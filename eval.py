"""
Evaluate UltraUNet segmentation checkpoints on labeled datasets (images/ + contours/).

Usage:
    python eval.py --checkpoint ./checkpoints/model.pth --dataset ./data/test

The dataset must contain:
    images/      *.png
    contours/    *.json   (paired by filename stem)

Metrics reported: soft Dice, binary Dice (full mask), MSD (skeleton distance).
"""

from __future__ import annotations

import json
import math
import os
import pickle
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.ultra_unet import UltraUNet
from utils.dataset import load_contour_data
from utils.metrics import DiceLoss
from utils.postprocessing import largest_connected_component
from utils.processing import (
    annotations_to_heatmap,
    histogram_match_tensor_array,
    msd,
    perform_skeletonization,
    resize_contour,
    skeleton_to_coordinates,
)

# Percentile normalisation (same as a0_2_normalizeImages)
UXTD_P_LOW = 1.0
UXTD_P_HIGH = 99.0
UXTD_NORM_SAMPLE_IMAGES = 500
UXTD_NORM_SAMPLE_SEED = 42
UXTD_NORM_MAX_PIXELS_PER_IMAGE = 50_000


# ============================================================================
# Intensity normalisation
# ============================================================================

def _normalize_frame(x_uint8: np.ndarray, *, lo01: float, hi01: float) -> np.ndarray:
    x01 = x_uint8.astype(np.float32) / 255.0
    lo, hi = float(lo01), float(hi01)
    if hi <= lo + 1e-12:
        return np.zeros_like(x01, dtype=np.float32)
    return (np.clip(x01, lo, hi) - lo) / (hi - lo)


def _to_uint8(x01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x01 * 255.0), 0, 255).astype(np.uint8)


def _compute_percentiles(png_paths: Sequence[Path], *, p_low: float, p_high: float) -> tuple:
    rng = np.random.default_rng(UXTD_NORM_SAMPLE_SEED)
    vals = []
    for p in png_paths:
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        x = (arr.astype(np.float32) / 255.0).ravel()
        if x.size > UXTD_NORM_MAX_PIXELS_PER_IMAGE:
            idx = rng.choice(x.size, size=UXTD_NORM_MAX_PIXELS_PER_IMAGE, replace=False)
            x = x[idx]
        vals.append(x)
    all_vals = np.concatenate(vals)
    return float(np.percentile(all_vals, p_low)), float(np.percentile(all_vals, p_high))


def _load_or_compute_norm(dataset_root: Path, results_dir: Path) -> tuple:
    cache_path = results_dir / "_norm_cache" / f"percentile_norm_{dataset_root.name}.json"
    if cache_path.is_file():
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
        return float(obj["lo01"]), float(obj["hi01"])

    images_dir = dataset_root / "images"
    pngs = sorted(p for p in images_dir.glob("*.png") if p.is_file())
    if not pngs:
        raise RuntimeError(f"No PNG files under {images_dir}")

    n_use = min(UXTD_NORM_SAMPLE_IMAGES, len(pngs))
    rng = np.random.default_rng(UXTD_NORM_SAMPLE_SEED)
    sampled = [pngs[i] for i in sorted(rng.choice(len(pngs), size=n_use, replace=False))]
    lo01, hi01 = _compute_percentiles(sampled, p_low=UXTD_P_LOW, p_high=UXTD_P_HIGH)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"lo01": lo01, "hi01": hi01, "dataset": str(dataset_root)}), encoding="utf-8")
    return lo01, hi01


# ============================================================================
# Data loading (contour-training compatible)
# ============================================================================

def load_eval_dataset(dataset_root: Path, height: int, width: int, *,
                       normalize: bool = False, norm_lo01: float = 0.0, norm_hi01: float = 1.0):
    """Wrapper around utils.dataset.load_contour_data with optional intensity normalisation."""
    if normalize:
        return _load_with_norm(dataset_root, height, width, norm_lo01, norm_hi01)
    return load_contour_data(str(dataset_root / "images"), str(dataset_root / "contours"),
                              type="circle", height=height, width=width)


def _load_with_norm(dataset_root, height, width, lo01, hi01):
    image_files = sorted(f for f in os.listdir(dataset_root / "images") if f.endswith(".png"))
    contour_files = sorted(f for f in os.listdir(dataset_root / "contours") if f.endswith(".json"))

    images, heatmaps, resized_contours, stems = [], [], [], []
    for img_f, ctr_f in zip(image_files, contour_files):
        if img_f.replace(".png", "") != ctr_f.replace(".json", ""):
            continue
        stem = img_f.replace(".png", "")
        with Image.open(dataset_root / "images" / img_f) as img:
            orig_size = img.size
            arr_u8 = np.asarray(img.convert("L"), dtype=np.uint8)
            x01 = _normalize_frame(arr_u8, lo01=lo01, hi01=hi01)
            img_resized = Image.fromarray(_to_uint8(x01), mode="L").resize((width, height), Image.BILINEAR)
            images.append(np.asarray(img_resized))

        with open(dataset_root / "contours" / ctr_f, "r", encoding="utf-8") as f:
            annot = json.load(f)
        rc = resize_contour(annot, orig_size, (width, height)) if annot else []
        resized_contours.append(rc)
        hm = annotations_to_heatmap(rc, height=height, width=width, resolution=None, type="circle")
        heatmaps.append(hm)
        stems.append(stem)

    images_t = torch.tensor(np.array(images), dtype=torch.float32) / 255.0
    heatmaps_t = torch.tensor(heatmaps, dtype=torch.float32)
    return images_t, heatmaps_t, resized_contours, stems


# ============================================================================
# Checkpoint loading
# ============================================================================

def load_model(checkpoint_path: Path, device: torch.device) -> UltraUNet:
    model = UltraUNet(img_ch=1, output_ch=1, n_channels=24).to(device)
    obj = torch.load(str(checkpoint_path), map_location=device)
    state_dict = None
    if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        state_dict = obj["model_state_dict"]
    elif isinstance(obj, dict):
        if any(torch.is_tensor(v) for v in obj.values()):
            state_dict = obj
    if state_dict is None:
        raise ValueError(f"Could not locate state_dict in {checkpoint_path}")
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
    model.eval()
    return model


# ============================================================================
# Metrics
# ============================================================================

def _binary_dice(pred_bin_hw: np.ndarray, gt_bin_hw: np.ndarray, *, smooth=1.0) -> float:
    p = pred_bin_hw.astype(np.float64).ravel()
    g = gt_bin_hw.astype(np.float64).ravel()
    inter = float((p * g).sum())
    return (2.0 * inter + smooth) / (float(p.sum()) + float(g.sum()) + smooth)


def _stats(xs: Sequence[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "median": float("nan")}
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)), "min": float(np.min(a)),
            "max": float(np.max(a)), "median": float(np.median(a))}


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_model(model, dataset, *, device, dice_loss, threshold=0.5, gt_binary_thresh=0.5,
                    lcc=True, show_progress=True) -> Dict[str, Any]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    dice_bin_valid, dice_soft_valid, msd_valid = [], [], []
    dice_bin_all, dice_soft_all, stems_all = [], [], []

    iterator = enumerate(loader)
    if show_progress:
        iterator = tqdm(iterator, total=len(loader), desc="eval", leave=True)

    for _, (image, heatmap, stem) in iterator:
        image = image.unsqueeze(1).float().to(device)
        heatmap = heatmap.unsqueeze(1).float().to(device)
        stem = stem[0] if isinstance(stem, (list, tuple)) else stem
        stems_all.append(stem)

        logits = model(image)
        probs = torch.sigmoid(logits)
        p2d = probs.squeeze(0).squeeze(0)
        h2d = heatmap.squeeze(0).squeeze(0)
        h_np = h2d.detach().cpu().numpy()
        outputs_np = p2d.detach().cpu().numpy()

        pred_bin_full = (outputs_np > threshold).astype(np.uint8)
        pred_bin_eval = largest_connected_component(outputs_np, threshold=threshold) if lcc else pred_bin_full
        pred_bin_eval = (np.asarray(pred_bin_eval) > 0.5).astype(np.uint8)

        gt_empty = float(h_np.sum()) <= 0.0
        if gt_empty:
            dice_soft_all.append(float("nan"))
            dice_bin_all.append(float("nan"))
        else:
            loss = dice_loss(p2d.unsqueeze(0), heatmap)
            dice_soft = 1.0 - float(loss.item())
            dice_soft_all.append(dice_soft)

            gt_bin = (h_np > gt_binary_thresh).astype(np.uint8)
            dice_bin = _binary_dice(pred_bin_eval, gt_bin)
            dice_bin_all.append(dice_bin)

            coord_in = perform_skeletonization(h_np, threshold=gt_binary_thresh)
            coord_in = skeleton_to_coordinates(coord_in)
            coord_out = perform_skeletonization(pred_bin_eval, threshold=threshold)
            coord_out = skeleton_to_coordinates(coord_out)

            if getattr(coord_in, "size", 0) == 0 or getattr(coord_out, "size", 0) == 0:
                msd_score = float("nan")
            else:
                msd_score = float(msd(coord_in, coord_out))

            if not math.isnan(msd_score) and not math.isinf(msd_score):
                dice_bin_valid.append(dice_bin)
                dice_soft_valid.append(dice_soft)
                msd_valid.append(msd_score)

    return {
        "dice_bin_stats": _stats(dice_bin_valid),
        "dice_soft_stats": _stats(dice_soft_valid),
        "msd_stats": _stats(msd_valid),
        "dice_bin_all": dice_bin_all,
        "dice_soft_all": dice_soft_all,
        "stems_all": stems_all,
        "n_total": len(stems_all),
        "n_used_dice": len(dice_bin_valid),
        "n_used_msd": len(msd_valid),
    }


def evaluate_checkpoint(checkpoint_path, *, dataset_path, results_dir="results/eval",
                         vis_every=0, normalize=False, lcc=True) -> Tuple[float, float]:
    """Evaluate a checkpoint and return (msd_mean, dice_mean)."""
    ckpt_path = Path(checkpoint_path)
    dataset_root = Path(dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalize:
        results_p = Path(results_dir)
        results_p.mkdir(parents=True, exist_ok=True)
        lo01, hi01 = _load_or_compute_norm(dataset_root, results_p)
    else:
        lo01, hi01 = 0.0, 1.0

    images_t, heatmaps_t, _, stems = load_eval_dataset(
        dataset_root, height=224, width=224,
        normalize=normalize, norm_lo01=lo01, norm_hi01=hi01,
    )

    class EvalDS(Dataset):
        def __init__(self):
            self.images = images_t
            self.heatmaps = heatmaps_t
            self.stems = stems
        def __len__(self):
            return len(self.stems)
        def __getitem__(self, idx):
            return self.images[idx], self.heatmaps[idx], self.stems[idx]

    ds = EvalDS()
    model = load_model(ckpt_path, device)
    dice_loss = DiceLoss()

    result = evaluate_model(model, ds, device=device, dice_loss=dice_loss,
                             threshold=0.5, gt_binary_thresh=0.5, lcc=lcc)

    print(f"\n=== {ckpt_path.name} ===")
    print(f"Dataset: {dataset_root}  (n={result['n_total']})")
    print(f"  Dice (binary): mean={result['dice_bin_stats']['mean']:.4f}  "
          f"std={result['dice_bin_stats']['std']:.4f}")
    print(f"  Dice (soft):   mean={result['dice_soft_stats']['mean']:.4f}  "
          f"std={result['dice_soft_stats']['std']:.4f}")
    print(f"  MSD:           mean={result['msd_stats']['mean']:.4f}  "
          f"std={result['msd_stats']['std']:.4f}")

    return float(result["msd_stats"]["mean"]), float(result["dice_bin_stats"]["mean"])


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate UltraUNet checkpoint on a labeled dataset.")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    p.add_argument("--dataset", required=True, help="Dataset root (images/ + contours/).")
    p.add_argument("--results_dir", default="results/eval")
    p.add_argument("--normalize", type=lambda x: x.lower() in ("true", "1", "yes"), default=False,
                   help="Apply percentile intensity normalisation (default: False).")
    p.add_argument("--no_lcc", action="store_true", help="Disable largest connected component post-processing.")
    args = p.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        results_dir=args.results_dir,
        normalize=args.normalize,
        lcc=not args.no_lcc,
    )


if __name__ == "__main__":
    main()
