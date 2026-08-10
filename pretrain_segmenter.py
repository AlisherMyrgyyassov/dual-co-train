r"""
Pretrain a segmentation model (UltraUNet) on a contour-labeled ultrasound dataset.

Usage:
    python pretrain_segmenter.py --data_root ./data/dataset --out_dir ./checkpoints

Expected layout:
    <data_root>/
        images/    *.png
        contours/  *.json   (paired by filename stem)

Checkpoint is saved as:
    <out_dir>/<data_root_name_lower>.pth

Training mode is chosen automatically:
    n_pairs <= 20  → few-shot (resampling, 20 epochs)
    n_pairs >  20  → full-source (one pass per epoch, 30 epochs)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from networks.ultra_unet import UltraUNet, init_weights
from utils.processing import annotations_to_heatmap, resize_contour
from utils.metrics import DiceLoss, FocalLoss


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class PretrainConfig:
    # Data
    data_root: str = "data"
    out_dir: str = "checkpoints/pretrained"
    image_size: int = 224
    heatmap_type: str = "gauss"       # "gauss" or "circle"

    # Few-shot threshold: n_pairs <= this → few-shot mode
    few_shot_threshold: int = 20
    skip_if_exists: bool = True

    # Few-shot hyperparams
    few_shot_max_epochs: int = 20
    few_shot_steps_per_epoch: int = 80
    few_shot_batch_size: int = 2
    few_shot_num_workers: int = 0
    few_shot_patience: int = 60

    # Full-source hyperparams
    full_source_max_epochs: int = 30
    full_source_batch_size: int = 8
    full_source_num_workers: int = 4
    full_source_patience: int = 15

    # Optimisation
    lr: float = 1e-4
    weight_decay: float = 2e-4
    grad_clip_norm: float = 1.5
    ema_beta: float = 0.98
    min_delta: float = 1e-4

    # Augmentations
    p_affine: float = 0.95
    rot_deg: float = 20.0
    scale_min: float = 0.75
    scale_max: float = 1.25
    translate_frac: float = 0.12

    p_elastic: float = 0.70
    elastic_alpha: float = 10.0
    elastic_sigma: float = 8.0

    p_intensity: float = 0.95
    brightness: float = 0.15
    contrast: float = 0.20
    gamma_min: float = 0.85
    gamma_max: float = 1.20

    p_speckle: float = 0.80
    speckle_sigma: float = 0.15

    p_blur: float = 0.25
    blur_kernel: int = 3
    blur_sigma_min: float = 0.2
    blur_sigma_max: float = 1.0

    # Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_weight: float = 0.2
    focal_weight: float = 0.8

    base_seed: int = 42


# ============================================================================
# Helpers
# ============================================================================

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dataset_seed(base_seed: int, dataset_name: str) -> int:
    h = 2166136261
    for b in dataset_name.lower().encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int((base_seed + h) % (2**31 - 1))


# ============================================================================
# Pair discovery: images/*.png  +  contours/*.json
# ============================================================================

def discover_pairs(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"data_root not found: {dataset_dir}")

    images_dir = dataset_dir / "images"
    contours_dir = dataset_dir / "contours"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ under {dataset_dir}")
    if not contours_dir.is_dir():
        raise FileNotFoundError(f"Expected contours/ under {dataset_dir}")

    image_paths = sorted(p for p in images_dir.glob("*.png") if p.is_file())
    contour_paths = sorted(p for p in contours_dir.glob("*.json") if p.is_file())

    if not image_paths:
        raise FileNotFoundError(f"No .png images found in {images_dir}")
    if not contour_paths:
        raise FileNotFoundError(f"No .json contours found in {contours_dir}")

    img_map = {}
    for p in image_paths:
        stem = p.stem
        if stem in img_map:
            raise RuntimeError(f"Duplicate image stem '{stem}' under {dataset_dir}")
        img_map[stem] = p

    ctr_map = {}
    for p in contour_paths:
        stem = p.stem
        if stem in ctr_map:
            raise RuntimeError(f"Duplicate contour stem '{stem}' under {dataset_dir}")
        ctr_map[stem] = p

    common = sorted(set(img_map) & set(ctr_map))
    missing_imgs = sorted(set(ctr_map) - set(img_map))
    missing_ctrs = sorted(set(img_map) - set(ctr_map))
    if missing_imgs or missing_ctrs:
        lines = [f"Stem mismatch under {dataset_dir}:"]
        if missing_imgs:
            lines.append(f"  Missing images for stems: {missing_imgs}")
        if missing_ctrs:
            lines.append(f"  Missing contours for stems: {missing_ctrs}")
        raise RuntimeError("\n".join(lines))

    pairs = [(img_map[s], ctr_map[s]) for s in common]
    if not pairs:
        raise RuntimeError(f"{dataset_dir} has no paired image+contour samples.")
    return pairs


# ============================================================================
# Loading & heatmap generation
# ============================================================================

def _load_png_gray01(path: Path, size: int) -> torch.Tensor:
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    im = im.resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _heatmap_from_contour(contour_json: list, orig_size: Tuple[int, int], out_size: int, heatmap_type: str) -> torch.Tensor:
    if not contour_json:
        return torch.zeros(1, out_size, out_size, dtype=torch.float32)
    resized = resize_contour(contour_json, orig_size, (out_size, out_size))
    hm = annotations_to_heatmap(resized, height=out_size, width=out_size, resolution=None, type=heatmap_type)
    hm = np.nan_to_num(np.asarray(hm, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    hm = np.clip(hm, 0.0, 1.0)
    return torch.from_numpy(hm).unsqueeze(0)


# ============================================================================
# Paired augmentations
# ============================================================================

def _rand_uniform(a: float, b: float) -> float:
    return a + (b - a) * random.random()


def _apply_affine_pair(x: torch.Tensor, y: torch.Tensor, *, rot_deg: float, scale_min: float,
                        scale_max: float, translate_frac: float) -> Tuple[torch.Tensor, torch.Tensor]:
    _, h, w = x.shape
    angle = _rand_uniform(-rot_deg, rot_deg)
    scale = _rand_uniform(scale_min, scale_max)
    max_dx = int(round(translate_frac * w))
    max_dy = int(round(translate_frac * h))
    trans = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))
    x2 = TF.affine(x, angle=angle, translate=trans, scale=scale, shear=[0.0, 0.0],
                    interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    y2 = TF.affine(y, angle=angle, translate=trans, scale=scale, shear=[0.0, 0.0],
                    interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    return x2, y2


def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _smooth_field(field: torch.Tensor, sigma: float) -> torch.Tensor:
    _, _, h, w = field.shape
    k = max(3, int(round(sigma * 4)) | 1)
    kernel = _gaussian_kernel2d(k, sigma, device=field.device, dtype=field.dtype).view(1, 1, k, k)
    return F.conv2d(field, kernel, padding=k // 2)


def _apply_elastic_pair(x: torch.Tensor, y: torch.Tensor, *, alpha: float, sigma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    _, h, w = x.shape
    device, dtype = x.device, x.dtype
    dx = _smooth_field(torch.randn(1, 1, h, w, device=device, dtype=dtype), sigma)
    dy = _smooth_field(torch.randn(1, 1, h, w, device=device, dtype=dtype), sigma)
    dx = dx * (alpha / max(1.0, float(w)))
    dy = dy * (alpha / max(1.0, float(h)))
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    disp = torch.stack([dx.squeeze(0).squeeze(0), dy.squeeze(0).squeeze(0)], dim=-1).unsqueeze(0)
    grid = (grid + disp).clamp(-1.2, 1.2)
    x2 = F.grid_sample(x.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(0)
    y2 = F.grid_sample(y.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(0)
    return x2, y2


def _apply_intensity(x: torch.Tensor, *, brightness: float, contrast: float, gamma_min: float, gamma_max: float) -> torch.Tensor:
    b = _rand_uniform(-brightness, brightness)
    c = _rand_uniform(1.0 - contrast, 1.0 + contrast)
    g = _rand_uniform(gamma_min, gamma_max)
    x2 = torch.clamp(x + b, 0.0, 1.0)
    x2 = torch.clamp((x2 - 0.5) * c + 0.5, 0.0, 1.0)
    return torch.clamp(torch.pow(torch.clamp(x2, 1e-6, 1.0), g), 0.0, 1.0)


def _apply_speckle(x: torch.Tensor, *, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + x * noise, 0.0, 1.0)


def _apply_blur(x: torch.Tensor, *, kernel: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    if kernel <= 1:
        return x
    sigma = _rand_uniform(sigma_min, sigma_max)
    k = kernel if kernel % 2 == 1 else kernel + 1
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


class PairedAugment:
    def __init__(self, cfg: PretrainConfig):
        self.cfg = cfg

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.cfg.p_affine:
            x, y = _apply_affine_pair(x, y, rot_deg=self.cfg.rot_deg, scale_min=self.cfg.scale_min,
                                       scale_max=self.cfg.scale_max, translate_frac=self.cfg.translate_frac)
        if random.random() < self.cfg.p_elastic:
            x, y = _apply_elastic_pair(x, y, alpha=self.cfg.elastic_alpha, sigma=self.cfg.elastic_sigma)
        if random.random() < self.cfg.p_intensity:
            x = _apply_intensity(x, brightness=self.cfg.brightness, contrast=self.cfg.contrast,
                                  gamma_min=self.cfg.gamma_min, gamma_max=self.cfg.gamma_max)
        if random.random() < self.cfg.p_speckle:
            x = _apply_speckle(x, sigma=self.cfg.speckle_sigma)
        if random.random() < self.cfg.p_blur:
            x = _apply_blur(x, kernel=self.cfg.blur_kernel, sigma_min=self.cfg.blur_sigma_min,
                            sigma_max=self.cfg.blur_sigma_max)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return x, y


# ============================================================================
# Dataset
# ============================================================================

class ContourHeatmapPairs(Dataset):
    def __init__(self, pairs: Sequence[Tuple[Path, Path]], cfg: PretrainConfig, augment: Optional[PairedAugment]):
        self.pairs = list(pairs)
        self.cfg = cfg
        self.augment = augment
        self.stems = [p[0].stem for p in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, ctr_path = self.pairs[int(idx)]
        im = Image.open(img_path)
        if im.mode != "L":
            im = im.convert("L")
        orig_size = im.size

        x = _load_png_gray01(img_path, size=self.cfg.image_size)
        contour_json = json.loads(ctr_path.read_text(encoding="utf-8"))
        y = _heatmap_from_contour(contour_json, orig_size=orig_size, out_size=self.cfg.image_size,
                                   heatmap_type=self.cfg.heatmap_type)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if self.augment is not None:
            x, y = self.augment(x, y)
        return x, y


def _collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


# ============================================================================
# Loss
# ============================================================================

_dice_loss_fn = DiceLoss(smooth=1.0)
_focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")


def _combined_loss(logits: torch.Tensor, target: torch.Tensor, *, dice_weight: float,
                    focal_weight: float, focal_alpha: float, focal_gamma: float) -> torch.Tensor:
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    probs = torch.sigmoid(logits)
    dice_term = _dice_loss_fn(probs, target)
    p = probs.clamp(1e-8, 1.0 - 1e-8)
    t = target
    pos = -focal_alpha * (1.0 - p) ** focal_gamma * t * torch.log(p)
    neg = -(1.0 - focal_alpha) * p ** focal_gamma * (1.0 - t) * torch.log(1.0 - p)
    focal_term = (pos + neg).mean()
    return focal_weight * focal_term + dice_weight * dice_term


# ============================================================================
# Training
# ============================================================================

def _train_step(model, opt, x, y, cfg, device):
    x = x.to(device, non_blocking=True).float()
    y = y.to(device, non_blocking=True).float()
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    loss = _combined_loss(logits, y, dice_weight=cfg.dice_weight, focal_weight=cfg.focal_weight,
                           focal_alpha=cfg.focal_alpha, focal_gamma=cfg.focal_gamma)
    if not torch.isfinite(loss):
        return None, {}, True
    loss.backward()
    if cfg.grad_clip_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
    opt.step()
    probs = torch.sigmoid(logits.detach())
    dice_soft = 1.0 - float(_dice_loss_fn(probs, y.detach()).item())
    return float(loss.item()), {"dice_soft": dice_soft}, False


def _train_one_dataset(dataset_name: str, pairs: Sequence[Tuple[Path, Path]],
                        cfg: PretrainConfig, device: torch.device) -> Path:
    n_pairs = len(pairs)
    is_few_shot = n_pairs <= cfg.few_shot_threshold

    out_path = Path(cfg.out_dir) / f"{dataset_name.lower()}.pth"
    if cfg.skip_if_exists and out_path.is_file():
        mode = "few_shot" if is_few_shot else "full_source"
        print(f"\n=== Pretrain ({mode}): {dataset_name} ===")
        print(f"[skip] checkpoint already exists: {out_path}")
        return out_path

    seed = _dataset_seed(cfg.base_seed, dataset_name)
    _seed_everything(seed)

    model = UltraUNet(img_ch=1, output_ch=1, n_channels=24).to(device)
    init_weights(model, init_type="kaiming", gain=0.02)

    aug = PairedAugment(cfg)
    ds = ContourHeatmapPairs(pairs, cfg=cfg, augment=aug)

    if is_few_shot:
        max_epochs = cfg.few_shot_max_epochs
        steps_per_epoch = cfg.few_shot_steps_per_epoch
        batch_size = cfg.few_shot_batch_size
        num_workers = cfg.few_shot_num_workers
        patience = cfg.few_shot_patience
    else:
        max_epochs = cfg.full_source_max_epochs
        steps_per_epoch = 0
        batch_size = cfg.full_source_batch_size
        num_workers = cfg.full_source_num_workers
        patience = cfg.full_source_patience

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                     pin_memory=(device.type == "cuda"), collate_fn=_collate,
                     drop_last=is_few_shot and (n_pairs >= batch_size),
                     persistent_workers=(num_workers > 0))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_ema = float("inf")
    best_epoch = -1
    best_state = None
    ema_loss = None
    bad_epochs = 0

    mode = "few_shot" if is_few_shot else "full_source"
    print(f"\n=== Pretrain ({mode}): {dataset_name} ===")
    print(f"seed={seed}  n_pairs={n_pairs}  image_size={cfg.image_size}  heatmap={cfg.heatmap_type}  batch_size={batch_size}")

    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        skipped = 0
        n_steps = steps_per_epoch if is_few_shot else len(dl)

        if is_few_shot:
            def _iter_loader():
                while True:
                    for batch in dl:
                        yield batch
            batch_iter = _iter_loader()
            for _ in range(steps_per_epoch):
                x, y = next(batch_iter)
                loss_val, metrics, was_skipped = _train_step(model, opt, x, y, cfg, device)
                if was_skipped:
                    skipped += 1
                    continue
                running += loss_val
        else:
            for x, y in dl:
                loss_val, metrics, was_skipped = _train_step(model, opt, x, y, cfg, device)
                if was_skipped:
                    skipped += 1
                    continue
                running += loss_val

        denom = max(1, n_steps - skipped)
        epoch_loss = running / denom

        ema_loss = epoch_loss if ema_loss is None else cfg.ema_beta * ema_loss + (1.0 - cfg.ema_beta) * epoch_loss

        print(f"epoch={epoch:03d}  loss={epoch_loss:.4f}  ema={ema_loss:.4f}  dice={metrics.get('dice_soft', 0.0):.3f}")
        if skipped:
            print(f"  [warn] skipped_nonfinite_steps={skipped}/{n_steps}")

        if (best_ema - ema_loss) > cfg.min_delta:
            best_ema = ema_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"[early-stop] patience reached at epoch={epoch} (best_epoch={best_epoch}, best_ema={best_ema:.4f})")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": best_state,
        "epoch": best_epoch,
        "best_ema_loss": best_ema,
        "dataset_folder": dataset_name,
        "n_pairs": n_pairs,
        "training_mode": mode,
        "sample_stems": list(ds.stems),
        "train_config": asdict(cfg),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "format_note": "Compatible with utils.pseudo_mask_infer.load_ultraunet_from_checkpoint",
    }
    torch.save(payload, str(out_path))
    print(f"[saved] {out_path}")
    return out_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pretrain UltraUNet on a contour-labeled dataset.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Dataset directory with images/ (PNG) and contours/ (JSON).")
    parser.add_argument("--out_dir", type=str, default="checkpoints/pretrained",
                        help="Directory to save checkpoints.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--heatmap_type", type=str, default="gauss", choices=["gauss", "circle"])
    parser.add_argument("--few_shot_threshold", type=int, default=20)
    parser.add_argument("--no_skip", action="store_true", help="Overwrite existing checkpoints.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = PretrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        image_size=args.image_size,
        heatmap_type=args.heatmap_type,
        few_shot_threshold=args.few_shot_threshold,
        skip_if_exists=not args.no_skip,
        base_seed=args.seed,
        lr=args.lr,
    )

    device = _resolve_device()
    print(f"Device: {device}")
    print(f"data_root: {cfg.data_root}")
    print(f"out_dir:  {cfg.out_dir}")
    print(f"few_shot_threshold: {cfg.few_shot_threshold} (n_pairs <= this -> few-shot, else full-source)\n")

    dataset_dir = Path(cfg.data_root)
    pairs = discover_pairs(dataset_dir)
    _train_one_dataset(dataset_dir.name, pairs, cfg=cfg, device=device)

    print("\nDone.")


if __name__ == "__main__":
    main()
