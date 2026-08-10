r"""
Dual Co-Training Pipeline: Segmentation-guided GAN synthesis and refinement.

This script trains a segmenter on unlabeled ultrasound images via a co-training loop
that alternates between:

1. Pseudo-mask inference with the current teacher model.
2. Clean/noisy split using contour quality checks.
3. Synthetic image generation via a conditional GAN with contour augmentations.
4. Supervised training on clean + synthetic samples, consistency on noisy samples.
5. Periodic GAN fine-tuning with the latest pseudo masks.

Usage:
    python train.py \
        --images_dir ./data/unlabeled/images \
        --seg_checkpoint ./checkpoints/pretrained/segmenter.pth \
        --val_dataset ./data/test \
        --work_dir ./runs/experiment

    # With optional GAN pretrain from scratch:
    python train.py \
        --images_dir ./data/unlabeled/images \
        --seg_checkpoint ./checkpoints/pretrained/segmenter.pth \
        --pretrain_gan
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks.ultra_unet import UltraUNet
from utils.metrics import FocalLoss, DiceLoss
from methods.mask import get_largest_connected_component
from augmentations_contour import (
    add_synthetic_artifacts_to_conditioning_mask,
    augment_mask,
    build_condmask_from_augmented_contour,
    build_synthetic_target_mask,
    extract_non_lcc_artifacts,
    normalize_synthetic_target_mask_mode,
)
from gan_utils import finetune_gan, pretrain_gan
from contour_check import TongueMaskQCConfig, check_mask_quality
from utils.pseudo_mask_infer import infer_pseudo_masks_u8_dict, load_masks_u8_from_dir, load_ultraunet_from_checkpoint


# ============================================================================
# Synthetic image augmentations (GAN output post-processing)
# ============================================================================

def _rand_float(lo: float, hi: float) -> float:
    return float(lo) + (float(hi) - float(lo)) * float(random.random())


def _apply_speckle_noise_t(x01: torch.Tensor, *, noise_std: float = 0.1) -> torch.Tensor:
    noise = torch.randn_like(x01) * float(noise_std)
    return x01 + x01 * noise


def _apply_psf_blur_t(x01: torch.Tensor, *, axial_std_dev: float, lateral_std_dev: float) -> torch.Tensor:
    a = max(1e-6, float(axial_std_dev))
    l = max(1e-6, float(lateral_std_dev))
    k_ax = int(6 * a + 1) | 1
    k_lat = int(6 * l + 1) | 1
    yy, xx = torch.meshgrid(
        torch.linspace(-(k_ax // 2), k_ax // 2, k_ax, device=x01.device, dtype=x01.dtype),
        torch.linspace(-(k_lat // 2), k_lat // 2, k_lat, device=x01.device, dtype=x01.dtype),
        indexing="ij",
    )
    kernel = torch.exp(-(xx**2 / (2 * l**2) + yy**2 / (2 * a**2)))
    kernel = kernel / (kernel.sum() + 1e-12)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return F.conv2d(x01, kernel, padding=(k_ax // 2, k_lat // 2), groups=1)


def _apply_global_scale_bias_np(x01: np.ndarray, *, gain_min=0.9, gain_max=1.1,
                                 bias_min=-0.05, bias_max=0.05) -> np.ndarray:
    a = _rand_float(gain_min, gain_max)
    b = _rand_float(bias_min, bias_max)
    return np.clip(a * x01 + b, 0.0, 1.0)


# ============================================================================
# Enums & Config
# ============================================================================

class SampleType(str, Enum):
    clean = "clean"
    noisy = "noisy"
    synthetic = "synthetic"


@dataclass(frozen=True)
class TrainConfig:
    images_dir: Path = Path("data/images")
    work_dir: Path = Path("runs/experiment")
    results_dir: Path = Path("results/experiment")

    # Segmentation model init
    seg_checkpoint: Path = Path("checkpoints/pretrained/segmenter.pth")

    # GAN
    pretrain_gan: bool = False                     # if True, pretrain GAN from scratch before co-training
    gan_checkpoint_dir: Path = Path("checkpoints/gan")
    gan_best_ckpt: Optional[Path] = None            # pre-existing GAN checkpoint (if not pretraining)

    # Co-training
    seg_epochs: int = 20
    seg_lr: float = 1e-4
    seg_weight_decay: float = 0.0
    seg_batch_size: int = 8
    seg_num_workers: int = 2
    seg_threshold: float = 0.5

    # Loss weights
    w_clean: float = 0.7
    w_noisy: float = 0.3
    w_synth: float = 1.0

    # Clean/noisy split
    qc_mode: str = "qc"                             # "qc" or "heuristic"
    lcc_ratio_threshold: float = 0.80

    # Mean Teacher
    ema_decay: float = 0.99
    consistency_mse_weight: float = 1.0

    # Synthetic generation
    gan_noise_amplitude: float = 0.1
    synthetic_per_epoch: int = 1000
    synthetic_seed: int = 42
    synthetic_pool_fraction: float = 0.1
    synthetic_deform_prob: float = 0.6
    synthetic_move_prob: float = 0.3
    synthetic_rotate_prob: float = 0.4
    synthetic_dilate_prob: float = 0.0
    synthetic_dilate_radius_px: int = 1
    synthetic_target_mask_mode: str = "gauss"
    synthetic_target_gaussian_sigma: float = 3.0

    # GAN-output image augs
    synthetic_speckle_prob: float = 0.3
    synthetic_speckle_noise_std: float = 0.05
    synthetic_psf_blur_prob: float = 0.3
    synthetic_psf_axial_std_min: float = 0.1
    synthetic_psf_axial_std_max: float = 0.7
    synthetic_psf_lateral_std_min: float = 0.1
    synthetic_psf_lateral_std_max: float = 0.7
    synthetic_global_gain_bias_prob: float = 0.0

    # Conditioning-mask artifacts
    condmask_contour_soft_prob: float = 0.10
    condmask_contour_partial_soft_prob: float = 0.10
    condmask_contour_partial_soft_frac_min: float = 0.25
    condmask_contour_partial_soft_frac_max: float = 0.50
    condmask_contour_soft_value: float = 0.9
    condmask_keep_existing_artifacts_prob: float = 0.10
    condmask_add_synthetic_artifacts_prob: float = 0.10
    condmask_artifact_value_min: float = 0.4
    condmask_artifact_value_max: float = 0.8
    condmask_artifact_count_min: int = 1
    condmask_artifact_count_max: int = 3
    condmask_artifact_dist_min_px: int = 20
    condmask_artifact_dist_max_px: int = 70
    condmask_artifact_angle_max_abs_deg: float = 30.0
    condmask_artifact_skeleton_len_min_px: int = 3
    condmask_artifact_skeleton_len_max_px: int = 6
    condmask_artifact_radius_px: int = 3
    condmask_artifact_safety_margin_px: int = 1

    # GAN fine-tuning (during co-training)
    gan_finetune_every: int = 4
    gan_finetune_epochs: int = 2
    gan_lr_g: float = 1e-5
    gan_lr_d: float = 1e-4
    gan_batch_size: int = 8
    gan_num_workers: int = 2
    gan_lambda_l1: float = 10.0
    gan_lambda_perceptual: float = 8.0
    gan_noise_std: float = 0.02
    gan_d_full_epochs: int = 5

    # Validation
    do_validation: bool = True
    val_every: int = 5
    val_dataset_path: Optional[str] = None
    val_normalize: bool = False

    # Misc
    device: str = ""
    seed: int = 42
    log_every: int = 50
    debug_samples_per_epoch: int = 12
    save_visualizations: bool = True
    discard_images: bool = True


# ============================================================================
# Helpers
# ============================================================================

def _resolve_device(device_str: str) -> torch.device:
    if device_str.strip():
        return torch.device(device_str.strip())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _iter_image_stems(images_dir: Path) -> List[str]:
    imgs = sorted(p for p in images_dir.glob("*.png") if p.is_file())
    return [p.stem for p in imgs]


def _rand_bool(p: float) -> bool:
    return random.random() < float(p)


def _lcc_ratio(mask01_hw: np.ndarray) -> float:
    m = (mask01_hw > 0.5).astype(np.uint8)
    total = float(m.sum())
    if total <= 0.0:
        return 0.0
    return float(get_largest_connected_component(m).sum()) / total


def _make_teacher(student: nn.Module) -> nn.Module:
    teacher = UltraUNet(img_ch=1, output_ch=1, n_channels=24)
    teacher.load_state_dict(student.state_dict(), strict=True)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def _ema_update(teacher: nn.Module, student: nn.Module, ema_decay: float) -> None:
    d = float(ema_decay)
    t_state = teacher.state_dict()
    s_state = student.state_dict()
    for k in t_state.keys():
        t = t_state[k]
        s = s_state[k]
        if not torch.is_tensor(t) or not torch.is_tensor(s):
            continue
        t_state[k].copy_(t * d + s * (1.0 - d))


def _stochastic_image_aug(x: torch.Tensor) -> torch.Tensor:
    y = x
    if _rand_bool(0.5):
        y = torch.clamp(y + torch.randn_like(y) * 0.03, 0.0, 1.0)
    if _rand_bool(0.5):
        gain = 1.0 + (random.random() * 0.2 - 0.1)
        bias = random.random() * 0.06 - 0.03
        y = torch.clamp(y * gain + bias, 0.0, 1.0)
    return y


def _overlay_mask_on_image(img01: np.ndarray, mask01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    mask01 = (mask01 > 0.5).astype(np.float32)
    base = np.stack([img01, img01, img01], axis=-1)
    color = np.zeros_like(base)
    color[..., 1] = 1.0
    out = base * (1.0 - alpha * mask01[..., None]) + color * (alpha * mask01[..., None])
    return np.clip(out, 0.0, 1.0)


# ============================================================================
# Clean / Noisy split
# ============================================================================

def split_clean_noisy(
    stems: Sequence[str],
    lcc_ratio_threshold: float,
    *,
    masks_u8: Mapping[str, np.ndarray],
    qc_mode: str = "qc",
) -> Tuple[List[str], List[str], Dict[str, float]]:
    clean: List[str] = []
    noisy: List[str] = []
    ratios: Dict[str, float] = {}

    mode = str(qc_mode).strip().lower()
    if mode not in ("qc", "heuristic"):
        raise ValueError(f"split_clean_noisy: unsupported qc_mode={qc_mode!r}")

    qc_cfg = TongueMaskQCConfig()

    for stem in stems:
        m = np.asarray(masks_u8[stem], dtype=np.uint8)
        m01 = (m > 127).astype(np.uint8)
        if mode == "heuristic":
            r = _lcc_ratio(m01.astype(np.float32))
            ratios[stem] = float(r)
            if m01.sum() == 0:
                noisy.append(stem)
            elif r >= float(lcc_ratio_threshold):
                clean.append(stem)
            else:
                noisy.append(stem)
        else:
            ok, info = check_mask_quality(m01 > 0, cfg=qc_cfg)
            ratios[stem] = float(info.get("dominant_area_ratio", 0.0))
            if ok:
                clean.append(stem)
            else:
                noisy.append(stem)

    return clean, noisy, ratios


def write_split_lists(out_dir: Path, clean: Sequence[str], noisy: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean_list.txt").write_text("\n".join(clean) + "\n", encoding="utf-8")
    (out_dir / "noisy_list.txt").write_text("\n".join(noisy) + "\n", encoding="utf-8")


# ============================================================================
# Datasets
# ============================================================================

class UXTDImageMaskDataset(Dataset):
    def __init__(self, images_dir: Path, stems: Sequence[str], *,
                 masks_u8: Mapping[str, np.ndarray]) -> None:
        self.images_dir = images_dir
        self.masks_u8 = dict(masks_u8)
        self.stems = list(stems)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        im = Image.open(self.images_dir / f"{stem}.png").convert("L")
        img = (np.asarray(im, dtype=np.float32) / 255.0)[None, ...]
        m_u8 = self.masks_u8[stem]
        mask = ((m_u8 > 127).astype(np.float32))[None, ...]
        return {"image": torch.from_numpy(img), "mask": torch.from_numpy(mask), "stem": stem}


class SyntheticDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, stems: Sequence[str], *,
                 target_mask_mode: str = "gauss") -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.stems = list(stems)
        self.target_mask_mode = normalize_synthetic_target_mask_mode(target_mask_mode)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        im = Image.open(self.images_dir / f"{stem}.png").convert("L")
        img = (np.asarray(im, dtype=np.float32) / 255.0)[None, ...]
        m = np.asarray(Image.open(self.masks_dir / f"{stem}.mask.png").convert("L"))
        if self.target_mask_mode == "gauss":
            mask = (m.astype(np.float32) / 255.0)[None, ...]
        else:
            mask = ((m > 127).astype(np.float32))[None, ...]
        return {"image": torch.from_numpy(img), "mask": torch.from_numpy(mask), "stem": stem}


class MixedSampleDataset(Dataset):
    def __init__(self, clean_ds, noisy_ds, synth_ds=None):
        self.items: List[Tuple[SampleType, Dataset, int]] = []
        for i in range(len(clean_ds)):
            self.items.append((SampleType.clean, clean_ds, i))
        for i in range(len(noisy_ds)):
            self.items.append((SampleType.noisy, noisy_ds, i))
        if synth_ds is not None:
            for i in range(len(synth_ds)):
                self.items.append((SampleType.synthetic, synth_ds, i))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        typ, ds, j = self.items[idx]
        sample = dict(ds[j])
        sample["sample_type"] = typ.value
        return sample


def _collate_mixed(batch):
    images = torch.stack([b["image"] for b in batch], 0)
    masks = torch.stack([b["mask"] for b in batch], 0)
    stems = [str(b["stem"]) for b in batch]
    types = [str(b["sample_type"]) for b in batch]
    return {"image": images, "mask": masks, "stem": stems, "sample_type": types}


# ============================================================================
# Mask pool (largest area)
# ============================================================================

def select_largest_mask_stems(stems, pool_fraction, *, masks_u8) -> List[str]:
    cands = [s for s in stems if s in masks_u8]
    if not cands:
        return []
    scored = [(s, int((masks_u8[s] > 127).sum())) for s in cands]
    scored.sort(key=lambda x: (-x[1], x[0]))
    k = max(1, int(np.ceil(float(pool_fraction) * len(scored))))
    return [s for s, _ in scored[:k]]


# ============================================================================
# GAN generator loading
# ============================================================================

def _load_gan_generator(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(str(checkpoint_path))
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    gen_state = ckpt.get("generator_state_dict", ckpt)
    from networks.unet import UNet as GanUNet
    generator = GanUNet(n_channels=2, n_classes=1)
    cleaned = {k.replace("module.", ""): v for k, v in gen_state.items()}
    generator.load_state_dict(cleaned, strict=False)
    generator.to(device)
    generator.eval()
    return generator


# ============================================================================
# Synthetic dataset generation
# ============================================================================

@torch.no_grad()
def generate_synthetic_dataset(
    images_dir, masks_dir, out_images_dir, out_masks_dir, out_condmasks_dir,
    stems, generator, device, noise_amplitude, n_generate, seed,
    *, dilate_prob=0.0, dilate_radius_px=1,
    deform_prob=0.6, move_prob=0.3, rotate_prob=0.4,
    condmask_contour_soft_prob=0.10,
    condmask_contour_partial_soft_prob=0.10,
    condmask_contour_partial_soft_frac_min=0.25,
    condmask_contour_partial_soft_frac_max=0.50,
    condmask_contour_soft_value=0.9,
    condmask_keep_existing_artifacts_prob=0.10,
    condmask_add_synthetic_artifacts_prob=0.10,
    condmask_artifact_value_min=0.4,
    condmask_artifact_value_max=0.8,
    condmask_artifact_count_min=1,
    condmask_artifact_count_max=3,
    condmask_artifact_dist_min_px=20,
    condmask_artifact_dist_max_px=40,
    condmask_artifact_angle_max_abs_deg=30.0,
    condmask_artifact_skeleton_len_min_px=3,
    condmask_artifact_skeleton_len_max_px=6,
    condmask_artifact_radius_px=3,
    condmask_artifact_safety_margin_px=1,
    synthetic_speckle_prob=0.3,
    synthetic_speckle_noise_std=0.05,
    synthetic_psf_blur_prob=0.3,
    synthetic_psf_axial_std_min=0.1,
    synthetic_psf_axial_std_max=0.7,
    synthetic_psf_lateral_std_min=0.1,
    synthetic_psf_lateral_std_max=0.7,
    synthetic_global_gain_bias_prob=0.2,
    synthetic_global_gain_min=0.9,
    synthetic_global_gain_max=1.1,
    synthetic_global_bias_min=-0.05,
    synthetic_global_bias_max=0.05,
    pseudo_masks_u8=None,
    target_mask_mode="gauss",
    target_gaussian_sigma=3.0,
) -> List[str]:
    target_mask_mode = normalize_synthetic_target_mask_mode(target_mask_mode)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    out_condmasks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    stems = list(stems)
    if not stems:
        return []
    if (masks_dir is None) == (pseudo_masks_u8 is None):
        raise ValueError("generate_synthetic_dataset: pass exactly one of masks_dir or pseudo_masks_u8")

    picked = [stems[int(rng.integers(0, len(stems)))] for _ in range(int(n_generate))]
    out_stems = []

    _MASK_CACHE_MAX = 512
    mask_cache = OrderedDict()

    def _get_cached_masks(base_stem):
        if base_stem in mask_cache:
            mask_cache.move_to_end(base_stem)
            return mask_cache[base_stem]
        if pseudo_masks_u8 is not None:
            m_u8 = np.asarray(pseudo_masks_u8[base_stem], dtype=np.uint8)
        else:
            m_u8 = np.asarray(Image.open(masks_dir / f"{base_stem}.mask.png").convert("L"), dtype=np.uint8)
        m01 = (m_u8 > 127).astype(np.uint8)
        m_lcc01 = get_largest_connected_component(m01)
        mask_cache[base_stem] = (m01, m_lcc01, None)
        if len(mask_cache) > _MASK_CACHE_MAX:
            mask_cache.popitem(last=False)
        return m01, m_lcc01, None

    def _save_image_u8(img01_hw, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(np.round(img01_hw * 255.0), 0, 255).astype(np.uint8), mode="L").save(out_path)

    def _save_mask_u8(mask01_hw, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(np.round(mask01_hw * 255.0), 0, 255).astype(np.uint8), mode="L").save(out_path)

    for i, base_stem in enumerate(tqdm(picked, desc="synthetic", unit="img")):
        syn_stem = f"{base_stem}__syn_{i:06d}"
        m01, m_lcc01, non_lcc_u8 = _get_cached_masks(base_stem)

        do_deform = _rand_bool(deform_prob)
        do_move = _rand_bool(move_prob)
        do_rotate = _rand_bool(rotate_prob)
        do_dilate = _rand_bool(dilate_prob)
        aug01, points_xy = augment_mask(
            mask=m_lcc01, image_size=m01.shape[0], circle_radius=3,
            do_deform=do_deform, do_move=do_move, do_rotate=do_rotate,
            skeleton_threshold=0.5, return_contour=True,
        )
        aug01 = np.asarray(aug01, dtype=np.uint8)

        cond01 = build_condmask_from_augmented_contour(
            points_xy, image_size=int(m01.shape[0]), circle_radius=3, rng=rng,
            prob_partial_soft=condmask_contour_partial_soft_prob,
            partial_soft_frac_min=condmask_contour_partial_soft_frac_min,
            partial_soft_frac_max=condmask_contour_partial_soft_frac_max,
            prob_full_soft=condmask_contour_soft_prob,
            soft_value=condmask_contour_soft_value,
            contour01_u8_fallback=aug01,
        )
        if _rand_bool(condmask_keep_existing_artifacts_prob):
            existing_u8 = extract_non_lcc_artifacts(m01)
            cond01 = np.maximum(cond01, existing_u8.astype(np.float32))
        if _rand_bool(condmask_add_synthetic_artifacts_prob):
            cond01 = add_synthetic_artifacts_to_conditioning_mask(
                cond01, gt_mask01_u8=(aug01 > 0).astype(np.uint8), rng=rng,
                n_artifacts_min=condmask_artifact_count_min,
                n_artifacts_max=condmask_artifact_count_max,
                value_min=condmask_artifact_value_min,
                value_max=condmask_artifact_value_max,
                dist_min_px=condmask_artifact_dist_min_px,
                dist_max_px=condmask_artifact_dist_max_px,
                angle_max_abs_deg=condmask_artifact_angle_max_abs_deg,
                skeleton_len_min_px=condmask_artifact_skeleton_len_min_px,
                skeleton_len_max_px=condmask_artifact_skeleton_len_max_px,
                artifact_radius_px=condmask_artifact_radius_px,
                safety_margin_px=condmask_artifact_safety_margin_px,
            )

        mask_t = torch.from_numpy(cond01.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        noise = torch.randn_like(mask_t) * float(noise_amplitude)
        cond = torch.cat([mask_t, noise], dim=1)
        fake_t = torch.sigmoid(generator(cond))

        if _rand_bool(synthetic_speckle_prob):
            fake_t = _apply_speckle_noise_t(fake_t, noise_std=synthetic_speckle_noise_std)
        if _rand_bool(synthetic_psf_blur_prob):
            fake_t = _apply_psf_blur_t(
                fake_t,
                axial_std_dev=_rand_float(synthetic_psf_axial_std_min, synthetic_psf_axial_std_max),
                lateral_std_dev=_rand_float(synthetic_psf_lateral_std_min, synthetic_psf_lateral_std_max),
            )

        fake = torch.clamp(fake_t, 0.0, 1.0).squeeze(0).squeeze(0).detach().cpu().numpy()
        if _rand_bool(synthetic_global_gain_bias_prob):
            fake = _apply_global_scale_bias_np(fake)

        gt01_out = build_synthetic_target_mask(
            points_xy, aug01, image_size=int(m01.shape[0]),
            mode=target_mask_mode, gaussian_sigma=target_gaussian_sigma,
        )

        _save_image_u8(fake, out_images_dir / f"{syn_stem}.png")
        _save_mask_u8(gt01_out.astype(np.float32), out_masks_dir / f"{syn_stem}.mask.png")
        _save_mask_u8(cond01.astype(np.float32), out_condmasks_dir / f"{syn_stem}.condmask.png")
        out_stems.append(syn_stem)

    return out_stems


# ============================================================================
# Segmentation training (one epoch)
# ============================================================================

def _dice_focal_loss(logits, target01, focal, dice):
    target01 = target01.float()
    return 0.8 * focal(logits, target01) + 0.2 * dice(torch.sigmoid(logits), target01)


def train_one_epoch(student, teacher, loader, optimizer, device, cfg) -> Dict[str, float]:
    student.train()
    teacher.eval()
    focal = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean").to(device)
    dice = DiceLoss(smooth=1.0).to(device)
    mse = nn.MSELoss(reduction="mean").to(device)

    totals = {"loss": 0.0, "loss_clean": 0.0, "loss_noisy": 0.0, "loss_synth": 0.0,
              "n": 0, "n_clean": 0, "n_noisy": 0, "n_synth": 0}

    pbar = tqdm(loader, desc="seg-train", unit="batch")
    for step, batch in enumerate(pbar):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        types = list(batch["sample_type"])

        optimizer.zero_grad(set_to_none=True)
        logits = student(x)
        loss_total = torch.zeros((), device=device)

        idx_clean = [i for i, t in enumerate(types) if t == SampleType.clean.value]
        if idx_clean:
            lc = _dice_focal_loss(logits[idx_clean], y[idx_clean], focal=focal, dice=dice)
            loss_total = loss_total + lc * float(cfg.w_clean)
            totals["loss_clean"] += float(lc.detach().cpu().item())
            totals["n_clean"] += len(idx_clean)

        idx_syn = [i for i, t in enumerate(types) if t == SampleType.synthetic.value]
        if idx_syn:
            ls = _dice_focal_loss(logits[idx_syn], y[idx_syn], focal=focal, dice=dice)
            loss_total = loss_total + ls * float(cfg.w_synth)
            totals["loss_synth"] += float(ls.detach().cpu().item())
            totals["n_synth"] += len(idx_syn)

        idx_noisy = [i for i, t in enumerate(types) if t == SampleType.noisy.value]
        if idx_noisy:
            x_n = x[idx_noisy]
            x_t = _stochastic_image_aug(x_n.detach())
            x_s = _stochastic_image_aug(x_n)
            with torch.no_grad():
                p_t = torch.sigmoid(teacher(x_t))
            p_s = torch.sigmoid(student(x_s))
            ln = mse(p_s, p_t) * float(cfg.consistency_mse_weight)
            loss_total = loss_total + ln * float(cfg.w_noisy)
            totals["loss_noisy"] += float(ln.detach().cpu().item())
            totals["n_noisy"] += len(idx_noisy)

        loss_total.backward()
        optimizer.step()
        _ema_update(teacher, student, ema_decay=cfg.ema_decay)

        totals["loss"] += float(loss_total.detach().cpu().item())
        totals["n"] += x.shape[0]

        if (step + 1) % cfg.log_every == 0:
            pbar.set_postfix(loss=f"{totals['loss'] / max(1, step + 1):.4f}",
                             clean=totals["n_clean"], noisy=totals["n_noisy"], synth=totals["n_synth"])

    n_batches = max(1, len(loader))
    return {k: totals[k] / n_batches if k.startswith("loss_") else float(totals[k])
            for k in ("loss", "loss_clean", "loss_noisy", "loss_synth", "n_clean", "n_noisy", "n_synth")}


# ============================================================================
# Save helpers
# ============================================================================

def _save_image_u8(img01_hw, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.round(img01_hw * 255.0), 0, 255).astype(np.uint8), mode="L").save(out_path)


def _save_mask_u8(mask01_hw, out_path):
    _save_image_u8(mask01_hw, out_path)


def _save_debug_grid(*, out_path, panels, ncols=4, cmap="gray"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    if n <= 0:
        return
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
    axes = np.asarray(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        title, img = panels[i]
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)


def _discard_epoch_images(epoch_dir: Path) -> int:
    if not epoch_dir.exists():
        return 0
    exts = {".png", ".jpg", ".jpeg"}
    removed = 0
    for p in list(epoch_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed


# ============================================================================
# Main training pipeline
# ============================================================================

def train(config: Optional[TrainConfig] = None, *, device=None, **overrides) -> Dict[str, Any]:
    if config is None:
        config = TrainConfig()
    if overrides:
        valid = set(asdict(config).keys())
        unknown = sorted(k for k in overrides if k not in valid)
        if unknown:
            raise TypeError(f"Unknown config override(s): {unknown}")
        config = replace(config, **overrides)

    cfg = config

    def _as_path(x):
        return x if isinstance(x, Path) else Path(str(x))

    cfg = replace(cfg,
        images_dir=_as_path(cfg.images_dir),
        work_dir=_as_path(cfg.work_dir),
        results_dir=_as_path(cfg.results_dir),
        seg_checkpoint=_as_path(cfg.seg_checkpoint),
        gan_checkpoint_dir=_as_path(cfg.gan_checkpoint_dir),
        gan_best_ckpt=_as_path(cfg.gan_best_ckpt) if cfg.gan_best_ckpt is not None else None,
    )

    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(cfg.seed)
    if device is None:
        device = _resolve_device(cfg.device)

    if not cfg.images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {cfg.images_dir}")
    if not cfg.seg_checkpoint.exists():
        raise FileNotFoundError(f"seg_checkpoint not found: {cfg.seg_checkpoint}")

    all_stems = _iter_image_stems(cfg.images_dir)
    if not all_stems:
        raise FileNotFoundError(f"No .png images found in {cfg.images_dir}")

    print(f"Device: {device}")
    print(f"Images: {cfg.images_dir} (n={len(all_stems)})")
    print(f"Segmenter checkpoint: {cfg.seg_checkpoint}")
    print(f"Work dir: {cfg.work_dir}")

    # ----- Step 1: GAN pretrain (optional) -----
    gan_ckpt = cfg.gan_best_ckpt
    if cfg.pretrain_gan:
        print("\n--- GAN Pretrain ---")
        gan_out = pretrain_gan(
            images_folder=str(cfg.images_dir),
            ultraunet_mask_checkpoint=str(cfg.seg_checkpoint),
            checkpoint_dir=str(cfg.gan_checkpoint_dir),
            epochs=30,
            device=device,
        )
        gan_ckpt = gan_out["best_model_path"]
        print(f"GAN pretrain complete: {gan_ckpt}")
    if gan_ckpt is None or not gan_ckpt.exists():
        raise FileNotFoundError(f"GAN checkpoint not found: {gan_ckpt}. Use --pretrain_gan or provide --gan_checkpoint.")

    # ----- Step 2: Initialize student & teacher -----
    student = load_ultraunet_from_checkpoint(cfg.seg_checkpoint, device=device).to(device)
    teacher = _make_teacher(student).to(device)
    optimizer = optim.Adam(student.parameters(), lr=cfg.seg_lr, weight_decay=cfg.seg_weight_decay)

    history: Dict[str, List[float]] = {
        "loss": [], "loss_clean": [], "loss_noisy": [], "loss_synth": [],
        "n_clean": [], "n_noisy": [], "n_synth": [],
    }

    seg_latest_ckpt_path = cfg.work_dir / "ultraunet_student_latest.pth"
    last_seg_ckpt_path = None

    # ----- Step 3: Co-training loop -----
    for epoch in range(cfg.seg_epochs):
        print(f"\n=== Epoch {epoch}/{cfg.seg_epochs - 1} ===")

        epoch_dir = cfg.work_dir / f"epoch_{epoch:03d}"
        _discard_epoch_images(epoch_dir)
        results_epoch_dir = cfg.results_dir / f"epoch-{epoch:03d}"
        results_epoch_dir.mkdir(parents=True, exist_ok=True)

        # 3a) Pseudo masks (teacher inference)
        pseudo_masks_u8 = infer_pseudo_masks_u8_dict(teacher, cfg.images_dir, all_stems, cfg.seg_threshold, device)

        # 3b) Clean/noisy split
        clean_stems, noisy_stems, _ = split_clean_noisy(all_stems, cfg.lcc_ratio_threshold,
                                                          masks_u8=pseudo_masks_u8, qc_mode=cfg.qc_mode)
        write_split_lists(epoch_dir, clean=clean_stems, noisy=noisy_stems)
        print(f"Split ({cfg.qc_mode}): clean={len(clean_stems)} noisy={len(noisy_stems)}")

        # 3c) Synthetic generation
        syn_root = epoch_dir / "synthetic"
        syn_images_dir = syn_root / "images"
        syn_masks_dir = syn_root / "masks"
        syn_condmasks_dir = syn_root / "condmasks"

        use_synthetic = cfg.synthetic_per_epoch > 0
        if use_synthetic:
            generator = _load_gan_generator(gan_ckpt, device=device)
            cands = clean_stems if clean_stems else all_stems
            pool_stems = select_largest_mask_stems(cands, pool_fraction=cfg.synthetic_pool_fraction,
                                                    masks_u8=pseudo_masks_u8) or list(cands)
            n_gen = cfg.synthetic_per_epoch
            print(f"Synth: pool={len(pool_stems)} (from |candidates|={len(cands)}), n_generate={n_gen}")
            synth_stems = generate_synthetic_dataset(
                images_dir=cfg.images_dir, masks_dir=None,
                out_images_dir=syn_images_dir, out_masks_dir=syn_masks_dir,
                out_condmasks_dir=syn_condmasks_dir,
                stems=pool_stems, generator=generator, device=device,
                noise_amplitude=cfg.gan_noise_amplitude, n_generate=n_gen,
                seed=cfg.synthetic_seed + epoch,
                dilate_prob=cfg.synthetic_dilate_prob,
                dilate_radius_px=cfg.synthetic_dilate_radius_px,
                deform_prob=cfg.synthetic_deform_prob,
                move_prob=cfg.synthetic_move_prob,
                rotate_prob=cfg.synthetic_rotate_prob,
                condmask_contour_soft_prob=cfg.condmask_contour_soft_prob,
                condmask_contour_partial_soft_prob=cfg.condmask_contour_partial_soft_prob,
                condmask_contour_partial_soft_frac_min=cfg.condmask_contour_partial_soft_frac_min,
                condmask_contour_partial_soft_frac_max=cfg.condmask_contour_partial_soft_frac_max,
                condmask_contour_soft_value=cfg.condmask_contour_soft_value,
                condmask_keep_existing_artifacts_prob=cfg.condmask_keep_existing_artifacts_prob,
                condmask_add_synthetic_artifacts_prob=cfg.condmask_add_synthetic_artifacts_prob,
                condmask_artifact_value_min=cfg.condmask_artifact_value_min,
                condmask_artifact_value_max=cfg.condmask_artifact_value_max,
                condmask_artifact_count_min=cfg.condmask_artifact_count_min,
                condmask_artifact_count_max=cfg.condmask_artifact_count_max,
                condmask_artifact_dist_min_px=cfg.condmask_artifact_dist_min_px,
                condmask_artifact_dist_max_px=cfg.condmask_artifact_dist_max_px,
                condmask_artifact_angle_max_abs_deg=cfg.condmask_artifact_angle_max_abs_deg,
                condmask_artifact_skeleton_len_min_px=cfg.condmask_artifact_skeleton_len_min_px,
                condmask_artifact_skeleton_len_max_px=cfg.condmask_artifact_skeleton_len_max_px,
                condmask_artifact_radius_px=cfg.condmask_artifact_radius_px,
                condmask_artifact_safety_margin_px=cfg.condmask_artifact_safety_margin_px,
                synthetic_speckle_prob=cfg.synthetic_speckle_prob,
                synthetic_speckle_noise_std=cfg.synthetic_speckle_noise_std,
                synthetic_psf_blur_prob=cfg.synthetic_psf_blur_prob,
                synthetic_psf_axial_std_min=cfg.synthetic_psf_axial_std_min,
                synthetic_psf_axial_std_max=cfg.synthetic_psf_axial_std_max,
                synthetic_psf_lateral_std_min=cfg.synthetic_psf_lateral_std_min,
                synthetic_psf_lateral_std_max=cfg.synthetic_psf_lateral_std_max,
                synthetic_global_gain_bias_prob=cfg.synthetic_global_gain_bias_prob,
                pseudo_masks_u8=pseudo_masks_u8,
                target_mask_mode=cfg.synthetic_target_mask_mode,
                target_gaussian_sigma=cfg.synthetic_target_gaussian_sigma,
            )
            print(f"Synthetic generated: {len(synth_stems)}")
        else:
            synth_stems = []

        # 3d) Train segmentation for one epoch
        clean_ds = UXTDImageMaskDataset(cfg.images_dir, clean_stems, masks_u8=pseudo_masks_u8)
        noisy_ds = UXTDImageMaskDataset(cfg.images_dir, noisy_stems, masks_u8=pseudo_masks_u8)
        synth_ds = SyntheticDataset(syn_images_dir, syn_masks_dir, synth_stems,
                                     target_mask_mode=cfg.synthetic_target_mask_mode) if synth_stems else None
        mixed = MixedSampleDataset(clean_ds=clean_ds, noisy_ds=noisy_ds, synth_ds=synth_ds)
        dl = DataLoader(mixed, batch_size=cfg.seg_batch_size, shuffle=True, num_workers=cfg.seg_num_workers,
                         pin_memory=(device.type == "cuda"), collate_fn=_collate_mixed)
        stats = train_one_epoch(student=student, teacher=teacher, loader=dl, optimizer=optimizer,
                                 device=device, cfg=cfg)
        for k in history:
            if k in stats:
                history[k].append(stats[k])
        print(f"Seg: loss={stats['loss']:.4f} (clean={stats['loss_clean']:.4f} "
              f"noisy={stats['loss_noisy']:.4f} synth={stats['loss_synth']:.4f})")

        # Save checkpoint
        torch.save({"epoch": epoch, "model_state_dict": student.state_dict(), "optimizer": optimizer.state_dict()},
                    seg_latest_ckpt_path)
        last_seg_ckpt_path = seg_latest_ckpt_path

        # Validation
        if cfg.do_validation and cfg.val_every > 0 and (epoch + 1) % cfg.val_every == 0 and cfg.val_dataset_path:
            try:
                from eval import evaluate_checkpoint
                msd_mean, dice_mean = evaluate_checkpoint(
                    checkpoint_path=str(seg_latest_ckpt_path),
                    dataset_path=str(cfg.val_dataset_path),
                    normalize=cfg.val_normalize,
                )
                print(f"VAL (epoch {epoch+1}): dice={dice_mean:.6f} msd={msd_mean:.6f}")
            except Exception as e:
                print(f"[val] Failed at epoch {epoch+1}: {type(e).__name__}: {e}")

        # 3e) Periodic GAN fine-tuning
        if use_synthetic and cfg.gan_finetune_every > 0 and (epoch + 1) % cfg.gan_finetune_every == 0:
            print(f"GAN fine-tune at epoch {epoch}")
            gan_masks_u8 = infer_pseudo_masks_u8_dict(teacher, cfg.images_dir, all_stems, cfg.seg_threshold, device)
            final_epoch_dir = cfg.work_dir / f"epoch_{(cfg.seg_epochs - 1):03d}"
            out_dir = final_epoch_dir / "gan_checkpoints"
            best_path = finetune_gan(
                ckpt_in=str(gan_ckpt), images_folder=str(cfg.images_dir), masks_folder=None,
                checkpoint_dir=str(out_dir), device=device, num_epochs=cfg.gan_finetune_epochs,
                batch_size=cfg.gan_batch_size, num_workers=cfg.gan_num_workers,
                learning_rate_g=cfg.gan_lr_g, learning_rate_d=cfg.gan_lr_d,
                lambda_l1=cfg.gan_lambda_l1, lambda_perceptual=cfg.gan_lambda_perceptual,
                noise_std=cfg.gan_noise_std, noise_amplitude=cfg.gan_noise_amplitude,
                seed=cfg.seed + epoch, masks_by_stem=gan_masks_u8,
            )
            gan_ckpt = Path(best_path)
            print(f"Updated GAN checkpoint: {gan_ckpt}")

        if cfg.discard_images:
            removed = _discard_epoch_images(epoch_dir)
            print(f"[cleanup] Discarded {removed} image files under {epoch_dir}")

    # ----- Save results -----
    out_dir = cfg.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "loss_history.csv"
    lines = ["epoch,loss,loss_clean,loss_noisy,loss_synth,n_clean,n_noisy,n_synth"]
    for i in range(len(history["loss"])):
        lines.append(f"{i+1},{history['loss'][i]},{history['loss_clean'][i]},{history['loss_noisy'][i]},"
                     f"{history['loss_synth'][i]},{history['n_clean'][i]},{history['n_noisy'][i]},{history['n_synth'][i]}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if cfg.save_visualizations:
        try:
            epochs = np.arange(1, len(history["loss"]) + 1)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(epochs, history["loss"], label="total")
            ax.plot(epochs, history["loss_clean"], label="clean")
            ax.plot(epochs, history["loss_noisy"], label="noisy")
            ax.plot(epochs, history["loss_synth"], label="synthetic")
            ax.set_xlabel("epoch"); ax.set_ylabel("loss")
            ax.grid(True, alpha=0.3); ax.legend()
            fig.tight_layout()
            fig.savefig(str(out_dir / "loss_curves.png"), dpi=200)
            plt.close(fig)
        except Exception:
            pass

    print("\nDone.")
    return {
        "config": cfg, "device": device, "history": history,
        "work_dir": cfg.work_dir, "results_dir": cfg.results_dir,
        "last_seg_checkpoint_path": last_seg_ckpt_path,
        "gan_best_ckpt": gan_ckpt,
    }


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Dual Co-Training: segmentation-guided GAN synthesis and refinement.")

    # Required
    p.add_argument("--images_dir", required=True, help="Directory of unlabeled target-domain PNG images.")
    p.add_argument("--seg_checkpoint", required=True, help="Path to pretrained segmenter checkpoint (.pth).")
    p.add_argument("--work_dir", default="runs/experiment")
    p.add_argument("--results_dir", default="results/experiment")

    # GAN
    p.add_argument("--pretrain_gan", action="store_true",
                   help="Pretrain GAN from scratch on unlabeled images before co-training.")
    p.add_argument("--gan_checkpoint", default=None, help="Pre-existing GAN checkpoint (required if not --pretrain_gan).")
    p.add_argument("--gan_checkpoint_dir", default="checkpoints/gan")

    # Co-training
    p.add_argument("--seg_epochs", type=int, default=20)
    p.add_argument("--seg_lr", type=float, default=1e-4)
    p.add_argument("--seg_batch_size", type=int, default=8)
    p.add_argument("--synthetic_per_epoch", type=int, default=1000)
    p.add_argument("--qc_mode", default="qc", choices=["qc", "heuristic"])
    p.add_argument("--gan_finetune_every", type=int, default=4)
    p.add_argument("--gan_finetune_epochs", type=int, default=2)

    # Validation
    p.add_argument("--val_dataset", default=None, help="Labeled test dataset root (images/ + contours/).")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--val_normalize", type=lambda x: x.lower() in ("true", "1", "yes"), default=False,
                   help="Apply percentile intensity normalisation during validation (default: False).")

    # Misc
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_visualizations", action="store_true")

    args = p.parse_args()

    return TrainConfig(
        images_dir=Path(args.images_dir),
        seg_checkpoint=Path(args.seg_checkpoint),
        work_dir=Path(args.work_dir),
        results_dir=Path(args.results_dir),
        pretrain_gan=args.pretrain_gan,
        gan_best_ckpt=Path(args.gan_checkpoint) if args.gan_checkpoint else None,
        gan_checkpoint_dir=Path(args.gan_checkpoint_dir),
        seg_epochs=args.seg_epochs,
        seg_lr=args.seg_lr,
        seg_batch_size=args.seg_batch_size,
        synthetic_per_epoch=args.synthetic_per_epoch,
        qc_mode=args.qc_mode,
        gan_finetune_every=args.gan_finetune_every,
        gan_finetune_epochs=args.gan_finetune_epochs,
        val_dataset_path=args.val_dataset,
        val_every=args.val_every,
        val_normalize=args.val_normalize,
        device=args.device,
        seed=args.seed,
        save_visualizations=not args.no_visualizations,
    )


def main():
    cfg = _parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
