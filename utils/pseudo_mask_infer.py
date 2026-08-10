"""
Shared helpers for UltraUNet grayscale loading, checkpoint restore, and in-memory pseudo masks.
Used by a2_1_train.py and a1_1_pretrainGAN.py (avoid circular imports between those scripts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm

from networks.ultra_unet import UltraUNet


def load_gray01_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    im = Image.open(image_path)
    if im.mode != "L":
        im = im.convert("L")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0)
    return x.to(device)


def load_ultraunet_from_checkpoint(checkpoint_path: Path, device: torch.device) -> UltraUNet:
    model = UltraUNet(img_ch=1, output_ch=1, n_channels=24).to(device)
    obj = torch.load(str(checkpoint_path), map_location=device)
    state_dict = None
    if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        state_dict = obj["model_state_dict"]
    elif isinstance(obj, dict):
        looks_like_state = any(torch.is_tensor(v) for v in obj.values())
        if looks_like_state:
            state_dict = obj
    if state_dict is None:
        raise ValueError(f"Could not locate state_dict in {checkpoint_path}")
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    return model


def load_masks_u8_from_dir(masks_dir: Path, stems: Sequence[str]) -> Dict[str, np.ndarray]:
    """Load ``stem.mask.png`` as uint8 (H,W) for each stem; raises if any file is missing."""
    out: Dict[str, np.ndarray] = {}
    for stem in stems:
        p = masks_dir / f"{stem}.mask.png"
        if not p.is_file():
            raise FileNotFoundError(f"Missing mask for stem {stem}: {p}")
        out[stem] = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
    return out


def iter_image_stems(images_dir: Path) -> List[str]:
    imgs = sorted([p for p in images_dir.glob("*.png") if p.is_file()])
    return [p.stem for p in imgs]


@torch.no_grad()
def infer_pseudo_masks_u8_dict(
    model: nn.Module,
    images_dir: Path,
    stems: Sequence[str],
    threshold: float,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run segmentation on each ``stem.png``; return uint8 (H,W) masks (0 / 255), not written to disk.
    """
    model.eval()
    out: Dict[str, np.ndarray] = {}
    for stem in tqdm(stems, desc="pseudo-masks (RAM)", unit="img"):
        img_path = images_dir / f"{stem}.png"
        x = load_gray01_tensor(img_path, device=device).unsqueeze(0)
        logits = model(x)
        probs = torch.sigmoid(logits)
        mask01 = (probs > float(threshold)).float().squeeze(0).squeeze(0).detach().cpu().numpy()
        u8 = (np.clip(mask01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        out[stem] = u8
    return out
