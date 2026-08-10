"""
Contour / mask quality checks for ultrasound tongue segmentation masks.

Goal: cheaply flag likely-bad masks ("noisy") vs acceptable ("clean") using:
- Connectedness check: dominant connected component ratio + component count
- Topology check: holes inside the ROI + broken/fragmented skeleton
- Shape plausibility: area, perimeter, eccentricity, solidity, elongation, thinness/spikiness

If run as a script:
    python contour_check.py "D:\\path\\to\\masks_folder"

It will create two subfolders inside that folder:
    clean/  noisy/
and copy each mask PNG into one of them, plus write `quality_report.csv`.

Assumptions:
- Input masks are PNG grayscale where foreground is >127 (same convention as this repo).
- Tongue masks are typically a single dominant blob with low fragmentation.
"""

from __future__ import annotations

import csv
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, binary_closing, disk
from skimage.measure import perimeter as sk_perimeter


@dataclass(frozen=True)
class TongueMaskQCConfig:
    # --- Connectedness ---
    # Conservative (lenient) defaults: allow a few small satellite blobs.
    max_components: int = 4
    dominant_area_ratio_min: float = 0.80  # dominant_cc_area / total_fg_area
    min_fg_area_px: int = 60  # below this treat as empty/noisy

    # --- Border-touch hard rule ---
    # If the dominant blob touches the outer band on any side, mark as noisy.
    # Example: 0.10 => outermost 10% of image on left/right/top/bottom.
    border_touch_frac: float = 0.10

    # --- Topology ---
    # Tongue masks may contain small internal voids; allow a few.
    max_holes: int = 6
    roi_pad_px: int = 6
    closing_radius_px: int = 3  # stabilize small gaps before checks
    # Skeleton fragmentation is a weak signal; be lenient.
    max_skeleton_components: int = 6

    # --- Shape plausibility ---
    # Use normalized stats to make thresholds resolution-robust.
    min_area_frac: float = 0.001  # area / (H*W)
    max_area_frac: float = 0.50
    # Lower solidity threshold to avoid over-flagging slightly concave shapes.
    min_solidity: float = 0.65
    max_eccentricity: float = 0.995  # very close to 1 => extremely elongated line-like
    max_elongation: float = 12.0  # major/minor axis ratio
    min_minor_axis_px: float = 3.0  # very thin masks are suspicious

    # Spikiness / fragmentation heuristics
    # Higher => more jagged; keep lenient by default.
    max_perimeter_area_ratio: float = 1.20  # perimeter / area (area in px)


def _load_mask_png_to_bool(mask_path: Path) -> np.ndarray:
    u8 = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return (u8 > 127)


def _count_holes_in_roi(mask: np.ndarray, *, roi_pad_px: int) -> int:
    """
    Count enclosed background components ("holes") inside a tight ROI around the mask.
    Cheap approximation: invert within ROI and count connected components not touching ROI border.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0
    h, w = mask.shape
    y0 = max(0, int(ys.min()) - int(roi_pad_px))
    y1 = min(h, int(ys.max()) + int(roi_pad_px) + 1)
    x0 = max(0, int(xs.min()) - int(roi_pad_px))
    x1 = min(w, int(xs.max()) + int(roi_pad_px) + 1)

    roi = mask[y0:y1, x0:x1]
    bg = ~roi
    # Use 8-connectivity consistently.
    lab = label(bg, connectivity=2)
    if lab.max() == 0:
        return 0

    holes = 0
    # Any background component that touches ROI boundary is "outside"; others are holes.
    for idx in range(1, int(lab.max()) + 1):
        cc = (lab == idx)
        if (
            cc[0, :].any()
            or cc[-1, :].any()
            or cc[:, 0].any()
            or cc[:, -1].any()
        ):
            continue
        holes += 1
    return int(holes)


def check_mask_quality(
    mask: np.ndarray,
    *,
    cfg: TongueMaskQCConfig = TongueMaskQCConfig(),
) -> Tuple[bool, Dict[str, Any]]:
    """
    Args:
        mask: HxW bool or 0/1 array (foreground=True/1)
        cfg: thresholds

    Returns:
        (is_clean, info_dict)
    """
    m = (mask > 0).astype(bool)
    h, w = m.shape
    info: Dict[str, Any] = {"h": int(h), "w": int(w)}

    fg_area = int(m.sum())
    info["fg_area_px"] = fg_area
    info["area_frac"] = float(fg_area) / float(max(1, h * w))

    reasons = []   # hard filters => noisy
    warnings = []  # secondary signals (reported, but do not flip clean->noisy)
    if fg_area < int(cfg.min_fg_area_px):
        reasons.append("empty_or_tiny")
        info["reasons"] = reasons
        info["warnings"] = warnings
        return False, info

    # Stabilize small gaps / jaggies (cheap)
    if int(cfg.closing_radius_px) > 0:
        m = binary_closing(m, disk(int(cfg.closing_radius_px)))

    # -----------------------------
    # Connectedness: components + dominant ratio
    # -----------------------------
    # Use 8-connectivity for tongue blobs.
    lab = label(m, connectivity=2)
    n_comp = int(lab.max())
    info["n_components"] = n_comp

    if n_comp == 0:
        reasons.append("empty_after_closing")
        info["reasons"] = reasons
        info["warnings"] = warnings
        return False, info

    props_all = regionprops(lab)
    areas = np.array([p.area for p in props_all], dtype=np.float64)
    dom_area = float(areas.max()) if areas.size else 0.0
    total_area = float(areas.sum()) if areas.size else 0.0
    dom_ratio = float(dom_area / max(1.0, total_area))
    info["dominant_area_ratio"] = dom_ratio

    if n_comp > int(cfg.max_components):
        reasons.append("too_many_components")
    if dom_ratio < float(cfg.dominant_area_ratio_min):
        reasons.append("dominant_component_too_small")

    # Pick dominant component as "main tongue blob" for shape checks
    dom_idx = int(np.argmax(areas)) if areas.size else 0
    dom_prop = props_all[dom_idx]
    dom_mask = (lab == int(dom_prop.label))

    # -----------------------------
    # Border-touch: hard noisy rule
    # -----------------------------
    bt = float(cfg.border_touch_frac)
    bt = max(0.0, min(0.49, bt))
    band_x = int(round(bt * w))
    band_y = int(round(bt * h))

    ys, xs = np.where(dom_mask)
    if ys.size > 0:
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        info["dom_bbox_xmin"] = x_min
        info["dom_bbox_xmax"] = x_max
        info["dom_bbox_ymin"] = y_min
        info["dom_bbox_ymax"] = y_max

        # Touching the 10% band on any side means bbox intersects that band.
        touch_left = (band_x > 0) and (x_min <= band_x)
        touch_right = (band_x > 0) and (x_max >= (w - 1 - band_x))
        touch_top = (band_y > 0) and (y_min <= band_y)
        touch_bottom = (band_y > 0) and (y_max >= (h - 1 - band_y))
        info["touch_left_band"] = bool(touch_left)
        info["touch_right_band"] = bool(touch_right)
        info["touch_top_band"] = bool(touch_top)
        info["touch_bottom_band"] = bool(touch_bottom)

        if touch_left or touch_right or touch_top or touch_bottom:
            reasons.append("touches_border_10pct_band")

    # -----------------------------
    # Topology: holes + broken skeleton
    # -----------------------------
    n_holes = _count_holes_in_roi(dom_mask, roi_pad_px=int(cfg.roi_pad_px))
    info["n_holes_roi"] = int(n_holes)
    if n_holes > int(cfg.max_holes):
        reasons.append("too_many_holes")

    # Skeleton fragmentation: multiple disconnected skeleton pieces suggests broken contour structure
    skel = skeletonize(dom_mask)
    sk_lab = label(skel, connectivity=2)
    n_skel = int(sk_lab.max())
    info["n_skeleton_components"] = n_skel
    if n_skel > int(cfg.max_skeleton_components):
        reasons.append("broken_skeleton")

    # -----------------------------
    # Shape plausibility: regionprops + perimeter heuristics
    # -----------------------------
    dom_area_px = float(dom_prop.area)
    info["dom_area_px"] = float(dom_area_px)
    info["dom_area_frac"] = float(dom_area_px) / float(max(1, h * w))

    # Shape checks are warnings only (secondary), not hard filters.
    if info["dom_area_frac"] < float(cfg.min_area_frac):
        warnings.append("area_too_small")
    if info["dom_area_frac"] > float(cfg.max_area_frac):
        warnings.append("area_too_large")

    # Regionprops on dominant CC
    info["eccentricity"] = float(getattr(dom_prop, "eccentricity", float("nan")))
    info["solidity"] = float(getattr(dom_prop, "solidity", float("nan")))
    major = float(getattr(dom_prop, "major_axis_length", 0.0))
    minor = float(getattr(dom_prop, "minor_axis_length", 0.0))
    info["major_axis_px"] = major
    info["minor_axis_px"] = minor
    elong = float(major / max(1e-6, minor))
    info["elongation"] = elong

    if np.isfinite(info["solidity"]) and info["solidity"] < float(cfg.min_solidity):
        warnings.append("low_solidity_spiky_or_concave")
    if np.isfinite(info["eccentricity"]) and info["eccentricity"] > float(cfg.max_eccentricity):
        warnings.append("too_line_like_eccentricity")
    if elong > float(cfg.max_elongation):
        warnings.append("too_elongated")
    if minor > 0.0 and minor < float(cfg.min_minor_axis_px):
        warnings.append("too_thin_minor_axis")

    # Perimeter/area spikiness proxy (higher => more jagged / fragmented)
    perim = float(sk_perimeter(dom_mask.astype(np.uint8), neighborhood=8))
    info["perimeter_px"] = perim
    pa = float(perim / max(1.0, dom_area_px))
    info["perimeter_area_ratio"] = pa
    if pa > float(cfg.max_perimeter_area_ratio):
        warnings.append("too_spiky_perimeter_area")

    info["reasons"] = reasons
    info["warnings"] = warnings
    is_clean = (len(reasons) == 0)
    return is_clean, info


def contour_check_folder(
    masks_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    cfg: TongueMaskQCConfig = TongueMaskQCConfig(),
    pattern: str = "*.png",
    recursive: bool = False,
    copy_mode: str = "copy",  # "copy" or "move"
) -> Dict[str, Any]:
    """
    Run QC on a directory of mask PNGs and split into clean/ and noisy/ subfolders.

    Writes:
    - `<masks_dir>/clean/*.png`
    - `<masks_dir>/noisy/*.png`
    - `<masks_dir>/quality_report.csv`
    """
    masks_dir = Path(masks_dir)
    if not masks_dir.is_dir():
        raise FileNotFoundError(str(masks_dir))

    out_root = masks_dir if output_dir is None else Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    clean_dir = out_root / "clean"
    noisy_dir = out_root / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    if recursive:
        paths = sorted(p for p in masks_dir.rglob(pattern) if p.is_file())
    else:
        paths = sorted(p for p in masks_dir.glob(pattern) if p.is_file())

    # Avoid reprocessing outputs if pattern matches them
    paths = [p for p in paths if clean_dir not in p.parents and noisy_dir not in p.parents]

    report_path = out_root / "quality_report.csv"
    rows: list[dict[str, Any]] = []

    n_clean = 0
    n_noisy = 0

    for p in paths:
        try:
            mask = _load_mask_png_to_bool(p)
            ok, info = check_mask_quality(mask, cfg=cfg)
            info_row = {
                "file": str(p.name),
                "ok": bool(ok),
                **{k: info.get(k) for k in info.keys()},
                "reasons": "|".join(info.get("reasons", [])),
                "warnings": "|".join(info.get("warnings", [])),
            }
            rows.append(info_row)

            dst = (clean_dir if ok else noisy_dir) / p.name
            if copy_mode == "move":
                shutil.move(str(p), str(dst))
            else:
                shutil.copy2(str(p), str(dst))

            if ok:
                n_clean += 1
            else:
                n_noisy += 1
        except Exception as e:
            rows.append({"file": str(p.name), "ok": False, "reasons": f"error:{type(e).__name__}:{e}"})
            dst = noisy_dir / p.name
            try:
                shutil.copy2(str(p), str(dst))
            except Exception:
                pass
            n_noisy += 1

    # Write CSV report
    # Determine headers as union of keys.
    headers: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return {
        "masks_dir": str(masks_dir),
        "output_dir": str(out_root),
        "pattern": pattern,
        "recursive": bool(recursive),
        "copy_mode": str(copy_mode),
        "cfg": asdict(cfg),
        "n_total": len(paths),
        "n_clean": int(n_clean),
        "n_noisy": int(n_noisy),
        "report_csv": str(report_path),
        "clean_dir": str(clean_dir),
        "noisy_dir": str(noisy_dir),
    }


def _print_usage() -> None:
    print("Usage:")
    print('  python contour_check.py "D:\\path\\to\\masks_folder"')
    print('  python contour_check.py "D:\\path\\to\\masks_folder" "D:\\path\\to\\output_dir"')


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        _print_usage()
        return 2
    masks_dir = Path(argv[1])
    output_dir = Path(argv[2]) if len(argv) >= 3 else None
    out = contour_check_folder(masks_dir, output_dir=output_dir)
    print(f"Done. n_total={out['n_total']} clean={out['n_clean']} noisy={out['n_noisy']}")
    print(f"clean: {out['clean_dir']}")
    print(f"noisy: {out['noisy_dir']}")
    print(f"report: {out['report_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))

