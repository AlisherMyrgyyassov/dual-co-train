import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
from pathlib import Path
from skimage import io, morphology
from scipy import ndimage

EXAMPLE_MASK_PATH = r"D:\Alisher_Data\Datasets\Segmentation Conditioned Project\UXTD\first-masks\14M_053D_f00370.mask.png"


def _largest_connected_component_u8(mask01_u8: np.ndarray) -> np.ndarray:
    """
    Return uint8 {0,1} mask with only the largest connected component.
    Local helper to avoid depending on other modules.
    """
    m = (mask01_u8 > 0).astype(np.uint8)
    labeled, n = ndimage.label(m > 0)
    if int(n) <= 1:
        return m
    counts = np.bincount(labeled.ravel())
    if len(counts) <= 1:
        return m
    counts[0] = 0
    lab = int(np.argmax(counts))
    return (labeled == lab).astype(np.uint8)


def extract_non_lcc_artifacts(mask01_u8: np.ndarray) -> np.ndarray:
    """
    Return mask pixels that are foreground but NOT in the largest component.
    Intended for adding back pre-existing artifacts into a conditioning mask.
    """
    m = (mask01_u8 > 0).astype(np.uint8)
    lcc = _largest_connected_component_u8(m)
    return ((m > 0) & (lcc == 0)).astype(np.uint8)


def load_mask(path: str, threshold: float = 0.5) -> np.ndarray:
    """Load an image mask and return it as (H,W) uint8 {0,1}."""
    mask = io.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > threshold).astype(np.uint8)


def _skeletonize_mask(mask, threshold=0.5):
    if len(mask.shape) == 3:
        # If RGB/RGBA, convert to grayscale
        mask = mask[:, :, 0] > threshold
    else:
        mask = mask > threshold
    return morphology.skeletonize(mask)


# Proximity-based sorting (sort by nearest neighbor)
def sort_by_proximity(points):
    """Sort points by nearest neighbor starting from leftmost point"""
    if len(points) == 0: return points

    # Start with the leftmost point
    leftmost_idx = np.argmin(points[:, 0])

    sorted_points = [points[leftmost_idx]]
    remaining_indices = list(range(len(points)))
    remaining_indices.remove(leftmost_idx)
    
    while remaining_indices:
        last_point = sorted_points[-1]
        remaining_points = points[remaining_indices]
        
        distances = np.linalg.norm(remaining_points - last_point, axis=1)
        nearest_idx = np.argmin(distances)
        
        # Add to sorted list
        sorted_points.append(remaining_points[nearest_idx])
        remaining_indices.pop(nearest_idx)
    
    return np.array(sorted_points)


def _skeleton_adjacency(coords_rc):
    """
    coords_rc: list of (row, col) skeleton pixels.
    Returns adjacency list for 8-connectivity on the skeleton.
    """
    n = len(coords_rc)
    pos_to_i = {(int(r), int(c)): i for i, (r, c) in enumerate(coords_rc)}
    adj = [[] for _ in range(n)]
    for i, (r, c) in enumerate(coords_rc):
        r, c = int(r), int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = pos_to_i.get((r + dr, c + dc))
                if j is not None:
                    adj[i].append(j)
    return adj


def _prune_skeleton_spurs(skeleton_bool: np.ndarray, *, max_spur_len_px: int) -> np.ndarray:
    """
    Prune short leaf branches ("spurs") from a 1px skeleton.

    Motivation: skeletonization of a slightly jagged/thick mask can create tiny branches.
    The downstream ordering logic uses the skeleton graph diameter; without pruning, a
    short spur can become an endpoint and hijack the ordered polyline.

    Strategy:
    - Build 8-connected adjacency graph of skeleton pixels.
    - If there are no junctions (deg>=3), do nothing (likely a simple curve).
    - For each leaf (deg==1), walk toward the nearest junction/end following deg==2 nodes.
      If we reach a junction and the leaf-to-junction path length <= max_spur_len_px,
      remove that path (excluding the junction), and repeat until stable.
    """
    rmax = int(max_spur_len_px)
    if rmax <= 0:
        return skeleton_bool

    ys, xs = np.where(skeleton_bool)
    if len(xs) == 0:
        return skeleton_bool

    coords_rc = list(zip(ys.astype(int).tolist(), xs.astype(int).tolist()))
    n = int(len(coords_rc))
    if n < 5:
        return skeleton_bool

    # Build adjacency in index space
    pos_to_i = {(int(r), int(c)): i for i, (r, c) in enumerate(coords_rc)}

    def build_adj():
        adj = [[] for _ in range(n)]
        for i, (r, c) in enumerate(coords_rc):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    j = pos_to_i.get((int(r) + dr, int(c) + dc))
                    if j is not None:
                        adj[i].append(j)
        return adj

    keep = np.ones((n,), dtype=bool)
    changed = True
    while changed:
        changed = False
        adj = build_adj()
        deg = np.array([sum(keep[j] for j in nbrs) if keep[i] else 0 for i, nbrs in enumerate(adj)], dtype=int)
        junctions = {i for i in range(n) if keep[i] and deg[i] >= 3}
        if not junctions:
            break

        leaves = [i for i in range(n) if keep[i] and deg[i] == 1]
        to_remove: set[int] = set()

        for leaf in leaves:
            if not keep[leaf]:
                continue
            # Walk toward a junction following the chain.
            path = [leaf]
            prev = -1
            cur = leaf
            steps = 0
            hit_junction = False

            while steps <= rmax:
                # find next kept neighbor (excluding prev)
                nbrs = [j for j in adj[cur] if keep[j] and j != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                prev, cur = cur, nxt
                path.append(cur)
                steps += 1
                if cur in junctions:
                    hit_junction = True
                    break
                # stop if we reached another leaf/end (deg != 2) without junction
                if deg[cur] != 2 and cur not in junctions:
                    break

            if hit_junction and (len(path) - 1) <= rmax:
                # remove everything except the junction itself
                for idx in path[:-1]:
                    to_remove.add(idx)

        if to_remove:
            for idx in to_remove:
                keep[idx] = False
            changed = True

    if np.all(keep):
        return skeleton_bool

    out = np.zeros_like(skeleton_bool, dtype=bool)
    for i, (r, c) in enumerate(coords_rc):
        if keep[i]:
            out[int(r), int(c)] = True
    return out


def _bfs_farthest_with_parent(adj, n: int, start: int):
    parent = [-1] * n
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    far = start
    while q:
        u = q.popleft()
        if dist[u] > dist[far]:
            far = u
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    return far, parent


def _path_from_parent(parent, end: int):
    """Vertex sequence from root of BFS tree (parent=-1) to `end`, inclusive."""
    out = []
    cur = end
    while cur != -1:
        out.append(cur)
        cur = parent[cur]
    out.reverse()
    return out


def _tree_diameter_path_indices(adj, n: int):
    """Longest shortest path on a tree; vertex indices along the skeleton graph."""
    if n <= 1:
        return list(range(n))
    a, _ = _bfs_farthest_with_parent(adj, n, 0)
    b, parent = _bfs_farthest_with_parent(adj, n, a)
    return _path_from_parent(parent, b)


def _order_skeleton_open_polyline_xy(skeleton_bool: np.ndarray) -> np.ndarray:
    """
    Order skeleton pixels as an *open* polyline along the 1-pixel graph (no shortcut jumps).

    Uses tree diameter (double BFS) when the skeleton graph has leaves — typical for
    tongue medial axes. For a simple cycle (all junction degree 2), walks the cycle once
    without re-adding the closing edge so endpoints are not neighbors on the skeleton.
    """
    ys, xs = np.where(skeleton_bool)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    n = int(len(xs))
    coords_rc = list(zip(ys.astype(int).tolist(), xs.astype(int).tolist()))
    adj = _skeleton_adjacency(coords_rc)

    def to_xy_rowcol(idx_list):
        arr = np.zeros((len(idx_list), 2), dtype=np.float32)
        for k, i in enumerate(idx_list):
            r, c = coords_rc[int(i)]
            arr[k, 0] = float(c)
            arr[k, 1] = float(r)
        return arr

    if n == 1:
        return to_xy_rowcol([0])

    leaves = [i for i in range(n) if len(adj[i]) == 1]
    all_deg2 = n >= 3 and all(len(adj[i]) == 2 for i in range(n))

    if all_deg2:
        # Simple cycle: walk all n vertices in one direction (open along the cycle graph).
        def xy_key(i):
            r, c = coords_rc[i]
            return (c, r)

        start = int(min(range(n), key=xy_key))
        nbrs = list(adj[start])
        if len(nbrs) != 2:
            path_idx = _tree_diameter_path_indices(adj, n)
        else:
            n1, n2 = nbrs[0], nbrs[1]
            if xy_key(n1) > xy_key(n2):
                n1, n2 = n2, n1
            path_idx = [start]
            prev, cur = start, n1
            while len(path_idx) < n:
                path_idx.append(cur)
                nxt_list = [x for x in adj[cur] if x != prev]
                if not nxt_list:
                    break
                prev, cur = cur, nxt_list[0]
    elif len(leaves) >= 1:
        path_idx = _tree_diameter_path_indices(adj, n)
    else:
        path_idx = _tree_diameter_path_indices(adj, n)

    if len(path_idx) < 2:
        return to_xy_rowcol(path_idx) if path_idx else np.zeros((0, 2), dtype=np.float32)

    # Prefer tongue tip on the left (smaller x); flip if needed.
    xy_path = to_xy_rowcol(path_idx)
    if xy_path[-1, 0] < xy_path[0, 0]:
        xy_path = xy_path[::-1]
    return xy_path


from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter

# Parameters for deformation
N_CONTROL_POINTS = 12  # Number of control points for B-spline
NOISE_STD = 6.0  # Standard deviation of Gaussian noise for perturbation (y-axis only)
# Regularization helps prevent zigzag oscillations caused by pointwise noise.
SMOOTHNESS = 6.0

def deform_contour(
    sorted_points,
    n_control_points: int = N_CONTROL_POINTS,
    noise_std: float = NOISE_STD,
    spline_smoothness: float = SMOOTHNESS,
    image_size: int = 224,
    apply_y_smoothing: bool = True,
    y_smoothing_window: int = 5,
    y_smoothing_polyorder: int = 1.0,
    apply_x_smoothing: bool = False,
    noise_smoothing_window: int = 3,
    apply_arclength_resample: bool = True,
    # Local slope constraint to remove remaining high-frequency zigzags.
    # After smoothing/resampling, we limit |dy| between consecutive points.
    max_abs_dy: float = 4.5,
):
    def _moving_average_1d(a: np.ndarray, window: int) -> np.ndarray:
        window = int(window)
        if window <= 1 or len(a) < 2:
            return a
        window = min(window, len(a))
        # Ensure odd window for symmetric smoothing.
        if window % 2 == 0:
            window -= 1
        window = max(1, window)
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(a.astype(np.float32), kernel, mode="same").astype(np.float32)

    def _resample_by_arclength(points_xy: np.ndarray, n_points: int) -> np.ndarray:
        if points_xy is None or len(points_xy) < 2:
            return points_xy
        pts = points_xy.astype(np.float32)
        diffs = np.diff(pts, axis=0)
        seg_len = np.sqrt(np.sum(diffs * diffs, axis=1))
        s = np.concatenate([[0.0], np.cumsum(seg_len, dtype=np.float32)])
        total = float(s[-1])
        if total <= 1e-6:
            return pts
        s_uniform = np.linspace(0.0, total, int(n_points), dtype=np.float32)
        x_new = np.interp(s_uniform, s, pts[:, 0]).astype(np.float32)
        y_new = np.interp(s_uniform, s, pts[:, 1]).astype(np.float32)
        return np.stack([x_new, y_new], axis=1)

    # Fit B-spline to the contour
    tck, u = splprep([sorted_points[:, 0], sorted_points[:, 1]], 
                    s=spline_smoothness,  
                    k=min(3, len(sorted_points)-1))  # Cubic spline if enough points

    # Resample to get N control points
    u_control = np.linspace(0, 1, n_control_points)
    control_points = np.array(splev(u_control, tck)).T  # Shape: (N_CONTROL_POINTS, 2)

    # print(f"Original contour points: {len(sorted_points)}")
    # print(f"Control points: {len(control_points)}")

    # Perturb control points with Gaussian noise - Y-AXIS ONLY
    perturbed_control_points = control_points.copy()
    noise_y = np.random.randn(n_control_points).astype(np.float32) * float(noise_std)
    # Smooth the noise itself across control points to remove pointwise alternations
    # that otherwise create zigzag patterns.
    noise_y = _moving_average_1d(noise_y, window=noise_smoothing_window)
    perturbed_control_points[:, 0] = control_points[:, 0]  # FREEZE X-AXIS
    perturbed_control_points[:, 1] += noise_y  # Only perturb Y-AXIS

    # Keep endpoints fixed for more realistic deformation (optional)
    perturbed_control_points[0] = control_points[0]
    perturbed_control_points[-1] = control_points[-1]

    # Fit new B-spline through perturbed control points
    # (use same smoothness to avoid exact interpolation oscillations)
    tck_perturbed, u_new = splprep(
        [perturbed_control_points[:, 0], perturbed_control_points[:, 1]],
        s=float(spline_smoothness),
        k=min(3, n_control_points - 1),
    )

    # Resample the deformed curve with same number of points as original
    u_resample = np.linspace(0, 1, len(sorted_points))
    deformed_contour = np.array(splev(u_resample, tck_perturbed)).T.astype(np.float32)

    # Resample by arc-length so adjacent points represent local progression consistently.
    if apply_arclength_resample and len(deformed_contour) >= 4:
        deformed_contour = _resample_by_arclength(deformed_contour, n_points=len(sorted_points))

    # Constraint: suppress high-frequency zigzags with (1) smoothing and (2) local slope clamp.
    if apply_y_smoothing and len(deformed_contour) >= int(max(5, y_smoothing_window)):
        window = int(y_smoothing_window)
        if window % 2 == 0:
            window += 1
        # Savitzky requires window <= n_points and odd.
        max_window = len(deformed_contour) - 1 if (len(deformed_contour) % 2 == 0) else len(deformed_contour)
        window = min(window, max_window)
        if window >= 5 and window > int(y_smoothing_polyorder):
            deformed_contour[:, 1] = savgol_filter(
                deformed_contour[:, 1],
                window_length=window,
                polyorder=int(y_smoothing_polyorder),
                mode="interp",
            ).astype(np.float32)

    if apply_x_smoothing and len(deformed_contour) >= int(max(5, y_smoothing_window)):
        window = int(y_smoothing_window)
        if window % 2 == 0:
            window += 1
        max_window = len(deformed_contour) - 1 if (len(deformed_contour) % 2 == 0) else len(deformed_contour)
        window = min(window, max_window)
        if window >= 5 and window > int(y_smoothing_polyorder):
            deformed_contour[:, 0] = savgol_filter(
                deformed_contour[:, 0],
                window_length=window,
                polyorder=int(y_smoothing_polyorder),
                mode="interp",
            ).astype(np.float32)

    # Clamp local slope (first derivative) to remove remaining sharp alternations.
    if len(deformed_contour) >= 3:
        dy = np.diff(deformed_contour[:, 1]).astype(np.float32)
        dy = np.clip(dy, -float(max_abs_dy), float(max_abs_dy))
        y0 = float(deformed_contour[0, 1])
        y_new = np.concatenate([[y0], y0 + np.cumsum(dy, dtype=np.float32)]).astype(np.float32)
        deformed_contour[:, 1] = y_new

    # Keep points inside image bounds to avoid degenerate disks.
    deformed_contour[:, 0] = np.clip(deformed_contour[:, 0], 0.0, float(image_size - 1))
    deformed_contour[:, 1] = np.clip(deformed_contour[:, 1], 0.0, float(image_size - 1))
    return deformed_contour



# Note: this file previously contained an inline demo snippet.
# It has been removed because it referenced `deformed_contour` at import time.


from typing import Dict, Optional, Tuple, Union


DEFAULT_CIRCLE_RADIUS = 3  # Must match your initial 224x224 mask thickening
DEFAULT_GAUSSIAN_SIGMA = 3.0  # Heatmap blur sigma in pixels (approx. circle radius)

SYNTHETIC_TARGET_MASK_MODES = frozenset({"gauss", "circles"})


def normalize_synthetic_target_mask_mode(mode: str) -> str:
    """Validate co-training synthetic seg-target mode: ``gauss`` or ``circles``."""
    key = str(mode).strip().lower()
    if key not in SYNTHETIC_TARGET_MASK_MODES:
        raise ValueError(
            f"synthetic target mask mode must be one of {sorted(SYNTHETIC_TARGET_MASK_MODES)!r}, got {mode!r}"
        )
    return key


def build_synthetic_target_mask(
    points_xy: np.ndarray,
    aug01_u8: np.ndarray,
    *,
    image_size: int,
    mode: str = "gauss",
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    do_dilate: bool = False,
    dilate_radius_px: int = 1,
) -> np.ndarray:
    """
    Build a synthetic segmentation target mask from augmented contour points.

    - ``gauss`` (default): Gaussian heatmap via ``contour_points_to_gaussian_heatmap``.
    - ``circles``: binary mask from ``aug01_u8`` (circle-stamped contour from ``augment_mask``).

    GAN conditioning masks always use circles separately (``build_condmask_from_augmented_contour``).
    """
    mode = normalize_synthetic_target_mask_mode(mode)
    H = int(image_size)

    if mode == "gauss":
        gt = contour_points_to_gaussian_heatmap(
            points_xy,
            image_size=H,
            sigma=float(gaussian_sigma),
        )
        if do_dilate:
            r = max(0, int(dilate_radius_px))
            if r > 0:
                sup = (np.asarray(aug01_u8, dtype=np.uint8) > 0).astype(np.uint8)
                sup = morphology.binary_dilation(sup > 0, morphology.disk(r)).astype(np.uint8)
                gt = ndimage.gaussian_filter(sup.astype(np.float32), sigma=float(gaussian_sigma)).astype(
                    np.float32
                )
                mx = float(np.max(gt))
                if mx > 0:
                    gt = gt / mx
                gt = np.clip(gt, 0.0, 1.0).astype(np.float32)
        return gt.astype(np.float32)

    gt_u8 = (np.asarray(aug01_u8, dtype=np.uint8) > 0).astype(np.uint8)
    if do_dilate:
        r = max(0, int(dilate_radius_px))
        if r > 0:
            gt_u8 = morphology.binary_dilation(gt_u8 > 0, morphology.disk(r)).astype(np.uint8)
    return (gt_u8 > 0).astype(np.float32)

CONDMASK_CONTOUR_SOFT_PROB_DEFAULT = 0.10
CONDMASK_CONTOUR_PARTIAL_SOFT_PROB_DEFAULT = 0.10
CONDMASK_CONTOUR_PARTIAL_SOFT_FRAC_MIN_DEFAULT = 0.25
CONDMASK_CONTOUR_PARTIAL_SOFT_FRAC_MAX_DEFAULT = 0.50
CONDMASK_CONTOUR_SOFT_VALUE_DEFAULT = 0.9
CONDMASK_CONTOUR_HARD_VALUE_DEFAULT = 1.0


def mask_to_ordered_contour_points(
    mask: np.ndarray,
    image_size: int = 224,
    skeleton_threshold: float = 0.5,
    max_points: int = 250,
    spur_prune_len_px: int = 12,
) -> np.ndarray:
    """
    Convert a binary mask into ordered contour points (x,y).

    Uses skeletonization + graph walk along 8-connected skeleton pixels (tree diameter
    or one turn around a rare simple cycle). This avoids greedy Euclidean chaining, which
    can jump across the curve and visually close the contour when rasterized.
    """
    if mask is None:
        return np.zeros((0, 2), dtype=np.float32)

    mask_bool = mask
    if mask_bool.ndim == 3:
        mask_bool = mask_bool[..., 0]
    mask_bool = mask_bool > skeleton_threshold

    if mask_bool.shape != (image_size, image_size):
        raise ValueError(f"Expected mask shape {(image_size, image_size)}, got {mask_bool.shape}")

    skeleton = _skeletonize_mask(mask_bool, threshold=skeleton_threshold)
    if not np.any(skeleton):
        return np.zeros((0, 2), dtype=np.float32)

    # Keep only the largest connected component (reduces contour branching).
    labeled = morphology.label(skeleton)
    if labeled.max() > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest_label = int(np.argmax(counts))
        skeleton = labeled == largest_label

    # Prune short leaf branches that can hijack the diameter ordering.
    skeleton = _prune_skeleton_spurs(skeleton, max_spur_len_px=int(spur_prune_len_px))

    if int(np.count_nonzero(skeleton)) < 4:
        return np.zeros((0, 2), dtype=np.float32)

    points_xy = _order_skeleton_open_polyline_xy(skeleton).astype(np.float32)

    if len(points_xy) > max_points:
        idx = np.linspace(0, len(points_xy) - 1, max_points).astype(int)
        points_xy = points_xy[idx]

    return points_xy


def _densify_contour_points(
    points_xy: np.ndarray,
    *,
    densify_step_px: float = 0.75,
    densify_max_points: int = 600,
) -> np.ndarray:
    """Resample contour polyline by arc-length for stable disk rasterization."""
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if len(points_xy) < 2 or densify_step_px is None or densify_step_px <= 0:
        return points_xy

    diffs = np.diff(points_xy, axis=0)
    seg_len = np.sqrt(np.sum(diffs * diffs, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg_len, dtype=np.float32)])
    total = float(s[-1])
    if total <= 1e-6:
        return points_xy

    n_points = int(total / float(densify_step_px)) + 1
    n_points = max(n_points, len(points_xy))
    n_points = min(n_points, int(densify_max_points))
    if n_points == len(points_xy):
        return points_xy

    s_uniform = np.linspace(0.0, total, n_points, dtype=np.float32)
    x_new = np.interp(s_uniform, s, points_xy[:, 0]).astype(np.float32)
    y_new = np.interp(s_uniform, s, points_xy[:, 1]).astype(np.float32)
    return np.stack([x_new, y_new], axis=1)


def contour_points_to_condmask(
    points_xy: np.ndarray,
    image_size: int = 224,
    radius: int = DEFAULT_CIRCLE_RADIUS,
    *,
    soft_point_indices: Optional[set[int]] = None,
    soft_value: float = CONDMASK_CONTOUR_SOFT_VALUE_DEFAULT,
    hard_value: float = CONDMASK_CONTOUR_HARD_VALUE_DEFAULT,
    densify_step_px: float = 0.75,
    densify_max_points: int = 600,
    densify_points: bool = True,
) -> np.ndarray:
    """
    Rasterize contour points as overlapping disks with per-point intensities.

    Selected indices in ``soft_point_indices`` are stamped at ``soft_value`` (0.9);
    all other points use ``hard_value`` (1.0). Overlaps use pixel-wise maximum.

    Circles are drawn with ``utils.processing.draw_circle`` (Euclidean disk), not
    ``skimage.draw.disk`` (some versions rasterize small radii as filled squares).
    """
    from utils.processing import draw_circle

    cond = np.zeros((image_size, image_size), dtype=np.float32)
    if points_xy is None or len(points_xy) == 0:
        return cond

    pts = np.asarray(points_xy, dtype=np.float32)
    if densify_points:
        pts = _densify_contour_points(
            pts,
            densify_step_px=densify_step_px,
            densify_max_points=densify_max_points,
        )

    soft_set = soft_point_indices or set()
    r = int(radius)
    for i, (x, y) in enumerate(pts):
        val = float(soft_value) if i in soft_set else float(hard_value)
        x_int = int(round(float(x)))
        y_int = int(round(float(y)))
        draw_circle(cond, x_int, y_int, r, value=val, use_max=True)

    return cond


def contour_points_to_gaussian_heatmap(
    points_xy: np.ndarray,
    *,
    image_size: int = 224,
    sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    densify_step_px: float = 0.75,
    densify_max_points: int = 600,
    densify_points: bool = True,
) -> np.ndarray:
    """
    Rasterize contour points as a Gaussian heatmap (float32 in [0,1]).

    Implementation mirrors `utils.processing.generate_heatmap`:
    - place 1.0 at (x,y) points
    - apply `scipy.ndimage.gaussian_filter`
    - normalize by max to [0,1]
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    H = int(image_size)
    W = int(image_size)
    heatmap = np.zeros((H, W), dtype=np.float32)
    if pts is None or len(pts) == 0:
        return heatmap

    if densify_points:
        pts = _densify_contour_points(
            pts,
            densify_step_px=float(densify_step_px),
            densify_max_points=int(densify_max_points),
        )

    # Stamp 1s at rounded coordinates (clipped to bounds)
    xs = np.round(pts[:, 0]).astype(np.int32)
    ys = np.round(pts[:, 1]).astype(np.int32)
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)
    heatmap[ys, xs] = 1.0

    sig = float(sigma)
    if sig > 0:
        heatmap = ndimage.gaussian_filter(heatmap, sigma=sig).astype(np.float32)

    mx = float(np.max(heatmap))
    if mx > 0:
        heatmap = (heatmap / mx).astype(np.float32)
    return np.clip(heatmap, 0.0, 1.0).astype(np.float32)


def contour_points_to_mask(
    points_xy: np.ndarray,
    image_size: int = 224,
    radius: int = DEFAULT_CIRCLE_RADIUS,
    densify_step_px: float = 0.75,
    densify_max_points: int = 600,
    post_smooth_radius: int = 0,
) -> np.ndarray:
    """Convert ordered contour points (x,y) back into a thick 224x224 mask."""
    mask_f = contour_points_to_condmask(
        points_xy,
        image_size=image_size,
        radius=radius,
        soft_point_indices=set(),
        hard_value=1.0,
        densify_step_px=densify_step_px,
        densify_max_points=densify_max_points,
        densify_points=True,
    )
    mask = (mask_f > 0.0).astype(np.uint8)

    # Post-process mask to remove small boundary jaggies from rasterization.
    if post_smooth_radius is not None and int(post_smooth_radius) > 0:
        mask_bool = mask > 0
        mask_bool = morphology.binary_closing(mask_bool, morphology.disk(int(post_smooth_radius)))
        mask = mask_bool.astype(np.uint8)

    return mask


def _sample_outside_point_near_mask(
    mask01_u8: np.ndarray,
    *,
    dist_min_px: int,
    dist_max_px: int,
    angle_max_abs_deg: float,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    """
    Sample a point outside the mask at a specified distance and within an \"upper-cone\" direction.

    - Distance is measured to the closest mask pixel.
    - Direction is measured as the angle between vector (nearest_mask_pixel -> candidate) and the upward axis.
      Upward axis is toward smaller y (row) in image coordinates.
    """
    m = (mask01_u8 > 0).astype(np.uint8)
    if int(m.sum()) <= 0:
        return None

    outside = (m == 0)
    dist, (nr, nc) = ndimage.distance_transform_edt(outside, return_indices=True)

    dmin = max(0.0, float(dist_min_px))
    dmax = max(dmin, float(dist_max_px))
    cand = outside & (dist >= dmin) & (dist <= dmax)
    ys, xs = np.where(cand)
    if len(xs) == 0:
        return None

    # Vector v = candidate - nearest_mask_pixel:
    vy = ys.astype(np.float32) - nr[ys, xs].astype(np.float32)
    vx = xs.astype(np.float32) - nc[ys, xs].astype(np.float32)
    # angle=0 means straight up (vy negative). Use atan2(vx, -vy).
    ang = np.degrees(np.arctan2(vx, -vy + 1e-6))
    ok = np.abs(ang) <= float(angle_max_abs_deg)
    ys2, xs2 = ys[ok], xs[ok]
    if len(xs2) == 0:
        return None

    j = int(rng.integers(0, len(xs2)))
    return int(ys2[j]), int(xs2[j])


def _make_short_polyline_from_anchor(
    anchor_yx: tuple[int, int],
    *,
    mask01_u8: np.ndarray,
    length_px: int,
    rng: np.random.Generator,
    jitter_deg: float = 12.0,
) -> np.ndarray:
    """
    Create a short polyline (x,y) starting at anchor and extending roughly away from the mask.
    """
    y0, x0 = int(anchor_yx[0]), int(anchor_yx[1])
    m = (mask01_u8 > 0).astype(np.uint8)
    outside = (m == 0)
    _, (nr, nc) = ndimage.distance_transform_edt(outside, return_indices=True)
    y_near = float(nr[y0, x0])
    x_near = float(nc[y0, x0])

    dy = float(y0) - y_near
    dx = float(x0) - x_near
    n = float(np.hypot(dx, dy))
    if n < 1e-6:
        dx, dy = 0.0, -1.0
        n = 1.0
    dx /= n
    dy /= n

    jitter = np.deg2rad(float(rng.uniform(-jitter_deg, jitter_deg)))
    c, s = float(np.cos(jitter)), float(np.sin(jitter))
    dxj = c * dx - s * dy
    dyj = s * dx + c * dy

    L = max(1, int(length_px))
    xs = [float(x0)]
    ys = [float(y0)]
    for k in range(1, L + 1):
        xs.append(float(x0) + dxj * float(k))
        ys.append(float(y0) + dyj * float(k))
    return np.stack([np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)], axis=1)


def _make_curved_polyline_from_anchor(
    anchor_yx: tuple[int, int],
    *,
    mask01_u8: np.ndarray,
    length_px: int,
    rng: np.random.Generator,
    jitter_deg: float = 12.0,
) -> np.ndarray:
    """
    Create a short *curved* polyline (x,y) starting at anchor.

    Motivation: straight rays look synthetic; ultrasound tongue artifacts more often appear
    as slightly curved streaks/arcs (palate-like) or curved vertical clutter.

    Strategy:
    - Start from the "away from mask" direction (nearest-mask -> anchor), with jitter.
    - With some probability, bias the direction toward horizontal to form palate-like arcs.
    - Add curvature by applying a smooth perpendicular offset along the path (sine profile),
      producing a C-shaped "tongue-like" streak.
    """
    y0, x0 = int(anchor_yx[0]), int(anchor_yx[1])
    m = (np.asarray(mask01_u8, dtype=np.uint8) > 0).astype(np.uint8)
    outside = (m == 0)
    _, (nr, nc) = ndimage.distance_transform_edt(outside, return_indices=True)
    y_near = float(nr[y0, x0])
    x_near = float(nc[y0, x0])

    # Base direction: away from nearest mask pixel.
    dy = float(y0) - y_near
    dx = float(x0) - x_near
    n = float(np.hypot(dx, dy))
    if n < 1e-6:
        dx, dy = 0.0, -1.0
        n = 1.0
    dx /= n
    dy /= n

    # Randomly bias toward horizontal "palate-like" arcs sometimes.
    # This keeps artifacts from always pointing radially away.
    if float(rng.uniform(0.0, 1.0)) < 0.55:
        # Choose left/right horizontal direction with mild upward/downward component.
        sx = float(rng.choice([-1.0, 1.0]))
        dx, dy = sx, float(rng.uniform(-0.25, 0.25))
        n2 = float(np.hypot(dx, dy))
        dx, dy = dx / max(1e-6, n2), dy / max(1e-6, n2)

    # Apply small angle jitter.
    jitter = np.deg2rad(float(rng.uniform(-jitter_deg, jitter_deg)))
    c, s = float(np.cos(jitter)), float(np.sin(jitter))
    dxj = c * dx - s * dy
    dyj = s * dx + c * dy

    L = max(1, int(length_px))
    t = np.linspace(0.0, 1.0, L + 1, dtype=np.float32)

    # Perpendicular unit vector to (dxj,dyj).
    px, py = -dyj, dxj
    pn = float(np.hypot(px, py))
    if pn < 1e-6:
        px, py = 0.0, 1.0
        pn = 1.0
    px /= pn
    py /= pn

    # Curvature amplitude in pixels (keep small but visible).
    # Scale with length so longer streaks bend more.
    amp = float(rng.uniform(0.35, 0.95)) * max(1.0, 0.35 * float(L))
    amp *= float(rng.choice([-1.0, 1.0]))

    # Sine profile for a single smooth bend.
    bend = np.sin(np.pi * t).astype(np.float32)  # 0 -> 1 -> 0

    # Base straight path.
    xs = float(x0) + dxj * (t * float(L))
    ys = float(y0) + dyj * (t * float(L))

    # Add perpendicular curvature.
    xs = xs + (bend * float(amp)) * float(px)
    ys = ys + (bend * float(amp)) * float(py)

    return np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)


def _rasterize_polyline_disks(
    points_xy: np.ndarray,
    *,
    image_size: int,
    radius_px: int,
) -> np.ndarray:
    """
    Rasterize a short polyline as a thick line by stamping disks along the points.
    Returns float32 mask in [0,1].
    """
    r = max(0, int(radius_px))
    if points_xy is None or len(points_xy) == 0 or r <= 0:
        return np.zeros((image_size, image_size), dtype=np.float32)

    out = np.zeros((image_size, image_size), dtype=np.float32)
    # NOTE: We intentionally use morphology.disk() footprint instead of skimage.draw.disk().
    # In some scikit-image versions, draw.disk with small radii (e.g. r=3) rasterizes as a
    # fully filled square (5x5). The morphology footprint is the expected discrete disk.
    fp = morphology.disk(r).astype(np.float32)
    fh, fw = fp.shape
    cy = fh // 2
    cx = fw // 2
    for x, y in np.asarray(points_xy, dtype=np.float32):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        y0 = yi - cy
        x0 = xi - cx
        y1 = y0 + fh
        x1 = x0 + fw

        if y1 <= 0 or x1 <= 0 or y0 >= image_size or x0 >= image_size:
            continue

        yy0 = max(0, y0)
        xx0 = max(0, x0)
        yy1 = min(image_size, y1)
        xx1 = min(image_size, x1)

        fy0 = yy0 - y0
        fx0 = xx0 - x0
        fy1 = fy0 + (yy1 - yy0)
        fx1 = fx0 + (xx1 - xx0)

        patch = fp[int(fy0):int(fy1), int(fx0):int(fx1)]
        out[int(yy0):int(yy1), int(xx0):int(xx1)] = np.maximum(
            out[int(yy0):int(yy1), int(xx0):int(xx1)],
            patch,
        )
    return out


def build_condmask_from_contour(
    contour01_u8: np.ndarray,
    *,
    prob_soft_contour: float = CONDMASK_CONTOUR_SOFT_PROB_DEFAULT,
    soft_contour_value: float = CONDMASK_CONTOUR_SOFT_VALUE_DEFAULT,
    hard_contour_value: float = CONDMASK_CONTOUR_HARD_VALUE_DEFAULT,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Initialize a float GAN conditioning mask from a binary contour mask.

    With probability ``prob_soft_contour`` (default 10%), contour pixels are set to
    ``soft_contour_value`` (0.9) instead of ``hard_contour_value`` (1.0).
    Artifact pixels added later via ``np.maximum`` are unchanged.
    """
    contour = np.asarray(contour01_u8, dtype=np.uint8) > 0
    val = float(hard_contour_value)
    p = float(prob_soft_contour)
    if p > 0.0:
        u = float(rng.random()) if rng is not None else float(np.random.random())
        if u < p:
            val = float(soft_contour_value)
    cond01 = np.zeros(contour.shape, dtype=np.float32)
    cond01[contour] = val
    return cond01


def build_condmask_from_augmented_contour(
    points_xy: np.ndarray,
    *,
    image_size: int,
    circle_radius: int = DEFAULT_CIRCLE_RADIUS,
    rng: np.random.Generator,
    prob_partial_soft: float = CONDMASK_CONTOUR_PARTIAL_SOFT_PROB_DEFAULT,
    partial_soft_frac_min: float = CONDMASK_CONTOUR_PARTIAL_SOFT_FRAC_MIN_DEFAULT,
    partial_soft_frac_max: float = CONDMASK_CONTOUR_PARTIAL_SOFT_FRAC_MAX_DEFAULT,
    prob_full_soft: float = CONDMASK_CONTOUR_SOFT_PROB_DEFAULT,
    soft_value: float = CONDMASK_CONTOUR_SOFT_VALUE_DEFAULT,
    hard_value: float = CONDMASK_CONTOUR_HARD_VALUE_DEFAULT,
    contour01_u8_fallback: Optional[np.ndarray] = None,
    densify_step_px: float = 0.75,
    densify_max_points: int = 600,
) -> np.ndarray:
    """
    Build a float GAN conditioning mask by rasterizing augmented contour points.

    Modes (mutually exclusive per call, checked in order):
    - **Partial soft** (``prob_partial_soft``, default 10%): pick 25–50% of densified
      contour points at random; stamp those disks at 0.9, others at 1.0.
    - **Full soft** (``prob_full_soft``, default 10%): all contour disks at 0.9.
    - **Default**: all contour disks at 1.0.

    Falls back to ``build_condmask_from_contour`` if no contour points are available.
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(pts) == 0:
        if contour01_u8_fallback is not None:
            return build_condmask_from_contour(
                contour01_u8_fallback,
                prob_soft_contour=float(prob_full_soft),
                soft_contour_value=float(soft_value),
                hard_contour_value=float(hard_value),
                rng=rng,
            )
        return np.zeros((int(image_size), int(image_size)), dtype=np.float32)

    pts_dense = _densify_contour_points(
        pts,
        densify_step_px=float(densify_step_px),
        densify_max_points=int(densify_max_points),
    )
    n = len(pts_dense)
    p_part = float(prob_partial_soft)
    p_full = float(prob_full_soft)
    r = float(rng.random())

    if r < p_part and n > 0:
        frac_lo = float(partial_soft_frac_min)
        frac_hi = float(partial_soft_frac_max)
        frac = float(rng.uniform(min(frac_lo, frac_hi), max(frac_lo, frac_hi)))
        n_soft = max(1, int(round(frac * n)))
        n_soft = min(n_soft, n)
        soft_idx = {int(i) for i in rng.choice(n, size=n_soft, replace=False)}
        return contour_points_to_condmask(
            pts_dense,
            image_size=int(image_size),
            radius=int(circle_radius),
            soft_point_indices=soft_idx,
            soft_value=float(soft_value),
            hard_value=float(hard_value),
            densify_points=False,
        )

    if r < (p_part + p_full):
        return contour_points_to_condmask(
            pts_dense,
            image_size=int(image_size),
            radius=int(circle_radius),
            soft_point_indices=set(range(n)),
            soft_value=float(soft_value),
            hard_value=float(hard_value),
            densify_points=False,
        )

    return contour_points_to_condmask(
        pts_dense,
        image_size=int(image_size),
        radius=int(circle_radius),
        soft_point_indices=set(),
        soft_value=float(soft_value),
        hard_value=float(hard_value),
        densify_points=False,
    )


def add_synthetic_artifacts_to_conditioning_mask(
    cond01_hw: np.ndarray,
    *,
    gt_mask01_u8: np.ndarray,
    rng: np.random.Generator,
    n_artifacts_min: int = 1,
    n_artifacts_max: int = 3,
    value_min: float = 0.4,
    value_max: float = 0.8,
    dist_min_px: int = 20,
    dist_max_px: int = 40,
    angle_max_abs_deg: float = 30.0,
    skeleton_len_min_px: int = 3,
    skeleton_len_max_px: int = 6,
    artifact_radius_px: int = 3,
    safety_margin_px: int = 1,
) -> np.ndarray:
    """
    Add 1–3 small line-like artifacts above the mask into a *conditioning* mask.

    The artifacts are short skeleton polylines (3–6 px) rasterized by stamping disks
    around the polyline points (similar to contour mask generation, but tiny).

    Returns a float32 mask in [0,1].
    """
    cond = np.asarray(cond01_hw, dtype=np.float32).copy()
    gt = (np.asarray(gt_mask01_u8, dtype=np.uint8) > 0).astype(np.uint8)
    H, W = gt.shape[:2]
    if cond.shape[:2] != (H, W):
        raise ValueError(f"cond/gt size mismatch: cond={cond.shape}, gt={gt.shape}")

    margin = max(0, int(safety_margin_px))
    forbidden = gt.astype(bool)
    if margin > 0:
        forbidden = morphology.binary_dilation(forbidden, morphology.disk(margin))

    n_min = int(n_artifacts_min)
    n_max = int(n_artifacts_max)
    n_max = max(n_min, n_max)
    n = int(rng.integers(n_min, n_max + 1))

    for _ in range(n):
        anchor = _sample_outside_point_near_mask(
            gt,
            dist_min_px=int(dist_min_px),
            dist_max_px=int(dist_max_px),
            angle_max_abs_deg=float(angle_max_abs_deg),
            rng=rng,
        )
        if anchor is None:
            continue
        y0, x0 = int(anchor[0]), int(anchor[1])
        if forbidden[y0, x0]:
            continue

        sk_len = int(rng.integers(int(skeleton_len_min_px), int(skeleton_len_max_px) + 1))
        pts_xy = _make_curved_polyline_from_anchor(anchor, mask01_u8=gt, length_px=sk_len, rng=rng)
        art01 = _rasterize_polyline_disks(pts_xy, image_size=H, radius_px=int(artifact_radius_px))

        art01 = art01 * (~forbidden).astype(np.float32)
        if float(art01.sum()) <= 0.0:
            continue

        val = float(rng.uniform(float(value_min), float(value_max)))
        cond = np.maximum(cond, art01 * val)

    return np.clip(cond, 0.0, 1.0).astype(np.float32)


def move_contour(
    points_xy: np.ndarray,
    move_px_range: Tuple[int, int] = (5, 15),
    left_right_margin_ratio: float = 0.10,
    image_size: int = 224,
    max_tries: int = 100,
) -> np.ndarray:
    """Move contour by a random translation of 5-15px in a random direction."""
    if points_xy is None or len(points_xy) == 0:
        return points_xy

    points_xy = np.asarray(points_xy, dtype=np.float32)
    min_px, max_px = int(move_px_range[0]), int(move_px_range[1])
    min_px = max(0, min_px)
    max_px = max(min_px, max_px)

    margin = int(round(left_right_margin_ratio * (image_size - 1)))
    x_min_allowed = float(margin)
    x_max_allowed = float((image_size - 1) - margin)

    for _ in range(int(max_tries)):
        magnitude = float(np.random.uniform(min_px, max_px))
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        dx = magnitude * float(np.cos(angle))
        dy = magnitude * float(np.sin(angle))

        moved = points_xy + np.array([dx, dy], dtype=np.float32)
        if np.all(moved[:, 0] >= x_min_allowed) and np.all(moved[:, 0] <= x_max_allowed):
            moved[:, 1] = np.clip(moved[:, 1], 0.0, float(image_size - 1))
            return moved

    # Fallback: hard clamp to respect constraints.
    magnitude = float(np.random.uniform(min_px, max_px))
    angle = float(np.random.uniform(0.0, 2.0 * np.pi))
    dx = magnitude * float(np.cos(angle))
    dy = magnitude * float(np.sin(angle))
    moved = points_xy + np.array([dx, dy], dtype=np.float32)
    moved[:, 0] = np.clip(moved[:, 0], x_min_allowed, x_max_allowed)
    moved[:, 1] = np.clip(moved[:, 1], 0.0, float(image_size - 1))
    return moved


def rotate_contour(
    points_xy: np.ndarray,
    angle_deg_range: Tuple[int, int] = (5, 20),
    image_size: int = 224,
) -> np.ndarray:
    """Rotate contour by a random angle in degrees (5..20), around centroid."""
    if points_xy is None or len(points_xy) == 0:
        return points_xy

    points_xy = np.asarray(points_xy, dtype=np.float32)
    min_deg, max_deg = int(angle_deg_range[0]), int(angle_deg_range[1])
    min_deg = max(0, min_deg)
    max_deg = max(min_deg, max_deg)

    angle_deg = float(np.random.uniform(min_deg, max_deg))
    angle_deg *= float(np.random.choice([-1.0, 1.0]))
    theta = np.deg2rad(angle_deg)

    centroid = points_xy.mean(axis=0, keepdims=True)  # (1,2)
    centered = points_xy - centroid

    c, s = float(np.cos(theta)), float(np.sin(theta))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)  # operates on (x,y)
    rotated = centered @ rot.T + centroid

    rotated[:, 0] = np.clip(rotated[:, 0], 0.0, float(image_size - 1))
    rotated[:, 1] = np.clip(rotated[:, 1], 0.0, float(image_size - 1))
    return rotated


def augment_mask(
    mask: np.ndarray,
    image_size: int = 224,
    circle_radius: int = DEFAULT_CIRCLE_RADIUS,
    do_deform: bool = False,
    do_move: bool = False,
    do_rotate: bool = False,
    do_dilate: bool = False,
    dilate_radius_px: int = 1,
    deformation_params: Optional[Dict] = None,
    move_params: Optional[Dict] = None,
    rotate_params: Optional[Dict] = None,
    skeleton_threshold: float = 0.5,
    return_contour: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply selected contour augmentations and convert back to 224x224 mask.

    Operations are applied in this order (only enabled ones run): deform -> move -> rotate
    """
    deformation_params = deformation_params or {}
    move_params = move_params or {}
    rotate_params = rotate_params or {}

    points_xy = mask_to_ordered_contour_points(
        mask=mask,
        image_size=image_size,
        skeleton_threshold=skeleton_threshold,
    )
    if len(points_xy) == 0:
        empty = np.zeros((image_size, image_size), dtype=np.uint8)
        return (empty, points_xy) if return_contour else empty

    # 1) Contour deformation (reference)
    if do_deform:
        points_xy = deform_contour(points_xy, **deformation_params)

    # 2) Random contour movement with X constraint
    if do_move:
        points_xy = move_contour(points_xy, image_size=image_size, **move_params)

    # 3) Contour rotation
    if do_rotate:
        points_xy = rotate_contour(points_xy, image_size=image_size, **rotate_params)

    out_mask = contour_points_to_mask(points_xy, image_size=image_size, radius=circle_radius)

    if do_dilate:
        r = int(dilate_radius_px)
        r = max(0, r)
        if r > 0:
            out_mask = morphology.binary_dilation(out_mask > 0, morphology.disk(r)).astype(np.uint8)
        else:
            out_mask = (out_mask > 0).astype(np.uint8)
    return (out_mask, points_xy) if return_contour else out_mask


def visualize_augmentations_for_one_mask(
    mask: np.ndarray,
    output_dir: str,
    image_size: int = 224,
    circle_radius: int = DEFAULT_CIRCLE_RADIUS,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    variants = [
        ("original", dict(do_deform=False, do_move=False, do_rotate=False)),
        ("dilate", dict(do_deform=False, do_move=False, do_rotate=False, do_dilate=True, dilate_radius_px=1)),
        ("deform", dict(do_deform=True, do_move=False, do_rotate=False)),
        ("move", dict(do_deform=False, do_move=True, do_rotate=False)),
        ("rotate", dict(do_deform=False, do_move=False, do_rotate=True)),
        ("deform+move", dict(do_deform=True, do_move=True, do_rotate=False)),
        ("deform+rotate", dict(do_deform=True, do_move=False, do_rotate=True)),
        ("move+rotate", dict(do_deform=False, do_move=True, do_rotate=True)),
        ("deform+move+rotate", dict(do_deform=True, do_move=True, do_rotate=True)),
        ("deform+move+rotate+dilate", dict(do_deform=True, do_move=True, do_rotate=True, do_dilate=True, dilate_radius_px=1)),
    ]

    cols = 4
    n = len(variants)
    rows = int(np.ceil(n / cols))

    # Grid A: masks only
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.asarray(axes).reshape(-1)

    for i, (name, flags) in enumerate(variants):
        aug = augment_mask(
            mask,
            image_size=image_size,
            circle_radius=circle_radius,
            **flags,
        )
        ax = axes[i]
        ax.imshow(aug, cmap="gray", interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    grid_a_path = os.path.join(output_dir, "augmentations_grid_masks.png")
    fig.savefig(grid_a_path, dpi=200)
    plt.close(fig)

    # Grid B: masks + contour points
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.asarray(axes).reshape(-1)

    for i, (name, flags) in enumerate(variants):
        aug_mask, aug_points = augment_mask(
            mask,
            image_size=image_size,
            circle_radius=circle_radius,
            return_contour=True,
            **flags,
        )
        ax = axes[i]
        ax.imshow(aug_mask, cmap="gray", interpolation="nearest")
        if len(aug_points) > 0:
            ax.scatter(aug_points[:, 0], aug_points[:, 1], s=6, c="red", linewidths=0)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    grid_b_path = os.path.join(output_dir, "augmentations_grid_masks_with_contour.png")
    fig.savefig(grid_b_path, dpi=200)
    plt.close(fig)


def visualize_random_deformed_contours(
    mask: np.ndarray,
    output_dir: str,
    n_samples: int = 10,
    image_size: int = 224,
    skeleton_threshold: float = 0.5,
    deformer_params: Optional[Dict] = None,
) -> None:
    """
    Save a 5x2 subplot of ONLY deformed contour lines (no masks).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    deformer_params = deformer_params or {}
    base_points = mask_to_ordered_contour_points(
        mask=mask,
        image_size=image_size,
        skeleton_threshold=skeleton_threshold,
    )
    if len(base_points) == 0:
        print("[augmentations_contour] No contour found; skipping random deformed contour visualization.")
        return

    n_samples = int(n_samples)
    # Requested fixed layout: 5x2 for 10 samples.
    rows, cols = 2, 5
    if n_samples != rows * cols:
        rows = int(np.ceil(n_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.asarray(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.set_xlim(0.0, float(image_size - 1))
        ax.set_ylim(float(image_size - 1), 0.0)  # invert for image-like orientation
        ax.axis("off")

        if i >= n_samples:
            continue

        pts = deform_contour(base_points, **deformer_params)
        ax.plot(pts[:, 0], pts[:, 1], color="blue", linewidth=2)
        ax.set_title(f"#{i+1}", fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "random_deformed_contours.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    def _parse_args() -> argparse.Namespace:
        repo_root = Path(__file__).resolve().parent
        default_ckpt_dir = repo_root / "checkpoints" / "pretrain-conditional"
        default_ckpt = default_ckpt_dir / "best_model.pth"

        p = argparse.ArgumentParser(description="Contour augmentations + GAN visualizations")
        p.add_argument(
            "--checkpoint",
            type=str,
            default=str(default_ckpt),
            help="Path to generator checkpoint .pth (default: best_model.pth in pretrain-conditional)",
        )
        p.add_argument("--noise-amplitude", type=float, default=0.1, help="Noise std used for GAN conditioning")
        p.add_argument("--noise-seed", type=int, default=42, help="Seed for deterministic GAN noise")
        p.add_argument("--device", type=str, default="", help="'' auto, or 'cpu'/'cuda'")
        p.add_argument("--num-variants", type=int, default=8, help="How many variants from the predefined list")
        return p.parse_args()

    def _load_gan_generator(checkpoint_path: str, device: str):
        import torch
        from networks.unet import UNet

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(str(ckpt_path))

        ckpt = torch.load(str(ckpt_path), map_location=device)
        gen_state = ckpt.get("generator_state_dict", ckpt)

        # Instantiate generator. Training used UNet(n_channels=2, n_classes=1)
        generator = UNet(n_channels=2, n_classes=1)

        # Be tolerant to possible 'module.' prefixes.
        cleaned_state = {}
        for k, v in gen_state.items():
            cleaned_state[k.replace("module.", "")] = v

        generator.load_state_dict(cleaned_state, strict=False)
        generator.to(device)
        generator.eval()
        return generator

    def _mask_to_generator_input(
        mask_224: np.ndarray,
        noise_amplitude: float,
        noise_seed: int,
        device: str,
        *,
        condmask_contour_soft_prob: float = CONDMASK_CONTOUR_SOFT_PROB_DEFAULT,
        condmask_contour_soft_value: float = CONDMASK_CONTOUR_SOFT_VALUE_DEFAULT,
    ):
        import torch

        # mask_224 expected shape (H,W), uint/float
        if mask_224.ndim != 2:
            raise ValueError(f"Expected 2D mask (H,W), got shape={mask_224.shape}")
        h, w = int(mask_224.shape[0]), int(mask_224.shape[1])
        rng = np.random.default_rng(int(noise_seed))
        m01 = (np.asarray(mask_224, dtype=np.uint8) > 0).astype(np.uint8)
        pts = mask_to_ordered_contour_points(m01, image_size=h, skeleton_threshold=0.5)
        mask = build_condmask_from_augmented_contour(
            pts,
            image_size=h,
            circle_radius=DEFAULT_CIRCLE_RADIUS,
            rng=rng,
            prob_full_soft=float(condmask_contour_soft_prob),
            soft_value=float(condmask_contour_soft_value),
            contour01_u8_fallback=m01,
        )

        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)  # (1,H,W)

        rng = torch.Generator(device=device)
        rng.manual_seed(int(noise_seed))
        noise = torch.randn((1, h, w), generator=rng, device=device) * float(noise_amplitude)
        noise = noise.to(device)

        input_cond = torch.cat([mask_t, noise], dim=0).unsqueeze(0)  # (1,2,H,W)
        return input_cond

    def _gan_visualize_for_variants(
        example_mask: np.ndarray,
        generator,
        device: str,
        output_dir: Path,
        circle_radius: int = DEFAULT_CIRCLE_RADIUS,
        image_size: int = 224,
        noise_amplitude: float = 0.1,
        noise_seed: int = 42,
        num_variants: int = 8,
    ) -> None:
        import torch

        output_dir.mkdir(parents=True, exist_ok=True)

        variants = [
            ("original", dict(do_deform=False, do_move=False, do_rotate=False)),
            ("deform", dict(do_deform=True, do_move=False, do_rotate=False)),
            ("move", dict(do_deform=False, do_move=True, do_rotate=False)),
            ("rotate", dict(do_deform=False, do_move=False, do_rotate=True)),
            ("deform+move", dict(do_deform=True, do_move=True, do_rotate=False)),
            ("deform+rotate", dict(do_deform=True, do_move=False, do_rotate=True)),
            ("move+rotate", dict(do_deform=False, do_move=True, do_rotate=True)),
            ("deform+move+rotate", dict(do_deform=True, do_move=True, do_rotate=True)),
        ][: int(num_variants)]

        # Generate augmented masks first.
        aug_masks = []
        for _name, flags in variants:
            aug_masks.append(
                augment_mask(
                    example_mask,
                    image_size=image_size,
                    circle_radius=circle_radius,
                    return_contour=False,
                    skeleton_threshold=0.5,
                    **flags,
                )
            )

        # Run generator
        gen_images = []
        with torch.no_grad():
            for i, m in enumerate(aug_masks):
                # Use different but deterministic seed per variant.
                input_cond = _mask_to_generator_input(m, noise_amplitude=noise_amplitude, noise_seed=noise_seed + i, device=device)
                fake = generator(input_cond)
                fake = torch.clamp(fake, 0, 1)
                gen_images.append(fake.squeeze(0).squeeze(0).cpu().numpy())  # (H,W)

        # Grid 1: mask + GAN output
        rows = len(variants)
        fig, axes = plt.subplots(rows, 2, figsize=(6, rows * 2.2))
        if rows == 1:
            axes = np.asarray(axes).reshape(1, 2)

        for r, ((name, _flags), aug_mask, gen_img) in enumerate(zip(variants, aug_masks, gen_images)):
            axes[r, 0].imshow(aug_mask, cmap="gray", interpolation="nearest")
            axes[r, 0].set_title(f"{name} (mask)", fontsize=10)
            axes[r, 0].axis("off")

            axes[r, 1].imshow(gen_img, cmap="gray", interpolation="nearest")
            axes[r, 1].set_title(f"{name} (GAN)", fontsize=10)
            axes[r, 1].axis("off")

        fig.tight_layout()
        out_path = output_dir / "gan_from_augmented_masks.png"
        fig.savefig(str(out_path), dpi=200)
        plt.close(fig)

        # Grid 2: GAN outputs only
        n = len(variants)
        cols = 4
        grid_rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 2.4, grid_rows * 2.4))
        axes = np.asarray(axes).reshape(-1)
        for i, ((name, _flags), gen_img) in enumerate(zip(variants, gen_images)):
            axes[i].imshow(gen_img, cmap="gray", interpolation="nearest")
            axes[i].set_title(name, fontsize=10)
            axes[i].axis("off")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        out_path = output_dir / "gan_outputs_only_grid.png"
        fig.savefig(str(out_path), dpi=200)
        plt.close(fig)

    # Save visualization grids to ./visualizations/augmentations/
    out_dir = Path(__file__).resolve().parent / "visualizations" / "augmentations"

    args = _parse_args()

    if not os.path.exists(EXAMPLE_MASK_PATH):
        print(f"[augmentations_contour] Example mask not found: {EXAMPLE_MASK_PATH}")
        print("[augmentations_contour] Update EXAMPLE_MASK_PATH in this file to a valid mask to generate visualizations.")
        raise SystemExit(1)

    example_mask = load_mask(EXAMPLE_MASK_PATH, threshold=0.5)
    visualize_augmentations_for_one_mask(example_mask, str(out_dir))
    print(f"[augmentations_contour] Saved augmentation visualizations to: {out_dir}")

    # Extra: 10 random deform-only contours (5x2) for quick sanity checks.
    visualize_random_deformed_contours(
        mask=example_mask,
        output_dir=str(out_dir),
        n_samples=10,
        image_size=224,
        skeleton_threshold=0.5,
        deformer_params={
            "apply_y_smoothing": True,
            "apply_arclength_resample": True,
            "apply_x_smoothing": False,
        },
    )

    # Optional GAN visualization
    try:
        import torch

        if args.device.strip() != "":
            device = args.device.strip()
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        generator = _load_gan_generator(args.checkpoint, device=device)
        gan_out_dir = out_dir / "gan_from_masks"
        _gan_visualize_for_variants(
            example_mask=example_mask,
            generator=generator,
            device=device,
            output_dir=gan_out_dir,
            noise_amplitude=args.noise_amplitude,
            noise_seed=args.noise_seed,
            num_variants=args.num_variants,
        )
        print(f"[augmentations_contour] Saved GAN visualizations to: {gan_out_dir}")
    except FileNotFoundError as e:
        print(f"[augmentations_contour] GAN checkpoint not found, skipping GAN visualizations: {e}")
    except Exception as e:
        import traceback
        print(f"[augmentations_contour] GAN visualization failed, skipping: {type(e).__name__}: {e}")
        traceback.print_exc()