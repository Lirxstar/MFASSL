"""Segmentation metrics: Dice and 95th-percentile Hausdorff distance (HD95)."""

from typing import Optional

import numpy as np

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:

    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    a, b = pred.sum(), gt.sum()
    if a == 0 and b == 0:
        return 1.0
    if a == 0 or b == 0:
        return 0.0
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (a + b))

def _surface_points(mask: np.ndarray) -> np.ndarray:

    mask = np.asarray(mask).astype(bool)
    try:
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
    except Exception:
        boundary = np.zeros_like(mask)
        idx = np.argwhere(mask)
        for coord in idx:
            for dim in range(mask.ndim):
                for step in (-1, 1):
                    nb = coord.copy()
                    nb[dim] += step
                    if (nb[dim] < 0 or nb[dim] >= mask.shape[dim]
                            or not mask[tuple(nb)]):
                        boundary[tuple(coord)] = True
                        break
    return np.argwhere(boundary).astype(float)

def _directed_distances(a_pts: np.ndarray, b_pts: np.ndarray, spacing) -> np.ndarray:

    diff = (a_pts[:, None, :] - b_pts[None, :, :]) * spacing
    d = np.sqrt((diff ** 2).sum(axis=-1))
    return d.min(axis=1)

def hd95(pred: np.ndarray, gt: np.ndarray, spacing: Optional[np.ndarray] = None) -> float:

    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    sp = np.ones(pred.ndim) if spacing is None else np.asarray(spacing, dtype=float)
    sp_pts = sp[: pred.ndim]
    a = _surface_points(pred)
    b = _surface_points(gt)
    d_ab = _directed_distances(a, b, sp_pts)
    d_ba = _directed_distances(b, a, sp_pts)
    combined = np.concatenate([d_ab, d_ba])
    return float(np.percentile(combined, 95))
