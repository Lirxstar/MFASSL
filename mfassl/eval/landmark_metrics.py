"""Facial-landmark metrics: NME, AUC@0.1, Failure@0.1, and flip-consistency (WFLW)."""

from typing import Optional

import numpy as np

def nme(pred: np.ndarray, gt: np.ndarray, norm_dist: np.ndarray) -> np.ndarray:

    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    err = np.sqrt(((pred - gt) ** 2).sum(axis=-1))
    return err.mean(axis=1) / np.asarray(norm_dist, dtype=float)

def auc_at(nme_values: np.ndarray, threshold: float = 0.1, n_steps: int = 1000) -> float:

    nme_values = np.asarray(nme_values, dtype=float)
    xs = np.linspace(0.0, threshold, n_steps + 1)
    ced = np.array([(nme_values <= x).mean() for x in xs])
    return float(np.trapz(ced, xs) / threshold)

def failure_rate(nme_values: np.ndarray, threshold: float = 0.1) -> float:

    return float((np.asarray(nme_values, dtype=float) > threshold).mean())

def landmark_flip_consistency(pred: np.ndarray, pred_flip: np.ndarray, image_width: float,
                              flip_index: Optional[np.ndarray] = None,
                              norm_dist: Optional[np.ndarray] = None) -> float:

    pred = np.asarray(pred, dtype=float)
    pred_flip = np.asarray(pred_flip, dtype=float).copy()
    pred_flip[..., 0] = image_width - pred_flip[..., 0]
    if flip_index is not None:
        pred_flip = pred_flip[:, np.asarray(flip_index), :]
    err = np.sqrt(((pred - pred_flip) ** 2).sum(axis=-1)).mean(axis=1)
    if norm_dist is not None:
        err = err / np.asarray(norm_dist, dtype=float)
    return float(np.clip(1.0 - err.mean(), 0.0, 1.0))
