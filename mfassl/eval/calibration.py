"""Calibration / proper-scoring metrics: ECE, NLL, Brier (NumPy)."""

import numpy as np

def _flatten(y_true, y_prob):
    return np.asarray(y_true).ravel().astype(float), np.asarray(y_prob).ravel().astype(float)

def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:

    y_true, y_prob = _flatten(y_true, y_prob)

    pred = (y_prob >= 0.5).astype(float)
    conf = np.where(pred == 1, y_prob, 1.0 - y_prob)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)

def bce_nll(y_true, y_prob, eps: float = 1e-7) -> float:

    y_true, y_prob = _flatten(y_true, y_prob)
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

def brier_score(y_true, y_prob) -> float:

    y_true, y_prob = _flatten(y_true, y_prob)
    return float(np.mean((y_prob - y_true) ** 2))
