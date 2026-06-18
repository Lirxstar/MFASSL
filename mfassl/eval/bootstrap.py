"""Paired test-set bootstrap for retraining-free confidence intervals (reproducibility utility)."""

from typing import Callable

import numpy as np

def paired_bootstrap(y_true: np.ndarray, score_baseline: np.ndarray, score_mfassl: np.ndarray,
                     metric_fn: Callable, n_boot: int = 10000, alpha: float = 0.05,
                     higher_is_better: bool = True, seed: int = 0) -> dict:

    y_true = np.asarray(y_true)
    score_baseline = np.asarray(score_baseline)
    score_mfassl = np.asarray(score_mfassl)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    sign = 1.0 if higher_is_better else -1.0

    observed = sign * (metric_fn(y_true, score_mfassl) - metric_fn(y_true, score_baseline))
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        m_b = metric_fn(y_true[idx], score_baseline[idx])
        m_m = metric_fn(y_true[idx], score_mfassl[idx])
        diffs[b] = sign * (m_m - m_b)

    lo = float(np.percentile(diffs, 100 * (alpha / 2)))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    p_one_sided = float(np.mean(diffs <= 0.0))
    return {
        "observed_gap": float(observed),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_one_sided,
        "mean_gap": float(diffs.mean()),
    }
