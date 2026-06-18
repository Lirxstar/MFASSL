"""Aggregate per-seed metric dicts into mean ± std."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

def seed_path(path: Optional[str], seed: int, n_seeds: int) -> Optional[str]:

    if path is None or n_seeds <= 1:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_seed{seed}{ext}"

def aggregate_metrics(runs: List[Dict[str, float]], ddof: int = 1
                      ) -> Dict[str, Tuple[float, float]]:

    if not runs:
        return {}
    keys = list(runs[0].keys())
    out: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals = np.array([r[k] for r in runs if k in r], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[k] = (float("nan"), float("nan"))
        elif vals.size == 1:
            out[k] = (float(vals[0]), 0.0)
        else:
            out[k] = (float(vals.mean()), float(vals.std(ddof=ddof)))
    return out

def format_aggregated(agg: Dict[str, Tuple[float, float]], fmt: str = ".4f") -> str:

    return "  ".join(f"{k}={m:{fmt}}±{s:{fmt}}" for k, (m, s) in agg.items())
