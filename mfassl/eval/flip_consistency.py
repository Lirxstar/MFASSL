"""Flip-Consistency: prediction agreement between an image and its horizontal reflection."""

import numpy as np

def multilabel_flip_consistency(prob: np.ndarray, prob_flip: np.ndarray,
                                threshold: float = 0.5,
                                reduction: str = "per_image") -> float:

    pred = (np.asarray(prob) >= threshold).astype(int)
    pred_flip = (np.asarray(prob_flip) >= threshold).astype(int)
    agree = (pred == pred_flip)
    if reduction == "per_image":
        return float(agree.all(axis=1).mean())
    if reduction == "per_label":
        return float(agree.mean())
    raise ValueError(f"unknown reduction '{reduction}' (expected 'per_image' or 'per_label')")
