"""Classification metrics (AUROC, AUPRC, F1) in pure NumPy."""

from typing import Optional

import numpy as np

def _rankdata(x: np.ndarray) -> np.ndarray:

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)

    xs = x[order]
    i = 0
    n = len(xs)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks

def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n_pos = float((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))

def f1_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:

    y_true = np.asarray(y_true).ravel()
    y_pred = (np.asarray(y_score).ravel() >= threshold).astype(int)
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    if tp == 0 and (fp == 0 or fn == 0):
        return 0.0 if (fp > 0 or fn > 0) else 1.0
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def _macro(metric_fn, y_true: np.ndarray, y_score: np.ndarray, **kw) -> float:

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim == 1:
        return metric_fn(y_true, y_score, **kw)
    vals = []
    for k in range(y_true.shape[1]):
        v = metric_fn(y_true[:, k], y_score[:, k], **kw)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")

def multilabel_auroc(y_true, y_score) -> float:
    return _macro(roc_auc_binary, y_true, y_score)

def multilabel_auprc(y_true, y_score) -> float:
    return _macro(average_precision_binary, y_true, y_score)

def multilabel_f1(y_true, y_score, threshold: float = 0.5) -> float:
    return _macro(f1_binary, y_true, y_score, threshold=threshold)

def multilabel_accuracy(y_true, y_score, threshold: float = 0.5) -> float:

    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float((y_pred == y_true.astype(int)).mean())
