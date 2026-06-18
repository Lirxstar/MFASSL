#!/usr/bin/env python3
"""Paired test-set bootstrap CI + p-value for a metric gap (reproducibility utility)."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mfassl.eval.bootstrap import paired_bootstrap
from mfassl.eval.metrics import multilabel_auroc, multilabel_auprc, multilabel_f1
from mfassl.eval.calibration import expected_calibration_error, bce_nll, brier_score

METRICS = {
    "auroc": (multilabel_auroc, True),
    "auprc": (multilabel_auprc, True),
    "f1": (multilabel_f1, True),
    "ece": (expected_calibration_error, False),
    "nll": (bce_nll, False),
    "brier": (brier_score, False),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help=".npz with y_true, y_score")
    p.add_argument("--mfassl", required=True, help=".npz with y_true, y_score")
    p.add_argument("--metric", default="auroc", choices=list(METRICS))
    p.add_argument("--n_boot", type=int, default=10000)
    args = p.parse_args()

    base = np.load(args.baseline)
    ours = np.load(args.mfassl)
    assert np.array_equal(base["y_true"], ours["y_true"]), "models must share the test set"
    metric_fn, higher = METRICS[args.metric]

    res = paired_bootstrap(base["y_true"], base["y_score"], ours["y_score"],
                           metric_fn, n_boot=args.n_boot, higher_is_better=higher)
    print(f"metric={args.metric}  gap={res['observed_gap']:+.4f}  "
          f"95% CI=[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]  p={res['p_value']:.4f}")

if __name__ == "__main__":
    main()
