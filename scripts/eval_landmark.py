#!/usr/bin/env python3
"""Downstream WFLW landmark-localization eval."""

import argparse
import os
import sys

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mfassl.utils.config import load_config
from mfassl.utils.seed import seed_everything
from mfassl.engine.build import resolve_device, load_encoder_from_checkpoint
from mfassl.data.datasets.wflw import (
    WFLWDataset, NUM_LANDMARKS, WFLW_FLIP_INDEX, WFLW_INTEROCULAR,
)
from mfassl.eval.landmark import LandmarkModel, train_landmark, evaluate_landmark
from mfassl.eval.aggregate import aggregate_metrics, format_aggregated

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="run once per seed and report mean±std; default: the config's seed")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = resolve_device(cfg.device)
    seeds = args.seeds if args.seeds is not None else [cfg.seed]

    root = cfg.data.wflw_root
    train_ds = WFLWDataset(root, cfg.data.get("train_ann", "list_98pt_rect_attr_train.txt"),
                           img_size=cfg.backbone.img_size, limit=cfg.data.get("limit"))
    test_ds = WFLWDataset(root, cfg.data.get("test_ann", "list_98pt_rect_attr_test.txt"),
                          img_size=cfg.backbone.img_size)
    bs = cfg.data.batch_size

    norm_index = tuple(cfg.data.get("norm_index", WFLW_INTEROCULAR))

    runs = []
    for seed in seeds:
        seed_everything(seed)
        encoder = load_encoder_from_checkpoint(cfg, args.checkpoint, device)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=cfg.data.num_workers)
        test_loader = DataLoader(test_ds, batch_size=bs, num_workers=cfg.data.num_workers)
        model = LandmarkModel(encoder, num_landmarks=NUM_LANDMARKS)
        train_landmark(model, train_loader, epochs=args.epochs, device=device)
        metrics = evaluate_landmark(model, test_loader, norm_index=norm_index,
                                    flip_index=WFLW_FLIP_INDEX, device=device)
        runs.append(metrics)
        print(f"seed={seed}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if len(runs) > 1:
        print(f"mean±std over {len(runs)} seeds {seeds}: "
              + format_aggregated(aggregate_metrics(runs)))

if __name__ == "__main__":
    main()
