#!/usr/bin/env python3
"""Downstream BraTS 2D segmentation eval."""

import argparse
import os
import sys

from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mfassl.utils.config import load_config
from mfassl.utils.seed import seed_everything
from mfassl.engine.build import resolve_device, load_encoder_from_checkpoint
from mfassl.data.datasets.brats import BraTS2DDataset, BRATS_SUBREGIONS, patient_wise_split
from mfassl.eval.segmentation import SegmentationModel, train_segmentation, evaluate_segmentation
from mfassl.eval.aggregate import aggregate_metrics, format_aggregated, seed_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="run once per seed and report mean±std; "
                        "default: the config's seed")
    p.add_argument("--save-predictions", default=None,
                   help=".npz of voxel-level y_true/y_score for run_bootstrap.py; a multi-seed "
                        "sweep adds a _seed{n} suffix per run")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = resolve_device(cfg.device)
    seeds = args.seeds if args.seeds is not None else [cfg.seed]

    ds = BraTS2DDataset(cfg.data.brats_root, img_size=cfg.backbone.img_size, labeled=True,
                        tumor_slices_only=cfg.data.get("tumor_slices_only", True),
                        limit=cfg.data.get("limit"))

    train_idx, val_idx = patient_wise_split(ds.index, val_frac=cfg.data.get("val_frac", 0.2),
                                            seed=cfg.seed)
    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)
    bs = cfg.data.batch_size
    spacing = ds.inplane_spacing_mm()

    runs = []
    for seed in seeds:
        seed_everything(seed)
        encoder = load_encoder_from_checkpoint(cfg, args.checkpoint, device)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=cfg.data.num_workers)
        val_loader = DataLoader(val_ds, batch_size=bs, num_workers=cfg.data.num_workers)
        model = SegmentationModel(encoder, num_classes=3, img_size=cfg.backbone.img_size)
        train_segmentation(model, train_loader, epochs=args.epochs, device=device)
        metrics = evaluate_segmentation(model, val_loader, class_names=BRATS_SUBREGIONS,
                                        device=device, spacing=spacing,
                                        save_predictions=seed_path(args.save_predictions, seed,
                                                                   len(seeds)))
        runs.append(metrics)
        print(f"seed={seed}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if len(runs) > 1:
        print(f"mean±std over {len(runs)} seeds {seeds}: "
              + format_aggregated(aggregate_metrics(runs)))

if __name__ == "__main__":
    main()
