#!/usr/bin/env python3
"""Downstream multi-label classification eval: linear probe + fine-tuning."""

import argparse
import os
import sys

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mfassl.utils.config import load_config
from mfassl.utils.seed import seed_everything
from mfassl.engine.build import resolve_device, load_encoder_from_checkpoint
from mfassl.eval.linear_probe import run_linear_probe
from mfassl.eval.finetune_cls import run_finetune
from mfassl.eval.aggregate import aggregate_metrics, format_aggregated, seed_path

def _datasets(cfg, img_size):
    name = cfg.data.name
    if name == "chexpert":
        from mfassl.data.datasets.chexpert import CheXpertDataset
        u_policy = cfg.data.get("u_policy", "zeros")
        frontal_only = cfg.data.get("frontal_only", True)
        train = CheXpertDataset(cfg.data.root, "train.csv", img_size, u_policy=u_policy,
                                frontal_only=frontal_only, limit=cfg.data.get("limit"))
        test = CheXpertDataset(cfg.data.root, cfg.data.get("val_csv", "valid.csv"), img_size,
                               u_policy=u_policy, frontal_only=frontal_only)
        return train, test, 14
    if name == "celeba":
        from mfassl.data.datasets.celeba import CelebAHQDataset, celeba_hq_split
        ds = CelebAHQDataset(cfg.data.root, img_size=img_size, limit=cfg.data.get("limit"))

        train_idx, test_idx = celeba_hq_split(ds, val_frac=cfg.data.get("val_frac", 0.1),
                                              seed=cfg.data.get("split_seed", 0))
        from torch.utils.data import Subset
        return Subset(ds, train_idx), Subset(ds, test_idx), 40
    raise NotImplementedError(f"classification eval for '{name}'")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["linear", "finetune"], default="finetune")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="run once per seed and report mean±std; default: the config's seed")
    p.add_argument("--save-predictions", default=None,
                   help=".npz of y_true/y_score for run_bootstrap.py; a multi-seed sweep adds a "
                        "_seed{n} suffix per run")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = resolve_device(cfg.device)
    seeds = args.seeds if args.seeds is not None else [cfg.seed]

    train_ds, test_ds, num_classes = _datasets(cfg, cfg.backbone.img_size)
    bs = cfg.data.batch_size
    runner = run_linear_probe if args.mode == "linear" else run_finetune
    lr = 1e-3 if args.mode == "linear" else 1e-4

    flip_reduction = "per_image" if cfg.data.name == "chexpert" else "per_label"

    runs = []
    for seed in seeds:
        seed_everything(seed)

        encoder = load_encoder_from_checkpoint(cfg, args.checkpoint, device)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=cfg.data.num_workers)
        test_loader = DataLoader(test_ds, batch_size=bs, num_workers=cfg.data.num_workers)
        metrics = runner(encoder, num_classes, train_loader, test_loader,
                         epochs=args.epochs, lr=lr, device=device,
                         flip_reduction=flip_reduction,
                         save_predictions=seed_path(args.save_predictions, seed, len(seeds)))
        runs.append(metrics)
        print(f"[{args.mode}] seed={seed} "
              + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if len(runs) > 1:
        print(f"[{args.mode}] mean±std over {len(runs)} seeds {seeds}: "
              + format_aggregated(aggregate_metrics(runs)))

if __name__ == "__main__":
    main()
