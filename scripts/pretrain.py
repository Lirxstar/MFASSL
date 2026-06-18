#!/usr/bin/env python3
"""MFASSL pretraining entry point."""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mfassl.utils.config import load_config
from mfassl.utils.seed import seed_everything
from mfassl.engine.build import (
    resolve_device, build_method, build_mfa, build_base_dataset,
    build_pretrain_loader, build_trainer, set_lr,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. seed=1")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    method = build_method(cfg).to(device)
    mfa = build_mfa(cfg, method).to(device)
    loader = build_pretrain_loader(cfg, build_base_dataset(cfg))
    trainer, optimizer = build_trainer(cfg, method, mfa)

    from mfassl.engine.schedules import cosine_momentum, teacher_temp as teacher_temp_sched

    epochs = cfg.pretrain.epochs
    max_steps = cfg.pretrain.get("max_steps")
    steps_per_epoch = max(1, len(loader))
    total_steps = epochs * steps_per_epoch
    base_lr = cfg.pretrain.base_lr
    warmup_steps = int(cfg.pretrain.get("warmup_epochs", 0) * steps_per_epoch)
    mom_base = cfg.pretrain.teacher_momentum
    mom_final = cfg.pretrain.get("teacher_momentum_final", 1.0)

    os.makedirs(cfg.log.out_dir, exist_ok=True)
    global_step = 0
    for epoch in range(epochs):
        for it, batch in enumerate(loader):
            batch = _to_device(batch, device)
            t = epoch + it / steps_per_epoch
            set_lr(optimizer, cosine := _lr(global_step, total_steps, base_lr, warmup_steps))

            if hasattr(method, "dino_loss"):
                method.dino_loss.teacher_temp = teacher_temp_sched(
                    t, cfg.pretrain.get("teacher_temp_warmup_epochs", 0),
                    start=cfg.dino.get("teacher_temp_start", cfg.dino.teacher_temp),
                    final=cfg.dino.teacher_temp)
            momentum = cosine_momentum(t, epochs, base=mom_base, final=mom_final)
            logs = trainer.step(batch, t=t, optimizer=optimizer, teacher_momentum=momentum)
            if global_step % cfg.log.log_every == 0:
                print(f"epoch {epoch} step {global_step} lr {cosine:.2e} "
                      f"loss {logs['loss'].item():.4f} base {logs['l_base'].item():.4f} "
                      f"eq {logs['l_eq'].item():.4f} mid {logs['l_mid'].item():.4f} "
                      f"w {logs['w']:.2f} r_t {logs['r_t']:.2f} mfa {logs['mfa_active']}")
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break
        if max_steps is not None and global_step >= max_steps:
            break

    ckpt_path = os.path.join(cfg.log.out_dir, "checkpoint.pth")
    torch.save({"method": method.state_dict(), "mfa": mfa.state_dict(),
                "config": _container(cfg)}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")

def _lr(step, total, base, warmup):
    from mfassl.engine.schedules import cosine_lr
    return cosine_lr(step, total, base, warmup, min_lr=base * 1e-3)

def _to_device(batch, device):
    batch["standard_crops"] = [c.to(device) for c in batch["standard_crops"]]
    x_l, x_r = batch["mirror"]
    batch["mirror"] = (x_l.to(device), x_r.to(device))
    return batch

def _container(cfg):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=True)

if __name__ == "__main__":
    main()
