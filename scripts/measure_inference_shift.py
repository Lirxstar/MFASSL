#!/usr/bin/env python3
"""Measure the MFA inference-time feature shift ||Z_L - X_L|| / ||X_L||."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mfassl.utils.config import load_config
from mfassl.engine.build import (
    resolve_device, build_method, build_mfa, build_base_dataset, build_pretrain_loader,
)
from mfassl.eval.inference_shift import measure_inference_shift

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max_batches", type=int, default=50)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = resolve_device(cfg.device)
    method = build_method(cfg).to(device)
    mfa = build_mfa(cfg, method).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    method.load_state_dict(ckpt["method"])
    mfa.load_state_dict(ckpt["mfa"])

    loader = build_pretrain_loader(cfg, build_base_dataset(cfg))
    res = measure_inference_shift(method.encoder, mfa, loader, layer=cfg.mfa.layer,
                                  device=device, max_batches=args.max_batches)
    print(f"inference shift: mean={100 * res['mean']:.2f}%  "
          f"std={100 * res['std']:.2f}%  (n={res['n']})")

if __name__ == "__main__":
    main()
