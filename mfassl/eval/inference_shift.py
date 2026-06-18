"""Measure the MFA training-time perturbation at the fusion layer."""

from typing import Optional

import torch

@torch.no_grad()
def measure_inference_shift(encoder, mfa, loader, layer: int = 8, device: str = "cpu",
                            r_t: float = 1.0, max_batches: Optional[int] = None) -> dict:

    encoder.to(device).eval()
    mfa.to(device).eval()
    rels = []
    for i, batch in enumerate(loader):
        x_l, x_r = batch["mirror"]
        x_l, x_r = x_l.to(device), x_r.to(device)
        out = encoder.forward_paired(x_l, x_r, mfa=mfa, layer=layer, r_t=r_t,
                                     mfa_active=True, final_norm=False)
        x = out["prefusion_l"]
        z, _, _ = mfa(out["prefusion_l"], out["prefusion_r"], r_t=r_t)
        num = torch.linalg.norm((z - x).reshape(x.size(0), -1), dim=1)
        den = torch.linalg.norm(x.reshape(x.size(0), -1), dim=1).clamp_min(1e-8)
        rels.append((num / den).cpu())
        if max_batches is not None and i + 1 >= max_batches:
            break
    rel = torch.cat(rels)
    return {"mean": float(rel.mean()), "std": float(rel.std()), "n": int(rel.numel())}
