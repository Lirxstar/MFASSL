"""Symmetry-aware objective."""

from typing import Optional

import torch
import torch.nn.functional as F

def _masked_mean_tokens(x: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:

    if valid_mask is None:
        return x.mean(dim=1)
    m = valid_mask.unsqueeze(-1).to(x.dtype)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom

def reflection_consistency_loss(
    x_l: torch.Tensor, x_r: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:

    s_l = F.normalize(_masked_mean_tokens(x_l, valid_mask), dim=-1)
    s_r = F.normalize(_masked_mean_tokens(x_r, valid_mask), dim=-1)
    cos = (s_l * s_r).sum(dim=-1)
    return (1.0 - cos).mean()

def mid_layer_consistency_loss(
    x_l: torch.Tensor, x_r: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:

    ph_l = F.normalize(x_l, dim=-1)
    ph_r = F.normalize(x_r, dim=-1)
    sq = ((ph_l - ph_r) ** 2).sum(dim=-1)
    if valid_mask is None:
        return sq.mean()
    m = valid_mask.to(sq.dtype)
    denom = m.sum(dim=1).clamp_min(1.0)
    per_sample = (sq * m).sum(dim=1) / denom
    return per_sample.mean()

class SymmetryLoss(torch.nn.Module):

    def __init__(self, lambda_eq: float = 0.5, lambda_mid: float = 1.0):
        super().__init__()
        self.lambda_eq = lambda_eq
        self.lambda_mid = lambda_mid

    def forward(self, x_l, x_r, valid_mask: Optional[torch.Tensor] = None):
        l_eq = reflection_consistency_loss(x_l, x_r, valid_mask)
        l_mid = mid_layer_consistency_loss(x_l, x_r, valid_mask)
        weighted = self.lambda_eq * l_eq + self.lambda_mid * l_mid
        return weighted, {"l_eq": l_eq.detach(), "l_mid": l_mid.detach()}
