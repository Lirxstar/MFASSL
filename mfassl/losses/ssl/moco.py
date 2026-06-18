"""MoCo-v3 self-supervised backbone for MFASSL."""

import copy
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models.vit import MFASSLViT

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, last_bn: bool = True) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
              nn.Linear(hidden_dim, out_dim)]
    if last_bn:
        layers.append(nn.BatchNorm1d(out_dim, affine=False))
    return nn.Sequential(*layers)

class MoCoV3(nn.Module):

    def __init__(self, vit_kwargs: dict, proj_dim: int = 256, proj_hidden: int = 4096,
                 pred_hidden: int = 4096, temperature: float = 0.2, **_ignored):
        super().__init__()
        self.temperature = temperature
        self.query_encoder = MFASSLViT(**vit_kwargs)
        embed_dim = self.query_encoder.embed_dim
        self.query_proj = _mlp(embed_dim, proj_hidden, proj_dim, last_bn=True)
        self.predictor = _mlp(proj_dim, pred_hidden, proj_dim, last_bn=False)

        self.key_encoder = copy.deepcopy(self.query_encoder)
        self.key_proj = copy.deepcopy(self.query_proj)
        for p in list(self.key_encoder.parameters()) + list(self.key_proj.parameters()):
            p.requires_grad = False

    @property
    def encoder(self) -> MFASSLViT:
        return self.query_encoder

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.query_proj(self.query_encoder(x)))

    @torch.no_grad()
    def _k(self, x: torch.Tensor) -> torch.Tensor:
        return self.key_proj(self.key_encoder(x))

    def _infonce(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = q @ k.t() / self.temperature
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels) * (2 * self.temperature)

    def base_loss(self, crops: Sequence[torch.Tensor], mirror_flags=None) -> torch.Tensor:

        if mirror_flags is not None:
            crops = [c for c, f in zip(crops, mirror_flags) if not f]
        v1, v2 = crops[0], crops[1]
        q1, q2 = self._q(v1), self._q(v2)
        with torch.no_grad():
            k1, k2 = self._k(v1), self._k(v2)
        return self._infonce(q1, k2) + self._infonce(q2, k1)

    def mirror_base_loss(self, tokens_l, tokens_r, x_l, x_r) -> torch.Tensor:

        q_l = self.predictor(self.query_proj(self.query_encoder.pool_tokens(tokens_l)))
        q_r = self.predictor(self.query_proj(self.query_encoder.pool_tokens(tokens_r)))
        with torch.no_grad():
            k_l = self.key_proj(self.key_encoder(x_l))
            k_r = self.key_proj(self.key_encoder(x_r))
        return self._infonce(q_l, k_r) + self._infonce(q_r, k_l)

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.99) -> None:
        for ps, pt in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
        for ps, pt in zip(self.query_proj.parameters(), self.key_proj.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
