"""DINO self-distillation backbone for MFASSL."""

import copy
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from ...models.vit import MFASSLViT

class DINOHead(nn.Module):

    def __init__(self, in_dim: int, out_dim: int = 4096, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256, nlayers: int = 3):
        super().__init__()
        nlayers = max(1, nlayers)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
        self.last_layer.parametrizations.weight.original0.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)

class DINOLoss(nn.Module):

    def __init__(self, out_dim: int, n_global: int = 2, student_temp: float = 0.1,
                 teacher_temp: float = 0.04, center_momentum: float = 0.9):
        super().__init__()
        self.n_global = n_global
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out: Sequence[torch.Tensor],
                teacher_out: Sequence[torch.Tensor], update_center: bool = True):

        student_lsm = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_out]
        teacher_sm = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_out
        ]

        total = 0.0
        n_terms = 0
        for ti, t in enumerate(teacher_sm):
            for si, s in enumerate(student_lsm):
                if si == ti:
                    continue
                total = total + (-(t * s).sum(dim=-1).mean())
                n_terms += 1
        loss = total / max(1, n_terms)

        if update_center:
            self._update_center(teacher_out)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_out: Sequence[torch.Tensor]) -> None:
        batch_center = torch.cat(teacher_out, dim=0).mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)

class DINO(nn.Module):

    def __init__(self, vit_kwargs: dict, out_dim: int = 4096, n_global: int = 2,
                 head_hidden_dim: int = 2048, head_bottleneck_dim: int = 256,
                 head_nlayers: int = 3, student_temp: float = 0.1,
                 teacher_temp: float = 0.04, center_momentum: float = 0.9):
        super().__init__()
        self.n_global = n_global
        self.student_encoder = MFASSLViT(**vit_kwargs)
        embed_dim = self.student_encoder.embed_dim
        self.student_head = DINOHead(embed_dim, out_dim, head_hidden_dim,
                                     head_bottleneck_dim, head_nlayers)

        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_head = copy.deepcopy(self.student_head)
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.dino_loss = DINOLoss(out_dim, n_global, student_temp, teacher_temp,
                                  center_momentum)

    @property
    def encoder(self) -> MFASSLViT:
        return self.student_encoder

    def _student_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student_head(self.student_encoder(x))

    @torch.no_grad()
    def _teacher_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher_head(self.teacher_encoder(x))

    def base_loss(self, crops: Sequence[torch.Tensor],
                  mirror_flags: Optional[Sequence[bool]] = None,
                  update_center: bool = True) -> torch.Tensor:

        if mirror_flags is not None:
            crops = [c for c, f in zip(crops, mirror_flags) if not f]
        student_out = [self._student_forward(c) for c in crops]
        with torch.no_grad():
            teacher_out = [self._teacher_forward(c) for c in crops[: self.n_global]]
        return self.dino_loss(student_out, teacher_out, update_center=update_center)

    def mirror_base_loss(self, tokens_l, tokens_r, x_l, x_r) -> torch.Tensor:

        s_l = self.student_head(self.student_encoder.pool_tokens(tokens_l))
        s_r = self.student_head(self.student_encoder.pool_tokens(tokens_r))
        with torch.no_grad():
            t_l = self.teacher_head(self.teacher_encoder(x_l))
            t_r = self.teacher_head(self.teacher_encoder(x_r))
        return self.dino_loss([s_l, s_r], [t_l, t_r], update_center=False)

    def cancel_last_layer_gradients(self) -> None:

        for p in self.student_head.last_layer.parameters():
            if p.grad is not None:
                p.grad = None

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
        for ps, pt in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
