"""MFASSL pretraining step -- the unified loop of Algorithm 1."""

from typing import Optional

import torch

from .schedules import StagedSchedule
from ..losses.symmetry import reflection_consistency_loss, mid_layer_consistency_loss

class MFASSLTrainer:

    def __init__(self, method, mfa, schedule: Optional[StagedSchedule] = None,
                 layer: int = 8, teacher_momentum: float = 0.996, beta: float = 1.0,
                 freeze_last_layer_epochs: float = 0.0):
        self.method = method
        self.mfa = mfa
        self.schedule = schedule or StagedSchedule()
        self.layer = layer
        self.teacher_momentum = teacher_momentum
        self.beta = beta
        self.freeze_last_layer_epochs = freeze_last_layer_epochs

    def compute_losses(self, batch: dict, t: float) -> dict:

        if hasattr(self.method, "mfassl_losses"):
            return self.method.mfassl_losses(batch, t, self.mfa, self.schedule, self.layer,
                                             self.beta)

        state = self.schedule.state(t)

        l_base_std = self.method.base_loss(batch["standard_crops"])

        x_l, x_r = batch["mirror"]
        valid_mask = batch.get("mirror_valid_mask")
        out = self.method.encoder.forward_paired(
            x_l, x_r, mfa=self.mfa, layer=self.layer,
            r_t=state["r_t"], mfa_active=state["mfa_active"], valid_mask=valid_mask,
        )

        l_mid = mid_layer_consistency_loss(out["prefusion_l"], out["prefusion_r"], valid_mask)
        l_eq = reflection_consistency_loss(out["prefusion_l"], out["prefusion_r"], valid_mask)

        l_base_mir = self.method.mirror_base_loss(out["tokens_l"], out["tokens_r"], x_l, x_r)
        l_base = l_base_std + self.beta * l_base_mir

        total = self.schedule.total_loss(l_base, l_eq, l_mid, t)
        return {
            "loss": total,
            "l_base": l_base.detach(),
            "l_base_std": l_base_std.detach(),
            "l_base_mir": l_base_mir.detach(),
            "l_eq": l_eq.detach(),
            "l_mid": l_mid.detach(),
            "w": state["w"],
            "r_t": state["r_t"],
            "mfa_active": state["mfa_active"],
            "gate_mean": (out["gate"].mean().detach() if out["gate"] is not None
                          else torch.tensor(0.0)),
        }

    def step(self, batch: dict, t: float, optimizer,
             teacher_momentum: Optional[float] = None) -> dict:

        optimizer.zero_grad(set_to_none=True)
        logs = self.compute_losses(batch, t)
        logs["loss"].backward()

        if t < self.freeze_last_layer_epochs and hasattr(self.method,
                                                          "cancel_last_layer_gradients"):
            self.method.cancel_last_layer_gradients()
        optimizer.step()
        if hasattr(self.method, "update_teacher"):
            m = self.teacher_momentum if teacher_momentum is None else teacher_momentum
            self.method.update_teacher(m)
        logs["loss"] = logs["loss"].detach()
        return logs
