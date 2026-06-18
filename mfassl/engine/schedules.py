"""Staged training schedules."""

import math
from dataclasses import dataclass

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def sym_loss_weight(t: float, t_sym: float = 10.0) -> float:

    if t_sym <= 0:
        return 1.0
    return _clip01(t / t_sym)

def mfa_active(t: float, t_mfa: float = 12.0) -> bool:

    return t >= t_mfa

def gate_ramp(t: float, t_mfa: float = 12.0, t_gate: float = 10.0) -> float:

    if t < t_mfa:
        return 0.0
    if t_gate <= 0:
        return 1.0
    return _clip01((t - t_mfa) / t_gate)

def total_loss(l_base, l_eq, l_mid, w: float, lambda_eq: float = 0.5,
               lambda_mid: float = 1.0):

    return l_base + w * (lambda_eq * l_eq + lambda_mid * l_mid)

def cosine_lr(step: int, total_steps: int, base_lr: float,
              warmup_steps: int = 0, min_lr: float = 0.0) -> float:

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def cosine_momentum(t: float, total_epochs: float, base: float = 0.996,
                    final: float = 1.0) -> float:

    if total_epochs <= 0:
        return base
    p = min(1.0, max(0.0, t / total_epochs))
    return final - (final - base) * 0.5 * (1.0 + math.cos(math.pi * p))

def teacher_temp(t: float, warmup_epochs: float, start: float = 0.04,
                 final: float = 0.04) -> float:

    if warmup_epochs <= 0:
        return final
    return start + (final - start) * min(1.0, max(0.0, t / warmup_epochs))

@dataclass
class StagedSchedule:

    t_sym: float = 10.0
    t_mfa: float = 12.0
    t_gate: float = 10.0
    lambda_eq: float = 0.5
    lambda_mid: float = 1.0

    def state(self, t: float) -> dict:

        return {
            "w": sym_loss_weight(t, self.t_sym),
            "r_t": gate_ramp(t, self.t_mfa, self.t_gate),
            "mfa_active": mfa_active(t, self.t_mfa),
        }

    def total_loss(self, l_base, l_eq, l_mid, t: float):
        w = sym_loss_weight(t, self.t_sym)
        return total_loss(l_base, l_eq, l_mid, w, self.lambda_eq, self.lambda_mid)
