"""Mirror-Fusion Attention (MFA)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def _inv_softplus(y: float) -> float:

    return math.log(math.expm1(y))

class MirrorFusionAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        eps: float = 1e-6,
        a_init: float = 0.5,
        b_init: float = 1.0,
        alpha_init: float = 0.1,
        gamma_init: float = 0.1,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = eps

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)

        self.a = nn.Parameter(torch.tensor(float(a_init)))
        self.b_raw = nn.Parameter(torch.tensor(_inv_softplus(float(b_init))))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    @property
    def b(self) -> torch.Tensor:

        return F.softplus(self.b_raw)

    def compute_gate(self, x_l: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:

        diff = x_l - x_r
        dist = torch.sqrt(diff * diff + self.eps ** 2).sum(dim=-1, keepdim=True)
        return torch.sigmoid(self.a - self.b * dist)

    def _attend(self, q_src, kv_src, valid_mask=None):

        b, n, d = q_src.shape
        q = self.q_proj(q_src).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_src).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_src).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if valid_mask is not None:

            key_mask = valid_mask[:, None, None, :]
            attn = attn.masked_fill(~key_mask, float("-inf"))
        attn = attn.softmax(dim=-1)

        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)

    def forward(self, x_l, x_r, r_t: float = 1.0, valid_mask=None):

        g = self.compute_gate(x_l, x_r)
        g_t = r_t * g

        a_lr = self._attend(x_l, x_r, valid_mask)
        a_rl = self._attend(x_r, x_l, valid_mask)

        z_l = x_l + self.alpha * (g_t * a_lr) + self.gamma * (x_l - x_r)
        z_r = x_r + self.alpha * (g_t * a_rl) + self.gamma * (x_r - x_l)

        if valid_mask is not None:
            keep = valid_mask.unsqueeze(-1)
            z_l = torch.where(keep, z_l, x_l)
            z_r = torch.where(keep, z_r, x_r)

        return z_l, z_r, g

    def added_param_count(self) -> int:

        return sum(p.numel() for p in self.parameters())
