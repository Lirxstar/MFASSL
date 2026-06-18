"""Mirror-paired view generation."""

from typing import Optional

import torch
import torch.nn.functional as F

def _resize(img: torch.Tensor, out_size: int) -> torch.Tensor:

    if img.shape[-2] == out_size and img.shape[-1] == out_size:
        return img
    x = img.unsqueeze(0).float()
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear",
                      align_corners=False, antialias=True)
    return x.squeeze(0)

def make_mirror_pair(
    img: torch.Tensor,
    axis_jitter: float = 0.03,
    out_size: int = 224,
    jitter_frac: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
):

    if img.dim() != 3:
        raise ValueError(f"expected (C, H, W), got shape {tuple(img.shape)}")
    _, _, w = img.shape

    if jitter_frac is None:
        if axis_jitter > 0:
            r = torch.rand((), generator=generator).item()
            jitter_frac = (2.0 * r - 1.0) * axis_jitter
        else:
            jitter_frac = 0.0

    axis = int(round(w / 2.0 + jitter_frac * w))
    axis = max(1, min(w - 1, axis))
    half = min(axis, w - axis)

    left = img[:, :, axis - half:axis]
    right = img[:, :, axis:axis + half]
    right = torch.flip(right, dims=[-1])

    x_l = _resize(left, out_size)
    x_r = _resize(right, out_size)
    return x_l, x_r, {"axis": axis, "jitter_frac": float(jitter_frac)}

class MirrorPairGenerator:

    def __init__(self, axis_jitter: float = 0.03, out_size: int = 224):
        self.axis_jitter = axis_jitter
        self.out_size = out_size

    def __call__(self, img: torch.Tensor, return_info: bool = False,
                 jitter_frac: Optional[float] = None,
                 generator: Optional[torch.Generator] = None):
        x_l, x_r, info = make_mirror_pair(
            img, axis_jitter=self.axis_jitter, out_size=self.out_size,
            jitter_frac=jitter_frac, generator=generator,
        )
        if return_info:
            return x_l, x_r, info
        return x_l, x_r
