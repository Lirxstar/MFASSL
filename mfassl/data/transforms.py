"""Multi-crop + mirror-pair transforms for MFASSL pretraining."""

from typing import List, Optional

import torch
import torch.nn.functional as F

from .mirror import make_mirror_pair

def random_resized_crop(img: torch.Tensor, size: int, scale=(0.3, 1.0),
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:

    c, h, w = img.shape
    area = h * w
    target = (scale[0] + (scale[1] - scale[0]) *
              torch.rand((), generator=generator).item()) * area
    ch = max(1, min(h, int(round(target ** 0.5))))
    cw = max(1, min(w, int(round(target ** 0.5))))
    top = int(torch.randint(0, h - ch + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, w - cw + 1, (1,), generator=generator).item())
    crop = img[:, top:top + ch, left:left + cw].unsqueeze(0).float()
    out = F.interpolate(crop, size=(size, size), mode="bilinear",
                        align_corners=False, antialias=True)
    return out.squeeze(0)

class MFASSLMultiCrop:

    def __init__(self, n_global: int = 2, n_local: int = 0, global_size: int = 224,
                 local_size: int = 96, mirror_size: int = 224, axis_jitter: float = 0.03,
                 augment: bool = False, pixel_range: str = "unit_centered"):
        self.n_global = n_global
        self.n_local = n_local
        self.global_size = global_size
        self.local_size = local_size
        self.mirror_size = mirror_size
        self.axis_jitter = axis_jitter
        self.photo = None
        if augment:
            from .augment import PhotometricAugment
            self.photo = PhotometricAugment(pixel_range=pixel_range)

    def _aug(self, crop: torch.Tensor) -> torch.Tensor:
        return self.photo(crop) if self.photo is not None else crop

    def __call__(self, img: torch.Tensor, generator: Optional[torch.Generator] = None) -> dict:
        crops: List[torch.Tensor] = []
        for _ in range(self.n_global):
            crops.append(self._aug(random_resized_crop(img, self.global_size, (0.4, 1.0),
                                                        generator)))
        for _ in range(self.n_local):
            crops.append(self._aug(random_resized_crop(img, self.local_size, (0.05, 0.4),
                                                        generator)))
        x_l, x_r, _ = make_mirror_pair(img, self.axis_jitter, self.mirror_size,
                                       generator=generator)
        return {"standard_crops": crops, "mirror": (self._aug(x_l), self._aug(x_r)),
                "n_global": self.n_global}

def collate_mfassl(batch: List[dict]) -> dict:

    n_crops = len(batch[0]["standard_crops"])
    standard = [torch.stack([item["standard_crops"][k] for item in batch], dim=0)
                for k in range(n_crops)]
    x_l = torch.stack([item["mirror"][0] for item in batch], dim=0)
    x_r = torch.stack([item["mirror"][1] for item in batch], dim=0)
    return {"standard_crops": standard, "mirror": (x_l, x_r),
            "n_global": batch[0]["n_global"]}
