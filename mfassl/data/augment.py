"""Standard SSL photometric augmentations for MFASSL pretraining."""

import torch

class PhotometricAugment:

    def __init__(self, pixel_range: str = "unit_centered", blur_kernel: int = 9):
        from torchvision.transforms import v2
        self.pixel_range = pixel_range
        if pixel_range == "zscore":

            self.tf = v2.RandomApply([v2.GaussianBlur(blur_kernel, sigma=(0.1, 2.0))], p=0.5)
        elif pixel_range == "unit_centered":
            self.tf = v2.Compose([
                v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(blur_kernel, sigma=(0.1, 2.0))], p=0.5),
                v2.RandomSolarize(threshold=0.5, p=0.2),
            ])
        else:
            raise ValueError(f"unknown pixel_range '{pixel_range}'")

    def __call__(self, crop: torch.Tensor) -> torch.Tensor:
        if self.pixel_range == "unit_centered":
            x = (crop * 0.5 + 0.5).clamp(0.0, 1.0)
            x = self.tf(x)
            return x * 2.0 - 1.0
        return self.tf(crop)
