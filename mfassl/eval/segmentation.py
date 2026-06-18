"""2D segmentation downstream (BraTS, slice-wise) -- ViT encoder + lightweight decoder."""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.vit import MFASSLViT
from .seg_metrics import dice_score, hd95
from .calibration import expected_calibration_error, bce_nll

class SegmentationModel(nn.Module):

    def __init__(self, encoder: MFASSLViT, num_classes: int, img_size: int = 224):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = encoder.backbone.patch_embed.patch_size[0]
        self.grid = img_size // self.patch_size
        d = encoder.embed_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(d, d // 2, 3, padding=1), nn.GroupNorm(8, d // 2), nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(d // 2, d // 4, 3, padding=1), nn.GroupNorm(8, d // 4), nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(d // 4, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.forward_features(x)
        npre = self.encoder.num_prefix_tokens
        patches = feats[:, npre:, :]
        b, n, d = patches.shape
        g = int(round(math.sqrt(n)))
        grid = patches.transpose(1, 2).reshape(b, d, g, g)
        logits = self.decoder(grid)
        if logits.shape[-1] != x.shape[-1]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                                   align_corners=False)
        return logits

def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):

    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    inter = (probs * target).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = 1.0 - (2 * inter + eps) / (denom + eps)
    return bce + dice.mean()

def train_segmentation(model, loader, epochs=1, lr=1e-4, device="cpu", max_steps=None):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    step = 0
    for _ in range(epochs):
        for img, target in loader:
            img, target = img.to(device), target.to(device)
            opt.zero_grad(set_to_none=True)
            dice_bce_loss(model(img), target).backward()
            opt.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                return

@torch.no_grad()
def evaluate_segmentation(model, loader, class_names: List[str], device="cpu",
                         threshold: float = 0.5, spacing=None, hd95_empty="skip",
                         save_predictions: Optional[str] = None) -> dict:

    model.to(device).eval()
    dices = {c: [] for c in class_names}
    hds = {c: [] for c in class_names}
    all_probs, all_targets = [], []
    for img, target in loader:
        img = img.to(device)
        probs = torch.sigmoid(model(img)).cpu().numpy()
        tgt = target.numpy()
        all_probs.append(probs.ravel())
        all_targets.append(tgt.ravel())
        pred = (probs >= threshold)
        for ci, c in enumerate(class_names):
            for b in range(pred.shape[0]):
                p_bc, g_bc = pred[b, ci], tgt[b, ci] > 0.5
                dices[c].append(dice_score(p_bc, g_bc))
                d = hd95(p_bc, g_bc, spacing)
                if not np.isnan(d):
                    hds[c].append(d)
                elif hd95_empty != "skip":

                    both_empty = (p_bc.sum() == 0 and g_bc.sum() == 0)
                    hds[c].append(0.0 if both_empty else float(hd95_empty))

    probs_flat = np.concatenate(all_probs)
    targets_flat = np.concatenate(all_targets)
    if save_predictions is not None:
        np.savez(save_predictions, y_true=targets_flat, y_score=probs_flat)
    def _mean(vals):
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    out = {}
    for c in class_names:
        out[f"Dice_{c}"] = _mean(dices[c])
        out[f"HD95_{c}"] = _mean(hds[c])
    out["Dice"] = _mean([out[f"Dice_{c}"] for c in class_names])
    out["HD95"] = _mean([out[f"HD95_{c}"] for c in class_names])
    out["ECE"] = expected_calibration_error(targets_flat, probs_flat)
    out["NLL"] = bce_nll(targets_flat, probs_flat)
    return out
