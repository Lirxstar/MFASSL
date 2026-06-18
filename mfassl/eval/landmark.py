"""Facial-landmark localization downstream (WFLW) -- ViT encoder + regression head."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..models.vit import MFASSLViT
from .landmark_metrics import nme, auc_at, failure_rate, landmark_flip_consistency

class LandmarkModel(nn.Module):

    def __init__(self, encoder: MFASSLViT, num_landmarks: int = 98, hidden: int = 256):
        super().__init__()
        self.encoder = encoder
        self.num_landmarks = num_landmarks
        self.head = nn.Sequential(
            nn.Linear(encoder.embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, num_landmarks * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        coords = self.head(feats).view(-1, self.num_landmarks, 2)
        return torch.sigmoid(coords)

def train_landmark(model, loader, epochs=1, lr=1e-4, device="cpu", max_steps=None):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    step = 0
    for _ in range(epochs):
        for img, target in loader:
            img, target = img.to(device), target.to(device)
            opt.zero_grad(set_to_none=True)
            criterion(model(img), target).backward()
            opt.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                return

@torch.no_grad()
def evaluate_landmark(model, loader, norm_index=None, flip_index=None, device="cpu") -> dict:

    model.to(device).eval()
    preds, gts, preds_flip = [], [], []
    for img, target in loader:
        img = img.to(device)
        preds.append(model(img).cpu().numpy())
        gts.append(target.numpy())
        preds_flip.append(model(torch.flip(img, dims=[-1])).cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    gt = np.concatenate(gts, axis=0)
    pred_flip = np.concatenate(preds_flip, axis=0)

    if norm_index is not None:
        norm = np.linalg.norm(gt[:, norm_index[0]] - gt[:, norm_index[1]], axis=-1)
        norm = np.clip(norm, 1e-6, None)
    else:
        mins = gt.min(axis=1)
        maxs = gt.max(axis=1)
        norm = np.linalg.norm(maxs - mins, axis=-1)
        norm = np.clip(norm, 1e-6, None)

    nme_vals = nme(pred, gt, norm)
    return {
        "NME": float(nme_vals.mean()),
        "AUC@0.1": auc_at(nme_vals, 0.1),
        "Fail@0.1": failure_rate(nme_vals, 0.1),
        "FlipConsistency": landmark_flip_consistency(
            pred, pred_flip, image_width=1.0, flip_index=flip_index, norm_dist=norm),
    }
