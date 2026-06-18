"""Downstream multi-label classification: linear probe and full fine-tuning."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..models.vit import MFASSLViT
from .metrics import multilabel_auroc, multilabel_auprc, multilabel_f1, multilabel_accuracy
from .calibration import expected_calibration_error, bce_nll, brier_score
from .flip_consistency import multilabel_flip_consistency

class ClassifierModel(nn.Module):

    def __init__(self, encoder: MFASSLViT, num_classes: int, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        return self.head(feats)

    def trainable_parameters(self):
        if self.freeze_encoder:
            return self.head.parameters()
        return self.parameters()

def train_classifier(model: ClassifierModel, loader, epochs: int = 1, lr: float = 1e-3,
                     device: str = "cpu", max_steps: Optional[int] = None) -> None:

    model.to(device).train()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    step = 0
    for _ in range(epochs):
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(img), label)
            loss.backward()
            opt.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                return

@torch.no_grad()
def evaluate_classifier(model: ClassifierModel, loader, device: str = "cpu",
                        flip_consistency: bool = True,
                        flip_reduction: str = "per_image",
                        save_predictions: Optional[str] = None) -> dict:

    model.to(device).eval()
    probs, labels, probs_flip = [], [], []
    for img, label in loader:
        img = img.to(device)
        p = torch.sigmoid(model(img)).cpu().numpy()
        probs.append(p)
        labels.append(label.numpy())
        if flip_consistency:
            pf = torch.sigmoid(model(torch.flip(img, dims=[-1]))).cpu().numpy()
            probs_flip.append(pf)

    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0)
    if save_predictions is not None:
        np.savez(save_predictions, y_true=y_true, y_score=y_prob)
    metrics = {
        "AUROC": multilabel_auroc(y_true, y_prob),
        "AUPRC": multilabel_auprc(y_true, y_prob),
        "F1": multilabel_f1(y_true, y_prob),
        "Acc": multilabel_accuracy(y_true, y_prob),
        "NLL": bce_nll(y_true, y_prob),
        "ECE": expected_calibration_error(y_true, y_prob),
        "Brier": brier_score(y_true, y_prob),
    }
    if flip_consistency:
        y_prob_flip = np.concatenate(probs_flip, axis=0)
        metrics["FlipConsistency"] = multilabel_flip_consistency(
            y_prob, y_prob_flip, reduction=flip_reduction)
    return metrics
