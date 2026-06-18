"""CheXpert loader."""

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

def _load_image(path: str, size: int) -> torch.Tensor:
    from PIL import Image
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None].repeat(3, 1, 1)

    return (t - 0.5) / 0.5

class CheXpertDataset(Dataset):

    def __init__(self, root: str, csv_file: str = "train.csv", img_size: int = 224,
                 u_policy: str = "zeros", frontal_only: bool = True,
                 limit: Optional[int] = None):
        import csv
        self.root = root
        self.img_size = img_size
        self.u_policy = u_policy
        self.frontal_only = frontal_only
        self.records = []
        csv_path = os.path.join(root, csv_file)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:

                if frontal_only and row.get("Frontal/Lateral", "Frontal") != "Frontal":
                    continue
                self.records.append(row)
                if limit is not None and len(self.records) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.records)

    def _label_vector(self, row) -> torch.Tensor:
        vals = []
        for name in CHEXPERT_LABELS:
            raw = row.get(name, "")
            if raw in ("", None):
                v = 0.0
            else:
                v = float(raw)
                if v == -1.0:
                    v = {"zeros": 0.0, "ones": 1.0, "ignore": -1.0}[self.u_policy]
            vals.append(v)
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        img = _load_image(os.path.join(self.root, row["Path"]), self.img_size)
        return img, self._label_vector(row)

    def patient_split(self, val_frac: float = 0.1, seed: int = 0):

        from ..splits import group_wise_split
        pids = [patient_id_from_path(r["Path"]) for r in self.records]
        return group_wise_split(pids, val_frac=val_frac, seed=seed)

def patient_id_from_path(path: str) -> str:

    for part in path.split("/"):
        if part.startswith("patient"):
            return part
    return path
