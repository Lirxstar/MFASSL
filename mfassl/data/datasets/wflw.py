"""WFLW loader -- 98 facial landmarks per image."""

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_LANDMARKS = 98

WFLW_FLIP_INDEX = [
    32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41,
    40, 39, 38, 51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62,
    61, 60, 67, 66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95,
    94, 93, 97, 96,
]

WFLW_INTEROCULAR = (60, 72)

class WFLWDataset(Dataset):

    def __init__(self, root: str, ann_file: str, img_size: int = 224,
                 limit: Optional[int] = None):
        self.root = root
        self.img_size = img_size
        self.records = self._read_ann(os.path.join(root, ann_file), limit)

    @staticmethod
    def _read_ann(path, limit):
        records = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                coords = np.array(parts[: NUM_LANDMARKS * 2], dtype=np.float32).reshape(-1, 2)
                rect = np.array(parts[NUM_LANDMARKS * 2: NUM_LANDMARKS * 2 + 4],
                                dtype=np.float32)
                img_path = parts[-1]
                records.append((img_path, coords, rect))
                if limit is not None and len(records) >= limit:
                    break
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image
        img_path, coords, rect = self.records[idx]
        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
        x1, y1, x2, y2 = rect

        img = img.crop((x1, y1, x2, y2)).resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = (torch.from_numpy(arr).permute(2, 0, 1) - 0.5) / 0.5
        box_w = max(x2 - x1, 1e-6)
        box_h = max(y2 - y1, 1e-6)
        norm = coords.copy()
        norm[:, 0] = (coords[:, 0] - x1) / box_w
        norm[:, 1] = (coords[:, 1] - y1) / box_h
        return t, torch.from_numpy(norm)
