"""OASIS-3 loader -- unlabeled T1-weighted MRI for joint MRI pretraining."""

import glob
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .brats import _normalize

class OASIS3Dataset(Dataset):

    def __init__(self, root: str, img_size: int = 224, pattern: str = "**/*T1w*.nii.gz",
                 in_chans: int = 1, limit: Optional[int] = None):
        self.root = root
        self.img_size = img_size
        self.in_chans = in_chans
        self.files = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
        self.index = self._build_index(limit)

    def _build_index(self, limit):
        import nibabel as nib
        index: List = []
        for f in self.files:
            n = nib.load(f).shape[2]
            for z in range(n // 4, 3 * n // 4):
                index.append((f, z))
                if limit is not None and len(index) >= limit:
                    return index
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        import nibabel as nib
        from PIL import Image
        f, z = self.index[idx]
        sl = _normalize(nib.load(f).get_fdata()[:, :, z])
        img = Image.fromarray(sl).resize((self.img_size, self.img_size), Image.BILINEAR)
        image = torch.from_numpy(np.asarray(img, dtype=np.float32))[None]
        if self.in_chans > 1:
            image = image.repeat(self.in_chans, 1, 1)
        return image, 0

class ConcatBalanced(Dataset):

    def __init__(self, ds_a: Dataset, ds_b: Dataset):
        self.ds_a, self.ds_b = ds_a, ds_b
        self.n = 2 * min(len(ds_a), len(ds_b))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if idx % 2 == 0:
            return self.ds_a[(idx // 2) % len(self.ds_a)]
        return self.ds_b[(idx // 2) % len(self.ds_b)]
