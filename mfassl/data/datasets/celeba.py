"""CelebA-HQ loader -- 30k face images with 40 binary attributes."""

import glob
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_HQ_PATTERNS = ["{idx}.jpg", "{idx}.png", "{idx:05d}.jpg", "{idx:05d}.png", "imgHQ{idx:05d}.png"]

_IMG_EXTS = (".jpg", ".jpeg", ".png")

def _read_attr(path: str) -> Tuple[Dict[str, List[float]], List[str]]:

    with open(path) as f:
        lines = f.read().splitlines()

    attr_names = lines[1].split()
    attr_by_file: Dict[str, List[float]] = {}
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        attr_by_file[parts[0]] = [(1.0 if int(v) == 1 else 0.0) for v in parts[1:]]
    return attr_by_file, attr_names

def _read_mapping(path: str) -> List[Tuple[int, str]]:

    rows: List[Tuple[int, str]] = []
    with open(path) as f:
        for line in f.read().splitlines():
            parts = line.split()
            if not parts or not parts[0].lstrip("-").isdigit():
                continue
            hq_idx = int(parts[0])
            orig = next((p for p in parts if p.lower().endswith(_IMG_EXTS)), None)
            if orig is None:
                raise ValueError(f"no image filename found in mapping row: {line!r}")
            rows.append((hq_idx, orig))
    if not rows:
        raise ValueError(f"no data rows parsed from mapping file {path}")
    return rows

def _read_identities(path: str) -> Dict[str, str]:

    ids: Dict[str, str] = {}
    with open(path) as f:
        for line in f.read().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                ids[parts[0]] = parts[1]
    return ids

class CelebAHQDataset(Dataset):

    def __init__(self, root: str, img_dir: str = "CelebA-HQ-img",
                 attr_file: str = "list_attr_celeba.txt",
                 mapping_file: str = "CelebA-HQ-to-CelebA-mapping.txt",
                 identity_file: str = "identity_CelebA.txt",
                 img_size: int = 224, limit: Optional[int] = None):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.img_size = img_size
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"CelebA-HQ image directory not found: {self.img_dir}")

        attr_by_file, self.attr_names = _read_attr(os.path.join(root, attr_file))
        ids_by_file = (_read_identities(os.path.join(root, identity_file))
                       if os.path.exists(os.path.join(root, identity_file)) else {})

        mapping_path = os.path.join(root, mapping_file)
        if os.path.exists(mapping_path):
            records, identities = self._build_from_mapping(mapping_path, attr_by_file,
                                                           ids_by_file)
        else:
            records, identities = self._build_from_folder(attr_by_file, ids_by_file)

        if limit is not None:
            records, identities = records[:limit], identities[:limit]
        self.records = records

        self.identities = identities if any(i is not None for i in identities) else None

    def _build_from_mapping(self, mapping_path, attr_by_file, ids_by_file):
        mapping = _read_mapping(mapping_path)
        pattern = self._detect_pattern(mapping)
        records, identities = [], []
        for hq_idx, orig in mapping:
            if orig not in attr_by_file:
                raise KeyError(f"mapping references '{orig}', absent from the attribute file")
            records.append((pattern.format(idx=hq_idx), attr_by_file[orig]))
            identities.append(ids_by_file.get(orig))
        return records, identities

    def _build_from_folder(self, attr_by_file, ids_by_file):
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(self.img_dir, "*"))
                       if p.lower().endswith(_IMG_EXTS))
        if not files:
            raise FileNotFoundError(f"no images found in {self.img_dir}")
        missing = [f for f in files if f not in attr_by_file]
        if missing:
            raise KeyError(
                f"{len(missing)} image(s) in {self.img_dir} have no attribute row (e.g. "
                f"{missing[0]}). The official CelebA-HQ format needs a mapping file "
                f"(CelebA-HQ-to-CelebA-mapping.txt / image_list.txt); see the loader docstring.")
        records = [(f, attr_by_file[f]) for f in files]
        identities = [ids_by_file.get(f) for f in files]
        return records, identities

    def _detect_pattern(self, mapping) -> str:

        first_idx = mapping[0][0]
        for pat in _HQ_PATTERNS:
            if os.path.exists(os.path.join(self.img_dir, pat.format(idx=first_idx))):
                return pat
        raise FileNotFoundError(
            f"could not locate HQ image for index {first_idx} in {self.img_dir} using any known "
            f"naming pattern {_HQ_PATTERNS}; pass a matching img_dir or rename the images.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image
        fname, vals = self.records[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - 0.5) / 0.5
        return t, torch.tensor(vals, dtype=torch.float32)

def celeba_hq_split(ds: CelebAHQDataset, val_frac: float = 0.1,
                    seed: int = 0) -> Tuple[List[int], List[int]]:

    n = len(ds)
    if ds.identities is not None:
        from ..splits import group_wise_split
        return group_wise_split(ds.identities, val_frac=val_frac, seed=seed)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(val_frac * n))) if n > 1 else 0
    test_idx, train_idx = idx[:n_val], idx[n_val:]
    return train_idx, test_idx
