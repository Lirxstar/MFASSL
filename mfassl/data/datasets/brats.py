"""BraTS 2023 loader -- 2D slice-wise multi-modal brain MRI."""

import glob
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

def _subregion_masks(label: np.ndarray) -> np.ndarray:
    et = (label == 4)
    tc = np.isin(label, [1, 4])
    wt = np.isin(label, [1, 2, 4])
    return np.stack([et, tc, wt], axis=0).astype(np.float32)

def select_labeled_slices(n_slices: int, seg: Optional[np.ndarray] = None,
                          tumor_slices_only: bool = True) -> List[int]:

    if tumor_slices_only:
        if seg is None:
            raise ValueError("tumor_slices_only=True requires the segmentation volume")
        return [z for z in range(n_slices) if seg[:, :, z].sum() > 0]
    return list(range(n_slices))

def _normalize(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32)
    mask = v > 0
    if mask.sum() > 0:
        mean, std = v[mask].mean(), v[mask].std() + 1e-6
        v = (v - mean) / std
    return v

class BraTS2DDataset(Dataset):

    def __init__(self, root: str, img_size: int = 224, modalities: Optional[List[str]] = None,
                 labeled: bool = True, tumor_slices_only: bool = True,
                 limit: Optional[int] = None):

        self.root = root
        self.img_size = img_size
        self.modalities = modalities or MODALITIES
        self.labeled = labeled
        self.tumor_slices_only = tumor_slices_only
        self.index = self._build_index(limit)

    def _case_dirs(self) -> List[str]:
        return sorted(d for d in glob.glob(os.path.join(self.root, "*")) if os.path.isdir(d))

    def _build_index(self, limit):
        import nibabel as nib
        index = []
        for case in self._case_dirs():
            cid = os.path.basename(case)
            seg_path = os.path.join(case, f"{cid}-seg.nii.gz")
            if self.labeled and not os.path.exists(seg_path):
                continue
            ref = os.path.join(case, f"{cid}-{self.modalities[0]}.nii.gz")
            if not os.path.exists(ref):
                continue
            n_slices = nib.load(ref).shape[2]
            if self.labeled:
                seg = nib.load(seg_path).get_fdata() if self.tumor_slices_only else None
                slices = select_labeled_slices(n_slices, seg, self.tumor_slices_only)
            else:
                slices = list(range(n_slices // 4, 3 * n_slices // 4))
            for z in slices:
                index.append((case, cid, z))
                if limit is not None and len(index) >= limit:
                    return index
        return index

    def __len__(self):
        return len(self.index)

    def inplane_spacing_mm(self) -> Optional[np.ndarray]:

        if not self.index:
            return None
        try:
            import nibabel as nib
            case, cid, _ = self.index[0]
            ref = nib.load(os.path.join(case, f"{cid}-{self.modalities[0]}.nii.gz"))
            zooms = ref.header.get_zooms()[:2]
            native = ref.shape[:2]
            return np.array([zooms[i] * native[i] / self.img_size for i in range(2)],
                            dtype=float)
        except Exception:
            return None

    def _load_slice(self, case, cid, z):
        import nibabel as nib
        from PIL import Image
        chans = []
        for m in self.modalities:
            vol = nib.load(os.path.join(case, f"{cid}-{m}.nii.gz")).get_fdata()
            sl = _normalize(vol[:, :, z])
            img = Image.fromarray(sl).resize((self.img_size, self.img_size), Image.BILINEAR)
            chans.append(np.asarray(img, dtype=np.float32))
        image = torch.from_numpy(np.stack(chans, axis=0))
        return image

    def __getitem__(self, idx):
        case, cid, z = self.index[idx]
        image = self._load_slice(case, cid, z)
        if not self.labeled:
            return image, 0
        import nibabel as nib
        from PIL import Image
        seg = nib.load(os.path.join(case, f"{cid}-seg.nii.gz")).get_fdata()[:, :, z]
        masks = _subregion_masks(seg)
        resized = []
        for ci in range(masks.shape[0]):
            m = Image.fromarray(masks[ci]).resize((self.img_size, self.img_size), Image.NEAREST)
            resized.append(np.asarray(m, dtype=np.float32))
        return image, torch.from_numpy(np.stack(resized, axis=0))

BRATS_SUBREGIONS = ["ET", "TC", "WT"]

def patient_wise_split(index, val_frac: float = 0.2, seed: int = 0):

    from ..splits import group_wise_split
    return group_wise_split([entry[1] for entry in index], val_frac=val_frac, seed=seed)
