"""Group-disjoint train/val splitting."""

import random
from typing import List, Sequence, Tuple

def group_wise_split(groups: Sequence, val_frac: float = 0.2,
                     seed: int = 0) -> Tuple[List[int], List[int]]:

    unique = sorted({g for g in groups}, key=str)
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(val_frac * len(unique)))) if len(unique) > 1 else 0
    val_groups = set(unique[:n_val])
    train_idx: List[int] = []
    val_idx: List[int] = []
    for i, g in enumerate(groups):
        (val_idx if g in val_groups else train_idx).append(i)
    return train_idx, val_idx
