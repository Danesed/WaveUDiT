"""The held-out validation split, built the same way for every evaluation script.

Seeded permutation of the case list, so every candidate model is scored on exactly the same cases.
"""
import numpy as np

from wavedit.data.brats_inpaint import BraTSInpaintingDataset


def build_val(root, cache_dir, target_shape, seed, val_split, val_max):
    ds = BraTSInpaintingDataset(root=root, target_shape=tuple(target_shape), train_mode=True,
                                augment=False, seed=seed, cache_dir=cache_dir, in_ram=False)
    ds.load_extra_masks = True  # recompute mh/mu via _slow_path (disk cache lacks them)
    n_total = len(ds.cases)
    n_val = max(1, int(round(n_total * val_split)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    val_indices = perm[:n_val].tolist()[:val_max]
    return ds, val_indices
