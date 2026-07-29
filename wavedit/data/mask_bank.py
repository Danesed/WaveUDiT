# wavedit/data/mask_bank.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Convention: all candidates stored in AXIAL-FIRST order (D, H, W), matching
# the rest of brats_inpaint.py after `_to_axial_first`.

def _to_axial_first(arr_xyz: np.ndarray) -> np.ndarray:
    """(X, Y, Z) -> (Z, X, Y). Mirror of brats_inpaint._to_axial_first."""
    return np.transpose(arr_xyz, (2, 0, 1))


class MaskBank:
    """Pool of healthy-region candidate masks sampled from tumor connected
    components across the BraTS training set. Thread-safe across DataLoader
    workers because the pool is built once at construction and read-only
    afterwards (numpy arrays in main-process memory, shared via fork-COW)."""

    def __init__(
        self,
        root: str,
        cache_path: Optional[str] = None,
        min_voxels: int = 800,
        verbose: bool = True,
    ):
        if not SCIPY_AVAILABLE or nib is None:
            raise RuntimeError("mask_bank requires scipy + nibabel")

        self.root = Path(root)
        self.min_voxels = int(min_voxels)
        self.cache_path = Path(cache_path) if cache_path is not None else None

        # Each candidate is stored as the dense bool bbox + its voxel count.
        self._candidates: List[np.ndarray] = []
        self._sizes: np.ndarray = np.zeros(0, dtype=np.int64)

        if self.cache_path is not None and self.cache_path.is_file():
            self._load_cache()
            if verbose:
                logger.info(f"MaskBank: loaded {len(self._candidates)} candidates from cache {self.cache_path}")
        else:
            self._build_pool(verbose=verbose)
            if self.cache_path is not None:
                self._save_cache()
                if verbose:
                    logger.info(f"MaskBank: saved pool to cache {self.cache_path}")

    # ------------------------------------------------------------------
    # Pool build / cache
    # ------------------------------------------------------------------

    def _build_pool(self, verbose: bool):
        # Source from the REAL inpainting masks (`*-mask.nii.gz` = healthy ∪ unhealthy),
        # kept WHOLE (multi-component, peri-lesional, full size): this matches the
        # validation target distribution (100% multi-component, median ~220-240k vox).
        all_masks = sorted(self.root.glob("BraTS-GLI-*/BraTS-GLI-*-*-mask.nii.gz"))
        mask_paths = [p for p in all_masks
                      if not (p.name.endswith("-mask-healthy.nii.gz")
                              or p.name.endswith("-mask-unhealthy.nii.gz"))]
        if not mask_paths:
            raise FileNotFoundError(
                f"No inpainting masks under {self.root} (pattern: BraTS-GLI-*/*-mask.nii.gz)")
        if verbose:
            logger.info(f"MaskBank: scanning {len(mask_paths)} REAL inpainting masks (whole, multi-component) ≥ {self.min_voxels} voxels")

        for i, p in enumerate(mask_paths):
            try:
                vol = nib.load(p).get_fdata().astype(bool)
                vol = _to_axial_first(vol)
                sz = int(vol.sum())
                if sz < self.min_voxels:
                    continue
                # Tight bbox of the WHOLE mask (preserves multi-component structure).
                coords = np.where(vol)
                z0, z1 = coords[0].min(), coords[0].max() + 1
                h0, h1 = coords[1].min(), coords[1].max() + 1
                w0, w1 = coords[2].min(), coords[2].max() + 1
                self._candidates.append(vol[z0:z1, h0:h1, w0:w1].copy())
            except Exception as e:
                logger.warning(f"MaskBank: skipping {p.name}: {e}")
            if verbose and (i + 1) % 100 == 0:
                logger.info(f"  scanned {i+1}/{len(mask_paths)} → pool size {len(self._candidates)}")

        if not self._candidates:
            raise RuntimeError(f"MaskBank: empty pool after scanning {len(mask_paths)} files")
        self._sizes = np.array([int(c.sum()) for c in self._candidates], dtype=np.int64)
        if verbose:
            logger.info(f"MaskBank: built pool with {len(self._candidates)} candidates "
                        f"(sizes p25={np.percentile(self._sizes,25):.0f} "
                        f"med={np.percentile(self._sizes,50):.0f} "
                        f"p75={np.percentile(self._sizes,75):.0f})")

    def _save_cache(self):
        # Save as object array of bool arrays. .npz with allow_pickle=True
        # is the simplest path that survives variable shapes.
        np.savez_compressed(
            self.cache_path,
            candidates=np.array(self._candidates, dtype=object),
            sizes=self._sizes,
            min_voxels=np.int64(self.min_voxels),
        )

    def _load_cache(self):
        data = np.load(self.cache_path, allow_pickle=True)
        cands = data["candidates"]
        # np.savez(... dtype=object) stores as a 0-d or 1-d array of objects
        self._candidates = [np.asarray(c, dtype=bool) for c in cands]
        self._sizes = data["sizes"].astype(np.int64)

    def __len__(self) -> int:
        return len(self._candidates)

    # ------------------------------------------------------------------
    # Sampling primitives
    # ------------------------------------------------------------------

    def _rotate_xy_yz(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Random rotation 0-360° on H-W plane (axes 1,2) and D-W plane (axes 0,2),
        i.e. Zhang's XY and YZ planes given our (D, H, W) layout (D=axial,
        H=L-R, W=A-P). Uses order=0 (nearest) to keep binary."""
        angle_hw = float(rng.uniform(0.0, 360.0))
        angle_dw = float(rng.uniform(0.0, 360.0))
        mask = ndimage.rotate(mask.astype(np.uint8), angle=angle_hw,
                              axes=(1, 2), order=0, reshape=True, mode="constant", cval=0)
        # Trim the zero-padding the reshape=True rotation added BEFORE the second rotation, so it
        # runs on a tight array instead of the grown one. Content-equivalent (rotating a
        # zero-padded volume == zero-padded rotation of the content), just much cheaper.
        m = mask.astype(bool)
        if m.any():
            c = np.where(m)
            mask = mask[c[0].min():c[0].max()+1, c[1].min():c[1].max()+1, c[2].min():c[2].max()+1]
        mask = ndimage.rotate(mask, angle=angle_dw,
                              axes=(0, 2), order=0, reshape=True, mode="constant", cval=0)
        return mask.astype(bool)

    def _mirror(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Independent 50% flip per dim."""
        for axis in (0, 1, 2):
            if bool(rng.integers(0, 2)):
                mask = np.flip(mask, axis=axis)
        return np.ascontiguousarray(mask)

    def _trim_to_bbox(self, mask: np.ndarray) -> np.ndarray:
        """Tight bbox of a (possibly rotated, possibly flipped) bool mask."""
        if not mask.any():
            return mask
        coords = np.where(mask)
        z0, z1 = coords[0].min(), coords[0].max() + 1
        h0, h1 = coords[1].min(), coords[1].max() + 1
        w0, w1 = coords[2].min(), coords[2].max() + 1
        return mask[z0:z1, h0:h1, w0:w1]

    # Sample one fully transformed candidate

    def sample_transformed(
        self,
        target_shape: Tuple[int, int, int],
        rng: np.random.Generator,
        existing_tumor_size_pct: Optional[float] = None,
    ) -> np.ndarray:
        """Pick a candidate (optionally anti-correlated by size with the case's
        existing tumor), apply random mirror + rotation, return as a bool
        bbox-sized array.

        Note: this does NOT place the mask yet. Use `place_in_volume` to do
        validity-checked placement.
        """
        n = len(self._candidates)
        if existing_tumor_size_pct is None:
            idx = int(rng.integers(0, n))
        else:
            # Anti-correlation: if target percentile is 80%, pick a candidate
            # at the 20% percentile.
            target_pct = 100.0 - float(existing_tumor_size_pct)
            target_size = float(np.percentile(self._sizes, target_pct))
            # Pick from the 50 nearest-by-size candidates.
            order = np.argsort(np.abs(self._sizes - target_size))
            idx = int(order[int(rng.integers(0, min(50, n)))])

        cand = self._candidates[idx].copy()
        cand = self._mirror(cand, rng)
        cand = self._rotate_xy_yz(cand, rng)
        cand = self._trim_to_bbox(cand)
        return cand

    def place_in_volume(
        self,
        cand: np.ndarray,
        brain_mask: np.ndarray,
        target_shape: Tuple[int, int, int],
        rng: np.random.Generator,
        tumor_mask: Optional[np.ndarray] = None,
        min_dist_to_tumor: int = 5,
        max_bg_overlap: float = 0.10,   # was 0.25: injected masks voided 16.2% background vs 6.2% for real case masks (2.6x), diluting the masked-loss denominator
        max_tries: int = 30,
        tumor_dilated: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Attempt to place `cand` inside `target_shape` such that:
          * the mask fits within the volume,
          * at most `max_bg_overlap` of mask voxels lie OUTSIDE the brain,
          * if `tumor_mask` is provided, the mask is ≥ `min_dist_to_tumor`
            voxels away (Euclidean) from any tumor voxel.

        Returns the placed mask (shape == target_shape, bool) or None if
        no valid placement was found within `max_tries`.

        Perf: `tumor_dilated` may be passed precomputed (so the expensive full-volume
        binary_dilation runs once per sample, not once per resample). The per-try checks
        operate on the candidate's sub-region only (no full-volume array alloc / AND per
        try); the full `placed` array is built once, on success. Semantics are identical.
        """
        D, H, W = target_shape
        cd, ch, cw = cand.shape
        if cd > D or ch > H or cw > W:
            return None

        if tumor_dilated is None and tumor_mask is not None and tumor_mask.any():
            tumor_dilated = ndimage.binary_dilation(tumor_mask, iterations=min_dist_to_tumor)

        cand_count = int(cand.sum())
        if cand_count == 0:
            return None

        bg = ~brain_mask                                    # once, not per try
        for _ in range(max_tries):
            # Sample placement TOP-LEFT corner uniformly within valid range.
            z0 = int(rng.integers(0, D - cd + 1))
            h0 = int(rng.integers(0, H - ch + 1))
            w0 = int(rng.integers(0, W - cw + 1))
            sl = (slice(z0, z0+cd), slice(h0, h0+ch), slice(w0, w0+cw))

            # Background overlap check on the candidate sub-region only.
            if int((cand & bg[sl]).sum()) / cand_count > max_bg_overlap:
                continue
            # Tumor proximity check on the sub-region only.
            if tumor_dilated is not None and (cand & tumor_dilated[sl]).any():
                continue

            placed = np.zeros(target_shape, dtype=bool)     # only on success
            placed[sl] = cand
            return placed

        return None

    def sample_and_place(
        self,
        brain_mask: np.ndarray,
        target_shape: Tuple[int, int, int],
        rng: np.random.Generator,
        tumor_mask: Optional[np.ndarray] = None,
        existing_tumor_size_pct: Optional[float] = None,
        min_dist_to_tumor: int = 5,
        max_bg_overlap: float = 0.10,
        max_resamples: int = 5,
    ) -> Optional[np.ndarray]:
        """Top-level convenience: sample a transformed candidate and place it
        in `target_shape`. If placement fails, resample up to `max_resamples`
        times. Returns None if nothing valid is found (caller should fall back
        to the original mask)."""
        tumor_dilated = None
        if tumor_mask is not None and tumor_mask.any():
            tumor_dilated = ndimage.binary_dilation(tumor_mask, iterations=min_dist_to_tumor)
        for _ in range(max_resamples):
            cand = self.sample_transformed(target_shape, rng, existing_tumor_size_pct)
            placed = self.place_in_volume(
                cand, brain_mask, target_shape, rng,
                tumor_mask=tumor_mask,
                min_dist_to_tumor=min_dist_to_tumor,
                max_bg_overlap=max_bg_overlap,
                tumor_dilated=tumor_dilated,
            )
            if placed is not None:
                return placed
        return None
