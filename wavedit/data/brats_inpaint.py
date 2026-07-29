# wavedit/data/brats_inpaint.py
#
# BraTS-2026 Local Synthesis (Inpainting) dataset.
#
# Per the official challenge spec:
#   - Each case folder is "BraTS-GLI-{ID}-{TP}/" containing:
#       BraTS-GLI-{ID}-{TP}-t1n.nii.gz           (training only) full T1 ground truth
#       BraTS-GLI-{ID}-{TP}-t1n-voided.nii.gz    T1 with mask region zeroed
#       BraTS-GLI-{ID}-{TP}-mask.nii.gz          binary mask of voxels to inpaint
#       BraTS-GLI-{ID}-{TP}-mask-healthy.nii.gz  (training only) the healthy patch
#       BraTS-GLI-{ID}-{TP}-mask-unhealthy.nii.gz (training only) the tumor region
#   - Native shape: (240, 240, 155) in nib (X, Y, Z).
#
# We transpose to put the 155-axial axis first -> (155, 240, 240) and pad to
# (160, 256, 256) so that the 3D Haar DWT yields a clean (80, 128, 128) wavelet
# map (slice dim 80, intra-slice 128x128 -> 16x16 patches with patch_size=(8,8)).
#
# Normalisation matches the official Pix2Pix3D baseline:
#   t1[t1<0] = 0;  max_v = t1.max();  t1 /= max_v;  t1 = 2*t1 - 1   in [-1, 1].
# We use the SAME max_v (from t1n if available; else from t1n-voided) for the
# voided volume so that the voided/GT scales are consistent.

import os
import re
import glob
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# Default pad-to shape (D, H, W) AFTER transposing axial-first.
DEFAULT_TARGET_SHAPE = (160, 256, 256)
# Native BraTS shape after the same transpose.
NATIVE_AXIAL_FIRST_SHAPE = (155, 240, 240)


def _compute_pad_amounts(native: Tuple[int, int, int],
                        target: Tuple[int, int, int]) -> List[int]:
    """Centred symmetric padding. Returns a 6-int list in F.pad order
    (last-dim first): (W_l, W_r, H_l, H_r, D_l, D_r)."""
    pad = []
    for n, t in zip(reversed(native), reversed(target)):
        total = max(0, t - n)
        left = total // 2
        right = total - left
        pad.extend([left, right])
    return pad


def _undo_pad(arr: np.ndarray,
              padded_shape: Tuple[int, int, int],
              native_shape: Tuple[int, int, int]) -> np.ndarray:
    """Strip the symmetric pad introduced by `_compute_pad_amounts`. `arr` is
    expected to be of shape `padded_shape` (D, H, W)."""
    sl = []
    for n, t in zip(native_shape, padded_shape):
        total = max(0, t - n)
        left = total // 2
        sl.append(slice(left, left + n))
    return arr[sl[0], sl[1], sl[2]]


def _to_axial_first(arr_xyz: np.ndarray) -> np.ndarray:
    """nib (X, Y, Z) -> (Z, X, Y) so the axial dim leads."""
    return np.transpose(arr_xyz, (2, 0, 1)).copy()


def _to_xyz(arr_axial_first: np.ndarray) -> np.ndarray:
    """Inverse of _to_axial_first."""
    return np.transpose(arr_axial_first, (1, 2, 0)).copy()


def _pad_volume(arr: np.ndarray,
                target_shape: Tuple[int, int, int],
                mode: str = "constant",
                value: float = 0.0) -> np.ndarray:
    """Pad a 3D numpy array (D, H, W) up to target_shape with symmetric centred
    padding. Uses torch.nn.functional.pad under the hood for `reflect` support."""
    cur = arr.shape
    if cur == target_shape:
        return arr
    if any(c > t for c, t in zip(cur, target_shape)):
        raise ValueError(f"Cannot pad: current shape {cur} larger than target {target_shape}")
    pad = _compute_pad_amounts(cur, target_shape)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()  # (1,1,D,H,W)
    t = F.pad(t, pad, mode=mode, value=value if mode == "constant" else 0.0)
    return t.squeeze(0).squeeze(0).numpy()


def _scan_dataset(root: str) -> List[Path]:
    """Return the list of case directories under `root`, one per BraTS-GLI case."""
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    cases = sorted([p for p in root_p.iterdir()
                    if p.is_dir() and p.name.startswith("BraTS-GLI-")])
    if len(cases) == 0:
        # Fall back: maybe a flat folder of NIfTIs (unsupported).
        raise RuntimeError(
            f"No BraTS-GLI-* case folders under {root}. Expected sub-folders per case."
        )
    return cases


# Filename helpers ------------------------------------------------------------

def _case_files(case_dir: Path):
    """Return a dict of expected file paths for a case folder."""
    name = case_dir.name
    return {
        "t1n":            case_dir / f"{name}-t1n.nii.gz",
        "voided":         case_dir / f"{name}-t1n-voided.nii.gz",
        "mask":           case_dir / f"{name}-mask.nii.gz",
        "mask_healthy":   case_dir / f"{name}-mask-healthy.nii.gz",
        "mask_unhealthy": case_dir / f"{name}-mask-unhealthy.nii.gz",
    }


def _load_nii(path: Path):
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine.copy(), img.header.copy()


# Normalisation ---------------------------------------------------------------

def _normalise_t1(t1: np.ndarray, max_v: Optional[float] = None):
    """BraTS-baseline normalisation: clip negatives, divide by max, scale [-1,1].
    Returns (normalised, max_v)."""
    t1 = t1.astype(np.float32, copy=True)
    t1[t1 < 0] = 0
    if max_v is None:
        max_v = float(t1.max())
    if max_v < 1e-8:
        max_v = 1.0
    t1n = t1 / max_v
    t1n = 2.0 * t1n - 1.0
    return t1n, max_v


# Augmentation ---------------------------------------------------------------

def _random_flip(arrays: List[np.ndarray], axes: Tuple[int, ...], rng: np.random.Generator):
    """Independently random-flip along each axis of the input axial-first volumes.
    All arrays in the list are flipped consistently."""
    flips = [bool(rng.integers(0, 2)) for _ in axes]
    out = []
    for a in arrays:
        b = a
        for ax, do in zip(axes, flips):
            if do:
                b = np.flip(b, axis=ax)
        out.append(np.ascontiguousarray(b))
    return out


# Dataset --------------------------------------------------------------------

class BraTSInpaintingDataset(Dataset):
    """Train/val dataset for the BraTS-2026 Local Synthesis (inpainting) task.

    Each __getitem__ returns a dict with float32 tensors of shape (1, D, H, W)
    where (D, H, W) == target_shape:

        - i_gt           T1 in [-1, 1]            (only if train_mode)
        - i_voided       T1-voided in [-1, 1]
        - mask           {0,1}, 1 = to-inpaint
        - mask_healthy   {0,1} (train_mode only)  subset of mask
        - mask_unhealthy {0,1} (train_mode only)  subset of mask
        - max_v          torch scalar — per-volume normalisation factor
        - name           str (case id)
        - native_shape   tuple (Z, X, Y) before padding
        - target_shape   tuple
        - affine         (4,4) float32 — to round-trip nib I/O
    """

    CACHE_VERSION = "v2" # v2: max_v is now derived from the VOIDED volume (inference convention) instead of t1n.

    def __init__(self,
                 root: str,
                 target_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
                 train_mode: bool = True,
                 augment: bool = True,
                 case_list: Optional[List[str]] = None,
                 seed: int = 42,
                 load_extra_masks: bool = False,
                 cache_dir: Optional[str] = None,
                 in_ram: bool = False,
                 ram_dtype: str = "float16",
                 aug_config: Optional[dict] = None,
                 mask_bank=None,
                 random_mask_prob: float = 0.0,
                 masks_per_case: int = 1,
                 ):
        self.root = root
        self.target_shape = tuple(target_shape)
        self.train_mode = train_mode
        self.augment = augment and train_mode
        self.aug_config = dict(aug_config or {})  # passed through to augment_inpaint()
        self.mask_bank = mask_bank
        self.random_mask_prob = float(random_mask_prob)
        self.masks_per_case = max(1, int(masks_per_case))
        self.load_extra_masks = load_extra_masks
        if cache_dir is None:
            self.cache_dir = None
        else:
            d, h, w = self.target_shape
            self.cache_dir = Path(cache_dir) / f"{self.CACHE_VERSION}_{d}x{h}x{w}"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        all_cases = _scan_dataset(root)
        if case_list is not None:
            case_set = set(case_list)
            all_cases = [c for c in all_cases if c.name in case_set]
        self.cases = all_cases

        if train_mode:
            # Filter cases that have the GT and healthy/unhealthy masks.
            kept = []
            for c in self.cases:
                f = _case_files(c)
                if f["t1n"].is_file() and f["voided"].is_file() and f["mask"].is_file():
                    kept.append(c)
                else:
                    logger.warning(f"Case {c.name}: missing required files for training. Skipping.")
            self.cases = kept
        else:
            kept = []
            for c in self.cases:
                f = _case_files(c)
                if f["voided"].is_file() and f["mask"].is_file():
                    kept.append(c)
            self.cases = kept

        if len(self.cases) == 0:
            raise RuntimeError(f"No usable cases in {root} for train_mode={train_mode}")

        logger.info(f"BraTSInpaintingDataset: {len(self.cases)} cases under {root} "
                    f"(train_mode={train_mode}, augment={self.augment}, "
                    f"in_ram={in_ram}, ram_dtype={ram_dtype}, "
                    f"cache_dir={'set' if self.cache_dir else 'none'})")

        self._rng = np.random.default_rng(seed)


        self._ram_cache: dict[str, dict] = {}
        self._ram_dtype = torch.float16 if ram_dtype == "float16" else torch.float32
        if in_ram:
            self._preload_to_ram()

    def set_curriculum(self, stage: str):
        """Update augmentation topology stage at runtime (compact/mixed/complex)."""
        if stage not in ("compact", "mixed", "complex"):
            raise ValueError(f"Unsupported curriculum stage: {stage}")
        self.aug_config["topology_stage"] = stage
        if stage == "compact":
            self.aug_config["mask_morph_prob"] = 0.15
            self.aug_config["mask_morph_max_iter"] = 1
        elif stage == "mixed":
            self.aug_config["mask_morph_prob"] = 0.3
            self.aug_config["mask_morph_max_iter"] = 2
        else:
            self.aug_config["mask_morph_prob"] = 0.5
            self.aug_config["mask_morph_max_iter"] = 3

    def __len__(self):
        return len(self.cases) * self.masks_per_case

    # Cache helpers

    def _cache_path(self, case_name: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        suffix = "_train" if self.train_mode else "_infer"
        if self.train_mode and self.load_extra_masks:
            suffix += "_extramasks"
        return self.cache_dir / f"{case_name}{suffix}.pt"

    def _slow_path(self, case_dir: Path) -> dict:
        """Read NIfTIs, normalise, transpose to axial-first, pad. Returns a
        dict of numpy arrays in axial-first padded space. No augmentation
        applied here — augmentation happens AFTER the cache hit."""
        files = _case_files(case_dir)
        voided_np, affine, _ = _load_nii(files["voided"])
        mask_np, _, _ = _load_nii(files["mask"])
        mask_np = (mask_np > 0).astype(np.float32)

        if self.train_mode:
            t1_np, _, _ = _load_nii(files["t1n"])
        else:
            t1_np = None

        if t1_np is not None:
            voided_norm, max_v = _normalise_t1(voided_np)
            t1_norm, _ = _normalise_t1(t1_np, max_v=max_v)
        else:
            voided_norm, max_v = _normalise_t1(voided_np)
            t1_norm = None

        mh_np = None
        mu_np = None
        if self.train_mode and self.load_extra_masks:
            if files["mask_healthy"].is_file():
                mh_np, _, _ = _load_nii(files["mask_healthy"])
                mh_np = (mh_np > 0).astype(np.float32)
            if files["mask_unhealthy"].is_file():
                mu_np, _, _ = _load_nii(files["mask_unhealthy"])
                mu_np = (mu_np > 0).astype(np.float32)

        voided_af = _to_axial_first(voided_norm)
        mask_af = _to_axial_first(mask_np)
        native_shape = voided_af.shape

        arrs_to_pad = {"voided": (voided_af, "reflect", 0.0),
                       "mask":   (mask_af,   "constant", 0.0)}
        if t1_norm is not None:
            arrs_to_pad["t1"] = (_to_axial_first(t1_norm), "reflect", 0.0)
        if mh_np is not None:
            arrs_to_pad["mh"] = (_to_axial_first(mh_np), "constant", 0.0)
        if mu_np is not None:
            arrs_to_pad["mu"] = (_to_axial_first(mu_np), "constant", 0.0)

        padded = {k: _pad_volume(v[0], self.target_shape, mode=v[1], value=v[2])
                  for k, v in arrs_to_pad.items()}
        padded["mask"] = (padded["mask"] > 0.5).astype(np.float32)
        if "mh" in padded:
            padded["mh"] = (padded["mh"] > 0.5).astype(np.float32)
        if "mu" in padded:
            padded["mu"] = (padded["mu"] > 0.5).astype(np.float32)

        return {
            "padded": padded,
            "max_v": float(max_v),
            "native_shape": tuple(int(x) for x in native_shape),
            "affine": affine.astype(np.float32),
        }

    def _save_cache(self, path: Path, data: dict) -> None:
        """Save preprocessed arrays compactly to disk. Images use float16,
        masks use uint8. Atomic rename to avoid partial files on crash."""
        padded = data["padded"]
        blob = {
            "voided": torch.from_numpy(padded["voided"]).half(),
            "mask":   torch.from_numpy(padded["mask"].astype(np.uint8)),
            "max_v":  data["max_v"],
            "native_shape": data["native_shape"],
            "affine": data["affine"],
        }
        if "t1" in padded:
            blob["t1"] = torch.from_numpy(padded["t1"]).half()
        if "mh" in padded:
            blob["mh"] = torch.from_numpy(padded["mh"].astype(np.uint8))
        if "mu" in padded:
            blob["mu"] = torch.from_numpy(padded["mu"].astype(np.uint8))
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        try:
            torch.save(blob, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def _preload_to_ram(self) -> None:
        """Fill ``self._ram_cache`` with every case's preprocessed tensors.
        Uses the disk cache when present; otherwise runs the slow NIfTI path
        (and writes to disk cache if ``cache_dir`` is set). Progress shown
        via tqdm + heartbeat log lines."""

        def _pack(data: dict) -> dict:
            """Convert a slow_path/load_cache 'data' dict to compact tensors
            for in-RAM storage."""
            padded = data["padded"]
            blob = {
                "voided":       torch.from_numpy(padded["voided"]).to(self._ram_dtype),
                "mask":         torch.from_numpy(padded["mask"].astype(np.uint8)),
                "max_v":        float(data["max_v"]),
                "native_shape": tuple(int(x) for x in data["native_shape"]),
                "affine":       np.asarray(data["affine"], dtype=np.float32),
            }
            if "t1" in padded:
                blob["t1"] = torch.from_numpy(padded["t1"]).to(self._ram_dtype)
            if "mh" in padded:
                blob["mh"] = torch.from_numpy(padded["mh"].astype(np.uint8))
            if "mu" in padded:
                blob["mu"] = torch.from_numpy(padded["mu"].astype(np.uint8))
            return blob

        def _go():
            for case_dir in self.cases:
                name = case_dir.name
                cache_path = self._cache_path(name)
                if cache_path is not None and cache_path.is_file():
                    try:
                        data = self._load_cache(cache_path)
                    except Exception as e:
                        logger.warning(f"RAM preload: cache read failed for {name}: {e}")
                        data = self._slow_path(case_dir)
                        if cache_path is not None:
                            try:
                                self._save_cache(cache_path, data)
                            except Exception:
                                pass
                else:
                    data = self._slow_path(case_dir)
                    if cache_path is not None:
                        try:
                            self._save_cache(cache_path, data)
                        except Exception as e:
                            logger.warning(f"RAM preload: cache write failed for {name}: {e}")
                self._ram_cache[name] = _pack(data)
                yield name


        from tqdm import tqdm
        n_total = len(self.cases)
        next_heartbeat = max(1, n_total // 10)
        bar = tqdm(_go(), total=n_total, desc="preload to RAM", leave=False)
        for i, name in enumerate(bar, start=1):
            if i == n_total or i % next_heartbeat == 0:
                logger.info(f"  RAM preload: {i}/{n_total} ({100*i//n_total}%) — {name}")
        bar.close()
        logger.info(f"BraTSInpaintingDataset: preloaded {len(self._ram_cache)} cases into RAM")

    def _ram_to_data(self, blob: dict) -> dict:
        """Convert an in-RAM compact blob back to a 'data' dict matching
        slow_path/_load_cache output (float32 numpy arrays in 'padded')."""
        padded = {
            "voided": blob["voided"].float().numpy(),
            "mask":   blob["mask"].float().numpy(),
        }
        if "t1" in blob:
            padded["t1"] = blob["t1"].float().numpy()
        if "mh" in blob:
            padded["mh"] = blob["mh"].float().numpy()
        if "mu" in blob:
            padded["mu"] = blob["mu"].float().numpy()
        return {
            "padded": padded,
            "max_v": float(blob["max_v"]),
            "native_shape": tuple(int(x) for x in blob["native_shape"]),
            "affine": np.asarray(blob["affine"], dtype=np.float32),
        }

    def _load_cache(self, path: Path) -> dict:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        padded = {
            "voided": blob["voided"].float().numpy(),
            "mask":   blob["mask"].float().numpy(),
        }
        if "t1" in blob:
            padded["t1"] = blob["t1"].float().numpy()
        if "mh" in blob:
            padded["mh"] = blob["mh"].float().numpy()
        if "mu" in blob:
            padded["mu"] = blob["mu"].float().numpy()
        return {
            "padded": padded,
            "max_v": float(blob["max_v"]),
            "native_shape": tuple(int(x) for x in blob["native_shape"]),
            "affine": np.asarray(blob["affine"], dtype=np.float32),
        }



    def __getitem__(self, idx):
        case_dir = self.cases[idx % len(self.cases)]
        name = case_dir.name

        # 1) Fast path: in-RAM
        if name in self._ram_cache:
            data = self._ram_to_data(self._ram_cache[name])
        else:
            cache_path = self._cache_path(name)
            # 2) Disk cache
            if cache_path is not None and cache_path.is_file():
                try:
                    data = self._load_cache(cache_path)
                except Exception as e:
                    logger.warning(f"Cache read failed for {name}: {e}; "
                                   f"falling back to NIfTI.")
                    data = self._slow_path(case_dir)
            # 3) Slow path: NIfTI -> normalise -> pad
            else:
                data = self._slow_path(case_dir)
                if cache_path is not None:
                    try:
                        self._save_cache(cache_path, data)
                    except Exception as e:
                        logger.warning(f"Cache write failed for {name}: {e}")

        padded = data["padded"]
        max_v = data["max_v"]
        native_shape = data["native_shape"]
        affine = data["affine"]

        # Random-mask injection 
        # Inject BEFORE augmentation so the new mask goes through the same
        # transforms (flip, rotation, etc.). Requires `t1` and optionally
        # `mu` (tumor mask) for placement.
        if (self.augment and self.mask_bank is not None
                and self.random_mask_prob > 0
                and "t1" in padded):
            try:
                rng = np.random.default_rng()  # worker-local randomness
                if rng.random() < self.random_mask_prob:
                    # Brain mask: voided/t1 in [-1,1] with background at -1.
                    # Threshold -0.99 to allow normalisation noise.
                    brain_mask_np = padded["t1"] > -0.99
                    new_mask = self.mask_bank.sample_and_place(
                        brain_mask=brain_mask_np,
                        target_shape=self.target_shape,
                        rng=rng,
                        tumor_mask=None,
                    )
                    if new_mask is not None:
                        new_mask_f = new_mask.astype(padded["mask"].dtype)
                        padded["mask"] = new_mask_f
                        padded["voided"] = np.where(
                            new_mask, np.float32(-1.0), padded["t1"]
                        ).astype(padded["voided"].dtype)
            except Exception as e:
                logger.warning(f"random-mask injection failed for {name}: {e}; "
                               f"using original mask")

        def _to_chw_tensor(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(a).unsqueeze(0).float()

        item = {
            "i_voided":     _to_chw_tensor(padded["voided"]),
            "mask":         _to_chw_tensor(padded["mask"]),
            "max_v":        torch.tensor(max_v, dtype=torch.float32),
            "name":         case_dir.name,
            "native_shape": native_shape,
            "target_shape": tuple(int(x) for x in self.target_shape),
            "affine":       affine,
        }
        if "t1" in padded:
            item["i_gt"] = _to_chw_tensor(padded["t1"])
        if "mh" in padded:
            item["mask_healthy"] = _to_chw_tensor(padded["mh"])
        if "mu" in padded:
            item["mask_unhealthy"] = _to_chw_tensor(padded["mu"])

        # Augmentation
        if self.augment and "i_gt" in item:
            from .augmentation import augment_inpaint
            sample = {
                "i_gt":     item["i_gt"].squeeze(0),       # (D, H, W)
                "i_voided": item["i_voided"].squeeze(0),
                "mask":     item["mask"].squeeze(0),
            }
            augment_inpaint(sample, cfg=self.aug_config,
                            skip_gpu_augs=getattr(self, "aug_on_gpu", False))
            item["i_gt"]     = sample["i_gt"].unsqueeze(0)
            item["i_voided"] = sample["i_voided"].unsqueeze(0)
            item["mask"]     = sample["mask"].unsqueeze(0)


        cs = getattr(self, "crop_size", None)
        if cs is not None and all(k in item for k in ("i_gt", "i_voided", "mask")):
            m = item["mask"]
            _, D, H, W = m.shape
            cd, ch, cw = [min(c, s) for c, s in zip(cs, (D, H, W))]
            idx = (m[0] > 0.5).nonzero(as_tuple=False)
            if idx.numel() == 0:
                cz, cy, cx = D // 2, H // 2, W // 2
            else:
                cz, cy, cx = [int(idx[:, k].float().mean().item()) for k in range(3)]
            j = 16
            cz += int(torch.randint(-j, j + 1, (1,)).item())
            cy += int(torch.randint(-j, j + 1, (1,)).item())
            cx += int(torch.randint(-j, j + 1, (1,)).item())
            z0 = min(max(cz - cd // 2, 0), D - cd)
            y0 = min(max(cy - ch // 2, 0), H - ch)
            x0 = min(max(cx - cw // 2, 0), W - cw)

            for k in ("i_gt", "i_voided", "mask", "mask_unhealthy", "mask_healthy"):
                if k in item:
                    item[k] = item[k][:, z0:z0 + cd, y0:y0 + ch, x0:x0 + cw].contiguous()

        return item


def collate_brats_inpainting(batch):
    """Stack tensor fields, keep python objects (str, tuple, ndarray) as lists."""
    if len(batch) == 0:
        return {}
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


# Inverse pipeline -----------------------------------------------------------

def reconstruct_full_volume(padded_pred_axial: np.ndarray,
                            native_shape_axial_first: Tuple[int, int, int],
                            target_shape: Tuple[int, int, int],
                            ) -> np.ndarray:
    """Inverse of the pad+transpose pipeline. Takes a padded prediction in
    axial-first order (D, H, W) and returns the array in nib order (X, Y, Z)
    with shape `native_shape_xyz` corresponding to the original case.

    `native_shape_axial_first` is the (Z, X, Y) shape recorded by the dataset.
    """
    unpadded = _undo_pad(padded_pred_axial, target_shape, native_shape_axial_first)
    return _to_xyz(unpadded)


def save_inpainting_prediction(pred_image_norm_axial: np.ndarray,
                               i_voided_original_xyz: np.ndarray,
                               mask_original_xyz: np.ndarray,
                               max_v: float,
                               native_shape_axial_first: Tuple[int, int, int],
                               target_shape: Tuple[int, int, int],
                               output_path: str,
                               affine: np.ndarray,
                               header: Optional[nib.Nifti1Header] = None,
                               ) -> np.ndarray:
    """End-to-end: given the model's prediction in [-1, 1] padded axial-first
    space, denormalise, composite with the voided original, and save the NIfTI
    in the submission format.

    Returns the saved volume (in nib X, Y, Z order)."""
    # 1) De-pad and transpose back to nib order
    pred_xyz = reconstruct_full_volume(pred_image_norm_axial,
                                       native_shape_axial_first, target_shape)
    # 2) De-normalise from [-1, 1] back to original intensity scale
    pred_xyz = (pred_xyz + 1.0) / 2.0          # -> [0, 1]
    pred_xyz = np.clip(pred_xyz, 0.0, 1.0)
    pred_xyz_full_scale = pred_xyz * max_v
    # 3) Compose: keep voided outside mask, predict inside mask
    mask_bool = mask_original_xyz > 0.5
    out = i_voided_original_xyz.astype(np.float32, copy=True)
    out[mask_bool] = pred_xyz_full_scale[mask_bool]
    # 4) Save
    img = nib.Nifti1Image(out, affine=affine, header=header)
    nib.save(img, output_path)
    return out
