# wavedit/evaluation/brats_metrics.py

from typing import Dict, Tuple

import numpy as np

try:
    import torch
    from torchmetrics.functional import structural_similarity_index_measure
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] == 1:
        return arr[0, 0]
    if arr.ndim != 3:
        raise ValueError(f"expected 3D array, got shape {arr.shape}")
    return arr


def mse_in_mask(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """MSE over mask voxels only. Inputs may be 3D arrays."""
    pred = _ensure_3d(pred).astype(np.float64)
    gt = _ensure_3d(gt).astype(np.float64)
    mask_b = _ensure_3d(mask) > 0.5
    n = int(mask_b.sum())
    if n == 0:
        return 0.0
    diff = (pred[mask_b] - gt[mask_b])
    return float((diff * diff).mean())


def psnr_in_mask(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                 data_range: float = None) -> float:
    """PSNR over mask voxels only. data_range defaults to gt.max() - gt.min()."""
    pred = _ensure_3d(pred).astype(np.float64)
    gt = _ensure_3d(gt).astype(np.float64)
    mask_b = _ensure_3d(mask) > 0.5
    if mask_b.sum() == 0:
        return 0.0
    mse = mse_in_mask(pred, gt, mask)
    if mse <= 0:
        return float("inf")
    if data_range is None:
        data_range = float(gt.max() - gt.min())
        if data_range < 1e-8:
            data_range = 1.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim_in_mask(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                 data_range: float = None) -> float:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torchmetrics + torch are required for SSIM")

    pred = _ensure_3d(pred).astype(np.float32)
    gt = _ensure_3d(gt).astype(np.float32)
    mask_b = _ensure_3d(mask) > 0.5
    if mask_b.sum() == 0:
        return 0.0

    if data_range is None:
        data_range = float(gt.max() - gt.min())
        if data_range < 1e-8:
            data_range = 1.0

    # Bounding box of the mask
    xs, ys, zs = np.where(mask_b)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    z0, z1 = zs.min(), zs.max() + 1

    # Pad bbox a bit so SSIM window has context; here SSIM uses 11x11 by default, give 6 voxels on each side.
    pad = 6
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad); z0 = max(0, z0 - pad)
    x1 = min(pred.shape[0], x1 + pad)
    y1 = min(pred.shape[1], y1 + pad)
    z1 = min(pred.shape[2], z1 + pad)

    pred_c = pred[x0:x1, y0:y1, z0:z1]
    gt_c = gt[x0:x1, y0:y1, z0:z1]
    mask_c = mask_b[x0:x1, y0:y1, z0:z1]

    # Per-slice 2D SSIM over Z, then average over mask voxels in each slice.
    # Tensors: (N_slices, 1, H, W) with N_slices = Z extent.
    Dz = pred_c.shape[2]
    if Dz == 0:
        return 0.0
    # Move Z to front so each "slice" is a (H, W) plane.
    p_2d = np.transpose(pred_c, (2, 0, 1))[:, None]  # (Dz, 1, X, Y)
    g_2d = np.transpose(gt_c,   (2, 0, 1))[:, None]
    m_2d = np.transpose(mask_c, (2, 0, 1))           # (Dz, X, Y) bool

    p_t = torch.from_numpy(p_2d).float()
    g_t = torch.from_numpy(g_2d).float()

    # `return_full_image=True` returns the per-voxel SSIM map.
    try:
        ssim_val, ssim_map = structural_similarity_index_measure(
            p_t, g_t, data_range=float(data_range), return_full_image=True
        )
        ssim_map = ssim_map.squeeze(1).numpy()  # (Dz, X, Y)
        weighted = ssim_map[m_2d]
        if weighted.size == 0:
            return 0.0
        return float(weighted.mean())
    except TypeError:
        # Older torchmetrics: no return_full_image. Fall back to scalar SSIM
        # on the bbox crop (slightly less precise but close).
        ssim_val = structural_similarity_index_measure(
            p_t, g_t, data_range=float(data_range)
        )
        return float(ssim_val)


def all_in_mask_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                        data_range: float = None) -> Dict[str, float]:
    """Return dict of SSIM / PSNR / MSE computed inside the mask only."""
    return {
        "ssim": ssim_in_mask(pred, gt, mask, data_range=data_range),
        "psnr": psnr_in_mask(pred, gt, mask, data_range=data_range),
        "mse":  mse_in_mask(pred, gt, mask),
    }


def official_metrics_from_normalised(
    pred_norm: np.ndarray,
    gt_norm: np.ndarray,
    voided_norm: np.ndarray,
    mask: np.ndarray,
    max_v: float,
    device=None,
) -> Dict[str, float]:

    if not TORCH_AVAILABLE:
        raise RuntimeError("torch required for official metrics")
    from .official_brats_metrics import generate_metrics

    pred_raw = (_ensure_3d(pred_norm).astype(np.float32) + 1.0) * 0.5 * float(max_v)
    gt_raw   = (_ensure_3d(gt_norm).astype(np.float32) + 1.0) * 0.5 * float(max_v)
    void_raw = (_ensure_3d(voided_norm).astype(np.float32) + 1.0) * 0.5 * float(max_v)
    mask_b   = _ensure_3d(mask) > 0.5

    # generate_metrics expects shape [1, X, Y, Z] in the ORIGINAL nifti axis order.
    _xyz = lambda a: np.ascontiguousarray(np.transpose(a, (1, 2, 0)))   # inverse of (2,0,1)
    p_t = torch.from_numpy(_xyz(pred_raw)).unsqueeze(0)
    g_t = torch.from_numpy(_xyz(gt_raw)).unsqueeze(0)
    v_t = torch.from_numpy(_xyz(void_raw)).unsqueeze(0)
    m_t = torch.from_numpy(_xyz(mask_b)).unsqueeze(0)
    if device is not None:
        p_t = p_t.to(device); g_t = g_t.to(device)
        v_t = v_t.to(device); m_t = m_t.to(device)

    return generate_metrics(prediction=p_t, target=g_t,
                            normalization_tensor=v_t, mask=m_t)


def aggregate_metrics(per_case: list) -> Dict[str, Dict[str, float]]:
    if not per_case:
        return {}
    keys = [k for k, v in per_case[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        vals = np.array([c[k] for c in per_case if k in c and np.isfinite(c[k])])
        if vals.size == 0:
            out[k] = {"mean": float("nan"), "median": float("nan"),
                      "iqr": float("nan"), "n": 0}
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        out[k] = {
            "mean":   float(vals.mean()),
            "median": float(np.median(vals)),
            "iqr":    float(q3 - q1),
            "min":    float(vals.min()),
            "max":    float(vals.max()),
            "n":      int(vals.size),
        }
    return out
