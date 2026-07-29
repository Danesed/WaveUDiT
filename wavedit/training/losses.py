import math
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torchmetrics.functional import structural_similarity_index_measure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ..evaluation.brats_metrics import (
    all_in_mask_metrics,
    aggregate_metrics,
    official_metrics_from_normalised,
)
from ..evaluation.visualization import visualize_inpainting_samples
from ..utils.logging_utils import get_logger
from .ema import EMA, lr_warmup_cosine

logger = get_logger(__name__)


def _atomic_save(blob: dict, path: str) -> None:
    """Write to a tmp file then os.replace. a SIGKILL/OOM mid-save (common on
    the shared box) can't corrupt the only checkpoint. Audit reliability fix."""
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(blob, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _ssim_masked_volume(pred, gt, mask):
    if not TORCHMETRICS_AVAILABLE:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    p = (pred.clamp(-1, 1) + 1.0) * 0.5
    g = (gt.clamp(-1, 1) + 1.0) * 0.5
    out = structural_similarity_index_measure(
        p, g, data_range=1.0, return_full_image=True
    )
    ssim_map = out[1] if isinstance(out, (tuple, list)) else out
    m = mask > 0.5
    if not bool(m.any()):
        return torch.ones((), device=pred.device, dtype=pred.dtype)
    # ssim_map matches input spatial shape (torchmetrics pads); index by mask.
    return ssim_map[m].mean()


def _ssim_per_slice_masked(pred, gt, mask, mask_threshold: int = 8):
    """Per-axial-slice 2D SSIM averaged over mask-touched slices. (B,1,D,H,W) in [-1,1]."""
    if not TORCHMETRICS_AVAILABLE:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    B, C, D, H, W = pred.shape
    p = (pred.clamp(-1, 1) + 1.0) * 0.5
    g = (gt.clamp(-1, 1) + 1.0) * 0.5
    p2d = p.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    g2d = g.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    m2d = mask.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    counts = m2d.flatten(1).sum(dim=1)
    keep = counts >= float(mask_threshold)
    if not bool(keep.any()):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return structural_similarity_index_measure(p2d[keep], g2d[keep], data_range=1.0)


def _ssim_full_volume(pred, gt):
    """Volumetric 3D SSIM on a (B,1,D,H,W) tensor in [-1,1]."""
    if not TORCHMETRICS_AVAILABLE:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    p = (pred.clamp(-1, 1) + 1.0) * 0.5
    g = (gt.clamp(-1, 1) + 1.0) * 0.5
    return structural_similarity_index_measure(p, g, data_range=1.0)


def _l1_boundary_weighted_in_mask(pred, gt, mask, alpha: float = 2.0,
                                  band_voxels: int = 6):
    if mask.sum() <= 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    with torch.no_grad():
        m = mask.float()
        m_outside = 1.0 - m
        dist = torch.zeros_like(m)
        running = m_outside
        for k in range(1, band_voxels + 1):
            running = F.max_pool3d(running, kernel_size=3, stride=1, padding=1)
            newly_covered = (running > 0.5) & (dist == 0) & (m > 0.5)
            dist = dist + newly_covered.float() * float(k)
        deep = (dist == 0) & (m > 0.5)
        dist = dist + deep.float() * float(band_voxels + 1)
        weight = m * (1.0 + alpha * torch.exp(-(dist - 1.0) / float(band_voxels)))
    diff = (pred - gt).abs() * weight
    denom = weight.sum().clamp_min(1.0)
    return diff.sum() / denom


class _SliceVGG16:
    _instance_by_device = {}

    @classmethod
    def get(cls, device):
        key = str(device)
        if key in cls._instance_by_device:
            return cls._instance_by_device[key]
        try:
            from torchvision.models import vgg16, VGG16_Weights
        except ImportError:
            logger.warning("torchvision not available; perceptual loss disabled")
            cls._instance_by_device[key] = None
            return None
        v = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:9].eval()
        for p in v.parameters():
            p.requires_grad = False
        v.to(device)
        cls._instance_by_device[key] = v
        return v


def _vgg_perceptual_slicewise(pred, gt, mask, k_slices: int = 8):
    """VGG16 relu2_2 perceptual L1 on axial slices that contain mask content.
    Up to k_slices per batch element. Pred/gt are (B,1,D,H,W) in [-1,1]."""
    vgg = _SliceVGG16.get(pred.device)
    if vgg is None:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    B, _, D, H, W = pred.shape
    m_slice = (mask.squeeze(1) > 0.5).any(dim=-1).any(dim=-1)  # (B, D)
    sel_p, sel_g = [], []
    for b in range(B):
        idx = m_slice[b].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        if idx.numel() > k_slices:
            perm = torch.randperm(idx.numel(), device=pred.device)[:k_slices]
            idx = idx[perm]
        sel_p.append(pred[b, 0].index_select(0, idx))
        sel_g.append(gt[b, 0].index_select(0, idx))
    if not sel_p:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    p = torch.cat(sel_p, dim=0).float()  # (K_total, H, W)
    g = torch.cat(sel_g, dim=0).float()
    p3 = ((p.clamp(-1, 1) + 1.0) * 0.5).unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    g3 = ((g.clamp(-1, 1) + 1.0) * 0.5).unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    with autocast(dtype=torch.float32, enabled=False):
        f_p = vgg(p3)
        f_g = vgg(g3)
        return (f_p - f_g).abs().mean()


def _grad_l1_in_mask(pred, gt, mask):
    if mask.sum() <= 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    total = pred.new_zeros(())
    denom = pred.new_zeros(())
    for ax in (2, 3, 4):  # D, H, W
        sl_lo = [slice(None)] * 5
        sl_hi = [slice(None)] * 5
        sl_lo[ax] = slice(None, -1)
        sl_hi[ax] = slice(1, None)
        gp = pred[tuple(sl_hi)] - pred[tuple(sl_lo)]
        gg = gt[tuple(sl_hi)] - gt[tuple(sl_lo)]
        m = torch.maximum(mask[tuple(sl_hi)], mask[tuple(sl_lo)])  # union
        total = total + ((gp - gg).abs() * m).sum()
        denom = denom + m.sum()
    return total / denom.clamp_min(1.0)


def _highfreq_l1_in_mask(pred, gt, mask, sigma: float = 1.0):
    if mask.sum() <= 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    from ..models.inpaint_extras import gaussian_blur3d
    hf_p = pred - gaussian_blur3d(pred, sigma)
    hf_g = gt - gaussian_blur3d(gt, sigma)
    diff = (hf_p - hf_g).abs() * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def _psnr_masked(pred, gt, mask):
    p = (pred.clamp(-1, 1) + 1.0) * 0.5
    g = (gt.clamp(-1, 1) + 1.0) * 0.5
    diff2 = ((p - g) ** 2) * mask
    mse = diff2.sum() / mask.sum().clamp_min(1.0)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-8))


def _rank_surrogate_loss(pred, gt, mask):
    ssim = _ssim_per_slice_masked(pred, gt, mask)
    psnr = _psnr_masked(pred, gt, mask)
    p = (pred.clamp(-1, 1) + 1.0) * 0.5
    g = (gt.clamp(-1, 1) + 1.0) * 0.5
    mse = (((p - g) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
    score = ssim + torch.clamp(psnr, max=40.0) / 40.0 + (1.0 - torch.clamp(mse * 10.0, max=1.0))
    return 3.0 - score, {"ssim": ssim, "psnr": psnr, "mse": mse}


def _set_topology_curriculum(train_loader, epoch: int, num_epochs: int):
    ds = getattr(train_loader, "dataset", None)
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    if ds is None or not hasattr(ds, "set_curriculum"):
        return
    progress = float(epoch + 1) / float(max(1, num_epochs))
    if progress < 0.33:
        stage = "compact"
    elif progress < 0.66:
        stage = "mixed"
    else:
        stage = "complex"
    ds.set_curriculum(stage=stage)


@torch.no_grad()
def _load_case(val_vol_dataset, vi):
    case = val_vol_dataset.cases[vi]
    if case.name in val_vol_dataset._ram_cache:
        blob = val_vol_dataset._ram_cache[case.name]
        return (case.name, blob["t1"].float(), blob["voided"].float(),
                blob["mask"].float(), float(blob["max_v"]))
    data = val_vol_dataset._slow_path(case)
    return (case.name,
            torch.from_numpy(data["padded"]["t1"]).float(),
            torch.from_numpy(data["padded"]["voided"]).float(),
            torch.from_numpy(data["padded"]["mask"]).float(),
            float(data["max_v"]))


@torch.no_grad()
def _validate(model, val_vol_dataset, val_indices, device, *, keep_for_viz: int = 4,
              use_mirror_tta: bool = False, eval_masks: dict = None):
    model.eval()
    per_case = []
    viz_buf = {"i_gt": [], "i_voided": [], "mask": [], "i_pred": []}
    for vi in val_indices:
        name, i_gt, voided, mask, max_v = _load_case(val_vol_dataset, vi)
        v = voided.unsqueeze(0).unsqueeze(0).to(device)
        m = mask.unsqueeze(0).unsqueeze(0).to(device)
        if use_mirror_tta and hasattr(model, "predict_mirror_consistent"):
            pred = model.predict_mirror_consistent(v, m).squeeze(0).squeeze(0).cpu()
        else:
            pred = model(v, m).squeeze(0).squeeze(0).cpu()
        score_mask = mask.numpy()
        if eval_masks is not None:
            mh = eval_masks.get(name)
            if mh is not None and float(mh.sum()) > 0:
                score_mask = mh
        metrics = official_metrics_from_normalised(
            pred_norm=pred.numpy(), gt_norm=i_gt.numpy(),
            voided_norm=voided.numpy(), mask=score_mask, max_v=max_v,
            device=device,
        )
        metrics["name"] = name
        per_case.append(metrics)
        if len(viz_buf["i_pred"]) < keep_for_viz:
            viz_buf["i_gt"].append(i_gt.numpy())
            viz_buf["i_voided"].append(voided.numpy())
            viz_buf["mask"].append(mask.numpy())
            viz_buf["i_pred"].append(pred.numpy())
    model.train()
    if not per_case:
        return None, None
    agg = aggregate_metrics(per_case)
    viz_arrays = {k: np.stack(v, axis=0) for k, v in viz_buf.items()} if viz_buf["i_pred"] else None
    return agg, viz_arrays
