# wavedit/data/augmentation.py

import math
import random
from typing import Dict

import torch
import torch.nn.functional as F


DEFAULT_AUG = {
    "flip_prob": 0.5,
    "rotation_deg": 10.0,
    "rotation_prob": 0.7,
    "gamma_range": (0.85, 1.15),
    "gamma_prob": 0.5,
    "brightness_range": (-0.05, 0.05),
    "brightness_prob": 0.5,
    "noise_sigma": 0.02,
    "noise_prob": 0.4,
    "bias_strength": 0.10,
    "bias_prob": 0.4,
    "mask_morph_prob": 0.3,
    "mask_morph_max_iter": 2,
    "topology_stage": "mixed",  # compact | mixed | complex
}


def _build_rot_matrix(rx: float, ry: float, rz: float) -> torch.Tensor:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotate_3d(sample: Dict[str, torch.Tensor], rx: float, ry: float, rz: float):
    D, H, W = sample["i_gt"].shape
    R = _build_rot_matrix(rx, ry, rz)
    theta = torch.zeros(1, 3, 4, dtype=torch.float32)
    theta[0, :, :3] = R.to(torch.float32)
    grid = F.affine_grid(theta, (1, 1, D, H, W), align_corners=False)
    for k, mode, pad in (("i_gt", "bilinear", "border"),
                         ("i_voided", "bilinear", "border"),
                         ("mask", "nearest", "zeros")):
        x = sample[k].unsqueeze(0).unsqueeze(0).float()
        y = F.grid_sample(x, grid, mode=mode, padding_mode=pad, align_corners=False)
        sample[k] = y.squeeze(0).squeeze(0).to(sample[k].dtype)


def _gamma(sample, gamma: float):
    for k in ("i_gt", "i_voided"):
        x = (sample[k] + 1.0) * 0.5
        sample[k] = x.clamp(0.0, 1.0).pow(gamma) * 2.0 - 1.0


def _brightness(sample, shift: float):
    for k in ("i_gt", "i_voided"):
        sample[k] = (sample[k] + shift).clamp(-1.0, 1.0)


def _noise(sample, sigma: float):
    sample["i_voided"] = (sample["i_voided"]
                          + torch.randn_like(sample["i_voided"]) * sigma).clamp(-1.0, 1.0)


def _bias_field(sample, strength: float):
    D, H, W = sample["i_gt"].shape
    low = torch.randn(1, 1, 8, 8, 8) * 0.3
    up = F.interpolate(low, size=(D, H, W), mode="trilinear", align_corners=False)
    up = up.squeeze(0).squeeze(0)
    up = up / up.abs().max().clamp_min(1e-6) * strength
    field = 1.0 + up
    for k in ("i_gt", "i_voided"):
        sample[k] = (sample[k] * field.to(sample[k].dtype)).clamp(-1.0, 1.0)


def _mask_morph(sample, iters: int, dilate: bool):
    m = sample["mask"].unsqueeze(0).unsqueeze(0).float()
    for _ in range(iters):
        if dilate:
            m = F.max_pool3d(m, 3, stride=1, padding=1)
        else:
            m = -F.max_pool3d(-m, 3, stride=1, padding=1)
    sample["mask"] = (m.squeeze(0).squeeze(0) > 0.5).to(sample["mask"].dtype)


def _mask_dropout_components(sample, drop_prob: float = 0.2):
    m = sample["mask"]
    if drop_prob <= 0:
        return
    drop = (torch.rand_like(m) < drop_prob).to(m.dtype)
    sample["mask"] = (m * (1.0 - drop) > 0.5).to(m.dtype)


def augment_inpaint(sample: Dict[str, torch.Tensor], cfg: dict = None,
                    skip_gpu_augs: bool = False) -> Dict[str, torch.Tensor]:
    """Apply the full pipeline to one sample dict (modifies in-place).
    `skip_gpu_augs=True` omits the expensive CPU intensity+geometric steps 2-6 (rotation, gamma,
    brightness, noise, bias field); they are instead applied batched on the GPU in the training
    loop (wavedit.data.gpu_augment.gpu_augment_batch), removing the ~4-5 s/sample bottleneck. Only
    the cheap, mask-related steps stay on CPU: flip (1) and mask morphology (7)."""
    c = {**DEFAULT_AUG, **(cfg or {})}

    # 1. flip — only along H (= L-R axis), the only physiologically symmetric
    # direction in the brain. Tensors here are 3D (D, H, W) with axes:
    #   0 = D (axial,  S-I direction) — NOT symmetric (top/bottom of head)
    #   1 = H (L-R)                  — symmetric (left/right hemispheres)
    #   2 = W (A-P)                  — NOT symmetric (front/back)
    # See wavedit/models/mirror_utils.py for the verification against the
    # BraTS LPS nifti axcodes.
    if random.random() < c["flip_prob"]:
        for k in ("i_gt", "i_voided", "mask"):
            sample[k] = sample[k].flip(1)

    # 2-6. rotation / gamma / brightness / noise / bias field -- the expensive intensity+geometric
    # steps. Skipped here when done batched on the GPU in the training loop (gpu_augment_batch).
    if not skip_gpu_augs:
        # 2. rotation
        if random.random() < c["rotation_prob"]:
            deg = c["rotation_deg"]
            rs = [random.uniform(-1, 1) * math.radians(deg) for _ in range(3)]
            _rotate_3d(sample, *rs)
        # 3. gamma
        if random.random() < c["gamma_prob"]:
            lo, hi = c["gamma_range"]
            _gamma(sample, random.uniform(lo, hi))
        # 4. brightness
        if random.random() < c["brightness_prob"]:
            lo, hi = c["brightness_range"]
            _brightness(sample, random.uniform(lo, hi))
        # 5. noise on voided
        if random.random() < c["noise_prob"]:
            _noise(sample, c["noise_sigma"])
        # 6. bias field
        if random.random() < c["bias_prob"]:
            _bias_field(sample, c["bias_strength"])

    # 7. mask morphology
    if random.random() < c["mask_morph_prob"]:
        stage = c.get("topology_stage", "mixed")
        if stage == "compact":
            iters = max(1, random.randint(1, c["mask_morph_max_iter"]))
            _mask_morph(sample, iters, dilate=False)
        elif stage == "complex":
            iters = max(2, random.randint(1, c["mask_morph_max_iter"] + 1))
            _mask_morph(sample, iters, dilate=(random.random() < 0.7))
            _mask_dropout_components(sample, drop_prob=0.1)
        else:  # mixed
            iters = random.randint(1, c["mask_morph_max_iter"])
            _mask_morph(sample, iters, dilate=(random.random() < 0.5))
        if "i_gt" in sample and "i_voided" in sample and "mask" in sample:
            m = sample["mask"] > 0.5
            sample["i_voided"] = torch.where(
                m, torch.full_like(sample["i_gt"], -1.0), sample["i_gt"]).to(sample["i_voided"].dtype)

    return sample
