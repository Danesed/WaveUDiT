# wavedit/data/gpu_augment.py

import math

import torch
import torch.nn.functional as F


def _batched_rot_theta(ang: torch.Tensor) -> torch.Tensor:
    """ang: (B,3)=(rx,ry,rz) radians -> (B,3,4) affine, R=Rz@Ry@Rx (matches _build_rot_matrix)."""
    B = ang.shape[0]; dev = ang.device
    rx, ry, rz = ang[:, 0], ang[:, 1], ang[:, 2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    z = torch.zeros(B, device=dev); o = torch.ones(B, device=dev)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).view(B, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(B, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).view(B, 3, 3)
    R = Rz @ Ry @ Rx
    theta = torch.zeros(B, 3, 4, device=dev, dtype=ang.dtype)
    theta[:, :, :3] = R
    return theta


def _pcol(B, dev):
    return None  # placeholder (kept for readability of per-sample helpers below)


@torch.no_grad()
def gpu_augment_batch(i_gt, voided, mask, cfg):
    """Batched GPU port of augment_inpaint steps 2-6 on a (B,1,D,H,W) triple. `cfg` is the merged
    aug config (DEFAULT_AUG | dataset.aug_config). Returns the augmented (i_gt, voided, mask)."""
    B = i_gt.shape[0]; D, H, W = i_gt.shape[-3:]; dev = i_gt.device
    ig = i_gt.float(); vo = voided.float(); mk = mask.float()

    def sel(prob):                                              # (B,1,1,1,1) per-sample gate
        return (torch.rand(B, device=dev) < prob).float().view(B, 1, 1, 1, 1)

    # --- 2. rotation ---
    do = (torch.rand(B, device=dev) < cfg["rotation_prob"]).float()
    ang = (torch.rand(B, 3, device=dev) * 2 - 1) * math.radians(cfg["rotation_deg"]) * do.unsqueeze(1)
    grid = F.affine_grid(_batched_rot_theta(ang), (B, 1, D, H, W), align_corners=False)
    ig = F.grid_sample(ig, grid, mode="bilinear", padding_mode="border", align_corners=False)
    vo = F.grid_sample(vo, grid, mode="bilinear", padding_mode="border", align_corners=False)
    mk = F.grid_sample(mk, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    # --- 3. gamma (on images): x=(v+1)/2 in [0,1] -> x**g -> *2-1 ; per-sample g ---
    g = sel(cfg["gamma_prob"])
    lo, hi = cfg["gamma_range"]
    gam = (torch.rand(B, 1, 1, 1, 1, device=dev) * (hi - lo) + lo)
    gam = torch.where(g > 0, gam, torch.ones_like(gam))        # gamma=1 where not selected (identity)
    for t in ("ig", "vo"):
        x = ((locals()[t] + 1.0) * 0.5).clamp(0.0, 1.0)
        y = x.pow(gam) * 2.0 - 1.0
        if t == "ig": ig = y
        else: vo = y

    # --- 4. brightness (+shift, clamp) ---
    b = sel(cfg["brightness_prob"])
    blo, bhi = cfg["brightness_range"]
    shift = (torch.rand(B, 1, 1, 1, 1, device=dev) * (bhi - blo) + blo) * b
    ig = (ig + shift).clamp(-1.0, 1.0); vo = (vo + shift).clamp(-1.0, 1.0)

    # --- 5. noise on voided (MODEL INPUT ONLY) ---
    # The composite is `voided*(1-mask) + raw*mask` and is compared against a CLEAN i_gt, so noising
    # `voided` used to inject an error OUTSIDE the mask that the network can never cancel: the masked
    # SSIM floors at ~0.9976 and the high-frequency term at ~2e-4 even with a PERFECT infill. Keeping
    # a clean copy for the composite preserves the augmentation (the network still sees noise) while
    # removing the permanent loss floor.
    n = sel(cfg["noise_prob"])
    vo_clean = vo
    vo = (vo + torch.randn_like(vo) * cfg["noise_sigma"] * n).clamp(-1.0, 1.0)

    # --- 6. bias field: low-res (8^3) random field, trilinear upsample, multiplicative ---
    bf = sel(cfg["bias_prob"])
    low = torch.randn(B, 1, 8, 8, 8, device=dev) * 0.3
    up = F.interpolate(low, size=(D, H, W), mode="trilinear", align_corners=False)
    up = up / up.abs().amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6) * cfg["bias_strength"]
    field = 1.0 + up * bf                                       # field=1 where not selected
    ig = (ig * field).clamp(-1.0, 1.0); vo = (vo * field).clamp(-1.0, 1.0)
    vo_clean = (vo_clean * field).clamp(-1.0, 1.0)              # same field, so it stays consistent with ig

    return (ig.to(i_gt.dtype), vo.to(voided.dtype), mk.to(mask.dtype),
            vo_clean.to(voided.dtype))
