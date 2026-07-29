import torch


def estimate_midline_h(x: torch.Tensor) -> torch.Tensor:
    """Intensity-weighted centre of mass along the H (= L-R) axis.

    Args:
        x: (B, 1, D, H, W) signal in [-1, 1]. Background is at -1 after
           normalisation, so we shift to [0, 1] before weighting — pure
           background contributes 0 to the centroid.

    Returns:
        center: (B,) fractional H index of the midline.
    """
    assert x.dim() == 5
    B, _, _, H, _ = x.shape
    w_pos = (x + 1.0) * 0.5            # [-1,1] -> [0,1], background ≈ 0
    hs = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, 1, H, 1)
    num = (w_pos * hs).flatten(1).sum(dim=1)
    den = w_pos.flatten(1).sum(dim=1).clamp_min(1.0)
    return num / den                   # (B,)


def mirror_h_around(x: torch.Tensor, center: torch.Tensor):
    """Mirror `x` along the H (= L-R) axis around a per-sample fractional `center`.

    Args:
        x:      (B, C, D, H, W) any tensor.
        center: (B,) fractional H index to mirror around.

    Returns:
        x_mirror: (B, C, D, H, W) — for each output index h, takes the value
                  at index round(2*center - h). Indices outside [0, H-1] are
                  clamped to the nearest in-bounds index (the `valid` mask
                  records where the gather was extrapolation).
        valid:    (B, 1, 1, H, 1) — 1 where the mirrored index was in-bounds,
                  else 0. Broadcastable against `x_mirror`.
    """
    assert x.dim() == 5
    B, C, D, H, W = x.shape
    fdtype = center.dtype
    hs = torch.arange(H, device=x.device, dtype=fdtype)                # (H,)
    mirror_idx = 2.0 * center.unsqueeze(-1) - hs.unsqueeze(0)          # (B, H)
    mirror_idx_int = mirror_idx.round().long()
    in_bounds = (mirror_idx_int >= 0) & (mirror_idx_int < H)           # (B, H)
    mirror_idx_safe = mirror_idx_int.clamp(0, H - 1)

    # Gather along H axis: index shape must match x.shape
    gather_idx = mirror_idx_safe.view(B, 1, 1, H, 1).expand(B, C, D, H, W)
    x_mirror = torch.gather(x, dim=-2, index=gather_idx)

    valid = in_bounds.view(B, 1, 1, H, 1).to(x.dtype)
    return x_mirror, valid


def contralateral_channels(voided: torch.Tensor, mask: torch.Tensor):
    """Build the two extra input channels for contralateral conditioning.

    Args:
        voided: (B, 1, D, H, W) in [-1, 1].
        mask:   (B, 1, D, H, W) in {0, 1}  the inpainting mask.

    Returns:
        voided_mirror: (B, 1, D, H, W) the mirrored anatomy.
        mirror_valid: (B, 1, D, H, W) 1 where the mirror voxel is trustworthy
            (in-bounds AND not itself inside the inpainting mask), else 0.
            The model can learn to ignore voided_mirror where this is 0.
    """
    center = estimate_midline_h(voided)
    voided_mirror, in_bounds = mirror_h_around(voided, center)
    mask_mirror, _ = mirror_h_around(mask, center)
    mirror_valid = in_bounds * (1.0 - mask_mirror)
    return voided_mirror, mirror_valid
