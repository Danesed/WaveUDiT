import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def void_depth_channel(mask: torch.Tensor, max_iter: int = 16) -> torch.Tensor:
    m = (mask > 0.5).float()
    cur = m
    depth = torch.zeros_like(m)
    for _ in range(max_iter):
        cur = -F.max_pool3d(-cur, kernel_size=3, stride=1, padding=1)  # binary erosion
        depth = depth + cur
        if float(cur.sum()) == 0.0:
            break
    return (depth / float(max_iter)).clamp(0.0, 1.0)


_GAUSS_CACHE: dict = {}


def _gaussian_kernel3d(sigma: float, device, dtype) -> torch.Tensor:
    key = (round(float(sigma), 4), str(device), str(dtype))
    k = _GAUSS_CACHE.get(key)
    if k is None:
        radius = max(1, int(math.ceil(2.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g1 = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g1 = g1 / g1.sum()
        g3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]      # (k,k,k)
        k = g3.view(1, 1, *g3.shape).to(device=device, dtype=dtype)
        _GAUSS_CACHE[key] = k
    return k


def gaussian_blur3d(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    if sigma <= 0:
        return x
    B, C, D, H, W = x.shape
    k = _gaussian_kernel3d(sigma, x.device, x.dtype)          # (1,1,k,k,k)
    r = k.shape[-1] // 2
    xp = F.pad(x, (r, r, r, r, r, r), mode="reflect")
    weight = k.expand(C, 1, *k.shape[2:])                     # depthwise
    return F.conv3d(xp, weight, groups=C)


_COORD_CACHE: dict = {}


def nerf_coord_features(shape, device, dtype, max_freqs: int = 6) -> torch.Tensor:
    D, H, W = shape[-3:]
    key = (D, H, W, str(device), str(dtype), int(max_freqs))
    feat = _COORD_CACHE.get(key)
    if feat is None:
        axes = []
        for n in (D, H, W):
            c = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float32)
            axes.append(c)
        gz, gy, gx = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")  # (D,H,W) each
        coords = torch.stack([gz, gy, gx], dim=0)                              # (3,D,H,W)
        freqs = (2.0 ** torch.arange(max_freqs, device=device, dtype=torch.float32)) * math.pi
        enc = coords[:, None] * freqs[None, :, None, None, None]               # (3,F,D,H,W)
        enc = torch.cat([torch.sin(enc), torch.cos(enc)], dim=1)              # (3,2F,D,H,W)
        feat = enc.reshape(1, 3 * 2 * max_freqs, D, H, W).to(dtype=dtype)
        _COORD_CACHE[key] = feat
    return feat


class _HFResBlock(nn.Module):
    """Per-voxel residual block (1x1x1 convs == a shared MLP over channels)."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch), nn.SiLU(), nn.Conv3d(ch, ch, 1),
            nn.GroupNorm(min(8, ch), ch), nn.SiLU(), nn.Conv3d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class DeCoHFHead(nn.Module):

    def __init__(self, feat_ch: int, model_channels: int = 64, num_blocks: int = 3,
                 max_freqs: int = 6, sigma: float = 1.0):
        super().__init__()
        self.max_freqs = max_freqs
        self.sigma = sigma
        coord_ch = 3 * 2 * max_freqs
        self.in_proj = nn.Conv3d(feat_ch + coord_ch + 1, model_channels, 1)   # +1 = mean
        self.blocks = nn.ModuleList([_HFResBlock(model_channels) for _ in range(num_blocks)])
        self.out = nn.Conv3d(model_channels, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        coords = nerf_coord_features(feats.shape, feats.device, feats.dtype, self.max_freqs)
        coords = coords.expand(feats.shape[0], -1, -1, -1, -1)
        x = self.in_proj(torch.cat([feats, coords, mean], dim=1))
        for b in self.blocks:
            x = b(x)
        hf = self.out(x)
        return hf - gaussian_blur3d(hf, self.sigma)   # high-pass: texture only
