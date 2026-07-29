# wavedit/models/hdit2d_udit_unet3d.py

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .udit_bottleneck import UDiTBottleneck3D, _channel_to_space_3d

try:
    from natten.functional import na2d
except Exception:  # natten optional at import time; required only when arch="hdit2d" is built
    na2d = None


def _gn(c: int) -> int:
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class Neighborhood2DBlock(nn.Module):


    def __init__(self, dim: int, d_head: int = 64, kernel: int = 7, dropout: float = 0.0,
                 ff_mult: int = 2, norm_groups: int = None):
        super().__init__()
        if na2d is None:
            raise ModuleNotFoundError("natten is required for arch='hdit2d' (na2d neighborhood attention)")
        d_head = next((dh for dh in range(min(d_head, dim), 0, -1) if dim % dh == 0), 1)
        self.n_heads = dim // d_head
        self.d_head = d_head
        self.kernel = kernel
        g = norm_groups if (norm_groups is not None and dim % norm_groups == 0) else _gn(dim)
        self.norm1 = nn.GroupNorm(g, dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.logit_scale = nn.Parameter(torch.full([self.n_heads, 1, 1], math.log(10.0)))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out.weight)                       # -> identity at init
        # depthwise-conv FFN at full 3D resolution (zero-init out -> identity at init)
        self.norm2 = nn.GroupNorm(g, dim)
        hidden = dim * ff_mult
        self.ff_in = nn.Conv3d(dim, hidden, 1)
        self.ff_dw = nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden)
        self.ff_act = nn.GELU()
        self.ff_out = nn.Conv3d(hidden, dim, 1)
        nn.init.zeros_(self.ff_out.weight); nn.init.zeros_(self.ff_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        # --- slice-wise 2D neighborhood attention over (H,W) ---
        h = self.norm1(x)
        h = h.permute(0, 2, 3, 4, 1).reshape(B * D, H, W, C)          # (B*D, H, W, C) channels-last
        qkv = self.qkv(h).reshape(B * D, H, W, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(3)                                       # each (B*D,H,W,nh,e)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp().view(1, 1, 1, self.n_heads, 1)
        q = q * scale
        q, k = q.to(v.dtype), k.to(v.dtype)   # F.normalize promotes to fp32 under autocast; na2d needs matching dtype
        k_eff = min(self.kernel, H, W)                                # kernel must fit the plane
        if k_eff % 2 == 0:
            k_eff -= 1                                                # na2d needs an odd window
        k_eff = max(1, k_eff)
        o = na2d(q, k, v, kernel_size=k_eff, scale=1.0)              # (B*D,H,W,nh,e)
        o = o.reshape(B * D, H, W, C)
        o = self.drop(self.out(o))
        o = o.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        x = x + o
        # --- conv FFN (full 3D) ---
        f = self.ff_in(self.norm2(x)); f = self.ff_dw(f); f = self.ff_act(f); f = self.ff_out(f)
        return x + f


class HDiT2DUDiTUNet3D(nn.Module):

    def __init__(self, in_channels: int = 2, base: int = 32, levels: int = 2, dropout: float = 0.0,
                 udit_downsample: int = 1, udit_d_head: int = 64, udit_blocks: int = 2,
                 nbhd_kernel: int = 7, blocks_per_level: int = 2, mask_attn: bool = False,
                 tokenrep: bool = False, norm_groups: int = 8, patch: int = 8,
                 out_channels: int = 1, final_tanh: bool = True):
        super().__init__()
        self.final_tanh = final_tanh
        self.mask_attn = mask_attn
        self.patch = patch
        chans: Sequence[int] = [base * (2 ** i) for i in range(levels)]

        def na_stack(dim):
            return nn.ModuleList([Neighborhood2DBlock(dim, d_head=udit_d_head, kernel=nbhd_kernel,
                                                      dropout=dropout, norm_groups=norm_groups)
                                  for _ in range(blocks_per_level)])

        self.patchify = nn.Conv3d(in_channels, chans[0], patch, stride=patch)
        self.enc = nn.ModuleList([na_stack(chans[0])])          # stage 0 processing (na2d)
        self.merge = nn.ModuleList()                            # token-merge between stages
        for i in range(len(chans) - 1):
            self.merge.append(nn.Conv3d(chans[i], chans[i + 1], 4, stride=2, padding=1, bias=False))
            self.enc.append(na_stack(chans[i + 1]))

        self.bottleneck = nn.ModuleList([
            UDiTBottleneck3D(chans[-1], downsample=udit_downsample, d_head=udit_d_head, dropout=dropout,
                             mask_attn=mask_attn, depth_attn=False, tokenrep=tokenrep, norm_groups=norm_groups)
            for _ in range(udit_blocks)])

        self.split = nn.ModuleList()                            # token-split (upsample)
        self.fuse = nn.ModuleList()                             # 1x1 fuse after skip-concat
        self.dec = nn.ModuleList()                              # na2d processing per decoder stage
        for i in reversed(range(len(chans) - 1)):
            self.split.append(nn.ConvTranspose3d(chans[i + 1], chans[i], 4, stride=2, padding=1, bias=False))
            self.fuse.append(nn.Conv3d(chans[i] * 2, chans[i], 1))
            self.dec.append(na_stack(chans[i]))


        self.out_channels = out_channels
        self.to_pixels = nn.Conv3d(chans[0], out_channels * (patch ** 3), 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h = self.patchify(x)
        skips = []
        for i, stage in enumerate(self.enc):
            for blk in stage:
                h = blk(h)
            skips.append(h)
            if i < len(self.merge):
                h = self.merge[i](h)
        hb = skips[-1]
        mask_bn = None
        if mask is not None and self.mask_attn:
            mask_bn = F.interpolate(mask.float(), size=hb.shape[-3:], mode="area")
        for blk in self.bottleneck:
            hb = blk(hb, mask=mask_bn)
        h = hb
        for split, fuse, stage, skip in zip(self.split, self.fuse, self.dec, reversed(skips[:-1])):
            h = split(h)
            if h.shape[-3:] != skip.shape[-3:]:
                h = F.interpolate(h, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            h = fuse(torch.cat([h, skip], dim=1))
            for blk in stage:
                h = blk(h)
        mean = _channel_to_space_3d(self.to_pixels(h), self.patch)   # (B, out, D, H, W)
        return mean.tanh() if self.final_tanh else mean


class HDiT2DWaveletUNet3D(nn.Module):

    def __init__(self, in_channels: int = 2, base: int = 32, levels: int = 2, dropout: float = 0.0,
                 udit_downsample: int = 1, udit_d_head: int = 64, udit_blocks: int = 2,
                 nbhd_kernel: int = 7, blocks_per_level: int = 2, mask_attn: bool = False,
                 tokenrep: bool = False, norm_groups: int = 8, patch: int = 4,
                 wavename: str = "haar"):
        super().__init__()
        from ..wavelets import DWT3D, IDWT3D
        self.dwt = DWT3D(wavename=wavename)
        self.idwt = IDWT3D(wavename=wavename)
        self.mask_attn = mask_attn
        self.body = HDiT2DUDiTUNet3D(
            in_channels=8 * in_channels, base=base, levels=levels, dropout=dropout,
            udit_downsample=udit_downsample, udit_d_head=udit_d_head, udit_blocks=udit_blocks,
            nbhd_kernel=nbhd_kernel, blocks_per_level=blocks_per_level, mask_attn=mask_attn,
            tokenrep=tokenrep, norm_groups=norm_groups, patch=patch,
            out_channels=8, final_tanh=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        subbands = self.dwt(x)                                  # 8 x (B, Cin, D/2, H/2, W/2)
        w = torch.cat(subbands, dim=1)                          # (B, 8*Cin, D/2, H/2, W/2)
        m_half = None
        if mask is not None and self.mask_attn:
            m_half = F.avg_pool3d(mask.float(), 2)              # half-res void fraction for attention
        o = self.body(w, mask=m_half)                          # (B, 8, D/2, H/2, W/2)
        rec = self.idwt(*[o[:, k:k + 1] for k in range(8)])    # (B, 1, D, H, W)
        return rec.tanh()
