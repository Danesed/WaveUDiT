from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_blocks import _ConvBlock
from .udit_bottleneck import UDiTBottleneck3D
from .inpaint_extras import void_depth_channel, gaussian_blur3d


class UDiTBottleneckUNet3D(nn.Module):
    def __init__(self, in_channels: int = 2, base: int = 32, levels: int = 4,
                 dropout: float = 0.0, udit_blocks: int = 2, udit_downsample: int = 2,
                 udit_d_head: int = 64, sharpen_head: bool = False, sharpen_sigma: float = 1.0,
                 mask_attn: bool = False, depth_attn: bool = False, contra_attn: bool = False,
                 deco_head: bool = False, deco_channels: int = 64, deco_blocks: int = 3,
                 tokenrep: bool = False, norm_groups: int = 8,
                 out_channels: int = 1, final_tanh: bool = True):
        super().__init__()
        chans: Sequence[int] = [base * (2 ** i) for i in range(levels)]
        self.mask_attn = mask_attn
        self.depth_attn = depth_attn
        self.final_tanh = final_tanh

        self.stem = _ConvBlock(in_channels, chans[0], dropout=dropout, groups=norm_groups)

        self.down = nn.ModuleList()
        for i in range(len(chans) - 1):
            self.down.append(nn.Sequential(
                nn.Conv3d(chans[i], chans[i + 1], 4, stride=2, padding=1, bias=False),
                _ConvBlock(chans[i + 1], chans[i + 1], dropout=dropout, groups=norm_groups),
            ))

        self.bottleneck = nn.ModuleList([
            UDiTBottleneck3D(chans[-1], downsample=udit_downsample,
                             d_head=udit_d_head, dropout=dropout,
                             mask_attn=mask_attn, depth_attn=depth_attn, contra_attn=contra_attn,
                             tokenrep=tokenrep, norm_groups=norm_groups)
            for _ in range(udit_blocks)
        ])

        self.up = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(chans) - 1)):
            self.up.append(nn.ConvTranspose3d(chans[i + 1], chans[i], 4, stride=2, padding=1, bias=False))
            self.up_blocks.append(_ConvBlock(chans[i] * 2, chans[i], dropout=dropout, groups=norm_groups))

        self.head = nn.Conv3d(chans[0], out_channels, 1)
        self.sharpen_head = sharpen_head
        self.sharpen_sigma = sharpen_sigma
        if sharpen_head:
            self.sharp_head = nn.Conv3d(chans[0], 1, 1)
            nn.init.zeros_(self.sharp_head.weight)
            nn.init.constant_(self.sharp_head.bias, -4.0)
        self.deco_head = deco_head
        if deco_head:
            from .inpaint_extras import DeCoHFHead
            self.deco = DeCoHFHead(chans[0], model_channels=deco_channels,
                                   num_blocks=deco_blocks, sigma=sharpen_sigma)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                contra_center: torch.Tensor = None, contra_valid: torch.Tensor = None) -> torch.Tensor:
        skips = [self.stem(x)]
        h = skips[0]
        for d in self.down:
            h = d(h)
            skips.append(h)
        hb = skips[-1]
        mask_bn = depth_bn = valid_bn = None
        if mask is not None and (self.mask_attn or self.depth_attn):
            mask_bn = F.interpolate(mask.float(), size=hb.shape[-3:], mode="area")
            if self.depth_attn:
                depth_bn = void_depth_channel(mask_bn)
        if contra_valid is not None:
            valid_bn = F.interpolate(contra_valid.float(), size=hb.shape[-3:], mode="area")
        for blk in self.bottleneck:
            hb = blk(hb, mask=mask_bn, depth=depth_bn,
                     contra_center=contra_center, contra_valid=valid_bn)
        h = hb
        for up, blk, skip in zip(self.up, self.up_blocks, reversed(skips[:-1])):
            h = up(h)
            h = blk(torch.cat([h, skip], dim=1))
        mean = self.head(h)
        if not self.final_tanh:
            return mean                      # linear output (e.g. wavelet coefficients)
        mean = mean.tanh()
        if self.sharpen_head:
            s = F.softplus(self.sharp_head(h))
            hf = mean - gaussian_blur3d(mean, self.sharpen_sigma)
            mean = (mean + s * hf).clamp(-1.0, 1.0)
        if self.deco_head:
            mean = (mean + self.deco(h, mean)).clamp(-1.0, 1.0)
        return mean
