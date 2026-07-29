"""The convolutional block shared by the encoder and the decoder of every architecture.

Kept as its own module because all three backbones build their resolution levels out of it.
"""
from typing import Sequence

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(groups, c_out),
            nn.SiLU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(groups, c_out),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)
