"""Model exports for the three architectures this repository ships.

`direct_unet` imports each architecture lazily inside its own dispatcher branch, so only the
module of the architecture you actually build is ever loaded. That is what keeps `--arch udit`
and `--arch wudit` free of the natten / dctorch dependencies that only WaveHUDiT needs.
"""
from .direct_unet import DirectInpaintModel

__all__ = ["DirectInpaintModel"]
