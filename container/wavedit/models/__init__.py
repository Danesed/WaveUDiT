"""Inference-only model exports for the submission container.

The full research package eagerly imports the HDiT stack (einops / natten / flash_attn / dctorch)
and the training utilities (wandb / torchmetrics / matplotlib). None of that is needed to run a
trained U-DiT checkpoint, and installing it would make the image large and fragile, so the
container ships a minimal package: `direct_unet` imports each architecture lazily inside its own
branch, and this image ships the U-DiT path only.
"""
from .direct_unet import DirectInpaintModel

__all__ = ["DirectInpaintModel"]
