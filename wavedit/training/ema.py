import math

import torch


class EMA:
    """Inverse-power-schedule EMA of `module.state_dict()` (float tensors only)."""

    def __init__(self, module: torch.nn.Module, decay_max: float = 0.9999,
                 power: float = 0.6667):
        self.decay_max = float(decay_max)
        self.power = float(power)
        self.step = 0
        self.shadow = {
            k: v.detach().clone()
            for k, v in module.state_dict().items()
            if v.is_floating_point()
        }

    def _current_decay(self) -> float:
        if self.step == 0:
            return min(self.decay_max, (1.0 + self.step) ** -self.power)
        return min(self.decay_max, 1.0 - (1.0 + self.step) ** -self.power)

    @torch.no_grad()
    def update(self, module: torch.nn.Module):
        self.step += 1
        d = self._current_decay()
        msd = module.state_dict()
        for k, v in self.shadow.items():
            v.mul_(d).add_(msd[k].detach().to(v.dtype), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, module: torch.nn.Module):
        prev = {k: v.detach().clone() for k, v in module.state_dict().items()
                if v.is_floating_point()}
        msd = module.state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v)
        module.load_state_dict(msd, strict=False)
        return prev

    @torch.no_grad()
    def restore(self, module: torch.nn.Module, prev: dict):
        msd = module.state_dict()
        for k, v in prev.items():
            msd[k].copy_(v)
        module.load_state_dict(msd, strict=False)


def lr_warmup_cosine(step: int, total_steps: int, warmup_steps: int = 1000,
                     lr_max: float = 5e-4, lr_min: float = 1e-6) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * min(p, 1.0)))
