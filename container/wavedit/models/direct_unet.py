import torch
import torch.nn as nn



class DirectInpaintModel(nn.Module):
    """Plain 3D U-Net inpainter.

    pred = unet(cat[voided, mask])  (tanh, in [-1,1]); the output is composited as
    out = voided*(1-mask) + pred*mask, so the known healthy tissue OUTSIDE the void
    is preserved exactly and only the hole is synthesised. Mirrors the official
    metric's compositing and matches the rank-1 inference convention.
    """

    def __init__(self, base: int = 32, levels: int = 4, dropout: float = 0.2,
                 arch: str = "cnn", udit_blocks: int = 2, udit_downsample: int = 2,
                 udit_d_head: int = 64, in_contra: bool = False, in_sdf: bool = False,
                 sharpen_head: bool = False, sharpen_sigma: float = 1.0,
                 nonlocal_healthy: bool = False, sdf_attn_bias: bool = False,
                 deco_head: bool = False, deco_channels: int = 64, deco_blocks: int = 3,
                 udit_tokenrep: bool = False, hdit_patch: int = 8,
                 contra_attn: bool = False):
        super().__init__()
        self.in_contra = in_contra
        self.in_sdf = in_sdf
        self.nonlocal_healthy = nonlocal_healthy
        self.contra_attn = contra_attn
        self.sdf_attn_bias = sdf_attn_bias
        in_channels = 2 + (2 if in_contra else 0) + (1 if in_sdf else 0)
        if arch == "udit":
            from .udit_unet3d import UDiTBottleneckUNet3D
            self.unet = UDiTBottleneckUNet3D(
                in_channels=in_channels, base=base, levels=levels, dropout=dropout,
                udit_blocks=udit_blocks, udit_downsample=udit_downsample,
                udit_d_head=udit_d_head, sharpen_head=sharpen_head,
                sharpen_sigma=sharpen_sigma, mask_attn=nonlocal_healthy,
                depth_attn=sdf_attn_bias, contra_attn=contra_attn, deco_head=deco_head,
                deco_channels=deco_channels, deco_blocks=deco_blocks,
                tokenrep=udit_tokenrep, norm_groups=max(8, base // 4))
        else:
            raise ValueError(f"unknown arch '{arch}': this container ships the U-DiT "
                             f"architecture only (the submitted model).")

    def _extra_channels(self, voided: torch.Tensor, mask: torch.Tensor):
        chans = []
        if self.in_contra:
            from .mirror_utils import contralateral_channels
            v_mirror, mirror_valid = contralateral_channels(voided, mask)
            chans += [v_mirror, mirror_valid]
        if self.in_sdf:
            from .inpaint_extras import void_depth_channel
            chans.append(void_depth_channel(mask))
        return chans

    def raw(self, voided: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([voided, mask, *self._extra_channels(voided, mask)], dim=1)
        if self.nonlocal_healthy or self.sdf_attn_bias:
            kw = {}
            if self.contra_attn:
                from .mirror_utils import estimate_midline_h, contralateral_channels
                kw["contra_center"] = estimate_midline_h(voided) / float(voided.shape[-2])
                _, mirror_valid = contralateral_channels(voided, mask)
                kw["contra_valid"] = mirror_valid
            return self.unet(inp, mask=mask, **kw)
        return self.unet(inp)

    def forward(self, voided: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw = self.raw(voided, mask)
        return voided * (1.0 - mask) + raw * mask

    @torch.no_grad()
    def predict_mirror_consistent(self, voided: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p1 = self.forward(voided, mask)
        vf = torch.flip(voided, dims=[-2])
        mf = torch.flip(mask, dims=[-2])
        p2 = torch.flip(self.forward(vf, mf), dims=[-2])
        return 0.5 * (p1 + p2)


class MeanEnsemble(nn.Module):

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, voided: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return sum(m(voided, mask) for m in self.models) / float(len(self.models))

    @torch.no_grad()
    def predict_mirror_consistent(self, voided: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return sum(m.predict_mirror_consistent(voided, mask)
                   for m in self.models) / float(len(self.models))
