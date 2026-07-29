#!/usr/bin/env python3
import torch

from wavedit.models.direct_unet import DirectInpaintModel


def load(ckpt, arch, base, dev):
    c = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = c.get("config", {}) if isinstance(c, dict) else {}
    g = lambda k, d: cfg.get(k, d)
    m = DirectInpaintModel(
        base=g("base", base), levels=g("levels", 4), dropout=g("dropout", 0.2),
        arch=g("arch", arch), udit_blocks=g("udit_blocks", 2),
        udit_downsample=g("udit_downsample", 2), udit_d_head=g("udit_d_head", 64),
        in_contra=g("in_contra", False), in_sdf=g("in_sdf", False),
        sharpen_head=g("sharpen_head", False), sharpen_sigma=g("sharpen_sigma", 1.0),
        nonlocal_healthy=g("nonlocal_healthy", False),
        sdf_attn_bias=g("sdf_attn_bias", False),
        deco_head=g("deco_head", False), deco_channels=g("deco_channels", 64),
        deco_blocks=g("deco_blocks", 3),
        udit_tokenrep=g("udit_tokenrep", False),
        hdit_patch=g("hdit_patch", 8), contra_attn=g("contra_attn", False)).to(dev)
    sd = next((c[k] for k in ["model", "ema", "state_dict"] if isinstance(c, dict) and k in c and isinstance(c[k], dict)), c)
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"  loaded {ckpt} arch={g('arch', arch)} "
          f"extra=[contra={g('in_contra', False)} sdf={g('in_sdf', False)} "
          f"sharp={g('sharpen_head', False)} nl={g('nonlocal_healthy', False)} "
          f"dattn={g('sdf_attn_bias', False)} deco={g('deco_head', False)} "
          f"tokenrep={g('udit_tokenrep', False)}] miss={len(miss)} unexp={len(unexp)}")
    m.eval(); return m


def predict(model, voided, mask, dev, tta):
    v = torch.from_numpy(voided).unsqueeze(0).unsqueeze(0).to(dev)
    m = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        if tta and hasattr(model, "predict_mirror_consistent"):
            p = model.predict_mirror_consistent(v, m)
        else:
            p = model(v, m)
    return p.squeeze(0).squeeze(0).cpu().numpy()
