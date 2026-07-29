#!/usr/bin/env python3
"""BraTS-2026 Local Synthesis (Inpainting) containerised inference entrypoint.

Fills the masked region of each voided T1n and writes ``<case>-t1n-inference.nii.gz``
to the output directory in the original image space (tissue outside the mask is
preserved). Mirror test-time augmentation is on by default.

Inputs per case: ``<case>-t1n-voided.nii.gz`` and ``<case>-mask.nii.gz`` (per-case
sub-directory or flat layout, auto-detected). Input/output default to /input and
/output; override with --input/--output or BRATS_INPUT/BRATS_OUTPUT.

  python predict.py --input /input --output /output --ckpt model/model.pth
"""
import argparse, os, glob
import numpy as np
import torch

from wavedit.models.direct_unet import DirectInpaintModel
from wavedit.data.brats_inpaint import (_load_nii, _normalise_t1, _to_axial_first,
                                        _pad_volume, save_inpainting_prediction)


def load(ckpt, device):
    """Rebuild the architecture from the checkpoint config, then load the weights."""
    c = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = c.get("config", {}) if isinstance(c, dict) else {}
    g = lambda k, d: cfg.get(k, d)
    m = DirectInpaintModel(
        base=g("base", 32), levels=g("levels", 4), dropout=g("dropout", 0.2),
        arch=g("arch", "udit"), udit_blocks=g("udit_blocks", 2),
        udit_downsample=g("udit_downsample", 2), udit_d_head=g("udit_d_head", 64),
        in_contra=g("in_contra", False), in_sdf=g("in_sdf", False),
        sharpen_head=g("sharpen_head", False), sharpen_sigma=g("sharpen_sigma", 1.0),
        nonlocal_healthy=g("nonlocal_healthy", False), sdf_attn_bias=g("sdf_attn_bias", False),
        deco_head=g("deco_head", False), deco_channels=g("deco_channels", 64),
        deco_blocks=g("deco_blocks", 3), udit_tokenrep=g("udit_tokenrep", False),
        hdit_patch=g("hdit_patch", 8), contra_attn=g("contra_attn", False)).to(device)
    sd = next((c[k] for k in ["model", "ema", "state_dict"]
               if isinstance(c, dict) and k in c and isinstance(c[k], dict)), c)
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"  loaded {ckpt}  (missing={len(miss)} unexpected={len(unexp)})")
    m.eval(); return m


@torch.no_grad()
def predict(model, voided, mask, device, tta=True):
    """Inference for one padded volume.

    BRATS_TTA selects the test-time averaging regime (see tta_ensemble.PRESETS):
      'mirror'    - prediction + its left-right flip (the historical default),
      'crop_only' - average over TRAINING-SIZE (144x208x208) crops around the void only.
    The network is trained on crops but was historically run on the full padded volume, which puts
    the bottleneck attention and the GroupNorm statistics outside their training regime; measured on
    the official metric, 'crop_only' beats 'mirror' on SSIM, PSNR and MSE simultaneously.
    """
    v = torch.from_numpy(voided).unsqueeze(0).unsqueeze(0).to(device)
    m = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    if not tta:
        p = model(v, m)
    else:
        preset = os.environ.get("BRATS_TTA", "crop_only")
        if preset == "mirror":
            p = model.predict_mirror_consistent(v, m)
        else:
            from tta_ensemble import predict_tta
            p = predict_tta(model, v, m, preset)
    return p.squeeze(0).squeeze(0).cpu().numpy()


def _find_cases(input_dir):
    cases = []
    for vp in sorted(glob.glob(os.path.join(input_dir, "**", "*-t1n-voided.nii.gz"),
                               recursive=True)):
        cid = os.path.basename(vp).replace("-t1n-voided.nii.gz", "")
        mp = os.path.join(os.path.dirname(vp), f"{cid}-mask.nii.gz")
        if os.path.isfile(mp):
            cases.append((cid, vp, mp))
        else:
            print(f"  [skip] {cid}: missing {cid}-mask.nii.gz")
    return cases


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", "--input", dest="input",
                   default=os.environ.get("BRATS_INPUT", "/input"))
    p.add_argument("--output-dir", "--output", dest="output",
                   default=os.environ.get("BRATS_OUTPUT", "/output"))
    p.add_argument("--ckpt", default=os.environ.get("BRATS_CKPT", "/opt/algorithm/model/model.pth"))
    p.add_argument("--target_shape", type=int, nargs=3, default=[160, 256, 256])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_tta", action="store_true")
    args = p.parse_args()
    ts = tuple(args.target_shape); tta = not args.no_tta
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading model: {args.ckpt} (device={args.device}, tta={tta})")
    model = load(args.ckpt, args.device)
    cases = _find_cases(args.input)
    print(f"Found {len(cases)} case(s) under {args.input} -> {args.output}")
    done = 0
    for cid, vp, mp in cases:
        voided_xyz, affine, header = _load_nii(vp)
        mask_xyz, _, _ = _load_nii(mp); mask_xyz = (mask_xyz > 0).astype(np.float32)
        voided_norm, max_v = _normalise_t1(voided_xyz)
        voided_af = _pad_volume(_to_axial_first(voided_norm), ts, mode="reflect", value=0.0)
        mask_af = (_pad_volume(_to_axial_first(mask_xyz), ts, mode="constant", value=0.0) > 0.5
                   ).astype(np.float32)
        native = _to_axial_first(voided_norm).shape
        pred_af = predict(model, voided_af, mask_af, args.device, tta)
        save_inpainting_prediction(
            pred_image_norm_axial=pred_af.astype(np.float32),
            i_voided_original_xyz=voided_xyz, mask_original_xyz=mask_xyz, max_v=max_v,
            native_shape_axial_first=native, target_shape=ts,
            output_path=os.path.join(args.output, f"{cid}-t1n-inference.nii.gz"),
            affine=affine, header=header)
        done += 1
        if done % 10 == 0 or done == len(cases):
            print(f"  [{done}/{len(cases)}] {cid}")
    print(f"DONE: wrote {done} inference file(s) to {args.output}")


if __name__ == "__main__":
    main()
