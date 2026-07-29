#!/usr/bin/env python3
"""Generate the 219-case official-validation leaderboard predictions for checkpoint.

IO matches the container entrypoint: per-case load voided+mask, normalise by voided-max, pad to
target_shape, mirror-TTA predict, unpad, de-normalise + composite (GT kept outside mask), write
BraTS-GLI-XXXXX-YYY-t1n-inference.nii.gz with the input affine/shape.

  CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=. python scripts/gen_val_predictions.py \
    --ckpt checkpoints_inpaint/WaveDiT_inpaint_contra_base48/unet_best_ssim.pth --base 48 \
    --out_dir submissions/val_predictions_base48 --zip submissions/val_predictions_base48.zip
"""
import os
import argparse, os, time, zipfile, numpy as np, torch
from pathlib import Path
from scripts.model_io import load
from wavedit.data.brats_inpaint import (_load_nii, _normalise_t1, _to_axial_first,
                                        _pad_volume, save_inpainting_prediction)

VAL_ROOT = os.environ.get("BRATS_VAL", "data/BraTS-Local-Synthesis-Validation")


def find_cases(val_root):
    root = Path(val_root)
    out = []
    for c in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("BraTS-GLI-")):
        v = c / f"{c.name}-t1n-voided.nii.gz"; m = c / f"{c.name}-mask.nii.gz"
        if v.is_file() and m.is_file():
            out.append((c.name, v, m))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--val_root", default=VAL_ROOT)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--zip", dest="zip_path", default=None)
    ap.add_argument("--target_shape", type=int, nargs=3, default=[160, 256, 256])
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    tshape = tuple(args.target_shape)
    torch.backends.cudnn.benchmark = True

    model = load(args.ckpt, "udit", args.base, dev)
    cases = find_cases(args.val_root)
    if args.limit:
        cases = cases[:args.limit]
    print(f"generating {len(cases)} cases -> {args.out_dir} (tta={not args.no_tta})", flush=True)

    t0 = time.time()
    for i, (name, vpath, mpath) in enumerate(cases, 1):
        voided_xyz, affine, header = _load_nii(vpath)
        mask_xyz, _, _ = _load_nii(mpath); mask_xyz = (mask_xyz > 0).astype(np.float32)
        voided_norm, max_v = _normalise_t1(voided_xyz)
        voided_af = _pad_volume(_to_axial_first(voided_norm), tshape, mode="reflect", value=0.0)
        mask_af = _pad_volume(_to_axial_first(mask_xyz), tshape, mode="constant", value=0.0)
        mask_af = (mask_af > 0.5).astype(np.float32)
        native_shape = _to_axial_first(voided_norm).shape
        v = torch.from_numpy(voided_af).unsqueeze(0).unsqueeze(0).float().to(dev)
        m = torch.from_numpy(mask_af).unsqueeze(0).unsqueeze(0).float().to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            if (not args.no_tta) and hasattr(model, "predict_mirror_consistent"):
                pred = model.predict_mirror_consistent(v, m)
            else:
                pred = model(v, m)
        pred_af = pred.float().squeeze(0).squeeze(0).cpu().numpy()
        save_inpainting_prediction(
            pred_image_norm_axial=pred_af, i_voided_original_xyz=voided_xyz,
            mask_original_xyz=mask_xyz, max_v=max_v, native_shape_axial_first=native_shape,
            target_shape=tshape, output_path=os.path.join(args.out_dir, f"{name}-t1n-inference.nii.gz"),
            affine=affine, header=header)
        if i % 20 == 0 or i == len(cases):
            print(f"  [{i}/{len(cases)}] {name}  ({(time.time()-t0)/i:.2f}s/case)", flush=True)

    n_out = len(list(Path(args.out_dir).glob("*-t1n-inference.nii.gz")))
    print(f"wrote {n_out} predictions in {time.time()-t0:.1f}s", flush=True)
    assert n_out == len(cases), f"expected {len(cases)} outputs, got {n_out}"

    if args.zip_path:
        with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(Path(args.out_dir).glob("*-t1n-inference.nii.gz")):
                z.write(f, arcname=f.name)   # flat zip (no top folder) per Synapse format
        print(f"ZIPPED -> {args.zip_path} ({os.path.getsize(args.zip_path)/1e6:.1f} MB, {n_out} files)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
