import os, numpy as np, torch, scipy.stats as st
from pathlib import Path
from scripts.model_io import load
from scripts.val_split import build_val
from scripts.tta_ensemble import predict_tta
from wavedit.data.brats_inpaint import (_load_nii, _normalise_t1, _to_axial_first,
                                        _pad_volume, save_inpainting_prediction)
from wavedit.evaluation.official_brats_metrics import generate_metrics

DATA = os.environ.get("BRATS_DATA", "data/BraTS-Local-Synthesis-Training")
CACHE = os.environ.get("BRATS_CACHE", "cache/inpaint")
TSHAPE = (160, 256, 256)
NCASES = int(os.environ.get("NCASES", 60))
PRESET = os.environ.get("PRESET", "crop_only")
MODELS = [tuple(x.split(":", 1)) for x in os.environ["MODELS"].split(",") if x.strip()]
dev = "cuda" if torch.cuda.is_available() else "cpu"
SCRATCH = "/tmp/claude-1001/select_tmp"; os.makedirs(SCRATCH, exist_ok=True)
print(f"device={dev} preset={PRESET} ncases={NCASES} models={[m[0] for m in MODELS]}", flush=True)

ds, vidx = build_val(DATA, CACHE, list(TSHAPE), 42, 0.05, NCASES)
names = []
for vi in vidx:
    d = ds._slow_path(ds.cases[vi]); pad = d["padded"]; mh = pad.get("mh")
    mk = pad["mask"].astype("float32")
    sm = (mh.astype("float32") if (mh is not None and mh.sum() > 0) else mk)
    if (sm > 0.5).sum() >= 10:
        names.append(ds.cases[vi].name)
print(f"cases {len(names)}", flush=True)


@torch.no_grad()
def score_case(model, name):
    c = Path(DATA) / name
    voided_xyz, affine, header = _load_nii(c / f"{name}-t1n-voided.nii.gz")
    mask_xyz, _, _ = _load_nii(c / f"{name}-mask.nii.gz"); mask_xyz = (mask_xyz > 0).astype(np.float32)
    voided_norm, max_v = _normalise_t1(voided_xyz)
    v_af = _pad_volume(_to_axial_first(voided_norm), TSHAPE, mode="reflect", value=0.0)
    m_af = (_pad_volume(_to_axial_first(mask_xyz), TSHAPE, mode="constant", value=0.0) > 0.5).astype(np.float32)
    native = _to_axial_first(voided_norm).shape
    v = torch.from_numpy(v_af)[None, None].float().to(dev)
    m = torch.from_numpy(m_af)[None, None].float().to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        p = predict_tta(model, v, m, PRESET)
    pred_xyz = save_inpainting_prediction(
        pred_image_norm_axial=p.float().squeeze(0).squeeze(0).cpu().numpy(),
        i_voided_original_xyz=voided_xyz, mask_original_xyz=mask_xyz, max_v=max_v,
        native_shape_axial_first=native, target_shape=TSHAPE,
        output_path=os.path.join(SCRATCH, f"{name}.nii.gz"), affine=affine, header=header)
    gt_xyz, _, _ = _load_nii(c / f"{name}-t1n.nii.gz")
    mh_xyz, _, _ = _load_nii(c / f"{name}-mask-healthy.nii.gz")
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))[None].to(dev)
    return generate_metrics(prediction=t(pred_xyz), target=t(gt_xyz),
                            normalization_tensor=t(voided_xyz),
                            mask=torch.from_numpy(mh_xyz > 0.5)[None].to(dev))


M = {n: {"ssim": [], "mse": [], "psnr": []} for n, _ in MODELS}
for n, ck in MODELS:
    model = load(ck, "udit", 64, dev)
    for i, name in enumerate(names, 1):
        r = score_case(model, name)
        M[n]["ssim"].append(r["ssim"]); M[n]["mse"].append(r["mse"]); M[n]["psnr"].append(r["psnr"])
        if i % 20 == 0 or i == len(names):
            print(f"  {n} [{i}/{len(names)}]", flush=True)
    del model; torch.cuda.empty_cache()

nm = [n for n, _ in MODELS]; N = len(names)
rank = {n: 0.0 for n in nm}
for i in range(N):
    rm = st.rankdata([M[n]["mse"][i] for n in nm])          # lower MSE = better
    rs = st.rankdata([-M[n]["ssim"][i] for n in nm])        # higher SSIM = better
    for j, n in enumerate(nm):
        rank[n] += (2.0 * rm[j] + rs[j]) / 3.0
rank = {n: rank[n] / N for n in nm}

print("\n%-14s %9s %9s %11s %11s" % ("model", "SSIMmean", "PSNRmed", "MSEmean", "rank(2:1)"))
for n in sorted(nm, key=lambda x: rank[x]):
    print("%-14s %.4f    %7.3f   %9.6f   %7.3f" % (
        n, np.mean(M[n]["ssim"]), np.median(M[n]["psnr"]), np.mean(M[n]["mse"]), rank[n]))
best = min(nm, key=lambda x: rank[x])
print(f"\nSUBMIT -> {best}")
print("(ranks are per-case; MSE weighted 2x because PSNR carries the same information)")
print("DONE")
