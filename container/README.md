# Now You Have My Healthy Attention: A U-DiT for Brain-MRI Inpainting.
## BraTS-2026 Local Synthesis (Inpainting) submission container

This image provide  a deterministic regression 3D U-Net with a U-DiT-style downsampled global self-attention bottleneck (3D rotary position embeddings), non-local healthy-only attention and a contralateral-symmetry input, running at full resolution.

Self-contained: bundles the model code (`wavedit/`, torch-only), the entrypoint (`predict.py`) and
the weights (`model/model.pth`); the architecture is reconstructed from the checkpoint configuration.

Packaged to the Sage Bionetworks challenge template
(<https://github.com/Sage-Bionetworks-Challenges/sample-model-templates#build-your-model>).

## Contract

| item | value |
|---|---|
| input mount | `/input` (read-only) |
| output mount | `/output` (read-write) |
| invocation | `--input-dir /input --output-dir /output` (also the default `CMD`) |
| input per case | `<case>-t1n-voided.nii.gz` and `<case>-mask.nii.gz`, per-case sub-directory or flat layout, auto-detected recursively |
| output per case | `<case>-t1n-inference.nii.gz`, in the original image space (native shape and affine) |
| network | **not needed at run time**: every dependency and the weights are baked in at build time |

Tissue outside the mask is copied through bit-exactly; only the void is synthesised. Falls back to
CPU if no GPU is available.

## Running the evaluation


```bash
docker run --rm --gpus '"device=0"' --network none \
  --volume /path/to/test_cases:/input:ro \
  --volume /path/to/predictions:/output \
  <image> --input-dir /input --output-dir /output
```

The arguments are also the default `CMD`, so `docker run ... <image>` with no arguments behaves
identically. Nothing has to be set through the environment.

`/input` is walked recursively for every `*-t1n-voided.nii.gz` that has a sibling
`*-mask.nii.gz`; the layout can be one sub-directory per case or a single flat directory. Each case is normalised by the maximum of its own voided volume, inpainted, then de-normalised and written to `/output/<case>-t1n-inference.nii.gz` in the original image space, with the input affine and header preserved. Voxels outside the mask are copied from the input unchanged, so the prediction differs
from the input only inside the void. A case whose mask file is missing is reported on stdout and skipped. Progress is printed every ten cases, and the run ends with
`DONE: wrote N inference file(s)`.

**Requirements, measured on this image.** About **16 GB of VRAM** at peak on one GPU.

Without a visible GPU the entrypoint falls back to CPU automatically. That path was tested and
completes, at 208 s per case instead of 2. Its output is numerically equivalent but not
bit-identical to the GPU one, as expected from different kernels and reduction orders.

**If something looks wrong.** The first line of the log prints the checkpoint path and `missing=0 unexpected=0`: any other pair of numbers means the weights did not match the architecture. `Found N case(s)` reports how many inputs were discovered, which is the quickest way to catch a mounting or layout problem.

## Contents

```
Dockerfile        # pinned deps, weights baked in, no network at run time
predict.py        # entrypoint: load checkpoint -> per-case inference -> write NIfTI
tta_ensemble.py   # test-time view averaging (exactly-invertible views)
wavedit/          # inference-only subset of the model package, U-DiT path only based on WaveDiT package.
model/model.pth   # trained weights
```

## Build

```bash
cp <checkpoint>.pth model/model.pth
docker buildx build  --tag udit-inpaint:v1 .
```

## Test locally exactly the way the harness runs it


```bash
docker run --rm --network none \
  --volume $PWD/sample_data:/input:ro \
  --volume $PWD/output:/output:rw \
  udit-inpaint:v1 --input-dir /input --output-dir /output
```

Add `--gpus all` for GPU inference, or pass `--device cpu` to force CPU.

## Options (environment variables)

| variable | default | meaning |
|---|---|---|
| `BRATS_TTA` | `crop_only` | Test-time averaging. `crop_only` averages predictions over training-size (144x208x208) crops around the void; `mirror` restores the earlier prediction + left-right-flip behaviour. Measured on the official metric, `crop_only` improves SSIM, PSNR and MSE simultaneously, because the network is trained on crops and running it on the whole padded volume puts the bottleneck attention and the normalisation statistics outside their training regime. |
| `BRATS_CKPT` | `/opt/algorithm/model/model.pth` | checkpoint path |
