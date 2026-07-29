#!/usr/bin/env bash
#
# ONE cosine cycle of the submitted U-DiT. The reported model is NOT a single run of this script:
# it is a chain of warm-started cycles, each starting from the best checkpoint of the previous one
# with the learning rate schedule restarted (--init_from loads weights only, not optimiser state):
#
#   cycle 1   50 epochs, lr 1e-4     from the previous base-64 champion
#   cycle 2   25 epochs, lr 1e-4     from cycle 1
#   cycle 3   25 epochs, lr 5e-5     from the best checkpoint of cycle 2   <- submitted model
#
# So roughly 100 epochs in total. Halving the learning rate in the last cycle matters: at the full
# 1e-4 the first step moves far enough to undo a good optimum, and the run needs most of the cycle
# just to recover it. Chain the cycles by passing INIT:
#
#   GPU=0 bash scripts/launch_udit_clean.sh                                  # cycle 1
#   GPU=0 INIT=<best of cycle 1> bash scripts/launch_udit_clean.sh           # cycle 2
#   GPU=0 LR=5e-5 INIT=<best of cycle 2> bash scripts/launch_udit_clean.sh   # cycle 3
#
# Use scripts/pick_best_aligned.py to choose the checkpoint each cycle starts from: the criterion
# that matches the official score is not the one the trainer's file names suggest.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"
BASE="${BASE:-64}"
EPOCHS="${EPOCHS:-25}"          # one cosine cycle; see the chain above for the full schedule
BATCH="${BATCH:-2}"             # measured: 2 fits in 66/94 GB and saturates the GPU, 3 goes OOM
LR="${LR:-1e-4}"                # use 5e-5 when warm-starting from an already good checkpoint
L2W="${L2W:-10.0}"
HALPHA="${HALPHA:-0.25}"
OUT="${OUT:-../checkpoints_inpaint/WaveDiT_inpaint_contra_udit_base${BASE}_clean}"
RUN="${RUN:-inpaint_contra_udit_base${BASE}_clean}"
DATA=${BRATS_DATA:?set BRATS_DATA to the challenge data root}
CACHE=${BRATS_CACHE:-cache/inpaint}
BANK=$CACHE/mask_bank_realmask_min400.npz

LOAD_ARG=""
if [ -n "${RESUME:-}" ]; then LOAD_ARG="--resume_from $RESUME"
elif [ -n "${INIT:-}" ]; then LOAD_ARG="--init_from $INIT"; fi
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
${PYTHON:-python3} -u scripts/train.py \
  --arch udit --base "$BASE" --levels 4 --dropout 0.2 \
  --udit_downsample 2 --udit_d_head 64 --udit_blocks 2 --sharpen_head --udit_tokenrep \
  --nonlocal_healthy --in_contra --contra_attn \
  $LOAD_ARG \
  --l2_weight "$L2W" --healthy_loss_alpha "$HALPHA" \
  --channels_last --tf32 \
  --hf_weight 0.5 --aug_on_gpu --crop_in_dataset --in_ram \
  --data_root "$DATA" --cache_dir "$CACHE" --mask_bank_cache "$BANK" \
  --target_shape 160 256 256 --crop_size 144 208 208 \
  --random_mask_prob 1.0 --masks_per_case 8 \
  --ssim_mode masked --mae_plain --eval_healthy --select_on mean \
  --batch_size "$BATCH" --grad_accum 1 \
  --lr "$LR" --lr_min 1e-7 --warmup_steps 300 --epochs "$EPOCHS" --patience 999 --ema_decay 0.995 \
  --checkpoint_dir "$OUT" --run_name "$RUN" \
  --num_workers 20 --wandb --wandb_project WaveDiT_challenge \
  >> "$OUT/train.log" 2>&1
