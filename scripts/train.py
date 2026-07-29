import argparse
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavedit.data.brats_inpaint import BraTSInpaintingDataset, collate_brats_inpainting
from wavedit.models.direct_unet import DirectInpaintModel
from wavedit.training.losses import (
    _ssim_masked_volume, _ssim_full_volume, _l1_boundary_weighted_in_mask,
    _grad_l1_in_mask, _vgg_perceptual_slicewise, _highfreq_l1_in_mask,
    _validate, _atomic_save,
)
from wavedit.training.ema import EMA, lr_warmup_cosine
from wavedit.evaluation.visualization import visualize_inpainting_samples
from wavedit.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _crop_to_void(tensors, mask, crop_dhw, jitter=16):
    """Crop a fixed-size (D,H,W) window around the void for each sample in the batch
    (Exp A: patch-crop training, the SOTA's compute-saving + regularizing trick).
    `tensors` is a list of (B,1,D,H,W) tensors cropped identically; `mask` drives the
    window center. Center = void centroid + uniform jitter, clamped to volume bounds.
    Returns the list of cropped tensors. If crop >= a dim, that dim is left full."""
    import torch as _t
    B, _, D, H, W = mask.shape
    cd, ch, cw = [min(c, s) for c, s in zip(crop_dhw, (D, H, W))]
    outs = [[] for _ in tensors]
    for b in range(B):
        idx = (mask[b, 0] > 0.5).nonzero(as_tuple=False)
        if idx.numel() == 0:
            cz = D // 2; cy = H // 2; cx = W // 2
        else:
            cz, cy, cx = [int(idx[:, k].float().mean().item()) for k in range(3)]
        if jitter > 0:
            cz += int(_t.randint(-jitter, jitter + 1, (1,)).item())
            cy += int(_t.randint(-jitter, jitter + 1, (1,)).item())
            cx += int(_t.randint(-jitter, jitter + 1, (1,)).item())
        z0 = min(max(cz - cd // 2, 0), D - cd)
        y0 = min(max(cy - ch // 2, 0), H - ch)
        x0 = min(max(cx - cw // 2, 0), W - cw)
        for i, t in enumerate(tensors):
            outs[i].append(t[b:b + 1, :, z0:z0 + cd, y0:y0 + ch, x0:x0 + cw])
    return [_t.cat(o, dim=0) for o in outs]


def parse_args():
    p = argparse.ArgumentParser()
    # Model.
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--arch", choices=["udit", "wudit", "whdit2d"], default="udit",
                   help="udit = U-DiT (submitted); wudit = WaveUDiT; whdit2d = WaveHUDiT")
    p.add_argument("--udit_blocks", type=int, default=2)
    p.add_argument("--udit_downsample", type=int, default=2)
    p.add_argument("--udit_d_head", type=int, default=64)
    # --- Round-2 ablation flags (fable_report.md §13), all default-off ---
    p.add_argument("--in_contra", action="store_true",
                   help="#2: add [mirrored-voided, mirror-valid] contralateral channels.")
    p.add_argument("--in_sdf", action="store_true",
                   help="A4: add the normalized void-depth coordinate as an input channel.")
    p.add_argument("--sharpen_head", action="store_true",
                   help="A2: 2nd head -> learned spatially-varying unsharp (needs --hf_weight>0).")
    p.add_argument("--sharpen_sigma", type=float, default=1.0)
    p.add_argument("--nonlocal_healthy", action="store_true",
                   help="A1: bottleneck void tokens attend only to known-healthy keys (udit only).")
    p.add_argument("--sdf_attn_bias", action="store_true",
                   help="A4: learned per-head depth key-bias in the bottleneck attention (udit only).")
    p.add_argument("--contra_attn", action="store_true",
                   help="A5: soft learned contralateral bias on the healthy-only attention logits "
                        "(boost keys near the query's L-R mirror; lambda init 0 -> exact warm-start). "
                        "Composes with --nonlocal_healthy; udit/wudit only.")
    p.add_argument("--deco_head", action="store_true",
                   help="DeCo per-voxel HF decode head (ported from WaveDiT-DEV); high-pass "
                        "residual, SSIM-safe anti-blur. Pairs well with --hf_weight.")
    p.add_argument("--deco_channels", type=int, default=64)
    p.add_argument("--deco_blocks", type=int, default=3)
    p.add_argument("--hdit_patch", type=int, default=8,
                   help="arch=hdit2d: non-overlapping patch-embed size (WaveDiT uses 8). "
                        "With levels=2 -> patch8 + 1 merge = /16 bottleneck (9x13x13).")
    p.add_argument("--udit_tokenrep", action="store_true",
                   help="TokenRep (ported from WaveDiT-DEV udit_tokenrep): band-grouped "
                        "overlapping-conv merge + depthwise-3x3 split in the U-DiT bottleneck "
                        "down/up of the QKV grid (udit only); zero-init residual, warm-start safe.")
    p.add_argument("--hf_weight", type=float, default=0.0,
                   help="Weight of the high-frequency anti-blur loss (anti-blur guard for A2 / realism lever).")
    p.add_argument("--hf_sigma", type=float, default=1.0)
    p.add_argument("--aug_strong", action="store_true",
                   help="#1: heavier intensity aug (gamma 0.7-1.5, stronger bias-field/brightness/noise).")
    p.add_argument("--init_from", default=None,
                   help="Partial weight load (strict=False) e.g. warm-start a udit model "
                        "from a trained cnn checkpoint. Loads weights only (no optimizer/epoch).")
    # Loss. Rank-1 recipe = MAE-on-mask + whole-volume SSIM (1:1), no grad/perceptual.
    p.add_argument("--l1_weight", type=float, default=1.0)
    p.add_argument("--healthy_loss_alpha", type=float, default=1.0,
                   help="Weight applied to void voxels that are TUMOUR tissue in the L1/L2 terms "
                        "(healthy voxels always weigh 1). Scoring is restricted to the healthy "
                        "sub-region, but the loss ran on the whole void, and measured over real "
                        "training draws 11.0%% of every injected void is tumour (median 5.8%%, 21%% of "
                        "samples >20%%) -- gradient spent teaching TUMOUR synthesis that is never "
                        "scored. 1.0 keeps the old behaviour; 0.0 is pure healthy-only.")
    p.add_argument("--l2_weight", type=float, default=0.0,
                   help="Weight of a masked MEAN-SQUARED-ERROR term. The official score is ~2/3 "
                        "squared error (per-case PSNR rank is exactly the inverse MSE rank, because "
                        "generate_metrics takes PSNR's data_range from the ground truth alone), and "
                        "MSE is minimised by the conditional MEAN whereas L1 gives the conditional "
                        "MEDIAN. Default 0.0 reproduces the previous objective exactly.")
    p.add_argument("--ssim_weight", type=float, default=1.0)
    p.add_argument("--grad_weight", type=float, default=0.0)
    p.add_argument("--perceptual_weight", type=float, default=0.0)
    p.add_argument("--eval_healthy", action="store_true",
                   help="Score/select validation SSIM on mask_healthy ONLY (the official metric "
                        "region), not the full void (~85%% tumor). THE correct convention.")
    p.add_argument("--mae_plain", action="store_true",
                   help="Plain masked-MAE (SOTA) instead of boundary-weighted L1.")
    p.add_argument("--select_on", choices=["median", "mean"], default="median",
                   help="Aggregate stat of healthy SSIM used for best-ckpt selection/early-stop.")
    p.add_argument("--ssim_mode", choices=["full", "masked"], default="full",
                   help="'full' = whole-volume SSIM on the RAW U-Net prediction (rank-1 Zhang recipe); "
                        "'masked' = SSIM averaged over mask voxels only (the official metric, our B1 setup).")
    # Data.
    p.add_argument("--data_root", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--in_ram", action="store_true")
    p.add_argument("--target_shape", type=int, nargs=3, default=[160, 256, 256])
    p.add_argument("--crop_size", type=int, nargs=3, default=None,
                   help="Exp A: train on a (D H W) crop around the void instead of the full "
                        "volume (SOTA patch-crop trick). Validation stays full-volume.")
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--val_max_cases", type=int, default=60)
    p.add_argument("--random_mask_prob", type=float, default=1.0,
                   help="Prob. of replacing the case mask with a fresh MaskBank sample. 1.0 = always.")
    p.add_argument("--masks_per_case", type=int, default=5,
                   help="Oversample each train case N times/epoch with different random masks (Zhang uses 5).")
    p.add_argument("--mask_bank_cache", default=None, help="Path to MaskBank .npz pool cache.")
    p.add_argument("--mask_bank_min_voxels", type=int, default=800)
    # Train.
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--run_name", default="unet_direct")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_every", type=int, default=2)
    p.add_argument("--patience", type=int, default=40, help="early-stop on best-SSIM stagnation")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="WaveDiT_challenge")
    p.add_argument("--wandb_id", default=None,
                   help="Resume a specific existing W&B run by id (e.g. after a crash/reboot). "
                        "If unset, a deterministic id derived from --run_name is used so a relaunch "
                        "of the same run_name re-attaches its run instead of forking a new one.")
    p.add_argument("--wandb_resume", default="allow",
                   help="W&B resume mode: allow (resume if id exists else create) | must | never.")
    p.add_argument("--crop_in_dataset", action="store_true",
                   help="Do the crop-to-void in the dataset workers")
    p.add_argument("--aug_on_gpu", action="store_true",
                   help="Do the 3D-rotation augmentation on the GPU (batched) instead of on CPU ")
    p.add_argument("--no_cudnn_benchmark", action="store_true",
                   help="Disable cudnn.benchmark.")
    p.add_argument("--channels_last", action="store_true",
                   help="Store the model + inputs in channels_last_3d memory format.")
    p.add_argument("--tf32", action="store_true",
                   help="Allow TF32 for the residual fp32 matmul/conv paths (bf16 autocast already "
                        "uses tensor cores for the bulk).")
    p.add_argument("--cudnn_benchmark_limit", type=int, default=0,
                   help="Cap the cudnn.benchmark autotune search to N algorithms (0 = unlimited).")
    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
    if args.cudnn_benchmark_limit > 0:
        torch.backends.cudnn.benchmark_limit = args.cudnn_benchmark_limit
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    viz_dir = os.path.join(args.checkpoint_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DirectInpaintModel(base=args.base, levels=args.levels, dropout=args.dropout,
                               arch=args.arch, udit_blocks=args.udit_blocks,
                               udit_downsample=args.udit_downsample,
                               udit_d_head=args.udit_d_head,
                               in_contra=args.in_contra, in_sdf=args.in_sdf,
                               sharpen_head=args.sharpen_head, sharpen_sigma=args.sharpen_sigma,
                               nonlocal_healthy=args.nonlocal_healthy,
                               sdf_attn_bias=args.sdf_attn_bias,
                               deco_head=args.deco_head, deco_channels=args.deco_channels,
                               deco_blocks=args.deco_blocks,
                               udit_tokenrep=args.udit_tokenrep,
                               hdit_patch=args.hdit_patch,
                               contra_attn=args.contra_attn).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
        logger.info("channels_last_3d memory format enabled (model + inputs)")
    n_par = sum(p.numel() for p in model.parameters())
    logger.info(f"DirectInpaintModel[{args.arch}]: base={args.base} levels={args.levels} "
                f"dropout={args.dropout}"
                + (f" udit_blocks={args.udit_blocks} ds={args.udit_downsample} "
                   f"d_head={args.udit_d_head}" if args.arch == "udit" else "")
                + f" -> {n_par:,} params")


    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        model_sd = model.state_dict()
        n_inflated, n_dropped = 0, 0
        for key in list(sd.keys()):
            if key in model_sd and sd[key].shape != model_sd[key].shape:
                src, dst = sd[key], model_sd[key]
                if (src.dim() == 5 and src.shape[0] == dst.shape[0]
                        and src.shape[1] < dst.shape[1] and src.shape[2:] == dst.shape[2:]):
                    new = dst.clone().zero_()
                    new[:, :src.shape[1]] = src
                    sd[key] = new
                    n_inflated += 1
                else:
                    del sd[key]
                    n_dropped += 1
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f"init_from {args.init_from}: strict=False "
                    f"({len(missing)} missing, {len(unexpected)} unexpected; "
                    f"{n_inflated} channel-inflated, {n_dropped} shape-dropped). "
                    f"Missing e.g.: {missing[:4]}")

    volume_ds = BraTSInpaintingDataset(root=args.data_root, target_shape=tuple(args.target_shape),
                                       train_mode=True, augment=True, seed=args.seed,
                                       cache_dir=args.cache_dir, in_ram=args.in_ram)
    volume_ds.aug_on_gpu = args.aug_on_gpu   # dataset then skips the expensive CPU 3D rotation
    if args.crop_in_dataset and args.crop_size is not None:
        volume_ds.crop_size = tuple(args.crop_size)   # workers crop -> small tensors over IPC
    if args.aug_strong:
        volume_ds.aug_config = {
            "gamma_range": (0.7, 1.5), "gamma_prob": 0.7,
            "brightness_range": (-0.1, 0.1), "brightness_prob": 0.6,
            "noise_sigma": 0.03, "noise_prob": 0.5,
            "bias_strength": 0.20, "bias_prob": 0.6,
        }
        logger.info(f"aug_strong: heavier intensity aug -> {volume_ds.aug_config}")
    n_total = len(volume_ds.cases)
    n_val = max(1, int(round(n_total * args.val_split)))
    rng = np.random.default_rng(args.seed)
    perm = np.arange(n_total); rng.shuffle(perm)
    val_indices = perm[:n_val].tolist()[: args.val_max_cases]
    train_indices = perm[n_val:].tolist()

    # Random-mask injection (the rank-1 lever). Attach MaskBank to the dataset.
    if args.random_mask_prob > 0:
        from wavedit.data.mask_bank import MaskBank
        volume_ds.mask_bank = MaskBank(root=args.data_root, cache_path=args.mask_bank_cache,
                                       min_voxels=args.mask_bank_min_voxels, verbose=True)
        volume_ds.random_mask_prob = float(args.random_mask_prob)
        volume_ds.load_extra_masks = True  # placement needs tumor/healthy masks
        logger.info(f"MaskBank attached: |pool|={len(volume_ds.mask_bank)} "
                    f"random_mask_prob={args.random_mask_prob}")
    # Zhang 2024: oversample each train case `masks_per_case` times/epoch (fresh mask each draw).
    if args.masks_per_case > 1 and args.random_mask_prob > 0:
        train_indices = train_indices * args.masks_per_case
        logger.info(f"masks_per_case={args.masks_per_case}: train samples/epoch -> {len(train_indices)}")
    logger.info(f"Split: train={len(train_indices)} val={len(val_indices)}")


    val_healthy = None
    if args.eval_healthy:
        volume_ds.load_extra_masks = True
        val_healthy = {}
        for vi in val_indices:
            case = volume_ds.cases[vi]
            padded = volume_ds._slow_path(case)["padded"]
            if "mh" in padded:
                val_healthy[case.name] = (padded["mh"] > 0.5).astype(np.float32)
        logger.info(f"eval_healthy: precomputed {len(val_healthy)} healthy val masks "
                    f"(scoring SSIM on mask_healthy, selecting on {args.select_on})")

    train_loader = DataLoader(Subset(volume_ds, train_indices), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_brats_inpainting, drop_last=True,
                              persistent_workers=(args.num_workers > 0),
                              prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model, decay_max=args.ema_decay)
    best_ssim = -1.0
    best_aggregate = -1e9   # 3-metric aggregate selection (SSIM + PSNR + MSE), never SSIM-only
    best_path = os.path.join(args.checkpoint_dir, "unet_best_ssim.pth")
    aggregate_path = os.path.join(args.checkpoint_dir, "unet_best_aggregate.pth")
    last_path = os.path.join(args.checkpoint_dir, "unet_last.pth")
    start_epoch, global_step, since_best = 0, 0, 0

    if args.resume_from and os.path.isfile(args.resume_from):
        ck = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); start_epoch = int(ck.get("epoch", -1)) + 1
        global_step = int(ck.get("global_step", 0)); best_ssim = float(ck.get("best_ssim", -1.0))
        if "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
        logger.info(f"Resumed from {args.resume_from}: epoch={start_epoch} best_ssim={best_ssim:.4f}")

    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            if args.wandb_id:
                wb_id, wb_resume = args.wandb_id, args.wandb_resume
            else:
                wb_id, wb_resume = None, None                 # None -> fresh id -> new run
            wandb.init(project=args.wandb_project, name=args.run_name, id=wb_id,
                       resume=wb_resume, config=vars(args),
                       dir=args.checkpoint_dir, settings=wandb.Settings(allow_val_change=True))
            logger.info(f"wandb run id={wb_id or 'auto (new run)'} resume={wb_resume}")
        except Exception as e:
            logger.warning(f"wandb init failed: {e}"); use_wandb = False

    total_steps = max(1, args.epochs * len(train_loader) // args.grad_accum)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss, run_n, accum = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"unet E{epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            i_gt = batch["i_gt"].to(device, non_blocking=True)
            voided = batch["i_voided"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            # Tumour mask, used only to down-weight never-scored voxels (see --healthy_loss_alpha).
            mu = (batch["mask_unhealthy"].to(device, non_blocking=True)
                  if (args.healthy_loss_alpha != 1.0 and "mask_unhealthy" in batch) else None)
            if args.channels_last:
                i_gt = i_gt.to(memory_format=torch.channels_last_3d)
                voided = voided.to(memory_format=torch.channels_last_3d)
                mask = mask.to(memory_format=torch.channels_last_3d)
            if args.crop_size is not None and not args.crop_in_dataset:
                tens = [i_gt, voided, mask] + ([mu] if mu is not None else [])
                out = _crop_to_void(tens, mask, args.crop_size)
                i_gt, voided, mask = out[0], out[1], out[2]
                if mu is not None: mu = out[3]
            if args.aug_on_gpu:

                from wavedit.data.gpu_augment import gpu_augment_batch
                from wavedit.data.augmentation import DEFAULT_AUG
                _augcfg = {**DEFAULT_AUG, **getattr(volume_ds, "aug_config", {})}
                _mk = torch.cat([mask, mu], dim=1) if mu is not None else mask
                i_gt, voided, _mk, voided_clean = gpu_augment_batch(i_gt, voided, _mk, _augcfg)
                if mu is not None: mask, mu = _mk[:, :1], _mk[:, 1:2]
                else: mask = _mk
            else:
                voided_clean = voided
            lr_now = lr_warmup_cosine(global_step, total_steps, warmup_steps=args.warmup_steps, lr_max=args.lr, lr_min=args.lr_min)
            for g in opt.param_groups: g["lr"] = lr_now
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                raw = model.raw(voided, mask)
                pred = voided_clean * (1.0 - mask) + raw * mask
                if mu is not None:
                    w = mask * (1.0 - (1.0 - args.healthy_loss_alpha) * (mu > 0.5).to(mask.dtype))
                else:
                    w = mask
                if args.mae_plain:
                    L_l1 = ((pred - i_gt).abs() * w).sum() / w.sum().clamp_min(1.0)
                else:
                    L_l1 = _l1_boundary_weighted_in_mask(pred, i_gt, mask)
                if args.ssim_mode == "full":
                    L_ssim = 1.0 - _ssim_full_volume(raw, i_gt)
                else:
                    L_ssim = 1.0 - _ssim_masked_volume(pred, i_gt, mask)
                L_grad = _grad_l1_in_mask(pred, i_gt, mask) if args.grad_weight > 0 else pred.new_zeros(())
                L_perc = _vgg_perceptual_slicewise(pred, i_gt, mask) if args.perceptual_weight > 0 else pred.new_zeros(())
                L_hf = _highfreq_l1_in_mask(pred, i_gt, mask, sigma=args.hf_sigma) \
                    if args.hf_weight > 0 else pred.new_zeros(())
                L_l2 = ((((pred - i_gt) ** 2) * w).sum() / w.sum().clamp_min(1.0)
                        if args.l2_weight > 0 else pred.new_zeros(()))
                loss = (args.l1_weight * L_l1 + args.l2_weight * L_l2 + args.ssim_weight * L_ssim
                        + args.grad_weight * L_grad + args.perceptual_weight * L_perc
                        + args.hf_weight * L_hf)
                (loss / args.grad_accum).backward()
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); accum = 0; continue
            accum += 1; run_loss += float(loss.detach()) * i_gt.size(0); run_n += i_gt.size(0)
            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step(); opt.zero_grad(set_to_none=True); ema.update(model); accum = 0; global_step += 1
            pbar.set_postfix({"loss": f"{float(loss.detach()):.4f}", "l1": f"{float(L_l1.detach()):.4f}",
                              "ssim": f"{float((1-L_ssim).detach()):.4f}", "lr": f"{lr_now:.2e}"})
        logger.info(f"unet E{epoch+1}/{args.epochs} | loss={run_loss/max(1,run_n):.5f} | step={global_step}")

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            prev = ema.apply_to(model)
            try:
                agg, viz_arrays = _validate(model, volume_ds, val_indices, device=device,
                                            keep_for_viz=4, use_mirror_tta=False,
                                            eval_masks=val_healthy)
                if agg:
                    s_med = agg["ssim"]["median"]; s_mean = agg["ssim"]["mean"]
                    ps = agg["psnr"]["median"]; ms = agg["mse"]["median"]
                    sel = agg["ssim"][args.select_on]
                    # 3-metric aggregate (BraTS-style: value SSIM+PSNR+MSE).
                    agg_score = s_mean + ps / 25.0 - ms / 0.005
                    region = "healthy" if val_healthy else "full"
                    logger.info(f"  unet val[{region}] (n={agg['ssim']['n']}): SSIM med={s_med:.4f} "
                                f"mean={s_mean:.4f} PSNR={ps:.2f} MSE={ms:.5f} AGG3={agg_score:.4f}")
                    if use_wandb:
                        import wandb; wandb.log({"unet/ssim": s_med, "unet/ssim_mean": s_mean,
                                                "unet/psnr": ps, "unet/mse": ms,
                                                "unet/agg3": agg_score}, step=global_step)
                    if sel > best_ssim:
                        best_ssim = sel; since_best = 0
                        _atomic_save({"model": model.state_dict(), "epoch": epoch,
                                      "global_step": global_step, "best_ssim": best_ssim,
                                      "ssim_at_best": s_med, "ssim_mean_at_best": s_mean,
                                      "psnr_at_best": ps, "mse_at_best": ms,
                                      "config": vars(args)}, best_path)
                        logger.info(f"  saved unet best-SSIM[{args.select_on}]={best_ssim:.4f}")
                    else:
                        since_best += args.val_every

                    if agg_score > best_aggregate:
                        best_aggregate = agg_score
                        _atomic_save({"model": model.state_dict(), "epoch": epoch,
                                      "global_step": global_step, "best_ssim": best_ssim,
                                      "ssim_at_best": s_med, "ssim_mean_at_best": s_mean,
                                      "psnr_at_best": ps, "mse_at_best": ms,
                                      "agg3_score": agg_score, "best_aggregate": best_aggregate,
                                      "config": vars(args)}, aggregate_path)
                        logger.info(f"  saved unet best-AGGREGATE3 score={agg_score:.4f} "
                                    f"(SSIM={s_mean:.4f} PSNR={ps:.2f} MSE={ms:.5f})")
                    # Triplane viz -> viz/epoch_XXXX.png
                    if viz_arrays is not None:
                        viz_path = os.path.join(viz_dir, f"epoch_{epoch+1:04d}.png")
                        try:
                            visualize_inpainting_samples(
                                i_gt=viz_arrays["i_gt"], i_voided=viz_arrays["i_voided"],
                                mask=viz_arrays["mask"], i_pred=viz_arrays["i_pred"],
                                out_path=viz_path, title_prefix=f"E{epoch+1} | ",
                                max_samples=4,
                                wandb_key=("val/inpaint_samples" if use_wandb else None),
                            )
                        except Exception as e:
                            logger.warning(f"viz failed: {e}")
            finally:
                ema.restore(model, prev)

        _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                      "epoch": epoch, "global_step": global_step, "best_ssim": best_ssim,
                      "config": vars(args)}, last_path)
        if since_best >= args.patience:
            logger.info(f"Early stop: no best-SSIM improvement in {since_best} epochs."); break

    logger.info(f"U-Net direct done. best_ssim={best_ssim:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
