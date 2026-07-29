import sys, glob, os, torch

SSIM_SPREAD, MSE_SPREAD = 0.090, 0.00362 # Leaderboard spread of the metrics.


def aligned(ssim, mse):
    return ssim / SSIM_SPREAD - 2.0 * mse / MSE_SPREAD


def main(d):
    cands = sorted(set(glob.glob(os.path.join(d, "unet_best_*.pth"))))
    best, best_s = None, -1e18
    for p in cands:
        if os.path.basename(p).startswith("unet_last"):
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}", file=sys.stderr)
            continue
        s, m = ck.get("ssim_mean_at_best"), ck.get("mse_at_best")
        if s is None or m is None or m <= 0:
            continue
        sc = aligned(float(s), float(m))
        print(f"  {os.path.basename(p):38s} E{str(ck.get('epoch','?')):3s} "
              f"ssim={s:.4f} mse={m:.5f} aligned={sc:.4f}", file=sys.stderr)
        if sc > best_s:
            best, best_s = p, sc
    if best is None:
        sys.exit(f"no scorable checkpoint in {d}")
    print(best)


if __name__ == "__main__":
    main(sys.argv[1])
