#!/usr/bin/env bash

set -u
PY=${PYTHON:-python3}
DIR=${1:?checkpoint dir}
INTERVAL=${2:-300}

epoch_of () { "$PY" - "$1" <<'EOF' 2>/dev/null
import sys, torch
try:
    print(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("epoch", ""))
except Exception:
    print("")
EOF
}

echo "[snap] watching $DIR every ${INTERVAL}s"
while true; do
  for base in unet_best_ssim unet_best_aggregate; do
    src="$DIR/$base.pth"
    [ -f "$src" ] || continue
    e=$(epoch_of "$src")
    [ -n "$e" ] || continue
    dst="$DIR/$base.E$e.KEEP.pth"
    if [ ! -f "$dst" ]; then
      cp "$src" "$dst.part" && mv "$dst.part" "$dst"
      echo "[snap] $(date -u +%H:%M:%S) saved $base E$e"
    fi
  done
  sleep "$INTERVAL"
done
