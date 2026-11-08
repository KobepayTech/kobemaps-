#!/usr/bin/env bash
# One-command live mapping demo.
#
#   ./demo_live.sh                      # replay the bundled corridor scene
#   ./demo_live.sh webcam               # map from an attached camera
#   ./demo_live.sh stream rtsp://host/live   # map from a phone / IP camera
#
# Downloads the checkpoint on first run (4.63 GB), then opens a live map at
# http://localhost:8080 that builds as frames stream in.

set -euo pipefail

LINGBOT_DIR="${LINGBOT_DIR:-$(pwd)}"
MODE="${1:-replay}"
OUT_DIR="${OUT_DIR:-demo_out}"

if [ ! -f "$LINGBOT_DIR/live_map.py" ]; then
  echo "live_map.py not found in $LINGBOT_DIR" >&2
  echo "Run this from your lingbot-map checkout, or set LINGBOT_DIR." >&2
  exit 1
fi

echo "==> Locating checkpoint"
CKPT=$(python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("robbyant/lingbot-map", "lingbot-map.pt"))
PY
)
echo "    $CKPT"

# Without a GPU the model runs ~10 s/frame, so keep the demo short.
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "==> GPU detected"
  FRAMES="${FRAMES:-200}"
  BACKEND=""
else
  echo "==> No GPU — using SDPA fallback, short demo (~2 min)"
  FRAMES="${FRAMES:-11}"
  BACKEND="--use_sdpa"
fi

COMMON=(--model_path "$CKPT" $BACKEND --view --save_cloud --save_every 10
        --cloud_stride 40 --out_dir "$OUT_DIR")

case "$MODE" in
  replay)
    echo "==> Replaying example/loop ($FRAMES frames)"
    python live_map.py "${COMMON[@]}" \
      --source replay --replay_dir example/loop --max_frames "$FRAMES"
    ;;
  webcam)
    echo "==> Capturing from camera ${CAMERA:-0} — Ctrl-C to stop"
    python live_map.py "${COMMON[@]}" \
      --source webcam --camera "${CAMERA:-0}"
    ;;
  stream)
    URL="${2:-}"
    [ -n "$URL" ] || { echo "usage: $0 stream <rtsp://...>" >&2; exit 1; }
    echo "==> Capturing from $URL — Ctrl-C to stop"
    python live_map.py "${COMMON[@]}" --source stream --stream_url "$URL"
    ;;
  *)
    echo "usage: $0 [replay|webcam|stream <url>]" >&2
    exit 1
    ;;
esac

echo
echo "==> Map written to $OUT_DIR/live_cloud.ply"
echo "    Open in MeshLab or CloudCompare, or revisit http://localhost:8080"
