#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$ROOT_DIR/app/radiology"
STUDIES_DIR="$ROOT_DIR/data/studies"
mkdir -p "$STUDIES_DIR"
export MONAILABEL_STUDIES="$STUDIES_DIR"
SEG_ENV="${1:-liver-seg-cpu}"
SEG_PROFILE="${2:-reference}"
SEG_PORT="${3:-8002}"
case "$SEG_PROFILE" in reference|fast) ;; *) echo "Profile must be reference or fast" >&2; exit 1 ;; esac
command -v conda >/dev/null 2>&1 || { echo "Conda must be on PATH" >&2; exit 1; }
conda run --no-capture-output -n "$SEG_ENV" python -m monailabel.main start_server \
  --app "$APP_DIR" \
  --studies "$STUDIES_DIR" \
  --host 127.0.0.1 \
  --port "$SEG_PORT" \
  --conf models "nnunet_liver,nnunet_liver_modelb,nnunet_liver_modelc" \
  --conf sam2 false \
  --conf scribbles false \
  --conf skip_trainers true \
  --conf inference_profile "$SEG_PROFILE" \
  --conf inference_threads 2
