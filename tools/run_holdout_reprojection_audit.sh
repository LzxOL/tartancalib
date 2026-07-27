#!/usr/bin/env bash
# Recreate Stage5 holdout overlays and Ours-vs-Kalibr prediction-difference crops.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 RESULT_DIR HOLDOUT_IMAGE_DIR [OUTPUT_DIR] [AUDIT_OPTIONS...]" >&2
  echo "Example: $0 RESULT_DIR IMAGE_DIR --frame-label 000087_right_518803198000_mono8" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
args=(
  --result-dir "$1"
  --image-dir "$2"
  --top-k 30
  --top-point-count 20
)
shift 2
if [[ $# -gt 0 && "$1" != --* ]]; then
  args+=(--output-dir "$1")
  shift
fi

python3 "$script_dir/re_render_holdout_reprojection_overlays.py" "${args[@]}" "$@"
