#!/usr/bin/env bash
# Recreate Stage5 holdout overlays and Ours-vs-Kalibr prediction-difference crops.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 RESULT_DIR HOLDOUT_IMAGE_DIR [OUTPUT_DIR] [AUDIT_OPTIONS...]" >&2
  echo "Example: $0 RESULT_DIR IMAGE_DIR --frame-label 000087_right_518803198000_mono8" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
result_dir="$1"
image_dir="$2"
output_dir="$result_dir/holdout_reprojection_visualizations_cross_circle"
args=(
  --result-dir "$result_dir"
  --image-dir "$image_dir"
  --top-k 30
  --top-point-count 20
)
shift 2
if [[ $# -gt 0 && "$1" != --* ]]; then
  output_dir="$1"
  shift
fi
args+=(--output-dir "$output_dir")

python3 "$script_dir/re_render_holdout_reprojection_overlays.py" "${args[@]}" "$@"

# The audit is based on the same frozen holdout artifact as the overlays.  Add
# coverage metrics beside the figures so a visually broad distribution can be
# compared numerically across datasets without rerunning Stage5.
points_csv="$result_dir/benchmark_holdout_points.csv"
if [[ -f "$points_csv" ]]; then
  python3 "$script_dir/analyze_holdout_reprojection_distribution.py" \
    --points "$points_csv" \
    --output-dir "$output_dir"
else
  echo "Warning: $points_csv not found; skipping distribution metrics." >&2
fi

# The holdout audit above is intentionally restricted to the frozen benchmark
# points.  Also export an aggregate image-plane figure for all successful
# frontend outer-corner rows when the Stage5 run contains that artifact.
frontend_views="$result_dir/auto_camera_initialization_bootstrap_views.csv"
if [[ -f "$frontend_views" ]]; then
  plot_python="python3"
  if ! python3 -c 'import matplotlib' >/dev/null 2>&1; then
    if /usr/bin/python3 -c 'import matplotlib' >/dev/null 2>&1; then
      plot_python="/usr/bin/python3"
    else
      echo "Warning: matplotlib not found; skipping frontend distribution figure." >&2
      plot_python=""
    fi
  fi
  if [[ -n "$plot_python" ]]; then
    "$plot_python" "$script_dir/plot_frontend_outer_corner_distribution.py" \
      --views "$frontend_views" \
      --output-dir "$output_dir"
  fi
else
  echo "Warning: $frontend_views not found; skipping frontend distribution figure." >&2
fi
