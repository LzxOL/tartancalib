#!/usr/bin/env bash
set -euo pipefail

# Run the four post-selection Pixel-Ray objectives under an identical Stage5
# protocol. The persistent selection BA stays pixel-only by design; this
# script changes only the training-only final-refinement objective.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right}"
MODEL="${MODEL:-ds-none}"
CONFIG="${CONFIG:-aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml}"
KALIBR_CAMCHAIN="${KALIBR_CAMCHAIN:-config/mono_fisheye_calib_3_25_right-camchain.yaml}"
CACHE_DIR="${CACHE_DIR:-result/.stage5_baseline_current_20260710_1444190clear_right_ds_topologyfix_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result_may/stage5_pixel_ray_hybrid_modes_$(date +%Y%m%d_%H%M%S)}"
LAMBDA_MIN="${LAMBDA_MIN:-0.2}"
LAMBDA_MAX="${LAMBDA_MAX:-0.8}"
TRANSITION_START_DEG="${TRANSITION_START_DEG:-30}"
TRANSITION_END_DEG="${TRANSITION_END_DEG:-70}"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

common=(
  --config "$CONFIG"
  --runtime-mode research
  --split-mode random_holdout_ratio
  --holdout-ratio 0.30
  --split-seed 1337
  --all
  --stage5-disable-selected-case-visualizations
  --stage5-enable-polar-angle-diagnostics
  --backend-residual-model image_plane
  --stage5-enable-hybrid-pixel-ray-final-refinement
  --stage5-hybrid-pixel-ray-max-iterations 12
  --image "$DATASET"
  --models "$MODEL"
  --kalibr-camchain "$KALIBR_CAMCHAIN"
  --cache-dir "$CACHE_DIR"
)

run_mode() {
  local mode="$1"
  shift
  local out="$OUTPUT_ROOT/$mode"
  mkdir -p "$out"
  ./build/run_stage5_backend "${common[@]}" "$@" --output "$out" \
    > "$out/run.log" 2>&1
}

# Same final-refinement budget across all four rows. lambda=0 and lambda=1
# are exact endpoint controls of the existing 4D Pixel-Ray residual.
run_mode pixel_only --stage5-hybrid-pixel-ray-lambda 0.0
run_mode global_hybrid_lambda_050 --stage5-hybrid-pixel-ray-lambda 0.5
run_mode ray_only --stage5-hybrid-pixel-ray-lambda 1.0
run_mode polar_adaptive \
  --stage5-hybrid-pixel-ray-lambda 0.5 \
  --stage5-hybrid-pixel-ray-polar-adaptive \
  --stage5-hybrid-pixel-ray-lambda-min "$LAMBDA_MIN" \
  --stage5-hybrid-pixel-ray-lambda-max "$LAMBDA_MAX" \
  --stage5-hybrid-pixel-ray-transition-start-deg "$TRANSITION_START_DEG" \
  --stage5-hybrid-pixel-ray-transition-end-deg "$TRANSITION_END_DEG"

summary="$OUTPUT_ROOT/hybrid_mode_heldout_summary.csv"
printf '%s\n' \
  'mode,heldout_multiboard_rmse_px,heldout_angular_rmse_rad,heldout_angular_rmse_deg' \
  > "$summary"
for mode in pixel_only global_hybrid_lambda_050 ray_only polar_adaptive; do
  holdout="$OUTPUT_ROOT/$mode/backend_holdout_summary.txt"
  pixel="$(awk '/^overall_rmse:/ {print $2}' "$holdout")"
  angular_rad="$(awk '/^overall_angular_rmse_rad:/ {print $2}' "$holdout")"
  angular_deg="$(awk '/^overall_angular_rmse_deg:/ {print $2}' "$holdout")"
  printf '%s,%s,%s,%s\n' "$mode" "$pixel" "$angular_rad" "$angular_deg" \
    >> "$summary"
done

printf 'Results: %s\n' "$summary"
