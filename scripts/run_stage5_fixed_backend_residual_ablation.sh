#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 IMAGE_DIR MANIFEST RESIDUAL_MODEL OUTPUT_DIR [EXTRA_ARGS...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_DIR="$1"
MANIFEST="$2"
RESIDUAL_MODEL="$3"
OUTPUT_DIR="$4"
shift 4
SPLIT_SEED="${SPLIT_SEED:-1337}"
CACHE_DIR="${CACHE_DIR:-result/.fixed_backend_residual_ablation_$(basename "$IMAGE_DIR")_seed${SPLIT_SEED}_cache}"
TEST_IMAGE_DIR="${TEST_IMAGE_DIR:-}"

case "$RESIDUAL_MODEL" in
  pixel_only|sphere_angular|polar_continuous_hybrid|pixel_ray_hybrid) ;;
  *) echo "Unsupported residual model: $RESIDUAL_MODEL" >&2; exit 2 ;;
esac

BACKEND_RESIDUAL_MODEL="$RESIDUAL_MODEL"
if [[ "$RESIDUAL_MODEL" == "pixel_ray_hybrid" ]]; then
  BACKEND_RESIDUAL_MODEL="image_plane"
fi

COMMAND=(./build/run_stage5_backend \
  --config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --runtime-mode research \
  --split-mode random_holdout_ratio \
  --holdout-ratio 0.30 \
  --split-seed "$SPLIT_SEED" \
  --all \
  --stage5-disable-selected-case-visualizations \
  --stage5-enable-polar-angle-diagnostics \
  --image "$IMAGE_DIR")

if [[ -n "$TEST_IMAGE_DIR" ]]; then
  COMMAND+=(
    --test-image "$TEST_IMAGE_DIR"
    --stage5-external-holdout-self-frontend-prepass
  )
fi

COMMAND+=( \
  --models ds-none \
  --kalibr-camchain config/mono_fisheye_calib_3_25_right-camchain.yaml \
  --backend-residual-model "$BACKEND_RESIDUAL_MODEL" \
  --backend-polar-continuous-hybrid-threshold-deg 50 \
  --backend-polar-continuous-hybrid-temperature-deg 10 \
  --stage5-trial-backend-selection-force-include-frame-board-list "$MANIFEST" \
  --stage5-trial-backend-selection-force-include-list-is-exact-input 1 \
  --output "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR")

if [[ "$RESIDUAL_MODEL" == "pixel_ray_hybrid" ]]; then
  COMMAND+=(
    --stage5-enable-hybrid-pixel-ray-final-refinement
    --stage5-hybrid-pixel-ray-lambda "${HYBRID_PIXEL_RAY_LAMBDA:-0.5}"
    --stage5-hybrid-pixel-ray-max-iterations "${HYBRID_PIXEL_RAY_MAX_ITERATIONS:-12}"
  )
fi

if [[ $# -gt 0 ]]; then
  COMMAND+=("$@")
fi

"${COMMAND[@]}"
