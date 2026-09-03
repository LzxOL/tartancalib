#!/usr/bin/env bash
set -euo pipefail

# Reproduce the frozen Pixel/Angular/Hybrid baseline suite. The three modes
# are separate runs over the same fixed backend manifest; they are never mixed
# into one optimization problem.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_DIR="${IMAGE_DIR:-image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right}"
DEFAULT_MANIFEST="paper/experiments/fixed_backend_residual_ablation_seed1337/manifests/1444190clear_pixel_backend.csv"
if [[ -n "${MANIFEST:-}" ]]; then
  MANIFEST_SET=1
else
  MANIFEST="$DEFAULT_MANIFEST"
  MANIFEST_SET=0
fi
PROFILE="${PROFILE:-all}"
SPLIT_SEED="${SPLIT_SEED:-1337}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result_may/stage5_frozen_residual_baseline_$(basename "$(dirname "$IMAGE_DIR")")_seed${SPLIT_SEED}}"

if [[ "$IMAGE_DIR" != *"1444190-clear/right"* && "$MANIFEST_SET" -eq 0 ]]; then
  echo "IMAGE_DIR is not the frozen 1444190-clear/right dataset." >&2
  echo "Set MANIFEST explicitly for another dataset." >&2
  exit 2
fi

case "$PROFILE" in
  pixel|angular|hybrid|all)
    ;;
  *)
    echo "Unsupported PROFILE: $PROFILE" >&2
    echo "Expected pixel, angular, hybrid, or all." >&2
    exit 2
    ;;
esac

run_profile() {
  local profile="$1"
  local model
  case "$profile" in
    pixel) model=pixel_only ;;
    angular) model=sphere_angular ;;
    hybrid) model=pixel_ray_hybrid ;;
  esac

  local output_dir="${OUTPUT_ROOT}/${profile}"
  local cache_dir="result/.stage5_frozen_residual_baseline_${profile}_$(basename "$(dirname "$IMAGE_DIR")")_seed${SPLIT_SEED}_cache"
  SPLIT_SEED="$SPLIT_SEED" \
    CACHE_DIR="$cache_dir" \
    HYBRID_PIXEL_RAY_LAMBDA="${HYBRID_PIXEL_RAY_LAMBDA:-0.5}" \
    HYBRID_PIXEL_RAY_MAX_ITERATIONS="${HYBRID_PIXEL_RAY_MAX_ITERATIONS:-12}" \
    scripts/run_stage5_fixed_backend_residual_ablation.sh \
      "$IMAGE_DIR" "$MANIFEST" "$model" "$output_dir"
}

case "$PROFILE" in
  all)
    run_profile pixel
    run_profile angular
    run_profile hybrid
    ;;
  *)
    run_profile "$PROFILE"
    ;;
esac
