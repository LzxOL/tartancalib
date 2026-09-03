#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_DIR="${IMAGE_DIR:-image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right}"
MODEL="${MODEL:-ds-none}"
KALIBR_CAMCHAIN="${KALIBR_CAMCHAIN:-config/mono_fisheye_calib_3_25_right-camchain.yaml}"
RESIDUAL_MODEL="${RESIDUAL_MODEL:-pixel_only}"
SPLIT_SEED="${SPLIT_SEED:-1337}"
CALIBRATION_SCOPE="${CALIBRATION_SCOPE:-random_holdout}"
HYBRID_THRESHOLD_DEG="${HYBRID_THRESHOLD_DEG:-50}"
HYBRID_TEMPERATURE_DEG="${HYBRID_TEMPERATURE_DEG:-10}"
CAMERA_AWARE_OUTER_RESCUE="${CAMERA_AWARE_OUTER_RESCUE:-1}"

case "$RESIDUAL_MODEL" in
  pixel|pixel_only)
    RESIDUAL_MODEL="pixel_only"
    ;;
  angular|sphere_angular)
    RESIDUAL_MODEL="sphere_angular"
    ;;
  hybrid|pixel_ray_hybrid)
    RESIDUAL_MODEL="pixel_ray_hybrid"
    ;;
  polar_continuous_hybrid)
    ;;
  *)
    echo "Unsupported baseline residual model: $RESIDUAL_MODEL" >&2
    echo "Expected pixel/pixel_only, angular/sphere_angular, hybrid/pixel_ray_hybrid, or polar_continuous_hybrid." >&2
    exit 2
    ;;
esac

BACKEND_RESIDUAL_MODEL="$RESIDUAL_MODEL"
HYBRID_REFINEMENT_ARGS=()
if [[ "$RESIDUAL_MODEL" == "pixel_ray_hybrid" ]]; then
  BACKEND_RESIDUAL_MODEL="image_plane"
  HYBRID_REFINEMENT_ARGS=(
    --stage5-enable-hybrid-pixel-ray-final-refinement
    --stage5-hybrid-pixel-ray-lambda "${HYBRID_PIXEL_RAY_LAMBDA:-0.5}"
    --stage5-hybrid-pixel-ray-max-iterations "${HYBRID_PIXEL_RAY_MAX_ITERATIONS:-12}"
  )
fi

case "$CAMERA_AWARE_OUTER_RESCUE" in
  1|true|on)
    OUTER_RESCUE_ARGS=(--stage5-enable-camera-aware-outer-rescue)
    ;;
  0|false|off)
    OUTER_RESCUE_ARGS=(--stage5-disable-camera-aware-outer-rescue)
    ;;
  *)
    echo "Unsupported CAMERA_AWARE_OUTER_RESCUE: $CAMERA_AWARE_OUTER_RESCUE" >&2
    echo "Expected 1/true/on or 0/false/off." >&2
    exit 2
    ;;
esac

case "$CALIBRATION_SCOPE" in
  random_holdout)
    SPLIT_ARGS=(
      --split-mode random_holdout_ratio
      --holdout-ratio 0.30
      --split-seed "$SPLIT_SEED"
    )
    ;;
  full_dataset)
    # Stage5 expects an evaluation split. Reusing the same image directory as
    # a frozen frontend prepass keeps every source frame in calibration while
    # providing diagnostics that cannot affect selection or optimization.
    SPLIT_ARGS=(
      --test-image "$IMAGE_DIR"
      --stage5-external-holdout-self-frontend-prepass
    )
    ;;
  *)
    echo "Unsupported CALIBRATION_SCOPE: $CALIBRATION_SCOPE" >&2
    echo "Expected random_holdout or full_dataset." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-result_may/stage5_baseline_current_camera_aware_outer_v2_1444190clear_right_ds_${RESIDUAL_MODEL}}"
CACHE_DIR="${CACHE_DIR:-result/.stage5_baseline_current_camera_aware_outer_v2_1444190clear_right_ds_${RESIDUAL_MODEL}_cache}"

./build/run_stage5_backend \
  --config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --runtime-mode research \
  "${SPLIT_ARGS[@]}" \
  --all \
  "${OUTER_RESCUE_ARGS[@]}" \
  --stage5-disable-selected-case-visualizations \
  --stage5-enable-global-scene-state-consistency-audit \
  --stage5-enable-polar-angle-diagnostics \
  --image "$IMAGE_DIR" \
  --models "$MODEL" \
  --kalibr-camchain "$KALIBR_CAMCHAIN" \
  --backend-residual-model "$BACKEND_RESIDUAL_MODEL" \
  --backend-polar-continuous-hybrid-threshold-deg "$HYBRID_THRESHOLD_DEG" \
  --backend-polar-continuous-hybrid-temperature-deg "$HYBRID_TEMPERATURE_DEG" \
  "${HYBRID_REFINEMENT_ARGS[@]}" \
  --output "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR"
