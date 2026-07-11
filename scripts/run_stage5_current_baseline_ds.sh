#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_DIR="${IMAGE_DIR:-image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right}"
MODEL="${MODEL:-ds-none}"
KALIBR_CAMCHAIN="${KALIBR_CAMCHAIN:-config/mono_fisheye_calib_3_25_right-camchain.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-result_may/stage5_baseline_current_1444190clear_right_ds}"
CACHE_DIR="${CACHE_DIR:-result/.stage5_baseline_current_1444190clear_right_ds_cache}"

./build/run_stage5_backend \
  --config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --runtime-mode research \
  --split-mode random_holdout_ratio \
  --holdout-ratio 0.30 \
  --split-seed 1337 \
  --all \
  --stage5-disable-selected-case-visualizations \
  --stage5-enable-polar-angle-diagnostics \
  --image "$IMAGE_DIR" \
  --models "$MODEL" \
  --kalibr-camchain "$KALIBR_CAMCHAIN" \
  --output "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR"
