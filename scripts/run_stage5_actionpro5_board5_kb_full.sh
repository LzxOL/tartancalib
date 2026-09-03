#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_DIR="${IMAGE_DIR:-image/Action Pro5/dataset-8-15-mono8/seq_fov155_mono8_curated_exclude15_20260819}"
CONFIG="${CONFIG:-aslam_cv/aslam_cameras_april/config/actionpro5_fov155_board5_apriltag_internal.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-result_may/stage5_actionpro5_board5_curated92_kb_full}"
CACHE_DIR="${CACHE_DIR:-result/.stage5_actionpro5_board5_curated92_kb_full_cache}"

# Stage5 currently requires a Kalibr camchain for comparison output. This file
# is not an Action Pro calibration and is not used by automatic initialization
# or BA. Do not interpret backend_vs_kalibr as an Action Pro benchmark.
KALIBR_COMPARISON_CAMCHAIN="${KALIBR_COMPARISON_CAMCHAIN:-config/mono_kb_calib_3_25_right-camchain.yaml}"

./build/run_stage5_backend \
  --image "$IMAGE_DIR" \
  --test-image "$IMAGE_DIR" \
  --stage5-external-holdout-self-frontend-prepass \
  --config "$CONFIG" \
  --output "$OUTPUT_DIR" \
  --kalibr-camchain "$KALIBR_COMPARISON_CAMCHAIN" \
  --models pinhole-equi \
  --runtime-mode research \
  --stage5-enable-frozen-recovery-baseline \
  --stage5-enable-two-layer-ba \
  --stage5-enable-global-scene-state-consistency-audit \
  --stage5-enable-polar-angle-diagnostics \
  --stage5-skip-heavy-overlays \
  --stage5-enable-progress \
  --stage5-progress-interval 10 \
  --cache-dir "$CACHE_DIR" \
  --all
