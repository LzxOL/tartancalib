#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASET="${1:-image/datatset_5_1/stereo_dataset_20260430_1444190-clear}"
OUTPUT_DIR="${2:-result_may/stage6_current_baseline_ds}"
HOLDOUT_OFFSET="${3:-0}"
CACHE_DIR="${STAGE6_CACHE_DIR:-result/.stage6_current_baseline_cache}"

cmake --build build --target run_stage6_stereo_extrinsic -j 8

./build/run_stage6_stereo_extrinsic \
  --left-image "${DATASET}/left" \
  --right-image "${DATASET}/right" \
  --left-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --right-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --models ds-none \
  --output "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --stage6-stereo-measurement-source all_valid \
  --stage6-frame-pairing-mode exact_timestamp \
  --holdout-stride 3 \
  --holdout-offset "${HOLDOUT_OFFSET}" \
  --stage6-solver-mode global_sparse_ba \
  --stage6-intrinsics-mode adaptive_regularized_joint_projection \
  --stage6-persistent-pose-structure independent_pair_board \
  --stage6-selection-ba-residual-mode pixel \
  --stage6-skip-final-global-ba \
  --stage6-enable-persistent-incremental-stereo-ba

python3 tools/verify_stage6_persistent_outputs.py \
  --expected-pose-structure independent_pair_board \
  --write-report \
  "${OUTPUT_DIR}"
