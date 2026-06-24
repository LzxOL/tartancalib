#!/bin/zsh
set -u

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 TRAIN_TAG TEST_TAG OUTPUT_DIR [--enable-final-ba]" >&2
  exit 2
fi

TRAIN_TAG="$1"
TEST_TAG="$2"
OUTPUT_DIR="$3"
FINAL_BA_MODE="${4:-}"

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

LOG_FILE="${OUTPUT_DIR}/run.log"
PID_FILE="${OUTPUT_DIR}/run.pid"
EXIT_FILE="${OUTPUT_DIR}/run.exit_code"
CACHE_DIR="result/.stage6_stereo_cache_clean_${TRAIN_TAG}_to_${TEST_TAG}"

mkdir -p "${OUTPUT_DIR}"
: > "${LOG_FILE}"
echo "$$" > "${PID_FILE}"
rm -f "${EXIT_FILE}"

EXTRA_ARGS=()
FINAL_STATE="after_incremental_batch_acceptance"
if [ "${FINAL_BA_MODE}" = "--enable-final-ba" ]; then
  EXTRA_ARGS+=(--stage6-enable-final-global-ba)
  FINAL_STATE="after_final_global_ba"
fi

{
  echo "[stage6-incremental-batch-acceptance] start: $(date)"
  echo "[stage6-incremental-batch-acceptance] pid: $$"
  echo "[stage6-incremental-batch-acceptance] train=${TRAIN_TAG} test=${TEST_TAG}"
  echo "[stage6-incremental-batch-acceptance] method=kalibr_style_incremental_batch_acceptance"
  echo "[stage6-incremental-batch-acceptance] final_state=${FINAL_STATE}"
  echo "[stage6-incremental-batch-acceptance] output=${OUTPUT_DIR}"
  echo "[stage6-incremental-batch-acceptance] cache=${CACHE_DIR}"
  ./build/run_stage6_stereo_extrinsic \
    --left-image "image/datatset_5_1/stereo_dataset_20260430_${TRAIN_TAG}/left" \
    --right-image "image/datatset_5_1/stereo_dataset_20260430_${TRAIN_TAG}/right" \
    --test-left-image "image/datatset_5_1/stereo_dataset_20260430_${TEST_TAG}/left" \
    --test-right-image "image/datatset_5_1/stereo_dataset_20260430_${TEST_TAG}/right" \
    --left-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
    --right-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
    --left-intrinsics config/mono_fisheye_calib_3_25_left-camchain.yaml \
    --right-intrinsics config/mono_fisheye_calib_3_25_right-camchain.yaml \
    --stereo-reference-camchain config/stereo_4_2-3-camchain.yaml \
    --output "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --stage6-stereo-measurement-source all_valid \
    --stage6-export-pair-board-consistency-audit \
    --stage6-enable-pair-only-stereo-ba-init \
    --stage6-pair-init-max-iterations 50 \
    --stage6-pair-init-convergence-threshold 1e-6 \
    --stage6-pair-init-use-huber-loss 1 \
    --stage6-enable-committing-pair-batch-selection \
    --stage6-pair-selection-budget-mode kalibr_style \
    --stage6-pairboard-selection-mode kalibr_style_batch \
    --stage6-pair-board-selection-budget-mode kalibr_style \
    --stage6-pair-board-selection-seed-count 50 \
    --stage6-pair-board-selection-max-candidate-additions 40 \
    --stage6-pair-board-selection-min-candidate-score 20 \
    --stage6-pair-board-selection-min-coverage-gain 0 \
    --stage6-pair-board-selection-max-accepted-per-pair 4 \
    --stage6-pair-board-selection-max-accepted-per-board 24 \
    --stage6-enable-pair-cohesion \
    --stage6-pair-cohesion-min-boards-per-pair 2 \
    --stage6-pair-cohesion-max-companions-per-pair 0 \
    --stage6-pair-cohesion-relax-score-gate 1 \
    --stage6-pair-cohesion-relax-cap-gates 1 \
    --stage6-single-board-pair-policy audit \
    --stage6-export-extrinsic-uncertainty-diagnostics \
    --stage6-export-stereo-reprojection-visualizations \
    --stage6-stereo-visualization-top-k 0 \
    "${EXTRA_ARGS[@]}"
  exit_code=$?
  echo "${exit_code}" > "${EXIT_FILE}"
  echo "[stage6-incremental-batch-acceptance] end: $(date)"
  echo "[stage6-incremental-batch-acceptance] exit_code: ${exit_code}"
  exit "${exit_code}"
} >> "${LOG_FILE}" 2>&1
