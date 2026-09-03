#!/bin/zsh
set -u

if [ "$#" -ne 3 ]; then
  echo "usage: $0 TRAIN_TAG TEST_TAG OUTPUT_DIR" >&2
  exit 2
fi

TRAIN_TAG="$1"
TEST_TAG="$2"
OUTPUT_DIR="$3"

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

LOG_FILE="${OUTPUT_DIR}/run.log"
PID_FILE="${OUTPUT_DIR}/run.pid"
EXIT_FILE="${OUTPUT_DIR}/run.exit_code"
CACHE_DIR="result/.stage6_stereo_cache_clean_${TRAIN_TAG}_to_${TEST_TAG}"

mkdir -p "${OUTPUT_DIR}"
: > "${LOG_FILE}"
echo "$$" > "${PID_FILE}"
rm -f "${EXIT_FILE}"

{
  echo "[stage6-valid-traversal-viz] start: $(date)"
  echo "[stage6-valid-traversal-viz] pid: $$"
  echo "[stage6-valid-traversal-viz] train=${TRAIN_TAG} test=${TEST_TAG}"
  echo "[stage6-valid-traversal-viz] mode=baseline: kalibr_style_batch valid_traversal pair_cohesion residual_score"
  echo "[stage6-valid-traversal-viz] output=${OUTPUT_DIR}"
  echo "[stage6-valid-traversal-viz] cache=${CACHE_DIR}"
  ./build/run_stage6_stereo_extrinsic \
    --left-image "image/datatset_5_1/stereo_dataset_20260430_${TRAIN_TAG}/left" \
    --right-image "image/datatset_5_1/stereo_dataset_20260430_${TRAIN_TAG}/right" \
    --test-left-image "image/datatset_5_1/stereo_dataset_20260430_${TEST_TAG}/left" \
    --test-right-image "image/datatset_5_1/stereo_dataset_20260430_${TEST_TAG}/right" \
    --left-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
    --right-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --models ds-none \
  --output "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --stage6-stereo-measurement-source all_valid \
    --stage6-export-pair-board-consistency-audit \
    --stage6-enable-pair-only-stereo-ba-init \
    --stage6-pair-init-max-iterations 50 \
    --stage6-pair-init-convergence-threshold 1e-6 \
    --stage6-pair-init-use-huber-loss 1 \
    --stage6-enable-kalibr-style-pair-selection \
    --stage6-batch-acceptance-policy residual_score \
    --stage6-pair-selection-budget-mode kalibr_style \
    --stage6-enable-pair-board-trial-selection \
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
    --stage6-stereo-visualization-top-k 0
  exit_code=$?
  echo "${exit_code}" > "${EXIT_FILE}"
  echo "[stage6-valid-traversal-viz] end: $(date)"
  echo "[stage6-valid-traversal-viz] exit_code: ${exit_code}"
  exit "${exit_code}"
} >> "${LOG_FILE}" 2>&1
