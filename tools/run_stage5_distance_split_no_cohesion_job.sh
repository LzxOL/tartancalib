#!/bin/zsh
set -u

if [ "$#" -ne 3 ]; then
  echo "usage: $0 TRAIN_IMAGE_DIR TEST_IMAGE_DIR OUTPUT_DIR" >&2
  exit 2
fi

TRAIN_IMAGE_DIR="$1"
TEST_IMAGE_DIR="$2"
OUTPUT_DIR="$3"

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

LOG_FILE="${OUTPUT_DIR}/run.log"
PID_FILE="${OUTPUT_DIR}/run.pid"
EXIT_FILE="${OUTPUT_DIR}/run.exit_code"

mkdir -p "${OUTPUT_DIR}"
: > "${LOG_FILE}"
echo "$$" > "${PID_FILE}"
rm -f "${EXIT_FILE}"

{
  echo "[stage5-no-cohesion-path] start: $(date)"
  echo "[stage5-no-cohesion-path] pid: $$"
  echo "[stage5-no-cohesion-path] train=${TRAIN_IMAGE_DIR}"
  echo "[stage5-no-cohesion-path] test=${TEST_IMAGE_DIR}"
  echo "[stage5-no-cohesion-path] mode=strict_rmse_no_cohesion"
  echo "[stage5-no-cohesion-path] output=${OUTPUT_DIR}"
  ./build/run_stage5_backend \
    --image "${TRAIN_IMAGE_DIR}" \
    --test-image "${TEST_IMAGE_DIR}" \
    --config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
    --kalibr-camchain config/mono_fisheye_calib_3_25_right-camchain.yaml \
    --output "${OUTPUT_DIR}" \
    --runtime-mode research \
    --cache-dir result/.stage5_backend_cache \
    --all \
    --stage5-enable-trial-backend-frame-board-selection \
    --stage5-trial-backend-selection-mode strict_rmse \
    --stage5-trial-backend-selection-budget-mode fixed \
    --stage5-trial-backend-selection-incremental 1 \
    --stage5-trial-backend-selection-max-iterations 5 \
    --stage5-trial-backend-selection-max-candidate-additions 20 \
    --stage5-trial-backend-selection-min-candidate-score 3.2 \
    --stage5-trial-backend-selection-min-coverage-gain 2.5 \
    --stage5-trial-backend-selection-max-accepted-per-board 4 \
    --stage5-trial-backend-selection-max-accepted-per-frame 1
  exit_code=$?
  echo "${exit_code}" > "${EXIT_FILE}"
  echo "[stage5-no-cohesion-path] end: $(date)"
  echo "[stage5-no-cohesion-path] exit_code: ${exit_code}"
  exit "${exit_code}"
} >> "${LOG_FILE}" 2>&1
