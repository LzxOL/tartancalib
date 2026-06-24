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
  echo "[stage5-trial-batch-path] start: $(date)"
  echo "[stage5-trial-batch-path] pid: $$"
  echo "[stage5-trial-batch-path] train=${TRAIN_IMAGE_DIR}"
  echo "[stage5-trial-batch-path] test=${TEST_IMAGE_DIR}"
  echo "[stage5-trial-batch-path] mode=kalibr_style_batch"
  echo "[stage5-trial-batch-path] output=${OUTPUT_DIR}"
  ./build/run_stage5_backend \
    --image "${TRAIN_IMAGE_DIR}" \
    --test-image "${TEST_IMAGE_DIR}" \
    --config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
    --kalibr-camchain config/mono_fisheye_calib_3_25_right-camchain.yaml \
    --output "${OUTPUT_DIR}" \
    --runtime-mode research \
    --cache-dir result/.stage5_backend_cache \
    --all
  exit_code=$?
  echo "${exit_code}" > "${EXIT_FILE}"
  echo "[stage5-trial-batch-path] end: $(date)"
  echo "[stage5-trial-batch-path] exit_code: ${exit_code}"
  exit "${exit_code}"
} >> "${LOG_FILE}" 2>&1
