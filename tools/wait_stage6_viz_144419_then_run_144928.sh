#!/bin/zsh
set -u

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

FIRST_DIR="result_stereo/stage6_valid_traversal_viz_20260609_144419_to_134853_kalibr_style_batch"
SECOND_DIR="result_stereo/stage6_valid_traversal_viz_20260609_144419_to_144928_kalibr_style_batch"
LOG_FILE="result_stereo/stage6_valid_traversal_viz_20260609_144419_wait_then_144928.log"

{
  echo "[wait-stage6-viz] start: $(date)"
  echo "[wait-stage6-viz] waiting for ${FIRST_DIR}/run.exit_code"
  while [ ! -f "${FIRST_DIR}/run.exit_code" ]; do
    sleep 30
  done
  first_exit="$(cat "${FIRST_DIR}/run.exit_code")"
  echo "[wait-stage6-viz] first_exit=${first_exit} at $(date)"
  if [ "${first_exit}" != "0" ]; then
    echo "[wait-stage6-viz] first run failed; not starting second run"
    exit 1
  fi
  echo "[wait-stage6-viz] starting second run: 144419 -> 144928"
  /bin/zsh tools/run_stage6_valid_traversal_viz_job.sh \
    144419 144928 "${SECOND_DIR}"
  second_exit=$?
  echo "[wait-stage6-viz] second_exit=${second_exit} at $(date)"
  exit "${second_exit}"
} >> "${LOG_FILE}" 2>&1
