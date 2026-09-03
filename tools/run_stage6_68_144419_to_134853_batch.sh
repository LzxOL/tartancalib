#!/bin/zsh
set -u

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

BATCH_DIR="result_stereo/stage6_68_next_20260608_144419_to_134853_batch_seq"
LOG_FILE="${BATCH_DIR}/batch.log"
PID_FILE="${BATCH_DIR}/batch.pid"
EXIT_FILE="${BATCH_DIR}/batch.exit_code"

mkdir -p "${BATCH_DIR}"
: > "${LOG_FILE}"
echo "$$" > "${PID_FILE}"
rm -f "${EXIT_FILE}"

run_one() {
  local mode="$1"
  local output_dir="$2"

  echo "[stage6-6.8-144419-to-134853] run_one start: $(date) mode=${mode}"
  zsh tools/run_stage6_68_compare_job.sh \
    144419 134853 "${mode}" "${output_dir}"
  local code=$?
  echo "[stage6-6.8-144419-to-134853] run_one end: $(date) mode=${mode} exit_code=${code}"
  return "${code}"
}

{
  echo "[stage6-6.8-144419-to-134853] start: $(date)"
  echo "[stage6-6.8-144419-to-134853] pid: $$"

  run_one no_cohesion \
    result_stereo/stage6_68_pairwise_20260608_144419_to_134853_no_cohesion
  code=$?
  if [ "${code}" -ne 0 ]; then
    echo "${code}" > "${EXIT_FILE}"
    exit "${code}"
  fi

  run_one under_target_rescue \
    result_stereo/stage6_68_pairwise_20260608_144419_to_134853_under_target_rescue
  code=$?
  if [ "${code}" -ne 0 ]; then
    echo "${code}" > "${EXIT_FILE}"
    exit "${code}"
  fi

  echo 0 > "${EXIT_FILE}"
  echo "[stage6-6.8-144419-to-134853] end: $(date)"
  echo "[stage6-6.8-144419-to-134853] exit_code: 0"
  exit 0
} >> "${LOG_FILE}" 2>&1
