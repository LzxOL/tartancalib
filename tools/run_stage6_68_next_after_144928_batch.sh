#!/bin/zsh
set -u

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

WAIT_BATCH_DIR="result_stereo/stage6_68_multiholdout_20260607_144928_batch_seq"
NEXT_BATCH_DIR="result_stereo/stage6_68_next_20260607_134853_to_144928_batch_seq"
LOG_FILE="${NEXT_BATCH_DIR}/batch.log"
PID_FILE="${NEXT_BATCH_DIR}/batch.pid"
EXIT_FILE="${NEXT_BATCH_DIR}/batch.exit_code"

mkdir -p "${NEXT_BATCH_DIR}"
: > "${LOG_FILE}"
echo "$$" > "${PID_FILE}"
rm -f "${EXIT_FILE}"

wait_for_previous_batch() {
  while [ ! -f "${WAIT_BATCH_DIR}/batch.exit_code" ]; do
    sleep 60
  done
  local previous_code
  previous_code="$(cat "${WAIT_BATCH_DIR}/batch.exit_code")"
  if [ "${previous_code}" != "0" ]; then
    echo "[stage6-6.8-next] previous batch failed exit_code=${previous_code}"
    echo "${previous_code}" > "${EXIT_FILE}"
    exit "${previous_code}"
  fi
}

run_one() {
  local mode="$1"
  local output_dir="$2"

  echo "[stage6-6.8-next] run_one start: $(date) train=134853 test=144928 mode=${mode}"
  zsh tools/run_stage6_68_compare_job.sh \
    134853 144928 "${mode}" "${output_dir}"
  local code=$?
  echo "[stage6-6.8-next] run_one end: $(date) train=134853 test=144928 mode=${mode} exit_code=${code}"
  return "${code}"
}

{
  echo "[stage6-6.8-next] start: $(date)"
  echo "[stage6-6.8-next] pid: $$"
  echo "[stage6-6.8-next] waiting_for=${WAIT_BATCH_DIR}/batch.exit_code"
  wait_for_previous_batch

  run_one no_cohesion \
    result_stereo/stage6_68_pairwise_20260607_134853_to_144928_no_cohesion
  code=$?
  if [ "${code}" -ne 0 ]; then
    echo "${code}" > "${EXIT_FILE}"
    exit "${code}"
  fi

  run_one under_target_rescue \
    result_stereo/stage6_68_pairwise_20260607_134853_to_144928_under_target_rescue
  code=$?
  if [ "${code}" -ne 0 ]; then
    echo "${code}" > "${EXIT_FILE}"
    exit "${code}"
  fi

  echo 0 > "${EXIT_FILE}"
  echo "[stage6-6.8-next] end: $(date)"
  echo "[stage6-6.8-next] exit_code: 0"
  exit 0
} >> "${LOG_FILE}" 2>&1
