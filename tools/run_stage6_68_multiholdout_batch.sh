#!/bin/zsh
set -u

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

BATCH_DIR="result_stereo/stage6_68_multiholdout_20260607_144928_batch_seq"
LOG_FILE="${BATCH_DIR}/batch.log"
PID_FILE="${BATCH_DIR}/batch.pid"
EXIT_FILE="${BATCH_DIR}/batch.exit_code"

mkdir -p "${BATCH_DIR}"
: > "${LOG_FILE}"
echo "$$" > "${PID_FILE}"
rm -f "${EXIT_FILE}"

run_one() {
  local test_tag="$1"
  local mode="$2"
  local output_dir="$3"

  echo "[stage6-6.8-batch] run_one start: $(date) test=${test_tag} mode=${mode}"
  zsh tools/run_stage6_68_compare_job.sh \
    144928 "${test_tag}" "${mode}" "${output_dir}"
  local code=$?
  echo "[stage6-6.8-batch] run_one end: $(date) test=${test_tag} mode=${mode} exit_code=${code}"
  return "${code}"
}

{
  echo "[stage6-6.8-batch] start: $(date)"
  echo "[stage6-6.8-batch] pid: $$"

  run_one 134853 no_cohesion \
    result_stereo/stage6_68_multiholdout_seq_20260607_144928_to_134853_no_cohesion
  code=$?
  if [ "${code}" -ne 0 ]; then
    echo "${code}" > "${EXIT_FILE}"
    exit "${code}"
  fi

  run_one 134853 under_target_rescue \
    result_stereo/stage6_68_multiholdout_seq_20260607_144928_to_134853_under_target_rescue
  code=$?
  if [ "${code}" -ne 0 ]; then
    echo "${code}" > "${EXIT_FILE}"
    exit "${code}"
  fi

  run_one 144419 no_cohesion \
    result_stereo/stage6_68_multiholdout_seq_20260607_144928_to_144419_no_cohesion
  code=$?
  if [ "${code}" -ne 0 ]; then
    echo "${code}" > "${EXIT_FILE}"
    exit "${code}"
  fi

  run_one 144419 under_target_rescue \
    result_stereo/stage6_68_multiholdout_seq_20260607_144928_to_144419_under_target_rescue
  code=$?
  if [ "${code}" -ne 0 ]; then
    echo "${code}" > "${EXIT_FILE}"
    exit "${code}"
  fi

  echo 0 > "${EXIT_FILE}"
  echo "[stage6-6.8-batch] end: $(date)"
  echo "[stage6-6.8-batch] exit_code: 0"
  exit 0
} >> "${LOG_FILE}" 2>&1
