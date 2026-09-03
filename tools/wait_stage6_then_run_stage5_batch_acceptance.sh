#!/bin/zsh
set -u

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

WATCH_LOG="result_may/stage5_trial_batch_acceptance_20260608_134853_right_val_144419_after_stage6/wait_stage6_then_run.log"
STAGE5_OUTPUT="result_may/stage5_trial_batch_acceptance_20260608_134853_right_val_144419_after_stage6"
STAGE6_DIRS=(
  result_stereo/stage6_pairboard_batch_acceptance_20260608_134853_to_144928_under_target_rescue
  result_stereo/stage6_pairboard_batch_acceptance_20260608_144419_to_134853_under_target_rescue
  result_stereo/stage6_pairboard_batch_acceptance_20260608_144928_to_144419_under_target_rescue
)

mkdir -p "${STAGE5_OUTPUT}"
: > "${WATCH_LOG}"

{
  echo "[wait-stage6-then-stage5] start: $(date)"
  while true; do
    all_done=1
    for dir in "${STAGE6_DIRS[@]}"; do
      exit_file="${dir}/run.exit_code"
      if [ ! -f "${exit_file}" ]; then
        all_done=0
        break
      fi
      exit_code="$(cat "${exit_file}")"
      if [ "${exit_code}" != "0" ]; then
        echo "[wait-stage6-then-stage5] stage6 failed: ${dir} exit_code=${exit_code}"
        echo "[wait-stage6-then-stage5] not starting stage5"
        exit 1
      fi
    done

    if [ "${all_done}" = "1" ]; then
      echo "[wait-stage6-then-stage5] all stage6 jobs finished successfully: $(date)"
      break
    fi

    sleep 60
  done

  echo "[wait-stage6-then-stage5] launching stage5: ${STAGE5_OUTPUT}"
  /bin/zsh tools/run_stage5_trial_batch_acceptance_job.sh \
    134853 \
    144419 \
    "${STAGE5_OUTPUT}"
  stage5_exit=$?
  echo "[wait-stage6-then-stage5] stage5 exit_code: ${stage5_exit}"
  echo "[wait-stage6-then-stage5] end: $(date)"
  exit "${stage5_exit}"
} >> "${WATCH_LOG}" 2>&1
