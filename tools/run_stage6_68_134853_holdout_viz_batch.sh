#!/bin/zsh
set -u

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

BATCH_DIR="result_stereo/stage6_68_holdout_134853_viz_20260608_batch_seq"
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
  local -a mode_args
  mode_args=()
  if [ "${mode}" = "under_target_rescue" ]; then
    mode_args=(
      --stage6-enable-pair-cohesion
      --stage6-pair-cohesion-min-boards-per-pair 2
      --stage6-pair-cohesion-max-companions-per-pair 0
      --stage6-pair-cohesion-relax-score-gate 1
      --stage6-pair-cohesion-relax-cap-gates 1
      --stage6-single-board-pair-policy audit
    )
  fi

  mkdir -p "${output_dir}"
  echo "[stage6-6.8-holdout-134853-viz] run_one start: $(date) mode=${mode}"
  ./build/run_stage6_stereo_extrinsic \
    --left-image image/datatset_5_1/stereo_dataset_20260430_144419/left \
    --right-image image/datatset_5_1/stereo_dataset_20260430_144419/right \
    --test-left-image image/datatset_5_1/stereo_dataset_20260430_134853/left \
    --test-right-image image/datatset_5_1/stereo_dataset_20260430_134853/right \
    --left-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
    --right-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --models ds-none \
  --output "${output_dir}" \
    --cache-dir result/.stage6_stereo_cache_clean_144419_to_134853 \
    --stage6-stereo-measurement-source all_valid \
    --stage6-export-pair-board-consistency-audit \
    --stage6-enable-pair-only-stereo-ba-init \
    --stage6-pair-init-max-iterations 50 \
    --stage6-pair-init-convergence-threshold 1e-6 \
    --stage6-pair-init-use-huber-loss 1 \
    --stage6-enable-kalibr-style-pair-selection \
    --stage6-enable-pair-board-trial-selection \
    --stage6-pair-board-selection-seed-count 50 \
    --stage6-pair-board-selection-max-candidate-additions 40 \
    --stage6-pair-board-selection-min-candidate-score 20 \
    --stage6-pair-board-selection-min-coverage-gain 0 \
    --stage6-pair-board-selection-max-accepted-per-pair 4 \
    --stage6-pair-board-selection-max-accepted-per-board 24 \
    --stage6-export-extrinsic-uncertainty-diagnostics \
    --stage6-export-stereo-reprojection-visualizations \
    --stage6-stereo-visualization-top-k 0 \
    "${mode_args[@]}"
  local code=$?
  echo "[stage6-6.8-holdout-134853-viz] run_one end: $(date) mode=${mode} exit_code=${code}"
  return "${code}"
}

{
  echo "[stage6-6.8-holdout-134853-viz] start: $(date)"
  echo "[stage6-6.8-holdout-134853-viz] pid: $$"

  run_one no_cohesion \
    result_stereo/stage6_68_holdout_134853_viz_20260608_144419_to_134853_no_cohesion
  code=$?
  if [ "${code}" -ne 0 ]; then
    echo "${code}" > "${EXIT_FILE}"
    exit "${code}"
  fi

  run_one under_target_rescue \
    result_stereo/stage6_68_holdout_134853_viz_20260608_144419_to_134853_under_target_rescue
  code=$?
  if [ "${code}" -ne 0 ]; then
    echo "${code}" > "${EXIT_FILE}"
    exit "${code}"
  fi

  echo 0 > "${EXIT_FILE}"
  echo "[stage6-6.8-holdout-134853-viz] end: $(date)"
  echo "[stage6-6.8-holdout-134853-viz] exit_code: 0"
  exit 0
} >> "${LOG_FILE}" 2>&1
