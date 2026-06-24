#!/bin/zsh
set -u

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

OUTPUT_DIR="result_stereo/stage6_68_train_134853_viz_all_training_20260608_to_144419_under_target_rescue"
LOG_FILE="${OUTPUT_DIR}/run.log"
PID_FILE="${OUTPUT_DIR}/run.pid"
EXIT_FILE="${OUTPUT_DIR}/run.exit_code"

mkdir -p "${OUTPUT_DIR}"
: > "${LOG_FILE}"
echo "$$" > "${PID_FILE}"
rm -f "${EXIT_FILE}"

{
  echo "[stage6-6.8-train-134853-viz-all-training] start: $(date)"
  echo "[stage6-6.8-train-134853-viz-all-training] pid: $$"
  echo "[stage6-6.8-train-134853-viz-all-training] train=134853 test=144419 mode=under_target_rescue"
  echo "[stage6-6.8-train-134853-viz-all-training] output=${OUTPUT_DIR}"

  ./build/run_stage6_stereo_extrinsic \
    --left-image image/datatset_5_1/stereo_dataset_20260430_134853/left \
    --right-image image/datatset_5_1/stereo_dataset_20260430_134853/right \
    --test-left-image image/datatset_5_1/stereo_dataset_20260430_144419/left \
    --test-right-image image/datatset_5_1/stereo_dataset_20260430_144419/right \
    --left-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
    --right-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
    --left-intrinsics config/mono_fisheye_calib_3_25_left-camchain.yaml \
    --right-intrinsics config/mono_fisheye_calib_3_25_right-camchain.yaml \
    --stereo-reference-camchain config/stereo_4_2-3-camchain.yaml \
    --output "${OUTPUT_DIR}" \
    --cache-dir result/.stage6_stereo_cache_clean_134853_to_144419 \
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
    --stage6-enable-pair-cohesion \
    --stage6-pair-cohesion-min-boards-per-pair 2 \
    --stage6-pair-cohesion-max-companions-per-pair 0 \
    --stage6-pair-cohesion-relax-score-gate 1 \
    --stage6-pair-cohesion-relax-cap-gates 1 \
    --stage6-single-board-pair-policy audit

  exit_code=$?
  echo "${exit_code}" > "${EXIT_FILE}"
  echo "[stage6-6.8-train-134853-viz-all-training] end: $(date)"
  echo "[stage6-6.8-train-134853-viz-all-training] exit_code: ${exit_code}"
  exit "${exit_code}"
} >> "${LOG_FILE}" 2>&1
