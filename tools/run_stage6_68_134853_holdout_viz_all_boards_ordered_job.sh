#!/bin/zsh
set -u

if [ "$#" -ne 2 ]; then
  echo "usage: $0 MODE OUTPUT_DIR" >&2
  exit 2
fi

MODE="$1"
OUTPUT_DIR="$2"

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

LOG_FILE="${OUTPUT_DIR}/run.log"
PID_FILE="${OUTPUT_DIR}/run.pid"
EXIT_FILE="${OUTPUT_DIR}/run.exit_code"

mkdir -p "${OUTPUT_DIR}"
: > "${LOG_FILE}"
echo "$$" > "${PID_FILE}"
rm -f "${EXIT_FILE}"

MODE_ARGS=()
case "${MODE}" in
  no_cohesion)
    ;;
  under_target_rescue)
    MODE_ARGS=(
      --stage6-enable-pair-cohesion
      --stage6-pair-cohesion-min-boards-per-pair 2
      --stage6-pair-cohesion-max-companions-per-pair 0
      --stage6-pair-cohesion-relax-score-gate 1
      --stage6-pair-cohesion-relax-cap-gates 1
      --stage6-single-board-pair-policy audit
    )
    ;;
  *)
    echo "unsupported MODE: ${MODE}" >&2
    exit 2
    ;;
esac

{
  echo "[stage6-6.8-holdout-134853-viz-all-boards-ordered] start: $(date)"
  echo "[stage6-6.8-holdout-134853-viz-all-boards-ordered] pid: $$"
  echo "[stage6-6.8-holdout-134853-viz-all-boards-ordered] mode=${MODE}"
  echo "[stage6-6.8-holdout-134853-viz-all-boards-ordered] output=${OUTPUT_DIR}"

  ./build/run_stage6_stereo_extrinsic \
    --left-image image/datatset_5_1/stereo_dataset_20260430_144419/left \
    --right-image image/datatset_5_1/stereo_dataset_20260430_144419/right \
    --test-left-image image/datatset_5_1/stereo_dataset_20260430_134853/left \
    --test-right-image image/datatset_5_1/stereo_dataset_20260430_134853/right \
    --left-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
    --right-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --models ds-none \
  --output "${OUTPUT_DIR}" \
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
    "${MODE_ARGS[@]}"

  exit_code=$?
  echo "${exit_code}" > "${EXIT_FILE}"
  echo "[stage6-6.8-holdout-134853-viz-all-boards-ordered] end: $(date)"
  echo "[stage6-6.8-holdout-134853-viz-all-boards-ordered] exit_code: ${exit_code}"
  exit "${exit_code}"
} >> "${LOG_FILE}" 2>&1
