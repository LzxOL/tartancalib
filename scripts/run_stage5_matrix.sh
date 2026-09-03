#!/bin/bash
set -euo pipefail

ROOT_DIR="/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib"
cd "$ROOT_DIR"

TRAIN_1="image/datatset_5_1/stereo_dataset_20260430_134853-clear/right"
TRAIN_2="image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right"
TRAIN_3="image/datatset_5_1/stereo_dataset_20260430_144928-clear/right"
TEST_1="image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right"
TEST_2="image/datatset_5_1/stereo_dataset_20260430_144928-clear/right"
TEST_3="image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right"

DS_CAMCHAIN="config/mono_fisheye_calib_3_25_right-camchain.yaml"
KB_CAMCHAIN="config/mono_kb_calib_3_25_right-camchain.yaml"
CONFIG="aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
CACHE_DIR="result/.stage5_backend_cache"
LOG_DIR="result_may/stage5_matrix_logs"
REPORT_DIR="result_may/stage5_matrix_reports"

mkdir -p "$LOG_DIR" "$REPORT_DIR"

run_one() {
  local model_label="$1"
  local camchain="$2"
  local train_dir="$3"
  local test_dir="$4"
  local model_family="ds-none"
  if [[ "$model_label" == "kb" ]]; then
    model_family="pinhole-equi"
  fi
  local out_dir="result_may/stage5_matrix_${model_label}_$(basename "$(dirname "$train_dir")")_to_$(basename "$(dirname "$test_dir")")"
  local log_file="$LOG_DIR/$(basename "$out_dir").log"

  if [[ -f "$out_dir/backend_training_summary.txt" &&
        -f "$out_dir/backend_holdout_summary.txt" &&
        -f "$out_dir/backend_vs_kalibr_summary.txt" ]]; then
    echo "skip existing ${model_label}: ${train_dir} -> ${test_dir}" | tee "$log_file"
  else
  echo "running ${model_label}: ${train_dir} -> ${test_dir}" | tee "$log_file"
  ./build/run_stage5_backend \
    --image "$train_dir" \
    --test-image "$test_dir" \
    --config "$CONFIG" \
    --kalibr-camchain "$camchain" \
    --models "$model_family" \
    --output "$out_dir" \
    --runtime-mode research \
    --cache-dir "$CACHE_DIR" \
    --all \
    --internal-regeneration-diagnostics \
    --stage5-export-internal-seed-step-overlays \
    >> "$log_file" 2>&1
  fi

  {
    echo ""
    echo "=== $(date '+%F %T') ${model_label} ${train_dir} -> ${test_dir} ==="
    sed -n '1,120p' "$out_dir/backend_training_summary.txt"
    sed -n '1,120p' "$out_dir/backend_holdout_summary.txt"
    sed -n '1,120p' "$out_dir/backend_vs_kalibr_summary.txt"
    if [[ -f "$out_dir/experiment_config_summary.txt" ]]; then
      sed -n '1,520p' "$out_dir/experiment_config_summary.txt" |
        rg -n "effective_geometry_prior|backend_skip_optimization|backend_final_state_label|effective_stage5_trial_backend_selection_accepted_candidates|accepted_frame_count|accepted_board_observation_count|camera_(xi|alpha|fu|fv|cu|cv)|overall_rmse|outer_only_rmse|internal_only_rmse" || true
    fi
  } >> "$REPORT_DIR/stage5_matrix_results.txt"
}

run_one "ds" "$DS_CAMCHAIN" "$TRAIN_1" "$TEST_1"
run_one "ds" "$DS_CAMCHAIN" "$TRAIN_2" "$TEST_2"
run_one "ds" "$DS_CAMCHAIN" "$TRAIN_3" "$TEST_3"
run_one "kb" "$KB_CAMCHAIN" "$TRAIN_1" "$TEST_1"
run_one "kb" "$KB_CAMCHAIN" "$TRAIN_2" "$TEST_2"
run_one "kb" "$KB_CAMCHAIN" "$TRAIN_3" "$TEST_3"

echo "done" | tee -a "$REPORT_DIR/stage5_matrix_results.txt"
