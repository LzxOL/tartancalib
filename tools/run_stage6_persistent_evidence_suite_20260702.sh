#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BIN="./build/run_stage6_stereo_extrinsic"
LEFT_CONFIG="aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
RIGHT_CONFIG="${LEFT_CONFIG}"
LEFT_INTRINSICS="config/mono_fisheye_calib_3_25_left-camchain.yaml"
RIGHT_INTRINSICS="config/mono_fisheye_calib_3_25_right-camchain.yaml"
REFERENCE_CAMCHAIN="config/stereo_4_2-3-camchain.yaml"

DATA_1444190_CLEAR="image/datatset_5_1/stereo_dataset_20260430_1444190-clear"
DATA_144928_CLEAR="image/datatset_5_1/stereo_dataset_20260430_144928-clear"

OUT_ROOT="result_may"
CACHE_ROOT="result"
TAG="20260702"

run_stage6_common() {
  local train_dataset="$1"
  local holdout_dataset="$2"
  local output_dir="$3"
  local cache_dir="$4"
  shift 4
  "${BIN}" \
    --left-image "${train_dataset}/left" \
    --right-image "${train_dataset}/right" \
    --test-left-image "${holdout_dataset}/left" \
    --test-right-image "${holdout_dataset}/right" \
    --left-config "${LEFT_CONFIG}" \
    --right-config "${RIGHT_CONFIG}" \
  --models ds-none \
  --output "${output_dir}" \
    --cache-dir "${cache_dir}" \
    --stage6-solver-mode global_sparse_ba \
    --stage6-skip-final-global-ba \
    --stage6-selection-ba-residual-mode pixel \
    --stage6-final-ba-residual-mode pixel \
    --stage6-training-pair-sample-count 12 \
    --stage6-training-pair-sample-seed 20260702 \
    "$@"
}

cmake --build build --target run_stage6_stereo_extrinsic -j 8

run_stage6_common \
  "${DATA_1444190_CLEAR}" \
  "${DATA_144928_CLEAR}" \
  "${OUT_ROOT}/stage6_persistent_incremental_ref_cross_clean_${TAG}_1444190clear_to_144928clear" \
  "${CACHE_ROOT}/.stage6_persistent_incremental_ref_cross_${TAG}_cache" \
  --stage6-persistent-incremental-max-iterations 6 \
  --stage6-incremental-mi-tol 0.05

run_stage6_common \
  "${DATA_144928_CLEAR}" \
  "${DATA_1444190_CLEAR}" \
  "${OUT_ROOT}/stage6_persistent_incremental_reverse_ref_cross_${TAG}_144928clear_to_1444190clear" \
  "${CACHE_ROOT}/.stage6_persistent_incremental_reverse_cross_${TAG}_cache" \
  --stage6-persistent-incremental-max-iterations 6 \
  --stage6-incremental-mi-tol 0.05

run_stage6_common \
  "${DATA_1444190_CLEAR}" \
  "${DATA_144928_CLEAR}" \
  "${OUT_ROOT}/stage6_legacy_selection_ref_cross_${TAG}_1444190clear_to_144928clear" \
  "${CACHE_ROOT}/.stage6_persistent_incremental_ref_cross_${TAG}_cache" \
  --stage6-disable-persistent-incremental-stereo-ba

run_stage6_common \
  "${DATA_144928_CLEAR}" \
  "${DATA_1444190_CLEAR}" \
  "${OUT_ROOT}/stage6_legacy_selection_reverse_cross_${TAG}_144928clear_to_1444190clear" \
  "${CACHE_ROOT}/.stage6_persistent_incremental_reverse_cross_${TAG}_cache" \
  --stage6-disable-persistent-incremental-stereo-ba

"${BIN}" \
  --left-image "${DATA_1444190_CLEAR}/left" \
  --right-image "${DATA_1444190_CLEAR}/right" \
  --test-left-image "${DATA_144928_CLEAR}/left" \
  --test-right-image "${DATA_144928_CLEAR}/right" \
  --left-config "${LEFT_CONFIG}" \
  --right-config "${RIGHT_CONFIG}" \
  --models ds-none \
  --output "${OUT_ROOT}/stage6_persistent_incremental_rejection_smoke_${TAG}_1444190clear_to_144928clear" \
  --cache-dir "${CACHE_ROOT}/.stage6_persistent_incremental_metricclean_smoke_${TAG}_cache" \
  --stage6-solver-mode global_sparse_ba \
  --stage6-skip-final-global-ba \
  --stage6-selection-ba-residual-mode pixel \
  --stage6-final-ba-residual-mode pixel \
  --stage6-training-pair-sample-count 3 \
  --stage6-training-pair-sample-seed 20260702 \
  --stage6-persistent-incremental-max-iterations 4 \
  --stage6-incremental-mi-tol 10.0

SUMMARY_DIR="${OUT_ROOT}/stage6_final_evidence_summary_${TAG}"
python3 tools/summarize_stage6_stereo_experiments.py \
  "${OUT_ROOT}/stage6_persistent_incremental_ref_cross_clean_${TAG}_1444190clear_to_144928clear" \
  "${OUT_ROOT}/stage6_legacy_selection_ref_cross_${TAG}_1444190clear_to_144928clear" \
  "${OUT_ROOT}/stage6_persistent_incremental_reverse_ref_cross_${TAG}_144928clear_to_1444190clear" \
  "${OUT_ROOT}/stage6_legacy_selection_reverse_cross_${TAG}_144928clear_to_1444190clear" \
  "${OUT_ROOT}/stage6_persistent_incremental_rejection_smoke_${TAG}_1444190clear_to_144928clear" \
  --output-dir "${SUMMARY_DIR}"

python3 tools/verify_stage6_persistent_outputs.py --write-report \
  "${OUT_ROOT}/stage6_persistent_incremental_ref_cross_clean_${TAG}_1444190clear_to_144928clear" \
  "${OUT_ROOT}/stage6_persistent_incremental_reverse_ref_cross_${TAG}_144928clear_to_1444190clear"

python3 tools/verify_stage6_persistent_outputs.py \
  \
  --require-rejection \
  --write-report \
  "${OUT_ROOT}/stage6_persistent_incremental_rejection_smoke_${TAG}_1444190clear_to_144928clear"

python3 tools/audit_stage6_persistent_vs_legacy.py \
  "${SUMMARY_DIR}/stage6_stereo_experiment_summary.csv" \
  --rmse-tolerance 0.05 \
  --output "${OUT_ROOT}/stage6_persistent_vs_legacy_bidirectional_reference_clean_summary_${TAG}/stage6_persistent_vs_legacy_audit.txt"

python3 tools/write_stage6_final_evidence_report.py \
  --summary-csv "${SUMMARY_DIR}/stage6_stereo_experiment_summary.csv" \
  --comparison-audit "${OUT_ROOT}/stage6_persistent_vs_legacy_bidirectional_reference_clean_summary_${TAG}/stage6_persistent_vs_legacy_audit.txt" \
  --verifier-report "${OUT_ROOT}/stage6_persistent_incremental_ref_cross_clean_${TAG}_1444190clear_to_144928clear/stage6_persistent_output_verification.txt" \
  --verifier-report "${OUT_ROOT}/stage6_persistent_incremental_reverse_ref_cross_${TAG}_144928clear_to_1444190clear/stage6_persistent_output_verification.txt" \
  --verifier-report "${OUT_ROOT}/stage6_persistent_incremental_rejection_smoke_${TAG}_1444190clear_to_144928clear/stage6_persistent_output_verification.txt" \
  --output "${SUMMARY_DIR}/stage6_persistent_final_evidence_report.md"

echo "Stage6 persistent evidence suite complete: ${SUMMARY_DIR}"
