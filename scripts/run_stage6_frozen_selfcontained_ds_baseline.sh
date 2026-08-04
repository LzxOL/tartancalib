#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: STAGE6_BA_MODE=pixel|angular|hybrid scripts/run_stage6_frozen_selfcontained_ds_baseline.sh [dataset] [output-dir] [holdout-offset]

Runs the frozen DS Stage6 protocol with strict timestamp pairing and a 2/3
training, 1/3 within-sequence holdout split. The final left/right camera YAMLs
are required to match the Stage6 optimized scene.

STAGE6_BA_MODE defaults to angular. `angular` selects the frozen
spherical-tangent persistent-selection protocol. `hybrid` selects the
pixel+tangent residual blocks with the same polar weighting as Stage6 BA.
The selected residual metric and its units are recorded in every summary.
EOF
  exit 0
fi

# Frozen 2026-08-02 Stage6 baseline. Intrinsics are learned from each side's
# image observations; no YAML, camchain, or external intrinsics are accepted.
DATASET="${1:-image/datatset_5_1/stereo_dataset_20260430_1444190-clear}"
OUTPUT_DIR="${2:-result_may/stage6_frozen_selfcontained_ds_baseline_20260802}"
HOLDOUT_OFFSET="${3:-0}"
OUTPUT_BASENAME="$(basename "${OUTPUT_DIR}")"
CACHE_DIR="${STAGE6_CACHE_DIR:-result/.${OUTPUT_BASENAME}_cache}"
STAGE6_BA_MODE="${STAGE6_BA_MODE:-angular}"

if [[ "${STAGE6_BA_MODE}" != "pixel" && "${STAGE6_BA_MODE}" != "angular" &&
      "${STAGE6_BA_MODE}" != "hybrid" ]]; then
  echo "STAGE6_BA_MODE must be pixel, angular, or hybrid, got: ${STAGE6_BA_MODE}" >&2
  exit 2
fi

cmake --build build --target run_stage6_stereo_extrinsic -j 8

./build/run_stage6_stereo_extrinsic \
  --left-image "${DATASET}/left" \
  --right-image "${DATASET}/right" \
  --left-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --right-config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --models ds-none \
  --output "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --stage6-stereo-measurement-source all_valid \
  --stage6-frame-pairing-mode exact_timestamp \
  --holdout-stride 3 \
  --holdout-offset "${HOLDOUT_OFFSET}" \
  --stage6-solver-mode global_sparse_ba \
  --stage6-intrinsics-mode adaptive_regularized_joint_projection \
  --stage6-persistent-pose-structure independent_pair_board \
  --stage6-ba-mode "${STAGE6_BA_MODE}" \
  --stage6-skip-final-global-ba \
  --stage6-enable-persistent-incremental-stereo-ba

python3 tools/verify_stage6_persistent_outputs.py \
  --expected-pose-structure independent_pair_board \
  --require-final-camera-yamls \
  --write-report \
  "${OUTPUT_DIR}"
