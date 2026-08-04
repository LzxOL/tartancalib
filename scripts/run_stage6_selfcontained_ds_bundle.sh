#!/usr/bin/env bash
set -euo pipefail

# Run one DS Stage6 calibration and freeze its final left/right intrinsics plus
# T_cam1_cam0 as an inseparable downstream bundle. No historical intrinsics or
# camchain may enter this workflow.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: STAGE6_BA_MODE=pixel|angular|hybrid scripts/run_stage6_selfcontained_ds_bundle.sh [dataset] [stage6-output] [bundle-dir] [holdout-offset]

Runs self-contained DS Stage6, verifies the native final left/right camera
YAMLs, and freezes those YAMLs plus T_cam1_cam0 into a non-overwritable bundle.
EOF
  exit 0
fi

DATASET="${1:-image/datatset_5_1/stereo_dataset_20260430_1444190-clear}"
RUN_ID="${STAGE6_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STAGE6_OUTPUT="${2:-result_may/stage6_selfcontained_ds_${RUN_ID}}"
BUNDLE_DIR="${3:-intrintic/catalog/current_baseline/stereo_bundles/${RUN_ID}_stage6_selfcontained_ds}"
HOLDOUT_OFFSET="${4:-0}"
STAGE6_BA_MODE="${STAGE6_BA_MODE:-angular}"

if [[ -e "${BUNDLE_DIR}" ]] && [[ -n "$(find "${BUNDLE_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite a frozen stereo bundle: ${BUNDLE_DIR}" >&2
  exit 2
fi

STAGE6_BA_MODE="${STAGE6_BA_MODE}" scripts/run_stage6_frozen_selfcontained_ds_baseline.sh \
  "${DATASET}" "${STAGE6_OUTPUT}" "${HOLDOUT_OFFSET}"

python3 tools/stereo_downstream/create_provenance_locked_stereo_bundle.py \
  --stage6-output "${STAGE6_OUTPUT}" \
  --use-stage6-final-intrinsics \
  --training-left-dir "${DATASET}/left" \
  --training-right-dir "${DATASET}/right" \
  --holdout-left-dir "${DATASET}/left" \
  --holdout-right-dir "${DATASET}/right" \
  --holdout-role within_sequence_holdout \
  --max-pair-delta-ms 0 \
  --bundle-dir "${BUNDLE_DIR}"

echo "stage6_output=${STAGE6_OUTPUT}"
echo "stereo_bundle=${BUNDLE_DIR}"
