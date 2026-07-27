#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BUNDLE_DIR="${BUNDLE_DIR:?Set BUNDLE_DIR to a provenance-locked stereo bundle directory}"
OUTPUT_DIR="${OUTPUT_DIR:-paper_experiments/2026_07_27_stereo_rectification_disparity_provenance_locked}"

for artifact in left_intrinsics.yaml right_intrinsics.yaml stereo_extrinsic.yaml stereo_bundle_manifest.json; do
  if [[ ! -f "$BUNDLE_DIR/$artifact" ]]; then
    echo "missing provenance-locked bundle artifact: $BUNDLE_DIR/$artifact" >&2
    exit 2
  fi
done

python3 tools/stereo_downstream/run_rectification_disparity_visualization.py \
  --ours-left-intrinsics "$BUNDLE_DIR/left_intrinsics.yaml" \
  --ours-right-intrinsics "$BUNDLE_DIR/right_intrinsics.yaml" \
  --ours-extrinsic "$BUNDLE_DIR/stereo_extrinsic.yaml" \
  --kalibr-camchain config/stereo_4_2-3-camchain.yaml \
  --left-dir image/mid_far_dataset/stereo_dataset_20260430_144928/left \
  --right-dir image/mid_far_dataset/stereo_dataset_20260430_144928/right \
  --calibration-left-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/left \
  --calibration-right-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right \
  --timestamp-tolerance-ms 1 \
  --output "$OUTPUT_DIR" \
  "$@"
