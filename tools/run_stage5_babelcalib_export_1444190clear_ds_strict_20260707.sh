#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib"
BIN="$ROOT/build/run_stage5_backend"
CONFIG="$ROOT/aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
IMAGE="$ROOT/image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right"
OUT="$ROOT/result_may/stage5_baseline_babelcalib_export_20260707_1444190clear_ds_strict"
CACHE="$ROOT/result/.stage5_baseline_babelcalib_export_20260707_1444190clear_ds_strict_cache"
EXPORT="$ROOT/image/babelcalib_multiboard_export_20260707_1444190clear_ds_strict"
DS_TARTAN="$ROOT/intrintic/tartan_result_right/mono_fisheye_calib_6_23_right-tartan-ds-camchain.yaml"

echo "[$(date '+%F %T')] start Stage5 baseline + BabelCalib export"
echo "image=$IMAGE"
echo "output=$OUT"
echo "cache=$CACHE"
echo "export=$EXPORT"

"$BIN" \
  --config "$CONFIG" \
  --runtime-mode research \
  --split-mode random_holdout_ratio \
  --holdout-ratio 0.30 \
  --split-seed 1337 \
  --all \
  --stage5-disable-selected-case-visualizations \
  --stage5-enable-angular-residual-diagnostics \
  --stage5-enable-polar-angle-diagnostics \
  --backend-residual-model hybrid_edge_angular \
  --backend-hybrid-angular-threshold-deg 50 \
  --image "$IMAGE" \
  --models ds-none \
  --kalibr-camchain "$DS_TARTAN" \
  --output "$OUT" \
  --cache-dir "$CACHE"

echo "[$(date '+%F %T')] Stage5 finished; exporting strict BabelCalib .mat"

python3 "$ROOT/tools/export_multiboard_babelcalib.py" \
  --source-run "$OUT" \
  --output-dir "$EXPORT" \
  --image-height 4512 \
  --image-width 4512 \
  --reference-board-id 1 \
  --seed 1337 \
  --test-ratio 0.30 \
  --use-source-split \
  --complete-canonical-grid \
  --no-include-metadata-fields \
  --overwrite

python3 "$ROOT/tools/check_babelcalib_multiboard_export.py" --export-dir "$EXPORT"

echo "[$(date '+%F %T')] done Stage5 baseline + BabelCalib export"
