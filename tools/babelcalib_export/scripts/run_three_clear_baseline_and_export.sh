#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib"
PY="/Users/linzhaoxian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
EXPORT_SCRIPT="$ROOT/tools/babelcalib_export/scripts/export_stage5_to_babelcalib.py"
CHECK_SCRIPT="$ROOT/tools/babelcalib_export/scripts/check_babelcalib_mat.py"
CONFIG="$ROOT/aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
CAMCHAIN="$ROOT/config/mono_fisheye_calib_3_25_right-camchain.yaml"
RUN_BASE="$ROOT/result_may/babelcalib_three_clear_stage5_baseline_runs"
EXPORT_BASE="$ROOT/tools/babelcalib_export/three_clear_baseline_exports"
LOG_DIR="$RUN_BASE/logs"

mkdir -p "$RUN_BASE" "$EXPORT_BASE" "$LOG_DIR"
cd "$ROOT"

run_one() {
  local tag="$1"
  local image_dir="$ROOT/image/datatset_5_1/stereo_dataset_20260430_${tag}/right"
  local out_dir="$RUN_BASE/${tag}"
  local export_dir="$EXPORT_BASE/${tag}"
  local cache_dir="$ROOT/result/.babelcalib_${tag}_stage5_baseline_cache"
  local log_file="$LOG_DIR/${tag}.log"

  echo "[$(date '+%F %T')] start ${tag}" | tee "$log_file"
  ./build/run_stage5_backend \
    --image "$image_dir" \
    --config "$CONFIG" \
    --kalibr-camchain "$CAMCHAIN" \
    --models ds-none \
    --output "$out_dir" \
    --runtime-mode research \
    --cache-dir "$cache_dir" \
    --split-mode random_holdout_ratio \
    --holdout-ratio 0.30 \
    --split-seed 1337 \
    --all \
    --internal-regeneration-diagnostics \
    --stage5-export-internal-seed-step-overlays \
    >> "$log_file" 2>&1

  local backend_points="$out_dir/backend_training_points.csv"
  if [[ ! -f "$backend_points" ]]; then
    backend_points="$out_dir/backend_points_from_used_frames.csv"
    "$PY" - "$out_dir" "$backend_points" <<'PY'
import csv
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
used_path = run_dir / "backend_used_frames.csv"
training_path = run_dir / "benchmark_training_points.csv"
if not used_path.exists():
    raise SystemExit(f"missing backend selection file: {used_path}")
if not training_path.exists():
    raise SystemExit(f"missing benchmark training points: {training_path}")
used = {}
with used_path.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        used[(row["frame_index"], row["frame_label"])] = set(row["used_board_ids"].split(";"))
rows = 0
frames = set()
with training_path.open(newline="") as fi, out_path.open("w", newline="") as fo:
    reader = csv.DictReader(fi)
    writer = csv.DictWriter(fo, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        boards = used.get((row["frame_index"], row["frame_label"]))
        if row.get("method") == "ours" and boards is not None and row["board_id"] in boards:
            writer.writerow(row)
            rows += 1
            frames.add((row["frame_index"], row["frame_label"]))
print(f"wrote {out_path}: rows={rows} frames={len(frames)} selected_frames={len(used)}")
PY
  fi

  echo "[$(date '+%F %T')] export ${tag}" | tee -a "$log_file"
  "$PY" "$EXPORT_SCRIPT" \
    --stage5-run-dir "$out_dir" \
    --output-dir "$export_dir" \
    --points-source benchmark \
    --method ours \
    --split-ratio 0.30 \
    --seed 1337 \
    --use-existing-stage5-split \
    --image-dir "$image_dir" \
    --backend-points-csv "$backend_points" \
    >> "$log_file" 2>&1

  "$PY" "$CHECK_SCRIPT" \
    "$export_dir/all.mat" \
    "$export_dir/train.mat" \
    "$export_dir/test.mat" \
    "$export_dir/backend.mat" \
    > "$export_dir/check_summary.json"
  echo "[$(date '+%F %T')] done ${tag}" | tee -a "$log_file"
}

run_one "$1"
