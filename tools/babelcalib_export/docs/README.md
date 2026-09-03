# TartanCalib Multi-Board to BabelCalib Export

This folder exports the current TartanCalib multi-board Stage5 observations to
BabelCalib `.mat` files.  It does not modify BabelCalib and does not rerun a new
detector.  It reuses Stage5 point diagnostics so the exported data matches the
same frontend/rescue logic used by our calibration pipeline.

## BabelCalib Format

The local BabelCalib code at `/Users/linzhaoxian/lzx-ws/project/calibr/babelcalib`
expects:

- `corners(n).x`: `2 x N` image observations.
- `corners(n).cspond`: `2 x N`, where row 1 is the fiducial index and row 2 is
  the BabelCalib board index.
- `boards(k).X`: `2 x M` board-local fiducial coordinates.
- `boards(k).Rt`: `3 x 4` fixed board layout pose.
- `imgsize`: `[height width]`.

The exporter remaps TartanCalib's original `point_id` values to dense 1-based
BabelCalib fiducial indices per board and writes `fiducial_map.jsonl` so the
mapping remains auditable.

## Important Layout Requirement

BabelCalib uses `boards(k).Rt` as the fixed multi-board target layout.  A real
multi-board export therefore needs a layout file containing complete board
poses, for example:

```csv
board_id,initialized,observation_count,rmse,T_reference_board_16
1,1,28,0.1,1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

The Stage5 code can write this as:

```text
stage5_intermediate_model/intermediate_board_poses.csv
```

If no layout is present, the exporter fails by default.  `--allow-identity-layout-for-debug`
exists only to test MATLAB struct formatting; it is not valid for real multi-board
calibration.

## Export

```bash
PY=/Users/linzhaoxian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3

$PY /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/tools/babelcalib_export/scripts/export_stage5_to_babelcalib.py \
  --stage5-run-dir /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage5_random7030_ds_20260629_144928clear \
  --layout-csv /path/to/intermediate_board_poses.csv \
  --output-dir /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/tools/babelcalib_export/examples/format_smoke_identity_144928clear_ds \
  --points-source benchmark \
  --method ours \
  --split-ratio 0.30 \
  --seed 1337
```

To reuse Stage5's existing training/holdout CSV split instead of rebuilding a
new random split:

```bash
  --use-existing-stage5-split
```

The output directory contains:

- `all.mat`
- `train.mat`
- `test.mat`
- `export_summary.json`
- `fiducial_map.jsonl`
- `frames_all.jsonl`
- `frames_train.jsonl`
- `frames_test.jsonl`
- `warnings.txt`

### Export from a single point CSV

For datasets that only exist as one Stage5 point CSV, pass it as the whole
dataset and let the exporter create a fixed-seed train/test split:

```bash
$PY /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/tools/babelcalib_export/scripts/export_stage5_to_babelcalib.py \
  --stage5-run-dir /path/to/stage5_run \
  --all-points-csv /path/to/benchmark_holdout_points.csv \
  --layout-csv /path/to/intermediate_board_poses.csv \
  --output-dir /path/to/babelcalib_export \
  --split-ratio 0.30 \
  --seed 1337
```

## Check

```bash
$PY /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/tools/babelcalib_export/scripts/check_babelcalib_mat.py \
  /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/tools/babelcalib_export/examples/format_smoke_identity_144928clear_ds/all.mat \
  /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/tools/babelcalib_export/examples/format_smoke_identity_144928clear_ds/train.mat \
  /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/tools/babelcalib_export/examples/format_smoke_identity_144928clear_ds/test.mat
```

The checker validates that every image point has a valid board/fiducial
correspondence and that each board has a complete `X` and `Rt`.

## Minimal MATLAB Validation

In MATLAB:

```matlab
babelcalib_root = '/Users/linzhaoxian/lzx-ws/project/calibr/babelcalib';
export_dir = '/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/tools/babelcalib_export/examples/format_smoke_identity_144928clear_ds';
run('/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/tools/babelcalib_export/matlab/run_babelcalib_minimal_validation.m');
```

The script loads `train.mat` and `test.mat`, then calls:

```matlab
model = calibrate(corners, boards, imgsize, ...);
test_model = get_poses(model, test_corners, boards, imgsize, ...);
```

It prints the train/test RMS values if BabelCalib accepts the exported data.
