# BabelCalib MAT input for Stage5

This adapter imports BabelCalib `corners`, `boards`, and `imgsize` data without
changing the native Stage5 calibration pipeline.

The Python adapter requires NumPy and SciPy (`pip install -r
tools/babelcalib_import/requirements.txt`). MATLAB v7.3/HDF5 files should first
be saved as a standard v7 MAT file supported by `scipy.io.loadmat`.

The data path is:

```text
BabelCalib MAT -> audited CSV/YAML interchange -> outer-only camera init
-> multi-board outer bootstrap -> Stage5 selection
-> persistent incremental selection BA -> backend/holdout evaluation
```

`boards.Rt` is validated during conversion but is not used to initialize the
camera or multi-board layout. Imported internal points are frozen observations;
Stage5 does not detect or regenerate points from images in this mode.

## Train/test MAT files

```bash
/Users/linzhaoxian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tools/babelcalib_import/run_stage5_from_mat.py \
  --train-mat image/babelcalib_export/mul-board/EXPORT/train.mat \
  --test-mat image/babelcalib_export/mul-board/EXPORT/test.mat \
  --config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --models ds-none \
  --kalibr-camchain config/mono_fisheye_calib_3_25_right-camchain.yaml \
  --output result_may/stage5_babelcalib_mat
```

Additional unknown arguments are forwarded to `run_stage5_backend`, so the
normal residual and persistent-selection controls remain available.

Use `--target-mode auto` (default), `single_board`, or `multi_board`. In
single-board mode the only board is fixed as the reference, candidate batches
are complete frames, and multi-board cohesion/consistency scoring is disabled.
Camera intrinsics and each frame's camera-to-board pose remain active in the
persistent incremental BA. Imported `boards.X` coordinates define the actual
outer target geometry, so rectangular checkerboards are not treated as square
AprilTags.

For a dense single-board grid, Stage5 enables the
`single_board_dense_grid` selection profile. One deterministically shuffled
view is the forced seed and every remaining view is an incremental candidate
batch. Every valid imported grid point in a view is kept together; there is no
within-view point selection. The acceptance boundary follows Kalibr's camera
calibration routine: accept when information gain is greater than `0.2` or the
camera-information rank increases, provided optimization is valid
(`JFinal < JStart`, finite state, and convergence before the 50-iteration
limit). Multi-board balance, cohesion, layout consistency, and residual-score
gates do not participate in checkerboard acceptance.

Dense-grid camera initialization also evaluates every camera seed on the same
complete observation set. Each view has an independent target pose and all
valid grid points contribute to the joint reprojection objective. The dense
initializer uses Kalibr's intrinsic-calibration iteration budget
(`maxIterations=200`, `convergenceDeltaX=1e-3`, `convergenceDeltaJ=1`);
coverage/information selection is not allowed to give different seeds
different view subsets.

## Deterministic split from all.mat

```bash
/Users/linzhaoxian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tools/babelcalib_import/run_stage5_from_mat.py \
  --mat path/to/all.mat --holdout-ratio 0.30 --split-seed 1337 \
  --config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --models ds-none --kalibr-camchain path/to/camchain.yaml \
  --output result_may/stage5_babelcalib_all_split
```

The generated interchange, split indices, manifest, and exact Stage5 command
are retained under `OUTPUT/precomputed_input/` for audit.

## All views for calibration

Use `--all-training` when every view in `all.mat` must participate in camera
initialization and selection BA:

```bash
/Users/linzhaoxian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tools/babelcalib_import/run_stage5_from_mat.py \
  --mat path/to/all.mat --all-training \
  --config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --models ds-none --kalibr-camchain path/to/reference-only-camchain.yaml \
  --output result_may/stage5_babelcalib_all_training
```

The current Stage5 benchmark entry point requires a second evaluation input,
so this mode imports the same observations under `precomputed_input/training_diagnostic/`.
That copy is only a training reprojection diagnostic: it is not independent,
and its metrics must not be reported as held-out or generalization results.
`calibration_protocol_summary.txt` and `run_manifest.json` record this explicitly.

The catalog wrapper exposes the same protocol as `--all`:

```bash
python3 tools/babelcalib_import/run_stage5_mat_catalog.py \
  --mat path/to/all.mat --dataset-id DATASET --camera left \
  --models ds kb --catalog-subdir checkerboard --all
```

## Current baseline plus canonical comparison

`run_stage5_mat_catalog.py` is the reproducible one-command wrapper for the
paper comparison. It runs the native Stage5 baseline, publishes the resulting
camera YAML under `intrintic/catalog/current_baseline`, and passes the matching
canonical YAMLs to Stage5 as evaluation-only references. The canonical files
are never used to generate the Stage5 seed or to update the BA state.

For one model with all training views and a separate frozen test MAT:

```bash
cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

python3 tools/babelcalib_import/run_stage5_mat_catalog.py \
  --train-mat image/babelcalib_export/checkerboard/checkboard_3_25_right_clear_export_11x8/all.mat \
  --test-mat image/babelcalib_export/checkerboard/stereo_4_2-3_images_right_export_11x8/all.mat \
  --dataset-id checkerboard-3-25-all \
  --camera right \
  --models ds \
  --canonical-root intrintic/catalog/canonical \
  --catalog-subdir checkerboard
```

The wrapper automatically selects the Kalibr YAML in
`canonical/right/checkerboard-3-25__right/ds/` as `--kalibr-camchain` and
passes the other same-dataset YAMLs as `--reference-intrinsics-yaml`. The run
directory includes:

```text
result_may/stage5_mat_<dataset>_<camera>_<timestamp>/
  ds/
    backend_training_summary.txt
    backend_holdout_summary.txt
    canonical_comparison.csv
  canonical_comparison_all_models.csv
  catalog_publish_manifest.json
```

`canonical_comparison.csv` reports the Ours row and every canonical reference
on both the training and frozen-test split. To run the original all-view
diagnostic protocol, use `--mat ... --all`; for a deterministic internal split,
omit `--test-mat` and use `--mat ... --split-seed 1337`.

Use `--reference-yaml LABEL:PATH` to add an explicit reference, or
`--disable-auto-canonical-references` to disable canonical auto-discovery.
`--canonical-root` can point at another catalog root without changing the
calibration code.

To run all supported camera models in one invocation, replace `--models ds`
with `--models all`. This expands to four independent Stage5 runs:

```bash
python3 tools/babelcalib_import/run_stage5_mat_catalog.py \
  --train-mat image/babelcalib_export/checkerboard/checkboard_3_25_right_clear_export_11x8/all.mat \
  --test-mat image/babelcalib_export/checkerboard/stereo_4_2-3_images_right_export_11x8/all.mat \
  --dataset-id checkerboard-3-25-all \
  --camera right \
  --models all \
  --canonical-root intrintic/catalog/canonical \
  --catalog-subdir checkerboard
```

`all` runs, in order, `ds`, `kb`, `eucm`, and native `ucm` (`omni-none`).
Each model receives its own Stage5 model family, Kalibr camchain, canonical
reference directory, output subdirectory, and current-baseline YAML. The
aggregate comparison is written to
`canonical_comparison_all_models.csv`.

## Validation

```bash
python3 tools/babelcalib_import/check_stage5_interchange.py \
  OUTPUT/precomputed_input/training
```

## Auxiliary initialization sessions

For datasets captured with the same camera but a different physical board
layout, additional precomputed sessions can constrain only the shared camera
initialization:

```bash
./build/run_stage5_backend \
  --stage5-precomputed-observations-dir PRIMARY_TRAINING \
  --stage5-precomputed-holdout-observations-dir FROZEN_HOLDOUT \
  --stage5-init-auxiliary-precomputed-observations-dir AUXILIARY_SESSION \
  --stage5-precomputed-init-use-all-points 1 \
  --stage5-precomputed-target-mode multi_board \
  --config CONFIG --models ds-none \
  --kalibr-camchain REFERENCE_ONLY.yaml --output OUTPUT --all
```

The auxiliary option is repeatable. Auxiliary frame-board observations use an
independent target pose during camera initialization. They never enter the
primary session's board layout, frame selection, persistent backend, or frozen
holdout evaluation. Stage5 rejects an auxiliary path that aliases the primary
training input, frozen holdout input, or another auxiliary session. Resolution,
target mode, board count, and reference board must match the primary input.
