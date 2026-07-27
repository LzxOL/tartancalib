# Stereo Rectification and Disparity Visualization

This tool is a qualitative downstream experiment for DS stereo calibration. It
compares exactly two complete systems: `Kalibr` and `Ours`. Each system uses
its own left/right intrinsics and stereo extrinsic; no intrinsic/extrinsic
cross-combination is permitted.

Run the frozen protocol only with a provenance-locked bundle:

```bash
BUNDLE_DIR=intrintic/catalog/current_baseline/stereo_bundles/2026_07_27_ours_ds_train1444190_timestamp1ms \
scripts/run_stereo_rectification_disparity_visualization.sh
```

## Provenance-locked stereo bundle

Do not combine a left intrinsic YAML, right intrinsic YAML, and an external
YAML from separate calibration runs. First estimate the stereo external on
strictly timestamp-synchronized *training* pairs with both intrinsics fixed,
then create the immutable three-file bundle below. The holdout sequence is
recorded only for validation and is excluded from the Stage6 optimization.

```bash
python3 tools/stereo_downstream/create_provenance_locked_stereo_bundle.py \
  --stage6-output paper_experiments/2026_07_27_provenance_locked_stereo_bundle/stage6_ours_fixedk_timestamp1ms_train1444190_testmidfar144928 \
  --left-intrinsics intrintic/catalog/canonical/left/ds/stereo-4-2-3__left__tartancalib__ds.yaml \
  --right-intrinsics intrintic/catalog/canonical/right/checkerboard-3-25__right/ds/checkerboard-3-25__right__tartancalib__ds.yaml \
  --training-left-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/left \
  --training-right-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right \
  --holdout-left-dir image/mid_far_dataset/stereo_dataset_20260430_144928/left \
  --holdout-right-dir image/mid_far_dataset/stereo_dataset_20260430_144928/right \
  --max-pair-delta-ms 1 \
  --bundle-dir intrintic/catalog/current_baseline/stereo_bundles/2026_07_27_ours_ds_train1444190_timestamp1ms
```

The generated `stereo_bundle_manifest.json` records SHA-256 checksums of the
three copied YAML files, the coordinate convention
`p_cam1 = R_cam1_from_cam0 * p_cam0 + t_cam1_from_cam0`, input paths, and the
strict pairing statistics. Run downstream evaluation using only those copied
YAML files:

```bash
python3 tools/stereo_downstream/run_rectification_disparity_visualization.py \
  --ours-left-intrinsics intrintic/catalog/current_baseline/stereo_bundles/2026_07_27_ours_ds_train1444190_timestamp1ms/left_intrinsics.yaml \
  --ours-right-intrinsics intrintic/catalog/current_baseline/stereo_bundles/2026_07_27_ours_ds_train1444190_timestamp1ms/right_intrinsics.yaml \
  --ours-extrinsic intrintic/catalog/current_baseline/stereo_bundles/2026_07_27_ours_ds_train1444190_timestamp1ms/stereo_extrinsic.yaml \
  --kalibr-camchain config/stereo_4_2-3-camchain.yaml \
  --left-dir image/mid_far_dataset/stereo_dataset_20260430_144928/left \
  --right-dir image/mid_far_dataset/stereo_dataset_20260430_144928/right \
  --calibration-left-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/left \
  --calibration-right-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right \
  --timestamp-tolerance-ms 1 \
  --output paper_experiments/2026_07_27_stereo_rectification_disparity_provenance_locked \
  --refresh-freeze
```

The default evaluates four blind-selected timestamp-synchronized pairs from
`image/mid_far_dataset/stereo_dataset_20260430_144928`, while the declared
calibration sequence is `1444190-clear`. The first run writes
`frame_manifest.yaml`, `frame_manifest.csv`, and `frozen_matches.csv`; future
runs reuse them. Use `--refresh-freeze` only to intentionally replace this
protocol.

Pairing is global greedy one-to-one by the capture timestamp embedded in each
filename, not by frame index. The default `--timestamp-tolerance-ms 1` rejects
delayed frames. The selected timestamp delta is written into the manifest and
`protocol.json`. A legacy manifest without this field is rejected and must be
regenerated with `--refresh-freeze`.

The output directory contains:

- `metrics.csv`: aggregate epipolar P95 and disparity-validity metrics.
- `per_frame_metrics.csv`: metrics for every frozen pair.
- `rectified_frozen_matches.csv`: every raw frozen match after each system's
  rectification, including validity and vertical error.
- `rectification_map_audit.csv`: source-map coverage for both systems.
- `stereo_rectification_disparity_figure.png/.pdf`: one composite figure.
- `stereo_rectification_disparity_table.tex`: the compact paper table.
- `protocol.json`: all calibration paths, virtual pinhole settings, SGBM
  settings, frozen frame IDs, and geometry checks.

The virtual camera is `2048x1536`, horizontal FoV is `120 deg`, and SGBM uses
`minDisparity=0`, `numDisparities=192`, and `blockSize=7`. Invalid map samples
and invalid disparity are shown in gray. The peripheral metrics use the fixed
elliptical annulus `rho in [0.65, 0.90]`; this region is independent of either
camera model.

`latex_includes.tex` contains a conservative figure/table insertion snippet.
The current default mid-far sequence is independent of the declared calibration
sequence but still contains the multi-board rig. It validates the downstream
protocol and must not be described as a natural-scene building/door-frame
example. For that paper panel, rerun the same command with a synchronized
natural-scene test directory and intentionally refresh the frozen manifest.

Quick smoke test:

```bash
python3 tools/stereo_downstream/test_rectification_disparity_visualization.py
python3 tools/stereo_downstream/run_rectification_disparity_visualization.py \
  --smoke --output result_may/stereo_rectification_disparity_smoke
```

## Epipolar-error audit

The original frozen SIFT correspondences are useful for a visualization, but a
mutual SIFT match is not an identity-preserved calibration correspondence on a
repetitive AprilTag rig. Before using `metrics.csv` as evidence, run the
independent audit:

```bash
cmake --build build --target detect_apriltag_internal -j 8
python3 tools/stereo_downstream/diagnose_rectification_epipolar_error.py
```

It writes a new directory rather than touching the frozen experiment. The
audit records the exact left/right filenames and timestamp deltas, both
directions of each stereo transform, DS project-unproject round-trip error,
synthetic 3D rectification error, all match-filter counts, frozen-SIFT
statistics, and same-AprilTag-ID/same-canonical-corner statistics. It also
tests whether inverting `T_cam1_cam0` could explain the result, exports the
per-frame essential-matrix rotation discrepancy, and renders the requested
match/error diagnostics. The C++ detector is used only to make the known-point
diagnostic; it does not run or modify calibration.

Default audit output:

```text
paper_experiments/2026_07_25_stereo_rectification_epipolar_audit/
```

Read `diagnostic_report.md` first. The original downstream P95 must not be
used as a paper claim when identity-preserved correspondences also show large
vertical error or when the input intrinsics/extrinsics do not have a
provenance-locked stereo calibration run.
