# Stereo Rectification and Disparity Visualization

This tool is a qualitative downstream experiment for DS stereo calibration. It
compares exactly two complete systems: `Kalibr` and `Ours`. Each system uses
its own left/right intrinsics and stereo extrinsic; no intrinsic/extrinsic
cross-combination is permitted.

## Peripheral Epipolar Consistency

`evaluate_peripheral_epipolar_consistency.py` is the first quantitative
downstream experiment. It evaluates complete `Kalibr` and `Ours` stereo
bundles on frozen, raw-image mutual-SIFT matches. The matching stage has no
access to either calibration. Each method then independently unprojects the
same pixel pairs, evaluates the same symmetric bearing-to-epipolar-plane angle
formula using its frozen `T_cam1_cam0`, and reports the same fixed sensor-radial
polar bands: central `[0,30)`, middle `[30,60)`, and peripheral `[60,80)` deg.
The bands are a method-independent equidistant sensor-polar proxy, so a method
cannot improve its score by changing which matches belong to the periphery.

The Ours input is guarded: its Stage6 persistent-selection summary must report
`spherical_tangent`. A pixel-refined bundle is rejected instead of silently
being included. Create the required self-contained Outer4+Internal angular
bundle as follows:

```bash
STAGE6_BA_MODE=angular scripts/run_stage6_selfcontained_ds_bundle.sh \
  image/datatset_5_1/stereo_dataset_20260430_1444190-clear \
  result_may/stage6_selfcontained_ds_angular_tangent_20260802 \
  intrintic/catalog/current_baseline/stereo_bundles/2026_08_02_stage6_selfcontained_ds_angular_tangent_1444190_stride3
```

Then run the independent evaluation. Exact capture timestamps are mandatory;
the test data must not be part of the Stage6 calibration input.

```bash
python3 tools/stereo_downstream/evaluate_peripheral_epipolar_consistency.py \
  --ours-bundle intrintic/catalog/current_baseline/stereo_bundles/2026_08_02_stage6_selfcontained_ds_angular_tangent_1444190_stride3 \
  --kalibr-camchain config/stereo_4_2-3-camchain.yaml \
  --left-dir image/mid_far_dataset/stereo_dataset_20260430_144928/left \
  --right-dir image/mid_far_dataset/stereo_dataset_20260430_144928/right \
  --calibration-left-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/left \
  --calibration-right-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right \
  --timestamp-tolerance-ms 0 \
  --output paper_experiments/2026_08_02_peripheral_epipolar_consistency_ds
```

It writes the immutable frame/match manifests, all per-match ray errors and
rectified coordinates, regional metrics, a compact LaTex table, a polar-band
curve, and a spatial peripheral overlay. It refuses to produce the final
table if any method/region has no valid angular or vertical-disparity samples.
`mid_far 144928` is an independent multi-board diagnostic sequence, not a
natural-scene result; point the same command at an explicitly frozen natural
test sequence for the paper's natural-scene claim.

Run the frozen protocol only with a provenance-locked bundle:

```bash
BUNDLE_DIR=intrintic/catalog/current_baseline/stereo_bundles/2026_08_02_stage6_selfcontained_ds_1444190_stride3 \
KALIBR_CAMCHAIN=config/stereo_4_2-3-camchain.yaml \
KALIBR_CALIBRATION_LABEL="checkerboard control" \
scripts/run_stereo_rectification_disparity_visualization.sh
```

To create a new self-contained bundle in one step, run:

```bash
scripts/run_stage6_selfcontained_ds_bundle.sh \
  image/datatset_5_1/stereo_dataset_20260430_1444190-clear \
  result_may/stage6_selfcontained_ds_20260802 \
  intrintic/catalog/current_baseline/stereo_bundles/2026_08_02_stage6_selfcontained_ds_1444190
```

The command rejects an existing nonempty bundle, verifies the two native final
camera YAMLs, and records checksums before any downstream evaluation can use
the result.

The default Kalibr camchain is the intended checkerboard-calibration control.
Its source label is written to `protocol.json`; use the same label in the paper
caption or table note, rather than describing it as a same-training-set run.

## Provenance-locked stereo bundle

Do not combine a left intrinsic YAML, right intrinsic YAML, and an external
YAML from separate calibration runs. The preferred protocol is a
self-contained Stage6 run: it creates both monocular seeds in process, jointly
optimizes left/right projection intrinsics and `T_cam1_cam0`, then exports all
three final YAMLs from the same `optimized_scene`.

## Self-contained Stage6 bundle

After a successful Stage6 run with
`--stage6-intrinsics-mode adaptive_regularized_joint_projection`, verify its
native final-camera exports and freeze them as one bundle:

```bash
python3 tools/verify_stage6_persistent_outputs.py \
  --expected-pose-structure independent_pair_board \
  --require-final-camera-yamls \
  result_may/stage6_frozen_selfcontained_ds_final_bundle_20260802

python3 tools/stereo_downstream/create_provenance_locked_stereo_bundle.py \
  --stage6-output result_may/stage6_frozen_selfcontained_ds_final_bundle_20260802 \
  --use-stage6-final-intrinsics \
  --training-left-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/left \
  --training-right-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right \
  --holdout-left-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/left \
  --holdout-right-dir image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right \
  --holdout-role within_sequence_holdout \
  --max-pair-delta-ms 0 \
  --bundle-dir intrintic/catalog/current_baseline/stereo_bundles/2026_08_02_stage6_selfcontained_ds_1444190_stride3
```

`--use-stage6-final-intrinsics` rejects historical YAML inputs and requires:
`stage6_uses_external_intrinsics=0`, active projection-intrinsics optimization,
and the two final YAMLs that match the Stage6 final summary. The holdout role
must be `within_sequence_holdout` when it is a split of the same sequence;
use `external_validation_only` only for a distinct frozen sequence.

## Fixed-intrinsics diagnostic bundle

The following legacy-style flow is retained only to isolate stereo-extrinsic
effects with fixed camera intrinsics. It is not the preferred final comparison
against Kalibr's jointly calibrated stereo system.

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

The wrapper passes `--ours-bundle-manifest`, which validates all three hashes
and requires `stage6_final_in_process` intrinsics with active Stage6 projection
optimization. Direct Python calls can use the same flag to enforce the check.

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
