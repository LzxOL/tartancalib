# Frozen Stage5 Baseline (v3)

`stage5_backend_frozen_v3_recovery_cache` is the default Stage5 protocol for
image-based calibration. It freezes the validated recovery and cache behavior
without freezing a dataset's physical board topology.

## Included behavior

- Direct multi-scale outer-tag detection and existing outer-corner validation.
- Camera-aware DS sphere-patch rescue after provisional outer-only camera
  initialization, followed by camera reinitialization when observations are
  recovered.
- Zero-detection atlas enabled by default. A frame with no direct decode still
  receives an exact-ID/Hamming-0 recovery attempt.
- Geometry-prior outer seeds and geometry-guided tag likelihood enabled for
  missing boards when a bootstrap/scene pose prior exists. Projected-only
  corners are never accepted as observations.
- Internal pose rescue and image-validated internal regeneration.
- One dataset-owned `--cache-dir` containing both `outer_detection_final` and
  `internal_refinement` stage artifacts. Configuration/state hashes isolate
  only the affected layer when an algorithm changes.

## Dataset-specific configuration

The YAML file remains responsible for the actual board/tag IDs and optional
costlier recovery coverage. For example, the four-board dataset uses IDs
`[1, 3, 4, 5]` and explicitly enables the extended sphere-patch atlas. Do not
copy that topology into a different dataset's baseline command.

## Explicit ablations

Use `--stage5-disable-camera-aware-outer-rescue` or
`--stage5-disable-camera-aware-sphere-patch-zero-detection` only for an
ablation. Such runs receive a deterministic protocol suffix and do not share
the frozen baseline's effective recovery configuration.

## Canonical Board5 KB profile

For the validated Board5 KB (`pinhole-equi`) workflow, use
`scripts/run_stage5_board5_kb_baseline.sh`. Its default inputs are
`board5_8_20-4`, the Board5 target configuration, a fixed 70/30 split, and
`split-seed=1337`. The profile explicitly enables:

- `--stage5-enable-frozen-recovery-baseline`;
- model-aware, eight-parameter KB information coreset selection;
- independent focal initialization; and
- global scene-state and polar-angle diagnostics; and
- reduced development overlays via `--stage5-skip-heavy-overlays`.

The skip flag suppresses the heavy per-case overlay workload. A small number of
legacy final diagnostic PNGs may still be emitted by the backend; they are not
used by selection, optimization, or reported metrics.

The script keeps seed-layout alignment and candidate-pose prefit disabled.
These are experiment controls, not part of the canonical profile. Its cache
directory is independent from the output directory; never reuse a cache across
different input datasets or algorithm profiles.

Round 2 performs the shared-layout optimization. During the later model-aware
Persistent selection, that established layout is fixed while camera intrinsics
and each candidate frame pose are evaluated. This prevents a single candidate
from compensating an intrinsics step by changing the multi-board geometry; it
does not remove the Round 2 layout optimization.

## Frozen holdout frontend

For image-based random holdout evaluation, the canonical script enables
`--stage5-external-holdout-self-frontend-prepass` by default. This runs the
holdout images through an independent frozen frontend prepass and evaluates
Ours and reference cameras on those same measured corners. It avoids deriving
holdout observations from the optimized training scene state.

Use this protocol for Board5 and other image datasets by overriding the
script's `IMAGE_DIR`, `CONFIG`, `KALIBR_CAMCHAIN`, `OUTPUT_DIR`, and
`CACHE_DIR` variables. Keep `FROZEN_HOLDOUT_FRONTEND=1` for reportable
holdout metrics. Set it to `0` only for a labeled legacy-regeneration
diagnostic; such values are not directly comparable to the frozen protocol.

## Final Five-Method Comparison

The Stage5 result is always reported as **Ours**. It is not the same result as
the original, independently-run TartanCalib pipeline. The reportable
five-method comparison is **Ours**, **TartanCalib**, **Kalibr**, **Basalt**,
and **BabelCalib**. Kalibr is the required primary reference. Supply the three
additional evaluator-ready camchains through `TARTANCALIB_CAMCHAIN`,
`BASALT_CAMCHAIN`, and `BABELCALIB_CAMCHAIN`; the script adds each as an
independent reference camera and evaluates every method on the same frozen
holdout observations. Do not use any external reference camera to initialize
Ours.

For a five-method run, the user-facing invocation should contain only paths:

```bash
IMAGE_DIR='...' CONFIG='...' OUTPUT_DIR='...' CACHE_DIR='...' \
KALIBR_CAMCHAIN='...' TARTANCALIB_CAMCHAIN='...' \
BABELCALIB_CAMCHAIN='...' BASALT_CAMCHAIN='...' \
scripts/run_stage5_board5_kb_baseline.sh
```

The script owns the fixed model, split, recovery, selection, and diagnostic
settings. The result must report equal point counts and successful pose refits
for all five methods before it is used as a comparison table.

`--stage5-model-aware-progressive-seed` is a separately validated experiment:
it improves accepted incremental batches on the currently tested KB datasets,
but remains opt-in until DS, EUCM, and UCM regressions are complete. Enable it
only by setting `PROGRESSIVE_SEED=1` when invoking the script.

## DS model validity contract

The frozen model-aware path uses the same dataset-independent lifecycle for
`ds-none` and the other camera families: the validated frozen seed uses all
valid Outer4 and internal measurements to establish camera information at the
current shared-layout state, without changing the layout or frame poses.
Candidate frames must pass the existing validity, trust-region,
residual-health, ray-validity, and rollback gates. No dataset name, frame ID,
Kalibr intrinsics, or dataset-specific acceptance threshold participates in
this path.

A run that rejects every Persistent candidate batch is not promoted to a final
shared-layout calibration. It exits unsuccessfully and does not write
`final_backend_camera.yaml`. When the initializer is valid, it additionally
writes `diagnostic_initializer_camera.yaml` and
`diagnostic_initializer_camera_summary.txt`. These preserve the independent
board-pose camera estimate and its RMSE/P95 audit, but are explicitly marked
`diagnostic_only` and `valid_for_rigid_shared_layout: 0`. This diagnostic may
be used in a separately labelled fixed-camera evaluation; it must not be
catalogued or reported as the final Stage5 result.

Do not silently switch a rejected DS run to KB. Model selection remains an
explicit experiment decision. Before declaring DS fully regression-frozen,
validate the same path on a DS-compatible rigid-layout dataset and require at
least one accepted Persistent candidate batch plus a successfully exported
final camera.

## External 6x6 AprilGrid Evaluation

Use `tools/evaluate_aprilgrid_intrinsics_catalog.py` for a reportable external
AprilGrid comparison. This is an evaluation protocol, not a Stage5 calibration
step. All cameras must be evaluated on one frozen set of measured corners:

1. Set `--detector-intrinsics-yaml` to the designated Kalibr camera. It
   initializes the frontend only and is never optimized.
2. Use the physical target's AprilGrid config with `--image-dir` and pass
   `--outer4-only`. A standard 6x6 AprilGrid has only decoded tag corners;
   generated `sphere_border_lattice` points are not physical measurements and
   must not enter this metric.
3. Leave geometry-prior rescue disabled and do not optimize any camera
   intrinsics. The evaluator independently refits only each tag pose for every
   candidate YAML.

The canonical shape is:

```bash
python3 ../tools/evaluate_aprilgrid_intrinsics_catalog.py \
  --image-dir '...' --detector-config '...' --outer4-only \
  --detector-intrinsics-yaml '...__kalibr__kb.yaml' \
  --params '...__ours__kb.yaml' '...__kalibr__kb.yaml' \
           '...__tartancalib__kb.yaml' '...__basalt__kb__camchain.yaml' \
           '...__babelcalib__kb.yaml' \
  --output '...'
```

Before reporting results, verify the generated `evaluation_protocol.json` says
`evaluation_point_scope: outer4_only`, identifies the Kalibr detector YAML,
and records shared observations across models. Do not compare a run that used
generated internal points with this protocol.
