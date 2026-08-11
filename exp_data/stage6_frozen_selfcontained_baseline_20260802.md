# Stage6 Frozen Self-Contained DS Baseline (2026-08-02)

## Scope

This baseline calibrates a stereo rig from the supplied left and right image
sequences. Each camera obtains its own seed from image observations through the
Stage5 monocular frontend. Stage6 then initializes and refines the stereo
extrinsic with persistent incremental BA.

## Frozen Protocol

| Item | Value |
| --- | --- |
| Camera family | `ds-none` |
| Intrinsics provenance | `stage6_auto_left/right_monocular_frontend` |
| External intrinsics / camchain | Disabled; required summary value is `stage6_uses_external_intrinsics: 0` |
| Frame pairing | `exact_timestamp` |
| Holdout | Every third paired frame, controlled by `--holdout-offset` |
| Measurement source | `all_valid` |
| Stereo pose structure | `independent_pair_board` |
| Intrinsics policy | `adaptive_regularized_joint_projection` |
| Main optimizer | Persistent incremental stereo BA |
| Final global BA | Disabled |
| Selection residual | Pixel reprojection residual |

## Reference Result

Dataset: `stereo_dataset_20260430_1444190-clear`, `holdout-offset=0`.

| Metric | Value |
| --- | ---: |
| Paired frames | 29 |
| Training / holdout pairs | 19 / 10 |
| Training RMSE | 0.965557 px |
| Extrinsic-only holdout RMSE | 0.827768 px |
| Baseline length | 65.339 mm |

Reference artifacts: `result_may/stage6_selfcontained_ds_full_20260802`.

## Invocation

Run `scripts/run_stage6_frozen_selfcontained_ds_baseline.sh`. Its optional
arguments are dataset root, output directory, and holdout offset, in that order.
