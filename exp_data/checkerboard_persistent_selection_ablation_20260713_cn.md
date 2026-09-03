# Checkerboard Persistent Selection Ablation (2026-07-13)

## Protocol

- Camera model: Double Sphere (`ds-none`)
- Training observations: `checkboard_3_25_right_clear_export_11x8/all.mat`
- Frozen-test observations: `stereo_4_2-3_images_right_export_11x8/all.mat`
- Training set: 126 frames, 11,088 control points
- Frozen test: 49 frames, 4,312 control points
- Each checkerboard corner is an independent unit-covariance measurement.
- All variants use the same initialization, candidate order, observations, and optimizer.
- Frozen-test evaluation is diagnostic only and never participates in selection,
  acceptance, rollback, stopping, or checkpoint choice.

## Main Results

| Variant | Kept views | Train RMSE (px) | Train P95 (px) | Frozen-test RMSE (px) | Frozen-test P95 (px) |
|---|---:|---:|---:|---:|---:|
| Initialization only | - | 1.19092 | 1.05939 | 0.331272 | 0.670154 |
| Persistent information selection (`miTol=0.2`) | 29 / 126 | 1.20252 | 1.13950 | 0.409182 | 0.847943 |
| All-view incremental BA | 126 / 126 | 1.19103 | **1.05009** | **0.322976** | **0.651531** |
| Kalibr DS, same training source | - | 1.19317 | 1.07404 | 0.349124 | 0.712778 |
| BabelCalib DS, same training source | - | 1.19322 | 1.07120 | 0.292798 | 0.575437 |

All variants have 100% pose-refit success on the 49 frozen-test frames.

## Camera Parameters

Parameter order: `xi, alpha, fu, fv, cu, cv`.

```text
Initialization only:
-0.181480, 0.618708, 1183.80, 1183.71, 2253.20, 2271.41

Persistent information selection:
-0.183487766871, 0.616711306828,
1178.35105251, 1178.37974502, 2253.68242183, 2271.64373848

All-view incremental BA:
-0.183871020129, 0.617769235864,
1180.10598550, 1180.05682735, 2253.46813597, 2271.31195707

Kalibr DS:
-0.179245, 0.619222, 1186.72, 1186.74, 2253.10, 2269.88

BabelCalib DS:
-0.188703, 0.615926, 1172.93, 1172.91, 2255.33, 2271.02
```

## Interpretation

The camera initializer is not the primary failure in this experiment. The
initialization-only camera already generalizes better than the default
information-selected result. Default information selection accepts 28 of 125
candidate batches and moves both full-training and frozen-test metrics in the
wrong direction. In contrast, all-view incremental BA improves frozen-test
RMSE and P95 over initialization-only and the same-source Kalibr DS baseline.

Changing checkerboard residual weighting from per-view normalization (`1/88`
per point) to independent unit-covariance measurements is required for Kalibr
objective parity, but it does not change the accepted subset. Incremental
information gain is a difference of log singular-value sums, so a uniform
residual scale cancels between consecutive states. The remaining issue is the
selected subset and candidate ordering/stopping behavior, not residual scale.

Although an intermediate accepted checkpoint reaches a lower frozen-test
error, selecting it using frozen-test RMSE would leak test data and is not a
valid calibration protocol.

## Reproducible Outputs

- Information selection:
  `result_may/stage5_checkerboard_ablation_information_selection_unit_corner_weight_20260713`
- All-view incremental BA:
  `result_may/stage5_checkerboard_ablation_all_valid_views_unit_corner_weight_20260713`
- Per-accepted-batch diagnostics:
  `persistent_camera_checkpoint_evaluations.csv` in each output directory
- Per-candidate optimizer and decision diagnostics:
  `trial_backend_frame_board_selection_decisions.csv` in each output directory

