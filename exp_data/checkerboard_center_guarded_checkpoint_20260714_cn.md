# Checkerboard Center-Guarded Checkpoint Selection (2026-07-14)

## Protocol

- Training: `checkboard_3_25_right_clear_export_11x8/all.mat`
  - 126 frames
  - 11088 uniform checkerboard control points
- Frozen test: `stereo_4_2-3_images_right_export_11x8/all.mat`
  - 49 frames
  - 4312 uniform checkerboard control points
- Frozen test is evaluation-only and is not used by initialization,
  selection, acceptance, checkpoint ranking, or parameter tuning.
- Canonical/Kalibr/BabelCalib parameters are reference-only.
- The final state comes from persistent incremental selection BA; no final BA
  is run.

## Method Change

The training-only checkerboard checkpoint selector now has two stages:

1. Detect the first checkpoint with cross-fold consensus:
   - fold-median mean improves by at least 5%;
   - worst-fold median does not regress;
   - frame median does not regress;
   - frame P90 is bounded to 1.10x initialization;
   - Huber-equivalent RMSE is bounded to 1.03x initialization.
2. Inspect the next four accepted checkpoints and replace the first consensus
   checkpoint only when:
   - at least three of five robust training metrics improve;
   - every metric remains within 1.03x of the selected checkpoint;
   - either fold-median mean or frame median improves.

If no consensus checkpoint exists, the conservative worst-fold selector is
used unchanged.

## Right Canonical Frozen-Test Results

| Model | Method | RMSE [px] | P95 [px] | Rank |
|---|---|---:|---:|---:|
| DS | Ours | **0.274020** | **0.530878** | **1 / 3** |
| DS | BabelCalib | 0.292798 | 0.575437 | 2 / 3 |
| DS | Kalibr | 0.349124 | 0.712778 | 3 / 3 |
| KB | BabelCalib | **0.292914** | 0.576194 | 1 / 4 |
| KB | Ours | 0.295013 | **0.572904** | **2 / 4** |
| KB | CamOdoCal | 0.330680 | 0.659595 | 3 / 4 |
| KB | TartanCalib | 0.341539 | 0.658440 | 4 / 4 |
| Omni-none | Ours | **0.439655** | **0.851091** | **1 / 3** |
| Omni-none | BabelCalib UCM | 0.447449 | 0.853383 | 2 / 3 |
| Omni-none | Kalibr | 0.544579 | 1.114180 | 3 / 3 |

Output directories:

- `result_may/stage5_checkerboard_final_center_guard_right_ds_126to49_20260714`
- `result_may/stage5_checkerboard_final_center_guard_right_kb_126to49_20260714`
- `result_may/stage5_checkerboard_final_center_guard_right_omni_126to49_20260714`

## Cross-Camera-Side Regression

The same selector was evaluated on the fixed left checkerboard 70/30 split:

| Model | Previous checkpoint result [px] | Final result [px] |
|---|---:|---:|
| DS | 0.369800 | 0.369800 |
| KB | 0.404521 | **0.389801** |
| Omni-none | 0.782547 | 0.782547 |

For left Omni-none, the current model still outperforms the available Kalibr
Omni-none reference on the same held-out observations (`1.520360 px`). The
selector correctly retains initialization because later checkpoints improve
central training metrics while strongly degrading the training tail.

## Canonical Outputs

- `intrintic/catalog/canonical/current_baseline/right/checkerboard-3-25-all__right__ours-baseline__ds.yaml`
- `intrintic/catalog/canonical/current_baseline/right/checkerboard-3-25-all__right__ours-baseline__kb.yaml`
- `intrintic/catalog/canonical/current_baseline/right/checkerboard-3-25-all__right__ours-baseline__omni.yaml`

All three files are reference-only. Stage5 summaries confirm:

```text
stage5_init_uses_yaml_intrinsics: 0
stage5_init_uses_kalibr_camchain_intrinsics: 0
used_config_intermediate_camera: 0
```
