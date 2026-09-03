# Frozen Residual Baseline Profiles

Frozen on 2026-07-19 for the DS residual ablation on
`stereo_dataset_20260430_1444190-clear/right`.

All three profiles use the same:

- random holdout ratio: `0.30`
- split seed: `1337`
- training frames: `38`
- holdout frames: `16`
- backend frame-board manifest: `manifests/1444190clear_pixel_backend.csv`
- backend input: `26` frames, `130` board observations, `4330` points
- holdout evaluation: `2689` points
- camera model: `ds-none`
- camera initialization: Stage5 automatic outer-only initialization
- final BA: disabled in the standard backend path

Profiles:

- `pixel/`: `pixel_only`, standard Pixel persistent incremental BA.
- `angular/`: `sphere_angular`, tangent-plane component-wise angular BA.
- `hybrid/`: Pixel persistent incremental BA followed by the explicitly enabled
  4D Pixel-Ray refinement with `lambda=0.5`.

The Hybrid refinement is committed only when its training objective is finite and
non-increasing. Holdout data is not used for scale computation or commit decisions.

The ordinary low-level Stage5 CLI default remains Pixel-only with Hybrid
refinement disabled. The frozen baseline suite entry point is
`scripts/run_stage5_frozen_residual_baseline.sh`; its default `PROFILE=all`
runs these three frozen profiles separately. They are baseline variants, not
three simultaneously enabled objectives.

The exact defaults are also recorded in `baseline_defaults.env`.

The copied summaries and manifests are the archival record. Original full run
directories remain under:

```text
result_may/stage5_residual_ablation_20260719_1444190clear_pixel
result_may/stage5_residual_ablation_20260719_1444190clear_angular
result_may/stage5_residual_ablation_20260719_1444190clear_hybrid
```
