# Stage5 Residual Baseline (2026-07-19)

This is the frozen three-profile DS residual baseline for the 1444190-clear
right-camera experiment. It supersedes earlier residual smoke tables for this
protocol.

Protocol:

```text
dataset: stereo_dataset_20260430_1444190-clear/right
model: ds-none
split: random_holdout_ratio
train/holdout: 70% / 30%
split_seed: 1337
training frames: 38
holdout frames: 16
backend input: 26 frames / 130 board observations / 4330 points
holdout points: 2689
```

The three frozen profiles are Pixel-only, tangent-plane Angular, and optional
Pixel-Ray Hybrid refinement. Their archival files are in:

```text
paper/experiments/fixed_backend_residual_ablation_seed1337/frozen_baseline_20260719
```

The reproducible suite command is:

```bash
scripts/run_stage5_frozen_residual_baseline.sh
```

Its default is `PROFILE=all`, which runs the three profiles separately over
the same fixed backend manifest. Hybrid is not injected into Pixel BA; it is a
separate frozen profile enabled after Pixel persistent BA. The low-level
`run_stage5_backend` CLI continues to default to Pixel-only.

The unified result table is `results.csv` in the frozen profile directory.
