# Stage5 Backend Full Experiment Summary

Baseline used in this table:
- `stage5_backend_full2_auto`

Notes:
- Baseline is available locally.
- The other full ablation rows were recorded from completed experiment outputs provided in chat and copied here as a local experiment ledger.
- Full-dataset Kalibr holdout values are abnormally large, so the safest conclusions are based on:
  - `frontend vs backend`
  - `baseline vs ablation`

## Main Table

| experiment | second_pass | intrinsics_mode | residual_gate | board_pose_gate | train_backend | holdout_backend | delta_holdout_vs_baseline | holdout_outer | holdout_internal | notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_full2_auto | 1 | delayed | 1 | 1 | 3.84853 | 6.64151 | +0.00000 | 3.20268 | 6.98910 | full delayed baseline |
| intrinsics_immediate | 1 | immediate | 1 | 1 | 3.84291 | 6.65832 | +0.01681 | 3.21588 | 7.00651 | basically tied with delayed |
| no_board_pose_gate | 1 | delayed | 1 | 0 | 4.25648 | 6.65489 | +0.01338 | 3.16398 | 7.00618 | slightly better outer, slightly worse overall |
| no_residual_gate | 1 | delayed | 0 | 1 | 7.17487 | 6.70191 | +0.06040 | 3.34209 | 7.04543 | residual gate should stay on |
| no_round2 | 0 | delayed | 1 | 1 | 3.82543 | 6.65644 | +0.01493 | 3.22221 | 7.00422 | round2 effect is small on this split |
| no_selection_gates | 1 | delayed | 0 | 0 | 10.30820 | 6.62850 | -0.01301 | 3.07936 | 6.98303 | best holdout overall here, but training degrades badly |
| pose_only | 1 | pose_only | 1 | 1 | 4.50705 | 14.00960 | +7.36809 | 35.08710 | 7.11114 | clearly fails; intrinsics must be optimized |

## Short Conclusions

- `pose_only` is decisively ruled out on full.
- `residual_sanity_gate` should remain enabled on full.
- `intrinsics_immediate` and delayed baseline are essentially tied on full.
- `round2` gives at most a very small gain on this full split.
- `board pose-fit gate` looks like a threshold-tuning problem, not a simple on/off problem.
- `no_selection_gates` is interesting as a signal, but not safe enough to replace the baseline because training quality degrades too much.
