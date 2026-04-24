# Stage5 Backend Experiment Summary

experiment_count: 10
baseline: stage5_backend_first20_auto

## Main Table

| experiment | second_pass | frontend_mode | backend_mode | residual_gate | board_pose_gate | train_backend | holdout_backend | holdout_outer | holdout_internal | delta_holdout_vs_baseline | notes |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| intrinsics_immediate | 1 | immediate | immediate | 1 | 1 | 4.08294 | 6.12397 | 4.10242 | 6.35389 | -0.00582 | failed_iters=9 |
| first20_auto |  |  |  |  |  | 4.08271 | 6.12979 | 4.08277 | 6.36203 | +0.00000 | failed_iters=9 |
| first20_stable |  |  |  |  |  | 4.08271 | 6.12979 | 4.08277 | 6.36203 | +0.00000 | failed_iters=9 |
| first20_fix |  |  |  |  |  | 4.08271 | 6.12980 | 4.08275 | 6.36204 | +0.00001 | not_fair,diagnostic_only,failed_iters=9 |
| no_residual_gate | 1 | delayed | delayed | 0 | 1 | 4.08239 | 6.13054 | 4.08194 | 6.36293 | +0.00075 | failed_iters=9 |
| no_round2 | 0 | delayed | delayed | 1 | 1 | 4.11308 | 6.17425 | 4.04411 | 6.41372 | +0.04446 | failed_iters=9 |
| no_board_pose_gate | 1 | delayed | delayed | 1 | 0 | 5.11335 | 6.20104 | 2.86284 | 6.52953 | +0.07125 | failed_iters=9 |
| no_selection_gates | 1 | delayed | delayed | 0 | 0 | 4.53587 | 6.33886 | 2.40117 | 6.70574 | +0.20907 | failed_iters=9 |
| pose_only | 1 | pose_only | pose_only | 1 | 1 | 4.98057 | 7.68775 | 7.80728 | 7.67108 | +1.55796 | failed_iters=8 |
| first20 |  |  |  |  |  | 4.95246 | 7.78300 | 8.07162 | 7.74209 | +1.65321 | not_fair,diagnostic_only,failed_iters=11 |

## Selection Scale

| experiment | round1_boards | round2_boards | round2_internal_points | auto_rmse | auto_fallback |
|---|---:|---:|---:|---:|---:|
| intrinsics_immediate | 30 | 30 | 877 | 3.01104 | 0 |
| first20_auto |  |  |  | 3.01104 | 0 |
| first20_stable |  |  |  |  |  |
| first20_fix |  |  |  |  |  |
| no_residual_gate | 30 | 30 | 877 | 3.01104 | 0 |
| no_round2 | 30 | 0 | 0 | 3.01104 | 0 |
| no_board_pose_gate | 45 | 43 | 1229 | 3.01104 | 0 |
| no_selection_gates | 43 | 43 | 1247 | 3.01104 | 0 |
| pose_only | 30 | 30 | 877 | 3.01104 | 0 |
| first20 |  |  |  |  |  |

## Optimization

| experiment | initial_backend_rmse | optimized_backend_rmse | initial_cost | optimized_cost | failed_iterations_total |
|---|---:|---:|---:|---:|---:|
| intrinsics_immediate | 4.15142 | 3.66237 | 464.17100 | 283.02200 | 9 |
| first20_auto | 4.17822 | 3.65969 | 464.54000 | 283.09500 | 9 |
| first20_stable | 4.17822 | 3.65969 | 464.54000 | 283.09500 | 9 |
| first20_fix | 4.17822 | 3.65970 | 464.54000 | 283.09300 | 9 |
| no_residual_gate | 4.17427 | 3.65941 | 463.64900 | 283.04600 | 9 |
| no_round2 | 4.64951 | 3.66129 | 558.14100 | 281.60400 | 9 |
| no_board_pose_gate | 5.61077 | 3.83194 | 1426.32000 | 589.24300 | 9 |
| no_selection_gates | 4.94036 | 3.09697 | 1237.31000 | 454.10000 | 9 |
| pose_only | 4.39265 | 3.95733 | 481.82400 | 415.81700 | 8 |
| first20 | 4.17822 | 4.17822 |  |  | 11 |
