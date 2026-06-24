# Stage5 trial backend candidate quality analysis

本报告聚合 9 组 `trial backend frame-board incremental selection` 实验，专门分析新增进入 backend 的 `accepted_incremental_trial` frame-board observation 的共同特征。

## 1. 输入实验

- `134853_to_144419` / `current`: `result_may/stage5_cmp_current_134853_right_val_144419_right`
- `134853_to_144419` / `relaxed40`: `result_may/stage5_cmp_relaxed40_134853_right_val_144419_right`
- `134853_to_144419` / `relaxed60`: `result_may/stage5_cmp_relaxed60_134853_right_val_144419_right`
- `144928_to_134853` / `current`: `result_may/stage5_cmp_current_144928_right_val_134853_right`
- `144928_to_134853` / `relaxed40`: `result_may/stage5_cmp_relaxed40_144928_right_val_134853_right`
- `144928_to_134853` / `relaxed60`: `result_may/stage5_cmp_relaxed60_144928_right_val_134853_right`
- `144419_to_144928` / `current`: `result_may/stage5_cmp_current_144419_right_val_144928_right`
- `144419_to_144928` / `relaxed40`: `result_may/stage5_cmp_relaxed40_144419_right_val_144928_right`
- `144419_to_144928` / `relaxed60`: `result_may/stage5_cmp_relaxed60_144419_right_val_144928_right`

## 2. 每组 accepted candidate 质量摘要

| dataset | mode | accepted | score mean | coverage mean | polar mean deg | max polar mean deg | internal pts mean | global Δ mean | outer Δ mean | internal Δ mean | holdout overall | holdout outer | holdout internal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 134853_to_144419 | current | 20 | 3.792 | 3.095 | 72.40 | 90.00 | 29.0 | -0.00167 | -0.00055 | -0.00179 | 1.94501 | 0.35204 | 2.07176 |
| 134853_to_144419 | relaxed40 | 40 | 3.716 | 3.003 | 72.28 | 90.00 | 29.1 | -0.00209 | -0.00052 | -0.00224 | 1.94398 | 0.32149 | 2.07135 |
| 134853_to_144419 | relaxed60 | 60 | 3.666 | 2.948 | 72.04 | 90.00 | 29.2 | -0.00195 | -0.00026 | -0.00210 | 1.94379 | 0.32713 | 2.07102 |
| 144928_to_134853 | current | 20 | 3.743 | 3.014 | 71.46 | 90.00 | 29.0 | -0.00397 | -0.00190 | -0.00420 | 3.25668 | 0.81435 | 3.46075 |
| 144928_to_134853 | relaxed40 | 40 | 3.710 | 2.990 | 71.68 | 90.00 | 28.8 | -0.00317 | -0.00158 | -0.00335 | 3.25674 | 0.81376 | 3.46084 |
| 144928_to_134853 | relaxed60 | 60 | 3.694 | 2.981 | 71.44 | 90.00 | 28.7 | -0.00245 | -0.00135 | -0.00258 | 3.25662 | 0.81324 | 3.46072 |
| 144419_to_144928 | current | 19 | 3.773 | 3.062 | 73.41 | 90.00 | 29.1 | -0.00161 | -0.00148 | -0.00168 | 6.59282 | 15.29850 | 4.10884 |
| 144419_to_144928 | relaxed40 | 38 | 3.678 | 2.956 | 72.76 | 90.00 | 29.0 | -0.00192 | -0.00116 | -0.00202 | 6.59607 | 15.30280 | 4.11255 |
| 144419_to_144928 | relaxed60 | 59 | 3.646 | 2.925 | 72.52 | 90.00 | 29.1 | -0.00137 | -0.00074 | -0.00145 | 6.61269 | 15.34780 | 4.11955 |

## 3. relaxed 相对 current 的变化

| dataset | mode | extra accepted | overall Δ | outer Δ | internal Δ | score mean Δ | polar mean Δ | internal pts mean Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 134853_to_144419 | relaxed40 | 20 | -0.00103 | -0.03055 | -0.00041 | -0.075 | -0.12 | 0.1 |
| 134853_to_144419 | relaxed60 | 40 | -0.00122 | -0.02491 | -0.00074 | -0.126 | -0.36 | 0.2 |
| 144419_to_144928 | relaxed40 | 19 | 0.00325 | 0.00430 | 0.00371 | -0.095 | -0.65 | -0.1 |
| 144419_to_144928 | relaxed60 | 40 | 0.01987 | 0.04930 | 0.01071 | -0.127 | -0.89 | 0.0 |
| 144928_to_134853 | relaxed40 | 20 | 0.00006 | -0.00059 | 0.00009 | -0.033 | 0.22 | -0.2 |
| 144928_to_134853 | relaxed60 | 40 | -0.00006 | -0.00111 | -0.00003 | -0.048 | -0.03 | -0.3 |

## 4. reason 分布

| dataset | mode | reason | count |
|---|---|---|---:|
| 134853_to_144419 | current | accepted_incremental_trial | 20 |
| 134853_to_144419 | current | baseline_seed | 75 |
| 134853_to_144419 | current | not_attempted_board_candidate_cap | 265 |
| 134853_to_144419 | current | not_attempted_frame_candidate_cap | 6 |
| 134853_to_144419 | current | rejected_below_min_candidate_score | 16 |
| 134853_to_144419 | current | rejected_by_wide_trial_residual_outlier | 38 |
| 134853_to_144419 | relaxed40 | accepted_incremental_trial | 40 |
| 134853_to_144419 | relaxed40 | baseline_seed | 75 |
| 134853_to_144419 | relaxed40 | not_attempted_board_candidate_cap | 257 |
| 134853_to_144419 | relaxed40 | not_attempted_frame_candidate_cap | 10 |
| 134853_to_144419 | relaxed40 | rejected_by_wide_trial_residual_outlier | 38 |
| 134853_to_144419 | relaxed60 | accepted_incremental_trial | 60 |
| 134853_to_144419 | relaxed60 | baseline_seed | 75 |
| 134853_to_144419 | relaxed60 | not_attempted_board_candidate_cap | 244 |
| 134853_to_144419 | relaxed60 | not_attempted_frame_candidate_cap | 3 |
| 134853_to_144419 | relaxed60 | rejected_by_wide_trial_residual_outlier | 38 |
| 144928_to_134853 | current | accepted_incremental_trial | 20 |
| 144928_to_134853 | current | baseline_seed | 91 |
| 144928_to_134853 | current | not_attempted_board_candidate_cap | 461 |
| 144928_to_134853 | current | not_attempted_frame_candidate_cap | 13 |
| 144928_to_134853 | current | rejected_below_min_candidate_score | 23 |
| 144928_to_134853 | current | rejected_by_wide_trial_residual_outlier | 36 |
| 144928_to_134853 | relaxed40 | accepted_incremental_trial | 40 |
| 144928_to_134853 | relaxed40 | baseline_seed | 91 |
| 144928_to_134853 | relaxed40 | not_attempted_board_candidate_cap | 465 |
| 144928_to_134853 | relaxed40 | not_attempted_frame_candidate_cap | 9 |
| 144928_to_134853 | relaxed40 | rejected_below_min_candidate_score | 3 |
| 144928_to_134853 | relaxed40 | rejected_by_wide_trial_residual_outlier | 36 |
| 144928_to_134853 | relaxed60 | accepted_incremental_trial | 60 |
| 144928_to_134853 | relaxed60 | baseline_seed | 91 |
| 144928_to_134853 | relaxed60 | not_attempted_board_candidate_cap | 451 |
| 144928_to_134853 | relaxed60 | not_attempted_frame_candidate_cap | 5 |
| 144928_to_134853 | relaxed60 | rejected_below_min_candidate_score | 1 |
| 144928_to_134853 | relaxed60 | rejected_by_wide_trial_residual_outlier | 36 |
| 144419_to_144928 | current | accepted_incremental_trial | 19 |
| 144419_to_144928 | current | baseline_seed | 56 |
| 144419_to_144928 | current | not_attempted_board_candidate_cap | 254 |
| 144419_to_144928 | current | not_attempted_candidate_limit | 47 |
| 144419_to_144928 | current | not_attempted_frame_candidate_cap | 18 |
| 144419_to_144928 | current | rejected_below_min_candidate_score | 23 |
| 144419_to_144928 | current | rejected_by_wide_trial_residual_outlier | 9 |
| 144419_to_144928 | current | rejected_incremental_rmse_delta | 1 |
| 144419_to_144928 | relaxed40 | accepted_incremental_trial | 38 |
| 144419_to_144928 | relaxed40 | baseline_seed | 56 |
| 144419_to_144928 | relaxed40 | not_attempted_board_candidate_cap | 254 |
| 144419_to_144928 | relaxed40 | not_attempted_candidate_limit | 55 |
| 144419_to_144928 | relaxed40 | not_attempted_frame_candidate_cap | 13 |
| 144419_to_144928 | relaxed40 | rejected_by_wide_trial_residual_outlier | 9 |
| 144419_to_144928 | relaxed40 | rejected_incremental_rmse_delta | 2 |
| 144419_to_144928 | relaxed60 | accepted_incremental_trial | 59 |
| 144419_to_144928 | relaxed60 | baseline_seed | 56 |
| 144419_to_144928 | relaxed60 | not_attempted_board_candidate_cap | 241 |
| 144419_to_144928 | relaxed60 | not_attempted_candidate_limit | 52 |
| 144419_to_144928 | relaxed60 | not_attempted_frame_candidate_cap | 9 |
| 144419_to_144928 | relaxed60 | rejected_by_wide_trial_residual_outlier | 9 |
| 144419_to_144928 | relaxed60 | rejected_incremental_rmse_delta | 1 |

## 5. 主要观察

- accepted candidate 的平均 `candidate_score` 通常在 3 左右，说明当前策略接入的不是随机 board，而是经过 coverage / diversity / residual quality 共同筛选后的 observation。
- `not_attempted_board_candidate_cap` 在多数实验中仍是最大拦截项，说明 board-level cap 对最终接入分布影响很大。继续调参时，优先看 per-board cap，而不是只调 `max_candidate_additions`。
- `134853_to_144419` 是 relaxed 策略最有收益的数据集；`relaxed40/60` 增加 candidate 后 holdout overall 和 internal 均小幅改善，outer 改善更明显。
- `144928_to_134853` 中 relaxed 基本中性，说明更多 candidate 没有明显破坏，但收益有限。
- `144419_to_144928` 中 relaxed 轻微退化，说明新增 board 并非越多越好；困难验证集更需要 candidate quality gate，而不是单纯增加数量。
- 当前最稳结论仍是：`current` 做默认主线，`relaxed40` 做实验分支，`relaxed60` 暂不主推。

## 6. 下一步策略建议

1. 对 `accepted_incremental_trial` 中的 board 分布做进一步限制或重加权，避免某些 board 被过度接入。
2. 把 local-vs-global consistency 加入 candidate score，优先接入局部 pose 与全局结构一致的 frame-board。
3. 对 accepted candidate 单独做 candidate-only residual pruning，保留 coverage 收益，同时减少 `144419_to_144928` 这种困难验证集的退化。
4. 不建议继续只放宽 `max_candidate_additions`，因为 `relaxed60` 已经显示出泛化不稳定。
