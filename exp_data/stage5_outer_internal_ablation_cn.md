# Stage5 内角点 / 外角点消融实验记录

本文记录 Stage5 中 **outer-only vs with-internal** 的消融实验。目标是判断：在当前 New Baseline 管线下，加入 regenerated internal points 是否相对只使用外四角点带来更稳定的跨数据集外部验证表现。

## 1. 实验定义

### New Baseline

New Baseline 指当前冻结的 `current` 配置：

- 使用 outer detection；
- 使用 outer-only DS auto-init；
- 使用 multi-board outer bootstrap；
- 使用 Round1 / Round2 internal regeneration；
- backend 输入同时包含 outer + internal points；
- 启用 Kalibr-style incremental frame-board trial selection；
- 启用 pre-backend point filter。

### outer-only ablation

outer-only 是严格消融，不是只在 final backend 关掉 internal，而是整个训练/优化链路都不让 internal 参与：

- `--include-internal-points 0`
- `--disable-second-pass`
- Round1 internal regeneration 跳过；
- Round2 跳过；
- final backend 只使用 outer residual；
- backend problem 中 internal point count 必须为 0。

### Kalibr reference

Kalibr reference 仍然是外部棋盘格 camchain 在同一 Stage5 evaluator 下的评估结果，不是 Kalibr 在当前多板 AprilTag 数据上重新标定的结果。

## 2. 实验结果

| Split | New Backend 输入 | outer-only Backend 输入 | New Baseline holdout | outer-only holdout | Kalibr holdout | outer-only - New | outer-only - Kalibr | outer-only status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `144928 -> 134853` | 39F / 115B / 460O / 3347I | 21F / 97B / 388O / 0I | 3.29735 | 3.28135 | 3.31184 | -0.01600 | -0.03049 | success |
| `134853 -> 144419` | 30F / 95B / 380O / 2728I | 14F / 69B / 276O / 0I | 1.94433 | 1.97010 | 1.98124 | +0.02577 | -0.01114 | success |
| `141444 -> 140151` | 45F / 132B / 528O / 3857I | 29F / 116B / 464O / 0I | 6.11854 | 6.12580 | 6.16768 | +0.00726 | -0.04188 | success |
| `144419 -> 144928` | 30F / 78B / 312O / 2223I | 12F / 60B / 240O / 0I | 6.57076 | 6.56313 | 6.58801 | -0.00763 | -0.02488 | success |
| `191538 -> 192347` | 31F / 83B / 332O / 2417I | 13F / 65B / 260O / 0I | 3.75082 | 3.83290 | 3.87601 | +0.08208 | -0.04311 | success |
| `192347 -> 191538` | 35F / 105B / 420O / 2990I | failed before backend input | 2.89692 | N/A | 2.94125 | N/A | N/A | failed: optimized residuals did not improve |

说明：

- `outer-only - New < 0` 表示 outer-only 比 New Baseline 更低。
- `outer-only - Kalibr < 0` 表示 outer-only 比 Kalibr reference 更低。
- `192347 -> 191538` 的 outer-only 后端没有成功，不能纳入 RMSE 均值对比。
- `F / B / O / I` 分别表示进入 final backend 的 frame count、board observation count、outer point count、internal point count。

## 3. 结果解读

### outer-only 已经很强

在 5 组成功的 outer-only 实验中，outer-only 全部优于 Kalibr reference。这说明当前 outer detection、outer-only DS auto-init、multi-board bootstrap 和 outer-only backend 本身已经能形成可用的标定链路。

### internal points 提升整体稳定性

New Baseline 在 5 组可比成功实验中赢了 3 组：

- `134853 -> 144419`
- `141444 -> 140151`
- `191538 -> 192347`

outer-only 赢了 2 组：

- `144928 -> 134853`
- `144419 -> 144928`

其中 `144419 -> 144928` 的优势只有 `0.00763 px`，非常小；`144928 -> 134853` 的优势为 `0.01600 px`，也属于小幅差异。

### outer-only 鲁棒性边界更窄

`192347 -> 191538` 中 outer-only 失败，失败原因是：

`optimized residuals did not improve`

这说明只靠 outer points 时，某些 split 下后端优化更容易进入不稳定状态。New Baseline 在同一 split 下成功，并且 holdout overall 为 `2.89692`。

## 4. 当前结论

这组消融不支持“internal points 每组都显著改善”的强结论，但支持更稳妥的结论：

- outer-only 链路已经很强，是当前方法的可靠基础；
- regenerated internal points 在多数 split 上改善或保持 holdout 表现；
- internal points 让 New Baseline 的成功率和整体稳定性更好；
- internal points 的收益依赖质量控制，后续仍需要继续研究 internal observation quality / covariance / weighting。

推荐汇报说法：

> 外四角点已经可以独立支撑一个可用的 DS 多板标定链路；在此基础上加入 regenerated internal points 后，New Baseline 在多数跨数据集验证中进一步提升或保持精度，并避免了某些 outer-only split 的优化失败。因此 internal regeneration 的价值主要体现在提供额外约束和提升整体稳定性，而不是在每个数据集上都带来单调的大幅 RMSE 降低。

## 5. 对应输出目录

outer-only 输出目录：

- `result_may/ablation_new_baseline_144928_right_val_134853_right_outer_only`
- `result_may/ablation_outer_only_134853_right_val_144419`
- `result_may/ablation_outer_only_141444_right_val_140151`
- `result_may/ablation_outer_only_144419_right_val_144928`
- `result_may/ablation_outer_only_191538_right_val_192347`
- `result_may/ablation_outer_only_192347_right_val_191538`

New Baseline 对应目录：

- `result_may/stage5_rerun_current_144928_right_val_134853_right_with_kalibr_compare`
- `result_may/stage5_rerun_current_134853_right_val_144419_right_with_kalibr_compare`
- `result_may/stage5_rerun_current_141444_right_val_140151_with_kalibr_compare`
- `result_may/stage5_rerun_current_144419_right_val_144928_right_with_kalibr_compare`
- `result_may/stage5_rerun_current_191538_right_val_192347_with_kalibr_compare`
- `result_may/stage5_rerun_current_192347_right_val_191538_with_kalibr_compare`
