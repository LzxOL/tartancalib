# Stage5 Trial Backend Frame-Board Selection 实验记录

本文记录 Stage5 中 `trial backend frame-board incremental selection` 这条实验线。它的核心目标是：在旧 Baseline 已经选中的观测之外，判断还能不能把更多可靠的 frame-board observation 接回 backend，从而增加多板覆盖、视场覆盖和边缘信息，同时不把坏观测带进最终 BA。

从当前实验结论出发，本文档固定以下命名：

- **Old Baseline / 旧 Baseline**：早期 strict selection 口径，作为历史对照。
- **New Baseline / 新 Baseline**：`current`，即 Kalibr-style incremental frame-board trial selection，作为后续 Stage5 默认对比主线。
- **Consistency-Soft Enhancement / 一致性软增强分支**：原 `v3-soft`，只作为可选增强参数，不作为默认 baseline。
- **relaxed40 / relaxed60**：更多 candidate 接回的覆盖率 ablation。

## 1. 为什么做这条线

Old Baseline 的 strict selection 比较保守：如果某个 board 的 internal generation / residual sanity 不够好，这个 frame-board observation 往往会被整块丢掉。这样做稳定，但代价是很多本来有用的 board 覆盖被浪费，尤其在多 board、大视场鱼眼场景中，少用一块 board 可能就少了一片视场约束。

因此我们参考 Kalibr 的思想：不要一次性相信所有候选观测，也不要只靠前端 hard gate 决定命运，而是让候选观测先进入 trial backend pool，经过短程 backend / residual 检查后，再决定是否保留。

一句话概括：

> 先保住 baseline 的稳定 seed，再把额外 frame-board observation 一个个试着接回来，能改善或不破坏优化的就保留。

## 2. Old Baseline：旧严格筛选 baseline

### 目的

Old Baseline 是历史对照组。它强调安全：宁可少用观测，也不把明显失败的 board observation 混入 backend。

### 方法

- 使用 Stage5 完整 baseline 流程。
- Round2 后只保留通过 strict selection 的 frame-board observation。
- 不做额外 candidate 接回。
- 不做 trial backend 增量选择。

### 解决的问题

它主要解决“坏 internal / 坏 board observation 污染 backend”的问题，但会牺牲多板覆盖率。

## 3. New Baseline：保守增量接回

### 目的

验证 baseline seed 之外的候选 frame-board 是否可以安全接回 backend，并观察多板覆盖增加后是否改善 external validation。

### 方法

1. 先保留 Old Baseline 已选中的 seed bundle。
2. 从完整 Round2 measurement 中构建 candidate pool。
3. 对每个 frame-board candidate 计算 score，主要考虑：
   - outer/internal point count；
   - coverage gain；
   - board diversity；
   - 当前 residual 质量。
4. 按 score 尝试增量加入候选。
5. 每次加入后跑 short trial backend。
6. 如果 global / outer / internal RMSE 退化超过阈值，就拒绝该 candidate。
7. 最终保留的 bundle 再进入 final backend。

### 配置

| 配置项 | New Baseline |
|---|---:|
| max_candidate_additions | 20 |
| min_candidate_score | 3.2 |
| min_coverage_gain | 2.5 |
| max_accepted_per_board | 4 |
| max_accepted_per_frame | 1 |

### 解决的问题

New Baseline 针对的是“Old Baseline 太保守，coverage 不够”的问题，但仍然保持较强安全边界。它对应代码/实验目录中的 `current` 配置；综合所有实验后，我们将 `current` 冻结为 **New Baseline**。

## 4. relaxed40 / relaxed60：放宽接回数量

### 目的

测试如果接回更多候选 board，是否能进一步提升泛化效果。

### 方法

与 New Baseline 相同，只是放宽 candidate 数量和 diversity cap，让更多 frame-board 有机会进入 backend。

| 配置项 | New Baseline | relaxed40 | relaxed60 |
|---|---:|---:|---:|
| max_candidate_additions | 20 | 40 | 60 |
| min_candidate_score | 3.2 | 2.8 | 2.4 |
| min_coverage_gain | 2.5 | 2.0 | 1.5 |
| max_accepted_per_board | 4 | 8 | 12 |
| max_accepted_per_frame | 1 | 2 | 3 |

### 解决的问题

这两组主要验证“更多 coverage 是否更好”。实验结果显示：更多不一定更好，尤其困难验证集上过度放宽可能轻微退化。

## 5. Consistency-Soft Enhancement：加入 local-vs-global consistency

### 目的

前面的 New Baseline / relaxed 主要看 coverage 和 residual，但没有显式判断“这个 board observation 的局部 pose 是否和全局多板结构一致”。因此我们加入 consistency score，希望避免把结构上不一致的 candidate 接回 backend。

### 方法

对每个候选 frame-board：

1. 用该 board 的 outer points 单独做 local pose refit，得到 `T_cam_board_local`。
2. 用当前全局多板结构得到 `T_cam_board_global = T_cam_reference * T_reference_board`。
3. 比较 local 和 global 的差异：
   - translation error；
   - rotation error；
   - local outer RMSE。
4. 把 consistency 作为 candidate score 的惩罚项。

### hard v3 与 Consistency-Soft Enhancement 区别

| 方案 | 方法 | 直观含义 |
|---|---|---|
| hard v3 | consistency score + 硬阈值 | 超过阈值直接拒绝 |
| Consistency-Soft Enhancement | 只作为 score penalty | 不直接拒绝，只降低排序优先级 |

hard v3 使用的硬阈值：

| 阈值 | 数值 |
|---|---:|
| max translation error | 8 mm |
| max rotation error | 5 deg |
| max local outer RMSE | 5 px |

### 解决的问题

Consistency-Soft Enhancement 主要针对“某些 candidate 从点级 residual 看不一定极端，但整体 board pose 和全局多板结构不一致”的问题。它是 New Baseline 的可选增强分支，不作为默认方案。

## 6. 三组 dataset_5_1 cross-dataset 结果

下面表格分成两层：

1. **Old Baseline 对比**：对应前期 `stage5_experiment_summary_cn.md` 中记录的旧严格筛选 baseline，用来说明我们最初遇到的问题和改进空间。
2. **当前代码 rerun 对比**：使用当前代码、同一 evaluator、同一 split 同时跑当前 strict 配置 / New Baseline / Kalibr，用来说明当前 selection / frontend / backend input 演进后的真实状态。

需要注意：Kalibr reference 指外部棋盘格 camchain 在同一 Stage5 evaluator 下的评估结果，不是 Kalibr 在当前多板 AprilTag 数据上重新标定。

### 6.0 Old Baseline vs New Baseline

这张表用于说明算法探索过程中的整体收益。Old Baseline 是早期 strict selection 口径；New Baseline 是加入改进后的 selection / frontend / backend input 管线后的默认策略，对应实验配置名 `current`。

| 数据集 | Old Baseline training | New Baseline training | training 变化 | Old Baseline holdout | New Baseline holdout | holdout 变化 |
|---|---:|---:|---:|---:|---:|---:|
| `134853 -> 144419` | 1.1330 | 0.807135 | -0.325865 | 2.1105 | 1.94433 | -0.16617 |
| `144928 -> 134853` | 1.8448 | 6.30290 | +4.45810 | 3.4419 | 3.29735 | -0.14455 |
| `144419 -> 144928` | 0.6771 | 0.690635 | +0.013535 | 6.8392 | 6.57076 | -0.26844 |

这说明：selection / frontend / backend input 的演进对 **holdout RMSE** 是稳定有收益的，三组 cross-dataset 都下降。但 training RMSE 不一定同步下降，尤其 `144928 -> 134853` 当前 training RMSE 变大，说明 New Baseline 不是简单“更拟合训练集”，而是更偏向改善外部验证泛化。

通俗地讲：我们不是把训练集压得更低来换结果，而是通过更合理的观测选择，让外部验证更稳。

### 6.1 当前代码同口径 rerun：strict 配置 / New Baseline / Kalibr

为了确认 Kalibr 对比口径没有变化，我们用当前代码重新跑了 strict 配置和 New Baseline。结果显示：同一个 split 下 strict 配置和 New Baseline 输出的 Kalibr 数值完全一致，因此当前对比口径是一致的。

| 数据集 | 方法 | accepted candidates | holdout overall | holdout outer | holdout internal | training overall | 与 Kalibr overall 差值 |
|---|---|---:|---:|---:|---:|---:|---:|
| `134853 -> 144419` | Kalibr reference | - | 1.95731 | 0.59390 | 2.07724 | 0.853244 | - |
| `134853 -> 144419` | rerun strict 配置 | 0 | 1.94608 | 0.380394 | 2.07217 | 0.807267 | -0.011232 |
| `134853 -> 144419` | rerun New Baseline | 20 | 1.94433 | 0.33110 | 2.07148 | 0.807135 | -0.012983 |
| `144928 -> 134853` | Kalibr reference | - | 3.32491 | 0.892417 | 3.53120 | 6.61579 | - |
| `144928 -> 134853` | rerun strict 配置 | 0 | 3.29954 | 0.923015 | 3.50291 | 6.18099 | -0.025378 |
| `144928 -> 134853` | rerun New Baseline | 20 | 3.29735 | 0.890119 | 3.50174 | 6.30290 | -0.027567 |
| `144419 -> 144928` | Kalibr reference | - | 6.58414 | 15.5712 | 3.94820 | 0.746269 | - |
| `144419 -> 144928` | rerun strict 配置 | 0 | 6.55632 | 15.2089 | 4.08984 | 0.690111 | -0.027819 |
| `144419 -> 144928` | rerun New Baseline | 19 | 6.57076 | 15.2445 | 4.09776 | 0.690635 | -0.013379 |

当前代码 rerun 的结论是：

- Kalibr reference 在当前 strict 配置 / New Baseline 两组中完全一致，说明对比口径没有变化。
- 当前 strict 配置本身已经明显强于 Old Baseline。
- New Baseline 相比当前 strict 配置的增益变小：两组健康数据略有提升，困难组 `144419 -> 144928` 略有退化。
- 当前 strict 配置和 New Baseline 的 holdout overall 都仍然优于 Kalibr reference。

这说明我们前面改进的 selection / frontend / backend input 已经把 strict 配置本身推高了，因此 trial selection 的边际收益自然变小。综合历史收益和当前稳定性，后续将 `current` 固定为 **New Baseline**。

### 6.2 历史探索：20260430_134853 right -> 20260430_144419 right

| 配置 | accepted candidates | backend 输入 | holdout overall | holdout outer | holdout internal | training overall | 结论 |
|---|---:|---|---:|---:|---:|---:|---|
| Kalibr reference | - | 外部棋盘格 camchain | 2.1411 | 0.8926 | 2.2609 | 1.1980 | 外部参考 |
| Old Baseline | 0 | 16 frames / 78 boards / 2575 pts | 2.1105 | 0.7641 | 2.2345 | 1.1330 | 保守稳定 |
| New Baseline | 20 | 35 frames / 95 boards / 3066 pts | 1.94501 | 0.35204 | 2.07176 | 0.808203 | 明显提升 |
| relaxed40 | 40 | 43 frames / 115 boards / 3731 pts | 1.94398 | 0.32149 | 2.07135 | 0.808173 | outer 改善最明显 |
| relaxed60 | 60 | 46 frames / 135 boards / 4399 pts | 1.94379 | 0.32713 | 2.07102 | 0.807828 | overall 最低但收益很小 |
| hard v3 | 16 | consistency hard gate | 1.94517 | 0.36016 | 2.07174 | - | 略差于 New Baseline |
| Consistency-Soft Enhancement | 20 | consistency soft score | 1.94551 | 0.36490 | 2.07195 | 0.807155 | 略差于 New Baseline |

**结论**：这组数据比较健康，New Baseline / relaxed 已经足够好。Consistency-Soft Enhancement 没有带来额外收益。

### 6.3 历史探索：20260430_144928 right -> 20260430_134853 right

| 配置 | accepted candidates | backend 输入 | holdout overall | holdout outer | holdout internal | training overall | 结论 |
|---|---:|---|---:|---:|---:|---:|---|
| Kalibr reference | - | 外部棋盘格 camchain | 3.4916 | 1.1723 | 3.6988 | 2.1157 | 外部参考 |
| Old Baseline | 0 | 19 frames / 85 boards / 2820 pts | 3.4419 | 1.1472 | 3.6464 | 1.8448 | 保守稳定 |
| New Baseline | 20 | 40 frames / 111 boards / 3596 pts | 3.25668 | 0.81435 | 3.46075 | 1.08004 | 当前最稳 |
| relaxed40 | 40 | 46 frames / 131 boards / 4249 pts | 3.25674 | 0.81376 | 3.46084 | 1.08032 | 基本持平 |
| relaxed60 | 60 | 53 frames / 151 boards / 4898 pts | 3.25662 | 0.81324 | 3.46072 | 1.08037 | 差异可忽略 |
| hard v3 | 20 | consistency hard gate | 3.29819 | 0.91505 | 3.50176 | 6.19850 | 明显差于 New Baseline |
| Consistency-Soft Enhancement | 20 | consistency soft score | 3.29819 | 0.91505 | 3.50176 | 6.19850 | 与 hard v3 相同 |

**结论**：这组数据里 consistency score 会打乱原本较好的 candidate 排序，导致结果变差。New Baseline 仍然是更好的选择。

### 6.4 历史探索：20260430_144419 right -> 20260430_144928 right

| 配置 | accepted candidates | backend 输入 | holdout overall | holdout outer | holdout internal | training overall | 结论 |
|---|---:|---|---:|---:|---:|---:|---|
| Kalibr reference | - | 外部棋盘格 camchain | 8.5933 | 21.4454 | 4.4658 | 0.7313 | 外部参考 |
| Old Baseline | 0 | 11 frames / 53 boards / 1749 pts | 6.8392 | 15.2107 | 4.5828 | 0.6771 | 验证集较困难 |
| New Baseline | 19 | 31 frames / 75 boards / 2425 pts | 6.59282 | 15.2985 | 4.10884 | 0.709491 | 稳定提升 |
| relaxed40 | 38 | 35 frames / 94 boards / 3051 pts | 6.59607 | 15.3028 | 4.11255 | 0.709756 | 略差 |
| relaxed60 | 59 | 40 frames / 115 boards / 3751 pts | 6.61269 | 15.3478 | 4.11955 | 0.710512 | 更差 |
| hard v3 | 13 | consistency hard gate | 6.58260 | 15.2724 | 4.10370 | 0.708902 | 小幅优于 New Baseline |
| Consistency-Soft Enhancement | 20 | consistency soft score | 6.57448 | 15.2566 | 4.09827 | 0.690333 | 本组最好 |

**结论**：这组是困难验证集，outer pose refit 本身很不稳定。Consistency-Soft Enhancement 在这里有帮助，说明 consistency score 对困难序列能起到一定保护作用。

## 7. 扩展数据集验证

为了判断 New Baseline 是否足够稳定，以及 Consistency-Soft Enhancement / relaxed40 是否应该升级为默认策略，我们又在 `20260421` 与 `20260427` 的四组 cross-dataset split 上继续实验。目前已完成其中四组 New Baseline，以及三组 Consistency-Soft Enhancement / relaxed40。

### 7.1 New Baseline 在扩展数据集上的结果

| 数据集 | New Baseline holdout overall | holdout outer | holdout internal | Kalibr holdout | New Baseline-Kalibr | accepted candidates | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `140151 -> 141444` | 10.8765 | 24.5540 | 6.94257 | 12.5853 | -1.70885 | 20 | 优于 Kalibr，但差于旧 DS baseline |
| `141444 -> 140151` | 6.11854 | 2.49702 | 6.46687 | 6.16428 | -0.04575 | 20 | 小幅优于 Kalibr |
| `191538 -> 192347` | 3.75082 | 2.08755 | 3.92436 | 3.83157 | -0.08075 | 18 | 小幅优于 Kalibr |
| `192347 -> 191538` | 2.89692 | 1.00722 | 3.06714 | 2.94125 | -0.04434 | 20 | 小幅优于 Kalibr |

这四组说明：New Baseline 对 Kalibr reference 仍然保持整体优势，但不是对旧 DS baseline 全面提升。尤其 `140151 -> 141444` 这组，New Baseline 改善 internal，但 outer / global geometry 变差，导致 overall 明显高于旧 DS baseline。

### 7.2 Consistency-Soft Enhancement 与 relaxed40 在扩展数据集上的结果

| 数据集 | New Baseline | Consistency-Soft Enhancement | relaxed40 | 最好方案 | 结论 |
|---|---:|---:|---:|---|---|
| `141444 -> 140151` | 6.11854 | 6.11836 | 6.11722 | relaxed40 | 差异极小 |
| `191538 -> 192347` | 3.75082 | 3.74094 | 3.74956 | Consistency-Soft Enhancement | Consistency-Soft Enhancement 小幅最好 |
| `192347 -> 191538` | 2.89692 | 2.89884 | 2.90026 | New Baseline | New Baseline 小幅最好 |

这三组差异都很小，没有任何一个方案稳定全胜：

- Consistency-Soft Enhancement 在 `191538 -> 192347` 上最好，但在 `192347 -> 191538` 略差；
- `relaxed40` 在 `141444 -> 140151` 上最好，但优势只有 `0.001` 量级；
- New Baseline 在 `192347 -> 191538` 上最好，并且整体最稳。

### 7.3 对 baseline 选择的影响

扩展实验进一步说明：不能简单把 `relaxed40` 或 Consistency-Soft Enhancement 直接升级为默认 baseline。

- `relaxed40` 证明“更多 candidate / 更多 coverage”不一定更好；
- Consistency-Soft Enhancement 证明“结构一致性 score”对部分困难序列有帮助，但不是稳定全局收益；
- New Baseline 虽然不是每组最优，但参数更少、行为更稳定，更适合作为 trial-selection 主线；
- 当前 strict 配置已经吸收了部分 selection / frontend / backend input 改进，因此在当前代码下也很强。

## 8. 总体结论

### 冻结命名

后续实验与汇报统一采用以下命名：

| 名称 | 对应方案 | 定位 |
|---|---|---|
| Old Baseline | 早期 strict selection 口径 | 历史对照 |
| New Baseline | `current` 配置 | 后续 Stage5 默认主线 |
| Consistency-Soft Enhancement | 原 v3-soft | New Baseline 的可选增强参数 |
| relaxed40 | max_candidate_additions=40 | coverage ablation |
| hard v3 | consistency hard gate | hard gate ablation |

### 历史探索结论

从 Old Baseline 到 New Baseline 的探索过程看，selection / frontend / backend input 的改进确实带来了明显收益：三组 cross-dataset 的 holdout overall 都下降。这个阶段说明，原始 strict baseline 的问题不是相机模型本身不行，而是观测选择和 backend 输入还不够合理。

### 当前代码 rerun 结论

当前代码下，strict 配置本身已经变强，和 New Baseline 的差距明显缩小。因此现在更准确的说法是：

- 当前 strict 配置和 New Baseline 都优于 Kalibr reference；
- New Baseline 相比当前 strict 配置只带来很小的边际变化；
- New Baseline 在 `134853 -> 144419`、`144928 -> 134853` 上略好；
- New Baseline 在困难组 `144419 -> 144928` 上略差；
- 这说明前面关于 selection / frontend / backend input 的改进已经被吸收到当前 baseline 中。

### 当前最稳方案

综合历史收益、扩展数据集和方法复杂度，我们冻结 `current` 为 **New Baseline**。  
需要强调的是：New Baseline 不是每个 split 都最优，而是当前综合稳定性、解释性和收益最均衡的方案。

### relaxed40 的定位

`relaxed40` 可以作为可选扩展。它在 `134853 -> 144419` 上有小幅收益，但在困难组 `144419 -> 144928` 上略差于 New Baseline。因此它不适合作为默认策略，但可以保留为实验分支。

### relaxed60 的定位

`relaxed60` 接入更多 candidate，但泛化收益不稳定。它证明“更多 board coverage 不一定更好”，暂时不建议主推。

### consistency v3 的定位

Consistency-Soft Enhancement 的结论比较微妙：

- 对健康数据集，它不稳定，甚至会轻微变差；
- 对困难数据集，尤其 `144419 -> 144928`，Consistency-Soft Enhancement 有帮助；
- 在扩展数据集上，Consistency-Soft Enhancement 只在 `191538 -> 192347` 小幅最好，没有形成全局优势；
- hard v3 不建议主推，因为硬阈值容易误伤健康样本；
- Consistency-Soft Enhancement 更适合作为“困难序列增强策略”，而不是默认 baseline。

## 9. 推荐汇报说法

可以这样概括这一阶段：

> 我们没有简单地把所有候选 board 全部加入 backend，而是实现了 Kalibr-style 的增量试加入机制。该机制先保留稳定 seed，再把额外 frame-board 作为 candidate，通过 coverage、diversity、residual quality 和 short trial backend 判断是否接回。历史实验表明，selection / frontend / backend input 的演进显著改善了 cross-dataset holdout RMSE。进一步扩展到 `20260421` 和 `20260427` 数据后，New Baseline 在多数 split 上仍优于 Kalibr reference，并且比 relaxed40 / Consistency-Soft Enhancement 更稳定。因此我们将 `current` 配置冻结为 New Baseline；Consistency-Soft Enhancement 作为可选增强分支，relaxed40 / hard v3 作为 ablation。

## 10. 当前推荐

| 使用场景 | 推荐方案 |
|---|---|
| 默认实验 / 后续 Stage5 主线 | New Baseline |
| 历史对照 | Old Baseline |
| 展示 selection/frontend 改进过程 | Old Baseline -> New Baseline |
| 想尝试增量接回 candidate | New Baseline |
| 希望增加覆盖、风险可控 | relaxed40 |
| 困难序列、outer pose refit 明显不稳定 | Consistency-Soft Enhancement |
| 健康数据集 | 不建议默认启用 Consistency-Soft Enhancement |
| 追求最多 candidate 数 | 不建议 relaxed60 作为主线 |

## 11. 相关输出文件

常用结果文件：

- `backend_holdout_summary.txt`
- `backend_training_summary.txt`
- `backend_vs_kalibr_summary.txt`
- `experiment_config_summary.txt`
- `trial_backend_frame_board_selection_decisions.csv`
- `trial_backend_frame_board_selection_summary.txt`

已生成的分析文件：

- `exp_data/stage5_trial_backend_candidate_quality_analysis_cn.md`
- `exp_data/stage5_trial_backend_accepted_candidates.csv`
- `exp_data/stage5_trial_backend_candidate_quality_summary.csv`
- `exp_data/stage5_trial_backend_candidate_reason_counts.csv`
- `exp_data/stage5_trial_backend_accepted_board_distribution.csv`
- `exp_data/stage5_trial_backend_relaxed_vs_current.csv`
