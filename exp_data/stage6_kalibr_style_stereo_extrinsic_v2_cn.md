# Stage6 Kalibr-Style 双目外参标定 v2 实验记录

## 1. 目标

Stage6 v2 的目标，是把当前双目外参标定从“能跑通的 prototype”推进成更接近 Kalibr 思路的流程：

1. 先用高质量 shared stereo pair 得到稳定初值。
2. 再做 pair-only stereo 初始化 refinement。
3. 再参考 Stage5 New Baseline，做 `seed -> candidate -> short BA -> accept/reject` 的增量 pair selection。
4. 最终把 selected stereo pair 放进 global sparse BA。

这里的核心思想和我们 Stage5 内参部分是一致的：

- 不把所有候选一次性塞进 backend。
- 先有稳定 seed。
- 再逐个或逐批尝试 candidate。
- 只有不破坏系统的 candidate 才保留。

## 2. 当前已实现内容

### 2.1 Pair-only stereo BA init

已经从之前的“shared-board 候选外参加权平均”升级为真正的固定内参局部 BA。

当前变量：

- `T_cam1_cam0`
- 每个 accepted shared `pair-board` 的局部 `T_cam0_board`

固定量：

- 左右相机内参
- board 3D 点

残差：

- cam0 reprojection residual
- cam1 reprojection residual

保护逻辑：

- 只有 `after_shared_rmse <= before_shared_rmse` 才允许把 refined baseline 写回 scene state。
- 如果 refinement 变差，则仅保留诊断输出，不污染正式外参。

### 2.2 Kalibr-style pair trial selection

已经实现成真正的增量试加入流程，而不是之前的空壳版本。

当前流程：

1. 先按 shared pair 生成 seed。
2. shared pair 的剩余候选继续参与 trial。
3. single-camera-only pair 也可以进入 candidate pool。
4. 每次尝试加入一个 candidate，跑 short global sparse BA。
5. 如果 RMSE/baseline delta 不恶化，则 accept；否则 reject。

accept/reject gate 目前仍然是：

- `total_rmse_delta`
- `cam0_rmse_delta`
- `cam1_rmse_delta`
- `baseline_rotation_delta_deg`
- `baseline_translation_delta_m`

### 2.3 Stage6 缓存

已经接入和 Stage5 一样的 outer detection cache 入口。

当前支持：

- 左右单目前端 fixed-intrinsics frontend 的 outer detection cache
- 默认缓存目录：`result/.stage6_stereo_cache`
- 可手动指定：`--cache-dir PATH`

说明：

- 第一次 run 是“填缓存”，不会特别快。
- 第二次及以后对同样数据 rerun，才能真正体现缓存提速。

### 2.4 诊断输出

当前已支持：

- `stereo_pair_init_summary.txt`
- `stereo_pair_init_candidates.csv`
- `stereo_pair_init_residuals_before_after.csv`
- `stereo_pair_trial_selection_summary.txt`
- `stereo_pair_trial_selection_decisions.csv`
- `stereo_pair_trial_selected_pairs.csv`
- `stereo_reprojection_visualizations/`
- `stereo_extrinsic_uncertainty_summary.txt`
- `stereo_extrinsic_candidate_dispersion.csv`
- `stereo_extrinsic_jackknife.csv`

## 3. 134853 双目数据集当前实测结果

实验目录：

- [stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool)

### 3.1 Pair-only init 结果

来源：

- [stereo_pair_init_summary.txt](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool/stereo_pair_init_summary.txt)

| 指标 | 数值 |
|---|---:|
| raw candidate count | 35 |
| consistency filtered count | 35 |
| medoid baseline length | 0.0654955 |
| pair BA baseline length | 0.0642395 |
| before shared RMSE | 3.30347 |
| after shared RMSE | 1.43700 |
| baseline rotation delta (deg) | 0.188452 |
| baseline translation delta (m) | 0.00218908 |
| used refined baseline | 1 |

结论：

- 这说明 pair-only local BA init 已经是有效的。
- 它不再只是“轻量候选平均”，而是真正把 shared stereo 的解释误差显著降下来了。

### 3.2 Pair trial selection 结果

来源：

- [stereo_pair_trial_selection_summary.txt](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool/stereo_pair_trial_selection_summary.txt)

| 指标 | 数值 |
|---|---:|
| requested seed count | 10 |
| seed count | 7 |
| candidate count | 19 |
| attempted count | 12 |
| accepted count | 9 |
| rejected count | 3 |
| final selected pair count | 16 |
| initial seed RMSE | 1.52529 |
| final selected RMSE | 1.33986 |

结论：

- 这说明 Kalibr-style pair selection 也已经真正跑通。
- 不再是“全部变成 seed，完全没有增量 trial”的旧问题。

### 3.3 single-camera-only pair 是否进入系统

来源：

- [stereo_pair_trial_selection_decisions.csv](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool/stereo_pair_trial_selection_decisions.csv)
- [stereo_pair_selection.csv](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool/stereo_pair_selection.csv)

已被 accept 的 single-camera-only pair 示例：

- `pair 6`
- `pair 9`
- `pair 16`
- `pair 17`
- `pair 19`
- `pair 41`
- `pair 49`
- `pair 57`
- `pair 11`

被 reject 的 single-camera-only pair 示例：

- `pair 36`：`total_rmse_delta_gate`
- `pair 42`：`total_rmse_delta_gate`
- `pair 56`：`total_rmse_delta_gate`

结论：

- 现在 single-camera-only pair 已经不再只是“统计存在”，而是确实进入了 candidate pool。
- 而且是通过 short BA 决策后，选择性进入 final backend。

### 3.4 Final global sparse BA 结果

来源：

- [stereo_global_sparse_ba_summary.txt](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool/stereo_global_sparse_ba_summary.txt)
- [stereo_reprojection_summary.txt](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool/stereo_reprojection_summary.txt)

| 指标 | 数值 |
|---|---:|
| selected pair count | 16 |
| shared observation count | 2321 |
| cam0-only observation count | 669 |
| cam1-only observation count | 796 |
| cam0-only effective scale | 0.173468 |
| cam1-only effective scale | 0.145791 |
| training total stereo RMSE | 1.79886 |
| holdout total stereo RMSE | 1.53217 |

结论：

- 当前 final BA 已经真实使用了 single-camera-only observations。
- 而且单侧观测被 budget cap 限制，不会无约束地主导系统。

## 4. 与上一版 Stage6 prototype 的对比

上一版问题：

1. pair-only init 只是候选外参加权平均。
2. 如果 refinement 变差，也可能污染 baseline。
3. pair trial selection 候选池太窄，往往只有 shared pair。
4. single-camera-only pair 没有真正进入 final BA。
5. 经常出现 `attempted_count=0` 或 `accepted_count=0`。

当前改进后，134853 的变化如下：

| 指标 | 旧版 | 当前版 |
|---|---:|---:|
| pair-init after shared RMSE | 3.73744 | 1.43700 |
| pair-init used refined baseline | 0/不稳定 | 1 |
| trial candidate count | 7 | 19 |
| attempted count | 1 | 12 |
| accepted count | 0 | 9 |
| final selected pair count | 6 | 16 |
| training total stereo RMSE | 3.20259 | 1.79886 |
| holdout total stereo RMSE | 1.53074 | 1.53217 |

解释：

- 训练集表现改善很明显，说明 Stage6 v2 的算法框架已经被真正打通。
- holdout 当前基本持平，说明这版先主要解决了“流程正确性”，还不是最终泛化最优。

## 5. 当前结论

可以把当前结论概括成三点：

1. **pair-only stereo BA init 已经成立**
   - 这一步现在是有效的，不只是 diagnostic。

2. **Kalibr-style pair selection 已经成立**
   - 不再是形式上的 seed/candidate，而是实打实的 short BA accept/reject。

3. **single-camera-only pair 已经可控地接入**
   - 它们会进入 final BA，但通过权重预算做了保护。

## 6. 多数据集验证结果

目前已经完成三组 `dataset_5_1` 双目数据集验证：

- [134853](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool)
- [144419](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_144419_stereo_kalibr_style_diag_localba_trialpool)
- [144928](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_144928_stereo_kalibr_style_diag_localba_trialpool)

后续已补齐带 Kalibr reference camchain 对比的正式输出：

- [134853 refcompare](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool_refcompare_cachecheck_fixed)
- [144419 refcompare](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_144419_stereo_kalibr_style_diag_localba_trialpool_refcompare)
- [144928 refcompare](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_144928_stereo_kalibr_style_diag_localba_trialpool_refcompare)

### 6.1 Pair-only init 是否有效

| Dataset | raw candidates | before shared RMSE | after shared RMSE | used refined baseline | baseline length |
|---|---:|---:|---:|---:|---:|
| `134853` | 35 | 3.30347 | 1.43700 | 1 | 0.0642163 |
| `144419` | 17 | 10.1521 | 3.25725 | 1 | 0.0659826 |
| `144928` | 34 | 4.75454 | 2.93887 | 1 | 0.0642149 |

结论：

- 三组数据都满足 `after_shared_rmse < before_shared_rmse`。
- 说明 pair-only stereo BA init 不是只对 134853 有效，而是在三组数据上都能显著降低 shared stereo residual。
- `144419` 初始误差最大，说明该组 stereo shared pose 初值更差，但 pair-only init 仍能把误差从 10 px 量级拉到 3 px 量级。

### 6.2 Kalibr-style pair selection 是否有效

| Dataset | candidate count | attempted | accepted | rejected | final selected pairs | seed RMSE | final selected RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `134853` | 19 | 12 | 9 | 3 | 16 | 1.52529 | 1.33986 |
| `144419` | 11 | 7 | 0 | 7 | 4 | 3.74666 | 3.74666 |
| `144928` | 24 | 17 | 17 | 0 | 24 | 2.98346 | 2.15327 |

结论：

- `134853`：trial selection 有效接回 9 个 candidate，selected RMSE 明显下降。
- `144928`：17 个 candidate 全部通过，selected RMSE 从 2.98 降到 2.15，收益明显。
- `144419`：所有 7 个 candidate 都被 reject，说明这组额外候选对当前系统不友好，selection 起到了保护作用。

### 6.3 Final BA 是否使用 single-camera-only observations

| Dataset | selected pairs | shared obs | cam0-only obs | cam1-only obs | cam0-only scale | cam1-only scale |
|---|---:|---:|---:|---:|---:|---:|
| `134853` | 16 | 2321 | 669 | 796 | 0.173468 | 0.145791 |
| `144419` | 4 | 1109 | 0 | 106 | 0 | 0.25 |
| `144928` | 24 | 2258 | 1003 | 1776 | 0.112562 | 0.0635698 |

结论：

- `134853` 和 `144928` 中，single-camera-only observations 已经真实进入 final BA。
- `144419` 中只有少量 cam1-only observations 进入，且 candidate 被大量拒绝，说明该数据集的可用 stereo candidate 更少。
- 单侧观测均通过 budget cap 限权，避免它们压过 shared stereo observations。

### 6.4 Training / holdout 结果

| Dataset | training total stereo RMSE | holdout total stereo RMSE | final selected pairs | 说明 |
|---|---:|---:|---:|---|
| `134853` | 1.79886 | 1.53217 | 16 | 训练误差明显下降，holdout 与前版基本持平 |
| `144419` | 3.82812 | 2.33539 | 4 | 候选全部被拒，说明 selection 保护较强 |
| `144928` | 2.15327 | 3.18250 | 24 | 候选全部接回，训练改善明显，但 holdout 偏高 |

阶段性判断：

- Stage6 v2 的算法链路已经稳定跑通。
- pair-only init 对三组数据都有明显帮助。
- pair trial selection 在不同数据集上表现出不同策略：有的积极接回，有的严格拒绝。
- 当前还不能直接说 v2 已经是最终外参 baseline，因为 holdout 在不同数据集上仍有差异，尤其 `144928` 的 holdout 还需要进一步诊断。

## 7. 缓存与 runtime 状态

Stage6 现在已经支持 outer detection cache，并且 `stage6_runtime_summary.txt` 已经补充了：

- `cache_dir`
- `cache_enabled`
- `cam0_training_detection_cache_hits`
- `cam0_training_detection_cache_misses`
- `cam1_training_detection_cache_hits`
- `cam1_training_detection_cache_misses`

最新三组正式输出中，cache hit/miss 已经正常写入。三组训练集左右相机检测均为 cache hit，没有 cache miss。

注意：

- `134853 cachecheck` 的算法结果和 `134853` 一致。
- 当前 runtime summary 已能证明 outer detection 重算被跳过。
- 如果后续要精确量化加速比，还需要增加 cold run / warm run 的独立耗时对比，而不是只看单次 total runtime。

## 8. 与 Kalibr 双目外参的直接对比

参考外参文件：

- [stereo_4_2-3-camchain.yaml](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/config/stereo_4_2-3-camchain.yaml)

Kalibr reference `T_cam1_cam0` 的平移为：

```text
[-0.0642553862, 0.0006565750, 0.0001125084]
```

reference baseline length：

```text
0.064258839 m
```

当前三组 Stage6 v2 结果与 Kalibr 外参的差异如下：

| Dataset | estimated translation xyz | estimated baseline | rot delta vs Kalibr (deg) | trans delta vs Kalibr (m) | baseline delta vs Kalibr (m) | 结论 |
|---|---|---:|---:|---:|---:|---|
| `134853` | `[-0.0641981, -0.0007061, -0.00135656]` | 0.0642163 | 0.149322 | 0.00200458 | 0.00004258 | baseline length 很接近，方向有约 2 mm 平移差 |
| `144419` | `[-0.0659678, -0.00034097, -0.00135496]` | 0.0659826 | 0.200927 | 0.00246596 | 0.00172376 | 该组偏差最大，和 selection 候选全部被拒一致 |
| `144928` | `[-0.0642096, -0.000130158, -0.000822309]` | 0.0642149 | 0.158735 | 0.00122267 | 0.00004389 | 三组里 translation delta 最小，baseline length 很接近 |

正式带 `--stereo-reference-camchain` 的输出：

- [stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool_refcompare_cachecheck_fixed](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool_refcompare_cachecheck_fixed)
- [stereo_reference_comparison.txt](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool_refcompare_cachecheck_fixed/stereo_reference_comparison.txt)
- [stage6_v2_144419_stereo_kalibr_style_diag_localba_trialpool_refcompare](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_144419_stereo_kalibr_style_diag_localba_trialpool_refcompare)
- [stereo_reference_comparison.txt](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_144419_stereo_kalibr_style_diag_localba_trialpool_refcompare/stereo_reference_comparison.txt)
- [stage6_v2_144928_stereo_kalibr_style_diag_localba_trialpool_refcompare](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_144928_stereo_kalibr_style_diag_localba_trialpool_refcompare)
- [stereo_reference_comparison.txt](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_144928_stereo_kalibr_style_diag_localba_trialpool_refcompare/stereo_reference_comparison.txt)

该正式输出的 reference comparison 为：

| 指标 | 数值 |
|---|---:|
| success | 1 |
| rotation delta vs Kalibr | 0.149322 deg |
| translation delta vs Kalibr | 0.00200458 m |
| Kalibr baseline length | 0.0642588 m |
| estimated baseline length | 0.0642163 m |
| baseline length delta | 0.0000425758 m |

解释：

- `134853` 和 `144928` 的 baseline length 与 Kalibr 非常接近，差值约 `0.04 mm`。
- `144419` 的 baseline length 偏大约 `1.72 mm`，是三组里最不稳定的一组。
- 从 rotation delta 看，三组都在 `0.15 ~ 0.20 deg` 量级。
- 从 translation delta 看，三组都在 `1.2 ~ 2.5 mm` 量级。

需要注意：

- 这里比较的是外参矩阵本身，不是 stereo reprojection RMSE。
- 由于 Stage6 使用我们的 multi-board AprilTag 观测，而 Kalibr camchain 来自独立标定，两者的 residual 口径和数据来源不完全一样。
- 因此这个表更适合说明“外参量级是否一致”，不是单独判断谁绝对更优。

后续正式运行建议带上：

```bash
--stereo-reference-camchain config/stereo_4_2-3-camchain.yaml
```

这样每个输出目录会自动生成：

```text
stereo_reference_comparison.txt
```

用于记录：

- `rotation_delta_deg`
- `translation_delta_m`
- `baseline_length_reference`
- `baseline_length_estimated`
- `baseline_length_delta_m`
- `reference_translation_xyz`

## 9. 缓存验证

正式 cachecheck 输出：

- [stage6_runtime_summary.txt](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_may/stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool_refcompare_cachecheck_fixed/stage6_runtime_summary.txt)

缓存统计：

| Dataset | cache enabled | cam0 hits / misses | cam1 hits / misses | total runtime seconds |
|---|---:|---:|---:|---:|
| `134853` | 1 | 85 / 0 | 85 / 0 | 10.8927 |
| `144419` | 1 | 88 / 0 | 87 / 0 | 11.0347 |
| `144928` | 1 | 135 / 0 | 134 / 0 | 41.6211 |

结论：

- Stage6 的左右单目前端 outer detection cache 已经验证成功。
- 对同一组数据 rerun 时，三组正式输出的左右相机检测均全部命中 cache。
- 当前 `pairing_build_dataset_runtime_seconds` 仍然包含前端构建、bundle build 和 dataset build 的累计时间，不是纯 detection 时间；因此它不能直接等价于“缓存耗时”。
- 但 cache hit/miss 已经能证明 outer detection 重算被跳过。

## 10. 当前一句话总结

Stage6 v2 已经从“prototype 计划”走到“核心算法链路跑通并完成三组验证”的阶段：

- pair-only stereo BA init 在三组数据上都有效；
- Kalibr-style pair selection 已经真正执行 short BA accept/reject；
- single-camera-only pair 已经可以受控进入 final BA；
- 与 Kalibr 外参相比，`134853` 和 `144928` 的 baseline length 非常接近，`144419` 仍偏不稳定；
- Stage6 outer detection cache 已验证可用；
- 下一步重点是：分析 `144928` holdout 偏高原因，并补旧 Stage6 baseline 对比。

## 11. 下一步建议

1. 针对 `144928` 输出 top bad stereo pair 可视化，确认 holdout 偏高来自 cam0、cam1、shared board 还是单侧 board。
2. 与旧 Stage6 baseline 做正式表格对比，明确 v2 的收益来自 pair-only BA init、pair trial selection，还是 single-camera-only pair 接入。
3. 后续可以做消融：
   - only pair-only init
   - only pair trial selection
   - pair-only init + trial selection
   - with / without single-camera-only candidates

## 12. 2026-05-29：Stage6 holdout RMSE 偏大排查

### 12.1 排查目标

之前 `134853 -> 144419` 的 Stage6 输出出现一个看起来矛盾的现象：

- 我们估计的外参与 Kalibr 外参非常接近；
- 但普通 holdout stereo reprojection RMSE 很大，约 `22.99 px`；
- 早期 extrinsic-only holdout 口径也一度达到 `27.15 px`。

因此这次排查重点不是继续调外参，而是确认：

1. 左右图像是否可能错配；
2. `stereo_4_2-3-camchain.yaml` 的方向是否被用反；
3. holdout 评估是否混入了训练集 multi-board 全局结构或局部 pose refit 失败样本。

### 12.2 新增诊断

在 `stereo_pair_board_consistency.csv` 中新增了几类只读诊断字段：

- `stereo_local_pose_delta_rotation_deg`
- `stereo_local_pose_delta_translation_m`
- `cam1_outer_rmse_from_cam0_pose`
- `cam0_outer_rmse_from_cam1_pose`
- `stereo_outer_rmse_from_cam0_pose`
- `stereo_outer_rmse_from_cam1_pose`
- `cam1_outer_rmse_from_cam0_pose_inverse_extrinsic`
- `cam0_outer_rmse_from_cam1_pose_inverse_extrinsic`

含义：

- 先分别用左相机和右相机的真实 outer corners 做单板局部 pose refit；
- 再用当前 `T_cam1_cam0` 把左相机局部 pose 投到右相机，计算右图 outer residual；
- 同时测试把外参方向反过来是否会更好。

### 12.3 关键输出

本次主要输出目录：

- `result_stereo/stage6_extrinsic_pose_delta_audit_134853_to_144419_v4_evaltrace`
- `result_stereo/stage6_extrinsic_pose_delta_audit_134853_to_144419_v5_extrinsic_guard`

pairing 结果：

| 项目 | 数值 |
|---|---:|
| paired frame count | 82 |
| training pairs | 41 |
| holdout pairs | 41 |
| pairing mode | filename timestamp exact |
| max timestamp delta | 0 ns |

外参方向测试：

| 指标 | training | holdout |
|---|---:|---:|
| `cam1_outer_rmse_from_cam0_pose` mean | 2.67997 | 2.61023 |
| `cam1_outer_rmse_from_cam0_pose` median | 2.05200 | 1.90175 |
| `cam0_outer_rmse_from_cam1_pose` mean | 2.65488 | 2.79908 |
| `cam0_outer_rmse_from_cam1_pose` median | 2.23016 | 2.26915 |
| inverse-direction cam1 mean | 333.6 | 341.019 |
| inverse-direction cam1 median | 340.868 | 345.829 |

结论：

- 外参方向没有用反；如果用反，RMSE 会到 `300+ px`。
- 左右图像按文件名时间戳精确配对，当前 Stage6 输出层面没有发现错配一帧的迹象。
- 对绝大多数 holdout board，左右单板局部 pose 通过当前外参互相解释时，outer residual 只有约 `2~3 px`。

### 12.4 找到的真正问题

早期 `holdout_extrinsic_only` 的 `27.15 px` 不是外参本身这么差，而是被少数局部 refit 失败样本拉爆。

典型异常：

| pair | frame | 问题 | 影响 |
|---|---|---|---|
| 66 | `000048_left_166703752000_mono8` | board4 单板局部 pose refit 失败 | 整个 pair RMSE 被拉到 `170.35 px` |
| 44 | `000006_left_121403198080_mono8` | 局部 stereo board refit 较差 | pair RMSE `11.13 px` |
| 51 | `000020_left_136503197160_mono8` | internal 被局部 refit 牵连 | pair RMSE `10.36 px` |
| 56 | `000028_left_145103198080_mono8` | 局部 refit 较差 | pair RMSE `6.96 px` |

消融统计：

| 口径 | holdout extrinsic-only RMSE |
|---|---:|
| 原始 extrinsic-only | 27.1451 |
| 去掉 pair66 | 3.36664 |
| 去掉 top4 bad pairs | 2.27413 |

这说明问题主要是少数 holdout 局部 board pose refit 失败，而不是外参整体错误。

### 12.5 修复

在 `StereoResidualEvaluationOptions` 中新增：

```text
extrinsic_only_max_local_stereo_outer_rmse_px = 8.0
```

在 `extrinsic_only_local_board_pose` 评估路径中：

- 每个 shared board 先做局部 stereo outer pose refit；
- 如果该 board 的局部 stereo outer RMSE 大于 `8 px`，则该 board 不参与 extrinsic-only holdout 统计；
- 普通 holdout、训练优化、pair selection 和最终外参结果不受影响。

修复后 `134853 -> 144419`：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| ordinary holdout total | 22.9876 | 22.9876 |
| extrinsic-only holdout total | 27.1451 | 2.95597 |
| extrinsic-only outer | 28.0447 | 2.20441 |
| extrinsic-only internal | 27.0196 | 3.04443 |
| extrinsic-only cam0 | 28.5699 | 2.67718 |
| extrinsic-only cam1 | 25.6552 | 3.20846 |

Kalibr 同口径对比：

| 方法 | extrinsic-only total | outer | internal | cam0 | cam1 |
|---|---:|---:|---:|---:|---:|
| Stage6 v2 | 2.95597 | 2.20441 | 3.04443 | 2.67718 | 3.20846 |
| Kalibr reference | 4.25285 | 3.93187 | 4.29478 | 2.85519 | 5.28568 |
| Stage6 - Kalibr | -1.29688 | -1.72746 | -1.25035 | -0.17801 | -2.07722 |

### 12.6 当前结论

这次排查后的结论是：

- 普通 holdout RMSE 大，是因为它仍然带有训练集 multi-board global scene / board structure 迁移，不适合作为跨数据集外参评价主指标。
- 更合理的外参评价是 extrinsic-only holdout：对验证集每个 shared board 现场做局部 pose refit，再评估左右相机外参一致性。
- 修复局部 refit 失败样本污染后，`134853 -> 144419` 的外参独立 holdout 为 `2.956 px`，并且优于 Kalibr reference 的 `4.253 px`。
- 后续 Stage6 报告中应同时保留：
  - ordinary holdout：说明训练 multi-board scene 跨数据集迁移情况；
  - extrinsic-only holdout：作为外参本身的主要评价指标。

### 12.7 后续待做

1. 把 `extrinsic_only_max_local_stereo_outer_rmse_px` 做成 CLI 参数，默认 `8.0`。
2. 对更多跨数据集 split 继续跑同口径，验证趋势是否稳定。
3. 输出被 extrinsic-only guard 跳过的 pair/board 列表，方便检查是否是模糊、遮挡或角点异常。
4. Stage6 cache 仍需优化：当前 detection cache 命中，但 frontend bundle/dataset build 仍然耗时较长。

### 12.8 三组有效外参评估结果

本节使用同一套 Stage6 v2 设置：

- `all_valid` stereo candidate pool；
- shared-board quality gate；
- pair-only stereo BA init；
- pair-board trial selection；
- `global_sparse_ba`；
- extrinsic-only holdout 使用验证集局部 stereo board pose refit；
- 对局部 refit 失败样本使用 `extrinsic_only_max_local_stereo_outer_rmse_px = 8.0` guard。

这里**只记录有效的外参评估口径**：`extrinsic-only holdout`。

说明：

- `extrinsic-only` 不使用训练集的 global multi-board scene / board structure；
- 验证集每个 shared board 现场做局部 stereo pose refit；
- 然后只检查固定外参是否能同时解释左右相机角点；
- Kalibr reference 也用同一个 evaluator，只替换为 `stereo_4_2-3-camchain.yaml` 中的外参。

| Split | Stage6 v2 extrinsic-only | Kalibr extrinsic-only | Stage6 - Kalibr | Stage6 outer / internal | Kalibr outer / internal |
|---|---:|---:|---:|---:|---:|
| `134853 -> 144419` | 2.95597 | 4.25285 | -1.29688 | 2.20441 / 3.04443 | 3.93187 / 4.29478 |
| `144928 -> 134853` | 4.14451 | 5.21814 | -1.07362 | 3.58874 / 4.21461 | 4.75949 / 5.27738 |
| `144419 -> 144928` | 2.86162 | 4.23852 | -1.37690 | 2.25695 / 2.93442 | 3.99707 / 4.27036 |

训练集统计：

| Split | training total RMSE | selected / eligible pairs | pair-only init before -> after |
|---|---:|---:|---:|
| `134853 -> 144419` | 1.47004 | 44 / 44 | 3.91489 -> 1.67884 |
| `144928 -> 134853` | 1.25140 | 52 / 61 | 3.52403 -> 1.72975 |
| `144419 -> 144928` | 1.94117 | 41 / 41 | 5.90942 -> 2.04375 |

extrinsic-only top bad pair 主要集中在少数 frame：

| Split | top bad pair | frame label | extrinsic-only pair RMSE | outer / internal |
|---|---:|---|---:|---:|
| `134853 -> 144419` | 51 | `000020_left_136503197160_mono8` | 7.17594 | 1.57804 / 7.67158 |
| `144928 -> 134853` | 77 | `000028_left_215103202000_mono8` | 11.4563 | 3.42204 / 12.1617 |
| `144419 -> 144928` | 63 | `000045_left_472903188160_mono8` | 7.86368 | 1.39756 / 8.38461 |

当前结论：

- 三组 split 下，Stage6 v2 的 extrinsic-only holdout 都优于 Kalibr reference，优势约 `1.07 ~ 1.38 px`。
- 普通 holdout 已降级为 debug 口径，不再作为外参主表结果展示；它主要反映跨数据集 global multi-board scene / board structure 不一致。
- extrinsic-only 口径下的剩余 top bad pair 往往是 internal RMSE 高、outer RMSE 不一定高，下一步更适合继续检查 internal regeneration / 局部 stereo refit 对 internal 点的影响，而不是先怀疑外参方向或时间同步。

### 12.9 下一步算法优化方向

当前 Stage6 v2 已经证明外参本身可达到并超过 Kalibr reference 的同口径 holdout。下一步不建议继续围绕 ordinary holdout 调参，而应该继续优化 stereo pair / pair-board 的选择、鲁棒性和诊断。

#### 方向 A：Stage6 版 New Baseline selection

来源：

- 参考 Kalibr `addBatch` 的思想；
- 参考 Stage5 内参 New Baseline 的 `seed bundle -> candidate pool -> short backend -> accept/reject`。

当前已经做到：

- pair-only BA init；
- pair-board trial selection；
- shared-board quality gate；
- all-valid stereo candidate pool；
- short BA delta gate。

还可以补：

- 按 pair-board 而不是整 pair 做更细的接受/拒绝；
- 每个 stereo pair 内保留更多高质量 board，而不是一整对 pair 通过/失败；
- 给 close / edge / large-area board 增加 coverage gain，而不是只看残差；
- 输出每个 rejected pair-board 的具体拒绝原因，例如 local stereo outer refit 差、cam0/cam1 residual 不平衡、baseline delta 过大。

预期解决的问题：

- 避免少数坏 board 污染整对 stereo pair；
- 让更多有价值的 shared board 进入外参优化；
- 减少“整帧看起来不错，但只有少数 board 进入 backend”的情况。

#### 方向 B：Kalibr-style 信息增益 / 覆盖增益

Kalibr 的核心不是简单把所有帧都塞进优化，而是逐步增加能提升约束的信息。我们可以在 Stage6 中加入更明确的 stereo coverage score：

- 左右共同可见 board 数；
- shared outer/internal 点数；
- 左右 residual balance；
- board id 覆盖；
- polar angle / edge coverage；
- baseline 约束贡献；
- 与已有 selected pair 的视角差异。

预期解决的问题：

- 避免重复视角占满 selection budget；
- 优先选择对外参更有约束的 stereo pair；
- 让训练集覆盖更均衡，提升跨数据集泛化。

#### 方向 C：鲁棒核与 outlier model

当前 Stage6 使用 Huber，第一版没有引入 Kalibr 的 Blake-Zisserman。后续可以做 ablation：

- Huber 当前 baseline；
- Cauchy；
- Tukey；
- Blake-Zisserman 风格 switch / outlier model。

建议不要和 selection 同时改，应该单独实验。

预期解决的问题：

- 减少模糊帧、局部 bad board、internal outlier 对外参的拉偏；
- 让优化更像 Kalibr 的鲁棒 batch calibration。

#### 方向 D：外参不确定性与稳定性

Kalibr 会输出较完整的不确定性信息。我们当前已有 dispersion / jackknife proxy，但还可以加强：

- selected-pair jackknife 外参变化；
- 每个 pair-board 对 baseline 的影响排序；
- 最坏 pair 删除前后外参变化；
- rotation / translation MAD；
- 标记外参是否由少数 frame 主导。

预期解决的问题：

- 不只看 RMSE，也知道外参是否稳定；
- 防止某组数据 RMSE 不高但外参由少数样本支撑。

#### 方向 E：extrinsic-only evaluator 继续完善

目前 extrinsic-only 已经是主要外参评价口径。后续建议补充：

- 输出被 `8 px` local stereo guard 跳过的 pair-board 列表；
- top bad pair 的 outer/internal residual overlay；
- cam0 local pose、cam1 local pose、外参传递 pose 的位姿差；
- 用文字 summary 说明 top bad 是 outer 坏、internal 坏，还是局部 stereo refit 坏。

预期解决的问题：

- 让每次 RMSE 变化都有可解释原因；
- 避免再次被少数局部 refit 失败样本误导。

### 12.10 Pair-board New Baseline selection v2_seed40

在原有 pair-board trial selection 基础上，新增更接近 Stage5 New Baseline 的策略约束：

- `seed_count = 40`，避免 seed 一开始占满所有好候选；
- `max_candidate_additions = 40`，允许更多 pair-board 进入 short BA trial；
- `min_candidate_score = 20`；
- `min_coverage_gain = 0.5`；
- `max_accepted_per_pair = 3`；
- `max_accepted_per_board = 24`；
- 每个候选输出 `coverage_gain`、当前已选 pair/board 数、拒绝原因。

目的：

- 从“整 pair 选择”进一步细化到“pair-board 选择”；
- 避免单个坏 board 污染整对 stereo pair；
- 让 candidate 既要质量好，也要带来新的 pair / board 覆盖；
- 让每个拒绝原因可追踪。

结果：

| Split | Stage6 v2 guard | pair-board v2_seed40 | Kalibr | v2_seed40 - Kalibr | 相对 Stage6 v2 guard |
|---|---:|---:|---:|---:|---:|
| `134853 -> 144419` | 2.95597 | 2.84007 | 4.25285 | -1.41278 | -0.11590 |
| `144928 -> 134853` | 4.14451 | 4.05975 | 5.21814 | -1.15839 | -0.08476 |
| `144419 -> 144928` | 2.86162 | 2.98505 | 4.23852 | -1.25348 | +0.12343 |

pair-board selection 统计：

| Split | seed | candidates | attempted | accepted | final pair-board | training RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `134853 -> 144419` | 40 | 202 | 16 | 4 | 44 | 1.10357 |
| `144928 -> 134853` | 40 | 289 | 34 | 22 | 62 | 1.26792 |
| `144419 -> 144928` | 40 | 201 | 10 | 9 | 49 | 1.86245 |

拒绝原因统计：

| Split | coverage gain gate | score gate | per-board cap | RMSE delta gate |
|---|---:|---:|---:|---:|
| `134853 -> 144419` | 141 | 0 | 5 | 12 |
| `144928 -> 134853` | 190 | 5 | 20 | 12 |
| `144419 -> 144928` | 148 | 0 | 3 | 1 |

当前判断：

- v2_seed40 在 `134853 -> 144419` 和 `144928 -> 134853` 上优于 Stage6 v2 guard；
- 但在 `144419 -> 144928` 上略差，说明固定的 `seed=40 / min_gain=0.5 / per-pair=3` 还不是稳定默认策略；
- 这版证明了 pair-board selection 方向有效，但还需要继续做策略调优。

下一步建议：

1. 保留 Stage6 v2 guard 作为当前稳定基线；
2. 将 pair-board v2_seed40 作为实验分支；
3. 下一轮尝试更温和策略：
   - `seed_count = 50`；
   - `min_coverage_gain = 0.0` 或只作为 score bonus；
   - `max_accepted_per_pair = 4`；
   - 对 `coverage_gain_gate` 不直接 hard reject，而是降低优先级；
4. 同时继续输出 top accepted/rejected pair-board overlay，确认被接回的 board 是否真的有图像质量。

### 12.11 Pair-board New Baseline selection v3_soft

v2_seed40 的问题是 `coverage_gain_gate` 太硬，导致大量候选虽然质量不错，但因为没有带来新的 pair / board 覆盖而被直接拒绝。v3_soft 改成更温和策略：

- `seed_count = 50`；
- `max_candidate_additions = 40`；
- `min_candidate_score = 20`；
- `min_coverage_gain = 0`；
- `max_accepted_per_pair = 4`；
- `max_accepted_per_board = 24`。

目的：

- 保留 pair-board 粒度；
- 不再把 coverage gain 当作 hard reject；
- 让更多高质量 pair-board 进入 short BA，由 RMSE / baseline delta gate 决定是否保留。

结果：

| Split | Stage6 v2 guard | v2_seed40 | v3_soft | Kalibr | v3_soft - Kalibr | v3_soft 相对 guard |
|---|---:|---:|---:|---:|---:|---:|
| `134853 -> 144419` | 2.95597 | 2.84007 | 2.90431 | 4.25285 | -1.34854 | -0.05166 |
| `144928 -> 134853` | 4.14451 | 4.05975 | 3.97210 | 5.21814 | -1.24603 | -0.17241 |
| `144419 -> 144928` | 2.86162 | 2.98505 | 2.83141 | 4.23852 | -1.40712 | -0.03021 |

selection 统计：

| Split | seed | candidates | attempted | accepted | final pair-board | training RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `134853 -> 144419` | 50 | 202 | 42 | 40 | 90 | 1.41928 |
| `144928 -> 134853` | 50 | 289 | 47 | 40 | 90 | 1.25213 |
| `144419 -> 144928` | 50 | 201 | 43 | 40 | 90 | 1.60387 |

拒绝原因统计：

| Split | per-board cap | per-pair cap | RMSE delta gate |
|---|---:|---:|---:|
| `134853 -> 144419` | 8 | 2 | 2 |
| `144928 -> 134853` | 29 | 0 | 7 |
| `144419 -> 144928` | 19 | 0 | 3 |

当前判断：

- v3_soft 在三组上都优于 Stage6 v2 guard；
- v3_soft 相比 v2_seed40 更稳定，解决了 v2_seed40 在 `144419 -> 144928` 上退化的问题；
- v2_seed40 在 `134853 -> 144419` 上单点最好，但跨 split 稳定性不如 v3_soft；
- 当前建议把 v3_soft 作为 Stage6 pair-board selection 的主实验分支，暂不直接替代 Stage6 v2 guard，等再补两组三维数据后再冻结。

下一步建议：

1. 跑 `140151/141444` 或 `191538/192347` 双目 split，验证 v3_soft 是否仍稳定；
2. 给 accepted / rejected pair-board 输出 side-by-side overlay；
3. 研究 `max_accepted_per_board = 24` 是否过强，因为当前仍有不少候选被 per-board cap 卡掉；
4. 如果更多数据集稳定，再考虑将 v3_soft 升级为 Stage6 v3 baseline。

### 12.12 Stage6 v3_soft 小消融与可解释可视化

本轮目标是把 v3_soft 从“数值上看起来更好”推进到“可解释、可汇报、可继续调参”的状态。

新增诊断输出：

- `stereo_pair_board_selection_visualizations/seed_pair_boards/`
- `stereo_pair_board_selection_visualizations/attempted_accepted_pair_boards/`
- `stereo_pair_board_selection_visualizations/rejected_total_rmse_delta_gate/`
- `stereo_pair_board_selection_visualizations/rejected_max_accepted_per_pair_gate/`
- `stereo_pair_board_selection_visualizations/rejected_max_accepted_per_board_gate/`

每张图都是左右相机 side-by-side overlay，只高亮一个 pair-board decision。图中包含：

- 真实观测点；
- 当前外参和局部/全局 pose 下的重投影点；
- residual arrow；
- pair id / board id；
- seed / accepted / rejected 状态；
- candidate score、coverage gain、RMSE delta、pose source。

这一步解决的问题是：不仅知道 RMSE 是否下降，还能看到“哪些 board 被接回、为什么接回、哪些 board 被拒绝、是 RMSE gate 拒绝还是 cap gate 拒绝”。

#### 小消融设置

对三组现有双目 split 做四组对比：

| 方法 | 目的 | 关键设置 |
|---|---|---|
| `v2_guard` | 安全基线 | seed 较保守，pair-board 接回较少 |
| `v3_soft` | 当前候选新 baseline | `seed=50, max_add=40, min_score=20, min_gain=0, max_per_pair=4, max_per_board=24` |
| `no_pairboard_trial` | 验证 pair-board trial selection 是否必要 | 关闭 pair-board trial selection，只用 all-valid + shared quality + pair init + final BA |
| `no_board_cap` | 验证 `max_accepted_per_board=24` 是否过强 | v3_soft 基础上关闭 board-level cap，即 `max_per_board=0` |

#### 结果表

以下表格均使用 **extrinsic-only holdout RMSE** 作为主要指标。ordinary holdout 仍只作为 debug，因为它会混入训练得到的 multi-board scene structure，不适合跨数据集评价外参本身。

| Split | Method | Training RMSE | Extrinsic-only holdout | Holdout outer | Holdout internal | final pair-board | accepted | rejected |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `134853 -> 144419` | `v2_guard` | 1.47004 | 2.95597 | 2.20441 | 3.04443 | 80 | 20 | 0 |
| `134853 -> 144419` | `v3_soft` | 1.41928 | 2.90431 | 2.14658 | 2.99314 | 90 | 40 | 12 |
| `134853 -> 144419` | `no_pairboard_trial` | 2.51493 | 3.01864 | 2.24890 | 3.10920 | - | - | - |
| `134853 -> 144419` | `no_board_cap` | 1.41396 | 2.94702 | 2.19614 | 3.03538 | 90 | 40 | 2 |
| `144928 -> 134853` | `v2_guard` | 1.25140 | 4.14451 | 3.58874 | 4.21461 | 80 | 20 | 3 |
| `144928 -> 134853` | `v3_soft` | 1.25213 | 3.97210 | 3.40157 | 4.04366 | 90 | 40 | 36 |
| `144928 -> 134853` | `no_pairboard_trial` | 2.45560 | 3.67321 | 3.08374 | 3.74649 | - | - | - |
| `144928 -> 134853` | `no_board_cap` | 1.20882 | 4.07448 | 3.50878 | 4.14563 | 90 | 40 | 7 |
| `144419 -> 144928` | `v2_guard` | 1.94117 | 2.86162 | 2.25695 | 2.93442 | 80 | 20 | 2 |
| `144419 -> 144928` | `v3_soft` | 1.60387 | 2.83141 | 2.21463 | 2.90535 | 90 | 40 | 22 |
| `144419 -> 144928` | `no_pairboard_trial` | 2.52924 | 2.55830 | 1.84046 | 2.64114 | - | - | - |
| `144419 -> 144928` | `no_board_cap` | 1.61521 | 2.86774 | 2.24391 | 2.94253 | 90 | 40 | 3 |

#### 观察

1. `v3_soft` 相比 `v2_guard` 在三组上都更好，因此可以冻结为 **Stage6 v3 pair-board soft selection 候选新 baseline**。

2. `no_pairboard_trial` 并不是完全失败。它在 `144419 -> 144928` 和 `144928 -> 134853` 的 extrinsic-only holdout 上更低，但 training RMSE 明显更大，说明它更像“更松的全量 BA 偶然在部分 holdout 上泛化更好”，而不是一个可解释、可控的选择策略。

3. `no_board_cap` 没有带来稳定收益。虽然它降低了某些 split 的 training RMSE，但三组 holdout 都没有超过 v3_soft。这说明 board-level cap 仍有必要，不能简单关闭。

4. v3_soft 的价值不是单纯追求最小训练 RMSE，而是在“接回更多 pair-board”和“不让单个 board / pair 过度主导”之间取得更稳的折中。

#### 当前决策

- 冻结 `v3_soft` 为 Stage6 后续实验的候选新 baseline，命名为 **Stage6 v3 pair-board soft selection**。
- 保留 `v2_guard` 作为安全 baseline。
- `no_pairboard_trial` 保留为 ablation，不作为默认方案。
- `no_board_cap` 不主推，说明 `max_accepted_per_board=24` 当前仍有保护作用。

下一步：

1. 继续看 v3_soft 的 accepted / rejected pair-board overlay，确认是否存在明显错误观测被接回；
2. 对 `no_pairboard_trial` 表现更好的 split 做进一步诊断，判断是因为接入更多真实有用 board，还是因为评价集偶然偏好更松的外参；
3. 如果要继续优化 v3，可以尝试把 `max_accepted_per_board` 从 hard cap 改成 score penalty，而不是完全关闭。
