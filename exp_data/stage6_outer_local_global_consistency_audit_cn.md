# Stage6 Outer / Local-vs-Global Consistency Audit

结果目录：
`/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result_stereo/stage6_outer_local_global_audit_134853_to_144419_v1`

程序侧已经补上自动输出，新的同类结果目录会额外包含：

- `stereo_pair_board_local_global_gap_summary.txt`

例如：

- `result_stereo/stage6_outer_local_global_audit_134853_to_144419_v2_summary/stereo_pair_board_local_global_gap_summary.txt`

## 1. 这次排查想回答什么

目标是把 Stage6 当前的 holdout 大 RMSE 进一步拆开，判断问题更像：

- `global scene / board structure / stereo extrinsic` 不一致；
- 还是 `单个 pair 的 outer pose 本身就不稳`；
- 还是 `角点检测本身就坏`。

核心做法：

1. 对每个 `pair-board` 计算 `local outer rmse`
   只看该 board 在左右相机中的真实 outer corners，分别做局部 pose refit。
2. 对同一个 `pair-board` 计算 `global outer rmse`
   用当前 Stage6 scene state 的 `T_cam0_world + T_world_board + T_cam1_cam0` 去重投影 outer corners。
3. 比较两者：
   - 如果 `local` 小、`global` 大：说明角点本身可解释，但全局 scene 对不上；
   - 如果 `local` 也大：才更像这个 board 本身观测不好。

这次额外修复了一点：

- 之前 holdout pair 不在训练优化得到的 `T_cam0_world_by_pair` 里，导致 holdout 的 `global_outer_rmse` 全是 `inf`；
- 现在改成：对 holdout pair 现场走一遍和 holdout evaluator 同口径的 `stereo symmetric outer refit`，再据此计算 per-board global outer residual。

所以这版 audit 才能真正分析 holdout。

## 2. 关键总结果

### 2.1 整体 RMSE

| 实验 | training total stereo RMSE | holdout total stereo RMSE | holdout outer RMSE | holdout internal RMSE |
|---|---:|---:|---:|---:|
| Stage6 current (`134853 -> 144419`) | 1.47004 | 22.9876 | 28.2695 | 22.1668 |
| Kalibr reference（同一 evaluator） | - | 23.1303 | 28.3851 | 22.3148 |

结论：

- 我们当前 Stage6 和 Kalibr reference 在同一 evaluator 下几乎一样差；
- 所以这不是“我们外参明显比 Kalibr 差”，而更像是 evaluator / scene-state / global board structure 这一层的问题。

### 2.2 consistency audit 总览

| split | row count | global outer mean / median / max | local outer mean / median / max | `global>=15 & local<=3` |
|---|---:|---:|---:|---:|
| training | 204 | 2.988 / 2.324 / 30.830 | 0.332 / 0.267 / 1.460 | 4 |
| holdout | 204 | 27.817 / 25.868 / 62.093 | 0.434 / 0.369 / 4.009 | 201 |

最重要的结论：

- `holdout` 的 204 个 `pair-board` 里，有 201 个都是 `local_good_global_bad`；
- 也就是绝大多数 holdout board：
  - `local outer refit` 很好；
  - 但 `global scene` 重投影非常差。

这说明：

> 当前 Stage6 holdout 大 RMSE 的主因，不是 outer corners 本身检测坏，而是“训练得到的全局 stereo scene / board structure / 外参约束”放到 holdout 上后，对不上 holdout 的真实几何。

## 3. Top bad holdout pair

按 holdout overall RMSE 排序，前 12 个最差 pair：

| pair | overall | outer | internal | shared boards | left frame |
|---|---:|---:|---:|---:|---|
| 79 | 27.545 | 35.865 | 26.191 | 3 | `000076_left_197103195080_mono8` |
| 81 | 27.339 | 33.690 | 26.329 | 5 | `000084_left_205699151000_mono8` |
| 45 | 27.245 | 31.952 | 26.533 | 5 | `000008_left_123603165240_mono8` |
| 80 | 26.943 | 33.480 | 25.909 | 4 | `000080_left_201503198080_mono8` |
| 43 | 26.100 | 32.633 | 25.055 | 5 | `000004_left_119203197240_mono8` |
| 78 | 24.840 | 30.041 | 24.041 | 5 | `000074_left_195003197240_mono8` |
| 42 | 24.057 | 30.245 | 23.063 | 5 | `000002_left_117203194080_mono8` |
| 65 | 23.475 | 27.695 | 22.846 | 5 | `000047_left_165603194080_mono8` |
| 67 | 23.473 | 27.661 | 22.851 | 5 | `000049_left_167903197160_mono8` |
| 66 | 23.454 | 27.669 | 22.821 | 5 | `000048_left_166703752000_mono8` |
| 46 | 23.362 | 27.321 | 22.759 | 5 | `000010_left_125699134000_mono8` |
| 51 | 23.206 | 26.960 | 22.634 | 5 | `000020_left_136503197160_mono8` |

观察：

- 这些最差 holdout pair 的 outer RMSE 普遍在 `27 ~ 36 px`；
- internal 也很大，但它是跟着 outer/global pose 一起被拉坏的，不是单独先坏。

## 4. Top bad holdout pair 的 per-board 细分

下面给几个代表性 pair 的 per-board 结果。格式是：

- `global_outer`：放进全局 scene 后的 outer RMSE
- `local_best`：单板局部 refit 后更好的那一侧 outer RMSE

### pair 42

| board | global_outer | local_best | diagnosis |
|---|---:|---:|---|
| 1 | 27.895 | 0.145 | local_good_global_bad |
| 2 | 28.837 | 0.476 | local_good_global_bad |
| 3 | 23.506 | 0.373 | local_good_global_bad |
| 4 | 39.008 | 0.609 | local_good_global_bad |
| 5 | 29.832 | 0.862 | local_good_global_bad |

### pair 43

| board | global_outer | local_best | diagnosis |
|---|---:|---:|---|
| 1 | 31.044 | 0.155 | local_good_global_bad |
| 2 | 30.580 | 0.531 | local_good_global_bad |
| 3 | 24.512 | 0.449 | local_good_global_bad |
| 4 | 41.759 | 0.694 | local_good_global_bad |
| 5 | 32.877 | 1.010 | local_good_global_bad |

### pair 45

| board | global_outer | local_best | diagnosis |
|---|---:|---:|---|
| 1 | 30.008 | 0.259 | local_good_global_bad |
| 2 | 30.183 | 0.325 | local_good_global_bad |
| 3 | 24.257 | 0.418 | local_good_global_bad |
| 4 | 41.045 | 0.452 | local_good_global_bad |
| 5 | 31.941 | 1.181 | local_good_global_bad |

### pair 79

| board | global_outer | local_best | diagnosis |
|---|---:|---:|---|
| 1 | 33.987 | 0.607 | local_good_global_bad |
| 2 | 35.586 | 0.517 | local_good_global_bad |
| 3 | 21.397 | 0.480 | local_good_global_bad |
| 4 | 62.093 | inf | insufficient_data |
| 5 | 31.165 | 0.606 | local_good_global_bad |

### pair 81

| board | global_outer | local_best | diagnosis |
|---|---:|---:|---|
| 1 | 32.410 | 0.393 | local_good_global_bad |
| 2 | 29.537 | 0.412 | local_good_global_bad |
| 3 | 25.772 | 0.520 | local_good_global_bad |
| 4 | 43.508 | 0.744 | local_good_global_bad |
| 5 | 34.571 | 1.100 | local_good_global_bad |

这里已经非常明确：

- 同一 pair 里几乎所有 board 都是 `local 好、global 差`；
- 而不是“只有单个 board 的 outer 本身很差”；
- 尤其 board4 经常是最差的那一块，但 board1/2/3/5 也都同步偏大。

这更像：

1. `holdout pair pose` 和 `训练 scene structure` 之间整体不一致；
2. 或者 `训练得到的 T_world_board_by_id` 对 holdout 不适配；
3. 而不是单独某块板的 detection 崩了。

## 5. Training 端有什么异常

training 并不是完全没问题，但问题集中在很少数 frame：

| pair | board | global_outer | local_outer | left frame |
|---|---:|---:|---:|---|
| 16 | 4 | 30.830 | 1.225 | `000028_left_215103202000_mono8` |
| 16 | 2 | 30.735 | 1.284 | `000028_left_215103202000_mono8` |
| 16 | 3 | 28.844 | 1.138 | `000028_left_215103202000_mono8` |
| 16 | 5 | 27.363 | 0.877 | `000028_left_215103202000_mono8` |

这说明 training 自己也存在少量 `local_good_global_bad` 的坏帧，只是数量远少于 holdout。

所以可以合理怀疑：

- 训练 scene structure 本身就被少量坏 pair 拉偏过；
- 然后 holdout 又被拿来强行套这套 structure，于是几乎全体 holdout board 都 global bad。

## 6. 当前最合理的归因

基于这次 audit，我认为优先级最高的判断是：

### 结论 A

> 当前 Stage6 的主要矛盾不是“外角点检测坏”，而是“global scene / board structure / extrinsic 这一层对 holdout 不一致”。

证据：

- holdout `local_outer_rmse` 基本都很好；
- holdout `global_outer_rmse` 基本全部很大；
- Kalibr reference 在同一 evaluator 下也同样很大。

### 结论 B

> 问题大概率不只是外参 `T_cam1_cam0` 单独错了。

原因：

- 如果只是外参明显错，而 Kalibr 外参明显更准，应该看到 Kalibr holdout 显著更小；
- 但现在 ours 和 Kalibr 非常接近。

更像是：

- holdout evaluator 中固定使用的 `T_world_board_by_id` / global board structure 不适配；
- 或者 training scene 里本来就有少量 `local_good_global_bad` 的 pair 污染了 structure；
- 然后 holdout 在这个 structure 下全部被拉大。

## 7. 这一步之后最值得做什么

这次 audit 已经把“问题属于哪一层”基本定出来了。下一步最值得做的不是继续调角点，而是：

1. 做 `Stage6 holdout local-vs-global` 的正式输出摘要
   把 top bad holdout pair / board 单独导出，后面每次实验都能直接对比。
2. 做 `training bad-pair suppression / consistency gate`
   把像 training `pair 16` 这种 `local_good_global_bad` 的坏 pair-board 从 stereo scene structure 里限制掉，避免污染 `T_world_board_by_id`。
3. 把 Stage5 的思路搬到 Stage6
   不只按 `pair` 选择，而是按 `pair-board` 逐步接回；
   对 `local_good_global_bad` 的坏 board 在 final BA 前做 gate。

## 8. consistency gate 快速对照实验

为了验证“是不是少量 training 坏板块在污染 scene structure”，又补了两组不改前端的对照：

- `gate15`
  - 条件：`local <= 3 px` 且 `global >= 15 px` 的 training `pair-board` 不进 final BA
- `gate5`
  - 条件：`local <= 3 px` 且 `global >= 5 px` 的 training `pair-board` 不进 final BA

结果目录：

- `result_stereo/stage6_outer_local_global_gate15_134853_to_144419`
- `result_stereo/stage6_outer_local_global_gate5_134853_to_144419`

### 8.1 数值结果

| 方案 | rejected training pair-board | training total RMSE | holdout total RMSE | holdout outer RMSE | holdout internal RMSE |
|---|---:|---:|---:|---:|---:|
| baseline audit v1 | 0 | 1.47004 | 22.9876 | 28.2695 | 22.1668 |
| gate15 | 4 | 1.47004 | 22.9876 | 28.2695 | 22.1668 |
| gate5 | 47 | 1.67116 | 24.3465 | 29.3837 | 23.5734 |

### 8.2 为什么 gate15 完全没效果

`gate15` 挡掉的 4 个坏板块全部来自：

- `pair 16 / board 2,3,4,5`

但这 4 个 `pair-board` 本来就**没有进入 final selected pair-board bundle**，所以：

- gate 虽然生效了；
- 但它挡掉的是“本来就没进 BA 的东西”；
- 因此最终训练和 holdout 数值完全不变。

### 8.3 为什么 gate5 反而更差

`gate5` 一共挡掉了 47 个 training `pair-board`，其中真正和 final selected bundle 有重叠的只有 2 个：

- `(pair 29, board 2)`，`global=6.748 px`，`local=0.108 px`
- `(pair 36, board 5)`，`global=7.286 px`，`local=0.093 px`

也就是说：

- `gate5` 开始挡掉一些“虽然 global 稍大，但其实仍然提供有效约束”的板块；
- 结果不是修好 structure，而是削弱了训练约束，导致 training / holdout 都变差。

### 8.4 gate 实验的结论

这组对照说明：

1. 当前最严重的 training `local_good_global_bad` 坏板块，已经大多没进入 final BA，不是 holdout 爆炸的主因；
2. 简单把阈值放宽到 `global>=5` 并不能修问题，反而会伤到正常约束；
3. 所以当前 Stage6 的核心问题，不是“再多挡一点 training 坏板块”就能解决的。

更像是：

- holdout evaluator 对 `training global board structure` 的依赖太强；
- 或者 cross-dataset holdout 本身不该直接复用训练的 `T_world_board_by_id` 作为评估世界结构。

## 9. 一个很关键的补充验证

为了进一步确认“是不是 holdout 观测本身其实没问题”，又基于现有 audit 做了一个近似统计：

- 对每个 holdout `pair-board`，用左右相机各自的 `local outer rmse` 合成一个近似的 stereo-local outer RMSE；
- 再和当前 `global outer rmse` 对比。

结果：

| 指标 | 数值 |
|---|---:|
| holdout board-level approx stereo-local outer RMSE（mean / median / max） | `0.850 / 0.665 / 4.009` |
| holdout board-level global outer RMSE（mean / median / max） | `27.599 / 25.844 / 45.440` |
| approx aggregate local stereo outer RMSE | `1.024` |
| 当前 holdout outer RMSE | `28.2695` |

这个对比几乎可以直接说明：

> holdout 这批 outer 观测本身是可以被很好解释的；真正把它拉到 28px 的，不是观测本身，而是“把它们塞进训练得到的 global stereo scene / board structure 后”的那一步。

## 10. 和已有 Stage6 结果的横向对比

再看一下之前几组 Stage6 v2 结果：

| 结果目录 | training total | holdout total | holdout outer | holdout internal |
|---|---:|---:|---:|---:|
| `stage6_v2_134853_stereo_kalibr_style_diag_localba_trialpool_refcompare_cachecheck_fixed` | 1.79886 | 1.53217 | 1.59022 | 1.52411 |
| `stage6_v2_144928_stereo_kalibr_style_diag_localba_trialpool_refcompare` | 2.15327 | 3.18250 | 2.01709 | 3.30952 |
| `stage6_v2_144419_stereo_kalibr_style_diag_localba_trialpool_refcompare` | 3.82812 | 2.33539 | 1.87787 | 2.39180 |
| **本次 cross-dataset `134853 -> 144419`** | **1.47004** | **22.9876** | **28.2695** | **22.1668** |

这说明：

- Stage6 不是“普遍都很差”；
- 同类 Stage6 流程在其它结果里可以跑到 `1.5 ~ 3.2 px` 的 holdout；
- 真正异常的是这次 **cross-dataset holdout**；
- 所以更像是 `cross-dataset + training global board structure reuse` 这个评估口径本身在放大误差。

## 11. 本次结论一句话版

> Stage6 这次排查已经明确：当前大 holdout RMSE 主要不是角点检测坏，而是“训练得到的全局多板 scene / board structure 放到 holdout 上整体对不上”。local outer refit 很好，global outer residual 却几乎全部爆炸；并且 Kalibr 在同一 evaluator 下也同样大，说明当前真正该修的是 Stage6 的 scene-structure / consistency 策略，而不是继续盯角点本身。

## 12. 进一步修正：新增 extrinsic-only holdout 口径

用户指出：如果我们真正想评估双目外参本身，就不应该把训练集的全局多板结构强行带到验证集。

因此代码中新增了一个并行输出口径：

- 旧口径：`holdout`
  - 使用训练优化得到的 `T_world_board_by_id` / global multi-board scene；
  - 再对 holdout 做重投影；
  - 如果训练集和验证集的 board 摆放关系不同，这个指标会被 scene structure mismatch 拉爆。
- 新口径：`holdout_extrinsic_only`
  - 不使用训练集 `T_world_board_by_id`；
  - 对验证集每个 `pair-board`，固定当前 `T_cam1_cam0`；
  - 现场局部 refit 一个 `T_cam0_board`，让左右相机 outer corners 共同解释得最好；
  - 再统计该 board 的 outer/internal residual；
  - 这个指标更接近“只看当前双目外参能否解释同一时刻左右图像”。

新增输出字段在：

- `stereo_reprojection_summary.txt`
  - `holdout_extrinsic_only_total_stereo_rmse`
  - `holdout_extrinsic_only_outer_only_rmse`
  - `holdout_extrinsic_only_internal_only_rmse`
  - `holdout_extrinsic_only_cam0_rmse`
  - `holdout_extrinsic_only_cam1_rmse`
- `stereo_reference_holdout_summary.txt`
  - `ours_extrinsic_only_holdout_*`
  - `reference_extrinsic_only_holdout_*`

### 12.1 实验结果

结果目录：

- `result_stereo/stage6_extrinsic_only_holdout_134853_to_144419_v2_localstereo`

| 评估口径 | Ours total | Ours outer | Ours internal | Kalibr total | Kalibr outer | Kalibr internal | Ours - Kalibr total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 global-structure holdout | 22.9876 | 28.2695 | 22.1668 | 23.1303 | 28.3851 | 22.3148 | -0.1427 |
| 新 extrinsic-only holdout | 27.1451 | 28.0447 | 27.0196 | 27.3827 | 28.3125 | 27.2529 | -0.2376 |

外参本身与 Kalibr 的差异：

| 指标 | 数值 |
|---|---:|
| rotation delta vs Kalibr | 0.193126 deg |
| translation delta vs Kalibr | 0.000842786 m |
| baseline length ours | 0.0639082 m |
| baseline length Kalibr | 0.0642588 m |
| baseline length delta | 0.000350686 m |

### 12.2 这个结果说明什么

这个结果有两层含义：

1. 我们的外参和 Kalibr 外参非常接近。
   - rotation 只差约 `0.19 deg`；
   - translation 只差约 `0.84 mm`；
   - baseline length 只差约 `0.35 mm`。
2. 即使改成不依赖训练 global board structure 的 `extrinsic-only` 口径，holdout 仍然约 `27 px`，而 Kalibr reference 也是约 `27 px`。

所以这一步推翻了一个过早判断：

> 大 RMSE 不只是训练 global board structure mismatch；至少在 `134853 -> 144419` 这组 cross-dataset 中，同一块 holdout board 的左右图像，在固定外参下也很难被一个局部双目 board pose 同时解释。

更通俗地说：

- 左相机自己看这块板，可以解释得很好；
- 右相机自己看这块板，也可以解释得很好；
- 但是要求“同一个 3D board pose + 同一个固定双目外参”同时解释左右两边时，就解释不好；
- 而且 Kalibr 外参也同样解释不好。

因此目前更可疑的是：

- holdout 左右图像 pairing 虽然文件名时间戳完全一致，但实际同步/相机流对应关系仍需继续核验；
- 左右相机内参或 camchain 中 cam0/cam1 方向、左右相机定义需要再次确认；
- 当前验证集和训练集的多板/target 几何在 stereo evaluator 中是否有额外约定差异；
- 不能仅仅归因于我们的 Stage6 外参优化结果差。

### 12.3 和前一版局部统计的区别

前面 audit 中的 `holdout_approx_stereo_local_outer_rmse_px ≈ 1.0021` 并不是严格双目外参指标。

它的含义是：

- 左相机单独 local pose refit 后 residual 很小；
- 右相机单独 local pose refit 后 residual 很小；
- 然后把两边分别统计合起来。

它没有要求左右两边共享同一个 `T_cam0_board` 和同一个 `T_cam1_cam0`。

新的 `holdout_extrinsic_only` 则要求：

- 同一块 board 只有一个局部 `T_cam0_board`；
- 右相机必须通过 `T_cam1_cam0 * T_cam0_board` 投影；
- 所以它才真正检查双目外参一致性。

这也是为什么两者数值会从 `~1 px` 跳到 `~27 px`。

## 13. 下一步建议

当前不应该继续盲目调 Stage6 selection 参数，而应该先做三个基础核验：

1. **左右图像同步核验**
   - 当前 `stereo_pairing_summary.txt` 显示：
     - `pairing_mode: filename_timestamp_exact`
     - `max_abs_pair_timestamp_delta_ms: 0`
   - 但仍建议回到 rosbag/header stamp 或原始采集日志确认左右硬件触发是否真正同步。
2. **左右相机定义和外参方向核验**
   - 确认 `stereo_4_2-3-camchain.yaml` 中的 `T_cam1_cam0` 是否和程序中的 left/right、cam0/cam1 完全一致；
   - 尤其要确认有没有把 `T_cam0_cam1` 当成 `T_cam1_cam0`，或左右相机路径反了。
3. **输出 pair-board 级 strict stereo-local residual**
   - 当前 summary 只有整体 `holdout_extrinsic_only`；
   - 下一步应额外输出每个 holdout `pair-board` 在 extrinsic-only 口径下的 outer/internal residual；
   - 这样能明确到底是所有 board 系统性偏，还是只有 board4/5 或某些时间段偏。

这一步做完后，再决定是否进入：

- Stage6 close-edge aware selection；
- pair-board soft weighting；
- 或者重新检查 stereo timestamp / camchain / 数据配对。
