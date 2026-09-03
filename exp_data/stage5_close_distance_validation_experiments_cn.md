# Stage5 Close-Distance 验证集实验记录

本文记录近期围绕 close-distance 验证集做的 Stage5 New Baseline 实验。重点不是把不同 close 图像简单混成一个大验证集，而是按数据来源分别评估，再汇总判断：

- close-distance 本身是否会导致泛化失败；
- 失败是否集中在特定 board / 视场边缘；
- New Baseline 与 Kalibr 在同一 holdout evaluator 下的差异；
- 不同 board layout 的 close 数据是否适合直接合并。

## 1. 验证集使用原则

不同数据集的多板摆放关系可能不同，也就是 `T_reference_board` 不同。因此这些 close 图像不适合直接混在一个 `test-image` 文件夹里当作单一 holdout 统一评估。

更合理的做法是：

1. 每个 close 子数据集单独评估；
2. 每个子集使用自己对应的多板结构 / scene 关系；
3. 最后在表格层面汇总 overall / outer / internal / board-level 结果。

因此本文把这些数据称为 **close-distance stress benchmark collection**，而不是一个单一验证集。

## 2. Close144419：普通 close 泛化验证

这组实验使用 `image/close_dis_dataset/stereo_dataset_20260430_144419` 作为 close holdout。结果表明：close-distance 场景本身不是问题，New Baseline 在 close144419 上稳定优于 Kalibr。

| 输出目录 | 训练集 | Backend holdout | Backend outer | Backend internal | Kalibr holdout | Kalibr outer | Kalibr internal | Backend - Kalibr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `stage5_newbaseline_134853_right_val_close_144419_v2a` | `134853 right` | `1.09904` | `0.23315` | `1.16983` | `1.17948` | `0.73143` | `1.22897` | `-0.08044` |
| `stage5_newbaseline_144928_right_val_close_144419_v2b` | `144928 right` | `1.08624` | `0.18930` | `1.15724` | `1.17839` | `0.73143` | `1.22778` | `-0.09215` |
| `stage5_newbaseline_134853_right_val_close_144419_v2c` | `134853 right` | `1.09904` | `0.23315` | `1.16983` | `1.17948` | `0.73143` | `1.22897` | `-0.08044` |
| `stage5_newbaseline_144419_right_val_close_144419_closeedge_diag` | `144419 right` | `1.08211` | `0.10997` | `1.15420` | `1.17843` | `0.73143` | `1.22778` | `-0.09632` |

补充：排除 reference board / board1 后，New Baseline 仍然优于 Kalibr。

| 输出目录 | Backend 不含 board1 | Kalibr 不含 board1 | Backend - Kalibr |
|---|---:|---:|---:|
| `stage5_newbaseline_134853_right_val_close_144419_v2a` | `0.94409` | `1.00631` | `-0.06222` |
| `stage5_newbaseline_144928_right_val_close_144419_v2b` | `0.93895` | `1.00396` | `-0.06501` |
| `stage5_newbaseline_134853_right_val_close_144419_v2c` | `0.94409` | `1.00631` | `-0.06222` |

说明：

- `v2b` 是三组 v2 中最好的，即 `144928 right -> close144419`。
- `v2a` 和 `v2c` 数值完全一致，说明它们大概率是同一配置重复运行，或者预期差异没有通过命令参数真正生效。
- close144419 的 outer RMSE 明显优于 Kalibr，说明 New Baseline 的外四角点 / pose refit 在该 close 场景下表现很好。

结论：

> close-distance 本身不是导致高 holdout RMSE 的根因。至少在 close144419 上，New Baseline 能稳定泛化，并且优于 Kalibr。

## 3. Close144928 / Close192347：close-edge board5 压力测试

另外两组 close 数据表现明显更难，主要问题集中在边缘 board5。

注意：本节前两行是 **outer subpix ROI clamp 修复前** 的历史结果，保留它们是为了说明问题来源。后续第 6 节记录了 close-edge outer subpix boost 修复后的新结果。

| Train -> close validation | Backend holdout | Backend outer | Backend internal | Kalibr holdout | Backend - Kalibr | 现象 |
|---|---:|---:|---:|---:|---:|---|
| `144928 -> close144928` | `10.6588` | `6.75481` | `11.0856` | `10.2290` | `+0.42986` | 极难 close-edge case，board5 主导误差 |
| `192347 -> close192347` | `10.2514` | `3.57713` | `10.8436` | `10.4448` | `-0.19347` | 极难 close-edge case，board5 主导误差 |

这里要注意：这两组不是简单说明 New Baseline 泛化差，因为 Kalibr 在相同 evaluator 下也出现类似高误差。

## 4. Force-Include Board5 实验

为了验证 board5 是否只是被 selection 错误过滤，做了强制候选实验：

输出目录：

`stage5_force_include_close_edge_board5_144928_right_val_close_144928_right`

候选文件：

`exp_data/close_edge_board5_force_include_candidates.csv`

实验方式：

- 选出 7 个 close-edge board5 候选；
- 强制绕过 score / coverage / board cap / frame cap；
- 但仍然必须通过 short backend 的 RMSE delta 检查；
- 如果加入后优化变差，就拒绝。

结果：

| Frame label | Board | Trial RMSE | Global delta | Outer delta | Internal delta | 决策 |
|---|---:|---:|---:|---:|---:|---|
| `000007_right_431303195240_mono8` | 5 | `6.89311` | `+0.18575` | `+1.24598` | `+0.02971` | rejected |
| `000009_right_433603197080_mono8` | 5 | `7.09748` | `+0.18111` | `+1.19814` | `+0.03514` | rejected |
| `000010_right_434603192000_mono8` | 5 | `8.93082` | `+0.29879` | `+1.74147` | `+0.04318` | rejected |
| `000090_right_522303194080_mono8` | 5 | `9.36993` | `+0.32009` | `+1.85821` | `+0.03578` | rejected |
| `000091_right_523303192000_mono8` | 5 | `9.30625` | `+0.31428` | `+1.82003` | `+0.03991` | rejected |
| `000092_right_524403198240_mono8` | 5 | `11.7358` | `+0.52409` | `+2.60480` | `+0.06329` | rejected |
| `000093_right_525603195160_mono8` | 5 | `8.13481` | `+0.28445` | `+1.69406` | `+0.03795` | rejected |

解释：

- 这些 board5 不是因为 score 或 cap 被误杀；
- 它们被真正试加入 backend 后，会让 RMSE 变差；
- 因此 trial backend selection 拒绝它们是合理的。

## 5. Board5 outer corner residual vector 诊断

为了判断 board5 的问题是否是 Stage5 独有，输出了四个 outer corner residual vector，对比：

- Backend DS；
- Kalibr reference。

输出目录：

`result_may/stage5_force_include_close_edge_board5_144928_right_val_close_144928_right/close_edge_board5_outer_corner_vector_diagnostics`

关键现象：

- Backend DS 和 Kalibr 的 residual arrow 方向高度一致；
- 7 个 board5 frame 中，大部分 residual direction cosine 接近 `1.0`；
- 说明两种相机模型在这些点上都往相似方向偏。

部分 residual 分解：

| Frame label | Method | Mean norm | Max norm | Dominant component |
|---|---|---:|---:|---|
| `000007_right_431303195240_mono8` | Backend DS | `15.80967` | `24.9269` | radial |
| `000007_right_431303195240_mono8` | Kalibr | `14.98538` | `23.4769` | radial |
| `000090_right_522303194080_mono8` | Backend DS | `15.44592` | `26.0161` | tangential |
| `000090_right_522303194080_mono8` | Kalibr | `14.82436` | `25.0581` | tangential |
| `000092_right_524403198240_mono8` | Backend DS | `16.96273` | `40.1633` | tangential |
| `000092_right_524403198240_mono8` | Kalibr | `15.58288` | `34.9183` | tangential |
| `000093_right_525603195160_mono8` | Backend DS | `14.78530` | `28.0063` | tangential |
| `000093_right_525603195160_mono8` | Kalibr | `13.34931` | `24.5582` | tangential |

结论：

> 如果 Backend DS 和 Kalibr 在同一 board5 上出现方向一致的大 residual，那么问题更像是 close-edge board/view 几何本身困难，或者边缘观测条件导致的共同困难，而不是 Stage5 单独泛化失败。

## 6. 最新进展：Close-edge outer subpix boost

### 6.1 问题回顾

进一步检查 `close144928` 的 holdout overlay 后发现，失败并不只是 backend residual model 的问题。很多 close-edge board4 / board5 的外四角点在近距离、大面积、边缘位置下亚像素 refinement 不够充分，导致外点 pose refit 和后续 internal generation 都被带偏。

旧实现中，虽然 `outer_subpix_scale` 可以调大，但实际 `final_subpix_window_radius` 会被 `verification_roi_radius - 2` clamp 住。结果是：

- 配置里 raw subpix radius 已经很大；
- 但真正传给 `cornerSubPix` 的窗口仍只有约 `14-39 px`；
- close-edge 大 board 的外四角点没有足够搜索范围。

### 6.2 方法

新的实验分支做了两步：

1. 去掉 `verification_roi_radius` 对 outer subpix window 的硬 clamp，让 `outer_subpix_scale` 真正生效；
2. 加入 close-edge aware boost：
   - 普通点使用 `outer_subpix_scale = 0.35`；
   - 只有满足大视场角 / 大面积 / 靠边条件的点才放大窗口；
   - 当前实验使用 `close_edge_outer_subpix_multiplier = 1.4`；
   - 如果 DS camera 暂时不可用，则用图像中心归一化半径作为 polar-like fallback，避免所有 polar 都变成 0 导致 boost 永远不触发。

### 6.3 关键实验结果

| 输出目录 | 训练 -> 验证 | outer subpix 策略 | boost 点数 | Backend holdout | Backend outer | Backend internal | Kalibr holdout | Backend - Kalibr |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `outer_subpix_polar_boost_proxy_144928_to_close144928` | `144928 -> close144928` | `scale=0.35 + polar/proxy boost x1.4` | `88 / 2576` | `0.99102` | `0.23905` | `1.05265` | `1.41363` | `-0.42262` |
| `outer_subpix_unclamped_144928_to_close144928` | `144928 -> close144928` | `scale=0.5, no boost, unclamped` | `0 / 2576` | `1.00466` | `0.25459` | `1.06680` | `1.40695` | `-0.40229` |
| `outer_subpix_polar_boost_144928_to_close144928` | `144928 -> close144928` | `scale=0.35 + boost x1.2, 但 polar 未触发` | `0 / 2576` | `4.07353` | `2.35534` | `4.25391` | `3.95085` | `+0.12268` |
| 历史 clamped 结果 | `144928 -> close144928` | `scale=0.8, 但被 ROI clamp` | `0 / 2576` | `10.6588` | `6.75481` | `11.0856` | `10.2290` | `+0.42986` |

### 6.4 Boost 触发情况

`outer_subpix_polar_boost_proxy_144928_to_close144928` 中：

| 数据 | 数值 |
|---|---:|
| outer corner debug 总数 | `2576` |
| close-edge boost 触发数 | `88` |
| 普通点 final radius 均值 | `39.84 px` |
| boost 点 raw radius 均值 | `55.64 px` |
| boost 点 final radius 均值 | `77.84 px` |
| boost 点 polar/proxy 均值 | `54.01 deg` |
| boost 点 area ratio 均值 | `0.0254` |

这说明新逻辑不是全局放大窗口，而是只对少量 close-edge hard cases 放大。

### 6.5 Cross-dataset sanity：144419 -> 144928

为了验证该策略不是只对 close144928 有效，又跑了 `144419 right -> 144928 right`。

| 输出目录 | 训练 -> 验证 | boost 点数 | Backend holdout | Backend outer | Backend internal | Kalibr holdout | Backend - Kalibr |
|---|---|---:|---:|---:|---:|---:|---:|
| `outer_subpix_polar_boost_proxy_144419_to_144928` | `144419 -> 144928` | `8 / 1708` | `2.26618` | `0.93815` | `2.39328` | `5.60878` | `-3.34260` |

对比历史 New Baseline：

| 方案 | Holdout overall | Holdout outer | Holdout internal |
|---|---:|---:|---:|
| 历史 New Baseline `144419 -> 144928` | `6.57076` | `15.2445` | `4.09776` |
| close-edge outer subpix boost proxy | `2.26618` | `0.93815` | `2.39328` |

这说明 `144419 -> 144928` 之前的高 outer RMSE 很大程度也来自 close-edge outer corner refinement 不充分，而不是单纯 backend selection 或 internal residual 的问题。

### 6.6 当前结论

当前 close-distance 实验可以重新分成三类：

1. **普通 close 泛化**：  
   `close144419` 表现很好，New Baseline 稳定优于 Kalibr。

2. **极端 close-edge stress case**：  
   旧实现下 `close144928` 和 `close192347` 的高误差主要集中在 board5，Kalibr 也出现类似失败。

3. **outer subpix 修复后的 close-edge case**：  
   `close144928` 已经从 `10.6588 px` 降到 `0.99102 px`，说明这类问题存在明确的 frontend 改进空间。

因此，汇报时建议这样表达：

> 我们额外构建了 close-distance stress benchmark，但没有直接把不同 board layout 的 close 图像混成一个统一验证集，而是按场景分别评估。早期结果显示，困难主要集中在近距离、大面积、边缘 board5；进一步诊断发现，问题很大一部分来自 outer corner subpixel refinement 的有效窗口被 ROI clamp 限制。去掉该限制后，再对大 polar / 大面积 / 靠边的少量点做 adaptive boost，`close144928` 的 holdout RMSE 从 `10.6588 px` 降到 `0.99102 px`。这说明 close-edge hard case 不是完全无解，而是需要更适配强鱼眼边缘几何的 frontend refinement。

## 7. 后续建议

1. close 数据继续按子场景单独跑，不建议直接混成一个 test-image。
2. 将 close-edge outer subpix boost 继续在 `134853 -> 144419`、`144928 -> 134853` 上验证，确认没有副作用。
3. 对 `close_edge_outer_subpix_multiplier` 做小范围 sweep，例如 `1.2 / 1.3 / 1.4`。
4. 对 close-edge board5 保留 residual-vector diagnostic，用于论文或汇报中解释困难样本。
5. 如果要进一步提升 close-edge 表现，应继续研究：
   - board5 边缘角点检测质量；
   - 近距离大视场下的 residual model；
   - board 平面姿态退化；
   - per-board / per-polar-bin 误差建模。
6. 当前不建议强行把 rejected board5 接入 backend，因为 force-include 实验已经说明旧 frontend 下它们会破坏优化；更合理的方向是先提升 outer corner refinement 质量。
