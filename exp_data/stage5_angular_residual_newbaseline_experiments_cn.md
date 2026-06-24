# Stage5 Angular Residual on New Baseline 实验记录

本页记录在 **New Baseline 默认配置** 下重新进行的 A/B/E50 angular residual 对比实验。

数据集：

- Train: `20260430_144928/right`
- Holdout: `20260430_134853/right`
- Camera model: DS
- Baseline: New Baseline 默认 selection/frontend

## 1. 实验组

| 组别 | backend residual model | 说明 |
|---|---|---|
| A | `image_plane` | New Baseline 默认 image-plane residual |
| B | `sphere_angular` | final backend 全部点使用球面 angular residual |
| E50 | `hybrid_edge_angular` | polar angle >= 50 deg 的点使用 angular，其余仍用 image-plane |

## 2. Backend residual 分配

| 组别 | image residual count | angular residual count | outer image / angular | internal image / angular |
|---|---:|---:|---:|---:|
| A | 3017 | 0 | 364 / 0 | 2653 / 0 |
| B | 0 | 3017 | 0 / 364 | 0 / 2653 |
| E50 | 2585 | 432 | 295 / 69 | 2290 / 363 |

## 3. Training / Holdout pixel RMSE

| 组别 | training overall | training outer | training internal | holdout overall | holdout outer | holdout internal | Kalibr holdout | ours - Kalibr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1.08017 | 0.203807 | 1.14941 | 3.25691 | 0.815071 | 3.46098 | 3.28411 | -0.02720 |
| B | 3.08053 | 2.75138 | 3.12299 | 4.04465 | 3.06447 | 4.16173 | 3.28411 | +0.76054 |
| E50 | 1.12710 | 0.366751 | 1.19424 | 3.27176 | 0.846025 | 3.47586 | 3.28411 | -0.01234 |

## 4. Backend angular diagnostics

| 组别 | optimized overall angular RMSE | outer angular RMSE | internal angular RMSE |
|---|---:|---:|---:|
| A | 0.000698022 | 0.000291529 | 0.000736495 |
| B | 0.002037070 | 0.002768100 | 0.001915130 |
| E50 | 0.000749146 | 0.000631181 | 0.000763912 |

## 5. Polar-bin angular RMSE

| 组别 | 0-30 deg | 30-50 deg | 50-70 deg | 70-85 deg | 85-100 deg |
|---|---:|---:|---:|---:|---:|
| A | 0.000864 | 0.000574 | 0.000474 | 0.000348 | N/A |
| B | 0.001948 | 0.002027 | 0.002270 | 0.005326 | N/A |
| E50 | 0.000861 | 0.000569 | 0.000869 | 0.002715 | N/A |

## 6. 简短结论

- A 仍然是本 split 上最好的方案：holdout overall 最低，backend angular RMSE 也最低。
- B full sphere angular 虽然 `success=1`，但 pixel RMSE 和 angular RMSE 都明显变差，当前不适合作为主线。
- E50 能正常构造 angular residual，约 14.3% 点进入 angular residual，但本组结果略差于 A。
- 这组 New Baseline 口径下，暂时没有看到 E50 优于 A；建议后续只作为候选分支继续多数据集验证，不替换默认 baseline。

## 7. 输出目录

| 组别 | output |
|---|---|
| A | `result_may/stage5_ang_A_newbaseline_default_144928_right_val_134853_right` |
| B | `result_may/stage5_ang_B_full_sphere_newbaseline_default_144928_right_val_134853_right` |
| E50 | `result_may/stage5_ang_E50_hybrid_newbaseline_default_144928_right_val_134853_right` |


## 8. 第二轮 angular 分支：B3 / B5 / B7

这一轮继续探索 full angular 失败后的替代方向：不再把所有 residual 直接替换成 angular，而是尝试辅助约束、outer/internal 分裂，以及连续 polar 权重。

需要特别注意：这一轮 B3/B5/B7 的输出实际启用了 `trial backend frame-board selection + pre-backend filter`，backend 输入点数为 **3596**；而上面的 A/B/E50 记录中 `effective_pre_backend_filter_mode=off`，backend 输入点数为 **3017**。因此前面的 A/B/E50 不能直接作为严格对照。

后来已补跑同口径 A-current：`image_plane` backend residual，backend 输入同样为 **3596** 点，可作为 B3/B5/B7 的严格对照。

### 8.1 实验组说明

| 组别 | 方法 | 目的 |
|---|---|---|
| A-current | image-plane | 同口径默认 image-plane 对照，backend input = 3596 |
| B3-0.05 | image-plane + angular auxiliary, weight=0.05 | 保留像素 residual，同时给每个点加低权重 angular 辅助约束 |
| B3-0.10 | image-plane + angular auxiliary, weight=0.10 | 检查 angular auxiliary 权重增大后是否带来变化 |
| B5a | outer angular + internal image | 判断 full angular 失败是否主要来自 outer 点 |
| B5b | outer image + internal angular | 判断 full angular 失败是否主要来自 internal 点 |
| B7 | polar continuous hybrid, threshold=50, temperature=10 | 用连续 sigmoid 权重替代 E50 hard threshold，避免残差类型突变 |

### 8.2 Backend residual 构造数量

| 组别 | backend points | image residual | angular residual | angular auxiliary | outer image / angular | internal image / angular |
|---|---:|---:|---:|---:|---:|---:|
| A-current | 3596 | 3596 | 0 | 0 | 444 / 0 | 3152 / 0 |
| B3-0.05 | 3596 | 3596 | 3596 | 3596 | 444 / 444 | 3152 / 3152 |
| B3-0.10 | 3596 | 3596 | 3596 | 3596 | 444 / 444 | 3152 / 3152 |
| B5a | 3596 | 3152 | 444 | 0 | 0 / 444 | 3152 / 0 |
| B5b | 3596 | 444 | 3152 | 0 | 444 / 0 | 0 / 3152 |
| B7 | 3596 | 3596 | 3596 | 0 | 444 / 444 | 3152 / 3152 |

B7 的 residual type assignment 显示平均权重：image-plane 约 0.7566，angular 约 0.2434；polar angle >= 50 deg 的点数为 548。

### 8.3 Training / Holdout pixel RMSE

| 组别 | training overall | training outer | training internal | holdout overall | holdout outer | holdout internal | Kalibr holdout | ours - Kalibr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-current | 1.08004 | 0.202844 | 1.14930 | 3.25668 | 0.814346 | 3.46075 | 3.28411 | -0.02743 |
| B3-0.05 | 1.08004 | 0.202844 | 1.14930 | 3.25668 | 0.814346 | 3.46075 | 3.28411 | -0.02743 |
| B3-0.10 | 1.08004 | 0.202844 | 1.14930 | 3.25668 | 0.814346 | 3.46075 | 3.28411 | -0.02743 |
| B5a | 1.09294 | 0.251767 | 1.16177 | 3.26435 | 0.824031 | 3.46865 | 3.28411 | -0.01976 |
| B5b | 1.08032 | 0.200340 | 1.14966 | 3.25627 | 0.812451 | 3.46037 | 3.28411 | -0.02784 |
| B7 | 1.08482 | 0.228989 | 1.15374 | 3.25851 | 0.817834 | 3.46260 | 3.28411 | -0.02560 |

### 8.4 Backend angular diagnostics

| 组别 | optimized angular RMSE | outer angular RMSE | internal angular RMSE |
|---|---:|---:|---:|
| A-current | 0.000599840 | 0.000267851 | 0.000632760 |
| B3-0.05 | 0.000599840 | 0.000267851 | 0.000632760 |
| B3-0.10 | 0.000599840 | 0.000267851 | 0.000632760 |
| B5a | 0.000601315 | 0.000590331 | 0.000602846 |
| B5b | 0.000634812 | 0.000229977 | 0.000672533 |
| B7 | 0.000600691 | 0.000289165 | 0.000632360 |

### 8.5 Polar-bin angular RMSE

| 组别 | 0-30 deg | 30-50 deg | 50-70 deg | 70-85 deg |
|---|---:|---:|---:|---:|
| A-current | 0.000725 | 0.000504 | 0.000435 | 0.000327 |
| B3-0.05 | 0.000725 | 0.000504 | 0.000435 | 0.000327 |
| B3-0.10 | 0.000725 | 0.000504 | 0.000435 | 0.000327 |
| B5a | 0.000718 | 0.000503 | 0.000465 | 0.001127 |
| B5b | 0.000766 | 0.000538 | 0.000445 | 0.000290 |
| B7 | 0.000722 | 0.000501 | 0.000464 | 0.000613 |

### 8.6 阶段结论

- 同口径 A-current 与 B3-0.05 / B3-0.10 数值完全相同，说明当前 auxiliary angular 在 0.05/0.10 权重下没有产生可观察影响。可能原因是 angular 辅助项相对 image-plane 项尺度过弱，或者虽然 residual block 被构造出来了，但其数值尺度不足以改变优化结果。
- B5a 明显伤害 outer：outer optimized RMSE 从约 0.20-0.23 量级升到 0.856，说明“outer 全部 angular”不适合作为方向。
- B5b 的 holdout overall 比 A-current 只低约 0.00041 px，差异极小；同时 internal angular RMSE 更高，说明 internal angular 没有形成可靠收益。
- B7 连续 polar hybrid 能正常构造连续权重，但 holdout 没有超过 B3/B5b，70-85 deg bin 也比 B3/B5b 更差。
- 同口径结论：A-current 仍然是最稳默认方案；B3 无可见影响，B5a 明显不合适，B5b 只有极微小 pixel holdout 变化且 angular 指标变差，B7 没有超过 A。

### 8.7 下一步建议

1. 不建议继续小范围调 B3 auxiliary weight；当前更像尺度问题，不是 0.05/0.10 这种权重能解决。
2. 若继续研究 angular，优先实现 B4 normalized angular residual：按 local pixel-to-ray sensitivity / projection Jacobian 归一化 angular residual，解决 radian residual 和 pixel residual 尺度不一致问题。
3. B5a、B7 暂时不作为主线；B5b 可作为失败/弱收益消融保留。
4. 默认 baseline 仍保持 A-current / image-plane。

## 9. 第三轮 angular 分支：B4 normalized angular residual

B4 的目标是解决 B full angular 的核心问题：原始 angular residual 的单位是 rad，而 image-plane residual 的单位是 pixel。直接替换时，优化器看到的是一个尺度不同的目标，容易导致 full angular 不稳定或把内参拉到不合适的位置。

B4 的做法是：对每个观测点，在当前 DS 相机模型下估计该点附近 **1 pixel 图像扰动对应多少 ray / tangent angular 扰动**，得到 `angular_sigma_per_pixel_rad`。随后将 angular residual 按这个局部尺度进行归一化，使其更接近“像素等价”的 residual。

### 9.1 实验组说明

| 组别 | 方法 | 目的 |
|---|---|---|
| B4-full | full normalized sphere angular | 所有点都使用 normalized angular，验证 full angular 是否能从灾难性失败恢复 |
| B4-outer | outer normalized angular + internal image-plane | 判断 outer 使用 normalized angular 是否更合理 |
| B4-internal | outer image-plane + internal normalized angular | 判断 internal 使用 normalized angular 是否更合理 |

### 9.2 Backend residual 构造数量

| 组别 | backend points | image residual | normalized angular residual | outer image / angular | internal image / angular |
|---|---:|---:|---:|---:|---:|
| A-current | 3596 | 3596 | 0 | 444 / 0 | 3152 / 0 |
| B4-full | 3596 | 0 | 3596 | 0 / 444 | 0 / 3152 |
| B4-outer | 3596 | 3152 | 444 | 0 / 444 | 3152 / 0 |
| B4-internal | 3596 | 444 | 3152 | 444 / 0 | 0 / 3152 |

### 9.3 Training / Holdout pixel RMSE

| 组别 | training overall | training outer | training internal | holdout overall | holdout outer | holdout internal | Kalibr holdout | ours - Kalibr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-current | 1.08004 | 0.202844 | 1.14930 | 3.25668 | 0.814346 | 3.46075 | 3.28411 | -0.02743 |
| B4-full | 1.07965 | 0.201698 | 1.14891 | 3.25634 | 0.815855 | 3.46034 | 3.28411 | -0.02777 |
| B4-outer | 1.08004 | 0.203740 | 1.14928 | 3.25707 | 0.816302 | 3.46111 | 3.28411 | -0.02704 |
| B4-internal | 1.07974 | 0.200523 | 1.14904 | 3.25615 | 0.813689 | 3.46021 | 3.28411 | -0.02796 |

### 9.4 Backend angular diagnostics

| 组别 | optimized angular RMSE | outer angular RMSE | internal angular RMSE |
|---|---:|---:|---:|
| A-current | 0.000599840 | 0.000267851 | 0.000632760 |
| B4-full | 0.000599195 | 0.000268720 | 0.000632010 |
| B4-outer | 0.000591570 | 0.000304530 | 0.000621440 |
| B4-internal | 0.000609760 | 0.000245820 | 0.000644724 |

### 9.5 Polar-bin angular RMSE

| 组别 | 0-30 deg | 30-50 deg | 50-70 deg | 70-85 deg |
|---|---:|---:|---:|---:|
| A-current | 0.000725 | 0.000504 | 0.000435 | 0.000327 |
| B4-full | 0.000724 | 0.000504 | 0.000432 | 0.000291 |
| B4-outer | 0.000715 | 0.000497 | 0.000430 | 0.000333 |
| B4-internal | 0.000737 | 0.000513 | 0.000436 | 0.000300 |

### 9.6 Normalized scale 诊断

| 组别 | sigma mean rad/px | sigma min | sigma max | mean normalized weight | max normalized weight |
|---|---:|---:|---:|---:|---:|
| B4-full | 0.000674059 | 0.000652343 | 0.000690607 | 1000000 | 1000000 |
| B4-outer | 0.000083147 | 0 | 0.000690591 | 123470.523 | 1000000 |
| B4-internal | 0.000590912 | 0 | 0.000690607 | 876529.477 | 1000000 |

说明：当前 `normalized_angular_max_weight_scale` 设置为 `1000000`，B4-full 的所有点都触顶，说明归一化尺度已经显著放大 angular residual。这个结果能解释为什么 B4 不再像 B full angular 一样崩坏，但也提示后续如果继续研究，需要做 weight cap / reference sigma 的敏感性实验。

### 9.7 阶段结论

- B4-full 相比 B full angular 是明显进步：B full angular 的 holdout overall 为 `4.04465`，而 B4-full 回到 `3.25634`，说明 normalized angular 成功修复了 naive full angular 的尺度灾难问题。
- B4-full 与 A-current 的差距非常小：holdout overall 只改善约 `0.00034 px`，不能作为显著提升。
- B4-internal 的 holdout overall 最低，为 `3.25615`，但 angular RMSE 反而比 A-current 更差，因此不能说几何意义上更优。
- B4-outer 的整体 angular RMSE 最低，但 holdout overall 变差，说明只优化 outer angular 会改变 ray-space 指标，但不一定改善最终 pixel holdout。
- 当前结论：B4 是一个比 B full angular 更合理的 angular residual 设计，证明“尺度归一化是必要的”；但它目前仍不足以替代 A-current image-plane baseline。

### 9.8 下一步建议

1. B4 应作为 angular residual 的重要消融保留：它说明 normalized angular 可以避免 full angular 崩坏。
2. 暂时不要把 B4 作为 New Baseline。
3. 如果继续探索 angular，下一步更合理的是 `image-plane + normalized angular auxiliary`，而不是 full replacement。
4. 建议补一个 weight-cap / reference-sigma sweep，例如：
   - max weight scale: `1e4 / 1e5 / 1e6`
   - reference sigma px: `0.5 / 1.0 / 2.0`
5. 默认主线仍保持 A-current image-plane。

## 10. normalized angular auxiliary 试验：B4-aux-norm005

这一组是在 `A-current` 的基础上，只额外加入 **normalized angular auxiliary**，主 residual 仍然是 image-plane。目标是验证：如果不做 full replacement，只把 normalized angular 作为辅助约束，是否能比 A-current 带来稳定增益。

### 10.1 实验组说明

| 组别 | 方法 | 目的 |
|---|---|---|
| B4-aux-norm005 | image-plane 主 residual + normalized angular auxiliary, weight=0.05 | 验证 normalized angular 作为辅助项是否能稳定改善 |

### 10.2 Backend residual 构造数量

| 组别 | backend points | image residual | angular residual | angular auxiliary | outer image / angular aux | internal image / angular aux |
|---|---:|---:|---:|---:|---:|---:|
| A-current | 3596 | 3596 | 0 | 0 | 444 / 0 | 3152 / 0 |
| B4-aux-norm005 | 3596 | 3596 | 3596 | 3596 | 444 / 444 | 3152 / 3152 |

### 10.3 Training / Holdout pixel RMSE

| 组别 | training overall | training outer | training internal | holdout overall | holdout outer | holdout internal | Kalibr holdout | ours - Kalibr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-current | 1.08004 | 0.202844 | 1.14930 | 3.25668 | 0.814346 | 3.46075 | 3.28411 | -0.02743 |
| B4-aux-norm005 | 1.07996 | 0.202422 | 1.14922 | 3.25662 | 0.814327 | 3.46069 | 3.28411 | -0.02749 |

### 10.4 Backend angular diagnostics

| 组别 | optimized angular RMSE | outer angular RMSE | internal angular RMSE |
|---|---:|---:|---:|
| A-current | 0.000599840 | 0.000267851 | 0.000632760 |
| B4-aux-norm005 | 0.000599819 | 0.000267701 | 0.000632746 |

### 10.5 阶段结论

- B4-aux-norm005 已经确认能把 normalized angular auxiliary 接到 backend 中。
- 但在 weight=0.05 时，它和 A-current 几乎完全一致，只带来极其微小的数值波动。
- 这说明 normalized auxiliary 的方向是可跑通的，但当前权重还不足以形成可见的优化收益。
- 后续如果继续，应优先做辅助权重和 weight cap 的 sweep，而不是直接把它当成新 baseline。

## 11. normalized angular auxiliary 权重 sweep

这一轮继续沿用 `image-plane 主 residual + normalized angular auxiliary`，只改变 auxiliary weight。目标是判断：B4 auxiliary 是不是因为权重太小所以没有产生效果。

注意：当前 `stage5_ang_B4_aux_norm005...` 目录中的 `backend_optimization_summary.txt` 实际记录为 `backend_angular_auxiliary_weight: 1`，说明该目录不再是严格的 0.05 输出，可能被后续运行覆盖。因此本节主要使用 0.1 / 0.2 / 0.5 / 1.0 四组有效输出，并保留 A-current 作为对照。

### 11.1 Training / Holdout pixel RMSE

| 组别 | auxiliary weight | training overall | training outer | training internal | holdout overall | holdout outer | holdout internal | Kalibr holdout | ours - Kalibr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A-current | 0 | 1.08004 | 0.202844 | 1.14930 | 3.25668 | 0.814346 | 3.46075 | 3.28411 | -0.02743 |
| B4-aux-norm010 | 0.1 | 1.07989 | 0.202048 | 1.14916 | 3.25657 | 0.814312 | 3.46064 | 3.28411 | -0.02754 |
| B4-aux-norm020 | 0.2 | 1.07977 | 0.201423 | 1.14904 | 3.25648 | 0.814288 | 3.46054 | 3.28411 | -0.02763 |
| B4-aux-norm050 | 0.5 | 1.07953 | 0.200154 | 1.14882 | 3.25630 | 0.814252 | 3.46035 | 3.28411 | -0.02781 |
| B4-aux-norm100 | 1.0 | 1.07933 | 0.199033 | 1.14863 | 3.25612 | 0.814240 | 3.46016 | 3.28411 | -0.02799 |

### 11.2 Backend angular diagnostics

| 组别 | auxiliary weight | optimized angular RMSE | outer angular RMSE | internal angular RMSE |
|---|---:|---:|---:|---:|
| A-current | 0 | 0.000599840 | 0.000267851 | 0.000632760 |
| B4-aux-norm010 | 0.1 | 0.000599801 | 0.000267575 | 0.000632734 |
| B4-aux-norm020 | 0.2 | 0.000599771 | 0.000267377 | 0.000632714 |
| B4-aux-norm050 | 0.5 | 0.000599713 | 0.000267031 | 0.000632672 |
| B4-aux-norm100 | 1.0 | 0.000599664 | 0.000266801 | 0.000632633 |

### 11.3 Polar-bin angular RMSE

| 组别 | auxiliary weight | 0-30 deg | 30-50 deg | 50-70 deg | 70-85 deg |
|---|---:|---:|---:|---:|---:|
| A-current | 0 | 0.000725 | 0.000504 | 0.000435 | 0.000327 |
| B4-aux-norm010 | 0.1 | 0.000725 | 0.000504 | 0.000435 | 0.000319 |
| B4-aux-norm020 | 0.2 | 0.000725 | 0.000504 | 0.000433 | 0.000313 |
| B4-aux-norm050 | 0.5 | 0.000725 | 0.000504 | 0.000433 | 0.000302 |
| B4-aux-norm100 | 1.0 | 0.000725 | 0.000504 | 0.000433 | 0.000292 |

### 11.4 阶段结论

- 随着 auxiliary weight 从 0.1 增加到 1.0，holdout overall 从 `3.25657` 缓慢降到 `3.25612`，趋势是单调改善，但幅度非常小。
- angular RMSE 也从 `0.000599801` 缓慢降到 `0.000599664`，说明 normalized auxiliary 确实在起作用，不再是完全无效项。
- 改善主要集中在 70-85 deg 边缘 bin：从 A-current 的 `0.000327` 降到 weight=1.0 的 `0.000292`。这符合“angular residual 更关注边缘大视场角”的预期。
- 但整体提升仍然很小：weight=1.0 相比 A-current 的 holdout overall 只改善约 `0.00056 px`，暂时不能作为强结论。
- 当前最合理表述：normalized angular auxiliary 有稳定、单调、弱改善趋势，尤其在 70-85 deg 边缘 bin；但收益量级太小，需要多数据集验证，且不能替代 A-current baseline。

### 11.5 下一步建议

1. 不建议继续只在这个 split 上调更大权重。
2. 建议把 weight=1.0 的 normalized auxiliary 跑到另外两到三组数据集，验证是否稳定单调改善。
3. 如果多数据集仍然只有极小收益，则 angular auxiliary 更适合作为论文中的消融探索，而不是主线方法。
4. 如果某些高边缘覆盖数据集收益更明显，再考虑做 point-wise adaptive angular auxiliary。

## 12. B4-aux-norm100 跨数据集验证

这一节验证 `image-plane 主 residual + normalized angular auxiliary, weight=1.0` 是否能跨数据集稳定改善。该方法在上一节的 `144928 -> 134853` split 上有单调但很弱的收益，因此这里继续在另外两组 20260430 数据上测试。

### 12.1 实验目的

| 组别 | 方法 | 想验证的问题 |
|---|---|---|
| Current / New Baseline | 默认 image-plane backend | 当前冻结的新 baseline |
| B4-aux-norm100 | image-plane + normalized angular auxiliary, weight=1.0 | angular auxiliary 的弱改善是否能跨数据集稳定复现 |

### 12.2 Backend residual 构造确认

| Split | 方法 | image residual count | angular auxiliary count | outer image / aux | internal image / aux |
|---|---|---:|---:|---:|---:|
| `134853 -> 144419` | B4-aux-norm100 | 3066 | 3066 | 380 / 380 | 2686 / 2686 |
| `144419 -> 144928` | B4-aux-norm100 | 2425 | 2425 | 300 / 300 | 2125 / 2125 |

说明：两组都确认 normalized angular auxiliary 已经真实进入 backend，不是只写 diagnostic。

### 12.3 Training / Holdout 对比

| Split | 方法 | training overall | training outer | training internal | holdout overall | holdout outer | holdout internal | Kalibr holdout | ours - Kalibr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `144928 -> 134853` | A-current | 1.08004 | 0.202844 | 1.14930 | 3.25668 | 0.814346 | 3.46075 | 3.28411 | -0.02743 |
| `144928 -> 134853` | B4-aux-norm100 | 1.07933 | 0.199033 | 1.14863 | 3.25612 | 0.814240 | 3.46016 | 3.28411 | -0.02799 |
| `134853 -> 144419` | Current | 0.807135 | 0.094511 | 0.860405 | 1.94433 | 0.331100 | 2.07148 | 1.95731 | -0.01298 |
| `134853 -> 144419` | B4-aux-norm100 | 0.807640 | 0.093222 | 0.860964 | 1.94402 | 0.339552 | 2.07098 | 1.95742 | -0.01340 |
| `144419 -> 144928` | Current | 0.690635 | 0.102534 | 0.735564 | 6.57076 | 15.2445 | 4.09776 | 6.58414 | -0.01338 |
| `144419 -> 144928` | B4-aux-norm100 | 0.709258 | 0.111626 | 0.755456 | 6.60131 | 15.3290 | 4.10851 | 6.58635 | 0.01495 |

### 12.4 Angular diagnostics

| Split | 方法 | optimized angular RMSE | outer angular RMSE | internal angular RMSE |
|---|---|---:|---:|---:|
| `144928 -> 134853` | A-current | 0.000599840 | 0.000267851 | 0.000632760 |
| `144928 -> 134853` | B4-aux-norm100 | 0.000599664 | 0.000266801 | 0.000632633 |
| `134853 -> 144419` | B4-aux-norm100 | 0.000506710 | 0.000144743 | 0.000538623 |
| `144419 -> 144928` | B4-aux-norm100 | 0.000426641 | 0.000155422 | 0.000452006 |

注：`134853 -> 144419` 和 `144419 -> 144928` 的 Current 对照当时没有开启 angular diagnostics，因此这里只记录 B4-aux-norm100 的 angular RMSE。若后续需要严格比较 angular RMSE，应重跑对应 Current 并开启 `--stage5-enable-angular-residual-diagnostics`。

### 12.5 阶段结论

- `B4-aux-norm100` 在 `144928 -> 134853` 上有非常小的改善：holdout overall 从 `3.25668` 到 `3.25612`。
- `B4-aux-norm100` 在 `134853 -> 144419` 上也只有极小改善：holdout overall 从 `1.94433` 到 `1.94402`，但 outer holdout 从 `0.33110` 变差到 `0.33955`。
- `B4-aux-norm100` 在 `144419 -> 144928` 上明显变差：holdout overall 从 `6.57076` 到 `6.60131`，outer/internal 都变差。
- 因此，normalized angular auxiliary 虽然工程上已经跑通，也能在部分 split 上带来弱改善，但跨数据集不稳定。
- 当前不建议把 B4-aux-norm100 纳入 New Baseline。
- 更合理的定位是：`B4 normalized angular auxiliary` 是一个可保留的 angular residual 消融分支，用来说明“normalized angular residual 比 naive full angular 更稳定”，但它还不是稳定提升方法。

### 12.6 后续建议

1. New Baseline 继续保持默认 image-plane backend。
2. 暂时不要继续简单增大全局 auxiliary weight。
3. 如果继续研究 angular residual，更值得尝试的是 point-wise / polar-aware auxiliary：
   - 只对高 polar angle 点启用 angular auxiliary；
   - 或根据 polar angle / corner quality / local residual 给 angular auxiliary 自适应权重；
   - 避免对中心区域和已稳定点施加不必要的 angular 约束。
4. 如果要严格比较 angular diagnostics，需要给 Current 对照也开启 angular diagnostic，避免只看到 B4 分支的 angular RMSE。
