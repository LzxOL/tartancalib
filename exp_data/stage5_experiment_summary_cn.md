# Stage5 Backend 实验总结

## 1. 历史 strict baseline / Old Baseline 基础结果

本节保留早期 strict selection 口径下的基础结果。根据后续 trial backend frame-board incremental selection 实验，正式命名已经更新为：

- **Old Baseline**：本节记录的早期 strict failed-board drop 口径，用作历史对照。
- **New Baseline**：后续冻结的 `current` 配置，即 Kalibr-style incremental frame-board trial selection，作为后续 Stage5 默认主线。
- **Consistency-Soft Enhancement**：原 `v3-soft`，作为 New Baseline 的可选增强分支。

Old Baseline 当时的方案是：

`auto-init + round2 + delayed intrinsics release + residual sanity gate + strict failed-board drop + Stage5 evaluator + ASLAM backend`

也就是说，Old Baseline 的核心定义是：

- 相机初始化：自动初始化，不依赖手写 Double Sphere 内参。
- 前端流程：保留 round2 / second-pass internal regeneration。
- 内参释放：前端与 backend 都使用 delayed intrinsics release。
- 观测接受策略：采用 Kalibr-style strict failed-board drop。
- board pose-fit gate：降级为 debug 选项，不属于论文主线。
- blur filter / pre-backend point filter：保留为实验和诊断工具，不进入 baseline。

strict failed-board drop 的含义是：如果某一帧中的某个 board 在 internal pose estimation / internal regeneration 阶段失败，则这个 board observation 整块丢弃；同一帧里其它正常 board 继续使用。这个策略更接近 Kalibr 的 target observation 接受逻辑，避免把“内部点生成失败但外部点还存在”的 board 继续混入后续评估或优化。

## 2. full 数据集主结果

本节分成三组结果：

- 原始四个 full 单目数据集：用于说明当前 Stage5 strict baseline 在主实验集上的整体稳定性。
- `dataset_5_1` 三个 stereo full 数据集的 right 相机跨数据集外部验证：用于检查同一套单目标定 baseline 是否能从一个 full 序列泛化到另一个 full 序列。
- 原始四个 full 单目数据集的成对 cross-dataset external validation：用于进一步检查原始主实验集内部的跨序列泛化。

所有表格都使用同一 Stage5 evaluator 口径。Kalibr reference 指的是外部棋盘格 camchain 在同一批 outer/internal observations 上的重投影评估结果，而不是 Kalibr 直接在多板 AprilTag 数据上重新标定的结果。

### 2.1 原始四个 full 单目数据集

下面这张表来自 `stage5_strict_baseline_4datasets.csv`，是当前最重要的 full 级别实验汇总。标准差列分别给出 Backend 与 Kalibr reference 的 x/y residual 标准差。

| 数据集 | 训练/验证帧 | 选中帧/板/内部点 | Backend 优化后 RMSE | 训练集 Backend / Kalibr | 验证集 Backend / Kalibr | 验证集 Backend x/y 标准差 | 验证集 Kalibr x/y 标准差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260421_140151 | 292 / 73 | 17 / 71 / 2057 | 3.444 | 4.0542 / 4.1163 | 6.6279 / 6.6896 | 5.7478 / 3.2946 | 5.8471 / 3.2321 |
| 20260421_141444 | 229 / 58 | 25 / 106 / 3050 | 4.6141 | 7.6197 / 9.0040 | 7.0268 / 7.7321 | 5.8610 / 3.8160 | 6.5351 / 4.0397 |
| 20260427_191538 | 91 / 23 | 13 / 65 / 1892 | 1.0451 | 1.4049 / 1.4470 | 2.9399 / 2.9392 | 2.0178 / 2.1206 | 2.0410 / 2.1037 |
| 20260427_192347 | 100 / 26 | 15 / 70 / 2027 | 3.1677 | 4.8985 / 5.3114 | 4.2005 / 4.6305 | 3.5487 / 2.1736 | 3.9052 / 2.3816 |

这四组数据的结论是：

- 在四个 full 数据集上，当前 strict baseline 都能稳定运行。
- 在 `20260421_140151` 和 `20260421_141444` 上，Backend 在 Stage5 evaluator 同口径下优于外部 Kalibr reference。
- 在 `20260427_191538` 上，Backend 和 Kalibr reference 几乎打平，而且误差已经接近棋盘格标定常见的 2 px 量级。
- 在 `20260427_192347` 上，Backend 仍优于 Kalibr reference，但 x 方向标准差明显偏大，主要来自运动模糊/重影图像。

一个重要 bookkeeping 备注：`20260427_192347` 的结果目前存放在 `result/stage5_backend_full_140151_strict_baseline_check`，这个目录名有误导性，因为当时复用了旧 output path。后续引用时应以 `stage5_strict_baseline_4datasets.csv` 中记录的真实图像路径为准。

### 2.2 `dataset_5_1` 三个 stereo full 数据集的 right 相机跨数据集验证

下面补充 `dataset_5_1` 中三个 full stereo 数据集右目单目标定结果。与前面的内部 stride holdout 不同，这里采用更严格的 cross-dataset external validation：每次使用一个 full 数据集训练，并使用另一个 full 数据集整体作为验证集。

| 训练数据集 -> 外部验证数据集 | 训练/验证帧 | 选中帧/板/内部点 | Backend 优化后 RMSE | 训练集 Backend / Kalibr | 外部验证 Backend / Kalibr | 外部验证 Backend x/y 标准差 | 外部验证 Kalibr x/y 标准差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260430_134853 right -> 20260430_144419 right | 85 / 87 | 16 / 78 / 2575 | 0.8856 | 1.1330 / 1.1980 | 2.1105 / 2.1411 | 1.7992 / 1.0851 | 1.8027 / 1.1149 |
| 20260430_144419 right -> 20260430_144928 right | 87 / 134 | 11 / 53 / 1749 | 0.6522 | 0.6771 / 0.7313 | 6.8392 / 8.5933 | 5.9253 / 3.3960 | 7.7359 / 3.7336 |
| 20260430_144928 right -> 20260430_134853 right | 134 / 85 | 19 / 85 / 2820 | 1.2116 | 1.8448 / 2.1157 | 3.4419 / 3.4916 | 3.0706 / 1.5142 | 3.0901 / 1.5484 |

新增 pose-only refit 诊断指标如下。这三组均使用最新 baseline 诊断输出重跑，`pose-only refit RMSE` 与 `outer-fit RMSE` 来自外部验证集上的 outer-only board pose refit；`internal evaluation RMSE` 是在该 refit pose 下评估 internal points。

| 训练数据集 -> 外部验证数据集 | pose-only success rate Backend / Kalibr | outer-fit RMSE Backend / Kalibr | outer-fit Δ(B-K) | internal RMSE Backend / Kalibr | internal Δ(B-K) | 诊断 |
|---|---:|---:|---:|---:|---:|---|
| 20260430_134853 right -> 20260430_144419 right | 100% / 100% | 0.7641 / 0.8926 | -0.1285 | 2.2345 / 2.2609 | -0.0264 | 正常，Backend 小幅更优 |
| 20260430_144419 right -> 20260430_144928 right | 100% / 100% | 15.2107 / 21.4454 | -6.2347 | 4.5828 / 4.4658 | +0.1170 | outer-fit 异常，验证序列 pose refit 很不稳定 |
| 20260430_144928 right -> 20260430_134853 right | 100% / 100% | 1.1472 / 1.1723 | -0.0251 | 3.6464 / 3.6988 | -0.0524 | 正常，Backend 小幅更优 |

这三组 cross-dataset external validation 的结论是：

- 训练集上三组 Backend 都优于 Kalibr reference，说明当前 backend refinement 对训练序列本身是稳定有效的。
- 外部验证上，三组 Backend 的 overall RMSE 都优于 Kalibr reference，但优势仍主要是小幅或局部的，不是数量级差异。
- `144419 -> 144928` 是明显异常组：pose-only / outer-fit RMSE 达到 `15.2107 px`，Kalibr reference 也达到 `21.4454 px`，说明该外部验证序列在 outer-only pose refit 层面已经很不稳定；这不是 internal 点单独造成的误差。
- `134853 -> 144419` 和 `144928 -> 134853` 两组更健康，pose-only refit success rate 均为 100%，outer-fit RMSE 较低，主要误差仍来自 internal evaluation RMSE 高于 outer-fit RMSE。
- 从 x/y 标准差看，三组外部验证的 Backend 与 Kalibr 基本处在同一量级；`144419 -> 144928` 的 x 方向标准差明显偏大，需要结合 worst-case frame / board diagnostics 单独解释。
- 因此，这组三组实验更适合作为“跨数据集外部验证 sanity check”：它证明方法没有明显过拟合到单一序列，但目前还不足以支撑“显著优于 Kalibr”的强结论。

### 2.3 原始 full 数据集的成对 cross-dataset external validation

下面补充原始四个 full 单目数据集的成对外部验证结果。这里分成两对相近采集条件的数据集互做验证：

- `20260421_140151 <-> 20260421_141444`
- `20260427_191538 <-> 20260427_192347`

| 训练数据集 -> 外部验证数据集 | 训练/验证帧 | 选中帧/板/内部点 | Backend 优化后 RMSE | 训练集 Backend / Kalibr | 外部验证 Backend / Kalibr | 外部验证 Backend x/y 标准差 | 外部验证 Kalibr x/y 标准差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260421_140151 -> 20260421_141444 | 365 / 287 | 17 / 73 / 2403 | 3.4057 | 4.0211 / 4.0841 | 7.6070 / 7.8450 | 6.2206 / 4.3442 | 6.4632 / 4.3973 |
| 20260421_141444 -> 20260421_140151 | 287 / 365 | 27 / 116 / 3815 | 4.5009 | 7.1138 / 8.5567 | 6.5873 / 6.7215 | 5.3441 / 3.8353 | 5.4591 / 3.8826 |
| 20260427_191538 -> 20260427_192347 | 114 / 126 | 13 / 65 / 2152 | 1.0201 | 1.3916 / 1.4326 | 4.6775 / 4.8072 | 3.9171 / 2.5018 | 3.9968 / 2.5748 |
| 20260427_192347 -> 20260427_191538 | 126 / 114 | 18 / 82 / 2701 | 3.1712 | 5.2167 / 5.7057 | 3.0087 / 3.0878 | 2.2244 / 2.0066 | 2.3058 / 2.0449 |

新增 pose-only refit 诊断指标如下。这里的 `pose-only refit RMSE` 与 `outer-fit RMSE` 均来自外部验证集上每个 board observation 的 outer-only pose refit；`internal evaluation RMSE` 则是在该 refit pose 下对 internal points 的重投影评估。

| 训练数据集 -> 外部验证数据集 | pose-only success rate Backend / Kalibr | outer-fit RMSE Backend / Kalibr | outer-fit Δ(B-K) | internal RMSE Backend / Kalibr | internal Δ(B-K) | 诊断 |
|---|---:|---:|---:|---:|---:|---|
| 20260421_140151 -> 20260421_141444 | 100% / 100% | 21.0196 / 30.2280 | -9.2084 | 7.9477 / 8.2084 | -0.2607 | outer-fit 异常，Backend 明显好于 Kalibr 但绝对误差仍高 |
| 20260421_141444 -> 20260421_140151 | 100% / 100% | 2.8544 / 2.5828 | +0.2716 | 6.9507 / 7.1096 | -0.1589 | outer-fit Kalibr 略优，internal Backend 略优 |
| 20260427_191538 -> 20260427_192347 | 100% / 100% | 2.5329 / 2.6001 | -0.0672 | 4.8989 / 5.0350 | -0.1361 | Backend 小幅更优 |
| 20260427_192347 -> 20260427_191538 | 100% / 100% | 1.1671 / 1.2095 | -0.0424 | 3.1796 / 3.2626 | -0.0830 | Backend 小幅更优 |

这四组成对 external validation 的结论是：

- 四组外部验证中，Backend 在 external validation overall RMSE 上都小幅优于 Kalibr reference。
- `20260427_191538 <-> 20260427_192347` 两组更健康，绝对误差更低，说明较新的 04-27 数据更适合作为方法效果展示。
- 对已补充 refit 诊断的 `20260427` 两组，pose-only refit success rate 均为 100%，说明外部验证误差不是由 refit 失败造成；同时 internal evaluation RMSE 明显高于 outer-fit RMSE，说明当前主要瓶颈更可能在 internal point 观测模型 / 检测系统偏移，而不是 outer pose refit 稳定性。
- `20260421_140151 <-> 20260421_141444` 两组绝对误差较高，但 Backend 仍保持小幅优势；结合 worst-case diagnostics，这两组误差主要受少数高残差 frame / board observation 主导。
- 这四组结果比 `dataset_5_1` 的三组 stereo-right cross validation 更整齐，支持“当前 Stage5 baseline 具备一定跨数据集泛化能力”的判断。
- 但优势仍然主要是小幅领先，而不是显著数量级提升；因此论文叙事不应强调“精度显著碾压 Kalibr”，而应强调鲁棒多板标定流程、失败观测处理和跨序列一致性。

### 2.4 KB 模型 cross-dataset external validation

完整 KB 模型实验表格已单独整理到：

- `exp_data/stage5_kb_model_experiments_cn.md`

为了判断“只是换 frontend 检测器/相机模型”到底能带来多大变化，我们又补跑了同一套 cross-dataset external validation，但将前端配置切到 `pinhole-equi`，作为当前仓库里的 KB 风格模型实验版。对应输出目录是：

- `result/stage5_backend_full_20260421_140151_external_val_141444_kb_baseline_refit_diag`
- `result/stage5_backend_full_20260421_141444_external_val_140151_kb_baseline_refit_diag`
- `result/stage5_backend_full_20260427_191538_external_val_192347_kb_baseline_refit_diag`
- `result/stage5_backend_full_20260427_192347_external_val_191538_kb_baseline_refit_diag`

这里仍然使用与前文一致的 external validation 口径，但需要特别说明两点：

- 这不是“独立的 Kalibr 重跑”，而是我们自己的 Stage5 backend 管线在 `pinhole-equi` 配置下的结果。
- 表中的 `Kalibr` 仍然指棋盘格 camchain reference 在同一 evaluator 下的评估结果。

| 训练数据集 -> 外部验证数据集 | Backend 优化后 RMSE | 训练集 Backend RMSE | 外部验证 Backend / Kalibr | 外部验证 Backend x/y 标准差 | 外部验证 Kalibr x/y 标准差 |
|---|---:|---:|---:|---:|---:|
| 20260421_140151 -> 20260421_141444 | 5.9302 | 6.6919 | 17.4588 / 6.7567 | 11.3249 / 13.2734 | 5.5303 / 3.8531 |
| 20260421_141444 -> 20260421_140151 | 4.7335 | 6.6682 | 6.6005 / 6.6826 | 5.3696 / 3.8244 | 5.4339 / 3.8707 |
| 20260427_191538 -> 20260427_192347 | 0.8940 | 0.9142 | 3.9603 / 3.9764 | 3.3737 / 2.0467 | 3.3703 / 2.0654 |
| 20260427_192347 -> 20260427_191538 | 2.0310 | 3.9286 | 2.3703 / 2.3447 | 1.8504 / 1.4580 | 1.8490 / 1.4375 |

对应的 pose-only refit / internal 诊断如下：

| 训练数据集 -> 外部验证数据集 | pose-only success rate Backend / Kalibr | outer-fit RMSE Backend / Kalibr | internal evaluation RMSE Backend / Kalibr | 结论 |
|---|---:|---:|---:|---|
| 20260421_140151 -> 20260421_141444 | 100% / 100% | 15.7971 / 3.7211 | 17.6758 / 7.0739 | 明显退化，KB 配置在这组上失稳 |
| 20260421_141444 -> 20260421_140151 | 100% / 100% | 2.4325 / 2.4332 | 6.9843 / 7.0726 | 基本打平，Backend 略优 |
| 20260427_191538 -> 20260427_192347 | 100% / 100% | 1.7914 / 1.8552 | 4.1698 / 4.1833 | 小幅优于 Kalibr，和 DS baseline 同量级 |
| 20260427_192347 -> 20260427_191538 | 100% / 100% | 1.0598 / 0.9953 | 2.4962 / 2.4722 | Kalibr 略优，差距很小 |

随后又补跑了 `dataset_5_1` 中三组 `right` 相机的 KB cross-dataset external validation：

- `result/stage5_backend_full_20260430_144928_right_external_val_134853_right_kb_baseline_refit_diag`
- `result/stage5_backend_full_20260430_134853_right_external_val_144928_right_kb_baseline_refit_diag`
- `result_may/stage5_backend_full_20260430_144419_right_external_val_134853_right_kb_baseline_refit_diag`

注意这里仍然是**跨模型参考对比**：

- 我们的方法使用 `pinhole-equi / KB-like` 配置；
- Kalibr reference 使用 `DS` camchain。

因此，这两组结果只能说明 KB 分支的**可用性与量级位置**，不能作为“同模型条件下优于 Kalibr”的直接证据。

| 训练数据集 -> 外部验证数据集 | Backend 优化后 RMSE | 训练集 Backend / Kalibr | 外部验证 Backend / Kalibr | 外部验证 Backend x/y 标准差 | 外部验证 Kalibr x/y 标准差 |
|---|---:|---:|---:|---:|---:|
| 20260430_144928 right -> 20260430_134853 right | 2.1113 | 4.1622 / 4.5440 | 3.3978 / 3.4191 | 3.1024 / 1.3337 | 3.1136 / 1.3452 |
| 20260430_134853 right -> 20260430_144928 right | 0.8160 | 0.8338 / 0.8740 | 3.7577 / 3.6698 | 2.9208 / 2.3498 | 2.8516 / 2.3049 |
| 20260430_144419 right -> 20260430_134853 right | 0.6509 | 0.6759 / 0.7464 | 3.4194 / 3.4310 | 3.1256 / 1.3606 | 3.1221 / 1.3606 |

| 训练数据集 -> 外部验证数据集 | pose-only success rate Backend / Kalibr | outer-fit RMSE Backend / Kalibr | internal evaluation RMSE Backend / Kalibr | 结论 |
|---|---:|---:|---:|---|
| 20260430_144928 right -> 20260430_134853 right | 100% / 100% | 1.1271 / 1.1540 | 3.5972 / 3.6188 | KB 小幅优于 Kalibr DS reference |
| 20260430_134853 right -> 20260430_144928 right | 100% / 100% | 2.3267 / 2.3137 | 3.9135 / 3.8185 | KB 略差于 Kalibr DS reference |
| 20260430_144419 right -> 20260430_134853 right | 100% / 100% | 1.1052 / 1.1522 | 3.6221 / 3.6325 | KB 极小幅优于 Kalibr DS reference |

这一轮 KB 结果的核心结论是：

- KB 配置并没有带来稳定、普遍的提升。
- `20260421_140151 -> 20260421_141444` 这一组出现了明显退化，external validation overall RMSE 从当前 DS baseline 的 `7.6070` 恶化到 `17.4588`，说明它在这组数据上的泛化稳定性不够。
- `20260421_141444 -> 20260421_140151` 和两组 `20260427` 数据上，KB 配置与 Kalibr reference 基本是打平或仅有极小波动，不存在显著优势。
- `dataset_5_1 right` 这三组也延续了同样趋势：`144928 -> 134853` 和 `144419 -> 134853` 都是小幅正结果，而 `134853 -> 144928` 略差，没有形成稳定一致的提升。
- 由于这些 `dataset_5_1` 结果本质上是 `KB-like` 对 `Kalibr DS reference` 的跨模型比较，它们更适合作为“KB 分支已经可跑通且误差量级合理”的补充证据，而不是论文主结论。
- 因而从论文角度看，这组结果再次印证了一个更重要的判断：当前限制因素不主要是“换一个相机模型”或“换一个 frontend 检测器”，而更可能是 internal observation 的误差建模、模糊场景下的系统偏差，以及 backend 如何处理观测不确定性。
- 也正因为如此，KB 这条线目前更适合放在“补充实验 / negative result”里，而不适合作为新的主 baseline。

### 2.5 EUCM 模型 cross-dataset external validation

完整 EUCM 模型实验表格已单独整理到：

- `exp_data/stage5_eucm_model_experiments_cn.md`

这一轮 EUCM 结果的核心判断是：

- `EUCM-none` 已经具备完整、稳定的 full 数据集运行能力。
- 在大多数 cross-dataset external validation 上，EUCM 与 Kalibr reference 持平或小幅更优。
- 与 KB 分支相比，EUCM 的稳定性和跨数据集一致性明显更好。
- 但它仍然没有形成“显著优于当前 DS baseline”的压倒性优势，因此目前更适合作为强候选模型族，而不是直接替换 DS 的唯一主线。

## 3. 与 Kalibr reference 的解释口径

这里的 Kalibr reference 不是“Kalibr 直接标定当前多板大 Tag 场景”的结果，而是使用棋盘格标定得到的 `mono_fisheye_calib_3_25_right-camchain.yaml`，再放到我们的 Stage5 evaluator 中同口径评估。

因此它应该被理解为：

- 外部参考内参。
- 棋盘格标定得到的相机模型。
- 在同一批 Stage5 outer/internal observations 上重新评估的 reference。

它不应该被理解为：

- Kalibr 在当前多板大 Tag 数据上的直接标定 baseline。
- 与我们完全同任务、同观测、同优化变量的竞争方法。

这个区别很重要。因为 Kalibr 自己输出的 reprojection error 通常只统计它自己优化时使用的 target corners，而我们的 Stage5 evaluator 同时拆分了 training / holdout、outer / internal，并且会暴露多板边缘、internal regeneration、运动模糊等问题。

## 4. 目前真正限制精度的主要因素

从四个 full 数据集和后续可视化来看，当前主要瓶颈不是相机模型整体错误，也不是 auto-init 失败，而是 internal point localization 的图像质量问题。

最明显的证据是：

- `20260427_191538` 数据较干净，训练集 x/y 标准差约为 `0.98 / 0.99`，验证集约为 `2.02 / 2.12`，说明算法在清晰图像上可以达到接近棋盘格标定的质量。
- `20260427_192347` 的 x 标准差明显大于 y，进一步可视化后发现主要由拍摄过程中的横向运动模糊和重影造成。
- 一些 worst frame 中，左右边缘 board 的 internal points 出现系统性偏移，而不是零散的单点 outlier。

因此，“x 方向误差总是更大”不是不可解决的物理限制，也不是 DS 模型天然不行。更准确的判断是：当前数据中存在方向性运动模糊，导致 internal localization 在 x 方向产生系统性偏差。

## 5. 多相机模型支持与 20260427_191538 对比

在当前 Stage5 baseline 基础上，我们已经完成了三种相机族的全链路接入：

- `ds-none`
- `eucm-none`
- `pinhole-equi`

这里的“全链路”包括：

- outer-only auto-init
- outer bootstrap
- internal regeneration
- Stage5 frontend
- ASLAM backend
- benchmark / summary 输出

其中需要特别记录一个 backend 修复点：`eucm-none` 在早期版本里会在 delayed intrinsics release 阶段崩溃，原因不是 EUCM 模型本身错误，而是 backend 在 intrinsics stage 中把 `NoDistortion` 家族的 `0` 维 distortion design variable 也一起激活了。修复后，`eucm-none` 已能稳定跑通 backend。

针对 `20260427_191538` 这组数据，目前最重要的多模型对比结果见：

- `stage5_camera_model_full_comparison_20260427_191538.csv`

当前结论如下：

### 5.1 DS-none

DS-none 仍然是 New Baseline 所使用的默认相机族。

在 `20260427_191538` full 数据集上：

- 训练集 Backend overall RMSE：`1.40492`
- 验证集 Backend overall RMSE：`2.93990`
- 与 Kalibr reference 几乎打平

这说明当前 baseline 在这组干净数据上已经足够强。

### 5.2 EUCM-none

EUCM-none 已经完整跑通 full 数据集。这里有一个 bookkeeping 细节：

- 结果目录名是 `stage5_backend_20260427_191538_first20_eucm_none`
- 但实际跑的是 full 数据集
- 可由 summary 中的 `training_frame_count=91`、`holdout_frame_count=23` 验证

EUCM-none 在这组数据上的结果是：

- Backend optimized overall RMSE：`0.920721`
- 训练集 Backend overall：`0.941787`
- 验证集 Backend overall：`2.99109`

与 DS baseline 对比：

- 训练集上，EUCM-none 明显更强。
- 验证集上，EUCM-none 没有赢过当前 DS baseline：
  - DS holdout overall：`2.93990`
  - EUCM holdout overall：`2.99109`

与当前 Kalibr reference 对比：

- EUCM holdout overall：`2.99109`
- Kalibr reference：`2.94375`

所以目前比较稳妥的说法是：

- `EUCM-none` 已经具备可用性，并且在训练集拟合上很强。
- 但在 `20260427_191538` 这组 full 数据上，它还没有稳定优于当前 `DS-none` 正式 baseline。
- 因此现阶段不建议直接用 EUCM 替换 DS baseline。

### 5.3 Pinhole-equi

`pinhole-equi` 的代码接入已经完成，但在 `20260427_191538` 这组数据上仍不够稳定。

当前现象是：

- auto-init 可以成功
- Stage5 round1 bundle 可以构建
- 但 benchmark 会给出 `optimized residuals did not improve`

因此：

- `pinhole-equi` 目前还不能作为正式 baseline 候选
- 更适合看作“已接入但尚未调稳”的模型族分支

### 5.4 当前多模型结论

这一轮多模型实验最重要的结论不是“DS 一定最好”或者“EUCM 一定最好”，而是：

- 当前框架已经不再是 DS-only。
- `EUCM-none` 已经具备完整运行能力。
- 在这组 full 数据上，`EUCM-none` 训练集更强，但 holdout 还没有稳定赢过 `DS-none`。
- `pinhole-equi` 仍需继续做初始化和前端稳定性改进。

因此，当前最合理的策略是：

- 正式 baseline 仍保持 `DS-none`
- `EUCM-none` 作为重点候选模型继续观察
- `pinhole-equi` 先不进入主线对比
## 6. 已完成的关键消融结论

### strict failed-board drop

141444 上的 failed-board 策略对比表明：

| 策略 | holdout overall | holdout outer | holdout internal | 结论 |
|---|---:|---:|---:|---|
| A / off-rescue | 7.0405 | 4.2109 | 7.3607 | 失败 board 仍可能影响 outer-only 评估 |
| strict failed-board drop | 7.0268 | 3.8359 | 7.3607 | 最稳，且不依赖 rescue |
| rescue gate8 | 7.3022 | 4.2109 | 7.6442 | 接回部分失败 board，但整体变差 |

结论：strict failed-board drop 比 wide-FOV rescue 更适合作为当前 baseline。rescue 保留为诊断或特殊场景实验，不建议作为主线。

### board pose-fit gate

board pose-fit gate 曾经用于防止个别 board pose fit 明显异常，但 full 和 first50 对照显示它并不是决定性收益来源。因此现在它被降级为：

- 稳定性保护。
- debug 开关。
- 非论文主线贡献。

### pre-backend point-level residual filter

我们实现了 Kalibr-style point-level residual outlier filter，并支持 `off / diagnostic / enabled`。实验结果显示它主要删除少量孤立 internal points，对最终指标帮助不明显。

原因是当前最主要问题不是“少数点 outlier”，而是“整块 board 或整帧由于模糊产生系统性偏移”。这种问题用 `mean + 2std` 的点级过滤并不理想。

### blur hard filter

后续又实现了 blur diagnostics 和 blur filter。保守 q05 版本可以删除最严重的一小部分 board：

- 140151 q05 删除约 5.44% internal points，但 holdout 略微变差。
- 141444 q05 删除约 5.41% internal points，holdout 只有极小改善。

结论：blur diagnostics 很有价值，但 hard filter 暂时不适合作为 baseline。它更适合帮助定位坏样本，而不是直接进入主算法。

### trial backend frame-board incremental selection

在 Old Baseline 的 strict selection 基础上，我们又实现并验证了一条默认关闭的 Kalibr-style `trial backend frame-board incremental selection` 实验线。经过多组实验后，`current` 配置已冻结为 **New Baseline**；原 `v3-soft` 统一命名为 **Consistency-Soft Enhancement**，作为可选增强分支。它的目标不是简单替换所有策略，而是验证：

- Old Baseline seed 之外的 frame-board observation 是否可以安全增量接回 backend；
- 接入更多 board coverage 后，是否能在不破坏稳定性的前提下改善 cross-dataset external validation；
- pre-backend point-level residual filter 与增量接入是否能一起工作。

这条实验线的核心做法是：

1. 先保留 Old Baseline 已经选中的 frame-board seed bundle；
2. 再从 full `round2` source measurement 中构建完整 candidate pool；
3. 对每个 candidate frame-board 计算 candidate score，结合：
   - outer/internal point count；
   - coverage gain；
   - board diversity；
   - 当前 residual 质量；
4. 按顺序增量尝试加入 candidate；
5. 每次加入一批后跑 short trial backend；
6. 如果 global / outer / internal RMSE 退化超阈值，则拒绝该 candidate；
7. 对最终保留的 bundle 再执行 pre-backend point-level residual pruning；
8. 特别修复了一个关键 bug：
   - pre-backend filter 删除的 internal points 不会在最终 backend input 中“复活”；
   - candidate pool 也不再错误继承 baseline 中非 seed board 的 `used=false` 状态。

本轮主要比较三档配置：

- `New Baseline`，对应实验配置名 `current`
  - `max_candidate_additions=20`
  - `min_candidate_score=3.2`
  - `min_coverage_gain=2.5`
  - `max_accepted_per_board=4`
  - `max_accepted_per_frame=1`
- `relaxed40`
  - `max_candidate_additions=40`
  - `min_candidate_score=2.8`
  - `min_coverage_gain=2.0`
  - `max_accepted_per_board=8`
  - `max_accepted_per_frame=2`
- `relaxed60`
  - `max_candidate_additions=60`
  - `min_candidate_score=2.4`
  - `min_coverage_gain=1.5`
  - `max_accepted_per_board=12`
  - `max_accepted_per_frame=3`

这三档都保持：

- `pre-backend-filter-mode=enabled`
- `pre-backend-filter-threshold-mode=mean_std`
- `pre-backend-filter-sigma=2.0`
- `pre-backend-filter-min-abs-threshold-px=0.2`

下面记录三组 `dataset_5_1` stereo-right cross-dataset external validation 的结果。为了方便和 New Baseline、Old Baseline 以及 Kalibr reference 直接对照，表中额外加入 `Kalibr reference` 与 `Old Baseline` 行；这两行来自前文 2.2 的结果，其中 `holdout outer / internal` 使用 2.2 中的 pose-only refit 诊断口径。

#### 20260430_134853 right -> 20260430_144419 right

| 配置 | accepted candidates | 最终 backend 输入 | holdout overall | holdout outer | holdout internal | training overall | 结论 |
|---|---:|---|---:|---:|---:|---:|---|
| Kalibr reference | - | 外部棋盘格 camchain reference | 2.1411 | 0.8926 | 2.2609 | 1.1980 | 外部参考内参 |
| Old Baseline | 0 | 16 frames / 78 boards / 2575 internal pts | 2.1105 | 0.7641 | 2.2345 | 1.1330 | 历史严格筛选对照 |
| New Baseline | 20 | 35 frames / 95 boards / 3066 pts | 1.94501 | 0.35204 | 2.07176 | 0.808203 | 默认主线，候选较保守 |
| relaxed40 | 40 | 43 frames / 115 boards / 3731 pts | 1.94398 | 0.32149 | 2.07135 | 0.808173 | 外部验证更好，outer 改善最明显 |
| relaxed60 | 60 | 46 frames / 135 boards / 4399 pts | 1.94379 | 0.32713 | 2.07102 | 0.807828 | overall 最低，但收益仍较小 |

这组数据说明：相对 Old Baseline，trial selection 系列整体明显降低 training / holdout RMSE；在 trial selection 内部继续放宽 candidate acceptance 也有小幅收益，尤其 `relaxed40` 对 outer holdout RMSE 改善较明显。

#### 20260430_144928 right -> 20260430_134853 right

| 配置 | accepted candidates | 最终 backend 输入 | holdout overall | holdout outer | holdout internal | training overall | 结论 |
|---|---:|---|---:|---:|---:|---:|---|
| Kalibr reference | - | 外部棋盘格 camchain reference | 3.4916 | 1.1723 | 3.6988 | 2.1157 | 外部参考内参 |
| Old Baseline | 0 | 19 frames / 85 boards / 2820 internal pts | 3.4419 | 1.1472 | 3.6464 | 1.8448 | 历史严格筛选对照 |
| New Baseline | 20 | 40 frames / 111 boards / 3596 pts | 3.25668 | 0.81435 | 3.46075 | 1.08004 | 当前最稳 |
| relaxed40 | 40 | 46 frames / 131 boards / 4249 pts | 3.25674 | 0.81376 | 3.46084 | 1.08032 | 基本持平，仅 outer 极小改善 |
| relaxed60 | 60 | 53 frames / 151 boards / 4898 pts | 3.25662 | 0.81324 | 3.46072 | 1.08037 | overall 略低，但幅度可以忽略 |

这组数据说明：相对 Old Baseline，trial selection 系列明显降低 overall / outer / internal RMSE；但在 trial selection 内部，`New Baseline / relaxed40 / relaxed60` 基本是中性差异，既没有明显收益，也没有明显破坏。

#### 20260430_144419 right -> 20260430_144928 right

| 配置 | accepted candidates | 最终 backend 输入 | holdout overall | holdout outer | holdout internal | training overall | 结论 |
|---|---:|---|---:|---:|---:|---:|---|
| Kalibr reference | - | 外部棋盘格 camchain reference | 8.5933 | 21.4454 | 4.4658 | 0.7313 | 外部参考内参 |
| Old Baseline | 0 | 11 frames / 53 boards / 1749 internal pts | 6.8392 | 15.2107 | 4.5828 | 0.6771 | 历史严格筛选对照 |
| New Baseline | 19 | 31 frames / 75 boards / 2425 pts | 6.59282 | 15.2985 | 4.10884 | 0.709491 | 当前最稳 |
| relaxed40 | 38 | 35 frames / 94 boards / 3051 pts | 6.59607 | 15.3028 | 4.11255 | 0.709756 | 略差 |
| relaxed60 | 59 | 40 frames / 115 boards / 3751 pts | 6.61269 | 15.3478 | 4.11955 | 0.710512 | 更差，不适合主推 |

这组数据说明：相对 Old Baseline，trial selection 的 New Baseline 降低了 holdout overall 和 internal RMSE，但 outer RMSE 略高；在 trial selection 内部，更多 candidate 并不是单调有益的，在困难外部验证集上，过度放宽反而会轻微退化。

#### 当前阶段性结论

从这三组 cross-dataset external validation 可以得到比较清楚的判断：

- `trial backend frame-board incremental selection` 这条线现在已经真正打通，不再只是诊断开关。
- candidate pool 能从 baseline seed 的几十个 frame-board 扩展到 full source measurement 的数百个 frame-board，再通过 short trial backend 逐步接回一部分 candidate。
- `New Baseline / relaxed40 / relaxed60` 三档都没有出现明显失稳，说明这套增量接回机制在工程上是可用的。
- 但收益具有明显数据集相关性：
  - `134853 -> 144419` 上，`relaxed40 / relaxed60` 有小幅正收益；
  - `144928 -> 134853` 上，relaxed 基本持平；
  - `144419 -> 144928` 上，relaxed 反而轻微变差。

因此当前建议是：

- `New Baseline` 继续作为默认实验主线，因为跨数据集最稳；
- `relaxed40` 保留为可选实验分支，因为它在部分数据集上有正收益，而且总体风险不大；
- `relaxed60` 暂时不建议提升为主线，因为它虽然能接入更多 board，但泛化收益不稳定。

也就是说，这一轮实验已经从“代码是否打通”进入“策略如何调得更好”的阶段。后续更值得做的不是继续无脑放宽 candidate 数量，而是分析：

- 哪些 accepted candidate 真正带来收益；
- 哪些 board/frame 会把 holdout 拉坏；
- 是否需要把 local-vs-global consistency、polar coverage、board diversity 更显式地写进 candidate score。

为此已经新增 accepted candidate 质量分析文件：

- `exp_data/stage5_trial_backend_candidate_quality_analysis_cn.md`
- `exp_data/stage5_trial_backend_accepted_candidates.csv`
- `exp_data/stage5_trial_backend_candidate_quality_summary.csv`
- `exp_data/stage5_trial_backend_candidate_reason_counts.csv`
- `exp_data/stage5_trial_backend_accepted_board_distribution.csv`
- `exp_data/stage5_trial_backend_relaxed_vs_current.csv`

这份分析的当前结论是：

- accepted candidate 的平均 `candidate_score` 通常在 3 左右，说明当前接入的不是随机 board，而是经过 coverage / diversity / residual quality 共同筛选后的 frame-board observation。
- `not_attempted_board_candidate_cap` 在多数实验中仍是最大拦截项，说明 per-board cap 对最终接入分布影响很大。
- `134853 -> 144419` 是 relaxed 策略最有收益的数据集；`144928 -> 134853` 基本中性；`144419 -> 144928` 则出现轻微退化。
- 下一步更值得做的是把 local-vs-global consistency 和 candidate-only residual pruning 加进策略，而不是继续单纯放宽 `max_candidate_additions`。

## 7. 可视化发现

针对 x 方向误差，我们已经生成了 worst frame 可视化：

- `result/stage5_backend_full_20260427_191538_xstd_viz/overlays/frame_000012_internal_localization.png`
- `result/stage5_backend_full_20260427_192347_xstd_viz/overlays/frame_000011_internal_localization.png`
- `result/stage5_backend_full_20260427_192347_xstd_viz/overlays/frame_000096_internal_localization.png`
- `result/stage5_backend_full_20260427_192347_xstd_viz/overlays/frame_000088_internal_localization.png`
- `result/stage5_backend_full_20260427_192347_xstd_viz/overlays/frame_000074_internal_localization.png`

可视化元素解释：

- 青色点表示实际生成并参与评估/优化的 internal observation。
- 洋红色十字表示当前模型投影出来的 reference / reprojection 位置。
- 黄色线段表示 residual vector。
- 线段越长，说明该 internal point 的观测与模型投影差异越大。

需要注意：洋红色十字不是物理意义上的“绝对真实角点”，而是当前 Stage5 evaluator / backend state 下的模型投影参考。它用于解释 residual 分布和系统性偏差。

## 8. 对“多板大 Tag 标定是否不如棋盘格”的判断

目前不能简单得出“多板大 Tag 标定天然不如棋盘格”的结论。

更合理的结论是：

- 在清晰数据 `20260427_191538` 上，我们已经能达到接近 Kalibr 棋盘格 reference 的表现。
- 在模糊数据上，误差变大主要来自 internal point localization，而不是相机模型或优化框架天然不足。
- 棋盘格角点通常更密、更规则、检测成熟度更高，对清晰图像尤其友好。
- 我们的大 Tag 多板方案优势在于大视场覆盖、多板布局和当前场景适配，但它对 internal point 生成质量更敏感。

所以，当前方法仍有提升空间，但提升方向不应该继续堆 gate，而应该针对 internal localization 的不确定性建模。

## 9. 下一步建议

优先级最高的下一步不是继续调阈值，而是做 internal measurement quality modeling。

推荐方向：

- 对 internal points 引入质量权重，而不是硬删除。
- 根据 blur / gradient / residual pattern 给 internal observations 设置 covariance。
- 研究 x/y 各向异性权重，尤其针对横向运动模糊导致的 x 方向误差。
- 将 board-level image quality 作为诊断量进入 summary，而不是直接作为 baseline filter。

可以作为下一组实验的候选分支：

- `internal_quality_weighting_diagnostic`
- `anisotropic_internal_covariance`
- `blur_aware_internal_weighting`

这条线比继续做 hard blur filter 更有论文价值，因为它不是“针对某个数据集删坏图”，而是把观测不确定性显式建模，逻辑上也更接近严谨的 calibration / bundle adjustment 方法。

## 10. 当前可用于论文的核心说法

当前实验已经支持下面几个比较稳的说法：

- 自动初始化替代手写 DS 内参后，完整 Stage5 + ASLAM backend pipeline 可以稳定工作。
- strict failed-board drop 是合理的 Kalibr-style target observation acceptance 策略，能避免失败 board 污染后续评估和优化。
- 在四个 full 数据集上，当前方法与棋盘格 Kalibr reference 在同一 Stage5 evaluator 下整体相当或更优。
- 对干净数据，方法可以达到接近 2 px 级别的验证集标准差。
- 对模糊数据，主要误差来自 internal localization 的系统性偏差，尤其是 x 方向运动模糊。
- 后续最有价值的提升方向是 internal observation quality / covariance modeling，而不是继续扩大手工过滤规则。
