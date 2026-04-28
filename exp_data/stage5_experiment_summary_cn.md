# Stage5 Backend 实验总结

## 1. 当前正式 baseline

当前建议作为正式 baseline 的方案是：

`auto-init + round2 + delayed intrinsics release + residual sanity gate + strict failed-board drop + Stage5 evaluator + ASLAM backend`

也就是说，当前 baseline 的核心定义是：

- 相机初始化：自动初始化，不依赖手写 Double Sphere 内参。
- 前端流程：保留 round2 / second-pass internal regeneration。
- 内参释放：前端与 backend 都使用 delayed intrinsics release。
- 观测接受策略：采用 Kalibr-style strict failed-board drop。
- board pose-fit gate：降级为 debug 选项，不属于论文主线。
- blur filter / pre-backend point filter：保留为实验和诊断工具，不进入 baseline。

strict failed-board drop 的含义是：如果某一帧中的某个 board 在 internal pose estimation / internal regeneration 阶段失败，则这个 board observation 整块丢弃；同一帧里其它正常 board 继续使用。这个策略更接近 Kalibr 的 target observation 接受逻辑，避免把“内部点生成失败但外部点还存在”的 board 继续混入后续评估或优化。

## 2. 四个 full 数据集主结果

下面这张表来自 `stage5_strict_baseline_4datasets.csv`，是当前最重要的 full 级别实验汇总。

| 数据集 | 训练/验证帧 | 选中帧/板/内部点 | Backend 优化后 RMSE | 训练集 Backend / Kalibr | 验证集 Backend / Kalibr | 验证集 x/y 标准差 |
|---|---:|---:|---:|---:|---:|---:|
| 20260421_140151 | 292 / 73 | 17 / 71 / 2057 | 3.444 | 4.0542 / 4.1163 | 6.6279 / 6.6896 | 5.7478 / 3.2946 |
| 20260421_141444 | 229 / 58 | 25 / 106 / 3050 | 4.6141 | 7.6197 / 9.0040 | 7.0268 / 7.7321 | 5.8610 / 3.8160 |
| 20260427_191538 | 91 / 23 | 13 / 65 / 1892 | 1.0451 | 1.4049 / 1.4470 | 2.9399 / 2.9392 | 2.0178 / 2.1206 |
| 20260427_192347 | 100 / 26 | 15 / 70 / 2027 | 3.1677 | 4.8985 / 5.3114 | 4.2005 / 4.6305 | 3.5487 / 2.1736 |

整体结论：

- 在四个 full 数据集上，当前 strict baseline 都能稳定运行。
- 在 `20260421_140151` 和 `20260421_141444` 上，Backend 在 Stage5 evaluator 同口径下优于外部 Kalibr reference。
- 在 `20260427_191538` 上，Backend 和 Kalibr reference 几乎打平，而且误差已经接近棋盘格标定常见的 2 px 量级。
- 在 `20260427_192347` 上，Backend 仍优于 Kalibr reference，但 x 方向标准差明显偏大，主要来自运动模糊/重影图像。

一个重要 bookkeeping 备注：`20260427_192347` 的结果目前存放在 `result/stage5_backend_full_140151_strict_baseline_check`，这个目录名有误导性，因为当时复用了旧 output path。后续引用时应以 `stage5_strict_baseline_4datasets.csv` 中记录的真实图像路径为准。

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

## 5. 已完成的关键消融结论

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

## 6. 可视化发现

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

## 7. 对“多板大 Tag 标定是否不如棋盘格”的判断

目前不能简单得出“多板大 Tag 标定天然不如棋盘格”的结论。

更合理的结论是：

- 在清晰数据 `20260427_191538` 上，我们已经能达到接近 Kalibr 棋盘格 reference 的表现。
- 在模糊数据上，误差变大主要来自 internal point localization，而不是相机模型或优化框架天然不足。
- 棋盘格角点通常更密、更规则、检测成熟度更高，对清晰图像尤其友好。
- 我们的大 Tag 多板方案优势在于大视场覆盖、多板布局和当前场景适配，但它对 internal point 生成质量更敏感。

所以，当前方法仍有提升空间，但提升方向不应该继续堆 gate，而应该针对 internal localization 的不确定性建模。

## 8. 下一步建议

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

## 9. 当前可用于论文的核心说法

当前实验已经支持下面几个比较稳的说法：

- 自动初始化替代手写 DS 内参后，完整 Stage5 + ASLAM backend pipeline 可以稳定工作。
- strict failed-board drop 是合理的 Kalibr-style target observation acceptance 策略，能避免失败 board 污染后续评估和优化。
- 在四个 full 数据集上，当前方法与棋盘格 Kalibr reference 在同一 Stage5 evaluator 下整体相当或更优。
- 对干净数据，方法可以达到接近 2 px 级别的验证集标准差。
- 对模糊数据，主要误差来自 internal localization 的系统性偏差，尤其是 x 方向运动模糊。
- 后续最有价值的提升方向是 internal observation quality / covariance modeling，而不是继续扩大手工过滤规则。

