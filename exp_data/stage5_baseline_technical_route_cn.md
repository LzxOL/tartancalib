# 当前 Stage5 Baseline 技术路线中文总结

## 1. 这份文档的目的

这份文档不是实验结果汇总，而是当前 baseline 的“技术路线说明书”。

它回答的问题是：

- 我们现在的 baseline 到底是什么。
- 它从输入图像到最终 backend 输出，中间到底经过了哪些模块。
- 每个模块的输入、输出、职责是什么。
- 哪些开关是 baseline 真正启用的，哪些虽然代码里有，但只是实验/诊断分支。
- 为什么当前 baseline 会长成这个样子。

如果把我们现在的方法用一句话概括，可以写成：

`多板 AprilTag 外角点检测 -> 自动相机初始化 -> outer bootstrap -> round1 internal regeneration + joint optimization -> round2 second-pass regeneration + joint optimization -> Stage5 training/holdout evaluator -> ASLAM backend refinement`

在当前正式 baseline 中，还需要额外加上两个限定：

- 使用 `strict failed-board drop`
- 使用 `delayed intrinsics release`

## 2. 当前 baseline 的准确含义

### 2.1 论文/实验意义上的 baseline

当前我们讨论的 baseline，不是“程序默认值”，而是“正式实验命令所对应的一整套固定协议”。

它的核心定义是：

- 相机初始化：`auto`
- 前端 second pass：`on`
- 前端内参优化：`on`
- 前端内参释放策略：`delayed`
- backend 内参优化：`on`
- backend 内参释放策略：`delayed`
- residual sanity gate：`on`
- board pose-fit gate：`off`
- strict board observation acceptance：`on`
- internal pose rescue：`off`
- pre-backend internal residual filter：`off`
- internal blur filter：`off`
- runtime mode：`fast`

### 2.2 代码默认值 vs 正式 baseline

这里有一个很重要的区别：

- `run_stage5_backend_main.cpp` 里有一些“代码默认值”
- 但我们现在认可的 baseline，是靠命令行参数显式覆盖出来的

例如：

- `strict_board_observation_acceptance` 在代码默认值里并不是自动开启的
- `board pose-fit gate` 默认也是关闭的
- `internal blur filter`、`pre-backend filter` 这些模块默认也都是关闭的

所以，不能简单把“当前代码能跑通的默认行为”理解成“正式 baseline”。

正式 baseline 更准确地说，是：

1. 固定了 protocol label：`stage5_backend_auto_v1`
2. 固定了一套命令行参数
3. 固定了一套 summary 输出口径
4. 固定了和 Kalibr reference 的比较方式

## 3. 一条标准 baseline 命令长什么样

以 full 数据集为例，当前 baseline 命令的结构应当是：

```bash
./build/run_stage5_backend \
  --image image/dataset_4-22/right_record_20260421_140151 \
  --config aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml \
  --output result/stage5_backend_full_140151_kalibr_style_failed_board_drop \
  --kalibr-camchain config/mono_fisheye_calib_3_25_right-camchain.yaml \
  --all \
  --holdout-stride 5 \
  --holdout-offset 0 \
  --camera-init-mode auto \
  --intrinsics-release-mode delayed \
  --runtime-mode fast \
  --strict-board-observation-acceptance
```

这条命令背后的真实含义是：

- 输入是整段图像序列
- 采用 stride=5 的 deterministic holdout 划分
- 训练集用于前端 + backend 优化
- holdout 集只用于评估，不参与训练
- Kalibr camchain 只作为 reference camera，不作为优化初值主线

## 4. 整体架构总览

当前 baseline 可以分成 7 个大阶段：

1. 数据收集与 split
2. training outer detection
3. automatic camera initialization
4. outer bootstrap
5. round1 / round2 前端优化
6. Stage5 benchmark 评估
7. ASLAM backend refinement

从实现位置上看，主控入口是：

- `aslam_cv/aslam_cameras_april/src/run_stage5_backend_main.cpp`

前端主流程是：

- `aslam_cv/aslam_cameras_april/src/apriltag_internal/FrozenRound2BaselinePipeline.cpp`

训练/验证评估与 backend 输入构建在：

- `aslam_cv/aslam_cameras_april/src/apriltag_internal/Stage5Benchmark.cpp`
- `aslam_cv/aslam_cameras_april/src/apriltag_internal/CalibrationStateBundle.cpp`

backend 优化在：

- `aslam_cv/aslam_cameras_april/src/apriltag_internal/AslamBackendCalibrationRunner.cpp`

## 5. 输入数据与配置

### 5.1 图像输入

程序接收的是一个图像目录。`run_stage5_backend_main.cpp` 会先把目录下图像路径收集出来，并按文件顺序组织成：

- `frame_index`
- `frame_label`
- `image_path`

内部使用的结构是：

- `FrozenRound2BaselineFrameSource`

这意味着整个系统从一开始就是“按帧组织”的。

### 5.2 标定板配置

配置文件当前是：

- `aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml`

里面最重要的字段包括：

- `tagIds: [1, 2, 3, 4, 5]`
- `tagSize`
- `tagSpacing`
- `canonical_pixels_per_module`
- `internal_projection_mode: sphere_border_lattice`
- `camera_initialization_mode: auto_with_manual_fallback`（配置文件默认）

不过在正式 baseline 命令里，我们会通过 CLI 再把相机初始化模式显式改成 `auto`。

### 5.3 Kalibr reference

当前默认只有一份外部参考内参：

- `config/mono_fisheye_calib_3_25_right-camchain.yaml`

它的作用是：

- 提供外部参考 camera
- 在 Stage5 evaluator 里做同口径比较

它不是当前主线优化的初值来源，也不是多板大 Tag 场景下的直接 Kalibr baseline。

## 6. 数据划分：training / holdout

Stage5 benchmark 的一个核心设计，是把数据分成 training 和 holdout 两部分。

对应结构体在：

- `CalibrationBenchmarkSplitOptions`
- `CalibrationBenchmarkSplit`

当前默认策略是：

- `mode = deterministic_stride`
- `holdout_stride = 5`
- `holdout_offset = 0`

意思是：每隔 5 帧抽 1 帧放到 holdout，其余用于 training。

这样做的目的有两个：

1. 不让所有指标都只在训练帧上报，避免“优化过的帧当然更好看”的误导
2. 保留一个固定、可重复的评估协议，便于不同实验之间公平比较

## 7. training outer detection

### 7.1 做什么

这一阶段只做一件事：

- 在每张训练图像上检测外层 AprilTag board 的 4 个 refined outer corners

这里“outer”是后续一切的基础，因为：

- auto-init 只依赖 outer
- outer bootstrap 只依赖 outer
- internal regeneration 的几何先验也依赖 outer pose

### 7.2 代码位置

主调用位于：

- `FrozenRound2BaselinePipeline::Run(...)`

使用的检测器是：

- `MultiScaleOuterTagDetector`

### 7.3 缓存

当前 baseline 默认开启 outer detection 磁盘缓存：

- `enable_outer_detection_cache = true`

缓存模块是：

- `OuterDetectionCache`

缓存只加速重复运行，不改变算法语义。

也就是说：

- 缓存命中时，复用已经保存的 outer detection
- 缓存未命中时，重新检测并存盘

这一步只影响运行时间，不影响结果。

## 8. automatic camera initialization

### 8.1 目标

这一步的目标是：

- 自动生成一个可用的 Double Sphere 初始相机
- 替代过去手写的 dataset-specific DS 内参

### 8.2 代码模块

对应模块是：

- `OuterOnlyCameraInitializer`

头文件：

- `aslam_cv/aslam_cameras_april/include/aslam/cameras/apriltag_internal/OuterOnlyCameraInitializer.hpp`

### 8.3 它依赖什么输入

它只依赖 outer observations：

- 每个 `(frame, board)` 的 4 个 refined outer corners
- 对应 board-frame 下的 outer 3D corner coordinates
- 图像分辨率

它**不依赖 internal regeneration**，这是设计上的关键点。因为 internal regeneration 本身就依赖于一个初始相机，如果反过来依赖 internal，会形成循环。

### 8.4 它怎么做

自动初始化的策略大致是：

1. 从图像尺寸推断主点初值，大致放在图像中心
2. 枚举一组焦距比例候选
3. 枚举一组 `xi` 候选
4. 枚举一组 `alpha` 候选
5. 形成 DS candidate grid
6. 对每个 candidate，在 outer observations 上做 pose fit
7. 用 pose success rate 和 outer reprojection RMSE 对 candidate 排序
8. 取最优 candidate，并可选地做一次 refined reevaluation

默认的候选网格包括：

- `focal_scale_candidates = {0.18, 0.22, 0.26, 0.30, 0.34, 0.40, 0.50, 0.60}`
- `xi_candidates = {-0.4, -0.2, 0.0, 0.2, 0.5, 1.0}`
- `alpha_candidates = {0.35, 0.45, 0.55, 0.65, 0.75}`

### 8.5 输出

输出结构是：

- `AutoCameraInitializationResult`

包含：

- 最终选中的相机参数
- 候选数量
- 采样/全量 observation 数量
- pose fit 成功/失败计数
- initialization RMSE
- candidate ranking
- residual diagnostics

### 8.6 fallback 的位置

代码仍然保留 manual camera 作为 fallback/debug 路径。

但是当前 baseline 命令用的是：

- `--camera-init-mode auto`

所以正式 baseline 的解释是：

- 自动初始化必须自己成功
- 不依赖 fallback

## 9. outer bootstrap

### 9.1 目标

有了初始 camera 之后，系统要做第一轮场景几何初始化，也就是 outer bootstrap。

它的目标是估计：

- 相机内参初值
- 每个 board 的位姿
- 每帧的相机位姿

### 9.2 模块

使用的模块是：

- `MultiBoardOuterBootstrap`

### 9.3 输入

输入是：

- training frames 上的 outer detections
- auto-init 选出的初始相机

### 9.4 输出

输出是：

- `OuterBootstrapResult`

它提供了最初的 scene state，后续 round1 internal regeneration 和 measurement builder 都会用它。

## 10. round1 internal regeneration

### 10.1 这一步为什么重要

我们的 pipeline 和传统“只用外角点”最大的不同之一，就是我们不只使用 outer corners，还会利用每个大 Tag 内部生成的大量 internal points。

这些 internal points 不是 detector 直接给的标准 AprilTag 角点，而是通过内部几何模型和图像证据“再生成”出来的。

### 10.2 模块

对应模块是：

- `MultiBoardInternalMeasurementRegenerator`

它的内部又会调用：

- `ApriltagInternalDetector`

### 10.3 输入

round1 regeneration 的输入是：

- 原始图像
- 当前帧 outer detections
- bootstrap 初始化得到的 scene state

### 10.4 internal regeneration 在做什么

直观地说，它在做以下事情：

1. 先根据 outer 几何和当前 camera/pose，预测内部点大概应该落在哪
2. 基于 canonical model 和 sphere-border-lattice 模式生成内部结构先验
3. 在图像上结合 template / gradient / peak / prior 等证据找更可信的位置
4. 做 ray refine / subpixel refine
5. 给出每个 internal point 是否有效、最终坐标、以及质量分数

### 10.5 当前 baseline 为什么容易受模糊影响

因为 internal points 的质量高度依赖图像局部边缘和梯度信息。

如果图像存在：

- 横向运动模糊
- 重影
- 边缘拉伸

那么 internal localization 往往会出现“整块 board 系统性偏移”，而不是少量离散 outlier。这也是为什么很多简单 point filter 效果一般。

## 11. joint measurement builder

### 11.1 作用

internal regeneration 完成后，系统要把 outer / internal 两类观测统一组织成一个 joint measurement dataset。

对应模块是：

- `JointReprojectionMeasurementBuilder`

### 11.2 输入

输入是每一帧的：

- outer detections
- regenerated internal measurements
- bootstrap result

### 11.3 它构建的对象

它会生成：

- `JointPointObservation`
- `JointBoardObservation`
- `JointMeasurementFrameResult`
- `JointMeasurementBuildResult`

### 11.4 strict failed-board drop 在这里生效

这一点非常关键。

builder 有一个选项：

- `include_outer_when_internal_failed`

在 baseline 中，因为启用了 `strict-board-observation-acceptance`，所以这个选项会被设成：

- `false`

它的含义是：

- 如果某个 board 的 internal regeneration 失败
- 那么这个 board 的 outer points 也不再保留进 solver dataset

也就是说，strict baseline 不是“只删 internal points”，而是“整块 board observation 不纳入后续求解”。

这和 Kalibr 更接近，因为 Kalibr 的 target observation 如果不成立，通常不会继续拿其中一部分 corner 进入优化。

### 11.5 builder 还会做什么

builder 还会给每个 point 附带：

- point type：outer / internal
- quality
- source kind
- used_in_solver
- rejection reason

所以它不仅是“拼接数据”，也是 solver 输入前的第一次语义化整理。

## 12. residual evaluation

### 12.1 作用

在 measurement build 完成后，系统会先在当前 scene state 下评估一次 residual。

对应模块是：

- `JointReprojectionResidualEvaluator`

### 12.2 它评估什么

它会统计：

- overall RMSE
- outer-only RMSE
- internal-only RMSE
- 每帧 / 每板的 residual 情况

这里的 residual evaluation 是后续 selection 和优化的依据之一。

## 13. selection

### 13.1 为什么要 selection

不是所有 frame 和 board observation 都要进入 solver。

selection 的作用是：

- 保证每个 board 至少有一定数量的有效视角
- 保留覆盖更好的观测
- 去掉明显不靠谱的候选 board observation

### 13.2 模块

对应模块是：

- `JointMeasurementSelection`

### 13.3 当前 baseline 的主要参数

从 `JointMeasurementSelectionOptions` 看，关键参数包括：

- `min_initial_views_per_board = 3`
- `residual_sanity_factor = 2.5`
- `max_pose_fit_outer_rmse = 8.0`

当前 baseline 中：

- `enable_residual_sanity_gate = true`
- `enable_board_pose_fit_gate = false`

### 13.4 这意味着什么

当前 baseline 的 selection 语义是：

- 保留 residual sanity gate 作为稳健性保护
- 不再把 board pose-fit gate 作为主线筛选条件

也就是说，我们现在认为：

- residual sanity gate 是保守但合理的
- board pose-fit gate 更像 debug/stability 工具，而不是核心算法贡献

## 14. round1 joint optimization

### 14.1 模块

对应模块是：

- `JointReprojectionOptimizer`

### 14.2 优化变量

前端 joint optimization 优化的是：

- frame poses
- board poses
- camera intrinsics（如果启用）

### 14.3 delayed intrinsics release

当前 baseline 的内参释放策略是 delayed。

对前端来说：

- round1 `intrinsics_release_iteration = 3`
- round2 `second_pass_intrinsics_release_iteration = 1`

它的含义是：

- 优化初期先主要稳定 pose/structure
- 到一定 iteration 后再放开 intrinsics

这样做是为了降低“初值还不稳时内参过早乱跑”的风险。

### 14.4 round1 输出

输出是：

- `JointOptimizationResult`

以及基于它构建的：

- `stage5_round1_bundle`

这个 bundle 已经是一个结构化的 Stage5 状态快照，可以被后续 evaluator 和 backend 使用。

## 15. round2 / second pass

### 15.1 为什么有 round2

round1 的优化结果会给出一个更好的：

- camera
- frame poses
- board poses

在这个更好的几何状态下，再重新做一遍 internal regeneration，通常能得到更稳的 internal observations。

所以 round2 的思路是：

- 用优化后的 state 反过来提升 measurement 质量

### 15.2 round2 的流程

round2 基本重复 round1 的链路：

1. 重新 internal regenerate
2. 重新 build joint measurement
3. 重新 residual evaluate
4. 重新 selection
5. 再做一次 joint optimization

### 15.3 round2 输出

最终输出：

- `final_stage5_bundle`

这个 bundle 是当前 baseline 真正送入 backend 的前端结果。

### 15.4 baseline 中 round2 的地位

当前 baseline 中：

- `run_second_pass = true`

也就是说 round2 是 baseline 的组成部分，不是额外实验项。

## 16. Stage5 bundle 的意义

Stage5 bundle 是前端与 backend 之间的桥梁。

它打包了：

- 当前 scene state
- 已接受的 measurement dataset
- residual statistics
- bundle metadata

一旦 bundle 被标记为 `ready_for_backend`，就可以进一步构造成 backend problem input。

## 17. training / holdout evaluation dataset

### 17.1 training dataset 怎么来

training evaluation dataset 直接从前端最终 bundle 中读取：

- 只取 `used_in_solver` 的点
- 保留 outer / internal 区分

也就是说，training evaluation 评估的是：

- 前端最终真正用于求解的那批观测

### 17.2 holdout dataset 怎么来

holdout dataset 不是简单从 training 复制出来，而是重新单独构建。

流程是：

1. 在 holdout 图像上重新做 outer detection
2. 基于前端优化后的 scene state 做 internal regeneration
3. 构造 holdout evaluation dataset

这里有一个重要点：

- holdout 不参与训练
- 但会用前端优化后的 state 来做投影与 regeneration

### 17.3 strict failed-board drop 在 holdout 里也生效

在 `Stage5Benchmark::BuildHoldoutEvaluationDataset(...)` 里也有同样逻辑：

- 如果 strict 开启，且某 board regeneration 失败
- 那么该 board 在 holdout evaluation dataset 中直接跳过

因此，strict 不是只作用于训练集 solver 输入，而是训练/验证口径一致。

## 18. Stage5 evaluator 的评估口径

### 18.1 它评估什么

Stage5 evaluator 的核心协议可以概括成：

- 用 camera intrinsics
- 对每个 board 先用 outer points refit 一个 board pose
- 再在这个 pose 下评估 outer 和 internal reprojection residual

因此 current protocol 不是简单的“把所有点直接投影一下”，而是：

- camera-only evaluation
- outer-refit-pose + outer/internal reprojection

### 18.2 为什么这样做

这样做可以尽量把评价集中在：

- camera model 是否合理
- internal localization 是否合理

而不是让 frame/board pose 的误差完全淹没掉 camera comparison。

### 18.3 输出指标

最常见的 summary 指标包括：

- overall RMSE
- outer-only RMSE
- internal-only RMSE
- mean residual x / y
- std residual x / y

这也是我们后面一直拿来和 Kalibr reference 做同口径对比的指标。

## 19. pre-backend curation / filters 在 baseline 里的位置

### 19.1 模块是存在的

当前代码里已经有两个“前 backend 观测整理”分支：

- `pre_backend_filter_mode`
- `internal_blur_filter_mode`

它们都位于：

- final Stage5 bundle 之后
- `BuildBackendProblemInput(...)` 之前

### 19.2 但 baseline 默认关闭

当前正式 baseline 中：

- `pre_backend_filter_mode = off`
- `internal_blur_filter_mode = off`

也就是说：

- backend 输入直接来自 final Stage5 bundle
- 不额外删点
- 不额外基于 blur 做 board 过滤

之所以这么定，是因为目前实验说明：

- 这些 filter 更有诊断价值
- 但还不足以稳定提升 full 数据集表现

## 20. backend problem input

### 20.1 构造位置

对应函数是：

- `BuildBackendProblemInput(...)`

在：

- `aslam_cv/aslam_cameras_april/src/apriltag_internal/CalibrationStateBundle.cpp`

### 20.2 它做什么

它把前端 bundle 转换为 backend 可消费的结构：

- scene_state
- measurement_dataset
- optimization masks
- priors
- diagnostics seed

### 20.3 backend 的优化掩码

当前 baseline backend 使用：

- `optimize_frame_poses = true`
- `optimize_board_poses = true`
- `optimize_intrinsics = true`
- `delayed_intrinsics_release = true`

也就是说 backend 不是只调内参，也不是只调 pose，而是 joint optimization。

### 20.4 intrinsics anchor prior

backend problem input 会自动启用：

- `use_intrinsics_anchor_prior = true`

也就是对内参加一个 anchor prior，防止优化过程中相机模型发散。

这个 prior 不是为了把结果“锁死”，而是为了提供合理约束。

## 21. ASLAM backend optimization

### 21.1 模块

对应模块是：

- `AslamBackendCalibrationRunner`

### 21.2 它优化什么

backend 进一步在 ASLAM 框架里构建设计变量和误差项，继续优化：

- frame poses
- board poses
- camera intrinsics

### 21.3 delayed release 在 backend 里的实现

backend 里 delayed release 的实现方式是分阶段：

1. 先跑 `pose_only`
2. 再跑 `intrinsics_released`

代码里会根据 `intrinsics_release_iteration` 自动拆分这两个阶段。

所以当前 baseline backend 的优化节奏是：

- 先稳定 pose
- 再放开内参与 pose 一起优化

### 21.4 backend 输出

backend 会产出：

- optimized scene state
- per-stage optimization summary
- final backend camera intrinsics
- training / holdout 再评估结果

对应常见输出文件包括：

- `backend_optimization_summary.txt`
- `backend_training_summary.txt`
- `backend_holdout_summary.txt`
- `backend_vs_frontend_summary.txt`

## 22. fast 模式与 research 模式

### 22.1 fast 模式保留什么

当前 baseline 命令使用：

- `--runtime-mode fast`

fast 模式保留：

- auto-init
- round1 / round2
- Stage5 training / holdout benchmark
- backend full optimization
- 各类 summary / csv

### 22.2 fast 模式关闭什么

fast 模式会关闭：

- minimal pose-only smoke
- Jacobian consistency check
- cost parity diagnostics
- 各类大批量 overlay 导出

### 22.3 为什么 baseline 用 fast

因为 fast 模式：

- 不改变核心算法语义
- 能显著缩短运行时间
- 保留我们论文和实验最关心的数值 summary

所以它更适合作为日常 baseline 运行方式。

## 23. 当前 baseline 为什么选择 strict failed-board drop

这是目前 baseline 中非常关键的一条设计选择。

之所以采用 strict，是因为实验显示：

- 当某个 board 的 internal regeneration 已经失败时
- 保留该 board 的 outer points 继续参与评估/求解，容易让这个 board 以“半有效状态”污染结果

strict 的逻辑更干净：

- 这个 board observation 成立，就整块使用
- 这个 board observation 不成立，就整块丢弃

这样有几个好处：

1. 语义更一致
2. 更接近 Kalibr 风格的 target observation acceptance
3. 在 141444 等边缘失败数据上，比 rescue 更稳

## 24. 当前 baseline 为什么不用 rescue / blur hard filter / pre-backend filter

### rescue

wide-FOV rescue 可以救回一部分失败 board，但 full 实验说明：

- 它会让一小部分本该失败的 board 重新进入系统
- 最终未必改善 overall / internal RMSE

因此 rescue 目前保留为实验支线，而不是 baseline。

### blur hard filter

blur hard filter 能删掉最差的一些模糊 board，但它的问题是：

- 容易损失覆盖率
- 收益不稳定
- 更像“删坏样本”而不是“改进观测建模”

因此目前不进 baseline。

### pre-backend residual filter

point-level residual filter 主要能清理少数 isolated outlier，但当前主要问题往往是：

- 整块 board 的系统性 internal 偏移

所以它不是当前最有效的主线改进。

## 25. 当前 baseline 的技术优势

如果从方法论角度总结，当前 baseline 的优势在于：

1. 不再依赖手写数据集内参，自动初始化已经打通。
2. 利用 outer + internal 两层观测，而不是只依赖单一 outer corners。
3. round2 second-pass 让 measurement quality 和 geometry 可以互相迭代提升。
4. 通过 training / holdout 分离，建立了更像 benchmark 的评估协议。
5. strict failed-board drop 让观测接受逻辑更干净、更接近 Kalibr 风格。
6. backend 继续做 joint refinement，而不是停留在前端 frozen 结果。

## 26. 当前 baseline 的主要短板

从技术上看，当前 baseline 的主要短板也很清楚：

1. internal localization 对图像质量非常敏感。
2. 横向运动模糊会把 x 方向 residual 拉大。
3. 当前对 internal observations 还是“接受/拒绝”风格为主，缺少更细致的不确定性建模。
4. hard filter 类方法对 full 数据集提升不稳定。

所以后续最值得做的，不是继续加更多 gate，而是：

- internal quality weighting
- blur-aware covariance
- x/y anisotropic uncertainty modeling

## 27. 你可以怎么读代码

如果你想顺着代码理解整条路线，建议按下面顺序看：

1. `run_stage5_backend_main.cpp`
   先看命令行参数如何被翻译成 requested/effective config。

2. `FrozenRound2BaselinePipeline.hpp / .cpp`
   看 training 前端主流程：outer detection -> auto-init -> bootstrap -> round1 -> round2。

3. `OuterOnlyCameraInitializer.hpp / .cpp`
   看自动相机初始化的 candidate grid 与筛选逻辑。

4. `MultiBoardInternalMeasurementRegenerator` 和 `ApriltagInternalDetector.cpp`
   看 internal regeneration 是如何从几何预测走到图像证据优化的。

5. `JointReprojectionMeasurementBuilder.hpp / .cpp`
   看 outer / internal observations 如何合并，以及 strict 是在哪里生效的。

6. `JointMeasurementSelection.hpp / .cpp`
   看 selection 的接受逻辑。

7. `Stage5Benchmark.hpp / .cpp`
   看 training/holdout dataset 如何构造，评估口径如何定义。

8. `CalibrationStateBundle.cpp`
   看前端 bundle 如何变成 backend problem input。

9. `AslamBackendCalibrationRunner.cpp`
   看 backend 的 delayed release 和最终联合优化。

## 28. 一句话总结

当前 baseline 的本质，不是一个单独的小技巧，而是一条完整、分层清晰的 calibration pipeline：

- 用 outer detection 保证几何入口稳定
- 用 auto-init 摆脱手写 DS 初值
- 用 round1 / round2 internal regeneration 把内部观测逐步做稳
- 用 strict failed-board drop 保证 board observation 的接受语义干净
- 用 Stage5 evaluator 建立 training/holdout benchmark 口径
- 用 ASLAM backend 做最后的联合精修

如果后面要继续提升，这条主线本身不需要推翻，最值得深挖的是：

- internal observation 的质量建模
- 模糊/重影条件下的各向异性误差处理

而不是重新回到手写初值或更多硬门限。

