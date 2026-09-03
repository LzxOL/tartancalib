# Stage6 外参链路更新审计（2026-07-13）

## Kalibr 对照

源码位置：

- `kalibr/aslam_offline_calibration/kalibr/python/kalibr_camera_calibration/CameraIntializers.py`
  - `stereoCalibrate()`：逐同步观测估计 target pose，以相对 pose 的中值生成 baseline seed；联合优化 baseline、每个 view pose 和 projection intrinsics，distortion 默认固定。
- `kalibr/aslam_offline_calibration/kalibr/python/kalibr_camera_calibration/CameraCalibrator.py`
  - `CalibrationTargetOptimizationProblem.fromTargetViewObservations()`：baseline 和 camera intrinsics 属于 calibration group；每个同步 view pose 是 batch nuisance variable。
  - `CameraCalibration.addTargetView()`：调用真实 `IncrementalEstimator.addBatch()`，由边缘 information gain/rank 决定 batch 接受与否。

Kalibr 没有在增量选择完成后额外依赖一次必需的 final BA；accepted batches 本身构成持久优化问题。

## 当前 Stage6 主路径

1. 先按 `exact_timestamp`（正式协议）或 frame-index 规则生成真实 stereo pairs，frontend 只处理这些 paired images；原始/unmatched 数量仍保留在 summary。
2. 用所有有效 shared-board outer pose 生成 `T_cam1_cam0` candidates。
3. 通过 SE(3) consistency 与 medoid 得到稳健 baseline seed。
4. pair-only initialization 联合优化 baseline 与每个 frame-board 的独立局部 pose；它只负责 seed，不负责 selection。
5. 按同步 frame 组织 pair-cohesive batch，同一 frame 的 shared boards 一起进入 trial。
6. persistent `IncrementalEstimator` 对 calibration group 做 nuisance marginalization 后的信息增益/rank判断。
7. 每个 `(synchronized pair, board)` 使用独立 `T_cam0_board` nuisance pose；同一 pair 的多个 board 只作为 cohesive batch 一起提交，不共享未知 layout。
8. accept 时保留 estimator 与 scene state；reject 时恢复 baseline、camera projection 和 independent pair-board poses。
9. 默认跳过 final global BA，最终状态来自 committed incremental problem。

`all_valid`（默认）会使用 Stage6 frontend 生成的全部有效 measurements，但不会继承 Stage5
persistent backend 的 accepted set；`backend_selected_only` 才表示只从两侧
monocular backend-selected observations 构造 stereo dataset。两种协议必须分开报告。

## 本次修复

- pair-only optimizer 不再无条件报告成功；检查 objective finite/decrease、linear solver failure 和 baseline finite。
- `pair_init_use_huber_loss` 现在真实控制初始化鲁棒核。
- candidate 的 before RMSE 在加入 trial batch 前从 committed state 计算。
- persistent rollback snapshot 增加左右 camera projection parameters。
- Stage6 persistent 主路径新增原生 `omni-none` 与 `omni-radtan` 同模型双目支持。
- 新增 `--stage6-intrinsics-mode`：
  - `fixed_stage5`：Stage6 不修改 Stage5 内参，calibration group 只有 stereo baseline。
  - `kalibr_joint_projection`：baseline 与左右 projection intrinsics 同属 calibration group；distortion 保持固定。
  - `regularized_joint_projection`：联合 projection refinement，并加入静态 Stage5 prior。
  - `adaptive_regularized_joint_projection`（默认）：数据充分时使用 regularized joint；数据不足时显式回退 `fixed_stage5`。

## 与 Kalibr 不完全相同的部分

- 我们是多板场景。默认 `independent_pair_board` 将每个 pair-board observation 作为独立 target pose，避免未知 global board layout 吸收外参或内参误差；`shared_frame_layout` 只保留为 legacy ablation。
- 我们在信息增量之前做 shared-board quality 和 pair-cohesion 预组织；最终 accept/reject 仍由真实 `IncrementalEstimator` 决定。
- 默认 `adaptive_regularized_joint_projection` 在数据充分时对应 Kalibr 的联合 projection refinement，并用 Stage5 prior 限制弱可观漂移；`fixed_stage5` 用于隔离内参与外参的显式消融。

## 验证协议

同一 frozen frontend、同一同步 pair、同一 residual 与随机种子比较：

1. `fixed_stage5`
2. `adaptive_regularized_joint_projection`
3. Kalibr reference extrinsic

核心指标：extrinsic-only holdout stereo RMSE、共享 outer/internal RMSE、rotation/translation delta、baseline length delta、information rank、accepted batch 数、jackknife 外参稳定性及运行时间。

## Independent pair-board 状态语义

- persistent problem 中的 nuisance state 是 `(pair_index, board_id) -> T_cam0_board`。
- committed nuisance poses 会写回 `StereoSceneState`，no-final training summary 直接评估 committed state，不再重新拟合 pose。
- holdout 不复用 training nuisance pose，仍从 holdout outer corners 独立拟合 `T_cam0_board`，再用所有 frozen observations 评估 stereo extrinsic。
- verifier 强制检查：
  - `persistent_incremental_pose_structure: independent_pair_board`
  - `persistent_incremental_layout_updates_extrinsic: 0`
  - `final_selected_rmse == training_total_stereo_rmse`

## 2026-07-13 exact-timestamp DS 三折结果

数据为 `stereo_dataset_20260430_1444190-clear`，共 29 个 exact timestamp pairs；`holdout_stride=3`，offset 为 0/1/2。左右内参来自 canonical Stage5 DS 结果。所有模式均使用 pixel residual、persistent incremental estimator、pair-cohesive batch 和 no-final-BA。

| Pose structure | Intrinsics policy | Training RMSE (px) | Holdout RMSE (px) | Baseline std (mm) |
|---|---|---:|---:|---:|
| independent pair-board | fixed Stage5 | 2.206 +/- 0.088 | 2.946 +/- 0.114 | 0.116 |
| independent pair-board | adaptive regularized joint | 0.862 +/- 0.132 | 1.104 +/- 0.374 | 0.110 |
| shared frame-layout (legacy) | fixed Stage5 | 2.527 +/- 0.124 | 2.953 +/- 0.073 | 0.106 |
| shared frame-layout (legacy) | joint projection | 0.916 +/- 0.154 | 1.098 +/- 0.383 | 0.102 |

independent-adaptive 与旧 shared-layout joint 的 holdout 几乎相同，但前者不允许 layout 补偿 camera model / baseline，因此应作为 Kalibr-style 联合双目标定主路径。`fixed_stage5` 保留为严格外参协议，用于证明不修改 Stage5 内参时的 baseline 可重复性；它不能与 joint mode 混在同一指标列中声称是同一任务。

三折 independent-adaptive 的最大 pairwise rotation difference 为 `0.0133 deg`，最大 translation-vector difference 为 `0.808 mm`；independent-fixed 分别为 `0.0206 deg` 和 `0.227 mm`。联合 projection 明显降低像素误差，但会把部分相机模型误差转移到 projection/baseline，因此论文中必须同时报告 intrinsics drift 与外参稳定性。

## 模型路径验证

在同一 3-pair exact-timestamp smoke 上，以下原生 persistent 路径均已通过输出 verifier：

| Model | Training RMSE (px) | Holdout RMSE (px) |
|---|---:|---:|
| DS | 2.665 | 3.096 |
| KB / pinhole-equi | 2.669 | 3.174 |
| EUCM-none | 2.722 | 3.249 |
| Omni-none | 2.636 | 2.863 |

这些数值只用于 runtime/path smoke，不是模型精度排名，因为各模型输入内参并非都来自同一 calibration method。另有高 information threshold 的 DS smoke 验证 independent candidate batch rejection、estimator reject 和 scene rollback。

## 推荐默认与论文协议

工程主路径：

```text
pairing = exact_timestamp
persistent_pose_structure = independent_pair_board
selection = persistent IncrementalEstimator, pair-cohesive batch
final_global_ba = disabled
```

论文应分别报告：

1. **Extrinsic-only**：`fixed_stage5`，冻结左右 Stage5 projection，只估计 baseline 与 independent pair-board nuisance poses。
2. **Joint stereo calibration**：`adaptive_regularized_joint_projection`，数据量充分时释放 projection 并加入静态 Stage5 prior；不足时显式回退 `fixed_stage5`。
3. `shared_frame_layout` 只作为 layout-coupling ablation，不再作为正式结果。
4. `frame_index` pairing 只作异步敏感性诊断；正式结果必须 exact timestamp。

默认 projection prior 使用 shape sigma `0.01`、focal relative sigma `0.01`
和 principal-point sigma `5 px`。三序列实验中，它相对旧的
`0.03 / 0.03 / 20 px` 设置将左右焦距的跨序列标准差从约 `13 px` 降至
约 `9 px`，holdout mean 仅从 `0.965 px` 变为 `0.971 px`。

## 多序列与跨序列验证

统一使用 exact timestamp、independent pair-board、adaptive regularized joint、
strong Stage5 prior、pixel residual 和 no-final-BA。

| Training sequence | Evaluation | Train pairs | Holdout pairs | Training RMSE (px) | Holdout RMSE (px) |
|---|---|---:|---:|---:|---:|
| 1444190-clear | stride-3 split | 19 | 10 | 0.965 | 0.831 |
| 144928-clear | stride-3 split | 21 | 11 | 0.685 | 0.987 |
| 134853-clear | stride-3 split | 19 | 10 | 0.873 | 1.094 |
| 1444190-clear | all 144928-clear | 29 | 32 | 0.875 | 1.095 |
| 144928-clear | all 1444190-clear | 32 | 29 | 0.729 | 1.175 |

三组同序列 holdout mean 为 `0.971 +/- 0.132 px`。双向跨序列测试更严格，
但仍保持在 `1.10--1.18 px`。这支持把 adaptive regularized joint 作为工程
默认；`fixed_stage5` 仍是独立的 extrinsic-only ablation，而不是精度最优模式。

当前 `config/stereo_4_2-3-camchain.yaml` 来自另一时间/装配状态，只能用于
reference sensitivity diagnostics，不能把其十几像素 frozen-measurement RMSE
解释为我们优于 Kalibr。正式论文比较需要在同一 rig 状态、同一同步图像上
重新运行 Kalibr stereo calibration。

当前 DS 主路径可用以下脚本复现：

```bash
scripts/run_stage6_current_baseline_ds.sh \
  image/datatset_5_1/stereo_dataset_20260430_1444190-clear \
  result_may/stage6_current_baseline_ds \
  0
```

## 已知运行时工作

Stage6 当前可以复用 outer-detection cache，但仍会针对左右相机重新运行
`FrozenRound2BaselinePipeline` 以重建 internal measurements 和 frontend scene。
Stage5 现有输出没有无损、带 schema/version 的 `CalibrationStateBundle` artifact，
诊断 CSV 也不足以恢复 point type、weight、board measurement 与 scene state。
后续性能改造应先定义并验证 frozen bundle serialization，再让 Stage5 导出、
Stage6 按 intrinsics/config/data fingerprint 加载；不能直接把诊断 CSV 当作优化输入。

数值 baseline 默认不导出 reprojection visualization。该输出在完整序列上可达
数百 MB 到 1.4 GB，必须作为显式诊断选项运行；output verifier 也只有在传入
`--require-visualizations` 时才强制检查图片。

exact-pair frontend prefilter 已避免未配对帧进入 `FrozenRound2BaselinePipeline`。
在 6+6 帧、实际 3 对的 smoke 中，frontend/dataset build 从约 `103.6 s`
下降到 `15.2 s`。`144928-clear` 从原始 134/75 帧缩减为 32/32 paired frames；
剩余开销来自 paired frames 的两遍 internal regeneration。
