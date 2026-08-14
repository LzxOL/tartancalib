# 板检测前端结构说明

本文档描述板检测前端当前已经存在的真实数据流，并约束后续重构的边界。重构的首要不变量是：不改变现有检测顺序、阈值、恢复触发条件、结果字段和诊断输出含义。

本套完整 board/tag 补救、恢复和几何优化逻辑的统一名称是
**Frozen Stage5 Board Rescue Recovery Stack（FS5-BRRS，Stage5 冻结式
Board Rescue Recovery Stack）**。后续提到 `FS5-BRRS`、`Stage5 frozen board
rescue` 或 `baseline board rescue recovery`，均指本文档列出的完整处理栈，
而不是某一个单独的 detector 分支。

## 当前数据流

```text
图像输入
  -> MultiScaleOuterTagDetector
       -> 多尺度 AprilTag 解码
       -> 外角点几何/边界有效性判断
       -> 球面外角点 refinement 与 cornerSubPix
       -> 多尺度角点融合
       -> camera-aware sphere-patch / zero-detection / geometry rescue
       -> OuterTagDetectionResult
  -> MultiBoardOuterBootstrap
       -> 根据有效外板建立 provisional 相机和板姿态初值
  -> CameraAwareOuterRescue
       -> provisional-camera patch recovery
       -> direct-layout geometry gate
       -> deterministic frame-order commit and rescue statistics
  -> MultiBoardInternalMeasurementRegenerator
       -> 几何先验外角点恢复模块（漏检板）
       -> 姿态先验选择、候选排序和结果提交
       -> ApriltagInternalDetector
            -> homography / pinhole bootstrap patch
            -> sphere lattice / sphere ray refine
            -> virtual patch / image subpixel / boundary seed
            -> 内角点拓扑、重复点和质量统一过滤
       -> InternalRegenerationFrameResult
  -> JointMeasurementBuilder / Stage5 backend
       -> 最终观测筛选、优化和诊断输出
```

## 文件职责

| 文件 | 当前职责 | 维护边界 |
| --- | --- | --- |
| `MultiScaleOuterTagDetector.hpp/.cpp` | 外层 tag 解码、多尺度候选、外角点 refinement、融合和 camera-aware 恢复 | 保留外层 detector 的图像级处理；Stage5 的 provisional-camera rescue 不回灌到此处 |
| `MultiBoardOuterBootstrap.hpp/.cpp` | 使用外板观测估计初始相机、板姿态和可见板状态 | 保持为 bootstrap 阶段，不承载单板检测算法 |
| `CameraAwareOuterRescue.hpp/.cpp` | Stage5 provisional camera 之后的 camera-aware patch rescue、zero-detection atlas、direct-layout 几何门控和确定性合并 | 保持为独立的外板恢复模块；不承载内点生成 |
| `GeometryPriorTopology.hpp/.cpp` | 四边形拓扑有效性、面积/中心/角点顺序归一化，以及 wrong-ID proposal 与几何先验的关联 | 只提供几何关联工具，不直接决定最终观测是否进入标定 |
| `GeometryPriorOuterRecovery.hpp/.cpp` | 几何先验外角投影、ROI/ray-patch 图像证据、corner refinement、tag likelihood、姿态一致性和候选最终评估 | 保持为无状态候选评估模块，不负责帧级调度和候选选择 |
| `MultiBoardInternalMeasurementRegenerator.hpp/.cpp` | 按帧/按板编排姿态先验、候选排序提交和内角点检测调用 | 继续保持 frame orchestration 与 result finalization；不重新实现几何恢复细节 |
| `ApriltagInternalDetector.hpp/.cpp` | 单板内角点生成及 homography、sphere、virtual-patch 策略 | 保持单板内点生成策略及其原有分支顺序；统一结果校验已下沉到 `InternalPointResultValidation` |
| `InternalPointResultValidation.hpp/.cpp` | 内点结果的拓扑归属、重复点抑制、越界修正和计数重算 | 作为所有内点生成策略之后的统一结果校验层 |
| `BoardDetectionPipeline.hpp/.cpp` | 上层阶段门面：统一“外板检测”和“板测量恢复”入口 | 逐步承接阶段状态和结果汇总，但不复制检测算法 |
| `OuterDetectionResultUtils.hpp` | 构造缺失板的统一失败结果 | 继续集中结果归一化的小型无状态工具 |
| `FrozenRound2BaselinePipeline.cpp` | Stage5 前端完整编排、缓存、bootstrap、两轮 regeneration 和后端衔接 | 只保留阶段调度和 runtime 汇总 |
| `Stage5BackendDiagnosticWriters.cpp` / `run_stage5_backend_main.cpp` | 诊断文件、CSV、overlay 和命令行流程 | 保持产物字段和路径兼容 |

## 外板恢复机制与边界

`BoardDetectionPipeline` 只负责组织调用；以下机制仍由其原有实现负责，
不能把“缺失结果占位”误认为恢复算法。

1. **自适应多尺度解码**：先在低分辨率尺度搜索，必要时回退到更高
   分辨率尺度；它是第一层外板恢复，适用于标签大小变化。
2. **Stage5 camera-aware sphere-patch rescue**：完成 provisional DS 相机
   初始化后，对缺失的已知 tag ID 做球面 patch 重解码，并只提交
   exact-ID / Hamming-0 的结果。对整帧无直接解码的情况，
   `camera_aware_sphere_patch_rescue_zero_detection_frames` 默认开启，确保
   不会因“零初检”而跳过恢复。
3. **扩展 atlas**：`camera_aware_sphere_patch_use_extended_atlas` 是更昂贵的
   数据集级加强项，增加边界和宽视场 patch；本 four-board 数据集显式开启。
4. **外四角 refinement 保护**：close-edge window boost、cornerSubPix 和
   unstable-rollback guard 改善或拒绝已解码标签的外角；它们不能凭空解码
   一个缺失 tag。
5. **geometry-prior / internal pose rescue**：依赖 bootstrap 或已有可见板，
   用于恢复姿态、外框种子和内点；它们不替代外层 exact-ID 解码，也不能
   恢复整帧零外板。

`enable_anonymous_tag_like_geometry_rescue` 保持关闭：该策略缺少稳定 ID
约束，在多板 rig 中可能把板身份分配错误，不能作为标定默认恢复路径。

## 当前内点与姿态恢复机制

这些机制仍然由 `ApriltagInternalDetector` 按原有分支顺序执行，统一结果
校验由 `InternalPointResultValidation` 收口：

1. 外四角或姿态先验提供 homography / pinhole bootstrap seed；
2. DS sphere-lattice、sphere-ray refine、border-conditioned seed 和
   virtual-patch seed，按 projection mode 选择；
3. 当标准外四角不足以稳定求姿态时，使用 internal pose rescue，并由射线角度、
   外角 RMSE 和姿态质量条件共同限制；
4. `force_internal_seed_from_prediction`、结构修正和 image-evidence/subpixel
   refinement 只负责生成或修正内点，不会创建缺失的外板；
5. 所有策略完成后统一执行拓扑归属、重复 refined corner 抑制、图像边界检查和
   corner count 重算；内点失败时是否保留已经确认的外四角由 measurement builder
   的既有策略决定。

因此 recovery 模块负责“候选恢复和姿态先验”，detector 负责“单板内点数值
生成”，validation 模块负责“结果一致性”；三者不会互相复制接受条件。

## 结果层次

1. `OuterTagDetectionResult`：一个请求板的外角点检测结果，包含 coarse/refined corner、角点有效 mask、failure reason 和恢复诊断。
2. `OuterTagMultiDetectionResult`：一帧所有请求板的外角点结果及 `OuterFrameMeasurementResult`。
3. `ApriltagInternalDetectionResult`：一个板的内角点结果，包含内角点有效性、质量、运行时统计和失败原因。
4. `InternalRegenerationFrameResult`：一帧所有板的最终板级测量、几何先验候选、warning 和 runtime breakdown。

缺失板必须仍然以带有 `board_id` 和 `NoDetectionsAtAll` 的结果占位，不能通过缩短数组或丢弃请求 ID 来表达漏检；这一约束现在由 `MakeMissingOuterTagDetection()` 统一维护。

## 已完成的低风险重构

- 新增 `BoardDetectionPipeline`，将上层调用明确为外层检测阶段和板测量恢复阶段。
- Stage5 冻结流程、multi-board regeneration 工具和 joint measurement prep 工具已使用该阶段门面。
- 重复的缺失板结果构造已集中到 `OuterDetectionResultUtils.hpp`。
- 将 Stage5 camera-aware outer rescue 从 `FrozenRound2BaselinePipeline.cpp` 移到
  `CameraAwareOuterRescue`，保留原有 patch、direct-layout gate、并行 worker
  和确定性合并顺序。
- 将 geometry-prior 使用的四边形拓扑检查和 wrong-ID proposal 关联移到
  `GeometryPriorTopology`，保留原有相对阈值和调用时机。
- 将 geometry-prior 外角恢复的投影、局部图像证据、亚像素、tag likelihood、
  姿态一致性和候选评估移到 `GeometryPriorOuterRecovery`，主编排文件只保留
  预测来源、候选排序和最终提交。
- 将内点结果的统一拓扑/重复点校验移到 `InternalPointResultValidation`，
  homography、sphere 和 virtual-patch 的生成分支及其顺序保持不变。
- 没有改变任何检测阈值、恢复条件、算法顺序或结果字段。

## 当前重构边界与验证要求

本轮结构整理已经完成到“阶段编排 / 外板恢复 / 几何先验候选 / 内点结果校验”四个职责层：

1. `FrozenRound2BaselinePipeline` 只编排 Stage5 阶段和诊断输出；camera-aware
   外板恢复由 `CameraAwareOuterRescue` 承担。
2. `MultiBoardInternalMeasurementRegenerator` 只负责编排帧级姿态先验、候选
   排序、板级提交和内点 detector 调用；几何先验外角恢复由
   `GeometryPriorOuterRecovery` 承担。
3. `ApriltagInternalDetector` 保留 homography、pinhole bootstrap、sphere
   lattice 和 virtual-patch 的数值生成分支。它们共享大量相同的图像证据、DS
   投影、边界模型和运行时状态；继续拆成多个文件会扩大接口面并提高行为漂移
   风险，因此本轮不做机械拆分。
4. 所有内点生成分支完成后统一经过 `InternalPointResultValidation`，避免每个
   策略重复维护拓扑归属、重复点抑制、越界处理和计数重算。

后续任何算法变更都必须保留当前阶段顺序、阈值、结果字段和诊断产物，并至少
通过编译、frame61（亚像素回退保护）、frame70（顶端板恢复）以及完整 109 帧
数据集回归。
