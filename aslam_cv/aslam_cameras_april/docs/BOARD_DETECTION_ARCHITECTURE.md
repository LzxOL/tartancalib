# 板检测前端结构说明

本文档描述板检测前端当前已经存在的真实数据流，并约束后续重构的边界。重构的首要不变量是：不改变现有检测顺序、阈值、恢复触发条件、结果字段和诊断输出含义。

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
       -> 根据有效外板建立相机和板姿态初值
  -> MultiBoardInternalMeasurementRegenerator
       -> 几何先验外角点恢复（漏检板）
       -> 图像证据、拓扑、tag likelihood 和姿态一致性检查
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

| 文件 | 当前职责 | 后续拆分方向 |
| --- | --- | --- |
| `MultiScaleOuterTagDetector.hpp/.cpp` | 外层 tag 解码、多尺度候选、外角点 refinement、融合和 camera-aware 恢复 | 保留公共结果类型；将候选几何、亚像素验证、恢复策略逐步移到内部模块 |
| `MultiBoardOuterBootstrap.hpp/.cpp` | 使用外板观测估计初始相机、板姿态和可见板状态 | 保持为 bootstrap 阶段，不承载单板检测算法 |
| `MultiBoardInternalMeasurementRegenerator.hpp/.cpp` | 按帧/按板编排恢复、几何先验候选评估和内角点检测调用 | 拆成 frame orchestration、geometry-prior rescue、result finalization 三层 |
| `ApriltagInternalDetector.hpp/.cpp` | 单板内角点生成及 homography、sphere、virtual-patch 策略 | 将三种内角点生成策略改为内部策略模块，保持当前分支顺序 |
| `BoardDetectionPipeline.hpp/.cpp` | 上层阶段门面：统一“外板检测”和“板测量恢复”入口 | 逐步承接阶段状态和结果汇总，但不复制检测算法 |
| `OuterDetectionResultUtils.hpp` | 构造缺失板的统一失败结果 | 继续集中结果归一化的小型无状态工具 |

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
| `FrozenRound2BaselinePipeline.cpp` | Stage5 前端完整编排、缓存、bootstrap、两轮 regeneration 和后端衔接 | 继续减少算法细节，只保留阶段调度和 runtime 汇总 |
| `Stage5BackendDiagnosticWriters.cpp` / `run_stage5_backend_main.cpp` | 诊断文件、CSV、overlay 和命令行流程 | 后续把 artifact writer 从主程序继续下沉 |

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
- 没有改变任何检测阈值、恢复条件、算法顺序或结果字段。

## 后续小步重构顺序

1. 在 `MultiBoardInternalMeasurementRegenerator.cpp` 内先提取不改变签名的 frame context 和 per-board finalization 函数。
2. 将 geometry-prior outer rescue 的纯候选评估移动到独立内部模块；恢复器只负责调用、候选排序和提交策略。
3. 将 `ApriltagInternalDetector.cpp` 中的内角点生成分支分别移到 homography、sphere-lattice、virtual-patch 内部实现文件。
4. 将 `MultiScaleOuterTagDetector.cpp` 中的恢复算法和 visualization 分离。
5. 最后整理 `run_stage5_backend_main.cpp` 的诊断写出函数，并保持 CSV 字段和已有产物路径兼容。

每一步都必须通过：编译、frame61（亚像素回退保护）、frame70（顶端板恢复）以及完整 109 帧数据集回归后才能继续下一步。
