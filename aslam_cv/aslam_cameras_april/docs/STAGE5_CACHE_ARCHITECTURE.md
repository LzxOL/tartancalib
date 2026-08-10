# Stage5 分层缓存约定

Stage5 的缓存必须以“结果的真实依赖关系”为边界，而不能以一次完整
命令为边界。这样只修改内点、姿态或后端逻辑时，外角检测结果仍可安全
复用；反过来，依赖相机参数的恢复结果也不会错误地用于新的相机状态。

## 当前已实现：v1 外角最终结果和内点重生成阶段

`OuterDetectionCache` 现在优先读写：

```text
<cache-dir>/stage5_cache_layout_v1/
  outer_detection_final/<semantic-config-hash>/
    manifest.yaml
    <image-identity-hash>.yml
```

原来的 `<cache-dir>/<config-hash>/<image-hash>.yml` 保留为只读兼容路径。
因此已有实验不用迁移，也不会被新代码覆盖。每个 `manifest.yaml` 记录：

- cache layout、阶段实现版本和 artifact schema；
- 该阶段的语义配置 hash；
- 上游 artifact hash；
- 用于人工核对的完整语义配置描述。

写入使用临时文件后重命名；若同一目录已有不同 manifest，缓存写入会
失败并保留旧文件，绝不覆盖。

在同一个 cache 根目录下，内点重生成结果现在写入：

```text
<cache-dir>/stage5_cache_layout_v1/
  internal_refinement/<internal-config-hash>/
    manifest.yaml
    <image-outer-result-state-hash>.yml
```

该阶段保存每帧各 board 的内点坐标、有效性、质量、内点调试信息、边界模型
和姿态恢复摘要。每个文件同时记录当前外角结果签名和 bootstrap/scene 状态
签名，因此不会把旧姿态下的内点错误复用到新姿态。内点算法参数变化只会
产生新的 `internal-config-hash` 子层；外角 cache 保持不变，旧的内点子层也
保留以便回溯。旧内点文件缺少新增运行计数时，读取端会根据保存的点集合
兼容重建计数。

## 一个数据集对应一个 cache 根目录

每个 cache 根目录首次使用时会创建 `dataset_manifest.yaml`。它绑定缓存到
一个绝对图像目录；后续若有不同图像目录尝试使用同一 cache，读取和写入
都会被拒绝，避免实验间静默混用。对于已有 legacy cache，首次绑定前会扫描
其中所有 `absolute_image_path`；只有全部属于同一图像目录时才允许接管。

因此，同一数据集的不同算法、不同阈值和不同 backend 实验应始终传入同一个
`--cache-dir`。算法差异通过根目录内部的 stage/config hash 区分，不需要再
为每个实验创建新的 cache 目录。

## 为什么尚未把当前外角结果直接拆为 raw/refine/rescue 三份

现有 `MultiScaleOuterTagDetector` 在一次 detector 调用中完成多尺度解码、
相机相关球面 refinement 和 camera-aware sphere-patch rescue。最终缓存
中的角点可能已经受相机模型影响。若只因为“decoder 参数没有变”而拿它
复用到新相机，会把旧相机下的角点当作新结果，破坏标定正确性。

因此，v1 先将最终外角结果隔离为一个独立阶段，并保留完全匹配的 hash
条件。下一步应先提取可序列化的 raw multi-scale decoder candidates，再
建立以下安全依赖图：

```text
image identity
  -> outer_decode
  -> outer_refinement
  -> outer_rescue (camera / scene dependent)
  -> internal_seed
  -> internal_refinement (implemented)
  -> pose_recovery
  -> frontend_measurements
  -> backend_input
  -> backend_optimization
```

## 失效规则

| 修改 | 可复用阶段 | 需要重新计算的阶段 |
| --- | --- | --- |
| 可视化、CSV 或诊断输出 | 所有计算结果 | diagnostics |
| 后端残差、鲁棒核、迭代次数 | frontend measurements | backend input / optimization |
| 内点 seed 或 subpixel 参数 | outer stages | internal 及下游 |
| 外角 refinement 参数 | outer_decode | outer refinement 及下游 |
| 相机模型或相机内参 | outer_decode | 相机相关 refinement/rescue 及下游 |
| AprilTag 解码/多尺度策略或图像变化 | image metadata | outer decode 及下游 |
| 全流程接口或序列化格式重大变化 | 无 | 提升 layout epoch 后重建全部 |

`implementation_version` 是阶段级版本，不是全局版本。正常算法修改只需
提升受影响阶段的版本；只有结果语义或 artifact 格式整体不兼容时，才提升
`stage5_cache_layout_v1` 的 layout epoch。
