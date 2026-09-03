# Stage5 单目内参 Baseline 代码阅读报告

## 1. Baseline 总体流程

基于代码审计，Stage5 baseline 的真实执行顺序如下：

```
outer detection (多尺度 outer corner detection)
    ↓
outer-only DS auto-init (DS candidate grid 枚举 + pose fit ranking)
    ↓
multi-board bootstrap (T_reference_board, T_camera_reference, reference board gauge fixed)
    ↓
Round1 internal regeneration (基于 bootstrap 粗相机状态)
    ↓
Round1 joint measurement build (outer + internal observations 合并)
    ↓
Round1 residual evaluation
    ↓
Round1 selection (residual sanity gate)
    ↓
Round1 backend optimization (delayed intrinsics release)
    ↓
[Round2 if enabled]
    ↓
Round2 internal regeneration (基于 Round1 optimized 相机状态)
    ↓
Round2 measurement rebuild (重新构建 joint measurements)
    ↓
Round2 residual evaluation
    ↓
Round2 selection
    ↓
Round2 backend optimization (second_pass intrinsics_release_iteration = 1)
    ↓
final evaluation / benchmark / holdout
```

**关键差异确认**：
- 实际代码中，Round1 和 Round2 都会独立执行 internal regeneration，而不是仅 Round2 重新生成
- Round2 的 internal regeneration 使用 Round1 的 `optimized_state`（更精确的相机参数）
- Round2 使用 `optimized_state` 中的 frame pose 和 board pose 作为 prior
- Final bundle 使用 Round2 结果（如果 Round2 可用且 non-degrading）

---

## 2. 主入口和关键配置

文件：`run_stage5_backend_main.cpp`

### 2.1 关键 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--camera-init-mode` | `auto` | `manual`, `auto`, `auto_with_manual_fallback` |
| `--disable-second-pass` | `false` | 是否跳过 Round2 |
| `--intrinsics-release-mode` | `delayed` | `delayed`, `immediate`, `pose_only` |
| `--intrinsics-release-iteration` | `3` | Round1 intrinsics release 迭代次数 |
| `--second-pass-intrinsics-release-iteration` | `1` | Round2 intrinsics release 迭代次数 |
| `--strict-board-observation-acceptance` | `false` | 是否严格要求 board observation 完整性 |
| `--backend-internal-anisotropic-weight-mode` | `off` | 各向异性权重 |
| `--backend-observation-role-weight-mode` | `balanced` | `balanced`, `outer_priority` |
| `--internal-blur-filter-mode` | `off` | blur 过滤 |
| `--internal-blur-board-weight-mode` | `off` | blur board 权重 |
| `--internal-observation-weight-mode` | `off` | observation quality 权重 |
| `--internal-joint-refine-mode` | `off` | joint refine |
| `--pre-backend-filter-mode` | `off` | pre-backend 过滤 |
| `--stage5-enable-polar-angle-diagnostics` | `false` | polar diagnostics 开关 |
| `--stage5-polar-angle-bin-edges` | `0,30,50,70,85,100` | polar angle 分桶边界 |

### 2.2 baseline 配置结构体

```cpp
// RequestedExperimentConfig 中的 baseline 选项：
struct RequestedExperimentConfig {
  CameraInitializationMode camera_init_mode = Auto;
  bool run_second_pass = true;
  IntrinsicsReleaseMode frontend_intrinsics_release_mode = Delayed;
  IntrinsicsReleaseMode backend_intrinsics_release_mode = Delayed;
  bool backend_delayed_intrinsics_release = true;
  int backend_intrinsics_release_iteration = 1;
  // ...
  InternalObservationWeightMode internal_observation_weight_mode = Off;
  InternalBlurBoardWeightMode internal_blur_board_weight_mode = Off;
};
```

### 2.3 输出文件

- `stage5_backend_result.txt` - 主结果文件
- `stage5_round1_bundle.json` / `stage5_round2_bundle.json` - calibration state bundle
- polar diagnostics 输出（当启用时）：
  - `polar_angle_residual_summary.txt`
  - `polar_angle_residual_bins.csv`

---

## 3. Auto-init 代码理解

文件：`OuterOnlyCameraInitializer.cpp` + `OuterOnlyCameraInitializer.hpp`

### 3.1 DS Candidate Grid 构造

```cpp
// focal_scale_candidates: {0.18, 0.22, 0.26, 0.30, 0.34, 0.40, 0.50, 0.60}
// xi_candidates: {-0.4, -0.2, 0.0, 0.2, 0.5, 1.0}
// alpha_candidates: {0.35, 0.45, 0.55, 0.65, 0.75}
```

对于 DS 模型，grid 大小为：`len(focal_scale) × len(xi) × len(alpha)` = `8 × 6 × 5 = 240` 个 candidates。

主点初值固定为图像中心：
```cpp
center_u = 0.5 * image_width;
center_v = 0.5 * image_height;
```

### 3.2 Pose Fit 与 Ranking

```cpp
// 每个 candidate 的评估：
for (const OuterObservationRecord& observation : observations) {
  if (!EstimatePoseFromObjectPoints(camera, object_points, image_points, &pose, &rmse)) {
    ++candidate.pose_failure_count;
    continue;
  }
  ++candidate.pose_success_count;
  total_squared_rmse += rmse * rmse;
}
```

Ranking 顺序（`CandidateIsBetter`）：
1. `valid` (是否有成功的 pose fit)
2. `pose_success_count` (成功率)
3. `success_rate` (成功率)
4. `mean_observation_rmse` (RMSE)
5. `successful_frame_count`
6. `successful_board_count`

### 3.3 Fallback 机制

```cpp
// IsAcceptableAutoCandidate 接受条件：
min_success = min(total_observation_count, 6);
min_success_rate = (total_observation_count >= 12) ? 0.4 : 0.25;
return candidate.valid &&
       candidate.pose_success_count >= min_success &&
       candidate.success_rate >= min_success_rate &&
       candidate.mean_observation_rmse < 20.0;
```

如果 auto 失败，自动降级到 manual generic seed 或 configured intermediate camera。

### 3.4 输出结果

`AutoCameraInitializationResult` 包含：
- `selected_camera` - 选中的相机参数
- `candidates` - 所有候选及其评估结果
- `initialization_rmse` - 初始化 RMSE
- `accepted_frame_count`, `accepted_board_observation_count`

该结果传递给 bootstrap 初始化器。

---

## 4. Multi-board Bootstrap 代码理解

文件：`MultiBoardOuterBootstrap.cpp` + `MultiBoardOuterBootstrap.hpp`

### 4.1 Reference Board 定义

```cpp
struct OuterBootstrapOptions {
  int reference_board_id = 1;  // 默认为 board 1
  // ...
};
```

Reference board 的 `T_reference_board = Identity()`，作为整个多板系统的 gauge。

### 4.2 Transform 方向确认

从代码中确认 transform 方向：

```cpp
// ComputeObservationRmse 中的关键计算：
const Eigen::Vector3d point_camera =
    T_camera_reference * (T_reference_board * board_points[index]);
```

这意味着：
- `board_points` 是 board 坐标系中的点
- `T_reference_board` 将 board 点变换到 reference board 坐标系
- `T_camera_reference` 将 reference board 坐标系中的点变换到相机坐标系

### 4.3 Board Pose 初始化

```cpp
// InitializeConnectedComponent 中：
reference_it->second.T_reference_board = Eigen::Isometry3d::Identity();
reference_it->second.initialized = true;

// 非 reference board 通过已初始化的 frame 估计：
Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
if (EstimateSingleObservationBoardPoseInCamera(state, intrinsics, observation,
                                               &T_camera_board, &observation_rmse)) {
  candidate.transform = frame.T_camera_reference.inverse() * T_camera_board;
  // 即 T_reference_board = T_camera_reference^(-1) * T_camera_board
}
```

### 4.4 输出给 Internal Regeneration 的状态

```cpp
struct OuterBootstrapResult {
  OuterBootstrapCameraIntrinsics coarse_camera;
  std::vector<OuterBootstrapBoardState> boards;  // 包含 T_reference_board
  std::vector<OuterBootstrapFrameState> frames;  // 包含 T_camera_reference
};
```

---

## 5. Internal Regeneration 代码理解

文件：`MultiBoardInternalMeasurementRegenerator.cpp` + `ApriltagInternalDetector.cpp`

### 5.1 实际使用的 Projection Mode

```cpp
// ApriltagInternalDetector.cpp 中的 InternalProjectionMode：
if (lowered == "sphere_lattice" || lowered == "sphere-lattice") {
  return InternalProjectionMode::SphereLattice;
}
if (lowered == "sphere_border_lattice" || lowered == "sphere-border-lattice") {
  return InternalProjectionMode::SphereBorderLattice;
}
if (lowered == "sphere_ray_refine" || lowered == "sphere-ray-refine") {
  return InternalProjectionMode::SphereRayRefine;
}
```

实际代码中使用了 `SphereBorderLattice` 作为主要模式（从 config 中的 `internal_projection_mode` 读取）。

### 5.2 Sphere-Border-Lattice 逻辑

核心数据结构：

```cpp
struct BoardSphereBoundaryModel {
  bool valid = false;
  std::array<Eigen::Vector3d, 4> outer_corner_rays{};  // 4 个 outer corner 的球面 ray
  std::array<BoardBoundaryEdgeModel, 4> edges;         // 4 条边的边界模型
};

struct SphereLatticeFrame {
  Eigen::Vector3d predicted_ray = Eigen::Vector3d::Zero();  // 预测的 ray
  Eigen::Vector3d tangent_u = Eigen::Vector3d::Zero();     // u 方向切线
  Eigen::Vector3d tangent_v = Eigen::Vector3d::Zero();     // v 方向切线
  // ...
};

struct RayRefinementEvaluation {
  Eigen::Vector3d ray = Eigen::Vector3d::Zero();  // 精化的 ray
  double template_quality = 0.0;
  double gradient_quality = 0.0;
  double final_quality = 0.0;
};
```

### 5.3 Round1 vs Round2 差异

```cpp
// FrozenRound2BaselinePipeline.cpp

// Round1：使用 bootstrap_result
result.round1.regeneration_results.push_back(
    regenerator.RegenerateFrame(image, regeneration_inputs[frame_index],
                                result.bootstrap_result));

// Round2：使用 Round1 的 optimized_state
result.round2.regeneration_results.push_back(
    regenerator.RegenerateFrame(
        image, regeneration_inputs[frame_index],
        result.round1.optimization_result.optimized_state));
```

关键差异：
- Round1 使用 `bootstrap_result.coarse_camera` 作为相机状态
- Round2 使用 `optimization_result.optimized_state.camera` 作为相机状态
- Round2 的 pose prior 更精确

### 5.4 Point Validity 判断

从 `ApriltagInternalDetector` 输出：
- `valid_internal_corner_count` - 有效 internal corner 数量
- 每个 point 有 `quality` 字段 (0.0 ~ 1.0)
- `final_quality` = template × gradient × edge 的组合

---

## 6. Joint Measurement / Residual / Backend

文件：
- `JointReprojectionMeasurementBuilder.cpp`
- `JointReprojectionCostCore.cpp`
- `JointReprojectionOptimizer.cpp`

### 6.1 Outer/Internal 合并

```cpp
struct JointMeasurementBuildResult {
  std::vector<JointMeasurementFrameResult> frames;
  std::vector<JointPointObservation> solver_observations;
  int used_outer_point_count = 0;
  int used_internal_point_count = 0;
  int used_total_point_count = 0;
  // ...
};

struct JointPointObservation {
  JointPointType point_type;  // Outer or Internal
  Eigen::Vector2d image_xy;
  Eigen::Vector3d target_xyz_board;
  double quality = 0.0;  // internal point 的质量
  // ...
};
```

### 6.2 Residual 计算

```cpp
// JointReprojectionCostCore::Evaluate 中的核心循环：
for (const JointPointObservation& obs : measurement_result.solver_observations) {
  // 1. 构建 T_camera_board
  const Eigen::Isometry3d T_camera_board =
      T_camera_reference * T_reference_board;

  // 2. 将 board 点变换到相机坐标系
  const Eigen::Vector3d point_camera = T_camera_board * obs.target_xyz_board;

  // 3. 投影到图像平面
  if (!camera.vsEuclideanToKeypoint(point_camera, &projected)) {
    // invalid projection penalty
  }

  // 4. 计算 residual
  residual = projected - obs.image_xy;
}
```

### 6.3 当前已有的 Weighting 机制

**1. Quality Weight (Observation Quality Weighting)**

```cpp
// JointReprojectionMeasurementBuilder.cpp
double ComputeInternalObservationQualityWeight(
    double quality,
    double quality_threshold,
    const JointMeasurementBuildOptions& options) {
  if (!options.enable_internal_observation_quality_weighting) {
    return 1.0;
  }
  // 根据 quality 计算 weight，范围 [min_weight, 1.0]
  return min_weight + (1.0 - min_weight) * pow(bounded_quality, exponent);
}
```

默认 `enable_internal_observation_quality_weighting = false`（baseline 关闭）。

**2. Role Weight (Outer/Internal Budget)**

```cpp
// backend_observation_role_weight_mode: "balanced", "outer_priority"
// backend_internal_role_budget_when_mixed: 0.5
```

**3. Anisotropic Weight**

```cpp
// backend_internal_anisotropic_weight_mode: "off", "fixed_xy_scale"
// backend_internal_anisotropic_x_scale, y_scale
```

**4. Huber Loss**

```cpp
// JointReprojectionCostOptions
double outer_huber_delta_pixels = 10.0;
double internal_huber_delta_pixels = 6.0;
```

### 6.4 Delayed Intrinsics Release 实现

```cpp
// JointReprojectionOptimizer.cpp
struct JointOptimizationOptions {
  int intrinsics_release_iteration = 1;  // 在第 N 次迭代后释放 intrinsics
  // ...
};

// 在 optimize 过程中：
// - 第 1 ~ (intrinsics_release_iteration - 1) 次迭代：intrinsics fixed
// - 第 intrinsics_release_iteration 次及之后：intrinsics released
```

---

## 7. Phase 0 审计项复核

| Feature | Status | Evidence Files | Notes |
|---------|--------|----------------|-------|
| Polar angle diagnostics | **exists** | `PolarAngleResidualDiagnostics.hpp/cpp` | 基础设施完整，默认关闭 |
| Polar-angle adaptive weighting | **NOT implemented** | - | 只有 diagnostics，无 backend weighting |
| Multi-board consistency diagnostics | **NOT implemented** | - | 无 board 间一致性统计 |
| Board-level consistency weighting | **NOT implemented** | - | 无 board 级别 consistency weight |
| Soft board rigidity prior | **NOT implemented** | - | 无 SE3 prior 或 rigidity prior |
| Round1/Round2 pipeline | **exists** | `FrozenRound2BaselinePipeline.cpp` | Round1/Round2 都完整实现 |
| Round1 delayed intrinsics release | **YES** | `intrinsics_release_iteration = 3` | 前 3 次迭代 intrinsics fixed |
| Round2 internal regeneration | **YES** | `regenerator.RegenerateFrame(..., result.round1.optimization_result.optimized_state)` | 使用 Round1 结果重新生成 |
| Round2 measurement rebuild | **YES** | `builder.Build(result.round2.joint_inputs, ...)` | 重新构建 joint measurements |
| Round2 selection | **YES** | `selector.Select(...)` | 重新 selection |
| Round2 optimization | **YES** | `round2_optimizer.Optimize(...)` | 重新优化 |

---

## 8. Phase 1 相关代码复核

### 8.1 Polar Diagnostics 启用方式

```cpp
// run_stage5_backend_main.cpp
ati::PolarAngleDiagnosticsOptions polar_options;
polar_options.bin_edges_deg = ParsePolarAngleBinEdges(args.polar_angle_bin_edges);
ati::PolarAngleResidualDiagnostics polar_diagnostics(polar_options);
const ati::PolarAngleDiagnosticsResult polar_result =
    polar_diagnostics.EvaluateWithResiduals(
        report.backend_problem_input.measurement_dataset,
        report.backend_problem_input.residual_result,
        report.backend_problem_input.optimized_scene_state,
        args.output_path);
```

### 8.2 默认状态

```cpp
// CmdArgs 中的默认值：
bool enable_polar_angle_diagnostics = false;
std::string polar_angle_bin_edges = "0,30,50,70,85,100";
```

### 8.3 确认只读不改变 backend

`PolarAngleResidualDiagnostics::EvaluateWithResiduals` 的实现：
1. 接收 `measurement_dataset`, `residual_result`, `scene_state` 作为输入
2. 仅读取数据，输出 `PolarAngleDiagnosticsResult`
3. **不修改** 输入数据
4. 输出文件：`polar_angle_residual_summary.txt`, `polar_angle_residual_bins.csv`

### 8.4 输出字段

```cpp
struct PolarAngleBinStatistics {
  double bin_min_deg, bin_max_deg;
  std::string point_type;  // "all", "outer", "internal"
  int point_count;
  double rmse;
  double mean_abs_x, mean_abs_y;
  double std_x, std_y;
  double median_residual, p90_residual, p95_residual, max_residual;
};
```

### 8.5 分桶输出

- `all_points_bins` - 所有点
- `outer_only_bins` - 仅 outer 点
- `internal_only_bins` - 仅 internal 点
- `per_board_bins` - 每个 board 的分桶
- `per_frame_bins` - 每个 frame 的分桶

---

## 9. 风险点和需要注意的地方

### 9.1 Transform 方向

**风险**：代码中使用 `T_camera_reference * T_reference_board` 的顺序，必须确保后续使用时一致。

**确认点**：
- `ComputeObservationRmse`: `T_camera_reference * (T_reference_board * point)`
- `MultiBoardInternalMeasurementRegenerator`: `ComposeCameraBoardTransform(T_camera_reference, T_reference_board)`
- `JointReprojectionCostCore`: 同样的 transform 顺序

**结论**：当前代码内部一致，但外部使用（如 benchmark）需确保与此处一致。

### 9.2 Reference Board Gauge Fixed

**风险**：将 board1 固定为 Identity 可能导致：
- 如果 board1 的观测质量差，会系统性影响所有 board pose
- 无法检测 board1 自身的 scale 或 orientation 偏差

**现状**：无其他机制检测此问题。

### 9.3 Round2 Measurement Rebuild

**确认**：Round2 使用 `result.round1.optimization_result.optimized_state` 重新执行：
1. Internal regeneration
2. Joint measurement build
3. Selection
4. Optimization

这意味着 Round2 不是简单的 re-optimize，而是完整的数据重建流程。

### 9.4 Polar Diagnostics 使用 optimized residual

**潜在风险**：`PolarAngleResidualDiagnostics` 使用的是 **optimized** 后的 residual。

如果用于 pre-backend weighting，需要：
1. 使用 pre-optimization 的 residual
2. 或者使用 training/holdout 外参验证的 reprojection error

### 9.5 Outer/Internal Role Weight 混合

```cpp
// backend_observation_role_weight_mode = "balanced"
// backend_internal_role_budget_when_mixed = 0.5
```

这意味着当同时存在 outer 和 internal points 时，internal points 的权重 budget 是 0.5（相对 outer）。需确认这是否与 polar-angle weighting 的目标一致。

### 9.6 Quality Weight 默认关闭

```cpp
// FrozenRound2BaselinePipeline.hpp
bool enable_internal_observation_quality_weighting = false;
```

Baseline 默认不启用 internal observation quality weighting。

---

## 10. 后续 Phase 2 前置建议

### 10.1 Polar-angle Adaptive Weighting 接入点

**最适合接入的文件**：`JointReprojectionMeasurementBuilder.cpp`

原因：
1. 已有 `ComputeInternalObservationQualityWeight` 的类比实现
2. 可以在 `ApplyInternalObservationQualityWeightsToResult` 附近增加 polar-angle weight
3. 与现有的 quality weight 形成统一的 weight 计算框架

**备选接入点**：`JointReprojectionCostCore.cpp`
- 优点：权重在 cost evaluation 时直接应用
- 缺点：需要修改 core evaluation 逻辑

### 10.2 Sigma_bin 估计来源

**建议使用 pre-optimization residual 统计**：
- 使用 Round1/Round2 selection 后的 residual
- 不使用 optimized residual（会低估 sigma）
- 或者使用 training/holdout 外参验证的 reprojection error

**具体实施**：
1. 在 `JointMeasurementSelection` 后收集 residual
2. 按 polar angle bin 分桶计算 RMSE 或 MAD
3. 使用 `sigma_bin = median_residual / 0.6745`（如果假设 Gaussian）
4. 或者直接使用 `sigma_bin = RMSE_bin`

### 10.3 保证 Baseline 不变

**关键原则**：
1. 所有新功能必须有对应的 `Mode` 或 `enable_*` flag
2. 默认值必须是 `Off` / `false`
3. 新增权重计算后，不改变现有的：
   - Point count
   - Frame/board observation acceptance
   - Solver observations list

**建议的 API 设计**：
```cpp
struct PolarAngleAdaptiveWeightOptions {
  bool enabled = false;
  std::vector<double> bin_edges_deg = {0, 30, 50, 70, 85, 100};
  double sigma_mode = "rmse";  // "rmse", "mad", "robust"
  double weight_floor = 0.1;
};
```

### 10.4 输出 polar_angle_weight_summary.txt

建议输出：
- 每个 bin 的 estimated sigma
- 每个 bin 的 point count
- 每个 bin 的 weight 统计（mean, min, max）
- 与 baseline（weight = 1.0）的对比

### 10.5 必须比较的指标

**Baseline 对比**：
1. `optimized_residual.overall_rmse` - overall RMSE
2. `optimized_residual.outer_rmse` - outer RMSE
3. `optimized_residual.internal_rmse` - internal RMSE
4. `optimized_residual.polar_angle_bins` - 每个 polar bin 的 RMSE
5. `optimized_residual.per_board_bins` - 每个 board 的 RMSE
6. `selection_result.accepted_*_count` - 接受的 point/board/frame 数
7. Training/Holdout RMSE 差异
8. External validation RMSE（如果有）

**防止回归**：
- 启用 weighting 后，Round2 的 `stage42_validation_pass` 应该仍然为 true
- Internal RMSE 不应显著增加
- Holdout RMSE 不应显著增加

---

## 附录：关键文件清单

| 文件路径 | 描述 |
|----------|------|
| `run_stage5_backend_main.cpp` | 主入口 |
| `FrozenRound2BaselinePipeline.cpp/hpp` | Round1/Round2 pipeline |
| `OuterOnlyCameraInitializer.cpp/hpp` | Auto-init |
| `MultiScaleOuterTagDetector.cpp/hpp` | Outer detection |
| `MultiBoardOuterBootstrap.cpp/hpp` | Multi-board bootstrap |
| `MultiBoardInternalMeasurementRegenerator.cpp/hpp` | Internal regeneration |
| `ApriltagInternalDetector.cpp/hpp` | Internal point detection (sphere-border-lattice) |
| `JointReprojectionMeasurementBuilder.cpp/hpp` | Measurement builder |
| `JointReprojectionCostCore.cpp/hpp` | Cost evaluation |
| `JointReprojectionOptimizer.cpp/hpp` | Backend optimizer |
| `JointReprojectionResidualEvaluator.cpp/hpp` | Residual evaluator |
| `JointMeasurementSelection.cpp/hpp` | Selection |
| `PolarAngleResidualDiagnostics.cpp/hpp` | Polar diagnostics |
| `JointMeasurementCuration.cpp/hpp` | Curation (blur weight, quality weight) |
