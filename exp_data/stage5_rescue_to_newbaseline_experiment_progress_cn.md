# Stage5 从 Rescue 到 New Baseline 的实验进展总结

本文整理近期围绕 Stage5 做的一系列实验探索。整体逻辑是：

> 先尝试救回漏检 board 和 internal 点，发现直接 rescue 不够稳定；随后转向更稳的 frame-board 级观测选择；最后冻结 New Baseline，并继续做 internal 消融和 angular residual 分支实验。

## 1. 起点问题：边缘 board 漏检和 internal 生成失败

### 问题

在强鱼眼、多 board AprilTag 场景中，部分边缘 board 会出现：

- AprilTag ID 没检测到；
- 外四角点没有可靠进入系统；
- internal points 生成失败；
- 某些 frame-board 在局部看合理，但放到全局 scene pose 下 residual 很大。

典型现象包括 frame 90/91/92/93 等边缘 frame 中的 board2 / board3 漏检或预测偏移。

### 第一阶段尝试：geometry prior outer seed

目的：

- 利用已有 `T_cam_reference` 和 `T_reference_board` 预测 missing board 的 ROI；
- 给漏检 board 提供一个几何先验；
- 尝试把未检测到 ID 的 board 找回来。

方法：

- 对 missing board 投影预测 outer quad；
- 输出 `geometry_prior_outer_seed_candidates.csv`；
- 加入 ROI overlay、corner seed overlay；
- 初期只做 diagnostic-only，避免把纯几何投影当作真实观测。

结果：

- 能大致指出 missing board 可能在哪里；
- 但如果直接把预测点当观测，会形成 self-confirming bias；
- 因此最终收敛为：几何投影只能作为 ROI / seed，不能直接进 backend。

结论：

> geometry prior 可以帮助定位候选区域，但不能直接替代真实图像观测。

实验对比：

| 方案 | 是否直接加入 backend | 是否改变 baseline | 主要输出 | 结果 |
|---|---|---|---|---|
| 无 geometry prior | 否 | 否 | 无 missing-board seed | 漏检 board 无额外 ROI 引导 |
| geometry prior outer seed diagnostic-only | 否 | 否 | `geometry_prior_outer_seed_candidates.csv` / ROI overlay | 能给出 missing board 的预测位置，但只作为 ROI / seed |
| 早期 geometry prior rescue | 曾尝试 | 是 | rescued outer / internal 诊断 | 风险较高，容易把几何预测当真实观测 |

这一组实验的结论不是 RMSE 提升，而是明确了边界：**几何预测只能当先验，不能直接当观测进入 backend**。

## 2. Rescue 图像验证：从几何预测到真实观测

### 尝试 1：ROI 内重新 AprilTag 检测

目的：

- 对 missing board 的预测 ROI 放大后重新跑 AprilTag detector；
- 如果能重新解出 tag ID，就走正常 outer refine 流程。

方法：

- missing board ROI bbox 扩大；
- 在 crop ROI 内重跑 detector；
- 后续加入 DS/ray-space rectified patch 尝试。

结果：

- 对部分样本有帮助；
- 但很多强畸变边缘 tag 仍然解不出 ID；
- 只靠 tag-id redetect 不能完全解决问题。

实验对比：

| 方案 | ROI 内处理方式 | 是否要求 tag ID | 是否允许进入 backend | 结果 |
|---|---|---|---|---|
| 原始 detector | 全图 AprilTag 检测 | 是 | 是 | 边缘强畸变 board2/3 容易漏检 |
| ROI redetect | missing board ROI 放大后重跑 detector | 是 | 仅 ID 成功时 | 部分样本可恢复，但大量边缘 tag 仍检测不到 ID |
| DS ray patch redetect | 基于 DS/ray-space rectified patch 后检测 | 是 | 仅 ID 成功时 | 比普通 crop 更合理，但仍不能稳定解出所有目标 tag |

结论是：**tag-id 成功是最安全证据，但在强鱼眼边缘不够稳定**。

### 尝试 2：边界 / 角点 evidence 验证

目的：

- 即使 tag ID 解不出来，也尝试通过 board 边界、角点响应、edge support 找到可信 outer corners。

方法：

- 在预测 quad 周围做 edge / corner evidence 检查；
- 加 adaptive subpix refinement；
- 加 outer pose refit 验证；
- 只有通过图像证据和 pose refit 的 candidate 才允许进入后续。

结果：

- 一些外四角点可以被救回；
- 但 early 版本有两个问题：
  - 如果使用当前全局 `T_cam_reference` 预测 missing board，某些 frame 的预测位置会整体偏；
  - 外四角点即使看起来不错，internal generation 仍可能被 seed / support 条件卡住。

实验对比：

| 方案 | 图像证据 | outer refit gate | 典型结果 |
|---|---|---:|---|
| 纯几何预测 | 无 | 无 | ROI 可视化有用，但不能作为真实观测 |
| edge / corner evidence validation | 有边缘/角点证据 | 约 3 px 固定阈值 | 更安全，但部分看起来合理的候选被卡住 |
| adaptive outer RMSE threshold | 有边缘/角点证据 | 按当前帧真实 board 质量自适应 | 比固定 3 px 更合理，但仍依赖 frame pose / visible-refit 质量 |

这一阶段说明：**outer rescue 的关键不是“预测得像不像”，而是预测后能否被图像证据和局部 pose refit 支撑**。

结论：

> Rescue 的核心不能只靠“预测位置”，必须有图像证据、局部 refit 和后续 residual 验证。

failed-board / rescue 策略对比：

| 方案 | holdout overall | holdout outer | holdout internal | 结论 |
|---|---:|---:|---:|---|
| A / off-rescue | 7.0405 | 4.2109 | 7.3607 | 失败 board 仍可能影响结果 |
| strict failed-board drop | 7.0268 | 3.8359 | 7.3607 | 最稳，不依赖 rescue |
| rescue gate8 | 7.3022 | 4.2109 | 7.6442 | 接回 2 个失败 board，但整体变差 |

这张表是早期重要转折点：**与其强行 rescue，不如先保证失败 board 不污染 backend**。

## 3. Intermediate model 与 visible-refit

### Outer-only intermediate calibration

目的：

- 参考 TartanCalib 的 intermediate model 思路；
- 先用所有有效 outer observations 做一轮 outer-only intermediate calibration；
- 得到更好的 DS intrinsics、board poses 和 frame poses，再服务后续 internal regeneration / missing-board ROI。

方法：

- 在 `outer-only DS auto-init -> multi-board bootstrap` 后加入 intermediate calibration；
- 第一版 diagnostic-only；
- 后续 I2/I3 允许用 intermediate state 做 Round1 或完整 frontend regeneration。

结果：

- intermediate model 能输出 outer-only calibration 诊断；
- 但在 missing-board ROI 上，旧 bootstrap 预测和 intermediate 预测有时差异很小；
- 对某些偏移严重的 frame，问题不一定来自 camera intrinsics，而更像当前 frame pose 估计不可靠。

实验对比：

| 阶段 | outer-only calibration RMSE before | outer-only calibration RMSE after | 结论 |
|---|---:|---:|---|
| outer-only intermediate calibration | 5.2384 | 3.8728 | intermediate 能明显降低 outer residual |

说明：intermediate calibration 本身有效，能让 outer-only scene 更好解释已检测 outer observations；但它对 missing-board ROI 的帮助取决于 frame pose 是否可靠。

结论：

> Intermediate model 是合理方向，但它不是万能修复；如果 `T_cam_reference` 本身被局部坏观测拉偏，单纯换 intrinsics 或 intermediate state 不一定能修好 missing-board ROI。

### visible-refit

目的：

- 对 missing-board ROI 不再完全依赖当前单帧全局 `T_cam_reference`；
- 改用当前帧中已经可见的正常 board 做局部 pose refit，再预测 missing board。

方法：

- 用可见 board 的真实 outer observations 重新估计当前 frame pose；
- 再根据全局 board structure 推断 missing board 位置；
- 对比 old / intermediate / visible-refit 三种预测。

结果：

- 对部分 frame 明显改善 missing board 位置；
- 说明问题不完全是 board 结构，也不完全是相机模型，而是某些 frame 的全局 pose 可能不够可靠。

实验对比：

| missing-board ROI 来源 | 依赖信息 | 可视化现象 | 结论 |
|---|---|---|---|
| old / optimized scene | 当前全局 `T_cam_reference` | frame90 等困难帧仍可能偏很多 | 单帧全局 pose 可能不可靠 |
| intermediate scene | intermediate camera / board / frame state | 与 old 预测常常差异不大 | 说明问题不一定是 intrinsics |
| visible-refit | 当前帧可见 board 的局部 pose refit | 对部分 frame 能把 ROI 拉回正确位置 | 更适合做 missing-board ROI 引导 |

这组实验最重要的判断是：**偏移严重时，优先怀疑当前 frame pose，而不是盲目继续调相机模型**。

结论：

> visible-refit 比单纯使用全局 `T_cam_reference` 更适合做 missing-board ROI 引导。

## 4. Internal generation 与过滤条件

### 问题

在 rescued outer corners 已经较准的情况下，internal overlay 中仍出现：

- 内点 seed 看起来贴近真实角点，但被标为 invalid；
- 放开过滤后，很多点会进入 backend，但 holdout 反而变差。

### 尝试 1：跳过 `SearchSphereLatticeSeed` / `RefineSphereSeedRayLocally` 过滤

目的：

- 验证是否是 seed 过滤太严格，误杀了大量看起来正常的 internal 点。

方法：

- 强制使用 predicted / refined seed；
- 输出 forced internal seed overlay；
- 可视化哪些点被过滤、哪些点进入 backend。

结果：

- 放开后确实能让更多 internal 点通过；
- 但也会带入一部分坏点，导致 residual 和 holdout 变差。

实验对比：

| 方案 | internal seed filter | backend training RMSE | holdout RMSE | 典型问题 |
|---|---|---:|---:|---|
| 默认 seed filter | 开启 | 待确认 | 待确认 | 会误杀很多看起来正常的点 |
| bypass seed filters | 跳过 `SearchSphereLatticeSeed` / `RefineSphereSeedRayLocally` 的过滤 | 0.8093 | 6.4801 | 更多点进入，但坏点也进入 |

说明：bypass 版本在该输出中能跑通，并且训练集 backend RMSE 很低，但可视化和 worst case 表明它会带入一些高 residual frame-board，因此不能直接作为 baseline。

结论：

> 原过滤条件确实过严，会误杀好点；但完全放开也不行，会引入坏点。

### 尝试 2：quality / residual adaptive 过滤

目的：

- 不再用单一 hard threshold；
- 尝试用 residual 分布和质量信息区分好点 / 坏点。

方法：

- 输出每个 internal 点的 quality、residual、是否进入 backend；
- 将可视化颜色拆分，避免“没进 backend”和“residual 大”混在一起；
- 增加 backend-used visualization。

结果：

- 诊断可读性提高；
- 但单纯点级过滤收益有限，因为很多问题是整块 board 或整帧系统性偏移，而不是孤立点 outlier。

实验对比：

| 方案 | backend 输入 | backend training RMSE | holdout RMSE | 结论 |
|---|---|---:|---:|---|
| bypass seed filters | 更多 internal 强制进入 | 0.8093 | 6.4801 | 点更多，但坏点风险仍在 |
| quality / residual adaptive | 按 quality 和 residual 过滤 internal | 0.7592 | 6.4825 | training 更低，holdout 基本持平 |

这说明点级 quality filter 有帮助，但主要问题仍不是少数孤立点，而是 frame-board 层级是否可靠。

结论：
å
> internal 点质量控制还值得继续研究，但当前更有效的层级不是单点，而是 frame-board observation。

## 5. Global scene consistency audit

### 目的

确认到底是 rescued internal 点坏，还是全局 scene state 中的 `T_cam_reference / T_reference_board` 有问题。

### 方法

对每个 frame-board observation 比较：

- local pose：只用当前 frame 当前 board 的 outer points refit；
- global pose：用 `T_cam_board = T_cam_reference * T_reference_board`；
- 输出 local-vs-global residual 和 pose delta。

### 结果

诊断发现一些关键现象：

- rescued internal 点在 local pose 下 RMSE 约 1 px，说明点本身不一定坏；
- 同一批点在 global scene pose 下 residual 可达几十甚至上百 px；
- frame90 board1 等 reference board 也出现 local 好、global 差。

实验对比：

| 观测集合 | local pose 下 residual | global scene 下 residual | 说明 |
|---|---:|---:|---|
| rescued internal mean | local internal RMSE ≈ 0.998 px | global internal RMSE ≈ 46.38 px | 点本身可解释，但全局 scene 对不上 |
| frame90 board2 rescued | local 可成功 refit | global internal RMSE ≈ 107.93 px | 更像 frame/global pose 问题 |
| frame92 board2 rescued | local 可成功 refit | global internal RMSE ≈ 86.83 px | 同上 |
| frame90 board1 normal/reference | local outer RMSE ≈ 0.723 px | global outer RMSE ≈ 54.997 px | reference board 自己也 global 差，优先怀疑 `T_cam_reference` |

### 结论

> 问题更像 frame pose / global scene state 与局部观测不一致，而不是 rescued internal 点本身必然错误。因此，直接把 rescued observation 接进 backend 风险较高。

## 6. 从 Rescue 转向观测选择：Kalibr-style trial backend selection

### 为什么转向

Rescue 的主要风险是：

- 如果候选 board 只靠几何预测，容易自我确认；
- 如果直接放开 internal 点，容易引入坏点；
- 单点过滤不能解决整块 board 的系统性偏移。

因此策略从“尽量救回所有点”转成：

> 先形成稳定 seed bundle，再把额外 frame-board observation 放入 candidate pool，通过 short trial backend 判断是否安全接回。

### New Baseline 方法

步骤：

1. 保留 Old Baseline 选中的稳定 seed bundle；
2. 从完整 Round2 measurement 构造 candidate pool；
3. 对每个 frame-board candidate 计算 score；
4. score 考虑：
   - outer/internal point count；
   - polar / view coverage；
   - board diversity；
   - residual quality；
5. 增量尝试加入 candidate；
6. 每次短程 backend 后检查 global / outer / internal RMSE；
7. 不破坏优化的 candidate 才进入 final backend。

New Baseline 配置：

| 参数 | 数值 |
|---|---:|
| max_candidate_additions | 20 |
| min_candidate_score | 3.2 |
| min_coverage_gain | 2.5 |
| max_accepted_per_board | 4 |
| max_accepted_per_frame | 1 |

### 解决的问题

- Old Baseline 太保守，coverage 不够；
- Rescue 太激进，容易把坏观测接进来；
- New Baseline 在二者之间，用 backend 试优化做安全验证。

### 结果

三组 `dataset_5_1` cross-dataset 上，New Baseline 相比 Old Baseline 的 holdout overall 都下降：

| Split | Old Baseline holdout | New Baseline holdout | 变化 |
|---|---:|---:|---:|
| `134853 -> 144419` | 2.1105 | 1.94433 | -0.16617 |
| `144928 -> 134853` | 3.4419 | 3.29735 | -0.14455 |
| `144419 -> 144928` | 6.8392 | 6.57076 | -0.26844 |

结论：

> 观测选择 / frontend / backend input 的改进，比直接 rescue 更稳定。因此 `current` 被冻结为 New Baseline。

## 7. relaxed40 / relaxed60 与 Consistency-Soft Enhancement

### relaxed40 / relaxed60

目的：

- 测试接回更多 candidate 是否能进一步提高 coverage 和 holdout 表现。

方法：

- relaxed40 / relaxed60 放宽 candidate 数量和 per-board / per-frame cap。

结果：

- 有些数据集略有收益；
- 有些困难验证集会轻微退化；
- 更多 candidate 并不是单调更好。

实验对比：

| Split | New Baseline | relaxed40 | relaxed60 | 结论 |
|---|---:|---:|---:|---|
| `134853 -> 144419` | 1.94501 | 1.94398 | 1.94379 | 放宽略有收益，但幅度很小 |
| `144928 -> 134853` | 3.25668 | 3.25674 | 3.25662 | 基本持平 |
| `144419 -> 144928` | 6.59282 | 6.59607 | 6.61269 | 放宽越多越容易退化 |

结论：

> relaxed40 可作为 ablation，relaxed60 暂不适合主推。

### Consistency-Soft Enhancement

目的：

- 避免接入局部 pose 与全局多板结构不一致的 candidate。

方法：

- 对候选 frame-board 做 local outer pose refit；
- 比较 `T_cam_board_local` 和 `T_cam_reference * T_reference_board`；
- 将 consistency error 作为 score penalty，而不是直接 hard reject。

结果：

- 在部分困难序列上有效；
- 但不是所有 split 稳定收益。

实验对比：

| Split | New Baseline | Consistency-Soft Enhancement | 变化 | 结论 |
|---|---:|---:|---:|---|
| `144928 -> 134853` | 3.25668 | 3.29819 | +0.04151 | consistency score 反而打乱较好排序 |
| `144419 -> 144928` | 6.59282 | 6.57448 | -0.01834 | 困难验证集有帮助 |
| `191538 -> 192347` | 3.75082 | 3.74094 | -0.00988 | 小幅改善 |
| `192347 -> 191538` | 2.89692 | 2.89884 | +0.00192 | 小幅变差 |

结论：

> Consistency-Soft Enhancement 作为可选增强分支保留，不作为默认 New Baseline。

## 8. New Baseline vs outer-only 消融

### 目的

验证 regenerated internal points 是否真的有价值。

### 方法

对比：

- New Baseline：outer + internal；
- outer-only：整个训练/优化链路都不使用 internal，backend 输入 internal count 必须为 0；
- Kalibr reference：外部棋盘格 camchain 在同一 evaluator 下评估。

### 结果

| Split | New Baseline holdout | outer-only holdout | Kalibr holdout | 结论 |
|---|---:|---:|---:|---|
| `144928 -> 134853` | 3.29735 | 3.28135 | 3.31184 | outer-only 略好 |
| `134853 -> 144419` | 1.94433 | 1.97010 | 1.98124 | New Baseline 更好 |
| `141444 -> 140151` | 6.11854 | 6.12580 | 6.16768 | New Baseline 更好 |
| `144419 -> 144928` | 6.57076 | 6.56313 | 6.58801 | outer-only 略好 |
| `191538 -> 192347` | 3.75082 | 3.83290 | 3.87601 | New Baseline 更好 |
| `192347 -> 191538` | 2.89692 | failed | 2.94125 | outer-only 失败 |

结论：

> outer-only 已经很强，说明外四角点链路本身可靠；但 internal points 提升了整体稳定性，并避免某些 split 的 outer-only 优化失败。因此 internal regeneration 的价值主要是“增加约束和稳定性”，而不是每组都单调降低 RMSE。

## 9. Angular residual：改到球面上优化

### 目的

验证强鱼眼边缘区域是否更适合用 sphere angular residual，而不是 image-plane residual。

### 方法

对比三组：

| 组别 | residual model | 说明 |
|---|---|---|
| A | `image_plane` | New Baseline 默认 |
| B | `sphere_angular` | 所有点都用球面 angular residual |
| E50 | `hybrid_edge_angular` | polar angle >= 50 deg 用 angular，其余用 image-plane |

### 结果

在 `144928 -> 134853` 的 New Baseline 口径下：

| 组别 | training overall | holdout overall | backend angular RMSE | 结论 |
|---|---:|---:|---:|---|
| A | 1.08017 | 3.25691 | 0.000698 | 当前最好 |
| B | 3.08053 | 4.04465 | 0.002037 | 明显变差 |
| E50 | 1.12710 | 3.27176 | 0.000749 | 可运行，但略差于 A |

E50 的 residual 分配：

- image residual count：2585；
- angular residual count：432；
- 约 14.3% 点进入 angular residual。

结论：

> Full angular residual 不是当前主线；E50 hybrid 能正常工作，但在 New Baseline 口径下还没有超过 A。Angular residual 作为候选分支保留，后续应继续做多数据集验证，而不是替换默认 baseline。

## 10. 当前总路线结论

这段时间的实验推进可以概括为四个阶段：

1. **Rescue 阶段**：尝试救回 missing board / internal 点，发现纯几何或过度放宽风险较高。
2. **诊断阶段**：通过 local-vs-global audit 发现一些问题来自 frame pose / global scene inconsistency，而不是单个 internal 点。
3. **选择阶段**：转向 Kalibr-style trial backend frame-board selection，用 short backend 判断 candidate 是否值得接回。
4. **冻结阶段**：将 `current` 冻结为 New Baseline，outer/internal 消融和 angular residual 都在 New Baseline 上继续验证。

最重要结论：

- Rescue 有价值，但目前更适合作为诊断和候选生成，不适合直接做 baseline；
- 单点 internal filter 不能解决主要问题，frame-board 级观测选择更有效；
- New Baseline 比 Old Baseline 在 cross-dataset holdout 上更稳；
- internal points 对整体稳定性有帮助；
- angular residual 目前只能作为候选分支，默认仍保持 image-plane residual。

## 11. 汇报时推荐说法

可以这样讲：

> 我们一开始试图直接救回强鱼眼边缘漏检的 board 和 internal 点，但发现这类 rescue 如果缺少图像证据和全局一致性检查，容易把坏观测带进 backend。随后我们通过 local-vs-global 诊断发现，很多异常并不是单个点坏，而是 frame-board 层级的全局一致性问题。因此最终方法从“硬救点”转为“候选观测增量接回”：以稳定 baseline 为 seed，对额外 frame-board observation 做 coverage / quality / diversity 评分，并通过 short trial backend 判断是否保留。这个策略在多组 cross-dataset 上比旧严格 baseline 更稳，因此冻结为 New Baseline。后续 outer/internal 消融说明 outer-only 已经很强，但 internal points 能提升整体稳定性；angular residual 目前作为候选分支，full angular 不稳定，hybrid E50 可运行但尚未超过默认 image-plane。

## 12. 后续建议

短期建议：

1. 继续以 New Baseline 作为默认主线；
2. Rescue 保留为 diagnostic / candidate source，不直接默认接入；
3. 对 accepted candidate 做更细的质量分析，尤其是哪些 board / polar bin 真正带来收益；
4. Angular residual 继续作为分支验证，优先多数据集验证 E50，不再主推 full angular；
5. internal quality 方向不要简单放宽过滤，而应研究更可靠的局部图像证据和 frame-board 级质量建模。
