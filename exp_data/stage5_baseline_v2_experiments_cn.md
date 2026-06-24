# Stage5 Baseline v2 实验记录

本文记录当前冻结的 **Baseline v2** 结果，并补充后续并入主线的 **Close-edge outer subpix boost**。从当前版本开始，Close-edge boost 已作为默认 baseline 前端策略的一部分，不再只是实验分支。

## 1. Baseline v2 定位

Baseline v2 的核心思想是：

- 保留 DS camera model 作为默认相机模型；
- 使用当前 Stage5 outer + internal pipeline；
- 使用 Kalibr-style trial backend frame-board selection；
- 以稳定 seed bundle 为基础，再增量尝试接回可靠 frame-board observation；
- 默认启用 close-edge outer subpix boost，用于改善近距离、大面积、边缘 board 的外四角点 refinement；
- 不默认启用 angular residual、soft rescue 等其它实验分支。

通俗地讲，Baseline v2 是当前“稳态主线”。Close-edge boost 已经从增强实验并入这条主线，angular residual / rescue 等仍然保留为分支实验。

## 2. Full 数据集 Cross Validation

这组结果来自 full 数据集互相做训练 / 验证。

| 数据集 | Baseline v2 holdout overall | holdout outer | holdout internal | Kalibr holdout | 与 Kalibr overall 差值 |
|---|---:|---:|---:|---:|---:|
| `140151 -> 141444` | **10.8765** | 24.5540 | 6.94257 | 12.5853 | -1.70885 |
| `141444 -> 140151` | **6.11854** | 2.49702 | 6.46687 | 6.16428 | -0.04575 |
| `191538 -> 192347` | **3.75082** | 2.08755 | 3.92436 | 3.83157 | -0.08075 |
| `192347 -> 191538` | **2.89692** | 1.00722 | 3.06714 | 2.94125 | -0.04434 |

### 观察

- 四组 full cross validation 中，Baseline v2 的 holdout overall 都低于 Kalibr reference。
- `140151 -> 141444` 虽然 overall 明显优于 Kalibr，但 holdout outer 仍然较高，说明该 split 仍存在边缘 board / outer pose refit 层面的困难。
- `191538 -> 192347` 和 `192347 -> 191538` 两组相对稳定，Baseline v2 比 Kalibr 小幅更好。

## 3. Dataset 5-1 Right Cross Validation

这组结果来自 `20260430` 三个 right 数据集之间的 cross validation。

| 数据集 | 方法 | holdout overall | holdout outer | holdout internal | training overall | 与 Kalibr overall 差值 |
|---|---|---:|---:|---:|---:|---:|
| `134853 -> 144419` | Kalibr reference | 1.95731 | 0.59390 | 2.07724 | 0.853244 | - |
| `134853 -> 144419` | Baseline v2 | **1.94433** | **0.33110** | **2.07148** | **0.807135** | -0.012983 |
| `144928 -> 134853` | Kalibr reference | 3.32491 | 0.892417 | 3.53120 | 6.61579 | - |
| `144928 -> 134853` | Baseline v2 | **3.29735** | **0.890119** | **3.50174** | 6.30290 | -0.027567 |
| `144419 -> 144928` | Kalibr reference | 6.58414 | 15.5712 | **3.94820** | 0.746269 | - |
| `144419 -> 144928` | Baseline v2 | **6.57076** | **15.2445** | 4.09776 | **0.690635** | -0.013379 |

### 观察

- 三组 dataset 5-1 cross validation 中，Baseline v2 的 holdout overall 都略优于 Kalibr。
- `134853 -> 144419` 和 `144928 -> 134853` 是健康组：outer / internal 都基本优于或接近 Kalibr。
- `144419 -> 144928` 是困难组：outer RMSE 极高，Kalibr 也同样极高，说明该验证集存在明显 outer pose refit / 边缘 board 解释困难。
- `144419 -> 144928` 中 Baseline v2 的 internal 比 Kalibr 略差，但 outer 和 overall 仍略优。

## 4. 当前总表结论

| 数据范围 | 实验组数 | Baseline v2 overall 优于 Kalibr 的组数 | 结论 |
|---|---:|---:|---|
| Full cross validation | 4 | 4 / 4 | Baseline v2 在 full 数据集上整体优于 Kalibr |
| Dataset 5-1 right cross validation | 3 | 3 / 3 | Baseline v2 在 20260430 right 数据集上整体小幅优于 Kalibr |
| 合计 | 7 | 7 / 7 | 当前 Baseline v2 作为主线 baseline 是合理的 |

## 5. 与后续实验的关系

Baseline v2 之后仍可继续做分支对照，但 close-edge outer subpix boost 已经并入当前默认前端：

- **Outer-only ablation**：验证 internal points 是否提供稳定性收益；
- **Angular residual / hybrid angular**：验证强鱼眼边缘是否需要球面残差；
- **Camera model ablation**：例如 DS / KB / EUCM / pinhole-equi；
- **Rescue / visible-refit**：作为 frontend candidate source 或 diagnostic，不直接替代 Baseline v2。

## 6. Baseline v2 与 Close-edge Outer Subpix Boost 对比

这里记录 Baseline v2 之后被并入主线的一个重要改动：**close-edge outer subpix boost**。

该方案最初作为实验分支提出，用来验证一个针对性问题：

> 近距离、大面积、边缘 board 的外四角点是否需要更大的 subpixel refinement window。

实验配置：

- 普通点：`outer_subpix_scale = 0.35`；
- close-edge 点：满足大 polar / 大面积 / 靠边条件后，subpix window 乘 `1.4`；
- 如果 DS camera 暂时不可用，则使用图像中心距离作为 polar-like fallback；
- 输出目录前缀：`outer_subpix_polar_boost_proxy_*`。

### 6.1 Dataset 5-1 三方 Holdout 对比

| 数据集 | 方法 | holdout overall | holdout outer | holdout internal | 与 Kalibr overall 差值 | 与 Baseline v2 overall 差值 |
|---|---|---:|---:|---:|---:|---:|
| `134853 -> 144419` | Kalibr reference | 2.26154 | 0.535958 | 2.40542 | - | - |
| `134853 -> 144419` | Baseline v2 | **1.94433** | 0.33110 | **2.07148** | -0.31721 | - |
| `134853 -> 144419` | Close-edge boost | 2.25466 | **0.278826** | 2.40412 | -0.00688 | +0.31033 |
| `144928 -> 134853` | Kalibr reference | 3.14645 | 0.457635 | 3.35210 | - | - |
| `144928 -> 134853` | Baseline v2 | 3.29735 | 0.890119 | 3.50174 | +0.15090 | - |
| `144928 -> 134853` | Close-edge boost | **3.12771** | **0.399680** | **3.33311** | -0.01875 | -0.16964 |
| `144419 -> 144928` | Kalibr reference | 5.60878 | 14.6837 | 2.41462 | - | - |
| `144419 -> 144928` | Baseline v2 | 6.57076 | 15.2445 | 4.09776 | +0.96198 | - |
| `144419 -> 144928` | Close-edge boost | **2.26618** | **0.938152** | **2.39328** | -3.34260 | -4.30458 |

### 6.2 Full 数据集三方 Holdout 对比

| 数据集 | 方法 | holdout overall | holdout outer | holdout internal | 与 Kalibr overall 差值 | 与 Baseline v2 overall 差值 |
|---|---|---:|---:|---:|---:|---:|
| `140151 -> 141444` | Kalibr reference | 8.93838 | 20.4112 | 5.61444 | - | - |
| `140151 -> 141444` | Baseline v2 | 10.8765 | 24.5540 | 6.94257 | +1.93812 | - |
| `140151 -> 141444` | Close-edge boost | **7.72839** | **15.9570** | **5.63288** | -1.20999 | -3.14811 |
| `141444 -> 140151` | Kalibr reference | **5.60626** | 1.30087 | **5.96681** | - | - |
| `141444 -> 140151` | Baseline v2 | 6.11854 | 2.49702 | 6.46687 | +0.51228 | - |
| `141444 -> 140151` | Close-edge boost | 5.60707 | **1.26030** | 5.96891 | +0.000814 | -0.51147 |
| `191538 -> 192347` | Kalibr reference | 3.21759 | 0.451782 | 3.42714 | - | - |
| `191538 -> 192347` | Baseline v2 | 3.75082 | 2.08755 | 3.92436 | +0.53323 | - |
| `191538 -> 192347` | Close-edge boost | **3.18788** | **0.297851** | **3.39776** | -0.02971 | -0.56294 |
| `192347 -> 191538` | Kalibr reference | 2.50293 | 0.341813 | 2.66628 | - | - |
| `192347 -> 191538` | Baseline v2 | 2.89692 | 1.00722 | 3.06714 | +0.39399 | - |
| `192347 -> 191538` | Close-edge boost | **2.49571** | **0.222597** | **2.66031** | -0.00722 | -0.40121 |

### 6.3 Boost 触发统计

| 数据集 | outer corner debug 总数 | boost 触发数 | 普通点 final radius 均值 | boost 点 final radius 均值 | boost 点 polar/proxy 均值 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| `134853 -> 144419` | 1662 | 0 | 39.77 | - | - | 没有 close-edge 点触发 boost |
| `144928 -> 134853` | 2576 | 88 | 39.84 | 77.84 | 54.01° | 存在 close-edge 点，boost 有效 |
| `144419 -> 144928` | 1708 | 8 | 40.72 | 71.75 | 52.46° | 少量关键边缘点触发，收益很大 |
| `140151 -> 141444` | 6004 | 0 | 43.62 | - | - | 没有 boost 触发，但普通点 unclamped subpix 后提升明显 |
| `141444 -> 140151` | 5000 | 296 | 43.17 | 73.52 | 53.40° | 大量 close-edge 点触发，整体接近 Kalibr |
| `191538 -> 192347` | 2188 | 0 | 43.96 | - | - | 没有 boost 触发，但普通点窗口放大后提升明显 |
| `192347 -> 191538` | 2520 | 36 | 44.08 | 73.83 | 51.87° | 少量 close-edge 点触发，整体优于 Kalibr |

### 6.4 结论

- `144928 -> 134853` 和 `144419 -> 144928` 说明：close-edge outer subpix boost 对边缘困难数据非常有效。
- `144419 -> 144928` 的提升最明显，holdout overall 从 `6.57076` 降到 `2.26618`，outer RMSE 从 `15.2445` 降到 `0.938152`。
- `134853 -> 144419` 没有任何 boost 触发，但 overall 从 `1.94433` 退化到 `2.25466`。这说明退化不是 boost 触发导致的，而是普通点 unclamped / 普通 subpix 策略也改变了。
- Full 数据集 4 组中，Close-edge boost 分支全部优于 Baseline v2，其中 `140151 -> 141444` 从 `10.8765` 降到 `7.72839`，说明外四角点 refinement 窗口限制确实是一个关键瓶颈。
- 需要注意，`140151 -> 141444` 和 `191538 -> 192347` 的 boost 触发数为 0，但结果仍然提升，说明真正起作用的不只是 close-edge multiplier，还包括移除旧 `verification_roi_radius` clamp 后，普通点 subpix 窗口也变大。
- 因此，该实验明确证明了自适应大窗口有价值。当前实现已将它并入 baseline，但仍需要继续监控健康组退化风险。

后续更稳的优化方向是：

- 继续保留 close-edge hard case 的大窗口收益；
- 对健康组进一步检查普通点 unclamped subpix 是否过强；
- 如果健康组退化稳定出现，再把“普通点窗口”和“close-edge 窗口”拆成更细的策略。

## 7. 汇报表述建议

可以这样表述：

> 我们将当前稳定主线冻结为 Baseline v2，并把 close-edge outer subpix boost 并入默认前端。Baseline v2 的核心仍然是 DS 模型、outer+internal frontend、以及 Kalibr-style trial backend selection；新增的 close-edge boost 主要解决近距离、大面积、边缘 board 的外四角点 refinement 窗口不足问题。

补充 close-edge boost 后，可以这样说：

> Close-edge outer subpix boost 证明，部分高 residual 并不是 backend 无法解释，而是前端外角点 refinement 窗口不足导致的。对大 polar / 大面积 / 靠边的少量点放大 subpixel window 后，`144419 -> 144928` 的 holdout overall 从 `6.57076` 降到 `2.26618`。Full 数据集 4 组也全部优于原 Baseline v2，因此该方案已并入当前 baseline。需要注意的是，`134853 -> 144419` 这类健康组曾出现退化，所以后续仍要继续观察普通点窗口是否需要更保守。

## 8. 待确认项

- `140151 -> 141444` 的 high outer RMSE 需要结合 worst-case board / frame 可视化进一步解释。
- Close-edge boost 已并入 baseline；若后续进一步区分普通点窗口和 close-edge 窗口，可另开 `Baseline v3`。

## 9. Trial Selection 增强：Frame-cohesion vs Close-distance Boost

在 Baseline v2 冻结之后，我们又围绕 `trial backend frame-board selection` 做了两类增强实验：

- `Frame-cohesion`
  目的：解决“一整帧图像看起来正常，但 backend 只接收其中 1 个 board”的问题。
  方法：当某个 frame 已经有 accepted observation 时，允许同帧其它高分 candidate 作为 companion board 再进入一次 short trial backend。

- `Close-distance boost`
  目的：进一步提高“近距离、大面积 board candidate”在 trial selection 中被尝试、被接受的概率。
  方法：在 `frame-cohesion` 基础上，对满足近距离条件的 candidate 增加 score bonus，使其更容易进入 incremental / frame-cohesion 尝试。

### 9.1 当前 Close-distance Boost 使用的条件

当前宽口径 `Close-distance boost` 使用的是以下条件：

- `projected_area_ratio >= 0.04`
- `max_polar_angle_deg >= 50`
- `outer_pose_refit_rmse <= 4.0 px`

满足后：

- candidate 额外加 `score bonus = 0.5`
- 在 frame-cohesion 中允许使用更低阈值：
  `close_distance_frame_cohesion_min_candidate_score = 2.8`
  而普通 candidate 仍使用 `3.2`

通俗讲，就是：

> 它不是直接强行把近距离 board 加进 backend，而是让“近距离 + 偏边缘 + 外角点还算靠谱”的 candidate 更容易获得一次 short trial backend 的机会。

### 9.2 `134853 -> 144419` 对比

| 方法 | holdout overall | holdout outer | holdout internal | training overall | Backend 输入 |
|---|---:|---:|---:|---:|---:|
| Baseline v2 | 2.25466 | 0.278826 | 2.40412 | 0.814461 | 95B |
| Frame-cohesion | 2.25389 | **0.194451** | 2.40445 | 0.815369 | 117B |
| Close-distance boost | **2.25236** | 0.213322 | **2.40259** | **0.813682** | 126B |
| Kalibr | 2.26154 | - | - | - | - |

补充统计：

- Frame-cohesion：
  `frame_cohesion_accepted_count = 25`
- Close-distance boost：
  `frame_cohesion_accepted_count = 33`
  `close_distance_candidate_count = 255`
  `close_distance_accepted_count = 41`

结论：

- `Frame-cohesion` 明显增加了 backend 输入，从 `95B` 提升到 `117B`；
- `Close-distance boost` 又进一步把 backend 输入提升到 `126B`；
- 这一组里 `Close-distance boost` 的 overall 最好，但提升幅度很小，属于“略优于 frame-cohesion”。

### 9.3 `144419 -> 144928` 对比

| 方法 | holdout overall | holdout outer | holdout internal | training overall | Backend 输入 |
|---|---:|---:|---:|---:|---:|
| Baseline v2 | 2.26618 | 0.938152 | 2.39328 | 0.775567 | 72B |
| Frame-cohesion | 2.26609 | 0.939285 | 2.39312 | 0.775592 | 99B |
| Close-distance boost | **2.26526** | **0.931088** | **2.39267** | 0.775738 | 94B |
| Kalibr | 5.60878 | - | - | - | - |

补充统计：

- Frame-cohesion：
  `frame_cohesion_accepted_count = 30`
- Close-distance boost：
  `frame_cohesion_accepted_count = 27`
  `close_distance_candidate_count = 266`
  `close_distance_accepted_count = 40`

结论：

- `Frame-cohesion` 已经解决了原来“一帧只进一个 board”的问题，backend 输入从 `72B` 提高到 `99B`；
- `Close-distance boost` 没有继续增加总 board 数，反而是把 `99B` 调整成 `94B`；
- 但它在这一组的 overall / outer / internal 三项都略优于 `Frame-cohesion`，说明它的价值不只是“多加样本”，而是“更偏向某类近距离大板 candidate”。

### 9.4 当前判断

从这两组补充实验看：

- `Frame-cohesion` 是稳定有效的结构性增强：
  它明确改善了 backend 输入的 frame-board 完整性。
- `Close-distance boost` 在这两组上都略优于 `Frame-cohesion`，
  说明“对近距离候选做轻度优先级倾斜”这个思路是有潜力的。
- 但当前宽口径 `Close-distance boost` 触发仍然偏多：
  `255 / 266` 个 candidate 被打标，说明它还不够聚焦。

因此目前更稳的判断是：

- `Frame-cohesion` 值得保留为 trial selection 的核心增强；
- `Close-distance boost` 可以保留为增强分支继续验证；
- 在更多 split 验证之前，不建议直接替代当前 Baseline v2 默认策略。

### 9.5 复查：04-27 两组 Trial Selection 增强实验

这次补充跑了 `20260427_191538 <-> 20260427_192347` 两个方向，用来检查：

- `Frame-cohesion trial selection` 是否能继续增加同帧 board 的接入；
- `Close-distance candidate score boost` 是否能在 `Frame-cohesion` 之外带来额外收益。

需要特别说明：本小节记录的是 **trial selection 层面的 close-distance score boost**，不是第 6 节的 **Close-edge outer subpix boost**。第 6 节的 Baseline v2 / Close-edge outer subpix boost 结果保持不变，本小节不替代 baseline。

本轮原计划还包含 `20260421_140151 <-> 20260421_141444`。这四组后续已经完成，因此一并补入下表。

| Split | 方法 | holdout overall | holdout outer | holdout internal | Kalibr holdout | Backend 输入 | 额外接受统计 |
|---|---|---:|---:|---:|---:|---:|---|
| `191538 -> 192347` | Frame-cohesion | 5.17482 | 4.92530 | 5.20812 | 3.21759 | 29F / 110B / 440O / 3127I | frame-cohesion accepted 29 |
| `191538 -> 192347` | Close-distance score boost | 5.17482 | 4.92530 | 5.20812 | 3.21759 | 29F / 113B / 452O / 3210I | frame-cohesion accepted 32; close-distance accepted 48 |
| `192347 -> 191538` | Frame-cohesion | 3.59634 | 3.48731 | 3.61105 | 2.50293 | 29F / 109B / 436O / 3079I | frame-cohesion accepted 18 |
| `192347 -> 191538` | Close-distance score boost | 3.59634 | 3.48731 | 3.61105 | 2.50293 | 30F / 117B / 468O / 3331I | frame-cohesion accepted 25; close-distance accepted 24 |
| `140151 -> 141444` | Frame-cohesion | 8.29669 | 11.9028 | 7.64008 | 8.93838 | 36F / 124B / 496O / 3524I | frame-cohesion accepted 34 |
| `140151 -> 141444` | Close-distance score boost | 8.29669 | 11.9028 | 7.64008 | 8.93838 | 36F / 128B / 512O / 3669I | frame-cohesion accepted 38; close-distance accepted 52 |
| `141444 -> 140151` | Frame-cohesion | 6.02448 | 3.81844 | 6.27226 | 5.60626 | 43F / 151B / 604O / 4302I | frame-cohesion accepted 24 |
| `141444 -> 140151` | Close-distance score boost | 6.02448 | 3.81844 | 6.27226 | 5.60626 | 40F / 150B / 600O / 4234I | frame-cohesion accepted 26; close-distance accepted 34 |

### 9.6 与 Baseline v2 / Close-edge Outer Subpix Boost 的直接对比

为了避免混淆，这里把三类方法放在同一张表里：

- `Baseline v2`：原本冻结的主线结果；
- `Frame-cohesion / Close-distance score boost`：本节 trial selection 层面的增强；
- `Close-edge outer subpix boost`：第 6 节前端 outer subpix 窗口策略，已经证明是更强的改进来源。

| Split | Baseline v2 overall | Frame-cohesion overall | Close-distance score boost overall | Close-edge outer subpix boost overall | Kalibr overall | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `140151 -> 141444` | 10.8765 | 8.29669 | 8.29669 | **7.72839** | 8.93838 | Trial selection 比 Baseline v2 好，但不如前端 subpix boost |
| `141444 -> 140151` | 6.11854 | 6.02448 | 6.02448 | **5.60707** | 5.60626 | Trial selection 小幅改善，但前端 subpix boost 几乎追平 Kalibr |
| `191538 -> 192347` | 3.75082 | 5.17482 | 5.17482 | **3.18788** | 3.21759 | Trial selection 明显退化，前端 subpix boost 最好 |
| `192347 -> 191538` | 2.89692 | 3.59634 | 3.59634 | **2.49571** | 2.50293 | Trial selection 明显退化，前端 subpix boost 最好 |

按 split 分析：

- `140151 -> 141444`：
  `Frame-cohesion / Close-distance score boost` 将 overall 从 `10.8765` 降到 `8.29669`，主要改善 outer，但 internal 反而从 `6.94257` 变差到 `7.64008`。第 6 节的 `Close-edge outer subpix boost` 进一步降到 `7.72839`，仍是更优方案。
- `141444 -> 140151`：
  Trial selection 从 `6.11854` 小幅降到 `6.02448`，改善主要来自 internal，但 outer 从 `2.49702` 变差到 `3.81844`。第 6 节的 `Close-edge outer subpix boost` 达到 `5.60707`，几乎追平 Kalibr `5.60626`。
- `191538 -> 192347`：
  Trial selection 从 Baseline v2 的 `3.75082` 退化到 `5.17482`，outer 从 `2.08755` 退化到 `4.92530`。第 6 节的 `Close-edge outer subpix boost` 为 `3.18788`，优于 Kalibr `3.21759`。
- `192347 -> 191538`：
  Trial selection 从 Baseline v2 的 `2.89692` 退化到 `3.59634`，outer 从 `1.00722` 退化到 `3.48731`。第 6 节的 `Close-edge outer subpix boost` 为 `2.49571`，优于 Kalibr `2.50293`。

补充观察：

- 这四个 split 里，`Close-distance score boost` 通常比纯 `Frame-cohesion` 多接入了一些 frame-board observation。
- 但是最终 holdout overall / outer / internal 完全没有变化，说明这些额外 observation 没有带来可见收益。
- 当前输出中的 Kalibr reference 为 `3.21759 / 2.50293`，与第 6 节 Close-edge boost 表格一致；它和更早 New Baseline 文档中的 `3.83157 / 2.94125` 不同，是因为后续 evaluator / holdout observation 口径已经更新。
- 本次我们的结果 `5.17482 / 3.59634` 明显差于第 6 节的 Close-edge outer subpix boost `3.18788 / 2.49571`，原因是两者不是同一个实验：本次只改变 trial selection candidate score，而第 6 节主要来自前端 outer subpix / unclamped window 的改动。

因此这次复查的结论是：

> 对 04-21 和 04-27 这四个 split，trial selection 层面的 close-distance score boost 能增加 backend 输入数量，但没有改善 holdout；Frame-cohesion 在 04-21 两组有一定收益，但在 04-27 两组明显退化。当前不应把这些 trial selection 增强并入 Baseline v2。真正稳定有效的改进仍是第 6 节的 Close-edge outer subpix boost。

本轮左相机 `134853 left -> 144419 left` intrinsics 结果已单独写入：

- `config/stage5_baseline_mono_134853_to_144419_left_intrinsics.yaml`
- `config/stage5_baseline_mono_134853_to_144419_right_intrinsics.yaml`
