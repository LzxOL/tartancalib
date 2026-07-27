# Stage5 Residual Baseline (2026-07-11)

本记录只包含修复 active-camera unprojection、normalized-ray Jacobian 和
residual-aware selection/acceptance 后的新版 persistent incremental BA。
旧版 Spherical `4 accepted / 40 rejected` 结果无效，不得用于论文比较。

## Robust Solver Profile Update

本日晚些时候进一步修复了 residual mode 共用 Pixel 数值配置的问题。当前
baseline 使用以下 mode-specific solver contract：

- Pixel：原生 pixel objective，保留短 candidate trial budget。
- Spherical：tangent residual 使用固定初始 `f_ref` 做常数 px-equivalent
  数值缩放；selection/health 仍使用 rad，不改变 angular objective 的最优解。
- Hybrid：pixel/angular 独立 residual blocks，angular block 使用固定
  `f_ref` 缩放，selection/health 使用 hybrid px-equivalent metric。
- Bearing 模式在 candidate joint solve 前先固定 camera 做 frame-pose prefit。
- Seed intrinsics warm-up 在 persistent estimator 接管前通过独立 problem
  完成，失败会显式 rollback，不改变 estimator 内部变量维度。
- Bearing joint solve 使用 incremental marginal solver 兼容的默认 trust
  region；通用 LM 只用于独立 warm-up，不用于 marginal system。
- 每轮最多 50 次，并在需要时自适应 continuation；相对 objective 或 camera
  parameter block 收敛后停止，最多追加 3 轮作为显式 runtime guard。
- 未收敛、非有限、objective 回升、residual health 失败的 candidate 均 rollback。

修复后 smoke：

| Dataset | Mode | Attempted | Accepted | Continuation | Guard hit | Holdout RMSE [px] |
|---|---|---:|---:|---:|---:|---:|
| 144928-clear | Pixel | 1 | 1 | 0 | 0 | 0.6316 |
| 144928-clear | Spherical | 1 | 1 | 2 | 0 | 0.6714 |
| 144928-clear | Hybrid | 1 | 1 | 3 | 0 | 0.6732 |
| 1444190-clear | Spherical | 1 | 1 | 2 | 0 | 0.5168 |
| 1444190-clear | Hybrid | 1 | 1 | 0 | 0 | 0.5182 |

`144928-clear` 连续 4-batch 测试中 Spherical 和 Hybrid 均接受 2、拒绝 2；
拒绝原因分别为 camera block 尚未收敛或 objective 回升，commit/rollback 状态
保持有效。下方早期 44-batch 表格是在 robust solver profile 合入前产生，仅用于
追踪问题来源，不能作为当前 baseline 的论文结果；论文表格必须重新跑完整数据。

## Baseline Modes

| Baseline | `RESIDUAL_MODEL` | BA residual | Selection / acceptance metric |
|---|---|---|---|
| Pixel | `pixel_only` | 2D image-plane | pixel |
| Spherical | `sphere_angular` | 2D tangent-plane component-wise | tangent angular |
| Hybrid | `polar_continuous_hybrid` | continuous pixel/angular hybrid | hybrid objective |

三组均使用同一 Stage5 camera initialization、random 70/30 split、frontend、
camera model 和 persistent incremental estimator。不存在 final BA。

统一入口：

```bash
RESIDUAL_MODEL=pixel_only scripts/run_stage5_current_baseline_ds.sh
RESIDUAL_MODEL=sphere_angular scripts/run_stage5_current_baseline_ds.sh
RESIDUAL_MODEL=polar_continuous_hybrid scripts/run_stage5_current_baseline_ds.sh
```

可通过 `IMAGE_DIR`、`OUTPUT_DIR`、`CACHE_DIR` 覆盖数据集和输出位置。

## 144928-clear DS Random 70/30

Split seed 为 `1337`，训练 60 帧，holdout 26 帧，三组 holdout 均为同一
`4174` 个点。

| Method | Holdout RMSE [px] | P95 [px] | Inlier@1px [%] | Accepted / Attempted | Objective decreased |
|---|---:|---:|---:|---:|---:|
| Pixel | **0.6300** | **1.3684** | **88.33** | 23 / 44 | 44 / 44 |
| Spherical | 0.6880 | 1.4291 | 85.75 | 22 / 44 | 44 / 44 |
| Hybrid | 0.6750 | 1.4405 | 86.99 | 29 / 44 | 43 / 44 |

最终 DS 参数：

| Method | xi | alpha | fu | fv | cu | cv |
|---|---:|---:|---:|---:|---:|---:|
| Pixel | -0.191159 | 0.615455 | 1173.07 | 1172.10 | 2242.20 | 2275.88 |
| Spherical | -0.213432 | 0.604883 | 1129.00 | 1129.51 | 2245.82 | 2280.71 |
| Hybrid | -0.219629 | 0.604822 | 1130.05 | 1129.46 | 2242.04 | 2275.59 |

新版 Spherical 已消除旧实现中 candidate objective 全部上升的问题，但在该
split 上仍基本停留在初始化焦距附近，因此当前数据不支持 Spherical 优于 Pixel
的结论。该现象应作为 residual/observability 研究问题保留，不应通过引用旧结果
或调整表格口径隐藏。

输出目录：

```text
result_may/paper_residual_ablation_20260711_144928clear_ds/pixel_only
result_may/paper_residual_ablation_20260711_144928clear_ds/tangent_plane
result_may/paper_residual_ablation_20260711_144928clear_ds/hybrid
```
