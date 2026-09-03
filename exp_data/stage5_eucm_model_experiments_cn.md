# Stage5 EUCM 模型实验汇总

## 1. 说明

本文档汇总 Stage5 中 `EUCM-none` 分支的 full 数据集实验结果。对应配置文件为：

- `aslam_cv/aslam_cameras_april/config/example_apriltag_internal_eucm_none.yaml`

与 KB 文档类似，这里的 `Kalibr` 仍然表示外部棋盘格 camchain reference 在同一 evaluator 下的评估结果，因此这些结果依然属于**跨模型参考对比**。但和 KB 分支相比，EUCM 的整体稳定性明显更好。

所有本轮 EUCM 结果都输出在：

- `result_may/`

并且每组都已经包含：

- `worst_reprojection_frame_backend_vs_kalibr.png`
- `worst_reprojection_board_backend_vs_kalibr.png`
- `worst_reprojection_summary.txt`

## 2. 原始 full 数据集成对 cross-dataset external validation

对应目录：

- `result_may/stage5_backend_full_20260421_140151_external_val_141444_eucm_none_refit_diag`
- `result_may/stage5_backend_full_20260421_141444_external_val_140151_eucm_none_refit_diag`
- `result_may/stage5_backend_full_20260427_191538_external_val_192347_eucm_none_refit_diag`
- `result_may/stage5_backend_full_20260427_192347_external_val_191538_eucm_none_refit_diag`

### 2.1 主结果表

| 训练数据集 -> 外部验证数据集 | Backend 优化后 RMSE | 训练集 Backend / Kalibr | 外部验证 Backend / Kalibr | 外部验证 Backend x/y 标准差 | 外部验证 Kalibr x/y 标准差 |
|---|---:|---:|---:|---:|---:|
| 20260421_140151 -> 20260421_141444 | 5.2787 | 6.3663 / 6.4474 | 7.4263 / 7.3025 | 6.1335 / 4.1556 | 6.0525 / 4.0431 |
| 20260421_141444 -> 20260421_140151 | 4.6442 | 7.0141 / 8.0110 | 6.5874 / 6.6777 | 5.3532 / 3.8202 | 5.4323 / 3.8602 |
| 20260427_191538 -> 20260427_192347 | 0.9002 | 0.9215 / 0.9546 | 4.1916 / 4.2636 | 3.5764 / 2.1399 | 3.6142 / 2.1848 |
| 20260427_192347 -> 20260427_191538 | 1.7077 | 4.2480 / 4.4169 | 3.0980 / 3.1038 | 2.3111 / 2.0497 | 2.3264 / 2.0471 |

### 2.2 pose-only / internal 诊断

| 训练数据集 -> 外部验证数据集 | pose-only success rate Backend / Kalibr | outer-fit RMSE Backend / Kalibr | internal evaluation RMSE Backend / Kalibr | 结论 |
|---|---:|---:|---:|---|
| 20260421_140151 -> 20260421_141444 | 100% / 100% | 4.1995 / 3.9526 | 7.7687 / 7.6521 | Kalibr 略优 |
| 20260421_141444 -> 20260421_140151 | 100% / 100% | 2.4545 / 2.4517 | 6.9683 / 7.0656 | Backend 基本打平并略优 |
| 20260427_191538 -> 20260427_192347 | 100% / 100% | 1.9752 / 2.0805 | 4.4091 / 4.4804 | Backend 小幅优于 Kalibr |
| 20260427_192347 -> 20260427_191538 | 100% / 100% | 1.1640 / 1.1825 | 3.2748 / 3.2801 | Backend 小幅优于 Kalibr |

### 2.3 解读

这一组结果说明：

- EUCM 分支在原始四组 full 数据集上是稳定可用的。
- `20260427` 这两组依然最健康，外部验证上继续保持对 Kalibr reference 的小幅优势。
- `20260421_140151 -> 20260421_141444` 这组略差于 Kalibr，但差距远小于 KB 分支同组的明显失稳。
- 总体上，EUCM 的表现明显比 KB 更稳，更接近“可进入主对比候选”的状态。

## 3. `dataset_5_1` stereo full 数据集 right 相机结果

对应目录：

- `result_may/stage5_backend_full_20260430_134853_right_external_val_144419_right_eucm_none_refit_diag`
- `result_may/stage5_backend_full_20260430_144419_right_external_val_144928_right_eucm_none_refit_diag`
- `result_may/stage5_backend_full_20260430_144928_right_external_val_134853_right_eucm_none_refit_diag`

### 3.1 主结果表

| 训练数据集 -> 外部验证数据集 | Backend 优化后 RMSE | 训练集 Backend / Kalibr | 外部验证 Backend / Kalibr | 外部验证 Backend x/y 标准差 | 外部验证 Kalibr x/y 标准差 |
|---|---:|---:|---:|---:|---:|
| 20260430_134853 right -> 20260430_144419 right | 0.8204 | 0.8379 / 0.8768 | 2.0987 / 2.1126 | 1.8411 / 0.9965 | 1.8379 / 1.0109 |
| 20260430_144419 right -> 20260430_144928 right | 0.6520 | 0.6777 / 0.7462 | 3.8952 / 3.7792 | 3.0225 / 2.4406 | 2.9353 / 2.3740 |
| 20260430_144928 right -> 20260430_134853 right | 1.9817 | 4.8826 / 5.3891 | 3.3948 / 3.4087 | 3.0747 / 1.3734 | 3.0835 / 1.3892 |

### 3.2 pose-only / internal 诊断

| 训练数据集 -> 外部验证数据集 | pose-only success rate Backend / Kalibr | outer-fit RMSE Backend / Kalibr | internal evaluation RMSE Backend / Kalibr | 结论 |
|---|---:|---:|---:|---|
| 20260430_134853 right -> 20260430_144419 right | 100% / 100% | 0.5996 / 0.7251 | 2.2260 / 2.2359 | Backend 小幅优于 Kalibr |
| 20260430_144419 right -> 20260430_144928 right | 100% / 100% | 2.4698 / 2.4377 | 4.0512 / 3.9271 | Kalibr 略优 |
| 20260430_144928 right -> 20260430_134853 right | 100% / 100% | 1.1559 / 1.1549 | 3.5930 / 3.6079 | 基本打平，Backend 极小幅优于 Kalibr |

### 3.3 解读

这一组三个 `right` 结果说明：

- EUCM 在 `134853 -> 144419` 和 `144928 -> 134853` 上都保持了小幅优势或基本打平。
- `144419 -> 144928` 仍然是更难的一组，在 EUCM 下也没有明显改善，仍略差于 Kalibr reference。
- 这说明 EUCM 没有神奇地解决所有泛化难点，但整体走势仍比 KB 更稳健。

## 4. 综合结论

当前 EUCM 模型实验最稳妥的结论是：

- `EUCM-none` 已经具备完整、稳定的 full 数据集运行能力。
- 它在大多数 cross-dataset external validation 上与 Kalibr reference 持平或小幅更优。
- 与 KB 分支相比，EUCM 的稳定性和跨数据集一致性明显更好。
- 但它仍然没有形成“显著优于 DS baseline / Kalibr DS reference”的压倒性优势。
- 因此，EUCM 目前最适合被视为**强候选模型族**：值得继续保留并与 DS baseline 并行观察，但还不适合直接替换 DS 成为唯一正式主线。
