# Stage5 KB 模型实验汇总

## 1. 说明

本文档单独汇总 Stage5 中 `pinhole-equi / KB-like` 分支的实验结果，避免与当前正式 baseline `DS-none` 混在一起。

需要特别强调：

- 这里的“KB 模型”指的是我们自己的 Stage5 backend 管线使用 `example_apriltag_internal_pinhole_equi.yaml` 的结果。
- 表格中的 `Kalibr` 仍然是外部棋盘格 camchain reference，在当前 evaluator 下的评估结果。
- 因此这些结果属于**跨模型参考对比**，主要用于回答“KB 分支是否可用、误差大致处于什么量级”，而不是严格意义上的同模型公平竞赛。

## 2. 原始 full 数据集成对 cross-dataset external validation

对应输出目录：

- `result/stage5_backend_full_20260421_140151_external_val_141444_kb_baseline_refit_diag`
- `result/stage5_backend_full_20260421_141444_external_val_140151_kb_baseline_refit_diag`
- `result/stage5_backend_full_20260427_191538_external_val_192347_kb_baseline_refit_diag`
- `result/stage5_backend_full_20260427_192347_external_val_191538_kb_baseline_refit_diag`

| 训练数据集 -> 外部验证数据集 | Backend 优化后 RMSE | 训练集 Backend RMSE | 外部验证 Backend / Kalibr | 外部验证 Backend x/y 标准差 | 外部验证 Kalibr x/y 标准差 |
|---|---:|---:|---:|---:|---:|
| 20260421_140151 -> 20260421_141444 | 5.9302 | 6.6919 | 17.4588 / 6.7567 | 11.3249 / 13.2734 | 5.5303 / 3.8531 |
| 20260421_141444 -> 20260421_140151 | 4.7335 | 6.6682 | 6.6005 / 6.6826 | 5.3696 / 3.8244 | 5.4339 / 3.8707 |
| 20260427_191538 -> 20260427_192347 | 0.8940 | 0.9142 | 3.9603 / 3.9764 | 3.3737 / 2.0467 | 3.3703 / 2.0654 |
| 20260427_192347 -> 20260427_191538 | 2.0310 | 3.9286 | 2.3703 / 2.3447 | 1.8504 / 1.4580 | 1.8490 / 1.4375 |

对应 pose-only / internal 诊断：

| 训练数据集 -> 外部验证数据集 | pose-only success rate Backend / Kalibr | outer-fit RMSE Backend / Kalibr | internal evaluation RMSE Backend / Kalibr | 结论 |
|---|---:|---:|---:|---|
| 20260421_140151 -> 20260421_141444 | 100% / 100% | 15.7971 / 3.7211 | 17.6758 / 7.0739 | 明显退化，KB 配置在这组上失稳 |
| 20260421_141444 -> 20260421_140151 | 100% / 100% | 2.4325 / 2.4332 | 6.9843 / 7.0726 | 基本打平，Backend 略优 |
| 20260427_191538 -> 20260427_192347 | 100% / 100% | 1.7914 / 1.8552 | 4.1698 / 4.1833 | 小幅优于 Kalibr，和 DS baseline 同量级 |
| 20260427_192347 -> 20260427_191538 | 100% / 100% | 1.0598 / 0.9953 | 2.4962 / 2.4722 | Kalibr 略优，差距很小 |

这一组实验说明：

- KB 分支在 `20260427` 这对较干净的数据上是可用的，并且能达到与 Kalibr reference 同量级的结果。
- 但在 `20260421_140151 -> 20260421_141444` 上出现了明显失稳，说明它的跨数据集稳定性不足。
- 因此，KB 分支目前不能替代 DS baseline 成为主线方案。

## 3. `dataset_5_1` stereo full 数据集 right 相机 KB 结果

对应输出目录：

- `result/stage5_backend_full_20260430_144928_right_external_val_134853_right_kb_baseline_refit_diag`
- `result/stage5_backend_full_20260430_134853_right_external_val_144928_right_kb_baseline_refit_diag`
- `result_may/stage5_backend_full_20260430_144419_right_external_val_134853_right_kb_baseline_refit_diag`

这两组同样属于跨模型参考对比：

- 我们的方法使用 `pinhole-equi / KB-like`；
- Kalibr reference 使用 `DS` camchain。

### 3.1 主结果表

| 训练数据集 -> 外部验证数据集 | Backend 优化后 RMSE | 训练集 Backend / Kalibr | 外部验证 Backend / Kalibr | 外部验证 Backend x/y 标准差 | 外部验证 Kalibr x/y 标准差 |
|---|---:|---:|---:|---:|---:|
| 20260430_144928 right -> 20260430_134853 right | 2.1113 | 4.1622 / 4.5440 | 3.3978 / 3.4191 | 3.1024 / 1.3337 | 3.1136 / 1.3452 |
| 20260430_134853 right -> 20260430_144928 right | 0.8160 | 0.8338 / 0.8740 | 3.7577 / 3.6698 | 2.9208 / 2.3498 | 2.8516 / 2.3049 |
| 20260430_144419 right -> 20260430_134853 right | 0.6509 | 0.6759 / 0.7464 | 3.4194 / 3.4310 | 3.1256 / 1.3606 | 3.1221 / 1.3606 |

### 3.2 pose-only / internal 诊断

| 训练数据集 -> 外部验证数据集 | pose-only success rate Backend / Kalibr | outer-fit RMSE Backend / Kalibr | internal evaluation RMSE Backend / Kalibr | 结论 |
|---|---:|---:|---:|---|
| 20260430_144928 right -> 20260430_134853 right | 100% / 100% | 1.1271 / 1.1540 | 3.5972 / 3.6188 | Backend 小幅优于 Kalibr |
| 20260430_134853 right -> 20260430_144928 right | 100% / 100% | 2.3267 / 2.3137 | 3.9135 / 3.8185 | Kalibr 略优，差距很小 |
| 20260430_144419 right -> 20260430_134853 right | 100% / 100% | 1.1052 / 1.1522 | 3.6221 / 3.6325 | Backend 极小幅优于 Kalibr |

### 3.3 解读

这三组结果比前面的 `20260421` 更平稳一些，但结论仍然一致：

- KB 分支不是完全失效，它在 `144928 -> 134853` 上可以小幅优于 Kalibr reference。
- 新补的 `144419 -> 134853` 也属于小幅正结果，外部验证、outer-fit 和 internal evaluation 都是 Backend 极小幅领先。
- 但它也没有形成稳定优势，在 `134853 -> 144928` 上反而略差。
- 所以最合理的说法不是“KB 更好”，而是“KB 分支已经可用，并能达到与 DS reference 接近的误差量级，但当前没有表现出稳定、跨数据集的一致改进”。

## 4. 总结

当前 KB 模型实验最稳妥的结论是：

- `pinhole-equi / KB-like` 分支已经打通并可运行。
- 在部分数据集上，它可以达到与 Kalibr DS reference 非常接近的结果。
- 但它没有形成稳定、普遍的提升，且在部分数据上会明显失稳。
- 因此 KB 分支目前更适合作为“已验证可用的模型族扩展”和“补充实验结果”，不适合作为替代 DS baseline 的主线方案。
