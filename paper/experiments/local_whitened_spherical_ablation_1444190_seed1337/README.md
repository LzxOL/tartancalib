# Local-Whitened Spherical Ablation

This experiment compares Pixel, the existing tangent-plane Spherical
residual, and a locally covariance-whitened Spherical residual on
`stereo_dataset_20260430_1444190-clear/right`.

All variants use the same 70/30 split (`seed=1337`), Stage5 initialization,
26-frame/130-board fixed backend manifest, and 2688 held-out points. Hybrid is
not included.

The local-whitening variant uses:

```text
pixel_sigma_px = 1.0
covariance_damping = 1e-12
min_sigma_rad = 1e-6
max_whitening_weight = 1e5
```

The implementation successfully whitened all 4330 backend angular residuals,
with no covariance failures or weight clamps. It did not improve the overall
or high-polar holdout metrics. This is consistent with the first-order
interpretation that whitening angular residuals by pixel-noise propagation
makes the objective approach a pixel-domain Mahalanobis objective rather than
increasing the influence of high-polar observations.

Files:

- `summary.csv`: overall metrics, final intrinsics, and whitening health
- `polar_holdout.csv`: shared-camera polar-bin metrics
- `protocol_summary.txt`: fairness checks and experiment configuration
- `table_local_whitening_ablation.tex`: single-column table

Regenerate with:

```bash
python3 scripts/generate_local_whitening_ablation.py
```
