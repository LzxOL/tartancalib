# Close-Calibration Pixel vs Spherical Ablation

## Protocol

- Calibration dataset: `image/close_dis_dataset/stereo_dataset_20260430_144928/right`
- Camera model: Double Sphere (`ds-none`)
- Split: random 70/30 holdout, seed `1337`
- Holdout indices: `0,6,7,8,9,13`
- Shared initialization: `0.283627,0.781634,1853.42,1852.61,2242.76,2275.2`
- Shared backend input: 11 frames, 55 board observations, 1830 points
- Shared holdout: 6 frames, 907 points
- Compared residuals: `pixel_only` and `sphere_angular`
- Hybrid is intentionally excluded.

The backend manifest is selected once by the Pixel baseline and then frozen.
Both fixed-input runs contain zero candidate batches, so selection and
acceptance cannot differ between residual modes.

## Outputs

- `summary.csv`: holdout, runtime, and final-intrinsics metrics
- `polar_holdout.csv`: holdout metrics in shared-camera polar bins
- `protocol_summary.txt`: machine-readable fairness checks
- `table_close_calibration_residual.tex`: single-column paper table
- `manifests/pixel_selected_backend.csv`: frozen frame-board input

Source run directories:

- `result_may/stage5_angular_closecalib_seed1337_fixed_pixel`
- `result_may/stage5_angular_closecalib_seed1337_fixed_spherical`

Regenerate the tables with:

```bash
python3 scripts/generate_close_calibration_residual_ablation.py
```

## Result

Spherical slightly reduces overall holdout RMSE, P95, and true angular RMSE,
but the changes are small. It slightly decreases Inlier@1px and provides
almost no gain in the >=50 degree angular bucket. This single 19-frame split
does not establish a material Spherical advantage; additional captures or
multiple fixed split seeds are required for a statistical claim.
