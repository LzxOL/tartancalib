# Sphere-Lattice `SS` Ablation Summary

## 1. Experiment Goal

This summary compares the current internal `sphere_lattice` pipeline under the same coarse initial camera:

- `with SS`: enable sphere seed search
- `no SS`: disable sphere seed search, so `SS = P`

The goal is to test whether `P -> SS` provides a real improvement before the final image-space subpixel step.

## 2. Dataset And Setup

- Dataset: `tartancalib/image/img_seq`
- Number of images: 12
- Groups: `image1`, `image2`, `image3`
- GT reference: use calibrated-camera run output `R_gt`
- Compared iteration: `iteration_0_initial_camera`
- Coarse initial camera:
  - `xi = -0.2`
  - `alpha = 0.6`
  - `fu = 2481.6`
  - `fv = 2481.6`
  - `cu = 2256`
  - `cv = 2256`

Config copy used in the run:

- [example_apriltag_internal.yaml](/Users/linzhaoxian/lzx-ws/project/calibr/tartancalib/result/meeting_report_20260418_ss_ablation_summary/example_apriltag_internal.yaml)

## 3. Key Result

| Setting | avg `|P - R_gt|` | avg `|SS - R_gt|` | avg `|R - R_gt|` | valid internal |
|---|---:|---:|---:|---:|
| with SS | 11.7354 px | 4.7540 px | 0.0088 px | 395 / 396 |
| no SS | 11.7354 px | 11.7354 px | 3.9354 px | 183 / 396 |

Main takeaways:

1. `SS` clearly improves the seed under the coarse initial camera.
2. The reduction from `|P - R_gt|` to `|SS - R_gt|` is about `6.98 px`.
3. Relative reduction is about `59.5%`.
4. Without `SS`, the pipeline loses many valid internal points and final `R` becomes much worse.

## 4. Materials In This Folder

### Result text files

- `with_ss_experiment_summary.txt`
- `with_ss_iter0_summary.txt`
- `no_ss_experiment_summary.txt`
- `no_ss_iter0_summary.txt`

### Representative experiment visualizations

- `gt_image1-1.png`
- `with_ss_image1-1.png`
- `no_ss_image1-1.png`
- `gt_image2-1.png`
- `with_ss_image2-1.png`
- `no_ss_image2-1.png`

These are good for side-by-side report screenshots:

- `GT` = calibrated-camera reference
- `with SS` = current proposed seed search
- `no SS` = ablation where `SS = P`

### Additional debug examples

- `internal_seed_debug_example.png`
  - Example internal `P / SS / R` visualization
- `outer_debug_example.png`
  - Example outer `C / SP` visualization

## 5. Notes For Report Writing

Recommended wording:

> Under the same coarse initial camera, enabling the proposed sphere-guided seed search reduces the average internal seed error from 11.74 px to 4.75 px relative to the calibrated-reference detection result, while also increasing valid internal detections from 183 to 395.

Important caveat:

> Here `R_gt` is not manually annotated ground truth, but the refined detection result produced by the calibrated-camera configuration.
