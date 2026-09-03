# Experiment Notes

## Current frontend detector / measurement generation baseline

The current Stage5 intrinsics and Stage6 stereo-extrinsics baselines should be
understood as using the same frontend measurement-generation family, not as
using a plain AprilTag detector followed by backend optimization.

Baseline frontend components:

- Round1 outer-corner geometry initialization: use reliable outer board corners
  to initialize the camera model, per-frame pose, and multi-board layout/board
  poses.
- Delayed intrinsics release / joint geometry optimization: refine the
  intermediate camera model and the camera-board geometry before regenerating
  internal measurements.
- Round2 internal feature generation: use the Round1 intermediate camera model
  and optimized poses to regenerate internal corners, rebuild joint
  measurements, reevaluate residuals, and rerun selection / joint optimization.
- Geometry-prior board rescue: for missing boards, use the current multi-board
  layout and visible-board pose context only to predict a search region. The
  accepted observation still has to come from image evidence: local/spherical
  subpixel refinement, edge support, and pose-consistency checks. Pure projected
  corners are not treated as backend observations.

This frontend is part of the baseline for both:

- Stage5 mono intrinsics calibration: it supplies the final board/corner
  observations used by the intrinsics-oriented backend selection and BA.
- Stage6 stereo extrinsics calibration: fixed-intrinsics monocular frontends
  first generate the per-camera board observations, then Stage6 performs
  stereo pair-board selection and global sparse BA on those observations.

Important evaluation convention:

- `raw holdout` is useful as a diagnostic because it freezes the original
  detector-visible observation set.
- `rescue-augmented holdout` is the stricter real-use setting because it keeps
  geometry rescue enabled on validation data as well. It may report higher RMSE
  because it evaluates additional close-distance / edge / high-polar boards
  that were previously missing. This should be reported as a harder, more
  complete observation set rather than as a direct regression of calibration
  quality.

For paper claims, the frontend detector contribution should be positioned as
multi-board feature coverage and measurement-generation robustness: it aims to
recover more valid board/corner observations, especially for close-distance,
edge, high-polar, and shifted-view cases, while keeping backend acceptance and
pose-consistency checks responsible for rejecting globally inconsistent
observations.

## Official full baseline: 140151 strict failed-board drop

Source: `result/stage5_backend_full_140151_kalibr_style_failed_board_drop`

This is the frozen `stage5_backend_auto_v1` full baseline after upgrading the acceptance policy to strict failed-board drop: auto-init, round2, delayed intrinsics release, residual sanity gate on, board pose-fit gate off, internal pose rescue off, fast runtime mode, and strict board-observation acceptance on.

Key result: backend holdout overall RMSE is 6.62787 px, with holdout std residual `(x, y) = (5.74781, 3.29464)`. Under the same Stage5 evaluator, the external Kalibr reference has holdout overall 6.68959 px and std `(5.84711, 3.23206)`. The previous gate-off baseline is retained as a historical reference but is no longer the comparison anchor.

Strict failed-board drop means: if internal target pose/regeneration fails for one board observation, that whole board observation is dropped, while other boards in the same frame remain usable. This follows the Kalibr-style acceptance principle and avoids silently retaining outer-only observations from a board whose internal generation failed.

## Repaired full 141444 result

Source: `result/stage5_backend_full_141444_benchmark_pose_rescue_fast`

This run records the wide-FOV benchmark pose rescue fix. The issue was not global intrinsics or board ID; it was edge-board outer-pose refit in Stage5 benchmark/visualization. For board4/board5 near the fisheye edge, the Kalibr-style 80-degree ray filter could leave too few points and trigger a bad pinhole fallback. Later experiments showed that strict failed-board drop is preferable to accepting rescued internal observations as the official policy.

Key result after repair: backend holdout overall RMSE is 7.04054 px, outer-only 4.21086 px, internal-only 7.3607 px. Holdout std `(x, y) = (5.84779, 3.86171)`, better than the external Kalibr reference `(10.3079, 4.74336)`.

## New 4-27 datasets

Sources:
- `result/stage5_backend_full_20260427_191538_strict_baseline`
- `result/stage5_backend_full_140151_strict_baseline_check`

Two additional 2026-04-27 datasets were evaluated with the same strict failed-board-drop baseline. The `191538` sequence is notably cleaner than the previous full datasets: backend training overall RMSE is `1.40492` px and holdout overall is `2.93990` px, essentially tied with the Kalibr reference `2.93917` px. Training residual std is approximately `(0.98, 0.99)` and holdout std `(2.02, 2.12)`, making it the cleanest dataset in the current local record.

The `192347` sequence is moderately harder than `191538` but still clearly better than the original 140151/141444 full datasets. Backend holdout overall RMSE is `4.20053` px versus Kalibr `4.63047` px, with holdout std `(3.55, 2.17)` better than Kalibr `(3.91, 2.38)`. One bookkeeping caveat: this result was written into the reused directory `result/stage5_backend_full_140151_strict_baseline_check`, so the directory name does not match the actual dataset. The new summary CSV records the true source dataset explicitly to avoid future confusion.

## Multi-camera-model support on 2026-04-27 / 191538

Source table: `exp_data/stage5_camera_model_full_comparison_20260427_191538.csv`

The Stage5 pipeline is no longer DS-only. `ds-none`, `eucm-none`, and `pinhole-equi` have all been wired into auto-init, frontend, backend, and summary export. A critical backend bug was fixed for `eucm-none`: during delayed intrinsics release, the backend was incorrectly activating a zero-dimensional `NoDistortion` design variable. After fixing this, EUCM can run end-to-end.

Current status on dataset `image/dataset_4_27/right_record_20260427_191538`:

- `ds-none`: official baseline, stable, holdout overall `2.93990` px.
- `eucm-none`: full run succeeds, training fit is stronger, but holdout overall `2.99109` px is still slightly worse than DS on this dataset.
- `pinhole-equi`: auto-init succeeds, but the current branch is not yet stable enough on this dataset; benchmark reports `optimized residuals did not improve`, so it should not be treated as a baseline candidate yet.

Interpretation: EUCM is now a real candidate model family, not a stub, but it has not yet shown a stable enough holdout win to replace the DS baseline. Pinhole-equi remains an integration-complete but not yet robust branch.

## Failed-board policy comparison

Source: `stage5_141444_failed_board_policy_comparison.csv`

On 141444 full, strict failed-board drop improved holdout overall RMSE from 7.04054 to 7.02680 and holdout outer-only RMSE from 4.21086 to 3.83590 without enabling rescue. Rescue-gate8 accepted two previously failed boards but worsened holdout overall/internal RMSE and residual standard deviation. On 140151 full, strict was essentially neutral relative to the previous gate-off baseline.

## Historical bad 141444 diagnostic

Sources: `result/stage5_backend_full_gate_off_baseline` and `result/stage5_backend_full_141444_gate_off_research_viz`

These are kept intentionally because they demonstrate the failure mode: individual board4/board5 observations produced hundreds of pixels of outer-pose RMSE while other boards in the same image were normal. They should not be used as final method performance after the benchmark/refit fix.

## Ablation takeaways

The available local ablations are primarily first20/first50 runs. They are useful for qualitative algorithm decisions, not as final paper-scale full-dataset proof unless rerun on full.

High-level conclusions retained so far:
- Round2 ablation did not show a decisive large full-scale benefit in early runs; still useful to report as a controlled component check.
- Pose-only / no intrinsics release is clearly weaker in earlier ablations and should be framed as supporting delayed intrinsics release.
- Board pose-fit gate has been demoted to a debug/stability option rather than a paper-mainline contribution.
- Residual sanity gate remains a practical robustness protection, but future paper claims should avoid over-centering on hand-tuned gates.

## Recommended use

For a paper table, use `stage5_full_kalibr_std_comparison.csv` for the two full datasets and `stage5_key_experiments.csv` for traceability. For ablation text, cite `stage5_ablation_and_runtime_summary.csv` but prefer rerunning selected ablations on full if they become central claims.

## Stage6 stereo extrinsic rollout

Source table: `exp_data/stage6_stereo_experiment_summary/stage6_stereo_experiment_summary.csv`

Stage6 is now tracked as a separate experimental line from Stage5. The current best branch is `stage6_fixed_intrinsics_global_sparse_ba_adaptive_v1`, built on top of two frozen monocular Stage5 frontends, a Kalibr-inspired finite graph bootstrap, selected-subset stereo view selection, and fixed-intrinsics `global_sparse_ba`.

Current archived subsets / comparisons:

- `first10`: `result/stage6_stereo_20260430_134853_first10_v2_smoke_fast`
- `first20`: `result/stage6_stereo_20260430_134853_first20_v2_smoke_fast`
- weighted / budget / adaptive comparisons on `first10` and `first20`

Key interpretation points so far:

- The previous infinite graph propagation bug is fixed. Both archived subsets terminate after `3` propagation iterations, and both stop through `stopped_by_no_progress=1` rather than by hitting the iteration cap.
- The bootstrap is now structurally interpretable. `first10` initializes `8/10` training pairs; `first20` initializes `11/20` training pairs, with the remaining failures split into graph-unreachable pairs and one reachable-but-failed numeric initialization.
- The `global_sparse_ba` line is now active and no longer blocked on single-camera-only weighting bugs.
- The current best weighting policy is `adaptive_independent_side_cap`, which improves `cam0/cam1` balance across multiple datasets more consistently than:
  - `fixed_scale = 0.25`
  - `fixed_scale = 0.10`
  - `per_side_budget_cap`
- The estimated stereo extrinsic remains numerically stable under the current best adaptive weighting, without obvious baseline / rotation drift in the tested subset runs.

Current recommendation:

- Treat `stage6_fixed_intrinsics_global_sparse_ba_adaptive_v1` as the current best documented Stage6 baseline.
- Do not yet promote it to the default solver path until `first50` confirms:
  - finite graph termination
  - stable initialized-pair counts
  - improved `cam0/cam1` balance
  - non-drifting `T_cam1_cam0`
  - no harmful regression in shared residual
- If `first50` remains stable, the next engineering step is `quality-selected full`, not unfiltered full.

## Stage5 close-distance edge stress test

Sources:

- `result_may/stage5_newbaseline_144419_right_val_close_144419_closeedge_diag`
- `result_may/stage5_newbaseline_144928_right_val_close_144928_right_closeedge_diag`
- `result_may/stage5_force_include_close_edge_board5_144928_right_val_close_144928_right`
- `result_may/stage5_newbaseline_192347_right_val_close_192347_closeedge_diag`
- `result_may/outer_subpix_polar_boost_proxy_144928_to_close144928`
- `result_may/outer_subpix_polar_boost_proxy_144419_to_144928`

This experiment checks whether poor close-distance edge validation is caused by Stage5 generalization failure, or by intrinsically difficult board/view geometry. The key diagnostic is board-level pose-only refit on the close validation set, with direct comparison between the final backend DS camera and the fixed Kalibr reference.

Summary table:

| Train -> close validation | Backend holdout overall | Kalibr holdout overall | Backend outer | Backend internal | Interpretation |
|---|---:|---:|---:|---:|---|
| `144419 -> close144419` | `1.08211` | `1.17843` | `0.109971` | `1.1542` | Close validation is handled well; backend is slightly better than Kalibr. |
| `144928 -> close144928` | `10.6588` | `10.2290` | `6.75481` | `11.0856` | Hard close-edge failure; backend and Kalibr both fail, dominated by board5. |
| `192347 -> close192347` | `10.2514` | `10.4448` | `3.57713` | `10.8436` | Hard close-edge failure; backend and Kalibr both fail, dominated by board5. |

Update after outer subpixel refinement diagnosis:

The `144928 -> close144928` hard failure was largely caused by the effective outer subpixel window being clamped by the verification ROI. After removing that clamp and enabling close-edge polar/proxy-aware subpixel boost, the same close holdout improved sharply:

| Run | Backend holdout overall | Kalibr holdout overall | Backend outer | Backend internal | Notes |
|---|---:|---:|---:|---:|---|
| `outer_subpix_polar_boost_proxy_144928_to_close144928` | `0.99102` | `1.41363` | `0.239045` | `1.05265` | `outer_subpix_scale=0.35`, close-edge boost `x1.4`, `88/2576` outer corners boosted. |
| `outer_subpix_polar_boost_proxy_144419_to_144928` | `2.26618` | `5.60878` | `0.938152` | `2.39328` | Cross-dataset sanity; `8/1708` outer corners boosted. |

This revises the earlier interpretation: close-edge board5 remains an important stress case, but it is not simply irreducible geometry. A large part of the failure can be addressed in the frontend by allowing a larger adaptive outer-corner subpixel window specifically for high-polar, large-area, near-border boards.

For `144928 -> close144928`, a force-include test was run using `exp_data/close_edge_board5_force_include_candidates.csv`. The seven close-edge board5 observations were all matched and attempted, but all were rejected by the incremental short-backend test:

| Frame label | Board | Trial RMSE | Global RMSE delta | Outer RMSE delta | Internal RMSE delta | Decision |
|---|---:|---:|---:|---:|---:|---|
| `000007_right_431303195240_mono8` | 5 | `6.89311` | `+0.185754` | `+1.24598` | `+0.0297072` | rejected |
| `000009_right_433603197080_mono8` | 5 | `7.09748` | `+0.181107` | `+1.19814` | `+0.0351364` | rejected |
| `000010_right_434603192000_mono8` | 5 | `8.93082` | `+0.298786` | `+1.74147` | `+0.0431819` | rejected |
| `000090_right_522303194080_mono8` | 5 | `9.36993` | `+0.320093` | `+1.85821` | `+0.0357769` | rejected |
| `000091_right_523303192000_mono8` | 5 | `9.30625` | `+0.314275` | `+1.82003` | `+0.0399062` | rejected |
| `000092_right_524403198240_mono8` | 5 | `11.7358` | `+0.524088` | `+2.6048` | `+0.0632888` | rejected |
| `000093_right_525603195160_mono8` | 5 | `8.13481` | `+0.284448` | `+1.69406` | `+0.0379528` | rejected |

The rejection reason is important: these observations were not missed by candidate scoring or board/frame caps. They were explicitly attempted, and the short-backend trial showed that adding them would worsen optimization, especially the outer residual.

Additional residual-vector diagnostics were generated under:

`result_may/stage5_force_include_close_edge_board5_144928_right_val_close_144928_right/close_edge_board5_outer_corner_vector_diagnostics`

The backend DS and Kalibr reference residual vectors point in almost the same direction on the failing board5 observations. Per-frame backend/Kalibr residual-direction cosine is near `1.0` for the seven tested frames, which means both models explain the same observations poorly in nearly the same way. This supports the interpretation that these are difficult close-edge board/view cases, not simply a Stage5-only generalization bug.

Takeaway:

- Close-distance validation is not inherently bad: `144419 -> close144419` works well.
- The bad close results are concentrated in specific board5 edge cases.
- For `144928` and `192347`, Kalibr also fails on the same close-edge board5 cases, so these should be reported as stress cases rather than used as direct evidence that Stage5 fails to generalize.
- The current trial-backend selection is behaving correctly by rejecting these observations instead of forcing them into the backend.
