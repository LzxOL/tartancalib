# Experiment Notes

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
