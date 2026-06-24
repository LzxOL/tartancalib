# Stage5/Stage6 Experiment Data Archive

Generated: 2026-04-26 14:32:27
Updated: 2026-04-26 19:30 strict failed-board drop baseline freeze
Root repo: /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib

This folder records the useful Stage5 backend experiments and the emerging Stage6 stereo-extrinsic experiments discussed so far. It is intentionally lightweight: CSV tables preserve the key numerical metrics and notes; source result directories remain under `result/`.

Files:
- `stage5_key_experiments.csv`: curated key experiments, including the official 140151 strict failed-board drop full baseline, repaired/diagnostic 141444 full runs, first20/first50 ablations, and runtime checks.
- `stage5_full_kalibr_std_comparison.csv`: focused backend-vs-Kalibr mean/std comparison for the two full datasets.
- `stage5_ablation_and_runtime_summary.csv`: ablation/runtime subset with deltas against the available first20 gate-off baseline where applicable.
- `stage6_stereo_experiment_summary/`: Stage6 fixed-intrinsics stereo experiment rollup, including the current best `stage6_fixed_intrinsics_global_sparse_ba_adaptive_v1` line and earlier fixed-scale / budget-cap comparisons.
- `experiment_notes.md`: human-readable interpretation and caveats.
- `source_result_dirs.txt`: source directories used by this archive.

Important interpretation rules:
- Official frozen full baseline is now `result/stage5_backend_full_140151_kalibr_style_failed_board_drop` for dataset `right_record_20260421_140151`.
- The previous gate-off baseline `result/stage5_backend_full_gate_off_baseline—1` is retained as a historical baseline, but no longer the comparison anchor.
- The repaired 141444 result is `result/stage5_backend_full_141444_benchmark_pose_rescue_fast`.
- The strict failed-board policy comparison is recorded in `stage5_141444_failed_board_policy_comparison.csv`.
- Kalibr camchain is an external reference from `config/mono_fisheye_calib_3_25_right-camchain.yaml`; it is not a same-task Kalibr baseline for internal-AprilTag calibration.
- Strict failed-board drop follows the Kalibr-style acceptance principle: if internal target pose/regeneration fails for a board observation, the whole board observation is dropped while other boards in the same frame remain usable.
- Stage6 currently records `fixed-intrinsics stereo extrinsic` experiments only.
- Current best documented Stage6 baseline is `stage6_fixed_intrinsics_global_sparse_ba_adaptive_v1`:
  - finite graph bootstrap
  - selected subset / view selection
  - `global_sparse_ba`
  - `adaptive_independent_side_cap`
  - `ba_single_camera_only_base_scale = 0.25`
  - `ba_adaptive_single_camera_only_per_side_cap_ratio = 0.05`
- This baseline is currently frozen as the recommended experimental configuration, but is not yet forced as the default CLI / default solver path.
