# Frozen Stage5 Baseline (v3)

`stage5_backend_frozen_v3_recovery_cache` is the default Stage5 protocol for
image-based calibration. It freezes the validated recovery and cache behavior
without freezing a dataset's physical board topology.

## Included behavior

- Direct multi-scale outer-tag detection and existing outer-corner validation.
- Camera-aware DS sphere-patch rescue after provisional outer-only camera
  initialization, followed by camera reinitialization when observations are
  recovered.
- Zero-detection atlas enabled by default. A frame with no direct decode still
  receives an exact-ID/Hamming-0 recovery attempt.
- Geometry-prior outer seeds and geometry-guided tag likelihood enabled for
  missing boards when a bootstrap/scene pose prior exists. Projected-only
  corners are never accepted as observations.
- Internal pose rescue and image-validated internal regeneration.
- One dataset-owned `--cache-dir` containing both `outer_detection_final` and
  `internal_refinement` stage artifacts. Configuration/state hashes isolate
  only the affected layer when an algorithm changes.

## Dataset-specific configuration

The YAML file remains responsible for the actual board/tag IDs and optional
costlier recovery coverage. For example, the four-board dataset uses IDs
`[1, 3, 4, 5]` and explicitly enables the extended sphere-patch atlas. Do not
copy that topology into a different dataset's baseline command.

## Explicit ablations

Use `--stage5-disable-camera-aware-outer-rescue` or
`--stage5-disable-camera-aware-sphere-patch-zero-detection` only for an
ablation. Such runs receive a deterministic protocol suffix and do not share
the frozen baseline's effective recovery configuration.
