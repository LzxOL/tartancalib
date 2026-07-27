#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

IMAGE_DIR="${IMAGE_DIR:-image/datatset_5_1/stereo_dataset_20260430_144928-clear/right}"
CONFIG_YAML="${CONFIG_YAML:-aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml}"
CAMERA_YAML="${CAMERA_YAML:-intrintic/catalog/current_baseline/mul-board/right/stereo-144928-clear__right__ours-baseline__ds.yaml}"
PATCH_SIZE="${PATCH_SIZE:-640}"
OUTPUT_DIR="${OUTPUT_DIR:-result_may/detect_apriltag_internal_outer_distortion_144928_right_$(date +%Y%m%d_%H%M%S)}"

cmake --build build --target detect_apriltag_internal -j 8

./build/detect_apriltag_internal \
  --image "${IMAGE_DIR}" \
  --all \
  --config "${CONFIG_YAML}" \
  --output "${OUTPUT_DIR}" \
  --outer-distortion-experiment \
  --outer-experiment-camera-yaml "${CAMERA_YAML}" \
  --outer-experiment-patch-size "${PATCH_SIZE}"

echo "Experiment output: ${OUTPUT_DIR}"
