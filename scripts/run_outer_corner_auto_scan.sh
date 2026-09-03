#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_DIR="${IMAGE_DIR:-image/dataset_8_8/board4_8-8-14_27/images/right}"
CONFIG="${CONFIG:-aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-result_may/outer_corner_auto_scan_board4_8_8}"
NO_OUTPUT_IMAGES="${NO_OUTPUT_IMAGES:-0}"

case "$NO_OUTPUT_IMAGES" in
  0|false|off)
    OUTPUT_IMAGE_ARGS=()
    ;;
  1|true|on)
    OUTPUT_IMAGE_ARGS=(--no-output-images)
    ;;
  *)
    echo "Unsupported NO_OUTPUT_IMAGES: $NO_OUTPUT_IMAGES" >&2
    exit 2
    ;;
esac

./build/detect_apriltag_internal \
  --image "$IMAGE_DIR" \
  --config "$CONFIG" \
  --all \
  --outer-corners-only \
  --auto-tag-ids \
  "${OUTPUT_IMAGE_ARGS[@]}" \
  --output "$OUTPUT_DIR"
