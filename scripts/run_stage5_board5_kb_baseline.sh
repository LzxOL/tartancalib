#!/usr/bin/env bash
set -euo pipefail

# Canonical Stage5 Board5 KB profile. Override paths through environment
# variables so each experiment keeps a distinct output and cache namespace.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_DIR="${IMAGE_DIR:-image/board5_8_20/board5_8_20-4}"
CONFIG="${CONFIG:-aslam_cv/aslam_cameras_april/config/board5_8_15_apriltag_internal.yaml}"
KALIBR_CAMCHAIN="${KALIBR_CAMCHAIN:-intrintic/catalog/canonical_aprilgrid/right/aprilgrid_8-22_3_right_____/kb/aprilgrid_8-22_3_right__kalibr__kb.yaml}"
TARTANCALIB_CAMCHAIN="${TARTANCALIB_CAMCHAIN:-}"
BABELCALIB_CAMCHAIN="${BABELCALIB_CAMCHAIN:-}"
BASALT_CAMCHAIN="${BASALT_CAMCHAIN:-}"
OUTPUT_DIR="${OUTPUT_DIR:-result_may/stage5_board5_8-20-4_right_kb_baseline_split1337}"
CACHE_DIR="${CACHE_DIR:-result_may/.stage5_board5_8-20-4_right_kb_baseline_split1337_cache}"
SPLIT_SEED="${SPLIT_SEED:-1337}"
PROGRESSIVE_SEED="${PROGRESSIVE_SEED:-0}"
FROZEN_HOLDOUT_FRONTEND="${FROZEN_HOLDOUT_FRONTEND:-1}"

case "$PROGRESSIVE_SEED" in
  0|false|off)
    PROGRESSIVE_SEED_ARG=""
    ;;
  1|true|on)
    PROGRESSIVE_SEED_ARG="--stage5-model-aware-progressive-seed"
    ;;
  *)
    echo "Unsupported PROGRESSIVE_SEED: $PROGRESSIVE_SEED" >&2
    echo "Expected 0/false/off or 1/true/on." >&2
    exit 2
    ;;
esac

case "$FROZEN_HOLDOUT_FRONTEND" in
  0|false|off)
    FROZEN_HOLDOUT_FRONTEND_ARG=""
    ;;
  1|true|on)
    FROZEN_HOLDOUT_FRONTEND_ARG="--stage5-external-holdout-self-frontend-prepass"
    ;;
  *)
    echo "Unsupported FROZEN_HOLDOUT_FRONTEND: $FROZEN_HOLDOUT_FRONTEND" >&2
    echo "Expected 0/false/off or 1/true/on." >&2
    exit 2
    ;;
esac

COMMAND=(
  ./build/run_stage5_backend
  --image "$IMAGE_DIR"
  --config "$CONFIG"
  --output "$OUTPUT_DIR"
  --cache-dir "$CACHE_DIR"
  --kalibr-camchain "$KALIBR_CAMCHAIN"
  --models pinhole-equi
  --runtime-mode research
  --split-mode random_holdout_ratio
  --holdout-ratio 0.30
  --split-seed "$SPLIT_SEED"
  --stage5-init-shared-focal 0
  --stage5-enable-frozen-recovery-baseline
  --stage5-enable-model-aware-information-coreset
  --stage5-model-aware-seed-layout-alignment 0
  --stage5-model-aware-candidate-pose-prefit 0
  --stage5-enable-global-scene-state-consistency-audit
  --stage5-enable-polar-angle-diagnostics
  --stage5-skip-heavy-overlays
  --stage5-enable-progress
  --stage5-progress-interval 10
)

if [[ -n "$FROZEN_HOLDOUT_FRONTEND_ARG" ]]; then
  COMMAND+=("$FROZEN_HOLDOUT_FRONTEND_ARG")
fi
if [[ -n "$PROGRESSIVE_SEED_ARG" ]]; then
  COMMAND+=("$PROGRESSIVE_SEED_ARG")
fi
if [[ -n "$TARTANCALIB_CAMCHAIN" ]]; then
  COMMAND+=(--reference-intrinsics-yaml "tartancalib:$TARTANCALIB_CAMCHAIN")
fi
if [[ -n "$BABELCALIB_CAMCHAIN" ]]; then
  COMMAND+=(--reference-intrinsics-yaml "babelcalib:$BABELCALIB_CAMCHAIN")
fi
if [[ -n "$BASALT_CAMCHAIN" ]]; then
  COMMAND+=(--reference-intrinsics-yaml "basalt:$BASALT_CAMCHAIN")
fi
COMMAND+=(--all)

"${COMMAND[@]}"
