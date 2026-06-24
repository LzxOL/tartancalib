#!/bin/zsh
set -u

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib || exit 1

HOLDOUT_LEFT="image/close_dis_dataset/stereo_dataset_20260430_144928/left"
HOLDOUT_RIGHT="image/close_dis_dataset/stereo_dataset_20260430_144928/right"
LEFT_CONFIG="aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
RIGHT_CONFIG="aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
LEFT_INTRINSICS="config/mono_fisheye_calib_3_25_left-camchain.yaml"
RIGHT_INTRINSICS="config/mono_fisheye_calib_3_25_right-camchain.yaml"
STEREO_REF="config/stereo_4_2-3-camchain.yaml"
STAMP="20260622_val_close144928"

run_case() {
  local train_label="$1"
  local train_left="$2"
  local train_right="$3"
  local residual="$4"
  local out="report/stage6_residual_${residual}_${train_label}_${STAMP}"
  local cache="result/.stage6_stereo_cache_residual_${residual}_${train_label}_${STAMP}"
  mkdir -p "${out}"
  echo "[residual_cmp] start train=${train_label} residual=${residual}" | tee "${out}/job_status.txt"

  local residual_args=()
  if [[ "${residual}" == "angular" ]]; then
    residual_args=(
      --stage6-selection-ba-residual-mode spherical_tangent
      --stage6-residual-mode spherical_tangent
      --stage6-spherical-use-normalize-jacobian false
    )
  fi

  ./build/run_stage6_stereo_extrinsic \
    --left-image "${train_left}" \
    --right-image "${train_right}" \
    --test-left-image "${HOLDOUT_LEFT}" \
    --test-right-image "${HOLDOUT_RIGHT}" \
    --left-config "${LEFT_CONFIG}" \
    --right-config "${RIGHT_CONFIG}" \
    --left-intrinsics "${LEFT_INTRINSICS}" \
    --right-intrinsics "${RIGHT_INTRINSICS}" \
    --stereo-reference-camchain "${STEREO_REF}" \
    --output "${out}" \
    --cache-dir "${cache}" \
    "${residual_args[@]}" \
    > "${out}/run.log" 2>&1
  local code=$?
  echo "[residual_cmp] done train=${train_label} residual=${residual} exit=${code}" | tee -a "${out}/job_status.txt"
  return ${code}
}

run_case \
  "1444190clear" \
  "image/datatset_5_1/stereo_dataset_20260430_1444190-clear/left" \
  "image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right" \
  "pixel"
run_case \
  "1444190clear" \
  "image/datatset_5_1/stereo_dataset_20260430_1444190-clear/left" \
  "image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right" \
  "angular"
run_case \
  "134853" \
  "image/datatset_5_1/stereo_dataset_20260430_134853/left" \
  "image/datatset_5_1/stereo_dataset_20260430_134853/right" \
  "pixel"
run_case \
  "134853" \
  "image/datatset_5_1/stereo_dataset_20260430_134853/left" \
  "image/datatset_5_1/stereo_dataset_20260430_134853/right" \
  "angular"

echo "[residual_cmp] all cases finished"
