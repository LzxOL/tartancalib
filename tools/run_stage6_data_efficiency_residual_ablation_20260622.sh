#!/bin/zsh
set -u

cd /Users/linzhaoxian/lzx-ws/project/calibr/tartancalib || exit 1

TRAIN_LEFT="image/datatset_5_1/stereo_dataset_20260430_1444190-clear/left"
TRAIN_RIGHT="image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right"
HOLDOUT_LEFT="image/close_dis_dataset/stereo_dataset_20260430_144928/left"
HOLDOUT_RIGHT="image/close_dis_dataset/stereo_dataset_20260430_144928/right"
LEFT_CONFIG="aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
RIGHT_CONFIG="aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
LEFT_INTRINSICS="config/mono_fisheye_calib_3_25_left-camchain.yaml"
RIGHT_INTRINSICS="config/mono_fisheye_calib_3_25_right-camchain.yaml"
STEREO_REF="config/stereo_4_2-3-camchain.yaml"
SEED=20260622
STAMP="20260622_1444190clear_to_close144928"

run_case() {
  local budget="$1"
  local group="$2"
  local residual="$3"
  local out="report/stage6_data_eff_${budget}frames_${group}_${residual}_${STAMP}"
  local cache="result/.stage6_stereo_cache_data_eff_${budget}frames_${group}_${residual}_${STAMP}"
  mkdir -p "${out}"
  echo "[data_eff] start budget=${budget} group=${group} residual=${residual}" | tee "${out}/job_status.txt"

  local group_args=()
  if [[ "${group}" == "single_board" ]]; then
    group_args=(--stage6-board-masking-ablation split_pair_boards)
  fi

  local residual_args=()
  if [[ "${residual}" == "angular" ]]; then
    residual_args=(
      --stage6-selection-ba-residual-mode spherical_tangent
      --stage6-residual-mode spherical_tangent
      --stage6-spherical-use-normalize-jacobian false
    )
  fi

  ./build/run_stage6_stereo_extrinsic \
    --left-image "${TRAIN_LEFT}" \
    --right-image "${TRAIN_RIGHT}" \
    --test-left-image "${HOLDOUT_LEFT}" \
    --test-right-image "${HOLDOUT_RIGHT}" \
    --left-config "${LEFT_CONFIG}" \
    --right-config "${RIGHT_CONFIG}" \
  --models ds-none \
  --output "${out}" \
    --cache-dir "${cache}" \
    --stage6-training-pair-sample-count "${budget}" \
    --stage6-training-pair-sample-seed "${SEED}" \
    "${group_args[@]}" \
    "${residual_args[@]}" \
    > "${out}/run.log" 2>&1
  local code=$?
  echo "[data_eff] done budget=${budget} group=${group} residual=${residual} exit=${code}" | tee -a "${out}/job_status.txt"
  return ${code}
}

for budget in 10; do
  for group in single_board multiboard; do
    for residual in pixel; do
      run_case "${budget}" "${group}" "${residual}"
    done
  done
done

echo "[data_eff] all cases finished"
