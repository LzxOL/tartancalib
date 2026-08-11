#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/ApriltagInternalDebugVisualization.hpp>
#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>
#include <aslam/cameras/apriltag_internal/JointMeasurementCuration.hpp>
#include <aslam/cameras/apriltag_internal/KalibrBenchmark.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardConsistencyDiagnostics.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/OuterOnlyCameraInitializer.hpp>
#include <aslam/cameras/apriltag_internal/PolarAngleResidualDiagnostics.hpp>
#include <aslam/cameras/apriltag_internal/PrecomputedObservationImporter.hpp>
#include <aslam/cameras/apriltag_internal/Stage5Benchmark.hpp>
#include <aslam/cameras/apriltag_internal/Stage5BackendDiagnosticWriters.hpp>
#include <aslam/cameras/apriltag_internal/Stage5CacheManifest.hpp>
#include <aslam/cameras/apriltag_internal/Stage5Runtime.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

namespace ati = aslam::cameras::apriltag_internal;
namespace fs = boost::filesystem;

std::string MatrixToCsv(const Eigen::Matrix4d& matrix);

constexpr const char kFrozenBaselineLabel[] =
    "stage5_backend_frozen_v3_recovery_cache";

enum class IntrinsicsReleaseMode {
  Delayed,
  Immediate,
  PoseOnly,
};

struct CmdArgs {
  std::string config_path;
  std::string image_path;
  std::string test_image_path;
  std::string precomputed_observations_dir;
  std::string precomputed_holdout_observations_dir;
  std::vector<std::string>
      precomputed_initialization_auxiliary_observation_dirs;
  std::string precomputed_target_mode = "auto";
  bool allow_image_training_with_precomputed_holdout = false;
  bool precomputed_init_use_all_points = false;
  std::string precomputed_initialization_point_scope = "all";
  bool external_holdout_self_frontend_prepass = false;
  bool stage5_holdout_evaluate_full_training_observations = false;
  bool stage5_frontend_only = false;
  std::string output_path;
  std::string kalibr_camchain_yaml;
  std::vector<std::string> reference_intrinsics_specs;
  std::string kalibr_training_split_signature;
  std::string camera_init_mode_override;
  std::string stage5_models = "ds-none";
  std::string stage5_init_refine_mode = "kalibr_outer_lm";
  std::string stage5_init_selection_scorer =
      "pose_marginalized_principal";
  bool stage5_enable_init_principal_profile = false;
  double stage5_init_principal_profile_radius_px = 10.0;
  bool stage5_enable_init_fixed_layout_diagnostic = false;
  bool stage5_enable_init_board_jackknife_diagnostic = false;
  bool stage5_enable_init_coverage_weighted_diagnostic = false;
  bool stage5_init_near_tie_prefer_lower_focal = false;
  double stage5_init_near_tie_relative_objective_tolerance = 0.0;
  bool stage5_camera_aware_outer_rescue = true;
  bool stage5_camera_aware_outer_rescue_zero_detection_frames = true;
  bool stage5_camera_aware_outer_rescue_zero_detection_frames_set = true;
  // Explicitly select the frozen recovery bundle used by the canonical
  // baseline command. This keeps future default changes from silently
  // disabling a recovery algorithm that has already been validated.
  bool stage5_enable_frozen_recovery_baseline = false;
  std::string experiment_tag;
  std::string cache_dir;
  bool all = false;
  int reference_board_id = 1;
  bool include_internal_points = true;
  int intrinsics_release_iteration = 3;
  int second_pass_intrinsics_release_iteration = 1;
  std::string split_mode = "random_holdout_ratio";
  bool stage5_no_holdout = false;
  double holdout_ratio = 0.30;
  unsigned int split_seed = 1337;
  int holdout_stride = 5;
  int holdout_offset = 0;
  bool disable_second_pass = false;
  IntrinsicsReleaseMode intrinsics_release_mode = IntrinsicsReleaseMode::Delayed;
  bool disable_residual_sanity_gate = false;
  bool enable_board_pose_fit_gate = false;
  std::string stage5_selection_mode = "baseline";
  double stage5_selection_residual_sanity_factor = 2.5;
  double stage5_selection_max_board_observation_rmse = 25.0;
  double stage5_selection_kalibr_style_outlier_sigma = 4.0;
  double stage5_selection_kalibr_style_min_abs_threshold_px = 1.0;
  int stage5_selection_kalibr_style_min_views_before_filter = 20;
  bool stage5_enable_trial_backend_frame_board_selection = true;
  std::string stage5_trial_backend_selection_mode = "kalibr_style_batch";
  std::string stage5_trial_backend_selection_budget_mode;
  bool stage5_trial_backend_selection_budget_mode_set = false;
  std::string stage5_trial_backend_selection_candidate_order;
  bool stage5_trial_backend_selection_candidate_order_set = false;
  std::string stage5_trial_backend_selection_info_gain_proxy_mode =
      "intrinsics_jacobian";
  std::string stage5_trial_backend_selection_batch_granularity =
      "frame";
  std::string stage5_trial_backend_selection_acceptance_policy =
      "kalibr_information_gain";
  double stage5_trial_backend_selection_mi_tol = 0.2;
  bool stage5_trial_backend_selection_mi_tol_set = false;
  double stage5_trial_backend_selection_rank_threshold = 1e-6;
  double stage5_checkerboard_huber_delta_pixels = 1.5;
  bool stage5_checkerboard_outlier_filter_enabled = true;
  double stage5_checkerboard_outlier_sigma = 4.0;
  double stage5_checkerboard_min_inlier_ratio = 0.5;
  int stage5_checkerboard_min_retained_points = 8;
  unsigned int stage5_trial_backend_selection_candidate_shuffle_seed = 0;
  bool stage5_trial_backend_selection_candidate_shuffle_seed_set = false;
  bool stage5_trial_backend_selection_incremental = true;
  bool stage5_trial_backend_selection_carry_accepted_trial_state = true;
  bool stage5_trial_backend_selection_optimize_intrinsics = true;
  bool stage5_trial_backend_selection_delayed_intrinsics_release = true;
  int stage5_trial_backend_selection_intrinsics_release_iteration = 1;
  bool stage5_trial_backend_selection_persistent_intrinsics_anchor_prior = false;
  bool stage5_trial_backend_selection_persistent_fix_board_layout = false;
  double stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_xi_alpha = 0.0;
  double stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_focal = 0.0;
  double stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_principal = 0.0;
  double stage5_trial_backend_selection_persistent_max_focal_relative_step = 0.0;
  double stage5_trial_backend_selection_persistent_max_principal_step_px = 0.0;
  double stage5_trial_backend_selection_persistent_max_xi_alpha_step = 0.0;
  int stage5_trial_backend_selection_max_iterations = 5;
  int stage5_trial_backend_selection_max_candidate_additions = 20;
  double stage5_trial_backend_selection_adaptive_budget_ratio = 0.10;
  int stage5_trial_backend_selection_adaptive_budget_min = 20;
  int stage5_trial_backend_selection_adaptive_budget_max = 120;
  int stage5_trial_backend_selection_runtime_safety_ceiling = 1000;
  double stage5_trial_backend_selection_outlier_sigma = 4.0;
  double stage5_trial_backend_selection_min_abs_threshold_px = 1.0;
  double stage5_trial_backend_selection_max_threshold_px = 25.0;
  double stage5_trial_backend_selection_accept_max_global_rmse_increase_px = 0.02;
  double stage5_trial_backend_selection_accept_max_outer_rmse_increase_px = 0.05;
  double stage5_trial_backend_selection_accept_max_internal_rmse_increase_px = 0.05;
  double stage5_trial_backend_selection_min_candidate_score = 3.2;
  double stage5_trial_backend_selection_min_coverage_gain = 2.5;
  bool stage5_trial_backend_selection_use_consistency_score = false;
  double stage5_trial_backend_selection_consistency_translation_sigma_mm = 3.0;
  double stage5_trial_backend_selection_consistency_rotation_sigma_deg = 2.0;
  double stage5_trial_backend_selection_consistency_penalty_weight = 1.0;
  double stage5_trial_backend_selection_consistency_max_translation_error_mm = -1.0;
  double stage5_trial_backend_selection_consistency_max_rotation_error_deg = -1.0;
  double stage5_trial_backend_selection_consistency_max_local_outer_rmse_px = -1.0;
  int stage5_trial_backend_selection_max_accepted_per_board = 0;
  int stage5_trial_backend_selection_max_accepted_per_frame = 0;
  bool stage5_trial_backend_selection_frame_cohesion = true;
  int stage5_trial_backend_selection_frame_cohesion_max_companions = 0;
  double stage5_trial_backend_selection_frame_cohesion_min_candidate_score = 3.2;
  int stage5_trial_backend_selection_min_keep_per_board = 5;
  std::string stage5_trial_backend_selection_force_include_frame_board_list;
  std::string stage5_trial_backend_selection_seed_frame_board_list;
  bool stage5_trial_backend_selection_force_include_list_is_exact_input = false;
  bool strict_board_observation_acceptance = false;
  bool preserve_frame_board_cohesion = false;
  bool ignore_image_evidence_min_quality = false;
  bool force_internal_seed_from_prediction = false;
  bool bypass_internal_seed_filters = false;
  std::string internal_corner_filter_mode = "sigma";
  double internal_corner_filter_max_reproj_error = -1.0;
  double internal_corner_filter_quality_min = 0.35;
  double internal_corner_filter_quality_relaxation_px = 1.0;
  double internal_corner_filter_adaptive_min_threshold_px = 1.0;
  std::string backend_polar_angle_weight_mode = "none";
  std::string backend_polar_angle_weight_bin_edges = "0,30,50,70,85,100";
  std::string backend_polar_angle_weight_fixed_bin_scales = "1.0,1.0,0.75,0.5,0.25";
  double backend_polar_angle_weight_adaptive_sigma_reference_deg = 50.0;
  double backend_polar_angle_weight_adaptive_sigma_growth = 1.0;
  double backend_polar_angle_weight_min_scale = 0.25;
  std::string backend_residual_model = "image_plane";
  double backend_hybrid_angular_threshold_deg = 50.0;
  std::string backend_outer_residual_model = "image_plane";
  std::string backend_internal_residual_model = "image_plane";
  bool backend_use_point_type_residual_split = false;
  bool backend_enable_angular_auxiliary_residual = false;
  double backend_angular_auxiliary_weight = 0.0;
  bool backend_angular_auxiliary_normalized = false;
  bool backend_angular_auxiliary_apply_to_outer = true;
  bool backend_angular_auxiliary_apply_to_internal = true;
  double backend_polar_continuous_hybrid_threshold_deg = 50.0;
  double backend_polar_continuous_hybrid_temperature_deg = 10.0;
  double backend_normalized_angular_reference_sigma_px = 1.0;
  double backend_normalized_angular_min_sigma_rad = 1e-6;
  double backend_normalized_angular_max_weight_scale = 1.0e8;
  double backend_pixel_residual_weight = 1.0;
  double backend_chordal_residual_weight = 1.0;
  bool enable_hybrid_pixel_ray_final_refinement = false;
  bool enable_large_intrinsic_perturbation = false;
  std::string large_intrinsic_perturbation_profile;
  double large_intrinsic_perturbation_scale = 1.0;
  bool large_intrinsic_perturbation_strict_scale = false;
  std::string large_intrinsic_perturbation_reference_scene_path;
  bool large_intrinsic_perturbation_outer_only_after_application = false;
  double hybrid_pixel_ray_lambda = 0.5;
  bool hybrid_pixel_ray_polar_adaptive = false;
  double hybrid_pixel_ray_lambda_min = 0.2;
  double hybrid_pixel_ray_lambda_max = 0.8;
  double hybrid_pixel_ray_transition_start_deg = 30.0;
  double hybrid_pixel_ray_transition_end_deg = 70.0;
  int hybrid_pixel_ray_max_iterations = 12;
  double hybrid_pixel_ray_pixel_scale_floor = 1e-3;
  double hybrid_pixel_ray_ray_scale_floor = 1e-6;
  bool backend_angular_use_normalize_jacobian = false;
  bool backend_angular_local_whitening = false;
  double backend_angular_local_whitening_pixel_sigma_px = 1.0;
  double backend_angular_local_whitening_covariance_damping = 1e-12;
  double backend_angular_local_whitening_min_sigma_rad = 1e-6;
  double backend_angular_local_whitening_max_weight = 1e5;
  std::string backend_angular_observed_ray_mode = "dynamic_current_camera";
  std::string backend_board_pose_parameterization = "reference_chain";
  bool backend_optimize_board_poses = true;
  bool backend_board_pose_prior = false;
  double backend_board_pose_prior_translation_sigma_mm = 20.0;
  double backend_board_pose_prior_rotation_sigma_deg = 5.0;
  bool backend_point_budget_control = false;
  int backend_point_budget_control_total_points = 0;
  unsigned int backend_point_budget_control_seed = 1337;
  int backend_max_boards_per_frame_for_ablation = -1;
  bool backend_fixed_intrinsics = false;
  bool enable_angular_residual_diagnostics = false;
  std::string angular_residual_bin_edges = "0,30,50,70,85,100";
  bool backend_multi_board_consistency_weighting = false;
  std::string backend_consistency_pose_source = "outer_only";
  std::string backend_consistency_weight_mode = "cauchy";
  double backend_consistency_translation_sigma_mm = 3.0;
  double backend_consistency_rotation_sigma_deg = 2.0;
  double backend_consistency_min_weight = 0.25;
  bool backend_consistency_apply_to_outer = true;
  bool backend_consistency_apply_to_internal = true;
  bool backend_consistency_hard_reject_enabled = false;
  double backend_consistency_hard_reject_translation_mm = 8.0;
  double backend_consistency_hard_reject_rotation_deg = 5.0;
  double backend_consistency_hard_reject_residual_px = 8.0;
  bool backend_consistency_dump_weight_summary = true;
  bool internal_regeneration_diagnostics = false;
  bool stage5_export_internal_seed_step_overlays = false;
  bool internal_blur_diagnostics = false;
  ati::InternalBlurFilterMode internal_blur_filter_mode =
      ati::InternalBlurFilterMode::Off;
  double internal_blur_filter_low_patch_gradient_quantile = 0.05;
  double internal_blur_filter_min_board_rmse_px = 5.0;
  double internal_blur_filter_min_board_p95_px = 5.0;
  ati::InternalPoseRescueMode internal_pose_rescue_mode =
      ati::InternalPoseRescueMode::Enabled;
  double internal_pose_rescue_max_ray_angle_deg = 88.0;
  double internal_pose_rescue_accept_max_outer_rmse = 8.0;
  bool enable_geometry_prior_outer_seed = true;
  bool geometry_prior_rescue_diagnostic_only = false;
  bool geometry_prior_rescue_use_as_observation = true;
  bool geometry_prior_rescue_keep_outer_on_internal_failure = false;
  bool geometry_prior_rescue_allow_geometry_only_pose_refit = true;
  int geometry_prior_rescue_subpix_window_radius = 0;
  double geometry_prior_rescue_max_corner_displacement_px = 0.0;
  double geometry_prior_rescue_min_corner_response_ratio = 0.03;
  bool geometry_prior_rescue_enable_spherical_refine = true;
  int geometry_prior_rescue_edge_sample_count = 80;
  int geometry_prior_rescue_edge_search_half_width_px = 6;
  double geometry_prior_rescue_min_edge_support_ratio = 0.45;
  double geometry_prior_rescue_min_edge_gradient_ratio = 0.02;
  double geometry_prior_rescue_accept_max_outer_rmse = 8.0;
  double geometry_prior_rescue_accept_max_rotation_error_deg = 5.0;
  double geometry_prior_rescue_accept_max_translation_error = 0.08;
  bool geometry_guided_tag_likelihood_enabled = true;
  int geometry_guided_tag_likelihood_min_visible_boards = 2;
  int geometry_guided_tag_likelihood_max_expected_hamming = 6;
  int geometry_guided_tag_likelihood_min_hamming_margin = 3;
  double geometry_guided_tag_likelihood_min_contrast = 0.10;
  bool geometry_guided_tag_likelihood_allow_single_anchor = false;
  double geometry_guided_tag_likelihood_single_anchor_max_outer_rmse = 0.50;
  int geometry_guided_tag_likelihood_single_anchor_max_expected_hamming = 2;
  int geometry_guided_tag_likelihood_single_anchor_min_hamming_margin = 6;
  double geometry_guided_tag_likelihood_single_anchor_min_contrast = 0.15;
  bool enable_outer_only_intermediate_calibration = false;
  bool intermediate_diagnostic_only = true;
  bool use_intermediate_for_round1_internal_regeneration = false;
  bool use_intermediate_for_full_frontend_regeneration = false;
  bool intermediate_optimize_intrinsics = true;
  bool intermediate_optimize_board_poses = true;
  bool intermediate_optimize_frame_poses = true;
  int intermediate_intrinsics_release_iteration = 1;
  double intermediate_max_outer_rmse_px = 8.0;
  int intermediate_min_visible_boards = 1;
  ati::PreBackendFilterMode pre_backend_filter_mode =
      ati::PreBackendFilterMode::Enabled;
  ati::PreBackendFilterThresholdMode pre_backend_filter_threshold_mode =
      ati::PreBackendFilterThresholdMode::MeanStd;
  double pre_backend_filter_sigma = 2.0;
  double pre_backend_filter_min_abs_threshold_px = 0.2;
  ati::InternalJointRefineMode internal_joint_refine_mode =
      ati::InternalJointRefineMode::Off;
  ati::InternalJointRefineTargetMode internal_joint_refine_target_mode =
      ati::InternalJointRefineTargetMode::HighResidualAndBlurBadBoard;
  double internal_joint_refine_search_radius_px = 2.0;
  double internal_joint_refine_max_displacement_px = 1.5;
  double internal_joint_refine_geometry_sigma_px = 1.0;
  double internal_joint_refine_observation_sigma_px = 1.0;
  int internal_joint_refine_subpix_window_radius = 1;
  double internal_joint_refine_min_objective_improvement = 5e-4;
  double internal_joint_refine_min_old_residual_px = 1.0;
  double internal_joint_refine_low_patch_gradient_quantile = 0.05;
  double internal_joint_refine_min_board_rmse_px = 5.0;
  double internal_joint_refine_min_board_p95_px = 5.0;
  double internal_joint_refine_min_corner_response_gain = 0.02;
  double internal_joint_refine_min_board_internal_improvement_px = 0.1;
  int internal_joint_refine_min_refined_point_count_per_board = 4;
  double internal_joint_refine_accept_max_global_outer_delta_px = 0.01;
  double internal_joint_refine_accept_max_frame_outer_delta_px = 0.05;
  int internal_joint_refine_acceptance_backend_max_iterations = 4;
  ati::InternalObservationWeightMode internal_observation_weight_mode =
      ati::InternalObservationWeightMode::Off;
  ati::InternalBlurBoardWeightMode internal_blur_board_weight_mode =
      ati::InternalBlurBoardWeightMode::Off;
  double internal_blur_board_weight_low_patch_gradient_quantile = 0.05;
  double internal_blur_board_weight_min_board_rmse_px = 5.0;
  double internal_blur_board_weight_min_board_p95_px = 5.0;
  double internal_blur_board_weight_min = 0.25;
  double internal_blur_board_weight_gradient_exponent = 1.0;
  double internal_observation_weight_low_quality_quantile = 0.2;
  double internal_observation_weight_min = 0.25;
  double internal_observation_weight_quality_exponent = 1.0;
  std::string internal_observation_weight_policy = "quality";
  double internal_observation_weight_residual_consistency_sigma = 2.0;
  double internal_observation_weight_residual_consistency_min_rmse = 0.5;
  double kalibr_runtime_seconds = -1.0;
  ati::Stage5RuntimeMode runtime_mode = ati::Stage5RuntimeMode::Research;
  bool enable_polar_angle_diagnostics = false;
  std::string polar_angle_bin_edges = "0,30,50,70,85,100";
  bool enable_multi_board_consistency_diagnostics = false;
  std::string multi_board_consistency_pose_source = "outer_only";
  int multi_board_consistency_min_outer_points = 4;
  bool enable_multiboard_rigidity_diagnostics = false;
  bool enable_global_scene_state_consistency_audit = false;
  bool enable_stage5_selection_diagnostics = false;
  bool export_holdout_reprojection_visualizations = false;
  bool export_selected_case_visualizations = true;
  int holdout_visualization_top_k = 30;
  int multiboard_rigidity_top_k = 30;
  double multiboard_rigidity_rotation_bad_threshold_deg = 3.0;
  double multiboard_rigidity_translation_bad_threshold = -1.0;
  double multiboard_rigidity_reprojection_delta_bad_threshold_px = 2.0;
  bool multiboard_rigidity_use_internal_points = true;
  bool multiboard_rigidity_use_outer_points = true;
};

std::string BuildKalibrSourceLabel(const std::string& kalibr_camchain_yaml) {
  return fs::path(kalibr_camchain_yaml).lexically_normal().generic_string();
}

std::string BuildReferenceSourceLabelFromPath(const std::string& yaml_path) {
  const fs::path path(yaml_path);
  const std::string stem = path.stem().string();
  return stem.empty() ? BuildKalibrSourceLabel(yaml_path) : stem;
}

ati::KalibrBenchmarkReference BuildAdditionalReferenceCamera(
    const std::string& spec,
    const std::string& split_signature) {
  ati::KalibrBenchmarkReference reference;
  const std::size_t separator = spec.find(':');
  if (separator != std::string::npos && separator > 0 &&
      separator + 1 < spec.size()) {
    reference.source_label = spec.substr(0, separator);
    reference.camchain_yaml = spec.substr(separator + 1);
  } else {
    reference.camchain_yaml = spec;
    reference.source_label = BuildReferenceSourceLabelFromPath(spec);
  }
  reference.training_split_signature = split_signature;
  return reference;
}

ati::CalibrationMeasurementDataset BuildAllOuterObservationDataset(
    const ati::JointMeasurementBuildResult& measurement_result,
    const std::string& source_stage_label,
    const std::string& dataset_label,
    const std::string& split_signature,
    const std::string& protocol_label) {
  ati::CalibrationMeasurementDataset dataset;
  dataset.reference_board_id = measurement_result.reference_board_id;
  dataset.bundle_version = "stage5_all_outer_observations_v1";
  dataset.baseline_protocol_label = protocol_label;
  dataset.training_split_signature = split_signature;
  dataset.frames = measurement_result.frames;
  dataset.dataset_label = dataset_label;
  dataset.source_stage_label = source_stage_label;
  dataset.warnings = measurement_result.warnings;
  dataset.failure_reason = measurement_result.failure_reason;

  for (const ati::JointMeasurementFrameResult& frame :
       measurement_result.frames) {
    bool frame_has_outer = false;
    for (const ati::JointBoardObservation& board : frame.board_observations) {
      int board_outer_count = 0;
      for (const ati::JointPointObservation& point : board.points) {
        if (point.point_type != ati::JointPointType::Outer) {
          continue;
        }
        dataset.solver_observations.push_back(point);
        ++board_outer_count;
      }
      if (board_outer_count > 0) {
        frame_has_outer = true;
        dataset.accepted_board_observation_keys.insert(
            std::make_pair(frame.frame_index, board.board_id));
        dataset.accepted_outer_point_count += board_outer_count;
      }
    }
    if (frame_has_outer) {
      dataset.accepted_frame_indices.insert(frame.frame_index);
    }
  }

  dataset.accepted_frame_count =
      static_cast<int>(dataset.accepted_frame_indices.size());
  dataset.accepted_board_observation_count =
      static_cast<int>(dataset.accepted_board_observation_keys.size());
  dataset.accepted_internal_point_count = 0;
  dataset.accepted_total_point_count =
      static_cast<int>(dataset.solver_observations.size());
  return dataset;
}

std::string BuildConfiguredCameraFamily(const ati::ApriltagInternalConfig& config) {
  ati::OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model =
      config.intermediate_camera.camera_model.empty()
          ? "ds"
          : config.intermediate_camera.camera_model;
  intrinsics.distortion_model =
      config.intermediate_camera.distortion_model.empty()
          ? (intrinsics.camera_model == "pinhole" ? "equi" : "none")
          : config.intermediate_camera.distortion_model;
  return intrinsics.NormalizedFamilyString();
}

std::string NormalizeStage5ModelFamily(std::string model) {
  std::transform(model.begin(), model.end(), model.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (model == "ds" || model == "double_sphere" ||
      model == "double-sphere") {
    return "ds-none";
  }
  if (model == "ds-none") {
    return model;
  }
  if (model == "kb" || model == "kannala-brandt" ||
      model == "kannala_brandt" || model == "equi" ||
      model == "equidistant" || model == "pinhole-equi") {
    return "pinhole-equi";
  }
  if (model == "eucm" || model == "eucm-none") {
    return "eucm-none";
  }
  if (model == "mei" || model == "ucm" || model == "omni" ||
      model == "omni-none" || model == "omni_none") {
    return "omni-none";
  }
  if (model == "omni-radtan" || model == "omni_radtan" ||
      model == "mei-radtan" || model == "radial-tangential-omni") {
    return "omni-radtan";
  }
  throw std::runtime_error(
      "Unsupported --models value: " + model +
      " (supported: ds-none, pinhole-equi/kb, eucm-none, mei/ucm/omni-none, omni-radtan)");
}

void ApplyStage5ModelFamily(const std::string& model_family,
                            ati::ApriltagInternalConfig* config) {
  if (config == nullptr) {
    throw std::runtime_error(
        "ApplyStage5ModelFamily requires a valid config pointer.");
  }
  const std::string family = NormalizeStage5ModelFamily(model_family);
  ati::IntermediateCameraConfig camera = config->intermediate_camera;
  camera.camera_yaml.clear();
  camera.intrinsics.clear();
  camera.distortion_coeffs.clear();
  if (family == "ds-none") {
    camera.camera_model = "ds";
    camera.distortion_model = "none";
  } else if (family == "pinhole-equi") {
    camera.camera_model = "pinhole";
    camera.distortion_model = "equi";
  } else if (family == "eucm-none") {
    camera.camera_model = "eucm";
    camera.distortion_model = "none";
  } else if (family == "omni-radtan") {
    camera.camera_model = "omni";
    camera.distortion_model = "radtan";
  } else if (family == "omni-none") {
    camera.camera_model = "omni";
    camera.distortion_model = "none";
  }
  config->intermediate_camera = camera;
}

double ElapsedSeconds(const std::chrono::steady_clock::time_point& start_time) {
  return std::chrono::duration_cast<std::chrono::duration<double> >(
             std::chrono::steady_clock::now() - start_time)
      .count();
}

void AddRuntimeStage(ati::Stage5RuntimeSummary* summary,
                     const std::string& stage_label,
                     double seconds,
                     bool skipped_in_fast_mode) {
  if (summary == nullptr) {
    return;
  }
  ati::Stage5RuntimeStageRecord record;
  record.stage_label = stage_label;
  record.wall_time_seconds = seconds;
  record.skipped_in_fast_mode = skipped_in_fast_mode;
  summary->stage_records.push_back(record);
}

struct RequestedExperimentConfig {
  std::string frozen_baseline_label = kFrozenBaselineLabel;
  std::string experiment_tag;
  std::string effective_protocol_label = kFrozenBaselineLabel;
  std::string models = "ds-none";
  ati::AutoCameraInitializationRefineMode init_refine_mode =
      ati::AutoCameraInitializationRefineMode::KalibrOuterLm;
  ati::AutoCameraInitializationSelectionScorer init_selection_scorer =
      ati::AutoCameraInitializationSelectionScorer::
          PoseMarginalizedPrincipal;
  ati::CameraInitializationMode camera_init_mode =
      ati::CameraInitializationMode::Auto;
  bool camera_aware_outer_rescue = true;
  bool camera_aware_outer_rescue_zero_detection_frames = true;
  bool camera_aware_outer_rescue_zero_detection_frames_set = true;
  bool frozen_recovery_baseline_preset = false;
  bool outer_only_ablation_mode = false;
  bool include_internal_points = true;
  bool run_second_pass = true;
  bool frontend_optimize_intrinsics = true;
  int frontend_intrinsics_release_iteration = 3;
  int frontend_second_pass_intrinsics_release_iteration = 1;
  IntrinsicsReleaseMode frontend_intrinsics_release_mode =
      IntrinsicsReleaseMode::Delayed;
  bool backend_optimize_intrinsics = true;
  bool backend_optimize_board_poses = true;
  bool backend_board_pose_prior = false;
  double backend_board_pose_prior_translation_sigma_mm = 20.0;
  double backend_board_pose_prior_rotation_sigma_deg = 5.0;
  bool backend_point_budget_control = false;
  int backend_point_budget_control_total_points = 0;
  unsigned int backend_point_budget_control_seed = 1337;
  int backend_max_boards_per_frame_for_ablation = -1;
  bool export_selected_case_visualizations = true;
  bool backend_delayed_intrinsics_release = true;
  int backend_intrinsics_release_iteration = 1;
  IntrinsicsReleaseMode backend_intrinsics_release_mode =
      IntrinsicsReleaseMode::Delayed;
  std::string backend_board_pose_parameterization = "reference_chain";
  bool enable_residual_sanity_gate = true;
  bool enable_board_pose_fit_gate = false;
  std::string selection_mode = "baseline";
  double selection_residual_sanity_factor = 2.5;
  double selection_max_board_observation_rmse = 25.0;
  double selection_kalibr_style_outlier_sigma = 4.0;
  double selection_kalibr_style_min_abs_threshold_px = 1.0;
  int selection_kalibr_style_min_views_before_filter = 20;
  bool enable_trial_backend_frame_board_selection = true;
  std::string trial_backend_selection_mode = "kalibr_style_batch";
  std::string trial_backend_selection_budget_mode;
  bool trial_backend_selection_budget_mode_set = false;
  std::string trial_backend_selection_candidate_order;
  bool trial_backend_selection_candidate_order_set = false;
  std::string trial_backend_selection_info_gain_proxy_mode =
      "intrinsics_jacobian";
  std::string trial_backend_selection_batch_granularity = "frame";
  std::string trial_backend_selection_acceptance_policy =
      "kalibr_information_gain";
  double trial_backend_selection_mi_tol = 0.2;
  bool trial_backend_selection_mi_tol_set = false;
  double trial_backend_selection_rank_threshold = 1e-6;
  double checkerboard_huber_delta_pixels = 1.5;
  bool checkerboard_outlier_filter_enabled = true;
  double checkerboard_outlier_sigma = 4.0;
  double checkerboard_min_inlier_ratio = 0.5;
  int checkerboard_min_retained_points = 8;
  unsigned int trial_backend_selection_candidate_shuffle_seed = 0;
  bool trial_backend_selection_candidate_shuffle_seed_set = false;
  bool trial_backend_selection_incremental = true;
  bool trial_backend_selection_carry_accepted_trial_state = true;
  bool trial_backend_selection_optimize_intrinsics = true;
  bool trial_backend_selection_delayed_intrinsics_release = true;
  int trial_backend_selection_intrinsics_release_iteration = 1;
  bool trial_backend_selection_persistent_intrinsics_anchor_prior = false;
  bool trial_backend_selection_persistent_fix_board_layout = false;
  double trial_backend_selection_persistent_intrinsics_anchor_weight_xi_alpha = 0.0;
  double trial_backend_selection_persistent_intrinsics_anchor_weight_focal = 0.0;
  double trial_backend_selection_persistent_intrinsics_anchor_weight_principal = 0.0;
  double trial_backend_selection_persistent_max_focal_relative_step = 0.0;
  double trial_backend_selection_persistent_max_principal_step_px = 0.0;
  double trial_backend_selection_persistent_max_xi_alpha_step = 0.0;
  int trial_backend_selection_max_iterations = 5;
  int trial_backend_selection_max_candidate_additions = 20;
  double trial_backend_selection_adaptive_budget_ratio = 0.10;
  int trial_backend_selection_adaptive_budget_min = 20;
  int trial_backend_selection_adaptive_budget_max = 120;
  int trial_backend_selection_runtime_safety_ceiling = 1000;
  double trial_backend_selection_outlier_sigma = 4.0;
  double trial_backend_selection_min_abs_threshold_px = 1.0;
  double trial_backend_selection_max_threshold_px = 25.0;
  double trial_backend_selection_accept_max_global_rmse_increase_px = 0.02;
  double trial_backend_selection_accept_max_outer_rmse_increase_px = 0.05;
  double trial_backend_selection_accept_max_internal_rmse_increase_px = 0.05;
  double trial_backend_selection_min_candidate_score = 3.2;
  double trial_backend_selection_min_coverage_gain = 2.5;
  bool trial_backend_selection_use_consistency_score = false;
  double trial_backend_selection_consistency_translation_sigma_mm = 3.0;
  double trial_backend_selection_consistency_rotation_sigma_deg = 2.0;
  double trial_backend_selection_consistency_penalty_weight = 1.0;
  double trial_backend_selection_consistency_max_translation_error_mm = -1.0;
  double trial_backend_selection_consistency_max_rotation_error_deg = -1.0;
  double trial_backend_selection_consistency_max_local_outer_rmse_px = -1.0;
  int trial_backend_selection_max_accepted_per_board = 0;
  int trial_backend_selection_max_accepted_per_frame = 0;
  bool trial_backend_selection_frame_cohesion = true;
  int trial_backend_selection_frame_cohesion_max_companions = 0;
  double trial_backend_selection_frame_cohesion_min_candidate_score = 3.2;
  int trial_backend_selection_min_keep_per_board = 5;
  std::string trial_backend_selection_force_include_frame_board_list;
  std::string trial_backend_selection_seed_frame_board_list;
  bool trial_backend_selection_force_include_list_is_exact_input = false;
  bool strict_board_observation_acceptance = false;
  bool preserve_frame_board_cohesion = false;
  bool ignore_image_evidence_min_quality = false;
  bool force_internal_seed_from_prediction = false;
  bool bypass_internal_seed_filters = false;
  std::string internal_corner_filter_mode = "sigma";
  double internal_corner_filter_max_reproj_error = -1.0;
  double internal_corner_filter_quality_min = 0.35;
  double internal_corner_filter_quality_relaxation_px = 1.0;
  double internal_corner_filter_adaptive_min_threshold_px = 1.0;
  std::string backend_polar_angle_weight_mode = "none";
  std::vector<double> backend_polar_angle_weight_bin_edges_deg = {0.0, 30.0, 50.0, 70.0, 85.0, 100.0};
  std::vector<double> backend_polar_angle_weight_fixed_bin_scales = {1.0, 1.0, 0.75, 0.5, 0.25};
  double backend_polar_angle_weight_adaptive_sigma_reference_deg = 50.0;
  double backend_polar_angle_weight_adaptive_sigma_growth = 1.0;
  double backend_polar_angle_weight_min_scale = 0.25;
  std::string backend_residual_model = "image_plane";
  double backend_hybrid_angular_threshold_deg = 50.0;
  std::string backend_outer_residual_model = "image_plane";
  std::string backend_internal_residual_model = "image_plane";
  bool backend_use_point_type_residual_split = false;
  bool backend_enable_angular_auxiliary_residual = false;
  double backend_angular_auxiliary_weight = 0.0;
  bool backend_angular_auxiliary_normalized = false;
  bool backend_angular_auxiliary_apply_to_outer = true;
  bool backend_angular_auxiliary_apply_to_internal = true;
  double backend_polar_continuous_hybrid_threshold_deg = 50.0;
  double backend_polar_continuous_hybrid_temperature_deg = 10.0;
  double backend_normalized_angular_reference_sigma_px = 1.0;
  double backend_normalized_angular_min_sigma_rad = 1e-6;
  double backend_normalized_angular_max_weight_scale = 1.0e8;
  double backend_pixel_residual_weight = 1.0;
  double backend_chordal_residual_weight = 1.0;
  bool enable_hybrid_pixel_ray_final_refinement = false;
  double hybrid_pixel_ray_lambda = 0.5;
  bool hybrid_pixel_ray_polar_adaptive = false;
  double hybrid_pixel_ray_lambda_min = 0.2;
  double hybrid_pixel_ray_lambda_max = 0.8;
  double hybrid_pixel_ray_transition_start_deg = 30.0;
  double hybrid_pixel_ray_transition_end_deg = 70.0;
  int hybrid_pixel_ray_max_iterations = 12;
  double hybrid_pixel_ray_pixel_scale_floor = 1e-3;
  double hybrid_pixel_ray_ray_scale_floor = 1e-6;
  bool backend_angular_use_normalize_jacobian = false;
  bool backend_angular_local_whitening = false;
  double backend_angular_local_whitening_pixel_sigma_px = 1.0;
  double backend_angular_local_whitening_covariance_damping = 1e-12;
  double backend_angular_local_whitening_min_sigma_rad = 1e-6;
  double backend_angular_local_whitening_max_weight = 1e5;
  std::string backend_angular_observed_ray_mode = "dynamic_current_camera";
  bool backend_fixed_intrinsics = false;
  bool enable_angular_residual_diagnostics = false;
  std::vector<double> angular_residual_bin_edges_deg = {0.0, 30.0, 50.0, 70.0, 85.0, 100.0};
  bool backend_multi_board_consistency_weighting = false;
  std::string backend_consistency_pose_source = "outer_only";
  std::string backend_consistency_weight_mode = "cauchy";
  double backend_consistency_translation_sigma_mm = 3.0;
  double backend_consistency_rotation_sigma_deg = 2.0;
  double backend_consistency_min_weight = 0.25;
  bool backend_consistency_apply_to_outer = true;
  bool backend_consistency_apply_to_internal = true;
  bool backend_consistency_hard_reject_enabled = false;
  double backend_consistency_hard_reject_translation_mm = 8.0;
  double backend_consistency_hard_reject_rotation_deg = 5.0;
  double backend_consistency_hard_reject_residual_px = 8.0;
  bool backend_consistency_dump_weight_summary = true;
  bool internal_regeneration_diagnostics = false;
  bool export_internal_seed_step_overlays = false;
  bool internal_blur_diagnostics = false;
  ati::InternalBlurFilterMode internal_blur_filter_mode =
      ati::InternalBlurFilterMode::Off;
  double internal_blur_filter_low_patch_gradient_quantile = 0.05;
  double internal_blur_filter_min_board_rmse_px = 5.0;
  double internal_blur_filter_min_board_p95_px = 5.0;
  ati::InternalPoseRescueMode internal_pose_rescue_mode =
      ati::InternalPoseRescueMode::Enabled;
  double internal_pose_rescue_max_ray_angle_deg = 88.0;
  double internal_pose_rescue_accept_max_outer_rmse = 8.0;
  bool enable_geometry_prior_outer_seed = true;
  bool geometry_prior_rescue_diagnostic_only = false;
  bool geometry_prior_rescue_use_as_observation = true;
  bool geometry_prior_rescue_keep_outer_on_internal_failure = false;
  bool geometry_prior_rescue_allow_geometry_only_pose_refit = true;
  int geometry_prior_rescue_subpix_window_radius = 0;
  double geometry_prior_rescue_max_corner_displacement_px = 0.0;
  double geometry_prior_rescue_min_corner_response_ratio = 0.03;
  bool geometry_prior_rescue_enable_spherical_refine = true;
  int geometry_prior_rescue_edge_sample_count = 80;
  int geometry_prior_rescue_edge_search_half_width_px = 6;
  double geometry_prior_rescue_min_edge_support_ratio = 0.45;
  double geometry_prior_rescue_min_edge_gradient_ratio = 0.02;
  double geometry_prior_rescue_accept_max_outer_rmse = 8.0;
  double geometry_prior_rescue_accept_max_rotation_error_deg = 5.0;
  double geometry_prior_rescue_accept_max_translation_error = 0.08;
  bool geometry_guided_tag_likelihood_enabled = true;
  int geometry_guided_tag_likelihood_min_visible_boards = 2;
  int geometry_guided_tag_likelihood_max_expected_hamming = 6;
  int geometry_guided_tag_likelihood_min_hamming_margin = 3;
  double geometry_guided_tag_likelihood_min_contrast = 0.10;
  bool geometry_guided_tag_likelihood_allow_single_anchor = false;
  double geometry_guided_tag_likelihood_single_anchor_max_outer_rmse = 0.50;
  int geometry_guided_tag_likelihood_single_anchor_max_expected_hamming = 2;
  int geometry_guided_tag_likelihood_single_anchor_min_hamming_margin = 6;
  double geometry_guided_tag_likelihood_single_anchor_min_contrast = 0.15;
  bool enable_outer_only_intermediate_calibration = false;
  bool intermediate_diagnostic_only = true;
  bool use_intermediate_for_round1_internal_regeneration = false;
  bool use_intermediate_for_full_frontend_regeneration = false;
  bool intermediate_optimize_intrinsics = true;
  bool intermediate_optimize_board_poses = true;
  bool intermediate_optimize_frame_poses = true;
  int intermediate_intrinsics_release_iteration = 1;
  double intermediate_max_outer_rmse_px = 8.0;
  int intermediate_min_visible_boards = 1;
  ati::PreBackendFilterMode pre_backend_filter_mode =
      ati::PreBackendFilterMode::Enabled;
  ati::PreBackendFilterThresholdMode pre_backend_filter_threshold_mode =
      ati::PreBackendFilterThresholdMode::MeanStd;
  double pre_backend_filter_sigma = 2.0;
  double pre_backend_filter_min_abs_threshold_px = 0.2;
  ati::InternalJointRefineMode internal_joint_refine_mode =
      ati::InternalJointRefineMode::Off;
  ati::InternalJointRefineTargetMode internal_joint_refine_target_mode =
      ati::InternalJointRefineTargetMode::HighResidualAndBlurBadBoard;
  double internal_joint_refine_search_radius_px = 2.0;
  double internal_joint_refine_max_displacement_px = 1.5;
  double internal_joint_refine_geometry_sigma_px = 1.0;
  double internal_joint_refine_observation_sigma_px = 1.0;
  int internal_joint_refine_subpix_window_radius = 1;
  double internal_joint_refine_min_objective_improvement = 5e-4;
  double internal_joint_refine_min_old_residual_px = 1.0;
  double internal_joint_refine_low_patch_gradient_quantile = 0.05;
  double internal_joint_refine_min_board_rmse_px = 5.0;
  double internal_joint_refine_min_board_p95_px = 5.0;
  double internal_joint_refine_min_corner_response_gain = 0.02;
  double internal_joint_refine_min_board_internal_improvement_px = 0.1;
  int internal_joint_refine_min_refined_point_count_per_board = 4;
  double internal_joint_refine_accept_max_global_outer_delta_px = 0.01;
  double internal_joint_refine_accept_max_frame_outer_delta_px = 0.05;
  int internal_joint_refine_acceptance_backend_max_iterations = 4;
  ati::InternalObservationWeightMode internal_observation_weight_mode =
      ati::InternalObservationWeightMode::Off;
  ati::InternalBlurBoardWeightMode internal_blur_board_weight_mode =
      ati::InternalBlurBoardWeightMode::Off;
  double internal_blur_board_weight_low_patch_gradient_quantile = 0.05;
  double internal_blur_board_weight_min_board_rmse_px = 5.0;
  double internal_blur_board_weight_min_board_p95_px = 5.0;
  double internal_blur_board_weight_min = 0.25;
  double internal_blur_board_weight_gradient_exponent = 1.0;
  double internal_observation_weight_low_quality_quantile = 0.2;
  double internal_observation_weight_min = 0.25;
  double internal_observation_weight_quality_exponent = 1.0;
  std::string internal_observation_weight_policy = "quality";
  double internal_observation_weight_residual_consistency_sigma = 2.0;
  double internal_observation_weight_residual_consistency_min_rmse = 0.5;
  bool enable_multi_board_consistency_diagnostics = false;
  std::string multi_board_consistency_pose_source = "outer_only";
  int multi_board_consistency_min_outer_points = 4;
  bool enable_multiboard_rigidity_diagnostics = false;
  bool enable_global_scene_state_consistency_audit = false;
  bool enable_stage5_selection_diagnostics = false;
  int multiboard_rigidity_top_k = 30;
  double multiboard_rigidity_rotation_bad_threshold_deg = 3.0;
  double multiboard_rigidity_translation_bad_threshold = -1.0;
  double multiboard_rigidity_reprojection_delta_bad_threshold_px = 2.0;
  bool multiboard_rigidity_use_internal_points = true;
  bool multiboard_rigidity_use_outer_points = true;
};

const char* ToString(IntrinsicsReleaseMode mode) {
  switch (mode) {
    case IntrinsicsReleaseMode::Delayed:
      return "delayed";
    case IntrinsicsReleaseMode::Immediate:
      return "immediate";
    case IntrinsicsReleaseMode::PoseOnly:
      return "pose_only";
  }
  return "unknown";
}

bool ParseBooleanFlagValue(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "1" || lowered == "true" || lowered == "yes" ||
      lowered == "on") {
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" ||
      lowered == "off") {
    return false;
  }
  throw std::runtime_error("Invalid boolean flag value: " + value);
}

IntrinsicsReleaseMode ParseIntrinsicsReleaseMode(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (lowered == "delayed") {
    return IntrinsicsReleaseMode::Delayed;
  }
  if (lowered == "immediate") {
    return IntrinsicsReleaseMode::Immediate;
  }
  if (lowered == "pose_only" || lowered == "pose-only") {
    return IntrinsicsReleaseMode::PoseOnly;
  }
  throw std::runtime_error("Unsupported intrinsics release mode: " + value);
}

ati::TrialBackendFrameBoardSelectionOptions::SelectionMode
ParseTrialBackendFrameBoardSelectionMode(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "strict_rmse" || lowered == "strict-rmse") {
    return ati::TrialBackendFrameBoardSelectionOptions::SelectionMode::StrictRmse;
  }
  if (lowered == "kalibr_style_batch" ||
      lowered == "kalibr-style-batch") {
    return ati::TrialBackendFrameBoardSelectionOptions::SelectionMode
        ::KalibrStyleBatch;
  }
  throw std::runtime_error(
      "Unsupported --stage5-trial-backend-selection-mode: " + value);
}

ati::TrialBackendFrameBoardSelectionOptions::BudgetMode
ParseTrialBackendFrameBoardSelectionBudgetMode(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "fixed") {
    return ati::TrialBackendFrameBoardSelectionOptions::BudgetMode::Fixed;
  }
  if (lowered == "adaptive") {
    return ati::TrialBackendFrameBoardSelectionOptions::BudgetMode::Adaptive;
  }
  if (lowered == "kalibr_style" || lowered == "kalibr-style") {
    return ati::TrialBackendFrameBoardSelectionOptions::BudgetMode::KalibrStyle;
  }
  throw std::runtime_error(
      "Unsupported --stage5-trial-backend-selection-budget-mode: " + value);
}

ati::TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
ParseTrialBackendFrameBoardCandidateOrderMode(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "score_sorted" || lowered == "score-sorted") {
    return ati::TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
        ::ScoreSorted;
  }
  if (lowered == "random_shuffle" || lowered == "random-shuffle") {
    return ati::TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
        ::RandomShuffle;
  }
  if (lowered == "intrinsics_information_greedy" ||
      lowered == "intrinsics-information-greedy" ||
      lowered == "information_greedy") {
    return ati::TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
        ::IntrinsicsInformationGreedy;
  }
  throw std::runtime_error(
      "Unsupported --stage5-trial-backend-selection-candidate-order: " +
      value);
}

ati::TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode
ParseTrialBackendFrameBoardInfoGainProxyMode(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "legacy" || lowered == "old" ||
      lowered == "old_info_gain_proxy") {
    return ati::TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode
        ::Legacy;
  }
  if (lowered == "intrinsics_jacobian" ||
      lowered == "intrinsics-jacobian" ||
      lowered == "jacobian") {
    return ati::TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode
        ::IntrinsicsJacobian;
  }
  throw std::runtime_error(
      "Unsupported --stage5-trial-backend-selection-info-gain-proxy-mode: " +
      value);
}

ati::TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity
ParseTrialBackendFrameBoardCandidateBatchGranularity(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "frame_board" || lowered == "frame-board" ||
      lowered == "board" || lowered == "frameboard") {
    return ati::TrialBackendFrameBoardSelectionOptions
        ::CandidateBatchGranularity::FrameBoard;
  }
  if (lowered == "frame" || lowered == "timestamp" ||
      lowered == "view") {
    return ati::TrialBackendFrameBoardSelectionOptions
        ::CandidateBatchGranularity::Frame;
  }
  if (lowered == "frame_board_then_frame" ||
      lowered == "frame-board-then-frame" ||
      lowered == "frame_consolidation" ||
      lowered == "frame-consolidation" ||
      lowered == "consolidated") {
    return ati::TrialBackendFrameBoardSelectionOptions
        ::CandidateBatchGranularity::FrameBoardThenFrame;
  }
  throw std::runtime_error(
      "Unsupported --stage5-trial-backend-selection-batch-granularity: " +
      value);
}

IntrinsicsReleaseMode DeriveFrontendIntrinsicsReleaseMode(
    bool optimize_intrinsics,
    bool run_second_pass,
    int round1_release_iteration,
    int round2_release_iteration) {
  if (!optimize_intrinsics) {
    return IntrinsicsReleaseMode::PoseOnly;
  }
  if (round1_release_iteration <= 0 &&
      (!run_second_pass || round2_release_iteration <= 0)) {
    return IntrinsicsReleaseMode::Immediate;
  }
  return IntrinsicsReleaseMode::Delayed;
}

IntrinsicsReleaseMode DeriveBackendIntrinsicsReleaseMode(
    bool optimize_intrinsics,
    bool delayed_intrinsics_release) {
  if (!optimize_intrinsics) {
    return IntrinsicsReleaseMode::PoseOnly;
  }
  return delayed_intrinsics_release ? IntrinsicsReleaseMode::Delayed
                                    : IntrinsicsReleaseMode::Immediate;
}

std::vector<double> ParsePolarAngleBinEdges(const std::string& edges_str) {
  std::vector<double> edges;
  std::istringstream stream(edges_str);
  std::string token;
  while (std::getline(stream, token, ',')) {
    try {
      edges.push_back(std::stod(token));
    } catch (const std::exception&) {
      throw std::runtime_error("Invalid polar angle bin edge: " + token);
    }
  }
  if (edges.size() < 2) {
    throw std::runtime_error("At least 2 polar angle bin edges are required.");
  }
  for (std::size_t i = 1; i < edges.size(); ++i) {
    if (edges[i] <= edges[i - 1]) {
      throw std::runtime_error(
          "Polar angle bin edges must be strictly increasing.");
    }
  }
  return edges;
}

std::vector<double> ParseCommaSeparatedDoubles(const std::string& values_str,
                                               const std::string& label) {
  std::vector<double> values;
  std::istringstream stream(values_str);
  std::string token;
  while (std::getline(stream, token, ',')) {
    try {
      values.push_back(std::stod(token));
    } catch (const std::exception&) {
      throw std::runtime_error("Invalid " + label + ": " + token);
    }
  }
  if (values.empty()) {
    throw std::runtime_error("At least one " + label + " value is required.");
  }
  return values;
}

bool HasAblationOverrides(const RequestedExperimentConfig& config) {
  return config.camera_init_mode != ati::CameraInitializationMode::Auto ||
         !config.camera_aware_outer_rescue ||
         !config.camera_aware_outer_rescue_zero_detection_frames ||
         config.outer_only_ablation_mode ||
         !config.include_internal_points ||
         !config.run_second_pass ||
         config.frontend_intrinsics_release_mode != IntrinsicsReleaseMode::Delayed ||
         config.backend_intrinsics_release_mode != IntrinsicsReleaseMode::Delayed ||
         !config.enable_residual_sanity_gate ||
         config.enable_board_pose_fit_gate ||
         config.strict_board_observation_acceptance ||
         config.preserve_frame_board_cohesion ||
         config.force_internal_seed_from_prediction ||
         config.bypass_internal_seed_filters ||
         config.internal_corner_filter_mode != "sigma" ||
         config.internal_corner_filter_max_reproj_error > 0.0 ||
         std::fabs(config.internal_corner_filter_quality_min - 0.35) > 1e-12 ||
         std::fabs(config.internal_corner_filter_quality_relaxation_px - 1.0) > 1e-12 ||
         std::fabs(config.internal_corner_filter_adaptive_min_threshold_px - 1.0) > 1e-12 ||
	         config.backend_residual_model != "image_plane" ||
	         config.enable_hybrid_pixel_ray_final_refinement ||
	         config.backend_angular_observed_ray_mode != "dynamic_current_camera" ||
	         !config.backend_optimize_board_poses ||
	         config.backend_point_budget_control ||
	         config.backend_max_boards_per_frame_for_ablation > 0 ||
	         config.backend_fixed_intrinsics ||
         config.backend_polar_angle_weight_mode != "none" ||
         config.backend_multi_board_consistency_weighting ||
         config.backend_consistency_pose_source != "outer_only" ||
         config.backend_consistency_weight_mode != "cauchy" ||
         std::fabs(config.backend_consistency_translation_sigma_mm - 3.0) >
             1e-12 ||
         std::fabs(config.backend_consistency_rotation_sigma_deg - 2.0) >
             1e-12 ||
         std::fabs(config.backend_consistency_min_weight - 0.25) > 1e-12 ||
         !config.backend_consistency_apply_to_outer ||
         !config.backend_consistency_apply_to_internal ||
         config.backend_consistency_hard_reject_enabled ||
         !config.backend_consistency_dump_weight_summary ||
         config.enable_angular_residual_diagnostics ||
         config.enable_multi_board_consistency_diagnostics ||
         config.enable_multiboard_rigidity_diagnostics ||
         config.internal_blur_filter_mode != ati::InternalBlurFilterMode::Off ||
         config.internal_pose_rescue_mode != ati::InternalPoseRescueMode::Enabled ||
         !config.enable_geometry_prior_outer_seed ||
         config.geometry_prior_rescue_diagnostic_only ||
         !config.geometry_prior_rescue_use_as_observation ||
         config.geometry_prior_rescue_keep_outer_on_internal_failure ||
         !config.geometry_prior_rescue_enable_spherical_refine ||
         config.enable_outer_only_intermediate_calibration ||
         config.internal_blur_board_weight_mode !=
             ati::InternalBlurBoardWeightMode::Off ||
         config.internal_joint_refine_mode != ati::InternalJointRefineMode::Off ||
         config.internal_observation_weight_mode !=
             ati::InternalObservationWeightMode::Off ||
         config.pre_backend_filter_mode != ati::PreBackendFilterMode::Off;
}

std::string BuildDeterministicExperimentTag(const RequestedExperimentConfig& config) {
  std::vector<std::string> parts;
  if (config.camera_init_mode != ati::CameraInitializationMode::Auto) {
    parts.push_back("caminit_" + std::string(ati::ToString(config.camera_init_mode)));
  }
  if (config.outer_only_ablation_mode) {
    parts.push_back("outer_only_ablation");
  }
  if (!config.run_second_pass) {
    parts.push_back("no_round2");
  }
  if (!config.include_internal_points) {
    parts.push_back("outer_only_calibration");
  }
  if (config.frontend_intrinsics_release_mode != IntrinsicsReleaseMode::Delayed) {
    parts.push_back("intrinsics_" +
                    std::string(ToString(config.frontend_intrinsics_release_mode)));
  }
  if (!config.enable_residual_sanity_gate) {
    parts.push_back("no_residual_gate");
  }
  if (config.enable_board_pose_fit_gate) {
    parts.push_back("board_pose_gate_debug");
  }
  if (config.preserve_frame_board_cohesion) {
    parts.push_back("frame_board_cohesion");
  }
  if (config.force_internal_seed_from_prediction) {
    parts.push_back("force_internal_seed");
  }
  if (config.bypass_internal_seed_filters) {
    parts.push_back("bypass_internal_seed_filters");
  }
  if (config.internal_corner_filter_mode != "sigma") {
    parts.push_back("internal_filter_" + config.internal_corner_filter_mode);
  }
  if (config.internal_corner_filter_max_reproj_error > 0.0) {
    std::ostringstream part;
    part << "internal_filter_cap_x10_"
         << static_cast<int>(std::round(
                config.internal_corner_filter_max_reproj_error * 10.0));
    parts.push_back(part.str());
  }
  if (std::fabs(config.internal_corner_filter_quality_min - 0.35) > 1e-12) {
    std::ostringstream part;
    part << "internal_filter_qmin_x100_"
         << static_cast<int>(std::round(
                config.internal_corner_filter_quality_min * 100.0));
    parts.push_back(part.str());
  }
  if (std::fabs(config.internal_corner_filter_quality_relaxation_px - 1.0) > 1e-12) {
    std::ostringstream part;
    part << "internal_filter_qrelax_x10_"
         << static_cast<int>(std::round(
                config.internal_corner_filter_quality_relaxation_px * 10.0));
    parts.push_back(part.str());
  }
  if (config.strict_board_observation_acceptance) {
    parts.push_back("kalibr_style_failed_board_drop");
  }
  if (config.internal_blur_filter_mode != ati::InternalBlurFilterMode::Off) {
    parts.push_back("internal_blur_filter_" +
                    std::string(ati::ToString(config.internal_blur_filter_mode)));
  }
  if (config.internal_pose_rescue_mode != ati::InternalPoseRescueMode::Enabled) {
    parts.push_back("internal_pose_rescue_" +
                    std::string(ati::ToString(config.internal_pose_rescue_mode)));
  }
  if (!config.camera_aware_outer_rescue) {
    parts.push_back("no_camera_aware_outer_rescue");
  }
  if (!config.camera_aware_outer_rescue_zero_detection_frames) {
    parts.push_back("no_zero_detection_atlas");
  }
  if (!config.enable_geometry_prior_outer_seed) {
    parts.push_back("no_geometry_prior_outer_seed");
  }
  if (config.geometry_prior_rescue_diagnostic_only) {
    parts.push_back("geometry_prior_rescue_diagnostic_only");
  }
  if (!config.geometry_prior_rescue_use_as_observation) {
    parts.push_back("geometry_prior_rescue_not_observation");
  }
  if (!config.geometry_prior_rescue_enable_spherical_refine) {
    parts.push_back("geometry_prior_rescue_no_spherical_refine");
  }
  if (!config.geometry_prior_rescue_allow_geometry_only_pose_refit) {
    parts.push_back("no_topology_identity_rescue");
  }
  if (config.geometry_prior_rescue_keep_outer_on_internal_failure) {
    parts.push_back("geometry_prior_rescue_outer_fallback");
  }
  if (config.internal_blur_board_weight_mode !=
      ati::InternalBlurBoardWeightMode::Off) {
    std::ostringstream blur_weight_tag;
    blur_weight_tag << "blur_board_weight_"
                    << ati::ToString(config.internal_blur_board_weight_mode);
    if (std::fabs(
            config.internal_blur_board_weight_low_patch_gradient_quantile - 0.05) >
        1e-12) {
      blur_weight_tag << "_q"
                      << static_cast<int>(std::lround(
                             config.internal_blur_board_weight_low_patch_gradient_quantile *
                             100.0));
    }
    if (std::fabs(config.internal_blur_board_weight_min - 0.25) > 1e-12) {
      blur_weight_tag << "_min"
                      << static_cast<int>(std::lround(
                             config.internal_blur_board_weight_min * 100.0));
    }
    parts.push_back(blur_weight_tag.str());
  }
  if (config.internal_joint_refine_mode != ati::InternalJointRefineMode::Off) {
    std::ostringstream joint_tag;
    joint_tag << "internal_joint_refine_"
              << ati::ToString(config.internal_joint_refine_mode);
    if (config.internal_joint_refine_target_mode !=
        ati::InternalJointRefineTargetMode::HighResidualAndBlurBadBoard) {
      joint_tag << "_" << ati::ToString(config.internal_joint_refine_target_mode);
    }
    if (std::fabs(config.internal_joint_refine_search_radius_px - 2.0) > 1e-12) {
      joint_tag << "_r"
                << static_cast<int>(
                       std::lround(config.internal_joint_refine_search_radius_px));
    }
    if (std::fabs(config.internal_joint_refine_min_old_residual_px - 1.0) >
        1e-12) {
      joint_tag << "_res"
                << static_cast<int>(
                       std::lround(config.internal_joint_refine_min_old_residual_px *
                                   10.0));
    }
    parts.push_back(joint_tag.str());
  }
  if (config.pre_backend_filter_mode != ati::PreBackendFilterMode::Off) {
    parts.push_back("pre_backend_filter_" +
                    std::string(ati::ToString(config.pre_backend_filter_mode)));
  }
  if (config.backend_polar_angle_weight_mode != "none") {
    std::ostringstream polar_tag;
    polar_tag << "polar_" << config.backend_polar_angle_weight_mode;
    if (config.backend_polar_angle_weight_mode == "fixed_bins") {
      polar_tag << "_bins";
      for (std::size_t scale_index = 0;
           scale_index < config.backend_polar_angle_weight_fixed_bin_scales.size();
           ++scale_index) {
        polar_tag << (scale_index == 0 ? "_" : "-")
                  << static_cast<int>(std::lround(
                         config.backend_polar_angle_weight_fixed_bin_scales[scale_index] *
                         100.0));
      }
    } else if (config.backend_polar_angle_weight_mode == "adaptive_sigma") {
      polar_tag << "_ref"
                << static_cast<int>(std::lround(
                       config.backend_polar_angle_weight_adaptive_sigma_reference_deg))
                << "_g"
                << static_cast<int>(std::lround(
                       config.backend_polar_angle_weight_adaptive_sigma_growth *
                       100.0))
                << "_min"
                << static_cast<int>(std::lround(
                       config.backend_polar_angle_weight_min_scale * 100.0));
    }
    parts.push_back(polar_tag.str());
  }
  if (config.enable_angular_residual_diagnostics) {
    parts.push_back("angular_residual_diag");
  }
  if (config.enable_multi_board_consistency_diagnostics) {
    std::ostringstream consistency_diag_tag;
    consistency_diag_tag << "multi_board_consistency_diag";
    if (config.multi_board_consistency_pose_source != "outer_only") {
      consistency_diag_tag << "_" << config.multi_board_consistency_pose_source;
    }
    if (config.multi_board_consistency_min_outer_points != 4) {
      consistency_diag_tag << "_min_outer"
                           << config.multi_board_consistency_min_outer_points;
    }
    parts.push_back(consistency_diag_tag.str());
  }
  if (config.backend_multi_board_consistency_weighting) {
    std::ostringstream consistency_tag;
    consistency_tag << "multi_board_consistency_weight";
    if (std::fabs(config.backend_consistency_translation_sigma_mm - 3.0) >
        1e-12) {
      consistency_tag << "_ts"
                      << static_cast<int>(std::lround(
                             config.backend_consistency_translation_sigma_mm *
                             10.0));
    }
    if (std::fabs(config.backend_consistency_rotation_sigma_deg - 2.0) >
        1e-12) {
      consistency_tag << "_rs"
                      << static_cast<int>(std::lround(
                             config.backend_consistency_rotation_sigma_deg *
                             10.0));
    }
    if (std::fabs(config.backend_consistency_min_weight - 0.25) > 1e-12) {
      consistency_tag << "_min"
                      << static_cast<int>(std::lround(
                             config.backend_consistency_min_weight * 100.0));
    }
    if (!config.backend_consistency_apply_to_outer) {
      consistency_tag << "_no_outer";
    }
    if (!config.backend_consistency_apply_to_internal) {
      consistency_tag << "_no_internal";
    }
    if (config.backend_consistency_hard_reject_enabled) {
      consistency_tag << "_hard_reject";
    }
    parts.push_back(consistency_tag.str());
  }
  if (config.enable_multiboard_rigidity_diagnostics) {
    std::ostringstream rigidity_tag;
    rigidity_tag << "multiboard_rigidity_diag";
    if (config.multiboard_rigidity_top_k != 30) {
      rigidity_tag << "_top" << config.multiboard_rigidity_top_k;
    }
    if (!config.multiboard_rigidity_use_internal_points) {
      rigidity_tag << "_no_internal";
    }
    if (!config.multiboard_rigidity_use_outer_points) {
      rigidity_tag << "_no_outer";
    }
    parts.push_back(rigidity_tag.str());
  }
  if (config.enable_outer_only_intermediate_calibration) {
    std::ostringstream intermediate_tag;
    intermediate_tag << "outer_only_intermediate";
    if (config.intermediate_diagnostic_only) {
      intermediate_tag << "_diag";
    }
    if (config.use_intermediate_for_round1_internal_regeneration) {
      intermediate_tag << "_round1";
    }
    if (std::fabs(config.intermediate_max_outer_rmse_px - 8.0) > 1e-12) {
      intermediate_tag << "_rmse"
                       << static_cast<int>(std::lround(
                              config.intermediate_max_outer_rmse_px));
    }
    if (config.intermediate_min_visible_boards != 1) {
      intermediate_tag << "_minb"
                       << config.intermediate_min_visible_boards;
    }
    parts.push_back(intermediate_tag.str());
  }
  if (config.internal_observation_weight_mode !=
      ati::InternalObservationWeightMode::Off) {
    std::ostringstream weight_tag;
    weight_tag << "internal_obs_weight_"
               << ati::ToString(config.internal_observation_weight_mode)
               << "_" << config.internal_observation_weight_policy;
    if (std::fabs(config.internal_observation_weight_low_quality_quantile - 0.2) >
        1e-12) {
      weight_tag << "_q"
                 << static_cast<int>(std::lround(
                        config.internal_observation_weight_low_quality_quantile *
                        100.0));
    }
    if (std::fabs(config.internal_observation_weight_min - 0.25) > 1e-12) {
      weight_tag << "_min"
                 << static_cast<int>(std::lround(
                        config.internal_observation_weight_min * 100.0));
    }
    if (std::fabs(config.internal_observation_weight_quality_exponent - 1.0) >
        1e-12) {
      weight_tag << "_p"
                 << static_cast<int>(std::lround(
                        config.internal_observation_weight_quality_exponent *
                        100.0));
    }
    parts.push_back(weight_tag.str());
  }
  if (config.backend_residual_model != "image_plane") {
    std::ostringstream residual_tag;
    residual_tag << "backend_" << config.backend_residual_model;
    if (config.backend_residual_model == "hybrid_edge_angular") {
      residual_tag << "_thr"
                   << static_cast<int>(std::lround(
                          config.backend_hybrid_angular_threshold_deg));
    } else if (config.backend_residual_model == "polar_continuous_hybrid" ||
               config.backend_residual_model == "hybrid" ||
               config.backend_residual_model == "paper_hybrid" ||
               config.backend_residual_model == "hybrid_paper") {
      residual_tag << "_theta"
                   << static_cast<int>(std::lround(
                          config.backend_polar_continuous_hybrid_threshold_deg))
                   << "_temp"
                   << static_cast<int>(std::lround(
                          config.backend_polar_continuous_hybrid_temperature_deg));
    }
    if (config.backend_angular_observed_ray_mode != "dynamic_current_camera") {
      residual_tag << "_"
                   << config.backend_angular_observed_ray_mode;
    }
    if (config.backend_fixed_intrinsics) {
      residual_tag << "_fixed_intrinsics";
    }
    parts.push_back(residual_tag.str());
  }
  if (config.enable_hybrid_pixel_ray_final_refinement) {
    std::ostringstream hybrid_tag;
    if (config.hybrid_pixel_ray_polar_adaptive) {
      hybrid_tag << "pixel_ray_adaptive_l"
                 << static_cast<int>(std::lround(
                        100.0 * config.hybrid_pixel_ray_lambda_min))
                 << "to" << static_cast<int>(std::lround(
                        100.0 * config.hybrid_pixel_ray_lambda_max))
                 << "_theta" << static_cast<int>(std::lround(
                        config.hybrid_pixel_ray_transition_start_deg))
                 << "to" << static_cast<int>(std::lround(
                        config.hybrid_pixel_ray_transition_end_deg));
    } else {
      hybrid_tag << "pixel_ray_refine_l"
                 << static_cast<int>(std::lround(
                        100.0 * config.hybrid_pixel_ray_lambda));
    }
    parts.push_back(hybrid_tag.str());
  }
	  if (!config.backend_optimize_board_poses) {
	    parts.push_back("backend_fixed_board_poses");
	  }
	  if (config.backend_board_pose_prior) {
	    std::ostringstream part;
	    part << "backend_board_pose_prior_t"
	         << static_cast<int>(std::lround(
	                config.backend_board_pose_prior_translation_sigma_mm))
	         << "mm_r"
	         << static_cast<int>(std::lround(
	                config.backend_board_pose_prior_rotation_sigma_deg))
	         << "deg";
	    parts.push_back(part.str());
	  }
	  if (config.backend_point_budget_control) {
	    std::ostringstream part;
	    part << "backend_point_budget_control";
	    if (config.backend_point_budget_control_total_points > 0) {
	      part << "_pts" << config.backend_point_budget_control_total_points;
	    }
	    parts.push_back(part.str());
	  }
	  if (config.backend_max_boards_per_frame_for_ablation > 0) {
	    std::ostringstream part;
	    part << "backend_frame_board_cap"
	         << config.backend_max_boards_per_frame_for_ablation;
	    parts.push_back(part.str());
	  }
	  if (parts.empty()) {
	    return std::string();
	  }
  std::ostringstream stream;
  for (std::size_t index = 0; index < parts.size(); ++index) {
    if (index > 0) {
      stream << "__";
    }
    stream << parts[index];
  }
  return stream.str();
}

RequestedExperimentConfig BuildRequestedExperimentConfig(const CmdArgs& args) {
  RequestedExperimentConfig config;
  config.camera_init_mode = args.camera_init_mode_override.empty()
                                ? ati::CameraInitializationMode::Auto
                                : ati::ParseCameraInitializationMode(
                                      args.camera_init_mode_override);
  config.models = NormalizeStage5ModelFamily(args.stage5_models);
  config.init_refine_mode =
      ati::ParseAutoCameraInitializationRefineMode(
          args.stage5_init_refine_mode);
  config.init_selection_scorer =
      ati::ParseAutoCameraInitializationSelectionScorer(
          args.stage5_init_selection_scorer);
  config.camera_aware_outer_rescue =
      args.stage5_camera_aware_outer_rescue;
  config.camera_aware_outer_rescue_zero_detection_frames =
      args.stage5_camera_aware_outer_rescue_zero_detection_frames;
  config.camera_aware_outer_rescue_zero_detection_frames_set =
      args.stage5_camera_aware_outer_rescue_zero_detection_frames_set;
  config.outer_only_ablation_mode = !args.include_internal_points;
  config.include_internal_points = args.include_internal_points;
  config.run_second_pass = !args.disable_second_pass;
  if (config.outer_only_ablation_mode) {
    config.include_internal_points = false;
    config.run_second_pass = false;
  }
  config.enable_residual_sanity_gate = !args.disable_residual_sanity_gate;
  config.enable_board_pose_fit_gate = args.enable_board_pose_fit_gate;
  config.selection_mode = args.stage5_selection_mode;
  config.selection_residual_sanity_factor =
      args.stage5_selection_residual_sanity_factor;
  config.selection_max_board_observation_rmse =
      args.stage5_selection_max_board_observation_rmse;
  config.selection_kalibr_style_outlier_sigma =
      args.stage5_selection_kalibr_style_outlier_sigma;
  config.selection_kalibr_style_min_abs_threshold_px =
      args.stage5_selection_kalibr_style_min_abs_threshold_px;
  config.selection_kalibr_style_min_views_before_filter =
      args.stage5_selection_kalibr_style_min_views_before_filter;
  config.enable_trial_backend_frame_board_selection =
      args.stage5_enable_trial_backend_frame_board_selection;
  config.trial_backend_selection_mode =
      args.stage5_trial_backend_selection_mode;
  config.trial_backend_selection_budget_mode =
      args.stage5_trial_backend_selection_budget_mode;
  config.trial_backend_selection_budget_mode_set =
      args.stage5_trial_backend_selection_budget_mode_set;
  config.trial_backend_selection_candidate_order =
      args.stage5_trial_backend_selection_candidate_order;
  config.trial_backend_selection_candidate_order_set =
      args.stage5_trial_backend_selection_candidate_order_set;
  config.trial_backend_selection_info_gain_proxy_mode =
      args.stage5_trial_backend_selection_info_gain_proxy_mode;
  config.trial_backend_selection_batch_granularity =
      args.stage5_trial_backend_selection_batch_granularity;
  config.trial_backend_selection_acceptance_policy =
      args.stage5_trial_backend_selection_acceptance_policy;
  config.trial_backend_selection_mi_tol =
      args.stage5_trial_backend_selection_mi_tol;
  config.trial_backend_selection_mi_tol_set =
      args.stage5_trial_backend_selection_mi_tol_set;
  config.trial_backend_selection_rank_threshold =
      args.stage5_trial_backend_selection_rank_threshold;
  config.checkerboard_huber_delta_pixels =
      args.stage5_checkerboard_huber_delta_pixels;
  config.checkerboard_outlier_filter_enabled =
      args.stage5_checkerboard_outlier_filter_enabled;
  config.checkerboard_outlier_sigma =
      args.stage5_checkerboard_outlier_sigma;
  config.checkerboard_min_inlier_ratio =
      args.stage5_checkerboard_min_inlier_ratio;
  config.checkerboard_min_retained_points =
      args.stage5_checkerboard_min_retained_points;
  config.trial_backend_selection_candidate_shuffle_seed =
      args.stage5_trial_backend_selection_candidate_shuffle_seed;
  config.trial_backend_selection_candidate_shuffle_seed_set =
      args.stage5_trial_backend_selection_candidate_shuffle_seed_set;
  config.trial_backend_selection_incremental =
      args.stage5_trial_backend_selection_incremental;
  config.trial_backend_selection_carry_accepted_trial_state =
      args.stage5_trial_backend_selection_carry_accepted_trial_state;
  config.trial_backend_selection_optimize_intrinsics =
      args.stage5_trial_backend_selection_optimize_intrinsics;
  config.trial_backend_selection_delayed_intrinsics_release =
      args.stage5_trial_backend_selection_delayed_intrinsics_release;
  config.trial_backend_selection_intrinsics_release_iteration =
      args.stage5_trial_backend_selection_intrinsics_release_iteration;
  config.trial_backend_selection_persistent_intrinsics_anchor_prior =
      args.stage5_trial_backend_selection_persistent_intrinsics_anchor_prior;
  config.trial_backend_selection_persistent_fix_board_layout =
      args.stage5_trial_backend_selection_persistent_fix_board_layout;
  config.trial_backend_selection_persistent_intrinsics_anchor_weight_xi_alpha =
      args.stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_xi_alpha;
  config.trial_backend_selection_persistent_intrinsics_anchor_weight_focal =
      args.stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_focal;
  config.trial_backend_selection_persistent_intrinsics_anchor_weight_principal =
      args.stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_principal;
  config.trial_backend_selection_persistent_max_focal_relative_step =
      args.stage5_trial_backend_selection_persistent_max_focal_relative_step;
  config.trial_backend_selection_persistent_max_principal_step_px =
      args.stage5_trial_backend_selection_persistent_max_principal_step_px;
  config.trial_backend_selection_persistent_max_xi_alpha_step =
      args.stage5_trial_backend_selection_persistent_max_xi_alpha_step;
  config.trial_backend_selection_max_iterations =
      args.stage5_trial_backend_selection_max_iterations;
  config.trial_backend_selection_max_candidate_additions =
      args.stage5_trial_backend_selection_max_candidate_additions;
  config.trial_backend_selection_adaptive_budget_ratio =
      args.stage5_trial_backend_selection_adaptive_budget_ratio;
  config.trial_backend_selection_adaptive_budget_min =
      args.stage5_trial_backend_selection_adaptive_budget_min;
  config.trial_backend_selection_adaptive_budget_max =
      args.stage5_trial_backend_selection_adaptive_budget_max;
  config.trial_backend_selection_runtime_safety_ceiling =
      args.stage5_trial_backend_selection_runtime_safety_ceiling;
  config.export_selected_case_visualizations =
      args.export_selected_case_visualizations;
  config.trial_backend_selection_outlier_sigma =
      args.stage5_trial_backend_selection_outlier_sigma;
  config.trial_backend_selection_min_abs_threshold_px =
      args.stage5_trial_backend_selection_min_abs_threshold_px;
  config.trial_backend_selection_max_threshold_px =
      args.stage5_trial_backend_selection_max_threshold_px;
  config.trial_backend_selection_accept_max_global_rmse_increase_px =
      args.stage5_trial_backend_selection_accept_max_global_rmse_increase_px;
  config.trial_backend_selection_accept_max_outer_rmse_increase_px =
      args.stage5_trial_backend_selection_accept_max_outer_rmse_increase_px;
  config.trial_backend_selection_accept_max_internal_rmse_increase_px =
      args.stage5_trial_backend_selection_accept_max_internal_rmse_increase_px;
  config.trial_backend_selection_min_candidate_score =
      args.stage5_trial_backend_selection_min_candidate_score;
  config.trial_backend_selection_min_coverage_gain =
      args.stage5_trial_backend_selection_min_coverage_gain;
  config.trial_backend_selection_use_consistency_score =
      args.stage5_trial_backend_selection_use_consistency_score;
  config.trial_backend_selection_consistency_translation_sigma_mm =
      args.stage5_trial_backend_selection_consistency_translation_sigma_mm;
  config.trial_backend_selection_consistency_rotation_sigma_deg =
      args.stage5_trial_backend_selection_consistency_rotation_sigma_deg;
  config.trial_backend_selection_consistency_penalty_weight =
      args.stage5_trial_backend_selection_consistency_penalty_weight;
  config.trial_backend_selection_consistency_max_translation_error_mm =
      args.stage5_trial_backend_selection_consistency_max_translation_error_mm;
  config.trial_backend_selection_consistency_max_rotation_error_deg =
      args.stage5_trial_backend_selection_consistency_max_rotation_error_deg;
  config.trial_backend_selection_consistency_max_local_outer_rmse_px =
      args.stage5_trial_backend_selection_consistency_max_local_outer_rmse_px;
  config.trial_backend_selection_max_accepted_per_board =
      args.stage5_trial_backend_selection_max_accepted_per_board;
  config.trial_backend_selection_max_accepted_per_frame =
      args.stage5_trial_backend_selection_max_accepted_per_frame;
  config.trial_backend_selection_frame_cohesion =
      args.stage5_trial_backend_selection_frame_cohesion;
  config.trial_backend_selection_frame_cohesion_max_companions =
      args.stage5_trial_backend_selection_frame_cohesion_max_companions;
  config.trial_backend_selection_frame_cohesion_min_candidate_score =
      args.stage5_trial_backend_selection_frame_cohesion_min_candidate_score;
  config.trial_backend_selection_min_keep_per_board =
      args.stage5_trial_backend_selection_min_keep_per_board;
  config.trial_backend_selection_force_include_frame_board_list =
      args.stage5_trial_backend_selection_force_include_frame_board_list;
  config.trial_backend_selection_seed_frame_board_list =
      args.stage5_trial_backend_selection_seed_frame_board_list;
  config.trial_backend_selection_force_include_list_is_exact_input =
      args.stage5_trial_backend_selection_force_include_list_is_exact_input;
  config.strict_board_observation_acceptance =
      args.strict_board_observation_acceptance;
  config.preserve_frame_board_cohesion = args.preserve_frame_board_cohesion;
  config.ignore_image_evidence_min_quality =
      args.ignore_image_evidence_min_quality;
  config.force_internal_seed_from_prediction =
      args.force_internal_seed_from_prediction;
  config.bypass_internal_seed_filters =
      args.bypass_internal_seed_filters;
  config.internal_corner_filter_mode =
      args.internal_corner_filter_mode;
  config.internal_corner_filter_max_reproj_error =
      args.internal_corner_filter_max_reproj_error;
  config.internal_corner_filter_quality_min =
      args.internal_corner_filter_quality_min;
  config.internal_corner_filter_quality_relaxation_px =
      args.internal_corner_filter_quality_relaxation_px;
  config.internal_corner_filter_adaptive_min_threshold_px =
      args.internal_corner_filter_adaptive_min_threshold_px;
  config.internal_regeneration_diagnostics =
      args.internal_regeneration_diagnostics;
  config.export_internal_seed_step_overlays =
      args.stage5_export_internal_seed_step_overlays;
  config.internal_blur_diagnostics = args.internal_blur_diagnostics;
  config.internal_blur_filter_mode = args.internal_blur_filter_mode;
  config.internal_blur_filter_low_patch_gradient_quantile =
      args.internal_blur_filter_low_patch_gradient_quantile;
  config.internal_blur_filter_min_board_rmse_px =
      args.internal_blur_filter_min_board_rmse_px;
  config.internal_blur_filter_min_board_p95_px =
      args.internal_blur_filter_min_board_p95_px;
  config.internal_pose_rescue_mode = args.internal_pose_rescue_mode;
  config.internal_pose_rescue_max_ray_angle_deg =
      args.internal_pose_rescue_max_ray_angle_deg;
  config.internal_pose_rescue_accept_max_outer_rmse =
      args.internal_pose_rescue_accept_max_outer_rmse;
  config.enable_geometry_prior_outer_seed =
      args.enable_geometry_prior_outer_seed;
  config.geometry_prior_rescue_diagnostic_only =
      args.geometry_prior_rescue_diagnostic_only;
  config.geometry_prior_rescue_use_as_observation =
      args.geometry_prior_rescue_use_as_observation;
  config.geometry_prior_rescue_keep_outer_on_internal_failure =
      args.geometry_prior_rescue_keep_outer_on_internal_failure;
  config.geometry_prior_rescue_allow_geometry_only_pose_refit =
      args.geometry_prior_rescue_allow_geometry_only_pose_refit;
  config.geometry_prior_rescue_subpix_window_radius =
      args.geometry_prior_rescue_subpix_window_radius;
  config.geometry_prior_rescue_max_corner_displacement_px =
      args.geometry_prior_rescue_max_corner_displacement_px;
  config.geometry_prior_rescue_min_corner_response_ratio =
      args.geometry_prior_rescue_min_corner_response_ratio;
  config.geometry_prior_rescue_enable_spherical_refine =
      args.geometry_prior_rescue_enable_spherical_refine;
  config.geometry_prior_rescue_edge_sample_count =
      args.geometry_prior_rescue_edge_sample_count;
  config.geometry_prior_rescue_edge_search_half_width_px =
      args.geometry_prior_rescue_edge_search_half_width_px;
  config.geometry_prior_rescue_min_edge_support_ratio =
      args.geometry_prior_rescue_min_edge_support_ratio;
  config.geometry_prior_rescue_min_edge_gradient_ratio =
      args.geometry_prior_rescue_min_edge_gradient_ratio;
  config.geometry_prior_rescue_accept_max_outer_rmse =
      args.geometry_prior_rescue_accept_max_outer_rmse;
  config.geometry_prior_rescue_accept_max_rotation_error_deg =
      args.geometry_prior_rescue_accept_max_rotation_error_deg;
  config.geometry_prior_rescue_accept_max_translation_error =
      args.geometry_prior_rescue_accept_max_translation_error;
  config.geometry_guided_tag_likelihood_enabled =
      args.geometry_guided_tag_likelihood_enabled;
  config.geometry_guided_tag_likelihood_min_visible_boards =
      args.geometry_guided_tag_likelihood_min_visible_boards;
  config.geometry_guided_tag_likelihood_max_expected_hamming =
      args.geometry_guided_tag_likelihood_max_expected_hamming;
  config.geometry_guided_tag_likelihood_min_hamming_margin =
      args.geometry_guided_tag_likelihood_min_hamming_margin;
  config.geometry_guided_tag_likelihood_min_contrast =
      args.geometry_guided_tag_likelihood_min_contrast;
  config.geometry_guided_tag_likelihood_allow_single_anchor =
      args.geometry_guided_tag_likelihood_allow_single_anchor;
  config.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse =
      args.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse;
  config.geometry_guided_tag_likelihood_single_anchor_max_expected_hamming =
      args.geometry_guided_tag_likelihood_single_anchor_max_expected_hamming;
  config.geometry_guided_tag_likelihood_single_anchor_min_hamming_margin =
      args.geometry_guided_tag_likelihood_single_anchor_min_hamming_margin;
  config.geometry_guided_tag_likelihood_single_anchor_min_contrast =
      args.geometry_guided_tag_likelihood_single_anchor_min_contrast;

  if (args.stage5_enable_frozen_recovery_baseline) {
    // Keep the canonical baseline recovery bundle explicit and centralized.
    // The topology identity branch is still image- and pose-validated; this
    // preset only prevents one of its prerequisite stages from being omitted.
    config.frozen_recovery_baseline_preset = true;
    config.camera_aware_outer_rescue = true;
    config.camera_aware_outer_rescue_zero_detection_frames = true;
    config.camera_aware_outer_rescue_zero_detection_frames_set = true;
    config.enable_geometry_prior_outer_seed = true;
    config.geometry_prior_rescue_diagnostic_only = false;
    config.geometry_prior_rescue_use_as_observation = true;
    config.geometry_prior_rescue_keep_outer_on_internal_failure = false;
    config.geometry_prior_rescue_allow_geometry_only_pose_refit = true;
    config.geometry_prior_rescue_enable_spherical_refine = true;
  }

  config.enable_outer_only_intermediate_calibration =
      args.enable_outer_only_intermediate_calibration;
  config.intermediate_diagnostic_only =
      args.intermediate_diagnostic_only;
  config.use_intermediate_for_round1_internal_regeneration =
      args.use_intermediate_for_round1_internal_regeneration;
  config.use_intermediate_for_full_frontend_regeneration =
      args.use_intermediate_for_full_frontend_regeneration;
  if (config.outer_only_ablation_mode) {
    config.internal_regeneration_diagnostics = false;
    config.export_internal_seed_step_overlays = false;
    config.internal_blur_diagnostics = false;
    config.use_intermediate_for_round1_internal_regeneration = false;
    config.use_intermediate_for_full_frontend_regeneration = false;
  }
  config.intermediate_optimize_intrinsics =
      args.intermediate_optimize_intrinsics;
  config.intermediate_optimize_board_poses =
      args.intermediate_optimize_board_poses;
  config.intermediate_optimize_frame_poses =
      args.intermediate_optimize_frame_poses;
  config.intermediate_intrinsics_release_iteration =
      args.intermediate_intrinsics_release_iteration;
  config.intermediate_max_outer_rmse_px =
      args.intermediate_max_outer_rmse_px;
  config.intermediate_min_visible_boards =
      args.intermediate_min_visible_boards;
  config.pre_backend_filter_mode = args.pre_backend_filter_mode;
  config.pre_backend_filter_threshold_mode =
      args.pre_backend_filter_threshold_mode;
  config.pre_backend_filter_sigma = args.pre_backend_filter_sigma;
  config.pre_backend_filter_min_abs_threshold_px =
      args.pre_backend_filter_min_abs_threshold_px;
  config.internal_joint_refine_mode = args.internal_joint_refine_mode;
  config.internal_joint_refine_target_mode =
      args.internal_joint_refine_target_mode;
  config.internal_joint_refine_search_radius_px =
      args.internal_joint_refine_search_radius_px;
  config.internal_joint_refine_max_displacement_px =
      args.internal_joint_refine_max_displacement_px;
  config.internal_joint_refine_geometry_sigma_px =
      args.internal_joint_refine_geometry_sigma_px;
  config.internal_joint_refine_observation_sigma_px =
      args.internal_joint_refine_observation_sigma_px;
  config.internal_joint_refine_subpix_window_radius =
      args.internal_joint_refine_subpix_window_radius;
  config.internal_joint_refine_min_objective_improvement =
      args.internal_joint_refine_min_objective_improvement;
  config.internal_joint_refine_min_old_residual_px =
      args.internal_joint_refine_min_old_residual_px;
  config.internal_joint_refine_low_patch_gradient_quantile =
      args.internal_joint_refine_low_patch_gradient_quantile;
  config.internal_joint_refine_min_board_rmse_px =
      args.internal_joint_refine_min_board_rmse_px;
  config.internal_joint_refine_min_board_p95_px =
      args.internal_joint_refine_min_board_p95_px;
  config.internal_joint_refine_min_corner_response_gain =
      args.internal_joint_refine_min_corner_response_gain;
  config.internal_joint_refine_min_board_internal_improvement_px =
      args.internal_joint_refine_min_board_internal_improvement_px;
  config.internal_joint_refine_min_refined_point_count_per_board =
      args.internal_joint_refine_min_refined_point_count_per_board;
  config.internal_joint_refine_accept_max_global_outer_delta_px =
      args.internal_joint_refine_accept_max_global_outer_delta_px;
  config.internal_joint_refine_accept_max_frame_outer_delta_px =
      args.internal_joint_refine_accept_max_frame_outer_delta_px;
  config.internal_joint_refine_acceptance_backend_max_iterations =
      args.internal_joint_refine_acceptance_backend_max_iterations;
  config.internal_observation_weight_mode =
      args.internal_observation_weight_mode;
  config.internal_observation_weight_policy =
      args.internal_observation_weight_policy;
  config.backend_polar_angle_weight_mode =
      args.backend_polar_angle_weight_mode;
  config.backend_polar_angle_weight_bin_edges_deg =
      ParseCommaSeparatedDoubles(args.backend_polar_angle_weight_bin_edges,
                                 "backend polar angle weight bin edge");
  config.backend_polar_angle_weight_fixed_bin_scales =
      ParseCommaSeparatedDoubles(
          args.backend_polar_angle_weight_fixed_bin_scales,
          "backend polar angle fixed bin scale");
  config.backend_polar_angle_weight_adaptive_sigma_reference_deg =
      args.backend_polar_angle_weight_adaptive_sigma_reference_deg;
  config.backend_polar_angle_weight_adaptive_sigma_growth =
      args.backend_polar_angle_weight_adaptive_sigma_growth;
  config.backend_polar_angle_weight_min_scale =
      args.backend_polar_angle_weight_min_scale;
  config.backend_residual_model = args.backend_residual_model;
  config.backend_hybrid_angular_threshold_deg =
      args.backend_hybrid_angular_threshold_deg;
  config.backend_outer_residual_model = args.backend_outer_residual_model;
  config.backend_internal_residual_model = args.backend_internal_residual_model;
  config.backend_use_point_type_residual_split =
      args.backend_use_point_type_residual_split;
  config.backend_enable_angular_auxiliary_residual =
      args.backend_enable_angular_auxiliary_residual;
  config.backend_angular_auxiliary_weight =
      args.backend_angular_auxiliary_weight;
  config.backend_angular_auxiliary_normalized =
      args.backend_angular_auxiliary_normalized;
  config.backend_angular_auxiliary_apply_to_outer =
      args.backend_angular_auxiliary_apply_to_outer;
  config.backend_angular_auxiliary_apply_to_internal =
      args.backend_angular_auxiliary_apply_to_internal;
  config.backend_polar_continuous_hybrid_threshold_deg =
      args.backend_polar_continuous_hybrid_threshold_deg;
  config.backend_polar_continuous_hybrid_temperature_deg =
      args.backend_polar_continuous_hybrid_temperature_deg;
  config.backend_normalized_angular_reference_sigma_px =
      args.backend_normalized_angular_reference_sigma_px;
  config.backend_normalized_angular_min_sigma_rad =
      args.backend_normalized_angular_min_sigma_rad;
  config.backend_normalized_angular_max_weight_scale =
      args.backend_normalized_angular_max_weight_scale;
  config.backend_pixel_residual_weight =
      args.backend_pixel_residual_weight;
  config.backend_chordal_residual_weight =
      args.backend_chordal_residual_weight;
  config.enable_hybrid_pixel_ray_final_refinement =
      args.enable_hybrid_pixel_ray_final_refinement;
  config.hybrid_pixel_ray_lambda = args.hybrid_pixel_ray_lambda;
  config.hybrid_pixel_ray_polar_adaptive =
      args.hybrid_pixel_ray_polar_adaptive;
  config.hybrid_pixel_ray_lambda_min = args.hybrid_pixel_ray_lambda_min;
  config.hybrid_pixel_ray_lambda_max = args.hybrid_pixel_ray_lambda_max;
  config.hybrid_pixel_ray_transition_start_deg =
      args.hybrid_pixel_ray_transition_start_deg;
  config.hybrid_pixel_ray_transition_end_deg =
      args.hybrid_pixel_ray_transition_end_deg;
  config.hybrid_pixel_ray_max_iterations =
      args.hybrid_pixel_ray_max_iterations;
  config.hybrid_pixel_ray_pixel_scale_floor =
      args.hybrid_pixel_ray_pixel_scale_floor;
  config.hybrid_pixel_ray_ray_scale_floor =
      args.hybrid_pixel_ray_ray_scale_floor;
  config.backend_angular_use_normalize_jacobian =
      args.backend_angular_use_normalize_jacobian;
	config.backend_angular_local_whitening =
	    args.backend_angular_local_whitening;
	config.backend_angular_local_whitening_pixel_sigma_px =
	    args.backend_angular_local_whitening_pixel_sigma_px;
	config.backend_angular_local_whitening_covariance_damping =
	    args.backend_angular_local_whitening_covariance_damping;
	config.backend_angular_local_whitening_min_sigma_rad =
	    args.backend_angular_local_whitening_min_sigma_rad;
	config.backend_angular_local_whitening_max_weight =
	    args.backend_angular_local_whitening_max_weight;
	  config.backend_angular_observed_ray_mode =
	      args.backend_angular_observed_ray_mode;
	  config.backend_board_pose_parameterization =
	      args.backend_board_pose_parameterization;
	  config.backend_optimize_board_poses = args.backend_optimize_board_poses;
	  config.backend_board_pose_prior = args.backend_board_pose_prior;
	  config.backend_board_pose_prior_translation_sigma_mm =
	      args.backend_board_pose_prior_translation_sigma_mm;
	  config.backend_board_pose_prior_rotation_sigma_deg =
	      args.backend_board_pose_prior_rotation_sigma_deg;
	  config.backend_point_budget_control = args.backend_point_budget_control;
	  config.backend_point_budget_control_total_points =
	      args.backend_point_budget_control_total_points;
	  config.backend_point_budget_control_seed =
	      args.backend_point_budget_control_seed;
	  config.backend_max_boards_per_frame_for_ablation =
	      args.backend_max_boards_per_frame_for_ablation;
	  config.backend_fixed_intrinsics = args.backend_fixed_intrinsics;
  config.enable_angular_residual_diagnostics =
      args.enable_angular_residual_diagnostics;
  config.angular_residual_bin_edges_deg =
      ParseCommaSeparatedDoubles(args.angular_residual_bin_edges,
                                 "angular residual bin edge");
  config.backend_multi_board_consistency_weighting =
      args.backend_multi_board_consistency_weighting;
  config.backend_consistency_pose_source =
      args.backend_consistency_pose_source;
  config.backend_consistency_weight_mode =
      args.backend_consistency_weight_mode;
  config.backend_consistency_translation_sigma_mm =
      args.backend_consistency_translation_sigma_mm;
  config.backend_consistency_rotation_sigma_deg =
      args.backend_consistency_rotation_sigma_deg;
  config.backend_consistency_min_weight =
      args.backend_consistency_min_weight;
  config.backend_consistency_apply_to_outer =
      args.backend_consistency_apply_to_outer;
  config.backend_consistency_apply_to_internal =
      args.backend_consistency_apply_to_internal;
  config.backend_consistency_hard_reject_enabled =
      args.backend_consistency_hard_reject_enabled;
  config.backend_consistency_hard_reject_translation_mm =
      args.backend_consistency_hard_reject_translation_mm;
  config.backend_consistency_hard_reject_rotation_deg =
      args.backend_consistency_hard_reject_rotation_deg;
  config.backend_consistency_hard_reject_residual_px =
      args.backend_consistency_hard_reject_residual_px;
  config.backend_consistency_dump_weight_summary =
      args.backend_consistency_dump_weight_summary;
  config.internal_blur_board_weight_mode =
      args.internal_blur_board_weight_mode;
  config.internal_blur_board_weight_low_patch_gradient_quantile =
      args.internal_blur_board_weight_low_patch_gradient_quantile;
  config.internal_blur_board_weight_min_board_rmse_px =
      args.internal_blur_board_weight_min_board_rmse_px;
  config.internal_blur_board_weight_min_board_p95_px =
      args.internal_blur_board_weight_min_board_p95_px;
  config.internal_blur_board_weight_min =
      args.internal_blur_board_weight_min;
  config.internal_blur_board_weight_gradient_exponent =
      args.internal_blur_board_weight_gradient_exponent;
  config.internal_observation_weight_low_quality_quantile =
      args.internal_observation_weight_low_quality_quantile;
  config.internal_observation_weight_min =
      args.internal_observation_weight_min;
  config.internal_observation_weight_quality_exponent =
      args.internal_observation_weight_quality_exponent;
  config.internal_observation_weight_residual_consistency_sigma =
      args.internal_observation_weight_residual_consistency_sigma;
  config.internal_observation_weight_residual_consistency_min_rmse =
      args.internal_observation_weight_residual_consistency_min_rmse;
  config.enable_multi_board_consistency_diagnostics =
      args.enable_multi_board_consistency_diagnostics;
  config.multi_board_consistency_pose_source =
      args.multi_board_consistency_pose_source;
  config.multi_board_consistency_min_outer_points =
      args.multi_board_consistency_min_outer_points;
  config.enable_multiboard_rigidity_diagnostics =
      args.enable_multiboard_rigidity_diagnostics;
  config.enable_global_scene_state_consistency_audit =
      args.enable_global_scene_state_consistency_audit;
  config.enable_stage5_selection_diagnostics =
      args.enable_stage5_selection_diagnostics;
  config.multiboard_rigidity_top_k = args.multiboard_rigidity_top_k;
  config.multiboard_rigidity_rotation_bad_threshold_deg =
      args.multiboard_rigidity_rotation_bad_threshold_deg;
  config.multiboard_rigidity_translation_bad_threshold =
      args.multiboard_rigidity_translation_bad_threshold;
  config.multiboard_rigidity_reprojection_delta_bad_threshold_px =
      args.multiboard_rigidity_reprojection_delta_bad_threshold_px;
  config.multiboard_rigidity_use_internal_points =
      args.multiboard_rigidity_use_internal_points;
  config.multiboard_rigidity_use_outer_points =
      args.multiboard_rigidity_use_outer_points;

  switch (args.intrinsics_release_mode) {
    case IntrinsicsReleaseMode::Delayed:
      config.frontend_optimize_intrinsics = true;
      config.frontend_intrinsics_release_iteration =
          args.intrinsics_release_iteration;
      config.frontend_second_pass_intrinsics_release_iteration =
          args.second_pass_intrinsics_release_iteration;
      config.frontend_intrinsics_release_mode = IntrinsicsReleaseMode::Delayed;
      config.backend_optimize_intrinsics = true;
      config.backend_delayed_intrinsics_release = true;
      config.backend_intrinsics_release_iteration =
          args.second_pass_intrinsics_release_iteration;
      config.backend_intrinsics_release_mode = IntrinsicsReleaseMode::Delayed;
      break;
    case IntrinsicsReleaseMode::Immediate:
      config.frontend_optimize_intrinsics = true;
      config.frontend_intrinsics_release_iteration = 0;
      config.frontend_second_pass_intrinsics_release_iteration = 0;
      config.frontend_intrinsics_release_mode = IntrinsicsReleaseMode::Immediate;
      config.backend_optimize_intrinsics = true;
      config.backend_delayed_intrinsics_release = false;
      config.backend_intrinsics_release_iteration = 0;
      config.backend_intrinsics_release_mode = IntrinsicsReleaseMode::Immediate;
      break;
    case IntrinsicsReleaseMode::PoseOnly:
      config.frontend_optimize_intrinsics = false;
      config.frontend_intrinsics_release_iteration = 0;
      config.frontend_second_pass_intrinsics_release_iteration = 0;
      config.frontend_intrinsics_release_mode = IntrinsicsReleaseMode::PoseOnly;
      config.backend_optimize_intrinsics = false;
      config.backend_delayed_intrinsics_release = false;
      config.backend_intrinsics_release_iteration = 0;
      config.backend_intrinsics_release_mode = IntrinsicsReleaseMode::PoseOnly;
      break;
  }

  if (config.backend_fixed_intrinsics) {
    config.backend_optimize_intrinsics = false;
    config.backend_delayed_intrinsics_release = false;
    config.backend_intrinsics_release_iteration = 0;
    config.backend_intrinsics_release_mode = IntrinsicsReleaseMode::PoseOnly;
  }

  config.experiment_tag = args.experiment_tag;
  if (config.experiment_tag.empty() && HasAblationOverrides(config)) {
    config.experiment_tag = BuildDeterministicExperimentTag(config);
  }
  if (!config.experiment_tag.empty()) {
    config.effective_protocol_label =
        config.frozen_baseline_label + "__" + config.experiment_tag;
  }
  return config;
}

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " (--image IMAGE_OR_DIR|--stage5-precomputed-observations-dir DIR)"
      << " --config APRILTAG_INTERNAL_YAML --output OUTPUT_DIR"
      << " --kalibr-camchain CAMCHAIN_YAML [--all]"
      << " [--test-image IMAGE_OR_DIR|--holdout-image IMAGE_OR_DIR]\n\n"
      << "Core options:\n"
      << "  --image PATH                         Training image or directory.\n"
      << "  --test-image PATH                    Optional holdout image or directory.\n"
      << "  --stage5-precomputed-observations-dir PATH\n"
      << "                                      Imported BabelCalib MAT training observations.\n"
      << "  --stage5-precomputed-holdout-observations-dir PATH\n"
      << "                                      Frozen imported MAT holdout observations.\n"
      << "  --stage5-allow-image-training-with-precomputed-holdout\n"
      << "                                      Explicit paper mode: image frontend training\n"
      << "                                      plus frozen precomputed holdout.\n"
      << "  --stage5-init-auxiliary-precomputed-observations-dir PATH\n"
      << "                                      Additional calibration session used only for\n"
      << "                                      shared-intrinsics initialization; repeatable.\n"
      << "  --stage5-precomputed-target-mode auto|single_board|multi_board\n"
      << "                                      Target topology mode; default auto.\n"
      << "  --stage5-precomputed-init-use-all-points 0|1\n"
      << "                                      Experimental: use imported outer and internal\n"
      << "                                      points in camera initialization LM; default 0.\n"
      << "  --stage5-precomputed-init-point-scope all|outer_only|internal_only\n"
      << "                                      Diagnostic control-point scope for the\n"
      << "                                      imported initialization LM; default all.\n"
      << "  --stage5-external-holdout-self-frontend-prepass\n"
      << "                                      For external --test-image evaluation,\n"
      << "                                      build test observations from a frozen\n"
      << "                                      frontend prepass on the test set itself.\n"
      << "  --stage5-frontend-only               Run detection, initialization, rescue, and\n"
      << "                                      observation regeneration on all frames; skip\n"
      << "                                      selection incremental BA and backend evaluation.\n"
      << "  --stage5-enable-camera-aware-sphere-patch-zero-detection\n"
      << "                                      Run the full-view sphere-patch atlas for raw\n"
      << "                                      zero-detection frames; exact-ID/Hamming-0 only.\n"
      << "  --stage5-enable-geometry-guided-tag-likelihood-single-anchor\n"
      << "                                      Allow a direct exact-ID single Tag to seed\n"
      << "                                      stricter geometry-guided recovery.\n"
      << "  --config PATH                        AprilTag-internal config YAML.\n"
      << "  --kalibr-camchain PATH               Kalibr reference camchain for comparison.\n"
      << "  --reference-intrinsics-yaml LABEL:PATH\n"
      << "                                      Extra reference camera YAML/camchain for\n"
      << "                                      evaluation only; repeatable. LABEL is\n"
      << "                                      optional when using just PATH.\n"
      << "  --models ds-none|pinhole-equi|kb|eucm-none|mei|ucm|omni-none|omni-radtan\n"
      << "                                      Stage5 calibration camera model family;\n"
      << "                                      default ds-none. This follows Kalibr-style\n"
      << "                                      explicit model selection instead of target YAML.\n"
      << "  --stage5-init-refine-mode kalibr_outer_lm|coordinate_search|none\n"
      << "                                      Camera initialization refinement mode;\n"
      << "                                      default kalibr_outer_lm.\n"
      << "  --stage5-init-selection-scorer pose_marginalized_principal|legacy_fixed_pose\n"
      << "                                      Outer-only initialization view scorer;\n"
      << "                                      default pose_marginalized_principal.\n"
      << "  --stage5-enable-init-principal-profile\n"
      << "                                      Diagnostic-only 3x3 fixed-cu/cv profile;\n"
      << "                                      does not update selected intrinsics.\n"
      << "  --stage5-init-principal-profile-radius-px PX\n"
      << "                                      Profile offset radius; default 10 px.\n"
      << "  --stage5-enable-init-fixed-layout-diagnostic\n"
      << "                                      Diagnostic-only frozen-layout/shared-frame\n"
      << "                                      pose LM; never commits intrinsics.\n"
      << "  --stage5-enable-init-board-jackknife-diagnostic\n"
      << "                                      Diagnostic-only leave-one-board-out\n"
      << "                                      camera refinement; never commits.\n"
      << "  --stage5-enable-init-coverage-weighted-diagnostic\n"
      << "                                      Diagnostic-only 4x4 inverse-density\n"
      << "                                      view weighting; never commits.\n"
      << "  --stage5-init-near-tie-prefer-lower-focal\n"
      << "                                      Explicit DS basin-selection ablation;\n"
      << "                                      default disabled.\n"
      << "  --stage5-init-near-tie-relative-objective-tolerance R\n"
      << "                                      Relative near-tie tolerance; default 0.\n"
      << "  --stage5-enable-camera-aware-outer-rescue\n"
      << "                                      Enable provisional-camera sphere-tile\n"
      << "                                      outer-tag rescue; enabled by default.\n"
      << "  --stage5-disable-camera-aware-outer-rescue\n"
      << "                                      Disable it for detector ablation.\n"
      << "  --stage5-enable-frozen-recovery-baseline\n"
      << "                                      Force the canonical recovery bundle:\n"
      << "                                      camera-aware rescue, zero-detection atlas,\n"
      << "                                      geometry-prior observation, spherical\n"
      << "                                      refinement, and topology identity rescue.\n"
      << "  --output PATH                        Output directory.\n"
      << "  --runtime-mode research|fast         Runtime preset.\n"
      << "  --cache-dir PATH                     Optional frontend cache directory.\n"
      << "  --split-mode random_holdout_ratio|deterministic_stride\n"
      << "                                      Internal train/holdout split when no\n"
      << "                                      --test-image is provided; default random.\n"
      << "  --holdout-ratio R                    Random holdout fraction; default 0.30.\n"
      << "  --split-seed N                       Random split seed; default 1337.\n"
      << "  --holdout-stride N --holdout-offset N\n"
      << "                                      Legacy deterministic split controls.\n"
      << "  --stage5-no-holdout                 Use all frames for training;\n"
      << "                                      evaluation is on the same frames.\n"
      << "  --all                                Run the full Stage5 pipeline.\n\n"
      << "Baseline controls:\n"
      << "  --include-internal-points 0|1        Internal/outer-only ablation.\n"
      << "  --backend-fixed-intrinsics           Pose/structure-only backend ablation.\n"
      << "  --disable-second-pass                Disable Round2 regeneration.\n"
      << "  --intrinsics-release-mode delayed|immediate|pose_only\n"
      << "  --intrinsics-release-iteration N\n"
      << "  --second-pass-intrinsics-release-iteration N\n\n"
      << "Trial backend selection:\n"
      << "  --stage5-enable-trial-backend-frame-board-selection\n"
      << "  --stage5-trial-backend-selection-mode strict_rmse|kalibr_style_batch"
      << " (default kalibr_style_batch; strict_rmse for ablation)\n"
      << "  --stage5-trial-backend-selection-budget-mode fixed|adaptive|kalibr_style\n"
      << "  --stage5-trial-backend-selection-candidate-order score_sorted|random_shuffle|intrinsics_information_greedy\n"
      << "  --stage5-trial-backend-selection-info-gain-proxy-mode legacy|intrinsics_jacobian\n"
      << "  --stage5-trial-backend-selection-batch-granularity frame_board|frame|frame_board_then_frame\n"
      << "  --stage5-trial-backend-selection-acceptance-policy residual_score|kalibr_information_gain\n"
      << "  --stage5-trial-backend-selection-mi-tol X\n"
      << "  --stage5-trial-backend-selection-rank-threshold X\n"
      << "  --stage5-checkerboard-huber-delta-px X"
      << " (dense single-board robust scale; default 1.5)\n"
      << "  --stage5-checkerboard-outlier-filter 0|1"
      << " (hard training-outlier filter after initial trial; default 1)\n"
      << "  --stage5-checkerboard-outlier-sigma X"
      << " (median/MAD threshold multiplier; default 4.0)\n"
      << "  --stage5-checkerboard-min-inlier-ratio X"
      << " (minimum retained fraction per view; default 0.5)\n"
      << "  --stage5-checkerboard-min-retained-points N"
      << " (minimum retained controls per view; default 8)\n"
      << "  --stage5-trial-backend-selection-candidate-shuffle-seed N"
      << " (optional reproducibility; default non-fixed seed in kalibr_style_batch)\n"
      << "  --stage5-trial-backend-selection-carry-accepted-trial-state 0|1"
      << " (default 1; Kalibr-style accepted state inheritance)\n"
      << "  --stage5-trial-backend-selection-optimize-intrinsics 0|1"
      << " (default 1; optimize intrinsics in trial BA)\n"
      << "  --stage5-trial-backend-selection-delayed-intrinsics-release 0|1"
      << " (default 1)\n"
      << "  --stage5-trial-backend-selection-intrinsics-release-iteration N"
      << " (default 1)\n"
      << "  --stage5-trial-backend-selection-persistent-intrinsics-anchor-prior 0|1"
      << " (default 1; trust-region anchor for candidate intrinsics)\n"
      << "  --stage5-trial-backend-selection-persistent-fix-board-layout 0|1"
      << " (default 0; freeze the estimated rigid board layout during persistent BA)\n"
      << "  --stage5-trial-backend-selection-persistent-intrinsics-anchor-weight-xi-alpha W\n"
      << "  --stage5-trial-backend-selection-persistent-intrinsics-anchor-weight-focal W\n"
      << "  --stage5-trial-backend-selection-persistent-intrinsics-anchor-weight-principal W\n"
      << "  --stage5-trial-backend-selection-persistent-max-focal-relative-step R\n"
      << "  --stage5-trial-backend-selection-persistent-max-principal-step-px PX\n"
      << "  --stage5-trial-backend-selection-persistent-max-xi-alpha-step X\n"
      << "  --stage5-trial-backend-selection-max-candidate-additions N\n"
      << "  --stage5-trial-backend-selection-adaptive-budget-ratio R\n"
      << "  --stage5-trial-backend-selection-adaptive-budget-min N\n"
      << "  --stage5-trial-backend-selection-adaptive-budget-max N\n"
      << "  --stage5-trial-backend-selection-runtime-safety-ceiling N\n"
      << "  --stage5-trial-backend-selection-min-candidate-score S\n"
      << "  --stage5-trial-backend-selection-min-coverage-gain G\n"
      << "  --stage5-trial-backend-selection-max-accepted-per-board N\n"
      << "  --stage5-trial-backend-selection-max-accepted-per-frame N\n"
      << "  --stage5-trial-backend-selection-frame-cohesion\n"
      << "  --stage5-trial-backend-selection-force-include-list-is-exact-input 0|1"
      << " (paper ablation: use force-list as the only backend input)\n"
      << "  --stage5-trial-backend-selection-seed-frame-board-list CSV"
      << " (paper ablation: exact common persistent seed input)\n"
      << "  --stage5-trial-backend-selection-frame-cohesion-max-companions N"
      << " (<=0 auto from observed boards)\n"
	      << "  --stage5-trial-backend-selection-close-distance-boost\n"
	      << "  --stage5-trial-backend-selection-close-distance-frame-admission\n\n"
	      << "Backend input ablations:\n"
	      << "  --backend-board-pose-parameterization "
              << "reference_chain|independent_frame_board_pose\n"
	      << "  --stage5-backend-board-pose-prior\n"
	      << "  --stage5-backend-board-pose-prior-translation-sigma-mm MM\n"
	      << "  --stage5-backend-board-pose-prior-rotation-sigma-deg DEG\n"
	      << "  --stage5-backend-point-budget-control\n"
	      << "  --stage5-backend-point-budget-control-total-points N\n"
	      << "  --stage5-backend-point-budget-control-seed N\n"
	      << "  --stage5-backend-max-boards-per-frame-for-ablation N\n\n"
	      << "Optional post-selection refinement:\n"
	      << "  --stage5-enable-hybrid-pixel-ray-final-refinement\n"
	      << "  --stage5-hybrid-pixel-ray-lambda L (default 0.5)\n"
	      << "  --stage5-hybrid-pixel-ray-polar-adaptive\n"
	      << "  --stage5-hybrid-pixel-ray-lambda-min L (default 0.2)\n"
	      << "  --stage5-hybrid-pixel-ray-lambda-max L (default 0.8)\n"
	      << "  --stage5-hybrid-pixel-ray-transition-start-deg D (default 30)\n"
	      << "  --stage5-hybrid-pixel-ray-transition-end-deg D (default 70)\n"
	      << "  --stage5-hybrid-pixel-ray-max-iterations N (default 12)\n"
	      << "  --stage5-hybrid-pixel-ray-pixel-scale-floor S (default 1e-3)\n"
      << "  --stage5-hybrid-pixel-ray-ray-scale-floor S (default 1e-6)\n\n"
      << "  --stage5-large-intrinsic-perturbation PROFILE\n"
      << "                                      Apply DS P1/P2/P3/P4 after\n"
      << "                                      internal recovery and before\n"
      << "                                      selection/incremental BA.\n\n"
      << "  --stage5-large-intrinsic-perturbation-scale S\n"
      << "                                      Unit-interval amount along the\n"
      << "                                      selected direction (default 1).\n\n"
      << "  --stage5-large-intrinsic-perturbation-reference-scene PATH\n"
      << "                                      Reuse a saved unperturbed scene\n"
      << "                                      for paired ablation runs.\n\n"
      << "  --stage5-large-intrinsic-perturbation-outer-only-after-application\n"
      << "                                      Drop frozen internal residuals only\n"
      << "                                      after perturbation, before selection.\n\n"
	      << "Diagnostics and visualizations:\n"
      << "  --internal-regeneration-diagnostics\n"
      << "  --stage5-enable-selection-diagnostics\n"
      << "  --stage5-export-holdout-reprojection-visualizations\n"
      << "  --stage5-disable-selected-case-visualizations\n"
      << "  --stage5-holdout-visualization-top-k N\n"
      << "  --stage5-enable-global-scene-state-consistency-audit\n"
      << "  --stage5-enable-polar-angle-diagnostics\n"
      << "  --stage5-enable-multi-board-consistency-diagnostics\n\n"
      << "Experimental branches are still parsed for existing scripts, but are\n"
      << "intentionally omitted from this short help to keep the baseline entry\n"
      << "point readable. See ParseArgs in this file for legacy/research flags.\n";
}

CmdArgs ParseArgs(int argc, char** argv) {
  CmdArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string token = argv[i];
    if (token == "--image" && i + 1 < argc) {
      args.image_path = argv[++i];
    } else if ((token == "--test-image" || token == "--holdout-image") &&
               i + 1 < argc) {
      args.test_image_path = argv[++i];
    } else if (token == "--stage5-precomputed-observations-dir" &&
               i + 1 < argc) {
      args.precomputed_observations_dir = argv[++i];
    } else if (token ==
                   "--stage5-precomputed-holdout-observations-dir" &&
               i + 1 < argc) {
      args.precomputed_holdout_observations_dir = argv[++i];
    } else if (
        token == "--stage5-allow-image-training-with-precomputed-holdout") {
      args.allow_image_training_with_precomputed_holdout = true;
    } else if (token ==
                   "--stage5-init-auxiliary-precomputed-observations-dir" &&
               i + 1 < argc) {
      args.precomputed_initialization_auxiliary_observation_dirs.push_back(
          argv[++i]);
    } else if (token == "--stage5-precomputed-target-mode" &&
               i + 1 < argc) {
      args.precomputed_target_mode = argv[++i];
    } else if (token == "--stage5-precomputed-init-use-all-points" &&
               i + 1 < argc) {
      args.precomputed_init_use_all_points =
          ParseBooleanFlagValue(argv[++i]);
    } else if (token == "--stage5-precomputed-init-point-scope" &&
               i + 1 < argc) {
      args.precomputed_initialization_point_scope = argv[++i];
    } else if (token == "--stage5-external-holdout-self-frontend-prepass") {
      args.external_holdout_self_frontend_prepass = true;
    } else if (token == "--stage5-frontend-only") {
      args.stage5_frontend_only = true;
    } else if (token ==
               "--stage5-holdout-evaluate-full-training-observations") {
      args.stage5_holdout_evaluate_full_training_observations = true;
    } else if (token == "--config" && i + 1 < argc) {
      args.config_path = argv[++i];
    } else if (token == "--output" && i + 1 < argc) {
      args.output_path = argv[++i];
    } else if (token == "--kalibr-camchain" && i + 1 < argc) {
      args.kalibr_camchain_yaml = argv[++i];
    } else if ((token == "--reference-intrinsics-yaml" ||
                token == "--reference-camera-yaml" ||
                token == "--extra-reference-intrinsics-yaml") &&
               i + 1 < argc) {
      args.reference_intrinsics_specs.push_back(argv[++i]);
    } else if (token == "--models" && i + 1 < argc) {
      args.stage5_models = NormalizeStage5ModelFamily(argv[++i]);
    } else if (token == "--stage5-init-refine-mode" && i + 1 < argc) {
      args.stage5_init_refine_mode = argv[++i];
    } else if (token == "--stage5-init-selection-scorer" && i + 1 < argc) {
      args.stage5_init_selection_scorer = argv[++i];
    } else if (token == "--stage5-enable-init-principal-profile") {
      args.stage5_enable_init_principal_profile = true;
    } else if (token == "--stage5-init-principal-profile-radius-px" &&
               i + 1 < argc) {
      args.stage5_init_principal_profile_radius_px = std::stod(argv[++i]);
    } else if (token == "--stage5-enable-init-fixed-layout-diagnostic") {
      args.stage5_enable_init_fixed_layout_diagnostic = true;
    } else if (token == "--stage5-enable-init-board-jackknife-diagnostic") {
      args.stage5_enable_init_board_jackknife_diagnostic = true;
    } else if (token == "--stage5-enable-init-coverage-weighted-diagnostic") {
      args.stage5_enable_init_coverage_weighted_diagnostic = true;
    } else if (token == "--stage5-init-near-tie-prefer-lower-focal") {
      args.stage5_init_near_tie_prefer_lower_focal = true;
    } else if (token ==
                   "--stage5-init-near-tie-relative-objective-tolerance" &&
               i + 1 < argc) {
      args.stage5_init_near_tie_relative_objective_tolerance =
          std::stod(argv[++i]);
    } else if (token == "--stage5-enable-camera-aware-outer-rescue") {
      args.stage5_camera_aware_outer_rescue = true;
    } else if (token == "--stage5-disable-camera-aware-outer-rescue") {
      args.stage5_camera_aware_outer_rescue = false;
    } else if (token == "--stage5-enable-frozen-recovery-baseline") {
      args.stage5_enable_frozen_recovery_baseline = true;
    } else if (token == "--kalibr-training-split-signature" && i + 1 < argc) {
      args.kalibr_training_split_signature = argv[++i];
    } else if (token == "--camera-init-mode" && i + 1 < argc) {
      args.camera_init_mode_override = argv[++i];
    } else if (token == "--experiment-tag" && i + 1 < argc) {
      args.experiment_tag = argv[++i];
    } else if (token == "--runtime-mode" && i + 1 < argc) {
      args.runtime_mode = ati::ParseStage5RuntimeMode(argv[++i]);
    } else if (token == "--cache-dir" && i + 1 < argc) {
      args.cache_dir = argv[++i];
    } else if (token == "--include-internal-points" && i + 1 < argc) {
      args.include_internal_points = std::stoi(argv[++i]) != 0;
    } else if (token == "--disable-second-pass") {
      args.disable_second_pass = true;
    } else if (token == "--intrinsics-release-mode" && i + 1 < argc) {
      args.intrinsics_release_mode = ParseIntrinsicsReleaseMode(argv[++i]);
    } else if (token == "--disable-residual-sanity-gate") {
      args.disable_residual_sanity_gate = true;
    } else if (token == "--enable-board-pose-fit-gate") {
      args.enable_board_pose_fit_gate = true;
    } else if (token == "--disable-board-pose-fit-gate") {
      args.enable_board_pose_fit_gate = false;
    } else if (token == "--stage5-selection-mode" && i + 1 < argc) {
      args.stage5_selection_mode = argv[++i];
    } else if (token == "--stage5-selection-residual-sanity-factor" &&
               i + 1 < argc) {
      args.stage5_selection_residual_sanity_factor =
          std::stod(argv[++i]);
    } else if (token == "--stage5-selection-max-board-observation-rmse" &&
               i + 1 < argc) {
      args.stage5_selection_max_board_observation_rmse =
          std::stod(argv[++i]);
    } else if (token == "--stage5-selection-kalibr-style-outlier-sigma" &&
               i + 1 < argc) {
      args.stage5_selection_kalibr_style_outlier_sigma =
          std::stod(argv[++i]);
    } else if (
        token == "--stage5-selection-kalibr-style-min-abs-threshold-px" &&
        i + 1 < argc) {
      args.stage5_selection_kalibr_style_min_abs_threshold_px =
          std::stod(argv[++i]);
    } else if (
        token == "--stage5-selection-kalibr-style-min-views-before-filter" &&
        i + 1 < argc) {
      args.stage5_selection_kalibr_style_min_views_before_filter =
          std::stoi(argv[++i]);
    } else if (token ==
               "--stage5-enable-trial-backend-frame-board-selection") {
      args.stage5_enable_trial_backend_frame_board_selection = true;
    } else if (token == "--stage5-trial-backend-selection-mode" &&
               i + 1 < argc) {
      args.stage5_trial_backend_selection_mode = argv[++i];
    } else if (token == "--stage5-trial-backend-selection-budget-mode" &&
               i + 1 < argc) {
      args.stage5_trial_backend_selection_budget_mode = argv[++i];
      args.stage5_trial_backend_selection_budget_mode_set = true;
    } else if (
        token == "--stage5-trial-backend-selection-candidate-order" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_candidate_order = argv[++i];
      args.stage5_trial_backend_selection_candidate_order_set = true;
    } else if (
        token ==
            "--stage5-trial-backend-selection-info-gain-proxy-mode" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_info_gain_proxy_mode = argv[++i];
    } else if (
        token ==
            "--stage5-trial-backend-selection-batch-granularity" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_batch_granularity = argv[++i];
    } else if (
        token ==
            "--stage5-trial-backend-selection-acceptance-policy" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_acceptance_policy = argv[++i];
    } else if (
        token == "--stage5-trial-backend-selection-mi-tol" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_mi_tol = std::stod(argv[++i]);
      args.stage5_trial_backend_selection_mi_tol_set = true;
    } else if (
        token == "--stage5-trial-backend-selection-rank-threshold" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_rank_threshold =
          std::stod(argv[++i]);
    } else if (token == "--stage5-checkerboard-huber-delta-px" &&
               i + 1 < argc) {
      args.stage5_checkerboard_huber_delta_pixels = std::stod(argv[++i]);
    } else if (token == "--stage5-checkerboard-outlier-filter" &&
               i + 1 < argc) {
      args.stage5_checkerboard_outlier_filter_enabled =
          ParseBooleanFlagValue(argv[++i]);
    } else if (token == "--stage5-checkerboard-outlier-sigma" &&
               i + 1 < argc) {
      args.stage5_checkerboard_outlier_sigma = std::stod(argv[++i]);
    } else if (token == "--stage5-checkerboard-min-inlier-ratio" &&
               i + 1 < argc) {
      args.stage5_checkerboard_min_inlier_ratio = std::stod(argv[++i]);
    } else if (token == "--stage5-checkerboard-min-retained-points" &&
               i + 1 < argc) {
      args.stage5_checkerboard_min_retained_points = std::stoi(argv[++i]);
    } else if (
        token == "--stage5-trial-backend-selection-candidate-shuffle-seed" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_candidate_shuffle_seed =
          static_cast<unsigned int>(std::stoul(argv[++i]));
      args.stage5_trial_backend_selection_candidate_shuffle_seed_set = true;
    } else if (token ==
                   "--stage5-trial-backend-selection-incremental" &&
               i + 1 < argc) {
      args.stage5_trial_backend_selection_incremental =
          std::stoi(argv[++i]) != 0;
    } else if (
        token ==
            "--stage5-trial-backend-selection-carry-accepted-trial-state" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_carry_accepted_trial_state =
          std::stoi(argv[++i]) != 0;
    } else if (
        token ==
            "--stage5-trial-backend-selection-optimize-intrinsics" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_optimize_intrinsics =
          std::stoi(argv[++i]) != 0;
    } else if (
        token ==
            "--stage5-trial-backend-selection-delayed-intrinsics-release" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_delayed_intrinsics_release =
          std::stoi(argv[++i]) != 0;
    } else if (
        token ==
            "--stage5-trial-backend-selection-intrinsics-release-iteration" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_intrinsics_release_iteration =
          std::stoi(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-persistent-intrinsics-anchor-prior" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_persistent_intrinsics_anchor_prior =
          std::stoi(argv[++i]) != 0;
    } else if (
        token ==
            "--stage5-trial-backend-selection-persistent-fix-board-layout" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_persistent_fix_board_layout =
          std::stoi(argv[++i]) != 0;
    } else if (
        token ==
            "--stage5-trial-backend-selection-persistent-intrinsics-anchor-weight-xi-alpha" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_xi_alpha =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-persistent-intrinsics-anchor-weight-focal" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_focal =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-persistent-intrinsics-anchor-weight-principal" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_principal =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-persistent-max-focal-relative-step" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_persistent_max_focal_relative_step =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-persistent-max-principal-step-px" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_persistent_max_principal_step_px =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-persistent-max-xi-alpha-step" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_persistent_max_xi_alpha_step =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage5-trial-backend-selection-max-iterations" &&
               i + 1 < argc) {
      args.stage5_trial_backend_selection_max_iterations =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-trial-backend-selection-max-candidate-additions" &&
               i + 1 < argc) {
      args.stage5_trial_backend_selection_max_candidate_additions =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-trial-backend-selection-adaptive-budget-ratio" &&
               i + 1 < argc) {
      args.stage5_trial_backend_selection_adaptive_budget_ratio =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage5-trial-backend-selection-adaptive-budget-min" &&
               i + 1 < argc) {
      args.stage5_trial_backend_selection_adaptive_budget_min =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-trial-backend-selection-adaptive-budget-max" &&
               i + 1 < argc) {
      args.stage5_trial_backend_selection_adaptive_budget_max =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-trial-backend-selection-runtime-safety-ceiling" &&
               i + 1 < argc) {
      args.stage5_trial_backend_selection_runtime_safety_ceiling =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-trial-backend-selection-outlier-sigma" &&
               i + 1 < argc) {
      args.stage5_trial_backend_selection_outlier_sigma =
          std::stod(argv[++i]);
    } else if (
        token == "--stage5-trial-backend-selection-min-abs-threshold-px" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_min_abs_threshold_px =
          std::stod(argv[++i]);
    } else if (
        token == "--stage5-trial-backend-selection-max-threshold-px" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_max_threshold_px =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-accept-max-global-rmse-increase-px" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_accept_max_global_rmse_increase_px =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-accept-max-outer-rmse-increase-px" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_accept_max_outer_rmse_increase_px =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-accept-max-internal-rmse-increase-px" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_accept_max_internal_rmse_increase_px =
          std::stod(argv[++i]);
    } else if (
        token == "--stage5-trial-backend-selection-min-candidate-score" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_min_candidate_score =
          std::stod(argv[++i]);
    } else if (
        token == "--stage5-trial-backend-selection-min-coverage-gain" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_min_coverage_gain =
          std::stod(argv[++i]);
    } else if (
        token == "--stage5-trial-backend-selection-use-consistency-score") {
      args.stage5_trial_backend_selection_use_consistency_score = true;
    } else if (
        token ==
            "--stage5-trial-backend-selection-consistency-translation-sigma-mm" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_consistency_translation_sigma_mm =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-consistency-rotation-sigma-deg" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_consistency_rotation_sigma_deg =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-consistency-penalty-weight" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_consistency_penalty_weight =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-consistency-max-translation-error-mm" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_consistency_max_translation_error_mm =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-consistency-max-rotation-error-deg" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_consistency_max_rotation_error_deg =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-consistency-max-local-outer-rmse-px" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_consistency_max_local_outer_rmse_px =
          std::stod(argv[++i]);
    } else if (
        token == "--stage5-trial-backend-selection-max-accepted-per-board" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_max_accepted_per_board =
          std::stoi(argv[++i]);
    } else if (
        token == "--stage5-trial-backend-selection-max-accepted-per-frame" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_max_accepted_per_frame =
          std::stoi(argv[++i]);
    } else if (
        token == "--stage5-trial-backend-selection-frame-cohesion") {
      args.stage5_trial_backend_selection_frame_cohesion = true;
    } else if (
        token == "--stage5-trial-backend-selection-frame-cohesion-max-companions" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_frame_cohesion_max_companions =
          std::stoi(argv[++i]);
    } else if (
        token == "--stage5-trial-backend-selection-frame-cohesion-min-candidate-score" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_frame_cohesion_min_candidate_score =
          std::stod(argv[++i]);
    } else if (
        token == "--stage5-trial-backend-selection-min-keep-per-board" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_min_keep_per_board =
          std::stoi(argv[++i]);
    } else if (
        token ==
            "--stage5-trial-backend-selection-force-include-frame-board-list" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_force_include_frame_board_list =
          argv[++i];
    } else if (
        token ==
            "--stage5-trial-backend-selection-seed-frame-board-list" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_seed_frame_board_list = argv[++i];
    } else if (
        token ==
            "--stage5-trial-backend-selection-force-include-list-is-exact-input" &&
        i + 1 < argc) {
      args.stage5_trial_backend_selection_force_include_list_is_exact_input =
          std::stoi(argv[++i]) != 0;
    } else if (token == "--strict-board-observation-acceptance") {
      args.strict_board_observation_acceptance = true;
    } else if (token == "--stage5-preserve-frame-board-cohesion") {
      args.preserve_frame_board_cohesion = true;
    } else if (token == "--backend-polar-angle-weight-mode" &&
               i + 1 < argc) {
      args.backend_polar_angle_weight_mode = argv[++i];
    } else if (token == "--backend-polar-angle-weight-bin-edges" &&
               i + 1 < argc) {
      args.backend_polar_angle_weight_bin_edges = argv[++i];
    } else if (token == "--backend-polar-angle-weight-fixed-bin-scales" &&
               i + 1 < argc) {
      args.backend_polar_angle_weight_fixed_bin_scales = argv[++i];
    } else if (
        token == "--backend-polar-angle-weight-adaptive-sigma-reference-deg" &&
        i + 1 < argc) {
      args.backend_polar_angle_weight_adaptive_sigma_reference_deg =
          std::stod(argv[++i]);
    } else if (
        token == "--backend-polar-angle-weight-adaptive-sigma-growth" &&
        i + 1 < argc) {
      args.backend_polar_angle_weight_adaptive_sigma_growth =
          std::stod(argv[++i]);
    } else if (token == "--backend-polar-angle-weight-min-scale" &&
               i + 1 < argc) {
      args.backend_polar_angle_weight_min_scale = std::stod(argv[++i]);
    } else if (token == "--backend-residual-model" && i + 1 < argc) {
      args.backend_residual_model = argv[++i];
    } else if (token == "--backend-hybrid-angular-threshold-deg" &&
               i + 1 < argc) {
      args.backend_hybrid_angular_threshold_deg = std::stod(argv[++i]);
    } else if (token == "--backend-use-point-type-residual-split") {
      args.backend_use_point_type_residual_split = true;
    } else if (token == "--backend-outer-residual-model" &&
               i + 1 < argc) {
      args.backend_outer_residual_model = argv[++i];
    } else if (token == "--backend-internal-residual-model" &&
               i + 1 < argc) {
      args.backend_internal_residual_model = argv[++i];
    } else if (token == "--backend-enable-angular-auxiliary-residual") {
      args.backend_enable_angular_auxiliary_residual = true;
    } else if (token == "--backend-angular-auxiliary-weight" &&
               i + 1 < argc) {
      args.backend_angular_auxiliary_weight = std::stod(argv[++i]);
    } else if (token == "--backend-angular-auxiliary-normalized" &&
               i + 1 < argc) {
      args.backend_angular_auxiliary_normalized = std::stoi(argv[++i]) != 0;
    } else if (token == "--backend-angular-auxiliary-apply-to-outer" &&
               i + 1 < argc) {
      args.backend_angular_auxiliary_apply_to_outer = std::stoi(argv[++i]) != 0;
    } else if (token == "--backend-angular-auxiliary-apply-to-internal" &&
               i + 1 < argc) {
      args.backend_angular_auxiliary_apply_to_internal =
          std::stoi(argv[++i]) != 0;
    } else if (token == "--backend-polar-continuous-hybrid-threshold-deg" &&
               i + 1 < argc) {
      args.backend_polar_continuous_hybrid_threshold_deg =
          std::stod(argv[++i]);
    } else if (token == "--backend-polar-continuous-hybrid-temperature-deg" &&
               i + 1 < argc) {
      args.backend_polar_continuous_hybrid_temperature_deg =
          std::stod(argv[++i]);
    } else if (token == "--backend-normalized-angular-reference-sigma-px" &&
               i + 1 < argc) {
      args.backend_normalized_angular_reference_sigma_px =
          std::stod(argv[++i]);
    } else if (token == "--backend-normalized-angular-min-sigma-rad" &&
               i + 1 < argc) {
      args.backend_normalized_angular_min_sigma_rad =
          std::stod(argv[++i]);
    } else if (token == "--backend-normalized-angular-max-weight-scale" &&
               i + 1 < argc) {
      args.backend_normalized_angular_max_weight_scale =
          std::stod(argv[++i]);
    } else if (token == "--backend-pixel-residual-weight" &&
               i + 1 < argc) {
      args.backend_pixel_residual_weight = std::stod(argv[++i]);
    } else if (token == "--backend-chordal-residual-weight" &&
               i + 1 < argc) {
      args.backend_chordal_residual_weight = std::stod(argv[++i]);
    } else if (token ==
               "--stage5-enable-hybrid-pixel-ray-final-refinement") {
      args.enable_hybrid_pixel_ray_final_refinement = true;
    } else if (token == "--stage5-large-intrinsic-perturbation" &&
               i + 1 < argc) {
      args.enable_large_intrinsic_perturbation = true;
      args.large_intrinsic_perturbation_profile = argv[++i];
    } else if (token == "--stage5-large-intrinsic-perturbation-scale" &&
               i + 1 < argc) {
      args.large_intrinsic_perturbation_scale = std::stod(argv[++i]);
    } else if (token == "--stage5-large-intrinsic-perturbation-strict-scale") {
      args.large_intrinsic_perturbation_strict_scale = true;
    } else if (token ==
                   "--stage5-large-intrinsic-perturbation-reference-scene" &&
               i + 1 < argc) {
      args.large_intrinsic_perturbation_reference_scene_path = argv[++i];
    } else if (token ==
               "--stage5-large-intrinsic-perturbation-outer-only-after-application") {
      args.large_intrinsic_perturbation_outer_only_after_application = true;
    } else if (token == "--stage5-hybrid-pixel-ray-lambda" &&
               i + 1 < argc) {
      args.hybrid_pixel_ray_lambda = std::stod(argv[++i]);
    } else if (token == "--stage5-hybrid-pixel-ray-polar-adaptive") {
      args.hybrid_pixel_ray_polar_adaptive = true;
    } else if (token == "--stage5-hybrid-pixel-ray-lambda-min" &&
               i + 1 < argc) {
      args.hybrid_pixel_ray_lambda_min = std::stod(argv[++i]);
    } else if (token == "--stage5-hybrid-pixel-ray-lambda-max" &&
               i + 1 < argc) {
      args.hybrid_pixel_ray_lambda_max = std::stod(argv[++i]);
    } else if (token == "--stage5-hybrid-pixel-ray-transition-start-deg" &&
               i + 1 < argc) {
      args.hybrid_pixel_ray_transition_start_deg = std::stod(argv[++i]);
    } else if (token == "--stage5-hybrid-pixel-ray-transition-end-deg" &&
               i + 1 < argc) {
      args.hybrid_pixel_ray_transition_end_deg = std::stod(argv[++i]);
    } else if (token == "--stage5-hybrid-pixel-ray-max-iterations" &&
               i + 1 < argc) {
      args.hybrid_pixel_ray_max_iterations = std::stoi(argv[++i]);
    } else if (token == "--stage5-hybrid-pixel-ray-pixel-scale-floor" &&
               i + 1 < argc) {
      args.hybrid_pixel_ray_pixel_scale_floor = std::stod(argv[++i]);
    } else if (token == "--stage5-hybrid-pixel-ray-ray-scale-floor" &&
               i + 1 < argc) {
      args.hybrid_pixel_ray_ray_scale_floor = std::stod(argv[++i]);
    } else if (token == "--backend-angular-use-normalize-jacobian" &&
               i + 1 < argc) {
      args.backend_angular_use_normalize_jacobian =
          ParseBooleanFlagValue(argv[++i]);
	} else if (token == "--backend-angular-local-whitening") {
	  args.backend_angular_local_whitening = true;
	} else if (token == "--backend-angular-local-whitening-pixel-sigma-px" &&
	           i + 1 < argc) {
	  args.backend_angular_local_whitening_pixel_sigma_px =
	      std::stod(argv[++i]);
	} else if (token ==
	               "--backend-angular-local-whitening-covariance-damping" &&
	           i + 1 < argc) {
	  args.backend_angular_local_whitening_covariance_damping =
	      std::stod(argv[++i]);
	} else if (token ==
	               "--backend-angular-local-whitening-min-sigma-rad" &&
	           i + 1 < argc) {
	  args.backend_angular_local_whitening_min_sigma_rad =
	      std::stod(argv[++i]);
	} else if (token == "--backend-angular-local-whitening-max-weight" &&
	           i + 1 < argc) {
	  args.backend_angular_local_whitening_max_weight =
	      std::stod(argv[++i]);
    } else if (token == "--backend-angular-observed-ray-mode" &&
               i + 1 < argc) {
      args.backend_angular_observed_ray_mode = argv[++i];
    } else if ((token == "--backend-board-pose-parameterization" ||
                token == "--stage5-backend-board-pose-parameterization") &&
               i + 1 < argc) {
      args.backend_board_pose_parameterization = argv[++i];
	    } else if ((token == "--stage5-backend-optimize-board-poses" ||
	                token == "--backend-optimize-board-poses") &&
	               i + 1 < argc) {
	      args.backend_optimize_board_poses = std::stoi(argv[++i]) != 0;
	    } else if (token == "--stage5-backend-board-pose-prior") {
	      args.backend_board_pose_prior = true;
	    } else if (token ==
	                   "--stage5-backend-board-pose-prior-translation-sigma-mm" &&
	               i + 1 < argc) {
	      args.backend_board_pose_prior_translation_sigma_mm =
	          std::stod(argv[++i]);
	    } else if (token ==
	                   "--stage5-backend-board-pose-prior-rotation-sigma-deg" &&
	               i + 1 < argc) {
	      args.backend_board_pose_prior_rotation_sigma_deg =
	          std::stod(argv[++i]);
	    } else if (token == "--stage5-backend-point-budget-control") {
	      args.backend_point_budget_control = true;
	    } else if (token == "--stage5-backend-point-budget-control-total-points" &&
	               i + 1 < argc) {
	      args.backend_point_budget_control_total_points =
	          std::stoi(argv[++i]);
	    } else if (token == "--stage5-backend-point-budget-control-seed" &&
	               i + 1 < argc) {
	      args.backend_point_budget_control_seed =
	          static_cast<unsigned int>(std::stoul(argv[++i]));
	    } else if (token == "--stage5-backend-max-boards-per-frame-for-ablation" &&
	               i + 1 < argc) {
	      args.backend_max_boards_per_frame_for_ablation =
	          std::stoi(argv[++i]);
	    } else if (token == "--backend-fixed-intrinsics") {
	      args.backend_fixed_intrinsics = true;
    } else if (token == "--stage5-enable-angular-residual-diagnostics") {
      args.enable_angular_residual_diagnostics = true;
    } else if (token == "--stage5-angular-residual-bin-edges" &&
               i + 1 < argc) {
      args.angular_residual_bin_edges = argv[++i];
    } else if (token == "--backend-multi-board-consistency-weighting") {
      args.backend_multi_board_consistency_weighting = true;
    } else if (token == "--backend-consistency-pose-source" &&
               i + 1 < argc) {
      args.backend_consistency_pose_source = argv[++i];
    } else if (token == "--backend-consistency-weight-mode" &&
               i + 1 < argc) {
      args.backend_consistency_weight_mode = argv[++i];
    } else if (token == "--backend-consistency-translation-sigma-mm" &&
               i + 1 < argc) {
      args.backend_consistency_translation_sigma_mm = std::stod(argv[++i]);
    } else if (token == "--backend-consistency-rotation-sigma-deg" &&
               i + 1 < argc) {
      args.backend_consistency_rotation_sigma_deg = std::stod(argv[++i]);
    } else if (token == "--backend-consistency-min-weight" &&
               i + 1 < argc) {
      args.backend_consistency_min_weight = std::stod(argv[++i]);
    } else if (token == "--backend-consistency-apply-to-outer" &&
               i + 1 < argc) {
      args.backend_consistency_apply_to_outer = std::stoi(argv[++i]) != 0;
    } else if (token == "--backend-consistency-apply-to-internal" &&
               i + 1 < argc) {
      args.backend_consistency_apply_to_internal = std::stoi(argv[++i]) != 0;
    } else if (token == "--backend-consistency-hard-reject-enabled" &&
               i + 1 < argc) {
      args.backend_consistency_hard_reject_enabled = std::stoi(argv[++i]) != 0;
    } else if (token == "--backend-consistency-hard-reject-translation-mm" &&
               i + 1 < argc) {
      args.backend_consistency_hard_reject_translation_mm = std::stod(argv[++i]);
    } else if (token == "--backend-consistency-hard-reject-rotation-deg" &&
               i + 1 < argc) {
      args.backend_consistency_hard_reject_rotation_deg = std::stod(argv[++i]);
    } else if (token == "--backend-consistency-hard-reject-residual-px" &&
               i + 1 < argc) {
      args.backend_consistency_hard_reject_residual_px = std::stod(argv[++i]);
    } else if (token == "--backend-consistency-dump-weight-summary" &&
               i + 1 < argc) {
      args.backend_consistency_dump_weight_summary = std::stoi(argv[++i]) != 0;
    } else if (token == "--internal-regeneration-diagnostics") {
      args.internal_regeneration_diagnostics = true;
    } else if (token == "--stage5-export-internal-seed-step-overlays") {
      args.stage5_export_internal_seed_step_overlays = true;
      args.internal_regeneration_diagnostics = true;
    } else if (token == "--internal-blur-diagnostics") {
      args.internal_blur_diagnostics = true;
    } else if (token == "--internal-blur-filter-mode" && i + 1 < argc) {
      args.internal_blur_filter_mode =
          ati::ParseInternalBlurFilterMode(argv[++i]);
    } else if (token == "--internal-blur-filter-low-patch-gradient-quantile" &&
               i + 1 < argc) {
      args.internal_blur_filter_low_patch_gradient_quantile =
          std::stod(argv[++i]);
    } else if (token == "--internal-blur-filter-min-board-rmse-px" &&
               i + 1 < argc) {
      args.internal_blur_filter_min_board_rmse_px = std::stod(argv[++i]);
    } else if (token == "--internal-blur-filter-min-board-p95-px" &&
               i + 1 < argc) {
      args.internal_blur_filter_min_board_p95_px = std::stod(argv[++i]);
    } else if (token == "--internal-pose-rescue-mode" && i + 1 < argc) {
      args.internal_pose_rescue_mode =
          ati::ParseInternalPoseRescueMode(argv[++i]);
    } else if (token == "--internal-pose-rescue-max-ray-angle-deg" && i + 1 < argc) {
      args.internal_pose_rescue_max_ray_angle_deg = std::stod(argv[++i]);
    } else if (token == "--internal-pose-rescue-accept-max-outer-rmse" &&
               i + 1 < argc) {
      args.internal_pose_rescue_accept_max_outer_rmse = std::stod(argv[++i]);
    } else if (token == "--stage5-ignore-image-evidence-min-quality") {
      args.ignore_image_evidence_min_quality = true;
    } else if (token == "--stage5-force-internal-seed-from-prediction") {
      args.force_internal_seed_from_prediction = true;
    } else if (token == "--stage5-bypass-internal-seed-filters") {
      args.bypass_internal_seed_filters = true;
    } else if (token == "--stage5-internal-corner-filter-mode" &&
               i + 1 < argc) {
      args.internal_corner_filter_mode = argv[++i];
    } else if (token == "--stage5-internal-corner-filter-max-reproj-error" &&
               i + 1 < argc) {
      args.internal_corner_filter_max_reproj_error = std::stod(argv[++i]);
    } else if (token == "--stage5-internal-corner-filter-quality-min" &&
               i + 1 < argc) {
      args.internal_corner_filter_quality_min = std::stod(argv[++i]);
    } else if (token == "--stage5-internal-corner-filter-quality-relaxation-px" &&
               i + 1 < argc) {
      args.internal_corner_filter_quality_relaxation_px = std::stod(argv[++i]);
    } else if (token == "--stage5-internal-corner-filter-adaptive-min-threshold-px" &&
               i + 1 < argc) {
      args.internal_corner_filter_adaptive_min_threshold_px = std::stod(argv[++i]);
    } else if (token == "--stage5-enable-geometry-prior-outer-seed") {
      args.enable_geometry_prior_outer_seed = true;
    } else if (token == "--stage5-disable-geometry-prior-outer-seed") {
      args.enable_geometry_prior_outer_seed = false;
    } else if (token == "--stage5-geometry-prior-rescue-diagnostic-only" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_diagnostic_only = std::stoi(argv[++i]) != 0;
    } else if (
        token == "--stage5-geometry-prior-rescue-use-as-observation") {
      args.geometry_prior_rescue_use_as_observation = true;
    } else if (
        token == "--stage5-geometry-prior-rescue-disable-use-as-observation") {
      args.geometry_prior_rescue_use_as_observation = false;
    } else if (
        token == "--stage5-geometry-prior-rescue-keep-outer-on-internal-failure") {
      args.geometry_prior_rescue_keep_outer_on_internal_failure = true;
    } else if (
        token == "--stage5-geometry-prior-rescue-allow-geometry-only-pose-refit") {
      args.geometry_prior_rescue_allow_geometry_only_pose_refit = true;
    } else if (
        token == "--stage5-geometry-prior-rescue-disable-geometry-only-pose-refit") {
      args.geometry_prior_rescue_allow_geometry_only_pose_refit = false;
    } else if (token == "--stage5-geometry-prior-rescue-subpix-window-radius" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_subpix_window_radius = std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-prior-rescue-max-corner-displacement-px" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_max_corner_displacement_px =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-prior-rescue-min-corner-response-ratio" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_min_corner_response_ratio =
          std::stod(argv[++i]);
    } else if (
        token == "--stage5-geometry-prior-rescue-enable-spherical-refine") {
      args.geometry_prior_rescue_enable_spherical_refine = true;
    } else if (
        token == "--stage5-geometry-prior-rescue-disable-spherical-refine") {
      args.geometry_prior_rescue_enable_spherical_refine = false;
    } else if (token ==
                   "--stage5-geometry-prior-rescue-edge-sample-count" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_edge_sample_count = std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-prior-rescue-edge-search-half-width-px" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_edge_search_half_width_px =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-prior-rescue-min-edge-support-ratio" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_min_edge_support_ratio =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-prior-rescue-min-edge-gradient-ratio" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_min_edge_gradient_ratio =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-prior-rescue-accept-max-outer-rmse" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_accept_max_outer_rmse = std::stod(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-prior-rescue-accept-max-rotation-error-deg" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_accept_max_rotation_error_deg =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-prior-rescue-accept-max-translation-error" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_accept_max_translation_error =
          std::stod(argv[++i]);
    } else if (token == "--stage5-enable-geometry-guided-tag-likelihood") {
      args.geometry_guided_tag_likelihood_enabled = true;
    } else if (token == "--stage5-disable-geometry-guided-tag-likelihood") {
      args.geometry_guided_tag_likelihood_enabled = false;
    } else if (token ==
                   "--stage5-geometry-guided-tag-likelihood-min-visible-boards" &&
               i + 1 < argc) {
      args.geometry_guided_tag_likelihood_min_visible_boards =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-guided-tag-likelihood-max-expected-hamming" &&
               i + 1 < argc) {
      args.geometry_guided_tag_likelihood_max_expected_hamming =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-guided-tag-likelihood-min-hamming-margin" &&
               i + 1 < argc) {
      args.geometry_guided_tag_likelihood_min_hamming_margin =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-guided-tag-likelihood-min-contrast" &&
               i + 1 < argc) {
      args.geometry_guided_tag_likelihood_min_contrast =
          std::stod(argv[++i]);
    } else if (token ==
               "--stage5-enable-camera-aware-sphere-patch-zero-detection") {
      args.stage5_camera_aware_outer_rescue_zero_detection_frames = true;
      args.stage5_camera_aware_outer_rescue_zero_detection_frames_set = true;
    } else if (token ==
               "--stage5-disable-camera-aware-sphere-patch-zero-detection") {
      args.stage5_camera_aware_outer_rescue_zero_detection_frames = false;
      args.stage5_camera_aware_outer_rescue_zero_detection_frames_set = true;
    } else if (token ==
                   "--stage5-enable-geometry-guided-tag-likelihood-single-anchor") {
      args.geometry_guided_tag_likelihood_allow_single_anchor = true;
    } else if (token ==
                   "--stage5-disable-geometry-guided-tag-likelihood-single-anchor") {
      args.geometry_guided_tag_likelihood_allow_single_anchor = false;
    } else if (token ==
                   "--stage5-geometry-guided-tag-likelihood-single-anchor-max-outer-rmse" &&
               i + 1 < argc) {
      args.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-guided-tag-likelihood-single-anchor-max-expected-hamming" &&
               i + 1 < argc) {
      args.geometry_guided_tag_likelihood_single_anchor_max_expected_hamming =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-guided-tag-likelihood-single-anchor-min-hamming-margin" &&
               i + 1 < argc) {
      args.geometry_guided_tag_likelihood_single_anchor_min_hamming_margin =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage5-geometry-guided-tag-likelihood-single-anchor-min-contrast" &&
               i + 1 < argc) {
      args.geometry_guided_tag_likelihood_single_anchor_min_contrast =
          std::stod(argv[++i]);
    } else if (token ==
               "--stage5-enable-outer-only-intermediate-calibration") {
      args.enable_outer_only_intermediate_calibration = true;
    } else if (token == "--stage5-intermediate-diagnostic-only" &&
               i + 1 < argc) {
      args.intermediate_diagnostic_only = std::stoi(argv[++i]) != 0;
    } else if (token ==
                   "--stage5-use-intermediate-for-round1-internal-regeneration" &&
               i + 1 < argc) {
      args.use_intermediate_for_round1_internal_regeneration =
          std::stoi(argv[++i]) != 0;
    } else if (
        token == "--stage5-use-intermediate-for-full-frontend-regeneration") {
      args.use_intermediate_for_full_frontend_regeneration = true;
      args.enable_outer_only_intermediate_calibration = true;
      args.intermediate_diagnostic_only = false;
      args.use_intermediate_for_round1_internal_regeneration = true;
      args.internal_regeneration_diagnostics = true;
      args.enable_geometry_prior_outer_seed = true;
    } else if (token == "--stage5-intermediate-optimize-intrinsics" &&
               i + 1 < argc) {
      args.intermediate_optimize_intrinsics = std::stoi(argv[++i]) != 0;
    } else if (token == "--stage5-intermediate-optimize-board-poses" &&
               i + 1 < argc) {
      args.intermediate_optimize_board_poses = std::stoi(argv[++i]) != 0;
    } else if (token == "--stage5-intermediate-optimize-frame-poses" &&
               i + 1 < argc) {
      args.intermediate_optimize_frame_poses = std::stoi(argv[++i]) != 0;
    } else if (
        token == "--stage5-intermediate-intrinsics-release-iteration" &&
        i + 1 < argc) {
      args.intermediate_intrinsics_release_iteration = std::stoi(argv[++i]);
    } else if (token == "--stage5-intermediate-max-outer-rmse-px" &&
               i + 1 < argc) {
      args.intermediate_max_outer_rmse_px = std::stod(argv[++i]);
    } else if (token == "--stage5-intermediate-min-visible-boards" &&
               i + 1 < argc) {
      args.intermediate_min_visible_boards = std::stoi(argv[++i]);
    } else if (token == "--pre-backend-filter-mode" && i + 1 < argc) {
      args.pre_backend_filter_mode =
          ati::ParsePreBackendFilterMode(argv[++i]);
    } else if (token == "--pre-backend-filter-threshold-mode" &&
               i + 1 < argc) {
      args.pre_backend_filter_threshold_mode =
          ati::ParsePreBackendFilterThresholdMode(argv[++i]);
    } else if (token == "--pre-backend-filter-sigma" && i + 1 < argc) {
      args.pre_backend_filter_sigma = std::stod(argv[++i]);
    } else if (token == "--pre-backend-filter-min-abs-threshold-px" &&
               i + 1 < argc) {
      args.pre_backend_filter_min_abs_threshold_px = std::stod(argv[++i]);
    } else if (token == "--internal-joint-refine-mode" && i + 1 < argc) {
      args.internal_joint_refine_mode =
          ati::ParseInternalJointRefineMode(argv[++i]);
    } else if (token == "--internal-joint-refine-target-mode" &&
               i + 1 < argc) {
      args.internal_joint_refine_target_mode =
          ati::ParseInternalJointRefineTargetMode(argv[++i]);
    } else if (token == "--internal-joint-refine-search-radius-px" &&
               i + 1 < argc) {
      args.internal_joint_refine_search_radius_px = std::stod(argv[++i]);
    } else if (token == "--internal-joint-refine-max-displacement-px" &&
               i + 1 < argc) {
      args.internal_joint_refine_max_displacement_px = std::stod(argv[++i]);
    } else if (token == "--internal-joint-refine-geometry-sigma-px" &&
               i + 1 < argc) {
      args.internal_joint_refine_geometry_sigma_px = std::stod(argv[++i]);
    } else if (token == "--internal-joint-refine-observation-sigma-px" &&
               i + 1 < argc) {
      args.internal_joint_refine_observation_sigma_px = std::stod(argv[++i]);
    } else if (token == "--internal-joint-refine-subpix-window-radius" &&
               i + 1 < argc) {
      args.internal_joint_refine_subpix_window_radius = std::stoi(argv[++i]);
    } else if (token ==
                   "--internal-joint-refine-min-objective-improvement" &&
               i + 1 < argc) {
      args.internal_joint_refine_min_objective_improvement =
          std::stod(argv[++i]);
    } else if (token == "--internal-joint-refine-min-old-residual-px" &&
               i + 1 < argc) {
      args.internal_joint_refine_min_old_residual_px = std::stod(argv[++i]);
    } else if (token ==
                   "--internal-joint-refine-low-patch-gradient-quantile" &&
               i + 1 < argc) {
      args.internal_joint_refine_low_patch_gradient_quantile =
          std::stod(argv[++i]);
    } else if (token == "--internal-joint-refine-min-board-rmse-px" &&
               i + 1 < argc) {
      args.internal_joint_refine_min_board_rmse_px = std::stod(argv[++i]);
    } else if (token == "--internal-joint-refine-min-board-p95-px" &&
               i + 1 < argc) {
      args.internal_joint_refine_min_board_p95_px = std::stod(argv[++i]);
    } else if (token == "--internal-joint-refine-min-corner-response-gain" &&
               i + 1 < argc) {
      args.internal_joint_refine_min_corner_response_gain = std::stod(argv[++i]);
    } else if (
        token == "--internal-joint-refine-min-board-internal-improvement-px" &&
        i + 1 < argc) {
      args.internal_joint_refine_min_board_internal_improvement_px =
          std::stod(argv[++i]);
    } else if (
        token == "--internal-joint-refine-min-refined-point-count-per-board" &&
        i + 1 < argc) {
      args.internal_joint_refine_min_refined_point_count_per_board =
          std::stoi(argv[++i]);
    } else if (
        token == "--internal-joint-refine-accept-max-global-outer-delta-px" &&
        i + 1 < argc) {
      args.internal_joint_refine_accept_max_global_outer_delta_px =
          std::stod(argv[++i]);
    } else if (
        token == "--internal-joint-refine-accept-max-frame-outer-delta-px" &&
        i + 1 < argc) {
      args.internal_joint_refine_accept_max_frame_outer_delta_px =
          std::stod(argv[++i]);
    } else if (
        token == "--internal-joint-refine-acceptance-backend-max-iterations" &&
        i + 1 < argc) {
      args.internal_joint_refine_acceptance_backend_max_iterations =
          std::stoi(argv[++i]);
    } else if (token == "--internal-blur-board-weight-mode" &&
               i + 1 < argc) {
      args.internal_blur_board_weight_mode =
          ati::ParseInternalBlurBoardWeightMode(argv[++i]);
    } else if (
        token == "--internal-blur-board-weight-low-patch-gradient-quantile" &&
        i + 1 < argc) {
      args.internal_blur_board_weight_low_patch_gradient_quantile =
          std::stod(argv[++i]);
    } else if (token == "--internal-blur-board-weight-min-board-rmse-px" &&
               i + 1 < argc) {
      args.internal_blur_board_weight_min_board_rmse_px = std::stod(argv[++i]);
    } else if (token == "--internal-blur-board-weight-min-board-p95-px" &&
               i + 1 < argc) {
      args.internal_blur_board_weight_min_board_p95_px = std::stod(argv[++i]);
    } else if (token == "--internal-blur-board-weight-min" &&
               i + 1 < argc) {
      args.internal_blur_board_weight_min = std::stod(argv[++i]);
    } else if (token == "--internal-blur-board-weight-gradient-exponent" &&
               i + 1 < argc) {
      args.internal_blur_board_weight_gradient_exponent =
          std::stod(argv[++i]);
    } else if (token == "--internal-observation-weight-mode" &&
               i + 1 < argc) {
      args.internal_observation_weight_mode =
          ati::ParseInternalObservationWeightMode(argv[++i]);
    } else if (token == "--internal-observation-weight-policy" &&
               i + 1 < argc) {
      args.internal_observation_weight_policy = argv[++i];
    } else if (token ==
                   "--internal-observation-weight-low-quality-quantile" &&
               i + 1 < argc) {
      args.internal_observation_weight_low_quality_quantile =
          std::stod(argv[++i]);
    } else if (token == "--internal-observation-weight-min" &&
               i + 1 < argc) {
      args.internal_observation_weight_min = std::stod(argv[++i]);
    } else if (token == "--internal-observation-weight-quality-exponent" &&
               i + 1 < argc) {
      args.internal_observation_weight_quality_exponent =
          std::stod(argv[++i]);
    } else if (
        token == "--internal-observation-weight-residual-consistency-sigma" &&
        i + 1 < argc) {
      args.internal_observation_weight_residual_consistency_sigma =
          std::stod(argv[++i]);
    } else if (
        token == "--internal-observation-weight-residual-consistency-min-rmse" &&
        i + 1 < argc) {
      args.internal_observation_weight_residual_consistency_min_rmse =
          std::stod(argv[++i]);
    } else if (token == "--kalibr-runtime-seconds" && i + 1 < argc) {
      args.kalibr_runtime_seconds = std::stod(argv[++i]);
    } else if (token == "--stage5-enable-polar-angle-diagnostics") {
      args.enable_polar_angle_diagnostics = true;
    } else if (token == "--stage5-polar-angle-bin-edges" && i + 1 < argc) {
      args.polar_angle_bin_edges = argv[++i];
    } else if (token == "--stage5-enable-multi-board-consistency-diagnostics") {
      args.enable_multi_board_consistency_diagnostics = true;
    } else if (token == "--stage5-multi-board-consistency-pose-source" &&
               i + 1 < argc) {
      args.multi_board_consistency_pose_source = argv[++i];
    } else if (token == "--stage5-multi-board-consistency-min-outer-points" &&
               i + 1 < argc) {
      args.multi_board_consistency_min_outer_points = std::stoi(argv[++i]);
    } else if (
        token == "--stage5-enable-global-scene-state-consistency-audit") {
      args.enable_global_scene_state_consistency_audit = true;
    } else if (token == "--stage5-enable-selection-diagnostics") {
      args.enable_stage5_selection_diagnostics = true;
    } else if (
        token == "--stage5-export-holdout-reprojection-visualizations") {
      args.export_holdout_reprojection_visualizations = true;
    } else if (token == "--stage5-disable-selected-case-visualizations") {
      args.export_selected_case_visualizations = false;
    } else if (token == "--stage5-enable-final-backend-ba") {
      throw std::runtime_error(
          "--stage5-enable-final-backend-ba has been removed. Stage5 uses only "
          "incremental selection BA; no final backend BA is available.");
    } else if (token == "--stage5-holdout-visualization-top-k" &&
               i + 1 < argc) {
      args.holdout_visualization_top_k = std::stoi(argv[++i]);
    } else if (token == "--enable-multiboard-rigidity-diagnostics") {
      args.enable_multiboard_rigidity_diagnostics = true;
    } else if (token == "--multiboard-rigidity-top-k" && i + 1 < argc) {
      args.multiboard_rigidity_top_k = std::stoi(argv[++i]);
    } else if (token == "--multiboard-rigidity-rotation-bad-threshold-deg" &&
               i + 1 < argc) {
      args.multiboard_rigidity_rotation_bad_threshold_deg = std::stod(argv[++i]);
    } else if (token == "--multiboard-rigidity-translation-bad-threshold" &&
               i + 1 < argc) {
      args.multiboard_rigidity_translation_bad_threshold = std::stod(argv[++i]);
    } else if (
        token == "--multiboard-rigidity-reprojection-delta-bad-threshold-px" &&
        i + 1 < argc) {
      args.multiboard_rigidity_reprojection_delta_bad_threshold_px =
          std::stod(argv[++i]);
    } else if (token == "--multiboard-rigidity-use-internal-points" &&
               i + 1 < argc) {
      args.multiboard_rigidity_use_internal_points =
          std::stoi(argv[++i]) != 0;
    } else if (token == "--multiboard-rigidity-use-outer-points" &&
               i + 1 < argc) {
      args.multiboard_rigidity_use_outer_points = std::stoi(argv[++i]) != 0;
    } else if (token == "--all") {
      args.all = true;
    } else if (token == "--stage5-no-holdout") {
      args.stage5_no_holdout = true;
    } else if (token == "--reference-board-id" && i + 1 < argc) {
      args.reference_board_id = std::stoi(argv[++i]);
    } else if (token == "--intrinsics-release-iteration" && i + 1 < argc) {
      args.intrinsics_release_iteration = std::stoi(argv[++i]);
    } else if (token == "--second-pass-intrinsics-release-iteration" && i + 1 < argc) {
      args.second_pass_intrinsics_release_iteration = std::stoi(argv[++i]);
    } else if (token == "--split-mode" && i + 1 < argc) {
      args.split_mode = argv[++i];
    } else if (token == "--holdout-ratio" && i + 1 < argc) {
      args.holdout_ratio = std::stod(argv[++i]);
    } else if (token == "--split-seed" && i + 1 < argc) {
      args.split_seed = static_cast<unsigned int>(std::stoul(argv[++i]));
    } else if (token == "--holdout-stride" && i + 1 < argc) {
      args.holdout_stride = std::stoi(argv[++i]);
    } else if (token == "--holdout-offset" && i + 1 < argc) {
      args.holdout_offset = std::stoi(argv[++i]);
    } else if (token == "--help" || token == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + token);
    }
  }

  const bool has_image_input = !args.image_path.empty();
  const bool has_precomputed_input = !args.precomputed_observations_dir.empty();
  if (has_image_input == has_precomputed_input) {
    throw std::runtime_error(
        "Specify exactly one of --image or --stage5-precomputed-observations-dir.");
  }
  if (args.config_path.empty() || args.output_path.empty() ||
      args.kalibr_camchain_yaml.empty()) {
    throw std::runtime_error(
        "--config, --output and --kalibr-camchain are required.");
  }
  if (has_precomputed_input && !args.test_image_path.empty()) {
    throw std::runtime_error(
        "Use --stage5-precomputed-holdout-observations-dir instead of --test-image with precomputed input.");
  }
  if (has_precomputed_input &&
      args.precomputed_holdout_observations_dir.empty()) {
    throw std::runtime_error(
        "Stage5 benchmark precomputed mode requires --stage5-precomputed-holdout-observations-dir to prevent train/holdout leakage.");
  }
  if (!has_precomputed_input &&
      !args.precomputed_holdout_observations_dir.empty() &&
      !args.allow_image_training_with_precomputed_holdout) {
    throw std::runtime_error(
        "Precomputed holdout observations require precomputed training observations.");
  }
  if (args.allow_image_training_with_precomputed_holdout &&
      has_precomputed_input) {
    throw std::runtime_error(
        "--stage5-allow-image-training-with-precomputed-holdout is only for image training input.");
  }
  if (args.allow_image_training_with_precomputed_holdout &&
      args.precomputed_holdout_observations_dir.empty()) {
    throw std::runtime_error(
        "--stage5-allow-image-training-with-precomputed-holdout requires frozen holdout observations.");
  }
  if (args.allow_image_training_with_precomputed_holdout &&
      !args.test_image_path.empty()) {
    throw std::runtime_error(
        "Use only precomputed holdout observations with image training in mixed frozen-test mode.");
  }
  if (!has_precomputed_input && args.precomputed_init_use_all_points) {
    throw std::runtime_error(
        "--stage5-precomputed-init-use-all-points requires precomputed observations.");
  }
  if (args.precomputed_initialization_point_scope != "all" &&
      args.precomputed_initialization_point_scope != "outer_only" &&
      args.precomputed_initialization_point_scope != "internal_only") {
    throw std::runtime_error(
        "--stage5-precomputed-init-point-scope must be all, outer_only, or "
        "internal_only.");
  }
  if (!args.precomputed_init_use_all_points &&
      args.precomputed_initialization_point_scope != "all") {
    throw std::runtime_error(
        "A non-default precomputed initialization point scope requires "
        "--stage5-precomputed-init-use-all-points 1.");
  }
  if (!std::isfinite(args.stage5_checkerboard_huber_delta_pixels) ||
      args.stage5_checkerboard_huber_delta_pixels < 0.0) {
    throw std::runtime_error(
        "--stage5-checkerboard-huber-delta-px must be finite and non-negative.");
  }
  if (!std::isfinite(args.stage5_checkerboard_outlier_sigma) ||
      args.stage5_checkerboard_outlier_sigma < 0.0) {
    throw std::runtime_error(
        "--stage5-checkerboard-outlier-sigma must be finite and non-negative.");
  }
  if (!std::isfinite(args.stage5_checkerboard_min_inlier_ratio) ||
      args.stage5_checkerboard_min_inlier_ratio < 0.0 ||
      args.stage5_checkerboard_min_inlier_ratio > 1.0) {
    throw std::runtime_error(
        "--stage5-checkerboard-min-inlier-ratio must be in [0, 1].");
  }
  if (args.stage5_checkerboard_min_retained_points < 4) {
    throw std::runtime_error(
        "--stage5-checkerboard-min-retained-points must be >= 4.");
  }
  if (args.backend_polar_angle_weight_mode != "none" &&
      args.backend_polar_angle_weight_mode != "diagnostic_only" &&
      args.backend_polar_angle_weight_mode != "fixed_bins" &&
      args.backend_polar_angle_weight_mode != "adaptive_sigma") {
    throw std::runtime_error(
        "--backend-polar-angle-weight-mode must be none, diagnostic_only, fixed_bins or adaptive_sigma.");
  }
  if (args.multi_board_consistency_pose_source != "outer_only") {
    throw std::runtime_error(
        "--stage5-multi-board-consistency-pose-source must be outer_only in Phase 3.");
  }
  if (args.multi_board_consistency_min_outer_points < 4) {
    throw std::runtime_error(
        "--stage5-multi-board-consistency-min-outer-points must be >= 4.");
  }
  if (args.backend_polar_angle_weight_min_scale < 0.0 ||
      args.backend_polar_angle_weight_min_scale > 1.0) {
    throw std::runtime_error(
        "--backend-polar-angle-weight-min-scale must be in [0, 1].");
  }
  if (args.backend_consistency_pose_source != "outer_only") {
    throw std::runtime_error(
        "--backend-consistency-pose-source must be outer_only.");
  }
  if (args.backend_consistency_weight_mode != "cauchy") {
    throw std::runtime_error(
        "--backend-consistency-weight-mode must be cauchy.");
  }
  if (args.backend_consistency_translation_sigma_mm <= 0.0) {
    throw std::runtime_error(
        "--backend-consistency-translation-sigma-mm must be > 0.");
  }
  if (args.backend_consistency_rotation_sigma_deg <= 0.0) {
    throw std::runtime_error(
        "--backend-consistency-rotation-sigma-deg must be > 0.");
  }
  if (args.backend_consistency_min_weight < 0.0 ||
      args.backend_consistency_min_weight > 1.0) {
    throw std::runtime_error(
        "--backend-consistency-min-weight must be in [0, 1].");
  }
  if (args.intermediate_min_visible_boards < 1) {
    throw std::runtime_error(
        "--stage5-intermediate-min-visible-boards must be >= 1.");
  }
  if (args.intermediate_max_outer_rmse_px < 0.0) {
    throw std::runtime_error(
        "--stage5-intermediate-max-outer-rmse-px must be >= 0.");
  }
  if (args.holdout_visualization_top_k < 1) {
    throw std::runtime_error(
        "--stage5-holdout-visualization-top-k must be >= 1.");
  }
  {
    const std::vector<double> bin_edges = ParseCommaSeparatedDoubles(
        args.backend_polar_angle_weight_bin_edges,
        "backend polar angle weight bin edge");
    if (bin_edges.size() < 2) {
      throw std::runtime_error(
          "--backend-polar-angle-weight-bin-edges requires at least 2 values.");
    }
    for (std::size_t i = 1; i < bin_edges.size(); ++i) {
      if (bin_edges[i] <= bin_edges[i - 1]) {
        throw std::runtime_error(
            "--backend-polar-angle-weight-bin-edges must be strictly increasing.");
      }
    }
    const std::vector<double> bin_scales = ParseCommaSeparatedDoubles(
        args.backend_polar_angle_weight_fixed_bin_scales,
        "backend polar angle fixed bin scale");
    if (bin_scales.size() != bin_edges.size() - 1) {
      throw std::runtime_error(
          "--backend-polar-angle-weight-fixed-bin-scales must have exactly one value per polar-angle bin.");
    }
    for (double scale : bin_scales) {
      if (scale < 0.0 || scale > 1.0) {
        throw std::runtime_error(
            "--backend-polar-angle-weight-fixed-bin-scales values must be in [0, 1].");
      }
    }
  }
  if (!args.stage5_no_holdout &&
      args.split_mode != "random_holdout_ratio" &&
      args.split_mode != "random_ratio" &&
      args.split_mode != "random_70_30" &&
      args.split_mode != "deterministic_stride" &&
      args.split_mode != "stride") {
    throw std::runtime_error(
        "--split-mode must be random_holdout_ratio or deterministic_stride.");
  }
  if (!args.stage5_no_holdout &&
      !(args.holdout_ratio > 0.0 && args.holdout_ratio < 1.0)) {
    throw std::runtime_error("--holdout-ratio must be in (0, 1).");
  }
  return args;
}

std::string InferDatasetLabel(const CmdArgs& args) {
  const fs::path output_dir(args.output_path);
  if (!output_dir.filename().string().empty()) {
    return output_dir.filename().string();
  }
  return fs::path(args.image_path).stem().string();
}

bool IsImageFile(const fs::path& path) {
  if (!fs::is_regular_file(path)) {
    return false;
  }
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
         extension == ".bmp" || extension == ".tif" || extension == ".tiff";
}

std::vector<std::string> CollectImagePaths(const std::string& image_path, bool all) {
  const fs::path input(image_path);
  if (!all) {
    return {image_path};
  }

  if (!fs::exists(input)) {
    throw std::runtime_error("Input path does not exist: " + image_path);
  }

  fs::path directory = input;
  if (fs::is_regular_file(input)) {
    directory = input.parent_path();
  }
  if (!fs::is_directory(directory)) {
    throw std::runtime_error("--all requires --image to point to a directory or a file inside it.");
  }

  std::vector<std::string> image_paths;
  for (fs::directory_iterator it(directory), end; it != end; ++it) {
    if (IsImageFile(it->path())) {
      image_paths.push_back(it->path().string());
    }
  }
  std::sort(image_paths.begin(), image_paths.end());
  if (image_paths.empty()) {
    throw std::runtime_error("No image files found in directory: " + directory.string());
  }
  return image_paths;
}

std::vector<ati::FrozenRound2BaselineFrameSource> BuildFrameSources(
    const std::vector<std::string>& image_paths,
    int frame_index_offset) {
  std::vector<ati::FrozenRound2BaselineFrameSource> frames;
  frames.reserve(image_paths.size());
  for (std::size_t index = 0; index < image_paths.size(); ++index) {
    ati::FrozenRound2BaselineFrameSource frame_source;
    frame_source.frame_index = frame_index_offset + static_cast<int>(index);
    frame_source.frame_label = fs::path(image_paths[index]).stem().string();
    frame_source.image_path = image_paths[index];
    frames.push_back(frame_source);
  }
  return frames;
}

void EnsureDirectoryExists(const fs::path& directory) {
  if (!directory.empty()) {
    fs::create_directories(directory);
  }
}

cv::Mat RenderLabeledCompare(
    const std::vector<std::pair<cv::Mat, std::string> >& images_and_labels) {
  std::vector<cv::Mat> valid_images;
  int target_height = 0;
  for (const auto& entry : images_and_labels) {
    if (entry.first.empty()) {
      return cv::Mat();
    }
    target_height = std::max(target_height, entry.first.rows);
  }
  if (target_height <= 0) {
    return cv::Mat();
  }

  const int banner_height = 30;
  valid_images.reserve(images_and_labels.size());
  for (const auto& entry : images_and_labels) {
    cv::Mat padded;
    const int bottom = std::max(0, target_height - entry.first.rows);
    cv::copyMakeBorder(entry.first, padded, banner_height, bottom, 0, 0,
                       cv::BORDER_CONSTANT, cv::Scalar(24, 24, 24));
    cv::putText(padded, entry.second, cv::Point(12, 21), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
    valid_images.push_back(padded);
  }

  cv::Mat compare;
  cv::hconcat(valid_images, compare);
  return compare;
}

std::string SanitizeFilenameComponent(const std::string& input) {
  std::string sanitized;
  sanitized.reserve(input.size());
  bool last_was_underscore = false;
  for (char ch : input) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch)) {
      sanitized.push_back(ch);
      last_was_underscore = false;
    } else if (!last_was_underscore) {
      sanitized.push_back('_');
      last_was_underscore = true;
    }
  }
  while (!sanitized.empty() && sanitized.back() == '_') {
    sanitized.pop_back();
  }
  if (sanitized.empty()) {
    return "case";
  }
  return sanitized;
}

std::string FormatDouble(double value) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << value;
  return stream.str();
}

double DistanceFromImageCenterPx(
    const ati::OuterBootstrapCameraIntrinsics& camera,
    const Eigen::Vector2d& xy) {
  const double cx =
      camera.resolution.width > 0 ? 0.5 * static_cast<double>(camera.resolution.width)
                                  : camera.cu;
  const double cy =
      camera.resolution.height > 0 ? 0.5 * static_cast<double>(camera.resolution.height)
                                   : camera.cv;
  return std::hypot(xy.x() - cx, xy.y() - cy);
}

double PolygonAreaPx(const std::vector<Eigen::Vector2d>& points) {
  if (points.size() < 3) {
    return 0.0;
  }
  double area = 0.0;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Eigen::Vector2d& a = points[index];
    const Eigen::Vector2d& b = points[(index + 1) % points.size()];
    area += a.x() * b.y() - b.x() * a.y();
  }
  return 0.5 * std::abs(area);
}

const ati::CameraModelRefitPointDiagnostics* FindRefitPointDiagnostics(
    const ati::CameraModelRefitEvaluationResult& evaluation,
    int frame_index,
    int board_id,
    int point_id,
    ati::JointPointType point_type) {
  for (const ati::CameraModelRefitPointDiagnostics& point :
       evaluation.point_diagnostics) {
    if (point.frame_index == frame_index && point.board_id == board_id &&
        point.point_id == point_id && point.point_type == point_type) {
      return &point;
    }
  }
  return nullptr;
}

const ati::CameraModelRefitFrameDiagnostics* FindWorstFrameByRmse(
    const ati::CameraModelRefitEvaluationResult& evaluation) {
  const ati::CameraModelRefitFrameDiagnostics* best = nullptr;
  for (const ati::CameraModelRefitFrameDiagnostics& frame :
       evaluation.frame_diagnostics) {
    if (best == nullptr || frame.rmse > best->rmse) {
      best = &frame;
    }
  }
  return best;
}

const ati::CameraModelRefitFrameDiagnostics* FindFrameDiagnosticsByIndex(
    const ati::CameraModelRefitEvaluationResult& evaluation,
    int frame_index) {
  for (const ati::CameraModelRefitFrameDiagnostics& frame :
       evaluation.frame_diagnostics) {
    if (frame.frame_index == frame_index) {
      return &frame;
    }
  }
  return nullptr;
}

const ati::CameraModelRefitBoardObservationDiagnostics*
FindWorstBoardObservationByRmse(
    const ati::CameraModelRefitEvaluationResult& evaluation) {
  const ati::CameraModelRefitBoardObservationDiagnostics* best = nullptr;
  for (const ati::CameraModelRefitBoardObservationDiagnostics& board :
       evaluation.board_observation_diagnostics) {
    if (!board.pose_only_refit_success) {
      continue;
    }
    if (best == nullptr || board.evaluation_rmse > best->evaluation_rmse) {
      best = &board;
    }
  }
  return best;
}

const ati::CameraModelRefitBoardObservationDiagnostics*
FindBoardObservationDiagnostics(
    const ati::CameraModelRefitEvaluationResult& evaluation,
    int frame_index,
    int board_id) {
  for (const ati::CameraModelRefitBoardObservationDiagnostics& board :
       evaluation.board_observation_diagnostics) {
    if (board.frame_index == frame_index && board.board_id == board_id) {
      return &board;
    }
  }
  return nullptr;
}

struct FrameDeltaCase {
  const ati::CameraModelRefitFrameDiagnostics* backend = nullptr;
  const ati::CameraModelRefitFrameDiagnostics* kalibr = nullptr;
  double backend_minus_kalibr = 0.0;
};

struct BoardDeltaCase {
  const ati::CameraModelRefitBoardObservationDiagnostics* backend = nullptr;
  const ati::CameraModelRefitBoardObservationDiagnostics* kalibr = nullptr;
  double backend_minus_kalibr = 0.0;
};

std::vector<const ati::CameraModelRefitFrameDiagnostics*> CollectWorstFrames(
    const ati::CameraModelRefitEvaluationResult& evaluation,
    std::size_t max_count) {
  std::vector<const ati::CameraModelRefitFrameDiagnostics*> frames;
  frames.reserve(evaluation.frame_diagnostics.size());
  for (const ati::CameraModelRefitFrameDiagnostics& frame :
       evaluation.frame_diagnostics) {
    frames.push_back(&frame);
  }
  std::sort(frames.begin(), frames.end(),
            [](const ati::CameraModelRefitFrameDiagnostics* lhs,
               const ati::CameraModelRefitFrameDiagnostics* rhs) {
              return lhs->rmse > rhs->rmse;
            });
  if (frames.size() > max_count) {
    frames.resize(max_count);
  }
  return frames;
}

std::vector<const ati::CameraModelRefitBoardObservationDiagnostics*>
CollectWorstBoards(const ati::CameraModelRefitEvaluationResult& evaluation,
                   std::size_t max_count) {
  std::vector<const ati::CameraModelRefitBoardObservationDiagnostics*> boards;
  boards.reserve(evaluation.board_observation_diagnostics.size());
  for (const ati::CameraModelRefitBoardObservationDiagnostics& board :
       evaluation.board_observation_diagnostics) {
    if (board.pose_only_refit_success) {
      boards.push_back(&board);
    }
  }
  std::sort(boards.begin(), boards.end(),
            [](const ati::CameraModelRefitBoardObservationDiagnostics* lhs,
               const ati::CameraModelRefitBoardObservationDiagnostics* rhs) {
              return lhs->evaluation_rmse > rhs->evaluation_rmse;
            });
  if (boards.size() > max_count) {
    boards.resize(max_count);
  }
  return boards;
}

std::vector<FrameDeltaCase> CollectBackendWorseFrames(
    const ati::CameraModelRefitEvaluationResult& backend_evaluation,
    const ati::CameraModelRefitEvaluationResult& kalibr_evaluation,
    std::size_t max_count) {
  std::vector<FrameDeltaCase> cases;
  for (const ati::CameraModelRefitFrameDiagnostics& backend_frame :
       backend_evaluation.frame_diagnostics) {
    const ati::CameraModelRefitFrameDiagnostics* kalibr_frame =
        FindFrameDiagnosticsByIndex(kalibr_evaluation, backend_frame.frame_index);
    if (kalibr_frame == nullptr || backend_frame.rmse <= kalibr_frame->rmse) {
      continue;
    }
    FrameDeltaCase item;
    item.backend = &backend_frame;
    item.kalibr = kalibr_frame;
    item.backend_minus_kalibr = backend_frame.rmse - kalibr_frame->rmse;
    cases.push_back(item);
  }
  std::sort(cases.begin(), cases.end(),
            [](const FrameDeltaCase& lhs, const FrameDeltaCase& rhs) {
              return lhs.backend_minus_kalibr > rhs.backend_minus_kalibr;
            });
  if (cases.size() > max_count) {
    cases.resize(max_count);
  }
  return cases;
}

std::vector<BoardDeltaCase> CollectBackendWorseBoards(
    const ati::CameraModelRefitEvaluationResult& backend_evaluation,
    const ati::CameraModelRefitEvaluationResult& kalibr_evaluation,
    std::size_t max_count) {
  std::vector<BoardDeltaCase> cases;
  for (const ati::CameraModelRefitBoardObservationDiagnostics& backend_board :
       backend_evaluation.board_observation_diagnostics) {
    if (!backend_board.pose_only_refit_success) {
      continue;
    }
    const ati::CameraModelRefitBoardObservationDiagnostics* kalibr_board =
        FindBoardObservationDiagnostics(kalibr_evaluation,
                                        backend_board.frame_index,
                                        backend_board.board_id);
    if (kalibr_board == nullptr || !kalibr_board->pose_only_refit_success ||
        backend_board.evaluation_rmse <= kalibr_board->evaluation_rmse) {
      continue;
    }
    BoardDeltaCase item;
    item.backend = &backend_board;
    item.kalibr = kalibr_board;
    item.backend_minus_kalibr =
        backend_board.evaluation_rmse - kalibr_board->evaluation_rmse;
    cases.push_back(item);
  }
  std::sort(cases.begin(), cases.end(),
            [](const BoardDeltaCase& lhs, const BoardDeltaCase& rhs) {
              return lhs.backend_minus_kalibr > rhs.backend_minus_kalibr;
            });
  if (cases.size() > max_count) {
    cases.resize(max_count);
  }
  return cases;
}

void WriteWorstReprojectionSummary(
    const fs::path& output_dir,
    const ati::CameraModelRefitFrameDiagnostics* worst_frame,
    const ati::CameraModelRefitBoardObservationDiagnostics* worst_board,
    const ati::CameraModelRefitEvaluationResult& backend_holdout_evaluation,
    const ati::CameraModelRefitEvaluationResult& kalibr_holdout_evaluation) {
  std::ofstream output((output_dir / "worst_reprojection_summary.txt").string().c_str());
  output << "selection_source: backend_holdout_evaluation\n";
  output << "backend_holdout_overall_rmse: "
         << backend_holdout_evaluation.overall_rmse << "\n";
  output << "kalibr_holdout_overall_rmse: "
         << kalibr_holdout_evaluation.overall_rmse << "\n";
  if (worst_frame != nullptr) {
    output << "worst_frame_index: " << worst_frame->frame_index << "\n";
    output << "worst_frame_label: " << worst_frame->frame_label << "\n";
    output << "worst_frame_backend_rmse: " << worst_frame->rmse << "\n";
    output << "worst_frame_backend_outer_rmse: " << worst_frame->outer_rmse << "\n";
    output << "worst_frame_backend_internal_rmse: "
           << worst_frame->internal_rmse << "\n";
    output << "worst_frame_backend_point_count: "
           << worst_frame->point_count << "\n";
  } else {
    output << "worst_frame_index: -1\n";
  }
  if (worst_board != nullptr) {
    output << "worst_board_frame_index: " << worst_board->frame_index << "\n";
    output << "worst_board_frame_label: " << worst_board->frame_label << "\n";
    output << "worst_board_id: " << worst_board->board_id << "\n";
    output << "worst_board_backend_rmse: "
           << worst_board->evaluation_rmse << "\n";
    output << "worst_board_backend_outer_rmse: "
           << worst_board->outer_evaluation_rmse << "\n";
    output << "worst_board_backend_internal_rmse: "
           << worst_board->internal_evaluation_rmse << "\n";
    output << "worst_board_backend_point_count: "
           << worst_board->point_count << "\n";
  } else {
    output << "worst_board_id: -1\n";
  }
  output << "worst_reprojection_frame_backend_png: "
         << (output_dir / "worst_reprojection_frame_backend.png").string() << "\n";
  output << "worst_reprojection_frame_backend_vs_kalibr_png: "
         << (output_dir / "worst_reprojection_frame_backend_vs_kalibr.png").string()
         << "\n";
  output << "worst_reprojection_board_backend_png: "
         << (output_dir / "worst_reprojection_board_backend.png").string() << "\n";
  output << "worst_reprojection_board_backend_vs_kalibr_png: "
         << (output_dir / "worst_reprojection_board_backend_vs_kalibr.png").string()
         << "\n";
}

void ExportWorstReprojectionOverlays(
    const fs::path& output_dir,
    const ati::Stage5Benchmark& benchmark,
    const ati::Stage5BenchmarkReport& report,
    const ati::CameraModelRefitEvaluationResult& backend_holdout_evaluation) {
  const ati::CameraModelRefitFrameDiagnostics* worst_frame =
      FindWorstFrameByRmse(backend_holdout_evaluation);
  const ati::CameraModelRefitBoardObservationDiagnostics* worst_board =
      FindWorstBoardObservationByRmse(backend_holdout_evaluation);

  if (worst_frame != nullptr) {
    const cv::Mat backend_overlay = benchmark.RenderEvaluationFrameOverlay(
        report, backend_holdout_evaluation, worst_frame->frame_index);
    const cv::Mat kalibr_overlay = benchmark.RenderEvaluationFrameOverlay(
        report, report.kalibr_holdout_evaluation, worst_frame->frame_index);
    if (!backend_overlay.empty()) {
      cv::imwrite((output_dir / "worst_reprojection_frame_backend.png").string(),
                  backend_overlay);
    }
    const cv::Mat compare = RenderLabeledCompare({
        std::make_pair(backend_overlay, "backend worst frame"),
        std::make_pair(kalibr_overlay, "kalibr same frame"),
    });
    if (!compare.empty()) {
      cv::imwrite(
          (output_dir / "worst_reprojection_frame_backend_vs_kalibr.png").string(),
          compare);
    }
  }

  if (worst_board != nullptr) {
    const cv::Mat backend_overlay =
        benchmark.RenderEvaluationBoardObservationOverlay(
            report, backend_holdout_evaluation, worst_board->frame_index,
            worst_board->board_id);
    const cv::Mat kalibr_overlay =
        benchmark.RenderEvaluationBoardObservationOverlay(
            report, report.kalibr_holdout_evaluation, worst_board->frame_index,
            worst_board->board_id);
    if (!backend_overlay.empty()) {
      cv::imwrite((output_dir / "worst_reprojection_board_backend.png").string(),
                  backend_overlay);
    }
    const cv::Mat compare = RenderLabeledCompare({
        std::make_pair(backend_overlay, "backend worst board"),
        std::make_pair(kalibr_overlay, "kalibr same board"),
    });
    if (!compare.empty()) {
      cv::imwrite(
          (output_dir / "worst_reprojection_board_backend_vs_kalibr.png").string(),
          compare);
    }
  }

  WriteWorstReprojectionSummary(output_dir, worst_frame, worst_board,
                                backend_holdout_evaluation,
                                report.kalibr_holdout_evaluation);
}

void ExportSelectedCaseVisualizations(
    const fs::path& output_dir,
    const ati::Stage5Benchmark& benchmark,
    const ati::Stage5BenchmarkReport& report,
    const ati::CameraModelRefitEvaluationResult& backend_holdout_evaluation) {
  const fs::path selected_dir = output_dir / "selected_case_visualizations";
  EnsureDirectoryExists(selected_dir);

  const std::vector<const ati::CameraModelRefitFrameDiagnostics*> worst_frames =
      CollectWorstFrames(backend_holdout_evaluation, 3);
  const std::vector<const ati::CameraModelRefitBoardObservationDiagnostics*>
      worst_boards = CollectWorstBoards(backend_holdout_evaluation, 3);
  const std::vector<FrameDeltaCase> backend_worse_frames =
      CollectBackendWorseFrames(backend_holdout_evaluation,
                                report.kalibr_holdout_evaluation, 3);
  const std::vector<BoardDeltaCase> backend_worse_boards =
      CollectBackendWorseBoards(backend_holdout_evaluation,
                                report.kalibr_holdout_evaluation, 3);

  std::ofstream summary((selected_dir / "selected_case_summary.txt").string().c_str());
  summary << "selection_source: backend_holdout_evaluation\n";
  summary << "backend_holdout_overall_rmse: "
          << backend_holdout_evaluation.overall_rmse << "\n";
  summary << "kalibr_holdout_overall_rmse: "
          << report.kalibr_holdout_evaluation.overall_rmse << "\n";
  summary << "worst_frame_case_count: " << worst_frames.size() << "\n";
  summary << "worst_board_case_count: " << worst_boards.size() << "\n";
  summary << "backend_worse_than_kalibr_frame_count: "
          << backend_worse_frames.size() << "\n";
  summary << "backend_worse_than_kalibr_board_count: "
          << backend_worse_boards.size() << "\n";

  for (std::size_t rank = 0; rank < worst_frames.size(); ++rank) {
    const ati::CameraModelRefitFrameDiagnostics* backend_frame = worst_frames[rank];
    const ati::CameraModelRefitFrameDiagnostics* kalibr_frame =
        FindFrameDiagnosticsByIndex(report.kalibr_holdout_evaluation,
                                    backend_frame->frame_index);
    const cv::Mat backend_overlay = benchmark.RenderEvaluationFrameOverlay(
        report, backend_holdout_evaluation, backend_frame->frame_index);
    const cv::Mat kalibr_overlay = benchmark.RenderEvaluationFrameOverlay(
        report, report.kalibr_holdout_evaluation, backend_frame->frame_index);
    const cv::Mat compare = RenderLabeledCompare({
        std::make_pair(
            backend_overlay,
            "backend rmse=" + FormatDouble(backend_frame->rmse)),
        std::make_pair(
            kalibr_overlay,
            "kalibr rmse=" +
                FormatDouble(kalibr_frame == nullptr ? 0.0 : kalibr_frame->rmse)),
    });
    if (!compare.empty()) {
      const std::string filename =
          "worst_backend_frame_rank" +
          std::to_string(static_cast<int>(rank + 1)) + "_frame_" +
          std::to_string(backend_frame->frame_index) + "_" +
          SanitizeFilenameComponent(backend_frame->frame_label) + ".png";
      cv::imwrite((selected_dir / filename).string(), compare);
      summary << "worst_backend_frame_rank" << (rank + 1) << ": "
              << filename << "\n";
    }
  }

  for (std::size_t rank = 0; rank < worst_boards.size(); ++rank) {
    const ati::CameraModelRefitBoardObservationDiagnostics* backend_board =
        worst_boards[rank];
    const ati::CameraModelRefitBoardObservationDiagnostics* kalibr_board =
        FindBoardObservationDiagnostics(report.kalibr_holdout_evaluation,
                                        backend_board->frame_index,
                                        backend_board->board_id);
    const cv::Mat backend_overlay =
        benchmark.RenderEvaluationBoardObservationOverlay(
            report, backend_holdout_evaluation, backend_board->frame_index,
            backend_board->board_id);
    const cv::Mat kalibr_overlay =
        benchmark.RenderEvaluationBoardObservationOverlay(
            report, report.kalibr_holdout_evaluation, backend_board->frame_index,
            backend_board->board_id);
    const cv::Mat compare = RenderLabeledCompare({
        std::make_pair(
            backend_overlay,
            "backend rmse=" + FormatDouble(backend_board->evaluation_rmse)),
        std::make_pair(
            kalibr_overlay,
            "kalibr rmse=" + FormatDouble(
                                  kalibr_board == nullptr
                                      ? 0.0
                                      : kalibr_board->evaluation_rmse)),
    });
    if (!compare.empty()) {
      const std::string filename =
          "worst_backend_board_rank" +
          std::to_string(static_cast<int>(rank + 1)) + "_frame_" +
          std::to_string(backend_board->frame_index) + "_board_" +
          std::to_string(backend_board->board_id) + "_" +
          SanitizeFilenameComponent(backend_board->frame_label) + ".png";
      cv::imwrite((selected_dir / filename).string(), compare);
      summary << "worst_backend_board_rank" << (rank + 1) << ": "
              << filename << "\n";
    }
  }

  for (std::size_t rank = 0; rank < backend_worse_frames.size(); ++rank) {
    const FrameDeltaCase& item = backend_worse_frames[rank];
    const cv::Mat backend_overlay = benchmark.RenderEvaluationFrameOverlay(
        report, backend_holdout_evaluation, item.backend->frame_index);
    const cv::Mat kalibr_overlay = benchmark.RenderEvaluationFrameOverlay(
        report, report.kalibr_holdout_evaluation, item.backend->frame_index);
    const cv::Mat compare = RenderLabeledCompare({
        std::make_pair(
            backend_overlay,
            "backend rmse=" + FormatDouble(item.backend->rmse)),
        std::make_pair(
            kalibr_overlay,
            "kalibr rmse=" + FormatDouble(item.kalibr->rmse)),
    });
    if (!compare.empty()) {
      const std::string filename =
          "backend_worse_than_kalibr_frame_rank" +
          std::to_string(static_cast<int>(rank + 1)) + "_frame_" +
          std::to_string(item.backend->frame_index) + "_" +
          SanitizeFilenameComponent(item.backend->frame_label) + ".png";
      cv::imwrite((selected_dir / filename).string(), compare);
      summary << "backend_worse_frame_rank" << (rank + 1)
              << "_backend_minus_kalibr: "
              << item.backend_minus_kalibr << "\n";
      summary << "backend_worse_frame_rank" << (rank + 1)
              << "_png: " << filename << "\n";
    }
  }

  for (std::size_t rank = 0; rank < backend_worse_boards.size(); ++rank) {
    const BoardDeltaCase& item = backend_worse_boards[rank];
    const cv::Mat backend_overlay =
        benchmark.RenderEvaluationBoardObservationOverlay(
            report, backend_holdout_evaluation, item.backend->frame_index,
            item.backend->board_id);
    const cv::Mat kalibr_overlay =
        benchmark.RenderEvaluationBoardObservationOverlay(
            report, report.kalibr_holdout_evaluation, item.backend->frame_index,
            item.backend->board_id);
    const cv::Mat compare = RenderLabeledCompare({
        std::make_pair(
            backend_overlay,
            "backend rmse=" + FormatDouble(item.backend->evaluation_rmse)),
        std::make_pair(
            kalibr_overlay,
            "kalibr rmse=" + FormatDouble(item.kalibr->evaluation_rmse)),
    });
    if (!compare.empty()) {
      const std::string filename =
          "backend_worse_than_kalibr_board_rank" +
          std::to_string(static_cast<int>(rank + 1)) + "_frame_" +
          std::to_string(item.backend->frame_index) + "_board_" +
          std::to_string(item.backend->board_id) + "_" +
          SanitizeFilenameComponent(item.backend->frame_label) + ".png";
      cv::imwrite((selected_dir / filename).string(), compare);
      summary << "backend_worse_board_rank" << (rank + 1)
              << "_backend_minus_kalibr: "
              << item.backend_minus_kalibr << "\n";
      summary << "backend_worse_board_rank" << (rank + 1)
              << "_png: " << filename << "\n";
    }
  }

}

void ExportHoldoutReprojectionVisualizations(
    const fs::path& output_dir,
    const ati::Stage5Benchmark& benchmark,
    const ati::Stage5BenchmarkReport& report,
    const ati::CameraModelRefitEvaluationResult& backend_holdout_evaluation,
    int top_k) {
  const fs::path viz_dir = output_dir / "holdout_reprojection_visualizations";
  const fs::path frame_dir = viz_dir / "frames";
  const fs::path board_dir = viz_dir / "boards";
  EnsureDirectoryExists(frame_dir);
  EnsureDirectoryExists(board_dir);

  const std::size_t max_count =
      static_cast<std::size_t>(std::max(1, top_k));
  const std::vector<const ati::CameraModelRefitFrameDiagnostics*> worst_frames =
      CollectWorstFrames(backend_holdout_evaluation, max_count);
  const std::vector<const ati::CameraModelRefitBoardObservationDiagnostics*>
      worst_boards = CollectWorstBoards(backend_holdout_evaluation, max_count);

  std::ofstream summary((viz_dir / "holdout_reprojection_visualization_summary.txt")
                            .string()
                            .c_str());
  summary << "purpose: visualize top holdout/test-image reprojection failures "
             "using the same pose-only refit evaluation that produces "
             "backend_holdout_summary.txt.\n";
  summary << "top_k_requested: " << top_k << "\n";
  summary << "backend_holdout_overall_rmse: "
          << backend_holdout_evaluation.overall_rmse << "\n";
  summary << "backend_holdout_outer_only_rmse: "
          << backend_holdout_evaluation.outer_only_rmse << "\n";
  summary << "backend_holdout_internal_only_rmse: "
          << backend_holdout_evaluation.internal_only_rmse << "\n";
  summary << "kalibr_holdout_overall_rmse: "
          << report.kalibr_holdout_evaluation.overall_rmse << "\n";
  summary << "frame_overlay_dir: " << frame_dir.string() << "\n";
  summary << "board_overlay_dir: " << board_dir.string() << "\n";
  summary << "legend: backend images show observed points, predicted points, "
             "and residual vectors from Stage5Benchmark evaluation overlays.\n";
  summary << "\n[frames]\n";

  for (std::size_t rank = 0; rank < worst_frames.size(); ++rank) {
    const ati::CameraModelRefitFrameDiagnostics* backend_frame =
        worst_frames[rank];
    const ati::CameraModelRefitFrameDiagnostics* kalibr_frame =
        FindFrameDiagnosticsByIndex(report.kalibr_holdout_evaluation,
                                    backend_frame->frame_index);
    const cv::Mat backend_overlay = benchmark.RenderEvaluationFrameOverlay(
        report, backend_holdout_evaluation, backend_frame->frame_index);
    const cv::Mat kalibr_overlay = benchmark.RenderEvaluationFrameOverlay(
        report, report.kalibr_holdout_evaluation, backend_frame->frame_index);

    const std::string stem =
        "rank" + std::to_string(static_cast<int>(rank + 1)) + "_frame_" +
        std::to_string(backend_frame->frame_index) + "_" +
        SanitizeFilenameComponent(backend_frame->frame_label);
    const std::string backend_filename = stem + "_backend.png";
    const std::string compare_filename = stem + "_backend_vs_kalibr.png";
    if (!backend_overlay.empty()) {
      cv::imwrite((frame_dir / backend_filename).string(), backend_overlay);
    }
    const cv::Mat compare = RenderLabeledCompare({
        std::make_pair(backend_overlay,
                       "backend frame rmse=" +
                           FormatDouble(backend_frame->rmse)),
        std::make_pair(kalibr_overlay,
                       "kalibr frame rmse=" +
                           FormatDouble(kalibr_frame == nullptr
                                            ? 0.0
                                            : kalibr_frame->rmse)),
    });
    if (!compare.empty()) {
      cv::imwrite((frame_dir / compare_filename).string(), compare);
    }
    summary << "rank" << (rank + 1)
            << ",frame_index=" << backend_frame->frame_index
            << ",frame_label=" << backend_frame->frame_label
            << ",backend_rmse=" << backend_frame->rmse
            << ",backend_outer_rmse=" << backend_frame->outer_rmse
            << ",backend_internal_rmse=" << backend_frame->internal_rmse
            << ",kalibr_rmse="
            << (kalibr_frame == nullptr ? 0.0 : kalibr_frame->rmse)
            << ",backend_png=" << backend_filename
            << ",compare_png=" << compare_filename << "\n";
  }

  summary << "\n[boards]\n";
  for (std::size_t rank = 0; rank < worst_boards.size(); ++rank) {
    const ati::CameraModelRefitBoardObservationDiagnostics* backend_board =
        worst_boards[rank];
    const ati::CameraModelRefitBoardObservationDiagnostics* kalibr_board =
        FindBoardObservationDiagnostics(report.kalibr_holdout_evaluation,
                                        backend_board->frame_index,
                                        backend_board->board_id);
    const cv::Mat backend_overlay =
        benchmark.RenderEvaluationBoardObservationOverlay(
            report, backend_holdout_evaluation, backend_board->frame_index,
            backend_board->board_id);
    const cv::Mat kalibr_overlay =
        benchmark.RenderEvaluationBoardObservationOverlay(
            report, report.kalibr_holdout_evaluation, backend_board->frame_index,
            backend_board->board_id);

    const std::string stem =
        "rank" + std::to_string(static_cast<int>(rank + 1)) + "_frame_" +
        std::to_string(backend_board->frame_index) + "_board_" +
        std::to_string(backend_board->board_id) + "_" +
        SanitizeFilenameComponent(backend_board->frame_label);
    const std::string backend_filename = stem + "_backend.png";
    const std::string compare_filename = stem + "_backend_vs_kalibr.png";
    if (!backend_overlay.empty()) {
      cv::imwrite((board_dir / backend_filename).string(), backend_overlay);
    }
    const cv::Mat compare = RenderLabeledCompare({
        std::make_pair(backend_overlay,
                       "backend board rmse=" +
                           FormatDouble(backend_board->evaluation_rmse)),
        std::make_pair(kalibr_overlay,
                       "kalibr board rmse=" +
                           FormatDouble(kalibr_board == nullptr
                                            ? 0.0
                                            : kalibr_board->evaluation_rmse)),
    });
    if (!compare.empty()) {
      cv::imwrite((board_dir / compare_filename).string(), compare);
    }
    summary << "rank" << (rank + 1)
            << ",frame_index=" << backend_board->frame_index
            << ",frame_label=" << backend_board->frame_label
            << ",board_id=" << backend_board->board_id
            << ",backend_rmse=" << backend_board->evaluation_rmse
            << ",backend_outer_rmse=" << backend_board->outer_evaluation_rmse
            << ",backend_internal_rmse="
            << backend_board->internal_evaluation_rmse
            << ",pose_fit_outer_rmse=" << backend_board->pose_fit_outer_rmse
            << ",point_count=" << backend_board->point_count
            << ",outer_point_count=" << backend_board->outer_point_count
            << ",internal_point_count=" << backend_board->internal_point_count
            << ",kalibr_rmse="
            << (kalibr_board == nullptr ? 0.0 : kalibr_board->evaluation_rmse)
            << ",backend_png=" << backend_filename
            << ",compare_png=" << compare_filename << "\n";
  }
}

std::string JoinStringVector(const std::vector<std::string>& values,
                             const std::string& separator);
std::string JoinDoubleVector(const std::vector<double>& values,
                             const std::string& separator);

void WriteEvaluationSummary(const std::string& path,
                            const ati::CameraModelRefitEvaluationResult& evaluation) {
  std::ofstream output(path.c_str());
  output << "success: " << (evaluation.success ? 1 : 0) << "\n";
  output << "failure_reason: " << evaluation.failure_reason << "\n";
  output << "method_label: " << evaluation.method_label << "\n";
  output << "split_label: " << evaluation.split_label << "\n";
  output << "split_signature: " << evaluation.split_signature << "\n";
  output << "evaluation_metric_name: pixel_reprojection\n";
  output << "evaluation_metric_unit: px\n";
  output << "evaluation_protocol: "
         << (evaluation.uniform_control_point_mode
                 ? "checkerboard_all_control_points_pose_refit_and_reprojection"
                 : "outer_pose_refit_plus_all_point_reprojection")
         << "\n";
  output << (evaluation.uniform_control_point_mode ? "test_pose_refit_rmse: "
                                                   : "pose_only_refit_rmse: ")
         << evaluation.pose_only_refit_rmse << "\n";
  output << (evaluation.uniform_control_point_mode
                 ? "test_pose_refit_success_rate: "
                 : "pose_only_refit_success_rate: ")
         << evaluation.pose_only_refit_success_rate << "\n";
  output << (evaluation.uniform_control_point_mode
                 ? "test_pose_refit_attempt_count: "
                 : "pose_only_refit_attempt_count: ")
         << evaluation.pose_only_refit_attempt_count << "\n";
  output << (evaluation.uniform_control_point_mode
                 ? "test_pose_refit_success_count: "
                 : "pose_only_refit_success_count: ")
         << evaluation.pose_only_refit_success_count << "\n";
  output << "overall_rmse: " << evaluation.overall_rmse << "\n";
  output << "overall_angular_rmse_rad: "
         << evaluation.overall_angular_rmse_rad << "\n";
  output << "overall_angular_rmse_deg: "
         << evaluation.overall_angular_rmse_deg << "\n";
  output << "angular_point_count: " << evaluation.angular_point_count << "\n";
  output << "p95_reprojection_error: "
         << evaluation.p95_reprojection_error << "\n";
  if (evaluation.uniform_control_point_mode) {
    output << "test_rmse_all_control_points: " << evaluation.overall_rmse << "\n";
  } else {
    output << "outer_only_rmse: " << evaluation.outer_only_rmse << "\n";
    output << "internal_only_rmse: " << evaluation.internal_only_rmse << "\n";
    output << "excluded_board_id_for_rmse: "
           << evaluation.excluded_board_id_for_rmse << "\n";
    output << "overall_rmse_excluding_board: "
           << evaluation.overall_rmse_excluding_board << "\n";
    output << "outer_only_rmse_excluding_board: "
           << evaluation.outer_only_rmse_excluding_board << "\n";
    output << "internal_only_rmse_excluding_board: "
           << evaluation.internal_only_rmse_excluding_board << "\n";
  }
  output << "mean_residual_x: " << evaluation.mean_residual_x << "\n";
  output << "mean_residual_y: " << evaluation.mean_residual_y << "\n";
  output << "std_residual_x: " << evaluation.std_residual_x << "\n";
  output << "std_residual_y: " << evaluation.std_residual_y << "\n";
  output << "point_count: " << evaluation.point_count << "\n";
  if (!evaluation.uniform_control_point_mode) {
    output << "outer_point_count: " << evaluation.outer_point_count << "\n";
    output << "internal_point_count: " << evaluation.internal_point_count << "\n";
    output << "point_count_excluding_board: "
           << evaluation.point_count_excluding_board << "\n";
    output << "outer_point_count_excluding_board: "
           << evaluation.outer_point_count_excluding_board << "\n";
    output << "internal_point_count_excluding_board: "
           << evaluation.internal_point_count_excluding_board << "\n";
  }
  output << "camera_xi: " << evaluation.camera.xi << "\n";
  output << "camera_alpha: " << evaluation.camera.alpha << "\n";
  output << "camera_fu: " << evaluation.camera.fu << "\n";
  output << "camera_fv: " << evaluation.camera.fv << "\n";
  output << "camera_cu: " << evaluation.camera.cu << "\n";
  output << "camera_cv: " << evaluation.camera.cv << "\n";
  output << "camera_model_family: "
         << evaluation.camera.NormalizedFamilyString() << "\n";
  output << "camera_model: " << evaluation.camera.camera_model << "\n";
  output << "camera_distortion_model: "
         << evaluation.camera.distortion_model << "\n";
  output << "camera_intrinsics_labels: "
         << JoinStringVector(evaluation.camera.IntrinsicsLabels(), ",") << "\n";
  output << "camera_intrinsics_csv: "
         << JoinDoubleVector(evaluation.camera.IntrinsicsVector(), ",") << "\n";
  output << "camera_distortion_labels: "
         << JoinStringVector(evaluation.camera.DistortionLabels(), ",") << "\n";
  output << "camera_distortion_csv: "
         << JoinDoubleVector(evaluation.camera.DistortionVector(), ",") << "\n";
  output << "camera_combined_labels: "
         << JoinStringVector(evaluation.camera.CombinedParameterLabels(), ",")
         << "\n";
  output << "camera_combined_csv: "
         << JoinDoubleVector(evaluation.camera.CombinedParameterVector(), ",")
         << "\n";
  for (const std::string& warning : evaluation.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteCommittedBackendTrainingSummary(
    const std::string& path,
    const ati::AslamBackendCalibrationResult& backend_result) {
  const ati::JointResidualEvaluationResult& evaluation =
      backend_result.optimized_residual;
  std::ofstream output(path.c_str());
  output << "success: " << (evaluation.success ? 1 : 0) << "\n";
  output << "failure_reason: " << evaluation.failure_reason << "\n";
  output << "method_label: backend_committed_state\n";
  output << "split_label: training\n";
  output << "split_signature: "
         << backend_result.training_split_signature
         << "_backend_committed_state\n";
  output << "evaluation_pose_mode: committed_backend_scene_state\n";
  output << "evaluation_metric_name: "
         << ati::ToString(backend_result.options.residual_model) << "\n";
  const ati::ResidualModel training_residual_model =
      backend_result.options.residual_model;
  output << "evaluation_metric_unit: "
         << (training_residual_model == ati::ResidualModel::ImagePlane
                 ? "px"
                 : (training_residual_model == ati::ResidualModel::SphereAngular ||
                            training_residual_model ==
                                ati::ResidualModel::NormalizedSphereAngular
                        ? "rad"
                        : (training_residual_model == ati::ResidualModel::Chordal
                               ? "unit_bearing_chord"
                               : "px_equivalent")))
         << "\n";
  output << "uniform_control_point_mode: "
         << (backend_result.options.uniform_control_point_mode ? 1 : 0) << "\n";
  output << "pose_only_refit_rmse: nan\n";
  output << "pose_only_refit_success_rate: nan\n";
  output << "pose_only_refit_attempt_count: 0\n";
  output << "pose_only_refit_success_count: 0\n";
  output << "overall_rmse: " << evaluation.overall_rmse << "\n";
  if (backend_result.options.uniform_control_point_mode) {
    output << "training_rmse_all_control_points: "
           << evaluation.overall_rmse << "\n";
  } else {
    output << "outer_only_rmse: " << evaluation.outer_only_rmse << "\n";
    output << "internal_only_rmse: " << evaluation.internal_only_rmse << "\n";
    output << "excluded_board_id_for_rmse: "
           << backend_result.optimized_scene_state.reference_board_id << "\n";
    output << "overall_rmse_excluding_board: nan\n";
    output << "outer_only_rmse_excluding_board: nan\n";
    output << "internal_only_rmse_excluding_board: nan\n";
  }
  output << "mean_residual_x: nan\n";
  output << "mean_residual_y: nan\n";
  output << "std_residual_x: nan\n";
  output << "std_residual_y: nan\n";
  output << "point_count: " << evaluation.point_diagnostics.size() << "\n";
  int outer_count = 0;
  int internal_count = 0;
  for (const ati::JointResidualPointDiagnostics& point :
       evaluation.point_diagnostics) {
    if (point.point_type == ati::JointPointType::Outer) {
      ++outer_count;
    } else {
      ++internal_count;
    }
  }
  if (!backend_result.options.uniform_control_point_mode) {
    output << "outer_point_count: " << outer_count << "\n";
    output << "internal_point_count: " << internal_count << "\n";
    output << "point_count_excluding_board: 0\n";
    output << "outer_point_count_excluding_board: 0\n";
    output << "internal_point_count_excluding_board: 0\n";
  }
  output << "camera_xi: "
         << backend_result.optimized_scene_state.camera.xi << "\n";
  output << "camera_alpha: "
         << backend_result.optimized_scene_state.camera.alpha << "\n";
  output << "camera_fu: "
         << backend_result.optimized_scene_state.camera.fu << "\n";
  output << "camera_fv: "
         << backend_result.optimized_scene_state.camera.fv << "\n";
  output << "camera_cu: "
         << backend_result.optimized_scene_state.camera.cu << "\n";
  output << "camera_cv: "
         << backend_result.optimized_scene_state.camera.cv << "\n";
  output << "camera_model_family: "
         << backend_result.optimized_scene_state.camera
                .NormalizedFamilyString()
         << "\n";
  output << "camera_model: "
         << backend_result.optimized_scene_state.camera.camera_model
         << "\n";
  output << "camera_distortion_model: "
         << backend_result.optimized_scene_state.camera
                .distortion_model
         << "\n";
  output << "camera_intrinsics_labels: "
         << JoinStringVector(
                backend_result.optimized_scene_state.camera
                    .IntrinsicsLabels(),
                ",")
         << "\n";
  output << "camera_intrinsics_csv: "
         << JoinDoubleVector(
                backend_result.optimized_scene_state.camera
                    .IntrinsicsVector(),
                ",")
         << "\n";
  output << "camera_distortion_labels: "
         << JoinStringVector(
                backend_result.optimized_scene_state.camera
                    .DistortionLabels(),
                ",")
         << "\n";
  output << "camera_distortion_csv: "
         << JoinDoubleVector(
                backend_result.optimized_scene_state.camera
                    .DistortionVector(),
                ",")
         << "\n";
  output << "camera_combined_labels: "
         << JoinStringVector(
                backend_result.optimized_scene_state.camera
                    .CombinedParameterLabels(),
                ",")
         << "\n";
  output << "camera_combined_csv: "
         << JoinDoubleVector(
                backend_result.optimized_scene_state.camera
                    .CombinedParameterVector(),
                ",")
         << "\n";
  output << "warning: training summary uses committed backend scene poses; "
            "holdout summary uses per-board pose-only refit because holdout "
            "frames are not in the optimized training scene.\n";
}

void WriteCommittedBackendTrainingPointsCsv(
    const std::string& path,
    const ati::JointResidualEvaluationResult& evaluation) {
  std::ofstream output(path.c_str());
  output << "method,split,frame_index,frame_label,board_id,point_id,point_type,"
         << "observed_x,observed_y,predicted_x,predicted_y,target_x,target_y,"
         << "target_z,residual_x,residual_y,residual_norm,debug_quality,"
         << "source_kind,source_point_index\n";
  for (const ati::JointResidualPointDiagnostics& point :
       evaluation.point_diagnostics) {
    output << "backend_committed_state,training,"
           << point.frame_index << ","
           << point.frame_label << ","
           << point.board_id << ","
           << point.point_id << ","
           << ati::ToString(point.point_type) << ","
           << point.observed_image_xy.x() << ","
           << point.observed_image_xy.y() << ","
           << point.predicted_image_xy.x() << ","
           << point.predicted_image_xy.y() << ","
           << point.target_xyz_board.x() << ","
           << point.target_xyz_board.y() << ","
           << point.target_xyz_board.z() << ","
           << point.residual_xy.x() << ","
           << point.residual_xy.y() << ","
           << point.residual_norm << ","
           << point.quality << ","
           << ati::ToString(point.source_kind) << ","
           << point.source_point_index << "\n";
  }
}

void WriteBackendBoardPosesCsv(
    const std::string& path,
    const ati::AslamBackendCalibrationResult& backend_result) {
  std::ofstream output(path.c_str());
  output << std::setprecision(12);
  output << "board_id,initialized,observation_count,rmse,T_reference_board_16\n";
  for (const ati::JointSceneBoardState& board :
       backend_result.optimized_scene_state.boards) {
    output << board.board_id << "," << (board.initialized ? 1 : 0)
           << "," << board.observation_count << "," << board.rmse << ","
           << MatrixToCsv(board.T_reference_board) << "\n";
  }
}

ati::CameraModelRefitEvaluationResult MakeCommittedBackendTrainingEvaluation(
    const ati::AslamBackendCalibrationResult& backend_result) {
  ati::CameraModelRefitEvaluationResult result;
  const ati::JointResidualEvaluationResult& residual =
      backend_result.optimized_residual;
  result.success = residual.success;
  result.method_label = "backend";
  result.split_label = "training";
  result.split_signature =
      backend_result.training_split_signature +
      "_backend_committed_state";
  result.camera = backend_result.optimized_scene_state.camera;
  result.uniform_control_point_mode =
      backend_result.options.uniform_control_point_mode;
  result.pose_only_refit_attempt_count = 0;
  result.pose_only_refit_success_count = 0;
  result.pose_only_refit_success_rate = 0.0;
  result.pose_only_refit_rmse = residual.outer_only_rmse;
  result.overall_rmse = residual.overall_rmse;
  result.outer_only_rmse = residual.outer_only_rmse;
  result.internal_only_rmse = residual.internal_only_rmse;
  result.excluded_board_id_for_rmse =
      backend_result.optimized_scene_state.reference_board_id;
  result.overall_rmse_excluding_board = std::numeric_limits<double>::quiet_NaN();
  result.outer_only_rmse_excluding_board =
      std::numeric_limits<double>::quiet_NaN();
  result.internal_only_rmse_excluding_board =
      std::numeric_limits<double>::quiet_NaN();
  result.mean_residual_x = std::numeric_limits<double>::quiet_NaN();
  result.mean_residual_y = std::numeric_limits<double>::quiet_NaN();
  result.std_residual_x = std::numeric_limits<double>::quiet_NaN();
  result.std_residual_y = std::numeric_limits<double>::quiet_NaN();

  for (const ati::JointResidualPointDiagnostics& point :
       residual.point_diagnostics) {
    ++result.point_count;
    if (point.point_type == ati::JointPointType::Outer) {
      ++result.outer_point_count;
    } else {
      ++result.internal_point_count;
    }
  }
  result.point_count_excluding_board = 0;
  result.outer_point_count_excluding_board = 0;
  result.internal_point_count_excluding_board = 0;
  result.evaluated_frame_count =
      static_cast<int>(residual.frame_diagnostics.size());
  result.evaluated_board_observation_count =
      static_cast<int>(residual.board_observation_diagnostics.size());
  result.warnings.push_back(
      "training evaluation uses committed backend scene poses; "
      "pose_only_refit fields are not used for training because accepted "
      "training frames already have optimized pose variables.");
  result.failure_reason = residual.failure_reason;
  return result;
}

void WriteBackendComparisonSummary(
    const std::string& path,
    const ati::AslamBackendCalibrationResult& backend_result,
    const ati::CameraModelRefitEvaluationResult& frontend_training,
    const ati::CameraModelRefitEvaluationResult& backend_training,
    const ati::CameraModelRefitEvaluationResult& kalibr_training,
    const ati::CameraModelRefitEvaluationResult& frontend_holdout,
    const ati::CameraModelRefitEvaluationResult& backend_holdout,
    const ati::CameraModelRefitEvaluationResult& kalibr_holdout) {
  std::ofstream output(path.c_str());
  output << "backend_success: " << (backend_result.success ? 1 : 0) << "\n";
  output << "backend_failure_reason: " << backend_result.failure_reason << "\n";
  output << "backend_initial_overall_rmse: "
         << backend_result.initial_residual.overall_rmse << "\n";
  output << "backend_optimized_overall_rmse: "
         << backend_result.optimized_residual.overall_rmse << "\n";
  output << "backend_observation_role_weight_mode: "
         << backend_result.options.observation_role_weight_mode << "\n";
  output << "backend_internal_role_budget_when_mixed: "
         << backend_result.options.internal_role_budget_when_mixed << "\n";
  output << "training_frontend_overall_rmse: " << frontend_training.overall_rmse << "\n";
  output << "training_backend_overall_rmse: " << backend_training.overall_rmse << "\n";
  output << "training_kalibr_overall_rmse: " << kalibr_training.overall_rmse << "\n";
  output << "training_frontend_outer_only_rmse: " << frontend_training.outer_only_rmse << "\n";
  output << "training_backend_outer_only_rmse: " << backend_training.outer_only_rmse << "\n";
  output << "training_kalibr_outer_only_rmse: " << kalibr_training.outer_only_rmse << "\n";
  output << "training_frontend_internal_only_rmse: "
         << frontend_training.internal_only_rmse << "\n";
  output << "training_backend_internal_only_rmse: "
         << backend_training.internal_only_rmse << "\n";
  output << "training_kalibr_internal_only_rmse: "
         << kalibr_training.internal_only_rmse << "\n";
  output << "training_backend_overall_rmse_excluding_board"
         << backend_training.excluded_board_id_for_rmse << ": "
         << backend_training.overall_rmse_excluding_board << "\n";
  output << "training_kalibr_overall_rmse_excluding_board"
         << kalibr_training.excluded_board_id_for_rmse << ": "
         << kalibr_training.overall_rmse_excluding_board << "\n";
  output << "training_frontend_mean_residual_x: " << frontend_training.mean_residual_x << "\n";
  output << "training_frontend_mean_residual_y: " << frontend_training.mean_residual_y << "\n";
  output << "training_frontend_std_residual_x: " << frontend_training.std_residual_x << "\n";
  output << "training_frontend_std_residual_y: " << frontend_training.std_residual_y << "\n";
  output << "training_backend_mean_residual_x: " << backend_training.mean_residual_x << "\n";
  output << "training_backend_mean_residual_y: " << backend_training.mean_residual_y << "\n";
  output << "training_backend_std_residual_x: " << backend_training.std_residual_x << "\n";
  output << "training_backend_std_residual_y: " << backend_training.std_residual_y << "\n";
  output << "training_kalibr_mean_residual_x: " << kalibr_training.mean_residual_x << "\n";
  output << "training_kalibr_mean_residual_y: " << kalibr_training.mean_residual_y << "\n";
  output << "training_kalibr_std_residual_x: " << kalibr_training.std_residual_x << "\n";
  output << "training_kalibr_std_residual_y: " << kalibr_training.std_residual_y << "\n";
  output << "holdout_frontend_overall_rmse: " << frontend_holdout.overall_rmse << "\n";
  output << "holdout_backend_overall_rmse: " << backend_holdout.overall_rmse << "\n";
  output << "holdout_kalibr_overall_rmse: " << kalibr_holdout.overall_rmse << "\n";
  output << "holdout_backend_pose_only_refit_rmse: "
         << backend_holdout.pose_only_refit_rmse << "\n";
  output << "holdout_backend_pose_only_refit_success_rate: "
         << backend_holdout.pose_only_refit_success_rate << "\n";
  output << "holdout_kalibr_pose_only_refit_rmse: "
         << kalibr_holdout.pose_only_refit_rmse << "\n";
  output << "holdout_kalibr_pose_only_refit_success_rate: "
         << kalibr_holdout.pose_only_refit_success_rate << "\n";
  output << "holdout_frontend_outer_only_rmse: " << frontend_holdout.outer_only_rmse << "\n";
  output << "holdout_backend_outer_only_rmse: " << backend_holdout.outer_only_rmse << "\n";
  output << "holdout_kalibr_outer_only_rmse: " << kalibr_holdout.outer_only_rmse << "\n";
  output << "holdout_frontend_internal_only_rmse: "
         << frontend_holdout.internal_only_rmse << "\n";
  output << "holdout_backend_internal_only_rmse: "
         << backend_holdout.internal_only_rmse << "\n";
  output << "holdout_kalibr_internal_only_rmse: "
         << kalibr_holdout.internal_only_rmse << "\n";
  output << "holdout_backend_overall_rmse_excluding_board"
         << backend_holdout.excluded_board_id_for_rmse << ": "
         << backend_holdout.overall_rmse_excluding_board << "\n";
  output << "holdout_kalibr_overall_rmse_excluding_board"
         << kalibr_holdout.excluded_board_id_for_rmse << ": "
         << kalibr_holdout.overall_rmse_excluding_board << "\n";
  output << "holdout_frontend_mean_residual_x: " << frontend_holdout.mean_residual_x << "\n";
  output << "holdout_frontend_mean_residual_y: " << frontend_holdout.mean_residual_y << "\n";
  output << "holdout_frontend_std_residual_x: " << frontend_holdout.std_residual_x << "\n";
  output << "holdout_frontend_std_residual_y: " << frontend_holdout.std_residual_y << "\n";
  output << "holdout_backend_mean_residual_x: " << backend_holdout.mean_residual_x << "\n";
  output << "holdout_backend_mean_residual_y: " << backend_holdout.mean_residual_y << "\n";
  output << "holdout_backend_std_residual_x: " << backend_holdout.std_residual_x << "\n";
  output << "holdout_backend_std_residual_y: " << backend_holdout.std_residual_y << "\n";
  output << "holdout_kalibr_mean_residual_x: " << kalibr_holdout.mean_residual_x << "\n";
  output << "holdout_kalibr_mean_residual_y: " << kalibr_holdout.mean_residual_y << "\n";
  output << "holdout_kalibr_std_residual_x: " << kalibr_holdout.std_residual_x << "\n";
  output << "holdout_kalibr_std_residual_y: " << kalibr_holdout.std_residual_y << "\n";
  output << "training_frontend_minus_kalibr: "
         << (frontend_training.overall_rmse - kalibr_training.overall_rmse) << "\n";
  output << "training_backend_minus_kalibr: "
         << (backend_training.overall_rmse - kalibr_training.overall_rmse) << "\n";
  output << "holdout_frontend_minus_kalibr: "
         << (frontend_holdout.overall_rmse - kalibr_holdout.overall_rmse) << "\n";
  output << "holdout_backend_minus_kalibr: "
         << (backend_holdout.overall_rmse - kalibr_holdout.overall_rmse) << "\n";
  output << "frontend_camera_xi: " << frontend_holdout.camera.xi << "\n";
  output << "frontend_camera_alpha: " << frontend_holdout.camera.alpha << "\n";
  output << "frontend_camera_fu: " << frontend_holdout.camera.fu << "\n";
  output << "frontend_camera_fv: " << frontend_holdout.camera.fv << "\n";
  output << "frontend_camera_cu: " << frontend_holdout.camera.cu << "\n";
  output << "frontend_camera_cv: " << frontend_holdout.camera.cv << "\n";
  output << "backend_camera_xi: " << backend_holdout.camera.xi << "\n";
  output << "backend_camera_alpha: " << backend_holdout.camera.alpha << "\n";
  output << "backend_camera_fu: " << backend_holdout.camera.fu << "\n";
  output << "backend_camera_fv: " << backend_holdout.camera.fv << "\n";
  output << "backend_camera_cu: " << backend_holdout.camera.cu << "\n";
  output << "backend_camera_cv: " << backend_holdout.camera.cv << "\n";
  output << "kalibr_camera_xi: " << kalibr_holdout.camera.xi << "\n";
  output << "kalibr_camera_alpha: " << kalibr_holdout.camera.alpha << "\n";
  output << "kalibr_camera_fu: " << kalibr_holdout.camera.fu << "\n";
  output << "kalibr_camera_fv: " << kalibr_holdout.camera.fv << "\n";
  output << "kalibr_camera_cu: " << kalibr_holdout.camera.cu << "\n";
  output << "kalibr_camera_cv: " << kalibr_holdout.camera.cv << "\n";
  for (const std::string& warning : backend_result.warnings) {
    output << "backend_warning: " << warning << "\n";
  }
}

void WriteBackendVsKalibrSummary(
    const fs::path& output_dir,
    const ati::KalibrBenchmarkReference& kalibr_reference,
    const ati::CameraModelRefitEvaluationResult& backend_training,
    const ati::CameraModelRefitEvaluationResult& kalibr_training,
    const ati::CameraModelRefitEvaluationResult& backend_holdout,
    const ati::CameraModelRefitEvaluationResult& kalibr_holdout) {
  const auto write_eval = [](std::ostream& output,
                             const std::string& prefix,
                             const ati::CameraModelRefitEvaluationResult& eval) {
    output << prefix << "_success: " << (eval.success ? 1 : 0) << "\n";
    output << prefix << "_overall_rmse: " << eval.overall_rmse << "\n";
    output << prefix << "_outer_only_rmse: " << eval.outer_only_rmse << "\n";
    output << prefix << "_internal_only_rmse: " << eval.internal_only_rmse
           << "\n";
    output << prefix << "_pose_only_refit_rmse: "
           << eval.pose_only_refit_rmse << "\n";
    output << prefix << "_pose_only_refit_success_rate: "
           << eval.pose_only_refit_success_rate << "\n";
    output << prefix << "_point_count: " << eval.point_count << "\n";
    output << prefix << "_outer_point_count: " << eval.outer_point_count
           << "\n";
    output << prefix << "_internal_point_count: "
           << eval.internal_point_count << "\n";
    output << prefix << "_excluded_board_id_for_rmse: "
           << eval.excluded_board_id_for_rmse << "\n";
    output << prefix << "_overall_rmse_excluding_board"
           << eval.excluded_board_id_for_rmse << ": "
           << eval.overall_rmse_excluding_board << "\n";
    output << prefix << "_point_count_excluding_board"
           << eval.excluded_board_id_for_rmse << ": "
           << eval.point_count_excluding_board << "\n";
    output << prefix << "_outer_point_count_excluding_board"
           << eval.excluded_board_id_for_rmse << ": "
           << eval.outer_point_count_excluding_board << "\n";
    output << prefix << "_internal_point_count_excluding_board"
           << eval.excluded_board_id_for_rmse << ": "
           << eval.internal_point_count_excluding_board << "\n";
    output << prefix << "_mean_residual_x: " << eval.mean_residual_x << "\n";
    output << prefix << "_mean_residual_y: " << eval.mean_residual_y << "\n";
    output << prefix << "_std_residual_x: " << eval.std_residual_x << "\n";
    output << prefix << "_std_residual_y: " << eval.std_residual_y << "\n";
  };

  const auto write_delta = [](
      std::ostream& output,
      const std::string& prefix,
      const ati::CameraModelRefitEvaluationResult& backend,
      const ati::CameraModelRefitEvaluationResult& kalibr) {
    output << prefix << "_backend_minus_kalibr_overall_rmse: "
           << backend.overall_rmse - kalibr.overall_rmse << "\n";
    output << prefix << "_backend_minus_kalibr_outer_only_rmse: "
           << backend.outer_only_rmse - kalibr.outer_only_rmse << "\n";
    output << prefix << "_backend_minus_kalibr_internal_only_rmse: "
           << backend.internal_only_rmse - kalibr.internal_only_rmse << "\n";
    output << prefix << "_backend_minus_kalibr_pose_only_refit_rmse: "
           << backend.pose_only_refit_rmse - kalibr.pose_only_refit_rmse
           << "\n";
    output << prefix << "_backend_minus_kalibr_overall_rmse_excluding_board"
           << backend.excluded_board_id_for_rmse << ": "
           << backend.overall_rmse_excluding_board -
                  kalibr.overall_rmse_excluding_board
           << "\n";
  };

  std::ofstream summary(
      (output_dir / "backend_vs_kalibr_summary.txt").string().c_str());
  summary << "purpose: compare the final backend camera after all enabled "
          << "Stage5 backend selection/refinement steps against the fixed "
          << "Kalibr camchain reference on the same evaluation split.\n";
  summary << "note: backend_* fields are copied from "
          << "backend_training_summary.txt / backend_holdout_summary.txt, not "
          << "from the pre-backend frontend benchmark report.\n";
  summary << "kalibr_camchain_yaml: " << kalibr_reference.camchain_yaml
          << "\n";
  summary << "kalibr_source_label: " << kalibr_reference.source_label << "\n";
  summary << "\n[training]\n";
  write_eval(summary, "backend_training", backend_training);
  write_eval(summary, "kalibr_training", kalibr_training);
  write_delta(summary, "training", backend_training, kalibr_training);
  summary << "\n[holdout]\n";
  write_eval(summary, "backend_holdout", backend_holdout);
  write_eval(summary, "kalibr_holdout", kalibr_holdout);
  write_delta(summary, "holdout", backend_holdout, kalibr_holdout);

  std::ofstream csv(
      (output_dir / "backend_vs_kalibr_summary.csv").string().c_str());
  csv << "split,method,success,overall_rmse,outer_only_rmse,"
      << "internal_only_rmse,pose_only_refit_rmse,"
      << "pose_only_refit_success_rate,point_count,outer_point_count,"
      << "internal_point_count,excluded_board_id_for_rmse,"
      << "overall_rmse_excluding_board,point_count_excluding_board,"
      << "outer_point_count_excluding_board,"
      << "internal_point_count_excluding_board,"
      << "mean_residual_x,mean_residual_y,"
      << "std_residual_x,std_residual_y\n";
  const auto write_csv_eval = [&csv](
      const std::string& split,
      const std::string& method,
      const ati::CameraModelRefitEvaluationResult& eval) {
    csv << split << "," << method << ","
        << (eval.success ? 1 : 0) << ","
        << eval.overall_rmse << ","
        << eval.outer_only_rmse << ","
        << eval.internal_only_rmse << ","
        << eval.pose_only_refit_rmse << ","
        << eval.pose_only_refit_success_rate << ","
        << eval.point_count << ","
        << eval.outer_point_count << ","
        << eval.internal_point_count << ","
        << eval.excluded_board_id_for_rmse << ","
        << eval.overall_rmse_excluding_board << ","
        << eval.point_count_excluding_board << ","
        << eval.outer_point_count_excluding_board << ","
        << eval.internal_point_count_excluding_board << ","
        << eval.mean_residual_x << ","
        << eval.mean_residual_y << ","
        << eval.std_residual_x << ","
        << eval.std_residual_y << "\n";
  };
  write_csv_eval("training", "backend_final", backend_training);
  write_csv_eval("training", "kalibr_reference", kalibr_training);
  write_csv_eval("holdout", "backend_final", backend_holdout);
  write_csv_eval("holdout", "kalibr_reference", kalibr_holdout);
}

std::string CsvEscape(const std::string& value);

void WriteCloseEdgeOuterPoseDiagnostics(
    const fs::path& output_dir,
    const ati::CameraModelRefitEvaluationResult& frontend_holdout,
    const ati::CameraModelRefitEvaluationResult& backend_holdout,
    const ati::CameraModelRefitEvaluationResult& kalibr_holdout) {
  std::ofstream board_csv(
      (output_dir / "close_edge_outer_pose_board_diagnostics.csv")
          .string()
          .c_str());
  board_csv
      << "method,frame_index,frame_label,board_id,point_count,"
      << "outer_point_count,internal_point_count,pose_fit_outer_rmse,"
      << "evaluation_rmse,outer_evaluation_rmse,internal_evaluation_rmse,"
      << "observed_outer_area_px,mean_observed_radius_px,"
      << "max_observed_radius_px,mean_outer_residual_norm,"
      << "max_outer_residual_norm\n";

  std::ofstream point_csv(
      (output_dir / "close_edge_outer_pose_corner_residuals.csv")
          .string()
          .c_str());
  point_csv
      << "method,frame_index,frame_label,board_id,point_id,"
      << "observed_x,observed_y,predicted_x,predicted_y,"
      << "residual_x,residual_y,residual_norm,observed_radius_px,"
      << "quality\n";

  const auto write_eval =
      [&](const std::string& method,
          const ati::CameraModelRefitEvaluationResult& evaluation) {
        for (const ati::CameraModelRefitBoardObservationDiagnostics& board :
             evaluation.board_observation_diagnostics) {
          if (!board.pose_only_refit_success ||
              (board.board_id != 4 && board.board_id != 5)) {
            continue;
          }
          std::vector<Eigen::Vector2d> outer_observed;
          std::vector<double> outer_residuals;
          double radius_sum = 0.0;
          double radius_max = 0.0;
          for (const ati::CameraModelRefitPointDiagnostics& point :
               evaluation.point_diagnostics) {
            if (point.frame_index != board.frame_index ||
                point.board_id != board.board_id ||
                point.point_type != ati::JointPointType::Outer) {
              continue;
            }
            outer_observed.push_back(point.observed_image_xy);
            outer_residuals.push_back(point.residual_norm);
            const double radius =
                DistanceFromImageCenterPx(evaluation.camera,
                                          point.observed_image_xy);
            radius_sum += radius;
            radius_max = std::max(radius_max, radius);
            point_csv
                << method << ","
                << point.frame_index << ","
                << CsvEscape(point.frame_label) << ","
                << point.board_id << ","
                << point.point_id << ","
                << point.observed_image_xy.x() << ","
                << point.observed_image_xy.y() << ","
                << point.predicted_image_xy.x() << ","
                << point.predicted_image_xy.y() << ","
                << point.residual_xy.x() << ","
                << point.residual_xy.y() << ","
                << point.residual_norm << ","
                << radius << ","
                << point.quality << "\n";
          }
          const double mean_radius =
              outer_observed.empty()
                  ? 0.0
                  : radius_sum / static_cast<double>(outer_observed.size());
          const double mean_outer_residual =
              outer_residuals.empty()
                  ? 0.0
                  : std::accumulate(outer_residuals.begin(),
                                    outer_residuals.end(), 0.0) /
                        static_cast<double>(outer_residuals.size());
          const double max_outer_residual =
              outer_residuals.empty()
                  ? 0.0
                  : *std::max_element(outer_residuals.begin(),
                                      outer_residuals.end());
          board_csv
              << method << ","
              << board.frame_index << ","
              << CsvEscape(board.frame_label) << ","
              << board.board_id << ","
              << board.point_count << ","
              << board.outer_point_count << ","
              << board.internal_point_count << ","
              << board.pose_fit_outer_rmse << ","
              << board.evaluation_rmse << ","
              << board.outer_evaluation_rmse << ","
              << board.internal_evaluation_rmse << ","
              << PolygonAreaPx(outer_observed) << ","
              << mean_radius << ","
              << radius_max << ","
              << mean_outer_residual << ","
              << max_outer_residual << "\n";
        }
      };

  write_eval("frontend_initial", frontend_holdout);
  write_eval("backend_final", backend_holdout);
  write_eval("kalibr_reference", kalibr_holdout);

  std::ofstream summary(
      (output_dir / "close_edge_outer_pose_diagnostics_summary.txt")
          .string()
          .c_str());
  summary << "purpose: diagnose close-distance edge board4/board5 holdout "
          << "failures by decomposing outer-corner pose-only refit residuals.\n";
  summary << "methods: frontend_initial, backend_final, kalibr_reference\n";
  summary << "board_csv: "
          << (output_dir / "close_edge_outer_pose_board_diagnostics.csv")
                 .string()
          << "\n";
  summary << "corner_csv: "
          << (output_dir / "close_edge_outer_pose_corner_residuals.csv")
                 .string()
          << "\n";
  summary << "note: this is diagnostic-only and does not affect frontend, "
          << "selection, backend optimization, or holdout evaluation.\n";
}

std::string MatrixToCsv(const Eigen::Matrix4d& matrix) {
  std::ostringstream stream;
  stream << std::setprecision(12);
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      if (row != 0 || col != 0) {
        stream << ",";
      }
      stream << matrix(row, col);
    }
  }
  return stream.str();
}

std::string JoinStringVector(const std::vector<std::string>& values,
                             const std::string& separator) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0u) {
      stream << separator;
    }
    stream << values[index];
  }
  return stream.str();
}

std::string JoinDoubleVector(const std::vector<double>& values,
                             const std::string& separator) {
  std::ostringstream stream;
  stream << std::setprecision(12);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0u) {
      stream << separator;
    }
    stream << values[index];
  }
  return stream.str();
}

std::string ResidualPointKey(
    const ati::JointResidualPointDiagnostics& point) {
  std::ostringstream stream;
  stream << point.frame_index << "|" << point.board_id << "|"
         << point.point_id << "|" << static_cast<int>(point.point_type)
         << "|" << static_cast<int>(point.source_kind);
  return stream.str();
}

void WriteIntermediateCameraYaml(
    const fs::path& path,
    const ati::OuterBootstrapCameraIntrinsics& camera) {
  std::ofstream output(path.string().c_str());
  output << std::setprecision(12);
  output << "camera_model: " << camera.camera_model << "\n";
  output << "distortion_model: " << camera.distortion_model << "\n";
  output << "resolution: [" << camera.resolution.width << ", "
         << camera.resolution.height << "]\n";
  output << "intrinsics: [" << camera.xi << ", " << camera.alpha << ", "
         << camera.fu << ", " << camera.fv << ", " << camera.cu << ", "
         << camera.cv << "]\n";
  output << "xi: " << camera.xi << "\n";
  output << "alpha: " << camera.alpha << "\n";
  output << "fu: " << camera.fu << "\n";
  output << "fv: " << camera.fv << "\n";
  output << "cu: " << camera.cu << "\n";
  output << "cv: " << camera.cv << "\n";
}

void WriteLargePerturbationSceneSnapshot(
    const fs::path& path,
    const ati::CalibrationSceneState& scene) {
  std::ofstream output(path.string().c_str());
  output << std::setprecision(17);
  output << "camera " << scene.camera.xi << " " << scene.camera.alpha << " "
         << scene.camera.fu << " " << scene.camera.fv << " "
         << scene.camera.cu << " " << scene.camera.cv << "\n";
  const std::vector<double> distortion = scene.camera.DistortionVector();
  output << "distortion " << distortion.size();
  for (double coefficient : distortion) {
    output << " " << coefficient;
  }
  output << "\n";
  for (const ati::JointSceneFrameState& frame : scene.frames) {
    output << "frame " << frame.frame_index << " "
           << (frame.initialized ? 1 : 0) << " " << frame.observation_count
           << " " << frame.rmse;
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        output << " " << frame.T_camera_reference(row, col);
      }
    }
    output << "\n";
  }
  for (const ati::JointSceneBoardState& board : scene.boards) {
    output << "board " << board.board_id << " "
           << (board.initialized ? 1 : 0) << " " << board.observation_count
           << " " << board.rmse;
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        output << " " << board.T_reference_board(row, col);
      }
    }
    output << "\n";
  }
}

void WriteIntermediateResidualPointsCsv(
    const fs::path& path,
    const ati::OuterOnlyIntermediateCalibrationResult& intermediate) {
  std::ofstream output(path.string().c_str());
  output << "stage,frame_index,frame_label,board_id,point_id,point_type,"
            "observed_u,observed_v,predicted_u,predicted_v,residual_x,"
            "residual_y,residual_norm,polar_angle_deg\n";
  const auto write_stage =
      [&output](const std::string& stage,
                const ati::JointResidualEvaluationResult& residuals) {
        for (const ati::JointResidualPointDiagnostics& point :
             residuals.point_diagnostics) {
          output << stage << "," << point.frame_index << ","
                 << point.frame_label << "," << point.board_id << ","
                 << point.point_id << "," << ati::ToString(point.point_type)
                 << "," << point.observed_image_xy.x() << ","
                 << point.observed_image_xy.y() << ","
                 << point.predicted_image_xy.x() << ","
                 << point.predicted_image_xy.y() << ","
                 << point.residual_xy.x() << ","
                 << point.residual_xy.y() << ","
                 << point.residual_norm << "," << point.polar_angle_deg
                 << "\n";
        }
      };
  if (intermediate.initial_residual_result.success) {
    write_stage("bootstrap", intermediate.initial_residual_result);
  }
  if (intermediate.optimization_result.optimized_residual.success) {
    write_stage("intermediate",
                intermediate.optimization_result.optimized_residual);
  }
}

void WriteIntermediatePredictionDiagnosticsCsv(
    const fs::path& path,
    const ati::OuterOnlyIntermediateCalibrationResult& intermediate) {
  std::ofstream output(path.string().c_str());
  output << "frame_index,frame_label,board_id,point_id,point_type,"
            "observed_u,observed_v,bootstrap_predicted_u,"
            "bootstrap_predicted_v,intermediate_predicted_u,"
            "intermediate_predicted_v,bootstrap_residual_norm,"
            "intermediate_residual_norm,mean_shift_px,max_shift_px,"
            "polar_angle_mean_deg,projected_scale,confidence_change,"
            "recommend_use_intermediate,observation_type\n";

  std::map<std::string, const ati::JointResidualPointDiagnostics*> optimized_by_key;
  if (intermediate.optimization_result.optimized_residual.success) {
    for (const ati::JointResidualPointDiagnostics& point :
         intermediate.optimization_result.optimized_residual.point_diagnostics) {
      optimized_by_key[ResidualPointKey(point)] = &point;
    }
  }

  for (const ati::JointResidualPointDiagnostics& initial_point :
       intermediate.initial_residual_result.point_diagnostics) {
    const auto optimized_it = optimized_by_key.find(ResidualPointKey(initial_point));
    if (optimized_it == optimized_by_key.end()) {
      continue;
    }
    const ati::JointResidualPointDiagnostics& optimized_point =
        *optimized_it->second;
    const double shift =
        (optimized_point.predicted_image_xy -
         initial_point.predicted_image_xy).norm();
    const bool better =
        optimized_point.residual_norm <= initial_point.residual_norm;
    output << initial_point.frame_index << "," << initial_point.frame_label
           << "," << initial_point.board_id << "," << initial_point.point_id
           << "," << ati::ToString(initial_point.point_type) << ","
           << initial_point.observed_image_xy.x() << ","
           << initial_point.observed_image_xy.y() << ","
           << initial_point.predicted_image_xy.x() << ","
           << initial_point.predicted_image_xy.y() << ","
           << optimized_point.predicted_image_xy.x() << ","
           << optimized_point.predicted_image_xy.y() << ","
           << initial_point.residual_norm << ","
           << optimized_point.residual_norm << "," << shift << ","
           << shift << "," << optimized_point.polar_angle_deg << ","
           << "0,0," << (better ? 1 : 0) << ",normal_detected\n";
  }
}

void WriteOuterOnlyIntermediateArtifacts(
    const fs::path& output_dir,
    const ati::FrozenRound2BaselineResult& baseline_result) {
  const ati::OuterOnlyIntermediateCalibrationResult& intermediate =
      baseline_result.outer_only_intermediate;
  if (!intermediate.enabled) {
    return;
  }

  const fs::path intermediate_dir = output_dir / "stage5_intermediate_model";
  EnsureDirectoryExists(intermediate_dir);

  {
    std::ofstream output(
        (intermediate_dir / "intermediate_calibration_summary.txt").string().c_str());
    output << std::setprecision(12);
    output << "enabled: " << (intermediate.enabled ? 1 : 0) << "\n";
    output << "diagnostic_only: " << (intermediate.diagnostic_only ? 1 : 0)
           << "\n";
    output << "use_for_round1_requested: "
           << (intermediate.use_for_round1_requested ? 1 : 0) << "\n";
    output << "use_for_full_frontend_regeneration_requested: "
           << (intermediate.use_for_full_frontend_regeneration_requested ? 1 : 0)
           << "\n";
    output << "used_for_round1_internal_regeneration: "
           << (intermediate.used_for_round1_internal_regeneration ? 1 : 0)
           << "\n";
    output << "used_for_full_frontend_regeneration: "
           << (intermediate.used_for_full_frontend_regeneration ? 1 : 0)
           << "\n";
    output << "success: " << (intermediate.success ? 1 : 0) << "\n";
    output << "state_source_label: " << intermediate.state_source_label << "\n";
    output << "failure_reason: " << intermediate.failure_reason << "\n";
    output << "max_outer_rmse_px: " << intermediate.max_outer_rmse_px << "\n";
    output << "min_visible_boards: " << intermediate.min_visible_boards << "\n";
    output << "total_outer_board_observation_count: "
           << intermediate.total_outer_board_observation_count << "\n";
    output << "used_outer_board_observation_count: "
           << intermediate.used_outer_board_observation_count << "\n";
    output << "rejected_outer_board_observation_count: "
           << intermediate.rejected_outer_board_observation_count << "\n";
    output << "used_outer_point_count: "
           << intermediate.used_outer_point_count << "\n";
    output << "used_internal_point_count: "
           << intermediate.used_internal_point_count << "\n";
    output << "outer_rmse_before: "
           << intermediate.initial_residual_result.outer_only_rmse << "\n";
    output << "overall_rmse_before: "
           << intermediate.initial_residual_result.overall_rmse << "\n";
    output << "outer_rmse_after: "
           << intermediate.optimization_result.optimized_residual.outer_only_rmse
           << "\n";
    output << "overall_rmse_after: "
           << intermediate.optimization_result.optimized_residual.overall_rmse
           << "\n";
    output << "initial_camera_xi: "
           << intermediate.optimization_result.initial_state.camera.xi << "\n";
    output << "initial_camera_alpha: "
           << intermediate.optimization_result.initial_state.camera.alpha << "\n";
    output << "initial_camera_fu: "
           << intermediate.optimization_result.initial_state.camera.fu << "\n";
    output << "initial_camera_fv: "
           << intermediate.optimization_result.initial_state.camera.fv << "\n";
    output << "initial_camera_cu: "
           << intermediate.optimization_result.initial_state.camera.cu << "\n";
    output << "initial_camera_cv: "
           << intermediate.optimization_result.initial_state.camera.cv << "\n";
    output << "intermediate_camera_xi: "
           << intermediate.optimization_result.optimized_state.camera.xi << "\n";
    output << "intermediate_camera_alpha: "
           << intermediate.optimization_result.optimized_state.camera.alpha
           << "\n";
    output << "intermediate_camera_fu: "
           << intermediate.optimization_result.optimized_state.camera.fu << "\n";
    output << "intermediate_camera_fv: "
           << intermediate.optimization_result.optimized_state.camera.fv << "\n";
    output << "intermediate_camera_cu: "
           << intermediate.optimization_result.optimized_state.camera.cu << "\n";
    output << "intermediate_camera_cv: "
           << intermediate.optimization_result.optimized_state.camera.cv << "\n";
    output << "optimizer_iteration_count: "
           << intermediate.optimization_result.iterations.size() << "\n";
    for (const std::string& warning : intermediate.warnings) {
      output << "warning: " << warning << "\n";
    }
  }

  if (intermediate.optimization_result.optimized_state.camera.IsValid()) {
    WriteIntermediateCameraYaml(
        intermediate_dir / "intermediate_camera.yaml",
        intermediate.optimization_result.optimized_state.camera);
  }

  {
    std::ofstream output(
        (intermediate_dir / "intermediate_board_poses.csv").string().c_str());
    output << "board_id,initialized,observation_count,rmse,T_reference_board_16\n";
    for (const ati::JointSceneBoardState& board :
         intermediate.optimization_result.optimized_state.boards) {
      output << board.board_id << "," << (board.initialized ? 1 : 0)
             << "," << board.observation_count << "," << board.rmse << ","
             << MatrixToCsv(board.T_reference_board) << "\n";
    }
  }

  {
    std::ofstream output(
        (intermediate_dir / "intermediate_frame_poses.csv").string().c_str());
    output << "frame_index,frame_label,initialized,observation_count,rmse,"
              "T_camera_reference_16\n";
    for (const ati::JointSceneFrameState& frame :
         intermediate.optimization_result.optimized_state.frames) {
      output << frame.frame_index << "," << frame.frame_label << ","
             << (frame.initialized ? 1 : 0) << ","
             << frame.observation_count << "," << frame.rmse << ","
             << MatrixToCsv(frame.T_camera_reference) << "\n";
    }
  }

  {
    std::ofstream output(
        (intermediate_dir / "intermediate_pose_delta.csv").string().c_str());
    output << "type,id,label,delta_norm\n";
    for (const ati::JointSceneBoardState& board :
         intermediate.optimization_result.optimized_state.boards) {
      const ati::JointSceneBoardState* initial_board =
          ati::FindJointSceneBoardState(
              intermediate.optimization_result.initial_state, board.board_id);
      if (initial_board == nullptr) {
        continue;
      }
      output << "board," << board.board_id << ",,"
             << ati::ComputePoseDeltaNorm(
                    ati::ToIsometry3d(initial_board->T_reference_board),
                    ati::ToIsometry3d(board.T_reference_board))
             << "\n";
    }
    for (const ati::JointSceneFrameState& frame :
         intermediate.optimization_result.optimized_state.frames) {
      const ati::JointSceneFrameState* initial_frame =
          ati::FindJointSceneFrameState(
              intermediate.optimization_result.initial_state, frame.frame_index);
      if (initial_frame == nullptr) {
        continue;
      }
      output << "frame," << frame.frame_index << "," << frame.frame_label
             << ","
             << ati::ComputePoseDeltaNorm(
                    ati::ToIsometry3d(initial_frame->T_camera_reference),
                    ati::ToIsometry3d(frame.T_camera_reference))
             << "\n";
    }
  }

  WriteIntermediateResidualPointsCsv(
      intermediate_dir / "intermediate_outer_residuals.csv", intermediate);
  WriteIntermediatePredictionDiagnosticsCsv(
      intermediate_dir / "intermediate_prediction_diagnostics.csv",
      intermediate);
}

void WriteCameraAwareOuterRescueArtifacts(
    const fs::path& output_dir,
    const ati::FrozenRound2BaselineResult& baseline_result) {
  const ati::CameraAwareOuterRescueSummary& rescue =
      baseline_result.camera_aware_outer_rescue;
  if (!rescue.requested) {
    return;
  }

  {
    std::ofstream output(
        (output_dir / "camera_aware_outer_rescue_summary.txt").string().c_str());
    output << std::setprecision(12);
    output << "stage5_camera_aware_outer_rescue_requested: "
           << (rescue.requested ? 1 : 0) << "\n";
    output << "stage5_camera_aware_outer_rescue_enabled: "
           << (rescue.enabled ? 1 : 0) << "\n";
    output << "stage5_camera_aware_outer_rescue_camera_family_supported: "
           << (rescue.camera_family_supported ? 1 : 0) << "\n";
    output << "stage5_camera_aware_outer_rescue_camera_source: "
           << rescue.camera_source << "\n";
    output << "stage5_camera_aware_outer_rescue_uses_yaml_intrinsics: "
           << rescue.uses_yaml_intrinsics << "\n";
    output << "stage5_camera_aware_outer_rescue_uses_kalibr_camchain_intrinsics: "
           << rescue.uses_kalibr_camchain_intrinsics << "\n";
    output << "stage5_camera_aware_outer_rescue_patch_plan: "
           << rescue.patch_plan << "\n";
    output << "stage5_camera_aware_outer_rescue_patch_size: "
           << rescue.patch_size << "\n";
    output << "stage5_camera_aware_outer_rescue_max_hamming: "
           << rescue.max_hamming << "\n";
    output << "stage5_camera_aware_outer_rescue_frame_count: "
           << rescue.frame_count << "\n";
    output << "stage5_camera_aware_outer_rescue_requested_board_observation_count: "
           << rescue.requested_board_observation_count << "\n";
    output << "stage5_camera_aware_outer_rescue_baseline_success_count: "
           << rescue.baseline_success_count << "\n";
    output << "stage5_camera_aware_outer_rescue_baseline_all_boards_frame_count: "
           << rescue.baseline_all_boards_frame_count << "\n";
    output << "stage5_camera_aware_outer_rescue_attempted_frame_count: "
           << rescue.attempted_frame_count << "\n";
    output << "stage5_camera_aware_outer_rescue_attempted_board_observation_count: "
           << rescue.attempted_board_observation_count << "\n";
    output << "stage5_camera_aware_outer_rescue_zero_detection_atlas_enabled: "
           << (rescue.zero_detection_atlas_enabled ? 1 : 0) << "\n";
    output << "stage5_camera_aware_outer_rescue_zero_detection_frame_count: "
           << rescue.zero_detection_frame_count << "\n";
    output << "stage5_camera_aware_outer_rescue_zero_detection_atlas_attempted_board_observation_count: "
           << rescue.zero_detection_atlas_attempted_board_observation_count
           << "\n";
    output << "stage5_camera_aware_outer_rescue_worker_count: "
           << rescue.worker_count << "\n";
    output << "stage5_camera_aware_outer_rescue_direct_layout_geometry_gate_enabled: "
           << (rescue.direct_layout_geometry_gate_enabled ? 1 : 0) << "\n";
    output << "stage5_camera_aware_outer_rescue_direct_layout_geometry_gate_available: "
           << (rescue.direct_layout_geometry_gate_available ? 1 : 0) << "\n";
    output << "stage5_camera_aware_outer_rescue_direct_layout_geometry_gate_max_rmse_px: "
           << rescue.direct_layout_geometry_gate_max_rmse_px << "\n";
    output << "stage5_camera_aware_outer_rescue_direct_layout_geometry_gate_evaluated_count: "
           << rescue.direct_layout_geometry_gate_evaluated_count << "\n";
    output << "stage5_camera_aware_outer_rescue_direct_layout_geometry_gate_accepted_count: "
           << rescue.direct_layout_geometry_gate_accepted_count << "\n";
    output << "stage5_camera_aware_outer_rescue_direct_layout_geometry_gate_rejected_count: "
           << rescue.direct_layout_geometry_gate_rejected_count << "\n";
    output << "stage5_camera_aware_outer_rescue_direct_layout_geometry_gate_not_evaluable_count: "
           << rescue.direct_layout_geometry_gate_not_evaluable_count << "\n";
    output << "stage5_camera_aware_outer_rescue_rescued_board_observation_count: "
           << rescue.rescued_board_observation_count << "\n";
    output << "stage5_camera_aware_outer_rescue_final_success_count: "
           << rescue.final_success_count << "\n";
    output << "stage5_camera_aware_outer_rescue_final_all_boards_frame_count: "
           << rescue.final_all_boards_frame_count << "\n";
    output << "stage5_camera_aware_outer_rescue_initialization_rerun: "
           << (rescue.camera_initialization_rerun ? 1 : 0) << "\n";
    output << "stage5_camera_aware_outer_rescue_initialization_rerun_success: "
           << (rescue.camera_initialization_rerun_success ? 1 : 0) << "\n";
    output << "stage5_camera_aware_outer_rescue_runtime_seconds: "
           << rescue.runtime_seconds << "\n";
    output << "stage5_camera_aware_outer_rescue_skip_reason: "
           << rescue.skip_reason << "\n";
    output << "provisional_camera_xi: " << rescue.provisional_camera.xi << "\n";
    output << "provisional_camera_alpha: " << rescue.provisional_camera.alpha << "\n";
    output << "provisional_camera_fu: " << rescue.provisional_camera.fu << "\n";
    output << "provisional_camera_fv: " << rescue.provisional_camera.fv << "\n";
    output << "provisional_camera_cu: " << rescue.provisional_camera.cu << "\n";
    output << "provisional_camera_cv: " << rescue.provisional_camera.cv << "\n";
    output << "final_initialization_camera_xi: "
           << rescue.final_initialization_camera.xi << "\n";
    output << "final_initialization_camera_alpha: "
           << rescue.final_initialization_camera.alpha << "\n";
    output << "final_initialization_camera_fu: "
           << rescue.final_initialization_camera.fu << "\n";
    output << "final_initialization_camera_fv: "
           << rescue.final_initialization_camera.fv << "\n";
    output << "final_initialization_camera_cu: "
           << rescue.final_initialization_camera.cu << "\n";
    output << "final_initialization_camera_cv: "
           << rescue.final_initialization_camera.cv << "\n";
  }

  {
    std::ofstream output(
        (output_dir / "camera_aware_outer_rescue_records.csv").string().c_str());
    output << "frame_index,frame_label,board_id,baseline_failure_reason,"
              "hamming,rescue_summary,corner0_x,corner0_y,corner1_x,corner1_y,"
              "corner2_x,corner2_y,corner3_x,corner3_y\n";
    for (const ati::CameraAwareOuterRescueRecord& record : rescue.records) {
      output << record.frame_index << "," << CsvEscape(record.frame_label)
             << "," << record.board_id << ","
             << CsvEscape(record.baseline_failure_reason) << ","
             << record.hamming << "," << CsvEscape(record.rescue_summary);
      for (const Eigen::Vector2d& corner : record.committed_corners) {
        output << "," << corner.x() << "," << corner.y();
      }
      output << "\n";
    }
  }

  if (rescue.records.empty()) {
    return;
  }
  const fs::path visualization_dir =
      output_dir / "camera_aware_outer_rescue_visualizations";
  EnsureDirectoryExists(visualization_dir);
  std::map<int, std::string> image_path_by_frame;
  for (const ati::FrozenRound2BaselineFrameSource& frame :
       baseline_result.frame_sources) {
    image_path_by_frame[frame.frame_index] = frame.image_path;
  }
  for (const ati::CameraAwareOuterRescueRecord& record : rescue.records) {
    const auto path_it = image_path_by_frame.find(record.frame_index);
    if (path_it == image_path_by_frame.end()) {
      continue;
    }
    cv::Mat image = cv::imread(path_it->second, cv::IMREAD_COLOR);
    if (image.empty()) {
      continue;
    }
    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      const Eigen::Vector2d& first =
          record.committed_corners[static_cast<std::size_t>(corner_index)];
      const Eigen::Vector2d& second = record.committed_corners[
          static_cast<std::size_t>((corner_index + 1) % 4)];
      cv::line(image, cv::Point2d(first.x(), first.y()),
               cv::Point2d(second.x(), second.y()),
               cv::Scalar(255, 220, 70), 5, cv::LINE_AA);
      cv::circle(image, cv::Point2d(first.x(), first.y()), 8,
                 cv::Scalar(255, 220, 70), -1, cv::LINE_AA);
    }
    const std::string title =
        "camera-aware rescue: frame=" + record.frame_label +
        " board=" + std::to_string(record.board_id) +
        " hamming=" + std::to_string(record.hamming);
    cv::putText(image, title, cv::Point(24, 48), cv::FONT_HERSHEY_SIMPLEX,
                0.9, cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
    cv::putText(image, title, cv::Point(24, 48), cv::FONT_HERSHEY_SIMPLEX,
                0.9, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    const std::string filename = record.frame_label + "_board" +
                                 std::to_string(record.board_id) + ".jpg";
    cv::imwrite((visualization_dir / filename).string(), image,
                std::vector<int>{cv::IMWRITE_JPEG_QUALITY, 92});
  }
}

void WriteExperimentConfigSummary(
    const std::string& path,
    const RequestedExperimentConfig& requested,
    const ati::Stage5BenchmarkReport& report,
    const ati::AslamBackendCalibrationResult* backend_result) {
  std::ofstream output(path.c_str());
  output << "frozen_baseline_label: " << requested.frozen_baseline_label << "\n";
  output << "experiment_tag: " << requested.experiment_tag << "\n";
  output << "effective_protocol_label: " << requested.effective_protocol_label << "\n";

  output << "requested_models: " << requested.models << "\n";
  output << "requested_stage5_init_refine_mode: "
         << ati::ToString(requested.init_refine_mode) << "\n";
  output << "requested_stage5_init_selection_scorer: "
         << ati::ToString(requested.init_selection_scorer) << "\n";
  output << "requested_camera_init_mode: "
         << ati::ToString(requested.camera_init_mode) << "\n";
  output << "requested_stage5_camera_aware_outer_rescue: "
         << (requested.camera_aware_outer_rescue ? 1 : 0) << "\n";
  output << "requested_stage5_camera_aware_outer_rescue_zero_detection_frames: "
         << (requested.camera_aware_outer_rescue_zero_detection_frames ? 1 : 0)
         << "\n";
  output << "requested_frozen_recovery_baseline_preset: "
         << (requested.frozen_recovery_baseline_preset ? 1 : 0) << "\n";
  output << "requested_geometry_guided_tag_likelihood_single_anchor: "
         << (requested.geometry_guided_tag_likelihood_allow_single_anchor ? 1 : 0)
         << "\n";
  output << "requested_outer_only_ablation_mode: "
         << (requested.outer_only_ablation_mode ? 1 : 0) << "\n";
  output << "requested_strict_outer_only_ablation: "
         << (!requested.include_internal_points ? 1 : 0) << "\n";
  output << "requested_internal_regeneration_skipped_for_outer_only: "
         << (requested.outer_only_ablation_mode ? 1 : 0) << "\n";
  output << "requested_ablation_mode: "
         << (requested.outer_only_ablation_mode ? "outer_only" : "with_internal")
         << "\n";
  output << "requested_include_internal_points: "
         << (requested.include_internal_points ? 1 : 0) << "\n";
  output << "requested_run_second_pass: " << (requested.run_second_pass ? 1 : 0) << "\n";
  output << "requested_frontend_optimize_intrinsics: "
         << (requested.frontend_optimize_intrinsics ? 1 : 0) << "\n";
  output << "requested_frontend_intrinsics_release_mode: "
         << ToString(requested.frontend_intrinsics_release_mode) << "\n";
  output << "requested_frontend_intrinsics_release_iteration: "
         << requested.frontend_intrinsics_release_iteration << "\n";
  output << "requested_frontend_second_pass_intrinsics_release_iteration: "
         << requested.frontend_second_pass_intrinsics_release_iteration << "\n";
  output << "requested_backend_optimize_intrinsics: "
         << (requested.backend_optimize_intrinsics ? 1 : 0) << "\n";
  output << "requested_backend_optimize_board_poses: "
         << (requested.backend_optimize_board_poses ? 1 : 0) << "\n";
  output << "requested_backend_board_pose_parameterization: "
         << requested.backend_board_pose_parameterization << "\n";
  output << "requested_backend_board_pose_prior: "
         << (requested.backend_board_pose_prior ? 1 : 0) << "\n";
  output << "requested_backend_board_pose_prior_translation_sigma_mm: "
         << requested.backend_board_pose_prior_translation_sigma_mm << "\n";
  output << "requested_backend_board_pose_prior_rotation_sigma_deg: "
         << requested.backend_board_pose_prior_rotation_sigma_deg << "\n";
  output << "requested_backend_point_budget_control: "
         << (requested.backend_point_budget_control ? 1 : 0) << "\n";
  output << "requested_backend_point_budget_control_total_points: "
         << requested.backend_point_budget_control_total_points << "\n";
  output << "requested_backend_point_budget_control_seed: "
         << requested.backend_point_budget_control_seed << "\n";
  output << "requested_backend_max_boards_per_frame_for_ablation: "
         << requested.backend_max_boards_per_frame_for_ablation << "\n";
  output << "requested_backend_delayed_intrinsics_release: "
         << (requested.backend_delayed_intrinsics_release ? 1 : 0) << "\n";
  output << "requested_backend_intrinsics_release_mode: "
         << ToString(requested.backend_intrinsics_release_mode) << "\n";
  output << "requested_backend_intrinsics_release_iteration: "
         << requested.backend_intrinsics_release_iteration << "\n";
  output << "requested_outer_subpix_scale: "
         << report.baseline_result.effective_options
                .config.outer_detector_config.outer_subpix_scale
         << "\n";
  output << "requested_outer_subpix_window_radius: "
         << report.baseline_result.effective_options
                .config.outer_detector_config.outer_subpix_window_radius
         << "\n";
  output << "requested_outer_subpix_window_scale: "
         << report.baseline_result.effective_options
                .config.outer_detector_config.outer_subpix_window_scale
         << "\n";
  output << "requested_outer_subpix_window_min: "
         << report.baseline_result.effective_options
                .config.outer_detector_config.outer_subpix_window_min
         << "\n";
  output << "requested_outer_subpix_window_max: "
         << report.baseline_result.effective_options
                .config.outer_detector_config.outer_subpix_window_max
         << "\n";
  output << "requested_enable_close_edge_outer_subpix_boost: "
         << (report.baseline_result.effective_options.config
                     .outer_detector_config.enable_close_edge_outer_subpix_boost
                 ? 1
                 : 0)
         << "\n";
  output << "requested_close_edge_outer_subpix_multiplier: "
         << report.baseline_result.effective_options
                .config.outer_detector_config.close_edge_outer_subpix_multiplier
         << "\n";
  output << "requested_close_edge_outer_subpix_max_multiplier: "
         << report.baseline_result.effective_options
                .config.outer_detector_config.close_edge_outer_subpix_max_multiplier
         << "\n";
  output << "requested_close_edge_outer_subpix_full_polar_deg: "
         << report.baseline_result.effective_options
                .config.outer_detector_config.close_edge_outer_subpix_full_polar_deg
         << "\n";
  output << "requested_enable_residual_sanity_gate: "
         << (requested.enable_residual_sanity_gate ? 1 : 0) << "\n";
  output << "requested_enable_board_pose_fit_gate: "
         << (requested.enable_board_pose_fit_gate ? 1 : 0) << "\n";
  output << "requested_stage5_selection_mode: "
         << requested.selection_mode << "\n";
  output << "requested_stage5_selection_residual_sanity_factor: "
         << requested.selection_residual_sanity_factor << "\n";
  output << "requested_stage5_selection_max_board_observation_rmse: "
         << requested.selection_max_board_observation_rmse << "\n";
  output << "requested_stage5_selection_kalibr_style_outlier_sigma: "
         << requested.selection_kalibr_style_outlier_sigma << "\n";
  output << "requested_stage5_selection_kalibr_style_min_abs_threshold_px: "
         << requested.selection_kalibr_style_min_abs_threshold_px << "\n";
  output << "requested_stage5_selection_kalibr_style_min_views_before_filter: "
         << requested.selection_kalibr_style_min_views_before_filter << "\n";
  output << "requested_stage5_enable_trial_backend_frame_board_selection: "
         << (requested.enable_trial_backend_frame_board_selection ? 1 : 0)
         << "\n";
  output << "requested_stage5_trial_backend_selection_mode: "
         << requested.trial_backend_selection_mode << "\n";
  output << "requested_stage5_trial_backend_selection_budget_mode: "
         << (requested.trial_backend_selection_budget_mode_set
                 ? requested.trial_backend_selection_budget_mode
                 : "auto")
         << "\n";
  output << "requested_stage5_trial_backend_selection_candidate_order: "
         << (requested.trial_backend_selection_candidate_order_set
                 ? requested.trial_backend_selection_candidate_order
                 : "auto")
         << "\n";
  output << "requested_stage5_trial_backend_selection_info_gain_proxy_mode: "
         << requested.trial_backend_selection_info_gain_proxy_mode << "\n";
  output << "requested_stage5_trial_backend_selection_batch_granularity: "
         << requested.trial_backend_selection_batch_granularity << "\n";
  output << "requested_stage5_trial_backend_selection_acceptance_policy: "
         << requested.trial_backend_selection_acceptance_policy << "\n";
  output << "requested_stage5_trial_backend_selection_mi_tol: "
         << requested.trial_backend_selection_mi_tol << "\n";
  output << "requested_stage5_trial_backend_selection_mi_tol_explicit: "
         << (requested.trial_backend_selection_mi_tol_set ? 1 : 0) << "\n";
  output << "requested_stage5_trial_backend_selection_rank_threshold: "
         << requested.trial_backend_selection_rank_threshold << "\n";
  output << "requested_stage5_checkerboard_huber_delta_pixels: "
         << requested.checkerboard_huber_delta_pixels << "\n";
  output << "requested_stage5_checkerboard_outlier_filter_enabled: "
         << (requested.checkerboard_outlier_filter_enabled ? 1 : 0) << "\n";
  output << "requested_stage5_checkerboard_outlier_sigma: "
         << requested.checkerboard_outlier_sigma << "\n";
  output << "requested_stage5_checkerboard_min_inlier_ratio: "
         << requested.checkerboard_min_inlier_ratio << "\n";
  output << "requested_stage5_checkerboard_min_retained_points: "
         << requested.checkerboard_min_retained_points << "\n";
  output << "requested_stage5_trial_backend_selection_candidate_shuffle_seed_set: "
         << (requested.trial_backend_selection_candidate_shuffle_seed_set ? 1
                                                                          : 0)
         << "\n";
  output << "requested_stage5_trial_backend_selection_candidate_shuffle_seed: "
         << requested.trial_backend_selection_candidate_shuffle_seed << "\n";
  output << "requested_stage5_trial_backend_selection_incremental: "
         << (requested.trial_backend_selection_incremental ? 1 : 0)
         << "\n";
  output << "requested_stage5_trial_backend_selection_carry_accepted_trial_state: "
         << (requested.trial_backend_selection_carry_accepted_trial_state ? 1
                                                                          : 0)
         << "\n";
  output << "requested_stage5_trial_backend_selection_optimize_intrinsics: "
         << (requested.trial_backend_selection_optimize_intrinsics ? 1 : 0)
         << "\n";
  output << "requested_stage5_trial_backend_selection_delayed_intrinsics_release: "
         << (requested.trial_backend_selection_delayed_intrinsics_release ? 1 : 0)
         << "\n";
  output << "requested_stage5_trial_backend_selection_intrinsics_release_iteration: "
         << requested.trial_backend_selection_intrinsics_release_iteration
         << "\n";
  output << "requested_stage5_trial_backend_selection_persistent_intrinsics_anchor_prior: "
         << (requested.trial_backend_selection_persistent_intrinsics_anchor_prior
                 ? 1
                 : 0)
         << "\n";
  output << "requested_stage5_trial_backend_selection_persistent_fix_board_layout: "
         << (requested.trial_backend_selection_persistent_fix_board_layout ? 1
                                                                           : 0)
         << "\n";
  output << "requested_stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_xi_alpha: "
         << requested
                .trial_backend_selection_persistent_intrinsics_anchor_weight_xi_alpha
         << "\n";
  output << "requested_stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_focal: "
         << requested
                .trial_backend_selection_persistent_intrinsics_anchor_weight_focal
         << "\n";
  output << "requested_stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_principal: "
         << requested
                .trial_backend_selection_persistent_intrinsics_anchor_weight_principal
         << "\n";
  output << "requested_stage5_trial_backend_selection_persistent_max_focal_relative_step: "
         << requested.trial_backend_selection_persistent_max_focal_relative_step
         << "\n";
  output << "requested_stage5_trial_backend_selection_persistent_max_principal_step_px: "
         << requested.trial_backend_selection_persistent_max_principal_step_px
         << "\n";
  output << "requested_stage5_trial_backend_selection_persistent_max_xi_alpha_step: "
         << requested.trial_backend_selection_persistent_max_xi_alpha_step
         << "\n";
  output << "requested_stage5_trial_backend_selection_max_iterations: "
         << requested.trial_backend_selection_max_iterations << "\n";
  output << "requested_stage5_trial_backend_selection_max_candidate_additions: "
         << requested.trial_backend_selection_max_candidate_additions << "\n";
  output << "requested_stage5_trial_backend_selection_adaptive_budget_ratio: "
         << requested.trial_backend_selection_adaptive_budget_ratio << "\n";
  output << "requested_stage5_trial_backend_selection_adaptive_budget_min: "
         << requested.trial_backend_selection_adaptive_budget_min << "\n";
  output << "requested_stage5_trial_backend_selection_adaptive_budget_max: "
         << requested.trial_backend_selection_adaptive_budget_max << "\n";
  output << "requested_stage5_trial_backend_selection_runtime_safety_ceiling: "
         << requested.trial_backend_selection_runtime_safety_ceiling << "\n";
  output << "requested_stage5_export_selected_case_visualizations: "
         << (requested.export_selected_case_visualizations ? 1 : 0) << "\n";
  output << "requested_stage5_trial_backend_selection_outlier_sigma: "
         << requested.trial_backend_selection_outlier_sigma << "\n";
  output << "requested_stage5_trial_backend_selection_min_abs_threshold_px: "
         << requested.trial_backend_selection_min_abs_threshold_px << "\n";
  output << "requested_stage5_trial_backend_selection_max_threshold_px: "
         << requested.trial_backend_selection_max_threshold_px << "\n";
  output << "requested_stage5_trial_backend_selection_accept_max_global_rmse_increase_px: "
         << requested.trial_backend_selection_accept_max_global_rmse_increase_px
         << "\n";
  output << "requested_stage5_trial_backend_selection_accept_max_outer_rmse_increase_px: "
         << requested.trial_backend_selection_accept_max_outer_rmse_increase_px
         << "\n";
  output << "requested_stage5_trial_backend_selection_accept_max_internal_rmse_increase_px: "
         << requested.trial_backend_selection_accept_max_internal_rmse_increase_px
         << "\n";
  output << "requested_stage5_trial_backend_selection_min_candidate_score: "
         << requested.trial_backend_selection_min_candidate_score << "\n";
  output << "requested_stage5_trial_backend_selection_min_coverage_gain: "
         << requested.trial_backend_selection_min_coverage_gain << "\n";
  output << "requested_stage5_trial_backend_selection_force_include_frame_board_list: "
         << requested.trial_backend_selection_force_include_frame_board_list
         << "\n";
  output << "requested_stage5_trial_backend_selection_seed_frame_board_list: "
         << requested.trial_backend_selection_seed_frame_board_list << "\n";
  output << "requested_stage5_trial_backend_selection_force_include_list_is_exact_input: "
         << (requested.trial_backend_selection_force_include_list_is_exact_input
                 ? 1
                 : 0)
         << "\n";
  output << "requested_stage5_trial_backend_selection_use_consistency_score: "
         << (requested.trial_backend_selection_use_consistency_score ? 1 : 0)
         << "\n";
  output << "requested_stage5_trial_backend_selection_consistency_translation_sigma_mm: "
         << requested
                .trial_backend_selection_consistency_translation_sigma_mm
         << "\n";
  output << "requested_stage5_trial_backend_selection_consistency_rotation_sigma_deg: "
         << requested
                .trial_backend_selection_consistency_rotation_sigma_deg
         << "\n";
  output << "requested_stage5_trial_backend_selection_consistency_penalty_weight: "
         << requested.trial_backend_selection_consistency_penalty_weight
         << "\n";
  output << "requested_stage5_trial_backend_selection_consistency_max_translation_error_mm: "
         << requested
                .trial_backend_selection_consistency_max_translation_error_mm
         << "\n";
  output << "requested_stage5_trial_backend_selection_consistency_max_rotation_error_deg: "
         << requested
                .trial_backend_selection_consistency_max_rotation_error_deg
         << "\n";
  output << "requested_stage5_trial_backend_selection_consistency_max_local_outer_rmse_px: "
         << requested
                .trial_backend_selection_consistency_max_local_outer_rmse_px
         << "\n";
  output << "requested_stage5_trial_backend_selection_max_accepted_per_board: "
         << requested.trial_backend_selection_max_accepted_per_board << "\n";
  output << "requested_stage5_trial_backend_selection_max_accepted_per_frame: "
         << requested.trial_backend_selection_max_accepted_per_frame << "\n";
  output << "requested_stage5_trial_backend_selection_frame_cohesion: "
         << (requested.trial_backend_selection_frame_cohesion ? 1 : 0) << "\n";
  output << "requested_stage5_trial_backend_selection_frame_cohesion_max_companions: "
         << requested.trial_backend_selection_frame_cohesion_max_companions << "\n";
  output << "requested_stage5_trial_backend_selection_frame_cohesion_min_candidate_score: "
         << requested.trial_backend_selection_frame_cohesion_min_candidate_score << "\n";
  output << "requested_stage5_trial_backend_selection_min_keep_per_board: "
         << requested.trial_backend_selection_min_keep_per_board << "\n";
  output << "requested_strict_board_observation_acceptance: "
         << (requested.strict_board_observation_acceptance ? 1 : 0) << "\n";
  output << "requested_preserve_frame_board_cohesion: "
         << (requested.preserve_frame_board_cohesion ? 1 : 0) << "\n";
  output << "requested_failed_board_drop_policy: "
         << (requested.strict_board_observation_acceptance
                 ? "drop_entire_board_observation"
                 : "keep_outer_when_internal_failed")
         << "\n";
  output << "requested_backend_polar_angle_weight_mode: "
         << requested.backend_polar_angle_weight_mode << "\n";
  output << "requested_backend_polar_angle_weight_bin_edges_deg: ";
  for (std::size_t index = 0;
       index < requested.backend_polar_angle_weight_bin_edges_deg.size();
       ++index) {
    if (index > 0) {
      output << ",";
    }
    output << requested.backend_polar_angle_weight_bin_edges_deg[index];
  }
  output << "\n";
  output << "requested_backend_polar_angle_weight_fixed_bin_scales: ";
  for (std::size_t index = 0;
       index < requested.backend_polar_angle_weight_fixed_bin_scales.size();
       ++index) {
    if (index > 0) {
      output << ",";
    }
    output << requested.backend_polar_angle_weight_fixed_bin_scales[index];
  }
  output << "\n";
  output << "requested_backend_polar_angle_weight_adaptive_sigma_reference_deg: "
         << requested.backend_polar_angle_weight_adaptive_sigma_reference_deg
         << "\n";
  output << "requested_backend_polar_angle_weight_adaptive_sigma_growth: "
         << requested.backend_polar_angle_weight_adaptive_sigma_growth << "\n";
  output << "requested_backend_polar_angle_weight_min_scale: "
         << requested.backend_polar_angle_weight_min_scale << "\n";
  output << "requested_backend_residual_model: "
         << requested.backend_residual_model << "\n";
  output << "requested_backend_hybrid_angular_threshold_deg: "
         << requested.backend_hybrid_angular_threshold_deg << "\n";
  output << "requested_backend_use_point_type_residual_split: "
         << (requested.backend_use_point_type_residual_split ? 1 : 0) << "\n";
  output << "requested_backend_outer_residual_model: "
         << requested.backend_outer_residual_model << "\n";
  output << "requested_backend_internal_residual_model: "
         << requested.backend_internal_residual_model << "\n";
  output << "requested_backend_enable_angular_auxiliary_residual: "
         << (requested.backend_enable_angular_auxiliary_residual ? 1 : 0)
         << "\n";
  output << "requested_backend_angular_auxiliary_weight: "
         << requested.backend_angular_auxiliary_weight << "\n";
  output << "requested_backend_angular_auxiliary_normalized: "
         << (requested.backend_angular_auxiliary_normalized ? 1 : 0) << "\n";
  output << "requested_backend_angular_auxiliary_apply_to_outer: "
         << (requested.backend_angular_auxiliary_apply_to_outer ? 1 : 0)
         << "\n";
  output << "requested_backend_angular_auxiliary_apply_to_internal: "
         << (requested.backend_angular_auxiliary_apply_to_internal ? 1 : 0)
         << "\n";
  output << "requested_backend_polar_continuous_hybrid_threshold_deg: "
         << requested.backend_polar_continuous_hybrid_threshold_deg << "\n";
  output << "requested_backend_polar_continuous_hybrid_temperature_deg: "
         << requested.backend_polar_continuous_hybrid_temperature_deg << "\n";
  output << "requested_backend_normalized_angular_reference_sigma_px: "
         << requested.backend_normalized_angular_reference_sigma_px << "\n";
  output << "requested_backend_normalized_angular_min_sigma_rad: "
         << requested.backend_normalized_angular_min_sigma_rad << "\n";
  output << "requested_backend_normalized_angular_max_weight_scale: "
         << requested.backend_normalized_angular_max_weight_scale << "\n";
  output << "requested_backend_pixel_residual_weight: "
         << requested.backend_pixel_residual_weight << "\n";
  output << "requested_backend_chordal_residual_weight: "
         << requested.backend_chordal_residual_weight << "\n";
  output << "requested_stage5_hybrid_pixel_ray_final_refinement_enabled: "
         << (requested.enable_hybrid_pixel_ray_final_refinement ? 1 : 0)
         << "\n";
  output << "requested_stage5_hybrid_pixel_ray_lambda: "
         << requested.hybrid_pixel_ray_lambda << "\n";
  output << "requested_stage5_hybrid_pixel_ray_polar_adaptive: "
         << (requested.hybrid_pixel_ray_polar_adaptive ? 1 : 0) << "\n";
  output << "requested_stage5_hybrid_pixel_ray_lambda_min: "
         << requested.hybrid_pixel_ray_lambda_min << "\n";
  output << "requested_stage5_hybrid_pixel_ray_lambda_max: "
         << requested.hybrid_pixel_ray_lambda_max << "\n";
  output << "requested_stage5_hybrid_pixel_ray_transition_start_deg: "
         << requested.hybrid_pixel_ray_transition_start_deg << "\n";
  output << "requested_stage5_hybrid_pixel_ray_transition_end_deg: "
         << requested.hybrid_pixel_ray_transition_end_deg << "\n";
  output << "requested_stage5_hybrid_pixel_ray_max_iterations: "
         << requested.hybrid_pixel_ray_max_iterations << "\n";
  output << "requested_stage5_hybrid_pixel_ray_pixel_scale_floor: "
         << requested.hybrid_pixel_ray_pixel_scale_floor << "\n";
  output << "requested_stage5_hybrid_pixel_ray_ray_scale_floor: "
         << requested.hybrid_pixel_ray_ray_scale_floor << "\n";
  output << "requested_backend_angular_use_normalize_jacobian: "
         << (requested.backend_angular_use_normalize_jacobian ? 1 : 0)
         << "\n";
  output << "requested_backend_angular_local_whitening: "
         << (requested.backend_angular_local_whitening ? 1 : 0) << "\n";
  output << "requested_backend_angular_local_whitening_pixel_sigma_px: "
         << requested.backend_angular_local_whitening_pixel_sigma_px << "\n";
  output << "requested_backend_angular_local_whitening_covariance_damping: "
         << requested.backend_angular_local_whitening_covariance_damping
         << "\n";
  output << "requested_backend_angular_local_whitening_min_sigma_rad: "
         << requested.backend_angular_local_whitening_min_sigma_rad << "\n";
  output << "requested_backend_angular_local_whitening_max_weight: "
         << requested.backend_angular_local_whitening_max_weight << "\n";
  output << "requested_backend_angular_observed_ray_mode: "
         << requested.backend_angular_observed_ray_mode << "\n";
  output << "requested_backend_fixed_intrinsics: "
         << (requested.backend_fixed_intrinsics ? 1 : 0) << "\n";
  output << "requested_enable_angular_residual_diagnostics: "
         << (requested.enable_angular_residual_diagnostics ? 1 : 0) << "\n";
  output << "requested_angular_residual_bin_edges_deg: ";
  for (std::size_t index = 0;
       index < requested.angular_residual_bin_edges_deg.size();
       ++index) {
    if (index > 0) {
      output << ",";
    }
    output << requested.angular_residual_bin_edges_deg[index];
  }
  output << "\n";
  output << "requested_backend_multi_board_consistency_weighting: "
         << (requested.backend_multi_board_consistency_weighting ? 1 : 0) << "\n";
  output << "requested_backend_consistency_pose_source: "
         << requested.backend_consistency_pose_source << "\n";
  output << "requested_backend_consistency_weight_mode: "
         << requested.backend_consistency_weight_mode << "\n";
  output << "requested_backend_consistency_translation_sigma_mm: "
         << requested.backend_consistency_translation_sigma_mm << "\n";
  output << "requested_backend_consistency_rotation_sigma_deg: "
         << requested.backend_consistency_rotation_sigma_deg << "\n";
  output << "requested_backend_consistency_min_weight: "
         << requested.backend_consistency_min_weight << "\n";
  output << "requested_backend_consistency_apply_to_outer: "
         << (requested.backend_consistency_apply_to_outer ? 1 : 0) << "\n";
  output << "requested_backend_consistency_apply_to_internal: "
         << (requested.backend_consistency_apply_to_internal ? 1 : 0) << "\n";
  output << "requested_backend_consistency_hard_reject_enabled: "
         << (requested.backend_consistency_hard_reject_enabled ? 1 : 0) << "\n";
  output << "requested_backend_consistency_hard_reject_translation_mm: "
         << requested.backend_consistency_hard_reject_translation_mm << "\n";
  output << "requested_backend_consistency_hard_reject_rotation_deg: "
         << requested.backend_consistency_hard_reject_rotation_deg << "\n";
  output << "requested_backend_consistency_hard_reject_residual_px: "
         << requested.backend_consistency_hard_reject_residual_px << "\n";
  output << "requested_backend_consistency_dump_weight_summary: "
         << (requested.backend_consistency_dump_weight_summary ? 1 : 0) << "\n";
  output << "requested_enable_multi_board_consistency_diagnostics: "
         << (requested.enable_multi_board_consistency_diagnostics ? 1 : 0) << "\n";
  output << "requested_multi_board_consistency_pose_source: "
         << requested.multi_board_consistency_pose_source << "\n";
  output << "requested_multi_board_consistency_min_outer_points: "
         << requested.multi_board_consistency_min_outer_points << "\n";
  output << "requested_enable_global_scene_state_consistency_audit: "
         << (requested.enable_global_scene_state_consistency_audit ? 1 : 0)
         << "\n";
  output << "requested_enable_stage5_selection_diagnostics: "
         << (requested.enable_stage5_selection_diagnostics ? 1 : 0)
         << "\n";
  output << "requested_enable_multiboard_rigidity_diagnostics: "
         << (requested.enable_multiboard_rigidity_diagnostics ? 1 : 0) << "\n";
  output << "requested_multiboard_rigidity_top_k: "
         << requested.multiboard_rigidity_top_k << "\n";
  output << "requested_multiboard_rigidity_rotation_bad_threshold_deg: "
         << requested.multiboard_rigidity_rotation_bad_threshold_deg << "\n";
  output << "requested_multiboard_rigidity_translation_bad_threshold: "
         << requested.multiboard_rigidity_translation_bad_threshold << "\n";
  output << "requested_multiboard_rigidity_reprojection_delta_bad_threshold_px: "
         << requested.multiboard_rigidity_reprojection_delta_bad_threshold_px << "\n";
  output << "requested_multiboard_rigidity_use_internal_points: "
         << (requested.multiboard_rigidity_use_internal_points ? 1 : 0) << "\n";
  output << "requested_multiboard_rigidity_use_outer_points: "
         << (requested.multiboard_rigidity_use_outer_points ? 1 : 0) << "\n";
  output << "requested_kalibr_style_pose_ray_limit_deg: 80\n";
  output << "requested_internal_regeneration_diagnostics: "
         << (requested.internal_regeneration_diagnostics ? 1 : 0) << "\n";
  output << "requested_stage5_export_internal_seed_step_overlays: "
         << (requested.export_internal_seed_step_overlays ? 1 : 0) << "\n";
  output << "requested_internal_blur_diagnostics: "
         << (requested.internal_blur_diagnostics ? 1 : 0) << "\n";
  output << "requested_internal_blur_filter_mode: "
         << ati::ToString(requested.internal_blur_filter_mode) << "\n";
  output << "requested_internal_blur_filter_low_patch_gradient_quantile: "
         << requested.internal_blur_filter_low_patch_gradient_quantile << "\n";
  output << "requested_internal_blur_filter_min_board_rmse_px: "
         << requested.internal_blur_filter_min_board_rmse_px << "\n";
  output << "requested_internal_blur_filter_min_board_p95_px: "
         << requested.internal_blur_filter_min_board_p95_px << "\n";
  output << "requested_internal_pose_rescue_mode: "
         << ati::ToString(requested.internal_pose_rescue_mode) << "\n";
  output << "requested_internal_pose_rescue_max_ray_angle_deg: "
         << requested.internal_pose_rescue_max_ray_angle_deg << "\n";
  output << "requested_internal_pose_rescue_accept_max_outer_rmse: "
         << requested.internal_pose_rescue_accept_max_outer_rmse << "\n";
  output << "requested_ignore_image_evidence_min_quality: "
         << (requested.ignore_image_evidence_min_quality ? 1 : 0) << "\n";
  output << "requested_force_internal_seed_from_prediction: "
         << (requested.force_internal_seed_from_prediction ? 1 : 0) << "\n";
  output << "requested_bypass_internal_seed_filters: "
         << (requested.bypass_internal_seed_filters ? 1 : 0) << "\n";
  output << "requested_internal_corner_filter_mode: "
         << requested.internal_corner_filter_mode << "\n";
  output << "requested_internal_corner_filter_max_reproj_error: "
         << requested.internal_corner_filter_max_reproj_error << "\n";
  output << "requested_internal_corner_filter_quality_min: "
         << requested.internal_corner_filter_quality_min << "\n";
  output << "requested_internal_corner_filter_quality_relaxation_px: "
         << requested.internal_corner_filter_quality_relaxation_px << "\n";
  output << "requested_internal_corner_filter_adaptive_min_threshold_px: "
         << requested.internal_corner_filter_adaptive_min_threshold_px << "\n";
  output << "requested_enable_geometry_prior_outer_seed: "
         << (requested.enable_geometry_prior_outer_seed ? 1 : 0) << "\n";
  output << "requested_geometry_prior_rescue_diagnostic_only: "
         << (requested.geometry_prior_rescue_diagnostic_only ? 1 : 0)
         << "\n";
  output << "requested_geometry_prior_rescue_use_as_observation: "
         << (requested.geometry_prior_rescue_use_as_observation ? 1 : 0)
         << "\n";
  output << "requested_geometry_prior_rescue_keep_outer_on_internal_failure: "
         << (requested.geometry_prior_rescue_keep_outer_on_internal_failure ? 1 : 0)
         << "\n";
  output << "requested_geometry_prior_rescue_allow_geometry_only_pose_refit: "
         << (requested.geometry_prior_rescue_allow_geometry_only_pose_refit ? 1 : 0)
         << "\n";
  output << "requested_geometry_prior_rescue_subpix_window_radius: "
         << requested.geometry_prior_rescue_subpix_window_radius << "\n";
  output << "requested_geometry_prior_rescue_max_corner_displacement_px: "
         << requested.geometry_prior_rescue_max_corner_displacement_px << "\n";
  output << "requested_geometry_prior_rescue_min_corner_response_ratio: "
         << requested.geometry_prior_rescue_min_corner_response_ratio << "\n";
  output << "requested_geometry_prior_rescue_enable_spherical_refine: "
         << (requested.geometry_prior_rescue_enable_spherical_refine ? 1 : 0)
         << "\n";
  output << "requested_geometry_prior_rescue_edge_sample_count: "
         << requested.geometry_prior_rescue_edge_sample_count << "\n";
  output << "requested_geometry_prior_rescue_edge_search_half_width_px: "
         << requested.geometry_prior_rescue_edge_search_half_width_px << "\n";
  output << "requested_geometry_prior_rescue_min_edge_support_ratio: "
         << requested.geometry_prior_rescue_min_edge_support_ratio << "\n";
  output << "requested_geometry_prior_rescue_min_edge_gradient_ratio: "
         << requested.geometry_prior_rescue_min_edge_gradient_ratio << "\n";
  output << "requested_geometry_prior_rescue_accept_max_outer_rmse: "
         << requested.geometry_prior_rescue_accept_max_outer_rmse << "\n";
  output << "requested_geometry_prior_rescue_accept_max_rotation_error_deg: "
         << requested.geometry_prior_rescue_accept_max_rotation_error_deg
         << "\n";
  output << "requested_geometry_prior_rescue_accept_max_translation_error: "
         << requested.geometry_prior_rescue_accept_max_translation_error
         << "\n";
  output << "requested_enable_outer_only_intermediate_calibration: "
         << (requested.enable_outer_only_intermediate_calibration ? 1 : 0)
         << "\n";
  output << "requested_intermediate_diagnostic_only: "
         << (requested.intermediate_diagnostic_only ? 1 : 0) << "\n";
  output << "requested_use_intermediate_for_round1_internal_regeneration: "
         << (requested.use_intermediate_for_round1_internal_regeneration ? 1 : 0)
         << "\n";
  output << "requested_use_intermediate_for_full_frontend_regeneration: "
         << (requested.use_intermediate_for_full_frontend_regeneration ? 1 : 0)
         << "\n";
  output << "requested_intermediate_optimize_intrinsics: "
         << (requested.intermediate_optimize_intrinsics ? 1 : 0) << "\n";
  output << "requested_intermediate_optimize_board_poses: "
         << (requested.intermediate_optimize_board_poses ? 1 : 0) << "\n";
  output << "requested_intermediate_optimize_frame_poses: "
         << (requested.intermediate_optimize_frame_poses ? 1 : 0) << "\n";
  output << "requested_intermediate_intrinsics_release_iteration: "
         << requested.intermediate_intrinsics_release_iteration << "\n";
  output << "requested_intermediate_max_outer_rmse_px: "
         << requested.intermediate_max_outer_rmse_px << "\n";
  output << "requested_intermediate_min_visible_boards: "
         << requested.intermediate_min_visible_boards << "\n";
  output << "requested_pre_backend_filter_mode: "
         << ati::ToString(requested.pre_backend_filter_mode) << "\n";
  output << "requested_pre_backend_filter_threshold_mode: "
         << ati::ToString(requested.pre_backend_filter_threshold_mode) << "\n";
  output << "requested_pre_backend_filter_sigma: "
         << requested.pre_backend_filter_sigma << "\n";
  output << "requested_pre_backend_filter_min_abs_threshold_px: "
         << requested.pre_backend_filter_min_abs_threshold_px << "\n";
  output << "requested_internal_joint_refine_mode: "
         << ati::ToString(requested.internal_joint_refine_mode) << "\n";
  output << "requested_internal_joint_refine_target_mode: "
         << ati::ToString(requested.internal_joint_refine_target_mode) << "\n";
  output << "requested_internal_joint_refine_search_radius_px: "
         << requested.internal_joint_refine_search_radius_px << "\n";
  output << "requested_internal_joint_refine_max_displacement_px: "
         << requested.internal_joint_refine_max_displacement_px << "\n";
  output << "requested_internal_joint_refine_geometry_sigma_px: "
         << requested.internal_joint_refine_geometry_sigma_px << "\n";
  output << "requested_internal_joint_refine_observation_sigma_px: "
         << requested.internal_joint_refine_observation_sigma_px << "\n";
  output << "requested_internal_joint_refine_subpix_window_radius: "
         << requested.internal_joint_refine_subpix_window_radius << "\n";
  output << "requested_internal_joint_refine_min_objective_improvement: "
         << requested.internal_joint_refine_min_objective_improvement << "\n";
  output << "requested_internal_joint_refine_min_old_residual_px: "
         << requested.internal_joint_refine_min_old_residual_px << "\n";
  output << "requested_internal_joint_refine_low_patch_gradient_quantile: "
         << requested.internal_joint_refine_low_patch_gradient_quantile << "\n";
  output << "requested_internal_joint_refine_min_board_rmse_px: "
         << requested.internal_joint_refine_min_board_rmse_px << "\n";
  output << "requested_internal_joint_refine_min_board_p95_px: "
         << requested.internal_joint_refine_min_board_p95_px << "\n";
  output << "requested_internal_joint_refine_min_corner_response_gain: "
         << requested.internal_joint_refine_min_corner_response_gain << "\n";
  output << "requested_internal_joint_refine_min_board_internal_improvement_px: "
         << requested.internal_joint_refine_min_board_internal_improvement_px
         << "\n";
  output << "requested_internal_joint_refine_min_refined_point_count_per_board: "
         << requested.internal_joint_refine_min_refined_point_count_per_board
         << "\n";
  output << "requested_internal_joint_refine_accept_max_global_outer_delta_px: "
         << requested.internal_joint_refine_accept_max_global_outer_delta_px
         << "\n";
  output << "requested_internal_joint_refine_accept_max_frame_outer_delta_px: "
         << requested.internal_joint_refine_accept_max_frame_outer_delta_px
         << "\n";
  output << "requested_internal_joint_refine_acceptance_backend_max_iterations: "
         << requested.internal_joint_refine_acceptance_backend_max_iterations
         << "\n";
  output << "requested_internal_blur_board_weight_mode: "
         << ati::ToString(requested.internal_blur_board_weight_mode) << "\n";
  output << "requested_internal_blur_board_weight_low_patch_gradient_quantile: "
         << requested.internal_blur_board_weight_low_patch_gradient_quantile << "\n";
  output << "requested_internal_blur_board_weight_min_board_rmse_px: "
         << requested.internal_blur_board_weight_min_board_rmse_px << "\n";
  output << "requested_internal_blur_board_weight_min_board_p95_px: "
         << requested.internal_blur_board_weight_min_board_p95_px << "\n";
  output << "requested_internal_blur_board_weight_min: "
         << requested.internal_blur_board_weight_min << "\n";
  output << "requested_internal_blur_board_weight_gradient_exponent: "
         << requested.internal_blur_board_weight_gradient_exponent << "\n";
  output << "requested_internal_observation_weight_mode: "
         << ati::ToString(requested.internal_observation_weight_mode) << "\n";
  output << "requested_internal_observation_weight_policy: "
         << requested.internal_observation_weight_policy << "\n";
  output << "requested_internal_observation_weight_low_quality_quantile: "
         << requested.internal_observation_weight_low_quality_quantile << "\n";
  output << "requested_internal_observation_weight_min: "
         << requested.internal_observation_weight_min << "\n";
  output << "requested_internal_observation_weight_quality_exponent: "
         << requested.internal_observation_weight_quality_exponent << "\n";
  output << "requested_internal_observation_weight_residual_consistency_sigma: "
         << requested.internal_observation_weight_residual_consistency_sigma << "\n";
  output << "requested_internal_observation_weight_residual_consistency_min_rmse: "
         << requested.internal_observation_weight_residual_consistency_min_rmse << "\n";

  output << "stage5_success: " << (report.success ? 1 : 0) << "\n";
  output << "stage5_failure_reason: " << report.failure_reason << "\n";
  output << "effective_stage5_camera_aware_outer_rescue: "
         << (report.baseline_result.camera_aware_outer_rescue.enabled ? 1 : 0)
         << "\n";
  output << "stage5_camera_aware_outer_rescue_camera_source: "
         << report.baseline_result.camera_aware_outer_rescue.camera_source
         << "\n";
  output << "stage5_camera_aware_outer_rescue_uses_yaml_intrinsics: "
         << report.baseline_result.camera_aware_outer_rescue.uses_yaml_intrinsics
         << "\n";
  output << "stage5_camera_aware_outer_rescue_uses_kalibr_camchain_intrinsics: "
         << report.baseline_result.camera_aware_outer_rescue
                .uses_kalibr_camchain_intrinsics
         << "\n";
  output << "stage5_camera_aware_outer_rescue_rescued_board_observation_count: "
         << report.baseline_result.camera_aware_outer_rescue
                .rescued_board_observation_count
         << "\n";
  output << "effective_stage5_baseline_protocol_label: "
         << report.baseline_protocol_label << "\n";
  output << "effective_camera_init_mode: "
         << ati::ToString(report.baseline_result.auto_camera_initialization.selected_mode) << "\n";
  output << "effective_camera_init_source: "
         << report.baseline_result.auto_camera_initialization.selected_source_label << "\n";
  output << "effective_camera_init_fallback_used: "
         << (report.baseline_result.auto_camera_initialization.fallback_used ? 1 : 0) << "\n";
  output << "stage5_init_refine_mode: "
         << ati::ToString(
                report.baseline_result.auto_camera_initialization.refine_mode)
         << "\n";
  output << "stage5_init_seed_method: "
         << report.baseline_result.auto_camera_initialization.stage5_init_seed_method
         << "\n";
  output << "stage5_init_seed_source: "
         << report.baseline_result.auto_camera_initialization.stage5_init_seed_source
         << "\n";
  output << "stage5_init_omni_gamma: "
         << report.baseline_result.auto_camera_initialization.stage5_init_omni_gamma
         << "\n";
  output << "stage5_init_omni_gamma_source: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_omni_gamma_source
         << "\n";
  output << "stage5_init_ds_mapping: "
         << report.baseline_result.auto_camera_initialization.stage5_init_ds_mapping
         << "\n";
  output << "stage5_init_ds_mapping_verified_against_kalibr_source: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_ds_mapping_verified_against_kalibr_source
         << "\n";
  output << "stage5_init_ds_grid_enumeration_enabled: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_ds_grid_enumeration_enabled
         << "\n";
  output << "stage5_init_uses_yaml_intrinsics: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_uses_yaml_intrinsics
         << "\n";
  output << "stage5_init_uses_kalibr_camchain_intrinsics: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_uses_kalibr_camchain_intrinsics
         << "\n";
  output << "stage5_init_outer_only: "
         << report.baseline_result.auto_camera_initialization.stage5_init_outer_only
         << "\n";
  output << "stage5_init_uses_layout_to_update_intrinsics: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_uses_layout_to_update_intrinsics
         << "\n";
  output << "stage5_init_layout_loo_diagnostics_only: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_layout_loo_diagnostics_only
         << "\n";
  output << "stage5_init_selection_prefilter: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_prefilter
         << "\n";
  output << "stage5_init_selection_scorer: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_scorer
         << "\n";
  output << "stage5_init_selection_uses_information_metric: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_uses_information_metric
         << "\n";
  output << "stage5_init_selection_is_exact_kalibr_information_theoretic: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_is_exact_kalibr_information_theoretic
         << "\n";
  output << "stage5_init_selection_pose_marginalized: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_pose_marginalized
         << "\n";
  output << "stage5_init_selection_principal_subspace_aware: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_principal_subspace_aware
         << "\n";
  output << "stage5_init_selection_camera_information_dimension: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_camera_information_dimension
         << "\n";
  output << "stage5_init_selection_camera_information_rank: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_camera_information_rank
         << "\n";
  output << "stage5_init_selection_principal_information_rank: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_principal_information_rank
         << "\n";
  output << "stage5_init_selection_pose_rank_min: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_pose_rank_min
         << "\n";
  output << "stage5_init_selection_pose_rank_max: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_pose_rank_max
         << "\n";
  output << "stage5_init_selection_pose_rank_deficient_count: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_pose_rank_deficient_count
         << "\n";
  output << "stage5_init_selection_principal_min_eigenvalue: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_principal_min_eigenvalue
         << "\n";
  output << "stage5_init_selection_principal_max_eigenvalue: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_principal_max_eigenvalue
         << "\n";
  output << "stage5_init_selection_cu_stddev_px: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_cu_stddev_px
         << "\n";
  output << "stage5_init_selection_cv_stddev_px: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_cv_stddev_px
         << "\n";
  output << "stage5_init_selection_weakest_eigenvalue: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_weakest_eigenvalue
         << "\n";
  output << "stage5_init_selection_weakest_direction: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_weakest_direction
         << "\n";
  output << "stage5_init_selection_weakest_principal_fraction: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_weakest_principal_fraction
         << "\n";
  output << "stage5_init_selection_weakest_focal_fraction: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_weakest_focal_fraction
         << "\n";
  output << "stage5_init_selection_information_linearization: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_information_linearization
         << "\n";
  output << "stage5_init_selection_all_pose_valid_observations_used: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_selection_all_pose_valid_observations_used
         << "\n";
  output << "stage5_init_runtime_seconds: "
         << report.baseline_result.auto_camera_initialization
                .stage5_init_runtime_seconds
         << "\n";
  output << "used_config_intermediate_camera: "
         << (report.baseline_result.auto_camera_initialization
                     .used_manual_intermediate_camera
                 ? 1
                 : 0)
         << "\n";
  output << "used_explicit_initial_camera: "
         << (report.baseline_result.auto_camera_initialization
                     .used_explicit_initial_camera
                 ? 1
                 : 0)
         << "\n";
  output << "effective_models: "
         << BuildConfiguredCameraFamily(
                report.baseline_result.effective_options.config)
         << "\n";
  output << "effective_stage5_camera_model_family: "
         << BuildConfiguredCameraFamily(
                report.baseline_result.effective_options.config)
         << "\n";
  const ati::OuterBootstrapCameraIntrinsics& effective_camera =
      report.baseline_result.auto_camera_initialization.selected_camera;
  const std::string effective_family = effective_camera.NormalizedFamilyString();
  std::string native_geometry = "unknown";
  if (effective_family == "ds-none") {
    native_geometry = "DoubleSphereCameraGeometry";
  } else if (effective_family == "eucm-none") {
    native_geometry = "ExtendedUnifiedCameraGeometry";
  } else if (effective_family == "pinhole-equi") {
    native_geometry = "EquidistantDistortedPinholeCameraGeometry";
  } else if (effective_family == "omni-none") {
    native_geometry = "OmniCameraGeometry";
  } else if (effective_family == "omni-radtan") {
    native_geometry = "DistortedOmniCameraGeometry";
  }
  output << "effective_stage5_native_camera_geometry: " << native_geometry << "\n";
  output << "effective_stage5_projection_parameter_count: "
         << effective_camera.IntrinsicsLabels().size() << "\n";
  output << "effective_stage5_distortion_parameter_count: "
         << effective_camera.DistortionLabels().size() << "\n";
  output << "effective_stage5_uses_native_no_distortion_geometry: "
         << ((effective_family == "ds-none" || effective_family == "eucm-none" ||
              effective_family == "omni-none")
                 ? 1
                 : 0)
         << "\n";
  output << "camera_intrinsics_labels: ";
  for (const std::string& label :
       report.baseline_result.auto_camera_initialization.selected_camera
           .IntrinsicsLabels()) {
    output << label << " ";
  }
  output << "\n";
  output << "camera_distortion_labels: ";
  for (const std::string& label :
       report.baseline_result.auto_camera_initialization.selected_camera
           .DistortionLabels()) {
    output << label << " ";
  }
  output << "\n";
  output << "effective_outer_only_ablation_mode: "
         << (report.baseline_result.effective_options.outer_only_ablation_mode ? 1 : 0)
         << "\n";
  output << "effective_strict_outer_only_ablation: "
         << (!report.baseline_result.effective_options.include_internal_points ? 1 : 0)
         << "\n";
  output << "effective_internal_regeneration_skipped_for_outer_only: "
         << (report.baseline_result.effective_options.outer_only_ablation_mode ? 1 : 0)
         << "\n";
  output << "effective_ablation_mode: "
         << (report.baseline_result.effective_options.outer_only_ablation_mode
                 ? "outer_only"
                 : "with_internal")
         << "\n";
  output << "effective_run_second_pass: "
         << (report.baseline_result.effective_options.run_second_pass ? 1 : 0) << "\n";
  output << "effective_round2_available: "
         << (report.baseline_result.round2_available ? 1 : 0) << "\n";
  output << "effective_frontend_optimize_intrinsics: "
         << (report.baseline_result.effective_options.optimize_intrinsics ? 1 : 0) << "\n";
  output << "effective_frontend_intrinsics_release_mode: "
         << ToString(DeriveFrontendIntrinsicsReleaseMode(
                report.baseline_result.effective_options.optimize_intrinsics,
                report.baseline_result.effective_options.run_second_pass,
                report.baseline_result.effective_options.intrinsics_release_iteration,
                report.baseline_result.effective_options
                    .second_pass_intrinsics_release_iteration))
         << "\n";
  output << "effective_frontend_intrinsics_release_iteration: "
         << report.baseline_result.round1.optimization_result.intrinsics_release_iteration << "\n";
  output << "effective_frontend_second_pass_intrinsics_release_iteration: "
         << (report.baseline_result.round2_available
                 ? report.baseline_result.round2.optimization_result
                       .intrinsics_release_iteration
                 : report.baseline_result.effective_options
                       .second_pass_intrinsics_release_iteration)
         << "\n";
  output << "effective_frontend_internal_observation_quality_weighting: "
         << (report.baseline_result.effective_options
                     .enable_internal_observation_quality_weighting
                 ? 1
                 : 0)
         << "\n";
  output << "effective_frontend_internal_observation_low_quality_quantile: "
         << report.baseline_result.effective_options
                .internal_observation_low_quality_quantile
         << "\n";
  output << "effective_frontend_internal_observation_min_weight: "
         << report.baseline_result.effective_options
                .internal_observation_min_weight
         << "\n";
  output << "effective_frontend_internal_observation_quality_exponent: "
         << report.baseline_result.effective_options
                .internal_observation_quality_exponent
         << "\n";
  output << "effective_enable_residual_sanity_gate: "
         << (report.baseline_result.effective_options.enable_residual_sanity_gate ? 1 : 0)
         << "\n";
  output << "effective_enable_board_pose_fit_gate: "
         << (report.baseline_result.effective_options.enable_board_pose_fit_gate ? 1 : 0)
         << "\n";
  output << "effective_strict_board_observation_acceptance: "
         << (report.baseline_result.effective_options
                     .strict_board_observation_acceptance
                 ? 1
                 : 0)
         << "\n";
  output << "effective_preserve_frame_board_cohesion: "
         << (report.baseline_result.effective_options
                     .preserve_frame_board_cohesion
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_selection_mode: "
         << ati::ToString(
                report.baseline_result.effective_options.selection_mode)
         << "\n";
  output << "effective_stage5_selection_residual_sanity_factor: "
         << report.baseline_result.effective_options
                .selection_residual_sanity_factor
         << "\n";
  output << "effective_stage5_selection_max_board_observation_rmse: "
         << report.baseline_result.effective_options
                .selection_max_board_observation_rmse
         << "\n";
  output << "effective_stage5_selection_kalibr_style_outlier_sigma: "
         << report.baseline_result.effective_options
                .selection_kalibr_style_outlier_sigma
         << "\n";
  output << "effective_stage5_selection_kalibr_style_min_abs_threshold_px: "
         << report.baseline_result.effective_options
                .selection_kalibr_style_min_abs_threshold_px
         << "\n";
  output << "effective_stage5_selection_kalibr_style_min_views_before_filter: "
         << report.baseline_result.effective_options
                .selection_kalibr_style_min_views_before_filter
         << "\n";
  output << "effective_stage5_trial_backend_frame_board_selection: "
         << (report.trial_backend_selection_result.enabled ? 1 : 0)
         << "\n";
  output << "effective_stage5_trial_backend_selection_mode: "
         << ati::ToString(report.trial_backend_selection_result.selection_mode)
         << "\n";
  output << "effective_stage5_trial_backend_selection_budget_mode: "
         << ati::ToString(report.trial_backend_selection_result.budget_mode)
         << "\n";
  output << "effective_stage5_trial_backend_selection_candidate_order: "
         << ati::ToString(
                report.trial_backend_selection_result.candidate_order_mode)
         << "\n";
  output << "effective_stage5_trial_backend_selection_info_gain_proxy_mode: "
         << ati::ToString(
                report.trial_backend_selection_result.info_gain_proxy_mode)
         << "\n";
  output << "effective_stage5_trial_backend_selection_batch_granularity: "
         << ati::ToString(report.trial_backend_selection_result
                              .candidate_batch_granularity)
         << "\n";
  output << "effective_stage5_trial_backend_selection_acceptance_policy: "
         << ati::ToString(
                report.trial_backend_selection_result.acceptance_policy)
         << "\n";
  output << "effective_stage5_trial_backend_selection_mi_tol: "
         << report.trial_backend_selection_result
                .acceptance_information_gain_threshold
         << "\n";
  output << "effective_stage5_trial_backend_selection_rank_threshold: "
         << report.trial_backend_selection_result
                .acceptance_rank_gain_threshold
         << "\n";
  output << "effective_stage5_checkerboard_huber_delta_pixels: "
         << report.trial_backend_selection_result
                .checkerboard_huber_delta_pixels
         << "\n";
  output << "effective_stage5_checkerboard_outlier_filter_enabled: "
         << (report.trial_backend_selection_result
                     .checkerboard_outlier_filter_enabled
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_checkerboard_outlier_sigma: "
         << report.trial_backend_selection_result.checkerboard_outlier_sigma
         << "\n";
  output << "effective_stage5_checkerboard_min_inlier_ratio: "
         << report.trial_backend_selection_result
                .checkerboard_min_inlier_ratio
         << "\n";
  output << "effective_stage5_checkerboard_min_retained_points: "
         << report.trial_backend_selection_result
                .checkerboard_min_retained_points
         << "\n";
  output << "effective_stage5_trial_backend_selection_candidate_shuffle_seed_set: "
         << (report.trial_backend_selection_result.candidate_shuffle_seed_set
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_trial_backend_selection_candidate_shuffle_seed: "
         << report.trial_backend_selection_result.candidate_shuffle_seed
         << "\n";
  output << "effective_stage5_trial_backend_selection_carry_accepted_trial_state: "
         << (report.trial_backend_selection_result.carry_accepted_trial_state
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_trial_backend_selection_optimize_intrinsics: "
         << (report.trial_backend_selection_result.optimize_intrinsics_in_trial
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_trial_backend_selection_delayed_intrinsics_release: "
         << (report.trial_backend_selection_result
                     .delayed_intrinsics_release_in_trial
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_trial_backend_selection_intrinsics_release_iteration: "
         << report.trial_backend_selection_result.intrinsics_release_iteration
         << "\n";
  output << "effective_stage5_trial_backend_selection_persistent_intrinsics_anchor_prior: "
         << (report.trial_backend_selection_result
                     .persistent_intrinsics_anchor_prior_enabled
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_trial_backend_selection_persistent_fix_board_layout: "
         << (report.trial_backend_selection_result
                     .persistent_incremental_board_layout_fixed
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_xi_alpha: "
         << report.trial_backend_selection_result
                .persistent_intrinsics_anchor_weight_xi_alpha
         << "\n";
  output << "effective_stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_focal: "
         << report.trial_backend_selection_result
                .persistent_intrinsics_anchor_weight_focal
         << "\n";
  output << "effective_stage5_trial_backend_selection_persistent_intrinsics_anchor_weight_principal: "
         << report.trial_backend_selection_result
                .persistent_intrinsics_anchor_weight_principal
         << "\n";
  output << "effective_stage5_trial_backend_selection_persistent_max_focal_relative_step: "
         << report.trial_backend_selection_result
                .persistent_max_focal_relative_step
         << "\n";
  output << "effective_stage5_trial_backend_selection_persistent_max_principal_step_px: "
         << report.trial_backend_selection_result
                .persistent_max_principal_step_px
         << "\n";
  output << "effective_stage5_trial_backend_selection_persistent_max_xi_alpha_step: "
         << report.trial_backend_selection_result.persistent_max_xi_alpha_step
         << "\n";
  int trial_optimizer_total_iterations = 0;
  int trial_optimizer_total_failed_iterations = 0;
  int trial_optimizer_linear_solver_failure_count = 0;
  int trial_optimizer_intrinsics_stage_count = 0;
  double trial_optimizer_objective_start_sum = 0.0;
  double trial_optimizer_objective_final_sum = 0.0;
  double trial_optimizer_last_fu_delta = 0.0;
  double trial_optimizer_last_fv_delta = 0.0;
  double trial_optimizer_last_alpha_delta = 0.0;
  double trial_optimizer_last_xi_delta = 0.0;
  for (const ati::TrialBackendOptimizationDiagnostics& diag :
       report.trial_backend_selection_result.trial_optimization_diagnostics) {
    trial_optimizer_total_iterations += diag.total_iterations;
    trial_optimizer_total_failed_iterations += diag.total_failed_iterations;
    trial_optimizer_linear_solver_failure_count +=
        diag.any_linear_solver_failure ? 1 : 0;
    trial_optimizer_intrinsics_stage_count +=
        diag.any_intrinsics_stage ? 1 : 0;
    trial_optimizer_objective_start_sum += diag.objective_start_sum;
    trial_optimizer_objective_final_sum += diag.objective_final_sum;
    trial_optimizer_last_fu_delta = diag.camera_fu_after - diag.camera_fu_before;
    trial_optimizer_last_fv_delta = diag.camera_fv_after - diag.camera_fv_before;
    trial_optimizer_last_alpha_delta =
        diag.camera_alpha_after - diag.camera_alpha_before;
    trial_optimizer_last_xi_delta = diag.camera_xi_after - diag.camera_xi_before;
  }
  output << "effective_stage5_trial_backend_optimizer_attempt_count: "
         << report.trial_backend_selection_result
                .trial_optimization_diagnostics.size()
         << "\n";
  output << "effective_stage5_trial_backend_optimizer_total_iterations: "
         << trial_optimizer_total_iterations << "\n";
  output << "effective_stage5_trial_backend_optimizer_total_failed_iterations: "
         << trial_optimizer_total_failed_iterations << "\n";
  output << "effective_stage5_trial_backend_optimizer_intrinsics_stage_count: "
         << trial_optimizer_intrinsics_stage_count << "\n";
  output << "effective_stage5_trial_backend_optimizer_linear_solver_failure_count: "
         << trial_optimizer_linear_solver_failure_count << "\n";
  output << "effective_stage5_trial_backend_optimizer_objective_start_sum: "
         << trial_optimizer_objective_start_sum << "\n";
  output << "effective_stage5_trial_backend_optimizer_objective_final_sum: "
         << trial_optimizer_objective_final_sum << "\n";
  output << "effective_stage5_trial_backend_optimizer_last_fu_delta: "
         << trial_optimizer_last_fu_delta << "\n";
  output << "effective_stage5_trial_backend_optimizer_last_fv_delta: "
         << trial_optimizer_last_fv_delta << "\n";
  output << "effective_stage5_trial_backend_optimizer_last_alpha_delta: "
         << trial_optimizer_last_alpha_delta << "\n";
  output << "effective_stage5_trial_backend_optimizer_last_xi_delta: "
         << trial_optimizer_last_xi_delta << "\n";
  output << "effective_stage5_persistent_incremental_backend_estimator_attempted: "
         << (report.trial_backend_selection_result
                     .persistent_incremental_backend_estimator_attempted
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_persistent_incremental_backend_estimator_used: "
         << (report.trial_backend_selection_result
                     .persistent_incremental_backend_estimator_used
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_persistent_incremental_backend_estimator_compatible: "
         << (report.trial_backend_selection_result
                     .persistent_incremental_backend_estimator_compatible
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_persistent_incremental_backend_estimator_fallback_reason: "
         << report.trial_backend_selection_result
                .persistent_incremental_backend_estimator_fallback_reason
         << "\n";
  output << "effective_stage5_persistent_incremental_backend_estimator_failure_reason: "
         << report.trial_backend_selection_result
                .persistent_incremental_backend_estimator_failure_reason
         << "\n";
  output << "effective_stage5_persistent_incremental_seed_batch_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_seed_batch_count
         << "\n";
  output << "effective_stage5_persistent_incremental_seed_frame_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_seed_frame_count
         << "\n";
  output << "effective_stage5_persistent_incremental_seed_board_observation_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_seed_board_observation_count
         << "\n";
  output << "effective_stage5_persistent_incremental_seed_point_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_seed_point_count
         << "\n";
  output << "effective_stage5_persistent_incremental_candidate_batch_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_candidate_batch_count
         << "\n";
  output << "effective_stage5_persistent_incremental_attempted_batch_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_attempted_batch_count
         << "\n";
  output << "effective_stage5_persistent_incremental_accepted_batch_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_accepted_batch_count
         << "\n";
  output << "effective_stage5_persistent_incremental_rejected_batch_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_rejected_batch_count
         << "\n";
  output << "effective_stage5_persistent_incremental_solver_profile_name: "
         << report.trial_backend_selection_result
                .persistent_incremental_solver_profile_name
         << "\n";
  output << "effective_stage5_persistent_incremental_solver_objective_unit: "
         << report.trial_backend_selection_result
                .persistent_incremental_solver_objective_unit
         << "\n";
  output << "effective_stage5_persistent_incremental_solver_max_iterations: "
         << report.trial_backend_selection_result
                .persistent_incremental_solver_max_iterations
         << "\n";
  output << "effective_stage5_persistent_incremental_solver_convergence_delta_j: "
         << report.trial_backend_selection_result
                .persistent_incremental_solver_convergence_delta_j
         << "\n";
  output << "effective_stage5_persistent_incremental_solver_bearing_reference_focal_px: "
         << report.trial_backend_selection_result
                .persistent_incremental_solver_bearing_reference_focal_px
         << "\n";
  output << "effective_stage5_persistent_incremental_solver_single_iteration_batch_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_solver_single_iteration_batch_count
         << "\n";
  output << "effective_stage5_persistent_incremental_solver_max_iteration_batch_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_solver_max_iteration_batch_count
         << "\n";
  output << "effective_stage5_persistent_incremental_solver_relative_objective_converged_batch_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_solver_relative_objective_converged_batch_count
         << "\n";
  output << "effective_stage5_persistent_incremental_total_elapsed_time_seconds: "
         << report.trial_backend_selection_result
                .persistent_incremental_total_elapsed_time_seconds
         << "\n";
  output << "effective_stage5_trial_backend_selection_valid_candidate_count: "
         << report.trial_backend_selection_result.valid_candidate_count
         << "\n";
  output << "effective_stage5_trial_backend_selection_valid_candidate_traversed_count: "
         << report.trial_backend_selection_result.valid_candidate_traversed_count
         << "\n";
  output << "effective_stage5_trial_backend_selection_safety_ceiling_hit: "
         << (report.trial_backend_selection_result.safety_ceiling_hit ? 1 : 0)
         << "\n";
  output << "effective_stage5_trial_backend_selection_runtime_safety_ceiling: "
         << report.trial_backend_selection_result.runtime_safety_ceiling
         << "\n";
  output << "effective_stage5_trial_backend_selection_max_candidate_additions_effective: "
         << report.trial_backend_selection_result
                .max_candidate_additions_effective
         << "\n";
  output << "effective_stage5_trial_backend_selection_success: "
         << (report.trial_backend_selection_result.success ? 1 : 0)
         << "\n";
  output << "effective_stage5_trial_backend_selection_threshold_px: "
         << report.trial_backend_selection_result.threshold_px << "\n";
  output << "effective_stage5_trial_backend_selection_baseline_seed_board_observations: "
         << report.trial_backend_selection_result
                .baseline_seed_board_observation_count
         << "\n";
  output << "effective_stage5_trial_backend_selection_candidate_board_observations: "
         << report.trial_backend_selection_result
                .candidate_board_observation_count
         << "\n";
  output << "effective_stage5_trial_backend_selection_attempted_candidates: "
         << report.trial_backend_selection_result.attempted_candidate_count
         << "\n";
  output << "effective_stage5_trial_backend_selection_accepted_candidates: "
         << report.trial_backend_selection_result.accepted_candidate_count
         << "\n";
  output << "effective_stage5_trial_backend_frame_batch_candidate_count: "
         << report.trial_backend_selection_result.frame_batch_candidate_count
         << "\n";
  output << "effective_stage5_trial_backend_frame_batch_attempted_count: "
         << report.trial_backend_selection_result.frame_batch_attempted_count
         << "\n";
  output << "effective_stage5_trial_backend_frame_batch_accepted_count: "
         << report.trial_backend_selection_result.frame_batch_accepted_count
         << "\n";
  output << "effective_stage5_trial_backend_frame_batch_rejected_count: "
         << report.trial_backend_selection_result.frame_batch_rejected_count
         << "\n";
  output << "effective_stage5_trial_backend_frame_consolidation_candidate_count: "
         << report.trial_backend_selection_result
                .frame_consolidation_candidate_count
         << "\n";
  output << "effective_stage5_trial_backend_frame_consolidation_accepted_count: "
         << report.trial_backend_selection_result
                .frame_consolidation_accepted_count
         << "\n";
  output << "effective_stage5_trial_backend_frame_consolidation_rejected_count: "
         << report.trial_backend_selection_result
                .frame_consolidation_rejected_count
         << "\n";
  output << "effective_stage5_trial_backend_frame_consolidation_dropped_board_observation_count: "
         << report.trial_backend_selection_result
                .frame_consolidation_dropped_board_observation_count
         << "\n";
  output << "effective_stage5_trial_backend_batch_acceptance_attempted_count: "
         << report.trial_backend_selection_result
                .batch_acceptance_attempted_count
         << "\n";
  output << "effective_stage5_trial_backend_batch_acceptance_accepted_count: "
         << report.trial_backend_selection_result
                .batch_acceptance_accepted_count
         << "\n";
  output << "effective_stage5_trial_backend_batch_acceptance_rescued_from_legacy_rmse_gate_count: "
         << report.trial_backend_selection_result
                .batch_acceptance_rescued_from_legacy_rmse_gate_count
         << "\n";
  output << "effective_stage5_trial_backend_batch_acceptance_rejected_hard_validity_count: "
         << report.trial_backend_selection_result
                .batch_acceptance_rejected_hard_validity_count
         << "\n";
  output << "effective_stage5_trial_backend_batch_acceptance_rejected_catastrophic_residual_count: "
         << report.trial_backend_selection_result
                .batch_acceptance_rejected_catastrophic_residual_count
         << "\n";
  output << "effective_stage5_trial_backend_batch_acceptance_rejected_score_count: "
         << report.trial_backend_selection_result
                .batch_acceptance_rejected_score_count
         << "\n";
  output << "effective_stage5_persistent_incremental_trust_region_backtracking_batch_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_trust_region_backtracking_batch_count
         << "\n";
  output << "effective_stage5_persistent_incremental_trust_region_backtracking_attempt_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_trust_region_backtracking_attempt_count
         << "\n";
  output << "effective_stage5_persistent_incremental_trust_region_backtracking_accepted_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_trust_region_backtracking_accepted_count
         << "\n";
  output << "effective_stage5_persistent_incremental_trust_region_backtracking_max_anchor_scale: "
         << report.trial_backend_selection_result
                .persistent_incremental_trust_region_backtracking_max_anchor_scale
         << "\n";
  output << "effective_stage5_persistent_incremental_normalize_information_gain_by_board_observation: "
         << (report.trial_backend_selection_result
                     .persistent_incremental_normalize_information_gain_by_board_observation
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_persistent_incremental_split_residual_health_gate_enabled: "
         << (report.trial_backend_selection_result
                     .persistent_incremental_split_residual_health_gate_enabled
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_persistent_incremental_split_residual_health_rejected_count: "
         << report.trial_backend_selection_result
                .persistent_incremental_split_residual_health_rejected_count
         << "\n";
  output << "effective_stage5_persistent_incremental_adaptive_saturation_stop_enabled: "
         << (report.trial_backend_selection_result
                     .persistent_incremental_adaptive_saturation_stop_enabled
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_persistent_incremental_adaptive_saturation_stop_hit: "
         << (report.trial_backend_selection_result
                     .persistent_incremental_adaptive_saturation_stop_hit
                 ? 1
                 : 0)
         << "\n";
  output << "effective_stage5_persistent_incremental_adaptive_saturation_min_accepted_batches: "
         << report.trial_backend_selection_result
                .persistent_incremental_adaptive_saturation_min_accepted_batches
         << "\n";
  output << "effective_stage5_persistent_incremental_adaptive_saturation_nonproductive_batch_limit: "
         << report.trial_backend_selection_result
                .persistent_incremental_adaptive_saturation_nonproductive_batch_limit
         << "\n";
  output << "effective_stage5_persistent_incremental_adaptive_saturation_consecutive_nonproductive_batches: "
         << report.trial_backend_selection_result
                .persistent_incremental_adaptive_saturation_consecutive_nonproductive_batches
         << "\n";
  output << "effective_stage5_persistent_incremental_adaptive_saturation_tail_ordering_score_threshold: "
         << report.trial_backend_selection_result
                .persistent_incremental_adaptive_saturation_tail_ordering_score_threshold
         << "\n";
  output << "effective_stage5_persistent_incremental_adaptive_saturation_next_ordering_score: "
         << report.trial_backend_selection_result
                .persistent_incremental_adaptive_saturation_next_ordering_score
         << "\n";
  output << "effective_stage5_persistent_incremental_adaptive_saturation_stop_reason: "
         << report.trial_backend_selection_result
                .persistent_incremental_adaptive_saturation_stop_reason
         << "\n";
  output << "effective_stage5_close_distance_candidate_count: "
         << report.trial_backend_selection_result.close_distance_candidate_count
         << "\n";
  output << "effective_stage5_close_distance_accepted_count: "
         << report.trial_backend_selection_result.close_distance_accepted_count
         << "\n";
  output << "effective_stage5_trial_backend_selection_rejected_board_observations: "
         << report.trial_backend_selection_result
                .rejected_board_observation_count
         << "\n";
  output << "effective_stage5_close_edge_hard_case_count: "
         << report.trial_backend_selection_result.close_edge_hard_case_count
         << "\n";
  output << "effective_stage5_close_edge_soft_candidate_count: "
         << report.trial_backend_selection_result.close_edge_soft_candidate_count
         << "\n";
  output << "effective_stage5_close_edge_soft_attempted_count: "
         << report.trial_backend_selection_result.close_edge_soft_attempted_count
         << "\n";
  output << "effective_stage5_close_edge_soft_accepted_count: "
         << report.trial_backend_selection_result.close_edge_soft_accepted_count
         << "\n";
  output << "effective_failed_board_drop_policy: "
         << (report.baseline_result.effective_options
                     .strict_board_observation_acceptance
                 ? "drop_entire_board_observation"
                 : "keep_outer_when_internal_failed")
         << "\n";
  output << "effective_kalibr_style_pose_ray_limit_deg: 80\n";
  output << "effective_internal_regeneration_diagnostics: "
         << (requested.internal_regeneration_diagnostics ? 1 : 0) << "\n";
  output << "effective_stage5_export_internal_seed_step_overlays: "
         << (requested.export_internal_seed_step_overlays ? 1 : 0) << "\n";
  output << "effective_internal_blur_diagnostics: "
         << (requested.internal_blur_diagnostics ? 1 : 0) << "\n";
  output << "effective_internal_blur_filter_mode: "
         << ati::ToString(report.internal_blur_filter_result.options.mode)
         << "\n";
  output << "effective_internal_blur_filter_diagnostic_only: "
         << (report.internal_blur_filter_result.diagnostic_only ? 1 : 0)
         << "\n";
  output << "effective_internal_blur_filter_backend_input_changed: "
         << (report.internal_blur_filter_result.backend_input_changed ? 1 : 0)
         << "\n";
  output << "effective_internal_blur_filter_low_patch_gradient_quantile: "
         << report.internal_blur_filter_result.options
                .low_patch_gradient_quantile
         << "\n";
  output << "effective_internal_blur_filter_patch_gradient_threshold: "
         << report.internal_blur_filter_result.patch_gradient_threshold << "\n";
  output << "effective_internal_blur_filter_min_board_rmse_px: "
         << report.internal_blur_filter_result.options
                .min_board_internal_rmse_px
         << "\n";
  output << "effective_internal_blur_filter_min_board_p95_px: "
         << report.internal_blur_filter_result.options
                .min_board_p95_residual_px
         << "\n";
  output << "effective_internal_blur_filter_input_internal_point_count: "
         << report.internal_blur_filter_result.input_internal_point_count
         << "\n";
  output << "effective_internal_blur_filter_filtered_internal_point_count: "
         << report.internal_blur_filter_result.filtered_internal_point_count
         << "\n";
  output << "effective_internal_blur_filter_remaining_internal_point_count: "
         << report.internal_blur_filter_result.remaining_internal_point_count
         << "\n";
  output << "effective_internal_blur_filter_filtered_board_observation_count: "
         << report.internal_blur_filter_result.filtered_board_observation_count
         << "\n";
  output << "effective_internal_pose_rescue_mode: "
         << ati::ToString(report.baseline_result.effective_options
                              .internal_pose_rescue_mode)
         << "\n";
  output << "effective_internal_pose_rescue_max_ray_angle_deg: "
         << report.baseline_result.effective_options
                .internal_pose_rescue_max_ray_angle_deg
         << "\n";
  output << "effective_internal_pose_rescue_accept_max_outer_rmse: "
         << report.baseline_result.effective_options
                .internal_pose_rescue_accept_max_outer_rmse
         << "\n";
  output << "effective_ignore_image_evidence_min_quality: "
         << (report.baseline_result.effective_options
                     .ignore_image_evidence_min_quality
                 ? 1
                 : 0)
         << "\n";
  output << "effective_force_internal_seed_from_prediction: "
         << (report.baseline_result.effective_options
                     .force_internal_seed_from_prediction
                 ? 1
                 : 0)
         << "\n";
  output << "effective_bypass_internal_seed_filters: "
         << (report.baseline_result.effective_options
                     .bypass_internal_seed_filters
                 ? 1
                 : 0)
         << "\n";
  output << "effective_internal_corner_filter_mode: "
         << report.baseline_result.effective_options
                .internal_corner_filter_mode
         << "\n";
  output << "effective_internal_corner_filter_max_reproj_error: "
         << report.baseline_result.effective_options
                .internal_corner_filter_max_reproj_error
         << "\n";
  output << "effective_internal_corner_filter_quality_min: "
         << report.baseline_result.effective_options
                .internal_corner_filter_quality_min
         << "\n";
  output << "effective_internal_corner_filter_quality_relaxation_px: "
         << report.baseline_result.effective_options
                .internal_corner_filter_quality_relaxation_px
         << "\n";
  output << "effective_internal_corner_filter_adaptive_min_threshold_px: "
         << report.baseline_result.effective_options
                .internal_corner_filter_adaptive_min_threshold_px
         << "\n";
  output << "effective_enable_geometry_prior_outer_seed: "
         << (report.baseline_result.effective_options
                     .enable_geometry_prior_outer_seed
                 ? 1
                 : 0)
         << "\n";
  output << "effective_geometry_prior_rescue_diagnostic_only: "
         << (report.baseline_result.effective_options
                     .geometry_prior_rescue_diagnostic_only
                 ? 1
                 : 0)
         << "\n";
  output << "effective_geometry_prior_rescue_use_as_observation: "
         << (report.baseline_result.effective_options
                     .geometry_prior_rescue_use_as_observation
                 ? 1
                 : 0)
         << "\n";
  output << "effective_geometry_prior_rescue_keep_outer_on_internal_failure: "
         << (report.baseline_result.effective_options
                     .geometry_prior_rescue_keep_outer_on_internal_failure
                 ? 1
                 : 0)
         << "\n";
  output << "effective_geometry_prior_rescue_allow_geometry_only_pose_refit: "
         << (report.baseline_result.effective_options
                     .geometry_prior_rescue_allow_geometry_only_pose_refit
                 ? 1
                 : 0)
         << "\n";
  output << "effective_geometry_only_updates_observations: "
         << (report.baseline_result.effective_options
                     .geometry_prior_rescue_allow_geometry_only_pose_refit &&
                 report.baseline_result.effective_options
                     .geometry_prior_rescue_use_as_observation &&
                 !report.baseline_result.effective_options
                      .geometry_prior_rescue_diagnostic_only
             ? 1
             : 0)
         << "\n";
  output << "effective_geometry_prior_quad_topology_guard: 1\n";
  output << "effective_geometry_prior_visible_frame_refit_mode: "
            "robust_multi_board_consensus\n";
  output << "effective_geometry_prior_rescue_subpix_window_radius: "
         << report.baseline_result.effective_options
                .geometry_prior_rescue_subpix_window_radius
         << "\n";
  output << "effective_geometry_prior_rescue_max_corner_displacement_px: "
         << report.baseline_result.effective_options
                .geometry_prior_rescue_max_corner_displacement_px
         << "\n";
  output << "effective_geometry_prior_corner_displacement_guard_mode: "
         << (report.baseline_result.effective_options
                         .geometry_prior_rescue_max_corner_displacement_px < 0.0
                 ? "disabled_debug"
                 : (report.baseline_result.effective_options
                                .geometry_prior_rescue_max_corner_displacement_px ==
                            0.0
                        ? "scale_adaptive"
                        : "fixed_px"))
         << "\n";
  output << "effective_geometry_prior_rescue_min_corner_response_ratio: "
         << report.baseline_result.effective_options
                .geometry_prior_rescue_min_corner_response_ratio
         << "\n";
  output << "effective_geometry_prior_rescue_enable_spherical_refine: "
         << (report.baseline_result.effective_options
                     .geometry_prior_rescue_enable_spherical_refine
                 ? 1
                 : 0)
         << "\n";
  output << "effective_geometry_prior_rescue_edge_sample_count: "
         << report.baseline_result.effective_options
                .geometry_prior_rescue_edge_sample_count
         << "\n";
  output << "effective_geometry_prior_rescue_edge_search_half_width_px: "
         << report.baseline_result.effective_options
                .geometry_prior_rescue_edge_search_half_width_px
         << "\n";
  output << "effective_geometry_prior_rescue_min_edge_support_ratio: "
         << report.baseline_result.effective_options
                .geometry_prior_rescue_min_edge_support_ratio
         << "\n";
  output << "effective_geometry_prior_rescue_min_edge_gradient_ratio: "
         << report.baseline_result.effective_options
                .geometry_prior_rescue_min_edge_gradient_ratio
         << "\n";
  output << "effective_geometry_prior_rescue_accept_max_outer_rmse: "
         << report.baseline_result.effective_options
                .geometry_prior_rescue_accept_max_outer_rmse
         << "\n";
  output << "effective_geometry_prior_rescue_accept_max_rotation_error_deg: "
         << report.baseline_result.effective_options
                .geometry_prior_rescue_accept_max_rotation_error_deg
         << "\n";
  output << "effective_geometry_prior_rescue_accept_max_translation_error: "
         << report.baseline_result.effective_options
                .geometry_prior_rescue_accept_max_translation_error
         << "\n";
    output << "effective_pre_backend_filter_mode: "
           << ati::ToString(report.pre_backend_filter_result.options.mode) << "\n";
    output << "effective_pre_backend_filter_threshold_mode: "
           << ati::ToString(
                  report.pre_backend_filter_result.options.threshold_mode)
           << "\n";
    output << "effective_pre_backend_filter_diagnostic_only: "
           << (report.pre_backend_filter_result.diagnostic_only ? 1 : 0) << "\n";
  output << "effective_pre_backend_filter_backend_input_changed: "
         << (report.pre_backend_filter_result.backend_input_changed ? 1 : 0)
         << "\n";
  output << "effective_pre_backend_filter_sigma: "
         << report.pre_backend_filter_result.options.sigma_threshold << "\n";
  output << "effective_pre_backend_filter_min_abs_threshold_px: "
         << report.pre_backend_filter_result.options.min_abs_threshold_px << "\n";
  output << "effective_pre_backend_filter_input_internal_point_count: "
         << report.pre_backend_filter_result.input_internal_point_count << "\n";
  output << "effective_pre_backend_filter_filtered_internal_point_count: "
         << report.pre_backend_filter_result.filtered_internal_point_count << "\n";
  output << "effective_pre_backend_filter_remaining_internal_point_count: "
         << report.pre_backend_filter_result.remaining_internal_point_count << "\n";
  output << "effective_internal_joint_refine_mode: "
         << ati::ToString(report.internal_joint_refine_result.options.mode)
         << "\n";
  output << "effective_internal_joint_refine_target_mode: "
         << ati::ToString(report.internal_joint_refine_result.options.target_mode)
         << "\n";
  output << "effective_internal_joint_refine_diagnostic_only: "
         << (report.internal_joint_refine_result.diagnostic_only ? 1 : 0)
         << "\n";
  output << "effective_internal_joint_refine_backend_input_changed: "
         << (report.internal_joint_refine_result.backend_input_changed ? 1 : 0)
         << "\n";
  output << "effective_internal_joint_refine_search_radius_px: "
         << report.internal_joint_refine_result.options.search_radius_px << "\n";
  output << "effective_internal_joint_refine_max_displacement_px: "
         << report.internal_joint_refine_result.options.max_displacement_px
         << "\n";
  output << "effective_internal_joint_refine_geometry_sigma_px: "
         << report.internal_joint_refine_result.options.geometry_sigma_px
         << "\n";
  output << "effective_internal_joint_refine_observation_sigma_px: "
         << report.internal_joint_refine_result.options.observation_sigma_px
         << "\n";
  output << "effective_internal_joint_refine_subpix_window_radius: "
         << report.internal_joint_refine_result.options.subpix_window_radius
         << "\n";
  output << "effective_internal_joint_refine_min_objective_improvement: "
         << report.internal_joint_refine_result.options.min_objective_improvement
         << "\n";
  output << "effective_internal_joint_refine_min_old_residual_px: "
         << report.internal_joint_refine_result.options.min_old_residual_px
         << "\n";
  output << "effective_internal_joint_refine_low_patch_gradient_quantile: "
         << report.internal_joint_refine_result.options.low_patch_gradient_quantile
         << "\n";
  output << "effective_internal_joint_refine_min_board_rmse_px: "
         << report.internal_joint_refine_result.options.min_board_internal_rmse_px
         << "\n";
  output << "effective_internal_joint_refine_min_board_p95_px: "
         << report.internal_joint_refine_result.options.min_board_p95_residual_px
         << "\n";
  output << "effective_internal_joint_refine_min_corner_response_gain: "
         << report.internal_joint_refine_result.options.min_corner_response_gain
         << "\n";
  output << "effective_internal_joint_refine_min_board_internal_improvement_px: "
         << report.internal_joint_refine_result.options
                .min_board_internal_rmse_improvement_px
         << "\n";
  output << "effective_internal_joint_refine_min_refined_point_count_per_board: "
         << report.internal_joint_refine_result.options
                .min_refined_point_count_per_board
         << "\n";
  output << "effective_internal_joint_refine_accept_max_global_outer_delta_px: "
         << report.internal_joint_refine_result.options
                .accept_max_global_outer_delta_px
         << "\n";
  output << "effective_internal_joint_refine_accept_max_frame_outer_delta_px: "
         << report.internal_joint_refine_result.options
                .accept_max_frame_outer_delta_px
         << "\n";
  output << "effective_internal_joint_refine_acceptance_backend_max_iterations: "
         << report.internal_joint_refine_result.options
                .acceptance_backend_max_iterations
         << "\n";
  output << "effective_internal_joint_refine_candidate_board_count: "
         << report.internal_joint_refine_result.candidate_board_count << "\n";
  output << "effective_internal_joint_refine_accepted_board_count: "
         << report.internal_joint_refine_result.accepted_board_count << "\n";
  output << "effective_internal_joint_refine_rolled_back_board_count: "
         << report.internal_joint_refine_result.rolled_back_board_count << "\n";
  output << "effective_internal_joint_refine_input_internal_point_count: "
         << report.internal_joint_refine_result.input_internal_point_count
         << "\n";
  output << "effective_internal_joint_refine_eligible_internal_point_count: "
         << report.internal_joint_refine_result.eligible_internal_point_count
         << "\n";
  output << "effective_internal_joint_refine_eligible_ratio: "
         << report.internal_joint_refine_result.eligible_ratio << "\n";
  output << "effective_internal_joint_refine_refined_internal_point_count: "
         << report.internal_joint_refine_result.refined_internal_point_count
         << "\n";
  output << "effective_internal_joint_refine_mean_displacement_px: "
         << report.internal_joint_refine_result.mean_displacement_px << "\n";
  output << "effective_internal_blur_board_weight_mode: "
         << ati::ToString(report.internal_blur_board_weight_result.options.mode)
         << "\n";
  output << "effective_internal_blur_board_weight_diagnostic_only: "
         << (report.internal_blur_board_weight_result.diagnostic_only ? 1 : 0)
         << "\n";
  output << "effective_internal_blur_board_weight_backend_input_changed: "
         << (report.internal_blur_board_weight_result.backend_input_changed ? 1 : 0)
         << "\n";
  output << "effective_internal_blur_board_weight_low_patch_gradient_quantile: "
         << report.internal_blur_board_weight_result.options
                .low_patch_gradient_quantile
         << "\n";
  output << "effective_internal_blur_board_weight_patch_gradient_threshold: "
         << report.internal_blur_board_weight_result.patch_gradient_threshold
         << "\n";
  output << "effective_internal_blur_board_weight_min_board_rmse_px: "
         << report.internal_blur_board_weight_result.options
                .min_board_internal_rmse_px
         << "\n";
  output << "effective_internal_blur_board_weight_min_board_p95_px: "
         << report.internal_blur_board_weight_result.options
                .min_board_p95_residual_px
         << "\n";
  output << "effective_internal_blur_board_weight_min_option: "
         << report.internal_blur_board_weight_result.options.min_weight << "\n";
  output << "effective_internal_blur_board_weight_gradient_exponent: "
         << report.internal_blur_board_weight_result.options.gradient_exponent
         << "\n";
  output << "effective_internal_blur_board_weight_input_internal_point_count: "
         << report.internal_blur_board_weight_result.input_internal_point_count
         << "\n";
  output << "effective_internal_blur_board_weight_downweighted_internal_point_count: "
         << report.internal_blur_board_weight_result.downweighted_internal_point_count
         << "\n";
  output << "effective_internal_blur_board_weight_downweighted_board_observation_count: "
         << report.internal_blur_board_weight_result.downweighted_board_observation_count
         << "\n";
  output << "effective_internal_blur_board_weight_mean_weight: "
         << report.internal_blur_board_weight_result.mean_weight << "\n";
  output << "effective_internal_observation_weight_mode: "
         << ati::ToString(
                report.internal_observation_weight_result.options.mode)
         << "\n";
  output << "effective_internal_observation_weight_policy: "
         << report.internal_observation_weight_result.policy << "\n";
  output << "effective_internal_observation_weight_diagnostic_only: "
         << (report.internal_observation_weight_result.diagnostic_only ? 1 : 0)
         << "\n";
  output << "effective_internal_observation_weight_backend_input_changed: "
         << (report.internal_observation_weight_result.backend_input_changed ? 1 : 0)
         << "\n";
  output << "effective_internal_observation_weight_low_quality_quantile: "
         << report.internal_observation_weight_result.options
                .low_quality_quantile
         << "\n";
  output << "effective_internal_observation_weight_quality_threshold: "
         << report.internal_observation_weight_result.quality_threshold
         << "\n";
  output << "effective_internal_observation_weight_min_option: "
         << report.internal_observation_weight_result.options.min_weight
         << "\n";
  output << "effective_internal_observation_weight_quality_exponent: "
         << report.internal_observation_weight_result.options.quality_exponent
         << "\n";
  output << "effective_internal_observation_weight_residual_consistency_sigma: "
         << report.internal_observation_weight_result
                .residual_consistency_sigma_multiplier
         << "\n";
  output << "effective_internal_observation_weight_residual_consistency_min_rmse: "
         << report.internal_observation_weight_result.residual_consistency_min_rmse
         << "\n";
  output << "effective_internal_observation_weight_residual_consistency_ratio_threshold: "
         << report.internal_observation_weight_result
                .residual_consistency_ratio_threshold
         << "\n";
  output << "effective_internal_observation_weight_input_internal_point_count: "
         << report.internal_observation_weight_result.input_internal_point_count
         << "\n";
  output << "effective_internal_observation_weight_downweighted_internal_point_count: "
         << report.internal_observation_weight_result
                .downweighted_internal_point_count
         << "\n";
  output << "effective_internal_observation_weight_mean_weight: "
         << report.internal_observation_weight_result.mean_weight << "\n";
  output << "effective_residual_sanity_factor: "
         << report.baseline_result.effective_options
                .selection_residual_sanity_factor
         << "\n";
  output << "effective_max_board_observation_rmse: "
         << report.baseline_result.effective_options
                .selection_max_board_observation_rmse
         << "\n";
  output << "effective_max_pose_fit_outer_rmse: 8\n";

  output << "effective_backend_problem_optimize_intrinsics: "
         << (report.backend_problem_input.optimization_masks.optimize_intrinsics ? 1 : 0)
         << "\n";
  output << "effective_backend_problem_optimize_board_poses: "
         << (report.backend_problem_input.optimization_masks.optimize_board_poses ? 1 : 0)
         << "\n";
  output << "effective_backend_problem_delayed_intrinsics_release: "
         << (report.backend_problem_input.optimization_masks.delayed_intrinsics_release ? 1 : 0)
         << "\n";
  output << "effective_backend_problem_intrinsics_release_mode: "
         << ToString(DeriveBackendIntrinsicsReleaseMode(
                report.backend_problem_input.optimization_masks.optimize_intrinsics,
                report.backend_problem_input.optimization_masks
                    .delayed_intrinsics_release))
         << "\n";
  output << "effective_backend_problem_intrinsics_release_iteration: "
         << report.backend_problem_input.optimization_masks.intrinsics_release_iteration << "\n";
  output << "effective_backend_problem_total_point_count: "
         << report.backend_problem_input.measurement_dataset.accepted_total_point_count
         << "\n";
  output << "effective_backend_problem_outer_point_count: "
         << report.backend_problem_input.measurement_dataset.accepted_outer_point_count
         << "\n";
  output << "effective_backend_problem_internal_point_count: "
         << report.backend_problem_input.measurement_dataset.accepted_internal_point_count
         << "\n";

  output << "round1_selected_frame_count: "
         << report.baseline_result.round1.selection_result.accepted_frame_count << "\n";
  output << "round1_selected_board_observation_count: "
         << report.baseline_result.round1.selection_result
                .accepted_board_observation_count
         << "\n";
  output << "round1_selected_internal_point_count: "
         << report.baseline_result.round1.selection_result.accepted_internal_point_count
         << "\n";
  output << "round2_selected_frame_count: "
         << (report.baseline_result.round2_available
                 ? report.baseline_result.round2.selection_result.accepted_frame_count
                 : 0)
         << "\n";
  output << "round2_selected_board_observation_count: "
         << (report.baseline_result.round2_available
                 ? report.baseline_result.round2.selection_result
                       .accepted_board_observation_count
                 : 0)
         << "\n";
  output << "round2_selected_internal_point_count: "
         << (report.baseline_result.round2_available
                 ? report.baseline_result.round2.selection_result
                       .accepted_internal_point_count
                 : 0)
         << "\n";

  if (backend_result != nullptr) {
    output << "backend_success: " << (backend_result->success ? 1 : 0) << "\n";
    output << "backend_failure_reason: " << backend_result->failure_reason << "\n";
    output << "backend_skip_optimization: "
           << (backend_result->options.skip_optimization ? 1 : 0) << "\n";
    output << "backend_final_state_label: after_incremental_selection_ba\n";
    output << "backend_board_pose_parameterization: "
           << backend_result->board_pose_parameterization << "\n";
    output << "effective_backend_runner_optimize_intrinsics: "
           << (backend_result->effective_problem_input.optimization_masks
                       .optimize_intrinsics
                   ? 1
                   : 0)
           << "\n";
    output << "effective_backend_runner_optimize_board_poses: "
           << (backend_result->effective_problem_input.optimization_masks
                       .optimize_board_poses
                   ? 1
                   : 0)
           << "\n";
    output << "effective_backend_runner_delayed_intrinsics_release: "
           << (backend_result->effective_problem_input.optimization_masks
                       .delayed_intrinsics_release
                   ? 1
                   : 0)
           << "\n";
    output << "effective_backend_runner_intrinsics_release_mode: "
           << ToString(DeriveBackendIntrinsicsReleaseMode(
                  backend_result->effective_problem_input.optimization_masks
                      .optimize_intrinsics,
                  backend_result->effective_problem_input.optimization_masks
                      .delayed_intrinsics_release))
           << "\n";
    output << "effective_backend_runner_intrinsics_release_iteration: "
           << backend_result->effective_problem_input.optimization_masks
                  .intrinsics_release_iteration
           << "\n";
  }
}

std::string CsvEscape(const std::string& value) {
  const bool needs_quotes =
      value.find(',') != std::string::npos ||
      value.find('"') != std::string::npos ||
      value.find('\n') != std::string::npos ||
      value.find('\r') != std::string::npos;
  if (!needs_quotes) {
    return value;
  }
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"') {
      escaped += "\"\"";
    } else {
      escaped += ch;
    }
  }
  escaped += "\"";
  return escaped;
}

double SafeRatio(int numerator, int denominator) {
  if (denominator <= 0) {
    return 0.0;
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

std::string TrimWhitespace(const std::string& value) {
  std::size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

bool TryParseInteger(const std::string& value, int* parsed_value) {
  if (parsed_value == nullptr) {
    return false;
  }
  const std::string trimmed = TrimWhitespace(value);
  if (trimmed.empty()) {
    return false;
  }
  char* end_ptr = nullptr;
  const long parsed = std::strtol(trimmed.c_str(), &end_ptr, 10);
  if (end_ptr == trimmed.c_str() || *end_ptr != '\0') {
    return false;
  }
  *parsed_value = static_cast<int>(parsed);
  return true;
}

void LoadForceIncludeFrameBoardList(
    const std::string& path,
    ati::TrialBackendFrameBoardSelectionOptions* options) {
  if (path.empty() || options == nullptr) {
    return;
  }
  std::ifstream input(path.c_str());
  if (!input) {
    throw std::runtime_error(
        "failed to open force-include frame-board list: " + path);
  }
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    line = TrimWhitespace(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::stringstream stream(line);
    std::string first;
    std::string second;
    if (!std::getline(stream, first, ',') ||
        !std::getline(stream, second, ',')) {
      throw std::runtime_error(
          "invalid force-include CSV row at line " +
          std::to_string(line_number) + ": " + line);
    }
    first = TrimWhitespace(first);
    second = TrimWhitespace(second);
    if ((first == "frame_index" || first == "frame_label") &&
        second == "board_id") {
      continue;
    }
    int board_id = -1;
    if (!TryParseInteger(second, &board_id)) {
      throw std::runtime_error(
          "invalid board_id in force-include CSV at line " +
          std::to_string(line_number) + ": " + line);
    }
    int frame_index = -1;
    if (TryParseInteger(first, &frame_index)) {
      options->force_include_frame_board_keys.insert(
          std::make_pair(frame_index, board_id));
    } else {
      options->force_include_frame_label_board_keys.insert(
          std::make_pair(first, board_id));
    }
  }
}


struct InternalBlurBoardRow {
  std::string split;
  int round_index = -1;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int internal_point_count = 0;
  int valid_internal_point_count = 0;
  int attempted_internal_point_count = 0;
  double valid_internal_ratio = 0.0;
  double internal_rmse = 0.0;
  double mean_residual_x = 0.0;
  double mean_residual_y = 0.0;
  double std_residual_x = 0.0;
  double std_residual_y = 0.0;
  double max_residual = 0.0;
  double p90_residual = 0.0;
  double p95_residual = 0.0;
  double board_center_radius = 0.0;
  double board_crop_sharpness_score = 0.0;
  double board_crop_laplacian_variance = 0.0;
  double mean_gradient_magnitude = 0.0;
  double corner_patch_mean_gradient = 0.0;
  double ghosting_or_double_edge_score = -1.0;
  bool crop_valid = false;
  std::string failure_reason;
  int used_in_backend = 0;
  int used_in_evaluation = 0;
};

struct InternalBlurFrameRow {
  std::string split;
  int round_index = -1;
  int frame_index = -1;
  std::string frame_label;
  int board_count = 0;
  int internal_point_count = 0;
  int valid_internal_point_count = 0;
  double internal_rmse = 0.0;
  double max_residual = 0.0;
  double p95_residual = 0.0;
  double mean_board_crop_sharpness_score = 0.0;
  double mean_laplacian_variance = 0.0;
  double mean_gradient_magnitude = 0.0;
  int low_sharpness_high_rmse_board_count = 0;
};

double Quantile(std::vector<double> values, double q) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double scaled = q * static_cast<double>(values.size() - 1);
  const std::size_t index =
      static_cast<std::size_t>(std::max(0.0, std::min(scaled, static_cast<double>(values.size() - 1))));
  return values[index];
}

std::map<int, std::string> BuildFrameImagePathMap(
    const std::vector<ati::FrozenRound2BaselineFrameSource>& frames) {
  std::map<int, std::string> paths;
  for (const ati::FrozenRound2BaselineFrameSource& frame : frames) {
    if (!frame.image_path.empty()) {
      paths[frame.frame_index] = frame.image_path;
    }
  }
  return paths;
}

void WriteAutoCameraInitializationBootstrapCornerOverlays(
    const fs::path& output_dir,
    const ati::AutoCameraInitializationResult& initialization,
    const std::vector<ati::FrozenRound2BaselineFrameSource>& frame_sources) {
  if (initialization.lm_bootstrap_observations.empty()) {
    return;
  }
  EnsureDirectoryExists(output_dir);
  const std::map<int, std::string> image_paths =
      BuildFrameImagePathMap(frame_sources);
  std::map<int, std::vector<const ati::AutoCameraInitializationBootstrapObservation*> >
      observations_by_frame;
  for (const ati::AutoCameraInitializationBootstrapObservation& observation :
       initialization.lm_bootstrap_observations) {
    observations_by_frame[observation.frame_index].push_back(&observation);
  }

  std::ofstream manifest((output_dir / "manifest.csv").string().c_str());
  manifest << "frame_index,frame_label,image_path,overlay_png,board_count,board_ids\n";
  for (const auto& entry : observations_by_frame) {
    const int frame_index = entry.first;
    const auto image_path_it = image_paths.find(frame_index);
    if (image_path_it == image_paths.end()) {
      continue;
    }
    cv::Mat image = cv::imread(image_path_it->second, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      continue;
    }
    cv::Mat overlay;
    if (image.channels() == 1) {
      cv::cvtColor(image, overlay, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
      cv::cvtColor(image, overlay, cv::COLOR_BGRA2BGR);
    } else {
      overlay = image.clone();
    }

    std::set<int> board_ids;
    std::string frame_label;
    for (const ati::AutoCameraInitializationBootstrapObservation* observation :
         entry.second) {
      if (observation == nullptr) {
        continue;
      }
      frame_label = observation->frame_label;
      board_ids.insert(observation->board_id);
      std::vector<cv::Point> polygon;
      polygon.reserve(4);
      for (int corner_index = 0; corner_index < 4; ++corner_index) {
        const Eigen::Vector2d& corner =
            observation->outer_corners[static_cast<std::size_t>(corner_index)];
        polygon.push_back(cv::Point(static_cast<int>(std::lround(corner.x())),
                                    static_cast<int>(std::lround(corner.y()))));
      }
      const cv::Scalar line_color(0, 220, 255);
      const cv::Scalar corner_color(0, 255, 80);
      const cv::Scalar text_color(255, 255, 255);
      const cv::Scalar text_shadow(0, 0, 0);
      for (int index = 0; index < 4; ++index) {
        cv::line(overlay, polygon[static_cast<std::size_t>(index)],
                 polygon[static_cast<std::size_t>((index + 1) % 4)],
                 line_color, 3, cv::LINE_AA);
      }
      for (int index = 0; index < 4; ++index) {
        cv::circle(overlay, polygon[static_cast<std::size_t>(index)],
                   7, corner_color, -1, cv::LINE_AA);
        cv::circle(overlay, polygon[static_cast<std::size_t>(index)],
                   7, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
      }
      cv::Point label_point = polygon.front() + cv::Point(12, -12);
      label_point.x = std::max(8, std::min(label_point.x, overlay.cols - 160));
      label_point.y = std::max(24, std::min(label_point.y, overlay.rows - 8));
      std::ostringstream label_stream;
      label_stream << "init board " << observation->board_id
                   << " lm=" << (observation->used_in_lm ? 1 : 0)
                   << " pose=" << (observation->pose_init_success ? 1 : 0);
      if (std::isfinite(observation->pose_fit_outer_rmse)) {
        label_stream << " rmse=" << std::fixed << std::setprecision(2)
                     << observation->pose_fit_outer_rmse;
      }
      const std::string label = label_stream.str();
      cv::putText(overlay, label, label_point + cv::Point(1, 1),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, text_shadow, 3, cv::LINE_AA);
      cv::putText(overlay, label, label_point,
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv::LINE_AA);
    }

    const std::string stem =
        "frame_" + std::to_string(frame_index) + "_" +
        SanitizeFilenameComponent(frame_label.empty()
                                      ? std::string("unknown")
                                      : frame_label);
    const std::string filename = stem + "_bootstrap_corners.png";
    cv::imwrite((output_dir / filename).string(), overlay);

    std::ostringstream board_stream;
    std::size_t offset = 0;
    for (int board_id : board_ids) {
      if (offset++ > 0) {
        board_stream << "|";
      }
      board_stream << board_id;
    }
    manifest << frame_index << ","
             << frame_label << ","
             << image_path_it->second << ","
             << filename << ","
             << board_ids.size() << ","
             << board_stream.str() << "\n";
  }
}

std::map<std::pair<int, int>, const ati::RegeneratedBoardMeasurement*>
BuildRegeneratedBoardMap(
    const std::vector<ati::InternalRegenerationFrameResult>& frames) {
  std::map<std::pair<int, int>, const ati::RegeneratedBoardMeasurement*> map;
  for (const ati::InternalRegenerationFrameResult& frame : frames) {
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      map[std::make_pair(frame.frame_index, measurement.board_id)] =
          &measurement;
    }
  }
  return map;
}

std::set<std::pair<int, int> > BuildBackendInternalBoardSet(
    const ati::CalibrationBackendProblemInput& backend_problem_input) {
  std::set<std::pair<int, int> > keys;
  for (const ati::JointPointObservation& point :
       backend_problem_input.measurement_dataset.solver_observations) {
    if (point.used_in_solver && point.point_type == ati::JointPointType::Internal) {
      keys.insert(std::make_pair(point.frame_index, point.board_id));
    }
  }
  return keys;
}

bool ComputeBoardCropMetrics(const cv::Mat& image,
                             const std::array<cv::Point2f, 4>& outer_corners,
                             double* center_radius,
                             double* laplacian_variance,
                             double* mean_gradient_magnitude,
                             std::string* failure_reason) {
  if (image.empty()) {
    if (failure_reason != nullptr) {
      *failure_reason = "image_empty";
    }
    return false;
  }
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }

  std::vector<cv::Point2f> corners(outer_corners.begin(), outer_corners.end());
  cv::Rect rect = cv::boundingRect(corners);
  const int margin =
      std::max(8, static_cast<int>(std::round(0.15 * std::max(rect.width, rect.height))));
  rect.x -= margin;
  rect.y -= margin;
  rect.width += 2 * margin;
  rect.height += 2 * margin;
  rect &= cv::Rect(0, 0, gray.cols, gray.rows);
  if (rect.width < 8 || rect.height < 8) {
    if (failure_reason != nullptr) {
      *failure_reason = "board_crop_too_small";
    }
    return false;
  }

  cv::Mat crop = gray(rect);
  cv::Mat laplacian;
  cv::Laplacian(crop, laplacian, CV_64F);
  cv::Scalar lap_mean;
  cv::Scalar lap_std;
  cv::meanStdDev(laplacian, lap_mean, lap_std);
  if (laplacian_variance != nullptr) {
    *laplacian_variance = lap_std[0] * lap_std[0];
  }

  cv::Mat grad_x;
  cv::Mat grad_y;
  cv::Sobel(crop, grad_x, CV_64F, 1, 0, 3);
  cv::Sobel(crop, grad_y, CV_64F, 0, 1, 3);
  cv::Mat magnitude;
  cv::magnitude(grad_x, grad_y, magnitude);
  if (mean_gradient_magnitude != nullptr) {
    *mean_gradient_magnitude = cv::mean(magnitude)[0];
  }

  if (center_radius != nullptr) {
    cv::Point2f center(0.0f, 0.0f);
    for (const cv::Point2f& corner : corners) {
      center += corner;
    }
    center *= 0.25f;
    const double dx = static_cast<double>(center.x) - 0.5 * gray.cols;
    const double dy = static_cast<double>(center.y) - 0.5 * gray.rows;
    const double half_diagonal =
        0.5 * std::sqrt(static_cast<double>(gray.cols * gray.cols + gray.rows * gray.rows));
    *center_radius = half_diagonal > 0.0 ? std::sqrt(dx * dx + dy * dy) / half_diagonal : 0.0;
  }
  return true;
}

double ComputeCornerPatchMeanGradient(
    const cv::Mat& image,
    const std::vector<ati::CameraModelRefitPointDiagnostics>& points) {
  if (image.empty() || points.empty()) {
    return 0.0;
  }
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Mat grad_x;
  cv::Mat grad_y;
  cv::Sobel(gray, grad_x, CV_64F, 1, 0, 3);
  cv::Sobel(gray, grad_y, CV_64F, 0, 1, 3);
  cv::Mat magnitude;
  cv::magnitude(grad_x, grad_y, magnitude);

  double sum = 0.0;
  int count = 0;
  for (const ati::CameraModelRefitPointDiagnostics& point : points) {
    const int x = static_cast<int>(std::round(point.observed_image_xy.x()));
    const int y = static_cast<int>(std::round(point.observed_image_xy.y()));
    cv::Rect rect(x - 4, y - 4, 9, 9);
    rect &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (rect.width <= 0 || rect.height <= 0) {
      continue;
    }
    sum += cv::mean(magnitude(rect))[0];
    ++count;
  }
  return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

std::vector<InternalBlurBoardRow> BuildInternalBlurBoardRowsForSplit(
    const std::string& split,
    int round_index,
    const ati::CalibrationEvaluationDataset& dataset,
    const ati::CameraModelRefitEvaluationResult& evaluation,
    const std::map<int, std::string>& image_paths,
    const ati::CalibrationBackendProblemInput& backend_problem_input) {
  std::map<std::pair<int, int>, std::vector<ati::CameraModelRefitPointDiagnostics> > points_by_board;
  for (const ati::CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.point_type != ati::JointPointType::Internal) {
      continue;
    }
    points_by_board[std::make_pair(point.frame_index, point.board_id)].push_back(point);
  }
  std::map<int, std::string> regeneration_frame_labels;
  for (const ati::InternalRegenerationFrameResult& frame :
       dataset.internal_regeneration_results) {
    regeneration_frame_labels[frame.frame_index] = frame.frame_label;
    for (const ati::RegeneratedBoardMeasurement& board :
         frame.board_measurements) {
      points_by_board[std::make_pair(frame.frame_index, board.board_id)];
    }
  }
  const auto regenerated_by_board =
      BuildRegeneratedBoardMap(dataset.internal_regeneration_results);
  const auto backend_internal_boards =
      BuildBackendInternalBoardSet(backend_problem_input);

  std::vector<InternalBlurBoardRow> rows;
  rows.reserve(points_by_board.size());
  std::map<int, cv::Mat> image_cache;
  for (const auto& entry : points_by_board) {
    const std::pair<int, int>& key = entry.first;
    const std::vector<ati::CameraModelRefitPointDiagnostics>& points = entry.second;
    InternalBlurBoardRow row;
    row.split = split;
    row.round_index = round_index;
    row.frame_index = key.first;
    row.board_id = key.second;
    row.frame_label = points.empty() ? std::string() : points.front().frame_label;
    if (row.frame_label.empty()) {
      const auto label_it = regeneration_frame_labels.find(row.frame_index);
      if (label_it != regeneration_frame_labels.end()) {
        row.frame_label = label_it->second;
      }
    }
    row.internal_point_count = static_cast<int>(points.size());
    row.used_in_evaluation = row.internal_point_count > 0 ? 1 : 0;
    row.used_in_backend =
        backend_internal_boards.count(key) > 0 && split == "training" ? 1 : 0;

    std::vector<double> norms;
    double sq_sum = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (const ati::CameraModelRefitPointDiagnostics& point : points) {
      norms.push_back(point.residual_norm);
      sq_sum += point.residual_norm * point.residual_norm;
      sum_x += point.residual_xy.x();
      sum_y += point.residual_xy.y();
    }
    if (!points.empty()) {
      row.internal_rmse =
          std::sqrt(sq_sum / static_cast<double>(points.size()));
      row.mean_residual_x = sum_x / static_cast<double>(points.size());
      row.mean_residual_y = sum_y / static_cast<double>(points.size());
      double var_x = 0.0;
      double var_y = 0.0;
      for (const ati::CameraModelRefitPointDiagnostics& point : points) {
        const double dx = point.residual_xy.x() - row.mean_residual_x;
        const double dy = point.residual_xy.y() - row.mean_residual_y;
        var_x += dx * dx;
        var_y += dy * dy;
      }
      row.std_residual_x =
          std::sqrt(var_x / static_cast<double>(points.size()));
      row.std_residual_y =
          std::sqrt(var_y / static_cast<double>(points.size()));
      row.max_residual = *std::max_element(norms.begin(), norms.end());
      row.p90_residual = Quantile(norms, 0.90);
      row.p95_residual = Quantile(norms, 0.95);
    }

    const auto regen_it = regenerated_by_board.find(key);
    if (regen_it != regenerated_by_board.end()) {
      const ati::ApriltagInternalDetectionResult& detection =
          regen_it->second->detection;
      row.valid_internal_point_count = detection.valid_internal_corner_count;
      row.attempted_internal_point_count =
          detection.runtime_breakdown.attempted_internal_corner_count;
      row.valid_internal_ratio =
          SafeRatio(row.valid_internal_point_count,
                    row.attempted_internal_point_count);
      if (!detection.success) {
        row.failure_reason = detection.failure_reason;
      }

      const auto image_path_it = image_paths.find(row.frame_index);
      if (image_path_it != image_paths.end()) {
        cv::Mat& image = image_cache[row.frame_index];
        if (image.empty()) {
          image = cv::imread(image_path_it->second, cv::IMREAD_UNCHANGED);
        }
        std::string crop_failure;
        row.crop_valid = ComputeBoardCropMetrics(
            image, detection.outer_corners, &row.board_center_radius,
            &row.board_crop_laplacian_variance,
            &row.mean_gradient_magnitude, &crop_failure);
        row.board_crop_sharpness_score = row.board_crop_laplacian_variance;
        row.corner_patch_mean_gradient =
            ComputeCornerPatchMeanGradient(image, points);
        if (!row.crop_valid && row.failure_reason.empty()) {
          row.failure_reason = crop_failure;
        }
      } else if (row.failure_reason.empty()) {
        row.failure_reason = "image_path_not_found";
      }
    } else {
      row.failure_reason = "regeneration_result_not_found";
    }
    rows.push_back(row);
  }
  return rows;
}

std::vector<InternalBlurFrameRow> BuildInternalBlurFrameRows(
    const std::vector<InternalBlurBoardRow>& board_rows) {
  std::map<std::pair<std::string, int>, InternalBlurFrameRow> frames;
  for (const InternalBlurBoardRow& board : board_rows) {
    InternalBlurFrameRow& frame =
        frames[std::make_pair(board.split, board.frame_index)];
    frame.split = board.split;
    frame.round_index = board.round_index;
    frame.frame_index = board.frame_index;
    frame.frame_label = board.frame_label;
    ++frame.board_count;
    frame.internal_point_count += board.internal_point_count;
    frame.valid_internal_point_count += board.valid_internal_point_count;
    frame.max_residual = std::max(frame.max_residual, board.max_residual);
    frame.mean_board_crop_sharpness_score += board.board_crop_sharpness_score;
    frame.mean_laplacian_variance += board.board_crop_laplacian_variance;
    frame.mean_gradient_magnitude += board.mean_gradient_magnitude;
  }

  std::map<std::pair<std::string, int>, std::vector<double> > residuals_by_frame;
  for (const InternalBlurBoardRow& board : board_rows) {
    residuals_by_frame[std::make_pair(board.split, board.frame_index)].push_back(
        board.internal_rmse);
  }

  std::vector<InternalBlurFrameRow> rows;
  rows.reserve(frames.size());
  for (auto& entry : frames) {
    InternalBlurFrameRow frame = entry.second;
    const std::vector<double>& rmses = residuals_by_frame[entry.first];
    double sq_sum = 0.0;
    for (double rmse : rmses) {
      sq_sum += rmse * rmse;
    }
    frame.internal_rmse =
        rmses.empty() ? 0.0
                      : std::sqrt(sq_sum / static_cast<double>(rmses.size()));
    frame.p95_residual = Quantile(rmses, 0.95);
    if (frame.board_count > 0) {
      frame.mean_board_crop_sharpness_score /= frame.board_count;
      frame.mean_laplacian_variance /= frame.board_count;
      frame.mean_gradient_magnitude /= frame.board_count;
    }
    rows.push_back(frame);
  }
  return rows;
}

std::string RadiusBin(double radius) {
  if (radius < 0.33) {
    return "center";
  }
  if (radius < 0.55) {
    return "mid";
  }
  if (radius < 0.75) {
    return "edge";
  }
  return "extreme_edge";
}

std::string SharpnessBin(double value, double low_quantile, double high_quantile) {
  if (value <= low_quantile) {
    return "low";
  }
  if (value >= high_quantile) {
    return "high";
  }
  return "medium";
}

void WriteInternalBlurDiagnostics(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report,
    const ati::CameraModelRefitEvaluationResult& backend_training_evaluation,
    const ati::CameraModelRefitEvaluationResult& backend_holdout_evaluation,
    const std::vector<ati::FrozenRound2BaselineFrameSource>& all_frames) {
  const std::map<int, std::string> image_paths = BuildFrameImagePathMap(all_frames);
  std::vector<InternalBlurBoardRow> board_rows =
      BuildInternalBlurBoardRowsForSplit(
          "training", report.baseline_result.final_stage5_bundle.round_index,
          report.training_dataset, backend_training_evaluation, image_paths,
          report.backend_problem_input);
  const std::vector<InternalBlurBoardRow> holdout_board_rows =
      BuildInternalBlurBoardRowsForSplit(
          "holdout", report.baseline_result.final_stage5_bundle.round_index,
          report.holdout_dataset, backend_holdout_evaluation, image_paths,
          report.backend_problem_input);
  board_rows.insert(board_rows.end(), holdout_board_rows.begin(),
                    holdout_board_rows.end());
  std::vector<InternalBlurFrameRow> frame_rows =
      BuildInternalBlurFrameRows(board_rows);

  std::vector<double> sharpness_values;
  sharpness_values.reserve(board_rows.size());
  for (const InternalBlurBoardRow& row : board_rows) {
    if (row.crop_valid) {
      sharpness_values.push_back(row.board_crop_sharpness_score);
    }
  }
  const double low_sharpness = Quantile(sharpness_values, 0.33);
  const double high_sharpness = Quantile(sharpness_values, 0.67);

  for (InternalBlurFrameRow& frame : frame_rows) {
    for (const InternalBlurBoardRow& board : board_rows) {
      if (board.split == frame.split && board.frame_index == frame.frame_index &&
          SharpnessBin(board.board_crop_sharpness_score, low_sharpness,
                       high_sharpness) == "low" &&
          board.internal_rmse >= 6.0) {
        ++frame.low_sharpness_high_rmse_board_count;
      }
    }
  }

  std::ofstream by_board(
      (output_dir / "internal_blur_diagnostics_by_board.csv").string().c_str());
  by_board
      << "split,round_index,frame_index,frame_label,board_id,"
      << "internal_point_count,valid_internal_point_count,"
      << "attempted_internal_point_count,valid_internal_ratio,internal_rmse,"
      << "mean_residual_x,mean_residual_y,std_residual_x,std_residual_y,"
      << "max_residual,p90_residual,p95_residual,board_center_radius,"
      << "radius_bin,sharpness_bin,board_crop_sharpness_score,"
      << "board_crop_laplacian_variance,mean_gradient_magnitude,"
      << "corner_patch_mean_gradient,ghosting_or_double_edge_score,"
      << "crop_valid,failure_reason,used_in_backend,used_in_evaluation\n";
  for (const InternalBlurBoardRow& row : board_rows) {
    by_board
        << row.split << ","
        << row.round_index << ","
        << row.frame_index << ","
        << CsvEscape(row.frame_label) << ","
        << row.board_id << ","
        << row.internal_point_count << ","
        << row.valid_internal_point_count << ","
        << row.attempted_internal_point_count << ","
        << row.valid_internal_ratio << ","
        << row.internal_rmse << ","
        << row.mean_residual_x << ","
        << row.mean_residual_y << ","
        << row.std_residual_x << ","
        << row.std_residual_y << ","
        << row.max_residual << ","
        << row.p90_residual << ","
        << row.p95_residual << ","
        << row.board_center_radius << ","
        << RadiusBin(row.board_center_radius) << ","
        << SharpnessBin(row.board_crop_sharpness_score, low_sharpness,
                        high_sharpness) << ","
        << row.board_crop_sharpness_score << ","
        << row.board_crop_laplacian_variance << ","
        << row.mean_gradient_magnitude << ","
        << row.corner_patch_mean_gradient << ","
        << row.ghosting_or_double_edge_score << ","
        << (row.crop_valid ? 1 : 0) << ","
        << CsvEscape(row.failure_reason) << ","
        << row.used_in_backend << ","
        << row.used_in_evaluation << "\n";
  }

  std::ofstream by_frame(
      (output_dir / "internal_blur_diagnostics_by_frame.csv").string().c_str());
  by_frame
      << "split,round_index,frame_index,frame_label,board_count,"
      << "internal_point_count,valid_internal_point_count,internal_rmse,"
      << "max_residual,p95_residual,mean_board_crop_sharpness_score,"
      << "mean_laplacian_variance,mean_gradient_magnitude,"
      << "low_sharpness_high_rmse_board_count\n";
  for (const InternalBlurFrameRow& row : frame_rows) {
    by_frame
        << row.split << ","
        << row.round_index << ","
        << row.frame_index << ","
        << CsvEscape(row.frame_label) << ","
        << row.board_count << ","
        << row.internal_point_count << ","
        << row.valid_internal_point_count << ","
        << row.internal_rmse << ","
        << row.max_residual << ","
        << row.p95_residual << ","
        << row.mean_board_crop_sharpness_score << ","
        << row.mean_laplacian_variance << ","
        << row.mean_gradient_magnitude << ","
        << row.low_sharpness_high_rmse_board_count << "\n";
  }

  std::vector<InternalBlurBoardRow> worst_boards = board_rows;
  std::sort(worst_boards.begin(), worst_boards.end(),
            [](const InternalBlurBoardRow& lhs, const InternalBlurBoardRow& rhs) {
              return lhs.internal_rmse > rhs.internal_rmse;
            });
  std::vector<InternalBlurFrameRow> worst_frames = frame_rows;
  std::sort(worst_frames.begin(), worst_frames.end(),
            [](const InternalBlurFrameRow& lhs, const InternalBlurFrameRow& rhs) {
              return lhs.internal_rmse > rhs.internal_rmse;
            });

  std::ofstream worst(
      (output_dir / "internal_blur_worst_cases.csv").string().c_str());
  worst << "case_type,rank,split,round_index,frame_index,frame_label,board_id,"
        << "internal_rmse,max_residual,p95_residual,sharpness_score,"
        << "mean_gradient_magnitude,board_center_radius,radius_bin,sharpness_bin\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(30, worst_boards.size()); ++i) {
    const InternalBlurBoardRow& row = worst_boards[i];
    worst << "board,"
          << (i + 1) << ","
          << row.split << ","
          << row.round_index << ","
          << row.frame_index << ","
          << CsvEscape(row.frame_label) << ","
          << row.board_id << ","
          << row.internal_rmse << ","
          << row.max_residual << ","
          << row.p95_residual << ","
          << row.board_crop_sharpness_score << ","
          << row.mean_gradient_magnitude << ","
          << row.board_center_radius << ","
          << RadiusBin(row.board_center_radius) << ","
          << SharpnessBin(row.board_crop_sharpness_score, low_sharpness,
                          high_sharpness) << "\n";
  }
  for (std::size_t i = 0; i < std::min<std::size_t>(30, worst_frames.size()); ++i) {
    const InternalBlurFrameRow& row = worst_frames[i];
    worst << "frame,"
          << (i + 1) << ","
          << row.split << ","
          << row.round_index << ","
          << row.frame_index << ","
          << CsvEscape(row.frame_label) << ","
          << -1 << ","
          << row.internal_rmse << ","
          << row.max_residual << ","
          << row.p95_residual << ","
          << row.mean_board_crop_sharpness_score << ","
          << row.mean_gradient_magnitude << ","
          << -1 << ","
          << "frame,"
          << SharpnessBin(row.mean_board_crop_sharpness_score, low_sharpness,
                          high_sharpness) << "\n";
  }

  std::map<int, std::pair<double, int> > rmse_by_board_id;
  std::map<std::string, std::pair<double, int> > rmse_by_radius_bin;
  std::map<std::string, std::pair<double, int> > rmse_by_sharpness_bin;
  int low_sharpness_high_rmse_count = 0;
  for (const InternalBlurBoardRow& row : board_rows) {
    rmse_by_board_id[row.board_id].first += row.internal_rmse;
    rmse_by_board_id[row.board_id].second += 1;
    rmse_by_radius_bin[RadiusBin(row.board_center_radius)].first += row.internal_rmse;
    rmse_by_radius_bin[RadiusBin(row.board_center_radius)].second += 1;
    const std::string sharpness_bin =
        SharpnessBin(row.board_crop_sharpness_score, low_sharpness,
                     high_sharpness);
    rmse_by_sharpness_bin[sharpness_bin].first += row.internal_rmse;
    rmse_by_sharpness_bin[sharpness_bin].second += 1;
    if (sharpness_bin == "low" && row.internal_rmse >= 6.0) {
      ++low_sharpness_high_rmse_count;
    }
  }

  std::ofstream summary(
      (output_dir / "internal_blur_diagnostics_summary.txt").string().c_str());
  summary << "enabled: 1\n";
  summary << "board_record_count: " << board_rows.size() << "\n";
  summary << "frame_record_count: " << frame_rows.size() << "\n";
  summary << "sharpness_low_quantile: " << low_sharpness << "\n";
  summary << "sharpness_high_quantile: " << high_sharpness << "\n";
  summary << "low_sharpness_high_internal_rmse_board_count: "
          << low_sharpness_high_rmse_count << "\n";
  summary << "ghosting_or_double_edge_score: not_computed_v1\n";
  summary << "\n[average_internal_rmse_by_board_id]\n";
  for (const auto& entry : rmse_by_board_id) {
    summary << "board_" << entry.first << ": "
            << entry.second.first / std::max(1, entry.second.second)
            << " count=" << entry.second.second << "\n";
  }
  summary << "\n[average_internal_rmse_by_radius_bin]\n";
  for (const auto& entry : rmse_by_radius_bin) {
    summary << entry.first << ": "
            << entry.second.first / std::max(1, entry.second.second)
            << " count=" << entry.second.second << "\n";
  }
  summary << "\n[average_internal_rmse_by_sharpness_bin]\n";
  for (const auto& entry : rmse_by_sharpness_bin) {
    summary << entry.first << ": "
            << entry.second.first / std::max(1, entry.second.second)
            << " count=" << entry.second.second << "\n";
  }
}

void WriteBackendDiagnosticArtifacts(const fs::path& output_dir,
                                     const std::string& prefix,
                                     const ati::AslamBackendCalibrationResult& result) {
  ati::WriteAslamBackendCalibrationSummary(
      (output_dir / (prefix + "_summary.txt")).string(), result);
  ati::WriteBackendResidualTypeAssignmentsCsv(
      (output_dir / (prefix + "_residual_type_per_point.csv")).string(),
      result);
  if (result.initial_cost_parity.success || !result.initial_cost_parity.failure_reason.empty()) {
    ati::WriteAslamBackendCostParitySummary(
        (output_dir / (prefix + "_cost_parity_initial_summary.txt")).string(),
        result.initial_cost_parity);
    ati::WriteAslamBackendCostParityCsv(
        (output_dir / (prefix + "_cost_parity_initial_points.csv")).string(),
        result.initial_cost_parity);
  }
  if (result.optimized_cost_parity.success ||
      !result.optimized_cost_parity.failure_reason.empty()) {
    ati::WriteAslamBackendCostParitySummary(
        (output_dir / (prefix + "_cost_parity_optimized_summary.txt")).string(),
        result.optimized_cost_parity);
    ati::WriteAslamBackendCostParityCsv(
        (output_dir / (prefix + "_cost_parity_optimized_points.csv")).string(),
        result.optimized_cost_parity);
  }
  if (result.jacobian_diagnostics.success ||
      !result.jacobian_diagnostics.failure_reason.empty()) {
    ati::WriteAslamBackendJacobianSummary(
        (output_dir / (prefix + "_jacobian_summary.txt")).string(),
        result.jacobian_diagnostics);
  }
  if (result.initial_variable_block_influence.success ||
      !result.initial_variable_block_influence.failure_reason.empty()) {
    ati::WriteAslamBackendVariableBlockInfluenceCsv(
        (output_dir / (prefix + "_variable_block_influence_initial.csv")).string(),
        result.initial_variable_block_influence);
  }
  if (result.optimized_variable_block_influence.success ||
      !result.optimized_variable_block_influence.failure_reason.empty()) {
    ati::WriteAslamBackendVariableBlockInfluenceCsv(
        (output_dir / (prefix + "_variable_block_influence_optimized.csv")).string(),
        result.optimized_variable_block_influence);
  }
}

struct PixelRayHybridLambdaBinAccumulator {
  int count = 0;
  double sum = 0.0;
  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();

  void Add(double value) {
    if (!std::isfinite(value)) {
      return;
    }
    ++count;
    sum += value;
    min = std::min(min, value);
    max = std::max(max, value);
  }
};

void WritePixelRayHybridAdaptiveDiagnostics(
    const fs::path& output_dir,
    const ati::AslamBackendCalibrationResult& backend_result,
    const ati::AngularResidualDiagnosticsResult& angular_diagnostics,
    const ati::CameraModelRefitEvaluationResult& holdout_evaluation) {
  std::ofstream observations(
      (output_dir / "hybrid_pixel_ray_adaptive_observations.csv").string().c_str());
  observations << "frame_index,frame_label,board_id,point_id,point_type,"
               << "pre_refinement_polar_angle_deg,lambda_i,"
               << "polar_adaptive_enabled\n";
  for (const ati::BackendResidualTypeAssignment& assignment :
       backend_result.residual_type_assignments) {
    if (assignment.residual_model_effective !=
        "pixel_ray_hybrid_final_refinement") {
      continue;
    }
    observations << assignment.frame_index << ","
                 << assignment.frame_label << ","
                 << assignment.board_id << ","
                 << assignment.point_id << ","
                 << ati::ToString(assignment.point_type) << ","
                 << assignment.polar_angle_deg << ","
                 << assignment.pixel_ray_hybrid_lambda << ","
                 << (assignment.pixel_ray_hybrid_polar_adaptive ? 1 : 0)
                 << "\n";
  }

  const std::vector<double>& edges =
      backend_result.options.angular_residual_bin_edges_deg;
  std::vector<PixelRayHybridLambdaBinAccumulator> lambda_bins;
  if (edges.size() >= 2) {
    lambda_bins.resize(edges.size() - 1);
    for (const ati::BackendResidualTypeAssignment& assignment :
         backend_result.residual_type_assignments) {
      if (assignment.residual_model_effective !=
              "pixel_ray_hybrid_final_refinement" ||
          !std::isfinite(assignment.polar_angle_deg)) {
        continue;
      }
      for (std::size_t index = 0; index + 1 < edges.size(); ++index) {
        const bool last_bin = index + 2 == edges.size();
        if (assignment.polar_angle_deg >= edges[index] &&
            (assignment.polar_angle_deg < edges[index + 1] ||
             (last_bin && assignment.polar_angle_deg <= edges[index + 1]))) {
          lambda_bins[index].Add(assignment.pixel_ray_hybrid_lambda);
          break;
        }
      }
    }
  }

  std::ofstream bins(
      (output_dir / "hybrid_pixel_ray_adaptive_bin_summary.csv").string().c_str());
  bins << "bin_min_deg,bin_max_deg,pre_refinement_observation_count,"
       << "lambda_min,lambda_mean,lambda_max,"
       << "final_training_pixel_rmse_px,final_training_angular_rmse_rad\n";
  const std::size_t bin_count = std::min(lambda_bins.size(),
                                         angular_diagnostics.all_points_bins.size());
  for (std::size_t index = 0; index < bin_count; ++index) {
    const PixelRayHybridLambdaBinAccumulator& lambda_bin = lambda_bins[index];
    const ati::AngularResidualBinStatistics& residual_bin =
        angular_diagnostics.all_points_bins[index];
    bins << residual_bin.bin_min_deg << "," << residual_bin.bin_max_deg << ","
         << lambda_bin.count << ","
         << (lambda_bin.count > 0 ? lambda_bin.min : 0.0) << ","
         << (lambda_bin.count > 0
                 ? lambda_bin.sum / static_cast<double>(lambda_bin.count)
                 : 0.0)
         << "," << (lambda_bin.count > 0 ? lambda_bin.max : 0.0) << ","
         << residual_bin.image_plane_rmse << "," << residual_bin.rmse << "\n";
  }

  std::ofstream heldout(
      (output_dir / "hybrid_pixel_ray_adaptive_heldout_metrics.csv").string().c_str());
  heldout << "polar_adaptive_enabled,lambda_min,lambda_max,"
          << "transition_start_deg,transition_end_deg,"
          << "heldout_multiboard_rmse_px,heldout_angular_rmse_rad,"
          << "heldout_angular_rmse_deg,heldout_angular_point_count\n";
  heldout << (backend_result.options.pixel_ray_hybrid_polar_adaptive_enabled
                  ? 1
                  : 0)
          << "," << backend_result.options.pixel_ray_hybrid_lambda_min
          << "," << backend_result.options.pixel_ray_hybrid_lambda_max
          << "," << backend_result.options.pixel_ray_hybrid_transition_start_deg
          << "," << backend_result.options.pixel_ray_hybrid_transition_end_deg
          << "," << holdout_evaluation.overall_rmse
          << "," << holdout_evaluation.overall_angular_rmse_rad
          << "," << holdout_evaluation.overall_angular_rmse_deg
          << "," << holdout_evaluation.angular_point_count << "\n";
}

void PrintProgress(const std::string& message) {
  std::cout << "[stage5_backend] " << message << std::endl;
}

void ValidateBackendResidualConfiguration(
    const RequestedExperimentConfig& config) {
  const ati::ResidualModel model =
      ati::ParseResidualModel(config.backend_residual_model);
  ati::ParseResidualModel(config.backend_outer_residual_model);
  ati::ParseResidualModel(config.backend_internal_residual_model);
  auto require_finite_nonnegative = [](double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::runtime_error(std::string(name) +
                               " must be finite and non-negative.");
    }
  };
  require_finite_nonnegative(config.backend_pixel_residual_weight,
                             "--backend-pixel-residual-weight");
  require_finite_nonnegative(config.backend_chordal_residual_weight,
                             "--backend-chordal-residual-weight");
  if (!std::isfinite(config.backend_hybrid_angular_threshold_deg) ||
      config.backend_hybrid_angular_threshold_deg < 0.0 ||
      config.backend_hybrid_angular_threshold_deg > 180.0) {
    throw std::runtime_error(
        "--backend-hybrid-angular-threshold-deg must be in [0, 180].");
  }
  if (!std::isfinite(
          config.backend_polar_continuous_hybrid_threshold_deg) ||
      config.backend_polar_continuous_hybrid_threshold_deg < 0.0 ||
      config.backend_polar_continuous_hybrid_threshold_deg > 180.0) {
    throw std::runtime_error(
        "--backend-polar-continuous-hybrid-threshold-deg must be in [0, 180].");
  }
  if (!std::isfinite(
          config.backend_polar_continuous_hybrid_temperature_deg) ||
      config.backend_polar_continuous_hybrid_temperature_deg <= 0.0) {
    throw std::runtime_error(
        "--backend-polar-continuous-hybrid-temperature-deg must be positive.");
  }
  if (model == ati::ResidualModel::PixelChordalHybrid &&
      config.backend_pixel_residual_weight == 0.0 &&
      config.backend_chordal_residual_weight == 0.0) {
    throw std::runtime_error(
        "pixel_chordal_hybrid requires a positive pixel or chordal weight.");
  }
  if (config.enable_hybrid_pixel_ray_final_refinement) {
    if (model != ati::ResidualModel::ImagePlane ||
        config.backend_use_point_type_residual_split ||
        config.backend_enable_angular_auxiliary_residual) {
      throw std::runtime_error(
          "Hybrid Pixel-Ray final refinement requires a standard pixel-only "
          "persistent selection BA (image_plane, no residual split or "
          "angular auxiliary block).");
    }
    if (config.backend_angular_observed_ray_mode !=
        "dynamic_current_camera") {
      throw std::runtime_error(
          "Hybrid Pixel-Ray final refinement requires "
          "dynamic_current_camera observed rays.");
    }
    if (!std::isfinite(config.hybrid_pixel_ray_lambda) ||
        config.hybrid_pixel_ray_lambda < 0.0 ||
        config.hybrid_pixel_ray_lambda > 1.0) {
      throw std::runtime_error(
          "--stage5-hybrid-pixel-ray-lambda must be in [0, 1].");
    }
    const bool valid_adaptive_lambdas =
        std::isfinite(config.hybrid_pixel_ray_lambda_min) &&
        std::isfinite(config.hybrid_pixel_ray_lambda_max) &&
        config.hybrid_pixel_ray_lambda_min >= 0.0 &&
        config.hybrid_pixel_ray_lambda_min <= 1.0 &&
        config.hybrid_pixel_ray_lambda_max >= 0.0 &&
        config.hybrid_pixel_ray_lambda_max <= 1.0 &&
        config.hybrid_pixel_ray_lambda_min <=
            config.hybrid_pixel_ray_lambda_max;
    if (!valid_adaptive_lambdas) {
      throw std::runtime_error(
          "Polar-adaptive Hybrid lambda limits must satisfy "
          "0 <= lambda_min <= lambda_max <= 1.");
    }
    const bool valid_adaptive_transition =
        std::isfinite(config.hybrid_pixel_ray_transition_start_deg) &&
        std::isfinite(config.hybrid_pixel_ray_transition_end_deg) &&
        config.hybrid_pixel_ray_transition_start_deg >= 0.0 &&
        config.hybrid_pixel_ray_transition_end_deg <= 180.0 &&
        config.hybrid_pixel_ray_transition_end_deg >
            config.hybrid_pixel_ray_transition_start_deg;
    if (!valid_adaptive_transition) {
      throw std::runtime_error(
          "Polar-adaptive Hybrid transition angles must satisfy "
          "0 <= start < end <= 180 degrees.");
    }
    if (config.hybrid_pixel_ray_max_iterations <= 0) {
      throw std::runtime_error(
          "--stage5-hybrid-pixel-ray-max-iterations must be positive.");
    }
    if (!std::isfinite(config.hybrid_pixel_ray_pixel_scale_floor) ||
        config.hybrid_pixel_ray_pixel_scale_floor <= 0.0 ||
        !std::isfinite(config.hybrid_pixel_ray_ray_scale_floor) ||
        config.hybrid_pixel_ray_ray_scale_floor <= 0.0) {
      throw std::runtime_error(
          "Hybrid Pixel-Ray scale floors must be finite and positive.");
    }
  }
  if (config.backend_angular_local_whitening &&
      model != ati::ResidualModel::SphereAngular) {
    throw std::runtime_error(
        "--backend-angular-local-whitening requires sphere_angular.");
  }
  if (config.backend_angular_local_whitening &&
      (!std::isfinite(
           config.backend_angular_local_whitening_pixel_sigma_px) ||
       config.backend_angular_local_whitening_pixel_sigma_px <= 0.0 ||
       !std::isfinite(
           config.backend_angular_local_whitening_covariance_damping) ||
       config.backend_angular_local_whitening_covariance_damping < 0.0 ||
       !std::isfinite(
           config.backend_angular_local_whitening_min_sigma_rad) ||
       config.backend_angular_local_whitening_min_sigma_rad <= 0.0 ||
       !std::isfinite(config.backend_angular_local_whitening_max_weight) ||
       config.backend_angular_local_whitening_max_weight < 1.0)) {
    throw std::runtime_error(
        "invalid angular local-whitening configuration.");
  }
}

void PrintEvaluationProgress(
    const std::string& label,
    const ati::CameraModelRefitEvaluationResult& evaluation,
    const std::string& metric_unit = "px") {
  std::cout << "[stage5_backend] " << label
            << " overall=" << evaluation.overall_rmse;
  if (!evaluation.uniform_control_point_mode) {
    std::cout << " outer=" << evaluation.outer_only_rmse
              << " internal=" << evaluation.internal_only_rmse;
  }
  std::cout << " unit=" << metric_unit
            << " points=" << evaluation.point_count << std::endl;
}

std::string BackendResidualMetricUnit(ati::ResidualModel model) {
  if (model == ati::ResidualModel::ImagePlane) {
    return "px";
  }
  if (model == ati::ResidualModel::SphereAngular ||
      model == ati::ResidualModel::NormalizedSphereAngular) {
    return "rad";
  }
  if (model == ati::ResidualModel::Chordal) {
    return "unit_bearing_chord";
  }
  return "px_equivalent";
}

void PrintBackendResultProgress(
    const ati::AslamBackendCalibrationResult& result) {
  std::cout << "[stage5_backend] backend success=" << (result.success ? 1 : 0)
            << " initial_rmse=" << result.initial_residual.overall_rmse
            << " optimized_rmse=" << result.optimized_residual.overall_rmse
            << std::endl;
  if (result.options.multi_board_consistency_weighting) {
    std::cout << "[stage5_backend] consistency obs="
              << result.consistency_observation_count
              << " successful=" << result.consistency_successful_observation_count
              << " downweighted=" << result.consistency_downweighted_observation_count
              << " mean_w=" << result.consistency_mean_weight
              << " min_w=" << result.consistency_min_applied_weight
              << std::endl;
  }
  for (const ati::AslamBackendOptimizationStageSummary& stage : result.stages) {
    std::cout << "[stage5_backend] stage " << stage.stage_label
              << " cost=" << stage.objective_start << " -> "
              << stage.objective_final
              << " iterations=" << stage.iterations
              << " failed=" << stage.failed_iterations
              << " lambda=" << stage.lm_lambda_final << std::endl;
  }
}

void WritePhase4BenchmarkSummary(
    const fs::path& path,
    const ati::AslamBackendCalibrationResult& backend_result,
    const ati::CameraModelRefitEvaluationResult& training_evaluation,
    const ati::CameraModelRefitEvaluationResult& holdout_evaluation) {
  std::ofstream output(path.string().c_str());
  output << "backend_multi_board_consistency_weighting: "
         << (backend_result.options.multi_board_consistency_weighting ? 1 : 0) << "\n";
  output << "backend_consistency_pose_source: "
         << backend_result.options.consistency_pose_source << "\n";
  output << "backend_consistency_weight_mode: "
         << ati::ToString(backend_result.options.consistency_weight_mode) << "\n";
  output << "training_overall_rmse: " << training_evaluation.overall_rmse << "\n";
  output << "training_outer_only_rmse: " << training_evaluation.outer_only_rmse << "\n";
  output << "training_internal_only_rmse: " << training_evaluation.internal_only_rmse << "\n";
  output << "holdout_overall_rmse: " << holdout_evaluation.overall_rmse << "\n";
  output << "holdout_outer_only_rmse: " << holdout_evaluation.outer_only_rmse << "\n";
  output << "holdout_internal_only_rmse: " << holdout_evaluation.internal_only_rmse << "\n";
  output << "holdout_pose_only_refit_rmse: "
         << holdout_evaluation.pose_only_refit_rmse << "\n";
  output << "holdout_pose_only_refit_success_rate: "
         << holdout_evaluation.pose_only_refit_success_rate << "\n";
  output << "holdout_std_residual_x: " << holdout_evaluation.std_residual_x << "\n";
  output << "holdout_std_residual_y: " << holdout_evaluation.std_residual_y << "\n";
  output << "consistency_observation_count: "
         << backend_result.consistency_observation_count << "\n";
  output << "consistency_successful_observation_count: "
         << backend_result.consistency_successful_observation_count << "\n";
  output << "consistency_downweighted_observation_count: "
         << backend_result.consistency_downweighted_observation_count << "\n";
  output << "consistency_hard_rejected_observation_count: "
         << backend_result.consistency_hard_rejected_observation_count << "\n";
  output << "consistency_mean_weight: "
         << backend_result.consistency_mean_weight << "\n";
  output << "consistency_min_applied_weight: "
         << backend_result.consistency_min_applied_weight << "\n";
  output << "consistency_max_translation_error_mm: "
         << backend_result.consistency_max_translation_error_mm << "\n";
  output << "consistency_max_rotation_error_deg: "
         << backend_result.consistency_max_rotation_error_deg << "\n";
  output << "note: Phase4 weighting affects backend BA residuals; holdout pose-only refit remains unweighted and only sees optimized intrinsics.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto total_start = std::chrono::steady_clock::now();
    const CmdArgs args = ParseArgs(argc, argv);
    const bool use_precomputed = !args.precomputed_observations_dir.empty();
    const RequestedExperimentConfig requested_config =
        BuildRequestedExperimentConfig(args);
    ValidateBackendResidualConfiguration(requested_config);
    ati::Stage5RuntimeSummary runtime_summary;
    runtime_summary.runtime_mode = args.runtime_mode;
    runtime_summary.cache_dir =
        args.cache_dir.empty() ? "result/.stage5_backend_cache" : args.cache_dir;
    runtime_summary.cache_layout_version = ati::Stage5CacheLayoutVersion();
    if (!use_precomputed) {
      const ati::Stage5DatasetCacheIdentity cache_dataset_identity =
          ati::MakeStage5DatasetCacheIdentity(args.image_path);
      runtime_summary.cache_dataset_label =
          cache_dataset_identity.dataset_label;
      runtime_summary.cache_dataset_image_root =
          cache_dataset_identity.absolute_image_root;
    }
    runtime_summary.cache_enabled = !use_precomputed;
    const std::string dataset_label = InferDatasetLabel(args);
    std::vector<std::string> image_paths;
    std::vector<std::string> test_image_paths;
    ati::FrozenPrecomputedMeasurementInput precomputed_training;
    ati::FrozenPrecomputedMeasurementInput precomputed_holdout;
    std::vector<ati::FrozenPrecomputedMeasurementInput>
        precomputed_initialization_auxiliary_sessions;
    {
      const auto stage_start = std::chrono::steady_clock::now();
      if (use_precomputed) {
        const ati::PrecomputedObservationImporter importer;
        precomputed_training =
            importer.Load(args.precomputed_observations_dir, 0,
                          args.precomputed_target_mode);
        if (!precomputed_training.success) {
          throw std::runtime_error(
              "Failed to import precomputed training observations: " +
              precomputed_training.failure_reason);
        }
        if (!args.precomputed_holdout_observations_dir.empty()) {
          precomputed_holdout = importer.Load(
              args.precomputed_holdout_observations_dir, 1000000,
              args.precomputed_target_mode);
          if (!precomputed_holdout.success) {
            throw std::runtime_error(
                "Failed to import precomputed holdout observations: " +
                precomputed_holdout.failure_reason);
          }
          if (precomputed_holdout.target_mode_resolved !=
                  precomputed_training.target_mode_resolved ||
              precomputed_holdout.board_count !=
                  precomputed_training.board_count) {
            throw std::runtime_error(
                "Precomputed training and holdout target topology do not match.");
          }
        }
        precomputed_initialization_auxiliary_sessions.reserve(
            args.precomputed_initialization_auxiliary_observation_dirs.size());
        for (std::size_t auxiliary_index = 0;
             auxiliary_index <
             args.precomputed_initialization_auxiliary_observation_dirs.size();
             ++auxiliary_index) {
          const std::string& auxiliary_dir =
              args.precomputed_initialization_auxiliary_observation_dirs[
                  auxiliary_index];
          if (fs::equivalent(fs::path(auxiliary_dir),
                             fs::path(args.precomputed_observations_dir)) ||
              (!args.precomputed_holdout_observations_dir.empty() &&
               fs::equivalent(
                   fs::path(auxiliary_dir),
                   fs::path(args.precomputed_holdout_observations_dir)))) {
            throw std::runtime_error(
                "Auxiliary initialization observations must be disjoint from "
                "primary training and frozen holdout inputs: " +
                auxiliary_dir);
          }
          for (std::size_t previous_index = 0;
               previous_index < auxiliary_index; ++previous_index) {
            if (fs::equivalent(
                    fs::path(auxiliary_dir),
                    fs::path(args
                                 .precomputed_initialization_auxiliary_observation_dirs[
                                     previous_index]))) {
              throw std::runtime_error(
                  "Duplicate auxiliary initialization observation session: " +
                  auxiliary_dir);
            }
          }
          ati::FrozenPrecomputedMeasurementInput auxiliary = importer.Load(
              auxiliary_dir,
              2000000 + static_cast<int>(auxiliary_index) * 1000000,
              args.precomputed_target_mode);
          if (!auxiliary.success) {
            throw std::runtime_error(
                "Failed to import auxiliary initialization observations: " +
                auxiliary.failure_reason);
          }
          if (auxiliary.image_size != precomputed_training.image_size ||
              auxiliary.reference_board_id !=
                  precomputed_training.reference_board_id ||
              auxiliary.target_mode_resolved !=
                  precomputed_training.target_mode_resolved ||
              auxiliary.board_count != precomputed_training.board_count) {
            throw std::runtime_error(
                "Auxiliary initialization session topology/resolution does not "
                "match the primary training observations: " +
                auxiliary_dir);
          }
          precomputed_initialization_auxiliary_sessions.push_back(
              std::move(auxiliary));
        }
      } else {
        image_paths = CollectImagePaths(args.image_path, args.all);
        if (!args.test_image_path.empty()) {
          test_image_paths = CollectImagePaths(args.test_image_path, args.all);
        }
        const ati::PrecomputedObservationImporter importer;
        precomputed_initialization_auxiliary_sessions.reserve(
            args.precomputed_initialization_auxiliary_observation_dirs.size());
        for (std::size_t auxiliary_index = 0;
             auxiliary_index <
             args.precomputed_initialization_auxiliary_observation_dirs.size();
             ++auxiliary_index) {
          const std::string& auxiliary_dir =
              args.precomputed_initialization_auxiliary_observation_dirs[
                  auxiliary_index];
          for (std::size_t previous_index = 0;
               previous_index < auxiliary_index; ++previous_index) {
            if (fs::equivalent(
                    fs::path(auxiliary_dir),
                    fs::path(args
                                 .precomputed_initialization_auxiliary_observation_dirs[
                                     previous_index]))) {
              throw std::runtime_error(
                  "Duplicate auxiliary initialization observation session: " +
                  auxiliary_dir);
            }
          }
          ati::FrozenPrecomputedMeasurementInput auxiliary = importer.Load(
              auxiliary_dir,
              2000000 + static_cast<int>(auxiliary_index) * 1000000,
              args.precomputed_target_mode);
          if (!auxiliary.success) {
            throw std::runtime_error(
                "Failed to import auxiliary initialization observations: " +
                auxiliary.failure_reason);
          }
          precomputed_initialization_auxiliary_sessions.push_back(
              std::move(auxiliary));
        }
      }
      if (!use_precomputed &&
          args.allow_image_training_with_precomputed_holdout &&
          !args.precomputed_holdout_observations_dir.empty()) {
        const ati::PrecomputedObservationImporter importer;
        precomputed_holdout = importer.Load(
            args.precomputed_holdout_observations_dir, 1000000,
            args.precomputed_target_mode);
        if (!precomputed_holdout.success) {
          throw std::runtime_error(
              "Failed to import precomputed holdout observations: " +
              precomputed_holdout.failure_reason);
        }
      }
      std::set<std::string> occupied_frame_labels;
      if (use_precomputed) {
        for (const ati::FrozenRound2BaselineFrameSource& frame :
             precomputed_training.frame_sources) {
          occupied_frame_labels.insert(frame.frame_label);
        }
        for (const ati::FrozenRound2BaselineFrameSource& frame :
             precomputed_holdout.frame_sources) {
          occupied_frame_labels.insert(frame.frame_label);
        }
      } else {
        for (const std::string& image_path : image_paths) {
          occupied_frame_labels.insert(
              fs::path(image_path).stem().string());
        }
        // An explicit raw holdout is allowed to point at the training set for
        // diagnostics, so only auxiliary overlap is prohibited here.
      }
      for (std::size_t auxiliary_index = 0;
           auxiliary_index <
           precomputed_initialization_auxiliary_sessions.size();
           ++auxiliary_index) {
        for (const ati::FrozenRound2BaselineFrameSource& frame :
             precomputed_initialization_auxiliary_sessions[auxiliary_index]
                 .frame_sources) {
          if (!frame.frame_label.empty() &&
              !occupied_frame_labels.insert(frame.frame_label).second) {
            throw std::runtime_error(
                "Auxiliary initialization frame overlaps a primary, holdout, "
                "or earlier auxiliary frame: " +
                frame.frame_label + " auxiliary_index=" +
                std::to_string(auxiliary_index));
          }
        }
      }
      AddRuntimeStage(&runtime_summary, "image_collection", ElapsedSeconds(stage_start),
                      false);
    }
    PrintProgress("dataset=" + dataset_label);
    PrintProgress("training_input=" +
                  (use_precomputed ? args.precomputed_observations_dir
                                   : args.image_path));
    if (!args.precomputed_holdout_observations_dir.empty()) {
      PrintProgress("holdout_input=" +
                    args.precomputed_holdout_observations_dir);
    } else if (!args.test_image_path.empty()) {
      PrintProgress("holdout_input=" + args.test_image_path);
    }
    for (const std::string& auxiliary_dir :
         args.precomputed_initialization_auxiliary_observation_dirs) {
      PrintProgress("initialization_auxiliary_input=" + auxiliary_dir);
    }
    PrintProgress("output=" + args.output_path);
    PrintProgress("collected_training_images=" + std::to_string(image_paths.size()));
    if (!args.test_image_path.empty()) {
      PrintProgress("collected_holdout_images=" +
                    std::to_string(test_image_paths.size()));
    }
    PrintProgress("runtime_mode=" +
                  std::string(ati::ToString(args.runtime_mode)));
    PrintProgress("cache_dir=" + runtime_summary.cache_dir);
    const std::string kalibr_source_label =
        BuildKalibrSourceLabel(args.kalibr_camchain_yaml);

    std::vector<ati::FrozenRound2BaselineFrameSource> all_frames =
        use_precomputed ? precomputed_training.frame_sources
                        : BuildFrameSources(image_paths, 0);
    std::vector<ati::FrozenRound2BaselineFrameSource> external_holdout_frames;
    if (precomputed_holdout.success) {
      external_holdout_frames = precomputed_holdout.frame_sources;
    } else if (!test_image_paths.empty()) {
      external_holdout_frames =
          BuildFrameSources(test_image_paths, static_cast<int>(all_frames.size()));
    }
    std::vector<ati::FrozenRound2BaselineFrameSource> all_frames_for_lookup =
        all_frames;
    all_frames_for_lookup.insert(all_frames_for_lookup.end(),
                                 external_holdout_frames.begin(),
                                 external_holdout_frames.end());

    ati::FrozenRound2BaselineOptions baseline_options;
    baseline_options.config = ati::ApriltagInternalDetector::LoadConfig(args.config_path);
    ApplyStage5ModelFamily(requested_config.models,
                           &baseline_options.config);
    if (requested_config.camera_aware_outer_rescue_zero_detection_frames_set) {
      baseline_options.config.outer_detector_config
          .camera_aware_sphere_patch_rescue_zero_detection_frames =
          requested_config.camera_aware_outer_rescue_zero_detection_frames;
    }
    if (!use_precomputed &&
        !precomputed_initialization_auxiliary_sessions.empty()) {
      const std::vector<int>& resolution =
          baseline_options.config.intermediate_camera.resolution;
      const cv::Size configured_size =
          resolution.size() == 2
              ? cv::Size(resolution[0], resolution[1])
              : cv::Size();
      const int configured_board_count =
          !baseline_options.config.tag_ids.empty()
              ? static_cast<int>(baseline_options.config.tag_ids.size())
              : 1;
      for (std::size_t auxiliary_index = 0;
           auxiliary_index <
           precomputed_initialization_auxiliary_sessions.size();
           ++auxiliary_index) {
        const ati::FrozenPrecomputedMeasurementInput& auxiliary =
            precomputed_initialization_auxiliary_sessions[auxiliary_index];
        if (configured_size.width <= 0 || configured_size.height <= 0 ||
            auxiliary.image_size != configured_size ||
            auxiliary.reference_board_id != args.reference_board_id ||
            auxiliary.board_count != configured_board_count) {
          throw std::runtime_error(
              "Auxiliary initialization session topology/resolution does not "
              "match the raw-image primary configuration: " +
              args.precomputed_initialization_auxiliary_observation_dirs[
                  auxiliary_index]);
        }
      }
    }
    baseline_options.config.camera_initialization_mode =
        requested_config.camera_init_mode;
    baseline_options.camera_initialization_refine_mode =
        requested_config.init_refine_mode;
    baseline_options.camera_initialization_selection_scorer =
        requested_config.init_selection_scorer;
    baseline_options.enable_camera_initialization_principal_profile =
        args.stage5_enable_init_principal_profile;
    baseline_options.camera_initialization_principal_profile_radius_px =
        std::max(0.0, args.stage5_init_principal_profile_radius_px);
    baseline_options.enable_camera_initialization_fixed_layout_diagnostic =
        args.stage5_enable_init_fixed_layout_diagnostic;
    baseline_options.enable_camera_initialization_board_jackknife_diagnostic =
        args.stage5_enable_init_board_jackknife_diagnostic;
    baseline_options.enable_camera_initialization_coverage_weighted_diagnostic =
        args.stage5_enable_init_coverage_weighted_diagnostic;
    baseline_options.camera_initialization_prefer_lower_focal_in_near_tie =
        args.stage5_init_near_tie_prefer_lower_focal;
    baseline_options.camera_initialization_near_tie_relative_objective_tolerance =
        std::max(0.0,
                 args.stage5_init_near_tie_relative_objective_tolerance);
    baseline_options.checkerboard_initialization_huber_delta_pixels =
        requested_config.checkerboard_huber_delta_pixels;
    baseline_options.precomputed_initialization_use_all_points =
        use_precomputed && args.precomputed_init_use_all_points;
    baseline_options.precomputed_initialization_point_scope =
        args.precomputed_initialization_point_scope;
    baseline_options.enable_camera_aware_outer_rescue =
        requested_config.camera_aware_outer_rescue && !use_precomputed;
    baseline_options.rerun_camera_initialization_after_outer_rescue = true;
    baseline_options.camera_aware_outer_rescue_max_hamming = 0;
    baseline_options.camera_initialization_auxiliary_session_count =
        static_cast<int>(
            precomputed_initialization_auxiliary_sessions.size());
    for (const ati::FrozenPrecomputedMeasurementInput& auxiliary :
         precomputed_initialization_auxiliary_sessions) {
      baseline_options.camera_initialization_auxiliary_bootstrap_frames.insert(
          baseline_options.camera_initialization_auxiliary_bootstrap_frames.end(),
          auxiliary.bootstrap_frames.begin(), auxiliary.bootstrap_frames.end());
    }
    baseline_options.reference_board_id = args.reference_board_id;
    baseline_options.frontend_only = args.stage5_frontend_only;
    baseline_options.outer_only_ablation_mode =
        requested_config.outer_only_ablation_mode;
    baseline_options.include_internal_points =
        requested_config.include_internal_points;
    baseline_options.optimize_intrinsics =
        requested_config.frontend_optimize_intrinsics;
    baseline_options.intrinsics_release_iteration =
        requested_config.frontend_intrinsics_release_iteration;
    baseline_options.run_second_pass = requested_config.run_second_pass;
    baseline_options.second_pass_intrinsics_release_iteration =
        requested_config.frontend_second_pass_intrinsics_release_iteration;
  baseline_options.enable_residual_sanity_gate =
      requested_config.enable_residual_sanity_gate;
  baseline_options.enable_board_pose_fit_gate =
      requested_config.enable_board_pose_fit_gate;
  baseline_options.selection_mode =
      ati::ParseJointMeasurementSelectionMode(
          requested_config.selection_mode);
  baseline_options.selection_residual_sanity_factor =
      requested_config.selection_residual_sanity_factor;
  baseline_options.selection_max_board_observation_rmse =
      requested_config.selection_max_board_observation_rmse;
  baseline_options.selection_kalibr_style_outlier_sigma =
      requested_config.selection_kalibr_style_outlier_sigma;
  baseline_options.selection_kalibr_style_min_abs_threshold_px =
      requested_config.selection_kalibr_style_min_abs_threshold_px;
  baseline_options.selection_kalibr_style_min_views_before_filter =
      requested_config.selection_kalibr_style_min_views_before_filter;
  baseline_options.strict_board_observation_acceptance =
      requested_config.strict_board_observation_acceptance;
    baseline_options.preserve_frame_board_cohesion =
      requested_config.preserve_frame_board_cohesion;
    if (use_precomputed && precomputed_training.single_board_mode) {
      baseline_options.preserve_frame_board_cohesion = false;
    }
    baseline_options.ignore_image_evidence_min_quality =
        requested_config.ignore_image_evidence_min_quality;
    baseline_options.force_internal_seed_from_prediction =
        requested_config.force_internal_seed_from_prediction;
    baseline_options.bypass_internal_seed_filters =
        requested_config.bypass_internal_seed_filters;
    baseline_options.internal_corner_filter_mode =
        requested_config.internal_corner_filter_mode;
    baseline_options.internal_corner_filter_max_reproj_error =
        requested_config.internal_corner_filter_max_reproj_error;
    baseline_options.internal_corner_filter_quality_min =
        requested_config.internal_corner_filter_quality_min;
    baseline_options.internal_corner_filter_quality_relaxation_px =
        requested_config.internal_corner_filter_quality_relaxation_px;
    baseline_options.internal_corner_filter_adaptive_min_threshold_px =
        requested_config.internal_corner_filter_adaptive_min_threshold_px;
    baseline_options.enable_internal_observation_quality_weighting =
        requested_config.internal_observation_weight_mode ==
            ati::InternalObservationWeightMode::Enabled &&
        requested_config.internal_observation_weight_policy == "quality";
    baseline_options.internal_observation_low_quality_quantile =
        requested_config.internal_observation_weight_low_quality_quantile;
    baseline_options.internal_observation_min_weight =
        requested_config.internal_observation_weight_min;
    baseline_options.internal_observation_quality_exponent =
        requested_config.internal_observation_weight_quality_exponent;
    baseline_options.internal_pose_rescue_mode =
        requested_config.internal_pose_rescue_mode;
    baseline_options.internal_pose_rescue_max_ray_angle_deg =
        requested_config.internal_pose_rescue_max_ray_angle_deg;
    baseline_options.internal_pose_rescue_accept_max_outer_rmse =
        requested_config.internal_pose_rescue_accept_max_outer_rmse;
    baseline_options.enable_geometry_prior_outer_seed =
        requested_config.enable_geometry_prior_outer_seed;
    baseline_options.geometry_prior_rescue_diagnostic_only =
        requested_config.geometry_prior_rescue_diagnostic_only;
    baseline_options.geometry_prior_rescue_use_as_observation =
        requested_config.geometry_prior_rescue_use_as_observation;
    baseline_options.geometry_prior_rescue_keep_outer_on_internal_failure =
        requested_config.geometry_prior_rescue_keep_outer_on_internal_failure;
    baseline_options.geometry_prior_rescue_allow_geometry_only_pose_refit =
        requested_config.geometry_prior_rescue_allow_geometry_only_pose_refit;
    baseline_options.geometry_prior_rescue_subpix_window_radius =
        requested_config.geometry_prior_rescue_subpix_window_radius;
    baseline_options.geometry_prior_rescue_max_corner_displacement_px =
        requested_config.geometry_prior_rescue_max_corner_displacement_px;
    baseline_options.geometry_prior_rescue_min_corner_response_ratio =
        requested_config.geometry_prior_rescue_min_corner_response_ratio;
    baseline_options.geometry_prior_rescue_enable_spherical_refine =
        requested_config.geometry_prior_rescue_enable_spherical_refine;
    baseline_options.geometry_prior_rescue_edge_sample_count =
        requested_config.geometry_prior_rescue_edge_sample_count;
    baseline_options.geometry_prior_rescue_edge_search_half_width_px =
        requested_config.geometry_prior_rescue_edge_search_half_width_px;
    baseline_options.geometry_prior_rescue_min_edge_support_ratio =
        requested_config.geometry_prior_rescue_min_edge_support_ratio;
    baseline_options.geometry_prior_rescue_min_edge_gradient_ratio =
        requested_config.geometry_prior_rescue_min_edge_gradient_ratio;
    baseline_options.geometry_prior_rescue_accept_max_outer_rmse =
        requested_config.geometry_prior_rescue_accept_max_outer_rmse;
    baseline_options.geometry_prior_rescue_accept_max_rotation_error_deg =
        requested_config.geometry_prior_rescue_accept_max_rotation_error_deg;
    baseline_options.geometry_prior_rescue_accept_max_translation_error =
        requested_config.geometry_prior_rescue_accept_max_translation_error;
    baseline_options.geometry_guided_tag_likelihood_enabled =
        requested_config.geometry_guided_tag_likelihood_enabled;
    baseline_options.geometry_guided_tag_likelihood_min_visible_boards =
        requested_config.geometry_guided_tag_likelihood_min_visible_boards;
    baseline_options.geometry_guided_tag_likelihood_max_expected_hamming =
        requested_config.geometry_guided_tag_likelihood_max_expected_hamming;
    baseline_options.geometry_guided_tag_likelihood_min_hamming_margin =
        requested_config.geometry_guided_tag_likelihood_min_hamming_margin;
    baseline_options.geometry_guided_tag_likelihood_min_contrast =
        requested_config.geometry_guided_tag_likelihood_min_contrast;
    baseline_options.geometry_guided_tag_likelihood_allow_single_anchor =
        requested_config.geometry_guided_tag_likelihood_allow_single_anchor;
    baseline_options.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse =
        requested_config.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse;
    baseline_options.geometry_guided_tag_likelihood_single_anchor_max_expected_hamming =
        requested_config.geometry_guided_tag_likelihood_single_anchor_max_expected_hamming;
    baseline_options.geometry_guided_tag_likelihood_single_anchor_min_hamming_margin =
        requested_config.geometry_guided_tag_likelihood_single_anchor_min_hamming_margin;
    baseline_options.geometry_guided_tag_likelihood_single_anchor_min_contrast =
        requested_config.geometry_guided_tag_likelihood_single_anchor_min_contrast;
    baseline_options.enable_outer_only_intermediate_calibration =
        requested_config.enable_outer_only_intermediate_calibration;
    baseline_options.intermediate_diagnostic_only =
        requested_config.intermediate_diagnostic_only;
    baseline_options.use_intermediate_for_round1_internal_regeneration =
        requested_config.use_intermediate_for_round1_internal_regeneration;
    baseline_options.use_intermediate_for_full_frontend_regeneration =
        requested_config.use_intermediate_for_full_frontend_regeneration;
    baseline_options.intermediate_optimize_intrinsics =
        requested_config.intermediate_optimize_intrinsics;
    baseline_options.intermediate_optimize_board_poses =
        requested_config.intermediate_optimize_board_poses;
    baseline_options.intermediate_optimize_frame_poses =
        requested_config.intermediate_optimize_frame_poses;
    baseline_options.intermediate_intrinsics_release_iteration =
        requested_config.intermediate_intrinsics_release_iteration;
    baseline_options.intermediate_max_outer_rmse_px =
        requested_config.intermediate_max_outer_rmse_px;
    baseline_options.intermediate_min_visible_boards =
        requested_config.intermediate_min_visible_boards;
    baseline_options.dataset_label = dataset_label;
    baseline_options.baseline_protocol_label =
        requested_config.effective_protocol_label;
    baseline_options.source_pipeline_label = "run_stage5_backend";
    baseline_options.enable_outer_detection_cache = true;
    baseline_options.outer_detection_cache_dir = runtime_summary.cache_dir;

    ati::BackendProblemOptions backend_options;
    backend_options.reference_board_id = args.reference_board_id;
    backend_options.optimize_frame_poses = true;
    backend_options.optimize_board_poses = true;
    backend_options.optimize_intrinsics =
        requested_config.backend_optimize_intrinsics;
    backend_options.delayed_intrinsics_release =
        requested_config.backend_delayed_intrinsics_release;
    backend_options.intrinsics_release_iteration =
        requested_config.backend_intrinsics_release_iteration;
    ati::BackendProblemOptions committed_backend_evaluation_options =
        backend_options;
    committed_backend_evaluation_options.optimize_board_poses =
        requested_config.backend_optimize_board_poses;

    ati::CalibrationBenchmarkSplitOptions split_options;
    split_options.mode = args.stage5_no_holdout
                             ? "all_frames_training_no_holdout"
                             : args.split_mode;
    split_options.all_frames_training = args.stage5_no_holdout;
    split_options.holdout_ratio = args.holdout_ratio;
    split_options.random_seed = args.split_seed;
    split_options.holdout_stride = args.holdout_stride;
    split_options.holdout_offset = args.holdout_offset;
    const ati::Stage5Benchmark benchmark(split_options);
    ati::CalibrationBenchmarkSplit preview_split;
    {
      const auto stage_start = std::chrono::steady_clock::now();
      if (args.stage5_frontend_only) {
        preview_split.success = true;
        preview_split.mode = "frontend_only_all_frames";
        preview_split.split_signature = "frontend_only_all_frames";
        preview_split.training_frames = all_frames;
      } else {
        preview_split = external_holdout_frames.empty()
                            ? benchmark.BuildDeterministicSplit(all_frames)
                            : benchmark.BuildExternalHoldoutSplit(
                                  all_frames,
                                  external_holdout_frames,
                                  use_precomputed
                                      ? fs::path(args.precomputed_holdout_observations_dir)
                                            .filename()
                                            .string()
                                      : fs::path(args.test_image_path).stem().string());
      }
      AddRuntimeStage(&runtime_summary, "split_preview", ElapsedSeconds(stage_start),
                      false);
    }
    if (preview_split.success) {
      PrintProgress("split=" + preview_split.split_signature +
                    " training=" +
                    std::to_string(preview_split.training_frames.size()) +
                    " holdout=" +
                    std::to_string(preview_split.holdout_frames.size()));
    } else {
      PrintProgress("split preview failed: " + preview_split.failure_reason);
    }
    const std::string kalibr_training_split_signature =
        !args.kalibr_training_split_signature.empty()
            ? args.kalibr_training_split_signature
            : (preview_split.success ? preview_split.split_signature : std::string());

    ati::KalibrBenchmarkReference kalibr_reference;
    kalibr_reference.camchain_yaml = args.kalibr_camchain_yaml;
    kalibr_reference.camera_model_family =
        BuildConfiguredCameraFamily(baseline_options.config);
    kalibr_reference.training_split_signature = kalibr_training_split_signature;
    kalibr_reference.runtime_seconds = args.kalibr_runtime_seconds;
    kalibr_reference.source_label = kalibr_source_label;

    std::vector<ati::KalibrBenchmarkReference> additional_camera_references;
    additional_camera_references.reserve(args.reference_intrinsics_specs.size());
    for (const std::string& reference_spec : args.reference_intrinsics_specs) {
      additional_camera_references.push_back(
          BuildAdditionalReferenceCamera(reference_spec,
                                         kalibr_training_split_signature));
    }

    ati::Stage5BenchmarkInput benchmark_input;
    benchmark_input.all_frames = all_frames;
    benchmark_input.external_holdout_frames = external_holdout_frames;
    benchmark_input.frontend_only = args.stage5_frontend_only;
    benchmark_input.use_precomputed_training_measurements = use_precomputed;
    benchmark_input.precomputed_training_measurements = precomputed_training;
    benchmark_input.use_precomputed_holdout_measurements =
        precomputed_holdout.success;
    benchmark_input.precomputed_holdout_measurements = precomputed_holdout;
    benchmark_input.external_holdout_label =
        use_precomputed
            ? fs::path(args.precomputed_holdout_observations_dir)
                  .filename()
                  .string()
            : (args.test_image_path.empty()
                   ? std::string()
                   : fs::path(args.test_image_path).stem().string());
    benchmark_input.use_external_holdout_self_frontend_prepass =
        args.external_holdout_self_frontend_prepass;
    benchmark_input.holdout_evaluate_full_training_observations =
        args.stage5_holdout_evaluate_full_training_observations;
    benchmark_input.baseline_options = baseline_options;
    benchmark_input.backend_options = backend_options;
    benchmark_input.committed_backend_evaluation_options =
        committed_backend_evaluation_options;
    benchmark_input.selection_backend_runner_options.residual_model =
        ati::ParseResidualModel(requested_config.backend_residual_model);
    benchmark_input.selection_backend_runner_options.hybrid_angular_threshold_deg =
        requested_config.backend_hybrid_angular_threshold_deg;
    benchmark_input.selection_backend_runner_options.outer_residual_model =
        ati::ParseResidualModel(requested_config.backend_outer_residual_model);
    benchmark_input.selection_backend_runner_options.internal_residual_model =
        ati::ParseResidualModel(requested_config.backend_internal_residual_model);
    benchmark_input.selection_backend_runner_options
        .use_point_type_residual_split =
        requested_config.backend_use_point_type_residual_split;
    benchmark_input.selection_backend_runner_options.angular_auxiliary_enabled =
        requested_config.backend_enable_angular_auxiliary_residual;
    benchmark_input.selection_backend_runner_options.angular_auxiliary_weight =
        requested_config.backend_angular_auxiliary_weight;
    benchmark_input.selection_backend_runner_options.angular_auxiliary_normalized =
        requested_config.backend_angular_auxiliary_normalized;
    benchmark_input.selection_backend_runner_options
        .angular_auxiliary_apply_to_outer =
        requested_config.backend_angular_auxiliary_apply_to_outer;
    benchmark_input.selection_backend_runner_options
        .angular_auxiliary_apply_to_internal =
        requested_config.backend_angular_auxiliary_apply_to_internal;
    benchmark_input.selection_backend_runner_options
        .polar_continuous_hybrid_threshold_deg =
        requested_config.backend_polar_continuous_hybrid_threshold_deg;
    benchmark_input.selection_backend_runner_options
        .polar_continuous_hybrid_temperature_deg =
        requested_config.backend_polar_continuous_hybrid_temperature_deg;
    benchmark_input.selection_backend_runner_options
        .normalized_angular_reference_sigma_px =
        requested_config.backend_normalized_angular_reference_sigma_px;
    benchmark_input.selection_backend_runner_options
        .normalized_angular_min_sigma_rad =
        requested_config.backend_normalized_angular_min_sigma_rad;
    benchmark_input.selection_backend_runner_options
        .normalized_angular_max_weight_scale =
        requested_config.backend_normalized_angular_max_weight_scale;
    benchmark_input.selection_backend_runner_options.pixel_residual_weight =
        requested_config.backend_pixel_residual_weight;
    benchmark_input.selection_backend_runner_options.chordal_residual_weight =
        requested_config.backend_chordal_residual_weight;
    benchmark_input.selection_backend_runner_options
        .angular_use_normalize_jacobian =
        requested_config.backend_angular_use_normalize_jacobian;
    benchmark_input.selection_backend_runner_options
        .angular_local_whitening_enabled =
        requested_config.backend_angular_local_whitening;
    benchmark_input.selection_backend_runner_options
        .angular_local_whitening_pixel_sigma_px =
        requested_config.backend_angular_local_whitening_pixel_sigma_px;
    benchmark_input.selection_backend_runner_options
        .angular_local_whitening_covariance_damping =
        requested_config.backend_angular_local_whitening_covariance_damping;
    benchmark_input.selection_backend_runner_options
        .angular_local_whitening_min_sigma_rad =
        requested_config.backend_angular_local_whitening_min_sigma_rad;
    benchmark_input.selection_backend_runner_options
        .angular_local_whitening_max_weight =
        requested_config.backend_angular_local_whitening_max_weight;
    benchmark_input.selection_backend_runner_options.angular_observed_ray_mode =
        ati::ParseAngularObservedRayMode(
            requested_config.backend_angular_observed_ray_mode);
    benchmark_input.pre_backend_filter_options.mode =
        requested_config.pre_backend_filter_mode;
    benchmark_input.pre_backend_filter_options.threshold_mode =
        requested_config.pre_backend_filter_threshold_mode;
    benchmark_input.pre_backend_filter_options.sigma_threshold =
        requested_config.pre_backend_filter_sigma;
    benchmark_input.pre_backend_filter_options.min_abs_threshold_px =
        requested_config.pre_backend_filter_min_abs_threshold_px;
    benchmark_input.internal_blur_filter_options.mode =
        requested_config.internal_blur_filter_mode;
    benchmark_input.internal_blur_filter_options.low_patch_gradient_quantile =
        requested_config.internal_blur_filter_low_patch_gradient_quantile;
    benchmark_input.internal_blur_filter_options.min_board_internal_rmse_px =
        requested_config.internal_blur_filter_min_board_rmse_px;
    benchmark_input.internal_blur_filter_options.min_board_p95_residual_px =
        requested_config.internal_blur_filter_min_board_p95_px;
    benchmark_input.internal_joint_refine_options.mode =
        requested_config.internal_joint_refine_mode;
    benchmark_input.internal_joint_refine_options.target_mode =
        requested_config.internal_joint_refine_target_mode;
    benchmark_input.internal_joint_refine_options.search_radius_px =
        requested_config.internal_joint_refine_search_radius_px;
    benchmark_input.internal_joint_refine_options.max_displacement_px =
        requested_config.internal_joint_refine_max_displacement_px;
    benchmark_input.internal_joint_refine_options.geometry_sigma_px =
        requested_config.internal_joint_refine_geometry_sigma_px;
    benchmark_input.internal_joint_refine_options.observation_sigma_px =
        requested_config.internal_joint_refine_observation_sigma_px;
    benchmark_input.internal_joint_refine_options.subpix_window_radius =
        requested_config.internal_joint_refine_subpix_window_radius;
    benchmark_input.internal_joint_refine_options.min_objective_improvement =
        requested_config.internal_joint_refine_min_objective_improvement;
    benchmark_input.internal_joint_refine_options.min_old_residual_px =
        requested_config.internal_joint_refine_min_old_residual_px;
    benchmark_input.internal_joint_refine_options.low_patch_gradient_quantile =
        requested_config.internal_joint_refine_low_patch_gradient_quantile;
    benchmark_input.internal_joint_refine_options.min_board_internal_rmse_px =
        requested_config.internal_joint_refine_min_board_rmse_px;
    benchmark_input.internal_joint_refine_options.min_board_p95_residual_px =
        requested_config.internal_joint_refine_min_board_p95_px;
    benchmark_input.internal_joint_refine_options.min_corner_response_gain =
        requested_config.internal_joint_refine_min_corner_response_gain;
    benchmark_input.internal_joint_refine_options
        .min_board_internal_rmse_improvement_px =
        requested_config.internal_joint_refine_min_board_internal_improvement_px;
    benchmark_input.internal_joint_refine_options
        .min_refined_point_count_per_board =
        requested_config.internal_joint_refine_min_refined_point_count_per_board;
    benchmark_input.internal_joint_refine_options
        .accept_max_global_outer_delta_px =
        requested_config.internal_joint_refine_accept_max_global_outer_delta_px;
    benchmark_input.internal_joint_refine_options
        .accept_max_frame_outer_delta_px =
        requested_config.internal_joint_refine_accept_max_frame_outer_delta_px;
    benchmark_input.internal_joint_refine_options
        .acceptance_backend_max_iterations =
        requested_config.internal_joint_refine_acceptance_backend_max_iterations;
    benchmark_input.internal_blur_board_weight_options.mode =
        requested_config.internal_blur_board_weight_mode;
    benchmark_input.internal_blur_board_weight_options
        .low_patch_gradient_quantile =
        requested_config.internal_blur_board_weight_low_patch_gradient_quantile;
    benchmark_input.internal_blur_board_weight_options
        .min_board_internal_rmse_px =
        requested_config.internal_blur_board_weight_min_board_rmse_px;
    benchmark_input.internal_blur_board_weight_options
        .min_board_p95_residual_px =
        requested_config.internal_blur_board_weight_min_board_p95_px;
    benchmark_input.internal_blur_board_weight_options.min_weight =
        requested_config.internal_blur_board_weight_min;
    benchmark_input.internal_blur_board_weight_options.gradient_exponent =
        requested_config.internal_blur_board_weight_gradient_exponent;
    benchmark_input.internal_observation_weight_options.mode =
        requested_config.internal_observation_weight_mode;
    benchmark_input.internal_observation_weight_options.policy =
        requested_config.internal_observation_weight_policy;
    benchmark_input.internal_observation_weight_options.low_quality_quantile =
        requested_config.internal_observation_weight_low_quality_quantile;
    benchmark_input.internal_observation_weight_options.min_weight =
        requested_config.internal_observation_weight_min;
    benchmark_input.internal_observation_weight_options.quality_exponent =
        requested_config.internal_observation_weight_quality_exponent;
    benchmark_input.internal_observation_weight_options
        .residual_consistency_sigma_multiplier =
        requested_config.internal_observation_weight_residual_consistency_sigma;
    benchmark_input.internal_observation_weight_options
        .residual_consistency_min_rmse =
        requested_config
            .internal_observation_weight_residual_consistency_min_rmse;
    benchmark_input.kalibr_reference = kalibr_reference;
    benchmark_input.additional_camera_references = additional_camera_references;
    benchmark_input.dataset_label = dataset_label;
    benchmark_input.enable_large_intrinsic_perturbation =
        args.enable_large_intrinsic_perturbation;
    benchmark_input.large_intrinsic_perturbation_profile =
        args.large_intrinsic_perturbation_profile;
    benchmark_input.large_intrinsic_perturbation_scale =
        args.large_intrinsic_perturbation_scale;
    benchmark_input.large_intrinsic_perturbation_strict_scale =
        args.large_intrinsic_perturbation_strict_scale;
    benchmark_input.large_intrinsic_perturbation_reference_scene_path =
        args.large_intrinsic_perturbation_reference_scene_path;
    benchmark_input
        .large_intrinsic_perturbation_outer_only_after_application =
        args.large_intrinsic_perturbation_outer_only_after_application;
    benchmark_input.enable_diagnostic_compare =
        args.runtime_mode == ati::Stage5RuntimeMode::Research;
    benchmark_input.multi_board_consistency_diagnostics_options.enabled =
        requested_config.enable_multi_board_consistency_diagnostics;
    benchmark_input.multi_board_consistency_diagnostics_options.pose_source =
        ati::ParseMultiBoardConsistencyPoseSource(
            requested_config.multi_board_consistency_pose_source);
    benchmark_input.multi_board_consistency_diagnostics_options.min_outer_points =
        requested_config.multi_board_consistency_min_outer_points;
    benchmark_input.trial_backend_selection_options.enabled =
        requested_config.enable_trial_backend_frame_board_selection;
    const ati::TrialBackendFrameBoardSelectionOptions::SelectionMode
        trial_backend_selection_mode =
            ParseTrialBackendFrameBoardSelectionMode(
                requested_config.trial_backend_selection_mode);
    benchmark_input.trial_backend_selection_options.selection_mode =
        trial_backend_selection_mode;
    if (requested_config.trial_backend_selection_budget_mode_set) {
      benchmark_input.trial_backend_selection_options.budget_mode =
          ParseTrialBackendFrameBoardSelectionBudgetMode(
              requested_config.trial_backend_selection_budget_mode);
    } else {
      benchmark_input.trial_backend_selection_options.budget_mode =
          trial_backend_selection_mode ==
                  ati::TrialBackendFrameBoardSelectionOptions::SelectionMode::
                      StrictRmse
              ? ati::TrialBackendFrameBoardSelectionOptions::BudgetMode::Fixed
              : ati::TrialBackendFrameBoardSelectionOptions::BudgetMode::
                    KalibrStyle;
    }
    if (requested_config.trial_backend_selection_candidate_order_set) {
      benchmark_input.trial_backend_selection_options.candidate_order_mode =
          ParseTrialBackendFrameBoardCandidateOrderMode(
              requested_config.trial_backend_selection_candidate_order);
      benchmark_input.trial_backend_selection_options
          .candidate_order_mode_explicit = true;
    } else {
      benchmark_input.trial_backend_selection_options.candidate_order_mode =
          ati::TrialBackendFrameBoardSelectionOptions::CandidateOrderMode::
              ScoreSorted;
      benchmark_input.trial_backend_selection_options
          .candidate_order_mode_explicit = false;
    }
    benchmark_input.trial_backend_selection_options.candidate_shuffle_seed_set =
        requested_config.trial_backend_selection_candidate_shuffle_seed_set;
    benchmark_input.trial_backend_selection_options.candidate_shuffle_seed =
        requested_config.trial_backend_selection_candidate_shuffle_seed;
    benchmark_input.trial_backend_selection_options.info_gain_proxy_mode =
        ParseTrialBackendFrameBoardInfoGainProxyMode(
            requested_config.trial_backend_selection_info_gain_proxy_mode);
    benchmark_input.trial_backend_selection_options
        .candidate_batch_granularity =
        ParseTrialBackendFrameBoardCandidateBatchGranularity(
            requested_config.trial_backend_selection_batch_granularity);
    benchmark_input.trial_backend_selection_options.acceptance_policy =
        ati::ParseKalibrStyleBatchAcceptancePolicy(
            requested_config.trial_backend_selection_acceptance_policy);
    benchmark_input.trial_backend_selection_options
        .acceptance_information_gain_threshold =
        requested_config.trial_backend_selection_mi_tol;
    benchmark_input.trial_backend_selection_options
        .acceptance_information_gain_threshold_explicit =
        requested_config.trial_backend_selection_mi_tol_set;
    benchmark_input.trial_backend_selection_options.acceptance_rank_gain_threshold =
        requested_config.trial_backend_selection_rank_threshold;
    benchmark_input.trial_backend_selection_options
        .checkerboard_huber_delta_pixels =
        requested_config.checkerboard_huber_delta_pixels;
    benchmark_input.trial_backend_selection_options
        .checkerboard_outlier_filter_enabled =
        requested_config.checkerboard_outlier_filter_enabled;
    benchmark_input.trial_backend_selection_options.checkerboard_outlier_sigma =
        requested_config.checkerboard_outlier_sigma;
    benchmark_input.trial_backend_selection_options
        .checkerboard_min_inlier_ratio =
        requested_config.checkerboard_min_inlier_ratio;
    benchmark_input.trial_backend_selection_options
        .checkerboard_min_retained_points =
        requested_config.checkerboard_min_retained_points;
    benchmark_input.trial_backend_selection_options.incremental_acceptance =
        requested_config.trial_backend_selection_incremental;
    benchmark_input.trial_backend_selection_options.carry_accepted_trial_state =
        requested_config.trial_backend_selection_carry_accepted_trial_state;
    benchmark_input.trial_backend_selection_options.optimize_intrinsics_in_trial =
        requested_config.trial_backend_selection_optimize_intrinsics;
    benchmark_input.trial_backend_selection_options
        .delayed_intrinsics_release_in_trial =
        requested_config.trial_backend_selection_delayed_intrinsics_release;
    benchmark_input.trial_backend_selection_options.intrinsics_release_iteration =
        requested_config.trial_backend_selection_intrinsics_release_iteration;
    benchmark_input.trial_backend_selection_options
        .persistent_intrinsics_anchor_prior_enabled =
        requested_config
            .trial_backend_selection_persistent_intrinsics_anchor_prior;
    benchmark_input.trial_backend_selection_options
        .persistent_fix_board_layout =
        requested_config.trial_backend_selection_persistent_fix_board_layout;
    benchmark_input.trial_backend_selection_options
        .persistent_intrinsics_anchor_weight_xi_alpha =
        requested_config
            .trial_backend_selection_persistent_intrinsics_anchor_weight_xi_alpha;
    benchmark_input.trial_backend_selection_options
        .persistent_intrinsics_anchor_weight_focal =
        requested_config
            .trial_backend_selection_persistent_intrinsics_anchor_weight_focal;
    benchmark_input.trial_backend_selection_options
        .persistent_intrinsics_anchor_weight_principal =
        requested_config
            .trial_backend_selection_persistent_intrinsics_anchor_weight_principal;
    benchmark_input.trial_backend_selection_options
        .persistent_max_focal_relative_step =
        requested_config
            .trial_backend_selection_persistent_max_focal_relative_step;
    benchmark_input.trial_backend_selection_options
        .persistent_max_principal_step_px =
        requested_config
            .trial_backend_selection_persistent_max_principal_step_px;
    benchmark_input.trial_backend_selection_options
        .persistent_max_xi_alpha_step =
        requested_config
            .trial_backend_selection_persistent_max_xi_alpha_step;
    benchmark_input.trial_backend_selection_options.max_iterations =
        requested_config.trial_backend_selection_max_iterations;
    benchmark_input.trial_backend_selection_options.max_candidate_additions =
        requested_config.trial_backend_selection_max_candidate_additions;
    benchmark_input.trial_backend_selection_options.adaptive_budget_ratio =
        requested_config.trial_backend_selection_adaptive_budget_ratio;
    benchmark_input.trial_backend_selection_options.adaptive_budget_min =
        requested_config.trial_backend_selection_adaptive_budget_min;
    benchmark_input.trial_backend_selection_options.adaptive_budget_max =
        requested_config.trial_backend_selection_adaptive_budget_max;
    benchmark_input.trial_backend_selection_options.runtime_safety_ceiling =
        requested_config.trial_backend_selection_runtime_safety_ceiling;
    benchmark_input.trial_backend_selection_options.outlier_sigma =
        requested_config.trial_backend_selection_outlier_sigma;
    benchmark_input.trial_backend_selection_options.min_abs_threshold_px =
        requested_config.trial_backend_selection_min_abs_threshold_px;
    benchmark_input.trial_backend_selection_options.max_threshold_px =
        requested_config.trial_backend_selection_max_threshold_px;
    benchmark_input.trial_backend_selection_options
        .accept_max_global_rmse_increase_px =
        requested_config
            .trial_backend_selection_accept_max_global_rmse_increase_px;
    benchmark_input.trial_backend_selection_options
        .accept_max_outer_rmse_increase_px =
        requested_config
            .trial_backend_selection_accept_max_outer_rmse_increase_px;
    benchmark_input.trial_backend_selection_options
        .accept_max_internal_rmse_increase_px =
        requested_config
            .trial_backend_selection_accept_max_internal_rmse_increase_px;
    benchmark_input.trial_backend_selection_options.min_candidate_score =
        requested_config.trial_backend_selection_min_candidate_score;
    benchmark_input.trial_backend_selection_options.min_coverage_gain =
        requested_config.trial_backend_selection_min_coverage_gain;
    benchmark_input.trial_backend_selection_options.use_consistency_score =
        requested_config.trial_backend_selection_use_consistency_score;
    benchmark_input.trial_backend_selection_options
        .consistency_translation_sigma_mm =
        requested_config
            .trial_backend_selection_consistency_translation_sigma_mm;
    benchmark_input.trial_backend_selection_options
        .consistency_rotation_sigma_deg =
        requested_config
            .trial_backend_selection_consistency_rotation_sigma_deg;
    benchmark_input.trial_backend_selection_options
        .consistency_penalty_weight =
        requested_config.trial_backend_selection_consistency_penalty_weight;
    benchmark_input.trial_backend_selection_options
        .consistency_max_translation_error_mm =
        requested_config
            .trial_backend_selection_consistency_max_translation_error_mm;
    benchmark_input.trial_backend_selection_options
        .consistency_max_rotation_error_deg =
        requested_config
            .trial_backend_selection_consistency_max_rotation_error_deg;
    benchmark_input.trial_backend_selection_options
        .consistency_max_local_outer_rmse_px =
        requested_config
            .trial_backend_selection_consistency_max_local_outer_rmse_px;
    benchmark_input.trial_backend_selection_options.max_accepted_per_board =
        requested_config.trial_backend_selection_max_accepted_per_board;
    benchmark_input.trial_backend_selection_options.max_accepted_per_frame =
        requested_config.trial_backend_selection_max_accepted_per_frame;
    benchmark_input.trial_backend_selection_options.frame_cohesion_enabled =
        requested_config.trial_backend_selection_frame_cohesion;
    benchmark_input.trial_backend_selection_options
        .frame_cohesion_max_companions_per_frame =
        requested_config
            .trial_backend_selection_frame_cohesion_max_companions;
    benchmark_input.trial_backend_selection_options
        .frame_cohesion_min_candidate_score =
        requested_config
            .trial_backend_selection_frame_cohesion_min_candidate_score;
    benchmark_input.trial_backend_selection_options.min_keep_observations_per_board =
        requested_config.trial_backend_selection_min_keep_per_board;
    benchmark_input.trial_backend_selection_options
        .force_include_list_is_exact_input =
        requested_config
            .trial_backend_selection_force_include_list_is_exact_input;
    benchmark_input.backend_input_ablation_options
        .point_budget_control_enabled =
        requested_config.backend_point_budget_control;
    benchmark_input.backend_input_ablation_options.point_budget_total_points =
        requested_config.backend_point_budget_control_total_points;
    benchmark_input.backend_input_ablation_options.point_budget_seed =
        requested_config.backend_point_budget_control_seed;
    benchmark_input.backend_input_ablation_options
        .max_boards_per_frame_for_ablation =
        requested_config.backend_max_boards_per_frame_for_ablation;
    LoadForceIncludeFrameBoardList(
        requested_config
            .trial_backend_selection_force_include_frame_board_list,
        &benchmark_input.trial_backend_selection_options);
    if (!requested_config.trial_backend_selection_seed_frame_board_list.empty()) {
      ati::TrialBackendFrameBoardSelectionOptions parsed_seed_options;
      LoadForceIncludeFrameBoardList(
          requested_config.trial_backend_selection_seed_frame_board_list,
          &parsed_seed_options);
      benchmark_input.trial_backend_selection_options
          .seed_override_frame_board_keys =
          parsed_seed_options.force_include_frame_board_keys;
      benchmark_input.trial_backend_selection_options
          .seed_override_frame_label_board_keys =
          parsed_seed_options.force_include_frame_label_board_keys;
    }
    if (use_precomputed && precomputed_training.single_board_mode) {
      benchmark_input.trial_backend_selection_options
          .candidate_batch_granularity =
          ati::TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity
              ::Frame;
      benchmark_input.trial_backend_selection_options.frame_cohesion_enabled =
          false;
      benchmark_input.trial_backend_selection_options.use_consistency_score =
          false;
      benchmark_input.multi_board_consistency_diagnostics_options.enabled =
          false;
    }

    const fs::path output_dir(args.output_path);
    EnsureDirectoryExists(output_dir);
    const auto write_runtime_summary = [&runtime_summary, &output_dir, &total_start]() {
      runtime_summary.total_runtime_seconds = ElapsedSeconds(total_start);
      ati::WriteStage5RuntimeSummary(
          (output_dir / "runtime_summary.txt").string(), runtime_summary);
    };

    PrintProgress("running frozen frontend baseline + Stage 5 benchmark...");
    const ati::Stage5BenchmarkReport report = benchmark.Run(benchmark_input);
    runtime_summary.training_detection_cache_hits =
        report.baseline_result.runtime_breakdown.training_detection_cache.cache_hits;
    runtime_summary.training_detection_cache_misses =
        report.baseline_result.runtime_breakdown.training_detection_cache.cache_misses;
    runtime_summary.training_detection_stage_layout_cache_hits =
        report.baseline_result.runtime_breakdown.training_detection_cache
            .stage_layout_cache_hits;
    runtime_summary.training_detection_legacy_layout_cache_hits =
        report.baseline_result.runtime_breakdown.training_detection_cache
            .legacy_layout_cache_hits;
    runtime_summary.training_internal_regeneration_cache_hits =
        report.baseline_result.runtime_breakdown
            .training_internal_regeneration_cache.cache_hits;
    runtime_summary.training_internal_regeneration_cache_misses =
        report.baseline_result.runtime_breakdown
            .training_internal_regeneration_cache.cache_misses;
    runtime_summary.holdout_detection_cache_hits =
        report.runtime_breakdown.holdout_detection_cache.cache_hits;
    runtime_summary.holdout_detection_cache_misses =
        report.runtime_breakdown.holdout_detection_cache.cache_misses;
    runtime_summary.holdout_detection_stage_layout_cache_hits =
        report.runtime_breakdown.holdout_detection_cache
            .stage_layout_cache_hits;
    runtime_summary.holdout_detection_legacy_layout_cache_hits =
        report.runtime_breakdown.holdout_detection_cache
            .legacy_layout_cache_hits;
    runtime_summary.holdout_internal_regeneration_cache_hits =
        report.runtime_breakdown.holdout_internal_regeneration_cache.cache_hits;
    runtime_summary.holdout_internal_regeneration_cache_misses =
        report.runtime_breakdown.holdout_internal_regeneration_cache.cache_misses;
    runtime_summary.round1_regeneration_attempted_internal_corners =
        report.baseline_result.runtime_breakdown
            .round1_regeneration_attempted_internal_corners;
    runtime_summary.round1_regeneration_valid_internal_corners =
        report.baseline_result.runtime_breakdown
            .round1_regeneration_valid_internal_corners;
    runtime_summary.round2_regeneration_attempted_internal_corners =
        report.baseline_result.runtime_breakdown
            .round2_regeneration_attempted_internal_corners;
    runtime_summary.round2_regeneration_valid_internal_corners =
        report.baseline_result.runtime_breakdown
            .round2_regeneration_valid_internal_corners;
    runtime_summary.round1_optimization_residual_evaluation_call_count =
        report.baseline_result.runtime_breakdown
            .round1_optimization_residual_evaluation_call_count;
    runtime_summary.round1_optimization_cost_evaluation_call_count =
        report.baseline_result.runtime_breakdown
            .round1_optimization_cost_evaluation_call_count;
    runtime_summary.round2_optimization_residual_evaluation_call_count =
        report.baseline_result.runtime_breakdown
            .round2_optimization_residual_evaluation_call_count;
    runtime_summary.round2_optimization_cost_evaluation_call_count =
        report.baseline_result.runtime_breakdown
            .round2_optimization_cost_evaluation_call_count;
    AddRuntimeStage(&runtime_summary, "training_outer_detection_load_build",
                    report.baseline_result.runtime_breakdown.training_outer_detection_seconds,
                    false);
    AddRuntimeStage(
        &runtime_summary, "camera_aware_outer_rescue",
        report.baseline_result.runtime_breakdown.camera_aware_outer_rescue_seconds,
        !report.baseline_result.effective_options.enable_camera_aware_outer_rescue);
    AddRuntimeStage(&runtime_summary, "auto_camera_initialization",
                    report.baseline_result.runtime_breakdown
                        .auto_camera_initialization_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "outer_bootstrap",
                    report.baseline_result.runtime_breakdown.outer_bootstrap_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "outer_only_intermediate_measurement_build",
                    report.baseline_result.runtime_breakdown
                        .outer_only_intermediate_measurement_build_seconds,
                    !requested_config.enable_outer_only_intermediate_calibration);
    AddRuntimeStage(&runtime_summary,
                    "outer_only_intermediate_residual_evaluation",
                    report.baseline_result.runtime_breakdown
                        .outer_only_intermediate_residual_evaluation_seconds,
                    !requested_config.enable_outer_only_intermediate_calibration);
    AddRuntimeStage(&runtime_summary, "outer_only_intermediate_selection",
                    report.baseline_result.runtime_breakdown
                        .outer_only_intermediate_selection_seconds,
                    !requested_config.enable_outer_only_intermediate_calibration);
    AddRuntimeStage(&runtime_summary, "outer_only_intermediate_optimization",
                    report.baseline_result.runtime_breakdown
                        .outer_only_intermediate_optimization_seconds,
                    !requested_config.enable_outer_only_intermediate_calibration);
    AddRuntimeStage(&runtime_summary, "round1_regeneration",
                    report.baseline_result.runtime_breakdown.round1_regeneration_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_pose_estimation",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_pose_estimation_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_boundary_model",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_boundary_model_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_seed_search",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_seed_search_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_ray_refine",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_ray_refine_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_image_evidence",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_image_evidence_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_subpix",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_subpix_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_measurement_build",
                    report.baseline_result.runtime_breakdown
                        .round1_measurement_build_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_residual_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round1_residual_evaluation_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_selection",
                    report.baseline_result.runtime_breakdown.round1_selection_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization",
                    report.baseline_result.runtime_breakdown.round1_optimization_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization_residual_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round1_optimization_residual_evaluation_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization_cost_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round1_optimization_cost_evaluation_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization_frame_updates",
                    report.baseline_result.runtime_breakdown
                        .round1_optimization_frame_update_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization_board_updates",
                    report.baseline_result.runtime_breakdown
                        .round1_optimization_board_update_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization_intrinsics_updates",
                    report.baseline_result.runtime_breakdown
                        .round1_optimization_intrinsics_update_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round2_regeneration",
                    report.baseline_result.runtime_breakdown.round2_regeneration_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_pose_estimation",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_pose_estimation_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_boundary_model",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_boundary_model_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_seed_search",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_seed_search_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_ray_refine",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_ray_refine_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_image_evidence",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_image_evidence_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_subpix",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_subpix_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_measurement_build",
                    report.baseline_result.runtime_breakdown
                        .round2_measurement_build_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_residual_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round2_residual_evaluation_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_selection",
                    report.baseline_result.runtime_breakdown.round2_selection_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization",
                    report.baseline_result.runtime_breakdown.round2_optimization_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization_residual_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round2_optimization_residual_evaluation_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization_cost_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round2_optimization_cost_evaluation_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization_frame_updates",
                    report.baseline_result.runtime_breakdown
                        .round2_optimization_frame_update_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization_board_updates",
                    report.baseline_result.runtime_breakdown
                        .round2_optimization_board_update_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization_intrinsics_updates",
                    report.baseline_result.runtime_breakdown
                        .round2_optimization_intrinsics_update_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "holdout_dataset_build",
                    report.runtime_breakdown.holdout_dataset_build_seconds, false);
    AddRuntimeStage(&runtime_summary, "pre_backend_filter",
                    report.runtime_breakdown.pre_backend_filter_seconds,
                    requested_config.pre_backend_filter_mode ==
                        ati::PreBackendFilterMode::Off);
    AddRuntimeStage(&runtime_summary, "internal_joint_refine",
                    report.runtime_breakdown.internal_joint_refine_seconds,
                    requested_config.internal_joint_refine_mode ==
                        ati::InternalJointRefineMode::Off);
    AddRuntimeStage(&runtime_summary, "internal_blur_filter",
                    report.runtime_breakdown.internal_blur_filter_seconds,
                    requested_config.internal_blur_filter_mode ==
                        ati::InternalBlurFilterMode::Off);
    AddRuntimeStage(&runtime_summary, "internal_blur_board_weight",
                    report.runtime_breakdown.internal_blur_board_weight_seconds,
                    requested_config.internal_blur_board_weight_mode ==
                        ati::InternalBlurBoardWeightMode::Off);
    AddRuntimeStage(&runtime_summary, "internal_observation_weighting",
                    report.runtime_breakdown
                        .internal_observation_weight_seconds,
                    requested_config.internal_observation_weight_mode ==
                        ati::InternalObservationWeightMode::Off);
    AddRuntimeStage(&runtime_summary, "diagnostic_compare",
                    report.runtime_breakdown.diagnostic_compare_seconds,
                    args.runtime_mode == ati::Stage5RuntimeMode::Fast);
    PrintProgress("writing Stage 5 benchmark artifacts...");

    if (args.stage5_frontend_only) {
      ati::WriteStage5BenchmarkProtocolSummary(
          (output_dir / "benchmark_protocol_summary.txt").string(), report);
      ati::WriteAutoCameraInitializationSummary(
          (output_dir / "auto_camera_initialization_summary.txt").string(),
          report.baseline_result.auto_camera_initialization);
      ati::WriteAutoCameraInitializationCandidatesCsv(
          (output_dir / "auto_camera_initialization_candidates.csv").string(),
          report.baseline_result.auto_camera_initialization);
      ati::WriteAutoCameraInitializationOuterResidualsCsv(
          (output_dir / "auto_camera_initialization_outer_residuals.csv").string(),
          report.baseline_result.auto_camera_initialization);
      ati::WriteAutoCameraInitializationBootstrapViewsCsv(
          (output_dir / "auto_camera_initialization_bootstrap_views.csv").string(),
          report.baseline_result.auto_camera_initialization);
      WriteCameraAwareOuterRescueArtifacts(output_dir, report.baseline_result);
      ati::WriteInternalRegenerationDiagnostics(output_dir, report);
      ati::WriteFrameBoardObservationFlowDiagnostics(
          output_dir, report, all_frames_for_lookup, nullptr, nullptr);
      if (requested_config.enable_geometry_prior_outer_seed) {
        ati::WriteGeometryPriorOuterSeedDiagnostics(output_dir, report);
      }
      if (requested_config.use_intermediate_for_full_frontend_regeneration) {
        ati::WriteIntermediateFrontendRegenerationSummary(output_dir, report);
      }
      if (requested_config.export_internal_seed_step_overlays) {
        ati::WriteInternalSeedStepOverlays(output_dir, report,
                                           all_frames_for_lookup);
      }

      std::ofstream summary((output_dir / "stage5_frontend_only_summary.txt")
                                .string()
                                .c_str());
      summary << "stage5_frontend_only: 1\n";
      summary << "selection_incremental_ba_run: 0\n";
      summary << "backend_optimization_run: 0\n";
      summary << "holdout_evaluation_run: 0\n";
      summary << "frontend_input_frame_count: "
              << report.split.training_frames.size() << "\n";
      summary << "reference_board_id: "
              << report.baseline_result.reference_board_id << "\n";
      summary << "round1_frame_count: "
              << report.baseline_result.round1.measurement_result.used_frame_count
              << "\n";
      summary << "round1_board_observation_count: "
              << report.baseline_result.round1.measurement_result
                     .used_board_observation_count
              << "\n";
      summary << "round1_outer_point_count: "
              << report.baseline_result.round1.measurement_result
                     .used_outer_point_count
              << "\n";
      summary << "round1_internal_point_count: "
              << report.baseline_result.round1.measurement_result
                     .used_internal_point_count
              << "\n";
      summary << "camera_aware_outer_rescue_requested: "
              << (report.baseline_result.camera_aware_outer_rescue.requested ? 1 : 0)
              << "\n";
      summary << "camera_aware_outer_rescue_count: "
              << report.baseline_result.camera_aware_outer_rescue
                     .rescued_board_observation_count
              << "\n";
      summary << "initialization_camera_family: "
              << report.baseline_result.auto_camera_initialization.selected_camera
                     .NormalizedFamilyString()
              << "\n";
      write_runtime_summary();
      PrintProgress("Stage5 frontend-only completed; no selection or backend BA ran.");
      return report.success ? 0 : 1;
    }

    if (report.baseline_result.stage5_round1_bundle.success) {
      ati::WriteCalibrationStateBundleSummary(
          (output_dir / "stage5_round1_bundle_summary.txt").string(),
          report.baseline_result.stage5_round1_bundle);
    }
    if (report.baseline_result.stage5_bundle_available) {
      ati::WriteCalibrationStateBundleSummary(
          (output_dir / "stage5_bundle_summary.txt").string(),
          report.baseline_result.final_stage5_bundle);
      ati::WriteCalibrationBackendProblemSummary(
          (output_dir / "stage5_backend_problem_summary.txt").string(),
          report.backend_problem_input);
    }
    if (report.final_backend_scene_available) {
      WriteLargePerturbationSceneSnapshot(
          output_dir / "final_persistent_backend_scene.txt",
          report.final_backend_scene);
    }
    ati::WriteBackendInputAblationSummary(
        (output_dir / "backend_input_ablation_summary.txt").string(),
        report.backend_input_ablation_result);
    ati::WriteAutoCameraInitializationSummary(
        (output_dir / "auto_camera_initialization_summary.txt").string(),
        report.baseline_result.auto_camera_initialization);
    ati::WriteAutoCameraInitializationCandidatesCsv(
        (output_dir / "auto_camera_initialization_candidates.csv").string(),
        report.baseline_result.auto_camera_initialization);
    ati::WriteAutoCameraInitializationRefinedBasinsCsv(
        (output_dir / "auto_camera_initialization_refined_basins.csv").string(),
        report.baseline_result.auto_camera_initialization);
    ati::WriteAutoCameraInitializationOuterResidualsCsv(
        (output_dir / "auto_camera_initialization_outer_residuals.csv").string(),
        report.baseline_result.auto_camera_initialization);
    ati::WriteAutoCameraInitializationBootstrapViewsCsv(
        (output_dir / "auto_camera_initialization_bootstrap_views.csv").string(),
        report.baseline_result.auto_camera_initialization);
    if (report.large_intrinsic_perturbation.enabled) {
      const ati::Stage5LargeIntrinsicPerturbationState& perturbation =
          report.large_intrinsic_perturbation;
      std::ofstream summary(
          (output_dir / "large_intrinsic_perturbation_summary.txt")
              .string()
              .c_str());
      summary << std::setprecision(12);
      summary << "enabled: 1\n";
      summary << "requested_profile: " << perturbation.requested_profile << "\n";
      summary << "effective_profile: " << perturbation.effective_profile << "\n";
      summary << "application_stage: after_internal_recovery_before_selection_incremental_ba\n";
      summary << "internal_observations_regenerated_after_perturbation: 0\n";
      summary << "internal_observations_refiltered_after_perturbation: 0\n";
      summary << "outer_only_after_application: "
              << (perturbation.outer_only_after_application ? 1 : 0) << "\n";
      summary << "frozen_internal_point_count_before_ablation: "
              << perturbation.frozen_internal_point_count_before_ablation
              << "\n";
      summary << "seed_internal_point_count_after_ablation: "
              << perturbation.seed_internal_point_count_after_ablation
              << "\n";
      summary << "candidate_pool_internal_point_count_after_ablation: "
              << perturbation.candidate_pool_internal_point_count_after_ablation
              << "\n";
      summary << "reference_scene_fingerprint: "
              << perturbation.reference_scene_fingerprint << "\n";
      summary << "perturbed_scene_fingerprint: "
              << perturbation.perturbed_scene_fingerprint << "\n";
      summary << "frozen_observation_fingerprint: "
              << perturbation.frozen_observation_fingerprint << "\n";
      summary << "requested_focal_scale: "
              << perturbation.requested_focal_scale << "\n";
      summary << "requested_xi_delta: " << perturbation.requested_xi_delta << "\n";
      summary << "requested_alpha_delta: "
              << perturbation.requested_alpha_delta << "\n";
      summary << "requested_scale: " << perturbation.requested_scale << "\n";
      summary << "effective_scale: " << perturbation.effective_scale << "\n";
      summary << "actual_focal_scale: " << perturbation.actual_focal_scale << "\n";
      summary << "actual_xi_delta: " << perturbation.actual_xi_delta << "\n";
      summary << "actual_alpha_delta: " << perturbation.actual_alpha_delta << "\n";
      summary << "projection_grid_width: "
              << perturbation.projection_grid_width << "\n";
      summary << "projection_grid_height: "
              << perturbation.projection_grid_height << "\n";
      summary << "valid_projection_grid_count: "
              << perturbation.valid_projection_grid_count << "\n";
      summary << "invalid_projection_grid_count: "
              << perturbation.invalid_projection_grid_count << "\n";
      summary << "valid_projection_grid: "
              << (perturbation.valid_projection_grid ? 1 : 0) << "\n";
      summary << "failure_reason: " << perturbation.failure_reason << "\n";
      summary << "reference_camera_intrinsics: "
              << JoinDoubleVector(perturbation.reference_camera.IntrinsicsVector(), ",")
              << "\n";
      summary << "reference_camera_distortion: "
              << JoinDoubleVector(perturbation.reference_camera.DistortionVector(), ",")
              << "\n";
      summary << "perturbed_camera_intrinsics: "
              << JoinDoubleVector(perturbation.perturbed_camera.IntrinsicsVector(), ",")
              << "\n";
      summary << "perturbed_camera_distortion: "
              << JoinDoubleVector(perturbation.perturbed_camera.DistortionVector(), ",")
              << "\n";
      summary << "selection_seed_camera_intrinsics: "
              << JoinDoubleVector(
                     perturbation.selection_seed_camera.IntrinsicsVector(), ",")
              << "\n";
      summary << "selection_seed_camera_distortion: "
              << JoinDoubleVector(
                     perturbation.selection_seed_camera.DistortionVector(), ",")
              << "\n";
      summary << "selection_candidate_camera_intrinsics: "
              << JoinDoubleVector(
                     perturbation.selection_candidate_camera.IntrinsicsVector(),
                     ",")
              << "\n";
      summary << "selection_candidate_camera_distortion: "
              << JoinDoubleVector(
                     perturbation.selection_candidate_camera.DistortionVector(),
                     ",")
              << "\n";
      summary << "selection_seed_matches_perturbed_camera: "
              << (perturbation.selection_seed_matches_perturbed_camera ? 1 : 0)
              << "\n";
      summary << "selection_candidate_matches_perturbed_camera: "
              << (perturbation.selection_candidate_matches_perturbed_camera ? 1
                                                                            : 0)
              << "\n";
      WriteIntermediateCameraYaml(output_dir / "large_perturbation_reference_camera.yaml",
                                  perturbation.reference_camera);
      WriteIntermediateCameraYaml(output_dir / "large_perturbation_initial_camera.yaml",
                                  perturbation.perturbed_camera);
      WriteLargePerturbationSceneSnapshot(
          output_dir / "large_intrinsic_perturbation_reference_scene.txt",
          perturbation.reference_scene);
      const ati::MultiBoardPoseOrientationEvaluationResult& pose_eval =
          report.large_perturbation_pose_orientation_evaluation;
      std::ofstream pose_summary(
          (output_dir / "large_perturbation_pose_orientation_summary.txt")
              .string().c_str());
      pose_summary << std::setprecision(12);
      pose_summary << "success: " << (pose_eval.success ? 1 : 0) << "\n";
      pose_summary << "evaluated_frame_count: " << pose_eval.evaluated_frame_count
                   << "\n";
      pose_summary << "pose_success_count: " << pose_eval.pose_success_count
                   << "\n";
      pose_summary << "pose_success_rate: " << pose_eval.pose_success_rate
                   << "\n";
      pose_summary << "orientation_median_deg: "
                   << pose_eval.orientation_median_deg << "\n";
      pose_summary << "orientation_p95_deg: "
                   << pose_eval.orientation_p95_deg << "\n";
      pose_summary << "failure_reason: " << pose_eval.failure_reason << "\n";
    }
    ati::WriteAutoCameraInitializationPoseExcitationCsv(
        (output_dir / "auto_camera_initialization_pose_excitation.csv").string(),
        report.baseline_result.auto_camera_initialization);
    ati::WriteAutoCameraInitializationPoseExcitationSamplesCsv(
        (output_dir /
         "auto_camera_initialization_pose_excitation_samples.csv").string(),
        report.baseline_result.auto_camera_initialization);
    if (report.baseline_result.auto_camera_initialization
            .stage5_init_principal_profile_enabled != 0) {
      ati::WriteAutoCameraInitializationPrincipalProfileCsv(
          (output_dir / "auto_camera_initialization_principal_profile.csv")
              .string(),
          report.baseline_result.auto_camera_initialization);
    }
    if (report.baseline_result.auto_camera_initialization
            .stage5_init_fixed_layout_principal_profile_enabled != 0) {
      ati::WriteAutoCameraInitializationFixedLayoutPrincipalProfileCsv(
          (output_dir /
           "auto_camera_initialization_fixed_layout_principal_profile.csv")
              .string(),
          report.baseline_result.auto_camera_initialization);
    }
    if (report.baseline_result.auto_camera_initialization
            .stage5_init_board_jackknife_diagnostic_enabled != 0) {
      ati::WriteAutoCameraInitializationBoardJackknifeCsv(
          (output_dir / "auto_camera_initialization_board_jackknife.csv")
              .string(),
          report.baseline_result.auto_camera_initialization);
    }
    if (report.baseline_result.auto_camera_initialization
            .stage5_init_coverage_weighted_diagnostic_enabled != 0) {
      ati::WriteAutoCameraInitializationCoverageWeightsCsv(
          (output_dir / "auto_camera_initialization_coverage_weights.csv")
              .string(),
          report.baseline_result.auto_camera_initialization);
    }
    WriteAutoCameraInitializationBootstrapCornerOverlays(
        output_dir / "auto_camera_initialization_bootstrap_corner_overlays",
        report.baseline_result.auto_camera_initialization,
        report.baseline_result.frame_sources);
    WriteOuterOnlyIntermediateArtifacts(output_dir, report.baseline_result);
    WriteCameraAwareOuterRescueArtifacts(output_dir, report.baseline_result);
    WriteExperimentConfigSummary(
        (output_dir / "experiment_config_summary.txt").string(),
        requested_config,
        report,
        nullptr);
    ati::WritePreBackendFilterSummary(
        (output_dir / "pre_backend_filter_summary.txt").string(),
        report.pre_backend_filter_result);
    ati::WritePreBackendFilterPointsCsv(
        (output_dir / "pre_backend_filter_points.csv").string(),
        report.pre_backend_filter_result);
    ati::WritePreBackendFilterBoardSummaryCsv(
        (output_dir / "pre_backend_filter_board_summary.csv").string(),
        report.pre_backend_filter_result);
    ati::WritePreBackendFilterFrameSummaryCsv(
        (output_dir / "pre_backend_filter_frame_summary.csv").string(),
        report.pre_backend_filter_result);
    ati::WriteInternalJointRefineSummary(
        (output_dir / "internal_joint_refine_summary.txt").string(),
        report.internal_joint_refine_result);
    ati::WriteInternalJointRefinePointsCsv(
        (output_dir / "internal_joint_refine_points.csv").string(),
        report.internal_joint_refine_result);
    ati::WriteInternalJointRefineBoardSummaryCsv(
        (output_dir / "internal_joint_refine_board_summary.csv").string(),
        report.internal_joint_refine_result);
    ati::WriteInternalJointRefineFrameSummaryCsv(
        (output_dir / "internal_joint_refine_frame_summary.csv").string(),
        report.internal_joint_refine_result);
    ati::WriteInternalBlurFilterSummary(
        (output_dir / "internal_blur_filter_summary.txt").string(),
        report.internal_blur_filter_result);
    ati::WriteInternalBlurFilterBoardDecisionsCsv(
        (output_dir / "internal_blur_filter_board_decisions.csv").string(),
        report.internal_blur_filter_result);
    ati::WriteInternalBlurFilterPointDecisionsCsv(
        (output_dir / "internal_blur_filter_point_decisions.csv").string(),
        report.internal_blur_filter_result);
    ati::WriteInternalBlurFilterFrameSummaryCsv(
        (output_dir / "internal_blur_filter_frame_summary.csv").string(),
        report.internal_blur_filter_result);
    ati::WriteInternalBlurBoardWeightSummary(
        (output_dir / "internal_blur_board_weight_summary.txt").string(),
        report.internal_blur_board_weight_result);
    ati::WriteInternalBlurBoardWeightPointsCsv(
        (output_dir / "internal_blur_board_weight_points.csv").string(),
        report.internal_blur_board_weight_result);
    ati::WriteInternalBlurBoardWeightBoardSummaryCsv(
        (output_dir / "internal_blur_board_weight_board_summary.csv")
            .string(),
        report.internal_blur_board_weight_result);
    ati::WriteInternalObservationWeightSummary(
        (output_dir / "internal_observation_weight_summary.txt").string(),
        report.internal_observation_weight_result);
    ati::WriteInternalObservationWeightsCsv(
        (output_dir / "internal_observation_weights.csv").string(),
        report.internal_observation_weight_result);
    ati::WriteInternalObservationWeightBoardSummaryCsv(
        (output_dir / "internal_observation_weight_board_summary.csv")
            .string(),
        report.internal_observation_weight_result);
    ati::WriteStage5BenchmarkProtocolSummary(
        (output_dir / "benchmark_protocol_summary.txt").string(), report);
    {
      std::ofstream trial_csv(
          (output_dir / "stage5_trial_backend_optimizer_diagnostics.csv")
              .string()
              .c_str());
      trial_csv
          << "label,success,design_variable_count,error_term_count,"
          << "initial_overall_rmse,optimized_overall_rmse,"
          << "initial_outer_rmse,optimized_outer_rmse,"
          << "initial_internal_rmse,optimized_internal_rmse,"
          << "xi_before,alpha_before,fu_before,fv_before,cu_before,cv_before,"
          << "xi_after,alpha_after,fu_after,fv_after,cu_after,cv_after,"
          << "stage_count,total_iterations,total_failed_iterations,"
          << "any_intrinsics_stage,any_linear_solver_failure,"
          << "objective_start_sum,objective_final_sum,"
          << "last_delta_x,last_delta_j,last_lm_lambda,failure_reason\n";
      for (const ati::TrialBackendOptimizationDiagnostics& diag :
           report.trial_backend_selection_result.trial_optimization_diagnostics) {
        trial_csv << CsvEscape(diag.label) << ","
                  << (diag.success ? 1 : 0) << ","
                  << diag.design_variable_count << ","
                  << diag.error_term_count << ","
                  << diag.initial_overall_rmse << ","
                  << diag.optimized_overall_rmse << ","
                  << diag.initial_outer_rmse << ","
                  << diag.optimized_outer_rmse << ","
                  << diag.initial_internal_rmse << ","
                  << diag.optimized_internal_rmse << ","
                  << diag.camera_xi_before << ","
                  << diag.camera_alpha_before << ","
                  << diag.camera_fu_before << ","
                  << diag.camera_fv_before << ","
                  << diag.camera_cu_before << ","
                  << diag.camera_cv_before << ","
                  << diag.camera_xi_after << ","
                  << diag.camera_alpha_after << ","
                  << diag.camera_fu_after << ","
                  << diag.camera_fv_after << ","
                  << diag.camera_cu_after << ","
                  << diag.camera_cv_after << ","
                  << diag.stage_count << ","
                  << diag.total_iterations << ","
                  << diag.total_failed_iterations << ","
                  << (diag.any_intrinsics_stage ? 1 : 0) << ","
                  << (diag.any_linear_solver_failure ? 1 : 0) << ","
                  << diag.objective_start_sum << ","
                  << diag.objective_final_sum << ","
                  << diag.last_delta_x << ","
                  << diag.last_delta_j << ","
                  << diag.last_lm_lambda << ","
                  << CsvEscape(diag.failure_reason) << "\n";
      }
    }
    ati::WriteStage5BenchmarkTrainingSummary(
        (output_dir / "benchmark_training_summary.txt").string(), report);
    ati::WriteStage5BenchmarkHoldoutSummary(
        (output_dir / "benchmark_holdout_summary.txt").string(), report);
    ati::WritePersistentCameraCheckpointEvaluationsCsv(
        (output_dir / "persistent_camera_checkpoint_evaluations.csv").string(),
        report);
    {
      std::vector<ati::CameraModelRefitEvaluationResult> training_evaluations{
          report.our_training_evaluation,
          report.kalibr_training_evaluation};
      training_evaluations.insert(training_evaluations.end(),
                                  report.additional_training_evaluations.begin(),
                                  report.additional_training_evaluations.end());
      ati::WriteCameraModelRefitPointsCsv(
          (output_dir / "benchmark_training_points.csv").string(),
          training_evaluations);
    }
    ati::WriteStage5BenchmarkHoldoutPointsCsv(
        (output_dir / "benchmark_holdout_points.csv").string(), report);
    ati::WriteStage5BenchmarkHoldoutBoardObservationsCsv(
        (output_dir / "benchmark_holdout_board_observations.csv").string(),
        report);
    ati::WriteStage5BenchmarkHoldoutFramesCsv(
        (output_dir / "benchmark_holdout_frames.csv").string(), report);
    ati::WriteCameraRayCurveSamplesCsv(
        (output_dir / "camera_ray_curve_samples.csv").string(),
        report.camera_ray_curve_diagnostics);
    ati::WriteCameraRayCurveSummaryCsv(
        (output_dir / "camera_ray_curve_summary.csv").string(),
        report.camera_ray_curve_diagnostics);
    ati::WriteStage5BenchmarkWorstCasesSummary(
        (output_dir / "benchmark_worst_cases_summary.txt").string(), report, 10);
    ati::WriteStage5BenchmarkHoldoutRobustOutlierSummary(
        (output_dir / "benchmark_holdout_robust_outlier_summary.txt").string(),
        report, 5.0);
    const bool write_selection_diagnostics =
        requested_config.enable_stage5_selection_diagnostics ||
        requested_config.enable_trial_backend_frame_board_selection ||
        requested_config.selection_mode == "kalibr_style_frame_board" ||
        requested_config.selection_mode == "kalibr_style_batch";
    if (requested_config.internal_regeneration_diagnostics) {
      ati::WriteInternalRegenerationDiagnostics(output_dir, report);
      if (requested_config.export_internal_seed_step_overlays) {
        ati::WriteInternalSeedStepOverlays(output_dir, report, all_frames_for_lookup);
      }
      if (requested_config.enable_geometry_prior_outer_seed) {
        ati::WriteGeometryPriorOuterSeedDiagnostics(output_dir, report);
      }
    }
    if (!use_precomputed &&
        (requested_config.internal_regeneration_diagnostics ||
         write_selection_diagnostics)) {
      ati::WriteFrameBoardObservationFlowDiagnostics(
          output_dir, report, all_frames_for_lookup, nullptr,
          nullptr);
    }
    if (requested_config.internal_regeneration_diagnostics &&
        requested_config.use_intermediate_for_full_frontend_regeneration) {
      ati::WriteIntermediateFrontendRegenerationSummary(output_dir, report);
    }
    if (requested_config.enable_global_scene_state_consistency_audit) {
      ati::WriteGlobalSceneStateConsistencyAudit(output_dir, report);
    }
    if (args.runtime_mode == ati::Stage5RuntimeMode::Research &&
        report.diagnostic_compare.success) {
      ati::WriteKalibrBenchmarkIntrinsicsCsv(
          (output_dir / "benchmark_intrinsics_compare.csv").string(),
          report.diagnostic_compare);
    }

    if (!use_precomputed &&
        args.runtime_mode == ati::Stage5RuntimeMode::Research) {
      const cv::Mat projection_compare = benchmark.RenderProjectionComparison(report);
      if (!projection_compare.empty()) {
        cv::imwrite((output_dir / "benchmark_projection_compare.png").string(),
                    projection_compare);
      }
    }

    if (!report.success) {
      write_runtime_summary();
      std::cout << "Stage 5 benchmark success: 0\n"
                << "Protocol summary: "
                << (output_dir / "benchmark_protocol_summary.txt").string() << "\n";
      return 1;
    }
    PrintProgress("Stage 5 benchmark completed.");
    PrintProgress("protocol=" + requested_config.effective_protocol_label);
    PrintProgress("camera init mode=" +
                  std::string(ati::ToString(
                      report.baseline_result.auto_camera_initialization.selected_mode)) +
                  " source=" +
                  report.baseline_result.auto_camera_initialization.selected_source_label +
                  " fallback=" +
                  std::to_string(
                      report.baseline_result.auto_camera_initialization.fallback_used ? 1 : 0));
    PrintEvaluationProgress("frontend training", report.our_training_evaluation);
    PrintEvaluationProgress("frontend holdout", report.our_holdout_evaluation);
    PrintEvaluationProgress("kalibr training", report.kalibr_training_evaluation);
    PrintEvaluationProgress("kalibr holdout", report.kalibr_holdout_evaluation);

    ati::AslamBackendCalibrationOptions runner_options;
    runner_options.uniform_control_point_mode =
        report.precomputed_single_board_ba_mode;
    runner_options.max_iterations = 12;
    runner_options.convergence_delta_j = 1e-3;
    runner_options.convergence_delta_x = 1e-4;
    runner_options.levenberg_marquardt_lambda_init = 1e-3;
    runner_options.linear_solver = "cholmod";
    runner_options.verbose = false;
    runner_options.use_huber_loss = true;
    runner_options.outer_huber_delta_pixels = 10.0;
    runner_options.internal_huber_delta_pixels = 6.0;
    runner_options.invalid_projection_penalty_pixels = 100.0;
    runner_options.polar_angle_weight_mode =
        ati::ParsePolarAngleWeightMode(
            requested_config.backend_polar_angle_weight_mode);
    runner_options.polar_angle_weight_bin_edges_deg =
        requested_config.backend_polar_angle_weight_bin_edges_deg;
    runner_options.polar_angle_weight_fixed_bin_scales =
        requested_config.backend_polar_angle_weight_fixed_bin_scales;
    runner_options.polar_angle_weight_adaptive_sigma_reference_deg =
        requested_config.backend_polar_angle_weight_adaptive_sigma_reference_deg;
    runner_options.polar_angle_weight_adaptive_sigma_growth =
        requested_config.backend_polar_angle_weight_adaptive_sigma_growth;
    runner_options.polar_angle_weight_min_scale =
        requested_config.backend_polar_angle_weight_min_scale;
    runner_options.residual_model =
        ati::ParseResidualModel(requested_config.backend_residual_model);
    runner_options.hybrid_angular_threshold_deg =
        requested_config.backend_hybrid_angular_threshold_deg;
    runner_options.outer_residual_model =
        ati::ParseResidualModel(requested_config.backend_outer_residual_model);
    runner_options.internal_residual_model =
        ati::ParseResidualModel(requested_config.backend_internal_residual_model);
    runner_options.use_point_type_residual_split =
        requested_config.backend_use_point_type_residual_split;
    runner_options.angular_auxiliary_enabled =
        requested_config.backend_enable_angular_auxiliary_residual;
    runner_options.angular_auxiliary_weight =
        requested_config.backend_angular_auxiliary_weight;
    runner_options.angular_auxiliary_normalized =
        requested_config.backend_angular_auxiliary_normalized;
    runner_options.angular_auxiliary_apply_to_outer =
        requested_config.backend_angular_auxiliary_apply_to_outer;
    runner_options.angular_auxiliary_apply_to_internal =
        requested_config.backend_angular_auxiliary_apply_to_internal;
    runner_options.polar_continuous_hybrid_threshold_deg =
        requested_config.backend_polar_continuous_hybrid_threshold_deg;
    runner_options.polar_continuous_hybrid_temperature_deg =
        requested_config.backend_polar_continuous_hybrid_temperature_deg;
    runner_options.normalized_angular_reference_sigma_px =
        requested_config.backend_normalized_angular_reference_sigma_px;
    runner_options.normalized_angular_min_sigma_rad =
        requested_config.backend_normalized_angular_min_sigma_rad;
    runner_options.normalized_angular_max_weight_scale =
        requested_config.backend_normalized_angular_max_weight_scale;
    runner_options.pixel_residual_weight =
        requested_config.backend_pixel_residual_weight;
    runner_options.chordal_residual_weight =
        requested_config.backend_chordal_residual_weight;
    runner_options.angular_use_normalize_jacobian =
        requested_config.backend_angular_use_normalize_jacobian;
    runner_options.angular_local_whitening_enabled =
        requested_config.backend_angular_local_whitening;
    runner_options.angular_local_whitening_pixel_sigma_px =
        requested_config.backend_angular_local_whitening_pixel_sigma_px;
    runner_options.angular_local_whitening_covariance_damping =
        requested_config.backend_angular_local_whitening_covariance_damping;
    runner_options.angular_local_whitening_min_sigma_rad =
        requested_config.backend_angular_local_whitening_min_sigma_rad;
    runner_options.angular_local_whitening_max_weight =
        requested_config.backend_angular_local_whitening_max_weight;
    runner_options.angular_observed_ray_mode =
        ati::ParseAngularObservedRayMode(
            requested_config.backend_angular_observed_ray_mode);
    runner_options.board_pose_parameterization =
        ati::ParseBoardPoseParameterization(
            requested_config.backend_board_pose_parameterization);
    runner_options.force_pose_only = requested_config.backend_fixed_intrinsics;
    runner_options.multi_board_consistency_weighting =
        requested_config.backend_multi_board_consistency_weighting;
    runner_options.consistency_pose_source =
        requested_config.backend_consistency_pose_source;
    runner_options.consistency_weight_mode =
        ati::ParseConsistencyWeightMode(
            requested_config.backend_consistency_weight_mode);
    runner_options.consistency_translation_sigma_mm =
        requested_config.backend_consistency_translation_sigma_mm;
    runner_options.consistency_rotation_sigma_deg =
        requested_config.backend_consistency_rotation_sigma_deg;
    runner_options.consistency_min_weight =
        requested_config.backend_consistency_min_weight;
    runner_options.consistency_apply_to_outer =
        requested_config.backend_consistency_apply_to_outer;
    runner_options.consistency_apply_to_internal =
        requested_config.backend_consistency_apply_to_internal;
    runner_options.consistency_hard_reject_enabled =
        requested_config.backend_consistency_hard_reject_enabled;
    runner_options.consistency_hard_reject_translation_mm =
        requested_config.backend_consistency_hard_reject_translation_mm;
    runner_options.consistency_hard_reject_rotation_deg =
        requested_config.backend_consistency_hard_reject_rotation_deg;
    runner_options.consistency_hard_reject_residual_px =
        requested_config.backend_consistency_hard_reject_residual_px;
    runner_options.consistency_dump_weight_summary =
        requested_config.backend_consistency_dump_weight_summary;
    runner_options.board_pose_prior_enabled =
        requested_config.backend_board_pose_prior;
    runner_options.board_pose_prior_translation_sigma_mm =
        requested_config.backend_board_pose_prior_translation_sigma_mm;
    runner_options.board_pose_prior_rotation_sigma_deg =
        requested_config.backend_board_pose_prior_rotation_sigma_deg;
    runner_options.enable_angular_residual_diagnostics =
        requested_config.enable_angular_residual_diagnostics;
    runner_options.angular_residual_bin_edges_deg =
        requested_config.angular_residual_bin_edges_deg;
    // Stage5 follows the Kalibr-style paper baseline: optimization happens
    // during incremental selection BA. The tail backend runner is diagnostics
    // only and must never perform a final BA.
    runner_options.skip_optimization = true;
    runner_options.export_cost_parity_diagnostics =
        args.runtime_mode == ati::Stage5RuntimeMode::Research;
    runner_options.export_variable_block_influence_diagnostics =
        runner_options.export_cost_parity_diagnostics;
    const ati::AslamBackendCalibrationRunner backend_runner(runner_options);
    PrintProgress(
        "final backend BA removed; evaluating incremental-selection committed state...");
    const auto backend_stage_start = std::chrono::steady_clock::now();
    ati::AslamBackendCalibrationResult backend_result =
        backend_runner.Run(report.backend_problem_input);
    AddRuntimeStage(&runtime_summary,
                    "backend_committed_state_evaluation",
                    ElapsedSeconds(backend_stage_start), false);
    PrintBackendResultProgress(backend_result);
    WriteBackendDiagnosticArtifacts(output_dir, "backend_optimization", backend_result);
    if (backend_result.success) {
      ati::WriteBoardLayoutPoseDeltaCsv(
          (output_dir / "board_layout_pose_delta.csv").string(),
          backend_result);
    }
    WriteExperimentConfigSummary(
        (output_dir / "experiment_config_summary.txt").string(),
        requested_config,
        report,
        &backend_result);

    if (!backend_result.success) {
      write_runtime_summary();
      std::cout << "Backend summary: "
                << (output_dir / "backend_optimization_summary.txt").string() << "\n";
      return 1;
    }

    {
      const bool enabled =
          requested_config.enable_hybrid_pixel_ray_final_refinement;
      bool attempted = false;
      bool succeeded = false;
      bool committed = false;
      std::string failure_reason = enabled ? "not_attempted" : "disabled";
      double runtime_seconds = 0.0;
      ati::AslamBackendCalibrationResult hybrid_result;

      if (enabled) {
        attempted = true;
        WriteBackendDiagnosticArtifacts(
            output_dir, "pixel_committed_state", backend_result);
        ati::CalibrationBackendProblemInput hybrid_input =
            backend_result.effective_problem_input;
        hybrid_input.scene_state.camera =
            backend_result.optimized_scene_state.camera;
        hybrid_input.scene_state.frames =
            backend_result.optimized_scene_state.frames;
        hybrid_input.scene_state.boards =
            backend_result.optimized_scene_state.boards;
        hybrid_input.scene_state.coarse_or_optimized_level =
            "pixel_committed_before_pixel_ray_hybrid_refinement";
        hybrid_input.optimization_masks.delayed_intrinsics_release = false;

        ati::AslamBackendCalibrationOptions hybrid_options = runner_options;
        hybrid_options.skip_optimization = false;
        hybrid_options.max_iterations =
            requested_config.hybrid_pixel_ray_max_iterations;
        hybrid_options.residual_model = ati::ResidualModel::ImagePlane;
        hybrid_options.use_point_type_residual_split = false;
        hybrid_options.angular_auxiliary_enabled = false;
        hybrid_options.angular_observed_ray_mode =
            ati::AslamBackendCalibrationOptions::AngularObservedRayMode::
                DynamicCurrentCamera;
        hybrid_options.pixel_ray_hybrid_refinement_mode = true;
        hybrid_options.pixel_ray_hybrid_lambda =
            requested_config.hybrid_pixel_ray_lambda;
        hybrid_options.pixel_ray_hybrid_polar_adaptive_enabled =
            requested_config.hybrid_pixel_ray_polar_adaptive;
        hybrid_options.pixel_ray_hybrid_lambda_min =
            requested_config.hybrid_pixel_ray_lambda_min;
        hybrid_options.pixel_ray_hybrid_lambda_max =
            requested_config.hybrid_pixel_ray_lambda_max;
        hybrid_options.pixel_ray_hybrid_transition_start_deg =
            requested_config.hybrid_pixel_ray_transition_start_deg;
        hybrid_options.pixel_ray_hybrid_transition_end_deg =
            requested_config.hybrid_pixel_ray_transition_end_deg;
        hybrid_options.pixel_ray_hybrid_pixel_scale_floor =
            requested_config.hybrid_pixel_ray_pixel_scale_floor;
        hybrid_options.pixel_ray_hybrid_ray_scale_floor =
            requested_config.hybrid_pixel_ray_ray_scale_floor;
        hybrid_options.export_cost_parity_diagnostics = false;
        hybrid_options.export_variable_block_influence_diagnostics =
            args.runtime_mode == ati::Stage5RuntimeMode::Research;
        hybrid_options.run_jacobian_consistency_check =
            args.runtime_mode == ati::Stage5RuntimeMode::Research;

        PrintProgress(
            "running optional Hybrid Pixel-Ray post-selection refinement...");
        const auto hybrid_start = std::chrono::steady_clock::now();
        const ati::AslamBackendCalibrationRunner hybrid_runner(hybrid_options);
        hybrid_result = hybrid_runner.Run(hybrid_input);
        runtime_seconds = ElapsedSeconds(hybrid_start);
        AddRuntimeStage(&runtime_summary,
                        "hybrid_pixel_ray_final_refinement",
                        runtime_seconds, false);
        WriteBackendDiagnosticArtifacts(
            output_dir, "hybrid_pixel_ray_refinement_backend", hybrid_result);
        bool objective_nonincreasing = !hybrid_result.stages.empty();
        for (const auto& stage : hybrid_result.stages) {
          objective_nonincreasing =
              objective_nonincreasing &&
              std::isfinite(stage.objective_start) &&
              std::isfinite(stage.objective_final) &&
              stage.objective_final <=
                  stage.objective_start +
                      1e-10 * std::max(1.0, std::abs(stage.objective_start));
        }
        succeeded = hybrid_result.success && objective_nonincreasing &&
                    hybrid_result.optimized_residual.success &&
                    hybrid_result.optimized_scene_state.camera.IsValid() &&
                    std::isfinite(
                        hybrid_result.optimized_residual.overall_rmse);
        if (succeeded) {
          backend_result = hybrid_result;
          committed = true;
          failure_reason.clear();
          WriteBackendDiagnosticArtifacts(
              output_dir, "backend_optimization", backend_result);
          ati::WriteBoardLayoutPoseDeltaCsv(
              (output_dir / "board_layout_pose_delta.csv").string(),
              backend_result);
          PrintProgress(
              "Hybrid Pixel-Ray refinement committed from training-only "
              "numerical health.");
        } else {
          failure_reason = hybrid_result.failure_reason.empty()
                               ? "hybrid_result_failed_numerical_health"
                               : hybrid_result.failure_reason;
          PrintProgress(
              "Hybrid Pixel-Ray refinement rolled back; keeping Pixel "
              "committed state: " + failure_reason);
        }
      }

      std::ofstream hybrid_summary(
          (output_dir /
           "hybrid_pixel_ray_final_refinement_summary.txt").string().c_str());
      hybrid_summary << std::setprecision(12);
      hybrid_summary << "stage5_hybrid_pixel_ray_final_refinement_enabled: "
                     << (enabled ? 1 : 0) << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_final_refinement_attempted: "
                     << (attempted ? 1 : 0) << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_final_refinement_success: "
                     << (succeeded ? 1 : 0) << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_final_refinement_committed: "
                     << (committed ? 1 : 0) << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_lambda: "
                     << requested_config.hybrid_pixel_ray_lambda << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_polar_adaptive_enabled: "
                     << (requested_config.hybrid_pixel_ray_polar_adaptive ? 1 : 0)
                     << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_lambda_min: "
                     << requested_config.hybrid_pixel_ray_lambda_min << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_lambda_max: "
                     << requested_config.hybrid_pixel_ray_lambda_max << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_transition_start_deg: "
                     << requested_config.hybrid_pixel_ray_transition_start_deg
                     << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_transition_end_deg: "
                     << requested_config.hybrid_pixel_ray_transition_end_deg
                     << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_residual_dimension: 4\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_scale_source: "
                     << (enabled ? "pixel_committed_training_state"
                                 : "disabled")
                     << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_scales_computed_once: "
                     << (hybrid_result.pixel_ray_hybrid_scales_computed_once
                             ? 1
                             : 0)
                     << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_valid_observation_count: "
                     << hybrid_result.pixel_ray_hybrid_valid_observation_count
                     << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_invalid_observation_count: "
                     << hybrid_result.pixel_ray_hybrid_invalid_observation_count
                     << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_s_px: "
                     << hybrid_result.pixel_ray_hybrid_pixel_scale << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_s_ray: "
                     << hybrid_result.pixel_ray_hybrid_ray_scale << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_initial_pixel_rmse: "
                     << hybrid_result.initial_residual.overall_image_plane_rmse
                     << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_final_pixel_rmse: "
                     << hybrid_result.optimized_residual.overall_image_plane_rmse
                     << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_initial_ray_rmse: "
                     << hybrid_result.initial_residual.overall_angular_rmse
                     << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_final_ray_rmse: "
                     << hybrid_result.optimized_residual.overall_angular_rmse
                     << "\n";
      int hybrid_iterations = 0;
      int hybrid_failed_iterations = 0;
      double hybrid_objective_start =
          std::numeric_limits<double>::quiet_NaN();
      double hybrid_objective_final =
          std::numeric_limits<double>::quiet_NaN();
      for (const auto& stage : hybrid_result.stages) {
        hybrid_iterations += stage.iterations;
        hybrid_failed_iterations += stage.failed_iterations;
        if (!std::isfinite(hybrid_objective_start)) {
          hybrid_objective_start = stage.objective_start;
        }
        hybrid_objective_final = stage.objective_final;
      }
      hybrid_summary << "stage5_hybrid_pixel_ray_iterations: "
                     << hybrid_iterations << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_runtime_seconds: "
                     << runtime_seconds << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_objective_start: "
                     << hybrid_objective_start << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_objective_final: "
                     << hybrid_objective_final << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_failed_iterations: "
                     << hybrid_failed_iterations << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_failure_reason: "
                     << failure_reason << "\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_holdout_used_for_scale: 0\n";
      hybrid_summary << "stage5_hybrid_pixel_ray_holdout_used_for_commit: 0\n";
      if (committed) {
        hybrid_summary << "stage5_hybrid_pixel_ray_final_camera_family: "
                       << backend_result.optimized_scene_state.camera
                              .NormalizedFamilyString()
                       << "\n";
        hybrid_summary << "stage5_hybrid_pixel_ray_final_intrinsics_csv: ";
        const std::vector<double> intrinsics =
            backend_result.optimized_scene_state.camera.IntrinsicsVector();
        for (std::size_t index = 0; index < intrinsics.size(); ++index) {
          if (index > 0) {
            hybrid_summary << ",";
          }
          hybrid_summary << intrinsics[index];
        }
        hybrid_summary << "\n";
      }
    }
    WriteExperimentConfigSummary(
        (output_dir / "experiment_config_summary.txt").string(),
        requested_config,
        report,
        &backend_result);

    ati::CameraModelRefitEvaluationResult backend_training_evaluation;
    ati::CameraModelRefitEvaluationResult backend_holdout_evaluation;
    {
      const auto stage_start = std::chrono::steady_clock::now();
      backend_training_evaluation =
          MakeCommittedBackendTrainingEvaluation(backend_result);
      AddRuntimeStage(&runtime_summary, "backend_training_evaluation",
                      ElapsedSeconds(stage_start), false);
    }
    {
      const auto stage_start = std::chrono::steady_clock::now();
      backend_holdout_evaluation = benchmark.EvaluateCameraModel(
          report.holdout_dataset,
          backend_result.optimized_scene_state.camera,
          "backend");
      AddRuntimeStage(&runtime_summary, "backend_holdout_evaluation",
                      ElapsedSeconds(stage_start), false);
    }
    if (!backend_training_evaluation.success || !backend_holdout_evaluation.success) {
      std::ofstream output((output_dir / "backend_vs_frontend_summary.txt").string().c_str());
      output << "backend_evaluation_failed: 1\n";
      output << "training_failure_reason: " << backend_training_evaluation.failure_reason << "\n";
      output << "holdout_failure_reason: " << backend_holdout_evaluation.failure_reason << "\n";
      write_runtime_summary();
      return 1;
    }
    PrintEvaluationProgress(
        "backend training objective", backend_training_evaluation,
        BackendResidualMetricUnit(runner_options.residual_model));
    PrintEvaluationProgress("backend holdout", backend_holdout_evaluation);

    WriteCommittedBackendTrainingSummary(
        (output_dir / "backend_training_summary.txt").string(),
        backend_result);
    WriteEvaluationSummary(
        (output_dir / "backend_holdout_summary.txt").string(),
        backend_holdout_evaluation);
    WriteCommittedBackendTrainingPointsCsv(
        (output_dir / "backend_training_points.csv").string(),
        backend_result.optimized_residual);
    WriteBackendBoardPosesCsv(
        (output_dir / "backend_board_poses.csv").string(),
        backend_result);
    ati::WriteCameraModelRefitPointsCsv(
        (output_dir / "backend_holdout_points.csv").string(),
        std::vector<ati::CameraModelRefitEvaluationResult>{
            backend_holdout_evaluation});
    WriteBackendVsKalibrSummary(
        output_dir,
        report.kalibr_reference,
        backend_training_evaluation,
        report.kalibr_training_evaluation,
        backend_holdout_evaluation,
        report.kalibr_holdout_evaluation);
    WriteCloseEdgeOuterPoseDiagnostics(
        output_dir,
        report.our_holdout_evaluation,
        backend_holdout_evaluation,
        report.kalibr_holdout_evaluation);
    if (args.export_holdout_reprojection_visualizations) {
      const auto viz_stage_start = std::chrono::steady_clock::now();
      ExportHoldoutReprojectionVisualizations(
          output_dir,
          benchmark,
          report,
          backend_holdout_evaluation,
          args.holdout_visualization_top_k);
      AddRuntimeStage(&runtime_summary,
                      "holdout_reprojection_visualization_export",
                      ElapsedSeconds(viz_stage_start), false);
      PrintProgress(
          "holdout reprojection visualizations=" +
          (output_dir / "holdout_reprojection_visualizations").string());
    }
    const bool write_backend_selection_diagnostics =
        requested_config.enable_stage5_selection_diagnostics ||
        requested_config.enable_trial_backend_frame_board_selection ||
        requested_config.selection_mode == "kalibr_style_frame_board" ||
        requested_config.selection_mode == "kalibr_style_batch";
    if (requested_config.internal_regeneration_diagnostics ||
        write_backend_selection_diagnostics) {
      ati::WriteFrameBoardObservationFlowDiagnostics(
          output_dir, report, all_frames_for_lookup,
          &backend_training_evaluation,
          &backend_result.optimized_residual);
    }
    if (requested_config.internal_regeneration_diagnostics &&
        requested_config.use_intermediate_for_full_frontend_regeneration) {
      ati::WriteIntermediateFrontendRegenerationSummary(output_dir, report);
    }
    if (requested_config.enable_angular_residual_diagnostics) {
      const ati::AngularResidualDiagnosticOptions angular_options{
          true, requested_config.angular_residual_bin_edges_deg};
      const ati::AngularResidualDiagnosticsResult angular_diagnostics =
          ati::EvaluateAngularResidualDiagnostics(
              backend_result.optimized_residual, angular_options);
      ati::WriteAngularResidualSummary(
          (output_dir / "angular_residual_summary.txt").string(),
          backend_result,
          angular_diagnostics);
      ati::WriteAngularResidualBinsCsv(
          (output_dir / "angular_residual_bins.csv").string(),
          angular_diagnostics);
      ati::WriteAngularResidualPointSelectionCsv(
          (output_dir / "angular_residual_point_selection.csv").string(),
          backend_result.optimized_residual);
    }
    if (requested_config.enable_hybrid_pixel_ray_final_refinement) {
      // This diagnostic is always emitted for Pixel-Ray final refinement, even
      // when the optional general polar diagnostic is disabled.
      const ati::AngularResidualDiagnosticOptions angular_options{
          true, requested_config.angular_residual_bin_edges_deg};
      const ati::AngularResidualDiagnosticsResult angular_diagnostics =
          ati::EvaluateAngularResidualDiagnostics(
              backend_result.optimized_residual, angular_options);
      WritePixelRayHybridAdaptiveDiagnostics(
          output_dir, backend_result, angular_diagnostics,
          backend_holdout_evaluation);
    }
    if (backend_result.options.multi_board_consistency_weighting &&
        backend_result.options.consistency_dump_weight_summary) {
      ati::WriteConsistencyWeightSummary(
          (output_dir / "phase4_consistency_weight_summary.txt").string(),
          backend_result);
      ati::WriteConsistencyPerBoardSummary(
          (output_dir / "phase4_per_board_weight_summary.txt").string(),
          backend_result);
      ati::WriteConsistencyPerFrameSummary(
          (output_dir / "phase4_per_frame_weight_summary.txt").string(),
          backend_result);
      ati::WriteTopDownweightedObservations(
          (output_dir / "phase4_top_downweighted_observations.txt").string(),
          backend_result);
      WritePhase4BenchmarkSummary(
          output_dir / "phase4_benchmark_summary.txt",
          backend_result,
          backend_training_evaluation,
          backend_holdout_evaluation);
    }
    if (requested_config.enable_multiboard_rigidity_diagnostics) {
      const auto rigidity_diag_start = std::chrono::steady_clock::now();
      ati::MultiBoardRigidityDiagnosticsOptions rigidity_options;
      rigidity_options.enabled = true;
      rigidity_options.top_k = requested_config.multiboard_rigidity_top_k;
      rigidity_options.rotation_bad_threshold_deg =
          requested_config.multiboard_rigidity_rotation_bad_threshold_deg;
      rigidity_options.translation_bad_threshold =
          requested_config.multiboard_rigidity_translation_bad_threshold;
      rigidity_options.reprojection_delta_bad_threshold_px =
          requested_config.multiboard_rigidity_reprojection_delta_bad_threshold_px;
      rigidity_options.use_internal_points =
          requested_config.multiboard_rigidity_use_internal_points;
      rigidity_options.use_outer_points =
          requested_config.multiboard_rigidity_use_outer_points;
      const ati::MultiBoardRigidityDiagnostics rigidity_diagnostics(
          rigidity_options);
      const ati::MultiBoardRigidityDiagnosticsResult rigidity_result =
          rigidity_diagnostics.Evaluate(
              backend_result.effective_problem_input.measurement_dataset,
              backend_result.optimized_scene_state);
      const ati::CalibrationMeasurementDataset round1_all_outer_dataset =
          BuildAllOuterObservationDataset(
              report.baseline_result.round1.measurement_result,
              "round1_all_outer_observations",
              report.dataset_label,
              report.split_signature,
              report.baseline_protocol_label);
      const ati::JointReprojectionSceneState round1_all_outer_scene_state =
          ati::BuildSceneStateFromBootstrap(report.baseline_result.bootstrap_result);
      const ati::MultiBoardRigidityDiagnosticsResult round1_all_outer_result =
          rigidity_diagnostics.Evaluate(round1_all_outer_dataset,
                                        round1_all_outer_scene_state);
      AddRuntimeStage(&runtime_summary, "multiboard_rigidity_diagnostics",
                      ElapsedSeconds(rigidity_diag_start), false);
      if (!rigidity_result.success) {
        std::cerr << "WARNING: Multi-board rigidity diagnostics failed: "
                  << rigidity_result.failure_reason << "\n";
      } else {
        ati::WriteFrameBoardConsistencyCsv(
            (output_dir / "frame_board_consistency.csv").string(),
            rigidity_result);
        ati::WriteFrameBoardConsistencySummary(
            (output_dir / "frame_board_consistency_summary.txt").string(),
            rigidity_result);
        ati::WriteTopBadFrameBoardObservations(
            (output_dir / "top_bad_frame_board_observations.txt").string(),
            rigidity_result,
            rigidity_options.top_k);
        ati::WriteBoardPairwiseConsistencyCsv(
            (output_dir / "board_pairwise_consistency.csv").string(),
            rigidity_result);
        ati::WriteBoardPairwiseConsistencySummary(
            (output_dir / "board_pairwise_consistency_summary.txt").string(),
            rigidity_result);
        ati::WriteTopBadBoardPairs(
            (output_dir / "top_bad_board_pairs.txt").string(),
            rigidity_result,
            rigidity_options.top_k);
      }
      if (!round1_all_outer_result.success) {
        std::cerr << "WARNING: Round1 all-outer rigidity diagnostics failed: "
                  << round1_all_outer_result.failure_reason << "\n";
      } else {
        ati::WriteFrameBoardConsistencyCsv(
            (output_dir / "round1_all_outer_frame_board_consistency.csv")
                .string(),
            round1_all_outer_result);
        ati::WriteFrameBoardConsistencySummary(
            (output_dir / "round1_all_outer_frame_board_consistency_summary.txt")
                .string(),
            round1_all_outer_result);
        ati::WriteTopBadFrameBoardObservations(
            (output_dir / "round1_all_outer_top_bad_frame_board_observations.txt")
                .string(),
            round1_all_outer_result,
            rigidity_options.top_k);
        ati::WriteBoardPairwiseConsistencyCsv(
            (output_dir / "round1_all_outer_board_pairwise_consistency.csv")
                .string(),
            round1_all_outer_result);
        ati::WriteBoardPairwiseConsistencySummary(
            (output_dir /
             "round1_all_outer_board_pairwise_consistency_summary.txt")
                .string(),
            round1_all_outer_result);
        ati::WriteTopBadBoardPairs(
            (output_dir / "round1_all_outer_top_bad_board_pairs.txt").string(),
            round1_all_outer_result,
            rigidity_options.top_k);
      }
    }
    if (requested_config.internal_blur_diagnostics) {
      WriteInternalBlurDiagnostics(
          output_dir, report, backend_training_evaluation,
          backend_holdout_evaluation, all_frames_for_lookup);
    }

    if (args.enable_polar_angle_diagnostics) {
      const auto polar_diag_start = std::chrono::steady_clock::now();
      ati::PolarAngleDiagnosticsOptions polar_options;
      polar_options.enabled = true;
      polar_options.bin_edges_deg = ParsePolarAngleBinEdges(args.polar_angle_bin_edges);
      ati::PolarAngleResidualDiagnostics polar_diagnostics(polar_options);

      const ati::PolarAngleDiagnosticsResult polar_result =
          polar_diagnostics.EvaluateWithResiduals(
              report.backend_problem_input.measurement_dataset,
              backend_result.optimized_residual,
              backend_result.optimized_scene_state,
              output_dir.string());

      AddRuntimeStage(&runtime_summary, "polar_angle_diagnostics",
                      ElapsedSeconds(polar_diag_start), false);

      if (!polar_result.success) {
        std::cerr << "WARNING: Polar angle diagnostics failed: "
                  << polar_result.failure_reason << "\n";
      }
    }

    if (requested_config.enable_multi_board_consistency_diagnostics &&
        report.multi_board_consistency_diagnostics.success) {
      ati::WriteMultiBoardConsistencySummary(
          (output_dir / "multi_board_consistency_summary.txt").string(),
          report.multi_board_consistency_diagnostics);
      ati::WriteMultiBoardConsistencyPerObservationCsv(
          (output_dir / "multi_board_consistency_per_observation.csv").string(),
          report.multi_board_consistency_diagnostics);
      ati::WriteMultiBoardConsistencyPerBoardCsv(
          (output_dir / "multi_board_consistency_per_board.csv").string(),
          report.multi_board_consistency_diagnostics);
      ati::WriteMultiBoardConsistencyPerFrameCsv(
          (output_dir / "multi_board_consistency_per_frame.csv").string(),
          report.multi_board_consistency_diagnostics);
    }

    WriteBackendComparisonSummary(
        (output_dir / "backend_vs_frontend_summary.txt").string(),
        backend_result,
        report.our_training_evaluation,
        backend_training_evaluation,
        report.kalibr_training_evaluation,
        report.our_holdout_evaluation,
        backend_holdout_evaluation,
        report.kalibr_holdout_evaluation);
    if (!use_precomputed) {
      const auto worst_overlay_stage_start = std::chrono::steady_clock::now();
      ExportWorstReprojectionOverlays(
          output_dir,
          benchmark,
          report,
          backend_holdout_evaluation);
      if (args.export_selected_case_visualizations) {
        ExportSelectedCaseVisualizations(
            output_dir,
            benchmark,
            report,
            backend_holdout_evaluation);
      }
      AddRuntimeStage(&runtime_summary, "worst_reprojection_overlay_export",
                      ElapsedSeconds(worst_overlay_stage_start), false);
    }

    AddRuntimeStage(&runtime_summary, "overlay_export", 0.0, true);

    write_runtime_summary();
    std::cout << "Stage 5 benchmark summary: "
              << (output_dir / "benchmark_protocol_summary.txt").string() << "\n"
              << "Backend optimization summary: "
              << (output_dir / "backend_optimization_summary.txt").string() << "\n"
              << "Backend training summary: "
              << (output_dir / "backend_training_summary.txt").string() << "\n"
              << "Backend holdout summary: "
              << (output_dir / "backend_holdout_summary.txt").string() << "\n"
              << "Backend vs frontend summary: "
              << (output_dir / "backend_vs_frontend_summary.txt").string() << "\n"
              << "Runtime summary: "
              << (output_dir / "runtime_summary.txt").string() << "\n";
    if (args.export_selected_case_visualizations) {
      std::cout << "Selected case visualizations: "
                << (output_dir / "selected_case_visualizations").string()
                << "\n";
    } else {
      std::cout << "Selected case visualizations: skipped\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
