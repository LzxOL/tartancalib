#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_ONLY_CAMERA_INITIALIZER_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_ONLY_CAMERA_INITIALIZER_HPP

#include <array>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

enum class AutoCameraInitializationRefineMode {
  None,
  CoordinateSearch,
  KalibrOuterLm,
};

enum class AutoCameraInitializationSelectionScorer {
  PoseMarginalizedPrincipal,
  LegacyFixedPose,
};

const char* ToString(AutoCameraInitializationRefineMode mode);
AutoCameraInitializationRefineMode ParseAutoCameraInitializationRefineMode(
    const std::string& value);
const char* ToString(AutoCameraInitializationSelectionScorer scorer);
AutoCameraInitializationSelectionScorer
ParseAutoCameraInitializationSelectionScorer(const std::string& value);

struct AutoCameraInitializationCandidate {
  int rank = -1;
  std::string source_label = "grid";
  std::string evaluation_scope = "sampled";
  OuterBootstrapCameraIntrinsics camera;
  int observation_count = 0;
  int pose_success_count = 0;
  int pose_failure_count = 0;
  int successful_frame_count = 0;
  int successful_board_count = 0;
  double success_rate = 0.0;
  double mean_observation_rmse = std::numeric_limits<double>::infinity();
  int leave_one_board_out_attempt_count = 0;
  int leave_one_board_out_success_count = 0;
  double leave_one_board_out_rmse = std::numeric_limits<double>::infinity();
  int relative_layout_pair_family_count = 0;
  int relative_layout_pair_sample_count = 0;
  double relative_layout_translation_rmse = std::numeric_limits<double>::infinity();
  double relative_layout_rotation_rmse_deg = std::numeric_limits<double>::infinity();
  double relative_layout_consistency_score = std::numeric_limits<double>::infinity();
  bool valid = false;
  std::string failure_reason;
};

struct AutoCameraInitializationRefinedBasinCandidate {
  int trial_index = -1;
  int seed_rank = -1;
  std::string seed_source_label;
  OuterBootstrapCameraIntrinsics seed_camera;
  OuterBootstrapCameraIntrinsics refined_camera;
  int selected_frame_count = 0;
  int selected_board_observation_count = 0;
  int residual_count = 0;
  int iteration_count = 0;
  double seed_objective = std::numeric_limits<double>::infinity();
  double full_outer_objective = std::numeric_limits<double>::infinity();
  double lm_initial_rmse = std::numeric_limits<double>::infinity();
  double lm_final_rmse = std::numeric_limits<double>::infinity();
  double lm_final_robust_rmse = std::numeric_limits<double>::infinity();
  // Diagnostics for the shared frame/board geometry pass.  The independent
  // board-pose LM remains available as the seed, but this pass is the
  // selection objective when it succeeds.
  int shared_layout_constraint_used = 0;
  int shared_layout_frame_count = 0;
  int shared_layout_board_count = 0;
  int shared_layout_observation_count = 0;
  double shared_layout_initial_rmse = std::numeric_limits<double>::infinity();
  double shared_layout_final_rmse = std::numeric_limits<double>::infinity();
  double shared_layout_final_robust_rmse =
      std::numeric_limits<double>::infinity();
  double combined_selection_objective =
      std::numeric_limits<double>::infinity();
  bool full_outer_health_acceptable = false;
  bool camera_step_finite = false;
  bool objective_improved_before_near_tie_policy = false;
  bool compared_as_near_tie = false;
  bool preferred_by_lower_focal_near_tie_policy = false;
  bool accepted_as_running_best = false;
  bool selected = false;
  int ray_comparison_sample_count = 0;
  double ray_rms_deg_to_selected =
      std::numeric_limits<double>::infinity();
  bool distinct_ray_basin_from_selected = false;
  std::string decision_reason;
};

struct AutoCameraInitializationResidual {
  std::string source_label;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  double quality = 0.0;
  bool used_local_patch_rescue = false;
  bool pose_success = false;
  double pose_fit_outer_rmse = std::numeric_limits<double>::quiet_NaN();
  std::string failure_reason;
};

struct AutoCameraInitializationBootstrapObservation {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  std::array<Eigen::Vector2d, 4> outer_corners{};
  bool used_in_lm = false;
  bool used_local_patch_rescue = false;
  bool pose_init_success = false;
  double pose_fit_outer_rmse = std::numeric_limits<double>::quiet_NaN();
};

struct AutoCameraInitializationPrincipalProfileSample {
  double delta_cu_px = 0.0;
  double delta_cv_px = 0.0;
  double fixed_cu = std::numeric_limits<double>::quiet_NaN();
  double fixed_cv = std::numeric_limits<double>::quiet_NaN();
  OuterBootstrapCameraIntrinsics optimized_camera;
  int expected_view_count = 0;
  int optimized_view_count = 0;
  int residual_count = 0;
  int iteration_count = 0;
  double final_rmse = std::numeric_limits<double>::infinity();
  double final_robust_rmse = std::numeric_limits<double>::infinity();
  double final_robust_cost = std::numeric_limits<double>::infinity();
  double delta_robust_cost = std::numeric_limits<double>::infinity();
  bool comparable = false;
};

struct AutoCameraInitializationBoardJackknifeSample {
  int excluded_board_id = -1;
  int expected_view_count = 0;
  int optimized_view_count = 0;
  int residual_count = 0;
  int iteration_count = 0;
  double final_rmse = std::numeric_limits<double>::infinity();
  double delta_xi = std::numeric_limits<double>::quiet_NaN();
  double delta_alpha = std::numeric_limits<double>::quiet_NaN();
  double delta_fu = std::numeric_limits<double>::quiet_NaN();
  double delta_fv = std::numeric_limits<double>::quiet_NaN();
  double delta_cu = std::numeric_limits<double>::quiet_NaN();
  double delta_cv = std::numeric_limits<double>::quiet_NaN();
  OuterBootstrapCameraIntrinsics optimized_camera;
  bool comparable = false;
};

struct AutoCameraInitializationCoverageWeightRecord {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int grid_x = -1;
  int grid_y = -1;
  double centroid_x = std::numeric_limits<double>::quiet_NaN();
  double centroid_y = std::numeric_limits<double>::quiet_NaN();
  double weight = 1.0;
};

struct AutoCameraInitializationPoseExcitationRecord {
  int board_id = -1;
  int observation_count = 0;
  int pose_success_count = 0;
  double normal_spread_median_deg =
      std::numeric_limits<double>::quiet_NaN();
  double normal_spread_p95_deg =
      std::numeric_limits<double>::quiet_NaN();
  double normal_spread_max_deg =
      std::numeric_limits<double>::quiet_NaN();
  double normal_xy_std_x = std::numeric_limits<double>::quiet_NaN();
  double normal_xy_std_y = std::numeric_limits<double>::quiet_NaN();
  double normal_xy_weak_variance =
      std::numeric_limits<double>::quiet_NaN();
  double normal_xy_strong_variance =
      std::numeric_limits<double>::quiet_NaN();
  double normal_xy_axis_balance_ratio =
      std::numeric_limits<double>::quiet_NaN();
  double normal_xy_dominant_axis_angle_deg =
      std::numeric_limits<double>::quiet_NaN();
  double tilt_min_deg = std::numeric_limits<double>::quiet_NaN();
  double tilt_max_deg = std::numeric_limits<double>::quiet_NaN();
  double tilt_range_deg = std::numeric_limits<double>::quiet_NaN();
  double centroid_min_x = std::numeric_limits<double>::quiet_NaN();
  double centroid_max_x = std::numeric_limits<double>::quiet_NaN();
  double centroid_min_y = std::numeric_limits<double>::quiet_NaN();
  double centroid_max_y = std::numeric_limits<double>::quiet_NaN();
  double centroid_span_x = std::numeric_limits<double>::quiet_NaN();
  double centroid_span_y = std::numeric_limits<double>::quiet_NaN();
};

struct AutoCameraInitializationPoseExcitationSample {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  double pose_rmse = std::numeric_limits<double>::quiet_NaN();
  double normal_x = std::numeric_limits<double>::quiet_NaN();
  double normal_y = std::numeric_limits<double>::quiet_NaN();
  double normal_z = std::numeric_limits<double>::quiet_NaN();
  double normal_deviation_from_board_mean_deg =
      std::numeric_limits<double>::quiet_NaN();
  double tilt_deg = std::numeric_limits<double>::quiet_NaN();
  double centroid_x = std::numeric_limits<double>::quiet_NaN();
  double centroid_y = std::numeric_limits<double>::quiet_NaN();
};

struct AutoCameraInitializationResult {
  bool success = false;
  CameraInitializationMode requested_mode =
      CameraInitializationMode::AutoWithManualFallback;
  CameraInitializationMode selected_mode = CameraInitializationMode::Manual;
  bool auto_attempted = false;
  bool fallback_used = false;
  bool used_manual_intermediate_camera = false;
  bool used_explicit_initial_camera = false;
  bool used_manual_generic_seed = false;
  bool selected_candidate_refined = false;
  AutoCameraInitializationRefineMode refine_mode =
      AutoCameraInitializationRefineMode::KalibrOuterLm;
  int lm_frame_count = 0;
  int lm_view_count = 0;
  int lm_residual_count = 0;
  int lm_invalid_projection_count = 0;
  int lm_nonfinite_count = 0;
  int lm_iteration_count = 0;
  double lm_initial_rmse = std::numeric_limits<double>::infinity();
  double lm_final_rmse = std::numeric_limits<double>::infinity();
  int lm_robust_loss_enabled = 0;
  std::string lm_robust_loss_type = "none";
  double lm_robust_loss_delta_pixels = 0.0;
  double lm_initial_robust_rmse = std::numeric_limits<double>::infinity();
  double lm_final_robust_rmse = std::numeric_limits<double>::infinity();
  int lm_initial_downweighted_point_count = 0;
  int lm_final_downweighted_point_count = 0;
  int stage5_init_pose_fit_outlier_gate_enabled = 1;
  int stage5_init_pose_fit_outlier_gate_applied = 0;
  int stage5_init_pose_fit_outlier_rejected_count = 0;
  double stage5_init_pose_fit_outlier_median_rmse =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_fit_outlier_mad_rmse =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_fit_outlier_threshold_rmse =
      std::numeric_limits<double>::quiet_NaN();
  int stage5_init_requires_all_configured_boards_per_frame = 1;
  std::string stage5_init_required_board_ids;
  int stage5_init_required_board_count = 0;
  int stage5_init_input_frame_count = 0;
  int stage5_init_complete_board_frame_count = 0;
  int stage5_init_incomplete_board_frame_rejected_count = 0;
  int stage5_init_observation_count_before_complete_frame_filter = 0;
  int stage5_init_observation_count_after_complete_frame_filter = 0;
  // The complete-frame counts above are retained for compatibility and
  // diagnostics.  Camera initialization evaluation now uses all valid outer
  // observations, including observations from incomplete frames.
  int stage5_init_camera_evaluation_observation_count = 0;
  int stage5_init_all_valid_outer_observations_used = 0;
  int stage5_init_rescued_outer_observation_count = 0;
  int stage5_init_rescued_outer_observation_pose_gate_rejected_count = 0;
  double stage5_init_rescued_outer_observation_lm_weight = 0.25;
  std::string stage5_init_seed_method;
  std::string stage5_init_seed_source;
  double stage5_init_omni_gamma = std::numeric_limits<double>::quiet_NaN();
  std::string stage5_init_omni_gamma_source;
  std::string stage5_init_ds_mapping;
  int stage5_init_ds_mapping_verified_against_kalibr_source = 0;
  int stage5_init_ds_grid_enumeration_enabled = 0;
  int stage5_init_near_tie_lower_focal_policy_enabled = 0;
  double stage5_init_near_tie_relative_objective_tolerance = 0.0;
  int stage5_init_refined_basin_candidate_count = 0;
  int stage5_init_refined_basin_valid_count = 0;
  int stage5_init_refined_basin_near_tie_count = 0;
  int stage5_init_selected_basin_seed_rank = -1;
  double stage5_init_selected_basin_objective =
      std::numeric_limits<double>::infinity();
  std::string stage5_init_selected_basin_reason = "unavailable";
  double stage5_init_basin_distinct_ray_rms_threshold_deg = 0.1;
  double stage5_init_basin_ambiguity_relative_objective_threshold = 0.05;
  int stage5_init_distinct_refined_basin_candidate_count = 0;
  int stage5_init_distinct_basin_runner_up_seed_rank = -1;
  std::string stage5_init_distinct_basin_alternate_selection =
      "unavailable";
  double stage5_init_distinct_basin_runner_up_objective =
      std::numeric_limits<double>::infinity();
  double stage5_init_distinct_basin_relative_objective_gap =
      std::numeric_limits<double>::infinity();
  double stage5_init_distinct_basin_ray_rms_deg =
      std::numeric_limits<double>::infinity();
  int stage5_init_distinct_basin_ambiguity_detected = 0;
  std::string stage5_init_kb_focal_source;
  int stage5_init_kb_row_circle_focal_available = 0;
  int stage5_init_kb_zero_distortion_seed = 0;
  int stage5_init_kb_zero_distortion_seed_included = 0;
  int stage5_init_kb_nonzero_distortion_seed_count = 0;
  int stage5_init_kb_distortion_released_in_lm = 0;
  int stage5_init_kb_distortion_fixed_zero_in_init_lm = 0;
  int stage5_init_kb_multistart_enabled = 0;
  std::string stage5_init_ucm_seed_source;
  int stage5_init_ucm_omni_gamma_available = 0;
  std::string stage5_init_ucm_mapping;
  int stage5_init_ucm_mapping_verified_against_kalibr_source = 0;
  int stage5_init_ucm_multistart_enabled = 0;
  int stage5_init_ucm_shape_released_in_lm = 0;
  int stage5_init_uses_yaml_intrinsics = 0;
  int stage5_init_uses_kalibr_camchain_intrinsics = 0;
  int stage5_init_outer_only = 1;
  int stage5_init_uses_layout_to_update_intrinsics = 0;
  int stage5_init_layout_loo_diagnostics_only = 1;
  int stage5_init_multiboard_frame_objective_enabled = 0;
  int stage5_init_fixed_layout_frame_constraint_used = 0;
  int stage5_init_optimizes_layout_variables = 0;
  int stage5_init_shared_frame_board_constraint_enabled = 0;
  int stage5_init_shared_frame_board_constraint_used = 0;
  int stage5_init_shared_layout_board_count = 0;
  int stage5_init_shared_layout_frame_count = 0;
  int stage5_init_shared_layout_observation_count = 0;
  double stage5_init_shared_layout_initial_rmse =
      std::numeric_limits<double>::infinity();
  double stage5_init_shared_layout_final_rmse =
      std::numeric_limits<double>::infinity();
  std::string stage5_init_lm_selection_objective;
  double stage5_init_lm_min_relative_objective_improvement = 0.0;
  std::string stage5_init_selection_prefilter;
  std::string stage5_init_selection_scorer;
  int stage5_init_selection_uses_information_metric = 0;
  int stage5_init_selection_is_exact_kalibr_information_theoretic = 0;
  int stage5_init_selection_pose_marginalized = 0;
  int stage5_init_selection_principal_subspace_aware = 0;
  int stage5_init_selection_camera_information_dimension = -1;
  int stage5_init_selection_camera_information_rank = -1;
  int stage5_init_selection_principal_information_rank = -1;
  int stage5_init_selection_pose_rank_min = -1;
  int stage5_init_selection_pose_rank_max = -1;
  int stage5_init_selection_pose_rank_deficient_count = 0;
  double stage5_init_selection_principal_min_eigenvalue = -1.0;
  double stage5_init_selection_principal_max_eigenvalue = -1.0;
  double stage5_init_selection_cu_stddev_px = -1.0;
  double stage5_init_selection_cv_stddev_px = -1.0;
  double stage5_init_selection_weakest_eigenvalue = -1.0;
  std::string stage5_init_selection_weakest_direction;
  double stage5_init_selection_weakest_principal_fraction = -1.0;
  double stage5_init_selection_weakest_focal_fraction = -1.0;
  std::string stage5_init_selection_information_linearization;
  int stage5_init_selection_all_pose_valid_observations_used = 0;
  int stage5_init_calibrate_intrinsics_enabled = 0;
  std::string stage5_init_calibrate_intrinsics_released_params;
  std::string stage5_init_calibrate_intrinsics_optimizer;
  double stage5_init_runtime_seconds = 0.0;
  int stage5_init_selected_pose_success_count = 0;
  int stage5_init_selected_pose_total_count = 0;
  double full_outer_pose_success_rate = 0.0;
  double full_outer_rmse = std::numeric_limits<double>::infinity();
  double full_outer_median_error = std::numeric_limits<double>::infinity();
  double full_outer_p95_error = std::numeric_limits<double>::infinity();
  double full_outer_robust_inlier_rmse = std::numeric_limits<double>::infinity();
  double full_outer_robust_outlier_threshold =
      std::numeric_limits<double>::infinity();
  int full_outer_robust_outlier_count = 0;
  int full_outer_projection_failure_count = 0;
  int full_outer_nonfinite_count = 0;
  int bootstrap_internal_points_used = 0;
  int stage5_init_dense_control_points_enabled = 0;
  std::string stage5_init_dense_control_points_scope = "disabled";
  int stage5_init_dense_control_point_count = 0;
  int stage5_init_primary_frame_count = 0;
  int stage5_init_auxiliary_session_count = 0;
  int stage5_init_auxiliary_frame_count = 0;
  int stage5_init_uses_auxiliary_sessions = 0;
  std::string selected_source_label;
  OuterBootstrapCameraIntrinsics selected_camera;
  cv::Size image_size;
  int candidate_count = 0;
  int sampled_observation_count = 0;
  int total_valid_outer_observation_count = 0;
  int accepted_pose_fit_observation_count = 0;
  int failed_pose_fit_observation_count = 0;
  int accepted_frame_count = 0;
  int accepted_board_observation_count = 0;
  double initialization_rmse = std::numeric_limits<double>::infinity();
  std::vector<AutoCameraInitializationCandidate> candidates;
  std::vector<AutoCameraInitializationRefinedBasinCandidate>
      refined_basin_candidates;
  std::vector<AutoCameraInitializationResidual> selected_residuals;
  std::vector<AutoCameraInitializationBootstrapObservation>
      lm_bootstrap_observations;
  int stage5_init_principal_profile_enabled = 0;
  double stage5_init_principal_profile_radius_px = 0.0;
  int stage5_init_principal_profile_observation_count = 0;
  int stage5_init_principal_profile_sample_count = 0;
  int stage5_init_principal_profile_comparable_sample_count = 0;
  double stage5_init_principal_profile_best_delta_cu_px =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_principal_profile_best_delta_cv_px =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_principal_profile_best_delta_robust_cost =
      std::numeric_limits<double>::quiet_NaN();
  std::vector<AutoCameraInitializationPrincipalProfileSample>
      principal_profile_samples;
  int stage5_init_fixed_layout_diagnostic_enabled = 0;
  int stage5_init_fixed_layout_diagnostic_updates_selected_intrinsics = 0;
  int stage5_init_fixed_layout_diagnostic_layout_success = 0;
  int stage5_init_fixed_layout_diagnostic_board_count = 0;
  int stage5_init_fixed_layout_diagnostic_frame_count = 0;
  int stage5_init_fixed_layout_diagnostic_board_observation_count = 0;
  int stage5_init_fixed_layout_diagnostic_iteration_count = 0;
  double stage5_init_fixed_layout_diagnostic_layout_bootstrap_rmse =
      std::numeric_limits<double>::infinity();
  double stage5_init_fixed_layout_diagnostic_initial_rmse =
      std::numeric_limits<double>::infinity();
  double stage5_init_fixed_layout_diagnostic_final_rmse =
      std::numeric_limits<double>::infinity();
  double stage5_init_fixed_layout_diagnostic_rig_axis_balance_ratio =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_fixed_layout_diagnostic_rig_dominant_axis_angle_deg =
      std::numeric_limits<double>::quiet_NaN();
  int stage5_init_fixed_layout_principal_profile_enabled = 0;
  double stage5_init_fixed_layout_principal_profile_radius_px = 0.0;
  int stage5_init_fixed_layout_principal_profile_sample_count = 0;
  int stage5_init_fixed_layout_principal_profile_comparable_sample_count = 0;
  double stage5_init_fixed_layout_principal_profile_best_delta_cu_px =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_fixed_layout_principal_profile_best_delta_cv_px =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_fixed_layout_principal_profile_best_delta_robust_cost =
      std::numeric_limits<double>::quiet_NaN();
  std::vector<AutoCameraInitializationPrincipalProfileSample>
      fixed_layout_principal_profile_samples;
  std::string stage5_init_fixed_layout_diagnostic_layout_source = "disabled";
  OuterBootstrapCameraIntrinsics stage5_init_fixed_layout_diagnostic_camera;
  int stage5_init_board_jackknife_diagnostic_enabled = 0;
  int stage5_init_board_jackknife_diagnostic_updates_selected_intrinsics = 0;
  int stage5_init_board_jackknife_diagnostic_sample_count = 0;
  int stage5_init_board_jackknife_diagnostic_comparable_sample_count = 0;
  std::vector<AutoCameraInitializationBoardJackknifeSample>
      board_jackknife_samples;
  int stage5_init_coverage_weighted_diagnostic_enabled = 0;
  int stage5_init_coverage_weighted_diagnostic_updates_selected_intrinsics = 0;
  int stage5_init_coverage_weighted_diagnostic_grid_rows = 0;
  int stage5_init_coverage_weighted_diagnostic_grid_cols = 0;
  int stage5_init_coverage_weighted_diagnostic_occupied_bin_count = 0;
  double stage5_init_coverage_weighted_diagnostic_min_weight = 1.0;
  double stage5_init_coverage_weighted_diagnostic_max_weight = 1.0;
  double stage5_init_coverage_weighted_diagnostic_initial_rmse =
      std::numeric_limits<double>::infinity();
  double stage5_init_coverage_weighted_diagnostic_final_rmse =
      std::numeric_limits<double>::infinity();
  OuterBootstrapCameraIntrinsics stage5_init_coverage_weighted_diagnostic_camera;
  std::vector<AutoCameraInitializationCoverageWeightRecord>
      coverage_weight_records;
  int stage5_init_pose_excitation_diagnostic_enabled = 0;
  int stage5_init_pose_excitation_diagnostic_updates_selected_intrinsics = 0;
  int stage5_init_pose_excitation_board_count = 0;
  int stage5_init_pose_excitation_pose_success_count = 0;
  int stage5_init_pose_excitation_pose_total_count = 0;
  double stage5_init_pose_excitation_min_board_normal_p95_deg =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_median_board_normal_p95_deg =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_max_board_normal_p95_deg =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_min_board_tilt_range_deg =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_median_board_tilt_range_deg =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_min_normal_xy_axis_balance_ratio =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_median_normal_xy_axis_balance_ratio =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_max_normal_xy_axis_balance_ratio =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_global_normal_xy_std_x =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_global_normal_xy_std_y =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_global_normal_xy_weak_variance =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_global_normal_xy_strong_variance =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_global_normal_xy_axis_balance_ratio =
      std::numeric_limits<double>::quiet_NaN();
  double stage5_init_pose_excitation_global_normal_xy_dominant_axis_angle_deg =
      std::numeric_limits<double>::quiet_NaN();
  int stage5_init_pose_excitation_single_axis_board_count = 0;
  int stage5_init_pose_excitation_principal_pseudo_observability_warning = 0;
  std::string stage5_init_pose_excitation_assessment = "unavailable";
  std::vector<AutoCameraInitializationPoseExcitationRecord>
      pose_excitation_records;
  std::vector<AutoCameraInitializationPoseExcitationSample>
      pose_excitation_samples;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct AutoCameraInitializationOptions {
  CameraInitializationMode mode = CameraInitializationMode::AutoWithManualFallback;
  AutoCameraInitializationRefineMode refine_mode =
      AutoCameraInitializationRefineMode::KalibrOuterLm;
  AutoCameraInitializationSelectionScorer selection_scorer =
      AutoCameraInitializationSelectionScorer::PoseMarginalizedPrincipal;
  bool use_explicit_initial_camera = false;
  OuterBootstrapCameraIntrinsics explicit_initial_camera;
  std::string explicit_initial_camera_source_label = "explicit_initial_camera";
  int max_candidate_observations = 80;
  int top_candidate_count = 10;
  bool refine_best_candidate = true;
  // Only frames containing valid outer corners for every configured board may
  // contribute to automatic camera initialization.
  bool require_all_configured_boards_per_frame = true;
  // Complete-frame filtering is kept for seed diagnostics, but valid outer
  // observations from incomplete frames are included in camera evaluation.
  bool include_all_valid_outer_observations_in_evaluation = true;
  // Camera-aware patch-rescue corners are useful evidence but are not as
  // trustworthy as ordinary subpixel-refined corners.  Gate them by their
  // own pose-fit threshold and downweight them in independent-pose LM.
  bool gate_rescued_outer_observations = true;
  double rescued_outer_observation_pose_rmse_gate_pixels = 8.0;
  double rescued_outer_observation_lm_weight = 0.25;
  // Estimate one common board layout and one pose per frame after the
  // independent-pose seed.  This is the actual selection pass; the older
  // fixed-layout diagnostic remains separately controllable.
  bool enable_shared_frame_board_constraint = true;
  // Use every directly imported control point in each observation while still
  // assigning an independent target pose to every frame-board pair.
  bool use_direct_dense_control_points = false;
  // Diagnostic scope for imported frozen control points. Supported values:
  // all, outer_only, internal_only.
  std::string direct_dense_control_point_scope = "all";
  // Diagnostic only. Fixed-cu/cv profile samples never update the selected
  // initialization camera.
  bool enable_principal_profile = false;
  double principal_profile_radius_px = 10.0;
  bool enable_fixed_layout_diagnostic = false;
  bool enable_board_jackknife_diagnostic = false;
  bool enable_coverage_weighted_diagnostic = false;
  bool prefer_lower_focal_in_near_tie = false;
  double near_tie_relative_objective_tolerance = 0.0;
  // Prevent numerically indistinguishable dense-grid DS minima from replacing
  // an earlier, better-supported multi-start basin.
  double dense_grid_lm_min_relative_objective_improvement = 1e-3;
  // Dense checkerboard/control-grid initialization uses point-norm Huber IRLS.
  // A non-positive value explicitly disables robust initialization.
  double dense_grid_lm_huber_delta_pixels = 1.5;
  std::vector<double> focal_scale_candidates{
      0.18, 0.22, 0.26, 0.30, 0.34, 0.40, 0.50, 0.60};
  std::vector<double> xi_candidates{-0.4, -0.2, 0.0, 0.2, 0.5, 1.0};
  std::vector<double> alpha_candidates{0.35, 0.45, 0.55, 0.65, 0.75};
  std::vector<double> eucm_alpha_candidates{0.35, 0.45, 0.55, 0.65, 0.75};
  std::vector<double> eucm_beta_candidates{0.6, 0.8, 1.0, 1.2, 1.5};
  std::vector<double> equidistant_k1_candidates{-0.15, 0.0, 0.15};
};

class OuterOnlyCameraInitializer {
 public:
  explicit OuterOnlyCameraInitializer(
      ApriltagInternalConfig config,
      AutoCameraInitializationOptions options = AutoCameraInitializationOptions{});

  AutoCameraInitializationResult Initialize(
      const std::vector<OuterBootstrapFrameInput>& frames) const;

 private:
  ApriltagInternalConfig config_;
  AutoCameraInitializationOptions options_;
};

void WriteAutoCameraInitializationSummary(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationCandidatesCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationRefinedBasinsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationOuterResidualsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationBootstrapViewsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationPrincipalProfileCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationFixedLayoutPrincipalProfileCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationBoardJackknifeCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationCoverageWeightsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationPoseExcitationCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationPoseExcitationSamplesCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_ONLY_CAMERA_INITIALIZER_HPP
