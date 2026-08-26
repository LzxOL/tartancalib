#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_INCREMENTAL_BACKEND_ESTIMATOR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_INCREMENTAL_BACKEND_ESTIMATOR_HPP

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct TrialBackendFrameBoardSelectionOptions;

struct Stage5IncrementalBackendEstimatorOptions {
  bool enabled = true;
  bool single_board_dense_grid_profile = false;
  double information_gain_threshold = 0.2;
  double rank_gain_threshold = 1e-6;
  int max_iterations = 5;
  double convergence_delta_j = 1e-3;
  double convergence_delta_x = 1e-4;
  int max_continuation_rounds = 3;
  bool verbose = false;
  bool check_validity = false;
  bool use_huber_loss = true;
  ResidualModel residual_model = ResidualModel::ImagePlane;
  double huber_delta_pixels = 8.0;
  double outer_huber_delta_radians = 0.02;
  double internal_huber_delta_radians = 0.015;
  double invalid_projection_penalty_pixels = 100.0;
  double invalid_projection_penalty_radians = 0.35;
  double hybrid_angular_threshold_deg = 50.0;
  double polar_continuous_hybrid_threshold_deg = 50.0;
  double polar_continuous_hybrid_temperature_deg = 10.0;
  double pixel_residual_weight = 1.0;
  double chordal_residual_weight = 1.0;
  bool angular_use_normalize_jacobian = false;
  bool angular_local_whitening_enabled = false;
  double angular_local_whitening_pixel_sigma_px = 1.0;
  double angular_local_whitening_covariance_damping = 1e-12;
  double angular_local_whitening_min_sigma_rad = 1e-6;
  double angular_local_whitening_max_weight = 1e5;
  AslamBackendCalibrationOptions::AngularObservedRayMode angular_observed_ray_mode =
      AslamBackendCalibrationOptions::AngularObservedRayMode::DynamicCurrentCamera;
  bool optimize_seed_intrinsics = false;
  bool independent_frame_board_camera_warmup = false;
  int independent_frame_board_camera_warmup_max_iterations = 200;
  bool optimize_candidate_intrinsics = true;
  bool fix_board_layout = false;
  bool use_candidate_intrinsics_anchor_prior = false;
  bool normalize_information_gain_by_board_observation = false;
  bool use_split_residual_health_gate = true;
  // A newly admitted frame has a different observation distribution from the
  // committed seed.  Comparing its standalone RMSE to the seed RMSE rejects
  // valid hard views solely because they are harder.  Model-aware selection
  // therefore uses the absolute candidate health checks plus committed-scene
  // regression checks instead.
  bool use_candidate_relative_residual_gate = true;
  bool use_bearing_pixel_safety_gate = true;
  bool use_full_training_pose_refit_health_gate = true;
  bool use_kb_distortion_guard = true;
  double candidate_intrinsics_anchor_weight_xi_alpha = 0.0;
  double candidate_intrinsics_anchor_weight_focal = 0.0;
  double candidate_intrinsics_anchor_weight_principal = 0.0;
  double max_candidate_focal_relative_step = 0.0;
  double max_candidate_principal_step_px = 0.0;
  double max_candidate_xi_alpha_step = 0.0;
  double split_residual_max_rmse_regression_ratio = 1.35;
  double split_residual_max_outer_rmse_regression_abs_px = 0.35;
  double split_residual_max_internal_rmse_regression_abs_px = 0.50;
  double split_residual_p95_threshold_scale = 1.5;
  double split_residual_internal_outer_rmse_ratio = 4.0;
  double full_training_pose_refit_max_rmse_regression_ratio = 1.005;
  double full_training_pose_refit_max_rmse_regression_abs_px = 0.002;
  double full_training_pose_refit_max_p95_regression_ratio = 1.01;
  double full_training_pose_refit_max_p95_regression_abs_px = 0.005;
  double full_training_pose_refit_max_pose_success_rate_drop = 0.01;
  bool full_training_instability_quarantine_enabled = true;
  double full_training_instability_quarantine_mad_scale = 8.0;
  double full_training_instability_quarantine_min_regression_px = 1.0;
  double full_training_instability_quarantine_min_regression_ratio = 2.0;
  double full_training_instability_quarantine_max_fraction = 0.02;
  bool adaptive_saturation_stop_enabled = true;
  int adaptive_saturation_min_accepted_batches = 0;
  int adaptive_saturation_nonproductive_batch_limit = 0;
};

struct Stage5IncrementalBackendBatchInput {
  int frame_index = -1;
  std::string frame_label;
  std::set<std::pair<int, int> > frame_board_keys;
  bool force = false;
  bool has_intrinsics_diversity_anchor = false;
  double ordering_score = 0.0;
  double coverage_gain = 0.0;
  double max_trial_rmse = 0.0;
  double residual_health_threshold_px = 0.0;
  double residual_health_threshold_metric = 0.0;
};

struct Stage5IncrementalBackendBatchResult {
  bool attempted = false;
  bool batch_accepted = false;
  bool force = false;
  bool optimization_success = false;
  bool objective_finite = false;
  bool objective_decreased = false;
  bool objective_gate_pass = false;
  bool information_gate_pass = false;
  bool residual_health_pass = true;
  bool split_residual_health_pass = true;
  bool pixel_safety_gate_pass = true;
  bool full_training_pose_refit_health_pass = true;
  bool ray_curve_validity_pass = true;
  bool kb_k3_released = true;
  bool kb_k4_released = true;
  bool trust_region_pass = true;
  bool trust_region_backtracking_used = false;
  bool solution_valid = false;
  int frame_index = -1;
  std::string frame_label;
  int batch_board_observation_count = 0;
  int batch_point_count = 0;
  int num_iterations = 0;
  int last_solver_pass_iterations = 0;
  int continuation_round_count = 0;
  bool continuation_guard_hit = false;
  bool pose_prefit_attempted = false;
  bool pose_prefit_success = false;
  int pose_prefit_iterations = 0;
  double pose_prefit_objective_start = 0.0;
  double pose_prefit_objective_final = 0.0;
  double pose_prefit_last_delta_j = 0.0;
  double pose_prefit_last_delta_x = 0.0;
  double information_gain = 0.0;
  double normalized_information_gain = 0.0;
  int information_gain_normalization_count = 1;
  double information_gain_threshold = 0.0;
  int rank_psi_after = -1;
  int rank_psi_deficiency_after = -1;
  int rank_theta_before = -1;
  int rank_theta_after = -1;
  int rank_theta_deficiency_after = -1;
  double rank_gain_threshold = 0.0;
  double svd_tolerance = 0.0;
  double qr_tolerance = 0.0;
  double objective_start = 0.0;
  double objective_final = 0.0;
  double objective_last_delta_j = 0.0;
  double state_last_delta_x = 0.0;
  bool linear_solver_failure = false;
  bool converged_by_relative_objective = false;
  bool converged_by_camera_step = false;
  double last_camera_shape_step = 0.0;
  double last_camera_focal_relative_step = 0.0;
  double last_camera_principal_step_px = 0.0;
  double elapsed_time_seconds = 0.0;
  double rmse_after = 0.0;
  double outer_rmse_after = 0.0;
  double internal_rmse_after = 0.0;
  double rmse_before = 0.0;
  double outer_rmse_before = 0.0;
  double internal_rmse_before = 0.0;
  std::string acceptance_metric_name;
  std::string acceptance_metric_unit;
  double acceptance_metric_threshold = 0.0;
  double acceptance_metric_before = 0.0;
  double acceptance_metric_after = 0.0;
  double acceptance_metric_candidate = 0.0;
  double acceptance_metric_candidate_p95 = 0.0;
  double acceptance_metric_candidate_outer = 0.0;
  double acceptance_metric_candidate_internal = 0.0;
  double total_p95_after = 0.0;
  double outer_p95_after = 0.0;
  double internal_p95_after = 0.0;
  double candidate_rmse_after = 0.0;
  double candidate_outer_rmse_after = 0.0;
  double candidate_internal_rmse_after = 0.0;
  double candidate_total_p95_after = 0.0;
  double candidate_outer_p95_after = 0.0;
  double candidate_internal_p95_after = 0.0;
  double pixel_rmse_before = 0.0;
  double pixel_rmse_after = 0.0;
  double pixel_p95_before = 0.0;
  double pixel_p95_after = 0.0;
  double candidate_pixel_rmse_after = 0.0;
  double candidate_pixel_p95_after = 0.0;
  double full_training_pixel_rmse_before = 0.0;
  double full_training_pixel_rmse_after = 0.0;
  double full_training_pixel_p95_before = 0.0;
  double full_training_pixel_p95_after = 0.0;
  double full_training_pose_success_rate_before = 0.0;
  double full_training_pose_success_rate_after = 0.0;
  int full_training_pose_success_count_before = 0;
  int full_training_pose_success_count_after = 0;
  int full_training_pose_total_count = 0;
  int full_training_invalid_projection_count_before = 0;
  int full_training_invalid_projection_count_after = 0;
  double ray_curve_rms_change_deg = 0.0;
  double ray_curve_max_change_deg = 0.0;
  double ray_curve_min_radial_derivative = 0.0;
  int image_plane_residual_count = 0;
  int angular_residual_count = 0;
  int chordal_residual_count = 0;
  int hybrid_angular_selected_count = 0;
  int hybrid_chordal_selected_count = 0;
  int angular_observation_geometry_failure_count = 0;
  int angular_local_whitening_success_count = 0;
  int angular_local_whitening_failure_count = 0;
  int angular_local_whitening_clamped_count = 0;
  double angular_local_whitening_sigma_sum_rad = 0.0;
  double angular_local_whitening_sigma_min_rad = 0.0;
  double angular_local_whitening_sigma_max_rad = 0.0;
  double angular_local_whitening_weight_sum = 0.0;
  double angular_local_whitening_weight_min = 0.0;
  double angular_local_whitening_weight_max = 0.0;
  double max_trial_rmse = 0.0;
  double residual_health_threshold_px = 0.0;
  double residual_health_threshold_metric = 0.0;
  int trust_region_retry_count = 0;
  double trust_region_violation_ratio = 1.0;
  double trust_region_anchor_weight_scale = 1.0;
  double camera_xi_before = 0.0;
  double camera_alpha_before = 0.0;
  double camera_fu_before = 0.0;
  double camera_fv_before = 0.0;
  double camera_cu_before = 0.0;
  double camera_cv_before = 0.0;
  double camera_xi_after = 0.0;
  double camera_alpha_after = 0.0;
  double camera_fu_after = 0.0;
  double camera_fv_after = 0.0;
  double camera_cu_after = 0.0;
  double camera_cv_after = 0.0;
  double camera_k1_before = 0.0;
  double camera_k2_before = 0.0;
  double camera_k3_before = 0.0;
  double camera_k4_before = 0.0;
  double camera_k1_after = 0.0;
  double camera_k2_after = 0.0;
  double camera_k3_after = 0.0;
  double camera_k4_after = 0.0;
  std::string accept_reason;
  std::string reject_reason;
  std::string committed_or_rollback;
};

struct Stage5IncrementalBackendEstimatorResult {
  bool attempted = false;
  bool success = false;
  bool compatible = false;
  std::string fallback_reason;
  std::string information_gain_target = "camera_intrinsics_only";
  bool board_layout_in_information_group = false;
  bool board_layout_fixed = false;
  int board_layout_pose_count = 0;
  double board_layout_max_matrix_abs_delta = 0.0;
  double board_layout_max_translation_delta = 0.0;
  double board_layout_max_rotation_delta_deg = 0.0;
  int camera_information_group_id = 0;
  int board_layout_group_id = 1;
  int transformation_group_id = 2;
  int seed_information_group_dim = -1;
  int seed_information_rank = -1;
  int seed_information_rank_deficiency = -1;
  bool seed_information_baseline_valid = false;
  double seed_information_scaled_min_singular_value = -1.0;
  double seed_information_scaled_max_singular_value = -1.0;
  double seed_information_scaled_condition_number = -1.0;
  double seed_information_ds_cu_stddev_px = -1.0;
  double seed_information_ds_cv_stddev_px = -1.0;
  int seed_batch_count = 0;
  int seed_frame_count = 0;
  int seed_board_observation_count = 0;
  int seed_point_count = 0;
  bool seed_outer_only_residuals = false;
  bool independent_frame_board_camera_warmup_requested = false;
  bool independent_frame_board_camera_warmup_attempted = false;
  bool independent_frame_board_camera_warmup_success = false;
  bool independent_frame_board_camera_warmup_committed = false;
  bool independent_frame_board_camera_warmup_health_pass = false;
  int independent_frame_board_camera_warmup_pose_count = 0;
  int independent_frame_board_camera_warmup_point_count = 0;
  int independent_frame_board_camera_warmup_iterations = 0;
  double independent_frame_board_camera_warmup_objective_start = 0.0;
  double independent_frame_board_camera_warmup_objective_final = 0.0;
  double independent_frame_board_camera_warmup_rmse_before = 0.0;
  double independent_frame_board_camera_warmup_rmse_after = 0.0;
  double independent_frame_board_camera_warmup_p95_before = 0.0;
  double independent_frame_board_camera_warmup_p95_after = 0.0;
  int independent_frame_board_camera_warmup_seed_quarantined_count = 0;
  int independent_frame_board_camera_warmup_instability_quarantined_count = 0;
  bool independent_frame_board_camera_warmup_quarantine_retry_attempted = false;
  bool independent_frame_board_camera_warmup_quarantine_retry_success = false;
  std::string independent_frame_board_camera_warmup_quarantine_reason;
  std::string independent_frame_board_camera_warmup_rollback_reason;
  bool seed_intrinsics_warmup_attempted = false;
  bool seed_intrinsics_warmup_success = false;
  bool seed_intrinsics_warmup_converged_by_relative_objective = false;
  int seed_intrinsics_warmup_iterations = 0;
  double seed_intrinsics_warmup_objective_start = 0.0;
  double seed_intrinsics_warmup_objective_final = 0.0;
  double seed_intrinsics_warmup_last_delta_j = 0.0;
  double seed_intrinsics_warmup_last_delta_x = 0.0;
  int candidate_batch_count = 0;
  int attempted_batch_count = 0;
  int accepted_batch_count = 0;
  int rejected_batch_count = 0;
  std::map<std::string, int> rejection_reason_counts;
  // Stable categories for aggregate reports.  rejection_reason_counts keeps
  // the detailed text for per-batch debugging; this map is intentionally not
  // keyed by dynamic numeric diagnostics.
  std::map<std::string, int> rejection_reason_code_counts;
  std::string solver_profile_name;
  std::string solver_objective_unit;
  int solver_max_iterations = 0;
  double solver_convergence_delta_j = 0.0;
  double solver_convergence_delta_x = 0.0;
  double solver_bearing_reference_focal_px = 1.0;
  double solver_bearing_residual_scale = 1.0;
  int solver_single_iteration_batch_count = 0;
  int solver_max_iteration_batch_count = 0;
  int solver_objective_decreased_batch_count = 0;
  int solver_relative_objective_converged_batch_count = 0;
  int solver_camera_step_converged_batch_count = 0;
  int solver_continuation_batch_count = 0;
  int solver_continuation_round_count = 0;
  int solver_continuation_guard_hit_count = 0;
  int image_plane_residual_count = 0;
  int angular_residual_count = 0;
  int chordal_residual_count = 0;
  int hybrid_angular_selected_count = 0;
  int hybrid_chordal_selected_count = 0;
  int angular_observation_geometry_failure_count = 0;
  int angular_local_whitening_success_count = 0;
  int angular_local_whitening_failure_count = 0;
  int angular_local_whitening_clamped_count = 0;
  double angular_local_whitening_sigma_sum_rad = 0.0;
  double angular_local_whitening_sigma_min_rad = 0.0;
  double angular_local_whitening_sigma_max_rad = 0.0;
  double angular_local_whitening_weight_sum = 0.0;
  double angular_local_whitening_weight_min = 0.0;
  double angular_local_whitening_weight_max = 0.0;
  int trust_region_backtracking_batch_count = 0;
  int trust_region_backtracking_attempt_count = 0;
  int trust_region_backtracking_accepted_count = 0;
  double trust_region_backtracking_max_anchor_scale = 1.0;
  bool normalize_information_gain_by_board_observation = false;
  bool split_residual_health_gate_enabled = false;
  int split_residual_health_rejected_count = 0;
  bool bearing_pixel_safety_gate_enabled = false;
  int bearing_pixel_safety_rejected_count = 0;
  bool full_training_pose_refit_health_gate_enabled = false;
  bool seed_intrinsics_warmup_full_training_health_pass = true;
  int full_training_pose_refit_health_rejected_count = 0;
  double initial_full_training_pixel_rmse = 0.0;
  double initial_full_training_pixel_p95 = 0.0;
  double initial_full_training_pose_success_rate = 0.0;
  int initial_full_training_pose_success_count = 0;
  int initial_full_training_pose_total_count = 0;
  int initial_full_training_invalid_projection_count = 0;
  int initial_full_training_invalid_outer_projection_count = 0;
  int initial_full_training_invalid_internal_projection_count = 0;
  double final_full_training_pixel_rmse = 0.0;
  double final_full_training_pixel_p95 = 0.0;
  double final_full_training_pose_success_rate = 0.0;
  int final_full_training_pose_success_count = 0;
  int final_full_training_pose_total_count = 0;
  int final_full_training_invalid_projection_count = 0;
  int final_full_training_invalid_outer_projection_count = 0;
  int final_full_training_invalid_internal_projection_count = 0;
  bool curated_bundle_state_consistency_pass = true;
  bool curated_bundle_shared_scene_health_pass = true;
  bool curated_bundle_used_validated_baseline_fallback = false;
  double committed_state_pixel_rmse = 0.0;
  double curated_bundle_pixel_rmse = 0.0;
  double curated_bundle_state_consistency_tolerance_px = 0.0;
  double validated_baseline_pixel_rmse = 0.0;
  double curated_bundle_shared_scene_rmse_limit_px = 0.0;
  bool kb_distortion_guard_enabled = false;
  int kb_ray_curve_validity_rejected_count = 0;
  bool adaptive_saturation_stop_enabled = false;
  std::string selection_metric_name;
  std::string selection_metric_unit;
  std::string residual_health_threshold_source;
  double residual_health_threshold_metric = 0.0;
  double seed_acceptance_metric_rmse = 0.0;
  double seed_acceptance_metric_p95 = 0.0;
  bool adaptive_saturation_stop_hit = false;
  int adaptive_saturation_min_accepted_batches = 0;
  int adaptive_saturation_nonproductive_batch_limit = 0;
  int adaptive_saturation_consecutive_nonproductive_batches = 0;
  double adaptive_saturation_tail_ordering_score_threshold = 0.0;
  double adaptive_saturation_next_ordering_score = 0.0;
  std::string adaptive_saturation_stop_reason;
  double total_elapsed_time_seconds = 0.0;
  CalibrationStateBundle curated_bundle;
  CalibrationSceneState optimized_scene_state;
  std::set<std::pair<int, int> > accepted_keys;
  std::set<std::pair<int, int> > quarantined_keys;
  std::vector<Stage5IncrementalBackendBatchResult> batch_results;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

bool IsStage5IncrementalBackendEstimatorCompatible(
    const CalibrationStateBundle& baseline_bundle,
    const CalibrationStateBundle& candidate_pool_bundle,
    const BackendProblemOptions& backend_options,
    const TrialBackendFrameBoardSelectionOptions& selection_options,
    const AslamBackendCalibrationOptions& backend_runner_options,
    std::string* reason);

Stage5IncrementalBackendEstimatorResult RunStage5IncrementalBackendEstimator(
    const CalibrationStateBundle& baseline_bundle,
    const CalibrationStateBundle& candidate_pool_bundle,
    const BackendProblemOptions& backend_options,
    const TrialBackendFrameBoardSelectionOptions& selection_options,
    const AslamBackendCalibrationOptions& backend_runner_options,
    const std::vector<Stage5IncrementalBackendBatchInput>& candidate_batches);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_INCREMENTAL_BACKEND_ESTIMATOR_HPP
