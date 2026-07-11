#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_INCREMENTAL_BACKEND_ESTIMATOR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_INCREMENTAL_BACKEND_ESTIMATOR_HPP

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
  double information_gain_threshold = 0.2;
  double rank_gain_threshold = 1e-6;
  int max_iterations = 5;
  double convergence_delta_j = 1e-3;
  double convergence_delta_x = 1e-4;
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
  AslamBackendCalibrationOptions::AngularObservedRayMode angular_observed_ray_mode =
      AslamBackendCalibrationOptions::AngularObservedRayMode::DynamicCurrentCamera;
  bool optimize_seed_intrinsics = false;
  bool optimize_candidate_intrinsics = true;
  bool use_candidate_intrinsics_anchor_prior = false;
  bool normalize_information_gain_by_board_observation = false;
  bool use_split_residual_health_gate = true;
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
  bool trust_region_pass = true;
  bool trust_region_backtracking_used = false;
  bool solution_valid = false;
  int frame_index = -1;
  std::string frame_label;
  int batch_board_observation_count = 0;
  int batch_point_count = 0;
  int num_iterations = 0;
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
  int image_plane_residual_count = 0;
  int angular_residual_count = 0;
  int chordal_residual_count = 0;
  int hybrid_angular_selected_count = 0;
  int hybrid_chordal_selected_count = 0;
  int angular_observation_geometry_failure_count = 0;
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
  int camera_information_group_id = 0;
  int board_layout_group_id = 1;
  int transformation_group_id = 2;
  int seed_information_group_dim = -1;
  int seed_batch_count = 0;
  int seed_frame_count = 0;
  int seed_board_observation_count = 0;
  int seed_point_count = 0;
  int candidate_batch_count = 0;
  int attempted_batch_count = 0;
  int accepted_batch_count = 0;
  int rejected_batch_count = 0;
  int image_plane_residual_count = 0;
  int angular_residual_count = 0;
  int chordal_residual_count = 0;
  int hybrid_angular_selected_count = 0;
  int hybrid_chordal_selected_count = 0;
  int angular_observation_geometry_failure_count = 0;
  int trust_region_backtracking_batch_count = 0;
  int trust_region_backtracking_attempt_count = 0;
  int trust_region_backtracking_accepted_count = 0;
  double trust_region_backtracking_max_anchor_scale = 1.0;
  bool normalize_information_gain_by_board_observation = false;
  bool split_residual_health_gate_enabled = false;
  int split_residual_health_rejected_count = 0;
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
