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
  double huber_delta_pixels = 8.0;
  double invalid_projection_penalty_pixels = 100.0;
  bool optimize_seed_intrinsics = false;
  bool optimize_candidate_intrinsics = true;
  bool use_candidate_intrinsics_anchor_prior = true;
  double candidate_intrinsics_anchor_weight_xi_alpha = 0.0;
  double candidate_intrinsics_anchor_weight_focal = 0.0;
  double candidate_intrinsics_anchor_weight_principal = 0.0;
  double max_candidate_focal_relative_step = 0.03;
  double max_candidate_principal_step_px = 20.0;
  double max_candidate_xi_alpha_step = 0.03;
};

struct Stage5IncrementalBackendBatchInput {
  int frame_index = -1;
  std::string frame_label;
  std::set<std::pair<int, int> > frame_board_keys;
  bool force = false;
  double max_trial_rmse = 0.0;
  double residual_health_threshold_px = 0.0;
};

struct Stage5IncrementalBackendBatchResult {
  bool attempted = false;
  bool batch_accepted = false;
  bool force = false;
  bool optimization_success = false;
  bool objective_finite = false;
  bool objective_decreased = false;
  bool information_gate_pass = false;
  bool residual_health_pass = true;
  bool solution_valid = false;
  int frame_index = -1;
  std::string frame_label;
  int batch_board_observation_count = 0;
  int batch_point_count = 0;
  int num_iterations = 0;
  double information_gain = 0.0;
  double information_gain_threshold = 0.0;
  int rank_theta_before = -1;
  int rank_theta_after = -1;
  double rank_gain_threshold = 0.0;
  double objective_start = 0.0;
  double objective_final = 0.0;
  double elapsed_time_seconds = 0.0;
  double rmse_after = 0.0;
  double outer_rmse_after = 0.0;
  double internal_rmse_after = 0.0;
  double max_trial_rmse = 0.0;
  double residual_health_threshold_px = 0.0;
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
  int seed_batch_count = 0;
  int seed_frame_count = 0;
  int seed_board_observation_count = 0;
  int seed_point_count = 0;
  int candidate_batch_count = 0;
  int attempted_batch_count = 0;
  int accepted_batch_count = 0;
  int rejected_batch_count = 0;
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
