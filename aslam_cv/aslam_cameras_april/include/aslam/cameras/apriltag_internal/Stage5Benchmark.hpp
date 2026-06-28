#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_BENCHMARK_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_BENCHMARK_HPP

#include <array>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>
#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>
#include <aslam/cameras/apriltag_internal/JointMeasurementCuration.hpp>
#include <aslam/cameras/apriltag_internal/KalibrStyleBatchAcceptance.hpp>
#include <aslam/cameras/apriltag_internal/KalibrBenchmark.hpp>
#include <aslam/cameras/apriltag_internal/OuterDetectionCache.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct CalibrationBenchmarkSplitOptions {
  std::string mode = "deterministic_stride";
  int holdout_stride = 5;
  int holdout_offset = 0;
  int minimum_training_frames = 3;
  int minimum_holdout_frames = 1;
};

struct CalibrationBenchmarkSplit {
  bool success = false;
  std::string mode = "deterministic_stride";
  int holdout_stride = 5;
  int holdout_offset = 0;
  std::string split_signature;
  std::vector<FrozenRound2BaselineFrameSource> training_frames;
  std::vector<FrozenRound2BaselineFrameSource> holdout_frames;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct CalibrationEvaluationPointObservation {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  JointPointType point_type = JointPointType::Outer;
  Eigen::Vector2d image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector3d target_xyz_board = Eigen::Vector3d::Zero();
  double quality = 0.0;
  int frame_storage_index = -1;
  int source_board_observation_index = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind = JointObservationSourceKind::OuterMeasurement;
};

struct CalibrationEvaluationBoardObservation {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  std::vector<CalibrationEvaluationPointObservation> points;
  int outer_point_count = 0;
  int internal_point_count = 0;
  bool has_pose_fit_outer_points = false;
};

struct CalibrationEvaluationFrameInput {
  int frame_index = -1;
  std::string frame_label;
  std::vector<int> visible_board_ids;
  std::vector<CalibrationEvaluationBoardObservation> board_observations;
};

struct CalibrationEvaluationDataset {
  bool success = false;
  std::string dataset_label;
  std::string split_label;
  std::string split_signature;
  std::vector<CalibrationEvaluationFrameInput> frames;
  int frame_count = 0;
  int board_observation_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  int total_point_count = 0;
  std::vector<InternalRegenerationFrameResult> internal_regeneration_results;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct CameraModelRefitPointDiagnostics {
  std::string method_label;
  std::string split_label;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  JointPointType point_type = JointPointType::Outer;
  Eigen::Vector2d observed_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d predicted_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector3d target_xyz_board = Eigen::Vector3d::Zero();
  Eigen::Vector2d residual_xy = Eigen::Vector2d::Zero();
  double residual_norm = 0.0;
  double quality = 0.0;
  int frame_storage_index = -1;
  int source_board_observation_index = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind = JointObservationSourceKind::OuterMeasurement;
};

struct CameraModelRefitBoardObservationDiagnostics {
  std::string method_label;
  std::string split_label;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  bool pose_only_refit_success = false;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double pose_fit_outer_rmse = 0.0;
  double evaluation_rmse = 0.0;
  double outer_evaluation_rmse = 0.0;
  double internal_evaluation_rmse = 0.0;
  std::string failure_reason;
};

struct CameraModelRefitFrameDiagnostics {
  std::string method_label;
  std::string split_label;
  int frame_index = -1;
  std::string frame_label;
  int pose_only_refit_attempt_count = 0;
  int pose_only_refit_success_count = 0;
  double pose_only_refit_success_rate = 0.0;
  double pose_only_refit_rmse = 0.0;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double rmse = 0.0;
  double outer_rmse = 0.0;
  double internal_rmse = 0.0;
};

struct CameraModelRefitEvaluationResult {
  bool success = false;
  std::string method_label;
  std::string split_label;
  std::string split_signature;
  OuterBootstrapCameraIntrinsics camera;
  int evaluated_frame_count = 0;
  int evaluated_board_observation_count = 0;
  int pose_only_refit_attempt_count = 0;
  int pose_only_refit_success_count = 0;
  double pose_only_refit_success_rate = 0.0;
  double pose_only_refit_rmse = 0.0;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double overall_rmse = 0.0;
  double outer_only_rmse = 0.0;
  double internal_only_rmse = 0.0;
  int excluded_board_id_for_rmse = 1;
  int point_count_excluding_board = 0;
  int outer_point_count_excluding_board = 0;
  int internal_point_count_excluding_board = 0;
  double overall_rmse_excluding_board = 0.0;
  double outer_only_rmse_excluding_board = 0.0;
  double internal_only_rmse_excluding_board = 0.0;
  double mean_residual_x = 0.0;
  double mean_residual_y = 0.0;
  double std_residual_x = 0.0;
  double std_residual_y = 0.0;
  std::vector<CameraModelRefitPointDiagnostics> point_diagnostics;
  std::vector<CameraModelRefitBoardObservationDiagnostics> board_observation_diagnostics;
  std::vector<CameraModelRefitFrameDiagnostics> frame_diagnostics;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct MultiBoardConsistencyObservationDiagnostics {
  int frame_id = -1;
  std::string frame_label;
  int board_id = -1;
  bool used_in_backend = false;
  bool local_pose_refit_success = false;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double local_outer_rmse = 0.0;
  double global_reprojection_rmse = 0.0;
  double translation_error_mm = 0.0;
  double rotation_error_deg = 0.0;
  double polar_angle_mean_deg = 0.0;
  double polar_angle_max_deg = 0.0;
  double residual_rmse = 0.0;
  double outer_rmse = 0.0;
  double internal_rmse = 0.0;
  std::string failure_reason;
};

struct MultiBoardConsistencyBoardDiagnostics {
  int board_id = -1;
  int support_observation_count = 0;
  double mean_translation_error_mm = 0.0;
  double median_translation_error_mm = 0.0;
  double p90_translation_error_mm = 0.0;
  double max_translation_error_mm = 0.0;
  double mean_rotation_error_deg = 0.0;
  double median_rotation_error_deg = 0.0;
  double p90_rotation_error_deg = 0.0;
  double max_rotation_error_deg = 0.0;
  int worst_frame_id = -1;
};

struct MultiBoardConsistencyFrameDiagnostics {
  int frame_id = -1;
  int observed_board_count = 0;
  double mean_translation_error_mm = 0.0;
  double max_translation_error_mm = 0.0;
  double mean_rotation_error_deg = 0.0;
  double max_rotation_error_deg = 0.0;
  int worst_board_id = -1;
  double frame_reprojection_rmse = 0.0;
};

enum class MultiBoardConsistencyPoseSource {
  OuterOnly,
};

const char* ToString(MultiBoardConsistencyPoseSource source);
MultiBoardConsistencyPoseSource ParseMultiBoardConsistencyPoseSource(
    const std::string& value);

struct MultiBoardConsistencyDiagnosticsOptions {
  bool enabled = false;
  MultiBoardConsistencyPoseSource pose_source =
      MultiBoardConsistencyPoseSource::OuterOnly;
  int min_outer_points = 4;
};

struct MultiBoardConsistencyDiagnosticsResult {
  bool success = false;
  bool training_only = true;
  std::string split_label = "training";
  std::string pose_source_label = "outer_only";
  bool optimized_intrinsics_fixed = true;
  bool tag_size_assumed_meters = true;
  int frame_count = 0;
  int board_observation_count = 0;
  int successful_local_pose_refit_count = 0;
  std::vector<MultiBoardConsistencyObservationDiagnostics>
      observation_diagnostics;
  std::vector<MultiBoardConsistencyBoardDiagnostics> board_diagnostics;
  std::vector<MultiBoardConsistencyFrameDiagnostics> frame_diagnostics;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct KalibrBenchmarkReference {
  std::string camchain_yaml;
  std::string camera_model_family = "ds";
  std::string training_split_signature;
  double runtime_seconds = -1.0;
  std::string source_label;
};

struct Stage5BenchmarkRuntimeBreakdown {
  double split_seconds = 0.0;
  double training_dataset_build_seconds = 0.0;
  double holdout_dataset_build_seconds = 0.0;
  double diagnostic_compare_seconds = 0.0;
  double internal_joint_refine_seconds = 0.0;
  double internal_blur_board_weight_seconds = 0.0;
  double internal_observation_weight_seconds = 0.0;
  double pre_backend_filter_seconds = 0.0;
  double internal_blur_filter_seconds = 0.0;
  OuterDetectionCacheStats holdout_detection_cache;
};

struct TrialBackendFrameBoardSelectionOptions {
  enum class SelectionMode {
    StrictRmse = 0,
    KalibrStyleBatch = 1,
  };
  enum class BudgetMode {
    Fixed = 0,
    Adaptive = 1,
    KalibrStyle = 2,
  };
  enum class CandidateOrderMode {
    ScoreSorted = 0,
    RandomShuffle = 1,
  };
  enum class InfoGainProxyMode {
    Legacy = 0,
    IntrinsicsJacobian = 1,
  };
  enum class CandidateBatchGranularity {
    FrameBoard = 0,
    Frame = 1,
    FrameBoardThenFrame = 2,
  };

  bool enabled = false;
  SelectionMode selection_mode = SelectionMode::KalibrStyleBatch;
  BudgetMode budget_mode = BudgetMode::KalibrStyle;
  CandidateOrderMode candidate_order_mode = CandidateOrderMode::RandomShuffle;
  InfoGainProxyMode info_gain_proxy_mode = InfoGainProxyMode::IntrinsicsJacobian;
  CandidateBatchGranularity candidate_batch_granularity =
      CandidateBatchGranularity::Frame;
  KalibrStyleBatchAcceptancePolicy acceptance_policy =
      KalibrStyleBatchAcceptancePolicy::KalibrInformationGain;
  double acceptance_information_gain_threshold = 0.2;
  double acceptance_rank_gain_threshold = 1e-6;
  bool candidate_shuffle_seed_set = false;
  unsigned int candidate_shuffle_seed = 0;
  bool incremental_acceptance = true;
  bool carry_accepted_trial_state = true;
  bool optimize_intrinsics_in_trial = true;
  bool delayed_intrinsics_release_in_trial = true;
  int intrinsics_release_iteration = 1;
  bool persistent_intrinsics_anchor_prior_enabled = true;
  double persistent_intrinsics_anchor_weight_xi_alpha = 0.0;
  double persistent_intrinsics_anchor_weight_focal = 0.0;
  double persistent_intrinsics_anchor_weight_principal = 0.0;
  double persistent_max_focal_relative_step = 0.03;
  double persistent_max_principal_step_px = 20.0;
  double persistent_max_xi_alpha_step = 0.03;
  int max_iterations = 5;
  int max_candidate_additions = 60;
  double adaptive_budget_ratio = 0.10;
  int adaptive_budget_min = 20;
  int adaptive_budget_max = 120;
  int runtime_safety_ceiling = 1000;
  double outlier_sigma = 4.0;
  double min_abs_threshold_px = 1.0;
  double max_threshold_px = 25.0;
  double accept_max_global_rmse_increase_px = 0.02;
  double accept_max_outer_rmse_increase_px = 0.05;
  double accept_max_internal_rmse_increase_px = 0.05;
  double min_candidate_score = 0.0;
  double min_coverage_gain = 0.0;
  bool use_consistency_score = false;
  double consistency_translation_sigma_mm = 3.0;
  double consistency_rotation_sigma_deg = 2.0;
  double consistency_penalty_weight = 1.0;
  double consistency_max_translation_error_mm = -1.0;
  double consistency_max_rotation_error_deg = -1.0;
  double consistency_max_local_outer_rmse_px = -1.0;
  int max_accepted_per_board = 0;
  int max_accepted_per_frame = 0;
  bool frame_cohesion_enabled = true;
  // <= 0 means rescue companions until the frame's observed board count.
  int frame_cohesion_max_companions_per_frame = 0;
  double frame_cohesion_min_candidate_score = 3.2;
  int min_keep_observations_per_board = 5;
  std::set<std::pair<int, int> > force_include_frame_board_keys;
  std::set<std::pair<std::string, int> > force_include_frame_label_board_keys;
};

struct TrialBackendFrameBoardObservationDecision {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double trial_rmse = 0.0;
  bool baseline_seed = false;
  bool attempted_incremental = false;
  double coverage_gain = 0.0;
  double candidate_score = 0.0;
  double polar_gain = 0.0;
  double edge_gain = 0.0;
  double board_balance_gain = 0.0;
  double frame_novelty_gain = 0.0;
  double grid_gain = 0.0;
  double covisibility_gain = 0.0;
  double residual_quality_score = 0.0;
  double consistency_score = 0.0;
  double consistency_penalty = 0.0;
  double consistency_translation_error_mm = 0.0;
  double consistency_rotation_error_deg = 0.0;
  double consistency_local_outer_rmse = 0.0;
  bool consistency_available = false;
  bool force_include_candidate = false;
  bool is_close_edge_hard_case = false;
  bool soft_candidate = false;
  bool soft_attempted = false;
  bool soft_accepted = false;
  bool frame_cohesion_candidate = false;
  bool frame_cohesion_attempted = false;
  bool frame_cohesion_accepted = false;
  bool frame_batch_candidate = false;
  bool frame_batch_attempted = false;
  bool frame_batch_accepted = false;
  bool frame_consolidation_candidate = false;
  bool frame_consolidation_accepted = false;
  bool close_distance_frame_admission_candidate = false;
  bool close_distance_frame_admission_attempted = false;
  bool close_distance_frame_admission_accepted = false;
  bool close_distance_candidate = false;
  bool intrinsics_diversity_anchor = false;
  double mean_polar_angle_deg = 0.0;
  double max_polar_angle_deg = 0.0;
  double projected_area_px = 0.0;
  double projected_area_ratio = 0.0;
  double outer_pose_refit_rmse = 0.0;
  double close_edge_score = 0.0;
  double close_distance_score_bonus = 0.0;
  double soft_weight = 1.0;
  double global_rmse_before = 0.0;
  double global_rmse_after = 0.0;
  double global_rmse_delta = 0.0;
  double outer_rmse_delta = 0.0;
  double internal_rmse_delta = 0.0;
  double soft_global_rmse_delta = 0.0;
  double soft_outer_rmse_delta = 0.0;
  double soft_internal_rmse_delta = 0.0;
  bool hard_validity_pass = false;
  bool legacy_rmse_pass = false;
  bool catastrophic_residual = false;
  double score_term = 0.0;
  double coverage_term = 0.0;
  double intrinsics_jacobian_logdet_gain = 0.0;
  double intrinsics_jacobian_trace_gain = 0.0;
  double intrinsics_jacobian_rank_gain = 0.0;
  double intrinsics_jacobian_info_term = 0.0;
  double frame_completion_bonus = 0.0;
  double new_board_bonus = 0.0;
  double cap_penalty = 0.0;
  double information_gain_proxy = 0.0;
  double residual_overage_penalty = 0.0;
  double batch_acceptance_score = 0.0;
  bool accepted_by_batch_acceptance = false;
  bool persistent_incremental_attempted = false;
  bool persistent_incremental_batch_accepted = false;
  bool persistent_incremental_force = false;
  double persistent_incremental_information_gain = 0.0;
  int persistent_incremental_rank_theta_before = -1;
  int persistent_incremental_rank_theta_after = -1;
  int persistent_incremental_iterations = 0;
  double persistent_incremental_objective_start = 0.0;
  double persistent_incremental_objective_final = 0.0;
  bool persistent_incremental_objective_decreased = false;
  double persistent_incremental_elapsed_time_seconds = 0.0;
  std::string persistent_incremental_commit_state;
  double persistent_incremental_camera_xi_before = 0.0;
  double persistent_incremental_camera_alpha_before = 0.0;
  double persistent_incremental_camera_fu_before = 0.0;
  double persistent_incremental_camera_fv_before = 0.0;
  double persistent_incremental_camera_cu_before = 0.0;
  double persistent_incremental_camera_cv_before = 0.0;
  double persistent_incremental_camera_xi_after = 0.0;
  double persistent_incremental_camera_alpha_after = 0.0;
  double persistent_incremental_camera_fu_after = 0.0;
  double persistent_incremental_camera_fv_after = 0.0;
  double persistent_incremental_camera_cu_after = 0.0;
  double persistent_incremental_camera_cv_after = 0.0;
  double left_rmse = 0.0;
  double right_rmse = 0.0;
  double center_side_rmse = 0.0;
  double edge_side_rmse = 0.0;
  bool kept = true;
  std::string reason;
};

struct TrialBackendOptimizationDiagnostics {
  std::string label;
  bool success = false;
  int design_variable_count = 0;
  int error_term_count = 0;
  double initial_overall_rmse = 0.0;
  double optimized_overall_rmse = 0.0;
  double initial_outer_rmse = 0.0;
  double optimized_outer_rmse = 0.0;
  double initial_internal_rmse = 0.0;
  double optimized_internal_rmse = 0.0;
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
  int stage_count = 0;
  int total_iterations = 0;
  int total_failed_iterations = 0;
  bool any_intrinsics_stage = false;
  bool any_linear_solver_failure = false;
  double objective_start_sum = 0.0;
  double objective_final_sum = 0.0;
  double last_delta_x = 0.0;
  double last_delta_j = 0.0;
  double last_lm_lambda = 0.0;
  std::string failure_reason;
};

struct TrialBackendFrameBoardSelectionResult {
  bool enabled = false;
  bool success = false;
  int source_joint_input_frame_count = 0;
  int source_measurement_frame_count = 0;
  int source_measurement_board_observation_count = 0;
  int source_measurement_total_point_count = 0;
  int source_measurement_outer_point_count = 0;
  int source_measurement_internal_point_count = 0;
  int source_measurement_hierarchical_frame_count = 0;
  int source_measurement_hierarchical_board_observation_count = 0;
  int source_measurement_hierarchical_total_point_count = 0;
  int source_measurement_hierarchical_outer_point_count = 0;
  int source_measurement_hierarchical_internal_point_count = 0;
  int source_measurement_flat_solver_observation_count = 0;
  int source_selection_frame_count = 0;
  int source_selection_board_observation_count = 0;
  int source_selection_total_point_count = 0;
  int source_selection_outer_point_count = 0;
  int source_selection_internal_point_count = 0;
  int candidate_pool_frame_count = 0;
  int candidate_pool_board_observation_count = 0;
  int candidate_pool_total_point_count = 0;
  int candidate_pool_outer_point_count = 0;
  int candidate_pool_internal_point_count = 0;
  int input_frame_count = 0;
  int input_board_observation_count = 0;
  int input_total_point_count = 0;
  int baseline_seed_frame_count = 0;
  int baseline_seed_board_observation_count = 0;
  int baseline_seed_total_point_count = 0;
  int baseline_seed_outer_point_count = 0;
  int baseline_seed_internal_point_count = 0;
  int candidate_board_observation_count = 0;
  TrialBackendFrameBoardSelectionOptions::BudgetMode budget_mode =
      TrialBackendFrameBoardSelectionOptions::BudgetMode::KalibrStyle;
  TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
      candidate_order_mode =
          TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
              ::RandomShuffle;
  TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode
      info_gain_proxy_mode =
          TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode::Legacy;
  TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity
      candidate_batch_granularity =
          TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity
              ::FrameBoard;
  KalibrStyleBatchAcceptancePolicy acceptance_policy =
      KalibrStyleBatchAcceptancePolicy::ResidualScore;
  double acceptance_information_gain_threshold = 0.0;
  double acceptance_rank_gain_threshold = 0.0;
  bool candidate_shuffle_seed_set = false;
  unsigned int candidate_shuffle_seed = 0;
  bool carry_accepted_trial_state = false;
  bool optimize_intrinsics_in_trial = false;
  bool delayed_intrinsics_release_in_trial = false;
  int intrinsics_release_iteration = 0;
  bool persistent_intrinsics_anchor_prior_enabled = false;
  double persistent_intrinsics_anchor_weight_xi_alpha = 0.0;
  double persistent_intrinsics_anchor_weight_focal = 0.0;
  double persistent_intrinsics_anchor_weight_principal = 0.0;
  double persistent_max_focal_relative_step = 0.0;
  double persistent_max_principal_step_px = 0.0;
  double persistent_max_xi_alpha_step = 0.0;
  int valid_candidate_count = 0;
  int valid_candidate_traversed_count = 0;
  bool safety_ceiling_hit = false;
  int runtime_safety_ceiling = 0;
  std::string max_candidate_additions_effective;
  int attempted_candidate_count = 0;
  int accepted_candidate_count = 0;
  TrialBackendFrameBoardSelectionOptions::SelectionMode selection_mode =
      TrialBackendFrameBoardSelectionOptions::SelectionMode::KalibrStyleBatch;
  int batch_acceptance_attempted_count = 0;
  int batch_acceptance_accepted_count = 0;
  int batch_acceptance_rescued_from_legacy_rmse_gate_count = 0;
  int batch_acceptance_rejected_hard_validity_count = 0;
  int batch_acceptance_rejected_catastrophic_residual_count = 0;
  int batch_acceptance_rejected_score_count = 0;
  bool persistent_incremental_backend_estimator_attempted = false;
  bool persistent_incremental_backend_estimator_used = false;
  bool persistent_incremental_backend_estimator_compatible = false;
  std::string persistent_incremental_backend_estimator_fallback_reason;
  std::string persistent_incremental_backend_estimator_failure_reason;
  int persistent_incremental_seed_batch_count = 0;
  int persistent_incremental_seed_frame_count = 0;
  int persistent_incremental_seed_board_observation_count = 0;
  int persistent_incremental_seed_point_count = 0;
  int persistent_incremental_candidate_batch_count = 0;
  int persistent_incremental_attempted_batch_count = 0;
  int persistent_incremental_accepted_batch_count = 0;
  int persistent_incremental_rejected_batch_count = 0;
  double persistent_incremental_total_elapsed_time_seconds = 0.0;
  int frame_cohesion_candidate_count = 0;
  int frame_cohesion_attempted_count = 0;
  int frame_cohesion_accepted_count = 0;
  int frame_cohesion_rejected_count = 0;
  int frame_batch_candidate_count = 0;
  int frame_batch_attempted_count = 0;
  int frame_batch_accepted_count = 0;
  int frame_batch_rejected_count = 0;
  int frame_consolidation_candidate_count = 0;
  int frame_consolidation_accepted_count = 0;
  int frame_consolidation_rejected_count = 0;
  int frame_consolidation_dropped_board_observation_count = 0;
  int close_distance_candidate_count = 0;
  int close_distance_accepted_count = 0;
  int close_distance_frame_admission_candidate_count = 0;
  int close_distance_frame_admission_attempted_count = 0;
  int close_distance_frame_admission_accepted_count = 0;
  int close_distance_frame_admission_rejected_count = 0;
  int intrinsics_diversity_anchor_candidate_count = 0;
  int intrinsics_diversity_anchor_accepted_count = 0;
  int intrinsics_diversity_anchor_rejected_count = 0;
  int close_edge_hard_case_count = 0;
  int close_edge_soft_candidate_count = 0;
  int close_edge_soft_attempted_count = 0;
  int close_edge_soft_accepted_count = 0;
  int close_edge_soft_rejected_count = 0;
  int kept_frame_count = 0;
  int kept_board_observation_count = 0;
  int kept_total_point_count = 0;
  int kept_outer_point_count = 0;
  int kept_internal_point_count = 0;
  int rejected_board_observation_count = 0;
  double median_board_rmse = 0.0;
  double robust_sigma_board_rmse = 0.0;
  double threshold_px = 0.0;
  AslamBackendCalibrationResult trial_backend_result;
  CalibrationStateBundle curated_bundle;
  std::vector<TrialBackendFrameBoardObservationDecision> decisions;
  std::vector<TrialBackendOptimizationDiagnostics> trial_optimization_diagnostics;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

const char* ToString(
    TrialBackendFrameBoardSelectionOptions::SelectionMode mode);
const char* ToString(
    TrialBackendFrameBoardSelectionOptions::BudgetMode mode);
const char* ToString(
    TrialBackendFrameBoardSelectionOptions::CandidateOrderMode mode);
const char* ToString(
    TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode mode);
const char* ToString(
    TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity mode);

struct BackendInputAblationOptions {
  bool point_budget_control_enabled = false;
  int point_budget_total_points = 0;
  unsigned int point_budget_seed = 1337;
  int max_boards_per_frame_for_ablation = -1;
};

struct BackendInputAblationResult {
  bool enabled = false;
  bool success = true;
  bool point_budget_control_enabled = false;
  int point_budget_total_points = 0;
  unsigned int point_budget_seed = 1337;
  int max_boards_per_frame_for_ablation = -1;
  int input_frame_count = 0;
  int input_board_observation_count = 0;
  int input_outer_point_count = 0;
  int input_internal_point_count = 0;
  int input_total_point_count = 0;
  int output_frame_count = 0;
  int output_board_observation_count = 0;
  int output_outer_point_count = 0;
  int output_internal_point_count = 0;
  int output_total_point_count = 0;
  int removed_board_observation_count = 0;
  int removed_internal_point_count = 0;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct CameraRayCurveSample {
  std::string reference_label;
  std::string reference_family;
  double image_x = 0.0;
  double image_y = 0.0;
  double radial_fraction = 0.0;
  double our_polar_deg = 0.0;
  double reference_polar_deg = 0.0;
  double angular_diff_deg = 0.0;
};

struct CameraRayCurveBucketSummary {
  std::string reference_label;
  std::string reference_family;
  std::string bucket_type;
  std::string bucket_label;
  int sample_count = 0;
  double mean_angular_diff_deg = 0.0;
  double rms_angular_diff_deg = 0.0;
  double max_angular_diff_deg = 0.0;
  double mean_our_polar_deg = 0.0;
  double mean_reference_polar_deg = 0.0;
};

struct CameraRayCurveDiagnostics {
  bool success = false;
  int grid_width = 0;
  int grid_height = 0;
  int comparison_count = 0;
  int sample_count = 0;
  int invalid_unprojection_count = 0;
  std::vector<CameraRayCurveSample> samples;
  std::vector<CameraRayCurveBucketSummary> bucket_summaries;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct Stage5BenchmarkInput {
  std::vector<FrozenRound2BaselineFrameSource> all_frames;
  std::vector<FrozenRound2BaselineFrameSource> external_holdout_frames;
  std::string external_holdout_label;
  FrozenRound2BaselineOptions baseline_options;
  BackendProblemOptions backend_options;
  BackendProblemOptions final_backend_options;
  InternalJointRefineOptions internal_joint_refine_options;
  InternalObservationWeightOptions internal_observation_weight_options;
  InternalBlurBoardWeightOptions internal_blur_board_weight_options;
  PreBackendObservationFilterOptions pre_backend_filter_options;
  InternalBlurObservationFilterOptions internal_blur_filter_options;
  KalibrBenchmarkReference kalibr_reference;
  std::vector<KalibrBenchmarkReference> additional_camera_references;
  std::string dataset_label;
  bool enable_diagnostic_compare = true;
  MultiBoardConsistencyDiagnosticsOptions
      multi_board_consistency_diagnostics_options;
  TrialBackendFrameBoardSelectionOptions trial_backend_selection_options;
  BackendInputAblationOptions backend_input_ablation_options;
};

struct Stage5BenchmarkReport {
  bool success = false;
  bool fair_protocol_matched = false;
  bool diagnostic_only = false;
  std::string dataset_label;
  std::string baseline_protocol_label;
  std::string split_signature;
  CalibrationBenchmarkSplit split;
  FrozenRound2BaselineResult baseline_result;
  InternalJointRefineResult internal_joint_refine_result;
  InternalObservationWeightResult internal_observation_weight_result;
  InternalBlurBoardWeightResult internal_blur_board_weight_result;
  PreBackendObservationFilterResult pre_backend_filter_result;
  InternalBlurObservationFilterResult internal_blur_filter_result;
  TrialBackendFrameBoardSelectionResult trial_backend_selection_result;
  BackendInputAblationResult backend_input_ablation_result;
  CalibrationBackendProblemInput backend_problem_input;
  CalibrationEvaluationDataset training_dataset;
  CalibrationEvaluationDataset holdout_dataset;
  CameraModelRefitEvaluationResult our_training_evaluation;
  CameraModelRefitEvaluationResult kalibr_training_evaluation;
  CameraModelRefitEvaluationResult our_holdout_evaluation;
  CameraModelRefitEvaluationResult kalibr_holdout_evaluation;
  std::vector<KalibrBenchmarkReference> additional_camera_references;
  std::vector<CameraModelRefitEvaluationResult> additional_training_evaluations;
  std::vector<CameraModelRefitEvaluationResult> additional_holdout_evaluations;
  CameraRayCurveDiagnostics camera_ray_curve_diagnostics;
  MultiBoardConsistencyDiagnosticsResult multi_board_consistency_diagnostics;
  KalibrBenchmarkReference kalibr_reference;
  KalibrBenchmarkReport diagnostic_compare;
  Stage5BenchmarkRuntimeBreakdown runtime_breakdown;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class Stage5Benchmark {
 public:
  explicit Stage5Benchmark(
      CalibrationBenchmarkSplitOptions split_options =
          CalibrationBenchmarkSplitOptions{});

  CalibrationBenchmarkSplit BuildDeterministicSplit(
      const std::vector<FrozenRound2BaselineFrameSource>& frames) const;
  CalibrationBenchmarkSplit BuildExternalHoldoutSplit(
      const std::vector<FrozenRound2BaselineFrameSource>& training_frames,
      const std::vector<FrozenRound2BaselineFrameSource>& holdout_frames,
      const std::string& holdout_label) const;

  Stage5BenchmarkReport Run(const Stage5BenchmarkInput& input) const;

  CameraModelRefitEvaluationResult EvaluateCameraModel(
      const CalibrationEvaluationDataset& dataset,
      const OuterBootstrapCameraIntrinsics& camera,
      const std::string& method_label) const;
  MultiBoardConsistencyDiagnosticsResult EvaluateMultiBoardConsistency(
      const CalibrationEvaluationDataset& dataset,
      const CameraModelRefitEvaluationResult& evaluation,
      const CalibrationBackendProblemInput& backend_problem_input,
      const CalibrationStateBundle& final_bundle,
      const MultiBoardConsistencyDiagnosticsOptions& options) const;

  cv::Mat RenderProjectionComparison(const Stage5BenchmarkReport& report,
                                     int max_width = 900,
                                     int max_height = 900) const;
  cv::Mat RenderEvaluationFrameOverlay(
      const Stage5BenchmarkReport& report,
      const CameraModelRefitEvaluationResult& evaluation,
      int frame_index) const;
  cv::Mat RenderEvaluationBoardObservationOverlay(
      const Stage5BenchmarkReport& report,
      const CameraModelRefitEvaluationResult& evaluation,
      int frame_index,
      int board_id) const;
  cv::Mat RenderOuterPoseFitFrameOverlay(
      const Stage5BenchmarkReport& report,
      const CameraModelRefitEvaluationResult& evaluation,
      int frame_index) const;
  cv::Mat RenderOuterPoseFitBoardOverlay(
      const Stage5BenchmarkReport& report,
      const CameraModelRefitEvaluationResult& evaluation,
      int frame_index,
      int board_id) const;

  const CalibrationBenchmarkSplitOptions& split_options() const {
    return split_options_;
  }

  CalibrationEvaluationDataset BuildHoldoutEvaluationDataset(
      const std::vector<FrozenRound2BaselineFrameSource>& holdout_frames,
      const FrozenRound2BaselineOptions& baseline_options,
      const JointReprojectionSceneState& optimized_scene_state,
      const std::string& split_signature,
      OuterDetectionCacheStats* cache_stats) const;

 private:
  CalibrationEvaluationDataset BuildTrainingEvaluationDataset(
      const CalibrationStateBundle& bundle) const;
  std::string FindFrameImagePath(const Stage5BenchmarkReport& report,
                                 int frame_index) const;

  CalibrationBenchmarkSplitOptions split_options_;
};

void WriteStage5BenchmarkProtocolSummary(const std::string& path,
                                         const Stage5BenchmarkReport& report);
void WriteStage5BenchmarkTrainingSummary(const std::string& path,
                                         const Stage5BenchmarkReport& report);
void WriteStage5BenchmarkHoldoutSummary(const std::string& path,
                                        const Stage5BenchmarkReport& report);
void WriteStage5BenchmarkHoldoutPointsCsv(const std::string& path,
                                          const Stage5BenchmarkReport& report);
void WriteCameraModelRefitPointsCsv(
    const std::string& path,
    const std::vector<CameraModelRefitEvaluationResult>& evaluations);
void WriteCameraModelRefitBoardObservationsCsv(
    const std::string& path,
    const std::vector<CameraModelRefitEvaluationResult>& evaluations);
void WriteCameraModelRefitFramesCsv(
    const std::string& path,
    const std::vector<CameraModelRefitEvaluationResult>& evaluations);
void WriteStage5BenchmarkHoldoutBoardObservationsCsv(
    const std::string& path,
    const Stage5BenchmarkReport& report);
void WriteStage5BenchmarkHoldoutFramesCsv(const std::string& path,
                                          const Stage5BenchmarkReport& report);
void WriteStage5BenchmarkWorstCasesSummary(const std::string& path,
                                           const Stage5BenchmarkReport& report,
                                           int top_k = 10);
void WriteCameraRayCurveSamplesCsv(const std::string& path,
                                   const CameraRayCurveDiagnostics& diagnostics);
void WriteCameraRayCurveSummaryCsv(const std::string& path,
                                   const CameraRayCurveDiagnostics& diagnostics);
void WriteMultiBoardConsistencySummary(
    const std::string& path,
    const MultiBoardConsistencyDiagnosticsResult& result);
void WriteMultiBoardConsistencyPerObservationCsv(
    const std::string& path,
    const MultiBoardConsistencyDiagnosticsResult& result);
void WriteMultiBoardConsistencyPerBoardCsv(
    const std::string& path,
    const MultiBoardConsistencyDiagnosticsResult& result);
void WriteMultiBoardConsistencyPerFrameCsv(
    const std::string& path,
    const MultiBoardConsistencyDiagnosticsResult& result);
void WriteBackendInputAblationSummary(
    const std::string& path,
    const BackendInputAblationResult& result);
void WriteBoardLayoutPoseDeltaCsv(
    const std::string& path,
    const AslamBackendCalibrationResult& backend_result);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_BENCHMARK_HPP
