#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_BENCHMARK_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_BENCHMARK_HPP

#include <array>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>
#include <aslam/cameras/apriltag_internal/InternalRegenerationCache.hpp>
#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>
#include <aslam/cameras/apriltag_internal/JointMeasurementCuration.hpp>
#include <aslam/cameras/apriltag_internal/KalibrStyleBatchAcceptance.hpp>
#include <aslam/cameras/apriltag_internal/KalibrBenchmark.hpp>
#include <aslam/cameras/apriltag_internal/OuterDetectionCache.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct CalibrationBenchmarkSplitOptions {
  std::string mode = "random_holdout_ratio";
  bool all_frames_training = false;
  int holdout_stride = 5;
  int holdout_offset = 0;
  double holdout_ratio = 0.30;
  unsigned int random_seed = 1337;
  int minimum_training_frames = 3;
  int minimum_holdout_frames = 1;
};

struct CalibrationBenchmarkSplit {
  bool success = false;
  std::string mode = "deterministic_stride";
  int holdout_stride = 5;
  int holdout_offset = 0;
  double holdout_ratio = 0.0;
  unsigned int random_seed = 0;
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
  int camera_aware_outer_rescue_attempted_board_count = 0;
  int camera_aware_outer_rescue_used_board_count = 0;
  bool uniform_control_point_mode = false;
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
  // A point can remain in the diagnostic CSV while being excluded from the
  // primary valid-observation metric (for example, its board pose failed).
  bool evaluation_included = true;
  std::string exclusion_reason;
};

struct CameraModelRefitBoardObservationDiagnostics {
  std::string method_label;
  std::string split_label;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  bool pose_only_refit_success = false;
  Eigen::Matrix4d T_camera_board = Eigen::Matrix4d::Identity();
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double pose_fit_outer_rmse = 0.0;
  bool all_point_pose_refit_success = false;
  int all_point_pose_refit_point_count = 0;
  double all_point_pose_refit_rmse = 0.0;
  double all_point_pose_refit_internal_rmse = 0.0;
  double evaluation_rmse = 0.0;
  double outer_evaluation_rmse = 0.0;
  double internal_evaluation_rmse = 0.0;
  bool evaluation_included = false;
  int invalid_projection_point_count = 0;
  int nonfinite_point_count = 0;
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
  bool uniform_control_point_mode = false;
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
  // Tangent-plane unit-bearing RMSE after the same held-out pose refit used
  // for pixel reprojection evaluation. This is an evaluation metric only.
  int angular_point_count = 0;
  double overall_angular_rmse_rad = 0.0;
  double overall_angular_rmse_deg = 0.0;
  double p95_reprojection_error = 0.0;
  double outer_only_rmse = 0.0;
  double internal_only_rmse = 0.0;
  // Primary metrics above use only valid board observations. These raw metrics
  // retain every observation for transparent stress testing and auditing.
  int valid_board_observation_count = 0;
  int raw_board_observation_count = 0;
  int raw_point_count = 0;
  int raw_outer_point_count = 0;
  int raw_internal_point_count = 0;
  double raw_overall_rmse = 0.0;
  double raw_outer_only_rmse = 0.0;
  double raw_internal_only_rmse = 0.0;
  double raw_p95_reprojection_error = 0.0;
  int invalid_board_observation_count = 0;
  int pose_init_failed_board_observation_count = 0;
  int residual_sanity_failed_board_observation_count = 0;
  int projection_invalid_board_observation_count = 0;
  int projection_failure_point_count = 0;
  int nonfinite_residual_point_count = 0;
  double residual_sanity_threshold_px = 25.0;
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

// Experimental-only held-out multiboard pose evaluation. Calibration never
// consumes this result.
struct MultiBoardPoseOrientationEvaluationResult {
  bool success = false;
  int evaluated_frame_count = 0;
  int pose_success_count = 0;
  double pose_success_rate = 0.0;
  double orientation_median_deg = 0.0;
  double orientation_p95_deg = 0.0;
  std::vector<double> orientation_errors_deg;
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
  double external_holdout_self_frontend_prepass_seconds = 0.0;
  double diagnostic_compare_seconds = 0.0;
  double internal_joint_refine_seconds = 0.0;
  double internal_blur_board_weight_seconds = 0.0;
  double internal_observation_weight_seconds = 0.0;
  double pre_backend_filter_seconds = 0.0;
  double internal_blur_filter_seconds = 0.0;
  OuterDetectionCacheStats holdout_detection_cache;
  InternalRegenerationCacheStats holdout_internal_regeneration_cache;
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
    IntrinsicsInformationGreedy = 2,
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
  // Single planar target profile: one dense target view per candidate batch.
  // The forced bootstrap batch contains an adaptive set of informative views
  // with intrinsics fixed; candidate batches then release intrinsics.
  bool single_board_dense_grid_profile = false;
  // One robust scale is shared by all dense-grid control points. The default
  // matches BabelCalib's pixel-domain Huber IRLS threshold.
  double checkerboard_huber_delta_pixels = 1.5;
  bool checkerboard_outlier_filter_enabled = true;
  double checkerboard_outlier_sigma = 4.0;
  double checkerboard_min_inlier_ratio = 0.5;
  int checkerboard_min_retained_points = 8;
  SelectionMode selection_mode = SelectionMode::KalibrStyleBatch;
  BudgetMode budget_mode = BudgetMode::KalibrStyle;
  CandidateOrderMode candidate_order_mode = CandidateOrderMode::ScoreSorted;
  bool candidate_order_mode_explicit = false;
  InfoGainProxyMode info_gain_proxy_mode = InfoGainProxyMode::IntrinsicsJacobian;
  CandidateBatchGranularity candidate_batch_granularity =
      CandidateBatchGranularity::Frame;
  KalibrStyleBatchAcceptancePolicy acceptance_policy =
      KalibrStyleBatchAcceptancePolicy::KalibrInformationGain;
  double acceptance_information_gain_threshold = 0.2;
  bool acceptance_information_gain_threshold_explicit = false;
  double acceptance_rank_gain_threshold = 1e-6;
  bool candidate_shuffle_seed_set = false;
  unsigned int candidate_shuffle_seed = 0;
  bool incremental_acceptance = true;
  bool carry_accepted_trial_state = true;
  bool optimize_intrinsics_in_trial = true;
  // Kalibr-style camera initialization: one temporary pose per frame-board,
  // fixed board-local geometry, and camera-only state commit.
  bool independent_frame_board_camera_warmup = false;
  bool delayed_intrinsics_release_in_trial = true;
  int intrinsics_release_iteration = 1;
  bool persistent_intrinsics_anchor_prior_enabled = false;
  bool persistent_fix_board_layout = false;
  // Experimental model-aware coreset: use the active camera family's full
  // parameter vector and independent frame-board pose-marginalized Fisher
  // information for frame-level ordering/acceptance diagnostics.
  bool model_aware_information_coreset = false;
  double persistent_intrinsics_anchor_weight_xi_alpha = 0.0;
  double persistent_intrinsics_anchor_weight_focal = 0.0;
  double persistent_intrinsics_anchor_weight_principal = 0.0;
  double persistent_max_focal_relative_step = 0.0;
  double persistent_max_principal_step_px = 0.0;
  double persistent_max_xi_alpha_step = 0.0;
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
  bool force_include_list_is_exact_input = false;
  std::set<std::pair<int, int> > force_include_frame_board_keys;
  std::set<std::pair<std::string, int> > force_include_frame_label_board_keys;
  std::set<std::pair<int, int> > seed_override_frame_board_keys;
  std::set<std::pair<std::string, int> > seed_override_frame_label_board_keys;
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
  double model_aware_information_gain = 0.0;
  double model_aware_rank_gain = 0.0;
  double model_aware_weak_direction_gain = 0.0;
  double frame_completion_bonus = 0.0;
  double new_board_bonus = 0.0;
  double cap_penalty = 0.0;
  double information_gain_proxy = 0.0;
  double residual_overage_penalty = 0.0;
  double batch_acceptance_score = 0.0;
  bool accepted_by_batch_acceptance = false;
  bool persistent_incremental_attempted = false;
  int persistent_incremental_attempt_order = -1;
  bool persistent_incremental_batch_accepted = false;
  bool persistent_incremental_force = false;
  bool persistent_incremental_trust_region_pass = true;
  bool persistent_incremental_trust_region_backtracking_used = false;
  bool persistent_incremental_split_residual_health_pass = true;
  bool persistent_incremental_pixel_safety_gate_pass = true;
  bool persistent_incremental_full_training_pose_refit_health_pass = true;
  bool persistent_incremental_ray_curve_validity_pass = true;
  bool persistent_incremental_kb_k3_released = true;
  bool persistent_incremental_kb_k4_released = true;
  double persistent_incremental_information_gain = 0.0;
  double persistent_incremental_normalized_information_gain = 0.0;
  int persistent_incremental_information_gain_normalization_count = 1;
  int persistent_incremental_rank_psi_after = -1;
  int persistent_incremental_rank_psi_deficiency_after = -1;
  int persistent_incremental_rank_theta_before = -1;
  int persistent_incremental_rank_theta_after = -1;
  int persistent_incremental_rank_theta_deficiency_after = -1;
  double persistent_incremental_svd_tolerance = 0.0;
  double persistent_incremental_qr_tolerance = 0.0;
  int persistent_incremental_iterations = 0;
  int persistent_incremental_last_solver_pass_iterations = 0;
  int persistent_incremental_continuation_round_count = 0;
  bool persistent_incremental_continuation_guard_hit = false;
  bool persistent_incremental_pose_prefit_attempted = false;
  bool persistent_incremental_pose_prefit_success = false;
  int persistent_incremental_pose_prefit_iterations = 0;
  double persistent_incremental_pose_prefit_objective_start = 0.0;
  double persistent_incremental_pose_prefit_objective_final = 0.0;
  double persistent_incremental_pose_prefit_last_delta_j = 0.0;
  double persistent_incremental_pose_prefit_last_delta_x = 0.0;
  double persistent_incremental_objective_start = 0.0;
  double persistent_incremental_objective_final = 0.0;
  double persistent_incremental_objective_last_delta_j = 0.0;
  double persistent_incremental_state_last_delta_x = 0.0;
  bool persistent_incremental_linear_solver_failure = false;
  bool persistent_incremental_converged_by_relative_objective = false;
  bool persistent_incremental_converged_by_camera_step = false;
  double persistent_incremental_last_camera_shape_step = 0.0;
  double persistent_incremental_last_camera_focal_relative_step = 0.0;
  double persistent_incremental_last_camera_principal_step_px = 0.0;
  bool persistent_incremental_objective_decreased = false;
  double persistent_incremental_rmse_before = 0.0;
  double persistent_incremental_outer_rmse_before = 0.0;
  double persistent_incremental_internal_rmse_before = 0.0;
  std::string persistent_incremental_acceptance_metric_name;
  std::string persistent_incremental_acceptance_metric_unit;
  double persistent_incremental_acceptance_metric_threshold = 0.0;
  double persistent_incremental_acceptance_metric_before = 0.0;
  double persistent_incremental_acceptance_metric_after = 0.0;
  double persistent_incremental_acceptance_metric_candidate = 0.0;
  double persistent_incremental_acceptance_metric_candidate_p95 = 0.0;
  double persistent_incremental_acceptance_metric_candidate_outer = 0.0;
  double persistent_incremental_acceptance_metric_candidate_internal = 0.0;
  double persistent_incremental_total_p95_after = 0.0;
  double persistent_incremental_outer_p95_after = 0.0;
  double persistent_incremental_internal_p95_after = 0.0;
  double persistent_incremental_candidate_rmse_after = 0.0;
  double persistent_incremental_candidate_outer_rmse_after = 0.0;
  double persistent_incremental_candidate_internal_rmse_after = 0.0;
  double persistent_incremental_candidate_total_p95_after = 0.0;
  double persistent_incremental_candidate_outer_p95_after = 0.0;
  double persistent_incremental_candidate_internal_p95_after = 0.0;
  double persistent_incremental_pixel_rmse_before = 0.0;
  double persistent_incremental_pixel_rmse_after = 0.0;
  double persistent_incremental_pixel_p95_before = 0.0;
  double persistent_incremental_pixel_p95_after = 0.0;
  double persistent_incremental_candidate_pixel_rmse_after = 0.0;
  double persistent_incremental_candidate_pixel_p95_after = 0.0;
  double persistent_incremental_full_training_pixel_rmse_before = 0.0;
  double persistent_incremental_full_training_pixel_rmse_after = 0.0;
  double persistent_incremental_full_training_pixel_p95_before = 0.0;
  double persistent_incremental_full_training_pixel_p95_after = 0.0;
  double persistent_incremental_full_training_pose_success_rate_before = 0.0;
  double persistent_incremental_full_training_pose_success_rate_after = 0.0;
  int persistent_incremental_full_training_pose_success_count_before = 0;
  int persistent_incremental_full_training_pose_success_count_after = 0;
  int persistent_incremental_full_training_pose_total_count = 0;
  int persistent_incremental_full_training_invalid_projection_count_before = 0;
  int persistent_incremental_full_training_invalid_projection_count_after = 0;
  double persistent_incremental_ray_curve_rms_change_deg = 0.0;
  double persistent_incremental_ray_curve_max_change_deg = 0.0;
  double persistent_incremental_ray_curve_min_radial_derivative = 0.0;
  int persistent_incremental_image_plane_residual_count = 0;
  int persistent_incremental_angular_residual_count = 0;
  int persistent_incremental_chordal_residual_count = 0;
  int persistent_incremental_hybrid_angular_selected_count = 0;
  int persistent_incremental_hybrid_chordal_selected_count = 0;
  int persistent_incremental_angular_geometry_failure_count = 0;
  int persistent_incremental_trust_region_retry_count = 0;
  double persistent_incremental_trust_region_violation_ratio = 1.0;
  double persistent_incremental_trust_region_anchor_weight_scale = 1.0;
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
  double persistent_incremental_camera_k1_before = 0.0;
  double persistent_incremental_camera_k2_before = 0.0;
  double persistent_incremental_camera_k3_before = 0.0;
  double persistent_incremental_camera_k4_before = 0.0;
  double persistent_incremental_camera_k1_after = 0.0;
  double persistent_incremental_camera_k2_after = 0.0;
  double persistent_incremental_camera_k3_after = 0.0;
  double persistent_incremental_camera_k4_after = 0.0;
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
  std::string selection_profile = "multi_board";
  bool selection_is_kalibr_checkerboard_style = false;
  double checkerboard_huber_delta_pixels = 0.0;
  bool checkerboard_force_all_valid_views = false;
  bool checkerboard_pose_marginalized_fisher = false;
  std::string checkerboard_seed_strategy;
  int checkerboard_seed_target_frame_count = 0;
  double checkerboard_seed_fisher_logdet = 0.0;
  double checkerboard_seed_fisher_rank_proxy = 0.0;
  bool checkerboard_outlier_filter_enabled = false;
  double checkerboard_outlier_sigma = 0.0;
  double checkerboard_min_inlier_ratio = 0.0;
  int checkerboard_min_retained_points = 0;
  double checkerboard_outlier_threshold_pixels = 0.0;
  double checkerboard_outlier_median_pixels = 0.0;
  double checkerboard_outlier_robust_sigma_pixels = 0.0;
  int checkerboard_outlier_removed_point_count = 0;
  int checkerboard_outlier_dropped_view_count = 0;
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
  bool model_aware_information_coreset_enabled = false;
  int model_aware_parameter_dimension = 0;
  std::vector<std::string> model_aware_parameter_labels;
  int model_aware_frame_count_scanned = 0;
  int model_aware_frame_count_accepted = 0;
  double model_aware_max_remaining_information_gain = 0.0;
  double model_aware_seed_fisher_rank = 0.0;
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
  std::string persistent_incremental_information_gain_target;
  bool persistent_incremental_board_layout_in_information_group = false;
  bool persistent_incremental_board_layout_fixed = false;
  int persistent_incremental_board_layout_pose_count = 0;
  double persistent_incremental_board_layout_max_matrix_abs_delta = 0.0;
  double persistent_incremental_board_layout_max_translation_delta = 0.0;
  double persistent_incremental_board_layout_max_rotation_delta_deg = 0.0;
  int persistent_incremental_camera_information_group_id = -1;
  int persistent_incremental_board_layout_group_id = -1;
  int persistent_incremental_transformation_group_id = -1;
  int persistent_incremental_seed_information_group_dim = -1;
  int persistent_incremental_seed_information_rank = -1;
  int persistent_incremental_seed_information_rank_deficiency = -1;
  bool persistent_incremental_seed_information_baseline_valid = false;
  double persistent_incremental_seed_information_scaled_min_singular_value = -1.0;
  double persistent_incremental_seed_information_scaled_max_singular_value = -1.0;
  double persistent_incremental_seed_information_scaled_condition_number = -1.0;
  double persistent_incremental_seed_information_ds_cu_stddev_px = -1.0;
  double persistent_incremental_seed_information_ds_cv_stddev_px = -1.0;
  int persistent_incremental_seed_batch_count = 0;
  int persistent_incremental_seed_frame_count = 0;
  int persistent_incremental_seed_board_observation_count = 0;
  int persistent_incremental_seed_point_count = 0;
  bool persistent_incremental_seed_outer_only_residuals = false;
  bool persistent_independent_camera_warmup_requested = false;
  bool persistent_independent_camera_warmup_attempted = false;
  bool persistent_independent_camera_warmup_success = false;
  bool persistent_independent_camera_warmup_committed = false;
  bool persistent_independent_camera_warmup_health_pass = false;
  int persistent_independent_camera_warmup_pose_count = 0;
  int persistent_independent_camera_warmup_point_count = 0;
  int persistent_independent_camera_warmup_iterations = 0;
  double persistent_independent_camera_warmup_objective_start = 0.0;
  double persistent_independent_camera_warmup_objective_final = 0.0;
  double persistent_independent_camera_warmup_rmse_before = 0.0;
  double persistent_independent_camera_warmup_rmse_after = 0.0;
  double persistent_independent_camera_warmup_p95_before = 0.0;
  double persistent_independent_camera_warmup_p95_after = 0.0;
  int persistent_independent_camera_warmup_seed_quarantined_count = 0;
  int persistent_independent_camera_warmup_instability_quarantined_count = 0;
  bool persistent_independent_camera_warmup_quarantine_retry_attempted = false;
  bool persistent_independent_camera_warmup_quarantine_retry_success = false;
  std::string persistent_independent_camera_warmup_quarantine_reason;
  std::string persistent_independent_camera_warmup_rollback_reason;
  bool persistent_incremental_seed_intrinsics_warmup_attempted = false;
  bool persistent_incremental_seed_intrinsics_warmup_success = false;
  bool persistent_incremental_seed_intrinsics_warmup_converged_by_relative_objective = false;
  int persistent_incremental_seed_intrinsics_warmup_iterations = 0;
  double persistent_incremental_seed_intrinsics_warmup_objective_start = 0.0;
  double persistent_incremental_seed_intrinsics_warmup_objective_final = 0.0;
  double persistent_incremental_seed_intrinsics_warmup_last_delta_j = 0.0;
  double persistent_incremental_seed_intrinsics_warmup_last_delta_x = 0.0;
  int persistent_incremental_candidate_batch_count = 0;
  int persistent_incremental_attempted_batch_count = 0;
  int persistent_incremental_accepted_batch_count = 0;
  int persistent_incremental_rejected_batch_count = 0;
  std::map<std::string, int> persistent_incremental_rejection_reason_counts;
  std::map<std::string, int>
      persistent_incremental_rejection_reason_code_counts;
  std::string persistent_incremental_solver_profile_name;
  std::string persistent_incremental_solver_objective_unit;
  int persistent_incremental_solver_max_iterations = 0;
  double persistent_incremental_solver_convergence_delta_j = 0.0;
  double persistent_incremental_solver_convergence_delta_x = 0.0;
  double persistent_incremental_solver_bearing_reference_focal_px = 1.0;
  double persistent_incremental_solver_bearing_residual_scale = 1.0;
  int persistent_incremental_solver_single_iteration_batch_count = 0;
  int persistent_incremental_solver_max_iteration_batch_count = 0;
  int persistent_incremental_solver_objective_decreased_batch_count = 0;
  int persistent_incremental_solver_relative_objective_converged_batch_count = 0;
  int persistent_incremental_solver_camera_step_converged_batch_count = 0;
  int persistent_incremental_solver_continuation_batch_count = 0;
  int persistent_incremental_solver_continuation_round_count = 0;
  int persistent_incremental_solver_continuation_guard_hit_count = 0;
  int persistent_incremental_image_plane_residual_count = 0;
  int persistent_incremental_angular_residual_count = 0;
  int persistent_incremental_chordal_residual_count = 0;
  int persistent_incremental_hybrid_angular_selected_count = 0;
  int persistent_incremental_hybrid_chordal_selected_count = 0;
  int persistent_incremental_angular_geometry_failure_count = 0;
  int persistent_incremental_angular_local_whitening_success_count = 0;
  int persistent_incremental_angular_local_whitening_failure_count = 0;
  int persistent_incremental_angular_local_whitening_clamped_count = 0;
  double persistent_incremental_angular_local_whitening_sigma_mean_rad = 0.0;
  double persistent_incremental_angular_local_whitening_sigma_min_rad = 0.0;
  double persistent_incremental_angular_local_whitening_sigma_max_rad = 0.0;
  double persistent_incremental_angular_local_whitening_weight_mean = 0.0;
  double persistent_incremental_angular_local_whitening_weight_min = 0.0;
  double persistent_incremental_angular_local_whitening_weight_max = 0.0;
  std::string persistent_incremental_selection_metric_name;
  std::string persistent_incremental_selection_metric_unit;
  std::string persistent_incremental_residual_health_threshold_source;
  double persistent_incremental_residual_health_threshold_metric = 0.0;
  double persistent_incremental_seed_acceptance_metric_rmse = 0.0;
  double persistent_incremental_seed_acceptance_metric_p95 = 0.0;
  int persistent_incremental_trust_region_backtracking_batch_count = 0;
  int persistent_incremental_trust_region_backtracking_attempt_count = 0;
  int persistent_incremental_trust_region_backtracking_accepted_count = 0;
  double persistent_incremental_trust_region_backtracking_max_anchor_scale = 1.0;
  bool persistent_incremental_normalize_information_gain_by_board_observation =
      false;
  bool persistent_incremental_split_residual_health_gate_enabled = false;
  int persistent_incremental_split_residual_health_rejected_count = 0;
  bool persistent_incremental_bearing_pixel_safety_gate_enabled = false;
  int persistent_incremental_bearing_pixel_safety_rejected_count = 0;
  bool persistent_incremental_full_training_pose_refit_health_gate_enabled =
      false;
  bool persistent_incremental_seed_intrinsics_warmup_full_training_health_pass =
      true;
  int persistent_incremental_full_training_pose_refit_health_rejected_count = 0;
  double persistent_incremental_initial_full_training_pixel_rmse = 0.0;
  double persistent_incremental_initial_full_training_pixel_p95 = 0.0;
  double persistent_incremental_initial_full_training_pose_success_rate = 0.0;
  int persistent_incremental_initial_full_training_pose_success_count = 0;
  int persistent_incremental_initial_full_training_pose_total_count = 0;
  int persistent_incremental_initial_full_training_invalid_projection_count = 0;
  int persistent_incremental_initial_full_training_invalid_outer_projection_count = 0;
  int persistent_incremental_initial_full_training_invalid_internal_projection_count = 0;
  double persistent_incremental_final_full_training_pixel_rmse = 0.0;
  double persistent_incremental_final_full_training_pixel_p95 = 0.0;
  double persistent_incremental_final_full_training_pose_success_rate = 0.0;
  int persistent_incremental_final_full_training_pose_success_count = 0;
  int persistent_incremental_final_full_training_pose_total_count = 0;
  int persistent_incremental_final_full_training_invalid_projection_count = 0;
  int persistent_incremental_final_full_training_invalid_outer_projection_count = 0;
  int persistent_incremental_final_full_training_invalid_internal_projection_count = 0;
  bool persistent_incremental_curated_bundle_state_consistency_pass = true;
  bool persistent_incremental_curated_bundle_shared_scene_health_pass = true;
  bool persistent_incremental_curated_bundle_used_validated_baseline_fallback =
      false;
  double persistent_incremental_committed_state_pixel_rmse = 0.0;
  double persistent_incremental_curated_bundle_pixel_rmse = 0.0;
  double persistent_incremental_curated_bundle_state_consistency_tolerance_px =
      0.0;
  double persistent_incremental_validated_baseline_pixel_rmse = 0.0;
  double persistent_incremental_curated_bundle_shared_scene_rmse_limit_px =
      0.0;
  bool persistent_incremental_kb_distortion_guard_enabled = false;
  int persistent_incremental_kb_ray_curve_validity_rejected_count = 0;
  bool persistent_incremental_adaptive_saturation_stop_enabled = false;
  bool persistent_incremental_adaptive_saturation_stop_hit = false;
  int persistent_incremental_adaptive_saturation_min_accepted_batches = 0;
  int persistent_incremental_adaptive_saturation_nonproductive_batch_limit = 0;
  int persistent_incremental_adaptive_saturation_consecutive_nonproductive_batches = 0;
  double persistent_incremental_adaptive_saturation_tail_ordering_score_threshold = 0.0;
  double persistent_incremental_adaptive_saturation_next_ordering_score = 0.0;
  std::string persistent_incremental_adaptive_saturation_stop_reason;
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
  std::string our_camera_source;
  OuterBootstrapCameraIntrinsics our_camera;
  std::vector<CameraRayCurveSample> samples;
  std::vector<CameraRayCurveBucketSummary> bucket_summaries;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct Stage5LargeIntrinsicPerturbationState {
  bool enabled = false;
  std::string requested_profile;
  std::string effective_profile;
  bool valid_projection_grid = false;
  int projection_grid_width = 0;
  int projection_grid_height = 0;
  int valid_projection_grid_count = 0;
  int invalid_projection_grid_count = 0;
  double requested_focal_scale = 1.0;
  double requested_xi_delta = 0.0;
  double requested_alpha_delta = 0.0;
  double requested_scale = 1.0;
  double effective_scale = 1.0;
  double actual_focal_scale = 1.0;
  double actual_xi_delta = 0.0;
  double actual_alpha_delta = 0.0;
  std::string reference_scene_fingerprint;
  std::string perturbed_scene_fingerprint;
  std::string frozen_observation_fingerprint;
  CalibrationSceneState reference_scene;
  OuterBootstrapCameraIntrinsics reference_camera;
  OuterBootstrapCameraIntrinsics perturbed_camera;
  OuterBootstrapCameraIntrinsics selection_seed_camera;
  OuterBootstrapCameraIntrinsics selection_candidate_camera;
  bool selection_seed_matches_perturbed_camera = false;
  bool selection_candidate_matches_perturbed_camera = false;
  bool outer_only_after_application = false;
  int frozen_internal_point_count_before_ablation = 0;
  int seed_internal_point_count_after_ablation = 0;
  int candidate_pool_internal_point_count_after_ablation = 0;
  std::string failure_reason;
};

struct Stage5BenchmarkInput {
  std::vector<FrozenRound2BaselineFrameSource> all_frames;
  std::vector<FrozenRound2BaselineFrameSource> external_holdout_frames;
  // Runs the complete image frontend on all_frames but intentionally does not
  // build a backend problem, select observations, or optimize a backend.
  bool frontend_only = false;
  bool use_precomputed_training_measurements = false;
  FrozenPrecomputedMeasurementInput precomputed_training_measurements;
  bool use_precomputed_holdout_measurements = false;
  FrozenPrecomputedMeasurementInput precomputed_holdout_measurements;
  std::string external_holdout_label;
  bool use_external_holdout_self_frontend_prepass = false;
  // Valid only when holdout frames exactly match the training frames. Reuse
  // the full frontend observations for a same-sequence evaluation, rather
  // than changing the observed board set through a second detector pass.
  bool holdout_evaluate_full_training_observations = false;
  FrozenRound2BaselineOptions baseline_options;
  BackendProblemOptions backend_options;
  BackendProblemOptions committed_backend_evaluation_options;
  AslamBackendCalibrationOptions selection_backend_runner_options;
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
  bool enable_large_intrinsic_perturbation = false;
  std::string large_intrinsic_perturbation_profile;
  // Unit interval amount along the selected P1--P4 direction. A value of one
  // is the full perturbation and zero is the paired clean baseline.
  double large_intrinsic_perturbation_scale = 1.0;
  // Semi-synthetic experiments preserve the requested scale exactly,
  // including values above one, instead of backing it off.
  bool large_intrinsic_perturbation_strict_scale = false;
  std::string large_intrinsic_perturbation_reference_scene_path;
  // Perturbation-ablation only: retain the already recovered/frozen internal
  // measurements through bootstrap, then remove them immediately after the
  // perturbation and before selection/incremental BA.
  bool large_intrinsic_perturbation_outer_only_after_application = false;
};

struct PersistentCameraCheckpointEvaluation {
  int attempt_order = -1;
  int frame_index = -1;
  std::string frame_label;
  double information_gain = 0.0;
  OuterBootstrapCameraIntrinsics camera;
  CameraModelRefitEvaluationResult training_evaluation;
  CameraModelRefitEvaluationResult holdout_evaluation;
};

struct Stage5BenchmarkReport {
  bool success = false;
  bool fair_protocol_matched = false;
  bool diagnostic_only = false;
  bool external_holdout_self_frontend_prepass_used = false;
  bool external_holdout_self_frontend_prepass_success = false;
  std::string external_holdout_observation_source = "training_scene_regeneration";
  std::string stage5_input_mode = "images";
  std::string precomputed_training_source;
  std::string precomputed_holdout_source;
  std::string precomputed_target_mode_requested = "auto";
  std::string precomputed_target_mode_resolved;
  int precomputed_board_count = 0;
  bool precomputed_single_board_ba_mode = false;
  int precomputed_training_frame_count = 0;
  int precomputed_training_board_observation_count = 0;
  int precomputed_training_outer_point_count = 0;
  int precomputed_training_internal_point_count = 0;
  int precomputed_holdout_frame_count = 0;
  int precomputed_holdout_board_observation_count = 0;
  int precomputed_holdout_outer_point_count = 0;
  int precomputed_holdout_internal_point_count = 0;
  bool precomputed_boards_rt_used_to_initialize_layout = false;
  std::string external_holdout_self_frontend_prepass_failure_reason;
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
  Stage5LargeIntrinsicPerturbationState large_intrinsic_perturbation;
  bool final_backend_scene_available = false;
  CalibrationSceneState final_backend_scene;
  MultiBoardPoseOrientationEvaluationResult
      large_perturbation_pose_orientation_evaluation;
  CalibrationBackendProblemInput backend_problem_input;
  CalibrationEvaluationDataset training_dataset;
  CalibrationEvaluationDataset holdout_dataset;
  CameraModelRefitEvaluationResult our_training_evaluation;
  CameraModelRefitEvaluationResult kalibr_training_evaluation;
  CameraModelRefitEvaluationResult our_holdout_evaluation;
  CameraModelRefitEvaluationResult kalibr_holdout_evaluation;
  CameraModelRefitEvaluationResult initialization_training_evaluation;
  CameraModelRefitEvaluationResult initialization_holdout_evaluation;
  CameraModelRefitEvaluationResult perturbation_boundary_training_evaluation;
  CameraModelRefitEvaluationResult perturbation_boundary_holdout_evaluation;
  std::vector<PersistentCameraCheckpointEvaluation>
      persistent_camera_checkpoint_evaluations;
  bool checkerboard_robust_checkpoint_selection_used = false;
  std::string checkerboard_robust_checkpoint_criterion;
  std::string checkerboard_robust_checkpoint_label;
  int checkerboard_robust_checkpoint_attempt_order = -1;
  double checkerboard_robust_checkpoint_frame_median_rmse = 0.0;
  double checkerboard_robust_checkpoint_frame_p90_rmse = 0.0;
  double checkerboard_robust_checkpoint_huber_rmse = 0.0;
  double checkerboard_robust_checkpoint_fold_median_mean_rmse = 0.0;
  double checkerboard_robust_checkpoint_fold_median_max_rmse = 0.0;
  double checkerboard_robust_checkpoint_fold_median_std_rmse = 0.0;
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
  MultiBoardPoseOrientationEvaluationResult
  EvaluateMultiBoardPoseOrientation(
      const CalibrationEvaluationDataset& dataset,
      const CalibrationSceneState& final_scene,
      const CalibrationSceneState& ground_truth_scene) const;
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
      OuterDetectionCacheStats* cache_stats,
      InternalRegenerationCacheStats* internal_cache_stats = nullptr) const;

 private:
  CalibrationEvaluationDataset BuildTrainingEvaluationDataset(
      const CalibrationStateBundle& bundle) const;
  CalibrationEvaluationDataset BuildEvaluationDatasetFromMeasurementResult(
      const JointMeasurementBuildResult& measurement_result,
      const std::vector<InternalRegenerationFrameResult>& regeneration_results,
      const std::string& dataset_label,
      const std::string& split_label,
      const std::string& split_signature,
      bool include_points_not_used_in_solver = false,
      bool require_internal_solver_support = false) const;
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
void WritePersistentCameraCheckpointEvaluationsCsv(
    const std::string& path,
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
void WriteStage5BenchmarkHoldoutRobustOutlierSummary(
    const std::string& path,
    const Stage5BenchmarkReport& report,
    double board_outlier_rmse_threshold_px = 5.0);
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
