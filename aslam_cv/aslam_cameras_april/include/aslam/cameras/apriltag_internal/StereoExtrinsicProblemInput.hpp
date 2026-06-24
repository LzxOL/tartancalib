#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STEREO_EXTRINSIC_PROBLEM_INPUT_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STEREO_EXTRINSIC_PROBLEM_INPUT_HPP

#include <array>
#include <map>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionMeasurementBuilder.hpp>
#include <aslam/cameras/apriltag_internal/KalibrStyleBatchAcceptance.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

enum class StereoPairPoseRefitMode {
  Cam0Only = 0,
  StereoSymmetric = 1,
};

enum class StereoViewSelectionMode {
  Off = 0,
  TopK = 1,
  KalibrStyleTrial = 2,
};

enum class StereoSolverMode {
  Alternating = 0,
  GlobalSparseBa = 1,
  SharedOnlyGlobalSparseBa = 2,
};

enum class StereoSingleCameraOnlyWeightMode {
  FixedScale = 0,
  PerSideBudgetCap = 1,
  AdaptiveIndependentSideCap = 2,
};

enum class StereoMeasurementSourceMode {
  BackendSelectedOnly = 0,
  AllValid = 1,
};

enum class StereoSingleBoardPairPolicy {
  Keep = 0,
  Audit = 1,
  Drop = 2,
  LowWeight = 3,
};

enum class StereoPairBoardSelectionMode {
  StrictRmse = 0,
  KalibrStyleBatch = 1,
};

enum class StereoFinalBaResidualMode {
  Pixel = 0,
  SphericalChordal = 1,
  SphericalTangent = 2,
  HybridPixelSpherical = 3,
};

enum class StereoSphericalUncertaintyMode {
  None = 0,
  Pixel = 1,
  Model = 2,
  PixelModel = 3,
};

enum class StereoCandidateBudgetMode {
  Fixed = 0,
  Adaptive = 1,
  KalibrStyle = 2,
};

enum class Stage6IncrementalInfoBlock {
  StereoExtrinsic = 0,
};

enum class StereoSelectionOptimizationMode {
  TrialBaCommit = 0,
  TrialBaNoCommit = 1,
  InformationGainOnly = 2,
};

enum class StereoRigParamMode {
  Cam0Reference = 0,
  RigCentricSymmetric = 1,
};

struct StereoFramePair {
  int pair_index = -1;
  int left_frame_index = -1;
  int right_frame_index = -1;
  std::string left_frame_label;
  std::string right_frame_label;
  std::string left_image_path;
  std::string right_image_path;
  bool is_training = true;
};

struct StereoCameraFixedCalibration {
  std::string camera_model_family;
  std::string camera_model;
  std::string distortion_model;
  std::vector<double> intrinsics;
  std::vector<double> distortion_coeffs;
  std::vector<int> resolution;
  std::string source_yaml_path;

  bool IsValid() const {
    return !camera_model_family.empty() &&
           !camera_model.empty() &&
           resolution.size() == 2 &&
           resolution[0] > 0 &&
           resolution[1] > 0 &&
           !intrinsics.empty();
  }
};

struct StereoObservation {
  int camera_index = -1;
  int pair_index = -1;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  JointPointType point_type = JointPointType::Outer;
  Eigen::Vector3d target_point_board = Eigen::Vector3d::Zero();
  Eigen::Vector2d observed_image_xy = Eigen::Vector2d::Zero();
  double weight = 1.0;
  double quality = 0.0;
  bool used_in_solver = false;
  int outer_subpix_window_radius = 0;
  int outer_pre_boost_subpix_window_radius = 0;
  int outer_boosted_raw_subpix_window_radius = 0;
  bool outer_close_edge_subpix_boost_applied = false;
  double outer_close_edge_subpix_area_ratio = 0.0;
  double outer_close_edge_subpix_max_polar_deg = 0.0;
};

struct StereoMeasurementDataset {
  bool success = false;
  int reference_board_id = 1;
  std::vector<StereoFramePair> frame_pairs;
  std::vector<StereoObservation> observations;
  std::vector<int> training_pair_indices;
  std::vector<int> holdout_pair_indices;
  int left_frame_count = 0;
  int right_frame_count = 0;
  int paired_frame_count = 0;
  int unmatched_left_count = 0;
  int unmatched_right_count = 0;
  std::string pairing_mode;
  long long max_pair_timestamp_delta_ns = 0;
  double mean_abs_pair_timestamp_delta_ms = 0.0;
  double max_abs_pair_timestamp_delta_ms = 0.0;
  int shared_board_observation_count = 0;
  int cam0_only_board_observation_count = 0;
  int cam1_only_board_observation_count = 0;
  std::map<int, int> pair_shared_board_count;
  std::map<int, int> pair_cam0_only_board_count;
  std::map<int, int> pair_cam1_only_board_count;
  std::map<int, std::set<int> > pair_shared_board_ids;
  std::map<int, std::set<int> > pair_cam0_only_board_ids;
  std::map<int, std::set<int> > pair_cam1_only_board_ids;
  std::map<int, std::set<int> > training_pair_board_ids;
  std::map<int, std::set<int> > holdout_pair_board_ids;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct StereoSceneState {
  bool success = false;
  bool cam0_is_reference = true;
  int gauge_fixed_board_id = 1;
  StereoCameraFixedCalibration cam0;
  StereoCameraFixedCalibration cam1;
  Eigen::Matrix4d T_cam1_cam0 = Eigen::Matrix4d::Identity();
  std::map<int, Eigen::Matrix4d> T_cam0_world_by_pair;
  std::map<int, Eigen::Matrix4d> T_world_board_by_id;
  std::set<int> excluded_training_pair_indices;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct Stage6RuntimeSummary {
  std::string cache_dir;
  bool cache_enabled = false;
  double total_runtime_seconds = 0.0;
  double pairing_build_dataset_runtime_seconds = 0.0;
  double initialization_runtime_seconds = 0.0;
  double training_optimization_runtime_seconds = 0.0;
  double global_sparse_ba_runtime_seconds = 0.0;
  double holdout_evaluation_runtime_seconds = 0.0;
  int cam0_training_detection_cache_hits = 0;
  int cam0_training_detection_cache_misses = 0;
  int cam0_training_detection_cache_load_failures = 0;
  int cam0_training_detection_cache_store_failures = 0;
  int cam1_training_detection_cache_hits = 0;
  int cam1_training_detection_cache_misses = 0;
  int cam1_training_detection_cache_load_failures = 0;
  int cam1_training_detection_cache_store_failures = 0;
  int symmetric_refit_call_count = 0;
  int symmetric_refit_improved_count = 0;
  int symmetric_refit_fallback_count = 0;
  int max_graph_propagation_iterations = 50;
  int graph_propagation_iteration_count = 0;
  int graph_propagation_new_pair_count = 0;
  int graph_propagation_new_board_count = 0;
  bool graph_propagation_stopped_by_no_progress = false;
  bool graph_propagation_stopped_by_iteration_limit = false;
  int runtime_guard_trigger_count = 0;
};

struct StereoExtrinsicSolverOptions {
  int reference_board_id = 1;
  int max_iterations = 10;
  double convergence_threshold = 1e-4;
  StereoViewSelectionMode view_selection_mode = StereoViewSelectionMode::Off;
  int selected_pair_count = 0;
  StereoSolverMode solver_mode = StereoSolverMode::GlobalSparseBa;
  StereoFinalBaResidualMode final_ba_residual_mode =
      StereoFinalBaResidualMode::Pixel;
  StereoFinalBaResidualMode selection_ba_residual_mode =
      StereoFinalBaResidualMode::Pixel;
  bool fixed_intrinsics_for_spherical = true;
  double spherical_weight = 1.0;
  bool spherical_polar_weighting = false;
  double spherical_min_polar_deg = 50.0;
  double spherical_max_weight = 4.0;
  StereoSphericalUncertaintyMode spherical_uncertainty_mode =
      StereoSphericalUncertaintyMode::None;
  double spherical_pixel_sigma_px = 0.5;
  std::array<double, 6> spherical_model_sigma{{0.02, 0.02, 2.0, 2.0, 2.0, 2.0}};
  double spherical_covariance_damping = 1e-8;
  double spherical_min_sigma_rad = 1e-5;
  double spherical_max_whitening_weight = 1e5;
  bool spherical_use_normalize_jacobian = false;
  StereoRigParamMode rig_param_mode = StereoRigParamMode::Cam0Reference;
  double rig_camera_prior_translation_weight = 1e-6;
  double rig_camera_prior_rotation_weight = 1e-6;
  double rig_stereo_relative_prior_weight = 1e-6;
  bool coobs_enable = false;
  std::string coobs_output_dir;
  int coobs_min_corners_per_group = 12;
  double coobs_high_polar_threshold_deg = 50.0;
  double coobs_very_high_polar_threshold_deg = 70.0;
  bool coobs_enable_rescue_suggestions = true;
  double coobs_score_alpha_high_polar = 1.0;
  double coobs_score_beta_multiboard = 2.0;
  double coobs_score_gamma_balance = 1.0;
  double coobs_score_eta_conflict = 1.0;
  double coobs_rescue_min_high_polar_score = 8.0;
  double coobs_rescue_bad_conflict_threshold = 5.0;
  bool selection_coobs_factor_ba_enable = false;
  bool selection_coobs_factor_ba_apply_stereo_factor = true;
  bool selection_coobs_factor_ba_apply_layout_factor = false;
  double selection_coobs_factor_ba_stereo_weight = 0.1;
  double selection_coobs_factor_ba_layout_weight = 0.05;
  bool coobs_aware_acceptance_enable = false;
  double coobs_aware_acceptance_min_score = 1.0;
  double coobs_aware_acceptance_max_total_rmse_delta = 0.05;
  double coobs_aware_acceptance_max_camera_rmse_delta = 0.08;
  bool coobs_aware_acceptance_balance_guard_enable = false;
  double coobs_aware_acceptance_max_camera_delta_imbalance = 0.03;
  double coobs_aware_acceptance_max_camera_delta_ratio = 2.0;
  bool coobs_aware_acceptance_require_pair_completion = true;
  double coobs_aware_acceptance_stereo_rot_scale_deg = 1.0;
  double coobs_aware_acceptance_layout_rot_scale_deg = 1.0;
  double coobs_aware_acceptance_stereo_trans_scale_m = 0.01;
  double coobs_aware_acceptance_layout_trans_scale_m = 0.005;
  bool coobs_factor_ba_enable = false;
  bool coobs_factor_ba_run_experiment_matrix = true;
  std::string coobs_factor_ba_output_dir_suffix = "coobs_factor_ba";
  int coobs_factor_ba_min_corners_per_cam_board = 12;
  double coobs_factor_ba_max_local_pose_rmse = 3.0;
  double coobs_factor_ba_huber_delta = 1.0;
  std::vector<double> coobs_factor_ba_stereo_weights{
      0.005, 0.01, 0.03, 0.05};
  std::vector<double> coobs_factor_ba_layout_weights{
      0.001, 0.003, 0.005, 0.01, 0.03};
  std::vector<double> coobs_factor_ba_combined_stereo_weights{0.01, 0.03};
  std::vector<double> coobs_factor_ba_combined_layout_weights{
      0.003, 0.005, 0.01};
  std::vector<std::pair<int, int> > coobs_factor_ba_layout_selected_pairs{
      {2, 4}, {1, 4}, {2, 5}};
  bool coobs_factor_ba_apply_stereo_factor = false;
  bool coobs_factor_ba_apply_layout_factor = false;
  double coobs_factor_ba_current_stereo_weight = 0.0;
  double coobs_factor_ba_current_layout_weight = 0.0;
  bool final_ba_optimize_intrinsics = false;
  bool final_ba_optimize_stereo_extrinsic = true;
  bool final_ba_optimize_pair_poses = true;
  bool final_ba_optimize_board_poses = true;
  bool skip_final_global_ba = true;
  int ba_max_iterations = 20;
  double ba_convergence_threshold = 1e-6;
  double ba_shared_observation_weight_scale = 1.0;
  double ba_single_camera_only_observation_weight_scale = 1.0;
  StereoSingleCameraOnlyWeightMode ba_single_camera_only_weight_mode =
      StereoSingleCameraOnlyWeightMode::FixedScale;
  double ba_single_camera_only_base_scale = 1.0;
  double ba_single_camera_only_per_side_budget_ratio = 0.10;
  double ba_adaptive_single_camera_only_per_side_cap_ratio = 0.05;
  StereoPairPoseRefitMode pair_pose_refit_mode =
      StereoPairPoseRefitMode::StereoSymmetric;
  int symmetric_refit_max_iterations = 8;
  double symmetric_refit_step = 1e-3;
  double pose_fit_guard_threshold_px = 8.0;
  double candidate_consistency_max_rotation_deg = 8.0;
  double candidate_consistency_max_translation_m = 0.08;
  bool require_symmetric_pair_refit = true;
  int min_shared_boards_for_extrinsic_candidate = 1;
  int max_graph_propagation_iterations = 50;
  bool enable_shared_board_quality_gate = true;
  bool enable_shared_board_quality_hard_gate = false;
  double shared_board_quality_max_outer_rmse_px = 3.0;
  int shared_board_quality_min_outer_points_per_camera = 4;
  int shared_board_quality_min_good_shared_boards = 1;
  bool shared_board_quality_filter_final_ba = true;
  bool export_pair_board_consistency_audit = false;
  bool enable_pair_board_consistency_gate = false;
  double pair_board_consistency_local_good_max_outer_rmse_px = 3.0;
  double pair_board_consistency_global_bad_min_outer_rmse_px = 15.0;

  bool enable_pair_only_stereo_ba_init = true;
  int pair_init_max_iterations = 50;
  double pair_init_convergence_threshold = 1e-6;
  bool pair_init_use_huber_loss = true;

  bool enable_kalibr_style_pair_selection = true;
  bool enable_committing_pair_batch_selection = false;
  bool enable_stage6_incremental_estimator = false;
  bool enable_incremental_pair_diversity_rescue = false;
  int incremental_pair_diversity_rescue_min_boards = 1;
  double incremental_mi_tol = 0.2;
  double incremental_rank_threshold = 1e-6;
  Stage6IncrementalInfoBlock incremental_info_block =
      Stage6IncrementalInfoBlock::StereoExtrinsic;
  StereoSelectionOptimizationMode selection_optimization_mode =
      StereoSelectionOptimizationMode::TrialBaCommit;
  KalibrStyleBatchAcceptancePolicy batch_acceptance_policy =
      KalibrStyleBatchAcceptancePolicy::ResidualScore;
  int pair_selection_seed_count = 10;
  StereoCandidateBudgetMode pair_selection_budget_mode =
      StereoCandidateBudgetMode::KalibrStyle;
  int pair_selection_max_candidate_additions = 30;
  double pair_selection_adaptive_budget_ratio = 0.20;
  int pair_selection_adaptive_budget_min = 30;
  int pair_selection_adaptive_budget_max = 120;
  int pair_selection_runtime_safety_ceiling = 1000;
  int pair_selection_min_shared_boards = 1;
  double pair_selection_max_rmse_delta = 0.02;
  double pair_selection_max_camera_rmse_delta = 0.05;
  double pair_selection_max_baseline_rotation_delta_deg = 0.2;
  double pair_selection_max_baseline_translation_delta_m = 0.005;

  bool enable_pair_board_trial_selection = true;
  StereoPairBoardSelectionMode pairboard_selection_mode =
      StereoPairBoardSelectionMode::KalibrStyleBatch;
  StereoCandidateBudgetMode pair_board_selection_budget_mode =
      StereoCandidateBudgetMode::KalibrStyle;
  int pair_board_selection_seed_count = 50;
  int pair_board_selection_max_candidate_additions = 40;
  double pair_board_selection_adaptive_budget_ratio = 0.15;
  int pair_board_selection_adaptive_budget_min = 40;
  int pair_board_selection_adaptive_budget_max = 200;
  int pair_board_selection_runtime_safety_ceiling = 1500;
  double pair_board_selection_min_candidate_score = 20.0;
  double pair_board_selection_min_coverage_gain = 0.0;
  int pair_board_selection_max_accepted_per_pair = 4;
  int pair_board_selection_max_accepted_per_board = 24;
  double pair_board_selection_max_rmse_delta = 0.02;
  double pair_board_selection_max_camera_rmse_delta = 0.05;
  double pair_board_selection_max_baseline_rotation_delta_deg = 0.2;
  double pair_board_selection_max_baseline_translation_delta_m = 0.005;
  bool enable_pair_cohesion = true;
  int pair_cohesion_min_boards_per_pair = 2;
  int pair_cohesion_max_companions_per_pair = 0;
  bool pair_cohesion_relax_score_gate = true;
  bool pair_cohesion_relax_cap_gates = true;
  StereoSingleBoardPairPolicy single_board_pair_policy =
      StereoSingleBoardPairPolicy::Audit;
  std::vector<std::pair<int, int> > ablation_excluded_pair_boards;

  bool export_stereo_reprojection_visualizations = false;
  int stereo_visualization_top_k = 50;
  bool export_extrinsic_uncertainty_diagnostics = false;
  bool export_angular_fixedk_diagnostic = false;
  bool board_masking_use_local_board_pose_ba = false;
};

struct StereoPairSelectionRow {
  int pair_index = -1;
  bool reachable = false;
  bool initialized = false;
  bool eligible = false;
  bool selected = false;
  int shared_board_count = 0;
  int shared_outer_point_count = 0;
  double pose_fit_rmse = std::numeric_limits<double>::infinity();
  int single_camera_only_board_count = 0;
  int covered_board_count = 0;
  int missing_board_coverage_count = 0;
  int score_shared_board_count = 0;
  int score_shared_outer_point_count = 0;
  double score_pose_fit_rmse = std::numeric_limits<double>::infinity();
  int score_single_camera_only_board_count = 0;
  std::vector<int> covered_board_ids;
  std::string rejection_reason;
};

struct StereoPairSelectionSummary {
  bool success = false;
  StereoViewSelectionMode mode = StereoViewSelectionMode::Off;
  int requested_pair_count = 0;
  int eligible_pair_count = 0;
  int selected_pair_count = 0;
  int reachable_pair_count = 0;
  int initialized_pair_count = 0;
  int selected_shared_board_pair_count = 0;
  int selected_single_camera_only_pair_count = 0;
  int selected_covered_board_count = 0;
  double selected_pose_fit_rmse_min = std::numeric_limits<double>::infinity();
  double selected_pose_fit_rmse_median = std::numeric_limits<double>::infinity();
  double selected_pose_fit_rmse_max = std::numeric_limits<double>::infinity();
  std::set<int> selected_pair_indices;
  std::set<std::pair<int, int> > selected_pair_board_keys;
  std::set<int> covered_board_ids;
  std::vector<StereoPairSelectionRow> rows;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct StereoGlobalSparseBaSummary {
  bool success = false;
  StereoSolverMode solver_mode = StereoSolverMode::Alternating;
  StereoFinalBaResidualMode residual_mode = StereoFinalBaResidualMode::Pixel;
  bool optimize_intrinsics = false;
  bool optimize_stereo_extrinsic = true;
  bool optimize_pair_poses = true;
  bool optimize_board_poses = true;
  int eligible_pair_count = 0;
  int selected_pair_count = 0;
  int active_board_count = 0;
  int reprojection_error_count = 0;
  int shared_observation_count = 0;
  int cam0_only_observation_count = 0;
  int cam1_only_observation_count = 0;
  int max_iterations = 0;
  double convergence_threshold = 0.0;
  double shared_observation_weight_scale = 1.0;
  double single_camera_only_observation_weight_scale = 0.25;
  StereoSingleCameraOnlyWeightMode single_camera_only_weight_mode =
      StereoSingleCameraOnlyWeightMode::FixedScale;
  double single_camera_only_base_scale = 0.25;
  double single_camera_only_per_side_budget_ratio = 0.10;
  double shared_total_base_weight = 0.0;
  double cam0_only_total_base_weight = 0.0;
  double cam1_only_total_base_weight = 0.0;
  double per_side_budget_limit = 0.0;
  double adaptive_single_camera_only_per_side_cap_ratio = 0.05;
  double cam0_only_cap = 0.0;
  double cam1_only_cap = 0.0;
  double cam0_only_effective_scale = 0.0;
  double cam1_only_effective_scale = 0.0;
  bool cam0_only_budget_clamped = false;
  bool cam1_only_budget_clamped = false;
  double shared_observation_weight_sum = 0.0;
  double cam0_only_observation_weight_sum = 0.0;
  double cam1_only_observation_weight_sum = 0.0;
  double initial_selected_rmse = 0.0;
  double final_selected_rmse = 0.0;
  double initial_selected_cam0_rmse = 0.0;
  double initial_selected_cam1_rmse = 0.0;
  double final_selected_cam0_rmse = 0.0;
  double final_selected_cam1_rmse = 0.0;
  double objective_start = 0.0;
  double objective_final = 0.0;
  int invalid_spherical_unprojection_count = 0;
  double spherical_weight = 1.0;
  bool spherical_polar_weighting = false;
  double spherical_min_polar_deg = 50.0;
  double spherical_max_weight = 4.0;
  StereoSphericalUncertaintyMode spherical_uncertainty_mode =
      StereoSphericalUncertaintyMode::None;
  double spherical_pixel_sigma_px = 0.5;
  std::array<double, 6> spherical_model_sigma{{0.02, 0.02, 2.0, 2.0, 2.0, 2.0}};
  double spherical_covariance_damping = 1e-8;
  double spherical_min_sigma_rad = 1e-5;
  double spherical_max_whitening_weight = 1e5;
  bool spherical_use_normalize_jacobian = false;
  int spherical_covariance_valid_count = 0;
  int spherical_covariance_invalid_count = 0;
  int spherical_covariance_damped_count = 0;
  int spherical_whitening_clamped_count = 0;
  double spherical_tangent_sigma_mean_rad = 0.0;
  double spherical_tangent_sigma_min_rad = 0.0;
  double spherical_tangent_sigma_max_rad = 0.0;
  double spherical_whitening_weight_mean = 0.0;
  double spherical_whitening_weight_min = 0.0;
  double spherical_whitening_weight_max = 0.0;
  StereoRigParamMode rig_param_mode = StereoRigParamMode::Cam0Reference;
  double rig_camera_prior_translation_weight = 0.0;
  double rig_camera_prior_rotation_weight = 0.0;
  double rig_stereo_relative_prior_weight = 0.0;
  double rig_projection_equivalence_max_pixel_diff = 0.0;
  double rig_projection_equivalence_max_angular_diff_rad = 0.0;
  double rig_stereo_relative_rotation_drift_deg = 0.0;
  double rig_stereo_relative_translation_drift_m = 0.0;
  int coobs_stereo_factor_count = 0;
  int coobs_layout_factor_count = 0;
  double coobs_stereo_factor_weight = 0.0;
  double coobs_layout_factor_weight = 0.0;
  double coobs_stereo_initial_rot_mean_deg = 0.0;
  double coobs_stereo_initial_rot_max_deg = 0.0;
  double coobs_stereo_initial_trans_mean_m = 0.0;
  double coobs_stereo_initial_trans_max_m = 0.0;
  double coobs_layout_initial_rot_mean_deg = 0.0;
  double coobs_layout_initial_rot_max_deg = 0.0;
  double coobs_layout_initial_trans_mean_m = 0.0;
  double coobs_layout_initial_trans_max_m = 0.0;
  bool board_masking_use_local_board_pose_ba = false;
  int iterations = 0;
  int failed_iterations = 0;
  bool linear_solver_failure = false;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct StereoExtrinsicProblemInput {
  std::string dataset_label;
  std::string left_image_path;
  std::string right_image_path;
  std::string left_config_path;
  std::string right_config_path;
  std::string left_intrinsics_path;
  std::string right_intrinsics_path;
  std::string split_signature;
  StereoMeasurementDataset measurement_dataset;
  StereoSceneState initial_scene;
  StereoExtrinsicSolverOptions solver_options;
  CalibrationStateBundle left_bundle;
  CalibrationStateBundle right_bundle;
};

struct StereoPairResidualSummary {
  int pair_index = -1;
  std::string left_frame_label;
  std::string right_frame_label;
  bool is_training = true;
  bool pose_refit_success = false;
  bool used_symmetric_refit = false;
  bool refit_fell_back_to_seed = false;
  bool used_in_metrics = false;
  int shared_board_count = 0;
  int cam0_only_board_count = 0;
  int cam1_only_board_count = 0;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  int cam0_point_count = 0;
  int cam1_point_count = 0;
  int shared_point_count = 0;
  int shared_outer_point_count = 0;
  int shared_internal_point_count = 0;
  double overall_rmse = 0.0;
  double cam0_rmse = 0.0;
  double cam1_rmse = 0.0;
  double outer_rmse = 0.0;
  double internal_rmse = 0.0;
  double shared_cam0_rmse = 0.0;
  double shared_cam1_rmse = 0.0;
  double shared_outer_rmse = 0.0;
  double shared_internal_rmse = 0.0;
  double cam0_only_rmse = 0.0;
  double cam1_only_rmse = 0.0;
  double mean_residual_x = 0.0;
  double mean_residual_y = 0.0;
  double std_residual_x = 0.0;
  double std_residual_y = 0.0;
  std::string pose_source;
  std::string failure_reason;
};

struct StereoBoardResidualSummary {
  int board_id = -1;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  int observation_count = 0;
  int cam0_point_count = 0;
  int cam1_point_count = 0;
  int shared_pair_count = 0;
  int shared_point_count = 0;
  int shared_cam0_point_count = 0;
  int shared_cam1_point_count = 0;
  int shared_outer_point_count = 0;
  int shared_internal_point_count = 0;
  double rmse = 0.0;
  double shared_cam0_rmse = 0.0;
  double shared_cam1_rmse = 0.0;
  double shared_outer_rmse = 0.0;
  double shared_internal_rmse = 0.0;
};

struct StereoCameraResidualSummary {
  int camera_index = -1;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double rmse = 0.0;
  int shared_point_count = 0;
  double shared_rmse = 0.0;
  int cam0_only_point_count = 0;
  double cam0_only_rmse = 0.0;
  int cam1_only_point_count = 0;
  double cam1_only_rmse = 0.0;
};

struct StereoResidualSummary {
  bool success = false;
  std::string split_label;
  bool holdout_refit = false;
  int pair_count = 0;
  int used_pair_count = 0;
  int unevaluable_pair_count = 0;
  int shared_board_pair_count = 0;
  int single_camera_only_pair_count = 0;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double total_stereo_rmse = 0.0;
  double cam0_rmse = 0.0;
  double cam1_rmse = 0.0;
  double cam1_over_cam0_rmse_ratio = 0.0;
  double cam_residual_balance_gap = 0.0;
  double outer_only_rmse = 0.0;
  double internal_only_rmse = 0.0;
  int shared_point_count = 0;
  int shared_outer_point_count = 0;
  int shared_internal_point_count = 0;
  double shared_total_rmse = 0.0;
  double shared_cam0_rmse = 0.0;
  double shared_cam1_rmse = 0.0;
  int cam0_only_point_count = 0;
  int cam1_only_point_count = 0;
  double cam0_only_total_rmse = 0.0;
  double cam1_only_total_rmse = 0.0;
  double mean_residual_x = 0.0;
  double mean_residual_y = 0.0;
  double std_residual_x = 0.0;
  double std_residual_y = 0.0;
  std::vector<StereoPairResidualSummary> pair_summaries;
  std::vector<StereoBoardResidualSummary> board_summaries;
  std::vector<StereoCameraResidualSummary> camera_summaries;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct StereoInitializationDiagnostics {
  bool success = false;
  int candidate_count = 0;
  int excluded_candidate_count = 0;
  int pair_pose_candidate_count = 0;
  int board_pose_candidate_count = 0;
  int candidate_rejected_pose_fit_count = 0;
  int candidate_rejected_consistency_count = 0;
  int graph_seed_pair_count = 0;
  int reachable_training_pair_count = 0;
  int unreachable_training_pair_count = 0;
  int excluded_training_pair_count = 0;
  int connected_component_count = 0;
  int gauge_connected_component_id = -1;
  int gauge_connected_pair_count = 0;
  int gauge_connected_board_count = 0;
  int graph_propagation_iteration_count = 0;
  int graph_propagation_new_pair_count = 0;
  int graph_propagation_new_board_count = 0;
  bool graph_propagation_stopped_by_no_progress = false;
  bool graph_propagation_stopped_by_iteration_limit = false;
  int initialized_training_pair_count = 0;
  int initialized_board_count = 0;
  int uninitialized_board_count = 0;
  int uninitialized_training_pair_count = 0;
  double medoid_score = 0.0;
  std::map<int, int> pair_component_ids;
  std::map<int, int> board_component_ids;
  std::vector<int> reachable_training_pair_indices;
  std::vector<int> unreachable_training_pair_indices;
  std::vector<int> reachable_board_ids;
  std::vector<int> unreachable_board_ids;
  std::vector<std::string> excluded_candidate_reasons;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct StereoPairInitCandidateRow {
  int pair_index = -1;
  int board_id = -1;
  bool raw_candidate = false;
  bool consistency_accepted = false;
  double cam0_outer_rmse = std::numeric_limits<double>::infinity();
  double cam1_outer_rmse = std::numeric_limits<double>::infinity();
  int shared_outer_point_count = 0;
  Eigen::Matrix4d T_cam1_cam0_candidate = Eigen::Matrix4d::Identity();
  double candidate_baseline_length = 0.0;
  std::string reject_reason;
};

struct StereoPairInitResidualRow {
  int pair_index = -1;
  int board_id = -1;
  int shared_point_count = 0;
  double before_rmse = std::numeric_limits<double>::infinity();
  double after_rmse = std::numeric_limits<double>::infinity();
};

struct StereoPairOnlyBaInitSummary {
  bool enabled = false;
  bool success = false;
  int raw_candidate_count = 0;
  int consistency_filtered_candidate_count = 0;
  int consistency_rejected_candidate_count = 0;
  int failed_pair_board_count = 0;
  double medoid_baseline_length = 0.0;
  double pair_ba_baseline_length = 0.0;
  double before_shared_rmse = std::numeric_limits<double>::infinity();
  double after_shared_rmse = std::numeric_limits<double>::infinity();
  double baseline_rotation_delta_deg = 0.0;
  double baseline_translation_delta_m = 0.0;
  bool used_refined_baseline = false;
  std::vector<StereoPairInitCandidateRow> candidates;
  std::vector<StereoPairInitResidualRow> residual_rows;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct StereoPairTrialSelectionDecision {
  int pair_index = -1;
  std::string left_frame_label;
  std::string right_frame_label;
  int shared_board_count = 0;
  int cam0_only_board_count = 0;
  int cam1_only_board_count = 0;
  int shared_outer_point_count = 0;
  int shared_internal_point_count = 0;
  double candidate_score = 0.0;
  double coverage_gain = 0.0;
  bool seed = false;
  bool attempted = false;
  bool accepted = false;
  double initial_total_rmse = std::numeric_limits<double>::infinity();
  double trial_total_rmse = std::numeric_limits<double>::infinity();
  double total_rmse_delta = std::numeric_limits<double>::infinity();
  double cam0_rmse_delta = std::numeric_limits<double>::infinity();
  double cam1_rmse_delta = std::numeric_limits<double>::infinity();
  double baseline_rotation_delta_deg = std::numeric_limits<double>::infinity();
  double baseline_translation_delta_m = std::numeric_limits<double>::infinity();
  bool incremental_estimator_enabled = false;
  std::string candidate_batch_type;
  bool batchAccepted = false;
  std::string accept_reason;
  bool solution_valid = false;
  bool optimization_success = false;
  int num_iterations = 0;
  double objective_before = std::numeric_limits<double>::infinity();
  double objective_after = std::numeric_limits<double>::infinity();
  double marginal_information_gain_proxy = 0.0;
  int rank_before = 0;
  int rank_after = 0;
  bool rank_proxy_increases = false;
  double info_gain_threshold = 0.0;
  std::string committed_or_rollback;
  std::string selection_optimization_mode;
  int trial_coobs_stereo_factor_count = 0;
  int trial_coobs_layout_factor_count = 0;
  double trial_coobs_stereo_initial_rot_mean_deg = 0.0;
  double trial_coobs_stereo_initial_rot_max_deg = 0.0;
  double trial_coobs_stereo_initial_trans_mean_m = 0.0;
  double trial_coobs_stereo_initial_trans_max_m = 0.0;
  double trial_coobs_layout_initial_rot_mean_deg = 0.0;
  double trial_coobs_layout_initial_rot_max_deg = 0.0;
  double trial_coobs_layout_initial_trans_mean_m = 0.0;
  double trial_coobs_layout_initial_trans_max_m = 0.0;
  double coobs_acceptance_score = 0.0;
  bool coobs_acceptance_health_pass = false;
  bool coobs_acceptance_structure_pass = false;
  bool coobs_acceptance_balance_pass = false;
  double coobs_acceptance_camera_delta_imbalance = 0.0;
  double coobs_acceptance_camera_delta_ratio = 0.0;
  bool accepted_by_coobs_aware_acceptance = false;
  bool force = false;
  std::string reject_reason;
};

struct StereoPairTrialSelectionSummary {
  bool enabled = false;
  bool success = false;
  int requested_seed_count = 0;
  int seed_count = 0;
  int candidate_count = 0;
  StereoCandidateBudgetMode budget_mode = StereoCandidateBudgetMode::KalibrStyle;
  int valid_candidate_count = 0;
  int valid_candidate_traversed_count = 0;
  bool safety_ceiling_hit = false;
  int runtime_safety_ceiling = 0;
  std::string max_candidate_additions_effective;
  int attempted_count = 0;
  int accepted_count = 0;
  int rejected_count = 0;
  int final_selected_pair_count = 0;
  double initial_seed_rmse = std::numeric_limits<double>::infinity();
  double final_selected_rmse = std::numeric_limits<double>::infinity();
  std::set<int> selected_pair_indices;
  std::vector<StereoPairTrialSelectionDecision> decisions;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct StereoPairBoardTrialSelectionDecision {
  int pair_index = -1;
  int board_id = -1;
  bool seed = false;
  bool pair_cohesion_candidate = false;
  bool pair_cohesion_cap_gate_relaxed = false;
  bool attempted = false;
  bool accepted = false;
  bool shared_board = false;
  int cam0_outer_point_count = 0;
  int cam1_outer_point_count = 0;
  int shared_point_count = 0;
  double cam0_outer_rmse = std::numeric_limits<double>::infinity();
  double cam1_outer_rmse = std::numeric_limits<double>::infinity();
  double candidate_score = 0.0;
  double coverage_gain = 0.0;
  int selected_pair_board_count_before = 0;
  int selected_board_count_before = 0;
  int selected_pair_count_before = 0;
  double initial_total_rmse = std::numeric_limits<double>::infinity();
  double trial_total_rmse = std::numeric_limits<double>::infinity();
  double total_rmse_delta = std::numeric_limits<double>::infinity();
  double cam0_rmse_delta = std::numeric_limits<double>::infinity();
  double cam1_rmse_delta = std::numeric_limits<double>::infinity();
  double baseline_rotation_delta_deg = std::numeric_limits<double>::infinity();
  double baseline_translation_delta_m = std::numeric_limits<double>::infinity();
  StereoPairBoardSelectionMode pairboard_selection_mode =
      StereoPairBoardSelectionMode::KalibrStyleBatch;
  bool hard_validity_pass = false;
  bool legacy_rmse_pass = false;
  bool catastrophic_residual = false;
  double score_term = 0.0;
  double coverage_term = 0.0;
  double pair_completion_bonus = 0.0;
  double new_board_bonus = 0.0;
  double cap_penalty = 0.0;
  double information_gain_proxy = 0.0;
  double residual_overage_penalty = 0.0;
  double batch_acceptance_score = 0.0;
  bool accepted_by_batch_acceptance = false;
  bool incremental_estimator_enabled = false;
  std::string candidate_batch_type;
  bool batchAccepted = false;
  std::string accept_reason;
  bool solution_valid = false;
  bool optimization_success = false;
  int num_iterations = 0;
  double objective_before = std::numeric_limits<double>::infinity();
  double objective_after = std::numeric_limits<double>::infinity();
  double marginal_information_gain_proxy = 0.0;
  int rank_before = 0;
  int rank_after = 0;
  bool rank_proxy_increases = false;
  double info_gain_threshold = 0.0;
  std::string committed_or_rollback;
  std::string selection_optimization_mode;
  int trial_coobs_stereo_factor_count = 0;
  int trial_coobs_layout_factor_count = 0;
  double trial_coobs_stereo_initial_rot_mean_deg = 0.0;
  double trial_coobs_stereo_initial_rot_max_deg = 0.0;
  double trial_coobs_stereo_initial_trans_mean_m = 0.0;
  double trial_coobs_stereo_initial_trans_max_m = 0.0;
  double trial_coobs_layout_initial_rot_mean_deg = 0.0;
  double trial_coobs_layout_initial_rot_max_deg = 0.0;
  double trial_coobs_layout_initial_trans_mean_m = 0.0;
  double trial_coobs_layout_initial_trans_max_m = 0.0;
  double coobs_acceptance_score = 0.0;
  bool coobs_acceptance_health_pass = false;
  bool coobs_acceptance_structure_pass = false;
  bool coobs_acceptance_balance_pass = false;
  double coobs_acceptance_camera_delta_imbalance = 0.0;
  double coobs_acceptance_camera_delta_ratio = 0.0;
  bool accepted_by_coobs_aware_acceptance = false;
  bool force = false;
  std::string reject_reason;
};

struct StereoPairBoardTrialSelectionSummary {
  bool enabled = false;
  bool success = false;
  int seed_count = 0;
  int candidate_count = 0;
  StereoCandidateBudgetMode budget_mode = StereoCandidateBudgetMode::KalibrStyle;
  int valid_candidate_count = 0;
  int valid_candidate_traversed_count = 0;
  bool safety_ceiling_hit = false;
  int runtime_safety_ceiling = 0;
  std::string max_candidate_additions_effective;
  int attempted_count = 0;
  int accepted_count = 0;
  int rejected_count = 0;
  StereoPairBoardSelectionMode pairboard_selection_mode =
      StereoPairBoardSelectionMode::KalibrStyleBatch;
  bool incremental_estimator_enabled = false;
  bool marginal_information_gain_proxy_enabled = false;
  bool rmse_delta_diagnostics_only = false;
  double incremental_mi_tol = 0.0;
  double incremental_rank_threshold = 0.0;
  std::string incremental_info_block;
  int batch_acceptance_attempted_count = 0;
  int batch_acceptance_accepted_count = 0;
  int batch_acceptance_rescued_from_legacy_rmse_gate_count = 0;
  int batch_acceptance_rejected_hard_validity_count = 0;
  int batch_acceptance_rejected_catastrophic_residual_count = 0;
  int batch_acceptance_rejected_score_count = 0;
  int pair_cohesion_candidate_count = 0;
  int pair_cohesion_attempted_count = 0;
  int pair_cohesion_accepted_count = 0;
  int pair_cohesion_rejected_count = 0;
  int pair_cohesion_auto_target_board_count = 0;
  int pair_cohesion_under_target_pair_count_before_rescue = 0;
  int pair_cohesion_under_target_pair_count_after_rescue = 0;
  int single_board_pair_count_before_rescue = 0;
  int single_board_pair_count_after_rescue = 0;
  int single_board_pair_count_after_policy = 0;
  int dropped_single_board_pair_count = 0;
  int final_selected_pair_board_count = 0;
  double initial_seed_rmse = std::numeric_limits<double>::infinity();
  double final_selected_rmse = std::numeric_limits<double>::infinity();
  std::set<std::pair<int, int> > selected_pair_board_keys;
  std::vector<StereoPairBoardTrialSelectionDecision> decisions;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct StereoExtrinsicCandidateDispersionRow {
  int pair_index = -1;
  int board_id = -1;
  bool consistency_accepted = false;
  double rotation_delta_deg = 0.0;
  double translation_delta_m = 0.0;
  double baseline_length = 0.0;
};

struct StereoExtrinsicJackknifeRow {
  int excluded_pair_index = -1;
  int remaining_candidate_count = 0;
  double rotation_delta_deg = std::numeric_limits<double>::infinity();
  double translation_delta_m = std::numeric_limits<double>::infinity();
  double baseline_length = std::numeric_limits<double>::infinity();
};

struct StereoExtrinsicUncertaintySummary {
  bool enabled = false;
  bool success = false;
  int candidate_count = 0;
  int accepted_candidate_count = 0;
  double rotation_delta_mean_deg = 0.0;
  double rotation_delta_median_deg = 0.0;
  double translation_delta_mean_m = 0.0;
  double translation_delta_median_m = 0.0;
  double baseline_length_mean = 0.0;
  double baseline_length_std = 0.0;
  double jackknife_rotation_max_deg = 0.0;
  double jackknife_translation_max_m = 0.0;
  int worst_jackknife_pair_index = -1;
  std::vector<StereoExtrinsicCandidateDispersionRow> candidate_rows;
  std::vector<StereoExtrinsicJackknifeRow> jackknife_rows;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct StereoPairBoardConsistencyRow {
  std::string split;
  int pair_index = -1;
  std::string left_frame_label;
  std::string right_frame_label;
  bool is_training = true;
  int board_id = -1;
  bool shared_board = false;
  bool cam0_only_board = false;
  bool cam1_only_board = false;
  int cam0_outer_point_count = 0;
  int cam1_outer_point_count = 0;
  int global_outer_point_count = 0;
  double global_outer_rmse = std::numeric_limits<double>::infinity();
  bool cam0_local_success = false;
  bool cam1_local_success = false;
  double cam0_local_outer_rmse = std::numeric_limits<double>::infinity();
  double cam1_local_outer_rmse = std::numeric_limits<double>::infinity();
  double local_outer_rmse = std::numeric_limits<double>::infinity();
  double cam0_pose_delta_rotation_deg = std::numeric_limits<double>::infinity();
  double cam0_pose_delta_translation_m = std::numeric_limits<double>::infinity();
  double cam1_pose_delta_rotation_deg = std::numeric_limits<double>::infinity();
  double cam1_pose_delta_translation_m = std::numeric_limits<double>::infinity();
  double stereo_local_pose_delta_rotation_deg =
      std::numeric_limits<double>::infinity();
  double stereo_local_pose_delta_translation_m =
      std::numeric_limits<double>::infinity();
  double cam1_outer_rmse_from_cam0_pose =
      std::numeric_limits<double>::infinity();
  double cam0_outer_rmse_from_cam1_pose =
      std::numeric_limits<double>::infinity();
  double stereo_outer_rmse_from_cam0_pose =
      std::numeric_limits<double>::infinity();
  double stereo_outer_rmse_from_cam1_pose =
      std::numeric_limits<double>::infinity();
  double cam1_outer_rmse_from_cam0_pose_inverse_extrinsic =
      std::numeric_limits<double>::infinity();
  double cam0_outer_rmse_from_cam1_pose_inverse_extrinsic =
      std::numeric_limits<double>::infinity();
  bool local_good_global_bad = false;
  bool rejected_by_consistency_gate = false;
  std::string diagnosis_label;
};

struct StereoPairBoardConsistencySummary {
  bool enabled = false;
  bool gate_enabled = false;
  double local_good_max_outer_rmse_px = 3.0;
  double global_bad_min_outer_rmse_px = 15.0;
  int row_count = 0;
  int training_row_count = 0;
  int holdout_row_count = 0;
  int local_good_global_bad_count = 0;
  int gate_rejected_pair_board_count = 0;
  std::set<std::pair<int, int> > gate_rejected_pair_boards;
  std::vector<StereoPairBoardConsistencyRow> rows;
  std::vector<std::string> warnings;
};

struct StereoExtrinsicCalibrationResult {
  bool success = false;
  StereoExtrinsicProblemInput problem_input;
  StereoSceneState optimized_scene;
  StereoSceneState post_initialization_scene;
  StereoSceneState pre_global_sparse_ba_scene;
  StereoInitializationDiagnostics initialization;
  StereoPairOnlyBaInitSummary pair_init_summary;
  StereoPairSelectionSummary pair_selection_summary;
  StereoPairTrialSelectionSummary pair_trial_selection_summary;
  StereoPairBoardTrialSelectionSummary pair_board_trial_selection_summary;
  StereoGlobalSparseBaSummary global_sparse_ba_summary;
  StereoExtrinsicUncertaintySummary extrinsic_uncertainty_summary;
  StereoPairBoardConsistencySummary pair_board_consistency_summary;
  Stage6RuntimeSummary runtime_summary;
  StereoResidualSummary training_residual_summary;
  StereoResidualSummary training_selected_initial_residual_summary;
  StereoResidualSummary training_selected_final_residual_summary;
  StereoResidualSummary holdout_residual_summary;
  StereoResidualSummary holdout_extrinsic_only_residual_summary;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

StereoMeasurementDataset BuildStereoMeasurementDataset(
    const std::vector<std::string>& left_image_paths,
    const std::vector<std::string>& right_image_paths,
    int holdout_stride,
    int holdout_offset,
    const CalibrationStateBundle& left_bundle,
    const CalibrationStateBundle& right_bundle,
    StereoMeasurementSourceMode source_mode =
        StereoMeasurementSourceMode::BackendSelectedOnly);

void WriteStereoExtrinsicYaml(const std::string& path,
                              const StereoExtrinsicCalibrationResult& result);
void WriteStereoExtrinsicSummary(const std::string& path,
                                 const StereoExtrinsicCalibrationResult& result);
void WriteStereoReprojectionSummary(const std::string& path,
                                    const StereoExtrinsicCalibrationResult& result);
void WriteStereoPerCameraResidualsCsv(const std::string& path,
                                      const StereoExtrinsicCalibrationResult& result);
void WriteStereoPerFrameResidualsCsv(const std::string& path,
                                     const StereoExtrinsicCalibrationResult& result);
void WriteStereoBaFrameFactorTraceCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoJacobianBlockDiagnosticsCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoPerBoardResidualsCsv(const std::string& path,
                                     const StereoExtrinsicCalibrationResult& result);
void WriteStereoIntrinsicsSanitySummary(const std::string& path,
                                        const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairingSummary(const std::string& path,
                               const StereoExtrinsicProblemInput& input);
void WriteStereoInitializationSummary(const std::string& path,
                                      const StereoExtrinsicCalibrationResult& result);
void WriteStereoGraphSummary(const std::string& path,
                             const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairSelectionSummary(const std::string& path,
                                     const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairSelectionCsv(const std::string& path,
                                 const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairInitSummary(const std::string& path,
                                const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairInitCandidatesCsv(const std::string& path,
                                      const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairInitResidualsCsv(const std::string& path,
                                     const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairTrialSelectionSummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairTrialSelectionDecisionsCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairTrialSelectedPairsCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairBoardTrialSelectionSummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairBoardTrialSelectionDecisionsCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairBoardTrialSelectedBoardsCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoGlobalSparseBaSummary(const std::string& path,
                                      const StereoExtrinsicCalibrationResult& result);
void WriteStereoGlobalSparseBaInitialVsFinal(const std::string& path,
                                             const StereoExtrinsicCalibrationResult& result);
void WriteStage6RuntimeSummary(const std::string& path,
                               const StereoExtrinsicCalibrationResult& result);
void WriteStereoRobustLossSummary(const std::string& path,
                                  const StereoExtrinsicCalibrationResult& result);
void WriteStereoExtrinsicUncertaintySummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoExtrinsicCandidateDispersionCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoExtrinsicJackknifeCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairBoardConsistencySummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairBoardLocalGlobalGapSummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoPairBoardConsistencyCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoSharedBoardQualityAuditCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoReprojectionVisualizations(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& result,
    int top_k);
void WriteStereoExtrinsicOnlyTopBadPairBoardVisualizations(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& result,
    const StereoSceneState& scene_state,
    const std::string& label,
    int top_k);
void WriteStereoBackendInputVisualizations(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& result,
    int top_k);
void WriteStereoAngularFixedKCornerTraceCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoAngularFixedKSummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteStereoHoldoutBoardPolarRmseCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result);
void WriteCoObsFactorBaExperiment(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& baseline_result);
void WriteStereoPairSelectionVisualizations(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& result,
    int top_k);
void WriteStereoPairBoardSelectionVisualizations(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& result,
    int top_k);

const char* ToString(StereoPairPoseRefitMode mode);
const char* ToString(StereoViewSelectionMode mode);
const char* ToString(StereoSolverMode mode);
const char* ToString(StereoSingleCameraOnlyWeightMode mode);
const char* ToString(StereoPairBoardSelectionMode mode);
const char* ToString(StereoFinalBaResidualMode mode);
const char* ToString(StereoSphericalUncertaintyMode mode);
const char* ToString(StereoCandidateBudgetMode mode);
const char* ToString(StereoSelectionOptimizationMode mode);
const char* ToString(StereoRigParamMode mode);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_STEREO_EXTRINSIC_PROBLEM_INPUT_HPP
