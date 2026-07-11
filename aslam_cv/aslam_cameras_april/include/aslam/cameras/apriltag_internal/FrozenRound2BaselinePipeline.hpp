#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_FROZEN_ROUND2_BASELINE_PIPELINE_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_FROZEN_ROUND2_BASELINE_PIPELINE_HPP

#include <string>
#include <vector>

#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/OuterDetectionCache.hpp>
#include <aslam/cameras/apriltag_internal/OuterOnlyCameraInitializer.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct FrozenRound2BaselineFrameSource {
  int frame_index = -1;
  std::string frame_label;
  std::string image_path;
};

struct JointMeasurementBuildValidationSummary {
  bool success = false;
  bool counting_consistent = false;
  bool flat_hierarchical_consistent = false;
  bool frame_order_invariant = false;
  bool label_mismatch_warning_observed = false;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct FrozenRoundArtifacts {
  std::vector<InternalRegenerationFrameResult> regeneration_results;
  std::vector<JointMeasurementFrameInput> joint_inputs;
  JointMeasurementBuildResult measurement_result;
  JointMeasurementBuildValidationSummary validation_summary;
  JointResidualEvaluationResult residual_result;
  JointMeasurementSelectionResult selection_result;
  JointOptimizationResult optimization_result;
};

struct OuterOnlyIntermediateCalibrationResult {
  bool enabled = false;
  bool diagnostic_only = true;
  bool use_for_round1_requested = false;
  bool use_for_full_frontend_regeneration_requested = false;
  bool used_for_round1_internal_regeneration = false;
  bool used_for_full_frontend_regeneration = false;
  bool success = false;
  std::string state_source_label = "bootstrap";
  JointMeasurementBuildResult measurement_result;
  JointResidualEvaluationResult initial_residual_result;
  JointMeasurementSelectionResult selection_result;
  JointOptimizationResult optimization_result;
  int total_outer_board_observation_count = 0;
  int used_outer_board_observation_count = 0;
  int rejected_outer_board_observation_count = 0;
  int used_outer_point_count = 0;
  int used_internal_point_count = 0;
  double max_outer_rmse_px = 8.0;
  int min_visible_boards = 1;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct FrozenRound2BaselineRuntimeBreakdown {
  OuterDetectionCacheStats training_detection_cache;
  double training_outer_detection_seconds = 0.0;
  double auto_camera_initialization_seconds = 0.0;
  double outer_bootstrap_seconds = 0.0;
  double outer_only_intermediate_measurement_build_seconds = 0.0;
  double outer_only_intermediate_residual_evaluation_seconds = 0.0;
  double outer_only_intermediate_selection_seconds = 0.0;
  double outer_only_intermediate_optimization_seconds = 0.0;
  double round1_regeneration_seconds = 0.0;
  double round1_regeneration_pose_estimation_seconds = 0.0;
  double round1_regeneration_boundary_model_seconds = 0.0;
  double round1_regeneration_seed_search_seconds = 0.0;
  double round1_regeneration_ray_refine_seconds = 0.0;
  double round1_regeneration_image_evidence_seconds = 0.0;
  double round1_regeneration_subpix_seconds = 0.0;
  int round1_regeneration_attempted_internal_corners = 0;
  int round1_regeneration_valid_internal_corners = 0;
  double round1_measurement_build_seconds = 0.0;
  double round1_residual_evaluation_seconds = 0.0;
  double round1_selection_seconds = 0.0;
  double round1_optimization_seconds = 0.0;
  double round1_optimization_residual_evaluation_seconds = 0.0;
  int round1_optimization_residual_evaluation_call_count = 0;
  double round1_optimization_cost_evaluation_seconds = 0.0;
  int round1_optimization_cost_evaluation_call_count = 0;
  double round1_optimization_frame_update_seconds = 0.0;
  double round1_optimization_board_update_seconds = 0.0;
  double round1_optimization_intrinsics_update_seconds = 0.0;
  double round2_regeneration_seconds = 0.0;
  double round2_regeneration_pose_estimation_seconds = 0.0;
  double round2_regeneration_boundary_model_seconds = 0.0;
  double round2_regeneration_seed_search_seconds = 0.0;
  double round2_regeneration_ray_refine_seconds = 0.0;
  double round2_regeneration_image_evidence_seconds = 0.0;
  double round2_regeneration_subpix_seconds = 0.0;
  int round2_regeneration_attempted_internal_corners = 0;
  int round2_regeneration_valid_internal_corners = 0;
  double round2_measurement_build_seconds = 0.0;
  double round2_residual_evaluation_seconds = 0.0;
  double round2_selection_seconds = 0.0;
  double round2_optimization_seconds = 0.0;
  double round2_optimization_residual_evaluation_seconds = 0.0;
  int round2_optimization_residual_evaluation_call_count = 0;
  double round2_optimization_cost_evaluation_seconds = 0.0;
  int round2_optimization_cost_evaluation_call_count = 0;
  double round2_optimization_frame_update_seconds = 0.0;
  double round2_optimization_board_update_seconds = 0.0;
  double round2_optimization_intrinsics_update_seconds = 0.0;
};

struct FrozenRound2BaselineOptions {
  ApriltagInternalConfig config;
  int reference_board_id = 1;
  bool outer_only_ablation_mode = false;
  bool include_internal_points = true;
  bool optimize_intrinsics = false;
  int intrinsics_release_iteration = 3;
  bool run_second_pass = true;
  int second_pass_intrinsics_release_iteration = 1;
  bool enable_residual_sanity_gate = true;
  bool enable_board_pose_fit_gate = false;
  JointMeasurementSelectionMode selection_mode =
      JointMeasurementSelectionMode::Baseline;
  double selection_residual_sanity_factor = 2.5;
  double selection_max_board_observation_rmse = 25.0;
  double selection_kalibr_style_outlier_sigma = 4.0;
  double selection_kalibr_style_min_abs_threshold_px = 1.0;
  int selection_kalibr_style_min_views_before_filter = 20;
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
  bool use_explicit_initial_camera = false;
  OuterBootstrapCameraIntrinsics explicit_initial_camera;
  std::string explicit_initial_camera_source_label = "explicit_initial_camera";
  AutoCameraInitializationRefineMode camera_initialization_refine_mode =
      AutoCameraInitializationRefineMode::KalibrOuterLm;
  bool enable_internal_observation_quality_weighting = false;
  double internal_observation_low_quality_quantile = 0.2;
  double internal_observation_min_weight = 0.25;
  double internal_observation_quality_exponent = 1.0;
  InternalPoseRescueMode internal_pose_rescue_mode =
      InternalPoseRescueMode::Enabled;
  double internal_pose_rescue_max_ray_angle_deg = 88.0;
  double internal_pose_rescue_accept_max_outer_rmse = 8.0;
  bool enable_geometry_prior_outer_seed = false;
  bool geometry_prior_rescue_diagnostic_only = true;
  bool geometry_prior_rescue_use_as_observation = false;
  bool geometry_prior_rescue_keep_outer_on_internal_failure = false;
  bool geometry_prior_rescue_allow_geometry_only_pose_refit = false;
  // 0 means adapt from the predicted board pixel scale, positive forces a
  // fixed radius, and negative disables geometry-prior subpixel refinement.
  int geometry_prior_rescue_subpix_window_radius = 0;
  // <= 0 disables the displacement upper bound. Geometry-prior rescue often
  // starts farther from the true corner than normal decoded-tag refinement.
  double geometry_prior_rescue_max_corner_displacement_px = 0.0;
  double geometry_prior_rescue_min_corner_response_ratio = 0.03;
  bool geometry_prior_rescue_enable_spherical_refine = false;
  int geometry_prior_rescue_edge_sample_count = 80;
  int geometry_prior_rescue_edge_search_half_width_px = 6;
  double geometry_prior_rescue_min_edge_support_ratio = 0.45;
  double geometry_prior_rescue_min_edge_gradient_ratio = 0.02;
  double geometry_prior_rescue_accept_max_outer_rmse = 8.0;
  double geometry_prior_rescue_accept_max_rotation_error_deg = 5.0;
  double geometry_prior_rescue_accept_max_translation_error = 0.08;
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
  std::string dataset_label;
  std::string training_split_signature = "all_frames";
  std::string baseline_protocol_label = "frozen_round2_v2_kalibr_corner_filter";
  std::string source_pipeline_label = "frozen_round2_baseline";
  bool enable_outer_detection_cache = false;
  std::string outer_detection_cache_dir;
};

struct FrozenRound2BaselineResult {
  bool success = false;
  std::string baseline_protocol_label = "frozen_round2_v2_kalibr_corner_filter";
  std::string dataset_label;
  std::string training_split_signature = "all_frames";
  int reference_board_id = 1;
  std::vector<FrozenRound2BaselineFrameSource> frame_sources;
  OuterBootstrapResult bootstrap_result;
  OuterOnlyIntermediateCalibrationResult outer_only_intermediate;
  FrozenRoundArtifacts round1;
  bool round2_available = false;
  FrozenRoundArtifacts round2;
  bool stage42_validation_pass = false;
  CalibrationStateBundle stage5_round1_bundle;
  CalibrationStateBundle final_stage5_bundle;
  bool stage5_bundle_available = false;
  AutoCameraInitializationResult auto_camera_initialization;
  FrozenRound2BaselineOptions effective_options;
  FrozenRound2BaselineRuntimeBreakdown runtime_breakdown;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class FrozenRound2BaselinePipeline {
 public:
  explicit FrozenRound2BaselinePipeline(
      FrozenRound2BaselineOptions options = FrozenRound2BaselineOptions{});

  FrozenRound2BaselineResult Run(
      const std::vector<FrozenRound2BaselineFrameSource>& frame_sources) const;

  const FrozenRound2BaselineOptions& options() const { return options_; }

 private:
  FrozenRound2BaselineOptions options_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_FROZEN_ROUND2_BASELINE_PIPELINE_HPP
