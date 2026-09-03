#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_FROZEN_ROUND2_BASELINE_PIPELINE_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_FROZEN_ROUND2_BASELINE_PIPELINE_HPP

#include <string>
#include <vector>

#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>
#include <aslam/cameras/apriltag_internal/InternalRegenerationCache.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/OuterDetectionCache.hpp>
#include <aslam/cameras/apriltag_internal/OuterOnlyCameraInitializer.hpp>
#include <aslam/cameras/apriltag_internal/Stage5RecoveryTypes.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

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
  InternalRegenerationCacheStats training_internal_regeneration_cache;
  double training_outer_detection_seconds = 0.0;
  double auto_camera_initialization_seconds = 0.0;
  double camera_aware_outer_rescue_seconds = 0.0;
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
  // Optional live progress reporting. Disabled by default so the frozen
  // baseline's numerical path and normal stdout remain unchanged.
  bool enable_progress_reporting = false;
  std::string progress_log_path;
  int progress_report_interval_frames = 1;
  // Complete detection/initialization/recovery and build frontend
  // observations, then stop before selection or any persistent backend BA.
  bool frontend_only = false;
  bool outer_only_ablation_mode = false;
  bool include_internal_points = true;
  bool optimize_intrinsics = false;
  bool optimize_bootstrap_intrinsics = true;
  // Keep disabled for the frozen baseline.  The explicit model-aware path
  // enables robust shared-layout initialization before Persistent BA.
  bool robust_board_layout_consensus = false;
  // Explicit measured-rig mode. Empty keeps the historical self-estimated
  // shared layout and all baseline behavior unchanged.
  FixedBoardLayout fixed_board_layout;
  std::string fixed_board_layout_source;
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
  // Round 1 still requires the reference board and refines the shared layout.
  // When enabled, Round 2 can add initialized non-reference-only frames while
  // keeping that layout fixed.
  bool allow_non_reference_board_frames_after_layout = false;
  bool strict_board_observation_acceptance = false;
  bool preserve_frame_board_cohesion = false;
  bool ignore_image_evidence_min_quality = false;
  bool force_internal_seed_from_prediction = false;
  bool bypass_internal_seed_filters = false;
  bool enable_internal_lattice_slot_ownership_check = true;
  std::string internal_corner_filter_mode = "sigma";
  double internal_corner_filter_max_reproj_error = -1.0;
  double internal_corner_filter_quality_min = 0.35;
  double internal_corner_filter_quality_relaxation_px = 1.0;
  double internal_corner_filter_adaptive_min_threshold_px = 1.0;
  bool enable_bidirectional_board_topology_consistency = true;
  bool use_explicit_initial_camera = false;
  OuterBootstrapCameraIntrinsics explicit_initial_camera;
  std::string explicit_initial_camera_source_label = "explicit_initial_camera";
  AutoCameraInitializationRefineMode camera_initialization_refine_mode =
      AutoCameraInitializationRefineMode::KalibrOuterLm;
  AutoCameraInitializationSelectionScorer camera_initialization_selection_scorer =
      AutoCameraInitializationSelectionScorer::PoseMarginalizedPrincipal;
  bool camera_initialization_shared_focal = false;
  // Select the initialization camera with one target pose per frame-board.
  // Scene bootstrap and backend BA still use the shared board layout.
  bool camera_initialization_use_independent_frame_board_poses = false;
  bool enable_camera_initialization_principal_profile = false;
  double camera_initialization_principal_profile_radius_px = 10.0;
  bool enable_camera_initialization_fixed_layout_diagnostic = false;
  bool enable_camera_initialization_board_jackknife_diagnostic = false;
  bool enable_camera_initialization_coverage_weighted_diagnostic = false;
  bool camera_initialization_prefer_lower_focal_in_near_tie = false;
  double camera_initialization_near_tie_relative_objective_tolerance = 0.0;
  double checkerboard_initialization_huber_delta_pixels = 1.5;
  bool precomputed_initialization_use_all_points = false;
  std::string precomputed_initialization_point_scope = "all";
  // Frozen baseline: recover missing exact-ID outer tags after provisional
  // camera initialization. The zero-detection atlas policy lives in
  // config.outer_detector_config and defaults to enabled.
  bool enable_camera_aware_outer_rescue = true;
  // Additive exact-ID detector fallback for unresolved boards. Its OpenCV
  // corner convention is normalized to the ETHZ/canonical board convention
  // before any refinement or internal regeneration.
  bool enable_opencv_apriltag_fallback = true;
  // Enables the multi-evidence missing-board path.  Direct exact-ID
  // detections remain unchanged; this flag only relaxes recovery-specific
  // visibility/refinement and boundary-model failure handling.
  bool enable_robust_missing_board_recovery = false;
  bool rerun_camera_initialization_after_outer_rescue = true;
  int camera_aware_outer_rescue_max_hamming = 0;
  // Zero selects a bounded automatic worker count.  Each worker owns its
  // AprilTag detector; therefore no detector state is shared across frames.
  int camera_aware_outer_rescue_worker_count = 0;
  // Auxiliary sessions participate only in camera initialization. Their
  // frame-board observations retain independent poses and never enter the
  // primary session's layout, selection, or backend problem.
  std::vector<OuterBootstrapFrameInput>
      camera_initialization_auxiliary_bootstrap_frames;
  int camera_initialization_auxiliary_session_count = 0;
  bool enable_internal_observation_quality_weighting = false;
  double internal_observation_low_quality_quantile = 0.2;
  double internal_observation_min_weight = 0.25;
  double internal_observation_quality_exponent = 1.0;
  InternalPoseRescueMode internal_pose_rescue_mode =
      InternalPoseRescueMode::Enabled;
  double internal_pose_rescue_max_ray_angle_deg = 88.0;
  double internal_pose_rescue_accept_max_outer_rmse = 8.0;
  // Frozen baseline: image-validated geometry-prior candidates may restore a
  // missing board when a bootstrap/scene pose prior exists. Pure projection is
  // still never committed without the existing image and pose checks.
  bool enable_geometry_prior_outer_seed = true;
  bool geometry_prior_rescue_diagnostic_only = false;
  bool geometry_prior_rescue_use_as_observation = true;
  bool geometry_prior_rescue_keep_outer_on_internal_failure = false;
  // A missing board can be identified by the rigid multi-board topology even
  // when its distorted AprilTag payload is not decodable. This remains gated
  // by independently visible boards, local image geometry, and pose refit.
  bool geometry_prior_rescue_allow_geometry_only_pose_refit = true;
  // 0 means adapt from the predicted board pixel scale, positive forces a
  // fixed radius, and negative disables geometry-prior subpixel refinement.
  int geometry_prior_rescue_subpix_window_radius = 0;
  // <= 0 disables the displacement upper bound. Geometry-prior rescue often
  // starts farther from the true corner than normal decoded-tag refinement.
  double geometry_prior_rescue_max_corner_displacement_px = 0.0;
  double geometry_prior_rescue_min_corner_response_ratio = 0.03;
  // Also enables the DS spherical edge-support bridge for recovered boards;
  // the existing spherical-refine disable flag ablates both together.
  bool geometry_prior_rescue_enable_spherical_refine = true;
  int geometry_prior_rescue_edge_sample_count = 80;
  int geometry_prior_rescue_edge_search_half_width_px = 6;
  double geometry_prior_rescue_min_edge_support_ratio = 0.45;
  double geometry_prior_rescue_min_edge_gradient_ratio = 0.02;
  double geometry_prior_rescue_accept_max_outer_rmse = 8.0;
  bool geometry_prior_rescue_scale_aware_outer_rmse_gate = true;
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
  std::string dataset_label;
  std::string training_split_signature = "all_frames";
  std::string baseline_protocol_label = "frozen_round2_v3_recovery_cache";
  std::string source_pipeline_label = "frozen_round2_baseline";
  // When enabled, this single dataset-owned root writes/reads both the
  // stage-scoped outer-detection artifacts and internal-regeneration artifacts.
  bool enable_outer_detection_cache = false;
  std::string outer_detection_cache_dir;
};

struct FrozenRound2BaselineResult {
  bool success = false;
  std::string baseline_protocol_label = "frozen_round2_v3_recovery_cache";
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
  CameraAwareOuterRescueSummary camera_aware_outer_rescue;
  FrozenRound2BaselineOptions effective_options;
  FrozenRound2BaselineRuntimeBreakdown runtime_breakdown;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct FrozenPrecomputedMeasurementInput {
  bool success = false;
  std::string schema_version = "stage5_precomputed_observations_v1";
  std::string source_path;
  cv::Size image_size;
  int reference_board_id = 1;
  std::string target_mode_requested = "auto";
  std::string target_mode_resolved = "multi_board";
  int board_count = 0;
  bool single_board_mode = false;
  std::vector<FrozenRound2BaselineFrameSource> frame_sources;
  std::vector<OuterBootstrapFrameInput> bootstrap_frames;
  JointMeasurementBuildResult measurement_result;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class FrozenRound2BaselinePipeline {
 public:
  explicit FrozenRound2BaselinePipeline(
      FrozenRound2BaselineOptions options = FrozenRound2BaselineOptions{});

  FrozenRound2BaselineResult Run(
      const std::vector<FrozenRound2BaselineFrameSource>& frame_sources) const;

  FrozenRound2BaselineResult RunPrecomputed(
      const FrozenPrecomputedMeasurementInput& input) const;

  const FrozenRound2BaselineOptions& options() const { return options_; }

 private:
  FrozenRound2BaselineOptions options_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_FROZEN_ROUND2_BASELINE_PIPELINE_HPP
