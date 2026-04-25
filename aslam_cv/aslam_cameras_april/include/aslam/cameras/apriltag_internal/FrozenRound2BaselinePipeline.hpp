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

struct FrozenRound2BaselineRuntimeBreakdown {
  OuterDetectionCacheStats training_detection_cache;
  double training_outer_detection_seconds = 0.0;
  double auto_camera_initialization_seconds = 0.0;
  double outer_bootstrap_seconds = 0.0;
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
  bool optimize_intrinsics = false;
  int intrinsics_release_iteration = 3;
  bool run_second_pass = true;
  int second_pass_intrinsics_release_iteration = 1;
  bool enable_residual_sanity_gate = true;
  bool enable_board_pose_fit_gate = false;
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
