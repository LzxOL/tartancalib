#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_FROZEN_ROUND2_BASELINE_PIPELINE_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_FROZEN_ROUND2_BASELINE_PIPELINE_HPP

#include <string>
#include <vector>

#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
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

struct FrozenRound2BaselineOptions {
  ApriltagInternalConfig config;
  int reference_board_id = 1;
  bool optimize_intrinsics = false;
  int intrinsics_release_iteration = 3;
  bool run_second_pass = true;
  int second_pass_intrinsics_release_iteration = 1;
  std::string dataset_label;
  std::string training_split_signature = "all_frames";
  std::string baseline_protocol_label = "frozen_round2_v2_kalibr_corner_filter";
  std::string source_pipeline_label = "frozen_round2_baseline";
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
