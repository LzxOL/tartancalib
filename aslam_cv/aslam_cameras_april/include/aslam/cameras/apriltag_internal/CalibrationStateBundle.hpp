#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_CALIBRATION_STATE_BUNDLE_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_CALIBRATION_STATE_BUNDLE_HPP

#include <set>
#include <string>
#include <utility>
#include <vector>

#include <aslam/cameras/apriltag_internal/JointReprojectionOptimizer.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct CalibrationSceneState {
  int reference_board_id = 1;
  std::string camera_model = "ds";
  std::string coarse_or_optimized_level;
  std::string bundle_version = "stage5_bundle_v1";
  std::string baseline_protocol_label = "frozen_round2_v1";
  std::string training_split_signature = "all_frames";
  OuterBootstrapCameraIntrinsics camera;
  std::vector<JointSceneBoardState> boards;
  std::vector<JointSceneFrameState> frames;
  std::string dataset_label;
  std::string source_pipeline_label;
  std::vector<std::string> warnings;
  std::string failure_reason;

  bool IsValid() const { return camera_model == "ds" && camera.IsValid(); }
};

struct CalibrationMeasurementDataset {
  int reference_board_id = 1;
  std::string bundle_version = "stage5_bundle_v1";
  std::string baseline_protocol_label = "frozen_round2_v1";
  std::string training_split_signature = "all_frames";
  std::vector<JointMeasurementFrameResult> frames;
  std::vector<JointPointObservation> solver_observations;
  std::set<int> accepted_frame_indices;
  std::set<std::pair<int, int> > accepted_board_observation_keys;
  int accepted_frame_count = 0;
  int accepted_board_observation_count = 0;
  int accepted_outer_point_count = 0;
  int accepted_internal_point_count = 0;
  int accepted_total_point_count = 0;
  std::string dataset_label;
  std::string source_stage_label;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct CalibrationStateBundle {
  bool success = false;
  std::string bundle_version = "stage5_bundle_v1";
  std::string baseline_protocol_label = "frozen_round2_v1";
  std::string training_split_signature = "all_frames";
  CalibrationSceneState scene_state;
  CalibrationMeasurementDataset measurement_dataset;
  JointResidualEvaluationResult residual_result;
  JointMeasurementSelectionResult selection_result;
  int round_index = -1;
  bool ready_for_backend = false;
  std::vector<std::string> warnings;
  std::string failure_reason;

  bool IsReadyForBackend() const { return success && ready_for_backend; }
  std::string SummaryString() const;
};

struct CalibrationBackendParameterization {
  std::string camera_model = "ds";
  std::string pose_parameterization = "se3";
  bool reference_board_fixed = true;
};

struct CalibrationOptimizationMasks {
  bool optimize_frame_poses = true;
  bool optimize_board_poses = true;
  bool optimize_intrinsics = false;
  bool delayed_intrinsics_release = true;
  int intrinsics_release_iteration = 3;
};

struct CalibrationPriorSettings {
  bool use_intrinsics_anchor_prior = true;
  double intrinsics_anchor_weight_xi_alpha = 1e-2;
  double intrinsics_anchor_weight_focal = 1e-4;
  double intrinsics_anchor_weight_principal = 1e-4;
};

struct CalibrationDiagnosticsSeed {
  double overall_rmse = 0.0;
  double outer_only_rmse = 0.0;
  double internal_only_rmse = 0.0;
  int accepted_total_point_count = 0;
  std::vector<std::string> warnings;
};

struct CalibrationBackendProblemInput {
  int reference_board_id = 1;
  std::string bundle_version = "stage5_bundle_v1";
  std::string baseline_protocol_label = "frozen_round2_v1";
  std::string training_split_signature = "all_frames";
  std::string dataset_label;
  CalibrationSceneState scene_state;
  CalibrationMeasurementDataset measurement_dataset;
  JointResidualEvaluationResult residual_result;
  CalibrationBackendParameterization parameterization;
  CalibrationOptimizationMasks optimization_masks;
  CalibrationPriorSettings priors;
  CalibrationDiagnosticsSeed diagnostics_seed;
};

struct CalibrationBundleMetadata {
  std::string bundle_version = "stage5_bundle_v1";
  std::string baseline_protocol_label = "frozen_round2_v1";
  std::string training_split_signature = "all_frames";
  std::string dataset_label;
  std::string source_pipeline_label = "run_joint_reprojection_optimization";
};

struct BackendProblemOptions {
  int reference_board_id = 1;
  bool optimize_frame_poses = true;
  bool optimize_board_poses = true;
  bool optimize_intrinsics = false;
  bool delayed_intrinsics_release = true;
  int intrinsics_release_iteration = 3;
  double intrinsics_anchor_weight_xi_alpha = 1e-2;
  double intrinsics_anchor_weight_focal = 1e-4;
  double intrinsics_anchor_weight_principal = 1e-4;
};

CalibrationSceneState BuildCalibrationSceneState(
    const JointReprojectionSceneState& scene_state,
    const std::string& level,
    const CalibrationBundleMetadata& metadata);

CalibrationMeasurementDataset BuildCalibrationMeasurementDataset(
    const JointMeasurementSelectionResult& selection_result,
    const JointMeasurementBuildResult& measurement_result,
    const std::string& source_stage_label,
    const CalibrationBundleMetadata& metadata);

CalibrationStateBundle BuildCalibrationStateBundleFromJointOptimizationResult(
    const JointOptimizationResult& optimization_result,
    const JointMeasurementSelectionResult& selection_result,
    const JointMeasurementBuildResult& measurement_result,
    int round_index,
    const std::string& dataset_label);

CalibrationStateBundle BuildCalibrationStateBundleFromJointOptimizationResult(
    const JointOptimizationResult& optimization_result,
    const JointMeasurementSelectionResult& selection_result,
    const JointMeasurementBuildResult& measurement_result,
    int round_index,
    const CalibrationBundleMetadata& metadata);

CalibrationBackendProblemInput BuildBackendProblemInput(
    const CalibrationStateBundle& bundle,
    const BackendProblemOptions& options = BackendProblemOptions{});

JointReprojectionSceneState BuildJointSceneStateFromCalibrationSceneState(
    const CalibrationSceneState& scene_state);

void WriteCalibrationStateBundleSummary(const std::string& path,
                                        const CalibrationStateBundle& bundle);
void WriteCalibrationBackendProblemSummary(
    const std::string& path,
    const CalibrationBackendProblemInput& backend_problem_input);

bool LoadCalibrationStateBundleSummary(const std::string& path,
                                       CalibrationStateBundle* bundle,
                                       std::string* error_message);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_CALIBRATION_STATE_BUNDLE_HPP
