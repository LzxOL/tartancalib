#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_CONSISTENCY_DIAGNOSTICS_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_CONSISTENCY_DIAGNOSTICS_HPP

#include <string>
#include <vector>

#include <Eigen/Core>

#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct MultiBoardRigidityDiagnosticsOptions {
  bool enabled = false;
  int top_k = 30;
  double rotation_bad_threshold_deg = 3.0;
  double translation_bad_threshold = -1.0;
  double reprojection_delta_bad_threshold_px = 2.0;
  bool use_internal_points = true;
  bool use_outer_points = true;
};

struct FrameBoardConsistencyRecord {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_count_outer = 0;
  int point_count_internal = 0;
  bool local_pose_success = false;
  std::string local_pose_source = "failed";
  double rotation_error_deg = 0.0;
  double translation_error_norm = 0.0;
  double local_reprojection_rmse = 0.0;
  double global_reprojection_rmse = 0.0;
  double reprojection_rmse_delta = 0.0;
  double mean_residual_x_global = 0.0;
  double mean_residual_y_global = 0.0;
  double rmse_x_global = 0.0;
  double rmse_y_global = 0.0;
  double polar_angle_mean = 0.0;
  double polar_angle_max = 0.0;
  bool is_reference_board = false;
  std::string diagnostic_status = "unknown";
};

struct BoardConsistencyAggregate {
  int board_id = -1;
  int observation_count = 0;
  int local_success_count = 0;
  double mean_rotation_error_deg = 0.0;
  double max_rotation_error_deg = 0.0;
  double mean_translation_error_norm = 0.0;
  double max_translation_error_norm = 0.0;
  double mean_global_rmse = 0.0;
  double mean_local_rmse = 0.0;
  double mean_rmse_delta = 0.0;
};

struct BoardPairwiseConsistencyRecord {
  int frame_index = -1;
  std::string frame_label;
  int board_i = -1;
  int board_j = -1;
  std::string pair_key;
  bool local_pose_success_i = false;
  bool local_pose_success_j = false;
  double pair_rotation_error_deg = 0.0;
  double pair_translation_error_norm = 0.0;
  int board_i_point_count = 0;
  int board_j_point_count = 0;
  double board_i_global_rmse = 0.0;
  double board_j_global_rmse = 0.0;
  double board_i_local_rmse = 0.0;
  double board_j_local_rmse = 0.0;
  std::string pair_diagnostic_status = "unknown";
};

struct BoardPairwiseConsistencyAggregate {
  std::string pair_key;
  int board_i = -1;
  int board_j = -1;
  int observation_count = 0;
  double mean_rotation_error_deg = 0.0;
  double median_rotation_error_deg = 0.0;
  double max_rotation_error_deg = 0.0;
  double mean_translation_error_norm = 0.0;
  double median_translation_error_norm = 0.0;
  double max_translation_error_norm = 0.0;
  double mean_board_i_global_rmse = 0.0;
  double mean_board_j_global_rmse = 0.0;
  int bad_count_by_rotation_threshold = 0;
  int bad_count_by_translation_threshold = 0;
};

struct MultiBoardRigidityDiagnosticsResult {
  bool success = false;
  bool training_only = false;
  int total_frame_board_observations = 0;
  int local_pose_success_count = 0;
  int local_pose_failure_count = 0;
  double mean_rotation_error_deg = 0.0;
  double median_rotation_error_deg = 0.0;
  double max_rotation_error_deg = 0.0;
  double mean_translation_error_norm = 0.0;
  double median_translation_error_norm = 0.0;
  double max_translation_error_norm = 0.0;
  double mean_global_reprojection_rmse = 0.0;
  double mean_local_reprojection_rmse = 0.0;
  double mean_reprojection_rmse_delta = 0.0;
  int bad_by_rotation_threshold_count = 0;
  int bad_by_reprojection_delta_threshold_count = 0;
  int total_pair_observations = 0;
  int unique_pair_count = 0;
  double mean_pair_rotation_error_deg = 0.0;
  double max_pair_rotation_error_deg = 0.0;
  double mean_pair_translation_error_norm = 0.0;
  double max_pair_translation_error_norm = 0.0;
  std::vector<FrameBoardConsistencyRecord> frame_board_records;
  std::vector<BoardConsistencyAggregate> board_aggregates;
  std::vector<BoardPairwiseConsistencyRecord> board_pairwise_records;
  std::vector<BoardPairwiseConsistencyAggregate> board_pairwise_aggregates;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class MultiBoardRigidityDiagnostics {
 public:
  explicit MultiBoardRigidityDiagnostics(
      MultiBoardRigidityDiagnosticsOptions options =
          MultiBoardRigidityDiagnosticsOptions{});

  MultiBoardRigidityDiagnosticsResult Evaluate(
      const CalibrationMeasurementDataset& observations,
      const JointReprojectionSceneState& scene_state) const;

  const MultiBoardRigidityDiagnosticsOptions& options() const {
    return options_;
  }

 private:
  MultiBoardRigidityDiagnosticsOptions options_;
};

void WriteFrameBoardConsistencyCsv(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result);
void WriteFrameBoardConsistencySummary(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result);
void WriteTopBadFrameBoardObservations(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result,
    int top_k);
void WriteBoardPairwiseConsistencyCsv(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result);
void WriteBoardPairwiseConsistencySummary(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result);
void WriteTopBadBoardPairs(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result,
    int top_k);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_CONSISTENCY_DIAGNOSTICS_HPP
