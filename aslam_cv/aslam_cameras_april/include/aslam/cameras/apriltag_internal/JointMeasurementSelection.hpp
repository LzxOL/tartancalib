#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_MEASUREMENT_SELECTION_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_MEASUREMENT_SELECTION_HPP

#include <set>
#include <string>
#include <vector>

#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionResidualEvaluator.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

enum class JointFrameSelectionReasonCode {
  None = 0,
  NoUsableBoardObservations,
  AcceptedMinViewsPerBoard,
  AcceptedNewBoardPair,
  AcceptedNewImageCoverage,
  RejectedRedundantView,
};

enum class JointBoardObservationSelectionReasonCode {
  None = 0,
  Accepted,
  RejectedNotSolverReady,
  RejectedResidualSanity,
  RejectedOuterPoseFit,
  RejectedFrameRejected,
};

const char* ToString(JointFrameSelectionReasonCode reason_code);
const char* ToString(JointBoardObservationSelectionReasonCode reason_code);

struct JointMeasurementSelectionOptions {
  int reference_board_id = 1;
  int coverage_grid_cols = 4;
  int coverage_grid_rows = 4;
  int min_initial_views_per_board = 3;
  double max_board_observation_rmse = 25.0;
  double residual_sanity_factor = 2.5;
  double max_pose_fit_outer_rmse = 8.0;
  bool enable_residual_sanity_gate = true;
  bool enable_board_pose_fit_gate = false;
};

struct JointBoardObservationSelectionDecision {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  bool accepted = false;
  JointBoardObservationSelectionReasonCode reason_code =
      JointBoardObservationSelectionReasonCode::None;
  std::string reason_detail;
  double rmse = 0.0;
  double pose_fit_outer_rmse = 0.0;
  double average_quality = 0.0;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  std::string coverage_signature;
};

struct JointFrameSelectionDecision {
  int frame_index = -1;
  std::string frame_label;
  bool accepted = false;
  std::vector<int> usable_board_ids;
  std::vector<int> accepted_board_ids;
  std::vector<JointFrameSelectionReasonCode> reason_codes;
  std::string reason_detail;
  int usable_board_observation_count = 0;
  int accepted_board_observation_count = 0;
};

struct JointMeasurementSelectionResult {
  bool success = false;
  int reference_board_id = 1;
  std::set<int> accepted_frame_indices;
  std::set<std::pair<int, int> > accepted_board_observation_keys;
  JointMeasurementBuildResult selected_measurement_result;
  std::vector<JointFrameSelectionDecision> frame_decisions;
  std::vector<JointBoardObservationSelectionDecision> board_observation_decisions;
  int accepted_frame_count = 0;
  int accepted_board_observation_count = 0;
  int accepted_outer_point_count = 0;
  int accepted_internal_point_count = 0;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class JointMeasurementSelection {
 public:
  explicit JointMeasurementSelection(
      JointMeasurementSelectionOptions options = JointMeasurementSelectionOptions{});

  JointMeasurementSelectionResult Select(
      const JointMeasurementBuildResult& measurement_result,
      const JointResidualEvaluationResult& residual_result,
      const JointReprojectionSceneState& scene_state) const;

  const JointMeasurementSelectionOptions& options() const { return options_; }

 private:
  JointMeasurementSelectionOptions options_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_MEASUREMENT_SELECTION_HPP
