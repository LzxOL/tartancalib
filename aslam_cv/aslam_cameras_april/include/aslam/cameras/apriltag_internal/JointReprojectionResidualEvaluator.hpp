#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_RESIDUAL_EVALUATOR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_RESIDUAL_EVALUATOR_HPP

#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct JointResidualEvaluationOptions {
  int top_k = 10;
  JointReprojectionCostOptions cost_options;
};

struct JointResidualPointDiagnostics {
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
  bool used_in_solver = false;
  int frame_storage_index = -1;
  int source_board_observation_index = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind = JointObservationSourceKind::OuterMeasurement;
};

struct JointResidualBoardObservationDiagnostics {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double rmse = 0.0;
};

struct JointResidualBoardDiagnostics {
  int board_id = -1;
  int observation_count = 0;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double rmse = 0.0;
};

struct JointResidualFrameDiagnostics {
  int frame_index = -1;
  std::string frame_label;
  std::vector<int> visible_board_ids;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double rmse = 0.0;
};

struct JointResidualEvaluationResult {
  bool success = false;
  int reference_board_id = 1;
  double overall_rmse = 0.0;
  double outer_only_rmse = 0.0;
  double internal_only_rmse = 0.0;
  std::vector<JointResidualPointDiagnostics> point_diagnostics;
  std::vector<JointResidualBoardObservationDiagnostics> board_observation_diagnostics;
  std::vector<JointResidualBoardDiagnostics> board_diagnostics;
  std::vector<JointResidualFrameDiagnostics> frame_diagnostics;
  std::vector<JointResidualPointDiagnostics> worst_points;
  std::vector<JointResidualBoardObservationDiagnostics> worst_board_observations;
  std::vector<JointResidualBoardDiagnostics> worst_boards;
  std::vector<JointResidualFrameDiagnostics> worst_frames;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class JointReprojectionResidualEvaluator {
 public:
  explicit JointReprojectionResidualEvaluator(
      JointResidualEvaluationOptions options = JointResidualEvaluationOptions{});

  JointResidualEvaluationResult Evaluate(
      const JointMeasurementBuildResult& measurement_result,
      const JointReprojectionSceneState& scene_state) const;

  JointResidualEvaluationResult Evaluate(
      const JointMeasurementBuildResult& measurement_result,
      const OuterBootstrapResult& bootstrap_result) const;

  void DrawFrameOverlay(const cv::Mat& image,
                        int frame_index,
                        const JointResidualEvaluationResult& evaluation_result,
                        cv::Mat* output_image) const;

  const JointResidualEvaluationOptions& options() const { return options_; }

 private:
  const JointResidualFrameDiagnostics* FindFrameDiagnostics(
      const JointResidualEvaluationResult& evaluation_result,
      int frame_index) const;

  JointResidualEvaluationOptions options_;
  JointReprojectionCostCore cost_core_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_RESIDUAL_EVALUATOR_HPP
