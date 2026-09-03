#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_COST_CORE_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_COST_CORE_HPP

#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/AngularResidualGeometry.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionMeasurementBuilder.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct JointSceneBoardState {
  int board_id = -1;
  bool initialized = false;
  Eigen::Matrix4d T_reference_board = Eigen::Matrix4d::Identity();
  int observation_count = 0;
  double rmse = 0.0;
};

struct JointSceneFrameState {
  int frame_index = -1;
  std::string frame_label;
  bool initialized = false;
  std::vector<int> visible_board_ids;
  Eigen::Matrix4d T_camera_reference = Eigen::Matrix4d::Identity();
  int observation_count = 0;
  double rmse = 0.0;
};

struct JointReprojectionSceneState {
  int reference_board_id = 1;
  OuterBootstrapCameraIntrinsics camera;
  std::vector<JointSceneBoardState> boards;
  std::vector<JointSceneFrameState> frames;
  std::vector<std::string> warnings;

  bool IsValid() const { return camera.IsValid(); }
};

struct JointReprojectionCostOptions {
  bool uniform_control_point_mode = false;
  ResidualModel residual_model = ResidualModel::ImagePlane;
  double hybrid_angular_threshold_deg = 50.0;
  double polar_continuous_hybrid_threshold_deg = 50.0;
  double polar_continuous_hybrid_temperature_deg = 10.0;
  double quality_weight_floor = 0.1;
  double outer_huber_delta_pixels = 10.0;
  double internal_huber_delta_pixels = 6.0;
  double outer_huber_delta_radians = 0.02;
  double internal_huber_delta_radians = 0.015;
  bool enable_invalid_projection_penalty = true;
  double invalid_projection_penalty_pixels = 100.0;
  double invalid_projection_penalty_radians = 0.35;
  std::string polar_angle_weight_mode = "none";
  std::vector<double> polar_angle_weight_bin_edges_deg = {0.0, 30.0, 50.0, 70.0, 85.0, 100.0};
  std::vector<double> polar_angle_weight_fixed_bin_scales = {1.0, 1.0, 1.0, 1.0, 1.0};
  double polar_angle_weight_adaptive_sigma_reference_deg = 50.0;
  double polar_angle_weight_adaptive_sigma_growth = 1.0;
  double polar_angle_weight_min_scale = 0.25;
  bool enable_angular_residual_diagnostics = false;
  std::vector<double> angular_residual_bin_edges_deg = {0.0, 30.0, 50.0, 70.0, 85.0, 100.0};
  bool multi_board_consistency_weighting = false;
  bool consistency_apply_to_outer = true;
  bool consistency_apply_to_internal = true;
};

struct JointCostPointEvaluation {
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
  Eigen::Vector3d observed_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d predicted_ray = Eigen::Vector3d::Zero();
  Eigen::Vector2d angular_residual_xy = Eigen::Vector2d::Zero();
  double angular_residual_norm = 0.0;
  ResidualModel residual_model_used = ResidualModel::ImagePlane;
  double quality = 0.0;
  bool used_in_solver = false;
  bool valid_projection = false;
  bool valid_angular_projection = false;
  double balance_weight = 0.0;
  double quality_weight = 0.0;
  double consistency_weight = 1.0;
  double huber_weight = 0.0;
  double polar_angle_weight = 1.0;
  double polar_angle_deg = 0.0;
  double final_weight = 0.0;
  double weighted_squared_error = 0.0;
  double active_image_plane_weight = 1.0;
  double active_angular_weight = 0.0;
  int frame_storage_index = -1;
  int source_board_observation_index = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind = JointObservationSourceKind::OuterMeasurement;
};

struct JointCostBoardObservationEvaluation {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double squared_error_sum = 0.0;
  double rmse = 0.0;
  double cost = 0.0;
  double average_quality = 0.0;
};

struct JointCostEvaluation {
  bool success = false;
  int reference_board_id = 1;
  std::vector<JointCostPointEvaluation> point_evaluations;
  std::vector<JointCostBoardObservationEvaluation> board_observation_evaluations;
  double total_squared_error = 0.0;
  double total_cost = 0.0;
  double total_image_squared_error = 0.0;
  double total_angular_squared_error = 0.0;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double overall_rmse = 0.0;
  double outer_rmse = 0.0;
  double internal_rmse = 0.0;
  double overall_image_plane_rmse = 0.0;
  double outer_image_plane_rmse = 0.0;
  double internal_image_plane_rmse = 0.0;
  double overall_angular_rmse = 0.0;
  double outer_angular_rmse = 0.0;
  double internal_angular_rmse = 0.0;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

JointReprojectionSceneState BuildSceneStateFromBootstrap(
    const OuterBootstrapResult& bootstrap_result);

const JointSceneFrameState* FindJointSceneFrameState(
    const JointReprojectionSceneState& scene_state,
    int frame_index);
const JointSceneBoardState* FindJointSceneBoardState(
    const JointReprojectionSceneState& scene_state,
    int board_id);
JointSceneFrameState* FindJointSceneFrameState(
    JointReprojectionSceneState* scene_state,
    int frame_index);
JointSceneBoardState* FindJointSceneBoardState(
    JointReprojectionSceneState* scene_state,
    int board_id);

IntermediateCameraConfig MakeIntermediateCameraConfig(
    const OuterBootstrapCameraIntrinsics& intrinsics);
Eigen::Isometry3d ToIsometry3d(const Eigen::Matrix4d& matrix);
Eigen::Matrix4d ToMatrix4d(const Eigen::Isometry3d& transform);
Eigen::Isometry3d ApplyPoseDelta(const Eigen::Isometry3d& pose,
                                 const Eigen::Matrix<double, 6, 1>& delta);
double ComputePoseDeltaNorm(const Eigen::Isometry3d& from,
                            const Eigen::Isometry3d& to);

bool EstimatePoseFromObjectPoints(const OuterBootstrapCameraIntrinsics& intrinsics,
                                  const std::vector<Eigen::Vector3d>& object_points,
                                  const std::vector<cv::Point2f>& image_points,
                                  Eigen::Isometry3d* pose,
                                  double* rmse);

// Initialization-only pose validation. Unlike the legacy helper above this
// never falls back to a pinhole solve or assigns a penalty to invalid
// projections. All supplied observations must be valid in the active camera
// model and the resulting pose must satisfy the supplied RMSE bound.
bool EstimatePoseFromObjectPointsStrict(
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>& image_points,
    double max_rmse,
    Eigen::Isometry3d* pose,
    double* rmse);

struct TransformCandidate {
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  double weight = 1.0;
};

Eigen::Isometry3d AverageTransforms(const std::vector<TransformCandidate>& candidates);

Eigen::VectorXd BuildWeightedResidualVector(const JointCostEvaluation& evaluation);
Eigen::VectorXd BuildWeightedResidualVectorForFrame(const JointCostEvaluation& evaluation,
                                                    int frame_index);
Eigen::VectorXd BuildWeightedResidualVectorForBoard(const JointCostEvaluation& evaluation,
                                                    int board_id);

class JointReprojectionCostCore {
 public:
  explicit JointReprojectionCostCore(
      JointReprojectionCostOptions options = JointReprojectionCostOptions{});

  JointCostEvaluation Evaluate(
      const JointMeasurementBuildResult& measurement_result,
      const JointReprojectionSceneState& scene_state) const;

  const JointReprojectionCostOptions& options() const { return options_; }

 private:
  JointReprojectionCostOptions options_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_COST_CORE_HPP
