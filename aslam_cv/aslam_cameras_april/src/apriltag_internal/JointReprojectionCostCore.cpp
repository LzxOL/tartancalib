#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <Eigen/Eigenvalues>

#include <opencv2/calib3d.hpp>

#include <aslam/cameras/apriltag_internal/AngularResidualGeometry.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/PolarAngleResidualDiagnostics.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

double ComputeRmseFromSquaredSum(double squared_sum, int count) {
  if (count <= 0) {
    return 0.0;
  }
  return std::sqrt(squared_sum / static_cast<double>(count));
}

void AppendUniqueWarning(const std::string& warning,
                         std::vector<std::string>* warnings) {
  if (warnings == nullptr || warning.empty()) {
    return;
  }
  if (std::find(warnings->begin(), warnings->end(), warning) == warnings->end()) {
    warnings->push_back(warning);
  }
}

std::vector<cv::Point3f> ToCvObjectPoints(const std::vector<Eigen::Vector3d>& points) {
  std::vector<cv::Point3f> cv_points;
  cv_points.reserve(points.size());
  for (const Eigen::Vector3d& point : points) {
    cv_points.push_back(cv::Point3f(static_cast<float>(point.x()),
                                    static_cast<float>(point.y()),
                                    static_cast<float>(point.z())));
  }
  return cv_points;
}

Eigen::Isometry3d PoseFromCv(const cv::Mat& rvec, const cv::Mat& tvec) {
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);

  Eigen::Matrix3d rotation_eigen = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      rotation_eigen(row, col) = rotation.at<double>(row, col);
    }
    translation[row] = tvec.at<double>(row, 0);
  }

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = rotation_eigen;
  pose.translation() = translation;
  return pose;
}

bool EvaluatePoseRmseWithSoftPenalty(const OuterBootstrapCameraIntrinsics& intrinsics,
                                     const std::vector<Eigen::Vector3d>& object_points,
                                     const std::vector<cv::Point2f>& image_points,
                                     const Eigen::Isometry3d& pose,
                                     double* rmse) {
  if (rmse == nullptr) {
    throw std::runtime_error("EvaluatePoseRmseWithSoftPenalty requires a valid output pointer.");
  }
  if (object_points.size() != image_points.size() || object_points.size() < 4) {
    return false;
  }

  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(intrinsics));
  constexpr double kInvalidProjectionPenalty = 100.0;
  double squared_error_sum = 0.0;
  int valid_projection_count = 0;
  for (std::size_t index = 0; index < object_points.size(); ++index) {
    Eigen::Vector2d projected = Eigen::Vector2d::Zero();
    if (!camera.vsEuclideanToKeypoint(pose * object_points[index], &projected)) {
      squared_error_sum += 2.0 * kInvalidProjectionPenalty * kInvalidProjectionPenalty;
      continue;
    }
    const double dx = projected.x() - static_cast<double>(image_points[index].x);
    const double dy = projected.y() - static_cast<double>(image_points[index].y);
    squared_error_sum += dx * dx + dy * dy;
    ++valid_projection_count;
  }

  if (valid_projection_count <= 0) {
    return false;
  }

  *rmse = std::sqrt(squared_error_sum / static_cast<double>(object_points.size()));
  return true;
}

bool EstimatePoseFromObjectPointsPinholeFallback(
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>& image_points,
    Eigen::Isometry3d* pose,
    double* rmse) {
  if (pose == nullptr || rmse == nullptr) {
    throw std::runtime_error(
        "EstimatePoseFromObjectPointsPinholeFallback requires valid output pointers.");
  }
  if (object_points.size() != image_points.size() || object_points.size() < 4) {
    return false;
  }

  cv::Mat rvec;
  cv::Mat tvec;
  const cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << intrinsics.fu, 0.0, intrinsics.cu,
                                 0.0, intrinsics.fv, intrinsics.cv,
                                 0.0, 0.0, 1.0);
  const cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

  bool success = false;
  const std::vector<cv::Point3f> cv_object_points = ToCvObjectPoints(object_points);
  if (cv_object_points.size() == 4) {
    success = cv::solvePnP(cv_object_points, image_points, camera_matrix, dist_coeffs,
                           rvec, tvec, false, cv::SOLVEPNP_IPPE);
  }
  if (!success) {
    success = cv::solvePnP(cv_object_points, image_points, camera_matrix, dist_coeffs,
                           rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
  }
  if (!success) {
    return false;
  }

  success = cv::solvePnP(cv_object_points, image_points, camera_matrix, dist_coeffs,
                         rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
  if (!success) {
    return false;
  }

  cv::Mat tvec64;
  tvec.convertTo(tvec64, CV_64F);
  if (tvec64.at<double>(2, 0) <= 0.0) {
    return false;
  }

  const Eigen::Isometry3d candidate_pose = PoseFromCv(rvec, tvec);
  double candidate_rmse = 0.0;
  if (!EvaluatePoseRmseWithSoftPenalty(intrinsics, object_points, image_points,
                                       candidate_pose, &candidate_rmse)) {
    return false;
  }

  *pose = candidate_pose;
  *rmse = candidate_rmse;
  return true;
}

double HuberWeight(double residual_norm, double delta) {
  if (!(delta > 0.0) || residual_norm <= 0.0 || residual_norm <= delta) {
    return 1.0;
  }
  return delta / residual_norm;
}

Eigen::Vector2d InvalidAngularResidual(double penalty_radians) {
  return Eigen::Vector2d::Constant(penalty_radians);
}

double ComputePolarAngleWeightScale(
    const DoubleSphereCameraModel& camera,
    const JointPointObservation& observation,
    const JointReprojectionCostOptions& options) {
  if ((!options.uniform_control_point_mode &&
       observation.point_type != JointPointType::Internal) ||
      options.polar_angle_weight_mode == "none" ||
      options.polar_angle_weight_mode == "diagnostic_only") {
    return 1.0;
  }

  const double polar_angle = ComputePolarAngleDeg(camera, observation.image_xy);
  if (!std::isfinite(polar_angle)) {
    return 1.0;
  }
  const double min_scale =
      std::max(0.0, std::min(1.0, options.polar_angle_weight_min_scale));

  if (options.polar_angle_weight_mode == "fixed_bins") {
    const std::vector<double>& edges = options.polar_angle_weight_bin_edges_deg;
    const std::vector<double>& scales = options.polar_angle_weight_fixed_bin_scales;
    const std::size_t bin_count = edges.size() >= 2 ? edges.size() - 1 : 0;
    for (std::size_t i = 0; i < bin_count && i < scales.size(); ++i) {
      if (polar_angle >= edges[i] && polar_angle < edges[i + 1]) {
        return std::max(min_scale, std::min(1.0, scales[i]));
      }
    }
    return 1.0;
  }

  const double reference_deg = options.polar_angle_weight_adaptive_sigma_reference_deg;
  const double growth = std::max(1e-6, options.polar_angle_weight_adaptive_sigma_growth);
  if (polar_angle <= reference_deg) {
    return 1.0;
  }
  const double normalized =
      (polar_angle - reference_deg) / std::max(1e-6, 90.0 - reference_deg);
  const double scale = 1.0 / (1.0 + growth * normalized * normalized);
  return std::max(min_scale, std::min(1.0, scale));
}

}  // namespace

JointReprojectionSceneState BuildSceneStateFromBootstrap(
    const OuterBootstrapResult& bootstrap_result) {
  JointReprojectionSceneState scene_state;
  scene_state.reference_board_id = bootstrap_result.reference_board_id;
  scene_state.camera = bootstrap_result.coarse_camera;
  scene_state.warnings = bootstrap_result.warnings;

  scene_state.boards.reserve(bootstrap_result.boards.size());
  for (const OuterBootstrapBoardState& board_state : bootstrap_result.boards) {
    JointSceneBoardState scene_board;
    scene_board.board_id = board_state.board_id;
    scene_board.initialized = board_state.initialized;
    scene_board.T_reference_board = board_state.T_reference_board;
    scene_board.observation_count = board_state.observation_count;
    scene_board.rmse = board_state.rmse;
    scene_state.boards.push_back(scene_board);
  }

  scene_state.frames.reserve(bootstrap_result.frames.size());
  for (const OuterBootstrapFrameState& frame_state : bootstrap_result.frames) {
    JointSceneFrameState scene_frame;
    scene_frame.frame_index = frame_state.frame_index;
    scene_frame.frame_label = frame_state.frame_label;
    scene_frame.initialized = frame_state.initialized;
    scene_frame.visible_board_ids = frame_state.visible_board_ids;
    scene_frame.T_camera_reference = frame_state.T_camera_reference;
    scene_frame.observation_count = frame_state.observation_count;
    scene_frame.rmse = frame_state.rmse;
    scene_state.frames.push_back(scene_frame);
  }
  return scene_state;
}

const JointSceneFrameState* FindJointSceneFrameState(
    const JointReprojectionSceneState& scene_state,
    int frame_index) {
  for (const JointSceneFrameState& frame_state : scene_state.frames) {
    if (frame_state.frame_index == frame_index) {
      return &frame_state;
    }
  }
  return nullptr;
}

const JointSceneBoardState* FindJointSceneBoardState(
    const JointReprojectionSceneState& scene_state,
    int board_id) {
  for (const JointSceneBoardState& board_state : scene_state.boards) {
    if (board_state.board_id == board_id) {
      return &board_state;
    }
  }
  return nullptr;
}

JointSceneFrameState* FindJointSceneFrameState(
    JointReprojectionSceneState* scene_state,
    int frame_index) {
  if (scene_state == nullptr) {
    return nullptr;
  }
  for (JointSceneFrameState& frame_state : scene_state->frames) {
    if (frame_state.frame_index == frame_index) {
      return &frame_state;
    }
  }
  return nullptr;
}

JointSceneBoardState* FindJointSceneBoardState(
    JointReprojectionSceneState* scene_state,
    int board_id) {
  if (scene_state == nullptr) {
    return nullptr;
  }
  for (JointSceneBoardState& board_state : scene_state->boards) {
    if (board_state.board_id == board_id) {
      return &board_state;
    }
  }
  return nullptr;
}

IntermediateCameraConfig MakeIntermediateCameraConfig(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  IntermediateCameraConfig config;
  config.camera_model = intrinsics.camera_model;
  config.distortion_model = intrinsics.distortion_model;
  config.intrinsics = intrinsics.IntrinsicsVector();
  config.distortion_coeffs = intrinsics.DistortionVector();
  config.resolution = {intrinsics.resolution.width, intrinsics.resolution.height};
  return config;
}

Eigen::Isometry3d ToIsometry3d(const Eigen::Matrix4d& matrix) {
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.matrix() = matrix;
  return transform;
}

Eigen::Matrix4d ToMatrix4d(const Eigen::Isometry3d& transform) {
  return transform.matrix();
}

Eigen::Isometry3d ApplyPoseDelta(const Eigen::Isometry3d& pose,
                                 const Eigen::Matrix<double, 6, 1>& delta) {
  Eigen::Isometry3d updated = pose;
  updated.translation() += delta.head<3>();

  const Eigen::Vector3d rotation_delta = delta.tail<3>();
  const double angle = rotation_delta.norm();
  if (angle > 1e-12) {
    const Eigen::AngleAxisd angle_axis(angle, rotation_delta / angle);
    updated.linear() = angle_axis.toRotationMatrix() * updated.linear();
  }
  return updated;
}

double ComputePoseDeltaNorm(const Eigen::Isometry3d& from,
                            const Eigen::Isometry3d& to) {
  const Eigen::Vector3d translation_delta = to.translation() - from.translation();
  const Eigen::Matrix3d relative_rotation = to.linear() * from.linear().transpose();
  Eigen::AngleAxisd angle_axis(relative_rotation);
  return std::sqrt(translation_delta.squaredNorm() +
                   angle_axis.angle() * angle_axis.angle());
}

bool EstimatePoseFromObjectPoints(const OuterBootstrapCameraIntrinsics& intrinsics,
                                  const std::vector<Eigen::Vector3d>& object_points,
                                  const std::vector<cv::Point2f>& image_points,
                                  Eigen::Isometry3d* pose,
                                  double* rmse) {
  if (pose == nullptr || rmse == nullptr) {
    throw std::runtime_error("EstimatePoseFromObjectPoints requires valid output pointers.");
  }
  if (object_points.size() != image_points.size() || object_points.size() < 4) {
    return false;
  }

  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(intrinsics));
  cv::Mat rvec;
  cv::Mat tvec;
  if (!camera.estimateTransformation(ToCvObjectPoints(object_points), image_points, &rvec, &tvec)) {
    return EstimatePoseFromObjectPointsPinholeFallback(
        intrinsics, object_points, image_points, pose, rmse);
  }

  const Eigen::Isometry3d candidate_pose = PoseFromCv(rvec, tvec);
  double candidate_rmse = 0.0;
  if (!EvaluatePoseRmseWithSoftPenalty(intrinsics, object_points, image_points,
                                       candidate_pose, &candidate_rmse)) {
    return EstimatePoseFromObjectPointsPinholeFallback(
        intrinsics, object_points, image_points, pose, rmse);
  }

  *pose = candidate_pose;
  *rmse = candidate_rmse;
  return true;
}

Eigen::Isometry3d AverageTransforms(const std::vector<TransformCandidate>& candidates) {
  if (candidates.empty()) {
    return Eigen::Isometry3d::Identity();
  }
  if (candidates.size() == 1) {
    return candidates.front().transform;
  }

  Eigen::Matrix4d quaternion_scatter = Eigen::Matrix4d::Zero();
  Eigen::Vector3d translation_sum = Eigen::Vector3d::Zero();
  double weight_sum = 0.0;

  Eigen::Quaterniond reference_quaternion(candidates.front().transform.linear());
  reference_quaternion.normalize();
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const double weight = std::max(1e-9, candidates[index].weight);
    Eigen::Quaterniond quaternion(candidates[index].transform.linear());
    quaternion.normalize();
    if (quaternion.dot(reference_quaternion) < 0.0) {
      quaternion.coeffs() *= -1.0;
    }
    const Eigen::Vector4d coeffs(quaternion.w(), quaternion.x(),
                                 quaternion.y(), quaternion.z());
    quaternion_scatter += weight * coeffs * coeffs.transpose();
    translation_sum += weight * candidates[index].transform.translation();
    weight_sum += weight;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(quaternion_scatter);
  const Eigen::Vector4d eigenvector = solver.eigenvectors().col(3);
  Eigen::Quaterniond average_quaternion(
      eigenvector[0], eigenvector[1], eigenvector[2], eigenvector[3]);
  average_quaternion.normalize();

  Eigen::Isometry3d averaged = Eigen::Isometry3d::Identity();
  averaged.linear() = average_quaternion.toRotationMatrix();
  averaged.translation() = translation_sum / std::max(1e-9, weight_sum);
  return averaged;
}

Eigen::VectorXd BuildWeightedResidualVector(const JointCostEvaluation& evaluation) {
  Eigen::VectorXd residuals = Eigen::VectorXd::Zero(
      2 * static_cast<int>(evaluation.point_evaluations.size()));
  int row = 0;
  for (const JointCostPointEvaluation& point : evaluation.point_evaluations) {
    const double scale = std::sqrt(std::max(0.0, point.final_weight));
    const Eigen::Vector2d active_residual =
        point.residual_model_used == ResidualModel::SphereAngular
            ? point.angular_residual_xy
            : point.residual_xy;
    residuals[row++] = scale * active_residual.x();
    residuals[row++] = scale * active_residual.y();
  }
  return residuals;
}

Eigen::VectorXd BuildWeightedResidualVectorForFrame(const JointCostEvaluation& evaluation,
                                                    int frame_index) {
  int count = 0;
  for (const JointCostPointEvaluation& point : evaluation.point_evaluations) {
    count += point.frame_index == frame_index ? 1 : 0;
  }
  Eigen::VectorXd residuals = Eigen::VectorXd::Zero(2 * count);
  int row = 0;
  for (const JointCostPointEvaluation& point : evaluation.point_evaluations) {
    if (point.frame_index != frame_index) {
      continue;
    }
    const double scale = std::sqrt(std::max(0.0, point.final_weight));
    const Eigen::Vector2d active_residual =
        point.residual_model_used == ResidualModel::SphereAngular
            ? point.angular_residual_xy
            : point.residual_xy;
    residuals[row++] = scale * active_residual.x();
    residuals[row++] = scale * active_residual.y();
  }
  return residuals;
}

Eigen::VectorXd BuildWeightedResidualVectorForBoard(const JointCostEvaluation& evaluation,
                                                    int board_id) {
  int count = 0;
  for (const JointCostPointEvaluation& point : evaluation.point_evaluations) {
    count += point.board_id == board_id ? 1 : 0;
  }
  Eigen::VectorXd residuals = Eigen::VectorXd::Zero(2 * count);
  int row = 0;
  for (const JointCostPointEvaluation& point : evaluation.point_evaluations) {
    if (point.board_id != board_id) {
      continue;
    }
    const double scale = std::sqrt(std::max(0.0, point.final_weight));
    const Eigen::Vector2d active_residual =
        point.residual_model_used == ResidualModel::SphereAngular
            ? point.angular_residual_xy
            : point.residual_xy;
    residuals[row++] = scale * active_residual.x();
    residuals[row++] = scale * active_residual.y();
  }
  return residuals;
}

JointReprojectionCostCore::JointReprojectionCostCore(JointReprojectionCostOptions options)
    : options_(std::move(options)) {}

JointCostEvaluation JointReprojectionCostCore::Evaluate(
    const JointMeasurementBuildResult& measurement_result,
    const JointReprojectionSceneState& scene_state) const {
  JointCostEvaluation result;
  result.reference_board_id = scene_state.reference_board_id;

  if (!measurement_result.success) {
    result.failure_reason = "measurement_result.success is false";
    return result;
  }
  if (!scene_state.IsValid()) {
    result.failure_reason = "scene_state camera is not valid";
    return result;
  }

  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(scene_state.camera));
  if (!camera.IsValid()) {
    result.failure_reason = "Failed to construct valid DS camera";
    return result;
  }

  struct ObservationBudget {
    int outer_count = 0;
    int internal_count = 0;
  };
  std::map<std::pair<int, int>, ObservationBudget> budgets;
  for (const JointPointObservation& point : measurement_result.solver_observations) {
    if (!point.used_in_solver) {
      continue;
    }
    ObservationBudget& budget = budgets[std::make_pair(point.frame_index, point.board_id)];
    if (point.point_type == JointPointType::Outer) {
      ++budget.outer_count;
    } else {
      ++budget.internal_count;
    }
  }

  if (budgets.empty()) {
    result.failure_reason = "No used_in_solver observations available for cost evaluation";
    return result;
  }

  std::map<std::pair<int, int>, JointCostBoardObservationEvaluation> board_observation_map;
  double total_outer_squared_sum = 0.0;
  int total_outer_count = 0;
  double total_internal_squared_sum = 0.0;
  int total_internal_count = 0;
  double total_outer_image_squared_sum = 0.0;
  double total_internal_image_squared_sum = 0.0;
  int total_outer_image_count = 0;
  int total_internal_image_count = 0;
  double total_outer_angular_squared_sum = 0.0;
  double total_internal_angular_squared_sum = 0.0;
  int total_outer_angular_count = 0;
  int total_internal_angular_count = 0;

  result.point_evaluations.reserve(measurement_result.solver_observations.size());
  for (const JointPointObservation& observation : measurement_result.solver_observations) {
    if (!observation.used_in_solver) {
      continue;
    }

    const JointSceneFrameState* frame_state =
        FindJointSceneFrameState(scene_state, observation.frame_index);
    if (frame_state == nullptr || !frame_state->initialized) {
      std::ostringstream warning;
      warning << "Missing or uninitialized frame " << observation.frame_index
              << " during cost evaluation";
      AppendUniqueWarning(warning.str(), &result.warnings);
      continue;
    }

    Eigen::Matrix4d T_reference_board = Eigen::Matrix4d::Identity();
    if (observation.board_id != scene_state.reference_board_id) {
      const JointSceneBoardState* board_state =
          FindJointSceneBoardState(scene_state, observation.board_id);
      if (board_state == nullptr || !board_state->initialized) {
        std::ostringstream warning;
        warning << "Missing or uninitialized board " << observation.board_id
                << " during cost evaluation";
        AppendUniqueWarning(warning.str(), &result.warnings);
        continue;
      }
      T_reference_board = board_state->T_reference_board;
    }

    const Eigen::Vector4d point_board(observation.target_xyz_board.x(),
                                      observation.target_xyz_board.y(),
                                      observation.target_xyz_board.z(),
                                      1.0);
    const Eigen::Vector4d point_camera_h =
        frame_state->T_camera_reference * (T_reference_board * point_board);
    const Eigen::Vector3d point_camera = point_camera_h.head<3>();

    JointCostPointEvaluation point_eval;
    point_eval.frame_index = observation.frame_index;
    point_eval.frame_label = observation.frame_label;
    point_eval.board_id = observation.board_id;
    point_eval.point_id = observation.point_id;
    point_eval.point_type = observation.point_type;
    point_eval.observed_image_xy = observation.image_xy;
    point_eval.target_xyz_board = observation.target_xyz_board;
    point_eval.quality = observation.quality;
    point_eval.used_in_solver = observation.used_in_solver;
    point_eval.frame_storage_index = observation.frame_storage_index;
    point_eval.source_board_observation_index = observation.source_board_observation_index;
    point_eval.source_point_index = observation.source_point_index;
    point_eval.source_kind = observation.source_kind;

    if (!camera.vsEuclideanToKeypoint(point_camera, &point_eval.predicted_image_xy)) {
      point_eval.valid_projection = false;
      point_eval.predicted_image_xy =
          Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN());
      if (options_.enable_invalid_projection_penalty) {
        point_eval.residual_xy =
            Eigen::Vector2d(options_.invalid_projection_penalty_pixels,
                            options_.invalid_projection_penalty_pixels);
      } else {
        point_eval.residual_xy = Eigen::Vector2d::Zero();
      }
    } else {
      point_eval.valid_projection = true;
      point_eval.residual_xy = point_eval.predicted_image_xy - point_eval.observed_image_xy;
    }
    point_eval.residual_norm = point_eval.residual_xy.norm();

    AngularObservationGeometry angular_observation_geometry;
    const bool have_angular_observation_geometry =
        ComputeAngularObservationGeometry(
            camera, point_eval.observed_image_xy, &angular_observation_geometry);
    AngularPredictionGeometry angular_prediction_geometry;
    const bool have_angular_prediction_geometry =
        ComputeAngularPredictionGeometry(
            camera, point_camera, &angular_prediction_geometry);
    point_eval.valid_angular_projection =
        have_angular_observation_geometry && have_angular_prediction_geometry;
    if (have_angular_observation_geometry) {
      point_eval.observed_ray = angular_observation_geometry.observed_ray;
      point_eval.polar_angle_deg = angular_observation_geometry.polar_angle_deg;
    } else {
      point_eval.polar_angle_deg = ComputePolarAngleDeg(camera, observation.image_xy);
    }
    if (have_angular_prediction_geometry) {
      point_eval.predicted_ray = angular_prediction_geometry.predicted_ray;
    }
    if (point_eval.valid_angular_projection) {
      point_eval.angular_residual_xy =
          ComputeAngularResidualTangent(
              angular_observation_geometry, angular_prediction_geometry);
    } else if (options_.enable_invalid_projection_penalty) {
      point_eval.angular_residual_xy =
          InvalidAngularResidual(options_.invalid_projection_penalty_radians);
    } else {
      point_eval.angular_residual_xy = Eigen::Vector2d::Zero();
    }
    point_eval.angular_residual_norm =
        ComputeAngularResidualNorm(point_eval.angular_residual_xy);

    const std::pair<int, int> observation_key(
        observation.frame_index, observation.board_id);
    const ObservationBudget& budget = budgets[observation_key];
    const bool has_outer = budget.outer_count > 0;
    const bool has_internal = budget.internal_count > 0;
    double type_budget = 1.0;
    int type_count = 1;
    if (options_.uniform_control_point_mode) {
      type_count = 1;
    } else if (has_outer && has_internal) {
      type_budget = observation.point_type == JointPointType::Outer ? 0.5 : 0.5;
      type_count = observation.point_type == JointPointType::Outer ?
          budget.outer_count : budget.internal_count;
    } else if (observation.point_type == JointPointType::Outer) {
      type_budget = 1.0;
      type_count = budget.outer_count;
    } else {
      type_budget = 1.0;
      type_count = budget.internal_count;
    }
    point_eval.balance_weight =
        type_budget / std::max(1, type_count);
    point_eval.quality_weight =
        std::max(0.0, observation.final_observation_weight);
    point_eval.consistency_weight =
        std::max(0.0, observation.consistency_weight);
    point_eval.polar_angle_weight =
        ComputePolarAngleWeightScale(camera, observation, options_);
    const bool use_continuous_hybrid =
        options_.residual_model == ResidualModel::PolarContinuousHybrid;
    const bool use_normalized_angular =
        options_.residual_model == ResidualModel::NormalizedSphereAngular;
    const bool use_angular_residual =
        !use_continuous_hybrid &&
        ShouldUseAngularResidual(
            options_.residual_model,
            point_eval.polar_angle_deg,
            options_.hybrid_angular_threshold_deg);
    if (use_continuous_hybrid) {
      point_eval.active_angular_weight = ComputePolarContinuousAngularWeight(
          point_eval.polar_angle_deg,
          options_.polar_continuous_hybrid_threshold_deg,
          options_.polar_continuous_hybrid_temperature_deg);
      point_eval.active_image_plane_weight =
          1.0 - point_eval.active_angular_weight;
    } else {
      point_eval.active_angular_weight = use_angular_residual ? 1.0 : 0.0;
      point_eval.active_image_plane_weight = use_angular_residual ? 0.0 : 1.0;
    }
    point_eval.residual_model_used =
        (use_angular_residual || point_eval.active_angular_weight >= 0.5)
            ? (use_normalized_angular ? ResidualModel::NormalizedSphereAngular
                                      : ResidualModel::SphereAngular)
            : ResidualModel::ImagePlane;
    const Eigen::Vector2d active_residual =
        point_eval.active_image_plane_weight * point_eval.residual_xy +
        point_eval.active_angular_weight * point_eval.angular_residual_xy;
    const double active_squared_norm = active_residual.squaredNorm();
    const double active_residual_norm = std::sqrt(std::max(0.0, active_squared_norm));
    const double huber_delta = options_.uniform_control_point_mode
        ? (use_angular_residual
               ? std::min(options_.outer_huber_delta_radians,
                          options_.internal_huber_delta_radians)
               : std::min(options_.outer_huber_delta_pixels,
                          options_.internal_huber_delta_pixels))
        : observation.point_type == JointPointType::Outer
              ? (use_angular_residual ? options_.outer_huber_delta_radians
                                      : options_.outer_huber_delta_pixels)
              : (use_angular_residual ? options_.internal_huber_delta_radians
                                      : options_.internal_huber_delta_pixels);
    point_eval.huber_weight = HuberWeight(active_residual_norm, huber_delta);
    point_eval.final_weight =
        point_eval.balance_weight * point_eval.quality_weight *
        point_eval.polar_angle_weight * point_eval.huber_weight;
    point_eval.weighted_squared_error =
        point_eval.final_weight * active_squared_norm;

    result.total_squared_error += active_squared_norm;
    result.total_cost += point_eval.weighted_squared_error;
    result.total_image_squared_error += point_eval.residual_xy.squaredNorm();
    result.total_angular_squared_error += point_eval.angular_residual_xy.squaredNorm();
    ++result.point_count;
    if (point_eval.point_type == JointPointType::Outer) {
      total_outer_squared_sum += active_squared_norm;
      ++total_outer_count;
      total_outer_image_squared_sum += point_eval.residual_xy.squaredNorm();
      ++total_outer_image_count;
      total_outer_angular_squared_sum += point_eval.angular_residual_xy.squaredNorm();
      ++total_outer_angular_count;
      ++result.outer_point_count;
    } else {
      total_internal_squared_sum += active_squared_norm;
      ++total_internal_count;
      total_internal_image_squared_sum += point_eval.residual_xy.squaredNorm();
      ++total_internal_image_count;
      total_internal_angular_squared_sum += point_eval.angular_residual_xy.squaredNorm();
      ++total_internal_angular_count;
      ++result.internal_point_count;
    }

    JointCostBoardObservationEvaluation& board_eval = board_observation_map[observation_key];
    board_eval.frame_index = observation.frame_index;
    board_eval.frame_label = observation.frame_label;
    board_eval.board_id = observation.board_id;
    ++board_eval.point_count;
    board_eval.average_quality += point_eval.quality;
    board_eval.squared_error_sum += active_squared_norm;
    board_eval.cost += point_eval.weighted_squared_error;
    if (point_eval.point_type == JointPointType::Outer) {
      ++board_eval.outer_point_count;
    } else {
      ++board_eval.internal_point_count;
    }

    result.point_evaluations.push_back(point_eval);
  }

  if (result.point_evaluations.empty()) {
    result.failure_reason = "No residual points could be evaluated";
    return result;
  }

  result.board_observation_evaluations.reserve(board_observation_map.size());
  for (auto& entry : board_observation_map) {
    JointCostBoardObservationEvaluation board_eval = entry.second;
    board_eval.average_quality /= std::max(1, board_eval.point_count);
    board_eval.rmse = ComputeRmseFromSquaredSum(board_eval.squared_error_sum,
                                               board_eval.point_count);
    result.board_observation_evaluations.push_back(board_eval);
  }

  result.overall_rmse = ComputeRmseFromSquaredSum(result.total_squared_error,
                                                  result.point_count);
  result.outer_rmse = ComputeRmseFromSquaredSum(total_outer_squared_sum,
                                                total_outer_count);
  result.internal_rmse = ComputeRmseFromSquaredSum(total_internal_squared_sum,
                                                   total_internal_count);
  result.overall_image_plane_rmse = ComputeRmseFromSquaredSum(
      result.total_image_squared_error, result.point_count);
  result.outer_image_plane_rmse = ComputeRmseFromSquaredSum(
      total_outer_image_squared_sum, total_outer_image_count);
  result.internal_image_plane_rmse = ComputeRmseFromSquaredSum(
      total_internal_image_squared_sum, total_internal_image_count);
  result.overall_angular_rmse = ComputeRmseFromSquaredSum(
      result.total_angular_squared_error, result.point_count);
  result.outer_angular_rmse = ComputeRmseFromSquaredSum(
      total_outer_angular_squared_sum, total_outer_angular_count);
  result.internal_angular_rmse = ComputeRmseFromSquaredSum(
      total_internal_angular_squared_sum, total_internal_angular_count);
  result.success = true;
  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
