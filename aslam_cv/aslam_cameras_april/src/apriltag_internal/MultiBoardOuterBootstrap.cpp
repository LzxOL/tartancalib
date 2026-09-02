#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

struct ObservationRecord {
  int frame_storage_index = -1;
  int board_id = -1;
  double quality = 0.0;
  std::array<cv::Point2f, 4> image_points{};
  bool reference_connected = false;
};

struct FrameNode {
  int frame_index = -1;
  std::string frame_label;
  cv::Size image_size;
  std::vector<int> requested_board_ids;
  std::vector<int> visible_board_ids;
  std::vector<int> observation_indices;
  bool reference_connected = false;
  bool initialized = false;
  Eigen::Isometry3d T_camera_reference = Eigen::Isometry3d::Identity();
  int connected_observation_count = 0;
  double rmse = std::numeric_limits<double>::infinity();
};

struct BoardNode {
  int board_id = -1;
  std::vector<int> observation_indices;
  bool reference_connected = false;
  bool initialized = false;
  Eigen::Isometry3d T_reference_board = Eigen::Isometry3d::Identity();
  int connected_observation_count = 0;
  double rmse = std::numeric_limits<double>::infinity();
};

struct SolverState {
  cv::Size image_size;
  std::vector<FrameNode> frames;
  std::vector<int> ordered_board_ids;
  std::map<int, BoardNode> boards;
  std::vector<ObservationRecord> observations;
  std::map<int, std::array<Eigen::Vector3d, 4> > board_corner_points;
};

struct TransformCandidate {
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  double weight = 1.0;
};

bool RefineBoardPoseFromInitializedFrames(const SolverState& state,
                                          const OuterBootstrapCameraIntrinsics& intrinsics,
                                          const std::vector<FrameNode>& frames,
                                          int board_id,
                                          bool robust_consensus,
                                          Eigen::Isometry3d* pose,
                                          double* rmse);

IntermediateCameraConfig MakeCameraConfig(const OuterBootstrapCameraIntrinsics& intrinsics) {
  IntermediateCameraConfig config;
  config.camera_model = intrinsics.camera_model;
  config.distortion_model = intrinsics.distortion_model;
  config.intrinsics = intrinsics.IntrinsicsVector();
  config.distortion_coeffs = intrinsics.DistortionVector();
  config.resolution.push_back(intrinsics.resolution.width);
  config.resolution.push_back(intrinsics.resolution.height);
  return config;
}

OuterBootstrapCameraIntrinsics MakeInitialIntrinsics(const cv::Size& resolution,
                                                     const OuterBootstrapOptions& options) {
  if (options.initial_camera.IsValid()) {
    OuterBootstrapCameraIntrinsics initial = options.initial_camera;
    initial.resolution = resolution;
    return initial;
  }
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model = "ds";
  intrinsics.distortion_model = "none";
  intrinsics.resolution = resolution;
  intrinsics.xi = options.init_xi;
  intrinsics.alpha = options.init_alpha;
  intrinsics.fu = options.init_fu_scale * static_cast<double>(resolution.width);
  intrinsics.fv = options.init_fv_scale * static_cast<double>(resolution.height);
  intrinsics.cu = 0.5 * static_cast<double>(resolution.width) + options.init_cu_offset;
  intrinsics.cv = 0.5 * static_cast<double>(resolution.height) + options.init_cv_offset;
  return intrinsics;
}

bool ClampIntrinsicsInPlace(OuterBootstrapCameraIntrinsics* intrinsics) {
  if (intrinsics == nullptr) {
    throw std::runtime_error("ClampIntrinsicsInPlace requires a valid pointer.");
  }
  intrinsics->fu = std::max(50.0, std::min(3.0 * intrinsics->resolution.width, intrinsics->fu));
  intrinsics->fv = std::max(50.0, std::min(3.0 * intrinsics->resolution.height, intrinsics->fv));
  intrinsics->cu =
      std::max(0.0, std::min(static_cast<double>(intrinsics->resolution.width), intrinsics->cu));
  intrinsics->cv =
      std::max(0.0, std::min(static_cast<double>(intrinsics->resolution.height), intrinsics->cv));
  const std::string family = intrinsics->NormalizedFamilyString();
  if (family == "ds-none") {
    intrinsics->xi = std::max(-0.95, std::min(2.5, intrinsics->xi));
    intrinsics->alpha = std::max(0.05, std::min(0.95, intrinsics->alpha));
    intrinsics->distortion_coeffs.clear();
  } else if (family == "eucm-none") {
    intrinsics->alpha = std::max(0.05, std::min(0.95, intrinsics->alpha));
    intrinsics->beta = std::max(0.05, std::min(2.5, intrinsics->beta));
    intrinsics->distortion_coeffs.clear();
  } else if (family == "pinhole-equi") {
    std::vector<double> clamped = intrinsics->distortion_coeffs;
    clamped.resize(4, 0.0);
    for (double& coeff : clamped) {
      coeff = std::max(-2.0, std::min(2.0, coeff));
    }
    intrinsics->distortion_coeffs = clamped;
  } else if (family == "omni-radtan") {
    intrinsics->xi = std::max(-0.95, std::min(3.0, intrinsics->xi));
    intrinsics->alpha = 0.0;
    intrinsics->beta = 0.0;
    std::vector<double> clamped = intrinsics->distortion_coeffs;
    clamped.resize(4, 0.0);
    for (double& coeff : clamped) {
      coeff = std::max(-2.0, std::min(2.0, coeff));
    }
    intrinsics->distortion_coeffs = clamped;
  } else if (family == "omni-none") {
    intrinsics->xi = std::max(-0.95, std::min(3.0, intrinsics->xi));
    intrinsics->alpha = 0.0;
    intrinsics->beta = 0.0;
    intrinsics->distortion_coeffs.clear();
  } else {
    return false;
  }
  return intrinsics->IsValid();
}

Eigen::VectorXd ToVector(const OuterBootstrapCameraIntrinsics& intrinsics) {
  const std::vector<double> parameters = intrinsics.CombinedParameterVector();
  Eigen::VectorXd vector(static_cast<int>(parameters.size()));
  for (int index = 0; index < vector.rows(); ++index) {
    vector[index] = parameters[static_cast<std::size_t>(index)];
  }
  return vector;
}

OuterBootstrapCameraIntrinsics FromVector(const Eigen::VectorXd& vector,
                                          const OuterBootstrapCameraIntrinsics& prototype) {
  OuterBootstrapCameraIntrinsics intrinsics = prototype;
  std::vector<double> parameters(static_cast<std::size_t>(vector.rows()));
  for (int index = 0; index < vector.rows(); ++index) {
    parameters[static_cast<std::size_t>(index)] = vector[index];
  }
  intrinsics.SetCombinedParameterVector(parameters);
  return intrinsics;
}

double ParameterStep(double value, double fallback_step) {
  return std::max(std::abs(value) * 1e-4, fallback_step);
}

Eigen::Isometry3d PoseFromCv(const cv::Mat& rvec, const cv::Mat& tvec) {
  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);

  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      rotation(row, col) = rotation_cv.at<double>(row, col);
    }
  }

  Eigen::Vector3d translation(
      tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0));
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = rotation;
  pose.translation() = translation;
  return pose;
}

Eigen::Matrix4d ToMatrix4d(const Eigen::Isometry3d& transform) {
  return transform.matrix();
}

std::array<Eigen::Vector3d, 4> BuildOuterCornerPoints(const ApriltagCanonicalModel& model) {
  const std::array<int, 4> point_ids{
      model.PointId(0, 0),
      model.PointId(model.ModuleDimension(), 0),
      model.PointId(model.ModuleDimension(), model.ModuleDimension()),
      model.PointId(0, model.ModuleDimension()),
  };
  std::array<Eigen::Vector3d, 4> points{};
  for (int index = 0; index < 4; ++index) {
    points[static_cast<std::size_t>(index)] = model.corner(point_ids[static_cast<std::size_t>(index)]).target_xyz;
  }
  return points;
}

std::vector<cv::Point3f> ToCvObjectPoints(const std::vector<Eigen::Vector3d>& points) {
  std::vector<cv::Point3f> cv_points;
  cv_points.reserve(points.size());
  for (const Eigen::Vector3d& point : points) {
    cv_points.push_back(cv::Point3f(static_cast<float>(point.x()), static_cast<float>(point.y()),
                                    static_cast<float>(point.z())));
  }
  return cv_points;
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
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(intrinsics));
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
    success = cv::solvePnP(cv_object_points, image_points, camera_matrix,
                           dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);
  }
  if (!success) {
    success = cv::solvePnP(cv_object_points, image_points, camera_matrix,
                           dist_coeffs, rvec, tvec, false,
                           cv::SOLVEPNP_ITERATIVE);
  }
  if (!success) {
    return false;
  }

  success = cv::solvePnP(cv_object_points, image_points, camera_matrix,
                         dist_coeffs, rvec, tvec, true,
                         cv::SOLVEPNP_ITERATIVE);
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

bool SolveRawImageSpacePinholePose(const OuterBootstrapCameraIntrinsics& intrinsics,
                                   const std::vector<Eigen::Vector3d>& object_points,
                                   const std::vector<cv::Point2f>& image_points,
                                   Eigen::Isometry3d* pose) {
  if (pose == nullptr) {
    throw std::runtime_error("SolveRawImageSpacePinholePose requires a valid output pointer.");
  }
  if (object_points.size() != image_points.size() || object_points.size() < 4) {
    return false;
  }

  cv::Mat rvec;
  cv::Mat tvec;
  const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << intrinsics.fu, 0.0, intrinsics.cu, 0.0,
                                 intrinsics.fv, intrinsics.cv, 0.0, 0.0, 1.0);
  const cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
  const std::vector<cv::Point3f> cv_object_points = ToCvObjectPoints(object_points);

  bool success = false;
  if (cv_object_points.size() == 4) {
    success = cv::solvePnP(cv_object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec,
                           false, cv::SOLVEPNP_IPPE);
  }
  if (!success) {
    success = cv::solvePnP(cv_object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec,
                           false, cv::SOLVEPNP_ITERATIVE);
  }
  if (!success) {
    return false;
  }

  success = cv::solvePnP(cv_object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec,
                         true, cv::SOLVEPNP_ITERATIVE);
  if (!success) {
    return false;
  }

  cv::Mat tvec64;
  tvec.convertTo(tvec64, CV_64F);
  if (tvec64.at<double>(2, 0) <= 0.0) {
    return false;
  }

  *pose = PoseFromCv(rvec, tvec);
  return true;
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
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(intrinsics));
  cv::Mat rvec;
  cv::Mat tvec;
  if (!camera.estimateTransformation(ToCvObjectPoints(object_points), image_points,
                                     &rvec, &tvec)) {
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

double ComputeObservationRmse(const OuterBootstrapCameraIntrinsics& intrinsics,
                              const Eigen::Isometry3d& T_camera_reference,
                              const Eigen::Isometry3d& T_reference_board,
                              const std::array<Eigen::Vector3d, 4>& board_points,
                              const std::array<cv::Point2f, 4>& image_points) {
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(intrinsics));
  double squared_error_sum = 0.0;
  for (int index = 0; index < 4; ++index) {
    Eigen::Vector2d projected = Eigen::Vector2d::Zero();
    const Eigen::Vector3d point_camera =
        T_camera_reference * (T_reference_board * board_points[static_cast<std::size_t>(index)]);
    if (!camera.vsEuclideanToKeypoint(point_camera, &projected)) {
      return 100.0;
    }
    const double dx = projected.x() - static_cast<double>(image_points[static_cast<std::size_t>(index)].x);
    const double dy = projected.y() - static_cast<double>(image_points[static_cast<std::size_t>(index)].y);
    squared_error_sum += dx * dx + dy * dy;
  }
  return std::sqrt(squared_error_sum / 4.0);
}

std::array<Eigen::Vector2d, 4> ComputeObservationCornerResiduals(
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const Eigen::Isometry3d& T_camera_reference,
    const Eigen::Isometry3d& T_reference_board,
    const std::array<Eigen::Vector3d, 4>& board_points,
    const std::array<cv::Point2f, 4>& image_points) {
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(intrinsics));
  std::array<Eigen::Vector2d, 4> residuals{};
  for (int index = 0; index < 4; ++index) {
    Eigen::Vector2d projected = Eigen::Vector2d::Zero();
    const Eigen::Vector3d point_camera =
        T_camera_reference * (T_reference_board * board_points[static_cast<std::size_t>(index)]);
    if (camera.vsEuclideanToKeypoint(point_camera, &projected)) {
      residuals[static_cast<std::size_t>(index)] =
          projected - Eigen::Vector2d(image_points[static_cast<std::size_t>(index)].x,
                                      image_points[static_cast<std::size_t>(index)].y);
    } else {
      residuals[static_cast<std::size_t>(index)] = Eigen::Vector2d(100.0, 100.0);
    }
  }
  return residuals;
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const double upper = values[middle];
  if (values.size() % 2 != 0) {
    return upper;
  }
  std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
  return 0.5 * (values[middle - 1] + upper);
}

double RotationDistanceRadians(const Eigen::Isometry3d& lhs,
                               const Eigen::Isometry3d& rhs) {
  const Eigen::Matrix3d delta = lhs.linear().transpose() * rhs.linear();
  const double cosine = std::max(-1.0, std::min(1.0, 0.5 * (delta.trace() - 1.0)));
  return std::acos(cosine);
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
    const Eigen::Vector4d coeffs(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z());
    quaternion_scatter += weight * coeffs * coeffs.transpose();
    translation_sum += weight * candidates[index].transform.translation();
    weight_sum += weight;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(quaternion_scatter);
  const Eigen::Vector4d eigenvector = solver.eigenvectors().col(3);
  Eigen::Quaterniond average_quaternion(eigenvector[0], eigenvector[1], eigenvector[2], eigenvector[3]);
  average_quaternion.normalize();

  Eigen::Isometry3d averaged = Eigen::Isometry3d::Identity();
  averaged.linear() = average_quaternion.toRotationMatrix();
  averaged.translation() = translation_sum / std::max(1e-9, weight_sum);
  return averaged;
}

Eigen::Isometry3d ConsensusTransform(const std::vector<TransformCandidate>& candidates) {
  if (candidates.size() < 3) {
    return AverageTransforms(candidates);
  }

  std::vector<double> pairwise_rotation;
  std::vector<double> pairwise_translation;
  pairwise_rotation.reserve(candidates.size() * (candidates.size() - 1) / 2);
  pairwise_translation.reserve(candidates.size() * (candidates.size() - 1) / 2);
  for (std::size_t lhs = 0; lhs < candidates.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < candidates.size(); ++rhs) {
      pairwise_rotation.push_back(
          RotationDistanceRadians(candidates[lhs].transform, candidates[rhs].transform));
      pairwise_translation.push_back(
          (candidates[lhs].transform.translation() -
           candidates[rhs].transform.translation()).norm());
    }
  }

  // The scales are derived from this board's measurements, rather than a
  // camera-specific pixel or metric threshold.  The medoid therefore selects
  // the densest transform cluster when a few independent planar poses branch.
  const double rotation_scale = std::max(1e-9, Median(pairwise_rotation));
  const double translation_scale = std::max(1e-9, Median(pairwise_translation));
  std::size_t best_index = 0;
  double best_score = std::numeric_limits<double>::infinity();
  double best_weight = -std::numeric_limits<double>::infinity();
  for (std::size_t candidate_index = 0;
       candidate_index < candidates.size(); ++candidate_index) {
    std::vector<double> distances;
    distances.reserve(candidates.size() - 1);
    for (std::size_t other_index = 0; other_index < candidates.size(); ++other_index) {
      if (candidate_index == other_index) {
        continue;
      }
      const double rotation = RotationDistanceRadians(
          candidates[candidate_index].transform, candidates[other_index].transform);
      const double translation =
          (candidates[candidate_index].transform.translation() -
           candidates[other_index].transform.translation()).norm();
      distances.push_back(rotation / rotation_scale + translation / translation_scale);
    }
    const double score = Median(distances);
    const double weight = candidates[candidate_index].weight;
    if (score < best_score ||
        (std::abs(score - best_score) < 1e-12 && weight > best_weight)) {
      best_index = candidate_index;
      best_score = score;
      best_weight = weight;
    }
  }
  return candidates[best_index].transform;
}

std::string JoinBoardIds(const std::vector<int>& board_ids) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < board_ids.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << board_ids[index];
  }
  return stream.str();
}

bool IsUsableBoardMeasurement(const OuterBoardMeasurement& measurement,
                              const OuterBootstrapOptions& options) {
  return measurement.success &&
         measurement.valid_refined_corner_count == 4 &&
         std::all_of(measurement.refined_corner_valid.begin(), measurement.refined_corner_valid.end(),
                     [](bool valid) { return valid; }) &&
         measurement.detection_quality >= options.min_detection_quality;
}

void AppendUniqueBoardId(int board_id, std::vector<int>* board_ids) {
  if (board_ids == nullptr || board_id < 0) {
    return;
  }
  if (std::find(board_ids->begin(), board_ids->end(), board_id) == board_ids->end()) {
    board_ids->push_back(board_id);
  }
}

SolverState BuildSolverState(const std::vector<OuterBootstrapFrameInput>& frame_inputs,
                             const std::map<int, std::array<Eigen::Vector3d, 4> >& board_corner_points,
                             const OuterBootstrapOptions& options,
                             std::vector<std::string>* warnings) {
  if (warnings == nullptr) {
    throw std::runtime_error("BuildSolverState requires a valid warnings pointer.");
  }

  SolverState state;
  state.board_corner_points = board_corner_points;

  std::set<int> ordered_board_id_set;
  for (std::size_t frame_storage_index = 0; frame_storage_index < frame_inputs.size(); ++frame_storage_index) {
    const OuterBootstrapFrameInput& frame_input = frame_inputs[frame_storage_index];
    FrameNode frame;
    frame.frame_index = frame_input.frame_index;
    frame.frame_label = frame_input.frame_label;
    frame.image_size = frame_input.measurements.image_size;
    frame.requested_board_ids = frame_input.measurements.requested_board_ids;

    if (frame.image_size.width <= 0 || frame.image_size.height <= 0) {
      throw std::runtime_error("Outer bootstrap requires frame measurements with a valid image_size.");
    }
    if (state.image_size.width == 0 && state.image_size.height == 0) {
      state.image_size = frame.image_size;
    } else if (frame.image_size != state.image_size) {
      throw std::runtime_error("Outer bootstrap requires all frames to share the same image size.");
    }

    for (std::size_t requested_index = 0; requested_index < frame.requested_board_ids.size(); ++requested_index) {
      ordered_board_id_set.insert(frame.requested_board_ids[requested_index]);
      state.boards[frame.requested_board_ids[requested_index]].board_id =
          frame.requested_board_ids[requested_index];
    }

    for (std::size_t measurement_index = 0;
         measurement_index < frame_input.measurements.board_measurements.size();
         ++measurement_index) {
      const OuterBoardMeasurement& measurement =
          frame_input.measurements.board_measurements[measurement_index];
      if (measurement.success) {
        AppendUniqueBoardId(measurement.board_id, &frame.visible_board_ids);
      }
      ordered_board_id_set.insert(measurement.board_id);
      state.boards[measurement.board_id].board_id = measurement.board_id;

      if (!IsUsableBoardMeasurement(measurement, options)) {
        if (measurement.success && measurement.valid_refined_corner_count > 0 &&
            measurement.detection_quality < options.min_detection_quality) {
          std::ostringstream warning;
          warning << "frame " << frame.frame_index << " board " << measurement.board_id
                  << " dropped by min_detection_quality=" << options.min_detection_quality;
          warnings->push_back(warning.str());
        }
        continue;
      }

      ObservationRecord observation;
      observation.frame_storage_index = static_cast<int>(frame_storage_index);
      observation.board_id = measurement.board_id;
      observation.quality = measurement.detection_quality;
      for (int corner_index = 0; corner_index < 4; ++corner_index) {
        const Eigen::Vector2d& point =
            measurement.refined_outer_corners_original_image[static_cast<std::size_t>(corner_index)];
        observation.image_points[static_cast<std::size_t>(corner_index)] =
            cv::Point2f(static_cast<float>(point.x()), static_cast<float>(point.y()));
      }

      const int observation_index = static_cast<int>(state.observations.size());
      state.observations.push_back(observation);
      frame.observation_indices.push_back(observation_index);
      state.boards[measurement.board_id].observation_indices.push_back(observation_index);
    }

    std::sort(frame.visible_board_ids.begin(), frame.visible_board_ids.end());
    state.frames.push_back(frame);
  }

  state.ordered_board_ids.assign(ordered_board_id_set.begin(), ordered_board_id_set.end());
  return state;
}

void MarkReferenceConnectedComponent(int reference_board_id, SolverState* state) {
  if (state == nullptr) {
    throw std::runtime_error("MarkReferenceConnectedComponent requires a valid state pointer.");
  }
  const auto reference_it = state->boards.find(reference_board_id);
  if (reference_it == state->boards.end()) {
    return;
  }

  std::queue<std::pair<bool, int> > queue;
  reference_it->second.reference_connected = true;
  queue.push(std::make_pair(true, reference_board_id));

  while (!queue.empty()) {
    const std::pair<bool, int> item = queue.front();
    queue.pop();
    if (item.first) {
      BoardNode& board = state->boards[item.second];
      for (std::size_t observation_offset = 0; observation_offset < board.observation_indices.size();
           ++observation_offset) {
        const int observation_index = board.observation_indices[observation_offset];
        FrameNode& frame =
            state->frames[static_cast<std::size_t>(state->observations[observation_index].frame_storage_index)];
        if (!frame.reference_connected) {
          frame.reference_connected = true;
          queue.push(std::make_pair(false, state->observations[observation_index].frame_storage_index));
        }
      }
    } else {
      FrameNode& frame = state->frames[static_cast<std::size_t>(item.second)];
      for (std::size_t observation_offset = 0; observation_offset < frame.observation_indices.size();
           ++observation_offset) {
        const int observation_index = frame.observation_indices[observation_offset];
        BoardNode& board = state->boards[state->observations[observation_index].board_id];
        if (!board.reference_connected) {
          board.reference_connected = true;
          queue.push(std::make_pair(true, board.board_id));
        }
      }
    }
  }

  for (std::size_t observation_index = 0; observation_index < state->observations.size();
       ++observation_index) {
    ObservationRecord& observation = state->observations[observation_index];
    const FrameNode& frame =
        state->frames[static_cast<std::size_t>(observation.frame_storage_index)];
    const BoardNode& board = state->boards[observation.board_id];
    observation.reference_connected = frame.reference_connected && board.reference_connected;
  }

  for (std::size_t frame_storage_index = 0; frame_storage_index < state->frames.size();
       ++frame_storage_index) {
    FrameNode& frame = state->frames[frame_storage_index];
    frame.connected_observation_count = 0;
    for (std::size_t observation_offset = 0; observation_offset < frame.observation_indices.size();
         ++observation_offset) {
      frame.connected_observation_count +=
          state->observations[frame.observation_indices[observation_offset]].reference_connected ? 1 : 0;
    }
  }

  for (auto board_it = state->boards.begin(); board_it != state->boards.end(); ++board_it) {
    BoardNode& board = board_it->second;
    board.connected_observation_count = 0;
    for (std::size_t observation_offset = 0; observation_offset < board.observation_indices.size();
         ++observation_offset) {
      board.connected_observation_count +=
          state->observations[board.observation_indices[observation_offset]].reference_connected ? 1 : 0;
    }
  }
}

bool EstimateFramePoseFromInitializedBoards(const SolverState& state,
                                            const OuterBootstrapCameraIntrinsics& intrinsics,
                                            int frame_storage_index,
                                            int reference_board_id,
                                            const std::map<int, BoardNode>& boards,
                                            Eigen::Isometry3d* pose,
                                            double* rmse) {
  if (pose == nullptr || rmse == nullptr) {
    throw std::runtime_error("EstimateFramePoseFromInitializedBoards requires valid output pointers.");
  }

  const FrameNode& frame = state.frames[static_cast<std::size_t>(frame_storage_index)];
  std::vector<Eigen::Vector3d> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(frame.observation_indices.size() * 4);
  image_points.reserve(frame.observation_indices.size() * 4);

  bool has_initialized_reference_observation = false;
  for (int observation_index : frame.observation_indices) {
    const ObservationRecord& observation = state.observations[observation_index];
    const auto board_it = boards.find(observation.board_id);
    if (observation.reference_connected &&
        observation.board_id == reference_board_id &&
        board_it != boards.end() && board_it->second.initialized) {
      has_initialized_reference_observation = true;
      break;
    }
  }

  for (std::size_t observation_offset = 0; observation_offset < frame.observation_indices.size();
       ++observation_offset) {
    const ObservationRecord& observation =
        state.observations[frame.observation_indices[observation_offset]];
    if (!observation.reference_connected) {
      continue;
    }
    if (has_initialized_reference_observation &&
        observation.board_id != reference_board_id) {
      continue;
    }
    const auto board_it = boards.find(observation.board_id);
    if (board_it == boards.end() || !board_it->second.initialized) {
      continue;
    }

    const std::array<Eigen::Vector3d, 4>& board_points =
        state.board_corner_points.at(observation.board_id);
    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      object_points.push_back(
          board_it->second.T_reference_board * board_points[static_cast<std::size_t>(corner_index)]);
      image_points.push_back(observation.image_points[static_cast<std::size_t>(corner_index)]);
    }
  }

  return EstimatePoseFromObjectPoints(intrinsics, object_points, image_points, pose, rmse);
}

bool EstimateSingleObservationBoardPoseInCamera(
    const SolverState& state,
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const ObservationRecord& observation,
    Eigen::Isometry3d* pose,
    double* rmse) {
  if (pose == nullptr || rmse == nullptr) {
    throw std::runtime_error(
        "EstimateSingleObservationBoardPoseInCamera requires valid output pointers.");
  }

  const std::array<Eigen::Vector3d, 4>& board_points =
      state.board_corner_points.at(observation.board_id);
  std::vector<Eigen::Vector3d> object_points(board_points.begin(), board_points.end());
  std::vector<cv::Point2f> image_points(observation.image_points.begin(), observation.image_points.end());
  return EstimatePoseFromObjectPoints(intrinsics, object_points, image_points, pose, rmse);
}

bool EstimateBoardPoseFromInitializedFrames(const SolverState& state,
                                            const OuterBootstrapCameraIntrinsics& intrinsics,
                                            const std::map<int, BoardNode>& boards,
                                            const std::vector<FrameNode>& frames,
                                            int board_id,
                                            bool robust_consensus,
                                            Eigen::Isometry3d* pose,
                                            double* rmse) {
  if (pose == nullptr || rmse == nullptr) {
    throw std::runtime_error("EstimateBoardPoseFromInitializedFrames requires valid output pointers.");
  }
  const auto board_it = boards.find(board_id);
  if (board_it == boards.end()) {
    return false;
  }

  std::vector<TransformCandidate> candidates;
  candidates.reserve(board_it->second.observation_indices.size());
  for (std::size_t observation_offset = 0;
       observation_offset < board_it->second.observation_indices.size();
       ++observation_offset) {
    const ObservationRecord& observation =
        state.observations[board_it->second.observation_indices[observation_offset]];
    if (!observation.reference_connected) {
      continue;
    }

    const FrameNode& frame =
        frames[static_cast<std::size_t>(observation.frame_storage_index)];
    if (!frame.initialized) {
      continue;
    }

    Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
    double observation_rmse = 0.0;
    if (!EstimateSingleObservationBoardPoseInCamera(state, intrinsics, observation,
                                                    &T_camera_board, &observation_rmse)) {
      continue;
    }

    TransformCandidate candidate;
    candidate.transform = frame.T_camera_reference.inverse() * T_camera_board;
    candidate.weight = observation.quality / std::max(1e-3, 1.0 + observation_rmse);
    candidates.push_back(candidate);
  }

  if (candidates.empty()) {
    return false;
  }

  *pose = robust_consensus
              ? ConsensusTransform(candidates)
              : AverageTransforms(candidates);
  double refined_rmse = 0.0;
  if (RefineBoardPoseFromInitializedFrames(state, intrinsics, frames, board_id,
                                           robust_consensus, pose,
                                           &refined_rmse)) {
    *rmse = refined_rmse;
    return true;
  }

  double squared_error_sum = 0.0;
  int count = 0;
  const std::array<Eigen::Vector3d, 4>& board_points = state.board_corner_points.at(board_id);
  for (std::size_t observation_offset = 0;
       observation_offset < board_it->second.observation_indices.size();
       ++observation_offset) {
    const ObservationRecord& observation =
        state.observations[board_it->second.observation_indices[observation_offset]];
    if (!observation.reference_connected) {
      continue;
    }
    const FrameNode& frame =
        frames[static_cast<std::size_t>(observation.frame_storage_index)];
    if (!frame.initialized) {
      continue;
    }
    const double observation_rmse =
        ComputeObservationRmse(intrinsics, frame.T_camera_reference, *pose,
                               board_points, observation.image_points);
    squared_error_sum += observation_rmse * observation_rmse;
    ++count;
  }
  *rmse = count > 0 ? std::sqrt(squared_error_sum / static_cast<double>(count)) : 0.0;
  return true;
}

std::string SummarizeBoardInitializationFailure(const SolverState& state,
                                                const OuterBootstrapCameraIntrinsics& intrinsics,
                                                const std::vector<FrameNode>& frames,
                                                int board_id) {
  const auto board_it = state.boards.find(board_id);
  if (board_it == state.boards.end()) {
    return "board missing from solver state";
  }

  const std::array<Eigen::Vector3d, 4>& board_points = state.board_corner_points.at(board_id);
  std::vector<Eigen::Vector3d> object_points(board_points.begin(), board_points.end());
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(intrinsics));

  int reference_connected_count = 0;
  int initialized_frame_count = 0;
  int ds_pose_success_count = 0;
  int raw_pinhole_success_count = 0;
  int fully_backprojectable_count = 0;

  for (std::size_t observation_offset = 0;
       observation_offset < board_it->second.observation_indices.size();
       ++observation_offset) {
    const ObservationRecord& observation =
        state.observations[board_it->second.observation_indices[observation_offset]];
    if (!observation.reference_connected) {
      continue;
    }
    ++reference_connected_count;

    const FrameNode& frame =
        frames[static_cast<std::size_t>(observation.frame_storage_index)];
    if (!frame.initialized) {
      continue;
    }
    ++initialized_frame_count;

    int valid_corner_count = 0;
    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      Eigen::Vector3d ray = Eigen::Vector3d::Zero();
      if (camera.keypointToEuclidean(
              Eigen::Vector2d(observation.image_points[static_cast<std::size_t>(corner_index)].x,
                              observation.image_points[static_cast<std::size_t>(corner_index)].y),
              &ray)) {
        ++valid_corner_count;
      }
    }
    if (valid_corner_count == 4) {
      ++fully_backprojectable_count;
    }

    Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
    double observation_rmse = 0.0;
    if (EstimateSingleObservationBoardPoseInCamera(state, intrinsics, observation,
                                                   &T_camera_board, &observation_rmse)) {
      ++ds_pose_success_count;
    }

    Eigen::Isometry3d pinhole_pose = Eigen::Isometry3d::Identity();
    std::vector<cv::Point2f> image_points(observation.image_points.begin(), observation.image_points.end());
    if (SolveRawImageSpacePinholePose(intrinsics, object_points, image_points, &pinhole_pose)) {
      ++raw_pinhole_success_count;
    }
  }

  std::ostringstream stream;
  stream << "connected_obs=" << reference_connected_count
         << " initialized_frame_obs=" << initialized_frame_count
         << " fully_backprojectable_obs=" << fully_backprojectable_count
         << " ds_pose_successes=" << ds_pose_success_count
         << " raw_pinhole_successes=" << raw_pinhole_success_count;
  return stream.str();
}

Eigen::VectorXd BuildBoardPoseResidualVector(const SolverState& state,
                                             const OuterBootstrapCameraIntrinsics& intrinsics,
                                             const std::vector<FrameNode>& frames,
                                             int board_id,
                                             const Eigen::Isometry3d& pose) {
  const auto board_it = state.boards.find(board_id);
  if (board_it == state.boards.end()) {
    return Eigen::VectorXd::Zero(0);
  }

  int corner_count = 0;
  for (std::size_t observation_offset = 0;
       observation_offset < board_it->second.observation_indices.size();
       ++observation_offset) {
    const ObservationRecord& observation =
        state.observations[board_it->second.observation_indices[observation_offset]];
    if (!observation.reference_connected) {
      continue;
    }
    const FrameNode& frame =
        frames[static_cast<std::size_t>(observation.frame_storage_index)];
    if (!frame.initialized) {
      continue;
    }
    corner_count += 4;
  }

  Eigen::VectorXd residuals = Eigen::VectorXd::Zero(2 * corner_count);
  if (corner_count <= 0) {
    return residuals;
  }

  const std::array<Eigen::Vector3d, 4>& board_points = state.board_corner_points.at(board_id);
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(intrinsics));
  int row = 0;
  for (std::size_t observation_offset = 0;
       observation_offset < board_it->second.observation_indices.size();
       ++observation_offset) {
    const ObservationRecord& observation =
        state.observations[board_it->second.observation_indices[observation_offset]];
    if (!observation.reference_connected) {
      continue;
    }
    const FrameNode& frame =
        frames[static_cast<std::size_t>(observation.frame_storage_index)];
    if (!frame.initialized) {
      continue;
    }

    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      const Eigen::Vector3d point_camera =
          frame.T_camera_reference * (pose * board_points[static_cast<std::size_t>(corner_index)]);
      Eigen::Vector2d projected = Eigen::Vector2d::Zero();
      if (camera.vsEuclideanToKeypoint(point_camera, &projected)) {
        residuals[row++] =
            projected.x() - static_cast<double>(observation.image_points[static_cast<std::size_t>(corner_index)].x);
        residuals[row++] =
            projected.y() - static_cast<double>(observation.image_points[static_cast<std::size_t>(corner_index)].y);
      } else {
        residuals[row++] = 100.0;
        residuals[row++] = 100.0;
      }
    }
  }

  return residuals;
}

double RobustBoardPoseScale(const Eigen::VectorXd& residuals) {
  std::vector<double> observation_rmse;
  observation_rmse.reserve(static_cast<std::size_t>(residuals.rows() / 8));
  for (int row = 0; row + 7 < residuals.rows(); row += 8) {
    observation_rmse.push_back(
        std::sqrt(residuals.segment(row, 8).squaredNorm() / 4.0));
  }
  if (observation_rmse.empty()) {
    return 1.0;
  }
  const double median = Median(observation_rmse);
  std::vector<double> deviations;
  deviations.reserve(observation_rmse.size());
  for (const double value : observation_rmse) {
    deviations.push_back(std::abs(value - median));
  }
  // This scale is wholly data-derived.  The median fallback keeps a compact
  // inlier set when the MAD is zero because several observations agree nearly
  // exactly.
  return std::max(1e-9, std::max(median, 1.4826 * Median(deviations)));
}

Eigen::VectorXd RobustBoardPoseSqrtWeights(const Eigen::VectorXd& residuals,
                                           double scale) {
  Eigen::VectorXd weights = Eigen::VectorXd::Ones(residuals.rows());
  for (int row = 0; row + 7 < residuals.rows(); row += 8) {
    const double observation_rmse =
        std::sqrt(residuals.segment(row, 8).squaredNorm() / 4.0);
    const double weight = std::min(1.0, scale / std::max(1e-9, observation_rmse));
    weights.segment(row, 8).setConstant(std::sqrt(weight));
  }
  return weights;
}

double RobustBoardPoseCost(const Eigen::VectorXd& residuals, double scale) {
  double cost = 0.0;
  for (int row = 0; row + 7 < residuals.rows(); row += 8) {
    const double observation_rmse =
        std::sqrt(residuals.segment(row, 8).squaredNorm() / 4.0);
    if (observation_rmse <= scale) {
      cost += 0.5 * observation_rmse * observation_rmse;
    } else {
      cost += scale * (observation_rmse - 0.5 * scale);
    }
  }
  return cost;
}

bool RefineBoardPoseFromInitializedFrames(const SolverState& state,
                                          const OuterBootstrapCameraIntrinsics& intrinsics,
                                          const std::vector<FrameNode>& frames,
                                          int board_id,
                                          bool robust_consensus,
                                          Eigen::Isometry3d* pose,
                                          double* rmse) {
  if (pose == nullptr || rmse == nullptr) {
    throw std::runtime_error("RefineBoardPoseFromInitializedFrames requires valid output pointers.");
  }

  Eigen::VectorXd residuals =
      BuildBoardPoseResidualVector(state, intrinsics, frames, board_id, *pose);
  if (residuals.size() <= 0) {
    return false;
  }

  double lambda = 1e-3;
  double robust_scale = robust_consensus ? RobustBoardPoseScale(residuals) : 0.0;
  double best_cost = robust_consensus ? RobustBoardPoseCost(residuals, robust_scale)
                                      : residuals.squaredNorm();
  for (int iteration = 0; iteration < 20; ++iteration) {
    Eigen::MatrixXd jacobian(residuals.rows(), 6);
    const Eigen::VectorXd sqrt_weights =
        robust_consensus ? RobustBoardPoseSqrtWeights(residuals, robust_scale)
                         : Eigen::VectorXd::Ones(residuals.rows());
    const Eigen::VectorXd weighted_residuals =
        sqrt_weights.array() * residuals.array();
    for (int column = 0; column < 6; ++column) {
      Eigen::Matrix<double, 6, 1> plus_delta = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> minus_delta = Eigen::Matrix<double, 6, 1>::Zero();
      const double step = column < 3 ? 1e-4 : 5e-4;
      plus_delta[column] = step;
      minus_delta[column] = -step;
      jacobian.col(column) =
          (BuildBoardPoseResidualVector(state, intrinsics, frames, board_id,
                                        ApplyPoseDelta(*pose, plus_delta)) -
           BuildBoardPoseResidualVector(state, intrinsics, frames, board_id,
                                        ApplyPoseDelta(*pose, minus_delta))) /
          (2.0 * step);
      if (robust_consensus) {
        jacobian.col(column).array() *= sqrt_weights.array();
      }
    }

    const Eigen::Matrix<double, 6, 6> hessian = jacobian.transpose() * jacobian;
    const Eigen::Matrix<double, 6, 1> gradient = jacobian.transpose() * weighted_residuals;
    const Eigen::Matrix<double, 6, 1> delta =
        (hessian + lambda * Eigen::Matrix<double, 6, 6>::Identity()).ldlt().solve(-gradient);
    if (!delta.allFinite()) {
      break;
    }

    const Eigen::Isometry3d candidate_pose = ApplyPoseDelta(*pose, delta);
    const Eigen::VectorXd candidate_residuals =
        BuildBoardPoseResidualVector(state, intrinsics, frames, board_id, candidate_pose);
    const double candidate_cost = robust_consensus
                                      ? RobustBoardPoseCost(candidate_residuals, robust_scale)
                                      : candidate_residuals.squaredNorm();
    if (candidate_cost < best_cost) {
      *pose = candidate_pose;
      residuals = candidate_residuals;
      if (robust_consensus) {
        robust_scale = RobustBoardPoseScale(residuals);
        best_cost = RobustBoardPoseCost(residuals, robust_scale);
      } else {
        best_cost = candidate_cost;
      }
      lambda *= 0.5;
      if (delta.norm() < 1e-5) {
        break;
      }
    } else {
      lambda *= 4.0;
    }
  }

  const int residual_corner_count = std::max(1, static_cast<int>(residuals.rows() / 2));
  *rmse = std::sqrt(residuals.squaredNorm() / static_cast<double>(residual_corner_count));
  return std::isfinite(*rmse);
}

bool InitializeConnectedComponent(const OuterBootstrapOptions& options,
                                  const SolverState& state,
                                  std::map<int, BoardNode>* boards,
                                  std::vector<FrameNode>* frames,
                                  std::vector<std::string>* warnings,
                                  const OuterBootstrapCameraIntrinsics& intrinsics) {
  if (boards == nullptr || frames == nullptr || warnings == nullptr) {
    throw std::runtime_error("InitializeConnectedComponent requires valid output pointers.");
  }

  const auto reference_it = boards->find(options.reference_board_id);
  if (reference_it == boards->end() || !reference_it->second.reference_connected ||
      reference_it->second.connected_observation_count <= 0) {
    return false;
  }

  reference_it->second.initialized = true;
  reference_it->second.T_reference_board = Eigen::Isometry3d::Identity();
  reference_it->second.rmse = 0.0;

  if (!options.fixed_board_layout.empty()) {
    for (const auto& layout_entry : options.fixed_board_layout) {
      const auto board_it = boards->find(layout_entry.first);
      if (board_it == boards->end() ||
          !board_it->second.reference_connected) {
        warnings->push_back(
            "fixed board layout contains board " +
            std::to_string(layout_entry.first) +
            " which is not in the reference-connected component");
        return false;
      }
      board_it->second.initialized = true;
      board_it->second.T_reference_board = layout_entry.second;
      board_it->second.rmse = 0.0;
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;

    for (std::size_t frame_storage_index = 0; frame_storage_index < frames->size(); ++frame_storage_index) {
      FrameNode& frame = (*frames)[frame_storage_index];
      if (!frame.reference_connected || frame.initialized) {
        continue;
      }

      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      double frame_rmse = 0.0;
      if (EstimateFramePoseFromInitializedBoards(state, intrinsics,
                                                 static_cast<int>(frame_storage_index),
                                                 options.reference_board_id,
                                                 *boards, &pose, &frame_rmse)) {
        frame.initialized = true;
        frame.T_camera_reference = pose;
        frame.rmse = frame_rmse;
        changed = true;
      }
    }

    for (auto board_it = boards->begin(); board_it != boards->end(); ++board_it) {
      BoardNode& board = board_it->second;
      if (!board.reference_connected || board.initialized || board.board_id == options.reference_board_id) {
        continue;
      }

      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      double board_rmse = 0.0;
      if (EstimateBoardPoseFromInitializedFrames(state, intrinsics, *boards, *frames, board.board_id,
                                                 options.robust_board_layout_consensus,
                                                 &pose, &board_rmse)) {
        board.initialized = true;
        board.T_reference_board = pose;
        board.rmse = board_rmse;
        changed = true;
      }
    }
  }

  for (std::size_t frame_storage_index = 0; frame_storage_index < frames->size(); ++frame_storage_index) {
    const FrameNode& frame = (*frames)[frame_storage_index];
    if (frame.reference_connected && !frame.initialized) {
      std::ostringstream warning;
      warning << "frame " << frame.frame_index
              << " stayed uninitialized after reference-connected propagation";
      warnings->push_back(warning.str());
    }
  }

  for (auto board_it = boards->begin(); board_it != boards->end(); ++board_it) {
    const BoardNode& board = board_it->second;
    if (board.reference_connected && !board.initialized) {
      std::ostringstream warning;
      warning << "board " << board.board_id
              << " stayed uninitialized after reference-connected propagation"
              << " (" << SummarizeBoardInitializationFailure(state, intrinsics, *frames, board.board_id)
              << ")";
      warnings->push_back(warning.str());
    }
  }

  return true;
}

Eigen::VectorXd BuildResidualVector(const SolverState& state,
                                    const OuterBootstrapCameraIntrinsics& intrinsics,
                                    const std::map<int, BoardNode>& boards,
                                    const std::vector<FrameNode>& frames,
                                    int* residual_corner_count) {
  int corner_count = 0;
  for (std::size_t observation_index = 0; observation_index < state.observations.size(); ++observation_index) {
    const ObservationRecord& observation = state.observations[observation_index];
    if (!observation.reference_connected) {
      continue;
    }
    const auto board_it = boards.find(observation.board_id);
    if (board_it == boards.end() || !board_it->second.initialized) {
      continue;
    }
    const FrameNode& frame =
        frames[static_cast<std::size_t>(observation.frame_storage_index)];
    if (!frame.initialized) {
      continue;
    }
    corner_count += 4;
  }

  if (residual_corner_count != nullptr) {
    *residual_corner_count = corner_count;
  }

  Eigen::VectorXd residuals = Eigen::VectorXd::Zero(2 * corner_count);
  int row = 0;
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(intrinsics));
  for (std::size_t observation_index = 0; observation_index < state.observations.size(); ++observation_index) {
    const ObservationRecord& observation = state.observations[observation_index];
    if (!observation.reference_connected) {
      continue;
    }
    const auto board_it = boards.find(observation.board_id);
    if (board_it == boards.end() || !board_it->second.initialized) {
      continue;
    }
    const FrameNode& frame =
        frames[static_cast<std::size_t>(observation.frame_storage_index)];
    if (!frame.initialized) {
      continue;
    }

    const std::array<Eigen::Vector3d, 4>& board_points =
        state.board_corner_points.at(observation.board_id);
    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      const Eigen::Vector3d point_camera =
          frame.T_camera_reference *
          (board_it->second.T_reference_board * board_points[static_cast<std::size_t>(corner_index)]);
      Eigen::Vector2d projected = Eigen::Vector2d::Zero();
      if (camera.vsEuclideanToKeypoint(point_camera, &projected)) {
        residuals[row++] =
            projected.x() - static_cast<double>(observation.image_points[static_cast<std::size_t>(corner_index)].x);
        residuals[row++] =
            projected.y() - static_cast<double>(observation.image_points[static_cast<std::size_t>(corner_index)].y);
      } else {
        residuals[row++] = 100.0;
        residuals[row++] = 100.0;
      }
    }
  }
  return residuals;
}

double ComputeRmse(const Eigen::VectorXd& residuals, int residual_corner_count) {
  if (residual_corner_count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(residuals.squaredNorm() / static_cast<double>(residual_corner_count));
}

bool OptimizeIntrinsics(const SolverState& state,
                        const std::map<int, BoardNode>& boards,
                        const std::vector<FrameNode>& frames,
                        const OuterBootstrapCameraIntrinsics& anchor_intrinsics,
                        OuterBootstrapCameraIntrinsics* intrinsics,
                        double* rmse) {
  if (intrinsics == nullptr || rmse == nullptr) {
    throw std::runtime_error("OptimizeIntrinsics requires valid output pointers.");
  }

  int residual_corner_count = 0;
  Eigen::VectorXd residuals = BuildResidualVector(state, *intrinsics, boards, frames, &residual_corner_count);
  if (residual_corner_count <= 0) {
    return false;
  }

  double lambda = 1e-3;
  double best_cost = residuals.squaredNorm();
  Eigen::VectorXd parameters = ToVector(*intrinsics);
  const Eigen::VectorXd anchor = ToVector(anchor_intrinsics);
  const std::vector<std::string> parameter_labels =
      intrinsics->CombinedParameterLabels();
  Eigen::VectorXd prior_sigma(parameters.rows());
  for (int index = 0; index < parameters.rows(); ++index) {
    const std::string& label = parameter_labels[static_cast<std::size_t>(index)];
    if (label == "xi" || label == "alpha" || label == "beta" ||
        label == "k1" || label == "k2" || label == "k3" || label == "k4") {
      prior_sigma[index] = 0.20;
    } else if (label == "fu") {
      prior_sigma[index] = 0.20 * static_cast<double>(intrinsics->resolution.width);
    } else if (label == "fv") {
      prior_sigma[index] = 0.20 * static_cast<double>(intrinsics->resolution.height);
    } else if (label == "cu") {
      prior_sigma[index] = 0.03 * static_cast<double>(intrinsics->resolution.width);
    } else if (label == "cv") {
      prior_sigma[index] = 0.03 * static_cast<double>(intrinsics->resolution.height);
    } else {
      prior_sigma[index] = 0.20;
    }
  }
  Eigen::VectorXd prior_weight(parameters.rows());
  for (int index = 0; index < parameters.rows(); ++index) {
    prior_weight[index] = 1.0 / std::max(1e-9, prior_sigma[index] * prior_sigma[index]);
  }

  for (int iteration = 0; iteration < 18; ++iteration) {
    Eigen::MatrixXd jacobian(residuals.rows(), parameters.rows());
    for (int column = 0; column < parameters.rows(); ++column) {
      Eigen::VectorXd plus = parameters;
      Eigen::VectorXd minus = parameters;
      const std::string& label = parameter_labels[static_cast<std::size_t>(column)];
      const double fallback_step =
          (label == "fu" || label == "fv" || label == "cu" || label == "cv") ? 1e-1 : 1e-3;
      const double step = ParameterStep(parameters[column], fallback_step);
      plus[column] += step;
      minus[column] -= step;

      OuterBootstrapCameraIntrinsics plus_intrinsics = FromVector(plus, *intrinsics);
      OuterBootstrapCameraIntrinsics minus_intrinsics = FromVector(minus, *intrinsics);
      ClampIntrinsicsInPlace(&plus_intrinsics);
      ClampIntrinsicsInPlace(&minus_intrinsics);

      jacobian.col(column) =
          (BuildResidualVector(state, plus_intrinsics, boards, frames, nullptr) -
           BuildResidualVector(state, minus_intrinsics, boards, frames, nullptr)) /
          (2.0 * step);
    }

    const Eigen::MatrixXd hessian = jacobian.transpose() * jacobian;
    const Eigen::VectorXd gradient = jacobian.transpose() * residuals;
    Eigen::MatrixXd prior_hessian =
        Eigen::MatrixXd::Zero(parameters.rows(), parameters.rows());
    Eigen::VectorXd prior_gradient = Eigen::VectorXd::Zero(parameters.rows());
    for (int index = 0; index < parameters.rows(); ++index) {
      prior_hessian(index, index) = prior_weight[index];
      prior_gradient[index] = prior_weight[index] * (parameters[index] - anchor[index]);
    }
    const Eigen::MatrixXd damped =
        hessian + prior_hessian + lambda * Eigen::MatrixXd::Identity(parameters.rows(), parameters.rows());
    const Eigen::VectorXd delta =
        damped.ldlt().solve(-(gradient + prior_gradient));
    if (!delta.allFinite()) {
      break;
    }

    OuterBootstrapCameraIntrinsics candidate = FromVector(parameters + delta, *intrinsics);
    ClampIntrinsicsInPlace(&candidate);
    const Eigen::VectorXd candidate_residuals =
        BuildResidualVector(state, candidate, boards, frames, nullptr);
    const Eigen::VectorXd candidate_vector = ToVector(candidate);
    double prior_cost = 0.0;
    for (int index = 0; index < parameters.rows(); ++index) {
      const double diff = candidate_vector[index] - anchor[index];
      prior_cost += prior_weight[index] * diff * diff;
    }
    const double candidate_cost = candidate_residuals.squaredNorm() + prior_cost;
    if (candidate_cost < best_cost) {
      parameters = candidate_vector;
      *intrinsics = candidate;
      residuals = candidate_residuals;
      best_cost = candidate_cost;
      lambda *= 0.5;
      if (delta.norm() < 1e-4) {
        break;
      }
    } else {
      lambda *= 4.0;
    }
  }

  *rmse = ComputeRmse(residuals, residual_corner_count);
  return true;
}

double ComputeFrameRmse(const SolverState& state,
                        const OuterBootstrapCameraIntrinsics& intrinsics,
                        const std::map<int, BoardNode>& boards,
                        const FrameNode& frame) {
  if (!frame.initialized) {
    return std::numeric_limits<double>::infinity();
  }

  double squared_error_sum = 0.0;
  int count = 0;
  for (std::size_t observation_offset = 0; observation_offset < frame.observation_indices.size();
       ++observation_offset) {
    const ObservationRecord& observation =
        state.observations[frame.observation_indices[observation_offset]];
    if (!observation.reference_connected) {
      continue;
    }
    const auto board_it = boards.find(observation.board_id);
    if (board_it == boards.end() || !board_it->second.initialized) {
      continue;
    }
    const double observation_rmse =
        ComputeObservationRmse(intrinsics, frame.T_camera_reference,
                               board_it->second.T_reference_board,
                               state.board_corner_points.at(observation.board_id),
                               observation.image_points);
    squared_error_sum += observation_rmse * observation_rmse;
    ++count;
  }

  return count > 0 ? std::sqrt(squared_error_sum / static_cast<double>(count))
                   : std::numeric_limits<double>::infinity();
}

double ComputeBoardRmse(const SolverState& state,
                        const OuterBootstrapCameraIntrinsics& intrinsics,
                        const std::vector<FrameNode>& frames,
                        const BoardNode& board) {
  if (!board.initialized) {
    return std::numeric_limits<double>::infinity();
  }

  double squared_error_sum = 0.0;
  int count = 0;
  for (std::size_t observation_offset = 0; observation_offset < board.observation_indices.size();
       ++observation_offset) {
    const ObservationRecord& observation =
        state.observations[board.observation_indices[observation_offset]];
    if (!observation.reference_connected) {
      continue;
    }
    const FrameNode& frame =
        frames[static_cast<std::size_t>(observation.frame_storage_index)];
    if (!frame.initialized) {
      continue;
    }
    const double observation_rmse =
        ComputeObservationRmse(intrinsics, frame.T_camera_reference,
                               board.T_reference_board,
                               state.board_corner_points.at(observation.board_id),
                               observation.image_points);
    squared_error_sum += observation_rmse * observation_rmse;
    ++count;
  }

  return count > 0 ? std::sqrt(squared_error_sum / static_cast<double>(count))
                   : std::numeric_limits<double>::infinity();
}

int CountInitializedObservationsForFrame(const SolverState& state,
                                         const std::map<int, BoardNode>& boards,
                                         const FrameNode& frame) {
  int count = 0;
  for (std::size_t observation_offset = 0; observation_offset < frame.observation_indices.size();
       ++observation_offset) {
    const ObservationRecord& observation =
        state.observations[frame.observation_indices[observation_offset]];
    if (!observation.reference_connected) {
      continue;
    }
    const auto board_it = boards.find(observation.board_id);
    if (board_it != boards.end() && board_it->second.initialized) {
      ++count;
    }
  }
  return count;
}

int CountInitializedObservationsForBoard(const SolverState& state,
                                         const std::vector<FrameNode>& frames,
                                         const BoardNode& board) {
  int count = 0;
  for (std::size_t observation_offset = 0; observation_offset < board.observation_indices.size();
       ++observation_offset) {
    const ObservationRecord& observation =
        state.observations[board.observation_indices[observation_offset]];
    if (!observation.reference_connected) {
      continue;
    }
    if (frames[static_cast<std::size_t>(observation.frame_storage_index)].initialized) {
      ++count;
    }
  }
  return count;
}

double ComputeGlobalRmse(const SolverState& state,
                         const OuterBootstrapCameraIntrinsics& intrinsics,
                         const std::map<int, BoardNode>& boards,
                         const std::vector<FrameNode>& frames,
                         int* used_observation_count,
                         int* used_corner_count) {
  int observation_count = 0;
  double squared_error_sum = 0.0;
  for (std::size_t observation_index = 0; observation_index < state.observations.size(); ++observation_index) {
    const ObservationRecord& observation = state.observations[observation_index];
    if (!observation.reference_connected) {
      continue;
    }
    const auto board_it = boards.find(observation.board_id);
    if (board_it == boards.end() || !board_it->second.initialized) {
      continue;
    }
    const FrameNode& frame =
        frames[static_cast<std::size_t>(observation.frame_storage_index)];
    if (!frame.initialized) {
      continue;
    }
    const double observation_rmse =
        ComputeObservationRmse(intrinsics, frame.T_camera_reference,
                               board_it->second.T_reference_board,
                               state.board_corner_points.at(observation.board_id),
                               observation.image_points);
    squared_error_sum += observation_rmse * observation_rmse;
    ++observation_count;
  }

  if (used_observation_count != nullptr) {
    *used_observation_count = observation_count;
  }
  if (used_corner_count != nullptr) {
    *used_corner_count = 4 * observation_count;
  }
  return observation_count > 0 ? std::sqrt(squared_error_sum / static_cast<double>(observation_count))
                               : std::numeric_limits<double>::infinity();
}

std::vector<OuterBootstrapObservationDiagnostics> BuildObservationDiagnostics(
    const SolverState& state,
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const std::map<int, BoardNode>& boards,
    const std::vector<FrameNode>& frames) {
  std::vector<OuterBootstrapObservationDiagnostics> diagnostics;
  diagnostics.reserve(state.observations.size());

  for (std::size_t observation_index = 0; observation_index < state.observations.size(); ++observation_index) {
    const ObservationRecord& observation = state.observations[observation_index];
    const FrameNode& frame =
        frames[static_cast<std::size_t>(observation.frame_storage_index)];
    const auto board_it = boards.find(observation.board_id);

    OuterBootstrapObservationDiagnostics diagnostics_entry;
    diagnostics_entry.frame_index = frame.frame_index;
    diagnostics_entry.frame_label = frame.frame_label;
    diagnostics_entry.board_id = observation.board_id;
    diagnostics_entry.detection_quality = observation.quality;
    diagnostics_entry.reference_connected = observation.reference_connected;
    diagnostics_entry.frame_initialized = frame.initialized;
    diagnostics_entry.board_initialized =
        board_it != boards.end() && board_it->second.initialized;
    diagnostics_entry.used_in_solve = diagnostics_entry.reference_connected &&
                                      diagnostics_entry.frame_initialized &&
                                      diagnostics_entry.board_initialized;
    diagnostics_entry.observation_rmse = std::numeric_limits<double>::infinity();

    if (diagnostics_entry.used_in_solve && board_it != boards.end()) {
      const std::array<Eigen::Vector3d, 4>& board_points =
          state.board_corner_points.at(observation.board_id);
      diagnostics_entry.corner_residuals_xy =
          ComputeObservationCornerResiduals(intrinsics,
                                           frame.T_camera_reference,
                                           board_it->second.T_reference_board,
                                           board_points,
                                           observation.image_points);
      diagnostics_entry.observation_rmse =
          ComputeObservationRmse(intrinsics,
                                 frame.T_camera_reference,
                                 board_it->second.T_reference_board,
                                 board_points,
                                 observation.image_points);
      for (int corner_index = 0; corner_index < 4; ++corner_index) {
        const Eigen::Vector2d& residual =
            diagnostics_entry.corner_residuals_xy[static_cast<std::size_t>(corner_index)];
        diagnostics_entry.max_abs_residual_x =
            std::max(diagnostics_entry.max_abs_residual_x, std::abs(residual.x()));
        diagnostics_entry.max_abs_residual_y =
            std::max(diagnostics_entry.max_abs_residual_y, std::abs(residual.y()));
      }
    }

    diagnostics.push_back(diagnostics_entry);
  }

  return diagnostics;
}

void OptimizeBootstrapState(const OuterBootstrapOptions& options,
                            const SolverState& state,
                            const OuterBootstrapCameraIntrinsics& anchor_intrinsics,
                            OuterBootstrapCameraIntrinsics* intrinsics,
                            std::map<int, BoardNode>* boards,
                            std::vector<FrameNode>* frames) {
  if (intrinsics == nullptr || boards == nullptr || frames == nullptr) {
    throw std::runtime_error("OptimizeBootstrapState requires valid output pointers.");
  }

  double previous_rmse = ComputeGlobalRmse(state, *intrinsics, *boards, *frames, nullptr, nullptr);
  for (int iteration = 0; iteration < options.max_coordinate_descent_iterations; ++iteration) {
    for (std::size_t frame_storage_index = 0; frame_storage_index < frames->size(); ++frame_storage_index) {
      FrameNode& frame = (*frames)[frame_storage_index];
      if (!frame.initialized) {
        continue;
      }

      Eigen::Isometry3d pose = frame.T_camera_reference;
      double frame_rmse = frame.rmse;
      if (EstimateFramePoseFromInitializedBoards(state, *intrinsics,
                                                 static_cast<int>(frame_storage_index),
                                                 options.reference_board_id,
                                                 *boards, &pose, &frame_rmse)) {
        frame.T_camera_reference = pose;
        frame.rmse = frame_rmse;
      }
    }

    for (auto board_it = boards->begin(); board_it != boards->end(); ++board_it) {
      BoardNode& board = board_it->second;
      if (!board.initialized || board.board_id == options.reference_board_id ||
          !options.fixed_board_layout.empty()) {
        continue;
      }

      Eigen::Isometry3d pose = board.T_reference_board;
      double board_rmse = board.rmse;
      if (EstimateBoardPoseFromInitializedFrames(state, *intrinsics, *boards, *frames,
                                                 board.board_id,
                                                 options.robust_board_layout_consensus,
                                                 &pose, &board_rmse)) {
        board.T_reference_board = pose;
        board.rmse = board_rmse;
      }
    }

    if (options.optimize_intrinsics) {
      double intrinsics_rmse = previous_rmse;
      OptimizeIntrinsics(state, *boards, *frames, anchor_intrinsics, intrinsics,
                         &intrinsics_rmse);
    }

    const double current_rmse =
        ComputeGlobalRmse(state, *intrinsics, *boards, *frames, nullptr, nullptr);
    if (!std::isfinite(current_rmse)) {
      break;
    }
    if (std::abs(previous_rmse - current_rmse) < options.convergence_threshold) {
      break;
    }
    previous_rmse = current_rmse;
  }

  for (std::size_t frame_storage_index = 0; frame_storage_index < frames->size(); ++frame_storage_index) {
    FrameNode& frame = (*frames)[frame_storage_index];
    frame.rmse = ComputeFrameRmse(state, *intrinsics, *boards, frame);
  }
  for (auto board_it = boards->begin(); board_it != boards->end(); ++board_it) {
    BoardNode& board = board_it->second;
    board.rmse = ComputeBoardRmse(state, *intrinsics, *frames, board);
  }
}

OuterBootstrapResult BuildFailureResult(const OuterBootstrapOptions& options,
                                        const SolverState& state,
                                        const OuterBootstrapCameraIntrinsics& intrinsics,
                                        const std::map<int, BoardNode>& boards,
                                        const std::vector<FrameNode>& frames,
                                        const std::vector<std::string>& warnings,
                                        const std::string& failure_reason) {
  OuterBootstrapResult result;
  result.success = false;
  result.reference_board_id = options.reference_board_id;
  result.coarse_camera = intrinsics;
  result.failure_reason = failure_reason;
  result.warnings = warnings;
  result.observation_diagnostics = BuildObservationDiagnostics(state, intrinsics, boards, frames);

  result.frames.reserve(frames.size());
  for (std::size_t frame_storage_index = 0; frame_storage_index < frames.size(); ++frame_storage_index) {
    OuterBootstrapFrameState frame_state;
    frame_state.frame_index = frames[frame_storage_index].frame_index;
    frame_state.frame_label = frames[frame_storage_index].frame_label;
    frame_state.initialized = frames[frame_storage_index].initialized;
    frame_state.visible_board_ids = frames[frame_storage_index].visible_board_ids;
    frame_state.T_camera_reference = ToMatrix4d(frames[frame_storage_index].T_camera_reference);
    frame_state.observation_count =
        CountInitializedObservationsForFrame(state, boards, frames[frame_storage_index]);
    frame_state.rmse = frames[frame_storage_index].rmse;
    result.frames.push_back(frame_state);
  }

  result.boards.reserve(state.ordered_board_ids.size());
  for (std::size_t board_index = 0; board_index < state.ordered_board_ids.size(); ++board_index) {
    const int board_id = state.ordered_board_ids[board_index];
    OuterBootstrapBoardState board_state;
    board_state.board_id = board_id;
    const auto board_it = boards.find(board_id);
    if (board_it != boards.end()) {
      board_state.initialized = board_it->second.initialized;
      board_state.T_reference_board = ToMatrix4d(board_it->second.T_reference_board);
      board_state.observation_count =
          CountInitializedObservationsForBoard(state, frames, board_it->second);
      board_state.rmse = board_it->second.rmse;
    }
    result.boards.push_back(board_state);
  }

  result.used_frame_count = static_cast<int>(std::count_if(
      frames.begin(), frames.end(), [](const FrameNode& frame) { return frame.initialized; }));
  result.global_rmse =
      ComputeGlobalRmse(state, intrinsics, boards, frames,
                        &result.used_board_observation_count, &result.used_corner_count);
  return result;
}

}  // namespace

bool LoadFixedBoardLayoutCsv(const std::string& path,
                             int reference_board_id,
                             FixedBoardLayout* layout,
                             std::string* failure_reason) {
  if (layout == nullptr) {
    if (failure_reason != nullptr) {
      *failure_reason = "fixed board layout output pointer is null";
    }
    return false;
  }
  layout->clear();
  std::ifstream input(path.c_str());
  if (!input.is_open()) {
    if (failure_reason != nullptr) {
      *failure_reason = "could not open fixed board layout CSV: " + path;
    }
    return false;
  }

  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> fields;
    std::stringstream line_stream(line);
    std::string field;
    while (std::getline(line_stream, field, ',')) {
      fields.push_back(field);
    }
    if (line_number == 1 && !fields.empty() && fields.front() == "board_id") {
      continue;
    }
    if (fields.size() != 20u) {
      if (failure_reason != nullptr) {
        *failure_reason = "fixed board layout CSV line " +
                          std::to_string(line_number) +
                          " must have board_id, initialized, observation_count, "
                          "rmse, and 16 matrix values";
      }
      layout->clear();
      return false;
    }
    int board_id = -1;
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    try {
      board_id = std::stoi(fields[0]);
      for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
          matrix(row, column) = std::stod(
              fields[static_cast<std::size_t>(4 + 4 * row + column)]);
        }
      }
    } catch (const std::exception&) {
      if (failure_reason != nullptr) {
        *failure_reason = "fixed board layout CSV line " +
                          std::to_string(line_number) +
                          " contains an invalid numeric value";
      }
      layout->clear();
      return false;
    }
    const Eigen::Matrix3d rotation = matrix.topLeftCorner<3, 3>();
    const bool rigid = matrix.allFinite() &&
                       (matrix.row(3) -
                        Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)).norm() < 1e-8 &&
                       (rotation.transpose() * rotation -
                        Eigen::Matrix3d::Identity()).norm() < 1e-5 &&
                       std::abs(rotation.determinant() - 1.0) < 1e-5;
    if (board_id < 0 || !rigid || layout->count(board_id) != 0u) {
      if (failure_reason != nullptr) {
        *failure_reason = "fixed board layout CSV line " +
                          std::to_string(line_number) +
                          " has a duplicate board id or a non-rigid transform";
      }
      layout->clear();
      return false;
    }
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.matrix() = matrix;
    layout->emplace(board_id, transform);
  }

  const auto reference_it = layout->find(reference_board_id);
  if (layout->empty() || reference_it == layout->end() ||
      (reference_it->second.matrix() - Eigen::Matrix4d::Identity()).norm() >
          1e-8) {
    if (failure_reason != nullptr) {
      *failure_reason = "fixed board layout must contain the reference board " +
                        std::to_string(reference_board_id) +
                        " with an identity transform";
    }
    layout->clear();
    return false;
  }
  return true;
}

MultiBoardOuterBootstrap::MultiBoardOuterBootstrap(
    ApriltagInternalConfig base_config, OuterBootstrapOptions options)
    : base_config_(std::move(base_config)), options_(std::move(options)) {}

ApriltagCanonicalModel MultiBoardOuterBootstrap::ModelForBoardId(int board_id) const {
  ApriltagInternalConfig config = base_config_;
  config.tag_id = board_id;
  config.tag_ids.clear();
  config.outer_detector_config.tag_id = board_id;
  config.outer_detector_config.tag_ids.clear();
  return ApriltagCanonicalModel(config);
}

OuterBootstrapResult MultiBoardOuterBootstrap::Solve(
    const std::vector<OuterBootstrapFrameInput>& frame_inputs) const {
  if (frame_inputs.empty()) {
    OuterBootstrapResult result;
    result.reference_board_id = options_.reference_board_id;
    result.failure_reason = "NoInputFrames";
    return result;
  }

  std::set<int> board_ids;
  for (std::size_t frame_index = 0; frame_index < frame_inputs.size(); ++frame_index) {
    for (std::size_t requested_index = 0;
         requested_index < frame_inputs[frame_index].measurements.requested_board_ids.size();
         ++requested_index) {
      board_ids.insert(frame_inputs[frame_index].measurements.requested_board_ids[requested_index]);
    }
    for (std::size_t measurement_index = 0;
         measurement_index < frame_inputs[frame_index].measurements.board_measurements.size();
         ++measurement_index) {
      board_ids.insert(
          frame_inputs[frame_index].measurements.board_measurements[measurement_index].board_id);
    }
  }
  board_ids.insert(options_.reference_board_id);

  std::map<int, std::array<Eigen::Vector3d, 4> > board_corner_points;
  for (auto board_it = board_ids.begin(); board_it != board_ids.end(); ++board_it) {
    board_corner_points[*board_it] = BuildOuterCornerPoints(ModelForBoardId(*board_it));
  }
  for (const OuterBootstrapFrameInput& frame : frame_inputs) {
    for (const OuterBoardMeasurement& measurement :
         frame.measurements.board_measurements) {
      if (!measurement.has_target_outer_corners) {
        continue;
      }
      auto geometry_it = board_corner_points.find(measurement.board_id);
      if (geometry_it == board_corner_points.end()) {
        board_corner_points[measurement.board_id] =
            measurement.target_outer_corners_board;
        continue;
      }
      bool matches_existing_imported_geometry = true;
      for (int corner_index = 0; corner_index < 4; ++corner_index) {
        const Eigen::Vector3d& expected =
            geometry_it->second[static_cast<std::size_t>(corner_index)];
        const Eigen::Vector3d& imported =
            measurement.target_outer_corners_board[
                static_cast<std::size_t>(corner_index)];
        if ((expected - imported).norm() > 1e-10) {
          matches_existing_imported_geometry = false;
          break;
        }
      }
      if (!matches_existing_imported_geometry) {
        geometry_it->second = measurement.target_outer_corners_board;
      }
    }
  }

  std::vector<std::string> warnings;
  SolverState state = BuildSolverState(frame_inputs, board_corner_points, options_, &warnings);
  MarkReferenceConnectedComponent(options_.reference_board_id, &state);

  OuterBootstrapCameraIntrinsics intrinsics = MakeInitialIntrinsics(state.image_size, options_);
  ClampIntrinsicsInPlace(&intrinsics);

  if (state.boards.find(options_.reference_board_id) == state.boards.end() ||
      state.boards[options_.reference_board_id].connected_observation_count <= 0) {
    warnings.push_back("reference board " + std::to_string(options_.reference_board_id) +
                       " was not observed in the input measurements");
    return BuildFailureResult(options_, state, intrinsics, state.boards, state.frames, warnings,
                              "ReferenceBoardNotObserved");
  }

  std::map<int, BoardNode> boards = state.boards;
  std::vector<FrameNode> frames = state.frames;
  if (!InitializeConnectedComponent(options_, state, &boards, &frames, &warnings, intrinsics)) {
    return BuildFailureResult(options_, state, intrinsics, boards, frames, warnings,
                              "ReferenceConnectedComponentInitializationFailed");
  }

  const OuterBootstrapCameraIntrinsics anchor_intrinsics = intrinsics;
  OptimizeBootstrapState(options_, state, anchor_intrinsics, &intrinsics, &boards, &frames);

  OuterBootstrapResult result =
      BuildFailureResult(options_, state, intrinsics, boards, frames, warnings, "");
  result.success = true;
  result.failure_reason.clear();
  result.global_rmse =
      ComputeGlobalRmse(state, intrinsics, boards, frames,
                        &result.used_board_observation_count, &result.used_corner_count);
  result.used_frame_count = static_cast<int>(std::count_if(
      frames.begin(), frames.end(), [](const FrameNode& frame) { return frame.initialized; }));

  for (auto board_it = boards.begin(); board_it != boards.end(); ++board_it) {
    if (!board_it->second.reference_connected) {
      std::ostringstream warning;
      warning << "board " << board_it->second.board_id
              << " is outside the connected component of reference board "
              << options_.reference_board_id;
      result.warnings.push_back(warning.str());
    }
  }
  for (std::size_t frame_storage_index = 0; frame_storage_index < frames.size(); ++frame_storage_index) {
    if (!frames[frame_storage_index].reference_connected) {
      std::ostringstream warning;
      warning << "frame " << frames[frame_storage_index].frame_index
              << " is outside the connected component of reference board "
              << options_.reference_board_id;
      result.warnings.push_back(warning.str());
    }
  }

  if (!std::isfinite(result.global_rmse) || result.used_board_observation_count <= 0) {
    result.success = false;
    result.failure_reason = "NoInitializedConnectedObservations";
  }
  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
