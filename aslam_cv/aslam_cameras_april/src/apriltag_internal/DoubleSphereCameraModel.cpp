#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>

#include <aslam/cameras.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

std::string Lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

std::string NormalizeCameraModelToken(const std::string& value) {
  const std::string lowered = Lowercase(value);
  if (lowered == "ds" || lowered == "double_sphere" || lowered == "double-sphere") {
    return "ds";
  }
  if (lowered == "eucm" || lowered == "extended_unified" || lowered == "extended-unified") {
    return "eucm";
  }
  if (lowered == "pinhole") {
    return "pinhole";
  }
  if (lowered == "mei" || lowered == "omni" || lowered == "omnidirectional") {
    return "omni";
  }
  return lowered;
}

std::string NormalizeDistortionModelToken(const std::string& value) {
  const std::string lowered = Lowercase(value);
  if (lowered.empty()) {
    return "none";
  }
  if (lowered == "none") {
    return "none";
  }
  if (lowered == "equi" || lowered == "equidistant") {
    return "equi";
  }
  if (lowered == "radtan" || lowered == "radial-tangential" ||
      lowered == "radial_tangential") {
    return "radtan";
  }
  return lowered;
}

std::string NormalizeFamilyString(const std::string& camera_model,
                                  const std::string& distortion_model) {
  const std::string normalized_camera = NormalizeCameraModelToken(camera_model);
  const std::string normalized_distortion =
      NormalizeDistortionModelToken(distortion_model);
  if (normalized_camera == "ds" && normalized_distortion == "none") {
    return "ds-none";
  }
  if (normalized_camera == "eucm" && normalized_distortion == "none") {
    return "eucm-none";
  }
  if (normalized_camera == "pinhole" && normalized_distortion == "equi") {
    return "pinhole-equi";
  }
  if (normalized_camera == "omni" && normalized_distortion == "radtan") {
    return "omni-radtan";
  }
  if (normalized_camera == "omni" && normalized_distortion == "none") {
    return "omni-none";
  }
  return "";
}

std::vector<double> DefaultEquidistantDistortionCoeffs(
    const std::vector<double>& input) {
  std::vector<double> result = input;
  result.resize(4, 0.0);
  if (result.size() > 4) {
    result.resize(4);
  }
  return result;
}

std::vector<double> DefaultRadtanDistortionCoeffs(
    const std::vector<double>& input) {
  std::vector<double> result = input;
  result.resize(4, 0.0);
  if (result.size() > 4) {
    result.resize(4);
  }
  return result;
}

boost::shared_ptr<CameraGeometryBase> MakeGeometry(
    const IntermediateCameraConfig& config) {
  const std::string family =
      NormalizeFamilyString(config.camera_model, config.distortion_model);
  if (family == "ds-none") {
    const std::vector<double>& intrinsics = config.intrinsics;
    DoubleSphereProjection<NoDistortion> projection(
        intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], intrinsics[4],
        intrinsics[5], config.resolution[0], config.resolution[1]);
    return boost::shared_ptr<CameraGeometryBase>(
        new DoubleSphereCameraGeometry(projection, GlobalShutter(), NoMask()));
  }
  if (family == "eucm-none") {
    const std::vector<double>& intrinsics = config.intrinsics;
    ExtendedUnifiedProjection<NoDistortion> projection(
        intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], intrinsics[4],
        intrinsics[5], config.resolution[0], config.resolution[1]);
    return boost::shared_ptr<CameraGeometryBase>(
        new ExtendedUnifiedCameraGeometry(projection, GlobalShutter(), NoMask()));
  }
  if (family == "pinhole-equi") {
    const std::vector<double>& intrinsics = config.intrinsics;
    const std::vector<double> distortion =
        DefaultEquidistantDistortionCoeffs(config.distortion_coeffs);
    EquidistantDistortion equidistant(
        distortion[0], distortion[1], distortion[2], distortion[3]);
    PinholeProjection<EquidistantDistortion> projection(
        intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], config.resolution[0],
        config.resolution[1], equidistant);
    return boost::shared_ptr<CameraGeometryBase>(
        new EquidistantDistortedPinholeCameraGeometry(
            projection, GlobalShutter(), NoMask()));
  }
  if (family == "omni-radtan") {
    const std::vector<double>& intrinsics = config.intrinsics;
    const std::vector<double> distortion =
        DefaultRadtanDistortionCoeffs(config.distortion_coeffs);
    RadialTangentialDistortion radtan(
        distortion[0], distortion[1], distortion[2], distortion[3]);
    OmniProjection<RadialTangentialDistortion> projection(
        intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], intrinsics[4],
        config.resolution[0], config.resolution[1], radtan);
    return boost::shared_ptr<CameraGeometryBase>(
        new DistortedOmniCameraGeometry(
            projection, GlobalShutter(), NoMask()));
  }
  if (family == "omni-none") {
    const std::vector<double>& intrinsics = config.intrinsics;
    OmniProjection<NoDistortion> projection(
        intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], intrinsics[4],
        config.resolution[0], config.resolution[1]);
    return boost::shared_ptr<CameraGeometryBase>(
        new OmniCameraGeometry(projection, GlobalShutter(), NoMask()));
  }
  throw std::runtime_error(
      "Unsupported camera family: camera_model=" + config.camera_model +
      " distortion_model=" + config.distortion_model);
}

bool HasExpectedIntrinsicsSize(const std::string& family, std::size_t size) {
  if (family == "ds-none" || family == "eucm-none") {
    return size == 6;
  }
  if (family == "pinhole-equi") {
    return size == 4;
  }
  if (family == "omni-radtan" || family == "omni-none") {
    return size == 5;
  }
  return false;
}

std::vector<double> ClampDistortionCoeffs(const std::vector<double>& values) {
  std::vector<double> result = DefaultEquidistantDistortionCoeffs(values);
  for (double& entry : result) {
    entry = std::max(-2.0, std::min(2.0, entry));
  }
  return result;
}

}  // namespace

std::string OuterBootstrapCameraIntrinsics::NormalizedCameraModel() const {
  return NormalizeCameraModelToken(camera_model);
}

std::string OuterBootstrapCameraIntrinsics::NormalizedDistortionModel() const {
  return NormalizeDistortionModelToken(distortion_model);
}

std::string OuterBootstrapCameraIntrinsics::NormalizedFamilyString() const {
  return NormalizeFamilyString(camera_model, distortion_model);
}

std::vector<double> OuterBootstrapCameraIntrinsics::IntrinsicsVector() const {
  const std::string family = NormalizedFamilyString();
  if (family == "ds-none") {
    return {xi, alpha, fu, fv, cu, cv};
  }
  if (family == "eucm-none") {
    return {alpha, beta, fu, fv, cu, cv};
  }
  if (family == "pinhole-equi") {
    return {fu, fv, cu, cv};
  }
  if (family == "omni-radtan" || family == "omni-none") {
    return {xi, fu, fv, cu, cv};
  }
  return {};
}

std::vector<double> OuterBootstrapCameraIntrinsics::DistortionVector() const {
  if (NormalizedFamilyString() == "pinhole-equi") {
    return DefaultEquidistantDistortionCoeffs(distortion_coeffs);
  }
  if (NormalizedFamilyString() == "omni-radtan") {
    return DefaultRadtanDistortionCoeffs(distortion_coeffs);
  }
  return {};
}

std::vector<double> OuterBootstrapCameraIntrinsics::CombinedParameterVector() const {
  std::vector<double> result = IntrinsicsVector();
  const std::vector<double> distortion = DistortionVector();
  result.insert(result.end(), distortion.begin(), distortion.end());
  return result;
}

std::vector<std::string> OuterBootstrapCameraIntrinsics::IntrinsicsLabels() const {
  const std::string family = NormalizedFamilyString();
  if (family == "ds-none") {
    return {"xi", "alpha", "fu", "fv", "cu", "cv"};
  }
  if (family == "eucm-none") {
    return {"alpha", "beta", "fu", "fv", "cu", "cv"};
  }
  if (family == "pinhole-equi") {
    return {"fu", "fv", "cu", "cv"};
  }
  if (family == "omni-radtan" || family == "omni-none") {
    return {"xi", "fu", "fv", "cu", "cv"};
  }
  return {};
}

std::vector<std::string> OuterBootstrapCameraIntrinsics::DistortionLabels() const {
  if (NormalizedFamilyString() == "pinhole-equi") {
    return {"k1", "k2", "k3", "k4"};
  }
  if (NormalizedFamilyString() == "omni-radtan") {
    return {"k1", "k2", "p1", "p2"};
  }
  return {};
}

std::vector<std::string> OuterBootstrapCameraIntrinsics::CombinedParameterLabels() const {
  std::vector<std::string> result = IntrinsicsLabels();
  const std::vector<std::string> distortion = DistortionLabels();
  result.insert(result.end(), distortion.begin(), distortion.end());
  return result;
}

bool OuterBootstrapCameraIntrinsics::SetIntrinsicsVector(const std::vector<double>& values) {
  const std::string family = NormalizedFamilyString();
  if (!HasExpectedIntrinsicsSize(family, values.size())) {
    return false;
  }
  if (family == "ds-none") {
    xi = values[0];
    alpha = values[1];
    fu = values[2];
    fv = values[3];
    cu = values[4];
    cv = values[5];
    return true;
  }
  if (family == "eucm-none") {
    alpha = values[0];
    beta = values[1];
    fu = values[2];
    fv = values[3];
    cu = values[4];
    cv = values[5];
    return true;
  }
  if (family == "pinhole-equi") {
    fu = values[0];
    fv = values[1];
    cu = values[2];
    cv = values[3];
    return true;
  }
  if (family == "omni-radtan" || family == "omni-none") {
    xi = values[0];
    fu = values[1];
    fv = values[2];
    cu = values[3];
    cv = values[4];
    return true;
  }
  return false;
}

bool OuterBootstrapCameraIntrinsics::SetDistortionVector(const std::vector<double>& values) {
  if (NormalizedFamilyString() == "pinhole-equi") {
    distortion_coeffs = DefaultEquidistantDistortionCoeffs(values);
    return true;
  }
  if (NormalizedFamilyString() == "omni-radtan") {
    distortion_coeffs = DefaultRadtanDistortionCoeffs(values);
    return true;
  }
  if (!values.empty()) {
    return false;
  }
  distortion_coeffs.clear();
  return true;
}

bool OuterBootstrapCameraIntrinsics::SetCombinedParameterVector(
    const std::vector<double>& values) {
  const std::size_t intrinsics_size = IntrinsicsLabels().size();
  if (values.size() < intrinsics_size) {
    return false;
  }
  const std::vector<double> intrinsics(values.begin(),
                                       values.begin() + intrinsics_size);
  const std::vector<double> distortion(values.begin() + intrinsics_size,
                                       values.end());
  return SetIntrinsicsVector(intrinsics) && SetDistortionVector(distortion);
}

DoubleSphereCameraModel DoubleSphereCameraModel::FromConfig(
    const IntermediateCameraConfig& config) {
  const std::string family =
      NormalizeFamilyString(config.camera_model, config.distortion_model);
  if (family.empty()) {
    throw std::runtime_error(
        "Unsupported camera_model/distortion_model combination: camera_model=" +
        config.camera_model + " distortion_model=" + config.distortion_model);
  }
  if (!HasExpectedIntrinsicsSize(family, config.intrinsics.size())) {
    throw std::runtime_error(
        "Malformed intrinsics for family " + family + ".");
  }
  if (config.resolution.size() != 2) {
    throw std::runtime_error("Camera resolution must be [width, height].");
  }
  if (family != "pinhole-equi" && family != "omni-radtan" &&
      !config.distortion_coeffs.empty()) {
    throw std::runtime_error(
        "This camera family expects empty distortion_coeffs.");
  }

  DoubleSphereCameraModel camera;
  camera.camera_model_ = NormalizeCameraModelToken(config.camera_model);
  camera.distortion_model_ = NormalizeDistortionModelToken(config.distortion_model);
  camera.resolution_ = cv::Size(config.resolution[0], config.resolution[1]);
  camera.geometry_ = MakeGeometry(config);
  camera.distortion_coeffs_ = ClampDistortionCoeffs(config.distortion_coeffs);
  if (family == "ds-none") {
    camera.xi_ = config.intrinsics[0];
    camera.alpha_ = config.intrinsics[1];
    camera.fu_ = config.intrinsics[2];
    camera.fv_ = config.intrinsics[3];
    camera.cu_ = config.intrinsics[4];
    camera.cv_ = config.intrinsics[5];
  } else if (family == "eucm-none") {
    camera.alpha_ = config.intrinsics[0];
    camera.beta_ = config.intrinsics[1];
    camera.fu_ = config.intrinsics[2];
    camera.fv_ = config.intrinsics[3];
    camera.cu_ = config.intrinsics[4];
    camera.cv_ = config.intrinsics[5];
  } else if (family == "pinhole-equi") {
    camera.fu_ = config.intrinsics[0];
    camera.fv_ = config.intrinsics[1];
    camera.cu_ = config.intrinsics[2];
    camera.cv_ = config.intrinsics[3];
  } else if (family == "omni-radtan" || family == "omni-none") {
    camera.xi_ = config.intrinsics[0];
    camera.fu_ = config.intrinsics[1];
    camera.fv_ = config.intrinsics[2];
    camera.cu_ = config.intrinsics[3];
    camera.cv_ = config.intrinsics[4];
  }
  camera.valid_ = true;
  return camera;
}

std::string DoubleSphereCameraModel::NormalizedFamilyString() const {
  return NormalizeFamilyString(camera_model_, distortion_model_);
}

bool DoubleSphereCameraModel::vsEuclideanToKeypoint(const Eigen::Vector3d& point,
                                                    Eigen::Vector2d* keypoint) const {
  if (keypoint == nullptr) {
    throw std::runtime_error("vsEuclideanToKeypoint requires a valid output pointer.");
  }
  if (!valid_ || geometry_ == nullptr) {
    return false;
  }
  Eigen::VectorXd projected;
  if (!geometry_->vsEuclideanToKeypoint(point, projected) ||
      projected.rows() != 2 || !projected.allFinite()) {
    return false;
  }
  (*keypoint)[0] = projected[0];
  (*keypoint)[1] = projected[1];
  return isValid(*keypoint);
}

bool DoubleSphereCameraModel::keypointToEuclidean(const Eigen::Vector2d& keypoint,
                                                  Eigen::Vector3d* ray) const {
  if (ray == nullptr) {
    throw std::runtime_error("keypointToEuclidean requires a valid output pointer.");
  }
  if (!valid_ || geometry_ == nullptr) {
    return false;
  }
  Eigen::Vector3d lifted = Eigen::Vector3d::Zero();
  if (!geometry_->vsKeypointToEuclidean(keypoint, lifted) || !lifted.allFinite()) {
    return false;
  }
  *ray = lifted;
  return true;
}

bool DoubleSphereCameraModel::estimateTransformation(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    cv::Mat* rvec,
    cv::Mat* tvec,
    const cv::Mat* initial_rvec,
    const cv::Mat* initial_tvec) const {
  if (rvec == nullptr || tvec == nullptr) {
    throw std::runtime_error("estimateTransformation requires valid output pointers.");
  }
  if (!valid_ || geometry_ == nullptr) {
    return false;
  }
  if (object_points.size() != image_points.size() || object_points.size() < 4) {
    return false;
  }

  std::vector<cv::Point3f> filtered_object_points;
  std::vector<cv::Point2f> normalized_points;
  std::vector<Eigen::Vector3d> observed_directions;
  filtered_object_points.reserve(object_points.size());
  normalized_points.reserve(image_points.size());
  observed_directions.reserve(image_points.size());
  Eigen::Vector3d mean_direction = Eigen::Vector3d::Zero();

  for (std::size_t i = 0; i < image_points.size(); ++i) {
    Eigen::Vector3d back_projection;
    if (!keypointToEuclidean(
            Eigen::Vector2d(image_points[i].x, image_points[i].y),
            &back_projection)) {
      continue;
    }
    const Eigen::Vector3d direction = back_projection.normalized();
    if (!direction.allFinite()) {
      continue;
    }
    filtered_object_points.push_back(object_points[i]);
    observed_directions.push_back(direction);
    mean_direction += direction;
  }

  if (filtered_object_points.size() < 4 || mean_direction.norm() < 1e-9) {
    return false;
  }

  const Eigen::Vector3d tangent_z = mean_direction.normalized();
  Eigen::Vector3d tangent_reference = Eigen::Vector3d::UnitY();
  if (std::abs(tangent_reference.dot(tangent_z)) > 0.95) {
    tangent_reference = Eigen::Vector3d::UnitX();
  }
  const Eigen::Vector3d tangent_x =
      tangent_reference.cross(tangent_z).normalized();
  const Eigen::Vector3d tangent_y = tangent_z.cross(tangent_x).normalized();
  Eigen::Matrix3d R_tangent_camera;
  R_tangent_camera.row(0) = tangent_x.transpose();
  R_tangent_camera.row(1) = tangent_y.transpose();
  R_tangent_camera.row(2) = tangent_z.transpose();
  if (!R_tangent_camera.allFinite()) {
    return false;
  }
  for (const Eigen::Vector3d& direction : observed_directions) {
    const Eigen::Vector3d tangent_direction =
        R_tangent_camera * direction;
    if (!tangent_direction.allFinite() || tangent_direction.z() <= 1e-3) {
      return false;
    }
    normalized_points.emplace_back(
        static_cast<float>(tangent_direction.x() / tangent_direction.z()),
        static_cast<float>(tangent_direction.y() / tangent_direction.z()));
  }

  const cv::Mat identity_camera = cv::Mat::eye(3, 3, CV_64F);
  const cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
  cv::Mat best_rvec;
  cv::Mat best_tvec;
  double best_rmse = std::numeric_limits<double>::infinity();

  auto evaluate_candidate =
      [this, &object_points, &image_points](const cv::Mat& candidate_rvec,
                                           const cv::Mat& candidate_tvec) {
        cv::Mat rotation_cv;
        cv::Rodrigues(candidate_rvec, rotation_cv);
        cv::Mat rotation64;
        cv::Mat translation64;
        rotation_cv.convertTo(rotation64, CV_64F);
        candidate_tvec.convertTo(translation64, CV_64F);
        if (rotation64.rows != 3 || rotation64.cols != 3 ||
            translation64.total() < 3u ||
            translation64.at<double>(2, 0) <= 0.0) {
          return std::numeric_limits<double>::infinity();
        }
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Zero();
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            rotation(row, column) = rotation64.at<double>(row, column);
          }
        }
        const Eigen::Vector3d translation(
            translation64.at<double>(0, 0),
            translation64.at<double>(1, 0),
            translation64.at<double>(2, 0));
        if (!rotation.allFinite() || !translation.allFinite()) {
          return std::numeric_limits<double>::infinity();
        }

        double squared_error_sum = 0.0;
        for (std::size_t index = 0; index < object_points.size(); ++index) {
          const cv::Point3f& object = object_points[index];
          const Eigen::Vector3d point_camera =
              rotation * Eigen::Vector3d(object.x, object.y, object.z) +
              translation;
          Eigen::Vector2d projected = Eigen::Vector2d::Zero();
          if (!vsEuclideanToKeypoint(point_camera, &projected) ||
              !projected.allFinite()) {
            return std::numeric_limits<double>::infinity();
          }
          const double dx =
              projected.x() - static_cast<double>(image_points[index].x);
          const double dy =
              projected.y() - static_cast<double>(image_points[index].y);
          squared_error_sum += dx * dx + dy * dy;
        }
        return std::sqrt(squared_error_sum /
                         static_cast<double>(object_points.size()));
      };

  auto tangent_pose_to_camera =
      [&R_tangent_camera](const cv::Mat& tangent_rvec,
                          const cv::Mat& tangent_tvec,
                          cv::Mat* camera_rvec,
                          cv::Mat* camera_tvec) {
        if (camera_rvec == nullptr || camera_tvec == nullptr) {
          return false;
        }
        cv::Mat tangent_rotation_cv;
        cv::Rodrigues(tangent_rvec, tangent_rotation_cv);
        cv::Mat tangent_rotation64;
        cv::Mat tangent_translation64;
        tangent_rotation_cv.convertTo(tangent_rotation64, CV_64F);
        tangent_tvec.convertTo(tangent_translation64, CV_64F);
        if (tangent_rotation64.rows != 3 || tangent_rotation64.cols != 3 ||
            tangent_translation64.total() < 3u) {
          return false;
        }
        Eigen::Matrix3d R_tangent_object = Eigen::Matrix3d::Zero();
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            R_tangent_object(row, column) =
                tangent_rotation64.at<double>(row, column);
          }
        }
        const Eigen::Vector3d t_tangent_object(
            tangent_translation64.at<double>(0, 0),
            tangent_translation64.at<double>(1, 0),
            tangent_translation64.at<double>(2, 0));
        const Eigen::Matrix3d R_camera_object =
            R_tangent_camera.transpose() * R_tangent_object;
        const Eigen::Vector3d t_camera_object =
            R_tangent_camera.transpose() * t_tangent_object;
        if (!R_camera_object.allFinite() || !t_camera_object.allFinite()) {
          return false;
        }
        cv::Mat camera_rotation(3, 3, CV_64F);
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            camera_rotation.at<double>(row, column) =
                R_camera_object(row, column);
          }
        }
        cv::Rodrigues(camera_rotation, *camera_rvec);
        *camera_tvec = (cv::Mat_<double>(3, 1)
            << t_camera_object.x(), t_camera_object.y(), t_camera_object.z());
        return true;
      };

  // A persistent scene pose is useful as a local optimizer seed for a
  // camera-aware rescued quad, but it must never bypass the current image
  // measurement.  Convert the seed into the tangent camera frame used by
  // solvePnP, refine it with the current four rays, and let the same camera
  // reprojection objective decide whether it is usable.
  if (initial_rvec != nullptr && initial_tvec != nullptr &&
      !initial_rvec->empty() && !initial_tvec->empty()) {
    // Keep the unrefined continuous pose as its own hypothesis. Tangent-plane
    // refinement can switch planar branches for close-edge boards even when
    // the previous pose remains valid under the updated camera.
    const double initial_candidate_rmse =
        evaluate_candidate(*initial_rvec, *initial_tvec);
    if (initial_candidate_rmse < best_rmse) {
      best_rmse = initial_candidate_rmse;
      best_rvec = initial_rvec->clone();
      best_tvec = initial_tvec->clone();
    }
    cv::Mat initial_rotation_cv;
    cv::Rodrigues(*initial_rvec, initial_rotation_cv);
    cv::Mat initial_rotation64;
    cv::Mat initial_translation64;
    initial_rotation_cv.convertTo(initial_rotation64, CV_64F);
    initial_tvec->convertTo(initial_translation64, CV_64F);
    if (initial_rotation64.rows == 3 && initial_rotation64.cols == 3 &&
        initial_translation64.total() == 3u) {
      Eigen::Matrix3d initial_rotation = Eigen::Matrix3d::Zero();
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          initial_rotation(row, column) = initial_rotation64.at<double>(row, column);
        }
      }
      const Eigen::Vector3d initial_translation(
          initial_translation64.at<double>(0, 0),
          initial_translation64.at<double>(1, 0),
          initial_translation64.at<double>(2, 0));
      const Eigen::Matrix3d tangent_rotation =
          R_tangent_camera * initial_rotation;
      const Eigen::Vector3d tangent_translation =
          R_tangent_camera * initial_translation;
      if (tangent_rotation.allFinite() && tangent_translation.allFinite() &&
          tangent_translation.z() > 0.0) {
        cv::Mat tangent_rotation_cv(3, 3, CV_64F);
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            tangent_rotation_cv.at<double>(row, column) =
                tangent_rotation(row, column);
          }
        }
        cv::Mat seed_rvec;
        cv::Rodrigues(tangent_rotation_cv, seed_rvec);
        cv::Mat seed_tvec = (cv::Mat_<double>(3, 1)
            << tangent_translation.x(), tangent_translation.y(),
               tangent_translation.z());
        cv::Mat refined_seed_rvec = seed_rvec.clone();
        cv::Mat refined_seed_tvec = seed_tvec.clone();
        bool refined = false;
        try {
          refined = cv::solvePnP(filtered_object_points,
                                 normalized_points,
                                 identity_camera,
                                 dist_coeffs,
                                 refined_seed_rvec,
                                 refined_seed_tvec,
                                 true,
                                 cv::SOLVEPNP_ITERATIVE);
        } catch (const cv::Exception&) {
          refined = false;
        }
        cv::Mat candidate_rvec;
        cv::Mat candidate_tvec;
        if (refined) {
          tangent_pose_to_camera(refined_seed_rvec,
                                 refined_seed_tvec,
                                 &candidate_rvec,
                                 &candidate_tvec);
        } else {
          candidate_rvec = *initial_rvec;
          candidate_tvec = *initial_tvec;
        }
        const double candidate_rmse =
            evaluate_candidate(candidate_rvec, candidate_tvec);
        if (candidate_rmse < best_rmse) {
          best_rmse = candidate_rmse;
          best_rvec = candidate_rvec.clone();
          best_tvec = candidate_tvec.clone();
        }
      }
    }
  }

  std::vector<cv::Mat> candidate_rvecs;
  std::vector<cv::Mat> candidate_tvecs;
  try {
    cv::solvePnPGeneric(filtered_object_points,
                        normalized_points,
                        identity_camera,
                        dist_coeffs,
                        candidate_rvecs,
                        candidate_tvecs,
                        false,
                        cv::SOLVEPNP_IPPE);
  } catch (const cv::Exception&) {
    candidate_rvecs.clear();
    candidate_tvecs.clear();
  }

  const std::size_t candidate_count =
      std::min(candidate_rvecs.size(), candidate_tvecs.size());
  for (std::size_t candidate_index = 0;
       candidate_index < candidate_count;
       ++candidate_index) {
    cv::Mat refined_rvec = candidate_rvecs[candidate_index].clone();
    cv::Mat refined_tvec = candidate_tvecs[candidate_index].clone();
    bool refined = false;
    try {
      refined = cv::solvePnP(filtered_object_points,
                             normalized_points,
                             identity_camera,
                             dist_coeffs,
                             refined_rvec,
                             refined_tvec,
                             true,
                             cv::SOLVEPNP_ITERATIVE);
    } catch (const cv::Exception&) {
      refined = false;
    }
    if (!refined) {
      refined_rvec = candidate_rvecs[candidate_index];
      refined_tvec = candidate_tvecs[candidate_index];
    }
    cv::Mat camera_candidate_rvec;
    cv::Mat camera_candidate_tvec;
    if (!tangent_pose_to_camera(refined_rvec,
                                refined_tvec,
                                &camera_candidate_rvec,
                                &camera_candidate_tvec)) {
      continue;
    }
    const double candidate_rmse =
        evaluate_candidate(camera_candidate_rvec, camera_candidate_tvec);
    if (candidate_rmse < best_rmse) {
      best_rmse = candidate_rmse;
      best_rvec = camera_candidate_rvec.clone();
      best_tvec = camera_candidate_tvec.clone();
    }
  }

  bool success = !best_rvec.empty() && !best_tvec.empty() &&
                 std::isfinite(best_rmse);
  if (!success) {
    cv::Mat local_rvec;
    cv::Mat local_tvec;
    const int method = filtered_object_points.size() >= 6u
                           ? cv::SOLVEPNP_ITERATIVE
                           : cv::SOLVEPNP_SQPNP;
    try {
      success = cv::solvePnP(filtered_object_points, normalized_points,
                             identity_camera, dist_coeffs, local_rvec,
                             local_tvec, false, method);
    } catch (const cv::Exception&) {
      success = false;
    }
    if (success) {
      try {
        success = cv::solvePnP(filtered_object_points,
                               normalized_points,
                               identity_camera,
                               dist_coeffs,
                               local_rvec,
                               local_tvec,
                               true,
                               cv::SOLVEPNP_ITERATIVE);
      } catch (const cv::Exception&) {
        success = false;
      }
    }
    cv::Mat camera_fallback_rvec;
    cv::Mat camera_fallback_tvec;
    const bool converted =
        success && tangent_pose_to_camera(local_rvec,
                                          local_tvec,
                                          &camera_fallback_rvec,
                                          &camera_fallback_tvec);
    if (converted &&
        std::isfinite(evaluate_candidate(camera_fallback_rvec,
                                         camera_fallback_tvec))) {
      best_rvec = camera_fallback_rvec;
      best_tvec = camera_fallback_tvec;
    } else {
      return false;
    }
  }

  cv::Mat tvec64;
  best_tvec.convertTo(tvec64, CV_64F);
  if (tvec64.at<double>(2, 0) <= 0.0) {
    return false;
  }

  *rvec = best_rvec;
  *tvec = best_tvec;
  return true;
}

bool DoubleSphereCameraModel::isValid(const Eigen::Vector2d& keypoint) const {
  return keypoint[0] >= 0.0 &&
         keypoint[0] < static_cast<double>(resolution_.width) &&
         keypoint[1] >= 0.0 &&
         keypoint[1] < static_cast<double>(resolution_.height);
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
