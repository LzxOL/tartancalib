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
  if (family == "omni-radtan") {
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
  if (family == "omni-radtan") {
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
  if (family == "omni-radtan") {
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
  if (family == "omni-radtan") {
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
  } else if (family == "omni-radtan") {
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
    cv::Mat* tvec) const {
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
  filtered_object_points.reserve(object_points.size());
  normalized_points.reserve(image_points.size());
  constexpr double kMaxRayAngleRadians =
      80.0 * 3.14159265358979323846 / 180.0;

  for (std::size_t i = 0; i < image_points.size(); ++i) {
    Eigen::Vector3d back_projection;
    if (!keypointToEuclidean(
            Eigen::Vector2d(image_points[i].x, image_points[i].y),
            &back_projection)) {
      continue;
    }
    const Eigen::Vector3d direction = back_projection.normalized();
    if (direction.z() <= std::cos(kMaxRayAngleRadians)) {
      continue;
    }

    filtered_object_points.push_back(object_points[i]);
    normalized_points.emplace_back(
        static_cast<float>(direction.x() / direction.z()),
        static_cast<float>(direction.y() / direction.z()));
  }

  if (filtered_object_points.size() < 4) {
    return false;
  }

  cv::Mat local_rvec;
  cv::Mat local_tvec;
  const cv::Mat identity_camera = cv::Mat::eye(3, 3, CV_64F);
  const cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

  bool success = false;
  if (filtered_object_points.size() == 4) {
    success = cv::solvePnP(filtered_object_points, normalized_points,
                           identity_camera, dist_coeffs, local_rvec, local_tvec,
                           false, cv::SOLVEPNP_IPPE);
  }
  if (!success) {
    success = cv::solvePnP(filtered_object_points, normalized_points,
                           identity_camera, dist_coeffs, local_rvec, local_tvec,
                           false, cv::SOLVEPNP_ITERATIVE);
  }
  if (!success) {
    return false;
  }

  success = cv::solvePnP(filtered_object_points, normalized_points,
                         identity_camera, dist_coeffs, local_rvec, local_tvec,
                         true, cv::SOLVEPNP_ITERATIVE);
  if (!success) {
    return false;
  }

  cv::Mat tvec64;
  local_tvec.convertTo(tvec64, CV_64F);
  if (tvec64.at<double>(2, 0) <= 0.0) {
    return false;
  }

  *rvec = local_rvec;
  *tvec = local_tvec;
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
