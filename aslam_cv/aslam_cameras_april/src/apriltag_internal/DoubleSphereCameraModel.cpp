#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <opencv2/calib3d.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

DoubleSphereCameraModel DoubleSphereCameraModel::FromConfig(const IntermediateCameraConfig& config) {
  if (config.camera_model != "ds") {
    throw std::runtime_error("Only Kalibr/TartanCalib double-sphere camera_model=ds is supported.");
  }
  if (!config.distortion_model.empty() && config.distortion_model != "none") {
    throw std::runtime_error("Double-sphere comparison path expects distortion_model=none.");
  }
  if (config.intrinsics.size() != 6) {
    throw std::runtime_error("Double-sphere intrinsics must be [xi, alpha, fu, fv, cu, cv].");
  }
  if (config.resolution.size() != 2) {
    throw std::runtime_error("Camera resolution must be [width, height].");
  }
  if (!config.distortion_coeffs.empty()) {
    throw std::runtime_error("Double-sphere comparison path expects empty distortion_coeffs.");
  }

  DoubleSphereCameraModel camera;
  camera.xi_ = config.intrinsics[0];
  camera.alpha_ = config.intrinsics[1];
  camera.fu_ = config.intrinsics[2];
  camera.fv_ = config.intrinsics[3];
  camera.cu_ = config.intrinsics[4];
  camera.cv_ = config.intrinsics[5];
  camera.resolution_ = cv::Size(config.resolution[0], config.resolution[1]);
  camera.updateTemporaries();
  camera.valid_ = true;
  return camera;
}

bool DoubleSphereCameraModel::vsEuclideanToKeypoint(const Eigen::Vector3d& point,
                                                    Eigen::Vector2d* keypoint) const {
  if (keypoint == nullptr) {
    throw std::runtime_error("vsEuclideanToKeypoint requires a valid output pointer.");
  }
  if (!valid_) {
    return false;
  }

  const double x = point.x();
  const double y = point.y();
  const double z = point.z();
  const double r2 = x * x + y * y;
  const double d1 = std::sqrt(r2 + z * z);

  if (z <= -(fov_parameter_ * d1)) {
    return false;
  }

  const double k = xi_ * d1 + z;
  const double d2 = std::sqrt(r2 + k * k);
  const double norm = alpha_ * d2 + (1.0 - alpha_) * k;
  if (std::abs(norm) < 1e-12) {
    return false;
  }

  const double inv_norm = 1.0 / norm;
  (*keypoint)[0] = fu_ * x * inv_norm + cu_;
  (*keypoint)[1] = fv_ * y * inv_norm + cv_;
  return isValid(*keypoint);
}

bool DoubleSphereCameraModel::keypointToEuclidean(const Eigen::Vector2d& keypoint,
                                                  Eigen::Vector3d* ray) const {
  if (ray == nullptr) {
    throw std::runtime_error("keypointToEuclidean requires a valid output pointer.");
  }
  if (!valid_) {
    return false;
  }

  const double mx = recip_fu_ * (keypoint[0] - cu_);
  const double my = recip_fv_ * (keypoint[1] - cv_);
  const double r2 = mx * mx + my * my;
  if (!isUndistortedKeypointValid(r2)) {
    return false;
  }

  const double mz = (1.0 - alpha_ * alpha_ * r2) /
                    (alpha_ * std::sqrt(1.0 - (2.0 * alpha_ - 1.0) * r2) + 1.0 - alpha_);
  const double mz2 = mz * mz;
  const double k =
      (mz * xi_ + std::sqrt(mz2 + (1.0 - xi_ * xi_) * r2)) / (mz2 + r2);

  (*ray)[0] = k * mx;
  (*ray)[1] = k * my;
  (*ray)[2] = k * mz - xi_;
  return true;
}

bool DoubleSphereCameraModel::estimateTransformation(const std::vector<cv::Point3f>& object_points,
                                                     const std::vector<cv::Point2f>& image_points,
                                                     cv::Mat* rvec,
                                                     cv::Mat* tvec) const {
  if (rvec == nullptr || tvec == nullptr) {
    throw std::runtime_error("estimateTransformation requires valid output pointers.");
  }
  if (object_points.size() != image_points.size() || object_points.size() < 4) {
    return false;
  }

  std::vector<cv::Point3f> filtered_object_points;
  std::vector<cv::Point2f> normalized_points;
  filtered_object_points.reserve(object_points.size());
  normalized_points.reserve(image_points.size());
  constexpr double kMaxRayAngleRadians = 80.0 * 3.14159265358979323846 / 180.0;

  for (std::size_t i = 0; i < image_points.size(); ++i) {
    Eigen::Vector3d back_projection;
    if (!keypointToEuclidean(Eigen::Vector2d(image_points[i].x, image_points[i].y), &back_projection)) {
      continue;
    }
    const Eigen::Vector3d direction = back_projection.normalized();
    if (direction.z() <= std::cos(kMaxRayAngleRadians)) {
      continue;
    }

    filtered_object_points.push_back(object_points[i]);
    normalized_points.emplace_back(static_cast<float>(direction.x() / direction.z()),
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
    success = cv::solvePnP(filtered_object_points, normalized_points, identity_camera, dist_coeffs,
                           local_rvec, local_tvec, false, cv::SOLVEPNP_IPPE);
  }
  if (!success) {
    success = cv::solvePnP(filtered_object_points, normalized_points, identity_camera, dist_coeffs,
                           local_rvec, local_tvec, false, cv::SOLVEPNP_ITERATIVE);
  }
  if (!success) {
    return false;
  }

  success = cv::solvePnP(filtered_object_points, normalized_points, identity_camera, dist_coeffs,
                         local_rvec, local_tvec, true, cv::SOLVEPNP_ITERATIVE);
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

void DoubleSphereCameraModel::updateTemporaries() {
  if (std::abs(fu_) < 1e-12 || std::abs(fv_) < 1e-12) {
    throw std::runtime_error("Double-sphere focal length must be non-zero.");
  }
  recip_fu_ = 1.0 / fu_;
  recip_fv_ = 1.0 / fv_;
  one_over_2alpha_m_1_ =
      alpha_ > 0.5 ? 1.0 / (2.0 * alpha_ - 1.0) : std::numeric_limits<double>::max();
  const double temp = alpha_ <= 0.5 ? alpha_ / (1.0 - alpha_) : (1.0 - alpha_) / alpha_;
  fov_parameter_ = (temp + xi_) / std::sqrt(2.0 * temp * xi_ + xi_ * xi_ + 1.0);
}

bool DoubleSphereCameraModel::isValid(const Eigen::Vector2d& keypoint) const {
  return keypoint[0] >= 0.0 && keypoint[0] < static_cast<double>(resolution_.width) &&
         keypoint[1] >= 0.0 && keypoint[1] < static_cast<double>(resolution_.height);
}

bool DoubleSphereCameraModel::isUndistortedKeypointValid(double rho2_d) const {
  return alpha_ <= 0.5 || rho2_d <= one_over_2alpha_m_1_;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
