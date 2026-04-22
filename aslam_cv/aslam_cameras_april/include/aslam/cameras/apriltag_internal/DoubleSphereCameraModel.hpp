#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_DOUBLE_SPHERE_CAMERA_MODEL_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_DOUBLE_SPHERE_CAMERA_MODEL_HPP

#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagCanonicalModel.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

class DoubleSphereCameraModel {
 public:
  static DoubleSphereCameraModel FromConfig(const IntermediateCameraConfig& config);

  bool IsValid() const { return valid_; }
  const cv::Size& resolution() const { return resolution_; }

  bool vsEuclideanToKeypoint(const Eigen::Vector3d& point, Eigen::Vector2d* keypoint) const;
  bool keypointToEuclidean(const Eigen::Vector2d& keypoint, Eigen::Vector3d* ray) const;
  bool estimateTransformation(const std::vector<cv::Point3f>& object_points,
                              const std::vector<cv::Point2f>& image_points,
                              cv::Mat* rvec,
                              cv::Mat* tvec) const;

 private:
  void updateTemporaries();
  bool isValid(const Eigen::Vector2d& keypoint) const;
  bool isUndistortedKeypointValid(double rho2_d) const;

  bool valid_ = false;
  double xi_ = 0.0;
  double alpha_ = 0.0;
  double fu_ = 0.0;
  double fv_ = 0.0;
  double cu_ = 0.0;
  double cv_ = 0.0;
  cv::Size resolution_{0, 0};

  double recip_fu_ = 0.0;
  double recip_fv_ = 0.0;
  double one_over_2alpha_m_1_ = 0.0;
  double fov_parameter_ = 0.0;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_DOUBLE_SPHERE_CAMERA_MODEL_HPP
