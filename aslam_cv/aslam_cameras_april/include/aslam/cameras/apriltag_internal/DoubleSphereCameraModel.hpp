#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_DOUBLE_SPHERE_CAMERA_MODEL_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_DOUBLE_SPHERE_CAMERA_MODEL_HPP

#include <string>
#include <vector>

#include <Eigen/Core>
#include <boost/shared_ptr.hpp>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagCanonicalModel.hpp>
#include <aslam/cameras/CameraGeometryBase.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

class DoubleSphereCameraModel {
 public:
  static DoubleSphereCameraModel FromConfig(const IntermediateCameraConfig& config);

  bool IsValid() const { return valid_; }
  const cv::Size& resolution() const { return resolution_; }
  const std::string& camera_model() const { return camera_model_; }
  const std::string& distortion_model() const { return distortion_model_; }
  std::string NormalizedFamilyString() const;
  const boost::shared_ptr<aslam::cameras::CameraGeometryBase>& geometry() const {
    return geometry_;
  }

  bool vsEuclideanToKeypoint(const Eigen::Vector3d& point, Eigen::Vector2d* keypoint) const;
  bool keypointToEuclidean(const Eigen::Vector2d& keypoint, Eigen::Vector3d* ray) const;
  bool estimateTransformation(const std::vector<cv::Point3f>& object_points,
                              const std::vector<cv::Point2f>& image_points,
                              cv::Mat* rvec,
                              cv::Mat* tvec) const;

 private:
  bool isValid(const Eigen::Vector2d& keypoint) const;

  bool valid_ = false;
  std::string camera_model_ = "ds";
  std::string distortion_model_ = "none";
  boost::shared_ptr<aslam::cameras::CameraGeometryBase> geometry_;
  double xi_ = 0.0;
  double alpha_ = 0.0;
  double beta_ = 1.0;
  double fu_ = 0.0;
  double fv_ = 0.0;
  double cu_ = 0.0;
  double cv_ = 0.0;
  std::vector<double> distortion_coeffs_;
  cv::Size resolution_{0, 0};
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_DOUBLE_SPHERE_CAMERA_MODEL_HPP
