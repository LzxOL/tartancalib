#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_CORNER_MEASUREMENT_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_CORNER_MEASUREMENT_HPP

#include <Eigen/Core>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

enum class CornerType {
  Outer,
  LCorner,
  XCorner,
};

struct CornerMeasurement {
  int board_id = -1;
  int point_id = -1;
  Eigen::Vector2d image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector3d target_xyz = Eigen::Vector3d::Zero();
  bool valid = false;
  CornerType corner_type = CornerType::LCorner;
  double quality = 0.0;
};

const char* ToString(CornerType corner_type);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_CORNER_MEASUREMENT_HPP
