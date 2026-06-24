#include <aslam/cameras/apriltag_internal/SymmetricStereoReprojectionError.hpp>

#include <Eigen/Core>
#include <iostream>

namespace ati = aslam::cameras::apriltag_internal;

int main() {
  const double intrinsics[4] = {400.0, 400.0, 320.0, 240.0};
  const double frame_pose[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  const double point[3] = {0.1, -0.05, 2.0};

  double ext_spread[6] = {0.0, 0.0, 0.0, 0.2, 0.0, 0.0};

  ati::SymmetricStereoReprojectionError cam0_error(
      0, 320.0, 240.0, Eigen::Vector3d(point[0], point[1], point[2]));
  ati::SymmetricStereoReprojectionError cam1_error(
      1, 320.0, 240.0, Eigen::Vector3d(point[0], point[1], point[2]));

  double cam0_residual[2] = {0.0, 0.0};
  double cam1_residual[2] = {0.0, 0.0};
  cam0_error(frame_pose, ext_spread, intrinsics, cam0_residual);
  cam1_error(frame_pose, ext_spread, intrinsics, cam1_residual);

  std::cout << "cam0_residual_px: " << cam0_residual[0] << ", "
            << cam0_residual[1] << "\n";
  std::cout << "cam1_residual_px: " << cam1_residual[0] << ", "
            << cam1_residual[1] << "\n";

  ext_spread[3] = 0.0;
  cam0_error(frame_pose, ext_spread, intrinsics, cam0_residual);
  cam1_error(frame_pose, ext_spread, intrinsics, cam1_residual);
  std::cout << "zero_spread_cam0_residual_px: " << cam0_residual[0] << ", "
            << cam0_residual[1] << "\n";
  std::cout << "zero_spread_cam1_residual_px: " << cam1_residual[0] << ", "
            << cam1_residual[1] << "\n";

  return 0;
}
