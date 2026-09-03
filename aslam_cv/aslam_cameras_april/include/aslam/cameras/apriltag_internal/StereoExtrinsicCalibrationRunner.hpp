#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STEREO_EXTRINSIC_CALIBRATION_RUNNER_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STEREO_EXTRINSIC_CALIBRATION_RUNNER_HPP

#include <aslam/cameras/apriltag_internal/StereoExtrinsicProblemInput.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

class StereoExtrinsicCalibrationRunner {
 public:
  explicit StereoExtrinsicCalibrationRunner(
      StereoExtrinsicSolverOptions options = StereoExtrinsicSolverOptions{});

  StereoExtrinsicCalibrationResult Run(
      const StereoExtrinsicProblemInput& input) const;

 private:
  StereoExtrinsicSolverOptions options_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_STEREO_EXTRINSIC_CALIBRATION_RUNNER_HPP
