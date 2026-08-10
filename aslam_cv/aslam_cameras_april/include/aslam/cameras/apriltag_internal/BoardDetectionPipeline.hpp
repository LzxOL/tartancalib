#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_BOARD_DETECTION_PIPELINE_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_BOARD_DETECTION_PIPELINE_HPP

#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

// High-level boundary for the board frontend.  The implementation deliberately
// delegates to the existing detectors so this class only makes the data flow
// explicit; it does not introduce another detection or recovery policy.
class BoardDetectionPipeline {
 public:
  BoardDetectionPipeline(
      ApriltagInternalConfig config,
      ApriltagInternalDetectionOptions options =
          ApriltagInternalDetectionOptions{});

  // Stage 1: decode and validate the outer tag observations, including all
  // existing multi-scale and camera-aware recovery paths.
  OuterTagMultiDetectionResult DetectOuter(const cv::Mat& image) const;

  // Stage 2: regenerate board measurements from the outer observations using
  // the existing bootstrap/optimized-scene recovery logic.
  InternalRegenerationFrameResult RegenerateFrame(
      const cv::Mat& image,
      const InternalRegenerationFrameInput& frame_input,
      const OuterBootstrapResult& bootstrap_result) const;
  InternalRegenerationFrameResult RegenerateFrame(
      const cv::Mat& image,
      const InternalRegenerationFrameInput& frame_input,
      const JointReprojectionSceneState& scene_state) const;

  const MultiScaleOuterTagDetector& outer_detector() const {
    return outer_detector_;
  }
  const MultiBoardInternalMeasurementRegenerator& measurement_regenerator() const {
    return measurement_regenerator_;
  }

 private:
  MultiScaleOuterTagDetector outer_detector_;
  MultiBoardInternalMeasurementRegenerator measurement_regenerator_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_BOARD_DETECTION_PIPELINE_HPP
