#include <aslam/cameras/apriltag_internal/BoardDetectionPipeline.hpp>

#include <utility>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

BoardDetectionPipeline::BoardDetectionPipeline(
    ApriltagInternalConfig config,
    ApriltagInternalDetectionOptions options)
    : outer_detector_(config.outer_detector_config),
      measurement_regenerator_(std::move(config), std::move(options)) {}

OuterTagMultiDetectionResult BoardDetectionPipeline::DetectOuter(
    const cv::Mat& image) const {
  return outer_detector_.DetectMultiple(image);
}

InternalRegenerationFrameResult BoardDetectionPipeline::RegenerateFrame(
    const cv::Mat& image,
    const InternalRegenerationFrameInput& frame_input,
    const OuterBootstrapResult& bootstrap_result) const {
  return measurement_regenerator_.RegenerateFrame(image, frame_input,
                                                  bootstrap_result);
}

InternalRegenerationFrameResult BoardDetectionPipeline::RegenerateFrame(
    const cv::Mat& image,
    const InternalRegenerationFrameInput& frame_input,
    const JointReprojectionSceneState& scene_state) const {
  return measurement_regenerator_.RegenerateFrame(image, frame_input,
                                                  scene_state);
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
