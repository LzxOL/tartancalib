#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_INTERNAL_MEASUREMENT_REGENERATOR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_INTERNAL_MEASUREMENT_REGENERATOR_HPP

#include <string>
#include <vector>

#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct InternalRegeneratedBoardResult {
  int board_id = -1;
  bool success = false;
  std::vector<CornerMeasurement> corners;
  ApriltagInternalDetectionResult detection;

  int ValidInternalCornerCount() const {
    int count = 0;
    for (const CornerMeasurement& corner : corners) {
      if (corner.valid && corner.corner_type != CornerType::Outer) {
        ++count;
      }
    }
    return count;
  }
};

struct InternalRegenerationFrameInput {
  int frame_index = -1;
  std::string frame_label;
  OuterTagMultiDetectionResult outer_detections;
};

struct InternalRegenerationFrameResult {
  int frame_index = -1;
  std::string frame_label;
  std::vector<InternalRegeneratedBoardResult> board_results;

  int SuccessfulBoardCount() const {
    int count = 0;
    for (const InternalRegeneratedBoardResult& board_result : board_results) {
      if (board_result.success) {
        ++count;
      }
    }
    return count;
  }

  int ValidInternalCornerCount() const {
    int count = 0;
    for (const InternalRegeneratedBoardResult& board_result : board_results) {
      count += board_result.ValidInternalCornerCount();
    }
    return count;
  }
};

class MultiBoardInternalMeasurementRegenerator {
 public:
  MultiBoardInternalMeasurementRegenerator(
      ApriltagInternalConfig config,
      ApriltagInternalDetectionOptions options = ApriltagInternalDetectionOptions{});

  InternalRegenerationFrameResult RegenerateFrame(
      const cv::Mat& image,
      const InternalRegenerationFrameInput& frame_input,
      const OuterBootstrapResult& bootstrap_result) const;

 private:
  ApriltagInternalConfig config_;
  ApriltagInternalDetectionOptions options_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_INTERNAL_MEASUREMENT_REGENERATOR_HPP
