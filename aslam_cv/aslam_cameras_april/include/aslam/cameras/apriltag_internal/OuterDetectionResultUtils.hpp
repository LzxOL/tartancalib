#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_DETECTION_RESULT_UTILS_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_DETECTION_RESULT_UTILS_HPP

#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

// Keep the representation of an absent requested board consistent across the
// outer detector, internal detector, and multi-board regeneration stages.
inline OuterTagDetectionResult MakeMissingOuterTagDetection(
    int board_id,
    OuterTagFailureReason reason = OuterTagFailureReason::NoDetectionsAtAll) {
  OuterTagDetectionResult detection;
  detection.board_id = board_id;
  detection.failure_reason = reason;
  detection.failure_reason_text = ToString(reason);
  return detection;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_DETECTION_RESULT_UTILS_HPP
