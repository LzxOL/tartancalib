#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_INTERNAL_POINT_RESULT_VALIDATION_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_INTERNAL_POINT_RESULT_VALIDATION_HPP

#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

// Final, mode-independent result validation. Generation strategies only
// populate measurements/debug evidence; this layer validates refined points,
// removes duplicate assignments, and recomputes counters.
void EnforceInternalTopologyAssignment(
    ApriltagInternalDetectionResult* result);
void SuppressWrongLatticeSlotAssignments(
    ApriltagInternalDetectionResult* result);
void SuppressDuplicateRefinedInternalCorners(
    ApriltagInternalDetectionResult* result);
void SuppressLocallyInconsistentRecoveredCorners(
    ApriltagInternalDetectionResult* result);
void SuppressZeroImageEvidenceRecoveredCorners(
    ApriltagInternalDetectionResult* result);
void RecomputeCornerCounts(ApriltagInternalDetectionResult* result);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif
