#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_CAMERA_AWARE_OUTER_RESCUE_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_CAMERA_AWARE_OUTER_RESCUE_HPP

#include <functional>

#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/OuterDetectionCache.hpp>
#include <aslam/cameras/apriltag_internal/Stage5RecoveryTypes.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

std::string MakeCameraAwareRescueSignature(
    const OuterBootstrapCameraIntrinsics& camera,
    const MultiScaleOuterTagDetectorConfig& config,
    int max_hamming,
    int reference_board_id);

// Runs the frozen camera-aware outer-board recovery stage.  The implementation
// owns patch detection, direct-layout validation, deterministic merging, and
// rescue bookkeeping; the baseline pipeline remains responsible only for
// stage ordering and passing the shared frame state through.
void RunCameraAwareOuterRescue(
    const std::vector<FrozenRound2BaselineFrameSource>& frame_sources,
    const ApriltagInternalConfig& config,
    const OuterBootstrapCameraIntrinsics& provisional_camera,
    int reference_board_id,
    int max_hamming,
    int requested_worker_count,
    std::vector<OuterBootstrapFrameInput>* bootstrap_frames,
    std::vector<InternalRegenerationFrameInput>* regeneration_inputs,
    CameraAwareOuterRescueSummary* summary,
    OuterDetectionCache* rescue_cache = nullptr,
    const std::function<void(std::size_t, std::size_t, const std::string&)>&
        progress_callback = {});

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif
