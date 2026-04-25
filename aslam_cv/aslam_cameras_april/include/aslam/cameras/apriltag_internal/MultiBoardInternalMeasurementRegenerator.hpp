#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_INTERNAL_MEASUREMENT_REGENERATOR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_INTERNAL_MEASUREMENT_REGENERATOR_HPP

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct JointReprojectionSceneState;
struct JointSceneFrameState;
struct JointSceneBoardState;

struct InternalRegenerationFrameInput {
  int frame_index = -1;
  std::string frame_label;
  OuterTagMultiDetectionResult outer_detections;
};

struct RegeneratedBoardMeasurement {
  int board_id = -1;
  bool frame_bootstrap_initialized = false;
  bool board_bootstrap_initialized = false;
  bool pose_prior_used = false;
  ApriltagInternalDetectionResult detection;
};

struct InternalRegenerationRuntimeBreakdown {
  double pose_estimation_seconds = 0.0;
  double boundary_model_seconds = 0.0;
  double seed_search_seconds = 0.0;
  double ray_refine_seconds = 0.0;
  double image_evidence_seconds = 0.0;
  double subpix_seconds = 0.0;
  int pose_estimation_call_count = 0;
  int boundary_model_build_count = 0;
  int attempted_internal_corner_count = 0;
  int valid_internal_corner_count = 0;
};

struct InternalRegenerationFrameResult {
  int frame_index = -1;
  std::string frame_label;
  bool frame_bootstrap_initialized = false;
  std::string state_source_label = "bootstrap";
  cv::Size image_size;
  std::vector<int> visible_board_ids;
  std::vector<std::string> warnings;
  std::vector<RegeneratedBoardMeasurement> board_measurements;
  InternalRegenerationRuntimeBreakdown runtime_breakdown;

  int SuccessfulBoardCount() const;
  int ValidInternalCornerCount() const;

  ApriltagInternalMultiDetectionResult AsMultiDetectionResult() const;
};

class MultiBoardInternalMeasurementRegenerator {
 public:
  explicit MultiBoardInternalMeasurementRegenerator(
      ApriltagInternalConfig config,
      ApriltagInternalDetectionOptions options = ApriltagInternalDetectionOptions{});

  InternalRegenerationFrameResult RegenerateFrame(
      const cv::Mat& image,
      const InternalRegenerationFrameInput& frame_input,
      const OuterBootstrapResult& bootstrap_result) const;
  InternalRegenerationFrameResult RegenerateFrame(
      const cv::Mat& image,
      const InternalRegenerationFrameInput& frame_input,
      const JointReprojectionSceneState& scene_state) const;

  void DrawFrameOverlay(const cv::Mat& image,
                        const InternalRegenerationFrameResult& frame_result,
                        cv::Mat* output_image) const;

  const ApriltagInternalDetector& detector() const { return detector_; }

 private:
  const OuterBootstrapFrameState* FindFrameState(
      const OuterBootstrapResult& bootstrap_result,
      const InternalRegenerationFrameInput& frame_input) const;
  const OuterBootstrapBoardState* FindBoardState(
      const OuterBootstrapResult& bootstrap_result,
      int board_id) const;
  const JointSceneFrameState* FindFrameState(
      const JointReprojectionSceneState& scene_state,
      const InternalRegenerationFrameInput& frame_input) const;
  const JointSceneBoardState* FindBoardState(
      const JointReprojectionSceneState& scene_state,
      int board_id) const;
  IntermediateCameraConfig MakeBootstrapCameraConfig(
      const OuterBootstrapCameraIntrinsics& intrinsics) const;
  IntermediateCameraConfig MakeSceneCameraConfig(
      const OuterBootstrapCameraIntrinsics& intrinsics) const;

  ApriltagInternalConfig config_;
  ApriltagInternalDetectionOptions options_;
  ApriltagInternalDetector detector_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_INTERNAL_MEASUREMENT_REGENERATOR_HPP
