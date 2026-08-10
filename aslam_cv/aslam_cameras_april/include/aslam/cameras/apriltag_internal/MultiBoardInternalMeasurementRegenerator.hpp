#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_INTERNAL_MEASUREMENT_REGENERATOR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_INTERNAL_MEASUREMENT_REGENERATOR_HPP

#include <limits>
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

struct GeometryPriorOuterSeedCandidate {
  int frame_index = -1;
  std::string frame_label;
  int missing_board_id = -1;
  std::string prediction_source_label;
  int frame_pose_refit_source_board_id = -1;
  double frame_pose_refit_outer_rmse = 0.0;
  std::vector<int> visible_boards_used;
  std::array<cv::Point2f, 4> predicted_corners{};
  std::array<cv::Point2f, 4> refined_corners{};
  double predicted_area_px = 0.0;
  double predicted_signed_area_px = 0.0;
  double refined_area_px = 0.0;
  double refined_signed_area_px = 0.0;
  double refined_to_predicted_area_ratio =
      std::numeric_limits<double>::quiet_NaN();
  bool predicted_quad_topology_valid = false;
  bool refined_quad_topology_valid = false;
  bool quad_topology_preserved = false;
  std::string quad_topology_summary;
  double local_corner_scale_px = 0.0;
  int subpix_window_radius = 0;
  bool spherical_refine_attempted = false;
  bool spherical_refine_success = false;
  int spherical_refine_successful_corner_count = 0;
  double spherical_refine_max_displacement_px = 0.0;
  double spherical_refine_min_quality = 0.0;
  double spherical_refine_min_support_count = 0.0;
  double spherical_refine_max_residual = 0.0;
  std::string spherical_refine_failure_summary;
  double max_corner_displacement_px = 0.0;
  double adaptive_max_corner_displacement_px =
      std::numeric_limits<double>::quiet_NaN();
  double min_corner_response_ratio = 0.0;
  double edge_support_ratio = 0.0;
  double mean_edge_gradient_ratio = 0.0;
  bool rectified_patch_checked = false;
  bool rectified_patch_decode_success = false;
  int rectified_patch_detected_tag_id = -1;
  int rectified_patch_hamming = -1;
  std::string rectified_patch_summary;
  bool geometry_guided_tag_likelihood_checked = false;
  bool geometry_guided_tag_likelihood_passed = false;
  std::string geometry_guided_tag_likelihood_mode;
  int geometry_guided_tag_likelihood_expected_hamming = -1;
  int geometry_guided_tag_likelihood_runner_up_id = -1;
  int geometry_guided_tag_likelihood_runner_up_hamming = -1;
  int geometry_guided_tag_likelihood_hamming_margin = -1;
  double geometry_guided_tag_likelihood_contrast = 0.0;
  std::string geometry_guided_tag_likelihood_summary;
  bool roi_redetect_checked = false;
  bool roi_redetect_success = false;
  int roi_redetect_detected_tag_id = -1;
  int roi_redetect_hamming = -1;
  cv::Rect roi_redetect_bbox;
  std::string roi_redetect_summary;
  bool roi_valid = false;
  bool image_evidence_checked = false;
  bool tag_id_validated = false;
  bool image_evidence_success = false;
  bool local_redetect_success = false;
  bool local_corner_refine_success = false;
  bool pose_refit_success = false;
  double local_vs_global_rotation_error_deg = 0.0;
  double local_vs_global_translation_error = 0.0;
  double outer_reprojection_rmse = 0.0;
  double frame_normal_outer_refit_rmse_median =
      std::numeric_limits<double>::quiet_NaN();
  double adaptive_accept_max_outer_rmse =
      std::numeric_limits<double>::quiet_NaN();
  bool accepted_as_rescued_observation = false;
  std::string reject_reason;
};

struct InternalRegenerationRuntimeBreakdown {
  double pose_estimation_seconds = 0.0;
  double boundary_model_seconds = 0.0;
  double seed_search_seconds = 0.0;
  double ray_refine_seconds = 0.0;
  double image_evidence_seconds = 0.0;
  double subpix_seconds = 0.0;
  int pose_estimation_call_count = 0;
  int pose_rescue_attempt_count = 0;
  int pose_rescue_success_count = 0;
  int pose_rescue_used_count = 0;
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
  std::vector<GeometryPriorOuterSeedCandidate> geometry_prior_outer_seed_candidates;
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
