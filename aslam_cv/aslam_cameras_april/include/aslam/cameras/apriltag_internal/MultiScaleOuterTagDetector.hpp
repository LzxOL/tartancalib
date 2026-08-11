#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_SCALE_OUTER_TAG_DETECTOR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_SCALE_OUTER_TAG_DETECTOR_HPP

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace AprilTags {
class TagDetector;
struct TagDetection;
}  // namespace AprilTags

namespace aslam {
namespace cameras {
namespace apriltag_internal {

class DoubleSphereCameraModel;

enum class OuterTagFailureReason {
  None = 0,
  NoDetectionsAtAll,
  DetectionsExistButNoMatchingTagId,
  MatchingTagIdButRejectedByBorder,
  MatchingTagIdButRefinementFailed,
  MatchingTagIdButAllScalesUnstable,
};

std::string ToString(OuterTagFailureReason reason);

struct OuterRefineCameraConfig {
  std::string camera_model;
  std::string distortion_model;
  std::vector<double> intrinsics;
  std::vector<double> distortion_coeffs;
  std::vector<int> resolution;

  bool IsConfigured() const {
    return !camera_model.empty() && !intrinsics.empty() && resolution.size() == 2;
  }
};

struct MultiScaleOuterTagDetectorConfig {
  int tag_id = 1;
  std::vector<int> tag_ids;
  // Frontend inspection only. When enabled, each frame first discovers the
  // decoded IDs present in that frame and then refines only those tags. It is
  // intentionally not a substitute for the explicit board topology required
  // by calibration and bundle adjustment.
  bool auto_discover_tag_ids = false;
  double min_border_distance = 4.0;
  int max_scales_to_try = 0;
  // Keep the legacy all-scale sweep by default. The adaptive cascade scans a
  // few low-resolution full images, validates their corners on the original
  // image, then visits higher-resolution full images only when required.
  bool enable_adaptive_scale_cascade = false;
  std::vector<double> adaptive_coarse_scale_divisors{4.5, 3.5, 2.5};
  std::vector<double> adaptive_fallback_scale_divisors{2.0, 1.5, 1.0};
  int adaptive_coarse_max_hamming = 0;
  bool enable_outer_spherical_refinement = true;
  bool do_outer_subpix_refinement = true;
  double outer_local_context_scale = 0.05;
  double outer_corner_marker_ratio = 0.0;
  double outer_subpix_scale = 0.35;
  bool enable_close_edge_outer_subpix_boost = true;
  double close_edge_outer_subpix_area_ratio = 0.02;
  double close_edge_outer_subpix_min_polar_deg = 50.0;
  double close_edge_outer_subpix_full_polar_deg = 78.0;
  double close_edge_outer_subpix_border_ratio = 0.15;
  double close_edge_outer_subpix_multiplier = 1.4;
  double close_edge_outer_subpix_max_multiplier = 2.4;
  double outer_refine_gate_scale = 0.025;
  double outer_refine_gate_min = 6.0;
  double min_detection_quality = 0.0;
  bool blur_before_detect = false;
  int blur_kernel = 7;
  double blur_sigma = 1.6;
  // Legacy detector-local id-bracket rescue is unsafe for the multi-board
  // calibration rig because board ids do not encode spatial order.
  bool enable_anonymous_tag_like_geometry_rescue = false;
  double anonymous_tag_like_rescue_max_center_error_scale = 0.90;
  double anonymous_tag_like_rescue_min_area_ratio = 0.30;
  double anonymous_tag_like_rescue_max_area_ratio = 3.50;
  // Camera-aware re-detection is active only when refine_camera is populated.
  // The production baseline accepts only exact-ID, zero-Hamming decodes and
  // keeps the distortion-aware patch-to-image mapping as the committed corner.
  bool enable_camera_aware_sphere_patch_rescue = true;
  int camera_aware_sphere_patch_max_hamming = 0;
  bool camera_aware_sphere_patch_commit_mapped_corners = true;
  // A full-view patch atlas is substantially more expensive for a frame with
  // no initial detections. It is enabled by default because otherwise a frame
  // with visible, strongly distorted boards but no direct AprilTag decode has
  // no recovery path at all. A recovered tag is still required to be an exact
  // requested-ID decode before it is used. Set this false only for an explicit
  // runtime/cost ablation.
  bool camera_aware_sphere_patch_rescue_zero_detection_frames = true;
  bool camera_aware_sphere_patch_use_extended_atlas = false;
  OuterRefineCameraConfig refine_camera;

  // Legacy compatibility fields. Old YAML keys may still populate these,
  // but the paper-facing C-S pipeline uses the high-level parameters above.
  std::vector<int> scale_candidates{3000, 2400, 1800, 1200, 1000, 800, 600, 500, 400, 300};
  std::vector<double> scale_divisors;
  int outer_subpix_window_radius = 0;
  double outer_subpix_window_scale = 0.015;
  int outer_subpix_window_min = 4;
  int outer_subpix_window_max = 48;
  double max_outer_refine_displacement = 6.0;
  double outer_refine_displacement_scale = 0.025;
  bool enable_outer_corner_layout_check = false;
  double outer_corner_verification_roi_scale = 0.035;
  int outer_corner_verification_roi_min = 12;
  int outer_corner_verification_roi_max = 48;
  double outer_corner_candidate_scale = 0.022;
  int outer_corner_candidate_min = 6;
  int outer_corner_candidate_max = 24;
  double outer_corner_branch_search_scale = 0.010;
  int outer_corner_branch_search_min = 3;
  int outer_corner_branch_search_max = 12;
  double outer_corner_min_direction_score = 0.35;
  double outer_corner_min_layout_score = 0.20;
};

struct OuterCornerScaleObservationDebugInfo {
  int target_longest_side = 0;
  double scale_factor = 1.0;
  double configured_scale_divisor = 0.0;
  cv::Point2f coarse_corner{};
  double deviation_from_consensus = 0.0;
  double deviation_from_fused = 0.0;
  bool rejected_as_outlier = false;
};

struct OuterCornerFusionDebugInfo {
  int corner_index = -1;
  int successful_scale_count = 0;
  int inlier_count = 0;
  int outlier_count = 0;
  double outlier_threshold = 0.0;
  double average_deviation_before = 0.0;
  double max_deviation_before = 0.0;
  double average_deviation_after = 0.0;
  double max_deviation_after = 0.0;
  bool used_outlier_rejection = false;
  bool stable_after_fusion = false;
  cv::Point2f consensus_corner{};
  cv::Point2f fused_corner{};
  std::vector<OuterCornerScaleObservationDebugInfo> scale_observations;
};

struct OuterCornerVerificationDebugInfo {
  int corner_index = -1;
  cv::Point2f coarse_corner{};
  cv::Point2f verified_corner{};
  cv::Point2f subpix_corner{};
  cv::Rect verification_roi;
  cv::Point2f prev_edge_direction{};
  cv::Point2f next_edge_direction{};
  std::vector<cv::Point2f> prev_marker_support_points;
  std::vector<cv::Point2f> next_marker_support_points;
  std::vector<cv::Point2f> prev_branch_points;
  std::vector<cv::Point2f> next_branch_points;
  double local_scale = 0.0;
  int verification_roi_radius = 0;
  int candidate_radius = 0;
  int branch_search_radius = 0;
  double direction_consistency_score = 0.0;
  double local_layout_score = 0.0;
  double verification_quality = 0.0;
  double coarse_to_verified_displacement = 0.0;
  double coarse_to_subpix_displacement = 0.0;
  double coarse_to_refined_displacement = 0.0;
  double corner_marker_width = 0.0;
  cv::Point2f image_line_corner{};
  double prev_image_line_residual = 0.0;
  double next_image_line_residual = 0.0;
  int prev_image_line_support_count = 0;
  int next_image_line_support_count = 0;
  bool image_line_valid = false;
  bool line_refinement_success = false;
  double line_refinement_quality = 0.0;
  double line_jump = 0.0;
  double line_jump_limit = 0.0;
  bool line_inside = false;
  double line_seed_gap = 0.0;
  bool line_seed_accepted = false;
  cv::Point2f spherical_corner{};
  std::vector<cv::Point2f> prev_spherical_curve_points;
  std::vector<cv::Point2f> next_spherical_curve_points;
  double prev_spherical_residual = 0.0;
  double next_spherical_residual = 0.0;
  int prev_spherical_support_count = 0;
  int next_spherical_support_count = 0;
  bool spherical_refinement_valid = false;
  bool spherical_refinement_applied = false;
  std::string spherical_failure_reason;
  bool close_edge_subpix_boost_applied = false;
  double close_edge_subpix_area_ratio = 0.0;
  double close_edge_subpix_max_polar_deg = 0.0;
  double close_edge_subpix_multiplier = 1.0;
  double configured_outer_subpix_scale = 0.0;
  double configured_outer_subpix_window_scale = 0.0;
  int configured_outer_subpix_window_radius = 0;
  int configured_outer_subpix_window_min = 0;
  int configured_outer_subpix_window_max = 0;
  int raw_subpix_window_radius = 0;
  int pre_boost_subpix_window_radius = 0;
  int boosted_raw_subpix_window_radius = 0;
  int subpix_window_clamp_limit = 0;
  bool subpix_window_clamped = false;
  int subpix_window_radius = 0;
  bool subpix_unstable_rollback_detected = false;
  int subpix_unstable_rollback_iteration = 0;
  double subpix_unstable_rollback_max_displacement = 0.0;
  double refine_displacement_limit = 0.0;
  bool refined_valid = false;
  bool verification_passed = false;
  bool subpix_applied = false;
  std::string failure_reason;
};

struct OuterWrongIdProposal {
  int detected_tag_id = -1;
  int hamming = -1;
  double area_px = 0.0;
  std::array<Eigen::Vector2d, 4> corners_original_image{};
  std::string source;
};

struct OuterSphericalCornerRefinementDebug {
  bool success = false;
  cv::Point2f refined_corner{};
  double quality = 0.0;
  double displacement_px = 0.0;
  double prev_edge_residual = 0.0;
  double next_edge_residual = 0.0;
  int prev_edge_support_count = 0;
  int next_edge_support_count = 0;
  std::string failure_reason;
};

struct OuterSphericalQuadRefinementResult {
  bool success = false;
  std::array<cv::Point2f, 4> refined_corners{};
  std::array<OuterSphericalCornerRefinementDebug, 4> corner_debug{};
  double max_displacement_px = 0.0;
  double min_quality = 0.0;
  int successful_corner_count = 0;
};

OuterSphericalQuadRefinementResult RefineOuterCornersBySphericalPlanes(
    const cv::Mat& gray,
    const DoubleSphereCameraModel& camera,
    const std::array<cv::Point2f, 4>& corner_seeds,
    const MultiScaleOuterTagDetectorConfig& config);

struct OuterTagScaleDebugInfo {
  int target_longest_side = 0;
  double scale_factor = 1.0;
  double configured_scale_divisor = 0.0;
  cv::Size scaled_size;
  bool attempted = false;
  int raw_detection_count = 0;
  int raw_good_detection_count = 0;
  int matching_tag_count = 0;
  int matching_good_tag_count = 0;
  int accepted_candidate_count = 0;
  int refined_success_count = 0;
  bool contributed_to_corner_fusion = false;
  std::vector<std::string> raw_detection_summaries;
  std::string rejection_summary;
};

struct OuterTagDetectionResult {
  bool success = false;
  int board_id = -1;
  int detected_tag_id = -1;
  int original_longest_side = 0;
  int chosen_scale_longest_side = 0;
  double chosen_scale_factor = 1.0;
  std::string scale_configuration_mode;
  int adaptive_coarse_scale_attempt_count = 0;
  int adaptive_fallback_scale_attempt_count = 0;
  bool adaptive_high_resolution_fallback_triggered = false;
  bool used_corner_fusion = false;
  int hamming = -1;
  bool good = false;
  bool attempted_local_patch_rescue = false;
  bool used_local_patch_rescue = false;
  std::string local_patch_rescue_summary;
  std::vector<OuterWrongIdProposal> wrong_id_proposals;
  std::array<Eigen::Vector2d, 4> coarse_corners_scaled_image{};
  std::array<Eigen::Vector2d, 4> coarse_corners_original_image{};
  std::array<Eigen::Vector2d, 4> refined_corners_original_image{};
  std::array<bool, 4> refined_valid{{false, false, false, false}};
  double quality = 0.0;
  OuterTagFailureReason failure_reason = OuterTagFailureReason::NoDetectionsAtAll;
  std::string failure_reason_text;
  std::vector<int> successful_scale_longest_sides;
  std::vector<OuterTagScaleDebugInfo> scale_debug;
  std::array<OuterCornerFusionDebugInfo, 4> corner_fusion_debug{};
  std::array<OuterCornerVerificationDebugInfo, 4> corner_verification_debug{};
};

struct OuterBoardMeasurement {
  int board_id = -1;
  int detected_tag_id = -1;
  bool success = false;
  bool attempted_local_patch_rescue = false;
  bool used_local_patch_rescue = false;
  std::string local_patch_rescue_summary;
  double detection_quality = 0.0;
  int valid_refined_corner_count = 0;
  std::array<Eigen::Vector2d, 4> refined_outer_corners_original_image{};
  std::array<bool, 4> refined_corner_valid{{false, false, false, false}};
  bool has_target_outer_corners = false;
  std::array<Eigen::Vector3d, 4> target_outer_corners_board{};
  bool has_direct_dense_control_points = false;
  std::vector<Eigen::Vector3d> direct_dense_target_points_board;
  std::vector<Eigen::Vector2d> direct_dense_image_points;
  std::vector<unsigned char> direct_dense_point_is_outer;
  std::array<OuterCornerVerificationDebugInfo, 4> corner_verification_debug{};
  OuterTagFailureReason failure_reason = OuterTagFailureReason::NoDetectionsAtAll;
  std::string failure_reason_text;
};

struct OuterFrameMeasurementResult {
  cv::Size image_size;
  std::vector<int> requested_board_ids;
  std::vector<OuterBoardMeasurement> board_measurements;

  bool AnySuccess() const {
    for (const OuterBoardMeasurement& measurement : board_measurements) {
      if (measurement.success) {
        return true;
      }
    }
    return false;
  }

  int SuccessfulBoardCount() const {
    int count = 0;
    for (const OuterBoardMeasurement& measurement : board_measurements) {
      count += measurement.success ? 1 : 0;
    }
    return count;
  }
};

struct OuterTagMultiDetectionResult {
  cv::Size image_size;
  std::vector<int> requested_board_ids;
  std::vector<OuterTagDetectionResult> detections;
  OuterFrameMeasurementResult frame_measurements;

  bool AnySuccess() const {
    return frame_measurements.AnySuccess();
  }

  int SuccessfulBoardCount() const {
    return frame_measurements.SuccessfulBoardCount();
  }
};

class MultiScaleOuterTagDetector {
 public:
  explicit MultiScaleOuterTagDetector(
      MultiScaleOuterTagDetectorConfig config = MultiScaleOuterTagDetectorConfig{});
  ~MultiScaleOuterTagDetector();

  static MultiScaleOuterTagDetectorConfig LoadConfig(const std::string& yaml_path);

  OuterTagDetectionResult Detect(const cv::Mat& image) const;
  OuterTagMultiDetectionResult DetectMultiple(const cv::Mat& image) const;
  std::vector<int> DiscoverTagIds(const cv::Mat& image) const;
  std::vector<OuterTagDetectionResult> DetectMultiple(
      const cv::Mat& image, const std::vector<int>& requested_tag_ids) const;
  void DrawDetection(const OuterTagDetectionResult& detection,
                     cv::Mat* output_image,
                     bool draw_debug) const;
  void DrawDetections(const OuterTagMultiDetectionResult& detections,
                      cv::Mat* output_image,
                      bool draw_debug) const;

  const MultiScaleOuterTagDetectorConfig& config() const { return config_; }
  const std::vector<int>& requested_board_ids() const { return requested_board_ids_; }

 private:
  cv::Mat ToGray(const cv::Mat& image) const;
  void DrawDetectionImpl(const OuterTagDetectionResult& detection,
                         cv::Mat* output_image,
                         bool draw_debug,
                         bool include_status_text) const;

  MultiScaleOuterTagDetectorConfig config_;
  std::vector<int> requested_board_ids_;
  std::unique_ptr<AprilTags::TagDetector> detector_;
  std::unique_ptr<DoubleSphereCameraModel> sphere_camera_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_SCALE_OUTER_TAG_DETECTOR_HPP
