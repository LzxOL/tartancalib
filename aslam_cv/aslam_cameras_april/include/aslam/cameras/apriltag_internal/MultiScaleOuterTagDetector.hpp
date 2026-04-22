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

enum class OuterTagFailureReason {
  None = 0,
  NoDetectionsAtAll,
  DetectionsExistButNoMatchingTagId,
  MatchingTagIdButRejectedByBorder,
  MatchingTagIdButRefinementFailed,
  MatchingTagIdButAllScalesUnstable,
};

std::string ToString(OuterTagFailureReason reason);

struct MultiScaleOuterTagDetectorConfig {
  int tag_id = 1;
  double min_border_distance = 4.0;
  int max_scales_to_try = 0;
  std::vector<int> scale_candidates{3000, 2400, 1800, 1200, 1000, 800, 600, 500, 400, 300};
  std::vector<double> scale_divisors;
  bool do_outer_subpix_refinement = true;
  int outer_subpix_window_radius = 0;
  double outer_subpix_window_scale = 0.015;
  int outer_subpix_window_min = 4;
  int outer_subpix_window_max = 16;
  double max_outer_refine_displacement = 6.0;
  double outer_refine_displacement_scale = 0.025;
  double min_detection_quality = 0.0;
  bool blur_before_detect = false;
  int blur_kernel = 7;
  double blur_sigma = 1.6;
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
  int subpix_window_radius = 0;
  double refine_displacement_limit = 0.0;
  bool refined_valid = false;
  bool verification_passed = false;
  bool subpix_applied = false;
  std::string failure_reason;
};

struct OuterTagScaleDebugInfo {
  int target_longest_side = 0;
  double scale_factor = 1.0;
  double configured_scale_divisor = 0.0;
  cv::Size scaled_size;
  bool attempted = false;
  int raw_detection_count = 0;
  int matching_tag_count = 0;
  int accepted_candidate_count = 0;
  int refined_success_count = 0;
  bool contributed_to_corner_fusion = false;
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
  bool used_corner_fusion = false;
  int hamming = -1;
  bool good = false;
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

class MultiScaleOuterTagDetector {
 public:
  explicit MultiScaleOuterTagDetector(
      MultiScaleOuterTagDetectorConfig config = MultiScaleOuterTagDetectorConfig{});
  ~MultiScaleOuterTagDetector();

  static MultiScaleOuterTagDetectorConfig LoadConfig(const std::string& yaml_path);

  OuterTagDetectionResult Detect(const cv::Mat& image) const;
  void DrawDetection(const OuterTagDetectionResult& detection,
                     cv::Mat* output_image,
                     bool draw_debug) const;

  const MultiScaleOuterTagDetectorConfig& config() const { return config_; }

 private:
  cv::Mat ToGray(const cv::Mat& image) const;

  MultiScaleOuterTagDetectorConfig config_;
  std::unique_ptr<AprilTags::TagDetector> detector_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_SCALE_OUTER_TAG_DETECTOR_HPP
