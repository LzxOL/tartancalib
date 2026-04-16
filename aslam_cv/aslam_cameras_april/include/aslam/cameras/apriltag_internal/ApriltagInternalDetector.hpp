#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_APRILTAG_INTERNAL_DETECTOR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_APRILTAG_INTERNAL_DETECTOR_HPP

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagCanonicalModel.hpp>
#include <aslam/cameras/apriltag_internal/CornerMeasurement.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct ApriltagInternalDetectionOptions {
  bool do_subpix_refinement = true;
  double max_subpix_displacement2 = 0.0;
  bool reject_duplicate_ids = true;
  double min_border_distance = 4.0;
  int canonical_pixels_per_module = 24;
  int refinement_window_radius = 0;
  double internal_subpix_window_scale = 0.5;
  int internal_subpix_window_min = 4;
  int internal_subpix_window_max = 16;
  double min_quality = 0.35;
  double min_template_contrast = 24.0;
  double virtual_patch_margin = 1.15;
  double internal_subpix_displacement_scale = 0.25;
  double max_internal_subpix_displacement = 6.0;
  MultiScaleOuterTagDetectorConfig outer_detector_config;
};

struct InternalCornerDebugInfo {
  int point_id = -1;
  CornerType corner_type = CornerType::LCorner;
  cv::Point2f predicted_image{};
  cv::Point2f refined_image{};
  cv::Point2f predicted_patch{};
  cv::Point2f refined_patch{};
  double local_module_scale = 0.0;
  int subpix_window_radius = 0;
  double subpix_displacement_limit = 0.0;
  int image_evidence_search_radius = 0;
  double q_refine = 0.0;
  double template_quality = 0.0;
  double gradient_quality = 0.0;
  double final_quality = 0.0;
  double image_template_quality = 0.0;
  double image_gradient_quality = 0.0;
  double image_centering_quality = 0.0;
  double image_final_quality = 0.0;
  double predicted_to_refined_displacement = 0.0;
  bool valid = false;
  bool image_evidence_valid = false;
};

struct ApriltagInternalDetectionResult {
  bool success = false;
  bool tag_detected = false;
  int board_id = -1;
  cv::Size image_size;
  InternalProjectionMode projection_mode = InternalProjectionMode::Homography;
  cv::Point2f tag_center;
  float observed_perimeter = 0.0f;
  std::array<cv::Point2f, 4> outer_corners{};
  std::array<bool, 4> outer_corner_valid{{false, false, false, false}};
  std::array<cv::Point2f, 4> patch_outer_corners{};
  std::vector<CornerMeasurement> corners;
  std::vector<InternalCornerDebugInfo> internal_corner_debug;
  int expected_visible_point_count = 0;
  int valid_corner_count = 0;
  int valid_internal_corner_count = 0;
  cv::Mat canonical_patch;
  OuterTagDetectionResult outer_detection;
};

class ApriltagInternalDetector {
 public:
  explicit ApriltagInternalDetector(
      ApriltagInternalConfig config,
      ApriltagInternalDetectionOptions options = ApriltagInternalDetectionOptions{});
  ~ApriltagInternalDetector();

  static ApriltagInternalConfig LoadConfig(const std::string& yaml_path);

  ApriltagInternalDetectionResult Detect(const cv::Mat& image) const;
  void DrawDetections(const ApriltagInternalDetectionResult& detections,
                      cv::Mat* output_image) const;
  void DrawCanonicalView(const ApriltagInternalDetectionResult& detections,
                         cv::Mat* output_image) const;

  const ApriltagCanonicalModel& model() const { return model_; }
  const ApriltagInternalConfig& config() const { return config_; }
  const ApriltagInternalDetectionOptions& options() const { return options_; }

 private:
  cv::Mat ToGray(const cv::Mat& image) const;

  ApriltagInternalConfig config_;
  ApriltagInternalDetectionOptions options_;
  ApriltagCanonicalModel model_;
  std::unique_ptr<MultiScaleOuterTagDetector> outer_detector_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_APRILTAG_INTERNAL_DETECTOR_HPP
