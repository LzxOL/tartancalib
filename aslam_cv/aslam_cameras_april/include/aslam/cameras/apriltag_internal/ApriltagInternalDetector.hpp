#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_APRILTAG_INTERNAL_DETECTOR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_APRILTAG_INTERNAL_DETECTOR_HPP

#include <array>
#include <map>
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
  cv::Point2f border_seed_image{};
  cv::Point2f sphere_seed_image{};
  cv::Point2f refined_image{};
  cv::Point2f predicted_patch{};
  cv::Point2f sphere_seed_patch{};
  cv::Point2f refined_patch{};
  cv::Vec3d predicted_ray{0.0, 0.0, 0.0};
  cv::Vec3d border_seed_ray{0.0, 0.0, 0.0};
  cv::Vec3d sphere_seed_ray{0.0, 0.0, 0.0};
  cv::Vec3d refined_ray{0.0, 0.0, 0.0};
  cv::Vec3d border_top_ray{0.0, 0.0, 0.0};
  cv::Vec3d border_bottom_ray{0.0, 0.0, 0.0};
  cv::Vec3d border_left_ray{0.0, 0.0, 0.0};
  cv::Vec3d border_right_ray{0.0, 0.0, 0.0};
  cv::Vec3d tangent_u_ray{0.0, 0.0, 0.0};
  cv::Vec3d tangent_v_ray{0.0, 0.0, 0.0};
  cv::Point2f module_u_axis{};
  cv::Point2f module_v_axis{};
  double local_module_scale = 0.0;
  double sphere_search_radius = 0.0;
  double adaptive_search_radius = 0.0;
  double sphere_template_quality = 0.0;
  double sphere_gradient_quality = 0.0;
  double sphere_prior_quality = 0.0;
  double sphere_peak_quality = 0.0;
  double sphere_raw_quality = 0.0;
  double sphere_seed_quality = 0.0;
  double ray_refine_edge_quality = 0.0;
  double ray_refine_photometric_quality = 0.0;
  double ray_refine_final_quality = 0.0;
  double ray_refine_trust_radius = 0.0;
  int ray_refine_iterations = 0;
  bool ray_refine_converged = false;
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
  double predicted_to_border_seed_displacement = 0.0;
  double predicted_to_seed_displacement = 0.0;
  double border_seed_to_sphere_seed_displacement = 0.0;
  double seed_to_refined_displacement = 0.0;
  double seed_to_refined_angular = 0.0;
  double predicted_to_refined_displacement = 0.0;
  bool border_seed_valid = false;
  bool border_seed_fallback_to_sphere_lattice = false;
  bool valid = false;
  bool image_evidence_valid = false;
};

struct ApriltagInternalRuntimeBreakdown {
  double total_seconds = 0.0;
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

struct ApriltagInternalDetectionResult {
  bool success = false;
  bool tag_detected = false;
  int board_id = -1;
  cv::Size image_size;
  std::string failure_reason;
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
  bool border_boundary_model_valid = false;
  std::array<bool, 4> border_edge_valid{{false, false, false, false}};
  std::array<double, 4> border_edge_rms_residual{{0.0, 0.0, 0.0, 0.0}};
  std::array<int, 4> border_edge_support_count{{0, 0, 0, 0}};
  std::array<std::vector<cv::Point2f>, 4> border_support_points{};
  std::array<std::vector<cv::Point2f>, 4> border_curves_image{};
  std::array<std::vector<cv::Vec3d>, 4> border_curves_ray{};
  cv::Mat canonical_patch;
  OuterTagDetectionResult outer_detection;
  ApriltagInternalRuntimeBreakdown runtime_breakdown;
};

struct ApriltagInternalMultiDetectionResult {
  cv::Size image_size;
  std::vector<int> requested_board_ids;
  std::vector<ApriltagInternalDetectionResult> detections;

  bool AnyTagDetected() const {
    for (const ApriltagInternalDetectionResult& detection : detections) {
      if (detection.tag_detected) {
        return true;
      }
    }
    return false;
  }

  bool AnySuccess() const {
    for (const ApriltagInternalDetectionResult& detection : detections) {
      if (detection.success) {
        return true;
      }
    }
    return false;
  }
};

class ApriltagInternalDetector {
 public:
  explicit ApriltagInternalDetector(
      ApriltagInternalConfig config,
      ApriltagInternalDetectionOptions options = ApriltagInternalDetectionOptions{});
  ~ApriltagInternalDetector();

  static ApriltagInternalConfig LoadConfig(const std::string& yaml_path);

  ApriltagInternalDetectionResult Detect(const cv::Mat& image) const;
  ApriltagInternalMultiDetectionResult DetectMultiple(const cv::Mat& image) const;
  ApriltagInternalDetectionResult DetectFromOuterDetection(
      const cv::Mat& image,
      int board_id,
      const OuterTagDetectionResult& outer_detection,
      const IntermediateCameraConfig* camera_override = nullptr,
      const Eigen::Matrix4d* T_camera_board_prior = nullptr) const;
  ApriltagInternalMultiDetectionResult DetectMultipleFromOuterDetections(
      const cv::Mat& image,
      const OuterTagMultiDetectionResult& outer_multi_detection,
      const IntermediateCameraConfig* camera_override = nullptr,
      const std::map<int, Eigen::Matrix4d>& T_camera_board_priors =
          std::map<int, Eigen::Matrix4d>()) const;
  void DrawDetections(const ApriltagInternalDetectionResult& detections,
                      cv::Mat* output_image) const;
  void DrawDetections(const ApriltagInternalMultiDetectionResult& detections,
                      cv::Mat* output_image) const;
  void DrawCanonicalView(const ApriltagInternalDetectionResult& detections,
                         cv::Mat* output_image) const;

  const ApriltagCanonicalModel& model() const {
    return board_runtimes_[default_board_index_].model;
  }
  const ApriltagInternalConfig& config() const { return config_; }
  const ApriltagInternalDetectionOptions& options() const { return options_; }
  const std::vector<int>& requested_board_ids() const { return requested_board_ids_; }

 private:
  struct BoardRuntime {
    explicit BoardRuntime(ApriltagInternalConfig board_config)
        : config(std::move(board_config)), model(config) {}

    ApriltagInternalConfig config;
    ApriltagCanonicalModel model;
  };

  cv::Mat ToGray(const cv::Mat& image) const;
  const BoardRuntime& RuntimeForBoardIdOrDefault(int board_id) const;
  ApriltagInternalDetectionResult DetectSingleBoardFromOuter(
      const cv::Mat& gray,
      const BoardRuntime& board_runtime,
      const OuterTagDetectionResult& outer_detection,
      const IntermediateCameraConfig* camera_override,
      const Eigen::Matrix4d* T_camera_board_prior) const;
  void DrawDetectionsImpl(const ApriltagInternalDetectionResult& detections,
                          const ApriltagCanonicalModel& model,
                          cv::Mat* output_image,
                          bool include_status_text) const;
  void DrawCanonicalViewImpl(const ApriltagInternalDetectionResult& detections,
                             const ApriltagCanonicalModel& model,
                             cv::Mat* output_image) const;

  ApriltagInternalConfig config_;
  ApriltagInternalDetectionOptions options_;
  std::vector<int> requested_board_ids_;
  std::vector<BoardRuntime> board_runtimes_;
  std::size_t default_board_index_ = 0;
  std::unique_ptr<MultiScaleOuterTagDetector> outer_detector_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_APRILTAG_INTERNAL_DETECTOR_HPP
