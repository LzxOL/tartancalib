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

enum class InternalPoseRescueMode {
  Off,
  Diagnostic,
  Enabled,
};

const char* ToString(InternalPoseRescueMode mode);
InternalPoseRescueMode ParseInternalPoseRescueMode(const std::string& value);

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
  bool ignore_image_evidence_min_quality = false;
  bool force_internal_seed_from_prediction = false;
  bool enable_internal_structure_correction_after_ss = true;
  // Experimental mode: skip SearchSphereLatticeSeed /
  // RefineSphereSeedRayLocally acceptance gates and start directly from the
  // predicted lattice ray. This is intentionally explicit because it bypasses
  // the normal image-evidence seed validation path.
  bool bypass_internal_seed_filters = false;
  InternalPoseRescueMode internal_pose_rescue_mode =
      InternalPoseRescueMode::Enabled;
  double internal_pose_rescue_max_ray_angle_deg = 88.0;
  double internal_pose_rescue_accept_max_outer_rmse = 8.0;
  bool enable_geometry_prior_outer_seed = false;
  bool geometry_prior_rescue_diagnostic_only = true;
  bool geometry_prior_rescue_use_as_observation = false;
  // Experimental and opt-in: if tag decoding/raw-quad context validation fails,
  // allow a geometry-prior seed to continue to pose refit only when strong
  // image edge evidence is present. Pure projected corners are still not
  // accepted without image evidence and pose consistency checks.
  bool geometry_prior_rescue_allow_geometry_only_pose_refit = false;
  // 0 means adapt from the predicted board pixel scale, positive forces a
  // fixed radius, and negative disables geometry-prior subpixel refinement.
  int geometry_prior_rescue_subpix_window_radius = 0;
  // Zero uses a scale-adaptive displacement upper bound, positive forces a
  // fixed pixel bound, and negative disables the bound for debug ablations.
  double geometry_prior_rescue_max_corner_displacement_px = 0.0;
  double geometry_prior_rescue_min_corner_response_ratio = 0.03;
  bool geometry_prior_rescue_enable_spherical_refine = false;
  int geometry_prior_rescue_edge_sample_count = 80;
  int geometry_prior_rescue_edge_search_half_width_px = 6;
  double geometry_prior_rescue_min_edge_support_ratio = 0.45;
  double geometry_prior_rescue_min_edge_gradient_ratio = 0.02;
  double geometry_prior_rescue_accept_max_outer_rmse = 8.0;
  double geometry_prior_rescue_accept_max_rotation_error_deg = 5.0;
  double geometry_prior_rescue_accept_max_translation_error = 0.08;
  // Opt-in large-tag recovery. A pose hypothesis from at least two visible
  // boards is refined against the image, then rendered back to a canonical
  // tag plane with the active camera model. The expected tag code must be a
  // clear photometric winner before the refined corners become an observation.
  bool geometry_guided_tag_likelihood_enabled = false;
  int geometry_guided_tag_likelihood_min_visible_boards = 2;
  int geometry_guided_tag_likelihood_max_expected_hamming = 6;
  int geometry_guided_tag_likelihood_min_hamming_margin = 3;
  double geometry_guided_tag_likelihood_min_contrast = 0.10;
  // A single anchor board cannot provide same-frame consensus. Permit it only
  // when the anchor is a direct exact-ID observation and require materially
  // stronger code evidence for every recovered board.
  bool geometry_guided_tag_likelihood_allow_single_anchor = false;
  double geometry_guided_tag_likelihood_single_anchor_max_outer_rmse = 0.50;
  int geometry_guided_tag_likelihood_single_anchor_max_expected_hamming = 2;
  int geometry_guided_tag_likelihood_single_anchor_min_hamming_margin = 6;
  double geometry_guided_tag_likelihood_single_anchor_min_contrast = 0.15;
  MultiScaleOuterTagDetectorConfig outer_detector_config;
};

struct InternalCornerDebugInfo {
  int point_id = -1;
  CornerType corner_type = CornerType::LCorner;
  cv::Point2f predicted_image{};
  cv::Point2f border_seed_image{};
  cv::Point2f sphere_seed_image{};
  cv::Point2f structure_corrected_image{};
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
  double sphere_seed_to_structure_corrected_displacement = 0.0;
  double seed_to_refined_displacement = 0.0;
  double seed_to_refined_angular = 0.0;
  double predicted_to_refined_displacement = 0.0;
  bool border_seed_valid = false;
  bool structure_correction_valid = false;
  bool structure_correction_group_by_column = false;
  double structure_correction_delta_px = 0.0;
  bool forced_prediction_seed = false;
  bool bypass_seed_filters = false;
  bool original_seed_filter_success = false;
  bool original_seed_filter_would_reject = false;
  cv::Point2f original_seed_filter_image{};
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
  int pose_rescue_attempt_count = 0;
  int pose_rescue_success_count = 0;
  int pose_rescue_used_count = 0;
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
  std::string internal_camera_source;
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
  bool pose_rescue_attempted = false;
  bool pose_rescue_success = false;
  bool pose_rescue_used = false;
  double pose_rescue_rmse = 0.0;
  double pose_rescue_max_ray_angle_deg = 0.0;
  double pose_rescue_ray_angle_limit_deg = 0.0;
  double pose_rescue_accept_max_outer_rmse = 0.0;
  std::string pose_rescue_failure_reason;
  bool border_boundary_model_valid = false;
  std::string border_boundary_model_failure_reason;
  std::array<bool, 4> border_edge_valid{{false, false, false, false}};
  std::array<double, 4> border_edge_rms_residual{{0.0, 0.0, 0.0, 0.0}};
  std::array<int, 4> border_edge_support_count{{0, 0, 0, 0}};
  std::array<int, 4> border_edge_support_ray_count{{0, 0, 0, 0}};
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
