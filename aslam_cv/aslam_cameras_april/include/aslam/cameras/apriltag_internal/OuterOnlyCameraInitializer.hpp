#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_ONLY_CAMERA_INITIALIZER_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_ONLY_CAMERA_INITIALIZER_HPP

#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct AutoCameraInitializationCandidate {
  int rank = -1;
  std::string source_label = "grid";
  std::string evaluation_scope = "sampled";
  OuterBootstrapCameraIntrinsics camera;
  int observation_count = 0;
  int pose_success_count = 0;
  int pose_failure_count = 0;
  int successful_frame_count = 0;
  int successful_board_count = 0;
  double success_rate = 0.0;
  double mean_observation_rmse = std::numeric_limits<double>::infinity();
  bool valid = false;
  std::string failure_reason;
};

struct AutoCameraInitializationResidual {
  std::string source_label;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  double quality = 0.0;
  bool pose_success = false;
  double pose_fit_outer_rmse = std::numeric_limits<double>::quiet_NaN();
  std::string failure_reason;
};

struct AutoCameraInitializationResult {
  bool success = false;
  CameraInitializationMode requested_mode =
      CameraInitializationMode::AutoWithManualFallback;
  CameraInitializationMode selected_mode = CameraInitializationMode::Manual;
  bool auto_attempted = false;
  bool fallback_used = false;
  bool used_manual_intermediate_camera = false;
  bool used_manual_generic_seed = false;
  bool selected_candidate_refined = false;
  std::string selected_source_label;
  OuterBootstrapCameraIntrinsics selected_camera;
  cv::Size image_size;
  int candidate_count = 0;
  int sampled_observation_count = 0;
  int total_valid_outer_observation_count = 0;
  int accepted_pose_fit_observation_count = 0;
  int failed_pose_fit_observation_count = 0;
  int accepted_frame_count = 0;
  int accepted_board_observation_count = 0;
  double initialization_rmse = std::numeric_limits<double>::infinity();
  std::vector<AutoCameraInitializationCandidate> candidates;
  std::vector<AutoCameraInitializationResidual> selected_residuals;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct AutoCameraInitializationOptions {
  CameraInitializationMode mode = CameraInitializationMode::AutoWithManualFallback;
  int max_candidate_observations = 80;
  int top_candidate_count = 10;
  bool refine_best_candidate = true;
  std::vector<double> focal_scale_candidates{
      0.18, 0.22, 0.26, 0.30, 0.34, 0.40, 0.50, 0.60};
  std::vector<double> xi_candidates{-0.4, -0.2, 0.0, 0.2, 0.5, 1.0};
  std::vector<double> alpha_candidates{0.35, 0.45, 0.55, 0.65, 0.75};
};

class OuterOnlyCameraInitializer {
 public:
  explicit OuterOnlyCameraInitializer(
      ApriltagInternalConfig config,
      AutoCameraInitializationOptions options = AutoCameraInitializationOptions{});

  AutoCameraInitializationResult Initialize(
      const std::vector<OuterBootstrapFrameInput>& frames) const;

 private:
  ApriltagInternalConfig config_;
  AutoCameraInitializationOptions options_;
};

void WriteAutoCameraInitializationSummary(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationCandidatesCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);
void WriteAutoCameraInitializationOuterResidualsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_ONLY_CAMERA_INITIALIZER_HPP
