#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_ONLY_CAMERA_INITIALIZER_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_ONLY_CAMERA_INITIALIZER_HPP

#include <array>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

enum class AutoCameraInitializationRefineMode {
  None,
  CoordinateSearch,
  KalibrOuterLm,
};

const char* ToString(AutoCameraInitializationRefineMode mode);
AutoCameraInitializationRefineMode ParseAutoCameraInitializationRefineMode(
    const std::string& value);

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
  int leave_one_board_out_attempt_count = 0;
  int leave_one_board_out_success_count = 0;
  double leave_one_board_out_rmse = std::numeric_limits<double>::infinity();
  int relative_layout_pair_family_count = 0;
  int relative_layout_pair_sample_count = 0;
  double relative_layout_translation_rmse = std::numeric_limits<double>::infinity();
  double relative_layout_rotation_rmse_deg = std::numeric_limits<double>::infinity();
  double relative_layout_consistency_score = std::numeric_limits<double>::infinity();
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

struct AutoCameraInitializationBootstrapObservation {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  std::array<Eigen::Vector2d, 4> outer_corners{};
  bool used_in_lm = false;
  bool pose_init_success = false;
  double pose_fit_outer_rmse = std::numeric_limits<double>::quiet_NaN();
};

struct AutoCameraInitializationResult {
  bool success = false;
  CameraInitializationMode requested_mode =
      CameraInitializationMode::AutoWithManualFallback;
  CameraInitializationMode selected_mode = CameraInitializationMode::Manual;
  bool auto_attempted = false;
  bool fallback_used = false;
  bool used_manual_intermediate_camera = false;
  bool used_explicit_initial_camera = false;
  bool used_manual_generic_seed = false;
  bool selected_candidate_refined = false;
  AutoCameraInitializationRefineMode refine_mode =
      AutoCameraInitializationRefineMode::KalibrOuterLm;
  int lm_frame_count = 0;
  int lm_view_count = 0;
  int lm_residual_count = 0;
  int lm_invalid_projection_count = 0;
  int lm_nonfinite_count = 0;
  int lm_iteration_count = 0;
  double lm_initial_rmse = std::numeric_limits<double>::infinity();
  double lm_final_rmse = std::numeric_limits<double>::infinity();
  std::string stage5_init_seed_method;
  std::string stage5_init_seed_source;
  double stage5_init_omni_gamma = std::numeric_limits<double>::quiet_NaN();
  std::string stage5_init_omni_gamma_source;
  std::string stage5_init_ds_mapping;
  int stage5_init_ds_mapping_verified_against_kalibr_source = 0;
  int stage5_init_ds_grid_enumeration_enabled = 0;
  std::string stage5_init_kb_focal_source;
  int stage5_init_kb_row_circle_focal_available = 0;
  int stage5_init_kb_zero_distortion_seed = 0;
  int stage5_init_kb_zero_distortion_seed_included = 0;
  int stage5_init_kb_nonzero_distortion_seed_count = 0;
  int stage5_init_kb_distortion_released_in_lm = 0;
  int stage5_init_kb_distortion_fixed_zero_in_init_lm = 0;
  int stage5_init_kb_multistart_enabled = 0;
  std::string stage5_init_ucm_seed_source;
  int stage5_init_ucm_omni_gamma_available = 0;
  std::string stage5_init_ucm_mapping;
  int stage5_init_ucm_mapping_verified_against_kalibr_source = 0;
  int stage5_init_ucm_multistart_enabled = 0;
  int stage5_init_ucm_shape_released_in_lm = 0;
  int stage5_init_uses_yaml_intrinsics = 0;
  int stage5_init_uses_kalibr_camchain_intrinsics = 0;
  int stage5_init_outer_only = 1;
  int stage5_init_uses_layout_to_update_intrinsics = 0;
  int stage5_init_layout_loo_diagnostics_only = 1;
  int stage5_init_multiboard_frame_objective_enabled = 0;
  int stage5_init_fixed_layout_frame_constraint_used = 0;
  int stage5_init_optimizes_layout_variables = 0;
  std::string stage5_init_lm_selection_objective;
  std::string stage5_init_selection_prefilter;
  std::string stage5_init_selection_scorer;
  int stage5_init_selection_uses_information_metric = 0;
  int stage5_init_selection_is_exact_kalibr_information_theoretic = 0;
  int stage5_init_calibrate_intrinsics_enabled = 0;
  std::string stage5_init_calibrate_intrinsics_released_params;
  std::string stage5_init_calibrate_intrinsics_optimizer;
  double stage5_init_runtime_seconds = 0.0;
  int stage5_init_selected_pose_success_count = 0;
  int stage5_init_selected_pose_total_count = 0;
  double full_outer_pose_success_rate = 0.0;
  double full_outer_rmse = std::numeric_limits<double>::infinity();
  double full_outer_median_error = std::numeric_limits<double>::infinity();
  double full_outer_p95_error = std::numeric_limits<double>::infinity();
  double full_outer_robust_inlier_rmse = std::numeric_limits<double>::infinity();
  double full_outer_robust_outlier_threshold =
      std::numeric_limits<double>::infinity();
  int full_outer_robust_outlier_count = 0;
  int full_outer_projection_failure_count = 0;
  int full_outer_nonfinite_count = 0;
  int bootstrap_internal_points_used = 0;
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
  std::vector<AutoCameraInitializationBootstrapObservation>
      lm_bootstrap_observations;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct AutoCameraInitializationOptions {
  CameraInitializationMode mode = CameraInitializationMode::AutoWithManualFallback;
  AutoCameraInitializationRefineMode refine_mode =
      AutoCameraInitializationRefineMode::KalibrOuterLm;
  bool use_explicit_initial_camera = false;
  OuterBootstrapCameraIntrinsics explicit_initial_camera;
  std::string explicit_initial_camera_source_label = "explicit_initial_camera";
  int max_candidate_observations = 80;
  int top_candidate_count = 10;
  bool refine_best_candidate = true;
  std::vector<double> focal_scale_candidates{
      0.18, 0.22, 0.26, 0.30, 0.34, 0.40, 0.50, 0.60};
  std::vector<double> xi_candidates{-0.4, -0.2, 0.0, 0.2, 0.5, 1.0};
  std::vector<double> alpha_candidates{0.35, 0.45, 0.55, 0.65, 0.75};
  std::vector<double> eucm_alpha_candidates{0.35, 0.45, 0.55, 0.65, 0.75};
  std::vector<double> eucm_beta_candidates{0.6, 0.8, 1.0, 1.2, 1.5};
  std::vector<double> equidistant_k1_candidates{-0.15, 0.0, 0.15};
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
void WriteAutoCameraInitializationBootstrapViewsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_ONLY_CAMERA_INITIALIZER_HPP
