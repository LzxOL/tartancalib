#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_OUTER_BOOTSTRAP_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_OUTER_BOOTSTRAP_HPP

#include <array>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagCanonicalModel.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct OuterBootstrapCameraIntrinsics {
  double xi = 0.0;
  double alpha = 0.0;
  double fu = 0.0;
  double fv = 0.0;
  double cu = 0.0;
  double cv = 0.0;
  cv::Size resolution;

  bool IsValid() const {
    return resolution.width > 0 && resolution.height > 0 && fu > 0.0 && fv > 0.0;
  }
};

struct OuterBootstrapOptions {
  int reference_board_id = 1;
  double init_xi = -0.2;
  double init_alpha = 0.6;
  double init_fu_scale = 0.55;
  double init_fv_scale = 0.55;
  double init_cu_offset = 0.0;
  double init_cv_offset = 0.0;
  int max_coordinate_descent_iterations = 6;
  double convergence_threshold = 1e-3;
  double min_detection_quality = 0.0;
};

struct OuterBootstrapFrameInput {
  int frame_index = -1;
  std::string frame_label;
  OuterFrameMeasurementResult measurements;
};

struct OuterBootstrapBoardState {
  int board_id = -1;
  bool initialized = false;
  Eigen::Matrix4d T_reference_board = Eigen::Matrix4d::Identity();
  int observation_count = 0;
  double rmse = 0.0;
};

struct OuterBootstrapFrameState {
  int frame_index = -1;
  std::string frame_label;
  bool initialized = false;
  std::vector<int> visible_board_ids;
  Eigen::Matrix4d T_camera_reference = Eigen::Matrix4d::Identity();
  int observation_count = 0;
  double rmse = 0.0;
};

struct OuterBootstrapObservationDiagnostics {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  double detection_quality = 0.0;
  bool reference_connected = false;
  bool frame_initialized = false;
  bool board_initialized = false;
  bool used_in_solve = false;
  double observation_rmse = 0.0;
  std::array<Eigen::Vector2d, 4> corner_residuals_xy{};
  double max_abs_residual_x = 0.0;
  double max_abs_residual_y = 0.0;
};

struct OuterBootstrapResult {
  bool success = false;
  int reference_board_id = 1;
  OuterBootstrapCameraIntrinsics coarse_camera;
  std::vector<OuterBootstrapBoardState> boards;
  std::vector<OuterBootstrapFrameState> frames;
  std::vector<OuterBootstrapObservationDiagnostics> observation_diagnostics;
  int used_frame_count = 0;
  int used_board_observation_count = 0;
  int used_corner_count = 0;
  double global_rmse = 0.0;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class MultiBoardOuterBootstrap {
 public:
  explicit MultiBoardOuterBootstrap(
      ApriltagInternalConfig base_config,
      OuterBootstrapOptions options = OuterBootstrapOptions{});

  OuterBootstrapResult Solve(const std::vector<OuterBootstrapFrameInput>& frames) const;

  const ApriltagInternalConfig& base_config() const { return base_config_; }
  const OuterBootstrapOptions& options() const { return options_; }

 private:
  ApriltagCanonicalModel ModelForBoardId(int board_id) const;

  ApriltagInternalConfig base_config_;
  OuterBootstrapOptions options_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_OUTER_BOOTSTRAP_HPP
