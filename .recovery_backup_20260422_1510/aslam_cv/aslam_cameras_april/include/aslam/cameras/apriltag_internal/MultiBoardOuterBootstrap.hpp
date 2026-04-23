#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_OUTER_BOOTSTRAP_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_OUTER_BOOTSTRAP_HPP

#include <string>
#include <vector>

#include <Eigen/Core>

#include <aslam/cameras/apriltag_internal/ApriltagCanonicalModel.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct OuterBootstrapFrameInput {
  int frame_index = -1;
  std::string frame_label;
  OuterFrameMeasurementResult measurements;
};

struct OuterBootstrapOptions {
  int reference_board_id = 1;
  double init_xi = 0.0;
  double init_alpha = 0.6;
  double init_fu_scale = 0.5;
  double init_fv_scale = 0.5;
  double init_cu_offset = 0.0;
  double init_cv_offset = 0.0;
  int image_width = 0;
  int image_height = 0;
  int max_initialization_passes = 8;
  int max_optimization_iterations = 4;
  double convergence_threshold = 1e-3;
};

struct OuterBootstrapCameraIntrinsics {
  double xi = 0.0;
  double alpha = 0.6;
  double fu = 0.0;
  double fv = 0.0;
  double cu = 0.0;
  double cv = 0.0;
  int width = 0;
  int height = 0;

  bool IsValid() const {
    return width > 0 && height > 0 && fu > 0.0 && fv > 0.0;
  }
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
  double rmse = 0.0;
};

struct OuterBootstrapResult {
  bool success = false;
  int reference_board_id = 1;
  OuterBootstrapCameraIntrinsics coarse_camera;
  std::vector<OuterBootstrapBoardState> boards;
  std::vector<OuterBootstrapFrameState> frames;
  int used_frame_count = 0;
  int used_board_observation_count = 0;
  int used_corner_count = 0;
  double global_rmse = 0.0;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class MultiBoardOuterBootstrap {
 public:
  MultiBoardOuterBootstrap(ApriltagInternalConfig config,
                           OuterBootstrapOptions options = OuterBootstrapOptions{});

  OuterBootstrapResult Solve(const std::vector<OuterBootstrapFrameInput>& frames) const;

  const ApriltagInternalConfig& config() const { return config_; }
  const OuterBootstrapOptions& options() const { return options_; }

 private:
  std::vector<Eigen::Vector3d> OuterCornerTargets() const;

  ApriltagInternalConfig config_;
  ApriltagCanonicalModel canonical_model_;
  OuterBootstrapOptions options_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_OUTER_BOOTSTRAP_HPP
