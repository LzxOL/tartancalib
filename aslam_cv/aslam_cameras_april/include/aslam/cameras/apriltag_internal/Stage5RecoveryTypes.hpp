#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_RECOVERY_TYPES_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_RECOVERY_TYPES_HPP

#include <array>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct FrozenRound2BaselineFrameSource {
  int frame_index = -1;
  std::string frame_label;
  std::string image_path;
};

struct CameraAwareOuterRescueRecord {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  std::string baseline_failure_reason;
  std::string rescue_summary;
  int hamming = -1;
  std::array<Eigen::Vector2d, 4> committed_corners{};
};

struct CameraAwareOuterRescueSummary {
  bool requested = false;
  bool enabled = false;
  bool camera_family_supported = false;
  std::string camera_source = "unavailable";
  int uses_yaml_intrinsics = 0;
  int uses_kalibr_camchain_intrinsics = 0;
  int patch_size = 640;
  std::string patch_plan = "dense_5x5_fov56_plus_wide_3x3_fov72";
  int max_hamming = 0;
  int frame_count = 0;
  int requested_board_observation_count = 0;
  int baseline_success_count = 0;
  int baseline_all_boards_frame_count = 0;
  int attempted_frame_count = 0;
  int attempted_board_observation_count = 0;
  bool zero_detection_atlas_enabled = false;
  int zero_detection_frame_count = 0;
  int zero_detection_atlas_attempted_board_observation_count = 0;
  int worker_count = 1;
  bool direct_layout_geometry_gate_enabled = false;
  bool direct_layout_geometry_gate_available = false;
  double direct_layout_geometry_gate_max_rmse_px = 25.0;
  int direct_layout_geometry_gate_evaluated_count = 0;
  int direct_layout_geometry_gate_accepted_count = 0;
  int direct_layout_geometry_gate_rejected_count = 0;
  int direct_layout_geometry_gate_not_evaluable_count = 0;
  int temporal_seed_attempted_board_observation_count = 0;
  int temporal_seed_rescued_board_observation_count = 0;
  int rescued_board_observation_count = 0;
  int final_success_count = 0;
  int final_all_boards_frame_count = 0;
  bool camera_initialization_rerun = false;
  bool camera_initialization_rerun_success = false;
  OuterBootstrapCameraIntrinsics provisional_camera;
  OuterBootstrapCameraIntrinsics final_initialization_camera;
  double runtime_seconds = 0.0;
  std::string skip_reason;
  std::vector<CameraAwareOuterRescueRecord> records;
  std::vector<std::string> warnings;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif
