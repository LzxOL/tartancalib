#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_MEASUREMENT_BUILDER_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_MEASUREMENT_BUILDER_HPP

#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagCanonicalModel.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct JointMeasurementFrameInput {
  int frame_index = -1;
  std::string frame_label;
  OuterTagMultiDetectionResult outer_detections;
  InternalRegenerationFrameResult regenerated_internal;
};

struct JointMeasurementBuildOptions {
  int reference_board_id = 1;
  bool include_outer_points = true;
  bool include_internal_points = true;
  bool include_outer_when_internal_failed = true;
  bool require_initialized_frame_and_board = true;
  bool filter_internal_corner_outliers = true;
  double filter_internal_corner_sigma_threshold = 2.0;
  double filter_internal_corner_min_reproj_error = 0.2;
};

enum class JointPointType {
  Outer,
  Internal,
};

enum class JointRejectionReasonCode {
  None = 0,
  FrameNotFoundInBootstrap,
  FrameLabelMismatch,
  FrameNotInitialized,
  BoardNotInitialized,
  NotReferenceConnected,
  MissingOuterBoardObservation,
  OuterMeasurementInvalid,
  MissingRegeneratedBoardResult,
  InternalPointInvalid,
  InternalPointReprojectionOutlier,
};

enum class JointObservationSourceKind {
  OuterMeasurement,
  InternalMeasurement,
};

const char* ToString(JointPointType point_type);
const char* ToString(JointRejectionReasonCode reason_code);
const char* ToString(JointObservationSourceKind source_kind);

struct JointPointObservation {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  JointPointType point_type = JointPointType::Outer;
  Eigen::Vector2d image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector3d target_xyz_board = Eigen::Vector3d::Zero();
  double quality = 0.0;
  bool used_in_solver = false;
  JointRejectionReasonCode rejection_reason_code = JointRejectionReasonCode::None;
  std::string rejection_detail;
  int frame_storage_index = -1;
  int source_board_observation_index = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind = JointObservationSourceKind::OuterMeasurement;
};

struct JointBoardObservation {
  int board_id = -1;
  bool frame_bootstrap_initialized = false;
  bool board_bootstrap_initialized = false;
  bool reference_connected = false;
  bool used_in_solver = false;
  int outer_point_count = 0;
  int internal_point_count = 0;
  std::vector<JointPointObservation> points;
};

struct JointMeasurementFrameResult {
  int frame_index = -1;
  std::string frame_label;
  bool frame_bootstrap_initialized = false;
  std::vector<int> visible_board_ids;
  std::vector<JointBoardObservation> board_observations;
};

struct JointMeasurementBuildResult {
  bool success = false;
  int reference_board_id = 1;
  OuterBootstrapResult bootstrap_seed;
  std::vector<JointMeasurementFrameResult> frames;
  std::vector<JointPointObservation> solver_observations;
  int used_frame_count = 0;
  int accepted_outer_board_observation_count = 0;
  int accepted_internal_board_observation_count = 0;
  int used_board_observation_count = 0;
  int used_outer_point_count = 0;
  int used_internal_point_count = 0;
  int used_total_point_count = 0;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class JointReprojectionMeasurementBuilder {
 public:
  explicit JointReprojectionMeasurementBuilder(
      ApriltagInternalConfig base_config,
      JointMeasurementBuildOptions options = JointMeasurementBuildOptions{});

  JointMeasurementBuildResult Build(
      const std::vector<JointMeasurementFrameInput>& frames,
      const OuterBootstrapResult& bootstrap_result) const;

  void DrawFrameOverlay(const cv::Mat& image,
                        const JointMeasurementFrameResult& frame_result,
                        cv::Mat* output_image) const;

  const ApriltagInternalConfig& base_config() const { return base_config_; }
  const JointMeasurementBuildOptions& options() const { return options_; }

 private:
  ApriltagCanonicalModel ModelForBoardId(int board_id) const;

  const OuterBootstrapFrameState* FindBootstrapFrameState(
      const OuterBootstrapResult& bootstrap_result,
      int frame_index) const;
  const OuterBootstrapBoardState* FindBootstrapBoardState(
      const OuterBootstrapResult& bootstrap_result,
      int board_id) const;
  const OuterBootstrapObservationDiagnostics* FindObservationDiagnostics(
      const OuterBootstrapResult& bootstrap_result,
      int frame_index,
      int board_id) const;

  ApriltagInternalConfig base_config_;
  JointMeasurementBuildOptions options_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_MEASUREMENT_BUILDER_HPP
