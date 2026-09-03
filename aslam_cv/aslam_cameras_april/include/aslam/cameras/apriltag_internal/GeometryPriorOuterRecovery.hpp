#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_GEOMETRY_PRIOR_OUTER_RECOVERY_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_GEOMETRY_PRIOR_OUTER_RECOVERY_HPP

#include <array>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

std::array<Eigen::Vector3d, 4> BuildOuterCornerPoints(
    const ApriltagCanonicalModel& model);
double RotationErrorDegrees(const Eigen::Matrix3d& lhs,
                            const Eigen::Matrix3d& rhs);
std::array<Eigen::Vector3d, 4> BuildOuterCornerPointsForBoard(
    const ApriltagInternalConfig& config,
    int board_id);
std::vector<Eigen::Vector3d> ToVector(
    const std::array<Eigen::Vector3d, 4>& values);
std::vector<cv::Point2f> ToVector(
    const std::array<cv::Point2f, 4>& values);
std::vector<cv::Point2f> ToImagePoints(
    const std::array<Eigen::Vector2d, 4>& values);

// Projects the canonical outer corners using the current board pose prior.
// The four projected corners must remain inside the image.
bool ProjectGeometryPriorOuterCorners(
    const IntermediateCameraConfig& camera_config,
    const ApriltagCanonicalModel& model,
    const Eigen::Matrix4d& T_camera_board_matrix,
    const cv::Size& image_size,
    std::array<Eigen::Vector2d, 4>* corners);

// Creates the diagnostic candidate record before image evidence is evaluated.
GeometryPriorOuterSeedCandidate BuildGeometryPriorOuterSeedCandidate(
    const InternalRegenerationFrameInput& frame_input,
    int board_id,
    const std::vector<int>& visible_boards_used,
    const std::array<Eigen::Vector2d, 4>& corners,
    const std::string& prediction_source_label,
    int frame_pose_refit_source_board_id,
    double frame_pose_refit_outer_rmse,
    const std::string& original_failure_reason);

// Runs the existing geometry-prior recovery chain: local ROI/ray-patch
// evidence, corner refinement, tag-likelihood checks, pose consistency, and
// final observation acceptance. The function intentionally owns no frame
// orchestration or board-selection policy.
GeometryPriorOuterSeedCandidate EvaluateGeometryPriorOuterSeedCandidate(
    const cv::Mat& gray,
    const ApriltagInternalConfig& config,
    const ApriltagInternalDetectionOptions& options,
    const IntermediateCameraConfig& camera_config,
    const InternalRegenerationFrameInput& frame_input,
    int board_id,
    const std::vector<int>& visible_boards_used,
    const std::array<Eigen::Vector2d, 4>& predicted_corners,
    const Eigen::Matrix4d& T_camera_board_matrix,
    const std::string& prediction_source_label,
    int frame_pose_refit_source_board_id,
    double frame_pose_refit_outer_rmse,
    double frame_normal_outer_refit_rmse_median,
    const std::string& original_failure_reason,
    const std::vector<std::array<Eigen::Vector2d, 4>>& competing_topology_slots,
    const OuterWrongIdProposal* wrong_id_proposal,
    OuterTagDetectionResult* rescued_detection);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif
