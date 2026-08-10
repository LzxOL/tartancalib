#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>

#include <aslam/cameras/apriltag_internal/OuterDetectionResultUtils.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

#include "apriltags/TagDetection.h"
#include "apriltags/TagDetector.h"
#include "apriltags/TagFamily.h"
#include "apriltags/Tag36h11.h"

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

void AppendUniqueBoardId(int board_id, std::vector<int>* board_ids) {
  if (board_ids == nullptr || board_id < 0) {
    return;
  }
  if (std::find(board_ids->begin(), board_ids->end(), board_id) == board_ids->end()) {
    board_ids->push_back(board_id);
  }
}

void CollectVisibleBoardIds(
    const InternalRegenerationFrameInput& frame_input,
    std::vector<int>* board_ids) {
  if (board_ids == nullptr) {
    return;
  }
  for (std::size_t index = 0;
       index < frame_input.outer_detections.requested_board_ids.size();
       ++index) {
    if (index < frame_input.outer_detections.detections.size() &&
        frame_input.outer_detections.detections[index].success) {
      AppendUniqueBoardId(
          frame_input.outer_detections.requested_board_ids[index], board_ids);
    }
  }
}

// A single board provides no same-frame layout consensus. It may seed
// geometry-guided recovery only when it is a genuine, exact-ID detector
// observation with four verified corners, rather than a prior rescue.
bool HasDirectExactOuterAnchor(
    const InternalRegenerationFrameInput& frame_input,
    int board_id) {
  if (board_id < 0) {
    return false;
  }
  bool detector_anchor_valid = false;
  for (std::size_t index = 0;
       index < frame_input.outer_detections.requested_board_ids.size();
       ++index) {
    if (frame_input.outer_detections.requested_board_ids[index] != board_id ||
        index >= frame_input.outer_detections.detections.size()) {
      continue;
    }
    const OuterTagDetectionResult& detection =
        frame_input.outer_detections.detections[index];
    detector_anchor_valid =
        detection.success && detection.good &&
        !detection.used_local_patch_rescue &&
        detection.detected_tag_id == board_id && detection.hamming == 0 &&
        std::all_of(detection.refined_valid.begin(), detection.refined_valid.end(),
                    [](bool valid) { return valid; });
    break;
  }
  if (!detector_anchor_valid) {
    return false;
  }
  for (const OuterBoardMeasurement& measurement :
       frame_input.outer_detections.frame_measurements.board_measurements) {
    if (measurement.board_id != board_id) {
      continue;
    }
    return measurement.success && !measurement.used_local_patch_rescue &&
           measurement.detected_tag_id == board_id &&
           measurement.valid_refined_corner_count == 4 &&
           std::all_of(measurement.refined_corner_valid.begin(),
                       measurement.refined_corner_valid.end(),
                       [](bool valid) { return valid; });
  }
  return false;
}

bool IsExactImageValidatedOuterMeasurement(
    const OuterBoardMeasurement& measurement) {
  if (!measurement.success || !measurement.used_local_patch_rescue ||
      measurement.detected_tag_id != measurement.board_id ||
      measurement.valid_refined_corner_count != 4) {
    return false;
  }
  // The outer-corner target geometry is always reconstructed from the board
  // configuration at the call site. The regular detector-side measurement
  // intentionally does not carry a duplicate target-corner array, so requiring
  // has_target_outer_corners here silently discarded exact sphere-patch
  // recoveries before the local-frame pose refit.
  return std::all_of(
      measurement.refined_corner_valid.begin(),
      measurement.refined_corner_valid.end(),
      [](bool valid) { return valid; });
}

void AppendUniqueWarning(const std::string& warning,
                         std::vector<std::string>* warnings) {
  if (warnings == nullptr || warning.empty()) {
    return;
  }
  if (std::find(warnings->begin(), warnings->end(), warning) == warnings->end()) {
    warnings->push_back(warning);
  }
}

std::string JoinBoardIds(const std::vector<int>& board_ids) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < board_ids.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << board_ids[index];
  }
  return stream.str();
}

Eigen::Matrix4d ComposeCameraBoardTransform(const Eigen::Matrix4d& T_camera_reference,
                                            const Eigen::Matrix4d& T_reference_board) {
  return T_camera_reference * T_reference_board;
}

std::array<Eigen::Vector3d, 4> BuildOuterCornerPoints(
    const ApriltagCanonicalModel& model) {
  const std::array<int, 4> point_ids{
      model.PointId(0, 0),
      model.PointId(model.ModuleDimension(), 0),
      model.PointId(model.ModuleDimension(), model.ModuleDimension()),
      model.PointId(0, model.ModuleDimension()),
  };
  std::array<Eigen::Vector3d, 4> points{};
  for (int index = 0; index < 4; ++index) {
    points[static_cast<std::size_t>(index)] =
        model.corner(point_ids[static_cast<std::size_t>(index)]).target_xyz;
  }
  return points;
}

bool ProjectGeometryPriorOuterCorners(
    const IntermediateCameraConfig& camera_config,
    const ApriltagCanonicalModel& model,
    const Eigen::Matrix4d& T_camera_board_matrix,
    const cv::Size& image_size,
    std::array<Eigen::Vector2d, 4>* corners) {
  if (corners == nullptr) {
    return false;
  }
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(camera_config);
  if (!camera.IsValid()) {
    return false;
  }
  const Eigen::Isometry3d T_camera_board = ToIsometry3d(T_camera_board_matrix);
  const std::array<Eigen::Vector3d, 4> board_points =
      BuildOuterCornerPoints(model);
  int inside_count = 0;
  for (int index = 0; index < 4; ++index) {
    const Eigen::Vector3d point_camera =
        T_camera_board * board_points[static_cast<std::size_t>(index)];
    Eigen::Vector2d keypoint;
    if (!camera.vsEuclideanToKeypoint(point_camera, &keypoint)) {
      return false;
    }
    (*corners)[static_cast<std::size_t>(index)] = keypoint;
    if (keypoint.x() >= 0.0 && keypoint.y() >= 0.0 &&
        keypoint.x() < image_size.width && keypoint.y() < image_size.height) {
      ++inside_count;
    }
  }
  return inside_count == 4;
}

double SignedPolygonArea(const std::array<cv::Point2f, 4>& corners) {
  double area = 0.0;
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f& a = corners[static_cast<std::size_t>(index)];
    const cv::Point2f& b = corners[static_cast<std::size_t>((index + 1) % 4)];
    area += static_cast<double>(a.x) * static_cast<double>(b.y) -
            static_cast<double>(b.x) * static_cast<double>(a.y);
  }
  return area * 0.5;
}

double PolygonArea(const std::array<cv::Point2f, 4>& corners) {
  return std::fabs(SignedPolygonArea(corners));
}

double Cross2d(const cv::Point2f& a,
               const cv::Point2f& b,
               const cv::Point2f& c) {
  return static_cast<double>(b.x - a.x) * static_cast<double>(c.y - a.y) -
         static_cast<double>(b.y - a.y) * static_cast<double>(c.x - a.x);
}

struct QuadTopologyCheck {
  bool valid = false;
  bool finite = false;
  bool convex = false;
  bool self_intersecting = false;
  double signed_area_px = 0.0;
  double area_px = 0.0;
  double min_edge_length_px = 0.0;
  std::string summary;
};

bool ProperSegmentsIntersect(const cv::Point2f& a,
                             const cv::Point2f& b,
                             const cv::Point2f& c,
                             const cv::Point2f& d) {
  constexpr double kCrossEpsilon = 1e-6;
  const double ab_c = Cross2d(a, b, c);
  const double ab_d = Cross2d(a, b, d);
  const double cd_a = Cross2d(c, d, a);
  const double cd_b = Cross2d(c, d, b);
  return ((ab_c > kCrossEpsilon && ab_d < -kCrossEpsilon) ||
          (ab_c < -kCrossEpsilon && ab_d > kCrossEpsilon)) &&
         ((cd_a > kCrossEpsilon && cd_b < -kCrossEpsilon) ||
          (cd_a < -kCrossEpsilon && cd_b > kCrossEpsilon));
}

QuadTopologyCheck CheckQuadTopology(
    const std::array<cv::Point2f, 4>& corners) {
  QuadTopologyCheck check;
  check.finite = std::all_of(
      corners.begin(), corners.end(), [](const cv::Point2f& point) {
        return std::isfinite(point.x) && std::isfinite(point.y);
      });
  if (!check.finite) {
    check.summary = "nonfinite_corner";
    return check;
  }

  check.signed_area_px = SignedPolygonArea(corners);
  check.area_px = std::fabs(check.signed_area_px);
  check.min_edge_length_px = std::numeric_limits<double>::infinity();
  int positive_turn_count = 0;
  int negative_turn_count = 0;
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f edge =
        corners[static_cast<std::size_t>((index + 1) % 4)] -
        corners[static_cast<std::size_t>(index)];
    check.min_edge_length_px = std::min(
        check.min_edge_length_px,
        std::hypot(static_cast<double>(edge.x), static_cast<double>(edge.y)));
    const double turn = Cross2d(
        corners[static_cast<std::size_t>(index)],
        corners[static_cast<std::size_t>((index + 1) % 4)],
        corners[static_cast<std::size_t>((index + 2) % 4)]);
    positive_turn_count += turn > 1e-6 ? 1 : 0;
    negative_turn_count += turn < -1e-6 ? 1 : 0;
  }
  check.convex = positive_turn_count == 4 || negative_turn_count == 4;
  check.self_intersecting =
      ProperSegmentsIntersect(corners[0], corners[1], corners[2], corners[3]) ||
      ProperSegmentsIntersect(corners[1], corners[2], corners[3], corners[0]);
  check.valid = check.area_px >= 4.0 &&
                check.min_edge_length_px >= 2.0 && check.convex &&
                !check.self_intersecting;
  std::ostringstream summary;
  summary << "finite=" << (check.finite ? 1 : 0)
          << " convex=" << (check.convex ? 1 : 0)
          << " self_intersecting=" << (check.self_intersecting ? 1 : 0)
          << " signed_area=" << check.signed_area_px
          << " min_edge=" << check.min_edge_length_px;
  check.summary = summary.str();
  return check;
}

struct WrongIdTopologyAssociation {
  bool compatible = false;
  std::array<Eigen::Vector2d, 4> ordered_corners{};
  int cyclic_shift = 0;
  bool reflected_order = false;
  double normalized_corner_rmse = std::numeric_limits<double>::infinity();
  double normalized_center_error = std::numeric_limits<double>::infinity();
  double area_ratio = std::numeric_limits<double>::quiet_NaN();
  std::string summary;
};

std::array<cv::Point2f, 4> ToCvCorners(
    const std::array<Eigen::Vector2d, 4>& corners) {
  std::array<cv::Point2f, 4> converted{};
  for (int index = 0; index < 4; ++index) {
    const Eigen::Vector2d& corner = corners[static_cast<std::size_t>(index)];
    converted[static_cast<std::size_t>(index)] =
        cv::Point2f(static_cast<float>(corner.x()),
                    static_cast<float>(corner.y()));
  }
  return converted;
}

// A decoder result with the wrong ID is not evidence for every missing board.
// Associate it to a topology-predicted board only when its image quadrilateral
// independently agrees in position, scale, and corner ordering.  The caller
// still performs image/code/pose validation before any observation is used.
WrongIdTopologyAssociation AssociateWrongIdProposalToTopology(
    const std::array<Eigen::Vector2d, 4>& topology_predicted_corners,
    const OuterWrongIdProposal& proposal) {
  WrongIdTopologyAssociation association;
  const std::array<cv::Point2f, 4> predicted_cv =
      ToCvCorners(topology_predicted_corners);
  const std::array<cv::Point2f, 4> proposal_cv =
      ToCvCorners(proposal.corners_original_image);
  const QuadTopologyCheck predicted_topology = CheckQuadTopology(predicted_cv);
  const QuadTopologyCheck proposal_topology = CheckQuadTopology(proposal_cv);
  if (!predicted_topology.valid || !proposal_topology.valid) {
    association.summary = "topology_invalid_quad predicted{" +
                          predicted_topology.summary + "} proposal{" +
                          proposal_topology.summary + "}";
    return association;
  }

  const double predicted_scale =
      std::sqrt(std::max(1.0, predicted_topology.area_px));
  Eigen::Vector2d predicted_center = Eigen::Vector2d::Zero();
  Eigen::Vector2d proposal_center = Eigen::Vector2d::Zero();
  for (int index = 0; index < 4; ++index) {
    predicted_center += topology_predicted_corners[static_cast<std::size_t>(index)];
    proposal_center += proposal.corners_original_image[static_cast<std::size_t>(index)];
  }
  predicted_center *= 0.25;
  proposal_center *= 0.25;
  association.normalized_center_error =
      (predicted_center - proposal_center).norm() / predicted_scale;
  association.area_ratio = proposal_topology.area_px / predicted_topology.area_px;

  double best_squared_error = std::numeric_limits<double>::infinity();
  for (int reflected = 0; reflected <= 1; ++reflected) {
    for (int shift = 0; shift < 4; ++shift) {
      std::array<Eigen::Vector2d, 4> ordered{};
      double squared_error = 0.0;
      for (int index = 0; index < 4; ++index) {
        const int proposal_index =
            reflected == 0 ? (shift + index) % 4 : (shift - index + 4) % 4;
        ordered[static_cast<std::size_t>(index)] =
            proposal.corners_original_image[static_cast<std::size_t>(proposal_index)];
        squared_error +=
            (topology_predicted_corners[static_cast<std::size_t>(index)] -
             ordered[static_cast<std::size_t>(index)]).squaredNorm();
      }
      if (squared_error < best_squared_error) {
        best_squared_error = squared_error;
        association.ordered_corners = ordered;
        association.cyclic_shift = shift;
        association.reflected_order = reflected != 0;
      }
    }
  }
  association.normalized_corner_rmse =
      std::sqrt(best_squared_error / 4.0) / predicted_scale;

  // Relative gates remain stable across image resolutions and Tag sizes. They
  // intentionally sit well above normal scene-prediction error, while a quad
  // belonging to a different physical board is usually separated by many Tag
  // widths in this capture geometry.
  constexpr double kMaxNormalizedCornerRmse = 0.18;
  constexpr double kMaxNormalizedCenterError = 0.15;
  constexpr double kMinAreaRatio = 0.55;
  constexpr double kMaxAreaRatio = 1.80;
  association.compatible =
      association.normalized_corner_rmse <= kMaxNormalizedCornerRmse &&
      association.normalized_center_error <= kMaxNormalizedCenterError &&
      association.area_ratio >= kMinAreaRatio &&
      association.area_ratio <= kMaxAreaRatio;
  std::ostringstream stream;
  stream << "topology_assoc compatible=" << (association.compatible ? 1 : 0)
         << " corner_rmse_norm=" << association.normalized_corner_rmse
         << " center_norm=" << association.normalized_center_error
         << " area_ratio=" << association.area_ratio
         << " cyclic_shift=" << association.cyclic_shift
         << " reflected=" << (association.reflected_order ? 1 : 0)
         << " proposal_id=" << proposal.detected_tag_id
         << " proposal_hamming=" << proposal.hamming
         << " proposal_source=" << proposal.source;
  association.summary = stream.str();
  return association;
}

GeometryPriorOuterSeedCandidate BuildGeometryPriorOuterSeedCandidate(
    const InternalRegenerationFrameInput& frame_input,
    int board_id,
    const std::vector<int>& visible_boards_used,
    const std::array<Eigen::Vector2d, 4>& corners,
    const std::string& prediction_source_label,
    int frame_pose_refit_source_board_id,
    double frame_pose_refit_outer_rmse,
    const std::string& original_failure_reason);

struct LocalFramePoseRefit {
  bool success = false;
  int source_board_id = -1;
  std::vector<int> support_board_ids;
  struct Candidate {
    int source_board_id = -1;
    double outer_rmse = std::numeric_limits<double>::infinity();
    Eigen::Matrix4d T_camera_reference = Eigen::Matrix4d::Identity();
  };
  std::vector<Candidate> candidates;
  double outer_rmse = 0.0;
  Eigen::Matrix4d T_camera_reference = Eigen::Matrix4d::Identity();
};

double PointDistance(const cv::Point2f& lhs, const cv::Point2f& rhs) {
  const double dx = static_cast<double>(lhs.x) - static_cast<double>(rhs.x);
  const double dy = static_cast<double>(lhs.y) - static_cast<double>(rhs.y);
  return std::sqrt(dx * dx + dy * dy);
}

double CornerLocalScale(const std::array<cv::Point2f, 4>& corners,
                        int corner_index) {
  const int prev_index = (corner_index + 3) % 4;
  const int next_index = (corner_index + 1) % 4;
  const cv::Point2f& corner = corners[static_cast<std::size_t>(corner_index)];
  const cv::Point2f prev_edge =
      corners[static_cast<std::size_t>(prev_index)] - corner;
  const cv::Point2f next_edge =
      corners[static_cast<std::size_t>(next_index)] - corner;
  return std::min(std::hypot(prev_edge.x, prev_edge.y),
                  std::hypot(next_edge.x, next_edge.y));
}

int ComputeGeometryPriorRescueSubpixWindowRadius(
    const ApriltagInternalDetectionOptions& options,
    const std::array<cv::Point2f, 4>& corners,
    double* local_corner_scale_px) {
  double scale_sum = 0.0;
  int scale_count = 0;
  for (int index = 0; index < 4; ++index) {
    const double scale = CornerLocalScale(corners, index);
    if (std::isfinite(scale) && scale > 0.0) {
      scale_sum += scale;
      ++scale_count;
    }
  }
  const double local_scale =
      scale_count > 0 ? scale_sum / static_cast<double>(scale_count) : 0.0;
  if (local_corner_scale_px != nullptr) {
    *local_corner_scale_px = local_scale;
  }

  if (options.geometry_prior_rescue_subpix_window_radius > 0) {
    return std::max(2, options.geometry_prior_rescue_subpix_window_radius);
  }
  if (options.geometry_prior_rescue_subpix_window_radius < 0) {
    return 0;
  }

  const MultiScaleOuterTagDetectorConfig& outer_config =
      options.outer_detector_config;
  // Match the normal outer-detector subpixel scale convention:
  //   radius = outer_subpix_scale * (tagSpacing * local_tag_edge_scale)
  // where tagSpacing is parsed into outer_corner_marker_ratio.
  //
  // Geometry-prior rescue does not have a decoded AprilTag quad yet, so its
  // local scale is estimated from the predicted/refined board quadrilateral.
  const double marker_width =
      outer_config.outer_corner_marker_ratio > 0.0
          ? outer_config.outer_corner_marker_ratio * local_scale
          : local_scale;
  const double scaled_radius =
      outer_config.outer_subpix_scale > 0.0
          ? outer_config.outer_subpix_scale * marker_width
          : static_cast<double>(outer_config.outer_subpix_window_min);
  const int min_radius = std::max(2, outer_config.outer_subpix_window_min);
  const int raw_radius = static_cast<int>(std::lround(scaled_radius));
  if (outer_config.outer_subpix_window_max > 0) {
    const int max_radius = std::max(min_radius, outer_config.outer_subpix_window_max);
    return std::max(min_radius, std::min(max_radius, raw_radius));
  }
  return std::max(min_radius, raw_radius);
}

bool IsInsideImage(const cv::Point2f& point,
                   const cv::Size& image_size,
                   double border) {
  return point.x >= border && point.y >= border &&
         point.x < static_cast<float>(image_size.width) - border &&
         point.y < static_cast<float>(image_size.height) - border;
}

std::array<cv::Point2f, 4> ExpandQuadAboutCenter(
    const std::array<cv::Point2f, 4>& corners,
    double scale) {
  cv::Point2f center(0.0f, 0.0f);
  for (const cv::Point2f& corner : corners) {
    center += corner;
  }
  center *= 0.25f;
  std::array<cv::Point2f, 4> expanded{};
  for (int index = 0; index < 4; ++index) {
    expanded[static_cast<std::size_t>(index)] =
        center + static_cast<float>(scale) *
                     (corners[static_cast<std::size_t>(index)] - center);
  }
  return expanded;
}

cv::Rect BuildExpandedBoundingRoi(const std::array<cv::Point2f, 4>& corners,
                                  const cv::Size& image_size,
                                  double scale) {
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  for (const cv::Point2f& point : corners) {
    min_x = std::min(min_x, point.x);
    min_y = std::min(min_y, point.y);
    max_x = std::max(max_x, point.x);
    max_y = std::max(max_y, point.y);
  }
  if (!std::isfinite(min_x) || !std::isfinite(min_y) ||
      !std::isfinite(max_x) || !std::isfinite(max_y) ||
      max_x <= min_x || max_y <= min_y) {
    return cv::Rect();
  }

  const float center_x = 0.5f * (min_x + max_x);
  const float center_y = 0.5f * (min_y + max_y);
  const float width = (max_x - min_x) * static_cast<float>(scale);
  const float height = (max_y - min_y) * static_cast<float>(scale);
  const int x0 = std::max(0, static_cast<int>(std::floor(center_x - 0.5f * width)));
  const int y0 = std::max(0, static_cast<int>(std::floor(center_y - 0.5f * height)));
  const int x1 = std::min(image_size.width,
                          static_cast<int>(std::ceil(center_x + 0.5f * width)));
  const int y1 = std::min(image_size.height,
                          static_cast<int>(std::ceil(center_y + 0.5f * height)));
  if (x1 <= x0 || y1 <= y0) {
    return cv::Rect();
  }
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

double QuadAreaFromTagDetection(const AprilTags::TagDetection& detection) {
  std::array<cv::Point2f, 4> corners{};
  for (int index = 0; index < 4; ++index) {
    corners[static_cast<std::size_t>(index)] =
        cv::Point2f(detection.p[index].first, detection.p[index].second);
  }
  return PolygonArea(corners);
}

bool NormalizeRay(Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    return false;
  }
  const double norm = ray->norm();
  if (!std::isfinite(norm) || norm <= 1e-12) {
    return false;
  }
  *ray /= norm;
  return true;
}

cv::Point2f QuadCenter(const std::array<cv::Point2f, 4>& corners) {
  cv::Point2f center(0.0f, 0.0f);
  for (const cv::Point2f& corner : corners) {
    center += corner;
  }
  center *= 0.25f;
  return center;
}

double MeanQuadEdgeLength(const std::array<cv::Point2f, 4>& corners) {
  double length_sum = 0.0;
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f delta =
        corners[static_cast<std::size_t>((index + 1) % 4)] -
        corners[static_cast<std::size_t>(index)];
    length_sum += std::hypot(delta.x, delta.y);
  }
  return 0.25 * length_sum;
}

bool QuadMatchesGeometryPriorSeed(
    const std::array<cv::Point2f, 4>& candidate_corners,
    const std::array<cv::Point2f, 4>& seed_corners,
    double* center_error_px,
    double* area_ratio,
    std::string* reject_reason) {
  const double seed_area = std::max(1.0, PolygonArea(seed_corners));
  const double candidate_area = std::max(1.0, PolygonArea(candidate_corners));
  const double ratio = candidate_area / seed_area;
  const double center_error =
      PointDistance(QuadCenter(candidate_corners), QuadCenter(seed_corners));
  const double seed_edge = std::max(1.0, MeanQuadEdgeLength(seed_corners));
  const double max_center_error = std::max(80.0, 0.75 * seed_edge);
  if (center_error_px != nullptr) {
    *center_error_px = center_error;
  }
  if (area_ratio != nullptr) {
    *area_ratio = ratio;
  }
  if (center_error > max_center_error) {
    if (reject_reason != nullptr) {
      std::ostringstream stream;
      stream << "center_error_too_large center_error_px=" << center_error
             << " max_center_error_px=" << max_center_error;
      *reject_reason = stream.str();
    }
    return false;
  }
  if (ratio < 0.20 || ratio > 5.00) {
    if (reject_reason != nullptr) {
      std::ostringstream stream;
      stream << "area_ratio_out_of_range area_ratio=" << ratio;
      *reject_reason = stream.str();
    }
    return false;
  }
  return true;
}

double CornerAssignmentCost(const std::array<cv::Point2f, 4>& candidate,
                            const std::array<cv::Point2f, 4>& seed) {
  double cost = 0.0;
  for (int index = 0; index < 4; ++index) {
    cost += PointDistance(candidate[static_cast<std::size_t>(index)],
                          seed[static_cast<std::size_t>(index)]);
  }
  return cost;
}

std::array<cv::Point2f, 4> ReorderQuadToMatchSeed(
    const std::array<cv::Point2f, 4>& raw_corners,
    const std::array<cv::Point2f, 4>& seed_corners,
    double* assignment_cost) {
  std::array<cv::Point2f, 4> best = raw_corners;
  double best_cost = std::numeric_limits<double>::infinity();
  for (int reverse = 0; reverse < 2; ++reverse) {
    for (int offset = 0; offset < 4; ++offset) {
      std::array<cv::Point2f, 4> ordered{};
      for (int index = 0; index < 4; ++index) {
        const int source_index =
            reverse ? (offset - index + 8) % 4 : (offset + index) % 4;
        ordered[static_cast<std::size_t>(index)] =
            raw_corners[static_cast<std::size_t>(source_index)];
      }
      const double cost = CornerAssignmentCost(ordered, seed_corners);
      if (cost < best_cost) {
        best_cost = cost;
        best = ordered;
      }
    }
  }
  if (assignment_cost != nullptr) {
    *assignment_cost = best_cost;
  }
  return best;
}

bool OrderContourQuad(const std::vector<cv::Point>& contour,
                      std::array<cv::Point2f, 4>* corners) {
  if (corners == nullptr || contour.size() != 4) {
    return false;
  }
  cv::Point2f center(0.0f, 0.0f);
  std::array<cv::Point2f, 4> ordered{};
  for (int index = 0; index < 4; ++index) {
    ordered[static_cast<std::size_t>(index)] =
        cv::Point2f(static_cast<float>(contour[static_cast<std::size_t>(index)].x),
                    static_cast<float>(contour[static_cast<std::size_t>(index)].y));
    center += ordered[static_cast<std::size_t>(index)];
  }
  center *= 0.25f;
  std::sort(ordered.begin(), ordered.end(),
            [center](const cv::Point2f& lhs, const cv::Point2f& rhs) {
              return std::atan2(lhs.y - center.y, lhs.x - center.x) <
                     std::atan2(rhs.y - center.y, rhs.x - center.x);
            });
  if (PolygonArea(ordered) <= 1.0) {
    return false;
  }
  *corners = ordered;
  return true;
}

std::array<cv::Point2f, 4> RectToQuad(const cv::RotatedRect& rect) {
  cv::Point2f points[4];
  rect.points(points);
  std::array<cv::Point2f, 4> corners{};
  for (int index = 0; index < 4; ++index) {
    corners[static_cast<std::size_t>(index)] = points[index];
  }
  return corners;
}

double SampleFloatAt(const cv::Mat& image, const cv::Point2f& point);

struct FittedImageLine {
  bool valid = false;
  cv::Point2f point;
  cv::Point2f direction;
  int support_count = 0;
  double mean_gradient_ratio = 0.0;
};

bool IntersectImageLines(const FittedImageLine& lhs,
                         const FittedImageLine& rhs,
                         cv::Point2f* intersection) {
  if (intersection == nullptr || !lhs.valid || !rhs.valid) {
    return false;
  }
  const double cross =
      static_cast<double>(lhs.direction.x) * rhs.direction.y -
      static_cast<double>(lhs.direction.y) * rhs.direction.x;
  if (!std::isfinite(cross) || std::abs(cross) < 1e-6) {
    return false;
  }
  const cv::Point2f delta = rhs.point - lhs.point;
  const double t =
      (static_cast<double>(delta.x) * rhs.direction.y -
       static_cast<double>(delta.y) * rhs.direction.x) /
      cross;
  *intersection = lhs.point + static_cast<float>(t) * lhs.direction;
  return std::isfinite(intersection->x) && std::isfinite(intersection->y);
}

FittedImageLine FitGeometryGuidedEdgeLine(
    const cv::Mat& grad_mag,
    double grad_norm,
    const cv::Size& image_size,
    const cv::Point2f& edge_start,
    const cv::Point2f& edge_end,
    int sample_count,
    int search_half_width,
    double min_gradient_ratio) {
  FittedImageLine fitted;
  const cv::Point2f edge = edge_end - edge_start;
  const double edge_len = std::max(1e-6, PointDistance(edge_start, edge_end));
  const cv::Point2f normal(-edge.y / static_cast<float>(edge_len),
                           edge.x / static_cast<float>(edge_len));
  std::vector<cv::Point2f> support_points;
  support_points.reserve(static_cast<std::size_t>(sample_count));
  double gradient_sum = 0.0;

  for (int sample = 0; sample < sample_count; ++sample) {
    const double t = (static_cast<double>(sample) + 0.5) /
                     static_cast<double>(sample_count);
    const cv::Point2f center = edge_start + static_cast<float>(t) * edge;
    double best_ratio = 0.0;
    cv::Point2f best_point = center;
    for (int offset = -search_half_width; offset <= search_half_width; ++offset) {
      const cv::Point2f probe =
          center + static_cast<float>(offset) * normal;
      if (!IsInsideImage(probe, image_size, 1.0)) {
        continue;
      }
      const double ratio = SampleFloatAt(grad_mag, probe) / grad_norm;
      if (ratio > best_ratio) {
        best_ratio = ratio;
        best_point = probe;
      }
    }
    if (best_ratio >= min_gradient_ratio) {
      support_points.push_back(best_point);
      gradient_sum += best_ratio;
    }
  }

  fitted.support_count = static_cast<int>(support_points.size());
  if (fitted.support_count > 0) {
    fitted.mean_gradient_ratio =
        gradient_sum / static_cast<double>(fitted.support_count);
  }
  const int min_support =
      std::max(8, static_cast<int>(std::lround(0.35 * sample_count)));
  if (fitted.support_count < min_support) {
    return fitted;
  }

  cv::Vec4f line;
  cv::fitLine(support_points, line, cv::DIST_HUBER, 0.0, 0.01, 0.01);
  fitted.direction = cv::Point2f(line[0], line[1]);
  fitted.point = cv::Point2f(line[2], line[3]);
  const double direction_norm =
      std::hypot(fitted.direction.x, fitted.direction.y);
  if (!std::isfinite(direction_norm) || direction_norm <= 1e-6) {
    return fitted;
  }
  fitted.direction.x /= static_cast<float>(direction_norm);
  fitted.direction.y /= static_cast<float>(direction_norm);
  fitted.valid = true;
  return fitted;
}

bool TryGeometryGuidedEdgeQuadProposal(
    const cv::Mat& gray,
    const std::array<cv::Point2f, 4>& predicted_corners,
    std::array<cv::Point2f, 4>* guided_corners,
    std::string* guided_summary) {
  if (guided_corners == nullptr || gray.empty()) {
    return false;
  }

  cv::Mat grad_x;
  cv::Mat grad_y;
  cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
  cv::Mat grad_mag;
  cv::magnitude(grad_x, grad_y, grad_mag);
  double grad_min = 0.0;
  double grad_max = 0.0;
  cv::minMaxLoc(grad_mag, &grad_min, &grad_max);
  const double grad_norm = std::max(1e-12, grad_max);

  const double seed_edge = std::max(1.0, MeanQuadEdgeLength(predicted_corners));
  const int sample_count =
      std::max(32, std::min(140, static_cast<int>(std::lround(seed_edge / 4.0))));
  const int search_half_width =
      std::max(10, std::min(48, static_cast<int>(std::lround(0.07 * seed_edge))));
  constexpr double kMinGradientRatio = 0.015;

  std::array<FittedImageLine, 4> lines{};
  int valid_line_count = 0;
  int support_total = 0;
  double gradient_ratio_sum = 0.0;
  for (int edge_index = 0; edge_index < 4; ++edge_index) {
    lines[static_cast<std::size_t>(edge_index)] =
        FitGeometryGuidedEdgeLine(
            grad_mag, grad_norm, gray.size(),
            predicted_corners[static_cast<std::size_t>(edge_index)],
            predicted_corners[static_cast<std::size_t>((edge_index + 1) % 4)],
            sample_count, search_half_width, kMinGradientRatio);
    if (lines[static_cast<std::size_t>(edge_index)].valid) {
      ++valid_line_count;
      support_total += lines[static_cast<std::size_t>(edge_index)].support_count;
      gradient_ratio_sum +=
          lines[static_cast<std::size_t>(edge_index)].mean_gradient_ratio;
    }
  }

  std::array<cv::Point2f, 4> intersections{};
  bool all_intersections_valid = valid_line_count == 4;
  for (int corner_index = 0; corner_index < 4 && all_intersections_valid;
       ++corner_index) {
    const FittedImageLine& previous =
        lines[static_cast<std::size_t>((corner_index + 3) % 4)];
    const FittedImageLine& current =
        lines[static_cast<std::size_t>(corner_index)];
    if (!IntersectImageLines(previous, current,
                             &intersections[static_cast<std::size_t>(corner_index)]) ||
        !IsInsideImage(intersections[static_cast<std::size_t>(corner_index)],
                       gray.size(), 1.0)) {
      all_intersections_valid = false;
    }
  }

  bool found = false;
  double center_error_px = 0.0;
  double area_ratio = 0.0;
  std::string reject_reason;
  if (all_intersections_valid) {
    double assignment_cost = 0.0;
    const std::array<cv::Point2f, 4> ordered =
        ReorderQuadToMatchSeed(intersections, predicted_corners,
                               &assignment_cost);
    if (QuadMatchesGeometryPriorSeed(ordered, predicted_corners,
                                     &center_error_px, &area_ratio,
                                     &reject_reason)) {
      *guided_corners = ordered;
      found = true;
    }
  }

  if (guided_summary != nullptr) {
    std::ostringstream stream;
    stream << "geometry_guided_edge_quad"
           << " valid_lines=" << valid_line_count
           << " support_total=" << support_total
           << " sample_count=" << sample_count
           << " search_half_width=" << search_half_width
           << " mean_line_gradient_ratio="
           << (valid_line_count > 0
                   ? gradient_ratio_sum / static_cast<double>(valid_line_count)
                   : 0.0);
    if (found) {
      stream << " center_error_px=" << center_error_px
             << " area_ratio=" << area_ratio;
    } else {
      stream << " reject="
             << (all_intersections_valid
                     ? (reject_reason.empty() ? "quad_prior_match_failed"
                                              : reject_reason)
                     : "line_intersection_failed");
    }
    *guided_summary = stream.str();
  }
  return found;
}

bool TryRelaxedRoiWeakQuadProposal(
    const cv::Mat& crop,
    const cv::Rect& roi,
    const std::array<cv::Point2f, 4>& predicted_corners,
    std::array<cv::Point2f, 4>* weak_corners,
    std::string* weak_summary) {
  if (weak_corners == nullptr || crop.empty()) {
    return false;
  }

  cv::Mat equalized;
  if (crop.channels() == 1) {
    cv::equalizeHist(crop, equalized);
  } else {
    cv::cvtColor(crop, equalized, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(equalized, equalized);
  }

  cv::Mat blurred;
  cv::GaussianBlur(equalized, blurred, cv::Size(3, 3), 0.0);

  std::vector<cv::Mat> edge_images;
  cv::Mat canny_low;
  cv::Canny(blurred, canny_low, 24.0, 72.0, 3, true);
  edge_images.push_back(canny_low);
  cv::Mat canny_mid;
  cv::Canny(blurred, canny_mid, 40.0, 120.0, 3, true);
  edge_images.push_back(canny_mid);
  cv::Mat adaptive;
  cv::adaptiveThreshold(blurred, adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY, 51, 3.0);
  cv::Mat adaptive_edges;
  cv::Canny(adaptive, adaptive_edges, 20.0, 80.0, 3, true);
  edge_images.push_back(adaptive_edges);

  const double seed_area = std::max(1.0, PolygonArea(predicted_corners));
  const double seed_edge = std::max(1.0, MeanQuadEdgeLength(predicted_corners));
  const double min_contour_area = std::max(64.0, 0.04 * seed_area);
  const double min_rect_area = std::max(64.0, 0.08 * seed_area);

  bool found = false;
  double best_score = std::numeric_limits<double>::infinity();
  double best_center_error_px = 0.0;
  double best_area_ratio = 0.0;
  double best_assignment_cost = 0.0;
  int contour_count_total = 0;
  int quad_candidate_count = 0;
  int matched_candidate_count = 0;
  std::string first_reject_reason;
  std::array<cv::Point2f, 4> best_corners{};

  for (const cv::Mat& edges : edge_images) {
    cv::Mat closed;
    cv::morphologyEx(edges, closed, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    contour_count_total += static_cast<int>(contours.size());
    for (const std::vector<cv::Point>& contour : contours) {
      const double contour_area = std::fabs(cv::contourArea(contour));
      if (contour_area < min_contour_area) {
        continue;
      }
      const double perimeter = cv::arcLength(contour, true);
      if (!std::isfinite(perimeter) || perimeter < 4.0 * std::sqrt(min_contour_area)) {
        continue;
      }

      std::vector<std::array<cv::Point2f, 4>> local_candidates;
      std::vector<cv::Point> approx;
      cv::approxPolyDP(contour, approx, 0.025 * perimeter, true);
      if (approx.size() == 4 && cv::isContourConvex(approx)) {
        std::array<cv::Point2f, 4> contour_quad{};
        if (OrderContourQuad(approx, &contour_quad)) {
          local_candidates.push_back(contour_quad);
        }
      }

      const cv::RotatedRect rect = cv::minAreaRect(contour);
      const double rect_area =
          static_cast<double>(rect.size.width) * static_cast<double>(rect.size.height);
      if (std::isfinite(rect_area) && rect_area >= min_rect_area) {
        local_candidates.push_back(RectToQuad(rect));
      }

      for (std::array<cv::Point2f, 4> local : local_candidates) {
        ++quad_candidate_count;
        std::array<cv::Point2f, 4> full{};
        for (int index = 0; index < 4; ++index) {
          full[static_cast<std::size_t>(index)] =
              local[static_cast<std::size_t>(index)] +
              cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
        }
        double assignment_cost = 0.0;
        const std::array<cv::Point2f, 4> ordered =
            ReorderQuadToMatchSeed(full, predicted_corners, &assignment_cost);
        double center_error_px = 0.0;
        double area_ratio = 0.0;
        std::string reject_reason;
        if (!QuadMatchesGeometryPriorSeed(ordered, predicted_corners,
                                          &center_error_px, &area_ratio,
                                          &reject_reason)) {
          if (first_reject_reason.empty()) {
            first_reject_reason = reject_reason;
          }
          continue;
        }
        ++matched_candidate_count;
        const double score =
            center_error_px / seed_edge +
            0.35 * std::abs(std::log(std::max(1e-6, area_ratio))) +
            0.04 * assignment_cost / seed_edge;
        if (score < best_score) {
          best_score = score;
          best_corners = ordered;
          best_center_error_px = center_error_px;
          best_area_ratio = area_ratio;
          best_assignment_cost = assignment_cost;
          found = true;
        }
      }
    }
  }

  if (weak_summary != nullptr) {
    std::ostringstream stream;
    stream << " relaxed_weak_quad contours=" << contour_count_total
           << " candidates=" << quad_candidate_count
           << " matched=" << matched_candidate_count;
    if (found) {
      stream << " center_error_px=" << best_center_error_px
             << " area_ratio=" << best_area_ratio
             << " assignment_cost_px=" << best_assignment_cost
             << " score=" << best_score;
    } else if (!first_reject_reason.empty()) {
      stream << " first_reject=" << first_reject_reason;
    }
    *weak_summary = stream.str();
  }
  if (!found) {
    return false;
  }
  *weak_corners = best_corners;
  return true;
}

bool BuildLocalDsPatchFrame(const DoubleSphereCameraModel& camera,
                            const cv::Point2f& image_center,
                            Eigen::Vector3d* center_ray,
                            Eigen::Vector3d* tangent_x,
                            Eigen::Vector3d* tangent_y) {
  if (center_ray == nullptr || tangent_x == nullptr || tangent_y == nullptr) {
    return false;
  }
  if (!camera.keypointToEuclidean(
          Eigen::Vector2d(image_center.x, image_center.y), center_ray) ||
      !NormalizeRay(center_ray)) {
    return false;
  }

  Eigen::Vector3d ray_x = Eigen::Vector3d::Zero();
  Eigen::Vector3d ray_y = Eigen::Vector3d::Zero();
  constexpr double kDeltaPx = 24.0;
  if (!camera.keypointToEuclidean(
          Eigen::Vector2d(image_center.x + kDeltaPx, image_center.y),
          &ray_x) ||
      !camera.keypointToEuclidean(
          Eigen::Vector2d(image_center.x, image_center.y + kDeltaPx),
          &ray_y) ||
      !NormalizeRay(&ray_x) || !NormalizeRay(&ray_y)) {
    return false;
  }

  *tangent_x = ray_x - (*center_ray) * center_ray->dot(ray_x);
  if (!NormalizeRay(tangent_x)) {
    return false;
  }
  *tangent_y = ray_y - (*center_ray) * center_ray->dot(ray_y);
  *tangent_y -= (*tangent_x) * tangent_x->dot(*tangent_y);
  if (!NormalizeRay(tangent_y)) {
    return false;
  }
  if (center_ray->dot(tangent_x->cross(*tangent_y)) < 0.0) {
    *tangent_y = -*tangent_y;
  }
  return true;
}

bool EstimateDsPatchFocalFromSeedCorners(
    const DoubleSphereCameraModel& camera,
    const std::array<cv::Point2f, 4>& seed_corners,
    const Eigen::Vector3d& center_ray,
    const Eigen::Vector3d& tangent_x,
    const Eigen::Vector3d& tangent_y,
    int patch_size,
    double* patch_focal,
    double* half_extent_out) {
  if (patch_focal == nullptr || patch_size <= 16) {
    return false;
  }
  double max_abs = 0.0;
  for (const cv::Point2f& corner : seed_corners) {
    Eigen::Vector3d ray = Eigen::Vector3d::Zero();
    if (!camera.keypointToEuclidean(Eigen::Vector2d(corner.x, corner.y), &ray) ||
        !NormalizeRay(&ray)) {
      return false;
    }
    const double denom = ray.dot(center_ray);
    if (!std::isfinite(denom) || denom <= 1e-6) {
      return false;
    }
    const double local_x = ray.dot(tangent_x) / denom;
    const double local_y = ray.dot(tangent_y) / denom;
    max_abs = std::max(max_abs, std::abs(local_x));
    max_abs = std::max(max_abs, std::abs(local_y));
  }
  if (!std::isfinite(max_abs) || max_abs <= 1e-4) {
    return false;
  }

  constexpr double kPatchMarginScale = 1.45;
  const double half_extent = max_abs * kPatchMarginScale;
  if (!std::isfinite(half_extent) || half_extent <= 1e-4) {
    return false;
  }
  *patch_focal = 0.5 * static_cast<double>(patch_size - 1) / half_extent;
  if (half_extent_out != nullptr) {
    *half_extent_out = half_extent;
  }
  return std::isfinite(*patch_focal) && *patch_focal > 1.0;
}

bool MapDsPatchPointToImage(const DoubleSphereCameraModel& camera,
                            const Eigen::Vector3d& center_ray,
                            const Eigen::Vector3d& tangent_x,
                            const Eigen::Vector3d& tangent_y,
                            double patch_focal,
                            int patch_size,
                            const cv::Point2f& patch_point,
                            Eigen::Vector2d* image_point) {
  if (image_point == nullptr || patch_focal <= 0.0 || patch_size <= 0) {
    return false;
  }
  const double cx = 0.5 * static_cast<double>(patch_size - 1);
  const double cy = 0.5 * static_cast<double>(patch_size - 1);
  const double nx = (static_cast<double>(patch_point.x) - cx) / patch_focal;
  const double ny = (static_cast<double>(patch_point.y) - cy) / patch_focal;
  Eigen::Vector3d ray = center_ray + nx * tangent_x + ny * tangent_y;
  if (!NormalizeRay(&ray)) {
    return false;
  }
  return camera.vsEuclideanToKeypoint(ray, image_point);
}

bool BuildDsRaySpacePatch(const cv::Mat& gray,
                          const DoubleSphereCameraModel& camera,
                          const Eigen::Vector3d& center_ray,
                          const Eigen::Vector3d& tangent_x,
                          const Eigen::Vector3d& tangent_y,
                          double patch_focal,
                          int patch_size,
                          cv::Mat* patch) {
  if (patch == nullptr || gray.empty() || patch_focal <= 0.0 ||
      patch_size <= 0) {
    return false;
  }
  patch->release();
  cv::Mat map_x(patch_size, patch_size, CV_32F);
  cv::Mat map_y(patch_size, patch_size, CV_32F);
  const double cx = 0.5 * static_cast<double>(patch_size - 1);
  const double cy = 0.5 * static_cast<double>(patch_size - 1);
  for (int y = 0; y < patch_size; ++y) {
    for (int x = 0; x < patch_size; ++x) {
      const double nx = (static_cast<double>(x) - cx) / patch_focal;
      const double ny = (static_cast<double>(y) - cy) / patch_focal;
      Eigen::Vector3d ray = center_ray + nx * tangent_x + ny * tangent_y;
      if (!NormalizeRay(&ray)) {
        map_x.at<float>(y, x) = -1.0f;
        map_y.at<float>(y, x) = -1.0f;
        continue;
      }

      Eigen::Vector2d image_point = Eigen::Vector2d::Zero();
      if (!camera.vsEuclideanToKeypoint(ray, &image_point)) {
        map_x.at<float>(y, x) = -1.0f;
        map_y.at<float>(y, x) = -1.0f;
        continue;
      }
      map_x.at<float>(y, x) = static_cast<float>(image_point.x());
      map_y.at<float>(y, x) = static_cast<float>(image_point.y());
    }
  }

  cv::remap(gray, *patch, map_x, map_y, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar(127));
  return !patch->empty();
}

bool TryExpandedRoiRedetect(
    const cv::Mat& gray,
    int board_id,
    const std::array<cv::Point2f, 4>& predicted_corners,
    std::array<cv::Point2f, 4>* decoded_corners,
    cv::Rect* roi_bbox,
    int* decoded_tag_id,
    int* hamming,
    std::array<cv::Point2f, 4>* weak_quad_corners_out,
    bool* weak_quad_found_out,
    std::string* summary) {
  if (decoded_corners == nullptr || gray.empty()) {
    return false;
  }
  if (weak_quad_found_out != nullptr) {
    *weak_quad_found_out = false;
  }
  constexpr double kRoiScale = 1.5;
  const cv::Rect roi = BuildExpandedBoundingRoi(predicted_corners, gray.size(), kRoiScale);
  if (roi_bbox != nullptr) {
    *roi_bbox = roi;
  }
  if (roi.empty()) {
    if (summary != nullptr) {
      *summary = "roi_redetect_invalid_bbox";
    }
    return false;
  }

  cv::Mat crop = gray(roi).clone();
  AprilTags::TagDetector detector(AprilTags::tagCodes36h11, 2);
  const std::vector<AprilTags::TagDetection> detections = detector.extractTags(crop);
  const AprilTags::TagDetection* best = nullptr;
  double best_area = -1.0;
  const AprilTags::TagDetection* context_best = nullptr;
  double context_best_area = -1.0;
  double context_best_center_error_px = 0.0;
  double context_best_area_ratio = 0.0;
  std::string context_best_reject_reason;
  std::ostringstream details;
  details << (detections.empty() ? "roi_redetect_no_detections"
                                 : "roi_redetect_detections")
          << "=" << detections.size()
          << " bbox=" << roi.x << ":" << roi.y << ":" << roi.width << ":" << roi.height;
  for (const AprilTags::TagDetection& detection : detections) {
    const double area = QuadAreaFromTagDetection(detection);
    details << " id=" << detection.id
            << " ham=" << detection.hammingDistance
            << " good=" << (detection.good ? 1 : 0)
            << " area=" << std::lround(area);
    if (detection.id == board_id && detection.good && area > best_area) {
      best = &detection;
      best_area = area;
    }
    if (detection.good) {
      std::array<cv::Point2f, 4> full_corners{};
      for (int index = 0; index < 4; ++index) {
        full_corners[static_cast<std::size_t>(index)] =
            cv::Point2f(detection.p[index].first + static_cast<float>(roi.x),
                        detection.p[index].second + static_cast<float>(roi.y));
      }
      double center_error_px = 0.0;
      double area_ratio = 0.0;
      std::string reject_reason;
      if (QuadMatchesGeometryPriorSeed(full_corners, predicted_corners,
                                       &center_error_px, &area_ratio,
                                       &reject_reason)) {
        if (area > context_best_area) {
          context_best = &detection;
          context_best_area = area;
          context_best_center_error_px = center_error_px;
          context_best_area_ratio = area_ratio;
        }
      } else if (context_best_reject_reason.empty()) {
        context_best_reject_reason = reject_reason;
      }
    }
  }
  const bool used_context_override = best == nullptr && context_best != nullptr;
  if (best == nullptr && context_best != nullptr) {
    best = context_best;
    best_area = context_best_area;
  }
  if (best == nullptr) {
    Eigen::MatrixXd raw_quads;
    try {
      raw_quads = detector.extractQuads(crop);
    } catch (const std::exception& error) {
      if (summary != nullptr) {
        *summary = details.str() + " raw_quad_extract_failed=" + error.what();
      }
      return false;
    }

    bool raw_quad_match = false;
    std::array<cv::Point2f, 4> raw_quad_corners{};
    double raw_best_score = std::numeric_limits<double>::infinity();
    double raw_best_center_error_px = 0.0;
    double raw_best_area_ratio = 0.0;
    double raw_best_assignment_cost = 0.0;
    std::string raw_reject_reason;
    const int raw_quad_count =
        raw_quads.rows() == 2 ? static_cast<int>(raw_quads.cols() / 4) : 0;
    for (int quad_index = 0; quad_index < raw_quad_count; ++quad_index) {
      std::array<cv::Point2f, 4> raw_full{};
      for (int corner_index = 0; corner_index < 4; ++corner_index) {
        raw_full[static_cast<std::size_t>(corner_index)] = cv::Point2f(
            static_cast<float>(raw_quads(0, quad_index * 4 + corner_index)) +
                static_cast<float>(roi.x),
            static_cast<float>(raw_quads(1, quad_index * 4 + corner_index)) +
                static_cast<float>(roi.y));
      }
      double assignment_cost = 0.0;
      const std::array<cv::Point2f, 4> ordered =
          ReorderQuadToMatchSeed(raw_full, predicted_corners,
                                 &assignment_cost);
      double center_error_px = 0.0;
      double area_ratio = 0.0;
      std::string reject_reason;
      if (!QuadMatchesGeometryPriorSeed(ordered, predicted_corners,
                                        &center_error_px, &area_ratio,
                                        &reject_reason)) {
        if (raw_reject_reason.empty()) {
          raw_reject_reason = reject_reason;
        }
        continue;
      }
      const double seed_edge =
          std::max(1.0, MeanQuadEdgeLength(predicted_corners));
      const double score =
          center_error_px / seed_edge +
          0.25 * std::abs(std::log(std::max(1e-6, area_ratio))) +
          0.05 * assignment_cost / seed_edge;
      if (score < raw_best_score) {
        raw_best_score = score;
        raw_quad_match = true;
        raw_quad_corners = ordered;
        raw_best_center_error_px = center_error_px;
        raw_best_area_ratio = area_ratio;
        raw_best_assignment_cost = assignment_cost;
      }
    }
	    if (raw_quad_match) {
	      *decoded_corners = raw_quad_corners;
	      if (decoded_tag_id != nullptr) {
	        *decoded_tag_id = -2;
      }
      if (hamming != nullptr) {
        *hamming = -1;
      }
      if (summary != nullptr) {
        std::ostringstream raw_summary;
        raw_summary << details.str()
                    << " raw_quad_context_override inferred_id=" << board_id
                    << " raw_quad_count=" << raw_quad_count
                    << " center_error_px=" << raw_best_center_error_px
                    << " area_ratio=" << raw_best_area_ratio
                    << " assignment_cost_px=" << raw_best_assignment_cost
                    << " score=" << raw_best_score;
        *summary = raw_summary.str();
	      }
	      return true;
	    }
	    std::array<cv::Point2f, 4> weak_quad_corners{};
	    std::string weak_quad_summary;
	    const bool collect_weak_quad =
	        weak_quad_corners_out != nullptr || weak_quad_found_out != nullptr;
	    const bool weak_quad_found =
	        collect_weak_quad &&
	        TryRelaxedRoiWeakQuadProposal(crop, roi, predicted_corners,
	                                      &weak_quad_corners,
	                                      &weak_quad_summary);
	    if (weak_quad_found) {
	      if (weak_quad_corners_out != nullptr) {
	        *weak_quad_corners_out = weak_quad_corners;
	      }
	      if (weak_quad_found_out != nullptr) {
	        *weak_quad_found_out = true;
	      }
	    }
	    if (summary != nullptr) {
	      *summary = details.str() + " no_matching_good_id"
	                 + (context_best_reject_reason.empty()
	                        ? ""
	                        : " context_reject=" + context_best_reject_reason)
	                 + " raw_quad_count=" + std::to_string(raw_quad_count)
	                 + (raw_reject_reason.empty()
	                        ? ""
	                        : " raw_context_reject=" + raw_reject_reason)
	                 + (weak_quad_found
	                        ? " weak_quad_diagnostic_found"
	                        : " weak_quad_diagnostic_not_found")
	                 + weak_quad_summary;
	    }
	    return false;
	  }

  for (int index = 0; index < 4; ++index) {
    (*decoded_corners)[static_cast<std::size_t>(index)] =
        cv::Point2f(best->p[index].first + static_cast<float>(roi.x),
                    best->p[index].second + static_cast<float>(roi.y));
  }
  if (decoded_tag_id != nullptr) {
    *decoded_tag_id = best->id;
  }
  if (hamming != nullptr) {
    *hamming = best->hammingDistance;
  }
  if (summary != nullptr) {
    std::ostringstream success;
    success << details.str()
            << (used_context_override ? " context_id_override decoded_id="
                                      : " matched_id=")
            << best->id;
    if (used_context_override) {
      success << " inferred_id=" << board_id
              << " center_error_px=" << context_best_center_error_px
              << " area_ratio=" << context_best_area_ratio;
    }
    *summary = success.str();
  }
	  return true;
	}

bool TryRelaxedRoiWeakQuadAlternative(
    const cv::Mat& gray,
    const std::array<cv::Point2f, 4>& predicted_corners,
    std::array<cv::Point2f, 4>* weak_quad_corners,
    cv::Rect* roi_bbox,
    std::string* summary) {
  if (weak_quad_corners == nullptr || gray.empty()) {
    return false;
  }
  constexpr double kRoiScale = 1.5;
  const cv::Rect roi =
      BuildExpandedBoundingRoi(predicted_corners, gray.size(), kRoiScale);
  if (roi_bbox != nullptr) {
    *roi_bbox = roi;
  }
  if (roi.empty()) {
    if (summary != nullptr) {
      *summary = "weak_quad_alternative_invalid_bbox";
    }
    return false;
  }
  const cv::Mat crop = gray(roi).clone();
  std::string weak_summary;
  const bool found =
      TryRelaxedRoiWeakQuadProposal(crop, roi, predicted_corners,
                                    weak_quad_corners, &weak_summary);
  if (summary != nullptr) {
    std::ostringstream stream;
    stream << "weak_quad_alternative bbox=" << roi.x << ":" << roi.y << ":"
           << roi.width << ":" << roi.height
           << (found ? " found" : " not_found") << weak_summary;
    *summary = stream.str();
  }
  return found;
}

bool TryDsRayPatchDecode(
    const cv::Mat& gray,
    const IntermediateCameraConfig& camera_config,
    int board_id,
    const std::array<cv::Point2f, 4>& seed_corners,
    std::array<cv::Point2f, 4>* decoded_corners,
    int* decoded_tag_id,
    int* hamming,
    std::string* summary) {
  if (decoded_corners == nullptr || gray.empty()) {
    return false;
  }
  constexpr int kPatchSize = 720;
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(camera_config);
  if (!camera.IsValid()) {
    if (summary != nullptr) {
      *summary = "ds_ray_patch_invalid_camera";
    }
    return false;
  }

  const cv::Point2f center = QuadCenter(seed_corners);
  if (!IsInsideImage(center, gray.size(), 1.0)) {
    if (summary != nullptr) {
      *summary = "ds_ray_patch_center_outside_image";
    }
    return false;
  }

  Eigen::Vector3d center_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangent_x = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangent_y = Eigen::Vector3d::Zero();
  if (!BuildLocalDsPatchFrame(camera, center, &center_ray, &tangent_x,
                              &tangent_y)) {
    if (summary != nullptr) {
      *summary = "ds_ray_patch_frame_failed";
    }
    return false;
  }

  double patch_focal = 0.0;
  double half_extent = 0.0;
  if (!EstimateDsPatchFocalFromSeedCorners(camera, seed_corners, center_ray,
                                           tangent_x, tangent_y, kPatchSize,
                                           &patch_focal, &half_extent)) {
    if (summary != nullptr) {
      *summary = "ds_ray_patch_focal_estimation_failed";
    }
    return false;
  }

  cv::Mat patch;
  if (!BuildDsRaySpacePatch(gray, camera, center_ray, tangent_x, tangent_y,
                            patch_focal, kPatchSize, &patch)) {
    if (summary != nullptr) {
      *summary = "ds_ray_patch_remap_failed";
    }
    return false;
  }

  AprilTags::TagDetector detector(AprilTags::tagCodes36h11, 2);
  const std::vector<AprilTags::TagDetection> detections =
      detector.extractTags(patch);
  if (detections.empty()) {
    if (summary != nullptr) {
      std::ostringstream stream;
      stream << "ds_ray_patch_no_detections"
             << " patch_size=" << kPatchSize
             << " patch_focal=" << patch_focal
             << " half_extent=" << half_extent;
      *summary = stream.str();
    }
    return false;
  }

  std::ostringstream details;
  details << "ds_ray_patch_detections=" << detections.size()
          << " patch_size=" << kPatchSize
          << " patch_focal=" << patch_focal
          << " half_extent=" << half_extent;
  bool have_context_override = false;
  int context_decoded_id = -1;
  int context_hamming = -1;
  double context_center_error_px = 0.0;
  double context_area_ratio = 0.0;
  double context_area = -1.0;
  std::array<cv::Point2f, 4> context_image_corners{};
  std::string context_reject_reason;
  for (const AprilTags::TagDetection& detection : detections) {
    details << " id=" << detection.id
            << " ham=" << detection.hammingDistance
            << " good=" << (detection.good ? 1 : 0);
    if (!detection.good) {
      continue;
    }
    std::array<cv::Point2f, 4> image_corners{};
    bool mapped_all_corners = true;
    for (int index = 0; index < 4; ++index) {
      Eigen::Vector2d image_point = Eigen::Vector2d::Zero();
      if (!MapDsPatchPointToImage(
              camera, center_ray, tangent_x, tangent_y, patch_focal,
              kPatchSize,
              cv::Point2f(detection.p[index].first, detection.p[index].second),
              &image_point)) {
        mapped_all_corners = false;
        break;
      }
      const cv::Point2f mapped(static_cast<float>(image_point.x()),
                               static_cast<float>(image_point.y()));
      if (!IsInsideImage(mapped, gray.size(), 1.0)) {
        mapped_all_corners = false;
        break;
      }
      image_corners[static_cast<std::size_t>(index)] = mapped;
    }
    if (!mapped_all_corners) {
      continue;
    }
    double center_error_px = 0.0;
    double area_ratio = 0.0;
    std::string reject_reason;
    const bool context_match = QuadMatchesGeometryPriorSeed(
        image_corners, seed_corners, &center_error_px, &area_ratio,
        &reject_reason);
    if (!context_match) {
      if (context_reject_reason.empty()) {
        context_reject_reason = reject_reason;
      }
      continue;
    }
    if (detection.id != board_id) {
      const double area = PolygonArea(image_corners);
      if (area > context_area) {
        have_context_override = true;
        context_decoded_id = detection.id;
        context_hamming = detection.hammingDistance;
        context_center_error_px = center_error_px;
        context_area_ratio = area_ratio;
        context_area = area;
        context_image_corners = image_corners;
      }
      continue;
    }
    for (int index = 0; index < 4; ++index) {
      (*decoded_corners)[static_cast<std::size_t>(index)] =
          image_corners[static_cast<std::size_t>(index)];
    }
    if (decoded_tag_id != nullptr) {
      *decoded_tag_id = detection.id;
    }
    if (hamming != nullptr) {
      *hamming = detection.hammingDistance;
    }
    if (summary != nullptr) {
      std::ostringstream success;
      success << details.str() << " matched_id=" << detection.id;
      *summary = success.str();
    }
    return true;
  }
  if (have_context_override) {
    for (int index = 0; index < 4; ++index) {
      (*decoded_corners)[static_cast<std::size_t>(index)] =
          context_image_corners[static_cast<std::size_t>(index)];
    }
    if (decoded_tag_id != nullptr) {
      *decoded_tag_id = context_decoded_id;
    }
    if (hamming != nullptr) {
      *hamming = context_hamming;
    }
    if (summary != nullptr) {
      std::ostringstream success;
      success << details.str()
              << " context_id_override decoded_id=" << context_decoded_id
              << " inferred_id=" << board_id
              << " center_error_px=" << context_center_error_px
              << " area_ratio=" << context_area_ratio;
      *summary = success.str();
    }
    return true;
  }
  if (summary != nullptr) {
    *summary = details.str() + " no_matching_good_id"
               + (context_reject_reason.empty()
                      ? ""
                      : " context_reject=" + context_reject_reason);
  }
  return false;
}

double SampleFloatAt(const cv::Mat& image, const cv::Point2f& point) {
  const int x = std::max(0, std::min(image.cols - 1,
                                     static_cast<int>(std::lround(point.x))));
  const int y = std::max(0, std::min(image.rows - 1,
                                     static_cast<int>(std::lround(point.y))));
  return static_cast<double>(image.at<float>(y, x));
}

struct EdgeEvidenceMetrics {
  double support_ratio = 0.0;
  double mean_gradient_ratio = 0.0;
};

EdgeEvidenceMetrics EvaluateQuadEdgeEvidence(
    const cv::Mat& gray,
    const std::array<cv::Point2f, 4>& corners,
    int sample_count,
    int search_half_width,
    double min_gradient_ratio) {
  EdgeEvidenceMetrics metrics;
  if (gray.empty() || sample_count <= 0 || search_half_width <= 0) {
    return metrics;
  }

  cv::Mat grad_x;
  cv::Mat grad_y;
  cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
  cv::Mat grad_mag;
  cv::magnitude(grad_x, grad_y, grad_mag);
  double grad_min = 0.0;
  double grad_max = 0.0;
  cv::minMaxLoc(grad_mag, &grad_min, &grad_max);
  const double grad_norm = std::max(1e-12, grad_max);

  int total_samples = 0;
  int support_samples = 0;
  double gradient_ratio_sum = 0.0;
  for (int edge_index = 0; edge_index < 4; ++edge_index) {
    const cv::Point2f a = corners[static_cast<std::size_t>(edge_index)];
    const cv::Point2f b = corners[static_cast<std::size_t>((edge_index + 1) % 4)];
    const cv::Point2f edge = b - a;
    const double edge_len = std::max(1e-6, PointDistance(a, b));
    const cv::Point2f normal(-edge.y / static_cast<float>(edge_len),
                             edge.x / static_cast<float>(edge_len));
    const int per_edge_samples = std::max(8, sample_count / 4);
    for (int sample = 0; sample < per_edge_samples; ++sample) {
      const double t = (static_cast<double>(sample) + 0.5) /
                       static_cast<double>(per_edge_samples);
      const cv::Point2f center = a + static_cast<float>(t) * edge;
      double best_ratio = 0.0;
      for (int offset = -search_half_width; offset <= search_half_width; ++offset) {
        const cv::Point2f probe =
            center + static_cast<float>(offset) * normal;
        if (!IsInsideImage(probe, gray.size(), 1.0)) {
          continue;
        }
        best_ratio = std::max(best_ratio, SampleFloatAt(grad_mag, probe) / grad_norm);
      }
      ++total_samples;
      gradient_ratio_sum += best_ratio;
      if (best_ratio >= min_gradient_ratio) {
        ++support_samples;
      }
    }
  }
  if (total_samples > 0) {
    metrics.support_ratio =
        static_cast<double>(support_samples) / static_cast<double>(total_samples);
    metrics.mean_gradient_ratio =
        gradient_ratio_sum / static_cast<double>(total_samples);
  }
  return metrics;
}

double RotationErrorDegrees(const Eigen::Matrix3d& lhs,
                            const Eigen::Matrix3d& rhs) {
  const Eigen::Matrix3d delta = lhs.transpose() * rhs;
  Eigen::AngleAxisd angle_axis(delta);
  return std::abs(angle_axis.angle()) * 180.0 / std::acos(-1.0);
}

std::array<Eigen::Vector3d, 4> BuildOuterCornerPointsForBoard(
    const ApriltagInternalConfig& config,
    int board_id) {
  ApriltagInternalConfig board_config = config;
  board_config.tag_id = board_id;
  const ApriltagCanonicalModel model(board_config);
  return BuildOuterCornerPoints(model);
}

std::vector<Eigen::Vector3d> ToVector(
    const std::array<Eigen::Vector3d, 4>& values) {
  return std::vector<Eigen::Vector3d>(values.begin(), values.end());
}

std::vector<cv::Point2f> ToVector(
    const std::array<cv::Point2f, 4>& values) {
  return std::vector<cv::Point2f>(values.begin(), values.end());
}

std::vector<cv::Point2f> ToImagePoints(
    const std::array<Eigen::Vector2d, 4>& values) {
  std::vector<cv::Point2f> points;
  points.reserve(values.size());
  for (const Eigen::Vector2d& value : values) {
    points.emplace_back(static_cast<float>(value.x()),
                        static_cast<float>(value.y()));
  }
  return points;
}

double MedianFinite(std::vector<double> values) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](double value) {
                                return !std::isfinite(value);
                              }),
               values.end());
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1] + values[mid]);
}

double ComputeFrameNormalOuterRefitRmseMedian(
    const ApriltagInternalConfig& config,
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const InternalRegenerationFrameInput& frame_input) {
  std::vector<double> rmses;
  for (const OuterBoardMeasurement& measurement :
       frame_input.outer_detections.frame_measurements.board_measurements) {
    const bool exact_image_validated_rescue =
        IsExactImageValidatedOuterMeasurement(measurement);
    if (!measurement.success ||
        (measurement.used_local_patch_rescue &&
         !exact_image_validated_rescue) ||
        measurement.valid_refined_corner_count < 4) {
      continue;
    }
    bool all_corners_valid = true;
    for (bool valid : measurement.refined_corner_valid) {
      all_corners_valid = all_corners_valid && valid;
    }
    if (!all_corners_valid) {
      continue;
    }
    const std::array<Eigen::Vector3d, 4> object_points_array =
        BuildOuterCornerPointsForBoard(config, measurement.board_id);
    Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
    double outer_rmse = 0.0;
    if (EstimatePoseFromObjectPoints(
            intrinsics, ToVector(object_points_array),
            ToImagePoints(measurement.refined_outer_corners_original_image),
            &T_camera_board, &outer_rmse) &&
        std::isfinite(outer_rmse)) {
      rmses.push_back(outer_rmse);
    }
  }
  return MedianFinite(rmses);
}

LocalFramePoseRefit EstimateLocalFramePoseFromVisibleBoards(
    const ApriltagInternalConfig& config,
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const InternalRegenerationFrameInput& frame_input,
    const std::function<bool(int, Eigen::Matrix4d*)>& lookup_reference_board) {
  struct BoardPoseSupport {
    int board_id = -1;
    double local_outer_rmse = std::numeric_limits<double>::infinity();
    Eigen::Isometry3d T_camera_reference = Eigen::Isometry3d::Identity();
    std::vector<Eigen::Vector3d> object_points_reference;
    std::vector<cv::Point2f> image_points;
  };

  LocalFramePoseRefit best;
  std::vector<BoardPoseSupport> supports;
  for (const OuterBoardMeasurement& measurement :
       frame_input.outer_detections.frame_measurements.board_measurements) {
    const bool exact_image_validated_rescue =
        IsExactImageValidatedOuterMeasurement(measurement);
    if (!measurement.success ||
        (measurement.used_local_patch_rescue &&
         !exact_image_validated_rescue) ||
        measurement.valid_refined_corner_count < 4) {
      continue;
    }
    bool all_corners_valid = true;
    for (bool valid : measurement.refined_corner_valid) {
      all_corners_valid = all_corners_valid && valid;
    }
    if (!all_corners_valid) {
      continue;
    }

    Eigen::Matrix4d T_reference_board_matrix = Eigen::Matrix4d::Identity();
    if (!lookup_reference_board(measurement.board_id,
                                &T_reference_board_matrix)) {
      continue;
    }

    const std::array<Eigen::Vector3d, 4> object_points_array =
        BuildOuterCornerPointsForBoard(config, measurement.board_id);
    Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
    double outer_rmse = 0.0;
    if (!EstimatePoseFromObjectPoints(
            intrinsics, ToVector(object_points_array),
            ToImagePoints(measurement.refined_outer_corners_original_image),
            &T_camera_board, &outer_rmse)) {
      continue;
    }

    const Eigen::Isometry3d T_reference_board =
        ToIsometry3d(T_reference_board_matrix);
    const Eigen::Isometry3d T_camera_reference =
        T_camera_board * T_reference_board.inverse();
    LocalFramePoseRefit::Candidate pose_candidate;
    pose_candidate.source_board_id = measurement.board_id;
    pose_candidate.outer_rmse = outer_rmse;
    pose_candidate.T_camera_reference = T_camera_reference.matrix();
    best.candidates.push_back(pose_candidate);
    BoardPoseSupport support;
    support.board_id = measurement.board_id;
    support.local_outer_rmse = outer_rmse;
    support.T_camera_reference = T_camera_reference;
    support.image_points =
        ToImagePoints(measurement.refined_outer_corners_original_image);
    support.object_points_reference.reserve(object_points_array.size());
    for (const Eigen::Vector3d& object_point : object_points_array) {
      support.object_points_reference.push_back(
          T_reference_board * object_point);
    }
    supports.push_back(std::move(support));
    if (!best.success || outer_rmse < best.outer_rmse) {
      best.success = true;
      best.source_board_id = measurement.board_id;
      best.support_board_ids = {measurement.board_id};
      best.outer_rmse = outer_rmse;
      best.T_camera_reference = T_camera_reference.matrix();
    }
  }

  std::vector<const BoardPoseSupport*> locally_valid_supports;
  for (const BoardPoseSupport& support : supports) {
    if (std::isfinite(support.local_outer_rmse) &&
        support.local_outer_rmse < 3.0) {
      locally_valid_supports.push_back(&support);
    }
  }
  if (locally_valid_supports.size() < 2) {
    return best;
  }

  const auto fit_supports =
      [&](const std::vector<const BoardPoseSupport*>& fit_set,
          Eigen::Isometry3d* pose, double* rmse) {
        std::vector<Eigen::Vector3d> object_points;
        std::vector<cv::Point2f> image_points;
        object_points.reserve(fit_set.size() * 4);
        image_points.reserve(fit_set.size() * 4);
        for (const BoardPoseSupport* support : fit_set) {
          object_points.insert(object_points.end(),
                               support->object_points_reference.begin(),
                               support->object_points_reference.end());
          image_points.insert(image_points.end(), support->image_points.begin(),
                              support->image_points.end());
        }
        return EstimatePoseFromObjectPoints(intrinsics, object_points,
                                            image_points, pose, rmse);
      };

  IntermediateCameraConfig camera_config;
  camera_config.camera_model = intrinsics.camera_model;
  camera_config.distortion_model = intrinsics.distortion_model;
  camera_config.intrinsics = intrinsics.IntrinsicsVector();
  camera_config.distortion_coeffs = intrinsics.DistortionVector();
  camera_config.resolution = {intrinsics.resolution.width,
                              intrinsics.resolution.height};
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(camera_config);
  if (!camera.IsValid()) {
    return best;
  }

  const auto board_rmse =
      [&](const Eigen::Isometry3d& pose, const BoardPoseSupport& support) {
        double squared_error_sum = 0.0;
        int valid_count = 0;
        for (std::size_t index = 0;
             index < support.object_points_reference.size(); ++index) {
          Eigen::Vector2d projected;
          if (!camera.vsEuclideanToKeypoint(
                  pose * support.object_points_reference[index], &projected)) {
            continue;
          }
          const cv::Point2f& observed = support.image_points[index];
          squared_error_sum +=
              (projected - Eigen::Vector2d(observed.x, observed.y)).squaredNorm();
          ++valid_count;
        }
        return valid_count > 0
                   ? std::sqrt(squared_error_sum /
                               static_cast<double>(valid_count))
                   : std::numeric_limits<double>::infinity();
      };

  struct PoseConsensusCandidate {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double fit_rmse = std::numeric_limits<double>::infinity();
    double median_board_rmse = std::numeric_limits<double>::infinity();
  };
  std::vector<PoseConsensusCandidate> pose_candidates;
  const auto append_pose_candidate =
      [&](const Eigen::Isometry3d& pose, double fit_rmse) {
        std::vector<double> board_rmses;
        board_rmses.reserve(locally_valid_supports.size());
        for (const BoardPoseSupport* support : locally_valid_supports) {
          board_rmses.push_back(board_rmse(pose, *support));
        }
        PoseConsensusCandidate candidate;
        candidate.pose = pose;
        candidate.fit_rmse = fit_rmse;
        candidate.median_board_rmse = MedianFinite(board_rmses);
        if (std::isfinite(candidate.median_board_rmse)) {
          pose_candidates.push_back(candidate);
        }
      };

  for (const BoardPoseSupport* support : locally_valid_supports) {
    append_pose_candidate(support->T_camera_reference,
                          support->local_outer_rmse);
  }
  Eigen::Isometry3d all_pose = Eigen::Isometry3d::Identity();
  double all_rmse = 0.0;
  if (fit_supports(locally_valid_supports, &all_pose, &all_rmse)) {
    append_pose_candidate(all_pose, all_rmse);
  }
  if (locally_valid_supports.size() >= 3) {
    for (std::size_t excluded = 0; excluded < locally_valid_supports.size();
         ++excluded) {
      std::vector<const BoardPoseSupport*> subset;
      subset.reserve(locally_valid_supports.size() - 1);
      for (std::size_t index = 0; index < locally_valid_supports.size();
           ++index) {
        if (index != excluded) {
          subset.push_back(locally_valid_supports[index]);
        }
      }
      Eigen::Isometry3d subset_pose = Eigen::Isometry3d::Identity();
      double subset_rmse = 0.0;
      if (fit_supports(subset, &subset_pose, &subset_rmse)) {
        append_pose_candidate(subset_pose, subset_rmse);
      }
    }
  }
  if (pose_candidates.empty()) {
    return best;
  }
  const PoseConsensusCandidate* consensus = &pose_candidates.front();
  for (const PoseConsensusCandidate& candidate : pose_candidates) {
    if (candidate.median_board_rmse < consensus->median_board_rmse) {
      consensus = &candidate;
    }
  }

  std::vector<double> consensus_board_rmses;
  consensus_board_rmses.reserve(locally_valid_supports.size());
  for (const BoardPoseSupport* support : locally_valid_supports) {
    consensus_board_rmses.push_back(board_rmse(consensus->pose, *support));
  }
  const double median_rmse = MedianFinite(consensus_board_rmses);
  std::vector<double> absolute_deviations;
  absolute_deviations.reserve(consensus_board_rmses.size());
  for (double value : consensus_board_rmses) {
    absolute_deviations.push_back(std::abs(value - median_rmse));
  }
  const double mad = MedianFinite(absolute_deviations);
  const double inlier_threshold =
      std::max(3.0, median_rmse + std::max(0.75, 3.0 * 1.4826 * mad));
  std::vector<const BoardPoseSupport*> inlier_supports;
  for (std::size_t index = 0; index < locally_valid_supports.size(); ++index) {
    if (consensus_board_rmses[index] <= inlier_threshold) {
      inlier_supports.push_back(locally_valid_supports[index]);
    }
  }
  if (inlier_supports.size() < 2) {
    return best;
  }

  Eigen::Isometry3d joint_pose = consensus->pose;
  double joint_rmse = consensus->fit_rmse;
  Eigen::Isometry3d refit_pose = Eigen::Isometry3d::Identity();
  double refit_rmse = 0.0;
  if (fit_supports(inlier_supports, &refit_pose, &refit_rmse)) {
    joint_pose = refit_pose;
    joint_rmse = refit_rmse;
  }
  best.success = true;
  best.T_camera_reference = joint_pose.matrix();
  best.outer_rmse = joint_rmse;
  best.support_board_ids.clear();
  for (const BoardPoseSupport* support : inlier_supports) {
    best.support_board_ids.push_back(support->board_id);
  }
  return best;
}

OuterTagDetectionResult BuildRescuedOuterDetection(
    int board_id,
    const cv::Size& image_size,
    const std::array<cv::Point2f, 4>& refined_corners,
    bool tag_id_validated,
    double quality,
    const std::string& summary) {
  OuterTagDetectionResult detection;
  detection.success = true;
  detection.board_id = board_id;
  detection.detected_tag_id = tag_id_validated ? board_id : -1;
  detection.good = true;
  detection.hamming = tag_id_validated ? 0 : -1;
  detection.original_longest_side = std::max(image_size.width, image_size.height);
  detection.chosen_scale_longest_side = detection.original_longest_side;
  detection.chosen_scale_factor = 1.0;
  detection.scale_configuration_mode = "geometry_guided_image_evidence";
  detection.used_local_patch_rescue = true;
  detection.attempted_local_patch_rescue = true;
  detection.local_patch_rescue_summary = summary;
  detection.quality = quality;
  detection.failure_reason = OuterTagFailureReason::None;
  detection.failure_reason_text = "None";
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f& corner = refined_corners[static_cast<std::size_t>(index)];
    const Eigen::Vector2d eigen_corner(corner.x, corner.y);
    detection.coarse_corners_scaled_image[static_cast<std::size_t>(index)] =
        eigen_corner;
    detection.coarse_corners_original_image[static_cast<std::size_t>(index)] =
        eigen_corner;
    detection.refined_corners_original_image[static_cast<std::size_t>(index)] =
        eigen_corner;
    detection.refined_valid[static_cast<std::size_t>(index)] = true;
  }
  return detection;
}

struct GeometryGuidedTagLikelihood {
  bool checked = false;
  bool passed = false;
  int expected_hamming = -1;
  int runner_up_id = -1;
  int runner_up_hamming = -1;
  int hamming_margin = -1;
  double contrast = 0.0;
  std::string summary;
};

int MinimumRotationHamming(unsigned long long observed_code,
                           unsigned long long candidate_code) {
  int best = std::numeric_limits<int>::max();
  unsigned long long rotated = observed_code;
  for (int rotation = 0; rotation < 4; ++rotation) {
    best = std::min(
        best, AprilTags::TagFamily::hammingDistance(rotated, candidate_code));
    rotated = AprilTags::TagFamily::rotate90(rotated, 6);
  }
  return best;
}

bool BuildModelAwareCanonicalTagPatch(
    const cv::Mat& gray,
    const DoubleSphereCameraModel& camera,
    const ApriltagCanonicalModel& model,
    const Eigen::Isometry3d& T_camera_board,
    int pixels_per_module,
    cv::Mat* patch,
    double* valid_ratio) {
  if (patch == nullptr || gray.empty() || !camera.IsValid() ||
      pixels_per_module < 4) {
    return false;
  }
  const int module_dimension = model.ModuleDimension();
  const int patch_size = module_dimension * pixels_per_module;
  if (patch_size <= 0) {
    return false;
  }
  cv::Mat map_x(patch_size, patch_size, CV_32F, cv::Scalar(-1.0f));
  cv::Mat map_y(patch_size, patch_size, CV_32F, cv::Scalar(-1.0f));
  int valid_count = 0;
  const double tag_size = model.config().tag_size;
  for (int row = 0; row < patch_size; ++row) {
    // Canonical image rows run top-to-bottom, while board y runs bottom-to-top.
    const double board_y =
        tag_size * (1.0 - (static_cast<double>(row) + 0.5) / patch_size);
    for (int col = 0; col < patch_size; ++col) {
      const double board_x =
          tag_size * (static_cast<double>(col) + 0.5) / patch_size;
      const Eigen::Vector3d point_camera =
          T_camera_board * Eigen::Vector3d(board_x, board_y, 0.0);
      Eigen::Vector2d image_point = Eigen::Vector2d::Zero();
      if (!camera.vsEuclideanToKeypoint(point_camera, &image_point) ||
          image_point.x() < 0.0 || image_point.y() < 0.0 ||
          image_point.x() >= gray.cols - 1.0 || image_point.y() >= gray.rows - 1.0) {
        continue;
      }
      map_x.at<float>(row, col) = static_cast<float>(image_point.x());
      map_y.at<float>(row, col) = static_cast<float>(image_point.y());
      ++valid_count;
    }
  }
  if (valid_ratio != nullptr) {
    *valid_ratio = static_cast<double>(valid_count) /
                   static_cast<double>(patch_size * patch_size);
  }
  if (valid_count < static_cast<int>(0.98 * patch_size * patch_size)) {
    return false;
  }
  cv::remap(gray, *patch, map_x, map_y, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar(127));
  return !patch->empty();
}

double MeanCanonicalModule(const cv::Mat& patch,
                           int module_x,
                           int module_y,
                           int module_dimension,
                           int pixels_per_module) {
  const int x0 = module_x * pixels_per_module + pixels_per_module / 4;
  const int y0 = (module_dimension - 1 - module_y) * pixels_per_module +
                 pixels_per_module / 4;
  const int side = std::max(1, pixels_per_module / 2);
  const cv::Rect roi(x0, y0, side, side);
  return cv::mean(patch(roi))[0];
}

GeometryGuidedTagLikelihood EvaluateGeometryGuidedTagLikelihood(
    const cv::Mat& gray,
    const ApriltagInternalConfig& config,
    const ApriltagInternalDetectionOptions& options,
    const DoubleSphereCameraModel& camera,
    int board_id,
    const Eigen::Isometry3d& T_camera_board,
    bool single_anchor) {
  GeometryGuidedTagLikelihood result;
  result.checked = true;
  if (board_id < 0 ||
      board_id >= static_cast<int>(AprilTags::tagCodes36h11.codes.size())) {
    result.summary = "invalid_expected_tag_id";
    return result;
  }

  ApriltagInternalConfig board_config = config;
  board_config.tag_id = board_id;
  const ApriltagCanonicalModel model(board_config);
  constexpr int kPixelsPerModule = 32;
  cv::Mat patch;
  double valid_ratio = 0.0;
  if (!BuildModelAwareCanonicalTagPatch(gray, camera, model, T_camera_board,
                                         kPixelsPerModule, &patch,
                                         &valid_ratio)) {
    std::ostringstream stream;
    stream << "canonical_patch_remap_failed valid_ratio=" << valid_ratio;
    result.summary = stream.str();
    return result;
  }

  const int code_dimension = ApriltagCanonicalModel::kCodeDimension;
  const int border = config.black_border_bits;
  const int module_dimension = model.ModuleDimension();
  std::vector<double> code_values;
  code_values.reserve(code_dimension * code_dimension);
  for (int y = 0; y < code_dimension; ++y) {
    for (int x = 0; x < code_dimension; ++x) {
      code_values.push_back(MeanCanonicalModule(
          patch, border + x, border + y, module_dimension, kPixelsPerModule));
    }
  }
  std::sort(code_values.begin(), code_values.end());
  const double low = code_values[static_cast<std::size_t>(code_values.size() / 5)];
  const double high = code_values[static_cast<std::size_t>(
      code_values.size() - 1 - code_values.size() / 5)];
  // Normalize code contrast in the rectified Tag patch.  Using the full image
  // range makes unrelated saturated lights or dark borders suppress a valid
  // Tag's local contrast, which is exactly what happens in frame 70.
  double patch_min = 0.0;
  double patch_max = 0.0;
  cv::minMaxLoc(patch, &patch_min, &patch_max);
  const double local_dynamic_range = patch_max - patch_min;
  result.contrast =
      (high - low) / std::max(1.0, local_dynamic_range);
  const double threshold = 0.5 * (low + high);

  unsigned long long observed_code = 0ULL;
  for (int y = code_dimension - 1; y >= 0; --y) {
    for (int x = 0; x < code_dimension; ++x) {
      const double value = MeanCanonicalModule(
          patch, border + x, border + y, module_dimension, kPixelsPerModule);
      observed_code <<= 1;
      if (value > threshold) {
        observed_code |= 1ULL;
      }
    }
  }

  const auto& codes = AprilTags::tagCodes36h11.codes;
  result.expected_hamming = MinimumRotationHamming(
      observed_code, codes[static_cast<std::size_t>(board_id)]);
  for (std::size_t id = 0; id < codes.size(); ++id) {
    if (static_cast<int>(id) == board_id) {
      continue;
    }
    const int hamming = MinimumRotationHamming(observed_code, codes[id]);
    if (hamming < result.runner_up_hamming || result.runner_up_id < 0) {
      result.runner_up_hamming = hamming;
      result.runner_up_id = static_cast<int>(id);
    }
  }
  result.hamming_margin = result.runner_up_hamming - result.expected_hamming;
  const int max_expected_hamming = single_anchor
                                       ? options.geometry_guided_tag_likelihood_single_anchor_max_expected_hamming
                                       : options.geometry_guided_tag_likelihood_max_expected_hamming;
  const int min_hamming_margin = single_anchor
                                      ? options.geometry_guided_tag_likelihood_single_anchor_min_hamming_margin
                                      : options.geometry_guided_tag_likelihood_min_hamming_margin;
  const double min_contrast = single_anchor
                                  ? options.geometry_guided_tag_likelihood_single_anchor_min_contrast
                                  : options.geometry_guided_tag_likelihood_min_contrast;
  result.passed = result.expected_hamming <= max_expected_hamming &&
                  result.hamming_margin >= min_hamming_margin &&
                  result.contrast >= min_contrast;
  std::ostringstream stream;
  stream << "model_aware_canonical_patch"
         << " valid_ratio=" << valid_ratio
         << " expected_id=" << board_id
         << " expected_hamming=" << result.expected_hamming
         << " runner_up_id=" << result.runner_up_id
         << " runner_up_hamming=" << result.runner_up_hamming
         << " hamming_margin=" << result.hamming_margin
         << " contrast=" << result.contrast
         << " mode=" << (single_anchor ? "single_anchor" : "multi_board")
         << " max_expected_hamming=" << max_expected_hamming
         << " min_hamming_margin=" << min_hamming_margin
         << " min_contrast=" << min_contrast
         << " contrast_range=canonical_patch[" << patch_min << ","
         << patch_max << "]"
         << " threshold=" << threshold
         << " passed=" << (result.passed ? 1 : 0);
  result.summary = stream.str();
  return result;
}

GeometryPriorOuterSeedCandidate FinalizeGeometryPriorOuterSeedCandidate(
    const cv::Mat& gray,
    const ApriltagInternalConfig& config,
    const ApriltagInternalDetectionOptions& options,
    const IntermediateCameraConfig& camera_config,
    int board_id,
    const std::array<cv::Point2f, 4>& initial_corners_input,
    const Eigen::Matrix4d& T_camera_board_matrix,
    bool tag_id_validated,
    bool single_anchor_is_direct_exact,
    const std::string& validation_source,
    GeometryPriorOuterSeedCandidate candidate,
    OuterTagDetectionResult* rescued_detection) {
  std::array<cv::Point2f, 4> initial_corners = initial_corners_input;
  candidate.tag_id_validated = tag_id_validated;
  const QuadTopologyCheck initial_topology =
      CheckQuadTopology(initial_corners);
  if (!initial_topology.valid) {
    candidate.quad_topology_summary =
        "initial_invalid:" + initial_topology.summary;
    candidate.reject_reason =
        validation_source + "_image_evidence_failed_initial_quad_topology";
    return candidate;
  }
  // A ray-patch decode may replace an in-bounds geometric prediction with a
  // mapped quadrilateral that reaches outside the source image. OpenCV's
  // cornerSubPix asserts instead of reporting this condition, so reject this
  // candidate before any refinement. This is an evidence failure, never a
  // reason to abort the whole frontend on an extreme-FOV frame.
  const double subpix_border =
      static_cast<double>(std::max(0, candidate.subpix_window_radius)) + 2.0;
  for (const cv::Point2f& corner : initial_corners) {
    if (!IsInsideImage(corner, gray.size(), subpix_border)) {
      candidate.reject_reason =
          validation_source + "_image_evidence_failed_refined_corner_near_border";
      return candidate;
    }
  }
  const DoubleSphereCameraModel sphere_camera =
      DoubleSphereCameraModel::FromConfig(camera_config);
  if (options.geometry_prior_rescue_enable_spherical_refine &&
      sphere_camera.IsValid()) {
    candidate.spherical_refine_attempted = true;
    const OuterSphericalQuadRefinementResult spherical_refinement =
        RefineOuterCornersBySphericalPlanes(
            gray, sphere_camera, initial_corners, options.outer_detector_config);
    candidate.spherical_refine_success = spherical_refinement.success;
    candidate.spherical_refine_successful_corner_count =
        spherical_refinement.successful_corner_count;
    candidate.spherical_refine_max_displacement_px =
        spherical_refinement.max_displacement_px;
    candidate.spherical_refine_min_quality = spherical_refinement.min_quality;
    double min_support_count = std::numeric_limits<double>::infinity();
    double max_residual = 0.0;
    std::ostringstream failure_summary;
    for (int index = 0; index < 4; ++index) {
      const OuterSphericalCornerRefinementDebug& debug =
          spherical_refinement.corner_debug[static_cast<std::size_t>(index)];
      min_support_count = std::min(
          min_support_count,
          static_cast<double>(std::min(debug.prev_edge_support_count,
                                       debug.next_edge_support_count)));
      if (std::isfinite(debug.prev_edge_residual)) {
        max_residual = std::max(max_residual, debug.prev_edge_residual);
      }
      if (std::isfinite(debug.next_edge_residual)) {
        max_residual = std::max(max_residual, debug.next_edge_residual);
      }
      if (!debug.success) {
        if (failure_summary.tellp() > 0) {
          failure_summary << ";";
        }
        failure_summary << index << ":" << debug.failure_reason;
      }
    }
    candidate.spherical_refine_min_support_count =
        std::isfinite(min_support_count) ? min_support_count : 0.0;
    candidate.spherical_refine_max_residual = max_residual;
    candidate.spherical_refine_failure_summary = failure_summary.str();
    if (spherical_refinement.successful_corner_count > 0) {
      initial_corners = spherical_refinement.refined_corners;
    }
  } else {
    candidate.spherical_refine_failure_summary =
        options.geometry_prior_rescue_enable_spherical_refine
            ? "invalid_camera"
            : "disabled";
  }

  std::vector<cv::Point2f> refined_points = ToVector(initial_corners);
  cv::cornerSubPix(gray, refined_points,
                   cv::Size(candidate.subpix_window_radius,
                            candidate.subpix_window_radius),
                   cv::Size(-1, -1),
                   cv::TermCriteria(cv::TermCriteria::EPS +
                                        cv::TermCriteria::COUNT,
                                    30, 0.01));
  std::array<cv::Point2f, 4> refined_corners{};
  double max_displacement = 0.0;
  for (int index = 0; index < 4; ++index) {
    refined_corners[static_cast<std::size_t>(index)] =
        refined_points[static_cast<std::size_t>(index)];
    candidate.refined_corners[static_cast<std::size_t>(index)] =
        refined_corners[static_cast<std::size_t>(index)];
    max_displacement = std::max(
        max_displacement,
        PointDistance(refined_corners[static_cast<std::size_t>(index)],
                      candidate.predicted_corners[static_cast<std::size_t>(index)]));
  }
  candidate.max_corner_displacement_px = max_displacement;
  if (options.geometry_prior_rescue_max_corner_displacement_px > 0.0) {
    candidate.adaptive_max_corner_displacement_px =
        options.geometry_prior_rescue_max_corner_displacement_px;
  } else if (options.geometry_prior_rescue_max_corner_displacement_px == 0.0) {
    candidate.adaptive_max_corner_displacement_px =
        std::min(40.0, std::max(4.0, 0.08 * candidate.local_corner_scale_px));
  }
  if (std::isfinite(candidate.adaptive_max_corner_displacement_px) &&
      max_displacement > candidate.adaptive_max_corner_displacement_px) {
    candidate.local_corner_refine_success = false;
    candidate.reject_reason =
        validation_source + "_image_evidence_failed_corner_displacement";
    return candidate;
  }

  const QuadTopologyCheck predicted_topology =
      CheckQuadTopology(candidate.predicted_corners);
  const QuadTopologyCheck refined_topology = CheckQuadTopology(refined_corners);
  candidate.predicted_quad_topology_valid = predicted_topology.valid;
  candidate.predicted_signed_area_px = predicted_topology.signed_area_px;
  candidate.refined_quad_topology_valid = refined_topology.valid;
  candidate.refined_area_px = refined_topology.area_px;
  candidate.refined_signed_area_px = refined_topology.signed_area_px;
  candidate.refined_to_predicted_area_ratio =
      predicted_topology.area_px > 1e-9
          ? refined_topology.area_px / predicted_topology.area_px
          : std::numeric_limits<double>::quiet_NaN();
  const bool orientation_preserved =
      predicted_topology.signed_area_px * refined_topology.signed_area_px > 0.0;
  const bool area_preserved =
      std::isfinite(candidate.refined_to_predicted_area_ratio) &&
      candidate.refined_to_predicted_area_ratio >= 0.5 &&
      candidate.refined_to_predicted_area_ratio <= 2.0;
  candidate.quad_topology_preserved =
      predicted_topology.valid && refined_topology.valid &&
      orientation_preserved && area_preserved;
  std::ostringstream topology_summary;
  topology_summary << "predicted{" << predicted_topology.summary << "}"
                   << " refined{" << refined_topology.summary << "}"
                   << " orientation_preserved="
                   << (orientation_preserved ? 1 : 0)
                   << " area_ratio="
                   << candidate.refined_to_predicted_area_ratio;
  candidate.quad_topology_summary = topology_summary.str();
  if (!candidate.quad_topology_preserved) {
    candidate.local_corner_refine_success = false;
    candidate.reject_reason =
        validation_source + "_image_evidence_failed_quad_topology";
    return candidate;
  }

  cv::Mat corner_response;
  cv::cornerMinEigenVal(gray, corner_response, 3, 3);
  double response_min = 0.0;
  double response_max = 0.0;
  cv::minMaxLoc(corner_response, &response_min, &response_max);
  const double response_norm = std::max(1e-12, response_max);
  double min_response_ratio = 1.0;
  for (const cv::Point2f& point : refined_corners) {
    min_response_ratio = std::min(
        min_response_ratio, SampleFloatAt(corner_response, point) / response_norm);
  }
  candidate.min_corner_response_ratio = min_response_ratio;
  const bool corner_response_ok =
      min_response_ratio >=
      options.geometry_prior_rescue_min_corner_response_ratio;
  const EdgeEvidenceMetrics edge_metrics = EvaluateQuadEdgeEvidence(
      gray, refined_corners,
      options.geometry_prior_rescue_edge_sample_count,
      options.geometry_prior_rescue_edge_search_half_width_px,
      options.geometry_prior_rescue_min_edge_gradient_ratio);
  candidate.edge_support_ratio = edge_metrics.support_ratio;
  candidate.mean_edge_gradient_ratio = edge_metrics.mean_gradient_ratio;
  const bool edge_evidence_ok =
      edge_metrics.support_ratio >=
          options.geometry_prior_rescue_min_edge_support_ratio &&
      edge_metrics.mean_gradient_ratio >=
          options.geometry_prior_rescue_min_edge_gradient_ratio;
  const bool multi_board_context_for_likelihood =
      candidate.visible_boards_used.size() >=
      static_cast<std::size_t>(std::max(
          2, options.geometry_guided_tag_likelihood_min_visible_boards));
  // A severely distorted but visible Tag can have weak corner response even
  // when its code remains recoverable after model-aware rectification. Keep a
  // deliberately weaker pre-gate for that path; the exact-ID likelihood below
  // is still mandatory before the observation can be committed.
  const bool weak_but_nonzero_edge_evidence =
      edge_metrics.support_ratio >=
          std::max(0.12, 0.33 * options.geometry_prior_rescue_min_edge_support_ratio) &&
      edge_metrics.mean_gradient_ratio >=
          std::max(0.005, 0.25 * options.geometry_prior_rescue_min_edge_gradient_ratio);
  const bool allow_model_aware_precheck =
      !tag_id_validated && options.geometry_guided_tag_likelihood_enabled &&
      multi_board_context_for_likelihood && weak_but_nonzero_edge_evidence;
  if (!corner_response_ok && !edge_evidence_ok && !allow_model_aware_precheck) {
    candidate.local_corner_refine_success = false;
    candidate.reject_reason =
        validation_source +
        "_image_evidence_failed_low_corner_and_edge_response";
    return candidate;
  }

  candidate.local_corner_refine_success = true;
  candidate.image_evidence_success = true;
  candidate.local_redetect_success = true;

  const bool geometry_only_pose_refit_candidate =
      !tag_id_validated &&
      options.geometry_prior_rescue_allow_geometry_only_pose_refit;
  const bool weak_quad_alternative =
      validation_source.find("weak_quad") != std::string::npos;
  const bool geometry_guided_edge_quad =
      validation_source.find("geometry_guided_edge_quad") != std::string::npos;
  const bool geometry_only_strong_edge_evidence =
      candidate.edge_support_ratio >=
          std::max(options.geometry_prior_rescue_min_edge_support_ratio, 0.90) &&
      candidate.mean_edge_gradient_ratio >=
          std::max(options.geometry_prior_rescue_min_edge_gradient_ratio, 0.01);
  const bool geometry_only_refine_ok =
      !options.geometry_prior_rescue_enable_spherical_refine ||
      candidate.spherical_refine_success ||
      candidate.spherical_refine_successful_corner_count >= 2;
  const bool geometry_only_observation_refine_ok =
      tag_id_validated ||
      (!options.geometry_prior_rescue_enable_spherical_refine ||
       candidate.spherical_refine_successful_corner_count >= 4);
  const bool geometry_only_observation_edge_ok =
      tag_id_validated || edge_evidence_ok;
  const bool multi_board_context =
      static_cast<int>(candidate.visible_boards_used.size()) >=
      std::max(2, options.geometry_guided_tag_likelihood_min_visible_boards);
  const bool single_anchor_context =
      options.geometry_guided_tag_likelihood_allow_single_anchor &&
      candidate.visible_boards_used.size() == 1u &&
      single_anchor_is_direct_exact &&
      candidate.frame_pose_refit_source_board_id >= 0 &&
      std::isfinite(candidate.frame_pose_refit_outer_rmse) &&
      candidate.frame_pose_refit_outer_rmse <=
          options.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse;
  const bool require_model_aware_tag_likelihood =
      !tag_id_validated &&
      options.geometry_guided_tag_likelihood_enabled &&
      (multi_board_context || single_anchor_context);
  if (require_model_aware_tag_likelihood &&
      !edge_evidence_ok && !weak_but_nonzero_edge_evidence) {
    candidate.local_corner_refine_success = false;
    candidate.reject_reason =
        validation_source + "_image_evidence_failed_edge_support";
    return candidate;
  }
  // The model-aware likelihood is evaluated only after a refined local pose is
  // available below. Keep the legacy geometry-only gate intact for callers
  // that did not opt into the new code-evidence path.
  if (!tag_id_validated && !require_model_aware_tag_likelihood &&
      (!geometry_only_pose_refit_candidate ||
       (weak_quad_alternative && !geometry_only_strong_edge_evidence) ||
       !geometry_only_refine_ok ||
       !geometry_only_observation_refine_ok ||
       !geometry_only_observation_edge_ok)) {
    candidate.reject_reason =
        validation_source + "_image_evidence_rejected_missing_tag_id_validation";
    if (geometry_only_pose_refit_candidate) {
      std::ostringstream stream;
      stream << candidate.reject_reason
             << "_geometry_only_gate_failed"
             << " weak_quad=" << (weak_quad_alternative ? 1 : 0)
             << " strong_edge=" << (geometry_only_strong_edge_evidence ? 1 : 0)
             << " refine_ok=" << (geometry_only_refine_ok ? 1 : 0)
             << " observation_refine_4corner="
             << (geometry_only_observation_refine_ok ? 1 : 0)
             << " observation_edge="
             << (geometry_only_observation_edge_ok ? 1 : 0);
      candidate.reject_reason = stream.str();
    }
    return candidate;
  }

  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model = camera_config.camera_model;
  intrinsics.distortion_model = camera_config.distortion_model;
  intrinsics.resolution = gray.size();
  if (!intrinsics.SetIntrinsicsVector(camera_config.intrinsics) ||
      !intrinsics.SetDistortionVector(camera_config.distortion_coeffs)) {
    candidate.reject_reason =
        validation_source + "_pose_refit_failed_missing_intrinsics";
    return candidate;
  }

  const std::array<Eigen::Vector3d, 4> object_points_array =
      BuildOuterCornerPointsForBoard(config, board_id);
  Eigen::Isometry3d local_pose = Eigen::Isometry3d::Identity();
  double outer_rmse = 0.0;
  if (!EstimatePoseFromObjectPoints(intrinsics, ToVector(object_points_array),
                                    ToVector(refined_corners), &local_pose,
                                    &outer_rmse)) {
    candidate.pose_refit_success = false;
    candidate.reject_reason = validation_source + "_pose_refit_failed";
    return candidate;
  }
  candidate.pose_refit_success = true;
  candidate.outer_reprojection_rmse = outer_rmse;

  const Eigen::Isometry3d global_pose = ToIsometry3d(T_camera_board_matrix);
  candidate.local_vs_global_rotation_error_deg =
      RotationErrorDegrees(global_pose.rotation(), local_pose.rotation());
  candidate.local_vs_global_translation_error =
      (global_pose.translation() - local_pose.translation()).norm();

  if (require_model_aware_tag_likelihood) {
    const GeometryGuidedTagLikelihood likelihood =
        EvaluateGeometryGuidedTagLikelihood(
            gray, config, options, sphere_camera, board_id, local_pose,
            single_anchor_context);
    candidate.geometry_guided_tag_likelihood_checked = likelihood.checked;
    candidate.geometry_guided_tag_likelihood_passed = likelihood.passed;
    candidate.geometry_guided_tag_likelihood_mode =
        single_anchor_context ? "single_anchor" : "multi_board";
    candidate.geometry_guided_tag_likelihood_expected_hamming =
        likelihood.expected_hamming;
    candidate.geometry_guided_tag_likelihood_runner_up_id =
        likelihood.runner_up_id;
    candidate.geometry_guided_tag_likelihood_runner_up_hamming =
        likelihood.runner_up_hamming;
    candidate.geometry_guided_tag_likelihood_hamming_margin =
        likelihood.hamming_margin;
    candidate.geometry_guided_tag_likelihood_contrast = likelihood.contrast;
    candidate.geometry_guided_tag_likelihood_summary = likelihood.summary;
    if (!likelihood.passed) {
      candidate.reject_reason =
          validation_source + "_image_evidence_rejected_tag_likelihood";
      return candidate;
    }
    tag_id_validated = true;
    candidate.tag_id_validated = true;
  }

  const bool visible_board_frame_pose_consistent =
      geometry_guided_edge_quad &&
      candidate.frame_pose_refit_source_board_id >= 0 &&
      std::isfinite(candidate.frame_pose_refit_outer_rmse) &&
      candidate.frame_pose_refit_outer_rmse > 0.0 &&
      candidate.edge_support_ratio >= options.geometry_prior_rescue_min_edge_support_ratio &&
      candidate.mean_edge_gradient_ratio >=
          options.geometry_prior_rescue_min_edge_gradient_ratio;
  const bool multi_board_context_high_confidence =
      !tag_id_validated &&
      candidate.visible_boards_used.size() >= 3 &&
      candidate.spherical_refine_successful_corner_count >= 4 &&
      candidate.edge_support_ratio >=
          std::max(options.geometry_prior_rescue_min_edge_support_ratio, 0.75) &&
      candidate.mean_edge_gradient_ratio >=
          std::max(options.geometry_prior_rescue_min_edge_gradient_ratio, 0.02);

  double accept_max_outer_rmse =
      options.geometry_prior_rescue_accept_max_outer_rmse;
  if (std::isfinite(candidate.frame_normal_outer_refit_rmse_median) &&
      candidate.frame_normal_outer_refit_rmse_median > 0.0) {
    if (tag_id_validated) {
      accept_max_outer_rmse = std::max(
          accept_max_outer_rmse,
          candidate.frame_normal_outer_refit_rmse_median + 1.0);
    } else {
      if (!multi_board_context_high_confidence) {
        accept_max_outer_rmse = std::min(
            accept_max_outer_rmse,
            std::max(1.5, candidate.frame_normal_outer_refit_rmse_median + 0.75));
      }
    }
  }
  candidate.adaptive_accept_max_outer_rmse = accept_max_outer_rmse;
  if (outer_rmse > accept_max_outer_rmse) {
    candidate.reject_reason =
        validation_source + "_pose_refit_rejected_outer_rmse";
    return candidate;
  }
  const double accept_max_rotation_error_deg =
      tag_id_validated
          ? options.geometry_prior_rescue_accept_max_rotation_error_deg
          : (visible_board_frame_pose_consistent
                 ? std::min(options.geometry_prior_rescue_accept_max_rotation_error_deg,
                            3.0)
                 : std::min(options.geometry_prior_rescue_accept_max_rotation_error_deg,
                            2.0));
  const double accept_max_translation_error =
      tag_id_validated
          ? options.geometry_prior_rescue_accept_max_translation_error
          : (visible_board_frame_pose_consistent
                 ? std::min(options.geometry_prior_rescue_accept_max_translation_error,
                            0.05)
                 : std::min(options.geometry_prior_rescue_accept_max_translation_error,
                            0.03));
  if (candidate.local_vs_global_rotation_error_deg >
      accept_max_rotation_error_deg) {
    candidate.reject_reason =
        validation_source + "_pose_refit_rejected_rotation_error";
    return candidate;
  }
  if (candidate.local_vs_global_translation_error >
      accept_max_translation_error) {
    candidate.reject_reason =
        validation_source + "_pose_refit_rejected_translation_error";
    return candidate;
  }

  candidate.accepted_as_rescued_observation = true;
  candidate.reject_reason = candidate.geometry_guided_tag_likelihood_passed
                                ? "accepted_geometry_guided_tag_likelihood_observation"
                                : (tag_id_validated
                                       ? "accepted_image_validated_rescued_observation"
                                       : "accepted_geometry_only_pose_refit_observation");
  if (!validation_source.empty() && validation_source != "primary") {
    candidate.reject_reason += "_" + validation_source;
  }
  if (rescued_detection != nullptr) {
    std::ostringstream summary;
    summary << "geometry_guided_refine"
            << " validation="
            << (tag_id_validated ? "tag_or_context_id" : "geometry_only")
            << " source=" << validation_source
            << " max_disp=" << max_displacement
            << " min_corner_response_ratio=" << min_response_ratio
            << " edge_support_ratio=" << edge_metrics.support_ratio
            << " mean_edge_gradient_ratio=" << edge_metrics.mean_gradient_ratio
            << " outer_rmse=" << outer_rmse
            << " rot_err_deg="
            << candidate.local_vs_global_rotation_error_deg
            << " trans_err="
            << candidate.local_vs_global_translation_error;
    *rescued_detection = BuildRescuedOuterDetection(
        board_id, gray.size(), refined_corners, tag_id_validated,
        std::max(min_response_ratio, edge_metrics.mean_gradient_ratio),
        summary.str());
  }
  return candidate;
}

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
    const OuterWrongIdProposal* wrong_id_proposal,
    OuterTagDetectionResult* rescued_detection) {
  std::array<Eigen::Vector2d, 4> effective_predicted_corners = predicted_corners;
  if (wrong_id_proposal != nullptr) {
    effective_predicted_corners = wrong_id_proposal->corners_original_image;
  }
  GeometryPriorOuterSeedCandidate candidate =
      BuildGeometryPriorOuterSeedCandidate(frame_input, board_id,
                                           visible_boards_used,
                                           effective_predicted_corners,
                                           prediction_source_label,
                                           frame_pose_refit_source_board_id,
                                           frame_pose_refit_outer_rmse,
                                           original_failure_reason);
  candidate.image_evidence_checked = true;
  candidate.frame_normal_outer_refit_rmse_median =
      frame_normal_outer_refit_rmse_median;
  candidate.adaptive_accept_max_outer_rmse =
      options.geometry_prior_rescue_accept_max_outer_rmse;
  if (rescued_detection != nullptr) {
    *rescued_detection = OuterTagDetectionResult{};
  }

  double local_corner_scale_px = 0.0;
  const int window_radius = ComputeGeometryPriorRescueSubpixWindowRadius(
      options, candidate.predicted_corners, &local_corner_scale_px);
  candidate.local_corner_scale_px = local_corner_scale_px;
  candidate.subpix_window_radius = window_radius;
  if (window_radius <= 0) {
    candidate.reject_reason = "image_evidence_disabled_subpix_window_radius";
    return candidate;
  }
  if (gray.empty()) {
    candidate.reject_reason = "image_evidence_failed_empty_image";
    return candidate;
  }

  for (const cv::Point2f& point : candidate.predicted_corners) {
    if (!IsInsideImage(point, gray.size(), window_radius + 2.0)) {
      candidate.reject_reason = "image_evidence_failed_corner_near_border";
      return candidate;
    }
  }

  std::array<cv::Point2f, 4> initial_corners = candidate.predicted_corners;
  candidate.roi_redetect_checked = true;
  std::array<cv::Point2f, 4> roi_redetected_corners{};
  std::string roi_redetect_summary;
  if (TryExpandedRoiRedetect(gray, board_id, candidate.predicted_corners,
                             &roi_redetected_corners,
                             &candidate.roi_redetect_bbox,
                             &candidate.roi_redetect_detected_tag_id,
                             &candidate.roi_redetect_hamming,
                             nullptr,
                             nullptr,
                             &roi_redetect_summary)) {
    candidate.roi_redetect_success = true;
    initial_corners = roi_redetected_corners;
  }
  candidate.roi_redetect_summary = roi_redetect_summary;

  candidate.rectified_patch_checked = true;
  std::array<cv::Point2f, 4> rectified_decoded_corners{};
  std::string rectified_patch_summary;
  if (!candidate.roi_redetect_success &&
      TryDsRayPatchDecode(gray, camera_config, board_id,
                          candidate.predicted_corners,
                          &rectified_decoded_corners,
                          &candidate.rectified_patch_detected_tag_id,
                          &candidate.rectified_patch_hamming,
                          &rectified_patch_summary)) {
    candidate.rectified_patch_decode_success = true;
    initial_corners = rectified_decoded_corners;
  }
  candidate.rectified_patch_summary = rectified_patch_summary;

  const bool tag_id_validated =
      candidate.roi_redetect_success ||
      candidate.rectified_patch_decode_success;
  const bool single_anchor_is_direct_exact =
      HasDirectExactOuterAnchor(frame_input, frame_pose_refit_source_board_id);
  GeometryPriorOuterSeedCandidate primary_candidate =
      FinalizeGeometryPriorOuterSeedCandidate(
          gray, config, options, camera_config, board_id, initial_corners,
          T_camera_board_matrix, tag_id_validated, single_anchor_is_direct_exact,
          "primary", candidate,
          rescued_detection);
  if (primary_candidate.accepted_as_rescued_observation || tag_id_validated) {
    return primary_candidate;
  }

  std::array<cv::Point2f, 4> guided_edge_corners{};
  std::string guided_edge_summary;
  if (TryGeometryGuidedEdgeQuadProposal(gray, candidate.predicted_corners,
                                        &guided_edge_corners,
                                        &guided_edge_summary)) {
    if (rescued_detection != nullptr) {
      *rescued_detection = OuterTagDetectionResult{};
    }
    GeometryPriorOuterSeedCandidate guided_candidate = candidate;
    guided_candidate.prediction_source_label += "_geometry_guided_edge_quad";
    guided_candidate.roi_redetect_checked = true;
    guided_candidate.roi_redetect_success = false;
    guided_candidate.roi_redetect_detected_tag_id = -4;
    guided_candidate.roi_redetect_hamming = -1;
    guided_candidate.roi_redetect_summary = guided_edge_summary;
    GeometryPriorOuterSeedCandidate guided_result =
        FinalizeGeometryPriorOuterSeedCandidate(
            gray, config, options, camera_config, board_id, guided_edge_corners,
            T_camera_board_matrix, false, single_anchor_is_direct_exact,
            "geometry_guided_edge_quad",
            guided_candidate, rescued_detection);
    if (guided_result.accepted_as_rescued_observation) {
      return guided_result;
    }
    primary_candidate.roi_redetect_summary +=
        " geometry_guided_edge_quad_reject=" + guided_result.reject_reason +
        " " + guided_edge_summary;
  } else {
    primary_candidate.roi_redetect_summary +=
        " geometry_guided_edge_quad_reject=no_candidate " + guided_edge_summary;
  }

  std::array<cv::Point2f, 4> weak_quad_corners{};
  cv::Rect weak_roi_bbox;
  std::string weak_quad_summary;
  if (!TryRelaxedRoiWeakQuadAlternative(gray, candidate.predicted_corners,
                                        &weak_quad_corners, &weak_roi_bbox,
                                        &weak_quad_summary)) {
    primary_candidate.roi_redetect_summary +=
        " weak_quad_alternative_reject=no_candidate " + weak_quad_summary;
    return primary_candidate;
  }

  if (rescued_detection != nullptr) {
    *rescued_detection = OuterTagDetectionResult{};
  }
  GeometryPriorOuterSeedCandidate weak_candidate = candidate;
  weak_candidate.prediction_source_label += "_weak_quad_alternative";
  weak_candidate.roi_redetect_checked = true;
  weak_candidate.roi_redetect_success = true;
  weak_candidate.roi_redetect_detected_tag_id = -3;
  weak_candidate.roi_redetect_hamming = -1;
  weak_candidate.roi_redetect_bbox = weak_roi_bbox;
  weak_candidate.roi_redetect_summary = weak_quad_summary;
  GeometryPriorOuterSeedCandidate weak_result =
      FinalizeGeometryPriorOuterSeedCandidate(
          gray, config, options, camera_config, board_id, weak_quad_corners,
          T_camera_board_matrix, false, single_anchor_is_direct_exact,
          "weak_quad_alternative",
          weak_candidate, rescued_detection);
  if (weak_result.accepted_as_rescued_observation) {
    return weak_result;
  }
  primary_candidate.roi_redetect_summary +=
      " weak_quad_alternative_reject=" + weak_result.reject_reason +
      " " + weak_quad_summary;
  return primary_candidate;
	}

GeometryPriorOuterSeedCandidate BuildGeometryPriorOuterSeedCandidate(
    const InternalRegenerationFrameInput& frame_input,
    int board_id,
    const std::vector<int>& visible_boards_used,
    const std::array<Eigen::Vector2d, 4>& corners,
    const std::string& prediction_source_label,
    int frame_pose_refit_source_board_id,
    double frame_pose_refit_outer_rmse,
    const std::string& original_failure_reason) {
  GeometryPriorOuterSeedCandidate candidate;
  candidate.frame_index = frame_input.frame_index;
  candidate.frame_label = frame_input.frame_label;
  candidate.missing_board_id = board_id;
  candidate.prediction_source_label = prediction_source_label;
  candidate.frame_pose_refit_source_board_id =
      frame_pose_refit_source_board_id;
  candidate.frame_pose_refit_outer_rmse = frame_pose_refit_outer_rmse;
  candidate.visible_boards_used = visible_boards_used;
  for (int index = 0; index < 4; ++index) {
    candidate.predicted_corners[static_cast<std::size_t>(index)] =
        cv::Point2f(static_cast<float>(corners[static_cast<std::size_t>(index)].x()),
                    static_cast<float>(corners[static_cast<std::size_t>(index)].y()));
    candidate.refined_corners[static_cast<std::size_t>(index)] =
        candidate.predicted_corners[static_cast<std::size_t>(index)];
  }
  const QuadTopologyCheck predicted_topology =
      CheckQuadTopology(candidate.predicted_corners);
  candidate.predicted_area_px = predicted_topology.area_px;
  candidate.predicted_signed_area_px = predicted_topology.signed_area_px;
  candidate.predicted_quad_topology_valid = predicted_topology.valid;
  candidate.quad_topology_summary =
      "predicted{" + predicted_topology.summary + "}";
  candidate.roi_valid = true;
  candidate.image_evidence_checked = false;
  candidate.image_evidence_success = false;
  candidate.local_redetect_success = false;
  candidate.local_corner_refine_success = false;
  candidate.pose_refit_success = false;
  candidate.accepted_as_rescued_observation = false;
  candidate.reject_reason =
      "diagnostic_only_no_image_evidence_validation_from_" +
      original_failure_reason;
  return candidate;
}

ApriltagInternalDetectionResult BuildFailedDetectionResult(
    int board_id,
    const cv::Size& image_size,
    const OuterTagDetectionResult& outer_detection,
    const std::string& failure_reason) {
  ApriltagInternalDetectionResult detection;
  detection.board_id = board_id;
  detection.image_size = image_size;
  detection.outer_detection = outer_detection;
  detection.tag_detected = outer_detection.success;
  detection.failure_reason = failure_reason;
  detection.internal_camera_source = "missing";
  return detection;
}

std::string BuildRegenerationFailureWarning(
    const InternalRegenerationFrameInput& frame_input,
    const std::string& state_source_label,
    int board_id,
    bool pose_prior_used,
    const std::string& failure_reason) {
  std::ostringstream stream;
  stream << "state=" << state_source_label
         << " frame=" << frame_input.frame_index;
  if (!frame_input.frame_label.empty()) {
    stream << " (" << frame_input.frame_label << ")";
  }
  stream << " board=" << board_id
         << " prior=" << (pose_prior_used ? 1 : 0)
         << " skipped: " << failure_reason;
  return stream.str();
}

void EmitRegenerationWarning(const std::string& warning,
                             std::vector<std::string>* warnings) {
  if (warning.empty()) {
    return;
  }
  AppendUniqueWarning(warning, warnings);
  std::cerr << "[internal_regen] " << warning << std::endl;
}

void AccumulateRuntimeBreakdown(
    const ApriltagInternalRuntimeBreakdown& detection_runtime,
    InternalRegenerationRuntimeBreakdown* frame_runtime) {
  if (frame_runtime == nullptr) {
    return;
  }
  frame_runtime->pose_estimation_seconds +=
      detection_runtime.pose_estimation_seconds;
  frame_runtime->boundary_model_seconds +=
      detection_runtime.boundary_model_seconds;
  frame_runtime->seed_search_seconds += detection_runtime.seed_search_seconds;
  frame_runtime->ray_refine_seconds += detection_runtime.ray_refine_seconds;
  frame_runtime->image_evidence_seconds +=
      detection_runtime.image_evidence_seconds;
  frame_runtime->subpix_seconds += detection_runtime.subpix_seconds;
  frame_runtime->pose_estimation_call_count +=
      detection_runtime.pose_estimation_call_count;
  frame_runtime->pose_rescue_attempt_count +=
      detection_runtime.pose_rescue_attempt_count;
  frame_runtime->pose_rescue_success_count +=
      detection_runtime.pose_rescue_success_count;
  frame_runtime->pose_rescue_used_count +=
      detection_runtime.pose_rescue_used_count;
  frame_runtime->boundary_model_build_count +=
      detection_runtime.boundary_model_build_count;
  frame_runtime->attempted_internal_corner_count +=
      detection_runtime.attempted_internal_corner_count;
  frame_runtime->valid_internal_corner_count +=
      detection_runtime.valid_internal_corner_count;
}

}  // namespace

int InternalRegenerationFrameResult::SuccessfulBoardCount() const {
  int count = 0;
  for (const RegeneratedBoardMeasurement& measurement : board_measurements) {
    count += measurement.detection.success ? 1 : 0;
  }
  return count;
}

int InternalRegenerationFrameResult::ValidInternalCornerCount() const {
  int count = 0;
  for (const RegeneratedBoardMeasurement& measurement : board_measurements) {
    count += measurement.detection.valid_internal_corner_count;
  }
  return count;
}

ApriltagInternalMultiDetectionResult InternalRegenerationFrameResult::AsMultiDetectionResult() const {
  ApriltagInternalMultiDetectionResult result;
  result.image_size = image_size;
  result.requested_board_ids.reserve(board_measurements.size());
  result.detections.reserve(board_measurements.size());
  for (const RegeneratedBoardMeasurement& measurement : board_measurements) {
    result.requested_board_ids.push_back(measurement.board_id);
    result.detections.push_back(measurement.detection);
  }
  return result;
}

MultiBoardInternalMeasurementRegenerator::MultiBoardInternalMeasurementRegenerator(
    ApriltagInternalConfig config,
    ApriltagInternalDetectionOptions options)
    : config_(std::move(config)),
      options_(std::move(options)),
      detector_(config_, options_) {}

InternalRegenerationFrameResult MultiBoardInternalMeasurementRegenerator::RegenerateFrame(
    const cv::Mat& image,
    const InternalRegenerationFrameInput& frame_input,
    const OuterBootstrapResult& bootstrap_result) const {
  if (image.empty()) {
    throw std::runtime_error("RegenerateFrame requires a non-empty image.");
  }

  InternalRegenerationFrameResult result;
  result.frame_index = frame_input.frame_index;
  result.frame_label = frame_input.frame_label;
  result.state_source_label = "bootstrap";
  result.image_size = image.size();
  CollectVisibleBoardIds(frame_input, &result.visible_board_ids);

  const OuterBootstrapFrameState* frame_state = FindFrameState(bootstrap_result, frame_input);
  result.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;

  const IntermediateCameraConfig camera_override =
      MakeBootstrapCameraConfig(bootstrap_result.coarse_camera);
  const LocalFramePoseRefit visible_frame_refit =
      EstimateLocalFramePoseFromVisibleBoards(
          config_, bootstrap_result.coarse_camera, frame_input,
          [&](int visible_board_id, Eigen::Matrix4d* T_reference_board) {
            if (T_reference_board == nullptr) {
              return false;
            }
            if (visible_board_id == bootstrap_result.reference_board_id) {
              *T_reference_board = Eigen::Matrix4d::Identity();
              return true;
            }
            const OuterBootstrapBoardState* visible_board_state =
                FindBoardState(bootstrap_result, visible_board_id);
            if (visible_board_state == nullptr ||
                !visible_board_state->initialized) {
              return false;
            }
            *T_reference_board = visible_board_state->T_reference_board;
            return true;
          });
  const double frame_normal_outer_refit_rmse_median =
      ComputeFrameNormalOuterRefitRmseMedian(
          config_, bootstrap_result.coarse_camera, frame_input);

  result.board_measurements.reserve(frame_input.outer_detections.detections.size());
  for (std::size_t index = 0; index < frame_input.outer_detections.requested_board_ids.size(); ++index) {
    const int board_id = frame_input.outer_detections.requested_board_ids[index];
    OuterTagDetectionResult outer_detection;
    if (index < frame_input.outer_detections.detections.size()) {
      outer_detection = frame_input.outer_detections.detections[index];
    } else {
      outer_detection = MakeMissingOuterTagDetection(board_id);
    }

    const OuterBootstrapBoardState* board_state = FindBoardState(bootstrap_result, board_id);
    const bool board_pose_available =
        board_state != nullptr && board_state->initialized;
    const bool scene_pose_prior_available =
        frame_state != nullptr && frame_state->initialized &&
        board_pose_available;
    const bool visible_refit_pose_prior_available =
        !scene_pose_prior_available && visible_frame_refit.success &&
        board_pose_available;
    const bool pose_prior_available =
        scene_pose_prior_available || visible_refit_pose_prior_available;
    Eigen::Matrix4d T_camera_board = Eigen::Matrix4d::Identity();
    if (scene_pose_prior_available) {
      T_camera_board = ComposeCameraBoardTransform(
          frame_state->T_camera_reference, board_state->T_reference_board);
    } else if (visible_refit_pose_prior_available) {
      T_camera_board =
          visible_frame_refit.T_camera_reference *
          board_state->T_reference_board;
    }

    const std::string original_outer_failure_reason =
        outer_detection.failure_reason_text.empty()
            ? ToString(outer_detection.failure_reason)
            : outer_detection.failure_reason_text;
    if (options_.enable_geometry_prior_outer_seed &&
        !outer_detection.success && pose_prior_available) {
      ApriltagInternalConfig board_config = config_;
      board_config.tag_id = board_id;
      const ApriltagCanonicalModel model(board_config);
      bool any_candidate_accepted = false;
      bool selected_rescued_detection = false;
      double selected_rescued_rmse = std::numeric_limits<double>::infinity();
      OuterTagDetectionResult selected_detection;
      const auto evaluate_prediction =
          [&](const Eigen::Matrix4d& T_camera_board_prediction,
              const std::string& prediction_source_label,
              int frame_pose_refit_source_board_id,
              double frame_pose_refit_outer_rmse,
              const std::vector<int>& visible_boards_used,
              const OuterWrongIdProposal* wrong_id_proposal = nullptr) {
            std::array<Eigen::Vector2d, 4> predicted_outer_corners{};
            if (!ProjectGeometryPriorOuterCorners(
                    camera_override, model, T_camera_board_prediction,
                    image.size(), &predicted_outer_corners)) {
              return;
            }
            OuterWrongIdProposal topology_ordered_proposal;
            const OuterWrongIdProposal* effective_wrong_id_proposal =
                wrong_id_proposal;
            std::string effective_prediction_source_label =
                prediction_source_label;
            std::string topology_association_summary;
            if (wrong_id_proposal != nullptr) {
              const WrongIdTopologyAssociation association =
                  AssociateWrongIdProposalToTopology(predicted_outer_corners,
                                                     *wrong_id_proposal);
              topology_association_summary = association.summary;
              if (!association.compatible) {
                GeometryPriorOuterSeedCandidate rejected_candidate =
                    BuildGeometryPriorOuterSeedCandidate(
                        frame_input, board_id, visible_boards_used,
                        predicted_outer_corners,
                        prediction_source_label + "_topology_rejected_from" +
                            std::to_string(wrong_id_proposal->detected_tag_id),
                        frame_pose_refit_source_board_id,
                        frame_pose_refit_outer_rmse,
                        original_outer_failure_reason);
                rejected_candidate.quad_topology_summary +=
                    " " + topology_association_summary;
                rejected_candidate.reject_reason =
                    "topology_id_association_rejected";
                result.geometry_prior_outer_seed_candidates.push_back(
                    std::move(rejected_candidate));
                return;
              }
              topology_ordered_proposal = *wrong_id_proposal;
              topology_ordered_proposal.corners_original_image =
                  association.ordered_corners;
              effective_wrong_id_proposal = &topology_ordered_proposal;
              effective_prediction_source_label += "_topology_assoc";
            }
        OuterTagDetectionResult rescued_detection;
        GeometryPriorOuterSeedCandidate candidate =
            EvaluateGeometryPriorOuterSeedCandidate(
                image, config_, options_, camera_override, frame_input,
                board_id, visible_boards_used, predicted_outer_corners,
                T_camera_board_prediction, effective_prediction_source_label,
                frame_pose_refit_source_board_id, frame_pose_refit_outer_rmse,
                frame_normal_outer_refit_rmse_median,
                original_outer_failure_reason,
                effective_wrong_id_proposal,
                &rescued_detection);
        if (!topology_association_summary.empty()) {
          candidate.quad_topology_summary +=
              " " + topology_association_summary;
        }
        result.geometry_prior_outer_seed_candidates.push_back(candidate);
            if (candidate.accepted_as_rescued_observation) {
              any_candidate_accepted = true;
              if (candidate.outer_reprojection_rmse < selected_rescued_rmse) {
                selected_rescued_rmse = candidate.outer_reprojection_rmse;
                selected_detection = rescued_detection;
                selected_rescued_detection = true;
              }
            }
          };
      evaluate_prediction(T_camera_board, result.state_source_label, -1, 0.0,
                          result.visible_board_ids);
      if (outer_detection.failure_reason ==
          OuterTagFailureReason::DetectionsExistButNoMatchingTagId) {
        for (const OuterWrongIdProposal& proposal :
             outer_detection.wrong_id_proposals) {
          evaluate_prediction(
              T_camera_board,
              result.state_source_label + "_wrong_id_proposal_from" +
                  std::to_string(proposal.detected_tag_id),
              -1, 0.0, result.visible_board_ids, &proposal);
        }
      }
      if (visible_frame_refit.success && board_pose_available) {
        const Eigen::Matrix4d T_camera_board_visible_refit =
            visible_frame_refit.T_camera_reference *
            board_state->T_reference_board;
        std::vector<int> visible_refit_board_ids =
            visible_frame_refit.support_board_ids;
        if (visible_refit_board_ids.empty()) {
          visible_refit_board_ids.push_back(
              visible_frame_refit.source_board_id);
        }
        const std::string visible_refit_label =
            result.state_source_label +
            (visible_frame_refit.support_board_ids.size() >= 2
                 ? "_visible_refit_consensus"
                 : "_visible_refit_single");
        evaluate_prediction(T_camera_board_visible_refit,
                            visible_refit_label,
                            visible_frame_refit.source_board_id,
                            visible_frame_refit.outer_rmse,
                            visible_refit_board_ids);
        std::vector<int> individual_refit_board_ids;
        for (const LocalFramePoseRefit::Candidate& pose_candidate :
             visible_frame_refit.candidates) {
          AppendUniqueBoardId(pose_candidate.source_board_id,
                              &individual_refit_board_ids);
        }
        if (individual_refit_board_ids.size() >= 2) {
          for (const LocalFramePoseRefit::Candidate& pose_candidate :
               visible_frame_refit.candidates) {
            const Eigen::Matrix4d T_camera_board_individual_refit =
                pose_candidate.T_camera_reference *
                board_state->T_reference_board;
            evaluate_prediction(
                T_camera_board_individual_refit,
                result.state_source_label + "_visible_refit_board" +
                    std::to_string(pose_candidate.source_board_id),
                pose_candidate.source_board_id, pose_candidate.outer_rmse,
                individual_refit_board_ids);
          }
        }
      }
      if (selected_rescued_detection &&
          options_.geometry_prior_rescue_use_as_observation &&
          !options_.geometry_prior_rescue_diagnostic_only) {
        outer_detection = selected_detection;
      } else if (any_candidate_accepted &&
                 options_.geometry_prior_rescue_use_as_observation &&
                 options_.geometry_prior_rescue_diagnostic_only) {
        EmitRegenerationWarning(
            "geometry prior rescue passed image validation but is kept "
            "diagnostic-only; pass --stage5-geometry-prior-rescue-diagnostic-only 0 "
            "to allow backend use.",
            &result.warnings);
      }
    }

    RegeneratedBoardMeasurement measurement;
    measurement.board_id = board_id;
    measurement.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;
    measurement.board_bootstrap_initialized = board_state != nullptr && board_state->initialized;
    const bool has_local_outer_pose_seed =
        outer_detection.success &&
        std::all_of(outer_detection.refined_valid.begin(),
                    outer_detection.refined_valid.end(),
                    [](bool valid) { return valid; });
    const bool use_pose_prior_for_internal_generation =
        pose_prior_available && !outer_detection.used_local_patch_rescue &&
        !has_local_outer_pose_seed;
    measurement.pose_prior_used = use_pose_prior_for_internal_generation;
    try {
      measurement.detection = detector_.DetectFromOuterDetection(
          image, board_id, outer_detection, &camera_override,
          use_pose_prior_for_internal_generation ? &T_camera_board : nullptr);
    } catch (const std::exception& error) {
      measurement.detection = BuildFailedDetectionResult(
          board_id, image.size(), outer_detection, error.what());
    }
    if (outer_detection.success && !measurement.detection.success &&
        !measurement.detection.failure_reason.empty()) {
      EmitRegenerationWarning(
          BuildRegenerationFailureWarning(
              frame_input, result.state_source_label, board_id,
              measurement.pose_prior_used, measurement.detection.failure_reason),
          &result.warnings);
    }
    AccumulateRuntimeBreakdown(measurement.detection.runtime_breakdown,
                               &result.runtime_breakdown);
    result.board_measurements.push_back(measurement);

    if (outer_detection.success) {
      AppendUniqueBoardId(board_id, &result.visible_board_ids);
    }
  }

  return result;
}

InternalRegenerationFrameResult MultiBoardInternalMeasurementRegenerator::RegenerateFrame(
    const cv::Mat& image,
    const InternalRegenerationFrameInput& frame_input,
    const JointReprojectionSceneState& scene_state) const {
  if (image.empty()) {
    throw std::runtime_error("RegenerateFrame requires a non-empty image.");
  }

  InternalRegenerationFrameResult result;
  result.frame_index = frame_input.frame_index;
  result.frame_label = frame_input.frame_label;
  result.state_source_label = "optimized_scene";
  result.image_size = image.size();
  CollectVisibleBoardIds(frame_input, &result.visible_board_ids);

  const JointSceneFrameState* frame_state = FindFrameState(scene_state, frame_input);
  result.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;

  const IntermediateCameraConfig camera_override =
      MakeSceneCameraConfig(scene_state.camera);
  const LocalFramePoseRefit visible_frame_refit =
      EstimateLocalFramePoseFromVisibleBoards(
          config_, scene_state.camera, frame_input,
          [&](int visible_board_id, Eigen::Matrix4d* T_reference_board) {
            if (T_reference_board == nullptr) {
              return false;
            }
            if (visible_board_id == scene_state.reference_board_id) {
              *T_reference_board = Eigen::Matrix4d::Identity();
              return true;
            }
            const JointSceneBoardState* visible_board_state =
                FindBoardState(scene_state, visible_board_id);
            if (visible_board_state == nullptr ||
                !visible_board_state->initialized) {
              return false;
            }
            *T_reference_board = visible_board_state->T_reference_board;
            return true;
          });
  const double frame_normal_outer_refit_rmse_median =
      ComputeFrameNormalOuterRefitRmseMedian(
          config_, scene_state.camera, frame_input);

  result.board_measurements.reserve(frame_input.outer_detections.detections.size());
  for (std::size_t index = 0; index < frame_input.outer_detections.requested_board_ids.size(); ++index) {
    const int board_id = frame_input.outer_detections.requested_board_ids[index];
    OuterTagDetectionResult outer_detection;
    if (index < frame_input.outer_detections.detections.size()) {
      outer_detection = frame_input.outer_detections.detections[index];
    } else {
      outer_detection = MakeMissingOuterTagDetection(board_id);
    }

    const JointSceneBoardState* board_state = FindBoardState(scene_state, board_id);
    const bool board_pose_available =
        board_state != nullptr && board_state->initialized;
    const bool scene_pose_prior_available =
        frame_state != nullptr && frame_state->initialized &&
        board_pose_available;
    const bool visible_refit_pose_prior_available =
        !scene_pose_prior_available && visible_frame_refit.success &&
        board_pose_available;
    const bool pose_prior_available =
        scene_pose_prior_available || visible_refit_pose_prior_available;
    Eigen::Matrix4d T_camera_board = Eigen::Matrix4d::Identity();
    if (scene_pose_prior_available) {
      T_camera_board = ComposeCameraBoardTransform(
          frame_state->T_camera_reference, board_state->T_reference_board);
    } else if (visible_refit_pose_prior_available) {
      T_camera_board =
          visible_frame_refit.T_camera_reference *
          board_state->T_reference_board;
    }

    const std::string original_outer_failure_reason =
        outer_detection.failure_reason_text.empty()
            ? ToString(outer_detection.failure_reason)
            : outer_detection.failure_reason_text;
    if (options_.enable_geometry_prior_outer_seed &&
        !outer_detection.success && pose_prior_available) {
      ApriltagInternalConfig board_config = config_;
      board_config.tag_id = board_id;
      const ApriltagCanonicalModel model(board_config);
      bool any_candidate_accepted = false;
      bool selected_rescued_detection = false;
      double selected_rescued_rmse = std::numeric_limits<double>::infinity();
      OuterTagDetectionResult selected_detection;
      const auto evaluate_prediction =
          [&](const Eigen::Matrix4d& T_camera_board_prediction,
              const std::string& prediction_source_label,
              int frame_pose_refit_source_board_id,
              double frame_pose_refit_outer_rmse,
              const std::vector<int>& visible_boards_used,
              const OuterWrongIdProposal* wrong_id_proposal = nullptr) {
            std::array<Eigen::Vector2d, 4> predicted_outer_corners{};
            if (!ProjectGeometryPriorOuterCorners(
                    camera_override, model, T_camera_board_prediction,
                    image.size(), &predicted_outer_corners)) {
              return;
            }
            OuterWrongIdProposal topology_ordered_proposal;
            const OuterWrongIdProposal* effective_wrong_id_proposal =
                wrong_id_proposal;
            std::string effective_prediction_source_label =
                prediction_source_label;
            std::string topology_association_summary;
            if (wrong_id_proposal != nullptr) {
              const WrongIdTopologyAssociation association =
                  AssociateWrongIdProposalToTopology(predicted_outer_corners,
                                                     *wrong_id_proposal);
              topology_association_summary = association.summary;
              if (!association.compatible) {
                GeometryPriorOuterSeedCandidate rejected_candidate =
                    BuildGeometryPriorOuterSeedCandidate(
                        frame_input, board_id, visible_boards_used,
                        predicted_outer_corners,
                        prediction_source_label + "_topology_rejected_from" +
                            std::to_string(wrong_id_proposal->detected_tag_id),
                        frame_pose_refit_source_board_id,
                        frame_pose_refit_outer_rmse,
                        original_outer_failure_reason);
                rejected_candidate.quad_topology_summary +=
                    " " + topology_association_summary;
                rejected_candidate.reject_reason =
                    "topology_id_association_rejected";
                result.geometry_prior_outer_seed_candidates.push_back(
                    std::move(rejected_candidate));
                return;
              }
              topology_ordered_proposal = *wrong_id_proposal;
              topology_ordered_proposal.corners_original_image =
                  association.ordered_corners;
              effective_wrong_id_proposal = &topology_ordered_proposal;
              effective_prediction_source_label += "_topology_assoc";
            }
        OuterTagDetectionResult rescued_detection;
        GeometryPriorOuterSeedCandidate candidate =
            EvaluateGeometryPriorOuterSeedCandidate(
                image, config_, options_, camera_override, frame_input,
                board_id, visible_boards_used, predicted_outer_corners,
                T_camera_board_prediction, effective_prediction_source_label,
                frame_pose_refit_source_board_id, frame_pose_refit_outer_rmse,
                frame_normal_outer_refit_rmse_median,
                original_outer_failure_reason,
                effective_wrong_id_proposal,
                &rescued_detection);
        if (!topology_association_summary.empty()) {
          candidate.quad_topology_summary +=
              " " + topology_association_summary;
        }
        result.geometry_prior_outer_seed_candidates.push_back(candidate);
            if (candidate.accepted_as_rescued_observation) {
              any_candidate_accepted = true;
              if (candidate.outer_reprojection_rmse < selected_rescued_rmse) {
                selected_rescued_rmse = candidate.outer_reprojection_rmse;
                selected_detection = rescued_detection;
                selected_rescued_detection = true;
              }
            }
          };
      evaluate_prediction(T_camera_board, result.state_source_label, -1, 0.0,
                          result.visible_board_ids);
      if (outer_detection.failure_reason ==
          OuterTagFailureReason::DetectionsExistButNoMatchingTagId) {
        for (const OuterWrongIdProposal& proposal :
             outer_detection.wrong_id_proposals) {
          evaluate_prediction(
              T_camera_board,
              result.state_source_label + "_wrong_id_proposal_from" +
                  std::to_string(proposal.detected_tag_id),
              -1, 0.0, result.visible_board_ids, &proposal);
        }
      }
      if (visible_frame_refit.success && board_pose_available) {
        const Eigen::Matrix4d T_camera_board_visible_refit =
            visible_frame_refit.T_camera_reference *
            board_state->T_reference_board;
        std::vector<int> visible_refit_board_ids =
            visible_frame_refit.support_board_ids;
        if (visible_refit_board_ids.empty()) {
          visible_refit_board_ids.push_back(
              visible_frame_refit.source_board_id);
        }
        const std::string visible_refit_label =
            result.state_source_label +
            (visible_frame_refit.support_board_ids.size() >= 2
                 ? "_visible_refit_consensus"
                 : "_visible_refit_single");
        evaluate_prediction(T_camera_board_visible_refit,
                            visible_refit_label,
                            visible_frame_refit.source_board_id,
                            visible_frame_refit.outer_rmse,
                            visible_refit_board_ids);
        std::vector<int> individual_refit_board_ids;
        for (const LocalFramePoseRefit::Candidate& pose_candidate :
             visible_frame_refit.candidates) {
          AppendUniqueBoardId(pose_candidate.source_board_id,
                              &individual_refit_board_ids);
        }
        if (individual_refit_board_ids.size() >= 2) {
          for (const LocalFramePoseRefit::Candidate& pose_candidate :
               visible_frame_refit.candidates) {
            const Eigen::Matrix4d T_camera_board_individual_refit =
                pose_candidate.T_camera_reference *
                board_state->T_reference_board;
            evaluate_prediction(
                T_camera_board_individual_refit,
                result.state_source_label + "_visible_refit_board" +
                    std::to_string(pose_candidate.source_board_id),
                pose_candidate.source_board_id, pose_candidate.outer_rmse,
                individual_refit_board_ids);
          }
        }
      }
      if (selected_rescued_detection &&
          options_.geometry_prior_rescue_use_as_observation &&
          !options_.geometry_prior_rescue_diagnostic_only) {
        outer_detection = selected_detection;
      } else if (any_candidate_accepted &&
                 options_.geometry_prior_rescue_use_as_observation &&
                 options_.geometry_prior_rescue_diagnostic_only) {
        EmitRegenerationWarning(
            "geometry prior rescue passed image validation but is kept "
            "diagnostic-only; pass --stage5-geometry-prior-rescue-diagnostic-only 0 "
            "to allow backend use.",
            &result.warnings);
      }
    }

    RegeneratedBoardMeasurement measurement;
    measurement.board_id = board_id;
    measurement.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;
    measurement.board_bootstrap_initialized = board_state != nullptr && board_state->initialized;
    const bool has_local_outer_pose_seed =
        outer_detection.success &&
        std::all_of(outer_detection.refined_valid.begin(),
                    outer_detection.refined_valid.end(),
                    [](bool valid) { return valid; });
    const bool use_pose_prior_for_internal_generation =
        pose_prior_available && !outer_detection.used_local_patch_rescue &&
        !has_local_outer_pose_seed;
    measurement.pose_prior_used = use_pose_prior_for_internal_generation;
    try {
      measurement.detection = detector_.DetectFromOuterDetection(
          image, board_id, outer_detection, &camera_override,
          use_pose_prior_for_internal_generation ? &T_camera_board : nullptr);
    } catch (const std::exception& error) {
      measurement.detection = BuildFailedDetectionResult(
          board_id, image.size(), outer_detection, error.what());
    }
    if (outer_detection.success && !measurement.detection.success &&
        !measurement.detection.failure_reason.empty()) {
      EmitRegenerationWarning(
          BuildRegenerationFailureWarning(
              frame_input, result.state_source_label, board_id,
              measurement.pose_prior_used, measurement.detection.failure_reason),
          &result.warnings);
    }
    AccumulateRuntimeBreakdown(measurement.detection.runtime_breakdown,
                               &result.runtime_breakdown);
    result.board_measurements.push_back(measurement);

    if (outer_detection.success) {
      AppendUniqueBoardId(board_id, &result.visible_board_ids);
    }
  }

  return result;
}

void MultiBoardInternalMeasurementRegenerator::DrawFrameOverlay(
    const cv::Mat& image,
    const InternalRegenerationFrameResult& frame_result,
    cv::Mat* output_image) const {
  if (output_image == nullptr) {
    throw std::runtime_error("DrawFrameOverlay requires a valid output pointer.");
  }
  *output_image = image.clone();
  ApriltagInternalMultiDetectionResult multi_detection = frame_result.AsMultiDetectionResult();
  detector_.DrawDetections(multi_detection, output_image);

  const int banner_height = 78;
  cv::rectangle(*output_image, cv::Rect(0, 0, output_image->cols, banner_height),
                cv::Scalar(18, 18, 18), cv::FILLED);

  std::ostringstream header;
  header << "frame " << frame_result.frame_index << "  state="
         << frame_result.state_source_label << "  frame_init="
         << (frame_result.frame_bootstrap_initialized ? "yes" : "no")
         << "  successful_boards=" << frame_result.SuccessfulBoardCount() << "/"
         << frame_result.board_measurements.size()
         << "  valid_internal=" << frame_result.ValidInternalCornerCount();
  cv::putText(*output_image, header.str(), cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.62,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

  std::ostringstream board_line;
  board_line << "visible=" << JoinBoardIds(frame_result.visible_board_ids);
  cv::putText(*output_image, board_line.str(), cv::Point(18, 53), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(180, 180, 180), 1, cv::LINE_AA);

  int x = 18;
  for (const RegeneratedBoardMeasurement& measurement : frame_result.board_measurements) {
    std::ostringstream token;
    token << "#" << measurement.board_id
          << " prior=" << (measurement.pose_prior_used ? "Y" : "N")
          << " int=" << measurement.detection.valid_internal_corner_count;
    cv::putText(*output_image, token.str(), cv::Point(x, 71), cv::FONT_HERSHEY_PLAIN, 1.0,
                measurement.detection.success ? cv::Scalar(100, 220, 120)
                                              : cv::Scalar(150, 150, 150),
                1, cv::LINE_AA);
    x += 145;
  }
}

const OuterBootstrapFrameState* MultiBoardInternalMeasurementRegenerator::FindFrameState(
    const OuterBootstrapResult& bootstrap_result,
    const InternalRegenerationFrameInput& frame_input) const {
  for (const OuterBootstrapFrameState& frame_state : bootstrap_result.frames) {
    if (frame_state.frame_index == frame_input.frame_index) {
      return &frame_state;
    }
  }
  for (const OuterBootstrapFrameState& frame_state : bootstrap_result.frames) {
    if (!frame_input.frame_label.empty() && frame_state.frame_label == frame_input.frame_label) {
      return &frame_state;
    }
  }
  return nullptr;
}

const OuterBootstrapBoardState* MultiBoardInternalMeasurementRegenerator::FindBoardState(
    const OuterBootstrapResult& bootstrap_result,
    int board_id) const {
  for (const OuterBootstrapBoardState& board_state : bootstrap_result.boards) {
    if (board_state.board_id == board_id) {
      return &board_state;
    }
  }
  return nullptr;
}

IntermediateCameraConfig MultiBoardInternalMeasurementRegenerator::MakeBootstrapCameraConfig(
    const OuterBootstrapCameraIntrinsics& intrinsics) const {
  return MakeSceneCameraConfig(intrinsics);
}

const JointSceneFrameState* MultiBoardInternalMeasurementRegenerator::FindFrameState(
    const JointReprojectionSceneState& scene_state,
    const InternalRegenerationFrameInput& frame_input) const {
  for (const JointSceneFrameState& frame_state : scene_state.frames) {
    if (frame_state.frame_index == frame_input.frame_index) {
      return &frame_state;
    }
  }
  for (const JointSceneFrameState& frame_state : scene_state.frames) {
    if (!frame_input.frame_label.empty() && frame_state.frame_label == frame_input.frame_label) {
      return &frame_state;
    }
  }
  return nullptr;
}

const JointSceneBoardState* MultiBoardInternalMeasurementRegenerator::FindBoardState(
    const JointReprojectionSceneState& scene_state,
    int board_id) const {
  for (const JointSceneBoardState& board_state : scene_state.boards) {
    if (board_state.board_id == board_id) {
      return &board_state;
    }
  }
  return nullptr;
}

IntermediateCameraConfig MultiBoardInternalMeasurementRegenerator::MakeSceneCameraConfig(
    const OuterBootstrapCameraIntrinsics& intrinsics) const {
  IntermediateCameraConfig config = config_.intermediate_camera;
  config.camera_model = intrinsics.camera_model;
  config.distortion_model = intrinsics.distortion_model;
  config.distortion_coeffs = intrinsics.DistortionVector();
  config.intrinsics = intrinsics.IntrinsicsVector();
  config.resolution = {intrinsics.resolution.width, intrinsics.resolution.height};
  return config;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
