#include <aslam/cameras/apriltag_internal/GeometryPriorOuterRecovery.hpp>
#include <aslam/cameras/apriltag_internal/GeometryPriorTopology.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "apriltags/TagDetection.h"
#include "apriltags/TagDetector.h"
#include "apriltags/TagFamily.h"
#include "apriltags/Tag36h11.h"

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

struct LocalCornerResponseEvidence {
  double refined_response = 0.0;
  double local_peak_response = 0.0;
  double peak_ratio = 0.0;
  bool passes = false;
};

LocalCornerResponseEvidence EvaluateLocalCornerResponseEvidence(
    const cv::Mat& gray, const cv::Point2f& corner, int subpix_radius) {
  LocalCornerResponseEvidence evidence;
  if (gray.empty()) return evidence;
  const int x = static_cast<int>(std::lround(corner.x));
  const int y = static_cast<int>(std::lround(corner.y));
  if (x < 0 || x >= gray.cols || y < 0 || y >= gray.rows) return evidence;

  const int response_radius =
      std::max(12, std::min(32, std::max(1, subpix_radius) / 3));
  const int margin = response_radius + 10;
  const int x0 = std::max(0, x - margin);
  const int y0 = std::max(0, y - margin);
  const int x1 = std::min(gray.cols, x + margin + 1);
  const int y1 = std::min(gray.rows, y + margin + 1);
  if (x1 - x0 < 20 || y1 - y0 < 20) return evidence;

  cv::Mat roi;
  gray(cv::Rect(x0, y0, x1 - x0, y1 - y0)).convertTo(roi, CV_32F);
  cv::Mat response;
  cv::cornerMinEigenVal(roi, response, 15, 3);
  const int cx = x - x0;
  const int cy = y - y0;
  for (int yy = std::max(0, cy - 2);
       yy <= std::min(response.rows - 1, cy + 2); ++yy) {
    for (int xx = std::max(0, cx - 2);
         xx <= std::min(response.cols - 1, cx + 2); ++xx) {
      evidence.refined_response = std::max(
          evidence.refined_response,
          static_cast<double>(response.at<float>(yy, xx)));
    }
  }
  for (int yy = std::max(0, cy - response_radius);
       yy <= std::min(response.rows - 1, cy + response_radius); ++yy) {
    for (int xx = std::max(0, cx - response_radius);
         xx <= std::min(response.cols - 1, cx + response_radius); ++xx) {
      evidence.local_peak_response = std::max(
          evidence.local_peak_response,
          static_cast<double>(response.at<float>(yy, xx)));
    }
  }
  evidence.peak_ratio = evidence.local_peak_response > 1e-9
                            ? evidence.refined_response /
                                  evidence.local_peak_response
                            : 0.0;
  constexpr double kReliableCornerPeakResponse = 8.0;
  constexpr double kMinimumPeakRatio = 0.20;
  evidence.passes = evidence.refined_response > 1e-9 &&
                    (evidence.local_peak_response <
                         kReliableCornerPeakResponse ||
                     evidence.peak_ratio >= kMinimumPeakRatio);
  return evidence;
}

}  // namespace

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

bool HasSufficientTopologyIdentityContext(
    int missing_board_id,
    const std::vector<int>& expected_board_ids,
    const std::vector<int>& visible_board_ids) {
  // The requested IDs belong to the immutable frame-level topology.  Do not
  // infer the expected set from a per-board detector config: sub-pipelines may
  // intentionally narrow that config to one board while recovering it.
  std::vector<int> expected_ids = expected_board_ids;
  std::sort(expected_ids.begin(), expected_ids.end());
  expected_ids.erase(std::unique(expected_ids.begin(), expected_ids.end()),
                     expected_ids.end());
  // Do not require exactly one missing board.  Two or more exact-ID boards
  // still provide sufficient same-frame topology context for assigning the
  // identity of a locally observed anonymous quad.  The image and pose gates
  // in the caller remain mandatory; this helper only enables that context.
  if (expected_ids.size() < 2 || visible_board_ids.size() < 2 ||
      std::find(expected_ids.begin(), expected_ids.end(),
                missing_board_id) == expected_ids.end() ||
      std::count(visible_board_ids.begin(), visible_board_ids.end(),
                 missing_board_id) != 0) {
    return false;
  }

  std::vector<int> unique_visible_ids = visible_board_ids;
  std::sort(unique_visible_ids.begin(), unique_visible_ids.end());
  if (std::adjacent_find(unique_visible_ids.begin(), unique_visible_ids.end()) !=
      unique_visible_ids.end()) {
    return false;
  }
  for (const int board_id : unique_visible_ids) {
    if (std::find(expected_ids.begin(), expected_ids.end(), board_id) ==
        expected_ids.end()) {
      return false;
    }
  }
  return true;
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

cv::Point2f RefineGeometryPriorCornerSubpixWithPadding(
    const cv::Mat& gray, const cv::Point2f& seed, int radius,
    const ApriltagInternalDetectionOptions& options) {
  if (gray.empty() || !std::isfinite(seed.x) || !std::isfinite(seed.y)) {
    return seed;
  }
  const int bounded_radius = std::max(2, radius);
  const int padding = bounded_radius + 2;
  const bool use_padding =
      options.outer_detector_config.enable_robust_missing_board_recovery &&
      (seed.x < padding || seed.y < padding ||
       seed.x > static_cast<float>(gray.cols - 1 - padding) ||
       seed.y > static_cast<float>(gray.rows - 1 - padding));
  cv::Mat padded;
  const cv::Mat* image = &gray;
  cv::Point2f point = seed;
  if (use_padding) {
    cv::copyMakeBorder(gray, padded, padding, padding, padding, padding,
                       cv::BORDER_REFLECT_101);
    image = &padded;
    point += cv::Point2f(static_cast<float>(padding),
                         static_cast<float>(padding));
  }
  std::vector<cv::Point2f> points{point};
  cv::cornerSubPix(
      *image, points, cv::Size(bounded_radius, bounded_radius), cv::Size(-1, -1),
      cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                       30, 0.1));
  return use_padding
             ? points.front() - cv::Point2f(static_cast<float>(padding),
                                            static_cast<float>(padding))
             : points.front();
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

// The direct AprilTag path records image support along each outer edge in
// corner_verification_debug.  Geometry/topology recovery used to commit only
// its four corners, which left the downstream spherical internal-point model
// without the edge rays it requires.  Sample the recovered edge on the camera
// sphere so a fisheye-projected straight board edge is followed as a curve in
// the image.  These points are evidence for the internal model only: they do
// not add a new acceptance gate for a recovered outer observation.
bool SlerpCameraRays(const Eigen::Vector3d& start_ray,
                     const Eigen::Vector3d& end_ray,
                     double alpha,
                     Eigen::Vector3d* interpolated_ray) {
  if (interpolated_ray == nullptr) {
    return false;
  }
  Eigen::Vector3d start = start_ray;
  Eigen::Vector3d end = end_ray;
  if (!NormalizeRay(&start) || !NormalizeRay(&end)) {
    return false;
  }

  alpha = std::max(0.0, std::min(1.0, alpha));
  double cosine = std::max(-1.0, std::min(1.0, start.dot(end)));
  if (cosine < 0.0) {
    end = -end;
    cosine = -cosine;
  }
  if (cosine > 0.9995) {
    *interpolated_ray = ((1.0 - alpha) * start + alpha * end).normalized();
    return true;
  }

  const double theta = std::acos(cosine);
  const double sin_theta = std::sin(theta);
  if (!std::isfinite(theta) || !std::isfinite(sin_theta) ||
      std::abs(sin_theta) <= 1e-9) {
    return false;
  }
  *interpolated_ray =
      (std::sin((1.0 - alpha) * theta) / sin_theta) * start +
      (std::sin(alpha * theta) / sin_theta) * end;
  return NormalizeRay(interpolated_ray);
}

std::array<std::vector<cv::Point2f>, 4> CollectRecoveredEdgeSupportPoints(
    const cv::Mat& gray,
    const IntermediateCameraConfig& camera_config,
    const ApriltagInternalDetectionOptions& options,
    const std::array<cv::Point2f, 4>& corners) {
  std::array<std::vector<cv::Point2f>, 4> edge_supports{};
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(camera_config);
  if (gray.empty() || !camera.IsValid()) {
    return edge_supports;
  }

  cv::Mat grad_x;
  cv::Mat grad_y;
  cv::Mat grad_mag;
  cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
  cv::magnitude(grad_x, grad_y, grad_mag);
  double gradient_max = 0.0;
  cv::minMaxLoc(grad_mag, nullptr, &gradient_max);
  const double gradient_norm = std::max(1e-12, gradient_max);

  const int sample_count = std::max(
      8, options.geometry_prior_rescue_edge_sample_count);
  const int search_half_width = std::max(
      1, options.geometry_prior_rescue_edge_search_half_width_px);
  const double min_gradient_ratio = std::max(
      0.0, options.geometry_prior_rescue_min_edge_gradient_ratio);
  const double tangent_delta = std::max(
      1.0 / static_cast<double>(sample_count), 0.002);

  for (int edge_index = 0; edge_index < 4; ++edge_index) {
    const cv::Point2f& edge_start =
        corners[static_cast<std::size_t>(edge_index)];
    const cv::Point2f& edge_end =
        corners[static_cast<std::size_t>((edge_index + 1) % 4)];
    Eigen::Vector3d start_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d end_ray = Eigen::Vector3d::Zero();
    if (!camera.keypointToEuclidean(
            Eigen::Vector2d(edge_start.x, edge_start.y), &start_ray) ||
        !camera.keypointToEuclidean(
            Eigen::Vector2d(edge_end.x, edge_end.y), &end_ray) ||
        !NormalizeRay(&start_ray) || !NormalizeRay(&end_ray)) {
      continue;
    }

    std::vector<cv::Point2f>& supports =
        edge_supports[static_cast<std::size_t>(edge_index)];
    supports.reserve(static_cast<std::size_t>(sample_count));
    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
      const double t = (static_cast<double>(sample_index) + 0.5) /
                       static_cast<double>(sample_count);
      Eigen::Vector3d ray = Eigen::Vector3d::Zero();
      Eigen::Vector3d before_ray = Eigen::Vector3d::Zero();
      Eigen::Vector3d after_ray = Eigen::Vector3d::Zero();
      if (!SlerpCameraRays(start_ray, end_ray, t, &ray) ||
          !SlerpCameraRays(start_ray, end_ray, std::max(0.0, t - tangent_delta),
                           &before_ray) ||
          !SlerpCameraRays(start_ray, end_ray, std::min(1.0, t + tangent_delta),
                           &after_ray)) {
        continue;
      }
      Eigen::Vector2d center_eigen;
      Eigen::Vector2d before_eigen;
      Eigen::Vector2d after_eigen;
      if (!camera.vsEuclideanToKeypoint(ray, &center_eigen) ||
          !camera.vsEuclideanToKeypoint(before_ray, &before_eigen) ||
          !camera.vsEuclideanToKeypoint(after_ray, &after_eigen)) {
        continue;
      }
      const cv::Point2f center(static_cast<float>(center_eigen.x()),
                               static_cast<float>(center_eigen.y()));
      const cv::Point2f tangent(
          static_cast<float>(after_eigen.x() - before_eigen.x()),
          static_cast<float>(after_eigen.y() - before_eigen.y()));
      const double tangent_norm = std::hypot(tangent.x, tangent.y);
      if (!IsInsideImage(center, gray.size(), 1.0) ||
          !std::isfinite(tangent_norm) || tangent_norm <= 1e-6) {
        continue;
      }
      const cv::Point2f normal(-tangent.y / static_cast<float>(tangent_norm),
                               tangent.x / static_cast<float>(tangent_norm));
      double best_ratio = 0.0;
      cv::Point2f best_point = center;
      for (int offset = -search_half_width; offset <= search_half_width; ++offset) {
        const cv::Point2f probe =
            center + static_cast<float>(offset) * normal;
        if (!IsInsideImage(probe, gray.size(), 1.0)) {
          continue;
        }
        const double ratio = SampleFloatAt(grad_mag, probe) / gradient_norm;
        if (ratio > best_ratio) {
          best_ratio = ratio;
          best_point = probe;
        }
      }
      if (best_ratio >= min_gradient_ratio) {
        supports.push_back(best_point);
      }
    }
  }
  return edge_supports;
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

OuterTagDetectionResult BuildRescuedOuterDetection(
    int board_id,
    const cv::Mat& gray,
    const IntermediateCameraConfig& camera_config,
    const ApriltagInternalDetectionOptions& options,
    const cv::Size& image_size,
    const std::array<cv::Point2f, 4>& coarse_corners,
    const std::array<cv::Point2f, 4>& refined_corners,
    int subpix_window_radius,
    bool tag_id_validated,
    bool topology_identity_assigned,
    double quality,
    const std::string& summary) {
  OuterTagDetectionResult detection;
  detection.success = true;
  detection.board_id = board_id;
  // Geometry may establish a unique physical board slot, but it is not a
  // decoded payload. Preserve that distinction with hamming=-1.
  detection.detected_tag_id =
      (tag_id_validated || topology_identity_assigned) ? board_id : -1;
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
  std::array<std::vector<cv::Point2f>, 4> edge_supports{};
  if (options.geometry_prior_rescue_enable_spherical_refine) {
    edge_supports = CollectRecoveredEdgeSupportPoints(
        gray, camera_config, options, refined_corners);
  }
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f& corner = refined_corners[static_cast<std::size_t>(index)];
    const Eigen::Vector2d eigen_corner(corner.x, corner.y);
    const cv::Point2f& coarse_corner =
        coarse_corners[static_cast<std::size_t>(index)];
    detection.coarse_corners_scaled_image[static_cast<std::size_t>(index)] =
        Eigen::Vector2d(coarse_corner.x, coarse_corner.y);
    detection.coarse_corners_original_image[static_cast<std::size_t>(index)] =
        Eigen::Vector2d(coarse_corner.x, coarse_corner.y);
    detection.refined_corners_original_image[static_cast<std::size_t>(index)] =
        eigen_corner;
    detection.refined_valid[static_cast<std::size_t>(index)] = true;

    OuterCornerVerificationDebugInfo& debug =
        detection.corner_verification_debug[static_cast<std::size_t>(index)];
    const cv::Point2f& previous_corner =
        refined_corners[static_cast<std::size_t>((index + 3) % 4)];
    const cv::Point2f& next_corner =
        refined_corners[static_cast<std::size_t>((index + 1) % 4)];
    debug.corner_index = index;
    debug.coarse_corner = coarse_corner;
    debug.verified_corner = corner;
    debug.subpix_corner = corner;
    debug.prev_edge_direction = previous_corner - corner;
    debug.next_edge_direction = next_corner - corner;
    debug.prev_marker_support_points =
        edge_supports[static_cast<std::size_t>((index + 3) % 4)];
    debug.next_marker_support_points =
        edge_supports[static_cast<std::size_t>(index)];
    debug.local_scale = CornerLocalScale(refined_corners, index);
    debug.corner_marker_width =
        options.outer_detector_config.outer_corner_marker_ratio > 0.0
            ? options.outer_detector_config.outer_corner_marker_ratio *
                  debug.local_scale
            : debug.local_scale;
    debug.subpix_window_radius = std::max(2, subpix_window_radius);
    debug.subpix_applied = true;
    debug.coarse_to_subpix_displacement =
        PointDistance(coarse_corner, corner);
    debug.coarse_to_refined_displacement =
        debug.coarse_to_subpix_displacement;
    debug.verification_quality = quality;
    debug.refined_valid = true;
    debug.verification_passed = true;
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
    const std::vector<int>& expected_board_ids,
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
      options.outer_detector_config.enable_robust_missing_board_recovery
          ? 0.0
          : static_cast<double>(std::max(0, candidate.subpix_window_radius)) + 2.0;
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
    candidate.spherical_refine_success =
        spherical_refinement.joint_fit_success;
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
    if (spherical_refinement.joint_fit_success) {
      initial_corners = spherical_refinement.refined_corners;
    }
  } else {
    candidate.spherical_refine_failure_summary =
        options.geometry_prior_rescue_enable_spherical_refine
            ? "invalid_camera"
            : "disabled";
  }

  if (options.geometry_prior_rescue_enable_spherical_refine &&
      (!candidate.spherical_refine_attempted ||
       !candidate.spherical_refine_success ||
       candidate.spherical_refine_successful_corner_count != 4)) {
    candidate.reject_reason =
        validation_source + "_image_evidence_failed_incomplete_spherical_refinement";
    return candidate;
  }

  // A successful joint fit supplies a camera-aware quad seed, but it is not a
  // substitute for the final image-domain subpixel measurement.  Refine that
  // complete quad with the same cornerSubPix stage used by ordinary image
  // observations. Partial spherical corners are never mixed in or committed.
  std::vector<cv::Point2f> refined_points;
  refined_points.reserve(initial_corners.size());
  for (const cv::Point2f& point : initial_corners) {
    refined_points.push_back(RefineGeometryPriorCornerSubpixWithPadding(
        gray, point, candidate.subpix_window_radius, options));
  }
  std::array<cv::Point2f, 4> refined_corners{};
  double max_prediction_displacement = 0.0;
  double max_refinement_displacement = 0.0;
  for (int index = 0; index < 4; ++index) {
    refined_corners[static_cast<std::size_t>(index)] =
        refined_points[static_cast<std::size_t>(index)];
    candidate.refined_corners[static_cast<std::size_t>(index)] =
        refined_corners[static_cast<std::size_t>(index)];
    max_prediction_displacement = std::max(
        max_prediction_displacement,
        PointDistance(refined_corners[static_cast<std::size_t>(index)],
                      candidate.predicted_corners[static_cast<std::size_t>(index)]));
    max_refinement_displacement = std::max(
        max_refinement_displacement,
        PointDistance(refined_corners[static_cast<std::size_t>(index)],
                      initial_corners_input[static_cast<std::size_t>(index)]));
  }
  for (const cv::Point2f& corner : refined_corners) {
    if (!std::isfinite(corner.x) || !std::isfinite(corner.y) ||
        !IsInsideImage(corner, gray.size(), 0.0)) {
      candidate.local_corner_refine_success = false;
      candidate.reject_reason =
          validation_source + "_image_evidence_failed_refined_corner_outside_image";
      return candidate;
    }
  }
  candidate.max_corner_displacement_px = max_prediction_displacement;
  candidate.max_refinement_displacement_px = max_refinement_displacement;
  // Refinement displacement is diagnostic only. A valid image-space edge or
  // decoded-quad observation can legitimately move far from a coarse seed;
  // acceptance is decided by image evidence, quad topology, ID likelihood,
  // and the downstream pose-consistency checks.

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
  candidate.refined_corner_response_pass_count = 0;
  candidate.min_refined_corner_local_peak_ratio = 1.0;
  for (int index = 0; index < 4; ++index) {
    const LocalCornerResponseEvidence evidence =
        EvaluateLocalCornerResponseEvidence(
            gray, refined_corners[static_cast<std::size_t>(index)],
            candidate.subpix_window_radius);
    candidate.refined_corner_local_responses[static_cast<std::size_t>(index)] =
        evidence.refined_response;
    candidate.refined_corner_local_peak_responses[
        static_cast<std::size_t>(index)] = evidence.local_peak_response;
    candidate.refined_corner_local_peak_ratios[static_cast<std::size_t>(index)] =
        evidence.peak_ratio;
    candidate.min_refined_corner_local_peak_ratio = std::min(
        candidate.min_refined_corner_local_peak_ratio, evidence.peak_ratio);
    if (evidence.passes) ++candidate.refined_corner_response_pass_count;
  }
  std::array<double, 4> ordered_local_responses =
      candidate.refined_corner_local_responses;
  std::sort(ordered_local_responses.begin(), ordered_local_responses.end());
  candidate.weakest_to_second_weakest_corner_response_ratio =
      ordered_local_responses[1] > 1e-9
          ? ordered_local_responses[0] / ordered_local_responses[1]
          : 0.0;
  // Geometry recovery must provide four independently supported physical
  // corners. An occluder can replace one board edge and still produce a
  // successful 4/4 spherical line fit; in that case one final intersection
  // is an isolated response outlier. Compare within the quad so uniformly
  // weak, highly distorted but visible boards are retained.
  constexpr double kMinimumAtomicCornerResponseBalance = 0.25;
  if (candidate.weakest_to_second_weakest_corner_response_ratio <
      kMinimumAtomicCornerResponseBalance) {
    candidate.local_corner_refine_success = false;
    candidate.reject_reason =
        validation_source +
        "_image_evidence_failed_atomic_corner_response_balance";
    return candidate;
  }
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
      candidate.spherical_refine_success;
  const bool geometry_only_observation_refine_ok =
      tag_id_validated ||
      (!options.geometry_prior_rescue_enable_spherical_refine ||
       candidate.spherical_refine_success);
  const bool geometry_only_observation_edge_ok =
      tag_id_validated || edge_evidence_ok;
  const bool multi_board_context =
      static_cast<int>(candidate.visible_boards_used.size()) >=
      std::max(2, options.geometry_guided_tag_likelihood_min_visible_boards);
  // Same-frame rigid topology may identify a locally observed, pose-consistent
  // quad even when more than one payload is undecodable. The image, quad, and
  // pose checks below remain mandatory; this never turns projected geometry
  // alone into an observation.
  const bool unique_topology_context = HasSufficientTopologyIdentityContext(
      board_id, expected_board_ids, candidate.visible_boards_used);
  const bool topology_visible_refit =
      candidate.prediction_source_label.find("visible_refit") !=
      std::string::npos;
  const bool topology_pose_ok =
      candidate.frame_pose_refit_source_board_id >= 0 &&
      std::isfinite(candidate.frame_pose_refit_outer_rmse) &&
      candidate.frame_pose_refit_outer_rmse <=
          options.geometry_prior_rescue_accept_max_outer_rmse;
  const bool topology_image_ok = corner_response_ok || edge_evidence_ok;
  const bool locally_observed_quad =
      candidate.roi_redetect_success ||
      candidate.rectified_patch_decode_success ||
      candidate.spherical_refine_success ||
      geometry_guided_edge_quad ||
      weak_quad_alternative;
  const bool topology_identity_context =
      geometry_only_pose_refit_candidate && unique_topology_context &&
      topology_visible_refit && topology_pose_ok && topology_image_ok &&
      locally_observed_quad &&
      // A visible-refit pose only supplies the geometric slot prediction.  It
      // does not establish the missing board's identity.  That identity is
      // established only after an observed wrong-ID quad has passed the
      // explicit topology association step (which appends _topology_assoc to
      // the prediction source label).
      candidate.prediction_source_label.find("_topology_assoc") !=
          std::string::npos &&
      // An edge/weak quad is only a geometric image candidate.  It has not
      // established the payload identity, so it must still pass the
      // model-aware tag-likelihood check below.  Otherwise a failed exact-ID
      // fallback could be promoted solely by projected topology, which is
      // precisely the failure mode seen for recovered B5 in frame 31.
      !geometry_guided_edge_quad && !weak_quad_alternative;
  const bool single_anchor_context =
      options.geometry_guided_tag_likelihood_allow_single_anchor &&
      candidate.visible_boards_used.size() == 1u &&
      single_anchor_is_direct_exact &&
      candidate.frame_pose_refit_source_board_id >= 0 &&
      std::isfinite(candidate.frame_pose_refit_outer_rmse) &&
      // At the periphery, fitting a pose to the four outer corners of the
      // only visible anchor is intrinsically less accurate than a same-frame
      // multi-board fit.  Do not discard a candidate solely because that
      // diagnostic is above the legacy 0.50 px threshold: it remains subject
      // to independent quad, edge, payload, and global-pose consistency gates
      // below.  Keep a finite, bounded sanity limit tied to the ordinary
      // geometry-recovery acceptance threshold.
      candidate.frame_pose_refit_outer_rmse <=
          std::max(options.geometry_prior_rescue_accept_max_outer_rmse,
                   options.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse);
  const bool require_model_aware_tag_likelihood =
      !tag_id_validated &&
      options.geometry_guided_tag_likelihood_enabled &&
      !topology_identity_context &&
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
      !topology_identity_context &&
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
      if (!multi_board_context_high_confidence && !topology_identity_context) {
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
  candidate.reject_reason =
      topology_identity_context
          ? "accepted_unique_topology_identity_observation"
          : (candidate.geometry_guided_tag_likelihood_passed
                 ? "accepted_geometry_guided_tag_likelihood_observation"
                 : (tag_id_validated
                        ? "accepted_image_validated_rescued_observation"
                        : "accepted_geometry_only_pose_refit_observation"));
  if (!validation_source.empty() && validation_source != "primary") {
    candidate.reject_reason += "_" + validation_source;
  }
  if (rescued_detection != nullptr) {
    std::ostringstream summary;
    summary << "geometry_guided_refine"
            << " validation="
            << (topology_identity_context
                    ? "unique_topology_identity"
                    : (candidate.geometry_guided_tag_likelihood_passed
                    ? (topology_identity_context
                           ? "topology_id_plus_model_aware_tag"
                           : "model_aware_tag")
                    : (tag_id_validated ? "decoded_tag_id"
                                        : "geometry_only")))
            << " source=" << validation_source
            << " prediction_disp=" << max_prediction_displacement
            << " refinement_disp=" << max_refinement_displacement
            << " min_corner_response_ratio=" << min_response_ratio
            << " edge_support_ratio=" << edge_metrics.support_ratio
            << " mean_edge_gradient_ratio=" << edge_metrics.mean_gradient_ratio
            << " outer_rmse=" << outer_rmse
            << " rot_err_deg="
            << candidate.local_vs_global_rotation_error_deg
            << " trans_err="
            << candidate.local_vs_global_translation_error;
    *rescued_detection = BuildRescuedOuterDetection(
        board_id, gray, camera_config, options, gray.size(), initial_corners,
        refined_corners, candidate.subpix_window_radius,
        tag_id_validated, topology_identity_context,
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

  const double required_border =
      options.outer_detector_config.enable_robust_missing_board_recovery
          ? 0.0
          : static_cast<double>(window_radius + 2);
  for (const cv::Point2f& point : candidate.predicted_corners) {
    if (!IsInsideImage(point, gray.size(), required_border)) {
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

  // A context override may return a geometrically compatible quad decoded as
  // another ID.  It is useful image evidence, but it is not exact-ID
  // validation and must be handled by topology identity association below.
  const bool roi_exact_id =
      candidate.roi_redetect_success &&
      candidate.roi_redetect_detected_tag_id == board_id;
  const bool rectified_exact_id =
      candidate.rectified_patch_decode_success &&
      candidate.rectified_patch_detected_tag_id == board_id;
  const bool tag_id_validated = roi_exact_id || rectified_exact_id;
  const bool single_anchor_is_direct_exact =
      HasDirectExactOuterAnchor(frame_input, frame_pose_refit_source_board_id);
  GeometryPriorOuterSeedCandidate primary_candidate =
      FinalizeGeometryPriorOuterSeedCandidate(
          gray, config, options, camera_config, board_id,
          frame_input.outer_detections.requested_board_ids, initial_corners,
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
            gray, config, options, camera_config, board_id,
            frame_input.outer_detections.requested_board_ids, guided_edge_corners,
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
          gray, config, options, camera_config, board_id,
          frame_input.outer_detections.requested_board_ids, weak_quad_corners,
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

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
