#include <aslam/cameras/apriltag_internal/InternalPointResultValidation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

bool IsFinitePoint(const cv::Point2f& point) {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

bool IsInsideImageWithBorder(const cv::Point2f& point,
                             const cv::Size& image_size,
                             double border_distance) {
  return point.x >= border_distance &&
         point.x <= static_cast<float>(image_size.width) - border_distance &&
         point.y >= border_distance &&
         point.y <= static_cast<float>(image_size.height) - border_distance;
}

double PointDistancePx(const cv::Point2f& lhs, const cv::Point2f& rhs) {
  const double dx = static_cast<double>(lhs.x) - static_cast<double>(rhs.x);
  const double dy = static_cast<double>(lhs.y) - static_cast<double>(rhs.y);
  return std::sqrt(dx * dx + dy * dy);
}

double DuplicateCandidateScore(const InternalCornerDebugInfo& debug) {
  // Image-evidence quality is diagnostic only.  Prefer the candidate that is
  // most consistent with its own lattice prediction, then use the ray seed
  // quality as a deterministic tie breaker.
  double score = 0.0;
  if (std::isfinite(debug.predicted_to_refined_displacement)) {
    score -= std::max(0.0, debug.predicted_to_refined_displacement);
  }
  if (std::isfinite(debug.predicted_to_seed_displacement)) {
    score -= 0.25 * std::max(0.0, debug.predicted_to_seed_displacement);
  }
  if (std::isfinite(debug.sphere_seed_quality)) {
    score += 0.5 * std::max(0.0, std::min(1.0, debug.sphere_seed_quality));
  }
  return score;
}

}  // namespace

void RecomputeCornerCounts(ApriltagInternalDetectionResult* result) {
  if (result == nullptr) {
    return;
  }
  result->valid_corner_count = 0;
  result->valid_internal_corner_count = 0;
  result->runtime_breakdown.valid_internal_corner_count = 0;
  for (const CornerMeasurement& measurement : result->corners) {
    if (!measurement.valid) {
      continue;
    }
    ++result->valid_corner_count;
    if (measurement.corner_type != CornerType::Outer) {
      ++result->valid_internal_corner_count;
      ++result->runtime_breakdown.valid_internal_corner_count;
    }
  }
}

void EnforceInternalTopologyAssignment(ApriltagInternalDetectionResult* result) {
  if (result == nullptr) {
    return;
  }

  for (InternalCornerDebugInfo& debug : result->internal_corner_debug) {
    if (debug.point_id < 0 || debug.corner_type == CornerType::Outer ||
        static_cast<std::size_t>(debug.point_id) >= result->corners.size()) {
      continue;
    }

    CornerMeasurement& measurement =
        result->corners[static_cast<std::size_t>(debug.point_id)];
    const bool refinement_required = result->outer_detection.used_local_patch_rescue;
    const bool refined_ok =
        IsFinitePoint(debug.refined_image) &&
        IsInsideImageWithBorder(debug.refined_image, result->image_size, 0.0) &&
        (!refinement_required || debug.image_refinement_applied);
    if (refined_ok) {
      measurement.image_xy =
          Eigen::Vector2d(debug.refined_image.x, debug.refined_image.y);
      measurement.valid = true;
      debug.valid = true;
    } else {
      measurement.valid = false;
      measurement.quality = 0.0;
      debug.valid = false;
      debug.final_quality = 0.0;
    }
  }

  RecomputeCornerCounts(result);
}

void SuppressDuplicateRefinedInternalCorners(ApriltagInternalDetectionResult* result) {
  if (result == nullptr || result->internal_corner_debug.size() < 2) {
    return;
  }
  constexpr double kDuplicateRefinedCornerDistancePx = 2.0;
  constexpr double kDuplicateRefinedCornerDistance2 =
      kDuplicateRefinedCornerDistancePx * kDuplicateRefinedCornerDistancePx;

  std::vector<bool> suppress(result->internal_corner_debug.size(), false);
  constexpr double kDuplicateWinnerMargin = 0.05;
  for (std::size_t i = 0; i < result->internal_corner_debug.size(); ++i) {
    const InternalCornerDebugInfo& a = result->internal_corner_debug[i];
    if (!a.valid || a.point_id < 0) {
      continue;
    }
    for (std::size_t j = i + 1; j < result->internal_corner_debug.size(); ++j) {
      const InternalCornerDebugInfo& b = result->internal_corner_debug[j];
      if (!b.valid || b.point_id < 0 || a.point_id == b.point_id) {
        continue;
      }
      const double dx = static_cast<double>(a.refined_image.x - b.refined_image.x);
      const double dy = static_cast<double>(a.refined_image.y - b.refined_image.y);
      if (dx * dx + dy * dy <= kDuplicateRefinedCornerDistance2) {
        const double score_a = DuplicateCandidateScore(a);
        const double score_b = DuplicateCandidateScore(b);
        if (score_a > score_b + kDuplicateWinnerMargin) {
          suppress[j] = true;
        } else if (score_b > score_a + kDuplicateWinnerMargin) {
          suppress[i] = true;
        } else {
          // Keep one deterministic, lattice-consistent observation.  Dropping
          // both points turns a local duplicate into a board-wide false
          // invalidation and was the source of many red crosses.  The lower
          // point id is only the final deterministic tie breaker.
          if (a.point_id <= b.point_id) {
            suppress[j] = true;
          } else {
            suppress[i] = true;
          }
        }
      }
    }
  }

  for (std::size_t i = 0; i < suppress.size(); ++i) {
    if (!suppress[i]) {
      continue;
    }
    InternalCornerDebugInfo& debug = result->internal_corner_debug[i];
    if (debug.point_id < 0 ||
        static_cast<std::size_t>(debug.point_id) >= result->corners.size()) {
      continue;
    }
    CornerMeasurement& measurement =
        result->corners[static_cast<std::size_t>(debug.point_id)];
    if (!measurement.valid) {
      continue;
    }
    measurement.valid = false;
    measurement.quality = 0.0;
    debug.valid = false;
    debug.image_evidence_valid = false;
    debug.final_quality = 0.0;
  }
  RecomputeCornerCounts(result);
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
