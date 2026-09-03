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
    const bool recovery_board = result->outer_detection.used_local_patch_rescue;
    const bool refined_ok =
        IsFinitePoint(debug.refined_image) &&
        IsInsideImageWithBorder(debug.refined_image, result->image_size, 0.0) &&
        (!recovery_board || debug.image_refinement_applied);
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

void SuppressWrongLatticeSlotAssignments(
    ApriltagInternalDetectionResult* result) {
  if (result == nullptr || result->internal_corner_debug.size() < 2) {
    return;
  }

  constexpr double kMinOwnSlotDistanceModules = 0.75;
  constexpr double kMaxOtherSlotDistanceModules = 0.25;
  constexpr double kMinOwnershipMarginModules = 0.50;
  std::vector<bool> suppress(result->internal_corner_debug.size(), false);
  for (std::size_t i = 0; i < result->internal_corner_debug.size(); ++i) {
    const InternalCornerDebugInfo& candidate = result->internal_corner_debug[i];
    if (!candidate.valid || candidate.point_id < 0 ||
        candidate.corner_type == CornerType::Outer ||
        candidate.local_module_scale <= 1.0 ||
        !IsFinitePoint(candidate.predicted_image) ||
        !IsFinitePoint(candidate.refined_image)) {
      continue;
    }

    const double own_distance =
        PointDistancePx(candidate.refined_image, candidate.predicted_image);
    const double scale = candidate.local_module_scale;
    if (own_distance < kMinOwnSlotDistanceModules * scale) {
      continue;
    }

    double nearest_other_distance = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < result->internal_corner_debug.size(); ++j) {
      if (i == j) {
        continue;
      }
      const InternalCornerDebugInfo& other = result->internal_corner_debug[j];
      if (other.point_id < 0 || other.point_id == candidate.point_id ||
          other.corner_type == CornerType::Outer ||
          !IsFinitePoint(other.predicted_image) ||
          !IsInsideImageWithBorder(other.predicted_image, result->image_size,
                                   1.0)) {
        continue;
      }
      nearest_other_distance = std::min(
          nearest_other_distance,
          PointDistancePx(candidate.refined_image, other.predicted_image));
    }

    if (nearest_other_distance <= kMaxOtherSlotDistanceModules * scale &&
        own_distance - nearest_other_distance >=
            kMinOwnershipMarginModules * scale) {
      suppress[i] = true;
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
    measurement.valid = false;
    measurement.quality = 0.0;
    debug.valid = false;
    debug.image_evidence_valid = false;
    debug.final_quality = 0.0;
  }
  RecomputeCornerCounts(result);
}

void SuppressDuplicateRefinedInternalCorners(ApriltagInternalDetectionResult* result) {
  if (result == nullptr || result->internal_corner_debug.size() < 2) {
    return;
  }
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
      const double pair_module_scale =
          a.local_module_scale > 1.0 && b.local_module_scale > 1.0
              ? std::min(a.local_module_scale, b.local_module_scale)
              : std::max(a.local_module_scale, b.local_module_scale);
      // Duplicate identity is a lattice-space property, not a native-pixel
      // property.  Use a small fraction of the locally projected module so
      // the same physical coincidence is detected at every image resolution.
      const double duplicate_distance =
          pair_module_scale > 1.0 ? 0.08 * pair_module_scale : 2.0;
      if (dx * dx + dy * dy <= duplicate_distance * duplicate_distance) {
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

void SuppressLocallyInconsistentRecoveredCorners(
    ApriltagInternalDetectionResult* result) {
  if (result == nullptr || !result->outer_detection.used_local_patch_rescue) {
    return;
  }
  std::vector<bool> suppress(result->internal_corner_debug.size(), false);
  constexpr double kMaxNormalizedGridResidual = 0.45;
  constexpr double kEvidenceWeakThreshold = 0.20;
  for (std::size_t i = 0; i < result->internal_corner_debug.size(); ++i) {
    const InternalCornerDebugInfo& candidate = result->internal_corner_debug[i];
    if (!candidate.valid || candidate.corner_type == CornerType::Outer ||
        candidate.lattice_u < 0 || candidate.lattice_v < 0 ||
        candidate.local_module_scale <= 1.0 || candidate.image_evidence_valid) {
      continue;
    }
    const InternalCornerDebugInfo* left = nullptr;
    const InternalCornerDebugInfo* right = nullptr;
    const InternalCornerDebugInfo* up = nullptr;
    const InternalCornerDebugInfo* down = nullptr;
    for (const InternalCornerDebugInfo& other : result->internal_corner_debug) {
      if (!other.valid || other.corner_type == CornerType::Outer) continue;
      if (other.lattice_v == candidate.lattice_v) {
        if (other.lattice_u < candidate.lattice_u &&
            (left == nullptr || other.lattice_u > left->lattice_u)) {
          left = &other;
        }
        if (other.lattice_u > candidate.lattice_u &&
            (right == nullptr || other.lattice_u < right->lattice_u)) {
          right = &other;
        }
      }
      if (other.lattice_u == candidate.lattice_u) {
        if (other.lattice_v < candidate.lattice_v &&
            (up == nullptr || other.lattice_v > up->lattice_v)) {
          up = &other;
        }
        if (other.lattice_v > candidate.lattice_v &&
            (down == nullptr || other.lattice_v < down->lattice_v)) {
          down = &other;
        }
      }
    }
    if (left == nullptr || right == nullptr || up == nullptr || down == nullptr) {
      continue;
    }
    const double row_alpha =
        static_cast<double>(candidate.lattice_u - left->lattice_u) /
        static_cast<double>(right->lattice_u - left->lattice_u);
    const double col_alpha =
        static_cast<double>(candidate.lattice_v - up->lattice_v) /
        static_cast<double>(down->lattice_v - up->lattice_v);
    const cv::Point2f row_mid =
        left->refined_image + static_cast<float>(row_alpha) *
                                  (right->refined_image - left->refined_image);
    const cv::Point2f col_mid =
        up->refined_image + static_cast<float>(col_alpha) *
                                (down->refined_image - up->refined_image);
    const double row_residual = PointDistancePx(candidate.refined_image, row_mid) /
                                candidate.local_module_scale;
    const double col_residual = PointDistancePx(candidate.refined_image, col_mid) /
                                candidate.local_module_scale;
    if (row_residual > kMaxNormalizedGridResidual &&
        col_residual > kMaxNormalizedGridResidual &&
        candidate.image_final_quality < kEvidenceWeakThreshold) {
      suppress[i] = true;
    }
  }
  for (std::size_t i = 0; i < suppress.size(); ++i) {
    if (!suppress[i]) continue;
    InternalCornerDebugInfo& debug = result->internal_corner_debug[i];
    if (debug.point_id < 0 ||
        static_cast<std::size_t>(debug.point_id) >= result->corners.size()) continue;
    CornerMeasurement& measurement =
        result->corners[static_cast<std::size_t>(debug.point_id)];
    measurement.valid = false;
    measurement.quality = 0.0;
    debug.valid = false;
    debug.image_evidence_valid = false;
    debug.final_quality = 0.0;
  }
  RecomputeCornerCounts(result);
}

void SuppressZeroImageEvidenceRecoveredCorners(
    ApriltagInternalDetectionResult* result) {
  if (result == nullptr || !result->outer_detection.used_local_patch_rescue) {
    return;
  }
  for (InternalCornerDebugInfo& debug : result->internal_corner_debug) {
    if (!debug.valid || debug.corner_type == CornerType::Outer ||
        (std::isfinite(debug.image_final_quality) &&
         debug.image_final_quality > 0.0)) {
      continue;
    }
    if (debug.point_id < 0 ||
        static_cast<std::size_t>(debug.point_id) >= result->corners.size()) {
      continue;
    }
    CornerMeasurement& measurement =
        result->corners[static_cast<std::size_t>(debug.point_id)];
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
