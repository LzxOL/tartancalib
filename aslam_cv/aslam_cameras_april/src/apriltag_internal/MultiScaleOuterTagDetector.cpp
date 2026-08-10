#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <opencv2/imgproc.hpp>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

#include "apriltags/TagFamily.h"
#include "apriltags/Tag36h11.h"
#include "apriltags/TagDetection.h"
#include "apriltags/TagDetector.h"

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

constexpr double kMinQuadAreaPixels = 64.0;
constexpr double kMinQuadEdgePixels = 8.0;
constexpr double kOuterLineDerivativeDelta = 1.5;
constexpr double kOuterLineResidualThreshold = 2.5;
constexpr int kMinLineSupportPoints = 6;
constexpr double kOuterLineMinQuality = 0.25;
constexpr double kOuterSpherePlaneResidualThreshold = 0.035;
constexpr int kOuterSphereMinSupportPoints = 4;
constexpr int kVerificationLineMinSupportPoints = 4;
constexpr int kOuterVerificationCandidateStepPixels = 2;
constexpr double kOuterDirectionAlignmentFloor = 0.75;
constexpr double kOuterLayoutContrastFloor = 8.0;
constexpr double kOuterLayoutContrastRange = 40.0;
constexpr int kOuterContextRadiusMin = 12;
constexpr int kOuterContextRadiusMax = 48;
constexpr int kOuterSubpixRadiusMin = 4;
constexpr int kOuterSubpixMaxIterations = 30;
constexpr double kOuterSubpixEpsilon = 0.1;
constexpr double kOuterSubpixNoMotionEpsilon = 1e-4;
constexpr double kOuterSubpixRollbackProbeFraction = 0.5;
constexpr double kPi = 3.14159265358979323846;
constexpr double kOuterFixedScaleDivisors[] = {1.0, 1.5, 2.0, 2.5, 3.5, 4.5, 6.0, 8.0, 12.0};
constexpr int kOuterLocalSpherePatchSize = 640;

struct ScaleCandidate {
  int target_longest_side = 0;
  double scale_factor = 1.0;
  double configured_scale_divisor = 0.0;
  cv::Size scaled_size;
  AprilTags::TagDetection detection;
  std::array<cv::Point2f, 4> scaled_corners{};
  std::array<cv::Point2f, 4> original_corners{};
  double scaled_area = 0.0;
  double min_edge = 0.0;
  double max_edge = 0.0;
  double shape_quality = 0.0;
  bool from_local_patch_rescue = false;
  std::string local_patch_label;
  double local_patch_center_distance =
      std::numeric_limits<double>::infinity();
};

struct RefinedCandidate {
  ScaleCandidate coarse;
  std::array<cv::Point2f, 4> coarse_original{};
  std::array<cv::Point2f, 4> refined_original{};
  std::array<bool, 4> refined_valid{{false, false, false, false}};
  std::array<OuterCornerVerificationDebugInfo, 4> verification_debug{};
  double refine_quality = 0.0;
  double quality = 0.0;
};

struct CornerSubpixResult {
  cv::Point2f refined_corner{};
  bool unstable_rollback_detected = false;
  int rollback_iteration = 0;
  double max_probe_displacement = 0.0;
};

CornerSubpixResult RefineCornerSubpixWithRollbackCheck(
    const cv::Mat& gray,
    const cv::Point2f& seed,
    int radius,
    bool check_unstable_rollback) {
  CornerSubpixResult result;
  result.refined_corner = seed;
  const cv::Size window(std::max(2, radius), std::max(2, radius));
  const cv::TermCriteria criteria(
      cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
      kOuterSubpixMaxIterations, kOuterSubpixEpsilon);
  std::vector<cv::Point2f> final_point{seed};
  cv::cornerSubPix(gray, final_point, window, cv::Size(-1, -1), criteria);
  result.refined_corner = final_point.front();
  const auto point_distance = [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
    return std::hypot(static_cast<double>(lhs.x - rhs.x),
                      static_cast<double>(lhs.y - rhs.y));
  };

  // OpenCV restores the original seed when its final iterate leaves the
  // search window. Its API does not expose that status. Probe only the
  // high-polar boosted path, and only after a zero-displacement result, so
  // ordinary already-converged corners stay untouched.
  if (!check_unstable_rollback ||
      point_distance(result.refined_corner, seed) > kOuterSubpixNoMotionEpsilon) {
    return result;
  }

  bool saw_large_excursion = false;
  for (int iteration = 1; iteration <= kOuterSubpixMaxIterations; ++iteration) {
    std::vector<cv::Point2f> probe_point{seed};
    const cv::TermCriteria probe_criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
        iteration, kOuterSubpixEpsilon);
    cv::cornerSubPix(gray, probe_point, window, cv::Size(-1, -1), probe_criteria);
    const double displacement = point_distance(probe_point.front(), seed);
    result.max_probe_displacement = std::max(result.max_probe_displacement, displacement);
    saw_large_excursion =
        saw_large_excursion ||
        displacement > kOuterSubpixRollbackProbeFraction * static_cast<double>(radius);
    if (iteration > 1 && saw_large_excursion &&
        displacement <= kOuterSubpixNoMotionEpsilon) {
      result.unstable_rollback_detected = true;
      result.rollback_iteration = iteration;
      break;
    }
  }
  return result;
}

struct FittedLine {
  bool valid = false;
  cv::Point2f anchor;
  cv::Point2f direction;
  double rms_residual = std::numeric_limits<double>::infinity();
  int support_count = 0;
};

struct DirectionalEdgeBranch {
  std::vector<cv::Point2f> support_points;
  FittedLine fitted_line;
  double score = 0.0;
  bool valid = false;
};

struct CornerLineRefinement {
  bool success = false;
  cv::Point2f refined_corner;
  double quality = 0.0;
};

struct SphericalEdgePlaneFit {
  std::vector<cv::Point2f> support_points;
  std::vector<Eigen::Vector3d> support_rays;
  Eigen::Vector3d plane_normal = Eigen::Vector3d::Zero();
  double rms_residual = std::numeric_limits<double>::infinity();
  int support_count = 0;
  bool valid = false;
};

struct SphericalCornerRefinement {
  bool success = false;
  cv::Point2f refined_corner{};
  Eigen::Vector3d refined_ray = Eigen::Vector3d::Zero();
  SphericalEdgePlaneFit prev_edge_fit;
  SphericalEdgePlaneFit next_edge_fit;
  std::vector<cv::Point2f> prev_curve_points;
  std::vector<cv::Point2f> next_curve_points;
  double quality = 0.0;
  std::string failure_reason;
};

struct ImageLineCornerRefinement {
  bool success = false;
  cv::Point2f refined_corner{};
  FittedLine prev_line;
  FittedLine next_line;
  double quality = 0.0;
  std::string failure_reason;
};

struct OuterCornerLocalVerificationResult {
  cv::Point2f verified_corner;
  cv::Rect verification_roi;
  cv::Point2f prev_edge_direction{};
  cv::Point2f next_edge_direction{};
  DirectionalEdgeBranch prev_branch;
  DirectionalEdgeBranch next_branch;
  double local_scale = 0.0;
  double corner_marker_width = 0.0;
  int verification_roi_radius = 0;
  int candidate_radius = 0;
  int branch_search_radius = 0;
  double direction_consistency_score = 0.0;
  double local_layout_score = 0.0;
  double verification_quality = 0.0;
  bool verification_passed = false;
  std::string failure_reason;
};

struct AdaptiveCornerSearchRadii {
  double local_scale = 0.0;
  int verification_roi_radius = 0;
  int candidate_radius = 0;
  int branch_search_radius = 0;
};

struct ScalePlanEntry {
  int target_longest_side = 0;
  double configured_scale_divisor = 0.0;
};

struct LocalSpherePatchPlan {
  std::string label;
  double normalized_x = 0.5;
  double normalized_y = 0.5;
  double fov_deg = 44.0;
};

struct LocalSpherePatchContext {
  std::string label;
  cv::Point2f center_image{};
  Eigen::Vector3d center_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangent_x = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangent_y = Eigen::Vector3d::Zero();
  double fov_deg = 0.0;
  double focal = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  int patch_size = 0;
  cv::Mat patch;
};

struct CornerFusionObservation {
  cv::Point2f point{};
  double weight = 1.0;
  int target_longest_side = 0;
  double scale_factor = 1.0;
  double configured_scale_divisor = 0.0;
};

struct CornerFusionOutcome {
  cv::Point2f consensus_corner{};
  cv::Point2f fused_corner{};
  double outlier_threshold = 0.0;
  double average_deviation_before = 0.0;
  double max_deviation_before = 0.0;
  double average_deviation_after = 0.0;
  double max_deviation_after = 0.0;
  int inlier_count = 0;
  int outlier_count = 0;
  bool used_outlier_rejection = false;
  bool stable_after_fusion = false;
  std::vector<bool> inlier_mask;
  std::vector<double> deviations_before;
  std::vector<double> deviations_after;
};

struct MultiScaleCornerFusionOutcome {
  bool valid = false;
  std::array<cv::Point2f, 4> fused_corners{};
  std::array<OuterCornerFusionDebugInfo, 4> debug{};
};

double ComputeQuadArea(const std::array<cv::Point2f, 4>& corners);
cv::Point2f ComputeQuadCenter(const std::array<cv::Point2f, 4>& corners);
std::pair<double, double> ComputeEdgeRange(const std::array<cv::Point2f, 4>& corners);
bool IntersectLines(const FittedLine& first, const FittedLine& second, cv::Point2f* intersection);
std::vector<cv::Point2f> CollectCornerMarkerEdgeSupportPoints(
    const cv::Mat& gray,
    const cv::Point2f& corner,
    const cv::Point2f& along_edge,
    double edge_length,
    const cv::Point2f& quad_center,
    double corner_marker_width,
    int verification_roi_radius);
ImageLineCornerRefinement RefineCornerByImageLineSupportIntersection(
    const std::vector<cv::Point2f>& prev_support_points,
    const std::vector<cv::Point2f>& next_support_points);

std::string Trim(const std::string& value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::string RemoveInlineComment(const std::string& line) {
  const auto pos = line.find('#');
  if (pos == std::string::npos) {
    return line;
  }
  return line.substr(0, pos);
}

std::string Unquote(const std::string& value) {
  if (value.size() >= 2) {
    const char first = value.front();
    const char last = value.back();
    if ((first == '\'' && last == '\'') || (first == '"' && last == '"')) {
      return value.substr(1, value.size() - 2);
    }
  }
  return value;
}

int ParseInt(const std::string& key, const std::string& value) {
  try {
    return std::stoi(value);
  } catch (const std::exception&) {
    throw std::runtime_error("Failed to parse integer field '" + key + "' from value '" + value + "'.");
  }
}

double ParseDouble(const std::string& key, const std::string& value) {
  try {
    return std::stod(value);
  } catch (const std::exception&) {
    throw std::runtime_error("Failed to parse float field '" + key + "' from value '" + value + "'.");
  }
}

bool ParseBool(const std::string& key, const std::string& value) {
  const std::string lowered = [&]() {
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return out;
  }();

  if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
    return false;
  }
  throw std::runtime_error("Failed to parse bool field '" + key + "' from value '" + value + "'.");
}

std::vector<double> ParseDoubleList(const std::string& key, const std::string& value) {
  const std::string trimmed = Trim(value);
  if (trimmed.size() < 2 || trimmed.front() != '[' || trimmed.back() != ']') {
    throw std::runtime_error("Expected list syntax for field '" + key + "', got '" + value + "'.");
  }

  const std::string inner = Trim(trimmed.substr(1, trimmed.size() - 2));
  std::vector<double> parsed;
  if (inner.empty()) {
    return parsed;
  }

  std::stringstream stream(inner);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const std::string cleaned_token = Trim(token);
    if (cleaned_token.empty()) {
      continue;
    }
    parsed.push_back(ParseDouble(key, cleaned_token));
  }
  return parsed;
}

std::vector<int> ParseIntList(const std::string& key, const std::string& value) {
  const std::vector<double> parsed_doubles = ParseDoubleList(key, value);
  std::vector<int> parsed;
  parsed.reserve(parsed_doubles.size());
  for (const double parsed_value : parsed_doubles) {
    const double rounded = std::round(parsed_value);
    if (std::abs(parsed_value - rounded) > 1e-9) {
      throw std::runtime_error(
          "Field '" + key + "' must contain integer-valued entries, got '" + value + "'.");
    }
    parsed.push_back(static_cast<int>(rounded));
  }
  return parsed;
}

std::vector<int> NormalizeBoardIds(const std::vector<int>& requested_ids,
                                   int fallback_tag_id) {
  std::vector<int> normalized_ids;
  normalized_ids.reserve(requested_ids.size() + 1);
  for (int board_id : requested_ids) {
    if (board_id < 0) {
      continue;
    }
    if (std::find(normalized_ids.begin(), normalized_ids.end(), board_id) ==
        normalized_ids.end()) {
      normalized_ids.push_back(board_id);
    }
  }

  if (normalized_ids.empty() && fallback_tag_id >= 0) {
    normalized_ids.push_back(fallback_tag_id);
  }
  return normalized_ids;
}

double ClampUnit(double value) {
  return std::max(0.0, std::min(1.0, value));
}

double Dot(const cv::Point2f& lhs, const cv::Point2f& rhs) {
  return static_cast<double>(lhs.x) * rhs.x + static_cast<double>(lhs.y) * rhs.y;
}

double Cross(const cv::Point2f& lhs, const cv::Point2f& rhs) {
  return static_cast<double>(lhs.x) * rhs.y - static_cast<double>(lhs.y) * rhs.x;
}

double Norm(const cv::Point2f& vector) {
  return std::hypot(vector.x, vector.y);
}

cv::Point2f NormalizeVector(const cv::Point2f& vector) {
  const double norm = Norm(vector);
  if (norm <= 1e-9) {
    return cv::Point2f(0.0f, 0.0f);
  }
  return cv::Point2f(static_cast<float>(vector.x / norm), static_cast<float>(vector.y / norm));
}

cv::Point2f PerpendicularLeft(const cv::Point2f& vector) {
  return cv::Point2f(-vector.y, vector.x);
}

cv::Point2f ToPoint(const Eigen::Vector2d& point) {
  return cv::Point2f(static_cast<float>(point.x()), static_cast<float>(point.y()));
}

Eigen::Vector2d ToEigen(const cv::Point2f& point) {
  return Eigen::Vector2d(point.x, point.y);
}

int CountValidCorners(const std::array<bool, 4>& valid_mask) {
  return static_cast<int>(std::count(valid_mask.begin(), valid_mask.end(), true));
}

IntermediateCameraConfig ToIntermediateCameraConfig(const OuterRefineCameraConfig& config) {
  IntermediateCameraConfig converted;
  converted.camera_model = config.camera_model;
  converted.distortion_model = config.distortion_model;
  converted.intrinsics = config.intrinsics;
  converted.distortion_coeffs = config.distortion_coeffs;
  converted.resolution = config.resolution;
  return converted;
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

bool ProjectRayToImage(const DoubleSphereCameraModel& camera,
                       const Eigen::Vector3d& ray,
                       cv::Point2f* image_point) {
  if (image_point == nullptr) {
    return false;
  }
  Eigen::Vector2d keypoint;
  if (!camera.vsEuclideanToKeypoint(ray, &keypoint)) {
    return false;
  }
  if (!std::isfinite(keypoint.x()) || !std::isfinite(keypoint.y())) {
    return false;
  }
  *image_point = cv::Point2f(static_cast<float>(keypoint.x()),
                             static_cast<float>(keypoint.y()));
  return true;
}

bool BuildLocalSpherePatchFrame(const DoubleSphereCameraModel& camera,
                                const cv::Point2f& center,
                                Eigen::Vector3d* center_ray,
                                Eigen::Vector3d* tangent_x,
                                Eigen::Vector3d* tangent_y) {
  if (center_ray == nullptr || tangent_x == nullptr || tangent_y == nullptr) {
    return false;
  }

  if (!camera.keypointToEuclidean(Eigen::Vector2d(center.x, center.y), center_ray) ||
      !NormalizeRay(center_ray)) {
    return false;
  }

  Eigen::Vector3d ray_x = Eigen::Vector3d::Zero();
  Eigen::Vector3d ray_y = Eigen::Vector3d::Zero();
  constexpr double kDeltaPx = 24.0;
  if (!camera.keypointToEuclidean(Eigen::Vector2d(center.x + kDeltaPx, center.y), &ray_x) ||
      !camera.keypointToEuclidean(Eigen::Vector2d(center.x, center.y + kDeltaPx), &ray_y) ||
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

std::vector<LocalSpherePatchPlan> BuildOuterLocalSpherePatchPlans(
    bool use_extended_atlas,
    bool use_zero_detection_fine_atlas) {
  std::vector<LocalSpherePatchPlan> plans;
  plans.reserve(34);
  const std::array<double, 5> dense_centers{{0.10, 0.30, 0.50, 0.70, 0.90}};
  for (std::size_t row = 0; row < dense_centers.size(); ++row) {
    for (std::size_t column = 0; column < dense_centers.size(); ++column) {
      LocalSpherePatchPlan plan;
      plan.label = "dense_r" + std::to_string(row) + "_c" +
                   std::to_string(column);
      plan.normalized_x = dense_centers[column];
      plan.normalized_y = dense_centers[row];
      plan.fov_deg = 56.0;
      plans.push_back(plan);
    }
  }
  const std::array<double, 3> wide_centers{{0.18, 0.50, 0.82}};
  for (std::size_t row = 0; row < wide_centers.size(); ++row) {
    for (std::size_t column = 0; column < wide_centers.size(); ++column) {
      LocalSpherePatchPlan plan;
      plan.label = "wide_r" + std::to_string(row) + "_c" +
                   std::to_string(column);
      plan.normalized_x = wide_centers[column];
      plan.normalized_y = wide_centers[row];
      plan.fov_deg = 72.0;
      plans.push_back(plan);
    }
  }
  if (use_extended_atlas) {
    // Zero-detection large-tag frames need both tighter overlap near the
    // image boundary and wider rectifications when a Tag spans a patch.
    const std::array<double, 6> edge_dense_centers{{
        0.06, 0.24, 0.42, 0.58, 0.76, 0.94}};
    for (std::size_t row = 0; row < edge_dense_centers.size(); ++row) {
      for (std::size_t column = 0; column < edge_dense_centers.size(); ++column) {
        LocalSpherePatchPlan plan;
        plan.label = "extended_dense_r" + std::to_string(row) + "_c" +
                     std::to_string(column);
        plan.normalized_x = edge_dense_centers[column];
        plan.normalized_y = edge_dense_centers[row];
        plan.fov_deg = 48.0;
        plans.push_back(plan);
      }
    }
    const std::array<double, 5> extended_wide_centers{{
        0.10, 0.30, 0.50, 0.70, 0.90}};
    for (std::size_t row = 0; row < extended_wide_centers.size(); ++row) {
      for (std::size_t column = 0; column < extended_wide_centers.size(); ++column) {
        LocalSpherePatchPlan plan;
        plan.label = "extended_wide_r" + std::to_string(row) + "_c" +
                     std::to_string(column);
        plan.normalized_x = extended_wide_centers[column];
        plan.normalized_y = extended_wide_centers[row];
        plan.fov_deg = 86.0;
        plans.push_back(plan);
      }
    }
    const std::array<double, 3> ultra_wide_centers{{0.20, 0.50, 0.80}};
    for (std::size_t row = 0; row < ultra_wide_centers.size(); ++row) {
      for (std::size_t column = 0; column < ultra_wide_centers.size(); ++column) {
        LocalSpherePatchPlan plan;
        plan.label = "extended_ultra_r" + std::to_string(row) + "_c" +
                     std::to_string(column);
        plan.normalized_x = ultra_wide_centers[column];
        plan.normalized_y = ultra_wide_centers[row];
        plan.fov_deg = 108.0;
        plans.push_back(plan);
      }
    }
  }
  if (use_zero_detection_fine_atlas) {
    // A frame with no direct decode has no usable image-space ROI.  This
    // dedicated tier samples the full usable fish-eye footprint densely enough
    // to place a large, highly distorted tag near the center of at least one
    // locally rectified view.  It is intentionally restricted to all-zero
    // frames because building these patches is substantially more expensive.
    const std::array<double, 9> zero_detection_centers{{
        0.02, 0.14, 0.26, 0.38, 0.50, 0.62, 0.74, 0.86, 0.98}};
    for (std::size_t row = 0; row < zero_detection_centers.size(); ++row) {
      for (std::size_t column = 0; column < zero_detection_centers.size(); ++column) {
        LocalSpherePatchPlan plan;
        plan.label = "zero_fine_r" + std::to_string(row) + "_c" +
                     std::to_string(column);
        plan.normalized_x = zero_detection_centers[column];
        plan.normalized_y = zero_detection_centers[row];
        plan.fov_deg = 42.0;
        plans.push_back(plan);
      }
    }
  }
  return plans;
}

bool BuildLocalSpherePatch(const cv::Mat& gray,
                           const DoubleSphereCameraModel& camera,
                           const LocalSpherePatchPlan& plan,
                           LocalSpherePatchContext* context) {
  if (context == nullptr) {
    return false;
  }

  context->patch.release();
  context->patch_size = kOuterLocalSpherePatchSize;
  context->label = std::string(plan.label) + "_fov" +
                   std::to_string(static_cast<int>(std::lround(plan.fov_deg)));
  context->center_image =
      cv::Point2f(static_cast<float>(plan.normalized_x * static_cast<double>(gray.cols - 1)),
                  static_cast<float>(plan.normalized_y * static_cast<double>(gray.rows - 1)));
  context->fov_deg = plan.fov_deg;
  context->cx = 0.5 * static_cast<double>(context->patch_size - 1);
  context->cy = 0.5 * static_cast<double>(context->patch_size - 1);
  context->focal =
      0.5 * static_cast<double>(context->patch_size) / std::tan(0.5 * plan.fov_deg * kPi / 180.0);

  if (!BuildLocalSpherePatchFrame(camera, context->center_image, &context->center_ray,
                                  &context->tangent_x, &context->tangent_y)) {
    return false;
  }

  cv::Mat map_x(context->patch_size, context->patch_size, CV_32F);
  cv::Mat map_y(context->patch_size, context->patch_size, CV_32F);
  for (int y = 0; y < context->patch_size; ++y) {
    for (int x = 0; x < context->patch_size; ++x) {
      const double nx = (static_cast<double>(x) - context->cx) / context->focal;
      const double ny = (static_cast<double>(y) - context->cy) / context->focal;
      Eigen::Vector3d ray = context->center_ray + nx * context->tangent_x + ny * context->tangent_y;
      if (!NormalizeRay(&ray)) {
        map_x.at<float>(y, x) = -1.0f;
        map_y.at<float>(y, x) = -1.0f;
        continue;
      }

      cv::Point2f image_point;
      if (!ProjectRayToImage(camera, ray, &image_point)) {
        map_x.at<float>(y, x) = -1.0f;
        map_y.at<float>(y, x) = -1.0f;
        continue;
      }

      map_x.at<float>(y, x) = image_point.x;
      map_y.at<float>(y, x) = image_point.y;
    }
  }

  cv::remap(gray, context->patch, map_x, map_y, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar(127));
  return !context->patch.empty();
}

bool PatchPixelToOriginalImage(const DoubleSphereCameraModel& camera,
                               const LocalSpherePatchContext& context,
                               const cv::Point2f& patch_point,
                               cv::Point2f* image_point) {
  if (image_point == nullptr) {
    return false;
  }
  const double nx = (static_cast<double>(patch_point.x) - context.cx) / context.focal;
  const double ny = (static_cast<double>(patch_point.y) - context.cy) / context.focal;
  Eigen::Vector3d ray = context.center_ray + nx * context.tangent_x + ny * context.tangent_y;
  if (!NormalizeRay(&ray)) {
    return false;
  }
  return ProjectRayToImage(camera, ray, image_point);
}

bool BuildScaleCandidateFromPatchDetection(const AprilTags::TagDetection& detection,
                                           const LocalSpherePatchContext& context,
                                           const DoubleSphereCameraModel& camera,
                                           ScaleCandidate* candidate) {
  if (candidate == nullptr) {
    return false;
  }

  candidate->target_longest_side = context.patch_size;
  candidate->scale_factor = 1.0;
  candidate->configured_scale_divisor = 0.0;
  candidate->scaled_size = cv::Size(context.patch_size, context.patch_size);
  candidate->detection = detection;
  candidate->from_local_patch_rescue = true;
  candidate->local_patch_label = context.label;

  for (int index = 0; index < 4; ++index) {
    const cv::Point2f patch_corner(detection.p[index].first, detection.p[index].second);
    candidate->scaled_corners[static_cast<std::size_t>(index)] = patch_corner;
    if (!PatchPixelToOriginalImage(camera, context, patch_corner,
                                   &candidate->original_corners[static_cast<std::size_t>(index)])) {
      return false;
    }
  }

  candidate->scaled_area = ComputeQuadArea(candidate->scaled_corners);
  const std::pair<double, double> edge_range = ComputeEdgeRange(candidate->scaled_corners);
  candidate->min_edge = edge_range.first;
  candidate->max_edge = edge_range.second;
  candidate->shape_quality =
      candidate->max_edge > 1e-6 ? ClampUnit(candidate->min_edge / candidate->max_edge) : 0.0;
  candidate->local_patch_center_distance = cv::norm(
      ComputeQuadCenter(candidate->scaled_corners) -
      cv::Point2f(static_cast<float>(context.cx),
                  static_cast<float>(context.cy)));
  return true;
}

std::string JoinReasons(const std::vector<std::string>& reasons) {
  if (reasons.empty()) {
    return "";
  }

  std::ostringstream stream;
  for (std::size_t index = 0; index < reasons.size(); ++index) {
    if (index > 0) {
      stream << "; ";
    }
    stream << reasons[index];
  }
  return stream.str();
}

std::string SummarizeRawDetection(const AprilTags::TagDetection& detection,
                                  const cv::Size& scaled_size) {
  std::ostringstream stream;
  std::array<cv::Point2f, 4> corners{};
  for (int index = 0; index < 4; ++index) {
    corners[static_cast<std::size_t>(index)] =
        cv::Point2f(detection.p[index].first, detection.p[index].second);
  }
  double signed_area_twice = 0.0;
  double min_edge = std::numeric_limits<double>::infinity();
  double max_edge = 0.0;
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f& current = corners[static_cast<std::size_t>(index)];
    const cv::Point2f& next = corners[static_cast<std::size_t>((index + 1) % 4)];
    signed_area_twice += static_cast<double>(current.x) * static_cast<double>(next.y) -
                         static_cast<double>(next.x) * static_cast<double>(current.y);
    const double edge_length = std::hypot(static_cast<double>(next.x - current.x),
                                          static_cast<double>(next.y - current.y));
    min_edge = std::min(min_edge, edge_length);
    max_edge = std::max(max_edge, edge_length);
  }
  const double area = 0.5 * std::abs(signed_area_twice);
  const std::pair<double, double> edge_range{min_edge, max_edge};
  const double shape_quality =
      edge_range.second > 1e-6 ? ClampUnit(edge_range.first / edge_range.second) : 0.0;
  bool inside_border = true;
  for (const cv::Point2f& corner : corners) {
    if (corner.x < 4.0f ||
        corner.x > static_cast<float>(scaled_size.width) - 4.0f ||
        corner.y < 4.0f ||
        corner.y > static_cast<float>(scaled_size.height) - 4.0f) {
      inside_border = false;
      break;
    }
  }

  stream << "id=" << detection.id
         << " good=" << (detection.good ? "1" : "0")
         << " ham=" << detection.hammingDistance
         << " area=" << std::fixed << std::setprecision(1) << area
         << " min_edge=" << edge_range.first
         << " shape=" << std::setprecision(2) << shape_quality
         << " inside=" << (inside_border ? "1" : "0");
  return stream.str();
}

MultiScaleOuterTagDetectorConfig ParseConfig(const std::string& yaml_path) {
  std::ifstream stream(yaml_path);
  if (!stream.is_open()) {
    throw std::runtime_error("Could not open config file: " + yaml_path);
  }

  MultiScaleOuterTagDetectorConfig config;
  std::string line;
  while (std::getline(stream, line)) {
    const std::string cleaned = Trim(RemoveInlineComment(line));
    if (cleaned.empty()) {
      continue;
    }

    const auto colon = cleaned.find(':');
    if (colon == std::string::npos) {
      continue;
    }

    const std::string key = Trim(cleaned.substr(0, colon));
    const std::string value = Unquote(Trim(cleaned.substr(colon + 1)));

    if (key == "tagId" || key == "tag_id") {
      config.tag_id = ParseInt(key, value);
    } else if (key == "tagIds" || key == "tag_ids") {
      config.tag_ids = ParseIntList(key, value);
    } else if (key == "autoDiscoverTagIds" || key == "auto_discover_tag_ids") {
      config.auto_discover_tag_ids = ParseBool(key, value);
    } else if (key == "minBorderDistance" || key == "min_border_distance") {
      config.min_border_distance = ParseDouble(key, value);
    } else if (key == "maxScalesToTry" || key == "max_scales_to_try") {
      config.max_scales_to_try = ParseInt(key, value);
    } else if (key == "enableAdaptiveScaleCascade" ||
               key == "enable_adaptive_scale_cascade") {
      config.enable_adaptive_scale_cascade = ParseBool(key, value);
    } else if (key == "adaptiveCoarseScaleDivisors" ||
               key == "adaptive_coarse_scale_divisors") {
      config.adaptive_coarse_scale_divisors = ParseDoubleList(key, value);
    } else if (key == "adaptiveFallbackScaleDivisors" ||
               key == "adaptive_fallback_scale_divisors") {
      config.adaptive_fallback_scale_divisors = ParseDoubleList(key, value);
    } else if (key == "adaptiveCoarseMaxHamming" ||
               key == "adaptive_coarse_max_hamming") {
      config.adaptive_coarse_max_hamming = ParseInt(key, value);
    } else if (key == "outerLocalContextScale" || key == "outer_local_context_scale") {
      config.outer_local_context_scale = ParseDouble(key, value);
    } else if (key == "outerCornerMarkerRatio" || key == "outer_corner_marker_ratio" ||
               key == "tagSpacing" || key == "tag_spacing") {
      config.outer_corner_marker_ratio = ParseDouble(key, value);
    } else if (key == "outerSubpixScale" || key == "outer_subpix_scale") {
      config.outer_subpix_scale = ParseDouble(key, value);
    } else if (key == "enableCloseEdgeOuterSubpixBoost" ||
               key == "enable_close_edge_outer_subpix_boost") {
      config.enable_close_edge_outer_subpix_boost = ParseBool(key, value);
    } else if (key == "closeEdgeOuterSubpixAreaRatio" ||
               key == "close_edge_outer_subpix_area_ratio") {
      config.close_edge_outer_subpix_area_ratio = ParseDouble(key, value);
    } else if (key == "closeEdgeOuterSubpixMinPolarDeg" ||
               key == "close_edge_outer_subpix_min_polar_deg") {
      config.close_edge_outer_subpix_min_polar_deg = ParseDouble(key, value);
    } else if (key == "closeEdgeOuterSubpixFullPolarDeg" ||
               key == "close_edge_outer_subpix_full_polar_deg") {
      config.close_edge_outer_subpix_full_polar_deg = ParseDouble(key, value);
    } else if (key == "closeEdgeOuterSubpixBorderRatio" ||
               key == "close_edge_outer_subpix_border_ratio") {
      config.close_edge_outer_subpix_border_ratio = ParseDouble(key, value);
    } else if (key == "closeEdgeOuterSubpixMultiplier" ||
               key == "close_edge_outer_subpix_multiplier") {
      config.close_edge_outer_subpix_multiplier = ParseDouble(key, value);
    } else if (key == "closeEdgeOuterSubpixMaxMultiplier" ||
               key == "close_edge_outer_subpix_max_multiplier") {
      config.close_edge_outer_subpix_max_multiplier = ParseDouble(key, value);
    } else if (key == "outerRefineGateScale" || key == "outer_refine_gate_scale") {
      config.outer_refine_gate_scale = ParseDouble(key, value);
    } else if (key == "outerRefineGateMin" || key == "outer_refine_gate_min") {
      config.outer_refine_gate_min = ParseDouble(key, value);
    } else if (key == "scaleCandidates" || key == "scale_candidates") {
      config.scale_candidates = ParseIntList(key, value);
    } else if (key == "scaleDivisors" || key == "scale_divisors") {
      config.scale_divisors = ParseDoubleList(key, value);
    } else if (key == "enableOuterSphericalRefinement" ||
               key == "enable_outer_spherical_refinement") {
      config.enable_outer_spherical_refinement = ParseBool(key, value);
    } else if (key == "doOuterSubpixRefinement" || key == "do_outer_subpix_refinement") {
      config.do_outer_subpix_refinement = ParseBool(key, value);
    } else if (key == "outerSubpixWindowRadius" || key == "outer_subpix_window_radius") {
      config.outer_subpix_window_radius = ParseInt(key, value);
    } else if (key == "outerSubpixWindowScale" || key == "outer_subpix_window_scale") {
      config.outer_subpix_window_scale = ParseDouble(key, value);
      config.outer_subpix_scale = config.outer_subpix_window_scale;
    } else if (key == "outerSubpixWindowMin" || key == "outer_subpix_window_min") {
      config.outer_subpix_window_min = ParseInt(key, value);
    } else if (key == "outerSubpixWindowMax" || key == "outer_subpix_window_max") {
      config.outer_subpix_window_max = ParseInt(key, value);
    } else if (key == "maxOuterRefineDisplacement" || key == "max_outer_refine_displacement") {
      config.max_outer_refine_displacement = ParseDouble(key, value);
      config.outer_refine_gate_min = config.max_outer_refine_displacement;
    } else if (key == "outerRefineDisplacementScale" || key == "outer_refine_displacement_scale") {
      config.outer_refine_displacement_scale = ParseDouble(key, value);
      config.outer_refine_gate_scale = config.outer_refine_displacement_scale;
    } else if (key == "minDetectionQuality" || key == "min_detection_quality") {
      config.min_detection_quality = ParseDouble(key, value);
    } else if (key == "blurBeforeDetect" || key == "blur_before_detect") {
      config.blur_before_detect = ParseBool(key, value);
    } else if (key == "blurKernel" || key == "blur_kernel") {
      config.blur_kernel = ParseInt(key, value);
    } else if (key == "blurSigma" || key == "blur_sigma") {
      config.blur_sigma = ParseDouble(key, value);
    } else if (key == "enableAnonymousTagLikeGeometryRescue" ||
               key == "enable_anonymous_tag_like_geometry_rescue") {
      config.enable_anonymous_tag_like_geometry_rescue = ParseBool(key, value);
    } else if (key == "anonymousTagLikeRescueMaxCenterErrorScale" ||
               key == "anonymous_tag_like_rescue_max_center_error_scale") {
      config.anonymous_tag_like_rescue_max_center_error_scale = ParseDouble(key, value);
    } else if (key == "anonymousTagLikeRescueMinAreaRatio" ||
               key == "anonymous_tag_like_rescue_min_area_ratio") {
      config.anonymous_tag_like_rescue_min_area_ratio = ParseDouble(key, value);
    } else if (key == "anonymousTagLikeRescueMaxAreaRatio" ||
               key == "anonymous_tag_like_rescue_max_area_ratio") {
      config.anonymous_tag_like_rescue_max_area_ratio = ParseDouble(key, value);
    } else if (key == "enableCameraAwareSpherePatchRescue" ||
               key == "enable_camera_aware_sphere_patch_rescue") {
      config.enable_camera_aware_sphere_patch_rescue = ParseBool(key, value);
    } else if (key == "cameraAwareSpherePatchMaxHamming" ||
               key == "camera_aware_sphere_patch_max_hamming") {
      config.camera_aware_sphere_patch_max_hamming = ParseInt(key, value);
    } else if (key == "cameraAwareSpherePatchCommitMappedCorners" ||
               key == "camera_aware_sphere_patch_commit_mapped_corners") {
      config.camera_aware_sphere_patch_commit_mapped_corners =
          ParseBool(key, value);
    } else if (key == "cameraAwareSpherePatchRescueZeroDetectionFrames" ||
               key == "camera_aware_sphere_patch_rescue_zero_detection_frames") {
      config.camera_aware_sphere_patch_rescue_zero_detection_frames =
          ParseBool(key, value);
    } else if (key == "cameraAwareSpherePatchUseExtendedAtlas" ||
               key == "camera_aware_sphere_patch_use_extended_atlas") {
      config.camera_aware_sphere_patch_use_extended_atlas =
          ParseBool(key, value);
    } else if (key == "camera_model") {
      config.refine_camera.camera_model = value;
    } else if (key == "distortion_model") {
      config.refine_camera.distortion_model = value;
    } else if (key == "intrinsics") {
      config.refine_camera.intrinsics = ParseDoubleList(key, value);
    } else if (key == "distortion_coeffs") {
      config.refine_camera.distortion_coeffs = ParseDoubleList(key, value);
    } else if (key == "resolution") {
      config.refine_camera.resolution = ParseIntList(key, value);
    } else if (key == "enableOuterCornerLocalVerification" ||
               key == "enable_outer_corner_local_verification") {
      // Legacy compatibility: the pipeline is now always C-S.
      (void)ParseBool(key, value);
    } else if (key == "enableOuterCornerLayoutCheck" ||
               key == "enable_outer_corner_layout_check") {
      config.enable_outer_corner_layout_check = ParseBool(key, value);
    } else if (key == "outerCornerVerificationRoiScale" ||
               key == "outer_corner_verification_roi_scale") {
      config.outer_corner_verification_roi_scale = ParseDouble(key, value);
      config.outer_local_context_scale = config.outer_corner_verification_roi_scale;
    } else if (key == "outerCornerVerificationRoiMin" ||
               key == "outer_corner_verification_roi_min") {
      config.outer_corner_verification_roi_min = ParseInt(key, value);
    } else if (key == "outerCornerVerificationRoiMax" ||
               key == "outer_corner_verification_roi_max") {
      config.outer_corner_verification_roi_max = ParseInt(key, value);
    } else if (key == "outerCornerCandidateScale" ||
               key == "outer_corner_candidate_scale") {
      config.outer_corner_candidate_scale = ParseDouble(key, value);
    } else if (key == "outerCornerCandidateMin" ||
               key == "outer_corner_candidate_min") {
      config.outer_corner_candidate_min = ParseInt(key, value);
    } else if (key == "outerCornerCandidateMax" ||
               key == "outer_corner_candidate_max") {
      config.outer_corner_candidate_max = ParseInt(key, value);
    } else if (key == "outerCornerBranchSearchScale" ||
               key == "outer_corner_branch_search_scale") {
      config.outer_corner_branch_search_scale = ParseDouble(key, value);
    } else if (key == "outerCornerBranchSearchMin" ||
               key == "outer_corner_branch_search_min") {
      config.outer_corner_branch_search_min = ParseInt(key, value);
    } else if (key == "outerCornerBranchSearchMax" ||
               key == "outer_corner_branch_search_max") {
      config.outer_corner_branch_search_max = ParseInt(key, value);
    } else if (key == "outerCornerVerificationRoiRadius" ||
               key == "outer_corner_verification_roi_radius") {
      const int fixed_radius = ParseInt(key, value);
      config.outer_corner_verification_roi_scale = 0.0;
      config.outer_corner_verification_roi_min = fixed_radius;
      config.outer_corner_verification_roi_max = fixed_radius;
    } else if (key == "outerCornerCandidateRadius" ||
               key == "outer_corner_candidate_radius") {
      const int fixed_radius = ParseInt(key, value);
      config.outer_corner_candidate_scale = 0.0;
      config.outer_corner_candidate_min = fixed_radius;
      config.outer_corner_candidate_max = fixed_radius;
    } else if (key == "outerCornerBranchSearchRadius" ||
               key == "outer_corner_branch_search_radius") {
      const int fixed_radius = ParseInt(key, value);
      config.outer_corner_branch_search_scale = 0.0;
      config.outer_corner_branch_search_min = fixed_radius;
      config.outer_corner_branch_search_max = fixed_radius;
    } else if (key == "outerCornerMinDirectionScore" ||
               key == "outer_corner_min_direction_score") {
      config.outer_corner_min_direction_score = ParseDouble(key, value);
    } else if (key == "outerCornerMinLayoutScore" ||
               key == "outer_corner_min_layout_score") {
      config.outer_corner_min_layout_score = ParseDouble(key, value);
    }
  }

  config.tag_ids = NormalizeBoardIds(config.tag_ids, config.tag_id);
  if (config.tag_ids.empty()) {
    throw std::runtime_error("Outer detector config requires at least one valid tag id.");
  }
  config.tag_id = config.tag_ids.front();
  if (config.anonymous_tag_like_rescue_max_center_error_scale < 0.0) {
    throw std::runtime_error(
        "anonymous_tag_like_rescue_max_center_error_scale must be non-negative.");
  }
  if (config.anonymous_tag_like_rescue_min_area_ratio < 0.0 ||
      config.anonymous_tag_like_rescue_max_area_ratio <
          config.anonymous_tag_like_rescue_min_area_ratio) {
    throw std::runtime_error(
        "anonymous_tag_like_rescue area ratio bounds must be non-negative and ordered.");
  }

  return config;
}

double ComputeQuadArea(const std::array<cv::Point2f, 4>& corners) {
  std::vector<cv::Point2f> polygon(corners.begin(), corners.end());
  return std::abs(cv::contourArea(polygon));
}

cv::Point2f ComputeQuadCenter(const std::array<cv::Point2f, 4>& corners) {
  cv::Point2f center(0.0f, 0.0f);
  for (const cv::Point2f& corner : corners) {
    center += corner;
  }
  return center * 0.25f;
}

std::pair<double, double> ComputeEdgeRange(const std::array<cv::Point2f, 4>& corners) {
  double min_edge = std::numeric_limits<double>::max();
  double max_edge = 0.0;
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f delta = corners[(index + 1) % 4] - corners[index];
    const double length = std::hypot(delta.x, delta.y);
    min_edge = std::min(min_edge, length);
    max_edge = std::max(max_edge, length);
  }
  if (!std::isfinite(min_edge)) {
    min_edge = 0.0;
  }
  return {min_edge, max_edge};
}

bool PassesBorderCheck(const std::array<cv::Point2f, 4>& corners, const cv::Size& size,
                       double min_border_distance) {
  for (const cv::Point2f& corner : corners) {
    if (corner.x < min_border_distance ||
        corner.x > static_cast<float>(size.width) - min_border_distance ||
        corner.y < min_border_distance ||
        corner.y > static_cast<float>(size.height) - min_border_distance) {
      return false;
    }
  }
  return true;
}

bool PassesRefinedCornerEdgeSupportCheck(
    const cv::Mat& gray,
    const std::array<cv::Point2f, 4>& refined_corners,
    int corner_index,
    const OuterCornerVerificationDebugInfo& debug,
    std::string* failure_reason) {
  if (corner_index < 0 || corner_index >= 4) {
    if (failure_reason != nullptr) {
      *failure_reason = "subpix_edge_support:index";
    }
    return false;
  }
  const int prev_index = (corner_index + 3) % 4;
  const int next_index = (corner_index + 1) % 4;
  const cv::Point2f corner = refined_corners[static_cast<std::size_t>(corner_index)];
  const cv::Point2f prev_edge =
      refined_corners[static_cast<std::size_t>(prev_index)] - corner;
  const cv::Point2f next_edge =
      refined_corners[static_cast<std::size_t>(next_index)] - corner;
  const double prev_length = Norm(prev_edge);
  const double next_length = Norm(next_edge);
  const cv::Point2f quad_center = ComputeQuadCenter(refined_corners);
  const std::vector<cv::Point2f> prev_support =
      CollectCornerMarkerEdgeSupportPoints(
          gray, corner, prev_edge, prev_length, quad_center,
          debug.corner_marker_width, debug.verification_roi_radius);
  const std::vector<cv::Point2f> next_support =
      CollectCornerMarkerEdgeSupportPoints(
          gray, corner, next_edge, next_length, quad_center,
          debug.corner_marker_width, debug.verification_roi_radius);
  const ImageLineCornerRefinement line_check =
      RefineCornerByImageLineSupportIntersection(prev_support, next_support);
  if (!line_check.success || line_check.quality < kOuterLineMinQuality) {
    if (failure_reason != nullptr) {
      std::ostringstream stream;
      stream << "subpix_edge_support:" << line_check.failure_reason
             << ":q=" << std::fixed << std::setprecision(2)
             << line_check.quality
             << ":support=(" << line_check.prev_line.support_count
             << "," << line_check.next_line.support_count << ")"
             << ":res=(" << line_check.prev_line.rms_residual
             << "," << line_check.next_line.rms_residual << ")";
      *failure_reason = stream.str();
    }
    return false;
  }
  return true;
}

bool PassesRefinedCornerResponseCheck(
    const cv::Mat& gray,
    const cv::Point2f& refined_corner,
    int subpix_radius,
    std::string* diagnostic) {
  if (gray.empty()) {
    if (diagnostic != nullptr) {
      *diagnostic = "corner_response:empty_image";
    }
    return false;
  }
  const int x = static_cast<int>(std::lround(refined_corner.x));
  const int y = static_cast<int>(std::lround(refined_corner.y));
  if (x < 0 || x >= gray.cols || y < 0 || y >= gray.rows) {
    if (diagnostic != nullptr) {
      *diagnostic = "corner_response:outside";
    }
    return false;
  }

  const int response_radius =
      std::max(12, std::min(32, std::max(1, subpix_radius) / 3));
  const int margin = response_radius + 10;
  const int x0 = std::max(0, x - margin);
  const int y0 = std::max(0, y - margin);
  const int x1 = std::min(gray.cols, x + margin + 1);
  const int y1 = std::min(gray.rows, y + margin + 1);
  if (x1 - x0 < 20 || y1 - y0 < 20) {
    if (diagnostic != nullptr) {
      *diagnostic = "corner_response:small_roi";
    }
    return false;
  }

  cv::Mat roi;
  gray(cv::Rect(x0, y0, x1 - x0, y1 - y0)).convertTo(roi, CV_32F);
  cv::Mat response;
  cv::cornerMinEigenVal(roi, response, 15, 3);

  const int cx = x - x0;
  const int cy = y - y0;
  const int peak_radius = 2;
  double refined_response = 0.0;
  for (int yy = std::max(0, cy - peak_radius);
       yy <= std::min(response.rows - 1, cy + peak_radius); ++yy) {
    for (int xx = std::max(0, cx - peak_radius);
         xx <= std::min(response.cols - 1, cx + peak_radius); ++xx) {
      refined_response =
          std::max(refined_response,
                   static_cast<double>(response.at<float>(yy, xx)));
    }
  }

  double local_peak_response = 0.0;
  for (int yy = std::max(0, cy - response_radius);
       yy <= std::min(response.rows - 1, cy + response_radius); ++yy) {
    for (int xx = std::max(0, cx - response_radius);
         xx <= std::min(response.cols - 1, cx + response_radius); ++xx) {
      local_peak_response =
          std::max(local_peak_response,
                   static_cast<double>(response.at<float>(yy, xx)));
    }
  }

  constexpr double kReliableCornerPeakResponse = 8.0;
  constexpr double kMinimumRefinedCornerResponse = 5.0;
  constexpr double kMinimumPeakRatio = 0.20;
  const double peak_ratio =
      local_peak_response > 1e-9 ? refined_response / local_peak_response : 0.0;
  const bool reliable_response = local_peak_response >= kReliableCornerPeakResponse;
  const bool passes =
      !reliable_response ||
      (refined_response >= kMinimumRefinedCornerResponse &&
       peak_ratio >= kMinimumPeakRatio);

  if (diagnostic != nullptr) {
    std::ostringstream stream;
    stream << "corner_response:r=" << std::fixed << std::setprecision(2)
           << refined_response << ":peak=" << local_peak_response
           << ":ratio=" << peak_ratio
           << ":reliable=" << (reliable_response ? 1 : 0);
    *diagnostic = stream.str();
  }
  return passes;
}

double SampleGrayBilinear(const cv::Mat& image, const cv::Point2f& point) {
  const float x = std::max(0.0f, std::min(point.x, static_cast<float>(image.cols - 1)));
  const float y = std::max(0.0f, std::min(point.y, static_cast<float>(image.rows - 1)));

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, image.cols - 1);
  const int y1 = std::min(y0 + 1, image.rows - 1);
  const float dx = x - static_cast<float>(x0);
  const float dy = y - static_cast<float>(y0);

  const float v00 = static_cast<float>(image.at<unsigned char>(y0, x0));
  const float v10 = static_cast<float>(image.at<unsigned char>(y0, x1));
  const float v01 = static_cast<float>(image.at<unsigned char>(y1, x0));
  const float v11 = static_cast<float>(image.at<unsigned char>(y1, x1));

  const float top = v00 * (1.0f - dx) + v10 * dx;
  const float bottom = v01 * (1.0f - dx) + v11 * dx;
  return static_cast<double>(top * (1.0f - dy) + bottom * dy);
}

double SampleDirectionalDerivative(const cv::Mat& image, const cv::Point2f& point,
                                   const cv::Point2f& direction) {
  const cv::Point2f delta = direction * static_cast<float>(kOuterLineDerivativeDelta);
  return SampleGrayBilinear(image, point + delta) - SampleGrayBilinear(image, point - delta);
}

bool IsInsideImage(const cv::Point2f& point, const cv::Size& size, float border = 0.0f) {
  return point.x >= border &&
         point.x <= static_cast<float>(size.width) - 1.0f - border &&
         point.y >= border &&
         point.y <= static_cast<float>(size.height) - 1.0f - border;
}

std::string BuildOuterChainLabel(const OuterCornerVerificationDebugInfo& verification) {
  if (verification.spherical_refinement_valid) {
    return verification.subpix_applied ? "C-SP-S" : "C-SP";
  }
  if (verification.subpix_applied) {
    return "C-S";
  }
  return "C";
}

int ClampRadiusFromScale(double ratio, double local_scale, int min_radius, int max_radius) {
  if (min_radius <= 0 || max_radius < min_radius) {
    throw std::runtime_error("Invalid adaptive radius bounds.");
  }
  const double scaled = ratio > 0.0 ? ratio * local_scale : static_cast<double>(min_radius);
  const int rounded = static_cast<int>(std::lround(scaled));
  return std::max(min_radius, std::min(max_radius, rounded));
}

AdaptiveCornerSearchRadii ComputeAdaptiveCornerSearchRadii(
    double local_scale,
    const MultiScaleOuterTagDetectorConfig& config) {
  AdaptiveCornerSearchRadii radii;
  radii.local_scale = local_scale;
  radii.verification_roi_radius =
      ClampRadiusFromScale(config.outer_local_context_scale, local_scale,
                           kOuterContextRadiusMin, kOuterContextRadiusMax);
  radii.candidate_radius = 0;
  radii.branch_search_radius = 0;
  return radii;
}

double ComputeOuterCornerMarkerWidth(double local_scale,
                                     const MultiScaleOuterTagDetectorConfig& config) {
  if (config.outer_corner_marker_ratio > 0.0) {
    return std::max(0.0, config.outer_corner_marker_ratio * local_scale);
  }
  return std::max(0.0, local_scale);
}

struct OuterSubpixRadiusComputation {
  double configured_scale = 0.0;
  double configured_window_scale = 0.0;
  int configured_fixed_radius = 0;
  int configured_min_radius = 0;
  int configured_max_radius = 0;
  double corner_marker_width = 0.0;
  double scaled_radius = 0.0;
  int raw_radius = 0;
  int clamp_limit = 0;
  int final_radius = 0;
  bool fixed_radius_mode = false;
  bool clamped = false;
};

OuterSubpixRadiusComputation ComputeAdaptiveOuterSubpixRadiusDebug(
    double local_scale,
    int verification_roi_radius,
    const MultiScaleOuterTagDetectorConfig& config) {
  OuterSubpixRadiusComputation result;
  result.configured_scale = config.outer_subpix_scale;
  result.configured_window_scale = config.outer_subpix_window_scale;
  result.configured_fixed_radius = config.outer_subpix_window_radius;
  result.configured_min_radius = config.outer_subpix_window_min;
  result.configured_max_radius = config.outer_subpix_window_max;
  result.corner_marker_width = ComputeOuterCornerMarkerWidth(local_scale, config);
  (void)verification_roi_radius;
  result.clamp_limit = 0;
  if (config.outer_subpix_window_radius > 0) {
    result.fixed_radius_mode = true;
    result.scaled_radius = static_cast<double>(config.outer_subpix_window_radius);
    result.raw_radius = std::max(2, config.outer_subpix_window_radius);
    const int min_radius = std::max(2, config.outer_subpix_window_min);
    if (config.outer_subpix_window_max > 0) {
      const int max_radius = std::max(min_radius, config.outer_subpix_window_max);
      result.clamp_limit = max_radius;
      result.final_radius = std::max(min_radius, std::min(max_radius, result.raw_radius));
    } else {
      result.clamp_limit = 0;
      result.final_radius = std::max(min_radius, result.raw_radius);
    }
    result.clamped = result.final_radius != result.raw_radius;
    return result;
  }

  result.scaled_radius =
      config.outer_subpix_scale > 0.0
          ? config.outer_subpix_scale * result.corner_marker_width
          : static_cast<double>(kOuterSubpixRadiusMin);
  result.raw_radius =
      std::max(kOuterSubpixRadiusMin,
               static_cast<int>(std::lround(result.scaled_radius)));
  const int min_radius = std::max(kOuterSubpixRadiusMin, config.outer_subpix_window_min);
  if (config.outer_subpix_window_max > 0) {
    const int max_radius = std::max(min_radius, config.outer_subpix_window_max);
    result.clamp_limit = max_radius;
    result.final_radius = std::max(min_radius, std::min(max_radius, result.raw_radius));
  } else {
    result.clamp_limit = 0;
    result.final_radius = std::max(min_radius, result.raw_radius);
  }
  result.clamped = result.final_radius != result.raw_radius;
  return result;
}

int ComputeAdaptiveOuterSubpixRadius(double local_scale,
                                     int verification_roi_radius,
                                     const MultiScaleOuterTagDetectorConfig& config) {
  return ComputeAdaptiveOuterSubpixRadiusDebug(local_scale,
                                               verification_roi_radius,
                                               config)
      .final_radius;
}

void PopulateOuterSubpixRadiusDebug(
    const OuterSubpixRadiusComputation& computation,
    OuterCornerVerificationDebugInfo* debug) {
  if (debug == nullptr) {
    return;
  }
  debug->configured_outer_subpix_scale = computation.configured_scale;
  debug->configured_outer_subpix_window_scale =
      computation.configured_window_scale;
  debug->configured_outer_subpix_window_radius =
      computation.configured_fixed_radius;
  debug->configured_outer_subpix_window_min =
      computation.configured_min_radius;
  debug->configured_outer_subpix_window_max =
      computation.configured_max_radius;
  debug->corner_marker_width = computation.corner_marker_width;
  debug->raw_subpix_window_radius = computation.raw_radius;
  debug->pre_boost_subpix_window_radius = computation.final_radius;
  debug->subpix_window_clamp_limit = computation.clamp_limit;
  debug->subpix_window_clamped = computation.clamped;
}

double MinDistanceToImageBorder(const std::array<cv::Point2f, 4>& corners,
                                const cv::Size& image_size) {
  double min_distance = std::numeric_limits<double>::infinity();
  for (const cv::Point2f& corner : corners) {
    min_distance = std::min(min_distance, static_cast<double>(corner.x));
    min_distance = std::min(min_distance, static_cast<double>(corner.y));
    min_distance = std::min(min_distance,
                            static_cast<double>(image_size.width - 1) -
                                static_cast<double>(corner.x));
    min_distance = std::min(min_distance,
                            static_cast<double>(image_size.height - 1) -
                                static_cast<double>(corner.y));
  }
  return min_distance;
}

bool ComputeMaxPolarAngleDeg(const std::array<cv::Point2f, 4>& corners,
                             const DoubleSphereCameraModel* sphere_camera,
                             double* max_polar_deg) {
  if (max_polar_deg == nullptr) {
    return false;
  }
  *max_polar_deg = 0.0;
  if (sphere_camera == nullptr) {
    return false;
  }
  bool has_valid_ray = false;
  for (const cv::Point2f& corner : corners) {
    Eigen::Vector3d ray = Eigen::Vector3d::Zero();
    if (!sphere_camera->keypointToEuclidean(Eigen::Vector2d(corner.x, corner.y), &ray) ||
        !NormalizeRay(&ray)) {
      continue;
    }
    const double radial_norm = std::hypot(ray.x(), ray.y());
    const double polar_rad = std::atan2(radial_norm, ray.z());
    const double polar_deg = polar_rad * 180.0 / kPi;
    if (std::isfinite(polar_deg)) {
      *max_polar_deg = std::max(*max_polar_deg, polar_deg);
      has_valid_ray = true;
    }
  }
  return has_valid_ray;
}

double ComputeMaxImageCenterPolarProxyDeg(const std::array<cv::Point2f, 4>& corners,
                                          const cv::Size& image_size) {
  if (image_size.width <= 0 || image_size.height <= 0) {
    return 0.0;
  }
  const cv::Point2d center(0.5 * static_cast<double>(image_size.width - 1),
                           0.5 * static_cast<double>(image_size.height - 1));
  const double half_diagonal = std::hypot(center.x, center.y);
  if (half_diagonal <= 1e-9) {
    return 0.0;
  }
  double max_normalized_radius = 0.0;
  for (const cv::Point2f& corner : corners) {
    const double radius =
        std::hypot(static_cast<double>(corner.x) - center.x,
                   static_cast<double>(corner.y) - center.y);
    max_normalized_radius =
        std::max(max_normalized_radius, radius / half_diagonal);
  }
  // Fallback only: map image-center radius to a polar-like score so close
  // border observations can still trigger the adaptive subpixel window when
  // no DS camera is available at this frontend stage.
  return 90.0 * std::min(1.0, std::max(0.0, max_normalized_radius));
}

bool ComputeMaxPolarOrImageProxyDeg(const std::array<cv::Point2f, 4>& corners,
                                    const cv::Size& image_size,
                                    const DoubleSphereCameraModel* sphere_camera,
                                    double* max_polar_deg) {
  if (ComputeMaxPolarAngleDeg(corners, sphere_camera, max_polar_deg)) {
    return true;
  }
  if (max_polar_deg != nullptr) {
    *max_polar_deg = ComputeMaxImageCenterPolarProxyDeg(corners, image_size);
  }
  return image_size.width > 0 && image_size.height > 0;
}

double ComputeAreaRatio(const std::array<cv::Point2f, 4>& corners,
                        const cv::Size& image_size) {
  if (image_size.width <= 0 || image_size.height <= 0) {
    return 0.0;
  }
  const double image_area =
      static_cast<double>(image_size.width) * static_cast<double>(image_size.height);
  return image_area > 0.0 ? ComputeQuadArea(corners) / image_area : 0.0;
}

struct CloseEdgeOuterSubpixBoostInfo {
  bool boost = false;
  double area_ratio = 0.0;
  double max_polar_deg = 0.0;
  double multiplier = 1.0;
};

CloseEdgeOuterSubpixBoostInfo ComputeCloseEdgeOuterSubpixBoostInfo(
    const std::array<cv::Point2f, 4>& corners,
    const cv::Size& image_size,
    const MultiScaleOuterTagDetectorConfig& config,
    const DoubleSphereCameraModel* sphere_camera) {
  CloseEdgeOuterSubpixBoostInfo info;
  if (!config.enable_close_edge_outer_subpix_boost ||
      config.close_edge_outer_subpix_multiplier <= 1.0 ||
      image_size.width <= 0 || image_size.height <= 0) {
    return info;
  }
  const double area_ratio = ComputeAreaRatio(corners, image_size);
  info.area_ratio = area_ratio;
  const bool near_board_by_area =
      area_ratio >= config.close_edge_outer_subpix_area_ratio;

  double max_polar_deg = 0.0;
  const bool have_polar =
      ComputeMaxPolarOrImageProxyDeg(corners, image_size, sphere_camera,
                                     &max_polar_deg);
  info.max_polar_deg = max_polar_deg;
  const bool high_polar =
      have_polar && max_polar_deg >= config.close_edge_outer_subpix_min_polar_deg;

  const double min_border_distance = MinDistanceToImageBorder(corners, image_size);
  const double border_threshold =
      config.close_edge_outer_subpix_border_ratio *
      static_cast<double>(std::min(image_size.width, image_size.height));
  const bool near_border = min_border_distance <= border_threshold;

  // The main reason to enlarge the outer-corner subpixel window is close range:
  // a near board occupies a larger image footprint, so the normal local window
  // can be too small for the true outer corner.  High-polar / near-border cases
  // are still useful auxiliary triggers, but they should not be mandatory.
  info.boost = near_board_by_area || (high_polar && near_border);
  if (!info.boost) {
    return info;
  }

  const double polar_span = std::max(
      1.0,
      config.close_edge_outer_subpix_full_polar_deg -
          config.close_edge_outer_subpix_min_polar_deg);
  const double polar_score =
      have_polar ? ClampUnit((max_polar_deg -
                              config.close_edge_outer_subpix_min_polar_deg) /
                             polar_span)
                 : 0.0;
  const double border_score =
      near_border && border_threshold > 1e-9
          ? ClampUnit(1.0 - min_border_distance / border_threshold)
          : 0.0;
  const double area_score =
      config.close_edge_outer_subpix_area_ratio > 1e-9
          ? ClampUnit((area_ratio / config.close_edge_outer_subpix_area_ratio -
                       1.0) /
                      2.0)
          : 0.0;
  const double severity = std::max({polar_score, border_score, area_score});
  const double base_multiplier =
      std::max(1.0, config.close_edge_outer_subpix_multiplier);
  const double max_multiplier =
      std::max(base_multiplier, config.close_edge_outer_subpix_max_multiplier);
  // Triggered observations get at least the base multiplier; high-polar,
  // near-border, or very large tags smoothly approach the configured maximum.
  info.multiplier =
      base_multiplier + (max_multiplier - base_multiplier) * severity;
  return info;
}

bool IsCloseEdgeOuterSubpixBoostCase(
    const std::array<cv::Point2f, 4>& corners,
    const cv::Size& image_size,
    const MultiScaleOuterTagDetectorConfig& config,
    const DoubleSphereCameraModel* sphere_camera) {
  return ComputeCloseEdgeOuterSubpixBoostInfo(corners, image_size, config,
                                              sphere_camera)
      .boost;
}

cv::Rect MakeCornerVerificationRoi(const cv::Point2f& corner,
                                   const cv::Size& size,
                                   int roi_radius) {
  const int x0 = std::max(0, static_cast<int>(std::floor(corner.x)) - roi_radius);
  const int y0 = std::max(0, static_cast<int>(std::floor(corner.y)) - roi_radius);
  const int x1 = std::min(size.width, static_cast<int>(std::ceil(corner.x)) + roi_radius + 1);
  const int y1 = std::min(size.height, static_cast<int>(std::ceil(corner.y)) + roi_radius + 1);
  return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

double SampleMeanAtPoints(const cv::Mat& image, const std::vector<cv::Point2f>& points) {
  double sum = 0.0;
  int count = 0;
  for (const cv::Point2f& point : points) {
    if (!IsInsideImage(point, image.size(), 1.0f)) {
      continue;
    }
    sum += SampleGrayBilinear(image, point);
    ++count;
  }
  if (count <= 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return sum / static_cast<double>(count);
}

std::vector<cv::Point2f> CollectDirectionalEdgeBranchPoints(const cv::Mat& gray,
                                                            const cv::Point2f& corner,
                                                            const cv::Point2f& edge_dir,
                                                            double edge_length,
                                                            const cv::Point2f& quad_center,
                                                            double start_offset,
                                                            double usable_extent,
                                                            int sample_count,
                                                            double search_radius) {
  if (Norm(edge_dir) <= 1e-9 || edge_length <= 1.0 || usable_extent <= 0.0 || sample_count <= 0) {
    return {};
  }

  cv::Point2f inward_normal = PerpendicularLeft(edge_dir);
  if (Dot(quad_center - corner, inward_normal) < 0.0) {
    inward_normal *= -1.0f;
  }

  std::vector<cv::Point2f> support_points;
  support_points.reserve(static_cast<std::size_t>(sample_count));

  for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
    const double alpha =
        sample_count == 1 ? 0.0 : static_cast<double>(sample_index) / static_cast<double>(sample_count - 1);
    const double edge_offset = start_offset + usable_extent * alpha;
    const cv::Point2f base_point = corner + edge_dir * static_cast<float>(edge_offset);

    double best_score = -std::numeric_limits<double>::infinity();
    cv::Point2f best_point = base_point;
    bool found = false;

    for (double offset = -search_radius; offset <= search_radius; offset += 1.0) {
      const cv::Point2f probe = base_point + inward_normal * static_cast<float>(offset);
      if (probe.x < 2.0f || probe.x > static_cast<float>(gray.cols - 3) ||
          probe.y < 2.0f || probe.y > static_cast<float>(gray.rows - 3)) {
        continue;
      }

      const double signed_derivative = -SampleDirectionalDerivative(gray, probe, inward_normal);
      const double fallback_magnitude = std::abs(signed_derivative);
      const double score = signed_derivative > 0.0 ? signed_derivative : 0.5 * fallback_magnitude;
      if (score > best_score) {
        best_score = score;
        best_point = probe;
        found = true;
      }
    }

    if (found && best_score > 4.0) {
      support_points.push_back(best_point);
    }
  }

  return support_points;
}

std::vector<cv::Point2f> CollectLocalEdgeSupportPoints(const cv::Mat& gray,
                                                       const cv::Point2f& corner,
                                                       const cv::Point2f& along_edge,
                                                       double edge_length,
                                                       const cv::Point2f& quad_center) {
  const cv::Point2f edge_dir = NormalizeVector(along_edge);
  if (Norm(edge_dir) <= 1e-9 || edge_length <= 1.0) {
    return {};
  }

  const double start_offset = std::min(std::max(6.0, edge_length * 0.04), edge_length * 0.15);
  const double segment_extent = std::min(std::max(40.0, edge_length * 0.25), 160.0);
  const double usable_extent = std::min(segment_extent, std::max(0.0, edge_length - start_offset - 2.0));
  if (usable_extent < 12.0) {
    return {};
  }

  const int sample_count =
      std::max(8, std::min(24, static_cast<int>(std::lround(usable_extent / 12.0))));
  const double search_radius = std::min(std::max(6.0, edge_length * 0.03), 24.0);
  return CollectDirectionalEdgeBranchPoints(gray, corner, edge_dir, edge_length, quad_center,
                                            start_offset, usable_extent, sample_count, search_radius);
}

std::vector<cv::Point2f> CollectCornerMarkerEdgeSupportPoints(
    const cv::Mat& gray,
    const cv::Point2f& corner,
    const cv::Point2f& along_edge,
    double edge_length,
    const cv::Point2f& quad_center,
    double corner_marker_width,
    int verification_roi_radius) {
  const cv::Point2f edge_dir = NormalizeVector(along_edge);
  if (Norm(edge_dir) <= 1e-9 || edge_length <= 1.0) {
    return {};
  }

  const double marker_extent = std::max(8.0, corner_marker_width);
  const double start_offset =
      std::min(std::max(1.5, 0.18 * marker_extent), std::max(2.0, edge_length * 0.18));
  const double local_extent =
      std::min(std::max(10.0, 1.20 * marker_extent), std::max(10.0, edge_length * 0.28));
  const double usable_extent =
      std::min(local_extent, std::max(0.0, edge_length - start_offset - 1.0));
  if (usable_extent < 6.0) {
    return {};
  }

  const int sample_count =
      std::max(5, std::min(16, static_cast<int>(std::lround(usable_extent / 3.0))));
  const double search_radius = std::max(
      2.0, std::min(std::max(4.0, static_cast<double>(verification_roi_radius) * 0.30),
                    std::max(3.0, 0.35 * marker_extent)));
  return CollectDirectionalEdgeBranchPoints(gray, corner, edge_dir, edge_length, quad_center,
                                            start_offset, usable_extent, sample_count, search_radius);
}

bool UnprojectSupportPointsToRays(const DoubleSphereCameraModel& camera,
                                  const std::vector<cv::Point2f>& image_points,
                                  std::vector<Eigen::Vector3d>* rays) {
  if (rays == nullptr) {
    throw std::runtime_error("UnprojectSupportPointsToRays requires a valid output pointer.");
  }

  rays->clear();
  rays->reserve(image_points.size());
  for (const cv::Point2f& point : image_points) {
    Eigen::Vector3d ray = Eigen::Vector3d::Zero();
    if (!camera.keypointToEuclidean(Eigen::Vector2d(point.x, point.y), &ray)) {
      continue;
    }
    const double norm = ray.norm();
    if (!std::isfinite(norm) || norm <= 1e-9) {
      continue;
    }
    rays->push_back(ray / norm);
  }
  return rays->size() >= static_cast<std::size_t>(kOuterSphereMinSupportPoints);
}

bool FitPlaneToRays(const std::vector<Eigen::Vector3d>& rays,
                    Eigen::Vector3d* plane_normal,
                    double* rms_residual) {
  if (plane_normal == nullptr || rms_residual == nullptr) {
    throw std::runtime_error("FitPlaneToRays requires valid output pointers.");
  }
  if (rays.size() < static_cast<std::size_t>(kOuterSphereMinSupportPoints)) {
    return false;
  }

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const Eigen::Vector3d& ray : rays) {
    covariance += ray * ray.transpose();
  }

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  Eigen::Vector3d normal = solver.eigenvectors().col(0);
  const double normal_norm = normal.norm();
  if (!std::isfinite(normal_norm) || normal_norm <= 1e-9) {
    return false;
  }
  normal /= normal_norm;

  double residual_sum_sq = 0.0;
  for (const Eigen::Vector3d& ray : rays) {
    const double residual = normal.dot(ray);
    residual_sum_sq += residual * residual;
  }

  *plane_normal = normal;
  *rms_residual = std::sqrt(residual_sum_sq / static_cast<double>(rays.size()));
  return std::isfinite(*rms_residual) && *rms_residual <= kOuterSpherePlaneResidualThreshold;
}

std::vector<cv::Point2f> ProjectSphericalPlaneCurve(const DoubleSphereCameraModel& camera,
                                                    const Eigen::Vector3d& plane_normal,
                                                    const Eigen::Vector3d& corner_ray,
                                                    const std::vector<Eigen::Vector3d>& support_rays) {
  std::vector<cv::Point2f> curve_points;
  if (support_rays.empty()) {
    return curve_points;
  }

  Eigen::Vector3d basis_a = corner_ray.normalized();
  Eigen::Vector3d basis_b = plane_normal.cross(basis_a);
  if (basis_b.norm() <= 1e-9) {
    return curve_points;
  }
  basis_b.normalize();

  double theta_min = 0.0;
  double theta_max = 0.0;
  bool initialized = false;
  for (const Eigen::Vector3d& ray : support_rays) {
    const double theta = std::atan2(ray.dot(basis_b), ray.dot(basis_a));
    if (!initialized) {
      theta_min = theta;
      theta_max = theta;
      initialized = true;
    } else {
      theta_min = std::min(theta_min, theta);
      theta_max = std::max(theta_max, theta);
    }
  }
  if (!initialized) {
    return curve_points;
  }

  theta_min = std::min(theta_min, 0.0) - 0.05;
  theta_max = std::max(theta_max, 0.0) + 0.05;
  const int sample_count = 24;
  curve_points.reserve(static_cast<std::size_t>(sample_count));
  for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
    const double alpha =
        sample_count == 1
            ? 0.0
            : static_cast<double>(sample_index) / static_cast<double>(sample_count - 1);
    const double theta = theta_min + (theta_max - theta_min) * alpha;
    const Eigen::Vector3d ray = (std::cos(theta) * basis_a + std::sin(theta) * basis_b).normalized();
    Eigen::Vector2d projected = Eigen::Vector2d::Zero();
    if (!camera.vsEuclideanToKeypoint(ray, &projected)) {
      continue;
    }
    curve_points.emplace_back(static_cast<float>(projected.x()), static_cast<float>(projected.y()));
  }
  return curve_points;
}

SphericalEdgePlaneFit FitSphericalEdgePlane(const cv::Mat& gray,
                                            const DoubleSphereCameraModel& camera,
                                            const cv::Point2f& corner,
                                            const cv::Point2f& edge,
                                            double edge_length,
                                            const cv::Point2f& quad_center,
                                            double corner_marker_width,
                                            int verification_roi_radius) {
  SphericalEdgePlaneFit fit;
  fit.support_points = CollectCornerMarkerEdgeSupportPoints(
      gray, corner, edge, edge_length, quad_center, corner_marker_width, verification_roi_radius);
  fit.support_count = static_cast<int>(fit.support_points.size());
  if (!UnprojectSupportPointsToRays(camera, fit.support_points, &fit.support_rays)) {
    return fit;
  }

  if (!FitPlaneToRays(fit.support_rays, &fit.plane_normal, &fit.rms_residual)) {
    return fit;
  }

  fit.valid = true;
  return fit;
}

SphericalCornerRefinement RefineCornerBySphericalPlanes(
    const cv::Mat& gray,
    const DoubleSphereCameraModel& camera,
    const std::array<cv::Point2f, 4>& corner_seeds,
    int corner_index,
    const MultiScaleOuterTagDetectorConfig& config) {
  SphericalCornerRefinement refinement;

  const int prev_index = (corner_index + 3) % 4;
  const int next_index = (corner_index + 1) % 4;
  const cv::Point2f corner = corner_seeds[static_cast<std::size_t>(corner_index)];
  const cv::Point2f prev_edge = corner_seeds[static_cast<std::size_t>(prev_index)] - corner;
  const cv::Point2f next_edge = corner_seeds[static_cast<std::size_t>(next_index)] - corner;
  const double prev_length = Norm(prev_edge);
  const double next_length = Norm(next_edge);
  if (prev_length <= 1.0 || next_length <= 1.0) {
    refinement.failure_reason = "short_edge";
    return refinement;
  }

  const double local_scale = std::min(prev_length, next_length);
  const double corner_marker_width = ComputeOuterCornerMarkerWidth(local_scale, config);
  const AdaptiveCornerSearchRadii radii =
      ComputeAdaptiveCornerSearchRadii(local_scale, config);
  const cv::Point2f quad_center = ComputeQuadCenter(corner_seeds);

  refinement.prev_edge_fit = FitSphericalEdgePlane(
      gray, camera, corner, prev_edge, prev_length, quad_center, corner_marker_width,
      radii.verification_roi_radius);
  refinement.next_edge_fit = FitSphericalEdgePlane(
      gray, camera, corner, next_edge, next_length, quad_center, corner_marker_width,
      radii.verification_roi_radius);
  if (!refinement.prev_edge_fit.valid || !refinement.next_edge_fit.valid) {
    refinement.failure_reason = "edge_fit";
    return refinement;
  }

  Eigen::Vector3d seed_ray = Eigen::Vector3d::Zero();
  if (!camera.keypointToEuclidean(Eigen::Vector2d(corner.x, corner.y), &seed_ray)) {
    refinement.failure_reason = "seed_ray";
    return refinement;
  }
  const double seed_norm = seed_ray.norm();
  if (!std::isfinite(seed_norm) || seed_norm <= 1e-9) {
    refinement.failure_reason = "seed_ray";
    return refinement;
  }
  seed_ray /= seed_norm;

  Eigen::Vector3d intersection_ray =
      refinement.prev_edge_fit.plane_normal.cross(refinement.next_edge_fit.plane_normal);
  const double intersection_norm = intersection_ray.norm();
  if (!std::isfinite(intersection_norm) || intersection_norm <= 1e-9) {
    refinement.failure_reason = "parallel_planes";
    return refinement;
  }
  intersection_ray /= intersection_norm;
  if (intersection_ray.dot(seed_ray) < 0.0) {
    intersection_ray = -intersection_ray;
  }

  Eigen::Vector2d projected = Eigen::Vector2d::Zero();
  if (!camera.vsEuclideanToKeypoint(intersection_ray, &projected)) {
    refinement.failure_reason = "projection";
    return refinement;
  }

  const cv::Point2f projected_corner(static_cast<float>(projected.x()),
                                     static_cast<float>(projected.y()));
  if (!IsInsideImage(projected_corner, gray.size(), 1.0f)) {
    refinement.failure_reason = "outside";
    return refinement;
  }

  refinement.refined_corner = projected_corner;
  refinement.refined_ray = intersection_ray;
  refinement.prev_curve_points = ProjectSphericalPlaneCurve(
      camera, refinement.prev_edge_fit.plane_normal, intersection_ray, refinement.prev_edge_fit.support_rays);
  refinement.next_curve_points = ProjectSphericalPlaneCurve(
      camera, refinement.next_edge_fit.plane_normal, intersection_ray, refinement.next_edge_fit.support_rays);

  const double residual_quality =
      ClampUnit(1.0 - std::max(refinement.prev_edge_fit.rms_residual,
                               refinement.next_edge_fit.rms_residual) /
                           kOuterSpherePlaneResidualThreshold);
  const double support_quality =
      ClampUnit(static_cast<double>(std::min(refinement.prev_edge_fit.support_count,
                                             refinement.next_edge_fit.support_count)) /
                8.0);
  refinement.quality = std::min(residual_quality, support_quality);
  refinement.success = true;
  refinement.failure_reason = "pass";
  return refinement;
}

bool FitSupportImageLine(const std::vector<cv::Point2f>& points, FittedLine* fitted_line) {
  if (fitted_line == nullptr) {
    throw std::runtime_error("FitSupportImageLine requires a valid output pointer.");
  }
  fitted_line->support_count = static_cast<int>(points.size());
  if (points.size() < 2) {
    fitted_line->valid = false;
    fitted_line->rms_residual = std::numeric_limits<double>::infinity();
    return false;
  }

  cv::Vec4f line;
  cv::fitLine(points, line, cv::DIST_L2, 0.0, 0.01, 0.01);
  fitted_line->anchor = cv::Point2f(line[2], line[3]);
  fitted_line->direction = NormalizeVector(cv::Point2f(line[0], line[1]));
  double residual_sum_sq = 0.0;
  for (const cv::Point2f& point : points) {
    const cv::Point2f delta = point - fitted_line->anchor;
    const double residual = std::abs(Cross(delta, fitted_line->direction));
    residual_sum_sq += residual * residual;
  }
  fitted_line->rms_residual = std::sqrt(residual_sum_sq / static_cast<double>(points.size()));
  fitted_line->valid = std::isfinite(fitted_line->rms_residual) &&
                       Norm(fitted_line->direction) > 1e-9;
  return fitted_line->valid;
}

ImageLineCornerRefinement RefineCornerByImageLineSupportIntersection(
    const std::vector<cv::Point2f>& prev_support_points,
    const std::vector<cv::Point2f>& next_support_points) {
  ImageLineCornerRefinement refinement;
  FitSupportImageLine(prev_support_points, &refinement.prev_line);
  FitSupportImageLine(next_support_points, &refinement.next_line);
  if (!refinement.prev_line.valid || !refinement.next_line.valid) {
    refinement.failure_reason = "line_fit";
    return refinement;
  }
  if (!IntersectLines(refinement.prev_line, refinement.next_line, &refinement.refined_corner)) {
    refinement.failure_reason = "parallel_lines";
    return refinement;
  }

  const double residual_quality =
      ClampUnit(1.0 - std::max(refinement.prev_line.rms_residual,
                               refinement.next_line.rms_residual) /
                           std::max(1.0, kOuterLineResidualThreshold * 2.0));
  const double support_quality =
      ClampUnit(static_cast<double>(std::min(refinement.prev_line.support_count,
                                             refinement.next_line.support_count)) /
                8.0);
  refinement.quality = std::min(residual_quality, support_quality);
  refinement.success = true;
  refinement.failure_reason = "pass";
  return refinement;
}

bool FitLineToPoints(const std::vector<cv::Point2f>& points,
                     FittedLine* fitted_line,
                     int min_support_points = kMinLineSupportPoints) {
  if (fitted_line == nullptr) {
    throw std::runtime_error("FitLineToPoints requires a valid output pointer.");
  }
  if (points.size() < static_cast<std::size_t>(std::max(2, min_support_points))) {
    return false;
  }

  cv::Vec4f line;
  cv::fitLine(points, line, cv::DIST_L2, 0.0, 0.01, 0.01);

  fitted_line->anchor = cv::Point2f(line[2], line[3]);
  fitted_line->direction = NormalizeVector(cv::Point2f(line[0], line[1]));
  fitted_line->support_count = static_cast<int>(points.size());

  double residual_sum_sq = 0.0;
  for (const cv::Point2f& point : points) {
    const cv::Point2f delta = point - fitted_line->anchor;
    const double residual = std::abs(Cross(delta, fitted_line->direction));
    residual_sum_sq += residual * residual;
  }

  fitted_line->rms_residual = std::sqrt(residual_sum_sq / static_cast<double>(points.size()));
  fitted_line->valid = std::isfinite(fitted_line->rms_residual) &&
                       fitted_line->rms_residual <= kOuterLineResidualThreshold;
  return fitted_line->valid;
}

DirectionalEdgeBranch ExtractDirectionalEdgeBranch(const cv::Mat& gray,
                                                   const cv::Point2f& candidate_corner,
                                                   const cv::Point2f& expected_edge,
                                                   double edge_length,
                                                   const cv::Point2f& quad_center,
                                                   const AdaptiveCornerSearchRadii& radii) {
  DirectionalEdgeBranch branch;
  const cv::Point2f edge_dir = NormalizeVector(expected_edge);
  if (Norm(edge_dir) <= 1e-9 || edge_length <= 1.0) {
    return branch;
  }

  const double start_offset = std::min(2.5, std::max(1.0, edge_length * 0.03));
  const double local_extent = std::min(
      std::max(8.0, static_cast<double>(radii.verification_roi_radius) * 0.85),
      std::max(8.0, edge_length * 0.20));
  const double usable_extent = std::min(local_extent, std::max(0.0, edge_length - start_offset - 1.0));
  if (usable_extent < 6.0) {
    return branch;
  }

  const int sample_count =
      std::max(4, std::min(10, static_cast<int>(std::lround(usable_extent / 2.5))));
  const double search_radius = static_cast<double>(std::max(2, radii.branch_search_radius));

  branch.support_points =
      CollectDirectionalEdgeBranchPoints(gray, candidate_corner, edge_dir, edge_length, quad_center,
                                         start_offset, usable_extent, sample_count, search_radius);
  if (!FitLineToPoints(branch.support_points, &branch.fitted_line, kVerificationLineMinSupportPoints)) {
    return branch;
  }

  branch.valid = true;
  return branch;
}

double ScoreDirectionalBranch(const DirectionalEdgeBranch& branch, const cv::Point2f& expected_edge) {
  if (!branch.valid) {
    return 0.0;
  }

  const cv::Point2f expected_dir = NormalizeVector(expected_edge);
  const double alignment =
      std::abs(Dot(branch.fitted_line.direction, expected_dir));
  const double alignment_score =
      ClampUnit((alignment - kOuterDirectionAlignmentFloor) / (1.0 - kOuterDirectionAlignmentFloor));
  const double support_score = ClampUnit(static_cast<double>(branch.fitted_line.support_count) / 6.0);
  const double residual_score =
      ClampUnit(1.0 - branch.fitted_line.rms_residual / (kOuterLineResidualThreshold * 1.2));
  return std::min(alignment_score, std::min(support_score, residual_score));
}

double ScoreCornerDirectionConsistency(DirectionalEdgeBranch* prev_branch,
                                       DirectionalEdgeBranch* next_branch,
                                       const cv::Point2f& prev_edge,
                                       const cv::Point2f& next_edge) {
  if (prev_branch == nullptr || next_branch == nullptr) {
    throw std::runtime_error("ScoreCornerDirectionConsistency requires valid branch pointers.");
  }

  prev_branch->score = ScoreDirectionalBranch(*prev_branch, prev_edge);
  next_branch->score = ScoreDirectionalBranch(*next_branch, next_edge);
  return std::min(prev_branch->score, next_branch->score);
}

double ScoreOuterCornerLocalLayout(const cv::Mat& gray,
                                   const cv::Point2f& candidate_corner,
                                   const cv::Point2f& prev_edge,
                                   const cv::Point2f& next_edge,
                                   const cv::Point2f& quad_center,
                                   const AdaptiveCornerSearchRadii& radii) {
  cv::Point2f inside_dir = NormalizeVector(quad_center - candidate_corner);
  if (Norm(inside_dir) <= 1e-9) {
    inside_dir = NormalizeVector(prev_edge + next_edge);
  }
  if (Norm(inside_dir) <= 1e-9) {
    return 0.0;
  }

  cv::Point2f lateral_dir = NormalizeVector(prev_edge - next_edge);
  if (Norm(lateral_dir) <= 1e-9) {
    lateral_dir = NormalizeVector(PerpendicularLeft(inside_dir));
  }
  if (Norm(lateral_dir) <= 1e-9) {
    return 0.0;
  }

  const double distance =
      std::min(8.0, std::max(3.0, static_cast<double>(radii.verification_roi_radius) * 0.28));
  const double spread =
      std::min(4.0, std::max(1.5, static_cast<double>(radii.verification_roi_radius) * 0.16));

  const std::vector<cv::Point2f> inside_points{
      candidate_corner + inside_dir * static_cast<float>(distance),
      candidate_corner + inside_dir * static_cast<float>(distance) + lateral_dir * static_cast<float>(spread),
      candidate_corner + inside_dir * static_cast<float>(distance) - lateral_dir * static_cast<float>(spread),
  };
  const std::vector<cv::Point2f> outside_points{
      candidate_corner - inside_dir * static_cast<float>(distance),
      candidate_corner - inside_dir * static_cast<float>(distance) + lateral_dir * static_cast<float>(spread),
      candidate_corner - inside_dir * static_cast<float>(distance) - lateral_dir * static_cast<float>(spread),
  };

  const double inside_mean = SampleMeanAtPoints(gray, inside_points);
  const double outside_mean = SampleMeanAtPoints(gray, outside_points);
  if (!std::isfinite(inside_mean) || !std::isfinite(outside_mean)) {
    return 0.0;
  }

  return ClampUnit((outside_mean - inside_mean - kOuterLayoutContrastFloor) /
                   kOuterLayoutContrastRange);
}

OuterCornerVerificationDebugInfo BuildVerificationDebugInfo(
    int corner_index,
    const cv::Point2f& coarse_corner,
    const OuterCornerLocalVerificationResult& verification) {
  OuterCornerVerificationDebugInfo debug;
  debug.corner_index = corner_index;
  debug.coarse_corner = coarse_corner;
  debug.verified_corner = verification.verified_corner;
  debug.subpix_corner = verification.verified_corner;
  debug.verification_roi = verification.verification_roi;
  debug.prev_edge_direction = verification.prev_edge_direction;
  debug.next_edge_direction = verification.next_edge_direction;
  debug.prev_branch_points = verification.prev_branch.support_points;
  debug.next_branch_points = verification.next_branch.support_points;
  debug.local_scale = verification.local_scale;
  debug.corner_marker_width = verification.corner_marker_width;
  debug.verification_roi_radius = verification.verification_roi_radius;
  debug.candidate_radius = verification.candidate_radius;
  debug.branch_search_radius = verification.branch_search_radius;
  debug.direction_consistency_score = verification.direction_consistency_score;
  debug.local_layout_score = verification.local_layout_score;
  debug.verification_quality = verification.verification_quality;
  debug.subpix_window_radius = 0;
  debug.verification_passed = verification.verification_passed;
  debug.subpix_applied = false;
  debug.failure_reason = verification.failure_reason;
  return debug;
}

OuterCornerVerificationDebugInfo BuildCoarseOnlyDebugInfo(
    int corner_index,
    const std::array<cv::Point2f, 4>& coarse_corners,
    const cv::Size& image_size,
    const MultiScaleOuterTagDetectorConfig& config) {
  OuterCornerVerificationDebugInfo debug;
  debug.corner_index = corner_index;
  debug.coarse_corner = coarse_corners[static_cast<std::size_t>(corner_index)];
  debug.verified_corner = debug.coarse_corner;
  debug.subpix_corner = debug.coarse_corner;
  const int prev_index = (corner_index + 3) % 4;
  const int next_index = (corner_index + 1) % 4;
  const cv::Point2f prev_edge =
      coarse_corners[static_cast<std::size_t>(prev_index)] - debug.coarse_corner;
  const cv::Point2f next_edge =
      coarse_corners[static_cast<std::size_t>(next_index)] - debug.coarse_corner;
  debug.prev_edge_direction = NormalizeVector(prev_edge);
  debug.next_edge_direction = NormalizeVector(next_edge);
  debug.local_scale = std::min(Norm(prev_edge), Norm(next_edge));
  debug.corner_marker_width = ComputeOuterCornerMarkerWidth(debug.local_scale, config);
  const AdaptiveCornerSearchRadii radii =
      ComputeAdaptiveCornerSearchRadii(debug.local_scale, config);
  debug.verification_roi_radius = radii.verification_roi_radius;
  debug.candidate_radius = radii.candidate_radius;
  debug.branch_search_radius = radii.branch_search_radius;
  debug.verification_roi =
      MakeCornerVerificationRoi(debug.coarse_corner, image_size, radii.verification_roi_radius);
  debug.verification_passed = true;
  debug.verification_quality = 1.0;
  debug.direction_consistency_score = 1.0;
  debug.local_layout_score = 1.0;
  debug.failure_reason = "cs_only";
  return debug;
}

OuterCornerLocalVerificationResult VerifyOuterCornerLocalStructure(
    const cv::Mat& gray,
    const std::array<cv::Point2f, 4>& coarse_corners,
    int corner_index,
    const MultiScaleOuterTagDetectorConfig& config) {
  OuterCornerLocalVerificationResult best_any;
  best_any.verification_quality = -1.0;
  best_any.local_layout_score = 0.0;
  best_any.direction_consistency_score = 0.0;

  const cv::Point2f coarse_corner = coarse_corners[static_cast<std::size_t>(corner_index)];
  const int prev_index = (corner_index + 3) % 4;
  const int next_index = (corner_index + 1) % 4;
  const cv::Point2f prev_edge =
      coarse_corners[static_cast<std::size_t>(prev_index)] - coarse_corner;
  const cv::Point2f next_edge =
      coarse_corners[static_cast<std::size_t>(next_index)] - coarse_corner;
  const double prev_length = Norm(prev_edge);
  const double next_length = Norm(next_edge);
  const double local_scale = std::min(prev_length, next_length);
  const AdaptiveCornerSearchRadii radii =
      ComputeAdaptiveCornerSearchRadii(local_scale, config);
  const cv::Point2f quad_center = ComputeQuadCenter(coarse_corners);

  best_any.verified_corner = coarse_corner;
  best_any.verification_roi =
      MakeCornerVerificationRoi(coarse_corner, gray.size(), radii.verification_roi_radius);
  best_any.prev_edge_direction = NormalizeVector(prev_edge);
  best_any.next_edge_direction = NormalizeVector(next_edge);
  best_any.local_scale = radii.local_scale;
  best_any.corner_marker_width = ComputeOuterCornerMarkerWidth(radii.local_scale, config);
  best_any.verification_roi_radius = radii.verification_roi_radius;
  best_any.candidate_radius = radii.candidate_radius;
  best_any.branch_search_radius = radii.branch_search_radius;

  const int candidate_radius = radii.candidate_radius;
  OuterCornerLocalVerificationResult best_passed;
  best_passed.verification_quality = -1.0;

  for (int dy = -candidate_radius; dy <= candidate_radius; dy += kOuterVerificationCandidateStepPixels) {
    for (int dx = -candidate_radius; dx <= candidate_radius; dx += kOuterVerificationCandidateStepPixels) {
      const cv::Point2f candidate_corner =
          coarse_corner + cv::Point2f(static_cast<float>(dx), static_cast<float>(dy));
      if (dx != 0 || dy != 0) {
        const cv::Point2f candidate_delta = candidate_corner - coarse_corner;
        if (std::hypot(candidate_delta.x, candidate_delta.y) >
            static_cast<double>(candidate_radius) + 1e-6) {
          continue;
        }
      }
      if (!IsInsideImage(candidate_corner, gray.size(), 2.0f)) {
        continue;
      }

      OuterCornerLocalVerificationResult candidate_result;
      candidate_result.verified_corner = candidate_corner;
      candidate_result.verification_roi = best_any.verification_roi;
      candidate_result.prev_edge_direction = NormalizeVector(prev_edge);
      candidate_result.next_edge_direction = NormalizeVector(next_edge);
      candidate_result.local_scale = radii.local_scale;
      candidate_result.corner_marker_width = ComputeOuterCornerMarkerWidth(radii.local_scale, config);
      candidate_result.verification_roi_radius = radii.verification_roi_radius;
      candidate_result.candidate_radius = radii.candidate_radius;
      candidate_result.branch_search_radius = radii.branch_search_radius;
      candidate_result.prev_branch = ExtractDirectionalEdgeBranch(
          gray, candidate_corner, prev_edge, prev_length, quad_center, radii);
      candidate_result.next_branch = ExtractDirectionalEdgeBranch(
          gray, candidate_corner, next_edge, next_length, quad_center, radii);
      candidate_result.direction_consistency_score =
          ScoreCornerDirectionConsistency(&candidate_result.prev_branch, &candidate_result.next_branch,
                                          prev_edge, next_edge);
      candidate_result.local_layout_score =
          config.enable_outer_corner_layout_check
              ? ScoreOuterCornerLocalLayout(gray, candidate_corner, prev_edge, next_edge, quad_center, radii)
              : 1.0;
      candidate_result.verification_quality = candidate_result.direction_consistency_score;
      candidate_result.verification_passed =
          candidate_result.direction_consistency_score >= config.outer_corner_min_direction_score &&
          (!config.enable_outer_corner_layout_check ||
           candidate_result.local_layout_score >= config.outer_corner_min_layout_score) &&
          candidate_result.prev_branch.valid &&
          candidate_result.next_branch.valid;

      if (candidate_result.verification_quality > best_any.verification_quality) {
        best_any = candidate_result;
      }
      if (candidate_result.verification_passed &&
          candidate_result.verification_quality > best_passed.verification_quality) {
        best_passed = candidate_result;
      }
    }
  }

  OuterCornerLocalVerificationResult final_result =
      best_passed.verification_quality >= 0.0 ? best_passed : best_any;
  if (final_result.verification_passed) {
    final_result.failure_reason = "pass";
    return final_result;
  }

  if (config.enable_outer_corner_layout_check &&
      final_result.direction_consistency_score < config.outer_corner_min_direction_score &&
      final_result.local_layout_score < config.outer_corner_min_layout_score) {
    final_result.failure_reason = "dir+layout";
  } else if (final_result.direction_consistency_score < config.outer_corner_min_direction_score) {
    final_result.failure_reason = "dir";
  } else if (config.enable_outer_corner_layout_check &&
             final_result.local_layout_score < config.outer_corner_min_layout_score) {
    final_result.failure_reason = "layout";
  } else if (!final_result.prev_branch.valid || !final_result.next_branch.valid) {
    final_result.failure_reason = "missing_branch";
  } else {
    final_result.failure_reason = "quality";
  }
  return final_result;
}

bool IntersectLines(const FittedLine& first, const FittedLine& second, cv::Point2f* intersection) {
  if (intersection == nullptr) {
    throw std::runtime_error("IntersectLines requires a valid output pointer.");
  }

  const double denominator = Cross(first.direction, second.direction);
  if (std::abs(denominator) <= 1e-6) {
    return false;
  }

  const cv::Point2f delta = second.anchor - first.anchor;
  const double distance_along_first = Cross(delta, second.direction) / denominator;
  const cv::Point2f point =
      first.anchor + first.direction * static_cast<float>(distance_along_first);
  if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
    return false;
  }

  *intersection = point;
  return true;
}

CornerLineRefinement RefineCornerByLineIntersection(const cv::Mat& gray,
                                                    const std::array<cv::Point2f, 4>& coarse_corners,
                                                    int corner_index) {
  CornerLineRefinement refinement;
  const int prev_index = (corner_index + 3) % 4;
  const int next_index = (corner_index + 1) % 4;

  const cv::Point2f corner = coarse_corners[static_cast<std::size_t>(corner_index)];
  const cv::Point2f prev = coarse_corners[static_cast<std::size_t>(prev_index)];
  const cv::Point2f next = coarse_corners[static_cast<std::size_t>(next_index)];
  const cv::Point2f quad_center = ComputeQuadCenter(coarse_corners);

  const cv::Point2f prev_edge = prev - corner;
  const cv::Point2f next_edge = next - corner;
  const double prev_length = Norm(prev_edge);
  const double next_length = Norm(next_edge);
  if (prev_length <= 1.0 || next_length <= 1.0) {
    return refinement;
  }

  const std::vector<cv::Point2f> prev_support =
      CollectLocalEdgeSupportPoints(gray, corner, prev_edge, prev_length, quad_center);
  const std::vector<cv::Point2f> next_support =
      CollectLocalEdgeSupportPoints(gray, corner, next_edge, next_length, quad_center);

  FittedLine prev_line;
  FittedLine next_line;
  if (!FitLineToPoints(prev_support, &prev_line) || !FitLineToPoints(next_support, &next_line)) {
    return refinement;
  }

  cv::Point2f intersection;
  if (!IntersectLines(prev_line, next_line, &intersection)) {
    return refinement;
  }

  refinement.success = true;
  refinement.refined_corner = intersection;
  const double residual_quality =
      ClampUnit(1.0 - std::max(prev_line.rms_residual, next_line.rms_residual) / kOuterLineResidualThreshold);
  const double support_quality =
      ClampUnit(static_cast<double>(std::min(prev_line.support_count, next_line.support_count)) / 12.0);
  refinement.quality = std::min(residual_quality, support_quality);
  return refinement;
}

bool IsCandidateBetter(const ScaleCandidate& lhs, const ScaleCandidate& rhs) {
  if (lhs.detection.good != rhs.detection.good) {
    return lhs.detection.good && !rhs.detection.good;
  }
  if (lhs.detection.hammingDistance != rhs.detection.hammingDistance) {
    return lhs.detection.hammingDistance < rhs.detection.hammingDistance;
  }
  if (std::abs(lhs.scaled_area - rhs.scaled_area) > 1e-6) {
    return lhs.scaled_area > rhs.scaled_area;
  }
  return lhs.target_longest_side > rhs.target_longest_side;
}

bool IsRefinedCandidateBetter(const RefinedCandidate& lhs, const RefinedCandidate& rhs) {
  if (lhs.coarse.detection.good != rhs.coarse.detection.good) {
    return lhs.coarse.detection.good && !rhs.coarse.detection.good;
  }
  if (lhs.coarse.detection.hammingDistance != rhs.coarse.detection.hammingDistance) {
    return lhs.coarse.detection.hammingDistance < rhs.coarse.detection.hammingDistance;
  }
  if (std::abs(lhs.coarse.scaled_area - rhs.coarse.scaled_area) > 1e-6) {
    return lhs.coarse.scaled_area > rhs.coarse.scaled_area;
  }
  if (std::abs(lhs.quality - rhs.quality) > 1e-6) {
    return lhs.quality > rhs.quality;
  }
  return lhs.coarse.target_longest_side > rhs.coarse.target_longest_side;
}

double ComputeCornerLocalScale(const std::array<cv::Point2f, 4>& corners, int corner_index) {
  const int prev_index = (corner_index + 3) % 4;
  const int next_index = (corner_index + 1) % 4;
  const cv::Point2f corner = corners[static_cast<std::size_t>(corner_index)];
  const cv::Point2f prev_edge = corners[static_cast<std::size_t>(prev_index)] - corner;
  const cv::Point2f next_edge = corners[static_cast<std::size_t>(next_index)] - corner;
  return std::min(Norm(prev_edge), Norm(next_edge));
}

cv::Mat MaybeBlur(const cv::Mat& image, const MultiScaleOuterTagDetectorConfig& config) {
  if (!config.blur_before_detect) {
    return image;
  }

  int kernel = std::max(1, config.blur_kernel);
  if (kernel % 2 == 0) {
    ++kernel;
  }

  cv::Mat blurred;
  cv::GaussianBlur(image, blurred, cv::Size(kernel, kernel), config.blur_sigma);
  return blurred;
}

double ComputeMedian(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return 0.5 * (values[middle - 1] + values[middle]);
  }
  return values[middle];
}

cv::Point2f ComputeMedianPoint(const std::vector<cv::Point2f>& points) {
  std::vector<double> xs;
  std::vector<double> ys;
  xs.reserve(points.size());
  ys.reserve(points.size());
  for (const cv::Point2f& point : points) {
    xs.push_back(static_cast<double>(point.x));
    ys.push_back(static_cast<double>(point.y));
  }
  return cv::Point2f(static_cast<float>(ComputeMedian(xs)),
                     static_cast<float>(ComputeMedian(ys)));
}

cv::Point2f ComputeWeightedAveragePoint(const std::vector<CornerFusionObservation>& observations,
                                        const std::vector<bool>& use_mask) {
  double weight_sum = 0.0;
  cv::Point2f weighted_sum(0.0f, 0.0f);
  for (std::size_t index = 0; index < observations.size(); ++index) {
    if (!use_mask.empty() && !use_mask[index]) {
      continue;
    }
    const double weight = std::max(1e-6, observations[index].weight);
    weighted_sum += observations[index].point * static_cast<float>(weight);
    weight_sum += weight;
  }

  if (weight_sum <= 1e-9) {
    return observations.empty() ? cv::Point2f() : observations.front().point;
  }
  return weighted_sum * static_cast<float>(1.0 / weight_sum);
}

double ComputeCandidateFusionWeight(const ScaleCandidate& candidate) {
  const double hamming_quality =
      1.0 / static_cast<double>(1 + std::max(0, candidate.detection.hammingDistance));
  const double scale_weight = std::max(0.25, candidate.scale_factor);
  return std::max(1e-3, hamming_quality * std::max(0.20, candidate.shape_quality) * scale_weight);
}

std::array<cv::Point2f, 4> ProjectOriginalCornersToScaledImage(
    const std::array<cv::Point2f, 4>& original_corners,
    const cv::Size& original_size,
    const cv::Size& scaled_size) {
  std::array<cv::Point2f, 4> scaled_corners{};
  const double scale_x =
      static_cast<double>(std::max(1, scaled_size.width)) / static_cast<double>(std::max(1, original_size.width));
  const double scale_y =
      static_cast<double>(std::max(1, scaled_size.height)) / static_cast<double>(std::max(1, original_size.height));
  for (int index = 0; index < 4; ++index) {
    scaled_corners[static_cast<std::size_t>(index)] = cv::Point2f(
        static_cast<float>(original_corners[static_cast<std::size_t>(index)].x * scale_x),
        static_cast<float>(original_corners[static_cast<std::size_t>(index)].y * scale_y));
  }
  return scaled_corners;
}

std::vector<ScalePlanEntry> BuildScalePlan(const cv::Size& original_size,
                                           const MultiScaleOuterTagDetectorConfig& config,
                                           std::string* scale_mode_used) {
  const int original_longest = std::max(original_size.width, original_size.height);
  std::vector<ScalePlanEntry> plan;
  if (scale_mode_used != nullptr) {
    *scale_mode_used = config.enable_adaptive_scale_cascade
                           ? "adaptive_cascade"
                           : "fixed_schedule";
  }

  auto append_entry = [&](int target_longest_side, double configured_scale_divisor) {
    if (target_longest_side <= 0) {
      return;
    }
    target_longest_side = std::min(target_longest_side, original_longest);
    if (std::find_if(plan.begin(), plan.end(), [&](const ScalePlanEntry& entry) {
          return entry.target_longest_side == target_longest_side;
        }) != plan.end()) {
      return;
    }
    ScalePlanEntry entry;
    entry.target_longest_side = target_longest_side;
    entry.configured_scale_divisor = configured_scale_divisor;
    plan.push_back(entry);
  };

  const auto append_divisors = [&](const std::vector<double>& divisors) {
    for (const double divisor : divisors) {
      if (divisor <= 0.0) {
        continue;
      }
      const int target_longest_side = std::max(
          1, static_cast<int>(std::lround(
                 static_cast<double>(original_longest) / divisor)));
      append_entry(target_longest_side, divisor);
    }
  };
  if (config.enable_adaptive_scale_cascade) {
    append_divisors(config.adaptive_coarse_scale_divisors);
    append_divisors(config.adaptive_fallback_scale_divisors);
  } else {
    append_divisors(std::vector<double>(
        std::begin(kOuterFixedScaleDivisors),
        std::end(kOuterFixedScaleDivisors)));
  }

  if (config.max_scales_to_try > 0 && static_cast<int>(plan.size()) > config.max_scales_to_try) {
    plan.resize(static_cast<std::size_t>(config.max_scales_to_try));
  }

  return plan;
}

cv::Size MakeScaledSize(const cv::Size& original_size, int target_longest_side) {
  const int original_longest = std::max(original_size.width, original_size.height);
  if (target_longest_side >= original_longest) {
    return original_size;
  }

  const double scale = static_cast<double>(target_longest_side) / static_cast<double>(original_longest);
  const int scaled_width = std::max(1, static_cast<int>(std::lround(original_size.width * scale)));
  const int scaled_height = std::max(1, static_cast<int>(std::lround(original_size.height * scale)));
  return cv::Size(scaled_width, scaled_height);
}

CornerFusionOutcome FuseCornerObservations(const std::vector<CornerFusionObservation>& observations) {
  CornerFusionOutcome outcome;
  if (observations.empty()) {
    return outcome;
  }

  std::vector<cv::Point2f> points;
  points.reserve(observations.size());
  for (const CornerFusionObservation& observation : observations) {
    points.push_back(observation.point);
  }

  outcome.consensus_corner = ComputeMedianPoint(points);
  outcome.inlier_mask.assign(observations.size(), true);
  outcome.deviations_before.reserve(observations.size());
  std::vector<double> distances;
  distances.reserve(observations.size());
  for (const CornerFusionObservation& observation : observations) {
    const double distance = Norm(observation.point - outcome.consensus_corner);
    outcome.deviations_before.push_back(distance);
    distances.push_back(distance);
  }

  if (!distances.empty()) {
    outcome.average_deviation_before =
        std::accumulate(distances.begin(), distances.end(), 0.0) /
        static_cast<double>(distances.size());
    outcome.max_deviation_before =
        *std::max_element(distances.begin(), distances.end());
  }

  if (observations.size() <= 2) {
    outcome.outlier_threshold = std::max(3.0, outcome.max_deviation_before);
  } else {
    const double median_distance = ComputeMedian(distances);
    std::vector<double> absolute_deviations;
    absolute_deviations.reserve(distances.size());
    for (const double distance : distances) {
      absolute_deviations.push_back(std::abs(distance - median_distance));
    }
    const double mad = ComputeMedian(absolute_deviations);
    outcome.outlier_threshold =
        std::max({3.0, 1.5 * median_distance, median_distance + 2.5 * std::max(mad, 0.75)});
    for (std::size_t index = 0; index < distances.size(); ++index) {
      outcome.inlier_mask[index] = distances[index] <= outcome.outlier_threshold + 1e-6;
    }
  }

  int inlier_count = static_cast<int>(std::count(outcome.inlier_mask.begin(),
                                                 outcome.inlier_mask.end(), true));
  const int min_required_inliers = observations.size() >= 3 ? 2 : 1;
  if (inlier_count < min_required_inliers) {
    std::fill(outcome.inlier_mask.begin(), outcome.inlier_mask.end(), false);
    std::vector<std::size_t> order(distances.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
      return distances[lhs] < distances[rhs];
    });
    for (int keep_index = 0; keep_index < min_required_inliers && keep_index < static_cast<int>(order.size());
         ++keep_index) {
      outcome.inlier_mask[order[static_cast<std::size_t>(keep_index)]] = true;
    }
    inlier_count = min_required_inliers;
  }

  outcome.inlier_count = inlier_count;
  outcome.outlier_count = static_cast<int>(observations.size()) - outcome.inlier_count;
  outcome.used_outlier_rejection = outcome.outlier_count > 0;
  outcome.fused_corner = ComputeWeightedAveragePoint(observations, outcome.inlier_mask);

  outcome.deviations_after.reserve(observations.size());
  double inlier_distance_sum = 0.0;
  for (std::size_t index = 0; index < observations.size(); ++index) {
    const double distance = Norm(observations[index].point - outcome.fused_corner);
    outcome.deviations_after.push_back(distance);
    if (!outcome.inlier_mask[index]) {
      continue;
    }
    inlier_distance_sum += distance;
    outcome.max_deviation_after = std::max(outcome.max_deviation_after, distance);
  }
  if (outcome.inlier_count > 0) {
    outcome.average_deviation_after = inlier_distance_sum / static_cast<double>(outcome.inlier_count);
  }
  outcome.stable_after_fusion =
      outcome.inlier_count > 0 &&
      outcome.max_deviation_after <= std::max(2.5, 0.75 * outcome.outlier_threshold + 1e-6);
  return outcome;
}

MultiScaleCornerFusionOutcome FuseMultiScaleCoarseCorners(
    const std::vector<ScaleCandidate>& coarse_candidates,
    const cv::Size& original_size,
    const MultiScaleOuterTagDetectorConfig& config) {
  MultiScaleCornerFusionOutcome outcome;
  if (coarse_candidates.empty()) {
    return outcome;
  }

  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    std::vector<CornerFusionObservation> observations;
    observations.reserve(coarse_candidates.size());
    for (const ScaleCandidate& candidate : coarse_candidates) {
      CornerFusionObservation observation;
      observation.point = candidate.original_corners[static_cast<std::size_t>(corner_index)];
      observation.weight = ComputeCandidateFusionWeight(candidate);
      observation.target_longest_side = candidate.target_longest_side;
      observation.scale_factor = candidate.scale_factor;
      observation.configured_scale_divisor = candidate.configured_scale_divisor;
      observations.push_back(observation);
    }

    const CornerFusionOutcome corner_outcome = FuseCornerObservations(observations);
    outcome.fused_corners[static_cast<std::size_t>(corner_index)] = corner_outcome.fused_corner;

    OuterCornerFusionDebugInfo debug;
    debug.corner_index = corner_index;
    debug.successful_scale_count = static_cast<int>(observations.size());
    debug.inlier_count = corner_outcome.inlier_count;
    debug.outlier_count = corner_outcome.outlier_count;
    debug.outlier_threshold = corner_outcome.outlier_threshold;
    debug.average_deviation_before = corner_outcome.average_deviation_before;
    debug.max_deviation_before = corner_outcome.max_deviation_before;
    debug.average_deviation_after = corner_outcome.average_deviation_after;
    debug.max_deviation_after = corner_outcome.max_deviation_after;
    debug.used_outlier_rejection = corner_outcome.used_outlier_rejection;
    debug.stable_after_fusion = corner_outcome.stable_after_fusion;
    debug.consensus_corner = corner_outcome.consensus_corner;
    debug.fused_corner = corner_outcome.fused_corner;
    for (std::size_t observation_index = 0; observation_index < observations.size(); ++observation_index) {
      OuterCornerScaleObservationDebugInfo observation_debug;
      observation_debug.target_longest_side = observations[observation_index].target_longest_side;
      observation_debug.scale_factor = observations[observation_index].scale_factor;
      observation_debug.configured_scale_divisor =
          observations[observation_index].configured_scale_divisor;
      observation_debug.coarse_corner = observations[observation_index].point;
      if (observation_index < corner_outcome.deviations_before.size()) {
        observation_debug.deviation_from_consensus =
            corner_outcome.deviations_before[observation_index];
      }
      if (observation_index < corner_outcome.deviations_after.size()) {
        observation_debug.deviation_from_fused =
            corner_outcome.deviations_after[observation_index];
      }
      if (observation_index < corner_outcome.inlier_mask.size()) {
        observation_debug.rejected_as_outlier =
            !corner_outcome.inlier_mask[observation_index];
      }
      debug.scale_observations.push_back(observation_debug);
    }
    outcome.debug[static_cast<std::size_t>(corner_index)] = debug;
  }

  const std::pair<double, double> fused_edge_range = ComputeEdgeRange(outcome.fused_corners);
  const double fused_shape_quality =
      fused_edge_range.second > 1e-6 ? ClampUnit(fused_edge_range.first / fused_edge_range.second) : 0.0;
  outcome.valid =
      PassesBorderCheck(outcome.fused_corners, original_size, config.min_border_distance) &&
      ComputeQuadArea(outcome.fused_corners) >= kMinQuadAreaPixels &&
      fused_edge_range.first >= kMinQuadEdgePixels &&
      fused_shape_quality > 0.10;
  return outcome;
}

RefinedCandidate RefineCoarseCandidate(const cv::Mat& gray_original,
                                       const ScaleCandidate& coarse_candidate,
                                       const std::array<cv::Point2f, 4>& coarse_original,
                                       const MultiScaleOuterTagDetectorConfig& config,
                                       const DoubleSphereCameraModel* sphere_camera) {
  RefinedCandidate refined_candidate;
  refined_candidate.coarse = coarse_candidate;
  refined_candidate.coarse_original = coarse_original;
  refined_candidate.refined_original = coarse_original;

  if (!PassesBorderCheck(coarse_original, gray_original.size(), config.min_border_distance)) {
    return refined_candidate;
  }

  std::array<bool, 4> method_valid{{false, false, false, false}};
  std::array<double, 4> method_quality{{0.0, 0.0, 0.0, 0.0}};
  const CloseEdgeOuterSubpixBoostInfo close_edge_boost_info =
      ComputeCloseEdgeOuterSubpixBoostInfo(coarse_original, gray_original.size(),
                                           config, sphere_camera);
  const bool close_edge_subpix_boost = close_edge_boost_info.boost;
  const double close_edge_area_ratio = close_edge_boost_info.area_ratio;
  const double close_edge_max_polar_deg = close_edge_boost_info.max_polar_deg;
  const double close_edge_subpix_multiplier =
      std::max(1.0, close_edge_boost_info.multiplier);

  for (int index = 0; index < 4; ++index) {
    refined_candidate.verification_debug[static_cast<std::size_t>(index)] =
        BuildCoarseOnlyDebugInfo(index, coarse_original, gray_original.size(), config);
    refined_candidate.refined_original[static_cast<std::size_t>(index)] =
        coarse_original[static_cast<std::size_t>(index)];
    method_valid[static_cast<std::size_t>(index)] = true;
    method_quality[static_cast<std::size_t>(index)] = 1.0;
    OuterCornerVerificationDebugInfo& debug =
        refined_candidate.verification_debug[static_cast<std::size_t>(index)];
    const double local_scale =
        debug.local_scale > 0.0 ? debug.local_scale : ComputeCornerLocalScale(coarse_original, index);
    debug.corner_marker_width = ComputeOuterCornerMarkerWidth(local_scale, config);
    debug.close_edge_subpix_boost_applied = close_edge_subpix_boost;
    debug.close_edge_subpix_area_ratio = close_edge_area_ratio;
    debug.close_edge_subpix_max_polar_deg = close_edge_max_polar_deg;
    debug.close_edge_subpix_multiplier = close_edge_subpix_multiplier;
    if (close_edge_subpix_boost) {
      debug.verification_roi_radius = std::max(
          debug.verification_roi_radius,
          static_cast<int>(std::lround(
              static_cast<double>(debug.verification_roi_radius) *
              close_edge_subpix_multiplier)));
      debug.verification_roi =
          MakeCornerVerificationRoi(debug.coarse_corner, gray_original.size(),
                                    debug.verification_roi_radius);
    }
    const OuterSubpixRadiusComputation subpix_computation =
        ComputeAdaptiveOuterSubpixRadiusDebug(local_scale,
                                              debug.verification_roi_radius,
                                              config);
    PopulateOuterSubpixRadiusDebug(subpix_computation, &debug);
    int subpix_radius = subpix_computation.final_radius;
    if (close_edge_subpix_boost) {
      subpix_radius = std::max(
          subpix_radius,
          static_cast<int>(std::lround(
              static_cast<double>(subpix_radius) *
              close_edge_subpix_multiplier)));
      if (config.outer_subpix_window_max > 0) {
        const int min_radius = std::max(kOuterSubpixRadiusMin, config.outer_subpix_window_min);
        const int max_radius = std::max(min_radius, config.outer_subpix_window_max);
        if (subpix_radius > max_radius) {
          subpix_radius = max_radius;
          debug.subpix_window_clamped = true;
          debug.subpix_window_clamp_limit = max_radius;
        }
      }
      debug.boosted_raw_subpix_window_radius = subpix_radius;
    } else {
      debug.boosted_raw_subpix_window_radius = 0;
    }
    debug.subpix_window_radius = std::max(2, subpix_radius);
    debug.spherical_corner = coarse_original[static_cast<std::size_t>(index)];

    const int prev_index = (index + 3) % 4;
    const int next_index = (index + 1) % 4;
    const cv::Point2f corner = coarse_original[static_cast<std::size_t>(index)];
    const cv::Point2f prev_edge = coarse_original[static_cast<std::size_t>(prev_index)] - corner;
    const cv::Point2f next_edge = coarse_original[static_cast<std::size_t>(next_index)] - corner;
    const double prev_length = Norm(prev_edge);
    const double next_length = Norm(next_edge);
    const cv::Point2f quad_center = ComputeQuadCenter(coarse_original);
    debug.prev_marker_support_points =
        CollectCornerMarkerEdgeSupportPoints(gray_original, corner, prev_edge, prev_length,
                                            quad_center, debug.corner_marker_width,
                                            debug.verification_roi_radius);
    debug.next_marker_support_points =
        CollectCornerMarkerEdgeSupportPoints(gray_original, corner, next_edge, next_length,
                                            quad_center, debug.corner_marker_width,
                                            debug.verification_roi_radius);
  }

  for (int index = 0; index < 4; ++index) {
    if (!method_valid[static_cast<std::size_t>(index)]) {
      continue;
    }

    OuterCornerVerificationDebugInfo& debug =
        refined_candidate.verification_debug[static_cast<std::size_t>(index)];
    const int prev_index = (index + 3) % 4;
    const int next_index = (index + 1) % 4;
    if (config.enable_outer_spherical_refinement &&
        sphere_camera != nullptr && sphere_camera->IsValid()) {
      const SphericalCornerRefinement spherical_refinement =
          RefineCornerBySphericalPlanes(gray_original, *sphere_camera, coarse_original, index, config);
      debug.prev_branch_points = spherical_refinement.prev_edge_fit.support_points;
      debug.next_branch_points = spherical_refinement.next_edge_fit.support_points;
      debug.prev_spherical_curve_points = spherical_refinement.prev_curve_points;
      debug.next_spherical_curve_points = spherical_refinement.next_curve_points;
      debug.prev_spherical_residual = spherical_refinement.prev_edge_fit.rms_residual;
      debug.next_spherical_residual = spherical_refinement.next_edge_fit.rms_residual;
      debug.prev_spherical_support_count = spherical_refinement.prev_edge_fit.support_count;
      debug.next_spherical_support_count = spherical_refinement.next_edge_fit.support_count;
      debug.spherical_refinement_valid = spherical_refinement.success;
      debug.spherical_failure_reason = spherical_refinement.failure_reason;
      debug.spherical_corner = spherical_refinement.success
                                   ? spherical_refinement.refined_corner
                                   : coarse_original[static_cast<std::size_t>(index)];
      debug.subpix_corner = debug.spherical_corner;

      const ImageLineCornerRefinement image_line_refinement =
          RefineCornerByImageLineSupportIntersection(
              spherical_refinement.prev_edge_fit.support_points,
              spherical_refinement.next_edge_fit.support_points);
      debug.image_line_valid = image_line_refinement.success;
      debug.image_line_corner = image_line_refinement.success
                                    ? image_line_refinement.refined_corner
                                    : coarse_original[static_cast<std::size_t>(index)];
      debug.prev_image_line_residual = image_line_refinement.prev_line.rms_residual;
      debug.next_image_line_residual = image_line_refinement.next_line.rms_residual;
      debug.prev_image_line_support_count = image_line_refinement.prev_line.support_count;
      debug.next_image_line_support_count = image_line_refinement.next_line.support_count;

      if (spherical_refinement.success) {
        refined_candidate.refined_original[static_cast<std::size_t>(index)] =
            spherical_refinement.refined_corner;
        method_quality[static_cast<std::size_t>(index)] =
            std::min(std::max(method_quality[static_cast<std::size_t>(index)],
                              spherical_refinement.quality),
                     1.0);
        debug.spherical_refinement_applied = true;
      }
      continue;
    }

    const CornerLineRefinement line_refinement =
        RefineCornerByLineIntersection(gray_original, coarse_original, index);
    const cv::Point2f coarse_corner = coarse_original[static_cast<std::size_t>(index)];
    const cv::Point2f delta = line_refinement.refined_corner - coarse_corner;
    const double line_jump = std::hypot(delta.x, delta.y);
    const bool line_inside =
        line_refinement.refined_corner.x >= config.min_border_distance &&
        line_refinement.refined_corner.x <= static_cast<float>(gray_original.cols) - config.min_border_distance &&
        line_refinement.refined_corner.y >= config.min_border_distance &&
        line_refinement.refined_corner.y <= static_cast<float>(gray_original.rows) - config.min_border_distance;
    const cv::Point2f line_seed_delta =
        line_refinement.refined_corner - coarse_original[static_cast<std::size_t>(index)];
    const double line_seed_gap = std::hypot(line_seed_delta.x, line_seed_delta.y);
    debug.line_refinement_success = line_refinement.success;
    debug.line_refinement_quality = line_refinement.quality;
    debug.line_jump = line_jump;
    debug.line_jump_limit = 0.0;
    debug.line_inside = line_inside;
    debug.line_seed_gap = line_seed_gap;

    bool accepted_line_seed =
        line_refinement.success &&
        line_refinement.quality >= kOuterLineMinQuality &&
        line_inside;
    cv::Point2f accepted_line_corner = line_refinement.refined_corner;
    double accepted_line_quality = line_refinement.quality;

    if (accepted_line_seed) {
      refined_candidate.refined_original[static_cast<std::size_t>(index)] =
          accepted_line_corner;
      method_quality[static_cast<std::size_t>(index)] =
          std::min(std::max(method_quality[static_cast<std::size_t>(index)],
                            accepted_line_quality),
                   1.0);
      debug.line_seed_accepted = true;
    }
    debug.subpix_corner = refined_candidate.refined_original[static_cast<std::size_t>(index)];
  }

  if (config.do_outer_subpix_refinement) {
    for (int index = 0; index < 4; ++index) {
      if (!method_valid[static_cast<std::size_t>(index)]) {
        continue;
      }
      const cv::Point2f point_seed =
          refined_candidate.refined_original[static_cast<std::size_t>(index)];
      const OuterCornerVerificationDebugInfo& debug_before_subpix =
          refined_candidate.verification_debug[static_cast<std::size_t>(index)];
      const int subpix_radius =
          debug_before_subpix.subpix_window_radius;
      const CornerSubpixResult subpix_result =
          RefineCornerSubpixWithRollbackCheck(
              gray_original, point_seed, subpix_radius,
              debug_before_subpix.close_edge_subpix_boost_applied);
      const cv::Point2f accepted_subpix = subpix_result.refined_corner;
      refined_candidate.refined_original[static_cast<std::size_t>(index)] = accepted_subpix;
      OuterCornerVerificationDebugInfo& debug_after_subpix =
          refined_candidate.verification_debug[static_cast<std::size_t>(index)];
      debug_after_subpix.subpix_corner = accepted_subpix;
      debug_after_subpix.subpix_applied = true;
      debug_after_subpix.subpix_unstable_rollback_detected =
          subpix_result.unstable_rollback_detected;
      debug_after_subpix.subpix_unstable_rollback_iteration =
          subpix_result.rollback_iteration;
      debug_after_subpix.subpix_unstable_rollback_max_displacement =
          subpix_result.max_probe_displacement;
    }
  } else {
    for (int index = 0; index < 4; ++index) {
      refined_candidate.verification_debug[static_cast<std::size_t>(index)].subpix_corner =
          refined_candidate.refined_original[static_cast<std::size_t>(index)];
    }
  }

  refined_candidate.refine_quality = 1.0;
  for (int index = 0; index < 4; ++index) {
    OuterCornerVerificationDebugInfo& debug =
        refined_candidate.verification_debug[static_cast<std::size_t>(index)];
    debug.coarse_to_verified_displacement =
        Norm(debug.verified_corner - coarse_original[static_cast<std::size_t>(index)]);
    debug.coarse_to_subpix_displacement =
        Norm(debug.subpix_corner - coarse_original[static_cast<std::size_t>(index)]);
    debug.coarse_to_refined_displacement =
        Norm(refined_candidate.refined_original[static_cast<std::size_t>(index)] -
             coarse_original[static_cast<std::size_t>(index)]);

    debug.refine_displacement_limit = 0.0;
    bool refined_valid = method_valid[static_cast<std::size_t>(index)];
    if (refined_valid && debug.subpix_unstable_rollback_detected) {
      refined_valid = false;
      std::ostringstream stream;
      stream << "subpix_unstable_rollback:iteration="
             << debug.subpix_unstable_rollback_iteration
             << ":max_displacement=" << std::fixed << std::setprecision(2)
             << debug.subpix_unstable_rollback_max_displacement
             << ":radius=" << debug.subpix_window_radius;
      debug.failure_reason = stream.str();
    } else if (refined_valid &&
        debug.close_edge_subpix_boost_applied &&
        debug.subpix_applied) {
      std::string support_failure_reason;
      const bool edge_support_ok = PassesRefinedCornerEdgeSupportCheck(
              gray_original, refined_candidate.refined_original, index, debug,
              &support_failure_reason);
      std::string response_diagnostic;
      const bool response_ok = PassesRefinedCornerResponseCheck(
          gray_original,
          refined_candidate.refined_original[static_cast<std::size_t>(index)],
          debug.subpix_window_radius,
          &response_diagnostic);
      // A strong corner response alone is insufficient: a large adaptive
      // window can converge to a different image corner. The refined point
      // must also be supported by the two marker edges of this decoded tag.
      if (!edge_support_ok || !response_ok) {
        refined_valid = false;
        debug.failure_reason = edge_support_ok
                                   ? response_diagnostic
                                   : support_failure_reason + ";" + response_diagnostic;
      }
    }
    debug.refined_valid = refined_valid;
    refined_candidate.refined_valid[static_cast<std::size_t>(index)] = debug.refined_valid;
    const double corner_quality = method_quality[static_cast<std::size_t>(index)];
    if (refined_candidate.refined_valid[static_cast<std::size_t>(index)]) {
      refined_candidate.refine_quality = std::min(refined_candidate.refine_quality, corner_quality);
    } else {
      refined_candidate.refine_quality = 0.0;
    }
  }

  const double hamming_quality =
      1.0 / static_cast<double>(1 + std::max(0, refined_candidate.coarse.detection.hammingDistance));
  const double area_quality = ClampUnit(refined_candidate.coarse.scaled_area / 2500.0);
  refined_candidate.quality = std::min(
      {hamming_quality, refined_candidate.coarse.shape_quality, refined_candidate.refine_quality, area_quality});
  return refined_candidate;
}

OuterSphericalQuadRefinementResult RefineOuterCornersBySphericalPlanesImpl(
    const cv::Mat& gray,
    const DoubleSphereCameraModel& camera,
    const std::array<cv::Point2f, 4>& corner_seeds,
    const MultiScaleOuterTagDetectorConfig& config) {
  OuterSphericalQuadRefinementResult result;
  result.refined_corners = corner_seeds;
  result.min_quality = 1.0;
  if (gray.empty() || !camera.IsValid()) {
    result.min_quality = 0.0;
    for (auto& debug : result.corner_debug) {
      debug.failure_reason = gray.empty() ? "empty_image" : "invalid_camera";
    }
    return result;
  }

  for (int index = 0; index < 4; ++index) {
    const SphericalCornerRefinement refinement =
        RefineCornerBySphericalPlanes(gray, camera, corner_seeds, index, config);
    OuterSphericalCornerRefinementDebug& debug =
        result.corner_debug[static_cast<std::size_t>(index)];
    debug.success = refinement.success;
    debug.refined_corner = refinement.success
                               ? refinement.refined_corner
                               : corner_seeds[static_cast<std::size_t>(index)];
    debug.quality = refinement.quality;
    debug.prev_edge_residual = refinement.prev_edge_fit.rms_residual;
    debug.next_edge_residual = refinement.next_edge_fit.rms_residual;
    debug.prev_edge_support_count = refinement.prev_edge_fit.support_count;
    debug.next_edge_support_count = refinement.next_edge_fit.support_count;
    debug.failure_reason = refinement.failure_reason;
    debug.displacement_px =
        Norm(debug.refined_corner - corner_seeds[static_cast<std::size_t>(index)]);

    if (refinement.success) {
      result.refined_corners[static_cast<std::size_t>(index)] =
          refinement.refined_corner;
      result.max_displacement_px =
          std::max(result.max_displacement_px, debug.displacement_px);
      result.min_quality = std::min(result.min_quality, refinement.quality);
      ++result.successful_corner_count;
    } else {
      result.min_quality = 0.0;
    }
  }
  result.success = result.successful_corner_count == 4;
  // Keep per-corner successful spherical refinements even when not all four
  // corners pass. The normal outer detector applies this refinement per corner;
  // geometry-prior rescue needs the same behavior so partially reliable board
  // edges can still improve the subsequent subpixel and pose checks.
  return result;
}

}  // namespace

std::string ToString(OuterTagFailureReason reason) {
  switch (reason) {
    case OuterTagFailureReason::None:
      return "None";
    case OuterTagFailureReason::NoDetectionsAtAll:
      return "NoDetectionsAtAll";
    case OuterTagFailureReason::DetectionsExistButNoMatchingTagId:
      return "DetectionsExistButNoMatchingTagId";
    case OuterTagFailureReason::MatchingTagIdButRejectedByBorder:
      return "MatchingTagIdButRejectedByBorder";
    case OuterTagFailureReason::MatchingTagIdButRefinementFailed:
      return "MatchingTagIdButRefinementFailed";
    case OuterTagFailureReason::MatchingTagIdButAllScalesUnstable:
      return "MatchingTagIdButAllScalesUnstable";
  }
  return "Unknown";
}

OuterSphericalQuadRefinementResult RefineOuterCornersBySphericalPlanes(
    const cv::Mat& gray,
    const DoubleSphereCameraModel& camera,
    const std::array<cv::Point2f, 4>& corner_seeds,
    const MultiScaleOuterTagDetectorConfig& config) {
  return RefineOuterCornersBySphericalPlanesImpl(gray, camera, corner_seeds, config);
}

MultiScaleOuterTagDetector::MultiScaleOuterTagDetector(MultiScaleOuterTagDetectorConfig config)
    : config_(std::move(config)) {
  requested_board_ids_ = NormalizeBoardIds(config_.tag_ids, config_.tag_id);
  if (requested_board_ids_.empty()) {
    throw std::runtime_error("MultiScaleOuterTagDetector requires at least one non-negative tag id.");
  }
  config_.tag_ids = requested_board_ids_;
  config_.tag_id = requested_board_ids_.front();
  if (config_.min_border_distance < 0.0) {
    throw std::runtime_error("min_border_distance must be non-negative.");
  }
  if (config_.max_scales_to_try < 0) {
    throw std::runtime_error("max_scales_to_try must be non-negative.");
  }
  if (config_.adaptive_coarse_max_hamming < 0) {
    throw std::runtime_error("adaptive_coarse_max_hamming must be non-negative.");
  }
  const auto validate_scale_divisors = [](const std::vector<double>& divisors,
                                          const char* field_name) {
    if (divisors.empty()) {
      throw std::runtime_error(std::string(field_name) + " must not be empty.");
    }
    for (const double divisor : divisors) {
      if (!std::isfinite(divisor) || divisor <= 0.0) {
        throw std::runtime_error(std::string(field_name) +
                                 " must contain positive finite values.");
      }
    }
  };
  validate_scale_divisors(config_.adaptive_coarse_scale_divisors,
                          "adaptive_coarse_scale_divisors");
  validate_scale_divisors(config_.adaptive_fallback_scale_divisors,
                          "adaptive_fallback_scale_divisors");
  if (config_.outer_local_context_scale < 0.0) {
    throw std::runtime_error("outer_local_context_scale must be non-negative.");
  }
  if (config_.outer_corner_marker_ratio < 0.0 || config_.outer_corner_marker_ratio > 1.0) {
    throw std::runtime_error("outer_corner_marker_ratio must be in [0, 1].");
  }
  if (config_.outer_subpix_scale < 0.0) {
    throw std::runtime_error("outer_subpix_scale must be non-negative.");
  }
  if (config_.close_edge_outer_subpix_area_ratio < 0.0) {
    throw std::runtime_error("close_edge_outer_subpix_area_ratio must be non-negative.");
  }
  if (config_.close_edge_outer_subpix_min_polar_deg < 0.0) {
    throw std::runtime_error("close_edge_outer_subpix_min_polar_deg must be non-negative.");
  }
  if (config_.close_edge_outer_subpix_full_polar_deg <=
      config_.close_edge_outer_subpix_min_polar_deg) {
    throw std::runtime_error(
        "close_edge_outer_subpix_full_polar_deg must be larger than "
        "close_edge_outer_subpix_min_polar_deg.");
  }
  if (config_.close_edge_outer_subpix_border_ratio < 0.0) {
    throw std::runtime_error("close_edge_outer_subpix_border_ratio must be non-negative.");
  }
  if (config_.close_edge_outer_subpix_multiplier <= 0.0) {
    throw std::runtime_error("close_edge_outer_subpix_multiplier must be positive.");
  }
  if (config_.close_edge_outer_subpix_max_multiplier <
      config_.close_edge_outer_subpix_multiplier) {
    throw std::runtime_error(
        "close_edge_outer_subpix_max_multiplier must be >= "
        "close_edge_outer_subpix_multiplier.");
  }
  if (config_.outer_refine_gate_scale < 0.0) {
    throw std::runtime_error("outer_refine_gate_scale must be non-negative.");
  }
  if (config_.outer_refine_gate_min <= 0.0) {
    throw std::runtime_error("outer_refine_gate_min must be positive.");
  }
  if (config_.outer_subpix_window_radius < 0) {
    throw std::runtime_error("outer_subpix_window_radius must be non-negative.");
  }
  if (config_.min_detection_quality < 0.0 || config_.min_detection_quality > 1.0) {
    throw std::runtime_error("min_detection_quality must be in [0, 1].");
  }
  if (config_.anonymous_tag_like_rescue_max_center_error_scale < 0.0) {
    throw std::runtime_error(
        "anonymous_tag_like_rescue_max_center_error_scale must be non-negative.");
  }
  if (config_.anonymous_tag_like_rescue_min_area_ratio < 0.0 ||
      config_.anonymous_tag_like_rescue_max_area_ratio <
          config_.anonymous_tag_like_rescue_min_area_ratio) {
    throw std::runtime_error(
        "anonymous_tag_like_rescue area ratio bounds must be non-negative and ordered.");
  }
  if (config_.camera_aware_sphere_patch_max_hamming < 0) {
    throw std::runtime_error(
        "camera_aware_sphere_patch_max_hamming must be non-negative.");
  }

  detector_ = std::make_unique<AprilTags::TagDetector>(AprilTags::tagCodes36h11, 2);
  if (config_.refine_camera.IsConfigured()) {
    sphere_camera_ = std::make_unique<DoubleSphereCameraModel>(
        DoubleSphereCameraModel::FromConfig(ToIntermediateCameraConfig(config_.refine_camera)));
  }
}

MultiScaleOuterTagDetector::~MultiScaleOuterTagDetector() = default;

MultiScaleOuterTagDetectorConfig MultiScaleOuterTagDetector::LoadConfig(const std::string& yaml_path) {
  return ParseConfig(yaml_path);
}

namespace {

struct PerTagOuterAggregationState {
  OuterTagDetectionResult result;
  bool saw_any_detection = false;
  bool saw_matching_tag_id = false;
  bool saw_border_rejection = false;
  bool saw_non_border_matching_rejection = false;
  bool attempted_local_patch_rescue = false;
  std::vector<std::string> local_patch_rescue_summaries;
  std::vector<ScaleCandidate> coarse_candidates;
  std::vector<OuterWrongIdProposal> wrong_id_proposals;
};

void StoreWrongIdProposal(const ScaleCandidate& candidate,
                          const std::string& source,
                          std::vector<OuterWrongIdProposal>* proposals) {
  if (proposals == nullptr || !candidate.detection.good ||
      candidate.detection.id < 0 || candidate.min_edge < kMinQuadEdgePixels ||
      candidate.scaled_area < kMinQuadAreaPixels ||
      candidate.shape_quality <= 0.10) {
    return;
  }
  OuterWrongIdProposal proposal;
  proposal.detected_tag_id = candidate.detection.id;
  proposal.hamming = candidate.detection.hammingDistance;
  proposal.area_px = candidate.scaled_area;
  proposal.source = source;
  for (int index = 0; index < 4; ++index) {
    proposal.corners_original_image[static_cast<std::size_t>(index)] =
        ToEigen(candidate.original_corners[static_cast<std::size_t>(index)]);
  }
  const auto existing = std::find_if(
      proposals->begin(), proposals->end(),
      [&proposal](const OuterWrongIdProposal& candidate_proposal) {
        return candidate_proposal.detected_tag_id == proposal.detected_tag_id &&
               candidate_proposal.source == proposal.source;
      });
  if (existing == proposals->end()) {
    proposals->push_back(std::move(proposal));
  } else if (proposal.hamming < existing->hamming ||
             (proposal.hamming == existing->hamming &&
              proposal.area_px > existing->area_px)) {
    *existing = std::move(proposal);
  } else {
    return;
  }
  constexpr std::size_t kMaxWrongIdProposalsPerBoard = 12;
  if (proposals->size() > kMaxWrongIdProposalsPerBoard) {
    std::stable_sort(
        proposals->begin(), proposals->end(),
        [](const OuterWrongIdProposal& lhs, const OuterWrongIdProposal& rhs) {
          if (lhs.hamming != rhs.hamming) {
            return lhs.hamming < rhs.hamming;
          }
          return lhs.area_px > rhs.area_px;
        });
    proposals->resize(kMaxWrongIdProposalsPerBoard);
  }
}

ScaleCandidate BuildScaleCandidateFromDetection(const AprilTags::TagDetection& detection,
                                                const OuterTagScaleDebugInfo& debug,
                                                const cv::Size& original_size) {
  ScaleCandidate candidate;
  candidate.target_longest_side = debug.target_longest_side;
  candidate.scale_factor = debug.scale_factor;
  candidate.configured_scale_divisor = debug.configured_scale_divisor;
  candidate.scaled_size = debug.scaled_size;
  candidate.detection = detection;
  for (int index = 0; index < 4; ++index) {
    candidate.scaled_corners[static_cast<std::size_t>(index)] =
        cv::Point2f(detection.p[index].first, detection.p[index].second);
  }

  const double scale_x =
      static_cast<double>(original_size.width) /
      static_cast<double>(std::max(1, candidate.scaled_size.width));
  const double scale_y =
      static_cast<double>(original_size.height) /
      static_cast<double>(std::max(1, candidate.scaled_size.height));
  for (int index = 0; index < 4; ++index) {
    candidate.original_corners[static_cast<std::size_t>(index)] = cv::Point2f(
        static_cast<float>(candidate.scaled_corners[static_cast<std::size_t>(index)].x * scale_x),
        static_cast<float>(candidate.scaled_corners[static_cast<std::size_t>(index)].y * scale_y));
  }

  candidate.scaled_area = ComputeQuadArea(candidate.scaled_corners);
  const std::pair<double, double> edge_range = ComputeEdgeRange(candidate.scaled_corners);
  candidate.min_edge = edge_range.first;
  candidate.max_edge = edge_range.second;
  candidate.shape_quality =
      candidate.max_edge > 1e-6 ? ClampUnit(candidate.min_edge / candidate.max_edge) : 0.0;
  return candidate;
}

const ScaleCandidate* BestCoarseCandidate(const PerTagOuterAggregationState& state) {
  const ScaleCandidate* best = nullptr;
  for (const ScaleCandidate& candidate : state.coarse_candidates) {
    if (best == nullptr || IsCandidateBetter(candidate, *best)) {
      best = &candidate;
    }
  }
  return best;
}

struct ResolvedBoardGeometry {
  int board_id = -1;
  cv::Point2f center{};
  double area = 0.0;
  double mean_edge = 0.0;
  double max_edge = 0.0;
};

bool BuildResolvedBoardGeometry(const PerTagOuterAggregationState& state,
                                ResolvedBoardGeometry* geometry) {
  if (geometry == nullptr) {
    return false;
  }
  const ScaleCandidate* candidate = BestCoarseCandidate(state);
  if (candidate == nullptr) {
    return false;
  }
  geometry->board_id = state.result.board_id;
  geometry->center = ComputeQuadCenter(candidate->original_corners);
  geometry->area = ComputeQuadArea(candidate->original_corners);
  const std::pair<double, double> edge_range = ComputeEdgeRange(candidate->original_corners);
  geometry->mean_edge = 0.25 * (
      Norm(candidate->original_corners[1] - candidate->original_corners[0]) +
      Norm(candidate->original_corners[2] - candidate->original_corners[1]) +
      Norm(candidate->original_corners[3] - candidate->original_corners[2]) +
      Norm(candidate->original_corners[0] - candidate->original_corners[3]));
  geometry->max_edge = edge_range.second;
  return geometry->area >= kMinQuadAreaPixels &&
         geometry->mean_edge >= kMinQuadEdgePixels;
}

ScaleCandidate MakeSyntheticAnonymousCandidateForTarget(
    const ScaleCandidate& anonymous_candidate,
    int target_board_id,
    const std::string& rescue_label) {
  ScaleCandidate synthetic = anonymous_candidate;
  synthetic.detection = AprilTags::TagDetection(target_board_id);
  synthetic.detection.good = true;
  synthetic.detection.id = target_board_id;
  // Keep anonymous-ID rescues lower ranked than true decoded IDs while still
  // allowing the normal refinement and quality gates to verify the quad.
  synthetic.detection.hammingDistance = 2;
  synthetic.detection.rotation = anonymous_candidate.detection.rotation;
  synthetic.detection.observedPerimeter = anonymous_candidate.detection.observedPerimeter;
  synthetic.detection.cxy = anonymous_candidate.detection.cxy;
  for (int index = 0; index < 4; ++index) {
    synthetic.detection.p[index] = anonymous_candidate.detection.p[index];
  }
  synthetic.from_local_patch_rescue = true;
  synthetic.local_patch_label = rescue_label;
  return synthetic;
}

void TryAnonymousTagLikeGeometryRescue(
    const cv::Size& image_size,
    const MultiScaleOuterTagDetectorConfig& config,
    const std::vector<ScaleCandidate>& anonymous_candidates,
    std::vector<PerTagOuterAggregationState>* states) {
  if (!config.enable_anonymous_tag_like_geometry_rescue || states == nullptr ||
      anonymous_candidates.empty() || states->size() < 3) {
    return;
  }

  std::vector<ResolvedBoardGeometry> resolved_boards;
  resolved_boards.reserve(states->size());
  for (const PerTagOuterAggregationState& state : *states) {
    if (state.coarse_candidates.empty()) {
      continue;
    }
    ResolvedBoardGeometry geometry;
    if (BuildResolvedBoardGeometry(state, &geometry)) {
      resolved_boards.push_back(geometry);
    }
  }
  if (resolved_boards.size() < 2) {
    return;
  }
  std::sort(resolved_boards.begin(), resolved_boards.end(),
            [](const ResolvedBoardGeometry& lhs, const ResolvedBoardGeometry& rhs) {
              return lhs.board_id < rhs.board_id;
            });

  std::vector<bool> anonymous_used(anonymous_candidates.size(), false);
  for (PerTagOuterAggregationState& state : *states) {
    if (!state.saw_any_detection || state.saw_matching_tag_id ||
        !state.coarse_candidates.empty()) {
      continue;
    }

    const int target_board_id = state.result.board_id;
    const ResolvedBoardGeometry* lower = nullptr;
    const ResolvedBoardGeometry* upper = nullptr;
    for (const ResolvedBoardGeometry& resolved : resolved_boards) {
      if (resolved.board_id < target_board_id) {
        if (lower == nullptr || resolved.board_id > lower->board_id) {
          lower = &resolved;
        }
      } else if (resolved.board_id > target_board_id) {
        if (upper == nullptr || resolved.board_id < upper->board_id) {
          upper = &resolved;
        }
      }
    }
    if (lower == nullptr || upper == nullptr || lower->board_id == upper->board_id) {
      continue;
    }

    const double id_alpha =
        static_cast<double>(target_board_id - lower->board_id) /
        static_cast<double>(upper->board_id - lower->board_id);
    if (id_alpha <= 0.0 || id_alpha >= 1.0) {
      continue;
    }
    const cv::Point2f expected_center =
        lower->center + (upper->center - lower->center) * static_cast<float>(id_alpha);
    const double expected_area =
        std::max(1.0, (1.0 - id_alpha) * lower->area + id_alpha * upper->area);
    const double expected_edge =
        std::max(kMinQuadEdgePixels,
                 (1.0 - id_alpha) * lower->mean_edge + id_alpha * upper->mean_edge);
    const double neighbor_span = Norm(upper->center - lower->center);
    const double max_center_error = std::max(
        config.anonymous_tag_like_rescue_max_center_error_scale * expected_edge,
        0.30 * neighbor_span);

    int best_index = -1;
    double best_score = std::numeric_limits<double>::infinity();
    double best_center_error = 0.0;
    double best_area_ratio = 0.0;
    for (std::size_t candidate_index = 0; candidate_index < anonymous_candidates.size();
         ++candidate_index) {
      if (anonymous_used[candidate_index]) {
        continue;
      }
      const ScaleCandidate& candidate = anonymous_candidates[candidate_index];
      if (!candidate.detection.good ||
          candidate.scaled_area < kMinQuadAreaPixels ||
          candidate.min_edge < kMinQuadEdgePixels ||
          candidate.shape_quality <= 0.10 ||
          !PassesBorderCheck(candidate.original_corners, image_size,
                             config.min_border_distance)) {
        continue;
      }

      const cv::Point2f candidate_center = ComputeQuadCenter(candidate.original_corners);
      bool overlaps_resolved_board = false;
      for (const ResolvedBoardGeometry& resolved : resolved_boards) {
        const double overlap_distance = Norm(candidate_center - resolved.center);
        const double overlap_limit = 0.35 * (candidate.max_edge + resolved.max_edge);
        if (overlap_distance < overlap_limit) {
          overlaps_resolved_board = true;
          break;
        }
      }
      if (overlaps_resolved_board) {
        continue;
      }

      const double candidate_area = std::max(1.0, ComputeQuadArea(candidate.original_corners));
      const double area_ratio = candidate_area / expected_area;
      if (area_ratio < config.anonymous_tag_like_rescue_min_area_ratio ||
          area_ratio > config.anonymous_tag_like_rescue_max_area_ratio) {
        continue;
      }
      const double center_error = Norm(candidate_center - expected_center);
      if (center_error > max_center_error) {
        continue;
      }
      const double score = center_error / std::max(1.0, max_center_error) +
                           0.25 * std::abs(std::log(area_ratio)) +
                           0.20 * (1.0 - candidate.shape_quality);
      if (score < best_score) {
        best_score = score;
        best_index = static_cast<int>(candidate_index);
        best_center_error = center_error;
        best_area_ratio = area_ratio;
      }
    }

    if (best_index < 0) {
      continue;
    }

    anonymous_used[static_cast<std::size_t>(best_index)] = true;
    std::ostringstream rescue_summary;
    rescue_summary << "anonymous_tag_like_geometry_rescue target_id=" << target_board_id
                   << " decoded_id=" << anonymous_candidates[static_cast<std::size_t>(best_index)].detection.id
                   << " bracket=" << lower->board_id << "," << upper->board_id
                   << " center_error_px=" << std::fixed << std::setprecision(2)
                   << best_center_error
                   << " max_center_error_px=" << max_center_error
                   << " area_ratio=" << best_area_ratio
                   << " score=" << best_score;
    state.saw_matching_tag_id = true;
    state.attempted_local_patch_rescue = true;
    state.coarse_candidates.push_back(MakeSyntheticAnonymousCandidateForTarget(
        anonymous_candidates[static_cast<std::size_t>(best_index)], target_board_id,
        rescue_summary.str()));
    state.result.successful_scale_longest_sides.push_back(
        anonymous_candidates[static_cast<std::size_t>(best_index)].target_longest_side);
    state.local_patch_rescue_summaries.push_back(rescue_summary.str());
  }
}

ScaleCandidate BuildSyntheticCandidateFromOriginalCorners(
    int target_board_id,
    const cv::Size& image_size,
    const std::array<cv::Point2f, 4>& original_corners,
    int hamming_distance,
    const std::string& rescue_label) {
  ScaleCandidate candidate;
  candidate.target_longest_side = std::max(image_size.width, image_size.height);
  candidate.scale_factor = 1.0;
  candidate.configured_scale_divisor = 0.0;
  candidate.scaled_size = image_size;
  candidate.detection = AprilTags::TagDetection(target_board_id);
  candidate.detection.good = true;
  candidate.detection.id = target_board_id;
  candidate.detection.hammingDistance = hamming_distance;
  candidate.scaled_corners = original_corners;
  candidate.original_corners = original_corners;
  const cv::Point2f center = ComputeQuadCenter(original_corners);
  candidate.detection.cxy = {center.x, center.y};
  candidate.detection.observedPerimeter = 0.0f;
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f& corner = original_corners[static_cast<std::size_t>(index)];
    candidate.detection.p[index] = {corner.x, corner.y};
    const cv::Point2f& next = original_corners[static_cast<std::size_t>((index + 1) % 4)];
    candidate.detection.observedPerimeter += static_cast<float>(Norm(next - corner));
  }
  candidate.scaled_area = ComputeQuadArea(candidate.scaled_corners);
  const std::pair<double, double> edge_range = ComputeEdgeRange(candidate.scaled_corners);
  candidate.min_edge = edge_range.first;
  candidate.max_edge = edge_range.second;
  candidate.shape_quality =
      candidate.max_edge > 1e-6 ? ClampUnit(candidate.min_edge / candidate.max_edge) : 0.0;
  candidate.from_local_patch_rescue = true;
  candidate.local_patch_label = rescue_label;
  return candidate;
}

void TryLocalSpherePatchRescue(
    const cv::Mat& gray_original,
    const MultiScaleOuterTagDetectorConfig& config,
    const DoubleSphereCameraModel* sphere_camera,
    const std::map<int, std::size_t>& requested_index_by_id,
    AprilTags::TagDetector* detector,
    std::vector<PerTagOuterAggregationState>* states) {
  if (!config.enable_camera_aware_sphere_patch_rescue || states == nullptr ||
      detector == nullptr || sphere_camera == nullptr ||
      !sphere_camera->IsValid()) {
    return;
  }

  const cv::Size image_size = gray_original.size();
  if (sphere_camera->resolution() != image_size) {
    return;
  }

  std::vector<std::size_t> unresolved_indices;
  unresolved_indices.reserve(states->size());
  for (std::size_t index = 0; index < states->size(); ++index) {
    PerTagOuterAggregationState& state = (*states)[index];
    const bool can_attempt_zero_detection_recovery =
        config.camera_aware_sphere_patch_rescue_zero_detection_frames &&
        !state.saw_any_detection;
    if (!state.saw_matching_tag_id &&
        (state.saw_any_detection || can_attempt_zero_detection_recovery)) {
      state.attempted_local_patch_rescue = true;
      unresolved_indices.push_back(index);
    }
  }
  if (unresolved_indices.empty()) {
    return;
  }

  const bool frame_has_no_direct_decode =
      std::all_of(states->begin(), states->end(),
                  [](const PerTagOuterAggregationState& state) {
                    return !state.saw_any_detection;
                  });
  const bool use_zero_detection_fine_atlas =
      config.camera_aware_sphere_patch_rescue_zero_detection_frames &&
      config.camera_aware_sphere_patch_use_extended_atlas &&
      frame_has_no_direct_decode;
  const std::vector<LocalSpherePatchPlan> patch_plans =
      BuildOuterLocalSpherePatchPlans(
          config.camera_aware_sphere_patch_use_extended_atlas,
          use_zero_detection_fine_atlas);
  std::map<std::size_t, ScaleCandidate> best_candidate_by_index;
  for (const LocalSpherePatchPlan& patch_plan : patch_plans) {
    LocalSpherePatchContext patch_context;
    if (!BuildLocalSpherePatch(gray_original, *sphere_camera, patch_plan, &patch_context)) {
      continue;
    }

    // Large Tags can exceed the AprilTag detector's most stable scale on a
    // local patch.  The original image cascade already searches scale; apply
    // the same idea to all-zero local-patch recovery and map detections back to
    // the native patch before converting them to distorted-image corners.
    const std::array<double, 3> patch_detection_scales =
        use_zero_detection_fine_atlas
            ? std::array<double, 3>{{0.50, 0.75, 0.0}}
            : std::array<double, 3>{{1.00, 1.00, 1.00}};
    for (const double patch_detection_scale : patch_detection_scales) {
      if (patch_detection_scale <= 0.0) {
        continue;
      }
      if (patch_detection_scale != 1.0 &&
          patch_detection_scale != 0.50 && patch_detection_scale != 0.75) {
        continue;
      }
      cv::Mat detection_patch;
      if (std::abs(patch_detection_scale - 1.0) <= 1e-9) {
        detection_patch = patch_context.patch;
      } else {
        cv::resize(patch_context.patch, detection_patch, cv::Size(),
                   patch_detection_scale, patch_detection_scale,
                   cv::INTER_AREA);
      }
      std::vector<cv::Mat> detection_variants;
      detection_variants.push_back(detection_patch);
      if (use_zero_detection_fine_atlas) {
        // Extreme-polar frames in this capture have a compressed grey-level
        // range even after sphere rectification.  Enhance only the recovery
        // copy; the committed corner still comes from the same ray mapping.
        cv::Mat equalized_patch;
        cv::equalizeHist(detection_patch, equalized_patch);
        detection_variants.push_back(equalized_patch);
      }
      for (std::size_t variant_index = 0;
           variant_index < detection_variants.size(); ++variant_index) {
        const std::vector<AprilTags::TagDetection> detections =
            detector->extractTags(detection_variants[variant_index]);
        for (const AprilTags::TagDetection& raw_detection : detections) {
        AprilTags::TagDetection detection = raw_detection;
        if (std::abs(patch_detection_scale - 1.0) > 1e-9) {
          for (int corner_index = 0; corner_index < 4; ++corner_index) {
            detection.p[corner_index].first /= patch_detection_scale;
            detection.p[corner_index].second /= patch_detection_scale;
          }
          detection.cxy.first /= patch_detection_scale;
          detection.cxy.second /= patch_detection_scale;
        }
      const auto requested_it = requested_index_by_id.find(detection.id);
      if (!detection.good) {
        continue;
      }

      ScaleCandidate candidate;
      if (!BuildScaleCandidateFromPatchDetection(detection, patch_context, *sphere_camera,
                                                 &candidate)) {
        continue;
      }
      if (std::abs(patch_detection_scale - 1.0) > 1e-9) {
        candidate.local_patch_label +=
            "_scale" + std::to_string(static_cast<int>(
                           std::lround(100.0 * patch_detection_scale)));
      }
      if (variant_index > 0) {
        candidate.local_patch_label += "_equalized";
      }
      if (candidate.scaled_area < kMinQuadAreaPixels || candidate.min_edge < kMinQuadEdgePixels ||
          candidate.shape_quality <= 0.10) {
        continue;
      }

      for (std::size_t unresolved_index : unresolved_indices) {
        PerTagOuterAggregationState& unresolved_state = (*states)[unresolved_index];
        if (candidate.detection.id != unresolved_state.result.board_id) {
          StoreWrongIdProposal(candidate, "camera_aware_sphere_patch",
                               &unresolved_state.wrong_id_proposals);
        }
      }

      if (requested_it == requested_index_by_id.end()) {
        continue;
      }

      PerTagOuterAggregationState& state = (*states)[requested_it->second];
      if (!state.attempted_local_patch_rescue ||
          detection.hammingDistance >
              config.camera_aware_sphere_patch_max_hamming) {
        continue;
      }

      const auto current_it = best_candidate_by_index.find(requested_it->second);
      const bool better = current_it == best_candidate_by_index.end() ||
                          candidate.detection.hammingDistance <
                              current_it->second.detection.hammingDistance ||
                          (candidate.detection.hammingDistance ==
                               current_it->second.detection.hammingDistance &&
                           (candidate.local_patch_center_distance <
                                current_it->second.local_patch_center_distance - 1e-6 ||
                            (std::abs(candidate.local_patch_center_distance -
                                      current_it->second.local_patch_center_distance) <= 1e-6 &&
                             candidate.scaled_area >
                                 current_it->second.scaled_area)));
      if (better) {
        best_candidate_by_index[requested_it->second] = candidate;
      }
    }
      }
    }
  }

  for (const auto& entry : best_candidate_by_index) {
    PerTagOuterAggregationState& state = (*states)[entry.first];
    const ScaleCandidate& candidate = entry.second;
    state.saw_matching_tag_id = true;
    state.coarse_candidates.push_back(candidate);
    std::ostringstream rescue_summary;
    rescue_summary << "camera_aware_sphere_patch label="
                   << candidate.local_patch_label
                   << " id=" << candidate.detection.id
                   << " ham=" << candidate.detection.hammingDistance
                   << " center_distance=" << std::fixed << std::setprecision(2)
                   << candidate.local_patch_center_distance
                   << " area=" << std::lround(candidate.scaled_area);
    state.local_patch_rescue_summaries.push_back(rescue_summary.str());
  }
}

OuterTagDetectionResult FinalizeOuterTagDetection(
    const cv::Mat& gray_original,
    const MultiScaleOuterTagDetectorConfig& config,
    const DoubleSphereCameraModel* sphere_camera,
    PerTagOuterAggregationState* state) {
  if (state == nullptr) {
    throw std::runtime_error("FinalizeOuterTagDetection requires a valid state pointer.");
  }

  OuterTagDetectionResult& result = state->result;
  result.attempted_local_patch_rescue = state->attempted_local_patch_rescue;
  result.wrong_id_proposals = state->wrong_id_proposals;
  if (state->attempted_local_patch_rescue) {
    result.local_patch_rescue_summary = state->local_patch_rescue_summaries.empty()
                                            ? "attempted local sphere-patch rescue, no matching id found"
                                            : JoinReasons(state->local_patch_rescue_summaries);
  }
  if (state->coarse_candidates.empty()) {
    if (!state->saw_any_detection) {
      result.failure_reason = OuterTagFailureReason::NoDetectionsAtAll;
    } else if (!state->saw_matching_tag_id) {
      result.failure_reason = OuterTagFailureReason::DetectionsExistButNoMatchingTagId;
    } else if (state->saw_border_rejection && !state->saw_non_border_matching_rejection) {
      result.failure_reason = OuterTagFailureReason::MatchingTagIdButRejectedByBorder;
    } else {
      result.failure_reason = OuterTagFailureReason::MatchingTagIdButAllScalesUnstable;
    }
    result.failure_reason_text = ToString(result.failure_reason);
    return result;
  }

  std::sort(state->coarse_candidates.begin(), state->coarse_candidates.end(), IsCandidateBetter);
  const auto reference_it = std::find_if(
      state->coarse_candidates.begin(), state->coarse_candidates.end(),
      [&](const ScaleCandidate& candidate) {
        return PassesBorderCheck(candidate.original_corners, gray_original.size(),
                                 config.min_border_distance);
      });
  if (reference_it == state->coarse_candidates.end()) {
    result.failure_reason = OuterTagFailureReason::MatchingTagIdButAllScalesUnstable;
    result.failure_reason_text = ToString(result.failure_reason);
    return result;
  }

  if (reference_it->from_local_patch_rescue &&
      config.camera_aware_sphere_patch_commit_mapped_corners) {
    result.used_local_patch_rescue = true;
    result.detected_tag_id = reference_it->detection.id;
    result.chosen_scale_longest_side = reference_it->target_longest_side;
    result.chosen_scale_factor = reference_it->scale_factor;
    result.hamming = reference_it->detection.hammingDistance;
    result.good = reference_it->detection.good;
    result.quality = 1.0;
    for (int index = 0; index < 4; ++index) {
      const std::size_t corner_index = static_cast<std::size_t>(index);
      result.coarse_corners_scaled_image[corner_index] =
          ToEigen(reference_it->scaled_corners[corner_index]);
      result.coarse_corners_original_image[corner_index] =
          ToEigen(reference_it->original_corners[corner_index]);
      result.refined_corners_original_image[corner_index] =
          ToEigen(reference_it->original_corners[corner_index]);
      result.refined_valid[corner_index] = true;
      result.corner_verification_debug[corner_index].coarse_corner =
          reference_it->original_corners[corner_index];
      result.corner_verification_debug[corner_index].verified_corner =
          reference_it->original_corners[corner_index];
      result.corner_verification_debug[corner_index].subpix_corner =
          reference_it->original_corners[corner_index];
      result.corner_verification_debug[corner_index].refined_valid = true;
      result.corner_verification_debug[corner_index].verification_passed = true;
    }
    result.success = true;
    result.failure_reason = OuterTagFailureReason::None;
    result.failure_reason_text = ToString(result.failure_reason);
    return result;
  }

  MultiScaleCornerFusionOutcome fusion =
      FuseMultiScaleCoarseCorners(state->coarse_candidates, gray_original.size(), config);
  result.corner_fusion_debug = fusion.debug;

  ScaleCandidate working_candidate = *reference_it;
  std::array<cv::Point2f, 4> working_coarse_original = reference_it->original_corners;
  if (fusion.valid) {
    working_coarse_original = fusion.fused_corners;
    working_candidate.scaled_corners = ProjectOriginalCornersToScaledImage(
        working_coarse_original, gray_original.size(), working_candidate.scaled_size);
    working_candidate.scaled_area = ComputeQuadArea(working_candidate.scaled_corners);
    const std::pair<double, double> fused_edge_range =
        ComputeEdgeRange(working_candidate.scaled_corners);
    working_candidate.min_edge = fused_edge_range.first;
    working_candidate.max_edge = fused_edge_range.second;
    working_candidate.shape_quality =
        fused_edge_range.second > 1e-6
            ? ClampUnit(fused_edge_range.first / fused_edge_range.second)
            : 0.0;
    result.used_corner_fusion = state->coarse_candidates.size() > 1;
    if (result.used_corner_fusion) {
      for (OuterTagScaleDebugInfo& debug : result.scale_debug) {
        if (std::find(result.successful_scale_longest_sides.begin(),
                      result.successful_scale_longest_sides.end(),
                      debug.target_longest_side) != result.successful_scale_longest_sides.end()) {
          debug.contributed_to_corner_fusion = true;
        }
      }
    }
  }

  const RefinedCandidate refined_candidate =
      RefineCoarseCandidate(gray_original, working_candidate, working_coarse_original, config,
                            sphere_camera);
  const bool refined_inside = PassesBorderCheck(refined_candidate.refined_original,
                                                gray_original.size(), config.min_border_distance);
  const bool all_refined_valid =
      std::all_of(refined_candidate.refined_valid.begin(), refined_candidate.refined_valid.end(),
                  [](bool valid) { return valid; });

  auto fill_result_from_candidate = [&](const RefinedCandidate& chosen_candidate) {
    result.used_local_patch_rescue = chosen_candidate.coarse.from_local_patch_rescue;
    if (result.used_local_patch_rescue && result.local_patch_rescue_summary.empty()) {
      result.local_patch_rescue_summary = chosen_candidate.coarse.local_patch_label;
    }
    result.detected_tag_id = chosen_candidate.coarse.detection.id;
    result.chosen_scale_longest_side = chosen_candidate.coarse.target_longest_side;
    result.chosen_scale_factor = chosen_candidate.coarse.scale_factor;
    result.hamming = chosen_candidate.coarse.detection.hammingDistance;
    result.good = chosen_candidate.coarse.detection.good;
    result.quality = chosen_candidate.quality;
    for (int index = 0; index < 4; ++index) {
      result.coarse_corners_scaled_image[static_cast<std::size_t>(index)] =
          ToEigen(chosen_candidate.coarse.scaled_corners[static_cast<std::size_t>(index)]);
      result.coarse_corners_original_image[static_cast<std::size_t>(index)] =
          ToEigen(chosen_candidate.coarse_original[static_cast<std::size_t>(index)]);
      result.refined_corners_original_image[static_cast<std::size_t>(index)] =
          ToEigen(chosen_candidate.refined_original[static_cast<std::size_t>(index)]);
      result.refined_valid[static_cast<std::size_t>(index)] =
          chosen_candidate.refined_valid[static_cast<std::size_t>(index)];
      result.corner_verification_debug[static_cast<std::size_t>(index)] =
          chosen_candidate.verification_debug[static_cast<std::size_t>(index)];
    }
  };
  fill_result_from_candidate(refined_candidate);

  std::vector<OuterTagScaleDebugInfo>::iterator debug_it =
      std::find_if(result.scale_debug.begin(), result.scale_debug.end(),
                   [&](const OuterTagScaleDebugInfo& info) {
                     return info.target_longest_side ==
                            refined_candidate.coarse.target_longest_side;
                   });

  if (refined_inside && all_refined_valid &&
      refined_candidate.quality >= config.min_detection_quality) {
    result.success = true;
    result.failure_reason = OuterTagFailureReason::None;
    result.failure_reason_text = ToString(result.failure_reason);
    if (debug_it != result.scale_debug.end()) {
      ++debug_it->refined_success_count;
    }
    return result;
  }

  if (!refined_inside || !all_refined_valid) {
    result.failure_reason = OuterTagFailureReason::MatchingTagIdButRefinementFailed;
  } else if (state->saw_border_rejection) {
    result.failure_reason = OuterTagFailureReason::MatchingTagIdButAllScalesUnstable;
  } else {
    result.failure_reason = OuterTagFailureReason::MatchingTagIdButAllScalesUnstable;
  }
  result.failure_reason_text = ToString(result.failure_reason);
  return result;
}

OuterBoardMeasurement BuildOuterBoardMeasurement(const OuterTagDetectionResult& detection) {
  OuterBoardMeasurement measurement;
  measurement.board_id = detection.board_id;
  measurement.detected_tag_id = detection.detected_tag_id;
  measurement.success = detection.success;
  measurement.attempted_local_patch_rescue = detection.attempted_local_patch_rescue;
  measurement.used_local_patch_rescue = detection.used_local_patch_rescue;
  measurement.local_patch_rescue_summary = detection.local_patch_rescue_summary;
  measurement.detection_quality = detection.quality;
  measurement.refined_outer_corners_original_image = detection.refined_corners_original_image;
  measurement.refined_corner_valid = detection.refined_valid;
  measurement.corner_verification_debug = detection.corner_verification_debug;
  measurement.valid_refined_corner_count = CountValidCorners(detection.refined_valid);
  measurement.failure_reason = detection.failure_reason;
  measurement.failure_reason_text = detection.failure_reason_text;
  return measurement;
}

OuterFrameMeasurementResult BuildOuterFrameMeasurementResult(
    const cv::Size& image_size,
    const std::vector<int>& requested_board_ids,
    const std::vector<OuterTagDetectionResult>& detections) {
  OuterFrameMeasurementResult result;
  result.image_size = image_size;
  result.requested_board_ids = requested_board_ids;
  result.board_measurements.reserve(detections.size());
  for (const OuterTagDetectionResult& detection : detections) {
    result.board_measurements.push_back(BuildOuterBoardMeasurement(detection));
  }
  return result;
}

}  // namespace

OuterTagMultiDetectionResult MultiScaleOuterTagDetector::DetectMultiple(
    const cv::Mat& image) const {
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  OuterTagMultiDetectionResult result;
  result.image_size = image.size();
  if (config_.auto_discover_tag_ids) {
    // The auto path discovers IDs while it evaluates each scale. This avoids
    // running the AprilTag decoder twice for every image.
    result.detections = DetectMultiple(image, std::vector<int>());
    for (const OuterTagDetectionResult& detection : result.detections) {
      result.requested_board_ids.push_back(detection.board_id);
    }
    std::sort(result.requested_board_ids.begin(), result.requested_board_ids.end());
  } else {
    result.requested_board_ids = requested_board_ids_;
    result.detections = DetectMultiple(image, result.requested_board_ids);
  }
  result.frame_measurements =
      BuildOuterFrameMeasurementResult(result.image_size,
                                       result.requested_board_ids,
                                       result.detections);
  return result;
}

std::vector<int> MultiScaleOuterTagDetector::DiscoverTagIds(
    const cv::Mat& image) const {
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  const cv::Mat gray_original = ToGray(image);
  std::string unused_scale_mode;
  const std::vector<ScalePlanEntry> scale_plan =
      BuildScalePlan(gray_original.size(), config_, &unused_scale_mode);
  std::set<int> discovered_ids;
  for (const ScalePlanEntry& plan_entry : scale_plan) {
    const cv::Size scaled_size =
        MakeScaledSize(gray_original.size(), plan_entry.target_longest_side);
    cv::Mat scaled_gray;
    if (scaled_size == gray_original.size()) {
      scaled_gray = gray_original;
    } else {
      cv::resize(gray_original, scaled_gray, scaled_size, 0.0, 0.0,
                 cv::INTER_AREA);
    }
    scaled_gray = MaybeBlur(scaled_gray, config_);
    const std::vector<AprilTags::TagDetection> detections =
        detector_->extractTags(scaled_gray);
    for (const AprilTags::TagDetection& detection : detections) {
      if (detection.good && detection.id >= 0) {
        discovered_ids.insert(detection.id);
      }
    }
  }
  return std::vector<int>(discovered_ids.begin(), discovered_ids.end());
}

std::vector<OuterTagDetectionResult> MultiScaleOuterTagDetector::DetectMultiple(
    const cv::Mat& image, const std::vector<int>& requested_tag_ids) const {
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  const bool discover_automatically =
      config_.auto_discover_tag_ids && requested_tag_ids.empty();
  const std::vector<int> normalized_tag_ids = discover_automatically
                                                  ? std::vector<int>()
                                                  : NormalizeBoardIds(
                                                        requested_tag_ids,
                                                        config_.tag_id);
  if (!discover_automatically && normalized_tag_ids.empty()) {
    throw std::runtime_error("DetectMultiple requires at least one valid requested tag id.");
  }

  const cv::Mat gray_original = ToGray(image);
  std::string scale_mode_used;
  const std::vector<ScalePlanEntry> scale_plan =
      BuildScalePlan(gray_original.size(), config_, &scale_mode_used);

  std::vector<PerTagOuterAggregationState> states;
  states.reserve(normalized_tag_ids.size());
  std::map<int, std::size_t> requested_index_by_id;
  std::vector<ScaleCandidate> anonymous_tag_like_candidates;
  const auto register_tag_id = [&](int tag_id) {
    if (tag_id < 0 || requested_index_by_id.count(tag_id) != 0) {
      return;
    }
    const std::size_t index = states.size();
    requested_index_by_id[tag_id] = index;
    states.emplace_back();
    states.back().result.board_id = tag_id;
    states.back().result.original_longest_side =
        std::max(gray_original.cols, gray_original.rows);
    states.back().result.scale_configuration_mode = scale_mode_used;
    states.back().result.failure_reason = OuterTagFailureReason::NoDetectionsAtAll;
    states.back().result.failure_reason_text =
        ToString(states.back().result.failure_reason);
  };
  for (int tag_id : normalized_tag_ids) {
    register_tag_id(tag_id);
  }

  const auto process_scale = [&](const ScalePlanEntry& plan_entry) {
    const cv::Size scaled_size =
        MakeScaledSize(gray_original.size(), plan_entry.target_longest_side);
    const double scale_factor =
        static_cast<double>(std::max(scaled_size.width, scaled_size.height)) /
        static_cast<double>(std::max(gray_original.cols, gray_original.rows));

    cv::Mat scaled_gray;
    if (scaled_size == gray_original.size()) {
      scaled_gray = gray_original;
    } else {
      cv::resize(gray_original, scaled_gray, scaled_size, 0.0, 0.0, cv::INTER_AREA);
    }
    scaled_gray = MaybeBlur(scaled_gray, config_);

    const std::vector<AprilTags::TagDetection> detections = detector_->extractTags(scaled_gray);
    if (discover_automatically) {
      for (const AprilTags::TagDetection& detection : detections) {
        if (detection.good) {
          register_tag_id(detection.id);
        }
      }
    }

    struct PerTagScaleState {
      OuterTagScaleDebugInfo debug;
      ScaleCandidate best_candidate;
      bool has_best_candidate = false;
      std::vector<std::string> rejection_reasons;
    };
    std::vector<PerTagScaleState> scale_states(states.size());
    for (std::size_t index = 0; index < scale_states.size(); ++index) {
      scale_states[index].debug.target_longest_side = plan_entry.target_longest_side;
      scale_states[index].debug.configured_scale_divisor = plan_entry.configured_scale_divisor;
      scale_states[index].debug.attempted = true;
      scale_states[index].debug.scaled_size = scaled_size;
      scale_states[index].debug.scale_factor = scale_factor;
      scale_states[index].debug.raw_detection_count = static_cast<int>(detections.size());
      scale_states[index].debug.raw_detection_summaries.reserve(detections.size());
      states[index].saw_any_detection = states[index].saw_any_detection || !detections.empty();
    }

    for (const AprilTags::TagDetection& detection : detections) {
      const std::string detection_summary = SummarizeRawDetection(detection, scaled_size);
      for (PerTagScaleState& scale_state : scale_states) {
        scale_state.debug.raw_detection_summaries.push_back(detection_summary);
        if (detection.good) {
          ++scale_state.debug.raw_good_detection_count;
        }
      }
      if (config_.enable_anonymous_tag_like_geometry_rescue && detection.good &&
          !scale_states.empty()) {
        ScaleCandidate anonymous_candidate =
            BuildScaleCandidateFromDetection(detection, scale_states.front().debug,
                                             gray_original.size());
        if (PassesBorderCheck(anonymous_candidate.scaled_corners, scaled_gray.size(),
                              config_.min_border_distance) &&
            anonymous_candidate.scaled_area >= kMinQuadAreaPixels &&
            anonymous_candidate.min_edge >= kMinQuadEdgePixels &&
            anonymous_candidate.shape_quality > 0.10) {
          anonymous_tag_like_candidates.push_back(anonymous_candidate);
        }
      }
    }

    for (const AprilTags::TagDetection& detection : detections) {
      const std::map<int, std::size_t>::const_iterator requested_it =
          requested_index_by_id.find(detection.id);

      if (detection.good && !scale_states.empty()) {
        ScaleCandidate proposal_candidate = BuildScaleCandidateFromDetection(
            detection, scale_states.front().debug, gray_original.size());
        if (PassesBorderCheck(proposal_candidate.scaled_corners, scaled_gray.size(),
                              config_.min_border_distance) &&
            proposal_candidate.scaled_area >= kMinQuadAreaPixels &&
            proposal_candidate.min_edge >= kMinQuadEdgePixels &&
            proposal_candidate.shape_quality > 0.10) {
          for (PerTagOuterAggregationState& state : states) {
            if (state.result.board_id != detection.id) {
              StoreWrongIdProposal(proposal_candidate, "full_image_scale",
                                   &state.wrong_id_proposals);
            }
          }
        }
      }
      if (requested_it == requested_index_by_id.end()) {
        continue;
      }

      PerTagOuterAggregationState& aggregate_state = states[requested_it->second];
      PerTagScaleState& scale_state = scale_states[requested_it->second];
      ++scale_state.debug.matching_tag_count;
      aggregate_state.saw_matching_tag_id = true;
      if (detection.good) {
        ++scale_state.debug.matching_good_tag_count;
      }

      if (!detection.good) {
        scale_state.rejection_reasons.push_back("matched tag id but detection.good=false");
        aggregate_state.saw_non_border_matching_rejection = true;
        continue;
      }

      ScaleCandidate candidate =
          BuildScaleCandidateFromDetection(detection, scale_state.debug, gray_original.size());

      if (!PassesBorderCheck(candidate.scaled_corners, scaled_gray.size(),
                             config_.min_border_distance)) {
        scale_state.rejection_reasons.push_back(
            "matched tag id but rejected by scaled-image border distance");
        aggregate_state.saw_border_rejection = true;
        continue;
      }

      if (candidate.scaled_area < kMinQuadAreaPixels || candidate.min_edge < kMinQuadEdgePixels ||
          candidate.shape_quality <= 0.10) {
        scale_state.rejection_reasons.push_back(
            "matched tag id but quad geometry is unstable");
        aggregate_state.saw_non_border_matching_rejection = true;
        continue;
      }

      ++scale_state.debug.accepted_candidate_count;
      if (!scale_state.has_best_candidate ||
          IsCandidateBetter(candidate, scale_state.best_candidate)) {
        scale_state.best_candidate = candidate;
        scale_state.has_best_candidate = true;
      }
    }

    for (std::size_t index = 0; index < states.size(); ++index) {
      if (!scale_states[index].has_best_candidate &&
          !scale_states[index].rejection_reasons.empty()) {
        scale_states[index].debug.rejection_summary =
            JoinReasons(scale_states[index].rejection_reasons);
      }
      if (scale_states[index].has_best_candidate) {
        states[index].coarse_candidates.push_back(scale_states[index].best_candidate);
        states[index].result.successful_scale_longest_sides.push_back(
            scale_states[index].best_candidate.target_longest_side);
      }
      states[index].result.scale_debug.push_back(scale_states[index].debug);
    }
  };

  const auto has_unresolved_requested_tag = [&]() {
    for (const PerTagOuterAggregationState& state : states) {
      // Probe on a copy: the probe is only a cascade decision and must not
      // overwrite the final scale-debug record or selected corners.
      PerTagOuterAggregationState probe = state;
      const OuterTagDetectionResult detection = FinalizeOuterTagDetection(
          gray_original, config_, sphere_camera_.get(), &probe);
      if (!detection.success ||
          detection.hamming > config_.adaptive_coarse_max_hamming) {
        return true;
      }
    }
    return false;
  };

  const std::size_t coarse_scale_count = std::min(
      config_.adaptive_coarse_scale_divisors.size(), scale_plan.size());
  int coarse_scale_attempt_count = 0;
  int fallback_scale_attempt_count = 0;
  bool high_resolution_fallback_triggered = false;
  if (!config_.enable_adaptive_scale_cascade || discover_automatically) {
    for (const ScalePlanEntry& plan_entry : scale_plan) {
      process_scale(plan_entry);
    }
    coarse_scale_attempt_count = static_cast<int>(scale_plan.size());
  } else {
    for (std::size_t index = 0; index < coarse_scale_count; ++index) {
      process_scale(scale_plan[index]);
      ++coarse_scale_attempt_count;
    }
    bool needs_high_resolution_fallback = has_unresolved_requested_tag();
    high_resolution_fallback_triggered = needs_high_resolution_fallback;
    for (std::size_t index = coarse_scale_count;
         index < scale_plan.size() && needs_high_resolution_fallback; ++index) {
      process_scale(scale_plan[index]);
      ++fallback_scale_attempt_count;
      needs_high_resolution_fallback = has_unresolved_requested_tag();
    }
  }

  for (PerTagOuterAggregationState& state : states) {
    state.result.adaptive_coarse_scale_attempt_count = coarse_scale_attempt_count;
    state.result.adaptive_fallback_scale_attempt_count = fallback_scale_attempt_count;
    state.result.adaptive_high_resolution_fallback_triggered =
        high_resolution_fallback_triggered;
  }

  TryLocalSpherePatchRescue(gray_original, config_, sphere_camera_.get(),
                            requested_index_by_id, detector_.get(), &states);
  TryAnonymousTagLikeGeometryRescue(gray_original.size(), config_,
                                    anonymous_tag_like_candidates, &states);

  std::vector<OuterTagDetectionResult> results;
  results.reserve(states.size());
  for (std::size_t index = 0; index < states.size(); ++index) {
    results.push_back(
        FinalizeOuterTagDetection(gray_original, config_, sphere_camera_.get(), &states[index]));
  }
  return results;
}

OuterTagDetectionResult MultiScaleOuterTagDetector::Detect(const cv::Mat& image) const {
  const std::vector<OuterTagDetectionResult> results = DetectMultiple(image, {config_.tag_id});
  return results.empty() ? OuterTagDetectionResult{} : results.front();
}

void MultiScaleOuterTagDetector::DrawDetectionImpl(const OuterTagDetectionResult& detection,
                                                   cv::Mat* output_image,
                                                   bool draw_debug,
                                                   bool include_status_text) const {
  if (output_image == nullptr || output_image->empty()) {
    throw std::runtime_error("DrawDetection requires a valid output image.");
  }

  if (output_image->channels() == 1) {
    cv::cvtColor(*output_image, *output_image, cv::COLOR_GRAY2BGR);
  } else if (output_image->channels() == 4) {
    cv::cvtColor(*output_image, *output_image, cv::COLOR_BGRA2BGR);
  }

  const bool has_coarse =
      std::any_of(detection.coarse_corners_original_image.begin(),
                  detection.coarse_corners_original_image.end(),
                  [](const Eigen::Vector2d& point) { return point.squaredNorm() > 0.0; });
  const double render_scale =
      std::max(1.0, static_cast<double>(std::max(output_image->cols, output_image->rows)) / 1800.0);
  const int coarse_radius = std::max(4, static_cast<int>(std::lround(4.0 * render_scale)));
  const int refined_radius = std::max(4, static_cast<int>(std::lround(3.0 * render_scale)));
  const int verified_radius = std::max(5, static_cast<int>(std::lround(5.0 * render_scale)));
  const int subpix_radius = std::max(5, static_cast<int>(std::lround(4.0 * render_scale)));
  const int fusion_observation_radius =
      std::max(2, static_cast<int>(std::lround(2.5 * render_scale)));
  const int fusion_marker_size =
      std::max(12, static_cast<int>(std::lround(11.0 * render_scale)));
  const int line_thickness = std::max(1, static_cast<int>(std::lround(render_scale)));
  const double label_scale = std::max(0.9, 0.7 * render_scale);

  if (draw_debug && has_coarse) {
    for (int index = 0; index < 4; ++index) {
      const cv::Point2f coarse = ToPoint(detection.coarse_corners_original_image[static_cast<std::size_t>(index)]);
      cv::circle(*output_image, coarse, coarse_radius, cv::Scalar(0, 165, 255), line_thickness);
    }
  }

  const std::array<cv::Scalar, 2> branch_colors{
      cv::Scalar(255, 180, 0),
      cv::Scalar(0, 220, 255),
  };
  const std::array<cv::Scalar, 4> fusion_colors{
      cv::Scalar(80, 120, 255),
      cv::Scalar(80, 220, 120),
      cv::Scalar(255, 120, 120),
      cv::Scalar(220, 120, 255),
  };
  if (draw_debug) {
    for (int index = 0; index < 4; ++index) {
      const OuterCornerFusionDebugInfo& fusion =
          detection.corner_fusion_debug[static_cast<std::size_t>(index)];
      if (fusion.corner_index < 0) {
        continue;
      }

      const cv::Scalar corner_color = fusion_colors[static_cast<std::size_t>(index)];
      for (const OuterCornerScaleObservationDebugInfo& observation : fusion.scale_observations) {
        if (observation.rejected_as_outlier) {
          cv::circle(*output_image, observation.coarse_corner, fusion_observation_radius + 1,
                     corner_color, line_thickness, cv::LINE_AA);
          cv::drawMarker(*output_image, observation.coarse_corner, cv::Scalar(0, 0, 255),
                         cv::MARKER_TILTED_CROSS, fusion_observation_radius * 4,
                         std::max(1, line_thickness));
        } else {
          cv::circle(*output_image, observation.coarse_corner, fusion_observation_radius,
                     corner_color, -1, cv::LINE_AA);
        }
      }

      cv::drawMarker(*output_image, fusion.fused_corner, corner_color,
                     cv::MARKER_DIAMOND, fusion_marker_size, std::max(1, line_thickness + 1));
      cv::putText(*output_image, "F" + std::to_string(index),
                  fusion.fused_corner + cv::Point2f(static_cast<float>(8.0 * render_scale),
                                                    static_cast<float>(-12.0 * render_scale)),
                  cv::FONT_HERSHEY_PLAIN, label_scale, corner_color, line_thickness);

      std::ostringstream fusion_label;
      fusion_label << "ms=" << fusion.successful_scale_count
                   << " in=" << fusion.inlier_count
                   << " out=" << fusion.outlier_count
                   << " avg=" << std::fixed << std::setprecision(1) << fusion.average_deviation_before
                   << " max=" << fusion.max_deviation_before;
      cv::putText(*output_image, fusion_label.str(),
                  fusion.fused_corner + cv::Point2f(static_cast<float>(8.0 * render_scale),
                                                    static_cast<float>(18.0 * render_scale)),
                  cv::FONT_HERSHEY_PLAIN, std::max(0.75, 0.55 * render_scale),
                  corner_color, line_thickness);
    }

    for (int index = 0; index < 4; ++index) {
      const OuterCornerVerificationDebugInfo& verification =
          detection.corner_verification_debug[static_cast<std::size_t>(index)];
      if (verification.corner_index < 0) {
        continue;
      }

      const cv::Point2f coarse = verification.coarse_corner;
      const cv::Point2f subpix = verification.subpix_corner;
      const cv::Point2f spherical =
          verification.spherical_refinement_valid ? verification.spherical_corner : subpix;
      const cv::Point2f final_corner =
          verification.subpix_applied
              ? verification.subpix_corner
              : (verification.spherical_refinement_valid ? verification.spherical_corner
                                                         : verification.coarse_corner);
      for (const cv::Point2f& point : verification.prev_branch_points) {
        cv::circle(*output_image, point, std::max(2, line_thickness + 1), branch_colors[0], -1, cv::LINE_AA);
      }
      for (const cv::Point2f& point : verification.next_branch_points) {
        cv::circle(*output_image, point, std::max(2, line_thickness + 1), branch_colors[1], -1, cv::LINE_AA);
      }
      for (std::size_t point_index = 1;
           point_index < verification.prev_spherical_curve_points.size();
           ++point_index) {
        cv::line(*output_image,
                 verification.prev_spherical_curve_points[point_index - 1],
                 verification.prev_spherical_curve_points[point_index],
                 branch_colors[0], line_thickness, cv::LINE_AA);
      }
      for (std::size_t point_index = 1;
           point_index < verification.next_spherical_curve_points.size();
           ++point_index) {
        cv::line(*output_image,
                 verification.next_spherical_curve_points[point_index - 1],
                 verification.next_spherical_curve_points[point_index],
                 branch_colors[1], line_thickness, cv::LINE_AA);
      }
      if (verification.subpix_applied) {
        const cv::Point2f subpix_start =
            verification.spherical_refinement_valid ? spherical : coarse;
        cv::line(*output_image, subpix_start, subpix, cv::Scalar(255, 220, 0), line_thickness,
                 cv::LINE_AA);
        // subpix_window_radius is already expressed in original-image pixels.
        // Keep it unscaled here; render_scale is only for marker/text sizes.
        const int pre_boost_subpix_window_radius_px =
            std::max(2, verification.pre_boost_subpix_window_radius);
        const int subpix_window_radius_px =
            std::max(2, verification.subpix_window_radius);
        const cv::Scalar base_subpix_window_color(255, 170, 60);
        const cv::Scalar final_subpix_window_color(255, 255, 120);
        if (pre_boost_subpix_window_radius_px > 0) {
	          cv::rectangle(
	              *output_image,
	              cv::Rect(static_cast<int>(std::lround(subpix.x)) -
	                           pre_boost_subpix_window_radius_px,
	                       static_cast<int>(std::lround(subpix.y)) -
	                           pre_boost_subpix_window_radius_px,
	                       pre_boost_subpix_window_radius_px * 2 + 1,
	                       pre_boost_subpix_window_radius_px * 2 + 1),
	              base_subpix_window_color,
	              pre_boost_subpix_window_radius_px != subpix_window_radius_px ? 2 : 1,
	              cv::LINE_AA);
	        }
	        cv::rectangle(
	            *output_image,
	            cv::Rect(static_cast<int>(std::lround(subpix.x)) - subpix_window_radius_px,
	                     static_cast<int>(std::lround(subpix.y)) - subpix_window_radius_px,
	                     subpix_window_radius_px * 2 + 1,
	                     subpix_window_radius_px * 2 + 1),
	            final_subpix_window_color, 2, cv::LINE_AA);
      }
      if (verification.spherical_refinement_valid) {
        cv::line(*output_image, coarse, spherical, cv::Scalar(255, 80, 255), line_thickness,
                 cv::LINE_AA);
      } else if (!verification.subpix_applied) {
        cv::line(*output_image, coarse, final_corner, cv::Scalar(255, 80, 255), line_thickness, cv::LINE_AA);
      }
      cv::circle(*output_image, coarse, coarse_radius, cv::Scalar(0, 165, 255), line_thickness);
      if (verification.subpix_applied) {
        cv::drawMarker(*output_image, subpix, cv::Scalar(255, 255, 0),
                       cv::MARKER_CROSS, subpix_radius * 3, line_thickness);
      }
      if (verification.spherical_refinement_valid) {
        cv::drawMarker(*output_image, spherical, cv::Scalar(255, 80, 255),
                       cv::MARKER_DIAMOND, verified_radius * 3, line_thickness);
      }

      std::ostringstream label;
      label << index << " " << BuildOuterChainLabel(verification)
            << " d=" << std::fixed << std::setprecision(1)
            << verification.coarse_to_refined_displacement
            << " subpix=" << verification.subpix_window_radius;
      cv::putText(*output_image, label.str(),
                  coarse + cv::Point2f(static_cast<float>(6.0 * render_scale),
                                       static_cast<float>(14.0 * render_scale)),
                  cv::FONT_HERSHEY_PLAIN, label_scale,
                  cv::Scalar(255, 220, 0),
                  line_thickness);
      if (verification.subpix_applied) {
        std::ostringstream base_label;
        base_label << "base outer_scale r="
                   << verification.pre_boost_subpix_window_radius;
        cv::putText(*output_image, base_label.str(),
                    subpix + cv::Point2f(static_cast<float>(8.0 * render_scale),
                                         static_cast<float>(12.0 * render_scale)),
                    cv::FONT_HERSHEY_PLAIN, std::max(0.85, 0.68 * render_scale),
                    cv::Scalar(255, 170, 60), line_thickness);
        std::ostringstream final_label;
        final_label << "final subpix r=" << verification.subpix_window_radius;
        cv::putText(*output_image, final_label.str(),
                    subpix + cv::Point2f(static_cast<float>(8.0 * render_scale),
                                         static_cast<float>(28.0 * render_scale)),
                    cv::FONT_HERSHEY_PLAIN, std::max(0.85, 0.68 * render_scale),
                    cv::Scalar(255, 255, 0), line_thickness);
      }
      cv::putText(*output_image, "C",
                  coarse + cv::Point2f(static_cast<float>(-10.0 * render_scale),
                                       static_cast<float>(-8.0 * render_scale)),
                  cv::FONT_HERSHEY_PLAIN, label_scale, cv::Scalar(0, 165, 255), line_thickness);
      if (verification.subpix_applied) {
        cv::putText(*output_image, "S",
                    subpix + cv::Point2f(static_cast<float>(6.0 * render_scale),
                                         static_cast<float>(-8.0 * render_scale)),
                    cv::FONT_HERSHEY_PLAIN, label_scale, cv::Scalar(255, 255, 0), line_thickness);
      }
      if (verification.spherical_refinement_valid) {
        cv::putText(*output_image, "SP",
                    spherical + cv::Point2f(static_cast<float>(6.0 * render_scale),
                                            static_cast<float>(-8.0 * render_scale)),
                    cv::FONT_HERSHEY_PLAIN, label_scale, cv::Scalar(255, 80, 255), line_thickness);
      }
    }
  }

  if (detection.success) {
    const std::array<cv::Scalar, 4> edge_colors{
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 0, 255),
    };

    for (int index = 0; index < 4; ++index) {
      const cv::Point2f start =
          ToPoint(detection.refined_corners_original_image[static_cast<std::size_t>(index)]);
      const cv::Point2f end =
          ToPoint(detection.refined_corners_original_image[static_cast<std::size_t>((index + 1) % 4)]);
      cv::line(*output_image, start, end, edge_colors[static_cast<std::size_t>(index)],
               std::max(2, line_thickness));
    }

    for (int index = 0; index < 4; ++index) {
      const cv::Point2f refined =
          ToPoint(detection.refined_corners_original_image[static_cast<std::size_t>(index)]);
      const cv::Scalar point_color =
          detection.refined_valid[static_cast<std::size_t>(index)] ? cv::Scalar(0, 255, 255)
                                                                   : cv::Scalar(0, 64, 255);
      cv::circle(*output_image, refined, refined_radius, point_color, -1);
      cv::putText(*output_image, std::to_string(index),
                  refined + cv::Point2f(static_cast<float>(4.0 * render_scale),
                                        static_cast<float>(-4.0 * render_scale)),
                  cv::FONT_HERSHEY_PLAIN, label_scale, point_color, line_thickness);
    }

    cv::Point2f board_center(0.0f, 0.0f);
    for (int index = 0; index < 4; ++index) {
      board_center += ToPoint(
          detection.refined_corners_original_image[static_cast<std::size_t>(index)]);
    }
    board_center *= 0.25f;
    cv::putText(*output_image, "#" + std::to_string(detection.board_id),
                board_center + cv::Point2f(static_cast<float>(6.0 * render_scale),
                                           static_cast<float>(-10.0 * render_scale)),
                cv::FONT_HERSHEY_SIMPLEX, std::max(0.7, 0.5 * render_scale),
                cv::Scalar(230, 230, 230), std::max(2, line_thickness), cv::LINE_AA);
  }

  if (include_status_text) {
    const std::string headline =
        detection.success ? "status: multi-scale outer tag detection success"
                          : "status: multi-scale outer tag detection failed";
    cv::putText(*output_image, headline, cv::Point(20, 28), cv::FONT_HERSHEY_SIMPLEX,
                std::max(0.6, 0.45 * render_scale), cv::Scalar(0, 255, 255),
                std::max(2, line_thickness));

    std::ostringstream summary;
    summary << "tagId=" << detection.board_id
            << " ref_scale=" << detection.chosen_scale_longest_side
            << " mode=" << detection.scale_configuration_mode
            << " fused=" << (detection.used_corner_fusion ? "yes" : "no")
            << " hamming=" << detection.hamming
            << " quality=" << std::fixed << std::setprecision(2) << detection.quality;
    cv::putText(*output_image, summary.str(), cv::Point(20, 56), cv::FONT_HERSHEY_SIMPLEX,
                std::max(0.55, 0.4 * render_scale), cv::Scalar(255, 255, 0),
                std::max(1, line_thickness));

    std::ostringstream failure;
    failure << "failure_reason=" << detection.failure_reason_text;
    cv::putText(*output_image, failure.str(), cv::Point(20, 84), cv::FONT_HERSHEY_SIMPLEX,
                std::max(0.55, 0.4 * render_scale), cv::Scalar(0, 200, 255),
                std::max(1, line_thickness));
  }
}

void MultiScaleOuterTagDetector::DrawDetection(const OuterTagDetectionResult& detection,
                                               cv::Mat* output_image,
                                               bool draw_debug) const {
  DrawDetectionImpl(detection, output_image, draw_debug, true);
}

void MultiScaleOuterTagDetector::DrawDetections(
    const OuterTagMultiDetectionResult& detections,
    cv::Mat* output_image,
    bool draw_debug) const {
  if (output_image == nullptr || output_image->empty()) {
    throw std::runtime_error("DrawDetections requires a valid output image.");
  }

  if (output_image->channels() == 1) {
    cv::cvtColor(*output_image, *output_image, cv::COLOR_GRAY2BGR);
  } else if (output_image->channels() == 4) {
    cv::cvtColor(*output_image, *output_image, cv::COLOR_BGRA2BGR);
  }

  for (const OuterTagDetectionResult& detection : detections.detections) {
    DrawDetectionImpl(detection, output_image, draw_debug, false);
  }

  const int banner_height = draw_debug ? 132 : 96;
  cv::rectangle(*output_image, cv::Rect(0, 0, output_image->cols, banner_height),
                cv::Scalar(20, 20, 20), cv::FILLED);

  int rescue_attempted_count = 0;
  int rescue_used_count = 0;
  for (const OuterTagDetectionResult& detection : detections.detections) {
    rescue_attempted_count += detection.attempted_local_patch_rescue ? 1 : 0;
    rescue_used_count += detection.used_local_patch_rescue ? 1 : 0;
  }

  std::ostringstream requested_ids_stream;
  for (std::size_t index = 0; index < detections.requested_board_ids.size(); ++index) {
    if (index > 0) {
      requested_ids_stream << ",";
    }
    requested_ids_stream << detections.requested_board_ids[index];
  }

  const std::string headline = detections.AnySuccess()
                                   ? "status: multi-board outer detection"
                                   : "status: no valid multi-board outer detection";
  cv::putText(*output_image, headline, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.68,
              cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

  std::ostringstream summary;
  summary << "requested boards: [" << requested_ids_stream.str() << "]  valid detections: "
          << detections.SuccessfulBoardCount() << "/"
          << detections.requested_board_ids.size();
  cv::putText(*output_image, summary.str(), cv::Point(20, 58), cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

  std::ostringstream measurements;
  measurements << "outer frame measurements: "
               << detections.frame_measurements.board_measurements.size()
               << "  local patch rescue used: " << rescue_used_count
               << "/" << rescue_attempted_count;
  cv::putText(*output_image, measurements.str(), cv::Point(20, 86), cv::FONT_HERSHEY_SIMPLEX,
              0.55, cv::Scalar(0, 200, 255), 2, cv::LINE_AA);

  if (draw_debug) {
    const std::string legend =
        "outer legend: C orange, S yellow, SP magenta, refined cyan, #id center label";
    cv::putText(*output_image, legend, cv::Point(20, 114), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  }
}

cv::Mat MultiScaleOuterTagDetector::ToGray(const cv::Mat& image) const {
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image.clone();
  } else if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    throw std::runtime_error("Unsupported image format: expected 1, 3 or 4 channels.");
  }

  if (gray.depth() == CV_16U) {
    gray.convertTo(gray, CV_8U, 1.0 / 256.0);
  } else if (gray.depth() != CV_8U) {
    gray.convertTo(gray, CV_8U);
  }

  return gray;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
