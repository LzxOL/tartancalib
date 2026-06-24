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
  const char* label = "";
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
std::pair<double, double> ComputeEdgeRange(const std::array<cv::Point2f, 4>& corners);
bool IntersectLines(const FittedLine& first, const FittedLine& second, cv::Point2f* intersection);

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

std::vector<LocalSpherePatchPlan> BuildOuterLocalSpherePatchPlans() {
  struct PlanSeed {
    const char* label = "";
    cv::Point2f center{};
    double fov_deg = 44.0;
  };
  const std::array<PlanSeed, 32> seeds{{
      {"top_inner", cv::Point2f(0.50f, 0.28f), 44.0},
      {"bottom_inner", cv::Point2f(0.50f, 0.72f), 44.0},
      {"left_inner", cv::Point2f(0.28f, 0.50f), 44.0},
      {"right_inner", cv::Point2f(0.72f, 0.50f), 44.0},
      {"top_left_inner", cv::Point2f(0.30f, 0.30f), 44.0},
      {"top_right_inner", cv::Point2f(0.70f, 0.30f), 44.0},
      {"bottom_left_inner", cv::Point2f(0.30f, 0.70f), 44.0},
      {"bottom_right_inner", cv::Point2f(0.70f, 0.70f), 44.0},
      {"top_outer", cv::Point2f(0.50f, 0.18f), 44.0},
      {"bottom_outer", cv::Point2f(0.50f, 0.82f), 44.0},
      {"left_outer", cv::Point2f(0.18f, 0.50f), 44.0},
      {"right_outer", cv::Point2f(0.82f, 0.50f), 44.0},
      {"top_left_outer", cv::Point2f(0.26f, 0.26f), 44.0},
      {"top_right_outer", cv::Point2f(0.74f, 0.26f), 44.0},
      {"bottom_left_outer", cv::Point2f(0.26f, 0.74f), 44.0},
      {"bottom_right_outer", cv::Point2f(0.74f, 0.74f), 44.0},
      // Extra stress-case patches for close boards near the fisheye boundary.
      // They are only used after ordinary multiscale detection sees tags but
      // fails to decode the requested board id, so they do not change normal
      // detections or relax the AprilTag id/hamming checks.
      {"left_edge_mid", cv::Point2f(0.10f, 0.50f), 58.0},
      {"left_edge_upper", cv::Point2f(0.10f, 0.38f), 58.0},
      {"left_edge_lower", cv::Point2f(0.10f, 0.62f), 58.0},
      {"left_edge_mid_wide", cv::Point2f(0.14f, 0.50f), 68.0},
      {"left_edge_upper_wide", cv::Point2f(0.14f, 0.36f), 68.0},
      {"left_edge_lower_wide", cv::Point2f(0.14f, 0.64f), 68.0},
      {"bottom_edge_mid", cv::Point2f(0.50f, 0.90f), 58.0},
      {"bottom_edge_left", cv::Point2f(0.38f, 0.90f), 58.0},
      {"bottom_edge_right", cv::Point2f(0.62f, 0.90f), 58.0},
      {"bottom_edge_mid_wide", cv::Point2f(0.50f, 0.86f), 68.0},
      {"bottom_edge_left_wide", cv::Point2f(0.36f, 0.86f), 68.0},
      {"bottom_edge_right_wide", cv::Point2f(0.64f, 0.86f), 68.0},
      {"bottom_left_edge", cv::Point2f(0.18f, 0.86f), 62.0},
      {"bottom_right_edge", cv::Point2f(0.82f, 0.86f), 62.0},
      {"top_left_edge", cv::Point2f(0.18f, 0.14f), 62.0},
      {"top_right_edge", cv::Point2f(0.82f, 0.14f), 62.0},
  }};

  std::vector<LocalSpherePatchPlan> plans;
  plans.reserve(seeds.size());
  for (const PlanSeed& seed : seeds) {
    LocalSpherePatchPlan plan;
    plan.label = seed.label;
    plan.normalized_x = seed.center.x;
    plan.normalized_y = seed.center.y;
    plan.fov_deg = seed.fov_deg;
    plans.push_back(plan);
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
    } else if (key == "minBorderDistance" || key == "min_border_distance") {
      config.min_border_distance = ParseDouble(key, value);
    } else if (key == "maxScalesToTry" || key == "max_scales_to_try") {
      config.max_scales_to_try = ParseInt(key, value);
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
    } else if (key == "closeEdgeOuterSubpixBorderRatio" ||
               key == "close_edge_outer_subpix_border_ratio") {
      config.close_edge_outer_subpix_border_ratio = ParseDouble(key, value);
    } else if (key == "closeEdgeOuterSubpixMultiplier" ||
               key == "close_edge_outer_subpix_multiplier") {
      config.close_edge_outer_subpix_multiplier = ParseDouble(key, value);
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
    } else if (key == "enableInterpolatedMissingBoardGeometryRescue" ||
               key == "enable_interpolated_missing_board_geometry_rescue") {
      config.enable_interpolated_missing_board_geometry_rescue = ParseBool(key, value);
    } else if (key == "anonymousTagLikeRescueMaxCenterErrorScale" ||
               key == "anonymous_tag_like_rescue_max_center_error_scale") {
      config.anonymous_tag_like_rescue_max_center_error_scale = ParseDouble(key, value);
    } else if (key == "anonymousTagLikeRescueMinAreaRatio" ||
               key == "anonymous_tag_like_rescue_min_area_ratio") {
      config.anonymous_tag_like_rescue_min_area_ratio = ParseDouble(key, value);
    } else if (key == "anonymousTagLikeRescueMaxAreaRatio" ||
               key == "anonymous_tag_like_rescue_max_area_ratio") {
      config.anonymous_tag_like_rescue_max_area_ratio = ParseDouble(key, value);
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
    result.final_radius = result.raw_radius;
    result.clamped = false;
    return result;
  }

  result.scaled_radius =
      config.outer_subpix_scale > 0.0
          ? config.outer_subpix_scale * result.corner_marker_width
          : static_cast<double>(kOuterSubpixRadiusMin);
  result.raw_radius =
      std::max(kOuterSubpixRadiusMin,
               static_cast<int>(std::lround(result.scaled_radius)));
  result.final_radius = result.raw_radius;
  result.clamped = false;
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

bool IsCloseEdgeOuterSubpixBoostCase(
    const std::array<cv::Point2f, 4>& corners,
    const cv::Size& image_size,
    const MultiScaleOuterTagDetectorConfig& config,
    const DoubleSphereCameraModel* sphere_camera) {
  if (!config.enable_close_edge_outer_subpix_boost ||
      config.close_edge_outer_subpix_multiplier <= 1.0 ||
      image_size.width <= 0 || image_size.height <= 0) {
    return false;
  }
  const double area_ratio = ComputeAreaRatio(corners, image_size);
  const bool near_board_by_area =
      area_ratio >= config.close_edge_outer_subpix_area_ratio;

  double max_polar_deg = 0.0;
  const bool have_polar =
      ComputeMaxPolarOrImageProxyDeg(corners, image_size, sphere_camera,
                                     &max_polar_deg);
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
  return near_board_by_area || (high_polar && near_border);
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

double ComputeRefineQuality(const std::array<cv::Point2f, 4>& coarse,
                            const std::array<cv::Point2f, 4>& refined,
                            double max_outer_refine_displacement,
                            std::array<bool, 4>* valid_mask,
                            std::array<double, 4>* quality_mask) {
  if (max_outer_refine_displacement <= 0.0) {
    throw std::runtime_error("max_outer_refine_displacement must be positive.");
  }

  double min_quality = 1.0;
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f delta = refined[index] - coarse[index];
    const double displacement = std::hypot(delta.x, delta.y);
    const bool valid = displacement <= max_outer_refine_displacement;
    const double quality = ClampUnit(1.0 - displacement / max_outer_refine_displacement);
    (*valid_mask)[static_cast<std::size_t>(index)] = valid;
    (*quality_mask)[static_cast<std::size_t>(index)] = quality;
    min_quality = std::min(min_quality, quality);
  }
  return min_quality;
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
    *scale_mode_used = "fixed_schedule";
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

  for (const double divisor : kOuterFixedScaleDivisors) {
    if (divisor <= 0.0) {
      continue;
    }
    const int target_longest_side =
        std::max(1, static_cast<int>(std::lround(static_cast<double>(original_longest) / divisor)));
    append_entry(target_longest_side, divisor);
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
  const bool close_edge_subpix_boost =
      IsCloseEdgeOuterSubpixBoostCase(coarse_original, gray_original.size(),
                                      config, sphere_camera);
  const double close_edge_area_ratio =
      ComputeAreaRatio(coarse_original, gray_original.size());
  double close_edge_max_polar_deg = 0.0;
  (void)ComputeMaxPolarOrImageProxyDeg(coarse_original, gray_original.size(),
                                       sphere_camera,
                                       &close_edge_max_polar_deg);

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
    if (close_edge_subpix_boost) {
      debug.verification_roi_radius = std::max(
          debug.verification_roi_radius,
          static_cast<int>(std::lround(
              static_cast<double>(debug.verification_roi_radius) *
              config.close_edge_outer_subpix_multiplier)));
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
              config.close_edge_outer_subpix_multiplier)));
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

    // The legacy local line intersection can fail on close/edge boards when a
    // single bad branch intersects far outside the image.  Refit the local
    // marker-edge support sets and use that intersection as a conservative
    // fallback seed before the final cornerSubPix step.  This fallback is
    // intentionally independent of close-edge window boosting: it improves the
    // seed without requiring a larger subpixel window.
    if (!accepted_line_seed) {
      const ImageLineCornerRefinement image_line_refinement =
          RefineCornerByImageLineSupportIntersection(
              debug.prev_marker_support_points,
              debug.next_marker_support_points);
      debug.image_line_valid = image_line_refinement.success;
      debug.image_line_corner = image_line_refinement.success
                                    ? image_line_refinement.refined_corner
                                    : coarse_original[static_cast<std::size_t>(index)];
      debug.prev_image_line_residual = image_line_refinement.prev_line.rms_residual;
      debug.next_image_line_residual = image_line_refinement.next_line.rms_residual;
      debug.prev_image_line_support_count =
          image_line_refinement.prev_line.support_count;
      debug.next_image_line_support_count =
          image_line_refinement.next_line.support_count;

      const cv::Point2f image_line_delta =
          image_line_refinement.refined_corner - coarse_corner;
      const double image_line_gap =
          std::hypot(image_line_delta.x, image_line_delta.y);
      const bool image_line_inside =
          image_line_refinement.refined_corner.x >= config.min_border_distance &&
          image_line_refinement.refined_corner.x <=
              static_cast<float>(gray_original.cols) - config.min_border_distance &&
          image_line_refinement.refined_corner.y >= config.min_border_distance &&
          image_line_refinement.refined_corner.y <=
              static_cast<float>(gray_original.rows) - config.min_border_distance;
      const double fallback_gap_limit =
          std::max(static_cast<double>(debug.verification_roi_radius),
                   1.5 * static_cast<double>(debug.subpix_window_radius));
      if (image_line_refinement.success &&
          image_line_refinement.quality >= kOuterLineMinQuality &&
          image_line_inside &&
          image_line_gap <= fallback_gap_limit) {
        accepted_line_seed = true;
        accepted_line_corner = image_line_refinement.refined_corner;
        accepted_line_quality = image_line_refinement.quality;
      } else {
        FittedLine prev_fallback_line = image_line_refinement.prev_line;
        FittedLine next_fallback_line = image_line_refinement.next_line;
        if (!prev_fallback_line.valid && next_fallback_line.valid) {
          prev_fallback_line.valid = true;
          prev_fallback_line.anchor =
              refined_candidate.refined_original[static_cast<std::size_t>(prev_index)];
          prev_fallback_line.direction =
              NormalizeVector(coarse_corner -
                              coarse_original[static_cast<std::size_t>(prev_index)]);
          prev_fallback_line.support_count = 2;
          prev_fallback_line.rms_residual = 0.0;
        }
        if (!next_fallback_line.valid && prev_fallback_line.valid) {
          next_fallback_line.valid = true;
          next_fallback_line.anchor =
              refined_candidate.refined_original[static_cast<std::size_t>(next_index)];
          next_fallback_line.direction =
              NormalizeVector(coarse_original[static_cast<std::size_t>(next_index)] -
                              coarse_corner);
          next_fallback_line.support_count = 2;
          next_fallback_line.rms_residual = 0.0;
        }
        cv::Point2f geometric_image_line_corner;
        if (prev_fallback_line.valid &&
            next_fallback_line.valid &&
            IntersectLines(prev_fallback_line, next_fallback_line,
                           &geometric_image_line_corner)) {
          const cv::Point2f geometric_delta =
              geometric_image_line_corner - coarse_corner;
          const double geometric_gap =
              std::hypot(geometric_delta.x, geometric_delta.y);
          const bool geometric_inside =
              geometric_image_line_corner.x >= config.min_border_distance &&
              geometric_image_line_corner.x <=
                  static_cast<float>(gray_original.cols) - config.min_border_distance &&
              geometric_image_line_corner.y >= config.min_border_distance &&
              geometric_image_line_corner.y <=
                  static_cast<float>(gray_original.rows) - config.min_border_distance;
          if (geometric_inside && geometric_gap <= fallback_gap_limit) {
            accepted_line_seed = true;
            accepted_line_corner = geometric_image_line_corner;
            accepted_line_quality =
                std::max(kOuterLineMinQuality,
                         image_line_refinement.quality);
            debug.image_line_valid = true;
            debug.image_line_corner = geometric_image_line_corner;
          }
        }
      }
    }

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
      std::vector<cv::Point2f> point_seed{
          refined_candidate.refined_original[static_cast<std::size_t>(index)]};
      const int subpix_radius =
          refined_candidate.verification_debug[static_cast<std::size_t>(index)].subpix_window_radius;
      cv::cornerSubPix(
          gray_original, point_seed,
          cv::Size(subpix_radius, subpix_radius),
          cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
      refined_candidate.refined_original[static_cast<std::size_t>(index)] = point_seed.front();
      refined_candidate.verification_debug[static_cast<std::size_t>(index)].subpix_corner =
          point_seed.front();
      refined_candidate.verification_debug[static_cast<std::size_t>(index)].subpix_applied = true;
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
    debug.refined_valid = method_valid[static_cast<std::size_t>(index)];
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
  if (config_.close_edge_outer_subpix_border_ratio < 0.0) {
    throw std::runtime_error("close_edge_outer_subpix_border_ratio must be non-negative.");
  }
  if (config_.close_edge_outer_subpix_multiplier <= 0.0) {
    throw std::runtime_error("close_edge_outer_subpix_multiplier must be positive.");
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
};

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

void TryInterpolatedMissingBoardGeometryRescue(
    const cv::Mat& gray_original,
    const MultiScaleOuterTagDetectorConfig& config,
    const DoubleSphereCameraModel* sphere_camera,
    std::vector<PerTagOuterAggregationState>* states) {
  if (!config.enable_interpolated_missing_board_geometry_rescue ||
      states == nullptr || states->size() < 3) {
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

  auto find_state_by_board = [&](int board_id) -> const PerTagOuterAggregationState* {
    for (const PerTagOuterAggregationState& state : *states) {
      if (state.result.board_id == board_id) {
        return &state;
      }
    }
    return nullptr;
  };

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

    const PerTagOuterAggregationState* lower_state = find_state_by_board(lower->board_id);
    const PerTagOuterAggregationState* upper_state = find_state_by_board(upper->board_id);
    const ScaleCandidate* lower_candidate =
        lower_state == nullptr ? nullptr : BestCoarseCandidate(*lower_state);
    const ScaleCandidate* upper_candidate =
        upper_state == nullptr ? nullptr : BestCoarseCandidate(*upper_state);
    if (lower_candidate == nullptr || upper_candidate == nullptr) {
      continue;
    }

    const double id_alpha =
        static_cast<double>(target_board_id - lower->board_id) /
        static_cast<double>(upper->board_id - lower->board_id);
    if (id_alpha <= 0.0 || id_alpha >= 1.0) {
      continue;
    }
    std::array<cv::Point2f, 4> predicted_corners{};
    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      predicted_corners[static_cast<std::size_t>(corner_index)] =
          lower_candidate->original_corners[static_cast<std::size_t>(corner_index)] +
          (upper_candidate->original_corners[static_cast<std::size_t>(corner_index)] -
           lower_candidate->original_corners[static_cast<std::size_t>(corner_index)]) *
              static_cast<float>(id_alpha);
    }
    if (!PassesBorderCheck(predicted_corners, gray_original.size(),
                           config.min_border_distance) ||
        ComputeQuadArea(predicted_corners) < kMinQuadAreaPixels) {
      continue;
    }

    std::ostringstream seed_label;
    seed_label << "interpolated_missing_board_geometry_seed target_id=" << target_board_id
               << " bracket=" << lower->board_id << "," << upper->board_id;
    ScaleCandidate seed_candidate = BuildSyntheticCandidateFromOriginalCorners(
        target_board_id, gray_original.size(), predicted_corners, 3, seed_label.str());

    const RefinedCandidate refined_candidate =
        RefineCoarseCandidate(gray_original, seed_candidate, predicted_corners,
                              config, sphere_camera);
    int image_evidence_corner_count = 0;
    double max_displacement = 0.0;
    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      const OuterCornerVerificationDebugInfo& debug =
          refined_candidate.verification_debug[static_cast<std::size_t>(corner_index)];
      if (debug.line_seed_accepted || debug.spherical_refinement_applied ||
          debug.image_line_valid) {
        ++image_evidence_corner_count;
      }
      max_displacement = std::max(
          max_displacement,
          Norm(refined_candidate.refined_original[static_cast<std::size_t>(corner_index)] -
               predicted_corners[static_cast<std::size_t>(corner_index)]));
    }
    const double expected_edge = std::max(
        kMinQuadEdgePixels, (1.0 - id_alpha) * lower->mean_edge + id_alpha * upper->mean_edge);
    const bool refined_inside =
        PassesBorderCheck(refined_candidate.refined_original, gray_original.size(),
                          config.min_border_distance);
    const double refined_area = ComputeQuadArea(refined_candidate.refined_original);
    const double max_allowed_displacement = std::max(18.0, 0.35 * expected_edge);
    if (image_evidence_corner_count < 3 ||
        max_displacement > max_allowed_displacement ||
        !refined_inside ||
        refined_area < kMinQuadAreaPixels) {
      state.attempted_local_patch_rescue = true;
      std::ostringstream reject_summary;
      reject_summary << "interpolated_missing_board_geometry_rejected target_id="
                     << target_board_id
                     << " bracket=" << lower->board_id << "," << upper->board_id
                     << " evidence_corners=" << image_evidence_corner_count
                     << " max_refine_displacement_px=" << std::fixed << std::setprecision(2)
                     << max_displacement
                     << " max_allowed_px=" << max_allowed_displacement
                     << " refined_inside=" << (refined_inside ? 1 : 0)
                     << " refined_area=" << refined_area;
      state.local_patch_rescue_summaries.push_back(reject_summary.str());
      continue;
    }

    std::ostringstream rescue_summary;
    rescue_summary << "interpolated_missing_board_geometry_rescue target_id=" << target_board_id
                   << " bracket=" << lower->board_id << "," << upper->board_id
                   << " evidence_corners=" << image_evidence_corner_count
                   << " max_refine_displacement_px=" << std::fixed << std::setprecision(2)
                   << max_displacement
                   << " quality=" << refined_candidate.quality;
    ScaleCandidate accepted_candidate = BuildSyntheticCandidateFromOriginalCorners(
        target_board_id, gray_original.size(), refined_candidate.refined_original,
        3, rescue_summary.str());
    state.saw_matching_tag_id = true;
    state.attempted_local_patch_rescue = true;
    state.coarse_candidates.push_back(accepted_candidate);
    state.result.successful_scale_longest_sides.push_back(
        accepted_candidate.target_longest_side);
    state.local_patch_rescue_summaries.push_back(rescue_summary.str());
  }
}

void TryLocalSpherePatchRescue(
    const cv::Mat& gray_original,
    const MultiScaleOuterTagDetectorConfig& config,
    const DoubleSphereCameraModel* sphere_camera,
    const std::map<int, std::size_t>& requested_index_by_id,
    AprilTags::TagDetector* detector,
    std::vector<PerTagOuterAggregationState>* states) {
  if (states == nullptr || detector == nullptr || sphere_camera == nullptr ||
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
    if (state.saw_any_detection && !state.saw_matching_tag_id) {
      state.attempted_local_patch_rescue = true;
      unresolved_indices.push_back(index);
    }
  }
  if (unresolved_indices.empty()) {
    return;
  }

  const std::vector<LocalSpherePatchPlan> patch_plans = BuildOuterLocalSpherePatchPlans();
  for (const LocalSpherePatchPlan& patch_plan : patch_plans) {
    if (unresolved_indices.empty()) {
      break;
    }

    LocalSpherePatchContext patch_context;
    if (!BuildLocalSpherePatch(gray_original, *sphere_camera, patch_plan, &patch_context)) {
      continue;
    }

    const std::vector<AprilTags::TagDetection> detections = detector->extractTags(patch_context.patch);
    if (detections.empty()) {
      continue;
    }

    for (const AprilTags::TagDetection& detection : detections) {
      const auto requested_it = requested_index_by_id.find(detection.id);
      if (requested_it == requested_index_by_id.end()) {
        continue;
      }

      PerTagOuterAggregationState& state = (*states)[requested_it->second];
      if (state.saw_matching_tag_id || !state.attempted_local_patch_rescue || !detection.good) {
        continue;
      }

      ScaleCandidate candidate;
      if (!BuildScaleCandidateFromPatchDetection(detection, patch_context, *sphere_camera,
                                                 &candidate)) {
        continue;
      }
      if (candidate.scaled_area < kMinQuadAreaPixels || candidate.min_edge < kMinQuadEdgePixels ||
          candidate.shape_quality <= 0.10) {
        continue;
      }

      state.saw_matching_tag_id = true;
      state.coarse_candidates.push_back(candidate);

      std::ostringstream rescue_summary;
      rescue_summary << "local_sphere_patch label=" << patch_context.label
                     << " id=" << detection.id
                     << " ham=" << detection.hammingDistance
                     << " area=" << std::lround(candidate.scaled_area);
      state.local_patch_rescue_summaries.push_back(rescue_summary.str());

      unresolved_indices.erase(
          std::remove(unresolved_indices.begin(), unresolved_indices.end(), requested_it->second),
          unresolved_indices.end());
    }
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
  result.requested_board_ids = requested_board_ids_;
  result.detections = DetectMultiple(image, requested_board_ids_);
  result.frame_measurements =
      BuildOuterFrameMeasurementResult(result.image_size,
                                       result.requested_board_ids,
                                       result.detections);
  return result;
}

std::vector<OuterTagDetectionResult> MultiScaleOuterTagDetector::DetectMultiple(
    const cv::Mat& image, const std::vector<int>& requested_tag_ids) const {
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  const std::vector<int> normalized_tag_ids =
      NormalizeBoardIds(requested_tag_ids, config_.tag_id);
  if (normalized_tag_ids.empty()) {
    throw std::runtime_error("DetectMultiple requires at least one valid requested tag id.");
  }

  const cv::Mat gray_original = ToGray(image);
  std::string scale_mode_used;
  const std::vector<ScalePlanEntry> scale_plan =
      BuildScalePlan(gray_original.size(), config_, &scale_mode_used);

  std::vector<PerTagOuterAggregationState> states(normalized_tag_ids.size());
  std::map<int, std::size_t> requested_index_by_id;
  std::vector<ScaleCandidate> anonymous_tag_like_candidates;
  for (std::size_t index = 0; index < normalized_tag_ids.size(); ++index) {
    requested_index_by_id[normalized_tag_ids[index]] = index;
    states[index].result.board_id = normalized_tag_ids[index];
    states[index].result.original_longest_side =
        std::max(gray_original.cols, gray_original.rows);
    states[index].result.scale_configuration_mode = scale_mode_used;
    states[index].result.failure_reason = OuterTagFailureReason::NoDetectionsAtAll;
    states[index].result.failure_reason_text = ToString(states[index].result.failure_reason);
  }

  for (const ScalePlanEntry& plan_entry : scale_plan) {
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
  }

  TryLocalSpherePatchRescue(gray_original, config_, sphere_camera_.get(),
                            requested_index_by_id, detector_.get(), &states);
  TryAnonymousTagLikeGeometryRescue(gray_original.size(), config_,
                                    anonymous_tag_like_candidates, &states);
  TryInterpolatedMissingBoardGeometryRescue(gray_original, config_, sphere_camera_.get(),
                                            &states);

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
        const int subpix_window_radius_px =
            std::max(2, verification.subpix_window_radius);
        cv::rectangle(
            *output_image,
            cv::Rect(static_cast<int>(std::lround(subpix.x)) - subpix_window_radius_px,
                     static_cast<int>(std::lround(subpix.y)) - subpix_window_radius_px,
                     subpix_window_radius_px * 2 + 1,
                     subpix_window_radius_px * 2 + 1),
            cv::Scalar(255, 255, 120), 1, cv::LINE_AA);
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
        std::ostringstream subpix_label;
        subpix_label << "r=" << verification.subpix_window_radius;
        cv::putText(*output_image, subpix_label.str(),
                    subpix + cv::Point2f(static_cast<float>(8.0 * render_scale),
                                         static_cast<float>(12.0 * render_scale)),
                    cv::FONT_HERSHEY_PLAIN, std::max(0.85, 0.75 * render_scale),
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
