#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

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
constexpr int kOuterRefinementWindowRadius = 12;
constexpr double kOuterLineDerivativeDelta = 1.5;
constexpr double kOuterLineResidualThreshold = 2.5;
constexpr int kMinLineSupportPoints = 6;
constexpr double kOuterLineMinQuality = 0.25;
constexpr double kOuterLineSubpixAgreementPixels = 6.0;
constexpr int kVerificationLineMinSupportPoints = 4;
constexpr int kOuterVerificationCandidateStepPixels = 2;
constexpr double kOuterDirectionAlignmentFloor = 0.75;
constexpr double kOuterLayoutContrastFloor = 8.0;
constexpr double kOuterLayoutContrastRange = 40.0;

struct ScaleCandidate {
  int target_longest_side = 0;
  double scale_factor = 1.0;
  cv::Size scaled_size;
  AprilTags::TagDetection detection;
  std::array<cv::Point2f, 4> scaled_corners{};
  double scaled_area = 0.0;
  double min_edge = 0.0;
  double max_edge = 0.0;
  double shape_quality = 0.0;
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

struct OuterCornerLocalVerificationResult {
  cv::Point2f verified_corner;
  cv::Rect verification_roi;
  cv::Point2f prev_edge_direction{};
  cv::Point2f next_edge_direction{};
  DirectionalEdgeBranch prev_branch;
  DirectionalEdgeBranch next_branch;
  double local_scale = 0.0;
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

std::vector<int> ParseIntList(const std::string& key, const std::string& value) {
  std::string cleaned = value;
  cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '['), cleaned.end());
  cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ']'), cleaned.end());

  std::replace(cleaned.begin(), cleaned.end(), ',', ' ');
  std::stringstream stream(cleaned);
  std::vector<int> parsed;
  std::string token;
  while (stream >> token) {
    parsed.push_back(ParseInt(key, token));
  }

  if (parsed.empty()) {
    throw std::runtime_error("Field '" + key + "' must contain at least one scale candidate.");
  }
  return parsed;
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
    } else if (key == "minBorderDistance" || key == "min_border_distance") {
      config.min_border_distance = ParseDouble(key, value);
    } else if (key == "maxScalesToTry" || key == "max_scales_to_try") {
      config.max_scales_to_try = ParseInt(key, value);
    } else if (key == "scaleCandidates" || key == "scale_candidates") {
      config.scale_candidates = ParseIntList(key, value);
    } else if (key == "doOuterSubpixRefinement" || key == "do_outer_subpix_refinement") {
      config.do_outer_subpix_refinement = ParseBool(key, value);
    } else if (key == "maxOuterRefineDisplacement" || key == "max_outer_refine_displacement") {
      config.max_outer_refine_displacement = ParseDouble(key, value);
    } else if (key == "minDetectionQuality" || key == "min_detection_quality") {
      config.min_detection_quality = ParseDouble(key, value);
    } else if (key == "blurBeforeDetect" || key == "blur_before_detect") {
      config.blur_before_detect = ParseBool(key, value);
    } else if (key == "blurKernel" || key == "blur_kernel") {
      config.blur_kernel = ParseInt(key, value);
    } else if (key == "blurSigma" || key == "blur_sigma") {
      config.blur_sigma = ParseDouble(key, value);
    } else if (key == "enableOuterCornerLocalVerification" ||
               key == "enable_outer_corner_local_verification") {
      config.enable_outer_corner_local_verification = ParseBool(key, value);
    } else if (key == "enableOuterCornerLayoutCheck" ||
               key == "enable_outer_corner_layout_check") {
      config.enable_outer_corner_layout_check = ParseBool(key, value);
    } else if (key == "outerCornerVerificationRoiScale" ||
               key == "outer_corner_verification_roi_scale") {
      config.outer_corner_verification_roi_scale = ParseDouble(key, value);
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
      ClampRadiusFromScale(config.outer_corner_verification_roi_scale, local_scale,
                           config.outer_corner_verification_roi_min,
                           config.outer_corner_verification_roi_max);
  radii.candidate_radius =
      ClampRadiusFromScale(config.outer_corner_candidate_scale, local_scale,
                           config.outer_corner_candidate_min,
                           config.outer_corner_candidate_max);
  radii.candidate_radius =
      std::min(radii.candidate_radius, std::max(0, radii.verification_roi_radius - 1));
  radii.branch_search_radius =
      ClampRadiusFromScale(config.outer_corner_branch_search_scale, local_scale,
                           config.outer_corner_branch_search_min,
                           config.outer_corner_branch_search_max);
  radii.branch_search_radius =
      std::min(radii.branch_search_radius, std::max(1, radii.verification_roi_radius - 1));
  return radii;
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
  debug.verification_roi_radius = verification.verification_roi_radius;
  debug.candidate_radius = verification.candidate_radius;
  debug.branch_search_radius = verification.branch_search_radius;
  debug.direction_consistency_score = verification.direction_consistency_score;
  debug.local_layout_score = verification.local_layout_score;
  debug.verification_quality = verification.verification_quality;
  debug.verification_passed = verification.verification_passed;
  debug.subpix_applied = false;
  debug.failure_reason = verification.failure_reason;
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
  best_any.verification_roi_radius = radii.verification_roi_radius;
  best_any.candidate_radius = radii.candidate_radius;
  best_any.branch_search_radius = radii.branch_search_radius;

  if (!config.enable_outer_corner_local_verification) {
    best_any.direction_consistency_score = 1.0;
    best_any.local_layout_score = 1.0;
    best_any.verification_quality = 1.0;
    best_any.verification_passed = true;
    best_any.failure_reason = "verification_disabled";
    return best_any;
  }

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

std::vector<int> BuildScaleList(const cv::Size& original_size,
                                const MultiScaleOuterTagDetectorConfig& config) {
  const int original_longest = std::max(original_size.width, original_size.height);
  std::vector<int> scales;
  scales.push_back(original_longest);

  for (const int candidate : config.scale_candidates) {
    if (candidate <= 0 || candidate >= original_longest) {
      continue;
    }
    if (std::find(scales.begin(), scales.end(), candidate) == scales.end()) {
      scales.push_back(candidate);
    }
  }

  if (config.max_scales_to_try > 0 && static_cast<int>(scales.size()) > config.max_scales_to_try) {
    scales.resize(static_cast<std::size_t>(config.max_scales_to_try));
  }

  return scales;
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

MultiScaleOuterTagDetector::MultiScaleOuterTagDetector(MultiScaleOuterTagDetectorConfig config)
    : config_(std::move(config)) {
  if (config_.tag_id < 0) {
    throw std::runtime_error("tag_id must be non-negative.");
  }
  if (config_.min_border_distance < 0.0) {
    throw std::runtime_error("min_border_distance must be non-negative.");
  }
  if (config_.max_outer_refine_displacement <= 0.0) {
    throw std::runtime_error("max_outer_refine_displacement must be positive.");
  }
  if (config_.min_detection_quality < 0.0 || config_.min_detection_quality > 1.0) {
    throw std::runtime_error("min_detection_quality must be in [0, 1].");
  }
  if (config_.scale_candidates.empty()) {
    throw std::runtime_error("scale_candidates must not be empty.");
  }
  if (config_.outer_corner_verification_roi_scale < 0.0) {
    throw std::runtime_error("outer_corner_verification_roi_scale must be non-negative.");
  }
  if (config_.outer_corner_verification_roi_min <= 0 ||
      config_.outer_corner_verification_roi_max < config_.outer_corner_verification_roi_min) {
    throw std::runtime_error("outer_corner_verification_roi_min/max must define a positive valid range.");
  }
  if (config_.outer_corner_candidate_scale < 0.0) {
    throw std::runtime_error("outer_corner_candidate_scale must be non-negative.");
  }
  if (config_.outer_corner_candidate_min < 0 ||
      config_.outer_corner_candidate_max < config_.outer_corner_candidate_min) {
    throw std::runtime_error("outer_corner_candidate_min/max must define a valid range.");
  }
  if (config_.outer_corner_branch_search_scale < 0.0) {
    throw std::runtime_error("outer_corner_branch_search_scale must be non-negative.");
  }
  if (config_.outer_corner_branch_search_min <= 0 ||
      config_.outer_corner_branch_search_max < config_.outer_corner_branch_search_min) {
    throw std::runtime_error("outer_corner_branch_search_min/max must define a positive valid range.");
  }
  if (config_.outer_corner_min_direction_score < 0.0 ||
      config_.outer_corner_min_direction_score > 1.0) {
    throw std::runtime_error("outer_corner_min_direction_score must be in [0, 1].");
  }
  if (config_.outer_corner_min_layout_score < 0.0 ||
      config_.outer_corner_min_layout_score > 1.0) {
    throw std::runtime_error("outer_corner_min_layout_score must be in [0, 1].");
  }

  detector_ = std::make_unique<AprilTags::TagDetector>(AprilTags::tagCodes36h11, 2);
}

MultiScaleOuterTagDetector::~MultiScaleOuterTagDetector() = default;

MultiScaleOuterTagDetectorConfig MultiScaleOuterTagDetector::LoadConfig(const std::string& yaml_path) {
  return ParseConfig(yaml_path);
}

OuterTagDetectionResult MultiScaleOuterTagDetector::Detect(const cv::Mat& image) const {
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  const cv::Mat gray_original = ToGray(image);
  const std::vector<int> scale_list = BuildScaleList(gray_original.size(), config_);

  OuterTagDetectionResult result;
  result.board_id = config_.tag_id;
  result.failure_reason = OuterTagFailureReason::NoDetectionsAtAll;
  result.failure_reason_text = ToString(result.failure_reason);

  bool saw_any_detection = false;
  bool saw_matching_tag_id = false;
  bool saw_border_rejection = false;
  bool saw_non_border_matching_rejection = false;
  bool saw_refinement_failure = false;
  bool saw_quality_failure = false;
  std::vector<ScaleCandidate> coarse_candidates;

  for (const int target_longest_side : scale_list) {
    OuterTagScaleDebugInfo debug;
    debug.target_longest_side = target_longest_side;
    debug.attempted = true;
    debug.scaled_size = MakeScaledSize(gray_original.size(), target_longest_side);
    debug.scale_factor =
        static_cast<double>(std::max(debug.scaled_size.width, debug.scaled_size.height)) /
        static_cast<double>(std::max(gray_original.cols, gray_original.rows));

    cv::Mat scaled_gray;
    if (debug.scaled_size == gray_original.size()) {
      scaled_gray = gray_original;
    } else {
      cv::resize(gray_original, scaled_gray, debug.scaled_size, 0.0, 0.0, cv::INTER_AREA);
    }
    scaled_gray = MaybeBlur(scaled_gray, config_);

    std::vector<AprilTags::TagDetection> detections = detector_->extractTags(scaled_gray);
    debug.raw_detection_count = static_cast<int>(detections.size());
    saw_any_detection = saw_any_detection || !detections.empty();

    ScaleCandidate best_candidate_for_scale;
    bool has_best_candidate_for_scale = false;
    std::vector<std::string> rejection_reasons;

    for (const AprilTags::TagDetection& detection : detections) {
      if (detection.id != config_.tag_id) {
        continue;
      }

      ++debug.matching_tag_count;
      saw_matching_tag_id = true;

      if (!detection.good) {
        rejection_reasons.push_back("matched tag id but detection.good=false");
        saw_non_border_matching_rejection = true;
        continue;
      }

      ScaleCandidate candidate;
      candidate.target_longest_side = target_longest_side;
      candidate.scale_factor = debug.scale_factor;
      candidate.scaled_size = debug.scaled_size;
      candidate.detection = detection;
      for (int index = 0; index < 4; ++index) {
        candidate.scaled_corners[static_cast<std::size_t>(index)] =
            cv::Point2f(detection.p[index].first, detection.p[index].second);
      }

      if (!PassesBorderCheck(candidate.scaled_corners, scaled_gray.size(), config_.min_border_distance)) {
        rejection_reasons.push_back("matched tag id but rejected by scaled-image border distance");
        saw_border_rejection = true;
        continue;
      }

      candidate.scaled_area = ComputeQuadArea(candidate.scaled_corners);
      const std::pair<double, double> edge_range = ComputeEdgeRange(candidate.scaled_corners);
      candidate.min_edge = edge_range.first;
      candidate.max_edge = edge_range.second;
      candidate.shape_quality =
          candidate.max_edge > 1e-6 ? ClampUnit(candidate.min_edge / candidate.max_edge) : 0.0;

      if (candidate.scaled_area < kMinQuadAreaPixels || candidate.min_edge < kMinQuadEdgePixels ||
          candidate.shape_quality <= 0.10) {
        rejection_reasons.push_back("matched tag id but quad geometry is unstable");
        saw_non_border_matching_rejection = true;
        continue;
      }

      ++debug.accepted_candidate_count;
      if (!has_best_candidate_for_scale || IsCandidateBetter(candidate, best_candidate_for_scale)) {
        best_candidate_for_scale = candidate;
        has_best_candidate_for_scale = true;
      }
    }

    if (!has_best_candidate_for_scale && !rejection_reasons.empty()) {
      debug.rejection_summary = JoinReasons(rejection_reasons);
    }

    if (has_best_candidate_for_scale) {
      coarse_candidates.push_back(best_candidate_for_scale);
      result.successful_scale_longest_sides.push_back(target_longest_side);
    }

    result.scale_debug.push_back(debug);
  }

  if (coarse_candidates.empty()) {
    if (!saw_any_detection) {
      result.failure_reason = OuterTagFailureReason::NoDetectionsAtAll;
    } else if (!saw_matching_tag_id) {
      result.failure_reason = OuterTagFailureReason::DetectionsExistButNoMatchingTagId;
    } else if (saw_border_rejection && !saw_non_border_matching_rejection) {
      result.failure_reason = OuterTagFailureReason::MatchingTagIdButRejectedByBorder;
    } else {
      result.failure_reason = OuterTagFailureReason::MatchingTagIdButAllScalesUnstable;
    }
    result.failure_reason_text = ToString(result.failure_reason);
    return result;
  }

  std::sort(coarse_candidates.begin(), coarse_candidates.end(), IsCandidateBetter);

  const double original_scale_x = static_cast<double>(gray_original.cols);
  const double original_scale_y = static_cast<double>(gray_original.rows);
  std::vector<RefinedCandidate> refined_candidates;
  bool has_best_failed_candidate = false;
  RefinedCandidate best_failed_candidate;

  for (ScaleCandidate& candidate : coarse_candidates) {
    RefinedCandidate refined_candidate;
    refined_candidate.coarse = candidate;

    const double scale_x =
        original_scale_x / static_cast<double>(std::max(1, candidate.scaled_size.width));
    const double scale_y =
        original_scale_y / static_cast<double>(std::max(1, candidate.scaled_size.height));

    std::array<bool, 4> method_valid{{false, false, false, false}};
    std::array<double, 4> method_quality{{0.0, 0.0, 0.0, 0.0}};

    for (int index = 0; index < 4; ++index) {
      const cv::Point2f coarse_original(
          static_cast<float>(candidate.scaled_corners[static_cast<std::size_t>(index)].x * scale_x),
          static_cast<float>(candidate.scaled_corners[static_cast<std::size_t>(index)].y * scale_y));
      refined_candidate.coarse_original[static_cast<std::size_t>(index)] = coarse_original;
      refined_candidate.refined_original[static_cast<std::size_t>(index)] = coarse_original;
    }

    bool coarse_original_is_valid =
        PassesBorderCheck(refined_candidate.coarse_original, gray_original.size(), config_.min_border_distance);
    if (!coarse_original_is_valid) {
      saw_border_rejection = true;
      continue;
    }

    std::array<cv::Point2f, 4> verification_seed_corners = refined_candidate.coarse_original;
    for (int index = 0; index < 4; ++index) {
      const OuterCornerLocalVerificationResult verification =
          VerifyOuterCornerLocalStructure(gray_original, refined_candidate.coarse_original, index, config_);
      refined_candidate.verification_debug[static_cast<std::size_t>(index)] =
          BuildVerificationDebugInfo(index,
                                     refined_candidate.coarse_original[static_cast<std::size_t>(index)],
                                     verification);
      const bool verification_passed =
          refined_candidate.verification_debug[static_cast<std::size_t>(index)].verification_passed;
      const cv::Point2f verification_seed =
          verification_passed ? verification.verified_corner
                              : refined_candidate.coarse_original[static_cast<std::size_t>(index)];
      refined_candidate.refined_original[static_cast<std::size_t>(index)] = verification_seed;
      method_valid[static_cast<std::size_t>(index)] = verification_passed;
      method_quality[static_cast<std::size_t>(index)] =
          verification_passed ? verification.verification_quality : 0.0;
      verification_seed_corners[static_cast<std::size_t>(index)] = verification_seed;
      refined_candidate.verification_debug[static_cast<std::size_t>(index)].subpix_corner =
          verification_seed;
    }

    std::array<cv::Point2f, 4> subpix_seed_corners = verification_seed_corners;
    if (config_.do_outer_subpix_refinement) {
      for (int index = 0; index < 4; ++index) {
        if (!method_valid[static_cast<std::size_t>(index)]) {
          continue;
        }
        std::vector<cv::Point2f> point_seed{verification_seed_corners[static_cast<std::size_t>(index)]};
        cv::cornerSubPix(
            gray_original, point_seed,
            cv::Size(kOuterRefinementWindowRadius, kOuterRefinementWindowRadius),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
        subpix_seed_corners[static_cast<std::size_t>(index)] = point_seed.front();
        refined_candidate.refined_original[static_cast<std::size_t>(index)] = point_seed.front();
        refined_candidate.verification_debug[static_cast<std::size_t>(index)].subpix_corner = point_seed.front();
        refined_candidate.verification_debug[static_cast<std::size_t>(index)].subpix_applied = true;
      }
    } else {
      for (int index = 0; index < 4; ++index) {
        refined_candidate.verification_debug[static_cast<std::size_t>(index)].subpix_corner =
            verification_seed_corners[static_cast<std::size_t>(index)];
      }
    }

    for (int index = 0; index < 4; ++index) {
      if (!method_valid[static_cast<std::size_t>(index)]) {
        continue;
      }

      const CornerLineRefinement line_refinement =
          RefineCornerByLineIntersection(gray_original, subpix_seed_corners, index);
      const cv::Point2f coarse_corner = refined_candidate.coarse_original[static_cast<std::size_t>(index)];
      const cv::Point2f delta = line_refinement.refined_corner - coarse_corner;
      const double line_jump = std::hypot(delta.x, delta.y);
      const double line_jump_limit = std::max(config_.max_outer_refine_displacement, 8.0);
      const bool line_inside =
          line_refinement.refined_corner.x >= config_.min_border_distance &&
          line_refinement.refined_corner.x <= static_cast<float>(gray_original.cols) - config_.min_border_distance &&
          line_refinement.refined_corner.y >= config_.min_border_distance &&
          line_refinement.refined_corner.y <= static_cast<float>(gray_original.rows) - config_.min_border_distance;
      const cv::Point2f subpix_delta =
          line_refinement.refined_corner - subpix_seed_corners[static_cast<std::size_t>(index)];
      const double line_subpix_gap = std::hypot(subpix_delta.x, subpix_delta.y);

      if (line_refinement.success &&
          line_refinement.quality >= kOuterLineMinQuality &&
          line_jump <= line_jump_limit &&
          line_inside &&
          line_subpix_gap <= kOuterLineSubpixAgreementPixels) {
        refined_candidate.refined_original[static_cast<std::size_t>(index)] = line_refinement.refined_corner;
        method_quality[static_cast<std::size_t>(index)] = std::min(
            std::max(method_quality[static_cast<std::size_t>(index)], line_refinement.quality),
            1.0);
      }
    }

    const bool refined_inside =
        PassesBorderCheck(refined_candidate.refined_original, gray_original.size(), config_.min_border_distance);
    const double displacement_gate = std::max(config_.max_outer_refine_displacement * 2.0, 12.0);
    std::array<bool, 4> displacement_valid{{false, false, false, false}};
    std::array<double, 4> displacement_quality{{0.0, 0.0, 0.0, 0.0}};
    ComputeRefineQuality(refined_candidate.coarse_original, refined_candidate.refined_original,
                         displacement_gate, &displacement_valid, &displacement_quality);

    refined_candidate.refine_quality = 1.0;
    for (int index = 0; index < 4; ++index) {
      refined_candidate.refined_valid[static_cast<std::size_t>(index)] =
          method_valid[static_cast<std::size_t>(index)] && displacement_valid[static_cast<std::size_t>(index)];
      const double corner_quality =
          std::min(method_quality[static_cast<std::size_t>(index)],
                   displacement_quality[static_cast<std::size_t>(index)]);
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

    if (!refined_inside ||
        std::any_of(refined_candidate.refined_valid.begin(), refined_candidate.refined_valid.end(),
                    [](bool valid) { return !valid; })) {
      saw_refinement_failure = true;
      if (!has_best_failed_candidate || IsRefinedCandidateBetter(refined_candidate, best_failed_candidate)) {
        best_failed_candidate = refined_candidate;
        has_best_failed_candidate = true;
      }
      continue;
    }

    if (refined_candidate.quality < config_.min_detection_quality) {
      saw_quality_failure = true;
      if (!has_best_failed_candidate || IsRefinedCandidateBetter(refined_candidate, best_failed_candidate)) {
        best_failed_candidate = refined_candidate;
        has_best_failed_candidate = true;
      }
      continue;
    }

    refined_candidates.push_back(refined_candidate);
    auto debug_it =
        std::find_if(result.scale_debug.begin(), result.scale_debug.end(),
                     [&](const OuterTagScaleDebugInfo& info) {
                       return info.target_longest_side == refined_candidate.coarse.target_longest_side;
                     });
    if (debug_it != result.scale_debug.end()) {
      ++debug_it->refined_success_count;
    }
  }

  auto fill_result_from_candidate = [&](const RefinedCandidate& chosen_candidate) {
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

  if (refined_candidates.empty()) {
    if (has_best_failed_candidate) {
      fill_result_from_candidate(best_failed_candidate);
    } else {
      const ScaleCandidate& fallback_candidate = coarse_candidates.front();
      result.detected_tag_id = fallback_candidate.detection.id;
      result.chosen_scale_longest_side = fallback_candidate.target_longest_side;
      result.chosen_scale_factor = fallback_candidate.scale_factor;
      result.hamming = fallback_candidate.detection.hammingDistance;
      result.good = fallback_candidate.detection.good;
      for (int index = 0; index < 4; ++index) {
        const cv::Point2f coarse_original(
            static_cast<float>(fallback_candidate.scaled_corners[static_cast<std::size_t>(index)].x * original_scale_x /
                               static_cast<double>(std::max(1, fallback_candidate.scaled_size.width))),
            static_cast<float>(fallback_candidate.scaled_corners[static_cast<std::size_t>(index)].y * original_scale_y /
                               static_cast<double>(std::max(1, fallback_candidate.scaled_size.height))));
        result.coarse_corners_scaled_image[static_cast<std::size_t>(index)] =
            ToEigen(fallback_candidate.scaled_corners[static_cast<std::size_t>(index)]);
        result.coarse_corners_original_image[static_cast<std::size_t>(index)] = ToEigen(coarse_original);
      }
    }
    if (saw_refinement_failure) {
      result.failure_reason = OuterTagFailureReason::MatchingTagIdButRefinementFailed;
    } else if (saw_quality_failure || saw_border_rejection) {
      result.failure_reason = OuterTagFailureReason::MatchingTagIdButAllScalesUnstable;
    } else {
      result.failure_reason = OuterTagFailureReason::MatchingTagIdButAllScalesUnstable;
    }
    result.failure_reason_text = ToString(result.failure_reason);
    return result;
  }

  std::sort(refined_candidates.begin(), refined_candidates.end(), IsRefinedCandidateBetter);
  const RefinedCandidate& best = refined_candidates.front();

  result.success = true;
  result.failure_reason = OuterTagFailureReason::None;
  result.failure_reason_text = ToString(result.failure_reason);
  fill_result_from_candidate(best);

  return result;
}

void MultiScaleOuterTagDetector::DrawDetection(const OuterTagDetectionResult& detection,
                                               cv::Mat* output_image) const {
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
  const int line_thickness = std::max(1, static_cast<int>(std::lround(render_scale)));
  const double label_scale = std::max(0.9, 0.7 * render_scale);

  if (has_coarse) {
    for (int index = 0; index < 4; ++index) {
      const cv::Point2f coarse = ToPoint(detection.coarse_corners_original_image[static_cast<std::size_t>(index)]);
      cv::circle(*output_image, coarse, coarse_radius, cv::Scalar(0, 165, 255), line_thickness);
    }
  }

  const std::array<cv::Scalar, 2> branch_colors{
      cv::Scalar(255, 180, 0),
      cv::Scalar(0, 220, 255),
  };
  for (int index = 0; index < 4; ++index) {
    const OuterCornerVerificationDebugInfo& verification =
        detection.corner_verification_debug[static_cast<std::size_t>(index)];
    if (verification.corner_index < 0) {
      continue;
    }

    const cv::Scalar roi_color =
        verification.verification_passed ? cv::Scalar(0, 180, 0) : cv::Scalar(0, 0, 255);
    if (verification.verification_roi.width > 0 && verification.verification_roi.height > 0) {
      cv::rectangle(*output_image, verification.verification_roi, roi_color, line_thickness);
    }

    const cv::Point2f coarse = verification.coarse_corner;
    const cv::Point2f verified = verification.verified_corner;
    const cv::Point2f subpix = verification.subpix_corner;
    const cv::Point2f anchor = verified;
    const float arrow_length = static_cast<float>(18.0 * render_scale);
    if (verification.candidate_radius > 0) {
      cv::circle(*output_image, coarse, verification.candidate_radius,
                 cv::Scalar(0, 110, 200), line_thickness);
    }
    if (verification.branch_search_radius > 0) {
      cv::circle(*output_image, verified, verification.branch_search_radius,
                 cv::Scalar(200, 255, 120), line_thickness);
    }
    if (Norm(verification.prev_edge_direction) > 1e-6) {
      cv::arrowedLine(*output_image, anchor,
                      anchor + verification.prev_edge_direction * arrow_length,
                      branch_colors[0], line_thickness, cv::LINE_AA, 0, 0.25);
    }
    if (Norm(verification.next_edge_direction) > 1e-6) {
      cv::arrowedLine(*output_image, anchor,
                      anchor + verification.next_edge_direction * arrow_length,
                      branch_colors[1], line_thickness, cv::LINE_AA, 0, 0.25);
    }

    for (const cv::Point2f& point : verification.prev_branch_points) {
      cv::circle(*output_image, point, std::max(2, line_thickness + 1), branch_colors[0], -1);
    }
    for (const cv::Point2f& point : verification.next_branch_points) {
      cv::circle(*output_image, point, std::max(2, line_thickness + 1), branch_colors[1], -1);
    }

    cv::line(*output_image, coarse, verified, cv::Scalar(120, 220, 120), line_thickness, cv::LINE_AA);
    cv::line(*output_image, verified, subpix, cv::Scalar(255, 220, 0), line_thickness, cv::LINE_AA);
    cv::circle(*output_image, coarse, coarse_radius, cv::Scalar(0, 165, 255), line_thickness);
    cv::circle(*output_image, verified, verified_radius,
               verification.verification_passed ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
               line_thickness);
    cv::drawMarker(*output_image, subpix, cv::Scalar(255, 255, 0),
                   cv::MARKER_CROSS, subpix_radius * 3, line_thickness);

    std::ostringstream label;
    label << (verification.verification_passed ? "pass" : "fail")
          << " dir=" << std::fixed << std::setprecision(2) << verification.direction_consistency_score
          << " lay=" << verification.local_layout_score
          << " Q=" << verification.verification_quality;
    if (!verification.verification_passed && !verification.failure_reason.empty()) {
      label << " " << verification.failure_reason;
    }
    cv::putText(*output_image, label.str(),
                verified + cv::Point2f(static_cast<float>(6.0 * render_scale),
                                       static_cast<float>(14.0 * render_scale)),
                cv::FONT_HERSHEY_PLAIN, label_scale,
                verification.verification_passed ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                line_thickness);
    std::ostringstream adaptive_label;
    adaptive_label << "s=" << std::fixed << std::setprecision(1) << verification.local_scale
                   << " roi=" << verification.verification_roi_radius
                   << " cand=" << verification.candidate_radius
                   << " br=" << verification.branch_search_radius;
    cv::putText(*output_image, adaptive_label.str(),
                verified + cv::Point2f(static_cast<float>(6.0 * render_scale),
                                       static_cast<float>(30.0 * render_scale)),
                cv::FONT_HERSHEY_PLAIN, std::max(0.8, 0.6 * render_scale),
                cv::Scalar(200, 255, 200), line_thickness);
    cv::putText(*output_image, "C",
                coarse + cv::Point2f(static_cast<float>(-10.0 * render_scale),
                                     static_cast<float>(-8.0 * render_scale)),
                cv::FONT_HERSHEY_PLAIN, label_scale, cv::Scalar(0, 165, 255), line_thickness);
    cv::putText(*output_image, "V",
                verified + cv::Point2f(static_cast<float>(-10.0 * render_scale),
                                       static_cast<float>(-8.0 * render_scale)),
                cv::FONT_HERSHEY_PLAIN, label_scale,
                verification.verification_passed ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                line_thickness);
    cv::putText(*output_image, "S",
                subpix + cv::Point2f(static_cast<float>(6.0 * render_scale),
                                     static_cast<float>(-8.0 * render_scale)),
                cv::FONT_HERSHEY_PLAIN, label_scale, cv::Scalar(255, 255, 0), line_thickness);
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
  }

  const std::string headline =
      detection.success ? "status: multi-scale outer tag detection success"
                        : "status: multi-scale outer tag detection failed";
  cv::putText(*output_image, headline, cv::Point(20, 28), cv::FONT_HERSHEY_SIMPLEX,
              std::max(0.6, 0.45 * render_scale), cv::Scalar(0, 255, 255),
              std::max(2, line_thickness));

  std::ostringstream summary;
  summary << "tagId=" << config_.tag_id << " chosen_scale=" << detection.chosen_scale_longest_side
          << " hamming=" << detection.hamming << " quality=" << std::fixed << std::setprecision(2)
          << detection.quality;
  cv::putText(*output_image, summary.str(), cv::Point(20, 56), cv::FONT_HERSHEY_SIMPLEX,
              std::max(0.55, 0.4 * render_scale), cv::Scalar(255, 255, 0),
              std::max(1, line_thickness));

  std::ostringstream failure;
  failure << "failure_reason=" << detection.failure_reason_text;
  cv::putText(*output_image, failure.str(), cv::Point(20, 84), cv::FONT_HERSHEY_SIMPLEX,
              std::max(0.55, 0.4 * render_scale), cv::Scalar(0, 200, 255),
              std::max(1, line_thickness));
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
