#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <boost/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

struct TemplateScore {
  double template_quality = 0.0;
  double gradient_quality = 0.0;
};

struct ImageEvidenceScore {
  TemplateScore point_score;
  TemplateScore best_score;
  cv::Point2f best_point{};
  double centering_quality = 0.0;
  double final_quality = 0.0;
};

struct VirtualPatchContext {
  cv::Mat patch;
  cv::Size patch_size;
  cv::Matx33d target_to_camera_rotation = cv::Matx33d::eye();
  Eigen::Vector3d target_to_camera_translation = Eigen::Vector3d::Zero();
  cv::Matx33d camera_to_virtual_rotation = cv::Matx33d::eye();
  cv::Matx33d virtual_to_camera_rotation = cv::Matx33d::eye();
  Eigen::Vector3d plane_normal_camera = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d plane_point_camera = Eigen::Vector3d::Zero();
  std::array<cv::Point2f, 4> outer_patch_corners{};
  double fu = 0.0;
  double fv = 0.0;
  double cu = 0.0;
  double cv = 0.0;
};

struct SphereLatticeFrame {
  cv::Point2f predicted_image{};
  cv::Point2f module_u_axis{};
  cv::Point2f module_v_axis{};
  Eigen::Vector3d predicted_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangent_u = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangent_v = Eigen::Vector3d::Zero();
  double theta_u = 0.0;
  double theta_v = 0.0;
  double theta_local = 0.0;
  double search_radius = 0.0;
  double local_module_scale = 0.0;
};

struct SphereSeedCandidate {
  bool valid = false;
  double alpha = 0.0;
  double beta = 0.0;
  double template_quality = 0.0;
  double gradient_quality = 0.0;
  double prior_quality = 0.0;
  double peak_quality = 0.0;
  double raw_quality = 0.0;
  double final_quality = 0.0;
  cv::Point2f image_point{};
  Eigen::Vector3d ray = Eigen::Vector3d::Zero();
};

struct PatchSeedFrame {
  cv::Point2f predicted_patch{};
  cv::Point2f predicted_image{};
  cv::Point2f module_u_axis{};
  cv::Point2f module_v_axis{};
  cv::Point2f unit_u{};
  cv::Point2f unit_v{};
  double local_module_scale = 0.0;
  double search_radius = 0.0;
};

struct PatchSeedCandidate {
  bool valid = false;
  double alpha = 0.0;
  double beta = 0.0;
  double template_quality = 0.0;
  double gradient_quality = 0.0;
  double prior_quality = 0.0;
  double peak_quality = 0.0;
  double raw_quality = 0.0;
  double final_quality = 0.0;
  cv::Point2f patch_point{};
  cv::Point2f image_point{};
};

struct RayRefinementEvaluation {
  bool valid = false;
  cv::Point2f image_point{};
  Eigen::Vector3d ray = Eigen::Vector3d::Zero();
  double template_quality = 0.0;
  double gradient_quality = 0.0;
  double edge_quality = 0.0;
  double photometric_quality = 0.0;
  double final_quality = 0.0;
};

struct RayRefinementResult {
  bool valid = false;
  bool converged = false;
  int iterations = 0;
  double trust_radius = 0.0;
  cv::Point2f refined_image{};
  Eigen::Vector3d refined_ray = Eigen::Vector3d::Zero();
  double template_quality = 0.0;
  double gradient_quality = 0.0;
  double edge_quality = 0.0;
  double photometric_quality = 0.0;
  double final_quality = 0.0;
};

struct BoardBoundaryEdgeModel {
  bool valid = false;
  std::vector<cv::Point2f> support_points;
  std::vector<Eigen::Vector3d> support_rays;
  Eigen::Vector3d plane_normal = Eigen::Vector3d::Zero();
  Eigen::Vector3d start_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d end_ray = Eigen::Vector3d::Zero();
  double rms_residual = std::numeric_limits<double>::infinity();
};

struct BoardSphereBoundaryModel {
  bool valid = false;
  std::array<Eigen::Vector3d, 4> outer_corner_rays{};
  std::array<BoardBoundaryEdgeModel, 4> edges;
};

struct BorderConditionedSeed {
  bool valid = false;
  cv::Point2f image_point{};
  Eigen::Vector3d ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d top_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d bottom_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d left_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_ray = Eigen::Vector3d::Zero();
};

constexpr double kSphereLatticeSearchRadiusScale = 0.5;
constexpr double kSphereLatticeSearchRadiusMin = 1e-3;
constexpr int kSphereLatticeCoarseGridSize = 9;
constexpr int kSphereLatticeFineGridSize = 5;
constexpr double kSphereSeedSampleRadiusScale = 0.55;
constexpr double kSphereSeedPriorInnerRatio = 0.35;
constexpr double kSphereSeedRawWeight = 0.65;
constexpr double kSphereSeedPeakWeight = 0.25;
constexpr double kSphereSeedPriorWeight = 0.10;
constexpr double kBoundaryProbeOffsetScale = 0.18;
constexpr double kBoundaryTangentOffsetScale = 0.30;
constexpr double kBoundaryProbeRadiusScale = 0.40;
constexpr double kRayRefineEdgeWeight = 0.60;
constexpr double kRayRefinePhotoWeight = 0.40;
constexpr double kRayRefineTrustRadiusScale = 0.28;
constexpr double kRayRefineTrustRadiusMin = 2e-4;
constexpr double kRayRefineDiffStepMin = 1e-4;
constexpr double kRayRefineDiffStepScale = 0.20;
constexpr double kRayRefineMinStep = 1e-5;
constexpr int kRayRefineMaxIterations = 12;
constexpr std::array<double, 4> kRayRefineLineSearchScales{{1.0, 0.5, 0.25, 0.125}};
constexpr double kPhotometricQuadrantOffsetScale = 0.28;
constexpr double kPhotometricProbeRadiusScale = 0.30;
constexpr double kPatchSeedSearchRadiusScale = 0.60;
constexpr double kPatchSeedSearchRadiusMinPx = 2.0;
constexpr int kPatchSeedCoarseGridSize = 9;
constexpr int kPatchSeedFineGridSize = 5;
constexpr int kBorderConditionedMinSupportPoints = 4;
constexpr double kBorderConditionedPlaneResidualThreshold = 0.035;
constexpr int kBorderBoundaryCurveSampleCount = 33;
constexpr double kBorderConditionedSearchRadiusGain = 0.75;
constexpr double kBorderConditionedSearchRadiusMaxScale = 3.0;

bool NormalizeRay(Eigen::Vector3d* ray);
Eigen::Vector3d ProjectOntoTangentPlane(const Eigen::Vector3d& vector,
                                        const Eigen::Vector3d& ray_anchor);
bool BuildLocalSphereOffsetRay(const Eigen::Vector3d& anchor_ray,
                               const Eigen::Vector3d& tangent_u,
                               const Eigen::Vector3d& tangent_v,
                               double du,
                               double dv,
                               Eigen::Vector3d* ray);
double ComputeLocalModuleSizePx(const cv::Point2f& module_u_axis,
                                const cv::Point2f& module_v_axis);
int ComputeSphereSearchRadiusOverlayPx(double local_module_scale);
int ComputeRayRefineTrustRadiusOverlayPx(double local_module_scale,
                                         int search_radius_px);
void ProjectOffsetToDisk(double radius, double* du, double* dv);
bool ProjectRayToImage(const DoubleSphereCameraModel& camera,
                       const Eigen::Vector3d& ray,
                       cv::Point2f* image_point);
bool UnprojectImagePointToRay(const DoubleSphereCameraModel& camera,
                              const cv::Point2f& image_point,
                              Eigen::Vector3d* ray);
bool IntersectVirtualPatchPixelWithTargetPlane(const VirtualPatchContext& context,
                                               const cv::Point2f& patch_point,
                                               Eigen::Vector3d* point_camera);

TemplateScore ComputeBoundaryAlignmentScoreAtPoint(const cv::Mat& gray,
                                                   const CanonicalCorner& corner_info,
                                                   const cv::Point2f& image_point,
                                                   const cv::Point2f& module_u_axis,
                                                   const cv::Point2f& module_v_axis,
                                                   double min_template_contrast,
                                                   double sample_radius_scale);

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

std::string Lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
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

std::vector<double> ParseDoubleList(const std::string& key, const std::string& value) {
  const std::string trimmed = Trim(value);
  if (trimmed.size() < 2 || trimmed.front() != '[' || trimmed.back() != ']') {
    throw std::runtime_error("Expected list syntax for field '" + key + "', got '" + value + "'.");
  }

  const std::string inner = Trim(trimmed.substr(1, trimmed.size() - 2));
  std::vector<double> values;
  if (inner.empty()) {
    return values;
  }

  std::stringstream stream(inner);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const std::string cleaned = Trim(token);
    if (!cleaned.empty()) {
      values.push_back(ParseDouble(key, cleaned));
    }
  }
  return values;
}

std::vector<int> ParseIntList(const std::string& key, const std::string& value) {
  const std::vector<double> parsed = ParseDoubleList(key, value);
  std::vector<int> values;
  values.reserve(parsed.size());
  for (double number : parsed) {
    values.push_back(static_cast<int>(std::lround(number)));
  }
  return values;
}

InternalProjectionMode ParseProjectionMode(const std::string& value) {
  const std::string lowered = Lowercase(value);
  if (lowered == "homography") {
    return InternalProjectionMode::Homography;
  }
  if (lowered == "virtual_pinhole_patch" || lowered == "virtual-pinhole-patch") {
    return InternalProjectionMode::VirtualPinholePatch;
  }
  if (lowered == "virtual_pinhole_image_subpix" ||
      lowered == "virtual-pinhole-image-subpix") {
    return InternalProjectionMode::VirtualPinholeImageSubpix;
  }
  if (lowered == "virtual_pinhole_patch_boundary_seed" ||
      lowered == "virtual-pinhole-patch-boundary-seed" ||
      lowered == "virtual_pinhole_patch_edge_seed" ||
      lowered == "virtual-pinhole-patch-edge-seed") {
    return InternalProjectionMode::VirtualPinholePatchBoundarySeed;
  }
  if (lowered == "sphere_lattice" || lowered == "sphere-lattice") {
    return InternalProjectionMode::SphereLattice;
  }
  if (lowered == "sphere_border_lattice" || lowered == "sphere-border-lattice") {
    return InternalProjectionMode::SphereBorderLattice;
  }
  if (lowered == "sphere_ray_refine" || lowered == "sphere-ray-refine") {
    return InternalProjectionMode::SphereRayRefine;
  }
  throw std::runtime_error("Unsupported internal_projection_mode '" + value + "'.");
}

bool UsesSphereSeedPipeline(InternalProjectionMode mode) {
  return mode == InternalProjectionMode::SphereLattice ||
         mode == InternalProjectionMode::SphereBorderLattice ||
         mode == InternalProjectionMode::SphereRayRefine;
}

bool UsesBorderConditionedSphereSeed(InternalProjectionMode mode) {
  return mode == InternalProjectionMode::SphereBorderLattice;
}

std::string ResolvePath(const std::string& path, const std::string& reference_yaml_path) {
  const boost::filesystem::path input_path(path);
  if (input_path.is_absolute()) {
    return input_path.lexically_normal().string();
  }

  const boost::filesystem::path reference_dir =
      boost::filesystem::absolute(boost::filesystem::path(reference_yaml_path)).parent_path();
  return (reference_dir / input_path).lexically_normal().string();
}

void ApplyCameraField(IntermediateCameraConfig* config,
                      const std::string& key,
                      const std::string& value,
                      const std::string& reference_yaml_path) {
  if (config == nullptr) {
    throw std::runtime_error("Camera config output pointer must not be null.");
  }

  if (key == "camera_yaml" || key == "intermediate_camera_yaml") {
    config->camera_yaml = ResolvePath(value, reference_yaml_path);
  } else if (key == "camera_model") {
    config->camera_model = Lowercase(value);
  } else if (key == "distortion_model") {
    config->distortion_model = Lowercase(value);
  } else if (key == "intrinsics") {
    config->intrinsics = ParseDoubleList(key, value);
  } else if (key == "distortion_coeffs") {
    config->distortion_coeffs = ParseDoubleList(key, value);
  } else if (key == "resolution") {
    config->resolution = ParseIntList(key, value);
  }
}

IntermediateCameraConfig LoadExternalCameraConfig(const std::string& camera_yaml_path) {
  std::ifstream stream(camera_yaml_path);
  if (!stream.is_open()) {
    throw std::runtime_error("Could not open camera config file: " + camera_yaml_path);
  }

  IntermediateCameraConfig config;
  config.camera_yaml = camera_yaml_path;

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
    ApplyCameraField(&config, key, value, camera_yaml_path);
  }

  return config;
}

IntermediateCameraConfig MakeCoarseInitialCameraConfig(
    const cv::Size& resolution,
    const ApriltagInternalConfig& config) {
  IntermediateCameraConfig camera_config;
  camera_config.camera_model = "ds";
  camera_config.distortion_model = "none";
  camera_config.distortion_coeffs.clear();
  camera_config.resolution = {resolution.width, resolution.height};
  camera_config.intrinsics = {
      config.sphere_lattice_init_xi,
      config.sphere_lattice_init_alpha,
      config.sphere_lattice_init_fu_scale * static_cast<double>(resolution.width),
      config.sphere_lattice_init_fv_scale * static_cast<double>(resolution.height),
      0.5 * static_cast<double>(resolution.width) + config.sphere_lattice_init_cu_offset,
      0.5 * static_cast<double>(resolution.height) + config.sphere_lattice_init_cv_offset,
  };
  return camera_config;
}

std::vector<int> NormalizeBoardIds(const std::vector<int>& configured_ids, int fallback_tag_id) {
  std::vector<int> normalized;
  const auto append_if_valid = [&](int board_id) {
    if (board_id < 0) {
      return;
    }
    if (std::find(normalized.begin(), normalized.end(), board_id) == normalized.end()) {
      normalized.push_back(board_id);
    }
  };

  for (int board_id : configured_ids) {
    append_if_valid(board_id);
  }
  if (normalized.empty()) {
    append_if_valid(fallback_tag_id);
  }
  if (normalized.empty()) {
    throw std::runtime_error("At least one non-negative tag id is required.");
  }
  return normalized;
}

ApriltagInternalConfig MakeBoardSpecificConfig(const ApriltagInternalConfig& base_config,
                                               int board_id) {
  ApriltagInternalConfig config = base_config;
  config.tag_id = board_id;
  config.tag_ids.clear();
  config.outer_detector_config.tag_ids.clear();
  config.outer_detector_config.tag_id = board_id;
  return config;
}

ApriltagInternalConfig ParseApriltagInternalConfig(const std::string& yaml_path) {
  std::ifstream stream(yaml_path);
  if (!stream.is_open()) {
    throw std::runtime_error("Could not open config file: " + yaml_path);
  }

  ApriltagInternalConfig config;
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

    if (key == "target_type") {
      config.target_type = value;
    } else if (key == "tagFamily" || key == "tag_family") {
      config.tag_family = value;
    } else if (key == "tagId" || key == "tag_id") {
      config.tag_id = ParseInt(key, value);
    } else if (key == "tagIds" || key == "tag_ids") {
      config.tag_ids = ParseIntList(key, value);
    } else if (key == "tagSize" || key == "tag_size") {
      config.tag_size = ParseDouble(key, value);
    } else if (key == "blackBorderBits" || key == "black_border_bits") {
      config.black_border_bits = ParseInt(key, value);
    } else if (key == "minVisiblePoints" || key == "min_visible_points") {
      config.min_visible_points = ParseInt(key, value);
    } else if (key == "canonicalPixelsPerModule" || key == "canonical_pixels_per_module") {
      config.canonical_pixels_per_module = ParseInt(key, value);
    } else if (key == "refinementWindowRadius" || key == "refinement_window_radius") {
      config.refinement_window_radius = ParseInt(key, value);
    } else if (key == "internalSubpixWindowScale" || key == "internal_subpix_window_scale") {
      config.internal_subpix_window_scale = ParseDouble(key, value);
    } else if (key == "internalSubpixWindowMin" || key == "internal_subpix_window_min") {
      config.internal_subpix_window_min = ParseInt(key, value);
    } else if (key == "internalSubpixWindowMax" || key == "internal_subpix_window_max") {
      config.internal_subpix_window_max = ParseInt(key, value);
    } else if (key == "maxSubpixDisplacement2" || key == "max_subpix_displacement2") {
      config.max_subpix_displacement2 = ParseDouble(key, value);
    } else if (key == "internalSubpixDisplacementScale" ||
               key == "internal_subpix_displacement_scale") {
      config.internal_subpix_displacement_scale = ParseDouble(key, value);
    } else if (key == "maxInternalSubpixDisplacement" ||
               key == "max_internal_subpix_displacement") {
      config.max_internal_subpix_displacement = ParseDouble(key, value);
    } else if (key == "enableDebugOutput" || key == "enable_debug_output") {
      config.enable_debug_output =
          Lowercase(value) == "1" || Lowercase(value) == "true" ||
          Lowercase(value) == "yes" || Lowercase(value) == "on";
    } else if (key == "internal_projection_mode") {
      config.internal_projection_mode = ParseProjectionMode(value);
    } else if (key == "sphereLatticeUseInitialCamera" ||
               key == "sphere_lattice_use_initial_camera") {
      config.sphere_lattice_use_initial_camera =
          Lowercase(value) == "1" || Lowercase(value) == "true" ||
          Lowercase(value) == "yes" || Lowercase(value) == "on";
    } else if (key == "outerSphericalUseInitialCamera" ||
               key == "outer_spherical_use_initial_camera") {
      config.outer_spherical_use_initial_camera =
          Lowercase(value) == "1" || Lowercase(value) == "true" ||
          Lowercase(value) == "yes" || Lowercase(value) == "on";
    } else if (key == "enableOuterSphericalRefinement" ||
               key == "enable_outer_spherical_refinement") {
      config.outer_detector_config.enable_outer_spherical_refinement =
          Lowercase(value) == "1" || Lowercase(value) == "true" ||
          Lowercase(value) == "yes" || Lowercase(value) == "on";
    } else if (key == "sphereLatticeEnableSeedSearch" ||
               key == "sphere_lattice_enable_seed_search") {
      config.sphere_lattice_enable_seed_search =
          Lowercase(value) == "1" || Lowercase(value) == "true" ||
          Lowercase(value) == "yes" || Lowercase(value) == "on";
    } else if (key == "sphereLatticeInitXi" || key == "sphere_lattice_init_xi") {
      config.sphere_lattice_init_xi = ParseDouble(key, value);
    } else if (key == "sphereLatticeInitAlpha" || key == "sphere_lattice_init_alpha") {
      config.sphere_lattice_init_alpha = ParseDouble(key, value);
    } else if (key == "sphereLatticeInitFuScale" || key == "sphere_lattice_init_fu_scale") {
      config.sphere_lattice_init_fu_scale = ParseDouble(key, value);
    } else if (key == "sphereLatticeInitFvScale" || key == "sphere_lattice_init_fv_scale") {
      config.sphere_lattice_init_fv_scale = ParseDouble(key, value);
    } else if (key == "sphereLatticeInitCuOffset" || key == "sphere_lattice_init_cu_offset") {
      config.sphere_lattice_init_cu_offset = ParseDouble(key, value);
    } else if (key == "sphereLatticeInitCvOffset" || key == "sphere_lattice_init_cv_offset") {
      config.sphere_lattice_init_cv_offset = ParseDouble(key, value);
    } else if (key == "minBorderDistance" || key == "min_border_distance") {
      config.outer_detector_config.min_border_distance = ParseDouble(key, value);
    } else if (key == "maxScalesToTry" || key == "max_scales_to_try") {
      config.outer_detector_config.max_scales_to_try = ParseInt(key, value);
    } else if (key == "outerLocalContextScale" || key == "outer_local_context_scale") {
      config.outer_detector_config.outer_local_context_scale = ParseDouble(key, value);
    } else if (key == "outerCornerMarkerRatio" || key == "outer_corner_marker_ratio" ||
               key == "tagSpacing" || key == "tag_spacing") {
      config.outer_detector_config.outer_corner_marker_ratio = ParseDouble(key, value);
    } else if (key == "outerSubpixScale" || key == "outer_subpix_scale") {
      config.outer_detector_config.outer_subpix_scale = ParseDouble(key, value);
    } else if (key == "outerRefineGateScale" || key == "outer_refine_gate_scale") {
      config.outer_detector_config.outer_refine_gate_scale = ParseDouble(key, value);
    } else if (key == "outerRefineGateMin" || key == "outer_refine_gate_min") {
      config.outer_detector_config.outer_refine_gate_min = ParseDouble(key, value);
    } else if (key == "scaleCandidates" || key == "scale_candidates") {
      config.outer_detector_config.scale_candidates = ParseIntList(key, value);
    } else if (key == "scaleDivisors" || key == "scale_divisors") {
      config.outer_detector_config.scale_divisors = ParseDoubleList(key, value);
    } else if (key == "doOuterSubpixRefinement" || key == "do_outer_subpix_refinement") {
      config.outer_detector_config.do_outer_subpix_refinement =
          Lowercase(value) == "1" || Lowercase(value) == "true" ||
          Lowercase(value) == "yes" || Lowercase(value) == "on";
    } else if (key == "outerSubpixWindowRadius" || key == "outer_subpix_window_radius") {
      config.outer_detector_config.outer_subpix_window_radius = ParseInt(key, value);
    } else if (key == "outerSubpixWindowScale" || key == "outer_subpix_window_scale") {
      config.outer_detector_config.outer_subpix_window_scale = ParseDouble(key, value);
      config.outer_detector_config.outer_subpix_scale =
          config.outer_detector_config.outer_subpix_window_scale;
    } else if (key == "outerSubpixWindowMin" || key == "outer_subpix_window_min") {
      config.outer_detector_config.outer_subpix_window_min = ParseInt(key, value);
    } else if (key == "outerSubpixWindowMax" || key == "outer_subpix_window_max") {
      config.outer_detector_config.outer_subpix_window_max = ParseInt(key, value);
    } else if (key == "maxOuterRefineDisplacement" || key == "max_outer_refine_displacement") {
      config.outer_detector_config.max_outer_refine_displacement = ParseDouble(key, value);
      config.outer_detector_config.outer_refine_gate_min =
          config.outer_detector_config.max_outer_refine_displacement;
    } else if (key == "outerRefineDisplacementScale" || key == "outer_refine_displacement_scale") {
      config.outer_detector_config.outer_refine_displacement_scale = ParseDouble(key, value);
      config.outer_detector_config.outer_refine_gate_scale =
          config.outer_detector_config.outer_refine_displacement_scale;
    } else if (key == "minDetectionQuality" || key == "min_detection_quality") {
      config.outer_detector_config.min_detection_quality = ParseDouble(key, value);
    } else if (key == "blurBeforeDetect" || key == "blur_before_detect") {
      config.outer_detector_config.blur_before_detect =
          Lowercase(value) == "1" || Lowercase(value) == "true" ||
          Lowercase(value) == "yes" || Lowercase(value) == "on";
    } else if (key == "blurKernel" || key == "blur_kernel") {
      config.outer_detector_config.blur_kernel = ParseInt(key, value);
    } else if (key == "blurSigma" || key == "blur_sigma") {
      config.outer_detector_config.blur_sigma = ParseDouble(key, value);
    } else if (key == "enableOuterCornerLocalVerification" ||
               key == "enable_outer_corner_local_verification") {
      // Legacy compatibility: the outer pipeline now always runs C-S.
      (void)(Lowercase(value) == "1" || Lowercase(value) == "true" ||
             Lowercase(value) == "yes" || Lowercase(value) == "on");
    } else if (key == "enableOuterCornerLayoutCheck" ||
               key == "enable_outer_corner_layout_check") {
      config.outer_detector_config.enable_outer_corner_layout_check =
          Lowercase(value) == "1" || Lowercase(value) == "true" ||
          Lowercase(value) == "yes" || Lowercase(value) == "on";
    } else if (key == "outerCornerVerificationRoiScale" ||
               key == "outer_corner_verification_roi_scale") {
      config.outer_detector_config.outer_corner_verification_roi_scale = ParseDouble(key, value);
      config.outer_detector_config.outer_local_context_scale =
          config.outer_detector_config.outer_corner_verification_roi_scale;
    } else if (key == "outerCornerVerificationRoiMin" ||
               key == "outer_corner_verification_roi_min") {
      config.outer_detector_config.outer_corner_verification_roi_min = ParseInt(key, value);
    } else if (key == "outerCornerVerificationRoiMax" ||
               key == "outer_corner_verification_roi_max") {
      config.outer_detector_config.outer_corner_verification_roi_max = ParseInt(key, value);
    } else if (key == "outerCornerCandidateScale" ||
               key == "outer_corner_candidate_scale") {
      config.outer_detector_config.outer_corner_candidate_scale = ParseDouble(key, value);
    } else if (key == "outerCornerCandidateMin" ||
               key == "outer_corner_candidate_min") {
      config.outer_detector_config.outer_corner_candidate_min = ParseInt(key, value);
    } else if (key == "outerCornerCandidateMax" ||
               key == "outer_corner_candidate_max") {
      config.outer_detector_config.outer_corner_candidate_max = ParseInt(key, value);
    } else if (key == "outerCornerBranchSearchScale" ||
               key == "outer_corner_branch_search_scale") {
      config.outer_detector_config.outer_corner_branch_search_scale = ParseDouble(key, value);
    } else if (key == "outerCornerBranchSearchMin" ||
               key == "outer_corner_branch_search_min") {
      config.outer_detector_config.outer_corner_branch_search_min = ParseInt(key, value);
    } else if (key == "outerCornerBranchSearchMax" ||
               key == "outer_corner_branch_search_max") {
      config.outer_detector_config.outer_corner_branch_search_max = ParseInt(key, value);
    } else if (key == "outerCornerVerificationRoiRadius" ||
               key == "outer_corner_verification_roi_radius") {
      const int fixed_radius = ParseInt(key, value);
      config.outer_detector_config.outer_corner_verification_roi_scale = 0.0;
      config.outer_detector_config.outer_corner_verification_roi_min = fixed_radius;
      config.outer_detector_config.outer_corner_verification_roi_max = fixed_radius;
    } else if (key == "outerCornerCandidateRadius" ||
               key == "outer_corner_candidate_radius") {
      const int fixed_radius = ParseInt(key, value);
      config.outer_detector_config.outer_corner_candidate_scale = 0.0;
      config.outer_detector_config.outer_corner_candidate_min = fixed_radius;
      config.outer_detector_config.outer_corner_candidate_max = fixed_radius;
    } else if (key == "outerCornerBranchSearchRadius" ||
               key == "outer_corner_branch_search_radius") {
      const int fixed_radius = ParseInt(key, value);
      config.outer_detector_config.outer_corner_branch_search_scale = 0.0;
      config.outer_detector_config.outer_corner_branch_search_min = fixed_radius;
      config.outer_detector_config.outer_corner_branch_search_max = fixed_radius;
    } else if (key == "outerCornerMinDirectionScore" ||
               key == "outer_corner_min_direction_score") {
      config.outer_detector_config.outer_corner_min_direction_score = ParseDouble(key, value);
    } else if (key == "outerCornerMinLayoutScore" ||
               key == "outer_corner_min_layout_score") {
      config.outer_detector_config.outer_corner_min_layout_score = ParseDouble(key, value);
    } else {
      ApplyCameraField(&config.intermediate_camera, key, value, yaml_path);
    }
  }

  if (!config.intermediate_camera.camera_yaml.empty()) {
    IntermediateCameraConfig loaded = LoadExternalCameraConfig(config.intermediate_camera.camera_yaml);
    if (!config.intermediate_camera.camera_model.empty()) {
      loaded.camera_model = config.intermediate_camera.camera_model;
    }
    if (!config.intermediate_camera.distortion_model.empty()) {
      loaded.distortion_model = config.intermediate_camera.distortion_model;
    }
    if (!config.intermediate_camera.intrinsics.empty()) {
      loaded.intrinsics = config.intermediate_camera.intrinsics;
    }
    if (!config.intermediate_camera.distortion_coeffs.empty()) {
      loaded.distortion_coeffs = config.intermediate_camera.distortion_coeffs;
    }
    if (!config.intermediate_camera.resolution.empty()) {
      loaded.resolution = config.intermediate_camera.resolution;
    }
    config.intermediate_camera = loaded;
  }

  config.tag_ids = NormalizeBoardIds(config.tag_ids, config.tag_id);
  config.tag_id = config.tag_ids.front();
  config.outer_detector_config.tag_id = config.tag_id;

  return config;
}

float ClampFloat(float value, float min_value, float max_value) {
  return std::max(min_value, std::min(max_value, value));
}

double ClampUnit(double value) {
  return std::max(0.0, std::min(1.0, value));
}

bool IsInsideImage(const cv::Point2f& point, const cv::Size& image_size) {
  return point.x >= 0.0f && point.x < static_cast<float>(image_size.width) &&
         point.y >= 0.0f && point.y < static_cast<float>(image_size.height);
}

bool IsInsideImageWithBorder(const cv::Point2f& point,
                             const cv::Size& image_size,
                             double border_distance) {
  return point.x >= border_distance &&
         point.x <= static_cast<float>(image_size.width) - border_distance &&
         point.y >= border_distance &&
         point.y <= static_cast<float>(image_size.height) - border_distance;
}

cv::Rect MakeClampedRoi(const cv::Point2f& center, int radius, const cv::Size& image_size) {
  const int left = std::max(0, static_cast<int>(std::round(center.x)) - radius);
  const int top = std::max(0, static_cast<int>(std::round(center.y)) - radius);
  const int right = std::min(image_size.width, static_cast<int>(std::round(center.x)) + radius + 1);
  const int bottom = std::min(image_size.height, static_cast<int>(std::round(center.y)) + radius + 1);
  return cv::Rect(left, top, std::max(0, right - left), std::max(0, bottom - top));
}

int ClampRadiusFromScale(double scale, double local_scale, int min_radius, int max_radius) {
  if (min_radius <= 0 || max_radius < min_radius) {
    throw std::runtime_error("Invalid adaptive radius bounds.");
  }
  const double scaled = scale > 0.0 ? scale * local_scale : static_cast<double>(min_radius);
  const int rounded = static_cast<int>(std::lround(scaled));
  return std::max(min_radius, std::min(max_radius, rounded));
}

double ComputeModuleScalePx(const cv::Point2f& module_u_axis,
                            const cv::Point2f& module_v_axis,
                            double fallback) {
  const double u_norm = std::hypot(module_u_axis.x, module_u_axis.y);
  const double v_norm = std::hypot(module_v_axis.x, module_v_axis.y);
  const bool u_valid = std::isfinite(u_norm) && u_norm > 1e-6;
  const bool v_valid = std::isfinite(v_norm) && v_norm > 1e-6;
  if (u_valid && v_valid) {
    return std::min(u_norm, v_norm);
  }
  if (u_valid) {
    return u_norm;
  }
  if (v_valid) {
    return v_norm;
  }
  return fallback;
}

int ComputeAdaptiveInternalSubpixRadius(double module_scale_px,
                                        const ApriltagInternalDetectionOptions& options) {
  if (options.refinement_window_radius > 0) {
    return options.refinement_window_radius;
  }
  return ClampRadiusFromScale(options.internal_subpix_window_scale, module_scale_px,
                              options.internal_subpix_window_min, options.internal_subpix_window_max);
}

double ComputeAdaptiveInternalSubpixDisplacementLimit(
    double module_scale_px,
    const ApriltagInternalDetectionOptions& options) {
  if (options.max_subpix_displacement2 > 0.0) {
    return std::sqrt(std::max(1e-9, options.max_subpix_displacement2));
  }
  const double scale_based_limit =
      options.internal_subpix_displacement_scale > 0.0
          ? options.internal_subpix_displacement_scale * std::max(0.0, module_scale_px)
          : 0.0;
  return std::max(options.max_internal_subpix_displacement, scale_based_limit);
}

int ComputeAdaptiveImageEvidenceSearchRadius(double module_scale_px,
                                             const ApriltagInternalDetectionOptions& options) {
  return std::max(1, 2 * ComputeAdaptiveInternalSubpixRadius(module_scale_px, options));
}

double MeanIntensity(const cv::Mat& patch, const cv::Point2f& center, int radius) {
  const cv::Rect roi = MakeClampedRoi(center, radius, patch.size());
  if (roi.width <= 0 || roi.height <= 0) {
    return 0.0;
  }
  return cv::mean(patch(roi))[0];
}

double SampleFloatAt(const cv::Mat& image, const cv::Point2f& point) {
  if (image.empty()) {
    return 0.0;
  }

  const float x = ClampFloat(point.x, 0.0f, static_cast<float>(image.cols - 1));
  const float y = ClampFloat(point.y, 0.0f, static_cast<float>(image.rows - 1));

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, image.cols - 1);
  const int y1 = std::min(y0 + 1, image.rows - 1);
  const float dx = x - static_cast<float>(x0);
  const float dy = y - static_cast<float>(y0);

  const float v00 = image.at<float>(y0, x0);
  const float v10 = image.at<float>(y0, x1);
  const float v01 = image.at<float>(y1, x0);
  const float v11 = image.at<float>(y1, x1);

  const float top = v00 * (1.0f - dx) + v10 * dx;
  const float bottom = v01 * (1.0f - dx) + v11 * dx;
  return static_cast<double>(top * (1.0f - dy) + bottom * dy);
}

cv::Point2f BoardToPatchPoint(const CanonicalCorner& corner_info,
                              int module_dimension,
                              int pixels_per_module) {
  const float patch_extent = static_cast<float>(module_dimension * pixels_per_module);
  return cv::Point2f(static_cast<float>(corner_info.lattice_u * pixels_per_module),
                     patch_extent - static_cast<float>(corner_info.lattice_v * pixels_per_module));
}

cv::Point2f PerspectiveTransformPoint(const cv::Mat& transform, const cv::Point2f& point) {
  std::vector<cv::Point2f> input{point};
  std::vector<cv::Point2f> output;
  cv::perspectiveTransform(input, output, transform);
  if (output.empty()) {
    return cv::Point2f();
  }
  return output.front();
}

cv::Point2f ProjectBoardPointToImage(const cv::Mat& board_to_image, double lattice_u, double lattice_v) {
  return PerspectiveTransformPoint(board_to_image, cv::Point2f(static_cast<float>(lattice_u),
                                                               static_cast<float>(lattice_v)));
}

TemplateScore ComputeTemplateScoreAtPoint(const cv::Mat& patch,
                                          const CanonicalCorner& corner_info,
                                          const cv::Point2f& patch_point,
                                          int pixels_per_module,
                                          double min_template_contrast) {
  if (corner_info.corner_type == CornerType::Outer || !corner_info.observable) {
    return {1.0, 1.0};
  }
  if (!IsInsideImage(patch_point, patch.size())) {
    return {0.0, 0.0};
  }

  const int sample_radius = std::max(1, pixels_per_module / 5);
  const double half_module = 0.5 * static_cast<double>(pixels_per_module);
  const double mean_ll =
      MeanIntensity(patch, cv::Point2f(patch_point.x - static_cast<float>(half_module),
                                       patch_point.y + static_cast<float>(half_module)),
                    sample_radius);
  const double mean_lr =
      MeanIntensity(patch, cv::Point2f(patch_point.x + static_cast<float>(half_module),
                                       patch_point.y + static_cast<float>(half_module)),
                    sample_radius);
  const double mean_ul =
      MeanIntensity(patch, cv::Point2f(patch_point.x - static_cast<float>(half_module),
                                       patch_point.y - static_cast<float>(half_module)),
                    sample_radius);
  const double mean_ur =
      MeanIntensity(patch, cv::Point2f(patch_point.x + static_cast<float>(half_module),
                                       patch_point.y - static_cast<float>(half_module)),
                    sample_radius);

  const std::array<double, 4> means{{mean_ll, mean_lr, mean_ul, mean_ur}};
  const std::pair<std::array<double, 4>::const_iterator, std::array<double, 4>::const_iterator> minmax =
      std::minmax_element(means.begin(), means.end());
  const std::array<double, 4>::const_iterator min_it = minmax.first;
  const std::array<double, 4>::const_iterator max_it = minmax.second;
  const double contrast = *max_it - *min_it;
  if (contrast < min_template_contrast) {
    return {0.0, 0.0};
  }

  std::array<double, 4> normalized{};
  for (std::size_t i = 0; i < means.size(); ++i) {
    normalized[i] = (means[i] - *min_it) / contrast;
  }

  double template_error = 0.0;
  for (std::size_t i = 0; i < normalized.size(); ++i) {
    const double expected_white = corner_info.module_pattern[i] ? 0.0 : 1.0;
    template_error += std::abs(normalized[i] - expected_white);
  }
  template_error /= static_cast<double>(normalized.size());

  const double delta_x = 0.5 * (std::abs(normalized[0] - normalized[1]) +
                                std::abs(normalized[2] - normalized[3]));
  const double delta_y = 0.5 * (std::abs(normalized[0] - normalized[2]) +
                                std::abs(normalized[1] - normalized[3]));
  const double contrast_quality = ClampUnit(contrast / 128.0);

  return {ClampUnit((1.0 - template_error) * contrast_quality),
          ClampUnit(std::min(delta_x, delta_y) * contrast_quality)};
}

double ExpectedBrightness(bool is_black) {
  return is_black ? 0.0 : 1.0;
}

double ComputeBoundaryTransitionSampleQuality(const cv::Mat& gray,
                                             const cv::Point2f& center,
                                             const cv::Point2f& tangent_unit,
                                             const cv::Point2f& normal_unit,
                                             double tangent_offset,
                                             double normal_offset,
                                             int probe_radius,
                                             bool negative_side_black,
                                             bool positive_side_black,
                                             double min_template_contrast) {
  const double expected_negative = ExpectedBrightness(negative_side_black);
  const double expected_positive = ExpectedBrightness(positive_side_black);
  const double expected_sign = expected_positive - expected_negative;
  if (std::abs(expected_sign) < 0.5) {
    return -1.0;
  }

  const cv::Point2f sample_center =
      center + static_cast<float>(tangent_offset) * tangent_unit;
  const cv::Point2f negative_point =
      sample_center - static_cast<float>(normal_offset) * normal_unit;
  const cv::Point2f positive_point =
      sample_center + static_cast<float>(normal_offset) * normal_unit;
  if (!IsInsideImageWithBorder(negative_point, gray.size(), probe_radius + 1.0) ||
      !IsInsideImageWithBorder(positive_point, gray.size(), probe_radius + 1.0)) {
    return -1.0;
  }

  const double negative_mean = MeanIntensity(gray, negative_point, probe_radius);
  const double positive_mean = MeanIntensity(gray, positive_point, probe_radius);
  const double signed_transition =
      expected_sign > 0.0 ? (positive_mean - negative_mean) : (negative_mean - positive_mean);
  return ClampUnit((signed_transition - min_template_contrast) / 96.0);
}

TemplateScore ComputeBoundaryAlignmentScoreAtPoint(const cv::Mat& gray,
                                                   const CanonicalCorner& corner_info,
                                                   const cv::Point2f& image_point,
                                                   const cv::Point2f& module_u_axis,
                                                   const cv::Point2f& module_v_axis,
                                                   double min_template_contrast,
                                                   double probe_radius_scale = 1.0) {
  if (corner_info.corner_type == CornerType::Outer || !corner_info.observable) {
    return {1.0, 1.0};
  }
  if (!IsInsideImage(image_point, gray.size())) {
    return {0.0, 0.0};
  }

  const double module_u_length = std::hypot(module_u_axis.x, module_u_axis.y);
  const double module_v_length = std::hypot(module_v_axis.x, module_v_axis.y);
  const double local_module_size = std::min(module_u_length, module_v_length);
  if (local_module_size < 1.0) {
    return {0.0, 0.0};
  }

  const float inverse_u_length = static_cast<float>(1.0 / std::max(1e-9, module_u_length));
  const float inverse_v_length = static_cast<float>(1.0 / std::max(1e-9, module_v_length));
  const cv::Point2f unit_u = module_u_axis * inverse_u_length;
  const cv::Point2f unit_v = module_v_axis * inverse_v_length;
  const double probe_offset = kBoundaryProbeOffsetScale * local_module_size;
  const double u_tangent_offset = kBoundaryTangentOffsetScale * module_v_length;
  const double v_tangent_offset = kBoundaryTangentOffsetScale * module_u_length;
  const int probe_radius =
      std::max(1, static_cast<int>(std::lround(probe_radius_scale *
                                               kBoundaryProbeRadiusScale *
                                               local_module_size / 6.0)));

  std::vector<double> u_samples;
  const double u_lower = ComputeBoundaryTransitionSampleQuality(
      gray, image_point, unit_v, unit_u, -u_tangent_offset, probe_offset, probe_radius,
      corner_info.module_pattern[0], corner_info.module_pattern[1], min_template_contrast);
  if (u_lower >= 0.0) {
    u_samples.push_back(u_lower);
  }
  const double u_upper = ComputeBoundaryTransitionSampleQuality(
      gray, image_point, unit_v, unit_u, u_tangent_offset, probe_offset, probe_radius,
      corner_info.module_pattern[2], corner_info.module_pattern[3], min_template_contrast);
  if (u_upper >= 0.0) {
    u_samples.push_back(u_upper);
  }

  std::vector<double> v_samples;
  const double v_left = ComputeBoundaryTransitionSampleQuality(
      gray, image_point, unit_u, unit_v, -v_tangent_offset, probe_offset, probe_radius,
      corner_info.module_pattern[0], corner_info.module_pattern[2], min_template_contrast);
  if (v_left >= 0.0) {
    v_samples.push_back(v_left);
  }
  const double v_right = ComputeBoundaryTransitionSampleQuality(
      gray, image_point, unit_u, unit_v, v_tangent_offset, probe_offset, probe_radius,
      corner_info.module_pattern[1], corner_info.module_pattern[3], min_template_contrast);
  if (v_right >= 0.0) {
    v_samples.push_back(v_right);
  }

  auto average_samples = [](const std::vector<double>& samples) {
    if (samples.empty()) {
      return 0.0;
    }
    double sum = 0.0;
    for (double sample : samples) {
      sum += sample;
    }
    return ClampUnit(sum / static_cast<double>(samples.size()));
  };

  return {average_samples(u_samples), average_samples(v_samples)};
}

double ComputePhotometricConsistencyScoreAtPoint(const cv::Mat& gray,
                                                 const CanonicalCorner& corner_info,
                                                 const cv::Point2f& image_point,
                                                 const cv::Point2f& module_u_axis,
                                                 const cv::Point2f& module_v_axis,
                                                 double min_template_contrast) {
  if (corner_info.corner_type == CornerType::Outer || !corner_info.observable) {
    return 1.0;
  }
  if (!IsInsideImage(image_point, gray.size())) {
    return 0.0;
  }

  const double local_module_size = ComputeLocalModuleSizePx(module_u_axis, module_v_axis);
  if (local_module_size < 1.0) {
    return 0.0;
  }

  const int probe_radius = std::max(
      1, static_cast<int>(std::lround(kPhotometricProbeRadiusScale * local_module_size / 6.0)));
  const std::array<cv::Point2f, 4> sample_centers{{
      image_point - static_cast<float>(kPhotometricQuadrantOffsetScale) * module_u_axis -
          static_cast<float>(kPhotometricQuadrantOffsetScale) * module_v_axis,
      image_point + static_cast<float>(kPhotometricQuadrantOffsetScale) * module_u_axis -
          static_cast<float>(kPhotometricQuadrantOffsetScale) * module_v_axis,
      image_point - static_cast<float>(kPhotometricQuadrantOffsetScale) * module_u_axis +
          static_cast<float>(kPhotometricQuadrantOffsetScale) * module_v_axis,
      image_point + static_cast<float>(kPhotometricQuadrantOffsetScale) * module_u_axis +
          static_cast<float>(kPhotometricQuadrantOffsetScale) * module_v_axis,
  }};

  std::vector<double> black_samples;
  std::vector<double> white_samples;
  black_samples.reserve(4);
  white_samples.reserve(4);
  for (std::size_t index = 0; index < sample_centers.size(); ++index) {
    if (!IsInsideImageWithBorder(sample_centers[index], gray.size(), probe_radius + 1.0)) {
      return 0.0;
    }
    const double mean = MeanIntensity(gray, sample_centers[index], probe_radius);
    if (corner_info.module_pattern[index]) {
      black_samples.push_back(mean);
    } else {
      white_samples.push_back(mean);
    }
  }
  if (black_samples.empty() || white_samples.empty()) {
    return 0.0;
  }

  auto compute_mean = [](const std::vector<double>& samples) {
    double sum = 0.0;
    for (double value : samples) {
      sum += value;
    }
    return sum / static_cast<double>(samples.size());
  };
  auto compute_std = [](const std::vector<double>& samples, double mean) {
    double sum_sq = 0.0;
    for (double value : samples) {
      const double delta = value - mean;
      sum_sq += delta * delta;
    }
    return std::sqrt(sum_sq / static_cast<double>(samples.size()));
  };

  const double black_mean = compute_mean(black_samples);
  const double white_mean = compute_mean(white_samples);
  const double contrast = white_mean - black_mean;
  if (!std::isfinite(contrast) || contrast < min_template_contrast) {
    return 0.0;
  }

  const double contrast_score = ClampUnit((contrast - min_template_contrast) / 96.0);
  const double std_black = compute_std(black_samples, black_mean);
  const double std_white = compute_std(white_samples, white_mean);
  const double consistency_penalty =
      (std_black + std_white) / std::max(contrast, 1e-6);
  const double consistency_score = ClampUnit(1.0 - consistency_penalty);
  return std::min(contrast_score, consistency_score);
}

RayRefinementEvaluation EvaluateRayRefinementObjective(
    const cv::Mat& gray,
    const DoubleSphereCameraModel& camera,
    const CanonicalCorner& corner_info,
    const SphereLatticeFrame& frame,
    const Eigen::Vector3d& seed_ray,
    const Eigen::Vector3d& seed_tangent_u,
    const Eigen::Vector3d& seed_tangent_v,
    const ApriltagInternalDetectionOptions& options,
    double du,
    double dv) {
  RayRefinementEvaluation evaluation;
  Eigen::Vector3d candidate_ray = Eigen::Vector3d::Zero();
  if (!BuildLocalSphereOffsetRay(seed_ray, seed_tangent_u, seed_tangent_v, du, dv, &candidate_ray)) {
    return evaluation;
  }

  cv::Point2f candidate_image;
  if (!ProjectRayToImage(camera, candidate_ray, &candidate_image) ||
      !IsInsideImage(candidate_image, gray.size())) {
    return evaluation;
  }

  const TemplateScore edge_score = ComputeBoundaryAlignmentScoreAtPoint(
      gray, corner_info, candidate_image, frame.module_u_axis, frame.module_v_axis,
      options.min_template_contrast, kSphereSeedSampleRadiusScale);
  const double edge_quality = std::min(edge_score.template_quality, edge_score.gradient_quality);
  const double photometric_quality = ComputePhotometricConsistencyScoreAtPoint(
      gray, corner_info, candidate_image, frame.module_u_axis, frame.module_v_axis,
      options.min_template_contrast);

  evaluation.valid = true;
  evaluation.image_point = candidate_image;
  evaluation.ray = candidate_ray;
  evaluation.template_quality = edge_score.template_quality;
  evaluation.gradient_quality = edge_score.gradient_quality;
  evaluation.edge_quality = edge_quality;
  evaluation.photometric_quality = photometric_quality;
  evaluation.final_quality = ClampUnit(kRayRefineEdgeWeight * edge_quality +
                                       kRayRefinePhotoWeight * photometric_quality);
  return evaluation;
}

RayRefinementResult RefineSphereSeedRayLocally(const cv::Mat& gray,
                                               const DoubleSphereCameraModel& camera,
                                               const CanonicalCorner& corner_info,
                                               const SphereLatticeFrame& frame,
                                               const ApriltagInternalDetectionOptions& options,
                                               const Eigen::Vector3d& sphere_seed_ray) {
  RayRefinementResult result;

  Eigen::Vector3d seed_tangent_u =
      ProjectOntoTangentPlane(frame.tangent_u, sphere_seed_ray);
  if (!NormalizeRay(&seed_tangent_u)) {
    return result;
  }
  Eigen::Vector3d seed_tangent_v =
      ProjectOntoTangentPlane(frame.tangent_v, sphere_seed_ray);
  seed_tangent_v =
      seed_tangent_v - seed_tangent_u * seed_tangent_u.dot(seed_tangent_v);
  if (!NormalizeRay(&seed_tangent_v)) {
    return result;
  }
  if (sphere_seed_ray.dot(seed_tangent_u.cross(seed_tangent_v)) < 0.0) {
    seed_tangent_v = -seed_tangent_v;
  }

  const double trust_radius = std::max(
      kRayRefineTrustRadiusMin,
      std::min(kRayRefineTrustRadiusScale * frame.theta_local, frame.search_radius));
  result.trust_radius = trust_radius;

  const RayRefinementEvaluation seed_evaluation = EvaluateRayRefinementObjective(
      gray, camera, corner_info, frame, sphere_seed_ray, seed_tangent_u, seed_tangent_v,
      options, 0.0, 0.0);
  if (!seed_evaluation.valid) {
    return result;
  }

  double current_du = 0.0;
  double current_dv = 0.0;
  RayRefinementEvaluation best = seed_evaluation;
  bool improved = false;
  double h = std::max(kRayRefineDiffStepMin, kRayRefineDiffStepScale * trust_radius);

  for (int iteration = 0; iteration < kRayRefineMaxIterations; ++iteration) {
    result.iterations = iteration + 1;
    if (h < kRayRefineMinStep) {
      break;
    }

    auto evaluate_offset = [&](double du, double dv,
                               double* projected_du,
                               double* projected_dv) {
      ProjectOffsetToDisk(trust_radius, &du, &dv);
      if (projected_du != nullptr) {
        *projected_du = du;
      }
      if (projected_dv != nullptr) {
        *projected_dv = dv;
      }
      return EvaluateRayRefinementObjective(
          gray, camera, corner_info, frame, sphere_seed_ray, seed_tangent_u, seed_tangent_v,
          options, du, dv);
    };

    double plus_u_du = 0.0;
    double plus_u_dv = 0.0;
    double minus_u_du = 0.0;
    double minus_u_dv = 0.0;
    double plus_v_du = 0.0;
    double plus_v_dv = 0.0;
    double minus_v_du = 0.0;
    double minus_v_dv = 0.0;
    const RayRefinementEvaluation plus_u =
        evaluate_offset(current_du + h, current_dv, &plus_u_du, &plus_u_dv);
    const RayRefinementEvaluation minus_u =
        evaluate_offset(current_du - h, current_dv, &minus_u_du, &minus_u_dv);
    const RayRefinementEvaluation plus_v =
        evaluate_offset(current_du, current_dv + h, &plus_v_du, &plus_v_dv);
    const RayRefinementEvaluation minus_v =
        evaluate_offset(current_du, current_dv - h, &minus_v_du, &minus_v_dv);

    double grad_u = 0.0;
    if (plus_u.valid && minus_u.valid &&
        std::abs(plus_u_du - minus_u_du) > 1e-9) {
      grad_u = (plus_u.final_quality - minus_u.final_quality) /
               (plus_u_du - minus_u_du);
    } else if (plus_u.valid && std::abs(plus_u_du - current_du) > 1e-9) {
      grad_u = (plus_u.final_quality - best.final_quality) /
               (plus_u_du - current_du);
    } else if (minus_u.valid && std::abs(current_du - minus_u_du) > 1e-9) {
      grad_u = (best.final_quality - minus_u.final_quality) /
               (current_du - minus_u_du);
    }

    double grad_v = 0.0;
    if (plus_v.valid && minus_v.valid &&
        std::abs(plus_v_dv - minus_v_dv) > 1e-9) {
      grad_v = (plus_v.final_quality - minus_v.final_quality) /
               (plus_v_dv - minus_v_dv);
    } else if (plus_v.valid && std::abs(plus_v_dv - current_dv) > 1e-9) {
      grad_v = (plus_v.final_quality - best.final_quality) /
               (plus_v_dv - current_dv);
    } else if (minus_v.valid && std::abs(current_dv - minus_v_dv) > 1e-9) {
      grad_v = (best.final_quality - minus_v.final_quality) /
               (current_dv - minus_v_dv);
    }

    const double grad_norm = std::hypot(grad_u, grad_v);
    if (!std::isfinite(grad_norm) || grad_norm < 1e-6) {
      break;
    }

    const double direction_u = grad_u / grad_norm;
    const double direction_v = grad_v / grad_norm;
    bool accepted = false;
    for (double scale : kRayRefineLineSearchScales) {
      double candidate_du = current_du + scale * h * direction_u;
      double candidate_dv = current_dv + scale * h * direction_v;
      ProjectOffsetToDisk(trust_radius, &candidate_du, &candidate_dv);
      const RayRefinementEvaluation candidate = EvaluateRayRefinementObjective(
          gray, camera, corner_info, frame, sphere_seed_ray, seed_tangent_u, seed_tangent_v,
          options, candidate_du, candidate_dv);
      if (candidate.valid && candidate.final_quality > best.final_quality + 1e-9) {
        current_du = candidate_du;
        current_dv = candidate_dv;
        best = candidate;
        improved = true;
        accepted = true;
        break;
      }
    }

    if (!accepted) {
      h *= 0.5;
    }
  }

  result.valid = true;
  result.converged = improved;
  result.refined_image = best.image_point;
  result.refined_ray = best.ray;
  result.template_quality = best.template_quality;
  result.gradient_quality = best.gradient_quality;
  result.edge_quality = best.edge_quality;
  result.photometric_quality = best.photometric_quality;
  result.final_quality = best.final_quality;
  return result;
}

TemplateScore ComputeImageEvidenceScoreAtPoint(const cv::Mat& gray,
                                               const CanonicalCorner& corner_info,
                                               const cv::Point2f& image_point,
                                               const cv::Point2f& module_u_axis,
                                               const cv::Point2f& module_v_axis,
                                               double min_template_contrast,
                                               double sample_radius_scale = 1.0) {
  if (corner_info.corner_type == CornerType::Outer || !corner_info.observable) {
    return {1.0, 1.0};
  }
  if (!IsInsideImage(image_point, gray.size())) {
    return {0.0, 0.0};
  }

  const double module_u_length = std::hypot(module_u_axis.x, module_u_axis.y);
  const double module_v_length = std::hypot(module_v_axis.x, module_v_axis.y);
  const double local_module_size = std::min(module_u_length, module_v_length);
  if (local_module_size < 1.0) {
    return {0.0, 0.0};
  }

  const int sample_radius = std::max(
      1, static_cast<int>(std::lround(sample_radius_scale * local_module_size / 6.0)));
  const std::array<cv::Point2f, 4> sample_centers{{
      image_point - 0.5f * module_u_axis - 0.5f * module_v_axis,
      image_point + 0.5f * module_u_axis - 0.5f * module_v_axis,
      image_point - 0.5f * module_u_axis + 0.5f * module_v_axis,
      image_point + 0.5f * module_u_axis + 0.5f * module_v_axis,
  }};

  for (const cv::Point2f& sample_center : sample_centers) {
    if (!IsInsideImageWithBorder(sample_center, gray.size(), sample_radius + 1.0)) {
      return {0.0, 0.0};
    }
  }

  const std::array<double, 4> means{{
      MeanIntensity(gray, sample_centers[0], sample_radius),
      MeanIntensity(gray, sample_centers[1], sample_radius),
      MeanIntensity(gray, sample_centers[2], sample_radius),
      MeanIntensity(gray, sample_centers[3], sample_radius),
  }};

  const std::pair<std::array<double, 4>::const_iterator, std::array<double, 4>::const_iterator> minmax =
      std::minmax_element(means.begin(), means.end());
  const std::array<double, 4>::const_iterator min_it = minmax.first;
  const std::array<double, 4>::const_iterator max_it = minmax.second;
  const double contrast = *max_it - *min_it;
  if (contrast < min_template_contrast) {
    return {0.0, 0.0};
  }

  std::array<double, 4> normalized{};
  for (std::size_t i = 0; i < means.size(); ++i) {
    normalized[i] = (means[i] - *min_it) / contrast;
  }

  double template_error = 0.0;
  for (std::size_t i = 0; i < normalized.size(); ++i) {
    const double expected_white = corner_info.module_pattern[i] ? 0.0 : 1.0;
    template_error += std::abs(normalized[i] - expected_white);
  }
  template_error /= static_cast<double>(normalized.size());

  const double delta_x = 0.5 * (std::abs(normalized[0] - normalized[1]) +
                                std::abs(normalized[2] - normalized[3]));
  const double delta_y = 0.5 * (std::abs(normalized[0] - normalized[2]) +
                                std::abs(normalized[1] - normalized[3]));
  const double contrast_quality = ClampUnit(contrast / 128.0);

  return {ClampUnit((1.0 - template_error) * contrast_quality),
          ClampUnit(std::min(delta_x, delta_y) * contrast_quality)};
}

ImageEvidenceScore EvaluateImageEvidenceAroundPoint(const cv::Mat& gray,
                                                    const CanonicalCorner& corner_info,
                                                    const cv::Point2f& image_point,
                                                    const cv::Point2f& module_u_axis,
                                                    const cv::Point2f& module_v_axis,
                                                    double min_template_contrast,
                                                    double max_center_error2,
                                                    int search_radius) {
  ImageEvidenceScore evidence;
  evidence.best_point = image_point;
  evidence.point_score = ComputeImageEvidenceScoreAtPoint(gray, corner_info, image_point,
                                                          module_u_axis, module_v_axis,
                                                          min_template_contrast);
  evidence.best_score = evidence.point_score;
  double best_raw_quality =
      std::min(evidence.best_score.template_quality, evidence.best_score.gradient_quality);
  double best_distance2 = 0.0;

  for (int dy = -search_radius; dy <= search_radius; ++dy) {
    for (int dx = -search_radius; dx <= search_radius; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;
      }

      const cv::Point2f candidate =
          image_point + cv::Point2f(static_cast<float>(dx), static_cast<float>(dy));
      const TemplateScore candidate_score =
          ComputeImageEvidenceScoreAtPoint(gray, corner_info, candidate, module_u_axis, module_v_axis,
                                           min_template_contrast);
      const double candidate_raw_quality =
          std::min(candidate_score.template_quality, candidate_score.gradient_quality);
      const double candidate_distance2 = static_cast<double>(dx * dx + dy * dy);
      if (candidate_raw_quality > best_raw_quality + 1e-9 ||
          (std::abs(candidate_raw_quality - best_raw_quality) <= 1e-9 &&
           candidate_distance2 < best_distance2)) {
        evidence.best_point = candidate;
        evidence.best_score = candidate_score;
        best_raw_quality = candidate_raw_quality;
        best_distance2 = candidate_distance2;
      }
    }
  }

  const double center_error2 =
      std::pow(static_cast<double>(evidence.best_point.x - image_point.x), 2.0) +
      std::pow(static_cast<double>(evidence.best_point.y - image_point.y), 2.0);
  const double point_raw_quality =
      std::min(evidence.point_score.template_quality, evidence.point_score.gradient_quality);
  const double local_module_size =
      0.5 * (std::hypot(module_u_axis.x, module_u_axis.y) + std::hypot(module_v_axis.x, module_v_axis.y));
  const double allowed_center_error =
      std::max(std::sqrt(std::max(1.0, max_center_error2)), 0.25 * local_module_size);
  const double distance_quality =
      ClampUnit(1.0 - center_error2 / std::max(1.0, allowed_center_error * allowed_center_error));
  const double ratio_quality =
      best_raw_quality > 1e-9 ? ClampUnit(point_raw_quality / best_raw_quality) : 0.0;
  evidence.centering_quality = ClampUnit(0.5 * distance_quality + 0.5 * ratio_quality);
  evidence.final_quality =
      std::min({evidence.best_score.template_quality, evidence.best_score.gradient_quality,
                evidence.centering_quality});
  return evidence;
}

cv::Point2f ComputeCenter(const std::array<cv::Point2f, 4>& corners) {
  cv::Point2f center(0.0f, 0.0f);
  for (const cv::Point2f& corner : corners) {
    center += corner;
  }
  return center * 0.25f;
}

float ComputePerimeter(const std::array<cv::Point2f, 4>& corners) {
  float perimeter = 0.0f;
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f delta = corners[(index + 1) % 4] - corners[index];
    perimeter += std::hypot(delta.x, delta.y);
  }
  return perimeter;
}

Eigen::Vector3d MatToEigenVector3d(const cv::Mat& vector) {
  cv::Mat vector64;
  vector.convertTo(vector64, CV_64F);
  if (vector64.total() != 3) {
    throw std::runtime_error("Expected a 3-vector.");
  }
  return Eigen::Vector3d(vector64.at<double>(0, 0), vector64.at<double>(1, 0), vector64.at<double>(2, 0));
}

cv::Matx33d Matrix4dToMatx33d(const Eigen::Matrix4d& transform) {
  cv::Matx33d rotation = cv::Matx33d::eye();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      rotation(row, col) = transform(row, col);
    }
  }
  return rotation;
}

Eigen::Vector3d Matrix4dToTranslation(const Eigen::Matrix4d& transform) {
  return Eigen::Vector3d(transform(0, 3), transform(1, 3), transform(2, 3));
}

Eigen::Vector3d Multiply(const cv::Matx33d& matrix, const Eigen::Vector3d& vector) {
  return Eigen::Vector3d(
      matrix(0, 0) * vector.x() + matrix(0, 1) * vector.y() + matrix(0, 2) * vector.z(),
      matrix(1, 0) * vector.x() + matrix(1, 1) * vector.y() + matrix(1, 2) * vector.z(),
      matrix(2, 0) * vector.x() + matrix(2, 1) * vector.y() + matrix(2, 2) * vector.z());
}

bool NormalizeRay(Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    throw std::runtime_error("NormalizeRay requires a valid pointer.");
  }
  const double norm = ray->norm();
  if (!std::isfinite(norm) || norm <= 1e-9) {
    return false;
  }
  *ray /= norm;
  return true;
}

double AngleBetweenRays(const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs) {
  const double lhs_norm = lhs.norm();
  const double rhs_norm = rhs.norm();
  if (!std::isfinite(lhs_norm) || !std::isfinite(rhs_norm) || lhs_norm <= 1e-9 || rhs_norm <= 1e-9) {
    return 0.0;
  }
  const double cosine = lhs.dot(rhs) / (lhs_norm * rhs_norm);
  return std::acos(std::max(-1.0, std::min(1.0, cosine)));
}

Eigen::Vector3d ProjectOntoTangentPlane(const Eigen::Vector3d& vector,
                                        const Eigen::Vector3d& ray_anchor) {
  return vector - ray_anchor * ray_anchor.dot(vector);
}

bool BuildLocalSphereOffsetRay(const Eigen::Vector3d& anchor_ray,
                               const Eigen::Vector3d& tangent_u,
                               const Eigen::Vector3d& tangent_v,
                               double du,
                               double dv,
                               Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    throw std::runtime_error("BuildLocalSphereOffsetRay requires a valid output pointer.");
  }
  *ray = anchor_ray + du * tangent_u + dv * tangent_v;
  return NormalizeRay(ray);
}

double ComputeLocalModuleSizePx(const cv::Point2f& module_u_axis,
                                const cv::Point2f& module_v_axis) {
  return std::min(std::hypot(module_u_axis.x, module_u_axis.y),
                  std::hypot(module_v_axis.x, module_v_axis.y));
}

int ComputeSphereSearchRadiusOverlayPx(double local_module_scale) {
  return std::max(
      6, static_cast<int>(std::lround(0.35 * std::max(1.0, local_module_scale))));
}

int ComputeRayRefineTrustRadiusOverlayPx(double local_module_scale,
                                         int search_radius_px) {
  const int trust_radius_px = std::max(
      4, static_cast<int>(std::lround(0.28 * std::max(1.0, local_module_scale))));
  return std::min(trust_radius_px, std::max(4, search_radius_px - 2));
}

void ProjectOffsetToDisk(double radius, double* du, double* dv) {
  if (du == nullptr || dv == nullptr) {
    throw std::runtime_error("ProjectOffsetToDisk requires valid pointers.");
  }
  if (!std::isfinite(radius) || radius <= 0.0) {
    *du = 0.0;
    *dv = 0.0;
    return;
  }
  const double norm = std::hypot(*du, *dv);
  if (!std::isfinite(norm) || norm <= radius) {
    return;
  }
  const double scale = radius / norm;
  *du *= scale;
  *dv *= scale;
}

bool UnprojectImagePointToRay(const DoubleSphereCameraModel& camera,
                              const cv::Point2f& image_point,
                              Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    throw std::runtime_error("UnprojectImagePointToRay requires a valid output pointer.");
  }
  if (!camera.keypointToEuclidean(Eigen::Vector2d(image_point.x, image_point.y), ray)) {
    return false;
  }
  return NormalizeRay(ray);
}

bool ProjectRayToImage(const DoubleSphereCameraModel& camera,
                       const Eigen::Vector3d& ray,
                       cv::Point2f* image_point) {
  if (image_point == nullptr) {
    throw std::runtime_error("ProjectRayToImage requires a valid output pointer.");
  }

  Eigen::Vector3d normalized_ray = ray;
  if (!NormalizeRay(&normalized_ray)) {
    return false;
  }

  Eigen::Vector2d keypoint = Eigen::Vector2d::Zero();
  if (!camera.vsEuclideanToKeypoint(normalized_ray, &keypoint)) {
    return false;
  }
  *image_point = cv::Point2f(static_cast<float>(keypoint.x()), static_cast<float>(keypoint.y()));
  return true;
}

Eigen::Vector3d TargetToCameraPoint(const cv::Point3f& target_point,
                                    const cv::Matx33d& target_to_camera_rotation,
                                    const Eigen::Vector3d& target_to_camera_translation) {
  return Multiply(target_to_camera_rotation,
                  Eigen::Vector3d(target_point.x, target_point.y, target_point.z)) +
         target_to_camera_translation;
}

cv::Point2f ProjectTargetPointToImage(const DoubleSphereCameraModel& camera,
                                      const cv::Matx33d& target_to_camera_rotation,
                                      const Eigen::Vector3d& target_to_camera_translation,
                                      const cv::Point3f& target_point,
                                      bool* visible) {
  const Eigen::Vector3d point_camera =
      TargetToCameraPoint(target_point, target_to_camera_rotation, target_to_camera_translation);
  if (visible != nullptr) {
    *visible = point_camera.z() > 0.0;
  }

  Eigen::Vector2d image_point = Eigen::Vector2d::Zero();
  const bool projection_success = camera.vsEuclideanToKeypoint(point_camera, &image_point);
  if (visible != nullptr) {
    *visible = *visible && projection_success;
  }
  return cv::Point2f(static_cast<float>(image_point.x()), static_cast<float>(image_point.y()));
}

bool EstimateTargetPose(const DoubleSphereCameraModel& camera,
                        const std::array<cv::Point2f, 4>& outer_corners,
                        const std::array<int, 4>& outer_point_ids,
                        const ApriltagCanonicalModel& model,
                        cv::Mat* rvec,
                        cv::Mat* tvec) {
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(4);
  image_points.reserve(4);

  for (int index = 0; index < 4; ++index) {
    const CanonicalCorner& corner_info = model.corner(outer_point_ids[index]);
    object_points.emplace_back(static_cast<float>(corner_info.target_xyz.x()),
                               static_cast<float>(corner_info.target_xyz.y()),
                               static_cast<float>(corner_info.target_xyz.z()));
    image_points.push_back(outer_corners[index]);
  }

  return camera.estimateTransformation(object_points, image_points, rvec, tvec);
}

cv::Point2f ProjectTargetPointToVirtualPatch(const VirtualPatchContext& context,
                                             const cv::Point3f& target_point,
                                             bool* visible) {
  const Eigen::Vector3d point_camera = TargetToCameraPoint(
      target_point, context.target_to_camera_rotation, context.target_to_camera_translation);
  const Eigen::Vector3d point_virtual = Multiply(context.camera_to_virtual_rotation, point_camera);
  const bool front_visible = point_virtual.z() > 1e-9;
  if (visible != nullptr) {
    *visible = front_visible;
  }
  if (!front_visible) {
    return cv::Point2f();
  }

  return cv::Point2f(static_cast<float>(context.fu * point_virtual.x() / point_virtual.z() + context.cu),
                     static_cast<float>(context.fv * point_virtual.y() / point_virtual.z() + context.cv));
}

bool ProjectImagePointToVirtualPatch(const DoubleSphereCameraModel& camera,
                                     const VirtualPatchContext& context,
                                     const cv::Point2f& image_point,
                                     cv::Point2f* patch_point) {
  if (patch_point == nullptr) {
    throw std::runtime_error("ProjectImagePointToVirtualPatch requires a valid output pointer.");
  }

  Eigen::Vector3d ray_camera = Eigen::Vector3d::Zero();
  if (!UnprojectImagePointToRay(camera, image_point, &ray_camera)) {
    return false;
  }

  const double denominator = context.plane_normal_camera.dot(ray_camera);
  if (std::abs(denominator) < 1e-9) {
    return false;
  }

  const double scale =
      context.plane_normal_camera.dot(context.plane_point_camera) / denominator;
  if (!std::isfinite(scale) || scale <= 1e-9) {
    return false;
  }

  const Eigen::Vector3d point_camera = scale * ray_camera;
  const Eigen::Vector3d point_virtual =
      Multiply(context.camera_to_virtual_rotation, point_camera);
  if (!std::isfinite(point_virtual.z()) || point_virtual.z() <= 1e-9) {
    return false;
  }

  *patch_point = cv::Point2f(
      static_cast<float>(context.fu * point_virtual.x() / point_virtual.z() + context.cu),
      static_cast<float>(context.fv * point_virtual.y() / point_virtual.z() + context.cv));
  return true;
}

bool ProjectVirtualPatchPointToImage(const DoubleSphereCameraModel& camera,
                                     const VirtualPatchContext& context,
                                     const cv::Point2f& patch_point,
                                     cv::Point2f* image_point) {
  if (image_point == nullptr) {
    throw std::runtime_error("ProjectVirtualPatchPointToImage requires a valid output pointer.");
  }

  Eigen::Vector3d point_camera = Eigen::Vector3d::Zero();
  if (!IntersectVirtualPatchPixelWithTargetPlane(context, patch_point, &point_camera)) {
    return false;
  }

  Eigen::Vector2d keypoint = Eigen::Vector2d::Zero();
  if (!camera.vsEuclideanToKeypoint(point_camera, &keypoint)) {
    return false;
  }

  *image_point =
      cv::Point2f(static_cast<float>(keypoint.x()), static_cast<float>(keypoint.y()));
  return true;
}

bool ComputeVirtualPatchAxes(const VirtualPatchContext& context,
                             const CanonicalCorner& corner_info,
                             double pitch,
                             cv::Point2f* module_u_axis,
                             cv::Point2f* module_v_axis) {
  if (module_u_axis == nullptr || module_v_axis == nullptr) {
    throw std::runtime_error("ComputeVirtualPatchAxes requires valid output pointers.");
  }

  const float half_pitch = static_cast<float>(0.5 * pitch);
  const cv::Point3f center(static_cast<float>(corner_info.target_xyz.x()),
                           static_cast<float>(corner_info.target_xyz.y()),
                           static_cast<float>(corner_info.target_xyz.z()));
  const cv::Point3f u_minus(center.x - half_pitch, center.y, center.z);
  const cv::Point3f u_plus(center.x + half_pitch, center.y, center.z);
  const cv::Point3f v_minus(center.x, center.y - half_pitch, center.z);
  const cv::Point3f v_plus(center.x, center.y + half_pitch, center.z);

  bool u_minus_visible = false;
  bool u_plus_visible = false;
  bool v_minus_visible = false;
  bool v_plus_visible = false;
  const cv::Point2f patch_u_minus =
      ProjectTargetPointToVirtualPatch(context, u_minus, &u_minus_visible);
  const cv::Point2f patch_u_plus =
      ProjectTargetPointToVirtualPatch(context, u_plus, &u_plus_visible);
  const cv::Point2f patch_v_minus =
      ProjectTargetPointToVirtualPatch(context, v_minus, &v_minus_visible);
  const cv::Point2f patch_v_plus =
      ProjectTargetPointToVirtualPatch(context, v_plus, &v_plus_visible);
  if (!(u_minus_visible && u_plus_visible && v_minus_visible && v_plus_visible)) {
    return false;
  }

  *module_u_axis = patch_u_plus - patch_u_minus;
  *module_v_axis = patch_v_plus - patch_v_minus;
  return true;
}

std::pair<cv::Point2f, cv::Point2f> ComputeHomographyLocalAxes(const cv::Mat& board_to_image,
                                                               const CanonicalCorner& corner_info) {
  const cv::Point2f u_minus =
      ProjectBoardPointToImage(board_to_image, corner_info.lattice_u - 0.5, corner_info.lattice_v);
  const cv::Point2f u_plus =
      ProjectBoardPointToImage(board_to_image, corner_info.lattice_u + 0.5, corner_info.lattice_v);
  const cv::Point2f v_minus =
      ProjectBoardPointToImage(board_to_image, corner_info.lattice_u, corner_info.lattice_v - 0.5);
  const cv::Point2f v_plus =
      ProjectBoardPointToImage(board_to_image, corner_info.lattice_u, corner_info.lattice_v + 0.5);
  return {u_plus - u_minus, v_plus - v_minus};
}

bool ComputeVirtualImageAxes(const DoubleSphereCameraModel& camera,
                             const cv::Matx33d& target_to_camera_rotation,
                             const Eigen::Vector3d& target_to_camera_translation,
                             const ApriltagCanonicalModel& model,
                             const CanonicalCorner& corner_info,
                             cv::Point2f* module_u_axis,
                             cv::Point2f* module_v_axis) {
  if (module_u_axis == nullptr || module_v_axis == nullptr) {
    throw std::runtime_error("ComputeVirtualImageAxes requires output pointers.");
  }

  const float half_pitch = static_cast<float>(0.5 * model.Pitch());
  const cv::Point3f center(static_cast<float>(corner_info.target_xyz.x()),
                           static_cast<float>(corner_info.target_xyz.y()),
                           static_cast<float>(corner_info.target_xyz.z()));
  const cv::Point3f u_minus(center.x - half_pitch, center.y, center.z);
  const cv::Point3f u_plus(center.x + half_pitch, center.y, center.z);
  const cv::Point3f v_minus(center.x, center.y - half_pitch, center.z);
  const cv::Point3f v_plus(center.x, center.y + half_pitch, center.z);

  bool u_minus_visible = false;
  bool u_plus_visible = false;
  bool v_minus_visible = false;
  bool v_plus_visible = false;
  const cv::Point2f image_u_minus =
      ProjectTargetPointToImage(camera, target_to_camera_rotation, target_to_camera_translation,
                                u_minus, &u_minus_visible);
  const cv::Point2f image_u_plus =
      ProjectTargetPointToImage(camera, target_to_camera_rotation, target_to_camera_translation,
                                u_plus, &u_plus_visible);
  const cv::Point2f image_v_minus =
      ProjectTargetPointToImage(camera, target_to_camera_rotation, target_to_camera_translation,
                                v_minus, &v_minus_visible);
  const cv::Point2f image_v_plus =
      ProjectTargetPointToImage(camera, target_to_camera_rotation, target_to_camera_translation,
                                v_plus, &v_plus_visible);
  if (!(u_minus_visible && u_plus_visible && v_minus_visible && v_plus_visible)) {
    return false;
  }

  *module_u_axis = image_u_plus - image_u_minus;
  *module_v_axis = image_v_plus - image_v_minus;
  return true;
}

bool BuildSphereLatticeFrame(const DoubleSphereCameraModel& camera,
                             const cv::Matx33d& target_to_camera_rotation,
                             const Eigen::Vector3d& target_to_camera_translation,
                             const ApriltagCanonicalModel& model,
                             const CanonicalCorner& corner_info,
                             SphereLatticeFrame* frame) {
  if (frame == nullptr) {
    throw std::runtime_error("BuildSphereLatticeFrame requires a valid output pointer.");
  }

  const float half_pitch = static_cast<float>(0.5 * model.Pitch());
  const cv::Point3f center(static_cast<float>(corner_info.target_xyz.x()),
                           static_cast<float>(corner_info.target_xyz.y()),
                           static_cast<float>(corner_info.target_xyz.z()));
  const cv::Point3f u_minus(center.x - half_pitch, center.y, center.z);
  const cv::Point3f u_plus(center.x + half_pitch, center.y, center.z);
  const cv::Point3f v_minus(center.x, center.y - half_pitch, center.z);
  const cv::Point3f v_plus(center.x, center.y + half_pitch, center.z);

  bool predicted_visible = false;
  frame->predicted_image = ProjectTargetPointToImage(camera, target_to_camera_rotation,
                                                     target_to_camera_translation, center,
                                                     &predicted_visible);
  if (!predicted_visible || !UnprojectImagePointToRay(camera, frame->predicted_image,
                                                      &frame->predicted_ray)) {
    return false;
  }

  bool u_minus_visible = false;
  bool u_plus_visible = false;
  bool v_minus_visible = false;
  bool v_plus_visible = false;
  const cv::Point2f image_u_minus =
      ProjectTargetPointToImage(camera, target_to_camera_rotation, target_to_camera_translation,
                                u_minus, &u_minus_visible);
  const cv::Point2f image_u_plus =
      ProjectTargetPointToImage(camera, target_to_camera_rotation, target_to_camera_translation,
                                u_plus, &u_plus_visible);
  const cv::Point2f image_v_minus =
      ProjectTargetPointToImage(camera, target_to_camera_rotation, target_to_camera_translation,
                                v_minus, &v_minus_visible);
  const cv::Point2f image_v_plus =
      ProjectTargetPointToImage(camera, target_to_camera_rotation, target_to_camera_translation,
                                v_plus, &v_plus_visible);
  if (!(u_minus_visible && u_plus_visible && v_minus_visible && v_plus_visible)) {
    return false;
  }

  Eigen::Vector3d ray_u_minus = Eigen::Vector3d::Zero();
  Eigen::Vector3d ray_u_plus = Eigen::Vector3d::Zero();
  Eigen::Vector3d ray_v_minus = Eigen::Vector3d::Zero();
  Eigen::Vector3d ray_v_plus = Eigen::Vector3d::Zero();
  if (!UnprojectImagePointToRay(camera, image_u_minus, &ray_u_minus) ||
      !UnprojectImagePointToRay(camera, image_u_plus, &ray_u_plus) ||
      !UnprojectImagePointToRay(camera, image_v_minus, &ray_v_minus) ||
      !UnprojectImagePointToRay(camera, image_v_plus, &ray_v_plus)) {
    return false;
  }

  frame->module_u_axis = image_u_plus - image_u_minus;
  frame->module_v_axis = image_v_plus - image_v_minus;
  frame->local_module_scale =
      ComputeModuleScalePx(frame->module_u_axis, frame->module_v_axis, model.Pitch());

  Eigen::Vector3d raw_tangent_u =
      ProjectOntoTangentPlane(ray_u_plus - ray_u_minus, frame->predicted_ray);
  Eigen::Vector3d raw_tangent_v =
      ProjectOntoTangentPlane(ray_v_plus - ray_v_minus, frame->predicted_ray);
  if (!NormalizeRay(&raw_tangent_u)) {
    return false;
  }
  raw_tangent_v =
      raw_tangent_v - raw_tangent_u * raw_tangent_u.dot(raw_tangent_v);
  if (!NormalizeRay(&raw_tangent_v)) {
    return false;
  }

  if (frame->predicted_ray.dot(raw_tangent_u.cross(raw_tangent_v)) < 0.0) {
    raw_tangent_v = -raw_tangent_v;
  }

  frame->tangent_u = raw_tangent_u;
  frame->tangent_v = raw_tangent_v;
  frame->theta_u = AngleBetweenRays(ray_u_minus, ray_u_plus);
  frame->theta_v = AngleBetweenRays(ray_v_minus, ray_v_plus);
  frame->theta_local = std::min(frame->theta_u, frame->theta_v);
  if (!std::isfinite(frame->theta_local) || frame->theta_local <= 1e-9) {
    return false;
  }
  frame->search_radius =
      std::max(kSphereLatticeSearchRadiusMin, kSphereLatticeSearchRadiusScale * frame->theta_local);
  return true;
}

bool BuildPatchSeedFrame(const DoubleSphereCameraModel& camera,
                         const VirtualPatchContext& context,
                         const ApriltagCanonicalModel& model,
                         const CanonicalCorner& corner_info,
                         PatchSeedFrame* frame) {
  if (frame == nullptr) {
    throw std::runtime_error("BuildPatchSeedFrame requires a valid output pointer.");
  }

  const cv::Point3f center(static_cast<float>(corner_info.target_xyz.x()),
                           static_cast<float>(corner_info.target_xyz.y()),
                           static_cast<float>(corner_info.target_xyz.z()));
  bool predicted_patch_visible = false;
  frame->predicted_patch = ProjectTargetPointToVirtualPatch(context, center, &predicted_patch_visible);

  bool predicted_image_visible = false;
  frame->predicted_image =
      ProjectTargetPointToImage(camera, context.target_to_camera_rotation,
                                context.target_to_camera_translation, center,
                                &predicted_image_visible);
  if (!(predicted_patch_visible && predicted_image_visible)) {
    return false;
  }

  if (!ComputeVirtualPatchAxes(context, corner_info, model.Pitch(), &frame->module_u_axis,
                               &frame->module_v_axis)) {
    return false;
  }

  frame->local_module_scale =
      ComputeModuleScalePx(frame->module_u_axis, frame->module_v_axis,
                           static_cast<double>(model.config().canonical_pixels_per_module));
  if (!std::isfinite(frame->local_module_scale) || frame->local_module_scale < 1.0) {
    return false;
  }

  const double u_norm = std::hypot(frame->module_u_axis.x, frame->module_u_axis.y);
  const double v_norm = std::hypot(frame->module_v_axis.x, frame->module_v_axis.y);
  if (!std::isfinite(u_norm) || !std::isfinite(v_norm) || u_norm <= 1e-9 || v_norm <= 1e-9) {
    return false;
  }
  frame->unit_u = frame->module_u_axis * static_cast<float>(1.0 / u_norm);
  frame->unit_v = frame->module_v_axis * static_cast<float>(1.0 / v_norm);
  frame->search_radius =
      std::max(kPatchSeedSearchRadiusMinPx, kPatchSeedSearchRadiusScale * frame->local_module_scale);
  return true;
}

PatchSeedCandidate EvaluatePatchSeedCandidate(const DoubleSphereCameraModel& camera,
                                              const cv::Mat& patch,
                                              const VirtualPatchContext& context,
                                              const CanonicalCorner& corner_info,
                                              const PatchSeedFrame& frame,
                                              const ApriltagInternalDetectionOptions& options,
                                              double alpha,
                                              double beta) {
  PatchSeedCandidate candidate;
  candidate.alpha = alpha;
  candidate.beta = beta;
  candidate.patch_point = frame.predicted_patch +
                          static_cast<float>(alpha) * frame.unit_u +
                          static_cast<float>(beta) * frame.unit_v;
  if (!IsInsideImage(candidate.patch_point, context.patch_size)) {
    return candidate;
  }

  if (!ProjectVirtualPatchPointToImage(camera, context, candidate.patch_point, &candidate.image_point)) {
    return candidate;
  }

  const TemplateScore score = ComputeBoundaryAlignmentScoreAtPoint(
      patch, corner_info, candidate.patch_point, frame.module_u_axis, frame.module_v_axis,
      options.min_template_contrast, kSphereSeedSampleRadiusScale);
  const double offset = std::hypot(alpha, beta);
  const double search_radius = std::max(kPatchSeedSearchRadiusMinPx, frame.search_radius);
  const double normalized_offset = offset / search_radius;
  const double prior_quality =
      normalized_offset <= kSphereSeedPriorInnerRatio
          ? 1.0
          : ClampUnit(1.0 - (normalized_offset - kSphereSeedPriorInnerRatio) /
                                 std::max(1e-6, 1.0 - kSphereSeedPriorInnerRatio));
  const double raw_quality = std::min(score.template_quality, score.gradient_quality);

  double best_neighbor_raw = 0.0;
  const std::array<cv::Point2f, 8> neighbor_offsets{{
      cv::Point2f(-1.0f, 0.0f), cv::Point2f(1.0f, 0.0f),  cv::Point2f(0.0f, -1.0f),
      cv::Point2f(0.0f, 1.0f),  cv::Point2f(-1.0f, -1.0f), cv::Point2f(1.0f, -1.0f),
      cv::Point2f(-1.0f, 1.0f), cv::Point2f(1.0f, 1.0f),
  }};
  for (const cv::Point2f& neighbor_offset : neighbor_offsets) {
    const cv::Point2f neighbor_patch = candidate.patch_point + neighbor_offset;
    if (!IsInsideImage(neighbor_patch, context.patch_size)) {
      continue;
    }
    const TemplateScore neighbor_score =
        ComputeBoundaryAlignmentScoreAtPoint(
            patch, corner_info, neighbor_patch, frame.module_u_axis, frame.module_v_axis,
            options.min_template_contrast, kSphereSeedSampleRadiusScale);
    best_neighbor_raw =
        std::max(best_neighbor_raw,
                 std::min(neighbor_score.template_quality, neighbor_score.gradient_quality));
  }
  const double peak_quality =
      raw_quality > 1e-9
          ? ClampUnit(0.5 + 0.5 * (raw_quality - best_neighbor_raw) /
                                 std::max(raw_quality, 1e-6))
          : 0.0;

  candidate.valid = true;
  candidate.template_quality = score.template_quality;
  candidate.gradient_quality = score.gradient_quality;
  candidate.prior_quality = prior_quality;
  candidate.peak_quality = peak_quality;
  candidate.raw_quality = raw_quality;
  candidate.final_quality = ClampUnit(kSphereSeedRawWeight * candidate.raw_quality +
                                      kSphereSeedPeakWeight * candidate.peak_quality +
                                      kSphereSeedPriorWeight * candidate.prior_quality);
  return candidate;
}

bool IsBetterPatchCandidate(const PatchSeedCandidate& candidate,
                            const PatchSeedCandidate& reference) {
  if (!candidate.valid) {
    return false;
  }
  if (!reference.valid) {
    return true;
  }
  if (candidate.final_quality > reference.final_quality + 1e-9) {
    return true;
  }
  if (std::abs(candidate.final_quality - reference.final_quality) > 1e-9) {
    return false;
  }
  const double candidate_offset = std::hypot(candidate.alpha, candidate.beta);
  const double reference_offset = std::hypot(reference.alpha, reference.beta);
  return candidate_offset < reference_offset;
}

PatchSeedCandidate SearchPatchSeedGrid(const DoubleSphereCameraModel& camera,
                                       const cv::Mat& patch,
                                       const VirtualPatchContext& context,
                                       const CanonicalCorner& corner_info,
                                       const PatchSeedFrame& frame,
                                       const ApriltagInternalDetectionOptions& options,
                                       double center_alpha,
                                       double center_beta,
                                       double radius,
                                       int grid_size) {
  PatchSeedCandidate best_candidate =
      EvaluatePatchSeedCandidate(camera, patch, context, corner_info, frame, options,
                                 center_alpha, center_beta);
  if (grid_size <= 1) {
    return best_candidate;
  }

  const double denominator = static_cast<double>(grid_size - 1);
  for (int row = 0; row < grid_size; ++row) {
    for (int col = 0; col < grid_size; ++col) {
      const double alpha =
          center_alpha - radius + 2.0 * radius * static_cast<double>(col) / denominator;
      const double beta =
          center_beta - radius + 2.0 * radius * static_cast<double>(row) / denominator;
      const PatchSeedCandidate candidate =
          EvaluatePatchSeedCandidate(camera, patch, context, corner_info, frame, options,
                                     alpha, beta);
      if (IsBetterPatchCandidate(candidate, best_candidate)) {
        best_candidate = candidate;
      }
    }
  }
  return best_candidate;
}

PatchSeedCandidate SearchVirtualPatchBoundarySeed(const DoubleSphereCameraModel& camera,
                                                  const cv::Mat& patch,
                                                  const VirtualPatchContext& context,
                                                  const CanonicalCorner& corner_info,
                                                  const PatchSeedFrame& frame,
                                                  const ApriltagInternalDetectionOptions& options) {
  const PatchSeedCandidate coarse_best =
      SearchPatchSeedGrid(camera, patch, context, corner_info, frame, options, 0.0, 0.0,
                          frame.search_radius, kPatchSeedCoarseGridSize);
  const double fine_radius = 0.5 * frame.search_radius;
  return SearchPatchSeedGrid(camera, patch, context, corner_info, frame, options,
                             coarse_best.alpha, coarse_best.beta, fine_radius,
                             kPatchSeedFineGridSize);
}

SphereSeedCandidate EvaluateSphereSeedCandidate(const cv::Mat& gray,
                                                const DoubleSphereCameraModel& camera,
                                                const CanonicalCorner& corner_info,
                                                const SphereLatticeFrame& frame,
                                                const ApriltagInternalDetectionOptions& options,
                                                double alpha,
                                                double beta) {
  SphereSeedCandidate candidate;
  candidate.alpha = alpha;
  candidate.beta = beta;

  Eigen::Vector3d candidate_ray =
      frame.predicted_ray + alpha * frame.tangent_u + beta * frame.tangent_v;
  if (!NormalizeRay(&candidate_ray)) {
    return candidate;
  }

  cv::Point2f candidate_image;
  if (!ProjectRayToImage(camera, candidate_ray, &candidate_image) ||
      !IsInsideImage(candidate_image, gray.size())) {
    return candidate;
  }

  const TemplateScore score = ComputeBoundaryAlignmentScoreAtPoint(
      gray, corner_info, candidate_image, frame.module_u_axis, frame.module_v_axis,
      options.min_template_contrast, kSphereSeedSampleRadiusScale);
  const double offset = std::hypot(alpha, beta);
  const double search_radius = std::max(kSphereLatticeSearchRadiusMin, frame.search_radius);
  const double normalized_offset = offset / search_radius;
  const double prior_quality =
      normalized_offset <= kSphereSeedPriorInnerRatio
          ? 1.0
          : ClampUnit(1.0 - (normalized_offset - kSphereSeedPriorInnerRatio) /
                                 std::max(1e-6, 1.0 - kSphereSeedPriorInnerRatio));
  const double raw_quality = std::min(score.template_quality, score.gradient_quality);

  double best_neighbor_raw = 0.0;
  const std::array<cv::Point2f, 8> neighbor_offsets{{
      cv::Point2f(-1.0f, 0.0f), cv::Point2f(1.0f, 0.0f),  cv::Point2f(0.0f, -1.0f),
      cv::Point2f(0.0f, 1.0f),  cv::Point2f(-1.0f, -1.0f), cv::Point2f(1.0f, -1.0f),
      cv::Point2f(-1.0f, 1.0f), cv::Point2f(1.0f, 1.0f),
  }};
  for (const cv::Point2f& neighbor_offset : neighbor_offsets) {
    const cv::Point2f neighbor_image = candidate_image + neighbor_offset;
    if (!IsInsideImage(neighbor_image, gray.size())) {
      continue;
    }
    const TemplateScore neighbor_score =
        ComputeBoundaryAlignmentScoreAtPoint(gray, corner_info, neighbor_image, frame.module_u_axis,
                                             frame.module_v_axis, options.min_template_contrast,
                                             kSphereSeedSampleRadiusScale);
    best_neighbor_raw =
        std::max(best_neighbor_raw,
                 std::min(neighbor_score.template_quality, neighbor_score.gradient_quality));
  }
  const double peak_quality =
      raw_quality > 1e-9
          ? ClampUnit(0.5 + 0.5 * (raw_quality - best_neighbor_raw) / std::max(raw_quality, 1e-6))
          : 0.0;

  candidate.valid = true;
  candidate.image_point = candidate_image;
  candidate.ray = candidate_ray;
  candidate.template_quality = score.template_quality;
  candidate.gradient_quality = score.gradient_quality;
  candidate.prior_quality = prior_quality;
  candidate.peak_quality = peak_quality;
  candidate.raw_quality = raw_quality;
  candidate.final_quality = ClampUnit(kSphereSeedRawWeight * candidate.raw_quality +
                                      kSphereSeedPeakWeight * candidate.peak_quality +
                                      kSphereSeedPriorWeight * candidate.prior_quality);
  return candidate;
}

bool IsBetterSphereCandidate(const SphereSeedCandidate& candidate,
                             const SphereSeedCandidate& reference) {
  if (!candidate.valid) {
    return false;
  }
  if (!reference.valid) {
    return true;
  }
  if (candidate.final_quality > reference.final_quality + 1e-9) {
    return true;
  }
  if (std::abs(candidate.final_quality - reference.final_quality) > 1e-9) {
    return false;
  }
  const double candidate_offset = std::hypot(candidate.alpha, candidate.beta);
  const double reference_offset = std::hypot(reference.alpha, reference.beta);
  return candidate_offset < reference_offset;
}

SphereSeedCandidate SearchSphereSeedGrid(const cv::Mat& gray,
                                         const DoubleSphereCameraModel& camera,
                                         const CanonicalCorner& corner_info,
                                         const SphereLatticeFrame& frame,
                                         const ApriltagInternalDetectionOptions& options,
                                         double center_alpha,
                                         double center_beta,
                                         double radius,
                                         int grid_size) {
  SphereSeedCandidate best_candidate =
      EvaluateSphereSeedCandidate(gray, camera, corner_info, frame, options, center_alpha, center_beta);
  if (grid_size <= 1) {
    return best_candidate;
  }

  const double denominator = static_cast<double>(grid_size - 1);
  for (int row = 0; row < grid_size; ++row) {
    for (int col = 0; col < grid_size; ++col) {
      const double alpha = center_alpha - radius + 2.0 * radius * static_cast<double>(col) / denominator;
      const double beta = center_beta - radius + 2.0 * radius * static_cast<double>(row) / denominator;
      const SphereSeedCandidate candidate =
          EvaluateSphereSeedCandidate(gray, camera, corner_info, frame, options, alpha, beta);
      if (IsBetterSphereCandidate(candidate, best_candidate)) {
        best_candidate = candidate;
      }
    }
  }
  return best_candidate;
}

SphereSeedCandidate SearchSphereLatticeSeed(const cv::Mat& gray,
                                            const DoubleSphereCameraModel& camera,
                                            const CanonicalCorner& corner_info,
                                            const SphereLatticeFrame& frame,
                                            const ApriltagInternalDetectionOptions& options) {
  const SphereSeedCandidate coarse_best =
      SearchSphereSeedGrid(gray, camera, corner_info, frame, options, 0.0, 0.0,
                           frame.search_radius, kSphereLatticeCoarseGridSize);
  const double fine_radius = 0.5 * frame.search_radius;
  return SearchSphereSeedGrid(gray, camera, corner_info, frame, options, coarse_best.alpha,
                              coarse_best.beta, fine_radius, kSphereLatticeFineGridSize);
}

bool SlerpRays(const Eigen::Vector3d& start_ray,
               const Eigen::Vector3d& end_ray,
               double alpha,
               Eigen::Vector3d* interpolated_ray) {
  if (interpolated_ray == nullptr) {
    throw std::runtime_error("SlerpRays requires a valid output pointer.");
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
  if (!std::isfinite(theta) || !std::isfinite(sin_theta) || std::abs(sin_theta) <= 1e-9) {
    return false;
  }

  *interpolated_ray =
      (std::sin((1.0 - alpha) * theta) / sin_theta) * start +
      (std::sin(alpha * theta) / sin_theta) * end;
  return NormalizeRay(interpolated_ray);
}

bool BuildSeedAnchoredSphereLatticeFrame(const SphereLatticeFrame& base_frame,
                                         const Eigen::Vector3d& anchor_ray,
                                         const cv::Point2f& anchor_image,
                                         double search_radius,
                                         SphereLatticeFrame* anchored_frame) {
  if (anchored_frame == nullptr) {
    throw std::runtime_error("BuildSeedAnchoredSphereLatticeFrame requires a valid output pointer.");
  }

  *anchored_frame = base_frame;
  anchored_frame->predicted_ray = anchor_ray;
  anchored_frame->predicted_image = anchor_image;
  anchored_frame->search_radius = search_radius;

  Eigen::Vector3d anchored_tangent_u =
      ProjectOntoTangentPlane(base_frame.tangent_u, anchor_ray);
  if (!NormalizeRay(&anchored_tangent_u)) {
    return false;
  }

  Eigen::Vector3d anchored_tangent_v =
      ProjectOntoTangentPlane(base_frame.tangent_v, anchor_ray);
  anchored_tangent_v =
      anchored_tangent_v - anchored_tangent_u * anchored_tangent_u.dot(anchored_tangent_v);
  if (!NormalizeRay(&anchored_tangent_v)) {
    return false;
  }

  if (anchor_ray.dot(anchored_tangent_u.cross(anchored_tangent_v)) < 0.0) {
    anchored_tangent_v = -anchored_tangent_v;
  }

  anchored_frame->tangent_u = anchored_tangent_u;
  anchored_frame->tangent_v = anchored_tangent_v;
  return true;
}

std::vector<cv::Point2f> CollectBoardEdgeSupportPoints(const OuterTagDetectionResult& outer_detection,
                                                       int edge_index) {
  const auto& debug = outer_detection.corner_verification_debug;
  auto select_supports = [](const OuterCornerVerificationDebugInfo& corner_debug,
                            bool use_next) -> const std::vector<cv::Point2f>& {
    const std::vector<cv::Point2f>& explicit_supports =
        use_next ? corner_debug.next_marker_support_points : corner_debug.prev_marker_support_points;
    if (!explicit_supports.empty()) {
      return explicit_supports;
    }
    return use_next ? corner_debug.next_branch_points : corner_debug.prev_branch_points;
  };

  std::vector<cv::Point2f> support_points;
  auto append_supports = [&support_points](const std::vector<cv::Point2f>& points) {
    support_points.insert(support_points.end(), points.begin(), points.end());
  };

  switch (edge_index) {
    case 0:
      append_supports(select_supports(debug[0], true));
      append_supports(select_supports(debug[1], false));
      break;
    case 1:
      append_supports(select_supports(debug[1], true));
      append_supports(select_supports(debug[2], false));
      break;
    case 2:
      append_supports(select_supports(debug[2], true));
      append_supports(select_supports(debug[3], false));
      break;
    case 3:
      append_supports(select_supports(debug[3], true));
      append_supports(select_supports(debug[0], false));
      break;
    default:
      break;
  }
  return support_points;
}

bool UnprojectSupportPointsToRays(const DoubleSphereCameraModel& camera,
                                  const std::vector<cv::Point2f>& image_points,
                                  std::vector<Eigen::Vector3d>* rays) {
  if (rays == nullptr) {
    throw std::runtime_error("UnprojectSupportPointsToRays requires a valid output pointer.");
  }

  rays->clear();
  rays->reserve(image_points.size());
  for (const cv::Point2f& image_point : image_points) {
    Eigen::Vector3d ray = Eigen::Vector3d::Zero();
    if (UnprojectImagePointToRay(camera, image_point, &ray)) {
      rays->push_back(ray);
    }
  }
  return rays->size() >= static_cast<std::size_t>(kBorderConditionedMinSupportPoints);
}

bool FitBoundaryPlaneToRays(const std::vector<Eigen::Vector3d>& rays,
                            Eigen::Vector3d* plane_normal,
                            double* rms_residual) {
  if (plane_normal == nullptr || rms_residual == nullptr) {
    throw std::runtime_error("FitBoundaryPlaneToRays requires valid output pointers.");
  }
  if (rays.size() < static_cast<std::size_t>(kBorderConditionedMinSupportPoints)) {
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
  if (!NormalizeRay(&normal)) {
    return false;
  }

  double residual_sum_sq = 0.0;
  for (const Eigen::Vector3d& ray : rays) {
    const double residual = normal.dot(ray);
    residual_sum_sq += residual * residual;
  }

  *plane_normal = normal;
  *rms_residual = std::sqrt(residual_sum_sq / static_cast<double>(rays.size()));
  return std::isfinite(*rms_residual) &&
         *rms_residual <= kBorderConditionedPlaneResidualThreshold;
}

bool ProjectRayOntoBoundaryPlane(const Eigen::Vector3d& plane_normal,
                                 const Eigen::Vector3d& reference_ray,
                                 Eigen::Vector3d* projected_ray) {
  if (projected_ray == nullptr) {
    throw std::runtime_error("ProjectRayOntoBoundaryPlane requires a valid output pointer.");
  }

  Eigen::Vector3d candidate = reference_ray - plane_normal * plane_normal.dot(reference_ray);
  if (!NormalizeRay(&candidate)) {
    return false;
  }
  if (candidate.dot(reference_ray) < 0.0) {
    candidate = -candidate;
  }
  *projected_ray = candidate;
  return true;
}

bool BuildBoardSphereBoundaryModel(const DoubleSphereCameraModel& camera,
                                   const OuterTagDetectionResult& outer_detection,
                                   BoardSphereBoundaryModel* boundary_model) {
  if (boundary_model == nullptr) {
    throw std::runtime_error("BuildBoardSphereBoundaryModel requires a valid output pointer.");
  }

  *boundary_model = BoardSphereBoundaryModel{};
  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    const Eigen::Vector2d& outer_corner =
        outer_detection.refined_corners_original_image[static_cast<std::size_t>(corner_index)];
    if (!UnprojectImagePointToRay(camera,
                                  cv::Point2f(static_cast<float>(outer_corner.x()),
                                              static_cast<float>(outer_corner.y())),
                                  &boundary_model->outer_corner_rays[static_cast<std::size_t>(corner_index)])) {
      return false;
    }
  }

  const std::array<int, 4> edge_start_corner{{0, 1, 3, 0}};
  const std::array<int, 4> edge_end_corner{{1, 2, 2, 3}};
  for (int edge_index = 0; edge_index < 4; ++edge_index) {
    BoardBoundaryEdgeModel& edge = boundary_model->edges[static_cast<std::size_t>(edge_index)];
    edge.support_points = CollectBoardEdgeSupportPoints(outer_detection, edge_index);
    if (!UnprojectSupportPointsToRays(camera, edge.support_points, &edge.support_rays)) {
      return false;
    }
    if (!FitBoundaryPlaneToRays(edge.support_rays, &edge.plane_normal, &edge.rms_residual)) {
      return false;
    }
    if (!ProjectRayOntoBoundaryPlane(
            edge.plane_normal,
            boundary_model->outer_corner_rays[static_cast<std::size_t>(edge_start_corner[edge_index])],
            &edge.start_ray) ||
        !ProjectRayOntoBoundaryPlane(
            edge.plane_normal,
            boundary_model->outer_corner_rays[static_cast<std::size_t>(edge_end_corner[edge_index])],
            &edge.end_ray)) {
      return false;
    }
    edge.valid = true;
  }

  boundary_model->valid = true;
  return true;
}

bool SampleBoundaryEdgeRay(const BoardSphereBoundaryModel& boundary_model,
                           int edge_index,
                           double alpha,
                           Eigen::Vector3d* ray);

void StoreBoardBoundaryDebugCurves(const BoardSphereBoundaryModel& boundary_model,
                                   const DoubleSphereCameraModel& camera,
                                   ApriltagInternalDetectionResult* result) {
  if (result == nullptr) {
    throw std::runtime_error("StoreBoardBoundaryDebugCurves requires a valid result pointer.");
  }

  result->border_boundary_model_valid = false;
  result->border_edge_valid = {{false, false, false, false}};
  result->border_edge_rms_residual = {{0.0, 0.0, 0.0, 0.0}};
  result->border_edge_support_count = {{0, 0, 0, 0}};
  for (std::size_t edge_index = 0; edge_index < result->border_curves_image.size(); ++edge_index) {
    result->border_support_points[edge_index].clear();
    result->border_curves_image[edge_index].clear();
    result->border_curves_ray[edge_index].clear();
  }
  if (!boundary_model.valid) {
    return;
  }

  for (int edge_index = 0; edge_index < 4; ++edge_index) {
    const BoardBoundaryEdgeModel& edge =
        boundary_model.edges[static_cast<std::size_t>(edge_index)];
    result->border_edge_valid[static_cast<std::size_t>(edge_index)] = edge.valid;
    result->border_edge_rms_residual[static_cast<std::size_t>(edge_index)] = edge.rms_residual;
    result->border_edge_support_count[static_cast<std::size_t>(edge_index)] =
        static_cast<int>(edge.support_points.size());
    result->border_support_points[static_cast<std::size_t>(edge_index)] = edge.support_points;

    std::vector<cv::Point2f>& image_curve =
        result->border_curves_image[static_cast<std::size_t>(edge_index)];
    std::vector<cv::Vec3d>& ray_curve =
        result->border_curves_ray[static_cast<std::size_t>(edge_index)];
    image_curve.reserve(kBorderBoundaryCurveSampleCount);
    ray_curve.reserve(kBorderBoundaryCurveSampleCount);

    for (int sample_index = 0; sample_index < kBorderBoundaryCurveSampleCount; ++sample_index) {
      const double alpha =
          kBorderBoundaryCurveSampleCount <= 1
              ? 0.0
              : static_cast<double>(sample_index) /
                    static_cast<double>(kBorderBoundaryCurveSampleCount - 1);
      Eigen::Vector3d ray = Eigen::Vector3d::Zero();
      if (!SampleBoundaryEdgeRay(boundary_model, edge_index, alpha, &ray)) {
        continue;
      }
      ray_curve.emplace_back(ray.x(), ray.y(), ray.z());

      cv::Point2f image_point{};
      if (ProjectRayToImage(camera, ray, &image_point)) {
        image_curve.push_back(image_point);
      }
    }
  }

  result->border_boundary_model_valid = true;
}

bool SampleBoundaryEdgeRay(const BoardSphereBoundaryModel& boundary_model,
                           int edge_index,
                           double alpha,
                           Eigen::Vector3d* ray) {
  if (edge_index < 0 || edge_index >= 4) {
    return false;
  }
  const BoardBoundaryEdgeModel& edge = boundary_model.edges[static_cast<std::size_t>(edge_index)];
  if (!edge.valid) {
    return false;
  }
  return SlerpRays(edge.start_ray, edge.end_ray, alpha, ray);
}

bool BuildBorderConditionedSeed(const BoardSphereBoundaryModel& boundary_model,
                                const DoubleSphereCameraModel& camera,
                                const ApriltagCanonicalModel& model,
                                const CanonicalCorner& corner_info,
                                BorderConditionedSeed* border_seed) {
  if (border_seed == nullptr) {
    throw std::runtime_error("BuildBorderConditionedSeed requires a valid output pointer.");
  }

  *border_seed = BorderConditionedSeed{};
  if (!boundary_model.valid) {
    return false;
  }

  const double module_dimension = static_cast<double>(model.ModuleDimension());
  if (!std::isfinite(module_dimension) || module_dimension <= 0.0) {
    return false;
  }
  const double s = static_cast<double>(corner_info.lattice_u) / module_dimension;
  const double t = static_cast<double>(corner_info.lattice_v) / module_dimension;

  if (!SampleBoundaryEdgeRay(boundary_model, 0, s, &border_seed->top_ray) ||
      !SampleBoundaryEdgeRay(boundary_model, 2, s, &border_seed->bottom_ray) ||
      !SampleBoundaryEdgeRay(boundary_model, 3, t, &border_seed->left_ray) ||
      !SampleBoundaryEdgeRay(boundary_model, 1, t, &border_seed->right_ray)) {
    return false;
  }

  Eigen::Vector3d vertical_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d horizontal_ray = Eigen::Vector3d::Zero();
  if (!SlerpRays(border_seed->top_ray, border_seed->bottom_ray, t, &vertical_ray) ||
      !SlerpRays(border_seed->left_ray, border_seed->right_ray, s, &horizontal_ray)) {
    return false;
  }

  border_seed->ray = vertical_ray + horizontal_ray;
  if (!NormalizeRay(&border_seed->ray)) {
    return false;
  }
  if (!ProjectRayToImage(camera, border_seed->ray, &border_seed->image_point)) {
    return false;
  }

  border_seed->valid = true;
  return true;
}

double ComputeAdaptiveBorderSeedSearchRadius(const SphereLatticeFrame& frame,
                                             const Eigen::Vector3d& border_seed_ray) {
  const double base_radius = std::max(kSphereLatticeSearchRadiusMin, frame.search_radius);
  const double disagreement = AngleBetweenRays(frame.predicted_ray, border_seed_ray);
  const double expanded_radius =
      std::max(base_radius, kBorderConditionedSearchRadiusGain * disagreement);
  return std::max(base_radius,
                  std::min(kBorderConditionedSearchRadiusMaxScale * base_radius, expanded_radius));
}

bool IntersectVirtualPatchPixelWithTargetPlane(const VirtualPatchContext& context,
                                               const cv::Point2f& patch_point,
                                               Eigen::Vector3d* point_camera) {
  if (point_camera == nullptr) {
    throw std::runtime_error("IntersectVirtualPatchPixelWithTargetPlane requires an output pointer.");
  }

  const Eigen::Vector3d ray_virtual((patch_point.x - context.cu) / context.fu,
                                    (patch_point.y - context.cv) / context.fv, 1.0);
  const Eigen::Vector3d ray_camera = Multiply(context.virtual_to_camera_rotation, ray_virtual);
  const double denominator = context.plane_normal_camera.dot(ray_camera);
  if (std::abs(denominator) < 1e-9) {
    return false;
  }

  const double scale = context.plane_normal_camera.dot(context.plane_point_camera) / denominator;
  if (scale <= 0.0) {
    return false;
  }

  *point_camera = scale * ray_camera;
  return true;
}

VirtualPatchContext BuildVirtualPatchContext(const cv::Mat& gray,
                                             const DoubleSphereCameraModel& camera,
                                             const cv::Mat& rvec,
                                             const cv::Mat& tvec,
                                             const ApriltagCanonicalModel& model,
                                             const std::array<int, 4>& outer_point_ids,
                                             const ApriltagInternalDetectionOptions& options) {
  VirtualPatchContext context;
  cv::Rodrigues(rvec, context.target_to_camera_rotation);
  context.target_to_camera_translation = MatToEigenVector3d(tvec);
  context.plane_normal_camera = Eigen::Vector3d(context.target_to_camera_rotation(0, 2),
                                                context.target_to_camera_rotation(1, 2),
                                                context.target_to_camera_rotation(2, 2));
  context.plane_point_camera = context.target_to_camera_translation;

  const double half_extent = 0.5 * model.ModuleDimension() * model.Pitch();
  const Eigen::Vector3d target_center(half_extent, half_extent, 0.0);
  const Eigen::Vector3d center_camera =
      Multiply(context.target_to_camera_rotation, target_center) + context.target_to_camera_translation;

  Eigen::Vector3d x_axis_camera(context.target_to_camera_rotation(0, 0),
                                context.target_to_camera_rotation(1, 0),
                                context.target_to_camera_rotation(2, 0));
  Eigen::Vector3d y_axis_camera(context.target_to_camera_rotation(0, 1),
                                context.target_to_camera_rotation(1, 1),
                                context.target_to_camera_rotation(2, 1));
  Eigen::Vector3d z_axis_virtual = center_camera.normalized();
  Eigen::Vector3d x_axis_virtual =
      x_axis_camera - x_axis_camera.dot(z_axis_virtual) * z_axis_virtual;
  if (x_axis_virtual.norm() < 1e-9) {
    x_axis_virtual = y_axis_camera - y_axis_camera.dot(z_axis_virtual) * z_axis_virtual;
  }
  x_axis_virtual.normalize();
  Eigen::Vector3d y_axis_virtual = z_axis_virtual.cross(x_axis_virtual).normalized();
  if (y_axis_virtual.dot(y_axis_camera) < 0.0) {
    x_axis_virtual = -x_axis_virtual;
    y_axis_virtual = -y_axis_virtual;
  }

  context.camera_to_virtual_rotation = cv::Matx33d(
      x_axis_virtual.x(), x_axis_virtual.y(), x_axis_virtual.z(),
      y_axis_virtual.x(), y_axis_virtual.y(), y_axis_virtual.z(),
      z_axis_virtual.x(), z_axis_virtual.y(), z_axis_virtual.z());
  context.virtual_to_camera_rotation = context.camera_to_virtual_rotation.t();

  const int patch_extent = model.ModuleDimension() * options.canonical_pixels_per_module;
  context.patch_size = cv::Size(patch_extent + 1, patch_extent + 1);
  context.cu = 0.5 * static_cast<double>(context.patch_size.width - 1);
  context.cv = 0.5 * static_cast<double>(context.patch_size.height - 1);

  double max_abs_x = 1e-3;
  double max_abs_y = 1e-3;
  for (int index = 0; index < 4; ++index) {
    const CanonicalCorner& corner_info = model.corner(outer_point_ids[index]);
    const Eigen::Vector3d point_camera = TargetToCameraPoint(
        cv::Point3f(static_cast<float>(corner_info.target_xyz.x()),
                    static_cast<float>(corner_info.target_xyz.y()),
                    static_cast<float>(corner_info.target_xyz.z())),
        context.target_to_camera_rotation, context.target_to_camera_translation);
    const Eigen::Vector3d point_virtual =
        Multiply(context.camera_to_virtual_rotation, point_camera);
    if (point_virtual.z() <= 1e-9) {
      continue;
    }
    max_abs_x = std::max(max_abs_x, std::abs(point_virtual.x() / point_virtual.z()));
    max_abs_y = std::max(max_abs_y, std::abs(point_virtual.y() / point_virtual.z()));
  }

  context.fu = 0.5 * static_cast<double>(context.patch_size.width - 1) /
               (max_abs_x * options.virtual_patch_margin);
  context.fv = 0.5 * static_cast<double>(context.patch_size.height - 1) /
               (max_abs_y * options.virtual_patch_margin);

  cv::Mat map_x(context.patch_size, CV_32FC1);
  cv::Mat map_y(context.patch_size, CV_32FC1);

  // TartanCalib-style intermediate image: build a local pinhole ray grid, then
  // remap it through the intermediate DS camera model back into the distorted image.
  for (int y = 0; y < context.patch_size.height; ++y) {
    for (int x = 0; x < context.patch_size.width; ++x) {
      const Eigen::Vector3d ray_virtual((static_cast<double>(x) - context.cu) / context.fu,
                                        (static_cast<double>(y) - context.cv) / context.fv, 1.0);
      const Eigen::Vector3d ray_camera = Multiply(context.virtual_to_camera_rotation, ray_virtual);
      Eigen::Vector2d image_point = Eigen::Vector2d::Zero();
      if (camera.vsEuclideanToKeypoint(ray_camera, &image_point)) {
        map_x.at<float>(y, x) = static_cast<float>(image_point.x());
        map_y.at<float>(y, x) = static_cast<float>(image_point.y());
      } else {
        map_x.at<float>(y, x) = -1.0f;
        map_y.at<float>(y, x) = -1.0f;
      }
    }
  }

  cv::remap(gray, context.patch, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));

  for (int index = 0; index < 4; ++index) {
    const CanonicalCorner& corner_info = model.corner(outer_point_ids[index]);
    bool visible = false;
    context.outer_patch_corners[index] =
        ProjectTargetPointToVirtualPatch(
            context,
            cv::Point3f(static_cast<float>(corner_info.target_xyz.x()),
                        static_cast<float>(corner_info.target_xyz.y()),
                        static_cast<float>(corner_info.target_xyz.z())),
            &visible);
  }

  return context;
}

cv::Scalar InternalCornerColor(const InternalCornerDebugInfo& debug) {
  if (!debug.valid) {
    return cv::Scalar(0, 64, 255);
  }
  if (debug.corner_type == CornerType::XCorner) {
    return cv::Scalar(255, 0, 255);
  }
  return cv::Scalar(0, 220, 0);
}

void PopulateInternalCornersFromHomography(const cv::Mat& gray,
                                           const cv::Mat& board_to_image,
                                           const cv::Mat& image_to_patch,
                                           const std::array<int, 4>& outer_point_ids,
                                           const ApriltagCanonicalModel& model,
                                           const ApriltagInternalDetectionOptions& options,
                                           ApriltagInternalDetectionResult* result) {
  if (result == nullptr) {
    throw std::runtime_error("Result pointer must not be null.");
  }

  for (const int point_id : model.VisiblePointIds()) {
    if (std::find(outer_point_ids.begin(), outer_point_ids.end(), point_id) != outer_point_ids.end()) {
      continue;
    }

    const CanonicalCorner& corner_info = model.corner(point_id);
    const cv::Point2f predicted_image = PerspectiveTransformPoint(
        board_to_image, cv::Point2f(static_cast<float>(corner_info.lattice_u),
                                    static_cast<float>(corner_info.lattice_v)));
    const cv::Point2f predicted_patch =
        BoardToPatchPoint(corner_info, model.ModuleDimension(), options.canonical_pixels_per_module);
    const std::pair<cv::Point2f, cv::Point2f> local_axes =
        ComputeHomographyLocalAxes(board_to_image, corner_info);
    const cv::Point2f module_u_axis = local_axes.first;
    const cv::Point2f module_v_axis = local_axes.second;
    const double module_scale_px =
        ComputeModuleScalePx(module_u_axis, module_v_axis,
                             static_cast<double>(options.canonical_pixels_per_module));
    const int subpix_window_radius =
        ComputeAdaptiveInternalSubpixRadius(module_scale_px, options);
    const double subpix_displacement_limit =
        ComputeAdaptiveInternalSubpixDisplacementLimit(module_scale_px, options);
    const int image_evidence_search_radius =
        ComputeAdaptiveImageEvidenceSearchRadius(module_scale_px, options);

    cv::Point2f refined_image = predicted_image;
    if (IsInsideImageWithBorder(predicted_image, gray.size(), options.min_border_distance) &&
        options.do_subpix_refinement) {
      std::vector<cv::Point2f> corners{refined_image};
      cv::cornerSubPix(gray, corners,
                       cv::Size(subpix_window_radius, subpix_window_radius),
                       cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
      refined_image = corners.front();
    }

    const cv::Point2f refined_patch = PerspectiveTransformPoint(image_to_patch, refined_image);
    const double displacement =
        std::hypot(refined_image.x - predicted_image.x, refined_image.y - predicted_image.y);
    const double q_refine = IsInsideImageWithBorder(predicted_image, gray.size(), options.min_border_distance)
                                ? (options.do_subpix_refinement
                                       ? ClampUnit(1.0 - (displacement * displacement) /
                                                           std::max(1e-9, subpix_displacement_limit *
                                                                             subpix_displacement_limit))
                                       : 1.0)
                                : 0.0;
    const TemplateScore template_score =
        ComputeTemplateScoreAtPoint(result->canonical_patch, corner_info, refined_patch,
                                    options.canonical_pixels_per_module, options.min_template_contrast);
    const ImageEvidenceScore image_score =
        EvaluateImageEvidenceAroundPoint(gray, corner_info, refined_image, module_u_axis, module_v_axis,
                                         options.min_template_contrast,
                                         subpix_displacement_limit * subpix_displacement_limit,
                                         image_evidence_search_radius);
    const double final_quality =
        std::min({template_score.template_quality, template_score.gradient_quality, q_refine});
    const double image_final_quality = image_score.final_quality;
    const bool valid =
        IsInsideImageWithBorder(refined_image, gray.size(), options.min_border_distance);
    const bool image_evidence_valid =
        IsInsideImageWithBorder(refined_image, gray.size(), options.min_border_distance) &&
        image_final_quality >= options.min_quality;

    CornerMeasurement& measurement = result->corners[static_cast<std::size_t>(point_id)];
    measurement.image_xy = Eigen::Vector2d(refined_image.x, refined_image.y);
    measurement.quality = final_quality;
    measurement.valid = valid;

    if (valid) {
      ++result->valid_corner_count;
      ++result->valid_internal_corner_count;
    }

    InternalCornerDebugInfo debug;
    debug.point_id = point_id;
    debug.corner_type = corner_info.corner_type;
    debug.predicted_image = predicted_image;
    debug.refined_image = refined_image;
    debug.predicted_patch = predicted_patch;
    debug.refined_patch = refined_patch;
    debug.module_u_axis = module_u_axis;
    debug.module_v_axis = module_v_axis;
    debug.local_module_scale = module_scale_px;
    debug.subpix_window_radius = subpix_window_radius;
    debug.subpix_displacement_limit = subpix_displacement_limit;
    debug.image_evidence_search_radius = image_evidence_search_radius;
    debug.q_refine = q_refine;
    debug.template_quality = template_score.template_quality;
    debug.gradient_quality = template_score.gradient_quality;
    debug.final_quality = final_quality;
    debug.image_template_quality = image_score.best_score.template_quality;
    debug.image_gradient_quality = image_score.best_score.gradient_quality;
    debug.image_centering_quality = image_score.centering_quality;
    debug.image_final_quality = image_final_quality;
    debug.predicted_to_refined_displacement = displacement;
    debug.valid = valid;
    debug.image_evidence_valid = image_evidence_valid;
    result->internal_corner_debug.push_back(debug);
  }
}

void PopulateInternalCornersFromSphereLattice(const cv::Mat& gray,
                                              const DoubleSphereCameraModel& camera,
                                              const cv::Matx33d& target_to_camera_rotation,
                                              const Eigen::Vector3d& target_to_camera_translation,
                                              const std::array<int, 4>& outer_point_ids,
                                              const OuterTagDetectionResult& outer_detection,
                                              const ApriltagCanonicalModel& model,
                                              const ApriltagInternalDetectionOptions& options,
                                              bool enable_seed_search,
                                              bool use_border_conditioned_seed,
                                              bool use_ray_domain_refine,
                                              ApriltagInternalDetectionResult* result) {
  if (result == nullptr) {
    throw std::runtime_error("Result pointer must not be null.");
  }

  BoardSphereBoundaryModel boundary_model;
  const bool has_boundary_model =
      use_border_conditioned_seed &&
      BuildBoardSphereBoundaryModel(camera, outer_detection, &boundary_model);
  StoreBoardBoundaryDebugCurves(boundary_model, camera, result);

  for (const int point_id : model.VisiblePointIds()) {
    if (std::find(outer_point_ids.begin(), outer_point_ids.end(), point_id) != outer_point_ids.end()) {
      continue;
    }

    const CanonicalCorner& corner_info = model.corner(point_id);
    InternalCornerDebugInfo debug;
    debug.point_id = point_id;
    debug.corner_type = corner_info.corner_type;

    SphereLatticeFrame frame;
    const bool has_frame =
        BuildSphereLatticeFrame(camera, target_to_camera_rotation, target_to_camera_translation,
                                model, corner_info, &frame);
    debug.predicted_image = frame.predicted_image;
    debug.predicted_ray = cv::Vec3d(frame.predicted_ray.x(), frame.predicted_ray.y(),
                                    frame.predicted_ray.z());
    debug.tangent_u_ray = cv::Vec3d(frame.tangent_u.x(), frame.tangent_u.y(), frame.tangent_u.z());
    debug.tangent_v_ray = cv::Vec3d(frame.tangent_v.x(), frame.tangent_v.y(), frame.tangent_v.z());
    debug.module_u_axis = frame.module_u_axis;
    debug.module_v_axis = frame.module_v_axis;
    debug.local_module_scale = frame.local_module_scale;
    debug.sphere_search_radius = frame.search_radius;
    debug.adaptive_search_radius = frame.search_radius;

    cv::Point2f border_seed_image{};
    Eigen::Vector3d border_seed_ray = Eigen::Vector3d::Zero();
    SphereLatticeFrame search_frame = frame;
    bool has_search_anchor = has_frame;
    if (use_border_conditioned_seed) {
      has_search_anchor = false;
      BorderConditionedSeed border_seed;
      if (has_frame && has_boundary_model &&
          BuildBorderConditionedSeed(boundary_model, camera, model, corner_info, &border_seed) &&
          IsInsideImage(border_seed.image_point, gray.size())) {
        border_seed_image = border_seed.image_point;
        border_seed_ray = border_seed.ray;
        debug.border_seed_image = border_seed.image_point;
        debug.border_seed_ray =
            cv::Vec3d(border_seed.ray.x(), border_seed.ray.y(), border_seed.ray.z());
        debug.border_seed_valid = true;
        debug.border_top_ray =
            cv::Vec3d(border_seed.top_ray.x(), border_seed.top_ray.y(), border_seed.top_ray.z());
        debug.border_bottom_ray = cv::Vec3d(border_seed.bottom_ray.x(), border_seed.bottom_ray.y(),
                                            border_seed.bottom_ray.z());
        debug.border_left_ray =
            cv::Vec3d(border_seed.left_ray.x(), border_seed.left_ray.y(), border_seed.left_ray.z());
        debug.border_right_ray = cv::Vec3d(border_seed.right_ray.x(), border_seed.right_ray.y(),
                                           border_seed.right_ray.z());
        debug.predicted_to_border_seed_displacement =
            std::hypot(border_seed_image.x - frame.predicted_image.x,
                       border_seed_image.y - frame.predicted_image.y);
        const double adaptive_search_radius =
            ComputeAdaptiveBorderSeedSearchRadius(frame, border_seed.ray);
        if (BuildSeedAnchoredSphereLatticeFrame(frame, border_seed.ray, border_seed.image_point,
                                                adaptive_search_radius, &search_frame)) {
          debug.adaptive_search_radius = adaptive_search_radius;
          has_search_anchor = true;
        } else {
          debug.border_seed_valid = false;
        }
      }
    }

    cv::Point2f sphere_seed_image = search_frame.predicted_image;
    Eigen::Vector3d sphere_seed_ray = search_frame.predicted_ray;
    SphereSeedCandidate anchor_candidate;
    SphereSeedCandidate seed_candidate;
    RayRefinementResult ray_refine_result;
    bool has_seed = false;
    if (has_search_anchor) {
      if (use_ray_domain_refine) {
        ray_refine_result =
            RefineSphereSeedRayLocally(gray, camera, corner_info, search_frame, options,
                                       search_frame.predicted_ray);
        if (ray_refine_result.valid) {
          sphere_seed_image = ray_refine_result.refined_image;
          sphere_seed_ray = ray_refine_result.refined_ray;
          has_seed = true;
        }
      } else {
        anchor_candidate =
            EvaluateSphereSeedCandidate(gray, camera, corner_info, search_frame, options, 0.0, 0.0);
        if (anchor_candidate.valid) {
          sphere_seed_image = anchor_candidate.image_point;
          sphere_seed_ray = anchor_candidate.ray;
          has_seed = true;
        }
        seed_candidate = enable_seed_search
                             ? SearchSphereLatticeSeed(gray, camera, corner_info, search_frame,
                                                       options)
                             : anchor_candidate;
        if (seed_candidate.valid) {
          sphere_seed_image = seed_candidate.image_point;
          sphere_seed_ray = seed_candidate.ray;
          has_seed = true;
        }
      }
    }

    debug.sphere_seed_image = sphere_seed_image;
    debug.sphere_seed_ray = cv::Vec3d(sphere_seed_ray.x(), sphere_seed_ray.y(), sphere_seed_ray.z());
    if (use_ray_domain_refine) {
      debug.sphere_template_quality = ray_refine_result.template_quality;
      debug.sphere_gradient_quality = ray_refine_result.gradient_quality;
      debug.sphere_prior_quality = 0.0;
      debug.sphere_peak_quality = 0.0;
      debug.sphere_raw_quality = ray_refine_result.edge_quality;
      debug.sphere_seed_quality = ray_refine_result.final_quality;
    } else {
      const SphereSeedCandidate& quality_candidate =
          seed_candidate.valid ? seed_candidate : anchor_candidate;
      debug.sphere_template_quality = quality_candidate.template_quality;
      debug.sphere_gradient_quality = quality_candidate.gradient_quality;
      debug.sphere_prior_quality = quality_candidate.prior_quality;
      debug.sphere_peak_quality = quality_candidate.peak_quality;
      debug.sphere_raw_quality = quality_candidate.raw_quality;
      debug.sphere_seed_quality = quality_candidate.final_quality;
    }
    debug.border_seed_to_sphere_seed_displacement =
        debug.border_seed_valid
            ? std::hypot(sphere_seed_image.x - debug.border_seed_image.x,
                         sphere_seed_image.y - debug.border_seed_image.y)
            : 0.0;
    debug.predicted_to_seed_displacement =
        std::hypot(sphere_seed_image.x - frame.predicted_image.x,
                   sphere_seed_image.y - frame.predicted_image.y);
    const int image_evidence_search_radius =
        ComputeAdaptiveImageEvidenceSearchRadius(frame.local_module_scale, options);

    cv::Point2f refined_image = sphere_seed_image;
    Eigen::Vector3d refined_ray = sphere_seed_ray;
    const int subpix_window_radius =
        ComputeAdaptiveInternalSubpixRadius(frame.local_module_scale, options);
    const double subpix_displacement_limit =
        ComputeAdaptiveInternalSubpixDisplacementLimit(frame.local_module_scale, options);
    double q_refine = 0.0;
    if (use_ray_domain_refine) {
      if (has_frame && has_seed &&
          IsInsideImageWithBorder(sphere_seed_image, gray.size(), options.min_border_distance) &&
          options.do_subpix_refinement) {
        std::vector<cv::Point2f> corners{refined_image};
        cv::cornerSubPix(gray, corners,
                         cv::Size(subpix_window_radius, subpix_window_radius),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
        refined_image = corners.front();
      }
      if (IsInsideImage(refined_image, gray.size())) {
        Eigen::Vector3d refined_ray_candidate = Eigen::Vector3d::Zero();
        if (UnprojectImagePointToRay(camera, refined_image, &refined_ray_candidate)) {
          refined_ray = refined_ray_candidate;
        }
      }
      q_refine =
          (has_frame && has_seed &&
           IsInsideImageWithBorder(sphere_seed_image, gray.size(), options.min_border_distance))
              ? (options.do_subpix_refinement
                     ? ClampUnit(1.0 - (std::hypot(refined_image.x - sphere_seed_image.x,
                                                   refined_image.y - sphere_seed_image.y) *
                                        std::hypot(refined_image.x - sphere_seed_image.x,
                                                   refined_image.y - sphere_seed_image.y)) /
                                           std::max(1e-9, subpix_displacement_limit *
                                                             subpix_displacement_limit))
                     : 1.0)
              : 0.0;
      debug.ray_refine_edge_quality = ray_refine_result.edge_quality;
      debug.ray_refine_photometric_quality = ray_refine_result.photometric_quality;
      debug.ray_refine_final_quality = ray_refine_result.final_quality;
      debug.ray_refine_trust_radius = ray_refine_result.trust_radius;
      debug.ray_refine_iterations = ray_refine_result.iterations;
      debug.ray_refine_converged = ray_refine_result.converged;
    } else {
      if (has_frame && has_seed &&
          IsInsideImageWithBorder(sphere_seed_image, gray.size(), options.min_border_distance) &&
          options.do_subpix_refinement) {
        std::vector<cv::Point2f> corners{refined_image};
        cv::cornerSubPix(gray, corners,
                         cv::Size(subpix_window_radius, subpix_window_radius),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
        refined_image = corners.front();
      }
      if (IsInsideImage(refined_image, gray.size())) {
        Eigen::Vector3d refined_ray_candidate = Eigen::Vector3d::Zero();
        if (UnprojectImagePointToRay(camera, refined_image, &refined_ray_candidate)) {
          refined_ray = refined_ray_candidate;
        }
      }
      q_refine =
          (has_frame && has_seed &&
           IsInsideImageWithBorder(sphere_seed_image, gray.size(), options.min_border_distance))
              ? (options.do_subpix_refinement
                     ? ClampUnit(1.0 - (std::hypot(refined_image.x - sphere_seed_image.x,
                                                   refined_image.y - sphere_seed_image.y) *
                                        std::hypot(refined_image.x - sphere_seed_image.x,
                                                   refined_image.y - sphere_seed_image.y)) /
                                           std::max(1e-9, subpix_displacement_limit *
                                                             subpix_displacement_limit))
                     : 1.0)
              : 0.0;
    }

    debug.refined_image = refined_image;
    debug.refined_ray = cv::Vec3d(refined_ray.x(), refined_ray.y(), refined_ray.z());
    debug.subpix_window_radius = subpix_window_radius;
    debug.subpix_displacement_limit = subpix_displacement_limit;
    debug.image_evidence_search_radius = image_evidence_search_radius;
    debug.seed_to_refined_displacement =
        std::hypot(refined_image.x - sphere_seed_image.x, refined_image.y - sphere_seed_image.y);
    debug.seed_to_refined_angular = AngleBetweenRays(sphere_seed_ray, refined_ray);
    debug.predicted_to_refined_displacement =
        std::hypot(refined_image.x - frame.predicted_image.x,
                   refined_image.y - frame.predicted_image.y);
    debug.q_refine = q_refine;

    ImageEvidenceScore image_score;
    if (has_frame && has_seed) {
      image_score = EvaluateImageEvidenceAroundPoint(
          gray, corner_info, refined_image, frame.module_u_axis, frame.module_v_axis,
          options.min_template_contrast,
          subpix_displacement_limit * subpix_displacement_limit,
          image_evidence_search_radius);
    }

    debug.template_quality = debug.sphere_template_quality;
    debug.gradient_quality = debug.sphere_gradient_quality;
    debug.image_template_quality = image_score.best_score.template_quality;
    debug.image_gradient_quality = image_score.best_score.gradient_quality;
    debug.image_centering_quality = image_score.centering_quality;
    debug.image_final_quality = image_score.final_quality;

    const double final_quality =
        std::min({debug.sphere_seed_quality, q_refine, image_score.final_quality});
    debug.final_quality = final_quality;

    const bool valid =
        has_frame && has_seed &&
        IsInsideImageWithBorder(refined_image, gray.size(), options.min_border_distance);
    const bool image_evidence_valid =
        has_frame && has_seed &&
        IsInsideImageWithBorder(refined_image, gray.size(), options.min_border_distance) &&
        image_score.final_quality >= options.min_quality;
    debug.valid = valid;
    debug.image_evidence_valid = image_evidence_valid;

    CornerMeasurement& measurement = result->corners[static_cast<std::size_t>(point_id)];
    measurement.image_xy = Eigen::Vector2d(refined_image.x, refined_image.y);
    measurement.quality = final_quality;
    measurement.valid = valid;

    if (valid) {
      ++result->valid_corner_count;
      ++result->valid_internal_corner_count;
    }

    result->internal_corner_debug.push_back(debug);
  }
}

void PopulateInternalCornersFromVirtualPatch(const cv::Mat& gray,
                                             const DoubleSphereCameraModel& camera,
                                             const VirtualPatchContext& context,
                                             const std::array<int, 4>& outer_point_ids,
                                             const ApriltagCanonicalModel& model,
                                             const ApriltagInternalDetectionOptions& options,
                                             ApriltagInternalDetectionResult* result) {
  if (result == nullptr) {
    throw std::runtime_error("Result pointer must not be null.");
  }

  for (const int point_id : model.VisiblePointIds()) {
    if (std::find(outer_point_ids.begin(), outer_point_ids.end(), point_id) != outer_point_ids.end()) {
      continue;
    }

    const CanonicalCorner& corner_info = model.corner(point_id);
    const cv::Point3f target_point(static_cast<float>(corner_info.target_xyz.x()),
                                   static_cast<float>(corner_info.target_xyz.y()),
                                   static_cast<float>(corner_info.target_xyz.z()));

    bool predicted_visible = false;
    const cv::Point2f predicted_patch =
        ProjectTargetPointToVirtualPatch(context, target_point, &predicted_visible);
    bool predicted_image_visible = false;
    const cv::Point2f predicted_image =
        ProjectTargetPointToImage(camera, context.target_to_camera_rotation,
                                  context.target_to_camera_translation, target_point,
                                  &predicted_image_visible);
    const double patch_module_scale_px = static_cast<double>(options.canonical_pixels_per_module);
    const int subpix_window_radius =
        ComputeAdaptiveInternalSubpixRadius(patch_module_scale_px, options);
    const double subpix_displacement_limit =
        ComputeAdaptiveInternalSubpixDisplacementLimit(patch_module_scale_px, options);

    cv::Point2f refined_patch = predicted_patch;
    if (predicted_visible &&
        IsInsideImageWithBorder(predicted_patch, context.patch_size, options.min_border_distance) &&
        options.do_subpix_refinement) {
      std::vector<cv::Point2f> corners{refined_patch};
      cv::cornerSubPix(context.patch, corners,
                       cv::Size(subpix_window_radius, subpix_window_radius),
                       cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
      refined_patch = corners.front();
    }

    cv::Point2f refined_image = predicted_image;
    bool refined_visible = predicted_image_visible;
    Eigen::Vector3d refined_point_camera;
    if (predicted_visible &&
        IsInsideImageWithBorder(refined_patch, context.patch_size, options.min_border_distance) &&
        IntersectVirtualPatchPixelWithTargetPlane(context, refined_patch, &refined_point_camera)) {
      Eigen::Vector2d reprojected_image;
      refined_visible = camera.vsEuclideanToKeypoint(refined_point_camera, &reprojected_image);
      refined_image =
          cv::Point2f(static_cast<float>(reprojected_image.x()), static_cast<float>(reprojected_image.y()));
    }

    const double displacement =
        std::hypot(refined_patch.x - predicted_patch.x, refined_patch.y - predicted_patch.y);
    const double q_refine = (predicted_visible && predicted_image_visible && refined_visible)
                                ? (options.do_subpix_refinement
                                       ? ClampUnit(1.0 - (displacement * displacement) /
                                                           std::max(1e-9, subpix_displacement_limit *
                                                                             subpix_displacement_limit))
                                       : 1.0)
                                : 0.0;
    const TemplateScore template_score =
        ComputeTemplateScoreAtPoint(result->canonical_patch, corner_info, refined_patch,
                                    options.canonical_pixels_per_module, options.min_template_contrast);
    cv::Point2f module_u_axis;
    cv::Point2f module_v_axis;
    const bool has_image_axes =
        ComputeVirtualImageAxes(camera, context.target_to_camera_rotation,
                                context.target_to_camera_translation, model, corner_info,
                                &module_u_axis, &module_v_axis);
    const double image_module_scale_px =
        has_image_axes ? ComputeModuleScalePx(module_u_axis, module_v_axis, patch_module_scale_px)
                       : patch_module_scale_px;
    const double image_subpix_displacement_limit =
        ComputeAdaptiveInternalSubpixDisplacementLimit(image_module_scale_px, options);
    const int image_evidence_search_radius =
        ComputeAdaptiveImageEvidenceSearchRadius(image_module_scale_px, options);
    const ImageEvidenceScore image_score =
        has_image_axes
            ? EvaluateImageEvidenceAroundPoint(gray, corner_info, refined_image, module_u_axis, module_v_axis,
                                               options.min_template_contrast,
                                               image_subpix_displacement_limit *
                                                   image_subpix_displacement_limit,
                                               image_evidence_search_radius)
            : ImageEvidenceScore{};
    const double final_quality =
        std::min({template_score.template_quality, template_score.gradient_quality, q_refine});
    const double image_final_quality = image_score.final_quality;
    const bool valid = refined_visible &&
                       IsInsideImageWithBorder(refined_image, result->image_size, options.min_border_distance);
    const bool image_evidence_valid =
        refined_visible &&
        IsInsideImageWithBorder(refined_image, result->image_size, options.min_border_distance) &&
        image_final_quality >= options.min_quality;

    CornerMeasurement& measurement = result->corners[static_cast<std::size_t>(point_id)];
    measurement.image_xy = Eigen::Vector2d(refined_image.x, refined_image.y);
    measurement.quality = final_quality;
    measurement.valid = valid;

    if (valid) {
      ++result->valid_corner_count;
      ++result->valid_internal_corner_count;
    }

    InternalCornerDebugInfo debug;
    debug.point_id = point_id;
    debug.corner_type = corner_info.corner_type;
    debug.predicted_image = predicted_image;
    debug.refined_image = refined_image;
    debug.predicted_patch = predicted_patch;
    debug.refined_patch = refined_patch;
    if (has_image_axes) {
      debug.module_u_axis = module_u_axis;
      debug.module_v_axis = module_v_axis;
    }
    debug.local_module_scale = patch_module_scale_px;
    debug.subpix_window_radius = subpix_window_radius;
    debug.subpix_displacement_limit = subpix_displacement_limit;
    debug.image_evidence_search_radius = image_evidence_search_radius;
    debug.q_refine = q_refine;
    debug.template_quality = template_score.template_quality;
    debug.gradient_quality = template_score.gradient_quality;
    debug.final_quality = final_quality;
    debug.image_template_quality = image_score.best_score.template_quality;
    debug.image_gradient_quality = image_score.best_score.gradient_quality;
    debug.image_centering_quality = image_score.centering_quality;
    debug.image_final_quality = image_final_quality;
    debug.predicted_to_refined_displacement = displacement;
    debug.valid = valid;
    debug.image_evidence_valid = image_evidence_valid;
    result->internal_corner_debug.push_back(debug);
  }
}

void PopulateInternalCornersFromVirtualPatchImageSubpix(
    const cv::Mat& gray,
    const DoubleSphereCameraModel& camera,
    const VirtualPatchContext& context,
    const std::array<int, 4>& outer_point_ids,
    const ApriltagCanonicalModel& model,
    const ApriltagInternalDetectionOptions& options,
    ApriltagInternalDetectionResult* result) {
  if (result == nullptr) {
    throw std::runtime_error("Result pointer must not be null.");
  }

  for (const int point_id : model.VisiblePointIds()) {
    if (std::find(outer_point_ids.begin(), outer_point_ids.end(), point_id) !=
        outer_point_ids.end()) {
      continue;
    }

    const CanonicalCorner& corner_info = model.corner(point_id);
    const cv::Point3f target_point(static_cast<float>(corner_info.target_xyz.x()),
                                   static_cast<float>(corner_info.target_xyz.y()),
                                   static_cast<float>(corner_info.target_xyz.z()));

    bool predicted_visible = false;
    const cv::Point2f predicted_patch =
        ProjectTargetPointToVirtualPatch(context, target_point, &predicted_visible);
    bool predicted_image_visible = false;
    const cv::Point2f predicted_image =
        ProjectTargetPointToImage(camera, context.target_to_camera_rotation,
                                  context.target_to_camera_translation, target_point,
                                  &predicted_image_visible);

    cv::Point2f module_u_axis;
    cv::Point2f module_v_axis;
    const bool has_image_axes =
        ComputeVirtualImageAxes(camera, context.target_to_camera_rotation,
                                context.target_to_camera_translation, model, corner_info,
                                &module_u_axis, &module_v_axis);
    const double fallback_module_scale_px =
        static_cast<double>(options.canonical_pixels_per_module);
    const double image_module_scale_px =
        has_image_axes
            ? ComputeModuleScalePx(module_u_axis, module_v_axis, fallback_module_scale_px)
            : fallback_module_scale_px;
    const int subpix_window_radius =
        ComputeAdaptiveInternalSubpixRadius(image_module_scale_px, options);
    const double subpix_displacement_limit =
        ComputeAdaptiveInternalSubpixDisplacementLimit(image_module_scale_px, options);

    cv::Point2f refined_image = predicted_image;
    if (predicted_visible && predicted_image_visible && has_image_axes &&
        IsInsideImageWithBorder(predicted_image, gray.size(), options.min_border_distance) &&
        options.do_subpix_refinement) {
      std::vector<cv::Point2f> corners{refined_image};
      cv::cornerSubPix(gray, corners,
                       cv::Size(subpix_window_radius, subpix_window_radius),
                       cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30,
                                        0.1));
      refined_image = corners.front();
    }

    cv::Point2f refined_patch = predicted_patch;
    if (predicted_visible &&
        IsInsideImageWithBorder(refined_image, gray.size(), options.min_border_distance)) {
      cv::Point2f mapped_patch;
      if (ProjectImagePointToVirtualPatch(camera, context, refined_image, &mapped_patch)) {
        refined_patch = mapped_patch;
      }
    }

    const double displacement =
        std::hypot(refined_image.x - predicted_image.x, refined_image.y - predicted_image.y);
    const double q_refine =
        (predicted_visible && predicted_image_visible && has_image_axes &&
         IsInsideImageWithBorder(predicted_image, gray.size(), options.min_border_distance))
            ? (options.do_subpix_refinement
                   ? ClampUnit(1.0 - (displacement * displacement) /
                                           std::max(1e-9, subpix_displacement_limit *
                                                             subpix_displacement_limit))
                   : 1.0)
            : 0.0;

    const TemplateScore patch_score =
        ComputeTemplateScoreAtPoint(result->canonical_patch, corner_info, refined_patch,
                                    options.canonical_pixels_per_module,
                                    options.min_template_contrast);
    const int image_evidence_search_radius =
        ComputeAdaptiveImageEvidenceSearchRadius(image_module_scale_px, options);
    const ImageEvidenceScore image_score =
        has_image_axes
            ? EvaluateImageEvidenceAroundPoint(gray, corner_info, refined_image, module_u_axis,
                                               module_v_axis, options.min_template_contrast,
                                               subpix_displacement_limit *
                                                   subpix_displacement_limit,
                                               image_evidence_search_radius)
            : ImageEvidenceScore{};
    const double final_quality = std::min(q_refine, image_score.final_quality);
    const double image_final_quality = image_score.final_quality;
    const bool valid =
        predicted_visible && predicted_image_visible && has_image_axes &&
        IsInsideImageWithBorder(refined_image, result->image_size, options.min_border_distance);
    const bool image_evidence_valid =
        predicted_visible && predicted_image_visible && has_image_axes &&
        IsInsideImageWithBorder(refined_image, result->image_size, options.min_border_distance) &&
        image_final_quality >= options.min_quality;

    CornerMeasurement& measurement = result->corners[static_cast<std::size_t>(point_id)];
    measurement.image_xy = Eigen::Vector2d(refined_image.x, refined_image.y);
    measurement.quality = final_quality;
    measurement.valid = valid;

    if (valid) {
      ++result->valid_corner_count;
      ++result->valid_internal_corner_count;
    }

    InternalCornerDebugInfo debug;
    debug.point_id = point_id;
    debug.corner_type = corner_info.corner_type;
    debug.predicted_image = predicted_image;
    debug.refined_image = refined_image;
    debug.predicted_patch = predicted_patch;
    debug.refined_patch = refined_patch;
    if (has_image_axes) {
      debug.module_u_axis = module_u_axis;
      debug.module_v_axis = module_v_axis;
    }
    debug.local_module_scale = image_module_scale_px;
    debug.subpix_window_radius = subpix_window_radius;
    debug.subpix_displacement_limit = subpix_displacement_limit;
    debug.image_evidence_search_radius = image_evidence_search_radius;
    debug.q_refine = q_refine;
    debug.template_quality = patch_score.template_quality;
    debug.gradient_quality = patch_score.gradient_quality;
    debug.final_quality = final_quality;
    debug.image_template_quality = image_score.best_score.template_quality;
    debug.image_gradient_quality = image_score.best_score.gradient_quality;
    debug.image_centering_quality = image_score.centering_quality;
    debug.image_final_quality = image_final_quality;
    debug.predicted_to_refined_displacement = displacement;
    debug.valid = valid;
    debug.image_evidence_valid = image_evidence_valid;
    result->internal_corner_debug.push_back(debug);
  }
}

void PopulateInternalCornersFromVirtualPatchBoundarySeed(
    const cv::Mat& gray,
    const DoubleSphereCameraModel& camera,
    const VirtualPatchContext& context,
    const std::array<int, 4>& outer_point_ids,
    const ApriltagCanonicalModel& model,
    const ApriltagInternalDetectionOptions& options,
    ApriltagInternalDetectionResult* result) {
  if (result == nullptr) {
    throw std::runtime_error("Result pointer must not be null.");
  }

  for (const int point_id : model.VisiblePointIds()) {
    if (std::find(outer_point_ids.begin(), outer_point_ids.end(), point_id) !=
        outer_point_ids.end()) {
      continue;
    }

    const CanonicalCorner& corner_info = model.corner(point_id);
    const cv::Point3f target_point(static_cast<float>(corner_info.target_xyz.x()),
                                   static_cast<float>(corner_info.target_xyz.y()),
                                   static_cast<float>(corner_info.target_xyz.z()));

    bool predicted_visible = false;
    const cv::Point2f predicted_patch =
        ProjectTargetPointToVirtualPatch(context, target_point, &predicted_visible);
    bool predicted_image_visible = false;
    const cv::Point2f predicted_image =
        ProjectTargetPointToImage(camera, context.target_to_camera_rotation,
                                  context.target_to_camera_translation, target_point,
                                  &predicted_image_visible);

    cv::Point2f image_module_u_axis;
    cv::Point2f image_module_v_axis;
    const bool has_image_axes =
        ComputeVirtualImageAxes(camera, context.target_to_camera_rotation,
                                context.target_to_camera_translation, model, corner_info,
                                &image_module_u_axis, &image_module_v_axis);
    const double fallback_module_scale_px =
        static_cast<double>(options.canonical_pixels_per_module);
    const double image_module_scale_px =
        has_image_axes
            ? ComputeModuleScalePx(image_module_u_axis, image_module_v_axis, fallback_module_scale_px)
            : fallback_module_scale_px;
    const int subpix_window_radius =
        ComputeAdaptiveInternalSubpixRadius(image_module_scale_px, options);
    const double subpix_displacement_limit =
        ComputeAdaptiveInternalSubpixDisplacementLimit(image_module_scale_px, options);
    const int image_evidence_search_radius =
        ComputeAdaptiveImageEvidenceSearchRadius(image_module_scale_px, options);

    PatchSeedFrame patch_frame;
    const bool has_patch_frame =
        BuildPatchSeedFrame(camera, context, model, corner_info, &patch_frame);
    PatchSeedCandidate seed_candidate;
    if (has_patch_frame) {
      seed_candidate = SearchVirtualPatchBoundarySeed(
          camera, context.patch, context, corner_info, patch_frame, options);
    }

    cv::Point2f seed_patch = predicted_patch;
    cv::Point2f seed_image = predicted_image;
    double seed_template_quality = 1.0;
    double seed_gradient_quality = 1.0;
    double seed_prior_quality = 1.0;
    double seed_peak_quality = 1.0;
    double seed_raw_quality = 1.0;
    double seed_quality = 1.0;
    if (seed_candidate.valid) {
      seed_patch = seed_candidate.patch_point;
      seed_image = seed_candidate.image_point;
      seed_template_quality = seed_candidate.template_quality;
      seed_gradient_quality = seed_candidate.gradient_quality;
      seed_prior_quality = seed_candidate.prior_quality;
      seed_peak_quality = seed_candidate.peak_quality;
      seed_raw_quality = seed_candidate.raw_quality;
      seed_quality = seed_candidate.final_quality;
    }

    cv::Point2f refined_image = seed_image;
    if (predicted_visible && predicted_image_visible && has_image_axes &&
        IsInsideImageWithBorder(seed_image, gray.size(), options.min_border_distance) &&
        options.do_subpix_refinement) {
      std::vector<cv::Point2f> corners{refined_image};
      cv::cornerSubPix(gray, corners,
                       cv::Size(subpix_window_radius, subpix_window_radius),
                       cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30,
                                        0.1));
      refined_image = corners.front();
    }

    cv::Point2f refined_patch = seed_patch;
    if (predicted_visible &&
        IsInsideImageWithBorder(refined_image, gray.size(), options.min_border_distance)) {
      cv::Point2f mapped_patch;
      if (ProjectImagePointToVirtualPatch(camera, context, refined_image, &mapped_patch)) {
        refined_patch = mapped_patch;
      }
    }

    const double displacement =
        std::hypot(refined_image.x - seed_image.x, refined_image.y - seed_image.y);
    const double q_refine =
        (predicted_visible && predicted_image_visible && has_image_axes &&
         IsInsideImageWithBorder(seed_image, gray.size(), options.min_border_distance))
            ? (options.do_subpix_refinement
                   ? ClampUnit(1.0 - (displacement * displacement) /
                                           std::max(1e-9, subpix_displacement_limit *
                                                             subpix_displacement_limit))
                   : 1.0)
            : 0.0;

    const ImageEvidenceScore image_score =
        has_image_axes
            ? EvaluateImageEvidenceAroundPoint(gray, corner_info, refined_image,
                                               image_module_u_axis, image_module_v_axis,
                                               options.min_template_contrast,
                                               subpix_displacement_limit *
                                                   subpix_displacement_limit,
                                               image_evidence_search_radius)
            : ImageEvidenceScore{};
    const double final_quality =
        std::min({seed_quality, q_refine, image_score.final_quality});
    const double image_final_quality = image_score.final_quality;
    const bool valid =
        predicted_visible && predicted_image_visible && has_image_axes &&
        IsInsideImageWithBorder(refined_image, result->image_size, options.min_border_distance);
    const bool image_evidence_valid =
        predicted_visible && predicted_image_visible && has_image_axes &&
        IsInsideImageWithBorder(refined_image, result->image_size, options.min_border_distance) &&
        image_final_quality >= options.min_quality;

    CornerMeasurement& measurement = result->corners[static_cast<std::size_t>(point_id)];
    measurement.image_xy = Eigen::Vector2d(refined_image.x, refined_image.y);
    measurement.quality = final_quality;
    measurement.valid = valid;

    if (valid) {
      ++result->valid_corner_count;
      ++result->valid_internal_corner_count;
    }

    InternalCornerDebugInfo debug;
    debug.point_id = point_id;
    debug.corner_type = corner_info.corner_type;
    debug.predicted_image = predicted_image;
    debug.sphere_seed_image = seed_image;
    debug.refined_image = refined_image;
    debug.predicted_patch = predicted_patch;
    debug.sphere_seed_patch = seed_patch;
    debug.refined_patch = refined_patch;
    if (has_image_axes) {
      debug.module_u_axis = image_module_u_axis;
      debug.module_v_axis = image_module_v_axis;
    }
    debug.local_module_scale = image_module_scale_px;
    debug.sphere_search_radius = has_patch_frame ? patch_frame.search_radius : 0.0;
    debug.sphere_template_quality = seed_template_quality;
    debug.sphere_gradient_quality = seed_gradient_quality;
    debug.sphere_prior_quality = seed_prior_quality;
    debug.sphere_peak_quality = seed_peak_quality;
    debug.sphere_raw_quality = seed_raw_quality;
    debug.sphere_seed_quality = seed_quality;
    debug.subpix_window_radius = subpix_window_radius;
    debug.subpix_displacement_limit = subpix_displacement_limit;
    debug.image_evidence_search_radius = image_evidence_search_radius;
    debug.q_refine = q_refine;
    debug.template_quality = seed_template_quality;
    debug.gradient_quality = seed_gradient_quality;
    debug.final_quality = final_quality;
    debug.image_template_quality = image_score.best_score.template_quality;
    debug.image_gradient_quality = image_score.best_score.gradient_quality;
    debug.image_centering_quality = image_score.centering_quality;
    debug.image_final_quality = image_final_quality;
    debug.predicted_to_seed_displacement =
        std::hypot(seed_image.x - predicted_image.x, seed_image.y - predicted_image.y);
    debug.seed_to_refined_displacement =
        std::hypot(refined_image.x - seed_image.x, refined_image.y - seed_image.y);
    debug.predicted_to_refined_displacement =
        std::hypot(refined_image.x - predicted_image.x, refined_image.y - predicted_image.y);
    debug.valid = valid;
    debug.image_evidence_valid = image_evidence_valid;
    result->internal_corner_debug.push_back(debug);
  }
}

}  // namespace

ApriltagInternalDetector::ApriltagInternalDetector(
    ApriltagInternalConfig config, ApriltagInternalDetectionOptions options)
    : config_(std::move(config)), options_(options) {
  if (options_.canonical_pixels_per_module <= 0) {
    throw std::runtime_error("canonical_pixels_per_module must be positive.");
  }
  if (options_.refinement_window_radius < 0) {
    throw std::runtime_error("refinement_window_radius must be non-negative.");
  }
  if (options_.internal_subpix_window_scale < 0.0) {
    throw std::runtime_error("internal_subpix_window_scale must be non-negative.");
  }
  if (options_.internal_subpix_window_min <= 0) {
    throw std::runtime_error("internal_subpix_window_min must be positive.");
  }
  if (options_.internal_subpix_window_max < options_.internal_subpix_window_min) {
    throw std::runtime_error("internal_subpix_window_max must be >= internal_subpix_window_min.");
  }
  if (options_.min_quality < 0.0 || options_.min_quality > 1.0) {
    throw std::runtime_error("min_quality must be in [0, 1].");
  }
  if (options_.virtual_patch_margin <= 1.0) {
    throw std::runtime_error("virtual_patch_margin must be greater than 1.0.");
  }
  if (options_.internal_subpix_displacement_scale < 0.0) {
    throw std::runtime_error("internal_subpix_displacement_scale must be non-negative.");
  }
  if (options_.max_internal_subpix_displacement <= 0.0) {
    throw std::runtime_error("max_internal_subpix_displacement must be positive.");
  }
  if (config_.sphere_lattice_init_fu_scale <= 0.0 || config_.sphere_lattice_init_fv_scale <= 0.0) {
    throw std::runtime_error("sphere_lattice_init_fu_scale and sphere_lattice_init_fv_scale must be positive.");
  }
  if (config_.sphere_lattice_init_alpha <= 0.0 || config_.sphere_lattice_init_alpha >= 1.0) {
    throw std::runtime_error("sphere_lattice_init_alpha must be in (0, 1).");
  }

  requested_board_ids_ = NormalizeBoardIds(config_.tag_ids, config_.tag_id);
  config_.tag_ids = requested_board_ids_;
  config_.tag_id = requested_board_ids_.front();
  default_board_index_ = 0;

  options_.min_border_distance = config_.outer_detector_config.min_border_distance;
  options_.outer_detector_config = config_.outer_detector_config;
  options_.outer_detector_config.tag_ids = requested_board_ids_;
  options_.outer_detector_config.tag_id = config_.tag_id;
  options_.outer_detector_config.min_border_distance = options_.min_border_distance;
  options_.outer_detector_config.do_outer_subpix_refinement =
      options_.do_subpix_refinement && config_.outer_detector_config.do_outer_subpix_refinement;
  IntermediateCameraConfig outer_refine_camera = config_.intermediate_camera;
  if (config_.outer_spherical_use_initial_camera &&
      options_.outer_detector_config.enable_outer_spherical_refinement) {
    if (config_.intermediate_camera.resolution.size() != 2) {
      throw std::runtime_error(
          "outer_spherical_use_initial_camera requires a configured image resolution.");
    }
    outer_refine_camera = MakeCoarseInitialCameraConfig(
        cv::Size(config_.intermediate_camera.resolution[0],
                 config_.intermediate_camera.resolution[1]),
        config_);
  }
  options_.outer_detector_config.refine_camera.camera_model = outer_refine_camera.camera_model;
  options_.outer_detector_config.refine_camera.distortion_model =
      outer_refine_camera.distortion_model;
  options_.outer_detector_config.refine_camera.intrinsics = outer_refine_camera.intrinsics;
  options_.outer_detector_config.refine_camera.distortion_coeffs =
      outer_refine_camera.distortion_coeffs;
  options_.outer_detector_config.refine_camera.resolution = outer_refine_camera.resolution;

  board_runtimes_.reserve(requested_board_ids_.size());
  for (int board_id : requested_board_ids_) {
    board_runtimes_.emplace_back(MakeBoardSpecificConfig(config_, board_id));
  }

  outer_detector_ = std::make_unique<MultiScaleOuterTagDetector>(options_.outer_detector_config);
}

ApriltagInternalDetector::~ApriltagInternalDetector() = default;

ApriltagInternalConfig ApriltagInternalDetector::LoadConfig(const std::string& yaml_path) {
  return ParseApriltagInternalConfig(yaml_path);
}

const ApriltagInternalDetector::BoardRuntime&
ApriltagInternalDetector::RuntimeForBoardIdOrDefault(int board_id) const {
  for (const BoardRuntime& runtime : board_runtimes_) {
    if (runtime.config.tag_id == board_id) {
      return runtime;
    }
  }
  return board_runtimes_[default_board_index_];
}

ApriltagInternalDetectionResult ApriltagInternalDetector::DetectSingleBoardFromOuter(
    const cv::Mat& gray,
    const BoardRuntime& board_runtime,
    const OuterTagDetectionResult& outer_detection,
    const IntermediateCameraConfig* camera_override,
    const Eigen::Matrix4d* T_camera_board_prior) const {
  const ApriltagInternalConfig& board_config = board_runtime.config;
  const ApriltagCanonicalModel& model = board_runtime.model;
  ApriltagInternalDetectionResult result;
  result.image_size = gray.size();
  result.board_id = board_config.tag_id;
  result.projection_mode = board_config.internal_projection_mode;
  result.expected_visible_point_count = model.ObservablePointCount();
  result.corners = model.MakeDefaultMeasurements();
  result.internal_corner_debug.reserve(model.VisiblePointIds().size());
  result.outer_detection = outer_detection;
  if (!result.outer_detection.success) {
    return result;
  }

  result.tag_detected = true;

  std::array<cv::Point2f, 4> outer_corners{};
  std::array<cv::Point2f, 4> raw_outer_corners{};
  for (int i = 0; i < 4; ++i) {
    outer_corners[i] = cv::Point2f(
        static_cast<float>(result.outer_detection.refined_corners_original_image[static_cast<std::size_t>(i)].x()),
        static_cast<float>(result.outer_detection.refined_corners_original_image[static_cast<std::size_t>(i)].y()));
    raw_outer_corners[i] = cv::Point2f(
        static_cast<float>(result.outer_detection.coarse_corners_original_image[static_cast<std::size_t>(i)].x()),
        static_cast<float>(result.outer_detection.coarse_corners_original_image[static_cast<std::size_t>(i)].y()));
  }

  result.outer_corners = outer_corners;
  result.tag_center = ComputeCenter(outer_corners);
  result.observed_perimeter = ComputePerimeter(outer_corners);

  const std::array<int, 4> outer_point_ids{
      model.PointId(0, 0),
      model.PointId(model.ModuleDimension(), 0),
      model.PointId(model.ModuleDimension(), model.ModuleDimension()),
      model.PointId(0, model.ModuleDimension()),
  };

  cv::Mat corner_response;
  cv::cornerMinEigenVal(gray, corner_response, 3, 3);
  double response_min = 0.0;
  double response_max = 0.0;
  cv::minMaxLoc(corner_response, &response_min, &response_max);
  const double gradient_norm = std::max(1e-6, response_max * 0.2);

  for (int i = 0; i < 4; ++i) {
    CornerMeasurement& measurement = result.corners[static_cast<std::size_t>(outer_point_ids[i])];
    measurement.image_xy = Eigen::Vector2d(outer_corners[i].x, outer_corners[i].y);

    const OuterCornerVerificationDebugInfo& outer_debug =
        result.outer_detection.corner_verification_debug[static_cast<std::size_t>(i)];
    const double q_refine = 1.0;
    const double q_gradient =
        ClampUnit(SampleFloatAt(corner_response, outer_corners[i]) / gradient_norm);
    measurement.quality = std::min(q_refine, q_gradient);
    measurement.valid = result.outer_detection.refined_valid[static_cast<std::size_t>(i)];
    result.outer_corner_valid[static_cast<std::size_t>(i)] = measurement.valid;
    if (measurement.valid) {
      ++result.valid_corner_count;
    }
  }

  if (board_config.internal_projection_mode == InternalProjectionMode::Homography) {
    std::vector<cv::Point2f> board_outer{
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(model.ModuleDimension()), 0.0f),
        cv::Point2f(static_cast<float>(model.ModuleDimension()),
                    static_cast<float>(model.ModuleDimension())),
        cv::Point2f(0.0f, static_cast<float>(model.ModuleDimension())),
    };
    std::vector<cv::Point2f> image_outer(outer_corners.begin(), outer_corners.end());
    const cv::Mat board_to_image = cv::getPerspectiveTransform(board_outer, image_outer);

    const int patch_extent = model.ModuleDimension() * options_.canonical_pixels_per_module;
    std::vector<cv::Point2f> patch_outer{
        cv::Point2f(0.0f, static_cast<float>(patch_extent)),
        cv::Point2f(static_cast<float>(patch_extent), static_cast<float>(patch_extent)),
        cv::Point2f(static_cast<float>(patch_extent), 0.0f),
        cv::Point2f(0.0f, 0.0f),
    };
    const cv::Mat image_to_patch = cv::getPerspectiveTransform(image_outer, patch_outer);
    cv::warpPerspective(gray, result.canonical_patch, image_to_patch,
                        cv::Size(patch_extent + 1, patch_extent + 1), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar(255));

    result.patch_outer_corners = {patch_outer[0], patch_outer[1], patch_outer[2], patch_outer[3]};
    PopulateInternalCornersFromHomography(gray, board_to_image, image_to_patch, outer_point_ids, model,
                                          options_, &result);
  } else {
    const bool sphere_lattice_has_initial_camera =
        UsesSphereSeedPipeline(board_config.internal_projection_mode) &&
        board_config.sphere_lattice_use_initial_camera;
    if (!board_config.intermediate_camera.IsConfigured() && !sphere_lattice_has_initial_camera) {
      throw std::runtime_error(
          std::string(ToString(board_config.internal_projection_mode)) +
          " mode requires an intermediate camera model in the config.");
    }

    const IntermediateCameraConfig internal_camera_config =
        camera_override != nullptr && camera_override->IsConfigured()
            ? *camera_override
            : ((UsesSphereSeedPipeline(board_config.internal_projection_mode) &&
                board_config.sphere_lattice_use_initial_camera)
                   ? MakeCoarseInitialCameraConfig(gray.size(), board_config)
                   : board_config.intermediate_camera);
    const DoubleSphereCameraModel camera =
        DoubleSphereCameraModel::FromConfig(internal_camera_config);
    if (gray.size() != camera.resolution()) {
      throw std::runtime_error("Input image size " + std::to_string(gray.cols) + "x" +
                               std::to_string(gray.rows) +
                               " does not match DS camera resolution " +
                               std::to_string(camera.resolution().width) + "x" +
                               std::to_string(camera.resolution().height) + ".");
    }

    cv::Matx33d target_to_camera_rotation = cv::Matx33d::eye();
    Eigen::Vector3d target_to_camera_translation = Eigen::Vector3d::Zero();
    if (T_camera_board_prior != nullptr) {
      target_to_camera_rotation = Matrix4dToMatx33d(*T_camera_board_prior);
      target_to_camera_translation = Matrix4dToTranslation(*T_camera_board_prior);
    } else {
      cv::Mat rvec;
      cv::Mat tvec;
      if (!EstimateTargetPose(camera, outer_corners, outer_point_ids, model, &rvec, &tvec)) {
        throw std::runtime_error("Failed to estimate target pose for " +
                                 std::string(ToString(board_config.internal_projection_mode)) + " mode.");
      }
      cv::Rodrigues(rvec, target_to_camera_rotation);
      target_to_camera_translation = MatToEigenVector3d(tvec);
    }

    if (UsesSphereSeedPipeline(board_config.internal_projection_mode)) {
      PopulateInternalCornersFromSphereLattice(gray, camera, target_to_camera_rotation,
                                               target_to_camera_translation, outer_point_ids,
                                               outer_detection, model, options_,
                                               board_config.sphere_lattice_enable_seed_search,
                                               UsesBorderConditionedSphereSeed(
                                                   board_config.internal_projection_mode),
                                               board_config.internal_projection_mode ==
                                                   InternalProjectionMode::SphereRayRefine,
                                               &result);
    } else {
      cv::Mat rvec;
      cv::Mat tvec;
      if (T_camera_board_prior != nullptr) {
        cv::Mat rotation_cv(3, 3, CV_64F);
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            rotation_cv.at<double>(row, col) = target_to_camera_rotation(row, col);
          }
        }
        cv::Rodrigues(rotation_cv, rvec);
        tvec = (cv::Mat_<double>(3, 1) << target_to_camera_translation.x(),
                target_to_camera_translation.y(),
                target_to_camera_translation.z());
      } else if (!EstimateTargetPose(camera, outer_corners, outer_point_ids, model, &rvec, &tvec)) {
        throw std::runtime_error("Failed to estimate target pose for " +
                                 std::string(ToString(board_config.internal_projection_mode)) + " mode.");
      }
      const VirtualPatchContext context =
          BuildVirtualPatchContext(gray, camera, rvec, tvec, model, outer_point_ids, options_);
      result.canonical_patch = context.patch.clone();
      result.patch_outer_corners = context.outer_patch_corners;

      if (board_config.internal_projection_mode ==
          InternalProjectionMode::VirtualPinholeImageSubpix) {
        PopulateInternalCornersFromVirtualPatchImageSubpix(
            gray, camera, context, outer_point_ids, model, options_, &result);
      } else if (board_config.internal_projection_mode ==
                 InternalProjectionMode::VirtualPinholePatchBoundarySeed) {
        PopulateInternalCornersFromVirtualPatchBoundarySeed(
            gray, camera, context, outer_point_ids, model, options_, &result);
      } else {
        PopulateInternalCornersFromVirtualPatch(gray, camera, context, outer_point_ids, model,
                                                options_, &result);
      }
    }
  }

  result.success =
      result.valid_internal_corner_count > 0 &&
      result.valid_corner_count >= board_config.min_visible_points;
  return result;
}

ApriltagInternalDetectionResult ApriltagInternalDetector::Detect(const cv::Mat& image) const {
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  const cv::Mat gray = ToGray(image);
  const BoardRuntime& runtime = board_runtimes_[default_board_index_];
  const OuterTagDetectionResult outer_detection = outer_detector_->Detect(gray);
  return DetectSingleBoardFromOuter(gray, runtime, outer_detection, nullptr, nullptr);
}

ApriltagInternalMultiDetectionResult ApriltagInternalDetector::DetectMultiple(
    const cv::Mat& image) const {
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  const cv::Mat gray = ToGray(image);
  ApriltagInternalMultiDetectionResult result;
  result.image_size = gray.size();
  result.requested_board_ids = requested_board_ids_;

  const OuterTagMultiDetectionResult outer_multi_detection =
      outer_detector_->DetectMultiple(gray);
  result.requested_board_ids = outer_multi_detection.requested_board_ids;
  result.detections.reserve(requested_board_ids_.size());
  for (std::size_t index = 0; index < requested_board_ids_.size(); ++index) {
    const int board_id = requested_board_ids_[index];
    const BoardRuntime& runtime = RuntimeForBoardIdOrDefault(board_id);
    if (index < outer_multi_detection.detections.size()) {
      result.detections.push_back(
          DetectSingleBoardFromOuter(
              gray, runtime, outer_multi_detection.detections[index], nullptr, nullptr));
    } else {
      OuterTagDetectionResult empty_outer;
      empty_outer.board_id = board_id;
      empty_outer.failure_reason = OuterTagFailureReason::NoDetectionsAtAll;
      empty_outer.failure_reason_text = ToString(empty_outer.failure_reason);
      result.detections.push_back(
          DetectSingleBoardFromOuter(gray, runtime, empty_outer, nullptr, nullptr));
    }
  }

  return result;
}

ApriltagInternalDetectionResult ApriltagInternalDetector::DetectFromOuterDetection(
    const cv::Mat& image,
    int board_id,
    const OuterTagDetectionResult& outer_detection,
    const IntermediateCameraConfig* camera_override,
    const Eigen::Matrix4d* T_camera_board_prior) const {
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  const cv::Mat gray = ToGray(image);
  const BoardRuntime& runtime = RuntimeForBoardIdOrDefault(board_id);
  return DetectSingleBoardFromOuter(
      gray, runtime, outer_detection, camera_override, T_camera_board_prior);
}

ApriltagInternalMultiDetectionResult ApriltagInternalDetector::DetectMultipleFromOuterDetections(
    const cv::Mat& image,
    const OuterTagMultiDetectionResult& outer_multi_detection,
    const IntermediateCameraConfig* camera_override,
    const std::map<int, Eigen::Matrix4d>& T_camera_board_priors) const {
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  const cv::Mat gray = ToGray(image);
  ApriltagInternalMultiDetectionResult result;
  result.image_size = gray.size();
  result.requested_board_ids = outer_multi_detection.requested_board_ids;
  result.detections.reserve(result.requested_board_ids.size());

  for (std::size_t index = 0; index < result.requested_board_ids.size(); ++index) {
    const int board_id = result.requested_board_ids[index];
    const BoardRuntime& runtime = RuntimeForBoardIdOrDefault(board_id);
    OuterTagDetectionResult outer_detection;
    if (index < outer_multi_detection.detections.size()) {
      outer_detection = outer_multi_detection.detections[index];
    } else {
      outer_detection.board_id = board_id;
      outer_detection.failure_reason = OuterTagFailureReason::NoDetectionsAtAll;
      outer_detection.failure_reason_text = ToString(outer_detection.failure_reason);
    }

    const auto prior_it = T_camera_board_priors.find(board_id);
    const Eigen::Matrix4d* prior_ptr =
        prior_it != T_camera_board_priors.end() ? &prior_it->second : nullptr;
    result.detections.push_back(
        DetectSingleBoardFromOuter(gray, runtime, outer_detection, camera_override, prior_ptr));
  }

  return result;
}

void ApriltagInternalDetector::DrawDetectionsImpl(
    const ApriltagInternalDetectionResult& detections,
    const ApriltagCanonicalModel& model,
    cv::Mat* output_image,
    bool include_status_text) const {
  if (output_image == nullptr || output_image->empty()) {
    throw std::runtime_error("DrawDetections requires a valid output image.");
  }

  if (output_image->channels() == 1) {
    cv::cvtColor(*output_image, *output_image, cv::COLOR_GRAY2BGR);
  } else if (output_image->channels() == 4) {
    cv::cvtColor(*output_image, *output_image, cv::COLOR_BGRA2BGR);
  }

  outer_detector_->DrawDetection(detections.outer_detection, output_image,
                                 config_.enable_debug_output);

  if (!detections.tag_detected) {
    return;
  }

  if (detections.tag_detected) {
    const cv::Scalar outer_outline_color(165, 165, 165);
    cv::line(*output_image, detections.outer_corners[0], detections.outer_corners[1], outer_outline_color, 2);
    cv::line(*output_image, detections.outer_corners[1], detections.outer_corners[2], outer_outline_color, 2);
    cv::line(*output_image, detections.outer_corners[2], detections.outer_corners[3], outer_outline_color, 2);
    cv::line(*output_image, detections.outer_corners[3], detections.outer_corners[0], outer_outline_color, 2);
    cv::circle(*output_image, detections.tag_center, 5, cv::Scalar(220, 220, 220), 2);
    cv::putText(*output_image, "#" + std::to_string(detections.board_id),
                cv::Point(static_cast<int>(detections.tag_center.x) + 8,
                          static_cast<int>(detections.tag_center.y) + 8),
                cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(220, 220, 220), 1);
  }

  if (config_.enable_debug_output) {
    if (detections.projection_mode == InternalProjectionMode::SphereBorderLattice &&
        detections.border_boundary_model_valid) {
      const std::array<cv::Scalar, 4> border_curve_colors{
          cv::Scalar(110, 110, 230),
          cv::Scalar(90, 180, 235),
          cv::Scalar(110, 210, 120),
          cv::Scalar(220, 150, 110)};
      for (std::size_t edge_index = 0; edge_index < detections.border_curves_image.size(); ++edge_index) {
        const std::vector<cv::Point2f>& curve =
            detections.border_curves_image[edge_index];
        if (curve.size() < 2) {
          continue;
        }
        for (std::size_t sample_index = 1; sample_index < curve.size(); ++sample_index) {
          cv::line(*output_image, curve[sample_index - 1], curve[sample_index],
                   border_curve_colors[edge_index], 1, cv::LINE_AA);
        }
        for (const cv::Point2f& point : detections.border_support_points[edge_index]) {
          if (!IsInsideImage(point, detections.image_size)) {
            continue;
          }
          cv::circle(*output_image, point, 2, border_curve_colors[edge_index], cv::FILLED,
                     cv::LINE_AA);
        }
      }
    }

    for (const auto& debug : detections.internal_corner_debug) {
      const cv::Scalar predicted_color(0, 165, 255);
      const cv::Scalar border_seed_color(255, 180, 0);
      const cv::Scalar seed_color(255, 80, 255);
      const cv::Scalar refined_color(0, 220, 80);
      const cv::Scalar arrow2_color(120, 190, 120);
      const cv::Scalar border_arrow_color(160, 160, 160);
      const cv::Scalar seed_arrow_color(160, 110, 200);
      const bool has_explicit_seed_stage =
          UsesSphereSeedPipeline(detections.projection_mode) ||
          detections.projection_mode ==
              InternalProjectionMode::VirtualPinholePatchBoundarySeed;
      const bool has_border_seed_stage =
          detections.projection_mode == InternalProjectionMode::SphereBorderLattice &&
          debug.border_seed_valid;
      if (IsInsideImage(debug.predicted_image, detections.image_size)) {
        cv::drawMarker(*output_image, debug.predicted_image, cv::Scalar(255, 255, 255),
                       cv::MARKER_CROSS, 8, 3, cv::LINE_AA);
        cv::drawMarker(*output_image, debug.predicted_image, predicted_color,
                       cv::MARKER_CROSS, 6, 1, cv::LINE_AA);
        cv::circle(*output_image, debug.predicted_image, 2, cv::Scalar(255, 255, 255), cv::FILLED,
                   cv::LINE_AA);
        cv::circle(*output_image, debug.predicted_image, 1, predicted_color, cv::FILLED,
                   cv::LINE_AA);
      }
      if (UsesSphereSeedPipeline(detections.projection_mode)) {
        const int base_search_radius =
            ComputeSphereSearchRadiusOverlayPx(debug.local_module_scale);
        const double search_scale =
            debug.sphere_search_radius > 1e-9
                ? std::max(1.0, debug.adaptive_search_radius / debug.sphere_search_radius)
                : 1.0;
        const int adaptive_search_radius =
            std::max(base_search_radius,
                     static_cast<int>(std::lround(base_search_radius * search_scale)));
        const cv::Point2f search_center =
            has_border_seed_stage ? debug.border_seed_image : debug.predicted_image;
        if (IsInsideImage(search_center, detections.image_size)) {
          cv::circle(*output_image, search_center, adaptive_search_radius,
                     cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
          if (detections.projection_mode == InternalProjectionMode::SphereRayRefine &&
              debug.ray_refine_trust_radius > 0.0) {
            const int trust_radius =
                ComputeRayRefineTrustRadiusOverlayPx(debug.local_module_scale, adaptive_search_radius);
            cv::circle(*output_image, search_center, trust_radius,
                       cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
          }
        }
        if (has_border_seed_stage &&
            IsInsideImage(debug.border_seed_image, detections.image_size)) {
          cv::drawMarker(*output_image, debug.border_seed_image, cv::Scalar(255, 255, 255),
                         cv::MARKER_TRIANGLE_UP, 8, 3, cv::LINE_AA);
          cv::drawMarker(*output_image, debug.border_seed_image, border_seed_color,
                         cv::MARKER_TRIANGLE_UP, 6, 1, cv::LINE_AA);
          if (IsInsideImage(debug.predicted_image, detections.image_size)) {
            cv::line(*output_image, debug.predicted_image, debug.border_seed_image,
                     border_arrow_color, 1);
          }
        }
        if (IsInsideImage(debug.sphere_seed_image, detections.image_size)) {
          cv::drawMarker(*output_image, debug.sphere_seed_image, cv::Scalar(255, 255, 255),
                         cv::MARKER_DIAMOND, 8, 3, cv::LINE_AA);
          cv::drawMarker(*output_image, debug.sphere_seed_image, seed_color,
                         cv::MARKER_DIAMOND, 6, 1, cv::LINE_AA);
        }
        if (has_border_seed_stage &&
            IsInsideImage(debug.border_seed_image, detections.image_size) &&
            IsInsideImage(debug.sphere_seed_image, detections.image_size)) {
          cv::line(*output_image, debug.border_seed_image, debug.sphere_seed_image,
                   seed_arrow_color, 1);
        } else if (IsInsideImage(debug.predicted_image, detections.image_size) &&
                   IsInsideImage(debug.sphere_seed_image, detections.image_size)) {
          cv::line(*output_image, debug.predicted_image, debug.sphere_seed_image,
                   border_arrow_color, 1);
        }
        if (IsInsideImage(debug.sphere_seed_image, detections.image_size) &&
            IsInsideImage(debug.refined_image, detections.image_size)) {
          cv::line(*output_image, debug.sphere_seed_image, debug.refined_image,
                   arrow2_color, 1);
        }
        if (IsInsideImage(debug.refined_image, detections.image_size)) {
          cv::drawMarker(*output_image, debug.refined_image, cv::Scalar(255, 255, 255),
                         cv::MARKER_SQUARE, 7, 3, cv::LINE_AA);
          cv::drawMarker(*output_image, debug.refined_image, refined_color,
                         cv::MARKER_SQUARE, 5, 1, cv::LINE_AA);
        }
      } else if (has_explicit_seed_stage) {
        if (IsInsideImage(debug.sphere_seed_image, detections.image_size)) {
          cv::drawMarker(*output_image, debug.sphere_seed_image, cv::Scalar(255, 255, 255),
                         cv::MARKER_DIAMOND, 8, 3, cv::LINE_AA);
          cv::drawMarker(*output_image, debug.sphere_seed_image, seed_color,
                         cv::MARKER_DIAMOND, 6, 1, cv::LINE_AA);
        }
        if (IsInsideImage(debug.predicted_image, detections.image_size) &&
            IsInsideImage(debug.sphere_seed_image, detections.image_size)) {
          cv::line(*output_image, debug.predicted_image, debug.sphere_seed_image,
                   cv::Scalar(180, 180, 180), 1);
        }
        if (IsInsideImage(debug.sphere_seed_image, detections.image_size) &&
            IsInsideImage(debug.refined_image, detections.image_size)) {
          cv::line(*output_image, debug.sphere_seed_image, debug.refined_image,
                   arrow2_color, 1);
        }
        if (IsInsideImage(debug.refined_image, detections.image_size)) {
          cv::drawMarker(*output_image, debug.refined_image, cv::Scalar(255, 255, 255),
                         cv::MARKER_SQUARE, 7, 3, cv::LINE_AA);
          cv::drawMarker(*output_image, debug.refined_image, refined_color,
                         cv::MARKER_SQUARE, 5, 1, cv::LINE_AA);
        }
      } else if (IsInsideImage(debug.predicted_image, detections.image_size) &&
                 IsInsideImage(debug.refined_image, detections.image_size)) {
        cv::line(*output_image, debug.predicted_image, debug.refined_image,
                 cv::Scalar(180, 180, 180), 1);
      }
    }
  }

  std::vector<const InternalCornerDebugInfo*> debug_by_point(model.PointCount(), nullptr);
  for (const auto& debug : detections.internal_corner_debug) {
    if (debug.point_id >= 0 &&
        static_cast<std::size_t>(debug.point_id) < debug_by_point.size()) {
      debug_by_point[static_cast<std::size_t>(debug.point_id)] = &debug;
    }
  }

  for (const auto& measurement : detections.corners) {
    const CanonicalCorner& canonical_corner = model.corner(measurement.point_id);
    if (!canonical_corner.observable) {
      continue;
    }

    const cv::Point2f point(static_cast<float>(measurement.image_xy.x()),
                            static_cast<float>(measurement.image_xy.y()));

    cv::Scalar color;
    if (!measurement.valid) {
      color = cv::Scalar(0, 64, 255);
    } else if (measurement.corner_type == CornerType::Outer) {
      color = cv::Scalar(0, 255, 255);
    } else if (measurement.corner_type == CornerType::XCorner) {
      color = cv::Scalar(255, 0, 255);
    } else {
      color = cv::Scalar(0, 220, 0);
    }

    const bool use_light_internal_marker =
        config_.enable_debug_output &&
        UsesSphereSeedPipeline(detections.projection_mode) &&
        measurement.corner_type != CornerType::Outer;
    const int radius = measurement.corner_type == CornerType::Outer ? 4 : 3;
    if (use_light_internal_marker) {
      if (measurement.valid) {
        cv::circle(*output_image, point, 2, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
        cv::circle(*output_image, point, 2, color, 1, cv::LINE_AA);
      } else {
        cv::circle(*output_image, point, 2, color, 1, cv::LINE_AA);
      }
    } else if (measurement.valid) {
      cv::circle(*output_image, point, radius, color, -1);
    } else {
      cv::circle(*output_image, point, radius, color, 1);
    }

    if (config_.enable_debug_output && measurement.valid) {
      std::ostringstream label;
      label << measurement.point_id;
      const InternalCornerDebugInfo* debug_info =
          debug_by_point[static_cast<std::size_t>(measurement.point_id)];
      if (debug_info != nullptr && measurement.corner_type != CornerType::Outer) {
        if (UsesSphereSeedPipeline(detections.projection_mode)) {
          label << ":" << std::lround(measurement.quality * 100.0);
          if (detections.projection_mode == InternalProjectionMode::SphereBorderLattice &&
              debug_info->border_seed_valid) {
            label << " dPB=" << std::fixed << std::setprecision(1)
                  << debug_info->predicted_to_border_seed_displacement;
          }
        } else {
          label << ":" << std::lround(measurement.quality * 100.0);
          label << " subpix=" << debug_info->subpix_window_radius
                << " gate=" << std::fixed << std::setprecision(1)
                << debug_info->subpix_displacement_limit;
        }
      } else {
        label << ":" << std::lround(measurement.quality * 100.0);
      }
      cv::putText(*output_image, label.str(),
                  cv::Point(static_cast<int>(point.x) + 4, static_cast<int>(point.y) - 4),
                  cv::FONT_HERSHEY_PLAIN, 0.8, color, 1);
    }
  }

  if (include_status_text) {
    const std::string status =
        detections.success ? "status: valid apriltag_internal observation"
                           : "status: outer tag ok, but internal observation below threshold";
    cv::putText(*output_image, status, cv::Point(20, 28), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 2);

    std::ostringstream summary;
    summary << "mode: " << ToString(detections.projection_mode) << "  valid corners: "
            << detections.valid_corner_count << "/" << detections.expected_visible_point_count
            << "  internal: " << detections.valid_internal_corner_count;
    cv::putText(*output_image, summary.str(), cv::Point(20, 56), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 0), 2);

    std::ostringstream outer_summary;
    outer_summary << "outer scale: " << detections.outer_detection.chosen_scale_longest_side
                  << "  outer status: " << detections.outer_detection.failure_reason_text;
    cv::putText(*output_image, outer_summary.str(), cv::Point(20, 84), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(0, 200, 255), 2);

    if (config_.enable_debug_output &&
        (UsesSphereSeedPipeline(detections.projection_mode) ||
         detections.projection_mode == InternalProjectionMode::VirtualPinholePatchBoundarySeed)) {
      const std::string legend =
          detections.projection_mode == InternalProjectionMode::SphereBorderLattice
              ? "internal legend: P orange cross, BC blue triangle, SS magenta diamond, R green square"
              : "internal legend: P orange cross, SS magenta diamond, R green square";
      cv::putText(*output_image, legend, cv::Point(20, 112), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                  cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
      cv::putText(*output_image, legend, cv::Point(20, 112), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                  cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
    }
  }

  if (config_.enable_debug_output &&
      detections.projection_mode == InternalProjectionMode::SphereBorderLattice) {
    const std::array<const char*, 4> edge_names{{"T", "R", "B", "L"}};
    int diagnostics_y = static_cast<int>(detections.tag_center.y) - 42;
    diagnostics_y = std::max(diagnostics_y, include_status_text ? 138 : 24);
    const int diagnostics_x =
        std::max(18, std::min(static_cast<int>(detections.tag_center.x) + 18,
                              output_image->cols - 220));

    std::ostringstream header;
    header << "BC #" << detections.board_id;
    cv::putText(*output_image, header.str(), cv::Point(diagnostics_x, diagnostics_y),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(*output_image, header.str(), cv::Point(diagnostics_x, diagnostics_y),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);

    for (std::size_t edge_index = 0; edge_index < edge_names.size(); ++edge_index) {
      std::ostringstream line;
      line << edge_names[edge_index] << ":";
      if (detections.border_edge_valid[edge_index]) {
        line << " rms=" << std::fixed << std::setprecision(4)
             << detections.border_edge_rms_residual[edge_index]
             << " n=" << detections.border_edge_support_count[edge_index];
      } else {
        line << " invalid";
      }
      const int y = diagnostics_y + 16 * static_cast<int>(edge_index + 1);
      cv::putText(*output_image, line.str(), cv::Point(diagnostics_x, y),
                  cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
      cv::putText(*output_image, line.str(), cv::Point(diagnostics_x, y),
                  cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
    }
  }
}

void ApriltagInternalDetector::DrawDetections(const ApriltagInternalDetectionResult& detections,
                                              cv::Mat* output_image) const {
  DrawDetectionsImpl(detections, RuntimeForBoardIdOrDefault(detections.board_id).model,
                     output_image, true);
}

void ApriltagInternalDetector::DrawDetections(
    const ApriltagInternalMultiDetectionResult& detections,
    cv::Mat* output_image) const {
  if (output_image == nullptr || output_image->empty()) {
    throw std::runtime_error("DrawDetections requires a valid output image.");
  }

  if (output_image->channels() == 1) {
    cv::cvtColor(*output_image, *output_image, cv::COLOR_GRAY2BGR);
  } else if (output_image->channels() == 4) {
    cv::cvtColor(*output_image, *output_image, cv::COLOR_BGRA2BGR);
  }

  int detected_count = 0;
  int success_count = 0;
  int valid_internal_count = 0;
  for (const ApriltagInternalDetectionResult& detection : detections.detections) {
    DrawDetectionsImpl(detection, RuntimeForBoardIdOrDefault(detection.board_id).model,
                       output_image, false);
    if (detection.tag_detected) {
      ++detected_count;
    }
    if (detection.success) {
      ++success_count;
    }
    valid_internal_count += detection.valid_internal_corner_count;
  }

  const int banner_height = config_.enable_debug_output ? 132 : 96;
  cv::rectangle(*output_image, cv::Rect(0, 0, output_image->cols, banner_height),
                cv::Scalar(20, 20, 20), cv::FILLED);

  std::ostringstream requested_ids_stream;
  for (std::size_t index = 0; index < detections.requested_board_ids.size(); ++index) {
    if (index > 0) {
      requested_ids_stream << ",";
    }
    requested_ids_stream << detections.requested_board_ids[index];
  }

  const std::string headline = detections.AnySuccess()
                                   ? "status: apriltag_internal multi-board detection"
                                   : "status: no valid apriltag_internal observation";
  cv::putText(*output_image, headline, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.68,
              cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

  std::ostringstream summary;
  summary << "requested boards: [" << requested_ids_stream.str() << "]  detected: "
          << detected_count << "/" << detections.requested_board_ids.size()
          << "  valid observations: " << success_count << "/"
          << detections.requested_board_ids.size();
  cv::putText(*output_image, summary.str(), cv::Point(20, 58), cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

  std::ostringstream counts;
  counts << "mode: " << ToString(config_.internal_projection_mode)
         << "  total valid internal points: " << valid_internal_count;
  cv::putText(*output_image, counts.str(), cv::Point(20, 86), cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(0, 200, 255), 2, cv::LINE_AA);

  if (config_.enable_debug_output) {
    const std::string legend =
        config_.internal_projection_mode == InternalProjectionMode::SphereBorderLattice
            ? "internal legend: P orange cross, BC blue triangle, SS magenta diamond, R green square"
            : "internal legend: P orange cross, SS magenta diamond, R green square";
    cv::putText(*output_image, legend, cv::Point(20, 114), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  }
}

void ApriltagInternalDetector::DrawCanonicalViewImpl(
    const ApriltagInternalDetectionResult& detections,
    const ApriltagCanonicalModel& model,
    cv::Mat* output_image) const {
  if (output_image == nullptr || output_image->empty()) {
    throw std::runtime_error("DrawCanonicalView requires a valid patch image.");
  }

  if (output_image->channels() == 1) {
    cv::cvtColor(*output_image, *output_image, cv::COLOR_GRAY2BGR);
  } else if (output_image->channels() == 4) {
    cv::cvtColor(*output_image, *output_image, cv::COLOR_BGRA2BGR);
  }

  if (detections.projection_mode == InternalProjectionMode::Homography) {
    const int module_dimension = model.ModuleDimension();
    const int pixels_per_module = options_.canonical_pixels_per_module;
    const int patch_extent = module_dimension * pixels_per_module;
    for (int module = 0; module <= module_dimension; ++module) {
      const int coord = module * pixels_per_module;
      const cv::Scalar grid_color = (module == 0 || module == module_dimension)
                                        ? cv::Scalar(255, 255, 0)
                                        : cv::Scalar(80, 80, 80);
      cv::line(*output_image, cv::Point(coord, 0), cv::Point(coord, patch_extent), grid_color, 1);
      cv::line(*output_image, cv::Point(0, coord), cv::Point(patch_extent, coord), grid_color, 1);
    }
  }

  const std::vector<cv::Point> outer_polygon{
      cv::Point(static_cast<int>(std::lround(detections.patch_outer_corners[0].x)),
                static_cast<int>(std::lround(detections.patch_outer_corners[0].y))),
      cv::Point(static_cast<int>(std::lround(detections.patch_outer_corners[1].x)),
                static_cast<int>(std::lround(detections.patch_outer_corners[1].y))),
      cv::Point(static_cast<int>(std::lround(detections.patch_outer_corners[2].x)),
                static_cast<int>(std::lround(detections.patch_outer_corners[2].y))),
      cv::Point(static_cast<int>(std::lround(detections.patch_outer_corners[3].x)),
                static_cast<int>(std::lround(detections.patch_outer_corners[3].y))),
  };
  cv::polylines(*output_image, outer_polygon, true, cv::Scalar(0, 255, 255), 2);

  for (const auto& debug : detections.internal_corner_debug) {
    if (config_.enable_debug_output && IsInsideImage(debug.predicted_patch, output_image->size())) {
      cv::drawMarker(*output_image, debug.predicted_patch, cv::Scalar(0, 165, 255),
                     cv::MARKER_CROSS, 8, 1);
    }
    if (config_.enable_debug_output &&
        detections.projection_mode == InternalProjectionMode::VirtualPinholePatchBoundarySeed &&
        IsInsideImage(debug.sphere_seed_patch, output_image->size())) {
      cv::drawMarker(*output_image, debug.sphere_seed_patch, cv::Scalar(255, 80, 255),
                     cv::MARKER_DIAMOND, 8, 1);
    }
    if (config_.enable_debug_output &&
        detections.projection_mode == InternalProjectionMode::VirtualPinholePatchBoundarySeed &&
        IsInsideImage(debug.predicted_patch, output_image->size()) &&
        IsInsideImage(debug.sphere_seed_patch, output_image->size())) {
      cv::line(*output_image, debug.predicted_patch, debug.sphere_seed_patch,
               cv::Scalar(180, 180, 180), 1);
    }
    if (config_.enable_debug_output &&
        detections.projection_mode != InternalProjectionMode::VirtualPinholePatchBoundarySeed &&
        IsInsideImage(debug.predicted_patch, output_image->size()) &&
        IsInsideImage(debug.refined_patch, output_image->size())) {
      cv::line(*output_image, debug.predicted_patch, debug.refined_patch,
               cv::Scalar(180, 180, 180), 1);
    } else if (config_.enable_debug_output &&
               detections.projection_mode == InternalProjectionMode::VirtualPinholePatchBoundarySeed &&
               IsInsideImage(debug.sphere_seed_patch, output_image->size()) &&
               IsInsideImage(debug.refined_patch, output_image->size())) {
      cv::line(*output_image, debug.sphere_seed_patch, debug.refined_patch,
               cv::Scalar(120, 190, 120), 1);
    }
    if (IsInsideImage(debug.refined_patch, output_image->size())) {
      const cv::Scalar color = InternalCornerColor(debug);
      if (debug.valid) {
        cv::circle(*output_image, debug.refined_patch, 3, color, -1);
      } else {
        cv::circle(*output_image, debug.refined_patch, 3, color, 1);
      }

      if (config_.enable_debug_output) {
        std::ostringstream label;
        label << debug.point_id;
        if (debug.valid) {
          label << ":" << std::lround(debug.final_quality * 100.0);
        }
        label << " subpix=" << debug.subpix_window_radius
              << " gate=" << std::fixed << std::setprecision(1)
              << debug.subpix_displacement_limit;
        cv::putText(*output_image, label.str(),
                    cv::Point(static_cast<int>(debug.refined_patch.x) + 4,
                              static_cast<int>(debug.refined_patch.y) - 4),
                    cv::FONT_HERSHEY_PLAIN, 0.8, color, 1);
      }
    }
  }

  const std::string headline = detections.success
                                   ? std::string(ToString(detections.projection_mode)) +
                                         " patch: valid observation"
                                   : std::string(ToString(detections.projection_mode)) +
                                         " patch: below threshold";
  cv::putText(*output_image, headline, cv::Point(12, 22), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 255), 1);

  std::ostringstream summary;
  summary << "outer scale=" << detections.outer_detection.chosen_scale_longest_side
          << " valid=" << detections.valid_corner_count << "/"
          << detections.expected_visible_point_count;
  cv::putText(*output_image, summary.str(), cv::Point(12, 42), cv::FONT_HERSHEY_SIMPLEX, 0.45,
              cv::Scalar(255, 255, 0), 1);
}

void ApriltagInternalDetector::DrawCanonicalView(
    const ApriltagInternalDetectionResult& detections,
    cv::Mat* output_image) const {
  DrawCanonicalViewImpl(detections, RuntimeForBoardIdOrDefault(detections.board_id).model,
                        output_image);
}

cv::Mat ApriltagInternalDetector::ToGray(const cv::Mat& image) const {
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
