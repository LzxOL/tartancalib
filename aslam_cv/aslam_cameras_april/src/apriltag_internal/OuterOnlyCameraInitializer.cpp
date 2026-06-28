#include <aslam/cameras/apriltag_internal/OuterOnlyCameraInitializer.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>

#include <boost/filesystem.hpp>

#include <opencv2/calib3d.hpp>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

const char* ToString(AutoCameraInitializationRefineMode mode) {
  switch (mode) {
    case AutoCameraInitializationRefineMode::None:
      return "none";
    case AutoCameraInitializationRefineMode::CoordinateSearch:
      return "coordinate_search";
    case AutoCameraInitializationRefineMode::KalibrOuterLm:
      return "kalibr_outer_lm";
  }
  return "unknown";
}

AutoCameraInitializationRefineMode ParseAutoCameraInitializationRefineMode(
    const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (normalized == "none") {
    return AutoCameraInitializationRefineMode::None;
  }
  if (normalized == "coordinate_search" || normalized == "coordinate-search" ||
      normalized == "legacy") {
    return AutoCameraInitializationRefineMode::CoordinateSearch;
  }
  if (normalized == "kalibr_outer_lm" || normalized == "kalibr-outer-lm" ||
      normalized == "kalibr_lm" || normalized == "lm") {
    return AutoCameraInitializationRefineMode::KalibrOuterLm;
  }
  throw std::runtime_error(
      "Unsupported --stage5-init-refine-mode: " + value +
      " (supported: kalibr_outer_lm, coordinate_search, none)");
}

namespace {

struct OuterObservationRecord {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  double quality = 0.0;
  std::vector<Eigen::Vector3d> object_points;
  std::vector<cv::Point2f> image_points;
};

struct SeedConstructionDiagnostics {
  std::string seed_method = "unknown";
  std::string seed_source = "unknown";
  double omni_gamma = std::numeric_limits<double>::quiet_NaN();
  std::string omni_gamma_source = "unavailable";
  std::string ds_mapping = "not_applicable";
  int ds_mapping_verified_against_kalibr_source = 0;
  int ds_grid_enumeration_enabled = 0;
  std::string fallback_reason;
};

void AppendUniqueWarning(const std::string& warning,
                         std::vector<std::string>* warnings) {
  if (warnings == nullptr || warning.empty()) {
    return;
  }
  if (std::find(warnings->begin(), warnings->end(), warning) == warnings->end()) {
    warnings->push_back(warning);
  }
}

std::string JoinLabels(const std::vector<std::string>& labels,
                       const std::string& delimiter) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    if (index > 0) {
      stream << delimiter;
    }
    stream << labels[index];
  }
  return stream.str();
}

bool ClampIntrinsicsInPlace(OuterBootstrapCameraIntrinsics* intrinsics) {
  if (intrinsics == nullptr) {
    throw std::runtime_error("ClampIntrinsicsInPlace requires a valid pointer.");
  }
  const std::string family = intrinsics->NormalizedFamilyString();
  if (family == "ds-none") {
    intrinsics->xi = std::max(-0.95, std::min(2.5, intrinsics->xi));
    intrinsics->alpha = std::max(0.05, std::min(0.95, intrinsics->alpha));
    intrinsics->beta = 1.0;
    intrinsics->distortion_coeffs.clear();
  } else if (family == "eucm-none") {
    intrinsics->alpha = std::max(0.05, std::min(0.95, intrinsics->alpha));
    intrinsics->beta = std::max(0.25, std::min(3.0, intrinsics->beta));
    intrinsics->xi = 0.0;
    intrinsics->distortion_coeffs.clear();
  } else if (family == "pinhole-equi") {
    intrinsics->xi = 0.0;
    intrinsics->alpha = 0.0;
    intrinsics->beta = 0.0;
    if (intrinsics->distortion_coeffs.size() != 4) {
      intrinsics->distortion_coeffs.resize(4, 0.0);
    }
    for (double& coefficient : intrinsics->distortion_coeffs) {
      coefficient = std::max(-1.5, std::min(1.5, coefficient));
    }
  } else {
    return false;
  }
  intrinsics->fu = std::max(50.0, std::min(3.0 * intrinsics->resolution.width, intrinsics->fu));
  intrinsics->fv = std::max(50.0, std::min(3.0 * intrinsics->resolution.height, intrinsics->fv));
  intrinsics->cu =
      std::max(0.0, std::min(static_cast<double>(intrinsics->resolution.width), intrinsics->cu));
  intrinsics->cv =
      std::max(0.0, std::min(static_cast<double>(intrinsics->resolution.height), intrinsics->cv));
  return intrinsics->IsValid();
}

OuterBootstrapCameraIntrinsics MakeGenericSeedIntrinsics(
    const cv::Size& resolution,
    const ApriltagInternalConfig& config) {
  constexpr double kDefaultXi = 0.0;
  constexpr double kDefaultAlpha = 0.5;
  constexpr double kDefaultFuScale = 0.55;
  constexpr double kDefaultFvScale = 0.55;
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model =
      config.intermediate_camera.camera_model.empty()
          ? "ds"
          : config.intermediate_camera.camera_model;
  intrinsics.distortion_model =
      config.intermediate_camera.distortion_model.empty()
          ? (intrinsics.camera_model == "pinhole" ? "equi" : "none")
          : config.intermediate_camera.distortion_model;
  intrinsics.resolution = resolution;
  intrinsics.fu = kDefaultFuScale * static_cast<double>(resolution.width);
  intrinsics.fv = kDefaultFvScale * static_cast<double>(resolution.height);
  intrinsics.cu = 0.5 * static_cast<double>(resolution.width);
  intrinsics.cv = 0.5 * static_cast<double>(resolution.height);
  if (intrinsics.NormalizedFamilyString() == "ds-none") {
    intrinsics.xi = kDefaultXi;
    intrinsics.alpha = kDefaultAlpha;
  } else if (intrinsics.NormalizedFamilyString() == "eucm-none") {
    intrinsics.alpha = kDefaultAlpha;
    intrinsics.beta = 1.0;
  } else if (intrinsics.NormalizedFamilyString() == "pinhole-equi") {
    intrinsics.distortion_coeffs = {0.0, 0.0, 0.0, 0.0};
  }
  ClampIntrinsicsInPlace(&intrinsics);
  return intrinsics;
}

double FisheyeResolutionFocalPrior(const cv::Size& resolution) {
  return static_cast<double>(std::max(resolution.width, resolution.height)) /
         M_PI;
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t middle = values.size() / 2u;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  double median = values[middle];
  if (values.size() % 2u == 0u) {
    const auto lower_it = std::max_element(values.begin(), values.begin() + middle);
    median = 0.5 * (median + *lower_it);
  }
  return median;
}

bool AddFocalSquaredCandidate(double focal_squared,
                              const cv::Size& image_size,
                              std::vector<double>* focal_guesses) {
  if (focal_guesses == nullptr || !std::isfinite(focal_squared) ||
      focal_squared <= 0.0) {
    return false;
  }
  const double focal = std::sqrt(focal_squared);
  const double min_reasonable_focal =
      0.05 * static_cast<double>(std::min(image_size.width, image_size.height));
  const double max_reasonable_focal =
      4.0 * static_cast<double>(std::max(image_size.width, image_size.height));
  if (!std::isfinite(focal) || focal < min_reasonable_focal ||
      focal > max_reasonable_focal) {
    return false;
  }
  focal_guesses->push_back(focal);
  return true;
}

bool EstimatePinholeFocalFromOuterHomographies(
    const cv::Size& image_size,
    const std::vector<OuterObservationRecord>& observations,
    double* focal) {
  if (focal == nullptr || image_size.width <= 0 || image_size.height <= 0) {
    return false;
  }
  *focal = std::numeric_limits<double>::quiet_NaN();
  const double center_u = 0.5 * static_cast<double>(image_size.width - 1);
  const double center_v = 0.5 * static_cast<double>(image_size.height - 1);
  std::vector<double> focal_guesses;
  focal_guesses.reserve(observations.size() * 2u);

  for (const OuterObservationRecord& observation : observations) {
    if (observation.object_points.size() != 4u ||
        observation.image_points.size() != 4u) {
      continue;
    }
    std::vector<cv::Point2f> object_points;
    std::vector<cv::Point2f> image_points;
    object_points.reserve(4u);
    image_points.reserve(4u);
    for (std::size_t index = 0; index < 4u; ++index) {
      object_points.emplace_back(
          static_cast<float>(observation.object_points[index].x()),
          static_cast<float>(observation.object_points[index].y()));
      image_points.emplace_back(
          static_cast<float>(static_cast<double>(observation.image_points[index].x) -
                             center_u),
          static_cast<float>(static_cast<double>(observation.image_points[index].y) -
                             center_v));
    }

    const cv::Mat homography = cv::findHomography(object_points, image_points, 0);
    if (homography.empty() || homography.rows != 3 || homography.cols != 3) {
      continue;
    }
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        H(row, col) = homography.at<double>(row, col);
      }
    }
    if (!H.allFinite()) {
      continue;
    }
    if (std::abs(H(2, 2)) > 1e-12) {
      H /= H(2, 2);
    }

    const Eigen::Vector3d h1 = H.col(0);
    const Eigen::Vector3d h2 = H.col(1);
    const double denom_orthogonal = h1.z() * h2.z();
    if (std::abs(denom_orthogonal) > 1e-12) {
      const double focal_squared =
          -(h1.x() * h2.x() + h1.y() * h2.y()) / denom_orthogonal;
      AddFocalSquaredCandidate(focal_squared, image_size, &focal_guesses);
    }
    const double denom_equal_norm = h1.z() * h1.z() - h2.z() * h2.z();
    if (std::abs(denom_equal_norm) > 1e-12) {
      const double focal_squared =
          -((h1.x() * h1.x() + h1.y() * h1.y()) -
            (h2.x() * h2.x() + h2.y() * h2.y())) /
          denom_equal_norm;
      AddFocalSquaredCandidate(focal_squared, image_size, &focal_guesses);
    }
  }

  if (focal_guesses.empty()) {
    return false;
  }
  *focal = Median(focal_guesses);
  return std::isfinite(*focal) && *focal > 0.0;
}

bool ComputeDirectOuterQuadOmniGammaCandidate(
    const cv::Size& image_size,
    const std::vector<cv::Point2f>& points,
    double* gamma) {
  if (gamma == nullptr || image_size.width <= 0 || image_size.height <= 0 ||
      points.size() < 4u) {
    return false;
  }
  *gamma = std::numeric_limits<double>::quiet_NaN();
  const double cu = 0.5 * static_cast<double>(image_size.width - 1);
  const double cv = 0.5 * static_cast<double>(image_size.height - 1);
  cv::Mat P(static_cast<int>(points.size()), 4, CV_64F);
  for (std::size_t index = 0; index < points.size(); ++index) {
    const double u = static_cast<double>(points[index].x) - cu;
    const double v = static_cast<double>(points[index].y) - cv;
    P.at<double>(static_cast<int>(index), 0) = u;
    P.at<double>(static_cast<int>(index), 1) = v;
    P.at<double>(static_cast<int>(index), 2) = 0.5;
    P.at<double>(static_cast<int>(index), 3) = -0.5 * (u * u + v * v);
  }
  cv::Mat C;
  cv::SVD::solveZ(P, C);
  if (C.empty() || C.rows < 4) {
    return false;
  }
  const double c0 = C.at<double>(0);
  const double c1 = C.at<double>(1);
  const double c2 = C.at<double>(2);
  const double c3 = C.at<double>(3);
  const double t = c0 * c0 + c1 * c1 + c2 * c3;
  if (!std::isfinite(t) || t < 0.0) {
    return false;
  }
  const double d = std::sqrt(1.0 / t);
  const double nx = c0 * d;
  const double ny = c1 * d;
  if (std::hypot(nx, ny) > 0.95) {
    return false;
  }
  const double nz_squared = 1.0 - nx * nx - ny * ny;
  if (!std::isfinite(nz_squared) || nz_squared <= 0.0) {
    return false;
  }
  const double nz = std::sqrt(nz_squared);
  const double candidate_gamma = std::abs(c2 * d / nz);
  const double min_reasonable_gamma =
      0.05 * static_cast<double>(std::min(image_size.width, image_size.height));
  const double max_reasonable_gamma =
      8.0 * static_cast<double>(std::max(image_size.width, image_size.height));
  if (!std::isfinite(candidate_gamma) ||
      candidate_gamma < min_reasonable_gamma ||
      candidate_gamma > max_reasonable_gamma) {
    return false;
  }
  *gamma = candidate_gamma;
  return true;
}

std::vector<double> CollectDirectOuterQuadOmniGammaCandidates(
    const cv::Size& image_size,
    const std::vector<OuterObservationRecord>& observations,
    int* outer_quad_group_count,
    int* valid_gamma_group_count) {
  if (outer_quad_group_count != nullptr) {
    *outer_quad_group_count = 0;
  }
  if (valid_gamma_group_count != nullptr) {
    *valid_gamma_group_count = 0;
  }
  std::vector<double> gamma_candidates;
  gamma_candidates.reserve(observations.size());
  for (const OuterObservationRecord& observation : observations) {
    if (outer_quad_group_count != nullptr) {
      ++(*outer_quad_group_count);
    }
    double candidate_gamma = std::numeric_limits<double>::quiet_NaN();
    if (ComputeDirectOuterQuadOmniGammaCandidate(
            image_size, observation.image_points, &candidate_gamma)) {
      if (valid_gamma_group_count != nullptr) {
        ++(*valid_gamma_group_count);
      }
      gamma_candidates.push_back(candidate_gamma);
    }
  }
  std::sort(gamma_candidates.begin(), gamma_candidates.end());
  gamma_candidates.erase(
      std::unique(gamma_candidates.begin(), gamma_candidates.end(),
                  [](double lhs, double rhs) {
                    return std::abs(lhs - rhs) < 1e-6;
                  }),
      gamma_candidates.end());
  return gamma_candidates;
}

bool EstimateOmniGammaFromDirectOuterQuads(
    const cv::Size& image_size,
    const std::vector<OuterObservationRecord>& observations,
    double* gamma,
    std::string* failure_reason) {
  if (gamma == nullptr || image_size.width <= 0 || image_size.height <= 0) {
    if (failure_reason != nullptr) {
      *failure_reason = "invalid_image_size_or_output";
    }
    return false;
  }
  *gamma = std::numeric_limits<double>::quiet_NaN();
  int outer_quad_group_count = 0;
  int outer_quad_group_with_four_points_count = 0;
  const std::vector<double> gamma_candidates =
      CollectDirectOuterQuadOmniGammaCandidates(
          image_size,
          observations,
          &outer_quad_group_count,
          &outer_quad_group_with_four_points_count);

  if (gamma_candidates.empty()) {
    if (failure_reason != nullptr) {
      std::ostringstream stream;
      stream << "outer_quad_direct_omni_no_valid_gamma; outer_quad_groups="
             << outer_quad_group_count
             << " outer_quad_groups_with_four_points="
             << outer_quad_group_with_four_points_count;
      *failure_reason = stream.str();
    }
    return false;
  }
  *gamma = Median(gamma_candidates);
  return std::isfinite(*gamma) && *gamma > 0.0;
}

OuterBootstrapCameraIntrinsics MakeKalibrLikeOuterSeedIntrinsics(
    const cv::Size& image_size,
    const ApriltagInternalConfig& config,
    const std::vector<OuterObservationRecord>& observations,
    std::string* source_label,
    SeedConstructionDiagnostics* seed_diagnostics,
    std::vector<std::string>* warnings) {
  OuterBootstrapCameraIntrinsics seed = MakeGenericSeedIntrinsics(image_size, config);
  const std::string family = seed.NormalizedFamilyString();
  double pinhole_focal = std::numeric_limits<double>::quiet_NaN();
  bool has_homography_focal = false;
  if (family != "ds-none") {
    has_homography_focal =
        EstimatePinholeFocalFromOuterHomographies(
            image_size, observations, &pinhole_focal);
    if (!has_homography_focal) {
      pinhole_focal =
          0.5 * static_cast<double>(std::max(image_size.width, image_size.height));
      AppendUniqueWarning(
          "Outer homography focal initialization failed for non-DS seed; using "
          "a resolution-derived focal fallback.",
          warnings);
    }
  }

  seed.cu = 0.5 * static_cast<double>(image_size.width - 1);
  seed.cv = 0.5 * static_cast<double>(image_size.height - 1);
  if (seed_diagnostics != nullptr) {
    seed_diagnostics->seed_method = "outer_only_model_seed";
    seed_diagnostics->seed_source = "model_family_and_observations";
    seed_diagnostics->ds_mapping = "not_applicable";
    seed_diagnostics->ds_mapping_verified_against_kalibr_source = 0;
    seed_diagnostics->ds_grid_enumeration_enabled = 0;
  }
  if (family == "ds-none") {
    double omni_gamma = std::numeric_limits<double>::quiet_NaN();
    std::string omni_failure_reason;
    const bool has_omni_gamma =
        EstimateOmniGammaFromDirectOuterQuads(
            image_size, observations, &omni_gamma, &omni_failure_reason);
    seed.beta = 1.0;
    seed.distortion_coeffs.clear();
    if (has_omni_gamma) {
      seed.xi = 0.0;
      seed.alpha = 0.5;
      seed.fu = 0.5 * omni_gamma;
      seed.fv = 0.5 * omni_gamma;
      if (source_label != nullptr) {
        *source_label = "kalibr_source_verified_omni_to_ds_seed";
      }
      if (seed_diagnostics != nullptr) {
        seed_diagnostics->seed_method =
            "kalibr_source_verified_omni_to_ds_seed";
        seed_diagnostics->seed_source =
            "outer_quad_direct_omni_seed_observations";
        seed_diagnostics->omni_gamma = omni_gamma;
        seed_diagnostics->omni_gamma_source =
            "outer_quad_direct_four_corner_groups";
        seed_diagnostics->ds_mapping =
            "xi_0_alpha_0p5_fu_fv_0p5_omni_gamma";
        seed_diagnostics->ds_mapping_verified_against_kalibr_source = 1;
      }
    } else {
      seed = MakeGenericSeedIntrinsics(image_size, config);
      seed.cu = 0.5 * static_cast<double>(image_size.width - 1);
      seed.cv = 0.5 * static_cast<double>(image_size.height - 1);
      seed.xi = 0.0;
      seed.alpha = 0.5;
      seed.fu = FisheyeResolutionFocalPrior(image_size);
      seed.fv = seed.fu;
      seed.beta = 1.0;
      seed.distortion_coeffs.clear();
      if (source_label != nullptr) {
        *source_label = "fallback_outer_only_ds_seed";
      }
      if (seed_diagnostics != nullptr) {
        seed_diagnostics->seed_method = "fallback_outer_only_ds_seed";
        seed_diagnostics->seed_source =
            "resolution_fisheye_fov_prior";
        seed_diagnostics->omni_gamma =
            std::numeric_limits<double>::quiet_NaN();
        seed_diagnostics->omni_gamma_source = "unavailable";
        seed_diagnostics->ds_mapping = "fallback_no_omni_gamma";
        seed_diagnostics->ds_mapping_verified_against_kalibr_source = 0;
        seed_diagnostics->fallback_reason = omni_failure_reason.empty()
                                                ? "kalibr_omni_gamma_unavailable"
                                                : omni_failure_reason;
      }
      AppendUniqueWarning(
          "Kalibr DS initializer source was found, but direct four-corner "
          "outer-quad Omni gamma initialization did not produce a valid gamma; "
          "using fallback_outer_only_ds_seed without 0.5 * "
          "pinhole-homography focal mapping. The fallback uses canonical "
          "xi=0, alpha=0.5 and a resolution/pi fisheye focal prior. "
          "fallback_reason=" +
              (omni_failure_reason.empty()
                   ? std::string("kalibr_omni_gamma_unavailable")
                   : omni_failure_reason),
          warnings);
    }
  } else if (family == "eucm-none") {
    seed.alpha = 0.5;
    seed.beta = 1.0;
    seed.xi = 0.0;
    seed.fu = 0.5 * pinhole_focal;
    seed.fv = 0.5 * pinhole_focal;
    seed.distortion_coeffs.clear();
    if (source_label != nullptr) {
      *source_label = has_homography_focal
                          ? "outer_homography_eucm_seed"
                          : "outer_resolution_eucm_seed";
    }
  } else if (family == "pinhole-equi") {
    seed.xi = 0.0;
    seed.alpha = 0.0;
    seed.beta = 0.0;
    seed.fu = pinhole_focal;
    seed.fv = pinhole_focal;
    seed.distortion_coeffs = {0.0, 0.0, 0.0, 0.0};
    if (source_label != nullptr) {
      *source_label = has_homography_focal
                          ? "outer_homography_pinhole_equi_seed"
                          : "outer_resolution_pinhole_equi_seed";
    }
  } else if (source_label != nullptr) {
    *source_label = "outer_unsupported_family_seed";
  }
  ClampIntrinsicsInPlace(&seed);
  return seed;
}

OuterBootstrapCameraIntrinsics BuildManualInitialCamera(
    const cv::Size& image_size,
    const ApriltagInternalConfig& config,
    const AutoCameraInitializationOptions& options,
    bool* used_manual_intermediate_camera,
    bool* used_explicit_initial_camera,
    bool* used_manual_generic_seed,
    std::string* source_label,
    std::vector<std::string>* warnings) {
  if (used_manual_intermediate_camera != nullptr) {
    *used_manual_intermediate_camera = false;
  }
  if (used_explicit_initial_camera != nullptr) {
    *used_explicit_initial_camera = false;
  }
  if (used_manual_generic_seed != nullptr) {
    *used_manual_generic_seed = false;
  }
  if (source_label != nullptr) {
    *source_label = "manual_generic_seed";
  }

  if (options.use_explicit_initial_camera) {
    OuterBootstrapCameraIntrinsics intrinsics =
        options.explicit_initial_camera;
    intrinsics.resolution = image_size;
    if (ClampIntrinsicsInPlace(&intrinsics)) {
      if (used_explicit_initial_camera != nullptr) {
        *used_explicit_initial_camera = true;
      }
      if (source_label != nullptr) {
        *source_label = options.explicit_initial_camera_source_label.empty()
                            ? "explicit_initial_camera"
                            : options.explicit_initial_camera_source_label;
      }
      return intrinsics;
    }
    AppendUniqueWarning(
        "Explicit initial camera is invalid after clamping; using a "
        "--models-derived generic seed instead.",
        warnings);
  }

  if (used_manual_generic_seed != nullptr) {
    *used_manual_generic_seed = true;
  }
  if (source_label != nullptr) {
    *source_label = "manual_generic_seed";
  }
  if (config.intermediate_camera.IsConfigured()) {
    AppendUniqueWarning(
        "Configured intermediate_camera intrinsics are ignored by Stage5 camera "
        "initialization; using a --models-derived generic seed instead.",
        warnings);
  }
  return MakeGenericSeedIntrinsics(image_size, config);
}

std::array<Eigen::Vector3d, 4> BuildOuterCornerPoints(const ApriltagInternalConfig& config,
                                                      int board_id) {
  ApriltagInternalConfig board_config = config;
  board_config.tag_id = board_id;
  board_config.tag_ids = {board_id};
  board_config.outer_detector_config.tag_id = board_id;
  board_config.outer_detector_config.tag_ids = {board_id};
  const ApriltagCanonicalModel model(board_config);
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

bool IsValidOuterMeasurement(const OuterBoardMeasurement& measurement) {
  if (!measurement.success || measurement.valid_refined_corner_count != 4) {
    return false;
  }
  for (bool valid : measurement.refined_corner_valid) {
    if (!valid) {
      return false;
    }
  }
  return true;
}

std::vector<OuterObservationRecord> CollectOuterObservations(
    const std::vector<OuterBootstrapFrameInput>& frames,
    const ApriltagInternalConfig& config) {
  std::vector<OuterObservationRecord> observations;
  for (const OuterBootstrapFrameInput& frame : frames) {
    for (const OuterBoardMeasurement& measurement : frame.measurements.board_measurements) {
      if (!IsValidOuterMeasurement(measurement)) {
        continue;
      }
      const std::array<Eigen::Vector3d, 4> object_points =
          BuildOuterCornerPoints(config, measurement.board_id);
      OuterObservationRecord observation;
      observation.frame_index = frame.frame_index;
      observation.frame_label = frame.frame_label;
      observation.board_id = measurement.board_id;
      observation.quality = measurement.detection_quality;
      observation.object_points.assign(object_points.begin(), object_points.end());
      observation.image_points.reserve(4);
      for (int index = 0; index < 4; ++index) {
        const Eigen::Vector2d& point =
            measurement.refined_outer_corners_original_image[static_cast<std::size_t>(index)];
        observation.image_points.push_back(
            cv::Point2f(static_cast<float>(point.x()), static_cast<float>(point.y())));
      }
      observations.push_back(observation);
    }
  }
  std::sort(observations.begin(), observations.end(),
            [](const OuterObservationRecord& lhs, const OuterObservationRecord& rhs) {
              if (lhs.frame_index != rhs.frame_index) {
                return lhs.frame_index < rhs.frame_index;
              }
              return lhs.board_id < rhs.board_id;
            });
  return observations;
}

cv::Size InferImageSize(const std::vector<OuterBootstrapFrameInput>& frames) {
  for (const OuterBootstrapFrameInput& frame : frames) {
    if (frame.measurements.image_size.width > 0 && frame.measurements.image_size.height > 0) {
      return frame.measurements.image_size;
    }
  }
  return cv::Size();
}

std::vector<OuterObservationRecord> SampleObservations(
    const std::vector<OuterObservationRecord>& observations,
    int max_count) {
  if (max_count <= 0 || static_cast<int>(observations.size()) <= max_count) {
    return observations;
  }
  std::vector<OuterObservationRecord> sampled;
  sampled.reserve(static_cast<std::size_t>(max_count));
  if (max_count == 1) {
    sampled.push_back(observations.front());
    return sampled;
  }

  for (int index = 0; index < max_count; ++index) {
    const double alpha = static_cast<double>(index) /
                         static_cast<double>(max_count - 1);
    const std::size_t sample_index = static_cast<std::size_t>(std::lround(
        alpha * static_cast<double>(observations.size() - 1)));
    sampled.push_back(observations[sample_index]);
  }
  return sampled;
}

double ObservationObjectExtent(const OuterObservationRecord& observation) {
  double max_distance = 0.0;
  for (std::size_t lhs = 0; lhs < observation.object_points.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < observation.object_points.size(); ++rhs) {
      max_distance =
          std::max(max_distance,
                   (observation.object_points[lhs] -
                    observation.object_points[rhs]).norm());
    }
  }
  return max_distance;
}

double RotationDistanceDeg(const Eigen::Matrix3d& lhs,
                           const Eigen::Matrix3d& rhs) {
  Eigen::AngleAxisd angle_axis(lhs.transpose() * rhs);
  double angle = std::abs(angle_axis.angle());
  if (angle > M_PI) {
    angle = 2.0 * M_PI - angle;
  }
  return angle * 180.0 / M_PI;
}

void EvaluateRelativeLayoutConsistency(
    const std::vector<OuterObservationRecord>& observations,
    const std::map<int, std::vector<std::pair<int, Eigen::Isometry3d>>>&
        successful_poses_by_frame,
    AutoCameraInitializationCandidate* candidate) {
  if (candidate == nullptr || successful_poses_by_frame.empty()) {
    return;
  }
  double board_extent_sum = 0.0;
  int board_extent_count = 0;
  for (const OuterObservationRecord& observation : observations) {
    const double extent = ObservationObjectExtent(observation);
    if (std::isfinite(extent) && extent > 1e-9) {
      board_extent_sum += extent;
      ++board_extent_count;
    }
  }
  const double board_extent =
      board_extent_count > 0
          ? board_extent_sum / static_cast<double>(board_extent_count)
          : 1.0;

  std::map<std::pair<int, int>, std::vector<Eigen::Isometry3d>> relatives_by_pair;
  for (const auto& frame_entry : successful_poses_by_frame) {
    const std::vector<std::pair<int, Eigen::Isometry3d>>& poses =
        frame_entry.second;
    if (poses.size() < 2u) {
      continue;
    }
    for (std::size_t lhs = 0; lhs < poses.size(); ++lhs) {
      for (std::size_t rhs = lhs + 1; rhs < poses.size(); ++rhs) {
        const int board_a = poses[lhs].first;
        const int board_b = poses[rhs].first;
        if (board_a == board_b) {
          continue;
        }
        const bool natural_order = board_a < board_b;
        const Eigen::Isometry3d& T_camera_a =
            natural_order ? poses[lhs].second : poses[rhs].second;
        const Eigen::Isometry3d& T_camera_b =
            natural_order ? poses[rhs].second : poses[lhs].second;
        const std::pair<int, int> key =
            natural_order ? std::make_pair(board_a, board_b)
                          : std::make_pair(board_b, board_a);
        relatives_by_pair[key].push_back(T_camera_a.inverse() * T_camera_b);
      }
    }
  }

  double translation_squared_sum = 0.0;
  double rotation_squared_sum = 0.0;
  int pairwise_difference_count = 0;
  int pair_family_count = 0;
  int pair_sample_count = 0;
  for (const auto& pair_entry : relatives_by_pair) {
    const std::vector<Eigen::Isometry3d>& relatives = pair_entry.second;
    pair_sample_count += static_cast<int>(relatives.size());
    if (relatives.size() < 2u) {
      continue;
    }
    ++pair_family_count;
    for (std::size_t lhs = 0; lhs < relatives.size(); ++lhs) {
      for (std::size_t rhs = lhs + 1; rhs < relatives.size(); ++rhs) {
        const double translation_error =
            (relatives[lhs].translation() -
             relatives[rhs].translation()).norm() /
            std::max(1e-9, board_extent);
        const double rotation_error_deg =
            RotationDistanceDeg(relatives[lhs].linear(), relatives[rhs].linear());
        if (!std::isfinite(translation_error) ||
            !std::isfinite(rotation_error_deg)) {
          continue;
        }
        translation_squared_sum += translation_error * translation_error;
        rotation_squared_sum += rotation_error_deg * rotation_error_deg;
        ++pairwise_difference_count;
      }
    }
  }

  candidate->relative_layout_pair_family_count = pair_family_count;
  candidate->relative_layout_pair_sample_count = pair_sample_count;
  if (pairwise_difference_count <= 0) {
    return;
  }
  candidate->relative_layout_translation_rmse =
      std::sqrt(translation_squared_sum /
                static_cast<double>(pairwise_difference_count));
  candidate->relative_layout_rotation_rmse_deg =
      std::sqrt(rotation_squared_sum /
                static_cast<double>(pairwise_difference_count));
  candidate->relative_layout_consistency_score =
      candidate->relative_layout_translation_rmse +
      0.02 * candidate->relative_layout_rotation_rmse_deg;
}

AutoCameraInitializationCandidate EvaluateCandidateOnObservations(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::string& source_label,
    const std::string& evaluation_scope,
    const std::vector<OuterObservationRecord>& observations) {
  AutoCameraInitializationCandidate candidate;
  candidate.source_label = source_label;
  candidate.evaluation_scope = evaluation_scope;
  candidate.camera = camera;
  candidate.observation_count = static_cast<int>(observations.size());

  if (!candidate.camera.IsValid()) {
    candidate.failure_reason = "candidate intrinsics are invalid";
    return candidate;
  }

  double total_squared_rmse = 0.0;
  std::set<int> successful_frames;
  std::set<int> successful_boards;
  std::map<int, std::vector<std::pair<int, Eigen::Isometry3d>>>
      successful_poses_by_frame;
  for (const OuterObservationRecord& observation : observations) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double observation_rmse = 0.0;
    if (!EstimatePoseFromObjectPoints(candidate.camera,
                                      observation.object_points,
                                      observation.image_points,
                                      &pose,
                                      &observation_rmse)) {
      ++candidate.pose_failure_count;
      continue;
    }
    ++candidate.pose_success_count;
    total_squared_rmse += observation_rmse * observation_rmse;
    successful_frames.insert(observation.frame_index);
    successful_boards.insert(observation.board_id);
    successful_poses_by_frame[observation.frame_index].push_back(
        std::make_pair(observation.board_id, pose));
  }

  candidate.successful_frame_count = static_cast<int>(successful_frames.size());
  candidate.successful_board_count = static_cast<int>(successful_boards.size());
  if (candidate.observation_count > 0) {
    candidate.success_rate =
        static_cast<double>(candidate.pose_success_count) /
        static_cast<double>(candidate.observation_count);
  }
  if (candidate.pose_success_count > 0) {
    candidate.mean_observation_rmse =
        std::sqrt(total_squared_rmse /
                  static_cast<double>(candidate.pose_success_count));
    candidate.valid = true;
  } else {
    candidate.failure_reason = "no outer pose fits succeeded";
  }
  EvaluateRelativeLayoutConsistency(
      observations, successful_poses_by_frame, &candidate);

  return candidate;
}

bool CandidateIsBetter(const AutoCameraInitializationCandidate& lhs,
                       const AutoCameraInitializationCandidate& rhs) {
  if (lhs.valid != rhs.valid) {
    return lhs.valid;
  }
  if (lhs.pose_success_count != rhs.pose_success_count) {
    return lhs.pose_success_count > rhs.pose_success_count;
  }
  if (std::abs(lhs.success_rate - rhs.success_rate) > 1e-12) {
    return lhs.success_rate > rhs.success_rate;
  }
  const bool lhs_basic_reprojection_health =
      std::isfinite(lhs.mean_observation_rmse) &&
      lhs.mean_observation_rmse < 80.0;
  const bool rhs_basic_reprojection_health =
      std::isfinite(rhs.mean_observation_rmse) &&
      rhs.mean_observation_rmse < 80.0;
  if (lhs_basic_reprojection_health != rhs_basic_reprojection_health) {
    return lhs_basic_reprojection_health;
  }
  if (!lhs_basic_reprojection_health &&
      std::abs(lhs.mean_observation_rmse - rhs.mean_observation_rmse) > 1e-12) {
    return lhs.mean_observation_rmse < rhs.mean_observation_rmse;
  }
  // Layout and leave-one-board-out metrics are diagnostics only here. They can
  // be explained by pose/layout compensation, so do not let them rank the
  // camera seed that enters the independent outer-corner LM.
  if (std::abs(lhs.mean_observation_rmse - rhs.mean_observation_rmse) > 1e-12) {
    return lhs.mean_observation_rmse < rhs.mean_observation_rmse;
  }
  if (lhs.successful_frame_count != rhs.successful_frame_count) {
    return lhs.successful_frame_count > rhs.successful_frame_count;
  }
  return lhs.successful_board_count > rhs.successful_board_count;
}

double CandidateObjective(const AutoCameraInitializationCandidate& candidate) {
  if (!candidate.valid || candidate.observation_count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  const double fail_fraction =
      static_cast<double>(candidate.pose_failure_count) /
      static_cast<double>(candidate.observation_count);
  const double rmse = candidate.mean_observation_rmse;
  return rmse * rmse +
         2500.0 * fail_fraction;
}

bool IsAcceptableAutoCandidate(const AutoCameraInitializationCandidate& candidate,
                               int total_observation_count) {
  const int min_success = std::min(total_observation_count, 6);
  const double min_success_rate = total_observation_count >= 12 ? 0.4 : 0.25;
  return candidate.valid &&
         candidate.pose_success_count >= min_success &&
         candidate.success_rate >= min_success_rate &&
         candidate.mean_observation_rmse < 20.0;
}

bool IsRefinableKalibrLikeSeed(const AutoCameraInitializationCandidate& candidate,
                               int total_observation_count) {
  const int min_success = std::min(total_observation_count, 6);
  const double min_success_rate = total_observation_count >= 12 ? 0.7 : 0.5;
  const bool outer_seed_source =
      candidate.source_label.find("outer") != std::string::npos ||
      candidate.source_label.find("seed") != std::string::npos;
  return outer_seed_source &&
         candidate.valid &&
         candidate.pose_success_count >= min_success &&
         candidate.success_rate >= min_success_rate &&
         std::isfinite(candidate.mean_observation_rmse) &&
         candidate.mean_observation_rmse < 150.0;
}

double ParameterStep(double value, double fallback_step) {
  return std::max(std::abs(value) * 0.05, fallback_step);
}

Eigen::VectorXd ToEigenVector(const std::vector<double>& values) {
  Eigen::VectorXd vector(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    vector[static_cast<Eigen::Index>(index)] = values[index];
  }
  return vector;
}

std::vector<double> ToStdVector(const Eigen::VectorXd& values) {
  std::vector<double> vector(static_cast<std::size_t>(values.rows()), 0.0);
  for (Eigen::Index index = 0; index < values.rows(); ++index) {
    vector[static_cast<std::size_t>(index)] = values[index];
  }
  return vector;
}

double ParameterFallbackStep(const std::string& label,
                             const OuterBootstrapCameraIntrinsics& camera) {
  if (label == "fu" || label == "fv") {
    return 40.0;
  }
  if (label == "cu" || label == "cv") {
    return 20.0;
  }
  if (label == "xi" || label == "alpha" || label == "beta") {
    return 0.08;
  }
  (void)camera;
  return 0.02;
}

OuterBootstrapCameraIntrinsics RefineCandidateCamera(
    const OuterBootstrapCameraIntrinsics& initial_camera,
    const std::vector<OuterObservationRecord>& observations) {
  OuterBootstrapCameraIntrinsics best = initial_camera;
  AutoCameraInitializationCandidate best_eval =
      EvaluateCandidateOnObservations(best, "auto_grid_refined", "full", observations);
  double best_objective = CandidateObjective(best_eval);
  if (!std::isfinite(best_objective)) {
    return initial_camera;
  }

  const std::vector<std::string> labels = best.CombinedParameterLabels();
  Eigen::VectorXd best_vector = ToEigenVector(best.CombinedParameterVector());
  std::vector<double> base_steps(labels.size(), 0.05);
  for (std::size_t index = 0; index < labels.size(); ++index) {
    base_steps[index] =
        ParameterStep(best_vector[static_cast<Eigen::Index>(index)],
                      ParameterFallbackStep(labels[index], best));
  }

  for (int round = 0; round < 4; ++round) {
    const double round_scale = std::pow(0.5, round);
    for (std::size_t parameter_index = 0; parameter_index < labels.size(); ++parameter_index) {
      for (int direction = -1; direction <= 1; direction += 2) {
        OuterBootstrapCameraIntrinsics candidate = best;
        Eigen::VectorXd candidate_vector =
            ToEigenVector(candidate.CombinedParameterVector());
        const double delta = static_cast<double>(direction) *
                             base_steps[parameter_index] * round_scale;
        candidate_vector[static_cast<Eigen::Index>(parameter_index)] += delta;
        candidate.SetCombinedParameterVector(ToStdVector(candidate_vector));
        if (!ClampIntrinsicsInPlace(&candidate)) {
          continue;
        }
        const AutoCameraInitializationCandidate candidate_eval =
            EvaluateCandidateOnObservations(candidate, "auto_grid_refined", "full", observations);
        const double candidate_objective = CandidateObjective(candidate_eval);
        if (candidate_objective + 1e-9 < best_objective) {
          best = candidate;
          best_eval = candidate_eval;
          best_objective = candidate_objective;
        }
      }
    }
  }

  return best;
}

struct KalibrOuterLmView {
  const OuterObservationRecord* observation = nullptr;
  Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
};

struct KalibrOuterLmFrameView {
  int frame_index = -1;
  std::string frame_label;
  std::vector<const OuterObservationRecord*> observations;
  Eigen::Isometry3d T_camera_reference = Eigen::Isometry3d::Identity();
};

struct KalibrOuterLmEvaluation {
  bool success = false;
  Eigen::VectorXd residuals;
  OuterBootstrapCameraIntrinsics camera;
  int point_count = 0;
  int invalid_projection_count = 0;
  int nonfinite_count = 0;
  double rmse = std::numeric_limits<double>::infinity();
};

struct KalibrOuterLmRefinementResult {
  OuterBootstrapCameraIntrinsics camera;
  int view_count = 0;
  int residual_count = 0;
  int invalid_projection_count = 0;
  int nonfinite_count = 0;
  int iteration_count = 0;
  double initial_rmse = std::numeric_limits<double>::infinity();
  double final_rmse = std::numeric_limits<double>::infinity();
  bool improved = false;
};

double FrameCohesionLmObjective(
    const KalibrOuterLmRefinementResult& refinement) {
  if (!std::isfinite(refinement.final_rmse) ||
      refinement.residual_count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  const double invalid_fraction =
      static_cast<double>(refinement.invalid_projection_count +
                          refinement.nonfinite_count) /
      static_cast<double>(std::max(1, refinement.residual_count));
  return refinement.final_rmse * refinement.final_rmse +
         10000.0 * invalid_fraction;
}

struct BootstrapLayout {
  bool success = false;
  int used_frame_count = 0;
  int used_board_observation_count = 0;
  double global_rmse = std::numeric_limits<double>::infinity();
  std::map<int, Eigen::Isometry3d> T_reference_board_by_id;
  std::map<int, Eigen::Isometry3d> T_camera_reference_by_frame;
  std::vector<std::string> warnings;
};

double NumericJacobianStep(int parameter_index,
                           int camera_parameter_count,
                           const std::vector<std::string>& camera_labels,
                           const Eigen::VectorXd& x);

struct KalibrOuterLmViewCandidate {
  KalibrOuterLmView view;
  Eigen::MatrixXd camera_information;
  int image_bin = -1;
  int radial_bin = -1;
  bool valid = false;
};

struct KalibrOuterLmFrameCandidate {
  int frame_index = -1;
  std::string frame_label;
  std::vector<KalibrOuterLmViewCandidate> views;
  Eigen::MatrixXd camera_information;
  std::set<int> board_ids;
  std::set<int> image_bins;
  std::set<int> radial_bins;
  bool valid = false;
};

Eigen::Matrix<double, 6, 1> PoseToVector(const Eigen::Isometry3d& pose) {
  Eigen::Matrix<double, 6, 1> vector;
  vector.head<3>() = pose.translation();
  Eigen::AngleAxisd angle_axis(pose.linear());
  if (!std::isfinite(angle_axis.angle()) || angle_axis.angle() < 1e-12) {
    vector.tail<3>().setZero();
  } else {
    vector.tail<3>() = angle_axis.angle() * angle_axis.axis();
  }
  return vector;
}

Eigen::Isometry3d VectorToPose(const Eigen::Matrix<double, 6, 1>& vector) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = vector.head<3>();
  const Eigen::Vector3d rotation_vector = vector.tail<3>();
  const double angle = rotation_vector.norm();
  if (angle > 1e-12) {
    pose.linear() =
        Eigen::AngleAxisd(angle, rotation_vector / angle).toRotationMatrix();
  }
  return pose;
}

Eigen::VectorXd BuildSingleObservationResidual(
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const OuterObservationRecord& observation,
    const Eigen::Isometry3d& T_camera_board) {
  Eigen::VectorXd residuals =
      Eigen::VectorXd::Constant(
          2 * static_cast<int>(observation.object_points.size()), 100.0);
  DoubleSphereCameraModel camera;
  try {
    camera = DoubleSphereCameraModel::FromConfig(
        MakeIntermediateCameraConfig(intrinsics));
  } catch (const std::exception&) {
    return residuals;
  }
  int row = 0;
  for (std::size_t point_index = 0;
       point_index < observation.object_points.size(); ++point_index) {
    Eigen::Vector2d projected = Eigen::Vector2d::Zero();
    if (!camera.vsEuclideanToKeypoint(
            T_camera_board * observation.object_points[point_index],
            &projected) ||
        !projected.allFinite()) {
      row += 2;
      continue;
    }
    residuals[row++] =
        projected.x() - static_cast<double>(observation.image_points[point_index].x);
    residuals[row++] =
        projected.y() - static_cast<double>(observation.image_points[point_index].y);
  }
  return residuals;
}

double SafeLogDet(const Eigen::MatrixXd& matrix) {
  if (matrix.rows() == 0 || matrix.cols() == 0) {
    return 0.0;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(matrix);
  if (solver.info() != Eigen::Success) {
    return -std::numeric_limits<double>::infinity();
  }
  double value = 0.0;
  for (Eigen::Index index = 0; index < solver.eigenvalues().rows(); ++index) {
    value += std::log(std::max(1e-12, solver.eigenvalues()[index]));
  }
  return value;
}

int ImageCoverageBin(const OuterObservationRecord& observation,
                     const cv::Size& image_size) {
  if (observation.image_points.empty() || image_size.width <= 0 ||
      image_size.height <= 0) {
    return -1;
  }
  double mean_x = 0.0;
  double mean_y = 0.0;
  for (const cv::Point2f& point : observation.image_points) {
    mean_x += static_cast<double>(point.x);
    mean_y += static_cast<double>(point.y);
  }
  mean_x /= static_cast<double>(observation.image_points.size());
  mean_y /= static_cast<double>(observation.image_points.size());
  const int col = std::max(0, std::min(3, static_cast<int>(
      std::floor(4.0 * mean_x / static_cast<double>(image_size.width)))));
  const int row = std::max(0, std::min(3, static_cast<int>(
      std::floor(4.0 * mean_y / static_cast<double>(image_size.height)))));
  return row * 4 + col;
}

int RadialCoverageBin(const OuterObservationRecord& observation,
                      const cv::Size& image_size) {
  if (observation.image_points.empty() || image_size.width <= 0 ||
      image_size.height <= 0) {
    return -1;
  }
  double mean_x = 0.0;
  double mean_y = 0.0;
  for (const cv::Point2f& point : observation.image_points) {
    mean_x += static_cast<double>(point.x);
    mean_y += static_cast<double>(point.y);
  }
  mean_x /= static_cast<double>(observation.image_points.size());
  mean_y /= static_cast<double>(observation.image_points.size());
  const double nx =
      (mean_x - 0.5 * static_cast<double>(image_size.width)) /
      std::max(1.0, 0.5 * static_cast<double>(image_size.width));
  const double ny =
      (mean_y - 0.5 * static_cast<double>(image_size.height)) /
      std::max(1.0, 0.5 * static_cast<double>(image_size.height));
  const double radius = std::sqrt(nx * nx + ny * ny);
  if (radius < 0.30) {
    return 0;
  }
  if (radius < 0.55) {
    return 1;
  }
  if (radius < 0.80) {
    return 2;
  }
  return 3;
}

Eigen::MatrixXd ComputeCameraInformationForObservation(
    const OuterBootstrapCameraIntrinsics& camera,
    const OuterObservationRecord& observation,
    const Eigen::Isometry3d& T_camera_board) {
  const std::vector<double> parameters = camera.CombinedParameterVector();
  const std::vector<std::string> labels = camera.CombinedParameterLabels();
  const int parameter_count = static_cast<int>(parameters.size());
  Eigen::MatrixXd jacobian(
      2 * static_cast<int>(observation.object_points.size()),
      parameter_count);
  jacobian.setZero();
  for (int column = 0; column < parameter_count; ++column) {
    const Eigen::VectorXd x = ToEigenVector(parameters);
    const double step = NumericJacobianStep(column, parameter_count, labels, x);
    Eigen::VectorXd plus_vector = x;
    Eigen::VectorXd minus_vector = x;
    plus_vector[column] += step;
    minus_vector[column] -= step;
    OuterBootstrapCameraIntrinsics plus_camera = camera;
    OuterBootstrapCameraIntrinsics minus_camera = camera;
    plus_camera.SetCombinedParameterVector(ToStdVector(plus_vector));
    minus_camera.SetCombinedParameterVector(ToStdVector(minus_vector));
    if (!ClampIntrinsicsInPlace(&plus_camera) ||
        !ClampIntrinsicsInPlace(&minus_camera)) {
      continue;
    }
    const Eigen::VectorXd plus_residual =
        BuildSingleObservationResidual(plus_camera, observation, T_camera_board);
    const Eigen::VectorXd minus_residual =
        BuildSingleObservationResidual(minus_camera, observation, T_camera_board);
    if (plus_residual.rows() != jacobian.rows() ||
        minus_residual.rows() != jacobian.rows()) {
      continue;
    }
    jacobian.col(column) = (plus_residual - minus_residual) / (2.0 * step);
  }
  return jacobian.transpose() * jacobian;
}

std::vector<OuterObservationRecord> SelectKalibrOuterLmObservations(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<OuterObservationRecord>& observations,
    int* pose_success_count,
    int* pose_failure_count) {
  if (pose_success_count != nullptr) {
    *pose_success_count = 0;
  }
  if (pose_failure_count != nullptr) {
    *pose_failure_count = 0;
  }

  struct ObservationCandidate {
    const OuterObservationRecord* observation = nullptr;
    std::set<std::string> tokens;
    double quality_score = 0.0;
    Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
    double pose_fit_outer_rmse = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd camera_information;
    bool pose_success = false;
  };

  std::vector<ObservationCandidate> candidates;
  candidates.reserve(observations.size());
  int min_frame_index = std::numeric_limits<int>::max();
  int max_frame_index = std::numeric_limits<int>::min();
  for (const OuterObservationRecord& observation : observations) {
    if (observation.image_points.size() < 4) {
      continue;
    }
    min_frame_index = std::min(min_frame_index, observation.frame_index);
    max_frame_index = std::max(max_frame_index, observation.frame_index);
    ObservationCandidate candidate;
    candidate.observation = &observation;
    candidate.tokens.insert("board:" + std::to_string(observation.board_id));
    const int radial_bin = RadialCoverageBin(observation, camera.resolution);
    if (radial_bin >= 0) {
      candidate.tokens.insert("radial:" + std::to_string(radial_bin));
    }
    const int image_bin = ImageCoverageBin(observation, camera.resolution);
    if (image_bin >= 0) {
      candidate.tokens.insert("quadrant:" + std::to_string(image_bin / 4) +
                              "_" + std::to_string(image_bin % 4));
    }

    const std::vector<cv::Point2f>& points = observation.image_points;
    const double area = std::abs(
        0.5 * ((points[0].x * points[1].y - points[1].x * points[0].y) +
               (points[1].x * points[2].y - points[2].x * points[1].y) +
               (points[2].x * points[3].y - points[3].x * points[2].y) +
               (points[3].x * points[0].y - points[0].x * points[3].y)));
    const double normalized_area =
        area / std::max(1.0, static_cast<double>(camera.resolution.area()));
    int area_bin = 0;
    if (normalized_area > 0.01) {
      area_bin = 2;
    } else if (normalized_area > 0.0025) {
      area_bin = 1;
    }
    candidate.tokens.insert("area:" + std::to_string(area_bin));

    const double edge_dx = static_cast<double>(points[1].x - points[0].x);
    const double edge_dy = static_cast<double>(points[1].y - points[0].y);
    double angle = std::atan2(edge_dy, edge_dx);
    if (angle < 0.0) {
      angle += std::acos(-1.0);
    }
    const int orientation_bin = std::max(
        0, std::min(7, static_cast<int>(
                           std::floor(8.0 * angle / std::acos(-1.0)))));
    candidate.tokens.insert("orientation:" + std::to_string(orientation_bin));

    const double width0 = std::hypot(points[1].x - points[0].x,
                                     points[1].y - points[0].y);
    const double height0 = std::hypot(points[3].x - points[0].x,
                                      points[3].y - points[0].y);
    const double aspect = width0 / std::max(1e-6, height0);
    const int aspect_bin = aspect < 0.75 ? 0 : (aspect > 1.33 ? 2 : 1);
    candidate.tokens.insert("shape:" + std::to_string(aspect_bin));

	    candidate.quality_score =
	        std::log1p(std::max(0.0, area)) + 0.1 * observation.quality;
    candidate.pose_success =
        EstimatePoseFromObjectPoints(camera,
                                     observation.object_points,
                                     observation.image_points,
                                     &candidate.T_camera_board,
                                     &candidate.pose_fit_outer_rmse);
    if (candidate.pose_success) {
      candidate.camera_information =
          ComputeCameraInformationForObservation(camera,
                                                 observation,
                                                 candidate.T_camera_board);
    }
	    candidates.push_back(candidate);
	  }

  if (candidates.empty()) {
    return {};
  }
  if (pose_success_count != nullptr || pose_failure_count != nullptr) {
    int success_count = 0;
    int failure_count = 0;
    for (const ObservationCandidate& candidate : candidates) {
      if (candidate.pose_success) {
        ++success_count;
      } else {
        ++failure_count;
      }
    }
    if (pose_success_count != nullptr) {
      *pose_success_count = success_count;
    }
    if (pose_failure_count != nullptr) {
      *pose_failure_count = failure_count;
    }
  }

  std::set<std::string> available_tokens;
  for (const ObservationCandidate& candidate : candidates) {
    available_tokens.insert(candidate.tokens.begin(), candidate.tokens.end());
  }

  std::set<int> selected_indices;
  std::set<int> selected_frames;
  std::set<std::string> covered_tokens;
  const int parameter_count =
      static_cast<int>(camera.CombinedParameterVector().size());
  Eigen::MatrixXd selected_information =
      Eigen::MatrixXd::Zero(parameter_count, parameter_count);
  double best_observed_information_gain = 0.0;
  while (true) {
    const bool coverage_complete =
        covered_tokens.size() >= available_tokens.size();
    const double current_logdet = SafeLogDet(selected_information);
    const double min_useful_information_gain =
        coverage_complete
            ? std::max(1e-6, 1e-6 * best_observed_information_gain)
            : 0.0;
    int best_index = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    int best_new_token_count = 0;
    double best_information_gain = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      if (selected_indices.count(static_cast<int>(index)) > 0) {
        continue;
      }
      const ObservationCandidate& candidate = candidates[index];
      if (!candidate.pose_success || candidate.camera_information.rows() == 0) {
        continue;
      }
      int new_token_count = 0;
      for (const std::string& token : candidate.tokens) {
        if (covered_tokens.count(token) == 0) {
          ++new_token_count;
        }
      }
      double spacing_bonus = 0.0;
      if (!selected_frames.empty() && max_frame_index > min_frame_index) {
        int min_spacing = std::numeric_limits<int>::max();
        for (int frame_index : selected_frames) {
          min_spacing = std::min(
              min_spacing,
              std::abs(candidate.observation->frame_index - frame_index));
        }
        spacing_bonus =
            static_cast<double>(min_spacing) /
            static_cast<double>(std::max(1, max_frame_index - min_frame_index));
      }
      const double candidate_logdet =
          SafeLogDet(selected_information + candidate.camera_information);
      const double information_gain =
          std::isfinite(candidate_logdet) && std::isfinite(current_logdet)
              ? candidate_logdet - current_logdet
              : 0.0;
      best_observed_information_gain =
          std::max(best_observed_information_gain,
                   std::max(0.0, information_gain));
      if (coverage_complete &&
          information_gain < min_useful_information_gain) {
        continue;
      }
      const double score =
          static_cast<double>(new_token_count) * 10.0 +
          0.2 * std::max(0.0, information_gain) +
          spacing_bonus + 1e-6 * candidate.quality_score;
      if (score > best_score) {
        best_score = score;
        best_index = static_cast<int>(index);
        best_new_token_count = new_token_count;
        best_information_gain = information_gain;
      }
    }
    if (best_index < 0) {
      break;
    }
    if (coverage_complete &&
        best_new_token_count <= 0 &&
        best_information_gain < min_useful_information_gain) {
      break;
    }

    selected_indices.insert(best_index);
    const ObservationCandidate& selected =
        candidates[static_cast<std::size_t>(best_index)];
    selected_frames.insert(selected.observation->frame_index);
    covered_tokens.insert(selected.tokens.begin(), selected.tokens.end());
    selected_information += selected.camera_information;
    best_observed_information_gain =
        std::max(best_observed_information_gain,
                 std::max(0.0, best_information_gain));
  }

  std::vector<OuterObservationRecord> selected_observations;
  selected_observations.reserve(selected_indices.size());
  if (pose_success_count != nullptr) {
    *pose_success_count = static_cast<int>(selected_indices.size());
  }
  if (pose_failure_count != nullptr) {
    *pose_failure_count = 0;
  }
  for (int index : selected_indices) {
    selected_observations.push_back(
        *candidates[static_cast<std::size_t>(index)].observation);
  }
  std::sort(selected_observations.begin(), selected_observations.end(),
            [](const OuterObservationRecord& lhs,
               const OuterObservationRecord& rhs) {
              if (lhs.frame_index != rhs.frame_index) {
                return lhs.frame_index < rhs.frame_index;
              }
              return lhs.board_id < rhs.board_id;
            });
  return selected_observations;
}

std::vector<OuterBootstrapFrameInput> FilterFramesForSelectedObservations(
    const std::vector<OuterBootstrapFrameInput>& frames,
    const std::vector<OuterObservationRecord>& selected_observations) {
  std::map<int, std::set<int>> selected_boards_by_frame;
  for (const OuterObservationRecord& observation : selected_observations) {
    selected_boards_by_frame[observation.frame_index].insert(observation.board_id);
  }
  std::vector<OuterBootstrapFrameInput> filtered_frames;
  filtered_frames.reserve(selected_boards_by_frame.size());
  for (const OuterBootstrapFrameInput& frame : frames) {
    const auto frame_it = selected_boards_by_frame.find(frame.frame_index);
    if (frame_it == selected_boards_by_frame.end()) {
      continue;
    }
    OuterBootstrapFrameInput filtered = frame;
    filtered.measurements.board_measurements.clear();
    for (const OuterBoardMeasurement& measurement :
         frame.measurements.board_measurements) {
      if (frame_it->second.count(measurement.board_id) > 0) {
        filtered.measurements.board_measurements.push_back(measurement);
      }
    }
    if (!filtered.measurements.board_measurements.empty()) {
      filtered_frames.push_back(filtered);
    }
  }
  return filtered_frames;
}

BootstrapLayout BuildBootstrapLayoutFromCamera(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<OuterBootstrapFrameInput>& frames,
    const ApriltagInternalConfig& config) {
  BootstrapLayout layout;
  OuterBootstrapOptions bootstrap_options;
  bootstrap_options.reference_board_id =
      config.tag_ids.empty() ? config.tag_id : config.tag_ids.front();
  if (bootstrap_options.reference_board_id < 0) {
    bootstrap_options.reference_board_id = 1;
  }
  bootstrap_options.initial_camera = camera;
  bootstrap_options.max_coordinate_descent_iterations = 0;
  bootstrap_options.min_detection_quality =
      config.outer_detector_config.min_detection_quality;

  MultiBoardOuterBootstrap bootstrap(config, bootstrap_options);
  const OuterBootstrapResult bootstrap_result = bootstrap.Solve(frames);
  layout.success = bootstrap_result.success;
  layout.used_frame_count = bootstrap_result.used_frame_count;
  layout.used_board_observation_count =
      bootstrap_result.used_board_observation_count;
  layout.global_rmse = bootstrap_result.global_rmse;
  layout.warnings = bootstrap_result.warnings;
  for (const OuterBootstrapBoardState& board : bootstrap_result.boards) {
    if (!board.initialized) {
      continue;
    }
    layout.T_reference_board_by_id[board.board_id] =
        Eigen::Isometry3d(board.T_reference_board);
  }
  for (const OuterBootstrapFrameState& frame : bootstrap_result.frames) {
    if (!frame.initialized) {
      continue;
    }
    layout.T_camera_reference_by_frame[frame.frame_index] =
        Eigen::Isometry3d(frame.T_camera_reference);
  }
  return layout;
}

bool EstimateFramePoseFromLayout(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<const OuterObservationRecord*>& observations,
    const std::map<int, Eigen::Isometry3d>& T_reference_board_by_id,
    Eigen::Isometry3d* pose,
    double* rmse) {
  if (pose == nullptr || rmse == nullptr) {
    throw std::runtime_error("EstimateFramePoseFromLayout requires valid output pointers.");
  }
  std::vector<Eigen::Vector3d> object_points;
  std::vector<cv::Point2f> image_points;
  for (const OuterObservationRecord* observation : observations) {
    if (observation == nullptr) {
      continue;
    }
    const auto layout_it =
        T_reference_board_by_id.find(observation->board_id);
    if (layout_it == T_reference_board_by_id.end()) {
      continue;
    }
    for (std::size_t point_index = 0;
         point_index < observation->object_points.size(); ++point_index) {
      object_points.push_back(
          layout_it->second * observation->object_points[point_index]);
      image_points.push_back(observation->image_points[point_index]);
    }
  }
  if (object_points.size() < 4) {
    return false;
  }
  return EstimatePoseFromObjectPoints(camera, object_points, image_points, pose, rmse);
}

void EvaluateLeaveOneBoardOutPrediction(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<OuterObservationRecord>& observations,
    const BootstrapLayout& layout,
    AutoCameraInitializationCandidate* candidate) {
  if (candidate == nullptr || !layout.success ||
      layout.T_reference_board_by_id.empty()) {
    return;
  }
  DoubleSphereCameraModel camera_model;
  try {
    camera_model = DoubleSphereCameraModel::FromConfig(
        MakeIntermediateCameraConfig(camera));
  } catch (const std::exception&) {
    return;
  }

  std::map<int, std::vector<const OuterObservationRecord*>> observations_by_frame;
  for (const OuterObservationRecord& observation : observations) {
    if (layout.T_reference_board_by_id.count(observation.board_id) == 0) {
      continue;
    }
    observations_by_frame[observation.frame_index].push_back(&observation);
  }

  double squared_error_sum = 0.0;
  int point_count = 0;
  for (const auto& frame_entry : observations_by_frame) {
    const std::vector<const OuterObservationRecord*>& frame_observations =
        frame_entry.second;
    if (frame_observations.size() < 2u) {
      continue;
    }
    for (const OuterObservationRecord* held_out : frame_observations) {
      if (held_out == nullptr) {
        continue;
      }
      std::vector<const OuterObservationRecord*> support_observations;
      support_observations.reserve(frame_observations.size() - 1u);
      for (const OuterObservationRecord* support : frame_observations) {
        if (support != nullptr && support != held_out) {
          support_observations.push_back(support);
        }
      }
      ++candidate->leave_one_board_out_attempt_count;
      Eigen::Isometry3d T_camera_reference = Eigen::Isometry3d::Identity();
      double support_rmse = 0.0;
      if (!EstimateFramePoseFromLayout(camera,
                                       support_observations,
                                       layout.T_reference_board_by_id,
                                       &T_camera_reference,
                                       &support_rmse)) {
        continue;
      }
      const auto layout_it =
          layout.T_reference_board_by_id.find(held_out->board_id);
      if (layout_it == layout.T_reference_board_by_id.end()) {
        continue;
      }
      int valid_projected_points = 0;
      double held_out_squared_error_sum = 0.0;
      for (std::size_t point_index = 0;
           point_index < held_out->object_points.size(); ++point_index) {
        const Eigen::Vector3d point_camera =
            T_camera_reference * layout_it->second *
            held_out->object_points[point_index];
        Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
        if (!camera_model.vsEuclideanToKeypoint(point_camera, &predicted) ||
            !predicted.allFinite()) {
          continue;
        }
        const Eigen::Vector2d observed(held_out->image_points[point_index].x,
                                       held_out->image_points[point_index].y);
        const double squared_error = (predicted - observed).squaredNorm();
        if (!std::isfinite(squared_error)) {
          continue;
        }
        held_out_squared_error_sum += squared_error;
        ++valid_projected_points;
      }
      if (valid_projected_points == 0) {
        continue;
      }
      ++candidate->leave_one_board_out_success_count;
      squared_error_sum += held_out_squared_error_sum;
      point_count += valid_projected_points;
    }
  }
  if (point_count > 0) {
    candidate->leave_one_board_out_rmse =
        std::sqrt(squared_error_sum / static_cast<double>(point_count));
  }
}

AutoCameraInitializationCandidate EvaluateCandidateWithMultiBoardLayout(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<OuterBootstrapFrameInput>& frames,
    const ApriltagInternalConfig& config,
    int total_observation_count) {
  AutoCameraInitializationCandidate candidate;
  candidate.source_label = "auto_grid_multiboard_layout";
  candidate.evaluation_scope = "layout_full";
  candidate.camera = camera;
  candidate.observation_count = total_observation_count;
  const BootstrapLayout layout =
      BuildBootstrapLayoutFromCamera(camera, frames, config);
  if (!layout.success || !std::isfinite(layout.global_rmse) ||
      layout.used_board_observation_count <= 0) {
    candidate.valid = false;
    candidate.failure_reason = "multi_board_layout_bootstrap_failed";
    return candidate;
  }
  candidate.valid = true;
  candidate.pose_success_count = layout.used_board_observation_count;
  candidate.pose_failure_count =
      std::max(0, total_observation_count - layout.used_board_observation_count);
  candidate.successful_frame_count = layout.used_frame_count;
  candidate.successful_board_count =
      static_cast<int>(layout.T_reference_board_by_id.size());
  candidate.success_rate =
      total_observation_count > 0
          ? static_cast<double>(layout.used_board_observation_count) /
                static_cast<double>(total_observation_count)
          : 0.0;
  candidate.mean_observation_rmse = layout.global_rmse;
  const std::vector<OuterObservationRecord> observations =
      CollectOuterObservations(frames, config);
  EvaluateLeaveOneBoardOutPrediction(camera, observations, layout, &candidate);
  return candidate;
}

double LayoutSeedObjective(const AutoCameraInitializationCandidate& candidate) {
  if (!candidate.valid || candidate.observation_count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  const double fail_fraction =
      static_cast<double>(candidate.pose_failure_count) /
      static_cast<double>(std::max(1, candidate.observation_count));
  const bool has_loo =
      candidate.leave_one_board_out_success_count > 0 &&
      std::isfinite(candidate.leave_one_board_out_rmse);
  const double rmse =
      has_loo ? candidate.leave_one_board_out_rmse
              : candidate.mean_observation_rmse;
  return rmse * rmse + 2500.0 * fail_fraction;
}

double LayoutSeedStepForLabel(const std::string& label,
                              const OuterBootstrapCameraIntrinsics& camera) {
  if (label == "xi") {
    return 0.20;
  }
  if (label == "alpha" || label == "beta") {
    return 0.15;
  }
  if (label == "fu") {
    return 0.08 * static_cast<double>(std::max(1, camera.resolution.width));
  }
  if (label == "fv") {
    return 0.08 * static_cast<double>(std::max(1, camera.resolution.height));
  }
  return 0.0;
}

OuterBootstrapCameraIntrinsics RefineKalibrLikeSeedWithLayoutObjective(
    const OuterBootstrapCameraIntrinsics& initial_camera,
    const std::vector<OuterBootstrapFrameInput>& frames,
    const ApriltagInternalConfig& config,
    int total_observation_count,
    std::vector<std::string>* warnings) {
  OuterBootstrapCameraIntrinsics best = initial_camera;
  AutoCameraInitializationCandidate best_eval =
      EvaluateCandidateWithMultiBoardLayout(
          best, frames, config, total_observation_count);
  double best_objective = LayoutSeedObjective(best_eval);
  if (!std::isfinite(best_objective)) {
    AppendUniqueWarning(
        "Kalibr-like layout-aware seed refinement skipped because the initial "
        "layout objective was invalid.",
        warnings);
    return initial_camera;
  }

  const std::vector<std::string> labels = best.CombinedParameterLabels();
  std::vector<double> base_steps(labels.size(), 0.0);
  for (std::size_t index = 0; index < labels.size(); ++index) {
    base_steps[index] = LayoutSeedStepForLabel(labels[index], best);
  }

  int accepted_update_count = 0;
  for (int round = 0; round < 5; ++round) {
    const double round_scale = std::pow(0.5, round);
    bool improved_this_round = false;
    for (std::size_t parameter_index = 0; parameter_index < labels.size();
         ++parameter_index) {
      const double base_step = base_steps[parameter_index];
      if (base_step <= 0.0) {
        continue;
      }
      for (int direction = -1; direction <= 1; direction += 2) {
        OuterBootstrapCameraIntrinsics candidate = best;
        Eigen::VectorXd candidate_vector =
            ToEigenVector(candidate.CombinedParameterVector());
        candidate_vector[static_cast<Eigen::Index>(parameter_index)] +=
            static_cast<double>(direction) * base_step * round_scale;
        candidate.SetCombinedParameterVector(ToStdVector(candidate_vector));
        candidate.cu = best.cu;
        candidate.cv = best.cv;
        if (!ClampIntrinsicsInPlace(&candidate)) {
          continue;
        }
        const AutoCameraInitializationCandidate candidate_eval =
            EvaluateCandidateWithMultiBoardLayout(
                candidate, frames, config, total_observation_count);
        const double candidate_objective = LayoutSeedObjective(candidate_eval);
        if (std::isfinite(candidate_objective) &&
            candidate_objective + 1e-6 < best_objective) {
          best = candidate;
          best_eval = candidate_eval;
          best_objective = candidate_objective;
          improved_this_round = true;
          ++accepted_update_count;
        }
      }
    }
    if (!improved_this_round) {
      continue;
    }
  }

  if (accepted_update_count > 0) {
    std::ostringstream stream;
    stream << "Kalibr-like seed was refined by continuous layout/LOO "
           << "optimization with " << accepted_update_count
           << " accepted parameter updates; final layout RMSE="
           << best_eval.mean_observation_rmse
           << " LOO RMSE=" << best_eval.leave_one_board_out_rmse << ".";
    AppendUniqueWarning(stream.str(), warnings);
  } else {
    AppendUniqueWarning(
        "Kalibr-like continuous layout/LOO seed refinement made no accepted "
        "parameter update.",
        warnings);
  }
  return best;
}

std::vector<KalibrOuterLmFrameView> SelectKalibrOuterLmFrameViews(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<OuterObservationRecord>& observations,
    const BootstrapLayout& layout) {
  if (!layout.success || layout.T_reference_board_by_id.empty()) {
    return {};
  }

  struct FrameCandidate {
    int frame_index = -1;
    std::string frame_label;
    std::vector<const OuterObservationRecord*> observations;
    std::set<std::string> tokens;
    double quality_score = 0.0;
    Eigen::Isometry3d T_camera_reference = Eigen::Isometry3d::Identity();
    Eigen::MatrixXd camera_information;
    bool pose_success = false;
  };

  std::map<int, FrameCandidate> frame_candidates;
  int min_frame_index = std::numeric_limits<int>::max();
  int max_frame_index = std::numeric_limits<int>::min();
  for (const OuterObservationRecord& observation : observations) {
    if (layout.T_reference_board_by_id.count(observation.board_id) == 0) {
      continue;
    }
    FrameCandidate& candidate = frame_candidates[observation.frame_index];
    candidate.frame_index = observation.frame_index;
    candidate.frame_label = observation.frame_label;
    candidate.observations.push_back(&observation);
    candidate.tokens.insert("board:" + std::to_string(observation.board_id));
    const int radial_bin = RadialCoverageBin(observation, camera.resolution);
    if (radial_bin >= 0) {
      candidate.tokens.insert("radial:" + std::to_string(radial_bin));
    }
    const int image_bin = ImageCoverageBin(observation, camera.resolution);
    if (image_bin >= 0) {
      candidate.tokens.insert("quadrant:" + std::to_string(image_bin / 4) +
                              "_" + std::to_string(image_bin % 4));
    }
    const std::vector<cv::Point2f>& points = observation.image_points;
    const double area = std::abs(
        0.5 * ((points[0].x * points[1].y - points[1].x * points[0].y) +
               (points[1].x * points[2].y - points[2].x * points[1].y) +
               (points[2].x * points[3].y - points[3].x * points[2].y) +
               (points[3].x * points[0].y - points[0].x * points[3].y)));
    const double normalized_area =
        area / std::max(1.0, static_cast<double>(camera.resolution.area()));
    const int area_bin =
        normalized_area > 0.01 ? 2 : (normalized_area > 0.0025 ? 1 : 0);
    candidate.tokens.insert("area:" + std::to_string(area_bin));
    const double edge_dx = static_cast<double>(points[1].x - points[0].x);
    const double edge_dy = static_cast<double>(points[1].y - points[0].y);
    double angle = std::atan2(edge_dy, edge_dx);
    if (angle < 0.0) {
      angle += std::acos(-1.0);
    }
    const int orientation_bin = std::max(
        0, std::min(7, static_cast<int>(
                           std::floor(8.0 * angle / std::acos(-1.0)))));
    candidate.tokens.insert("orientation:" + std::to_string(orientation_bin));
    candidate.quality_score +=
        std::log1p(std::max(0.0, area)) + 0.1 * observation.quality;
    min_frame_index = std::min(min_frame_index, observation.frame_index);
    max_frame_index = std::max(max_frame_index, observation.frame_index);
  }

  std::vector<FrameCandidate> candidates;
  candidates.reserve(frame_candidates.size());
  const int camera_parameter_count =
      static_cast<int>(camera.CombinedParameterVector().size());
  for (auto& entry : frame_candidates) {
    FrameCandidate& candidate = entry.second;
    const auto pose_it =
        layout.T_camera_reference_by_frame.find(candidate.frame_index);
    if (pose_it != layout.T_camera_reference_by_frame.end()) {
      candidate.T_camera_reference = pose_it->second;
      candidate.pose_success = true;
    } else {
      double rmse = 0.0;
      candidate.pose_success =
          EstimateFramePoseFromLayout(camera,
                                      candidate.observations,
                                      layout.T_reference_board_by_id,
                                      &candidate.T_camera_reference,
                                      &rmse);
    }
    if (!candidate.pose_success || candidate.observations.empty()) {
      continue;
    }
    candidate.camera_information =
        Eigen::MatrixXd::Zero(camera_parameter_count, camera_parameter_count);
    for (const OuterObservationRecord* observation : candidate.observations) {
      if (observation == nullptr) {
        continue;
      }
      const auto board_pose_it =
          layout.T_reference_board_by_id.find(observation->board_id);
      if (board_pose_it == layout.T_reference_board_by_id.end()) {
        continue;
      }
      const Eigen::Isometry3d T_camera_board =
          candidate.T_camera_reference * board_pose_it->second;
      candidate.camera_information +=
          ComputeCameraInformationForObservation(
              camera, *observation, T_camera_board);
    }
    candidate.tokens.insert("board_count:" +
                            std::to_string(std::min<int>(
                                5, candidate.observations.size())));
    candidates.push_back(candidate);
  }
  if (candidates.empty()) {
    return {};
  }

  std::set<std::string> available_tokens;
  for (const FrameCandidate& candidate : candidates) {
    available_tokens.insert(candidate.tokens.begin(), candidate.tokens.end());
  }

  std::set<int> selected_indices;
  std::set<int> selected_frames;
  std::set<std::string> covered_tokens;
  Eigen::MatrixXd selected_information =
      Eigen::MatrixXd::Zero(camera_parameter_count, camera_parameter_count);
  double best_observed_information_gain = 0.0;
  while (true) {
    const bool coverage_complete =
        covered_tokens.size() >= available_tokens.size();
    const double current_logdet = SafeLogDet(selected_information);
    const double min_useful_information_gain =
        coverage_complete
            ? std::max(1e-6, 1e-6 * best_observed_information_gain)
            : 0.0;
    int best_index = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    int best_new_token_count = 0;
    double best_information_gain = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      if (selected_indices.count(static_cast<int>(index)) > 0) {
        continue;
      }
      const FrameCandidate& candidate = candidates[index];
      int new_token_count = 0;
      for (const std::string& token : candidate.tokens) {
        if (covered_tokens.count(token) == 0) {
          ++new_token_count;
        }
      }
      double spacing_bonus = 0.0;
      if (!selected_frames.empty() && max_frame_index > min_frame_index) {
        int min_spacing = std::numeric_limits<int>::max();
        for (int frame_index : selected_frames) {
          min_spacing =
              std::min(min_spacing,
                       std::abs(candidate.frame_index - frame_index));
        }
        spacing_bonus =
            static_cast<double>(min_spacing) /
            static_cast<double>(std::max(1, max_frame_index - min_frame_index));
      }
      const double multiboard_bonus =
          0.25 * static_cast<double>(std::max<int>(
                     0, static_cast<int>(candidate.observations.size()) - 1));
      const double candidate_logdet =
          SafeLogDet(selected_information + candidate.camera_information);
      const double information_gain =
          std::isfinite(candidate_logdet) && std::isfinite(current_logdet)
              ? candidate_logdet - current_logdet
              : 0.0;
      best_observed_information_gain =
          std::max(best_observed_information_gain,
                   std::max(0.0, information_gain));
      if (coverage_complete &&
          information_gain < min_useful_information_gain) {
        continue;
      }
      const double score =
          static_cast<double>(new_token_count) * 10.0 +
          0.2 * std::max(0.0, information_gain) +
          spacing_bonus + multiboard_bonus + 1e-6 * candidate.quality_score;
      if (score > best_score) {
        best_score = score;
        best_index = static_cast<int>(index);
        best_new_token_count = new_token_count;
        best_information_gain = information_gain;
      }
    }
    if (best_index < 0) {
      break;
    }
    if (coverage_complete &&
        best_new_token_count <= 0 &&
        best_information_gain < min_useful_information_gain) {
      break;
    }
    selected_indices.insert(best_index);
    const FrameCandidate& selected =
        candidates[static_cast<std::size_t>(best_index)];
    selected_frames.insert(selected.frame_index);
    covered_tokens.insert(selected.tokens.begin(), selected.tokens.end());
    selected_information += selected.camera_information;
    best_observed_information_gain =
        std::max(best_observed_information_gain,
                 std::max(0.0, best_information_gain));
  }

  std::vector<KalibrOuterLmFrameView> selected_views;
  selected_views.reserve(selected_indices.size());
  for (int index : selected_indices) {
    const FrameCandidate& candidate =
        candidates[static_cast<std::size_t>(index)];
    KalibrOuterLmFrameView view;
    view.frame_index = candidate.frame_index;
    view.frame_label = candidate.frame_label;
    view.observations = candidate.observations;
    view.T_camera_reference = candidate.T_camera_reference;
    selected_views.push_back(view);
  }
  std::sort(selected_views.begin(), selected_views.end(),
            [](const KalibrOuterLmFrameView& lhs,
               const KalibrOuterLmFrameView& rhs) {
              return lhs.frame_index < rhs.frame_index;
            });
  return selected_views;
}

std::vector<KalibrOuterLmFrameView> DownsampleFrameViewsUniformly(
    const std::vector<KalibrOuterLmFrameView>& views,
    int max_view_count) {
  if (max_view_count <= 0 ||
      static_cast<int>(views.size()) <= max_view_count) {
    return views;
  }
  std::vector<KalibrOuterLmFrameView> sampled;
  sampled.reserve(static_cast<std::size_t>(max_view_count));
  std::set<std::size_t> used_indices;
  const double last = static_cast<double>(views.size() - 1u);
  for (int index = 0; index < max_view_count; ++index) {
    const double alpha =
        max_view_count <= 1
            ? 0.0
            : static_cast<double>(index) /
                  static_cast<double>(max_view_count - 1);
    std::size_t source_index = static_cast<std::size_t>(
        std::round(alpha * last));
    while (source_index < views.size() &&
           used_indices.count(source_index) > 0) {
      ++source_index;
    }
    if (source_index >= views.size()) {
      source_index = views.size() - 1u;
      while (source_index > 0u && used_indices.count(source_index) > 0) {
        --source_index;
      }
    }
    if (used_indices.insert(source_index).second) {
      sampled.push_back(views[source_index]);
    }
  }
  return sampled;
}

double NumericJacobianStep(int parameter_index,
                           int camera_parameter_count,
                           const std::vector<std::string>& camera_labels,
                           const Eigen::VectorXd& x) {
  if (parameter_index < camera_parameter_count) {
    const std::string& label =
        camera_labels[static_cast<std::size_t>(parameter_index)];
    if (label == "fu" || label == "fv" || label == "cu" || label == "cv") {
      return std::max(1e-3, std::abs(x[parameter_index]) * 1e-6);
    }
    return std::max(1e-6, std::abs(x[parameter_index]) * 1e-6);
  }
  const int pose_offset = parameter_index - camera_parameter_count;
  return (pose_offset % 6) < 3 ? 1e-6 : 1e-7;
}

KalibrOuterLmEvaluation EvaluateKalibrOuterLmState(
    const Eigen::VectorXd& x,
    const OuterBootstrapCameraIntrinsics& camera_prototype,
    const std::vector<KalibrOuterLmView>& views) {
  KalibrOuterLmEvaluation evaluation;
  const int camera_parameter_count =
      static_cast<int>(camera_prototype.CombinedParameterVector().size());
  if (x.rows() < camera_parameter_count) {
    ++evaluation.nonfinite_count;
    return evaluation;
  }
  for (Eigen::Index index = 0; index < x.rows(); ++index) {
    if (!std::isfinite(x[index])) {
      ++evaluation.nonfinite_count;
      return evaluation;
    }
  }

  evaluation.camera = camera_prototype;
  const Eigen::VectorXd camera_vector = x.head(camera_parameter_count);
  evaluation.camera.SetCombinedParameterVector(ToStdVector(camera_vector));
  if (!ClampIntrinsicsInPlace(&evaluation.camera)) {
    ++evaluation.nonfinite_count;
    return evaluation;
  }

  DoubleSphereCameraModel camera;
  try {
    camera = DoubleSphereCameraModel::FromConfig(
        MakeIntermediateCameraConfig(evaluation.camera));
  } catch (const std::exception&) {
    ++evaluation.nonfinite_count;
    return evaluation;
  }

  int point_count = 0;
  for (const KalibrOuterLmView& view : views) {
    if (view.observation != nullptr) {
      point_count += static_cast<int>(view.observation->object_points.size());
    }
  }
  evaluation.point_count = point_count;
  evaluation.residuals = Eigen::VectorXd::Zero(std::max(0, 2 * point_count));
  if (point_count <= 0) {
    return evaluation;
  }

  constexpr double kInvalidProjectionPenalty = 100.0;
  int row = 0;
  double squared_error_sum = 0.0;
  for (std::size_t view_index = 0; view_index < views.size(); ++view_index) {
    const KalibrOuterLmView& view = views[view_index];
    const OuterObservationRecord& observation = *view.observation;
    Eigen::Matrix<double, 6, 1> pose_vector;
    pose_vector = x.segment<6>(
        camera_parameter_count + static_cast<int>(view_index) * 6);
    const Eigen::Isometry3d T_camera_board = VectorToPose(pose_vector);
    for (std::size_t point_index = 0;
         point_index < observation.object_points.size(); ++point_index) {
      Eigen::Vector2d projected = Eigen::Vector2d::Zero();
      const bool valid_projection = camera.vsEuclideanToKeypoint(
          T_camera_board * observation.object_points[point_index], &projected);
      if (!valid_projection || !projected.allFinite()) {
        evaluation.residuals[row++] = kInvalidProjectionPenalty;
        evaluation.residuals[row++] = kInvalidProjectionPenalty;
        squared_error_sum += 2.0 * kInvalidProjectionPenalty *
                             kInvalidProjectionPenalty;
        ++evaluation.invalid_projection_count;
        continue;
      }
      const double dx =
          projected.x() - static_cast<double>(observation.image_points[point_index].x);
      const double dy =
          projected.y() - static_cast<double>(observation.image_points[point_index].y);
      if (!std::isfinite(dx) || !std::isfinite(dy)) {
        evaluation.residuals[row++] = kInvalidProjectionPenalty;
        evaluation.residuals[row++] = kInvalidProjectionPenalty;
        squared_error_sum += 2.0 * kInvalidProjectionPenalty *
                             kInvalidProjectionPenalty;
        ++evaluation.nonfinite_count;
        continue;
      }
      evaluation.residuals[row++] = dx;
      evaluation.residuals[row++] = dy;
      squared_error_sum += dx * dx + dy * dy;
    }
  }

  evaluation.rmse = std::sqrt(squared_error_sum /
                              static_cast<double>(std::max(1, point_count)));
  evaluation.success = evaluation.nonfinite_count == 0;
  return evaluation;
}

KalibrOuterLmEvaluation EvaluateKalibrOuterLmFrameState(
    const Eigen::VectorXd& x,
    const OuterBootstrapCameraIntrinsics& camera_prototype,
    const std::vector<KalibrOuterLmFrameView>& frame_views,
    const std::map<int, Eigen::Isometry3d>& T_reference_board_by_id) {
  KalibrOuterLmEvaluation evaluation;
  const int camera_parameter_count =
      static_cast<int>(camera_prototype.CombinedParameterVector().size());
  if (x.rows() < camera_parameter_count) {
    ++evaluation.nonfinite_count;
    return evaluation;
  }
  for (Eigen::Index index = 0; index < x.rows(); ++index) {
    if (!std::isfinite(x[index])) {
      ++evaluation.nonfinite_count;
      return evaluation;
    }
  }

  evaluation.camera = camera_prototype;
  evaluation.camera.SetCombinedParameterVector(
      ToStdVector(x.head(camera_parameter_count)));
  if (!ClampIntrinsicsInPlace(&evaluation.camera)) {
    ++evaluation.nonfinite_count;
    return evaluation;
  }

  DoubleSphereCameraModel camera;
  try {
    camera = DoubleSphereCameraModel::FromConfig(
        MakeIntermediateCameraConfig(evaluation.camera));
  } catch (const std::exception&) {
    ++evaluation.nonfinite_count;
    return evaluation;
  }

  int point_count = 0;
  for (const KalibrOuterLmFrameView& frame_view : frame_views) {
    for (const OuterObservationRecord* observation : frame_view.observations) {
      if (observation != nullptr &&
          T_reference_board_by_id.count(observation->board_id) > 0) {
        point_count += static_cast<int>(observation->object_points.size());
      }
    }
  }
  evaluation.point_count = point_count;
  evaluation.residuals = Eigen::VectorXd::Zero(std::max(0, 2 * point_count));
  if (point_count <= 0) {
    return evaluation;
  }

  constexpr double kInvalidProjectionPenalty = 100.0;
  int row = 0;
  double squared_error_sum = 0.0;
  for (std::size_t frame_index = 0; frame_index < frame_views.size();
       ++frame_index) {
    const Eigen::Matrix<double, 6, 1> pose_vector = x.segment<6>(
        camera_parameter_count + static_cast<int>(frame_index) * 6);
    const Eigen::Isometry3d T_camera_reference = VectorToPose(pose_vector);
    for (const OuterObservationRecord* observation :
         frame_views[frame_index].observations) {
      if (observation == nullptr) {
        continue;
      }
      const auto board_pose_it =
          T_reference_board_by_id.find(observation->board_id);
      if (board_pose_it == T_reference_board_by_id.end()) {
        continue;
      }
      for (std::size_t point_index = 0;
           point_index < observation->object_points.size(); ++point_index) {
        const Eigen::Vector3d point_reference =
            board_pose_it->second * observation->object_points[point_index];
        Eigen::Vector2d projected = Eigen::Vector2d::Zero();
        const bool valid_projection = camera.vsEuclideanToKeypoint(
            T_camera_reference * point_reference, &projected);
        if (!valid_projection || !projected.allFinite()) {
          evaluation.residuals[row++] = kInvalidProjectionPenalty;
          evaluation.residuals[row++] = kInvalidProjectionPenalty;
          squared_error_sum += 2.0 * kInvalidProjectionPenalty *
                               kInvalidProjectionPenalty;
          ++evaluation.invalid_projection_count;
          continue;
        }
        const double dx =
            projected.x() -
            static_cast<double>(observation->image_points[point_index].x);
        const double dy =
            projected.y() -
            static_cast<double>(observation->image_points[point_index].y);
        if (!std::isfinite(dx) || !std::isfinite(dy)) {
          evaluation.residuals[row++] = kInvalidProjectionPenalty;
          evaluation.residuals[row++] = kInvalidProjectionPenalty;
          squared_error_sum += 2.0 * kInvalidProjectionPenalty *
                               kInvalidProjectionPenalty;
          ++evaluation.nonfinite_count;
          continue;
        }
        evaluation.residuals[row++] = dx;
        evaluation.residuals[row++] = dy;
        squared_error_sum += dx * dx + dy * dy;
      }
    }
  }

  evaluation.rmse = std::sqrt(squared_error_sum /
                              static_cast<double>(std::max(1, point_count)));
  evaluation.success = evaluation.nonfinite_count == 0;
  return evaluation;
}

struct KalibrOuterLmSchurStep {
  bool success = false;
  Eigen::VectorXd step;
};

KalibrOuterLmSchurStep ComputeKalibrOuterLmSchurStep(
    const Eigen::VectorXd& x,
    const OuterBootstrapCameraIntrinsics& camera_prototype,
    const std::vector<KalibrOuterLmView>& views,
    double lambda) {
  KalibrOuterLmSchurStep result;
  const int camera_parameter_count =
      static_cast<int>(camera_prototype.CombinedParameterVector().size());
  const int parameter_count =
      camera_parameter_count + static_cast<int>(views.size()) * 6;
  result.step = Eigen::VectorXd::Zero(parameter_count);
  if (x.rows() != parameter_count || camera_parameter_count <= 0) {
    return result;
  }

  OuterBootstrapCameraIntrinsics camera = camera_prototype;
  camera.SetCombinedParameterVector(
      ToStdVector(x.head(camera_parameter_count)));
  if (!ClampIntrinsicsInPlace(&camera)) {
    return result;
  }

  const std::vector<std::string> camera_labels =
      camera_prototype.CombinedParameterLabels();
  Eigen::MatrixXd Hcc =
      Eigen::MatrixXd::Zero(camera_parameter_count, camera_parameter_count);
  Eigen::VectorXd gc = Eigen::VectorXd::Zero(camera_parameter_count);

  struct PoseBlock {
    int pose_offset = 0;
    Eigen::Matrix<double, 6, 6> Hpp;
    Eigen::MatrixXd Hpc;
    Eigen::Matrix<double, 6, 1> gp;
  };
  std::vector<PoseBlock> pose_blocks;
  pose_blocks.reserve(views.size());

  for (std::size_t view_index = 0; view_index < views.size(); ++view_index) {
    const KalibrOuterLmView& view = views[view_index];
    if (view.observation == nullptr) {
      continue;
    }
    const int pose_offset =
        camera_parameter_count + static_cast<int>(view_index) * 6;
    const Eigen::Matrix<double, 6, 1> pose_vector =
        x.segment<6>(pose_offset);
    const Eigen::Isometry3d T_camera_board = VectorToPose(pose_vector);
    const Eigen::VectorXd residual =
        BuildSingleObservationResidual(
            camera, *view.observation, T_camera_board);
    if (residual.rows() <= 0 || !residual.allFinite()) {
      continue;
    }

    Eigen::MatrixXd Jc(residual.rows(), camera_parameter_count);
    Jc.setZero();
    for (int column = 0; column < camera_parameter_count; ++column) {
      const double step =
          NumericJacobianStep(column, camera_parameter_count, camera_labels, x);
      Eigen::VectorXd plus_vector = x.head(camera_parameter_count);
      Eigen::VectorXd minus_vector = x.head(camera_parameter_count);
      plus_vector[column] += step;
      minus_vector[column] -= step;
      OuterBootstrapCameraIntrinsics plus_camera = camera_prototype;
      OuterBootstrapCameraIntrinsics minus_camera = camera_prototype;
      plus_camera.SetCombinedParameterVector(ToStdVector(plus_vector));
      minus_camera.SetCombinedParameterVector(ToStdVector(minus_vector));
      if (!ClampIntrinsicsInPlace(&plus_camera) ||
          !ClampIntrinsicsInPlace(&minus_camera)) {
        continue;
      }
      const Eigen::VectorXd plus_residual =
          BuildSingleObservationResidual(
              plus_camera, *view.observation, T_camera_board);
      const Eigen::VectorXd minus_residual =
          BuildSingleObservationResidual(
              minus_camera, *view.observation, T_camera_board);
      if (plus_residual.rows() != residual.rows() ||
          minus_residual.rows() != residual.rows() ||
          !plus_residual.allFinite() || !minus_residual.allFinite()) {
        continue;
      }
      Jc.col(column) = (plus_residual - minus_residual) / (2.0 * step);
    }

    Eigen::Matrix<double, Eigen::Dynamic, 6> Jp(residual.rows(), 6);
    Jp.setZero();
    for (int column = 0; column < 6; ++column) {
      const double step =
          NumericJacobianStep(pose_offset + column,
                              camera_parameter_count,
                              camera_labels,
                              x);
      Eigen::Matrix<double, 6, 1> plus_pose_vector = pose_vector;
      Eigen::Matrix<double, 6, 1> minus_pose_vector = pose_vector;
      plus_pose_vector[column] += step;
      minus_pose_vector[column] -= step;
      const Eigen::VectorXd plus_residual =
          BuildSingleObservationResidual(
              camera,
              *view.observation,
              VectorToPose(plus_pose_vector));
      const Eigen::VectorXd minus_residual =
          BuildSingleObservationResidual(
              camera,
              *view.observation,
              VectorToPose(minus_pose_vector));
      if (plus_residual.rows() != residual.rows() ||
          minus_residual.rows() != residual.rows() ||
          !plus_residual.allFinite() || !minus_residual.allFinite()) {
        continue;
      }
      Jp.col(column) = (plus_residual - minus_residual) / (2.0 * step);
    }

    Hcc.noalias() += Jc.transpose() * Jc;
    gc.noalias() += Jc.transpose() * residual;

    PoseBlock block;
    block.pose_offset = pose_offset;
    block.Hpp = Jp.transpose() * Jp;
    block.Hpc = Jp.transpose() * Jc;
    block.gp = Jp.transpose() * residual;
    for (int diagonal = 0; diagonal < 6; ++diagonal) {
      block.Hpp(diagonal, diagonal) +=
          lambda * std::max(1.0, std::abs(block.Hpp(diagonal, diagonal))) +
          1e-9;
    }
    pose_blocks.push_back(block);
  }

  if (pose_blocks.empty() || !Hcc.allFinite() || !gc.allFinite()) {
    return result;
  }

  Eigen::MatrixXd schur_H = Hcc;
  for (int diagonal = 0; diagonal < camera_parameter_count; ++diagonal) {
    schur_H(diagonal, diagonal) +=
        lambda * std::max(1.0, std::abs(Hcc(diagonal, diagonal))) + 1e-9;
  }
  Eigen::VectorXd schur_g = gc;
  for (const PoseBlock& block : pose_blocks) {
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(block.Hpp);
    if (ldlt.info() != Eigen::Success) {
      return result;
    }
    const Eigen::MatrixXd Hpp_inv_Hpc = ldlt.solve(block.Hpc);
    const Eigen::Matrix<double, 6, 1> Hpp_inv_gp = ldlt.solve(block.gp);
    if (!Hpp_inv_Hpc.allFinite() || !Hpp_inv_gp.allFinite()) {
      return result;
    }
    schur_H.noalias() -= block.Hpc.transpose() * Hpp_inv_Hpc;
    schur_g.noalias() -= block.Hpc.transpose() * Hpp_inv_gp;
  }

  if (!schur_H.allFinite() || !schur_g.allFinite()) {
    return result;
  }
  Eigen::LDLT<Eigen::MatrixXd> camera_ldlt(schur_H);
  if (camera_ldlt.info() != Eigen::Success) {
    return result;
  }
  const Eigen::VectorXd camera_step = camera_ldlt.solve(-schur_g);
  if (camera_step.rows() != camera_parameter_count ||
      !camera_step.allFinite()) {
    return result;
  }

  result.step.head(camera_parameter_count) = camera_step;
  for (const PoseBlock& block : pose_blocks) {
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(block.Hpp);
    if (ldlt.info() != Eigen::Success) {
      return KalibrOuterLmSchurStep();
    }
    const Eigen::Matrix<double, 6, 1> pose_rhs =
        block.gp + block.Hpc * camera_step;
    const Eigen::Matrix<double, 6, 1> pose_step =
        ldlt.solve(-pose_rhs);
    if (!pose_step.allFinite()) {
      return KalibrOuterLmSchurStep();
    }
    result.step.segment<6>(block.pose_offset) = pose_step;
  }
  result.success = true;
  return result;
}

KalibrOuterLmRefinementResult RefineCandidateCameraKalibrOuterLm(
    const OuterBootstrapCameraIntrinsics& initial_camera,
    const std::vector<OuterObservationRecord>& observations) {
  KalibrOuterLmRefinementResult result;
  result.camera = initial_camera;

  std::vector<KalibrOuterLmView> views;
  views.reserve(observations.size());
  for (const OuterObservationRecord& observation : observations) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double rmse = 0.0;
    if (!EstimatePoseFromObjectPoints(initial_camera,
                                      observation.object_points,
                                      observation.image_points,
                                      &pose,
                                      &rmse)) {
      continue;
    }
    KalibrOuterLmView view;
    view.observation = &observation;
    view.T_camera_board = pose;
    views.push_back(view);
  }
  result.view_count = static_cast<int>(views.size());
  if (views.empty()) {
    return result;
  }

  const std::vector<double> camera_parameters =
      initial_camera.CombinedParameterVector();
  const int camera_parameter_count = static_cast<int>(camera_parameters.size());
  const int parameter_count =
      camera_parameter_count + static_cast<int>(views.size()) * 6;
  Eigen::VectorXd x(parameter_count);
  x.head(camera_parameter_count) = ToEigenVector(camera_parameters);
  for (std::size_t view_index = 0; view_index < views.size(); ++view_index) {
    x.segment<6>(camera_parameter_count + static_cast<int>(view_index) * 6) =
        PoseToVector(views[view_index].T_camera_board);
  }

  KalibrOuterLmEvaluation current =
      EvaluateKalibrOuterLmState(x, initial_camera, views);
  result.initial_rmse = current.rmse;
  result.residual_count = static_cast<int>(current.residuals.rows());
  if (!current.success || current.residuals.rows() <= 0 ||
      !std::isfinite(current.rmse)) {
    result.invalid_projection_count = current.invalid_projection_count;
    result.nonfinite_count = current.nonfinite_count;
    return result;
  }

  const std::vector<std::string> camera_labels =
      initial_camera.CombinedParameterLabels();
  double lambda = 1e-3;
  constexpr int kMaxIterations = 25;
  for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
    const KalibrOuterLmSchurStep schur_step =
        ComputeKalibrOuterLmSchurStep(x, initial_camera, views, lambda);
    if (!schur_step.success ||
        schur_step.step.rows() != parameter_count ||
        !schur_step.step.allFinite()) {
      lambda *= 10.0;
      continue;
    }
    const Eigen::VectorXd& step = schur_step.step;
    if (step.norm() < 1e-8) {
      result.iteration_count = iteration + 1;
      break;
    }

    Eigen::VectorXd candidate_x = x + step;
    KalibrOuterLmEvaluation candidate =
        EvaluateKalibrOuterLmState(candidate_x, initial_camera, views);
    if (candidate.success && std::isfinite(candidate.rmse) &&
        candidate.rmse + 1e-9 < current.rmse) {
      x = candidate_x;
      x.head(camera_parameter_count) =
          ToEigenVector(candidate.camera.CombinedParameterVector());
      current = EvaluateKalibrOuterLmState(x, initial_camera, views);
      lambda = std::max(1e-9, lambda * 0.3);
      result.iteration_count = iteration + 1;
      if (std::abs(result.initial_rmse - current.rmse) < 1e-9) {
        break;
      }
    } else {
      lambda *= 10.0;
      if (lambda > 1e9) {
        result.iteration_count = iteration + 1;
        break;
      }
    }
  }

  result.camera = current.camera;
  result.final_rmse = current.rmse;
  result.invalid_projection_count = current.invalid_projection_count;
  result.nonfinite_count = current.nonfinite_count;
  result.improved = std::isfinite(result.initial_rmse) &&
                    std::isfinite(result.final_rmse) &&
                    result.final_rmse + 1e-9 < result.initial_rmse;
  return result;
}

KalibrOuterLmRefinementResult RefineCandidateCameraKalibrFrameCohesionLm(
    const OuterBootstrapCameraIntrinsics& initial_camera,
    const std::vector<KalibrOuterLmFrameView>& frame_views,
    const std::map<int, Eigen::Isometry3d>& T_reference_board_by_id) {
  KalibrOuterLmRefinementResult result;
  result.camera = initial_camera;
  result.view_count = 0;
  for (const KalibrOuterLmFrameView& frame_view : frame_views) {
    result.view_count += static_cast<int>(frame_view.observations.size());
  }
  if (frame_views.empty()) {
    return result;
  }

  const std::vector<double> camera_parameters =
      initial_camera.CombinedParameterVector();
  const int camera_parameter_count = static_cast<int>(camera_parameters.size());
  const int parameter_count =
      camera_parameter_count + static_cast<int>(frame_views.size()) * 6;
  Eigen::VectorXd x(parameter_count);
  x.head(camera_parameter_count) = ToEigenVector(camera_parameters);
  for (std::size_t frame_index = 0; frame_index < frame_views.size();
       ++frame_index) {
    x.segment<6>(camera_parameter_count + static_cast<int>(frame_index) * 6) =
        PoseToVector(frame_views[frame_index].T_camera_reference);
  }

  KalibrOuterLmEvaluation current =
      EvaluateKalibrOuterLmFrameState(
          x, initial_camera, frame_views, T_reference_board_by_id);
  result.initial_rmse = current.rmse;
  result.residual_count = static_cast<int>(current.residuals.rows());
  if (!current.success || current.residuals.rows() <= 0 ||
      !std::isfinite(current.rmse)) {
    result.invalid_projection_count = current.invalid_projection_count;
    result.nonfinite_count = current.nonfinite_count;
    return result;
  }

  const std::vector<std::string> camera_labels =
      initial_camera.CombinedParameterLabels();
  double lambda = 1e-3;
  constexpr int kMaxIterations = 25;
  for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
    Eigen::MatrixXd jacobian(current.residuals.rows(), parameter_count);
    for (int column = 0; column < parameter_count; ++column) {
      const double step =
          NumericJacobianStep(column, camera_parameter_count, camera_labels, x);
      Eigen::VectorXd x_plus = x;
      Eigen::VectorXd x_minus = x;
      x_plus[column] += step;
      x_minus[column] -= step;
      const KalibrOuterLmEvaluation plus =
          EvaluateKalibrOuterLmFrameState(
              x_plus, initial_camera, frame_views, T_reference_board_by_id);
      const KalibrOuterLmEvaluation minus =
          EvaluateKalibrOuterLmFrameState(
              x_minus, initial_camera, frame_views, T_reference_board_by_id);
      if (!plus.success || !minus.success ||
          plus.residuals.rows() != current.residuals.rows() ||
          minus.residuals.rows() != current.residuals.rows()) {
        jacobian.col(column).setZero();
        continue;
      }
      jacobian.col(column) = (plus.residuals - minus.residuals) / (2.0 * step);
    }

    const Eigen::MatrixXd hessian = jacobian.transpose() * jacobian;
    const Eigen::VectorXd gradient = jacobian.transpose() * current.residuals;
    Eigen::MatrixXd damped = hessian;
    for (int diagonal = 0; diagonal < damped.rows(); ++diagonal) {
      damped(diagonal, diagonal) +=
          lambda * std::max(1.0, std::abs(hessian(diagonal, diagonal)));
    }
    const Eigen::VectorXd step = damped.ldlt().solve(-gradient);
    if (step.rows() != parameter_count || !step.allFinite()) {
      lambda *= 10.0;
      continue;
    }
    if (step.norm() < 1e-8) {
      result.iteration_count = iteration + 1;
      break;
    }

    const Eigen::VectorXd candidate_x = x + step;
    const KalibrOuterLmEvaluation candidate =
        EvaluateKalibrOuterLmFrameState(candidate_x,
                                        initial_camera,
                                        frame_views,
                                        T_reference_board_by_id);
    if (candidate.success && std::isfinite(candidate.rmse) &&
        candidate.rmse + 1e-9 < current.rmse) {
      x = candidate_x;
      x.head(camera_parameter_count) =
          ToEigenVector(candidate.camera.CombinedParameterVector());
      current = EvaluateKalibrOuterLmFrameState(
          x, initial_camera, frame_views, T_reference_board_by_id);
      lambda = std::max(1e-9, lambda * 0.3);
      result.iteration_count = iteration + 1;
      if (std::abs(result.initial_rmse - current.rmse) < 1e-9) {
        break;
      }
    } else {
      lambda *= 10.0;
      if (lambda > 1e9) {
        result.iteration_count = iteration + 1;
        break;
      }
    }
  }

  result.camera = current.camera;
  result.final_rmse = current.rmse;
  result.invalid_projection_count = current.invalid_projection_count;
  result.nonfinite_count = current.nonfinite_count;
  result.improved = std::isfinite(result.initial_rmse) &&
                    std::isfinite(result.final_rmse) &&
                    result.final_rmse + 1e-9 < result.initial_rmse;
  return result;
}

std::vector<AutoCameraInitializationResidual> EvaluateSelectedResiduals(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::string& source_label,
    const std::vector<OuterObservationRecord>& observations) {
  std::vector<AutoCameraInitializationResidual> residuals;
  residuals.reserve(observations.size());
  for (const OuterObservationRecord& observation : observations) {
    AutoCameraInitializationResidual residual;
    residual.source_label = source_label;
    residual.frame_index = observation.frame_index;
    residual.frame_label = observation.frame_label;
    residual.board_id = observation.board_id;
    residual.quality = observation.quality;

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double pose_fit_outer_rmse = 0.0;
    residual.pose_success =
        EstimatePoseFromObjectPoints(camera,
                                     observation.object_points,
                                     observation.image_points,
                                     &pose,
                                     &pose_fit_outer_rmse);
    if (residual.pose_success) {
      residual.pose_fit_outer_rmse = pose_fit_outer_rmse;
    } else {
      residual.failure_reason = "pose_fit_failed";
    }
    residuals.push_back(residual);
  }
  return residuals;
}

void ApplySelectedResidualStats(
    const std::vector<AutoCameraInitializationResidual>& residuals,
    AutoCameraInitializationResult* result) {
  if (result == nullptr) {
    throw std::runtime_error("ApplySelectedResidualStats requires a valid result pointer.");
  }
  result->accepted_pose_fit_observation_count = 0;
  result->failed_pose_fit_observation_count = 0;
  result->accepted_frame_count = 0;
  result->accepted_board_observation_count = 0;
  result->initialization_rmse = std::numeric_limits<double>::infinity();

  std::set<int> accepted_frames;
  double total_squared_rmse = 0.0;
  std::vector<double> successful_errors;
  for (const AutoCameraInitializationResidual& residual : residuals) {
    if (!residual.pose_success) {
      ++result->failed_pose_fit_observation_count;
      continue;
    }
    ++result->accepted_pose_fit_observation_count;
    accepted_frames.insert(residual.frame_index);
    total_squared_rmse += residual.pose_fit_outer_rmse * residual.pose_fit_outer_rmse;
    successful_errors.push_back(residual.pose_fit_outer_rmse);
  }

  result->accepted_frame_count = static_cast<int>(accepted_frames.size());
  result->accepted_board_observation_count = result->accepted_pose_fit_observation_count;
  if (result->accepted_pose_fit_observation_count > 0) {
    result->initialization_rmse =
        std::sqrt(total_squared_rmse /
                  static_cast<double>(result->accepted_pose_fit_observation_count));
  }
  result->full_outer_projection_failure_count =
      result->failed_pose_fit_observation_count;
  result->full_outer_nonfinite_count = 0;
  if (!residuals.empty()) {
    result->full_outer_pose_success_rate =
        static_cast<double>(result->accepted_pose_fit_observation_count) /
        static_cast<double>(residuals.size());
  }
  result->full_outer_rmse = result->initialization_rmse;
	  if (!successful_errors.empty()) {
	    std::sort(successful_errors.begin(), successful_errors.end());
	    const std::size_t median_index = successful_errors.size() / 2;
	    result->full_outer_median_error = successful_errors[median_index];
	    const std::size_t p95_index = std::min(
        successful_errors.size() - 1,
        static_cast<std::size_t>(std::ceil(
	            0.95 * static_cast<double>(successful_errors.size()))) - 1);
	    result->full_outer_p95_error = successful_errors[p95_index];
	    std::vector<double> absolute_deviations;
	    absolute_deviations.reserve(successful_errors.size());
	    for (double error : successful_errors) {
	      absolute_deviations.push_back(
	          std::abs(error - result->full_outer_median_error));
	    }
	    std::sort(absolute_deviations.begin(), absolute_deviations.end());
	    const double mad = absolute_deviations[absolute_deviations.size() / 2];
	    result->full_outer_robust_outlier_threshold =
	        result->full_outer_median_error +
	        std::max(1.0, 8.0 * 1.4826 * mad);
	    double inlier_squared_sum = 0.0;
	    int inlier_count = 0;
	    result->full_outer_robust_outlier_count = 0;
	    for (double error : successful_errors) {
	      if (error <= result->full_outer_robust_outlier_threshold) {
	        inlier_squared_sum += error * error;
	        ++inlier_count;
	      } else {
	        ++result->full_outer_robust_outlier_count;
	      }
	    }
	    if (inlier_count > 0) {
	      result->full_outer_robust_inlier_rmse =
	          std::sqrt(inlier_squared_sum / static_cast<double>(inlier_count));
	    }
	  }
	}

void AppendBootstrapObservationRecord(
    const OuterObservationRecord& observation,
    bool used_in_lm,
    bool pose_init_success,
    double pose_fit_outer_rmse,
    std::vector<AutoCameraInitializationBootstrapObservation>* records) {
  if (records == nullptr) {
    return;
  }
  AutoCameraInitializationBootstrapObservation record;
  record.frame_index = observation.frame_index;
  record.frame_label = observation.frame_label;
  record.board_id = observation.board_id;
  record.used_in_lm = used_in_lm;
  record.pose_init_success = pose_init_success;
  record.pose_fit_outer_rmse = pose_fit_outer_rmse;
  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    record.outer_corners[static_cast<std::size_t>(corner_index)] =
        Eigen::Vector2d(
            observation.image_points[static_cast<std::size_t>(corner_index)].x,
            observation.image_points[static_cast<std::size_t>(corner_index)].y);
  }
  records->push_back(record);
}

void AppendDsSeedCandidate(const OuterBootstrapCameraIntrinsics& seed,
                           const std::string& source_label,
                           std::vector<AutoCameraInitializationCandidate>* candidates) {
  if (candidates == nullptr) {
    return;
  }
  AutoCameraInitializationCandidate candidate;
  candidate.source_label = source_label;
  candidate.evaluation_scope = "outer_seed";
  candidate.camera = seed;
  candidates->push_back(candidate);
}

std::vector<AutoCameraInitializationCandidate> GenerateCandidateGrid(
    const cv::Size& image_size,
    const ApriltagInternalConfig& config,
    const AutoCameraInitializationOptions& options,
    const std::vector<OuterBootstrapFrameInput>& frames,
    const std::vector<OuterObservationRecord>& observations,
    SeedConstructionDiagnostics* seed_diagnostics,
    std::vector<std::string>* warnings) {
  std::vector<AutoCameraInitializationCandidate> candidates;
  std::string source_label;
  const OuterBootstrapCameraIntrinsics kalibr_like_seed =
      MakeKalibrLikeOuterSeedIntrinsics(
          image_size, config, observations, &source_label, seed_diagnostics,
          warnings);
  if (kalibr_like_seed.IsValid()) {
    OuterBootstrapCameraIntrinsics refined_seed = kalibr_like_seed;
	    if (kalibr_like_seed.NormalizedFamilyString() == "ds-none") {
	      AppendUniqueWarning(
	          "Layout/LOO diagnostics are not allowed to update DS intrinsics; "
	          "the selected initialization seed may only be refined by the "
	          "outer-corner LM with independent per-board poses.",
	          warnings);
	    }
    AutoCameraInitializationCandidate candidate;
    candidate.source_label = source_label;
    candidate.evaluation_scope = "outer_seed";
    candidate.camera = refined_seed;
    candidates.push_back(candidate);
  }

  const double center_u = 0.5 * static_cast<double>(image_size.width);
  const double center_v = 0.5 * static_cast<double>(image_size.height);
  OuterBootstrapCameraIntrinsics seed = MakeGenericSeedIntrinsics(image_size, config);
  const std::string family = seed.NormalizedFamilyString();
  if (family == "ds-none") {
    candidates.clear();
    if (source_label == "kalibr_source_verified_omni_to_ds_seed") {
      const std::vector<double> direct_gammas =
          CollectDirectOuterQuadOmniGammaCandidates(
              image_size, observations, nullptr, nullptr);
      if (!direct_gammas.empty()) {
        candidates.reserve(direct_gammas.size());
        for (double gamma : direct_gammas) {
          OuterBootstrapCameraIntrinsics direct_seed = seed;
          direct_seed.xi = 0.0;
          direct_seed.alpha = 0.5;
          direct_seed.beta = 1.0;
          direct_seed.fu = 0.5 * gamma;
          direct_seed.fv = 0.5 * gamma;
          direct_seed.cu = 0.5 * static_cast<double>(image_size.width - 1);
          direct_seed.cv = 0.5 * static_cast<double>(image_size.height - 1);
          direct_seed.distortion_coeffs.clear();
          ClampIntrinsicsInPlace(&direct_seed);
          AppendDsSeedCandidate(direct_seed,
                                "outer_quad_direct_omni_to_ds_seed",
                                &candidates);
        }
        AppendUniqueWarning(
            "DS grid enumeration is disabled; using direct four-corner "
            "outer-quad Omni gamma candidates only as diagnostics; final seed "
            "selection is based on multi-observation pose health.",
            warnings);
      }
    }
    const double base_focal = FisheyeResolutionFocalPrior(image_size);
    const std::array<double, 9> focal_scales{
        0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50, 1.70};
    for (double focal_scale : focal_scales) {
      OuterBootstrapCameraIntrinsics physical_seed =
          MakeGenericSeedIntrinsics(image_size, config);
      physical_seed.xi = 0.0;
      physical_seed.alpha = 0.5;
      physical_seed.beta = 1.0;
      physical_seed.fu = focal_scale * base_focal;
      physical_seed.fv = physical_seed.fu;
      physical_seed.cu = 0.5 * static_cast<double>(image_size.width - 1);
      physical_seed.cv = 0.5 * static_cast<double>(image_size.height - 1);
      physical_seed.distortion_coeffs.clear();
      ClampIntrinsicsInPlace(&physical_seed);
      AppendDsSeedCandidate(physical_seed,
                            std::abs(focal_scale - 1.0) < 1e-12
                                ? "fallback_outer_only_ds_seed"
                                : "multi_observation_outer_physical_ds_seed",
                            &candidates);
    }
    AppendUniqueWarning(
        "Added a compact physically plausible DS seed set around resolution/pi; "
        "selection uses all outer observations, not a single tag gamma median.",
        warnings);
    if (candidates.empty()) {
      AppendUniqueWarning(
          "DS outer seed was invalid and DS parameter grid enumeration is "
          "disabled for the Stage5 baseline.",
          warnings);
    } else {
      AppendUniqueWarning(
          "DS parameter grid enumeration is disabled; using outer-only DS "
          "initialization candidates.",
          warnings);
    }
    return candidates;
  }
  if (!candidates.empty() &&
      options.refine_mode == AutoCameraInitializationRefineMode::KalibrOuterLm) {
    AppendUniqueWarning(
        "Using the outer-only model seed as the primary initialization "
        "candidate; legacy parameter grid is skipped in "
        "kalibr_outer_lm mode.",
        warnings);
    return candidates;
  }

  for (double focal_scale : options.focal_scale_candidates) {
    if (family == "eucm-none") {
      for (double alpha : options.eucm_alpha_candidates) {
        for (double beta : options.eucm_beta_candidates) {
          AutoCameraInitializationCandidate candidate;
          candidate.source_label = "auto_grid";
          candidate.evaluation_scope = "sampled";
          candidate.camera = seed;
          candidate.camera.alpha = alpha;
          candidate.camera.beta = beta;
          candidate.camera.fu = focal_scale * static_cast<double>(image_size.width);
          candidate.camera.fv = focal_scale * static_cast<double>(image_size.height);
          candidate.camera.cu = center_u;
          candidate.camera.cv = center_v;
          ClampIntrinsicsInPlace(&candidate.camera);
          candidates.push_back(candidate);
        }
      }
    } else if (family == "pinhole-equi") {
      for (double k1 : options.equidistant_k1_candidates) {
        AutoCameraInitializationCandidate candidate;
        candidate.source_label = "auto_grid";
        candidate.evaluation_scope = "sampled";
        candidate.camera = seed;
        candidate.camera.fu = focal_scale * static_cast<double>(image_size.width);
        candidate.camera.fv = focal_scale * static_cast<double>(image_size.height);
        candidate.camera.cu = center_u;
        candidate.camera.cv = center_v;
        candidate.camera.distortion_coeffs = {k1, 0.0, 0.0, 0.0};
        ClampIntrinsicsInPlace(&candidate.camera);
        candidates.push_back(candidate);
      }
    }
  }
  return candidates;
}

}  // namespace

OuterOnlyCameraInitializer::OuterOnlyCameraInitializer(
    ApriltagInternalConfig config,
    AutoCameraInitializationOptions options)
    : config_(std::move(config)), options_(std::move(options)) {}

AutoCameraInitializationResult OuterOnlyCameraInitializer::Initialize(
    const std::vector<OuterBootstrapFrameInput>& frames) const {
  const auto initialization_start = std::chrono::steady_clock::now();
  AutoCameraInitializationResult result;
  result.requested_mode = options_.mode;
  result.selected_mode = CameraInitializationMode::Manual;
  result.refine_mode = options_.refine_mode;
  result.image_size = InferImageSize(frames);
  result.stage5_init_seed_method = "not_attempted";
  result.stage5_init_seed_source = "none";
  result.stage5_init_omni_gamma_source = "not_attempted";
  result.stage5_init_ds_mapping = "not_applicable";
  result.stage5_init_ds_grid_enumeration_enabled = 0;
  result.stage5_init_uses_yaml_intrinsics = 0;
  result.stage5_init_uses_kalibr_camchain_intrinsics = 0;
  result.stage5_init_outer_only = 1;
  result.stage5_init_uses_layout_to_update_intrinsics = 0;
  result.stage5_init_layout_loo_diagnostics_only = 1;
  result.stage5_init_multiboard_frame_objective_enabled = 0;
  result.stage5_init_fixed_layout_frame_constraint_used = 0;
  result.stage5_init_optimizes_layout_variables = 0;
	  result.stage5_init_lm_selection_objective =
	      "independent_board_pose_outer_lm_final_reprojection";
  result.stage5_init_selection_prefilter = "coverage_diversity_spacing";
  result.stage5_init_selection_scorer =
      "outer_initializer_intrinsics_jacobian_information_proxy";
  result.stage5_init_selection_uses_information_metric = 1;
  result.stage5_init_selection_is_exact_kalibr_information_theoretic = 0;
  auto stamp_runtime = [&result, initialization_start]() {
    const auto now = std::chrono::steady_clock::now();
    result.stage5_init_runtime_seconds =
        std::chrono::duration<double>(now - initialization_start).count();
  };

  if (result.image_size.width <= 0 || result.image_size.height <= 0) {
    result.failure_reason = "Could not infer image size from outer observations.";
    stamp_runtime();
    return result;
  }

  bool used_manual_intermediate_camera = false;
  bool used_explicit_initial_camera = false;
  bool used_manual_generic_seed = false;
  std::string manual_source_label;
  const OuterBootstrapCameraIntrinsics manual_camera =
      BuildManualInitialCamera(result.image_size,
                               config_,
                               options_,
                               &used_manual_intermediate_camera,
                               &used_explicit_initial_camera,
                               &used_manual_generic_seed,
                               &manual_source_label,
                               &result.warnings);

  if (options_.mode == CameraInitializationMode::Manual) {
    result.success = manual_camera.IsValid();
    result.selected_mode = CameraInitializationMode::Manual;
    result.selected_source_label = manual_source_label;
    result.selected_camera = manual_camera;
    result.used_manual_intermediate_camera = used_manual_intermediate_camera;
    result.used_explicit_initial_camera = used_explicit_initial_camera;
    result.used_manual_generic_seed = used_manual_generic_seed;
    const std::vector<OuterObservationRecord> observations =
        CollectOuterObservations(frames, config_);
    result.total_valid_outer_observation_count = static_cast<int>(observations.size());
    result.selected_residuals =
        EvaluateSelectedResiduals(result.selected_camera,
                                  result.selected_source_label,
                                  observations);
    ApplySelectedResidualStats(result.selected_residuals, &result);
	    if (!result.success) {
	      result.failure_reason = "Manual initialization produced invalid intrinsics.";
	    }
    result.stage5_init_seed_method = manual_source_label;
    result.stage5_init_seed_source = manual_source_label;
    result.stage5_init_selection_uses_information_metric = 0;
    result.stage5_init_selection_scorer = "not_used_manual_initialization";
    stamp_runtime();
	    return result;
	  }

  result.auto_attempted = true;
  const std::vector<OuterObservationRecord> all_observations =
      CollectOuterObservations(frames, config_);
  result.total_valid_outer_observation_count =
      static_cast<int>(all_observations.size());

  if (all_observations.empty()) {
    result.failure_reason =
        "No valid outer observations with four refined corners were available for "
        "automatic camera initialization.";
  } else {
    const std::vector<OuterObservationRecord> sampled_observations =
        SampleObservations(all_observations, options_.max_candidate_observations);
    result.sampled_observation_count =
        static_cast<int>(sampled_observations.size());

    SeedConstructionDiagnostics seed_diagnostics;
    std::vector<AutoCameraInitializationCandidate> candidates =
        GenerateCandidateGrid(
            result.image_size, config_, options_, frames, all_observations,
            &seed_diagnostics, &result.warnings);
    result.stage5_init_seed_method = seed_diagnostics.seed_method;
    result.stage5_init_seed_source = seed_diagnostics.seed_source;
    result.stage5_init_omni_gamma = seed_diagnostics.omni_gamma;
    result.stage5_init_omni_gamma_source = seed_diagnostics.omni_gamma_source;
    result.stage5_init_ds_mapping = seed_diagnostics.ds_mapping;
    result.stage5_init_ds_mapping_verified_against_kalibr_source =
        seed_diagnostics.ds_mapping_verified_against_kalibr_source;
    result.stage5_init_ds_grid_enumeration_enabled =
        seed_diagnostics.ds_grid_enumeration_enabled;
    result.stage5_init_selection_prefilter = "coverage_diversity_spacing";
    result.stage5_init_selection_scorer =
        "outer_initializer_intrinsics_jacobian_information_proxy";
    result.stage5_init_selection_uses_information_metric = 1;
    result.stage5_init_selection_is_exact_kalibr_information_theoretic = 0;
    if (!seed_diagnostics.fallback_reason.empty()) {
      AppendUniqueWarning("fallback_reason: " + seed_diagnostics.fallback_reason,
                          &result.warnings);
    }
    for (AutoCameraInitializationCandidate& candidate : candidates) {
      candidate = EvaluateCandidateOnObservations(
          candidate.camera, candidate.source_label, "sampled", sampled_observations);
    }

    std::sort(candidates.begin(), candidates.end(), CandidateIsBetter);
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      candidates[index].rank = static_cast<int>(index + 1);
    }
    result.candidate_count = static_cast<int>(candidates.size());
    result.candidates = candidates;

    if (!candidates.empty() &&
        (IsAcceptableAutoCandidate(
             candidates.front(), result.total_valid_outer_observation_count) ||
         (options_.refine_best_candidate &&
          options_.refine_mode ==
              AutoCameraInitializationRefineMode::KalibrOuterLm &&
          IsRefinableKalibrLikeSeed(
              candidates.front(), result.total_valid_outer_observation_count)))) {
      AutoCameraInitializationCandidate best_candidate = candidates.front();
      OuterBootstrapCameraIntrinsics selected_camera = best_candidate.camera;
      std::string selected_source_label = best_candidate.source_label;
      if (options_.refine_best_candidate &&
          options_.refine_mode ==
              AutoCameraInitializationRefineMode::CoordinateSearch) {
        const OuterBootstrapCameraIntrinsics refined_camera =
            RefineCandidateCamera(best_candidate.camera, all_observations);
        const AutoCameraInitializationCandidate refined_eval =
            EvaluateCandidateOnObservations(refined_camera,
                                            "auto_grid_refined",
                                            "full",
                                            all_observations);
        if (CandidateObjective(refined_eval) + 1e-9 <
            CandidateObjective(best_candidate)) {
          best_candidate = refined_eval;
          selected_camera = refined_camera;
          selected_source_label = "auto_grid_refined";
          result.selected_candidate_refined = true;
        }
      } else if (options_.refine_best_candidate &&
                 options_.refine_mode ==
                     AutoCameraInitializationRefineMode::KalibrOuterLm) {
        result.stage5_init_calibrate_intrinsics_enabled = 1;
        result.stage5_init_calibrate_intrinsics_released_params =
            JoinLabels(best_candidate.camera.CombinedParameterLabels(), ",");
	        result.stage5_init_calibrate_intrinsics_optimizer =
	            "outer_corner_reprojection_lm_camera_intrinsics_and_pose";
	        result.stage5_init_multiboard_frame_objective_enabled = 0;
	        result.stage5_init_fixed_layout_frame_constraint_used = 0;
	        result.stage5_init_optimizes_layout_variables = 0;
	        result.stage5_init_uses_layout_to_update_intrinsics = 0;
	        result.stage5_init_layout_loo_diagnostics_only = 1;
		        result.stage5_init_lm_selection_objective =
		            "independent_board_pose_outer_lm_final_reprojection";
	        const int max_lm_seed_count =
	            std::max(
	                1,
	                std::min(options_.top_candidate_count,
	                         static_cast<int>(candidates.size())));
        int lm_seed_attempt_count = 0;
        int lm_seed_accept_count = 0;
        bool frame_cohesion_diagnostics_skip_logged = false;
	        double best_objective = CandidateObjective(best_candidate);
	        double best_lm_selection_objective =
	            std::numeric_limits<double>::infinity();
	        KalibrOuterLmRefinementResult best_lm_refined;
	        AutoCameraInitializationCandidate best_lm_seed = best_candidate;
	        std::string best_lm_source_label;
        std::set<int> best_lm_frame_indices;
        std::vector<OuterObservationRecord> best_selected_lm_observations;
        int best_selected_pose_success_count = 0;
        int best_selected_pose_failure_count = 0;

	        for (const AutoCameraInitializationCandidate& seed_candidate :
	             candidates) {
	          const double max_image_extent = static_cast<double>(
	              std::max(result.image_size.width, result.image_size.height));
	          const bool priority_rank_seed =
	              seed_candidate.rank <= max_lm_seed_count;
	          const bool low_focal_ray_curve_probe =
	              max_image_extent > 0.0 &&
	              seed_candidate.camera.fu >= 0.23 * max_image_extent &&
	              seed_candidate.camera.fu <= 0.27 * max_image_extent;
	          if (!priority_rank_seed && !low_focal_ray_curve_probe) {
	            continue;
	          }
	          if (lm_seed_attempt_count >= max_lm_seed_count &&
	              !low_focal_ray_curve_probe) {
	            continue;
	          }
	          if (!IsRefinableKalibrLikeSeed(
	                  seed_candidate, result.total_valid_outer_observation_count)) {
	            continue;
          }
          ++lm_seed_attempt_count;

          std::string lm_source_label =
              "outer_only_independent_board_pose_lm";
          KalibrOuterLmRefinementResult lm_refined;
          int selected_pose_success_count = 0;
          int selected_pose_failure_count = 0;
          const std::vector<OuterObservationRecord> lm_observations =
              SelectKalibrOuterLmObservations(
                  seed_candidate.camera,
                  all_observations,
                  &selected_pose_success_count,
                  &selected_pose_failure_count);
          std::set<int> lm_frame_indices;
          std::vector<OuterObservationRecord> selected_lm_observations;
          std::ostringstream label;
          label << "outer_only_independent_board_pose_lm_rank"
                << seed_candidate.rank;
          lm_source_label = label.str();
          selected_lm_observations = lm_observations;
          for (const OuterObservationRecord& observation :
               selected_lm_observations) {
            lm_frame_indices.insert(observation.frame_index);
          }
          lm_refined =
              RefineCandidateCameraKalibrOuterLm(seed_candidate.camera,
                                                 selected_lm_observations);

          KalibrOuterLmRefinementResult selection_refined = lm_refined;
          std::string selection_objective_label =
              "independent_board_pose_outer_lm_final_reprojection";
          std::set<int> selection_frame_indices = lm_frame_indices;
          std::vector<OuterObservationRecord> selection_lm_observations =
              selected_lm_observations;
          bool used_fixed_layout_frame_constraint = false;
          if (!frame_cohesion_diagnostics_skip_logged) {
            AppendUniqueWarning(
                "Fixed-layout frame-cohesion LM diagnostics skipped by "
                "default: diagnostics_only=1, layout_updates_intrinsics=0, "
                "and this diagnostic can dominate full-dataset initialization "
                "runtime. Intrinsics are selected by independent outer-only "
                "board-pose LM plus full outer health.",
                &result.warnings);
            frame_cohesion_diagnostics_skip_logged = true;
          }

          const AutoCameraInitializationCandidate refined_eval =
              EvaluateCandidateOnObservations(selection_refined.camera,
	                                              lm_source_label,
	                                              "full",
	                                              all_observations);
	          const double seed_objective = CandidateObjective(seed_candidate);
	          const double refined_objective = CandidateObjective(refined_eval);
	          const double lm_selection_objective =
	              std::isfinite(selection_refined.final_rmse)
	                  ? selection_refined.final_rmse *
	                        selection_refined.final_rmse
	                  : std::numeric_limits<double>::infinity();
	          std::ostringstream trial_warning;
	          trial_warning
	              << "Kalibr-style multi-start LM trial rank="
              << seed_candidate.rank
              << " source=" << seed_candidate.source_label
              << " seed_fu=" << seed_candidate.camera.fu
              << " seed_alpha=" << seed_candidate.camera.alpha
              << " refined_fu=" << selection_refined.camera.fu
	              << " refined_alpha=" << selection_refined.camera.alpha
	              << " refined_xi=" << selection_refined.camera.xi
	              << " seed_objective=" << seed_objective
	              << " refined_objective=" << refined_objective
	              << " lm_initial_rmse=" << selection_refined.initial_rmse
	              << " lm_final_rmse=" << selection_refined.final_rmse
	              << " lm_selection_objective=" << lm_selection_objective
	              << " selected_frames=" << selection_frame_indices.size()
	              << " selected_board_observations="
	              << selection_refined.view_count
              << " fixed_layout_frame_constraint="
              << (used_fixed_layout_frame_constraint ? 1 : 0)
              << " layout_updates_intrinsics=0"
              << " layout_variables_optimized=0.";
	          AppendUniqueWarning(trial_warning.str(), &result.warnings);

		          const bool full_outer_health_acceptable =
		              selection_refined.improved &&
		              std::isfinite(selection_refined.final_rmse) &&
		              selection_refined.residual_count > 0 &&
		              refined_eval.valid &&
		              IsRefinableKalibrLikeSeed(
		                  refined_eval,
		                  result.total_valid_outer_observation_count);
	          const bool frame_objective_improved =
	              std::isfinite(lm_selection_objective) &&
	              lm_selection_objective + 1e-9 <
	                  best_lm_selection_objective;
	          if (!full_outer_health_acceptable || !frame_objective_improved) {
	            continue;
	          }
	          best_objective = refined_objective;
	          best_lm_selection_objective = lm_selection_objective;
	          best_lm_refined = selection_refined;
          best_lm_seed = seed_candidate;
          best_lm_source_label = lm_source_label;
          best_lm_frame_indices = selection_frame_indices;
          best_selected_lm_observations = selection_lm_observations;
          best_selected_pose_success_count = selected_pose_success_count;
          best_selected_pose_failure_count = selected_pose_failure_count;
          best_candidate = refined_eval;
          selected_camera = selection_refined.camera;
          selected_source_label = lm_source_label;
          result.stage5_init_multiboard_frame_objective_enabled =
              used_fixed_layout_frame_constraint ? 1 : 0;
          result.stage5_init_fixed_layout_frame_constraint_used =
              used_fixed_layout_frame_constraint ? 1 : 0;
          result.stage5_init_optimizes_layout_variables = 0;
          result.stage5_init_lm_selection_objective =
              selection_objective_label;
          result.selected_candidate_refined = true;
          ++lm_seed_accept_count;
        }

        if (result.selected_candidate_refined &&
            !best_lm_source_label.empty()) {
          result.stage5_init_selected_pose_success_count =
              best_selected_pose_success_count;
          result.stage5_init_selected_pose_total_count =
              best_selected_pose_success_count +
              best_selected_pose_failure_count;
          for (const OuterObservationRecord& observation :
               best_selected_lm_observations) {
            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            double pose_rmse = std::numeric_limits<double>::quiet_NaN();
            const bool pose_success =
                EstimatePoseFromObjectPoints(best_lm_seed.camera,
                                             observation.object_points,
                                             observation.image_points,
                                             &pose,
                                             &pose_rmse);
            AppendBootstrapObservationRecord(
                observation,
                true,
                pose_success,
                pose_rmse,
                &result.lm_bootstrap_observations);
          }
          result.lm_frame_count =
              static_cast<int>(best_lm_frame_indices.size());
          result.lm_view_count = best_lm_refined.view_count;
          result.lm_residual_count = best_lm_refined.residual_count;
          result.lm_invalid_projection_count =
              best_lm_refined.invalid_projection_count;
          result.lm_nonfinite_count = best_lm_refined.nonfinite_count;
          result.lm_iteration_count = best_lm_refined.iteration_count;
          result.lm_initial_rmse = best_lm_refined.initial_rmse;
          result.lm_final_rmse = best_lm_refined.final_rmse;
          std::ostringstream selection_warning;
          selection_warning
              << "Outer-only multi-start selected-observation "
              << best_lm_source_label
              << " selected " << result.lm_frame_count
              << " frames / " << result.lm_view_count
              << " pose-init-valid board observations from adaptive "
                 "coverage/diversity plus intrinsics-information selection"
              << "; tried " << lm_seed_attempt_count
              << " candidate seeds and accepted " << lm_seed_accept_count
              << " objective-improving updates; final health was evaluated on "
                 "all observations.";
          AppendUniqueWarning(selection_warning.str(), &result.warnings);
        } else {
          std::ostringstream health_warning;
          health_warning
              << "Kalibr-style multi-start LM did not improve the selected "
                 "seed candidate under full outer health; tried "
              << lm_seed_attempt_count
              << " candidate seeds. Keeping the unrefined initialization.";
          AppendUniqueWarning(health_warning.str(), &result.warnings);
        }
      }

      result.success = true;
      result.selected_mode = CameraInitializationMode::Auto;
      result.selected_source_label = selected_source_label;
      result.selected_camera = selected_camera;
      result.selected_residuals =
          EvaluateSelectedResiduals(result.selected_camera,
                                    result.selected_source_label,
                                    all_observations);
      ApplySelectedResidualStats(result.selected_residuals, &result);
    } else if (!candidates.empty()) {
      result.failure_reason =
          "Automatic outer-only camera initialization did not find a sufficiently "
          "stable outer-only camera candidate.";
    }
  }

  if (!result.success &&
      options_.mode == CameraInitializationMode::AutoWithManualFallback &&
      manual_camera.IsValid()) {
    std::ostringstream warning;
    warning << "Auto camera initialization failed: "
            << (result.failure_reason.empty() ? "unknown failure"
                                              : result.failure_reason)
            << "; falling back to " << manual_source_label << ".";
    AppendUniqueWarning(warning.str(), &result.warnings);
    result.success = true;
    result.fallback_used = true;
    result.selected_mode = CameraInitializationMode::Manual;
    result.selected_source_label = manual_source_label;
    result.selected_camera = manual_camera;
    result.used_manual_intermediate_camera = used_manual_intermediate_camera;
    result.used_explicit_initial_camera = used_explicit_initial_camera;
    result.used_manual_generic_seed = used_manual_generic_seed;
    result.selected_residuals =
        EvaluateSelectedResiduals(result.selected_camera,
                                  result.selected_source_label,
                                  all_observations);
    ApplySelectedResidualStats(result.selected_residuals, &result);
    result.failure_reason.clear();
  } else if (!result.success && options_.mode == CameraInitializationMode::Auto) {
    AppendUniqueWarning(result.failure_reason, &result.warnings);
  } else {
    result.used_manual_intermediate_camera =
        (result.selected_mode == CameraInitializationMode::Manual) &&
        used_manual_intermediate_camera;
    result.used_explicit_initial_camera =
        (result.selected_mode == CameraInitializationMode::Manual) &&
        used_explicit_initial_camera;
    result.used_manual_generic_seed =
        (result.selected_mode == CameraInitializationMode::Manual) &&
        used_manual_generic_seed;
  }

  stamp_runtime();
	  return result;
	}

void WriteAutoCameraInitializationSummary(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "requested_mode: " << ToString(result.requested_mode) << "\n";
  output << "selected_mode: " << ToString(result.selected_mode) << "\n";
  output << "fallback_used: " << (result.fallback_used ? 1 : 0) << "\n";
  output << "auto_attempted: " << (result.auto_attempted ? 1 : 0) << "\n";
  output << "used_manual_intermediate_camera: "
         << (result.used_manual_intermediate_camera ? 1 : 0) << "\n";
  output << "used_config_intermediate_camera: "
         << (result.used_manual_intermediate_camera ? 1 : 0) << "\n";
  output << "used_explicit_initial_camera: "
         << (result.used_explicit_initial_camera ? 1 : 0) << "\n";
  output << "used_manual_generic_seed: "
         << (result.used_manual_generic_seed ? 1 : 0) << "\n";
  output << "selected_candidate_refined: "
         << (result.selected_candidate_refined ? 1 : 0) << "\n";
  output << "stage5_init_refine_mode: " << ToString(result.refine_mode) << "\n";
  output << "stage5_init_seed_method: " << result.stage5_init_seed_method << "\n";
  output << "stage5_init_seed_source: " << result.stage5_init_seed_source << "\n";
  output << "stage5_init_omni_gamma: " << result.stage5_init_omni_gamma << "\n";
  output << "stage5_init_omni_gamma_source: "
         << result.stage5_init_omni_gamma_source << "\n";
  output << "stage5_init_ds_mapping: " << result.stage5_init_ds_mapping << "\n";
  output << "stage5_init_ds_mapping_verified_against_kalibr_source: "
         << result.stage5_init_ds_mapping_verified_against_kalibr_source << "\n";
  output << "stage5_init_ds_grid_enumeration_enabled: "
         << result.stage5_init_ds_grid_enumeration_enabled << "\n";
  output << "stage5_init_uses_yaml_intrinsics: "
         << result.stage5_init_uses_yaml_intrinsics << "\n";
  output << "stage5_init_uses_kalibr_camchain_intrinsics: "
         << result.stage5_init_uses_kalibr_camchain_intrinsics << "\n";
  output << "stage5_init_outer_only: " << result.stage5_init_outer_only << "\n";
  output << "stage5_init_uses_layout_to_update_intrinsics: "
         << result.stage5_init_uses_layout_to_update_intrinsics << "\n";
  output << "stage5_init_layout_loo_diagnostics_only: "
         << result.stage5_init_layout_loo_diagnostics_only << "\n";
  output << "stage5_init_multiboard_frame_objective_enabled: "
         << result.stage5_init_multiboard_frame_objective_enabled << "\n";
  output << "stage5_init_fixed_layout_frame_constraint_used: "
         << result.stage5_init_fixed_layout_frame_constraint_used << "\n";
  output << "stage5_init_optimizes_layout_variables: "
         << result.stage5_init_optimizes_layout_variables << "\n";
  output << "stage5_init_lm_selection_objective: "
         << result.stage5_init_lm_selection_objective << "\n";
  output << "stage5_init_selection_prefilter: "
         << result.stage5_init_selection_prefilter << "\n";
  output << "stage5_init_selection_scorer: "
         << result.stage5_init_selection_scorer << "\n";
  output << "stage5_init_selection_uses_information_metric: "
         << result.stage5_init_selection_uses_information_metric << "\n";
  output << "stage5_init_selection_is_exact_kalibr_information_theoretic: "
         << result.stage5_init_selection_is_exact_kalibr_information_theoretic
         << "\n";
  output << "stage5_init_calibrate_intrinsics_enabled: "
         << result.stage5_init_calibrate_intrinsics_enabled << "\n";
  output << "stage5_init_calibrate_intrinsics_released_params: "
         << result.stage5_init_calibrate_intrinsics_released_params << "\n";
  output << "stage5_init_calibrate_intrinsics_optimizer: "
         << result.stage5_init_calibrate_intrinsics_optimizer << "\n";
  output << "stage5_init_runtime_seconds: "
         << result.stage5_init_runtime_seconds << "\n";
  output << "stage5_init_selected_frame_count: " << result.lm_frame_count << "\n";
  output << "stage5_init_selected_board_observation_count: "
         << result.lm_view_count << "\n";
  output << "stage5_init_selected_pose_success_count: "
         << result.stage5_init_selected_pose_success_count << "\n";
  output << "stage5_init_selected_pose_total_count: "
         << result.stage5_init_selected_pose_total_count << "\n";
  output << "stage5_init_full_outer_pose_success_count: "
         << result.accepted_pose_fit_observation_count << "\n";
  output << "stage5_init_full_outer_pose_total_count: "
         << result.total_valid_outer_observation_count << "\n";
  output << "stage5_init_pose_success_rate: "
         << result.full_outer_pose_success_rate << "\n";
  output << "stage5_init_full_outer_rmse: " << result.full_outer_rmse << "\n";
  output << "stage5_init_full_outer_median: "
         << result.full_outer_median_error << "\n";
	  output << "stage5_init_full_outer_p95: " << result.full_outer_p95_error << "\n";
	  output << "stage5_init_full_outer_robust_inlier_rmse: "
	         << result.full_outer_robust_inlier_rmse << "\n";
	  output << "stage5_init_full_outer_robust_outlier_threshold: "
	         << result.full_outer_robust_outlier_threshold << "\n";
	  output << "stage5_init_full_outer_robust_outlier_count: "
	         << result.full_outer_robust_outlier_count << "\n";
	  output << "stage5_init_full_outer_projection_failure_count: "
	         << result.full_outer_projection_failure_count << "\n";
  output << "stage5_init_full_outer_nonfinite_residual_count: "
         << result.full_outer_nonfinite_count << "\n";
  output << "kalibr_outer_lm_frame_count: " << result.lm_frame_count << "\n";
  output << "kalibr_outer_lm_view_count: " << result.lm_view_count << "\n";
  output << "kalibr_outer_lm_residual_count: " << result.lm_residual_count << "\n";
  output << "kalibr_outer_lm_invalid_projection_count: "
         << result.lm_invalid_projection_count << "\n";
  output << "kalibr_outer_lm_nonfinite_count: "
         << result.lm_nonfinite_count << "\n";
  output << "kalibr_outer_lm_iteration_count: "
         << result.lm_iteration_count << "\n";
  output << "kalibr_outer_lm_initial_rmse: " << result.lm_initial_rmse << "\n";
  output << "kalibr_outer_lm_final_rmse: " << result.lm_final_rmse << "\n";
  output << "full_outer_pose_success_rate: "
         << result.full_outer_pose_success_rate << "\n";
  output << "full_outer_rmse: " << result.full_outer_rmse << "\n";
  output << "full_outer_median_error: "
         << result.full_outer_median_error << "\n";
	  output << "full_outer_p95_error: " << result.full_outer_p95_error << "\n";
	  output << "full_outer_robust_inlier_rmse: "
	         << result.full_outer_robust_inlier_rmse << "\n";
	  output << "full_outer_robust_outlier_threshold: "
	         << result.full_outer_robust_outlier_threshold << "\n";
	  output << "full_outer_robust_outlier_count: "
	         << result.full_outer_robust_outlier_count << "\n";
	  output << "full_outer_projection_failure_count: "
	         << result.full_outer_projection_failure_count << "\n";
  output << "full_outer_nonfinite_count: "
         << result.full_outer_nonfinite_count << "\n";
  output << "bootstrap_internal_points_used: "
         << result.bootstrap_internal_points_used << "\n";
  output << "selected_source_label: " << result.selected_source_label << "\n";
  output << "image_width: " << result.image_size.width << "\n";
  output << "image_height: " << result.image_size.height << "\n";
  output << "selected_camera_model_family: "
         << result.selected_camera.NormalizedFamilyString() << "\n";
  output << "selected_camera_model: " << result.selected_camera.NormalizedCameraModel() << "\n";
  output << "selected_distortion_model: "
         << result.selected_camera.NormalizedDistortionModel() << "\n";
  output << "selected_intrinsics_labels: ";
  for (const std::string& label : result.selected_camera.IntrinsicsLabels()) {
    output << label << " ";
  }
  output << "\n";
  output << "selected_intrinsics_csv: ";
  const std::vector<double> selected_intrinsics = result.selected_camera.IntrinsicsVector();
  for (std::size_t index = 0; index < selected_intrinsics.size(); ++index) {
    if (index > 0) {
      output << ",";
    }
    output << selected_intrinsics[index];
  }
  output << "\n";
  output << "selected_distortion_labels: ";
  for (const std::string& label : result.selected_camera.DistortionLabels()) {
    output << label << " ";
  }
  output << "\n";
  output << "selected_distortion_csv: ";
  const std::vector<double> selected_distortion = result.selected_camera.DistortionVector();
  for (std::size_t index = 0; index < selected_distortion.size(); ++index) {
    if (index > 0) {
      output << ",";
    }
    output << selected_distortion[index];
  }
  output << "\n";
  output << "selected_xi: " << result.selected_camera.xi << "\n";
  output << "selected_alpha: " << result.selected_camera.alpha << "\n";
  output << "selected_fu: " << result.selected_camera.fu << "\n";
  output << "selected_fv: " << result.selected_camera.fv << "\n";
  output << "selected_cu: " << result.selected_camera.cu << "\n";
  output << "selected_cv: " << result.selected_camera.cv << "\n";
  output << "candidate_count: " << result.candidate_count << "\n";
  output << "sampled_observation_count: " << result.sampled_observation_count << "\n";
  output << "total_valid_outer_observation_count: "
         << result.total_valid_outer_observation_count << "\n";
  output << "accepted_pose_fit_observation_count: "
         << result.accepted_pose_fit_observation_count << "\n";
  output << "failed_pose_fit_observation_count: "
         << result.failed_pose_fit_observation_count << "\n";
  output << "accepted_frame_count: " << result.accepted_frame_count << "\n";
  output << "accepted_board_observation_count: "
         << result.accepted_board_observation_count << "\n";
  output << "best_candidate_rmse: " << result.initialization_rmse << "\n";
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteAutoCameraInitializationCandidatesCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "rank,source_label,evaluation_scope,camera_model_family,camera_model,"
         << "distortion_model,intrinsics_labels,intrinsics_csv,distortion_labels,distortion_csv,"
         << "xi,alpha,beta,fu,fv,cu,cv,"
         << "observation_count,pose_success_count,pose_failure_count,"
         << "successful_frame_count,successful_board_count,success_rate,"
         << "mean_observation_rmse,leave_one_board_out_attempt_count,"
         << "leave_one_board_out_success_count,leave_one_board_out_rmse,"
         << "relative_layout_pair_family_count,relative_layout_pair_sample_count,"
         << "relative_layout_translation_rmse,relative_layout_rotation_rmse_deg,"
         << "relative_layout_consistency_score,"
         << "valid,failure_reason\n";
  for (const AutoCameraInitializationCandidate& candidate : result.candidates) {
    std::ostringstream intrinsics_labels_stream;
    const std::vector<std::string> intrinsics_labels = candidate.camera.IntrinsicsLabels();
    for (std::size_t index = 0; index < intrinsics_labels.size(); ++index) {
      if (index > 0) {
        intrinsics_labels_stream << "|";
      }
      intrinsics_labels_stream << intrinsics_labels[index];
    }
    std::ostringstream intrinsics_values_stream;
    const std::vector<double> intrinsics_values = candidate.camera.IntrinsicsVector();
    for (std::size_t index = 0; index < intrinsics_values.size(); ++index) {
      if (index > 0) {
        intrinsics_values_stream << "|";
      }
      intrinsics_values_stream << intrinsics_values[index];
    }
    std::ostringstream distortion_labels_stream;
    const std::vector<std::string> distortion_labels = candidate.camera.DistortionLabels();
    for (std::size_t index = 0; index < distortion_labels.size(); ++index) {
      if (index > 0) {
        distortion_labels_stream << "|";
      }
      distortion_labels_stream << distortion_labels[index];
    }
    std::ostringstream distortion_values_stream;
    const std::vector<double> distortion_values = candidate.camera.DistortionVector();
    for (std::size_t index = 0; index < distortion_values.size(); ++index) {
      if (index > 0) {
        distortion_values_stream << "|";
      }
      distortion_values_stream << distortion_values[index];
    }
    output << candidate.rank << ","
           << candidate.source_label << ","
           << candidate.evaluation_scope << ","
           << candidate.camera.NormalizedFamilyString() << ","
           << candidate.camera.NormalizedCameraModel() << ","
           << candidate.camera.NormalizedDistortionModel() << ","
           << intrinsics_labels_stream.str() << ","
           << intrinsics_values_stream.str() << ","
           << distortion_labels_stream.str() << ","
           << distortion_values_stream.str() << ","
           << candidate.camera.xi << ","
           << candidate.camera.alpha << ","
           << candidate.camera.beta << ","
           << candidate.camera.fu << ","
           << candidate.camera.fv << ","
           << candidate.camera.cu << ","
           << candidate.camera.cv << ","
           << candidate.observation_count << ","
           << candidate.pose_success_count << ","
           << candidate.pose_failure_count << ","
           << candidate.successful_frame_count << ","
           << candidate.successful_board_count << ","
           << candidate.success_rate << ","
           << candidate.mean_observation_rmse << ","
           << candidate.leave_one_board_out_attempt_count << ","
           << candidate.leave_one_board_out_success_count << ","
           << candidate.leave_one_board_out_rmse << ","
           << candidate.relative_layout_pair_family_count << ","
           << candidate.relative_layout_pair_sample_count << ","
           << candidate.relative_layout_translation_rmse << ","
           << candidate.relative_layout_rotation_rmse_deg << ","
           << candidate.relative_layout_consistency_score << ","
           << (candidate.valid ? 1 : 0) << ","
           << candidate.failure_reason << "\n";
  }
}

void WriteAutoCameraInitializationOuterResidualsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "source_label,frame_index,frame_label,board_id,quality,pose_success,"
         << "pose_fit_outer_rmse,failure_reason\n";
  for (const AutoCameraInitializationResidual& residual : result.selected_residuals) {
    output << residual.source_label << ","
           << residual.frame_index << ","
           << residual.frame_label << ","
           << residual.board_id << ","
           << residual.quality << ","
           << (residual.pose_success ? 1 : 0) << ","
           << residual.pose_fit_outer_rmse << ","
           << residual.failure_reason << "\n";
  }
}

void WriteAutoCameraInitializationBootstrapViewsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "frame_index,frame_label,board_id,used_in_lm,pose_init_success,"
         << "pose_fit_outer_rmse,corner_index,x,y\n";
  for (const AutoCameraInitializationBootstrapObservation& observation :
       result.lm_bootstrap_observations) {
    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      const Eigen::Vector2d& corner =
          observation.outer_corners[static_cast<std::size_t>(corner_index)];
      output << observation.frame_index << ","
             << observation.frame_label << ","
             << observation.board_id << ","
             << (observation.used_in_lm ? 1 : 0) << ","
             << (observation.pose_init_success ? 1 : 0) << ","
             << observation.pose_fit_outer_rmse << ","
             << corner_index << ","
             << corner.x() << ","
             << corner.y() << "\n";
    }
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
