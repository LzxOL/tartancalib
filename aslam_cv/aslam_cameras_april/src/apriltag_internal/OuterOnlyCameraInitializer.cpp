#include <aslam/cameras/apriltag_internal/OuterOnlyCameraInitializer.hpp>

#include <algorithm>
#include <atomic>
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
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

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

const char* ToString(AutoCameraInitializationSelectionScorer scorer) {
  switch (scorer) {
    case AutoCameraInitializationSelectionScorer::PoseMarginalizedPrincipal:
      return "pose_marginalized_principal";
    case AutoCameraInitializationSelectionScorer::LegacyFixedPose:
      return "legacy_fixed_pose";
  }
  return "unknown";
}

AutoCameraInitializationSelectionScorer
ParseAutoCameraInitializationSelectionScorer(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (normalized == "pose_marginalized_principal" ||
      normalized == "pose-marginalized-principal" ||
      normalized == "principal") {
    return AutoCameraInitializationSelectionScorer::
        PoseMarginalizedPrincipal;
  }
  if (normalized == "legacy_fixed_pose" ||
      normalized == "legacy-fixed-pose" || normalized == "legacy") {
    return AutoCameraInitializationSelectionScorer::LegacyFixedPose;
  }
  throw std::runtime_error(
      "Unsupported --stage5-init-selection-scorer: " + value +
      " (supported: pose_marginalized_principal, legacy_fixed_pose)");
}

namespace {

constexpr double kInitializationPoseMaxRmsePx = 25.0;

struct OuterObservationRecord {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  double quality = 0.0;
  std::vector<Eigen::Vector3d> object_points;
  std::vector<cv::Point2f> image_points;
  bool used_direct_dense_control_points = false;
  // Propagated from OuterBoardMeasurement so camera initialization can keep
  // rescued corners visible, but evaluate/gate them separately.
  bool used_local_patch_rescue = false;
  std::string local_patch_rescue_summary;
  int internal_point_count = 0;
  std::array<cv::Point2f, 4> diagnostic_outer_image_points{};
};

struct SeedConstructionDiagnostics {
  std::string seed_method = "unknown";
  std::string seed_source = "unknown";
  double omni_gamma = std::numeric_limits<double>::quiet_NaN();
  std::string omni_gamma_source = "unavailable";
  std::string ds_mapping = "not_applicable";
  int ds_mapping_verified_against_kalibr_source = 0;
  int ds_grid_enumeration_enabled = 0;
  std::string ucm_seed_source = "not_applicable";
  std::string ucm_mapping = "not_applicable";
  int ucm_mapping_verified_against_kalibr_source = 0;
  int ucm_multistart_enabled = 0;
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

bool IsPinholeEquiDistortionLabel(const std::string& label) {
  return label == "k1" || label == "k2" || label == "k3" || label == "k4";
}

bool IsRadtanDistortionLabel(const std::string& label) {
  return label == "k1" || label == "k2" || label == "p1" || label == "p2";
}

bool IsCameraDistortionLabel(const std::string& label) {
  return IsPinholeEquiDistortionLabel(label) || IsRadtanDistortionLabel(label);
}

bool HasAnyMeaningfulDistortionCoefficient(
    const OuterBootstrapCameraIntrinsics& camera) {
  for (double coefficient : camera.DistortionVector()) {
    if (std::isfinite(coefficient) && std::abs(coefficient) > 1e-12) {
      return true;
    }
  }
  return false;
}

bool FixKalibrOuterLmInitializationLabel(const std::string& family,
                                         const std::string& label) {
  if (family == "pinhole-equi") {
    // The KB initializer must use the same eight-parameter camera model as
    // its Fisher scoring and downstream backend. Freezing k3/k4 makes focal
    // length and low-order terms compensate for real high-order distortion.
    return false;
  }
  if (family == "omni-radtan") {
    return IsCameraDistortionLabel(label);
  }
  return false;
}

std::vector<std::string> KalibrOuterLmReleasedLabels(
    const OuterBootstrapCameraIntrinsics& camera) {
  std::vector<std::string> labels = camera.CombinedParameterLabels();
  const std::string family = camera.NormalizedFamilyString();
  labels.erase(std::remove_if(labels.begin(),
                              labels.end(),
                              [&family](const std::string& label) {
                                return FixKalibrOuterLmInitializationLabel(
                                    family, label);
                              }),
               labels.end());
  return labels;
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
      coefficient = std::max(-0.6, std::min(0.6, coefficient));
    }
  } else if (family == "omni-radtan") {
    intrinsics->xi = std::max(-0.95, std::min(3.0, intrinsics->xi));
    intrinsics->alpha = 0.0;
    intrinsics->beta = 0.0;
    if (intrinsics->distortion_coeffs.size() != 4) {
      intrinsics->distortion_coeffs.resize(4, 0.0);
    }
    for (double& coefficient : intrinsics->distortion_coeffs) {
      coefficient = std::max(-0.6, std::min(0.6, coefficient));
    }
  } else if (family == "omni-none") {
    intrinsics->xi = std::max(-0.95, std::min(3.0, intrinsics->xi));
    intrinsics->alpha = 0.0;
    intrinsics->beta = 0.0;
    intrinsics->distortion_coeffs.clear();
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
          ? (intrinsics.camera_model == "pinhole"
                 ? "equi"
                 : (intrinsics.camera_model == "omni" ||
                    intrinsics.camera_model == "mei"
                        ? "radtan"
                        : "none"))
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
  } else if (intrinsics.NormalizedFamilyString() == "omni-radtan") {
    intrinsics.xi = 1.0;
    intrinsics.distortion_coeffs = {0.0, 0.0, 0.0, 0.0};
  } else if (intrinsics.NormalizedFamilyString() == "omni-none") {
    intrinsics.xi = 1.0;
    intrinsics.distortion_coeffs.clear();
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

double Quantile(std::vector<double> values, double probability) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const double clamped_probability =
      std::max(0.0, std::min(1.0, probability));
  const double position =
      clamped_probability * static_cast<double>(values.size() - 1u);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double alpha = position - static_cast<double>(lower);
  return (1.0 - alpha) * values[lower] + alpha * values[upper];
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

bool EstimateKalibrPinholeFocalFromOuterObservations(
    const cv::Size& image_size,
    const std::vector<OuterObservationRecord>& observations,
    double* focal,
    std::string* source_label,
    std::string* failure_reason) {
  if (focal == nullptr) {
    return false;
  }
  *focal = std::numeric_limits<double>::quiet_NaN();
  // Kalibr's pinhole-equi initializer computes a pinhole focal from target
  // row/column circles and initializes equidistant distortion to zero. Our
  // Stage5 bootstrap intentionally only has each large tag's outer four
  // corners, so the exact row-circle estimator is not observable here. Do not
  // substitute a planar homography focal and call it Kalibr-equivalent; that
  // path strongly favors high-focal single-board pose compensation on fisheye
  // data. Use a fisheye field-of-view focal prior as an explicit outer-only
  // fallback and report the fallback reason in the summary.
  const double fallback_focal = FisheyeResolutionFocalPrior(image_size);
  if (std::isfinite(fallback_focal) && fallback_focal > 0.0) {
    *focal = fallback_focal;
    if (source_label != nullptr) {
      *source_label = "outer_resolution_fisheye_focal_prior";
    }
    if (failure_reason != nullptr) {
      *failure_reason =
          "kalibr_pinhole_circle_focal_unavailable_for_outer_four_corner_tags";
    }
    return true;
  }
  if (failure_reason != nullptr) {
    *failure_reason = "invalid_image_size_for_fisheye_focal_prior";
  }
  return false;
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
  std::string pinhole_focal_source;
  std::string pinhole_focal_failure_reason;
  if (family == "pinhole-equi") {
    has_homography_focal =
        EstimateKalibrPinholeFocalFromOuterObservations(
            image_size,
            observations,
            &pinhole_focal,
            &pinhole_focal_source,
            &pinhole_focal_failure_reason);
    if (!has_homography_focal) {
      pinhole_focal =
          0.5 * static_cast<double>(std::max(image_size.width, image_size.height));
      pinhole_focal_source = "outer_resolution_half_extent_fallback";
    }
  } else if (family != "ds-none" && family != "omni-radtan" &&
             family != "omni-none" &&
             family != "eucm-none") {
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
  } else if (family == "omni-radtan" || family == "omni-none") {
    double omni_gamma = std::numeric_limits<double>::quiet_NaN();
    std::string omni_failure_reason;
    const bool has_omni_gamma =
        EstimateOmniGammaFromDirectOuterQuads(
            image_size, observations, &omni_gamma, &omni_failure_reason);
    seed.xi = 1.0;
    seed.alpha = 0.0;
    seed.beta = 0.0;
    seed.fu = has_omni_gamma ? omni_gamma : FisheyeResolutionFocalPrior(image_size);
    seed.fv = seed.fu;
    seed.distortion_coeffs =
        family == "omni-radtan" ? std::vector<double>{0.0, 0.0, 0.0, 0.0}
                                 : std::vector<double>{};
    if (source_label != nullptr) {
      *source_label = has_omni_gamma
                          ? "kalibr_omni_outer_direct_gamma_seed"
                          : (family == "omni-none"
                                 ? "fallback_outer_only_omni_none_seed"
                                 : "fallback_outer_only_omni_radtan_seed");
    }
    if (seed_diagnostics != nullptr) {
      seed_diagnostics->seed_method =
          has_omni_gamma ? "kalibr_omni_outer_direct_gamma_seed"
                         : (family == "omni-none"
                                ? "fallback_outer_only_omni_none_seed"
                                : "fallback_outer_only_omni_radtan_seed");
      seed_diagnostics->seed_source =
          has_omni_gamma ? "outer_quad_direct_omni_seed_observations"
                         : "resolution_fisheye_fov_prior";
      seed_diagnostics->omni_gamma =
          has_omni_gamma ? omni_gamma : std::numeric_limits<double>::quiet_NaN();
      seed_diagnostics->omni_gamma_source =
          has_omni_gamma ? "outer_quad_direct_four_corner_groups"
                         : "unavailable";
      seed_diagnostics->ucm_seed_source = seed_diagnostics->seed_source;
      seed_diagnostics->ucm_mapping =
          has_omni_gamma ? "omni_xi_1_fu_fv_omni_gamma"
                         : "fallback_no_omni_gamma";
      seed_diagnostics->ucm_mapping_verified_against_kalibr_source =
          has_omni_gamma ? 1 : 0;
      seed_diagnostics->fallback_reason = omni_failure_reason;
    }
    if (!has_omni_gamma) {
      AppendUniqueWarning(
          "Kalibr Omni initializer source was found, but Stage5 outer-only "
          "four-corner observations did not produce a valid direct gamma; "
          "using xi=1, no distortion for omni-none (or zero radtan for the "
          "explicit omni-radtan family), and a resolution/pi fisheye "
          "focal prior. fallback_reason=" +
              (omni_failure_reason.empty()
                   ? std::string("kalibr_omni_gamma_unavailable")
                   : omni_failure_reason),
          warnings);
    }
  } else if (family == "eucm-none") {
    double omni_gamma = std::numeric_limits<double>::quiet_NaN();
    std::string omni_failure_reason;
    const bool has_omni_gamma =
        EstimateOmniGammaFromDirectOuterQuads(
            image_size, observations, &omni_gamma, &omni_failure_reason);
    seed.alpha = 0.5;
    seed.beta = 1.0;
    seed.xi = 0.0;
    seed.fu = has_omni_gamma ? 0.5 * omni_gamma
                              : FisheyeResolutionFocalPrior(image_size);
    seed.fv = seed.fu;
    seed.distortion_coeffs.clear();
    if (source_label != nullptr) {
      *source_label = has_omni_gamma
                          ? "kalibr_source_verified_omni_to_eucm_seed"
                          : "fallback_outer_only_eucm_seed";
    }
    if (seed_diagnostics != nullptr) {
      seed_diagnostics->seed_method =
          has_omni_gamma ? "kalibr_source_verified_omni_to_eucm_seed"
                         : "fallback_outer_only_eucm_seed";
      seed_diagnostics->seed_source =
          has_omni_gamma ? "outer_quad_direct_omni_seed_observations"
                         : "resolution_fisheye_fov_prior";
      seed_diagnostics->omni_gamma =
          has_omni_gamma ? omni_gamma : std::numeric_limits<double>::quiet_NaN();
      seed_diagnostics->omni_gamma_source =
          has_omni_gamma ? "outer_quad_direct_four_corner_groups"
                         : "unavailable";
      seed_diagnostics->ucm_seed_source = seed_diagnostics->seed_source;
      seed_diagnostics->ucm_mapping =
          has_omni_gamma
              ? "alpha_0p5_beta_1_fu_fv_0p5_omni_gamma"
              : "fallback_no_omni_gamma";
      seed_diagnostics->ucm_mapping_verified_against_kalibr_source =
          has_omni_gamma ? 1 : 0;
      if (!has_omni_gamma) {
        seed_diagnostics->fallback_reason =
            omni_failure_reason.empty()
                ? "kalibr_omni_gamma_unavailable"
                : omni_failure_reason;
      }
    }
    if (!has_omni_gamma) {
      AppendUniqueWarning(
          "Kalibr EUCM initializer source was found and maps Omni to EUCM as "
          "alpha=0.5*omni.xi, beta=1, fu/fv=0.5*omni.fu/fv. Stage5 outer-only "
          "four-corner observations did not produce a valid direct Omni gamma, "
          "so the EUCM seed uses alpha=0.5, beta=1 and a resolution/pi fisheye "
          "focal prior. No pinhole-homography focal is mapped into EUCM. "
          "fallback_reason=" +
              (omni_failure_reason.empty()
                   ? std::string("kalibr_omni_gamma_unavailable")
                   : omni_failure_reason),
          warnings);
    }
  } else if (family == "pinhole-equi") {
    seed.xi = 0.0;
    seed.alpha = 0.0;
    seed.beta = 0.0;
    seed.fu = pinhole_focal;
    seed.fv = pinhole_focal;
    seed.distortion_coeffs = {0.0, 0.0, 0.0, 0.0};
    if (source_label != nullptr) {
      *source_label = "kalibr_pinhole_equi_outer_fisheye_prior_zero_distortion_seed";
    }
    if (seed_diagnostics != nullptr) {
      seed_diagnostics->seed_method =
          "kalibr_pinhole_equi_zero_distortion_seed";
      seed_diagnostics->seed_source = pinhole_focal_source.empty()
                                          ? "outer_fisheye_prior"
                                          : pinhole_focal_source;
      seed_diagnostics->fallback_reason = pinhole_focal_failure_reason;
    }
    if (!pinhole_focal_failure_reason.empty()) {
      AppendUniqueWarning(
	          "Kalibr pinhole-equi row-circle focal initialization cannot be "
	          "exactly applied to Stage5 outer-four-corner observations; using a "
	          "reported outer-only fisheye focal fallback. The initialization "
	          "candidate set uses a compact focal/k1 multistart; the lightweight "
	          "outer-only LM releases k1/k2 and keeps k3/k4 for the later backend, "
	          "where internal points and selection provide more evidence. "
	          "fallback_reason=" +
	              pinhole_focal_failure_reason,
	          warnings);
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
    const ApriltagInternalConfig& config,
    bool use_direct_dense_control_points = false,
    const std::string& direct_dense_control_point_scope = "all") {
  if (direct_dense_control_point_scope != "all" &&
      direct_dense_control_point_scope != "outer_only" &&
      direct_dense_control_point_scope != "internal_only") {
    throw std::runtime_error(
        "Unsupported direct dense control-point scope: " +
        direct_dense_control_point_scope);
  }
  std::vector<OuterObservationRecord> observations;
  for (const OuterBootstrapFrameInput& frame : frames) {
    for (const OuterBoardMeasurement& measurement : frame.measurements.board_measurements) {
      const bool dense_storage_valid =
          use_direct_dense_control_points &&
          measurement.has_direct_dense_control_points &&
          measurement.direct_dense_target_points_board.size() ==
              measurement.direct_dense_image_points.size();
      const bool dense_type_metadata_valid =
          measurement.direct_dense_point_is_outer.size() ==
          measurement.direct_dense_image_points.size();
      std::vector<std::size_t> dense_point_indices;
      if (dense_storage_valid &&
          (direct_dense_control_point_scope == "all" ||
           dense_type_metadata_valid)) {
        dense_point_indices.reserve(
            measurement.direct_dense_image_points.size());
        for (std::size_t point_index = 0;
             point_index < measurement.direct_dense_image_points.size();
             ++point_index) {
          const bool is_outer =
              dense_type_metadata_valid &&
              measurement.direct_dense_point_is_outer[point_index] != 0u;
          if (direct_dense_control_point_scope == "outer_only" && !is_outer) {
            continue;
          }
          if (direct_dense_control_point_scope == "internal_only" && is_outer) {
            continue;
          }
          dense_point_indices.push_back(point_index);
        }
      }
      const bool has_dense_control_points = dense_point_indices.size() >= 4u;
      if (!IsValidOuterMeasurement(measurement) &&
          !has_dense_control_points) {
        continue;
      }
      OuterObservationRecord observation;
      observation.frame_index = frame.frame_index;
      observation.frame_label = frame.frame_label;
      observation.board_id = measurement.board_id;
      observation.quality = measurement.detection_quality;
      observation.used_local_patch_rescue =
          measurement.used_local_patch_rescue;
      observation.local_patch_rescue_summary =
          measurement.local_patch_rescue_summary;
      for (int index = 0; index < 4; ++index) {
        if (!measurement.refined_corner_valid[
                static_cast<std::size_t>(index)]) {
          continue;
        }
        const Eigen::Vector2d& point =
            measurement.refined_outer_corners_original_image[
                static_cast<std::size_t>(index)];
        observation.diagnostic_outer_image_points[
            static_cast<std::size_t>(index)] =
            cv::Point2f(static_cast<float>(point.x()),
                        static_cast<float>(point.y()));
      }
      if (has_dense_control_points) {
        observation.used_direct_dense_control_points = true;
        observation.object_points.reserve(dense_point_indices.size());
        observation.image_points.reserve(dense_point_indices.size());
        for (const std::size_t point_index : dense_point_indices) {
          observation.object_points.push_back(
              measurement.direct_dense_target_points_board[point_index]);
          const Eigen::Vector2d& point =
              measurement.direct_dense_image_points[point_index];
          observation.image_points.emplace_back(
              static_cast<float>(point.x()), static_cast<float>(point.y()));
          const bool is_outer =
              dense_type_metadata_valid &&
              measurement.direct_dense_point_is_outer[point_index] != 0u;
          if (!is_outer) {
            ++observation.internal_point_count;
          }
        }
        if (!dense_type_metadata_valid) {
          observation.internal_point_count = std::max(
              0, static_cast<int>(dense_point_indices.size()) - 4);
        }
      } else {
        const std::array<Eigen::Vector3d, 4> object_points =
            measurement.has_target_outer_corners
                ? measurement.target_outer_corners_board
                : BuildOuterCornerPoints(config, measurement.board_id);
        observation.object_points.assign(object_points.begin(),
                                         object_points.end());
        observation.image_points.reserve(4);
        for (int index = 0; index < 4; ++index) {
          const Eigen::Vector2d& point =
              measurement.refined_outer_corners_original_image[
                  static_cast<std::size_t>(index)];
          observation.image_points.emplace_back(
              static_cast<float>(point.x()), static_cast<float>(point.y()));
        }
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

std::set<int> ConfiguredBoardIds(const ApriltagInternalConfig& config) {
  std::set<int> board_ids(config.tag_ids.begin(), config.tag_ids.end());
  if (board_ids.empty()) {
    board_ids.insert(config.tag_id);
  }
  return board_ids;
}

std::string FormatBoardIds(const std::set<int>& board_ids) {
  std::ostringstream stream;
  bool first = true;
  for (int board_id : board_ids) {
    if (!first) {
      stream << ",";
    }
    stream << board_id;
    first = false;
  }
  return stream.str();
}

std::set<int> CompleteBoardFrameIndices(
    const std::vector<OuterObservationRecord>& outer_observations,
    const std::set<int>& required_board_ids) {
  std::map<int, std::set<int>> observed_boards_by_frame;
  for (const OuterObservationRecord& observation : outer_observations) {
    observed_boards_by_frame[observation.frame_index].insert(
        observation.board_id);
  }
  std::set<int> complete_frame_indices;
  for (const auto& frame_entry : observed_boards_by_frame) {
    const std::set<int>& observed_board_ids = frame_entry.second;
    if (std::includes(observed_board_ids.begin(), observed_board_ids.end(),
                      required_board_ids.begin(), required_board_ids.end())) {
      complete_frame_indices.insert(frame_entry.first);
    }
  }
  return complete_frame_indices;
}

std::vector<OuterObservationRecord> FilterObservationsByFrameIndices(
    const std::vector<OuterObservationRecord>& observations,
    const std::set<int>& accepted_frame_indices) {
  std::vector<OuterObservationRecord> filtered;
  filtered.reserve(observations.size());
  for (const OuterObservationRecord& observation : observations) {
    if (accepted_frame_indices.count(observation.frame_index) != 0u) {
      filtered.push_back(observation);
    }
  }
  return filtered;
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
    const std::vector<OuterObservationRecord>& observations,
    double rescued_observation_weight = 1.0) {
  AutoCameraInitializationCandidate candidate;
  candidate.source_label = source_label;
  candidate.evaluation_scope = evaluation_scope;
  candidate.camera = camera;
  candidate.observation_count = static_cast<int>(observations.size());

  if (!candidate.camera.IsValid()) {
    candidate.failure_reason = "candidate intrinsics are invalid";
    return candidate;
  }

  double total_weighted_squared_rmse = 0.0;
  double successful_observation_weight = 0.0;
  std::vector<double> successful_observation_rmses;
  successful_observation_rmses.reserve(observations.size());
  double max_observation_rmse = -std::numeric_limits<double>::infinity();
  int worst_observation_frame_index = -1;
  int worst_observation_board_id = -1;
  double robust_loss_sum = 0.0;
  constexpr double kOuterHealthHuberDeltaPx = 3.0;
  const auto huber_loss = [](double residual, double delta) {
    const double absolute_residual = std::abs(residual);
    if (absolute_residual <= delta) {
      return 0.5 * residual * residual;
    }
    return delta * (absolute_residual - 0.5 * delta);
  };
  std::set<int> successful_frames;
  std::set<int> successful_boards;
  std::map<int, std::vector<std::pair<int, Eigen::Isometry3d>>>
      successful_poses_by_frame;
  for (const OuterObservationRecord& observation : observations) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double observation_rmse = 0.0;
    if (!EstimatePoseFromObjectPointsStrict(
            candidate.camera, observation.object_points,
            observation.image_points, kInitializationPoseMaxRmsePx, &pose,
            &observation_rmse)) {
      ++candidate.pose_failure_count;
      continue;
    }
    ++candidate.pose_success_count;
    const double observation_weight =
        observation.used_local_patch_rescue
            ? std::max(0.0, rescued_observation_weight)
            : 1.0;
    total_weighted_squared_rmse +=
        observation_weight * observation_rmse * observation_rmse;
    robust_loss_sum += observation_weight *
                       huber_loss(observation_rmse,
                                  kOuterHealthHuberDeltaPx);
    successful_observation_weight += observation_weight;
    successful_observation_rmses.push_back(observation_rmse);
    if (observation_rmse > max_observation_rmse) {
      max_observation_rmse = observation_rmse;
      worst_observation_frame_index = observation.frame_index;
      worst_observation_board_id = observation.board_id;
    }
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
  if (candidate.pose_success_count > 0 && successful_observation_weight > 0.0) {
    candidate.mean_observation_rmse =
        std::sqrt(total_weighted_squared_rmse /
                  successful_observation_weight);
    candidate.robust_observation_rmse = std::sqrt(
        2.0 * robust_loss_sum / successful_observation_weight);
    std::sort(successful_observation_rmses.begin(),
              successful_observation_rmses.end());
    const std::size_t median_index =
        successful_observation_rmses.size() / 2u;
    candidate.median_observation_rmse =
        successful_observation_rmses[median_index];
    const std::size_t p95_index = std::min(
        successful_observation_rmses.size() - 1u,
        static_cast<std::size_t>(std::ceil(
            0.95 * static_cast<double>(successful_observation_rmses.size()))) -
            1u);
    candidate.p95_observation_rmse = successful_observation_rmses[p95_index];
    candidate.max_observation_rmse = max_observation_rmse;
    candidate.worst_observation_frame_index = worst_observation_frame_index;
    candidate.worst_observation_board_id = worst_observation_board_id;
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
  if (lhs.projection_failure_count != rhs.projection_failure_count) {
    return lhs.projection_failure_count < rhs.projection_failure_count;
  }
  if (std::abs(lhs.success_rate - rhs.success_rate) > 1e-12) {
    return lhs.success_rate > rhs.success_rate;
  }
  const bool lhs_p95_finite = std::isfinite(lhs.p95_observation_rmse);
  const bool rhs_p95_finite = std::isfinite(rhs.p95_observation_rmse);
  if (lhs_p95_finite != rhs_p95_finite) {
    return lhs_p95_finite;
  }
  if (lhs_p95_finite &&
      std::abs(lhs.p95_observation_rmse - rhs.p95_observation_rmse) > 1e-12) {
    return lhs.p95_observation_rmse < rhs.p95_observation_rmse;
  }
  if (std::abs(lhs.robust_observation_rmse - rhs.robust_observation_rmse) >
      1e-12) {
    return lhs.robust_observation_rmse < rhs.robust_observation_rmse;
  }
  if (std::abs(lhs.median_observation_rmse - rhs.median_observation_rmse) >
      1e-12) {
    return lhs.median_observation_rmse < rhs.median_observation_rmse;
  }
  if (lhs.successful_frame_count != rhs.successful_frame_count) {
    return lhs.successful_frame_count > rhs.successful_frame_count;
  }
  return lhs.successful_board_count > rhs.successful_board_count;
}

// Use the same independent Outer4 observation set to compare every refined
// basin.  Projection validity is a hard feasibility condition.  Among
// feasible basins, residual quality comes first: a handful of difficult
// frame-board poses must not select a globally worse camera basin merely
// because that basin reports more local pose initializations.
bool RefinedOuterEvaluationIsBetter(
    const AutoCameraInitializationCandidate& candidate,
    double candidate_lm_objective,
    const AutoCameraInitializationCandidate& incumbent,
    double incumbent_lm_objective) {
  if (candidate.projection_failure_count !=
      incumbent.projection_failure_count) {
    return candidate.projection_failure_count <
           incumbent.projection_failure_count;
  }
  const bool candidate_p95_finite =
      std::isfinite(candidate.p95_observation_rmse);
  const bool incumbent_p95_finite =
      std::isfinite(incumbent.p95_observation_rmse);
  if (candidate_p95_finite != incumbent_p95_finite) {
    return candidate_p95_finite;
  }
  if (candidate_p95_finite &&
      std::abs(candidate.p95_observation_rmse -
               incumbent.p95_observation_rmse) > 1e-12) {
    return candidate.p95_observation_rmse < incumbent.p95_observation_rmse;
  }
  if (std::abs(candidate.robust_observation_rmse -
               incumbent.robust_observation_rmse) > 1e-12) {
    return candidate.robust_observation_rmse <
           incumbent.robust_observation_rmse;
  }
  if (std::abs(candidate.median_observation_rmse -
               incumbent.median_observation_rmse) > 1e-12) {
    return candidate.median_observation_rmse <
           incumbent.median_observation_rmse;
  }
  if (candidate.pose_success_count != incumbent.pose_success_count) {
    return candidate.pose_success_count > incumbent.pose_success_count;
  }
  if (std::abs(candidate_lm_objective - incumbent_lm_objective) > 1e-12) {
    return candidate_lm_objective < incumbent_lm_objective;
  }
  return candidate.source_label < incumbent.source_label;
}

double CandidateObjective(const AutoCameraInitializationCandidate& candidate) {
  if (!candidate.valid || candidate.observation_count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  const double robust_rmse = candidate.robust_observation_rmse;
  const double rmse = candidate.mean_observation_rmse;
  const double effective_rmse =
      std::isfinite(robust_rmse) ? robust_rmse : rmse;
  return effective_rmse * effective_rmse;
}

bool IsAcceptableAutoCandidate(const AutoCameraInitializationCandidate& candidate,
                               int total_observation_count) {
  const int min_success = std::min(total_observation_count, 6);
  const double min_success_rate = total_observation_count >= 12 ? 0.4 : 0.25;
  return candidate.valid &&
         candidate.pose_success_count >= min_success &&
         candidate.success_rate >= min_success_rate &&
         std::isfinite(candidate.robust_observation_rmse) &&
         candidate.robust_observation_rmse < 20.0;
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
         std::isfinite(candidate.robust_observation_rmse) &&
         candidate.robust_observation_rmse < 150.0;
}

bool IsPhysicallyPlausibleOmniNoneSeed(
    const OuterBootstrapCameraIntrinsics& camera) {
  if (camera.NormalizedFamilyString() != "omni-none" ||
      camera.resolution.width <= 0 || camera.resolution.height <= 0 ||
      !(camera.xi > -0.9 && camera.xi < 5.0)) {
    return false;
  }
  const double denominator = 1.0 + camera.xi;
  if (!(denominator > 0.1)) {
    return false;
  }
  const double extent = static_cast<double>(
      std::max(camera.resolution.width, camera.resolution.height));
  const double equivalent_fu = camera.fu / denominator;
  const double equivalent_fv = camera.fv / denominator;
  return std::isfinite(equivalent_fu) && std::isfinite(equivalent_fv) &&
         equivalent_fu >= 0.18 * extent && equivalent_fu <= 0.55 * extent &&
         equivalent_fv >= 0.18 * extent && equivalent_fv <= 0.55 * extent;
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
    const std::vector<OuterObservationRecord>& observations,
    double rescued_observation_weight) {
  OuterBootstrapCameraIntrinsics best = initial_camera;
  AutoCameraInitializationCandidate best_eval =
      EvaluateCandidateOnObservations(best, "auto_grid_refined", "full",
                                      observations, rescued_observation_weight);
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
            EvaluateCandidateOnObservations(candidate, "auto_grid_refined", "full",
                                            observations,
                                            rescued_observation_weight);
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
  double observation_weight = 1.0;
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
  int downweighted_point_count = 0;
  double rmse = std::numeric_limits<double>::infinity();
  double robust_cost = std::numeric_limits<double>::infinity();
  double robust_rmse = std::numeric_limits<double>::infinity();
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
  double robust_loss_delta_pixels = 0.0;
  double initial_robust_rmse = std::numeric_limits<double>::infinity();
  double final_robust_rmse = std::numeric_limits<double>::infinity();
  double initial_robust_cost = std::numeric_limits<double>::infinity();
  double final_robust_cost = std::numeric_limits<double>::infinity();
  int initial_downweighted_point_count = 0;
  int final_downweighted_point_count = 0;
  bool improved = false;
};

struct PoseFitOutlierGateResult {
  std::vector<OuterObservationRecord> accepted_observations;
  int pose_failure_count = 0;
  int rejected_outlier_count = 0;
  int rescued_observation_count = 0;
  int rescued_pose_gate_rejected_count = 0;
  bool gate_applied = false;
  double median_rmse = std::numeric_limits<double>::quiet_NaN();
  double mad_rmse = std::numeric_limits<double>::quiet_NaN();
  double threshold_rmse = std::numeric_limits<double>::quiet_NaN();
};

PoseFitOutlierGateResult FilterPoseFitOutliersForInitialization(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<OuterObservationRecord>& observations,
    bool gate_rescued_observations = true,
    double rescued_observation_pose_rmse_gate_pixels = 8.0) {
  PoseFitOutlierGateResult result;
  struct SuccessfulPoseFit {
    const OuterObservationRecord* observation = nullptr;
    double rmse = std::numeric_limits<double>::infinity();
  };
  std::vector<SuccessfulPoseFit> successful;
  successful.reserve(observations.size());
  std::vector<double> errors;
  errors.reserve(observations.size());
  for (const OuterObservationRecord& observation : observations) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double rmse = std::numeric_limits<double>::infinity();
    if (!EstimatePoseFromObjectPointsStrict(
            camera, observation.object_points, observation.image_points,
            kInitializationPoseMaxRmsePx, &pose, &rmse) ||
        !std::isfinite(rmse)) {
      ++result.pose_failure_count;
      continue;
    }
    if (observation.used_local_patch_rescue) {
      ++result.rescued_observation_count;
      if (gate_rescued_observations &&
          rescued_observation_pose_rmse_gate_pixels > 0.0 &&
          rmse > rescued_observation_pose_rmse_gate_pixels) {
        // A patch rescue is camera-aware and therefore has a different
        // failure mode from an ordinary subpixel corner.  Do not let one
        // badly mapped rescue corner pull the camera basin; reject it with a
        // dedicated gate before the generic MAD gate is computed.
        ++result.rescued_pose_gate_rejected_count;
        result.gate_applied = true;
        continue;
      }
    }
    successful.push_back(SuccessfulPoseFit{&observation, rmse});
    errors.push_back(rmse);
  }
  if (successful.empty()) {
    return result;
  }

  result.median_rmse = Median(errors);
  std::vector<double> absolute_deviations;
  absolute_deviations.reserve(errors.size());
  for (double error : errors) {
    absolute_deviations.push_back(std::abs(error - result.median_rmse));
  }
  result.mad_rmse = Median(absolute_deviations);
  result.threshold_rmse = std::max(
      8.0,
      result.median_rmse +
          std::max(1.0, 8.0 * 1.4826 * result.mad_rmse));

  int candidate_outlier_count = 0;
  for (const SuccessfulPoseFit& fit : successful) {
    if (fit.rmse > result.threshold_rmse) {
      ++candidate_outlier_count;
    }
  }
  const int max_isolated_outlier_count = std::max(
      2, static_cast<int>(std::floor(0.10 * successful.size())));
  result.gate_applied = candidate_outlier_count > 0 &&
                        candidate_outlier_count <= max_isolated_outlier_count &&
                        static_cast<int>(successful.size()) -
                                candidate_outlier_count >=
                            4;
  result.accepted_observations.reserve(successful.size());
  for (const SuccessfulPoseFit& fit : successful) {
    if (result.gate_applied && fit.rmse > result.threshold_rmse) {
      ++result.rejected_outlier_count;
      continue;
    }
    result.accepted_observations.push_back(*fit.observation);
  }
  return result;
}

double PointNormHuberWeight(double dx, double dy, double delta_pixels) {
  if (!(delta_pixels > 0.0) || !std::isfinite(delta_pixels)) {
    return 1.0;
  }
  const double norm = std::hypot(dx, dy);
  if (!std::isfinite(norm) || norm <= delta_pixels || norm <= 1e-12) {
    return 1.0;
  }
  return delta_pixels / norm;
}

double PointNormHuberCost(double dx, double dy, double delta_pixels) {
  const double squared_norm = dx * dx + dy * dy;
  if (!(delta_pixels > 0.0) || !std::isfinite(delta_pixels)) {
    return squared_norm;
  }
  const double norm = std::sqrt(std::max(0.0, squared_norm));
  if (norm <= delta_pixels) {
    return squared_norm;
  }
  return 2.0 * delta_pixels * norm - delta_pixels * delta_pixels;
}

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
    const Eigen::Isometry3d& T_camera_board,
    std::vector<unsigned char>* valid_points = nullptr) {
  Eigen::VectorXd residuals =
      Eigen::VectorXd::Constant(
          2 * static_cast<int>(observation.object_points.size()), 100.0);
  if (valid_points != nullptr) {
    valid_points->assign(observation.object_points.size(), 0u);
  }
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
    if (valid_points != nullptr) {
      (*valid_points)[point_index] = 1u;
    }
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

struct CameraInformationResult {
  Eigen::MatrixXd information;
  int pose_rank = -1;
  bool success = false;
};

struct SelectionInformationDiagnostics {
  int camera_rank = -1;
  int principal_rank = -1;
  int pose_rank_min = -1;
  int pose_rank_max = -1;
  int pose_rank_deficient_count = 0;
  double principal_min_eigenvalue = -1.0;
  double principal_max_eigenvalue = -1.0;
  double cu_stddev_px = -1.0;
  double cv_stddev_px = -1.0;
  double weakest_eigenvalue = -1.0;
  Eigen::VectorXd weakest_direction;
  double weakest_principal_fraction = -1.0;
  double weakest_focal_fraction = -1.0;
};

struct SymmetricInformationAnalysis {
  int rank = 0;
  double log_pseudodeterminant = 0.0;
  double min_positive_eigenvalue = -1.0;
  double max_eigenvalue = -1.0;
  Eigen::MatrixXd pseudoinverse;
  double weakest_eigenvalue = -1.0;
  Eigen::VectorXd weakest_direction;
};

SymmetricInformationAnalysis AnalyzeSymmetricInformation(
    const Eigen::MatrixXd& input) {
  SymmetricInformationAnalysis result;
  if (input.rows() <= 0 || input.rows() != input.cols() ||
      !input.allFinite()) {
    return result;
  }
  const Eigen::MatrixXd symmetric = 0.5 * (input + input.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);
  if (solver.info() != Eigen::Success) {
    return result;
  }
  const Eigen::VectorXd eigenvalues = solver.eigenvalues();
  if (eigenvalues.size() > 0) {
    result.weakest_eigenvalue = eigenvalues[0];
    result.weakest_direction = solver.eigenvectors().col(0);
    Eigen::Index dominant_index = 0;
    result.weakest_direction.cwiseAbs().maxCoeff(&dominant_index);
    if (result.weakest_direction[dominant_index] < 0.0) {
      result.weakest_direction *= -1.0;
    }
  }
  const double maximum =
      eigenvalues.size() > 0 ? std::max(0.0, eigenvalues.maxCoeff()) : 0.0;
  // This matrix is J^T J, so Kalibr's relative SVD threshold of 1e-6 on J
  // corresponds to approximately 1e-12 on its eigenvalues.
  const double tolerance = std::max(1e-15, 1e-12 * maximum);
  Eigen::VectorXd inverse_values = Eigen::VectorXd::Zero(eigenvalues.size());
  for (Eigen::Index index = 0; index < eigenvalues.size(); ++index) {
    const double value = eigenvalues[index];
    if (!std::isfinite(value) || value <= tolerance) {
      continue;
    }
    inverse_values[index] = 1.0 / value;
    ++result.rank;
    result.log_pseudodeterminant += std::log(value);
    if (result.min_positive_eigenvalue < 0.0) {
      result.min_positive_eigenvalue = value;
    }
    result.max_eigenvalue = std::max(result.max_eigenvalue, value);
  }
  result.pseudoinverse =
      solver.eigenvectors() * inverse_values.asDiagonal() *
      solver.eigenvectors().transpose();
  return result;
}

std::vector<double> CameraInformationParameterScales(
    const OuterBootstrapCameraIntrinsics& camera) {
  const std::vector<std::string> labels = camera.CombinedParameterLabels();
  const double image_scale = std::max(
      1.0, static_cast<double>(
               std::max(camera.resolution.width, camera.resolution.height)));
  std::vector<double> scales(labels.size(), 1.0);
  for (std::size_t index = 0; index < labels.size(); ++index) {
    if (labels[index] == "fu" || labels[index] == "fv" ||
        labels[index] == "cu" || labels[index] == "cv") {
      scales[index] = image_scale;
    }
  }
  return scales;
}

Eigen::MatrixXd ScaledCameraInformation(
    const OuterBootstrapCameraIntrinsics& camera,
    const Eigen::MatrixXd& information) {
  const std::vector<double> scales = CameraInformationParameterScales(camera);
  if (information.rows() != static_cast<int>(scales.size()) ||
      information.cols() != static_cast<int>(scales.size())) {
    return Eigen::MatrixXd();
  }
  Eigen::VectorXd scale_vector(static_cast<int>(scales.size()));
  for (int index = 0; index < scale_vector.rows(); ++index) {
    scale_vector[index] = scales[static_cast<std::size_t>(index)];
  }
  return scale_vector.asDiagonal() * information *
         scale_vector.asDiagonal();
}

Eigen::MatrixXd PrincipalMarginalInformation(
    const OuterBootstrapCameraIntrinsics& camera,
    const Eigen::MatrixXd& scaled_information) {
  const std::vector<std::string> labels = camera.CombinedParameterLabels();
  std::vector<int> principal_indices;
  std::vector<int> nuisance_indices;
  for (int index = 0; index < static_cast<int>(labels.size()); ++index) {
    if (labels[static_cast<std::size_t>(index)] == "cu" ||
        labels[static_cast<std::size_t>(index)] == "cv") {
      principal_indices.push_back(index);
    } else {
      nuisance_indices.push_back(index);
    }
  }
  if (principal_indices.size() != 2u ||
      scaled_information.rows() != static_cast<int>(labels.size()) ||
      scaled_information.cols() != static_cast<int>(labels.size())) {
    return Eigen::MatrixXd();
  }
  Eigen::Matrix2d Hpp = Eigen::Matrix2d::Zero();
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      Hpp(row, col) = scaled_information(principal_indices[row],
                                         principal_indices[col]);
    }
  }
  if (nuisance_indices.empty()) {
    return 0.5 * (Hpp + Hpp.transpose());
  }
  Eigen::MatrixXd Hnn(nuisance_indices.size(), nuisance_indices.size());
  Eigen::MatrixXd Hpn(2, nuisance_indices.size());
  for (int row = 0; row < static_cast<int>(nuisance_indices.size()); ++row) {
    for (int col = 0; col < static_cast<int>(nuisance_indices.size()); ++col) {
      Hnn(row, col) = scaled_information(nuisance_indices[row],
                                         nuisance_indices[col]);
    }
    for (int principal = 0; principal < 2; ++principal) {
      Hpn(principal, row) =
          scaled_information(principal_indices[principal],
                             nuisance_indices[row]);
    }
  }
  const SymmetricInformationAnalysis nuisance =
      AnalyzeSymmetricInformation(Hnn);
  Eigen::Matrix2d marginal =
      Hpp - Hpn * nuisance.pseudoinverse * Hpn.transpose();
  marginal = 0.5 * (marginal + marginal.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(marginal);
  if (solver.info() != Eigen::Success) {
    return Eigen::MatrixXd();
  }
  Eigen::Vector2d clamped = solver.eigenvalues().cwiseMax(0.0);
  return solver.eigenvectors() * clamped.asDiagonal() *
         solver.eigenvectors().transpose();
}

double PrincipalAwareInformationScore(
    const OuterBootstrapCameraIntrinsics& camera,
    const Eigen::MatrixXd& information,
    SelectionInformationDiagnostics* diagnostics = nullptr) {
  const Eigen::MatrixXd scaled = ScaledCameraInformation(camera, information);
  const SymmetricInformationAnalysis camera_analysis =
      AnalyzeSymmetricInformation(scaled);
  const Eigen::MatrixXd principal =
      PrincipalMarginalInformation(camera, scaled);
  const SymmetricInformationAnalysis principal_analysis =
      AnalyzeSymmetricInformation(principal);
  if (diagnostics != nullptr) {
    diagnostics->camera_rank = camera_analysis.rank;
    diagnostics->principal_rank = principal_analysis.rank;
    diagnostics->principal_min_eigenvalue =
        principal_analysis.min_positive_eigenvalue;
    diagnostics->principal_max_eigenvalue = principal_analysis.max_eigenvalue;
    diagnostics->weakest_eigenvalue = camera_analysis.weakest_eigenvalue;
    diagnostics->weakest_direction = camera_analysis.weakest_direction;
    const std::vector<std::string> camera_labels =
        camera.CombinedParameterLabels();
    double principal_squared_norm = 0.0;
    double focal_squared_norm = 0.0;
    if (camera_analysis.weakest_direction.size() ==
        static_cast<int>(camera_labels.size())) {
      for (int index = 0; index < static_cast<int>(camera_labels.size());
           ++index) {
        const double component = camera_analysis.weakest_direction[index];
        const std::string& label = camera_labels[static_cast<std::size_t>(index)];
        if (label == "cu" || label == "cv") {
          principal_squared_norm += component * component;
        } else if (label == "fu" || label == "fv") {
          focal_squared_norm += component * component;
        }
      }
      diagnostics->weakest_principal_fraction =
          std::sqrt(std::max(0.0, principal_squared_norm));
      diagnostics->weakest_focal_fraction =
          std::sqrt(std::max(0.0, focal_squared_norm));
    }

    if (principal_analysis.rank == 2 &&
        principal_analysis.pseudoinverse.rows() == 2) {
      const double image_scale = std::max(
          1.0, static_cast<double>(std::max(camera.resolution.width,
                                            camera.resolution.height)));
      const double cu_variance = principal_analysis.pseudoinverse(0, 0);
      const double cv_variance = principal_analysis.pseudoinverse(1, 1);
      if (std::isfinite(cu_variance) && cu_variance >= 0.0) {
        diagnostics->cu_stddev_px =
            image_scale * std::sqrt(cu_variance);
      }
      if (std::isfinite(cv_variance) && cv_variance >= 0.0) {
        diagnostics->cv_stddev_px =
            image_scale * std::sqrt(cv_variance);
      }
    }

    const SymmetricInformationAnalysis raw_analysis =
        AnalyzeSymmetricInformation(information);
    const std::vector<std::string> labels = camera.CombinedParameterLabels();
    if ((diagnostics->cu_stddev_px < 0.0 ||
         diagnostics->cv_stddev_px < 0.0) &&
        raw_analysis.rank == information.rows() &&
        raw_analysis.pseudoinverse.rows() == information.rows()) {
      for (int index = 0; index < static_cast<int>(labels.size()); ++index) {
        const double variance = raw_analysis.pseudoinverse(index, index);
        if (!std::isfinite(variance) || variance < 0.0) {
          continue;
        }
        if (labels[static_cast<std::size_t>(index)] == "cu") {
          diagnostics->cu_stddev_px = std::sqrt(variance);
        } else if (labels[static_cast<std::size_t>(index)] == "cv") {
          diagnostics->cv_stddev_px = std::sqrt(variance);
        }
      }
    }
  }
  return 30.0 * static_cast<double>(camera_analysis.rank) +
         20.0 * static_cast<double>(principal_analysis.rank) +
         0.05 * camera_analysis.log_pseudodeterminant +
         principal_analysis.log_pseudodeterminant;
}

std::size_t ChooseSelectionWorkerCount(std::size_t candidate_count) {
  if (candidate_count == 0) {
    return 0;
  }
  const unsigned int hardware_workers = std::thread::hardware_concurrency();
  const std::size_t usable_workers =
      hardware_workers > 1 ? static_cast<std::size_t>(hardware_workers - 1) : 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(candidate_count,
                               std::min<std::size_t>(usable_workers, 8)));
}

CameraInformationResult ComputeCameraInformationForObservation(
    const OuterBootstrapCameraIntrinsics& camera,
    const OuterObservationRecord& observation,
    const Eigen::Isometry3d& T_camera_board,
    AutoCameraInitializationSelectionScorer scorer) {
  CameraInformationResult result;
  const std::vector<double> parameters = camera.CombinedParameterVector();
  const std::vector<std::string> labels = camera.CombinedParameterLabels();
  const int parameter_count = static_cast<int>(parameters.size());
  Eigen::MatrixXd camera_jacobian(
      2 * static_cast<int>(observation.object_points.size()),
      parameter_count);
  camera_jacobian.setZero();
  std::vector<unsigned char> base_valid_points;
  const Eigen::VectorXd base_residual = BuildSingleObservationResidual(
      camera, observation, T_camera_board, &base_valid_points);
  if (base_residual.rows() != camera_jacobian.rows() ||
      !base_residual.allFinite()) {
    return result;
  }
  const Eigen::VectorXd x = ToEigenVector(parameters);
  for (int column = 0; column < parameter_count; ++column) {
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
    std::vector<unsigned char> plus_valid_points;
    std::vector<unsigned char> minus_valid_points;
    const Eigen::VectorXd plus_residual = BuildSingleObservationResidual(
        plus_camera, observation, T_camera_board, &plus_valid_points);
    const Eigen::VectorXd minus_residual = BuildSingleObservationResidual(
        minus_camera, observation, T_camera_board, &minus_valid_points);
    if (plus_residual.rows() != camera_jacobian.rows() ||
        minus_residual.rows() != camera_jacobian.rows()) {
      continue;
    }
    for (std::size_t point_index = 0;
         point_index < observation.object_points.size(); ++point_index) {
      if (point_index >= base_valid_points.size() ||
          point_index >= plus_valid_points.size() ||
          point_index >= minus_valid_points.size() ||
          base_valid_points[point_index] == 0u ||
          plus_valid_points[point_index] == 0u ||
          minus_valid_points[point_index] == 0u) {
        continue;
      }
      const int row = 2 * static_cast<int>(point_index);
      camera_jacobian.block<2, 1>(row, column) =
          (plus_residual.segment<2>(row) - minus_residual.segment<2>(row)) /
          (2.0 * step);
    }
  }

  if (scorer == AutoCameraInitializationSelectionScorer::LegacyFixedPose) {
    result.information = camera_jacobian.transpose() * camera_jacobian;
    result.success = result.information.allFinite();
    return result;
  }

  Eigen::Matrix<double, Eigen::Dynamic, 6> pose_jacobian(
      camera_jacobian.rows(), 6);
  pose_jacobian.setZero();
  const Eigen::Matrix<double, 6, 1> pose_vector =
      PoseToVector(T_camera_board);
  for (int column = 0; column < 6; ++column) {
    const double step = column < 3 ? 1e-6 : 1e-7;
    Eigen::Matrix<double, 6, 1> plus_pose = pose_vector;
    Eigen::Matrix<double, 6, 1> minus_pose = pose_vector;
    plus_pose[column] += step;
    minus_pose[column] -= step;
    std::vector<unsigned char> plus_valid_points;
    std::vector<unsigned char> minus_valid_points;
    const Eigen::VectorXd plus_residual = BuildSingleObservationResidual(
        camera, observation, VectorToPose(plus_pose), &plus_valid_points);
    const Eigen::VectorXd minus_residual = BuildSingleObservationResidual(
        camera, observation, VectorToPose(minus_pose), &minus_valid_points);
    if (plus_residual.rows() != pose_jacobian.rows() ||
        minus_residual.rows() != pose_jacobian.rows()) {
      continue;
    }
    for (std::size_t point_index = 0;
         point_index < observation.object_points.size(); ++point_index) {
      if (point_index >= base_valid_points.size() ||
          point_index >= plus_valid_points.size() ||
          point_index >= minus_valid_points.size() ||
          base_valid_points[point_index] == 0u ||
          plus_valid_points[point_index] == 0u ||
          minus_valid_points[point_index] == 0u) {
        continue;
      }
      const int row = 2 * static_cast<int>(point_index);
      pose_jacobian.block<2, 1>(row, column) =
          (plus_residual.segment<2>(row) - minus_residual.segment<2>(row)) /
          (2.0 * step);
    }
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> pose_svd(
      pose_jacobian, Eigen::ComputeThinU);
  if (pose_svd.info() != Eigen::Success ||
      pose_svd.singularValues().size() == 0) {
    return result;
  }
  const double maximum_singular_value =
      pose_svd.singularValues().maxCoeff();
  const double pose_rank_tolerance =
      std::max(1e-10, 1e-8 * maximum_singular_value);
  int pose_rank = 0;
  for (Eigen::Index index = 0;
       index < pose_svd.singularValues().size(); ++index) {
    if (pose_svd.singularValues()[index] > pose_rank_tolerance) {
      ++pose_rank;
    }
  }
  result.pose_rank = pose_rank;
  Eigen::MatrixXd marginalized_jacobian = camera_jacobian;
  if (pose_rank > 0) {
    const Eigen::MatrixXd pose_basis = pose_svd.matrixU().leftCols(pose_rank);
    marginalized_jacobian.noalias() -=
        pose_basis * (pose_basis.transpose() * camera_jacobian);
  }
  result.information =
      marginalized_jacobian.transpose() * marginalized_jacobian;
  result.information =
      0.5 * (result.information + result.information.transpose());
  result.success = result.information.allFinite();
  return result;
}

std::vector<OuterObservationRecord> SelectKalibrOuterLmObservations(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<OuterObservationRecord>& observations,
    AutoCameraInitializationSelectionScorer scorer,
    int* pose_success_count,
    int* pose_failure_count,
    SelectionInformationDiagnostics* selection_diagnostics) {
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
    int pose_information_rank = -1;
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
    candidate.pose_success = EstimatePoseFromObjectPointsStrict(
        camera, observation.object_points, observation.image_points,
        kInitializationPoseMaxRmsePx, &candidate.T_camera_board,
        &candidate.pose_fit_outer_rmse);
    if (candidate.pose_success) {
      const CameraInformationResult information =
          ComputeCameraInformationForObservation(
              camera, observation, candidate.T_camera_board, scorer);
      candidate.camera_information = information.information;
      candidate.pose_information_rank = information.pose_rank;
      candidate.pose_success = information.success;
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
    const double current_information_score =
        scorer == AutoCameraInitializationSelectionScorer::LegacyFixedPose
            ? SafeLogDet(selected_information)
            : PrincipalAwareInformationScore(camera, selected_information);
    const double min_useful_information_gain =
        coverage_complete
            ? std::max(1e-6, 1e-6 * best_observed_information_gain)
            : 0.0;

    struct CandidateEvaluation {
      bool usable = false;
      int new_token_count = 0;
      double information_gain = 0.0;
      double score = -std::numeric_limits<double>::infinity();
    };
    std::vector<CandidateEvaluation> evaluations(candidates.size());
    std::atomic<std::size_t> next_candidate(0);
    const auto evaluate_candidates = [&]() {
      for (;;) {
        const std::size_t index = next_candidate.fetch_add(1);
        if (index >= candidates.size()) {
          return;
        }
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
        const Eigen::MatrixXd candidate_total_information =
            selected_information + candidate.camera_information;
        const double candidate_information_score =
            scorer == AutoCameraInitializationSelectionScorer::LegacyFixedPose
                ? SafeLogDet(candidate_total_information)
                : PrincipalAwareInformationScore(camera,
                                                 candidate_total_information);
        const double information_gain =
            std::isfinite(candidate_information_score) &&
                    std::isfinite(current_information_score)
                ? candidate_information_score - current_information_score
                : 0.0;
        evaluations[index].usable = true;
        evaluations[index].new_token_count = new_token_count;
        evaluations[index].information_gain = information_gain;
        evaluations[index].score =
            static_cast<double>(new_token_count) * 10.0 +
            0.2 * std::max(0.0, information_gain) + spacing_bonus +
            1e-6 * candidate.quality_score;
      }
    };
    const std::size_t selection_worker_count =
        ChooseSelectionWorkerCount(candidates.size());
    std::vector<std::thread> selection_workers;
    selection_workers.reserve(selection_worker_count);
    for (std::size_t worker_index = 0;
         worker_index < selection_worker_count; ++worker_index) {
      selection_workers.emplace_back(evaluate_candidates);
    }
    for (std::thread& worker : selection_workers) {
      worker.join();
    }

    int best_index = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    int best_new_token_count = 0;
    double best_information_gain = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      const CandidateEvaluation& evaluation = evaluations[index];
      if (!evaluation.usable) {
        continue;
      }
      best_observed_information_gain =
          std::max(best_observed_information_gain,
                   std::max(0.0, evaluation.information_gain));
      if (coverage_complete &&
          evaluation.information_gain < min_useful_information_gain) {
        continue;
      }
      if (evaluation.score > best_score) {
        best_score = evaluation.score;
        best_index = static_cast<int>(index);
        best_new_token_count = evaluation.new_token_count;
        best_information_gain = evaluation.information_gain;
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
  if (selection_diagnostics != nullptr) {
    PrincipalAwareInformationScore(camera, selected_information,
                                   selection_diagnostics);
    for (int index : selected_indices) {
      const int rank = candidates[static_cast<std::size_t>(index)]
                           .pose_information_rank;
      if (rank < 0) {
        continue;
      }
      if (selection_diagnostics->pose_rank_min < 0) {
        selection_diagnostics->pose_rank_min = rank;
        selection_diagnostics->pose_rank_max = rank;
      } else {
        selection_diagnostics->pose_rank_min =
            std::min(selection_diagnostics->pose_rank_min, rank);
        selection_diagnostics->pose_rank_max =
            std::max(selection_diagnostics->pose_rank_max, rank);
      }
      if (rank < 6) {
        ++selection_diagnostics->pose_rank_deficient_count;
      }
    }
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

SelectionInformationDiagnostics EvaluateSelectionInformationAtCamera(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<OuterObservationRecord>& observations,
    AutoCameraInitializationSelectionScorer scorer) {
  SelectionInformationDiagnostics diagnostics;
  const int parameter_count =
      static_cast<int>(camera.CombinedParameterVector().size());
  Eigen::MatrixXd accumulated =
      Eigen::MatrixXd::Zero(parameter_count, parameter_count);
  for (const OuterObservationRecord& observation : observations) {
    Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
    double pose_rmse = std::numeric_limits<double>::infinity();
    if (!EstimatePoseFromObjectPointsStrict(
            camera, observation.object_points, observation.image_points,
            kInitializationPoseMaxRmsePx, &T_camera_board, &pose_rmse)) {
      continue;
    }
    const CameraInformationResult information =
        ComputeCameraInformationForObservation(
            camera, observation, T_camera_board, scorer);
    if (!information.success ||
        information.information.rows() != parameter_count) {
      continue;
    }
    accumulated += information.information;
    if (information.pose_rank >= 0) {
      if (diagnostics.pose_rank_min < 0) {
        diagnostics.pose_rank_min = information.pose_rank;
        diagnostics.pose_rank_max = information.pose_rank;
      } else {
        diagnostics.pose_rank_min =
            std::min(diagnostics.pose_rank_min, information.pose_rank);
        diagnostics.pose_rank_max =
            std::max(diagnostics.pose_rank_max, information.pose_rank);
      }
      if (information.pose_rank < 6) {
        ++diagnostics.pose_rank_deficient_count;
      }
    }
  }
  PrincipalAwareInformationScore(camera, accumulated, &diagnostics);
  return diagnostics;
}

std::string FormatWeakestCameraDirection(
    const OuterBootstrapCameraIntrinsics& camera,
    const SelectionInformationDiagnostics& diagnostics) {
  const std::vector<std::string> labels = camera.CombinedParameterLabels();
  if (diagnostics.weakest_direction.size() !=
      static_cast<int>(labels.size())) {
    return "unavailable";
  }
  std::ostringstream stream;
  stream.precision(9);
  for (int index = 0; index < static_cast<int>(labels.size()); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << labels[static_cast<std::size_t>(index)] << "="
           << diagnostics.weakest_direction[index];
  }
  return stream.str();
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
    const ApriltagInternalConfig& config,
    int reference_board_id) {
  BootstrapLayout layout;
  OuterBootstrapOptions bootstrap_options;
  bootstrap_options.reference_board_id = reference_board_id;
  if (bootstrap_options.reference_board_id < 0) {
    bootstrap_options.reference_board_id =
        config.tag_ids.empty() ? config.tag_id : config.tag_ids.front();
  }
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
  return EstimatePoseFromObjectPointsStrict(
      camera, object_points, image_points, kInitializationPoseMaxRmsePx, pose,
      rmse);
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
    int total_observation_count,
    int reference_board_id) {
  AutoCameraInitializationCandidate candidate;
  candidate.source_label = "auto_grid_multiboard_layout";
  candidate.evaluation_scope = "layout_full";
  candidate.camera = camera;
  candidate.observation_count = total_observation_count;
  const BootstrapLayout layout =
      BuildBootstrapLayoutFromCamera(camera, frames, config,
                                     reference_board_id);
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
  const bool has_loo =
      candidate.leave_one_board_out_success_count > 0 &&
      std::isfinite(candidate.leave_one_board_out_rmse);
  const double rmse =
      has_loo ? candidate.leave_one_board_out_rmse
              : candidate.mean_observation_rmse;
  return rmse * rmse;
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
    int reference_board_id,
    std::vector<std::string>* warnings) {
  OuterBootstrapCameraIntrinsics best = initial_camera;
  AutoCameraInitializationCandidate best_eval =
      EvaluateCandidateWithMultiBoardLayout(
          best, frames, config, total_observation_count, reference_board_id);
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
                candidate, frames, config, total_observation_count,
                reference_board_id);
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
      const CameraInformationResult information =
          ComputeCameraInformationForObservation(
              camera, *observation, T_camera_board,
              AutoCameraInitializationSelectionScorer::
                  PoseMarginalizedPrincipal);
      if (information.success) {
        candidate.camera_information += information.information;
      }
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

std::vector<KalibrOuterLmFrameView> BuildAllKalibrOuterLmFrameViews(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<OuterObservationRecord>& observations,
    const BootstrapLayout& layout) {
  if (!layout.success || layout.T_reference_board_by_id.empty()) {
    return {};
  }
  std::map<int, KalibrOuterLmFrameView> views_by_frame;
  for (const OuterObservationRecord& observation : observations) {
    if (layout.T_reference_board_by_id.count(observation.board_id) == 0) {
      continue;
    }
    KalibrOuterLmFrameView& view = views_by_frame[observation.frame_index];
    view.frame_index = observation.frame_index;
    view.frame_label = observation.frame_label;
    view.observations.push_back(&observation);
  }

  std::vector<KalibrOuterLmFrameView> views;
  views.reserve(views_by_frame.size());
  for (auto& entry : views_by_frame) {
    KalibrOuterLmFrameView& view = entry.second;
    const auto bootstrap_pose_it =
        layout.T_camera_reference_by_frame.find(view.frame_index);
    if (bootstrap_pose_it != layout.T_camera_reference_by_frame.end()) {
      view.T_camera_reference = bootstrap_pose_it->second;
    } else {
      double pose_rmse = std::numeric_limits<double>::infinity();
      if (!EstimateFramePoseFromLayout(camera,
                                       view.observations,
                                       layout.T_reference_board_by_id,
                                       &view.T_camera_reference,
                                       &pose_rmse)) {
        continue;
      }
    }
    views.push_back(view);
  }
  return views;
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
    const std::vector<KalibrOuterLmView>& views,
    double robust_loss_delta_pixels) {
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
  double robust_cost_sum = 0.0;
  double point_weight_sum = 0.0;
  for (std::size_t view_index = 0; view_index < views.size(); ++view_index) {
    const KalibrOuterLmView& view = views[view_index];
    const OuterObservationRecord& observation = *view.observation;
    const double observation_weight =
        std::max(0.0, view.observation_weight);
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
        squared_error_sum += observation_weight *
                             2.0 * kInvalidProjectionPenalty *
                             kInvalidProjectionPenalty;
        robust_cost_sum += observation_weight *
                           2.0 * kInvalidProjectionPenalty *
                           kInvalidProjectionPenalty;
        point_weight_sum += observation_weight;
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
        squared_error_sum += observation_weight *
                             2.0 * kInvalidProjectionPenalty *
                             kInvalidProjectionPenalty;
        robust_cost_sum += observation_weight *
                           2.0 * kInvalidProjectionPenalty *
                           kInvalidProjectionPenalty;
        point_weight_sum += observation_weight;
        ++evaluation.nonfinite_count;
        continue;
      }
      evaluation.residuals[row++] = dx;
      evaluation.residuals[row++] = dy;
      squared_error_sum += observation_weight * (dx * dx + dy * dy);
      robust_cost_sum += observation_weight *
                         PointNormHuberCost(
                             dx, dy, robust_loss_delta_pixels);
      point_weight_sum += observation_weight;
      if (PointNormHuberWeight(dx, dy, robust_loss_delta_pixels) < 1.0) {
        ++evaluation.downweighted_point_count;
      }
    }
  }

  evaluation.rmse = std::sqrt(squared_error_sum /
                              std::max(1e-12, point_weight_sum));
  evaluation.robust_cost = robust_cost_sum;
  evaluation.robust_rmse = std::sqrt(
      robust_cost_sum / std::max(1e-12, point_weight_sum));
  evaluation.success = evaluation.nonfinite_count == 0;
  return evaluation;
}

KalibrOuterLmEvaluation EvaluateKalibrOuterLmFrameState(
    const Eigen::VectorXd& x,
    const OuterBootstrapCameraIntrinsics& camera_prototype,
    const std::vector<KalibrOuterLmFrameView>& frame_views,
    const std::map<int, Eigen::Isometry3d>& T_reference_board_by_id,
    double robust_loss_delta_pixels = 0.0) {
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
  double robust_cost_sum = 0.0;
  double point_weight_sum = 0.0;
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
          robust_cost_sum += PointNormHuberCost(
              kInvalidProjectionPenalty, kInvalidProjectionPenalty,
              robust_loss_delta_pixels);
          point_weight_sum += 1.0;
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
          robust_cost_sum += PointNormHuberCost(
              kInvalidProjectionPenalty, kInvalidProjectionPenalty,
              robust_loss_delta_pixels);
          point_weight_sum += 1.0;
          ++evaluation.nonfinite_count;
          continue;
        }
        evaluation.residuals[row++] = dx;
        evaluation.residuals[row++] = dy;
        squared_error_sum += dx * dx + dy * dy;
        robust_cost_sum +=
            PointNormHuberCost(dx, dy, robust_loss_delta_pixels);
        point_weight_sum += 1.0;
        if (PointNormHuberWeight(dx, dy, robust_loss_delta_pixels) < 1.0) {
          ++evaluation.downweighted_point_count;
        }
      }
    }
  }

  evaluation.rmse = std::sqrt(squared_error_sum /
                              static_cast<double>(std::max(1, point_count)));
  evaluation.robust_cost = robust_cost_sum;
  evaluation.robust_rmse = std::sqrt(
      robust_cost_sum / std::max(1e-12, point_weight_sum));
  evaluation.success = evaluation.nonfinite_count == 0;
  return evaluation;
}

Eigen::VectorXd BuildSingleFrameResidual(
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const KalibrOuterLmFrameView& frame_view,
    const std::map<int, Eigen::Isometry3d>& T_reference_board_by_id,
    const Eigen::Isometry3d& T_camera_reference,
    std::vector<unsigned char>* valid_points = nullptr) {
  int point_count = 0;
  for (const OuterObservationRecord* observation : frame_view.observations) {
    if (observation != nullptr &&
        T_reference_board_by_id.count(observation->board_id) > 0) {
      point_count += static_cast<int>(observation->object_points.size());
    }
  }
  Eigen::VectorXd residuals =
      Eigen::VectorXd::Constant(std::max(0, 2 * point_count), 100.0);
  if (valid_points != nullptr) {
    valid_points->assign(static_cast<std::size_t>(std::max(0, point_count)),
                         0u);
  }
  if (point_count <= 0) {
    return residuals;
  }

  DoubleSphereCameraModel camera;
  try {
    camera = DoubleSphereCameraModel::FromConfig(
        MakeIntermediateCameraConfig(intrinsics));
  } catch (const std::exception&) {
    return residuals;
  }

  int row = 0;
  std::size_t flat_point_index = 0u;
  for (const OuterObservationRecord* observation : frame_view.observations) {
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
      if (camera.vsEuclideanToKeypoint(
              T_camera_reference * point_reference, &projected) &&
          projected.allFinite()) {
        residuals[row] =
            projected.x() -
            static_cast<double>(observation->image_points[point_index].x);
        residuals[row + 1] =
            projected.y() -
            static_cast<double>(observation->image_points[point_index].y);
        if (valid_points != nullptr) {
          (*valid_points)[flat_point_index] = 1u;
        }
      }
      row += 2;
      ++flat_point_index;
    }
  }
  return residuals;
}

struct KalibrOuterLmSchurStep {
  bool success = false;
  Eigen::VectorXd step;
};

bool FindFocalParameterIndices(const std::vector<std::string>& labels,
                               int* fu_index,
                               int* fv_index) {
  if (fu_index == nullptr || fv_index == nullptr) {
    return false;
  }
  *fu_index = -1;
  *fv_index = -1;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    if (labels[index] == "fu") {
      *fu_index = static_cast<int>(index);
    } else if (labels[index] == "fv") {
      *fv_index = static_cast<int>(index);
    }
  }
  return *fu_index >= 0 && *fv_index >= 0 && *fu_index != *fv_index;
}

bool NormalizeSharedFocalInPlace(OuterBootstrapCameraIntrinsics* camera) {
  if (camera == nullptr) {
    return false;
  }
  const std::vector<std::string> labels = camera->CombinedParameterLabels();
  std::vector<double> parameters = camera->CombinedParameterVector();
  int fu_index = -1;
  int fv_index = -1;
  if (!FindFocalParameterIndices(labels, &fu_index, &fv_index) ||
      fu_index >= static_cast<int>(parameters.size()) ||
      fv_index >= static_cast<int>(parameters.size())) {
    return false;
  }
  const double fu = parameters[static_cast<std::size_t>(fu_index)];
  const double fv = parameters[static_cast<std::size_t>(fv_index)];
  if (!std::isfinite(fu) || !std::isfinite(fv) || fu <= 0.0 || fv <= 0.0) {
    return false;
  }
  const double shared_focal = std::sqrt(fu * fv);
  if (!std::isfinite(shared_focal) || shared_focal <= 0.0) {
    return false;
  }
  parameters[static_cast<std::size_t>(fu_index)] = shared_focal;
  parameters[static_cast<std::size_t>(fv_index)] = shared_focal;
  camera->SetCombinedParameterVector(parameters);
  return ClampIntrinsicsInPlace(camera);
}

bool SolveCameraSchurStep(const Eigen::MatrixXd& schur_H,
                          const Eigen::VectorXd& schur_g,
                          const std::vector<std::string>& camera_labels,
                          bool shared_focal,
                          Eigen::VectorXd* camera_step) {
  if (camera_step == nullptr || schur_H.rows() != schur_H.cols() ||
      schur_H.rows() != schur_g.rows() || !schur_H.allFinite() ||
      !schur_g.allFinite()) {
    return false;
  }
  const int parameter_count = static_cast<int>(schur_g.rows());
  if (!shared_focal) {
    Eigen::LDLT<Eigen::MatrixXd> ldlt(schur_H);
    if (ldlt.info() != Eigen::Success) {
      return false;
    }
    *camera_step = ldlt.solve(-schur_g);
    return camera_step->rows() == parameter_count && camera_step->allFinite();
  }

  int fu_index = -1;
  int fv_index = -1;
  if (!FindFocalParameterIndices(camera_labels, &fu_index, &fv_index) ||
      parameter_count < 2) {
    return false;
  }
  Eigen::MatrixXd reduction =
      Eigen::MatrixXd::Zero(parameter_count, parameter_count - 1);
  int reduced_column = 0;
  int shared_focal_column = -1;
  for (int full_column = 0; full_column < parameter_count; ++full_column) {
    if (full_column == fv_index) {
      continue;
    }
    reduction(full_column, reduced_column) = 1.0;
    if (full_column == fu_index) {
      shared_focal_column = reduced_column;
    }
    ++reduced_column;
  }
  if (shared_focal_column < 0) {
    return false;
  }
  reduction(fv_index, shared_focal_column) = 1.0;
  const Eigen::MatrixXd reduced_H =
      reduction.transpose() * schur_H * reduction;
  const Eigen::VectorXd reduced_g = reduction.transpose() * schur_g;
  Eigen::LDLT<Eigen::MatrixXd> reduced_ldlt(reduced_H);
  if (reduced_ldlt.info() != Eigen::Success) {
    return false;
  }
  const Eigen::VectorXd reduced_step = reduced_ldlt.solve(-reduced_g);
  if (!reduced_step.allFinite()) {
    return false;
  }
  *camera_step = reduction * reduced_step;
  return camera_step->rows() == parameter_count && camera_step->allFinite();
}

KalibrOuterLmSchurStep ComputeKalibrOuterLmFrameSchurStep(
    const Eigen::VectorXd& x,
    const OuterBootstrapCameraIntrinsics& camera_prototype,
    const std::vector<KalibrOuterLmFrameView>& frame_views,
    const std::map<int, Eigen::Isometry3d>& T_reference_board_by_id,
    double lambda,
    double robust_loss_delta_pixels,
    const std::set<std::string>& additional_fixed_camera_labels,
    bool shared_focal) {
  KalibrOuterLmSchurStep result;
  const int camera_parameter_count =
      static_cast<int>(camera_prototype.CombinedParameterVector().size());
  const int parameter_count =
      camera_parameter_count + static_cast<int>(frame_views.size()) * 6;
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
  const std::string initialization_family =
      camera_prototype.NormalizedFamilyString();
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
  pose_blocks.reserve(frame_views.size());

  for (std::size_t frame_index = 0; frame_index < frame_views.size();
       ++frame_index) {
    const int pose_offset =
        camera_parameter_count + static_cast<int>(frame_index) * 6;
    const Eigen::Matrix<double, 6, 1> pose_vector =
        x.segment<6>(pose_offset);
    const Eigen::Isometry3d T_camera_reference = VectorToPose(pose_vector);
    std::vector<unsigned char> valid_points;
    const Eigen::VectorXd residual = BuildSingleFrameResidual(
        camera, frame_views[frame_index], T_reference_board_by_id,
        T_camera_reference, &valid_points);
    if (residual.rows() <= 0 || !residual.allFinite()) {
      continue;
    }

    Eigen::MatrixXd Jc(residual.rows(), camera_parameter_count);
    Jc.setZero();
    for (int column = 0; column < camera_parameter_count; ++column) {
      if (column < static_cast<int>(camera_labels.size()) &&
          (FixKalibrOuterLmInitializationLabel(
               initialization_family,
               camera_labels[static_cast<std::size_t>(column)]) ||
           additional_fixed_camera_labels.count(
               camera_labels[static_cast<std::size_t>(column)]) > 0)) {
        continue;
      }
      const double numeric_step =
          NumericJacobianStep(column, camera_parameter_count, camera_labels, x);
      Eigen::VectorXd plus_vector = x.head(camera_parameter_count);
      Eigen::VectorXd minus_vector = x.head(camera_parameter_count);
      plus_vector[column] += numeric_step;
      minus_vector[column] -= numeric_step;
      OuterBootstrapCameraIntrinsics plus_camera = camera_prototype;
      OuterBootstrapCameraIntrinsics minus_camera = camera_prototype;
      plus_camera.SetCombinedParameterVector(ToStdVector(plus_vector));
      minus_camera.SetCombinedParameterVector(ToStdVector(minus_vector));
      if (!ClampIntrinsicsInPlace(&plus_camera) ||
          !ClampIntrinsicsInPlace(&minus_camera)) {
        continue;
      }
      const Eigen::VectorXd plus_residual = BuildSingleFrameResidual(
          plus_camera, frame_views[frame_index], T_reference_board_by_id,
          T_camera_reference);
      const Eigen::VectorXd minus_residual = BuildSingleFrameResidual(
          minus_camera, frame_views[frame_index], T_reference_board_by_id,
          T_camera_reference);
      if (plus_residual.rows() != residual.rows() ||
          minus_residual.rows() != residual.rows() ||
          !plus_residual.allFinite() || !minus_residual.allFinite()) {
        continue;
      }
      Jc.col(column) =
          (plus_residual - minus_residual) / (2.0 * numeric_step);
    }

    Eigen::Matrix<double, Eigen::Dynamic, 6> Jp(residual.rows(), 6);
    Jp.setZero();
    for (int column = 0; column < 6; ++column) {
      const double numeric_step = NumericJacobianStep(
          pose_offset + column, camera_parameter_count, camera_labels, x);
      Eigen::Matrix<double, 6, 1> plus_pose = pose_vector;
      Eigen::Matrix<double, 6, 1> minus_pose = pose_vector;
      plus_pose[column] += numeric_step;
      minus_pose[column] -= numeric_step;
      const Eigen::VectorXd plus_residual = BuildSingleFrameResidual(
          camera, frame_views[frame_index], T_reference_board_by_id,
          VectorToPose(plus_pose));
      const Eigen::VectorXd minus_residual = BuildSingleFrameResidual(
          camera, frame_views[frame_index], T_reference_board_by_id,
          VectorToPose(minus_pose));
      if (plus_residual.rows() != residual.rows() ||
          minus_residual.rows() != residual.rows() ||
          !plus_residual.allFinite() || !minus_residual.allFinite()) {
        continue;
      }
      Jp.col(column) =
          (plus_residual - minus_residual) / (2.0 * numeric_step);
    }

    Eigen::VectorXd weighted_residual = residual;
    Eigen::MatrixXd weighted_Jc = Jc;
    Eigen::Matrix<double, Eigen::Dynamic, 6> weighted_Jp = Jp;
    for (Eigen::Index row = 0; row + 1 < residual.rows(); row += 2) {
      const std::size_t point_index = static_cast<std::size_t>(row / 2);
      const bool valid = point_index < valid_points.size() &&
                         valid_points[point_index] != 0u;
      const double weight =
          valid ? PointNormHuberWeight(residual[row], residual[row + 1],
                                       robust_loss_delta_pixels)
                : 1.0;
      const double sqrt_weight = std::sqrt(std::max(0.0, weight));
      weighted_residual.segment<2>(row) *= sqrt_weight;
      weighted_Jc.middleRows(row, 2) *= sqrt_weight;
      weighted_Jp.middleRows(row, 2) *= sqrt_weight;
    }

    Hcc.noalias() += weighted_Jc.transpose() * weighted_Jc;
    gc.noalias() += weighted_Jc.transpose() * weighted_residual;
    PoseBlock block;
    block.pose_offset = pose_offset;
    block.Hpp = weighted_Jp.transpose() * weighted_Jp;
    block.Hpc = weighted_Jp.transpose() * weighted_Jc;
    block.gp = weighted_Jp.transpose() * weighted_residual;
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
  Eigen::VectorXd camera_step;
  if (!SolveCameraSchurStep(schur_H, schur_g, camera_labels, shared_focal,
                            &camera_step)) {
    return result;
  }
  result.step.head(camera_parameter_count) = camera_step;
  for (int column = 0; column < camera_parameter_count; ++column) {
    if (column < static_cast<int>(camera_labels.size()) &&
        (FixKalibrOuterLmInitializationLabel(
             initialization_family,
             camera_labels[static_cast<std::size_t>(column)]) ||
         additional_fixed_camera_labels.count(
             camera_labels[static_cast<std::size_t>(column)]) > 0)) {
      result.step[column] = 0.0;
    }
  }
  for (const PoseBlock& block : pose_blocks) {
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(block.Hpp);
    if (ldlt.info() != Eigen::Success) {
      return KalibrOuterLmSchurStep();
    }
    const Eigen::Matrix<double, 6, 1> pose_step = ldlt.solve(
        -(block.gp + block.Hpc * camera_step));
    if (!pose_step.allFinite()) {
      return KalibrOuterLmSchurStep();
    }
    result.step.segment<6>(block.pose_offset) = pose_step;
  }
  result.success = true;
  return result;
}

KalibrOuterLmSchurStep ComputeKalibrOuterLmSchurStep(
    const Eigen::VectorXd& x,
    const OuterBootstrapCameraIntrinsics& camera_prototype,
    const std::vector<KalibrOuterLmView>& views,
    double lambda,
    double robust_loss_delta_pixels,
    const std::set<std::string>& additional_fixed_camera_labels,
    bool shared_focal) {
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
  const std::string initialization_family =
      camera_prototype.NormalizedFamilyString();
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
    std::vector<unsigned char> valid_points;
    const Eigen::VectorXd residual = BuildSingleObservationResidual(
        camera, *view.observation, T_camera_board, &valid_points);
    if (residual.rows() <= 0 || !residual.allFinite()) {
      continue;
    }

    Eigen::MatrixXd Jc(residual.rows(), camera_parameter_count);
    Jc.setZero();
    for (int column = 0; column < camera_parameter_count; ++column) {
      if (column < static_cast<int>(camera_labels.size()) &&
          (FixKalibrOuterLmInitializationLabel(
               initialization_family,
               camera_labels[static_cast<std::size_t>(column)]) ||
           additional_fixed_camera_labels.count(
               camera_labels[static_cast<std::size_t>(column)]) > 0)) {
        continue;
      }
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

    Eigen::VectorXd weighted_residual = residual;
    Eigen::MatrixXd weighted_Jc = Jc;
    Eigen::Matrix<double, Eigen::Dynamic, 6> weighted_Jp = Jp;
    for (Eigen::Index row = 0; row + 1 < residual.rows(); row += 2) {
      const std::size_t point_index = static_cast<std::size_t>(row / 2);
      const bool valid = point_index < valid_points.size() &&
                         valid_points[point_index] != 0u;
      const double weight =
          std::max(0.0, view.observation_weight) *
          (valid ? PointNormHuberWeight(residual[row], residual[row + 1],
                                        robust_loss_delta_pixels)
                 : 1.0);
      const double sqrt_weight = std::sqrt(std::max(0.0, weight));
      weighted_residual.segment<2>(row) *= sqrt_weight;
      weighted_Jc.middleRows(row, 2) *= sqrt_weight;
      weighted_Jp.middleRows(row, 2) *= sqrt_weight;
    }

    Hcc.noalias() += weighted_Jc.transpose() * weighted_Jc;
    gc.noalias() += weighted_Jc.transpose() * weighted_residual;

    PoseBlock block;
    block.pose_offset = pose_offset;
    block.Hpp = weighted_Jp.transpose() * weighted_Jp;
    block.Hpc = weighted_Jp.transpose() * weighted_Jc;
    block.gp = weighted_Jp.transpose() * weighted_residual;
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
  Eigen::VectorXd camera_step;
  if (!SolveCameraSchurStep(schur_H, schur_g, camera_labels, shared_focal,
                            &camera_step)) {
    return result;
  }

  result.step.head(camera_parameter_count) = camera_step;
  for (int column = 0; column < camera_parameter_count; ++column) {
    if (column < static_cast<int>(camera_labels.size()) &&
        (FixKalibrOuterLmInitializationLabel(
             initialization_family,
             camera_labels[static_cast<std::size_t>(column)]) ||
         additional_fixed_camera_labels.count(
             camera_labels[static_cast<std::size_t>(column)]) > 0)) {
      result.step[column] = 0.0;
    }
  }
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
    const std::vector<OuterObservationRecord>& observations,
    double robust_loss_delta_pixels,
    const std::set<std::string>& additional_fixed_camera_labels = {},
    const std::map<std::pair<int, int>, double>* observation_weights =
        nullptr,
    double rescued_observation_lm_weight = 1.0,
    bool shared_focal = false) {
  KalibrOuterLmRefinementResult result;
  OuterBootstrapCameraIntrinsics refinement_camera = initial_camera;
  if (shared_focal && !NormalizeSharedFocalInPlace(&refinement_camera)) {
    return result;
  }
  result.camera = refinement_camera;
  result.robust_loss_delta_pixels =
      std::max(0.0, robust_loss_delta_pixels);

  std::vector<KalibrOuterLmView> views;
  views.reserve(observations.size());
  for (const OuterObservationRecord& observation : observations) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double rmse = 0.0;
    if (!EstimatePoseFromObjectPointsStrict(
            refinement_camera, observation.object_points,
            observation.image_points, kInitializationPoseMaxRmsePx, &pose,
            &rmse)) {
      continue;
    }
    KalibrOuterLmView view;
    view.observation = &observation;
    view.T_camera_board = pose;
    if (observation.used_local_patch_rescue &&
        std::isfinite(rescued_observation_lm_weight) &&
        rescued_observation_lm_weight > 0.0) {
      view.observation_weight = rescued_observation_lm_weight;
    }
    if (observation_weights != nullptr) {
      const auto weight_it = observation_weights->find(
          std::make_pair(observation.frame_index, observation.board_id));
      if (weight_it != observation_weights->end() &&
          std::isfinite(weight_it->second) && weight_it->second > 0.0) {
        view.observation_weight *= weight_it->second;
      }
    }
    views.push_back(view);
  }
  result.view_count = static_cast<int>(views.size());
  if (views.empty()) {
    return result;
  }

  const std::vector<double> camera_parameters =
      refinement_camera.CombinedParameterVector();
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
      EvaluateKalibrOuterLmState(
          x, refinement_camera, views, result.robust_loss_delta_pixels);
  result.initial_rmse = current.rmse;
  result.initial_robust_rmse = current.robust_rmse;
  result.initial_robust_cost = current.robust_cost;
  result.initial_downweighted_point_count = current.downweighted_point_count;
  result.residual_count = static_cast<int>(current.residuals.rows());
  if (!current.success || current.residuals.rows() <= 0 ||
      !std::isfinite(current.rmse)) {
    result.invalid_projection_count = current.invalid_projection_count;
    result.nonfinite_count = current.nonfinite_count;
    return result;
  }

  const std::vector<std::string> camera_labels =
      refinement_camera.CombinedParameterLabels();
  double lambda = 1e-3;
  const bool dense_grid = std::any_of(
      observations.begin(), observations.end(),
      [](const OuterObservationRecord& observation) {
        return observation.object_points.size() > 4u;
      });
  const int max_iterations = dense_grid ? 200 : 25;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const KalibrOuterLmSchurStep schur_step =
        ComputeKalibrOuterLmSchurStep(
            x, refinement_camera, views, lambda,
            result.robust_loss_delta_pixels,
            additional_fixed_camera_labels, shared_focal);
    if (!schur_step.success ||
        schur_step.step.rows() != parameter_count ||
        !schur_step.step.allFinite()) {
      lambda *= 10.0;
      continue;
    }
    const Eigen::VectorXd& step = schur_step.step;
    const double step_norm = step.norm();
    if (step_norm < (dense_grid ? 1e-3 : 1e-8)) {
      result.iteration_count = iteration + 1;
      break;
    }

    const double previous_cost = current.robust_cost;
    Eigen::VectorXd candidate_x = x + step;
    KalibrOuterLmEvaluation candidate =
        EvaluateKalibrOuterLmState(candidate_x, refinement_camera, views,
                                   result.robust_loss_delta_pixels);
    if (candidate.success && std::isfinite(candidate.robust_cost) &&
        candidate.invalid_projection_count <= current.invalid_projection_count &&
        candidate.robust_cost + 1e-9 < current.robust_cost) {
      x = candidate_x;
      x.head(camera_parameter_count) =
          ToEigenVector(candidate.camera.CombinedParameterVector());
      current = EvaluateKalibrOuterLmState(
          x, refinement_camera, views, result.robust_loss_delta_pixels);
      lambda = std::max(1e-9, lambda * 0.3);
      result.iteration_count = iteration + 1;
      const double current_cost = current.robust_cost;
      if (dense_grid && std::isfinite(previous_cost) &&
          std::isfinite(current_cost) &&
          previous_cost - current_cost >= 0.0 &&
          previous_cost - current_cost < 1.0) {
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
  result.final_robust_rmse = current.robust_rmse;
  result.final_robust_cost = current.robust_cost;
  result.final_downweighted_point_count = current.downweighted_point_count;
  result.invalid_projection_count = current.invalid_projection_count;
  result.nonfinite_count = current.nonfinite_count;
  result.improved = std::isfinite(result.initial_robust_cost) &&
                    std::isfinite(result.final_robust_cost) &&
                    result.final_robust_cost + 1e-9 <
                        result.initial_robust_cost;
  return result;
}

void PopulatePrincipalProfile(
    const OuterBootstrapCameraIntrinsics& selected_camera,
    const std::vector<OuterObservationRecord>& observations,
    double robust_loss_delta_pixels,
    double radius_px,
    AutoCameraInitializationResult* result) {
  if (result == nullptr) {
    return;
  }
  result->stage5_init_principal_profile_enabled = 1;
  result->stage5_init_principal_profile_radius_px =
      std::max(0.0, std::abs(radius_px));

  std::vector<OuterObservationRecord> profile_observations;
  profile_observations.reserve(observations.size());
  for (const OuterObservationRecord& observation : observations) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double pose_rmse = std::numeric_limits<double>::infinity();
    if (EstimatePoseFromObjectPointsStrict(
            selected_camera, observation.object_points,
            observation.image_points, kInitializationPoseMaxRmsePx, &pose,
            &pose_rmse)) {
      profile_observations.push_back(observation);
    }
  }
  result->stage5_init_principal_profile_observation_count =
      static_cast<int>(profile_observations.size());
  if (profile_observations.empty()) {
    AppendUniqueWarning(
        "Principal profile diagnostic skipped because no observation has a "
        "valid independent pose at the selected initialization camera.",
        &result->warnings);
    return;
  }

  const double radius = result->stage5_init_principal_profile_radius_px;
  const std::vector<double> offsets =
      radius > 0.0 ? std::vector<double>{-radius, 0.0, radius}
                   : std::vector<double>{0.0};
  const std::set<std::string> fixed_principal_labels{"cu", "cv"};
  result->principal_profile_samples.clear();
  result->principal_profile_samples.reserve(offsets.size() * offsets.size());

  for (double delta_cu : offsets) {
    for (double delta_cv : offsets) {
      AutoCameraInitializationPrincipalProfileSample sample;
      sample.delta_cu_px = delta_cu;
      sample.delta_cv_px = delta_cv;
      sample.fixed_cu = selected_camera.cu + delta_cu;
      sample.fixed_cv = selected_camera.cv + delta_cv;
      sample.expected_view_count =
          static_cast<int>(profile_observations.size());

      OuterBootstrapCameraIntrinsics profile_camera = selected_camera;
      profile_camera.cu = sample.fixed_cu;
      profile_camera.cv = sample.fixed_cv;
      if (!ClampIntrinsicsInPlace(&profile_camera)) {
        result->principal_profile_samples.push_back(sample);
        continue;
      }
      const KalibrOuterLmRefinementResult refined =
          RefineCandidateCameraKalibrOuterLm(profile_camera,
                                             profile_observations,
                                             robust_loss_delta_pixels,
                                             fixed_principal_labels);
      sample.optimized_camera = refined.camera;
      sample.optimized_view_count = refined.view_count;
      sample.residual_count = refined.residual_count;
      sample.iteration_count = refined.iteration_count;
      sample.final_rmse = refined.final_rmse;
      sample.final_robust_rmse = refined.final_robust_rmse;
      sample.final_robust_cost = refined.final_robust_cost;
      sample.comparable =
          refined.view_count == sample.expected_view_count &&
          refined.residual_count > 0 &&
          std::isfinite(refined.final_robust_cost) &&
          std::isfinite(refined.final_rmse) &&
          std::abs(refined.camera.cu - sample.fixed_cu) < 1e-9 &&
          std::abs(refined.camera.cv - sample.fixed_cv) < 1e-9;
      result->principal_profile_samples.push_back(sample);
    }
  }

  double center_cost = std::numeric_limits<double>::infinity();
  for (const AutoCameraInitializationPrincipalProfileSample& sample :
       result->principal_profile_samples) {
    if (sample.comparable && std::abs(sample.delta_cu_px) < 1e-12 &&
        std::abs(sample.delta_cv_px) < 1e-12) {
      center_cost = sample.final_robust_cost;
      break;
    }
  }

  double best_cost = std::numeric_limits<double>::infinity();
  for (AutoCameraInitializationPrincipalProfileSample& sample :
       result->principal_profile_samples) {
    if (!sample.comparable) {
      continue;
    }
    ++result->stage5_init_principal_profile_comparable_sample_count;
    if (std::isfinite(center_cost)) {
      sample.delta_robust_cost = sample.final_robust_cost - center_cost;
    }
    if (sample.final_robust_cost < best_cost) {
      best_cost = sample.final_robust_cost;
      result->stage5_init_principal_profile_best_delta_cu_px =
          sample.delta_cu_px;
      result->stage5_init_principal_profile_best_delta_cv_px =
          sample.delta_cv_px;
      result->stage5_init_principal_profile_best_delta_robust_cost =
          sample.delta_robust_cost;
    }
  }
  result->stage5_init_principal_profile_sample_count =
      static_cast<int>(result->principal_profile_samples.size());

  std::ostringstream warning;
  warning << "Principal profile diagnostic evaluated "
          << result->stage5_init_principal_profile_comparable_sample_count
          << "/" << result->stage5_init_principal_profile_sample_count
          << " comparable fixed-cu/cv samples on "
          << result->stage5_init_principal_profile_observation_count
          << " common independent-pose observations; best grid offset=("
          << result->stage5_init_principal_profile_best_delta_cu_px << ","
          << result->stage5_init_principal_profile_best_delta_cv_px
          << ") px. Diagnostic samples do not update selected intrinsics.";
  AppendUniqueWarning(warning.str(), &result->warnings);
}

void PopulateBoardJackknifeDiagnostic(
    const OuterBootstrapCameraIntrinsics& selected_camera,
    const std::vector<OuterObservationRecord>& observations,
    double robust_loss_delta_pixels,
    AutoCameraInitializationResult* result) {
  if (result == nullptr) {
    return;
  }
  result->stage5_init_board_jackknife_diagnostic_enabled = 1;
  result->stage5_init_board_jackknife_diagnostic_updates_selected_intrinsics = 0;
  std::set<int> board_ids;
  for (const OuterObservationRecord& observation : observations) {
    board_ids.insert(observation.board_id);
  }

  result->board_jackknife_samples.clear();
  result->board_jackknife_samples.reserve(board_ids.size());
  for (int excluded_board_id : board_ids) {
    std::vector<OuterObservationRecord> retained_observations;
    retained_observations.reserve(observations.size());
    for (const OuterObservationRecord& observation : observations) {
      if (observation.board_id != excluded_board_id) {
        retained_observations.push_back(observation);
      }
    }

    AutoCameraInitializationBoardJackknifeSample sample;
    sample.excluded_board_id = excluded_board_id;
    sample.expected_view_count =
        static_cast<int>(retained_observations.size());
    const KalibrOuterLmRefinementResult refinement =
        RefineCandidateCameraKalibrOuterLm(selected_camera,
                                           retained_observations,
                                           robust_loss_delta_pixels);
    sample.optimized_camera = refinement.camera;
    sample.optimized_view_count = refinement.view_count;
    sample.residual_count = refinement.residual_count;
    sample.iteration_count = refinement.iteration_count;
    sample.final_rmse = refinement.final_rmse;
    sample.delta_xi = refinement.camera.xi - selected_camera.xi;
    sample.delta_alpha = refinement.camera.alpha - selected_camera.alpha;
    sample.delta_fu = refinement.camera.fu - selected_camera.fu;
    sample.delta_fv = refinement.camera.fv - selected_camera.fv;
    sample.delta_cu = refinement.camera.cu - selected_camera.cu;
    sample.delta_cv = refinement.camera.cv - selected_camera.cv;
    sample.comparable =
        !retained_observations.empty() &&
        refinement.view_count == sample.expected_view_count &&
        refinement.residual_count > 0 &&
        std::isfinite(refinement.final_rmse) &&
        refinement.camera.IsValid();
    if (sample.comparable) {
      ++result->stage5_init_board_jackknife_diagnostic_comparable_sample_count;
    }
    result->board_jackknife_samples.push_back(sample);
  }
  result->stage5_init_board_jackknife_diagnostic_sample_count =
      static_cast<int>(result->board_jackknife_samples.size());

  std::ostringstream warning;
  warning << "Board jackknife diagnostic evaluated "
          << result->stage5_init_board_jackknife_diagnostic_comparable_sample_count
          << "/" << result->stage5_init_board_jackknife_diagnostic_sample_count
          << " comparable leave-one-board-out camera refinements. Diagnostic "
             "cameras were not committed.";
  AppendUniqueWarning(warning.str(), &result->warnings);
}

void PopulateCoverageWeightedDiagnostic(
    const OuterBootstrapCameraIntrinsics& selected_camera,
    const std::vector<OuterObservationRecord>& observations,
    double robust_loss_delta_pixels,
    AutoCameraInitializationResult* result) {
  if (result == nullptr) {
    return;
  }
  constexpr int kGridRows = 4;
  constexpr int kGridCols = 4;
  result->stage5_init_coverage_weighted_diagnostic_enabled = 1;
  result->stage5_init_coverage_weighted_diagnostic_updates_selected_intrinsics =
      0;
  result->stage5_init_coverage_weighted_diagnostic_grid_rows = kGridRows;
  result->stage5_init_coverage_weighted_diagnostic_grid_cols = kGridCols;
  result->coverage_weight_records.clear();
  result->coverage_weight_records.reserve(observations.size());

  std::map<int, int> bin_counts;
  for (const OuterObservationRecord& observation : observations) {
    if (observation.image_points.empty()) {
      continue;
    }
    double centroid_x = 0.0;
    double centroid_y = 0.0;
    for (const cv::Point2f& point : observation.image_points) {
      centroid_x += static_cast<double>(point.x);
      centroid_y += static_cast<double>(point.y);
    }
    centroid_x /= static_cast<double>(observation.image_points.size());
    centroid_y /= static_cast<double>(observation.image_points.size());
    const int grid_x = std::max(
        0, std::min(kGridCols - 1,
                    static_cast<int>(std::floor(
                        kGridCols * centroid_x /
                        std::max(1, selected_camera.resolution.width)))));
    const int grid_y = std::max(
        0, std::min(kGridRows - 1,
                    static_cast<int>(std::floor(
                        kGridRows * centroid_y /
                        std::max(1, selected_camera.resolution.height)))));
    AutoCameraInitializationCoverageWeightRecord record;
    record.frame_index = observation.frame_index;
    record.frame_label = observation.frame_label;
    record.board_id = observation.board_id;
    record.grid_x = grid_x;
    record.grid_y = grid_y;
    record.centroid_x = centroid_x;
    record.centroid_y = centroid_y;
    result->coverage_weight_records.push_back(record);
    ++bin_counts[grid_y * kGridCols + grid_x];
  }
  result->stage5_init_coverage_weighted_diagnostic_occupied_bin_count =
      static_cast<int>(bin_counts.size());
  if (result->coverage_weight_records.empty() || bin_counts.empty()) {
    AppendUniqueWarning(
        "Coverage-weighted initialization diagnostic had no valid "
        "observation centroids; selected intrinsics were left unchanged.",
        &result->warnings);
    return;
  }

  const double observation_count =
      static_cast<double>(result->coverage_weight_records.size());
  const double occupied_bin_count = static_cast<double>(bin_counts.size());
  double clipped_weight_sum = 0.0;
  for (AutoCameraInitializationCoverageWeightRecord& record :
       result->coverage_weight_records) {
    const int bin = record.grid_y * kGridCols + record.grid_x;
    const int count = std::max(1, bin_counts[bin]);
    const double equal_bin_weight =
        observation_count / (occupied_bin_count * static_cast<double>(count));
    record.weight = std::max(0.25, std::min(4.0, equal_bin_weight));
    clipped_weight_sum += record.weight;
  }
  const double normalization =
      clipped_weight_sum > 0.0 ? observation_count / clipped_weight_sum : 1.0;
  std::map<std::pair<int, int>, double> observation_weights;
  double min_weight = std::numeric_limits<double>::infinity();
  double max_weight = 0.0;
  for (AutoCameraInitializationCoverageWeightRecord& record :
       result->coverage_weight_records) {
    record.weight *= normalization;
    min_weight = std::min(min_weight, record.weight);
    max_weight = std::max(max_weight, record.weight);
    observation_weights[std::make_pair(record.frame_index, record.board_id)] =
        record.weight;
  }
  result->stage5_init_coverage_weighted_diagnostic_min_weight = min_weight;
  result->stage5_init_coverage_weighted_diagnostic_max_weight = max_weight;

  const KalibrOuterLmRefinementResult refinement =
      RefineCandidateCameraKalibrOuterLm(selected_camera,
                                         observations,
                                         robust_loss_delta_pixels,
                                         {},
                                         &observation_weights);
  result->stage5_init_coverage_weighted_diagnostic_camera = refinement.camera;
  result->stage5_init_coverage_weighted_diagnostic_initial_rmse =
      refinement.initial_rmse;
  result->stage5_init_coverage_weighted_diagnostic_final_rmse =
      refinement.final_rmse;

  std::ostringstream warning;
  warning << "Coverage-weighted 4x4 diagnostic used " << bin_counts.size()
          << " occupied bins and observation weights [" << min_weight << ","
          << max_weight << "]; weighted RMSE " << refinement.initial_rmse
          << " -> " << refinement.final_rmse << "; diagnostic cu/cv=("
          << refinement.camera.cu << "," << refinement.camera.cv
          << "). The diagnostic camera was not committed.";
  AppendUniqueWarning(warning.str(), &result->warnings);
}

void PopulatePoseExcitationDiagnostic(
    const OuterBootstrapCameraIntrinsics& selected_camera,
    const std::vector<OuterObservationRecord>& observations,
    AutoCameraInitializationResult* result) {
  if (result == nullptr) {
    return;
  }
  result->stage5_init_pose_excitation_diagnostic_enabled = 1;
  result->stage5_init_pose_excitation_diagnostic_updates_selected_intrinsics =
      0;

  struct PoseSample {
    int frame_index = -1;
    std::string frame_label;
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double pose_rmse = std::numeric_limits<double>::quiet_NaN();
    double centroid_x = 0.0;
    double centroid_y = 0.0;
  };
  std::map<int, int> observation_count_by_board;
  std::map<int, std::vector<PoseSample>> samples_by_board;
  for (const OuterObservationRecord& observation : observations) {
    ++observation_count_by_board[observation.board_id];
    ++result->stage5_init_pose_excitation_pose_total_count;
    if (observation.image_points.empty()) {
      continue;
    }
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double pose_rmse = std::numeric_limits<double>::infinity();
    if (!EstimatePoseFromObjectPointsStrict(
            selected_camera, observation.object_points,
            observation.image_points, kInitializationPoseMaxRmsePx, &pose,
            &pose_rmse)) {
      continue;
    }
    const Eigen::Vector3d normal = pose.linear().col(2).normalized();
    if (!normal.allFinite()) {
      continue;
    }
    PoseSample sample;
    sample.frame_index = observation.frame_index;
    sample.frame_label = observation.frame_label;
    sample.normal = normal;
    sample.pose_rmse = pose_rmse;
    for (const cv::Point2f& point : observation.image_points) {
      sample.centroid_x += static_cast<double>(point.x);
      sample.centroid_y += static_cast<double>(point.y);
    }
    sample.centroid_x /= static_cast<double>(observation.image_points.size());
    sample.centroid_y /= static_cast<double>(observation.image_points.size());
    samples_by_board[observation.board_id].push_back(sample);
    ++result->stage5_init_pose_excitation_pose_success_count;
  }

  result->pose_excitation_records.clear();
  result->pose_excitation_samples.clear();
  result->pose_excitation_records.reserve(observation_count_by_board.size());
  result->pose_excitation_samples.reserve(
      static_cast<std::size_t>(result->stage5_init_pose_excitation_pose_success_count));
  std::vector<double> board_normal_p95_values;
  std::vector<double> board_tilt_range_values;
  std::vector<double> board_normal_xy_axis_balance_values;
  std::vector<Eigen::Vector2d> global_aligned_normal_xy;
  constexpr double kRadiansToDegrees =
      180.0 / 3.14159265358979323846;
  for (const auto& board_entry : observation_count_by_board) {
    AutoCameraInitializationPoseExcitationRecord record;
    record.board_id = board_entry.first;
    record.observation_count = board_entry.second;
    const std::vector<PoseSample>& samples = samples_by_board[record.board_id];
    record.pose_success_count = static_cast<int>(samples.size());
    if (!samples.empty()) {
      const Eigen::Vector3d reference_normal = samples.front().normal;
      Eigen::Vector3d normal_sum = Eigen::Vector3d::Zero();
      std::vector<Eigen::Vector3d> aligned_normals;
      aligned_normals.reserve(samples.size());
      for (const PoseSample& sample : samples) {
        Eigen::Vector3d aligned = sample.normal;
        if (aligned.dot(reference_normal) < 0.0) {
          aligned = -aligned;
        }
        aligned_normals.push_back(aligned);
        normal_sum += aligned;
        const Eigen::Vector3d optical_aligned =
            aligned.z() < 0.0 ? -aligned : aligned;
        global_aligned_normal_xy.push_back(optical_aligned.head<2>());
      }
      const Eigen::Vector3d mean_normal =
          normal_sum.norm() > 1e-9 ? normal_sum.normalized()
                                   : reference_normal;
      Eigen::Vector2d normal_xy_mean = Eigen::Vector2d::Zero();
      for (const Eigen::Vector3d& normal : aligned_normals) {
        normal_xy_mean += normal.head<2>();
      }
      normal_xy_mean /= static_cast<double>(aligned_normals.size());
      Eigen::Matrix2d normal_xy_covariance = Eigen::Matrix2d::Zero();
      for (const Eigen::Vector3d& normal : aligned_normals) {
        const Eigen::Vector2d centered = normal.head<2>() - normal_xy_mean;
        normal_xy_covariance += centered * centered.transpose();
      }
      normal_xy_covariance /=
          static_cast<double>(std::max<std::size_t>(1u,
                                                    aligned_normals.size()));
      record.normal_xy_std_x =
          std::sqrt(std::max(0.0, normal_xy_covariance(0, 0)));
      record.normal_xy_std_y =
          std::sqrt(std::max(0.0, normal_xy_covariance(1, 1)));
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> normal_solver(
          normal_xy_covariance);
      if (normal_solver.info() == Eigen::Success &&
          normal_solver.eigenvalues().allFinite() &&
          normal_solver.eigenvectors().allFinite()) {
        record.normal_xy_weak_variance =
            std::max(0.0, normal_solver.eigenvalues()[0]);
        record.normal_xy_strong_variance =
            std::max(0.0, normal_solver.eigenvalues()[1]);
        record.normal_xy_axis_balance_ratio =
            record.normal_xy_strong_variance > 1e-15
                ? std::sqrt(record.normal_xy_weak_variance /
                            record.normal_xy_strong_variance)
                : 0.0;
        const Eigen::Vector2d dominant_axis =
            normal_solver.eigenvectors().col(1);
        record.normal_xy_dominant_axis_angle_deg =
            std::atan2(dominant_axis.y(), dominant_axis.x()) *
            kRadiansToDegrees;
      }
      std::vector<double> normal_spreads;
      std::vector<double> tilts;
      std::vector<double> centroid_x_values;
      std::vector<double> centroid_y_values;
      for (std::size_t index = 0; index < samples.size(); ++index) {
        const double normal_cosine = std::max(
            -1.0, std::min(1.0, aligned_normals[index].dot(mean_normal)));
        const double normal_deviation_deg =
            std::acos(normal_cosine) * kRadiansToDegrees;
        normal_spreads.push_back(normal_deviation_deg);
        const double optical_axis_cosine =
            std::max(0.0, std::min(1.0,
                                  std::abs(aligned_normals[index].z())));
        const double tilt_deg =
            std::acos(optical_axis_cosine) * kRadiansToDegrees;
        tilts.push_back(tilt_deg);
        centroid_x_values.push_back(samples[index].centroid_x);
        centroid_y_values.push_back(samples[index].centroid_y);

        AutoCameraInitializationPoseExcitationSample public_sample;
        public_sample.frame_index = samples[index].frame_index;
        public_sample.frame_label = samples[index].frame_label;
        public_sample.board_id = record.board_id;
        public_sample.pose_rmse = samples[index].pose_rmse;
        public_sample.normal_x = aligned_normals[index].x();
        public_sample.normal_y = aligned_normals[index].y();
        public_sample.normal_z = aligned_normals[index].z();
        public_sample.normal_deviation_from_board_mean_deg =
            normal_deviation_deg;
        public_sample.tilt_deg = tilt_deg;
        public_sample.centroid_x = samples[index].centroid_x;
        public_sample.centroid_y = samples[index].centroid_y;
        result->pose_excitation_samples.push_back(public_sample);
      }
      record.normal_spread_median_deg = Quantile(normal_spreads, 0.5);
      record.normal_spread_p95_deg = Quantile(normal_spreads, 0.95);
      record.normal_spread_max_deg = Quantile(normal_spreads, 1.0);
      record.tilt_min_deg = Quantile(tilts, 0.0);
      record.tilt_max_deg = Quantile(tilts, 1.0);
      record.tilt_range_deg = record.tilt_max_deg - record.tilt_min_deg;
      record.centroid_min_x = Quantile(centroid_x_values, 0.0);
      record.centroid_max_x = Quantile(centroid_x_values, 1.0);
      record.centroid_min_y = Quantile(centroid_y_values, 0.0);
      record.centroid_max_y = Quantile(centroid_y_values, 1.0);
      record.centroid_span_x = record.centroid_max_x - record.centroid_min_x;
      record.centroid_span_y = record.centroid_max_y - record.centroid_min_y;
      board_normal_p95_values.push_back(record.normal_spread_p95_deg);
      board_tilt_range_values.push_back(record.tilt_range_deg);
      if (std::isfinite(record.normal_xy_axis_balance_ratio)) {
        board_normal_xy_axis_balance_values.push_back(
            record.normal_xy_axis_balance_ratio);
        if (record.normal_xy_axis_balance_ratio < 0.2) {
          ++result->stage5_init_pose_excitation_single_axis_board_count;
        }
      }
    }
    result->pose_excitation_records.push_back(record);
  }

  result->stage5_init_pose_excitation_board_count =
      static_cast<int>(result->pose_excitation_records.size());
  if (!board_normal_p95_values.empty()) {
    result->stage5_init_pose_excitation_min_board_normal_p95_deg =
        Quantile(board_normal_p95_values, 0.0);
    result->stage5_init_pose_excitation_median_board_normal_p95_deg =
        Quantile(board_normal_p95_values, 0.5);
    result->stage5_init_pose_excitation_max_board_normal_p95_deg =
        Quantile(board_normal_p95_values, 1.0);
  }
  if (!board_tilt_range_values.empty()) {
    result->stage5_init_pose_excitation_min_board_tilt_range_deg =
        Quantile(board_tilt_range_values, 0.0);
    result->stage5_init_pose_excitation_median_board_tilt_range_deg =
        Quantile(board_tilt_range_values, 0.5);
  }
  if (!board_normal_xy_axis_balance_values.empty()) {
    result->stage5_init_pose_excitation_min_normal_xy_axis_balance_ratio =
        Quantile(board_normal_xy_axis_balance_values, 0.0);
    result->stage5_init_pose_excitation_median_normal_xy_axis_balance_ratio =
        Quantile(board_normal_xy_axis_balance_values, 0.5);
    result->stage5_init_pose_excitation_max_normal_xy_axis_balance_ratio =
        Quantile(board_normal_xy_axis_balance_values, 1.0);
  }
  if (!global_aligned_normal_xy.empty()) {
    Eigen::Vector2d global_mean = Eigen::Vector2d::Zero();
    for (const Eigen::Vector2d& normal_xy : global_aligned_normal_xy) {
      global_mean += normal_xy;
    }
    global_mean /= static_cast<double>(global_aligned_normal_xy.size());
    Eigen::Matrix2d global_covariance = Eigen::Matrix2d::Zero();
    for (const Eigen::Vector2d& normal_xy : global_aligned_normal_xy) {
      const Eigen::Vector2d centered = normal_xy - global_mean;
      global_covariance += centered * centered.transpose();
    }
    global_covariance /=
        static_cast<double>(global_aligned_normal_xy.size());
    result->stage5_init_pose_excitation_global_normal_xy_std_x =
        std::sqrt(std::max(0.0, global_covariance(0, 0)));
    result->stage5_init_pose_excitation_global_normal_xy_std_y =
        std::sqrt(std::max(0.0, global_covariance(1, 1)));
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> global_solver(
        global_covariance);
    if (global_solver.info() == Eigen::Success &&
        global_solver.eigenvalues().allFinite() &&
        global_solver.eigenvectors().allFinite()) {
      result->stage5_init_pose_excitation_global_normal_xy_weak_variance =
          std::max(0.0, global_solver.eigenvalues()[0]);
      result->stage5_init_pose_excitation_global_normal_xy_strong_variance =
          std::max(0.0, global_solver.eigenvalues()[1]);
      result
          ->stage5_init_pose_excitation_global_normal_xy_axis_balance_ratio =
          result
                      ->stage5_init_pose_excitation_global_normal_xy_strong_variance >
                  1e-15
              ? std::sqrt(
                    result
                        ->stage5_init_pose_excitation_global_normal_xy_weak_variance /
                    result
                        ->stage5_init_pose_excitation_global_normal_xy_strong_variance)
              : 0.0;
      const Eigen::Vector2d dominant_axis =
          global_solver.eigenvectors().col(1);
      result
          ->stage5_init_pose_excitation_global_normal_xy_dominant_axis_angle_deg =
          std::atan2(dominant_axis.y(), dominant_axis.x()) *
          kRadiansToDegrees;
    }
  }

  const double median_spread =
      result->stage5_init_pose_excitation_median_board_normal_p95_deg;
  const double median_axis_balance =
      result
          ->stage5_init_pose_excitation_median_normal_xy_axis_balance_ratio;
  const double global_axis_balance =
      result
          ->stage5_init_pose_excitation_global_normal_xy_axis_balance_ratio;
  if (!std::isfinite(median_spread)) {
    result->stage5_init_pose_excitation_assessment = "unavailable";
  } else if (median_spread < 3.0) {
    result->stage5_init_pose_excitation_assessment =
        "low_rotation_excitation_diagnostic";
  } else if ((std::isfinite(global_axis_balance) &&
              global_axis_balance < 0.5) ||
             (std::isfinite(median_axis_balance) &&
              median_axis_balance < 0.2 &&
              result->stage5_init_pose_excitation_single_axis_board_count ==
                  result->stage5_init_pose_excitation_board_count)) {
    result->stage5_init_pose_excitation_assessment =
        "single_axis_rotation_excitation_principal_risk";
    result
        ->stage5_init_pose_excitation_principal_pseudo_observability_warning =
        1;
  } else if (median_spread < 8.0) {
    result->stage5_init_pose_excitation_assessment =
        "moderate_rotation_excitation_diagnostic";
  } else {
    result->stage5_init_pose_excitation_assessment =
        "strong_rotation_excitation_diagnostic";
  }

  std::ostringstream warning;
  warning << "Pose-excitation diagnostic: board-normal p95 spread min/median/max="
          << result->stage5_init_pose_excitation_min_board_normal_p95_deg
          << "/"
          << result->stage5_init_pose_excitation_median_board_normal_p95_deg
          << "/"
          << result->stage5_init_pose_excitation_max_board_normal_p95_deg
          << " deg; median tilt range="
          << result->stage5_init_pose_excitation_median_board_tilt_range_deg
          << " deg; normal-xy axis-balance min/median/max="
          << result
                 ->stage5_init_pose_excitation_min_normal_xy_axis_balance_ratio
          << "/"
          << result
                 ->stage5_init_pose_excitation_median_normal_xy_axis_balance_ratio
          << "/"
          << result
                 ->stage5_init_pose_excitation_max_normal_xy_axis_balance_ratio
          << "; single-axis boards="
          << result->stage5_init_pose_excitation_single_axis_board_count
          << "/" << result->stage5_init_pose_excitation_board_count
          << "; global axis balance=" << global_axis_balance
          << "; assessment="
          << result->stage5_init_pose_excitation_assessment
          << ". Thresholds are diagnostic only and never gate initialization.";
  AppendUniqueWarning(warning.str(), &result->warnings);
}

KalibrOuterLmRefinementResult RefineCandidateCameraKalibrFrameCohesionLm(
    const OuterBootstrapCameraIntrinsics& initial_camera,
    const std::vector<KalibrOuterLmFrameView>& frame_views,
    const std::map<int, Eigen::Isometry3d>& T_reference_board_by_id,
    double robust_loss_delta_pixels = 0.0,
    const std::set<std::string>& additional_fixed_camera_labels = {},
    bool shared_focal = false) {
  KalibrOuterLmRefinementResult result;
  OuterBootstrapCameraIntrinsics refinement_camera = initial_camera;
  if (shared_focal && !NormalizeSharedFocalInPlace(&refinement_camera)) {
    return result;
  }
  result.camera = refinement_camera;
  result.robust_loss_delta_pixels =
      std::max(0.0, robust_loss_delta_pixels);
  result.view_count = 0;
  for (const KalibrOuterLmFrameView& frame_view : frame_views) {
    result.view_count += static_cast<int>(frame_view.observations.size());
  }
  if (frame_views.empty()) {
    return result;
  }

  const std::vector<double> camera_parameters =
      refinement_camera.CombinedParameterVector();
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
          x, refinement_camera, frame_views, T_reference_board_by_id,
          result.robust_loss_delta_pixels);
  result.initial_rmse = current.rmse;
  result.initial_robust_rmse = current.robust_rmse;
  result.initial_robust_cost = current.robust_cost;
  result.initial_downweighted_point_count = current.downweighted_point_count;
  result.residual_count = static_cast<int>(current.residuals.rows());
  if (!current.success || current.residuals.rows() <= 0 ||
      !std::isfinite(current.rmse) ||
      !std::isfinite(current.robust_cost)) {
    result.invalid_projection_count = current.invalid_projection_count;
    result.nonfinite_count = current.nonfinite_count;
    return result;
  }

  double lambda = 1e-3;
  const bool dense_grid = std::any_of(
      frame_views.begin(), frame_views.end(),
      [](const KalibrOuterLmFrameView& frame_view) {
        return std::any_of(
            frame_view.observations.begin(), frame_view.observations.end(),
            [](const OuterObservationRecord* observation) {
              return observation != nullptr &&
                     observation->object_points.size() > 4u;
            });
      });
  const int max_iterations = dense_grid ? 100 : 25;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const KalibrOuterLmSchurStep schur_step =
        ComputeKalibrOuterLmFrameSchurStep(
            x, refinement_camera, frame_views, T_reference_board_by_id, lambda,
            result.robust_loss_delta_pixels,
            additional_fixed_camera_labels, shared_focal);
    if (!schur_step.success ||
        schur_step.step.rows() != parameter_count ||
        !schur_step.step.allFinite()) {
      lambda *= 10.0;
      continue;
    }
    const Eigen::VectorXd& step = schur_step.step;
    if (step.norm() < (dense_grid ? 1e-3 : 1e-8)) {
      result.iteration_count = iteration + 1;
      break;
    }

    const double previous_cost = current.robust_cost;
    const Eigen::VectorXd candidate_x = x + step;
    const KalibrOuterLmEvaluation candidate =
        EvaluateKalibrOuterLmFrameState(candidate_x,
                                        refinement_camera,
                                        frame_views,
                                        T_reference_board_by_id,
                                        result.robust_loss_delta_pixels);
    if (candidate.success && std::isfinite(candidate.robust_cost) &&
        candidate.invalid_projection_count <= current.invalid_projection_count &&
        candidate.robust_cost + 1e-9 < current.robust_cost) {
      x = candidate_x;
      x.head(camera_parameter_count) =
          ToEigenVector(candidate.camera.CombinedParameterVector());
      current = EvaluateKalibrOuterLmFrameState(
          x, refinement_camera, frame_views, T_reference_board_by_id,
          result.robust_loss_delta_pixels);
      lambda = std::max(1e-9, lambda * 0.3);
      result.iteration_count = iteration + 1;
      if (dense_grid && previous_cost - current.robust_cost >= 0.0 &&
          previous_cost - current.robust_cost < 1.0) {
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
  result.final_robust_rmse = current.robust_rmse;
  result.final_robust_cost = current.robust_cost;
  result.final_downweighted_point_count = current.downweighted_point_count;
  result.invalid_projection_count = current.invalid_projection_count;
  result.nonfinite_count = current.nonfinite_count;
  result.improved = std::isfinite(result.initial_robust_cost) &&
                    std::isfinite(result.final_robust_cost) &&
                    result.final_robust_cost + 1e-9 <
                        result.initial_robust_cost;
  return result;
}

void PopulateFixedLayoutDiagnostic(
    const OuterBootstrapCameraIntrinsics& selected_camera,
    const std::vector<OuterBootstrapFrameInput>& frames,
    const std::vector<OuterObservationRecord>& observations,
    const ApriltagInternalConfig& config,
    int reference_board_id,
    double robust_loss_delta_pixels,
    bool enable_principal_profile,
    double principal_profile_radius_px,
    AutoCameraInitializationResult* result) {
  if (result == nullptr) {
    return;
  }
  result->stage5_init_fixed_layout_diagnostic_enabled = 1;
  result->stage5_init_fixed_layout_diagnostic_updates_selected_intrinsics = 0;
  result->stage5_init_fixed_layout_diagnostic_layout_source =
      "multiboard_outer_bootstrap_at_selected_camera_then_frozen";

  const BootstrapLayout layout =
      BuildBootstrapLayoutFromCamera(selected_camera, frames, config,
                                     reference_board_id);
  result->stage5_init_fixed_layout_diagnostic_layout_success =
      layout.success ? 1 : 0;
  result->stage5_init_fixed_layout_diagnostic_board_count =
      static_cast<int>(layout.T_reference_board_by_id.size());
  result->stage5_init_fixed_layout_diagnostic_layout_bootstrap_rmse =
      layout.global_rmse;
  if (!layout.success || layout.T_reference_board_by_id.empty()) {
    AppendUniqueWarning(
        "Fixed-layout initialization diagnostic could not estimate a common "
        "multi-board layout; selected intrinsics were left unchanged.",
        &result->warnings);
    return;
  }

  const std::vector<KalibrOuterLmFrameView> frame_views =
      BuildAllKalibrOuterLmFrameViews(selected_camera, observations, layout);
  result->stage5_init_fixed_layout_diagnostic_frame_count =
      static_cast<int>(frame_views.size());
  if (frame_views.empty()) {
    AppendUniqueWarning(
        "Fixed-layout initialization diagnostic had no valid shared-rig "
        "frame poses; selected intrinsics were left unchanged.",
        &result->warnings);
    return;
  }

  std::vector<Eigen::Vector2d> rig_normal_xy;
  rig_normal_xy.reserve(frame_views.size());
  for (const KalibrOuterLmFrameView& frame_view : frame_views) {
    Eigen::Vector3d normal =
        frame_view.T_camera_reference.linear().col(2).normalized();
    if (!normal.allFinite()) {
      continue;
    }
    if (normal.z() < 0.0) {
      normal = -normal;
    }
    rig_normal_xy.push_back(normal.head<2>());
  }
  if (rig_normal_xy.size() >= 2u) {
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (const Eigen::Vector2d& normal_xy : rig_normal_xy) {
      mean += normal_xy;
    }
    mean /= static_cast<double>(rig_normal_xy.size());
    Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
    for (const Eigen::Vector2d& normal_xy : rig_normal_xy) {
      const Eigen::Vector2d centered = normal_xy - mean;
      covariance.noalias() += centered * centered.transpose();
    }
    covariance /= static_cast<double>(rig_normal_xy.size());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
    if (solver.info() == Eigen::Success &&
        solver.eigenvalues().allFinite() &&
        solver.eigenvectors().allFinite()) {
      const double weak_variance =
          std::max(0.0, solver.eigenvalues()[0]);
      const double strong_variance =
          std::max(0.0, solver.eigenvalues()[1]);
      result->stage5_init_fixed_layout_diagnostic_rig_axis_balance_ratio =
          strong_variance > 1e-15
              ? std::sqrt(weak_variance / strong_variance)
              : 0.0;
      const Eigen::Vector2d dominant_axis = solver.eigenvectors().col(1);
      result->stage5_init_fixed_layout_diagnostic_rig_dominant_axis_angle_deg =
          std::atan2(dominant_axis.y(), dominant_axis.x()) *
          (180.0 / 3.14159265358979323846);
    }
  }

  const KalibrOuterLmRefinementResult refinement =
      RefineCandidateCameraKalibrFrameCohesionLm(
          selected_camera, frame_views, layout.T_reference_board_by_id,
          robust_loss_delta_pixels);
  result->stage5_init_fixed_layout_diagnostic_camera = refinement.camera;
  result->stage5_init_fixed_layout_diagnostic_board_observation_count =
      refinement.view_count;
  result->stage5_init_fixed_layout_diagnostic_iteration_count =
      refinement.iteration_count;
  result->stage5_init_fixed_layout_diagnostic_initial_rmse =
      refinement.initial_rmse;
  result->stage5_init_fixed_layout_diagnostic_final_rmse =
      refinement.final_rmse;

  if (enable_principal_profile && refinement.camera.IsValid()) {
    result->stage5_init_fixed_layout_principal_profile_enabled = 1;
    result->stage5_init_fixed_layout_principal_profile_radius_px =
        std::max(0.0, std::abs(principal_profile_radius_px));
    const double radius =
        result->stage5_init_fixed_layout_principal_profile_radius_px;
    const std::vector<double> offsets =
        radius > 0.0 ? std::vector<double>{-radius, 0.0, radius}
                     : std::vector<double>{0.0};
    const std::set<std::string> fixed_principal_labels{"cu", "cv"};
    result->fixed_layout_principal_profile_samples.clear();
    result->fixed_layout_principal_profile_samples.reserve(
        offsets.size() * offsets.size());

    for (double delta_cu : offsets) {
      for (double delta_cv : offsets) {
        AutoCameraInitializationPrincipalProfileSample sample;
        sample.delta_cu_px = delta_cu;
        sample.delta_cv_px = delta_cv;
        sample.fixed_cu = refinement.camera.cu + delta_cu;
        sample.fixed_cv = refinement.camera.cv + delta_cv;
        sample.expected_view_count = refinement.view_count;

        OuterBootstrapCameraIntrinsics profile_camera = refinement.camera;
        profile_camera.cu = sample.fixed_cu;
        profile_camera.cv = sample.fixed_cv;
        if (!ClampIntrinsicsInPlace(&profile_camera)) {
          result->fixed_layout_principal_profile_samples.push_back(sample);
          continue;
        }
        const KalibrOuterLmRefinementResult profiled =
            RefineCandidateCameraKalibrFrameCohesionLm(
                profile_camera, frame_views, layout.T_reference_board_by_id,
                robust_loss_delta_pixels, fixed_principal_labels);
        sample.optimized_camera = profiled.camera;
        sample.optimized_view_count = profiled.view_count;
        sample.residual_count = profiled.residual_count;
        sample.iteration_count = profiled.iteration_count;
        sample.final_rmse = profiled.final_rmse;
        sample.final_robust_rmse = profiled.final_robust_rmse;
        sample.final_robust_cost = profiled.final_robust_cost;
        sample.comparable =
            profiled.view_count == sample.expected_view_count &&
            profiled.residual_count > 0 &&
            std::isfinite(profiled.final_robust_cost) &&
            std::isfinite(profiled.final_rmse) &&
            std::abs(profiled.camera.cu - sample.fixed_cu) < 1e-9 &&
            std::abs(profiled.camera.cv - sample.fixed_cv) < 1e-9;
        result->fixed_layout_principal_profile_samples.push_back(sample);
      }
    }

    double center_cost = std::numeric_limits<double>::infinity();
    for (const AutoCameraInitializationPrincipalProfileSample& sample :
         result->fixed_layout_principal_profile_samples) {
      if (sample.comparable && std::abs(sample.delta_cu_px) < 1e-12 &&
          std::abs(sample.delta_cv_px) < 1e-12) {
        center_cost = sample.final_robust_cost;
        break;
      }
    }
    double best_cost = std::numeric_limits<double>::infinity();
    for (AutoCameraInitializationPrincipalProfileSample& sample :
         result->fixed_layout_principal_profile_samples) {
      if (!sample.comparable) {
        continue;
      }
      ++result->stage5_init_fixed_layout_principal_profile_comparable_sample_count;
      if (std::isfinite(center_cost)) {
        sample.delta_robust_cost = sample.final_robust_cost - center_cost;
      }
      if (sample.final_robust_cost < best_cost) {
        best_cost = sample.final_robust_cost;
        result->stage5_init_fixed_layout_principal_profile_best_delta_cu_px =
            sample.delta_cu_px;
        result->stage5_init_fixed_layout_principal_profile_best_delta_cv_px =
            sample.delta_cv_px;
        result
            ->stage5_init_fixed_layout_principal_profile_best_delta_robust_cost =
            sample.delta_robust_cost;
      }
    }
    result->stage5_init_fixed_layout_principal_profile_sample_count =
        static_cast<int>(
            result->fixed_layout_principal_profile_samples.size());
  }

  std::ostringstream warning;
  warning << "Fixed-layout shared-frame-pose diagnostic used "
          << frame_views.size() << " frames / " << refinement.view_count
          << " board observations; RMSE " << refinement.initial_rmse << " -> "
          << refinement.final_rmse << "; diagnostic cu/cv=("
          << refinement.camera.cu << "," << refinement.camera.cv
          << "); rig axis balance="
          << result->stage5_init_fixed_layout_diagnostic_rig_axis_balance_ratio
          << ". The diagnostic camera was not committed.";
  AppendUniqueWarning(warning.str(), &result->warnings);
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
    residual.used_local_patch_rescue = observation.used_local_patch_rescue;

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double pose_fit_outer_rmse = 0.0;
    residual.pose_success = EstimatePoseFromObjectPointsStrict(
        camera, observation.object_points, observation.image_points,
        kInitializationPoseMaxRmsePx, &pose, &pose_fit_outer_rmse);
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
  record.used_local_patch_rescue = observation.used_local_patch_rescue;
  record.pose_init_success = pose_init_success;
  record.pose_fit_outer_rmse = pose_fit_outer_rmse;
  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    record.outer_corners[static_cast<std::size_t>(corner_index)] =
        Eigen::Vector2d(
            observation.diagnostic_outer_image_points[
                static_cast<std::size_t>(corner_index)].x,
            observation.diagnostic_outer_image_points[
                static_cast<std::size_t>(corner_index)].y);
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

void AppendSeedCandidate(
    const OuterBootstrapCameraIntrinsics& seed,
    const std::string& source_label,
    std::vector<AutoCameraInitializationCandidate>* candidates) {
  if (candidates == nullptr || !seed.IsValid()) {
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
  const bool strict_internal_only =
      options.use_direct_dense_control_points &&
      options.direct_dense_control_point_scope == "internal_only";
  OuterBootstrapCameraIntrinsics kalibr_like_seed;
  if (strict_internal_only) {
    source_label = "internal_only_resolution_seed";
    if (seed_diagnostics != nullptr) {
      seed_diagnostics->seed_method =
          "internal_only_resolution_multistart_seed";
      seed_diagnostics->seed_source =
          "resolution_fisheye_fov_prior_no_outer_measurements";
      seed_diagnostics->omni_gamma =
          std::numeric_limits<double>::quiet_NaN();
      seed_diagnostics->omni_gamma_source =
          "not_used_strict_internal_only";
      seed_diagnostics->ds_mapping =
          "not_used_strict_internal_only";
      seed_diagnostics->ds_mapping_verified_against_kalibr_source = 0;
      seed_diagnostics->ds_grid_enumeration_enabled = 0;
    }
    AppendUniqueWarning(
        "Strict internal-only initialization does not consume outer-corner "
        "measurements during seed construction; candidates come only from "
        "the model family and image-resolution focal prior.",
        warnings);
  } else {
    kalibr_like_seed = MakeKalibrLikeOuterSeedIntrinsics(
        image_size, config, observations, &source_label, seed_diagnostics,
        warnings);
  }
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
      const bool canonical_seed = std::abs(focal_scale - 1.0) < 1e-12;
      AppendDsSeedCandidate(
          physical_seed,
          strict_internal_only
              ? (canonical_seed
                     ? "internal_only_resolution_ds_seed"
                     : "internal_only_resolution_multistart_ds_seed")
              : (canonical_seed
                     ? "fallback_outer_only_ds_seed"
                     : "multi_observation_outer_physical_ds_seed"),
          &candidates);
    }
    if (strict_internal_only) {
      AppendUniqueWarning(
          "Added a compact physically plausible DS seed set around "
          "resolution/pi; candidate ranking and refinement use internal "
          "control points only.",
          warnings);
    } else {
      AppendUniqueWarning(
          "Added a compact physically plausible DS seed set around "
          "resolution/pi; selection uses all outer observations, not a "
          "single tag gamma median.",
          warnings);
    }
    if (candidates.empty()) {
      AppendUniqueWarning(
          "DS outer seed was invalid and DS parameter grid enumeration is "
          "disabled for the Stage5 baseline.",
          warnings);
    } else {
      AppendUniqueWarning(
          strict_internal_only
              ? "DS parameter grid enumeration is disabled; using "
                "observation-free physical seeds followed by strict "
                "internal-only optimization."
              : "DS parameter grid enumeration is disabled; using outer-only "
                "DS initialization candidates.",
          warnings);
    }
    return candidates;
  }
  if (!candidates.empty() &&
      options.refine_mode == AutoCameraInitializationRefineMode::KalibrOuterLm) {
    if (family == "pinhole-equi") {
      std::vector<AutoCameraInitializationCandidate> kb_candidates;
      const double base_focal = FisheyeResolutionFocalPrior(image_size);
      const std::array<double, 8> focal_scales{
          0.75, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50};
      const std::array<double, 5> k1_candidates{-0.08, -0.04, 0.0, 0.04, 0.08};
      for (double focal_scale : focal_scales) {
        for (double k1 : k1_candidates) {
          OuterBootstrapCameraIntrinsics physical_seed =
              MakeGenericSeedIntrinsics(image_size, config);
          physical_seed.xi = 0.0;
          physical_seed.alpha = 0.0;
          physical_seed.beta = 0.0;
          physical_seed.fu = focal_scale * base_focal;
          physical_seed.fv = physical_seed.fu;
          physical_seed.cu = 0.5 * static_cast<double>(image_size.width - 1);
          physical_seed.cv = 0.5 * static_cast<double>(image_size.height - 1);
          physical_seed.distortion_coeffs = {k1, 0.0, 0.0, 0.0};
          ClampIntrinsicsInPlace(&physical_seed);
          const bool canonical_seed =
              std::abs(focal_scale - 1.0) < 1e-12 && std::abs(k1) < 1e-12;
          AppendSeedCandidate(
              physical_seed,
              canonical_seed
                  ? "kalibr_pinhole_equi_outer_fisheye_prior_zero_distortion_seed"
                  : "outer_only_pinhole_equi_physical_multistart_seed",
              &kb_candidates);
        }
      }
      if (!kb_candidates.empty()) {
        candidates.swap(kb_candidates);
      }
      AppendUniqueWarning(
          "Using Kalibr-style pinhole-equi initialization: image-center "
          "principal point and an explicit outer-only fisheye focal fallback "
          "because complete target-row circle focal initialization is not "
          "observable from one large tag's four outer corners. The Stage5 "
          "baseline uses a compact KB focal/k1 multistart and then releases "
          "all four KB distortion coefficients in the outer-only LM so "
          "selection evaluates the same camera model used by the backend.",
          warnings);
      return candidates;
    }
    if (family == "eucm-none") {
      std::vector<AutoCameraInitializationCandidate> eucm_candidates;
      if (kalibr_like_seed.IsValid()) {
        AutoCameraInitializationCandidate candidate;
        candidate.source_label = source_label;
        candidate.evaluation_scope = "outer_seed";
        candidate.camera = kalibr_like_seed;
        eucm_candidates.push_back(candidate);
      }
      const std::vector<double> direct_gammas =
          CollectDirectOuterQuadOmniGammaCandidates(
              image_size, observations, nullptr, nullptr);
      for (double gamma : direct_gammas) {
        OuterBootstrapCameraIntrinsics direct_seed =
            MakeGenericSeedIntrinsics(image_size, config);
        direct_seed.alpha = 0.5;
        direct_seed.beta = 1.0;
        direct_seed.xi = 0.0;
        direct_seed.fu = 0.5 * gamma;
        direct_seed.fv = direct_seed.fu;
        direct_seed.cu = 0.5 * static_cast<double>(image_size.width - 1);
        direct_seed.cv = 0.5 * static_cast<double>(image_size.height - 1);
        direct_seed.distortion_coeffs.clear();
        ClampIntrinsicsInPlace(&direct_seed);
        AppendSeedCandidate(direct_seed,
                            "outer_quad_direct_omni_to_eucm_seed",
                            &eucm_candidates);
      }
      const double base_focal = FisheyeResolutionFocalPrior(image_size);
      const std::array<double, 8> focal_scales{
          0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50};
      const std::array<std::pair<double, double>, 7> shape_seeds{{
          {0.50, 1.00},
          {0.45, 1.00},
          {0.55, 1.00},
          {0.50, 0.80},
          {0.50, 1.20},
          {0.40, 0.80},
          {0.60, 1.20},
      }};
      for (double focal_scale : focal_scales) {
        for (const auto& shape_seed : shape_seeds) {
          OuterBootstrapCameraIntrinsics physical_seed =
              MakeGenericSeedIntrinsics(image_size, config);
          physical_seed.alpha = shape_seed.first;
          physical_seed.beta = shape_seed.second;
          physical_seed.xi = 0.0;
          physical_seed.fu = focal_scale * base_focal;
          physical_seed.fv = physical_seed.fu;
          physical_seed.cu = 0.5 * static_cast<double>(image_size.width - 1);
          physical_seed.cv = 0.5 * static_cast<double>(image_size.height - 1);
          physical_seed.distortion_coeffs.clear();
          ClampIntrinsicsInPlace(&physical_seed);
          const bool canonical_seed =
              std::abs(focal_scale - 1.0) < 1e-12 &&
              std::abs(physical_seed.alpha - 0.5) < 1e-12 &&
              std::abs(physical_seed.beta - 1.0) < 1e-12;
          AppendSeedCandidate(
              physical_seed,
              canonical_seed ? "fallback_outer_only_eucm_seed"
                             : "outer_only_eucm_physical_multistart_seed",
              &eucm_candidates);
        }
      }
      if (!eucm_candidates.empty()) {
        candidates.swap(eucm_candidates);
      }
      if (seed_diagnostics != nullptr) {
        seed_diagnostics->ucm_multistart_enabled =
            candidates.size() > 1u ? 1 : 0;
      }
      AppendUniqueWarning(
          "Using Kalibr-style EUCM initialization: Kalibr's EUCM source maps "
          "an Omni initializer to alpha=0.5*omni.xi, beta=1, and "
          "fu/fv=0.5*omni.fu/fv. Because a single large tag contributes only "
          "four outer corners, Stage5 also evaluates a compact EUCM focal/"
          "alpha/beta multistart set around the same physical seed. This is "
          "not the legacy wide DS-style grid, and no YAML/camchain intrinsics "
          "or pinhole-homography focal are used as EUCM parameters.",
          warnings);
      return candidates;
    }
    if (family == "omni-radtan" || family == "omni-none") {
      std::vector<AutoCameraInitializationCandidate> omni_candidates;
      if (kalibr_like_seed.IsValid()) {
        AutoCameraInitializationCandidate candidate;
        candidate.source_label = source_label;
        candidate.evaluation_scope = "outer_seed";
        candidate.camera = kalibr_like_seed;
        omni_candidates.push_back(candidate);
      }
      const double base_extent =
          static_cast<double>(std::max(image_size.width, image_size.height));
      const std::array<double, 7> focal_scales{
          0.50, 0.65, 0.80, 0.95, 1.10, 1.25, 1.45};
      for (double focal_scale : focal_scales) {
        OuterBootstrapCameraIntrinsics physical_seed =
            MakeGenericSeedIntrinsics(image_size, config);
        physical_seed.xi = 1.0;
        physical_seed.alpha = 0.0;
        physical_seed.beta = 0.0;
        physical_seed.fu = focal_scale * base_extent;
        physical_seed.fv = physical_seed.fu;
        physical_seed.cu = 0.5 * static_cast<double>(image_size.width - 1);
        physical_seed.cv = 0.5 * static_cast<double>(image_size.height - 1);
        physical_seed.distortion_coeffs = {0.0, 0.0, 0.0, 0.0};
        ClampIntrinsicsInPlace(&physical_seed);
        AutoCameraInitializationCandidate candidate;
        candidate.source_label =
            std::abs(focal_scale - 0.95) < 1e-12
                ? "kalibr_omni_resolution_scale_seed"
                : "outer_only_physical_omni_radtan_seed";
        candidate.evaluation_scope = "outer_seed";
        candidate.camera = physical_seed;
        omni_candidates.push_back(candidate);
      }
      candidates.swap(omni_candidates);
      AppendUniqueWarning(
          "Using Kalibr Omni/MEI parameterization for initialization: xi=1, "
          "image-center principal point, zero radial-tangential distortion seed, "
          "outer-only Omni gamma when observable, plus a compact resolution-scale "
          "focal seed set because one large tag's four corners can produce an "
          "unstable direct gamma. Radtan distortion is kept fixed during Stage5 "
          "initialization LM and released later in backend.",
          warnings);
      return candidates;
    }
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
    } else if (family == "omni-radtan" || family == "omni-none") {
      AutoCameraInitializationCandidate candidate;
      candidate.source_label = "auto_grid";
      candidate.evaluation_scope = "sampled";
      candidate.camera = seed;
      candidate.camera.xi = 1.0;
      candidate.camera.fu = focal_scale * static_cast<double>(image_size.width);
      candidate.camera.fv = focal_scale * static_cast<double>(image_size.height);
      candidate.camera.cu = center_u;
      candidate.camera.cv = center_v;
      candidate.camera.distortion_coeffs =
          family == "omni-radtan" ? std::vector<double>{0.0, 0.0, 0.0, 0.0}
                                   : std::vector<double>{};
      ClampIntrinsicsInPlace(&candidate.camera);
      candidates.push_back(candidate);
    }
  }
  return candidates;
}

bool ComputeCameraRayRmsDifferenceDeg(
    const OuterBootstrapCameraIntrinsics& lhs,
    const OuterBootstrapCameraIntrinsics& rhs,
    const cv::Size& image_size,
    double* rms_deg,
    int* common_sample_count) {
  if (rms_deg == nullptr || common_sample_count == nullptr ||
      !lhs.IsValid() || !rhs.IsValid() || image_size.width <= 0 ||
      image_size.height <= 0) {
    return false;
  }
  auto make_camera = [&](const OuterBootstrapCameraIntrinsics& intrinsics) {
    IntermediateCameraConfig config;
    config.camera_model = intrinsics.NormalizedCameraModel();
    config.distortion_model = intrinsics.NormalizedDistortionModel();
    config.intrinsics = intrinsics.IntrinsicsVector();
    config.distortion_coeffs = intrinsics.DistortionVector();
    config.resolution = {image_size.width, image_size.height};
    return DoubleSphereCameraModel::FromConfig(config);
  };

  const DoubleSphereCameraModel lhs_camera = make_camera(lhs);
  const DoubleSphereCameraModel rhs_camera = make_camera(rhs);
  if (!lhs_camera.IsValid() || !rhs_camera.IsValid()) {
    return false;
  }

  constexpr int kGridSize = 11;
  constexpr double kPi = 3.14159265358979323846;
  double squared_angle_sum = 0.0;
  int count = 0;
  for (int row = 0; row < kGridSize; ++row) {
    const double y =
        (0.05 + 0.90 * static_cast<double>(row) /
                    static_cast<double>(kGridSize - 1)) *
        static_cast<double>(image_size.height - 1);
    for (int col = 0; col < kGridSize; ++col) {
      const double x =
          (0.05 + 0.90 * static_cast<double>(col) /
                      static_cast<double>(kGridSize - 1)) *
          static_cast<double>(image_size.width - 1);
      Eigen::Vector3d lhs_ray = Eigen::Vector3d::Zero();
      Eigen::Vector3d rhs_ray = Eigen::Vector3d::Zero();
      if (!lhs_camera.keypointToEuclidean(Eigen::Vector2d(x, y), &lhs_ray) ||
          !rhs_camera.keypointToEuclidean(Eigen::Vector2d(x, y), &rhs_ray) ||
          !lhs_ray.allFinite() || !rhs_ray.allFinite() ||
          lhs_ray.norm() <= 1e-12 || rhs_ray.norm() <= 1e-12) {
        continue;
      }
      lhs_ray.normalize();
      rhs_ray.normalize();
      const double cosine =
          std::max(-1.0, std::min(1.0, lhs_ray.dot(rhs_ray)));
      const double angle_deg = std::acos(cosine) * 180.0 / kPi;
      if (!std::isfinite(angle_deg)) {
        continue;
      }
      squared_angle_sum += angle_deg * angle_deg;
      ++count;
    }
  }
  if (count < 25) {
    return false;
  }
  *common_sample_count = count;
  *rms_deg = std::sqrt(squared_angle_sum / static_cast<double>(count));
  return std::isfinite(*rms_deg);
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
  result.stage5_init_shared_focal_requested =
      options_.shared_focal_during_outer_lm ? 1 : 0;
  result.image_size = InferImageSize(frames);
  result.stage5_init_seed_method = "not_attempted";
  result.stage5_init_seed_source = "none";
  result.stage5_init_omni_gamma_source = "not_attempted";
  result.stage5_init_ds_mapping = "not_applicable";
  result.stage5_init_ds_grid_enumeration_enabled = 0;
  result.stage5_init_near_tie_lower_focal_policy_enabled =
      options_.prefer_lower_focal_in_near_tie ? 1 : 0;
  result.stage5_init_near_tie_relative_objective_tolerance =
      std::max(0.0, options_.near_tie_relative_objective_tolerance);
  result.stage5_init_shared_frame_board_constraint_enabled =
      options_.enable_shared_frame_board_constraint ? 1 : 0;
  result.stage5_init_rescued_outer_observation_lm_weight =
      options_.rescued_outer_observation_lm_weight;
  result.stage5_init_reference_board_id = options_.reference_board_id;
  result.stage5_init_kb_focal_source = "not_applicable";
  result.stage5_init_kb_row_circle_focal_available = 0;
  result.stage5_init_kb_zero_distortion_seed = 0;
  result.stage5_init_kb_zero_distortion_seed_included = 0;
  result.stage5_init_kb_nonzero_distortion_seed_count = 0;
  result.stage5_init_kb_distortion_released_in_lm = 0;
  result.stage5_init_kb_distortion_fixed_zero_in_init_lm = 0;
  result.stage5_init_kb_multistart_enabled = 0;
  result.stage5_init_ucm_seed_source = "not_applicable";
  result.stage5_init_ucm_omni_gamma_available = 0;
  result.stage5_init_ucm_mapping = "not_applicable";
  result.stage5_init_ucm_mapping_verified_against_kalibr_source = 0;
  result.stage5_init_ucm_multistart_enabled = 0;
  result.stage5_init_ucm_shape_released_in_lm = 0;
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
  result.stage5_init_lm_min_relative_objective_improvement =
      options_.use_direct_dense_control_points
          ? std::max(
                0.0,
                options_.dense_grid_lm_min_relative_objective_improvement)
          : 0.0;
  result.stage5_init_requires_all_configured_boards_per_frame =
      options_.require_all_configured_boards_per_frame ? 1 : 0;
  const std::set<int> configured_board_ids = ConfiguredBoardIds(config_);
  result.stage5_init_required_board_ids = FormatBoardIds(configured_board_ids);
  result.stage5_init_required_board_count =
      static_cast<int>(configured_board_ids.size());
  result.stage5_init_input_frame_count = static_cast<int>(frames.size());
  result.stage5_init_selection_prefilter =
      options_.require_all_configured_boards_per_frame
          ? "all_configured_boards_per_frame_then_coverage_diversity_spacing"
          : "coverage_diversity_spacing";
  result.stage5_init_selection_scorer = ToString(options_.selection_scorer);
  result.stage5_init_selection_uses_information_metric = 1;
  result.stage5_init_selection_is_exact_kalibr_information_theoretic = 0;
  result.stage5_init_selection_pose_marginalized =
      options_.selection_scorer ==
              AutoCameraInitializationSelectionScorer::
                  PoseMarginalizedPrincipal
          ? 1
          : 0;
  result.stage5_init_selection_principal_subspace_aware =
      result.stage5_init_selection_pose_marginalized;
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
        CollectOuterObservations(
            frames, config_,
            options_.use_direct_dense_control_points,
            options_.direct_dense_control_point_scope);
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
  const bool strict_internal_only =
      options_.use_direct_dense_control_points &&
      options_.direct_dense_control_point_scope == "internal_only";
  const std::vector<OuterObservationRecord> unfiltered_outer_observations =
      CollectOuterObservations(frames, config_, false);
  const std::vector<OuterObservationRecord> full_health_observations =
      CollectOuterObservations(
          frames, config_,
          options_.use_direct_dense_control_points,
          options_.direct_dense_control_point_scope);
  std::set<int> initialization_frame_indices;
  if (options_.require_all_configured_boards_per_frame) {
    initialization_frame_indices = CompleteBoardFrameIndices(
        unfiltered_outer_observations, configured_board_ids);
  } else {
    for (const OuterBootstrapFrameInput& frame : frames) {
      initialization_frame_indices.insert(frame.frame_index);
    }
  }
  result.stage5_init_complete_board_frame_count =
      static_cast<int>(initialization_frame_indices.size());
  result.stage5_init_incomplete_board_frame_rejected_count =
      options_.require_all_configured_boards_per_frame
          ? std::max(0,
                     result.stage5_init_input_frame_count -
                         result.stage5_init_complete_board_frame_count)
          : 0;
  result.stage5_init_observation_count_before_complete_frame_filter =
      static_cast<int>(full_health_observations.size());
  const std::vector<OuterObservationRecord> seed_observations =
      strict_internal_only
          ? std::vector<OuterObservationRecord>{}
          : FilterObservationsByFrameIndices(
                unfiltered_outer_observations, initialization_frame_indices);
  const std::vector<OuterObservationRecord> complete_frame_observations =
      FilterObservationsByFrameIndices(
          full_health_observations, initialization_frame_indices);
  result.stage5_init_observation_count_after_complete_frame_filter =
      static_cast<int>(complete_frame_observations.size());
  const std::vector<OuterObservationRecord> all_observations =
      options_.include_all_valid_outer_observations_in_evaluation
          ? full_health_observations
          : complete_frame_observations;
  result.stage5_init_camera_evaluation_observation_count =
      static_cast<int>(all_observations.size());
  result.stage5_init_all_valid_outer_observations_used =
      options_.include_all_valid_outer_observations_in_evaluation ? 1 : 0;
  result.stage5_init_rescued_outer_observation_count = 0;
  for (const OuterObservationRecord& observation : all_observations) {
    if (observation.used_local_patch_rescue) {
      ++result.stage5_init_rescued_outer_observation_count;
    }
  }
  if (options_.require_all_configured_boards_per_frame) {
    std::ostringstream warning;
    warning << "Automatic camera initialization requires every configured "
               "board in each contributing frame: required_board_ids="
            << result.stage5_init_required_board_ids
            << ", accepted_frames="
            << result.stage5_init_complete_board_frame_count << "/"
            << result.stage5_init_input_frame_count
            << ", accepted_observations="
            << result.stage5_init_observation_count_after_complete_frame_filter
            << "/"
            << result.stage5_init_observation_count_before_complete_frame_filter
            << ". Complete-frame observations remain the seed-construction "
               "diagnostic set; all valid outer observations are used for "
               "camera evaluation/LM when enabled, including incomplete "
               "frames.";
    AppendUniqueWarning(warning.str(), &result.warnings);
  }
  if (options_.use_direct_dense_control_points) {
    int direct_dense_observation_count = 0;
    for (const OuterObservationRecord& observation : all_observations) {
      if (!observation.used_direct_dense_control_points) {
        continue;
      }
      ++direct_dense_observation_count;
      result.stage5_init_dense_control_point_count +=
          static_cast<int>(observation.object_points.size());
      result.bootstrap_internal_points_used +=
          observation.internal_point_count;
    }
    if (direct_dense_observation_count > 0) {
      result.stage5_init_dense_control_points_enabled = 1;
      result.stage5_init_dense_control_points_scope =
          "independent_frame_board_pose_" +
          options_.direct_dense_control_point_scope;
      result.stage5_init_outer_only =
          result.bootstrap_internal_points_used == 0 ? 1 : 0;
      result.stage5_init_selection_prefilter =
          options_.require_all_configured_boards_per_frame
              ? "all_configured_boards_per_frame_then_dense_grid_coverage_diversity_spacing"
              : "dense_grid_coverage_diversity_spacing";
      result.stage5_init_selection_scorer =
          "all_valid_observations_no_initialization_selection";
      result.stage5_init_selection_uses_information_metric = 0;
      result.stage5_init_selection_pose_marginalized = 0;
      result.stage5_init_selection_principal_subspace_aware = 0;
      result.stage5_init_lm_selection_objective =
          "independent_target_pose_dense_grid_lm";
    }
  }
  const int eligible_initialization_observation_count =
      static_cast<int>(all_observations.size());
  result.total_valid_outer_observation_count =
      static_cast<int>(full_health_observations.size());

  if (all_observations.empty()) {
    result.failure_reason = options_.require_all_configured_boards_per_frame
                                ? "No frame contained valid outer corners for every configured board; automatic camera initialization has no eligible observations."
                                : "No valid outer observations with four refined corners were available for automatic camera initialization.";
  } else {
    const std::vector<OuterObservationRecord> sampled_observations =
        SampleObservations(all_observations,
                           options_.max_candidate_observations);
    result.sampled_observation_count =
        static_cast<int>(sampled_observations.size());

    SeedConstructionDiagnostics seed_diagnostics;
    std::vector<AutoCameraInitializationCandidate> candidates =
        GenerateCandidateGrid(
            result.image_size, config_, options_, frames, seed_observations,
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
    const std::string initialized_family =
        MakeGenericSeedIntrinsics(result.image_size, config_).NormalizedFamilyString();
    if (initialized_family == "eucm-none" ||
        initialized_family == "omni-radtan" ||
        initialized_family == "omni-none") {
      result.stage5_init_ucm_seed_source = seed_diagnostics.ucm_seed_source;
      result.stage5_init_ucm_omni_gamma_available =
          std::isfinite(seed_diagnostics.omni_gamma) ? 1 : 0;
      result.stage5_init_ucm_mapping = seed_diagnostics.ucm_mapping;
      result.stage5_init_ucm_mapping_verified_against_kalibr_source =
          seed_diagnostics.ucm_mapping_verified_against_kalibr_source;
      result.stage5_init_ucm_multistart_enabled =
          seed_diagnostics.ucm_multistart_enabled;
    }
    if (initialized_family == "pinhole-equi") {
      int nonzero_kb_seed_count = 0;
      bool zero_kb_seed_included = false;
      for (const AutoCameraInitializationCandidate& candidate : candidates) {
        if (candidate.camera.NormalizedFamilyString() != "pinhole-equi") {
          continue;
        }
        if (HasAnyMeaningfulDistortionCoefficient(candidate.camera)) {
          ++nonzero_kb_seed_count;
        } else {
          zero_kb_seed_included = true;
        }
      }
      result.stage5_init_kb_focal_source = seed_diagnostics.seed_source;
      result.stage5_init_kb_row_circle_focal_available =
          seed_diagnostics.seed_source == "kalibr_row_circle_pinhole_focal" ? 1 : 0;
      result.stage5_init_kb_zero_distortion_seed =
          zero_kb_seed_included ? 1 : 0;
      result.stage5_init_kb_zero_distortion_seed_included =
          zero_kb_seed_included ? 1 : 0;
      result.stage5_init_kb_nonzero_distortion_seed_count =
          nonzero_kb_seed_count;
      result.stage5_init_kb_distortion_fixed_zero_in_init_lm = 0;
      result.stage5_init_kb_distortion_released_in_lm = 1;
      result.stage5_init_kb_multistart_enabled =
          candidates.size() > 1u ? 1 : 0;
    }
    result.stage5_init_selection_prefilter =
        result.stage5_init_dense_control_points_enabled != 0
            ? (options_.require_all_configured_boards_per_frame
                   ? "all_configured_boards_per_frame_then_dense_grid_coverage_diversity_spacing"
                   : "dense_grid_coverage_diversity_spacing")
            : (options_.require_all_configured_boards_per_frame
                   ? "all_configured_boards_per_frame_then_coverage_diversity_spacing"
                   : "coverage_diversity_spacing");
    result.stage5_init_selection_scorer =
        result.stage5_init_dense_control_points_enabled != 0
            ? "all_valid_observations_no_initialization_selection"
            : ToString(options_.selection_scorer);
    result.stage5_init_selection_uses_information_metric =
        result.stage5_init_dense_control_points_enabled != 0 ? 0 : 1;
    result.stage5_init_selection_pose_marginalized =
        result.stage5_init_dense_control_points_enabled == 0 &&
                options_.selection_scorer ==
                    AutoCameraInitializationSelectionScorer::
                        PoseMarginalizedPrincipal
            ? 1
            : 0;
    result.stage5_init_selection_principal_subspace_aware =
        result.stage5_init_selection_pose_marginalized;
    result.stage5_init_selection_is_exact_kalibr_information_theoretic = 0;
    if (!seed_diagnostics.fallback_reason.empty()) {
      AppendUniqueWarning("fallback_reason: " + seed_diagnostics.fallback_reason,
                          &result.warnings);
    }
    for (AutoCameraInitializationCandidate& candidate : candidates) {
      candidate = EvaluateCandidateOnObservations(
          candidate.camera, candidate.source_label, "sampled",
          sampled_observations, 1.0);
    }

    std::sort(candidates.begin(), candidates.end(), CandidateIsBetter);
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      candidates[index].rank = static_cast<int>(index + 1);
    }
    result.candidate_count = static_cast<int>(candidates.size());
    result.candidates = candidates;

    if (!candidates.empty() &&
        (IsAcceptableAutoCandidate(
             candidates.front(), eligible_initialization_observation_count) ||
         (options_.refine_best_candidate &&
          options_.refine_mode ==
              AutoCameraInitializationRefineMode::KalibrOuterLm &&
          IsRefinableKalibrLikeSeed(
              candidates.front(), eligible_initialization_observation_count)))) {
      AutoCameraInitializationCandidate best_candidate = candidates.front();
      OuterBootstrapCameraIntrinsics selected_camera = best_candidate.camera;
      std::string selected_source_label = best_candidate.source_label;
      if (options_.refine_best_candidate &&
          options_.refine_mode ==
              AutoCameraInitializationRefineMode::CoordinateSearch) {
        const OuterBootstrapCameraIntrinsics refined_camera =
            RefineCandidateCamera(
                best_candidate.camera, all_observations,
                options_.rescued_outer_observation_lm_weight);
        const AutoCameraInitializationCandidate refined_eval =
            EvaluateCandidateOnObservations(refined_camera,
                                            "auto_grid_refined",
                                            "full",
                                            all_observations,
                                            options_.rescued_outer_observation_lm_weight);
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
            JoinLabels(KalibrOuterLmReleasedLabels(best_candidate.camera), ",");
        if (best_candidate.camera.NormalizedFamilyString() == "pinhole-equi") {
          result.stage5_init_kb_distortion_released_in_lm = 1;
          result.stage5_init_kb_distortion_fixed_zero_in_init_lm = 0;
          result.stage5_init_kb_multistart_enabled =
              candidates.size() > 1u ? 1 : 0;
        }
        if (best_candidate.camera.NormalizedFamilyString() == "eucm-none" ||
            best_candidate.camera.NormalizedFamilyString() == "omni-radtan" ||
            best_candidate.camera.NormalizedFamilyString() == "omni-none") {
          result.stage5_init_ucm_multistart_enabled =
              candidates.size() > 1u ? 1 : 0;
          result.stage5_init_ucm_shape_released_in_lm = 1;
        }
	        result.stage5_init_calibrate_intrinsics_optimizer =
		            result.stage5_init_dense_control_points_enabled != 0
	                ? "dense_grid_reprojection_lm_camera_intrinsics_and_per_view_pose"
	                : "outer_corner_reprojection_lm_camera_intrinsics_and_pose";
	        result.stage5_init_multiboard_frame_objective_enabled = 0;
	        result.stage5_init_fixed_layout_frame_constraint_used = 0;
	        result.stage5_init_optimizes_layout_variables = 0;
	        result.stage5_init_uses_layout_to_update_intrinsics = 0;
	        result.stage5_init_layout_loo_diagnostics_only = 1;
		        result.stage5_init_lm_selection_objective =
		            "independent_board_pose_outer_lm_plus_full_outer_health";
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
	        double best_lm_objective =
	            std::numeric_limits<double>::infinity();
	        KalibrOuterLmRefinementResult best_lm_refined;
	        AutoCameraInitializationCandidate best_lm_seed = best_candidate;
	        std::string best_lm_source_label;
        std::set<int> best_lm_frame_indices;
        std::vector<OuterObservationRecord> best_selected_lm_observations;
        int best_selected_pose_success_count = 0;
        int best_selected_pose_failure_count = 0;
        PoseFitOutlierGateResult best_pose_fit_gate;
        SelectionInformationDiagnostics best_selection_diagnostics;
        int selected_refined_basin_index = -1;

	        for (const AutoCameraInitializationCandidate& seed_candidate :
	             candidates) {
	          OuterBootstrapCameraIntrinsics refinement_seed_camera =
	              seed_candidate.camera;
	          if (options_.shared_focal_during_outer_lm &&
	              !NormalizeSharedFocalInPlace(&refinement_seed_camera)) {
	            continue;
	          }
	          const double max_image_extent = static_cast<double>(
	              std::max(result.image_size.width, result.image_size.height));
	          const std::string seed_family =
	              seed_candidate.camera.NormalizedFamilyString();
	          const bool priority_rank_seed =
	              seed_candidate.rank <= max_lm_seed_count;
	          const bool low_focal_ray_curve_probe =
	              (seed_family == "ds-none" ||
	               seed_family == "pinhole-equi" ||
	               seed_family == "eucm-none" ||
	               seed_family == "omni-radtan" ||
	               seed_family == "omni-none") &&
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
	                  seed_candidate, eligible_initialization_observation_count)) {
	            continue;
          }
          ++lm_seed_attempt_count;

          const bool dense_grid_lm =
              result.stage5_init_dense_control_points_enabled != 0;
          std::string lm_source_label =
              dense_grid_lm
                  ? "dense_grid_independent_target_pose_lm"
                  : "outer_only_independent_board_pose_lm";
          KalibrOuterLmRefinementResult lm_refined;
          int selected_pose_success_count = 0;
          int selected_pose_failure_count = 0;
          std::vector<OuterObservationRecord> lm_observations;
          SelectionInformationDiagnostics selection_diagnostics;
          if (dense_grid_lm) {
            lm_observations = all_observations;
            for (const OuterObservationRecord& observation : all_observations) {
              Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
              double pose_rmse = std::numeric_limits<double>::infinity();
              if (EstimatePoseFromObjectPointsStrict(
                      refinement_seed_camera, observation.object_points,
                      observation.image_points, kInitializationPoseMaxRmsePx,
                      &pose, &pose_rmse)) {
                ++selected_pose_success_count;
              } else {
                ++selected_pose_failure_count;
              }
            }
          } else {
            lm_observations = SelectKalibrOuterLmObservations(
                refinement_seed_camera,
                all_observations,
                options_.selection_scorer,
                &selected_pose_success_count,
                &selected_pose_failure_count,
                &selection_diagnostics);
          }
          const PoseFitOutlierGateResult pose_fit_gate =
              FilterPoseFitOutliersForInitialization(refinement_seed_camera,
                                                     lm_observations,
                                                     options_.gate_rescued_outer_observations,
                                                     options_.rescued_outer_observation_pose_rmse_gate_pixels);
          PoseFitOutlierGateResult effective_pose_fit_gate = pose_fit_gate;
          int cumulative_pose_fit_outlier_count =
              pose_fit_gate.rejected_outlier_count;
          selected_pose_success_count = static_cast<int>(
              pose_fit_gate.accepted_observations.size());
          const int pose_failure_count_before_outlier_rejection =
              (dense_grid_lm ? pose_fit_gate.pose_failure_count
                             : selected_pose_failure_count);
          selected_pose_failure_count =
              pose_failure_count_before_outlier_rejection +
              cumulative_pose_fit_outlier_count;
          std::set<int> lm_frame_indices;
          std::vector<OuterObservationRecord> selected_lm_observations;
          std::ostringstream label;
          label << (dense_grid_lm
                        ? "dense_grid_independent_target_pose_lm_rank"
                        : "outer_only_independent_board_pose_lm_rank")
                << seed_candidate.rank;
          lm_source_label = label.str();
          selected_lm_observations = pose_fit_gate.accepted_observations;
          for (const OuterObservationRecord& observation :
               selected_lm_observations) {
            lm_frame_indices.insert(observation.frame_index);
          }
	          lm_refined =
	              RefineCandidateCameraKalibrOuterLm(refinement_seed_camera,
	                                                 selected_lm_observations,
	                                                 dense_grid_lm
	                                                     ? options_.dense_grid_lm_huber_delta_pixels
	                                                     : 0.0,
	                                                 {}, nullptr,
	                                                 options_.rescued_outer_observation_lm_weight,
	                                                 options_.shared_focal_during_outer_lm);

          // Intrinsics refinement can change the independent PnP basin of an
          // otherwise successful observation. Recheck pose health at the
          // refined camera and trim only isolated MAD outliers before the
          // camera seed is allowed to compete with other basins.
          for (int cleanup_pass = 0; cleanup_pass < 2; ++cleanup_pass) {
            const PoseFitOutlierGateResult refined_pose_fit_gate =
                FilterPoseFitOutliersForInitialization(
                    lm_refined.camera, selected_lm_observations,
                    options_.gate_rescued_outer_observations,
                    options_.rescued_outer_observation_pose_rmse_gate_pixels);
            if (!refined_pose_fit_gate.gate_applied ||
                refined_pose_fit_gate.rejected_outlier_count <= 0) {
              if (cumulative_pose_fit_outlier_count == 0) {
                effective_pose_fit_gate = refined_pose_fit_gate;
              }
              break;
            }

            cumulative_pose_fit_outlier_count +=
                refined_pose_fit_gate.rejected_outlier_count;
            effective_pose_fit_gate = refined_pose_fit_gate;
            effective_pose_fit_gate.gate_applied = true;
            effective_pose_fit_gate.rejected_outlier_count =
                cumulative_pose_fit_outlier_count;
            selected_lm_observations =
                refined_pose_fit_gate.accepted_observations;
            lm_frame_indices.clear();
            for (const OuterObservationRecord& observation :
                 selected_lm_observations) {
              lm_frame_indices.insert(observation.frame_index);
            }

            const KalibrOuterLmRefinementResult previous_refinement =
                lm_refined;
            KalibrOuterLmRefinementResult cleaned_refinement =
                RefineCandidateCameraKalibrOuterLm(
                    previous_refinement.camera,
                    selected_lm_observations,
                    dense_grid_lm
                        ? options_.dense_grid_lm_huber_delta_pixels
                        : 0.0,
                    {}, nullptr,
                    options_.rescued_outer_observation_lm_weight,
                    options_.shared_focal_during_outer_lm);
            cleaned_refinement.initial_rmse =
                previous_refinement.initial_rmse;
            cleaned_refinement.initial_robust_rmse =
                previous_refinement.initial_robust_rmse;
            cleaned_refinement.initial_robust_cost =
                previous_refinement.initial_robust_cost;
            cleaned_refinement.initial_downweighted_point_count =
                previous_refinement.initial_downweighted_point_count;
            cleaned_refinement.iteration_count +=
                previous_refinement.iteration_count;
            cleaned_refinement.improved =
                cleaned_refinement.improved ||
                (std::isfinite(cleaned_refinement.initial_robust_cost) &&
                 std::isfinite(cleaned_refinement.final_robust_cost) &&
                 cleaned_refinement.final_robust_cost + 1e-12 <
                     cleaned_refinement.initial_robust_cost);
            lm_refined = cleaned_refinement;
          }

          // Reclassify against the complete candidate set after cleanup. A
          // transient PnP branch at an intermediate camera must not
          // permanently discard an observation that is healthy under the
          // cleaned camera. Iterating to a stable observation set supports
          // both removal and re-admission without dataset-specific rules.
          const auto observation_key_set = [](
              const std::vector<OuterObservationRecord>& observations) {
            std::set<std::pair<int, int>> keys;
            for (const OuterObservationRecord& observation : observations) {
              keys.insert(std::make_pair(observation.frame_index,
                                         observation.board_id));
            }
            return keys;
          };
          for (int reconciliation_pass = 0;
               reconciliation_pass < 4;
               ++reconciliation_pass) {
            const PoseFitOutlierGateResult reconciled_pose_fit_gate =
                FilterPoseFitOutliersForInitialization(
                    lm_refined.camera, lm_observations,
                    options_.gate_rescued_outer_observations,
                    options_.rescued_outer_observation_pose_rmse_gate_pixels);
            effective_pose_fit_gate = reconciled_pose_fit_gate;
            cumulative_pose_fit_outlier_count =
                reconciled_pose_fit_gate.rejected_outlier_count;
            if (observation_key_set(
                    reconciled_pose_fit_gate.accepted_observations) ==
                observation_key_set(selected_lm_observations)) {
              break;
            }

            selected_lm_observations =
                reconciled_pose_fit_gate.accepted_observations;
            lm_frame_indices.clear();
            for (const OuterObservationRecord& observation :
                 selected_lm_observations) {
              lm_frame_indices.insert(observation.frame_index);
            }
            const KalibrOuterLmRefinementResult previous_refinement =
                lm_refined;
            KalibrOuterLmRefinementResult reconciled_refinement =
                RefineCandidateCameraKalibrOuterLm(
                    previous_refinement.camera,
                    selected_lm_observations,
                    dense_grid_lm
                        ? options_.dense_grid_lm_huber_delta_pixels
                        : 0.0,
                    {}, nullptr,
                    options_.rescued_outer_observation_lm_weight,
                    options_.shared_focal_during_outer_lm);
            reconciled_refinement.initial_rmse =
                previous_refinement.initial_rmse;
            reconciled_refinement.initial_robust_rmse =
                previous_refinement.initial_robust_rmse;
            reconciled_refinement.initial_robust_cost =
                previous_refinement.initial_robust_cost;
            reconciled_refinement.initial_downweighted_point_count =
                previous_refinement.initial_downweighted_point_count;
            reconciled_refinement.iteration_count +=
                previous_refinement.iteration_count;
            reconciled_refinement.improved =
                reconciled_refinement.improved ||
                (std::isfinite(reconciled_refinement.initial_robust_cost) &&
                 std::isfinite(reconciled_refinement.final_robust_cost) &&
                 reconciled_refinement.final_robust_cost + 1e-12 <
                     reconciled_refinement.initial_robust_cost);
            lm_refined = reconciled_refinement;
          }
          selected_pose_success_count =
              static_cast<int>(selected_lm_observations.size());
          selected_pose_failure_count =
              pose_failure_count_before_outlier_rejection +
              cumulative_pose_fit_outlier_count;

          KalibrOuterLmRefinementResult selection_refined = lm_refined;
          std::string selection_objective_label =
              dense_grid_lm
                  ? "independent_target_pose_dense_grid_lm"
                  : "independent_board_pose_outer_lm_plus_full_outer_health";
          std::set<int> selection_frame_indices = lm_frame_indices;
          std::vector<OuterObservationRecord> selection_lm_observations =
              selected_lm_observations;
          bool used_fixed_layout_frame_constraint = false;
          BootstrapLayout shared_layout;
          KalibrOuterLmRefinementResult shared_layout_refinement;
          int shared_layout_frame_count = 0;
          int shared_layout_board_count = 0;
          int shared_layout_observation_count = 0;
          const bool use_fixed_measured_layout =
              !options_.fixed_board_layout.empty();
          if (!dense_grid_lm &&
              (options_.enable_shared_frame_board_constraint ||
               use_fixed_measured_layout)) {
            // The independent LM can explain a wrong camera by giving every
            // frame-board pair its own pose.  Re-estimate a common board
            // layout, then optimize one pose per frame.  The resulting
            // camera is used for basin selection; the board layout itself is
            // not committed to the later Stage5 backend.
            if (use_fixed_measured_layout) {
              shared_layout.success = true;
              shared_layout.T_reference_board_by_id =
                  options_.fixed_board_layout;
              shared_layout.used_board_observation_count =
                  static_cast<int>(all_observations.size());
            } else {
              shared_layout = BuildBootstrapLayoutFromCamera(
                  selection_refined.camera, frames, config_,
                  options_.reference_board_id);
            }
            if (shared_layout.success &&
                !shared_layout.T_reference_board_by_id.empty()) {
              const std::vector<KalibrOuterLmFrameView> shared_frame_views =
                  BuildAllKalibrOuterLmFrameViews(
                      selection_refined.camera, all_observations, shared_layout);
              shared_layout_frame_count =
                  static_cast<int>(shared_frame_views.size());
              shared_layout_board_count = static_cast<int>(
                  shared_layout.T_reference_board_by_id.size());
              for (const KalibrOuterLmFrameView& frame_view :
                   shared_frame_views) {
                shared_layout_observation_count +=
                    static_cast<int>(frame_view.observations.size());
              }
              if (!shared_frame_views.empty()) {
                shared_layout_refinement =
                    RefineCandidateCameraKalibrFrameCohesionLm(
                        selection_refined.camera, shared_frame_views,
                        shared_layout.T_reference_board_by_id, 0.0, {},
                        options_.shared_focal_during_outer_lm);
                if (shared_layout_refinement.camera.IsValid() &&
                    shared_layout_refinement.residual_count > 0 &&
                    std::isfinite(shared_layout_refinement.final_robust_rmse)) {
                  selection_refined = shared_layout_refinement;
                  selection_objective_label =
                      "outer_only_shared_frame_board_constraint_lm_plus_full_outer_health";
                  lm_source_label =
                      "outer_only_shared_frame_board_constraint_lm_rank" +
                      std::to_string(seed_candidate.rank);
                  used_fixed_layout_frame_constraint = true;
                  selection_frame_indices.clear();
                  for (const KalibrOuterLmFrameView& frame_view :
                       shared_frame_views) {
                    selection_frame_indices.insert(frame_view.frame_index);
                  }
                  selection_lm_observations = all_observations;
                  selected_pose_success_count =
                      shared_layout_observation_count;
                  selected_pose_failure_count = std::max(
                      0, static_cast<int>(all_observations.size()) -
                             shared_layout_observation_count);
                }
              }
            }
          }
          if (!frame_cohesion_diagnostics_skip_logged) {
            AppendUniqueWarning(
                dense_grid_lm
                    ? "Dense control-point initialization uses an independent target pose for every frame-board observation and all valid imported control points; no board-layout variables update intrinsics."
                    : (options_.enable_shared_frame_board_constraint
                           ? "Outer-only initialization uses an independent-pose LM only for basin seeding, then selects with a shared board-layout / shared frame-pose LM; layout variables remain fixed after bootstrap."
                           : "Shared frame-board initialization constraint disabled; selecting with independent outer-only board-pose LM plus full outer health."),
                &result.warnings);
            frame_cohesion_diagnostics_skip_logged = true;
          }

          const AutoCameraInitializationCandidate refined_eval =
              EvaluateCandidateOnObservations(selection_refined.camera,
	                                              lm_source_label,
	                                              "full",
	                                              all_observations,
	                                              options_.rescued_outer_observation_lm_weight);
	          const double seed_objective = CandidateObjective(seed_candidate);
	          const double refined_objective = CandidateObjective(refined_eval);
	          const double lm_selection_objective =
	              std::isfinite(selection_refined.final_robust_rmse)
	                  ? selection_refined.final_robust_rmse *
	                        selection_refined.final_robust_rmse
	                  : std::numeric_limits<double>::infinity();
	          const bool omni_none_seed =
	              selection_refined.camera.NormalizedFamilyString() ==
	              "omni-none";
	          // The independent LM is a basin refiner, not the final camera
	          // ranking objective.  Rank every basin on the same full Outer4
	          // observation set, using Huber loss so a small number of bad
	          // pose hypotheses cannot make a physically good basin lose to a
	          // worse one.  Keep the LM term only as a deterministic tie-break.
	          const double full_outer_health_selection_weight = 1.0;
          const double full_outer_health_objective =
              std::isfinite(refined_eval.robust_observation_rmse)
                  ? refined_eval.robust_observation_rmse *
                        refined_eval.robust_observation_rmse
                  : std::numeric_limits<double>::infinity();
          const double combined_lm_selection_objective =
              std::isfinite(lm_selection_objective) &&
                      std::isfinite(full_outer_health_objective)
                  ? full_outer_health_selection_weight *
                        full_outer_health_objective +
                        1e-3 * lm_selection_objective
                  : std::numeric_limits<double>::infinity();
	          std::ostringstream trial_warning;
	          trial_warning
	              << "Kalibr-style multi-start LM trial rank="
              << seed_candidate.rank
              << " source=" << seed_candidate.source_label
              << " seed_fu=" << seed_candidate.camera.fu
              << " seed_alpha=" << seed_candidate.camera.alpha
              << " seed_beta=" << seed_candidate.camera.beta
              << " refined_fu=" << selection_refined.camera.fu
	              << " refined_alpha=" << selection_refined.camera.alpha
	              << " refined_beta=" << selection_refined.camera.beta
	              << " refined_xi=" << selection_refined.camera.xi
	              << " seed_objective=" << seed_objective
	              << " refined_objective=" << refined_objective
	              << " lm_initial_rmse=" << selection_refined.initial_rmse
	              << " lm_final_rmse=" << selection_refined.final_rmse
	              << " lm_robust_loss_delta_px="
	              << selection_refined.robust_loss_delta_pixels
	              << " lm_initial_robust_rmse="
	              << selection_refined.initial_robust_rmse
	              << " lm_final_robust_rmse="
	              << selection_refined.final_robust_rmse
	              << " lm_initial_downweighted_points="
	              << selection_refined.initial_downweighted_point_count
	              << " lm_final_downweighted_points="
	              << selection_refined.final_downweighted_point_count
	              << " lm_selection_objective=" << lm_selection_objective
	              << " full_outer_health_weight="
	              << full_outer_health_selection_weight
              << " combined_lm_selection_objective="
              << combined_lm_selection_objective
              << " selected_frames=" << selection_frame_indices.size()
              << " selected_board_observations="
              << selection_refined.view_count
              << " shared_layout_constraint_used="
              << (used_fixed_layout_frame_constraint ? 1 : 0)
              << " shared_layout_frames=" << shared_layout_frame_count
              << " shared_layout_boards=" << shared_layout_board_count
              << " shared_layout_observations="
              << shared_layout_observation_count
              << " shared_layout_initial_rmse="
              << shared_layout_refinement.initial_rmse
              << " shared_layout_final_rmse="
              << shared_layout_refinement.final_rmse
              << " fixed_layout_frame_constraint="
              << (used_fixed_layout_frame_constraint ? 1 : 0)
              << " layout_updates_intrinsics=0"
              << " layout_variables_optimized=0"
              << " camera_step_gate=finite_model_validity_only"
              << " pose_fit_gate_applied="
              << (effective_pose_fit_gate.gate_applied ? 1 : 0)
              << " pose_fit_gate_rejected="
              << effective_pose_fit_gate.rejected_outlier_count
              << " pose_fit_gate_median_rmse="
              << effective_pose_fit_gate.median_rmse
              << " pose_fit_gate_mad_rmse="
              << effective_pose_fit_gate.mad_rmse
              << " pose_fit_gate_threshold_rmse="
              << effective_pose_fit_gate.threshold_rmse << ".";
	          AppendUniqueWarning(trial_warning.str(), &result.warnings);

	          const bool shared_constraint_finite =
	              used_fixed_layout_frame_constraint &&
	              selection_refined.camera.IsValid() &&
	              std::isfinite(selection_refined.final_robust_cost) &&
	              std::isfinite(selection_refined.final_robust_rmse);
	          const bool refinement_health_acceptable =
	              selection_refined.improved || shared_constraint_finite;
	          const bool full_outer_health_acceptable =
	              refinement_health_acceptable &&
	              std::isfinite(selection_refined.final_robust_rmse) &&
	              selection_refined.residual_count > 0 &&
	              refined_eval.projection_failure_count == 0 &&
	              std::isfinite(refined_eval.p95_observation_rmse) &&
	              refined_eval.p95_observation_rmse < 50.0 &&
	              (options_.use_direct_dense_control_points
	                   ? selection_refined.camera.IsValid()
	                   : refined_eval.valid &&
	                         (omni_none_seed
	                              ? IsPhysicallyPlausibleOmniNoneSeed(
	                                    selection_refined.camera)
	                              : IsRefinableKalibrLikeSeed(
	                                    refined_eval,
	                                    eligible_initialization_observation_count)));
          const bool camera_init_step_finite =
              seed_candidate.camera.IsValid() &&
              selection_refined.camera.IsValid() &&
              std::isfinite(seed_candidate.camera.fu) &&
              std::isfinite(seed_candidate.camera.fv) &&
              std::isfinite(selection_refined.camera.fu) &&
              std::isfinite(selection_refined.camera.fv);
          // A measured layout removes the camera/layout ambiguity. In that
          // explicit mode rank basins by the one-pose-per-frame objective,
          // rather than independent frame-board PnP which can hide a wrong
          // camera behind an independently fitted pose for each board.
          const bool objective_improved_before_near_tie_policy =
              selected_refined_basin_index < 0 ||
              (use_fixed_measured_layout
                   ? lm_selection_objective + 1e-12 < best_lm_objective
                   : RefinedOuterEvaluationIsBetter(
                         refined_eval, lm_selection_objective, best_candidate,
                         best_lm_objective));
          bool frame_objective_improved =
              objective_improved_before_near_tie_policy;
		          bool compared_as_near_tie = false;
		          bool preferred_by_lower_focal_near_tie_policy = false;
		          if (options_.prefer_lower_focal_in_near_tie &&
	              !options_.use_direct_dense_control_points &&
	              selection_refined.camera.NormalizedFamilyString() ==
	                  "ds-none" &&
	              std::isfinite(combined_lm_selection_objective) &&
	              std::isfinite(best_lm_selection_objective) &&
	              best_lm_refined.camera.IsValid()) {
	            const double tolerance =
	                std::max(0.0,
	                         options_.near_tie_relative_objective_tolerance);
	            const double objective_scale = std::max(
	                1e-12,
	                std::min(std::abs(combined_lm_selection_objective),
	                         std::abs(best_lm_selection_objective)));
		            const bool near_tie =
	                std::abs(combined_lm_selection_objective -
	                         best_lm_selection_objective) <=
	                tolerance * objective_scale;
		            if (near_tie) {
		              compared_as_near_tie = true;
		              const double candidate_focal = std::sqrt(
	                  selection_refined.camera.fu *
	                  selection_refined.camera.fv);
	              const double best_focal = std::sqrt(
	                  best_lm_refined.camera.fu *
	                  best_lm_refined.camera.fv);
		              frame_objective_improved =
		                  std::isfinite(candidate_focal) &&
		                  std::isfinite(best_focal) &&
		                  candidate_focal + 1e-9 < best_focal;
		              preferred_by_lower_focal_near_tie_policy =
		                  frame_objective_improved;
		            }
		          }
	          if (frame_objective_improved &&
	              options_.use_direct_dense_control_points &&
	              std::isfinite(best_lm_selection_objective)) {
	            const double required_improvement =
	                std::max(0.0,
	                         options_
	                             .dense_grid_lm_min_relative_objective_improvement) *
	                std::max(1e-12, std::abs(best_lm_selection_objective));
	            frame_objective_improved =
	                combined_lm_selection_objective + required_improvement <
	                best_lm_selection_objective;
	          }

          AutoCameraInitializationRefinedBasinCandidate basin_record;
          basin_record.trial_index =
              static_cast<int>(result.refined_basin_candidates.size());
          basin_record.seed_rank = seed_candidate.rank;
          basin_record.seed_source_label = seed_candidate.source_label;
          basin_record.seed_camera = seed_candidate.camera;
          basin_record.refined_camera = selection_refined.camera;
          basin_record.selected_frame_count =
              static_cast<int>(selection_frame_indices.size());
          basin_record.selected_board_observation_count =
              selection_refined.view_count;
          basin_record.residual_count = selection_refined.residual_count;
          basin_record.iteration_count = selection_refined.iteration_count;
          basin_record.seed_objective = seed_objective;
          basin_record.full_outer_objective = refined_objective;
          basin_record.full_outer_robust_rmse =
              refined_eval.robust_observation_rmse;
          basin_record.full_outer_median_rmse =
              refined_eval.median_observation_rmse;
          basin_record.full_outer_p95_rmse = refined_eval.p95_observation_rmse;
          basin_record.full_outer_pose_success_count =
              refined_eval.pose_success_count;
          basin_record.full_outer_pose_failure_count =
              refined_eval.pose_failure_count;
          basin_record.full_outer_max_rmse = refined_eval.max_observation_rmse;
          basin_record.full_outer_worst_frame_index =
              refined_eval.worst_observation_frame_index;
          basin_record.full_outer_worst_board_id =
              refined_eval.worst_observation_board_id;
          basin_record.full_outer_projection_failure_count =
              refined_eval.projection_failure_count;
          basin_record.lm_initial_rmse = selection_refined.initial_rmse;
          basin_record.lm_final_rmse = selection_refined.final_rmse;
          basin_record.lm_final_robust_rmse =
              selection_refined.final_robust_rmse;
          basin_record.shared_layout_constraint_used =
              used_fixed_layout_frame_constraint ? 1 : 0;
          basin_record.shared_layout_frame_count = shared_layout_frame_count;
          basin_record.shared_layout_board_count = shared_layout_board_count;
          basin_record.shared_layout_observation_count =
              shared_layout_observation_count;
          basin_record.shared_layout_initial_rmse =
              shared_layout_refinement.initial_rmse;
          basin_record.shared_layout_final_rmse =
              shared_layout_refinement.final_rmse;
          basin_record.shared_layout_final_robust_rmse =
              shared_layout_refinement.final_robust_rmse;
          basin_record.combined_selection_objective =
              combined_lm_selection_objective;
          basin_record.full_outer_health_acceptable =
              full_outer_health_acceptable;
          basin_record.camera_step_finite = camera_init_step_finite;
          basin_record.objective_improved_before_near_tie_policy =
              objective_improved_before_near_tie_policy;
          basin_record.compared_as_near_tie = compared_as_near_tie;
          basin_record.preferred_by_lower_focal_near_tie_policy =
              preferred_by_lower_focal_near_tie_policy;
          basin_record.accepted_as_running_best =
              full_outer_health_acceptable && camera_init_step_finite &&
              frame_objective_improved;
          if (!full_outer_health_acceptable) {
            basin_record.decision_reason = "rejected_full_outer_health";
          } else if (!camera_init_step_finite) {
            basin_record.decision_reason = "rejected_nonfinite_camera_step";
          } else if (!frame_objective_improved) {
            basin_record.decision_reason = compared_as_near_tie
                                               ? "rejected_near_tie_higher_focal"
                                               : "rejected_selection_objective";
          } else if (preferred_by_lower_focal_near_tie_policy) {
            basin_record.decision_reason = "accepted_near_tie_lower_focal";
          } else {
            basin_record.decision_reason = "accepted_lower_objective";
          }
          result.refined_basin_candidates.push_back(basin_record);
          if (!basin_record.accepted_as_running_best) {
		            continue;
		          }
		          if (selected_refined_basin_index >= 0) {
		            result.refined_basin_candidates[
		                static_cast<std::size_t>(selected_refined_basin_index)]
		                .selected = false;
		          }
		          selected_refined_basin_index = basin_record.trial_index;
		          result.refined_basin_candidates.back().selected = true;
	          best_objective = refined_objective;
	          best_lm_selection_objective = combined_lm_selection_objective;
	          best_lm_objective = lm_selection_objective;
	          best_lm_refined = selection_refined;
          best_lm_seed = seed_candidate;
          best_lm_source_label = lm_source_label;
          best_lm_frame_indices = selection_frame_indices;
          best_selected_lm_observations = selection_lm_observations;
          best_selected_pose_success_count = selected_pose_success_count;
          best_selected_pose_failure_count = selected_pose_failure_count;
          best_pose_fit_gate = effective_pose_fit_gate;
          best_selection_diagnostics = selection_diagnostics;
          best_candidate = refined_eval;
          selected_camera = selection_refined.camera;
          selected_source_label = lm_source_label;
          result.stage5_init_multiboard_frame_objective_enabled =
              used_fixed_layout_frame_constraint ? 1 : 0;
          result.stage5_init_fixed_layout_frame_constraint_used =
              used_fixed_layout_frame_constraint ? 1 : 0;
          result.stage5_init_optimizes_layout_variables = 0;
          result.stage5_init_shared_frame_board_constraint_enabled =
              options_.enable_shared_frame_board_constraint ? 1 : 0;
          result.stage5_init_shared_frame_board_constraint_used =
              used_fixed_layout_frame_constraint ? 1 : 0;
          result.stage5_init_uses_layout_to_update_intrinsics =
              used_fixed_layout_frame_constraint ? 1 : 0;
          result.stage5_init_layout_loo_diagnostics_only =
              used_fixed_layout_frame_constraint ? 0 : 1;
          result.stage5_init_shared_layout_board_count =
              shared_layout_board_count;
          result.stage5_init_shared_layout_frame_count =
              shared_layout_frame_count;
          result.stage5_init_shared_layout_observation_count =
              shared_layout_observation_count;
          result.stage5_init_shared_layout_initial_rmse =
              shared_layout_refinement.initial_rmse;
          result.stage5_init_shared_layout_final_rmse =
              shared_layout_refinement.final_rmse;
          result.stage5_init_lm_selection_objective =
              selection_objective_label;
          result.selected_candidate_refined = true;
	          result.stage5_init_shared_focal_effective =
	              options_.shared_focal_during_outer_lm ? 1 : 0;
	          result.stage5_init_shared_focal_released_after_initialization =
	              result.stage5_init_shared_focal_effective;
	          ++lm_seed_accept_count;
        }

        result.stage5_init_refined_basin_candidate_count =
            static_cast<int>(result.refined_basin_candidates.size());
        result.stage5_init_refined_basin_valid_count = 0;
        result.stage5_init_refined_basin_near_tie_count = 0;
        for (const AutoCameraInitializationRefinedBasinCandidate& basin :
             result.refined_basin_candidates) {
          if (basin.full_outer_health_acceptable && basin.camera_step_finite) {
            ++result.stage5_init_refined_basin_valid_count;
          }
          if (basin.compared_as_near_tie) {
            ++result.stage5_init_refined_basin_near_tie_count;
          }
          if (basin.selected) {
            result.stage5_init_selected_basin_seed_rank = basin.seed_rank;
            result.stage5_init_selected_basin_objective =
                basin.combined_selection_objective;
            result.stage5_init_selected_basin_reason = basin.decision_reason;
          }
        }

        if (selected_refined_basin_index >= 0) {
          AutoCameraInitializationRefinedBasinCandidate& selected_basin =
              result.refined_basin_candidates[
                  static_cast<std::size_t>(selected_refined_basin_index)];
          selected_basin.ray_comparison_sample_count = 121;
          selected_basin.ray_rms_deg_to_selected = 0.0;
          selected_basin.distinct_ray_basin_from_selected = false;
        }

        AutoCameraInitializationRefinedBasinCandidate* best_distinct_by_objective =
            nullptr;
        AutoCameraInitializationRefinedBasinCandidate*
            most_distinct_near_optimal = nullptr;
        for (AutoCameraInitializationRefinedBasinCandidate& basin :
             result.refined_basin_candidates) {
          if (!basin.full_outer_health_acceptable ||
              !basin.camera_step_finite || basin.selected) {
            continue;
          }
          if (!ComputeCameraRayRmsDifferenceDeg(
                  basin.refined_camera, selected_camera, result.image_size,
                  &basin.ray_rms_deg_to_selected,
                  &basin.ray_comparison_sample_count)) {
            continue;
          }
          basin.distinct_ray_basin_from_selected =
              basin.ray_rms_deg_to_selected >=
              result.stage5_init_basin_distinct_ray_rms_threshold_deg;
          if (!basin.distinct_ray_basin_from_selected) {
            continue;
          }
          ++result.stage5_init_distinct_refined_basin_candidate_count;
          if (best_distinct_by_objective == nullptr ||
              basin.combined_selection_objective <
                  best_distinct_by_objective->combined_selection_objective) {
            best_distinct_by_objective = &basin;
          }
          if (std::isfinite(result.stage5_init_selected_basin_objective) &&
              std::isfinite(basin.combined_selection_objective)) {
            const double objective_scale = std::max(
                1e-12,
                std::min(
                    std::abs(result.stage5_init_selected_basin_objective),
                    std::abs(basin.combined_selection_objective)));
            const double relative_gap =
                std::abs(basin.combined_selection_objective -
                         result.stage5_init_selected_basin_objective) /
                objective_scale;
            if (relative_gap <=
                    result
                        .stage5_init_basin_ambiguity_relative_objective_threshold &&
                (most_distinct_near_optimal == nullptr ||
                 basin.ray_rms_deg_to_selected >
                     most_distinct_near_optimal->ray_rms_deg_to_selected)) {
              most_distinct_near_optimal = &basin;
            }
          }
        }
        AutoCameraInitializationRefinedBasinCandidate* distinct_runner_up =
            most_distinct_near_optimal != nullptr
                ? most_distinct_near_optimal
                : best_distinct_by_objective;
        result.stage5_init_distinct_basin_alternate_selection =
            most_distinct_near_optimal != nullptr
                ? "maximum_ray_separation_within_relative_objective_threshold"
                : (best_distinct_by_objective != nullptr
                       ? "minimum_objective_distinct_candidate_outside_threshold"
                       : "unavailable");
        if (distinct_runner_up != nullptr &&
            std::isfinite(result.stage5_init_selected_basin_objective) &&
            std::isfinite(
                distinct_runner_up->combined_selection_objective)) {
          result.stage5_init_distinct_basin_runner_up_seed_rank =
              distinct_runner_up->seed_rank;
          result.stage5_init_distinct_basin_runner_up_objective =
              distinct_runner_up->combined_selection_objective;
          result.stage5_init_distinct_basin_ray_rms_deg =
              distinct_runner_up->ray_rms_deg_to_selected;
          const double objective_scale = std::max(
              1e-12,
              std::min(
                  std::abs(result.stage5_init_selected_basin_objective),
                  std::abs(
                      distinct_runner_up->combined_selection_objective)));
          result.stage5_init_distinct_basin_relative_objective_gap =
              std::abs(
                  distinct_runner_up->combined_selection_objective -
                  result.stage5_init_selected_basin_objective) /
              objective_scale;
          result.stage5_init_distinct_basin_ambiguity_detected =
              most_distinct_near_optimal != nullptr ? 1 : 0;
          if (result.stage5_init_distinct_basin_ambiguity_detected != 0) {
            std::ostringstream warning;
            warning
                << "Camera initialization found a geometrically distinct "
                   "near-optimal basin: selected seed rank="
                << result.stage5_init_selected_basin_seed_rank
                << " objective="
                << result.stage5_init_selected_basin_objective
                << ", alternate seed rank="
                << result.stage5_init_distinct_basin_runner_up_seed_rank
                << " objective="
                << result.stage5_init_distinct_basin_runner_up_objective
                << ", relative objective gap="
                << result.stage5_init_distinct_basin_relative_objective_gap
                << ", ray RMS separation="
                << result.stage5_init_distinct_basin_ray_rms_deg
                << " deg. Selection remains objective-driven; the alternate "
                   "basin is reported for external validation.";
            AppendUniqueWarning(warning.str(), &result.warnings);
          }
        }

        if (result.selected_candidate_refined &&
            !best_lm_source_label.empty()) {
          result.stage5_init_selected_pose_success_count =
              best_selected_pose_success_count;
          result.stage5_init_selected_pose_total_count =
              best_selected_pose_success_count +
              best_selected_pose_failure_count;
          result.stage5_init_pose_fit_outlier_gate_applied =
              best_pose_fit_gate.gate_applied ? 1 : 0;
          result.stage5_init_pose_fit_outlier_rejected_count =
              best_pose_fit_gate.rejected_outlier_count;
          result.stage5_init_rescued_outer_observation_pose_gate_rejected_count =
              best_pose_fit_gate.rescued_pose_gate_rejected_count;
          result.stage5_init_pose_fit_outlier_median_rmse =
              best_pose_fit_gate.median_rmse;
          result.stage5_init_pose_fit_outlier_mad_rmse =
              best_pose_fit_gate.mad_rmse;
          result.stage5_init_pose_fit_outlier_threshold_rmse =
              best_pose_fit_gate.threshold_rmse;
          if (result.stage5_init_dense_control_points_enabled == 0) {
            best_selection_diagnostics = EvaluateSelectionInformationAtCamera(
                selected_camera, best_selected_lm_observations,
                options_.selection_scorer);
            result.stage5_init_selection_camera_information_dimension =
                static_cast<int>(best_lm_seed.camera
                                     .CombinedParameterVector()
                                     .size());
            result.stage5_init_selection_camera_information_rank =
                best_selection_diagnostics.camera_rank;
            result.stage5_init_selection_principal_information_rank =
                best_selection_diagnostics.principal_rank;
            result.stage5_init_selection_pose_rank_min =
                best_selection_diagnostics.pose_rank_min;
            result.stage5_init_selection_pose_rank_max =
                best_selection_diagnostics.pose_rank_max;
            result.stage5_init_selection_pose_rank_deficient_count =
                best_selection_diagnostics.pose_rank_deficient_count;
            result.stage5_init_selection_principal_min_eigenvalue =
                best_selection_diagnostics.principal_min_eigenvalue;
            result.stage5_init_selection_principal_max_eigenvalue =
                best_selection_diagnostics.principal_max_eigenvalue;
            result.stage5_init_selection_cu_stddev_px =
                best_selection_diagnostics.cu_stddev_px;
            result.stage5_init_selection_cv_stddev_px =
                best_selection_diagnostics.cv_stddev_px;
            result.stage5_init_selection_weakest_eigenvalue =
                best_selection_diagnostics.weakest_eigenvalue;
            result.stage5_init_selection_weakest_direction =
                FormatWeakestCameraDirection(
                    selected_camera, best_selection_diagnostics);
            result.stage5_init_selection_weakest_principal_fraction =
                best_selection_diagnostics.weakest_principal_fraction;
            result.stage5_init_selection_weakest_focal_fraction =
                best_selection_diagnostics.weakest_focal_fraction;
            result.stage5_init_selection_information_linearization =
                "selected_refined_camera_with_independent_pose_refit";
            result.stage5_init_selection_all_pose_valid_observations_used =
                best_selected_pose_failure_count == 0 &&
                        best_selected_pose_success_count ==
                            eligible_initialization_observation_count
                    ? 1
                    : 0;
          }
          std::set<std::pair<int, int>> used_observation_keys;
          for (const OuterObservationRecord& observation :
               best_selected_lm_observations) {
            used_observation_keys.insert(
                std::make_pair(observation.frame_index,
                               observation.board_id));
          }
          for (const OuterObservationRecord& observation :
               full_health_observations) {
            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            double pose_rmse = std::numeric_limits<double>::quiet_NaN();
            const bool pose_success = EstimatePoseFromObjectPointsStrict(
                best_lm_seed.camera, observation.object_points,
                observation.image_points, kInitializationPoseMaxRmsePx, &pose,
                &pose_rmse);
            AppendBootstrapObservationRecord(
                observation,
                used_observation_keys.count(
                    std::make_pair(observation.frame_index,
                                   observation.board_id)) > 0,
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
	          result.lm_robust_loss_enabled =
	              best_lm_refined.robust_loss_delta_pixels > 0.0 ? 1 : 0;
	          result.lm_robust_loss_type =
	              result.lm_robust_loss_enabled != 0
	                  ? "point_norm_huber_irls"
	                  : "none";
	          result.lm_robust_loss_delta_pixels =
	              best_lm_refined.robust_loss_delta_pixels;
	          result.lm_initial_robust_rmse =
	              best_lm_refined.initial_robust_rmse;
	          result.lm_final_robust_rmse =
	              best_lm_refined.final_robust_rmse;
	          result.lm_initial_downweighted_point_count =
	              best_lm_refined.initial_downweighted_point_count;
	          result.lm_final_downweighted_point_count =
	              best_lm_refined.final_downweighted_point_count;
          std::ostringstream selection_warning;
          selection_warning
	              << (result.stage5_init_dense_control_points_enabled != 0
                      ? "Dense-grid multi-start selected-view "
                      : "Outer-only multi-start selected-observation ")
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
                                    full_health_observations);
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
                                  full_health_observations);
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

  if (options_.enable_principal_profile && result.success &&
      result.selected_camera.IsValid()) {
    const PoseFitOutlierGateResult profile_pose_fit_gate =
        FilterPoseFitOutliersForInitialization(result.selected_camera,
                                               all_observations,
                                               options_.gate_rescued_outer_observations,
                                               options_.rescued_outer_observation_pose_rmse_gate_pixels);
    PopulatePrincipalProfile(
        result.selected_camera,
        profile_pose_fit_gate.accepted_observations,
        options_.use_direct_dense_control_points
            ? options_.dense_grid_lm_huber_delta_pixels
            : 0.0,
        options_.principal_profile_radius_px,
        &result);
  }
  if (options_.enable_fixed_layout_diagnostic && result.success &&
      result.selected_camera.IsValid()) {
    PopulateFixedLayoutDiagnostic(result.selected_camera,
                                  frames,
                                  all_observations,
                                  config_,
                                  options_.reference_board_id,
                                  options_.use_direct_dense_control_points
                                      ? options_.dense_grid_lm_huber_delta_pixels
                                      : 0.0,
                                  options_.enable_principal_profile,
                                  options_.principal_profile_radius_px,
                                  &result);
  }
  if (options_.enable_board_jackknife_diagnostic && result.success &&
      result.selected_camera.IsValid()) {
    PopulateBoardJackknifeDiagnostic(
        result.selected_camera,
        all_observations,
        options_.use_direct_dense_control_points
            ? options_.dense_grid_lm_huber_delta_pixels
            : 0.0,
        &result);
  }
  if (options_.enable_coverage_weighted_diagnostic && result.success &&
      result.selected_camera.IsValid()) {
    PopulateCoverageWeightedDiagnostic(
        result.selected_camera,
        all_observations,
        options_.use_direct_dense_control_points
            ? options_.dense_grid_lm_huber_delta_pixels
            : 0.0,
        &result);
  }
  if (result.success && result.selected_camera.IsValid()) {
    result.stage5_init_selected_fu_minus_fv =
        result.selected_camera.fu - result.selected_camera.fv;
    PopulatePoseExcitationDiagnostic(result.selected_camera,
                                     all_observations,
                                     &result);
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
  output << "stage5_init_shared_focal_requested: "
         << result.stage5_init_shared_focal_requested << "\n";
  output << "stage5_init_shared_focal_effective: "
         << result.stage5_init_shared_focal_effective << "\n";
  output << "stage5_init_shared_focal_released_after_initialization: "
         << result.stage5_init_shared_focal_released_after_initialization << "\n";
  output << "stage5_init_selected_fu_minus_fv: "
         << result.stage5_init_selected_fu_minus_fv << "\n";
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
  output << "stage5_init_near_tie_lower_focal_policy_enabled: "
         << result.stage5_init_near_tie_lower_focal_policy_enabled << "\n";
  output << "stage5_init_near_tie_relative_objective_tolerance: "
         << result.stage5_init_near_tie_relative_objective_tolerance << "\n";
  output << "stage5_init_refined_basin_candidate_count: "
         << result.stage5_init_refined_basin_candidate_count << "\n";
  output << "stage5_init_refined_basin_valid_count: "
         << result.stage5_init_refined_basin_valid_count << "\n";
  output << "stage5_init_refined_basin_near_tie_count: "
         << result.stage5_init_refined_basin_near_tie_count << "\n";
  output << "stage5_init_selected_basin_seed_rank: "
         << result.stage5_init_selected_basin_seed_rank << "\n";
  output << "stage5_init_selected_basin_objective: "
         << result.stage5_init_selected_basin_objective << "\n";
  output << "stage5_init_selected_basin_reason: "
         << result.stage5_init_selected_basin_reason << "\n";
  output << "stage5_init_basin_distinct_ray_rms_threshold_deg: "
         << result.stage5_init_basin_distinct_ray_rms_threshold_deg << "\n";
  output << "stage5_init_basin_ambiguity_relative_objective_threshold: "
         << result.stage5_init_basin_ambiguity_relative_objective_threshold
         << "\n";
  output << "stage5_init_distinct_refined_basin_candidate_count: "
         << result.stage5_init_distinct_refined_basin_candidate_count << "\n";
  output << "stage5_init_distinct_basin_runner_up_seed_rank: "
         << result.stage5_init_distinct_basin_runner_up_seed_rank << "\n";
  output << "stage5_init_distinct_basin_alternate_selection: "
         << result.stage5_init_distinct_basin_alternate_selection << "\n";
  output << "stage5_init_distinct_basin_runner_up_objective: "
         << result.stage5_init_distinct_basin_runner_up_objective << "\n";
  output << "stage5_init_distinct_basin_relative_objective_gap: "
         << result.stage5_init_distinct_basin_relative_objective_gap << "\n";
  output << "stage5_init_distinct_basin_ray_rms_deg: "
         << result.stage5_init_distinct_basin_ray_rms_deg << "\n";
  output << "stage5_init_distinct_basin_ambiguity_detected: "
         << result.stage5_init_distinct_basin_ambiguity_detected << "\n";
  output << "stage5_init_kb_focal_source: "
         << result.stage5_init_kb_focal_source << "\n";
  output << "stage5_init_kb_row_circle_focal_available: "
         << result.stage5_init_kb_row_circle_focal_available << "\n";
  output << "stage5_init_kb_zero_distortion_seed: "
         << result.stage5_init_kb_zero_distortion_seed << "\n";
  output << "stage5_init_kb_zero_distortion_seed_included: "
         << result.stage5_init_kb_zero_distortion_seed_included << "\n";
  output << "stage5_init_kb_nonzero_distortion_seed_count: "
         << result.stage5_init_kb_nonzero_distortion_seed_count << "\n";
  output << "stage5_init_kb_distortion_released_in_lm: "
         << result.stage5_init_kb_distortion_released_in_lm << "\n";
  output << "stage5_init_kb_distortion_fixed_zero_in_init_lm: "
         << result.stage5_init_kb_distortion_fixed_zero_in_init_lm << "\n";
  output << "stage5_init_kb_multistart_enabled: "
         << result.stage5_init_kb_multistart_enabled << "\n";
  output << "stage5_init_ucm_seed_source: "
         << result.stage5_init_ucm_seed_source << "\n";
  output << "stage5_init_ucm_omni_gamma_available: "
         << result.stage5_init_ucm_omni_gamma_available << "\n";
  output << "stage5_init_ucm_mapping: "
         << result.stage5_init_ucm_mapping << "\n";
  output << "stage5_init_ucm_mapping_verified_against_kalibr_source: "
         << result.stage5_init_ucm_mapping_verified_against_kalibr_source
         << "\n";
  output << "stage5_init_ucm_multistart_enabled: "
         << result.stage5_init_ucm_multistart_enabled << "\n";
  output << "stage5_init_ucm_shape_released_in_lm: "
         << result.stage5_init_ucm_shape_released_in_lm << "\n";
  output << "stage5_init_uses_yaml_intrinsics: "
         << result.stage5_init_uses_yaml_intrinsics << "\n";
  output << "stage5_init_uses_kalibr_camchain_intrinsics: "
         << result.stage5_init_uses_kalibr_camchain_intrinsics << "\n";
  output << "stage5_init_outer_only: " << result.stage5_init_outer_only << "\n";
  output << "stage5_init_dense_control_points_enabled: "
         << result.stage5_init_dense_control_points_enabled << "\n";
  output << "stage5_init_dense_control_points_scope: "
         << result.stage5_init_dense_control_points_scope << "\n";
  output << "stage5_init_dense_control_point_count: "
         << result.stage5_init_dense_control_point_count << "\n";
  output << "stage5_init_primary_frame_count: "
         << result.stage5_init_primary_frame_count << "\n";
  output << "stage5_init_auxiliary_session_count: "
         << result.stage5_init_auxiliary_session_count << "\n";
  output << "stage5_init_auxiliary_frame_count: "
         << result.stage5_init_auxiliary_frame_count << "\n";
  output << "stage5_init_uses_auxiliary_sessions: "
         << result.stage5_init_uses_auxiliary_sessions << "\n";
  output << "stage5_init_dense_internal_point_count: "
         << result.bootstrap_internal_points_used << "\n";
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
  output << "stage5_init_shared_frame_board_constraint_enabled: "
         << result.stage5_init_shared_frame_board_constraint_enabled << "\n";
  output << "stage5_init_shared_frame_board_constraint_used: "
         << result.stage5_init_shared_frame_board_constraint_used << "\n";
  output << "stage5_init_shared_layout_board_count: "
         << result.stage5_init_shared_layout_board_count << "\n";
  output << "stage5_init_shared_layout_frame_count: "
         << result.stage5_init_shared_layout_frame_count << "\n";
  output << "stage5_init_shared_layout_observation_count: "
         << result.stage5_init_shared_layout_observation_count << "\n";
  output << "stage5_init_shared_layout_initial_rmse: "
         << result.stage5_init_shared_layout_initial_rmse << "\n";
  output << "stage5_init_shared_layout_final_rmse: "
         << result.stage5_init_shared_layout_final_rmse << "\n";
  output << "stage5_init_lm_selection_objective: "
         << result.stage5_init_lm_selection_objective << "\n";
  output << "stage5_init_lm_min_relative_objective_improvement: "
         << result.stage5_init_lm_min_relative_objective_improvement << "\n";
  output << "stage5_init_selection_prefilter: "
         << result.stage5_init_selection_prefilter << "\n";
  output << "stage5_init_selection_scorer: "
         << result.stage5_init_selection_scorer << "\n";
  output << "stage5_init_selection_uses_information_metric: "
         << result.stage5_init_selection_uses_information_metric << "\n";
  output << "stage5_init_selection_is_exact_kalibr_information_theoretic: "
         << result.stage5_init_selection_is_exact_kalibr_information_theoretic
         << "\n";
  output << "stage5_init_selection_pose_marginalized: "
         << result.stage5_init_selection_pose_marginalized << "\n";
  output << "stage5_init_selection_principal_subspace_aware: "
         << result.stage5_init_selection_principal_subspace_aware << "\n";
  output << "stage5_init_selection_camera_information_dimension: "
         << result.stage5_init_selection_camera_information_dimension << "\n";
  output << "stage5_init_selection_camera_information_rank: "
         << result.stage5_init_selection_camera_information_rank << "\n";
  output << "stage5_init_selection_principal_information_rank: "
         << result.stage5_init_selection_principal_information_rank << "\n";
  output << "stage5_init_selection_pose_rank_min: "
         << result.stage5_init_selection_pose_rank_min << "\n";
  output << "stage5_init_selection_pose_rank_max: "
         << result.stage5_init_selection_pose_rank_max << "\n";
  output << "stage5_init_selection_pose_rank_deficient_count: "
         << result.stage5_init_selection_pose_rank_deficient_count << "\n";
  output << "stage5_init_selection_principal_min_eigenvalue: "
         << result.stage5_init_selection_principal_min_eigenvalue << "\n";
  output << "stage5_init_selection_principal_max_eigenvalue: "
         << result.stage5_init_selection_principal_max_eigenvalue << "\n";
  output << "stage5_init_selection_cu_stddev_px: "
         << result.stage5_init_selection_cu_stddev_px << "\n";
  output << "stage5_init_selection_cv_stddev_px: "
         << result.stage5_init_selection_cv_stddev_px << "\n";
  output << "stage5_init_selection_weakest_eigenvalue: "
         << result.stage5_init_selection_weakest_eigenvalue << "\n";
  output << "stage5_init_selection_weakest_direction: "
         << result.stage5_init_selection_weakest_direction << "\n";
  output << "stage5_init_selection_weakest_principal_fraction: "
         << result.stage5_init_selection_weakest_principal_fraction << "\n";
  output << "stage5_init_selection_weakest_focal_fraction: "
         << result.stage5_init_selection_weakest_focal_fraction << "\n";
  output << "stage5_init_selection_information_linearization: "
         << result.stage5_init_selection_information_linearization << "\n";
  output << "stage5_init_principal_profile_enabled: "
         << result.stage5_init_principal_profile_enabled << "\n";
  output << "stage5_init_principal_profile_radius_px: "
         << result.stage5_init_principal_profile_radius_px << "\n";
  output << "stage5_init_principal_profile_observation_count: "
         << result.stage5_init_principal_profile_observation_count << "\n";
  output << "stage5_init_principal_profile_sample_count: "
         << result.stage5_init_principal_profile_sample_count << "\n";
  output << "stage5_init_principal_profile_comparable_sample_count: "
         << result.stage5_init_principal_profile_comparable_sample_count
         << "\n";
  output << "stage5_init_principal_profile_best_delta_cu_px: "
         << result.stage5_init_principal_profile_best_delta_cu_px << "\n";
  output << "stage5_init_principal_profile_best_delta_cv_px: "
         << result.stage5_init_principal_profile_best_delta_cv_px << "\n";
  output << "stage5_init_principal_profile_best_delta_robust_cost: "
         << result.stage5_init_principal_profile_best_delta_robust_cost
         << "\n";
  output << "stage5_init_fixed_layout_diagnostic_enabled: "
         << result.stage5_init_fixed_layout_diagnostic_enabled << "\n";
  output << "stage5_init_fixed_layout_diagnostic_updates_selected_intrinsics: "
         << result.stage5_init_fixed_layout_diagnostic_updates_selected_intrinsics
         << "\n";
  output << "stage5_init_fixed_layout_diagnostic_layout_source: "
         << result.stage5_init_fixed_layout_diagnostic_layout_source << "\n";
  output << "stage5_init_fixed_layout_diagnostic_layout_success: "
         << result.stage5_init_fixed_layout_diagnostic_layout_success << "\n";
  output << "stage5_init_fixed_layout_diagnostic_board_count: "
         << result.stage5_init_fixed_layout_diagnostic_board_count << "\n";
  output << "stage5_init_fixed_layout_diagnostic_frame_count: "
         << result.stage5_init_fixed_layout_diagnostic_frame_count << "\n";
  output << "stage5_init_fixed_layout_diagnostic_board_observation_count: "
         << result.stage5_init_fixed_layout_diagnostic_board_observation_count
         << "\n";
  output << "stage5_init_fixed_layout_diagnostic_iteration_count: "
         << result.stage5_init_fixed_layout_diagnostic_iteration_count << "\n";
  output << "stage5_init_fixed_layout_diagnostic_layout_bootstrap_rmse: "
         << result.stage5_init_fixed_layout_diagnostic_layout_bootstrap_rmse
         << "\n";
  output << "stage5_init_fixed_layout_diagnostic_initial_rmse: "
         << result.stage5_init_fixed_layout_diagnostic_initial_rmse << "\n";
  output << "stage5_init_fixed_layout_diagnostic_final_rmse: "
         << result.stage5_init_fixed_layout_diagnostic_final_rmse << "\n";
  output << "stage5_init_fixed_layout_diagnostic_rig_axis_balance_ratio: "
         << result.stage5_init_fixed_layout_diagnostic_rig_axis_balance_ratio
         << "\n";
  output
      << "stage5_init_fixed_layout_diagnostic_rig_dominant_axis_angle_deg: "
      << result
             .stage5_init_fixed_layout_diagnostic_rig_dominant_axis_angle_deg
      << "\n";
  output << "stage5_init_fixed_layout_principal_profile_enabled: "
         << result.stage5_init_fixed_layout_principal_profile_enabled << "\n";
  output << "stage5_init_fixed_layout_principal_profile_radius_px: "
         << result.stage5_init_fixed_layout_principal_profile_radius_px << "\n";
  output << "stage5_init_fixed_layout_principal_profile_sample_count: "
         << result.stage5_init_fixed_layout_principal_profile_sample_count
         << "\n";
  output
      << "stage5_init_fixed_layout_principal_profile_comparable_sample_count: "
      << result
             .stage5_init_fixed_layout_principal_profile_comparable_sample_count
      << "\n";
  output << "stage5_init_fixed_layout_principal_profile_best_delta_cu_px: "
         << result.stage5_init_fixed_layout_principal_profile_best_delta_cu_px
         << "\n";
  output << "stage5_init_fixed_layout_principal_profile_best_delta_cv_px: "
         << result.stage5_init_fixed_layout_principal_profile_best_delta_cv_px
         << "\n";
  output
      << "stage5_init_fixed_layout_principal_profile_best_delta_robust_cost: "
      << result
             .stage5_init_fixed_layout_principal_profile_best_delta_robust_cost
      << "\n";
  output << "stage5_init_fixed_layout_diagnostic_xi: "
         << result.stage5_init_fixed_layout_diagnostic_camera.xi << "\n";
  output << "stage5_init_fixed_layout_diagnostic_alpha: "
         << result.stage5_init_fixed_layout_diagnostic_camera.alpha << "\n";
  output << "stage5_init_fixed_layout_diagnostic_fu: "
         << result.stage5_init_fixed_layout_diagnostic_camera.fu << "\n";
  output << "stage5_init_fixed_layout_diagnostic_fv: "
         << result.stage5_init_fixed_layout_diagnostic_camera.fv << "\n";
  output << "stage5_init_fixed_layout_diagnostic_cu: "
         << result.stage5_init_fixed_layout_diagnostic_camera.cu << "\n";
  output << "stage5_init_fixed_layout_diagnostic_cv: "
         << result.stage5_init_fixed_layout_diagnostic_camera.cv << "\n";
  output << "stage5_init_board_jackknife_diagnostic_enabled: "
         << result.stage5_init_board_jackknife_diagnostic_enabled << "\n";
  output << "stage5_init_board_jackknife_diagnostic_updates_selected_intrinsics: "
         << result.stage5_init_board_jackknife_diagnostic_updates_selected_intrinsics
         << "\n";
  output << "stage5_init_board_jackknife_diagnostic_sample_count: "
         << result.stage5_init_board_jackknife_diagnostic_sample_count << "\n";
  output << "stage5_init_board_jackknife_diagnostic_comparable_sample_count: "
         << result.stage5_init_board_jackknife_diagnostic_comparable_sample_count
         << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_enabled: "
         << result.stage5_init_coverage_weighted_diagnostic_enabled << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_updates_selected_intrinsics: "
         << result.stage5_init_coverage_weighted_diagnostic_updates_selected_intrinsics
         << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_grid_rows: "
         << result.stage5_init_coverage_weighted_diagnostic_grid_rows << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_grid_cols: "
         << result.stage5_init_coverage_weighted_diagnostic_grid_cols << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_occupied_bin_count: "
         << result.stage5_init_coverage_weighted_diagnostic_occupied_bin_count
         << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_min_weight: "
         << result.stage5_init_coverage_weighted_diagnostic_min_weight << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_max_weight: "
         << result.stage5_init_coverage_weighted_diagnostic_max_weight << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_initial_rmse: "
         << result.stage5_init_coverage_weighted_diagnostic_initial_rmse
         << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_final_rmse: "
         << result.stage5_init_coverage_weighted_diagnostic_final_rmse << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_xi: "
         << result.stage5_init_coverage_weighted_diagnostic_camera.xi << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_alpha: "
         << result.stage5_init_coverage_weighted_diagnostic_camera.alpha << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_fu: "
         << result.stage5_init_coverage_weighted_diagnostic_camera.fu << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_fv: "
         << result.stage5_init_coverage_weighted_diagnostic_camera.fv << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_cu: "
         << result.stage5_init_coverage_weighted_diagnostic_camera.cu << "\n";
  output << "stage5_init_coverage_weighted_diagnostic_cv: "
         << result.stage5_init_coverage_weighted_diagnostic_camera.cv << "\n";
  output << "stage5_init_pose_excitation_diagnostic_enabled: "
         << result.stage5_init_pose_excitation_diagnostic_enabled << "\n";
  output << "stage5_init_pose_excitation_diagnostic_updates_selected_intrinsics: "
         << result.stage5_init_pose_excitation_diagnostic_updates_selected_intrinsics
         << "\n";
  output << "stage5_init_pose_excitation_board_count: "
         << result.stage5_init_pose_excitation_board_count << "\n";
  output << "stage5_init_pose_excitation_pose_success_count: "
         << result.stage5_init_pose_excitation_pose_success_count << "\n";
  output << "stage5_init_pose_excitation_pose_total_count: "
         << result.stage5_init_pose_excitation_pose_total_count << "\n";
  output << "stage5_init_pose_excitation_min_board_normal_p95_deg: "
         << result.stage5_init_pose_excitation_min_board_normal_p95_deg << "\n";
  output << "stage5_init_pose_excitation_median_board_normal_p95_deg: "
         << result.stage5_init_pose_excitation_median_board_normal_p95_deg
         << "\n";
  output << "stage5_init_pose_excitation_max_board_normal_p95_deg: "
         << result.stage5_init_pose_excitation_max_board_normal_p95_deg << "\n";
  output << "stage5_init_pose_excitation_min_board_tilt_range_deg: "
         << result.stage5_init_pose_excitation_min_board_tilt_range_deg << "\n";
  output << "stage5_init_pose_excitation_median_board_tilt_range_deg: "
         << result.stage5_init_pose_excitation_median_board_tilt_range_deg
         << "\n";
  output
      << "stage5_init_pose_excitation_min_normal_xy_axis_balance_ratio: "
      << result
             .stage5_init_pose_excitation_min_normal_xy_axis_balance_ratio
      << "\n";
  output
      << "stage5_init_pose_excitation_median_normal_xy_axis_balance_ratio: "
      << result
             .stage5_init_pose_excitation_median_normal_xy_axis_balance_ratio
      << "\n";
  output
      << "stage5_init_pose_excitation_max_normal_xy_axis_balance_ratio: "
      << result
             .stage5_init_pose_excitation_max_normal_xy_axis_balance_ratio
      << "\n";
  output << "stage5_init_pose_excitation_global_normal_xy_std_x: "
         << result.stage5_init_pose_excitation_global_normal_xy_std_x << "\n";
  output << "stage5_init_pose_excitation_global_normal_xy_std_y: "
         << result.stage5_init_pose_excitation_global_normal_xy_std_y << "\n";
  output
      << "stage5_init_pose_excitation_global_normal_xy_weak_variance: "
      << result
             .stage5_init_pose_excitation_global_normal_xy_weak_variance
      << "\n";
  output
      << "stage5_init_pose_excitation_global_normal_xy_strong_variance: "
      << result
             .stage5_init_pose_excitation_global_normal_xy_strong_variance
      << "\n";
  output
      << "stage5_init_pose_excitation_global_normal_xy_axis_balance_ratio: "
      << result
             .stage5_init_pose_excitation_global_normal_xy_axis_balance_ratio
      << "\n";
  output
      << "stage5_init_pose_excitation_global_normal_xy_dominant_axis_angle_deg: "
      << result
             .stage5_init_pose_excitation_global_normal_xy_dominant_axis_angle_deg
      << "\n";
  output << "stage5_init_pose_excitation_single_axis_board_count: "
         << result.stage5_init_pose_excitation_single_axis_board_count << "\n";
  output
      << "stage5_init_pose_excitation_principal_pseudo_observability_warning: "
      << result
             .stage5_init_pose_excitation_principal_pseudo_observability_warning
      << "\n";
  output << "stage5_init_pose_excitation_assessment: "
         << result.stage5_init_pose_excitation_assessment << "\n";
  output << "stage5_init_selection_all_pose_valid_observations_used: "
         << result.stage5_init_selection_all_pose_valid_observations_used
         << "\n";
  output << "stage5_init_calibrate_intrinsics_enabled: "
         << result.stage5_init_calibrate_intrinsics_enabled << "\n";
  output << "stage5_init_calibrate_intrinsics_released_params: "
         << result.stage5_init_calibrate_intrinsics_released_params << "\n";
  output << "stage5_init_calibrate_intrinsics_optimizer: "
         << result.stage5_init_calibrate_intrinsics_optimizer << "\n";
  output << "stage5_init_pose_fit_outlier_gate_enabled: "
         << result.stage5_init_pose_fit_outlier_gate_enabled << "\n";
  output << "stage5_init_pose_fit_outlier_gate_applied: "
         << result.stage5_init_pose_fit_outlier_gate_applied << "\n";
  output << "stage5_init_pose_fit_outlier_rejected_count: "
         << result.stage5_init_pose_fit_outlier_rejected_count << "\n";
  output << "stage5_init_rescued_outer_observation_count: "
         << result.stage5_init_rescued_outer_observation_count << "\n";
  output << "stage5_init_reference_board_id: "
         << result.stage5_init_reference_board_id << "\n";
  output << "stage5_init_rescued_outer_observation_pose_gate_rejected_count: "
         << result.stage5_init_rescued_outer_observation_pose_gate_rejected_count
         << "\n";
  output << "stage5_init_rescued_outer_observation_lm_weight: "
         << result.stage5_init_rescued_outer_observation_lm_weight << "\n";
  output << "stage5_init_pose_fit_outlier_median_rmse: "
         << result.stage5_init_pose_fit_outlier_median_rmse << "\n";
  output << "stage5_init_pose_fit_outlier_mad_rmse: "
         << result.stage5_init_pose_fit_outlier_mad_rmse << "\n";
  output << "stage5_init_pose_fit_outlier_threshold_rmse: "
         << result.stage5_init_pose_fit_outlier_threshold_rmse << "\n";
  output << "stage5_init_requires_all_configured_boards_per_frame: "
         << result.stage5_init_requires_all_configured_boards_per_frame
         << "\n";
  output << "stage5_init_required_board_ids: "
         << result.stage5_init_required_board_ids << "\n";
  output << "stage5_init_required_board_count: "
         << result.stage5_init_required_board_count << "\n";
  output << "stage5_init_input_frame_count: "
         << result.stage5_init_input_frame_count << "\n";
  output << "stage5_init_complete_board_frame_count: "
         << result.stage5_init_complete_board_frame_count << "\n";
  output << "stage5_init_incomplete_board_frame_rejected_count: "
         << result.stage5_init_incomplete_board_frame_rejected_count << "\n";
  output << "stage5_init_observation_count_before_complete_frame_filter: "
         << result.stage5_init_observation_count_before_complete_frame_filter
         << "\n";
  output << "stage5_init_observation_count_after_complete_frame_filter: "
         << result.stage5_init_observation_count_after_complete_frame_filter
         << "\n";
  output << "stage5_init_camera_evaluation_observation_count: "
         << result.stage5_init_camera_evaluation_observation_count << "\n";
  output << "stage5_init_all_valid_outer_observations_used: "
         << result.stage5_init_all_valid_outer_observations_used << "\n";
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
  output << "stage5_init_lm_robust_loss_enabled: "
         << result.lm_robust_loss_enabled << "\n";
  output << "stage5_init_lm_robust_loss_type: "
         << result.lm_robust_loss_type << "\n";
  output << "stage5_init_lm_robust_loss_delta_pixels: "
         << result.lm_robust_loss_delta_pixels << "\n";
  output << "stage5_init_lm_initial_robust_rmse: "
         << result.lm_initial_robust_rmse << "\n";
  output << "stage5_init_lm_final_robust_rmse: "
         << result.lm_final_robust_rmse << "\n";
  output << "stage5_init_lm_initial_downweighted_point_count: "
         << result.lm_initial_downweighted_point_count << "\n";
  output << "stage5_init_lm_final_downweighted_point_count: "
         << result.lm_final_downweighted_point_count << "\n";
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
         << "mean_observation_rmse,robust_observation_rmse,"
         << "median_observation_rmse,p95_observation_rmse,max_observation_rmse,"
         << "worst_observation_frame_index,worst_observation_board_id,"
         << "projection_failure_count,leave_one_board_out_attempt_count,"
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
           << candidate.robust_observation_rmse << ","
           << candidate.median_observation_rmse << ","
           << candidate.p95_observation_rmse << ","
           << candidate.max_observation_rmse << ","
           << candidate.worst_observation_frame_index << ","
           << candidate.worst_observation_board_id << ","
           << candidate.projection_failure_count << ","
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

void WriteAutoCameraInitializationRefinedBasinsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output
      << "trial_index,seed_rank,seed_source_label,camera_model_family,"
      << "seed_xi,seed_alpha,seed_beta,seed_fu,seed_fv,seed_cu,seed_cv,"
      << "refined_xi,refined_alpha,refined_beta,refined_fu,refined_fv,"
      << "refined_cu,refined_cv,selected_frame_count,"
      << "selected_board_observation_count,residual_count,iteration_count,"
      << "seed_objective,full_outer_objective,lm_initial_rmse,lm_final_rmse,"
      << "lm_final_robust_rmse,shared_layout_constraint_used,"
      << "shared_layout_frame_count,shared_layout_board_count,"
      << "shared_layout_observation_count,shared_layout_initial_rmse,"
      << "shared_layout_final_rmse,shared_layout_final_robust_rmse,"
      << "full_outer_robust_rmse,full_outer_median_rmse,full_outer_p95_rmse,"
      << "full_outer_pose_success_count,full_outer_pose_failure_count,"
      << "full_outer_max_rmse,full_outer_worst_frame_index,"
      << "full_outer_worst_board_id,full_outer_projection_failure_count,"
      << "combined_selection_objective,"
      << "full_outer_health_acceptable,camera_step_finite,"
      << "objective_improved_before_near_tie_policy,compared_as_near_tie,"
      << "preferred_by_lower_focal_near_tie_policy,"
      << "accepted_as_running_best,selected,ray_comparison_sample_count,"
      << "ray_rms_deg_to_selected,distinct_ray_basin_from_selected,"
      << "decision_reason\n";
  for (const AutoCameraInitializationRefinedBasinCandidate& basin :
       result.refined_basin_candidates) {
    output << basin.trial_index << "," << basin.seed_rank << ","
           << basin.seed_source_label << ","
           << basin.refined_camera.NormalizedFamilyString() << ","
           << basin.seed_camera.xi << "," << basin.seed_camera.alpha << ","
           << basin.seed_camera.beta << "," << basin.seed_camera.fu << ","
           << basin.seed_camera.fv << "," << basin.seed_camera.cu << ","
           << basin.seed_camera.cv << "," << basin.refined_camera.xi << ","
           << basin.refined_camera.alpha << ","
           << basin.refined_camera.beta << ","
           << basin.refined_camera.fu << "," << basin.refined_camera.fv
           << "," << basin.refined_camera.cu << ","
           << basin.refined_camera.cv << "," << basin.selected_frame_count
           << "," << basin.selected_board_observation_count << ","
           << basin.residual_count << "," << basin.iteration_count << ","
           << basin.seed_objective << "," << basin.full_outer_objective
           << "," << basin.lm_initial_rmse << "," << basin.lm_final_rmse
           << "," << basin.lm_final_robust_rmse << ","
           << basin.shared_layout_constraint_used << ","
           << basin.shared_layout_frame_count << ","
           << basin.shared_layout_board_count << ","
           << basin.shared_layout_observation_count << ","
           << basin.shared_layout_initial_rmse << ","
           << basin.shared_layout_final_rmse << ","
           << basin.shared_layout_final_robust_rmse << ","
           << basin.full_outer_robust_rmse << ","
           << basin.full_outer_median_rmse << ","
           << basin.full_outer_p95_rmse << ","
           << basin.full_outer_pose_success_count << ","
           << basin.full_outer_pose_failure_count << ","
           << basin.full_outer_max_rmse << ","
           << basin.full_outer_worst_frame_index << ","
           << basin.full_outer_worst_board_id << ","
           << basin.full_outer_projection_failure_count << ","
           << basin.combined_selection_objective << ","
           << (basin.full_outer_health_acceptable ? 1 : 0) << ","
           << (basin.camera_step_finite ? 1 : 0) << ","
           << (basin.objective_improved_before_near_tie_policy ? 1 : 0)
           << "," << (basin.compared_as_near_tie ? 1 : 0) << ","
           << (basin.preferred_by_lower_focal_near_tie_policy ? 1 : 0)
           << "," << (basin.accepted_as_running_best ? 1 : 0) << ","
           << (basin.selected ? 1 : 0) << ","
           << basin.ray_comparison_sample_count << ","
           << basin.ray_rms_deg_to_selected << ","
           << (basin.distinct_ray_basin_from_selected ? 1 : 0) << ","
           << basin.decision_reason << "\n";
  }
}

void WriteAutoCameraInitializationOuterResidualsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "source_label,frame_index,frame_label,board_id,quality,"
         << "used_local_patch_rescue,pose_success,"
         << "pose_fit_outer_rmse,failure_reason\n";
  for (const AutoCameraInitializationResidual& residual : result.selected_residuals) {
    output << residual.source_label << ","
           << residual.frame_index << ","
           << residual.frame_label << ","
           << residual.board_id << ","
           << residual.quality << ","
           << (residual.used_local_patch_rescue ? 1 : 0) << ","
           << (residual.pose_success ? 1 : 0) << ","
           << residual.pose_fit_outer_rmse << ","
           << residual.failure_reason << "\n";
  }
}

void WriteAutoCameraInitializationBootstrapViewsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "frame_index,frame_label,board_id,used_in_lm,"
         << "used_local_patch_rescue,pose_init_success,"
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
             << (observation.used_local_patch_rescue ? 1 : 0) << ","
             << (observation.pose_init_success ? 1 : 0) << ","
             << observation.pose_fit_outer_rmse << ","
             << corner_index << ","
             << corner.x() << ","
             << corner.y() << "\n";
    }
  }
}

void WriteAutoCameraInitializationPrincipalProfileCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "delta_cu_px,delta_cv_px,fixed_cu,fixed_cv,comparable,"
            "expected_view_count,optimized_view_count,residual_count,"
            "iterations,final_rmse,final_robust_rmse,final_robust_cost,"
            "delta_robust_cost,camera_model,distortion_model,xi,alpha,beta,"
            "fu,fv,cu,cv\n";
  for (const AutoCameraInitializationPrincipalProfileSample& sample :
       result.principal_profile_samples) {
    output << sample.delta_cu_px << ","
           << sample.delta_cv_px << ","
           << sample.fixed_cu << ","
           << sample.fixed_cv << ","
           << (sample.comparable ? 1 : 0) << ","
           << sample.expected_view_count << ","
           << sample.optimized_view_count << ","
           << sample.residual_count << ","
           << sample.iteration_count << ","
           << sample.final_rmse << ","
           << sample.final_robust_rmse << ","
           << sample.final_robust_cost << ","
           << sample.delta_robust_cost << ","
           << sample.optimized_camera.NormalizedCameraModel() << ","
           << sample.optimized_camera.NormalizedDistortionModel() << ","
           << sample.optimized_camera.xi << ","
           << sample.optimized_camera.alpha << ","
           << sample.optimized_camera.beta << ","
           << sample.optimized_camera.fu << ","
           << sample.optimized_camera.fv << ","
           << sample.optimized_camera.cu << ","
           << sample.optimized_camera.cv << "\n";
  }
}

void WriteAutoCameraInitializationFixedLayoutPrincipalProfileCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "delta_cu_px,delta_cv_px,fixed_cu,fixed_cv,comparable,"
            "expected_board_observation_count,"
            "optimized_board_observation_count,residual_count,iterations,"
            "final_rmse,final_robust_rmse,final_robust_cost,"
            "delta_robust_cost,camera_model,distortion_model,xi,alpha,beta,"
            "fu,fv,cu,cv\n";
  for (const AutoCameraInitializationPrincipalProfileSample& sample :
       result.fixed_layout_principal_profile_samples) {
    output << sample.delta_cu_px << ","
           << sample.delta_cv_px << ","
           << sample.fixed_cu << ","
           << sample.fixed_cv << ","
           << (sample.comparable ? 1 : 0) << ","
           << sample.expected_view_count << ","
           << sample.optimized_view_count << ","
           << sample.residual_count << ","
           << sample.iteration_count << ","
           << sample.final_rmse << ","
           << sample.final_robust_rmse << ","
           << sample.final_robust_cost << ","
           << sample.delta_robust_cost << ","
           << sample.optimized_camera.NormalizedCameraModel() << ","
           << sample.optimized_camera.NormalizedDistortionModel() << ","
           << sample.optimized_camera.xi << ","
           << sample.optimized_camera.alpha << ","
           << sample.optimized_camera.beta << ","
           << sample.optimized_camera.fu << ","
           << sample.optimized_camera.fv << ","
           << sample.optimized_camera.cu << ","
           << sample.optimized_camera.cv << "\n";
  }
}

void WriteAutoCameraInitializationBoardJackknifeCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "excluded_board_id,comparable,expected_view_count,"
            "optimized_view_count,residual_count,iterations,final_rmse,"
            "delta_xi,delta_alpha,delta_fu,delta_fv,delta_cu,delta_cv,"
            "xi,alpha,beta,fu,fv,cu,cv\n";
  for (const AutoCameraInitializationBoardJackknifeSample& sample :
       result.board_jackknife_samples) {
    output << sample.excluded_board_id << ","
           << (sample.comparable ? 1 : 0) << ","
           << sample.expected_view_count << ","
           << sample.optimized_view_count << ","
           << sample.residual_count << ","
           << sample.iteration_count << ","
           << sample.final_rmse << ","
           << sample.delta_xi << ","
           << sample.delta_alpha << ","
           << sample.delta_fu << ","
           << sample.delta_fv << ","
           << sample.delta_cu << ","
           << sample.delta_cv << ","
           << sample.optimized_camera.xi << ","
           << sample.optimized_camera.alpha << ","
           << sample.optimized_camera.beta << ","
           << sample.optimized_camera.fu << ","
           << sample.optimized_camera.fv << ","
           << sample.optimized_camera.cu << ","
           << sample.optimized_camera.cv << "\n";
  }
}

void WriteAutoCameraInitializationCoverageWeightsCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "frame_index,frame_label,board_id,grid_x,grid_y,centroid_x,"
            "centroid_y,weight\n";
  for (const AutoCameraInitializationCoverageWeightRecord& record :
       result.coverage_weight_records) {
    output << record.frame_index << ","
           << record.frame_label << ","
           << record.board_id << ","
           << record.grid_x << ","
           << record.grid_y << ","
           << record.centroid_x << ","
           << record.centroid_y << ","
           << record.weight << "\n";
  }
}

void WriteAutoCameraInitializationPoseExcitationCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "board_id,observation_count,pose_success_count,"
            "normal_spread_median_deg,normal_spread_p95_deg,"
            "normal_spread_max_deg,normal_xy_std_x,normal_xy_std_y,"
            "normal_xy_weak_variance,normal_xy_strong_variance,"
            "normal_xy_axis_balance_ratio,"
            "normal_xy_dominant_axis_angle_deg,tilt_min_deg,tilt_max_deg,"
            "tilt_range_deg,centroid_min_x,centroid_max_x,"
            "centroid_min_y,centroid_max_y,centroid_span_x,centroid_span_y\n";
  for (const AutoCameraInitializationPoseExcitationRecord& record :
       result.pose_excitation_records) {
    output << record.board_id << ","
           << record.observation_count << ","
           << record.pose_success_count << ","
           << record.normal_spread_median_deg << ","
           << record.normal_spread_p95_deg << ","
           << record.normal_spread_max_deg << ","
           << record.normal_xy_std_x << ","
           << record.normal_xy_std_y << ","
           << record.normal_xy_weak_variance << ","
           << record.normal_xy_strong_variance << ","
           << record.normal_xy_axis_balance_ratio << ","
           << record.normal_xy_dominant_axis_angle_deg << ","
           << record.tilt_min_deg << ","
           << record.tilt_max_deg << ","
           << record.tilt_range_deg << ","
           << record.centroid_min_x << ","
           << record.centroid_max_x << ","
           << record.centroid_min_y << ","
           << record.centroid_max_y << ","
           << record.centroid_span_x << ","
           << record.centroid_span_y << "\n";
  }
}

void WriteAutoCameraInitializationPoseExcitationSamplesCsv(
    const std::string& path,
    const AutoCameraInitializationResult& result) {
  std::ofstream output(path.c_str());
  output << "frame_index,frame_label,board_id,pose_rmse,normal_x,normal_y,"
            "normal_z,normal_deviation_from_board_mean_deg,tilt_deg,"
            "centroid_x,centroid_y\n";
  for (const AutoCameraInitializationPoseExcitationSample& sample :
       result.pose_excitation_samples) {
    output << sample.frame_index << ","
           << sample.frame_label << ","
           << sample.board_id << ","
           << sample.pose_rmse << ","
           << sample.normal_x << ","
           << sample.normal_y << ","
           << sample.normal_z << ","
           << sample.normal_deviation_from_board_mean_deg << ","
           << sample.tilt_deg << ","
           << sample.centroid_x << ","
           << sample.centroid_y << "\n";
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
