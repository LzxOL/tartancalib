#include <aslam/cameras/apriltag_internal/Stage5Benchmark.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <Eigen/Eigenvalues>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagCanonicalModel.hpp>
#include <aslam/cameras/apriltag_internal/AngularResidualGeometry.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>
#include <aslam/cameras/apriltag_internal/PolarAngleResidualDiagnostics.hpp>
#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>
#include <aslam/cameras/apriltag_internal/Stage5IncrementalBackendEstimator.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

constexpr double kInvalidProjectionPenaltyPixels = 100.0;
constexpr double kWideFovPoseRescueTriggerRmse = 50.0;
constexpr double kWideFovPoseRescueMaxRayAngleRadians =
    85.0 * 3.14159265358979323846 / 180.0;
constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

std::string SanitizeMetricKey(const std::string& input) {
  std::string sanitized;
  sanitized.reserve(input.size());
  bool last_was_underscore = false;
  for (char ch : input) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch)) {
      sanitized.push_back(static_cast<char>(std::tolower(uch)));
      last_was_underscore = false;
    } else if (!last_was_underscore) {
      sanitized.push_back('_');
      last_was_underscore = true;
    }
  }
  while (!sanitized.empty() && sanitized.back() == '_') {
    sanitized.pop_back();
  }
  return sanitized.empty() ? "reference" : sanitized;
}

std::string JoinStrings(const std::vector<std::string>& values,
                        const std::string& delimiter) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      stream << delimiter;
    }
    stream << values[index];
  }
  return stream.str();
}

std::string JoinDoubles(const std::vector<double>& values,
                        const std::string& delimiter) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      stream << delimiter;
    }
    stream << values[index];
  }
  return stream.str();
}

std::string EscapeCsvCell(const std::string& value) {
  const bool needs_quotes = value.find_first_of(",\"\n\r") != std::string::npos;
  if (!needs_quotes) {
    return value;
  }
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"') {
      escaped += "\"\"";
    } else {
      escaped.push_back(ch);
    }
  }
  escaped += "\"";
  return escaped;
}

std::string RangeBucketLabel(double low, double high, const std::string& suffix) {
  std::ostringstream label;
  label << low << "_" << high << suffix;
  std::string result = label.str();
  std::replace(result.begin(), result.end(), '.', 'p');
  return result;
}

std::string ValueBucketLabel(double value,
                             const std::vector<double>& edges,
                             const std::string& suffix) {
  if (edges.size() < 2) {
    return "all";
  }
  for (std::size_t i = 0; i + 1 < edges.size(); ++i) {
    if (value >= edges[i] && value < edges[i + 1]) {
      return RangeBucketLabel(edges[i], edges[i + 1], suffix);
    }
  }
  std::ostringstream label;
  label << edges.back() << "_plus" << suffix;
  std::string result = label.str();
  std::replace(result.begin(), result.end(), '.', 'p');
  return result;
}

double PolarAngleDegrees(const Eigen::Vector3d& ray) {
  const Eigen::Vector3d unit = ray.normalized();
  const double z = std::max(-1.0, std::min(1.0, unit.z()));
  return std::acos(z) * kRadiansToDegrees;
}

double AngularDifferenceDegrees(const Eigen::Vector3d& a,
                                const Eigen::Vector3d& b) {
  const double dot =
      std::max(-1.0, std::min(1.0, a.normalized().dot(b.normalized())));
  return std::acos(dot) * kRadiansToDegrees;
}

void ReevaluateCalibrationStateBundle(CalibrationStateBundle* bundle);

bool UnprojectPinholeEquidistantRayCurve(
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const Eigen::Vector2d& pixel,
    Eigen::Vector3d* ray);

bool UnprojectForRayCurve(const OuterBootstrapCameraIntrinsics& intrinsics,
                          const DoubleSphereCameraModel& camera,
                          const Eigen::Vector2d& pixel,
                          Eigen::Vector3d* ray);

struct LargeDsPerturbationDefinition {
  std::string label;
  double focal_scale = 1.0;
  double xi_delta = 0.0;
  double alpha_delta = 0.0;
};

bool GetLargeDsPerturbationDefinition(
    const std::string& requested,
    LargeDsPerturbationDefinition* definition) {
  if (definition == nullptr) {
    return false;
  }
  std::string label = requested;
  std::transform(label.begin(), label.end(), label.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::toupper(ch));
                 });
  if (label == "P1") {
    *definition = {"P1", 0.70, -0.20, 0.10};
  } else if (label == "P1F50") {
    *definition = {"P1F50", 0.50, -0.20, 0.10};
  } else if (label == "P2") {
    *definition = {"P2", 0.70, 0.20, -0.10};
  } else if (label == "P2F50") {
    *definition = {"P2F50", 0.50, 0.20, -0.10};
  } else if (label == "P3") {
    *definition = {"P3", 1.30, -0.20, 0.10};
  } else if (label == "P4") {
    *definition = {"P4", 1.30, 0.20, -0.10};
  } else if (label == "N0") {
    *definition = {"N0", 0.70, -0.20, 0.10};
  } else if (label == "N1") {
    *definition = {"N1", std::exp(0.9 * std::log(0.70)), -0.20, 0.10};
  } else if (label == "N2") {
    *definition = {"N2", std::exp(1.1 * std::log(0.70)), -0.20, 0.10};
  } else if (label == "N3") {
    *definition = {"N3", 0.70, -0.18, 0.11};
  } else if (label == "N4") {
    *definition = {"N4", 0.70, -0.22, 0.09};
  } else {
    return false;
  }
  return true;
}

std::string LargePerturbationSceneFingerprint(
    const CalibrationSceneState& scene) {
  std::ostringstream stream;
  stream << std::setprecision(17);
  stream << scene.camera.NormalizedFamilyString() << ":";
  for (double value : scene.camera.CombinedParameterVector()) {
    stream << value << ",";
  }
  stream << "|";
  for (const JointSceneFrameState& frame : scene.frames) {
    stream << "f" << frame.frame_index << ":";
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        stream << frame.T_camera_reference(row, col) << ",";
      }
    }
  }
  for (const JointSceneBoardState& board : scene.boards) {
    stream << "b" << board.board_id << ":";
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        stream << board.T_reference_board(row, col) << ",";
      }
    }
  }
  const std::string serialized = stream.str();
  std::uint64_t hash = 1469598103934665603ULL;
  for (unsigned char value : serialized) {
    hash ^= static_cast<std::uint64_t>(value);
    hash *= 1099511628211ULL;
  }
  std::ostringstream digest;
  digest << "fnv1a64:" << std::hex << hash;
  return digest.str();
}

std::string LargePerturbationObservationFingerprint(
    const CalibrationMeasurementDataset& dataset) {
  std::vector<const JointPointObservation*> observations;
  for (const JointMeasurementFrameResult& frame : dataset.frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      for (const JointPointObservation& point : board.points) {
        observations.push_back(&point);
      }
    }
  }
  std::sort(observations.begin(), observations.end(),
            [](const JointPointObservation* lhs,
               const JointPointObservation* rhs) {
              return std::make_tuple(
                         lhs->frame_index, lhs->board_id, lhs->point_id,
                         lhs->point_type, lhs->image_xy.x(), lhs->image_xy.y()) <
                     std::make_tuple(
                         rhs->frame_index, rhs->board_id, rhs->point_id,
                         rhs->point_type, rhs->image_xy.x(), rhs->image_xy.y());
            });
  std::ostringstream stream;
  stream << std::setprecision(17);
  for (const JointPointObservation* point : observations) {
    stream << point->frame_index << "," << point->board_id << ","
           << point->point_id << "," << static_cast<int>(point->point_type)
           << "," << point->image_xy.x() << "," << point->image_xy.y()
           << "," << point->target_xyz_board.x() << ","
           << point->target_xyz_board.y() << ","
           << point->target_xyz_board.z() << "|";
  }
  const std::string serialized = stream.str();
  std::uint64_t hash = 1469598103934665603ULL;
  for (unsigned char value : serialized) {
    hash ^= static_cast<std::uint64_t>(value);
    hash *= 1099511628211ULL;
  }
  std::ostringstream digest;
  digest << "fnv1a64:" << std::hex << hash;
  return digest.str();
}

bool SameLargePerturbationCamera(
    const OuterBootstrapCameraIntrinsics& lhs,
    const OuterBootstrapCameraIntrinsics& rhs) {
  const std::vector<double> lhs_values = lhs.CombinedParameterVector();
  const std::vector<double> rhs_values = rhs.CombinedParameterVector();
  if (lhs.NormalizedFamilyString() != rhs.NormalizedFamilyString() ||
      lhs_values.size() != rhs_values.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs_values.size(); ++index) {
    const double scale = std::max(
        1.0, std::max(std::abs(lhs_values[index]), std::abs(rhs_values[index])));
    if (std::abs(lhs_values[index] - rhs_values[index]) > 1e-12 * scale) {
      return false;
    }
  }
  return true;
}

bool LoadLargePerturbationSceneSnapshot(
    const std::string& path,
    CalibrationSceneState* scene,
    std::string* error_message) {
  if (scene == nullptr || path.empty()) {
    if (error_message != nullptr) {
      *error_message = "Scene snapshot path or output scene is empty.";
    }
    return false;
  }
  std::ifstream input(path.c_str());
  if (!input.is_open()) {
    if (error_message != nullptr) {
      *error_message = "Failed to open scene snapshot: " + path;
    }
    return false;
  }
  scene->frames.clear();
  scene->boards.clear();
  std::string tag;
  bool have_camera = false;
  while (input >> tag) {
    if (tag == "camera") {
      if (!(input >> scene->camera.xi >> scene->camera.alpha >>
            scene->camera.fu >> scene->camera.fv >> scene->camera.cu >>
            scene->camera.cv)) {
        break;
      }
      have_camera = true;
    } else if (tag == "distortion") {
      std::size_t coefficient_count = 0;
      if (!(input >> coefficient_count)) {
        break;
      }
      std::vector<double> coefficients(coefficient_count, 0.0);
      for (std::size_t index = 0; index < coefficient_count; ++index) {
        if (!(input >> coefficients[index])) {
          break;
        }
      }
      if (!scene->camera.SetDistortionVector(coefficients)) {
        if (error_message != nullptr) {
          *error_message = "Scene snapshot has invalid distortion coefficients: " +
                           path;
        }
        return false;
      }
    } else if (tag == "frame") {
      JointSceneFrameState frame;
      int initialized = 0;
      if (!(input >> frame.frame_index >> initialized >> frame.observation_count >>
            frame.rmse)) {
        break;
      }
      frame.initialized = initialized != 0;
      for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
          if (!(input >> frame.T_camera_reference(row, col))) {
            break;
          }
        }
      }
      bool replaced = false;
      for (JointSceneFrameState& current : scene->frames) {
        if (current.frame_index == frame.frame_index) {
          current.initialized = frame.initialized;
          current.observation_count = frame.observation_count;
          current.rmse = frame.rmse;
          current.T_camera_reference = frame.T_camera_reference;
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        scene->frames.push_back(frame);
      }
    } else if (tag == "board") {
      JointSceneBoardState board;
      int initialized = 0;
      if (!(input >> board.board_id >> initialized >> board.observation_count >>
            board.rmse)) {
        break;
      }
      board.initialized = initialized != 0;
      for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
          if (!(input >> board.T_reference_board(row, col))) {
            break;
          }
        }
      }
      bool replaced = false;
      for (JointSceneBoardState& current : scene->boards) {
        if (current.board_id == board.board_id) {
          current.initialized = board.initialized;
          current.observation_count = board.observation_count;
          current.rmse = board.rmse;
          current.T_reference_board = board.T_reference_board;
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        scene->boards.push_back(board);
      }
    } else {
      std::string ignored;
      std::getline(input, ignored);
    }
  }
  if (!have_camera || !scene->camera.IsValid()) {
    if (error_message != nullptr) {
      *error_message = "Scene snapshot has no valid camera record: " + path;
    }
    return false;
  }
  return true;
}

bool EvaluateLargeDsProjectionGrid(
    const OuterBootstrapCameraIntrinsics& camera,
    int grid_width,
    int grid_height,
    int* valid_count,
    int* invalid_count,
    const OuterBootstrapCameraIntrinsics* reference_domain_camera = nullptr) {
  if (valid_count == nullptr || invalid_count == nullptr ||
      grid_width < 2 || grid_height < 2 ||
      (camera.NormalizedFamilyString() != "ds-none" &&
       camera.NormalizedFamilyString() != "pinhole-equi")) {
    return false;
  }
  *valid_count = 0;
  *invalid_count = 0;
  try {
    const DoubleSphereCameraModel model =
        DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(camera));
    std::unique_ptr<DoubleSphereCameraModel> reference_model;
    double reference_radius_px = 0.0;
    if (reference_domain_camera != nullptr) {
      reference_model.reset(new DoubleSphereCameraModel(
          DoubleSphereCameraModel::FromConfig(
              MakeIntermediateCameraConfig(*reference_domain_camera))));
      reference_radius_px = std::min(
          std::min(reference_domain_camera->cu, reference_domain_camera->cv),
          std::min(reference_domain_camera->resolution.width - 1.0 -
                       reference_domain_camera->cu,
                   reference_domain_camera->resolution.height - 1.0 -
                       reference_domain_camera->cv));
      if (!(reference_radius_px > 0.0) || !std::isfinite(reference_radius_px)) {
        return false;
      }
    }
    for (int y_index = 0; y_index < grid_height; ++y_index) {
      const double y = static_cast<double>(y_index) *
                       static_cast<double>(camera.resolution.height - 1) /
                       static_cast<double>(grid_height - 1);
      for (int x_index = 0; x_index < grid_width; ++x_index) {
        const double x = static_cast<double>(x_index) *
                         static_cast<double>(camera.resolution.width - 1) /
                         static_cast<double>(grid_width - 1);
        const Eigen::Vector2d pixel(x, y);
        if (reference_model) {
          const double radius_px =
              (pixel - Eigen::Vector2d(reference_domain_camera->cu,
                                       reference_domain_camera->cv)).norm();
          if (radius_px > reference_radius_px) {
            continue;
          }
          Eigen::Vector3d reference_ray = Eigen::Vector3d::Zero();
          Eigen::Vector2d reference_roundtrip = Eigen::Vector2d::Zero();
          const bool reference_valid =
              UnprojectForRayCurve(*reference_domain_camera, *reference_model,
                                   pixel, &reference_ray) &&
              reference_ray.allFinite() && reference_ray.norm() > 1e-12 &&
              reference_model->vsEuclideanToKeypoint(reference_ray,
                                                     &reference_roundtrip) &&
              reference_roundtrip.allFinite();
          if (!reference_valid) {
            continue;
          }
        }
        Eigen::Vector3d ray = Eigen::Vector3d::Zero();
        Eigen::Vector2d roundtrip = Eigen::Vector2d::Zero();
        const bool valid = UnprojectForRayCurve(camera, model, pixel, &ray) &&
                           ray.allFinite() && ray.norm() > 1e-12 &&
                           model.vsEuclideanToKeypoint(ray, &roundtrip) &&
                           roundtrip.allFinite();
        if (valid) {
          ++(*valid_count);
        } else {
          ++(*invalid_count);
        }
      }
    }
  } catch (const std::exception&) {
    *valid_count = 0;
    *invalid_count = grid_width * grid_height;
    return false;
  }
  return *invalid_count == 0;
}

bool ApplyLargeDsIntrinsicPerturbation(
    const Stage5BenchmarkInput& input,
    CalibrationStateBundle* bundle,
    Stage5LargeIntrinsicPerturbationState* state) {
  if (bundle == nullptr || state == nullptr) {
    return false;
  }
  state->enabled = input.enable_large_intrinsic_perturbation;
  state->requested_profile = input.large_intrinsic_perturbation_profile;
  if (!state->enabled) {
    return true;
  }
  CalibrationSceneState reference_scene = bundle->scene_state;
  if (!input.large_intrinsic_perturbation_reference_scene_path.empty()) {
    std::string snapshot_error;
    if (!LoadLargePerturbationSceneSnapshot(
            input.large_intrinsic_perturbation_reference_scene_path,
            &reference_scene, &snapshot_error)) {
      state->failure_reason = snapshot_error;
      return false;
    }
  }
  bundle->scene_state.camera = reference_scene.camera;
  bundle->scene_state.frames = reference_scene.frames;
  bundle->scene_state.boards = reference_scene.boards;
  state->reference_scene = reference_scene;
  state->reference_camera = reference_scene.camera;
  state->reference_scene_fingerprint =
      LargePerturbationSceneFingerprint(bundle->scene_state);
  LargeDsPerturbationDefinition definition;
  if (!GetLargeDsPerturbationDefinition(
          input.large_intrinsic_perturbation_profile, &definition)) {
    state->failure_reason =
        "Unknown large DS perturbation profile; expected P1-P4, P1F50, P2F50, or N0-N4.";
    return false;
  }
  state->effective_profile = definition.label;
  state->requested_focal_scale = definition.focal_scale;
  state->requested_xi_delta = definition.xi_delta;
  state->requested_alpha_delta = definition.alpha_delta;
  state->requested_scale = input.large_intrinsic_perturbation_scale;
  const double maximum_scale =
      input.large_intrinsic_perturbation_strict_scale ? 2.0 : 1.0;
  if (!std::isfinite(state->requested_scale) ||
      state->requested_scale < 0.0 ||
      state->requested_scale > maximum_scale) {
    state->failure_reason =
        "Large DS perturbation scale is outside the configured valid range.";
    return false;
  }
  const std::string camera_family =
      state->reference_camera.NormalizedFamilyString();
  if (camera_family != "ds-none" && camera_family != "pinhole-equi") {
    state->failure_reason =
        "Large intrinsic perturbation is implemented only for ds-none and pinhole-equi.";
    return false;
  }
  if (camera_family == "pinhole-equi") {
    if (definition.label != "P1") {
      state->failure_reason =
          "KB perturbation currently supports only the explicitly defined P1 branch.";
      return false;
    }
    definition.xi_delta = 0.0;
    definition.alpha_delta = 0.0;
    state->effective_profile = "KB-P1-focal-plus-principal-point";
    state->requested_xi_delta = 0.0;
    state->requested_alpha_delta = 0.0;
  }

  constexpr int kGridWidth = 41;
  constexpr int kGridHeight = 41;
  state->projection_grid_width = kGridWidth;
  state->projection_grid_height = kGridHeight;
  // Back off the perturbation vector as a whole. This preserves the requested
  // direction and never silently clips xi or alpha to a model boundary.
  const int max_backoff_index =
      (input.large_intrinsic_perturbation_strict_scale ||
       state->requested_scale == 0.0) ? 0 : 9;
  for (int backoff_index = 0; backoff_index <= max_backoff_index;
       ++backoff_index) {
    const double backoff = 1.0 - 0.1 * static_cast<double>(backoff_index);
    const double effective_scale = state->requested_scale * backoff;
    OuterBootstrapCameraIntrinsics candidate = state->reference_camera;
    // A logarithmic focal interpolation preserves the profile's multiplicative
    // definition: scale=0 is exactly clean and scale=1 is P1--P4.
    candidate.fu = state->reference_camera.fu *
                   std::exp(effective_scale * std::log(definition.focal_scale));
    candidate.fv = state->reference_camera.fv *
                   std::exp(effective_scale * std::log(definition.focal_scale));
    if (camera_family == "ds-none") {
      candidate.xi = state->reference_camera.xi +
                     definition.xi_delta * effective_scale;
      candidate.alpha =
          state->reference_camera.alpha +
          definition.alpha_delta * effective_scale;
      if (input.large_intrinsic_perturbation_strict_scale) {
        candidate.cu = state->reference_camera.cu + 150.0 * effective_scale;
        candidate.cv = state->reference_camera.cv - 150.0 * effective_scale;
      }
    } else if (input.large_intrinsic_perturbation_strict_scale) {
      // KB has no xi/alpha shape parameters. The strict robustness protocol
      // therefore couples its focal P1 direction with a fixed signed
      // projection-center displacement, recorded through the initial camera.
      candidate.cu = state->reference_camera.cu + 150.0 * effective_scale;
      candidate.cv = state->reference_camera.cv - 150.0 * effective_scale;
    }
    int valid_count = 0;
    int invalid_count = 0;
    EvaluateLargeDsProjectionGrid(candidate, kGridWidth, kGridHeight,
                                  &valid_count, &invalid_count,
                                  &state->reference_camera);
    bool parameter_domain_valid =
        candidate.IsValid() && std::isfinite(candidate.fu) &&
        std::isfinite(candidate.fv);
    if (camera_family == "ds-none") {
      parameter_domain_valid =
          parameter_domain_valid && std::isfinite(candidate.xi) &&
          std::isfinite(candidate.alpha) && candidate.alpha > 0.0 &&
          candidate.alpha < 1.0 && candidate.xi > -1.0 && candidate.xi < 1.0;
    } else {
      const std::vector<double> distortion = candidate.DistortionVector();
      parameter_domain_valid =
          parameter_domain_valid && distortion.size() == 4u &&
          std::all_of(distortion.begin(), distortion.end(),
                      [](double value) { return std::isfinite(value); });
    }
    // Pixel-domain validity is a measured outcome, not a reason to weaken an
    // otherwise legal P1--P4 parameter perturbation. Invalid fixed-mask pixels
    // are retained in valid_grid_ratio/invalid_grid_count diagnostics.
    if (!parameter_domain_valid || valid_count <= 0) {
      state->valid_projection_grid_count = valid_count;
      state->invalid_projection_grid_count = invalid_count;
      continue;
    }
    state->actual_focal_scale = candidate.fu / state->reference_camera.fu;
    state->actual_xi_delta = candidate.xi - state->reference_camera.xi;
    state->actual_alpha_delta = candidate.alpha - state->reference_camera.alpha;
    state->effective_scale = effective_scale;
    state->valid_projection_grid_count = valid_count;
    state->invalid_projection_grid_count = invalid_count;
    state->valid_projection_grid = invalid_count == 0;
    state->perturbed_camera = candidate;
    bundle->scene_state.camera = candidate;
    bundle->scene_state.coarse_or_optimized_level =
        "large_intrinsic_perturbation_before_selection";
    // This only recomputes residual diagnostics for already imported/frozen
    // observations. It never regenerates or filters internal image points.
    ReevaluateCalibrationStateBundle(bundle);
    state->perturbed_scene_fingerprint =
        LargePerturbationSceneFingerprint(bundle->scene_state);
    return bundle->scene_state.IsValid();
  }
  state->failure_reason =
      "Requested camera perturbation has no usable projection/inverse-projection grid.";
  return false;
}

bool UnprojectPinholeEquidistantRayCurve(
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const Eigen::Vector2d& pixel,
    Eigen::Vector3d* ray) {
  if (ray == nullptr || intrinsics.fu <= 0.0 || intrinsics.fv <= 0.0) {
    return false;
  }
  const std::vector<double> distortion = intrinsics.DistortionVector();
  if (distortion.size() < 4u) {
    return false;
  }
  const double xd = (pixel.x() - intrinsics.cu) / intrinsics.fu;
  const double yd = (pixel.y() - intrinsics.cv) / intrinsics.fv;
  const double rd = std::hypot(xd, yd);
  if (!std::isfinite(rd)) {
    return false;
  }
  if (rd < 1e-12) {
    *ray = Eigen::Vector3d(0.0, 0.0, 1.0);
    return true;
  }
  const auto theta_distorted = [&distortion](double theta) {
    const double theta2 = theta * theta;
    const double theta4 = theta2 * theta2;
    const double theta6 = theta4 * theta2;
    const double theta8 = theta4 * theta4;
    return theta *
           (1.0 + distortion[0] * theta2 + distortion[1] * theta4 +
            distortion[2] * theta6 + distortion[3] * theta8);
  };
  const double max_theta = 0.5 * std::acos(-1.0) - 1e-9;
  const double max_rd = theta_distorted(max_theta);
  if (!std::isfinite(max_rd) || rd > max_rd + 1e-9) {
    return false;
  }

  double low = 0.0;
  double high = max_theta;
  for (int iteration = 0; iteration < 80; ++iteration) {
    const double mid = 0.5 * (low + high);
    if (theta_distorted(mid) < rd) {
      low = mid;
    } else {
      high = mid;
    }
  }
  const double theta = 0.5 * (low + high);
  const double ru = std::tan(theta);
  const double scale = ru / rd;
  *ray = Eigen::Vector3d(xd * scale, yd * scale, 1.0);
  return ray->allFinite() && ray->norm() > 1e-12;
}

bool UnprojectForRayCurve(const OuterBootstrapCameraIntrinsics& intrinsics,
                          const DoubleSphereCameraModel& camera,
                          const Eigen::Vector2d& pixel,
                          Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    return false;
  }
  if (intrinsics.NormalizedFamilyString() == "pinhole-equi") {
    return UnprojectPinholeEquidistantRayCurve(intrinsics, pixel, ray);
  }
  return camera.keypointToEuclidean(pixel, ray);
}

struct CameraRayCurveReference {
  std::string label;
  OuterBootstrapCameraIntrinsics intrinsics;
};

struct CameraRayCurveBucketAccumulator {
  std::string reference_label;
  std::string reference_family;
  std::string bucket_type;
  std::string bucket_label;
  int sample_count = 0;
  double angular_sum = 0.0;
  double angular_square_sum = 0.0;
  double max_angular = 0.0;
  double our_polar_sum = 0.0;
  double reference_polar_sum = 0.0;

  void Add(const CameraRayCurveSample& sample) {
    ++sample_count;
    angular_sum += sample.angular_diff_deg;
    angular_square_sum += sample.angular_diff_deg * sample.angular_diff_deg;
    max_angular = std::max(max_angular, sample.angular_diff_deg);
    our_polar_sum += sample.our_polar_deg;
    reference_polar_sum += sample.reference_polar_deg;
  }

  CameraRayCurveBucketSummary Summary() const {
    CameraRayCurveBucketSummary summary;
    summary.reference_label = reference_label;
    summary.reference_family = reference_family;
    summary.bucket_type = bucket_type;
    summary.bucket_label = bucket_label;
    summary.sample_count = sample_count;
    if (sample_count > 0) {
      summary.mean_angular_diff_deg =
          angular_sum / static_cast<double>(sample_count);
      summary.rms_angular_diff_deg =
          std::sqrt(angular_square_sum / static_cast<double>(sample_count));
      summary.max_angular_diff_deg = max_angular;
      summary.mean_our_polar_deg =
          our_polar_sum / static_cast<double>(sample_count);
      summary.mean_reference_polar_deg =
          reference_polar_sum / static_cast<double>(sample_count);
    }
    return summary;
  }
};

void AddRayCurveBucketSample(
    const CameraRayCurveSample& sample,
    const std::string& bucket_type,
    const std::string& bucket_label,
    std::map<std::string, CameraRayCurveBucketAccumulator>* accumulators) {
  if (accumulators == nullptr) {
    return;
  }
  const std::string key = sample.reference_label + "|" + bucket_type + "|" +
                          bucket_label;
  CameraRayCurveBucketAccumulator& accumulator = (*accumulators)[key];
  accumulator.reference_label = sample.reference_label;
  accumulator.reference_family = sample.reference_family;
  accumulator.bucket_type = bucket_type;
  accumulator.bucket_label = bucket_label;
  accumulator.Add(sample);
}

CameraRayCurveDiagnostics ComputeCameraRayCurveDiagnostics(
    const OuterBootstrapCameraIntrinsics& our_intrinsics,
    const std::string& our_camera_source,
    const std::vector<CameraRayCurveReference>& references) {
  CameraRayCurveDiagnostics diagnostics;
  diagnostics.grid_width = 41;
  diagnostics.grid_height = 41;
  diagnostics.comparison_count = static_cast<int>(references.size());
  diagnostics.our_camera_source = our_camera_source;
  diagnostics.our_camera = our_intrinsics;
  if (!our_intrinsics.IsValid()) {
    diagnostics.failure_reason = "ours camera intrinsics are invalid";
    return diagnostics;
  }
  if (references.empty()) {
    diagnostics.failure_reason = "no reference camera intrinsics available";
    return diagnostics;
  }

  DoubleSphereCameraModel our_camera;
  try {
    our_camera = DoubleSphereCameraModel::FromConfig(
        MakeIntermediateCameraConfig(our_intrinsics));
  } catch (const std::exception& e) {
    diagnostics.failure_reason =
        std::string("failed to construct ours camera model: ") + e.what();
    return diagnostics;
  }

  const cv::Size resolution = our_intrinsics.resolution;
  const Eigen::Vector2d image_center(
      0.5 * static_cast<double>(std::max(1, resolution.width - 1)),
      0.5 * static_cast<double>(std::max(1, resolution.height - 1)));
  const double max_radius =
      std::max(1e-9, std::sqrt(image_center.x() * image_center.x() +
                               image_center.y() * image_center.y()));
  const std::vector<double> radial_edges = {0.0, 0.2, 0.4, 0.6, 0.8, 1.01};
  const std::vector<double> polar_edges = {0.0, 30.0, 50.0, 70.0, 90.0};
  std::map<std::string, CameraRayCurveBucketAccumulator> bucket_accumulators;

  for (const CameraRayCurveReference& reference : references) {
    if (!reference.intrinsics.IsValid()) {
      diagnostics.warnings.push_back("skipped invalid reference camera: " +
                                     reference.label);
      continue;
    }
    DoubleSphereCameraModel reference_camera;
    try {
      reference_camera = DoubleSphereCameraModel::FromConfig(
          MakeIntermediateCameraConfig(reference.intrinsics));
    } catch (const std::exception& e) {
      diagnostics.warnings.push_back("skipped reference camera " +
                                     reference.label + ": " + e.what());
      continue;
    }

    for (int y_index = 0; y_index < diagnostics.grid_height; ++y_index) {
      const double y = diagnostics.grid_height <= 1
                           ? image_center.y()
                           : static_cast<double>(y_index) *
                                 static_cast<double>(resolution.height - 1) /
                                 static_cast<double>(diagnostics.grid_height - 1);
      for (int x_index = 0; x_index < diagnostics.grid_width; ++x_index) {
        const double x = diagnostics.grid_width <= 1
                             ? image_center.x()
                             : static_cast<double>(x_index) *
                                   static_cast<double>(resolution.width - 1) /
                                   static_cast<double>(diagnostics.grid_width - 1);
        Eigen::Vector3d our_ray = Eigen::Vector3d::Zero();
        Eigen::Vector3d reference_ray = Eigen::Vector3d::Zero();
        const Eigen::Vector2d pixel(x, y);
        if (!UnprojectForRayCurve(our_intrinsics, our_camera, pixel, &our_ray) ||
            !UnprojectForRayCurve(reference.intrinsics, reference_camera, pixel,
                                  &reference_ray) ||
            our_ray.norm() <= 1e-12 || reference_ray.norm() <= 1e-12 ||
            !our_ray.allFinite() || !reference_ray.allFinite()) {
          ++diagnostics.invalid_unprojection_count;
          continue;
        }
        CameraRayCurveSample sample;
        sample.reference_label = SanitizeMetricKey(reference.label);
        sample.reference_family = reference.intrinsics.NormalizedFamilyString();
        sample.image_x = x;
        sample.image_y = y;
        sample.radial_fraction = (pixel - image_center).norm() / max_radius;
        sample.our_polar_deg = PolarAngleDegrees(our_ray);
        sample.reference_polar_deg = PolarAngleDegrees(reference_ray);
        sample.angular_diff_deg =
            AngularDifferenceDegrees(our_ray, reference_ray);
        diagnostics.samples.push_back(sample);

        AddRayCurveBucketSample(
            sample, "radial",
            ValueBucketLabel(sample.radial_fraction, radial_edges, "r"),
            &bucket_accumulators);
        AddRayCurveBucketSample(
            sample, "ours_polar",
            ValueBucketLabel(sample.our_polar_deg, polar_edges, "deg"),
            &bucket_accumulators);
        AddRayCurveBucketSample(
            sample, "all", "all", &bucket_accumulators);
      }
    }
  }

  diagnostics.sample_count = static_cast<int>(diagnostics.samples.size());
  diagnostics.bucket_summaries.reserve(bucket_accumulators.size());
  for (const auto& kv : bucket_accumulators) {
    diagnostics.bucket_summaries.push_back(kv.second.Summary());
  }
  diagnostics.success = diagnostics.sample_count > 0;
  if (!diagnostics.success && diagnostics.failure_reason.empty()) {
    diagnostics.failure_reason = "all ray-curve samples were invalid";
  }
  return diagnostics;
}

struct TrialBoardRmseAccumulator {
  std::string frame_label;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double squared_error_sum = 0.0;
};

struct DenseGridOutlierFilterResult {
  CalibrationStateBundle bundle;
  bool enabled = false;
  double median_pixels = 0.0;
  double robust_sigma_pixels = 0.0;
  double threshold_pixels = 0.0;
  int removed_point_count = 0;
  int dropped_view_count = 0;
};

struct TrialBackendMetrics {
  bool success = false;
  double overall_rmse = 0.0;
  double outer_rmse = 0.0;
  double internal_rmse = 0.0;
};

struct TrialBackendTraversalBudget {
  TrialBackendFrameBoardSelectionOptions::BudgetMode mode =
      TrialBackendFrameBoardSelectionOptions::BudgetMode::Fixed;
  int traversal_limit = 0;
  int runtime_safety_ceiling = 0;
  std::string max_candidate_additions_effective;
};

TrialBackendTraversalBudget ComputeTrialBackendTraversalBudget(
    const TrialBackendFrameBoardSelectionOptions& options,
    int valid_candidate_count) {
  TrialBackendTraversalBudget budget;
  budget.mode = options.budget_mode;
  budget.runtime_safety_ceiling =
      std::max(0, options.runtime_safety_ceiling);
  const int valid_count = std::max(0, valid_candidate_count);
  if (options.budget_mode ==
      TrialBackendFrameBoardSelectionOptions::BudgetMode::KalibrStyle) {
    budget.traversal_limit =
        budget.runtime_safety_ceiling > 0
            ? budget.runtime_safety_ceiling
            : std::numeric_limits<int>::max();
    budget.max_candidate_additions_effective =
        "ignored_in_kalibr_style_batch";
    return budget;
  }
  if (options.budget_mode ==
      TrialBackendFrameBoardSelectionOptions::BudgetMode::Adaptive) {
    const int min_budget = std::max(0, options.adaptive_budget_min);
    const int max_budget =
        options.adaptive_budget_max > 0
            ? std::max(min_budget, options.adaptive_budget_max)
            : std::numeric_limits<int>::max();
    int effective_budget = static_cast<int>(
        std::ceil(static_cast<double>(valid_count) *
                  std::max(0.0, options.adaptive_budget_ratio)));
    effective_budget = std::max(effective_budget, min_budget);
    effective_budget = std::min(effective_budget, max_budget);
    budget.traversal_limit = std::max(0, effective_budget);
  } else {
    budget.traversal_limit = std::max(0, options.max_candidate_additions);
  }
  budget.max_candidate_additions_effective =
      std::to_string(budget.traversal_limit);
  return budget;
}

struct CandidateCoverageScoreBreakdown {
  double total_score = 0.0;
  double coverage_gain = 0.0;
  double polar_gain = 0.0;
  double edge_gain = 0.0;
  double board_balance_gain = 0.0;
  double frame_novelty_gain = 0.0;
  double grid_gain = 0.0;
  double covisibility_gain = 0.0;
  double residual_quality_score = 0.0;
  double mean_polar_angle_deg = 0.0;
  double max_polar_angle_deg = 0.0;
};

struct CandidateConsistencyScoreBreakdown {
  bool available = false;
  double score = 0.0;
  double penalty = 0.0;
  double translation_error_mm = 0.0;
  double rotation_error_deg = 0.0;
  double local_outer_rmse = 0.0;
};

struct BoardObservationGeometrySummary {
  bool found = false;
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  int point_count = 0;
  int visible_board_count_in_frame = 0;
  double mean_radius_px = 0.0;
  double max_radius_px = 0.0;
  double mean_polar_angle_deg = 0.0;
  double max_polar_angle_deg = 0.0;
  double projected_area_px = 0.0;
  double projected_area_ratio = 0.0;
};

struct IntrinsicsInformationGainProxyBreakdown {
  double score_term = 0.0;
  double coverage_term = 0.0;
  double intrinsics_jacobian_logdet_gain = 0.0;
  double intrinsics_jacobian_trace_gain = 0.0;
  double intrinsics_jacobian_rank_gain = 0.0;
  double intrinsics_jacobian_info_term = 0.0;
  double frame_completion_bonus = 0.0;
  double new_board_bonus = 0.0;
  double information_gain_proxy = 0.0;
};

struct IntrinsicsJacobianInformationSummary {
  bool available = false;
  int point_count = 0;
  Eigen::Matrix<double, 6, 6> fisher =
      Eigen::Matrix<double, 6, 6>::Zero();
  double logdet = 0.0;
  double trace = 0.0;
  double rank_proxy = 0.0;
};

struct BackendDatasetStats {
  std::set<int> accepted_frame_indices;
  std::set<std::pair<int, int> > accepted_board_observation_keys;
  int accepted_frame_count = 0;
  int accepted_board_observation_count = 0;
  int accepted_outer_point_count = 0;
  int accepted_internal_point_count = 0;
  int accepted_total_point_count = 0;
};

struct EvaluationRobustSummary {
  double frame_median_rmse = std::numeric_limits<double>::quiet_NaN();
  double frame_p90_rmse = std::numeric_limits<double>::quiet_NaN();
  double trimmed90_rmse = std::numeric_limits<double>::quiet_NaN();
  double huber15_rmse = std::numeric_limits<double>::quiet_NaN();
  double fold_median_mean_rmse = std::numeric_limits<double>::quiet_NaN();
  double fold_median_max_rmse = std::numeric_limits<double>::quiet_NaN();
  double fold_median_std_rmse = std::numeric_limits<double>::quiet_NaN();
};

using FrameBoardKey = std::pair<int, int>;

void ReevaluateCalibrationStateBundle(CalibrationStateBundle* bundle);

double Quantile(std::vector<double> values, double q) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const double position =
      std::max(0.0, std::min(1.0, q)) *
      static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

EvaluationRobustSummary SummarizeEvaluationRobustness(
    const CameraModelRefitEvaluationResult& evaluation) {
  EvaluationRobustSummary summary;
  std::vector<double> frame_rmses;
  std::array<std::vector<double>, 5> fold_frame_rmses;
  for (const CameraModelRefitFrameDiagnostics& frame :
       evaluation.frame_diagnostics) {
    if (frame.point_count > 0 && std::isfinite(frame.rmse)) {
      frame_rmses.push_back(frame.rmse);
      std::uint32_t hash = 2166136261u;
      for (const unsigned char ch : frame.frame_label) {
        hash ^= static_cast<std::uint32_t>(ch);
        hash *= 16777619u;
      }
      fold_frame_rmses[hash % fold_frame_rmses.size()].push_back(frame.rmse);
    }
  }
  summary.frame_median_rmse = Quantile(frame_rmses, 0.5);
  summary.frame_p90_rmse = Quantile(frame_rmses, 0.9);
  std::vector<double> fold_medians;
  for (const std::vector<double>& fold_values : fold_frame_rmses) {
    if (!fold_values.empty()) {
      fold_medians.push_back(Quantile(fold_values, 0.5));
    }
  }
  if (!fold_medians.empty()) {
    summary.fold_median_mean_rmse =
        std::accumulate(fold_medians.begin(), fold_medians.end(), 0.0) /
        static_cast<double>(fold_medians.size());
    summary.fold_median_max_rmse =
        *std::max_element(fold_medians.begin(), fold_medians.end());
    double variance = 0.0;
    for (const double value : fold_medians) {
      const double delta = value - summary.fold_median_mean_rmse;
      variance += delta * delta;
    }
    summary.fold_median_std_rmse = std::sqrt(
        variance / static_cast<double>(fold_medians.size()));
  }

  std::vector<double> residuals;
  for (const CameraModelRefitPointDiagnostics& point :
       evaluation.point_diagnostics) {
    if (std::isfinite(point.residual_norm)) {
      residuals.push_back(point.residual_norm);
    }
  }
  if (residuals.empty()) {
    return summary;
  }
  std::sort(residuals.begin(), residuals.end());
  const std::size_t retained = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::ceil(0.9 * static_cast<double>(residuals.size()))));
  double trimmed_squared_sum = 0.0;
  double huber_cost_sum = 0.0;
  constexpr double kHuberDeltaPixels = 1.5;
  for (std::size_t index = 0; index < residuals.size(); ++index) {
    const double residual = residuals[index];
    if (index < retained) {
      trimmed_squared_sum += residual * residual;
    }
    huber_cost_sum +=
        residual <= kHuberDeltaPixels
            ? 0.5 * residual * residual
            : kHuberDeltaPixels *
                  (residual - 0.5 * kHuberDeltaPixels);
  }
  summary.trimmed90_rmse = std::sqrt(
      trimmed_squared_sum / static_cast<double>(retained));
  summary.huber15_rmse = std::sqrt(
      2.0 * huber_cost_sum / static_cast<double>(residuals.size()));
  return summary;
}

bool ApplySingleBoardCameraAndPoseRefit(
    const OuterBootstrapCameraIntrinsics& camera,
    const CameraModelRefitEvaluationResult& evaluation,
    CalibrationStateBundle* bundle) {
  if (bundle == nullptr || !camera.IsValid() || !evaluation.success) {
    return false;
  }
  bundle->scene_state.camera = camera;
  int updated_frame_count = 0;
  for (const CameraModelRefitBoardObservationDiagnostics& board_diag :
       evaluation.board_observation_diagnostics) {
    if (!board_diag.pose_only_refit_success) {
      continue;
    }
    const JointSceneBoardState* board_state = nullptr;
    for (const JointSceneBoardState& board : bundle->scene_state.boards) {
      if (board.board_id == board_diag.board_id && board.initialized) {
        board_state = &board;
        break;
      }
    }
    if (board_state == nullptr) {
      continue;
    }
    const Eigen::Matrix4d T_camera_reference =
        board_diag.T_camera_board * board_state->T_reference_board.inverse();
    for (JointSceneFrameState& frame : bundle->scene_state.frames) {
      if (frame.frame_index == board_diag.frame_index) {
        frame.T_camera_reference = T_camera_reference;
        frame.initialized = true;
        ++updated_frame_count;
        break;
      }
    }
  }
  ReevaluateCalibrationStateBundle(bundle);
  return updated_frame_count > 0 && bundle->IsReadyForBackend();
}

struct BackendInputAblationWorkingResult {
  CalibrationStateBundle bundle;
  BackendInputAblationResult result;
};

bool ScoreSortedBefore(
    const TrialBackendFrameBoardObservationDecision& lhs,
    const TrialBackendFrameBoardObservationDecision& rhs) {
  if (lhs.kept != rhs.kept) {
    return lhs.kept && !rhs.kept;
  }
  if (lhs.candidate_score != rhs.candidate_score) {
    return lhs.candidate_score > rhs.candidate_score;
  }
  if (lhs.coverage_gain != rhs.coverage_gain) {
    return lhs.coverage_gain > rhs.coverage_gain;
  }
  if (lhs.trial_rmse != rhs.trial_rmse) {
    return lhs.trial_rmse < rhs.trial_rmse;
  }
  if (lhs.frame_index != rhs.frame_index) {
    return lhs.frame_index < rhs.frame_index;
  }
  return lhs.board_id < rhs.board_id;
}

const JointSceneFrameState* FindSceneFrameState(
    const CalibrationStateBundle& bundle,
    int frame_index);
const JointSceneBoardState* FindSceneBoardState(
    const CalibrationStateBundle& bundle,
    int board_id);
double ComputeRotationAngleDeg(const Eigen::Matrix3d& rotation);
bool EstimatePoseForBenchmarkRefit(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<Eigen::Vector3d>& outer_targets,
    const std::vector<cv::Point2f>& outer_pixels,
    Eigen::Isometry3d* pose,
    double* rmse);

struct TrialObservationKey {
  int frame_index = -1;
  int board_id = -1;
  int point_id = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind =
      JointObservationSourceKind::OuterMeasurement;

  bool operator<(const TrialObservationKey& other) const {
    if (frame_index != other.frame_index) {
      return frame_index < other.frame_index;
    }
    if (board_id != other.board_id) {
      return board_id < other.board_id;
    }
    if (point_id != other.point_id) {
      return point_id < other.point_id;
    }
    if (source_point_index != other.source_point_index) {
      return source_point_index < other.source_point_index;
    }
    return static_cast<int>(source_kind) < static_cast<int>(other.source_kind);
  }
};

TrialObservationKey MakeTrialObservationKey(
    const JointPointObservation& point) {
  TrialObservationKey key;
  key.frame_index = point.frame_index;
  key.board_id = point.board_id;
  key.point_id = point.point_id;
  key.source_point_index = point.source_point_index;
  key.source_kind = point.source_kind;
  return key;
}

struct TrialObservationState {
  bool used_in_solver = false;
  bool present_in_solver_observations = false;
  JointRejectionReasonCode rejection_reason_code =
      JointRejectionReasonCode::None;
  std::string rejection_detail;
};

struct HierarchicalMeasurementCounts {
  int frame_count = 0;
  int board_observation_count = 0;
  int total_point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
};

HierarchicalMeasurementCounts ComputeHierarchicalMeasurementCounts(
    const JointMeasurementBuildResult& measurement_result) {
  HierarchicalMeasurementCounts counts;
  std::set<int> used_frame_indices;
  std::set<std::pair<int, int> > used_board_keys;
  for (const JointMeasurementFrameResult& frame : measurement_result.frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      bool board_used = false;
      for (const JointPointObservation& point : board.points) {
        if (!point.used_in_solver) {
          continue;
        }
        board_used = true;
        ++counts.total_point_count;
        if (point.point_type == JointPointType::Outer) {
          ++counts.outer_point_count;
        } else {
          ++counts.internal_point_count;
        }
      }
      if (board_used) {
        used_frame_indices.insert(frame.frame_index);
        used_board_keys.insert(std::make_pair(frame.frame_index, board.board_id));
      }
    }
  }
  counts.frame_count = static_cast<int>(used_frame_indices.size());
  counts.board_observation_count = static_cast<int>(used_board_keys.size());
  return counts;
}

std::map<TrialObservationKey, TrialObservationState> BuildObservationStateMap(
    const CalibrationMeasurementDataset& dataset) {
  std::map<TrialObservationKey, TrialObservationState> states;
  for (const JointMeasurementFrameResult& frame : dataset.frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      for (const JointPointObservation& point : board.points) {
        states[MakeTrialObservationKey(point)] =
            TrialObservationState{point.used_in_solver,
                                  false,
                                  point.rejection_reason_code,
                                  point.rejection_detail};
      }
    }
  }
  for (const JointPointObservation& point : dataset.solver_observations) {
    states[MakeTrialObservationKey(point)] =
        TrialObservationState{point.used_in_solver,
                              true,
                              point.rejection_reason_code,
                              point.rejection_detail};
  }
  return states;
}

double FindBoardObservationRmse(
    const JointResidualEvaluationResult& residual,
    const FrameBoardKey& key) {
  for (const JointResidualBoardObservationDiagnostics& diagnostics :
       residual.board_observation_diagnostics) {
    if (diagnostics.frame_index == key.first &&
        diagnostics.board_id == key.second) {
      return diagnostics.rmse;
    }
  }
  return std::numeric_limits<double>::infinity();
}

constexpr double kTrialSelectionImageCenterX = 2304.0;
constexpr double kTrialSelectionImageCenterY = 2304.0;
constexpr double kTrialSelectionHalfDiagonalPx = 3258.0;
constexpr double kTrialSelectionImageAreaPx =
    4608.0 * 4608.0;

double ComputeMedian(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t size = values.size();
  if (size % 2 == 1) {
    return values[size / 2];
  }
  return 0.5 * (values[size / 2 - 1] + values[size / 2]);
}

double ComputeMedianAbsoluteDeviation(const std::vector<double>& values,
                                      double median) {
  if (values.empty()) {
    return 0.0;
  }
  std::vector<double> deviations;
  deviations.reserve(values.size());
  for (double value : values) {
    deviations.push_back(std::abs(value - median));
  }
  return ComputeMedian(deviations);
}

std::set<FrameBoardKey> CollectAcceptedFrameBoardKeys(
    const CalibrationMeasurementDataset& dataset) {
  std::set<FrameBoardKey> keys;
  for (const JointMeasurementFrameResult& frame : dataset.frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      bool used = false;
      for (const JointPointObservation& point : board.points) {
        if (point.used_in_solver) {
          used = true;
          break;
        }
      }
      if (used) {
        keys.insert(std::make_pair(frame.frame_index, board.board_id));
      }
    }
  }
  return keys;
}

std::map<int, int> CountAcceptedObservationsByBoard(
    const CalibrationMeasurementDataset& dataset,
    const std::set<FrameBoardKey>& accepted_keys) {
  std::map<int, int> counts;
  for (const FrameBoardKey& key : accepted_keys) {
    ++counts[key.second];
  }
  for (const JointMeasurementFrameResult& frame : dataset.frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      if (counts.find(board.board_id) == counts.end()) {
        counts[board.board_id] = 0;
      }
    }
  }
  return counts;
}

std::map<int, int> CountFrameBoardObservationCapacity(
    const std::set<FrameBoardKey>& frame_board_keys) {
  std::map<int, int> counts;
  for (const FrameBoardKey& key : frame_board_keys) {
    ++counts[key.first];
  }
  return counts;
}

BoardObservationGeometrySummary SummarizeBoardObservationGeometry(
    const FrameBoardKey& key,
    const CalibrationMeasurementDataset& dataset) {
  BoardObservationGeometrySummary summary;
  for (const JointMeasurementFrameResult& frame : dataset.frames) {
    if (frame.frame_index != key.first) {
      continue;
    }
    summary.visible_board_count_in_frame =
        static_cast<int>(frame.board_observations.size());
    for (const JointBoardObservation& board : frame.board_observations) {
      if (board.board_id != key.second) {
        continue;
      }
      summary.found = true;
      double radius_sum = 0.0;
      double polar_sum = 0.0;
      std::vector<cv::Point2f> image_points;
      for (const JointPointObservation& point : board.points) {
        summary.center += point.image_xy;
        image_points.emplace_back(static_cast<float>(point.image_xy.x()),
                                  static_cast<float>(point.image_xy.y()));
        const double dx = point.image_xy.x() - kTrialSelectionImageCenterX;
        const double dy = point.image_xy.y() - kTrialSelectionImageCenterY;
        const double radius = std::sqrt(dx * dx + dy * dy);
        radius_sum += radius;
        summary.max_radius_px = std::max(summary.max_radius_px, radius);
        const double normalized_radius =
            std::max(0.0, std::min(1.0, radius / kTrialSelectionHalfDiagonalPx));
        const double polar_deg = normalized_radius * 90.0;
        polar_sum += polar_deg;
        summary.max_polar_angle_deg =
            std::max(summary.max_polar_angle_deg, polar_deg);
        ++summary.point_count;
      }
      if (summary.point_count > 0) {
        summary.center /= static_cast<double>(summary.point_count);
        summary.mean_radius_px =
            radius_sum / static_cast<double>(summary.point_count);
        summary.mean_polar_angle_deg =
            polar_sum / static_cast<double>(summary.point_count);
      }
      if (image_points.size() >= 3) {
        std::vector<cv::Point2f> hull;
        cv::convexHull(image_points, hull);
        if (hull.size() >= 3) {
          summary.projected_area_px = std::abs(cv::contourArea(hull));
          summary.projected_area_ratio =
              summary.projected_area_px / kTrialSelectionImageAreaPx;
        }
      }
      return summary;
    }
  }
  return summary;
}

std::vector<const JointPointObservation*> CollectFrameBoardPoints(
    const CalibrationMeasurementDataset& dataset,
    const FrameBoardKey& key,
    bool only_used_in_solver) {
  std::vector<const JointPointObservation*> points;
  for (const JointMeasurementFrameResult& frame : dataset.frames) {
    if (frame.frame_index != key.first) {
      continue;
    }
    for (const JointBoardObservation& board : frame.board_observations) {
      if (board.board_id != key.second) {
        continue;
      }
      for (const JointPointObservation& point : board.points) {
        if (only_used_in_solver && !point.used_in_solver) {
          continue;
        }
        points.push_back(&point);
      }
      return points;
    }
  }
  return points;
}

OuterBootstrapCameraIntrinsics PerturbIntrinsicsParameter(
    const OuterBootstrapCameraIntrinsics& camera,
    int parameter_index,
    double delta) {
  OuterBootstrapCameraIntrinsics perturbed = camera;
  switch (parameter_index) {
    case 0:
      perturbed.xi += delta;
      break;
    case 1:
      perturbed.alpha += delta;
      break;
    case 2:
      perturbed.fu += delta;
      break;
    case 3:
      perturbed.fv += delta;
      break;
    case 4:
      perturbed.cu += delta;
      break;
    case 5:
      perturbed.cv += delta;
      break;
    default:
      break;
  }
  return perturbed;
}

double IntrinsicsFiniteDifferenceStep(
    const OuterBootstrapCameraIntrinsics& camera,
    int parameter_index) {
  switch (parameter_index) {
    case 0:
    case 1:
      return 1e-4;
    case 2:
    case 3:
    case 4:
    case 5:
      return 1e-2 * std::max(1.0, std::max(camera.fu, camera.fv));
    default:
      return 1e-4;
  }
}

double IntrinsicsColumnScale(
    const OuterBootstrapCameraIntrinsics& camera,
    int parameter_index) {
  switch (parameter_index) {
    case 0:
    case 1:
      return 0.05;
    case 2:
      return 0.01 * std::max(1, camera.resolution.width);
    case 3:
      return 0.01 * std::max(1, camera.resolution.height);
    case 4:
      return 0.01 * std::max(1, camera.resolution.width);
    case 5:
      return 0.01 * std::max(1, camera.resolution.height);
    default:
      return 1.0;
  }
}

double RegularizedFisherLogDet(
    const Eigen::Matrix<double, 6, 6>& fisher) {
  const Eigen::Matrix<double, 6, 6> sym =
      0.5 * (fisher + fisher.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6> > solver(sym);
  if (solver.info() != Eigen::Success) {
    return 0.0;
  }
  double logdet = 0.0;
  for (int i = 0; i < 6; ++i) {
    logdet += std::log1p(std::max(0.0, solver.eigenvalues()[i]));
  }
  return logdet;
}

double FisherRankProxy(const Eigen::Matrix<double, 6, 6>& fisher) {
  const Eigen::Matrix<double, 6, 6> sym =
      0.5 * (fisher + fisher.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6> > solver(sym);
  if (solver.info() != Eigen::Success) {
    return 0.0;
  }
  double rank_proxy = 0.0;
  for (int i = 0; i < 6; ++i) {
    const double lambda = std::max(0.0, solver.eigenvalues()[i]);
    rank_proxy += lambda / (lambda + 1.0);
  }
  return rank_proxy;
}

IntrinsicsJacobianInformationSummary ComputeIntrinsicsJacobianInformation(
    const CalibrationStateBundle& bundle,
    const FrameBoardKey& key) {
  IntrinsicsJacobianInformationSummary summary;
  const JointSceneFrameState* frame_state =
      FindSceneFrameState(bundle, key.first);
  const JointSceneBoardState* board_state =
      FindSceneBoardState(bundle, key.second);
  if (frame_state == nullptr || board_state == nullptr ||
      !frame_state->initialized || !board_state->initialized ||
      !bundle.scene_state.camera.IsValid()) {
    return summary;
  }

  const Eigen::Isometry3d T_camera_reference(
      frame_state->T_camera_reference);
  const Eigen::Isometry3d T_reference_board(
      board_state->T_reference_board);
  const Eigen::Isometry3d T_camera_board =
      T_camera_reference * T_reference_board;
  const std::vector<const JointPointObservation*> points =
      CollectFrameBoardPoints(bundle.measurement_dataset, key, true);
  if (points.empty()) {
    return summary;
  }

  const DoubleSphereCameraModel base_model =
      DoubleSphereCameraModel::FromConfig(
          MakeIntermediateCameraConfig(bundle.scene_state.camera));
  if (!base_model.IsValid()) {
    return summary;
  }

  std::array<DoubleSphereCameraModel, 6> plus_models;
  std::array<DoubleSphereCameraModel, 6> minus_models;
  std::array<double, 6> steps{};
  std::array<double, 6> column_scales{};
  for (int param = 0; param < 6; ++param) {
    steps[static_cast<std::size_t>(param)] =
        IntrinsicsFiniteDifferenceStep(bundle.scene_state.camera, param);
    column_scales[static_cast<std::size_t>(param)] =
        IntrinsicsColumnScale(bundle.scene_state.camera, param);
    plus_models[static_cast<std::size_t>(param)] =
        DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(
            PerturbIntrinsicsParameter(
                bundle.scene_state.camera, param,
                steps[static_cast<std::size_t>(param)])));
    minus_models[static_cast<std::size_t>(param)] =
        DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(
            PerturbIntrinsicsParameter(
                bundle.scene_state.camera, param,
                -steps[static_cast<std::size_t>(param)])));
    if (!plus_models[static_cast<std::size_t>(param)].IsValid() ||
        !minus_models[static_cast<std::size_t>(param)].IsValid()) {
      return summary;
    }
  }

  Eigen::Matrix<double, 6, 6> Hcc =
      Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> Hcp =
      Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> Hpp =
      Eigen::Matrix<double, 6, 6>::Zero();
  constexpr double kTranslationStep = 1e-5;
  constexpr double kRotationStep = 1e-6;
  for (const JointPointObservation* point : points) {
    if (point == nullptr) {
      continue;
    }
    const Eigen::Vector3d point_camera =
        T_camera_board * point->target_xyz_board;
    Eigen::Matrix<double, 2, 6> jacobian =
        Eigen::Matrix<double, 2, 6>::Zero();
    Eigen::Matrix<double, 2, 6> pose_jacobian =
        Eigen::Matrix<double, 2, 6>::Zero();
    bool valid_point = true;
    for (int param = 0; param < 6; ++param) {
      Eigen::Vector2d plus = Eigen::Vector2d::Zero();
      Eigen::Vector2d minus = Eigen::Vector2d::Zero();
      if (!plus_models[static_cast<std::size_t>(param)]
               .vsEuclideanToKeypoint(point_camera, &plus) ||
          !minus_models[static_cast<std::size_t>(param)]
               .vsEuclideanToKeypoint(point_camera, &minus)) {
        valid_point = false;
        break;
      }
      const double inv_two_step =
          0.5 / std::max(steps[static_cast<std::size_t>(param)], 1e-12);
      jacobian.col(param) =
          (plus - minus) * inv_two_step *
          column_scales[static_cast<std::size_t>(param)];
    }
    if (!valid_point || !jacobian.allFinite()) {
      continue;
    }

    for (int axis = 0; axis < 3; ++axis) {
      Eigen::Vector3d plus_point = point_camera;
      Eigen::Vector3d minus_point = point_camera;
      plus_point[axis] += kTranslationStep;
      minus_point[axis] -= kTranslationStep;
      Eigen::Vector2d plus = Eigen::Vector2d::Zero();
      Eigen::Vector2d minus = Eigen::Vector2d::Zero();
      if (!base_model.vsEuclideanToKeypoint(plus_point, &plus) ||
          !base_model.vsEuclideanToKeypoint(minus_point, &minus)) {
        valid_point = false;
        break;
      }
      pose_jacobian.col(axis) =
          (plus - minus) * (0.5 / kTranslationStep);
    }
    for (int axis = 0; axis < 3 && valid_point; ++axis) {
      const Eigen::Vector3d unit_axis =
          Eigen::Vector3d::Unit(axis);
      const Eigen::Vector3d plus_point =
          Eigen::AngleAxisd(kRotationStep, unit_axis) * point_camera;
      const Eigen::Vector3d minus_point =
          Eigen::AngleAxisd(-kRotationStep, unit_axis) * point_camera;
      Eigen::Vector2d plus = Eigen::Vector2d::Zero();
      Eigen::Vector2d minus = Eigen::Vector2d::Zero();
      if (!base_model.vsEuclideanToKeypoint(plus_point, &plus) ||
          !base_model.vsEuclideanToKeypoint(minus_point, &minus)) {
        valid_point = false;
        break;
      }
      pose_jacobian.col(3 + axis) =
          (plus - minus) * (0.5 / kRotationStep);
    }
    if (!valid_point || !pose_jacobian.allFinite()) {
      continue;
    }
    Hcc.noalias() += jacobian.transpose() * jacobian;
    Hcp.noalias() += jacobian.transpose() * pose_jacobian;
    Hpp.noalias() += pose_jacobian.transpose() * pose_jacobian;
    ++summary.point_count;
  }

  if (summary.point_count <= 0) {
    return summary;
  }
  const double pose_damping =
      1e-9 * std::max(1.0, Hpp.trace() / 6.0);
  Hpp.diagonal().array() += pose_damping;
  const Eigen::LDLT<Eigen::Matrix<double, 6, 6> > pose_ldlt(Hpp);
  if (pose_ldlt.info() != Eigen::Success) {
    return IntrinsicsJacobianInformationSummary{};
  }
  const Eigen::Matrix<double, 6, 6> Hpp_inv_Hpc =
      pose_ldlt.solve(Hcp.transpose());
  if (!Hpp_inv_Hpc.allFinite()) {
    return IntrinsicsJacobianInformationSummary{};
  }
  Eigen::Matrix<double, 6, 6> marginalized_fisher =
      Hcc - Hcp * Hpp_inv_Hpc;
  marginalized_fisher =
      0.5 * (marginalized_fisher + marginalized_fisher.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6> > fisher_solver(
      marginalized_fisher);
  if (fisher_solver.info() != Eigen::Success) {
    return IntrinsicsJacobianInformationSummary{};
  }
  summary.fisher =
      fisher_solver.eigenvectors() *
      fisher_solver.eigenvalues().cwiseMax(0.0).asDiagonal() *
      fisher_solver.eigenvectors().transpose();
  summary.fisher /= static_cast<double>(summary.point_count);
  summary.trace = std::max(0.0, summary.fisher.trace());
  summary.logdet = RegularizedFisherLogDet(summary.fisher);
  summary.rank_proxy = FisherRankProxy(summary.fisher);
  summary.available = true;
  return summary;
}

std::set<std::pair<int, int> > CollectAcceptedGridCells(
    const CalibrationMeasurementDataset& dataset,
    const std::set<FrameBoardKey>& accepted_keys) {
  std::set<std::pair<int, int> > cells;
  for (const FrameBoardKey& key : accepted_keys) {
    const BoardObservationGeometrySummary summary =
        SummarizeBoardObservationGeometry(key, dataset);
    if (!summary.found || summary.point_count <= 0) {
      continue;
    }
    cells.insert(std::make_pair(
        static_cast<int>(std::floor(summary.center.x() / 320.0)),
        static_cast<int>(std::floor(summary.center.y() / 240.0))));
  }
  return cells;
}

std::set<int> CollectAcceptedFrameIds(
    const std::set<FrameBoardKey>& accepted_keys) {
  std::set<int> frame_ids;
  for (const FrameBoardKey& key : accepted_keys) {
    frame_ids.insert(key.first);
  }
  return frame_ids;
}

BackendDatasetStats RecomputeBackendDatasetStats(
    CalibrationMeasurementDataset* dataset) {
  if (dataset == nullptr) {
    throw std::runtime_error("RecomputeBackendDatasetStats requires a dataset.");
  }

  BackendDatasetStats stats;
  dataset->solver_observations.clear();

  for (JointMeasurementFrameResult& frame : dataset->frames) {
    bool frame_used = false;
    for (JointBoardObservation& board : frame.board_observations) {
      board.used_in_solver = false;
      board.outer_point_count = 0;
      board.internal_point_count = 0;
      for (JointPointObservation& point : board.points) {
        if (!point.used_in_solver) {
          continue;
        }
        board.used_in_solver = true;
        frame_used = true;
        dataset->solver_observations.push_back(point);
        stats.accepted_board_observation_keys.insert(
            std::make_pair(point.frame_index, point.board_id));
        ++stats.accepted_total_point_count;
        if (point.point_type == JointPointType::Outer) {
          ++stats.accepted_outer_point_count;
          ++board.outer_point_count;
        } else {
          ++stats.accepted_internal_point_count;
          ++board.internal_point_count;
        }
      }
    }
    if (frame_used) {
      stats.accepted_frame_indices.insert(frame.frame_index);
    }
  }

  stats.accepted_frame_count =
      static_cast<int>(stats.accepted_frame_indices.size());
  stats.accepted_board_observation_count =
      static_cast<int>(stats.accepted_board_observation_keys.size());

  dataset->accepted_frame_indices = stats.accepted_frame_indices;
  dataset->accepted_board_observation_keys =
      stats.accepted_board_observation_keys;
  dataset->accepted_frame_count = stats.accepted_frame_count;
  dataset->accepted_board_observation_count =
      stats.accepted_board_observation_count;
  dataset->accepted_outer_point_count = stats.accepted_outer_point_count;
  dataset->accepted_internal_point_count = stats.accepted_internal_point_count;
  dataset->accepted_total_point_count = stats.accepted_total_point_count;
  return stats;
}

JointMeasurementBuildResult BuildTrialMeasurementResultFromDataset(
    const CalibrationMeasurementDataset& dataset,
    int reference_board_id) {
  JointMeasurementBuildResult result;
  result.reference_board_id = reference_board_id;
  result.frames = dataset.frames;
  result.solver_observations = dataset.solver_observations;
  result.warnings = dataset.warnings;
  result.used_frame_count = dataset.accepted_frame_count;
  result.used_board_observation_count =
      dataset.accepted_board_observation_count;
  result.used_outer_point_count = dataset.accepted_outer_point_count;
  result.used_internal_point_count = dataset.accepted_internal_point_count;
  result.used_total_point_count = dataset.accepted_total_point_count;
  result.success = !result.frames.empty() && result.used_total_point_count > 0;
  if (!result.success) {
    result.failure_reason =
        dataset.failure_reason.empty()
            ? "CalibrationMeasurementDataset has no used-in-solver observations."
            : dataset.failure_reason;
  }
  return result;
}

CalibrationStateBundle BuildCandidateBackendPoolBundle(
    const CalibrationStateBundle& selected_bundle,
    const FrozenRoundArtifacts& artifacts) {
  CalibrationStateBundle candidate_bundle;
  candidate_bundle.success = selected_bundle.success;
  candidate_bundle.bundle_version = selected_bundle.bundle_version;
  candidate_bundle.baseline_protocol_label =
      selected_bundle.baseline_protocol_label;
  candidate_bundle.training_split_signature =
      selected_bundle.training_split_signature;
  candidate_bundle.scene_state = selected_bundle.scene_state;
  candidate_bundle.round_index = selected_bundle.round_index;
  candidate_bundle.measurement_dataset.reference_board_id =
      selected_bundle.measurement_dataset.reference_board_id;
  candidate_bundle.measurement_dataset.bundle_version =
      selected_bundle.measurement_dataset.bundle_version;
  candidate_bundle.measurement_dataset.baseline_protocol_label =
      selected_bundle.measurement_dataset.baseline_protocol_label;
  candidate_bundle.measurement_dataset.training_split_signature =
      selected_bundle.measurement_dataset.training_split_signature;
  candidate_bundle.measurement_dataset.dataset_label =
      selected_bundle.measurement_dataset.dataset_label;
  candidate_bundle.measurement_dataset.source_stage_label =
      selected_bundle.measurement_dataset.source_stage_label +
      "_candidate_backend_pool";
  candidate_bundle.measurement_dataset.frames =
      artifacts.measurement_result.frames;
  candidate_bundle.measurement_dataset.solver_observations.clear();
  candidate_bundle.warnings = selected_bundle.warnings;

  const auto selected_point_state =
      BuildObservationStateMap(selected_bundle.measurement_dataset);
  const std::set<FrameBoardKey> selected_frame_board_keys =
      CollectAcceptedFrameBoardKeys(selected_bundle.measurement_dataset);

  for (JointMeasurementFrameResult& frame :
       candidate_bundle.measurement_dataset.frames) {
    for (JointBoardObservation& board : frame.board_observations) {
      const FrameBoardKey frame_board_key(frame.frame_index, board.board_id);
      const bool is_selected_seed =
          selected_frame_board_keys.find(frame_board_key) !=
          selected_frame_board_keys.end();
      if (is_selected_seed) {
        continue;
      }

      board.used_in_solver = false;
      board.outer_point_count = 0;
      board.internal_point_count = 0;
      for (JointPointObservation& point : board.points) {
        if (point.used_in_solver) {
          point.rejection_reason_code = JointRejectionReasonCode::None;
          point.rejection_detail.clear();
        }
        if (point.used_in_solver) {
          board.used_in_solver = true;
          if (point.point_type == JointPointType::Outer) {
            ++board.outer_point_count;
          } else {
            ++board.internal_point_count;
          }
        }
      }
    }
  }

  for (JointMeasurementFrameResult& frame :
       candidate_bundle.measurement_dataset.frames) {
    for (JointBoardObservation& board : frame.board_observations) {
      const FrameBoardKey frame_board_key(frame.frame_index, board.board_id);
      if (selected_frame_board_keys.find(frame_board_key) ==
          selected_frame_board_keys.end()) {
        continue;
      }
      for (JointPointObservation& point : board.points) {
        const auto it =
            selected_point_state.find(MakeTrialObservationKey(point));
        if (it == selected_point_state.end()) {
          continue;
        }
        point.used_in_solver = it->second.used_in_solver;
        point.rejection_reason_code = it->second.rejection_reason_code;
        point.rejection_detail = it->second.rejection_detail;
      }
    }
  }

  for (JointPointObservation& point :
       candidate_bundle.measurement_dataset.solver_observations) {
    const auto it = selected_point_state.find(MakeTrialObservationKey(point));
    if (it == selected_point_state.end()) {
      continue;
    }
    point.used_in_solver = it->second.used_in_solver;
    point.rejection_reason_code = it->second.rejection_reason_code;
    point.rejection_detail = it->second.rejection_detail;
  }

  RecomputeBackendDatasetStats(&candidate_bundle.measurement_dataset);

  const JointMeasurementBuildResult candidate_measurement =
      BuildTrialMeasurementResultFromDataset(
          candidate_bundle.measurement_dataset,
          candidate_bundle.scene_state.reference_board_id);
  JointResidualEvaluationOptions residual_options;
  residual_options.top_k = 10;
  const JointReprojectionResidualEvaluator residual_evaluator(
      residual_options);
  candidate_bundle.residual_result = residual_evaluator.Evaluate(
      candidate_measurement,
      BuildJointSceneStateFromCalibrationSceneState(
          candidate_bundle.scene_state));
  candidate_bundle.ready_for_backend =
      candidate_bundle.measurement_dataset.accepted_total_point_count > 0 &&
      candidate_bundle.residual_result.success;
  candidate_bundle.success = candidate_bundle.ready_for_backend;
  if (!candidate_bundle.success) {
    candidate_bundle.failure_reason =
        candidate_bundle.residual_result.failure_reason.empty()
            ? "candidate backend pool bundle is not ready"
            : candidate_bundle.residual_result.failure_reason;
  }
  return candidate_bundle;
}

void ReevaluateCalibrationStateBundle(CalibrationStateBundle* bundle) {
  if (bundle == nullptr) {
    return;
  }
  RecomputeBackendDatasetStats(&bundle->measurement_dataset);
  const JointMeasurementBuildResult measurement =
      BuildTrialMeasurementResultFromDataset(
          bundle->measurement_dataset,
          bundle->scene_state.reference_board_id);
  JointResidualEvaluationOptions residual_options;
  residual_options.top_k = 10;
  const JointReprojectionResidualEvaluator residual_evaluator(
      residual_options);
  bundle->residual_result = residual_evaluator.Evaluate(
      measurement,
      BuildJointSceneStateFromCalibrationSceneState(bundle->scene_state));
  bundle->ready_for_backend =
      bundle->measurement_dataset.accepted_total_point_count > 0 &&
      bundle->residual_result.success;
  bundle->success = bundle->ready_for_backend;
  if (!bundle->success) {
    bundle->failure_reason =
        bundle->residual_result.failure_reason.empty()
            ? "bundle has no accepted solver observations"
            : bundle->residual_result.failure_reason;
  } else {
    bundle->failure_reason.clear();
  }
}

void DisableFrozenInternalObservationsForPerturbationAblation(
    CalibrationStateBundle* bundle) {
  if (bundle == nullptr) {
    return;
  }
  for (JointMeasurementFrameResult& frame :
       bundle->measurement_dataset.frames) {
    for (JointBoardObservation& board : frame.board_observations) {
      for (JointPointObservation& point : board.points) {
        if (point.point_type != JointPointType::Internal) {
          continue;
        }
        point.used_in_solver = false;
        point.rejection_detail =
            "frozen internal residual disabled after intrinsic perturbation";
      }
    }
  }
  ReevaluateCalibrationStateBundle(bundle);
}

CalibrationStateBundle BuildBundleForAcceptedFrameBoardKeys(
    const CalibrationStateBundle& scene_template_bundle,
    const CalibrationStateBundle& candidate_pool_bundle,
    const std::set<FrameBoardKey>& accepted_keys,
    const std::string& source_stage_label) {
  CalibrationStateBundle bundle = scene_template_bundle;
  bundle.measurement_dataset = candidate_pool_bundle.measurement_dataset;
  bundle.measurement_dataset.source_stage_label = source_stage_label;
  const auto scene_template_point_state =
      BuildObservationStateMap(scene_template_bundle.measurement_dataset);
  for (JointMeasurementFrameResult& frame : bundle.measurement_dataset.frames) {
    for (JointBoardObservation& board : frame.board_observations) {
      const FrameBoardKey key(frame.frame_index, board.board_id);
      const bool keep = accepted_keys.find(key) != accepted_keys.end();
      board.used_in_solver = false;
      board.outer_point_count = 0;
      board.internal_point_count = 0;
      for (JointPointObservation& point : board.points) {
        const bool original_used = point.used_in_solver;
        bool allowed_by_scene_template = true;
        const auto scene_template_it =
            scene_template_point_state.find(MakeTrialObservationKey(point));
        if (scene_template_it != scene_template_point_state.end()) {
          allowed_by_scene_template =
              scene_template_it->second.used_in_solver;
          if (!allowed_by_scene_template) {
            point.rejection_reason_code =
                scene_template_it->second.rejection_reason_code;
            point.rejection_detail =
                scene_template_it->second.rejection_detail;
          }
        }
        point.used_in_solver =
            keep && original_used && allowed_by_scene_template;
        if (!keep) {
          point.rejection_detail =
              "trial_backend_incremental_not_selected";
        }
      }
    }
  }
  ReevaluateCalibrationStateBundle(&bundle);
  return bundle;
}

CalibrationBundleMetadata BuildMetadataFromBundle(
    const CalibrationStateBundle& bundle,
    const std::string& source_pipeline_label) {
  CalibrationBundleMetadata metadata;
  metadata.bundle_version = bundle.bundle_version;
  metadata.baseline_protocol_label = bundle.baseline_protocol_label;
  metadata.training_split_signature = bundle.training_split_signature;
  metadata.dataset_label = bundle.scene_state.dataset_label;
  if (metadata.dataset_label.empty()) {
    metadata.dataset_label = bundle.measurement_dataset.dataset_label;
  }
  metadata.source_pipeline_label = source_pipeline_label;
  return metadata;
}

bool UpdateSceneTemplateFromAcceptedTrialBackend(
    const AslamBackendCalibrationResult& accepted_backend,
    const CalibrationStateBundle& metadata_source_bundle,
    const std::string& level,
    const std::string& source_pipeline_label,
    CalibrationStateBundle* scene_template_bundle) {
  if (scene_template_bundle == nullptr ||
      !accepted_backend.success ||
      !accepted_backend.optimized_scene_state.IsValid()) {
    return false;
  }
  const CalibrationBundleMetadata metadata =
      BuildMetadataFromBundle(metadata_source_bundle, source_pipeline_label);
  scene_template_bundle->scene_state = BuildCalibrationSceneState(
      accepted_backend.optimized_scene_state, level, metadata);
  return true;
}

JointReprojectionSceneState MergeAcceptedTrialStateIntoScene(
    const JointReprojectionSceneState& full_candidate_scene,
    const JointReprojectionSceneState& accepted_optimized_scene) {
  JointReprojectionSceneState merged = full_candidate_scene;
  if (!accepted_optimized_scene.IsValid()) {
    return merged;
  }
  merged.reference_board_id = accepted_optimized_scene.reference_board_id;
  merged.camera = accepted_optimized_scene.camera;
  std::map<int, JointSceneBoardState> optimized_boards_by_id;
  for (const JointSceneBoardState& board : accepted_optimized_scene.boards) {
    if (board.board_id >= 0 && board.initialized) {
      optimized_boards_by_id[board.board_id] = board;
    }
  }
  for (JointSceneBoardState& board : merged.boards) {
    const auto it = optimized_boards_by_id.find(board.board_id);
    if (it != optimized_boards_by_id.end()) {
      board = it->second;
    }
  }
  std::map<int, JointSceneFrameState> optimized_frames_by_index;
  for (const JointSceneFrameState& frame : accepted_optimized_scene.frames) {
    if (frame.frame_index >= 0 && frame.initialized) {
      optimized_frames_by_index[frame.frame_index] = frame;
    }
  }
  for (JointSceneFrameState& frame : merged.frames) {
    const auto it = optimized_frames_by_index.find(frame.frame_index);
    if (it != optimized_frames_by_index.end()) {
      frame = it->second;
    }
  }
  merged.warnings = accepted_optimized_scene.warnings;
  return merged;
}

bool UpdateSceneTemplateFromAcceptedTrialBackendMerged(
    const AslamBackendCalibrationResult& accepted_backend,
    const CalibrationStateBundle& full_candidate_bundle,
    const std::string& level,
    const std::string& source_pipeline_label,
    CalibrationStateBundle* scene_template_bundle) {
  if (scene_template_bundle == nullptr ||
      !accepted_backend.success ||
      !accepted_backend.optimized_scene_state.IsValid() ||
      !full_candidate_bundle.scene_state.IsValid()) {
    return false;
  }
  const JointReprojectionSceneState full_candidate_scene =
      BuildJointSceneStateFromCalibrationSceneState(
          full_candidate_bundle.scene_state);
  const JointReprojectionSceneState merged_scene =
      MergeAcceptedTrialStateIntoScene(
          full_candidate_scene,
          accepted_backend.optimized_scene_state);
  const CalibrationBundleMetadata metadata =
      BuildMetadataFromBundle(full_candidate_bundle, source_pipeline_label);
  scene_template_bundle->scene_state =
      BuildCalibrationSceneState(merged_scene, level, metadata);
  return true;
}

void MarkFrameBoardObservationUnused(CalibrationStateBundle* bundle,
                                     const FrameBoardKey& key,
                                     const std::string& rejection_detail) {
  if (bundle == nullptr) {
    return;
  }
  for (JointMeasurementFrameResult& frame : bundle->measurement_dataset.frames) {
    if (frame.frame_index != key.first) {
      continue;
    }
    for (JointBoardObservation& board : frame.board_observations) {
      if (board.board_id != key.second) {
        continue;
      }
      board.used_in_solver = false;
      board.outer_point_count = 0;
      board.internal_point_count = 0;
      for (JointPointObservation& point : board.points) {
        if (point.used_in_solver) {
          point.rejection_detail = rejection_detail;
        }
        point.used_in_solver = false;
      }
      return;
    }
  }
}

void ApplyMaxBoardsPerFrameAblation(CalibrationStateBundle* bundle,
                                    const std::set<FrameBoardKey>& seed_keys,
                                    int max_boards_per_frame,
                                    BackendInputAblationResult* result) {
  if (bundle == nullptr || max_boards_per_frame <= 0) {
    return;
  }

  struct FrameBoardCandidate {
    FrameBoardKey key;
    bool seed = false;
    double rmse = std::numeric_limits<double>::infinity();
  };

  std::map<int, std::vector<FrameBoardCandidate> > candidates_by_frame;
  for (const FrameBoardKey& key :
       bundle->measurement_dataset.accepted_board_observation_keys) {
    FrameBoardCandidate candidate;
    candidate.key = key;
    candidate.seed = seed_keys.find(key) != seed_keys.end();
    candidate.rmse = FindBoardObservationRmse(bundle->residual_result, key);
    candidates_by_frame[key.first].push_back(candidate);
  }

  int removed_count = 0;
  for (auto& entry : candidates_by_frame) {
    std::vector<FrameBoardCandidate>& candidates = entry.second;
    if (static_cast<int>(candidates.size()) <= max_boards_per_frame) {
      continue;
    }
    std::sort(candidates.begin(),
              candidates.end(),
              [](const FrameBoardCandidate& lhs,
                 const FrameBoardCandidate& rhs) {
                if (lhs.seed != rhs.seed) {
                  return lhs.seed && !rhs.seed;
                }
                const bool lhs_finite = std::isfinite(lhs.rmse);
                const bool rhs_finite = std::isfinite(rhs.rmse);
                if (lhs_finite != rhs_finite) {
                  return lhs_finite && !rhs_finite;
                }
                if (lhs.rmse != rhs.rmse) {
                  return lhs.rmse < rhs.rmse;
                }
                return lhs.key.second < rhs.key.second;
              });
    for (std::size_t index = static_cast<std::size_t>(max_boards_per_frame);
         index < candidates.size(); ++index) {
      MarkFrameBoardObservationUnused(
          bundle,
          candidates[index].key,
          "backend_input_max_boards_per_frame_ablation");
      ++removed_count;
    }
  }

  if (removed_count > 0 && result != nullptr) {
    std::ostringstream warning;
    warning << "Applied backend input max-boards-per-frame ablation: cap="
            << max_boards_per_frame
            << " removed_board_observations=" << removed_count;
    result->warnings.push_back(warning.str());
  }
  ReevaluateCalibrationStateBundle(bundle);
}

void ApplyPointBudgetControl(CalibrationStateBundle* bundle,
                             const CalibrationStateBundle& seed_bundle,
                             const BackendInputAblationOptions& options,
                             BackendInputAblationResult* result) {
  if (bundle == nullptr || !options.point_budget_control_enabled) {
    return;
  }

  const int target_total_points =
      options.point_budget_total_points > 0
          ? options.point_budget_total_points
          : seed_bundle.measurement_dataset.accepted_total_point_count;
  if (result != nullptr) {
    result->point_budget_total_points = target_total_points;
  }
  if (target_total_points <= 0) {
    if (result != nullptr) {
      result->warnings.push_back(
          "Point-budget control skipped because target_total_points <= 0.");
    }
    return;
  }

  const int current_outer =
      bundle->measurement_dataset.accepted_outer_point_count;
  const int current_internal =
      bundle->measurement_dataset.accepted_internal_point_count;
  const int target_internal =
      std::max(0, target_total_points - current_outer);
  if (current_outer > target_total_points && result != nullptr) {
    std::ostringstream warning;
    warning << "Point-budget target is below current outer point count; "
            << "kept all outer points and reduced internal target to 0. "
            << "target_total_points=" << target_total_points
            << " outer_points=" << current_outer;
    result->warnings.push_back(warning.str());
  }
  if (target_internal >= current_internal) {
    return;
  }

  std::vector<TrialObservationKey> internal_point_keys;
  internal_point_keys.reserve(static_cast<std::size_t>(current_internal));
  for (const JointMeasurementFrameResult& frame :
       bundle->measurement_dataset.frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      for (const JointPointObservation& point : board.points) {
        if (point.used_in_solver &&
            point.point_type == JointPointType::Internal) {
          internal_point_keys.push_back(MakeTrialObservationKey(point));
        }
      }
    }
  }

  std::mt19937 rng(options.point_budget_seed);
  std::shuffle(internal_point_keys.begin(), internal_point_keys.end(), rng);
  std::set<TrialObservationKey> kept_internal_keys;
  const int clamped_target_internal =
      std::max(0, std::min(target_internal,
                          static_cast<int>(internal_point_keys.size())));
  for (int index = 0; index < clamped_target_internal; ++index) {
    kept_internal_keys.insert(internal_point_keys[static_cast<std::size_t>(index)]);
  }

  int removed_internal_count = 0;
  for (JointMeasurementFrameResult& frame : bundle->measurement_dataset.frames) {
    for (JointBoardObservation& board : frame.board_observations) {
      for (JointPointObservation& point : board.points) {
        if (!point.used_in_solver ||
            point.point_type != JointPointType::Internal) {
          continue;
        }
        if (kept_internal_keys.find(MakeTrialObservationKey(point)) !=
            kept_internal_keys.end()) {
          continue;
        }
        point.used_in_solver = false;
        point.rejection_detail = "backend_input_point_budget_control";
        ++removed_internal_count;
      }
    }
  }

  if (removed_internal_count > 0 && result != nullptr) {
    std::ostringstream warning;
    warning << "Applied backend input point-budget control: target_total_points="
            << target_total_points
            << " target_internal_points=" << clamped_target_internal
            << " removed_internal_points=" << removed_internal_count;
    result->warnings.push_back(warning.str());
  }
  ReevaluateCalibrationStateBundle(bundle);
}

BackendInputAblationWorkingResult ApplyBackendInputAblationControls(
    const CalibrationStateBundle& selected_bundle,
    const CalibrationStateBundle& seed_bundle,
    const BackendInputAblationOptions& options) {
  BackendInputAblationWorkingResult working;
  working.bundle = selected_bundle;
  BackendInputAblationResult& result = working.result;
  result.enabled =
      options.point_budget_control_enabled ||
      options.max_boards_per_frame_for_ablation > 0;
  result.point_budget_control_enabled =
      options.point_budget_control_enabled;
  result.point_budget_total_points = options.point_budget_total_points;
  result.point_budget_seed = options.point_budget_seed;
  result.max_boards_per_frame_for_ablation =
      options.max_boards_per_frame_for_ablation;
  result.input_frame_count =
      selected_bundle.measurement_dataset.accepted_frame_count;
  result.input_board_observation_count =
      selected_bundle.measurement_dataset.accepted_board_observation_count;
  result.input_outer_point_count =
      selected_bundle.measurement_dataset.accepted_outer_point_count;
  result.input_internal_point_count =
      selected_bundle.measurement_dataset.accepted_internal_point_count;
  result.input_total_point_count =
      selected_bundle.measurement_dataset.accepted_total_point_count;

  if (result.enabled) {
    const std::set<FrameBoardKey> seed_keys =
        CollectAcceptedFrameBoardKeys(seed_bundle.measurement_dataset);
    ApplyMaxBoardsPerFrameAblation(
        &working.bundle,
        seed_keys,
        options.max_boards_per_frame_for_ablation,
        &result);
    ApplyPointBudgetControl(
        &working.bundle,
        seed_bundle,
        options,
        &result);
    if (!working.bundle.IsReadyForBackend()) {
      result.success = false;
      result.failure_reason =
          working.bundle.failure_reason.empty()
              ? "Backend input ablation produced a bundle that is not ready for backend."
              : working.bundle.failure_reason;
    }
  }

  result.output_frame_count =
      working.bundle.measurement_dataset.accepted_frame_count;
  result.output_board_observation_count =
      working.bundle.measurement_dataset.accepted_board_observation_count;
  result.output_outer_point_count =
      working.bundle.measurement_dataset.accepted_outer_point_count;
  result.output_internal_point_count =
      working.bundle.measurement_dataset.accepted_internal_point_count;
  result.output_total_point_count =
      working.bundle.measurement_dataset.accepted_total_point_count;
  result.removed_board_observation_count =
      std::max(0,
               result.input_board_observation_count -
                   result.output_board_observation_count);
  result.removed_internal_point_count =
      std::max(0,
               result.input_internal_point_count -
                   result.output_internal_point_count);
  return working;
}

TrialBackendMetrics ExtractTrialBackendMetrics(
    const AslamBackendCalibrationResult& backend_result) {
  TrialBackendMetrics metrics;
  metrics.success = backend_result.success;
  metrics.overall_rmse = backend_result.optimized_residual.overall_rmse;
  metrics.outer_rmse = backend_result.optimized_residual.outer_only_rmse;
  metrics.internal_rmse = backend_result.optimized_residual.internal_only_rmse;
  return metrics;
}

TrialBackendOptimizationDiagnostics SummarizeTrialBackendOptimization(
    const std::string& label,
    const AslamBackendCalibrationResult& backend_result) {
  TrialBackendOptimizationDiagnostics summary;
  summary.label = label;
  summary.success = backend_result.success;
  summary.design_variable_count = backend_result.design_variable_count;
  summary.error_term_count = backend_result.error_term_count;
  summary.initial_overall_rmse = backend_result.initial_residual.overall_rmse;
  summary.optimized_overall_rmse = backend_result.optimized_residual.overall_rmse;
  summary.initial_outer_rmse = backend_result.initial_residual.outer_only_rmse;
  summary.optimized_outer_rmse = backend_result.optimized_residual.outer_only_rmse;
  summary.initial_internal_rmse = backend_result.initial_residual.internal_only_rmse;
  summary.optimized_internal_rmse =
      backend_result.optimized_residual.internal_only_rmse;
  summary.camera_xi_before = backend_result.anchor_camera.xi;
  summary.camera_alpha_before = backend_result.anchor_camera.alpha;
  summary.camera_fu_before = backend_result.anchor_camera.fu;
  summary.camera_fv_before = backend_result.anchor_camera.fv;
  summary.camera_cu_before = backend_result.anchor_camera.cu;
  summary.camera_cv_before = backend_result.anchor_camera.cv;
  summary.camera_xi_after = backend_result.optimized_scene_state.camera.xi;
  summary.camera_alpha_after = backend_result.optimized_scene_state.camera.alpha;
  summary.camera_fu_after = backend_result.optimized_scene_state.camera.fu;
  summary.camera_fv_after = backend_result.optimized_scene_state.camera.fv;
  summary.camera_cu_after = backend_result.optimized_scene_state.camera.cu;
  summary.camera_cv_after = backend_result.optimized_scene_state.camera.cv;
  summary.stage_count = static_cast<int>(backend_result.stages.size());
  summary.failure_reason = backend_result.failure_reason;
  for (const AslamBackendOptimizationStageSummary& stage :
       backend_result.stages) {
    summary.total_iterations += stage.iterations;
    summary.total_failed_iterations += stage.failed_iterations;
    summary.any_intrinsics_stage =
        summary.any_intrinsics_stage || stage.optimize_intrinsics;
    summary.any_linear_solver_failure =
        summary.any_linear_solver_failure || stage.linear_solver_failure;
    summary.objective_start_sum += stage.objective_start;
    summary.objective_final_sum += stage.objective_final;
    summary.last_delta_x = stage.delta_x_final;
    summary.last_delta_j = stage.delta_j_final;
    summary.last_lm_lambda = stage.lm_lambda_final;
  }
  return summary;
}

AslamBackendCalibrationResult RunShortTrialBackend(
    const CalibrationStateBundle& bundle,
    const BackendProblemOptions& backend_options,
    const TrialBackendFrameBoardSelectionOptions& options,
    const std::string& label_suffix) {
  BackendProblemOptions trial_backend_options = backend_options;
  trial_backend_options.optimize_frame_poses = true;
  trial_backend_options.optimize_board_poses = true;
  trial_backend_options.optimize_intrinsics =
      options.optimize_intrinsics_in_trial;
  trial_backend_options.delayed_intrinsics_release =
      options.optimize_intrinsics_in_trial &&
      options.delayed_intrinsics_release_in_trial;
  trial_backend_options.intrinsics_release_iteration =
      std::max(0, options.intrinsics_release_iteration);
  CalibrationBackendProblemInput trial_input =
      BuildBackendProblemInput(bundle, trial_backend_options);
  trial_input.dataset_label += label_suffix;

  AslamBackendCalibrationOptions runner_options;
  runner_options.uniform_control_point_mode =
      options.single_board_dense_grid_profile;
  runner_options.max_iterations = std::max(1, options.max_iterations);
  runner_options.convergence_delta_j = 1e-3;
  runner_options.convergence_delta_x = 1e-4;
  runner_options.levenberg_marquardt_lambda_init = 1e-3;
  runner_options.linear_solver = "cholmod";
  runner_options.verbose = false;
  runner_options.use_huber_loss = true;
  const double checkerboard_huber_delta =
      options.single_board_dense_grid_profile
          ? std::max(0.0, options.checkerboard_huber_delta_pixels)
          : 0.0;
  runner_options.outer_huber_delta_pixels =
      checkerboard_huber_delta > 0.0 ? checkerboard_huber_delta : 10.0;
  runner_options.internal_huber_delta_pixels =
      checkerboard_huber_delta > 0.0 ? checkerboard_huber_delta : 6.0;
  runner_options.invalid_projection_penalty_pixels =
      kInvalidProjectionPenaltyPixels;
  runner_options.export_cost_parity_diagnostics = false;
  runner_options.run_jacobian_consistency_check = false;
  runner_options.skip_optimization = false;

  const AslamBackendCalibrationRunner runner(runner_options);
  return runner.Run(trial_input);
}

AslamBackendCalibrationOptions MakeTrialBackendRunnerOptions(
    const TrialBackendFrameBoardSelectionOptions& options,
    const AslamBackendCalibrationOptions& residual_options) {
  AslamBackendCalibrationOptions runner_options;
  runner_options.uniform_control_point_mode =
      residual_options.uniform_control_point_mode;
  runner_options.max_iterations = std::max(1, options.max_iterations);
  runner_options.convergence_delta_j = 1e-3;
  runner_options.convergence_delta_x = 1e-4;
  runner_options.levenberg_marquardt_lambda_init = 1e-3;
  runner_options.linear_solver = "cholmod";
  runner_options.verbose = false;
  runner_options.use_huber_loss = true;
  const double checkerboard_huber_delta =
      options.single_board_dense_grid_profile
          ? std::max(0.0, options.checkerboard_huber_delta_pixels)
          : 0.0;
  runner_options.outer_huber_delta_pixels =
      checkerboard_huber_delta > 0.0 ? checkerboard_huber_delta : 10.0;
  runner_options.internal_huber_delta_pixels =
      checkerboard_huber_delta > 0.0 ? checkerboard_huber_delta : 6.0;
  runner_options.invalid_projection_penalty_pixels =
      kInvalidProjectionPenaltyPixels;
  runner_options.export_cost_parity_diagnostics = false;
  runner_options.run_jacobian_consistency_check = false;
  runner_options.skip_optimization = false;
  runner_options.residual_model = residual_options.residual_model;
  runner_options.outer_huber_delta_radians =
      residual_options.outer_huber_delta_radians;
  runner_options.internal_huber_delta_radians =
      residual_options.internal_huber_delta_radians;
  runner_options.invalid_projection_penalty_radians =
      residual_options.invalid_projection_penalty_radians;
  runner_options.hybrid_angular_threshold_deg =
      residual_options.hybrid_angular_threshold_deg;
  runner_options.outer_residual_model = residual_options.outer_residual_model;
  runner_options.internal_residual_model =
      residual_options.internal_residual_model;
  runner_options.use_point_type_residual_split =
      residual_options.use_point_type_residual_split;
  runner_options.angular_auxiliary_enabled =
      residual_options.angular_auxiliary_enabled;
  runner_options.angular_auxiliary_weight =
      residual_options.angular_auxiliary_weight;
  runner_options.angular_auxiliary_normalized =
      residual_options.angular_auxiliary_normalized;
  runner_options.angular_auxiliary_apply_to_outer =
      residual_options.angular_auxiliary_apply_to_outer;
  runner_options.angular_auxiliary_apply_to_internal =
      residual_options.angular_auxiliary_apply_to_internal;
  runner_options.polar_continuous_hybrid_threshold_deg =
      residual_options.polar_continuous_hybrid_threshold_deg;
  runner_options.polar_continuous_hybrid_temperature_deg =
      residual_options.polar_continuous_hybrid_temperature_deg;
  runner_options.normalized_angular_reference_sigma_px =
      residual_options.normalized_angular_reference_sigma_px;
  runner_options.normalized_angular_min_sigma_rad =
      residual_options.normalized_angular_min_sigma_rad;
	  runner_options.normalized_angular_max_weight_scale =
	      residual_options.normalized_angular_max_weight_scale;
	  runner_options.pixel_residual_weight =
	      residual_options.pixel_residual_weight;
	  runner_options.chordal_residual_weight =
	      residual_options.chordal_residual_weight;
	  runner_options.angular_use_normalize_jacobian =
	      residual_options.angular_use_normalize_jacobian;
  runner_options.angular_local_whitening_enabled =
      residual_options.angular_local_whitening_enabled;
  runner_options.angular_local_whitening_pixel_sigma_px =
      residual_options.angular_local_whitening_pixel_sigma_px;
  runner_options.angular_local_whitening_covariance_damping =
      residual_options.angular_local_whitening_covariance_damping;
  runner_options.angular_local_whitening_min_sigma_rad =
      residual_options.angular_local_whitening_min_sigma_rad;
  runner_options.angular_local_whitening_max_weight =
      residual_options.angular_local_whitening_max_weight;
  runner_options.angular_observed_ray_mode =
      residual_options.angular_observed_ray_mode;
  return runner_options;
}

void CopyPersistentIncrementalSummary(
    const Stage5IncrementalBackendEstimatorResult& incremental_result,
    TrialBackendFrameBoardSelectionResult* result) {
  if (result == nullptr) {
    return;
  }
  result->persistent_incremental_backend_estimator_attempted =
      incremental_result.attempted;
  result->persistent_incremental_backend_estimator_compatible =
      incremental_result.compatible;
  result->persistent_incremental_backend_estimator_fallback_reason =
      incremental_result.fallback_reason;
  result->persistent_incremental_backend_estimator_failure_reason =
      incremental_result.failure_reason;
  result->persistent_incremental_information_gain_target =
      incremental_result.information_gain_target;
  result->persistent_incremental_board_layout_in_information_group =
      incremental_result.board_layout_in_information_group;
  result->persistent_incremental_board_layout_fixed =
      incremental_result.board_layout_fixed;
  result->persistent_incremental_board_layout_pose_count =
      incremental_result.board_layout_pose_count;
  result->persistent_incremental_board_layout_max_matrix_abs_delta =
      incremental_result.board_layout_max_matrix_abs_delta;
  result->persistent_incremental_board_layout_max_translation_delta =
      incremental_result.board_layout_max_translation_delta;
  result->persistent_incremental_board_layout_max_rotation_delta_deg =
      incremental_result.board_layout_max_rotation_delta_deg;
  result->persistent_incremental_camera_information_group_id =
      incremental_result.camera_information_group_id;
  result->persistent_incremental_board_layout_group_id =
      incremental_result.board_layout_group_id;
  result->persistent_incremental_transformation_group_id =
      incremental_result.transformation_group_id;
  result->persistent_incremental_seed_information_group_dim =
      incremental_result.seed_information_group_dim;
  result->persistent_incremental_seed_information_rank =
      incremental_result.seed_information_rank;
  result->persistent_incremental_seed_information_rank_deficiency =
      incremental_result.seed_information_rank_deficiency;
  result->persistent_incremental_seed_information_baseline_valid =
      incremental_result.seed_information_baseline_valid;
  result->persistent_incremental_seed_information_scaled_min_singular_value =
      incremental_result.seed_information_scaled_min_singular_value;
  result->persistent_incremental_seed_information_scaled_max_singular_value =
      incremental_result.seed_information_scaled_max_singular_value;
  result->persistent_incremental_seed_information_scaled_condition_number =
      incremental_result.seed_information_scaled_condition_number;
  result->persistent_incremental_seed_information_ds_cu_stddev_px =
      incremental_result.seed_information_ds_cu_stddev_px;
  result->persistent_incremental_seed_information_ds_cv_stddev_px =
      incremental_result.seed_information_ds_cv_stddev_px;
  result->persistent_incremental_seed_batch_count =
      incremental_result.seed_batch_count;
  result->persistent_incremental_seed_frame_count =
      incremental_result.seed_frame_count;
  result->persistent_incremental_seed_board_observation_count =
      incremental_result.seed_board_observation_count;
  result->persistent_incremental_seed_point_count =
      incremental_result.seed_point_count;
  result->persistent_incremental_seed_intrinsics_warmup_attempted =
      incremental_result.seed_intrinsics_warmup_attempted;
  result->persistent_incremental_seed_intrinsics_warmup_success =
      incremental_result.seed_intrinsics_warmup_success;
  result
      ->persistent_incremental_seed_intrinsics_warmup_converged_by_relative_objective =
      incremental_result.seed_intrinsics_warmup_converged_by_relative_objective;
  result->persistent_incremental_seed_intrinsics_warmup_iterations =
      incremental_result.seed_intrinsics_warmup_iterations;
  result->persistent_incremental_seed_intrinsics_warmup_objective_start =
      incremental_result.seed_intrinsics_warmup_objective_start;
  result->persistent_incremental_seed_intrinsics_warmup_objective_final =
      incremental_result.seed_intrinsics_warmup_objective_final;
  result->persistent_incremental_seed_intrinsics_warmup_last_delta_j =
      incremental_result.seed_intrinsics_warmup_last_delta_j;
  result->persistent_incremental_seed_intrinsics_warmup_last_delta_x =
      incremental_result.seed_intrinsics_warmup_last_delta_x;
  result->persistent_incremental_candidate_batch_count =
      incremental_result.candidate_batch_count;
  result->persistent_incremental_attempted_batch_count =
      incremental_result.attempted_batch_count;
  result->persistent_incremental_accepted_batch_count =
      incremental_result.accepted_batch_count;
  result->persistent_incremental_rejected_batch_count =
      incremental_result.rejected_batch_count;
  result->persistent_incremental_solver_profile_name =
      incremental_result.solver_profile_name;
  result->persistent_incremental_solver_objective_unit =
      incremental_result.solver_objective_unit;
  result->persistent_incremental_solver_max_iterations =
      incremental_result.solver_max_iterations;
  result->persistent_incremental_solver_convergence_delta_j =
      incremental_result.solver_convergence_delta_j;
  result->persistent_incremental_solver_convergence_delta_x =
      incremental_result.solver_convergence_delta_x;
  result->persistent_incremental_solver_bearing_reference_focal_px =
      incremental_result.solver_bearing_reference_focal_px;
  result->persistent_incremental_solver_bearing_residual_scale =
      incremental_result.solver_bearing_residual_scale;
  result->persistent_incremental_solver_single_iteration_batch_count =
      incremental_result.solver_single_iteration_batch_count;
  result->persistent_incremental_solver_max_iteration_batch_count =
      incremental_result.solver_max_iteration_batch_count;
  result->persistent_incremental_solver_objective_decreased_batch_count =
      incremental_result.solver_objective_decreased_batch_count;
  result
      ->persistent_incremental_solver_relative_objective_converged_batch_count =
      incremental_result.solver_relative_objective_converged_batch_count;
  result->persistent_incremental_solver_camera_step_converged_batch_count =
      incremental_result.solver_camera_step_converged_batch_count;
  result->persistent_incremental_solver_continuation_batch_count =
      incremental_result.solver_continuation_batch_count;
  result->persistent_incremental_solver_continuation_round_count =
      incremental_result.solver_continuation_round_count;
  result->persistent_incremental_solver_continuation_guard_hit_count =
      incremental_result.solver_continuation_guard_hit_count;
  result->persistent_incremental_image_plane_residual_count =
      incremental_result.image_plane_residual_count;
	  result->persistent_incremental_angular_residual_count =
	      incremental_result.angular_residual_count;
	  result->persistent_incremental_chordal_residual_count =
	      incremental_result.chordal_residual_count;
	  result->persistent_incremental_hybrid_angular_selected_count =
	      incremental_result.hybrid_angular_selected_count;
	  result->persistent_incremental_hybrid_chordal_selected_count =
	      incremental_result.hybrid_chordal_selected_count;
	  result->persistent_incremental_angular_geometry_failure_count =
	      incremental_result.angular_observation_geometry_failure_count;
  result->persistent_incremental_angular_local_whitening_success_count =
      incremental_result.angular_local_whitening_success_count;
  result->persistent_incremental_angular_local_whitening_failure_count =
      incremental_result.angular_local_whitening_failure_count;
  result->persistent_incremental_angular_local_whitening_clamped_count =
      incremental_result.angular_local_whitening_clamped_count;
  if (incremental_result.angular_local_whitening_success_count > 0) {
    const double count = static_cast<double>(
        incremental_result.angular_local_whitening_success_count);
    result->persistent_incremental_angular_local_whitening_sigma_mean_rad =
        incremental_result.angular_local_whitening_sigma_sum_rad / count;
    result->persistent_incremental_angular_local_whitening_sigma_min_rad =
        incremental_result.angular_local_whitening_sigma_min_rad;
    result->persistent_incremental_angular_local_whitening_sigma_max_rad =
        incremental_result.angular_local_whitening_sigma_max_rad;
    result->persistent_incremental_angular_local_whitening_weight_mean =
        incremental_result.angular_local_whitening_weight_sum / count;
    result->persistent_incremental_angular_local_whitening_weight_min =
        incremental_result.angular_local_whitening_weight_min;
    result->persistent_incremental_angular_local_whitening_weight_max =
        incremental_result.angular_local_whitening_weight_max;
  }
  result->persistent_incremental_selection_metric_name =
      incremental_result.selection_metric_name;
  result->persistent_incremental_selection_metric_unit =
      incremental_result.selection_metric_unit;
  result->persistent_incremental_residual_health_threshold_source =
      incremental_result.residual_health_threshold_source;
  result->persistent_incremental_residual_health_threshold_metric =
      incremental_result.residual_health_threshold_metric;
  result->persistent_incremental_seed_acceptance_metric_rmse =
      incremental_result.seed_acceptance_metric_rmse;
  result->persistent_incremental_seed_acceptance_metric_p95 =
      incremental_result.seed_acceptance_metric_p95;
  result->persistent_incremental_trust_region_backtracking_batch_count =
      incremental_result.trust_region_backtracking_batch_count;
  result->persistent_incremental_trust_region_backtracking_attempt_count =
      incremental_result.trust_region_backtracking_attempt_count;
  result->persistent_incremental_trust_region_backtracking_accepted_count =
      incremental_result.trust_region_backtracking_accepted_count;
  result->persistent_incremental_trust_region_backtracking_max_anchor_scale =
      incremental_result.trust_region_backtracking_max_anchor_scale;
  result
      ->persistent_incremental_normalize_information_gain_by_board_observation =
      incremental_result.normalize_information_gain_by_board_observation;
  result->persistent_incremental_split_residual_health_gate_enabled =
      incremental_result.split_residual_health_gate_enabled;
  result->persistent_incremental_split_residual_health_rejected_count =
      incremental_result.split_residual_health_rejected_count;
  result->persistent_incremental_bearing_pixel_safety_gate_enabled =
      incremental_result.bearing_pixel_safety_gate_enabled;
  result->persistent_incremental_bearing_pixel_safety_rejected_count =
      incremental_result.bearing_pixel_safety_rejected_count;
  result->persistent_incremental_full_training_pose_refit_health_gate_enabled =
      incremental_result.full_training_pose_refit_health_gate_enabled;
  result
      ->persistent_incremental_seed_intrinsics_warmup_full_training_health_pass =
      incremental_result.seed_intrinsics_warmup_full_training_health_pass;
  result
      ->persistent_incremental_full_training_pose_refit_health_rejected_count =
      incremental_result.full_training_pose_refit_health_rejected_count;
  result->persistent_incremental_initial_full_training_pixel_rmse =
      incremental_result.initial_full_training_pixel_rmse;
  result->persistent_incremental_initial_full_training_pixel_p95 =
      incremental_result.initial_full_training_pixel_p95;
  result->persistent_incremental_initial_full_training_pose_success_rate =
      incremental_result.initial_full_training_pose_success_rate;
  result->persistent_incremental_initial_full_training_pose_success_count =
      incremental_result.initial_full_training_pose_success_count;
  result->persistent_incremental_initial_full_training_pose_total_count =
      incremental_result.initial_full_training_pose_total_count;
  result->persistent_incremental_initial_full_training_invalid_projection_count =
      incremental_result.initial_full_training_invalid_projection_count;
  result->persistent_incremental_final_full_training_pixel_rmse =
      incremental_result.final_full_training_pixel_rmse;
  result->persistent_incremental_final_full_training_pixel_p95 =
      incremental_result.final_full_training_pixel_p95;
  result->persistent_incremental_final_full_training_pose_success_rate =
      incremental_result.final_full_training_pose_success_rate;
  result->persistent_incremental_final_full_training_pose_success_count =
      incremental_result.final_full_training_pose_success_count;
  result->persistent_incremental_final_full_training_pose_total_count =
      incremental_result.final_full_training_pose_total_count;
  result->persistent_incremental_final_full_training_invalid_projection_count =
      incremental_result.final_full_training_invalid_projection_count;
  result->persistent_incremental_kb_distortion_guard_enabled =
      incremental_result.kb_distortion_guard_enabled;
  result->persistent_incremental_kb_ray_curve_validity_rejected_count =
      incremental_result.kb_ray_curve_validity_rejected_count;
  result->persistent_incremental_adaptive_saturation_stop_enabled =
      incremental_result.adaptive_saturation_stop_enabled;
  result->persistent_incremental_adaptive_saturation_stop_hit =
      incremental_result.adaptive_saturation_stop_hit;
  result->persistent_incremental_adaptive_saturation_min_accepted_batches =
      incremental_result.adaptive_saturation_min_accepted_batches;
  result->persistent_incremental_adaptive_saturation_nonproductive_batch_limit =
      incremental_result.adaptive_saturation_nonproductive_batch_limit;
  result
      ->persistent_incremental_adaptive_saturation_consecutive_nonproductive_batches =
      incremental_result.adaptive_saturation_consecutive_nonproductive_batches;
  result
      ->persistent_incremental_adaptive_saturation_tail_ordering_score_threshold =
      incremental_result.adaptive_saturation_tail_ordering_score_threshold;
  result->persistent_incremental_adaptive_saturation_next_ordering_score =
      incremental_result.adaptive_saturation_next_ordering_score;
  result->persistent_incremental_adaptive_saturation_stop_reason =
      incremental_result.adaptive_saturation_stop_reason;
  result->persistent_incremental_total_elapsed_time_seconds =
      incremental_result.total_elapsed_time_seconds;
}

double ComputeFrameBoardCoverageGain(
    const FrameBoardKey& key,
    const CalibrationMeasurementDataset& candidate_dataset,
    const std::set<FrameBoardKey>& accepted_keys) {
  double gain = 0.0;
  bool frame_seen = false;
  bool board_seen = false;
  bool frame_board_counted = false;
  Eigen::Vector2d candidate_center = Eigen::Vector2d::Zero();
  int candidate_point_count = 0;
  for (const JointMeasurementFrameResult& frame : candidate_dataset.frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      const FrameBoardKey current_key(frame.frame_index, board.board_id);
      const bool accepted = accepted_keys.find(current_key) != accepted_keys.end();
      if (accepted) {
        if (frame.frame_index == key.first) {
          frame_seen = true;
        }
        if (board.board_id == key.second) {
          board_seen = true;
        }
      }
      if (current_key != key) {
        continue;
      }
      for (const JointPointObservation& point : board.points) {
        candidate_center += point.image_xy;
        ++candidate_point_count;
      }
      frame_board_counted = true;
    }
  }
  if (!frame_board_counted || candidate_point_count <= 0) {
    return 0.0;
  }
  candidate_center /= static_cast<double>(candidate_point_count);
  if (!frame_seen) {
    gain += 1.0;
  }
  if (!board_seen) {
    gain += 0.5;
  }

  const int candidate_cell_x =
      static_cast<int>(std::floor(candidate_center.x() / 320.0));
  const int candidate_cell_y =
      static_cast<int>(std::floor(candidate_center.y() / 240.0));
  bool cell_seen = false;
  for (const JointMeasurementFrameResult& frame : candidate_dataset.frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      const FrameBoardKey current_key(frame.frame_index, board.board_id);
      if (accepted_keys.find(current_key) == accepted_keys.end()) {
        continue;
      }
      Eigen::Vector2d center = Eigen::Vector2d::Zero();
      int point_count = 0;
      for (const JointPointObservation& point : board.points) {
        center += point.image_xy;
        ++point_count;
      }
      if (point_count <= 0) {
        continue;
      }
      center /= static_cast<double>(point_count);
      const int cell_x =
          static_cast<int>(std::floor(center.x() / 320.0));
      const int cell_y =
          static_cast<int>(std::floor(center.y() / 240.0));
      if (cell_x == candidate_cell_x && cell_y == candidate_cell_y) {
        cell_seen = true;
        break;
      }
    }
    if (cell_seen) {
      break;
    }
  }
  if (!cell_seen) {
    gain += 0.5;
  }
  return gain;
}

CandidateCoverageScoreBreakdown ComputeCandidateCoverageScore(
    const FrameBoardKey& key,
    const CalibrationMeasurementDataset& candidate_dataset,
    const std::set<FrameBoardKey>& accepted_keys,
    double trial_rmse,
    double threshold_px,
    bool use_pixel_trial_residual_quality) {
  CandidateCoverageScoreBreakdown score;
  const BoardObservationGeometrySummary geometry =
      SummarizeBoardObservationGeometry(key, candidate_dataset);
  if (!geometry.found || geometry.point_count <= 0) {
    return score;
  }

  const std::set<int> accepted_frame_ids =
      CollectAcceptedFrameIds(accepted_keys);
  const std::map<int, int> board_counts =
      CountAcceptedObservationsByBoard(candidate_dataset, accepted_keys);
  const std::set<std::pair<int, int> > grid_cells =
      CollectAcceptedGridCells(candidate_dataset, accepted_keys);

  score.mean_polar_angle_deg = geometry.mean_polar_angle_deg;
  score.max_polar_angle_deg = geometry.max_polar_angle_deg;
  score.frame_novelty_gain =
      accepted_frame_ids.find(key.first) == accepted_frame_ids.end() ? 1.0 : 0.0;

  const auto board_it = board_counts.find(key.second);
  const int board_count =
      board_it == board_counts.end() ? 0 : board_it->second;
  int max_board_count = 0;
  for (const auto& entry : board_counts) {
    max_board_count = std::max(max_board_count, entry.second);
  }
  if (max_board_count > 0) {
    score.board_balance_gain =
        std::max(0.0,
                 static_cast<double>(max_board_count - board_count) /
                     static_cast<double>(max_board_count));
  }

  const std::pair<int, int> cell(
      static_cast<int>(std::floor(geometry.center.x() / 320.0)),
      static_cast<int>(std::floor(geometry.center.y() / 240.0)));
  score.grid_gain = grid_cells.find(cell) == grid_cells.end() ? 1.0 : 0.0;

  const double mean_polar_norm =
      std::max(0.0, std::min(1.0, geometry.mean_polar_angle_deg / 70.0));
  score.polar_gain = mean_polar_norm;
  score.edge_gain =
      geometry.max_polar_angle_deg >= 50.0
          ? std::max(0.0, std::min(1.0,
                                   (geometry.max_polar_angle_deg - 50.0) /
                                       35.0))
          : 0.0;
  score.covisibility_gain =
      geometry.visible_board_count_in_frame >= 2
          ? std::min(1.0,
                     static_cast<double>(
                         geometry.visible_board_count_in_frame - 1) /
                         4.0)
          : 0.0;
  const double safe_threshold = std::max(1e-6, threshold_px);
  score.residual_quality_score =
      use_pixel_trial_residual_quality
          ? std::max(0.0, 1.0 - std::min(1.0, trial_rmse / safe_threshold))
          : 0.5;
  score.coverage_gain =
      0.9 * score.polar_gain +
      0.8 * score.edge_gain +
      0.8 * score.board_balance_gain +
      0.6 * score.frame_novelty_gain +
      0.5 * score.grid_gain +
      0.5 * score.covisibility_gain;
  score.total_score =
      score.coverage_gain +
      (use_pixel_trial_residual_quality ? 0.8 * score.residual_quality_score
                                        : 0.0);
  return score;
}

double Clamp01(double value) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  return std::max(0.0, std::min(1.0, value));
}

int PolarAngleBin(double angle_deg) {
  if (!std::isfinite(angle_deg)) {
    return -1;
  }
  return static_cast<int>(std::floor(
      std::max(0.0, std::min(100.0, angle_deg)) / 10.0));
}

int ProjectedAreaRatioBin(double area_ratio) {
  if (!std::isfinite(area_ratio) || area_ratio <= 0.0) {
    return -1;
  }
  if (area_ratio < 0.005) {
    return 0;
  }
  if (area_ratio < 0.015) {
    return 1;
  }
  if (area_ratio < 0.030) {
    return 2;
  }
  if (area_ratio < 0.060) {
    return 3;
  }
  if (area_ratio < 0.100) {
    return 4;
  }
  return 5;
}

bool IsIntrinsicsDiversityAnchorGeometry(
    const BoardObservationGeometrySummary& geometry) {
  if (!geometry.found || geometry.point_count <= 0) {
    return false;
  }
  return geometry.max_polar_angle_deg >= 50.0 ||
         geometry.projected_area_ratio >= 0.04;
}

std::pair<int, int> ImageCoverageCell(const Eigen::Vector2d& center) {
  return std::make_pair(
      static_cast<int>(std::floor(center.x() / 320.0)),
      static_cast<int>(std::floor(center.y() / 240.0)));
}

IntrinsicsInformationGainProxyBreakdown ComputeIntrinsicsInformationGainProxy(
    const TrialBackendFrameBoardObservationDecision& candidate,
    const CalibrationMeasurementDataset& candidate_dataset,
    const std::set<FrameBoardKey>& accepted_keys,
    const std::map<int, int>& accepted_observation_count_by_frame,
    const std::map<int, int>& candidate_pool_board_capacity_by_frame,
    const Eigen::Matrix<double, 6, 6>& accepted_intrinsics_fisher,
    const IntrinsicsJacobianInformationSummary& candidate_intrinsics_info,
    double min_candidate_score,
    TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode
        info_gain_proxy_mode) {
  IntrinsicsInformationGainProxyBreakdown gain;
  const FrameBoardKey key(candidate.frame_index, candidate.board_id);
  const BoardObservationGeometrySummary geometry =
      SummarizeBoardObservationGeometry(key, candidate_dataset);
  if (!geometry.found || geometry.point_count <= 0) {
    return gain;
  }

  std::set<std::pair<int, int> > accepted_cells;
  std::set<int> accepted_mean_polar_bins;
  std::set<int> accepted_max_polar_bins;
  std::set<int> accepted_area_bins;
  int selected_board_count = 0;
  int selected_frame_observation_count = 0;
  for (const FrameBoardKey& accepted_key : accepted_keys) {
    const BoardObservationGeometrySummary accepted_geometry =
        SummarizeBoardObservationGeometry(accepted_key, candidate_dataset);
    if (!accepted_geometry.found || accepted_geometry.point_count <= 0) {
      continue;
    }
    accepted_cells.insert(ImageCoverageCell(accepted_geometry.center));
    const int mean_polar_bin =
        PolarAngleBin(accepted_geometry.mean_polar_angle_deg);
    if (mean_polar_bin >= 0) {
      accepted_mean_polar_bins.insert(mean_polar_bin);
    }
    const int max_polar_bin =
        PolarAngleBin(accepted_geometry.max_polar_angle_deg);
    if (max_polar_bin >= 0) {
      accepted_max_polar_bins.insert(max_polar_bin);
    }
    const int area_bin =
        ProjectedAreaRatioBin(accepted_geometry.projected_area_ratio);
    if (area_bin >= 0) {
      accepted_area_bins.insert(area_bin);
    }
    if (accepted_key.second == candidate.board_id) {
      ++selected_board_count;
    }
    if (accepted_key.first == candidate.frame_index) {
      ++selected_frame_observation_count;
    }
  }

  const double score_denominator = std::max(min_candidate_score, 1.0);
  const double raw_score_term =
      std::isfinite(candidate.candidate_score)
          ? candidate.candidate_score / score_denominator
          : 0.0;
  gain.score_term = std::min(2.0, std::max(0.0, raw_score_term));

  const std::pair<int, int> cell = ImageCoverageCell(geometry.center);
  const int mean_polar_bin = PolarAngleBin(geometry.mean_polar_angle_deg);
  const int max_polar_bin = PolarAngleBin(geometry.max_polar_angle_deg);
  const int area_bin = ProjectedAreaRatioBin(geometry.projected_area_ratio);
  const bool new_image_cell = accepted_cells.find(cell) == accepted_cells.end();
  const bool new_mean_polar_bin =
      mean_polar_bin >= 0 &&
      accepted_mean_polar_bins.find(mean_polar_bin) ==
          accepted_mean_polar_bins.end();
  const bool new_max_polar_bin =
      max_polar_bin >= 0 &&
      accepted_max_polar_bins.find(max_polar_bin) ==
          accepted_max_polar_bins.end();
  const bool new_area_bin =
      area_bin >= 0 &&
      accepted_area_bins.find(area_bin) == accepted_area_bins.end();

  const double image_cell_bonus = new_image_cell ? 0.80 : 0.0;
  const double polar_bin_bonus = new_mean_polar_bin ? 0.75 : 0.0;
  const double scale_bin_bonus = new_area_bin ? 0.55 : 0.0;
  const double edge_bin_bonus =
      new_max_polar_bin && geometry.max_polar_angle_deg >= 50.0 ? 0.35 : 0.0;
  const double diversity_anchor_bonus =
      info_gain_proxy_mode ==
              TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode
                  ::IntrinsicsJacobian &&
              IsIntrinsicsDiversityAnchorGeometry(geometry)
          ? 0.75
          : 0.0;
  const double board_balance_bonus =
      0.35 * Clamp01(candidate.board_balance_gain);
  const double covisibility_bonus =
      0.20 * Clamp01(candidate.covisibility_gain);
  const double frame_novelty_bonus =
      selected_frame_observation_count == 0 ? 0.20 : 0.0;
  const double score_hint_bonus = 0.10 * gain.score_term;
  gain.coverage_term =
      image_cell_bonus + polar_bin_bonus + scale_bin_bonus + edge_bin_bonus +
      diversity_anchor_bonus + board_balance_bonus + covisibility_bonus +
      frame_novelty_bonus + score_hint_bonus;

  if (info_gain_proxy_mode ==
          TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode
              ::IntrinsicsJacobian &&
      candidate_intrinsics_info.available) {
    const Eigen::Matrix<double, 6, 6> combined_fisher =
        accepted_intrinsics_fisher + candidate_intrinsics_info.fisher;
    const double accepted_logdet =
        RegularizedFisherLogDet(accepted_intrinsics_fisher);
    const double combined_logdet =
        RegularizedFisherLogDet(combined_fisher);
    const double accepted_rank =
        FisherRankProxy(accepted_intrinsics_fisher);
    const double combined_rank =
        FisherRankProxy(combined_fisher);
    gain.intrinsics_jacobian_logdet_gain =
        std::max(0.0, combined_logdet - accepted_logdet);
    gain.intrinsics_jacobian_trace_gain =
        std::max(0.0, combined_fisher.trace() -
                          accepted_intrinsics_fisher.trace());
    gain.intrinsics_jacobian_rank_gain =
        std::max(0.0, combined_rank - accepted_rank);
    const double logdet_term =
        std::min(2.0, gain.intrinsics_jacobian_logdet_gain / 1.5);
    const double trace_term =
        std::min(1.0, std::log1p(gain.intrinsics_jacobian_trace_gain) / 6.0);
    const double rank_term =
        std::min(1.0, gain.intrinsics_jacobian_rank_gain / 0.5);
    gain.intrinsics_jacobian_info_term =
        logdet_term + 0.35 * trace_term + 0.35 * rank_term;
  }

  const auto frame_capacity_it =
      candidate_pool_board_capacity_by_frame.find(candidate.frame_index);
  const int observed_board_capacity =
      frame_capacity_it == candidate_pool_board_capacity_by_frame.end()
          ? 0
          : frame_capacity_it->second;
  const auto selected_frame_it =
      accepted_observation_count_by_frame.find(candidate.frame_index);
  const int accepted_frame_board_count =
      selected_frame_it == accepted_observation_count_by_frame.end()
          ? 0
          : selected_frame_it->second;
  gain.frame_completion_bonus =
      accepted_frame_board_count > 0 &&
              accepted_frame_board_count < observed_board_capacity
          ? 0.35
          : 0.0;
  gain.new_board_bonus = selected_board_count == 0 ? 0.35 : 0.0;
  gain.information_gain_proxy =
      gain.coverage_term + gain.intrinsics_jacobian_info_term +
      gain.frame_completion_bonus + gain.new_board_bonus;
  return gain;
}

CandidateConsistencyScoreBreakdown ComputeCandidateConsistencyScore(
    const FrameBoardKey& key,
    const CalibrationStateBundle& candidate_pool_bundle,
    const TrialBackendFrameBoardSelectionOptions& options) {
  CandidateConsistencyScoreBreakdown score;
  if (!options.use_consistency_score) {
    return score;
  }

  const JointSceneFrameState* frame_state =
      FindSceneFrameState(candidate_pool_bundle, key.first);
  const JointSceneBoardState* board_state =
      FindSceneBoardState(candidate_pool_bundle, key.second);
  if (frame_state == nullptr || !frame_state->initialized ||
      board_state == nullptr || !board_state->initialized) {
    return score;
  }

  std::vector<Eigen::Vector3d> outer_targets;
  std::vector<cv::Point2f> outer_pixels;
  for (const JointMeasurementFrameResult& frame :
       candidate_pool_bundle.measurement_dataset.frames) {
    if (frame.frame_index != key.first) {
      continue;
    }
    for (const JointBoardObservation& board : frame.board_observations) {
      if (board.board_id != key.second) {
        continue;
      }
      for (const JointPointObservation& point : board.points) {
        if (!point.used_in_solver || point.point_type != JointPointType::Outer) {
          continue;
        }
        outer_targets.push_back(point.target_xyz_board);
        outer_pixels.emplace_back(
            static_cast<float>(point.image_xy.x()),
            static_cast<float>(point.image_xy.y()));
      }
      break;
    }
    break;
  }

  if (outer_targets.size() < 4) {
    return score;
  }

  Eigen::Isometry3d T_camera_board_local = Eigen::Isometry3d::Identity();
  double local_outer_rmse = 0.0;
  if (!EstimatePoseForBenchmarkRefit(candidate_pool_bundle.scene_state.camera,
                                     outer_targets,
                                     outer_pixels,
                                     &T_camera_board_local,
                                     &local_outer_rmse)) {
    return score;
  }

  const Eigen::Isometry3d T_camera_reference(
      frame_state->T_camera_reference);
  const Eigen::Isometry3d T_reference_board_global(
      board_state->T_reference_board);
  const Eigen::Isometry3d T_reference_board_obs =
      T_camera_reference.inverse() * T_camera_board_local;
  const Eigen::Isometry3d delta =
      T_reference_board_global.inverse() * T_reference_board_obs;

  score.available = true;
  score.local_outer_rmse = local_outer_rmse;
  score.translation_error_mm = delta.translation().norm() * 1000.0;
  score.rotation_error_deg = ComputeRotationAngleDeg(delta.rotation());

  const double translation_sigma =
      std::max(1e-6, options.consistency_translation_sigma_mm);
  const double rotation_sigma =
      std::max(1e-6, options.consistency_rotation_sigma_deg);
  const double t_norm = score.translation_error_mm / translation_sigma;
  const double r_norm = score.rotation_error_deg / rotation_sigma;
  const double e_cons = std::sqrt(t_norm * t_norm + r_norm * r_norm);
  score.score = 1.0 / (1.0 + e_cons * e_cons);
  score.penalty = options.consistency_penalty_weight * (1.0 - score.score);
  return score;
}

bool IsForceIncludeFrameBoardCandidate(
    const FrameBoardKey& key,
    const std::string& frame_label,
    const TrialBackendFrameBoardSelectionOptions& options) {
  if (options.force_include_frame_board_keys.find(key) !=
      options.force_include_frame_board_keys.end()) {
    return true;
  }
  return options.force_include_frame_label_board_keys.find(
             std::make_pair(frame_label, key.second)) !=
         options.force_include_frame_label_board_keys.end();
}

std::map<FrameBoardKey, TrialBoardRmseAccumulator> ComputeTrialBoardResiduals(
    const AslamBackendCalibrationResult& backend_result) {
  std::map<FrameBoardKey, TrialBoardRmseAccumulator> accumulators;
  for (const JointResidualPointDiagnostics& point :
       backend_result.optimized_residual.point_diagnostics) {
    if (!point.used_in_solver) {
      continue;
    }
    const FrameBoardKey key(point.frame_index, point.board_id);
    TrialBoardRmseAccumulator& accumulator = accumulators[key];
    accumulator.frame_label = point.frame_label;
    ++accumulator.point_count;
    accumulator.squared_error_sum +=
        point.residual_norm * point.residual_norm;
    if (point.point_type == JointPointType::Outer) {
      ++accumulator.outer_point_count;
    } else {
      ++accumulator.internal_point_count;
    }
  }
  return accumulators;
}

TrialObservationKey MakeTrialObservationKey(
    const JointResidualPointDiagnostics& point) {
  TrialObservationKey key;
  key.frame_index = point.frame_index;
  key.board_id = point.board_id;
  key.point_id = point.point_id;
  key.source_point_index = point.source_point_index;
  key.source_kind = point.source_kind;
  return key;
}

DenseGridOutlierFilterResult FilterDenseGridOutliers(
    const CalibrationStateBundle& candidate_pool_bundle,
    const AslamBackendCalibrationResult& trial_backend_result,
    const std::set<FrameBoardKey>& /*protected_seed_keys*/,
    const TrialBackendFrameBoardSelectionOptions& options) {
  DenseGridOutlierFilterResult result;
  result.bundle = candidate_pool_bundle;
  result.enabled = options.single_board_dense_grid_profile &&
                   options.checkerboard_outlier_filter_enabled;
  if (!result.enabled) {
    return result;
  }

  std::vector<double> residual_norms;
  std::map<FrameBoardKey, int> total_count_by_view;
  std::map<FrameBoardKey, int> inlier_count_by_view;
  for (const JointResidualPointDiagnostics& point :
       trial_backend_result.optimized_residual.point_diagnostics) {
    if (!point.used_in_solver || !std::isfinite(point.residual_norm)) {
      continue;
    }
    residual_norms.push_back(point.residual_norm);
  }
  if (residual_norms.empty()) {
    result.bundle.warnings.push_back(
        "Dense-grid outlier filtering skipped: no finite trial residuals.");
    return result;
  }

  result.median_pixels = ComputeMedian(residual_norms);
  const double mad = ComputeMedianAbsoluteDeviation(
      residual_norms, result.median_pixels);
  result.robust_sigma_pixels = 1.4826 * mad;
  result.threshold_pixels = std::max(
      std::max(1e-6, options.checkerboard_huber_delta_pixels),
      result.median_pixels +
          std::max(0.0, options.checkerboard_outlier_sigma) *
              result.robust_sigma_pixels);

  std::set<TrialObservationKey> outlier_points;
  for (const JointResidualPointDiagnostics& point :
       trial_backend_result.optimized_residual.point_diagnostics) {
    if (!point.used_in_solver || !std::isfinite(point.residual_norm)) {
      continue;
    }
    const FrameBoardKey view_key(point.frame_index, point.board_id);
    ++total_count_by_view[view_key];
    if (point.residual_norm <= result.threshold_pixels) {
      ++inlier_count_by_view[view_key];
    } else {
      outlier_points.insert(MakeTrialObservationKey(point));
    }
  }

  std::set<FrameBoardKey> dropped_views;
  for (const auto& entry : total_count_by_view) {
    const int total_count = entry.second;
    const int inlier_count = inlier_count_by_view[entry.first];
    const int minimum_count = std::max(
        std::max(4, options.checkerboard_min_retained_points),
        static_cast<int>(std::ceil(
            std::max(0.0, std::min(1.0,
                                   options.checkerboard_min_inlier_ratio)) *
            static_cast<double>(total_count))));
    if (inlier_count < minimum_count) {
      dropped_views.insert(entry.first);
    }
  }

  for (JointMeasurementFrameResult& frame :
       result.bundle.measurement_dataset.frames) {
    for (JointBoardObservation& board : frame.board_observations) {
      const FrameBoardKey view_key(frame.frame_index, board.board_id);
      const bool drop_view = dropped_views.count(view_key) > 0;
      for (JointPointObservation& point : board.points) {
        if (!point.used_in_solver) {
          continue;
        }
        if (!drop_view &&
            outlier_points.count(MakeTrialObservationKey(point)) == 0) {
          continue;
        }
        point.used_in_solver = false;
        point.rejection_detail = drop_view
            ? "checkerboard_robust_filter_dropped_view"
            : "checkerboard_robust_filter_point_outlier";
        ++result.removed_point_count;
      }
    }
  }
  result.dropped_view_count = static_cast<int>(dropped_views.size());
  std::ostringstream warning;
  warning << "Dense-grid robust filtering: threshold_px="
          << result.threshold_pixels
          << " median_px=" << result.median_pixels
          << " robust_sigma_px=" << result.robust_sigma_pixels
          << " removed_points=" << result.removed_point_count
          << " dropped_views=" << result.dropped_view_count;
  result.bundle.warnings.push_back(warning.str());
  ReevaluateCalibrationStateBundle(&result.bundle);
  return result;
}

bool SelectionUsesPixelTrialResidualGate(
    const AslamBackendCalibrationOptions& selection_runner_options) {
  return selection_runner_options.residual_model == ResidualModel::ImagePlane &&
         !selection_runner_options.use_point_type_residual_split &&
         !selection_runner_options.angular_auxiliary_enabled;
}

TrialBackendFrameBoardSelectionResult ApplyTrialBackendFrameBoardSelection(
    const CalibrationStateBundle& baseline_bundle,
    const CalibrationStateBundle& candidate_pool_bundle,
    const BackendProblemOptions& backend_options,
    const TrialBackendFrameBoardSelectionOptions& options,
    const AslamBackendCalibrationOptions& selection_runner_options) {
  TrialBackendFrameBoardSelectionResult result;
  result.enabled = options.enabled;
  result.selection_mode = options.selection_mode;
  result.selection_profile = options.single_board_dense_grid_profile
                                 ? "single_board_dense_grid"
                                 : "multi_board";
  result.selection_is_kalibr_checkerboard_style =
      options.single_board_dense_grid_profile;
  result.checkerboard_huber_delta_pixels =
      options.single_board_dense_grid_profile
          ? options.checkerboard_huber_delta_pixels
          : 0.0;
  result.checkerboard_force_all_valid_views =
      options.single_board_dense_grid_profile &&
      options.acceptance_information_gain_threshold < 0.0;
  result.curated_bundle = baseline_bundle;
  result.input_frame_count =
      candidate_pool_bundle.measurement_dataset.accepted_frame_count;
  result.input_board_observation_count =
      candidate_pool_bundle.measurement_dataset.accepted_board_observation_count;
  result.input_total_point_count =
      candidate_pool_bundle.measurement_dataset.accepted_total_point_count;
  result.baseline_seed_frame_count =
      baseline_bundle.measurement_dataset.accepted_frame_count;
  result.baseline_seed_board_observation_count =
      baseline_bundle.measurement_dataset.accepted_board_observation_count;
  result.baseline_seed_outer_point_count =
      baseline_bundle.measurement_dataset.accepted_outer_point_count;
  result.baseline_seed_internal_point_count =
      baseline_bundle.measurement_dataset.accepted_internal_point_count;
  const bool use_pixel_trial_residual_gate =
      SelectionUsesPixelTrialResidualGate(selection_runner_options) &&
      !options.single_board_dense_grid_profile;
  if (!use_pixel_trial_residual_gate) {
    result.warnings.push_back(
        options.single_board_dense_grid_profile
            ? "Single-board dense-grid profile: trial RMSE is diagnostic only; persistent accept/reject follows Kalibr's information/rank and optimizer-validity boundary."
            : "Residual-aware Stage5 selection: disabled pixel trial-RMSE pre-rejection/strong ordering for non-pixel BA residual mode; persistent accept/reject uses the active residual metric.");
  }
  result.baseline_seed_total_point_count =
      baseline_bundle.measurement_dataset.accepted_total_point_count;

  if (!options.enabled) {
    result.success = true;
    result.failure_reason = "disabled";
    result.kept_frame_count = result.baseline_seed_frame_count;
    result.kept_board_observation_count =
        result.baseline_seed_board_observation_count;
    result.kept_outer_point_count = result.baseline_seed_outer_point_count;
    result.kept_internal_point_count =
        result.baseline_seed_internal_point_count;
    result.kept_total_point_count = result.baseline_seed_total_point_count;
    return result;
  }

  if (!baseline_bundle.IsReadyForBackend()) {
    result.failure_reason = "baseline seed bundle is not ready for backend";
    return result;
  }
  if (!candidate_pool_bundle.IsReadyForBackend()) {
    result.failure_reason = "candidate pool bundle is not ready for backend";
    return result;
  }

  result.trial_backend_result = RunShortTrialBackend(
      candidate_pool_bundle,
      backend_options,
      options,
      "_trial_backend_frame_board_pool");
  result.trial_optimization_diagnostics.push_back(
      SummarizeTrialBackendOptimization(
          "frame_board_pool", result.trial_backend_result));
  if (!result.trial_backend_result.success) {
    result.failure_reason =
        "trial backend failed: " +
        result.trial_backend_result.failure_reason;
    result.warnings = result.trial_backend_result.warnings;
    return result;
  }

  const std::map<FrameBoardKey, TrialBoardRmseAccumulator> accumulators =
      ComputeTrialBoardResiduals(result.trial_backend_result);

  std::vector<double> board_rmses;
  board_rmses.reserve(accumulators.size());
  for (const auto& entry : accumulators) {
    const TrialBoardRmseAccumulator& accumulator = entry.second;
    if (accumulator.point_count <= 0) {
      continue;
    }
    board_rmses.push_back(std::sqrt(
        accumulator.squared_error_sum /
        static_cast<double>(accumulator.point_count)));
  }
  if (board_rmses.empty()) {
    result.failure_reason = "trial backend produced no board residuals";
    return result;
  }

  result.median_board_rmse = ComputeMedian(board_rmses);
  const double mad =
      ComputeMedianAbsoluteDeviation(board_rmses, result.median_board_rmse);
  result.robust_sigma_board_rmse = 1.4826 * mad;
  result.threshold_px =
      result.median_board_rmse +
      options.outlier_sigma *
          std::max(result.robust_sigma_board_rmse,
                   options.min_abs_threshold_px);
  if (options.max_threshold_px > 0.0) {
    result.threshold_px = std::min(result.threshold_px,
                                   options.max_threshold_px);
  }
  const double checkerboard_view_rmse_threshold =
      options.single_board_dense_grid_profile
          ? std::max(0.25,
                     result.median_board_rmse +
                         options.outlier_sigma *
                             std::max(1e-6,
                                      result.robust_sigma_board_rmse))
          : result.threshold_px;

  const std::set<FrameBoardKey> baseline_keys =
      CollectAcceptedFrameBoardKeys(baseline_bundle.measurement_dataset);
  const std::set<FrameBoardKey> candidate_pool_keys =
      CollectAcceptedFrameBoardKeys(candidate_pool_bundle.measurement_dataset);
  const DenseGridOutlierFilterResult dense_grid_filter =
      FilterDenseGridOutliers(candidate_pool_bundle,
                              result.trial_backend_result,
                              baseline_keys,
                              options);
  result.checkerboard_outlier_filter_enabled = dense_grid_filter.enabled;
  result.checkerboard_outlier_sigma = options.checkerboard_outlier_sigma;
  result.checkerboard_min_inlier_ratio =
      options.checkerboard_min_inlier_ratio;
  result.checkerboard_min_retained_points =
      options.checkerboard_min_retained_points;
  result.checkerboard_outlier_threshold_pixels =
      dense_grid_filter.threshold_pixels;
  result.checkerboard_outlier_median_pixels = dense_grid_filter.median_pixels;
  result.checkerboard_outlier_robust_sigma_pixels =
      dense_grid_filter.robust_sigma_pixels;
  result.checkerboard_outlier_removed_point_count =
      dense_grid_filter.removed_point_count;
  result.checkerboard_outlier_dropped_view_count =
      dense_grid_filter.dropped_view_count;
  const CalibrationStateBundle& persistent_candidate_pool_bundle =
      dense_grid_filter.enabled ? dense_grid_filter.bundle
                                : candidate_pool_bundle;
  const std::set<FrameBoardKey> persistent_candidate_pool_keys =
      CollectAcceptedFrameBoardKeys(
          persistent_candidate_pool_bundle.measurement_dataset);
  CalibrationStateBundle persistent_baseline_bundle = baseline_bundle;
  if (dense_grid_filter.enabled) {
    std::set<FrameBoardKey> filtered_seed_keys;
    std::set_intersection(
        baseline_keys.begin(), baseline_keys.end(),
        persistent_candidate_pool_keys.begin(),
        persistent_candidate_pool_keys.end(),
        std::inserter(filtered_seed_keys, filtered_seed_keys.end()));
    persistent_baseline_bundle = BuildBundleForAcceptedFrameBoardKeys(
        persistent_candidate_pool_bundle,
        persistent_candidate_pool_bundle,
        filtered_seed_keys,
        baseline_bundle.measurement_dataset.source_stage_label +
            "_dense_grid_filtered_seed");
    if (!persistent_baseline_bundle.IsReadyForBackend()) {
      result.failure_reason =
          "checkerboard outlier filtering removed every bootstrap view";
      return result;
    }
  }
  if (dense_grid_filter.enabled) {
    std::ostringstream warning;
    warning << "Checkerboard robust filtering retained "
            << persistent_candidate_pool_bundle.measurement_dataset
                   .accepted_total_point_count
            << " / "
            << candidate_pool_bundle.measurement_dataset
                   .accepted_total_point_count
            << " points and "
            << persistent_candidate_pool_bundle.measurement_dataset
                   .accepted_frame_count
            << " / "
            << candidate_pool_bundle.measurement_dataset.accepted_frame_count
            << " views.";
    result.warnings.push_back(warning.str());
  }
  result.candidate_board_observation_count =
      static_cast<int>(candidate_pool_keys.size() > baseline_keys.size()
                           ? candidate_pool_keys.size() - baseline_keys.size()
                           : 0);

  std::map<int, int> kept_count_by_board;
  for (const auto& entry : accumulators) {
    const int board_id = entry.first.second;
    const TrialBoardRmseAccumulator& accumulator = entry.second;
    const double rmse =
        accumulator.point_count > 0
            ? std::sqrt(accumulator.squared_error_sum /
                        static_cast<double>(accumulator.point_count))
            : 0.0;
    if (rmse <= result.threshold_px) {
      ++kept_count_by_board[board_id];
    }
  }

  std::map<FrameBoardKey, TrialBackendFrameBoardObservationDecision>
      decision_by_key;
  for (const FrameBoardKey& key : baseline_keys) {
    TrialBackendFrameBoardObservationDecision decision;
    decision.frame_index = key.first;
    decision.board_id = key.second;
    decision.baseline_seed = true;
    decision.kept = true;
    decision.reason = "baseline_seed";
    const auto accum_it = accumulators.find(key);
    if (accum_it != accumulators.end()) {
      decision.frame_label = accum_it->second.frame_label;
      decision.point_count = accum_it->second.point_count;
      decision.outer_point_count = accum_it->second.outer_point_count;
      decision.internal_point_count = accum_it->second.internal_point_count;
      decision.trial_rmse =
          accum_it->second.point_count > 0
              ? std::sqrt(accum_it->second.squared_error_sum /
                          static_cast<double>(accum_it->second.point_count))
              : 0.0;
    }
    decision_by_key[key] = decision;
  }

  std::set<FrameBoardKey> rejected_keys;
  std::vector<TrialBackendFrameBoardObservationDecision> candidate_decisions;
  for (const auto& entry : accumulators) {
    if (baseline_keys.find(entry.first) != baseline_keys.end()) {
      continue;
    }
    TrialBackendFrameBoardObservationDecision decision;
    decision.frame_index = entry.first.first;
    decision.board_id = entry.first.second;
    decision.frame_label = entry.second.frame_label;
    decision.point_count = entry.second.point_count;
    decision.outer_point_count = entry.second.outer_point_count;
    decision.internal_point_count = entry.second.internal_point_count;
    decision.trial_rmse =
        entry.second.point_count > 0
            ? std::sqrt(entry.second.squared_error_sum /
                        static_cast<double>(entry.second.point_count))
            : 0.0;
    decision.kept = true;
    decision.reason = "kept";
    if (use_pixel_trial_residual_gate &&
        decision.trial_rmse > result.threshold_px) {
      const int board_kept_count = kept_count_by_board[decision.board_id];
      if (board_kept_count >= options.min_keep_observations_per_board) {
        decision.kept = false;
        decision.reason = "trial_backend_residual_outlier";
        rejected_keys.insert(entry.first);
      } else {
        decision.reason =
            "kept_to_preserve_min_observations_per_board";
      }
    }
    candidate_decisions.push_back(decision);
    decision_by_key[entry.first] = decision;
  }

  if (!options.incremental_acceptance) {
    if (rejected_keys.empty()) {
      result.curated_bundle = candidate_pool_bundle;
      result.success = true;
      result.kept_frame_count = result.input_frame_count;
      result.kept_board_observation_count = result.input_board_observation_count;
      result.kept_outer_point_count =
          result.curated_bundle.measurement_dataset.accepted_outer_point_count;
      result.kept_internal_point_count =
          result.curated_bundle.measurement_dataset
              .accepted_internal_point_count;
      result.kept_total_point_count = result.input_total_point_count;
      result.rejected_board_observation_count = 0;
      result.decisions.reserve(decision_by_key.size());
      for (const auto& entry : decision_by_key) {
        result.decisions.push_back(entry.second);
      }
      return result;
    }

    std::set<FrameBoardKey> kept_keys = candidate_pool_keys;
    for (const FrameBoardKey& rejected_key : rejected_keys) {
      kept_keys.erase(rejected_key);
    }
    CalibrationStateBundle curated = BuildBundleForAcceptedFrameBoardKeys(
        candidate_pool_bundle,
        candidate_pool_bundle,
        kept_keys,
        candidate_pool_bundle.measurement_dataset.source_stage_label +
            "_trial_backend_frame_board_selected");
    curated.warnings.push_back(
        "Applied one-shot trial-backend frame-board observation selection.");
    result.curated_bundle = curated;
    result.success = true;
    result.rejected_board_observation_count =
        static_cast<int>(rejected_keys.size());
    result.kept_frame_count =
        curated.measurement_dataset.accepted_frame_count;
    result.kept_board_observation_count =
        curated.measurement_dataset.accepted_board_observation_count;
    result.kept_outer_point_count =
        curated.measurement_dataset.accepted_outer_point_count;
    result.kept_internal_point_count =
        curated.measurement_dataset.accepted_internal_point_count;
    result.kept_total_point_count =
        curated.measurement_dataset.accepted_total_point_count;
    result.decisions.reserve(decision_by_key.size());
    for (const auto& entry : decision_by_key) {
      result.decisions.push_back(entry.second);
    }
    return result;
  }

  for (TrialBackendFrameBoardObservationDecision& candidate :
       candidate_decisions) {
    const FrameBoardKey key(candidate.frame_index, candidate.board_id);
    const CandidateCoverageScoreBreakdown score =
        ComputeCandidateCoverageScore(
            key,
            candidate_pool_bundle.measurement_dataset,
            baseline_keys,
            candidate.trial_rmse,
            result.threshold_px,
            use_pixel_trial_residual_gate);
    const CandidateConsistencyScoreBreakdown consistency =
        ComputeCandidateConsistencyScore(
            key,
            candidate_pool_bundle,
            options);
    candidate.coverage_gain = score.coverage_gain;
    candidate.candidate_score = score.total_score - consistency.penalty;
    candidate.polar_gain = score.polar_gain;
    candidate.edge_gain = score.edge_gain;
    candidate.board_balance_gain = score.board_balance_gain;
    candidate.frame_novelty_gain = score.frame_novelty_gain;
    candidate.grid_gain = score.grid_gain;
    candidate.covisibility_gain = score.covisibility_gain;
    candidate.residual_quality_score = score.residual_quality_score;
    candidate.consistency_available = consistency.available;
    candidate.consistency_score = consistency.score;
    candidate.consistency_penalty = consistency.penalty;
    candidate.consistency_translation_error_mm =
        consistency.translation_error_mm;
    candidate.consistency_rotation_error_deg =
        consistency.rotation_error_deg;
    candidate.consistency_local_outer_rmse =
        consistency.local_outer_rmse;
    candidate.force_include_candidate =
        IsForceIncludeFrameBoardCandidate(key, candidate.frame_label, options);
    candidate.mean_polar_angle_deg = score.mean_polar_angle_deg;
    candidate.max_polar_angle_deg = score.max_polar_angle_deg;
    candidate.soft_weight = 1.0;
    const BoardObservationGeometrySummary geometry =
        SummarizeBoardObservationGeometry(
            key,
            candidate_pool_bundle.measurement_dataset);
    candidate.projected_area_px = geometry.projected_area_px;
    candidate.projected_area_ratio = geometry.projected_area_ratio;
    candidate.intrinsics_diversity_anchor =
        IsIntrinsicsDiversityAnchorGeometry(geometry);
    if (candidate.intrinsics_diversity_anchor) {
      ++result.intrinsics_diversity_anchor_candidate_count;
    }
    candidate.outer_pose_refit_rmse = 0.0;
    if (candidate.coverage_gain == 0.0) {
      candidate.coverage_gain = ComputeFrameBoardCoverageGain(
        key,
        candidate_pool_bundle.measurement_dataset,
        baseline_keys);
      candidate.candidate_score =
          candidate.coverage_gain +
          (use_pixel_trial_residual_gate
               ? 0.8 * candidate.residual_quality_score
               : 0.0) -
          candidate.consistency_penalty;
    }
  }
  std::sort(candidate_decisions.begin(), candidate_decisions.end(),
            ScoreSortedBefore);

  std::set<FrameBoardKey> accepted_keys = baseline_keys;
  CalibrationStateBundle current_scene_template_bundle = candidate_pool_bundle;
  const std::map<int, int> candidate_pool_board_capacity_by_frame =
      CountFrameBoardObservationCapacity(candidate_pool_keys);
  AslamBackendCalibrationResult current_backend = RunShortTrialBackend(
      baseline_bundle,
      backend_options,
      options,
      "_trial_backend_incremental_seed");
  result.trial_optimization_diagnostics.push_back(
      SummarizeTrialBackendOptimization(
          "incremental_seed", current_backend));
  if (!current_backend.success) {
    result.failure_reason =
        "incremental seed backend failed: " +
        current_backend.failure_reason;
    result.warnings = current_backend.warnings;
    return result;
  }
  TrialBackendMetrics current_metrics =
      ExtractTrialBackendMetrics(current_backend);
  const TrialBackendTraversalBudget traversal_budget =
      ComputeTrialBackendTraversalBudget(
          options, static_cast<int>(candidate_decisions.size()));
  result.budget_mode = traversal_budget.mode;
  result.valid_candidate_count =
      static_cast<int>(candidate_decisions.size());
  result.runtime_safety_ceiling =
      traversal_budget.runtime_safety_ceiling;
  result.max_candidate_additions_effective =
      traversal_budget.max_candidate_additions_effective;
  result.info_gain_proxy_mode = options.info_gain_proxy_mode;
  result.candidate_batch_granularity =
      options.candidate_batch_granularity;
  result.acceptance_policy = options.acceptance_policy;
  result.acceptance_information_gain_threshold =
      options.acceptance_information_gain_threshold;
  result.acceptance_rank_gain_threshold =
      options.acceptance_rank_gain_threshold;
  result.carry_accepted_trial_state = options.carry_accepted_trial_state;
  result.optimize_intrinsics_in_trial = options.optimize_intrinsics_in_trial;
  result.delayed_intrinsics_release_in_trial =
      options.delayed_intrinsics_release_in_trial;
  result.intrinsics_release_iteration = options.intrinsics_release_iteration;
  result.persistent_intrinsics_anchor_prior_enabled =
      options.persistent_intrinsics_anchor_prior_enabled;
  result.persistent_intrinsics_anchor_weight_xi_alpha =
      options.persistent_intrinsics_anchor_weight_xi_alpha;
  result.persistent_intrinsics_anchor_weight_focal =
      options.persistent_intrinsics_anchor_weight_focal;
  result.persistent_intrinsics_anchor_weight_principal =
      options.persistent_intrinsics_anchor_weight_principal;
  result.persistent_max_focal_relative_step =
      options.persistent_max_focal_relative_step;
  result.persistent_max_principal_step_px =
      options.persistent_max_principal_step_px;
  result.persistent_max_xi_alpha_step =
      options.persistent_max_xi_alpha_step;
  std::map<int, int> accepted_candidate_count_by_board;
  std::map<int, int> accepted_candidate_count_by_frame;
  std::map<int, int> accepted_frame_cohesion_count_by_frame;
  std::map<int, int> accepted_observation_count_by_frame;
  for (const FrameBoardKey& key : baseline_keys) {
    ++accepted_observation_count_by_frame[key.first];
  }
  const std::map<int, int> baseline_observation_count_by_frame =
      accepted_observation_count_by_frame;
  std::map<FrameBoardKey, IntrinsicsJacobianInformationSummary>
      intrinsics_jacobian_info_by_key;
  Eigen::Matrix<double, 6, 6> accepted_intrinsics_fisher =
      Eigen::Matrix<double, 6, 6>::Zero();
  for (const FrameBoardKey& key : candidate_pool_keys) {
    const IntrinsicsJacobianInformationSummary info =
        ComputeIntrinsicsJacobianInformation(candidate_pool_bundle, key);
    intrinsics_jacobian_info_by_key[key] = info;
    if (baseline_keys.find(key) != baseline_keys.end() && info.available) {
      accepted_intrinsics_fisher += info.fisher;
    }
  }
  const bool kalibr_style_batch_mode =
      options.selection_mode ==
      TrialBackendFrameBoardSelectionOptions::SelectionMode::KalibrStyleBatch;
  const bool frame_batch_mode =
      kalibr_style_batch_mode &&
      options.candidate_batch_granularity ==
          TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity
              ::Frame;
  const bool frame_consolidation_mode =
      kalibr_style_batch_mode &&
      options.candidate_batch_granularity ==
          TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity
              ::FrameBoardThenFrame;
  result.candidate_order_mode =
      kalibr_style_batch_mode
          ? options.candidate_order_mode
          : TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
                ::ScoreSorted;
  result.candidate_shuffle_seed_set =
      result.candidate_order_mode ==
      TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
          ::RandomShuffle;
  if (result.candidate_order_mode ==
      TrialBackendFrameBoardSelectionOptions::CandidateOrderMode::
          IntrinsicsInformationGreedy) {
    std::vector<TrialBackendFrameBoardObservationDecision> remaining =
        candidate_decisions;
    std::vector<TrialBackendFrameBoardObservationDecision> ordered;
    ordered.reserve(remaining.size());
    Eigen::Matrix<double, 6, 6> ordering_fisher =
        accepted_intrinsics_fisher;
    while (!remaining.empty()) {
      const double current_logdet =
          RegularizedFisherLogDet(ordering_fisher);
      std::size_t best_index = remaining.size();
      double best_gain = -std::numeric_limits<double>::infinity();
      double best_coverage = -std::numeric_limits<double>::infinity();
      for (std::size_t index = 0; index < remaining.size(); ++index) {
        const FrameBoardKey key(remaining[index].frame_index,
                                remaining[index].board_id);
        const auto info_it = intrinsics_jacobian_info_by_key.find(key);
        if (info_it == intrinsics_jacobian_info_by_key.end() ||
            !info_it->second.available) {
          continue;
        }
        const double gain = std::max(
            0.0,
            RegularizedFisherLogDet(ordering_fisher + info_it->second.fisher) -
                current_logdet);
        const double coverage = remaining[index].coverage_gain;
        if (gain > best_gain + 1e-12 ||
            (std::abs(gain - best_gain) <= 1e-12 &&
             coverage > best_coverage + 1e-12)) {
          best_index = index;
          best_gain = gain;
          best_coverage = coverage;
        }
      }
      if (best_index >= remaining.size()) {
        std::sort(remaining.begin(), remaining.end(), ScoreSortedBefore);
        ordered.insert(ordered.end(), remaining.begin(), remaining.end());
        break;
      }
      const FrameBoardKey best_key(remaining[best_index].frame_index,
                                   remaining[best_index].board_id);
      ordering_fisher += intrinsics_jacobian_info_by_key[best_key].fisher;
      ordered.push_back(remaining[best_index]);
      remaining.erase(remaining.begin() +
                      static_cast<std::ptrdiff_t>(best_index));
    }
    candidate_decisions.swap(ordered);
  } else if (result.candidate_shuffle_seed_set) {
    // Randomized controlled ablations must not inherit the residual-dependent
    // score ordering that precedes this block. Canonicalize identities first
    // so equal seeds produce an equal frame-board schedule across residual
    // models.
    std::sort(
        candidate_decisions.begin(), candidate_decisions.end(),
        [](const TrialBackendFrameBoardObservationDecision& lhs,
           const TrialBackendFrameBoardObservationDecision& rhs) {
          if (lhs.frame_index != rhs.frame_index) {
            return lhs.frame_index < rhs.frame_index;
          }
          if (lhs.board_id != rhs.board_id) {
            return lhs.board_id < rhs.board_id;
          }
          return lhs.frame_label < rhs.frame_label;
        });
    unsigned int deterministic_seed = 2166136261u;
    for (const TrialBackendFrameBoardObservationDecision& candidate :
         candidate_decisions) {
      deterministic_seed ^= static_cast<unsigned int>(
          candidate.frame_index * 73856093u);
      deterministic_seed *= 16777619u;
      deterministic_seed ^= static_cast<unsigned int>(
          candidate.board_id * 19349663u);
      deterministic_seed *= 16777619u;
    }
    result.candidate_shuffle_seed =
        options.candidate_shuffle_seed_set
            ? options.candidate_shuffle_seed
            : deterministic_seed;
    std::mt19937 rng(result.candidate_shuffle_seed);
    std::shuffle(candidate_decisions.begin(), candidate_decisions.end(), rng);
  }
  if (!options.force_include_frame_board_keys.empty() ||
      !options.force_include_frame_label_board_keys.empty()) {
    std::stable_partition(
        candidate_decisions.begin(), candidate_decisions.end(),
        [&options](const TrialBackendFrameBoardObservationDecision& decision) {
          return IsForceIncludeFrameBoardCandidate(
              FrameBoardKey(decision.frame_index, decision.board_id),
              decision.frame_label, options);
        });
  }
  const auto traversal_budget_exhausted = [&]() {
    return result.valid_candidate_traversed_count >=
           traversal_budget.traversal_limit;
  };
  const auto mark_runtime_safety_ceiling =
      [&](TrialBackendFrameBoardObservationDecision* candidate) {
        if (candidate != nullptr) {
          candidate->kept = false;
          candidate->reason = "runtime_safety_ceiling";
        }
        if (!result.safety_ceiling_hit) {
          result.safety_ceiling_hit = true;
          result.warnings.push_back(
              "runtime_safety_ceiling_hit; result is runtime-capped");
        }
      };
  const auto record_candidate_traversal = [&]() {
    ++result.valid_candidate_traversed_count;
  };
  const auto batch_acceptance_pass =
      [&](TrialBackendFrameBoardObservationDecision* candidate,
          const TrialBackendMetrics& tentative_metrics,
          const AslamBackendCalibrationResult& tentative_backend,
          bool board_cap_exceeded,
          bool frame_cap_exceeded,
          const std::vector<TrialBackendFrameBoardObservationDecision>*
              batch_members) {
        if (candidate == nullptr) {
          return false;
        }
        ++result.batch_acceptance_attempted_count;
        const bool residual_finite =
            tentative_metrics.success &&
            std::isfinite(tentative_metrics.overall_rmse) &&
            std::isfinite(tentative_metrics.outer_rmse) &&
            std::isfinite(tentative_metrics.internal_rmse) &&
            std::isfinite(candidate->global_rmse_delta) &&
            std::isfinite(candidate->outer_rmse_delta) &&
            std::isfinite(candidate->internal_rmse_delta);
        bool objective_valid = true;
        bool linear_solver_failure = false;
        for (const AslamBackendOptimizationStageSummary& stage :
             tentative_backend.stages) {
          objective_valid =
              objective_valid && std::isfinite(stage.objective_start) &&
              std::isfinite(stage.objective_final) &&
              !(stage.objective_final > 10.0 * stage.objective_start);
          linear_solver_failure =
              linear_solver_failure || stage.linear_solver_failure;
        }
        candidate->hard_validity_pass =
            tentative_backend.success && residual_finite &&
            objective_valid && !linear_solver_failure;
        if (!candidate->hard_validity_pass) {
          candidate->reason = "hard_validity_gate";
          ++result.batch_acceptance_rejected_hard_validity_count;
          return false;
        }

        candidate->legacy_rmse_pass =
            candidate->global_rmse_delta <=
                options.accept_max_global_rmse_increase_px &&
            candidate->outer_rmse_delta <=
                options.accept_max_outer_rmse_increase_px &&
            candidate->internal_rmse_delta <=
                options.accept_max_internal_rmse_increase_px;

        const double max_global_delta =
            std::max(options.accept_max_global_rmse_increase_px, 1e-12);
        const double max_outer_delta =
            std::max(options.accept_max_outer_rmse_increase_px, 1e-12);
        const double max_internal_delta =
            std::max(options.accept_max_internal_rmse_increase_px, 1e-12);
        candidate->catastrophic_residual =
            candidate->global_rmse_delta >
                5.0 * max_global_delta ||
            candidate->outer_rmse_delta >
                4.0 * max_outer_delta ||
            candidate->internal_rmse_delta >
                4.0 * max_internal_delta;
        if (candidate->catastrophic_residual) {
          candidate->reason = "batch_catastrophic_residual_gate";
          ++result.batch_acceptance_rejected_catastrophic_residual_count;
          return false;
        }

        const TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode
            effective_info_gain_mode =
                options.acceptance_policy ==
                        KalibrStyleBatchAcceptancePolicy
                            ::KalibrInformationGain
                    ? TrialBackendFrameBoardSelectionOptions
                          ::InfoGainProxyMode::IntrinsicsJacobian
                    : options.info_gain_proxy_mode;
        const IntrinsicsInformationGainProxyBreakdown gain =
            ComputeIntrinsicsInformationGainProxy(
                *candidate,
                candidate_pool_bundle.measurement_dataset,
                accepted_keys,
                accepted_observation_count_by_frame,
                candidate_pool_board_capacity_by_frame,
                accepted_intrinsics_fisher,
                [&]() -> IntrinsicsJacobianInformationSummary {
                  const FrameBoardKey candidate_key(
                      candidate->frame_index, candidate->board_id);
                  const auto info_it =
                      intrinsics_jacobian_info_by_key.find(candidate_key);
                  return info_it == intrinsics_jacobian_info_by_key.end()
                             ? IntrinsicsJacobianInformationSummary{}
                             : info_it->second;
                }(),
                options.min_candidate_score,
                effective_info_gain_mode);
        candidate->score_term = gain.score_term;
        candidate->coverage_term = gain.coverage_term;
        candidate->intrinsics_jacobian_logdet_gain =
            gain.intrinsics_jacobian_logdet_gain;
        candidate->intrinsics_jacobian_trace_gain =
            gain.intrinsics_jacobian_trace_gain;
        candidate->intrinsics_jacobian_rank_gain =
            gain.intrinsics_jacobian_rank_gain;
        candidate->intrinsics_jacobian_info_term =
            gain.intrinsics_jacobian_info_term;
        candidate->frame_completion_bonus = gain.frame_completion_bonus;
        candidate->new_board_bonus = gain.new_board_bonus;
        candidate->information_gain_proxy =
            options.acceptance_policy ==
                    KalibrStyleBatchAcceptancePolicy::KalibrInformationGain
                ? candidate->intrinsics_jacobian_logdet_gain
                : gain.information_gain_proxy;
        if (batch_members != nullptr && !batch_members->empty()) {
          Eigen::Matrix<double, 6, 6> batch_fisher =
              Eigen::Matrix<double, 6, 6>::Zero();
          double aggregate_score_term = 0.0;
          double aggregate_coverage_term = 0.0;
          double aggregate_frame_completion_bonus = 0.0;
          double aggregate_new_board_bonus = 0.0;
          bool aggregate_intrinsics_anchor = false;
          int available_jacobian_count = 0;
          for (const TrialBackendFrameBoardObservationDecision& member :
               *batch_members) {
            const FrameBoardKey member_key(member.frame_index,
                                           member.board_id);
            const auto info_it =
                intrinsics_jacobian_info_by_key.find(member_key);
            IntrinsicsJacobianInformationSummary member_info;
            if (info_it != intrinsics_jacobian_info_by_key.end()) {
              member_info = info_it->second;
              if (member_info.available) {
                batch_fisher += member_info.fisher;
                ++available_jacobian_count;
              }
            }
            const IntrinsicsInformationGainProxyBreakdown member_gain =
                ComputeIntrinsicsInformationGainProxy(
                    member,
                    candidate_pool_bundle.measurement_dataset,
                    accepted_keys,
                    accepted_observation_count_by_frame,
                    candidate_pool_board_capacity_by_frame,
                    accepted_intrinsics_fisher,
                    member_info,
                    options.min_candidate_score,
                    effective_info_gain_mode);
            aggregate_score_term += member_gain.score_term;
            aggregate_coverage_term += member_gain.coverage_term;
            aggregate_frame_completion_bonus =
                std::max(aggregate_frame_completion_bonus,
                         member_gain.frame_completion_bonus);
            aggregate_new_board_bonus += member_gain.new_board_bonus;
            aggregate_intrinsics_anchor =
                aggregate_intrinsics_anchor ||
                member.intrinsics_diversity_anchor;
          }
          const double batch_member_count =
              static_cast<double>(std::max<std::size_t>(
                  1u, batch_members->size()));
          const Eigen::Matrix<double, 6, 6> combined_fisher =
              accepted_intrinsics_fisher + batch_fisher;
          const double accepted_logdet =
              RegularizedFisherLogDet(accepted_intrinsics_fisher);
          const double combined_logdet =
              RegularizedFisherLogDet(combined_fisher);
          const double accepted_rank =
              FisherRankProxy(accepted_intrinsics_fisher);
          const double combined_rank = FisherRankProxy(combined_fisher);
          candidate->score_term =
              aggregate_score_term / batch_member_count;
          candidate->coverage_term =
              aggregate_coverage_term / batch_member_count;
          candidate->intrinsics_jacobian_logdet_gain =
              available_jacobian_count > 0
                  ? std::max(0.0, combined_logdet - accepted_logdet)
                  : 0.0;
          candidate->intrinsics_jacobian_trace_gain =
              available_jacobian_count > 0
                  ? std::max(0.0, combined_fisher.trace() -
                                      accepted_intrinsics_fisher.trace())
                  : 0.0;
          candidate->intrinsics_jacobian_rank_gain =
              available_jacobian_count > 0
                  ? std::max(0.0, combined_rank - accepted_rank)
                  : 0.0;
          const double logdet_term =
              std::min(2.0,
                       candidate->intrinsics_jacobian_logdet_gain / 1.5);
          const double trace_term =
              std::min(1.0,
                       std::log1p(candidate->intrinsics_jacobian_trace_gain) /
                           6.0);
          const double rank_term =
              std::min(1.0,
                       candidate->intrinsics_jacobian_rank_gain / 0.5);
          candidate->intrinsics_jacobian_info_term =
              logdet_term + 0.35 * trace_term + 0.35 * rank_term;
          candidate->frame_completion_bonus =
              aggregate_frame_completion_bonus;
          candidate->new_board_bonus =
              aggregate_new_board_bonus / batch_member_count;
          candidate->information_gain_proxy =
              options.acceptance_policy ==
                      KalibrStyleBatchAcceptancePolicy::KalibrInformationGain
                  ? candidate->intrinsics_jacobian_logdet_gain
                  : candidate->coverage_term +
                        candidate->intrinsics_jacobian_info_term +
                        candidate->frame_completion_bonus +
                        candidate->new_board_bonus;
          candidate->intrinsics_diversity_anchor =
              candidate->intrinsics_diversity_anchor ||
              aggregate_intrinsics_anchor;
        }
        candidate->cap_penalty =
            (board_cap_exceeded ? 0.5 : 0.0) +
            (frame_cap_exceeded ? 0.5 : 0.0);
        const double global_overage =
            std::max(0.0,
                     candidate->global_rmse_delta / max_global_delta - 1.0);
        const double outer_overage =
            std::max(0.0,
                     candidate->outer_rmse_delta / max_outer_delta - 1.0);
        const double internal_overage =
            std::max(0.0,
                     candidate->internal_rmse_delta / max_internal_delta -
                         1.0);
        candidate->residual_overage_penalty =
            global_overage + 0.5 * outer_overage + 0.5 * internal_overage;
        candidate->batch_acceptance_score =
            candidate->information_gain_proxy -
            candidate->residual_overage_penalty - candidate->cap_penalty;
        KalibrStyleBatchAcceptanceOptions acceptance_options;
        acceptance_options.policy = options.acceptance_policy;
        acceptance_options.information_gain_threshold =
            options.acceptance_information_gain_threshold;
        acceptance_options.rank_gain_threshold =
            options.acceptance_rank_gain_threshold;
        KalibrStyleBatchAcceptanceInput acceptance_input;
        acceptance_input.hard_validity_pass = candidate->hard_validity_pass;
        acceptance_input.catastrophic_residual =
            candidate->catastrophic_residual;
        acceptance_input.companion_completion =
            candidate->frame_cohesion_candidate &&
            candidate->frame_completion_bonus > 0.0;
        acceptance_input.critical_view =
            candidate->intrinsics_diversity_anchor;
        acceptance_input.information_gain_proxy =
            options.acceptance_policy ==
                    KalibrStyleBatchAcceptancePolicy
                        ::KalibrInformationGain
                ? candidate->intrinsics_jacobian_logdet_gain
                : candidate->information_gain_proxy;
        acceptance_input.rank_gain_proxy =
            options.acceptance_policy ==
                    KalibrStyleBatchAcceptancePolicy::KalibrInformationGain
                ? 0.0
                : candidate->intrinsics_jacobian_rank_gain;
        acceptance_input.residual_score = candidate->batch_acceptance_score;
        acceptance_input.residual_overage_penalty =
            candidate->residual_overage_penalty;
        const KalibrStyleBatchAcceptanceDecision acceptance_decision =
            EvaluateKalibrStyleBatchAcceptance(acceptance_options,
                                               acceptance_input);
        if (acceptance_decision.accepted) {
          candidate->accepted_by_batch_acceptance = true;
          candidate->reason = acceptance_decision.reason;
          ++result.batch_acceptance_accepted_count;
          if (!candidate->legacy_rmse_pass) {
            ++result.batch_acceptance_rescued_from_legacy_rmse_gate_count;
          }
          return true;
        }
        candidate->reason = acceptance_decision.reason;
        ++result.batch_acceptance_rejected_score_count;
        return false;
      };

  if (frame_batch_mode) {
    std::map<int, std::vector<std::size_t> > candidate_indices_by_frame;
    std::vector<int> frame_order;
    for (std::size_t index = 0; index < candidate_decisions.size(); ++index) {
      const int frame_index = candidate_decisions[index].frame_index;
      if (candidate_indices_by_frame[frame_index].empty()) {
        frame_order.push_back(frame_index);
      }
      candidate_indices_by_frame[frame_index].push_back(index);
    }
    result.frame_batch_candidate_count =
        static_cast<int>(candidate_indices_by_frame.size());

    BackendProblemOptions persistent_backend_options = backend_options;
    persistent_backend_options.optimize_frame_poses = true;
    persistent_backend_options.optimize_board_poses = true;
    persistent_backend_options.optimize_intrinsics =
        options.optimize_intrinsics_in_trial;
    persistent_backend_options.delayed_intrinsics_release =
        options.optimize_intrinsics_in_trial &&
        options.delayed_intrinsics_release_in_trial;
    persistent_backend_options.intrinsics_release_iteration =
        std::max(0, options.intrinsics_release_iteration);
    result.persistent_intrinsics_anchor_prior_enabled =
        options.persistent_intrinsics_anchor_prior_enabled;
    result.persistent_intrinsics_anchor_weight_xi_alpha =
        options.persistent_intrinsics_anchor_weight_xi_alpha;
    result.persistent_intrinsics_anchor_weight_focal =
        options.persistent_intrinsics_anchor_weight_focal;
    result.persistent_intrinsics_anchor_weight_principal =
        options.persistent_intrinsics_anchor_weight_principal;
    result.persistent_max_focal_relative_step =
        options.persistent_max_focal_relative_step;
    result.persistent_max_principal_step_px =
        options.persistent_max_principal_step_px;
    result.persistent_max_xi_alpha_step =
        options.persistent_max_xi_alpha_step;
    const AslamBackendCalibrationOptions persistent_runner_options =
        MakeTrialBackendRunnerOptions(options, selection_runner_options);

    std::vector<Stage5IncrementalBackendBatchInput> persistent_batches;
    std::map<int, Stage5IncrementalBackendBatchInput> persistent_batch_by_frame;
    bool persistent_safety_ceiling_hit = false;
    int persistent_candidate_observation_attempt_count = 0;
    const bool persistent_uses_fixed_traversal_limit =
        options.budget_mode ==
        TrialBackendFrameBoardSelectionOptions::BudgetMode::Fixed;
    const int persistent_traversal_limit =
        persistent_uses_fixed_traversal_limit
            ? traversal_budget.traversal_limit
            : (traversal_budget.runtime_safety_ceiling > 0
                   ? traversal_budget.runtime_safety_ceiling
                   : std::numeric_limits<int>::max());
    if (options.budget_mode ==
        TrialBackendFrameBoardSelectionOptions::BudgetMode::Adaptive) {
      result.warnings.push_back(
          "Persistent incremental estimator ignores adaptive attempted-batch "
          "truncation and traverses candidate frame batches until information "
          "acceptance saturates or runtime_safety_ceiling is reached.");
    }
    for (int frame_index : frame_order) {
      if (static_cast<int>(persistent_batches.size()) >=
          persistent_traversal_limit) {
        persistent_safety_ceiling_hit = true;
        break;
      }
      const std::vector<std::size_t>& indices =
          candidate_indices_by_frame[frame_index];
      Stage5IncrementalBackendBatchInput batch_input;
      batch_input.frame_index = frame_index;
      batch_input.residual_health_threshold_px =
          checkerboard_view_rmse_threshold;
      double ordering_score_sum = 0.0;
      double coverage_gain_sum = 0.0;
      int ordering_score_count = 0;
      for (std::size_t index : indices) {
        const TrialBackendFrameBoardObservationDecision& member =
            candidate_decisions[index];
        const FrameBoardKey key(member.frame_index, member.board_id);
        if (accepted_keys.find(key) != accepted_keys.end()) {
          continue;
        }
        if (persistent_candidate_pool_keys.find(key) ==
            persistent_candidate_pool_keys.end()) {
          continue;
        }
        if (batch_input.frame_label.empty()) {
          batch_input.frame_label = member.frame_label;
        }
        batch_input.max_trial_rmse =
            std::max(batch_input.max_trial_rmse, member.trial_rmse);
        batch_input.frame_board_keys.insert(key);
        batch_input.force =
            batch_input.force || member.force_include_candidate;
        batch_input.has_intrinsics_diversity_anchor =
            batch_input.has_intrinsics_diversity_anchor ||
            member.intrinsics_diversity_anchor;
        ordering_score_sum +=
            std::isfinite(member.candidate_score) ? member.candidate_score
                                                  : 0.0;
        coverage_gain_sum +=
            std::isfinite(member.coverage_gain) ? member.coverage_gain : 0.0;
        ++ordering_score_count;
      }
      if (batch_input.frame_board_keys.empty()) {
        continue;
      }
      const double ordering_denominator =
          static_cast<double>(std::max(1, ordering_score_count));
      batch_input.ordering_score = ordering_score_sum / ordering_denominator;
      batch_input.coverage_gain = coverage_gain_sum / ordering_denominator;
      persistent_candidate_observation_attempt_count +=
          static_cast<int>(batch_input.frame_board_keys.size());
      persistent_batch_by_frame[frame_index] = batch_input;
      persistent_batches.push_back(batch_input);
    }

    const Stage5IncrementalBackendEstimatorResult persistent_result =
        RunStage5IncrementalBackendEstimator(
            persistent_baseline_bundle, persistent_candidate_pool_bundle,
            persistent_backend_options,
            options, persistent_runner_options, persistent_batches);
    CopyPersistentIncrementalSummary(persistent_result, &result);

    const bool persistent_required_for_residual_ablation =
        persistent_runner_options.residual_model != ResidualModel::ImagePlane ||
        persistent_runner_options.use_point_type_residual_split ||
        persistent_runner_options.angular_auxiliary_enabled;
    if (!persistent_result.success && persistent_required_for_residual_ablation) {
      result.failure_reason =
          "Requested backend residual model must be applied inside persistent "
          "incremental selection BA, but the persistent estimator is not "
          "compatible: " + persistent_result.fallback_reason;
      result.warnings.insert(result.warnings.end(),
                             persistent_result.warnings.begin(),
                             persistent_result.warnings.end());
      return result;
    }

    if (persistent_result.success) {
      result.persistent_incremental_backend_estimator_used = true;
      result.valid_candidate_traversed_count =
          persistent_result.attempted_batch_count;
      result.safety_ceiling_hit = persistent_safety_ceiling_hit;
      if (result.safety_ceiling_hit) {
        result.warnings.push_back(
            "runtime_safety_ceiling_hit; persistent incremental result is runtime-capped");
      }
      result.frame_batch_attempted_count =
          persistent_result.attempted_batch_count;
      result.frame_batch_accepted_count =
          persistent_result.accepted_batch_count;
      result.frame_batch_rejected_count =
          persistent_result.rejected_batch_count;
      result.batch_acceptance_attempted_count =
          persistent_result.attempted_batch_count;
      result.batch_acceptance_accepted_count =
          persistent_result.accepted_batch_count;
      result.batch_acceptance_rejected_score_count =
          persistent_result.rejected_batch_count;
      result.attempted_candidate_count =
          persistent_candidate_observation_attempt_count;
      accepted_keys = persistent_result.accepted_keys;
      result.curated_bundle = persistent_result.curated_bundle;
      result.success = true;
      result.rejected_board_observation_count =
          static_cast<int>(candidate_pool_keys.size() > accepted_keys.size()
                               ? candidate_pool_keys.size() -
                                     accepted_keys.size()
                               : 0);
      result.kept_frame_count =
          result.curated_bundle.measurement_dataset.accepted_frame_count;
      result.kept_board_observation_count =
          result.curated_bundle.measurement_dataset
              .accepted_board_observation_count;
      result.kept_outer_point_count =
          result.curated_bundle.measurement_dataset.accepted_outer_point_count;
      result.kept_internal_point_count =
          result.curated_bundle.measurement_dataset
              .accepted_internal_point_count;
      result.kept_total_point_count =
          result.curated_bundle.measurement_dataset.accepted_total_point_count;
      result.warnings.insert(result.warnings.end(),
                             persistent_result.warnings.begin(),
                             persistent_result.warnings.end());
      result.warnings.push_back(
          "Used persistent incremental Stage5 backend estimator for frame-batch selection.");

      std::map<int, Stage5IncrementalBackendBatchResult>
          persistent_batch_result_by_frame;
      std::map<int, int> persistent_batch_attempt_order_by_frame;
      int persistent_batch_attempt_order = 0;
      for (const Stage5IncrementalBackendBatchResult& batch_result :
           persistent_result.batch_results) {
        persistent_batch_result_by_frame[batch_result.frame_index] =
            batch_result;
        persistent_batch_attempt_order_by_frame[batch_result.frame_index] =
            persistent_batch_attempt_order++;
      }

      for (TrialBackendFrameBoardObservationDecision candidate :
           candidate_decisions) {
        const FrameBoardKey key(candidate.frame_index, candidate.board_id);
        const auto batch_it =
            persistent_batch_result_by_frame.find(candidate.frame_index);
        const bool in_attempted_batch =
            batch_it != persistent_batch_result_by_frame.end() &&
            persistent_batch_by_frame[candidate.frame_index]
                    .frame_board_keys.count(key) > 0;
        candidate.frame_batch_candidate = true;
        candidate.persistent_incremental_attempted = in_attempted_batch;
        if (in_attempted_batch) {
          candidate.persistent_incremental_attempt_order =
              persistent_batch_attempt_order_by_frame[candidate.frame_index];
        }
        if (!in_attempted_batch) {
          candidate.kept = false;
          if (result.safety_ceiling_hit) {
            candidate.reason = "runtime_safety_ceiling";
          } else if (
              result.persistent_incremental_adaptive_saturation_stop_hit) {
            candidate.reason =
                result.persistent_incremental_adaptive_saturation_stop_reason
                    .empty()
                    ? "adaptive_information_saturation"
                    : result
                          .persistent_incremental_adaptive_saturation_stop_reason;
          } else {
            candidate.reason = "not_attempted_persistent_incremental";
          }
          decision_by_key[key] = candidate;
          continue;
        }
        const Stage5IncrementalBackendBatchResult& batch_result =
            batch_it->second;
        candidate.attempted_incremental = true;
        candidate.frame_batch_attempted = batch_result.attempted;
        candidate.frame_batch_accepted = batch_result.batch_accepted;
        candidate.persistent_incremental_batch_accepted =
            batch_result.batch_accepted;
        candidate.persistent_incremental_force = batch_result.force;
        candidate.persistent_incremental_trust_region_pass =
            batch_result.trust_region_pass;
        candidate.persistent_incremental_trust_region_backtracking_used =
            batch_result.trust_region_backtracking_used;
        candidate.persistent_incremental_split_residual_health_pass =
            batch_result.split_residual_health_pass;
        candidate.persistent_incremental_pixel_safety_gate_pass =
            batch_result.pixel_safety_gate_pass;
        candidate.persistent_incremental_full_training_pose_refit_health_pass =
            batch_result.full_training_pose_refit_health_pass;
        candidate.persistent_incremental_ray_curve_validity_pass =
            batch_result.ray_curve_validity_pass;
        candidate.persistent_incremental_kb_k3_released =
            batch_result.kb_k3_released;
        candidate.persistent_incremental_kb_k4_released =
            batch_result.kb_k4_released;
        candidate.persistent_incremental_information_gain =
            batch_result.information_gain;
        candidate.persistent_incremental_normalized_information_gain =
            batch_result.normalized_information_gain;
        candidate
            .persistent_incremental_information_gain_normalization_count =
            batch_result.information_gain_normalization_count;
        candidate.persistent_incremental_rank_psi_after =
            batch_result.rank_psi_after;
        candidate.persistent_incremental_rank_psi_deficiency_after =
            batch_result.rank_psi_deficiency_after;
        candidate.persistent_incremental_rank_theta_before =
            batch_result.rank_theta_before;
        candidate.persistent_incremental_rank_theta_after =
            batch_result.rank_theta_after;
        candidate.persistent_incremental_rank_theta_deficiency_after =
            batch_result.rank_theta_deficiency_after;
        candidate.persistent_incremental_svd_tolerance =
            batch_result.svd_tolerance;
        candidate.persistent_incremental_qr_tolerance =
            batch_result.qr_tolerance;
        candidate.persistent_incremental_iterations =
            batch_result.num_iterations;
        candidate.persistent_incremental_last_solver_pass_iterations =
            batch_result.last_solver_pass_iterations;
        candidate.persistent_incremental_continuation_round_count =
            batch_result.continuation_round_count;
        candidate.persistent_incremental_continuation_guard_hit =
            batch_result.continuation_guard_hit;
        candidate.persistent_incremental_pose_prefit_attempted =
            batch_result.pose_prefit_attempted;
        candidate.persistent_incremental_pose_prefit_success =
            batch_result.pose_prefit_success;
        candidate.persistent_incremental_pose_prefit_iterations =
            batch_result.pose_prefit_iterations;
        candidate.persistent_incremental_pose_prefit_objective_start =
            batch_result.pose_prefit_objective_start;
        candidate.persistent_incremental_pose_prefit_objective_final =
            batch_result.pose_prefit_objective_final;
        candidate.persistent_incremental_pose_prefit_last_delta_j =
            batch_result.pose_prefit_last_delta_j;
        candidate.persistent_incremental_pose_prefit_last_delta_x =
            batch_result.pose_prefit_last_delta_x;
        candidate.persistent_incremental_objective_start =
            batch_result.objective_start;
        candidate.persistent_incremental_objective_final =
            batch_result.objective_final;
        candidate.persistent_incremental_objective_last_delta_j =
            batch_result.objective_last_delta_j;
        candidate.persistent_incremental_state_last_delta_x =
            batch_result.state_last_delta_x;
        candidate.persistent_incremental_linear_solver_failure =
            batch_result.linear_solver_failure;
        candidate.persistent_incremental_converged_by_relative_objective =
            batch_result.converged_by_relative_objective;
        candidate.persistent_incremental_converged_by_camera_step =
            batch_result.converged_by_camera_step;
        candidate.persistent_incremental_last_camera_shape_step =
            batch_result.last_camera_shape_step;
        candidate.persistent_incremental_last_camera_focal_relative_step =
            batch_result.last_camera_focal_relative_step;
        candidate.persistent_incremental_last_camera_principal_step_px =
            batch_result.last_camera_principal_step_px;
        candidate.persistent_incremental_objective_decreased =
            batch_result.objective_decreased;
        candidate.persistent_incremental_rmse_before =
            batch_result.rmse_before;
        candidate.persistent_incremental_outer_rmse_before =
            batch_result.outer_rmse_before;
        candidate.persistent_incremental_internal_rmse_before =
            batch_result.internal_rmse_before;
        candidate.persistent_incremental_acceptance_metric_name =
            batch_result.acceptance_metric_name;
        candidate.persistent_incremental_acceptance_metric_unit =
            batch_result.acceptance_metric_unit;
        candidate.persistent_incremental_acceptance_metric_threshold =
            batch_result.acceptance_metric_threshold;
        candidate.persistent_incremental_acceptance_metric_before =
            batch_result.acceptance_metric_before;
        candidate.persistent_incremental_acceptance_metric_after =
            batch_result.acceptance_metric_after;
        candidate.persistent_incremental_acceptance_metric_candidate =
            batch_result.acceptance_metric_candidate;
        candidate.persistent_incremental_acceptance_metric_candidate_p95 =
            batch_result.acceptance_metric_candidate_p95;
        candidate.persistent_incremental_acceptance_metric_candidate_outer =
            batch_result.acceptance_metric_candidate_outer;
        candidate.persistent_incremental_acceptance_metric_candidate_internal =
            batch_result.acceptance_metric_candidate_internal;
        candidate.persistent_incremental_total_p95_after =
            batch_result.total_p95_after;
        candidate.persistent_incremental_outer_p95_after =
            batch_result.outer_p95_after;
        candidate.persistent_incremental_internal_p95_after =
            batch_result.internal_p95_after;
        candidate.persistent_incremental_candidate_rmse_after =
            batch_result.candidate_rmse_after;
        candidate.persistent_incremental_candidate_outer_rmse_after =
            batch_result.candidate_outer_rmse_after;
        candidate.persistent_incremental_candidate_internal_rmse_after =
            batch_result.candidate_internal_rmse_after;
        candidate.persistent_incremental_candidate_total_p95_after =
            batch_result.candidate_total_p95_after;
        candidate.persistent_incremental_candidate_outer_p95_after =
            batch_result.candidate_outer_p95_after;
        candidate.persistent_incremental_candidate_internal_p95_after =
            batch_result.candidate_internal_p95_after;
        candidate.persistent_incremental_pixel_rmse_before =
            batch_result.pixel_rmse_before;
        candidate.persistent_incremental_pixel_rmse_after =
            batch_result.pixel_rmse_after;
        candidate.persistent_incremental_pixel_p95_before =
            batch_result.pixel_p95_before;
        candidate.persistent_incremental_pixel_p95_after =
            batch_result.pixel_p95_after;
        candidate.persistent_incremental_candidate_pixel_rmse_after =
            batch_result.candidate_pixel_rmse_after;
        candidate.persistent_incremental_candidate_pixel_p95_after =
            batch_result.candidate_pixel_p95_after;
        candidate.persistent_incremental_full_training_pixel_rmse_before =
            batch_result.full_training_pixel_rmse_before;
        candidate.persistent_incremental_full_training_pixel_rmse_after =
            batch_result.full_training_pixel_rmse_after;
        candidate.persistent_incremental_full_training_pixel_p95_before =
            batch_result.full_training_pixel_p95_before;
        candidate.persistent_incremental_full_training_pixel_p95_after =
            batch_result.full_training_pixel_p95_after;
        candidate
            .persistent_incremental_full_training_pose_success_rate_before =
            batch_result.full_training_pose_success_rate_before;
        candidate
            .persistent_incremental_full_training_pose_success_rate_after =
            batch_result.full_training_pose_success_rate_after;
        candidate
            .persistent_incremental_full_training_pose_success_count_before =
            batch_result.full_training_pose_success_count_before;
        candidate
            .persistent_incremental_full_training_pose_success_count_after =
            batch_result.full_training_pose_success_count_after;
        candidate.persistent_incremental_full_training_pose_total_count =
            batch_result.full_training_pose_total_count;
        candidate
            .persistent_incremental_full_training_invalid_projection_count_before =
            batch_result.full_training_invalid_projection_count_before;
        candidate
            .persistent_incremental_full_training_invalid_projection_count_after =
            batch_result.full_training_invalid_projection_count_after;
        candidate.persistent_incremental_ray_curve_rms_change_deg =
            batch_result.ray_curve_rms_change_deg;
        candidate.persistent_incremental_ray_curve_max_change_deg =
            batch_result.ray_curve_max_change_deg;
        candidate.persistent_incremental_ray_curve_min_radial_derivative =
            batch_result.ray_curve_min_radial_derivative;
        candidate.persistent_incremental_image_plane_residual_count =
            batch_result.image_plane_residual_count;
	        candidate.persistent_incremental_angular_residual_count =
	            batch_result.angular_residual_count;
	        candidate.persistent_incremental_chordal_residual_count =
	            batch_result.chordal_residual_count;
	        candidate.persistent_incremental_hybrid_angular_selected_count =
	            batch_result.hybrid_angular_selected_count;
	        candidate.persistent_incremental_hybrid_chordal_selected_count =
	            batch_result.hybrid_chordal_selected_count;
	        candidate.persistent_incremental_angular_geometry_failure_count =
	            batch_result.angular_observation_geometry_failure_count;
        candidate.persistent_incremental_trust_region_retry_count =
            batch_result.trust_region_retry_count;
        candidate.persistent_incremental_trust_region_violation_ratio =
            batch_result.trust_region_violation_ratio;
        candidate.persistent_incremental_trust_region_anchor_weight_scale =
            batch_result.trust_region_anchor_weight_scale;
        candidate.persistent_incremental_elapsed_time_seconds =
            batch_result.elapsed_time_seconds;
        candidate.persistent_incremental_commit_state =
            batch_result.committed_or_rollback;
        candidate.persistent_incremental_camera_xi_before =
            batch_result.camera_xi_before;
        candidate.persistent_incremental_camera_alpha_before =
            batch_result.camera_alpha_before;
        candidate.persistent_incremental_camera_fu_before =
            batch_result.camera_fu_before;
        candidate.persistent_incremental_camera_fv_before =
            batch_result.camera_fv_before;
        candidate.persistent_incremental_camera_cu_before =
            batch_result.camera_cu_before;
        candidate.persistent_incremental_camera_cv_before =
            batch_result.camera_cv_before;
        candidate.persistent_incremental_camera_xi_after =
            batch_result.camera_xi_after;
        candidate.persistent_incremental_camera_alpha_after =
            batch_result.camera_alpha_after;
        candidate.persistent_incremental_camera_fu_after =
            batch_result.camera_fu_after;
        candidate.persistent_incremental_camera_fv_after =
            batch_result.camera_fv_after;
        candidate.persistent_incremental_camera_cu_after =
            batch_result.camera_cu_after;
        candidate.persistent_incremental_camera_cv_after =
            batch_result.camera_cv_after;
        candidate.persistent_incremental_camera_k1_before =
            batch_result.camera_k1_before;
        candidate.persistent_incremental_camera_k2_before =
            batch_result.camera_k2_before;
        candidate.persistent_incremental_camera_k3_before =
            batch_result.camera_k3_before;
        candidate.persistent_incremental_camera_k4_before =
            batch_result.camera_k4_before;
        candidate.persistent_incremental_camera_k1_after =
            batch_result.camera_k1_after;
        candidate.persistent_incremental_camera_k2_after =
            batch_result.camera_k2_after;
        candidate.persistent_incremental_camera_k3_after =
            batch_result.camera_k3_after;
        candidate.persistent_incremental_camera_k4_after =
            batch_result.camera_k4_after;
        candidate.information_gain_proxy = batch_result.information_gain;
        candidate.intrinsics_jacobian_info_term =
            batch_result.information_gain;
        candidate.intrinsics_jacobian_rank_gain =
            batch_result.rank_theta_after >= 0 &&
                    batch_result.rank_theta_before >= 0
                ? static_cast<double>(batch_result.rank_theta_after -
                                      batch_result.rank_theta_before)
                : 0.0;
        candidate.global_rmse_after = batch_result.rmse_after;
        candidate.outer_rmse_delta = batch_result.outer_rmse_after;
        candidate.internal_rmse_delta = batch_result.internal_rmse_after;
        candidate.hard_validity_pass = batch_result.objective_finite;
        candidate.legacy_rmse_pass = batch_result.objective_decreased;
        candidate.accepted_by_batch_acceptance =
            batch_result.batch_accepted && !batch_result.force;
        candidate.kept = accepted_keys.find(key) != accepted_keys.end();
        if (candidate.kept) {
          candidate.reason = batch_result.accept_reason.empty()
                                 ? "accepted_persistent_incremental_batch"
                                 : batch_result.accept_reason;
          ++result.accepted_candidate_count;
          if (candidate.intrinsics_diversity_anchor) {
            ++result.intrinsics_diversity_anchor_accepted_count;
          }
        } else {
          candidate.reason = batch_result.reject_reason.empty()
                                 ? "rejected_persistent_incremental_batch"
                                 : batch_result.reject_reason;
          if (candidate.intrinsics_diversity_anchor) {
            ++result.intrinsics_diversity_anchor_rejected_count;
          }
        }
        decision_by_key[key] = candidate;
      }
      result.decisions.reserve(decision_by_key.size());
      for (const auto& entry : decision_by_key) {
        result.decisions.push_back(entry.second);
      }
      return result;
    }

    if (persistent_result.attempted) {
      result.warnings.push_back(
          "Persistent incremental Stage5 backend estimator fallback: " +
          (persistent_result.failure_reason.empty()
               ? persistent_result.fallback_reason
               : persistent_result.failure_reason));
    }

    for (int frame_index : frame_order) {
      if (options.budget_mode ==
              TrialBackendFrameBoardSelectionOptions::BudgetMode::KalibrStyle &&
          traversal_budget_exhausted()) {
        if (!result.safety_ceiling_hit) {
          result.safety_ceiling_hit = true;
          result.warnings.push_back(
              "runtime_safety_ceiling_hit; result is runtime-capped");
        }
        break;
      }
      if (options.budget_mode !=
              TrialBackendFrameBoardSelectionOptions::BudgetMode::KalibrStyle &&
          result.valid_candidate_traversed_count >=
              traversal_budget.traversal_limit) {
        break;
      }

      const std::vector<std::size_t>& indices =
          candidate_indices_by_frame[frame_index];
      std::vector<TrialBackendFrameBoardObservationDecision> batch_members;
      std::set<FrameBoardKey> batch_keys;
      for (std::size_t index : indices) {
        const TrialBackendFrameBoardObservationDecision& member =
            candidate_decisions[index];
        const FrameBoardKey key(member.frame_index, member.board_id);
        if (accepted_keys.find(key) != accepted_keys.end()) {
          continue;
        }
        batch_members.push_back(member);
        batch_keys.insert(key);
      }
      if (batch_members.empty()) {
        continue;
      }

      TrialBackendFrameBoardObservationDecision frame_decision =
          batch_members.front();
      frame_decision.frame_batch_candidate = true;
      frame_decision.frame_batch_attempted = true;
      frame_decision.board_id = -1;
      frame_decision.point_count = 0;
      frame_decision.outer_point_count = 0;
      frame_decision.internal_point_count = 0;
      frame_decision.candidate_score = 0.0;
      frame_decision.coverage_gain = 0.0;
      frame_decision.trial_rmse = 0.0;
      frame_decision.intrinsics_diversity_anchor = false;
      for (const TrialBackendFrameBoardObservationDecision& member :
           batch_members) {
        frame_decision.point_count += member.point_count;
        frame_decision.outer_point_count += member.outer_point_count;
        frame_decision.internal_point_count += member.internal_point_count;
        frame_decision.candidate_score += member.candidate_score;
        frame_decision.coverage_gain += member.coverage_gain;
        frame_decision.trial_rmse =
            std::max(frame_decision.trial_rmse, member.trial_rmse);
        frame_decision.intrinsics_diversity_anchor =
            frame_decision.intrinsics_diversity_anchor ||
            member.intrinsics_diversity_anchor;
      }
      const double batch_member_count =
          static_cast<double>(std::max<std::size_t>(
              1u, batch_members.size()));
      frame_decision.candidate_score /= batch_member_count;
      frame_decision.coverage_gain /= batch_member_count;

      record_candidate_traversal();
      ++result.frame_batch_attempted_count;
      result.attempted_candidate_count +=
          static_cast<int>(batch_members.size());

      std::set<FrameBoardKey> tentative_keys = accepted_keys;
      tentative_keys.insert(batch_keys.begin(), batch_keys.end());
      CalibrationStateBundle tentative_bundle =
          BuildBundleForAcceptedFrameBoardKeys(
              current_scene_template_bundle,
              candidate_pool_bundle,
              tentative_keys,
              candidate_pool_bundle.measurement_dataset.source_stage_label +
                  "_trial_backend_frame_batch_tentative");
      if (!tentative_bundle.IsReadyForBackend()) {
        ++result.frame_batch_rejected_count;
        frame_decision.kept = false;
        frame_decision.reason = "hard_validity_gate";
        for (const TrialBackendFrameBoardObservationDecision& member :
             batch_members) {
          TrialBackendFrameBoardObservationDecision rejected = member;
          rejected.frame_batch_candidate = true;
          rejected.frame_batch_attempted = true;
          rejected.kept = false;
          rejected.reason = "rejected_frame_batch_hard_validity";
          decision_by_key[FrameBoardKey(rejected.frame_index,
                                        rejected.board_id)] = rejected;
        }
        continue;
      }

      AslamBackendCalibrationResult tentative_backend = RunShortTrialBackend(
          tentative_bundle,
          backend_options,
          options,
          "_trial_backend_frame_batch_candidate");
      result.trial_optimization_diagnostics.push_back(
          SummarizeTrialBackendOptimization(
              "frame_batch_candidate", tentative_backend));
      if (!tentative_backend.success) {
        ++result.frame_batch_rejected_count;
        for (const TrialBackendFrameBoardObservationDecision& member :
             batch_members) {
          TrialBackendFrameBoardObservationDecision rejected = member;
          rejected.frame_batch_candidate = true;
          rejected.frame_batch_attempted = true;
          rejected.kept = false;
          rejected.reason = "rejected_frame_batch_hard_validity";
          decision_by_key[FrameBoardKey(rejected.frame_index,
                                        rejected.board_id)] = rejected;
        }
        continue;
      }

      const TrialBackendMetrics tentative_metrics =
          ExtractTrialBackendMetrics(tentative_backend);
      frame_decision.global_rmse_before = current_metrics.overall_rmse;
      frame_decision.global_rmse_after = tentative_metrics.overall_rmse;
      frame_decision.global_rmse_delta =
          tentative_metrics.overall_rmse - current_metrics.overall_rmse;
      frame_decision.outer_rmse_delta =
          tentative_metrics.outer_rmse - current_metrics.outer_rmse;
      frame_decision.internal_rmse_delta =
          tentative_metrics.internal_rmse - current_metrics.internal_rmse;
      frame_decision.legacy_rmse_pass =
          frame_decision.global_rmse_delta <=
              options.accept_max_global_rmse_increase_px &&
          frame_decision.outer_rmse_delta <=
              options.accept_max_outer_rmse_increase_px &&
          frame_decision.internal_rmse_delta <=
              options.accept_max_internal_rmse_increase_px;
      const bool accept_frame_batch =
          batch_acceptance_pass(&frame_decision, tentative_metrics,
                                tentative_backend, false, false,
                                &batch_members);
      if (!accept_frame_batch) {
        ++result.frame_batch_rejected_count;
        for (const TrialBackendFrameBoardObservationDecision& member :
             batch_members) {
          TrialBackendFrameBoardObservationDecision rejected = member;
          rejected.frame_batch_candidate = true;
          rejected.frame_batch_attempted = true;
          rejected.kept = false;
          rejected.reason = frame_decision.reason.empty()
                                ? "rejected_frame_batch_trial"
                                : frame_decision.reason;
          decision_by_key[FrameBoardKey(rejected.frame_index,
                                        rejected.board_id)] = rejected;
        }
        continue;
      }

      ++result.frame_batch_accepted_count;
      accepted_keys.insert(batch_keys.begin(), batch_keys.end());
      current_metrics = tentative_metrics;
      current_backend = tentative_backend;
      if (options.carry_accepted_trial_state) {
        UpdateSceneTemplateFromAcceptedTrialBackendMerged(
            current_backend,
            candidate_pool_bundle,
            "trial_backend_incremental_optimized",
            candidate_pool_bundle.measurement_dataset.source_stage_label +
                "_trial_backend_frame_batch_state_carry",
            &current_scene_template_bundle);
      }
      for (const TrialBackendFrameBoardObservationDecision& member :
           batch_members) {
        TrialBackendFrameBoardObservationDecision accepted = member;
        accepted.frame_batch_candidate = true;
        accepted.frame_batch_attempted = true;
        accepted.frame_batch_accepted = true;
        accepted.attempted_incremental = true;
        accepted.kept = true;
        accepted.reason = "accepted_frame_batch_trial";
        accepted.global_rmse_before = frame_decision.global_rmse_before;
        accepted.global_rmse_after = frame_decision.global_rmse_after;
        accepted.global_rmse_delta = frame_decision.global_rmse_delta;
        accepted.outer_rmse_delta = frame_decision.outer_rmse_delta;
        accepted.internal_rmse_delta = frame_decision.internal_rmse_delta;
        accepted.hard_validity_pass = frame_decision.hard_validity_pass;
        accepted.legacy_rmse_pass = frame_decision.legacy_rmse_pass;
        accepted.catastrophic_residual = frame_decision.catastrophic_residual;
        accepted.score_term = frame_decision.score_term;
        accepted.coverage_term = frame_decision.coverage_term;
        accepted.intrinsics_jacobian_logdet_gain =
            frame_decision.intrinsics_jacobian_logdet_gain;
        accepted.intrinsics_jacobian_trace_gain =
            frame_decision.intrinsics_jacobian_trace_gain;
        accepted.intrinsics_jacobian_rank_gain =
            frame_decision.intrinsics_jacobian_rank_gain;
        accepted.intrinsics_jacobian_info_term =
            frame_decision.intrinsics_jacobian_info_term;
        accepted.frame_completion_bonus =
            frame_decision.frame_completion_bonus;
        accepted.new_board_bonus = frame_decision.new_board_bonus;
        accepted.cap_penalty = frame_decision.cap_penalty;
        accepted.information_gain_proxy =
            frame_decision.information_gain_proxy;
        accepted.residual_overage_penalty =
            frame_decision.residual_overage_penalty;
        accepted.batch_acceptance_score =
            frame_decision.batch_acceptance_score;
        accepted.accepted_by_batch_acceptance =
            frame_decision.accepted_by_batch_acceptance;
        ++accepted_candidate_count_by_board[accepted.board_id];
        ++accepted_candidate_count_by_frame[accepted.frame_index];
        ++accepted_observation_count_by_frame[accepted.frame_index];
        ++result.accepted_candidate_count;
        if (accepted.intrinsics_diversity_anchor) {
          ++result.intrinsics_diversity_anchor_accepted_count;
        }
        const FrameBoardKey key(accepted.frame_index, accepted.board_id);
        const auto info_it = intrinsics_jacobian_info_by_key.find(key);
        if (info_it != intrinsics_jacobian_info_by_key.end() &&
            info_it->second.available) {
          accepted_intrinsics_fisher += info_it->second.fisher;
        }
        decision_by_key[key] = accepted;
      }
      current_metrics = tentative_metrics;
      current_backend = tentative_backend;
      if (options.carry_accepted_trial_state) {
        UpdateSceneTemplateFromAcceptedTrialBackendMerged(
            current_backend,
            candidate_pool_bundle,
            "trial_backend_incremental_optimized",
            candidate_pool_bundle.measurement_dataset.source_stage_label +
                "_trial_backend_frame_batch_state_carry",
            &current_scene_template_bundle);
      }
    }
  }

  for (TrialBackendFrameBoardObservationDecision& candidate :
       candidate_decisions) {
    const FrameBoardKey key(candidate.frame_index, candidate.board_id);
    if (accepted_keys.find(key) != accepted_keys.end()) {
      continue;
    }
    candidate.attempted_incremental = false;
    candidate.global_rmse_before = current_metrics.overall_rmse;
    const bool force_include_candidate =
        candidate.force_include_candidate ||
        IsForceIncludeFrameBoardCandidate(
            key, candidate.frame_label, options);
    candidate.force_include_candidate = force_include_candidate;
    const auto can_try_frame_cohesion_candidate = [&]() {
      const auto frame_capacity_it =
          candidate_pool_board_capacity_by_frame.find(candidate.frame_index);
      const int observed_board_capacity =
          frame_capacity_it == candidate_pool_board_capacity_by_frame.end()
              ? 0
              : frame_capacity_it->second;
      const int selected_board_count =
          accepted_observation_count_by_frame[candidate.frame_index];
      const bool has_observed_companion_to_rescue =
          selected_board_count > 0 &&
          selected_board_count < observed_board_capacity;
      const bool within_companion_cap =
          options.frame_cohesion_max_companions_per_frame <= 0 ||
          accepted_frame_cohesion_count_by_frame[candidate.frame_index] <
              options.frame_cohesion_max_companions_per_frame;
      return options.frame_cohesion_enabled &&
             has_observed_companion_to_rescue &&
             within_companion_cap &&
             candidate.candidate_score >=
                 options.frame_cohesion_min_candidate_score;
    };
    const bool board_cap_exceeded =
        options.max_accepted_per_board > 0 &&
        accepted_candidate_count_by_board[candidate.board_id] >=
            options.max_accepted_per_board;
    const bool frame_cap_exceeded =
        options.max_accepted_per_frame > 0 &&
        accepted_candidate_count_by_frame[candidate.frame_index] >=
            options.max_accepted_per_frame;
    if (!kalibr_style_batch_mode && !candidate.kept &&
        !force_include_candidate) {
      candidate.reason = "rejected_by_wide_trial_residual_outlier";
      decision_by_key[key] = candidate;
      continue;
    }
    if (!kalibr_style_batch_mode &&
        !force_include_candidate &&
        candidate.candidate_score < options.min_candidate_score) {
      candidate.kept = false;
      candidate.reason = "rejected_below_min_candidate_score";
      decision_by_key[key] = candidate;
      continue;
    }
    if (!kalibr_style_batch_mode &&
        !force_include_candidate &&
        candidate.coverage_gain < options.min_coverage_gain) {
      candidate.kept = false;
      candidate.reason = "rejected_below_min_coverage_gain";
      decision_by_key[key] = candidate;
      continue;
    }
    if (options.use_consistency_score && candidate.consistency_available &&
        options.consistency_max_translation_error_mm > 0.0 &&
        candidate.consistency_translation_error_mm >
            options.consistency_max_translation_error_mm) {
      candidate.kept = false;
      candidate.reason = "rejected_by_consistency_translation";
      decision_by_key[key] = candidate;
      continue;
    }
    if (options.use_consistency_score && candidate.consistency_available &&
        options.consistency_max_rotation_error_deg > 0.0 &&
        candidate.consistency_rotation_error_deg >
            options.consistency_max_rotation_error_deg) {
      candidate.kept = false;
      candidate.reason = "rejected_by_consistency_rotation";
      decision_by_key[key] = candidate;
      continue;
    }
    if (options.use_consistency_score && candidate.consistency_available &&
        options.consistency_max_local_outer_rmse_px > 0.0 &&
        candidate.consistency_local_outer_rmse >
            options.consistency_max_local_outer_rmse_px) {
      candidate.kept = false;
      candidate.reason = "rejected_by_consistency_local_outer_rmse";
      decision_by_key[key] = candidate;
      continue;
    }
    if (!kalibr_style_batch_mode &&
        !force_include_candidate && board_cap_exceeded) {
      if (can_try_frame_cohesion_candidate()) {
        candidate.frame_cohesion_candidate = true;
        candidate.reason = "frame_cohesion_candidate_from_board_cap";
        ++result.frame_cohesion_candidate_count;
      } else {
        candidate.kept = false;
        candidate.reason = "not_attempted_board_candidate_cap";
        decision_by_key[key] = candidate;
        continue;
      }
    }
    if (!kalibr_style_batch_mode &&
        !force_include_candidate && !candidate.frame_cohesion_candidate &&
        frame_cap_exceeded) {
      if (can_try_frame_cohesion_candidate()) {
        candidate.frame_cohesion_candidate = true;
        candidate.reason = "frame_cohesion_candidate_from_frame_cap";
        ++result.frame_cohesion_candidate_count;
      } else {
        candidate.kept = false;
        candidate.reason = "not_attempted_frame_candidate_cap";
        decision_by_key[key] = candidate;
        continue;
      }
    }
    if (!kalibr_style_batch_mode &&
        !force_include_candidate &&
        !candidate.frame_cohesion_candidate &&
        options.budget_mode !=
            TrialBackendFrameBoardSelectionOptions::BudgetMode::KalibrStyle &&
        result.attempted_candidate_count >=
            traversal_budget.traversal_limit) {
      candidate.kept = false;
      candidate.reason = "not_attempted_candidate_limit";
      decision_by_key[key] = candidate;
      continue;
    }
    if (options.budget_mode ==
            TrialBackendFrameBoardSelectionOptions::BudgetMode::KalibrStyle &&
        traversal_budget_exhausted()) {
      mark_runtime_safety_ceiling(&candidate);
      decision_by_key[key] = candidate;
      continue;
    }
    if (kalibr_style_batch_mode && !force_include_candidate &&
        options.budget_mode !=
            TrialBackendFrameBoardSelectionOptions::BudgetMode::KalibrStyle &&
        !candidate.frame_cohesion_candidate &&
        result.valid_candidate_traversed_count >=
            traversal_budget.traversal_limit) {
      candidate.kept = false;
      candidate.reason = "batch_trial_budget_limit";
      decision_by_key[key] = candidate;
      continue;
    }
    record_candidate_traversal();
    candidate.attempted_incremental = true;
    if (candidate.frame_cohesion_candidate) {
      candidate.frame_cohesion_attempted = true;
      ++result.frame_cohesion_attempted_count;
    } else {
      ++result.attempted_candidate_count;
    }
    std::set<FrameBoardKey> tentative_keys = accepted_keys;
    tentative_keys.insert(key);
    CalibrationStateBundle tentative_bundle =
        BuildBundleForAcceptedFrameBoardKeys(
            current_scene_template_bundle,
            candidate_pool_bundle,
            tentative_keys,
            candidate_pool_bundle.measurement_dataset.source_stage_label +
                "_trial_backend_incremental_tentative");
    if (!tentative_bundle.IsReadyForBackend()) {
      candidate.kept = false;
      candidate.reason =
          kalibr_style_batch_mode ? "hard_validity_gate"
                                  : "tentative_bundle_not_ready";
      if (kalibr_style_batch_mode) {
        ++result.batch_acceptance_attempted_count;
        ++result.batch_acceptance_rejected_hard_validity_count;
      }
      decision_by_key[key] = candidate;
      continue;
    }
    AslamBackendCalibrationResult tentative_backend = RunShortTrialBackend(
        tentative_bundle,
        backend_options,
        options,
        "_trial_backend_incremental_candidate");
    result.trial_optimization_diagnostics.push_back(
        SummarizeTrialBackendOptimization(
            "incremental_candidate", tentative_backend));
    if (!tentative_backend.success) {
      candidate.kept = false;
      candidate.reason =
          kalibr_style_batch_mode ? "hard_validity_gate"
                                  : "tentative_backend_failed";
      if (kalibr_style_batch_mode) {
        ++result.batch_acceptance_attempted_count;
        ++result.batch_acceptance_rejected_hard_validity_count;
      }
      decision_by_key[key] = candidate;
      continue;
    }
    const TrialBackendMetrics tentative_metrics =
        ExtractTrialBackendMetrics(tentative_backend);
    candidate.global_rmse_after = tentative_metrics.overall_rmse;
    candidate.global_rmse_delta =
        tentative_metrics.overall_rmse - current_metrics.overall_rmse;
    candidate.outer_rmse_delta =
        tentative_metrics.outer_rmse - current_metrics.outer_rmse;
    candidate.internal_rmse_delta =
        tentative_metrics.internal_rmse - current_metrics.internal_rmse;
    candidate.legacy_rmse_pass =
        candidate.global_rmse_delta <=
            options.accept_max_global_rmse_increase_px &&
        candidate.outer_rmse_delta <=
            options.accept_max_outer_rmse_increase_px &&
        candidate.internal_rmse_delta <=
            options.accept_max_internal_rmse_increase_px;
    const bool accept =
        kalibr_style_batch_mode
            ? batch_acceptance_pass(&candidate, tentative_metrics,
                                    tentative_backend, board_cap_exceeded,
                                    frame_cap_exceeded, nullptr)
            : candidate.legacy_rmse_pass;
    if (accept) {
      candidate.kept = true;
      candidate.reason = "accepted_incremental_trial";
      accepted_keys.insert(key);
      const auto info_it = intrinsics_jacobian_info_by_key.find(key);
      if (info_it != intrinsics_jacobian_info_by_key.end() &&
          info_it->second.available) {
        accepted_intrinsics_fisher += info_it->second.fisher;
      }
      current_metrics = tentative_metrics;
      current_backend = tentative_backend;
      if (options.carry_accepted_trial_state) {
        UpdateSceneTemplateFromAcceptedTrialBackendMerged(
            current_backend,
            candidate_pool_bundle,
            "trial_backend_incremental_optimized",
            candidate_pool_bundle.measurement_dataset.source_stage_label +
                "_trial_backend_incremental_state_carry",
            &current_scene_template_bundle);
      }
      if (candidate.intrinsics_diversity_anchor) {
        ++result.intrinsics_diversity_anchor_accepted_count;
      }
      if (candidate.frame_cohesion_candidate) {
        candidate.frame_cohesion_accepted = true;
        ++result.frame_cohesion_accepted_count;
        ++accepted_frame_cohesion_count_by_frame[candidate.frame_index];
        ++accepted_observation_count_by_frame[candidate.frame_index];
        ++accepted_candidate_count_by_board[candidate.board_id];
        ++accepted_candidate_count_by_frame[candidate.frame_index];
        candidate.reason = "accepted_frame_cohesion_trial";
      } else {
        ++result.accepted_candidate_count;
        ++accepted_candidate_count_by_board[candidate.board_id];
        ++accepted_candidate_count_by_frame[candidate.frame_index];
        ++accepted_observation_count_by_frame[candidate.frame_index];
      }
    } else {
      candidate.kept = false;
      if (!kalibr_style_batch_mode || candidate.reason.empty()) {
        candidate.reason =
            candidate.frame_cohesion_candidate
                ? "rejected_frame_cohesion_rmse_delta"
                : "rejected_incremental_rmse_delta";
      }
      if (candidate.frame_cohesion_candidate) {
        ++result.frame_cohesion_rejected_count;
      }
      if (candidate.intrinsics_diversity_anchor) {
        ++result.intrinsics_diversity_anchor_rejected_count;
      }
    }
    decision_by_key[key] = candidate;
  }

  if (options.frame_cohesion_enabled) {
    std::map<int, std::vector<TrialBackendFrameBoardObservationDecision> >
        post_pass_companions_by_frame;
    for (const TrialBackendFrameBoardObservationDecision& candidate :
         candidate_decisions) {
      const FrameBoardKey key(candidate.frame_index, candidate.board_id);
      if (accepted_keys.find(key) != accepted_keys.end()) {
        continue;
      }
      const auto frame_capacity_it =
          candidate_pool_board_capacity_by_frame.find(candidate.frame_index);
      if (frame_capacity_it == candidate_pool_board_capacity_by_frame.end()) {
        continue;
      }
      const int observed_board_capacity = frame_capacity_it->second;
      if (accepted_observation_count_by_frame[candidate.frame_index] <= 0 ||
          accepted_observation_count_by_frame[candidate.frame_index] >=
              observed_board_capacity) {
        continue;
      }
      TrialBackendFrameBoardObservationDecision companion = candidate;
      companion.frame_cohesion_candidate = true;
      companion.frame_cohesion_attempted = false;
      companion.frame_cohesion_accepted = false;
      companion.soft_weight = 1.0;
      companion.reason = "frame_cohesion_post_pass_candidate";
      post_pass_companions_by_frame[candidate.frame_index].push_back(companion);
    }
    int post_pass_companion_count = 0;
    for (const auto& entry : post_pass_companions_by_frame) {
      post_pass_companion_count += static_cast<int>(entry.second.size());
    }
    result.valid_candidate_count += post_pass_companion_count;

    for (auto& entry : post_pass_companions_by_frame) {
      std::sort(
          entry.second.begin(),
          entry.second.end(), ScoreSortedBefore);
      if (result.candidate_order_mode ==
          TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
              ::RandomShuffle) {
        std::mt19937 companion_rng(
            result.candidate_shuffle_seed ^
            static_cast<unsigned int>(entry.first * 2654435761u));
        std::shuffle(entry.second.begin(), entry.second.end(),
                     companion_rng);
      }
    }

    bool stop_frame_cohesion_traversal = false;
    for (auto& entry : post_pass_companions_by_frame) {
      if (stop_frame_cohesion_traversal) {
        break;
      }
      const int frame_index = entry.first;
      const auto frame_capacity_it =
          candidate_pool_board_capacity_by_frame.find(frame_index);
      if (frame_capacity_it == candidate_pool_board_capacity_by_frame.end()) {
        continue;
      }
      const int observed_board_capacity = frame_capacity_it->second;
      for (TrialBackendFrameBoardObservationDecision candidate : entry.second) {
        if (stop_frame_cohesion_traversal) {
          break;
        }
        if (accepted_observation_count_by_frame[frame_index] >=
            observed_board_capacity) {
          break;
        }
        if (options.frame_cohesion_max_companions_per_frame > 0 &&
            accepted_frame_cohesion_count_by_frame[frame_index] >=
                options.frame_cohesion_max_companions_per_frame) {
          break;
        }
        const FrameBoardKey key(candidate.frame_index, candidate.board_id);
        if (accepted_keys.find(key) != accepted_keys.end()) {
          continue;
        }
        const bool board_cap_exceeded =
            options.max_accepted_per_board > 0 &&
            accepted_candidate_count_by_board[candidate.board_id] >=
                options.max_accepted_per_board;
        const bool frame_cap_exceeded =
            options.max_accepted_per_frame > 0 &&
            accepted_candidate_count_by_frame[frame_index] >=
                options.max_accepted_per_frame;

        ++result.frame_cohesion_candidate_count;
        if (options.budget_mode ==
                TrialBackendFrameBoardSelectionOptions::BudgetMode::
                    KalibrStyle &&
            traversal_budget_exhausted()) {
          mark_runtime_safety_ceiling(&candidate);
          decision_by_key[key] = candidate;
          stop_frame_cohesion_traversal = true;
          break;
        }
        record_candidate_traversal();
        candidate.attempted_incremental = true;
        candidate.frame_cohesion_attempted = true;
        candidate.global_rmse_before = current_metrics.overall_rmse;
        ++result.frame_cohesion_attempted_count;

        std::set<FrameBoardKey> tentative_keys = accepted_keys;
        tentative_keys.insert(key);
        CalibrationStateBundle tentative_bundle =
            BuildBundleForAcceptedFrameBoardKeys(
                current_scene_template_bundle,
                candidate_pool_bundle,
                tentative_keys,
                candidate_pool_bundle.measurement_dataset.source_stage_label +
                    "_trial_backend_frame_cohesion_post_pass_tentative");
        if (!tentative_bundle.IsReadyForBackend()) {
          candidate.kept = false;
          candidate.reason =
              kalibr_style_batch_mode ? "hard_validity_gate"
                                      : "tentative_bundle_not_ready";
          if (kalibr_style_batch_mode) {
            ++result.batch_acceptance_attempted_count;
            ++result.batch_acceptance_rejected_hard_validity_count;
          }
          ++result.frame_cohesion_rejected_count;
          decision_by_key[key] = candidate;
          continue;
        }
        AslamBackendCalibrationResult tentative_backend = RunShortTrialBackend(
            tentative_bundle,
            backend_options,
            options,
            "_trial_backend_frame_cohesion_post_pass_candidate");
        result.trial_optimization_diagnostics.push_back(
            SummarizeTrialBackendOptimization(
                "frame_cohesion_post_pass_candidate", tentative_backend));
        if (!tentative_backend.success) {
          candidate.kept = false;
          candidate.reason =
              kalibr_style_batch_mode ? "hard_validity_gate"
                                      : "tentative_backend_failed";
          if (kalibr_style_batch_mode) {
            ++result.batch_acceptance_attempted_count;
            ++result.batch_acceptance_rejected_hard_validity_count;
          }
          ++result.frame_cohesion_rejected_count;
          decision_by_key[key] = candidate;
          continue;
        }
        const TrialBackendMetrics tentative_metrics =
            ExtractTrialBackendMetrics(tentative_backend);
        candidate.global_rmse_after = tentative_metrics.overall_rmse;
        candidate.global_rmse_delta =
            tentative_metrics.overall_rmse - current_metrics.overall_rmse;
        candidate.outer_rmse_delta =
            tentative_metrics.outer_rmse - current_metrics.outer_rmse;
        candidate.internal_rmse_delta =
            tentative_metrics.internal_rmse - current_metrics.internal_rmse;
        candidate.legacy_rmse_pass =
            candidate.global_rmse_delta <=
                options.accept_max_global_rmse_increase_px &&
            candidate.outer_rmse_delta <=
                options.accept_max_outer_rmse_increase_px &&
            candidate.internal_rmse_delta <=
                options.accept_max_internal_rmse_increase_px;
        const bool accept =
            kalibr_style_batch_mode
                ? batch_acceptance_pass(&candidate, tentative_metrics,
                                        tentative_backend, board_cap_exceeded,
                                        frame_cap_exceeded, nullptr)
                : candidate.legacy_rmse_pass;
        if (accept) {
          candidate.kept = true;
          candidate.frame_cohesion_accepted = true;
          candidate.reason = "accepted_frame_cohesion_trial";
          accepted_keys.insert(key);
          const auto info_it = intrinsics_jacobian_info_by_key.find(key);
          if (info_it != intrinsics_jacobian_info_by_key.end() &&
              info_it->second.available) {
            accepted_intrinsics_fisher += info_it->second.fisher;
          }
          current_metrics = tentative_metrics;
          current_backend = tentative_backend;
          if (options.carry_accepted_trial_state) {
            UpdateSceneTemplateFromAcceptedTrialBackendMerged(
                current_backend,
                candidate_pool_bundle,
                "trial_backend_incremental_optimized",
                candidate_pool_bundle.measurement_dataset.source_stage_label +
                    "_trial_backend_frame_cohesion_state_carry",
                &current_scene_template_bundle);
          }
          if (candidate.intrinsics_diversity_anchor) {
            ++result.intrinsics_diversity_anchor_accepted_count;
          }
          ++result.frame_cohesion_accepted_count;
          ++accepted_frame_cohesion_count_by_frame[frame_index];
          ++accepted_observation_count_by_frame[frame_index];
          ++accepted_candidate_count_by_board[candidate.board_id];
          ++accepted_candidate_count_by_frame[frame_index];
        } else {
          candidate.kept = false;
          if (!kalibr_style_batch_mode || candidate.reason.empty()) {
            candidate.reason = "rejected_frame_cohesion_rmse_delta";
          }
          ++result.frame_cohesion_rejected_count;
          if (candidate.intrinsics_diversity_anchor) {
            ++result.intrinsics_diversity_anchor_rejected_count;
          }
        }
        decision_by_key[key] = candidate;
      }
    }
  }

  if (frame_consolidation_mode) {
    std::map<int, std::vector<FrameBoardKey> > accepted_keys_by_frame;
    for (const FrameBoardKey& key : accepted_keys) {
      accepted_keys_by_frame[key.first].push_back(key);
    }
    std::set<int> baseline_frame_indices;
    for (const FrameBoardKey& key : baseline_keys) {
      baseline_frame_indices.insert(key.first);
    }

    std::vector<int> candidate_frame_order;
    std::set<int> seen_candidate_frames;
    for (const TrialBackendFrameBoardObservationDecision& decision :
         candidate_decisions) {
      if (accepted_keys_by_frame.find(decision.frame_index) ==
          accepted_keys_by_frame.end()) {
        continue;
      }
      if (baseline_frame_indices.count(decision.frame_index) > 0) {
        continue;
      }
      if (seen_candidate_frames.insert(decision.frame_index).second) {
        candidate_frame_order.push_back(decision.frame_index);
      }
    }

    std::set<FrameBoardKey> consolidated_keys;
    Eigen::Matrix<double, 6, 6> consolidated_fisher =
        Eigen::Matrix<double, 6, 6>::Zero();
    for (const auto& entry : accepted_keys_by_frame) {
      const int frame_index = entry.first;
      if (baseline_frame_indices.count(frame_index) == 0) {
        continue;
      }
      for (const FrameBoardKey& key : entry.second) {
        consolidated_keys.insert(key);
        const auto info_it = intrinsics_jacobian_info_by_key.find(key);
        if (info_it != intrinsics_jacobian_info_by_key.end() &&
            info_it->second.available) {
          consolidated_fisher += info_it->second.fisher;
        }
      }
    }

    result.frame_consolidation_candidate_count =
        static_cast<int>(candidate_frame_order.size());
    const double mi_threshold =
        std::max(0.0, options.acceptance_information_gain_threshold);
    const double critical_view_threshold = 0.25 * mi_threshold;

    for (int frame_index : candidate_frame_order) {
      const auto frame_it = accepted_keys_by_frame.find(frame_index);
      if (frame_it == accepted_keys_by_frame.end()) {
        continue;
      }
      Eigen::Matrix<double, 6, 6> frame_fisher =
          Eigen::Matrix<double, 6, 6>::Zero();
      bool has_information = false;
      bool critical_view = false;
      int board_count = 0;
      for (const FrameBoardKey& key : frame_it->second) {
        const auto decision_it = decision_by_key.find(key);
        if (decision_it != decision_by_key.end()) {
          critical_view =
              critical_view || decision_it->second.intrinsics_diversity_anchor;
        }
        const auto info_it = intrinsics_jacobian_info_by_key.find(key);
        if (info_it != intrinsics_jacobian_info_by_key.end() &&
            info_it->second.available) {
          frame_fisher += info_it->second.fisher;
          has_information = true;
        }
        ++board_count;
      }

      const double before_logdet =
          RegularizedFisherLogDet(consolidated_fisher);
      const double before_rank = FisherRankProxy(consolidated_fisher);
      const Eigen::Matrix<double, 6, 6> candidate_fisher =
          consolidated_fisher + frame_fisher;
      const double after_logdet =
          RegularizedFisherLogDet(candidate_fisher);
      const double after_rank = FisherRankProxy(candidate_fisher);
      const double information_gain =
          has_information ? std::max(0.0, after_logdet - before_logdet) : 0.0;
      const double rank_gain =
          has_information ? std::max(0.0, after_rank - before_rank) : 0.0;
      const bool accept_frame =
          has_information &&
          (information_gain > mi_threshold ||
           rank_gain > options.acceptance_rank_gain_threshold ||
           (critical_view && information_gain > critical_view_threshold));

      if (accept_frame) {
        ++result.frame_consolidation_accepted_count;
        consolidated_fisher = candidate_fisher;
        for (const FrameBoardKey& key : frame_it->second) {
          consolidated_keys.insert(key);
          auto decision_it = decision_by_key.find(key);
          if (decision_it != decision_by_key.end()) {
            decision_it->second.frame_consolidation_candidate = true;
            decision_it->second.frame_consolidation_accepted = true;
            decision_it->second.intrinsics_jacobian_logdet_gain =
                information_gain;
            decision_it->second.intrinsics_jacobian_rank_gain = rank_gain;
            decision_it->second.intrinsics_jacobian_info_term =
                information_gain;
            if (decision_it->second.reason.empty() ||
                decision_it->second.reason ==
                    "accepted_incremental_trial" ||
                decision_it->second.reason ==
                    "accepted_frame_cohesion_trial") {
              decision_it->second.reason =
                  "accepted_frame_consolidation";
            }
          }
        }
      } else {
        ++result.frame_consolidation_rejected_count;
        result.frame_consolidation_dropped_board_observation_count +=
            board_count;
        for (const FrameBoardKey& key : frame_it->second) {
          auto decision_it = decision_by_key.find(key);
          if (decision_it != decision_by_key.end()) {
            decision_it->second.kept = false;
            decision_it->second.frame_consolidation_candidate = true;
            decision_it->second.frame_consolidation_accepted = false;
            decision_it->second.intrinsics_jacobian_logdet_gain =
                information_gain;
            decision_it->second.intrinsics_jacobian_rank_gain = rank_gain;
            decision_it->second.intrinsics_jacobian_info_term =
                information_gain;
            decision_it->second.reason =
                critical_view
                    ? "rejected_frame_consolidation_low_information_gain"
                    : "rejected_frame_consolidation_redundant";
          }
        }
      }
    }

    accepted_keys = consolidated_keys;
  }

  CalibrationStateBundle scene_template_for_curated = candidate_pool_bundle;
  const bool can_carry_accepted_trial_state =
      options.carry_accepted_trial_state &&
      current_scene_template_bundle.scene_state.IsValid();
  if (can_carry_accepted_trial_state) {
    scene_template_for_curated.scene_state =
        current_scene_template_bundle.scene_state;
    scene_template_for_curated.warnings.push_back(
        "Carried accepted trial backend optimized state into final curated bundle.");
  }

  CalibrationStateBundle curated = BuildBundleForAcceptedFrameBoardKeys(
      scene_template_for_curated,
      candidate_pool_bundle,
      accepted_keys,
      candidate_pool_bundle.measurement_dataset.source_stage_label +
          "_trial_backend_incremental_selected");
  if (can_carry_accepted_trial_state) {
    curated.warnings.push_back(
        "Stage5 trial backend selection carried accepted optimized state; "
        "frame-board keys still come from the frozen baseline-compatible acceptance path.");
  }
  curated.warnings.push_back(
      "Applied incremental trial-backend frame-board observation selection.");

  result.curated_bundle = curated;
  result.success = true;
  result.rejected_board_observation_count =
      static_cast<int>(candidate_pool_keys.size() > accepted_keys.size()
                           ? candidate_pool_keys.size() - accepted_keys.size()
                           : 0);
  result.kept_frame_count =
      curated.measurement_dataset.accepted_frame_count;
  result.kept_board_observation_count =
      curated.measurement_dataset.accepted_board_observation_count;
  result.kept_outer_point_count =
      curated.measurement_dataset.accepted_outer_point_count;
  result.kept_internal_point_count =
      curated.measurement_dataset.accepted_internal_point_count;
  result.kept_total_point_count =
      curated.measurement_dataset.accepted_total_point_count;
  result.decisions.reserve(decision_by_key.size());
  for (const auto& entry : decision_by_key) {
    result.decisions.push_back(entry.second);
  }
  return result;
}

double ComputePercentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double clamped = std::max(0.0, std::min(1.0, percentile));
  const double index = clamped * static_cast<double>(values.size() - 1);
  const std::size_t lower =
      static_cast<std::size_t>(std::floor(index));
  const std::size_t upper =
      static_cast<std::size_t>(std::ceil(index));
  if (lower == upper) {
    return values[lower];
  }
  const double blend = index - static_cast<double>(lower);
  return (1.0 - blend) * values[lower] + blend * values[upper];
}

double ElapsedSeconds(const std::chrono::steady_clock::time_point& start_time) {
  return std::chrono::duration_cast<std::chrono::duration<double> >(
             std::chrono::steady_clock::now() - start_time)
      .count();
}

void ComputeKalibrStyleResidualStatistics(
    const std::vector<CameraModelRefitPointDiagnostics>& point_diagnostics,
    double* mean_residual_x,
    double* mean_residual_y,
    double* std_residual_x,
    double* std_residual_y) {
  if (mean_residual_x == nullptr || mean_residual_y == nullptr ||
      std_residual_x == nullptr || std_residual_y == nullptr) {
    return;
  }
  *mean_residual_x = 0.0;
  *mean_residual_y = 0.0;
  *std_residual_x = 0.0;
  *std_residual_y = 0.0;
  if (point_diagnostics.empty()) {
    return;
  }

  double sum_x = 0.0;
  double sum_y = 0.0;
  for (const CameraModelRefitPointDiagnostics& point_diag : point_diagnostics) {
    sum_x += point_diag.residual_xy.x();
    sum_y += point_diag.residual_xy.y();
  }
  const double count = static_cast<double>(point_diagnostics.size());
  *mean_residual_x = sum_x / count;
  *mean_residual_y = sum_y / count;

  double sum_sq_x = 0.0;
  double sum_sq_y = 0.0;
  for (const CameraModelRefitPointDiagnostics& point_diag : point_diagnostics) {
    const double dx = point_diag.residual_xy.x() - *mean_residual_x;
    const double dy = point_diag.residual_xy.y() - *mean_residual_y;
    sum_sq_x += dx * dx;
    sum_sq_y += dy * dy;
  }
  *std_residual_x = std::sqrt(sum_sq_x / count);
  *std_residual_y = std::sqrt(sum_sq_y / count);
}

void WriteKalibrStyleResidualStatistics(
    std::ostream& output,
    const std::string& prefix,
    const CameraModelRefitEvaluationResult& evaluation) {
  output << prefix << "_mean_residual_x: " << evaluation.mean_residual_x << "\n";
  output << prefix << "_mean_residual_y: " << evaluation.mean_residual_y << "\n";
  output << prefix << "_std_residual_x: " << evaluation.std_residual_x << "\n";
  output << prefix << "_std_residual_y: " << evaluation.std_residual_y << "\n";
}

std::vector<int> NormalizeBoardIds(const std::vector<int>& configured_ids,
                                   int fallback_tag_id) {
  std::vector<int> board_ids;
  const auto append_if_valid = [&board_ids](int board_id) {
    if (board_id < 0) {
      return;
    }
    if (std::find(board_ids.begin(), board_ids.end(), board_id) == board_ids.end()) {
      board_ids.push_back(board_id);
    }
  };
  for (int board_id : configured_ids) {
    append_if_valid(board_id);
  }
  if (board_ids.empty()) {
    append_if_valid(fallback_tag_id);
  }
  return board_ids;
}

ApriltagInternalConfig NormalizeConfig(ApriltagInternalConfig config) {
  config.tag_ids = NormalizeBoardIds(config.tag_ids, config.tag_id);
  if (!config.tag_ids.empty()) {
    config.tag_id = config.tag_ids.front();
  }
  config.outer_detector_config.tag_ids = config.tag_ids;
  config.outer_detector_config.tag_id = config.tag_id;
  return config;
}

ApriltagInternalConfig BoardConfigForId(const ApriltagInternalConfig& config, int board_id) {
  ApriltagInternalConfig board_config = config;
  board_config.tag_id = board_id;
  board_config.tag_ids = {board_id};
  board_config.outer_detector_config.tag_id = board_id;
  board_config.outer_detector_config.tag_ids = {board_id};
  return board_config;
}

ApriltagInternalDetectionOptions MakeDetectionOptions(
    const ApriltagInternalConfig& config) {
  ApriltagInternalDetectionOptions options;
  options.do_subpix_refinement = true;
  options.max_subpix_displacement2 = config.max_subpix_displacement2;
  options.min_border_distance = 4.0;
  options.canonical_pixels_per_module = config.canonical_pixels_per_module;
  options.refinement_window_radius = config.refinement_window_radius;
  options.internal_subpix_window_scale = config.internal_subpix_window_scale;
  options.internal_subpix_window_min = config.internal_subpix_window_min;
  options.internal_subpix_window_max = config.internal_subpix_window_max;
  options.internal_subpix_displacement_scale = config.internal_subpix_displacement_scale;
  options.max_internal_subpix_displacement = config.max_internal_subpix_displacement;
  options.ignore_image_evidence_min_quality =
      config.ignore_image_evidence_min_quality;
  options.force_internal_seed_from_prediction =
      config.force_internal_seed_from_prediction;
  options.outer_detector_config = config.outer_detector_config;
  return options;
}

std::string JoinIndices(const std::vector<FrozenRound2BaselineFrameSource>& frames) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < frames.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << frames[index].frame_index;
  }
  return stream.str();
}

void AppendVisibleBoardId(int board_id, std::vector<int>* visible_board_ids) {
  if (visible_board_ids == nullptr || board_id < 0) {
    return;
  }
  if (std::find(visible_board_ids->begin(), visible_board_ids->end(), board_id) ==
      visible_board_ids->end()) {
    visible_board_ids->push_back(board_id);
  }
}

std::array<Eigen::Vector3d, 4> BuildOuterCornerTargets(const ApriltagInternalConfig& config,
                                                       int board_id) {
  const ApriltagCanonicalModel model(BoardConfigForId(config, board_id));
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

Eigen::Isometry3d PoseFromCv(const cv::Mat& rvec, const cv::Mat& tvec) {
  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);

  cv::Mat rotation64;
  cv::Mat tvec64;
  rotation_cv.convertTo(rotation64, CV_64F);
  tvec.convertTo(tvec64, CV_64F);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      pose.linear()(row, col) = rotation64.at<double>(row, col);
    }
    pose.translation()[row] = tvec64.at<double>(row, 0);
  }
  return pose;
}

bool EvaluatePoseRmse(
    const DoubleSphereCameraModel& camera_model,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const Eigen::Isometry3d& pose,
    double* rmse) {
  if (rmse == nullptr ||
      object_points.size() != image_points.size() ||
      object_points.empty()) {
    return false;
  }

  double squared_error_sum = 0.0;
  for (std::size_t index = 0; index < object_points.size(); ++index) {
    Eigen::Vector2d projected;
    if (!camera_model.vsEuclideanToKeypoint(pose * object_points[index], &projected)) {
      squared_error_sum += 1e12;
      continue;
    }
    const Eigen::Vector2d observed(image_points[index].x, image_points[index].y);
    squared_error_sum += (projected - observed).squaredNorm();
  }
  *rmse = std::sqrt(squared_error_sum / static_cast<double>(object_points.size()));
  return true;
}

bool TryWideFovOuterPoseRescue(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>& image_points,
    Eigen::Isometry3d* pose,
    double* rmse) {
  if (pose == nullptr || rmse == nullptr ||
      object_points.size() != image_points.size() ||
      object_points.size() < 4 || !camera.IsValid()) {
    return false;
  }

  const DoubleSphereCameraModel camera_model =
      DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(camera));

  std::vector<cv::Point3f> filtered_object_points;
  std::vector<cv::Point2f> normalized_points;
  filtered_object_points.reserve(object_points.size());
  normalized_points.reserve(image_points.size());
  for (std::size_t index = 0; index < image_points.size(); ++index) {
    Eigen::Vector3d ray;
    if (!camera_model.keypointToEuclidean(
            Eigen::Vector2d(image_points[index].x, image_points[index].y), &ray)) {
      continue;
    }
    const Eigen::Vector3d direction = ray.normalized();
    if (direction.z() <= std::cos(kWideFovPoseRescueMaxRayAngleRadians)) {
      continue;
    }
    filtered_object_points.emplace_back(
        static_cast<float>(object_points[index].x()),
        static_cast<float>(object_points[index].y()),
        static_cast<float>(object_points[index].z()));
    normalized_points.emplace_back(
        static_cast<float>(direction.x() / direction.z()),
        static_cast<float>(direction.y() / direction.z()));
  }

  if (filtered_object_points.size() < 4) {
    return false;
  }

  const cv::Mat identity_camera = cv::Mat::eye(3, 3, CV_64F);
  const cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
  Eigen::Isometry3d best_pose = Eigen::Isometry3d::Identity();
  double best_rmse = std::numeric_limits<double>::infinity();

  auto evaluate_candidate = [&](const cv::Mat& candidate_rvec,
                                const cv::Mat& candidate_tvec) {
    if (candidate_rvec.empty() || candidate_tvec.empty()) {
      return;
    }
    cv::Mat candidate_tvec64;
    candidate_tvec.convertTo(candidate_tvec64, CV_64F);
    if (candidate_tvec64.at<double>(2, 0) <= 0.0) {
      return;
    }
    const Eigen::Isometry3d candidate_pose = PoseFromCv(candidate_rvec, candidate_tvec);
    double candidate_rmse = 0.0;
    if (!EvaluatePoseRmse(camera_model, object_points, image_points,
                          candidate_pose, &candidate_rmse)) {
      return;
    }
    if (candidate_rmse < best_rmse) {
      best_rmse = candidate_rmse;
      best_pose = candidate_pose;
    }
  };

  auto refine_and_evaluate_candidate = [&](const cv::Mat& seed_rvec,
                                           const cv::Mat& seed_tvec) {
    evaluate_candidate(seed_rvec, seed_tvec);
    cv::Mat refined_rvec = seed_rvec.clone();
    cv::Mat refined_tvec = seed_tvec.clone();
    if (cv::solvePnP(filtered_object_points, normalized_points, identity_camera,
                     dist_coeffs, refined_rvec, refined_tvec, true,
                     cv::SOLVEPNP_ITERATIVE)) {
      evaluate_candidate(refined_rvec, refined_tvec);
    }
  };

  if (filtered_object_points.size() == 4) {
    std::vector<cv::Mat> ippe_rvecs;
    std::vector<cv::Mat> ippe_tvecs;
    cv::solvePnPGeneric(filtered_object_points, normalized_points, identity_camera,
                        dist_coeffs, ippe_rvecs, ippe_tvecs, false,
                        cv::SOLVEPNP_IPPE);
    for (std::size_t index = 0; index < ippe_rvecs.size(); ++index) {
      refine_and_evaluate_candidate(ippe_rvecs[index], ippe_tvecs[index]);
    }
  }

  cv::Mat iterative_rvec;
  cv::Mat iterative_tvec;
  if (cv::solvePnP(filtered_object_points, normalized_points, identity_camera,
                   dist_coeffs, iterative_rvec, iterative_tvec, false,
                   cv::SOLVEPNP_ITERATIVE)) {
    refine_and_evaluate_candidate(iterative_rvec, iterative_tvec);
  }

  if (!std::isfinite(best_rmse)) {
    return false;
  }

  *pose = best_pose;
  *rmse = best_rmse;
  return true;
}

bool EstimatePoseForBenchmarkRefit(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<Eigen::Vector3d>& outer_targets,
    const std::vector<cv::Point2f>& outer_pixels,
    Eigen::Isometry3d* pose,
    double* rmse) {
  if (pose == nullptr || rmse == nullptr) {
    return false;
  }

  Eigen::Isometry3d standard_pose = Eigen::Isometry3d::Identity();
  double standard_rmse = std::numeric_limits<double>::infinity();
  const bool standard_success =
      EstimatePoseFromObjectPoints(camera, outer_targets, outer_pixels,
                                   &standard_pose, &standard_rmse);
  if (standard_success && standard_rmse <= kWideFovPoseRescueTriggerRmse) {
    *pose = standard_pose;
    *rmse = standard_rmse;
    return true;
  }

  Eigen::Isometry3d rescue_pose = Eigen::Isometry3d::Identity();
  double rescue_rmse = std::numeric_limits<double>::infinity();
  const bool rescue_success =
      TryWideFovOuterPoseRescue(camera, outer_targets, outer_pixels,
                                &rescue_pose, &rescue_rmse);
  if (rescue_success && (!standard_success || rescue_rmse < standard_rmse)) {
    *pose = rescue_pose;
    *rmse = rescue_rmse;
    return true;
  }

  if (standard_success) {
    *pose = standard_pose;
    *rmse = standard_rmse;
    return true;
  }
  return false;
}

bool IsOuterPoint(const CalibrationEvaluationPointObservation& point) {
  return point.point_type == JointPointType::Outer;
}

bool IsInternalPoint(const CalibrationEvaluationPointObservation& point) {
  return point.point_type == JointPointType::Internal;
}

cv::Rect ClampRectToImage(const cv::Rect& rect, const cv::Size& image_size) {
  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  return rect & image_rect;
}

const CameraModelRefitFrameDiagnostics* FindFrameDiagnostics(
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index) {
  for (const CameraModelRefitFrameDiagnostics& frame : evaluation.frame_diagnostics) {
    if (frame.frame_index == frame_index) {
      return &frame;
    }
  }
  return nullptr;
}

const CameraModelRefitBoardObservationDiagnostics* FindBoardDiagnostics(
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index,
    int board_id) {
  for (const CameraModelRefitBoardObservationDiagnostics& board :
       evaluation.board_observation_diagnostics) {
    if (board.frame_index == frame_index && board.board_id == board_id) {
      return &board;
    }
  }
  return nullptr;
}

double ComputeOuterRmseForFrame(const CameraModelRefitEvaluationResult& evaluation,
                                int frame_index) {
  double squared_error_sum = 0.0;
  int point_count = 0;
  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index || point.point_type != JointPointType::Outer) {
      continue;
    }
    squared_error_sum += point.residual_xy.squaredNorm();
    ++point_count;
  }
  if (point_count <= 0) {
    return 0.0;
  }
  return std::sqrt(squared_error_sum / static_cast<double>(point_count));
}

bool IsFiniteImagePoint(const Eigen::Vector2d& point) {
  return std::isfinite(point.x()) && std::isfinite(point.y());
}

const JointSceneFrameState* FindSceneFrameState(
    const CalibrationStateBundle& bundle,
    int frame_index) {
  for (const JointSceneFrameState& frame_state : bundle.scene_state.frames) {
    if (frame_state.frame_index == frame_index) {
      return &frame_state;
    }
  }
  return nullptr;
}

const JointSceneBoardState* FindSceneBoardState(
    const CalibrationStateBundle& bundle,
    int board_id) {
  for (const JointSceneBoardState& board_state : bundle.scene_state.boards) {
    if (board_state.board_id == board_id) {
      return &board_state;
    }
  }
  return nullptr;
}

bool IsBoardObservationUsedInBackend(
    const CalibrationBackendProblemInput& backend_problem_input,
    int frame_index,
    int board_id) {
  const std::pair<int, int> key(frame_index, board_id);
  return backend_problem_input.measurement_dataset
             .accepted_board_observation_keys.find(key) !=
         backend_problem_input.measurement_dataset
             .accepted_board_observation_keys.end();
}

double ComputeRotationAngleDeg(const Eigen::Matrix3d& rotation) {
  const Eigen::AngleAxisd angle_axis(rotation);
  return std::abs(angle_axis.angle()) * 180.0 / M_PI;
}

const CalibrationEvaluationBoardObservation* FindDatasetBoardObservation(
    const CalibrationEvaluationDataset& dataset,
    int frame_index,
    int board_id) {
  for (const CalibrationEvaluationFrameInput& frame : dataset.frames) {
    if (frame.frame_index != frame_index) {
      continue;
    }
    for (const CalibrationEvaluationBoardObservation& board :
         frame.board_observations) {
      if (board.board_id == board_id) {
        return &board;
      }
    }
  }
  return nullptr;
}

void DrawObservedPredictedPoint(cv::Mat* image,
                                const CameraModelRefitPointDiagnostics& point,
                                const cv::Scalar& observed_color,
                                int radius,
                                bool annotate_point_id) {
  if (image == nullptr || !IsFiniteImagePoint(point.observed_image_xy) ||
      !IsFiniteImagePoint(point.predicted_image_xy)) {
    return;
  }

  const cv::Point observed(static_cast<int>(std::lround(point.observed_image_xy.x())),
                           static_cast<int>(std::lround(point.observed_image_xy.y())));
  const cv::Point predicted(static_cast<int>(std::lround(point.predicted_image_xy.x())),
                            static_cast<int>(std::lround(point.predicted_image_xy.y())));
  cv::circle(*image, observed, radius, observed_color, 2, cv::LINE_AA);
  cv::drawMarker(*image, predicted, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10, 1,
                 cv::LINE_AA);
  cv::line(*image, observed, predicted, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
  if (annotate_point_id) {
    cv::putText(*image, std::to_string(point.point_id),
                observed + cv::Point(6, -6), cv::FONT_HERSHEY_PLAIN, 1.0,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  }
}

bool AccumulatePointBounds(const CameraModelRefitPointDiagnostics& point,
                           double* min_x,
                           double* min_y,
                           double* max_x,
                           double* max_y) {
  if (min_x == nullptr || min_y == nullptr || max_x == nullptr || max_y == nullptr ||
      !IsFiniteImagePoint(point.observed_image_xy) ||
      !IsFiniteImagePoint(point.predicted_image_xy)) {
    return false;
  }

  *min_x = std::min(*min_x, std::min(point.observed_image_xy.x(), point.predicted_image_xy.x()));
  *min_y = std::min(*min_y, std::min(point.observed_image_xy.y(), point.predicted_image_xy.y()));
  *max_x = std::max(*max_x, std::max(point.observed_image_xy.x(), point.predicted_image_xy.x()));
  *max_y = std::max(*max_y, std::max(point.observed_image_xy.y(), point.predicted_image_xy.y()));
  return true;
}

}  // namespace

Stage5Benchmark::Stage5Benchmark(CalibrationBenchmarkSplitOptions split_options)
    : split_options_(std::move(split_options)) {}

CalibrationBenchmarkSplit Stage5Benchmark::BuildDeterministicSplit(
    const std::vector<FrozenRound2BaselineFrameSource>& frames) const {
  CalibrationBenchmarkSplit split;
  split.mode = split_options_.mode;
  split.holdout_stride = split_options_.holdout_stride;
  split.holdout_offset = split_options_.holdout_offset;
  split.holdout_ratio = split_options_.holdout_ratio;
  split.random_seed = split_options_.random_seed;

  if (frames.empty()) {
    split.failure_reason = "Stage 5 benchmark split requires non-empty frame sources.";
    return split;
  }

  const std::string mode =
      split_options_.mode.empty() ? "deterministic_stride" : split_options_.mode;
  if (split_options_.all_frames_training ||
      mode == "all_frames_training_no_holdout") {
    split.training_frames = frames;
    split.holdout_frames = frames;
    split.mode = "all_frames_training_no_holdout";
    split.holdout_ratio = 0.0;
    split.random_seed = 0;
    split.split_signature = "all_frames_training_no_holdout_frame_count_" +
                            std::to_string(frames.size());
  } else if (mode == "random_holdout_ratio" || mode == "random_ratio" ||
      mode == "random_70_30") {
    if (!(split_options_.holdout_ratio > 0.0 &&
          split_options_.holdout_ratio < 1.0)) {
      split.failure_reason =
          "holdout_ratio must be in (0, 1) for random_holdout_ratio split.";
      return split;
    }
    if (static_cast<int>(frames.size()) <
        split_options_.minimum_training_frames +
            split_options_.minimum_holdout_frames) {
      split.failure_reason =
          "Not enough frames for requested minimum train/holdout split.";
      return split;
    }

    std::vector<std::size_t> indices(frames.size());
    std::iota(indices.begin(), indices.end(), 0u);
    std::mt19937 generator(split_options_.random_seed);
    std::shuffle(indices.begin(), indices.end(), generator);
    int holdout_count = static_cast<int>(std::lround(
        static_cast<double>(frames.size()) * split_options_.holdout_ratio));
    holdout_count = std::max(holdout_count, split_options_.minimum_holdout_frames);
    holdout_count = std::min(
        holdout_count,
        static_cast<int>(frames.size()) - split_options_.minimum_training_frames);
    if (holdout_count <= 0) {
      split.failure_reason =
          "Random split produced zero holdout frames after minimum constraints.";
      return split;
    }

    std::set<std::size_t> holdout_indices;
    for (int index = 0; index < holdout_count; ++index) {
      holdout_indices.insert(indices[static_cast<std::size_t>(index)]);
    }
    for (std::size_t index = 0; index < frames.size(); ++index) {
      if (holdout_indices.find(index) != holdout_indices.end()) {
        split.holdout_frames.push_back(frames[index]);
      } else {
        split.training_frames.push_back(frames[index]);
      }
    }
    split.mode = "random_holdout_ratio";
    split.split_signature =
        "random_holdout_ratio_" + std::to_string(split_options_.holdout_ratio) +
        "_seed_" + std::to_string(split_options_.random_seed) +
        "_holdout_indices_" + JoinIndices(split.holdout_frames);
  } else if (mode == "deterministic_stride" || mode == "stride") {
    if (split_options_.holdout_stride <= 1) {
      split.failure_reason =
          "holdout_stride must be greater than 1 for deterministic split.";
      return split;
    }

    for (std::size_t index = 0; index < frames.size(); ++index) {
      const int normalized_offset =
          ((split_options_.holdout_offset % split_options_.holdout_stride) +
           split_options_.holdout_stride) %
          split_options_.holdout_stride;
      const bool is_holdout =
          static_cast<int>((index + split_options_.holdout_stride - normalized_offset) %
                           split_options_.holdout_stride) == 0;
      if (is_holdout) {
        split.holdout_frames.push_back(frames[index]);
      } else {
        split.training_frames.push_back(frames[index]);
      }
    }

    split.split_signature =
        "deterministic_stride_" + std::to_string(split.holdout_stride) +
        "_offset_" + std::to_string(split.holdout_offset) +
        "_holdout_indices_" + JoinIndices(split.holdout_frames);
  } else {
    split.failure_reason = "Unsupported Stage 5 split mode: " + mode;
    return split;
  }

  if (static_cast<int>(split.training_frames.size()) <
      split_options_.minimum_training_frames) {
    split.failure_reason = "Training split is too small for Stage 5 benchmark.";
    return split;
  }
  if (static_cast<int>(split.holdout_frames.size()) <
      split_options_.minimum_holdout_frames) {
    split.failure_reason = "Hold-out split is too small for Stage 5 benchmark.";
    return split;
  }

  split.success = true;
  return split;
}

CalibrationBenchmarkSplit Stage5Benchmark::BuildExternalHoldoutSplit(
    const std::vector<FrozenRound2BaselineFrameSource>& training_frames,
    const std::vector<FrozenRound2BaselineFrameSource>& holdout_frames,
    const std::string& holdout_label) const {
  CalibrationBenchmarkSplit split;
  split.mode = "explicit_external_holdout";
  split.holdout_stride = 0;
  split.holdout_offset = 0;
  split.holdout_ratio = 0.0;
  split.random_seed = 0;
  split.training_frames = training_frames;
  split.holdout_frames = holdout_frames;

  if (training_frames.empty()) {
    split.failure_reason =
        "External holdout split requires non-empty training frame sources.";
    return split;
  }
  if (holdout_frames.empty()) {
    split.failure_reason =
        "External holdout split requires non-empty holdout frame sources.";
    return split;
  }
  if (static_cast<int>(training_frames.size()) < split_options_.minimum_training_frames) {
    split.failure_reason = "Training split is too small for Stage 5 benchmark.";
    return split;
  }
  if (static_cast<int>(holdout_frames.size()) < split_options_.minimum_holdout_frames) {
    split.failure_reason = "Hold-out split is too small for Stage 5 benchmark.";
    return split;
  }

  const std::string normalized_holdout_label =
      holdout_label.empty() ? "external_holdout" : holdout_label;
  split.split_signature =
      "explicit_train_" + std::to_string(training_frames.size()) +
      "_holdout_" + std::to_string(holdout_frames.size()) +
      "_holdout_label_" + normalized_holdout_label;
  split.success = true;
  return split;
}

CalibrationEvaluationDataset Stage5Benchmark::BuildTrainingEvaluationDataset(
    const CalibrationStateBundle& bundle) const {
  CalibrationEvaluationDataset dataset;
  dataset.dataset_label = bundle.scene_state.dataset_label;
  dataset.split_label = "training";
  dataset.split_signature = bundle.training_split_signature;

  for (const JointMeasurementFrameResult& frame_result :
       bundle.measurement_dataset.frames) {
    CalibrationEvaluationFrameInput frame_input;
    frame_input.frame_index = frame_result.frame_index;
    frame_input.frame_label = frame_result.frame_label;
    frame_input.visible_board_ids = frame_result.visible_board_ids;

    for (const JointBoardObservation& board_observation :
         frame_result.board_observations) {
      CalibrationEvaluationBoardObservation eval_board;
      eval_board.frame_index = frame_result.frame_index;
      eval_board.frame_label = frame_result.frame_label;
      eval_board.board_id = board_observation.board_id;

      for (const JointPointObservation& point : board_observation.points) {
        if (!point.used_in_solver) {
          continue;
        }
        CalibrationEvaluationPointObservation eval_point;
        eval_point.frame_index = point.frame_index;
        eval_point.frame_label = point.frame_label;
        eval_point.board_id = point.board_id;
        eval_point.point_id = point.point_id;
        eval_point.point_type = point.point_type;
        eval_point.image_xy = point.image_xy;
        eval_point.target_xyz_board = point.target_xyz_board;
        eval_point.quality = point.quality;
        eval_point.frame_storage_index = point.frame_storage_index;
        eval_point.source_board_observation_index = point.source_board_observation_index;
        eval_point.source_point_index = point.source_point_index;
        eval_point.source_kind = point.source_kind;
        eval_board.points.push_back(eval_point);
        if (eval_point.point_type == JointPointType::Outer) {
          ++eval_board.outer_point_count;
        } else {
          ++eval_board.internal_point_count;
        }
      }

      eval_board.has_pose_fit_outer_points = (eval_board.outer_point_count >= 4);
      if (!eval_board.points.empty()) {
        frame_input.board_observations.push_back(eval_board);
        ++dataset.board_observation_count;
        dataset.outer_point_count += eval_board.outer_point_count;
        dataset.internal_point_count += eval_board.internal_point_count;
      }
    }

    if (!frame_input.board_observations.empty()) {
      dataset.frames.push_back(frame_input);
    }
  }

  dataset.frame_count = static_cast<int>(dataset.frames.size());
  dataset.total_point_count = dataset.outer_point_count + dataset.internal_point_count;
  dataset.success = dataset.frame_count > 0 && dataset.board_observation_count > 0 &&
                    dataset.total_point_count > 0;
  if (!dataset.success) {
    dataset.failure_reason = "Training evaluation dataset is empty.";
  }
  return dataset;
}

CalibrationEvaluationDataset
Stage5Benchmark::BuildEvaluationDatasetFromMeasurementResult(
    const JointMeasurementBuildResult& measurement_result,
    const std::vector<InternalRegenerationFrameResult>& regeneration_results,
    const std::string& dataset_label,
    const std::string& split_label,
    const std::string& split_signature,
    bool include_points_not_used_in_solver) const {
  CalibrationEvaluationDataset dataset;
  dataset.dataset_label = dataset_label;
  dataset.split_label = split_label;
  dataset.split_signature = split_signature;
  dataset.internal_regeneration_results = regeneration_results;

  for (const JointMeasurementFrameResult& frame_result :
       measurement_result.frames) {
    CalibrationEvaluationFrameInput frame_input;
    frame_input.frame_index = frame_result.frame_index;
    frame_input.frame_label = frame_result.frame_label;
    frame_input.visible_board_ids = frame_result.visible_board_ids;

    for (const JointBoardObservation& board_observation :
         frame_result.board_observations) {
      CalibrationEvaluationBoardObservation eval_board;
      eval_board.frame_index = frame_result.frame_index;
      eval_board.frame_label = frame_result.frame_label;
      eval_board.board_id = board_observation.board_id;

      for (const JointPointObservation& point : board_observation.points) {
        if (!include_points_not_used_in_solver && !point.used_in_solver) {
          continue;
        }
        CalibrationEvaluationPointObservation eval_point;
        eval_point.frame_index = point.frame_index;
        eval_point.frame_label = point.frame_label;
        eval_point.board_id = point.board_id;
        eval_point.point_id = point.point_id;
        eval_point.point_type = point.point_type;
        eval_point.image_xy = point.image_xy;
        eval_point.target_xyz_board = point.target_xyz_board;
        eval_point.quality = point.quality;
        eval_point.frame_storage_index = point.frame_storage_index;
        eval_point.source_board_observation_index =
            point.source_board_observation_index;
        eval_point.source_point_index = point.source_point_index;
        eval_point.source_kind = point.source_kind;
        eval_board.points.push_back(eval_point);
        if (eval_point.point_type == JointPointType::Outer) {
          ++eval_board.outer_point_count;
        } else {
          ++eval_board.internal_point_count;
        }
      }

      eval_board.has_pose_fit_outer_points =
          (eval_board.outer_point_count >= 4);
      if (!eval_board.points.empty() && eval_board.has_pose_fit_outer_points) {
        frame_input.board_observations.push_back(eval_board);
        ++dataset.board_observation_count;
        dataset.outer_point_count += eval_board.outer_point_count;
        dataset.internal_point_count += eval_board.internal_point_count;
      }
    }

    if (!frame_input.board_observations.empty()) {
      dataset.frames.push_back(frame_input);
    }
  }

  dataset.frame_count = static_cast<int>(dataset.frames.size());
  dataset.total_point_count =
      dataset.outer_point_count + dataset.internal_point_count;
  dataset.success = dataset.frame_count > 0 &&
                    dataset.board_observation_count > 0 &&
                    dataset.total_point_count > 0;
  if (!measurement_result.success) {
    dataset.warnings.push_back(
        "Source measurement_result.success is false: " +
        measurement_result.failure_reason);
  }
  for (const std::string& warning : measurement_result.warnings) {
    dataset.warnings.push_back(warning);
  }
  if (!dataset.success) {
    dataset.failure_reason = split_label + " evaluation dataset is empty.";
  }
  return dataset;
}

CalibrationEvaluationDataset Stage5Benchmark::BuildHoldoutEvaluationDataset(
    const std::vector<FrozenRound2BaselineFrameSource>& holdout_frames,
    const FrozenRound2BaselineOptions& baseline_options,
    const JointReprojectionSceneState& optimized_scene_state,
    const std::string& split_signature,
    OuterDetectionCacheStats* cache_stats,
    InternalRegenerationCacheStats* internal_cache_stats) const {
  CalibrationEvaluationDataset dataset;
  dataset.dataset_label = baseline_options.dataset_label;
  dataset.split_label = "holdout";
  dataset.split_signature = split_signature;

  ApriltagInternalConfig config = NormalizeConfig(baseline_options.config);
  if (baseline_options.enable_camera_aware_outer_rescue &&
      optimized_scene_state.camera.IsValid() &&
      optimized_scene_state.camera.NormalizedFamilyString() == "ds-none") {
    OuterRefineCameraConfig& refine_camera =
        config.outer_detector_config.refine_camera;
    refine_camera.camera_model =
        optimized_scene_state.camera.NormalizedCameraModel();
    refine_camera.distortion_model =
        optimized_scene_state.camera.NormalizedDistortionModel();
    refine_camera.intrinsics =
        optimized_scene_state.camera.IntrinsicsVector();
    refine_camera.distortion_coeffs =
        optimized_scene_state.camera.DistortionVector();
    refine_camera.resolution = {
        optimized_scene_state.camera.resolution.width,
        optimized_scene_state.camera.resolution.height};
    config.outer_detector_config.enable_camera_aware_sphere_patch_rescue =
        true;
    config.outer_detector_config.camera_aware_sphere_patch_max_hamming =
        std::max(0, baseline_options.camera_aware_outer_rescue_max_hamming);
    config.outer_detector_config
        .camera_aware_sphere_patch_commit_mapped_corners = true;
  } else {
    config.outer_detector_config.refine_camera = OuterRefineCameraConfig{};
  }
  ApriltagInternalDetectionOptions detection_options = MakeDetectionOptions(config);
  detection_options.internal_pose_rescue_mode =
      baseline_options.internal_pose_rescue_mode;
  detection_options.internal_pose_rescue_max_ray_angle_deg =
      baseline_options.internal_pose_rescue_max_ray_angle_deg;
  detection_options.internal_pose_rescue_accept_max_outer_rmse =
      baseline_options.internal_pose_rescue_accept_max_outer_rmse;
  detection_options.ignore_image_evidence_min_quality =
      baseline_options.ignore_image_evidence_min_quality;
  detection_options.force_internal_seed_from_prediction =
      baseline_options.force_internal_seed_from_prediction;
  detection_options.bypass_internal_seed_filters =
      baseline_options.bypass_internal_seed_filters;
  detection_options.enable_geometry_prior_outer_seed =
      baseline_options.enable_geometry_prior_outer_seed;
  detection_options.geometry_prior_rescue_diagnostic_only =
      baseline_options.geometry_prior_rescue_diagnostic_only;
  detection_options.geometry_prior_rescue_use_as_observation =
      baseline_options.geometry_prior_rescue_use_as_observation;
  detection_options.geometry_prior_rescue_allow_geometry_only_pose_refit =
      baseline_options.geometry_prior_rescue_allow_geometry_only_pose_refit;
  detection_options.geometry_prior_rescue_subpix_window_radius =
      baseline_options.geometry_prior_rescue_subpix_window_radius;
  detection_options.geometry_prior_rescue_max_corner_displacement_px =
      baseline_options.geometry_prior_rescue_max_corner_displacement_px;
  detection_options.geometry_prior_rescue_min_corner_response_ratio =
      baseline_options.geometry_prior_rescue_min_corner_response_ratio;
  detection_options.geometry_prior_rescue_enable_spherical_refine =
      baseline_options.geometry_prior_rescue_enable_spherical_refine;
  detection_options.geometry_prior_rescue_edge_sample_count =
      baseline_options.geometry_prior_rescue_edge_sample_count;
  detection_options.geometry_prior_rescue_edge_search_half_width_px =
      baseline_options.geometry_prior_rescue_edge_search_half_width_px;
  detection_options.geometry_prior_rescue_min_edge_support_ratio =
      baseline_options.geometry_prior_rescue_min_edge_support_ratio;
  detection_options.geometry_prior_rescue_min_edge_gradient_ratio =
      baseline_options.geometry_prior_rescue_min_edge_gradient_ratio;
  detection_options.geometry_prior_rescue_accept_max_outer_rmse =
      baseline_options.geometry_prior_rescue_accept_max_outer_rmse;
  detection_options.geometry_prior_rescue_accept_max_rotation_error_deg =
      baseline_options.geometry_prior_rescue_accept_max_rotation_error_deg;
  detection_options.geometry_prior_rescue_accept_max_translation_error =
      baseline_options.geometry_prior_rescue_accept_max_translation_error;
  detection_options.geometry_guided_tag_likelihood_enabled =
      baseline_options.geometry_guided_tag_likelihood_enabled;
  detection_options.geometry_guided_tag_likelihood_min_visible_boards =
      baseline_options.geometry_guided_tag_likelihood_min_visible_boards;
  detection_options.geometry_guided_tag_likelihood_max_expected_hamming =
      baseline_options.geometry_guided_tag_likelihood_max_expected_hamming;
  detection_options.geometry_guided_tag_likelihood_min_hamming_margin =
      baseline_options.geometry_guided_tag_likelihood_min_hamming_margin;
  detection_options.geometry_guided_tag_likelihood_min_contrast =
      baseline_options.geometry_guided_tag_likelihood_min_contrast;
  detection_options.geometry_guided_tag_likelihood_allow_single_anchor =
      baseline_options.geometry_guided_tag_likelihood_allow_single_anchor;
  detection_options.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse =
      baseline_options.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse;
  detection_options.geometry_guided_tag_likelihood_single_anchor_max_expected_hamming =
      baseline_options.geometry_guided_tag_likelihood_single_anchor_max_expected_hamming;
  detection_options.geometry_guided_tag_likelihood_single_anchor_min_hamming_margin =
      baseline_options.geometry_guided_tag_likelihood_single_anchor_min_hamming_margin;
  detection_options.geometry_guided_tag_likelihood_single_anchor_min_contrast =
      baseline_options.geometry_guided_tag_likelihood_single_anchor_min_contrast;
  const MultiScaleOuterTagDetector outer_detector(config.outer_detector_config);
  const MultiBoardInternalMeasurementRegenerator regenerator(config, detection_options);
  const OuterDetectionCache detection_cache(
      config.outer_detector_config,
      OuterDetectionCacheOptions{baseline_options.enable_outer_detection_cache,
                                 baseline_options.outer_detection_cache_dir});
  const InternalRegenerationCache internal_regeneration_cache(
      config, detection_options,
      InternalRegenerationCacheOptions{
          baseline_options.enable_outer_detection_cache,
          baseline_options.outer_detection_cache_dir});

  for (std::size_t frame_storage_index = 0; frame_storage_index < holdout_frames.size();
       ++frame_storage_index) {
    const FrozenRound2BaselineFrameSource& frame_source = holdout_frames[frame_storage_index];
    const cv::Mat image = cv::imread(frame_source.image_path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      dataset.warnings.push_back("Failed to read hold-out image: " + frame_source.image_path);
      continue;
    }

    OuterTagMultiDetectionResult outer_detection;
    std::string cache_warning;
    OuterDetectionCacheLoadSource cache_load_source =
        OuterDetectionCacheLoadSource::None;
    if (detection_cache.Load(frame_source.image_path, &outer_detection,
                             &cache_warning, &cache_load_source)) {
      if (cache_stats != nullptr) {
        ++cache_stats->cache_hits;
        if (cache_load_source == OuterDetectionCacheLoadSource::StageLayout) {
          ++cache_stats->stage_layout_cache_hits;
        } else if (cache_load_source ==
                   OuterDetectionCacheLoadSource::LegacyLayout) {
          ++cache_stats->legacy_layout_cache_hits;
        }
      }
    } else {
      if (cache_stats != nullptr) {
        ++cache_stats->cache_misses;
      }
      if (!cache_warning.empty()) {
        if (cache_stats != nullptr) {
          ++cache_stats->load_failures;
        }
        dataset.warnings.push_back("Outer detection cache load warning: " + cache_warning);
      }
      outer_detection = outer_detector.DetectMultiple(image);
      if (detection_cache.enabled() &&
          !detection_cache.Save(frame_source.image_path, outer_detection, &cache_warning)) {
        if (cache_stats != nullptr) {
          ++cache_stats->store_failures;
        }
        if (!cache_warning.empty()) {
          dataset.warnings.push_back("Outer detection cache store warning: " + cache_warning);
        }
      }
    }
    for (const OuterTagDetectionResult& detection :
         outer_detection.detections) {
      if (detection.attempted_local_patch_rescue) {
        ++dataset.camera_aware_outer_rescue_attempted_board_count;
      }
      if (detection.used_local_patch_rescue) {
        ++dataset.camera_aware_outer_rescue_used_board_count;
      }
    }
    InternalRegenerationFrameInput regen_input;
    regen_input.frame_index = frame_source.frame_index;
    regen_input.frame_label = frame_source.frame_label;
    regen_input.outer_detections = outer_detection;
    InternalRegenerationFrameResult regen_result;
    const std::string internal_cache_state_signature =
        InternalRegenerationCache::MakeSceneStateSignature(
            optimized_scene_state);
    std::string internal_cache_warning;
    const bool internal_cache_hit = internal_regeneration_cache.Load(
        frame_source.image_path, regen_input, internal_cache_state_signature,
        &regen_result, &internal_cache_warning);
    if (!internal_cache_hit) {
      if (!internal_cache_warning.empty()) {
        dataset.warnings.push_back(
            "Internal regeneration cache load warning: " +
            internal_cache_warning);
      }
      regen_result = regenerator.RegenerateFrame(
          image, regen_input, optimized_scene_state);
      if (internal_regeneration_cache.enabled() &&
          !internal_regeneration_cache.Save(
              frame_source.image_path, regen_input,
              internal_cache_state_signature, regen_result,
              &internal_cache_warning) &&
          !internal_cache_warning.empty()) {
        dataset.warnings.push_back(
            "Internal regeneration cache store warning: " +
            internal_cache_warning);
      }
    }
    if (internal_cache_stats != nullptr) {
      *internal_cache_stats = internal_regeneration_cache.stats();
    }
    for (const std::string& warning : regen_result.warnings) {
      dataset.warnings.push_back(warning);
    }
    dataset.internal_regeneration_results.push_back(regen_result);

    CalibrationEvaluationFrameInput frame_input;
    frame_input.frame_index = frame_source.frame_index;
    frame_input.frame_label = frame_source.frame_label;

    for (std::size_t board_obs_index = 0;
         board_obs_index < outer_detection.frame_measurements.board_measurements.size();
         ++board_obs_index) {
      const OuterBoardMeasurement& outer_measurement =
          outer_detection.frame_measurements.board_measurements[board_obs_index];
      CalibrationEvaluationBoardObservation eval_board;
      eval_board.frame_index = frame_source.frame_index;
      eval_board.frame_label = frame_source.frame_label;
      eval_board.board_id = outer_measurement.board_id;

      const RegeneratedBoardMeasurement* regenerated_board = nullptr;
      for (const RegeneratedBoardMeasurement& measurement : regen_result.board_measurements) {
        if (measurement.board_id == outer_measurement.board_id) {
          regenerated_board = &measurement;
          break;
        }
      }
      if (baseline_options.strict_board_observation_acceptance &&
          outer_measurement.success &&
          (regenerated_board == nullptr ||
           !regenerated_board->detection.success)) {
        continue;
      }

      if (outer_measurement.success && outer_measurement.valid_refined_corner_count == 4) {
        const std::array<Eigen::Vector3d, 4> outer_targets =
            BuildOuterCornerTargets(config, outer_measurement.board_id);
        for (int corner_index = 0; corner_index < 4; ++corner_index) {
          if (!outer_measurement.refined_corner_valid[static_cast<std::size_t>(corner_index)]) {
            continue;
          }
          CalibrationEvaluationPointObservation point;
          point.frame_index = frame_source.frame_index;
          point.frame_label = frame_source.frame_label;
          point.board_id = outer_measurement.board_id;
          point.point_id = corner_index;
          point.point_type = JointPointType::Outer;
          point.image_xy =
              outer_measurement.refined_outer_corners_original_image[static_cast<std::size_t>(
                  corner_index)];
          point.target_xyz_board = outer_targets[static_cast<std::size_t>(corner_index)];
          point.quality = outer_measurement.detection_quality;
          point.frame_storage_index = static_cast<int>(frame_storage_index);
          point.source_board_observation_index = static_cast<int>(board_obs_index);
          point.source_point_index = corner_index;
          point.source_kind = JointObservationSourceKind::OuterMeasurement;
          eval_board.points.push_back(point);
          ++eval_board.outer_point_count;
        }
        eval_board.has_pose_fit_outer_points = (eval_board.outer_point_count == 4);
        AppendVisibleBoardId(outer_measurement.board_id, &frame_input.visible_board_ids);
      }

      if (regenerated_board != nullptr) {
        for (std::size_t corner_index = 0;
             corner_index < regenerated_board->detection.corners.size();
             ++corner_index) {
          const CornerMeasurement& corner =
              regenerated_board->detection.corners[corner_index];
          if (!corner.valid || corner.corner_type == CornerType::Outer) {
            continue;
          }
          CalibrationEvaluationPointObservation point;
          point.frame_index = frame_source.frame_index;
          point.frame_label = frame_source.frame_label;
          point.board_id = outer_measurement.board_id;
          point.point_id = corner.point_id;
          point.point_type = JointPointType::Internal;
          point.image_xy = corner.image_xy;
          point.target_xyz_board = corner.target_xyz;
          point.quality = corner.quality;
          point.frame_storage_index = static_cast<int>(frame_storage_index);
          point.source_board_observation_index = static_cast<int>(board_obs_index);
          point.source_point_index = static_cast<int>(corner_index);
          point.source_kind = JointObservationSourceKind::InternalMeasurement;
          eval_board.points.push_back(point);
          ++eval_board.internal_point_count;
        }
      }

      if (!eval_board.points.empty() && eval_board.has_pose_fit_outer_points) {
        frame_input.board_observations.push_back(eval_board);
        ++dataset.board_observation_count;
        dataset.outer_point_count += eval_board.outer_point_count;
        dataset.internal_point_count += eval_board.internal_point_count;
      }
    }

    if (!frame_input.board_observations.empty()) {
      dataset.frames.push_back(frame_input);
    }
  }

  dataset.frame_count = static_cast<int>(dataset.frames.size());
  dataset.total_point_count = dataset.outer_point_count + dataset.internal_point_count;
  dataset.success = dataset.frame_count > 0 && dataset.board_observation_count > 0 &&
                    dataset.total_point_count > 0;
  if (!dataset.success) {
    dataset.failure_reason = "Hold-out evaluation dataset is empty.";
  }
  return dataset;
}

const char* ToString(MultiBoardConsistencyPoseSource source) {
  switch (source) {
    case MultiBoardConsistencyPoseSource::OuterOnly:
      return "outer_only";
  }
  return "outer_only";
}

const char* ToString(
    TrialBackendFrameBoardSelectionOptions::SelectionMode mode) {
  switch (mode) {
    case TrialBackendFrameBoardSelectionOptions::SelectionMode::StrictRmse:
      return "strict_rmse";
    case TrialBackendFrameBoardSelectionOptions::SelectionMode::KalibrStyleBatch:
      return "kalibr_style_batch";
  }
  return "strict_rmse";
}

const char* ToString(
    TrialBackendFrameBoardSelectionOptions::BudgetMode mode) {
  switch (mode) {
    case TrialBackendFrameBoardSelectionOptions::BudgetMode::Fixed:
      return "fixed";
    case TrialBackendFrameBoardSelectionOptions::BudgetMode::Adaptive:
      return "adaptive";
    case TrialBackendFrameBoardSelectionOptions::BudgetMode::KalibrStyle:
      return "kalibr_style";
  }
  return "fixed";
}

const char* ToString(
    TrialBackendFrameBoardSelectionOptions::CandidateOrderMode mode) {
  switch (mode) {
    case TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
        ::ScoreSorted:
      return "score_sorted";
    case TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
        ::RandomShuffle:
      return "random_shuffle";
    case TrialBackendFrameBoardSelectionOptions::CandidateOrderMode
        ::IntrinsicsInformationGreedy:
      return "intrinsics_information_greedy";
  }
  return "score_sorted";
}

const char* ToString(
    TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode mode) {
  switch (mode) {
    case TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode::Legacy:
      return "legacy";
    case TrialBackendFrameBoardSelectionOptions::InfoGainProxyMode
        ::IntrinsicsJacobian:
      return "intrinsics_jacobian";
  }
  return "intrinsics_jacobian";
}

const char* ToString(
    TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity mode) {
  switch (mode) {
    case TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity
        ::FrameBoard:
      return "frame_board";
    case TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity
        ::Frame:
      return "frame";
    case TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity
        ::FrameBoardThenFrame:
      return "frame_board_then_frame";
  }
  return "frame_board";
}

MultiBoardConsistencyPoseSource ParseMultiBoardConsistencyPoseSource(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "outer_only" || lowered == "outeronly") {
    return MultiBoardConsistencyPoseSource::OuterOnly;
  }
  throw std::runtime_error(
      "Unsupported multi-board consistency pose source: " + value);
}

CameraModelRefitEvaluationResult Stage5Benchmark::EvaluateCameraModel(
    const CalibrationEvaluationDataset& dataset,
    const OuterBootstrapCameraIntrinsics& camera,
    const std::string& method_label) const {
  CameraModelRefitEvaluationResult result;
  result.method_label = method_label;
  result.split_label = dataset.split_label;
  result.split_signature = dataset.split_signature;
  result.camera = camera;
  result.uniform_control_point_mode = dataset.uniform_control_point_mode;

  if (!dataset.success) {
    result.failure_reason =
        "EvaluateCameraModel requires a successful evaluation dataset.";
    return result;
  }
  if (!camera.IsValid()) {
    result.failure_reason = "EvaluateCameraModel requires valid DS intrinsics.";
    return result;
  }

  const DoubleSphereCameraModel camera_model =
      DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(camera));
  double total_squared_error = 0.0;
  double total_angular_squared_error = 0.0;
  double outer_squared_error = 0.0;
  double internal_squared_error = 0.0;
  double total_squared_error_excluding_board = 0.0;
  double outer_squared_error_excluding_board = 0.0;
  double internal_squared_error_excluding_board = 0.0;
  int total_point_count = 0;
  int total_angular_point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  int total_point_count_excluding_board = 0;
  int outer_point_count_excluding_board = 0;
  int internal_point_count_excluding_board = 0;
  double pose_fit_outer_squared_rmse = 0.0;

  for (const CalibrationEvaluationFrameInput& frame : dataset.frames) {
    double frame_squared_error = 0.0;
    double frame_outer_squared_error = 0.0;
    double frame_internal_squared_error = 0.0;
    double frame_pose_fit_outer_squared_rmse = 0.0;
    int frame_point_count = 0;
    int frame_outer_count = 0;
    int frame_internal_count = 0;
    int frame_pose_refit_attempt_count = 0;
    int frame_pose_refit_success_count = 0;

    for (const CalibrationEvaluationBoardObservation& board : frame.board_observations) {
      CameraModelRefitBoardObservationDiagnostics board_diag;
      board_diag.method_label = method_label;
      board_diag.split_label = dataset.split_label;
      board_diag.frame_index = frame.frame_index;
      board_diag.frame_label = frame.frame_label;
      board_diag.board_id = board.board_id;

      std::vector<Eigen::Vector3d> outer_targets;
      std::vector<cv::Point2f> outer_pixels;
      std::vector<Eigen::Vector3d> all_targets;
      std::vector<cv::Point2f> all_pixels;
      all_targets.reserve(board.points.size());
      all_pixels.reserve(board.points.size());
      for (const CalibrationEvaluationPointObservation& point : board.points) {
        all_targets.push_back(point.target_xyz_board);
        all_pixels.emplace_back(static_cast<float>(point.image_xy.x()),
                                static_cast<float>(point.image_xy.y()));
        if (IsOuterPoint(point)) {
          outer_targets.push_back(point.target_xyz_board);
          outer_pixels.push_back(
              cv::Point2f(static_cast<float>(point.image_xy.x()),
                          static_cast<float>(point.image_xy.y())));
        }
      }
      const std::vector<Eigen::Vector3d>& pose_targets =
          dataset.uniform_control_point_mode ? all_targets : outer_targets;
      const std::vector<cv::Point2f>& pose_pixels =
          dataset.uniform_control_point_mode ? all_pixels : outer_pixels;
      if (pose_targets.size() < 4) {
        ++result.pose_only_refit_attempt_count;
        ++frame_pose_refit_attempt_count;
        board_diag.failure_reason = dataset.uniform_control_point_mode
                                        ? "insufficient_control_points"
                                        : "insufficient_outer_pose_fit_points";
        result.board_observation_diagnostics.push_back(board_diag);
        result.warnings.push_back(
            "Skipped board observation without enough pose-fit points: frame=" +
            frame.frame_label + " board=" + std::to_string(board.board_id));
        continue;
      }

      Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
      double pose_fit_outer_rmse = 0.0;
      ++result.pose_only_refit_attempt_count;
      ++frame_pose_refit_attempt_count;
      if (!EstimatePoseForBenchmarkRefit(camera, pose_targets, pose_pixels,
                                         &T_camera_board, &pose_fit_outer_rmse)) {
        board_diag.failure_reason = "pose_only_refit_failed";
        result.board_observation_diagnostics.push_back(board_diag);
        result.warnings.push_back(
            "Pose refit failed: frame=" + frame.frame_label + " board=" +
            std::to_string(board.board_id));
        continue;
      }
      ++result.pose_only_refit_success_count;
      ++frame_pose_refit_success_count;
      pose_fit_outer_squared_rmse +=
          pose_fit_outer_rmse * pose_fit_outer_rmse;
      frame_pose_fit_outer_squared_rmse +=
          pose_fit_outer_rmse * pose_fit_outer_rmse;
      board_diag.pose_only_refit_success = true;
      board_diag.T_camera_board = T_camera_board.matrix();
      board_diag.pose_fit_outer_rmse = pose_fit_outer_rmse;

      board_diag.all_point_pose_refit_point_count =
          static_cast<int>(all_targets.size());
      Eigen::Isometry3d T_camera_board_all_points =
          dataset.uniform_control_point_mode
              ? T_camera_board
              : Eigen::Isometry3d::Identity();
      double all_point_pose_refit_rmse = dataset.uniform_control_point_mode
                                             ? pose_fit_outer_rmse
                                             : 0.0;
      if (dataset.uniform_control_point_mode ||
          (all_targets.size() >= 4 &&
          EstimatePoseForBenchmarkRefit(camera, all_targets, all_pixels,
                                        &T_camera_board_all_points,
                                        &all_point_pose_refit_rmse))) {
        board_diag.all_point_pose_refit_success = true;
        board_diag.all_point_pose_refit_rmse = all_point_pose_refit_rmse;
        if (!dataset.uniform_control_point_mode) {
          double all_point_internal_squared_error = 0.0;
          int all_point_internal_count = 0;
          for (const CalibrationEvaluationPointObservation& point : board.points) {
            if (point.point_type != JointPointType::Internal) {
              continue;
            }
            Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
            if (!camera_model.vsEuclideanToKeypoint(
                    T_camera_board_all_points * point.target_xyz_board,
                    &predicted)) {
              continue;
            }
            all_point_internal_squared_error +=
                (predicted - point.image_xy).squaredNorm();
            ++all_point_internal_count;
          }
          board_diag.all_point_pose_refit_internal_rmse =
              all_point_internal_count > 0
                  ? std::sqrt(all_point_internal_squared_error /
                              static_cast<double>(all_point_internal_count))
                  : 0.0;
        }
      }

      double board_squared_error = 0.0;
      double board_outer_squared_error = 0.0;
      double board_internal_squared_error = 0.0;
      int board_point_count = 0;
      for (const CalibrationEvaluationPointObservation& point : board.points) {
        CameraModelRefitPointDiagnostics point_diag;
        point_diag.method_label = method_label;
        point_diag.split_label = dataset.split_label;
        point_diag.frame_index = point.frame_index;
        point_diag.frame_label = point.frame_label;
        point_diag.board_id = point.board_id;
        point_diag.point_id = point.point_id;
        point_diag.point_type = point.point_type;
        point_diag.observed_image_xy = point.image_xy;
        point_diag.target_xyz_board = point.target_xyz_board;
        point_diag.quality = point.quality;
        point_diag.frame_storage_index = point.frame_storage_index;
        point_diag.source_board_observation_index = point.source_board_observation_index;
        point_diag.source_point_index = point.source_point_index;
        point_diag.source_kind = point.source_kind;

        Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
        double squared_error = 0.0;
        if (!camera_model.vsEuclideanToKeypoint(T_camera_board * point.target_xyz_board,
                                                &predicted)) {
          point_diag.predicted_image_xy = Eigen::Vector2d::Constant(
              std::numeric_limits<double>::quiet_NaN());
          point_diag.residual_xy = Eigen::Vector2d::Constant(kInvalidProjectionPenaltyPixels);
          point_diag.residual_norm = std::sqrt(2.0) * kInvalidProjectionPenaltyPixels;
          squared_error = 2.0 * kInvalidProjectionPenaltyPixels *
                          kInvalidProjectionPenaltyPixels;
        } else {
          point_diag.predicted_image_xy = predicted;
          point_diag.residual_xy = predicted - point.image_xy;
          point_diag.residual_norm = point_diag.residual_xy.norm();
          squared_error = point_diag.residual_xy.squaredNorm();

          AngularObservationGeometry observation_geometry;
          AngularPredictionGeometry prediction_geometry;
          if (ComputeAngularObservationGeometry(camera_model, point.image_xy,
                                                &observation_geometry) &&
              ComputeAngularPredictionGeometryFromPoint(
                  T_camera_board * point.target_xyz_board, predicted,
                  &prediction_geometry)) {
            const Eigen::Vector2d angular_residual =
                ComputeAngularResidualTangent(observation_geometry,
                                              prediction_geometry);
            if (angular_residual.allFinite()) {
              total_angular_squared_error += angular_residual.squaredNorm();
              ++total_angular_point_count;
            }
          }
        }

        result.point_diagnostics.push_back(point_diag);
        board_squared_error += squared_error;
        ++board_point_count;
        if (point.board_id != result.excluded_board_id_for_rmse) {
          total_squared_error_excluding_board += squared_error;
          ++total_point_count_excluding_board;
        }
        if (dataset.uniform_control_point_mode) {
          // Checkerboard control points are one homogeneous measurement set.
        } else if (point.point_type == JointPointType::Outer) {
          outer_squared_error += squared_error;
          board_outer_squared_error += squared_error;
          frame_outer_squared_error += squared_error;
          ++outer_point_count;
          ++board_diag.outer_point_count;
          ++frame_outer_count;
          if (point.board_id != result.excluded_board_id_for_rmse) {
            outer_squared_error_excluding_board += squared_error;
            ++outer_point_count_excluding_board;
          }
        } else {
          internal_squared_error += squared_error;
          board_internal_squared_error += squared_error;
          frame_internal_squared_error += squared_error;
          ++internal_point_count;
          ++board_diag.internal_point_count;
          ++frame_internal_count;
          if (point.board_id != result.excluded_board_id_for_rmse) {
            internal_squared_error_excluding_board += squared_error;
            ++internal_point_count_excluding_board;
          }
        }
      }

      if (board_point_count <= 0) {
        continue;
      }
      board_diag.point_count = board_point_count;
      board_diag.evaluation_rmse =
          std::sqrt(board_squared_error / static_cast<double>(board_point_count));
      board_diag.outer_evaluation_rmse =
          board_diag.outer_point_count > 0
              ? std::sqrt(board_outer_squared_error /
                          static_cast<double>(board_diag.outer_point_count))
              : 0.0;
      board_diag.internal_evaluation_rmse =
          board_diag.internal_point_count > 0
              ? std::sqrt(board_internal_squared_error /
                          static_cast<double>(board_diag.internal_point_count))
              : 0.0;
      result.board_observation_diagnostics.push_back(board_diag);
      total_squared_error += board_squared_error;
      total_point_count += board_point_count;
      frame_squared_error += board_squared_error;
      frame_point_count += board_point_count;
    }

    if (frame_point_count > 0) {
      CameraModelRefitFrameDiagnostics frame_diag;
      frame_diag.method_label = method_label;
      frame_diag.split_label = dataset.split_label;
      frame_diag.frame_index = frame.frame_index;
      frame_diag.frame_label = frame.frame_label;
      frame_diag.pose_only_refit_attempt_count = frame_pose_refit_attempt_count;
      frame_diag.pose_only_refit_success_count = frame_pose_refit_success_count;
      frame_diag.pose_only_refit_success_rate =
          frame_pose_refit_attempt_count > 0
              ? static_cast<double>(frame_pose_refit_success_count) /
                    static_cast<double>(frame_pose_refit_attempt_count)
              : 0.0;
      frame_diag.pose_only_refit_rmse =
          frame_pose_refit_success_count > 0
              ? std::sqrt(frame_pose_fit_outer_squared_rmse /
                          static_cast<double>(frame_pose_refit_success_count))
              : 0.0;
      frame_diag.point_count = frame_point_count;
      frame_diag.outer_point_count = frame_outer_count;
      frame_diag.internal_point_count = frame_internal_count;
      frame_diag.rmse =
          std::sqrt(frame_squared_error / static_cast<double>(frame_point_count));
      frame_diag.outer_rmse =
          frame_outer_count > 0
              ? std::sqrt(frame_outer_squared_error /
                          static_cast<double>(frame_outer_count))
              : 0.0;
      frame_diag.internal_rmse =
          frame_internal_count > 0
              ? std::sqrt(frame_internal_squared_error /
                          static_cast<double>(frame_internal_count))
              : 0.0;
      result.frame_diagnostics.push_back(frame_diag);
    }
  }

  result.evaluated_frame_count = static_cast<int>(result.frame_diagnostics.size());
  result.evaluated_board_observation_count =
      static_cast<int>(result.board_observation_diagnostics.size());
  result.pose_only_refit_success_rate =
      result.pose_only_refit_attempt_count > 0
          ? static_cast<double>(result.pose_only_refit_success_count) /
                static_cast<double>(result.pose_only_refit_attempt_count)
          : 0.0;
  result.pose_only_refit_rmse =
      result.pose_only_refit_success_count > 0
          ? std::sqrt(pose_fit_outer_squared_rmse /
                      static_cast<double>(result.pose_only_refit_success_count))
          : 0.0;
  result.point_count = total_point_count;
  result.angular_point_count = total_angular_point_count;
  result.outer_point_count = outer_point_count;
  result.internal_point_count = internal_point_count;
  result.point_count_excluding_board = total_point_count_excluding_board;
  result.outer_point_count_excluding_board = outer_point_count_excluding_board;
  result.internal_point_count_excluding_board = internal_point_count_excluding_board;
  if (total_point_count <= 0) {
    result.failure_reason = "Camera-only refit evaluation produced zero valid points.";
    return result;
  }

  result.overall_rmse =
      std::sqrt(total_squared_error / static_cast<double>(total_point_count));
  result.overall_angular_rmse_rad =
      total_angular_point_count > 0
          ? std::sqrt(total_angular_squared_error /
                      static_cast<double>(total_angular_point_count))
          : 0.0;
  result.overall_angular_rmse_deg =
      result.overall_angular_rmse_rad * kRadiansToDegrees;
  std::vector<double> residual_norms;
  residual_norms.reserve(result.point_diagnostics.size());
  for (const CameraModelRefitPointDiagnostics& point :
       result.point_diagnostics) {
    if (std::isfinite(point.residual_norm)) {
      residual_norms.push_back(point.residual_norm);
    }
  }
  result.p95_reprojection_error = ComputePercentile(residual_norms, 0.95);
  result.outer_only_rmse =
      outer_point_count > 0
          ? std::sqrt(outer_squared_error / static_cast<double>(outer_point_count))
          : 0.0;
  result.internal_only_rmse =
      internal_point_count > 0
          ? std::sqrt(internal_squared_error / static_cast<double>(internal_point_count))
          : 0.0;
  result.overall_rmse_excluding_board =
      total_point_count_excluding_board > 0
          ? std::sqrt(total_squared_error_excluding_board /
                      static_cast<double>(total_point_count_excluding_board))
          : 0.0;
  result.outer_only_rmse_excluding_board =
      outer_point_count_excluding_board > 0
          ? std::sqrt(outer_squared_error_excluding_board /
                      static_cast<double>(outer_point_count_excluding_board))
          : 0.0;
  result.internal_only_rmse_excluding_board =
      internal_point_count_excluding_board > 0
          ? std::sqrt(internal_squared_error_excluding_board /
                      static_cast<double>(internal_point_count_excluding_board))
          : 0.0;
  ComputeKalibrStyleResidualStatistics(
      result.point_diagnostics,
      &result.mean_residual_x,
      &result.mean_residual_y,
      &result.std_residual_x,
      &result.std_residual_y);
  result.success = true;
  return result;
}

MultiBoardPoseOrientationEvaluationResult
Stage5Benchmark::EvaluateMultiBoardPoseOrientation(
    const CalibrationEvaluationDataset& dataset,
    const CalibrationSceneState& final_scene,
    const CalibrationSceneState& ground_truth_scene) const {
  MultiBoardPoseOrientationEvaluationResult result;
  if (!dataset.success || !final_scene.IsValid() ||
      !ground_truth_scene.IsValid()) {
    result.failure_reason =
        "Multiboard pose orientation evaluation requires valid datasets and scenes.";
    return result;
  }

  for (const CalibrationEvaluationFrameInput& frame : dataset.frames) {
    // Imported holdout frames are offset by the Stage5 interchange loader to
    // avoid collisions with training ids. Their GT scene uses source ids.
    const int ground_truth_frame_index =
        frame.frame_index >= 1000000 ? frame.frame_index - 1000000
                                     : frame.frame_index;
    const JointSceneFrameState* gt_frame = nullptr;
    for (const JointSceneFrameState& candidate : ground_truth_scene.frames) {
      if (candidate.frame_index == ground_truth_frame_index &&
          candidate.initialized) {
        gt_frame = &candidate;
        break;
      }
    }
    if (gt_frame == nullptr) {
      continue;
    }
    ++result.evaluated_frame_count;
    std::vector<Eigen::Vector3d> reference_targets;
    std::vector<cv::Point2f> outer_pixels;
    for (const CalibrationEvaluationBoardObservation& observation :
         frame.board_observations) {
      const JointSceneBoardState* board = nullptr;
      for (const JointSceneBoardState& candidate : final_scene.boards) {
        if (candidate.board_id == observation.board_id && candidate.initialized) {
          board = &candidate;
          break;
        }
      }
      if (board == nullptr) {
        continue;
      }
      const Eigen::Isometry3d T_reference_board(
          board->T_reference_board);
      for (const CalibrationEvaluationPointObservation& point :
           observation.points) {
        if (point.point_type != JointPointType::Outer) {
          continue;
        }
        reference_targets.push_back(
            T_reference_board * point.target_xyz_board);
        outer_pixels.emplace_back(static_cast<float>(point.image_xy.x()),
                                  static_cast<float>(point.image_xy.y()));
      }
    }
    if (reference_targets.size() < 4u) {
      continue;
    }
    Eigen::Isometry3d T_camera_reference = Eigen::Isometry3d::Identity();
    double pose_rmse = 0.0;
    if (!EstimatePoseForBenchmarkRefit(final_scene.camera, reference_targets,
                                       outer_pixels, &T_camera_reference,
                                       &pose_rmse)) {
      continue;
    }
    const Eigen::Matrix3d R_delta =
        Eigen::Isometry3d(gt_frame->T_camera_reference).rotation().transpose() *
        T_camera_reference.rotation();
    const double cosine = std::max(
        -1.0, std::min(1.0, 0.5 * (R_delta.trace() - 1.0)));
    constexpr double kDegreesPerRadian = 57.2957795130823208768;
    result.orientation_errors_deg.push_back(
        std::acos(cosine) * kDegreesPerRadian);
    ++result.pose_success_count;
  }
  result.pose_success_rate = result.evaluated_frame_count > 0
      ? static_cast<double>(result.pose_success_count) /
            static_cast<double>(result.evaluated_frame_count)
      : 0.0;
  if (result.orientation_errors_deg.empty()) {
    result.failure_reason = "No pose-evaluation frame could be refit.";
    return result;
  }
  result.orientation_median_deg =
      ComputePercentile(result.orientation_errors_deg, 0.50);
  result.orientation_p95_deg =
      ComputePercentile(result.orientation_errors_deg, 0.95);
  result.success = true;
  return result;
}

MultiBoardConsistencyDiagnosticsResult
Stage5Benchmark::EvaluateMultiBoardConsistency(
    const CalibrationEvaluationDataset& dataset,
    const CameraModelRefitEvaluationResult& evaluation,
    const CalibrationBackendProblemInput& backend_problem_input,
    const CalibrationStateBundle& final_bundle,
    const MultiBoardConsistencyDiagnosticsOptions& options) const {
  MultiBoardConsistencyDiagnosticsResult result;
  result.pose_source_label = ToString(options.pose_source);
  if (!options.enabled) {
    result.failure_reason = "disabled";
    return result;
  }
  if (!dataset.success) {
    result.failure_reason =
        "Multi-board consistency diagnostics require a successful dataset.";
    return result;
  }
  if (!evaluation.success) {
    result.failure_reason =
        "Multi-board consistency diagnostics require a successful evaluation.";
    return result;
  }
  if (options.pose_source != MultiBoardConsistencyPoseSource::OuterOnly) {
    result.failure_reason = "Only outer_only pose source is supported in Phase 3.";
    return result;
  }

  const DoubleSphereCameraModel camera_model =
      DoubleSphereCameraModel::FromConfig(
          MakeIntermediateCameraConfig(final_bundle.scene_state.camera));

  for (const CalibrationEvaluationFrameInput& frame : dataset.frames) {
    const JointSceneFrameState* frame_state =
        FindSceneFrameState(final_bundle, frame.frame_index);
    if (frame_state == nullptr || !frame_state->initialized) {
      result.warnings.push_back(
          "Missing initialized training frame pose for frame=" +
          std::to_string(frame.frame_index));
      continue;
    }

    for (const CalibrationEvaluationBoardObservation& board :
         frame.board_observations) {
      MultiBoardConsistencyObservationDiagnostics row;
      row.frame_id = frame.frame_index;
      row.frame_label = frame.frame_label;
      row.board_id = board.board_id;
      row.used_in_backend = IsBoardObservationUsedInBackend(
          backend_problem_input, frame.frame_index, board.board_id);
      row.outer_point_count = board.outer_point_count;
      row.internal_point_count = board.internal_point_count;

      const CameraModelRefitBoardObservationDiagnostics* board_diag =
          FindBoardDiagnostics(evaluation, frame.frame_index, board.board_id);
      if (board_diag != nullptr) {
        row.local_outer_rmse = board_diag->pose_fit_outer_rmse;
        row.global_reprojection_rmse = board_diag->evaluation_rmse;
        row.residual_rmse = board_diag->evaluation_rmse;
        row.outer_rmse = board_diag->outer_evaluation_rmse;
        row.internal_rmse = board_diag->internal_evaluation_rmse;
      }

      double polar_sum = 0.0;
      int polar_count = 0;
      for (const CalibrationEvaluationPointObservation& point : board.points) {
        const double polar_deg =
            ComputePolarAngleDeg(camera_model, point.image_xy);
        if (!std::isfinite(polar_deg)) {
          continue;
        }
        polar_sum += polar_deg;
        ++polar_count;
        row.polar_angle_max_deg =
            std::max(row.polar_angle_max_deg, polar_deg);
      }
      row.polar_angle_mean_deg =
          polar_count > 0 ? polar_sum / static_cast<double>(polar_count) : 0.0;

      std::vector<Eigen::Vector3d> outer_targets;
      std::vector<cv::Point2f> outer_pixels;
      for (const CalibrationEvaluationPointObservation& point : board.points) {
        if (!IsOuterPoint(point)) {
          continue;
        }
        outer_targets.push_back(point.target_xyz_board);
        outer_pixels.emplace_back(
            static_cast<float>(point.image_xy.x()),
            static_cast<float>(point.image_xy.y()));
      }

      if (static_cast<int>(outer_targets.size()) < options.min_outer_points) {
        row.failure_reason = "insufficient_outer_points";
        result.observation_diagnostics.push_back(row);
        continue;
      }

      const JointSceneBoardState* board_state =
          FindSceneBoardState(final_bundle, board.board_id);
      if (board_state == nullptr || !board_state->initialized) {
        row.failure_reason = "board_not_initialized";
        result.observation_diagnostics.push_back(row);
        continue;
      }

      Eigen::Isometry3d T_camera_board_local = Eigen::Isometry3d::Identity();
      double local_outer_rmse = 0.0;
      if (!EstimatePoseForBenchmarkRefit(final_bundle.scene_state.camera,
                                         outer_targets,
                                         outer_pixels,
                                         &T_camera_board_local,
                                         &local_outer_rmse)) {
        row.failure_reason = "local_pose_refit_failed";
        result.observation_diagnostics.push_back(row);
        continue;
      }

      row.local_pose_refit_success = true;
      row.local_outer_rmse = local_outer_rmse;
      const Eigen::Isometry3d T_camera_reference =
          Eigen::Isometry3d(frame_state->T_camera_reference);
      const Eigen::Isometry3d T_reference_board_global =
          Eigen::Isometry3d(board_state->T_reference_board);
      const Eigen::Isometry3d T_reference_board_obs =
          T_camera_reference.inverse() * T_camera_board_local;
      const Eigen::Isometry3d delta =
          T_reference_board_global.inverse() * T_reference_board_obs;
      row.translation_error_mm = delta.translation().norm() * 1000.0;
      row.rotation_error_deg = ComputeRotationAngleDeg(delta.rotation());
      result.observation_diagnostics.push_back(row);
      ++result.successful_local_pose_refit_count;
    }
  }

  result.frame_count = static_cast<int>(dataset.frames.size());
  result.board_observation_count =
      static_cast<int>(result.observation_diagnostics.size());

  std::map<int, std::vector<const MultiBoardConsistencyObservationDiagnostics*> >
      by_board;
  std::map<int, std::vector<const MultiBoardConsistencyObservationDiagnostics*> >
      by_frame;
  for (const MultiBoardConsistencyObservationDiagnostics& row :
       result.observation_diagnostics) {
    by_frame[row.frame_id].push_back(&row);
    if (row.local_pose_refit_success) {
      by_board[row.board_id].push_back(&row);
    }
  }

  for (const auto& entry : by_board) {
    MultiBoardConsistencyBoardDiagnostics board_diag;
    board_diag.board_id = entry.first;
    board_diag.support_observation_count =
        static_cast<int>(entry.second.size());
    std::vector<double> translation_errors;
    std::vector<double> rotation_errors;
    translation_errors.reserve(entry.second.size());
    rotation_errors.reserve(entry.second.size());
    for (const MultiBoardConsistencyObservationDiagnostics* row : entry.second) {
      translation_errors.push_back(row->translation_error_mm);
      rotation_errors.push_back(row->rotation_error_deg);
      if (row->translation_error_mm >= board_diag.max_translation_error_mm) {
        board_diag.max_translation_error_mm = row->translation_error_mm;
        board_diag.worst_frame_id = row->frame_id;
      }
      board_diag.max_rotation_error_deg =
          std::max(board_diag.max_rotation_error_deg, row->rotation_error_deg);
    }
    if (!translation_errors.empty()) {
      board_diag.mean_translation_error_mm =
          std::accumulate(translation_errors.begin(), translation_errors.end(), 0.0) /
          static_cast<double>(translation_errors.size());
      board_diag.median_translation_error_mm =
          ComputeMedian(translation_errors);
      board_diag.p90_translation_error_mm =
          ComputePercentile(translation_errors, 0.90);
      board_diag.mean_rotation_error_deg =
          std::accumulate(rotation_errors.begin(), rotation_errors.end(), 0.0) /
          static_cast<double>(rotation_errors.size());
      board_diag.median_rotation_error_deg =
          ComputeMedian(rotation_errors);
      board_diag.p90_rotation_error_deg =
          ComputePercentile(rotation_errors, 0.90);
    }
    result.board_diagnostics.push_back(board_diag);
  }

  for (const auto& entry : by_frame) {
    MultiBoardConsistencyFrameDiagnostics frame_diag;
    frame_diag.frame_id = entry.first;
    const CameraModelRefitFrameDiagnostics* existing_frame =
        FindFrameDiagnostics(evaluation, entry.first);
    if (existing_frame != nullptr) {
      frame_diag.frame_reprojection_rmse = existing_frame->rmse;
    }

    std::vector<double> translation_errors;
    std::vector<double> rotation_errors;
    for (const MultiBoardConsistencyObservationDiagnostics* row : entry.second) {
      if (!row->local_pose_refit_success) {
        continue;
      }
      ++frame_diag.observed_board_count;
      translation_errors.push_back(row->translation_error_mm);
      rotation_errors.push_back(row->rotation_error_deg);
      if (row->translation_error_mm >= frame_diag.max_translation_error_mm) {
        frame_diag.max_translation_error_mm = row->translation_error_mm;
        frame_diag.worst_board_id = row->board_id;
      }
      frame_diag.max_rotation_error_deg =
          std::max(frame_diag.max_rotation_error_deg, row->rotation_error_deg);
    }
    if (!translation_errors.empty()) {
      frame_diag.mean_translation_error_mm =
          std::accumulate(translation_errors.begin(), translation_errors.end(), 0.0) /
          static_cast<double>(translation_errors.size());
      frame_diag.mean_rotation_error_deg =
          std::accumulate(rotation_errors.begin(), rotation_errors.end(), 0.0) /
          static_cast<double>(rotation_errors.size());
    }
    result.frame_diagnostics.push_back(frame_diag);
  }

  result.success = true;
  return result;
}

Stage5BenchmarkReport Stage5Benchmark::Run(const Stage5BenchmarkInput& input) const {
  Stage5BenchmarkReport report;
  report.dataset_label = input.dataset_label.empty()
                             ? input.baseline_options.dataset_label
                             : input.dataset_label;
  if (input.use_precomputed_training_measurements) {
    report.stage5_input_mode = "babelcalib_mat_precomputed";
    report.precomputed_training_source =
        input.precomputed_training_measurements.source_path;
    report.precomputed_target_mode_requested =
        input.precomputed_training_measurements.target_mode_requested;
    report.precomputed_target_mode_resolved =
        input.precomputed_training_measurements.target_mode_resolved;
    report.precomputed_board_count =
        input.precomputed_training_measurements.board_count;
    report.precomputed_single_board_ba_mode =
        input.precomputed_training_measurements.single_board_mode;
    report.precomputed_training_frame_count =
        input.precomputed_training_measurements.measurement_result.used_frame_count;
    report.precomputed_training_board_observation_count =
        input.precomputed_training_measurements.measurement_result
            .used_board_observation_count;
    report.precomputed_training_outer_point_count =
        input.precomputed_training_measurements.measurement_result
            .used_outer_point_count;
    report.precomputed_training_internal_point_count =
        input.precomputed_training_measurements.measurement_result
            .used_internal_point_count;
  }
  if (input.use_precomputed_holdout_measurements) {
    report.precomputed_holdout_source =
        input.precomputed_holdout_measurements.source_path;
    report.precomputed_holdout_frame_count =
        input.precomputed_holdout_measurements.measurement_result.used_frame_count;
    report.precomputed_holdout_board_observation_count =
        input.precomputed_holdout_measurements.measurement_result
            .used_board_observation_count;
    report.precomputed_holdout_outer_point_count =
        input.precomputed_holdout_measurements.measurement_result
            .used_outer_point_count;
    report.precomputed_holdout_internal_point_count =
        input.precomputed_holdout_measurements.measurement_result
            .used_internal_point_count;
  }

  if (input.frontend_only) {
    report.split.success = true;
    report.split.mode = "frontend_only_all_frames";
    report.split.split_signature = "frontend_only_all_frames";
    report.split.training_frames = input.all_frames;
  } else {
    const auto stage_start = std::chrono::steady_clock::now();
    report.split = input.external_holdout_frames.empty()
                       ? BuildDeterministicSplit(input.all_frames)
                       : BuildExternalHoldoutSplit(input.all_frames,
                                                   input.external_holdout_frames,
                                                   input.external_holdout_label);
    report.runtime_breakdown.split_seconds = ElapsedSeconds(stage_start);
  }
  if (!report.split.success) {
    report.failure_reason = report.split.failure_reason;
    return report;
  }
  const auto same_frame_sequence = [](const std::vector<FrozenRound2BaselineFrameSource>& lhs,
                                      const std::vector<FrozenRound2BaselineFrameSource>& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index) {
      if (lhs[index].frame_label != rhs[index].frame_label ||
          lhs[index].image_path != rhs[index].image_path) {
        return false;
      }
    }
    return true;
  };
  if (input.holdout_evaluate_full_training_observations &&
      (input.use_precomputed_training_measurements ||
       !same_frame_sequence(report.split.training_frames,
                            report.split.holdout_frames))) {
    report.failure_reason =
        "Full-training-observation holdout evaluation requires identical "
        "image-based training and holdout frame sequences.";
    return report;
  }
  report.split_signature = report.split.split_signature;
  report.kalibr_reference = input.kalibr_reference;
  report.external_holdout_self_frontend_prepass_used =
      !input.use_precomputed_holdout_measurements &&
      input.use_external_holdout_self_frontend_prepass &&
      !input.external_holdout_frames.empty();
  report.external_holdout_observation_source =
      input.use_precomputed_holdout_measurements
          ? "babelcalib_mat_frozen_precomputed_measurements"
          : input.holdout_evaluate_full_training_observations
          ? "same_sequence_full_frontend_observations"
          : report.external_holdout_self_frontend_prepass_used
          ? "external_holdout_self_frontend_prepass_full_measurements"
          : "training_scene_regeneration";

  FrozenRound2BaselineOptions baseline_options = input.baseline_options;
  baseline_options.dataset_label = report.dataset_label;
  baseline_options.training_split_signature = report.split.split_signature;
  const FrozenRound2BaselinePipeline baseline_pipeline(baseline_options);
  report.baseline_result = input.use_precomputed_training_measurements
                               ? baseline_pipeline.RunPrecomputed(
                                     input.precomputed_training_measurements)
                               : baseline_pipeline.Run(
                                     report.split.training_frames);
  report.baseline_protocol_label = report.baseline_result.baseline_protocol_label;
  if (!report.baseline_result.success) {
    report.failure_reason = report.baseline_result.failure_reason;
    return report;
  }
  if (input.frontend_only) {
    report.success = true;
    report.diagnostic_only = true;
    report.warnings.push_back(
        "Stage5 frontend-only mode: selection incremental BA, backend BA, "
        "and holdout evaluation were not run.");
    return report;
  }
  if (!report.baseline_result.stage5_bundle_available) {
    report.failure_reason =
        "Frozen round2 baseline completed but final Stage 5 bundle is not ready.";
    return report;
  }

  std::map<int, std::string> frame_image_paths;
  for (const FrozenRound2BaselineFrameSource& frame : input.all_frames) {
    frame_image_paths[frame.frame_index] = frame.image_path;
  }
  for (const FrozenRound2BaselineFrameSource& frame : input.external_holdout_frames) {
    frame_image_paths[frame.frame_index] = frame.image_path;
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    InternalJointRefineOptions refine_options =
        input.internal_joint_refine_options;
    if (input.use_precomputed_training_measurements) {
      refine_options.mode = InternalJointRefineMode::Off;
    }
    report.internal_joint_refine_result = ApplyInternalJointRefinement(
        report.baseline_result.final_stage5_bundle,
        frame_image_paths,
        refine_options);
    report.runtime_breakdown.internal_joint_refine_seconds =
        ElapsedSeconds(stage_start);
  }
  if (!report.internal_joint_refine_result.success) {
    report.failure_reason =
        report.internal_joint_refine_result.failure_reason.empty()
            ? "Internal joint refinement failed."
            : report.internal_joint_refine_result.failure_reason;
    return report;
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    PreBackendObservationFilterOptions pre_backend_filter_options =
        input.pre_backend_filter_options;
    if (report.precomputed_single_board_ba_mode) {
      pre_backend_filter_options.mode = PreBackendFilterMode::Off;
    }
    report.pre_backend_filter_result = ApplyPreBackendObservationFilter(
        report.internal_joint_refine_result.curated_bundle,
        pre_backend_filter_options);
    report.runtime_breakdown.pre_backend_filter_seconds =
        ElapsedSeconds(stage_start);
  }
  if (!report.pre_backend_filter_result.success) {
    report.failure_reason =
        report.pre_backend_filter_result.failure_reason.empty()
            ? "Pre-backend observation filter failed."
            : report.pre_backend_filter_result.failure_reason;
    return report;
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    InternalBlurObservationFilterOptions blur_filter_options =
        input.internal_blur_filter_options;
    if (input.use_precomputed_training_measurements) {
      blur_filter_options.mode = InternalBlurFilterMode::Off;
    }
    report.internal_blur_filter_result = ApplyInternalBlurObservationFilter(
        report.pre_backend_filter_result.curated_bundle,
        frame_image_paths,
        blur_filter_options);
    report.runtime_breakdown.internal_blur_filter_seconds =
        ElapsedSeconds(stage_start);
  }
  if (!report.internal_blur_filter_result.success) {
    report.failure_reason =
        report.internal_blur_filter_result.failure_reason.empty()
            ? "Internal blur observation filter failed."
            : report.internal_blur_filter_result.failure_reason;
    return report;
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    InternalBlurBoardWeightOptions blur_weight_options =
        input.internal_blur_board_weight_options;
    if (input.use_precomputed_training_measurements) {
      blur_weight_options.mode = InternalBlurBoardWeightMode::Off;
    }
    report.internal_blur_board_weight_result =
        ApplyInternalBlurBoardWeights(
            report.internal_blur_filter_result.curated_bundle,
            frame_image_paths,
            blur_weight_options);
    report.runtime_breakdown.internal_blur_board_weight_seconds =
        ElapsedSeconds(stage_start);
  }
  if (!report.internal_blur_board_weight_result.success) {
    report.failure_reason =
        report.internal_blur_board_weight_result.failure_reason.empty()
            ? "Internal blur board weighting failed."
            : report.internal_blur_board_weight_result.failure_reason;
    return report;
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    InternalObservationWeightOptions observation_weight_options =
        input.internal_observation_weight_options;
    if (report.precomputed_single_board_ba_mode) {
      observation_weight_options.mode = InternalObservationWeightMode::Off;
    }
    report.internal_observation_weight_result =
        ApplyInternalObservationWeights(
            report.internal_blur_board_weight_result.curated_bundle,
            observation_weight_options);
    report.runtime_breakdown.internal_observation_weight_seconds =
        ElapsedSeconds(stage_start);
  }
  if (!report.internal_observation_weight_result.success) {
    report.failure_reason =
        report.internal_observation_weight_result.failure_reason.empty()
            ? "Internal observation weighting failed."
            : report.internal_observation_weight_result.failure_reason;
    return report;
  }

  CalibrationStateBundle curated_backend_seed_bundle =
      report.internal_observation_weight_result.curated_bundle;
  if (input.enable_large_intrinsic_perturbation) {
    report.large_intrinsic_perturbation.outer_only_after_application =
        input.large_intrinsic_perturbation_outer_only_after_application;
    report.large_intrinsic_perturbation
        .frozen_internal_point_count_before_ablation =
        curated_backend_seed_bundle.measurement_dataset
            .accepted_internal_point_count;
    report.large_intrinsic_perturbation.frozen_observation_fingerprint =
        LargePerturbationObservationFingerprint(
            curated_backend_seed_bundle.measurement_dataset);
    if (!ApplyLargeDsIntrinsicPerturbation(
            input, &curated_backend_seed_bundle,
            &report.large_intrinsic_perturbation)) {
      report.failure_reason =
          report.large_intrinsic_perturbation.failure_reason.empty()
              ? "Failed to apply large intrinsic perturbation before selection."
              : report.large_intrinsic_perturbation.failure_reason;
      return report;
    }
    if (input.large_intrinsic_perturbation_outer_only_after_application) {
      DisableFrozenInternalObservationsForPerturbationAblation(
          &curated_backend_seed_bundle);
      if (!curated_backend_seed_bundle.IsReadyForBackend() ||
          curated_backend_seed_bundle.measurement_dataset
                  .accepted_internal_point_count != 0) {
        report.failure_reason =
            "Failed to remove frozen internal residuals after perturbation.";
        return report;
      }
    }
    report.large_intrinsic_perturbation
        .seed_internal_point_count_after_ablation =
        curated_backend_seed_bundle.measurement_dataset
            .accepted_internal_point_count;
  }
  const FrozenRoundArtifacts& trial_source_artifacts =
      report.baseline_result.round2_available
          ? report.baseline_result.round2
          : report.baseline_result.round1;
  const bool single_board_dense_grid_profile =
      input.use_precomputed_training_measurements &&
      input.precomputed_training_measurements.single_board_mode &&
      input.trial_backend_selection_options.enabled;
  CalibrationStateBundle trial_candidate_bundle =
      input.trial_backend_selection_options.enabled
          ? BuildCandidateBackendPoolBundle(
                curated_backend_seed_bundle,
                trial_source_artifacts)
          : curated_backend_seed_bundle;
  if (input.enable_large_intrinsic_perturbation &&
      input.large_intrinsic_perturbation_outer_only_after_application) {
    DisableFrozenInternalObservationsForPerturbationAblation(
        &trial_candidate_bundle);
    if (!trial_candidate_bundle.IsReadyForBackend() ||
        trial_candidate_bundle.measurement_dataset
                .accepted_internal_point_count != 0) {
      report.failure_reason =
          "Failed to remove frozen internal residuals from the perturbation candidate pool.";
      return report;
    }
  }
  if (input.enable_large_intrinsic_perturbation) {
    report.large_intrinsic_perturbation
        .candidate_pool_internal_point_count_after_ablation =
        trial_candidate_bundle.measurement_dataset
            .accepted_internal_point_count;
  }
  if (input.enable_large_intrinsic_perturbation) {
    report.large_intrinsic_perturbation.selection_seed_camera =
        curated_backend_seed_bundle.scene_state.camera;
    report.large_intrinsic_perturbation.selection_candidate_camera =
        trial_candidate_bundle.scene_state.camera;
    report.large_intrinsic_perturbation.selection_seed_matches_perturbed_camera =
        SameLargePerturbationCamera(
            report.large_intrinsic_perturbation.selection_seed_camera,
            report.large_intrinsic_perturbation.perturbed_camera);
    report.large_intrinsic_perturbation
        .selection_candidate_matches_perturbed_camera =
        SameLargePerturbationCamera(
            report.large_intrinsic_perturbation.selection_candidate_camera,
            report.large_intrinsic_perturbation.perturbed_camera);
    if (!report.large_intrinsic_perturbation
             .selection_seed_matches_perturbed_camera ||
        !report.large_intrinsic_perturbation
             .selection_candidate_matches_perturbed_camera) {
      report.failure_reason =
          "Large DS perturbation was not preserved at the selection boundary.";
      return report;
    }
  }
  if (single_board_dense_grid_profile) {
    // A checkerboard target view is atomic: every valid imported grid point
    // belongs to the same incremental batch. Do not inherit point-level
    // filtering from the generic multi-board round-1 selection.
    trial_candidate_bundle.measurement_dataset.frames =
        trial_source_artifacts.measurement_result.frames;
    trial_candidate_bundle.measurement_dataset.solver_observations =
        trial_source_artifacts.measurement_result.solver_observations;
    trial_candidate_bundle.measurement_dataset.source_stage_label =
        trial_source_artifacts.measurement_result.frames.empty()
            ? trial_candidate_bundle.measurement_dataset.source_stage_label
            : trial_candidate_bundle.measurement_dataset.source_stage_label +
                  "_single_board_dense_grid_atomic_views";
    ReevaluateCalibrationStateBundle(&trial_candidate_bundle);
  }

  TrialBackendFrameBoardSelectionOptions effective_selection_options =
      input.trial_backend_selection_options;
  if (single_board_dense_grid_profile) {
    effective_selection_options.single_board_dense_grid_profile = true;
    effective_selection_options.selection_mode =
        TrialBackendFrameBoardSelectionOptions::SelectionMode::KalibrStyleBatch;
    effective_selection_options.budget_mode =
        TrialBackendFrameBoardSelectionOptions::BudgetMode::KalibrStyle;
    if (!effective_selection_options.candidate_order_mode_explicit) {
      effective_selection_options.candidate_order_mode =
          TrialBackendFrameBoardSelectionOptions::CandidateOrderMode::
              RandomShuffle;
    }
    effective_selection_options.candidate_batch_granularity =
        TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity::Frame;
    effective_selection_options.acceptance_policy =
        KalibrStyleBatchAcceptancePolicy::KalibrInformationGain;
    // Kalibr's command-line default is miTol=0.2; miTol=-1 is the explicit
    // all-view override. Keep the same default boundary here.
    if (!effective_selection_options
             .acceptance_information_gain_threshold_explicit) {
      effective_selection_options.acceptance_information_gain_threshold = 0.2;
    }
    effective_selection_options.optimize_intrinsics_in_trial = true;
    effective_selection_options.delayed_intrinsics_release_in_trial = false;
    effective_selection_options.persistent_intrinsics_anchor_prior_enabled = false;
    effective_selection_options.persistent_max_focal_relative_step = 0.0;
    effective_selection_options.persistent_max_principal_step_px = 0.0;
    effective_selection_options.persistent_max_xi_alpha_step = 0.0;
    effective_selection_options.frame_cohesion_enabled = false;
    effective_selection_options.use_consistency_score = false;
    effective_selection_options.max_accepted_per_board = 0;
    effective_selection_options.max_accepted_per_frame = 0;
    effective_selection_options.max_iterations = 50;
    if (!effective_selection_options.candidate_shuffle_seed_set) {
      effective_selection_options.candidate_shuffle_seed_set = true;
      effective_selection_options.candidate_shuffle_seed = 1337u;
    }
  }

  CalibrationStateBundle exact_trial_candidate_bundle;
  CalibrationStateBundle single_board_seed_bundle;
  CalibrationStateBundle controlled_seed_override_bundle;
  std::string checkerboard_seed_strategy = "not_applicable";
  int checkerboard_seed_target_frame_count = 0;
  double checkerboard_seed_fisher_logdet = 0.0;
  double checkerboard_seed_fisher_rank_proxy = 0.0;
  const CalibrationStateBundle* selection_seed_bundle =
      &curated_backend_seed_bundle;
  const CalibrationStateBundle* selection_candidate_bundle =
      &trial_candidate_bundle;
  const bool force_list_exact_input =
      effective_selection_options.force_include_list_is_exact_input &&
      (!effective_selection_options.force_include_frame_board_keys
            .empty() ||
       !effective_selection_options.force_include_frame_label_board_keys
            .empty());
  const bool controlled_seed_override =
      !effective_selection_options.seed_override_frame_board_keys.empty() ||
      !effective_selection_options.seed_override_frame_label_board_keys.empty();
  if (controlled_seed_override) {
    std::set<FrameBoardKey> seed_override_keys;
    for (const JointMeasurementFrameResult& frame :
         trial_candidate_bundle.measurement_dataset.frames) {
      for (const JointBoardObservation& board : frame.board_observations) {
        const FrameBoardKey key(frame.frame_index, board.board_id);
        if (effective_selection_options.seed_override_frame_board_keys.count(key) > 0 ||
            effective_selection_options.seed_override_frame_label_board_keys.count(
                std::make_pair(frame.frame_label, board.board_id)) > 0) {
          seed_override_keys.insert(key);
        }
      }
    }
    controlled_seed_override_bundle = BuildBundleForAcceptedFrameBoardKeys(
        trial_candidate_bundle, trial_candidate_bundle, seed_override_keys,
        trial_candidate_bundle.measurement_dataset.source_stage_label +
            "_controlled_seed_override");
    if (!controlled_seed_override_bundle.IsReadyForBackend()) {
      report.failure_reason =
          "Controlled persistent seed override produced no valid backend input.";
      return report;
    }
    controlled_seed_override_bundle.warnings.push_back(
        "Stage5 used a controlled frame-board seed override for a paired perturbation experiment.");
    selection_seed_bundle = &controlled_seed_override_bundle;
  }
  if (single_board_dense_grid_profile && !force_list_exact_input) {
    const std::set<FrameBoardKey> candidate_keys =
        CollectAcceptedFrameBoardKeys(
            trial_candidate_bundle.measurement_dataset);
    std::set<FrameBoardKey> selected_seed_keys =
        CollectAcceptedFrameBoardKeys(
            curated_backend_seed_bundle.measurement_dataset);
    if (selected_seed_keys.size() < 3 && candidate_keys.size() >= 3) {
      for (const FrameBoardKey& key : candidate_keys) {
        selected_seed_keys.insert(key);
        if (selected_seed_keys.size() >= 3) {
          break;
        }
      }
    }
    if (!selected_seed_keys.empty()) {
      checkerboard_seed_target_frame_count =
          static_cast<int>(selected_seed_keys.size());
      Eigen::Matrix<double, 6, 6> cumulative_fisher =
          Eigen::Matrix<double, 6, 6>::Zero();
      for (const FrameBoardKey& key : selected_seed_keys) {
        const IntrinsicsJacobianInformationSummary info =
            ComputeIntrinsicsJacobianInformation(trial_candidate_bundle, key);
        if (info.available) {
          cumulative_fisher += info.fisher;
        }
      }
      checkerboard_seed_strategy =
          "existing_stage5_coverage_observability_selection_with_pose_marginalized_fisher_audit";
      checkerboard_seed_fisher_logdet =
          RegularizedFisherLogDet(cumulative_fisher);
      checkerboard_seed_fisher_rank_proxy =
          FisherRankProxy(cumulative_fisher);
      single_board_seed_bundle = BuildBundleForAcceptedFrameBoardKeys(
          trial_candidate_bundle,
          trial_candidate_bundle,
          selected_seed_keys,
          trial_candidate_bundle.measurement_dataset.source_stage_label +
              "_single_board_dense_grid_existing_selection_seed");
      if (single_board_seed_bundle.IsReadyForBackend()) {
        selection_seed_bundle = &single_board_seed_bundle;
      }
    }
  }
  if (force_list_exact_input) {
    std::set<FrameBoardKey> exact_keys;
    for (const JointMeasurementFrameResult& frame :
         trial_candidate_bundle.measurement_dataset.frames) {
      for (const JointBoardObservation& board : frame.board_observations) {
        const FrameBoardKey key(frame.frame_index, board.board_id);
        if (IsForceIncludeFrameBoardCandidate(
                key, frame.frame_label,
                effective_selection_options)) {
          exact_keys.insert(key);
        }
      }
    }
    exact_trial_candidate_bundle = BuildBundleForAcceptedFrameBoardKeys(
        trial_candidate_bundle,
        trial_candidate_bundle,
        exact_keys,
        trial_candidate_bundle.measurement_dataset.source_stage_label +
            "_force_include_exact_backend_input");
    exact_trial_candidate_bundle.warnings.push_back(
        "Stage5 trial backend selection force-include list was used as the exact backend input for a controlled ablation.");
    selection_seed_bundle = &exact_trial_candidate_bundle;
    selection_candidate_bundle = &exact_trial_candidate_bundle;
  }
  AslamBackendCalibrationOptions effective_selection_runner_options =
      input.selection_backend_runner_options;
  effective_selection_runner_options.uniform_control_point_mode =
      single_board_dense_grid_profile;
  if (single_board_dense_grid_profile &&
      effective_selection_options.checkerboard_huber_delta_pixels > 0.0) {
    effective_selection_runner_options.outer_huber_delta_pixels =
        effective_selection_options.checkerboard_huber_delta_pixels;
    effective_selection_runner_options.internal_huber_delta_pixels =
        effective_selection_options.checkerboard_huber_delta_pixels;
  }
  effective_selection_runner_options.use_point_type_residual_split =
      single_board_dense_grid_profile
          ? false
          : effective_selection_runner_options.use_point_type_residual_split;
  report.trial_backend_selection_result =
      ApplyTrialBackendFrameBoardSelection(
          *selection_seed_bundle,
          *selection_candidate_bundle,
          input.backend_options,
          effective_selection_options,
          effective_selection_runner_options);
  report.trial_backend_selection_result
      .checkerboard_pose_marginalized_fisher =
      single_board_dense_grid_profile;
  report.trial_backend_selection_result.checkerboard_seed_strategy =
      checkerboard_seed_strategy;
  report.trial_backend_selection_result.checkerboard_seed_target_frame_count =
      checkerboard_seed_target_frame_count;
  report.trial_backend_selection_result.checkerboard_seed_fisher_logdet =
      checkerboard_seed_fisher_logdet;
  report.trial_backend_selection_result.checkerboard_seed_fisher_rank_proxy =
      checkerboard_seed_fisher_rank_proxy;
  report.trial_backend_selection_result.source_joint_input_frame_count =
      static_cast<int>(trial_source_artifacts.joint_inputs.size());
  const HierarchicalMeasurementCounts source_hierarchical_counts =
      ComputeHierarchicalMeasurementCounts(
          trial_source_artifacts.measurement_result);
  report.trial_backend_selection_result.source_measurement_frame_count =
      trial_source_artifacts.measurement_result.used_frame_count;
  report.trial_backend_selection_result
      .source_measurement_board_observation_count =
      trial_source_artifacts.measurement_result.used_board_observation_count;
  report.trial_backend_selection_result.source_measurement_total_point_count =
      trial_source_artifacts.measurement_result.used_total_point_count;
  report.trial_backend_selection_result.source_measurement_outer_point_count =
      trial_source_artifacts.measurement_result.used_outer_point_count;
  report.trial_backend_selection_result
      .source_measurement_internal_point_count =
      trial_source_artifacts.measurement_result.used_internal_point_count;
  report.trial_backend_selection_result
      .source_measurement_hierarchical_frame_count =
      source_hierarchical_counts.frame_count;
  report.trial_backend_selection_result
      .source_measurement_hierarchical_board_observation_count =
      source_hierarchical_counts.board_observation_count;
  report.trial_backend_selection_result
      .source_measurement_hierarchical_total_point_count =
      source_hierarchical_counts.total_point_count;
  report.trial_backend_selection_result
      .source_measurement_hierarchical_outer_point_count =
      source_hierarchical_counts.outer_point_count;
  report.trial_backend_selection_result
      .source_measurement_hierarchical_internal_point_count =
      source_hierarchical_counts.internal_point_count;
  report.trial_backend_selection_result
      .source_measurement_flat_solver_observation_count =
      static_cast<int>(
          trial_source_artifacts.measurement_result.solver_observations.size());
  report.trial_backend_selection_result.source_selection_frame_count =
      trial_source_artifacts.selection_result.accepted_frame_count;
  report.trial_backend_selection_result
      .source_selection_board_observation_count =
      trial_source_artifacts.selection_result.accepted_board_observation_count;
  report.trial_backend_selection_result.source_selection_total_point_count =
      trial_source_artifacts.selection_result.selected_measurement_result
          .used_total_point_count;
  report.trial_backend_selection_result.source_selection_outer_point_count =
      trial_source_artifacts.selection_result.accepted_outer_point_count;
  report.trial_backend_selection_result.source_selection_internal_point_count =
      trial_source_artifacts.selection_result.accepted_internal_point_count;
  report.trial_backend_selection_result.candidate_pool_frame_count =
      trial_candidate_bundle.measurement_dataset.accepted_frame_count;
  report.trial_backend_selection_result.candidate_pool_board_observation_count =
      trial_candidate_bundle.measurement_dataset
          .accepted_board_observation_count;
  report.trial_backend_selection_result.candidate_pool_total_point_count =
      trial_candidate_bundle.measurement_dataset.accepted_total_point_count;
  report.trial_backend_selection_result.candidate_pool_outer_point_count =
      trial_candidate_bundle.measurement_dataset.accepted_outer_point_count;
  report.trial_backend_selection_result.candidate_pool_internal_point_count =
      trial_candidate_bundle.measurement_dataset.accepted_internal_point_count;
  if (!report.trial_backend_selection_result.success) {
    report.failure_reason =
        report.trial_backend_selection_result.failure_reason.empty()
            ? "Trial-backend frame-board selection failed."
            : report.trial_backend_selection_result.failure_reason;
    return report;
  }

  const CalibrationStateBundle& selected_final_backend_bundle =
      input.trial_backend_selection_options.enabled
          ? report.trial_backend_selection_result.curated_bundle
          : curated_backend_seed_bundle;
  BackendInputAblationWorkingResult backend_input_ablation =
      ApplyBackendInputAblationControls(
          selected_final_backend_bundle,
          curated_backend_seed_bundle,
          input.backend_input_ablation_options);
  report.backend_input_ablation_result =
      backend_input_ablation.result;
  if (!report.backend_input_ablation_result.success) {
    report.failure_reason =
        report.backend_input_ablation_result.failure_reason.empty()
            ? "Backend input ablation failed."
            : report.backend_input_ablation_result.failure_reason;
    return report;
  }
  report.warnings.insert(
      report.warnings.end(),
      report.backend_input_ablation_result.warnings.begin(),
      report.backend_input_ablation_result.warnings.end());
  CalibrationStateBundle final_backend_bundle =
      backend_input_ablation.bundle;
  report.backend_problem_input = BuildBackendProblemInput(
      final_backend_bundle,
      input.committed_backend_evaluation_options);
  {
    const auto stage_start = std::chrono::steady_clock::now();
    if (input.use_precomputed_training_measurements) {
      report.training_dataset = BuildEvaluationDatasetFromMeasurementResult(
          input.precomputed_training_measurements.measurement_result,
          std::vector<InternalRegenerationFrameResult>{},
          report.dataset_label + "_precomputed_training", "training",
          report.split.split_signature + "_frozen_precomputed_training");
    } else {
      report.training_dataset = BuildTrainingEvaluationDataset(
          report.baseline_result.final_stage5_bundle);
      report.training_dataset.internal_regeneration_results =
          report.baseline_result.round2_available
              ? report.baseline_result.round2.regeneration_results
              : report.baseline_result.round1.regeneration_results;
    }
    report.runtime_breakdown.training_dataset_build_seconds =
        ElapsedSeconds(stage_start);
  }
  if (!report.training_dataset.success) {
    report.failure_reason = report.training_dataset.failure_reason;
    return report;
  }

  if (input.use_precomputed_holdout_measurements) {
    const auto stage_start = std::chrono::steady_clock::now();
    report.holdout_dataset = BuildEvaluationDatasetFromMeasurementResult(
        input.precomputed_holdout_measurements.measurement_result,
        std::vector<InternalRegenerationFrameResult>{},
        input.external_holdout_label.empty()
            ? report.dataset_label + "_precomputed_holdout"
            : input.external_holdout_label,
        "holdout", report.split.split_signature + "_frozen_precomputed");
    report.runtime_breakdown.holdout_dataset_build_seconds =
        ElapsedSeconds(stage_start);
  } else if (input.holdout_evaluate_full_training_observations) {
    const auto stage_start = std::chrono::steady_clock::now();
    const FrozenRoundArtifacts& training_frontend_artifacts =
        report.baseline_result.round2_available
            ? report.baseline_result.round2
            : report.baseline_result.round1;
    report.holdout_dataset = BuildEvaluationDatasetFromMeasurementResult(
        training_frontend_artifacts.measurement_result,
        training_frontend_artifacts.regeneration_results,
        report.dataset_label + "_same_sequence_full_frontend",
        "holdout",
        report.split.split_signature +
            "_same_sequence_full_frontend_observations",
        true);
    report.runtime_breakdown.holdout_dataset_build_seconds =
        ElapsedSeconds(stage_start);
  } else if (report.external_holdout_self_frontend_prepass_used) {
    const auto stage_start = std::chrono::steady_clock::now();
    FrozenRound2BaselineOptions holdout_prepass_options = baseline_options;
    holdout_prepass_options.outer_only_ablation_mode = false;
    holdout_prepass_options.include_internal_points = true;
    holdout_prepass_options.run_second_pass = true;
    holdout_prepass_options.dataset_label =
        input.external_holdout_label.empty()
            ? report.dataset_label + "_external_holdout_self_prepass"
            : input.external_holdout_label + "_self_frontend_prepass";
    holdout_prepass_options.training_split_signature =
        report.split.split_signature + "_external_holdout_self_frontend_prepass";
    const FrozenRound2BaselinePipeline holdout_prepass_pipeline(
        holdout_prepass_options);
    const FrozenRound2BaselineResult holdout_prepass_result =
        holdout_prepass_pipeline.Run(report.split.holdout_frames);
    report.runtime_breakdown.external_holdout_self_frontend_prepass_seconds =
        ElapsedSeconds(stage_start);
    if (!holdout_prepass_result.success ||
        !holdout_prepass_result.stage5_bundle_available) {
      report.external_holdout_self_frontend_prepass_success = false;
      report.external_holdout_self_frontend_prepass_failure_reason =
          holdout_prepass_result.failure_reason.empty()
              ? "External holdout self frontend prepass failed."
              : holdout_prepass_result.failure_reason;
      report.failure_reason =
          report.external_holdout_self_frontend_prepass_failure_reason;
      return report;
    }
    report.external_holdout_self_frontend_prepass_success = true;
    {
      const auto dataset_start = std::chrono::steady_clock::now();
      const FrozenRoundArtifacts& holdout_frontend_artifacts =
          holdout_prepass_result.round2_available
              ? holdout_prepass_result.round2
              : holdout_prepass_result.round1;
      report.holdout_dataset = BuildEvaluationDatasetFromMeasurementResult(
          holdout_frontend_artifacts.measurement_result,
          holdout_frontend_artifacts.regeneration_results,
          holdout_prepass_options.dataset_label,
          "holdout",
          report.split.split_signature +
              "_external_holdout_self_frontend_prepass_full_measurements");
      report.holdout_dataset.split_label = "holdout";
      report.runtime_breakdown.holdout_dataset_build_seconds =
          ElapsedSeconds(dataset_start);
    }
  } else {
    const JointReprojectionSceneState optimized_scene_state =
        report.baseline_result.round2_available
            ? report.baseline_result.round2.optimization_result.optimized_state
            : report.baseline_result.round1.optimization_result.optimized_state;
    const auto stage_start = std::chrono::steady_clock::now();
    report.holdout_dataset = BuildHoldoutEvaluationDataset(
        report.split.holdout_frames, baseline_options, optimized_scene_state,
        report.split.split_signature, &report.runtime_breakdown.holdout_detection_cache,
        &report.runtime_breakdown.holdout_internal_regeneration_cache);
    report.runtime_breakdown.holdout_dataset_build_seconds =
        ElapsedSeconds(stage_start);
  }
  if (!report.holdout_dataset.success) {
    report.failure_reason = report.holdout_dataset.failure_reason;
    return report;
  }
  if (report.precomputed_single_board_ba_mode) {
    report.training_dataset.uniform_control_point_mode = true;
    report.holdout_dataset.uniform_control_point_mode = true;
  }

  OuterBootstrapCameraIntrinsics kalibr_intrinsics;
  std::string intrinsics_error;
  if (!LoadKalibrCamchainIntrinsics(input.kalibr_reference.camchain_yaml,
                                    &kalibr_intrinsics,
                                    &intrinsics_error)) {
    report.failure_reason = intrinsics_error;
    return report;
  }

  const OuterBootstrapCameraIntrinsics& final_backend_camera =
      final_backend_bundle.scene_state.camera;
  const std::string our_family =
      final_backend_camera.NormalizedFamilyString();
  const std::string kalibr_family = kalibr_intrinsics.NormalizedFamilyString();
  report.fair_protocol_matched =
      (input.kalibr_reference.camera_model_family == kalibr_family) &&
      (our_family == kalibr_family) &&
      !input.kalibr_reference.training_split_signature.empty() &&
      input.kalibr_reference.training_split_signature == report.split.split_signature;
  report.diagnostic_only = !report.fair_protocol_matched;
  if (!report.fair_protocol_matched) {
    std::ostringstream warning;
    warning << "Kalibr comparison downgraded to diagnostic-only: ours=" << our_family
            << " kalibr=" << kalibr_family
            << " source_label_family=" << input.kalibr_reference.camera_model_family
            << " split_match="
            << ((!input.kalibr_reference.training_split_signature.empty() &&
                 input.kalibr_reference.training_split_signature ==
                     report.split.split_signature)
                    ? 1
                    : 0);
    report.warnings.push_back(warning.str());
  }

  report.our_training_evaluation = EvaluateCameraModel(
      report.training_dataset,
      final_backend_camera,
      "ours");
  const OuterBootstrapCameraIntrinsics& initialization_camera =
      report.baseline_result.auto_camera_initialization.selected_camera;
  report.initialization_training_evaluation = EvaluateCameraModel(
      report.training_dataset, initialization_camera, "initialization_only");
  report.initialization_holdout_evaluation = EvaluateCameraModel(
      report.holdout_dataset, initialization_camera, "initialization_only");
  if (report.large_intrinsic_perturbation.enabled) {
    report.perturbation_boundary_training_evaluation = EvaluateCameraModel(
        report.training_dataset,
        report.large_intrinsic_perturbation.perturbed_camera,
        "perturbation_boundary");
    report.perturbation_boundary_holdout_evaluation = EvaluateCameraModel(
        report.holdout_dataset,
        report.large_intrinsic_perturbation.perturbed_camera,
        "perturbation_boundary");
  }
  report.kalibr_training_evaluation =
      EvaluateCameraModel(report.training_dataset, kalibr_intrinsics, "kalibr");
  report.our_holdout_evaluation = EvaluateCameraModel(
      report.holdout_dataset,
      final_backend_camera,
      "ours");
  report.kalibr_holdout_evaluation =
      EvaluateCameraModel(report.holdout_dataset, kalibr_intrinsics, "kalibr");

  const std::string checkpoint_camera_family =
      final_backend_camera.NormalizedFamilyString();
  if ((checkpoint_camera_family == "ds-none" ||
       checkpoint_camera_family == "pinhole-equi" ||
       checkpoint_camera_family == "omni-none")) {
    std::vector<const TrialBackendFrameBoardObservationDecision*>
        accepted_checkpoints;
    for (const TrialBackendFrameBoardObservationDecision& decision :
         report.trial_backend_selection_result.decisions) {
      if (decision.persistent_incremental_attempt_order >= 0 &&
          decision.persistent_incremental_batch_accepted) {
        accepted_checkpoints.push_back(&decision);
      }
    }
    std::sort(
        accepted_checkpoints.begin(), accepted_checkpoints.end(),
        [](const TrialBackendFrameBoardObservationDecision* lhs,
           const TrialBackendFrameBoardObservationDecision* rhs) {
          return lhs->persistent_incremental_attempt_order <
                 rhs->persistent_incremental_attempt_order;
        });
    for (const TrialBackendFrameBoardObservationDecision* decision :
         accepted_checkpoints) {
      PersistentCameraCheckpointEvaluation checkpoint;
      checkpoint.attempt_order =
          decision->persistent_incremental_attempt_order;
      checkpoint.frame_index = decision->frame_index;
      checkpoint.frame_label = decision->frame_label;
      checkpoint.information_gain =
          decision->persistent_incremental_information_gain;
      checkpoint.camera = final_backend_camera;
      checkpoint.camera.xi =
          decision->persistent_incremental_camera_xi_after;
      checkpoint.camera.alpha =
          decision->persistent_incremental_camera_alpha_after;
      checkpoint.camera.fu =
          decision->persistent_incremental_camera_fu_after;
      checkpoint.camera.fv =
          decision->persistent_incremental_camera_fv_after;
      checkpoint.camera.cu =
          decision->persistent_incremental_camera_cu_after;
      checkpoint.camera.cv =
          decision->persistent_incremental_camera_cv_after;
      if (checkpoint_camera_family == "pinhole-equi") {
        checkpoint.camera.distortion_coeffs = {
            decision->persistent_incremental_camera_k1_after,
            decision->persistent_incremental_camera_k2_after,
            decision->persistent_incremental_camera_k3_after,
            decision->persistent_incremental_camera_k4_after};
      }
      checkpoint.training_evaluation = EvaluateCameraModel(
          report.training_dataset, checkpoint.camera,
          "persistent_checkpoint_training");
      checkpoint.holdout_evaluation = EvaluateCameraModel(
          report.holdout_dataset, checkpoint.camera,
          "persistent_checkpoint_holdout");
      report.persistent_camera_checkpoint_evaluations.push_back(
          std::move(checkpoint));
    }

    // Multi-board runs expose checkpoints strictly as diagnostics. The robust
    // checkpoint selector below was designed for one board with independent
    // per-view poses and must never replace a committed multi-board scene.
    if (report.precomputed_single_board_ba_mode) {
    const EvaluationRobustSummary initialization_robust =
        SummarizeEvaluationRobustness(
            report.initialization_training_evaluation);
    OuterBootstrapCameraIntrinsics selected_camera = initialization_camera;
    CameraModelRefitEvaluationResult selected_training_evaluation =
        report.initialization_training_evaluation;
    EvaluationRobustSummary selected_robust = initialization_robust;
    std::string selected_label = "initialization_only";
    int selected_attempt_order = -1;
    bool selected_by_early_cross_fold_consensus = false;
    std::size_t first_consensus_checkpoint_index =
        report.persistent_camera_checkpoint_evaluations.size();
    constexpr std::size_t kConsensusRefinementWindow = 4;
    for (std::size_t checkpoint_index = 0;
         checkpoint_index <
         report.persistent_camera_checkpoint_evaluations.size();
         ++checkpoint_index) {
      const PersistentCameraCheckpointEvaluation& checkpoint =
          report.persistent_camera_checkpoint_evaluations[checkpoint_index];
      if (selected_by_early_cross_fold_consensus &&
          checkpoint_index > first_consensus_checkpoint_index +
                                 kConsensusRefinementWindow) {
        break;
      }
      const EvaluationRobustSummary candidate_robust =
          SummarizeEvaluationRobustness(checkpoint.training_evaluation);
      const bool finite_candidate =
          std::isfinite(candidate_robust.frame_median_rmse) &&
          std::isfinite(candidate_robust.frame_p90_rmse) &&
          std::isfinite(candidate_robust.huber15_rmse) &&
          std::isfinite(candidate_robust.fold_median_mean_rmse) &&
          std::isfinite(candidate_robust.fold_median_max_rmse);
      const bool cross_fold_consensus =
          finite_candidate &&
          candidate_robust.fold_median_mean_rmse <=
              0.95 * initialization_robust.fold_median_mean_rmse &&
          candidate_robust.fold_median_max_rmse <=
              initialization_robust.fold_median_max_rmse;
      const bool center_healthy =
          candidate_robust.frame_median_rmse <=
          initialization_robust.frame_median_rmse;
      const bool bounded_tail =
          candidate_robust.frame_p90_rmse <=
              1.10 * initialization_robust.frame_p90_rmse &&
          candidate_robust.huber15_rmse <=
              1.03 * initialization_robust.huber15_rmse;
      if (!cross_fold_consensus || !center_healthy || !bounded_tail) {
        continue;
      }
      if (selected_by_early_cross_fold_consensus) {
        const std::array<std::pair<double, double>, 5> metric_pairs{{
            {candidate_robust.fold_median_mean_rmse,
             selected_robust.fold_median_mean_rmse},
            {candidate_robust.fold_median_max_rmse,
             selected_robust.fold_median_max_rmse},
            {candidate_robust.frame_median_rmse,
             selected_robust.frame_median_rmse},
            {candidate_robust.frame_p90_rmse,
             selected_robust.frame_p90_rmse},
            {candidate_robust.huber15_rmse,
             selected_robust.huber15_rmse},
        }};
        int improved_metric_count = 0;
        bool bounded_relative_regression = true;
        for (const std::pair<double, double>& metric_pair : metric_pairs) {
          if (metric_pair.first < metric_pair.second) {
            ++improved_metric_count;
          }
          if (metric_pair.first > 1.03 * metric_pair.second) {
            bounded_relative_regression = false;
          }
        }
        const bool central_metric_improved =
            candidate_robust.fold_median_mean_rmse <
                selected_robust.fold_median_mean_rmse ||
            candidate_robust.frame_median_rmse <
                selected_robust.frame_median_rmse;
        if (improved_metric_count < 3 || !bounded_relative_regression ||
            !central_metric_improved) {
          continue;
        }
      } else {
        first_consensus_checkpoint_index = checkpoint_index;
      }
      selected_camera = checkpoint.camera;
      selected_training_evaluation = checkpoint.training_evaluation;
      selected_robust = candidate_robust;
      selected_label = "accepted_batch";
      selected_attempt_order = checkpoint.attempt_order;
      selected_by_early_cross_fold_consensus = true;
    }

    if (!selected_by_early_cross_fold_consensus) {
      for (const PersistentCameraCheckpointEvaluation& checkpoint :
           report.persistent_camera_checkpoint_evaluations) {
        const EvaluationRobustSummary candidate_robust =
            SummarizeEvaluationRobustness(checkpoint.training_evaluation);
        const bool finite_candidate =
            std::isfinite(candidate_robust.frame_median_rmse) &&
            std::isfinite(candidate_robust.frame_p90_rmse) &&
            std::isfinite(candidate_robust.huber15_rmse);
        const bool worst_fold_improved =
            finite_candidate &&
            std::isfinite(candidate_robust.fold_median_max_rmse) &&
            candidate_robust.fold_median_max_rmse <
                0.995 * selected_robust.fold_median_max_rmse;
        const bool fold_mean_healthy =
            candidate_robust.fold_median_mean_rmse <=
                1.005 * initialization_robust.fold_median_mean_rmse;
        const bool tail_healthy =
            candidate_robust.frame_p90_rmse <=
                1.02 * initialization_robust.frame_p90_rmse;
        const bool robust_cost_healthy =
            candidate_robust.huber15_rmse <=
                1.01 * initialization_robust.huber15_rmse;
        if (!worst_fold_improved || !fold_mean_healthy || !tail_healthy ||
            !robust_cost_healthy) {
          continue;
        }
        selected_camera = checkpoint.camera;
        selected_training_evaluation = checkpoint.training_evaluation;
        selected_robust = candidate_robust;
        selected_label = "accepted_batch";
        selected_attempt_order = checkpoint.attempt_order;
      }
    }

    if (ApplySingleBoardCameraAndPoseRefit(
            selected_camera, selected_training_evaluation,
            &final_backend_bundle)) {
      report.checkerboard_robust_checkpoint_selection_used = true;
      report.checkerboard_robust_checkpoint_criterion =
          selected_by_early_cross_fold_consensus
              ? "training_early_cross_fold_consensus_with_center_guarded_pareto_refinement"
              : "training_worst_fold_median_with_fold_mean_frame_p90_and_huber_guards";
      report.checkerboard_robust_checkpoint_label = selected_label;
      report.checkerboard_robust_checkpoint_attempt_order =
          selected_attempt_order;
      report.checkerboard_robust_checkpoint_frame_median_rmse =
          selected_robust.frame_median_rmse;
      report.checkerboard_robust_checkpoint_frame_p90_rmse =
          selected_robust.frame_p90_rmse;
      report.checkerboard_robust_checkpoint_huber_rmse =
          selected_robust.huber15_rmse;
      report.checkerboard_robust_checkpoint_fold_median_mean_rmse =
          selected_robust.fold_median_mean_rmse;
      report.checkerboard_robust_checkpoint_fold_median_max_rmse =
          selected_robust.fold_median_max_rmse;
      report.checkerboard_robust_checkpoint_fold_median_std_rmse =
          selected_robust.fold_median_std_rmse;
      report.backend_problem_input = BuildBackendProblemInput(
          final_backend_bundle,
          input.committed_backend_evaluation_options);
      report.our_training_evaluation = EvaluateCameraModel(
          report.training_dataset, final_backend_bundle.scene_state.camera,
          "ours");
      report.our_holdout_evaluation = EvaluateCameraModel(
          report.holdout_dataset, final_backend_bundle.scene_state.camera,
          "ours");
      std::ostringstream warning;
      warning << "Checkerboard training-only robust checkpoint selected "
              << selected_label
              << " attempt_order=" << selected_attempt_order
              << " frame_median_rmse="
              << selected_robust.frame_median_rmse
              << " frame_p90_rmse=" << selected_robust.frame_p90_rmse
              << " huber15_rmse=" << selected_robust.huber15_rmse
              << " fold_median_mean_rmse="
              << selected_robust.fold_median_mean_rmse
              << " fold_median_max_rmse="
              << selected_robust.fold_median_max_rmse
              << " fold_median_std_rmse="
              << selected_robust.fold_median_std_rmse
              << "; frozen holdout was not used for checkpoint selection.";
      report.warnings.push_back(warning.str());
    } else {
      report.warnings.push_back(
          "Checkerboard robust checkpoint selection could not refit the "
          "committed single-board poses; retained the persistent final state.");
    }
    }
  }

  report.additional_camera_references = input.additional_camera_references;
  std::vector<CameraRayCurveReference> ray_curve_references;
  ray_curve_references.push_back(CameraRayCurveReference{"kalibr", kalibr_intrinsics});
  for (std::size_t reference_index = 0;
       reference_index < input.additional_camera_references.size();
       ++reference_index) {
    const KalibrBenchmarkReference& reference =
        input.additional_camera_references[reference_index];
    OuterBootstrapCameraIntrinsics reference_intrinsics;
    std::string reference_error;
    if (!LoadKalibrCamchainIntrinsics(reference.camchain_yaml,
                                      &reference_intrinsics,
                                      &reference_error)) {
      report.failure_reason = reference_error;
      return report;
    }
    std::string method_label = reference.source_label.empty()
                                   ? ("reference_" +
                                      std::to_string(reference_index + 1))
                                   : reference.source_label;
    method_label = SanitizeMetricKey(method_label);
    ray_curve_references.push_back(
        CameraRayCurveReference{method_label, reference_intrinsics});
    report.additional_training_evaluations.push_back(
        EvaluateCameraModel(report.training_dataset,
                            reference_intrinsics,
                            method_label));
    report.additional_holdout_evaluations.push_back(
        EvaluateCameraModel(report.holdout_dataset,
                            reference_intrinsics,
                            method_label));
  }
  report.camera_ray_curve_diagnostics = ComputeCameraRayCurveDiagnostics(
      final_backend_camera,
      "stage5_final_backend_bundle_after_selection_ablation",
      ray_curve_references);
  if (!report.camera_ray_curve_diagnostics.success &&
      !report.camera_ray_curve_diagnostics.failure_reason.empty()) {
    report.warnings.push_back(
        "Camera ray-curve diagnostics failed: " +
        report.camera_ray_curve_diagnostics.failure_reason);
  }
  report.warnings.insert(
      report.warnings.end(),
      report.camera_ray_curve_diagnostics.warnings.begin(),
      report.camera_ray_curve_diagnostics.warnings.end());

  if (input.multi_board_consistency_diagnostics_options.enabled) {
    report.multi_board_consistency_diagnostics = EvaluateMultiBoardConsistency(
        report.training_dataset,
	        report.our_training_evaluation,
	        report.backend_problem_input,
	        final_backend_bundle,
	        input.multi_board_consistency_diagnostics_options);
    if (!report.multi_board_consistency_diagnostics.success &&
        report.multi_board_consistency_diagnostics.failure_reason != "disabled") {
      report.warnings.push_back(
          "Multi-board consistency diagnostics failed: " +
          report.multi_board_consistency_diagnostics.failure_reason);
    }
  }

  if (!report.our_training_evaluation.success ||
      !report.kalibr_training_evaluation.success ||
      !report.our_holdout_evaluation.success ||
      !report.kalibr_holdout_evaluation.success) {
    report.failure_reason = "Camera-only refit evaluation failed for one or more methods.";
    return report;
  }

  if (input.enable_diagnostic_compare) {
    const auto stage_start = std::chrono::steady_clock::now();
    KalibrBenchmarkInput diagnostic_input;
    diagnostic_input.dataset_label = report.dataset_label;
    diagnostic_input.kalibr_camchain_yaml = input.kalibr_reference.camchain_yaml;
    diagnostic_input.our_bundle = report.baseline_result.final_stage5_bundle;
    const KalibrBenchmark diagnostic_benchmark;
    report.diagnostic_compare = diagnostic_benchmark.Compare(diagnostic_input);
    report.runtime_breakdown.diagnostic_compare_seconds =
        ElapsedSeconds(stage_start);
    if (!report.diagnostic_compare.success) {
      report.warnings.push_back(
          "Low-level Stage 5 diagnostic projection compare failed: " +
          report.diagnostic_compare.failure_reason);
    }
  } else {
    report.diagnostic_compare.failure_reason = "Skipped by runtime mode.";
  }

  report.final_backend_scene = final_backend_bundle.scene_state;
  report.final_backend_scene_available = report.final_backend_scene.IsValid();
  if (report.large_intrinsic_perturbation.enabled &&
      report.final_backend_scene_available) {
    report.large_perturbation_pose_orientation_evaluation =
        EvaluateMultiBoardPoseOrientation(
            report.holdout_dataset, report.final_backend_scene,
            report.large_intrinsic_perturbation.reference_scene);
  }

  report.success = true;
  report.warnings.insert(report.warnings.end(),
                         report.baseline_result.warnings.begin(),
                         report.baseline_result.warnings.end());
  return report;
}

cv::Mat Stage5Benchmark::RenderProjectionComparison(const Stage5BenchmarkReport& report,
                                                    int max_width,
                                                    int max_height) const {
  if (!report.diagnostic_compare.success) {
    return cv::Mat();
  }
  const KalibrBenchmark diagnostic_benchmark;
  return diagnostic_benchmark.RenderProjectionComparison(
      report.diagnostic_compare, max_width, max_height);
}

std::string Stage5Benchmark::FindFrameImagePath(const Stage5BenchmarkReport& report,
                                                int frame_index) const {
  const auto search = [frame_index](const std::vector<FrozenRound2BaselineFrameSource>& frames)
      -> std::string {
    for (const FrozenRound2BaselineFrameSource& frame : frames) {
      if (frame.frame_index == frame_index) {
        return frame.image_path;
      }
    }
    return std::string();
  };

  std::string image_path = search(report.split.holdout_frames);
  if (!image_path.empty()) {
    return image_path;
  }
  image_path = search(report.split.training_frames);
  if (!image_path.empty()) {
    return image_path;
  }
  return std::string();
}

cv::Mat Stage5Benchmark::RenderEvaluationFrameOverlay(
    const Stage5BenchmarkReport& report,
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index) const {
  const std::string image_path = FindFrameImagePath(report, frame_index);
  if (image_path.empty()) {
    return cv::Mat();
  }

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    return cv::Mat();
  }

  cv::Mat output = image.clone();
  int point_count = 0;
  int outer_count = 0;
  int internal_count = 0;
  double worst_residual = 0.0;

  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index) {
      continue;
    }
    ++point_count;
    if (point.point_type == JointPointType::Outer) {
      ++outer_count;
    } else {
      ++internal_count;
    }
    worst_residual = std::max(worst_residual, point.residual_norm);

    const cv::Point observed(static_cast<int>(std::lround(point.observed_image_xy.x())),
                             static_cast<int>(std::lround(point.observed_image_xy.y())));
    const cv::Point predicted(static_cast<int>(std::lround(point.predicted_image_xy.x())),
                              static_cast<int>(std::lround(point.predicted_image_xy.y())));
    const cv::Scalar projected_color =
        point.point_type == JointPointType::Outer ? cv::Scalar(60, 220, 80)
                                                  : cv::Scalar(40, 180, 255);
    cv::line(output, observed, predicted, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
    // Outer points use cross/circle markers. Dense internal observations use
    // compact filled dots, with the prediction remaining in the foreground.
    if (point.point_type == JointPointType::Outer) {
      cv::drawMarker(output, observed, cv::Scalar(0, 0, 255),
                     cv::MARKER_CROSS, 13, 1, cv::LINE_AA);
      cv::circle(output, predicted, 8, projected_color, 2, cv::LINE_AA);
    } else {
      cv::circle(output, observed, 2, cv::Scalar(0, 0, 255), cv::FILLED,
                 cv::LINE_AA);
      cv::circle(output, predicted, 3, projected_color, cv::FILLED,
                 cv::LINE_AA);
    }
  }

  double frame_rmse = 0.0;
  std::string frame_label;
  for (const CameraModelRefitFrameDiagnostics& frame : evaluation.frame_diagnostics) {
    if (frame.frame_index == frame_index) {
      frame_rmse = frame.rmse;
      frame_label = frame.frame_label;
      break;
    }
  }

  const int banner_height = 82;
  cv::rectangle(output, cv::Rect(0, 0, output.cols, banner_height),
                cv::Scalar(18, 18, 18), cv::FILLED);
  std::ostringstream header;
  header << evaluation.method_label << " " << evaluation.split_label
         << " frame=" << frame_index;
  if (!frame_label.empty()) {
    header << " (" << frame_label << ")";
  }
  header << " rmse=" << frame_rmse;
  cv::putText(output, header.str(), cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.62,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

  std::ostringstream summary;
  summary << "points=" << point_count << " outer=" << outer_count
          << " internal=" << internal_count
          << " worst=" << worst_residual;
  cv::putText(output, summary.str(), cv::Point(18, 54), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(195, 195, 195), 1, cv::LINE_AA);

  return output;
}

cv::Mat Stage5Benchmark::RenderEvaluationBoardObservationOverlay(
    const Stage5BenchmarkReport& report,
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index,
    int board_id) const {
  cv::Mat full_overlay = RenderEvaluationFrameOverlay(report, evaluation, frame_index);
  if (full_overlay.empty()) {
    return cv::Mat();
  }

  bool has_points = false;
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  int point_count = 0;
  double board_rmse = 0.0;

  for (const CameraModelRefitBoardObservationDiagnostics& board :
       evaluation.board_observation_diagnostics) {
    if (board.frame_index == frame_index && board.board_id == board_id) {
      board_rmse = board.evaluation_rmse;
      break;
    }
  }

  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index || point.board_id != board_id) {
      continue;
    }
    has_points = true;
    ++point_count;
    min_x = std::min(min_x, std::min(point.observed_image_xy.x(), point.predicted_image_xy.x()));
    min_y = std::min(min_y, std::min(point.observed_image_xy.y(), point.predicted_image_xy.y()));
    max_x = std::max(max_x, std::max(point.observed_image_xy.x(), point.predicted_image_xy.x()));
    max_y = std::max(max_y, std::max(point.observed_image_xy.y(), point.predicted_image_xy.y()));
  }

  if (!has_points) {
    return cv::Mat();
  }

  const int padding = 80;
  cv::Rect crop_rect(static_cast<int>(std::floor(min_x)) - padding,
                     static_cast<int>(std::floor(min_y)) - padding,
                     static_cast<int>(std::ceil(max_x - min_x)) + 2 * padding,
                     static_cast<int>(std::ceil(max_y - min_y)) + 2 * padding);
  crop_rect = ClampRectToImage(crop_rect, full_overlay.size());
  if (crop_rect.width <= 0 || crop_rect.height <= 0) {
    return cv::Mat();
  }

  cv::Mat cropped = full_overlay(crop_rect).clone();
  cv::rectangle(cropped, cv::Rect(0, 0, cropped.cols, 54), cv::Scalar(18, 18, 18), cv::FILLED);
  std::ostringstream banner;
  banner << evaluation.method_label << " frame=" << frame_index
         << " board=" << board_id
         << " rmse=" << board_rmse
         << " points=" << point_count;
  cv::putText(cropped, banner.str(), cv::Point(12, 24), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
  cv::putText(cropped, "outer: red cross/green circle; internal: red/orange dots",
              cv::Point(12, 44), cv::FONT_HERSHEY_PLAIN, 1.0,
              cv::Scalar(190, 190, 190), 1, cv::LINE_AA);
  return cropped;
}

cv::Mat Stage5Benchmark::RenderOuterPoseFitFrameOverlay(
    const Stage5BenchmarkReport& report,
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index) const {
  const std::string image_path = FindFrameImagePath(report, frame_index);
  if (image_path.empty()) {
    return cv::Mat();
  }

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    return cv::Mat();
  }

  cv::Mat output = image.clone();
  int outer_count = 0;
  double worst_outer_residual = 0.0;
  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index || point.point_type != JointPointType::Outer) {
      continue;
    }
    ++outer_count;
    worst_outer_residual = std::max(worst_outer_residual, point.residual_norm);
    DrawObservedPredictedPoint(&output, point, cv::Scalar(60, 220, 80), 5, true);
  }

  const CameraModelRefitFrameDiagnostics* frame_diag =
      FindFrameDiagnostics(evaluation, frame_index);
  const double outer_frame_rmse = ComputeOuterRmseForFrame(evaluation, frame_index);
  const int banner_height = 82;
  cv::rectangle(output, cv::Rect(0, 0, output.cols, banner_height),
                cv::Scalar(18, 18, 18), cv::FILLED);

  std::ostringstream header;
  header << evaluation.method_label << " " << evaluation.split_label
         << " outer-only frame=" << frame_index;
  if (frame_diag != nullptr && !frame_diag->frame_label.empty()) {
    header << " (" << frame_diag->frame_label << ")";
  }
  header << " outer_only_rmse=" << outer_frame_rmse;
  cv::putText(output, header.str(), cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.62,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

  std::ostringstream summary;
  summary << "outer_points=" << outer_count
          << " eval_outer_only_rmse=" << evaluation.outer_only_rmse
          << " worst_outer=" << worst_outer_residual;
  cv::putText(output, summary.str(), cv::Point(18, 54), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(195, 195, 195), 1, cv::LINE_AA);
  return output;
}

cv::Mat Stage5Benchmark::RenderOuterPoseFitBoardOverlay(
    const Stage5BenchmarkReport& report,
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index,
    int board_id) const {
  const std::string image_path = FindFrameImagePath(report, frame_index);
  if (image_path.empty()) {
    return cv::Mat();
  }

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    return cv::Mat();
  }

  bool has_outer_points = false;
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  int outer_point_count = 0;
  double worst_outer_residual = 0.0;
  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index || point.board_id != board_id ||
        point.point_type != JointPointType::Outer) {
      continue;
    }
    if (!AccumulatePointBounds(point, &min_x, &min_y, &max_x, &max_y)) {
      continue;
    }
    has_outer_points = true;
    ++outer_point_count;
    worst_outer_residual = std::max(worst_outer_residual, point.residual_norm);
  }
  if (!has_outer_points) {
    return cv::Mat();
  }

  const int padding = 80;
  cv::Rect crop_rect(static_cast<int>(std::floor(min_x)) - padding,
                     static_cast<int>(std::floor(min_y)) - padding,
                     static_cast<int>(std::ceil(max_x - min_x)) + 2 * padding,
                     static_cast<int>(std::ceil(max_y - min_y)) + 2 * padding);
  crop_rect = ClampRectToImage(crop_rect, image.size());
  if (crop_rect.width <= 0 || crop_rect.height <= 0) {
    return cv::Mat();
  }

  cv::Mat cropped = image(crop_rect).clone();
  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index || point.board_id != board_id ||
        point.point_type != JointPointType::Outer ||
        !IsFiniteImagePoint(point.observed_image_xy) ||
        !IsFiniteImagePoint(point.predicted_image_xy)) {
      continue;
    }

    CameraModelRefitPointDiagnostics shifted_point = point;
    shifted_point.observed_image_xy -=
        Eigen::Vector2d(static_cast<double>(crop_rect.x), static_cast<double>(crop_rect.y));
    shifted_point.predicted_image_xy -=
        Eigen::Vector2d(static_cast<double>(crop_rect.x), static_cast<double>(crop_rect.y));
    DrawObservedPredictedPoint(&cropped, shifted_point, cv::Scalar(60, 220, 80), 5, true);
  }

  const CameraModelRefitBoardObservationDiagnostics* board_diag =
      FindBoardDiagnostics(evaluation, frame_index, board_id);
  cv::rectangle(cropped, cv::Rect(0, 0, cropped.cols, 54), cv::Scalar(18, 18, 18),
                cv::FILLED);
  std::ostringstream banner;
  banner << evaluation.method_label << " frame=" << frame_index
         << " board=" << board_id
         << " pose_fit_outer_rmse="
         << (board_diag != nullptr ? board_diag->pose_fit_outer_rmse : 0.0)
         << " outer_points=" << outer_point_count;
  cv::putText(cropped, banner.str(), cv::Point(12, 24), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
  std::ostringstream detail;
  detail << "green: observed outer, red cross: predicted, worst_outer="
         << worst_outer_residual;
  cv::putText(cropped, detail.str(), cv::Point(12, 44), cv::FONT_HERSHEY_PLAIN, 1.0,
              cv::Scalar(190, 190, 190), 1, cv::LINE_AA);
  return cropped;
}

void WriteStage5BenchmarkProtocolSummary(const std::string& path,
                                         const Stage5BenchmarkReport& report) {
  std::ofstream output(path.c_str());
  output << "success: " << (report.success ? 1 : 0) << "\n";
  output << "failure_reason: " << report.failure_reason << "\n";
  output << "baseline_protocol_label: " << report.baseline_protocol_label << "\n";
  output << "dataset_label: " << report.dataset_label << "\n";
  output << "split_signature: " << report.split_signature << "\n";
  output << "split_mode: " << report.split.mode << "\n";
  output << "holdout_stride: " << report.split.holdout_stride << "\n";
  output << "holdout_offset: " << report.split.holdout_offset << "\n";
  output << "holdout_ratio: " << report.split.holdout_ratio << "\n";
  output << "split_random_seed: " << report.split.random_seed << "\n";
  output << "training_frame_count: " << report.split.training_frames.size() << "\n";
  output << "holdout_frame_count: " << report.split.holdout_frames.size() << "\n";
  output << "stage5_input_mode: " << report.stage5_input_mode << "\n";
  output << "precomputed_training_source: "
         << report.precomputed_training_source << "\n";
  output << "precomputed_holdout_source: "
         << report.precomputed_holdout_source << "\n";
  output << "precomputed_target_mode_requested: "
         << report.precomputed_target_mode_requested << "\n";
  output << "precomputed_target_mode_resolved: "
         << report.precomputed_target_mode_resolved << "\n";
  output << "precomputed_board_count: "
         << report.precomputed_board_count << "\n";
  output << "precomputed_single_board_ba_mode: "
         << (report.precomputed_single_board_ba_mode ? 1 : 0) << "\n";
  output << "checkerboard_robust_checkpoint_selection_used: "
         << (report.checkerboard_robust_checkpoint_selection_used ? 1 : 0)
         << "\n";
  output << "checkerboard_robust_checkpoint_criterion: "
         << report.checkerboard_robust_checkpoint_criterion << "\n";
  output << "checkerboard_robust_checkpoint_label: "
         << report.checkerboard_robust_checkpoint_label << "\n";
  output << "checkerboard_robust_checkpoint_attempt_order: "
         << report.checkerboard_robust_checkpoint_attempt_order << "\n";
  output << "checkerboard_robust_checkpoint_frame_median_rmse: "
         << report.checkerboard_robust_checkpoint_frame_median_rmse << "\n";
  output << "checkerboard_robust_checkpoint_frame_p90_rmse: "
         << report.checkerboard_robust_checkpoint_frame_p90_rmse << "\n";
  output << "checkerboard_robust_checkpoint_huber_rmse: "
         << report.checkerboard_robust_checkpoint_huber_rmse << "\n";
  output << "checkerboard_robust_checkpoint_fold_median_mean_rmse: "
         << report.checkerboard_robust_checkpoint_fold_median_mean_rmse
         << "\n";
  output << "checkerboard_robust_checkpoint_fold_median_max_rmse: "
         << report.checkerboard_robust_checkpoint_fold_median_max_rmse
         << "\n";
  output << "checkerboard_robust_checkpoint_fold_median_std_rmse: "
         << report.checkerboard_robust_checkpoint_fold_median_std_rmse
         << "\n";
  output << "precomputed_uniform_control_point_mode: "
         << (report.precomputed_single_board_ba_mode ? 1 : 0) << "\n";
  output << "precomputed_training_frame_count: "
         << report.precomputed_training_frame_count << "\n";
  output << "precomputed_training_board_observation_count: "
         << report.precomputed_training_board_observation_count << "\n";
  if (report.precomputed_single_board_ba_mode) {
    output << "precomputed_training_control_point_count: "
           << report.precomputed_training_outer_point_count +
                  report.precomputed_training_internal_point_count
           << "\n";
  } else {
    output << "precomputed_training_outer_point_count: "
           << report.precomputed_training_outer_point_count << "\n";
    output << "precomputed_training_internal_point_count: "
           << report.precomputed_training_internal_point_count << "\n";
  }
  output << "precomputed_holdout_frame_count: "
         << report.precomputed_holdout_frame_count << "\n";
  output << "precomputed_holdout_board_observation_count: "
         << report.precomputed_holdout_board_observation_count << "\n";
  if (report.precomputed_single_board_ba_mode) {
    output << "precomputed_holdout_control_point_count: "
           << report.precomputed_holdout_outer_point_count +
                  report.precomputed_holdout_internal_point_count
           << "\n";
  } else {
    output << "precomputed_holdout_outer_point_count: "
           << report.precomputed_holdout_outer_point_count << "\n";
    output << "precomputed_holdout_internal_point_count: "
           << report.precomputed_holdout_internal_point_count << "\n";
  }
  output << "precomputed_boards_rt_used_to_initialize_layout: "
         << (report.precomputed_boards_rt_used_to_initialize_layout ? 1 : 0)
         << "\n";
  output << "external_holdout_observation_source: "
         << report.external_holdout_observation_source << "\n";
  output << "external_holdout_self_frontend_prepass_used: "
         << (report.external_holdout_self_frontend_prepass_used ? 1 : 0)
         << "\n";
  output << "external_holdout_self_frontend_prepass_success: "
         << (report.external_holdout_self_frontend_prepass_success ? 1 : 0)
         << "\n";
  output << "external_holdout_self_frontend_prepass_failure_reason: "
         << report.external_holdout_self_frontend_prepass_failure_reason
         << "\n";
  output << "external_holdout_self_frontend_prepass_seconds: "
         << report.runtime_breakdown.external_holdout_self_frontend_prepass_seconds
         << "\n";
  output << "fair_protocol_matched: " << (report.fair_protocol_matched ? 1 : 0) << "\n";
  output << "diagnostic_only: " << (report.diagnostic_only ? 1 : 0) << "\n";
  output << "kalibr_camera_model_family: " << report.kalibr_reference.camera_model_family
         << "\n";
  output << "kalibr_training_split_signature: "
         << report.kalibr_reference.training_split_signature << "\n";
  output << "kalibr_source_label: " << report.kalibr_reference.source_label << "\n";
  output << "additional_reference_camera_count: "
         << report.additional_camera_references.size() << "\n";
  for (std::size_t index = 0;
       index < report.additional_camera_references.size();
       ++index) {
    const KalibrBenchmarkReference& reference =
        report.additional_camera_references[index];
    const std::string prefix =
        "additional_reference_" + std::to_string(index) + "_";
    output << prefix << "source_label: " << reference.source_label << "\n";
    output << prefix << "camchain_yaml: " << reference.camchain_yaml << "\n";
    output << prefix << "training_split_signature: "
           << reference.training_split_signature << "\n";
  }
  output << "internal_joint_refine_mode: "
         << ToString(report.internal_joint_refine_result.options.mode)
         << "\n";
  output << "internal_joint_refine_target_mode: "
         << ToString(report.internal_joint_refine_result.options.target_mode)
         << "\n";
  output << "internal_joint_refine_backend_input_changed: "
         << (report.internal_joint_refine_result.backend_input_changed ? 1 : 0)
         << "\n";
  output << "internal_joint_refine_candidate_board_count: "
         << report.internal_joint_refine_result.candidate_board_count << "\n";
  output << "internal_joint_refine_accepted_board_count: "
         << report.internal_joint_refine_result.accepted_board_count << "\n";
  output << "internal_joint_refine_rolled_back_board_count: "
         << report.internal_joint_refine_result.rolled_back_board_count << "\n";
  output << "internal_joint_refine_eligible_internal_point_count: "
         << report.internal_joint_refine_result.eligible_internal_point_count
         << "\n";
  output << "internal_joint_refine_refined_internal_point_count: "
         << report.internal_joint_refine_result.refined_internal_point_count
         << "\n";
  output << "internal_joint_refine_mean_displacement_px: "
         << report.internal_joint_refine_result.mean_displacement_px << "\n";
  output << "internal_blur_board_weight_mode: "
         << ToString(report.internal_blur_board_weight_result.options.mode)
         << "\n";
  output << "internal_blur_board_weight_backend_input_changed: "
         << (report.internal_blur_board_weight_result.backend_input_changed ? 1 : 0)
         << "\n";
  output << "internal_blur_board_weight_low_patch_gradient_quantile: "
         << report.internal_blur_board_weight_result.options
                .low_patch_gradient_quantile
         << "\n";
  output << "internal_blur_board_weight_patch_gradient_threshold: "
         << report.internal_blur_board_weight_result.patch_gradient_threshold
         << "\n";
  output << "internal_blur_board_weight_downweighted_board_observation_count: "
         << report.internal_blur_board_weight_result
                .downweighted_board_observation_count
         << "\n";
  output << "internal_blur_board_weight_downweighted_internal_point_count: "
         << report.internal_blur_board_weight_result
                .downweighted_internal_point_count
         << "\n";
  output << "internal_blur_board_weight_mean_weight: "
         << report.internal_blur_board_weight_result.mean_weight << "\n";
  output << "internal_observation_weight_mode: "
         << ToString(report.internal_observation_weight_result.options.mode)
         << "\n";
  output << "internal_observation_weight_policy: "
         << report.internal_observation_weight_result.policy << "\n";
  output << "internal_observation_weight_backend_input_changed: "
         << (report.internal_observation_weight_result.backend_input_changed ? 1 : 0)
         << "\n";
  output << "internal_observation_weight_low_quality_quantile: "
         << report.internal_observation_weight_result.options.low_quality_quantile
         << "\n";
  output << "internal_observation_weight_quality_threshold: "
         << report.internal_observation_weight_result.quality_threshold << "\n";
  output << "internal_observation_weight_residual_consistency_ratio_threshold: "
         << report.internal_observation_weight_result
                .residual_consistency_ratio_threshold
         << "\n";
  output << "internal_observation_weight_downweighted_internal_point_count: "
         << report.internal_observation_weight_result
                .downweighted_internal_point_count
         << "\n";
  output << "internal_observation_weight_mean_weight: "
         << report.internal_observation_weight_result.mean_weight << "\n";
  output << "comparison_scope: output_level_and_evaluator_level_only\n";
  output << "evaluation_protocol: "
         << (report.precomputed_single_board_ba_mode
                 ? "checkerboard_all_control_points_pose_refit_and_reprojection"
                 : "camera_only_outer_refit_pose_plus_outer_internal_reprojection")
         << "\n";
  output << "camera_aware_outer_rescue_attempted_board_count: "
         << report.holdout_dataset
                .camera_aware_outer_rescue_attempted_board_count
         << "\n";
  output << "camera_aware_outer_rescue_used_board_count: "
         << report.holdout_dataset.camera_aware_outer_rescue_used_board_count
         << "\n";
  for (const std::string& warning : report.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteEvaluationCameraSummary(std::ostream& output,
                                  const std::string& prefix,
                                  const CameraModelRefitEvaluationResult& evaluation) {
  const OuterBootstrapCameraIntrinsics& camera = evaluation.camera;
  output << prefix << "camera_model_family: "
         << camera.NormalizedFamilyString() << "\n";
  output << prefix << "camera_model: " << camera.NormalizedCameraModel() << "\n";
  output << prefix << "distortion_model: "
         << camera.NormalizedDistortionModel() << "\n";
  output << prefix << "camera_intrinsics_labels: "
         << JoinStrings(camera.IntrinsicsLabels(), ",") << "\n";
  output << prefix << "camera_intrinsics_csv: ";
  const std::vector<double> intrinsics = camera.IntrinsicsVector();
  for (std::size_t index = 0; index < intrinsics.size(); ++index) {
    if (index > 0) {
      output << ",";
    }
    output << intrinsics[index];
  }
  output << "\n";
  output << prefix << "camera_distortion_labels: "
         << JoinStrings(camera.DistortionLabels(), ",") << "\n";
  output << prefix << "camera_distortion_csv: ";
  const std::vector<double> distortion = camera.DistortionVector();
  for (std::size_t index = 0; index < distortion.size(); ++index) {
    if (index > 0) {
      output << ",";
    }
    output << distortion[index];
  }
  output << "\n";
}

void WriteStage5BenchmarkTrainingSummary(const std::string& path,
                                         const Stage5BenchmarkReport& report) {
  std::ofstream output(path.c_str());
  output << "split_label: training\n";
  output << "split_signature: " << report.split_signature << "\n";
  WriteEvaluationCameraSummary(output, "our_", report.our_training_evaluation);
  WriteEvaluationCameraSummary(output, "kalibr_", report.kalibr_training_evaluation);
  output << "our_overall_rmse: " << report.our_training_evaluation.overall_rmse << "\n";
  output << "our_p95_reprojection_error: "
         << report.our_training_evaluation.p95_reprojection_error << "\n";
  WriteEvaluationCameraSummary(
      output, "initialization_only_",
      report.initialization_training_evaluation);
  output << "initialization_only_overall_rmse: "
         << report.initialization_training_evaluation.overall_rmse << "\n";
  output << "initialization_only_p95_reprojection_error: "
         << report.initialization_training_evaluation.p95_reprojection_error
         << "\n";
  if (!report.precomputed_single_board_ba_mode) {
    output << "our_outer_only_rmse: " << report.our_training_evaluation.outer_only_rmse << "\n";
    output << "our_internal_only_rmse: " << report.our_training_evaluation.internal_only_rmse
           << "\n";
    output << "our_overall_rmse_excluding_board"
           << report.our_training_evaluation.excluded_board_id_for_rmse << ": "
           << report.our_training_evaluation.overall_rmse_excluding_board << "\n";
    output << "our_internal_only_rmse_excluding_board"
           << report.our_training_evaluation.excluded_board_id_for_rmse << ": "
           << report.our_training_evaluation.internal_only_rmse_excluding_board
           << "\n";
  }
  output << "kalibr_overall_rmse: " << report.kalibr_training_evaluation.overall_rmse << "\n";
  output << "kalibr_p95_reprojection_error: "
         << report.kalibr_training_evaluation.p95_reprojection_error << "\n";
  if (!report.precomputed_single_board_ba_mode) {
    output << "kalibr_outer_only_rmse: " << report.kalibr_training_evaluation.outer_only_rmse
           << "\n";
    output << "kalibr_internal_only_rmse: "
           << report.kalibr_training_evaluation.internal_only_rmse << "\n";
    output << "kalibr_overall_rmse_excluding_board"
           << report.kalibr_training_evaluation.excluded_board_id_for_rmse << ": "
           << report.kalibr_training_evaluation.overall_rmse_excluding_board << "\n";
    output << "kalibr_internal_only_rmse_excluding_board"
           << report.kalibr_training_evaluation.excluded_board_id_for_rmse << ": "
           << report.kalibr_training_evaluation.internal_only_rmse_excluding_board
           << "\n";
  }
  WriteKalibrStyleResidualStatistics(output, "our", report.our_training_evaluation);
  WriteKalibrStyleResidualStatistics(output, "kalibr", report.kalibr_training_evaluation);
  output << "our_point_count: " << report.our_training_evaluation.point_count << "\n";
  if (!report.precomputed_single_board_ba_mode) {
    output << "our_point_count_excluding_board"
           << report.our_training_evaluation.excluded_board_id_for_rmse << ": "
           << report.our_training_evaluation.point_count_excluding_board << "\n";
  }
  output << "kalibr_point_count: " << report.kalibr_training_evaluation.point_count << "\n";
  if (!report.precomputed_single_board_ba_mode) {
    output << "kalibr_point_count_excluding_board"
           << report.kalibr_training_evaluation.excluded_board_id_for_rmse << ": "
           << report.kalibr_training_evaluation.point_count_excluding_board << "\n";
  }
  output << "additional_reference_camera_count: "
         << report.additional_training_evaluations.size() << "\n";
  for (std::size_t index = 0;
       index < report.additional_training_evaluations.size();
       ++index) {
    const CameraModelRefitEvaluationResult& evaluation =
        report.additional_training_evaluations[index];
    const std::string prefix =
        "reference_" + SanitizeMetricKey(evaluation.method_label) + "_";
    WriteEvaluationCameraSummary(output, prefix, evaluation);
    output << prefix << "overall_rmse: " << evaluation.overall_rmse << "\n";
    output << prefix << "p95_reprojection_error: "
           << evaluation.p95_reprojection_error << "\n";
    output << prefix << "outer_only_rmse: " << evaluation.outer_only_rmse << "\n";
    output << prefix << "internal_only_rmse: "
           << evaluation.internal_only_rmse << "\n";
    output << prefix << "overall_rmse_excluding_board"
           << evaluation.excluded_board_id_for_rmse << ": "
           << evaluation.overall_rmse_excluding_board << "\n";
    output << prefix << "internal_only_rmse_excluding_board"
           << evaluation.excluded_board_id_for_rmse << ": "
           << evaluation.internal_only_rmse_excluding_board << "\n";
    WriteKalibrStyleResidualStatistics(output,
                                       prefix.substr(0, prefix.size() - 1),
                                       evaluation);
    output << prefix << "point_count: " << evaluation.point_count << "\n";
    output << prefix << "point_count_excluding_board"
           << evaluation.excluded_board_id_for_rmse << ": "
           << evaluation.point_count_excluding_board << "\n";
  }
}

void WriteStage5BenchmarkHoldoutSummary(const std::string& path,
                                        const Stage5BenchmarkReport& report) {
  std::ofstream output(path.c_str());
  output << "split_label: holdout\n";
  output << "split_signature: " << report.split_signature << "\n";
  output << "evaluation_protocol: "
         << (report.precomputed_single_board_ba_mode
                 ? "checkerboard_all_control_points_pose_refit_and_reprojection"
                 : "outer_pose_refit_plus_all_point_reprojection")
         << "\n";
  output << "camera_aware_outer_rescue_attempted_board_count: "
         << report.holdout_dataset
                .camera_aware_outer_rescue_attempted_board_count
         << "\n";
  output << "camera_aware_outer_rescue_used_board_count: "
         << report.holdout_dataset.camera_aware_outer_rescue_used_board_count
         << "\n";
  WriteEvaluationCameraSummary(output, "our_", report.our_holdout_evaluation);
  WriteEvaluationCameraSummary(output, "kalibr_", report.kalibr_holdout_evaluation);
  output << "our_overall_rmse: " << report.our_holdout_evaluation.overall_rmse << "\n";
  output << "our_p95_reprojection_error: "
         << report.our_holdout_evaluation.p95_reprojection_error << "\n";
  WriteEvaluationCameraSummary(
      output, "initialization_only_",
      report.initialization_holdout_evaluation);
  output << "initialization_only_overall_rmse: "
         << report.initialization_holdout_evaluation.overall_rmse << "\n";
  output << "initialization_only_p95_reprojection_error: "
         << report.initialization_holdout_evaluation.p95_reprojection_error
         << "\n";
  if (report.precomputed_single_board_ba_mode) {
    output << "our_test_rmse_all_control_points: "
           << report.our_holdout_evaluation.overall_rmse << "\n";
  }
  const std::string pose_prefix = report.precomputed_single_board_ba_mode
                                      ? "test_pose_refit_"
                                      : "pose_only_refit_";
  output << "our_" << pose_prefix << "rmse: "
         << report.our_holdout_evaluation.pose_only_refit_rmse << "\n";
  output << "our_" << pose_prefix << "success_rate: "
         << report.our_holdout_evaluation.pose_only_refit_success_rate << "\n";
  output << "our_" << pose_prefix << "attempt_count: "
         << report.our_holdout_evaluation.pose_only_refit_attempt_count << "\n";
  output << "our_" << pose_prefix << "success_count: "
         << report.our_holdout_evaluation.pose_only_refit_success_count << "\n";
  if (!report.precomputed_single_board_ba_mode) {
    output << "our_outer_only_rmse: " << report.our_holdout_evaluation.outer_only_rmse << "\n";
    output << "our_internal_only_rmse: " << report.our_holdout_evaluation.internal_only_rmse
           << "\n";
    output << "our_overall_rmse_excluding_board"
           << report.our_holdout_evaluation.excluded_board_id_for_rmse << ": "
           << report.our_holdout_evaluation.overall_rmse_excluding_board << "\n";
    output << "our_internal_only_rmse_excluding_board"
           << report.our_holdout_evaluation.excluded_board_id_for_rmse << ": "
           << report.our_holdout_evaluation.internal_only_rmse_excluding_board
           << "\n";
  }
  output << "kalibr_overall_rmse: " << report.kalibr_holdout_evaluation.overall_rmse << "\n";
  output << "kalibr_p95_reprojection_error: "
         << report.kalibr_holdout_evaluation.p95_reprojection_error << "\n";
  if (report.precomputed_single_board_ba_mode) {
    output << "kalibr_test_rmse_all_control_points: "
           << report.kalibr_holdout_evaluation.overall_rmse << "\n";
  }
  output << "kalibr_" << pose_prefix << "rmse: "
         << report.kalibr_holdout_evaluation.pose_only_refit_rmse << "\n";
  output << "kalibr_" << pose_prefix << "success_rate: "
         << report.kalibr_holdout_evaluation.pose_only_refit_success_rate
         << "\n";
  output << "kalibr_" << pose_prefix << "attempt_count: "
         << report.kalibr_holdout_evaluation.pose_only_refit_attempt_count
         << "\n";
  output << "kalibr_" << pose_prefix << "success_count: "
         << report.kalibr_holdout_evaluation.pose_only_refit_success_count
         << "\n";
  if (!report.precomputed_single_board_ba_mode) {
    output << "kalibr_outer_only_rmse: " << report.kalibr_holdout_evaluation.outer_only_rmse
           << "\n";
    output << "kalibr_internal_only_rmse: "
           << report.kalibr_holdout_evaluation.internal_only_rmse << "\n";
    output << "kalibr_overall_rmse_excluding_board"
           << report.kalibr_holdout_evaluation.excluded_board_id_for_rmse << ": "
           << report.kalibr_holdout_evaluation.overall_rmse_excluding_board << "\n";
    output << "kalibr_internal_only_rmse_excluding_board"
           << report.kalibr_holdout_evaluation.excluded_board_id_for_rmse << ": "
           << report.kalibr_holdout_evaluation.internal_only_rmse_excluding_board
           << "\n";
  }
  WriteKalibrStyleResidualStatistics(output, "our", report.our_holdout_evaluation);
  WriteKalibrStyleResidualStatistics(output, "kalibr", report.kalibr_holdout_evaluation);
  output << "our_point_count: " << report.our_holdout_evaluation.point_count << "\n";
  output << "our_point_count_excluding_board"
         << report.our_holdout_evaluation.excluded_board_id_for_rmse << ": "
         << report.our_holdout_evaluation.point_count_excluding_board << "\n";
  output << "kalibr_point_count: " << report.kalibr_holdout_evaluation.point_count << "\n";
  output << "kalibr_point_count_excluding_board"
         << report.kalibr_holdout_evaluation.excluded_board_id_for_rmse << ": "
         << report.kalibr_holdout_evaluation.point_count_excluding_board << "\n";
  output << "additional_reference_camera_count: "
         << report.additional_holdout_evaluations.size() << "\n";
  output << "camera_ray_curve_success: "
         << (report.camera_ray_curve_diagnostics.success ? 1 : 0) << "\n";
  output << "camera_ray_curve_comparison_count: "
         << report.camera_ray_curve_diagnostics.comparison_count << "\n";
  output << "camera_ray_curve_sample_count: "
         << report.camera_ray_curve_diagnostics.sample_count << "\n";
  output << "camera_ray_curve_invalid_unprojection_count: "
         << report.camera_ray_curve_diagnostics.invalid_unprojection_count
         << "\n";
  output << "camera_ray_curve_our_camera_source: "
         << report.camera_ray_curve_diagnostics.our_camera_source << "\n";
  output << "camera_ray_curve_our_camera_family: "
         << report.camera_ray_curve_diagnostics.our_camera
                .NormalizedFamilyString()
         << "\n";
  output << "camera_ray_curve_our_camera_labels: "
         << JoinStrings(report.camera_ray_curve_diagnostics.our_camera
                            .CombinedParameterLabels(),
                        ",")
         << "\n";
  output << "camera_ray_curve_our_camera_csv: "
         << JoinDoubles(report.camera_ray_curve_diagnostics.our_camera
                            .CombinedParameterVector(),
                        ",")
         << "\n";
  if (!report.camera_ray_curve_diagnostics.failure_reason.empty()) {
    output << "camera_ray_curve_failure_reason: "
           << report.camera_ray_curve_diagnostics.failure_reason << "\n";
  }
  for (std::size_t index = 0;
       index < report.additional_holdout_evaluations.size();
       ++index) {
    const CameraModelRefitEvaluationResult& evaluation =
        report.additional_holdout_evaluations[index];
    const std::string prefix =
        "reference_" + SanitizeMetricKey(evaluation.method_label) + "_";
    WriteEvaluationCameraSummary(output, prefix, evaluation);
    output << prefix << "overall_rmse: " << evaluation.overall_rmse << "\n";
    output << prefix << "p95_reprojection_error: "
           << evaluation.p95_reprojection_error << "\n";
    if (report.precomputed_single_board_ba_mode) {
      output << prefix << "test_rmse_all_control_points: "
             << evaluation.overall_rmse << "\n";
    }
    output << prefix << pose_prefix << "rmse: "
           << evaluation.pose_only_refit_rmse << "\n";
    output << prefix << pose_prefix << "success_rate: "
           << evaluation.pose_only_refit_success_rate << "\n";
    output << prefix << pose_prefix << "attempt_count: "
           << evaluation.pose_only_refit_attempt_count << "\n";
    output << prefix << pose_prefix << "success_count: "
           << evaluation.pose_only_refit_success_count << "\n";
    if (!report.precomputed_single_board_ba_mode) {
      output << prefix << "outer_only_rmse: " << evaluation.outer_only_rmse << "\n";
      output << prefix << "internal_only_rmse: "
             << evaluation.internal_only_rmse << "\n";
      output << prefix << "overall_rmse_excluding_board"
             << evaluation.excluded_board_id_for_rmse << ": "
             << evaluation.overall_rmse_excluding_board << "\n";
      output << prefix << "internal_only_rmse_excluding_board"
             << evaluation.excluded_board_id_for_rmse << ": "
             << evaluation.internal_only_rmse_excluding_board << "\n";
    }
    WriteKalibrStyleResidualStatistics(output, prefix.substr(0, prefix.size() - 1),
                                       evaluation);
    output << prefix << "point_count: " << evaluation.point_count << "\n";
    output << prefix << "point_count_excluding_board"
           << evaluation.excluded_board_id_for_rmse << ": "
           << evaluation.point_count_excluding_board << "\n";
  }
}

void WritePersistentCameraCheckpointEvaluationsCsv(
    const std::string& path,
    const Stage5BenchmarkReport& report) {
  std::ofstream output(path.c_str());
  output << "checkpoint,attempt_order,frame_index,frame_label,information_gain,"
         << "xi,alpha,fu,fv,cu,cv,train_rmse,train_p95,"
         << "train_frame_median_rmse,train_frame_p90_rmse,"
         << "train_trimmed90_rmse,train_huber15_rmse,"
         << "train_fold_median_mean_rmse,train_fold_median_max_rmse,"
         << "train_fold_median_std_rmse,test_rmse,test_p95,"
         << "train_pose_success_rate,test_pose_success_rate\n";
  const auto write_row = [&output](
      const std::string& checkpoint_label,
      int attempt_order,
      int frame_index,
      const std::string& frame_label,
      double information_gain,
      const OuterBootstrapCameraIntrinsics& camera,
      const CameraModelRefitEvaluationResult& training,
      const CameraModelRefitEvaluationResult& holdout) {
    const EvaluationRobustSummary training_robust =
        SummarizeEvaluationRobustness(training);
    output << checkpoint_label << "," << attempt_order << ","
           << frame_index << "," << frame_label << ","
           << information_gain << "," << camera.xi << ","
           << camera.alpha << "," << camera.fu << "," << camera.fv
           << "," << camera.cu << "," << camera.cv << ","
           << training.overall_rmse << ","
           << training.p95_reprojection_error << ","
           << training_robust.frame_median_rmse << ","
           << training_robust.frame_p90_rmse << ","
           << training_robust.trimmed90_rmse << ","
           << training_robust.huber15_rmse << ","
           << training_robust.fold_median_mean_rmse << ","
           << training_robust.fold_median_max_rmse << ","
           << training_robust.fold_median_std_rmse << ","
           << holdout.overall_rmse << ","
           << holdout.p95_reprojection_error << ","
           << training.pose_only_refit_success_rate << ","
           << holdout.pose_only_refit_success_rate << "\n";
  };
  write_row("initialization_only", -1, -1, "", 0.0,
            report.initialization_holdout_evaluation.camera,
            report.initialization_training_evaluation,
            report.initialization_holdout_evaluation);
  if (report.large_intrinsic_perturbation.enabled) {
    write_row("perturbation_boundary", -2, -1, "", 0.0,
              report.large_intrinsic_perturbation.perturbed_camera,
              report.perturbation_boundary_training_evaluation,
              report.perturbation_boundary_holdout_evaluation);
  }
  for (const PersistentCameraCheckpointEvaluation& checkpoint :
       report.persistent_camera_checkpoint_evaluations) {
    write_row("accepted_batch", checkpoint.attempt_order,
              checkpoint.frame_index, checkpoint.frame_label,
              checkpoint.information_gain, checkpoint.camera,
              checkpoint.training_evaluation,
              checkpoint.holdout_evaluation);
  }
}

void WriteStage5BenchmarkHoldoutPointsCsv(const std::string& path,
                                          const Stage5BenchmarkReport& report) {
  std::vector<CameraModelRefitEvaluationResult> evaluations{
      report.our_holdout_evaluation,
      report.kalibr_holdout_evaluation};
  evaluations.insert(evaluations.end(),
                     report.additional_holdout_evaluations.begin(),
                     report.additional_holdout_evaluations.end());
  WriteCameraModelRefitPointsCsv(path, evaluations);
}

void WriteStage5BenchmarkHoldoutBoardObservationsCsv(
    const std::string& path,
    const Stage5BenchmarkReport& report) {
  std::vector<CameraModelRefitEvaluationResult> evaluations{
      report.our_holdout_evaluation,
      report.kalibr_holdout_evaluation};
  evaluations.insert(evaluations.end(),
                     report.additional_holdout_evaluations.begin(),
                     report.additional_holdout_evaluations.end());
  WriteCameraModelRefitBoardObservationsCsv(path, evaluations);
}

void WriteStage5BenchmarkHoldoutFramesCsv(const std::string& path,
                                          const Stage5BenchmarkReport& report) {
  std::vector<CameraModelRefitEvaluationResult> evaluations{
      report.our_holdout_evaluation,
      report.kalibr_holdout_evaluation};
  evaluations.insert(evaluations.end(),
                     report.additional_holdout_evaluations.begin(),
                     report.additional_holdout_evaluations.end());
  WriteCameraModelRefitFramesCsv(path, evaluations);
}

void WriteCameraRayCurveSamplesCsv(
    const std::string& path,
    const CameraRayCurveDiagnostics& diagnostics) {
  std::ofstream output(path.c_str());
  output << "reference_label,reference_family,image_x,image_y,radial_fraction,"
         << "our_polar_deg,reference_polar_deg,angular_diff_deg\n";
  for (const CameraRayCurveSample& sample : diagnostics.samples) {
    output << sample.reference_label << ","
           << sample.reference_family << ","
           << sample.image_x << ","
           << sample.image_y << ","
           << sample.radial_fraction << ","
           << sample.our_polar_deg << ","
           << sample.reference_polar_deg << ","
           << sample.angular_diff_deg << "\n";
  }
}

void WriteCameraRayCurveSummaryCsv(
    const std::string& path,
    const CameraRayCurveDiagnostics& diagnostics) {
  std::ofstream output(path.c_str());
  output << "our_camera_source,our_camera_family,our_camera_labels,"
         << "our_camera_csv,reference_label,reference_family,bucket_type,bucket_label,"
         << "sample_count,mean_angular_diff_deg,rms_angular_diff_deg,"
         << "max_angular_diff_deg,mean_our_polar_deg,mean_reference_polar_deg\n";
  for (const CameraRayCurveBucketSummary& summary :
       diagnostics.bucket_summaries) {
    output << EscapeCsvCell(diagnostics.our_camera_source) << ","
           << diagnostics.our_camera.NormalizedFamilyString() << ","
           << EscapeCsvCell(JoinStrings(
                  diagnostics.our_camera.CombinedParameterLabels(), ";"))
           << ","
           << EscapeCsvCell(JoinDoubles(
                  diagnostics.our_camera.CombinedParameterVector(), ";"))
           << ","
           << summary.reference_label << ","
           << summary.reference_family << ","
           << summary.bucket_type << ","
           << summary.bucket_label << ","
           << summary.sample_count << ","
           << summary.mean_angular_diff_deg << ","
           << summary.rms_angular_diff_deg << ","
           << summary.max_angular_diff_deg << ","
           << summary.mean_our_polar_deg << ","
           << summary.mean_reference_polar_deg << "\n";
  }
}

void WriteCameraModelRefitPointsCsv(
    const std::string& path,
    const std::vector<CameraModelRefitEvaluationResult>& evaluations) {
  std::ofstream output(path.c_str());
  output << "method,split,frame_index,frame_label,board_id,point_id,point_type,"
         << "observed_x,observed_y,predicted_x,predicted_y,target_x,target_y,target_z,"
         << "residual_x,residual_y,residual_norm,debug_quality,source_kind,source_point_index\n";
  const auto write_points = [&output](const CameraModelRefitEvaluationResult& evaluation) {
    for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
      output << point.method_label << ","
             << point.split_label << ","
             << point.frame_index << ","
             << point.frame_label << ","
             << point.board_id << ","
             << point.point_id << ","
             << ToString(point.point_type) << ","
             << point.observed_image_xy.x() << ","
             << point.observed_image_xy.y() << ","
             << point.predicted_image_xy.x() << ","
             << point.predicted_image_xy.y() << ","
             << point.target_xyz_board.x() << ","
             << point.target_xyz_board.y() << ","
             << point.target_xyz_board.z() << ","
             << point.residual_xy.x() << ","
             << point.residual_xy.y() << ","
             << point.residual_norm << ","
             << point.quality << ","
             << ToString(point.source_kind) << ","
             << point.source_point_index << "\n";
    }
  };
  for (const CameraModelRefitEvaluationResult& evaluation : evaluations) {
    write_points(evaluation);
  }
}

void WriteCameraModelRefitBoardObservationsCsv(
    const std::string& path,
    const std::vector<CameraModelRefitEvaluationResult>& evaluations) {
  std::ofstream output(path.c_str());
  output << "method,split,frame_index,frame_label,board_id,"
         << "pose_only_refit_success,pose_fit_outer_rmse,evaluation_rmse,"
         << "outer_evaluation_rmse,internal_evaluation_rmse,"
         << "all_point_pose_refit_success,all_point_pose_refit_point_count,"
         << "all_point_pose_refit_rmse,all_point_pose_refit_internal_rmse,"
         << "point_count,outer_point_count,internal_point_count,failure_reason\n";
  for (const CameraModelRefitEvaluationResult& evaluation : evaluations) {
    for (const CameraModelRefitBoardObservationDiagnostics& board :
         evaluation.board_observation_diagnostics) {
      output << board.method_label << ","
             << board.split_label << ","
             << board.frame_index << ","
             << board.frame_label << ","
             << board.board_id << ","
             << (board.pose_only_refit_success ? 1 : 0) << ","
             << board.pose_fit_outer_rmse << ","
             << board.evaluation_rmse << ","
             << board.outer_evaluation_rmse << ","
             << board.internal_evaluation_rmse << ","
             << (board.all_point_pose_refit_success ? 1 : 0) << ","
             << board.all_point_pose_refit_point_count << ","
             << board.all_point_pose_refit_rmse << ","
             << board.all_point_pose_refit_internal_rmse << ","
             << board.point_count << ","
             << board.outer_point_count << ","
             << board.internal_point_count << ","
             << board.failure_reason << "\n";
    }
  }
}

void WriteCameraModelRefitFramesCsv(
    const std::string& path,
    const std::vector<CameraModelRefitEvaluationResult>& evaluations) {
  std::ofstream output(path.c_str());
  output << "method,split,frame_index,frame_label,"
         << "pose_only_refit_attempt_count,pose_only_refit_success_count,"
         << "pose_only_refit_success_rate,pose_only_refit_rmse,"
         << "rmse,outer_rmse,internal_rmse,"
         << "point_count,outer_point_count,internal_point_count\n";
  for (const CameraModelRefitEvaluationResult& evaluation : evaluations) {
    for (const CameraModelRefitFrameDiagnostics& frame :
         evaluation.frame_diagnostics) {
      output << frame.method_label << ","
             << frame.split_label << ","
             << frame.frame_index << ","
             << frame.frame_label << ","
             << frame.pose_only_refit_attempt_count << ","
             << frame.pose_only_refit_success_count << ","
             << frame.pose_only_refit_success_rate << ","
             << frame.pose_only_refit_rmse << ","
             << frame.rmse << ","
             << frame.outer_rmse << ","
             << frame.internal_rmse << ","
             << frame.point_count << ","
             << frame.outer_point_count << ","
             << frame.internal_point_count << "\n";
    }
  }
}

void WriteStage5BenchmarkWorstCasesSummary(const std::string& path,
                                           const Stage5BenchmarkReport& report,
                                           int top_k) {
  std::ofstream output(path.c_str());
  output << "dataset_label: " << report.dataset_label << "\n";
  output << "split_signature: " << report.split_signature << "\n";
  output << "top_k: " << top_k << "\n";

  const auto write_eval = [&output, top_k](const CameraModelRefitEvaluationResult& evaluation) {
    output << "\n[" << evaluation.method_label << "_" << evaluation.split_label << "]\n";
    output << "pose_only_refit_rmse: " << evaluation.pose_only_refit_rmse << "\n";
    output << "pose_only_refit_success_rate: "
           << evaluation.pose_only_refit_success_rate << "\n";
    output << "pose_only_refit_attempt_count: "
           << evaluation.pose_only_refit_attempt_count << "\n";
    output << "pose_only_refit_success_count: "
           << evaluation.pose_only_refit_success_count << "\n";
    output << "overall_rmse: " << evaluation.overall_rmse << "\n";
    output << "outer_only_rmse: " << evaluation.outer_only_rmse << "\n";
    output << "internal_only_rmse: " << evaluation.internal_only_rmse << "\n";

    std::vector<CameraModelRefitFrameDiagnostics> worst_frames =
        evaluation.frame_diagnostics;
    std::sort(worst_frames.begin(), worst_frames.end(),
              [](const CameraModelRefitFrameDiagnostics& lhs,
                 const CameraModelRefitFrameDiagnostics& rhs) {
                return lhs.rmse > rhs.rmse;
              });
    if (top_k >= 0 && static_cast<int>(worst_frames.size()) > top_k) {
      worst_frames.resize(static_cast<std::size_t>(top_k));
    }
    output << "worst_frames:\n";
    for (const CameraModelRefitFrameDiagnostics& frame : worst_frames) {
      output << "  frame_index=" << frame.frame_index
             << " frame_label=" << frame.frame_label
             << " rmse=" << frame.rmse
             << " outer_rmse=" << frame.outer_rmse
             << " internal_rmse=" << frame.internal_rmse
             << " pose_only_refit_success_rate="
             << frame.pose_only_refit_success_rate
             << " pose_only_refit_rmse=" << frame.pose_only_refit_rmse
             << " point_count=" << frame.point_count
             << " outer_point_count=" << frame.outer_point_count
             << " internal_point_count=" << frame.internal_point_count << "\n";
    }

    std::vector<CameraModelRefitBoardObservationDiagnostics> worst_boards =
        evaluation.board_observation_diagnostics;
    std::sort(worst_boards.begin(), worst_boards.end(),
              [](const CameraModelRefitBoardObservationDiagnostics& lhs,
                 const CameraModelRefitBoardObservationDiagnostics& rhs) {
                return lhs.evaluation_rmse > rhs.evaluation_rmse;
              });
    if (top_k >= 0 && static_cast<int>(worst_boards.size()) > top_k) {
      worst_boards.resize(static_cast<std::size_t>(top_k));
    }
    output << "worst_board_observations:\n";
    for (const CameraModelRefitBoardObservationDiagnostics& board : worst_boards) {
      output << "  frame_index=" << board.frame_index
             << " frame_label=" << board.frame_label
             << " board_id=" << board.board_id
             << " pose_only_refit_success="
             << (board.pose_only_refit_success ? 1 : 0)
             << " rmse=" << board.evaluation_rmse
             << " pose_fit_outer_rmse=" << board.pose_fit_outer_rmse
             << " outer_rmse=" << board.outer_evaluation_rmse
             << " internal_rmse=" << board.internal_evaluation_rmse
             << " point_count=" << board.point_count
             << " outer_point_count=" << board.outer_point_count
             << " internal_point_count=" << board.internal_point_count
             << " failure_reason=" << board.failure_reason << "\n";
    }
  };

  write_eval(report.our_holdout_evaluation);
  write_eval(report.kalibr_holdout_evaluation);
}

namespace {

struct RmseAccumulator {
  double squared_error_sum = 0.0;
  int point_count = 0;

  void Add(double rmse, int count) {
    if (count <= 0 || !std::isfinite(rmse)) {
      return;
    }
    squared_error_sum += rmse * rmse * static_cast<double>(count);
    point_count += count;
  }

  double Rmse() const {
    return point_count > 0
               ? std::sqrt(squared_error_sum /
                           static_cast<double>(point_count))
               : std::numeric_limits<double>::quiet_NaN();
  }
};

struct RobustHoldoutAggregate {
  RmseAccumulator overall;
  RmseAccumulator outer;
  RmseAccumulator internal;
};

RobustHoldoutAggregate AggregateBoardObservations(
    const std::vector<CameraModelRefitBoardObservationDiagnostics>& boards) {
  RobustHoldoutAggregate aggregate;
  for (const CameraModelRefitBoardObservationDiagnostics& board : boards) {
    aggregate.overall.Add(board.evaluation_rmse, board.point_count);
    aggregate.outer.Add(board.outer_evaluation_rmse, board.outer_point_count);
    aggregate.internal.Add(board.internal_evaluation_rmse,
                           board.internal_point_count);
  }
  return aggregate;
}

double BoardObservationSquaredErrorContribution(
    const CameraModelRefitBoardObservationDiagnostics& board) {
  if (board.point_count <= 0 || !std::isfinite(board.evaluation_rmse)) {
    return 0.0;
  }
  return board.evaluation_rmse * board.evaluation_rmse *
         static_cast<double>(board.point_count);
}

}  // namespace

void WriteStage5BenchmarkHoldoutRobustOutlierSummary(
    const std::string& path,
    const Stage5BenchmarkReport& report,
    double board_outlier_rmse_threshold_px) {
  std::ofstream output(path.c_str());
  output << "dataset_label: " << report.dataset_label << "\n";
  output << "split_signature: " << report.split_signature << "\n";
  output << "board_outlier_rmse_threshold_px: "
         << board_outlier_rmse_threshold_px << "\n";
  output << "purpose: quantify whether holdout RMSE is dominated by a small "
         << "number of high-residual frame-board observations; this diagnostic "
         << "does not change backend selection or optimization.\n";

  const auto write_eval =
      [&output, board_outlier_rmse_threshold_px](
          const CameraModelRefitEvaluationResult& evaluation) {
        output << "\n[" << evaluation.method_label << "_"
               << evaluation.split_label << "]\n";
        output << "overall_rmse: " << evaluation.overall_rmse << "\n";
        output << "outer_only_rmse: " << evaluation.outer_only_rmse << "\n";
        output << "internal_only_rmse: " << evaluation.internal_only_rmse
               << "\n";
        output << "pose_only_refit_rmse: "
               << evaluation.pose_only_refit_rmse << "\n";
        output << "pose_only_refit_success_rate: "
               << evaluation.pose_only_refit_success_rate << "\n";

        std::vector<CameraModelRefitBoardObservationDiagnostics> sorted_boards =
            evaluation.board_observation_diagnostics;
        std::sort(sorted_boards.begin(), sorted_boards.end(),
                  [](const CameraModelRefitBoardObservationDiagnostics& lhs,
                     const CameraModelRefitBoardObservationDiagnostics& rhs) {
                    return BoardObservationSquaredErrorContribution(lhs) >
                           BoardObservationSquaredErrorContribution(rhs);
                  });

        std::vector<CameraModelRefitBoardObservationDiagnostics> inlier_boards;
        std::vector<CameraModelRefitBoardObservationDiagnostics> outlier_boards;
        for (const CameraModelRefitBoardObservationDiagnostics& board :
             evaluation.board_observation_diagnostics) {
          if (std::isfinite(board.evaluation_rmse) &&
              board.evaluation_rmse > board_outlier_rmse_threshold_px) {
            outlier_boards.push_back(board);
          } else {
            inlier_boards.push_back(board);
          }
        }
        const RobustHoldoutAggregate inlier_aggregate =
            AggregateBoardObservations(inlier_boards);
        output << "outlier_board_observation_count: "
               << outlier_boards.size() << "\n";
        output << "inlier_board_observation_count: "
               << inlier_boards.size() << "\n";
        output << "rmse_excluding_outlier_board_observations: "
               << inlier_aggregate.overall.Rmse() << "\n";
        output << "outer_rmse_excluding_outlier_board_observations: "
               << inlier_aggregate.outer.Rmse() << "\n";
        output << "internal_rmse_excluding_outlier_board_observations: "
               << inlier_aggregate.internal.Rmse() << "\n";
        output << "point_count_excluding_outlier_board_observations: "
               << inlier_aggregate.overall.point_count << "\n";
        output << "internal_point_count_excluding_outlier_board_observations: "
               << inlier_aggregate.internal.point_count << "\n";

        const int exclusion_counts[] = {1, 2, 3, 5, 8, 9, 10, 12, 15};
        for (const int exclusion_count : exclusion_counts) {
          std::vector<CameraModelRefitBoardObservationDiagnostics> kept_boards;
          for (std::size_t index = 0; index < sorted_boards.size(); ++index) {
            if (index >= static_cast<std::size_t>(exclusion_count)) {
              kept_boards.push_back(sorted_boards[index]);
            }
          }
          const RobustHoldoutAggregate kept =
              AggregateBoardObservations(kept_boards);
          output << "rmse_excluding_top" << exclusion_count
                 << "_sse_board_observations: " << kept.overall.Rmse()
                 << "\n";
          output << "internal_rmse_excluding_top" << exclusion_count
                 << "_sse_board_observations: " << kept.internal.Rmse()
                 << "\n";
          output << "outer_rmse_excluding_top" << exclusion_count
                 << "_sse_board_observations: " << kept.outer.Rmse()
                 << "\n";
          output << "point_count_excluding_top" << exclusion_count
                 << "_sse_board_observations: "
                 << kept.overall.point_count << "\n";
        }

        output << "outlier_board_observations:\n";
        for (const CameraModelRefitBoardObservationDiagnostics& board :
             outlier_boards) {
          output << "  frame_index=" << board.frame_index
                 << " frame_label=" << board.frame_label
                 << " board_id=" << board.board_id
                 << " rmse=" << board.evaluation_rmse
                 << " outer_rmse=" << board.outer_evaluation_rmse
                 << " internal_rmse=" << board.internal_evaluation_rmse
                 << " pose_fit_outer_rmse=" << board.pose_fit_outer_rmse
                 << " point_count=" << board.point_count
                 << " outer_point_count=" << board.outer_point_count
                 << " internal_point_count=" << board.internal_point_count
                 << "\n";
        }

        output << "per_board_aggregate:\n";
        std::map<int, std::vector<CameraModelRefitBoardObservationDiagnostics> >
            boards_by_id;
        for (const CameraModelRefitBoardObservationDiagnostics& board :
             evaluation.board_observation_diagnostics) {
          boards_by_id[board.board_id].push_back(board);
        }
        for (const auto& entry : boards_by_id) {
          const RobustHoldoutAggregate aggregate =
              AggregateBoardObservations(entry.second);
          output << "  board_id=" << entry.first
                 << " observation_count=" << entry.second.size()
                 << " overall_rmse=" << aggregate.overall.Rmse()
                 << " outer_rmse=" << aggregate.outer.Rmse()
                 << " internal_rmse=" << aggregate.internal.Rmse()
                 << " point_count=" << aggregate.overall.point_count
                 << " internal_point_count=" << aggregate.internal.point_count
                 << "\n";
        }
      };

  write_eval(report.our_holdout_evaluation);
  write_eval(report.kalibr_holdout_evaluation);
  for (const CameraModelRefitEvaluationResult& evaluation :
       report.additional_holdout_evaluations) {
    write_eval(evaluation);
  }
}

void WriteMultiBoardConsistencyPerObservationCsv(
    const std::string& path,
    const MultiBoardConsistencyDiagnosticsResult& result) {
  std::ofstream output(path.c_str());
  output << "frame_id,frame_label,board_id,used_in_backend,"
         << "local_pose_refit_success,outer_point_count,internal_point_count,"
         << "local_outer_rmse,global_reprojection_rmse,translation_error_mm,"
         << "rotation_error_deg,polar_angle_mean_deg,polar_angle_max_deg,"
         << "residual_rmse,outer_rmse,internal_rmse,failure_reason\n";
  for (const MultiBoardConsistencyObservationDiagnostics& row :
       result.observation_diagnostics) {
    output << row.frame_id << ","
           << row.frame_label << ","
           << row.board_id << ","
           << (row.used_in_backend ? 1 : 0) << ","
           << (row.local_pose_refit_success ? 1 : 0) << ","
           << row.outer_point_count << ","
           << row.internal_point_count << ","
           << row.local_outer_rmse << ","
           << row.global_reprojection_rmse << ","
           << row.translation_error_mm << ","
           << row.rotation_error_deg << ","
           << row.polar_angle_mean_deg << ","
           << row.polar_angle_max_deg << ","
           << row.residual_rmse << ","
           << row.outer_rmse << ","
           << row.internal_rmse << ","
           << row.failure_reason << "\n";
  }
}

void WriteMultiBoardConsistencyPerBoardCsv(
    const std::string& path,
    const MultiBoardConsistencyDiagnosticsResult& result) {
  std::ofstream output(path.c_str());
  output << "board_id,support_observation_count,mean_translation_error_mm,"
         << "median_translation_error_mm,p90_translation_error_mm,"
         << "max_translation_error_mm,mean_rotation_error_deg,"
         << "median_rotation_error_deg,p90_rotation_error_deg,"
         << "max_rotation_error_deg,worst_frame_id\n";
  for (const MultiBoardConsistencyBoardDiagnostics& row :
       result.board_diagnostics) {
    output << row.board_id << ","
           << row.support_observation_count << ","
           << row.mean_translation_error_mm << ","
           << row.median_translation_error_mm << ","
           << row.p90_translation_error_mm << ","
           << row.max_translation_error_mm << ","
           << row.mean_rotation_error_deg << ","
           << row.median_rotation_error_deg << ","
           << row.p90_rotation_error_deg << ","
           << row.max_rotation_error_deg << ","
           << row.worst_frame_id << "\n";
  }
}

void WriteMultiBoardConsistencyPerFrameCsv(
    const std::string& path,
    const MultiBoardConsistencyDiagnosticsResult& result) {
  std::ofstream output(path.c_str());
  output << "frame_id,observed_board_count,mean_translation_error_mm,"
         << "max_translation_error_mm,mean_rotation_error_deg,"
         << "max_rotation_error_deg,worst_board_id,frame_reprojection_rmse\n";
  for (const MultiBoardConsistencyFrameDiagnostics& row :
       result.frame_diagnostics) {
    output << row.frame_id << ","
           << row.observed_board_count << ","
           << row.mean_translation_error_mm << ","
           << row.max_translation_error_mm << ","
           << row.mean_rotation_error_deg << ","
           << row.max_rotation_error_deg << ","
           << row.worst_board_id << ","
           << row.frame_reprojection_rmse << "\n";
  }
}

void WriteMultiBoardConsistencySummary(
    const std::string& path,
    const MultiBoardConsistencyDiagnosticsResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "training_only_diagnostics: " << (result.training_only ? 1 : 0) << "\n";
  output << "split_label: " << result.split_label << "\n";
  output << "pose_source: " << result.pose_source_label << "\n";
  output << "optimized_intrinsics_fixed: "
         << (result.optimized_intrinsics_fixed ? 1 : 0) << "\n";
  output << "translation_error_unit_assumption: tag_size_meters_to_mm\n";
  output << "outer_only_pose_interpretation_boundary: 1\n";
  output << "frame_count: " << result.frame_count << "\n";
  output << "board_observation_count: " << result.board_observation_count << "\n";
  output << "successful_local_pose_refit_count: "
         << result.successful_local_pose_refit_count << "\n";

  std::vector<MultiBoardConsistencyObservationDiagnostics> worst_translation =
      result.observation_diagnostics;
  std::sort(worst_translation.begin(), worst_translation.end(),
            [](const MultiBoardConsistencyObservationDiagnostics& lhs,
               const MultiBoardConsistencyObservationDiagnostics& rhs) {
              return lhs.translation_error_mm > rhs.translation_error_mm;
            });
  std::vector<MultiBoardConsistencyObservationDiagnostics> worst_rotation =
      result.observation_diagnostics;
  std::sort(worst_rotation.begin(), worst_rotation.end(),
            [](const MultiBoardConsistencyObservationDiagnostics& lhs,
               const MultiBoardConsistencyObservationDiagnostics& rhs) {
              return lhs.rotation_error_deg > rhs.rotation_error_deg;
            });

  output << "\n[board_rankings]\n";
  for (const MultiBoardConsistencyBoardDiagnostics& board : result.board_diagnostics) {
    output << "board_id=" << board.board_id
           << " mean_translation_error_mm=" << board.mean_translation_error_mm
           << " mean_rotation_error_deg=" << board.mean_rotation_error_deg
           << " max_translation_error_mm=" << board.max_translation_error_mm
           << " max_rotation_error_deg=" << board.max_rotation_error_deg
           << " support_observation_count=" << board.support_observation_count
           << " worst_frame_id=" << board.worst_frame_id << "\n";
  }

  output << "\n[frame_rankings]\n";
  for (const MultiBoardConsistencyFrameDiagnostics& frame : result.frame_diagnostics) {
    output << "frame_id=" << frame.frame_id
           << " mean_translation_error_mm=" << frame.mean_translation_error_mm
           << " mean_rotation_error_deg=" << frame.mean_rotation_error_deg
           << " max_translation_error_mm=" << frame.max_translation_error_mm
           << " max_rotation_error_deg=" << frame.max_rotation_error_deg
           << " worst_board_id=" << frame.worst_board_id
           << " frame_reprojection_rmse=" << frame.frame_reprojection_rmse << "\n";
  }

  output << "\n[worst_observations_by_translation]\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(10, worst_translation.size()); ++i) {
    const MultiBoardConsistencyObservationDiagnostics& row = worst_translation[i];
    output << "rank=" << (i + 1)
           << " frame_id=" << row.frame_id
           << " board_id=" << row.board_id
           << " translation_error_mm=" << row.translation_error_mm
           << " rotation_error_deg=" << row.rotation_error_deg
           << " residual_rmse=" << row.residual_rmse
           << " polar_angle_mean_deg=" << row.polar_angle_mean_deg
           << " local_pose_refit_success=" << (row.local_pose_refit_success ? 1 : 0)
           << "\n";
  }

  output << "\n[worst_observations_by_rotation]\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(10, worst_rotation.size()); ++i) {
    const MultiBoardConsistencyObservationDiagnostics& row = worst_rotation[i];
    output << "rank=" << (i + 1)
           << " frame_id=" << row.frame_id
           << " board_id=" << row.board_id
           << " translation_error_mm=" << row.translation_error_mm
           << " rotation_error_deg=" << row.rotation_error_deg
           << " residual_rmse=" << row.residual_rmse
           << " polar_angle_mean_deg=" << row.polar_angle_mean_deg
           << " local_pose_refit_success=" << (row.local_pose_refit_success ? 1 : 0)
           << "\n";
  }

  auto find_board = [&result](int board_id)
      -> const MultiBoardConsistencyBoardDiagnostics* {
    for (const MultiBoardConsistencyBoardDiagnostics& board :
         result.board_diagnostics) {
      if (board.board_id == board_id) {
        return &board;
      }
    }
    return nullptr;
  };
  const MultiBoardConsistencyBoardDiagnostics* board4 = find_board(4);
  const MultiBoardConsistencyBoardDiagnostics* board5 = find_board(5);

  output << "\n[key_questions]\n";
  output << "board4_has_larger_consistency_error: "
         << ((board4 != nullptr && board5 != nullptr &&
              board4->mean_translation_error_mm > board5->mean_translation_error_mm)
                 ? 1
                 : 0)
         << "\n";
  output << "board5_has_larger_consistency_error: "
         << ((board4 != nullptr && board5 != nullptr &&
              board5->mean_translation_error_mm > board4->mean_translation_error_mm)
                 ? 1
                 : 0)
         << "\n";
  output << "high_consistency_error_vs_high_residual: inspect_top_ranked_lists\n";
  output << "high_consistency_error_vs_high_polar_angle: inspect_top_ranked_lists\n";
  output << "residual_not_extreme_but_structure_inconsistent_possible: 1\n";
  output << "phase4_board_level_consistency_weighting_recommendation: "
         << (result.successful_local_pose_refit_count > 0 ? "inspect_summary" : "not_ready")
         << "\n";

  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteBackendInputAblationSummary(
    const std::string& path,
    const BackendInputAblationResult& result) {
  std::ofstream output(path.c_str());
  output << "enabled: " << (result.enabled ? 1 : 0) << "\n";
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "point_budget_control_enabled: "
         << (result.point_budget_control_enabled ? 1 : 0) << "\n";
  output << "point_budget_total_points: "
         << result.point_budget_total_points << "\n";
  output << "point_budget_seed: " << result.point_budget_seed << "\n";
  output << "max_boards_per_frame_for_ablation: "
         << result.max_boards_per_frame_for_ablation << "\n";
  output << "input_frame_count: " << result.input_frame_count << "\n";
  output << "input_board_observation_count: "
         << result.input_board_observation_count << "\n";
  output << "input_outer_point_count: "
         << result.input_outer_point_count << "\n";
  output << "input_internal_point_count: "
         << result.input_internal_point_count << "\n";
  output << "input_total_point_count: "
         << result.input_total_point_count << "\n";
  output << "output_frame_count: " << result.output_frame_count << "\n";
  output << "output_board_observation_count: "
         << result.output_board_observation_count << "\n";
  output << "output_outer_point_count: "
         << result.output_outer_point_count << "\n";
  output << "output_internal_point_count: "
         << result.output_internal_point_count << "\n";
  output << "output_total_point_count: "
         << result.output_total_point_count << "\n";
  output << "removed_board_observation_count: "
         << result.removed_board_observation_count << "\n";
  output << "removed_internal_point_count: "
         << result.removed_internal_point_count << "\n";
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteBoardLayoutPoseDeltaCsv(
    const std::string& path,
    const AslamBackendCalibrationResult& backend_result) {
  const auto find_initial_board =
      [&backend_result](int board_id) -> const JointSceneBoardState* {
    return FindJointSceneBoardState(
        backend_result.initial_scene_state,
        board_id);
  };
  const auto find_board_rmse =
      [](const JointResidualEvaluationResult& residual,
         int board_id) -> double {
    for (const JointResidualBoardDiagnostics& board :
         residual.board_diagnostics) {
      if (board.board_id == board_id) {
        return board.rmse;
      }
    }
    return 0.0;
  };
  const auto find_board_observation_count =
      [](const JointResidualEvaluationResult& residual,
         int board_id) -> int {
    for (const JointResidualBoardDiagnostics& board :
         residual.board_diagnostics) {
      if (board.board_id == board_id) {
        return board.observation_count;
      }
    }
    return 0;
  };

  std::map<int, std::set<int> > frames_by_board;
  std::map<int, std::set<int> > boards_by_frame;
  for (const std::pair<int, int>& key :
       backend_result.effective_problem_input.measurement_dataset
           .accepted_board_observation_keys) {
    frames_by_board[key.second].insert(key.first);
    boards_by_frame[key.first].insert(key.second);
  }

  std::ofstream output(path.c_str());
  output << "board_id,is_reference,observation_count,coobserved_frame_count,"
         << "translation_delta_mm,rotation_delta_deg,initial_rmse,"
         << "optimized_rmse\n";
  for (const JointSceneBoardState& optimized_board :
       backend_result.optimized_scene_state.boards) {
    const int board_id = optimized_board.board_id;
    const JointSceneBoardState* initial_board = find_initial_board(board_id);
    if (initial_board == nullptr) {
      continue;
    }
    const bool is_reference =
        board_id == backend_result.effective_problem_input.reference_board_id;
    const Eigen::Isometry3d T_initial =
        is_reference
            ? Eigen::Isometry3d::Identity()
            : ToIsometry3d(initial_board->T_reference_board);
    const Eigen::Isometry3d T_optimized =
        is_reference
            ? Eigen::Isometry3d::Identity()
            : ToIsometry3d(optimized_board.T_reference_board);
    const Eigen::Isometry3d delta = T_initial.inverse() * T_optimized;
    int coobserved_frame_count = 0;
    const auto frame_it = frames_by_board.find(board_id);
    if (frame_it != frames_by_board.end()) {
      for (int frame_index : frame_it->second) {
        const auto boards_it = boards_by_frame.find(frame_index);
        if (boards_it != boards_by_frame.end() &&
            boards_it->second.size() > 1) {
          ++coobserved_frame_count;
        }
      }
    }

    output << board_id << ","
           << (is_reference ? 1 : 0) << ","
           << find_board_observation_count(
                  backend_result.optimized_residual,
                  board_id) << ","
           << coobserved_frame_count << ","
           << delta.translation().norm() * 1000.0 << ","
           << ComputeRotationAngleDeg(delta.rotation()) << ","
           << find_board_rmse(backend_result.initial_residual, board_id) << ","
           << find_board_rmse(backend_result.optimized_residual, board_id)
           << "\n";
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
