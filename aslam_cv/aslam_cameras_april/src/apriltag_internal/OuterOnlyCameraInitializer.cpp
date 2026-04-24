#include <aslam/cameras/apriltag_internal/OuterOnlyCameraInitializer.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

struct OuterObservationRecord {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  double quality = 0.0;
  std::vector<Eigen::Vector3d> object_points;
  std::vector<cv::Point2f> image_points;
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

bool ClampIntrinsicsInPlace(OuterBootstrapCameraIntrinsics* intrinsics) {
  if (intrinsics == nullptr) {
    throw std::runtime_error("ClampIntrinsicsInPlace requires a valid pointer.");
  }
  intrinsics->xi = std::max(-0.95, std::min(2.5, intrinsics->xi));
  intrinsics->alpha = std::max(0.05, std::min(0.95, intrinsics->alpha));
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
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.resolution = resolution;
  intrinsics.xi = config.sphere_lattice_init_xi;
  intrinsics.alpha = config.sphere_lattice_init_alpha;
  intrinsics.fu = config.sphere_lattice_init_fu_scale * static_cast<double>(resolution.width);
  intrinsics.fv = config.sphere_lattice_init_fv_scale * static_cast<double>(resolution.height);
  intrinsics.cu = 0.5 * static_cast<double>(resolution.width) +
                  config.sphere_lattice_init_cu_offset;
  intrinsics.cv = 0.5 * static_cast<double>(resolution.height) +
                  config.sphere_lattice_init_cv_offset;
  ClampIntrinsicsInPlace(&intrinsics);
  return intrinsics;
}

OuterBootstrapCameraIntrinsics MakeIntermediateIntrinsics(
    const IntermediateCameraConfig& camera_config) {
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.xi = camera_config.intrinsics[0];
  intrinsics.alpha = camera_config.intrinsics[1];
  intrinsics.fu = camera_config.intrinsics[2];
  intrinsics.fv = camera_config.intrinsics[3];
  intrinsics.cu = camera_config.intrinsics[4];
  intrinsics.cv = camera_config.intrinsics[5];
  intrinsics.resolution =
      cv::Size(camera_config.resolution[0], camera_config.resolution[1]);
  return intrinsics;
}

OuterBootstrapCameraIntrinsics BuildManualInitialCamera(
    const cv::Size& image_size,
    const ApriltagInternalConfig& config,
    bool* used_manual_intermediate_camera,
    bool* used_manual_generic_seed,
    std::string* source_label,
    std::vector<std::string>* warnings) {
  if (used_manual_intermediate_camera != nullptr) {
    *used_manual_intermediate_camera = false;
  }
  if (used_manual_generic_seed != nullptr) {
    *used_manual_generic_seed = false;
  }
  if (source_label != nullptr) {
    *source_label = "manual_generic_seed";
  }

  if (config.intermediate_camera.IsConfigured() &&
      config.intermediate_camera.camera_model == "ds" &&
      config.intermediate_camera.intrinsics.size() == 6 &&
      config.intermediate_camera.resolution.size() == 2) {
    const cv::Size configured_size(config.intermediate_camera.resolution[0],
                                   config.intermediate_camera.resolution[1]);
    if (configured_size == image_size) {
      OuterBootstrapCameraIntrinsics intrinsics =
          MakeIntermediateIntrinsics(config.intermediate_camera);
      if (ClampIntrinsicsInPlace(&intrinsics)) {
        if (used_manual_intermediate_camera != nullptr) {
          *used_manual_intermediate_camera = true;
        }
        if (source_label != nullptr) {
          *source_label = "manual_intermediate_camera";
        }
        return intrinsics;
      }
      AppendUniqueWarning(
          "Configured intermediate_camera is invalid after clamping; using generic "
          "sphere_lattice seed instead.",
          warnings);
    } else {
      std::ostringstream stream;
      stream << "Configured intermediate_camera resolution "
             << configured_size.width << "x" << configured_size.height
             << " does not match image resolution " << image_size.width << "x"
             << image_size.height
             << "; using generic sphere_lattice seed instead.";
      AppendUniqueWarning(stream.str(), warnings);
    }
  }

  if (used_manual_generic_seed != nullptr) {
    *used_manual_generic_seed = true;
  }
  if (source_label != nullptr) {
    *source_label = "manual_generic_seed";
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
  return candidate.mean_observation_rmse * candidate.mean_observation_rmse +
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

double ParameterStep(double value, double fallback_step) {
  return std::max(std::abs(value) * 0.05, fallback_step);
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

  const double base_steps[6] = {
      0.15,
      0.10,
      ParameterStep(best.fu, 40.0),
      ParameterStep(best.fv, 40.0),
      ParameterStep(best.cu, 20.0),
      ParameterStep(best.cv, 20.0),
  };

  for (int round = 0; round < 4; ++round) {
    const double round_scale = std::pow(0.5, round);
    for (int parameter_index = 0; parameter_index < 6; ++parameter_index) {
      for (int direction = -1; direction <= 1; direction += 2) {
        OuterBootstrapCameraIntrinsics candidate = best;
        const double delta = static_cast<double>(direction) *
                             base_steps[parameter_index] * round_scale;
        switch (parameter_index) {
          case 0:
            candidate.xi += delta;
            break;
          case 1:
            candidate.alpha += delta;
            break;
          case 2:
            candidate.fu += delta;
            break;
          case 3:
            candidate.fv += delta;
            break;
          case 4:
            candidate.cu += delta;
            break;
          case 5:
            candidate.cv += delta;
            break;
        }
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
  for (const AutoCameraInitializationResidual& residual : residuals) {
    if (!residual.pose_success) {
      ++result->failed_pose_fit_observation_count;
      continue;
    }
    ++result->accepted_pose_fit_observation_count;
    accepted_frames.insert(residual.frame_index);
    total_squared_rmse += residual.pose_fit_outer_rmse * residual.pose_fit_outer_rmse;
  }

  result->accepted_frame_count = static_cast<int>(accepted_frames.size());
  result->accepted_board_observation_count = result->accepted_pose_fit_observation_count;
  if (result->accepted_pose_fit_observation_count > 0) {
    result->initialization_rmse =
        std::sqrt(total_squared_rmse /
                  static_cast<double>(result->accepted_pose_fit_observation_count));
  }
}

std::vector<AutoCameraInitializationCandidate> GenerateCandidateGrid(
    const cv::Size& image_size,
    const AutoCameraInitializationOptions& options) {
  std::vector<AutoCameraInitializationCandidate> candidates;
  const double center_u = 0.5 * static_cast<double>(image_size.width);
  const double center_v = 0.5 * static_cast<double>(image_size.height);
  for (double focal_scale : options.focal_scale_candidates) {
    for (double xi : options.xi_candidates) {
      for (double alpha : options.alpha_candidates) {
        AutoCameraInitializationCandidate candidate;
        candidate.source_label = "auto_grid";
        candidate.evaluation_scope = "sampled";
        candidate.camera.resolution = image_size;
        candidate.camera.xi = xi;
        candidate.camera.alpha = alpha;
        candidate.camera.fu = focal_scale * static_cast<double>(image_size.width);
        candidate.camera.fv = focal_scale * static_cast<double>(image_size.height);
        candidate.camera.cu = center_u;
        candidate.camera.cv = center_v;
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
  AutoCameraInitializationResult result;
  result.requested_mode = options_.mode;
  result.selected_mode = CameraInitializationMode::Manual;
  result.image_size = InferImageSize(frames);

  if (result.image_size.width <= 0 || result.image_size.height <= 0) {
    result.failure_reason = "Could not infer image size from outer observations.";
    return result;
  }

  bool used_manual_intermediate_camera = false;
  bool used_manual_generic_seed = false;
  std::string manual_source_label;
  const OuterBootstrapCameraIntrinsics manual_camera =
      BuildManualInitialCamera(result.image_size,
                               config_,
                               &used_manual_intermediate_camera,
                               &used_manual_generic_seed,
                               &manual_source_label,
                               &result.warnings);

  if (options_.mode == CameraInitializationMode::Manual) {
    result.success = manual_camera.IsValid();
    result.selected_mode = CameraInitializationMode::Manual;
    result.selected_source_label = manual_source_label;
    result.selected_camera = manual_camera;
    result.used_manual_intermediate_camera = used_manual_intermediate_camera;
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

    std::vector<AutoCameraInitializationCandidate> candidates =
        GenerateCandidateGrid(result.image_size, options_);
    for (AutoCameraInitializationCandidate& candidate : candidates) {
      candidate = EvaluateCandidateOnObservations(
          candidate.camera, candidate.source_label, "sampled", sampled_observations);
    }

    std::sort(candidates.begin(), candidates.end(), CandidateIsBetter);
    const int top_candidate_count =
        std::max(1, std::min(options_.top_candidate_count,
                              static_cast<int>(candidates.size())));
    for (int index = 0; index < top_candidate_count; ++index) {
      AutoCameraInitializationCandidate reevaluated =
          EvaluateCandidateOnObservations(candidates[static_cast<std::size_t>(index)].camera,
                                          candidates[static_cast<std::size_t>(index)].source_label,
                                          "full",
                                          all_observations);
      candidates[static_cast<std::size_t>(index)] = reevaluated;
    }

    std::sort(candidates.begin(), candidates.end(), CandidateIsBetter);
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      candidates[index].rank = static_cast<int>(index + 1);
    }
    result.candidate_count = static_cast<int>(candidates.size());
    result.candidates = candidates;

    if (!candidates.empty() &&
        IsAcceptableAutoCandidate(candidates.front(),
                                  result.total_valid_outer_observation_count)) {
      AutoCameraInitializationCandidate best_candidate = candidates.front();
      OuterBootstrapCameraIntrinsics selected_camera = best_candidate.camera;
      std::string selected_source_label = "auto_grid";
      if (options_.refine_best_candidate) {
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
          "stable DS candidate.";
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
    result.used_manual_generic_seed =
        (result.selected_mode == CameraInitializationMode::Manual) &&
        used_manual_generic_seed;
  }

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
  output << "used_manual_generic_seed: "
         << (result.used_manual_generic_seed ? 1 : 0) << "\n";
  output << "selected_candidate_refined: "
         << (result.selected_candidate_refined ? 1 : 0) << "\n";
  output << "selected_source_label: " << result.selected_source_label << "\n";
  output << "image_width: " << result.image_size.width << "\n";
  output << "image_height: " << result.image_size.height << "\n";
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
  output << "rank,source_label,evaluation_scope,xi,alpha,fu,fv,cu,cv,"
         << "observation_count,pose_success_count,pose_failure_count,"
         << "successful_frame_count,successful_board_count,success_rate,"
         << "mean_observation_rmse,valid,failure_reason\n";
  for (const AutoCameraInitializationCandidate& candidate : result.candidates) {
    output << candidate.rank << ","
           << candidate.source_label << ","
           << candidate.evaluation_scope << ","
           << candidate.camera.xi << ","
           << candidate.camera.alpha << ","
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

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
