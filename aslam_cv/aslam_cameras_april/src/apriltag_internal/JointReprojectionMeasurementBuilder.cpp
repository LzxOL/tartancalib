#include <aslam/cameras/apriltag_internal/JointReprojectionMeasurementBuilder.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

#include <Eigen/LU>
#include <Eigen/QR>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

void AppendUniqueBoardId(int board_id, std::vector<int>* board_ids) {
  if (board_ids == nullptr || board_id < 0) {
    return;
  }
  if (std::find(board_ids->begin(), board_ids->end(), board_id) == board_ids->end()) {
    board_ids->push_back(board_id);
  }
}

void AppendUniqueWarning(const std::string& warning, std::vector<std::string>* warnings) {
  if (warnings == nullptr || warning.empty()) {
    return;
  }
  if (std::find(warnings->begin(), warnings->end(), warning) == warnings->end()) {
    warnings->push_back(warning);
  }
}

std::string JoinBoardIds(const std::vector<int>& board_ids) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < board_ids.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << board_ids[index];
  }
  return stream.str();
}

int CountUsedPoints(const JointBoardObservation& board_observation,
                    JointPointType point_type) {
  int count = 0;
  for (const JointPointObservation& point : board_observation.points) {
    if (point.used_in_solver && point.point_type == point_type) {
      ++count;
    }
  }
  return count;
}

int RejectDuplicateInternalImageLocations(
    const JointMeasurementBuildOptions& options,
    JointBoardObservation* board_observation) {
  if (board_observation == nullptr ||
      !(options.internal_duplicate_image_distance_px > 0.0)) {
    return 0;
  }

  std::vector<std::size_t> internal_indices;
  for (std::size_t index = 0; index < board_observation->points.size(); ++index) {
    const JointPointObservation& point = board_observation->points[index];
    if (point.used_in_solver && point.point_type == JointPointType::Internal &&
        point.image_xy.allFinite()) {
      internal_indices.push_back(index);
    }
  }

  const double threshold_squared =
      options.internal_duplicate_image_distance_px *
      options.internal_duplicate_image_distance_px;
  std::vector<int> component(internal_indices.size(), -1);
  int component_count = 0;
  for (std::size_t seed = 0; seed < internal_indices.size(); ++seed) {
    if (component[seed] >= 0) {
      continue;
    }
    component[seed] = component_count;
    std::vector<std::size_t> frontier{seed};
    for (std::size_t cursor = 0; cursor < frontier.size(); ++cursor) {
      const std::size_t current = frontier[cursor];
      const JointPointObservation& current_point =
          board_observation->points[internal_indices[current]];
      for (std::size_t candidate = 0; candidate < internal_indices.size();
           ++candidate) {
        if (component[candidate] >= 0) {
          continue;
        }
        const JointPointObservation& candidate_point =
            board_observation->points[internal_indices[candidate]];
        if (candidate_point.point_id == current_point.point_id ||
            (candidate_point.image_xy - current_point.image_xy).squaredNorm() >
                threshold_squared) {
          continue;
        }
        component[candidate] = component_count;
        frontier.push_back(candidate);
      }
    }
    ++component_count;
  }

  int rejected_count = 0;
  for (int component_id = 0; component_id < component_count; ++component_id) {
    std::vector<std::size_t> members;
    for (std::size_t index = 0; index < component.size(); ++index) {
      if (component[index] == component_id) {
        members.push_back(internal_indices[index]);
      }
    }
    if (members.size() <= 1) {
      continue;
    }

    double best_quality = -std::numeric_limits<double>::infinity();
    for (std::size_t index : members) {
      const double quality = board_observation->points[index].quality;
      best_quality = std::max(
          best_quality, std::isfinite(quality) ? quality : 0.0);
    }
    int best_count = 0;
    std::size_t best_index = members.front();
    for (std::size_t index : members) {
      const double quality = board_observation->points[index].quality;
      const double finite_quality = std::isfinite(quality) ? quality : 0.0;
      if (std::abs(finite_quality - best_quality) <= 1e-9) {
        ++best_count;
        best_index = index;
      }
    }

    std::ostringstream competing_ids;
    for (std::size_t member_index = 0; member_index < members.size();
         ++member_index) {
      if (member_index > 0) {
        competing_ids << ";";
      }
      competing_ids << board_observation->points[members[member_index]].point_id;
    }
    for (std::size_t index : members) {
      if (best_count == 1 && index == best_index) {
        continue;
      }
      JointPointObservation& point = board_observation->points[index];
      point.used_in_solver = false;
      point.rejection_reason_code =
          JointRejectionReasonCode::DuplicateInternalImageLocation;
      std::ostringstream detail;
      detail << "different internal point IDs share one refined image location"
             << " competing_point_ids=" << competing_ids.str()
             << " threshold_px="
             << options.internal_duplicate_image_distance_px;
      if (best_count != 1) {
        detail << " ambiguous_equal_quality=1";
      }
      point.rejection_detail = detail.str();
      ++rejected_count;
    }
  }
  return rejected_count;
}

double ComputeInternalObservationQualityWeight(
    double quality,
    double quality_threshold,
    const JointMeasurementBuildOptions& options) {
  if (!options.enable_internal_observation_quality_weighting) {
    return 1.0;
  }
  const double bounded_quality =
      std::max(0.0, std::min(1.0, std::isfinite(quality) ? quality : 0.0));
  if (bounded_quality > quality_threshold) {
    return 1.0;
  }
  const double min_weight =
      std::max(0.0, std::min(1.0, options.internal_observation_min_weight));
  const double exponent =
      options.internal_observation_quality_exponent > 0.0
          ? options.internal_observation_quality_exponent
          : 1.0;
  return min_weight + (1.0 - min_weight) *
                          std::pow(bounded_quality, exponent);
}

double Quantile(std::vector<double> values, double q) {
  if (values.empty()) {
    return 0.0;
  }
  if (q <= 0.0) {
    return *std::min_element(values.begin(), values.end());
  }
  if (q >= 1.0) {
    return *std::max_element(values.begin(), values.end());
  }
  std::sort(values.begin(), values.end());
  const double scaled = q * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(scaled));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(scaled));
  if (lower == upper) {
    return values[lower];
  }
  const double t = scaled - static_cast<double>(lower);
  return (1.0 - t) * values[lower] + t * values[upper];
}

double ClampUnitLocal(double value) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  return std::max(0.0, std::min(1.0, value));
}

void ApplyInternalObservationQualityWeightsToResult(
    JointMeasurementBuildResult* result,
    const JointMeasurementBuildOptions& options) {
  if (result == nullptr) {
    return;
  }

  if (options.robust_missing_board_recovery) {
    // Robust recovery has already passed image refinement and topology gates.
    // Keep its accepted internal points at unit weight; otherwise this later
    // quality pass would silently reintroduce the low-quality down-weighting
    // that the recovery mode is designed to remove.
    result->solver_observations.clear();
    result->used_outer_point_count = 0;
    result->used_internal_point_count = 0;
    result->used_total_point_count = 0;
    std::set<std::pair<int, int> > accepted_outer_keys;
    std::set<std::pair<int, int> > accepted_internal_keys;
    std::set<std::pair<int, int> > used_board_keys;
    std::set<int> used_frame_indices;
    for (JointMeasurementFrameResult& frame : result->frames) {
      for (JointBoardObservation& board : frame.board_observations) {
        board.outer_point_count = 0;
        board.internal_point_count = 0;
        board.used_in_solver = false;
        for (JointPointObservation& point : board.points) {
          if (point.point_type == JointPointType::Internal &&
              point.used_in_solver) {
            point.observation_weight = 1.0;
            point.consistency_weight = 1.0;
            point.final_observation_weight = 1.0;
          }
          if (point.used_in_solver) {
            result->solver_observations.push_back(point);
            ++result->used_total_point_count;
            board.used_in_solver = true;
            used_frame_indices.insert(frame.frame_index);
            used_board_keys.insert(
                std::make_pair(frame.frame_index, board.board_id));
            if (point.point_type == JointPointType::Internal) {
              ++result->used_internal_point_count;
              ++board.internal_point_count;
              accepted_internal_keys.insert(
                  std::make_pair(frame.frame_index, board.board_id));
            } else {
              ++result->used_outer_point_count;
              ++board.outer_point_count;
              accepted_outer_keys.insert(
                  std::make_pair(frame.frame_index, board.board_id));
            }
          }
        }
      }
    }
    result->used_frame_count = static_cast<int>(used_frame_indices.size());
    result->used_board_observation_count =
        static_cast<int>(used_board_keys.size());
    result->accepted_outer_board_observation_count =
        static_cast<int>(accepted_outer_keys.size());
    result->accepted_internal_board_observation_count =
        static_cast<int>(accepted_internal_keys.size());
    return;
  }

  if (!options.enable_internal_observation_quality_weighting) {
    return;
  }

  std::vector<double> qualities;
  for (const JointMeasurementFrameResult& frame : result->frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      for (const JointPointObservation& point : board.points) {
        if (point.used_in_solver && point.point_type == JointPointType::Internal) {
          qualities.push_back(
              std::max(0.0, std::min(1.0, std::isfinite(point.quality) ? point.quality : 0.0)));
        }
      }
    }
  }
  if (qualities.empty()) {
    return;
  }

  const double threshold = Quantile(
      qualities,
      std::max(0.0, std::min(1.0, options.internal_observation_low_quality_quantile)));

  result->solver_observations.clear();
  result->used_internal_point_count = 0;
  result->used_total_point_count = 0;
  for (JointMeasurementFrameResult& frame : result->frames) {
    for (JointBoardObservation& board : frame.board_observations) {
      for (JointPointObservation& point : board.points) {
        if (point.point_type == JointPointType::Internal) {
          point.observation_weight = point.used_in_solver
                                         ? ComputeInternalObservationQualityWeight(
                                               point.quality, threshold, options)
                                         : 1.0;
        }
        if (point.used_in_solver) {
          result->solver_observations.push_back(point);
          ++result->used_total_point_count;
          if (point.point_type == JointPointType::Internal) {
            ++result->used_internal_point_count;
          }
        }
      }
    }
  }
}

std::vector<int> CollectBoardIds(const JointMeasurementFrameInput& frame_input) {
  std::vector<int> board_ids;
  for (int board_id : frame_input.outer_detections.requested_board_ids) {
    AppendUniqueBoardId(board_id, &board_ids);
  }
  for (const OuterBoardMeasurement& measurement :
       frame_input.outer_detections.frame_measurements.board_measurements) {
    AppendUniqueBoardId(measurement.board_id, &board_ids);
  }
  for (const OuterTagDetectionResult& detection : frame_input.outer_detections.detections) {
    AppendUniqueBoardId(detection.board_id, &board_ids);
  }
  for (const RegeneratedBoardMeasurement& measurement :
       frame_input.regenerated_internal.board_measurements) {
    AppendUniqueBoardId(measurement.board_id, &board_ids);
  }
  return board_ids;
}

const OuterBoardMeasurement* FindOuterBoardMeasurement(
    const JointMeasurementFrameInput& frame_input,
    int board_id,
    int* measurement_index) {
  const std::vector<OuterBoardMeasurement>& measurements =
      frame_input.outer_detections.frame_measurements.board_measurements;
  for (std::size_t index = 0; index < measurements.size(); ++index) {
    if (measurements[index].board_id == board_id) {
      if (measurement_index != nullptr) {
        *measurement_index = static_cast<int>(index);
      }
      return &measurements[index];
    }
  }
  if (measurement_index != nullptr) {
    *measurement_index = -1;
  }
  return nullptr;
}

OuterBoardMeasurement BuildOuterBoardMeasurementFromDetection(
    const OuterTagDetectionResult& detection) {
  OuterBoardMeasurement measurement;
  measurement.board_id = detection.board_id;
  measurement.detected_tag_id = detection.detected_tag_id;
  measurement.success = detection.success;
  measurement.attempted_local_patch_rescue = detection.attempted_local_patch_rescue;
  measurement.used_local_patch_rescue = detection.used_local_patch_rescue;
  measurement.local_patch_rescue_summary = detection.local_patch_rescue_summary;
  measurement.detection_quality = detection.quality;
  measurement.valid_refined_corner_count = 0;
  measurement.refined_outer_corners_original_image =
      detection.refined_corners_original_image;
  measurement.refined_corner_valid = detection.refined_valid;
  measurement.corner_verification_debug = detection.corner_verification_debug;
  for (bool valid : detection.refined_valid) {
    measurement.valid_refined_corner_count += valid ? 1 : 0;
  }
  measurement.failure_reason = detection.failure_reason;
  measurement.failure_reason_text = detection.failure_reason_text;
  return measurement;
}

const RegeneratedBoardMeasurement* FindRegeneratedBoardMeasurement(
    const JointMeasurementFrameInput& frame_input,
    int board_id,
    int* measurement_index) {
  const std::vector<RegeneratedBoardMeasurement>& measurements =
      frame_input.regenerated_internal.board_measurements;
  for (std::size_t index = 0; index < measurements.size(); ++index) {
    if (measurements[index].board_id == board_id) {
      if (measurement_index != nullptr) {
        *measurement_index = static_cast<int>(index);
      }
      return &measurements[index];
    }
  }
  if (measurement_index != nullptr) {
    *measurement_index = -1;
  }
  return nullptr;
}

JointRejectionReasonCode ComputeBoardLevelReason(
    const OuterBootstrapFrameState* frame_state,
    const OuterBootstrapBoardState* board_state,
    const OuterBootstrapObservationDiagnostics* observation_diagnostics,
    bool require_initialized_frame_and_board) {
  if (frame_state == nullptr) {
    return JointRejectionReasonCode::FrameNotFoundInBootstrap;
  }
  if (require_initialized_frame_and_board && !frame_state->initialized) {
    return JointRejectionReasonCode::FrameNotInitialized;
  }
  if (board_state == nullptr || (require_initialized_frame_and_board && !board_state->initialized)) {
    return JointRejectionReasonCode::BoardNotInitialized;
  }
  if (observation_diagnostics != nullptr && !observation_diagnostics->reference_connected) {
    return JointRejectionReasonCode::NotReferenceConnected;
  }
  return JointRejectionReasonCode::None;
}

std::string BuildBoardLevelReasonDetail(
    const JointMeasurementFrameInput& frame_input,
    const OuterBootstrapFrameState* frame_state,
    const OuterBootstrapBoardState* board_state,
    const OuterBootstrapObservationDiagnostics* observation_diagnostics) {
  std::ostringstream stream;
  if (frame_state == nullptr) {
    stream << "frame_index=" << frame_input.frame_index << " missing from bootstrap_result.frames";
    return stream.str();
  }
  if (!frame_input.frame_label.empty() && !frame_state->frame_label.empty() &&
      frame_input.frame_label != frame_state->frame_label) {
    stream << "frame_index=" << frame_input.frame_index
           << " label mismatch input=" << frame_input.frame_label
           << " bootstrap=" << frame_state->frame_label;
    return stream.str();
  }
  if (!frame_state->initialized) {
    stream << "bootstrap frame was not initialized";
    return stream.str();
  }
  if (board_state == nullptr) {
    stream << "board_id missing from bootstrap_result.boards";
    return stream.str();
  }
  if (!board_state->initialized) {
    stream << "bootstrap board was not initialized";
    return stream.str();
  }
  if (observation_diagnostics != nullptr && !observation_diagnostics->reference_connected) {
    stream << "bootstrap observation is not reference-connected";
    return stream.str();
  }
  return std::string();
}

std::array<CanonicalCorner, 4> OuterCanonicalCorners(const ApriltagCanonicalModel& model) {
  return {model.corner(model.PointId(0, 0)),
          model.corner(model.PointId(model.ModuleDimension(), 0)),
          model.corner(model.PointId(model.ModuleDimension(), model.ModuleDimension())),
          model.corner(model.PointId(0, model.ModuleDimension()))};
}

struct BoardTopologyConsistencyOutcome {
  bool evaluated = false;
  int rejected_internal_point_count = 0;
  std::string diagnostic;
};

struct InternalTopologySurface {
  Eigen::Vector2d target_center = Eigen::Vector2d::Zero();
  Eigen::Vector2d target_scale = Eigen::Vector2d::Ones();
  Eigen::Matrix<double, 10, 1> coeff_u =
      Eigen::Matrix<double, 10, 1>::Zero();
  Eigen::Matrix<double, 10, 1> coeff_v =
      Eigen::Matrix<double, 10, 1>::Zero();
  double rmse = std::numeric_limits<double>::infinity();
  double residual_median = 0.0;
  double residual_sigma = 0.0;

  Eigen::Matrix<double, 10, 1> Terms(
      const Eigen::Vector3d& target) const {
    const double x = (target.x() - target_center.x()) / target_scale.x();
    const double y = (target.y() - target_center.y()) / target_scale.y();
    Eigen::Matrix<double, 10, 1> terms;
    terms << 1.0, x, y, x * x, x * y, y * y,
        x * x * x, x * x * y, x * y * y, y * y * y;
    return terms;
  }

  Eigen::Vector2d Evaluate(const Eigen::Vector3d& target) const {
    const Eigen::Matrix<double, 10, 1> terms = Terms(target);
    return Eigen::Vector2d(coeff_u.dot(terms), coeff_v.dot(terms));
  }

  Eigen::Matrix2d Jacobian(const Eigen::Vector3d& target) const {
    const double x = (target.x() - target_center.x()) / target_scale.x();
    const double y = (target.y() - target_center.y()) / target_scale.y();
    Eigen::Matrix<double, 10, 1> dx;
    Eigen::Matrix<double, 10, 1> dy;
    dx << 0.0, 1.0, 0.0, 2.0 * x, y, 0.0,
        3.0 * x * x, 2.0 * x * y, y * y, 0.0;
    dy << 0.0, 0.0, 1.0, 0.0, x, 2.0 * y,
        0.0, x * x, 2.0 * x * y, 3.0 * y * y;
    Eigen::Matrix2d jacobian;
    jacobian(0, 0) = coeff_u.dot(dx) / target_scale.x();
    jacobian(0, 1) = coeff_u.dot(dy) / target_scale.y();
    jacobian(1, 0) = coeff_v.dot(dx) / target_scale.x();
    jacobian(1, 1) = coeff_v.dot(dy) / target_scale.y();
    return jacobian;
  }
};

bool FitInternalTopologySurfaceOnce(
    const std::vector<JointPointObservation*>& points,
    const Eigen::Vector2d& target_center,
    const Eigen::Vector2d& target_scale,
    InternalTopologySurface* surface,
    std::vector<double>* residuals) {
  if (surface == nullptr || residuals == nullptr || points.size() < 10) {
    return false;
  }
  InternalTopologySurface fitted;
  fitted.target_center = target_center;
  fitted.target_scale = target_scale;
  Eigen::MatrixXd design(points.size(), 10);
  Eigen::VectorXd observed_u(points.size());
  Eigen::VectorXd observed_v(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    if (points[index] == nullptr || !points[index]->image_xy.allFinite() ||
        !points[index]->target_xyz_board.allFinite()) {
      return false;
    }
    design.row(static_cast<Eigen::Index>(index)) =
        fitted.Terms(points[index]->target_xyz_board).transpose();
    observed_u[static_cast<Eigen::Index>(index)] = points[index]->image_xy.x();
    observed_v[static_cast<Eigen::Index>(index)] = points[index]->image_xy.y();
  }
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> decomposition(design);
  if (decomposition.rank() < 10) {
    return false;
  }
  fitted.coeff_u = decomposition.solve(observed_u);
  fitted.coeff_v = decomposition.solve(observed_v);
  if (!fitted.coeff_u.allFinite() || !fitted.coeff_v.allFinite()) {
    return false;
  }
  residuals->clear();
  residuals->reserve(points.size());
  double squared_error_sum = 0.0;
  for (const JointPointObservation* point : points) {
    const double residual =
        (fitted.Evaluate(point->target_xyz_board) - point->image_xy).norm();
    residuals->push_back(residual);
    squared_error_sum += residual * residual;
  }
  fitted.rmse =
      std::sqrt(squared_error_sum / static_cast<double>(points.size()));
  fitted.residual_median = Quantile(*residuals, 0.5);
  std::vector<double> deviations;
  deviations.reserve(residuals->size());
  for (double residual : *residuals) {
    deviations.push_back(std::abs(residual - fitted.residual_median));
  }
  fitted.residual_sigma = 1.4826 * Quantile(deviations, 0.5);
  *surface = fitted;
  return true;
}

bool FitReliableInternalTopologySurface(
    const JointMeasurementBuildOptions& options,
    const ApriltagCanonicalModel& model,
    const std::vector<JointPointObservation*>& internal_points,
    InternalTopologySurface* surface,
    double* module_scale_px) {
  if (surface == nullptr || module_scale_px == nullptr ||
      static_cast<int>(internal_points.size()) <
          options.board_topology_min_internal_point_count) {
    return false;
  }
  Eigen::Vector2d min_target = internal_points.front()->target_xyz_board.head<2>();
  Eigen::Vector2d max_target = min_target;
  for (const JointPointObservation* point : internal_points) {
    min_target = min_target.cwiseMin(point->target_xyz_board.head<2>());
    max_target = max_target.cwiseMax(point->target_xyz_board.head<2>());
  }
  const Eigen::Vector2d target_scale = 0.5 * (max_target - min_target);
  if (target_scale.x() <= 1e-9 || target_scale.y() <= 1e-9) {
    return false;
  }
  const Eigen::Vector2d target_center = 0.5 * (min_target + max_target);
  std::vector<double> residuals;
  InternalTopologySurface fitted;
  if (!FitInternalTopologySurfaceOnce(internal_points, target_center,
                                      target_scale, &fitted, &residuals)) {
    return false;
  }

  const double trim_threshold =
      std::max(2.0, fitted.residual_median +
                        4.0 * std::max(0.25, fitted.residual_sigma));
  std::vector<JointPointObservation*> trimmed_points;
  for (std::size_t index = 0; index < residuals.size(); ++index) {
    if (residuals[index] <= trim_threshold) {
      trimmed_points.push_back(internal_points[index]);
    }
  }
  if (trimmed_points.size() < internal_points.size() &&
      static_cast<int>(trimmed_points.size()) >=
          options.board_topology_min_internal_point_count) {
    InternalTopologySurface trimmed;
    if (FitInternalTopologySurfaceOnce(trimmed_points, target_center,
                                       target_scale, &trimmed, &residuals)) {
      fitted = trimmed;
    }
  }

  const std::array<CanonicalCorner, 4> canonical_outer =
      OuterCanonicalCorners(model);
  std::array<Eigen::Vector2d, 4> predicted_outer{};
  for (std::size_t index = 0; index < canonical_outer.size(); ++index) {
    predicted_outer[index] = fitted.Evaluate(canonical_outer[index].target_xyz);
  }
  std::vector<double> module_scales;
  for (std::size_t index = 0; index < predicted_outer.size(); ++index) {
    double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t other = 0; other < predicted_outer.size(); ++other) {
      if (other != index) {
        nearest = std::min(
            nearest, (predicted_outer[index] - predicted_outer[other]).norm());
      }
    }
    module_scales.push_back(
        nearest / std::max(1.0, static_cast<double>(model.ModuleDimension())));
  }
  const double scale = Quantile(module_scales, 0.5);
  if (!std::isfinite(scale) || scale <= 1.0 || !std::isfinite(fitted.rmse) ||
      fitted.rmse >
          options.board_topology_max_internal_surface_rmse_module_ratio *
              scale) {
    return false;
  }
  *surface = fitted;
  *module_scale_px = scale;
  return true;
}

BoardTopologyConsistencyOutcome EnforceInternalBoardTopologyConsistency(
    const ApriltagCanonicalModel& model,
    const JointMeasurementBuildOptions& options,
    JointBoardObservation* board_observation) {
  BoardTopologyConsistencyOutcome outcome;
  if (board_observation == nullptr ||
      !options.enable_bidirectional_board_topology_consistency) {
    return outcome;
  }

  std::vector<JointPointObservation*> outer_points;
  std::vector<JointPointObservation*> internal_points;
  for (JointPointObservation& point : board_observation->points) {
    if (!point.used_in_solver) {
      continue;
    }
    if (point.point_type == JointPointType::Outer) {
      outer_points.push_back(&point);
    } else {
      internal_points.push_back(&point);
    }
  }
  if (outer_points.size() != 4 ||
      static_cast<int>(internal_points.size()) <
          options.board_topology_min_internal_point_count) {
    return outcome;
  }

  InternalTopologySurface internal_surface;
  double module_scale_px = 0.0;
  if (!FitReliableInternalTopologySurface(
          options, model, internal_points, &internal_surface,
          &module_scale_px)) {
    return outcome;
  }
  outcome.evaluated = true;

  // The surface is fitted after a MAD trim.  A small number of points that
  // still disagree with that robust surface are semantically unsafe even if
  // cornerSubPix reported success.  Only remove an isolated minority; a
  // broad disagreement means the surface itself is not trustworthy and the
  // existing measurement path must remain unchanged.
  const double internal_residual_threshold = std::max(
      {options.board_topology_min_outer_residual_px,
       options.board_topology_max_outer_residual_module_ratio * module_scale_px,
       internal_surface.residual_median +
           6.0 * std::max(0.25, internal_surface.residual_sigma)});
  std::vector<std::size_t> internal_outlier_indices;
  for (std::size_t index = 0; index < internal_points.size(); ++index) {
    const double residual =
        (internal_surface.Evaluate(internal_points[index]->target_xyz_board) -
         internal_points[index]->image_xy)
            .norm();
    if (std::isfinite(residual) && residual > internal_residual_threshold) {
      internal_outlier_indices.push_back(index);
    }
  }
  if (!internal_outlier_indices.empty() &&
      internal_outlier_indices.size() <=
          std::max<std::size_t>(2, internal_points.size() / 10)) {
    for (std::size_t index : internal_outlier_indices) {
      JointPointObservation& rejected = *internal_points[index];
      rejected.used_in_solver = false;
      rejected.rejection_reason_code =
          JointRejectionReasonCode::InternalPointReprojectionOutlier;
      rejected.rejection_detail =
          "bidirectional_board_topology surface_internal_residual=" +
          std::to_string(internal_residual_threshold) +
          " action=reject_isolated_internal_topology_mismatch";
      ++outcome.rejected_internal_point_count;
    }
  }

  return outcome;
}

void FilterInternalPointsByReprojectionError(
    const OuterBootstrapCameraIntrinsics& camera,
    const JointMeasurementBuildOptions& options,
    JointBoardObservation* board_observation,
    bool gross_topology_only) {
  if (board_observation == nullptr ||
      (!gross_topology_only && !options.filter_internal_corner_outliers) ||
      (gross_topology_only &&
       !options.filter_gross_internal_topology_outliers)) {
    return;
  }

  std::vector<Eigen::Vector3d> outer_targets;
  std::vector<cv::Point2f> outer_pixels;
  std::vector<JointPointObservation*> internal_points;
  outer_targets.reserve(4);
  outer_pixels.reserve(4);
  for (JointPointObservation& point : board_observation->points) {
    if (!point.used_in_solver) {
      continue;
    }
    if (point.point_type == JointPointType::Outer) {
      outer_targets.push_back(point.target_xyz_board);
      outer_pixels.push_back(
          cv::Point2f(static_cast<float>(point.image_xy.x()),
                      static_cast<float>(point.image_xy.y())));
    } else {
      internal_points.push_back(&point);
    }
  }

  if (outer_targets.size() < 4 || internal_points.empty() || !camera.IsValid()) {
    return;
  }

  Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
  double pose_fit_rmse = 0.0;
  if (!EstimatePoseFromObjectPoints(camera, outer_targets, outer_pixels,
                                    &T_camera_board, &pose_fit_rmse)) {
    return;
  }
  if (gross_topology_only &&
      options.gross_internal_topology_max_outer_pose_rmse_px > 0.0 &&
      pose_fit_rmse >
          options.gross_internal_topology_max_outer_pose_rmse_px) {
    return;
  }

  const DoubleSphereCameraModel camera_model =
      DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(camera));
  std::vector<double> residual_norms;
  residual_norms.reserve(internal_points.size());
  std::vector<double> per_point_residuals(
      internal_points.size(), std::numeric_limits<double>::infinity());
  for (std::size_t index = 0; index < internal_points.size(); ++index) {
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    if (!camera_model.vsEuclideanToKeypoint(
            T_camera_board * internal_points[index]->target_xyz_board, &predicted)) {
      continue;
    }
    per_point_residuals[index] = (predicted - internal_points[index]->image_xy).norm();
    residual_norms.push_back(per_point_residuals[index]);
  }

  if (residual_norms.empty()) {
    for (JointPointObservation* point : internal_points) {
      point->used_in_solver = false;
      point->rejection_reason_code = JointRejectionReasonCode::InternalPointReprojectionOutlier;
      point->rejection_detail = "projection invalid under outer-only pose refit";
    }
    return;
  }

  const double mean_residual = std::accumulate(
      residual_norms.begin(), residual_norms.end(), 0.0) /
      static_cast<double>(residual_norms.size());
  double variance = 0.0;
  for (double residual : residual_norms) {
    const double delta = residual - mean_residual;
    variance += delta * delta;
  }
  variance /= static_cast<double>(residual_norms.size());
  const double std_residual = std::sqrt(std::max(0.0, variance));
  const double threshold =
      mean_residual + options.filter_internal_corner_sigma_threshold * std_residual;
  double effective_threshold = threshold;
  const bool has_absolute_cap =
      options.filter_internal_corner_max_reproj_error > 0.0;
  const bool use_quality_residual_adaptive =
      options.filter_internal_corner_mode == "quality_residual_adaptive";
  if (gross_topology_only) {
    const double median_residual = Quantile(residual_norms, 0.5);
    std::vector<double> absolute_deviations;
    absolute_deviations.reserve(residual_norms.size());
    for (double residual : residual_norms) {
      absolute_deviations.push_back(std::abs(residual - median_residual));
    }
    const double robust_sigma =
        1.4826 * Quantile(absolute_deviations, 0.5);
    const double robust_scale = std::max(0.25, robust_sigma);
    effective_threshold = std::max(
        options.gross_internal_topology_min_reproj_error_px,
        median_residual +
            options.gross_internal_topology_sigma_threshold * robust_scale);
  } else if (options.filter_internal_corner_mode == "local_residual_cap" &&
      has_absolute_cap) {
    effective_threshold = options.filter_internal_corner_max_reproj_error;
  } else if (options.filter_internal_corner_mode == "sigma_with_cap" &&
             has_absolute_cap) {
    effective_threshold =
        std::min(threshold, options.filter_internal_corner_max_reproj_error);
  } else if (use_quality_residual_adaptive) {
    const double robust_base =
        Quantile(residual_norms, 0.75) +
        options.filter_internal_corner_sigma_threshold *
            std::max(0.0, Quantile(residual_norms, 0.75) -
                              Quantile(residual_norms, 0.25));
    effective_threshold = std::max(
        std::max(threshold, robust_base),
        options.filter_internal_corner_adaptive_min_threshold_px);
    if (has_absolute_cap) {
      effective_threshold = std::min(
          effective_threshold, options.filter_internal_corner_max_reproj_error);
    }
  }

  for (std::size_t index = 0; index < internal_points.size(); ++index) {
    JointPointObservation* point = internal_points[index];
    const double residual = per_point_residuals[index];
    double per_point_threshold = effective_threshold;
    bool low_quality = false;
    if (!gross_topology_only && use_quality_residual_adaptive) {
      const double quality = ClampUnitLocal(point->quality);
      low_quality = quality < options.filter_internal_corner_quality_min;
      if (low_quality) {
        const double low_quality_threshold = std::max(
            options.filter_internal_corner_min_reproj_error,
            0.5 * effective_threshold);
        per_point_threshold = std::min(per_point_threshold, low_quality_threshold);
      } else {
        const double relaxation =
            options.filter_internal_corner_quality_relaxation_px *
            (quality - options.filter_internal_corner_quality_min) /
            std::max(1e-9, 1.0 - options.filter_internal_corner_quality_min);
        per_point_threshold += std::max(0.0, relaxation);
        if (has_absolute_cap) {
          per_point_threshold =
              std::min(per_point_threshold,
                       options.filter_internal_corner_max_reproj_error +
                           std::max(0.0, options.filter_internal_corner_quality_relaxation_px));
        }
      }
    }
    const bool invalid_projection = !std::isfinite(residual);
    const bool over_threshold =
        residual > per_point_threshold &&
        residual >
            (gross_topology_only
                 ? options.gross_internal_topology_min_reproj_error_px
                 : options.filter_internal_corner_min_reproj_error);
    if (!invalid_projection && !over_threshold) {
      continue;
    }
    point->used_in_solver = false;
    point->rejection_reason_code = JointRejectionReasonCode::InternalPointReprojectionOutlier;
    std::ostringstream detail;
    if (invalid_projection) {
      detail << "projection invalid under outer-only pose refit";
    } else {
      detail << "reprojection_error=" << residual
             << " threshold=" << per_point_threshold
             << " sigma_threshold=" << threshold
             << " filter_mode="
             << (gross_topology_only
                     ? "gross_internal_topology_outlier"
                     : options.filter_internal_corner_mode)
             << " mean=" << mean_residual
             << " std=" << std_residual
             << " pose_fit_outer_rmse=" << pose_fit_rmse;
      if (!gross_topology_only && use_quality_residual_adaptive) {
        detail << " base_threshold=" << effective_threshold
               << " point_quality=" << ClampUnitLocal(point->quality)
               << " low_quality=" << (low_quality ? 1 : 0)
               << " quality_min=" << options.filter_internal_corner_quality_min
               << " quality_relaxation_px="
               << options.filter_internal_corner_quality_relaxation_px
               << " adaptive_min_threshold_px="
               << options.filter_internal_corner_adaptive_min_threshold_px;
      }
      if (gross_topology_only) {
        detail << " gross_min_reproj_error_px="
               << options.gross_internal_topology_min_reproj_error_px
               << " gross_sigma_threshold="
               << options.gross_internal_topology_sigma_threshold
               << " gross_max_outer_pose_rmse_px="
               << options.gross_internal_topology_max_outer_pose_rmse_px;
      }
    }
    point->rejection_detail = detail.str();
  }
}

}  // namespace

const char* ToString(JointPointType point_type) {
  switch (point_type) {
    case JointPointType::Outer:
      return "outer";
    case JointPointType::Internal:
      return "internal";
  }
  return "unknown";
}

const char* ToString(JointRejectionReasonCode reason_code) {
  switch (reason_code) {
    case JointRejectionReasonCode::None:
      return "none";
    case JointRejectionReasonCode::FrameNotFoundInBootstrap:
      return "frame_not_found_in_bootstrap";
    case JointRejectionReasonCode::FrameLabelMismatch:
      return "frame_label_mismatch";
    case JointRejectionReasonCode::FrameNotInitialized:
      return "frame_not_initialized";
    case JointRejectionReasonCode::BoardNotInitialized:
      return "board_not_initialized";
    case JointRejectionReasonCode::NotReferenceConnected:
      return "not_reference_connected";
    case JointRejectionReasonCode::MissingOuterBoardObservation:
      return "missing_outer_board_observation";
    case JointRejectionReasonCode::OuterMeasurementInvalid:
      return "outer_measurement_invalid";
    case JointRejectionReasonCode::MissingRegeneratedBoardResult:
      return "missing_regenerated_board_result";
    case JointRejectionReasonCode::InternalRegenerationFailed:
      return "internal_regeneration_failed";
    case JointRejectionReasonCode::InternalPointInvalid:
      return "internal_point_invalid";
    case JointRejectionReasonCode::DuplicateInternalImageLocation:
      return "duplicate_internal_image_location";
    case JointRejectionReasonCode::InternalPointReprojectionOutlier:
      return "internal_point_reprojection_outlier";
  }
  return "unknown";
}

const char* ToString(JointObservationSourceKind source_kind) {
  switch (source_kind) {
    case JointObservationSourceKind::OuterMeasurement:
      return "outer_measurement";
    case JointObservationSourceKind::InternalMeasurement:
      return "internal_measurement";
  }
  return "unknown";
}

JointReprojectionMeasurementBuilder::JointReprojectionMeasurementBuilder(
    ApriltagInternalConfig base_config,
    JointMeasurementBuildOptions options)
    : base_config_(std::move(base_config)), options_(std::move(options)) {}

JointMeasurementBuildResult JointReprojectionMeasurementBuilder::Build(
    const std::vector<JointMeasurementFrameInput>& frames,
    const OuterBootstrapResult& bootstrap_result) const {
  JointMeasurementBuildResult result;
  result.reference_board_id = bootstrap_result.reference_board_id;
  result.bootstrap_seed = bootstrap_result;

  if (!bootstrap_result.success) {
    result.failure_reason = "bootstrap_result.success is false";
    return result;
  }

  if (options_.reference_board_id != bootstrap_result.reference_board_id) {
    std::ostringstream warning;
    warning << "joint measurement builder reference_board_id=" << options_.reference_board_id
            << " differs from bootstrap_result.reference_board_id="
            << bootstrap_result.reference_board_id
            << "; using bootstrap_result.reference_board_id";
    AppendUniqueWarning(warning.str(), &result.warnings);
  }

  std::set<int> used_frame_indices;
  std::set<std::pair<int, int> > used_board_keys;
  std::set<std::pair<int, int> > accepted_outer_board_keys;
  std::set<std::pair<int, int> > accepted_internal_board_keys;

  result.frames.reserve(frames.size());
  for (std::size_t frame_storage_index = 0; frame_storage_index < frames.size(); ++frame_storage_index) {
    const JointMeasurementFrameInput& frame_input = frames[frame_storage_index];

    JointMeasurementFrameResult frame_result;
    frame_result.frame_index = frame_input.frame_index;
    frame_result.frame_label = frame_input.frame_label;

    const OuterBootstrapFrameState* frame_state =
        FindBootstrapFrameState(bootstrap_result, frame_input.frame_index);
    frame_result.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;

    const bool label_mismatch =
        frame_state != nullptr &&
        !frame_input.frame_label.empty() &&
        !frame_state->frame_label.empty() &&
        frame_input.frame_label != frame_state->frame_label;
    if (frame_state == nullptr) {
      std::ostringstream warning;
      warning << "frame_index " << frame_input.frame_index
              << " is missing from bootstrap_result.frames";
      AppendUniqueWarning(warning.str(), &result.warnings);
    } else if (label_mismatch) {
      std::ostringstream warning;
      warning << "frame_index " << frame_input.frame_index
              << " label mismatch input=" << frame_input.frame_label
              << " bootstrap=" << frame_state->frame_label;
      AppendUniqueWarning(warning.str(), &result.warnings);
    }

    frame_result.visible_board_ids = frame_input.regenerated_internal.visible_board_ids;
    for (const OuterBoardMeasurement& measurement :
         frame_input.outer_detections.frame_measurements.board_measurements) {
      if (measurement.success) {
        AppendUniqueBoardId(measurement.board_id, &frame_result.visible_board_ids);
      }
    }

    const std::vector<int> board_ids = CollectBoardIds(frame_input);
    frame_result.board_observations.reserve(board_ids.size());
    for (int board_id : board_ids) {
      JointBoardObservation board_observation;
      board_observation.board_id = board_id;
      board_observation.frame_bootstrap_initialized =
          frame_state != nullptr && frame_state->initialized;

      const OuterBootstrapBoardState* board_state =
          FindBootstrapBoardState(bootstrap_result, board_id);
      board_observation.board_bootstrap_initialized =
          board_state != nullptr && board_state->initialized;

      const OuterBootstrapObservationDiagnostics* observation_diagnostics =
          FindObservationDiagnostics(bootstrap_result, frame_input.frame_index, board_id);
      board_observation.reference_connected =
          observation_diagnostics != nullptr && observation_diagnostics->reference_connected;

      const JointRejectionReasonCode board_level_reason = ComputeBoardLevelReason(
          frame_state, board_state, observation_diagnostics,
          options_.require_initialized_frame_and_board);
      const std::string board_level_detail = BuildBoardLevelReasonDetail(
          frame_input, frame_state, board_state, observation_diagnostics);

      const ApriltagCanonicalModel model = ModelForBoardId(board_id);
      const std::array<CanonicalCorner, 4> outer_corners = OuterCanonicalCorners(model);

      int outer_measurement_index = -1;
      const OuterBoardMeasurement* outer_measurement =
          FindOuterBoardMeasurement(frame_input, board_id, &outer_measurement_index);
      int regenerated_measurement_index = -1;
      const RegeneratedBoardMeasurement* regenerated_measurement =
          FindRegeneratedBoardMeasurement(frame_input, board_id,
                                          &regenerated_measurement_index);
      OuterBoardMeasurement regenerated_rescued_outer_measurement;
      if (options_.use_regenerated_rescued_outer_measurements &&
          regenerated_measurement != nullptr &&
          regenerated_measurement->detection.outer_detection.success &&
          regenerated_measurement->detection.outer_detection.used_local_patch_rescue &&
          (outer_measurement == nullptr || !outer_measurement->success)) {
        regenerated_rescued_outer_measurement =
            BuildOuterBoardMeasurementFromDetection(
                regenerated_measurement->detection.outer_detection);
        outer_measurement = &regenerated_rescued_outer_measurement;
        outer_measurement_index = regenerated_measurement_index;
      }
      // A board observation is only geometrically usable when all four
      // outer corners have passed refinement.  This must apply to direct
      // detections as well as rescue detections: keeping three outer points
      // (or internal points) can create a plausible-looking but wrong board
      // pose when one image edge is occluded or lacks line support.
      bool outer_refinement_invalid = false;
      if (outer_measurement != nullptr) {
        outer_refinement_invalid =
            !outer_measurement->success ||
            outer_measurement->valid_refined_corner_count != 4;
        for (bool valid : outer_measurement->refined_corner_valid) {
          outer_refinement_invalid = outer_refinement_invalid || !valid;
        }
      }
      // Supporting-line coverage is diagnostic-only. Under a wide-angle
      // projection a visually valid curved border can have weak straight-line
      // support. The independently refined outer corners and regenerated
      // internal lattice remain the authoritative image measurements.
      const bool board_geometry_invalid = outer_refinement_invalid;
      bool internal_regeneration_failed_for_board = false;
      std::string internal_regeneration_failure_detail;
      const bool reject_entire_board_observation =
          regenerated_measurement != nullptr &&
          regenerated_measurement->detection.reject_entire_board_observation;
      if ((reject_entire_board_observation ||
           !options_.include_outer_when_internal_failed) &&
          outer_measurement != nullptr && outer_measurement->success) {
        if (regenerated_measurement == nullptr) {
          internal_regeneration_failed_for_board = true;
          internal_regeneration_failure_detail =
              "missing regenerated internal measurement";
        } else if (!regenerated_measurement->detection.success) {
          internal_regeneration_failed_for_board = true;
          internal_regeneration_failure_detail =
              regenerated_measurement->detection.failure_reason.empty()
                  ? "internal regeneration failed"
                  : regenerated_measurement->detection.failure_reason;
        }
      }
      if (options_.include_outer_points && outer_measurement != nullptr) {
        for (int corner_index = 0; corner_index < 4; ++corner_index) {
          JointPointObservation point;
          point.frame_index = frame_input.frame_index;
          point.frame_label = frame_input.frame_label;
          point.board_id = board_id;
          point.point_id = outer_corners[static_cast<std::size_t>(corner_index)].point_id;
          point.point_type = JointPointType::Outer;
          point.image_xy =
              outer_measurement->refined_outer_corners_original_image[static_cast<std::size_t>(corner_index)];
          point.target_xyz_board =
              outer_corners[static_cast<std::size_t>(corner_index)].target_xyz;
          point.quality = outer_measurement->detection_quality;
          point.frame_storage_index = static_cast<int>(frame_storage_index);
          point.source_board_observation_index = outer_measurement_index;
          point.source_point_index = corner_index;
          point.source_kind = JointObservationSourceKind::OuterMeasurement;
          const OuterCornerVerificationDebugInfo& outer_debug =
              outer_measurement->corner_verification_debug[static_cast<std::size_t>(corner_index)];
          point.outer_subpix_window_radius = outer_debug.subpix_window_radius;
          point.outer_pre_boost_subpix_window_radius =
              outer_debug.pre_boost_subpix_window_radius;
          point.outer_boosted_raw_subpix_window_radius =
              outer_debug.boosted_raw_subpix_window_radius;
          point.outer_close_edge_subpix_boost_applied =
              outer_debug.close_edge_subpix_boost_applied;
          point.outer_close_edge_subpix_area_ratio =
              outer_debug.close_edge_subpix_area_ratio;
          point.outer_close_edge_subpix_max_polar_deg =
              outer_debug.close_edge_subpix_max_polar_deg;

          if (label_mismatch) {
            point.rejection_detail = BuildBoardLevelReasonDetail(
                frame_input, frame_state, board_state, observation_diagnostics);
          }
          if (board_level_reason != JointRejectionReasonCode::None) {
            point.rejection_reason_code = board_level_reason;
            point.rejection_detail = board_level_detail;
          } else if (reject_entire_board_observation) {
            point.rejection_reason_code =
                JointRejectionReasonCode::InternalRegenerationFailed;
            point.rejection_detail =
                regenerated_measurement->detection.failure_reason.empty()
                    ? "rescued board rejected by downstream image evidence"
                    : regenerated_measurement->detection.failure_reason;
          } else if (board_geometry_invalid) {
            point.rejection_reason_code =
                JointRejectionReasonCode::OuterMeasurementInvalid;
            point.rejection_detail =
                "board rejected because all four outer corners were not refined";
          } else if (!outer_measurement->success ||
                     !outer_measurement->refined_corner_valid[
                         static_cast<std::size_t>(corner_index)]) {
            point.rejection_reason_code = JointRejectionReasonCode::OuterMeasurementInvalid;
            const std::string& corner_failure_reason =
                outer_debug.failure_reason.empty()
                    ? outer_measurement->failure_reason_text
                    : outer_debug.failure_reason;
            if (!corner_failure_reason.empty()) {
              point.rejection_detail = corner_failure_reason;
            } else {
              std::ostringstream detail;
              detail << "success=" << (outer_measurement->success ? 1 : 0)
                     << " corner_index=" << corner_index
                     << " refined_valid=0";
              point.rejection_detail = detail.str();
            }
          } else if (internal_regeneration_failed_for_board &&
                     !reject_entire_board_observation &&
                     !(options_.include_rescued_outer_when_internal_failed &&
                       outer_measurement->used_local_patch_rescue)) {
            point.rejection_reason_code =
                JointRejectionReasonCode::InternalRegenerationFailed;
            point.rejection_detail = internal_regeneration_failure_detail;
          } else {
            point.used_in_solver = true;
            if (internal_regeneration_failed_for_board &&
                !reject_entire_board_observation &&
                outer_measurement->used_local_patch_rescue) {
              point.rejection_detail =
                  "rescued outer retained despite internal regeneration failure: " +
                  internal_regeneration_failure_detail;
            }
          }

          board_observation.points.push_back(point);
        }
      }

      if (options_.include_internal_points) {
        if (regenerated_measurement == nullptr) {
          if (options_.include_outer_points && outer_measurement != nullptr) {
            std::ostringstream warning;
            warning << "frame " << frame_input.frame_index << " board " << board_id
                    << " is missing regenerated internal measurement";
            AppendUniqueWarning(warning.str(), &result.warnings);
          }
        } else {
          const ApriltagInternalDetectionResult& detection = regenerated_measurement->detection;
          std::set<int> attempted_internal_point_ids;
          for (const InternalCornerDebugInfo& debug :
               detection.internal_corner_debug) {
            if (debug.corner_type != CornerType::Outer && debug.point_id >= 0) {
              attempted_internal_point_ids.insert(debug.point_id);
            }
          }
          for (std::size_t point_index = 0; point_index < detection.corners.size(); ++point_index) {
            const CornerMeasurement& measurement = detection.corners[point_index];
            if (measurement.corner_type == CornerType::Outer) {
              continue;
            }
            // MakeDefaultMeasurements contains the canonical union of all
            // possible points. Only points visited by this board's active
            // lattice are observations. Treating the untouched defaults as
            // invalid measurements inflated diagnostics and rendered dozens
            // of false crosses for every board, especially after pose failure.
            if (attempted_internal_point_ids.count(measurement.point_id) == 0) {
              continue;
            }

            JointPointObservation point;
            point.frame_index = frame_input.frame_index;
            point.frame_label = frame_input.frame_label;
            point.board_id = board_id;
            point.point_id = measurement.point_id;
            point.point_type = JointPointType::Internal;
            point.image_xy = measurement.image_xy;
            point.target_xyz_board = measurement.target_xyz;
            point.quality = measurement.quality;
            point.observation_weight = 1.0;
            point.frame_storage_index = static_cast<int>(frame_storage_index);
            point.source_board_observation_index = regenerated_measurement_index;
            point.source_point_index = static_cast<int>(point_index);
            point.source_kind = JointObservationSourceKind::InternalMeasurement;

            if (board_level_reason != JointRejectionReasonCode::None) {
              point.rejection_reason_code = board_level_reason;
              point.rejection_detail = board_level_detail;
            } else if (board_geometry_invalid) {
              point.rejection_reason_code =
                  JointRejectionReasonCode::OuterMeasurementInvalid;
              point.rejection_detail =
                  "board rejected because all four outer corners were not refined";
            } else if (reject_entire_board_observation ||
                      (!options_.include_outer_when_internal_failed &&
                       !detection.success)) {
              point.rejection_reason_code =
                  JointRejectionReasonCode::InternalRegenerationFailed;
              point.rejection_detail =
                  detection.failure_reason.empty()
                      ? "internal regeneration failed"
                      : detection.failure_reason;
            } else if (!measurement.valid) {
              point.rejection_reason_code = JointRejectionReasonCode::InternalPointInvalid;
              point.rejection_detail = "CornerMeasurement.valid is false";
            } else {
              point.used_in_solver = true;
            }

            board_observation.points.push_back(point);
          }
        }
      }

      const BoardTopologyConsistencyOutcome topology_outcome =
          EnforceInternalBoardTopologyConsistency(
              model, options_, &board_observation);
      if (topology_outcome.rejected_internal_point_count > 0) {
        std::ostringstream warning;
        warning << "frame " << frame_input.frame_index << " board "
                << board_id << " topology consistency rejected internal="
                << topology_outcome.rejected_internal_point_count << " "
                << topology_outcome.diagnostic;
        AppendUniqueWarning(warning.str(), &result.warnings);
      }

      if (!options_.robust_missing_board_recovery) {
        FilterInternalPointsByReprojectionError(
            bootstrap_result.coarse_camera, options_, &board_observation,
            false);
      } else {
        FilterInternalPointsByReprojectionError(
            bootstrap_result.coarse_camera, options_, &board_observation,
            true);
      }
      const int duplicate_internal_rejected_count =
          RejectDuplicateInternalImageLocations(options_, &board_observation);
      if (duplicate_internal_rejected_count > 0) {
        std::ostringstream warning;
        warning << "frame " << frame_input.frame_index << " board " << board_id
                << " rejected " << duplicate_internal_rejected_count
                << " internal point(s) with duplicate refined image locations";
        AppendUniqueWarning(warning.str(), &result.warnings);
      }

      board_observation.outer_point_count =
          CountUsedPoints(board_observation, JointPointType::Outer);
      board_observation.internal_point_count =
          CountUsedPoints(board_observation, JointPointType::Internal);
      board_observation.used_in_solver =
          board_observation.outer_point_count > 0 || board_observation.internal_point_count > 0;

      if (board_observation.outer_point_count == 4) {
        accepted_outer_board_keys.insert(std::make_pair(frame_input.frame_index, board_id));
      }
      if (board_observation.internal_point_count > 0) {
        accepted_internal_board_keys.insert(std::make_pair(frame_input.frame_index, board_id));
      }
      if (board_observation.used_in_solver) {
        used_frame_indices.insert(frame_input.frame_index);
        used_board_keys.insert(std::make_pair(frame_input.frame_index, board_id));
      }

      for (const JointPointObservation& point : board_observation.points) {
        if (point.used_in_solver) {
          result.solver_observations.push_back(point);
        }
      }

      frame_result.board_observations.push_back(board_observation);
    }

    result.frames.push_back(frame_result);
  }

  result.used_frame_count = static_cast<int>(used_frame_indices.size());
  result.accepted_outer_board_observation_count =
      static_cast<int>(accepted_outer_board_keys.size());
  result.accepted_internal_board_observation_count =
      static_cast<int>(accepted_internal_board_keys.size());
  result.used_board_observation_count = static_cast<int>(used_board_keys.size());
  result.used_outer_point_count = 0;
  for (const JointPointObservation& point : result.solver_observations) {
    if (point.point_type == JointPointType::Outer) {
      ++result.used_outer_point_count;
    }
  }
  result.used_internal_point_count = 0;
  for (const JointPointObservation& point : result.solver_observations) {
    if (point.point_type == JointPointType::Internal) {
      ++result.used_internal_point_count;
    }
  }
  result.used_total_point_count = static_cast<int>(result.solver_observations.size());

  if (result.used_outer_point_count + result.used_internal_point_count !=
      result.used_total_point_count) {
    std::ostringstream stream;
    stream << "count mismatch: outer=" << result.used_outer_point_count
           << " internal=" << result.used_internal_point_count
           << " total=" << result.used_total_point_count;
    result.failure_reason = stream.str();
    return result;
  }

  if (result.used_total_point_count == 0) {
    result.failure_reason = "No solver-ready joint observations were built.";
    return result;
  }

  ApplyInternalObservationQualityWeightsToResult(&result, options_);

  result.success = true;
  return result;
}

void JointReprojectionMeasurementBuilder::DrawFrameOverlay(
    const cv::Mat& image,
    const JointMeasurementFrameResult& frame_result,
    cv::Mat* output_image) const {
  if (output_image == nullptr) {
    throw std::runtime_error("DrawFrameOverlay requires a valid output pointer.");
  }
  *output_image = image.clone();

  int used_outer = 0;
  int used_internal = 0;
  int rejected_points = 0;
  for (const JointBoardObservation& board_observation : frame_result.board_observations) {
    for (const JointPointObservation& point : board_observation.points) {
      if (!point.used_in_solver) {
        ++rejected_points;
        continue;
      }

      const cv::Point pixel(static_cast<int>(std::lround(point.image_xy.x())),
                            static_cast<int>(std::lround(point.image_xy.y())));
      const cv::Scalar color = point.point_type == JointPointType::Outer
                                   ? cv::Scalar(60, 220, 80)
                                   : cv::Scalar(40, 180, 255);
      const int radius = point.point_type == JointPointType::Outer ? 5 : 3;
      cv::circle(*output_image, pixel, radius, color, 2, cv::LINE_AA);
      if (point.point_type == JointPointType::Outer) {
        ++used_outer;
      } else {
        ++used_internal;
      }
    }

    if (board_observation.used_in_solver) {
      for (const JointPointObservation& point : board_observation.points) {
        if (point.used_in_solver) {
          const cv::Point anchor(static_cast<int>(std::lround(point.image_xy.x())),
                                 static_cast<int>(std::lround(point.image_xy.y())));
          std::ostringstream label;
          label << "#" << board_observation.board_id;
          cv::putText(*output_image, label.str(), anchor + cv::Point(6, -6),
                      cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
          break;
        }
      }
    }
  }

  const int banner_height = 76;
  cv::rectangle(*output_image, cv::Rect(0, 0, output_image->cols, banner_height),
                cv::Scalar(18, 18, 18), cv::FILLED);

  std::ostringstream header;
  header << "frame " << frame_result.frame_index
         << " bootstrap_frame=" << (frame_result.frame_bootstrap_initialized ? "yes" : "no")
         << " used_outer=" << used_outer
         << " used_internal=" << used_internal
         << " rejected=" << rejected_points;
  cv::putText(*output_image, header.str(), cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.62,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

  std::ostringstream board_line;
  board_line << "visible=" << JoinBoardIds(frame_result.visible_board_ids);
  cv::putText(*output_image, board_line.str(), cv::Point(18, 54), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(185, 185, 185), 1, cv::LINE_AA);
}

ApriltagCanonicalModel JointReprojectionMeasurementBuilder::ModelForBoardId(int board_id) const {
  ApriltagInternalConfig config = base_config_;
  config.tag_id = board_id;
  config.tag_ids.clear();
  config.outer_detector_config.tag_id = board_id;
  config.outer_detector_config.tag_ids.clear();
  return ApriltagCanonicalModel(config);
}

const OuterBootstrapFrameState* JointReprojectionMeasurementBuilder::FindBootstrapFrameState(
    const OuterBootstrapResult& bootstrap_result,
    int frame_index) const {
  for (const OuterBootstrapFrameState& frame_state : bootstrap_result.frames) {
    if (frame_state.frame_index == frame_index) {
      return &frame_state;
    }
  }
  return nullptr;
}

const OuterBootstrapBoardState* JointReprojectionMeasurementBuilder::FindBootstrapBoardState(
    const OuterBootstrapResult& bootstrap_result,
    int board_id) const {
  for (const OuterBootstrapBoardState& board_state : bootstrap_result.boards) {
    if (board_state.board_id == board_id) {
      return &board_state;
    }
  }
  return nullptr;
}

const OuterBootstrapObservationDiagnostics*
JointReprojectionMeasurementBuilder::FindObservationDiagnostics(
    const OuterBootstrapResult& bootstrap_result,
    int frame_index,
    int board_id) const {
  for (const OuterBootstrapObservationDiagnostics& diagnostics :
       bootstrap_result.observation_diagnostics) {
    if (diagnostics.frame_index == frame_index && diagnostics.board_id == board_id) {
      return &diagnostics;
    }
  }
  return nullptr;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
