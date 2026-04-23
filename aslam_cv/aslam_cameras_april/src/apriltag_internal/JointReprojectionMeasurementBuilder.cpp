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

void FilterInternalPointsByReprojectionError(
    const OuterBootstrapCameraIntrinsics& camera,
    const JointMeasurementBuildOptions& options,
    JointBoardObservation* board_observation) {
  if (board_observation == nullptr || !options.filter_internal_corner_outliers) {
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

  for (std::size_t index = 0; index < internal_points.size(); ++index) {
    JointPointObservation* point = internal_points[index];
    const double residual = per_point_residuals[index];
    const bool invalid_projection = !std::isfinite(residual);
    const bool over_threshold =
        residual > threshold &&
        residual > options.filter_internal_corner_min_reproj_error;
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
             << " threshold=" << threshold
             << " mean=" << mean_residual
             << " std=" << std_residual
             << " pose_fit_outer_rmse=" << pose_fit_rmse;
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
    case JointRejectionReasonCode::InternalPointInvalid:
      return "internal_point_invalid";
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
      if (options_.include_outer_points && outer_measurement != nullptr) {
        bool outer_measurement_valid = outer_measurement->success;
        for (bool valid : outer_measurement->refined_corner_valid) {
          outer_measurement_valid = outer_measurement_valid && valid;
        }

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

          if (label_mismatch) {
            point.rejection_detail = BuildBoardLevelReasonDetail(
                frame_input, frame_state, board_state, observation_diagnostics);
          }
          if (board_level_reason != JointRejectionReasonCode::None) {
            point.rejection_reason_code = board_level_reason;
            point.rejection_detail = board_level_detail;
          } else if (!outer_measurement_valid) {
            point.rejection_reason_code = JointRejectionReasonCode::OuterMeasurementInvalid;
            if (!outer_measurement->failure_reason_text.empty()) {
              point.rejection_detail = outer_measurement->failure_reason_text;
            } else {
              std::ostringstream detail;
              detail << "success=" << (outer_measurement->success ? 1 : 0)
                     << " valid_refined_corner_count="
                     << outer_measurement->valid_refined_corner_count;
              point.rejection_detail = detail.str();
            }
          } else {
            point.used_in_solver = true;
          }

          board_observation.points.push_back(point);
        }
      }

      int regenerated_measurement_index = -1;
      const RegeneratedBoardMeasurement* regenerated_measurement =
          FindRegeneratedBoardMeasurement(frame_input, board_id, &regenerated_measurement_index);
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
          for (std::size_t point_index = 0; point_index < detection.corners.size(); ++point_index) {
            const CornerMeasurement& measurement = detection.corners[point_index];
            if (measurement.corner_type == CornerType::Outer) {
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
            point.frame_storage_index = static_cast<int>(frame_storage_index);
            point.source_board_observation_index = regenerated_measurement_index;
            point.source_point_index = static_cast<int>(point_index);
            point.source_kind = JointObservationSourceKind::InternalMeasurement;

            if (board_level_reason != JointRejectionReasonCode::None) {
              point.rejection_reason_code = board_level_reason;
              point.rejection_detail = board_level_detail;
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

      FilterInternalPointsByReprojectionError(
          bootstrap_result.coarse_camera, options_, &board_observation);

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
  result.used_outer_point_count = 4 * result.accepted_outer_board_observation_count;
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
