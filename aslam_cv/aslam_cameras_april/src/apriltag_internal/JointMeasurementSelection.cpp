#include <aslam/cameras/apriltag_internal/JointMeasurementSelection.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

struct CandidateBoardObservation {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double average_quality = 0.0;
  double rmse = std::numeric_limits<double>::infinity();
  double pose_fit_outer_rmse = std::numeric_limits<double>::infinity();
  std::string coverage_signature;
};

void AppendUniqueInt(int value, std::vector<int>* values) {
  if (values == nullptr) {
    return;
  }
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(value);
  }
}

double ComputeMedian(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  if ((values.size() % 2) == 0) {
    return 0.5 * (values[mid - 1] + values[mid]);
  }
  return values[mid];
}

double AverageQuality(const JointBoardObservation& board_observation) {
  double quality_sum = 0.0;
  int count = 0;
  for (const JointPointObservation& point : board_observation.points) {
    if (!point.used_in_solver) {
      continue;
    }
    quality_sum += point.quality;
    ++count;
  }
  return count > 0 ? quality_sum / static_cast<double>(count) : 0.0;
}

std::vector<Eigen::Vector2d> UsedPointImagePositions(const JointBoardObservation& board_observation) {
  std::vector<Eigen::Vector2d> positions;
  for (const JointPointObservation& point : board_observation.points) {
    if (point.used_in_solver) {
      positions.push_back(point.image_xy);
    }
  }
  return positions;
}

double AverageOuterEdgeLength(const JointBoardObservation& board_observation) {
  std::vector<Eigen::Vector2d> outer_points;
  outer_points.reserve(4);
  for (const JointPointObservation& point : board_observation.points) {
    if (point.used_in_solver && point.point_type == JointPointType::Outer) {
      outer_points.push_back(point.image_xy);
    }
  }
  if (outer_points.size() == 4) {
    double edge_length_sum = 0.0;
    for (std::size_t index = 0; index < outer_points.size(); ++index) {
      edge_length_sum += (outer_points[index] -
          outer_points[(index + 1) % outer_points.size()]).norm();
    }
    return edge_length_sum / 4.0;
  }

  const std::vector<Eigen::Vector2d> positions = UsedPointImagePositions(board_observation);
  if (positions.size() < 2) {
    return 0.0;
  }
  double max_distance = 0.0;
  for (std::size_t i = 0; i < positions.size(); ++i) {
    for (std::size_t j = i + 1; j < positions.size(); ++j) {
      max_distance = std::max(max_distance, (positions[i] - positions[j]).norm());
    }
  }
  return max_distance;
}

int ScaleBinForBoardObservation(const JointBoardObservation& board_observation,
                                const cv::Size& image_size) {
  const double average_scale =
      AverageOuterEdgeLength(board_observation) /
      static_cast<double>(std::max(image_size.width, image_size.height));
  if (average_scale < 0.04) {
    return 0;
  }
  if (average_scale < 0.08) {
    return 1;
  }
  if (average_scale < 0.14) {
    return 2;
  }
  return 3;
}

std::string BuildCoverageSignature(const JointBoardObservation& board_observation,
                                   const cv::Size& image_size,
                                   int grid_cols,
                                   int grid_rows) {
  const std::vector<Eigen::Vector2d> positions = UsedPointImagePositions(board_observation);
  if (positions.empty()) {
    return "";
  }

  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const Eigen::Vector2d& point : positions) {
    centroid += point;
  }
  centroid /= static_cast<double>(positions.size());

  const int cell_x = std::max(0, std::min(grid_cols - 1,
      static_cast<int>(std::floor(grid_cols * centroid.x() /
                                  std::max(1, image_size.width)))));
  const int cell_y = std::max(0, std::min(grid_rows - 1,
      static_cast<int>(std::floor(grid_rows * centroid.y() /
                                  std::max(1, image_size.height)))));
  const int scale_bin = ScaleBinForBoardObservation(board_observation, image_size);

  std::ostringstream stream;
  stream << cell_x << ":" << cell_y << ":" << scale_bin;
  return stream.str();
}

double ComputeOuterPoseFitRmse(const JointBoardObservation& board_observation,
                               const OuterBootstrapCameraIntrinsics& camera) {
  if (!camera.IsValid()) {
    return std::numeric_limits<double>::infinity();
  }

  std::vector<Eigen::Vector3d> outer_targets;
  std::vector<cv::Point2f> outer_pixels;
  outer_targets.reserve(4);
  outer_pixels.reserve(4);
  for (const JointPointObservation& point : board_observation.points) {
    if (!point.used_in_solver || point.point_type != JointPointType::Outer) {
      continue;
    }
    outer_targets.push_back(point.target_xyz_board);
    outer_pixels.push_back(
        cv::Point2f(static_cast<float>(point.image_xy.x()),
                    static_cast<float>(point.image_xy.y())));
  }

  if (outer_targets.size() < 4) {
    return std::numeric_limits<double>::infinity();
  }

  Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
  double pose_fit_outer_rmse = std::numeric_limits<double>::infinity();
  if (!EstimatePoseFromObjectPoints(camera, outer_targets, outer_pixels,
                                    &T_camera_board, &pose_fit_outer_rmse)) {
    return std::numeric_limits<double>::infinity();
  }
  return pose_fit_outer_rmse;
}

void RecomputeMeasurementCounts(JointMeasurementBuildResult* result) {
  if (result == nullptr) {
    return;
  }

  result->solver_observations.clear();
  result->used_frame_count = 0;
  result->accepted_outer_board_observation_count = 0;
  result->accepted_internal_board_observation_count = 0;
  result->used_board_observation_count = 0;
  result->used_outer_point_count = 0;
  result->used_internal_point_count = 0;
  result->used_total_point_count = 0;

  std::set<int> used_frames;
  std::set<std::pair<int, int> > used_board_keys;
  std::set<std::pair<int, int> > accepted_outer_keys;
  std::set<std::pair<int, int> > accepted_internal_keys;
  for (JointMeasurementFrameResult& frame_result : result->frames) {
    for (JointBoardObservation& board_observation : frame_result.board_observations) {
      board_observation.used_in_solver = false;
      board_observation.outer_point_count = 0;
      board_observation.internal_point_count = 0;
      for (JointPointObservation& point : board_observation.points) {
        if (!point.used_in_solver) {
          continue;
        }
        result->solver_observations.push_back(point);
        board_observation.used_in_solver = true;
        used_frames.insert(frame_result.frame_index);
        used_board_keys.insert(std::make_pair(frame_result.frame_index, board_observation.board_id));
        if (point.point_type == JointPointType::Outer) {
          ++board_observation.outer_point_count;
          ++result->used_outer_point_count;
          accepted_outer_keys.insert(std::make_pair(frame_result.frame_index, board_observation.board_id));
        } else {
          ++board_observation.internal_point_count;
          ++result->used_internal_point_count;
          accepted_internal_keys.insert(std::make_pair(frame_result.frame_index, board_observation.board_id));
        }
      }
    }
  }

  result->used_frame_count = static_cast<int>(used_frames.size());
  result->used_board_observation_count = static_cast<int>(used_board_keys.size());
  result->accepted_outer_board_observation_count = static_cast<int>(accepted_outer_keys.size());
  result->accepted_internal_board_observation_count = static_cast<int>(accepted_internal_keys.size());
  result->used_total_point_count =
      result->used_outer_point_count + result->used_internal_point_count;
  result->success = result->used_total_point_count > 0;
}

}  // namespace

const char* ToString(JointFrameSelectionReasonCode reason_code) {
  switch (reason_code) {
    case JointFrameSelectionReasonCode::None:
      return "None";
    case JointFrameSelectionReasonCode::NoUsableBoardObservations:
      return "NoUsableBoardObservations";
    case JointFrameSelectionReasonCode::AcceptedMinViewsPerBoard:
      return "AcceptedMinViewsPerBoard";
    case JointFrameSelectionReasonCode::AcceptedNewBoardPair:
      return "AcceptedNewBoardPair";
    case JointFrameSelectionReasonCode::AcceptedNewImageCoverage:
      return "AcceptedNewImageCoverage";
    case JointFrameSelectionReasonCode::RejectedRedundantView:
      return "RejectedRedundantView";
  }
  return "Unknown";
}

const char* ToString(JointBoardObservationSelectionReasonCode reason_code) {
  switch (reason_code) {
    case JointBoardObservationSelectionReasonCode::None:
      return "None";
    case JointBoardObservationSelectionReasonCode::Accepted:
      return "Accepted";
    case JointBoardObservationSelectionReasonCode::RejectedNotSolverReady:
      return "RejectedNotSolverReady";
    case JointBoardObservationSelectionReasonCode::RejectedResidualSanity:
      return "RejectedResidualSanity";
    case JointBoardObservationSelectionReasonCode::RejectedOuterPoseFit:
      return "RejectedOuterPoseFit";
    case JointBoardObservationSelectionReasonCode::RejectedFrameRejected:
      return "RejectedFrameRejected";
  }
  return "Unknown";
}

JointMeasurementSelection::JointMeasurementSelection(
    JointMeasurementSelectionOptions options)
    : options_(std::move(options)) {}

JointMeasurementSelectionResult JointMeasurementSelection::Select(
    const JointMeasurementBuildResult& measurement_result,
    const JointResidualEvaluationResult& residual_result,
    const JointReprojectionSceneState& scene_state) const {
  JointMeasurementSelectionResult result;
  result.reference_board_id = scene_state.reference_board_id;
  result.selected_measurement_result = measurement_result;

  if (!measurement_result.success) {
    result.failure_reason = "measurement_result.success is false";
    return result;
  }
  if (!residual_result.success) {
    result.failure_reason = "residual_result.success is false";
    return result;
  }
  if (!scene_state.IsValid()) {
    result.failure_reason = "scene_state is not valid";
    return result;
  }

  std::map<std::pair<int, int>, const JointResidualBoardObservationDiagnostics*> residual_by_key;
  std::vector<double> valid_rmse_values;
  for (const JointResidualBoardObservationDiagnostics& diagnostics :
       residual_result.board_observation_diagnostics) {
    residual_by_key[std::make_pair(diagnostics.frame_index, diagnostics.board_id)] = &diagnostics;
    if (std::isfinite(diagnostics.rmse)) {
      valid_rmse_values.push_back(diagnostics.rmse);
    }
  }

  const double median_rmse = ComputeMedian(valid_rmse_values);
  const double residual_sanity_threshold =
      valid_rmse_values.empty() ? options_.max_board_observation_rmse :
      std::max(5.0, std::min(options_.max_board_observation_rmse,
                             options_.residual_sanity_factor * median_rmse));
  {
    std::ostringstream warning;
    warning << "selection residual_sanity_threshold=" << residual_sanity_threshold
            << " median_rmse=" << median_rmse;
    result.warnings.push_back(warning.str());
  }
  {
    std::ostringstream warning;
    warning << "selection enable_residual_sanity_gate="
            << (options_.enable_residual_sanity_gate ? 1 : 0);
    result.warnings.push_back(warning.str());
  }
  {
    std::ostringstream warning;
    warning << "selection max_pose_fit_outer_rmse="
            << options_.max_pose_fit_outer_rmse;
    result.warnings.push_back(warning.str());
  }
  {
    std::ostringstream warning;
    warning << "selection enable_board_pose_fit_gate="
            << (options_.enable_board_pose_fit_gate ? 1 : 0);
    result.warnings.push_back(warning.str());
  }

  std::map<std::pair<int, int>, CandidateBoardObservation> candidates;
  for (const JointMeasurementFrameResult& frame_result : measurement_result.frames) {
    const JointSceneFrameState* scene_frame =
        FindJointSceneFrameState(scene_state, frame_result.frame_index);
    for (const JointBoardObservation& board_observation : frame_result.board_observations) {
      CandidateBoardObservation candidate;
      candidate.frame_index = frame_result.frame_index;
      candidate.frame_label = frame_result.frame_label;
      candidate.board_id = board_observation.board_id;
      candidate.average_quality = AverageQuality(board_observation);
      candidate.coverage_signature = BuildCoverageSignature(
          board_observation, measurement_result.bootstrap_seed.coarse_camera.resolution,
          options_.coverage_grid_cols, options_.coverage_grid_rows);
      candidate.pose_fit_outer_rmse =
          ComputeOuterPoseFitRmse(board_observation, scene_state.camera);
      const auto residual_it = residual_by_key.find(
          std::make_pair(frame_result.frame_index, board_observation.board_id));
      if (residual_it != residual_by_key.end()) {
        candidate.rmse = residual_it->second->rmse;
        candidate.point_count = residual_it->second->point_count;
        candidate.outer_point_count = residual_it->second->outer_point_count;
        candidate.internal_point_count = residual_it->second->internal_point_count;
      }

      JointBoardObservationSelectionDecision decision;
      decision.frame_index = frame_result.frame_index;
      decision.frame_label = frame_result.frame_label;
      decision.board_id = board_observation.board_id;
      decision.rmse = candidate.rmse;
      decision.pose_fit_outer_rmse = candidate.pose_fit_outer_rmse;
      decision.average_quality = candidate.average_quality;
      decision.point_count = candidate.point_count;
      decision.outer_point_count = candidate.outer_point_count;
      decision.internal_point_count = candidate.internal_point_count;
      decision.coverage_signature = candidate.coverage_signature;

      const bool solver_ready = board_observation.used_in_solver &&
          scene_frame != nullptr && scene_frame->initialized &&
          (board_observation.board_id == scene_state.reference_board_id ||
           (FindJointSceneBoardState(scene_state, board_observation.board_id) != nullptr &&
            FindJointSceneBoardState(scene_state, board_observation.board_id)->initialized));
      if (!solver_ready || candidate.point_count <= 0) {
        decision.reason_code = JointBoardObservationSelectionReasonCode::RejectedNotSolverReady;
        decision.reason_detail = "board observation is not solver-ready for Stage 4 selection";
        result.board_observation_decisions.push_back(decision);
        continue;
      }

      if (options_.enable_board_pose_fit_gate &&
          (!std::isfinite(candidate.pose_fit_outer_rmse) ||
           (options_.max_pose_fit_outer_rmse > 0.0 &&
            candidate.pose_fit_outer_rmse > options_.max_pose_fit_outer_rmse))) {
        decision.reason_code = JointBoardObservationSelectionReasonCode::RejectedOuterPoseFit;
        std::ostringstream detail;
        detail << "pose_fit_outer_rmse=" << candidate.pose_fit_outer_rmse
               << " threshold=" << options_.max_pose_fit_outer_rmse;
        decision.reason_detail = detail.str();
        result.board_observation_decisions.push_back(decision);
        continue;
      }

      if (options_.enable_residual_sanity_gate &&
          (!std::isfinite(candidate.rmse) || candidate.rmse > residual_sanity_threshold)) {
        decision.reason_code = JointBoardObservationSelectionReasonCode::RejectedResidualSanity;
        std::ostringstream detail;
        detail << "rmse=" << candidate.rmse
               << " threshold=" << residual_sanity_threshold;
        decision.reason_detail = detail.str();
        result.board_observation_decisions.push_back(decision);
        continue;
      }

      candidates[std::make_pair(candidate.frame_index, candidate.board_id)] = candidate;
      result.board_observation_decisions.push_back(decision);
    }
  }

  std::map<int, int> accepted_observation_count_per_board;
  std::map<int, std::set<std::string> > accepted_signatures_per_board;
  std::set<std::pair<int, int> > accepted_board_pairs;

  for (const JointMeasurementFrameResult& frame_result : measurement_result.frames) {
    JointFrameSelectionDecision frame_decision;
    frame_decision.frame_index = frame_result.frame_index;
    frame_decision.frame_label = frame_result.frame_label;

    std::vector<CandidateBoardObservation> sane_board_observations;
    for (const JointBoardObservation& board_observation : frame_result.board_observations) {
      const auto candidate_it = candidates.find(
          std::make_pair(frame_result.frame_index, board_observation.board_id));
      if (candidate_it == candidates.end()) {
        continue;
      }
      sane_board_observations.push_back(candidate_it->second);
      AppendUniqueInt(board_observation.board_id, &frame_decision.usable_board_ids);
    }
    std::sort(frame_decision.usable_board_ids.begin(), frame_decision.usable_board_ids.end());
    frame_decision.usable_board_observation_count =
        static_cast<int>(sane_board_observations.size());

    if (sane_board_observations.empty()) {
      frame_decision.accepted = false;
      frame_decision.reason_codes.push_back(
          JointFrameSelectionReasonCode::NoUsableBoardObservations);
      frame_decision.reason_detail = "no residual-sane board observations";
      result.frame_decisions.push_back(frame_decision);
      continue;
    }

    bool needs_min_view_coverage = false;
    for (const CandidateBoardObservation& candidate : sane_board_observations) {
      if (accepted_observation_count_per_board[candidate.board_id] <
          options_.min_initial_views_per_board) {
        needs_min_view_coverage = true;
        break;
      }
    }
    if (needs_min_view_coverage) {
      frame_decision.reason_codes.push_back(
          JointFrameSelectionReasonCode::AcceptedMinViewsPerBoard);
    }

    bool introduces_new_board_pair = false;
    std::vector<int> sane_board_ids = frame_decision.usable_board_ids;
    for (std::size_t first = 0; first < sane_board_ids.size() && !introduces_new_board_pair; ++first) {
      for (std::size_t second = first + 1; second < sane_board_ids.size(); ++second) {
        const std::pair<int, int> board_pair(sane_board_ids[first], sane_board_ids[second]);
        if (accepted_board_pairs.find(board_pair) == accepted_board_pairs.end()) {
          introduces_new_board_pair = true;
          break;
        }
      }
    }
    if (introduces_new_board_pair) {
      frame_decision.reason_codes.push_back(
          JointFrameSelectionReasonCode::AcceptedNewBoardPair);
    }

    bool introduces_new_image_coverage = false;
    for (const CandidateBoardObservation& candidate : sane_board_observations) {
      if (candidate.coverage_signature.empty() ||
          accepted_signatures_per_board[candidate.board_id].find(candidate.coverage_signature) ==
              accepted_signatures_per_board[candidate.board_id].end()) {
        introduces_new_image_coverage = true;
        break;
      }
    }
    if (introduces_new_image_coverage) {
      frame_decision.reason_codes.push_back(
          JointFrameSelectionReasonCode::AcceptedNewImageCoverage);
    }

    frame_decision.accepted = !frame_decision.reason_codes.empty();
    if (frame_decision.accepted) {
      for (const CandidateBoardObservation& candidate : sane_board_observations) {
        result.accepted_board_observation_keys.insert(
            std::make_pair(candidate.frame_index, candidate.board_id));
        result.accepted_frame_indices.insert(candidate.frame_index);
        AppendUniqueInt(candidate.board_id, &frame_decision.accepted_board_ids);
        ++accepted_observation_count_per_board[candidate.board_id];
        accepted_signatures_per_board[candidate.board_id].insert(candidate.coverage_signature);
      }
      for (std::size_t first = 0; first < sane_board_ids.size(); ++first) {
        for (std::size_t second = first + 1; second < sane_board_ids.size(); ++second) {
          accepted_board_pairs.insert(std::make_pair(sane_board_ids[first], sane_board_ids[second]));
        }
      }
      frame_decision.accepted_board_observation_count =
          static_cast<int>(frame_decision.accepted_board_ids.size());
      std::ostringstream detail;
      for (std::size_t i = 0; i < frame_decision.reason_codes.size(); ++i) {
        if (i > 0) {
          detail << ",";
        }
        detail << ToString(frame_decision.reason_codes[i]);
      }
      frame_decision.reason_detail = detail.str();
    } else {
      frame_decision.reason_codes.push_back(
          JointFrameSelectionReasonCode::RejectedRedundantView);
      frame_decision.reason_detail = "residual-sane but redundant coverage";
    }

    result.frame_decisions.push_back(frame_decision);
  }

  for (JointBoardObservationSelectionDecision& decision : result.board_observation_decisions) {
    const std::pair<int, int> key(decision.frame_index, decision.board_id);
    if (result.accepted_board_observation_keys.find(key) !=
        result.accepted_board_observation_keys.end()) {
      decision.accepted = true;
      decision.reason_code = JointBoardObservationSelectionReasonCode::Accepted;
      decision.reason_detail = "accepted by frame-level coverage/redundancy control";
    } else if (decision.reason_code ==
               JointBoardObservationSelectionReasonCode::None) {
      decision.reason_code = JointBoardObservationSelectionReasonCode::RejectedFrameRejected;
      decision.reason_detail = "frame rejected as redundant after board-observation sanity pass";
    }
  }

  for (JointMeasurementFrameResult& frame_result : result.selected_measurement_result.frames) {
    for (JointBoardObservation& board_observation : frame_result.board_observations) {
      const bool board_selected =
          result.accepted_board_observation_keys.find(
              std::make_pair(frame_result.frame_index, board_observation.board_id)) !=
          result.accepted_board_observation_keys.end();
      for (JointPointObservation& point : board_observation.points) {
        point.used_in_solver = point.used_in_solver && board_selected;
      }
    }
  }
  RecomputeMeasurementCounts(&result.selected_measurement_result);

  result.accepted_frame_count = static_cast<int>(result.accepted_frame_indices.size());
  result.accepted_board_observation_count =
      static_cast<int>(result.accepted_board_observation_keys.size());
  result.accepted_outer_point_count =
      result.selected_measurement_result.used_outer_point_count;
  result.accepted_internal_point_count =
      result.selected_measurement_result.used_internal_point_count;

  if (result.accepted_board_observation_count <= 0 ||
      result.selected_measurement_result.used_total_point_count <= 0) {
    result.failure_reason = "selection produced no accepted Stage 4 observations";
    result.selected_measurement_result.success = false;
    return result;
  }

  result.success = true;
  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
