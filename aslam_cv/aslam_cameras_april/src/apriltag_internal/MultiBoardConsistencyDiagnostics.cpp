#include <aslam/cameras/apriltag_internal/MultiBoardConsistencyDiagnostics.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <utility>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>
#include <aslam/cameras/apriltag_internal/PolarAngleResidualDiagnostics.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

struct LocalPoseEstimate {
  bool success = false;
  std::string source = "failed";
  Eigen::Isometry3d T_cam_board = Eigen::Isometry3d::Identity();
  double reprojection_rmse = 0.0;
};

const JointSceneFrameState* FindSceneFrameState(
    const JointReprojectionSceneState& scene_state,
    int frame_index) {
  for (const JointSceneFrameState& frame_state : scene_state.frames) {
    if (frame_state.frame_index == frame_index) {
      return &frame_state;
    }
  }
  return nullptr;
}

const JointSceneBoardState* FindSceneBoardState(
    const JointReprojectionSceneState& scene_state,
    int board_id) {
  for (const JointSceneBoardState& board_state : scene_state.boards) {
    if (board_state.board_id == board_id) {
      return &board_state;
    }
  }
  return nullptr;
}

double ComputeRotationAngleDeg(const Eigen::Matrix3d& rotation) {
  const Eigen::AngleAxisd angle_axis(rotation);
  return std::abs(angle_axis.angle()) * 180.0 / M_PI;
}

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

double ComputeMean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

std::string MakePairKey(int board_i, int board_j) {
  std::ostringstream stream;
  stream << board_i << "-" << board_j;
  return stream.str();
}

bool IsOuterPoint(const JointPointObservation& observation) {
  return observation.point_type == JointPointType::Outer;
}

bool IsInternalPoint(const JointPointObservation& observation) {
  return observation.point_type == JointPointType::Internal;
}

bool IncludePoint(const JointPointObservation& observation,
                  const MultiBoardRigidityDiagnosticsOptions& options) {
  if (IsOuterPoint(observation)) {
    return options.use_outer_points;
  }
  if (IsInternalPoint(observation)) {
    return options.use_internal_points;
  }
  return false;
}

double ComputeBoardReprojectionRmse(
    const DoubleSphereCameraModel& camera_model,
    const std::vector<const JointPointObservation*>& points,
    const Eigen::Isometry3d& T_cam_board,
    double* mean_residual_x,
    double* mean_residual_y,
    double* rmse_x,
    double* rmse_y) {
  if (mean_residual_x != nullptr) {
    *mean_residual_x = 0.0;
  }
  if (mean_residual_y != nullptr) {
    *mean_residual_y = 0.0;
  }
  if (rmse_x != nullptr) {
    *rmse_x = 0.0;
  }
  if (rmse_y != nullptr) {
    *rmse_y = 0.0;
  }
  if (points.empty()) {
    return 0.0;
  }

  double sum_sq = 0.0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  std::vector<double> residual_xs;
  std::vector<double> residual_ys;
  residual_xs.reserve(points.size());
  residual_ys.reserve(points.size());
  int valid_count = 0;
  for (const JointPointObservation* point : points) {
    if (point == nullptr) {
      continue;
    }
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    if (!camera_model.vsEuclideanToKeypoint(
            T_cam_board * point->target_xyz_board, &predicted)) {
      continue;
    }
    const Eigen::Vector2d residual = predicted - point->image_xy;
    sum_sq += residual.squaredNorm();
    sum_x += residual.x();
    sum_y += residual.y();
    residual_xs.push_back(residual.x());
    residual_ys.push_back(residual.y());
    ++valid_count;
  }

  if (valid_count <= 0) {
    return 0.0;
  }

  if (mean_residual_x != nullptr) {
    *mean_residual_x = sum_x / static_cast<double>(valid_count);
  }
  if (mean_residual_y != nullptr) {
    *mean_residual_y = sum_y / static_cast<double>(valid_count);
  }

  double sum_sq_x = 0.0;
  double sum_sq_y = 0.0;
  for (int index = 0; index < valid_count; ++index) {
    const double dx = residual_xs[static_cast<std::size_t>(index)];
    const double dy = residual_ys[static_cast<std::size_t>(index)];
    sum_sq_x += dx * dx;
    sum_sq_y += dy * dy;
  }
  if (rmse_x != nullptr) {
    *rmse_x = std::sqrt(sum_sq_x / static_cast<double>(valid_count));
  }
  if (rmse_y != nullptr) {
    *rmse_y = std::sqrt(sum_sq_y / static_cast<double>(valid_count));
  }
  return std::sqrt(sum_sq / static_cast<double>(valid_count));
}

LocalPoseEstimate EstimateLocalPose(
    const OuterBootstrapCameraIntrinsics& camera,
    const std::vector<const JointPointObservation*>& points,
    const MultiBoardRigidityDiagnosticsOptions& options) {
  LocalPoseEstimate result;
  std::vector<Eigen::Vector3d> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(points.size());
  image_points.reserve(points.size());
  for (const JointPointObservation* point : points) {
    if (point == nullptr || !IsOuterPoint(*point)) {
      continue;
    }
    object_points.push_back(point->target_xyz_board);
    image_points.emplace_back(
        static_cast<float>(point->image_xy.x()),
        static_cast<float>(point->image_xy.y()));
  }

  if (object_points.size() < 4) {
    return result;
  }

  if (EstimatePoseFromObjectPoints(camera, object_points, image_points,
                                   &result.T_cam_board, &result.reprojection_rmse)) {
    result.success = true;
    result.source = "outer_only";
  }
  (void)options;
  return result;
}

struct IndexedBoardObservation {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  std::vector<const JointPointObservation*> points;
};

std::vector<IndexedBoardObservation> IndexBoardObservations(
    const CalibrationMeasurementDataset& observations) {
  std::map<std::pair<int, int>, IndexedBoardObservation> grouped;
  for (const JointPointObservation& observation : observations.solver_observations) {
    std::pair<int, int> key(observation.frame_index, observation.board_id);
    IndexedBoardObservation& entry = grouped[key];
    entry.frame_index = observation.frame_index;
    entry.frame_label = observation.frame_label;
    entry.board_id = observation.board_id;
    entry.points.push_back(&observation);
  }

  std::vector<IndexedBoardObservation> indexed;
  indexed.reserve(grouped.size());
  for (const auto& kv : grouped) {
    indexed.push_back(kv.second);
  }
  return indexed;
}

}  // namespace

MultiBoardRigidityDiagnostics::MultiBoardRigidityDiagnostics(
    MultiBoardRigidityDiagnosticsOptions options)
    : options_(std::move(options)) {}

MultiBoardRigidityDiagnosticsResult MultiBoardRigidityDiagnostics::Evaluate(
    const CalibrationMeasurementDataset& observations,
    const JointReprojectionSceneState& scene_state) const {
  MultiBoardRigidityDiagnosticsResult result;
  result.training_only = false;
  if (!options_.enabled) {
    result.failure_reason = "disabled";
    return result;
  }

  const DoubleSphereCameraModel camera_model =
      DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(scene_state.camera));
  if (!camera_model.IsValid()) {
    result.failure_reason = "invalid_camera_model";
    return result;
  }

  const std::vector<IndexedBoardObservation> indexed =
      IndexBoardObservations(observations);
  std::map<std::pair<int, int>, LocalPoseEstimate> local_pose_map;
  std::map<std::pair<int, int>, FrameBoardConsistencyRecord> frame_board_map;

  for (const IndexedBoardObservation& indexed_obs : indexed) {
    FrameBoardConsistencyRecord row;
    row.frame_index = indexed_obs.frame_index;
    row.frame_label = indexed_obs.frame_label;
    row.board_id = indexed_obs.board_id;
    row.is_reference_board = (indexed_obs.board_id == scene_state.reference_board_id);

    const JointSceneFrameState* frame_state =
        FindSceneFrameState(scene_state, indexed_obs.frame_index);
    if (frame_state == nullptr || !frame_state->initialized) {
      row.diagnostic_status = "frame_not_initialized";
      result.frame_board_records.push_back(row);
      continue;
    }

    Eigen::Matrix4d T_reference_board_matrix = Eigen::Matrix4d::Identity();
    if (!row.is_reference_board) {
      const JointSceneBoardState* board_state =
          FindSceneBoardState(scene_state, indexed_obs.board_id);
      if (board_state == nullptr || !board_state->initialized) {
        row.diagnostic_status = "board_not_initialized";
        result.frame_board_records.push_back(row);
        continue;
      }
      T_reference_board_matrix = board_state->T_reference_board;
    }
    const Eigen::Isometry3d T_cam_reference(frame_state->T_camera_reference);
    const Eigen::Isometry3d T_reference_board(T_reference_board_matrix);
    const Eigen::Isometry3d T_cam_board_global =
        T_cam_reference * T_reference_board;

    std::vector<const JointPointObservation*> used_points;
    used_points.reserve(indexed_obs.points.size());
    double polar_sum = 0.0;
    int polar_count = 0;
    for (const JointPointObservation* point : indexed_obs.points) {
      if (point == nullptr || !IncludePoint(*point, options_)) {
        continue;
      }
      used_points.push_back(point);
      if (IsOuterPoint(*point)) {
        ++row.point_count_outer;
      } else if (IsInternalPoint(*point)) {
        ++row.point_count_internal;
      }
      const double polar_deg = ComputePolarAngleDeg(camera_model, point->image_xy);
      if (std::isfinite(polar_deg)) {
        polar_sum += polar_deg;
        ++polar_count;
        row.polar_angle_max = std::max(row.polar_angle_max, polar_deg);
      }
    }
    row.polar_angle_mean =
        polar_count > 0 ? polar_sum / static_cast<double>(polar_count) : 0.0;

    if (row.point_count_outer < 4) {
      row.diagnostic_status = "insufficient_points";
      result.frame_board_records.push_back(row);
      continue;
    }

    double global_mean_x = 0.0;
    double global_mean_y = 0.0;
    row.global_reprojection_rmse = ComputeBoardReprojectionRmse(
        camera_model, used_points, T_cam_board_global,
        &global_mean_x, &global_mean_y, &row.rmse_x_global, &row.rmse_y_global);
    row.mean_residual_x_global = global_mean_x;
    row.mean_residual_y_global = global_mean_y;

    const LocalPoseEstimate local_pose =
        EstimateLocalPose(scene_state.camera, used_points, options_);
    local_pose_map[std::make_pair(indexed_obs.frame_index, indexed_obs.board_id)] =
        local_pose;
    if (!local_pose.success) {
      row.local_pose_success = false;
      row.local_pose_source = "failed";
      row.diagnostic_status = "local_pose_failed";
      result.frame_board_records.push_back(row);
      continue;
    }

    row.local_pose_success = true;
    row.local_pose_source = local_pose.source;
    row.local_reprojection_rmse = local_pose.reprojection_rmse;
    const Eigen::Isometry3d delta =
        T_cam_board_global.inverse() * local_pose.T_cam_board;
    row.rotation_error_deg = ComputeRotationAngleDeg(delta.rotation());
    row.translation_error_norm = delta.translation().norm();
    row.reprojection_rmse_delta =
        row.global_reprojection_rmse - row.local_reprojection_rmse;
    row.diagnostic_status = "ok";

    result.frame_board_records.push_back(row);
    frame_board_map[std::make_pair(indexed_obs.frame_index, indexed_obs.board_id)] = row;
  }

  result.total_frame_board_observations =
      static_cast<int>(result.frame_board_records.size());

  std::map<int, std::vector<FrameBoardConsistencyRecord>> by_board;
  std::map<int, std::vector<FrameBoardConsistencyRecord>> by_frame;
  std::vector<double> rotation_errors;
  std::vector<double> translation_errors;
  std::vector<double> global_rmses;
  std::vector<double> local_rmses;
  std::vector<double> reprojection_deltas;
  for (const FrameBoardConsistencyRecord& row : result.frame_board_records) {
    by_board[row.board_id].push_back(row);
    by_frame[row.frame_index].push_back(row);
    if (!row.local_pose_success) {
      ++result.local_pose_failure_count;
      continue;
    }
    ++result.local_pose_success_count;
    rotation_errors.push_back(row.rotation_error_deg);
    translation_errors.push_back(row.translation_error_norm);
    global_rmses.push_back(row.global_reprojection_rmse);
    local_rmses.push_back(row.local_reprojection_rmse);
    reprojection_deltas.push_back(row.reprojection_rmse_delta);
    if (row.rotation_error_deg >= options_.rotation_bad_threshold_deg) {
      ++result.bad_by_rotation_threshold_count;
    }
    if (row.reprojection_rmse_delta >= options_.reprojection_delta_bad_threshold_px) {
      ++result.bad_by_reprojection_delta_threshold_count;
    }
  }
  result.mean_rotation_error_deg = ComputeMean(rotation_errors);
  result.median_rotation_error_deg = ComputeMedian(rotation_errors);
  result.max_rotation_error_deg =
      rotation_errors.empty()
          ? 0.0
          : *std::max_element(rotation_errors.begin(), rotation_errors.end());
  result.mean_translation_error_norm = ComputeMean(translation_errors);
  result.median_translation_error_norm = ComputeMedian(translation_errors);
  result.max_translation_error_norm =
      translation_errors.empty()
          ? 0.0
          : *std::max_element(translation_errors.begin(), translation_errors.end());
  result.mean_global_reprojection_rmse = ComputeMean(global_rmses);
  result.mean_local_reprojection_rmse = ComputeMean(local_rmses);
  result.mean_reprojection_rmse_delta = ComputeMean(reprojection_deltas);

  for (const auto& kv : by_board) {
    BoardConsistencyAggregate agg;
    agg.board_id = kv.first;
    agg.observation_count = static_cast<int>(kv.second.size());
    std::vector<double> rotation_values;
    std::vector<double> translation_values;
    std::vector<double> global_rmse_values;
    std::vector<double> local_rmse_values;
    std::vector<double> delta_values;
    for (const FrameBoardConsistencyRecord& row : kv.second) {
      if (!row.local_pose_success) {
        continue;
      }
      ++agg.local_success_count;
      rotation_values.push_back(row.rotation_error_deg);
      translation_values.push_back(row.translation_error_norm);
      global_rmse_values.push_back(row.global_reprojection_rmse);
      local_rmse_values.push_back(row.local_reprojection_rmse);
      delta_values.push_back(row.reprojection_rmse_delta);
      agg.max_rotation_error_deg =
          std::max(agg.max_rotation_error_deg, row.rotation_error_deg);
      agg.max_translation_error_norm =
          std::max(agg.max_translation_error_norm, row.translation_error_norm);
    }
    agg.mean_rotation_error_deg = ComputeMean(rotation_values);
    agg.mean_translation_error_norm = ComputeMean(translation_values);
    agg.mean_global_rmse = ComputeMean(global_rmse_values);
    agg.mean_local_rmse = ComputeMean(local_rmse_values);
    agg.mean_rmse_delta = ComputeMean(delta_values);
    result.board_aggregates.push_back(agg);
  }

  std::map<std::string, std::vector<BoardPairwiseConsistencyRecord>> by_pair;
  for (const auto& frame_entry : by_frame) {
    std::vector<FrameBoardConsistencyRecord> successful_rows;
    for (const FrameBoardConsistencyRecord& row : frame_entry.second) {
      if (row.local_pose_success) {
        successful_rows.push_back(row);
      }
    }
    std::sort(successful_rows.begin(), successful_rows.end(),
              [](const FrameBoardConsistencyRecord& lhs,
                 const FrameBoardConsistencyRecord& rhs) {
                return lhs.board_id < rhs.board_id;
              });
    for (std::size_t i = 0; i < successful_rows.size(); ++i) {
      for (std::size_t j = i + 1; j < successful_rows.size(); ++j) {
        const FrameBoardConsistencyRecord& row_i = successful_rows[i];
        const FrameBoardConsistencyRecord& row_j = successful_rows[j];
        const auto key_i = std::make_pair(row_i.frame_index, row_i.board_id);
        const auto key_j = std::make_pair(row_j.frame_index, row_j.board_id);
        const auto local_i_it = local_pose_map.find(key_i);
        const auto local_j_it = local_pose_map.find(key_j);
        if (local_i_it == local_pose_map.end() || local_j_it == local_pose_map.end() ||
            !local_i_it->second.success || !local_j_it->second.success) {
          continue;
        }

        Eigen::Matrix4d T_ref_i_matrix = Eigen::Matrix4d::Identity();
        if (row_i.board_id != scene_state.reference_board_id) {
          const JointSceneBoardState* board_i_state =
              FindSceneBoardState(scene_state, row_i.board_id);
          if (board_i_state == nullptr || !board_i_state->initialized) {
            continue;
          }
          T_ref_i_matrix = board_i_state->T_reference_board;
        }
        Eigen::Matrix4d T_ref_j_matrix = Eigen::Matrix4d::Identity();
        if (row_j.board_id != scene_state.reference_board_id) {
          const JointSceneBoardState* board_j_state =
              FindSceneBoardState(scene_state, row_j.board_id);
          if (board_j_state == nullptr || !board_j_state->initialized) {
            continue;
          }
          T_ref_j_matrix = board_j_state->T_reference_board;
        }

        const Eigen::Isometry3d T_board_i_board_j_local =
            local_i_it->second.T_cam_board.inverse() * local_j_it->second.T_cam_board;
        const Eigen::Isometry3d T_board_i_board_j_global =
            Eigen::Isometry3d(T_ref_i_matrix).inverse() * Eigen::Isometry3d(T_ref_j_matrix);
        const Eigen::Isometry3d delta =
            T_board_i_board_j_global.inverse() * T_board_i_board_j_local;

        BoardPairwiseConsistencyRecord pair_row;
        pair_row.frame_index = row_i.frame_index;
        pair_row.frame_label = row_i.frame_label;
        pair_row.board_i = row_i.board_id;
        pair_row.board_j = row_j.board_id;
        pair_row.pair_key = MakePairKey(row_i.board_id, row_j.board_id);
        pair_row.local_pose_success_i = true;
        pair_row.local_pose_success_j = true;
        pair_row.pair_rotation_error_deg = ComputeRotationAngleDeg(delta.rotation());
        pair_row.pair_translation_error_norm = delta.translation().norm();
        pair_row.board_i_point_count =
            row_i.point_count_outer + row_i.point_count_internal;
        pair_row.board_j_point_count =
            row_j.point_count_outer + row_j.point_count_internal;
        pair_row.board_i_global_rmse = row_i.global_reprojection_rmse;
        pair_row.board_j_global_rmse = row_j.global_reprojection_rmse;
        pair_row.board_i_local_rmse = row_i.local_reprojection_rmse;
        pair_row.board_j_local_rmse = row_j.local_reprojection_rmse;
        pair_row.pair_diagnostic_status = "ok";
        result.board_pairwise_records.push_back(pair_row);
        by_pair[pair_row.pair_key].push_back(pair_row);
      }
    }
  }

  result.total_pair_observations =
      static_cast<int>(result.board_pairwise_records.size());
  result.unique_pair_count = static_cast<int>(by_pair.size());
  std::vector<double> pair_rotations;
  std::vector<double> pair_translations;
  for (const auto& kv : by_pair) {
    BoardPairwiseConsistencyAggregate agg;
    agg.pair_key = kv.first;
    if (!kv.second.empty()) {
      agg.board_i = kv.second.front().board_i;
      agg.board_j = kv.second.front().board_j;
    }
    agg.observation_count = static_cast<int>(kv.second.size());
    std::vector<double> rotations;
    std::vector<double> translations;
    std::vector<double> board_i_global_rmses;
    std::vector<double> board_j_global_rmses;
    for (const BoardPairwiseConsistencyRecord& row : kv.second) {
      rotations.push_back(row.pair_rotation_error_deg);
      translations.push_back(row.pair_translation_error_norm);
      board_i_global_rmses.push_back(row.board_i_global_rmse);
      board_j_global_rmses.push_back(row.board_j_global_rmse);
      if (row.pair_rotation_error_deg >= options_.rotation_bad_threshold_deg) {
        ++agg.bad_count_by_rotation_threshold;
      }
      if (options_.translation_bad_threshold >= 0.0 &&
          row.pair_translation_error_norm >= options_.translation_bad_threshold) {
        ++agg.bad_count_by_translation_threshold;
      }
    }
    agg.mean_rotation_error_deg = ComputeMean(rotations);
    agg.median_rotation_error_deg = ComputeMedian(rotations);
    agg.max_rotation_error_deg =
        rotations.empty() ? 0.0
                          : *std::max_element(rotations.begin(), rotations.end());
    agg.mean_translation_error_norm = ComputeMean(translations);
    agg.median_translation_error_norm = ComputeMedian(translations);
    agg.max_translation_error_norm =
        translations.empty() ? 0.0
                             : *std::max_element(translations.begin(), translations.end());
    agg.mean_board_i_global_rmse = ComputeMean(board_i_global_rmses);
    agg.mean_board_j_global_rmse = ComputeMean(board_j_global_rmses);
    result.board_pairwise_aggregates.push_back(agg);
    pair_rotations.insert(pair_rotations.end(), rotations.begin(), rotations.end());
    pair_translations.insert(pair_translations.end(), translations.begin(), translations.end());
  }
  result.mean_pair_rotation_error_deg = ComputeMean(pair_rotations);
  result.max_pair_rotation_error_deg =
      pair_rotations.empty()
          ? 0.0
          : *std::max_element(pair_rotations.begin(), pair_rotations.end());
  result.mean_pair_translation_error_norm = ComputeMean(pair_translations);
  result.max_pair_translation_error_norm =
      pair_translations.empty()
          ? 0.0
          : *std::max_element(pair_translations.begin(), pair_translations.end());

  result.success = true;
  return result;
}

void WriteFrameBoardConsistencyCsv(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result) {
  std::ofstream output(path.c_str());
  output << "frame_index,frame_label,board_id,point_count_outer,"
         << "point_count_internal,local_pose_success,local_pose_source,"
         << "rotation_error_deg,translation_error_norm,local_reprojection_rmse,"
         << "global_reprojection_rmse,reprojection_rmse_delta,"
         << "mean_residual_x_global,mean_residual_y_global,rmse_x_global,"
         << "rmse_y_global,polar_angle_mean,polar_angle_max,is_reference_board,"
         << "diagnostic_status\n";
  for (const FrameBoardConsistencyRecord& row : result.frame_board_records) {
    output << row.frame_index << ","
           << row.frame_label << ","
           << row.board_id << ","
           << row.point_count_outer << ","
           << row.point_count_internal << ","
           << (row.local_pose_success ? 1 : 0) << ","
           << row.local_pose_source << ","
           << row.rotation_error_deg << ","
           << row.translation_error_norm << ","
           << row.local_reprojection_rmse << ","
           << row.global_reprojection_rmse << ","
           << row.reprojection_rmse_delta << ","
           << row.mean_residual_x_global << ","
           << row.mean_residual_y_global << ","
           << row.rmse_x_global << ","
           << row.rmse_y_global << ","
           << row.polar_angle_mean << ","
           << row.polar_angle_max << ","
           << (row.is_reference_board ? 1 : 0) << ","
           << row.diagnostic_status << "\n";
  }
}

void WriteFrameBoardConsistencySummary(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "training_only_diagnostics: " << (result.training_only ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "total_frame_board_observations: "
         << result.total_frame_board_observations << "\n";
  output << "local_pose_success_count: " << result.local_pose_success_count << "\n";
  output << "local_pose_failure_count: " << result.local_pose_failure_count << "\n";
  output << "mean_rotation_error_deg: " << result.mean_rotation_error_deg << "\n";
  output << "median_rotation_error_deg: " << result.median_rotation_error_deg << "\n";
  output << "max_rotation_error_deg: " << result.max_rotation_error_deg << "\n";
  output << "mean_translation_error_norm: " << result.mean_translation_error_norm << "\n";
  output << "median_translation_error_norm: " << result.median_translation_error_norm << "\n";
  output << "max_translation_error_norm: " << result.max_translation_error_norm << "\n";
  output << "mean_global_reprojection_rmse: "
         << result.mean_global_reprojection_rmse << "\n";
  output << "mean_local_reprojection_rmse: "
         << result.mean_local_reprojection_rmse << "\n";
  output << "mean_reprojection_rmse_delta: "
         << result.mean_reprojection_rmse_delta << "\n";
  output << "bad_by_rotation_threshold_count: "
         << result.bad_by_rotation_threshold_count << "\n";
  output << "bad_by_reprojection_delta_threshold_count: "
         << result.bad_by_reprojection_delta_threshold_count << "\n";
  output << "\n[by_board]\n";
  for (const BoardConsistencyAggregate& row : result.board_aggregates) {
    output << "board_id=" << row.board_id
           << " observation_count=" << row.observation_count
           << " local_success_count=" << row.local_success_count
           << " mean_rotation_error_deg=" << row.mean_rotation_error_deg
           << " max_rotation_error_deg=" << row.max_rotation_error_deg
           << " mean_translation_error_norm=" << row.mean_translation_error_norm
           << " max_translation_error_norm=" << row.max_translation_error_norm
           << " mean_global_rmse=" << row.mean_global_rmse
           << " mean_local_rmse=" << row.mean_local_rmse
           << " mean_rmse_delta=" << row.mean_rmse_delta << "\n";
  }
}

void WriteTopBadFrameBoardObservations(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result,
    int top_k) {
  std::ofstream output(path.c_str());
  std::vector<FrameBoardConsistencyRecord> successful;
  for (const FrameBoardConsistencyRecord& row : result.frame_board_records) {
    if (row.local_pose_success) {
      successful.push_back(row);
    }
  }
  const auto write_top = [&output, top_k](
                             std::vector<FrameBoardConsistencyRecord> rows,
                             const std::string& title,
                             const std::function<bool(const FrameBoardConsistencyRecord&,
                                                      const FrameBoardConsistencyRecord&)>& cmp) {
    std::sort(rows.begin(), rows.end(), cmp);
    output << "[" << title << "]\n";
    for (int index = 0;
         index < std::min<int>(top_k, static_cast<int>(rows.size()));
         ++index) {
      const FrameBoardConsistencyRecord& row = rows[static_cast<std::size_t>(index)];
      output << "rank=" << (index + 1)
             << " frame_index=" << row.frame_index
             << " frame_label=" << row.frame_label
             << " board_id=" << row.board_id
             << " rotation_error_deg=" << row.rotation_error_deg
             << " translation_error_norm=" << row.translation_error_norm
             << " reprojection_rmse_delta=" << row.reprojection_rmse_delta
             << " global_reprojection_rmse=" << row.global_reprojection_rmse
             << " local_reprojection_rmse=" << row.local_reprojection_rmse
             << " polar_angle_mean=" << row.polar_angle_mean
             << " polar_angle_max=" << row.polar_angle_max
             << " point_count_outer=" << row.point_count_outer
             << " point_count_internal=" << row.point_count_internal
             << "\n";
    }
    output << "\n";
  };
  write_top(successful, "top_by_rotation_error_deg",
            [](const FrameBoardConsistencyRecord& lhs,
               const FrameBoardConsistencyRecord& rhs) {
              return lhs.rotation_error_deg > rhs.rotation_error_deg;
            });
  write_top(successful, "top_by_translation_error_norm",
            [](const FrameBoardConsistencyRecord& lhs,
               const FrameBoardConsistencyRecord& rhs) {
              return lhs.translation_error_norm > rhs.translation_error_norm;
            });
  write_top(successful, "top_by_reprojection_rmse_delta",
            [](const FrameBoardConsistencyRecord& lhs,
               const FrameBoardConsistencyRecord& rhs) {
              return lhs.reprojection_rmse_delta > rhs.reprojection_rmse_delta;
            });
  write_top(successful, "top_by_global_reprojection_rmse",
            [](const FrameBoardConsistencyRecord& lhs,
               const FrameBoardConsistencyRecord& rhs) {
              return lhs.global_reprojection_rmse > rhs.global_reprojection_rmse;
            });
}

void WriteBoardPairwiseConsistencyCsv(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result) {
  std::ofstream output(path.c_str());
  output << "frame_index,frame_label,board_i,board_j,pair_key,"
         << "local_pose_success_i,local_pose_success_j,"
         << "pair_rotation_error_deg,pair_translation_error_norm,"
         << "board_i_point_count,board_j_point_count,"
         << "board_i_global_rmse,board_j_global_rmse,"
         << "board_i_local_rmse,board_j_local_rmse,pair_diagnostic_status\n";
  for (const BoardPairwiseConsistencyRecord& row : result.board_pairwise_records) {
    output << row.frame_index << ","
           << row.frame_label << ","
           << row.board_i << ","
           << row.board_j << ","
           << row.pair_key << ","
           << (row.local_pose_success_i ? 1 : 0) << ","
           << (row.local_pose_success_j ? 1 : 0) << ","
           << row.pair_rotation_error_deg << ","
           << row.pair_translation_error_norm << ","
           << row.board_i_point_count << ","
           << row.board_j_point_count << ","
           << row.board_i_global_rmse << ","
           << row.board_j_global_rmse << ","
           << row.board_i_local_rmse << ","
           << row.board_j_local_rmse << ","
           << row.pair_diagnostic_status << "\n";
  }
}

void WriteBoardPairwiseConsistencySummary(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "total_pair_observations: " << result.total_pair_observations << "\n";
  output << "unique_pair_count: " << result.unique_pair_count << "\n";
  output << "mean_pair_rotation_error_deg: "
         << result.mean_pair_rotation_error_deg << "\n";
  output << "max_pair_rotation_error_deg: "
         << result.max_pair_rotation_error_deg << "\n";
  output << "mean_pair_translation_error_norm: "
         << result.mean_pair_translation_error_norm << "\n";
  output << "max_pair_translation_error_norm: "
         << result.max_pair_translation_error_norm << "\n";
  output << "\n[by_pair]\n";
  for (const BoardPairwiseConsistencyAggregate& row :
       result.board_pairwise_aggregates) {
    output << "pair_key=" << row.pair_key
           << " board_i=" << row.board_i
           << " board_j=" << row.board_j
           << " observation_count=" << row.observation_count
           << " mean_rotation_error_deg=" << row.mean_rotation_error_deg
           << " median_rotation_error_deg=" << row.median_rotation_error_deg
           << " max_rotation_error_deg=" << row.max_rotation_error_deg
           << " mean_translation_error_norm=" << row.mean_translation_error_norm
           << " median_translation_error_norm=" << row.median_translation_error_norm
           << " max_translation_error_norm=" << row.max_translation_error_norm
           << " mean_board_i_global_rmse=" << row.mean_board_i_global_rmse
           << " mean_board_j_global_rmse=" << row.mean_board_j_global_rmse
           << " bad_count_by_rotation_threshold=" << row.bad_count_by_rotation_threshold
           << " bad_count_by_translation_threshold="
           << row.bad_count_by_translation_threshold
           << "\n";
  }
}

void WriteTopBadBoardPairs(
    const std::string& path,
    const MultiBoardRigidityDiagnosticsResult& result,
    int top_k) {
  std::ofstream output(path.c_str());
  std::vector<BoardPairwiseConsistencyRecord> rows =
      result.board_pairwise_records;
  std::sort(rows.begin(), rows.end(),
            [](const BoardPairwiseConsistencyRecord& lhs,
               const BoardPairwiseConsistencyRecord& rhs) {
              if (lhs.pair_rotation_error_deg != rhs.pair_rotation_error_deg) {
                return lhs.pair_rotation_error_deg > rhs.pair_rotation_error_deg;
              }
              return lhs.pair_translation_error_norm > rhs.pair_translation_error_norm;
            });
  for (int index = 0; index < std::min<int>(top_k, static_cast<int>(rows.size()));
       ++index) {
    const BoardPairwiseConsistencyRecord& row = rows[static_cast<std::size_t>(index)];
    output << "rank=" << (index + 1)
           << " frame_index=" << row.frame_index
           << " frame_label=" << row.frame_label
           << " pair_key=" << row.pair_key
           << " board_i=" << row.board_i
           << " board_j=" << row.board_j
           << " pair_rotation_error_deg=" << row.pair_rotation_error_deg
           << " pair_translation_error_norm=" << row.pair_translation_error_norm
           << " board_i_global_rmse=" << row.board_i_global_rmse
           << " board_j_global_rmse=" << row.board_j_global_rmse
           << " board_i_local_rmse=" << row.board_i_local_rmse
           << " board_j_local_rmse=" << row.board_j_local_rmse
           << "\n";
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
