#include <aslam/cameras/apriltag_internal/Stage6IncrementalBatchEstimator.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

Eigen::Matrix3d Skew(const Eigen::Vector3d& value) {
  Eigen::Matrix3d skew;
  skew << 0.0, -value.z(), value.y(),
      value.z(), 0.0, -value.x(),
      -value.y(), value.x(), 0.0;
  return skew;
}

bool IsSelectedPairBoard(const std::set<std::pair<int, int> >& keys,
                         int pair_index,
                         int board_id) {
  return keys.count(std::make_pair(pair_index, board_id)) > 0;
}

bool TransformPoint(const Eigen::Matrix4d& transform,
                    const Eigen::Vector3d& point,
                    Eigen::Vector3d* transformed) {
  if (transformed == nullptr) {
    return false;
  }
  const Eigen::Vector4d homogeneous(point.x(), point.y(), point.z(), 1.0);
  const Eigen::Vector4d result = transform * homogeneous;
  if (!result.allFinite()) {
    return false;
  }
  *transformed = result.head<3>();
  return transformed->allFinite();
}

Eigen::Matrix<double, 6, 6> ComputeStereoExtrinsicInformationMatrix(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene,
    const std::set<std::pair<int, int> >& selected_pair_boards) {
  Eigen::Matrix<double, 6, 6> hessian =
      Eigen::Matrix<double, 6, 6>::Zero();
  if (selected_pair_boards.empty()) {
    return hessian;
  }

  const Eigen::Matrix4d T_cam1_cam0 = scene.T_cam1_cam0;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.camera_index != 1 ||
        !IsSelectedPairBoard(selected_pair_boards, observation.pair_index,
                             observation.board_id)) {
      continue;
    }
    const auto pair_pose_it =
        scene.T_cam0_world_by_pair.find(observation.pair_index);
    const auto board_pose_it =
        scene.T_world_board_by_id.find(observation.board_id);
    if (pair_pose_it == scene.T_cam0_world_by_pair.end() ||
        board_pose_it == scene.T_world_board_by_id.end()) {
      continue;
    }

    Eigen::Vector3d point_world;
    Eigen::Vector3d point_cam0;
    Eigen::Vector3d point_cam1;
    if (!TransformPoint(board_pose_it->second, observation.target_point_board,
                        &point_world) ||
        !TransformPoint(pair_pose_it->second, point_world, &point_cam0) ||
        !TransformPoint(T_cam1_cam0, point_cam0, &point_cam1)) {
      continue;
    }
    const double z = point_cam1.z();
    if (!std::isfinite(z) || std::abs(z) < 1e-9) {
      continue;
    }

    Eigen::Matrix<double, 2, 3> J_project;
    J_project << 1.0 / z, 0.0, -point_cam1.x() / (z * z),
        0.0, 1.0 / z, -point_cam1.y() / (z * z);

    Eigen::Matrix<double, 3, 6> J_extrinsic;
    J_extrinsic.block<3, 3>(0, 0).setIdentity();
    J_extrinsic.block<3, 3>(0, 3) = -Skew(point_cam1);

    const Eigen::Matrix<double, 2, 6> J = J_project * J_extrinsic;
    const double weight = std::isfinite(observation.weight)
                              ? std::max(0.0, observation.weight)
                              : 1.0;
    hessian += weight * J.transpose() * J;
  }
  return hessian;
}

double DampedLog2EigenvalueSum(const Eigen::Matrix<double, 6, 6>& hessian,
                               double damping,
                               double rank_threshold,
                               int* rank) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6> > solver(
      hessian);
  if (solver.info() != Eigen::Success) {
    if (rank != nullptr) {
      *rank = 0;
    }
    return 0.0;
  }
  const Eigen::Matrix<double, 6, 1> values = solver.eigenvalues();
  double log_sum = 0.0;
  int effective_rank = 0;
  const double safe_damping = std::max(damping, 1e-12);
  const double safe_rank_threshold = std::max(rank_threshold, 0.0);
  for (int i = 0; i < values.rows(); ++i) {
    const double value = std::max(0.0, values(i));
    if (value > safe_rank_threshold) {
      ++effective_rank;
    }
    log_sum += std::log(value + safe_damping) / std::log(2.0);
  }
  if (rank != nullptr) {
    *rank = effective_rank;
  }
  return log_sum;
}

bool IsFiniteResidual(const StereoResidualSummary& residual) {
  return residual.success && residual.point_count > 0 &&
         std::isfinite(residual.total_stereo_rmse) &&
         std::isfinite(residual.cam0_rmse) &&
         std::isfinite(residual.cam1_rmse);
}

double RotationDistanceDeg(const Eigen::Matrix4d& lhs,
                           const Eigen::Matrix4d& rhs) {
  constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
  const Eigen::Matrix3d rotation_delta =
      lhs.block<3, 3>(0, 0).transpose() * rhs.block<3, 3>(0, 0);
  const Eigen::AngleAxisd angle_axis(rotation_delta);
  return std::abs(angle_axis.angle()) * kRadiansToDegrees;
}

double TranslationDistance(const Eigen::Matrix4d& lhs,
                           const Eigen::Matrix4d& rhs) {
  return (lhs.block<3, 1>(0, 3) - rhs.block<3, 1>(0, 3)).norm();
}

}  // namespace

const char* ToString(Stage6IncrementalInfoBlock block) {
  switch (block) {
    case Stage6IncrementalInfoBlock::StereoExtrinsic:
      return "stereo_extrinsic";
  }
  return "unknown";
}

Stage6IncrementalBatchEstimator::Stage6IncrementalBatchEstimator(
    Stage6IncrementalBatchEstimatorOptions options)
    : options_(options) {}

double Stage6IncrementalBatchEstimator::ComputeMarginalInformationGainProxy(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene,
    const std::set<std::pair<int, int> >& selected_pair_boards,
    int* rank) const {
  const Eigen::Matrix<double, 6, 6> hessian =
      ComputeStereoExtrinsicInformationMatrix(dataset, scene,
                                              selected_pair_boards);
  return DampedLog2EigenvalueSum(hessian, options_.rank_threshold,
                                 options_.rank_threshold, rank);
}

Stage6IncrementalBatchResult Stage6IncrementalBatchEstimator::AddBatch(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_before,
    const StereoSceneState& scene_after,
    const StereoResidualSummary& residual_before,
    const StereoResidualSummary& residual_after,
    const StereoGlobalSparseBaSummary& optimization_summary,
    const Stage6IncrementalBatchCandidate& candidate) const {
  Stage6IncrementalBatchResult result;
  result.force = candidate.force;
  result.info_gain_threshold = options_.info_gain_threshold;
  result.rank_threshold = options_.rank_threshold;
  result.info_block = ToString(options_.info_block);
  result.num_iterations = optimization_summary.iterations;
  result.objective_before = optimization_summary.objective_start;
  result.objective_after = optimization_summary.objective_final;
  result.rmse_before = residual_before.total_stereo_rmse;
  result.rmse_after = residual_after.total_stereo_rmse;
  result.total_rmse_delta =
      residual_after.total_stereo_rmse - residual_before.total_stereo_rmse;
  result.cam0_rmse_delta = residual_after.cam0_rmse - residual_before.cam0_rmse;
  result.cam1_rmse_delta = residual_after.cam1_rmse - residual_before.cam1_rmse;
  result.baseline_rotation_delta_deg =
      RotationDistanceDeg(scene_before.T_cam1_cam0, scene_after.T_cam1_cam0);
  result.baseline_translation_delta_m =
      TranslationDistance(scene_before.T_cam1_cam0, scene_after.T_cam1_cam0);

  const double before_log_sum = ComputeMarginalInformationGainProxy(
      dataset, scene_before, candidate.selected_pair_boards_before,
      &result.rank_before);
  const double after_log_sum = ComputeMarginalInformationGainProxy(
      dataset, scene_after, candidate.selected_pair_boards_after,
      &result.rank_after);
  result.marginal_information_gain_proxy =
      0.5 * (after_log_sum - before_log_sum);
  result.rank_proxy_increases = result.rank_after > result.rank_before;

  result.residual_finite = IsFiniteResidual(residual_after) &&
                           std::isfinite(result.total_rmse_delta) &&
                           std::isfinite(result.cam0_rmse_delta) &&
                           std::isfinite(result.cam1_rmse_delta);
  result.objective_finite = std::isfinite(result.objective_before) &&
                            std::isfinite(result.objective_after);
  result.objective_decreased =
      result.objective_finite &&
      result.objective_after < result.objective_before;
  result.no_solver_failure = !optimization_summary.linear_solver_failure;
  const bool iteration_valid =
      optimization_summary.max_iterations <= 0 ||
      optimization_summary.iterations < optimization_summary.max_iterations;
  result.optimization_success =
      optimization_summary.success ||
      (result.objective_finite && result.objective_decreased &&
       result.no_solver_failure && iteration_valid);
  result.baseline_stable =
      std::isfinite(result.baseline_rotation_delta_deg) &&
      std::isfinite(result.baseline_translation_delta_m) &&
      result.baseline_rotation_delta_deg <=
          options_.max_baseline_rotation_delta_deg &&
      result.baseline_translation_delta_m <=
          options_.max_baseline_translation_delta_m;
  result.solution_valid = result.optimization_success && result.residual_finite &&
                          result.objective_finite &&
                          result.objective_decreased &&
                          result.no_solver_failure && iteration_valid &&
                          result.baseline_stable;

  if (candidate.force) {
    result.batchAccepted = result.optimization_success;
    result.accept_reason = result.batchAccepted ? "force" : "";
    result.reject_reason =
        result.batchAccepted ? "" : "forced_batch_optimization_failed";
  } else if (!result.solution_valid) {
    result.batchAccepted = false;
    result.reject_reason = "hard_validity_gate";
  } else if (result.marginal_information_gain_proxy >
             options_.info_gain_threshold) {
    result.batchAccepted = true;
    result.accept_reason = "marginal_information_gain";
  } else if (result.rank_proxy_increases) {
    result.batchAccepted = true;
    result.accept_reason = "rank_proxy_increase";
  } else {
    result.batchAccepted = false;
    result.reject_reason = "marginal_information_gain_gate";
  }
  result.committed_or_rollback =
      result.batchAccepted ? "committed" : "rollback";
  return result;
}

Stage6IncrementalBatchResult
Stage6IncrementalBatchEstimator::EvaluateInformationGainOnly(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene,
    const Stage6IncrementalBatchCandidate& candidate) const {
  Stage6IncrementalBatchResult result;
  result.force = candidate.force;
  result.info_gain_threshold = options_.info_gain_threshold;
  result.rank_threshold = options_.rank_threshold;
  result.info_block = ToString(options_.info_block);
  result.optimization_success = true;
  result.residual_finite = true;
  result.objective_finite = true;
  result.objective_decreased = true;
  result.no_solver_failure = true;
  result.baseline_stable = true;
  result.solution_valid = true;
  result.committed_or_rollback = "selection_only";

  const double before_log_sum = ComputeMarginalInformationGainProxy(
      dataset, scene, candidate.selected_pair_boards_before,
      &result.rank_before);
  const double after_log_sum = ComputeMarginalInformationGainProxy(
      dataset, scene, candidate.selected_pair_boards_after,
      &result.rank_after);
  result.marginal_information_gain_proxy =
      0.5 * (after_log_sum - before_log_sum);
  result.rank_proxy_increases = result.rank_after > result.rank_before;

  if (candidate.force) {
    result.batchAccepted = true;
    result.accept_reason = "force_information_gain_only";
  } else if (result.marginal_information_gain_proxy >
             options_.info_gain_threshold) {
    result.batchAccepted = true;
    result.accept_reason = "marginal_information_gain";
  } else if (result.rank_proxy_increases) {
    result.batchAccepted = true;
    result.accept_reason = "rank_proxy_increase";
  } else {
    result.batchAccepted = false;
    result.reject_reason = "marginal_information_gain_gate";
  }
  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
