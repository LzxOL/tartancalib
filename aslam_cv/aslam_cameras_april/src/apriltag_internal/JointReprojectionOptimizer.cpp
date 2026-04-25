#include <aslam/cameras/apriltag_internal/JointReprojectionOptimizer.hpp>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

double ElapsedSeconds(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

double ComputeWeightedVectorRmse(const Eigen::VectorXd& residuals) {
  if (residuals.size() <= 0) {
    return 0.0;
  }
  return std::sqrt(residuals.squaredNorm() / std::max(1, static_cast<int>(residuals.rows() / 2)));
}

double ParameterStep(double value, double fallback_step) {
  return std::max(std::abs(value) * 1e-4, fallback_step);
}

double HuberWeight(double residual_norm, double delta) {
  if (!(delta > 0.0) || residual_norm <= 0.0 || residual_norm <= delta) {
    return 1.0;
  }
  return delta / residual_norm;
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

Eigen::Matrix<double, 6, 1> ToVector(const OuterBootstrapCameraIntrinsics& intrinsics) {
  Eigen::Matrix<double, 6, 1> vector;
  vector << intrinsics.xi, intrinsics.alpha, intrinsics.fu, intrinsics.fv,
      intrinsics.cu, intrinsics.cv;
  return vector;
}

OuterBootstrapCameraIntrinsics FromVector(const Eigen::Matrix<double, 6, 1>& vector,
                                          const cv::Size& resolution) {
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.resolution = resolution;
  intrinsics.xi = vector[0];
  intrinsics.alpha = vector[1];
  intrinsics.fu = vector[2];
  intrinsics.fv = vector[3];
  intrinsics.cu = vector[4];
  intrinsics.cv = vector[5];
  return intrinsics;
}

void AppendUniqueWarning(const std::string& warning,
                         std::vector<std::string>* warnings) {
  if (warnings == nullptr || warning.empty()) {
    return;
  }
  if (std::find(warnings->begin(), warnings->end(), warning) == warnings->end()) {
    warnings->push_back(warning);
  }
}

std::vector<int> AcceptedFrameIndices(const JointMeasurementSelectionResult& selection_result) {
  return std::vector<int>(selection_result.accepted_frame_indices.begin(),
                          selection_result.accepted_frame_indices.end());
}

std::vector<int> AcceptedBoardIds(const JointMeasurementSelectionResult& selection_result,
                                  int reference_board_id) {
  std::set<int> board_ids;
  for (const std::pair<int, int>& key : selection_result.accepted_board_observation_keys) {
    board_ids.insert(key.second);
  }
  board_ids.erase(reference_board_id);
  return std::vector<int>(board_ids.begin(), board_ids.end());
}

std::vector<const JointPointObservation*> CollectAcceptedPointsForFrame(
    const JointMeasurementBuildResult& measurement_result,
    int frame_index) {
  std::vector<const JointPointObservation*> points;
  for (const JointMeasurementFrameResult& frame_result : measurement_result.frames) {
    if (frame_result.frame_index != frame_index) {
      continue;
    }
    for (const JointBoardObservation& board_observation : frame_result.board_observations) {
      if (!board_observation.used_in_solver) {
        continue;
      }
      for (const JointPointObservation& point : board_observation.points) {
        if (point.used_in_solver) {
          points.push_back(&point);
        }
      }
    }
  }
  return points;
}

std::vector<const JointPointObservation*> CollectAcceptedPointsForBoardObservation(
    const JointMeasurementBuildResult& measurement_result,
    int frame_index,
    int board_id) {
  std::vector<const JointPointObservation*> points;
  for (const JointMeasurementFrameResult& frame_result : measurement_result.frames) {
    if (frame_result.frame_index != frame_index) {
      continue;
    }
    for (const JointBoardObservation& board_observation : frame_result.board_observations) {
      if (board_observation.board_id != board_id || !board_observation.used_in_solver) {
        continue;
      }
      for (const JointPointObservation& point : board_observation.points) {
        if (point.used_in_solver) {
          points.push_back(&point);
        }
      }
    }
  }
  return points;
}

bool EstimateFramePoseFromSelectedPoints(
    const JointMeasurementBuildResult& measurement_result,
    const JointReprojectionSceneState& scene_state,
    int frame_index,
    Eigen::Isometry3d* pose,
    double* rmse) {
  if (pose == nullptr || rmse == nullptr) {
    throw std::runtime_error("EstimateFramePoseFromSelectedPoints requires valid output pointers.");
  }

  const std::vector<const JointPointObservation*> points =
      CollectAcceptedPointsForFrame(measurement_result, frame_index);
  if (points.size() < 4) {
    return false;
  }

  std::vector<Eigen::Vector3d> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(points.size());
  image_points.reserve(points.size());
  for (const JointPointObservation* point : points) {
    const JointSceneBoardState* board_state =
        point->board_id == scene_state.reference_board_id ? nullptr :
        FindJointSceneBoardState(scene_state, point->board_id);
    if (point->board_id != scene_state.reference_board_id &&
        (board_state == nullptr || !board_state->initialized)) {
      continue;
    }
    const Eigen::Vector4d board_point(point->target_xyz_board.x(),
                                      point->target_xyz_board.y(),
                                      point->target_xyz_board.z(), 1.0);
    const Eigen::Vector4d reference_point =
        point->board_id == scene_state.reference_board_id ?
        board_point : board_state->T_reference_board * board_point;
    object_points.push_back(reference_point.head<3>());
    image_points.push_back(cv::Point2f(static_cast<float>(point->image_xy.x()),
                                       static_cast<float>(point->image_xy.y())));
  }

  return EstimatePoseFromObjectPoints(scene_state.camera, object_points, image_points,
                                      pose, rmse);
}

bool RefineBoardPoseFromSelection(
    const JointMeasurementBuildResult& measurement_result,
    const JointReprojectionCostCore& cost_core,
    const JointReprojectionSceneState& current_state,
    int board_id,
    Eigen::Isometry3d* pose,
    double* rmse,
    JointOptimizationRuntimeBreakdown* runtime_breakdown) {
  if (pose == nullptr || rmse == nullptr) {
    throw std::runtime_error("RefineBoardPoseFromSelection requires valid output pointers.");
  }

  struct LocalBoardResidualEvaluation {
    bool success = false;
    Eigen::VectorXd weighted_residuals;
    double cost = 0.0;
  };
  const auto evaluate_board_residuals =
      [&measurement_result, &cost_core, board_id](
          const JointReprojectionSceneState& scene_state) -> LocalBoardResidualEvaluation {
    LocalBoardResidualEvaluation result;
    if (!scene_state.IsValid()) {
      return result;
    }

    const DoubleSphereCameraModel camera =
        DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(scene_state.camera));
    if (!camera.IsValid()) {
      return result;
    }

    struct ObservationBudget {
      int outer_count = 0;
      int internal_count = 0;
    };
    std::map<std::pair<int, int>, ObservationBudget> budgets;
    int selected_point_count = 0;
    for (const JointPointObservation& point : measurement_result.solver_observations) {
      if (!point.used_in_solver || point.board_id != board_id) {
        continue;
      }
      ObservationBudget& budget =
          budgets[std::make_pair(point.frame_index, point.board_id)];
      if (point.point_type == JointPointType::Outer) {
        ++budget.outer_count;
      } else {
        ++budget.internal_count;
      }
      ++selected_point_count;
    }
    if (selected_point_count <= 0) {
      return result;
    }

    result.weighted_residuals.resize(2 * selected_point_count);
    int row = 0;
    for (const JointPointObservation& point : measurement_result.solver_observations) {
      if (!point.used_in_solver || point.board_id != board_id) {
        continue;
      }

      const JointSceneFrameState* frame_state =
          FindJointSceneFrameState(scene_state, point.frame_index);
      if (frame_state == nullptr || !frame_state->initialized) {
        continue;
      }

      Eigen::Matrix4d T_reference_board = Eigen::Matrix4d::Identity();
      if (point.board_id != scene_state.reference_board_id) {
        const JointSceneBoardState* board_state =
            FindJointSceneBoardState(scene_state, point.board_id);
        if (board_state == nullptr || !board_state->initialized) {
          continue;
        }
        T_reference_board = board_state->T_reference_board;
      }

      const Eigen::Vector4d point_board(point.target_xyz_board.x(),
                                        point.target_xyz_board.y(),
                                        point.target_xyz_board.z(),
                                        1.0);
      const Eigen::Vector4d point_camera_h =
          frame_state->T_camera_reference * (T_reference_board * point_board);
      const Eigen::Vector3d point_camera = point_camera_h.head<3>();

      Eigen::Vector2d predicted_image_xy = Eigen::Vector2d::Zero();
      Eigen::Vector2d residual_xy = Eigen::Vector2d::Zero();
      if (!camera.vsEuclideanToKeypoint(point_camera, &predicted_image_xy)) {
        if (cost_core.options().enable_invalid_projection_penalty) {
          residual_xy = Eigen::Vector2d(
              cost_core.options().invalid_projection_penalty_pixels,
              cost_core.options().invalid_projection_penalty_pixels);
        }
      } else {
        residual_xy = predicted_image_xy - point.image_xy;
      }

      const double residual_norm = residual_xy.norm();
      const ObservationBudget& budget =
          budgets[std::make_pair(point.frame_index, point.board_id)];
      const bool has_outer = budget.outer_count > 0;
      const bool has_internal = budget.internal_count > 0;
      double type_budget = 1.0;
      int type_count = 1;
      if (has_outer && has_internal) {
        type_budget = 0.5;
        type_count = point.point_type == JointPointType::Outer
                         ? budget.outer_count
                         : budget.internal_count;
      } else if (point.point_type == JointPointType::Outer) {
        type_count = budget.outer_count;
      } else {
        type_count = budget.internal_count;
      }

      const double balance_weight =
          type_budget / std::max(1, type_count);
      const double huber_delta =
          point.point_type == JointPointType::Outer
              ? cost_core.options().outer_huber_delta_pixels
              : cost_core.options().internal_huber_delta_pixels;
      const double huber_weight = HuberWeight(residual_norm, huber_delta);
      const double final_weight = balance_weight * huber_weight;
      const double scale = std::sqrt(std::max(0.0, final_weight));
      result.weighted_residuals[row++] = scale * residual_xy.x();
      result.weighted_residuals[row++] = scale * residual_xy.y();
      result.cost += final_weight * residual_xy.squaredNorm();
    }

    if (row <= 0) {
      result.weighted_residuals.resize(0);
      return result;
    }
    if (row != result.weighted_residuals.rows()) {
      result.weighted_residuals.conservativeResize(row);
    }
    result.success = true;
    return result;
  };

  JointReprojectionSceneState working_state = current_state;
  JointSceneBoardState* working_board = FindJointSceneBoardState(&working_state, board_id);
  if (working_board == nullptr || !working_board->initialized) {
    return false;
  }
  working_board->T_reference_board = ToMatrix4d(*pose);
  const auto evaluation_start = std::chrono::steady_clock::now();
  LocalBoardResidualEvaluation evaluation = evaluate_board_residuals(working_state);
  if (runtime_breakdown != nullptr) {
    runtime_breakdown->cost_evaluation_seconds += ElapsedSeconds(evaluation_start);
    ++runtime_breakdown->cost_evaluation_call_count;
  }
  Eigen::VectorXd residuals = evaluation.weighted_residuals;
  if (!evaluation.success || residuals.size() <= 0) {
    return false;
  }

  double lambda = 1e-3;
  double best_cost = evaluation.cost;
  for (int iteration = 0; iteration < 12; ++iteration) {
    Eigen::MatrixXd jacobian(residuals.rows(), 6);
    for (int column = 0; column < 6; ++column) {
      Eigen::Matrix<double, 6, 1> plus_delta = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> minus_delta = Eigen::Matrix<double, 6, 1>::Zero();
      const double step = column < 3 ? 1e-4 : 5e-4;
      plus_delta[column] = step;
      minus_delta[column] = -step;

      JointReprojectionSceneState plus_state = working_state;
      FindJointSceneBoardState(&plus_state, board_id)->T_reference_board =
          ToMatrix4d(ApplyPoseDelta(*pose, plus_delta));
      JointReprojectionSceneState minus_state = working_state;
      FindJointSceneBoardState(&minus_state, board_id)->T_reference_board =
          ToMatrix4d(ApplyPoseDelta(*pose, minus_delta));

      const auto plus_start = std::chrono::steady_clock::now();
      const LocalBoardResidualEvaluation plus_eval = evaluate_board_residuals(plus_state);
      if (runtime_breakdown != nullptr) {
        runtime_breakdown->cost_evaluation_seconds += ElapsedSeconds(plus_start);
        ++runtime_breakdown->cost_evaluation_call_count;
      }
      const auto minus_start = std::chrono::steady_clock::now();
      const LocalBoardResidualEvaluation minus_eval = evaluate_board_residuals(minus_state);
      if (runtime_breakdown != nullptr) {
        runtime_breakdown->cost_evaluation_seconds += ElapsedSeconds(minus_start);
        ++runtime_breakdown->cost_evaluation_call_count;
      }
      if (!plus_eval.success || !minus_eval.success) {
        return false;
      }
      jacobian.col(column) =
          (plus_eval.weighted_residuals - minus_eval.weighted_residuals) /
          (2.0 * step);
    }

    const Eigen::Matrix<double, 6, 6> hessian = jacobian.transpose() * jacobian;
    const Eigen::Matrix<double, 6, 1> gradient = jacobian.transpose() * residuals;
    const Eigen::Matrix<double, 6, 1> delta =
        (hessian + lambda * Eigen::Matrix<double, 6, 6>::Identity()).ldlt().solve(-gradient);
    if (!delta.allFinite()) {
      break;
    }

    const Eigen::Isometry3d candidate_pose = ApplyPoseDelta(*pose, delta);
    JointReprojectionSceneState candidate_state = working_state;
    FindJointSceneBoardState(&candidate_state, board_id)->T_reference_board =
        ToMatrix4d(candidate_pose);
    const auto candidate_start = std::chrono::steady_clock::now();
    const LocalBoardResidualEvaluation candidate_eval =
        evaluate_board_residuals(candidate_state);
    if (runtime_breakdown != nullptr) {
      runtime_breakdown->cost_evaluation_seconds += ElapsedSeconds(candidate_start);
      ++runtime_breakdown->cost_evaluation_call_count;
    }
    if (!candidate_eval.success) {
      lambda *= 4.0;
      continue;
    }

    const Eigen::VectorXd& candidate_residuals = candidate_eval.weighted_residuals;
    const double candidate_cost = candidate_eval.cost;
    if (candidate_cost < best_cost) {
      *pose = candidate_pose;
      working_state = candidate_state;
      residuals = candidate_residuals;
      best_cost = candidate_cost;
      lambda *= 0.5;
      if (delta.norm() < 1e-5) {
        break;
      }
    } else {
      lambda *= 4.0;
    }
  }

  *rmse = ComputeWeightedVectorRmse(residuals);
  return std::isfinite(*rmse);
}

bool EstimateBoardPoseFromSelectedFrames(
    const JointMeasurementBuildResult& measurement_result,
    const JointReprojectionCostCore& cost_core,
    const JointReprojectionSceneState& scene_state,
    int board_id,
    Eigen::Isometry3d* pose,
    double* rmse,
    JointOptimizationRuntimeBreakdown* runtime_breakdown) {
  if (pose == nullptr || rmse == nullptr) {
    throw std::runtime_error("EstimateBoardPoseFromSelectedFrames requires valid output pointers.");
  }

  std::vector<TransformCandidate> candidates;
  for (const JointMeasurementFrameResult& frame_result : measurement_result.frames) {
    const JointSceneFrameState* frame_state =
        FindJointSceneFrameState(scene_state, frame_result.frame_index);
    if (frame_state == nullptr || !frame_state->initialized) {
      continue;
    }

    const std::vector<const JointPointObservation*> points =
        CollectAcceptedPointsForBoardObservation(measurement_result,
                                                frame_result.frame_index,
                                                board_id);
    if (points.size() < 4) {
      continue;
    }

    std::vector<Eigen::Vector3d> object_points;
    std::vector<cv::Point2f> image_points;
    object_points.reserve(points.size());
    image_points.reserve(points.size());
    double quality_sum = 0.0;
    for (const JointPointObservation* point : points) {
      object_points.push_back(point->target_xyz_board);
      image_points.push_back(cv::Point2f(static_cast<float>(point->image_xy.x()),
                                         static_cast<float>(point->image_xy.y())));
      quality_sum += point->quality;
    }

    Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
    double observation_rmse = 0.0;
    if (!EstimatePoseFromObjectPoints(scene_state.camera, object_points, image_points,
                                      &T_camera_board, &observation_rmse)) {
      continue;
    }

    TransformCandidate candidate;
    candidate.transform = ToIsometry3d(frame_state->T_camera_reference).inverse() * T_camera_board;
    candidate.weight =
        (quality_sum / std::max<std::size_t>(1, points.size())) /
        std::max(1e-3, 1.0 + observation_rmse);
    candidates.push_back(candidate);
  }

  if (candidates.empty()) {
    return false;
  }

  *pose = AverageTransforms(candidates);
  if (!RefineBoardPoseFromSelection(measurement_result, cost_core, scene_state,
                                    board_id, pose, rmse, runtime_breakdown)) {
    return false;
  }
  return true;
}

bool OptimizeIntrinsicsIfEnabled(const JointMeasurementBuildResult& measurement_result,
                                 const JointReprojectionCostCore& cost_core,
                                 const JointOptimizationOptions& options,
                                 const JointReprojectionSceneState& anchor_state,
                                 JointReprojectionSceneState* scene_state,
                                 double* step_norm,
                                 JointOptimizationRuntimeBreakdown* runtime_breakdown) {
  if (scene_state == nullptr || step_norm == nullptr) {
    throw std::runtime_error("OptimizeIntrinsicsIfEnabled requires valid output pointers.");
  }
  *step_norm = 0.0;

  struct LocalGlobalResidualEvaluation {
    bool success = false;
    Eigen::VectorXd weighted_residuals;
    double total_cost = 0.0;
  };
  const auto evaluate_global_residuals =
      [&measurement_result, &cost_core](
          const JointReprojectionSceneState& current_state) -> LocalGlobalResidualEvaluation {
    LocalGlobalResidualEvaluation result;
    if (!measurement_result.success || !current_state.IsValid()) {
      return result;
    }

    const DoubleSphereCameraModel camera =
        DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(current_state.camera));
    if (!camera.IsValid()) {
      return result;
    }

    struct ObservationBudget {
      int outer_count = 0;
      int internal_count = 0;
    };
    std::map<std::pair<int, int>, ObservationBudget> budgets;
    int selected_point_count = 0;
    for (const JointPointObservation& point : measurement_result.solver_observations) {
      if (!point.used_in_solver) {
        continue;
      }
      ObservationBudget& budget =
          budgets[std::make_pair(point.frame_index, point.board_id)];
      if (point.point_type == JointPointType::Outer) {
        ++budget.outer_count;
      } else {
        ++budget.internal_count;
      }
      ++selected_point_count;
    }
    if (selected_point_count <= 0) {
      return result;
    }

    result.weighted_residuals.resize(2 * selected_point_count);
    int row = 0;
    for (const JointPointObservation& point : measurement_result.solver_observations) {
      if (!point.used_in_solver) {
        continue;
      }

      const JointSceneFrameState* frame_state =
          FindJointSceneFrameState(current_state, point.frame_index);
      if (frame_state == nullptr || !frame_state->initialized) {
        continue;
      }

      Eigen::Matrix4d T_reference_board = Eigen::Matrix4d::Identity();
      if (point.board_id != current_state.reference_board_id) {
        const JointSceneBoardState* board_state =
            FindJointSceneBoardState(current_state, point.board_id);
        if (board_state == nullptr || !board_state->initialized) {
          continue;
        }
        T_reference_board = board_state->T_reference_board;
      }

      const Eigen::Vector4d point_board(point.target_xyz_board.x(),
                                        point.target_xyz_board.y(),
                                        point.target_xyz_board.z(),
                                        1.0);
      const Eigen::Vector4d point_camera_h =
          frame_state->T_camera_reference * (T_reference_board * point_board);
      const Eigen::Vector3d point_camera = point_camera_h.head<3>();

      Eigen::Vector2d predicted_image_xy = Eigen::Vector2d::Zero();
      Eigen::Vector2d residual_xy = Eigen::Vector2d::Zero();
      if (!camera.vsEuclideanToKeypoint(point_camera, &predicted_image_xy)) {
        if (cost_core.options().enable_invalid_projection_penalty) {
          residual_xy = Eigen::Vector2d(
              cost_core.options().invalid_projection_penalty_pixels,
              cost_core.options().invalid_projection_penalty_pixels);
        }
      } else {
        residual_xy = predicted_image_xy - point.image_xy;
      }

      const double residual_norm = residual_xy.norm();
      const ObservationBudget& budget =
          budgets[std::make_pair(point.frame_index, point.board_id)];
      const bool has_outer = budget.outer_count > 0;
      const bool has_internal = budget.internal_count > 0;
      double type_budget = 1.0;
      int type_count = 1;
      if (has_outer && has_internal) {
        type_budget = 0.5;
        type_count = point.point_type == JointPointType::Outer
                         ? budget.outer_count
                         : budget.internal_count;
      } else if (point.point_type == JointPointType::Outer) {
        type_count = budget.outer_count;
      } else {
        type_count = budget.internal_count;
      }

      const double balance_weight =
          type_budget / std::max(1, type_count);
      const double huber_delta =
          point.point_type == JointPointType::Outer
              ? cost_core.options().outer_huber_delta_pixels
              : cost_core.options().internal_huber_delta_pixels;
      const double huber_weight = HuberWeight(residual_norm, huber_delta);
      const double final_weight = balance_weight * huber_weight;
      const double scale = std::sqrt(std::max(0.0, final_weight));
      result.weighted_residuals[row++] = scale * residual_xy.x();
      result.weighted_residuals[row++] = scale * residual_xy.y();
      result.total_cost += final_weight * residual_xy.squaredNorm();
    }

    if (row <= 0) {
      result.weighted_residuals.resize(0);
      return result;
    }
    if (row != result.weighted_residuals.rows()) {
      result.weighted_residuals.conservativeResize(row);
    }
    result.success = true;
    return result;
  };

  const auto evaluation_start = std::chrono::steady_clock::now();
  const LocalGlobalResidualEvaluation evaluation = evaluate_global_residuals(*scene_state);
  if (runtime_breakdown != nullptr) {
    runtime_breakdown->cost_evaluation_seconds += ElapsedSeconds(evaluation_start);
    ++runtime_breakdown->cost_evaluation_call_count;
  }
  Eigen::VectorXd residuals = evaluation.weighted_residuals;
  if (!evaluation.success || residuals.size() <= 0) {
    return false;
  }

  Eigen::Matrix<double, 6, 1> parameters = ToVector(scene_state->camera);
  const Eigen::Matrix<double, 6, 1> anchor = ToVector(anchor_state.camera);
  Eigen::Matrix<double, 6, 1> prior_weight = Eigen::Matrix<double, 6, 1>::Zero();
  prior_weight[0] = options.intrinsics_anchor_weight_xi_alpha;
  prior_weight[1] = options.intrinsics_anchor_weight_xi_alpha;
  prior_weight[2] = options.intrinsics_anchor_weight_focal;
  prior_weight[3] = options.intrinsics_anchor_weight_focal;
  prior_weight[4] = options.intrinsics_anchor_weight_principal;
  prior_weight[5] = options.intrinsics_anchor_weight_principal;

  double lambda = 1e-3;
  double best_cost = evaluation.total_cost;
  for (int iteration = 0; iteration < 10; ++iteration) {
    Eigen::MatrixXd jacobian(residuals.rows(), 6);
    for (int column = 0; column < 6; ++column) {
      Eigen::Matrix<double, 6, 1> plus = parameters;
      Eigen::Matrix<double, 6, 1> minus = parameters;
      const double step = ParameterStep(parameters[column], column < 2 ? 1e-4 : 1e-2);
      plus[column] += step;
      minus[column] -= step;

      JointReprojectionSceneState plus_state = *scene_state;
      JointReprojectionSceneState minus_state = *scene_state;
      plus_state.camera = FromVector(plus, scene_state->camera.resolution);
      minus_state.camera = FromVector(minus, scene_state->camera.resolution);
      ClampIntrinsicsInPlace(&plus_state.camera);
      ClampIntrinsicsInPlace(&minus_state.camera);

      const auto plus_start = std::chrono::steady_clock::now();
      const LocalGlobalResidualEvaluation plus_eval = evaluate_global_residuals(plus_state);
      if (runtime_breakdown != nullptr) {
        runtime_breakdown->cost_evaluation_seconds += ElapsedSeconds(plus_start);
        ++runtime_breakdown->cost_evaluation_call_count;
      }
      const auto minus_start = std::chrono::steady_clock::now();
      const LocalGlobalResidualEvaluation minus_eval = evaluate_global_residuals(minus_state);
      if (runtime_breakdown != nullptr) {
        runtime_breakdown->cost_evaluation_seconds += ElapsedSeconds(minus_start);
        ++runtime_breakdown->cost_evaluation_call_count;
      }
      if (!plus_eval.success || !minus_eval.success) {
        return false;
      }
      jacobian.col(column) =
          (plus_eval.weighted_residuals -
           minus_eval.weighted_residuals) /
          (2.0 * step);
    }

    const Eigen::Matrix<double, 6, 6> hessian = jacobian.transpose() * jacobian;
    const Eigen::Matrix<double, 6, 1> gradient = jacobian.transpose() * residuals;
    Eigen::Matrix<double, 6, 6> prior_hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> prior_gradient = Eigen::Matrix<double, 6, 1>::Zero();
    for (int index = 0; index < 6; ++index) {
      prior_hessian(index, index) = prior_weight[index];
      prior_gradient[index] = prior_weight[index] * (parameters[index] - anchor[index]);
    }

    const Eigen::Matrix<double, 6, 1> delta =
        (hessian + prior_hessian +
         lambda * Eigen::Matrix<double, 6, 6>::Identity()).ldlt().solve(-(gradient + prior_gradient));
    if (!delta.allFinite()) {
      break;
    }

    JointReprojectionSceneState candidate_state = *scene_state;
    candidate_state.camera = FromVector(parameters + delta, scene_state->camera.resolution);
    ClampIntrinsicsInPlace(&candidate_state.camera);
    const auto candidate_start = std::chrono::steady_clock::now();
    const LocalGlobalResidualEvaluation candidate_eval =
        evaluate_global_residuals(candidate_state);
    if (runtime_breakdown != nullptr) {
      runtime_breakdown->cost_evaluation_seconds += ElapsedSeconds(candidate_start);
      ++runtime_breakdown->cost_evaluation_call_count;
    }
    if (!candidate_eval.success) {
      lambda *= 4.0;
      continue;
    }

    double prior_cost = 0.0;
    const Eigen::Matrix<double, 6, 1> candidate_vector = ToVector(candidate_state.camera);
    for (int index = 0; index < 6; ++index) {
      const double diff = candidate_vector[index] - anchor[index];
      prior_cost += prior_weight[index] * diff * diff;
    }
    const double candidate_cost = candidate_eval.total_cost + prior_cost;
    if (candidate_cost < best_cost) {
      *step_norm = delta.norm();
      *scene_state = candidate_state;
      parameters = candidate_vector;
      residuals = candidate_eval.weighted_residuals;
      best_cost = candidate_cost;
      lambda *= 0.5;
      if (delta.norm() < 1e-4) {
        break;
      }
    } else {
      lambda *= 4.0;
    }
  }

  return *step_norm > 0.0;
}

void UpdateStateDiagnostics(const JointMeasurementBuildResult& measurement_result,
                            const JointResidualEvaluationResult& residual_result,
                            JointReprojectionSceneState* scene_state) {
  if (scene_state == nullptr) {
    return;
  }

  for (JointSceneFrameState& frame_state : scene_state->frames) {
    frame_state.observation_count = 0;
    frame_state.rmse = 0.0;
    for (const JointResidualFrameDiagnostics& diagnostics : residual_result.frame_diagnostics) {
      if (diagnostics.frame_index == frame_state.frame_index) {
        frame_state.observation_count = 0;
        for (const JointResidualBoardObservationDiagnostics& board_observation :
             residual_result.board_observation_diagnostics) {
          if (board_observation.frame_index == frame_state.frame_index) {
            ++frame_state.observation_count;
          }
        }
        frame_state.rmse = diagnostics.rmse;
        break;
      }
    }
  }

  for (JointSceneBoardState& board_state : scene_state->boards) {
    board_state.observation_count = 0;
    board_state.rmse = 0.0;
    for (const JointResidualBoardDiagnostics& diagnostics : residual_result.board_diagnostics) {
      if (diagnostics.board_id == board_state.board_id) {
        board_state.observation_count = diagnostics.observation_count;
        board_state.rmse = diagnostics.rmse;
        break;
      }
    }
  }
  (void)measurement_result;
}

}  // namespace

JointReprojectionOptimizer::JointReprojectionOptimizer(JointOptimizationOptions options)
    : options_(std::move(options)),
      cost_core_(options_.cost_options),
      residual_evaluator_(JointResidualEvaluationOptions{10, options_.cost_options}) {}

JointOptimizationResult JointReprojectionOptimizer::Optimize(
    const JointMeasurementSelectionResult& selection_result,
    const JointReprojectionSceneState& initial_state) const {
  JointOptimizationResult result;
  result.reference_board_id = initial_state.reference_board_id;
  result.optimize_intrinsics = options_.optimize_intrinsics;
  result.intrinsics_release_iteration = options_.intrinsics_release_iteration;
  result.selection_result = selection_result;
  result.initial_state = initial_state;
  result.optimized_state = initial_state;

  if (!selection_result.success) {
    result.failure_reason = "selection_result.success is false";
    return result;
  }
  if (!initial_state.IsValid()) {
    result.failure_reason = "initial_state is not valid";
    return result;
  }

  {
    const auto start = std::chrono::steady_clock::now();
    result.initial_residual =
        residual_evaluator_.Evaluate(selection_result.selected_measurement_result, result.initial_state);
    result.runtime_breakdown.residual_evaluation_seconds += ElapsedSeconds(start);
    ++result.runtime_breakdown.residual_evaluation_call_count;
  }
  if (!result.initial_residual.success) {
    result.failure_reason = "failed to evaluate initial residuals";
    return result;
  }
  UpdateStateDiagnostics(selection_result.selected_measurement_result,
                         result.initial_residual,
                         &result.initial_state);
  result.optimized_state = result.initial_state;

  JointCostEvaluation current_eval;
  {
    const auto start = std::chrono::steady_clock::now();
    current_eval =
        cost_core_.Evaluate(selection_result.selected_measurement_result, result.optimized_state);
    result.runtime_breakdown.cost_evaluation_seconds += ElapsedSeconds(start);
    ++result.runtime_breakdown.cost_evaluation_call_count;
  }
  if (!current_eval.success) {
    result.failure_reason = current_eval.failure_reason;
    result.warnings = current_eval.warnings;
    return result;
  }

  const std::vector<int> frame_indices = AcceptedFrameIndices(selection_result);
  const std::vector<int> board_ids = AcceptedBoardIds(
      selection_result, initial_state.reference_board_id);
  bool intrinsics_release_window_reached = false;
  bool any_intrinsics_attempted = false;
  bool any_intrinsics_accepted = false;

  for (int iteration = 0; iteration < options_.max_joint_iterations; ++iteration) {
    JointOptimizationIterationSummary summary;
    summary.iteration_index = iteration;
    summary.cost_before = current_eval.total_cost;
    summary.overall_rmse_before = current_eval.overall_rmse;

    if (options_.optimize_frame_poses) {
      const auto frame_update_start = std::chrono::steady_clock::now();
      for (int frame_index : frame_indices) {
        JointSceneFrameState* frame_state =
            FindJointSceneFrameState(&result.optimized_state, frame_index);
        if (frame_state == nullptr || !frame_state->initialized) {
          ++summary.rejected_frame_updates;
          continue;
        }

        Eigen::Isometry3d candidate_pose = ToIsometry3d(frame_state->T_camera_reference);
        double pose_rmse = 0.0;
        if (!EstimateFramePoseFromSelectedPoints(selection_result.selected_measurement_result,
                                                 result.optimized_state,
                                                 frame_index,
                                                 &candidate_pose,
                                                 &pose_rmse)) {
          ++summary.rejected_frame_updates;
          continue;
        }

        JointReprojectionSceneState candidate_state = result.optimized_state;
        FindJointSceneFrameState(&candidate_state, frame_index)->T_camera_reference =
            ToMatrix4d(candidate_pose);
        const auto candidate_start = std::chrono::steady_clock::now();
        const JointCostEvaluation candidate_eval =
            cost_core_.Evaluate(selection_result.selected_measurement_result, candidate_state);
        result.runtime_breakdown.cost_evaluation_seconds += ElapsedSeconds(candidate_start);
        ++result.runtime_breakdown.cost_evaluation_call_count;
        if (candidate_eval.success && candidate_eval.total_cost < current_eval.total_cost) {
          summary.max_parameter_delta = std::max(
              summary.max_parameter_delta,
              ComputePoseDeltaNorm(ToIsometry3d(frame_state->T_camera_reference), candidate_pose));
          result.optimized_state = candidate_state;
          current_eval = candidate_eval;
          ++summary.accepted_frame_updates;
        } else {
          ++summary.rejected_frame_updates;
        }
      }
      result.runtime_breakdown.frame_update_seconds += ElapsedSeconds(frame_update_start);
    }

    if (options_.optimize_board_poses) {
      const auto board_update_start = std::chrono::steady_clock::now();
      for (int board_id : board_ids) {
        JointSceneBoardState* board_state =
            FindJointSceneBoardState(&result.optimized_state, board_id);
        if (board_state == nullptr || !board_state->initialized) {
          ++summary.rejected_board_updates;
          continue;
        }

        Eigen::Isometry3d candidate_pose = ToIsometry3d(board_state->T_reference_board);
        double pose_rmse = 0.0;
        if (!EstimateBoardPoseFromSelectedFrames(selection_result.selected_measurement_result,
                                                 cost_core_,
                                                 result.optimized_state,
                                                 board_id,
                                                 &candidate_pose,
                                                 &pose_rmse,
                                                 &result.runtime_breakdown)) {
          ++summary.rejected_board_updates;
          continue;
        }

        JointReprojectionSceneState candidate_state = result.optimized_state;
        FindJointSceneBoardState(&candidate_state, board_id)->T_reference_board =
            ToMatrix4d(candidate_pose);
        const auto candidate_start = std::chrono::steady_clock::now();
        const JointCostEvaluation candidate_eval =
            cost_core_.Evaluate(selection_result.selected_measurement_result, candidate_state);
        result.runtime_breakdown.cost_evaluation_seconds += ElapsedSeconds(candidate_start);
        ++result.runtime_breakdown.cost_evaluation_call_count;
        if (candidate_eval.success && candidate_eval.total_cost < current_eval.total_cost) {
          summary.max_parameter_delta = std::max(
              summary.max_parameter_delta,
              ComputePoseDeltaNorm(ToIsometry3d(board_state->T_reference_board), candidate_pose));
          result.optimized_state = candidate_state;
          current_eval = candidate_eval;
          ++summary.accepted_board_updates;
        } else {
          ++summary.rejected_board_updates;
        }
      }
      result.runtime_breakdown.board_update_seconds += ElapsedSeconds(board_update_start);
    }

    if (options_.optimize_intrinsics &&
        iteration >= options_.intrinsics_release_iteration) {
      const auto intrinsics_start = std::chrono::steady_clock::now();
      intrinsics_release_window_reached = true;
      summary.attempted_intrinsics_update = true;
      any_intrinsics_attempted = true;
      JointReprojectionSceneState candidate_state = result.optimized_state;
      double intrinsics_step_norm = 0.0;
      if (OptimizeIntrinsicsIfEnabled(selection_result.selected_measurement_result,
                                      cost_core_,
                                      options_,
                                      result.initial_state,
                                      &candidate_state,
                                      &intrinsics_step_norm,
                                      &result.runtime_breakdown)) {
        const auto candidate_start = std::chrono::steady_clock::now();
        const JointCostEvaluation candidate_eval =
            cost_core_.Evaluate(selection_result.selected_measurement_result, candidate_state);
        result.runtime_breakdown.cost_evaluation_seconds += ElapsedSeconds(candidate_start);
        ++result.runtime_breakdown.cost_evaluation_call_count;
        if (candidate_eval.success && candidate_eval.total_cost < current_eval.total_cost) {
          result.optimized_state = candidate_state;
          current_eval = candidate_eval;
          summary.accepted_intrinsics_update = true;
          any_intrinsics_accepted = true;
          summary.intrinsics_step_norm = intrinsics_step_norm;
          summary.max_parameter_delta = std::max(summary.max_parameter_delta, intrinsics_step_norm);
        } else {
          summary.rejected_intrinsics_update = true;
        }
      } else {
        summary.rejected_intrinsics_update = true;
      }
      result.runtime_breakdown.intrinsics_update_seconds += ElapsedSeconds(intrinsics_start);
    }

    summary.cost_after = current_eval.total_cost;
    summary.cost_delta = summary.cost_before - summary.cost_after;
    summary.overall_rmse_after = current_eval.overall_rmse;
    summary.outer_rmse_after = current_eval.outer_rmse;
    summary.internal_rmse_after = current_eval.internal_rmse;
    result.iterations.push_back(summary);

    if (summary.cost_delta < options_.convergence_threshold) {
      break;
    }
  }

  {
    const auto start = std::chrono::steady_clock::now();
    result.optimized_residual =
        residual_evaluator_.Evaluate(selection_result.selected_measurement_result, result.optimized_state);
    result.runtime_breakdown.residual_evaluation_seconds += ElapsedSeconds(start);
    ++result.runtime_breakdown.residual_evaluation_call_count;
  }
  if (!result.optimized_residual.success) {
    result.failure_reason = "failed to evaluate optimized residuals";
    return result;
  }
  UpdateStateDiagnostics(selection_result.selected_measurement_result,
                         result.optimized_residual,
                         &result.optimized_state);

  result.success = result.optimized_residual.overall_rmse <=
      result.initial_residual.overall_rmse + 1e-9;
  if (!result.success) {
    result.failure_reason = "optimized residuals did not improve";
  }

  if (options_.optimize_intrinsics) {
    if (!intrinsics_release_window_reached) {
      AppendUniqueWarning("intrinsics release window was never reached during optimization",
                          &result.warnings);
    } else if (!any_intrinsics_attempted) {
      AppendUniqueWarning("intrinsics release was enabled but no intrinsics update was attempted",
                          &result.warnings);
    } else if (!any_intrinsics_accepted) {
      AppendUniqueWarning("intrinsics release was enabled but all intrinsics updates were rejected",
                          &result.warnings);
    }
  }
  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
