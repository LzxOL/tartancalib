#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>
#include <aslam/cameras/apriltag_internal/AngularResidualGeometry.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/PolarAngleResidualDiagnostics.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <aslam/backend/CameraDesignVariable.hpp>
#include <aslam/backend/DesignVariable.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/HomogeneousExpression.hpp>
#include <aslam/backend/JacobianContainer.hpp>
#include <aslam/backend/MEstimatorPolicies.hpp>
#include <aslam/backend/MapTransformation.hpp>
#include <aslam/backend/MappedEuclideanPoint.hpp>
#include <aslam/backend/MappedRotationQuaternion.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/Optimizer.hpp>
#include <aslam/backend/TransformationExpression.hpp>
#include <aslam/cameras.hpp>
#include <sm/kinematics/Transformation.hpp>
#include <sm/kinematics/quaternion_algebra.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

const char* ToString(AslamBackendCalibrationOptions::PolarAngleWeightMode mode) {
  switch (mode) {
    case AslamBackendCalibrationOptions::PolarAngleWeightMode::None:
      return "none";
    case AslamBackendCalibrationOptions::PolarAngleWeightMode::DiagnosticOnly:
      return "diagnostic_only";
    case AslamBackendCalibrationOptions::PolarAngleWeightMode::FixedBins:
      return "fixed_bins";
    case AslamBackendCalibrationOptions::PolarAngleWeightMode::AdaptiveSigma:
      return "adaptive_sigma";
  }
  return "none";
}

const char* ToString(AslamBackendCalibrationOptions::ConsistencyWeightMode mode) {
  switch (mode) {
    case AslamBackendCalibrationOptions::ConsistencyWeightMode::Cauchy:
      return "cauchy";
  }
  return "cauchy";
}

const char* ToString(
    AslamBackendCalibrationOptions::BoardPoseParameterization mode) {
  switch (mode) {
    case AslamBackendCalibrationOptions::BoardPoseParameterization::
        ReferenceChain:
      return "reference_chain";
    case AslamBackendCalibrationOptions::BoardPoseParameterization::
        IndependentFrameBoardPose:
      return "independent_frame_board_pose";
  }
  return "reference_chain";
}

const char* ToString(AslamBackendCalibrationOptions::AngularObservedRayMode mode) {
  switch (mode) {
    case AslamBackendCalibrationOptions::AngularObservedRayMode::DynamicCurrentCamera:
      return "dynamic_current_camera";
    case AslamBackendCalibrationOptions::AngularObservedRayMode::FrozenAnchorCamera:
      return "frozen_anchor_camera";
  }
  return "dynamic_current_camera";
}

AslamBackendCalibrationOptions::PolarAngleWeightMode ParsePolarAngleWeightMode(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "none") {
    return AslamBackendCalibrationOptions::PolarAngleWeightMode::None;
  }
  if (lowered == "diagnostic_only" || lowered == "diagnosticonly") {
    return AslamBackendCalibrationOptions::PolarAngleWeightMode::DiagnosticOnly;
  }
  if (lowered == "fixed_bins" || lowered == "fixedbins") {
    return AslamBackendCalibrationOptions::PolarAngleWeightMode::FixedBins;
  }
  if (lowered == "adaptive_sigma" || lowered == "adaptivesigma") {
    return AslamBackendCalibrationOptions::PolarAngleWeightMode::AdaptiveSigma;
  }
  throw std::runtime_error("Unknown polar angle weight mode: " + value);
}

AslamBackendCalibrationOptions::BoardPoseParameterization
ParseBoardPoseParameterization(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "reference_chain" || lowered == "reference-chain" ||
      lowered == "chain" || lowered == "layout") {
    return AslamBackendCalibrationOptions::BoardPoseParameterization::
        ReferenceChain;
  }
  if (lowered == "independent_frame_board_pose" ||
      lowered == "independent-frame-board-pose" ||
      lowered == "independent_board_pose" ||
      lowered == "independent-board-pose" ||
      lowered == "local_board_pose" || lowered == "local-board-pose") {
    return AslamBackendCalibrationOptions::BoardPoseParameterization::
        IndependentFrameBoardPose;
  }
  throw std::runtime_error("Unknown board pose parameterization: " + value);
}

AslamBackendCalibrationOptions::ConsistencyWeightMode ParseConsistencyWeightMode(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "cauchy") {
    return AslamBackendCalibrationOptions::ConsistencyWeightMode::Cauchy;
  }
  throw std::runtime_error("Unknown consistency weight mode: " + value);
}

AslamBackendCalibrationOptions::AngularObservedRayMode ParseAngularObservedRayMode(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "dynamic_current_camera" || lowered == "dynamiccurrentcamera" ||
      lowered == "dynamic") {
    return AslamBackendCalibrationOptions::AngularObservedRayMode::
        DynamicCurrentCamera;
  }
  if (lowered == "frozen_anchor_camera" || lowered == "frozenanchorcamera" ||
      lowered == "frozen") {
    return AslamBackendCalibrationOptions::AngularObservedRayMode::
        FrozenAnchorCamera;
  }
  throw std::runtime_error("Unknown angular observed ray mode: " + value);
}

namespace {

using DsGeometry = aslam::cameras::DoubleSphereCameraGeometry;
using DsProjection = aslam::cameras::DoubleSphereProjection<aslam::cameras::NoDistortion>;
using EucmGeometry = aslam::cameras::ExtendedUnifiedCameraGeometry;
using EucmProjection = aslam::cameras::ExtendedUnifiedProjection<aslam::cameras::NoDistortion>;
using PinholeEquiGeometry = aslam::cameras::EquidistantDistortedPinholeCameraGeometry;
using PinholeEquiProjection =
    aslam::cameras::PinholeProjection<aslam::cameras::EquidistantDistortion>;
using MeiGeometry = aslam::cameras::DistortedOmniCameraGeometry;
using MeiProjection =
    aslam::cameras::OmniProjection<aslam::cameras::RadialTangentialDistortion>;
using OmniNoneGeometry = aslam::cameras::OmniCameraGeometry;
using OmniNoneProjection =
    aslam::cameras::OmniProjection<aslam::cameras::NoDistortion>;

enum class CameraParameterBlock {
  kProjection,
  kDistortion,
};

// Optimizer error terms may be evaluated concurrently. Keep numerical camera
// perturbations local so one residual cannot alter another residual's camera.
template <int ResidualDimension, typename GeometryT, typename AdapterT,
          typename ResidualEvaluator>
void AddThreadSafeCameraFiniteDifferenceJacobian(
    const boost::shared_ptr<AdapterT>& design_variable_adapter,
    const GeometryT& base_camera,
    CameraParameterBlock parameter_block,
    const ResidualEvaluator& evaluate_residual,
    aslam::backend::JacobianContainer* jacobians) {
  if (jacobians == nullptr || design_variable_adapter == nullptr ||
      !design_variable_adapter->isActive()) {
    return;
  }
  const Eigen::MatrixXd base_parameters =
      design_variable_adapter->getParameters();
  const int dimension = static_cast<int>(base_parameters.size());
  if (dimension <= 0) {
    return;
  }

  Eigen::Matrix<double, ResidualDimension, Eigen::Dynamic> camera_jacobian(
      ResidualDimension, dimension);
  camera_jacobian.setZero();
  for (int index = 0; index < dimension; ++index) {
    const Eigen::Index row =
        static_cast<Eigen::Index>(index % base_parameters.rows());
    const Eigen::Index col =
        static_cast<Eigen::Index>(index / base_parameters.rows());
    const double base_value = base_parameters(row, col);
    const double epsilon =
        std::max(1e-7, 1e-6 * std::max(1.0, std::fabs(base_value)));

    Eigen::MatrixXd positive_parameters = base_parameters;
    positive_parameters(row, col) = base_value + epsilon;
    GeometryT positive_camera(base_camera);
    if (parameter_block == CameraParameterBlock::kProjection) {
      positive_camera.projection().setParameters(positive_parameters);
    } else {
      positive_camera.projection().distortion().setParameters(
          positive_parameters);
    }
    bool positive_valid = false;
    const Eigen::Matrix<double, ResidualDimension, 1> positive_residual =
        evaluate_residual(positive_camera, &positive_valid);

    Eigen::MatrixXd negative_parameters = base_parameters;
    negative_parameters(row, col) = base_value - epsilon;
    GeometryT negative_camera(base_camera);
    if (parameter_block == CameraParameterBlock::kProjection) {
      negative_camera.projection().setParameters(negative_parameters);
    } else {
      negative_camera.projection().distortion().setParameters(
          negative_parameters);
    }
    bool negative_valid = false;
    const Eigen::Matrix<double, ResidualDimension, 1> negative_residual =
        evaluate_residual(negative_camera, &negative_valid);

    if (positive_valid && negative_valid && positive_residual.allFinite() &&
        negative_residual.allFinite()) {
      camera_jacobian.col(index) =
          (positive_residual - negative_residual) / (2.0 * epsilon);
    }
  }
  jacobians->add(design_variable_adapter.get(), camera_jacobian);
}

struct ConsistencyWeightSummary {
  bool success = false;
  std::string pose_source = "outer_only";
  int observation_count = 0;
  int successful_observation_count = 0;
  int downweighted_observation_count = 0;
  int hard_rejected_observation_count = 0;
  double mean_consistency_weight = 1.0;
  double min_consistency_weight = 1.0;
  double max_translation_error_mm = 0.0;
  double max_rotation_error_deg = 0.0;
  std::vector<ConsistencyObservationWeightSummaryEntry> observations;
  std::vector<std::string> warnings;
  std::string failure_reason;
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
  const std::string family = intrinsics->NormalizedFamilyString();
  if (family == "ds-none") {
    intrinsics->xi = std::max(-0.95, std::min(2.5, intrinsics->xi));
    intrinsics->alpha = std::max(0.05, std::min(0.95, intrinsics->alpha));
    intrinsics->beta = 1.0;
    intrinsics->distortion_coeffs.clear();
  } else if (family == "eucm-none") {
    intrinsics->xi = 0.0;
    intrinsics->alpha = std::max(0.05, std::min(0.95, intrinsics->alpha));
    intrinsics->beta = std::max(0.25, std::min(3.0, intrinsics->beta));
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
  } else if (family == "omni-radtan") {
    intrinsics->xi = std::max(-0.95, std::min(3.0, intrinsics->xi));
    intrinsics->alpha = 0.0;
    intrinsics->beta = 0.0;
    if (intrinsics->distortion_coeffs.size() != 4) {
      intrinsics->distortion_coeffs.resize(4, 0.0);
    }
    for (double& coefficient : intrinsics->distortion_coeffs) {
      coefficient = std::max(-1.5, std::min(1.5, coefficient));
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

double ComputeRotationAngleDeg(const Eigen::Matrix3d& rotation) {
  Eigen::AngleAxisd angle_axis(rotation);
  return std::fabs(angle_axis.angle()) * 180.0 / M_PI;
}

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

double ComputeBoardObservationResidualRmse(
    const JointMeasurementBuildResult& measurement_result,
    const JointReprojectionSceneState& scene_state,
    int frame_index,
    int board_id) {
  const DoubleSphereCameraModel camera_model =
      DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(scene_state.camera));
  if (!camera_model.IsValid()) {
    return 0.0;
  }
  const JointSceneFrameState* frame_state =
      FindSceneFrameState(scene_state, frame_index);
  if (frame_state == nullptr || !frame_state->initialized) {
    return 0.0;
  }
  Eigen::Matrix4d T_reference_board = Eigen::Matrix4d::Identity();
  if (board_id != scene_state.reference_board_id) {
    const JointSceneBoardState* board_state =
        FindSceneBoardState(scene_state, board_id);
    if (board_state == nullptr || !board_state->initialized) {
      return 0.0;
    }
    T_reference_board = board_state->T_reference_board;
  }

  double squared_sum = 0.0;
  int count = 0;
  for (const JointPointObservation& observation : measurement_result.solver_observations) {
    if (!observation.used_in_solver ||
        observation.frame_index != frame_index ||
        observation.board_id != board_id) {
      continue;
    }
    const Eigen::Vector4d point_board(observation.target_xyz_board.x(),
                                      observation.target_xyz_board.y(),
                                      observation.target_xyz_board.z(),
                                      1.0);
    const Eigen::Vector4d point_camera_h =
        frame_state->T_camera_reference * (T_reference_board * point_board);
    const Eigen::Vector3d point_camera = point_camera_h.head<3>();
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    if (!camera_model.vsEuclideanToKeypoint(point_camera, &predicted)) {
      continue;
    }
    squared_sum += (predicted - observation.image_xy).squaredNorm();
    ++count;
  }
  if (count <= 0) {
    return 0.0;
  }
  return std::sqrt(squared_sum / static_cast<double>(count));
}

ConsistencyWeightSummary ComputeConsistencyWeightSummary(
    const CalibrationBackendProblemInput& problem_input,
    const JointMeasurementBuildResult& measurement_result,
    const JointReprojectionSceneState& scene_state,
    const AslamBackendCalibrationOptions& options) {
  ConsistencyWeightSummary summary;
  summary.pose_source = options.consistency_pose_source;
  if (!options.multi_board_consistency_weighting) {
    summary.failure_reason = "disabled";
    return summary;
  }
  if (options.consistency_pose_source != "outer_only") {
    summary.failure_reason = "Only outer_only consistency pose source is supported.";
    return summary;
  }

  const DoubleSphereCameraModel camera_model =
      DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(scene_state.camera));
  if (!camera_model.IsValid()) {
    summary.failure_reason = "Failed to construct camera for consistency weighting.";
    return summary;
  }

  struct OuterObservationBuffer {
    std::string frame_label;
    std::vector<Eigen::Vector3d> outer_targets;
    std::vector<cv::Point2f> outer_pixels;
    int num_outer_points = 0;
    int num_internal_points = 0;
    double polar_angle_sum = 0.0;
    int polar_angle_count = 0;
    double polar_angle_max_deg = 0.0;
  };

  std::map<std::pair<int, int>, OuterObservationBuffer> buffers;
  for (const JointPointObservation& observation : measurement_result.solver_observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    OuterObservationBuffer& buffer =
        buffers[std::make_pair(observation.frame_index, observation.board_id)];
    buffer.frame_label = observation.frame_label;
    if (observation.point_type == JointPointType::Outer) {
      buffer.outer_targets.push_back(observation.target_xyz_board);
      buffer.outer_pixels.emplace_back(
          static_cast<float>(observation.image_xy.x()),
          static_cast<float>(observation.image_xy.y()));
      ++buffer.num_outer_points;
    } else {
      ++buffer.num_internal_points;
    }
    const double polar_angle_deg = ComputePolarAngleDeg(camera_model, observation.image_xy);
    if (std::isfinite(polar_angle_deg)) {
      buffer.polar_angle_sum += polar_angle_deg;
      ++buffer.polar_angle_count;
      buffer.polar_angle_max_deg = std::max(buffer.polar_angle_max_deg, polar_angle_deg);
    }
  }

  std::map<std::pair<int, int>, Eigen::Isometry3d> local_pose_by_key;
  for (const auto& entry : buffers) {
    const OuterObservationBuffer& buffer = entry.second;
    if (static_cast<int>(buffer.outer_targets.size()) < 4) {
      continue;
    }
    Eigen::Isometry3d local_pose = Eigen::Isometry3d::Identity();
    double local_outer_rmse = 0.0;
    if (EstimatePoseFromObjectPoints(scene_state.camera,
                                     buffer.outer_targets,
                                     buffer.outer_pixels,
                                     &local_pose,
                                     &local_outer_rmse)) {
      local_pose_by_key[entry.first] = local_pose;
    }
  }

  double weight_sum = 0.0;
  for (const auto& entry : buffers) {
    const int frame_index = entry.first.first;
    const int board_id = entry.first.second;
    const OuterObservationBuffer& buffer = entry.second;

    ConsistencyObservationWeightSummaryEntry row;
    row.frame_index = frame_index;
    row.frame_label = buffer.frame_label;
    row.board_id = board_id;
    row.num_outer_points = buffer.num_outer_points;
    row.num_internal_points = buffer.num_internal_points;
    row.polar_angle_deg = buffer.polar_angle_count > 0
                              ? buffer.polar_angle_sum /
                                    static_cast<double>(buffer.polar_angle_count)
                              : 0.0;

    ++summary.observation_count;

    if (static_cast<int>(buffer.outer_targets.size()) < 4) {
      row.failure_reason = "insufficient_outer_points";
      summary.observations.push_back(row);
      continue;
    }

    const JointSceneFrameState* frame_state =
        FindSceneFrameState(scene_state, frame_index);
    const JointSceneBoardState* board_state =
        FindSceneBoardState(scene_state, board_id);
    if (frame_state == nullptr || !frame_state->initialized) {
      row.failure_reason = "frame_not_initialized";
      summary.observations.push_back(row);
      continue;
    }
    if (board_id != scene_state.reference_board_id &&
        (board_state == nullptr || !board_state->initialized)) {
      row.failure_reason = "board_not_initialized";
      summary.observations.push_back(row);
      continue;
    }

    const auto local_pose_it = local_pose_by_key.find(entry.first);
    if (local_pose_it == local_pose_by_key.end()) {
      row.failure_reason = "local_pose_refit_failed";
      summary.observations.push_back(row);
      continue;
    }

    row.local_pose_refit_success = true;
    row.residual_rmse =
        ComputeBoardObservationResidualRmse(measurement_result,
                                            scene_state,
                                            frame_index,
                                            board_id);
    Eigen::Isometry3d T_camera_reference(frame_state->T_camera_reference);
    const auto local_reference_it = local_pose_by_key.find(
        std::make_pair(frame_index, scene_state.reference_board_id));
    if (local_reference_it != local_pose_by_key.end()) {
      T_camera_reference = local_reference_it->second;
      row.reference_pose_from_local_refit = true;
    }
    const Eigen::Isometry3d& T_camera_board_local = local_pose_it->second;
    const Eigen::Isometry3d T_reference_board_global =
        board_id == scene_state.reference_board_id
            ? Eigen::Isometry3d::Identity()
            : Eigen::Isometry3d(board_state->T_reference_board);
    const Eigen::Isometry3d T_reference_board_obs =
        T_camera_reference.inverse() * T_camera_board_local;
    const Eigen::Isometry3d delta =
        T_reference_board_global.inverse() * T_reference_board_obs;
    row.translation_correction_mm = delta.translation() * 1000.0;
    row.translation_error_mm = row.translation_correction_mm.norm();
    const Eigen::AngleAxisd delta_angle_axis(delta.rotation());
    row.rotation_correction_deg =
        delta_angle_axis.axis() * delta_angle_axis.angle() * 180.0 / M_PI;
    row.rotation_error_deg = row.rotation_correction_deg.norm();

    const double sigma_t =
        std::max(1e-6, options.consistency_translation_sigma_mm);
    const double sigma_r =
        std::max(1e-6, options.consistency_rotation_sigma_deg);
    const double e_t = row.translation_error_mm / sigma_t;
    const double e_r = row.rotation_error_deg / sigma_r;
    const double e_cons = std::sqrt(e_t * e_t + e_r * e_r);

    const bool is_reference_board =
        board_id == scene_state.reference_board_id;
    double weight = 1.0;
    if (!is_reference_board) {
      switch (options.consistency_weight_mode) {
        case AslamBackendCalibrationOptions::ConsistencyWeightMode::Cauchy:
          weight = 1.0 / (1.0 + e_cons * e_cons);
          break;
      }
    }
    weight = std::max(options.consistency_min_weight, std::min(1.0, weight));
    row.consistency_weight = weight;
    row.final_weight = weight;

    if (!is_reference_board && options.consistency_hard_reject_enabled &&
        row.translation_error_mm >= options.consistency_hard_reject_translation_mm &&
        row.rotation_error_deg >= options.consistency_hard_reject_rotation_deg &&
        row.residual_rmse >= options.consistency_hard_reject_residual_px) {
      row.hard_rejected = true;
      ++summary.hard_rejected_observation_count;
    }

    ++summary.successful_observation_count;
    if (weight < 0.999999) {
      ++summary.downweighted_observation_count;
    }
    summary.min_consistency_weight =
        std::min(summary.min_consistency_weight, weight);
    summary.max_translation_error_mm =
        std::max(summary.max_translation_error_mm, row.translation_error_mm);
    summary.max_rotation_error_deg =
        std::max(summary.max_rotation_error_deg, row.rotation_error_deg);
    weight_sum += weight;
    summary.observations.push_back(row);
  }

  if (summary.successful_observation_count > 0) {
    summary.mean_consistency_weight =
        weight_sum / static_cast<double>(summary.successful_observation_count);
    summary.success = true;
  } else {
    summary.failure_reason = "No successful consistency observations.";
  }
  return summary;
}

void ApplyConsistencyWeightSummaryToDataset(
    const ConsistencyWeightSummary& summary,
    CalibrationMeasurementDataset* dataset,
    const AslamBackendCalibrationOptions& options) {
  if (dataset == nullptr || !summary.success) {
    return;
  }
  std::map<std::pair<int, int>, ConsistencyObservationWeightSummaryEntry> by_key;
  for (const ConsistencyObservationWeightSummaryEntry& row : summary.observations) {
    by_key[std::make_pair(row.frame_index, row.board_id)] = row;
  }

  for (JointPointObservation& observation : dataset->solver_observations) {
    const auto it = by_key.find(std::make_pair(observation.frame_index, observation.board_id));
    observation.consistency_weight = 1.0;
    observation.final_observation_weight = observation.observation_weight;
    observation.consistency_hard_rejected = false;
    if (it == by_key.end()) {
      continue;
    }
    const ConsistencyObservationWeightSummaryEntry& row = it->second;
    bool apply = false;
    if (observation.point_type == JointPointType::Outer) {
      apply = options.consistency_apply_to_outer;
    } else {
      apply = options.consistency_apply_to_internal;
    }
    if (!apply) {
      continue;
    }
    observation.consistency_weight = row.consistency_weight;
    observation.final_observation_weight =
        observation.observation_weight * observation.consistency_weight;
    observation.consistency_hard_rejected = row.hard_rejected;
    if (row.hard_rejected && options.consistency_hard_reject_enabled) {
      observation.used_in_solver = false;
      observation.final_observation_weight = 0.0;
    }
  }
}

void WriteConsistencyWeightSummary(
    const std::string& path,
    const ConsistencyWeightSummary& summary) {
  std::ofstream output(path.c_str());
  output << "success: " << (summary.success ? 1 : 0) << "\n";
  output << "failure_reason: " << summary.failure_reason << "\n";
  output << "pose_source: " << summary.pose_source << "\n";
  output << "observation_count: " << summary.observation_count << "\n";
  output << "successful_observation_count: "
         << summary.successful_observation_count << "\n";
  output << "downweighted_observation_count: "
         << summary.downweighted_observation_count << "\n";
  output << "hard_rejected_observation_count: "
         << summary.hard_rejected_observation_count << "\n";
  output << "mean_consistency_weight: " << summary.mean_consistency_weight << "\n";
  output << "min_consistency_weight: " << summary.min_consistency_weight << "\n";
  output << "max_translation_error_mm: " << summary.max_translation_error_mm << "\n";
  output << "max_rotation_error_deg: " << summary.max_rotation_error_deg << "\n";
  for (const std::string& warning : summary.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteConsistencyPerBoardSummary(
    const std::string& path,
    const ConsistencyWeightSummary& summary) {
  std::ofstream output(path.c_str());
  output << "board_id,support_observation_count,mean_consistency_weight,min_consistency_weight,"
         << "mean_translation_error_mm,max_translation_error_mm,mean_rotation_error_deg,"
         << "max_rotation_error_deg,hard_rejected_count\n";
  std::map<int, std::vector<ConsistencyObservationWeightSummaryEntry> > grouped;
  for (const ConsistencyObservationWeightSummaryEntry& row : summary.observations) {
    if (!row.local_pose_refit_success) {
      continue;
    }
    grouped[row.board_id].push_back(row);
  }
  for (const auto& entry : grouped) {
    double weight_sum = 0.0;
    double min_weight = 1.0;
    double translation_sum = 0.0;
    double max_translation = 0.0;
    double rotation_sum = 0.0;
    double max_rotation = 0.0;
    int hard_rejected_count = 0;
    for (const ConsistencyObservationWeightSummaryEntry& row : entry.second) {
      weight_sum += row.consistency_weight;
      min_weight = std::min(min_weight, row.consistency_weight);
      translation_sum += row.translation_error_mm;
      max_translation = std::max(max_translation, row.translation_error_mm);
      rotation_sum += row.rotation_error_deg;
      max_rotation = std::max(max_rotation, row.rotation_error_deg);
      if (row.hard_rejected) {
        ++hard_rejected_count;
      }
    }
    const double count = static_cast<double>(entry.second.size());
    output << entry.first << ","
           << entry.second.size() << ","
           << (count > 0.0 ? weight_sum / count : 1.0) << ","
           << min_weight << ","
           << (count > 0.0 ? translation_sum / count : 0.0) << ","
           << max_translation << ","
           << (count > 0.0 ? rotation_sum / count : 0.0) << ","
           << max_rotation << ","
           << hard_rejected_count << "\n";
  }
}

void WriteConsistencyPerFrameSummary(
    const std::string& path,
    const ConsistencyWeightSummary& summary) {
  std::ofstream output(path.c_str());
  output << "frame_index,support_observation_count,mean_consistency_weight,min_consistency_weight,"
         << "mean_translation_error_mm,max_translation_error_mm,mean_rotation_error_deg,"
         << "max_rotation_error_deg,worst_board_id\n";
  std::map<int, std::vector<ConsistencyObservationWeightSummaryEntry> > grouped;
  for (const ConsistencyObservationWeightSummaryEntry& row : summary.observations) {
    if (!row.local_pose_refit_success) {
      continue;
    }
    grouped[row.frame_index].push_back(row);
  }
  for (const auto& entry : grouped) {
    double weight_sum = 0.0;
    double min_weight = 1.0;
    double translation_sum = 0.0;
    double max_translation = 0.0;
    double rotation_sum = 0.0;
    double max_rotation = 0.0;
    int worst_board_id = -1;
    for (const ConsistencyObservationWeightSummaryEntry& row : entry.second) {
      weight_sum += row.consistency_weight;
      min_weight = std::min(min_weight, row.consistency_weight);
      translation_sum += row.translation_error_mm;
      rotation_sum += row.rotation_error_deg;
      if (row.translation_error_mm >= max_translation) {
        max_translation = row.translation_error_mm;
        worst_board_id = row.board_id;
      }
      max_rotation = std::max(max_rotation, row.rotation_error_deg);
    }
    const double count = static_cast<double>(entry.second.size());
    output << entry.first << ","
           << entry.second.size() << ","
           << (count > 0.0 ? weight_sum / count : 1.0) << ","
           << min_weight << ","
           << (count > 0.0 ? translation_sum / count : 0.0) << ","
           << max_translation << ","
           << (count > 0.0 ? rotation_sum / count : 0.0) << ","
           << max_rotation << ","
           << worst_board_id << "\n";
  }
}

void WriteTopDownweightedObservations(
    const std::string& path,
    const ConsistencyWeightSummary& summary) {
  std::ofstream output(path.c_str());
  output << "frame_index,frame_label,board_id,translation_error_mm,rotation_error_deg,"
         << "translation_x_mm,translation_y_mm,translation_z_mm,"
         << "rotation_x_deg,rotation_y_deg,rotation_z_deg,"
         << "residual_rmse,polar_angle_deg,consistency_weight,final_weight,"
         << "num_outer_points,num_internal_points,hard_rejected,local_pose_refit_success,"
         << "reference_pose_from_local_refit\n";
  std::vector<ConsistencyObservationWeightSummaryEntry> rows = summary.observations;
  std::sort(rows.begin(), rows.end(),
            [](const ConsistencyObservationWeightSummaryEntry& lhs,
               const ConsistencyObservationWeightSummaryEntry& rhs) {
              if (lhs.consistency_weight != rhs.consistency_weight) {
                return lhs.consistency_weight < rhs.consistency_weight;
              }
              return lhs.translation_error_mm > rhs.translation_error_mm;
            });
  for (const ConsistencyObservationWeightSummaryEntry& row : rows) {
    output << row.frame_index << ","
           << row.frame_label << ","
           << row.board_id << ","
           << row.translation_error_mm << ","
           << row.rotation_error_deg << ","
           << row.translation_correction_mm.x() << ","
           << row.translation_correction_mm.y() << ","
           << row.translation_correction_mm.z() << ","
           << row.rotation_correction_deg.x() << ","
           << row.rotation_correction_deg.y() << ","
           << row.rotation_correction_deg.z() << ","
           << row.residual_rmse << ","
           << row.polar_angle_deg << ","
           << row.consistency_weight << ","
           << row.final_weight << ","
           << row.num_outer_points << ","
           << row.num_internal_points << ","
           << (row.hard_rejected ? 1 : 0) << ","
           << (row.local_pose_refit_success ? 1 : 0) << ","
           << (row.reference_pose_from_local_refit ? 1 : 0) << "\n";
  }
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

double PriorWeightForLabel(const CalibrationPriorSettings& priors,
                           const std::string& label) {
  if (label == "fu" || label == "fv") {
    return priors.intrinsics_anchor_weight_focal;
  }
  if (label == "cu" || label == "cv") {
    return priors.intrinsics_anchor_weight_principal;
  }
  return priors.intrinsics_anchor_weight_xi_alpha;
}

struct MeasurementSelectionStats {
  std::set<int> accepted_frame_indices;
  std::set<std::pair<int, int> > accepted_board_observation_keys;
  int accepted_frame_count = 0;
  int accepted_board_observation_count = 0;
  int accepted_outer_board_observation_count = 0;
  int accepted_internal_board_observation_count = 0;
  int accepted_outer_point_count = 0;
  int accepted_internal_point_count = 0;
  int accepted_total_point_count = 0;
};

MeasurementSelectionStats ComputeMeasurementSelectionStats(
    const std::vector<JointPointObservation>& solver_observations) {
  MeasurementSelectionStats stats;
  std::set<std::pair<int, int> > outer_board_keys;
  std::set<std::pair<int, int> > internal_board_keys;

  for (const JointPointObservation& observation : solver_observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    const std::pair<int, int> key(observation.frame_index, observation.board_id);
    stats.accepted_frame_indices.insert(observation.frame_index);
    stats.accepted_board_observation_keys.insert(key);
    ++stats.accepted_total_point_count;
    if (observation.point_type == JointPointType::Outer) {
      ++stats.accepted_outer_point_count;
      outer_board_keys.insert(key);
    } else {
      ++stats.accepted_internal_point_count;
      internal_board_keys.insert(key);
    }
  }

  stats.accepted_frame_count = static_cast<int>(stats.accepted_frame_indices.size());
  stats.accepted_board_observation_count =
      static_cast<int>(stats.accepted_board_observation_keys.size());
  stats.accepted_outer_board_observation_count =
      static_cast<int>(outer_board_keys.size());
  stats.accepted_internal_board_observation_count =
      static_cast<int>(internal_board_keys.size());
  return stats;
}

void ApplyMeasurementSelectionStatsToDataset(const MeasurementSelectionStats& stats,
                                             CalibrationMeasurementDataset* dataset) {
  if (dataset == nullptr) {
    throw std::runtime_error(
        "ApplyMeasurementSelectionStatsToDataset requires a valid dataset pointer.");
  }
  dataset->accepted_frame_indices = stats.accepted_frame_indices;
  dataset->accepted_board_observation_keys = stats.accepted_board_observation_keys;
  dataset->accepted_frame_count = stats.accepted_frame_count;
  dataset->accepted_board_observation_count = stats.accepted_board_observation_count;
  dataset->accepted_outer_point_count = stats.accepted_outer_point_count;
  dataset->accepted_internal_point_count = stats.accepted_internal_point_count;
  dataset->accepted_total_point_count = stats.accepted_total_point_count;
}

JointMeasurementBuildResult BuildMeasurementResult(
    const CalibrationMeasurementDataset& dataset,
    int reference_board_id) {
  JointMeasurementBuildResult result;
  result.reference_board_id = reference_board_id;
  result.frames = dataset.frames;
  result.solver_observations = dataset.solver_observations;
  result.warnings = dataset.warnings;

  const MeasurementSelectionStats stats =
      ComputeMeasurementSelectionStats(dataset.solver_observations);
  result.used_frame_count = stats.accepted_frame_count;
  result.accepted_outer_board_observation_count =
      stats.accepted_outer_board_observation_count;
  result.accepted_internal_board_observation_count =
      stats.accepted_internal_board_observation_count;
  result.used_board_observation_count = stats.accepted_board_observation_count;
  result.used_outer_point_count = stats.accepted_outer_point_count;
  result.used_internal_point_count = stats.accepted_internal_point_count;
  result.used_total_point_count = stats.accepted_total_point_count;

  result.success = !result.frames.empty() && result.used_total_point_count > 0;
  if (!result.success) {
    result.failure_reason =
        dataset.failure_reason.empty()
            ? "CalibrationMeasurementDataset has no used-in-solver observations."
            : dataset.failure_reason;
  }
  return result;
}

template <typename GeometryT>
OuterBootstrapCameraIntrinsics GeometryToIntrinsics(const GeometryT& geometry);

template <>
OuterBootstrapCameraIntrinsics GeometryToIntrinsics<DsGeometry>(const DsGeometry& geometry) {
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model = "ds";
  intrinsics.distortion_model = "none";
  intrinsics.xi = geometry.projection().xi();
  intrinsics.alpha = geometry.projection().alpha();
  intrinsics.fu = geometry.projection().fu();
  intrinsics.fv = geometry.projection().fv();
  intrinsics.cu = geometry.projection().cu();
  intrinsics.cv = geometry.projection().cv();
  intrinsics.resolution =
      cv::Size(geometry.projection().width(), geometry.projection().height());
  return intrinsics;
}

template <>
OuterBootstrapCameraIntrinsics GeometryToIntrinsics<EucmGeometry>(const EucmGeometry& geometry) {
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model = "eucm";
  intrinsics.distortion_model = "none";
  intrinsics.alpha = geometry.projection().alpha();
  intrinsics.beta = geometry.projection().beta();
  intrinsics.fu = geometry.projection().fu();
  intrinsics.fv = geometry.projection().fv();
  intrinsics.cu = geometry.projection().cu();
  intrinsics.cv = geometry.projection().cv();
  intrinsics.resolution =
      cv::Size(geometry.projection().width(), geometry.projection().height());
  return intrinsics;
}

template <>
OuterBootstrapCameraIntrinsics GeometryToIntrinsics<PinholeEquiGeometry>(
    const PinholeEquiGeometry& geometry) {
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model = "pinhole";
  intrinsics.distortion_model = "equi";
  intrinsics.fu = geometry.projection().fu();
  intrinsics.fv = geometry.projection().fv();
  intrinsics.cu = geometry.projection().cu();
  intrinsics.cv = geometry.projection().cv();
  Eigen::MatrixXd distortion_parameters;
  geometry.projection().distortion().getParameters(distortion_parameters);
  intrinsics.distortion_coeffs.resize(
      static_cast<std::size_t>(distortion_parameters.rows()), 0.0);
  for (Eigen::Index index = 0; index < distortion_parameters.rows(); ++index) {
    intrinsics.distortion_coeffs[static_cast<std::size_t>(index)] =
        distortion_parameters(index, 0);
  }
  intrinsics.resolution =
      cv::Size(geometry.projection().width(), geometry.projection().height());
  return intrinsics;
}

template <>
OuterBootstrapCameraIntrinsics GeometryToIntrinsics<MeiGeometry>(
    const MeiGeometry& geometry) {
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model = "omni";
  intrinsics.distortion_model = "radtan";
  intrinsics.xi = geometry.projection().xi();
  intrinsics.fu = geometry.projection().fu();
  intrinsics.fv = geometry.projection().fv();
  intrinsics.cu = geometry.projection().cu();
  intrinsics.cv = geometry.projection().cv();
  Eigen::MatrixXd distortion_parameters;
  geometry.projection().distortion().getParameters(distortion_parameters);
  intrinsics.distortion_coeffs.resize(
      static_cast<std::size_t>(distortion_parameters.rows()), 0.0);
  for (Eigen::Index index = 0; index < distortion_parameters.rows(); ++index) {
    intrinsics.distortion_coeffs[static_cast<std::size_t>(index)] =
        distortion_parameters(index, 0);
  }
  intrinsics.resolution =
      cv::Size(geometry.projection().width(), geometry.projection().height());
  return intrinsics;
}

template <>
OuterBootstrapCameraIntrinsics GeometryToIntrinsics<OmniNoneGeometry>(
    const OmniNoneGeometry& geometry) {
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model = "omni";
  intrinsics.distortion_model = "none";
  intrinsics.xi = geometry.projection().xi();
  intrinsics.fu = geometry.projection().fu();
  intrinsics.fv = geometry.projection().fv();
  intrinsics.cu = geometry.projection().cu();
  intrinsics.cv = geometry.projection().cv();
  intrinsics.distortion_coeffs.clear();
  intrinsics.resolution =
      cv::Size(geometry.projection().width(), geometry.projection().height());
  return intrinsics;
}

template <typename GeometryT>
bool ComputeObservationGeometryForCamera(
    const GeometryT& camera,
    const Eigen::Vector2d& observed_image_xy,
    AngularObservationGeometry* geometry) {
  Eigen::Vector3d observed_ray = Eigen::Vector3d::Zero();
  if (!camera.keypointToEuclidean(observed_image_xy, observed_ray)) {
    return false;
  }
  return ComputeAngularObservationGeometryFromRay(
      observed_image_xy, observed_ray, geometry);
}

template <typename GeometryT>
bool ComputePredictionGeometryForCamera(
    const GeometryT& camera,
    const Eigen::Vector4d& point_camera,
    AngularPredictionGeometry* geometry) {
  Eigen::Vector2d predicted_image_xy = Eigen::Vector2d::Zero();
  if (!camera.homogeneousToKeypoint(point_camera, predicted_image_xy)) {
    return false;
  }
  return ComputeAngularPredictionGeometryFromPoint(
      point_camera.head<3>(), predicted_image_xy, geometry);
}

template <typename GeometryT>
double EstimateAngularSigmaPerPixelForCamera(
    const GeometryT& camera,
    const Eigen::Vector2d& observed_image_xy,
    const AngularObservationGeometry& observation_geometry,
    double finite_difference_step_px = 1.0) {
  if (!observation_geometry.success ||
      !(finite_difference_step_px > 0.0) ||
      !std::isfinite(finite_difference_step_px)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double squared_sum = 0.0;
  int count = 0;
  const Eigen::Vector2d offsets[] = {
      Eigen::Vector2d(finite_difference_step_px, 0.0),
      Eigen::Vector2d(-finite_difference_step_px, 0.0),
      Eigen::Vector2d(0.0, finite_difference_step_px),
      Eigen::Vector2d(0.0, -finite_difference_step_px),
  };
  for (const Eigen::Vector2d& offset : offsets) {
    AngularObservationGeometry shifted_geometry;
    if (!ComputeObservationGeometryForCamera(
            camera, observed_image_xy + offset, &shifted_geometry)) {
      continue;
    }
    const Eigen::Vector3d ray_delta =
        shifted_geometry.observed_ray - observation_geometry.observed_ray;
    const double sigma =
        (observation_geometry.tangent_basis.transpose() * ray_delta).norm() /
        finite_difference_step_px;
    if (sigma > 0.0 && std::isfinite(sigma)) {
      squared_sum += sigma * sigma;
      ++count;
    }
  }
  return count > 0
             ? std::sqrt(squared_sum / static_cast<double>(count))
             : std::numeric_limits<double>::quiet_NaN();
}

template <typename GeometryT>
boost::shared_ptr<GeometryT> MakeTypedGeometry(
    const OuterBootstrapCameraIntrinsics& intrinsics);

template <>
boost::shared_ptr<DsGeometry> MakeTypedGeometry<DsGeometry>(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  DsProjection projection(intrinsics.xi, intrinsics.alpha, intrinsics.fu, intrinsics.fv,
                          intrinsics.cu, intrinsics.cv, intrinsics.resolution.width,
                          intrinsics.resolution.height);
  return boost::shared_ptr<DsGeometry>(
      new DsGeometry(projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask()));
}

template <>
boost::shared_ptr<EucmGeometry> MakeTypedGeometry<EucmGeometry>(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  EucmProjection projection(intrinsics.alpha, intrinsics.beta, intrinsics.fu, intrinsics.fv,
                            intrinsics.cu, intrinsics.cv, intrinsics.resolution.width,
                            intrinsics.resolution.height);
  return boost::shared_ptr<EucmGeometry>(
      new EucmGeometry(projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask()));
}

template <>
boost::shared_ptr<PinholeEquiGeometry> MakeTypedGeometry<PinholeEquiGeometry>(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  const std::vector<double> distortion =
      intrinsics.distortion_coeffs.size() == 4
          ? intrinsics.distortion_coeffs
          : std::vector<double>{0.0, 0.0, 0.0, 0.0};
  PinholeEquiProjection projection(
      intrinsics.fu, intrinsics.fv, intrinsics.cu, intrinsics.cv,
      intrinsics.resolution.width, intrinsics.resolution.height,
      aslam::cameras::EquidistantDistortion(
          distortion[0], distortion[1], distortion[2], distortion[3]));
  return boost::shared_ptr<PinholeEquiGeometry>(
      new PinholeEquiGeometry(
          projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask()));
}

template <>
boost::shared_ptr<MeiGeometry> MakeTypedGeometry<MeiGeometry>(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  const std::vector<double> distortion =
      intrinsics.distortion_coeffs.size() == 4
          ? intrinsics.distortion_coeffs
          : std::vector<double>{0.0, 0.0, 0.0, 0.0};
  MeiProjection projection(
      intrinsics.xi, intrinsics.fu, intrinsics.fv, intrinsics.cu,
      intrinsics.cv, intrinsics.resolution.width, intrinsics.resolution.height,
      aslam::cameras::RadialTangentialDistortion(
          distortion[0], distortion[1], distortion[2], distortion[3]));
  return boost::shared_ptr<MeiGeometry>(
      new MeiGeometry(
          projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask()));
}

template <>
boost::shared_ptr<OmniNoneGeometry> MakeTypedGeometry<OmniNoneGeometry>(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  OmniNoneProjection projection(
      intrinsics.xi, intrinsics.fu, intrinsics.fv, intrinsics.cu,
      intrinsics.cv, intrinsics.resolution.width, intrinsics.resolution.height);
  return boost::shared_ptr<OmniNoneGeometry>(new OmniNoneGeometry(
      projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask()));
}

struct PoseVariableState {
  sm::kinematics::Transformation transform;
  boost::shared_ptr<aslam::backend::MappedRotationQuaternion> rotation_dv;
  boost::shared_ptr<aslam::backend::MappedEuclideanPoint> translation_dv;
  aslam::backend::TransformationExpression expression;
};

struct ObservationBudget {
  int outer_count = 0;
  int internal_count = 0;
};

std::map<std::pair<int, int>, ObservationBudget> BuildObservationBudgets(
    const JointMeasurementBuildResult& measurement_result) {
  std::map<std::pair<int, int>, ObservationBudget> budgets;
  for (const JointPointObservation& observation : measurement_result.solver_observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    ObservationBudget& budget =
        budgets[std::make_pair(observation.frame_index, observation.board_id)];
    if (observation.point_type == JointPointType::Outer) {
      ++budget.outer_count;
    } else {
      ++budget.internal_count;
    }
  }
  return budgets;
}

bool ComposeCameraBoardPoseFromReferenceChain(
    const JointReprojectionSceneState& scene_state,
    int frame_index,
    int board_id,
    Eigen::Matrix4d* T_camera_board) {
  if (T_camera_board == nullptr) {
    return false;
  }
  const JointSceneFrameState* frame_state =
      FindJointSceneFrameState(scene_state, frame_index);
  if (frame_state == nullptr || !frame_state->initialized) {
    return false;
  }
  Eigen::Matrix4d T_reference_board = Eigen::Matrix4d::Identity();
  if (board_id != scene_state.reference_board_id) {
    const JointSceneBoardState* board_state =
        FindJointSceneBoardState(scene_state, board_id);
    if (board_state == nullptr || !board_state->initialized) {
      return false;
    }
    T_reference_board = board_state->T_reference_board;
  }
  *T_camera_board = frame_state->T_camera_reference * T_reference_board;
  return true;
}

double RmseFromSquaredSum(double squared_sum, int count) {
  return count > 0 ? std::sqrt(squared_sum / static_cast<double>(count)) : 0.0;
}

template <typename GeometryT>
JointResidualEvaluationResult EvaluateIndependentFrameBoardPoseResiduals(
    const JointMeasurementBuildResult& measurement_result,
    const boost::shared_ptr<GeometryT>& camera_geometry,
    const std::map<std::pair<int, int>, PoseVariableState>& local_pose_variables,
    int reference_board_id,
    int top_k) {
  JointResidualEvaluationResult result;
  result.reference_board_id = reference_board_id;
  if (!measurement_result.success) {
    result.failure_reason = "measurement_result.success is false";
    return result;
  }
  if (camera_geometry == nullptr) {
    result.failure_reason = "missing camera geometry";
    return result;
  }

  std::map<std::pair<int, int>, std::pair<double, int> >
      board_observation_accumulators;
  std::map<std::pair<int, int>, std::pair<int, std::string> >
      board_observation_labels;
  std::map<int, std::tuple<double, int, int, int> > board_accumulators;
  std::map<int, std::pair<double, int> > frame_accumulators;
  std::map<int, std::pair<int, std::string> > frame_labels;
  std::map<int, std::vector<int> > frame_visible_boards;

  double total_squared_sum = 0.0;
  double outer_squared_sum = 0.0;
  double internal_squared_sum = 0.0;
  int total_count = 0;
  int outer_count = 0;
  int internal_count = 0;

  for (const JointPointObservation& observation :
       measurement_result.solver_observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    const std::pair<int, int> key(observation.frame_index,
                                  observation.board_id);
    const auto pose_it = local_pose_variables.find(key);
    if (pose_it == local_pose_variables.end()) {
      continue;
    }
    const Eigen::Vector4d point_board(observation.target_xyz_board.x(),
                                      observation.target_xyz_board.y(),
                                      observation.target_xyz_board.z(),
                                      1.0);
    const Eigen::Vector4d point_camera_h =
        pose_it->second.expression.toTransformationMatrix() * point_board;
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    const bool valid_projection =
        camera_geometry->homogeneousToKeypoint(point_camera_h, predicted) &&
        predicted.allFinite();

    JointResidualPointDiagnostics point;
    point.frame_index = observation.frame_index;
    point.frame_label = observation.frame_label;
    point.board_id = observation.board_id;
    point.point_id = observation.point_id;
    point.point_type = observation.point_type;
    point.observed_image_xy = observation.image_xy;
    point.target_xyz_board = observation.target_xyz_board;
    point.predicted_image_xy =
        valid_projection ? predicted
                         : Eigen::Vector2d(
                               std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::quiet_NaN());
    if (valid_projection) {
      point.residual_xy = predicted - observation.image_xy;
    } else {
      point.residual_xy = Eigen::Vector2d::Constant(100.0);
    }
    point.residual_norm = point.residual_xy.norm();
    point.quality = observation.quality;
    point.used_in_solver = observation.used_in_solver;
    point.frame_storage_index = observation.frame_storage_index;
    point.source_board_observation_index =
        observation.source_board_observation_index;
    point.source_point_index = observation.source_point_index;
    point.source_kind = observation.source_kind;
    result.point_diagnostics.push_back(point);

    const double squared = point.residual_xy.squaredNorm();
    total_squared_sum += squared;
    ++total_count;
    if (observation.point_type == JointPointType::Outer) {
      outer_squared_sum += squared;
      ++outer_count;
    } else {
      internal_squared_sum += squared;
      ++internal_count;
    }
    board_observation_accumulators[key].first += squared;
    board_observation_accumulators[key].second += 1;
    board_observation_labels[key] =
        std::make_pair(observation.board_id, observation.frame_label);

    std::tuple<double, int, int, int>& board_acc =
        board_accumulators[observation.board_id];
    std::get<0>(board_acc) += squared;
    std::get<1>(board_acc) += 1;
    if (observation.point_type == JointPointType::Outer) {
      std::get<2>(board_acc) += 1;
    } else {
      std::get<3>(board_acc) += 1;
    }
    frame_accumulators[observation.frame_index].first += squared;
    frame_accumulators[observation.frame_index].second += 1;
    frame_labels[observation.frame_index] =
        std::make_pair(observation.frame_index, observation.frame_label);
    std::vector<int>& visible_boards =
        frame_visible_boards[observation.frame_index];
    if (std::find(visible_boards.begin(), visible_boards.end(),
                  observation.board_id) == visible_boards.end()) {
      visible_boards.push_back(observation.board_id);
    }
  }

  if (total_count <= 0) {
    result.failure_reason =
        "No used observations with independent frame-board poses";
    return result;
  }
  result.overall_rmse = RmseFromSquaredSum(total_squared_sum, total_count);
  result.outer_only_rmse = RmseFromSquaredSum(outer_squared_sum, outer_count);
  result.internal_only_rmse =
      RmseFromSquaredSum(internal_squared_sum, internal_count);
  result.overall_image_plane_rmse = result.overall_rmse;
  result.outer_only_image_plane_rmse = result.outer_only_rmse;
  result.internal_only_image_plane_rmse = result.internal_only_rmse;

  for (const auto& entry : board_observation_accumulators) {
    JointResidualBoardObservationDiagnostics diagnostics;
    diagnostics.frame_index = entry.first.first;
    diagnostics.board_id = entry.first.second;
    diagnostics.frame_label = board_observation_labels[entry.first].second;
    diagnostics.point_count = entry.second.second;
    diagnostics.rmse =
        RmseFromSquaredSum(entry.second.first, entry.second.second);
    for (const JointResidualPointDiagnostics& point :
         result.point_diagnostics) {
      if (point.frame_index == diagnostics.frame_index &&
          point.board_id == diagnostics.board_id) {
        if (point.point_type == JointPointType::Outer) {
          ++diagnostics.outer_point_count;
        } else {
          ++diagnostics.internal_point_count;
        }
      }
    }
    result.board_observation_diagnostics.push_back(diagnostics);
  }
  for (const auto& entry : board_accumulators) {
    JointResidualBoardDiagnostics diagnostics;
    diagnostics.board_id = entry.first;
    diagnostics.point_count = std::get<1>(entry.second);
    diagnostics.outer_point_count = std::get<2>(entry.second);
    diagnostics.internal_point_count = std::get<3>(entry.second);
    diagnostics.rmse =
        RmseFromSquaredSum(std::get<0>(entry.second),
                           std::get<1>(entry.second));
    for (const JointResidualBoardObservationDiagnostics& observation :
         result.board_observation_diagnostics) {
      if (observation.board_id == diagnostics.board_id) {
        ++diagnostics.observation_count;
      }
    }
    result.board_diagnostics.push_back(diagnostics);
  }
  for (const auto& entry : frame_accumulators) {
    JointResidualFrameDiagnostics diagnostics;
    diagnostics.frame_index = entry.first;
    diagnostics.frame_label = frame_labels[entry.first].second;
    diagnostics.visible_board_ids = frame_visible_boards[entry.first];
    diagnostics.point_count = entry.second.second;
    diagnostics.rmse = RmseFromSquaredSum(entry.second.first,
                                          entry.second.second);
    for (const JointResidualPointDiagnostics& point :
         result.point_diagnostics) {
      if (point.frame_index == diagnostics.frame_index) {
        if (point.point_type == JointPointType::Outer) {
          ++diagnostics.outer_point_count;
        } else {
          ++diagnostics.internal_point_count;
        }
      }
    }
    result.frame_diagnostics.push_back(diagnostics);
  }
  result.worst_points = result.point_diagnostics;
  std::sort(result.worst_points.begin(), result.worst_points.end(),
            [](const JointResidualPointDiagnostics& lhs,
               const JointResidualPointDiagnostics& rhs) {
              return lhs.residual_norm > rhs.residual_norm;
            });
  if (top_k >= 0 && static_cast<int>(result.worst_points.size()) > top_k) {
    result.worst_points.resize(static_cast<std::size_t>(top_k));
  }
  result.worst_board_observations = result.board_observation_diagnostics;
  std::sort(result.worst_board_observations.begin(),
            result.worst_board_observations.end(),
            [](const JointResidualBoardObservationDiagnostics& lhs,
               const JointResidualBoardObservationDiagnostics& rhs) {
              return lhs.rmse > rhs.rmse;
            });
  if (top_k >= 0 &&
      static_cast<int>(result.worst_board_observations.size()) > top_k) {
    result.worst_board_observations.resize(static_cast<std::size_t>(top_k));
  }
  result.worst_boards = result.board_diagnostics;
  std::sort(result.worst_boards.begin(), result.worst_boards.end(),
            [](const JointResidualBoardDiagnostics& lhs,
               const JointResidualBoardDiagnostics& rhs) {
              return lhs.rmse > rhs.rmse;
            });
  if (top_k >= 0 && static_cast<int>(result.worst_boards.size()) > top_k) {
    result.worst_boards.resize(static_cast<std::size_t>(top_k));
  }
  result.worst_frames = result.frame_diagnostics;
  std::sort(result.worst_frames.begin(), result.worst_frames.end(),
            [](const JointResidualFrameDiagnostics& lhs,
               const JointResidualFrameDiagnostics& rhs) {
              return lhs.rmse > rhs.rmse;
            });
  if (top_k >= 0 && static_cast<int>(result.worst_frames.size()) > top_k) {
    result.worst_frames.resize(static_cast<std::size_t>(top_k));
  }
  result.success = true;
  return result;
}

double ComputeBalanceWeight(const ObservationBudget& budget,
                            JointPointType point_type,
                            const AslamBackendCalibrationOptions& options) {
  if (options.observation_role_weight_mode == "unweighted_points") {
    return 1.0;
  }
  if (options.uniform_control_point_mode) {
    // Dense calibration targets contribute one independent measurement per
    // control point, matching Kalibr's per-corner unit covariance model.
    return 1.0;
  }
  const bool has_outer = budget.outer_count > 0;
  const bool has_internal = budget.internal_count > 0;
  double type_budget = 1.0;
  int type_count = 1;
  if (has_outer && has_internal) {
    const double internal_budget =
        options.observation_role_weight_mode == "outer_priority"
            ? std::max(0.0, std::min(1.0,
                                     options.internal_role_budget_when_mixed))
            : 0.5;
    type_budget = point_type == JointPointType::Internal
                      ? internal_budget
                      : 1.0 - internal_budget;
    type_count = point_type == JointPointType::Outer ? budget.outer_count
                                                     : budget.internal_count;
  } else if (point_type == JointPointType::Outer) {
    type_budget = 1.0;
    type_count = budget.outer_count;
  } else {
    type_budget = 1.0;
    type_count = budget.internal_count;
  }
  return type_budget / std::max(1, type_count);
}

template <typename GeometryT>
double ComputePolarAngleWeightScale(
    const GeometryT& camera,
    const JointPointObservation& observation,
    const AslamBackendCalibrationOptions& options) {
  if (!options.uniform_control_point_mode &&
      observation.point_type != JointPointType::Internal) {
    return 1.0;
  }
  if (options.polar_angle_weight_mode ==
          AslamBackendCalibrationOptions::PolarAngleWeightMode::None ||
      options.polar_angle_weight_mode ==
          AslamBackendCalibrationOptions::PolarAngleWeightMode::DiagnosticOnly) {
    return 1.0;
  }

  AngularObservationGeometry geometry;
  const double polar_angle =
      ComputeObservationGeometryForCamera(camera, observation.image_xy,
                                          &geometry)
          ? geometry.polar_angle_deg
          : std::numeric_limits<double>::quiet_NaN();
  if (!std::isfinite(polar_angle)) {
    return 1.0;
  }
  const double min_scale =
      std::max(0.0, std::min(1.0, options.polar_angle_weight_min_scale));

  if (options.polar_angle_weight_mode ==
      AslamBackendCalibrationOptions::PolarAngleWeightMode::FixedBins) {
    const std::vector<double>& edges = options.polar_angle_weight_bin_edges_deg;
    const std::vector<double>& scales = options.polar_angle_weight_fixed_bin_scales;
    const std::size_t bin_count = edges.size() >= 2 ? edges.size() - 1 : 0;
    for (std::size_t i = 0; i < bin_count && i < scales.size(); ++i) {
      if (polar_angle >= edges[i] && polar_angle < edges[i + 1]) {
        return std::max(min_scale, std::min(1.0, scales[i]));
      }
    }
    return 1.0;
  }

  const double reference_deg =
      options.polar_angle_weight_adaptive_sigma_reference_deg;
  const double growth =
      std::max(1e-6, options.polar_angle_weight_adaptive_sigma_growth);
  if (polar_angle <= reference_deg) {
    return 1.0;
  }
  const double normalized =
      (polar_angle - reference_deg) / std::max(1e-6, 90.0 - reference_deg);
  const double scale = 1.0 / (1.0 + growth * normalized * normalized);
  return std::max(min_scale, std::min(1.0, scale));
}

std::string ToCostCorePolarAngleWeightMode(
    AslamBackendCalibrationOptions::PolarAngleWeightMode mode) {
  switch (mode) {
    case AslamBackendCalibrationOptions::PolarAngleWeightMode::None:
      return "none";
    case AslamBackendCalibrationOptions::PolarAngleWeightMode::DiagnosticOnly:
      return "diagnostic_only";
    case AslamBackendCalibrationOptions::PolarAngleWeightMode::FixedBins:
      return "fixed_bins";
    case AslamBackendCalibrationOptions::PolarAngleWeightMode::AdaptiveSigma:
      return "adaptive_sigma";
  }
  return "none";
}

JointReprojectionCostOptions MakeCostOptionsForBackendResidualEvaluation(
    const AslamBackendCalibrationOptions& options) {
  JointReprojectionCostOptions cost_options;
  cost_options.uniform_control_point_mode = options.uniform_control_point_mode;
  cost_options.residual_model = options.residual_model;
  cost_options.hybrid_angular_threshold_deg =
      options.hybrid_angular_threshold_deg;
  cost_options.polar_continuous_hybrid_threshold_deg =
      options.polar_continuous_hybrid_threshold_deg;
  cost_options.polar_continuous_hybrid_temperature_deg =
      options.polar_continuous_hybrid_temperature_deg;
  cost_options.outer_huber_delta_pixels =
      options.use_huber_loss ? options.outer_huber_delta_pixels : 0.0;
  cost_options.internal_huber_delta_pixels =
      options.use_huber_loss ? options.internal_huber_delta_pixels : 0.0;
  cost_options.outer_huber_delta_radians =
      options.use_huber_loss ? options.outer_huber_delta_radians : 0.0;
  cost_options.internal_huber_delta_radians =
      options.use_huber_loss ? options.internal_huber_delta_radians : 0.0;
  cost_options.enable_invalid_projection_penalty =
      options.invalid_projection_penalty_pixels > 0.0 ||
      options.invalid_projection_penalty_radians > 0.0;
  cost_options.invalid_projection_penalty_pixels =
      options.invalid_projection_penalty_pixels;
  cost_options.invalid_projection_penalty_radians =
      options.invalid_projection_penalty_radians;
  cost_options.polar_angle_weight_mode =
      ToCostCorePolarAngleWeightMode(options.polar_angle_weight_mode);
  cost_options.polar_angle_weight_bin_edges_deg =
      options.polar_angle_weight_bin_edges_deg;
  cost_options.polar_angle_weight_fixed_bin_scales =
      options.polar_angle_weight_fixed_bin_scales;
  cost_options.polar_angle_weight_adaptive_sigma_reference_deg =
      options.polar_angle_weight_adaptive_sigma_reference_deg;
  cost_options.polar_angle_weight_adaptive_sigma_growth =
      options.polar_angle_weight_adaptive_sigma_growth;
  cost_options.polar_angle_weight_min_scale =
      options.polar_angle_weight_min_scale;
  cost_options.enable_angular_residual_diagnostics =
      options.enable_angular_residual_diagnostics;
  cost_options.angular_residual_bin_edges_deg =
      options.angular_residual_bin_edges_deg;
  cost_options.multi_board_consistency_weighting =
      options.multi_board_consistency_weighting;
  cost_options.consistency_apply_to_outer =
      options.consistency_apply_to_outer;
  cost_options.consistency_apply_to_internal =
      options.consistency_apply_to_internal;
  return cost_options;
}

Eigen::Matrix2d ComputeBackendInverseCovariance(
    double scalar_weight,
    JointPointType point_type,
    const AslamBackendCalibrationOptions& options) {
  const double safe_weight = std::max(0.0, scalar_weight);
  Eigen::Matrix2d inverse_covariance =
      safe_weight * Eigen::Matrix2d::Identity();
  if (point_type != JointPointType::Internal ||
      options.internal_anisotropic_weight_mode != "fixed_xy_scale") {
    return inverse_covariance;
  }

  const double x_scale =
      std::max(0.0, options.internal_anisotropic_x_scale);
  const double y_scale =
      std::max(0.0, options.internal_anisotropic_y_scale);
  inverse_covariance.setZero();
  inverse_covariance(0, 0) = safe_weight * x_scale;
  inverse_covariance(1, 1) = safe_weight * y_scale;
  return inverse_covariance;
}

CalibrationMeasurementDataset FilterMeasurementDataset(
    const CalibrationMeasurementDataset& dataset,
    const std::set<int>& selected_frame_indices,
    const std::set<int>& selected_board_ids) {
  CalibrationMeasurementDataset filtered = dataset;
  filtered.frames.clear();
  filtered.solver_observations.clear();

  for (const JointMeasurementFrameResult& frame : dataset.frames) {
    if (selected_frame_indices.find(frame.frame_index) == selected_frame_indices.end()) {
      continue;
    }
    JointMeasurementFrameResult filtered_frame = frame;
    filtered_frame.visible_board_ids.clear();
    filtered_frame.board_observations.clear();
    for (int board_id : frame.visible_board_ids) {
      if (selected_board_ids.find(board_id) != selected_board_ids.end()) {
        filtered_frame.visible_board_ids.push_back(board_id);
      }
    }
    for (const JointBoardObservation& board_observation : frame.board_observations) {
      if (selected_board_ids.find(board_observation.board_id) == selected_board_ids.end()) {
        continue;
      }
      filtered_frame.board_observations.push_back(board_observation);
    }
    if (!filtered_frame.board_observations.empty()) {
      filtered.frames.push_back(filtered_frame);
    }
  }

  for (const JointPointObservation& observation : dataset.solver_observations) {
    if (selected_frame_indices.find(observation.frame_index) == selected_frame_indices.end()) {
      continue;
    }
    if (selected_board_ids.find(observation.board_id) == selected_board_ids.end()) {
      continue;
    }
    filtered.solver_observations.push_back(observation);
  }

  const MeasurementSelectionStats stats =
      ComputeMeasurementSelectionStats(filtered.solver_observations);
  ApplyMeasurementSelectionStatsToDataset(stats, &filtered);
  if (filtered.accepted_total_point_count <= 0) {
    filtered.failure_reason = "Filtered backend debug subset is empty.";
  } else {
    filtered.failure_reason.clear();
  }
  return filtered;
}

CalibrationSceneState FilterSceneState(const CalibrationSceneState& scene_state,
                                       const std::set<int>& selected_frame_indices,
                                       const std::set<int>& selected_board_ids) {
  CalibrationSceneState filtered = scene_state;
  filtered.frames.clear();
  filtered.boards.clear();

  for (const JointSceneFrameState& frame : scene_state.frames) {
    if (selected_frame_indices.find(frame.frame_index) != selected_frame_indices.end()) {
      filtered.frames.push_back(frame);
    }
  }
  for (const JointSceneBoardState& board : scene_state.boards) {
    if (selected_board_ids.find(board.board_id) != selected_board_ids.end()) {
      filtered.boards.push_back(board);
    }
  }
  return filtered;
}

CalibrationBackendProblemInput BuildEffectiveProblemInput(
    const CalibrationBackendProblemInput& input,
    const AslamBackendCalibrationOptions& options) {
  const bool use_subset =
      options.debug_max_frames > 0 || options.debug_max_nonreference_boards >= 0;
  if (!use_subset && !options.force_pose_only) {
    return input;
  }

  std::set<int> used_frame_indices;
  for (const JointPointObservation& observation : input.measurement_dataset.solver_observations) {
    if (observation.used_in_solver) {
      used_frame_indices.insert(observation.frame_index);
    }
  }
  if (used_frame_indices.empty()) {
    return input;
  }

  std::vector<int> ordered_frame_indices;
  ordered_frame_indices.reserve(used_frame_indices.size());
  for (const JointMeasurementFrameResult& frame : input.measurement_dataset.frames) {
    if (used_frame_indices.find(frame.frame_index) != used_frame_indices.end()) {
      ordered_frame_indices.push_back(frame.frame_index);
    }
  }
  if (options.debug_max_frames > 0 &&
      static_cast<int>(ordered_frame_indices.size()) > options.debug_max_frames) {
    ordered_frame_indices.resize(static_cast<std::size_t>(options.debug_max_frames));
  }
  std::set<int> selected_frame_indices(ordered_frame_indices.begin(),
                                       ordered_frame_indices.end());

  std::map<int, int> board_point_counts;
  for (const JointPointObservation& observation : input.measurement_dataset.solver_observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    if (selected_frame_indices.find(observation.frame_index) == selected_frame_indices.end()) {
      continue;
    }
    if (observation.board_id == input.reference_board_id) {
      continue;
    }
    ++board_point_counts[observation.board_id];
  }

  std::vector<std::pair<int, int> > ranked_boards;
  ranked_boards.reserve(board_point_counts.size());
  for (const auto& entry : board_point_counts) {
    ranked_boards.push_back(std::make_pair(entry.first, entry.second));
  }
  std::sort(ranked_boards.begin(), ranked_boards.end(),
            [](const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
            });

  std::set<int> selected_board_ids;
  selected_board_ids.insert(input.reference_board_id);
  if (options.debug_max_nonreference_boards < 0) {
    for (const auto& entry : ranked_boards) {
      selected_board_ids.insert(entry.first);
    }
  } else {
    const int max_nonreference_boards = options.debug_max_nonreference_boards;
    for (std::size_t index = 0; index < ranked_boards.size() &&
                                static_cast<int>(index) < max_nonreference_boards;
         ++index) {
      selected_board_ids.insert(ranked_boards[index].first);
    }
  }

  CalibrationBackendProblemInput effective = input;
  effective.scene_state = FilterSceneState(
      input.scene_state, selected_frame_indices, selected_board_ids);
  effective.measurement_dataset = FilterMeasurementDataset(
      input.measurement_dataset, selected_frame_indices, selected_board_ids);
  if (options.force_pose_only) {
    effective.optimization_masks.optimize_intrinsics = false;
    effective.optimization_masks.delayed_intrinsics_release = false;
  }
  return effective;
}

struct ReprojectionDebugSample {
  bool valid_projection = false;
  Eigen::Vector2d predicted_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d residual_xy = Eigen::Vector2d::Zero();
  double residual_norm = 0.0;
  double backend_inv_r_scale = 0.0;
  double backend_m_estimator_weight = 0.0;
  double backend_raw_squared_error = 0.0;
  double backend_weighted_squared_error = 0.0;
};

template <typename ProjectionT>
class ProjectionAnchorError
    : public aslam::backend::ErrorTermFs<ProjectionT::DesignVariableDimension> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using projection_dv_t = aslam::backend::DesignVariableAdapter<ProjectionT>;
  static constexpr int kDimension = ProjectionT::DesignVariableDimension;
  using vector_t = Eigen::Matrix<double, kDimension, 1>;
  using matrix_t = Eigen::Matrix<double, kDimension, kDimension>;

  ProjectionAnchorError(
      const ProjectionT* projection,
      const boost::shared_ptr<projection_dv_t>& projection_dv,
      const vector_t& anchor,
      const vector_t& prior_weight)
      : projection_(projection),
        projection_dv_(projection_dv),
        anchor_(anchor) {
    if (projection_ == nullptr || projection_dv_ == nullptr) {
      throw std::runtime_error("ProjectionAnchorError requires valid camera data.");
    }
    const matrix_t inverse_covariance = prior_weight.asDiagonal();
    parent_t::setInvR(inverse_covariance);
    parent_t::setDesignVariables(projection_dv_.get());
  }

 protected:
  double evaluateErrorImplementation() override {
    Eigen::MatrixXd parameters_matrix;
    projection_->getParameters(parameters_matrix);
    vector_t parameters = parameters_matrix;
    parent_t::setError(parameters - anchor_);
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    jacobians.add(projection_dv_.get(), matrix_t::Identity());
  }

 private:
  using parent_t =
      aslam::backend::ErrorTermFs<ProjectionT::DesignVariableDimension>;

  const ProjectionT* projection_;
  boost::shared_ptr<projection_dv_t> projection_dv_;
  vector_t anchor_;
};

class EquidistantDistortionAnchorError
    : public aslam::backend::ErrorTermFs<aslam::cameras::EquidistantDistortion::DesignVariableDimension> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using distortion_t = aslam::cameras::EquidistantDistortion;
  using distortion_dv_t = aslam::backend::DesignVariableAdapter<distortion_t>;
  static constexpr int kDimension = distortion_t::DesignVariableDimension;
  using vector_t = Eigen::Matrix<double, kDimension, 1>;
  using matrix_t = Eigen::Matrix<double, kDimension, kDimension>;

  EquidistantDistortionAnchorError(
      const distortion_t* distortion,
      const boost::shared_ptr<distortion_dv_t>& distortion_dv,
      const vector_t& anchor,
      const vector_t& prior_weight)
      : distortion_(distortion),
        distortion_dv_(distortion_dv),
        anchor_(anchor) {
    if (distortion_ == nullptr || distortion_dv_ == nullptr) {
      throw std::runtime_error(
          "EquidistantDistortionAnchorError requires valid camera data.");
    }
    const matrix_t inverse_covariance = prior_weight.asDiagonal();
    parent_t::setInvR(inverse_covariance);
    parent_t::setDesignVariables(distortion_dv_.get());
  }

 protected:
  double evaluateErrorImplementation() override {
    Eigen::MatrixXd parameters_matrix;
    distortion_->getParameters(parameters_matrix);
    vector_t parameters = parameters_matrix;
    parent_t::setError(parameters - anchor_);
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    jacobians.add(distortion_dv_.get(), matrix_t::Identity());
  }

 private:
  using parent_t =
      aslam::backend::ErrorTermFs<aslam::cameras::EquidistantDistortion::DesignVariableDimension>;

  const distortion_t* distortion_;
  boost::shared_ptr<distortion_dv_t> distortion_dv_;
  vector_t anchor_;
};

class BoardPoseAnchorError : public aslam::backend::ErrorTermFs<6> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BoardPoseAnchorError(
      const boost::shared_ptr<aslam::backend::MappedRotationQuaternion>& rotation_dv,
      const boost::shared_ptr<aslam::backend::MappedEuclideanPoint>& translation_dv,
      const Eigen::Vector4d& rotation_anchor,
      const Eigen::Vector3d& translation_anchor,
      double translation_sigma_mm,
      double rotation_sigma_deg)
      : rotation_dv_(rotation_dv),
        translation_dv_(translation_dv),
        rotation_anchor_(rotation_anchor),
        translation_anchor_(translation_anchor) {
    if (rotation_dv_ == nullptr || translation_dv_ == nullptr) {
      throw std::runtime_error("BoardPoseAnchorError requires valid board pose DVs.");
    }
    const double sigma_t_m =
        std::max(1e-9, translation_sigma_mm * 1.0e-3);
    const double kPi = 3.141592653589793238462643383279502884;
    const double sigma_r_rad =
        std::max(1e-12, rotation_sigma_deg * kPi / 180.0);
    Eigen::Matrix<double, 6, 6> inverse_covariance =
        Eigen::Matrix<double, 6, 6>::Zero();
    inverse_covariance.block<3, 3>(0, 0) =
        (1.0 / (sigma_t_m * sigma_t_m)) * Eigen::Matrix3d::Identity();
    inverse_covariance.block<3, 3>(3, 3) =
        (1.0 / (sigma_r_rad * sigma_r_rad)) * Eigen::Matrix3d::Identity();
    parent_t::setInvR(inverse_covariance);
    parent_t::setDesignVariables(translation_dv_.get(), rotation_dv_.get());
  }

 protected:
  double evaluateErrorImplementation() override {
    Eigen::Matrix<double, 6, 1> error;
    error.head<3>() = CurrentTranslation() - translation_anchor_;
    error.tail<3>() = sm::kinematics::qlog(
        sm::kinematics::qplus(sm::kinematics::quatInv(rotation_anchor_),
                              CurrentRotation()));
    parent_t::setError(error);
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    Eigen::Matrix<double, 6, 3> translation_chain =
        Eigen::Matrix<double, 6, 3>::Zero();
    translation_chain.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 3> rotation_chain =
        Eigen::Matrix<double, 6, 3>::Zero();
    rotation_chain.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();
    jacobians.add(translation_dv_.get(), translation_chain);
    jacobians.add(rotation_dv_.get(), rotation_chain);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<6>;

  Eigen::Vector3d CurrentTranslation() const {
    Eigen::MatrixXd parameters;
    translation_dv_->getParameters(parameters);
    return Eigen::Vector3d(parameters(0, 0), parameters(1, 0), parameters(2, 0));
  }

  Eigen::Vector4d CurrentRotation() const {
    Eigen::MatrixXd parameters;
    rotation_dv_->getParameters(parameters);
    return Eigen::Vector4d(parameters(0, 0), parameters(1, 0),
                           parameters(2, 0), parameters(3, 0));
  }

  boost::shared_ptr<aslam::backend::MappedRotationQuaternion> rotation_dv_;
  boost::shared_ptr<aslam::backend::MappedEuclideanPoint> translation_dv_;
  Eigen::Vector4d rotation_anchor_ = Eigen::Vector4d::Zero();
  Eigen::Vector3d translation_anchor_ = Eigen::Vector3d::Zero();
};

template <typename GeometryT>
class CameraReprojectionError : public aslam::backend::ErrorTermFs<2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using camera_dv_t = aslam::backend::CameraDesignVariable<GeometryT>;
  using measurement_t = Eigen::Vector2d;
  using inverse_covariance_t = Eigen::Matrix2d;

  CameraReprojectionError(const measurement_t& measurement,
                          const inverse_covariance_t& inverse_covariance,
                          double huber_delta_pixels,
                          bool use_huber_loss,
                          const aslam::backend::HomogeneousExpression& point_camera,
                          const camera_dv_t& camera_dv,
                          double invalid_projection_penalty_pixels)
      : measurement_(measurement),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        inverse_covariance_(inverse_covariance.cwiseMax(0.0)),
        balance_weight_(
            0.5 * (inverse_covariance_(0, 0) + inverse_covariance_(1, 1))),
        huber_delta_pixels_(huber_delta_pixels),
        invalid_projection_penalty_pixels_(invalid_projection_penalty_pixels) {
    parent_t::setInvR(inverse_covariance_);
    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    camera_dv_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(), design_variables.end());

    if (use_huber_loss && huber_delta_pixels_ > 0.0 && balance_weight_ > 0.0) {
      const double scaled_delta = std::sqrt(balance_weight_) * huber_delta_pixels_;
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(scaled_delta)));
    }
  }

  ReprojectionDebugSample BuildDebugSample() const {
    ReprojectionDebugSample sample;
    sample.backend_inv_r_scale = balance_weight_;

    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    bool valid_projection = false;
    Eigen::Vector2d residual = ComputeResidual(&predicted, &valid_projection);
    sample.valid_projection = valid_projection;
    sample.predicted_image_xy = predicted;
    sample.residual_xy = residual;
    sample.residual_norm = residual.norm();
    sample.backend_raw_squared_error =
        residual.transpose() * inverse_covariance_ * residual;
    sample.backend_m_estimator_weight =
        parent_t::getMEstimatorWeight(sample.backend_raw_squared_error);
    sample.backend_weighted_squared_error =
        sample.backend_m_estimator_weight * sample.backend_raw_squared_error;
    return sample;
  }

 protected:
  double evaluateErrorImplementation() override {
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    bool valid_projection = false;
    parent_t::setError(ComputeResidual(&predicted, &valid_projection));
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    typename GeometryT::jacobian_homogeneous_t projection_jacobian;
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    const bool valid_projection =
        camera_dv_.camera()->homogeneousToKeypoint(point_homogeneous, predicted,
                                                   projection_jacobian) &&
        predicted.allFinite() && projection_jacobian.allFinite();
    if (!valid_projection) {
      return;
    }

    // Residual is defined as measurement - predicted, so the projection Jacobians
    // enter with a negative sign for both the pose chain and camera intrinsics.
    point_camera_.evaluateJacobians(jacobians, -projection_jacobian);
    camera_dv_.evaluateJacobians(jacobians, point_homogeneous);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<2>;

  Eigen::Vector2d ComputeResidual(Eigen::Vector2d* predicted,
                                  bool* valid_projection) const {
    if (predicted == nullptr || valid_projection == nullptr) {
      throw std::runtime_error("ComputeResidual requires valid output pointers.");
    }
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    *predicted = Eigen::Vector2d::Zero();
    *valid_projection =
        camera_dv_.camera()->homogeneousToKeypoint(point_homogeneous, *predicted) &&
        predicted->allFinite();
    if (!(*valid_projection)) {
      *predicted = Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::quiet_NaN());
      return Eigen::Vector2d::Constant(invalid_projection_penalty_pixels_);
    }
    return measurement_ - *predicted;
  }

  measurement_t measurement_;
  aslam::backend::HomogeneousExpression point_camera_;
  camera_dv_t camera_dv_;
  inverse_covariance_t inverse_covariance_ = inverse_covariance_t::Identity();
  double balance_weight_ = 1.0;
  double huber_delta_pixels_ = 0.0;
  double invalid_projection_penalty_pixels_ = 100.0;
};

template <typename GeometryT>
class CameraAngularReprojectionError : public aslam::backend::ErrorTermFs<2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using camera_dv_t = aslam::backend::CameraDesignVariable<GeometryT>;
  using measurement_t = Eigen::Vector2d;
  using inverse_covariance_t = Eigen::Matrix2d;

  CameraAngularReprojectionError(
      const Eigen::Vector2d& observed_image_xy,
      const inverse_covariance_t& inverse_covariance,
      double huber_delta_radians,
      bool use_huber_loss,
      const aslam::backend::HomogeneousExpression& point_camera,
      const camera_dv_t& camera_dv,
      double invalid_projection_penalty_radians,
      bool use_normalize_jacobian = true,
      AslamBackendCalibrationOptions::AngularObservedRayMode observed_ray_mode =
          AslamBackendCalibrationOptions::AngularObservedRayMode::
              DynamicCurrentCamera,
      const AngularObservationGeometry& frozen_observation_geometry =
          AngularObservationGeometry())
      : observed_image_xy_(observed_image_xy),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        inverse_covariance_(inverse_covariance.cwiseMax(0.0)),
        balance_weight_(
            0.5 * (inverse_covariance_(0, 0) + inverse_covariance_(1, 1))),
        huber_delta_radians_(huber_delta_radians),
        invalid_projection_penalty_radians_(invalid_projection_penalty_radians),
        use_normalize_jacobian_(use_normalize_jacobian),
        observed_ray_mode_(observed_ray_mode),
        frozen_observation_geometry_(frozen_observation_geometry) {
    parent_t::setInvR(inverse_covariance_);
    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    camera_dv_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(), design_variables.end());

    if (use_huber_loss && huber_delta_radians_ > 0.0 && balance_weight_ > 0.0) {
      const double scaled_delta =
          std::sqrt(balance_weight_) * huber_delta_radians_;
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(scaled_delta)));
    }
  }

  ReprojectionDebugSample BuildDebugSample() const {
    ReprojectionDebugSample sample;
    sample.backend_inv_r_scale = balance_weight_;
    AngularPredictionGeometry predicted_geometry;
    AngularObservationGeometry observation_geometry;
    bool valid_projection = false;
    const Eigen::Vector2d residual =
        ComputeResidual(&observation_geometry, &predicted_geometry, &valid_projection);
    sample.valid_projection = valid_projection;
    sample.predicted_image_xy = predicted_geometry.predicted_image_xy;
    sample.residual_xy = residual;
    sample.residual_norm = residual.norm();
    sample.backend_raw_squared_error =
        residual.transpose() * inverse_covariance_ * residual;
    sample.backend_m_estimator_weight =
        parent_t::getMEstimatorWeight(sample.backend_raw_squared_error);
    sample.backend_weighted_squared_error =
        sample.backend_m_estimator_weight * sample.backend_raw_squared_error;
    return sample;
  }

 protected:
  double evaluateErrorImplementation() override {
    AngularObservationGeometry observation_geometry;
    AngularPredictionGeometry predicted_geometry;
    bool valid_projection = false;
    parent_t::setError(
        ComputeResidual(&observation_geometry, &predicted_geometry, &valid_projection));
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    AngularObservationGeometry observation_geometry;
    AngularPredictionGeometry predicted_geometry;
    bool valid_projection = false;
    const Eigen::Vector2d residual =
        ComputeResidual(&observation_geometry, &predicted_geometry, &valid_projection);
    if (!valid_projection || !observation_geometry.success) {
      return;
    }

    const Eigen::Vector3d point_camera = point_homogeneous.head<3>();
    const double point_norm = point_camera.norm();
    if (!(point_norm > 1e-12) || !std::isfinite(point_norm)) {
      return;
    }
    const Eigen::Vector3d unit_ray = point_camera / point_norm;
    const Eigen::Matrix3d d_unit_d_point =
        (Eigen::Matrix3d::Identity() - unit_ray * unit_ray.transpose()) /
        point_norm;

    const Eigen::Matrix<double, 2, 3> tangent_t =
        observation_geometry.tangent_basis.transpose();
    const Eigen::Matrix<double, 2, 3> d_residual_d_point =
        tangent_t * d_unit_d_point;
    const Eigen::Matrix<double, 2, 4> d_residual_d_homogeneous =
        (Eigen::Matrix<double, 2, 4>() <<
             d_residual_d_point(0, 0), d_residual_d_point(0, 1),
             d_residual_d_point(0, 2), 0.0,
             d_residual_d_point(1, 0), d_residual_d_point(1, 1),
             d_residual_d_point(1, 2), 0.0)
            .finished();

    // Angular residual is defined as predicted_ray - observed_ray in the
    // observed tangent plane, so the pose/board point chain enters with a
    // positive sign. This differs from the image-plane residual above, which is
    // measurement - predicted and therefore uses a negative projection Jacobian.
    point_camera_.evaluateJacobians(jacobians, d_residual_d_homogeneous);

    // In frozen-anchor mode the measured ray is a fixed datum, not a function of
    // the current intrinsics.  We therefore do not add the dynamic camera
    // finite-difference Jacobian that B0 uses through unproject(current_camera).
    if (observed_ray_mode_ ==
        AslamBackendCalibrationOptions::AngularObservedRayMode::
            FrozenAnchorCamera) {
      return;
    }

    const GeometryT& base_camera = *camera_dv_.camera();
    const auto evaluate_residual =
        [this](const GeometryT& camera, bool* valid) {
          AngularObservationGeometry observation;
          AngularPredictionGeometry prediction;
          return ComputeResidualForCamera(camera, &observation, &prediction,
                                          valid);
        };
    AddThreadSafeCameraFiniteDifferenceJacobian<2>(
        const_cast<camera_dv_t&>(camera_dv_).projectionDesignVariable(),
        base_camera, CameraParameterBlock::kProjection, evaluate_residual,
        &jacobians);
    AddThreadSafeCameraFiniteDifferenceJacobian<2>(
        const_cast<camera_dv_t&>(camera_dv_).distortionDesignVariable(),
        base_camera, CameraParameterBlock::kDistortion, evaluate_residual,
        &jacobians);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<2>;

  Eigen::Vector2d ComputeResidual(AngularObservationGeometry* observation_geometry,
                                  AngularPredictionGeometry* predicted_geometry,
                                  bool* valid_projection) const {
    return ComputeResidualForCamera(*camera_dv_.camera(), observation_geometry,
                                    predicted_geometry, valid_projection);
  }

  Eigen::Vector2d ComputeResidualForCamera(
      const GeometryT& camera,
      AngularObservationGeometry* observation_geometry,
      AngularPredictionGeometry* predicted_geometry,
      bool* valid_projection) const {
    if (observation_geometry == nullptr || predicted_geometry == nullptr ||
        valid_projection == nullptr) {
      throw std::runtime_error(
          "CameraAngularReprojectionError requires valid output pointers.");
    }
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    bool observation_valid = false;
    if (observed_ray_mode_ ==
        AslamBackendCalibrationOptions::AngularObservedRayMode::
            FrozenAnchorCamera) {
      *observation_geometry = frozen_observation_geometry_;
      observation_valid = observation_geometry->success;
    } else {
      observation_valid = ComputeObservationGeometryForCamera(
          camera, observed_image_xy_, observation_geometry);
    }
    *valid_projection = observation_valid &&
        ComputePredictionGeometryForCamera(camera, point_homogeneous,
                                           predicted_geometry);
    if (!(*valid_projection) || !observation_geometry->success) {
      return Eigen::Vector2d::Constant(invalid_projection_penalty_radians_);
    }
    return ComputeAngularResidualTangent(*observation_geometry, *predicted_geometry);
  }

  Eigen::Vector2d observed_image_xy_ = Eigen::Vector2d::Zero();
  aslam::backend::HomogeneousExpression point_camera_;
  camera_dv_t camera_dv_;
  inverse_covariance_t inverse_covariance_ = inverse_covariance_t::Identity();
  double balance_weight_ = 1.0;
  double huber_delta_radians_ = 0.0;
  double invalid_projection_penalty_radians_ = 0.35;
  bool use_normalize_jacobian_ = true;
  AslamBackendCalibrationOptions::AngularObservedRayMode observed_ray_mode_ =
      AslamBackendCalibrationOptions::AngularObservedRayMode::DynamicCurrentCamera;
  AngularObservationGeometry frozen_observation_geometry_;
};

template <typename GeometryT>
class CameraPixelRayHybridReprojectionError
    : public aslam::backend::ErrorTermFs<4> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using camera_dv_t = aslam::backend::CameraDesignVariable<GeometryT>;
  using inverse_covariance_t = Eigen::Matrix2d;

  CameraPixelRayHybridReprojectionError(
      const Eigen::Vector2d& observed_image_xy,
      const inverse_covariance_t& inverse_covariance,
      double lambda,
      double huber_delta,
      bool use_huber_loss,
      const aslam::backend::HomogeneousExpression& point_camera,
      const camera_dv_t& camera_dv,
      double invalid_projection_penalty_pixels,
      double invalid_projection_penalty_radians)
      : observed_image_xy_(observed_image_xy),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        inverse_covariance_(inverse_covariance.cwiseMax(0.0)),
        balance_weight_(
            0.5 * (inverse_covariance_(0, 0) + inverse_covariance_(1, 1))),
        lambda_(lambda),
        huber_delta_(huber_delta),
        invalid_projection_penalty_pixels_(invalid_projection_penalty_pixels),
        invalid_projection_penalty_radians_(
            invalid_projection_penalty_radians) {
    Eigen::Matrix4d hybrid_inverse_covariance = Eigen::Matrix4d::Zero();
    hybrid_inverse_covariance.topLeftCorner<2, 2>() = inverse_covariance_;
    hybrid_inverse_covariance.bottomRightCorner<2, 2>() = inverse_covariance_;
    parent_t::setInvR(hybrid_inverse_covariance);

    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    camera_dv_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(),
                                         design_variables.end());

    if (use_huber_loss && huber_delta_ > 0.0 && balance_weight_ > 0.0) {
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(
                  std::sqrt(balance_weight_) * huber_delta_)));
    }
  }

  bool BuildRawComponents(Eigen::Vector2d* pixel_residual,
                          Eigen::Vector2d* ray_residual) const {
    return ComputeRawComponentsForCamera(*camera_dv_.camera(), pixel_residual,
                                         ray_residual);
  }

  void SetFixedScales(double pixel_scale, double ray_scale) {
    if (!(pixel_scale > 0.0) || !(ray_scale > 0.0) ||
        !std::isfinite(pixel_scale) || !std::isfinite(ray_scale)) {
      throw std::runtime_error(
          "Pixel-ray hybrid scales must be finite and positive.");
    }
    pixel_scale_ = pixel_scale;
    ray_scale_ = ray_scale;
  }

 protected:
  double evaluateErrorImplementation() override {
    parent_t::setError(ComputeScaledResidualForCamera(*camera_dv_.camera()));
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    Eigen::Vector2d pixel_residual = Eigen::Vector2d::Zero();
    Eigen::Vector2d ray_residual = Eigen::Vector2d::Zero();
    if (!ComputeRawComponentsForCamera(*camera_dv_.camera(), &pixel_residual,
                                       &ray_residual)) {
      return;
    }

    typename GeometryT::jacobian_homogeneous_t projection_jacobian;
    Eigen::Vector2d predicted_image = Eigen::Vector2d::Zero();
    if (!camera_dv_.camera()->homogeneousToKeypoint(
            point_homogeneous, predicted_image, projection_jacobian) ||
        !predicted_image.allFinite() || !projection_jacobian.allFinite()) {
      return;
    }

    AngularObservationGeometry observation_geometry;
    if (!ComputeObservationGeometryForCamera(*camera_dv_.camera(),
                                             observed_image_xy_,
                                             &observation_geometry)) {
      return;
    }
    const Eigen::Vector3d point = point_homogeneous.head<3>();
    const double point_norm = point.norm();
    if (!(point_norm > 1e-12) || !std::isfinite(point_norm)) {
      return;
    }
    const Eigen::Vector3d unit_ray = point / point_norm;
    const Eigen::Matrix3d d_unit_d_point =
        (Eigen::Matrix3d::Identity() - unit_ray * unit_ray.transpose()) /
        point_norm;

    Eigen::Matrix<double, 4, 4> d_residual_d_homogeneous =
        Eigen::Matrix<double, 4, 4>::Zero();
    d_residual_d_homogeneous.topRows<2>() =
        pixel_weight() * projection_jacobian;
    d_residual_d_homogeneous.block<2, 3>(2, 0) =
        ray_weight() * observation_geometry.tangent_basis.transpose() *
        d_unit_d_point;
    point_camera_.evaluateJacobians(jacobians,
                                    d_residual_d_homogeneous);

    const GeometryT& base_camera = *camera_dv_.camera();
    const auto evaluate_residual = [this](const GeometryT& camera,
                                          bool* valid) {
      Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
      Eigen::Vector2d ray = Eigen::Vector2d::Zero();
      *valid = ComputeRawComponentsForCamera(camera, &pixel, &ray);
      return *valid ? BuildScaledResidual(pixel, ray)
                    : InvalidScaledResidual();
    };
    AddThreadSafeCameraFiniteDifferenceJacobian<4>(
        const_cast<camera_dv_t&>(camera_dv_).projectionDesignVariable(),
        base_camera, CameraParameterBlock::kProjection, evaluate_residual,
        &jacobians);
    AddThreadSafeCameraFiniteDifferenceJacobian<4>(
        const_cast<camera_dv_t&>(camera_dv_).distortionDesignVariable(),
        base_camera, CameraParameterBlock::kDistortion, evaluate_residual,
        &jacobians);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<4>;

  double pixel_weight() const {
    return std::sqrt(std::max(0.0, 1.0 - lambda_)) / pixel_scale_;
  }

  double ray_weight() const {
    return std::sqrt(std::max(0.0, lambda_)) / ray_scale_;
  }

  bool ComputeRawComponentsForCamera(const GeometryT& camera,
                                     Eigen::Vector2d* pixel_residual,
                                     Eigen::Vector2d* ray_residual) const {
    if (pixel_residual == nullptr || ray_residual == nullptr) {
      throw std::runtime_error(
          "CameraPixelRayHybridReprojectionError requires output pointers.");
    }
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    Eigen::Vector2d predicted_image = Eigen::Vector2d::Zero();
    const bool image_valid = camera.homogeneousToKeypoint(
                                 point_homogeneous, predicted_image) &&
                             predicted_image.allFinite();
    AngularObservationGeometry observation_geometry;
    AngularPredictionGeometry prediction_geometry;
    const bool ray_valid = ComputeObservationGeometryForCamera(
                               camera, observed_image_xy_,
                               &observation_geometry) &&
                           ComputePredictionGeometryForCamera(
                               camera, point_homogeneous,
                               &prediction_geometry);
    if (!image_valid || !ray_valid || !observation_geometry.success ||
        !prediction_geometry.valid_projection) {
      return false;
    }
    // The paper convention is prediction minus observation for both blocks.
    *pixel_residual = predicted_image - observed_image_xy_;
    *ray_residual = ComputeAngularResidualTangent(observation_geometry,
                                                  prediction_geometry);
    return pixel_residual->allFinite() && ray_residual->allFinite();
  }

  Eigen::Vector4d BuildScaledResidual(
      const Eigen::Vector2d& pixel_residual,
      const Eigen::Vector2d& ray_residual) const {
    Eigen::Vector4d residual;
    residual.head<2>() = pixel_weight() * pixel_residual;
    residual.tail<2>() = ray_weight() * ray_residual;
    return residual;
  }

  Eigen::Vector4d InvalidScaledResidual() const {
    Eigen::Vector4d residual;
    residual.head<2>().setConstant(pixel_weight() *
                                   invalid_projection_penalty_pixels_);
    residual.tail<2>().setConstant(ray_weight() *
                                   invalid_projection_penalty_radians_);
    return residual;
  }

  Eigen::Vector4d ComputeScaledResidualForCamera(const GeometryT& camera) const {
    Eigen::Vector2d pixel_residual = Eigen::Vector2d::Zero();
    Eigen::Vector2d ray_residual = Eigen::Vector2d::Zero();
    if (!ComputeRawComponentsForCamera(camera, &pixel_residual,
                                       &ray_residual)) {
      return InvalidScaledResidual();
    }
    return BuildScaledResidual(pixel_residual, ray_residual);
  }

  Eigen::Vector2d observed_image_xy_ = Eigen::Vector2d::Zero();
  aslam::backend::HomogeneousExpression point_camera_;
  camera_dv_t camera_dv_;
  inverse_covariance_t inverse_covariance_ = inverse_covariance_t::Identity();
  double balance_weight_ = 1.0;
  double lambda_ = 0.5;
  double pixel_scale_ = 1.0;
  double ray_scale_ = 1.0;
  double huber_delta_ = 3.0;
  double invalid_projection_penalty_pixels_ = 100.0;
  double invalid_projection_penalty_radians_ = 0.35;
};

double ComputeFixedPolarAdaptivePixelRayLambda(
    const AslamBackendCalibrationOptions& options,
    double polar_angle_deg) {
  if (!options.pixel_ray_hybrid_polar_adaptive_enabled) {
    return options.pixel_ray_hybrid_lambda;
  }
  const double normalized = std::max(
      0.0, std::min(1.0,
          (polar_angle_deg - options.pixel_ray_hybrid_transition_start_deg) /
              (options.pixel_ray_hybrid_transition_end_deg -
               options.pixel_ray_hybrid_transition_start_deg)));
  // Cubic smoothstep provides a monotone transition with zero slope at both
  // endpoints, without introducing a hard residual-model switch.
  const double smoothstep = normalized * normalized * (3.0 - 2.0 * normalized);
  return std::max(
      0.0, std::min(1.0,
          options.pixel_ray_hybrid_lambda_min +
              (options.pixel_ray_hybrid_lambda_max -
               options.pixel_ray_hybrid_lambda_min) * smoothstep));
}

template <typename GeometryT>
class CameraChordalReprojectionError : public aslam::backend::ErrorTermFs<3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using camera_dv_t = aslam::backend::CameraDesignVariable<GeometryT>;
  using inverse_covariance_t = Eigen::Matrix3d;

  CameraChordalReprojectionError(
      const Eigen::Vector2d& observed_image_xy,
      const inverse_covariance_t& inverse_covariance,
      double huber_delta_chordal,
      bool use_huber_loss,
      const aslam::backend::HomogeneousExpression& point_camera,
      const camera_dv_t& camera_dv,
      double invalid_projection_penalty_chordal,
      bool use_normalize_jacobian = true,
      AslamBackendCalibrationOptions::AngularObservedRayMode observed_ray_mode =
          AslamBackendCalibrationOptions::AngularObservedRayMode::
              DynamicCurrentCamera,
      const AngularObservationGeometry& frozen_observation_geometry =
          AngularObservationGeometry())
      : observed_image_xy_(observed_image_xy),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        inverse_covariance_(inverse_covariance.cwiseMax(0.0)),
        balance_weight_((inverse_covariance_(0, 0) +
                         inverse_covariance_(1, 1) +
                         inverse_covariance_(2, 2)) /
                        3.0),
        huber_delta_chordal_(huber_delta_chordal),
        invalid_projection_penalty_chordal_(invalid_projection_penalty_chordal),
        use_normalize_jacobian_(use_normalize_jacobian),
        observed_ray_mode_(observed_ray_mode),
        frozen_observation_geometry_(frozen_observation_geometry) {
    parent_t::setInvR(inverse_covariance_);
    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    camera_dv_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(),
                                         design_variables.end());

    if (use_huber_loss && huber_delta_chordal_ > 0.0 &&
        balance_weight_ > 0.0) {
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(
                  std::sqrt(balance_weight_) * huber_delta_chordal_)));
    }
  }

 protected:
  double evaluateErrorImplementation() override {
    AngularObservationGeometry observation_geometry;
    AngularPredictionGeometry prediction_geometry;
    bool valid_projection = false;
    parent_t::setError(
        ComputeResidual(&observation_geometry, &prediction_geometry,
                        &valid_projection));
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    AngularObservationGeometry observation_geometry;
    AngularPredictionGeometry prediction_geometry;
    bool valid_projection = false;
    const Eigen::Matrix<double, 3, 1> residual =
        ComputeResidual(&observation_geometry, &prediction_geometry,
                        &valid_projection);
    (void)residual;
    if (!valid_projection || !observation_geometry.success ||
        !prediction_geometry.valid_projection) {
      return;
    }

    const Eigen::Vector3d point_camera = point_homogeneous.head<3>();
    const double point_norm = point_camera.norm();
    if (!(point_norm > 1e-12) || !std::isfinite(point_norm)) {
      return;
    }
    const Eigen::Vector3d unit_ray = point_camera / point_norm;
    const Eigen::Matrix3d d_unit_d_point =
        (Eigen::Matrix3d::Identity() - unit_ray * unit_ray.transpose()) /
        point_norm;
    const Eigen::Matrix<double, 3, 4> d_residual_d_homogeneous =
        (Eigen::Matrix<double, 3, 4>() <<
             d_unit_d_point(0, 0), d_unit_d_point(0, 1),
             d_unit_d_point(0, 2), 0.0,
             d_unit_d_point(1, 0), d_unit_d_point(1, 1),
             d_unit_d_point(1, 2), 0.0,
             d_unit_d_point(2, 0), d_unit_d_point(2, 1),
             d_unit_d_point(2, 2), 0.0)
            .finished();
    point_camera_.evaluateJacobians(jacobians, d_residual_d_homogeneous);

    if (observed_ray_mode_ ==
        AslamBackendCalibrationOptions::AngularObservedRayMode::
            FrozenAnchorCamera) {
      return;
    }

    const GeometryT& base_camera = *camera_dv_.camera();
    const auto evaluate_residual =
        [this](const GeometryT& camera, bool* valid) {
          AngularObservationGeometry observation;
          AngularPredictionGeometry prediction;
          return ComputeResidualForCamera(camera, &observation, &prediction,
                                          valid);
        };
    AddThreadSafeCameraFiniteDifferenceJacobian<3>(
        const_cast<camera_dv_t&>(camera_dv_).projectionDesignVariable(),
        base_camera, CameraParameterBlock::kProjection, evaluate_residual,
        &jacobians);
    AddThreadSafeCameraFiniteDifferenceJacobian<3>(
        const_cast<camera_dv_t&>(camera_dv_).distortionDesignVariable(),
        base_camera, CameraParameterBlock::kDistortion, evaluate_residual,
        &jacobians);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<3>;

  Eigen::Matrix<double, 3, 1> ComputeResidual(
      AngularObservationGeometry* observation_geometry,
      AngularPredictionGeometry* prediction_geometry,
      bool* valid_projection) const {
    return ComputeResidualForCamera(*camera_dv_.camera(), observation_geometry,
                                    prediction_geometry, valid_projection);
  }

  Eigen::Matrix<double, 3, 1> ComputeResidualForCamera(
      const GeometryT& camera,
      AngularObservationGeometry* observation_geometry,
      AngularPredictionGeometry* prediction_geometry,
      bool* valid_projection) const {
    if (observation_geometry == nullptr || prediction_geometry == nullptr ||
        valid_projection == nullptr) {
      throw std::runtime_error(
          "CameraChordalReprojectionError requires valid output pointers.");
    }
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    bool observation_valid = false;
    if (observed_ray_mode_ ==
        AslamBackendCalibrationOptions::AngularObservedRayMode::
            FrozenAnchorCamera) {
      *observation_geometry = frozen_observation_geometry_;
      observation_valid = observation_geometry->success;
    } else {
      observation_valid = ComputeObservationGeometryForCamera(
          camera, observed_image_xy_, observation_geometry);
    }
    *valid_projection = observation_valid &&
        ComputePredictionGeometryForCamera(camera, point_homogeneous,
                                           prediction_geometry);
    if (!(*valid_projection) || !observation_geometry->success ||
        !prediction_geometry->valid_projection ||
        !observation_geometry->observed_ray.allFinite() ||
        !prediction_geometry->predicted_ray.allFinite()) {
      return Eigen::Matrix<double, 3, 1>::Constant(
          invalid_projection_penalty_chordal_);
    }
    return prediction_geometry->predicted_ray - observation_geometry->observed_ray;
  }

  Eigen::Vector2d observed_image_xy_ = Eigen::Vector2d::Zero();
  aslam::backend::HomogeneousExpression point_camera_;
  camera_dv_t camera_dv_;
  inverse_covariance_t inverse_covariance_ = inverse_covariance_t::Identity();
  double balance_weight_ = 1.0;
  double huber_delta_chordal_ = 0.0;
  double invalid_projection_penalty_chordal_ = 0.35;
  bool use_normalize_jacobian_ = true;
  AslamBackendCalibrationOptions::AngularObservedRayMode observed_ray_mode_ =
      AslamBackendCalibrationOptions::AngularObservedRayMode::DynamicCurrentCamera;
  AngularObservationGeometry frozen_observation_geometry_;
};

template <typename GeometryT>
class CameraPolarContinuousHybridReprojectionError
    : public aslam::backend::ErrorTermFs<2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using camera_dv_t = aslam::backend::CameraDesignVariable<GeometryT>;
  using inverse_covariance_t = Eigen::Matrix2d;

  CameraPolarContinuousHybridReprojectionError(
      const Eigen::Vector2d& observed_image_xy,
      const inverse_covariance_t& inverse_covariance,
      double huber_delta_pixels,
      bool use_huber_loss,
      const aslam::backend::HomogeneousExpression& point_camera,
      const camera_dv_t& camera_dv,
      double invalid_projection_penalty_pixels,
      double invalid_projection_penalty_radians,
      bool use_normalize_jacobian,
      AslamBackendCalibrationOptions::AngularObservedRayMode observed_ray_mode,
      const AngularObservationGeometry& frozen_observation_geometry,
      double threshold_deg,
      double temperature_deg)
      : observed_image_xy_(observed_image_xy),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        inverse_covariance_(inverse_covariance.cwiseMax(0.0)),
        balance_weight_(
            0.5 * (inverse_covariance_(0, 0) + inverse_covariance_(1, 1))),
        invalid_projection_penalty_pixels_(invalid_projection_penalty_pixels),
        invalid_projection_penalty_radians_(invalid_projection_penalty_radians),
        use_normalize_jacobian_(use_normalize_jacobian),
        observed_ray_mode_(observed_ray_mode),
        frozen_observation_geometry_(frozen_observation_geometry),
        threshold_deg_(threshold_deg),
        temperature_deg_(temperature_deg) {
    parent_t::setInvR(inverse_covariance_);
    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    camera_dv_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(),
                                         design_variables.end());

    if (use_huber_loss && huber_delta_pixels > 0.0 && balance_weight_ > 0.0) {
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(
                  std::sqrt(balance_weight_) * huber_delta_pixels)));
    }
  }

  ReprojectionDebugSample BuildDebugSample() const {
    ReprojectionDebugSample sample;
    sample.backend_inv_r_scale = balance_weight_;
    HybridEvaluation evaluation;
    const Eigen::Vector2d residual = ComputeResidual(&evaluation);
    sample.valid_projection = evaluation.valid_projection;
    sample.predicted_image_xy = evaluation.predicted_image_xy;
    sample.residual_xy = residual;
    sample.residual_norm = residual.norm();
    sample.backend_raw_squared_error =
        residual.transpose() * inverse_covariance_ * residual;
    sample.backend_m_estimator_weight =
        parent_t::getMEstimatorWeight(sample.backend_raw_squared_error);
    sample.backend_weighted_squared_error =
        sample.backend_m_estimator_weight * sample.backend_raw_squared_error;
    return sample;
  }

 protected:
  double evaluateErrorImplementation() override {
    HybridEvaluation evaluation;
    parent_t::setError(ComputeResidual(&evaluation));
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    HybridEvaluation evaluation;
    const Eigen::Vector2d residual = ComputeResidual(&evaluation);
    (void)residual;
    if (!evaluation.valid_projection ||
        !evaluation.observation_geometry.success ||
        !evaluation.prediction_geometry.valid_projection) {
      return;
    }

    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    typename GeometryT::jacobian_homogeneous_t projection_jacobian;
    Eigen::Vector2d predicted_image = Eigen::Vector2d::Zero();
    const bool valid_image_jacobian =
        camera_dv_.camera()->homogeneousToKeypoint(
            point_homogeneous, predicted_image, projection_jacobian) &&
        predicted_image.allFinite() && projection_jacobian.allFinite();
    if (!valid_image_jacobian) {
      return;
    }

    const Eigen::Vector3d point_camera = point_homogeneous.head<3>();
    const double point_norm = point_camera.norm();
    if (!(point_norm > 1e-12) || !std::isfinite(point_norm)) {
      return;
    }
    const Eigen::Vector3d unit_ray = point_camera / point_norm;
    Eigen::Matrix3d d_unit_d_point = Eigen::Matrix3d::Identity();
    if (use_normalize_jacobian_) {
      d_unit_d_point =
          (Eigen::Matrix3d::Identity() - unit_ray * unit_ray.transpose()) /
          point_norm;
    }
    const Eigen::Matrix<double, 2, 3> d_angular_d_point =
        evaluation.observation_geometry.tangent_basis.transpose() *
        d_unit_d_point;
    const Eigen::Matrix<double, 2, 4> angular_jacobian =
        (Eigen::Matrix<double, 2, 4>() <<
             d_angular_d_point(0, 0), d_angular_d_point(0, 1),
             d_angular_d_point(0, 2), 0.0,
             d_angular_d_point(1, 0), d_angular_d_point(1, 1),
             d_angular_d_point(1, 2), 0.0)
            .finished();
    const Eigen::Matrix<double, 2, 4> hybrid_jacobian =
        (1.0 - evaluation.angular_weight) * projection_jacobian +
        evaluation.angular_weight * angular_jacobian;
    point_camera_.evaluateJacobians(jacobians, hybrid_jacobian);

    const GeometryT& base_camera = *camera_dv_.camera();
    const auto evaluate_residual =
        [this](const GeometryT& camera, bool* valid) {
          HybridEvaluation evaluation;
          const Eigen::Vector2d residual =
              ComputeResidualForCamera(camera, &evaluation);
          *valid = evaluation.valid_projection;
          return residual;
        };
    AddThreadSafeCameraFiniteDifferenceJacobian<2>(
        const_cast<camera_dv_t&>(camera_dv_).projectionDesignVariable(),
        base_camera, CameraParameterBlock::kProjection, evaluate_residual,
        &jacobians);
    AddThreadSafeCameraFiniteDifferenceJacobian<2>(
        const_cast<camera_dv_t&>(camera_dv_).distortionDesignVariable(),
        base_camera, CameraParameterBlock::kDistortion, evaluate_residual,
        &jacobians);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<2>;

  struct HybridEvaluation {
    bool valid_projection = false;
    double angular_weight = 0.0;
    Eigen::Vector2d predicted_image_xy = Eigen::Vector2d::Zero();
    AngularObservationGeometry observation_geometry;
    AngularPredictionGeometry prediction_geometry;
  };

  Eigen::Vector2d ComputeResidual(HybridEvaluation* evaluation) const {
    return ComputeResidualForCamera(*camera_dv_.camera(), evaluation);
  }

  Eigen::Vector2d ComputeResidualForCamera(
      const GeometryT& camera, HybridEvaluation* evaluation) const {
    if (evaluation == nullptr) {
      throw std::runtime_error(
          "CameraPolarContinuousHybridReprojectionError requires output.");
    }
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    bool observation_valid = false;
    if (observed_ray_mode_ ==
        AslamBackendCalibrationOptions::AngularObservedRayMode::
            FrozenAnchorCamera) {
      evaluation->observation_geometry = frozen_observation_geometry_;
      observation_valid = evaluation->observation_geometry.success;
    } else {
      observation_valid = ComputeObservationGeometryForCamera(
          camera, observed_image_xy_, &evaluation->observation_geometry);
    }
    const bool image_valid =
        camera.homogeneousToKeypoint(point_homogeneous,
                                     evaluation->predicted_image_xy) &&
        evaluation->predicted_image_xy.allFinite();
    const bool angular_valid = observation_valid &&
        ComputePredictionGeometryForCamera(camera, point_homogeneous,
                                           &evaluation->prediction_geometry);
    evaluation->valid_projection =
        image_valid && angular_valid &&
        evaluation->observation_geometry.success;
    if (!evaluation->valid_projection) {
      return Eigen::Vector2d::Constant(
          0.5 * (invalid_projection_penalty_pixels_ +
                 invalid_projection_penalty_radians_));
    }
    evaluation->angular_weight = ComputePolarContinuousAngularWeight(
        evaluation->observation_geometry.polar_angle_deg,
        threshold_deg_,
        temperature_deg_);
    const Eigen::Vector2d pixel_residual =
        evaluation->predicted_image_xy - observed_image_xy_;
    const Eigen::Vector2d angular_residual =
        ComputeAngularResidualTangent(evaluation->observation_geometry,
                                      evaluation->prediction_geometry);
    return (1.0 - evaluation->angular_weight) * pixel_residual +
           evaluation->angular_weight * angular_residual;
  }

  Eigen::Vector2d observed_image_xy_ = Eigen::Vector2d::Zero();
  aslam::backend::HomogeneousExpression point_camera_;
  camera_dv_t camera_dv_;
  inverse_covariance_t inverse_covariance_ = inverse_covariance_t::Identity();
  double balance_weight_ = 1.0;
  double invalid_projection_penalty_pixels_ = 100.0;
  double invalid_projection_penalty_radians_ = 0.35;
  bool use_normalize_jacobian_ = true;
  AslamBackendCalibrationOptions::AngularObservedRayMode observed_ray_mode_ =
      AslamBackendCalibrationOptions::AngularObservedRayMode::
          DynamicCurrentCamera;
  AngularObservationGeometry frozen_observation_geometry_;
  double threshold_deg_ = 50.0;
  double temperature_deg_ = 10.0;
};

template <typename GeometryT>
using CameraDv = aslam::backend::CameraDesignVariable<GeometryT>;

double EvaluateTotalProblemObjective(
    aslam::backend::OptimizationProblem* problem) {
  if (problem == nullptr) {
    throw std::runtime_error("EvaluateTotalProblemObjective requires a valid problem.");
  }
  double total_cost = 0.0;
  for (std::size_t index = 0; index < problem->numErrorTerms(); ++index) {
    total_cost += problem->errorTerm(index)->evaluateError();
  }
  return total_cost;
}

struct FrozenWeightErrorState {
  aslam::backend::ErrorTerm* error_term = nullptr;
  double frozen_m_estimator_weight = 1.0;
};

std::vector<FrozenWeightErrorState> CaptureFrozenWeightErrorStates(
    const std::set<aslam::backend::ErrorTerm*>& error_terms) {
  std::vector<FrozenWeightErrorState> frozen_states;
  frozen_states.reserve(error_terms.size());
  for (aslam::backend::ErrorTerm* error_term : error_terms) {
    if (error_term == nullptr) {
      continue;
    }
    error_term->evaluateError();
    FrozenWeightErrorState frozen_state;
    frozen_state.error_term = error_term;
    frozen_state.frozen_m_estimator_weight = error_term->getCurrentMEstimatorWeight();
    frozen_states.push_back(frozen_state);
  }
  return frozen_states;
}

double EvaluateFrozenWeightObjective(
    const std::vector<FrozenWeightErrorState>& frozen_states) {
  double total_cost = 0.0;
  for (const FrozenWeightErrorState& frozen_state : frozen_states) {
    if (frozen_state.error_term == nullptr) {
      continue;
    }
    frozen_state.error_term->evaluateError();
    total_cost +=
        frozen_state.frozen_m_estimator_weight *
        frozen_state.error_term->getRawSquaredError();
  }
  return total_cost;
}

struct DesignVariableIndexState {
  aslam::backend::DesignVariable* design_variable = nullptr;
  int block_index = -1;
  int column_base = -1;
};

struct BackendErrorTermInfluenceMetadata {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  JointPointType point_type = JointPointType::Outer;
  std::string residual_family;
};

std::vector<DesignVariableIndexState> CaptureAndAssignBlockIndices(
    const boost::shared_ptr<aslam::backend::OptimizationProblem>& problem) {
  if (problem == nullptr) {
    throw std::runtime_error("CaptureAndAssignBlockIndices requires a valid problem.");
  }
  std::vector<DesignVariableIndexState> states;
  states.reserve(problem->numDesignVariables());

  int next_block_index = 0;
  int next_column_base = 0;
  for (std::size_t index = 0; index < problem->numDesignVariables(); ++index) {
    aslam::backend::DesignVariable* design_variable = problem->designVariable(index);
    DesignVariableIndexState state;
    state.design_variable = design_variable;
    state.block_index = design_variable->blockIndex();
    state.column_base = design_variable->columnBase();
    states.push_back(state);

    if (design_variable->isActive()) {
      design_variable->setBlockIndex(next_block_index++);
      design_variable->setColumnBase(next_column_base);
      next_column_base += design_variable->minimalDimensions();
    } else {
      design_variable->setBlockIndex(-1);
      design_variable->setColumnBase(-1);
    }
  }
  return states;
}

void RestoreBlockIndices(const std::vector<DesignVariableIndexState>& states) {
  for (const DesignVariableIndexState& state : states) {
    if (state.design_variable == nullptr) {
      continue;
    }
    state.design_variable->setBlockIndex(state.block_index);
    state.design_variable->setColumnBase(state.column_base);
  }
}

AslamBackendJacobianBlockDiagnostics RunJacobianBlockCheck(
    const std::string& block_label,
    aslam::backend::DesignVariable* design_variable,
    const boost::shared_ptr<aslam::backend::OptimizationProblem>& problem,
    double finite_difference_epsilon,
    std::vector<std::string>* warnings) {
  AslamBackendJacobianBlockDiagnostics diagnostics;
  diagnostics.block_label = block_label;
  if (design_variable == nullptr) {
    AppendUniqueWarning(block_label + " Jacobian check skipped: null design variable.", warnings);
    return diagnostics;
  }

  diagnostics.dimension = design_variable->minimalDimensions();
  diagnostics.analytic_gradient.assign(
      static_cast<std::size_t>(diagnostics.dimension), 0.0);
  diagnostics.finite_difference_gradient.assign(
      static_cast<std::size_t>(diagnostics.dimension), 0.0);

  const bool original_active = design_variable->isActive();
  design_variable->setActive(true);
  const std::vector<DesignVariableIndexState> saved_indices =
      CaptureAndAssignBlockIndices(problem);

  std::set<aslam::backend::ErrorTerm*> attached_error_terms;
  problem->getErrors(design_variable, attached_error_terms);
  if (attached_error_terms.empty()) {
    AppendUniqueWarning(block_label + " Jacobian check skipped: no attached error terms.",
                        warnings);
    RestoreBlockIndices(saved_indices);
    design_variable->setActive(original_active);
    return diagnostics;
  }

  const std::vector<FrozenWeightErrorState> frozen_error_states =
      CaptureFrozenWeightErrorStates(attached_error_terms);

  Eigen::VectorXd analytic_gradient =
      Eigen::VectorXd::Zero(design_variable->minimalDimensions());
  for (aslam::backend::ErrorTerm* error_term : attached_error_terms) {
    error_term->evaluateError();
    aslam::backend::JacobianContainer jacobians(static_cast<int>(error_term->dimension()));
    error_term->getWeightedJacobians(jacobians, true);
    Eigen::VectorXd weighted_error;
    error_term->getWeightedError(weighted_error, true);
    for (aslam::backend::JacobianContainer::map_t::const_iterator it = jacobians.begin();
         it != jacobians.end(); ++it) {
      if (it->first == design_variable) {
        analytic_gradient += 2.0 * it->second.transpose() * weighted_error;
      }
    }
  }

  const int dimension = design_variable->minimalDimensions();
  for (int index = 0; index < dimension; ++index) {
    Eigen::VectorXd positive_step = Eigen::VectorXd::Zero(dimension);
    positive_step[index] = finite_difference_epsilon;
    design_variable->update(positive_step.data(), dimension);
    const double positive_cost = EvaluateFrozenWeightObjective(frozen_error_states);
    design_variable->revertUpdate();

    Eigen::VectorXd negative_step = Eigen::VectorXd::Zero(dimension);
    negative_step[index] = -finite_difference_epsilon;
    design_variable->update(negative_step.data(), dimension);
    const double negative_cost = EvaluateFrozenWeightObjective(frozen_error_states);
    design_variable->revertUpdate();

    const double finite_difference =
        (positive_cost - negative_cost) / (2.0 * finite_difference_epsilon);
    diagnostics.analytic_gradient[static_cast<std::size_t>(index)] =
        analytic_gradient[index];
    diagnostics.finite_difference_gradient[static_cast<std::size_t>(index)] =
        finite_difference;
    diagnostics.max_abs_difference = std::max(
        diagnostics.max_abs_difference,
        std::fabs(analytic_gradient[index] - finite_difference));
  }

  RestoreBlockIndices(saved_indices);
  design_variable->setActive(original_active);
  return diagnostics;
}

template <typename GeometryT>
AslamBackendJacobianDiagnostics RunJacobianDiagnostics(
    const boost::shared_ptr<aslam::backend::OptimizationProblem>& problem,
    CameraDv<GeometryT>* camera_dv,
    const std::map<int, PoseVariableState>& frame_variables,
    const std::map<int, PoseVariableState>& board_variables,
    double finite_difference_epsilon) {
  AslamBackendJacobianDiagnostics diagnostics;
  diagnostics.finite_difference_epsilon = finite_difference_epsilon;
  diagnostics.objective_model = "irls_frozen_weighted_cost";

  std::vector<std::string> warnings;
  if (!frame_variables.empty()) {
    const auto& frame_entry = *frame_variables.begin();
    diagnostics.block_diagnostics.push_back(RunJacobianBlockCheck(
        "frame_rotation_frame_" + std::to_string(frame_entry.first),
        frame_entry.second.rotation_dv.get(), problem,
        finite_difference_epsilon, &warnings));
    diagnostics.block_diagnostics.push_back(RunJacobianBlockCheck(
        "frame_translation_frame_" + std::to_string(frame_entry.first),
        frame_entry.second.translation_dv.get(), problem,
        finite_difference_epsilon, &warnings));
  } else {
    AppendUniqueWarning("Frame Jacobian check skipped: no frame pose variables.", &warnings);
  }

  if (!board_variables.empty()) {
    const auto& board_entry = *board_variables.begin();
    diagnostics.block_diagnostics.push_back(RunJacobianBlockCheck(
        "board_rotation_board_" + std::to_string(board_entry.first),
        board_entry.second.rotation_dv.get(), problem,
        finite_difference_epsilon, &warnings));
    diagnostics.block_diagnostics.push_back(RunJacobianBlockCheck(
        "board_translation_board_" + std::to_string(board_entry.first),
        board_entry.second.translation_dv.get(), problem,
        finite_difference_epsilon, &warnings));
  } else {
    AppendUniqueWarning("Board Jacobian check skipped: no non-reference board variables.",
                        &warnings);
  }

  if (camera_dv != nullptr) {
    diagnostics.block_diagnostics.push_back(RunJacobianBlockCheck(
        "camera_intrinsics",
        camera_dv->projectionDesignVariable().get(),
        problem,
        finite_difference_epsilon,
        &warnings));
  } else {
    AppendUniqueWarning("Camera Jacobian check skipped: null camera DV.", &warnings);
  }

  diagnostics.warnings = warnings;
  diagnostics.success = !diagnostics.block_diagnostics.empty();
  if (!diagnostics.success) {
    diagnostics.failure_reason = "No Jacobian blocks could be checked.";
  }
  return diagnostics;
}

std::string VariableScopeFromLabel(const std::string& label) {
  if (label.find("camera_intrinsics") == 0) {
    return "camera_model";
  }
  if (label.find("frame_") == 0) {
    return "T_camera_reference";
  }
  if (label.find("board_") == 0) {
    return "T_reference_board";
  }
  return "other";
}

double RegularizedSymmetricLogDet(const Eigen::MatrixXd& hessian) {
  if (hessian.rows() == 0 || hessian.cols() == 0 ||
      hessian.rows() != hessian.cols()) {
    return 0.0;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(hessian);
  if (solver.info() != Eigen::Success) {
    return 0.0;
  }
  double value = 0.0;
  for (int index = 0; index < solver.eigenvalues().rows(); ++index) {
    value += std::log1p(std::max(0.0, solver.eigenvalues()[index]));
  }
  return value;
}

double SymmetricRankProxy(const Eigen::MatrixXd& hessian) {
  if (hessian.rows() == 0 || hessian.cols() == 0 ||
      hessian.rows() != hessian.cols()) {
    return 0.0;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(hessian);
  if (solver.info() != Eigen::Success) {
    return 0.0;
  }
  double value = 0.0;
  for (int index = 0; index < solver.eigenvalues().rows(); ++index) {
    const double lambda = std::max(0.0, solver.eigenvalues()[index]);
    value += lambda / (lambda + 1.0);
  }
  return value;
}

struct VariableBlockInfluenceAccumulator {
  std::string stage_label;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  std::string point_type;
  std::string residual_family;
  std::string variable_block;
  std::string variable_scope;
  int residual_count = 0;
  int residual_dimension = 0;
  int jacobian_columns = 0;
  double weighted_cost = 0.0;
  Eigen::MatrixXd hessian;
  Eigen::VectorXd gradient;

  void Add(const Eigen::MatrixXd& jacobian,
           const Eigen::VectorXd& weighted_error,
           double error_cost) {
    if (jacobian.rows() <= 0 || jacobian.cols() <= 0) {
      return;
    }
    if (hessian.rows() == 0) {
      hessian = Eigen::MatrixXd::Zero(jacobian.cols(), jacobian.cols());
      gradient = Eigen::VectorXd::Zero(jacobian.cols());
      jacobian_columns = static_cast<int>(jacobian.cols());
    }
    if (hessian.rows() != jacobian.cols()) {
      return;
    }
    hessian += jacobian.transpose() * jacobian;
    if (weighted_error.rows() == jacobian.rows()) {
      gradient += jacobian.transpose() * weighted_error;
    }
    ++residual_count;
    residual_dimension += static_cast<int>(jacobian.rows());
    weighted_cost += error_cost;
  }

  AslamBackendVariableBlockInfluenceEntry ToEntry() const {
    AslamBackendVariableBlockInfluenceEntry entry;
    entry.stage_label = stage_label;
    entry.frame_index = frame_index;
    entry.frame_label = frame_label;
    entry.board_id = board_id;
    entry.point_type = point_type;
    entry.residual_family = residual_family;
    entry.variable_block = variable_block;
    entry.variable_scope = variable_scope;
    entry.residual_count = residual_count;
    entry.residual_dimension = residual_dimension;
    entry.jacobian_columns = jacobian_columns;
    entry.weighted_cost = weighted_cost;
    if (hessian.rows() > 0) {
      entry.hessian_trace = hessian.trace();
      entry.hessian_frobenius_norm = hessian.norm();
      entry.hessian_logdet = RegularizedSymmetricLogDet(hessian);
      entry.hessian_rank_proxy = SymmetricRankProxy(hessian);
    }
    if (gradient.rows() > 0) {
      entry.gradient_norm = gradient.norm();
    }
    return entry;
  }
};

std::string InfluenceAggregationKey(
    const std::string& stage_label,
    const BackendErrorTermInfluenceMetadata& metadata,
    const std::string& variable_block) {
  std::ostringstream stream;
  stream << stage_label << "|"
         << metadata.frame_index << "|"
         << metadata.frame_label << "|"
         << metadata.board_id << "|"
         << ToString(metadata.point_type) << "|"
         << metadata.residual_family << "|"
         << variable_block;
  return stream.str();
}

AslamBackendVariableBlockInfluenceDiagnostics
EvaluateVariableBlockInfluenceDiagnostics(
    const std::string& stage_label,
    const boost::shared_ptr<aslam::backend::OptimizationProblem>& problem,
    const std::vector<aslam::backend::ErrorTerm*>& error_terms,
    const std::vector<BackendErrorTermInfluenceMetadata>& metadata,
    const std::map<aslam::backend::DesignVariable*, std::string>&
        design_variable_labels) {
  AslamBackendVariableBlockInfluenceDiagnostics diagnostics;
  if (problem == nullptr) {
    diagnostics.failure_reason =
        "Variable block influence diagnostics require a valid problem.";
    return diagnostics;
  }
  if (error_terms.size() != metadata.size()) {
    diagnostics.failure_reason =
        "Variable block influence metadata count does not match error terms.";
    return diagnostics;
  }

  const std::vector<DesignVariableIndexState> saved_indices =
      CaptureAndAssignBlockIndices(problem);
  std::map<std::string, VariableBlockInfluenceAccumulator> accumulators;
  for (std::size_t index = 0; index < error_terms.size(); ++index) {
    aslam::backend::ErrorTerm* error_term = error_terms[index];
    if (error_term == nullptr) {
      continue;
    }
    error_term->evaluateError();
    Eigen::VectorXd weighted_error;
    error_term->getWeightedError(weighted_error, true);
    aslam::backend::JacobianContainer jacobians(
        static_cast<int>(error_term->dimension()));
    error_term->getWeightedJacobians(jacobians, true);
    const double weighted_cost =
        weighted_error.rows() > 0 ? weighted_error.squaredNorm()
                                  : error_term->getRawSquaredError();
    for (aslam::backend::JacobianContainer::map_t::const_iterator it =
             jacobians.begin();
         it != jacobians.end(); ++it) {
      if (it->first == nullptr || !it->first->isActive()) {
        continue;
      }
      std::string variable_block = "unknown";
      const auto label_it = design_variable_labels.find(it->first);
      if (label_it != design_variable_labels.end()) {
        variable_block = label_it->second;
      } else {
        const int block_index = it->first->blockIndex();
        if (block_index >= 0) {
        variable_block = "block_" + std::to_string(block_index);
        }
      }
      const std::string key =
          InfluenceAggregationKey(stage_label, metadata[index], variable_block);
      VariableBlockInfluenceAccumulator& acc = accumulators[key];
      if (acc.residual_count == 0) {
        acc.stage_label = stage_label;
        acc.frame_index = metadata[index].frame_index;
        acc.frame_label = metadata[index].frame_label;
        acc.board_id = metadata[index].board_id;
        acc.point_type = ToString(metadata[index].point_type);
        acc.residual_family = metadata[index].residual_family;
        acc.variable_block = variable_block;
        acc.variable_scope = VariableScopeFromLabel(variable_block);
      }
      acc.Add(it->second, weighted_error, weighted_cost);
    }
  }
  RestoreBlockIndices(saved_indices);

  diagnostics.entries.reserve(accumulators.size());
  for (const auto& entry : accumulators) {
    diagnostics.entries.push_back(entry.second.ToEntry());
  }
  diagnostics.success = !diagnostics.entries.empty();
  if (!diagnostics.success) {
    diagnostics.failure_reason =
        "No active variable block Jacobians were collected.";
  }
  return diagnostics;
}

template <typename ReprojectionErrorT>
AslamBackendCostParityDiagnostics EvaluateCostParityDiagnostics(
    const std::string& stage_label,
    const boost::shared_ptr<aslam::backend::OptimizationProblem>& problem,
    const JointMeasurementBuildResult& measurement_result,
    const JointReprojectionSceneState& scene_state,
    const std::vector<boost::shared_ptr<ReprojectionErrorT> >& reprojection_errors,
    const JointReprojectionCostOptions& frontend_cost_options) {
  AslamBackendCostParityDiagnostics diagnostics;
  diagnostics.stage_label = stage_label;

  if (problem == nullptr) {
    diagnostics.failure_reason = "Parity diagnostics require a valid optimization problem.";
    return diagnostics;
  }

  const JointReprojectionCostCore frontend_cost_core(frontend_cost_options);
  const JointCostEvaluation frontend_evaluation =
      frontend_cost_core.Evaluate(measurement_result, scene_state);
  if (!frontend_evaluation.success) {
    diagnostics.failure_reason = frontend_evaluation.failure_reason;
    diagnostics.warnings = frontend_evaluation.warnings;
    return diagnostics;
  }

  if (frontend_evaluation.point_evaluations.size() != reprojection_errors.size()) {
    diagnostics.failure_reason =
        "Frontend/backend parity point count mismatch: frontend=" +
        std::to_string(frontend_evaluation.point_evaluations.size()) +
        " backend=" + std::to_string(reprojection_errors.size());
    return diagnostics;
  }

  diagnostics.frontend_total_squared_error = frontend_evaluation.total_squared_error;
  diagnostics.frontend_total_cost = frontend_evaluation.total_cost;
  diagnostics.backend_problem_total_weighted_cost =
      EvaluateTotalProblemObjective(problem.get());
  diagnostics.point_diagnostics.reserve(reprojection_errors.size());

  for (std::size_t index = 0; index < reprojection_errors.size(); ++index) {
    const JointCostPointEvaluation& frontend_point =
        frontend_evaluation.point_evaluations[index];
    const ReprojectionDebugSample backend_point =
        reprojection_errors[index]->BuildDebugSample();

    AslamBackendPointCostParityDiagnostics point_diagnostics;
    point_diagnostics.frame_index = frontend_point.frame_index;
    point_diagnostics.frame_label = frontend_point.frame_label;
    point_diagnostics.board_id = frontend_point.board_id;
    point_diagnostics.point_id = frontend_point.point_id;
    point_diagnostics.point_type = frontend_point.point_type;
    point_diagnostics.observed_image_xy = frontend_point.observed_image_xy;
    point_diagnostics.frontend_predicted_image_xy = frontend_point.predicted_image_xy;
    point_diagnostics.backend_predicted_image_xy = backend_point.predicted_image_xy;
    point_diagnostics.frontend_residual_xy = frontend_point.residual_xy;
    point_diagnostics.backend_residual_xy = backend_point.residual_xy;
    point_diagnostics.frontend_valid_projection = frontend_point.valid_projection;
    point_diagnostics.backend_valid_projection = backend_point.valid_projection;
    point_diagnostics.frontend_balance_weight = frontend_point.balance_weight;
    point_diagnostics.frontend_huber_weight = frontend_point.huber_weight;
    point_diagnostics.frontend_final_weight = frontend_point.final_weight;
    point_diagnostics.frontend_weighted_squared_error =
        frontend_point.weighted_squared_error;
    point_diagnostics.backend_inv_r_scale = backend_point.backend_inv_r_scale;
    point_diagnostics.backend_m_estimator_weight =
        backend_point.backend_m_estimator_weight;
    point_diagnostics.backend_raw_squared_error =
        backend_point.backend_raw_squared_error;
    point_diagnostics.backend_weighted_squared_error =
        backend_point.backend_weighted_squared_error;

    if (frontend_point.valid_projection && backend_point.valid_projection &&
        frontend_point.predicted_image_xy.allFinite() &&
        backend_point.predicted_image_xy.allFinite()) {
      point_diagnostics.predicted_difference_norm =
          (frontend_point.predicted_image_xy -
           backend_point.predicted_image_xy).norm();
      point_diagnostics.residual_sign_consistency_norm =
          (frontend_point.residual_xy + backend_point.residual_xy).norm();
    } else if (!frontend_point.valid_projection && !backend_point.valid_projection) {
      point_diagnostics.predicted_difference_norm = 0.0;
      point_diagnostics.residual_sign_consistency_norm =
          (frontend_point.residual_xy - backend_point.residual_xy).norm();
    } else {
      point_diagnostics.predicted_difference_norm =
          std::numeric_limits<double>::infinity();
      point_diagnostics.residual_sign_consistency_norm =
          std::numeric_limits<double>::infinity();
    }

    point_diagnostics.weighted_cost_difference =
        point_diagnostics.backend_weighted_squared_error -
        point_diagnostics.frontend_weighted_squared_error;

    diagnostics.backend_reprojection_total_raw_squared_error +=
        point_diagnostics.backend_raw_squared_error;
    diagnostics.backend_reprojection_total_weighted_cost +=
        point_diagnostics.backend_weighted_squared_error;
    diagnostics.total_abs_weighted_cost_difference +=
        std::fabs(point_diagnostics.weighted_cost_difference);
    diagnostics.max_abs_weighted_cost_difference = std::max(
        diagnostics.max_abs_weighted_cost_difference,
        std::fabs(point_diagnostics.weighted_cost_difference));
    diagnostics.max_predicted_difference_norm = std::max(
        diagnostics.max_predicted_difference_norm,
        point_diagnostics.predicted_difference_norm);
    diagnostics.max_residual_sign_consistency_norm = std::max(
        diagnostics.max_residual_sign_consistency_norm,
        point_diagnostics.residual_sign_consistency_norm);
    diagnostics.point_diagnostics.push_back(point_diagnostics);
  }

  diagnostics.compared_point_count =
      static_cast<int>(diagnostics.point_diagnostics.size());
  diagnostics.success = true;
  return diagnostics;
}

template <typename GeometryT>
AslamBackendOptimizationStageSummary RunOptimizationStage(
    const std::string& stage_label,
    bool optimize_intrinsics,
    int max_iterations,
    const AslamBackendCalibrationOptions& options,
    const boost::shared_ptr<aslam::backend::OptimizationProblem>& problem,
    CameraDv<GeometryT>* camera_dv) {
  if (problem == nullptr || camera_dv == nullptr) {
    throw std::runtime_error("RunOptimizationStage requires a valid problem and camera DV.");
  }
  constexpr bool kHasDistortionDv =
      GeometryT::projection_t::distortion_t::DesignVariableDimension > 0;
  camera_dv->setActive(optimize_intrinsics,
                       optimize_intrinsics && kHasDistortionDv,
                       false);

  AslamBackendOptimizationStageSummary summary;
  summary.stage_label = stage_label;
  summary.optimize_intrinsics = optimize_intrinsics;
  summary.max_iterations = max_iterations;
  if (max_iterations <= 0) {
    return summary;
  }

  aslam::backend::OptimizerOptions optimizer_options;
  optimizer_options.maxIterations = max_iterations;
  optimizer_options.convergenceDeltaJ = options.convergence_delta_j;
  optimizer_options.convergenceDeltaX = options.convergence_delta_x;
  optimizer_options.levenbergMarquardtLambdaInit =
      options.levenberg_marquardt_lambda_init;
  optimizer_options.doLevenbergMarquardt = true;
  optimizer_options.doSchurComplement = false;
  optimizer_options.verbose = options.verbose;
  optimizer_options.linearSolver = options.linear_solver;

  aslam::backend::Optimizer optimizer(optimizer_options);
  optimizer.setProblem(problem);
  const aslam::backend::SolutionReturnValue solution = optimizer.optimize();

  summary.objective_start = solution.JStart;
  summary.objective_final = solution.JFinal;
  summary.iterations = solution.iterations;
  summary.failed_iterations = solution.failedIterations;
  summary.lm_lambda_final = solution.lmLambdaFinal;
  summary.delta_x_final = solution.dXFinal;
  summary.delta_j_final = solution.dJFinal;
  summary.linear_solver_failure = solution.linearSolverFailure;
  return summary;
}

template <typename GeometryT>
void MaybeAddDistortionAnchorPrior(
    const CalibrationBackendProblemInput&,
    const boost::shared_ptr<GeometryT>&,
    CameraDv<GeometryT>*,
    const boost::shared_ptr<aslam::backend::OptimizationProblem>&) {}

template <>
void MaybeAddDistortionAnchorPrior<PinholeEquiGeometry>(
    const CalibrationBackendProblemInput& input,
    const boost::shared_ptr<PinholeEquiGeometry>& camera_geometry,
    CameraDv<PinholeEquiGeometry>* camera_dv,
    const boost::shared_ptr<aslam::backend::OptimizationProblem>& problem) {
  if (!input.priors.use_intrinsics_anchor_prior) {
    return;
  }
  if (camera_dv == nullptr) {
    throw std::runtime_error(
        "MaybeAddDistortionAnchorPrior<PinholeEquiGeometry> requires a valid camera DV.");
  }
  Eigen::Matrix<double, 4, 1> anchor = Eigen::Matrix<double, 4, 1>::Zero();
  const std::vector<double> distortion =
      input.scene_state.camera.DistortionVector();
  for (std::size_t index = 0; index < std::min<std::size_t>(4, distortion.size()); ++index) {
    anchor[static_cast<Eigen::Index>(index)] = distortion[index];
  }
  Eigen::Matrix<double, 4, 1> prior_weight = Eigen::Matrix<double, 4, 1>::Constant(
      input.priors.intrinsics_anchor_weight_xi_alpha);
  boost::shared_ptr<EquidistantDistortionAnchorError> prior(
      new EquidistantDistortionAnchorError(
          &camera_geometry->projection().distortion(),
          camera_dv->distortionDesignVariable(),
          anchor,
          prior_weight));
  problem->addErrorTerm(prior);
}

template <typename GeometryT>
bool ExecuteBackendOptimization(
    AslamBackendCalibrationResult* result,
    const JointMeasurementBuildResult& measurement_result,
    const JointReprojectionResidualEvaluator& residual_evaluator,
    const JointReprojectionCostOptions& frontend_cost_options) {
  if (result == nullptr) {
    throw std::runtime_error("ExecuteBackendOptimization requires a valid result pointer.");
  }

	  boost::shared_ptr<GeometryT> camera_geometry =
	      MakeTypedGeometry<GeometryT>(result->initial_scene_state.camera);
	  const boost::shared_ptr<GeometryT> anchor_camera_geometry =
	      MakeTypedGeometry<GeometryT>(result->anchor_camera);
	  CameraDv<GeometryT> camera_dv(camera_geometry);
	  camera_dv.setActive(false, false, false);
	  const double chordal_reference_focal_px = std::sqrt(
	      std::max(1.0, std::abs(result->initial_scene_state.camera.fu)) *
	      std::max(1.0, std::abs(result->initial_scene_state.camera.fv)));

  boost::shared_ptr<aslam::backend::OptimizationProblem> problem(
      new aslam::backend::OptimizationProblem);
  problem->addDesignVariable(camera_dv.projectionDesignVariable());
  problem->addDesignVariable(camera_dv.distortionDesignVariable());
  problem->addDesignVariable(camera_dv.shutterDesignVariable());

  const bool independent_frame_board_pose =
      result->options.board_pose_parameterization ==
      AslamBackendCalibrationOptions::BoardPoseParameterization::
          IndependentFrameBoardPose;
  std::map<int, PoseVariableState> frame_variables;
  std::map<int, PoseVariableState> board_variables;
  std::map<std::pair<int, int>, PoseVariableState> local_pose_variables;
  std::map<aslam::backend::DesignVariable*, std::string> design_variable_labels;
  design_variable_labels[camera_dv.projectionDesignVariable().get()] =
      "camera_intrinsics";
  design_variable_labels[camera_dv.distortionDesignVariable().get()] =
      "camera_distortion";
  design_variable_labels[camera_dv.shutterDesignVariable().get()] =
      "camera_shutter";

  const std::map<std::pair<int, int>, ObservationBudget> observation_budgets =
      BuildObservationBudgets(measurement_result);
  if (independent_frame_board_pose) {
    const bool active =
        result->effective_problem_input.optimization_masks.optimize_frame_poses;
    for (const auto& entry : observation_budgets) {
      Eigen::Matrix4d T_camera_board = Eigen::Matrix4d::Identity();
      if (!ComposeCameraBoardPoseFromReferenceChain(
              result->initial_scene_state, entry.first.first,
              entry.first.second, &T_camera_board)) {
        continue;
      }
      PoseVariableState& variable = local_pose_variables[entry.first];
      variable.transform = sm::kinematics::Transformation(T_camera_board);
      variable.expression = aslam::backend::transformationToExpression(
          variable.transform, variable.rotation_dv, variable.translation_dv);
      variable.rotation_dv->setActive(active);
      variable.translation_dv->setActive(active);
      problem->addDesignVariable(variable.rotation_dv);
      problem->addDesignVariable(variable.translation_dv);
      const std::string key_label =
          "frame_" + std::to_string(entry.first.first) + "_board_" +
          std::to_string(entry.first.second);
      design_variable_labels[variable.rotation_dv.get()] =
          "local_board_rotation_" + key_label;
      design_variable_labels[variable.translation_dv.get()] =
          "local_board_translation_" + key_label;
    }
    result->initial_residual =
        EvaluateIndependentFrameBoardPoseResiduals(
            measurement_result, camera_geometry, local_pose_variables,
            result->effective_problem_input.reference_board_id, 10);
    if (!result->initial_residual.success) {
      result->failure_reason = result->initial_residual.failure_reason;
      return false;
    }
  } else {
    for (const JointSceneFrameState& frame_state :
         result->initial_scene_state.frames) {
      if (!frame_state.initialized) {
        continue;
      }
      PoseVariableState& variable = frame_variables[frame_state.frame_index];
      variable.transform =
          sm::kinematics::Transformation(frame_state.T_camera_reference);
      variable.expression = aslam::backend::transformationToExpression(
          variable.transform, variable.rotation_dv, variable.translation_dv);
      const bool active =
          result->effective_problem_input.optimization_masks.optimize_frame_poses;
      variable.rotation_dv->setActive(active);
      variable.translation_dv->setActive(active);
      problem->addDesignVariable(variable.rotation_dv);
      problem->addDesignVariable(variable.translation_dv);
      design_variable_labels[variable.rotation_dv.get()] =
          "frame_rotation_frame_" + std::to_string(frame_state.frame_index);
      design_variable_labels[variable.translation_dv.get()] =
          "frame_translation_frame_" + std::to_string(frame_state.frame_index);
    }

    for (const JointSceneBoardState& board_state :
         result->initial_scene_state.boards) {
      if (!board_state.initialized ||
          board_state.board_id ==
              result->effective_problem_input.reference_board_id) {
        continue;
      }
      PoseVariableState& variable = board_variables[board_state.board_id];
      variable.transform =
          sm::kinematics::Transformation(board_state.T_reference_board);
      variable.expression = aslam::backend::transformationToExpression(
          variable.transform, variable.rotation_dv, variable.translation_dv);
      const bool active =
          result->effective_problem_input.optimization_masks.optimize_board_poses;
      variable.rotation_dv->setActive(active);
      variable.translation_dv->setActive(active);
      problem->addDesignVariable(variable.rotation_dv);
      problem->addDesignVariable(variable.translation_dv);
      design_variable_labels[variable.rotation_dv.get()] =
          "board_rotation_board_" + std::to_string(board_state.board_id);
      design_variable_labels[variable.translation_dv.get()] =
          "board_translation_board_" + std::to_string(board_state.board_id);
    }

  }
  const aslam::backend::TransformationExpression identity_transform(
      Eigen::Matrix4d::Identity());
  int skipped_point_count = 0;
  std::vector<boost::shared_ptr<CameraReprojectionError<GeometryT> > > reprojection_errors;
  std::vector<boost::shared_ptr<CameraAngularReprojectionError<GeometryT> > >
      angular_reprojection_errors;
  std::vector<
      boost::shared_ptr<CameraPolarContinuousHybridReprojectionError<GeometryT> > >
      hybrid_reprojection_errors;
  std::vector<
      boost::shared_ptr<CameraPixelRayHybridReprojectionError<GeometryT> > >
      pixel_ray_hybrid_reprojection_errors;
  std::vector<double> pixel_ray_hybrid_pixel_norms;
  std::vector<double> pixel_ray_hybrid_ray_norms;
  std::vector<aslam::backend::ErrorTerm*> influence_error_terms;
  std::vector<BackendErrorTermInfluenceMetadata> influence_metadata;
  reprojection_errors.reserve(
      static_cast<std::size_t>(measurement_result.used_total_point_count));
  angular_reprojection_errors.reserve(
      static_cast<std::size_t>(measurement_result.used_total_point_count));
  hybrid_reprojection_errors.reserve(
      static_cast<std::size_t>(measurement_result.used_total_point_count));
  pixel_ray_hybrid_reprojection_errors.reserve(
      static_cast<std::size_t>(measurement_result.used_total_point_count));
  pixel_ray_hybrid_pixel_norms.reserve(
      static_cast<std::size_t>(measurement_result.used_total_point_count));
  pixel_ray_hybrid_ray_norms.reserve(
      static_cast<std::size_t>(measurement_result.used_total_point_count));
  influence_error_terms.reserve(
      static_cast<std::size_t>(measurement_result.used_total_point_count));
  influence_metadata.reserve(
      static_cast<std::size_t>(measurement_result.used_total_point_count));
  for (const JointPointObservation& observation : measurement_result.solver_observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    const Eigen::Vector4d zero_h = Eigen::Vector4d::Zero();
    aslam::backend::HomogeneousExpression point_camera(zero_h);
    const aslam::backend::HomogeneousExpression point_board(
        observation.target_xyz_board);
    if (independent_frame_board_pose) {
      const auto pose_it = local_pose_variables.find(
          std::make_pair(observation.frame_index, observation.board_id));
      if (pose_it == local_pose_variables.end()) {
        ++skipped_point_count;
        continue;
      }
      point_camera = pose_it->second.expression * point_board;
    } else {
      const auto frame_it = frame_variables.find(observation.frame_index);
      if (frame_it == frame_variables.end()) {
        ++skipped_point_count;
        continue;
      }

      aslam::backend::TransformationExpression board_expression =
          identity_transform;
      if (observation.board_id !=
          result->effective_problem_input.reference_board_id) {
        const auto board_it = board_variables.find(observation.board_id);
        if (board_it == board_variables.end()) {
          ++skipped_point_count;
          continue;
        }
        board_expression = board_it->second.expression;
      }

      const aslam::backend::HomogeneousExpression point_reference =
          board_expression * point_board;
      point_camera = frame_it->second.expression * point_reference;
    }

    const ObservationBudget& budget =
        observation_budgets.find(std::make_pair(observation.frame_index, observation.board_id))
            ->second;
    const double polar_angle_weight =
        ComputePolarAngleWeightScale(*camera_geometry, observation,
                                     result->options);
    const bool use_unweighted_points =
        result->options.observation_role_weight_mode == "unweighted_points";
    const double balance_weight =
        ComputeBalanceWeight(budget, observation.point_type, result->options) *
        (use_unweighted_points
             ? 1.0
             : std::max(0.0, observation.final_observation_weight) *
                   polar_angle_weight);
    const Eigen::Matrix2d inverse_covariance = ComputeBackendInverseCovariance(
        balance_weight, observation.point_type, result->options);
    const GeometryT& observation_ray_camera =
        result->options.angular_observed_ray_mode ==
                AslamBackendCalibrationOptions::AngularObservedRayMode::
                    FrozenAnchorCamera
            ? *anchor_camera_geometry
            : *camera_geometry;
    AngularObservationGeometry angular_observation_geometry;
    const bool have_angular_observation_geometry =
        ComputeObservationGeometryForCamera(
            observation_ray_camera, observation.image_xy,
            &angular_observation_geometry);
    const double observed_polar_angle_deg =
        have_angular_observation_geometry
            ? angular_observation_geometry.polar_angle_deg
            : std::numeric_limits<double>::quiet_NaN();
    ResidualModel requested_point_residual_model =
        result->options.residual_model;
    if (result->options.use_point_type_residual_split) {
      requested_point_residual_model =
          observation.point_type == JointPointType::Outer
              ? result->options.outer_residual_model
              : result->options.internal_residual_model;
    }
	    const bool bearing_space_mode =
	        result->options.pixel_ray_hybrid_refinement_mode ||
	        requested_point_residual_model == ResidualModel::SphereAngular ||
	        requested_point_residual_model ==
	            ResidualModel::NormalizedSphereAngular ||
	        requested_point_residual_model == ResidualModel::HybridEdgeAngular ||
	        requested_point_residual_model ==
	            ResidualModel::PolarContinuousHybrid ||
	        requested_point_residual_model == ResidualModel::Chordal ||
	        requested_point_residual_model ==
	            ResidualModel::PixelChordalHybrid;
	    if (bearing_space_mode && !have_angular_observation_geometry) {
	      ++skipped_point_count;
	      continue;
	    }
	    const bool use_continuous_hybrid =
	        requested_point_residual_model == ResidualModel::PolarContinuousHybrid;
	    const bool use_normalized_angular =
	        requested_point_residual_model == ResidualModel::NormalizedSphereAngular;
	    const bool use_pixel_chordal_hybrid =
	        requested_point_residual_model == ResidualModel::PixelChordalHybrid;
	    const double continuous_angular_weight = use_continuous_hybrid
	        ? ComputePolarContinuousAngularWeight(
              observed_polar_angle_deg,
              result->options.polar_continuous_hybrid_threshold_deg,
              result->options.polar_continuous_hybrid_temperature_deg)
        : 0.0;
	    const bool use_angular_residual =
	        !use_continuous_hybrid &&
	        ShouldUseAngularResidual(
	            requested_point_residual_model,
	            observed_polar_angle_deg,
	            result->options.hybrid_angular_threshold_deg);
	    const bool use_chordal_residual =
	        (requested_point_residual_model == ResidualModel::Chordal ||
	         requested_point_residual_model == ResidualModel::PixelChordalHybrid) &&
	        have_angular_observation_geometry;
	    const bool add_pixel_residual =
	        result->options.pixel_ray_hybrid_refinement_mode
	            ? false
	            : use_pixel_chordal_hybrid
	            ? result->options.pixel_residual_weight > 0.0
	            : (!use_angular_residual && !use_chordal_residual) ||
	                  use_continuous_hybrid;
	    const bool add_angular_primary =
	        use_angular_residual || continuous_angular_weight > 0.0;
    const bool add_chordal_primary =
        use_chordal_residual &&
        result->options.chordal_residual_weight > 0.0;
    const double pixel_ray_hybrid_lambda =
        result->options.pixel_ray_hybrid_refinement_mode
            ? ComputeFixedPolarAdaptivePixelRayLambda(
                  result->options, observed_polar_angle_deg)
            : 0.0;
    const bool add_angular_auxiliary =
        result->options.angular_auxiliary_enabled &&
        result->options.angular_auxiliary_weight > 0.0 &&
        ((observation.point_type == JointPointType::Outer &&
          result->options.angular_auxiliary_apply_to_outer) ||
         (observation.point_type == JointPointType::Internal &&
          result->options.angular_auxiliary_apply_to_internal));
	    const double pixel_weight_scale = use_pixel_chordal_hybrid
	        ? std::max(0.0, result->options.pixel_residual_weight)
	        : (use_continuous_hybrid
	               ? std::max(0.0, 1.0 - continuous_angular_weight)
	               : 1.0);
	    const double angular_primary_weight_scale = use_continuous_hybrid
	        ? continuous_angular_weight *
	              chordal_reference_focal_px * chordal_reference_focal_px
	        : 1.0;
	    const double chordal_weight_scale =
	        add_chordal_primary
	            ? std::max(0.0, result->options.chordal_residual_weight)
	            : 0.0;
    double angular_sigma_per_pixel_rad = std::numeric_limits<double>::quiet_NaN();
    double normalized_angular_weight_scale = 1.0;
    const bool need_normalized_angular_scale =
        (use_normalized_angular ||
         (add_angular_auxiliary &&
          result->options.angular_auxiliary_normalized)) &&
        have_angular_observation_geometry;
    if (need_normalized_angular_scale) {
      angular_sigma_per_pixel_rad =
          EstimateAngularSigmaPerPixelForCamera(
              observation_ray_camera, observation.image_xy,
              angular_observation_geometry);
      const double safe_sigma = std::max(
          result->options.normalized_angular_min_sigma_rad,
          angular_sigma_per_pixel_rad);
      const double reference_sigma_px = std::max(
          1e-12, result->options.normalized_angular_reference_sigma_px);
      if (std::isfinite(safe_sigma) && safe_sigma > 0.0) {
        normalized_angular_weight_scale =
            std::min(result->options.normalized_angular_max_weight_scale,
                     (reference_sigma_px * reference_sigma_px) /
                         (safe_sigma * safe_sigma));
      }
    }
    BackendResidualTypeAssignment residual_type_assignment;
    residual_type_assignment.frame_index = observation.frame_index;
    residual_type_assignment.frame_label = observation.frame_label;
    residual_type_assignment.board_id = observation.board_id;
    residual_type_assignment.point_id = observation.point_id;
    residual_type_assignment.point_type = observation.point_type;
    residual_type_assignment.polar_angle_deg =
        observed_polar_angle_deg;
    residual_type_assignment.residual_model_requested =
        ToString(requested_point_residual_model);
	    if (result->options.pixel_ray_hybrid_refinement_mode) {
	      residual_type_assignment.residual_model_effective =
	          "pixel_ray_hybrid_final_refinement";
	    } else if (use_angular_residual && have_angular_observation_geometry) {
	      residual_type_assignment.residual_model_effective =
	          result->options.angular_local_whitening_enabled
	              ? "locally_whitened_sphere_angular"
	              : (use_normalized_angular ? "normalized_sphere_angular"
	                                        : "sphere_angular");
	    } else if (use_continuous_hybrid) {
	      residual_type_assignment.residual_model_effective =
	          "polar_continuous_hybrid";
	    } else if (requested_point_residual_model == ResidualModel::Chordal) {
	      residual_type_assignment.residual_model_effective = "chordal";
	    } else if (use_pixel_chordal_hybrid) {
	      residual_type_assignment.residual_model_effective =
	          "pixel_chordal_hybrid";
	    } else {
	      residual_type_assignment.residual_model_effective = "image_plane";
	    }
    residual_type_assignment.angular_observation_geometry_success =
        have_angular_observation_geometry;
    residual_type_assignment.pixel_ray_hybrid_polar_adaptive =
        result->options.pixel_ray_hybrid_refinement_mode &&
        result->options.pixel_ray_hybrid_polar_adaptive_enabled;
    residual_type_assignment.pixel_ray_hybrid_lambda =
        pixel_ray_hybrid_lambda;
    residual_type_assignment.image_plane_weight_scale =
        result->options.pixel_ray_hybrid_refinement_mode
            ? 1.0 - pixel_ray_hybrid_lambda
            : (add_pixel_residual ? pixel_weight_scale : 0.0);
    residual_type_assignment.angular_weight_scale =
        result->options.pixel_ray_hybrid_refinement_mode
            ? pixel_ray_hybrid_lambda
            : add_angular_primary
            ? angular_primary_weight_scale *
                  (use_normalized_angular ? normalized_angular_weight_scale : 1.0)
            : 0.0;
    residual_type_assignment.angular_sigma_per_pixel_rad =
        std::isfinite(angular_sigma_per_pixel_rad)
            ? angular_sigma_per_pixel_rad
            : 0.0;
    residual_type_assignment.normalized_angular_weight_scale =
        need_normalized_angular_scale ? normalized_angular_weight_scale : 0.0;
    residual_type_assignment.angular_auxiliary_enabled =
        add_angular_auxiliary && have_angular_observation_geometry;
    residual_type_assignment.angular_auxiliary_normalized =
        residual_type_assignment.angular_auxiliary_enabled &&
        result->options.angular_auxiliary_normalized;
    result->residual_type_assignments.push_back(residual_type_assignment);

    auto append_influence_metadata =
        [&](aslam::backend::ErrorTerm* error_term,
            const std::string& residual_family) {
          if (error_term == nullptr) {
            return;
          }
          BackendErrorTermInfluenceMetadata meta;
          meta.frame_index = observation.frame_index;
          meta.frame_label = observation.frame_label;
          meta.board_id = observation.board_id;
          meta.point_type = observation.point_type;
          meta.residual_family = residual_family;
          influence_error_terms.push_back(error_term);
          influence_metadata.push_back(meta);
        };

	    if (result->options.pixel_ray_hybrid_refinement_mode) {
	      if (!(balance_weight > 0.0) || !std::isfinite(balance_weight)) {
	        ++result->pixel_ray_hybrid_invalid_observation_count;
	        ++skipped_point_count;
	        continue;
	      }
	      boost::shared_ptr<CameraPixelRayHybridReprojectionError<GeometryT> > error(
	          new CameraPixelRayHybridReprojectionError<GeometryT>(
	              observation.image_xy,
	              inverse_covariance,
	              pixel_ray_hybrid_lambda,
	              result->options.pixel_ray_hybrid_huber_delta,
	              result->options.use_huber_loss,
	              point_camera,
	              camera_dv,
	              result->options.invalid_projection_penalty_pixels,
	              result->options.invalid_projection_penalty_radians));
	      Eigen::Vector2d raw_pixel = Eigen::Vector2d::Zero();
	      Eigen::Vector2d raw_ray = Eigen::Vector2d::Zero();
	      if (!error->BuildRawComponents(&raw_pixel, &raw_ray)) {
	        ++result->pixel_ray_hybrid_invalid_observation_count;
	        ++skipped_point_count;
	        continue;
	      }
	      pixel_ray_hybrid_pixel_norms.push_back(raw_pixel.norm());
	      pixel_ray_hybrid_ray_norms.push_back(raw_ray.norm());
	      pixel_ray_hybrid_reprojection_errors.push_back(error);
	      problem->addErrorTerm(error);
	      append_influence_metadata(error.get(), "pixel_ray_hybrid_4d");
	      ++result->residual_block_construction.pixel_ray_hybrid_residual_count;
	      if (observation.point_type == JointPointType::Outer) {
	        ++result->residual_block_construction
	              .outer_pixel_ray_hybrid_residual_count;
	      } else {
	        ++result->residual_block_construction
	              .internal_pixel_ray_hybrid_residual_count;
	      }
	      continue;
	    }

	    auto add_angular_error = [&](double weight_scale, bool auxiliary) {
      if (!have_angular_observation_geometry || !(weight_scale > 0.0)) {
        return;
      }
	  Eigen::Matrix2d angular_inverse_covariance =
	      inverse_covariance * weight_scale;
	  if (result->options.angular_local_whitening_enabled && !auxiliary) {
	    BearingCovarianceOptions covariance_options;
	    covariance_options.use_pixel_uncertainty = true;
	    covariance_options.use_model_uncertainty = false;
	    covariance_options.pixel_sigma_px =
	        result->options.angular_local_whitening_pixel_sigma_px;
	    covariance_options.covariance_damping =
	        result->options.angular_local_whitening_covariance_damping;
	    covariance_options.min_sigma_rad =
	        result->options.angular_local_whitening_min_sigma_rad;
	    covariance_options.max_whitening_weight =
	        result->options.angular_local_whitening_max_weight;
	    BearingCovarianceResult covariance_result;
	    if (!ComputeBearingTangentCovariance(
	            MakeIntermediateCameraConfig(
	                GeometryToIntrinsics<GeometryT>(observation_ray_camera)),
	            observation.image_xy, angular_observation_geometry,
	            covariance_options, &covariance_result)) {
	      ++skipped_point_count;
	      return;
	    }
	    angular_inverse_covariance =
	        balance_weight * covariance_result.sqrt_information.transpose() *
	        covariance_result.sqrt_information;
	  }
      const double huber_delta_radians = result->options.uniform_control_point_mode
          ? std::min(result->options.outer_huber_delta_radians,
                     result->options.internal_huber_delta_radians)
          : observation.point_type == JointPointType::Outer
                ? result->options.outer_huber_delta_radians
                : result->options.internal_huber_delta_radians;
      boost::shared_ptr<CameraAngularReprojectionError<GeometryT> > error(
          new CameraAngularReprojectionError<GeometryT>(
              observation.image_xy,
              angular_inverse_covariance,
              huber_delta_radians,
              result->options.use_huber_loss,
              point_camera,
              camera_dv,
              result->options.invalid_projection_penalty_radians,
              result->options.angular_use_normalize_jacobian,
              result->options.angular_observed_ray_mode,
              angular_observation_geometry));
      problem->addErrorTerm(error);
      angular_reprojection_errors.push_back(error);
      append_influence_metadata(
          error.get(), auxiliary ? "angular_auxiliary" : "angular_primary");
      ++result->residual_block_construction.angular_residual_count;
      if (observation.point_type == JointPointType::Outer) {
        ++result->residual_block_construction.outer_angular_residual_count;
      } else {
        ++result->residual_block_construction.internal_angular_residual_count;
      }
      if (auxiliary) {
        ++result->residual_block_construction.angular_auxiliary_residual_count;
        if (observation.point_type == JointPointType::Outer) {
          ++result->residual_block_construction
                .outer_angular_auxiliary_residual_count;
        } else {
          ++result->residual_block_construction
                .internal_angular_auxiliary_residual_count;
        }
      }
	    };

	    auto add_chordal_error = [&]() {
	      if (!have_angular_observation_geometry || !(chordal_weight_scale > 0.0)) {
	        return;
	      }
	      const double huber_delta_chordal = result->options.uniform_control_point_mode
	          ? std::min(result->options.outer_huber_delta_radians,
	                     result->options.internal_huber_delta_radians)
	          : observation.point_type == JointPointType::Outer
	                ? result->options.outer_huber_delta_radians
	                : result->options.internal_huber_delta_radians;
	      const Eigen::Matrix3d chordal_inverse_covariance =
	          balance_weight * chordal_weight_scale *
	          chordal_reference_focal_px * chordal_reference_focal_px *
	          Eigen::Matrix3d::Identity();
	      boost::shared_ptr<CameraChordalReprojectionError<GeometryT> > error(
	          new CameraChordalReprojectionError<GeometryT>(
	              observation.image_xy,
	              chordal_inverse_covariance,
	              huber_delta_chordal,
	              result->options.use_huber_loss,
	              point_camera,
	              camera_dv,
	              result->options.invalid_projection_penalty_radians,
	              result->options.angular_use_normalize_jacobian,
	              result->options.angular_observed_ray_mode,
	              angular_observation_geometry));
	      problem->addErrorTerm(error);
	      append_influence_metadata(error.get(), "chordal");
	      ++result->residual_block_construction.chordal_residual_count;
	      if (observation.point_type == JointPointType::Outer) {
	        ++result->residual_block_construction.outer_chordal_residual_count;
	      } else {
	        ++result->residual_block_construction.internal_chordal_residual_count;
	      }
	    };

    if (add_pixel_residual && pixel_weight_scale > 0.0) {
      const double huber_delta = result->options.uniform_control_point_mode
                                     ? std::min(result->options.outer_huber_delta_pixels,
                                                result->options.internal_huber_delta_pixels)
                                     : observation.point_type == JointPointType::Outer
                                           ? result->options.outer_huber_delta_pixels
                                           : result->options.internal_huber_delta_pixels;
      boost::shared_ptr<CameraReprojectionError<GeometryT> > error(
          new CameraReprojectionError<GeometryT>(
              observation.image_xy,
              inverse_covariance * pixel_weight_scale,
              huber_delta,
              result->options.use_huber_loss,
              point_camera,
              camera_dv,
              result->options.invalid_projection_penalty_pixels));
      problem->addErrorTerm(error);
      reprojection_errors.push_back(error);
      append_influence_metadata(error.get(), "image_plane");
      ++result->residual_block_construction.image_plane_residual_count;
      if (observation.point_type == JointPointType::Outer) {
        ++result->residual_block_construction.outer_image_plane_residual_count;
      } else {
        ++result->residual_block_construction.internal_image_plane_residual_count;
      }
    }
	    if (add_angular_primary) {
	      add_angular_error(
	          angular_primary_weight_scale *
	              (use_normalized_angular ? normalized_angular_weight_scale : 1.0),
	          false);
	    }
	    if (add_chordal_primary) {
	      add_chordal_error();
	    }
    if (add_angular_auxiliary) {
      const double auxiliary_normalized_scale =
          result->options.angular_auxiliary_normalized
              ? normalized_angular_weight_scale
              : 1.0;
      add_angular_error(
          result->options.angular_auxiliary_weight *
              auxiliary_normalized_scale,
          true);
    }
  }

  if (result->options.pixel_ray_hybrid_refinement_mode) {
    result->pixel_ray_hybrid_valid_observation_count =
        static_cast<int>(pixel_ray_hybrid_reprojection_errors.size());
    if (pixel_ray_hybrid_reprojection_errors.empty()) {
      result->failure_reason =
          "Pixel-ray hybrid refinement has zero valid training observations.";
      return false;
    }
    auto median = [](std::vector<double> values) {
      const std::size_t count = values.size();
      const std::size_t middle = count / 2;
      std::nth_element(values.begin(), values.begin() + middle, values.end());
      const double upper = values[middle];
      if ((count % 2) != 0) {
        return upper;
      }
      const double lower =
          *std::max_element(values.begin(), values.begin() + middle);
      return 0.5 * (lower + upper);
    };
    const double pixel_median = median(pixel_ray_hybrid_pixel_norms);
    const double ray_median = median(pixel_ray_hybrid_ray_norms);
    if (!std::isfinite(pixel_median) || !std::isfinite(ray_median)) {
      result->failure_reason =
          "Pixel-ray hybrid refinement produced non-finite median scales.";
      return false;
    }
    result->pixel_ray_hybrid_pixel_scale = std::max(
        pixel_median, result->options.pixel_ray_hybrid_pixel_scale_floor);
    result->pixel_ray_hybrid_ray_scale = std::max(
        ray_median, result->options.pixel_ray_hybrid_ray_scale_floor);
    if (!(result->pixel_ray_hybrid_pixel_scale > 0.0) ||
        !(result->pixel_ray_hybrid_ray_scale > 0.0) ||
        !std::isfinite(result->pixel_ray_hybrid_pixel_scale) ||
        !std::isfinite(result->pixel_ray_hybrid_ray_scale)) {
      result->failure_reason =
          "Pixel-ray hybrid refinement scales are not finite and positive.";
      return false;
    }
    for (const auto& error : pixel_ray_hybrid_reprojection_errors) {
      error->SetFixedScales(result->pixel_ray_hybrid_pixel_scale,
                            result->pixel_ray_hybrid_ray_scale);
    }
    result->pixel_ray_hybrid_scales_computed_once = true;
  }
  result->residual_block_construction.skipped_solver_observation_count =
      skipped_point_count;

  if (skipped_point_count > 0) {
    AppendUniqueWarning(
        "Skipped " + std::to_string(skipped_point_count) +
            " solver observations while building the backend problem.",
        &result->warnings);
  }

  if (result->effective_problem_input.priors.use_intrinsics_anchor_prior) {
    const std::vector<double> anchor_values =
        result->effective_problem_input.scene_state.camera.IntrinsicsVector();
    const std::vector<std::string> anchor_labels =
        result->effective_problem_input.scene_state.camera.IntrinsicsLabels();
    using projection_t = typename GeometryT::projection_t;
    typedef ProjectionAnchorError<projection_t> prior_t;
    typename prior_t::vector_t anchor_matrix = prior_t::vector_t::Zero();
    typename prior_t::vector_t prior_matrix = prior_t::vector_t::Zero();
    for (std::size_t index = 0; index < anchor_values.size(); ++index) {
      anchor_matrix[static_cast<Eigen::Index>(index)] = anchor_values[index];
      prior_matrix[static_cast<Eigen::Index>(index)] =
          PriorWeightForLabel(result->effective_problem_input.priors, anchor_labels[index]);
    }
    boost::shared_ptr<prior_t> prior(
        new prior_t(&camera_geometry->projection(),
                    camera_dv.projectionDesignVariable(),
                    anchor_matrix,
                    prior_matrix));
    problem->addErrorTerm(prior);
    MaybeAddDistortionAnchorPrior<GeometryT>(
        result->effective_problem_input, camera_geometry, &camera_dv, problem);
  }

  result->board_pose_prior_translation_sigma_mm =
      result->options.board_pose_prior_translation_sigma_mm;
  result->board_pose_prior_rotation_sigma_deg =
      result->options.board_pose_prior_rotation_sigma_deg;
  if (result->options.board_pose_prior_enabled &&
      result->effective_problem_input.optimization_masks.optimize_board_poses &&
      !independent_frame_board_pose) {
    for (const auto& board_entry : board_variables) {
      const PoseVariableState& variable = board_entry.second;
      boost::shared_ptr<BoardPoseAnchorError> prior(
          new BoardPoseAnchorError(
              variable.rotation_dv,
              variable.translation_dv,
              variable.transform.q(),
              variable.transform.t(),
              result->options.board_pose_prior_translation_sigma_mm,
              result->options.board_pose_prior_rotation_sigma_deg));
      problem->addErrorTerm(prior);
      ++result->board_pose_prior_count;
    }
  }

  result->design_variable_count = static_cast<int>(problem->numDesignVariables());
  result->error_term_count = static_cast<int>(problem->numErrorTerms());
  if (result->error_term_count <= 0) {
    result->failure_reason = "ASLAM backend problem contains zero error terms.";
    return false;
  }

  if (result->options.export_cost_parity_diagnostics &&
      !independent_frame_board_pose) {
    if (!hybrid_reprojection_errors.empty()) {
      result->initial_cost_parity = EvaluateCostParityDiagnostics(
          "initial",
          problem,
          measurement_result,
          result->initial_scene_state,
          hybrid_reprojection_errors,
          frontend_cost_options);
    } else {
      result->initial_cost_parity = EvaluateCostParityDiagnostics(
          "initial",
          problem,
          measurement_result,
          result->initial_scene_state,
          reprojection_errors,
          frontend_cost_options);
    }
  } else if (result->options.export_cost_parity_diagnostics &&
             independent_frame_board_pose) {
    AppendUniqueWarning(
        "Cost parity diagnostics are skipped for independent_frame_board_pose "
        "because the parity evaluator is reference-chain scene based.",
        &result->warnings);
  }
  if (result->options.export_variable_block_influence_diagnostics) {
    result->initial_variable_block_influence =
        EvaluateVariableBlockInfluenceDiagnostics(
            "initial",
            problem,
            influence_error_terms,
            influence_metadata,
            design_variable_labels);
  }
  if (result->options.run_jacobian_consistency_check &&
      !independent_frame_board_pose) {
    result->jacobian_diagnostics = RunJacobianDiagnostics<GeometryT>(
        problem,
        &camera_dv,
        frame_variables,
        board_variables,
        result->options.jacobian_finite_difference_epsilon);
  } else if (result->options.run_jacobian_consistency_check &&
             independent_frame_board_pose) {
    AppendUniqueWarning(
        "Jacobian consistency diagnostics are skipped for "
        "independent_frame_board_pose because the diagnostic helper currently "
        "expects reference-chain frame/board variable blocks.",
        &result->warnings);
  }

  if (result->options.skip_optimization) {
    AppendUniqueWarning("ASLAM backend optimization skipped; returning the "
                        "selection committed state for Kalibr-style baseline.",
                        &result->warnings);
  } else if (problem->countActiveDesignVariables() <= 0) {
    AppendUniqueWarning("ASLAM backend problem has zero active design variables; "
                        "returning the frozen baseline state.",
                        &result->warnings);
  } else {
    const bool optimize_intrinsics =
        result->effective_problem_input.optimization_masks.optimize_intrinsics;
    if (optimize_intrinsics &&
        result->effective_problem_input.optimization_masks.delayed_intrinsics_release) {
      const int pose_only_iterations =
          std::max(0, std::min(result->options.max_iterations - 1,
                               result->effective_problem_input.optimization_masks
                                   .intrinsics_release_iteration));
      const int released_iterations =
          std::max(1, result->options.max_iterations - pose_only_iterations);
      if (pose_only_iterations > 0) {
        result->stages.push_back(RunOptimizationStage<GeometryT>(
            "pose_only", false, pose_only_iterations, result->options, problem, &camera_dv));
      }
      result->stages.push_back(RunOptimizationStage<GeometryT>(
          "intrinsics_released", true, released_iterations, result->options, problem, &camera_dv));
    } else {
      result->stages.push_back(RunOptimizationStage<GeometryT>(
          optimize_intrinsics ? "joint_full" : "pose_only",
          optimize_intrinsics, result->options.max_iterations, result->options, problem, &camera_dv));
    }
  }
  result->optimized_scene_state.camera = GeometryToIntrinsics<GeometryT>(*camera_geometry);
  if (!result->optimized_scene_state.camera.IsValid()) {
    const OuterBootstrapCameraIntrinsics unclamped_camera =
        result->optimized_scene_state.camera;
    ClampIntrinsicsInPlace(&result->optimized_scene_state.camera);
    AppendUniqueWarning(
        "ASLAM backend returned non-clamped intrinsics; final camera was clamped "
        "before evaluation.",
        &result->warnings);
    if (!unclamped_camera.IsValid() && !result->optimized_scene_state.camera.IsValid()) {
      result->failure_reason = "ASLAM backend produced invalid camera intrinsics.";
      return false;
    }
  }

  if (independent_frame_board_pose) {
    result->optimized_residual =
        EvaluateIndependentFrameBoardPoseResiduals(
            measurement_result, camera_geometry, local_pose_variables,
            result->effective_problem_input.reference_board_id, 10);
  } else {
    for (JointSceneFrameState& frame_state :
         result->optimized_scene_state.frames) {
      const auto frame_it = frame_variables.find(frame_state.frame_index);
      if (frame_it == frame_variables.end()) {
        continue;
      }
      frame_state.T_camera_reference = frame_it->second.transform.T();
    }
    for (JointSceneBoardState& board_state :
         result->optimized_scene_state.boards) {
      if (board_state.board_id ==
          result->effective_problem_input.reference_board_id) {
        board_state.T_reference_board = Eigen::Matrix4d::Identity();
        continue;
      }
      const auto board_it = board_variables.find(board_state.board_id);
      if (board_it == board_variables.end()) {
        continue;
      }
      board_state.T_reference_board = board_it->second.transform.T();
    }
    result->optimized_residual =
        residual_evaluator.Evaluate(measurement_result,
                                    result->optimized_scene_state);
  }
  if (!result->optimized_residual.success) {
    result->failure_reason = result->optimized_residual.failure_reason;
    result->warnings.insert(result->warnings.end(),
                            result->optimized_residual.warnings.begin(),
                            result->optimized_residual.warnings.end());
    return false;
  }

  if (result->options.export_cost_parity_diagnostics &&
      !independent_frame_board_pose) {
    if (!hybrid_reprojection_errors.empty()) {
      result->optimized_cost_parity = EvaluateCostParityDiagnostics(
          "optimized",
          problem,
          measurement_result,
          result->optimized_scene_state,
          hybrid_reprojection_errors,
          frontend_cost_options);
    } else {
      result->optimized_cost_parity = EvaluateCostParityDiagnostics(
          "optimized",
          problem,
          measurement_result,
          result->optimized_scene_state,
          reprojection_errors,
          frontend_cost_options);
    }
  }
  if (result->options.export_variable_block_influence_diagnostics) {
    result->optimized_variable_block_influence =
        EvaluateVariableBlockInfluenceDiagnostics(
            "optimized",
            problem,
            influence_error_terms,
            influence_metadata,
            design_variable_labels);
  }

  return true;
}

}  // namespace

AslamBackendCalibrationRunner::AslamBackendCalibrationRunner(
    AslamBackendCalibrationOptions options)
    : options_(std::move(options)) {}

AslamBackendCalibrationResult AslamBackendCalibrationRunner::Run(
    const CalibrationBackendProblemInput& input) const {
  AslamBackendCalibrationResult result;
  result.dataset_label = input.dataset_label;
  result.baseline_protocol_label = input.baseline_protocol_label;
  result.training_split_signature = input.training_split_signature;
  result.board_pose_parameterization =
      ToString(options_.board_pose_parameterization);
  result.problem_input = input;
  result.effective_problem_input = BuildEffectiveProblemInput(input, options_);
  result.options = options_;
  result.anchor_camera = input.scene_state.camera;
  result.initial_scene_state = BuildJointSceneStateFromCalibrationSceneState(
      result.effective_problem_input.scene_state);
  result.optimized_scene_state = result.initial_scene_state;
  result.warnings = input.diagnostics_seed.warnings;

  if (options_.pixel_ray_hybrid_refinement_mode) {
    if (!std::isfinite(options_.pixel_ray_hybrid_lambda) ||
        options_.pixel_ray_hybrid_lambda < 0.0 ||
        options_.pixel_ray_hybrid_lambda > 1.0) {
      result.failure_reason =
          "Pixel-ray hybrid lambda must be finite and in [0, 1].";
      return result;
    }
    if (!std::isfinite(options_.pixel_ray_hybrid_pixel_scale_floor) ||
        options_.pixel_ray_hybrid_pixel_scale_floor <= 0.0 ||
        !std::isfinite(options_.pixel_ray_hybrid_ray_scale_floor) ||
        options_.pixel_ray_hybrid_ray_scale_floor <= 0.0) {
      result.failure_reason =
          "Pixel-ray hybrid scale floors must be finite and positive.";
      return result;
    }
    if (options_.angular_observed_ray_mode !=
        AslamBackendCalibrationOptions::AngularObservedRayMode::
            DynamicCurrentCamera) {
      result.failure_reason =
          "Pixel-ray hybrid refinement requires dynamic_current_camera rays.";
      return result;
    }
    if (options_.use_point_type_residual_split ||
        options_.angular_auxiliary_enabled) {
      result.failure_reason =
          "Pixel-ray hybrid refinement cannot be combined with point-type "
          "residual splitting or an auxiliary angular residual.";
      return result;
    }
  }

  if (result.effective_problem_input.optimization_masks.optimize_intrinsics !=
      input.optimization_masks.optimize_intrinsics) {
    AppendUniqueWarning("Backend debug mode forced pose-only optimization "
                        "(intrinsics release disabled for this run).",
                        &result.warnings);
  }
  if (options_.angular_observed_ray_mode ==
          AslamBackendCalibrationOptions::AngularObservedRayMode::
              FrozenAnchorCamera &&
      options_.residual_model == ResidualModel::SphereAngular &&
      result.effective_problem_input.optimization_masks.optimize_intrinsics) {
    AppendUniqueWarning(
        "Frozen-anchor full angular residuals keep observed rays fixed and do "
        "not provide the dynamic observed-ray intrinsics Jacobian used by B0; "
        "optimized intrinsics may be weakly constrained unless additional "
        "priors or non-angular residuals are present.",
        &result.warnings);
  }
  if (options_.debug_max_frames > 0 || options_.debug_max_nonreference_boards >= 0) {
    std::ostringstream stream;
    stream << "Backend debug subset active: frames="
           << result.effective_problem_input.measurement_dataset.accepted_frame_count
           << " boards="
           << result.effective_problem_input.measurement_dataset.accepted_board_observation_count
           << " points="
           << result.effective_problem_input.measurement_dataset.accepted_total_point_count;
    AppendUniqueWarning(stream.str(), &result.warnings);
  }

  JointMeasurementBuildResult measurement_result =
      BuildMeasurementResult(result.effective_problem_input.measurement_dataset,
                             result.effective_problem_input.reference_board_id);
  if (!measurement_result.success) {
    result.failure_reason = measurement_result.failure_reason;
    return result;
  }

  ConsistencyWeightSummary consistency_weight_summary =
      ComputeConsistencyWeightSummary(result.effective_problem_input,
                                      measurement_result,
                                      result.initial_scene_state,
                                      options_);
  if (options_.multi_board_consistency_weighting && consistency_weight_summary.success) {
    ApplyConsistencyWeightSummaryToDataset(
        consistency_weight_summary,
        &result.effective_problem_input.measurement_dataset,
        options_);
    measurement_result =
        BuildMeasurementResult(result.effective_problem_input.measurement_dataset,
                               result.effective_problem_input.reference_board_id);
    if (!measurement_result.success) {
      result.failure_reason = measurement_result.failure_reason;
      return result;
    }
  }
  result.consistency_observation_count =
      consistency_weight_summary.observation_count;
  result.consistency_successful_observation_count =
      consistency_weight_summary.successful_observation_count;
  result.consistency_downweighted_observation_count =
      consistency_weight_summary.downweighted_observation_count;
  result.consistency_hard_rejected_observation_count =
      consistency_weight_summary.hard_rejected_observation_count;
  result.consistency_mean_weight =
      consistency_weight_summary.mean_consistency_weight;
  result.consistency_min_applied_weight =
      consistency_weight_summary.min_consistency_weight;
  result.consistency_max_translation_error_mm =
      consistency_weight_summary.max_translation_error_mm;
  result.consistency_max_rotation_error_deg =
      consistency_weight_summary.max_rotation_error_deg;
  result.consistency_observation_summaries =
      consistency_weight_summary.observations;

  const JointReprojectionCostOptions frontend_cost_options =
      MakeCostOptionsForBackendResidualEvaluation(options_);

  JointResidualEvaluationOptions residual_options;
  residual_options.top_k = 10;
  residual_options.cost_options = frontend_cost_options;
  const JointReprojectionResidualEvaluator residual_evaluator(residual_options);
  result.initial_residual =
      residual_evaluator.Evaluate(measurement_result, result.initial_scene_state);
  if (!result.initial_residual.success) {
    result.failure_reason = result.initial_residual.failure_reason;
    result.warnings.insert(result.warnings.end(),
                           result.initial_residual.warnings.begin(),
                           result.initial_residual.warnings.end());
    return result;
  }

  const std::string family = result.initial_scene_state.camera.NormalizedFamilyString();
  bool backend_success = false;
  if (family == "ds-none") {
    backend_success = ExecuteBackendOptimization<DsGeometry>(
        &result, measurement_result, residual_evaluator, frontend_cost_options);
  } else if (family == "eucm-none") {
    backend_success = ExecuteBackendOptimization<EucmGeometry>(
        &result, measurement_result, residual_evaluator, frontend_cost_options);
  } else if (family == "pinhole-equi") {
    backend_success = ExecuteBackendOptimization<PinholeEquiGeometry>(
        &result, measurement_result, residual_evaluator, frontend_cost_options);
  } else if (family == "omni-radtan") {
    backend_success = ExecuteBackendOptimization<MeiGeometry>(
        &result, measurement_result, residual_evaluator, frontend_cost_options);
  } else if (family == "omni-none") {
    backend_success = ExecuteBackendOptimization<OmniNoneGeometry>(
        &result, measurement_result, residual_evaluator, frontend_cost_options);
  } else {
    result.failure_reason = "Unsupported backend camera family: " + family;
    return result;
  }
  if (!backend_success) {
    return result;
  }

  result.success = true;
  result.warnings.insert(result.warnings.end(),
                         measurement_result.warnings.begin(),
                         measurement_result.warnings.end());
  return result;
}

void WriteAslamBackendCalibrationSummary(
    const std::string& path,
    const AslamBackendCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "dataset_label: " << result.dataset_label << "\n";
  output << "baseline_protocol_label: " << result.baseline_protocol_label << "\n";
  output << "training_split_signature: " << result.training_split_signature << "\n";
  output << "board_pose_parameterization: "
         << result.board_pose_parameterization << "\n";
  output << "uses_reference_chain_board_layout: "
         << (result.options.board_pose_parameterization ==
                     AslamBackendCalibrationOptions::
                         BoardPoseParameterization::ReferenceChain
                 ? 1
                 : 0)
         << "\n";
  output << "uses_independent_frame_board_poses: "
         << (result.options.board_pose_parameterization ==
                     AslamBackendCalibrationOptions::
                         BoardPoseParameterization::IndependentFrameBoardPose
                 ? 1
                 : 0)
         << "\n";
  output << "effective_frame_count: "
         << result.effective_problem_input.measurement_dataset.accepted_frame_count << "\n";
  output << "effective_board_observation_count: "
         << result.effective_problem_input.measurement_dataset.accepted_board_observation_count
         << "\n";
  output << "effective_total_point_count: "
         << result.effective_problem_input.measurement_dataset.accepted_total_point_count
         << "\n";
  output << "design_variable_count: " << result.design_variable_count << "\n";
  output << "error_term_count: " << result.error_term_count << "\n";
  output << "initial_overall_rmse: " << result.initial_residual.overall_rmse << "\n";
  output << "initial_outer_only_rmse: " << result.initial_residual.outer_only_rmse << "\n";
  output << "initial_internal_only_rmse: " << result.initial_residual.internal_only_rmse << "\n";
  output << "optimized_overall_rmse: " << result.optimized_residual.overall_rmse << "\n";
  output << "optimized_outer_only_rmse: " << result.optimized_residual.outer_only_rmse << "\n";
  output << "optimized_internal_only_rmse: "
         << result.optimized_residual.internal_only_rmse << "\n";
  output << "anchor_camera_xi: " << result.anchor_camera.xi << "\n";
  output << "anchor_camera_alpha: " << result.anchor_camera.alpha << "\n";
  output << "anchor_camera_fu: " << result.anchor_camera.fu << "\n";
  output << "anchor_camera_fv: " << result.anchor_camera.fv << "\n";
  output << "anchor_camera_cu: " << result.anchor_camera.cu << "\n";
  output << "anchor_camera_cv: " << result.anchor_camera.cv << "\n";
  output << "optimized_camera_xi: " << result.optimized_scene_state.camera.xi << "\n";
  output << "optimized_camera_alpha: " << result.optimized_scene_state.camera.alpha << "\n";
  output << "optimized_camera_fu: " << result.optimized_scene_state.camera.fu << "\n";
  output << "optimized_camera_fv: " << result.optimized_scene_state.camera.fv << "\n";
  output << "optimized_camera_cu: " << result.optimized_scene_state.camera.cu << "\n";
  output << "optimized_camera_cv: " << result.optimized_scene_state.camera.cv << "\n";
  output << "backend_max_iterations: " << result.options.max_iterations << "\n";
  output << "backend_convergence_delta_j: " << result.options.convergence_delta_j << "\n";
  output << "backend_convergence_delta_x: " << result.options.convergence_delta_x << "\n";
  output << "backend_linear_solver: " << result.options.linear_solver << "\n";
  output << "backend_export_cost_parity_diagnostics: "
         << (result.options.export_cost_parity_diagnostics ? 1 : 0) << "\n";
  output << "backend_run_jacobian_consistency_check: "
         << (result.options.run_jacobian_consistency_check ? 1 : 0) << "\n";
  output << "backend_internal_anisotropic_weight_mode: "
         << result.options.internal_anisotropic_weight_mode << "\n";
  output << "backend_internal_anisotropic_x_scale: "
         << result.options.internal_anisotropic_x_scale << "\n";
  output << "backend_internal_anisotropic_y_scale: "
         << result.options.internal_anisotropic_y_scale << "\n";
  output << "backend_observation_role_weight_mode: "
         << result.options.observation_role_weight_mode << "\n";
  output << "backend_internal_role_budget_when_mixed: "
         << result.options.internal_role_budget_when_mixed << "\n";
  output << "backend_polar_angle_weight_mode: "
         << ToString(result.options.polar_angle_weight_mode) << "\n";
  output << "backend_polar_angle_weight_bin_edges_deg: ";
  for (std::size_t i = 0; i < result.options.polar_angle_weight_bin_edges_deg.size(); ++i) {
    if (i > 0) {
      output << ",";
    }
    output << result.options.polar_angle_weight_bin_edges_deg[i];
  }
  output << "\n";
  output << "backend_polar_angle_weight_fixed_bin_scales: ";
  for (std::size_t i = 0; i < result.options.polar_angle_weight_fixed_bin_scales.size(); ++i) {
    if (i > 0) {
      output << ",";
    }
    output << result.options.polar_angle_weight_fixed_bin_scales[i];
  }
  output << "\n";
  output << "backend_polar_angle_weight_adaptive_sigma_reference_deg: "
         << result.options.polar_angle_weight_adaptive_sigma_reference_deg << "\n";
  output << "backend_polar_angle_weight_adaptive_sigma_growth: "
         << result.options.polar_angle_weight_adaptive_sigma_growth << "\n";
  output << "backend_polar_angle_weight_min_scale: "
         << result.options.polar_angle_weight_min_scale << "\n";
  output << "backend_multi_board_consistency_weighting: "
         << (result.options.multi_board_consistency_weighting ? 1 : 0) << "\n";
  output << "backend_consistency_pose_source: "
         << result.options.consistency_pose_source << "\n";
  output << "backend_consistency_weight_mode: "
         << ToString(result.options.consistency_weight_mode) << "\n";
  output << "backend_consistency_translation_sigma_mm: "
         << result.options.consistency_translation_sigma_mm << "\n";
  output << "backend_consistency_rotation_sigma_deg: "
         << result.options.consistency_rotation_sigma_deg << "\n";
  output << "backend_consistency_min_weight: "
         << result.options.consistency_min_weight << "\n";
  output << "backend_consistency_apply_to_outer: "
         << (result.options.consistency_apply_to_outer ? 1 : 0) << "\n";
  output << "backend_consistency_apply_to_internal: "
         << (result.options.consistency_apply_to_internal ? 1 : 0) << "\n";
  output << "backend_consistency_hard_reject_enabled: "
         << (result.options.consistency_hard_reject_enabled ? 1 : 0) << "\n";
  output << "backend_consistency_hard_reject_translation_mm: "
         << result.options.consistency_hard_reject_translation_mm << "\n";
  output << "backend_consistency_hard_reject_rotation_deg: "
         << result.options.consistency_hard_reject_rotation_deg << "\n";
  output << "backend_consistency_hard_reject_residual_px: "
         << result.options.consistency_hard_reject_residual_px << "\n";
  output << "backend_consistency_dump_weight_summary: "
         << (result.options.consistency_dump_weight_summary ? 1 : 0) << "\n";
  output << "backend_board_pose_prior_enabled: "
         << (result.options.board_pose_prior_enabled ? 1 : 0) << "\n";
  output << "backend_board_pose_prior_translation_sigma_mm: "
         << result.options.board_pose_prior_translation_sigma_mm << "\n";
  output << "backend_board_pose_prior_rotation_sigma_deg: "
         << result.options.board_pose_prior_rotation_sigma_deg << "\n";
  output << "backend_board_pose_prior_count: "
         << result.board_pose_prior_count << "\n";
  output << "backend_residual_model: "
         << ToString(result.options.residual_model) << "\n";
  output << "backend_effective_residual_model: "
         << (result.options.pixel_ray_hybrid_refinement_mode
                 ? "pixel_ray_hybrid_final_refinement"
                 : ToString(result.options.residual_model))
         << "\n";
  output << "backend_residual_metric_unit: "
         << (result.options.pixel_ray_hybrid_refinement_mode
                 ? "normalized_pixel_ray_4d"
                 : result.options.angular_local_whitening_enabled
                 ? "normalized"
                 : (result.options.residual_model == ResidualModel::ImagePlane
                 ? "px"
                 : (result.options.residual_model == ResidualModel::SphereAngular ||
                            result.options.residual_model ==
                                ResidualModel::NormalizedSphereAngular
                        ? "rad"
                        : (result.options.residual_model == ResidualModel::Chordal
                               ? "unit_bearing_chord"
                               : "px_equivalent"))))
         << "\n";
  output << "backend_hybrid_angular_threshold_deg: "
         << result.options.hybrid_angular_threshold_deg << "\n";
  output << "backend_use_point_type_residual_split: "
         << (result.options.use_point_type_residual_split ? 1 : 0) << "\n";
  output << "backend_outer_residual_model: "
         << ToString(result.options.outer_residual_model) << "\n";
  output << "backend_internal_residual_model: "
         << ToString(result.options.internal_residual_model) << "\n";
  output << "backend_angular_auxiliary_enabled: "
         << (result.options.angular_auxiliary_enabled ? 1 : 0) << "\n";
  output << "backend_angular_auxiliary_weight: "
         << result.options.angular_auxiliary_weight << "\n";
  output << "backend_angular_auxiliary_normalized: "
         << (result.options.angular_auxiliary_normalized ? 1 : 0) << "\n";
  output << "backend_angular_auxiliary_apply_to_outer: "
         << (result.options.angular_auxiliary_apply_to_outer ? 1 : 0) << "\n";
  output << "backend_angular_auxiliary_apply_to_internal: "
         << (result.options.angular_auxiliary_apply_to_internal ? 1 : 0) << "\n";
  output << "backend_polar_continuous_hybrid_threshold_deg: "
         << result.options.polar_continuous_hybrid_threshold_deg << "\n";
  output << "backend_polar_continuous_hybrid_temperature_deg: "
         << result.options.polar_continuous_hybrid_temperature_deg << "\n";
  output << "backend_normalized_angular_reference_sigma_px: "
         << result.options.normalized_angular_reference_sigma_px << "\n";
  output << "backend_normalized_angular_min_sigma_rad: "
         << result.options.normalized_angular_min_sigma_rad << "\n";
  output << "backend_normalized_angular_max_weight_scale: "
         << result.options.normalized_angular_max_weight_scale << "\n";
  output << "backend_pixel_residual_weight: "
         << result.options.pixel_residual_weight << "\n";
  output << "backend_chordal_residual_weight: "
         << result.options.chordal_residual_weight << "\n";
  output << "backend_pixel_ray_hybrid_refinement_mode: "
         << (result.options.pixel_ray_hybrid_refinement_mode ? 1 : 0) << "\n";
  output << "backend_pixel_ray_hybrid_lambda: "
         << result.options.pixel_ray_hybrid_lambda << "\n";
  output << "backend_pixel_ray_hybrid_pixel_scale_floor: "
         << result.options.pixel_ray_hybrid_pixel_scale_floor << "\n";
  output << "backend_pixel_ray_hybrid_ray_scale_floor: "
         << result.options.pixel_ray_hybrid_ray_scale_floor << "\n";
  output << "backend_pixel_ray_hybrid_huber_delta: "
         << result.options.pixel_ray_hybrid_huber_delta << "\n";
  output << "backend_angular_use_normalize_jacobian: "
         << (result.options.angular_use_normalize_jacobian ? 1 : 0)
         << "\n";
  output << "backend_angular_local_whitening: "
         << (result.options.angular_local_whitening_enabled ? 1 : 0) << "\n";
  output << "backend_angular_local_whitening_pixel_sigma_px: "
         << result.options.angular_local_whitening_pixel_sigma_px << "\n";
  output << "backend_angular_local_whitening_covariance_damping: "
         << result.options.angular_local_whitening_covariance_damping << "\n";
  output << "backend_angular_local_whitening_min_sigma_rad: "
         << result.options.angular_local_whitening_min_sigma_rad << "\n";
  output << "backend_angular_local_whitening_max_weight: "
         << result.options.angular_local_whitening_max_weight << "\n";
  output << "backend_angular_observed_ray_mode: "
         << ToString(result.options.angular_observed_ray_mode) << "\n";
  output << "angular_observed_ray_anchor_camera_xi: "
         << result.anchor_camera.xi << "\n";
  output << "angular_observed_ray_anchor_camera_alpha: "
         << result.anchor_camera.alpha << "\n";
  output << "angular_observed_ray_anchor_camera_fu: "
         << result.anchor_camera.fu << "\n";
  output << "angular_observed_ray_anchor_camera_fv: "
         << result.anchor_camera.fv << "\n";
  output << "angular_observed_ray_anchor_camera_cu: "
         << result.anchor_camera.cu << "\n";
  output << "angular_observed_ray_anchor_camera_cv: "
         << result.anchor_camera.cv << "\n";
  output << "backend_enable_angular_residual_diagnostics: "
         << (result.options.enable_angular_residual_diagnostics ? 1 : 0) << "\n";
  output << "constructed_image_plane_residual_count: "
         << result.residual_block_construction.image_plane_residual_count << "\n";
  output << "constructed_angular_residual_count: "
         << result.residual_block_construction.angular_residual_count << "\n";
  output << "constructed_chordal_residual_count: "
         << result.residual_block_construction.chordal_residual_count << "\n";
  output << "constructed_pixel_ray_hybrid_residual_count: "
         << result.residual_block_construction.pixel_ray_hybrid_residual_count
         << "\n";
  output << "constructed_angular_auxiliary_residual_count: "
         << result.residual_block_construction.angular_auxiliary_residual_count << "\n";
  output << "constructed_outer_image_plane_residual_count: "
         << result.residual_block_construction.outer_image_plane_residual_count << "\n";
  output << "constructed_outer_angular_residual_count: "
         << result.residual_block_construction.outer_angular_residual_count << "\n";
  output << "constructed_outer_chordal_residual_count: "
         << result.residual_block_construction.outer_chordal_residual_count << "\n";
  output << "constructed_outer_pixel_ray_hybrid_residual_count: "
         << result.residual_block_construction
                .outer_pixel_ray_hybrid_residual_count
         << "\n";
  output << "constructed_outer_angular_auxiliary_residual_count: "
         << result.residual_block_construction
                .outer_angular_auxiliary_residual_count << "\n";
  output << "constructed_internal_image_plane_residual_count: "
         << result.residual_block_construction.internal_image_plane_residual_count << "\n";
  output << "constructed_internal_angular_residual_count: "
         << result.residual_block_construction.internal_angular_residual_count << "\n";
  output << "constructed_internal_chordal_residual_count: "
         << result.residual_block_construction.internal_chordal_residual_count << "\n";
  output << "constructed_internal_pixel_ray_hybrid_residual_count: "
         << result.residual_block_construction
                .internal_pixel_ray_hybrid_residual_count
         << "\n";
  output << "constructed_internal_angular_auxiliary_residual_count: "
         << result.residual_block_construction
                .internal_angular_auxiliary_residual_count << "\n";
  output << "skipped_solver_observation_count: "
         << result.residual_block_construction.skipped_solver_observation_count << "\n";
  output << "residual_type_assignment_count: "
         << result.residual_type_assignments.size() << "\n";
  output << "pixel_ray_hybrid_residual_dimension: 4\n";
  output << "pixel_ray_hybrid_scales_computed_once: "
         << (result.pixel_ray_hybrid_scales_computed_once ? 1 : 0) << "\n";
  output << "pixel_ray_hybrid_scale_source: "
         << (result.options.pixel_ray_hybrid_refinement_mode
                 ? "pixel_committed_training_state"
                 : "disabled")
         << "\n";
  output << "pixel_ray_hybrid_valid_observation_count: "
         << result.pixel_ray_hybrid_valid_observation_count << "\n";
  output << "pixel_ray_hybrid_invalid_observation_count: "
         << result.pixel_ray_hybrid_invalid_observation_count << "\n";
  output << "pixel_ray_hybrid_s_px: "
         << result.pixel_ray_hybrid_pixel_scale << "\n";
  output << "pixel_ray_hybrid_s_ray: "
         << result.pixel_ray_hybrid_ray_scale << "\n";
  output << "pixel_ray_hybrid_pixel_objective_weight: "
         << (1.0 - result.options.pixel_ray_hybrid_lambda) << "\n";
  output << "pixel_ray_hybrid_ray_objective_weight: "
         << result.options.pixel_ray_hybrid_lambda << "\n";
  output << "pixel_ray_hybrid_polar_adaptive_enabled: "
         << (result.options.pixel_ray_hybrid_polar_adaptive_enabled ? 1 : 0)
         << "\n";
  output << "pixel_ray_hybrid_polar_adaptive_lambda_min: "
         << result.options.pixel_ray_hybrid_lambda_min << "\n";
  output << "pixel_ray_hybrid_polar_adaptive_lambda_max: "
         << result.options.pixel_ray_hybrid_lambda_max << "\n";
  output << "pixel_ray_hybrid_polar_adaptive_transition_start_deg: "
         << result.options.pixel_ray_hybrid_transition_start_deg << "\n";
  output << "pixel_ray_hybrid_polar_adaptive_transition_end_deg: "
         << result.options.pixel_ray_hybrid_transition_end_deg << "\n";
  output << "initial_overall_image_plane_rmse: "
         << result.initial_residual.overall_image_plane_rmse << "\n";
  output << "initial_overall_angular_rmse: "
         << result.initial_residual.overall_angular_rmse << "\n";
  output << "optimized_overall_image_plane_rmse: "
         << result.optimized_residual.overall_image_plane_rmse << "\n";
  output << "optimized_overall_angular_rmse: "
         << result.optimized_residual.overall_angular_rmse << "\n";
  output << "backend_consistency_observation_count: "
         << result.consistency_observation_count << "\n";
  output << "backend_consistency_successful_observation_count: "
         << result.consistency_successful_observation_count << "\n";
  output << "backend_consistency_downweighted_observation_count: "
         << result.consistency_downweighted_observation_count << "\n";
  output << "backend_consistency_hard_rejected_observation_count: "
         << result.consistency_hard_rejected_observation_count << "\n";
  output << "backend_consistency_mean_weight: "
         << result.consistency_mean_weight << "\n";
  output << "backend_consistency_min_applied_weight: "
         << result.consistency_min_applied_weight << "\n";
  output << "backend_consistency_max_translation_error_mm: "
         << result.consistency_max_translation_error_mm << "\n";
  output << "backend_consistency_max_rotation_error_deg: "
         << result.consistency_max_rotation_error_deg << "\n";
  output << "backend_debug_max_frames: " << result.options.debug_max_frames << "\n";
  output << "backend_debug_max_nonreference_boards: "
         << result.options.debug_max_nonreference_boards << "\n";
  output << "backend_force_pose_only: " << (result.options.force_pose_only ? 1 : 0) << "\n";
  output << "backend_skip_optimization: "
         << (result.options.skip_optimization ? 1 : 0) << "\n";
  output << "backend_final_state_label: "
         << (result.options.skip_optimization
                 ? "after_incremental_selection_ba"
                 : (result.options.pixel_ray_hybrid_refinement_mode
                        ? "after_optional_pixel_ray_hybrid_refinement"
                        : "unexpected_backend_optimization"))
         << "\n";
  if (result.initial_cost_parity.success) {
    output << "initial_frontend_total_cost: "
           << result.initial_cost_parity.frontend_total_cost << "\n";
    output << "initial_backend_reprojection_total_cost: "
           << result.initial_cost_parity.backend_reprojection_total_weighted_cost << "\n";
    output << "initial_backend_problem_total_cost: "
           << result.initial_cost_parity.backend_problem_total_weighted_cost << "\n";
  }
  if (result.optimized_cost_parity.success) {
    output << "optimized_frontend_total_cost: "
           << result.optimized_cost_parity.frontend_total_cost << "\n";
    output << "optimized_backend_reprojection_total_cost: "
           << result.optimized_cost_parity.backend_reprojection_total_weighted_cost << "\n";
    output << "optimized_backend_problem_total_cost: "
           << result.optimized_cost_parity.backend_problem_total_weighted_cost << "\n";
  }
  if (result.jacobian_diagnostics.success) {
    output << "jacobian_block_count: "
           << result.jacobian_diagnostics.block_diagnostics.size() << "\n";
    for (const AslamBackendJacobianBlockDiagnostics& block :
         result.jacobian_diagnostics.block_diagnostics) {
      output << "jacobian_block_label: " << block.block_label << "\n";
      output << "jacobian_block_max_abs_difference: "
             << block.max_abs_difference << "\n";
    }
  }
  for (const AslamBackendOptimizationStageSummary& stage : result.stages) {
    output << "stage_label: " << stage.stage_label << "\n";
    output << "stage_optimize_intrinsics: " << (stage.optimize_intrinsics ? 1 : 0) << "\n";
    output << "stage_max_iterations: " << stage.max_iterations << "\n";
    output << "stage_objective_start: " << stage.objective_start << "\n";
    output << "stage_objective_final: " << stage.objective_final << "\n";
    output << "stage_iterations: " << stage.iterations << "\n";
    output << "stage_failed_iterations: " << stage.failed_iterations << "\n";
    output << "stage_lm_lambda_final: " << stage.lm_lambda_final << "\n";
    output << "stage_delta_x_final: " << stage.delta_x_final << "\n";
    output << "stage_delta_j_final: " << stage.delta_j_final << "\n";
    output << "stage_linear_solver_failure: "
           << (stage.linear_solver_failure ? 1 : 0) << "\n";
  }
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteAslamBackendCostParitySummary(
    const std::string& path,
    const AslamBackendCostParityDiagnostics& diagnostics) {
  std::ofstream output(path.c_str());
  output << "success: " << (diagnostics.success ? 1 : 0) << "\n";
  output << "failure_reason: " << diagnostics.failure_reason << "\n";
  output << "stage_label: " << diagnostics.stage_label << "\n";
  output << "compared_point_count: " << diagnostics.compared_point_count << "\n";
  output << "frontend_total_squared_error: "
         << diagnostics.frontend_total_squared_error << "\n";
  output << "frontend_total_cost: " << diagnostics.frontend_total_cost << "\n";
  output << "backend_reprojection_total_raw_squared_error: "
         << diagnostics.backend_reprojection_total_raw_squared_error << "\n";
  output << "backend_reprojection_total_weighted_cost: "
         << diagnostics.backend_reprojection_total_weighted_cost << "\n";
  output << "backend_problem_total_weighted_cost: "
         << diagnostics.backend_problem_total_weighted_cost << "\n";
  output << "total_abs_weighted_cost_difference: "
         << diagnostics.total_abs_weighted_cost_difference << "\n";
  output << "max_abs_weighted_cost_difference: "
         << diagnostics.max_abs_weighted_cost_difference << "\n";
  output << "max_predicted_difference_norm: "
         << diagnostics.max_predicted_difference_norm << "\n";
  output << "max_residual_sign_consistency_norm: "
         << diagnostics.max_residual_sign_consistency_norm << "\n";
  for (const std::string& warning : diagnostics.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteAslamBackendCostParityCsv(
    const std::string& path,
    const AslamBackendCostParityDiagnostics& diagnostics) {
  std::ofstream output(path.c_str());
  output << "frame_index,frame_label,board_id,point_id,point_type,"
         << "observed_x,observed_y,"
         << "frontend_predicted_x,frontend_predicted_y,"
         << "backend_predicted_x,backend_predicted_y,"
         << "frontend_residual_x,frontend_residual_y,"
         << "backend_residual_x,backend_residual_y,"
         << "frontend_valid_projection,backend_valid_projection,"
         << "frontend_balance_weight,frontend_huber_weight,frontend_final_weight,"
         << "frontend_weighted_squared_error,"
         << "backend_inv_r_scale,backend_m_estimator_weight,"
         << "backend_raw_squared_error,backend_weighted_squared_error,"
         << "predicted_difference_norm,residual_sign_consistency_norm,"
         << "weighted_cost_difference\n";
  for (const AslamBackendPointCostParityDiagnostics& point : diagnostics.point_diagnostics) {
    output << point.frame_index << ","
           << point.frame_label << ","
           << point.board_id << ","
           << point.point_id << ","
           << ToString(point.point_type) << ","
           << point.observed_image_xy.x() << ","
           << point.observed_image_xy.y() << ","
           << point.frontend_predicted_image_xy.x() << ","
           << point.frontend_predicted_image_xy.y() << ","
           << point.backend_predicted_image_xy.x() << ","
           << point.backend_predicted_image_xy.y() << ","
           << point.frontend_residual_xy.x() << ","
           << point.frontend_residual_xy.y() << ","
           << point.backend_residual_xy.x() << ","
           << point.backend_residual_xy.y() << ","
           << (point.frontend_valid_projection ? 1 : 0) << ","
           << (point.backend_valid_projection ? 1 : 0) << ","
           << point.frontend_balance_weight << ","
           << point.frontend_huber_weight << ","
           << point.frontend_final_weight << ","
           << point.frontend_weighted_squared_error << ","
           << point.backend_inv_r_scale << ","
           << point.backend_m_estimator_weight << ","
           << point.backend_raw_squared_error << ","
           << point.backend_weighted_squared_error << ","
           << point.predicted_difference_norm << ","
           << point.residual_sign_consistency_norm << ","
           << point.weighted_cost_difference << "\n";
  }
}

void WriteAslamBackendJacobianSummary(
    const std::string& path,
    const AslamBackendJacobianDiagnostics& diagnostics) {
  std::ofstream output(path.c_str());
  output << "success: " << (diagnostics.success ? 1 : 0) << "\n";
  output << "failure_reason: " << diagnostics.failure_reason << "\n";
  output << "finite_difference_epsilon: "
         << diagnostics.finite_difference_epsilon << "\n";
  output << "objective_model: " << diagnostics.objective_model << "\n";
  for (const AslamBackendJacobianBlockDiagnostics& block :
       diagnostics.block_diagnostics) {
    output << "block_label: " << block.block_label << "\n";
    output << "block_dimension: " << block.dimension << "\n";
    output << "block_max_abs_difference: " << block.max_abs_difference << "\n";
    output << "analytic_gradient:";
    for (double value : block.analytic_gradient) {
      output << " " << value;
    }
    output << "\n";
    output << "finite_difference_gradient:";
    for (double value : block.finite_difference_gradient) {
      output << " " << value;
    }
    output << "\n";
  }
  for (const std::string& warning : diagnostics.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteAslamBackendVariableBlockInfluenceCsv(
    const std::string& path,
    const AslamBackendVariableBlockInfluenceDiagnostics& diagnostics) {
  std::ofstream output(path.c_str());
  output << "stage_label,frame_index,frame_label,board_id,point_type,"
         << "residual_family,variable_block,variable_scope,residual_count,"
         << "residual_dimension,jacobian_columns,weighted_cost,"
         << "hessian_trace,hessian_frobenius_norm,hessian_logdet,"
         << "hessian_rank_proxy,gradient_norm\n";
  for (const AslamBackendVariableBlockInfluenceEntry& entry :
       diagnostics.entries) {
    output << entry.stage_label << ","
           << entry.frame_index << ","
           << entry.frame_label << ","
           << entry.board_id << ","
           << entry.point_type << ","
           << entry.residual_family << ","
           << entry.variable_block << ","
           << entry.variable_scope << ","
           << entry.residual_count << ","
           << entry.residual_dimension << ","
           << entry.jacobian_columns << ","
           << entry.weighted_cost << ","
           << entry.hessian_trace << ","
           << entry.hessian_frobenius_norm << ","
           << entry.hessian_logdet << ","
           << entry.hessian_rank_proxy << ","
           << entry.gradient_norm << "\n";
  }
}

void WriteBackendResidualTypeAssignmentsCsv(
    const std::string& path,
    const AslamBackendCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "frame_index,frame_label,board_id,point_id,point_type,"
         << "polar_angle_deg,residual_model_requested,residual_model_effective,"
         << "angular_observation_geometry_success,"
         << "pixel_ray_hybrid_polar_adaptive,pixel_ray_hybrid_lambda,"
         << "image_plane_weight_scale,"
         << "angular_weight_scale,angular_sigma_per_pixel_rad,"
         << "normalized_angular_weight_scale,angular_auxiliary_enabled,"
         << "angular_auxiliary_normalized\n";
  for (const BackendResidualTypeAssignment& assignment :
       result.residual_type_assignments) {
    output << assignment.frame_index << ","
           << assignment.frame_label << ","
           << assignment.board_id << ","
           << assignment.point_id << ","
           << ToString(assignment.point_type) << ","
           << assignment.polar_angle_deg << ","
           << assignment.residual_model_requested << ","
           << assignment.residual_model_effective << ","
           << (assignment.angular_observation_geometry_success ? 1 : 0) << ","
           << (assignment.pixel_ray_hybrid_polar_adaptive ? 1 : 0) << ","
           << assignment.pixel_ray_hybrid_lambda << ","
           << assignment.image_plane_weight_scale << ","
           << assignment.angular_weight_scale << ","
           << assignment.angular_sigma_per_pixel_rad << ","
           << assignment.normalized_angular_weight_scale << ","
           << (assignment.angular_auxiliary_enabled ? 1 : 0) << ","
           << (assignment.angular_auxiliary_normalized ? 1 : 0)
           << "\n";
  }
}

void WriteConsistencyWeightSummary(
    const std::string& path,
    const AslamBackendCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "backend_multi_board_consistency_weighting: "
         << (result.options.multi_board_consistency_weighting ? 1 : 0) << "\n";
  output << "pose_source: " << result.options.consistency_pose_source << "\n";
  output << "observation_count: " << result.consistency_observation_count << "\n";
  output << "successful_observation_count: "
         << result.consistency_successful_observation_count << "\n";
  output << "downweighted_observation_count: "
         << result.consistency_downweighted_observation_count << "\n";
  output << "hard_rejected_observation_count: "
         << result.consistency_hard_rejected_observation_count << "\n";
  output << "mean_consistency_weight: " << result.consistency_mean_weight << "\n";
  output << "min_consistency_weight: " << result.consistency_min_applied_weight << "\n";
  output << "max_translation_error_mm: "
         << result.consistency_max_translation_error_mm << "\n";
  output << "max_rotation_error_deg: "
         << result.consistency_max_rotation_error_deg << "\n";
}

void WriteConsistencyPerBoardSummary(
    const std::string& path,
    const AslamBackendCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "board_id,support_observation_count,mean_consistency_weight,min_consistency_weight,"
         << "mean_translation_error_mm,max_translation_error_mm,mean_rotation_error_deg,"
         << "max_rotation_error_deg,hard_rejected_count\n";
  std::map<int, std::vector<ConsistencyObservationWeightSummaryEntry> > grouped;
  for (const ConsistencyObservationWeightSummaryEntry& row :
       result.consistency_observation_summaries) {
    if (!row.local_pose_refit_success) {
      continue;
    }
    grouped[row.board_id].push_back(row);
  }
  for (const auto& entry : grouped) {
    double weight_sum = 0.0;
    double min_weight = 1.0;
    double translation_sum = 0.0;
    double max_translation = 0.0;
    double rotation_sum = 0.0;
    double max_rotation = 0.0;
    int hard_rejected_count = 0;
    for (const ConsistencyObservationWeightSummaryEntry& row : entry.second) {
      weight_sum += row.consistency_weight;
      min_weight = std::min(min_weight, row.consistency_weight);
      translation_sum += row.translation_error_mm;
      max_translation = std::max(max_translation, row.translation_error_mm);
      rotation_sum += row.rotation_error_deg;
      max_rotation = std::max(max_rotation, row.rotation_error_deg);
      if (row.hard_rejected) {
        ++hard_rejected_count;
      }
    }
    const double count = static_cast<double>(entry.second.size());
    output << entry.first << ","
           << entry.second.size() << ","
           << (count > 0.0 ? weight_sum / count : 1.0) << ","
           << min_weight << ","
           << (count > 0.0 ? translation_sum / count : 0.0) << ","
           << max_translation << ","
           << (count > 0.0 ? rotation_sum / count : 0.0) << ","
           << max_rotation << ","
           << hard_rejected_count << "\n";
  }
}

void WriteConsistencyPerFrameSummary(
    const std::string& path,
    const AslamBackendCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "frame_index,support_observation_count,mean_consistency_weight,min_consistency_weight,"
         << "mean_translation_error_mm,max_translation_error_mm,mean_rotation_error_deg,"
         << "max_rotation_error_deg,worst_board_id\n";
  std::map<int, std::vector<ConsistencyObservationWeightSummaryEntry> > grouped;
  for (const ConsistencyObservationWeightSummaryEntry& row :
       result.consistency_observation_summaries) {
    if (!row.local_pose_refit_success) {
      continue;
    }
    grouped[row.frame_index].push_back(row);
  }
  for (const auto& entry : grouped) {
    double weight_sum = 0.0;
    double min_weight = 1.0;
    double translation_sum = 0.0;
    double max_translation = 0.0;
    double rotation_sum = 0.0;
    double max_rotation = 0.0;
    int worst_board_id = -1;
    for (const ConsistencyObservationWeightSummaryEntry& row : entry.second) {
      weight_sum += row.consistency_weight;
      min_weight = std::min(min_weight, row.consistency_weight);
      translation_sum += row.translation_error_mm;
      rotation_sum += row.rotation_error_deg;
      if (row.translation_error_mm >= max_translation) {
        max_translation = row.translation_error_mm;
        worst_board_id = row.board_id;
      }
      max_rotation = std::max(max_rotation, row.rotation_error_deg);
    }
    const double count = static_cast<double>(entry.second.size());
    output << entry.first << ","
           << entry.second.size() << ","
           << (count > 0.0 ? weight_sum / count : 1.0) << ","
           << min_weight << ","
           << (count > 0.0 ? translation_sum / count : 0.0) << ","
           << max_translation << ","
           << (count > 0.0 ? rotation_sum / count : 0.0) << ","
           << max_rotation << ","
           << worst_board_id << "\n";
  }
}

void WriteTopDownweightedObservations(
    const std::string& path,
    const AslamBackendCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "frame_index,frame_label,board_id,translation_error_mm,rotation_error_deg,"
         << "translation_x_mm,translation_y_mm,translation_z_mm,"
         << "rotation_x_deg,rotation_y_deg,rotation_z_deg,"
         << "residual_rmse,polar_angle_deg,consistency_weight,final_weight,"
         << "num_outer_points,num_internal_points,hard_rejected,local_pose_refit_success,"
         << "reference_pose_from_local_refit\n";
  std::vector<ConsistencyObservationWeightSummaryEntry> rows =
      result.consistency_observation_summaries;
  std::sort(rows.begin(), rows.end(),
            [](const ConsistencyObservationWeightSummaryEntry& lhs,
               const ConsistencyObservationWeightSummaryEntry& rhs) {
              if (lhs.consistency_weight != rhs.consistency_weight) {
                return lhs.consistency_weight < rhs.consistency_weight;
              }
              return lhs.translation_error_mm > rhs.translation_error_mm;
            });
  for (const ConsistencyObservationWeightSummaryEntry& row : rows) {
    output << row.frame_index << ","
           << row.frame_label << ","
           << row.board_id << ","
           << row.translation_error_mm << ","
           << row.rotation_error_deg << ","
           << row.translation_correction_mm.x() << ","
           << row.translation_correction_mm.y() << ","
           << row.translation_correction_mm.z() << ","
           << row.rotation_correction_deg.x() << ","
           << row.rotation_correction_deg.y() << ","
           << row.rotation_correction_deg.z() << ","
           << row.residual_rmse << ","
           << row.polar_angle_deg << ","
           << row.consistency_weight << ","
           << row.final_weight << ","
           << row.num_outer_points << ","
           << row.num_internal_points << ","
           << (row.hard_rejected ? 1 : 0) << ","
           << (row.local_pose_refit_success ? 1 : 0) << ","
           << (row.reference_pose_from_local_refit ? 1 : 0) << "\n";
  }
}

double ComputePercentileFromSorted(const std::vector<double>& values,
                                   double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  const double clamped = std::max(0.0, std::min(100.0, percentile));
  const double position =
      (clamped / 100.0) * static_cast<double>(values.size() - 1);
  const std::size_t lower_index =
      static_cast<std::size_t>(std::floor(position));
  const std::size_t upper_index =
      static_cast<std::size_t>(std::ceil(position));
  if (lower_index == upper_index) {
    return values[lower_index];
  }
  const double alpha = position - static_cast<double>(lower_index);
  return (1.0 - alpha) * values[lower_index] + alpha * values[upper_index];
}

AngularResidualBinStatistics BuildAngularBinStatistics(
    double bin_min_deg,
    double bin_max_deg,
    const std::vector<const JointResidualPointDiagnostics*>& points) {
  AngularResidualBinStatistics statistics;
  statistics.bin_min_deg = bin_min_deg;
  statistics.bin_max_deg = bin_max_deg;
  statistics.point_count = static_cast<int>(points.size());
  for (const JointResidualPointDiagnostics* point : points) {
    if (point->point_type == JointPointType::Outer) {
      ++statistics.outer_count;
    } else {
      ++statistics.internal_count;
    }
  }
  if (points.empty()) {
    return statistics;
  }

  std::vector<double> residual_norms;
  residual_norms.reserve(points.size());
  double squared_sum = 0.0;
  double image_plane_squared_sum = 0.0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  for (const JointResidualPointDiagnostics* point : points) {
    const double residual_norm = point->angular_residual_norm;
    residual_norms.push_back(residual_norm);
    squared_sum += residual_norm * residual_norm;
    image_plane_squared_sum += point->residual_norm * point->residual_norm;
    sum_x += point->angular_residual_xy.x();
    sum_y += point->angular_residual_xy.y();
    statistics.max_residual =
        std::max(statistics.max_residual, residual_norm);
  }
  std::sort(residual_norms.begin(), residual_norms.end());
  const double point_count = static_cast<double>(points.size());
  statistics.rmse = std::sqrt(squared_sum / point_count);
  statistics.image_plane_rmse = std::sqrt(image_plane_squared_sum / point_count);
  statistics.median_residual = ComputePercentileFromSorted(residual_norms, 50.0);
  statistics.p90_residual = ComputePercentileFromSorted(residual_norms, 90.0);
  statistics.p95_residual = ComputePercentileFromSorted(residual_norms, 95.0);
  const double mean_x = sum_x / point_count;
  const double mean_y = sum_y / point_count;
  double variance_x = 0.0;
  double variance_y = 0.0;
  for (const JointResidualPointDiagnostics* point : points) {
    const double dx = point->angular_residual_xy.x() - mean_x;
    const double dy = point->angular_residual_xy.y() - mean_y;
    variance_x += dx * dx;
    variance_y += dy * dy;
  }
  statistics.std_x = std::sqrt(variance_x / point_count);
  statistics.std_y = std::sqrt(variance_y / point_count);
  return statistics;
}

std::vector<AngularResidualBinStatistics> BuildAngularBins(
    const std::vector<JointResidualPointDiagnostics>& points,
    const std::vector<double>& bin_edges_deg,
    JointPointType* point_type_filter) {
  std::vector<AngularResidualBinStatistics> bins;
  if (bin_edges_deg.size() < 2) {
    return bins;
  }
  bins.reserve(bin_edges_deg.size() - 1);
  for (std::size_t bin_index = 1; bin_index < bin_edges_deg.size(); ++bin_index) {
    const double bin_min_deg = bin_edges_deg[bin_index - 1];
    const double bin_max_deg = bin_edges_deg[bin_index];
    std::vector<const JointResidualPointDiagnostics*> bin_points;
    for (const JointResidualPointDiagnostics& point : points) {
      if (point_type_filter != nullptr && point.point_type != *point_type_filter) {
        continue;
      }
      if (!std::isfinite(point.polar_angle_deg) ||
          !std::isfinite(point.angular_residual_norm)) {
        continue;
      }
      const bool in_bin =
          point.polar_angle_deg >= bin_min_deg &&
          (bin_index + 1 == bin_edges_deg.size()
               ? point.polar_angle_deg <= bin_max_deg
               : point.polar_angle_deg < bin_max_deg);
      if (in_bin) {
        bin_points.push_back(&point);
      }
    }
    bins.push_back(BuildAngularBinStatistics(bin_min_deg, bin_max_deg, bin_points));
  }
  return bins;
}

void WriteAngularBinTable(std::ostream& output,
                          const std::string& section_label,
                          const std::vector<AngularResidualBinStatistics>& bins) {
  output << "[" << section_label << "]\n";
  output << "bin_min_deg,bin_max_deg,point_count,outer_count,internal_count,rmse,image_plane_rmse,std_x,std_y,median_residual,p90_residual,p95_residual,max_residual\n";
  for (const AngularResidualBinStatistics& bin : bins) {
    const bool empty = bin.point_count <= 0;
    output << bin.bin_min_deg << ","
           << bin.bin_max_deg << ","
           << bin.point_count << ","
           << bin.outer_count << ","
           << bin.internal_count << ","
           << (empty ? "N/A" : std::to_string(bin.rmse)) << ","
           << (empty ? "N/A" : std::to_string(bin.image_plane_rmse)) << ","
           << (empty ? "N/A" : std::to_string(bin.std_x)) << ","
           << (empty ? "N/A" : std::to_string(bin.std_y)) << ","
           << (empty ? "N/A" : std::to_string(bin.median_residual)) << ","
           << (empty ? "N/A" : std::to_string(bin.p90_residual)) << ","
           << (empty ? "N/A" : std::to_string(bin.p95_residual)) << ","
           << (empty ? "N/A" : std::to_string(bin.max_residual)) << "\n";
  }
  output << "\n";
}

void WriteConsistencyWeightSummary(
    const std::string& path,
    const AslamBackendCalibrationResult& result);

void WriteTopDownweightedObservations(
    const std::string& path,
    const AslamBackendCalibrationResult& result);

AngularResidualDiagnosticsResult EvaluateAngularResidualDiagnostics(
    const JointResidualEvaluationResult& evaluation,
    const AngularResidualDiagnosticOptions& options) {
  AngularResidualDiagnosticsResult diagnostics;
  if (!options.enabled) {
    diagnostics.failure_reason = "Angular residual diagnostics disabled.";
    return diagnostics;
  }
  if (!evaluation.success) {
    diagnostics.failure_reason =
        "Angular residual diagnostics require a successful evaluation.";
    return diagnostics;
  }
  diagnostics.all_points_bins =
      BuildAngularBins(evaluation.point_diagnostics, options.bin_edges_deg, nullptr);
  JointPointType outer_type = JointPointType::Outer;
  diagnostics.outer_only_bins =
      BuildAngularBins(evaluation.point_diagnostics, options.bin_edges_deg, &outer_type);
  JointPointType internal_type = JointPointType::Internal;
  diagnostics.internal_only_bins =
      BuildAngularBins(evaluation.point_diagnostics, options.bin_edges_deg, &internal_type);
  double polar_angle_sum = 0.0;
  for (const JointResidualPointDiagnostics& point : evaluation.point_diagnostics) {
    if (std::isfinite(point.polar_angle_deg)) {
      if (diagnostics.finite_polar_angle_count == 0) {
        diagnostics.polar_angle_min_deg = point.polar_angle_deg;
        diagnostics.polar_angle_max_deg = point.polar_angle_deg;
      } else {
        diagnostics.polar_angle_min_deg =
            std::min(diagnostics.polar_angle_min_deg, point.polar_angle_deg);
        diagnostics.polar_angle_max_deg =
            std::max(diagnostics.polar_angle_max_deg, point.polar_angle_deg);
      }
      ++diagnostics.finite_polar_angle_count;
      polar_angle_sum += point.polar_angle_deg;
    }
    const bool uses_angular =
        point.residual_model_used == ResidualModel::SphereAngular;
    if (uses_angular) {
      ++diagnostics.angular_residual_count;
      if (point.point_type == JointPointType::Outer) {
        ++diagnostics.outer_angular_residual_count;
      } else {
        ++diagnostics.internal_angular_residual_count;
      }
    } else {
      ++diagnostics.image_plane_residual_count;
      if (point.point_type == JointPointType::Outer) {
        ++diagnostics.outer_image_plane_residual_count;
      } else {
        ++diagnostics.internal_image_plane_residual_count;
      }
    }
  }
  if (diagnostics.finite_polar_angle_count > 0) {
    diagnostics.polar_angle_mean_deg =
        polar_angle_sum /
        static_cast<double>(diagnostics.finite_polar_angle_count);
  }
  diagnostics.success = true;
  return diagnostics;
}

void WriteAngularResidualSummary(
    const std::string& path,
    const AslamBackendCalibrationResult& result,
    const AngularResidualDiagnosticsResult& diagnostics) {
  const JointResidualEvaluationResult& evaluation = result.optimized_residual;
  std::ofstream output(path.c_str());
  output << "success: " << (diagnostics.success ? 1 : 0) << "\n";
  output << "failure_reason: " << diagnostics.failure_reason << "\n";
  output << "evaluation_success: " << (evaluation.success ? 1 : 0) << "\n";
  output << "active_residual_model: "
         << ToString(result.options.residual_model) << "\n";
  output << "hybrid_angular_threshold_deg: "
         << result.options.hybrid_angular_threshold_deg << "\n";
  output << "constructed_image_plane_residual_count: "
         << result.residual_block_construction.image_plane_residual_count << "\n";
  output << "constructed_angular_residual_count: "
         << result.residual_block_construction.angular_residual_count << "\n";
  output << "constructed_angular_auxiliary_residual_count: "
         << result.residual_block_construction.angular_auxiliary_residual_count
         << "\n";
  output << "constructed_outer_image_plane_residual_count: "
         << result.residual_block_construction.outer_image_plane_residual_count << "\n";
  output << "constructed_outer_angular_residual_count: "
         << result.residual_block_construction.outer_angular_residual_count << "\n";
  output << "constructed_outer_angular_auxiliary_residual_count: "
         << result.residual_block_construction
                .outer_angular_auxiliary_residual_count << "\n";
  output << "constructed_internal_image_plane_residual_count: "
         << result.residual_block_construction.internal_image_plane_residual_count
         << "\n";
  output << "constructed_internal_angular_residual_count: "
         << result.residual_block_construction.internal_angular_residual_count
         << "\n";
  output << "constructed_internal_angular_auxiliary_residual_count: "
         << result.residual_block_construction
                .internal_angular_auxiliary_residual_count << "\n";
  output << "skipped_solver_observation_count: "
         << result.residual_block_construction.skipped_solver_observation_count
         << "\n";
  output << "image_plane_residual_count: "
         << diagnostics.image_plane_residual_count << "\n";
  output << "angular_residual_count: "
         << diagnostics.angular_residual_count << "\n";
  output << "outer_image_plane_residual_count: "
         << diagnostics.outer_image_plane_residual_count << "\n";
  output << "outer_angular_residual_count: "
         << diagnostics.outer_angular_residual_count << "\n";
  output << "internal_image_plane_residual_count: "
         << diagnostics.internal_image_plane_residual_count << "\n";
  output << "internal_angular_residual_count: "
         << diagnostics.internal_angular_residual_count << "\n";
  output << "finite_polar_angle_count: "
         << diagnostics.finite_polar_angle_count << "\n";
  output << "polar_angle_min_deg: " << diagnostics.polar_angle_min_deg << "\n";
  output << "polar_angle_mean_deg: " << diagnostics.polar_angle_mean_deg << "\n";
  output << "polar_angle_max_deg: " << diagnostics.polar_angle_max_deg << "\n";
  output << "overall_image_plane_rmse: " << evaluation.overall_image_plane_rmse << "\n";
  output << "overall_angular_rmse: " << evaluation.overall_angular_rmse << "\n";
  output << "outer_only_image_plane_rmse: " << evaluation.outer_only_image_plane_rmse << "\n";
  output << "outer_only_angular_rmse: " << evaluation.outer_only_angular_rmse << "\n";
  output << "internal_only_image_plane_rmse: "
         << evaluation.internal_only_image_plane_rmse << "\n";
  output << "internal_only_angular_rmse: " << evaluation.internal_only_angular_rmse << "\n";
  WriteAngularBinTable(output, "all_points_bins", diagnostics.all_points_bins);
  WriteAngularBinTable(output, "outer_only_bins", diagnostics.outer_only_bins);
  WriteAngularBinTable(output, "internal_only_bins", diagnostics.internal_only_bins);
}

void WriteAngularResidualBinsCsv(
    const std::string& path,
    const AngularResidualDiagnosticsResult& diagnostics) {
  std::ofstream output(path.c_str());
  output << "subset,bin_min_deg,bin_max_deg,point_count,outer_count,internal_count,rmse,image_plane_rmse,std_x,std_y,median_residual,p90_residual,p95_residual,max_residual\n";
  const auto write_bins = [&output](const std::string& subset,
                                    const std::vector<AngularResidualBinStatistics>& bins) {
    for (const AngularResidualBinStatistics& bin : bins) {
      const bool empty = bin.point_count <= 0;
      output << subset << ","
             << bin.bin_min_deg << ","
             << bin.bin_max_deg << ","
             << bin.point_count << ","
             << bin.outer_count << ","
             << bin.internal_count << ","
             << (empty ? "N/A" : std::to_string(bin.rmse)) << ","
             << (empty ? "N/A" : std::to_string(bin.image_plane_rmse)) << ","
             << (empty ? "N/A" : std::to_string(bin.std_x)) << ","
             << (empty ? "N/A" : std::to_string(bin.std_y)) << ","
             << (empty ? "N/A" : std::to_string(bin.median_residual)) << ","
             << (empty ? "N/A" : std::to_string(bin.p90_residual)) << ","
             << (empty ? "N/A" : std::to_string(bin.p95_residual)) << ","
             << (empty ? "N/A" : std::to_string(bin.max_residual)) << "\n";
    }
  };
  write_bins("all_points", diagnostics.all_points_bins);
  write_bins("outer_only", diagnostics.outer_only_bins);
  write_bins("internal_only", diagnostics.internal_only_bins);
}

void WriteAngularResidualPointSelectionCsv(
    const std::string& path,
    const JointResidualEvaluationResult& evaluation) {
  std::ofstream output(path.c_str());
  output << "frame_index,frame_label,board_id,point_id,point_type,source_kind,"
         << "observed_x,observed_y,predicted_x,predicted_y,"
         << "polar_angle_deg,selected_residual_type,"
         << "image_residual_x,image_residual_y,image_residual_norm,"
         << "angular_residual_x,angular_residual_y,angular_residual_norm,"
         << "used_in_solver\n";
  for (const JointResidualPointDiagnostics& point : evaluation.point_diagnostics) {
    output << point.frame_index << ","
           << point.frame_label << ","
           << point.board_id << ","
           << point.point_id << ","
           << ToString(point.point_type) << ","
           << ToString(point.source_kind) << ","
           << point.observed_image_xy.x() << ","
           << point.observed_image_xy.y() << ","
           << point.predicted_image_xy.x() << ","
           << point.predicted_image_xy.y() << ","
           << point.polar_angle_deg << ","
           << (point.residual_model_used == ResidualModel::SphereAngular
                   ? "sphere_angular"
                   : "image_plane")
           << ","
           << point.residual_xy.x() << ","
           << point.residual_xy.y() << ","
           << point.residual_norm << ","
           << point.angular_residual_xy.x() << ","
           << point.angular_residual_xy.y() << ","
           << point.angular_residual_norm << ","
           << (point.used_in_solver ? 1 : 0) << "\n";
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
