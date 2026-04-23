#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
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

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

using DsGeometry = aslam::cameras::DoubleSphereCameraGeometry;
using DsProjection = aslam::cameras::DoubleSphereProjection<aslam::cameras::NoDistortion>;
using CameraDv = aslam::backend::CameraDesignVariable<DsGeometry>;

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

OuterBootstrapCameraIntrinsics GeometryToIntrinsics(const DsGeometry& geometry) {
  OuterBootstrapCameraIntrinsics intrinsics;
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

boost::shared_ptr<DsGeometry> MakeGeometry(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  DsProjection projection(intrinsics.xi, intrinsics.alpha, intrinsics.fu, intrinsics.fv,
                          intrinsics.cu, intrinsics.cv, intrinsics.resolution.width,
                          intrinsics.resolution.height);
  return boost::shared_ptr<DsGeometry>(
      new DsGeometry(projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask()));
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

double ComputeBalanceWeight(const ObservationBudget& budget,
                            JointPointType point_type) {
  const bool has_outer = budget.outer_count > 0;
  const bool has_internal = budget.internal_count > 0;
  double type_budget = 1.0;
  int type_count = 1;
  if (has_outer && has_internal) {
    type_budget = 0.5;
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

struct DsReprojectionDebugSample {
  bool valid_projection = false;
  Eigen::Vector2d predicted_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d residual_xy = Eigen::Vector2d::Zero();
  double residual_norm = 0.0;
  double backend_inv_r_scale = 0.0;
  double backend_m_estimator_weight = 0.0;
  double backend_raw_squared_error = 0.0;
  double backend_weighted_squared_error = 0.0;
};

class DsIntrinsicsAnchorError : public aslam::backend::ErrorTermFs<6> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using projection_dv_t = aslam::backend::DesignVariableAdapter<DsProjection>;

  DsIntrinsicsAnchorError(
      const boost::shared_ptr<DsGeometry>& camera_geometry,
      const boost::shared_ptr<projection_dv_t>& projection_dv,
      const Eigen::Matrix<double, 6, 1>& anchor,
      const Eigen::Matrix<double, 6, 1>& prior_weight)
      : camera_geometry_(camera_geometry),
        projection_dv_(projection_dv),
        anchor_(anchor) {
    if (camera_geometry_ == nullptr || projection_dv_ == nullptr) {
      throw std::runtime_error("DsIntrinsicsAnchorError requires valid camera data.");
    }
    const Eigen::Matrix<double, 6, 6> inverse_covariance = prior_weight.asDiagonal();
    parent_t::setInvR(inverse_covariance);
    parent_t::setDesignVariables(projection_dv_.get());
  }

 protected:
  double evaluateErrorImplementation() override {
    Eigen::MatrixXd parameters_matrix;
    camera_geometry_->projection().getParameters(parameters_matrix);
    Eigen::Matrix<double, 6, 1> parameters = parameters_matrix;
    parent_t::setError(parameters - anchor_);
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    jacobians.add(projection_dv_.get(), Eigen::Matrix<double, 6, 6>::Identity());
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<6>;

  boost::shared_ptr<DsGeometry> camera_geometry_;
  boost::shared_ptr<projection_dv_t> projection_dv_;
  Eigen::Matrix<double, 6, 1> anchor_;
};

class DsReprojectionError : public aslam::backend::ErrorTermFs<2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using measurement_t = Eigen::Vector2d;
  using inverse_covariance_t = Eigen::Matrix2d;

  DsReprojectionError(const measurement_t& measurement,
                      double balance_weight,
                      double huber_delta_pixels,
                      bool use_huber_loss,
                      const aslam::backend::HomogeneousExpression& point_camera,
                      const CameraDv& camera_dv,
                      double invalid_projection_penalty_pixels)
      : measurement_(measurement),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        balance_weight_(std::max(0.0, balance_weight)),
        huber_delta_pixels_(huber_delta_pixels),
        invalid_projection_penalty_pixels_(invalid_projection_penalty_pixels) {
    const inverse_covariance_t inverse_covariance =
        balance_weight_ * inverse_covariance_t::Identity();
    parent_t::setInvR(inverse_covariance);
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

  DsReprojectionDebugSample BuildDebugSample() const {
    DsReprojectionDebugSample sample;
    sample.backend_inv_r_scale = balance_weight_;

    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    bool valid_projection = false;
    Eigen::Vector2d residual = ComputeResidual(&predicted, &valid_projection);
    sample.valid_projection = valid_projection;
    sample.predicted_image_xy = predicted;
    sample.residual_xy = residual;
    sample.residual_norm = residual.norm();
    sample.backend_raw_squared_error = balance_weight_ * residual.squaredNorm();
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
    DsReprojectionError* mutable_this = const_cast<DsReprojectionError*>(this);
    mutable_this->evaluateJacobiansFiniteDifference(jacobians);
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
  CameraDv camera_dv_;
  double balance_weight_ = 1.0;
  double huber_delta_pixels_ = 0.0;
  double invalid_projection_penalty_pixels_ = 100.0;
};

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

double EvaluateSelectedErrorTermsObjective(
    const std::set<aslam::backend::ErrorTerm*>& error_terms) {
  double total_cost = 0.0;
  for (aslam::backend::ErrorTerm* error_term : error_terms) {
    total_cost += error_term->evaluateError();
  }
  return total_cost;
}

struct DesignVariableIndexState {
  aslam::backend::DesignVariable* design_variable = nullptr;
  int block_index = -1;
  int column_base = -1;
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
    const double positive_cost = EvaluateSelectedErrorTermsObjective(attached_error_terms);
    design_variable->revertUpdate();

    Eigen::VectorXd negative_step = Eigen::VectorXd::Zero(dimension);
    negative_step[index] = -finite_difference_epsilon;
    design_variable->update(negative_step.data(), dimension);
    const double negative_cost = EvaluateSelectedErrorTermsObjective(attached_error_terms);
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

AslamBackendJacobianDiagnostics RunJacobianDiagnostics(
    const boost::shared_ptr<aslam::backend::OptimizationProblem>& problem,
    CameraDv* camera_dv,
    const std::map<int, PoseVariableState>& frame_variables,
    const std::map<int, PoseVariableState>& board_variables,
    double finite_difference_epsilon) {
  AslamBackendJacobianDiagnostics diagnostics;
  diagnostics.finite_difference_epsilon = finite_difference_epsilon;

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

AslamBackendCostParityDiagnostics EvaluateCostParityDiagnostics(
    const std::string& stage_label,
    const boost::shared_ptr<aslam::backend::OptimizationProblem>& problem,
    const JointMeasurementBuildResult& measurement_result,
    const JointReprojectionSceneState& scene_state,
    const std::vector<boost::shared_ptr<DsReprojectionError> >& reprojection_errors,
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
    const DsReprojectionDebugSample backend_point =
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

AslamBackendOptimizationStageSummary RunOptimizationStage(
    const std::string& stage_label,
    bool optimize_intrinsics,
    int max_iterations,
    const AslamBackendCalibrationOptions& options,
    const boost::shared_ptr<aslam::backend::OptimizationProblem>& problem,
    CameraDv* camera_dv) {
  if (problem == nullptr || camera_dv == nullptr) {
    throw std::runtime_error("RunOptimizationStage requires a valid problem and camera DV.");
  }

  camera_dv->setActive(optimize_intrinsics, false, false);

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
  result.problem_input = input;
  result.effective_problem_input = BuildEffectiveProblemInput(input, options_);
  result.options = options_;
  result.anchor_camera = input.scene_state.camera;
  result.initial_scene_state = BuildJointSceneStateFromCalibrationSceneState(
      result.effective_problem_input.scene_state);
  result.optimized_scene_state = result.initial_scene_state;
  result.warnings = input.diagnostics_seed.warnings;

  if (result.effective_problem_input.optimization_masks.optimize_intrinsics !=
      input.optimization_masks.optimize_intrinsics) {
    AppendUniqueWarning("Backend debug mode forced pose-only optimization "
                        "(intrinsics release disabled for this run).",
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

  const JointMeasurementBuildResult measurement_result =
      BuildMeasurementResult(result.effective_problem_input.measurement_dataset,
                             result.effective_problem_input.reference_board_id);
  if (!measurement_result.success) {
    result.failure_reason = measurement_result.failure_reason;
    return result;
  }

  JointReprojectionCostOptions frontend_cost_options;
  frontend_cost_options.outer_huber_delta_pixels =
      options_.use_huber_loss ? options_.outer_huber_delta_pixels : 0.0;
  frontend_cost_options.internal_huber_delta_pixels =
      options_.use_huber_loss ? options_.internal_huber_delta_pixels : 0.0;
  frontend_cost_options.enable_invalid_projection_penalty =
      options_.invalid_projection_penalty_pixels > 0.0;
  frontend_cost_options.invalid_projection_penalty_pixels =
      options_.invalid_projection_penalty_pixels;

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

  boost::shared_ptr<DsGeometry> camera_geometry =
      MakeGeometry(result.initial_scene_state.camera);
  CameraDv camera_dv(camera_geometry);
  camera_dv.setActive(false, false, false);

  boost::shared_ptr<aslam::backend::OptimizationProblem> problem(
      new aslam::backend::OptimizationProblem);
  problem->addDesignVariable(camera_dv.projectionDesignVariable());
  problem->addDesignVariable(camera_dv.distortionDesignVariable());
  problem->addDesignVariable(camera_dv.shutterDesignVariable());

  std::map<int, PoseVariableState> frame_variables;
  std::map<int, PoseVariableState> board_variables;

  for (const JointSceneFrameState& frame_state : result.initial_scene_state.frames) {
    if (!frame_state.initialized) {
      continue;
    }
    PoseVariableState& variable = frame_variables[frame_state.frame_index];
    variable.transform = sm::kinematics::Transformation(frame_state.T_camera_reference);
    variable.expression = aslam::backend::transformationToExpression(
        variable.transform, variable.rotation_dv, variable.translation_dv);
    const bool active =
        result.effective_problem_input.optimization_masks.optimize_frame_poses;
    variable.rotation_dv->setActive(active);
    variable.translation_dv->setActive(active);
    problem->addDesignVariable(variable.rotation_dv);
    problem->addDesignVariable(variable.translation_dv);
  }

  for (const JointSceneBoardState& board_state : result.initial_scene_state.boards) {
    if (!board_state.initialized ||
        board_state.board_id == result.effective_problem_input.reference_board_id) {
      continue;
    }
    PoseVariableState& variable = board_variables[board_state.board_id];
    variable.transform = sm::kinematics::Transformation(board_state.T_reference_board);
    variable.expression = aslam::backend::transformationToExpression(
        variable.transform, variable.rotation_dv, variable.translation_dv);
    const bool active =
        result.effective_problem_input.optimization_masks.optimize_board_poses;
    variable.rotation_dv->setActive(active);
    variable.translation_dv->setActive(active);
    problem->addDesignVariable(variable.rotation_dv);
    problem->addDesignVariable(variable.translation_dv);
  }

  const std::map<std::pair<int, int>, ObservationBudget> observation_budgets =
      BuildObservationBudgets(measurement_result);
  const aslam::backend::TransformationExpression identity_transform(
      Eigen::Matrix4d::Identity());
  int skipped_point_count = 0;
  std::vector<boost::shared_ptr<DsReprojectionError> > reprojection_errors;
  reprojection_errors.reserve(
      static_cast<std::size_t>(measurement_result.used_total_point_count));
  for (const JointPointObservation& observation : measurement_result.solver_observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    const auto frame_it = frame_variables.find(observation.frame_index);
    if (frame_it == frame_variables.end()) {
      ++skipped_point_count;
      continue;
    }

    aslam::backend::TransformationExpression board_expression = identity_transform;
    if (observation.board_id != result.effective_problem_input.reference_board_id) {
      const auto board_it = board_variables.find(observation.board_id);
      if (board_it == board_variables.end()) {
        ++skipped_point_count;
        continue;
      }
      board_expression = board_it->second.expression;
    }

    const aslam::backend::HomogeneousExpression point_board(
        observation.target_xyz_board);
    const aslam::backend::HomogeneousExpression point_reference =
        board_expression * point_board;
    const aslam::backend::HomogeneousExpression point_camera =
        frame_it->second.expression * point_reference;

    const ObservationBudget& budget =
        observation_budgets.find(std::make_pair(observation.frame_index, observation.board_id))
            ->second;
    const double balance_weight = ComputeBalanceWeight(budget, observation.point_type);
    const double huber_delta = observation.point_type == JointPointType::Outer
                                   ? options_.outer_huber_delta_pixels
                                   : options_.internal_huber_delta_pixels;
    boost::shared_ptr<DsReprojectionError> error(
        new DsReprojectionError(observation.image_xy,
                                balance_weight,
                                huber_delta,
                                options_.use_huber_loss,
                                point_camera,
                                camera_dv,
                                options_.invalid_projection_penalty_pixels));
    problem->addErrorTerm(error);
    reprojection_errors.push_back(error);
  }

  if (skipped_point_count > 0) {
    AppendUniqueWarning(
        "Skipped " + std::to_string(skipped_point_count) +
            " solver observations while building the backend problem.",
        &result.warnings);
  }

  if (result.effective_problem_input.priors.use_intrinsics_anchor_prior) {
    Eigen::Matrix<double, 6, 1> anchor;
    anchor << result.effective_problem_input.scene_state.camera.xi,
        result.effective_problem_input.scene_state.camera.alpha,
        result.effective_problem_input.scene_state.camera.fu,
        result.effective_problem_input.scene_state.camera.fv,
        result.effective_problem_input.scene_state.camera.cu,
        result.effective_problem_input.scene_state.camera.cv;
    Eigen::Matrix<double, 6, 1> prior_weight = Eigen::Matrix<double, 6, 1>::Zero();
    prior_weight[0] =
        result.effective_problem_input.priors.intrinsics_anchor_weight_xi_alpha;
    prior_weight[1] =
        result.effective_problem_input.priors.intrinsics_anchor_weight_xi_alpha;
    prior_weight[2] =
        result.effective_problem_input.priors.intrinsics_anchor_weight_focal;
    prior_weight[3] =
        result.effective_problem_input.priors.intrinsics_anchor_weight_focal;
    prior_weight[4] =
        result.effective_problem_input.priors.intrinsics_anchor_weight_principal;
    prior_weight[5] =
        result.effective_problem_input.priors.intrinsics_anchor_weight_principal;
    boost::shared_ptr<DsIntrinsicsAnchorError> prior(
        new DsIntrinsicsAnchorError(camera_geometry,
                                    camera_dv.projectionDesignVariable(),
                                    anchor,
                                    prior_weight));
    problem->addErrorTerm(prior);
  }

  result.design_variable_count = static_cast<int>(problem->numDesignVariables());
  result.error_term_count = static_cast<int>(problem->numErrorTerms());
  if (result.error_term_count <= 0) {
    result.failure_reason = "ASLAM backend problem contains zero error terms.";
    return result;
  }

  if (options_.export_cost_parity_diagnostics) {
    result.initial_cost_parity = EvaluateCostParityDiagnostics(
        "initial",
        problem,
        measurement_result,
        result.initial_scene_state,
        reprojection_errors,
        frontend_cost_options);
  }
  if (options_.run_jacobian_consistency_check) {
    result.jacobian_diagnostics = RunJacobianDiagnostics(
        problem,
        &camera_dv,
        frame_variables,
        board_variables,
        options_.jacobian_finite_difference_epsilon);
  }

  if (problem->countActiveDesignVariables() <= 0) {
    AppendUniqueWarning("ASLAM backend problem has zero active design variables; "
                        "returning the frozen baseline state.",
                        &result.warnings);
  } else {
    const bool optimize_intrinsics =
        result.effective_problem_input.optimization_masks.optimize_intrinsics;
    if (optimize_intrinsics &&
        result.effective_problem_input.optimization_masks.delayed_intrinsics_release) {
      const int pose_only_iterations =
          std::max(0, std::min(options_.max_iterations - 1,
                               result.effective_problem_input.optimization_masks
                                   .intrinsics_release_iteration));
      const int released_iterations =
          std::max(1, options_.max_iterations - pose_only_iterations);
      if (pose_only_iterations > 0) {
        result.stages.push_back(RunOptimizationStage(
            "pose_only", false, pose_only_iterations, options_, problem, &camera_dv));
      }
      result.stages.push_back(RunOptimizationStage(
          "intrinsics_released", true, released_iterations, options_, problem, &camera_dv));
    } else {
      result.stages.push_back(RunOptimizationStage(
          optimize_intrinsics ? "joint_full" : "pose_only",
          optimize_intrinsics, options_.max_iterations, options_, problem, &camera_dv));
    }
  }

  result.optimized_scene_state.camera = GeometryToIntrinsics(*camera_geometry);
  if (!result.optimized_scene_state.camera.IsValid()) {
    const OuterBootstrapCameraIntrinsics unclamped_camera =
        result.optimized_scene_state.camera;
    ClampIntrinsicsInPlace(&result.optimized_scene_state.camera);
    AppendUniqueWarning(
        "ASLAM backend returned non-clamped double-sphere intrinsics; final camera "
        "was clamped before evaluation.",
        &result.warnings);
    if (!unclamped_camera.IsValid() && !result.optimized_scene_state.camera.IsValid()) {
      result.failure_reason = "ASLAM backend produced invalid double-sphere intrinsics.";
      return result;
    }
  }

  for (JointSceneFrameState& frame_state : result.optimized_scene_state.frames) {
    const auto frame_it = frame_variables.find(frame_state.frame_index);
    if (frame_it == frame_variables.end()) {
      continue;
    }
    frame_state.T_camera_reference = frame_it->second.transform.T();
  }
  for (JointSceneBoardState& board_state : result.optimized_scene_state.boards) {
    if (board_state.board_id == result.effective_problem_input.reference_board_id) {
      board_state.T_reference_board = Eigen::Matrix4d::Identity();
      continue;
    }
    const auto board_it = board_variables.find(board_state.board_id);
    if (board_it == board_variables.end()) {
      continue;
    }
    board_state.T_reference_board = board_it->second.transform.T();
  }

  result.optimized_residual =
      residual_evaluator.Evaluate(measurement_result, result.optimized_scene_state);
  if (!result.optimized_residual.success) {
    result.failure_reason = result.optimized_residual.failure_reason;
    result.warnings.insert(result.warnings.end(),
                           result.optimized_residual.warnings.begin(),
                           result.optimized_residual.warnings.end());
    return result;
  }

  if (options_.export_cost_parity_diagnostics) {
    result.optimized_cost_parity = EvaluateCostParityDiagnostics(
        "optimized",
        problem,
        measurement_result,
        result.optimized_scene_state,
        reprojection_errors,
        frontend_cost_options);
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
  output << "backend_debug_max_frames: " << result.options.debug_max_frames << "\n";
  output << "backend_debug_max_nonreference_boards: "
         << result.options.debug_max_nonreference_boards << "\n";
  output << "backend_force_pose_only: " << (result.options.force_pose_only ? 1 : 0) << "\n";
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

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
