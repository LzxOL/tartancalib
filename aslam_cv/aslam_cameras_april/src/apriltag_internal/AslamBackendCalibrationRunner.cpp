#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <boost/shared_ptr.hpp>

#include <aslam/backend/CameraDesignVariable.hpp>
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

JointMeasurementBuildResult BuildMeasurementResult(
    const CalibrationMeasurementDataset& dataset,
    int reference_board_id) {
  JointMeasurementBuildResult result;
  result.success = !dataset.frames.empty() && !dataset.solver_observations.empty();
  result.reference_board_id = reference_board_id;
  result.frames = dataset.frames;
  result.solver_observations = dataset.solver_observations;
  result.used_frame_count = dataset.accepted_frame_count;
  result.accepted_outer_board_observation_count =
      dataset.accepted_board_observation_count;
  result.accepted_internal_board_observation_count =
      dataset.accepted_board_observation_count;
  result.used_board_observation_count = dataset.accepted_board_observation_count;
  result.used_outer_point_count = dataset.accepted_outer_point_count;
  result.used_internal_point_count = dataset.accepted_internal_point_count;
  result.used_total_point_count = dataset.accepted_total_point_count;
  result.warnings = dataset.warnings;
  if (!result.success) {
    result.failure_reason =
        dataset.failure_reason.empty() ?
        "CalibrationMeasurementDataset has no solver observations." :
        dataset.failure_reason;
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
                      const inverse_covariance_t& inverse_covariance,
                      const aslam::backend::HomogeneousExpression& point_camera,
                      const CameraDv& camera_dv,
                      double invalid_projection_penalty_pixels)
      : measurement_(measurement),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        invalid_projection_penalty_pixels_(invalid_projection_penalty_pixels) {
    parent_t::setInvR(inverse_covariance);
    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    camera_dv_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(), design_variables.end());
  }

 protected:
  double evaluateErrorImplementation() override {
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    const bool projection_ok =
        camera_dv_.camera()->homogeneousToKeypoint(point_homogeneous, predicted);
    if (!projection_ok || !predicted.allFinite()) {
      parent_t::setError(
          Eigen::Vector2d::Constant(invalid_projection_penalty_pixels_));
    } else {
      parent_t::setError(measurement_ - predicted);
    }
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    DsReprojectionError* mutable_this = const_cast<DsReprojectionError*>(this);
    mutable_this->evaluateJacobiansFiniteDifference(jacobians);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<2>;

  measurement_t measurement_;
  aslam::backend::HomogeneousExpression point_camera_;
  CameraDv camera_dv_;
  double invalid_projection_penalty_pixels_ = 100.0;
};

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
  result.options = options_;
  result.anchor_camera = input.scene_state.camera;
  result.initial_scene_state = BuildJointSceneStateFromCalibrationSceneState(
      input.scene_state);
  result.optimized_scene_state = result.initial_scene_state;
  result.warnings = input.diagnostics_seed.warnings;

  const JointMeasurementBuildResult measurement_result =
      BuildMeasurementResult(input.measurement_dataset, input.reference_board_id);
  if (!measurement_result.success) {
    result.failure_reason = measurement_result.failure_reason;
    return result;
  }

  JointResidualEvaluationOptions residual_options;
  residual_options.top_k = 10;
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
    PoseVariableState variable;
    variable.transform = sm::kinematics::Transformation(frame_state.T_camera_reference);
    variable.expression = aslam::backend::transformationToExpression(
        variable.transform, variable.rotation_dv, variable.translation_dv);
    const bool active = input.optimization_masks.optimize_frame_poses;
    variable.rotation_dv->setActive(active);
    variable.translation_dv->setActive(active);
    problem->addDesignVariable(variable.rotation_dv);
    problem->addDesignVariable(variable.translation_dv);
    frame_variables[frame_state.frame_index] = variable;
  }

  for (const JointSceneBoardState& board_state : result.initial_scene_state.boards) {
    if (!board_state.initialized ||
        board_state.board_id == input.reference_board_id) {
      continue;
    }
    PoseVariableState variable;
    variable.transform = sm::kinematics::Transformation(board_state.T_reference_board);
    variable.expression = aslam::backend::transformationToExpression(
        variable.transform, variable.rotation_dv, variable.translation_dv);
    const bool active = input.optimization_masks.optimize_board_poses;
    variable.rotation_dv->setActive(active);
    variable.translation_dv->setActive(active);
    problem->addDesignVariable(variable.rotation_dv);
    problem->addDesignVariable(variable.translation_dv);
    board_variables[board_state.board_id] = variable;
  }

  const aslam::backend::TransformationExpression identity_transform(
      Eigen::Matrix4d::Identity());
  int skipped_point_count = 0;
  for (const JointPointObservation& observation : measurement_result.solver_observations) {
    const auto frame_it = frame_variables.find(observation.frame_index);
    if (frame_it == frame_variables.end()) {
      ++skipped_point_count;
      continue;
    }

    aslam::backend::TransformationExpression board_expression = identity_transform;
    if (observation.board_id != input.reference_board_id) {
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

    boost::shared_ptr<DsReprojectionError> error(
        new DsReprojectionError(observation.image_xy, Eigen::Matrix2d::Identity(),
                                point_camera, camera_dv,
                                options_.invalid_projection_penalty_pixels));
    if (options_.use_huber_loss) {
      const double huber_delta =
          observation.point_type == JointPointType::Outer ?
          options_.outer_huber_delta_pixels :
          options_.internal_huber_delta_pixels;
      error->setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(huber_delta)));
    }
    problem->addErrorTerm(error);
  }

  if (skipped_point_count > 0) {
    AppendUniqueWarning(
        "Skipped " + std::to_string(skipped_point_count) +
        " solver observations while building the backend problem.",
        &result.warnings);
  }

  if (input.priors.use_intrinsics_anchor_prior) {
    Eigen::Matrix<double, 6, 1> anchor;
    anchor << input.scene_state.camera.xi,
        input.scene_state.camera.alpha,
        input.scene_state.camera.fu,
        input.scene_state.camera.fv,
        input.scene_state.camera.cu,
        input.scene_state.camera.cv;
    Eigen::Matrix<double, 6, 1> prior_weight = Eigen::Matrix<double, 6, 1>::Zero();
    prior_weight[0] = input.priors.intrinsics_anchor_weight_xi_alpha;
    prior_weight[1] = input.priors.intrinsics_anchor_weight_xi_alpha;
    prior_weight[2] = input.priors.intrinsics_anchor_weight_focal;
    prior_weight[3] = input.priors.intrinsics_anchor_weight_focal;
    prior_weight[4] = input.priors.intrinsics_anchor_weight_principal;
    prior_weight[5] = input.priors.intrinsics_anchor_weight_principal;
    boost::shared_ptr<DsIntrinsicsAnchorError> prior(
        new DsIntrinsicsAnchorError(camera_geometry,
                                    camera_dv.projectionDesignVariable(),
                                    anchor, prior_weight));
    problem->addErrorTerm(prior);
  }

  result.design_variable_count = static_cast<int>(problem->numDesignVariables());
  result.error_term_count = static_cast<int>(problem->numErrorTerms());
  if (result.error_term_count <= 0) {
    result.failure_reason = "ASLAM backend problem contains zero error terms.";
    return result;
  }

  if (problem->countActiveDesignVariables() <= 0) {
    AppendUniqueWarning("ASLAM backend problem has zero active design variables; "
                        "returning the frozen baseline state.",
                        &result.warnings);
  } else {
    const bool optimize_intrinsics = input.optimization_masks.optimize_intrinsics;
    if (optimize_intrinsics && input.optimization_masks.delayed_intrinsics_release) {
      const int pose_only_iterations =
          std::max(0, std::min(options_.max_iterations - 1,
                               input.optimization_masks.intrinsics_release_iteration));
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
    if (board_state.board_id == input.reference_board_id) {
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

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
