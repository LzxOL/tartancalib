#include <aslam/cameras/apriltag_internal/Stage5IncrementalBackendEstimator.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <aslam/backend/CameraDesignVariable.hpp>
#include <aslam/backend/DesignVariable.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/HomogeneousExpression.hpp>
#include <aslam/backend/JacobianContainer.hpp>
#include <aslam/backend/MapTransformation.hpp>
#include <aslam/backend/MappedEuclideanPoint.hpp>
#include <aslam/backend/MappedRotationQuaternion.hpp>
#include <aslam/backend/MEstimatorPolicies.hpp>
#include <aslam/backend/TransformationExpression.hpp>
#include <aslam/cameras.hpp>
#include <aslam/calibration/core/IncrementalEstimator.h>
#include <aslam/calibration/core/OptimizationProblem.h>
#include <aslam/calibration/core/LinearSolverOptions.h>
#include <sm/kinematics/Transformation.hpp>

#include <aslam/cameras/apriltag_internal/JointReprojectionResidualEvaluator.hpp>
#include <aslam/cameras/apriltag_internal/Stage5Benchmark.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

using FrameBoardKey = std::pair<int, int>;
using DsGeometry = aslam::cameras::DoubleSphereCameraGeometry;
using DsProjection =
    aslam::cameras::DoubleSphereProjection<aslam::cameras::NoDistortion>;
using DsCameraDv = aslam::backend::CameraDesignVariable<DsGeometry>;
using CalibrationBatch = aslam::calibration::OptimizationProblem;

constexpr std::size_t kCalibrationGroupId = 0;
constexpr std::size_t kTransformationGroupId = 1;

enum class CameraOptimizationPhase {
  kSeedFixedIntrinsics,
  kCandidateTrustRegion,
};

boost::shared_ptr<DsGeometry> MakeDsGeometry(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  DsProjection projection(intrinsics.xi, intrinsics.alpha, intrinsics.fu,
                          intrinsics.fv, intrinsics.cu, intrinsics.cv,
                          intrinsics.resolution.width,
                          intrinsics.resolution.height);
  return boost::make_shared<DsGeometry>(
      projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask());
}

OuterBootstrapCameraIntrinsics CameraToIntrinsics(const DsGeometry& geometry) {
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

struct PoseVariable {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PoseVariable() = default;
  PoseVariable(const PoseVariable&) = delete;
  PoseVariable& operator=(const PoseVariable&) = delete;
  PoseVariable(PoseVariable&&) = delete;
  PoseVariable& operator=(PoseVariable&&) = delete;

  sm::kinematics::Transformation transform;
  boost::shared_ptr<aslam::backend::MappedRotationQuaternion> rotation_dv;
  boost::shared_ptr<aslam::backend::MappedEuclideanPoint> translation_dv;
  aslam::backend::TransformationExpression expression;
};

using PoseVariableMap =
    std::map<int, PoseVariable, std::less<int>,
             Eigen::aligned_allocator<std::pair<const int, PoseVariable> > >;
using PoseMatrixMap =
    std::map<int, Eigen::Matrix4d, std::less<int>,
             Eigen::aligned_allocator<
                 std::pair<const int, Eigen::Matrix4d> > >;

PoseVariable& GetOrCreatePoseVariable(PoseVariableMap* variables, int id) {
  auto it = variables->find(id);
  if (it != variables->end()) {
    return it->second;
  }
  return variables
      ->emplace(std::piecewise_construct, std::forward_as_tuple(id),
                std::forward_as_tuple())
      .first->second;
}

void InitializePoseVariable(PoseVariable* variable,
                            const Eigen::Matrix4d& transform_matrix,
                            bool active) {
  if (variable == nullptr) {
    return;
  }
  variable->transform = sm::kinematics::Transformation(transform_matrix);
  variable->expression = aslam::backend::transformationToExpression(
      variable->transform, variable->rotation_dv, variable->translation_dv);
  variable->rotation_dv->setActive(active);
  variable->translation_dv->setActive(active);
}

void AddPoseVariableDvs(const PoseVariable& variable,
                        std::size_t group_id,
                        const boost::shared_ptr<CalibrationBatch>& batch) {
  batch->addDesignVariable(variable.rotation_dv, group_id);
  batch->addDesignVariable(variable.translation_dv, group_id);
}

void SetPoseVariableFromMatrix(PoseVariable* variable,
                               const Eigen::Matrix4d& matrix) {
  if (variable == nullptr) {
    return;
  }
  variable->transform.set(matrix);
  variable->rotation_dv->set(variable->transform.q());
  variable->translation_dv->set(variable->transform.t());
}

bool IsPoseMatrixFinite(const Eigen::Matrix4d& matrix) {
  return matrix.allFinite() &&
         matrix.topLeftCorner<3, 3>().allFinite() &&
         matrix.topRightCorner<3, 1>().allFinite();
}

bool IsPoseVariableFinite(const PoseVariable& variable) {
  const Eigen::Vector4d& q = variable.transform.q();
  const Eigen::Vector3d& t = variable.transform.t();
  return q.allFinite() && t.allFinite() &&
         std::abs(q.norm() - 1.0) < 1e-3;
}

std::string FormatDsCameraForReason(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  std::ostringstream stream;
  stream << "xi=" << intrinsics.xi << " alpha=" << intrinsics.alpha
         << " fu=" << intrinsics.fu << " fv=" << intrinsics.fv
         << " cu=" << intrinsics.cu << " cv=" << intrinsics.cv
         << " width=" << intrinsics.resolution.width
         << " height=" << intrinsics.resolution.height
         << " family=" << intrinsics.NormalizedFamilyString();
  return stream.str();
}

bool IsCameraStateValid(const DsGeometry& geometry, std::string* reason) {
  const OuterBootstrapCameraIntrinsics intrinsics =
      CameraToIntrinsics(geometry);
  const std::vector<double> parameters = intrinsics.CombinedParameterVector();
  const bool finite = std::all_of(
      parameters.begin(), parameters.end(),
      [](double value) { return std::isfinite(value); });
  if (!finite) {
    if (reason != nullptr) {
      *reason = "nonfinite_camera_parameters " +
                FormatDsCameraForReason(intrinsics);
    }
    return false;
  }
  if (!intrinsics.IsValid()) {
    if (reason != nullptr) {
      *reason = "invalid_camera_intrinsics " +
                FormatDsCameraForReason(intrinsics);
    }
    return false;
  }
  const double width = static_cast<double>(std::max(1, intrinsics.resolution.width));
  const double height = static_cast<double>(std::max(1, intrinsics.resolution.height));
  const double min_focal = 0.05 * std::min(width, height);
  const double max_focal = 2.0 * std::max(width, height);
  if (!(intrinsics.fu > min_focal && intrinsics.fv > min_focal &&
        intrinsics.fu < max_focal && intrinsics.fv < max_focal)) {
    if (reason != nullptr) {
      *reason = "physically_implausible_focal " +
                FormatDsCameraForReason(intrinsics);
    }
    return false;
  }
  if (!(intrinsics.alpha > 1e-4 && intrinsics.alpha < 0.999) ||
      !(intrinsics.xi > -1.5 && intrinsics.xi < 2.5)) {
    if (reason != nullptr) {
      *reason = "physically_implausible_ds_shape " +
                FormatDsCameraForReason(intrinsics);
    }
    return false;
  }
  if (reason != nullptr) {
    reason->clear();
  }
  return true;
}

void FillCameraDiagnostics(const OuterBootstrapCameraIntrinsics& camera,
                           double* xi,
                           double* alpha,
                           double* fu,
                           double* fv,
                           double* cu,
                           double* cv) {
  if (xi != nullptr) {
    *xi = camera.xi;
  }
  if (alpha != nullptr) {
    *alpha = camera.alpha;
  }
  if (fu != nullptr) {
    *fu = camera.fu;
  }
  if (fv != nullptr) {
    *fv = camera.fv;
  }
  if (cu != nullptr) {
    *cu = camera.cu;
  }
  if (cv != nullptr) {
    *cv = camera.cv;
  }
}

class DsProjectionAnchorError : public aslam::backend::ErrorTermFs<6> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DsProjectionAnchorError(
      const DsProjection* projection,
      const boost::shared_ptr<aslam::backend::DesignVariableAdapter<DsProjection> >&
          projection_dv,
      const Eigen::Matrix<double, 6, 1>& anchor,
      const Eigen::Matrix<double, 6, 1>& prior_weight)
      : projection_(projection),
        projection_dv_(projection_dv),
        anchor_(anchor) {
    if (projection_ == nullptr || projection_dv_ == nullptr) {
      throw std::runtime_error("DsProjectionAnchorError requires valid projection data.");
    }
    const Eigen::Matrix<double, 6, 6> inverse_covariance =
        prior_weight.asDiagonal();
    parent_t::setInvR(inverse_covariance);
    parent_t::setDesignVariables(projection_dv_.get());
  }

 protected:
  double evaluateErrorImplementation() override {
    Eigen::MatrixXd parameters_matrix;
    projection_->getParameters(parameters_matrix);
    Eigen::Matrix<double, 6, 1> parameters =
        Eigen::Matrix<double, 6, 1>::Zero();
    parameters = parameters_matrix;
    parent_t::setError(parameters - anchor_);
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    jacobians.add(projection_dv_.get(), Eigen::Matrix<double, 6, 6>::Identity());
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<6>;

  const DsProjection* projection_ = nullptr;
  boost::shared_ptr<aslam::backend::DesignVariableAdapter<DsProjection> >
      projection_dv_;
  Eigen::Matrix<double, 6, 1> anchor_ = Eigen::Matrix<double, 6, 1>::Zero();
};

double PositiveOrDefault(double value, double fallback) {
  return value > 0.0 && std::isfinite(value) ? value : fallback;
}

double PriorWeightFromSigma(double sigma) {
  const double safe_sigma = PositiveOrDefault(sigma, 1.0);
  return 1.0 / (safe_sigma * safe_sigma);
}

class IncrementalDsReprojectionError : public aslam::backend::ErrorTermFs<2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IncrementalDsReprojectionError(
      const Eigen::Vector2d& measurement,
      const Eigen::Matrix2d& inverse_covariance,
      double huber_delta_pixels,
      bool use_huber_loss,
      const aslam::backend::HomogeneousExpression& point_camera,
      const DsCameraDv& camera_dv,
      double invalid_projection_penalty_pixels)
      : measurement_(measurement),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        inverse_covariance_(inverse_covariance),
        invalid_projection_penalty_pixels_(invalid_projection_penalty_pixels) {
    parent_t::setInvR(inverse_covariance_);
    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    camera_dv_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(),
                                         design_variables.end());
    const double balance_weight =
        0.5 * (inverse_covariance_(0, 0) + inverse_covariance_(1, 1));
    if (use_huber_loss && huber_delta_pixels > 0.0 && balance_weight > 0.0) {
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(
                  std::sqrt(balance_weight) * huber_delta_pixels)));
    }
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
    DsGeometry::jacobian_homogeneous_t projection_jacobian;
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    const bool valid_projection =
        camera_dv_.camera()->homogeneousToKeypoint(point_homogeneous,
                                                   predicted,
                                                   projection_jacobian) &&
        predicted.allFinite() && projection_jacobian.allFinite();
    if (!valid_projection) {
      return;
    }
    point_camera_.evaluateJacobians(jacobians, -projection_jacobian);
    camera_dv_.evaluateJacobians(jacobians, point_homogeneous);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<2>;

  Eigen::Vector2d ComputeResidual(Eigen::Vector2d* predicted,
                                  bool* valid_projection) const {
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    *valid_projection =
        camera_dv_.camera()->homogeneousToKeypoint(point_homogeneous,
                                                   *predicted) &&
        predicted->allFinite();
    if (!(*valid_projection)) {
      *predicted = Eigen::Vector2d::Constant(
          std::numeric_limits<double>::quiet_NaN());
      return Eigen::Vector2d::Constant(invalid_projection_penalty_pixels_);
    }
    return measurement_ - *predicted;
  }

  Eigen::Vector2d measurement_ = Eigen::Vector2d::Zero();
  aslam::backend::HomogeneousExpression point_camera_;
  const DsCameraDv& camera_dv_;
  Eigen::Matrix2d inverse_covariance_ = Eigen::Matrix2d::Identity();
  double invalid_projection_penalty_pixels_ = 100.0;
};

std::set<FrameBoardKey> CollectAcceptedKeys(
    const CalibrationMeasurementDataset& dataset) {
  std::set<FrameBoardKey> keys;
  for (const JointMeasurementFrameResult& frame : dataset.frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      if (board.used_in_solver) {
        keys.insert(FrameBoardKey(frame.frame_index, board.board_id));
      }
    }
  }
  return keys;
}

const JointSceneFrameState* FindFrameState(const CalibrationSceneState& scene,
                                           int frame_index) {
  for (const JointSceneFrameState& frame : scene.frames) {
    if (frame.frame_index == frame_index) {
      return &frame;
    }
  }
  return nullptr;
}

const JointSceneBoardState* FindBoardState(const CalibrationSceneState& scene,
                                           int board_id) {
  for (const JointSceneBoardState& board : scene.boards) {
    if (board.board_id == board_id) {
      return &board;
    }
  }
  return nullptr;
}

const JointSceneFrameState* FindPreferredFrameState(
    const CalibrationSceneState& baseline_scene,
    const CalibrationSceneState& candidate_pool_scene,
    int frame_index) {
  const JointSceneFrameState* baseline_frame =
      FindFrameState(baseline_scene, frame_index);
  if (baseline_frame != nullptr && baseline_frame->initialized) {
    return baseline_frame;
  }
  return FindFrameState(candidate_pool_scene, frame_index);
}

const JointSceneBoardState* FindPreferredBoardState(
    const CalibrationSceneState& baseline_scene,
    const CalibrationSceneState& candidate_pool_scene,
    int board_id) {
  const JointSceneBoardState* baseline_board =
      FindBoardState(baseline_scene, board_id);
  if (baseline_board != nullptr && baseline_board->initialized) {
    return baseline_board;
  }
  return FindBoardState(candidate_pool_scene, board_id);
}

const JointMeasurementFrameResult* FindMeasurementFrame(
    const CalibrationMeasurementDataset& dataset,
    int frame_index) {
  for (const JointMeasurementFrameResult& frame : dataset.frames) {
    if (frame.frame_index == frame_index) {
      return &frame;
    }
  }
  return nullptr;
}

int CountBatchPoints(const CalibrationMeasurementDataset& dataset,
                     const std::set<FrameBoardKey>& keys) {
  int count = 0;
  for (const JointPointObservation& observation : dataset.solver_observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    if (keys.count(FrameBoardKey(observation.frame_index,
                                 observation.board_id)) > 0) {
      ++count;
    }
  }
  return count;
}

struct ResidualStats {
  int total_count = 0;
  int outer_count = 0;
  int internal_count = 0;
  double total_squared_error = 0.0;
  double outer_squared_error = 0.0;
  double internal_squared_error = 0.0;
  int invalid_projection_count = 0;

  double Rmse() const {
    return total_count > 0
               ? std::sqrt(total_squared_error / static_cast<double>(total_count))
               : 0.0;
  }
  double OuterRmse() const {
    return outer_count > 0
               ? std::sqrt(outer_squared_error / static_cast<double>(outer_count))
               : 0.0;
  }
  double InternalRmse() const {
    return internal_count > 0
               ? std::sqrt(internal_squared_error /
                           static_cast<double>(internal_count))
               : 0.0;
  }
};

class PersistentProblemBuilder {
 public:
  PersistentProblemBuilder(const CalibrationStateBundle& baseline_bundle,
                           const CalibrationStateBundle& candidate_pool_bundle,
                           const Stage5IncrementalBackendEstimatorOptions& options)
      : baseline_bundle_(baseline_bundle),
        candidate_pool_bundle_(candidate_pool_bundle),
        options_(options),
        camera_geometry_(MakeDsGeometry(baseline_bundle.scene_state.camera)),
        camera_dv_(camera_geometry_) {
    camera_dv_.setActive(false, false, false);
    BuildBoardVariables();
  }

  struct StateSnapshot;

  boost::shared_ptr<CalibrationBatch> BuildBatch(
      const std::set<FrameBoardKey>& keys,
      bool force_add_frame_variables,
      CameraOptimizationPhase camera_phase,
      const StateSnapshot* camera_anchor_state) {
    boost::shared_ptr<CalibrationBatch> batch =
        boost::make_shared<CalibrationBatch>();
    AddCameraVariables(camera_phase, batch);
    MaybeAddCameraAnchorPrior(camera_phase, camera_anchor_state, batch);
    AddBoardVariables(batch);
    AddFrameVariables(keys, force_add_frame_variables, batch);
    AddResiduals(keys, batch);
    return batch;
  }

  ResidualStats EvaluateAccepted(
      const std::set<FrameBoardKey>& accepted_keys) const {
    ResidualStats stats;
    for (const JointPointObservation& observation :
         candidate_pool_bundle_.measurement_dataset.solver_observations) {
      if (!observation.used_in_solver ||
          accepted_keys.count(FrameBoardKey(observation.frame_index,
                                            observation.board_id)) == 0) {
        continue;
      }
      const PoseVariable* frame_variable =
          FindFrameVariable(observation.frame_index);
      if (frame_variable == nullptr) {
        ++stats.invalid_projection_count;
        continue;
      }
      Eigen::Matrix4d T_reference_board = Eigen::Matrix4d::Identity();
      if (observation.board_id !=
          candidate_pool_bundle_.scene_state.reference_board_id) {
        const PoseVariable* board_variable =
            FindBoardVariable(observation.board_id);
        if (board_variable == nullptr) {
          ++stats.invalid_projection_count;
          continue;
        }
        T_reference_board = board_variable->expression.toTransformationMatrix();
      }
      const Eigen::Vector4d point_board(observation.target_xyz_board.x(),
                                        observation.target_xyz_board.y(),
                                        observation.target_xyz_board.z(),
                                        1.0);
      const Eigen::Vector4d point_camera =
          frame_variable->expression.toTransformationMatrix() *
          T_reference_board * point_board;
      Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
      if (!camera_geometry_->homogeneousToKeypoint(point_camera, predicted) ||
          !predicted.allFinite()) {
        ++stats.invalid_projection_count;
        continue;
      }
      const double squared_error =
          (predicted - observation.image_xy).squaredNorm();
      ++stats.total_count;
      stats.total_squared_error += squared_error;
      if (observation.point_type == JointPointType::Outer) {
        ++stats.outer_count;
        stats.outer_squared_error += squared_error;
      } else {
        ++stats.internal_count;
        stats.internal_squared_error += squared_error;
      }
    }
    return stats;
  }

  CalibrationSceneState BuildSceneState() const {
    CalibrationSceneState scene = candidate_pool_bundle_.scene_state;
    scene.camera = CameraToIntrinsics(*camera_geometry_);
    scene.camera_model = scene.camera.NormalizedCameraModel();
    scene.distortion_model = scene.camera.NormalizedDistortionModel();
    for (JointSceneFrameState& frame : scene.frames) {
      const PoseVariable* variable = FindFrameVariable(frame.frame_index);
      if (variable != nullptr) {
        frame.T_camera_reference =
            variable->expression.toTransformationMatrix();
        frame.initialized = true;
      }
    }
    for (JointSceneBoardState& board : scene.boards) {
      if (board.board_id == scene.reference_board_id) {
        board.T_reference_board = Eigen::Matrix4d::Identity();
        board.initialized = true;
        continue;
      }
      const PoseVariable* variable = FindBoardVariable(board.board_id);
      if (variable != nullptr) {
        board.T_reference_board =
            variable->expression.toTransformationMatrix();
        board.initialized = true;
      }
    }
    scene.coarse_or_optimized_level = "stage5_persistent_incremental_backend";
    scene.source_pipeline_label = "stage5_persistent_incremental_backend";
    return scene;
  }

  struct StateSnapshot {
    Eigen::MatrixXd projection_parameters;
    PoseMatrixMap frame_poses;
    PoseMatrixMap board_poses;
  };

  StateSnapshot CaptureState() const {
    StateSnapshot snapshot;
    camera_geometry_->projection().getParameters(
        snapshot.projection_parameters);
    for (const auto& entry : frame_variables_) {
      snapshot.frame_poses[entry.first] =
          entry.second.expression.toTransformationMatrix();
    }
    for (const auto& entry : board_variables_) {
      snapshot.board_poses[entry.first] =
          entry.second.expression.toTransformationMatrix();
    }
    return snapshot;
  }

  OuterBootstrapCameraIntrinsics CurrentCamera() const {
    return CameraToIntrinsics(*camera_geometry_);
  }

  void RestoreState(const StateSnapshot& snapshot) {
    if (snapshot.projection_parameters.size() > 0) {
      camera_dv_.projectionDesignVariable()->setParameters(
          snapshot.projection_parameters);
    }
    for (auto it = frame_variables_.begin(); it != frame_variables_.end();) {
      if (snapshot.frame_poses.count(it->first) == 0) {
        it = frame_variables_.erase(it);
      } else {
        ++it;
      }
    }
    for (const auto& entry : snapshot.frame_poses) {
      auto variable_it = frame_variables_.find(entry.first);
      if (variable_it != frame_variables_.end()) {
        SetPoseVariableFromMatrix(&variable_it->second, entry.second);
      }
    }
    for (const auto& entry : snapshot.board_poses) {
      auto variable_it = board_variables_.find(entry.first);
      if (variable_it != board_variables_.end()) {
        SetPoseVariableFromMatrix(&variable_it->second, entry.second);
      }
    }
  }

  bool CurrentStateFinite(std::string* reason) const {
    if (!IsCameraStateValid(*camera_geometry_, reason)) {
      return false;
    }
    for (const auto& entry : frame_variables_) {
      if (!IsPoseVariableFinite(entry.second)) {
        if (reason != nullptr) {
          *reason = "nonfinite_frame_pose_" + std::to_string(entry.first);
        }
        return false;
      }
    }
    for (const auto& entry : board_variables_) {
      if (!IsPoseVariableFinite(entry.second)) {
        if (reason != nullptr) {
          *reason = "nonfinite_board_pose_" + std::to_string(entry.first);
        }
        return false;
      }
    }
    if (reason != nullptr) {
      reason->clear();
    }
    return true;
  }

  bool CandidateCameraStepWithinTrustRegion(const StateSnapshot& anchor,
                                            std::string* reason) const {
    if (!options_.optimize_candidate_intrinsics) {
      if (reason != nullptr) {
        reason->clear();
      }
      return true;
    }
    if (options_.use_candidate_intrinsics_anchor_prior) {
      if (reason != nullptr) {
        reason->clear();
      }
      return true;
    }
    if (anchor.projection_parameters.rows() < 6 ||
        anchor.projection_parameters.cols() < 1) {
      if (reason != nullptr) {
        *reason = "missing_camera_trust_region_anchor";
      }
      return false;
    }
    Eigen::MatrixXd current_parameters;
    camera_geometry_->projection().getParameters(current_parameters);
    if (current_parameters.rows() < 6 || current_parameters.cols() < 1 ||
        !current_parameters.allFinite()) {
      if (reason != nullptr) {
        *reason = "nonfinite_camera_parameters_after_candidate";
      }
      return false;
    }
    const double anchor_fu = std::max(1.0, std::abs(anchor.projection_parameters(2, 0)));
    const double anchor_fv = std::max(1.0, std::abs(anchor.projection_parameters(3, 0)));
    const double max_fu_step =
        options_.max_candidate_focal_relative_step * anchor_fu;
    const double max_fv_step =
        options_.max_candidate_focal_relative_step * anchor_fv;
    const double d_xi = std::abs(current_parameters(0, 0) -
                                 anchor.projection_parameters(0, 0));
    const double d_alpha = std::abs(current_parameters(1, 0) -
                                    anchor.projection_parameters(1, 0));
    const double d_fu = std::abs(current_parameters(2, 0) -
                                 anchor.projection_parameters(2, 0));
    const double d_fv = std::abs(current_parameters(3, 0) -
                                 anchor.projection_parameters(3, 0));
    const double d_cu = std::abs(current_parameters(4, 0) -
                                 anchor.projection_parameters(4, 0));
    const double d_cv = std::abs(current_parameters(5, 0) -
                                 anchor.projection_parameters(5, 0));
    if (d_xi > options_.max_candidate_xi_alpha_step ||
        d_alpha > options_.max_candidate_xi_alpha_step ||
        d_fu > max_fu_step ||
        d_fv > max_fv_step ||
        d_cu > options_.max_candidate_principal_step_px ||
        d_cv > options_.max_candidate_principal_step_px) {
      if (reason != nullptr) {
        std::ostringstream stream;
        stream << "camera_trust_region_gate "
               << "dxi=" << d_xi << " dalpha=" << d_alpha
               << " dfu=" << d_fu << "/" << max_fu_step
               << " dfv=" << d_fv << "/" << max_fv_step
               << " dcu=" << d_cu << " dcv=" << d_cv;
        *reason = stream.str();
      }
      return false;
    }
    if (reason != nullptr) {
      reason->clear();
    }
    return true;
  }

 private:
  void AddCameraVariables(CameraOptimizationPhase camera_phase,
                          const boost::shared_ptr<CalibrationBatch>& batch) {
    const bool projection_active =
        camera_phase == CameraOptimizationPhase::kCandidateTrustRegion &&
        options_.optimize_candidate_intrinsics;
    camera_dv_.setActive(projection_active, false, false);
    batch->addDesignVariable(camera_dv_.projectionDesignVariable(),
                             kCalibrationGroupId);
    batch->addDesignVariable(camera_dv_.distortionDesignVariable(),
                             kCalibrationGroupId);
    batch->addDesignVariable(camera_dv_.shutterDesignVariable(),
                             kCalibrationGroupId);
  }

  void MaybeAddCameraAnchorPrior(
      CameraOptimizationPhase camera_phase,
      const StateSnapshot* camera_anchor_state,
      const boost::shared_ptr<CalibrationBatch>& batch) {
    if (camera_phase != CameraOptimizationPhase::kCandidateTrustRegion ||
        !options_.optimize_candidate_intrinsics ||
        !options_.use_candidate_intrinsics_anchor_prior ||
        camera_anchor_state == nullptr ||
        camera_anchor_state->projection_parameters.rows() < 6 ||
        camera_anchor_state->projection_parameters.cols() < 1) {
      return;
    }
    Eigen::Matrix<double, 6, 1> anchor = Eigen::Matrix<double, 6, 1>::Zero();
    anchor = camera_anchor_state->projection_parameters;
    Eigen::Matrix<double, 6, 1> weights = Eigen::Matrix<double, 6, 1>::Zero();
    const double focal_sigma_u = std::max(
        1.0, options_.max_candidate_focal_relative_step *
                 std::max(1.0, std::abs(anchor[2])));
    const double focal_sigma_v = std::max(
        1.0, options_.max_candidate_focal_relative_step *
                 std::max(1.0, std::abs(anchor[3])));
    const double principal_sigma =
        PositiveOrDefault(options_.max_candidate_principal_step_px, 20.0);
    const double xi_alpha_sigma =
        PositiveOrDefault(options_.max_candidate_xi_alpha_step, 0.03);
    const double xi_alpha_weight =
        PositiveOrDefault(options_.candidate_intrinsics_anchor_weight_xi_alpha,
                          PriorWeightFromSigma(xi_alpha_sigma));
    const double focal_weight_u =
        PositiveOrDefault(options_.candidate_intrinsics_anchor_weight_focal,
                          PriorWeightFromSigma(focal_sigma_u));
    const double focal_weight_v =
        PositiveOrDefault(options_.candidate_intrinsics_anchor_weight_focal,
                          PriorWeightFromSigma(focal_sigma_v));
    const double principal_weight =
        PositiveOrDefault(options_.candidate_intrinsics_anchor_weight_principal,
                          PriorWeightFromSigma(principal_sigma));
    weights[0] = xi_alpha_weight;
    weights[1] = xi_alpha_weight;
    weights[2] = focal_weight_u;
    weights[3] = focal_weight_v;
    weights[4] = principal_weight;
    weights[5] = principal_weight;
    boost::shared_ptr<DsProjectionAnchorError> prior(
        new DsProjectionAnchorError(
            &camera_geometry_->projection(),
            camera_dv_.projectionDesignVariable(),
            anchor,
            weights));
    batch->addErrorTerm(prior);
  }

  void BuildBoardVariables() {
    std::set<int> board_ids;
    for (const JointSceneBoardState& board : baseline_bundle_.scene_state.boards) {
      board_ids.insert(board.board_id);
    }
    for (const JointSceneBoardState& board :
         candidate_pool_bundle_.scene_state.boards) {
      board_ids.insert(board.board_id);
    }
    for (int board_id : board_ids) {
      if (board_id == candidate_pool_bundle_.scene_state.reference_board_id) {
        continue;
      }
      const JointSceneBoardState* board = FindPreferredBoardState(
          baseline_bundle_.scene_state, candidate_pool_bundle_.scene_state,
          board_id);
      if (board == nullptr || !board->initialized) {
        continue;
      }
      PoseVariable& variable =
          GetOrCreatePoseVariable(&board_variables_, board_id);
      InitializePoseVariable(&variable, board->T_reference_board, true);
    }
  }

  void AddBoardVariables(const boost::shared_ptr<CalibrationBatch>& batch) {
    for (const auto& entry : board_variables_) {
      AddPoseVariableDvs(entry.second, kCalibrationGroupId, batch);
    }
  }

  void AddFrameVariables(const std::set<FrameBoardKey>& keys,
                         bool force_add_frame_variables,
                         const boost::shared_ptr<CalibrationBatch>& batch) {
    std::set<int> frames;
    for (const FrameBoardKey& key : keys) {
      frames.insert(key.first);
    }
    for (int frame_index : frames) {
      auto frame_it = frame_variables_.find(frame_index);
      if (frame_it == frame_variables_.end()) {
        const JointSceneFrameState* frame_state =
            FindPreferredFrameState(baseline_bundle_.scene_state,
                                    candidate_pool_bundle_.scene_state,
                                    frame_index);
        if (frame_state == nullptr || !frame_state->initialized) {
          throw std::runtime_error(
              "Stage5 incremental estimator missing initialized frame pose.");
        }
        PoseVariable& variable =
            GetOrCreatePoseVariable(&frame_variables_, frame_index);
        InitializePoseVariable(&variable, frame_state->T_camera_reference,
                               true);
        frame_it = frame_variables_.find(frame_index);
      } else if (!force_add_frame_variables) {
        continue;
      }
      AddPoseVariableDvs(frame_it->second, kTransformationGroupId, batch);
    }
  }

  const PoseVariable* FindFrameVariable(int frame_index) const {
    const auto it = frame_variables_.find(frame_index);
    return it == frame_variables_.end() ? nullptr : &it->second;
  }

  const PoseVariable* FindBoardVariable(int board_id) const {
    const auto it = board_variables_.find(board_id);
    return it == board_variables_.end() ? nullptr : &it->second;
  }

  void AddResiduals(const std::set<FrameBoardKey>& keys,
                    const boost::shared_ptr<CalibrationBatch>& batch) {
    const std::map<FrameBoardKey, int> point_count_by_key =
        CountPointsByKey(keys);
    const aslam::backend::TransformationExpression identity_transform(
        Eigen::Matrix4d::Identity());
    for (const JointPointObservation& observation :
         candidate_pool_bundle_.measurement_dataset.solver_observations) {
      if (!observation.used_in_solver ||
          keys.count(FrameBoardKey(observation.frame_index,
                                   observation.board_id)) == 0) {
        continue;
      }
      const PoseVariable* frame_variable =
          FindFrameVariable(observation.frame_index);
      if (frame_variable == nullptr) {
        continue;
      }
      aslam::backend::TransformationExpression board_expression =
          identity_transform;
      if (observation.board_id !=
          candidate_pool_bundle_.scene_state.reference_board_id) {
        const PoseVariable* board_variable =
            FindBoardVariable(observation.board_id);
        if (board_variable == nullptr) {
          continue;
        }
        board_expression = board_variable->expression;
      }
      const aslam::backend::HomogeneousExpression point_board(
          observation.target_xyz_board);
      const aslam::backend::HomogeneousExpression point_camera =
          frame_variable->expression * (board_expression * point_board);
      const int key_point_count =
          std::max(1, point_count_by_key.at(
                          FrameBoardKey(observation.frame_index,
                                        observation.board_id)));
      const double weight =
          std::max(0.0, observation.final_observation_weight) /
          static_cast<double>(key_point_count);
      if (!(weight > 0.0)) {
        continue;
      }
      const Eigen::Matrix2d inv_r =
          weight * Eigen::Matrix2d::Identity();
      boost::shared_ptr<IncrementalDsReprojectionError> error(
          new IncrementalDsReprojectionError(
              observation.image_xy, inv_r, options_.huber_delta_pixels,
              options_.use_huber_loss, point_camera, camera_dv_,
              options_.invalid_projection_penalty_pixels));
      batch->addErrorTerm(error);
    }
  }

  std::map<FrameBoardKey, int> CountPointsByKey(
      const std::set<FrameBoardKey>& keys) const {
    std::map<FrameBoardKey, int> counts;
    for (const FrameBoardKey& key : keys) {
      counts[key] = 0;
    }
    for (const JointPointObservation& observation :
         candidate_pool_bundle_.measurement_dataset.solver_observations) {
      if (!observation.used_in_solver) {
        continue;
      }
      const FrameBoardKey key(observation.frame_index, observation.board_id);
      const auto it = counts.find(key);
      if (it != counts.end()) {
        ++it->second;
      }
    }
    return counts;
  }

  const CalibrationStateBundle& baseline_bundle_;
  const CalibrationStateBundle& candidate_pool_bundle_;
  Stage5IncrementalBackendEstimatorOptions options_;
  boost::shared_ptr<DsGeometry> camera_geometry_;
  DsCameraDv camera_dv_;
  PoseVariableMap frame_variables_;
  PoseVariableMap board_variables_;
};

Stage5IncrementalBackendEstimatorOptions MakeOptions(
    const TrialBackendFrameBoardSelectionOptions& selection_options,
    const AslamBackendCalibrationOptions& backend_runner_options) {
  Stage5IncrementalBackendEstimatorOptions options;
  options.enabled = true;
  options.information_gain_threshold =
      selection_options.acceptance_information_gain_threshold;
  options.rank_gain_threshold =
      selection_options.acceptance_rank_gain_threshold;
  options.max_iterations =
      selection_options.max_iterations > 0
          ? selection_options.max_iterations
          : backend_runner_options.max_iterations;
  options.convergence_delta_j = backend_runner_options.convergence_delta_j;
  options.convergence_delta_x = backend_runner_options.convergence_delta_x;
  options.verbose = backend_runner_options.verbose;
  options.check_validity = false;
  options.use_huber_loss = backend_runner_options.use_huber_loss;
  options.huber_delta_pixels =
      std::min(backend_runner_options.outer_huber_delta_pixels,
               backend_runner_options.internal_huber_delta_pixels);
  options.invalid_projection_penalty_pixels =
      backend_runner_options.invalid_projection_penalty_pixels;
  options.optimize_seed_intrinsics = false;
  options.optimize_candidate_intrinsics =
      selection_options.optimize_intrinsics_in_trial;
  options.use_candidate_intrinsics_anchor_prior =
      selection_options.optimize_intrinsics_in_trial &&
      selection_options.persistent_intrinsics_anchor_prior_enabled;
  options.candidate_intrinsics_anchor_weight_xi_alpha =
      selection_options.persistent_intrinsics_anchor_weight_xi_alpha;
  options.candidate_intrinsics_anchor_weight_focal =
      selection_options.persistent_intrinsics_anchor_weight_focal;
  options.candidate_intrinsics_anchor_weight_principal =
      selection_options.persistent_intrinsics_anchor_weight_principal;
  options.max_candidate_focal_relative_step =
      PositiveOrDefault(selection_options.persistent_max_focal_relative_step,
                        options.max_candidate_focal_relative_step);
  options.max_candidate_principal_step_px =
      PositiveOrDefault(selection_options.persistent_max_principal_step_px,
                        options.max_candidate_principal_step_px);
  options.max_candidate_xi_alpha_step =
      PositiveOrDefault(selection_options.persistent_max_xi_alpha_step,
                        options.max_candidate_xi_alpha_step);
  return options;
}

JointMeasurementBuildResult BuildPersistentMeasurementResultFromDataset(
    const CalibrationMeasurementDataset& dataset,
    int reference_board_id) {
  JointMeasurementBuildResult result;
  result.reference_board_id = reference_board_id;
  result.frames = dataset.frames;
  result.solver_observations = dataset.solver_observations;
  result.warnings = dataset.warnings;
  result.used_frame_count = dataset.accepted_frame_count;
  result.used_board_observation_count = dataset.accepted_board_observation_count;
  result.used_outer_point_count = dataset.accepted_outer_point_count;
  result.used_internal_point_count = dataset.accepted_internal_point_count;
  result.used_total_point_count = dataset.accepted_total_point_count;
  result.success = !result.frames.empty() && result.used_total_point_count > 0;
  if (!result.success) {
    result.failure_reason =
        dataset.failure_reason.empty()
            ? "CalibrationMeasurementDataset has no used-in-solver observations."
            : dataset.failure_reason;
  }
  return result;
}

void ReevaluateCuratedBundleResidual(CalibrationStateBundle* bundle) {
  if (bundle == nullptr) {
    return;
  }
  const JointMeasurementBuildResult measurement =
      BuildPersistentMeasurementResultFromDataset(
          bundle->measurement_dataset, bundle->scene_state.reference_board_id);
  if (!measurement.success) {
    bundle->success = false;
    bundle->ready_for_backend = false;
    bundle->failure_reason = measurement.failure_reason;
    return;
  }
  JointResidualEvaluationOptions residual_options;
  residual_options.top_k = 10;
  const JointReprojectionResidualEvaluator residual_evaluator(residual_options);
  bundle->residual_result = residual_evaluator.Evaluate(
      measurement,
      BuildJointSceneStateFromCalibrationSceneState(bundle->scene_state));
  bundle->ready_for_backend =
      bundle->measurement_dataset.accepted_total_point_count > 0 &&
      bundle->scene_state.IsValid() &&
      bundle->residual_result.success;
  bundle->success = bundle->ready_for_backend;
  if (bundle->success) {
    bundle->failure_reason.clear();
  } else {
    bundle->failure_reason =
        bundle->residual_result.failure_reason.empty()
            ? "persistent incremental curated bundle residual evaluation failed"
            : bundle->residual_result.failure_reason;
  }
}

CalibrationStateBundle BuildCuratedBundle(
    const CalibrationStateBundle& scene_template,
    const CalibrationStateBundle& candidate_pool,
    const CalibrationSceneState& optimized_scene,
    const std::set<FrameBoardKey>& accepted_keys) {
  CalibrationStateBundle bundle = scene_template;
  bundle.scene_state = optimized_scene;
  bundle.measurement_dataset = candidate_pool.measurement_dataset;
  bundle.measurement_dataset.source_stage_label =
      candidate_pool.measurement_dataset.source_stage_label +
      "_persistent_incremental_backend";
  bundle.measurement_dataset.accepted_frame_indices.clear();
  bundle.measurement_dataset.accepted_board_observation_keys.clear();
  bundle.measurement_dataset.accepted_frame_count = 0;
  bundle.measurement_dataset.accepted_board_observation_count = 0;
  bundle.measurement_dataset.accepted_outer_point_count = 0;
  bundle.measurement_dataset.accepted_internal_point_count = 0;
  bundle.measurement_dataset.accepted_total_point_count = 0;
  for (JointMeasurementFrameResult& frame : bundle.measurement_dataset.frames) {
    bool frame_used = false;
    for (JointBoardObservation& board : frame.board_observations) {
      const FrameBoardKey key(frame.frame_index, board.board_id);
      const bool keep = accepted_keys.count(key) > 0;
      board.used_in_solver = keep;
      board.outer_point_count = 0;
      board.internal_point_count = 0;
      for (JointPointObservation& point : board.points) {
        const bool used = keep && point.used_in_solver;
        point.used_in_solver = used;
        if (used) {
          ++bundle.measurement_dataset.accepted_total_point_count;
          if (point.point_type == JointPointType::Outer) {
            ++board.outer_point_count;
            ++bundle.measurement_dataset.accepted_outer_point_count;
          } else {
            ++board.internal_point_count;
            ++bundle.measurement_dataset.accepted_internal_point_count;
          }
        } else if (!keep) {
          point.rejection_detail =
              "persistent_incremental_backend_not_selected";
        }
      }
      if (keep) {
        frame_used = true;
        bundle.measurement_dataset.accepted_board_observation_keys.insert(key);
        ++bundle.measurement_dataset.accepted_board_observation_count;
      }
    }
    if (frame_used) {
      bundle.measurement_dataset.accepted_frame_indices.insert(
          frame.frame_index);
    }
  }
  bundle.measurement_dataset.accepted_frame_count =
      static_cast<int>(bundle.measurement_dataset.accepted_frame_indices.size());
  bundle.measurement_dataset.solver_observations.clear();
  for (const JointMeasurementFrameResult& frame :
       bundle.measurement_dataset.frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      for (const JointPointObservation& point : board.points) {
        if (point.used_in_solver) {
          bundle.measurement_dataset.solver_observations.push_back(point);
        }
      }
    }
  }
  if (bundle.measurement_dataset.accepted_total_point_count <= 0) {
    bundle.ready_for_backend = false;
    bundle.success = false;
    bundle.failure_reason =
        "persistent incremental backend produced no points";
  } else if (!bundle.scene_state.IsValid()) {
    bundle.ready_for_backend = false;
    bundle.success = false;
    bundle.failure_reason =
        "persistent incremental backend produced invalid scene state";
  } else {
    ReevaluateCuratedBundleResidual(&bundle);
  }
  return bundle;
}

}  // namespace

bool IsStage5IncrementalBackendEstimatorCompatible(
    const CalibrationStateBundle& baseline_bundle,
    const CalibrationStateBundle& candidate_pool_bundle,
    const BackendProblemOptions& backend_options,
    const TrialBackendFrameBoardSelectionOptions& selection_options,
    const AslamBackendCalibrationOptions& backend_runner_options,
    std::string* reason) {
  auto set_reason = [&](const std::string& value) {
    if (reason != nullptr) {
      *reason = value;
    }
    return false;
  };
  if (!selection_options.enabled) {
    return set_reason("trial backend selection disabled");
  }
  if (selection_options.selection_mode !=
      TrialBackendFrameBoardSelectionOptions::SelectionMode::KalibrStyleBatch) {
    return set_reason("selection mode is not kalibr_style_batch");
  }
  if (selection_options.candidate_batch_granularity !=
      TrialBackendFrameBoardSelectionOptions::CandidateBatchGranularity::Frame) {
    return set_reason("candidate batch granularity is not frame");
  }
  if (!selection_options.optimize_intrinsics_in_trial) {
    return set_reason("trial intrinsics are not active");
  }
  if (backend_runner_options.residual_model != ResidualModel::ImagePlane ||
      backend_runner_options.use_point_type_residual_split ||
      backend_runner_options.angular_auxiliary_enabled) {
    return set_reason("only image-plane residual is supported in first version");
  }
  if (!baseline_bundle.IsReadyForBackend() ||
      !candidate_pool_bundle.IsReadyForBackend()) {
    return set_reason("bundle is not ready for backend");
  }
  if (baseline_bundle.scene_state.camera.NormalizedFamilyString() != "ds-none" ||
      candidate_pool_bundle.scene_state.camera.NormalizedFamilyString() !=
          "ds-none") {
    return set_reason("only ds-none camera model is supported");
  }
  if (!backend_options.optimize_frame_poses ||
      !backend_options.optimize_board_poses) {
    return set_reason("frame and board pose optimization must be enabled");
  }
  if (reason != nullptr) {
    reason->clear();
  }
  return true;
}

Stage5IncrementalBackendEstimatorResult RunStage5IncrementalBackendEstimator(
    const CalibrationStateBundle& baseline_bundle,
    const CalibrationStateBundle& candidate_pool_bundle,
    const BackendProblemOptions& backend_options,
    const TrialBackendFrameBoardSelectionOptions& selection_options,
    const AslamBackendCalibrationOptions& backend_runner_options,
    const std::vector<Stage5IncrementalBackendBatchInput>& candidate_batches) {
  const auto time_start = std::chrono::steady_clock::now();
  Stage5IncrementalBackendEstimatorResult result;
  result.attempted = true;
  result.curated_bundle = baseline_bundle;
  result.optimized_scene_state = baseline_bundle.scene_state;
  result.accepted_keys =
      CollectAcceptedKeys(baseline_bundle.measurement_dataset);
  result.candidate_batch_count = static_cast<int>(candidate_batches.size());

  std::string incompatible_reason;
  result.compatible = IsStage5IncrementalBackendEstimatorCompatible(
      baseline_bundle, candidate_pool_bundle, backend_options,
      selection_options, backend_runner_options, &incompatible_reason);
  if (!result.compatible) {
    result.fallback_reason = incompatible_reason;
    result.failure_reason = incompatible_reason;
    return result;
  }

  const Stage5IncrementalBackendEstimatorOptions estimator_options =
      MakeOptions(selection_options, backend_runner_options);

  aslam::calibration::IncrementalEstimator::Options inc_options;
  inc_options.infoGainDelta =
      estimator_options.information_gain_threshold;
  inc_options.checkValidity = estimator_options.check_validity;
  inc_options.verbose = estimator_options.verbose;
  aslam::calibration::LinearSolverOptions solver_options;
  solver_options.columnScaling = true;
  aslam::backend::Optimizer2Options optimizer_options;
  optimizer_options.maxIterations = estimator_options.max_iterations;
  optimizer_options.convergenceDeltaJ =
      estimator_options.convergence_delta_j;
  optimizer_options.convergenceDeltaX =
      estimator_options.convergence_delta_x;
  optimizer_options.verbose = estimator_options.verbose;
  optimizer_options.nThreads = 4;

  try {
    PersistentProblemBuilder builder(baseline_bundle, candidate_pool_bundle,
                                     estimator_options);
    aslam::calibration::IncrementalEstimator estimator(
        kCalibrationGroupId, inc_options, solver_options, optimizer_options);

    const std::set<FrameBoardKey> seed_keys = result.accepted_keys;
    result.seed_board_observation_count = static_cast<int>(seed_keys.size());
    std::set<int> seed_frames;
    for (const FrameBoardKey& key : seed_keys) {
      seed_frames.insert(key.first);
    }
    result.seed_frame_count = static_cast<int>(seed_frames.size());
    result.seed_point_count = CountBatchPoints(
        candidate_pool_bundle.measurement_dataset, seed_keys);
    const PersistentProblemBuilder::StateSnapshot seed_state =
        builder.CaptureState();
    boost::shared_ptr<CalibrationBatch> seed_batch =
        builder.BuildBatch(seed_keys, true,
                           CameraOptimizationPhase::kSeedFixedIntrinsics,
                           nullptr);
    const aslam::calibration::IncrementalEstimator::ReturnValue seed_ret =
        estimator.addBatch(seed_batch, true);
    result.seed_batch_count = seed_ret.batchAccepted ? 1 : 0;
    if (!seed_ret.batchAccepted) {
      result.failure_reason = "forced seed batch was not accepted";
      return result;
    }
    std::string seed_state_invalid_reason;
    bool seed_state_restored = false;
    if (!builder.CurrentStateFinite(&seed_state_invalid_reason)) {
      builder.RestoreState(seed_state);
      seed_state_restored = true;
      result.warnings.push_back(
          "Forced persistent seed optimization produced an invalid state and "
          "was restored to the incoming backend seed: " +
          seed_state_invalid_reason);
    }

    int current_rank =
        seed_state_restored ? -1 : static_cast<int>(seed_ret.rankTheta);
    for (const Stage5IncrementalBackendBatchInput& input :
         candidate_batches) {
      Stage5IncrementalBackendBatchResult batch_result;
      batch_result.attempted = true;
      batch_result.force = input.force;
      batch_result.frame_index = input.frame_index;
      batch_result.frame_label = input.frame_label;
      batch_result.max_trial_rmse = input.max_trial_rmse;
      batch_result.residual_health_threshold_px =
          input.residual_health_threshold_px;
      batch_result.batch_board_observation_count =
          static_cast<int>(input.frame_board_keys.size());
      batch_result.batch_point_count = CountBatchPoints(
          candidate_pool_bundle.measurement_dataset, input.frame_board_keys);
      batch_result.information_gain_threshold =
          estimator_options.information_gain_threshold;
      batch_result.rank_gain_threshold =
          estimator_options.rank_gain_threshold;
      batch_result.rank_theta_before = current_rank;
      FillCameraDiagnostics(builder.CurrentCamera(),
                            &batch_result.camera_xi_before,
                            &batch_result.camera_alpha_before,
                            &batch_result.camera_fu_before,
                            &batch_result.camera_fv_before,
                            &batch_result.camera_cu_before,
                            &batch_result.camera_cv_before);
      ++result.attempted_batch_count;

      if (input.frame_board_keys.empty() ||
          batch_result.batch_point_count <= 0) {
        batch_result.batch_accepted = false;
        batch_result.reject_reason = "empty_batch";
        batch_result.committed_or_rollback = "rollback";
        ++result.rejected_batch_count;
        result.batch_results.push_back(batch_result);
        continue;
      }

      const PersistentProblemBuilder::StateSnapshot batch_state =
          builder.CaptureState();
      boost::shared_ptr<CalibrationBatch> batch =
          builder.BuildBatch(input.frame_board_keys, true,
                             CameraOptimizationPhase::kCandidateTrustRegion,
                             &batch_state);
      aslam::calibration::IncrementalEstimator::ReturnValue ret;
      try {
        ret = estimator.addBatch(batch, input.force);
      } catch (const std::exception& exception) {
        builder.RestoreState(batch_state);
        batch_result.batch_accepted = false;
        batch_result.optimization_success = false;
        batch_result.objective_finite = false;
        batch_result.solution_valid = false;
        batch_result.reject_reason =
            std::string("incremental_add_batch_exception: ") +
            exception.what();
        batch_result.committed_or_rollback = "rollback";
        ++result.rejected_batch_count;
        result.batch_results.push_back(batch_result);
        continue;
      }
      const bool incremental_accepted = ret.batchAccepted;
      batch_result.batch_accepted = incremental_accepted;
      batch_result.num_iterations = static_cast<int>(ret.numIterations);
      batch_result.information_gain = ret.informationGain;
      batch_result.rank_theta_after = static_cast<int>(ret.rankTheta);
      batch_result.objective_start = ret.JStart;
      batch_result.objective_final = ret.JFinal;
      batch_result.elapsed_time_seconds = ret.elapsedTime;
      batch_result.objective_finite =
          std::isfinite(ret.JStart) && std::isfinite(ret.JFinal);
      batch_result.objective_decreased =
          batch_result.objective_finite && ret.JFinal < ret.JStart;
      batch_result.information_gate_pass =
          (ret.informationGain >
               estimator_options.information_gain_threshold ||
           (batch_result.rank_theta_before >= 0 &&
            (static_cast<double>(ret.rankTheta) -
             static_cast<double>(batch_result.rank_theta_before)) >
                estimator_options.rank_gain_threshold));
      batch_result.residual_health_pass =
          input.force ||
          input.residual_health_threshold_px <= 0.0 ||
          input.max_trial_rmse <= input.residual_health_threshold_px;
      FillCameraDiagnostics(builder.CurrentCamera(),
                            &batch_result.camera_xi_after,
                            &batch_result.camera_alpha_after,
                            &batch_result.camera_fu_after,
                            &batch_result.camera_fv_after,
                            &batch_result.camera_cu_after,
                            &batch_result.camera_cv_after);
      batch_result.optimization_success =
          ret.numIterations < static_cast<std::size_t>(
                                  std::max(1, estimator_options.max_iterations));
      batch_result.solution_valid =
          input.force ||
          (batch_result.objective_finite &&
           batch_result.objective_decreased &&
           batch_result.information_gate_pass &&
           batch_result.residual_health_pass);
      std::string state_invalid_reason;
      batch_result.solution_valid =
          batch_result.solution_valid &&
          builder.CurrentStateFinite(&state_invalid_reason);
      if (batch_result.solution_valid) {
        batch_result.solution_valid =
            builder.CandidateCameraStepWithinTrustRegion(batch_state,
                                                         &state_invalid_reason);
      }
      batch_result.batch_accepted =
          incremental_accepted && batch_result.solution_valid;
      batch_result.committed_or_rollback =
          batch_result.batch_accepted ? "committed" : "rollback";
      if (batch_result.batch_accepted) {
        result.accepted_keys.insert(input.frame_board_keys.begin(),
                                    input.frame_board_keys.end());
        current_rank = static_cast<int>(ret.rankTheta);
        const ResidualStats stats = builder.EvaluateAccepted(
            result.accepted_keys);
        batch_result.rmse_after = stats.Rmse();
        batch_result.outer_rmse_after = stats.OuterRmse();
        batch_result.internal_rmse_after = stats.InternalRmse();
        batch_result.accept_reason =
            input.force ? "force"
                        : (ret.informationGain >
                                   estimator_options.information_gain_threshold
                               ? "incremental_information_gain"
                               : "incremental_rank_gain");
        ++result.accepted_batch_count;
      } else {
        if (incremental_accepted) {
          estimator.rejectBatch(batch);
        }
        builder.RestoreState(batch_state);
        if (!state_invalid_reason.empty()) {
          batch_result.reject_reason = state_invalid_reason;
        } else if (!batch_result.objective_finite) {
          batch_result.reject_reason = "incremental_nonfinite_objective";
        } else if (!batch_result.objective_decreased) {
          std::ostringstream stream;
          stream << "incremental_objective_increase_gate JStart="
                 << ret.JStart << " JFinal=" << ret.JFinal;
          batch_result.reject_reason = stream.str();
        } else if (!batch_result.information_gate_pass) {
          batch_result.reject_reason = "incremental_information_gain_gate";
        } else if (!batch_result.residual_health_pass) {
          std::ostringstream stream;
          stream << "incremental_residual_health_gate max_trial_rmse="
                 << input.max_trial_rmse
                 << " threshold="
                 << input.residual_health_threshold_px;
          batch_result.reject_reason = stream.str();
        } else {
          batch_result.reject_reason =
              incremental_accepted
                  ? "incremental_solution_validity_gate"
                  : "incremental_backend_rejected_batch";
        }
        ++result.rejected_batch_count;
      }
      result.batch_results.push_back(batch_result);
    }

    result.optimized_scene_state = builder.BuildSceneState();
    result.curated_bundle = BuildCuratedBundle(
        baseline_bundle, candidate_pool_bundle, result.optimized_scene_state,
        result.accepted_keys);
    result.success = result.curated_bundle.IsReadyForBackend();
    if (!result.success) {
      result.failure_reason =
          result.curated_bundle.failure_reason.empty()
              ? "persistent incremental backend produced invalid curated bundle"
              : result.curated_bundle.failure_reason;
    } else if (result.accepted_batch_count == 0) {
      result.warnings.push_back(
          "Persistent incremental estimator kept the seed backend and rejected all candidate batches.");
    }
  } catch (const std::exception& exception) {
    result.success = false;
    result.failure_reason = exception.what();
  }

  const auto time_end = std::chrono::steady_clock::now();
  result.total_elapsed_time_seconds =
      std::chrono::duration<double>(time_end - time_start).count();
  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
