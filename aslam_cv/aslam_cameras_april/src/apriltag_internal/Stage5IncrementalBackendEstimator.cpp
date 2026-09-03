#include <aslam/cameras/apriltag_internal/Stage5IncrementalBackendEstimator.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
#include <numeric>
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
#include <aslam/backend/LevenbergMarquardtTrustRegionPolicy.hpp>
#include <aslam/backend/MapTransformation.hpp>
#include <aslam/backend/MappedEuclideanPoint.hpp>
#include <aslam/backend/MappedRotationQuaternion.hpp>
#include <aslam/backend/MEstimatorPolicies.hpp>
#include <aslam/backend/Optimizer2.hpp>
#include <aslam/backend/TransformationExpression.hpp>
#include <aslam/cameras.hpp>
#include <aslam/calibration/core/IncrementalEstimator.h>
#include <aslam/calibration/core/OptimizationProblem.h>
#include <aslam/calibration/core/LinearSolverOptions.h>
#include <sm/kinematics/Transformation.hpp>

#include <aslam/cameras/apriltag_internal/AngularResidualGeometry.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>
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
using EucmGeometry = aslam::cameras::ExtendedUnifiedCameraGeometry;
using EucmProjection =
    aslam::cameras::ExtendedUnifiedProjection<aslam::cameras::NoDistortion>;
using PinholeEquiGeometry =
    aslam::cameras::EquidistantDistortedPinholeCameraGeometry;
using PinholeEquiProjection =
    aslam::cameras::PinholeProjection<aslam::cameras::EquidistantDistortion>;
using MeiGeometry = aslam::cameras::DistortedOmniCameraGeometry;
using MeiProjection =
    aslam::cameras::OmniProjection<aslam::cameras::RadialTangentialDistortion>;
using OmniNoneGeometry = aslam::cameras::OmniCameraGeometry;
using OmniNoneProjection =
    aslam::cameras::OmniProjection<aslam::cameras::NoDistortion>;
using CalibrationBatch = aslam::calibration::OptimizationProblem;

// The incremental estimator computes information gain on the marginalized
// group. Keep that group restricted to camera intrinsics; board layout and
// per-frame poses remain optimized nuisance variables.
constexpr std::size_t kCameraInformationGroupId = 0;
constexpr std::size_t kBoardLayoutGroupId = 1;
constexpr std::size_t kTransformationGroupId = 2;
constexpr int kMaxTrustRegionBacktrackingRetries = 3;
constexpr double kMaxTrustRegionAnchorWeightScale = 4096.0;

enum class CameraOptimizationPhase {
  kPosePrefitFixedIntrinsics,
  kSeedFixedIntrinsics,
  // Model-aware selection needs an information baseline before candidates are
  // tried.  This seed uses the normal shared frame/board state and every
  // valid observation; it is not an independent-pose or outer-only path.
  kSeedActiveCamera,
  // A multi-frame information seed first retains its frame poses as nuisance
  // variables while building the camera Fisher matrix, then freezes them
  // before any candidate is optimized. This preserves the correct Schur
  // marginalization without carrying those poses into later solves.
  kSeedActiveCameraFixedFramePoses,
  kCandidateTrustRegion,
};

enum class SelectionResidualMetric {
  kPixel,
  kAngularTangent,
  kLocallyWhitenedAngularTangent,
  kChordal,
  kHybridObjective,
  kPixelChordalHybridObjective,
};

enum class CameraParameterBlock {
  kProjection,
  kDistortion,
};

struct KbDistortionReleaseState {
  bool k3_released = true;
  bool k4_released = true;
};

struct KbRayCurveHealth {
  bool applicable = false;
  bool valid = true;
  int sample_count = 0;
  double rms_change_deg = 0.0;
  double max_change_deg = 0.0;
  double min_radial_derivative = 1.0;
  std::string failure_reason;
};

// Error terms are evaluated in parallel. Numerical camera Jacobians must
// perturb a private camera copy rather than the shared camera design variable.
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

bool ConvergedByBearingRelativeObjective(
    ResidualModel residual_model,
    const aslam::calibration::IncrementalEstimator::ReturnValue& value,
    int max_iterations) {
  if (residual_model == ResidualModel::ImagePlane ||
      value.linearSolverFailure ||
      value.numIterations <
          static_cast<std::size_t>(std::max(1, max_iterations)) ||
      !std::isfinite(value.JFinal) || !std::isfinite(value.dJFinal) ||
      value.dJFinal < 0.0) {
    return false;
  }
  constexpr double kBearingRelativeObjectiveTolerance = 1e-3;
  const double relative_last_objective_decrease =
      std::abs(value.dJFinal) / std::max(1.0, std::abs(value.JFinal));
  return std::isfinite(relative_last_objective_decrease) &&
         relative_last_objective_decrease <=
             kBearingRelativeObjectiveTolerance;
}

struct CameraStepConvergence {
  bool converged = false;
  double shape_step = 0.0;
  double focal_relative_step = 0.0;
  double principal_step_px = 0.0;
};

CameraStepConvergence EvaluateCameraStepConvergence(
    const OuterBootstrapCameraIntrinsics& before,
    const OuterBootstrapCameraIntrinsics& after) {
  CameraStepConvergence result;
  result.shape_step = std::max(
      std::max(std::abs(after.xi - before.xi),
               std::abs(after.alpha - before.alpha)),
      std::abs(after.beta - before.beta));
  if (before.distortion_coeffs.size() == after.distortion_coeffs.size()) {
    for (std::size_t index = 0; index < before.distortion_coeffs.size();
         ++index) {
      result.shape_step = std::max(
          result.shape_step,
          std::abs(after.distortion_coeffs[index] -
                   before.distortion_coeffs[index]));
    }
  } else {
    result.shape_step = std::numeric_limits<double>::infinity();
  }
  result.focal_relative_step = std::max(
      std::abs(after.fu - before.fu) / std::max(1.0, std::abs(before.fu)),
      std::abs(after.fv - before.fv) / std::max(1.0, std::abs(before.fv)));
  result.principal_step_px =
      std::max(std::abs(after.cu - before.cu),
               std::abs(after.cv - before.cv));
  // These thresholds are parameter-scale based, independent of residual
  // units and image count. They only terminate continuation after pose prefit.
  constexpr double kShapeStepTolerance = 5e-4;
  constexpr double kFocalRelativeStepTolerance = 5e-4;
  constexpr double kPrincipalStepTolerancePx = 0.25;
  result.converged = std::isfinite(result.shape_step) &&
      std::isfinite(result.focal_relative_step) &&
      std::isfinite(result.principal_step_px) &&
      result.shape_step <= kShapeStepTolerance &&
      result.focal_relative_step <= kFocalRelativeStepTolerance &&
      result.principal_step_px <= kPrincipalStepTolerancePx;
  return result;
}

bool ConvergedByBearingRelativeObjective(
    ResidualModel residual_model,
    const aslam::backend::SolutionReturnValue& value,
    int max_iterations) {
  if (residual_model == ResidualModel::ImagePlane ||
      value.linearSolverFailure ||
      value.iterations < std::max(1, max_iterations) ||
      !std::isfinite(value.JFinal) || !std::isfinite(value.dJFinal) ||
      value.dJFinal < 0.0) {
    return false;
  }
  constexpr double kBearingRelativeObjectiveTolerance = 1e-3;
  const double relative_last_objective_decrease =
      std::abs(value.dJFinal) / std::max(1.0, std::abs(value.JFinal));
  return std::isfinite(relative_last_objective_decrease) &&
         relative_last_objective_decrease <=
             kBearingRelativeObjectiveTolerance;
}

SelectionResidualMetric SelectionMetricForResidualModel(
    ResidualModel residual_model,
    bool angular_local_whitening_enabled) {
  switch (residual_model) {
    case ResidualModel::SphereAngular:
      return angular_local_whitening_enabled
                 ? SelectionResidualMetric::kLocallyWhitenedAngularTangent
                 : SelectionResidualMetric::kAngularTangent;
    case ResidualModel::NormalizedSphereAngular:
      return SelectionResidualMetric::kAngularTangent;
    case ResidualModel::Chordal:
      return SelectionResidualMetric::kChordal;
    case ResidualModel::PixelChordalHybrid:
      return SelectionResidualMetric::kPixelChordalHybridObjective;
    case ResidualModel::HybridEdgeAngular:
    case ResidualModel::PolarContinuousHybrid:
      return SelectionResidualMetric::kHybridObjective;
    case ResidualModel::ImagePlane:
      return SelectionResidualMetric::kPixel;
  }
  return SelectionResidualMetric::kPixel;
}

const char* SelectionMetricName(SelectionResidualMetric metric) {
  switch (metric) {
    case SelectionResidualMetric::kPixel:
      return "pixel";
    case SelectionResidualMetric::kAngularTangent:
      return "tangent_angular";
    case SelectionResidualMetric::kLocallyWhitenedAngularTangent:
      return "locally_whitened_tangent_angular";
    case SelectionResidualMetric::kChordal:
      return "chordal";
    case SelectionResidualMetric::kHybridObjective:
      return "hybrid_objective";
    case SelectionResidualMetric::kPixelChordalHybridObjective:
      return "pixel_chordal_hybrid_objective";
  }
  return "pixel";
}

const char* SelectionMetricUnit(SelectionResidualMetric metric) {
  switch (metric) {
    case SelectionResidualMetric::kPixel:
      return "px";
    case SelectionResidualMetric::kAngularTangent:
      return "rad";
    case SelectionResidualMetric::kLocallyWhitenedAngularTangent:
      return "normalized";
    case SelectionResidualMetric::kChordal:
      return "chordal";
    case SelectionResidualMetric::kHybridObjective:
      return "px_equivalent";
    case SelectionResidualMetric::kPixelChordalHybridObjective:
      return "px_equivalent";
  }
  return "px";
}

bool SelectionMetricUsesAngularHealth(SelectionResidualMetric metric) {
  return metric == SelectionResidualMetric::kAngularTangent ||
         metric == SelectionResidualMetric::kChordal;
}

bool SelectionMetricUsesResidualAwareScoreGate(
    SelectionResidualMetric metric) {
  return metric == SelectionResidualMetric::kAngularTangent ||
         metric == SelectionResidualMetric::kLocallyWhitenedAngularTangent ||
         metric == SelectionResidualMetric::kChordal ||
         metric == SelectionResidualMetric::kHybridObjective ||
         metric == SelectionResidualMetric::kPixelChordalHybridObjective;
}

bool SelectionMetricIsHybridObjective(SelectionResidualMetric metric) {
  return metric == SelectionResidualMetric::kHybridObjective ||
         metric == SelectionResidualMetric::kPixelChordalHybridObjective;
}

bool IsBearingResidualModel(ResidualModel model) {
  return model != ResidualModel::ImagePlane;
}

bool ResidualConstructionMatchesMode(ResidualModel model,
                                     double pixel_weight,
                                     double chordal_weight,
                                     int image_plane_count,
                                     int angular_count,
                                     int chordal_count,
                                     std::string* reason) {
  const int total_count = image_plane_count + angular_count + chordal_count;
  bool matches = total_count > 0;
  switch (model) {
    case ResidualModel::ImagePlane:
      matches = matches && image_plane_count > 0 && angular_count == 0 &&
                chordal_count == 0;
      break;
    case ResidualModel::SphereAngular:
      matches = matches && image_plane_count == 0 && angular_count > 0 &&
                chordal_count == 0;
      break;
    case ResidualModel::HybridEdgeAngular:
    case ResidualModel::PolarContinuousHybrid:
      matches = matches && chordal_count == 0 &&
                image_plane_count + angular_count > 0;
      break;
    case ResidualModel::Chordal:
      matches = matches && image_plane_count == 0 && angular_count == 0 &&
                chordal_count > 0;
      break;
    case ResidualModel::PixelChordalHybrid:
      matches = matches && angular_count == 0 &&
                (pixel_weight <= 0.0 || image_plane_count > 0) &&
                (chordal_weight <= 0.0 || chordal_count > 0);
      break;
    case ResidualModel::NormalizedSphereAngular:
      matches = false;
      break;
  }
  if (!matches && reason != nullptr) {
    std::ostringstream stream;
    stream << "residual_mode_contract_violation model=" << ToString(model)
           << " image_plane=" << image_plane_count
           << " angular=" << angular_count
           << " chordal=" << chordal_count;
    *reason = stream.str();
  } else if (reason != nullptr) {
    reason->clear();
  }
  return matches;
}

boost::shared_ptr<DsGeometry> MakeDsGeometry(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  DsProjection projection(intrinsics.xi, intrinsics.alpha, intrinsics.fu,
                          intrinsics.fv, intrinsics.cu, intrinsics.cv,
                          intrinsics.resolution.width,
                          intrinsics.resolution.height);
  return boost::make_shared<DsGeometry>(
      projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask());
}

boost::shared_ptr<EucmGeometry> MakeEucmGeometry(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  EucmProjection projection(intrinsics.alpha, intrinsics.beta, intrinsics.fu,
                            intrinsics.fv, intrinsics.cu, intrinsics.cv,
                            intrinsics.resolution.width,
                            intrinsics.resolution.height);
  return boost::make_shared<EucmGeometry>(
      projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask());
}

boost::shared_ptr<PinholeEquiGeometry> MakePinholeEquiGeometry(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  const std::vector<double> distortion =
      intrinsics.distortion_coeffs.size() == 4u
          ? intrinsics.distortion_coeffs
          : std::vector<double>{0.0, 0.0, 0.0, 0.0};
  PinholeEquiProjection projection(
      intrinsics.fu, intrinsics.fv, intrinsics.cu, intrinsics.cv,
      intrinsics.resolution.width, intrinsics.resolution.height,
      aslam::cameras::EquidistantDistortion(
          distortion[0], distortion[1], distortion[2], distortion[3]));
  return boost::make_shared<PinholeEquiGeometry>(
      projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask());
}

boost::shared_ptr<MeiGeometry> MakeMeiGeometry(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  const std::vector<double> distortion =
      intrinsics.distortion_coeffs.size() == 4u
          ? intrinsics.distortion_coeffs
          : std::vector<double>{0.0, 0.0, 0.0, 0.0};
  MeiProjection projection(
      intrinsics.xi, intrinsics.fu, intrinsics.fv, intrinsics.cu,
      intrinsics.cv, intrinsics.resolution.width, intrinsics.resolution.height,
      aslam::cameras::RadialTangentialDistortion(
          distortion[0], distortion[1], distortion[2], distortion[3]));
  return boost::make_shared<MeiGeometry>(
      projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask());
}

boost::shared_ptr<OmniNoneGeometry> MakeOmniNoneGeometry(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  OmniNoneProjection projection(
      intrinsics.xi, intrinsics.fu, intrinsics.fv, intrinsics.cu,
      intrinsics.cv, intrinsics.resolution.width, intrinsics.resolution.height);
  return boost::make_shared<OmniNoneGeometry>(
      projection, aslam::cameras::GlobalShutter(), aslam::cameras::NoMask());
}

template <typename GeometryT>
boost::shared_ptr<GeometryT> MakePersistentGeometry(
    const OuterBootstrapCameraIntrinsics& intrinsics);

template <>
boost::shared_ptr<DsGeometry> MakePersistentGeometry<DsGeometry>(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  return MakeDsGeometry(intrinsics);
}

template <>
boost::shared_ptr<EucmGeometry> MakePersistentGeometry<EucmGeometry>(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  return MakeEucmGeometry(intrinsics);
}

template <>
boost::shared_ptr<PinholeEquiGeometry>
MakePersistentGeometry<PinholeEquiGeometry>(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  return MakePinholeEquiGeometry(intrinsics);
}

template <>
boost::shared_ptr<MeiGeometry> MakePersistentGeometry<MeiGeometry>(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  return MakeMeiGeometry(intrinsics);
}

template <>
boost::shared_ptr<OmniNoneGeometry> MakePersistentGeometry<OmniNoneGeometry>(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  return MakeOmniNoneGeometry(intrinsics);
}

template <typename GeometryT>
OuterBootstrapCameraIntrinsics CameraToIntrinsics(const GeometryT& geometry);

template <>
OuterBootstrapCameraIntrinsics CameraToIntrinsics<DsGeometry>(
    const DsGeometry& geometry) {
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
OuterBootstrapCameraIntrinsics CameraToIntrinsics<EucmGeometry>(
    const EucmGeometry& geometry) {
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
OuterBootstrapCameraIntrinsics CameraToIntrinsics<PinholeEquiGeometry>(
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
OuterBootstrapCameraIntrinsics CameraToIntrinsics<MeiGeometry>(
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
OuterBootstrapCameraIntrinsics CameraToIntrinsics<OmniNoneGeometry>(
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

IntermediateCameraConfig MakePersistentIntermediateCameraConfig(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  IntermediateCameraConfig config;
  config.camera_model = intrinsics.camera_model;
  config.distortion_model = intrinsics.distortion_model;
  config.intrinsics = intrinsics.IntrinsicsVector();
  config.distortion_coeffs = intrinsics.DistortionVector();
  config.resolution = {intrinsics.resolution.width,
                       intrinsics.resolution.height};
  return config;
}

template <typename GeometryT>
bool ComputeLocalBearingWhiteningForCamera(
    const GeometryT&,
    const Eigen::Vector2d&,
    const AngularObservationGeometry&,
    const Stage5IncrementalBackendEstimatorOptions&,
    BearingCovarianceResult*) {
  return false;
}

template <>
bool ComputeLocalBearingWhiteningForCamera<DsGeometry>(
    const DsGeometry& camera,
    const Eigen::Vector2d& observed_image_xy,
    const AngularObservationGeometry& observation_geometry,
    const Stage5IncrementalBackendEstimatorOptions& options,
    BearingCovarianceResult* result) {
  BearingCovarianceOptions covariance_options;
  covariance_options.use_pixel_uncertainty = true;
  covariance_options.use_model_uncertainty = false;
  covariance_options.pixel_sigma_px =
      options.angular_local_whitening_pixel_sigma_px;
  covariance_options.covariance_damping =
      options.angular_local_whitening_covariance_damping;
  covariance_options.min_sigma_rad =
      options.angular_local_whitening_min_sigma_rad;
  covariance_options.max_whitening_weight =
      options.angular_local_whitening_max_weight;
  return ComputeBearingTangentCovariance(
      MakePersistentIntermediateCameraConfig(
          CameraToIntrinsics<DsGeometry>(camera)),
      observed_image_xy, observation_geometry, covariance_options, result);
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
         << " beta=" << intrinsics.beta
         << " fu=" << intrinsics.fu << " fv=" << intrinsics.fv
         << " cu=" << intrinsics.cu << " cv=" << intrinsics.cv
         << " width=" << intrinsics.resolution.width
         << " height=" << intrinsics.resolution.height
         << " family=" << intrinsics.NormalizedFamilyString();
  return stream.str();
}

template <typename GeometryT>
bool IsCameraStateValid(const GeometryT& geometry, std::string* reason) {
  const OuterBootstrapCameraIntrinsics intrinsics =
      CameraToIntrinsics<GeometryT>(geometry);
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
  if (intrinsics.NormalizedFamilyString() == "ds-none" &&
      (!(intrinsics.alpha > 1e-4 && intrinsics.alpha < 0.999) ||
       !(intrinsics.xi > -1.5 && intrinsics.xi < 2.5))) {
    if (reason != nullptr) {
      *reason = "physically_implausible_ds_shape " +
                FormatDsCameraForReason(intrinsics);
    }
    return false;
  }
  if (intrinsics.NormalizedFamilyString() == "eucm-none" &&
      (!(intrinsics.alpha > 1e-4 && intrinsics.alpha < 0.999) ||
       !(intrinsics.beta > 1e-4 && intrinsics.beta < 5.0))) {
    if (reason != nullptr) {
      *reason = "physically_implausible_eucm_shape " +
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

void FillDistortionDiagnostics(const OuterBootstrapCameraIntrinsics& camera,
                               double* k1,
                               double* k2,
                               double* k3,
                               double* k4) {
  const std::vector<double> distortion = camera.DistortionVector();
  double* outputs[] = {k1, k2, k3, k4};
  for (std::size_t index = 0; index < 4u; ++index) {
    if (outputs[index] != nullptr) {
      *outputs[index] = index < distortion.size() ? distortion[index] : 0.0;
    }
  }
}

template <int Dimension>
class ProjectionAnchorError : public aslam::backend::ErrorTermFs<Dimension> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ProjectionAnchorError(
      aslam::backend::DesignVariable* projection_dv,
      const Eigen::MatrixXd& anchor,
      const Eigen::VectorXd& prior_weight)
      : projection_dv_(projection_dv) {
    if (projection_dv_ == nullptr || anchor.cols() != 1 ||
        anchor.rows() != Dimension || prior_weight.rows() != Dimension ||
        prior_weight.cols() != 1) {
      throw std::runtime_error(
          "ProjectionAnchorError requires valid projection prior data.");
    }
    for (int index = 0; index < Dimension; ++index) {
      anchor_[index] = anchor(index, 0);
      prior_weight_[index] = prior_weight[index];
    }
    const Eigen::Matrix<double, Dimension, Dimension> inverse_covariance =
        prior_weight_.asDiagonal();
    parent_t::setInvR(inverse_covariance);
    parent_t::setDesignVariables(projection_dv_);
  }

 protected:
  double evaluateErrorImplementation() override {
    Eigen::MatrixXd parameters_matrix;
    projection_dv_->getParameters(parameters_matrix);
    typename parent_t::error_t parameters = parent_t::error_t::Zero();
    const Eigen::Map<const Eigen::VectorXd> flat(parameters_matrix.data(),
                                                parameters_matrix.size());
    for (int index = 0; index < Dimension; ++index) {
      parameters[index] = flat[index];
    }
    parent_t::setError(parameters - anchor_);
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    jacobians.add(projection_dv_,
                  Eigen::Matrix<double, Dimension, Dimension>::Identity());
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<Dimension>;

  aslam::backend::DesignVariable* projection_dv_ = nullptr;
  Eigen::Matrix<double, Dimension, 1> anchor_ =
      Eigen::Matrix<double, Dimension, 1>::Zero();
  Eigen::Matrix<double, Dimension, 1> prior_weight_ =
      Eigen::Matrix<double, Dimension, 1>::Ones();
};

class SquarePixelFocalError : public aslam::backend::ErrorTermFs<1> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SquarePixelFocalError(
      aslam::backend::DesignVariable* projection_dv)
      : projection_dv_(projection_dv) {
    if (projection_dv_ == nullptr) {
      throw std::runtime_error(
          "SquarePixelFocalError requires a projection design variable.");
    }
    // This is an effectively hard physical square-pixel constraint while
    // retaining the existing projection design variable and rollback path.
    constexpr double kFocalDifferenceSigmaPx = 0.01;
    Eigen::Matrix<double, 1, 1> inverse_covariance;
    inverse_covariance(0, 0) =
        1.0 / (kFocalDifferenceSigmaPx * kFocalDifferenceSigmaPx);
    parent_t::setInvR(inverse_covariance);
    parent_t::setDesignVariables(projection_dv_);
  }

 protected:
  double evaluateErrorImplementation() override {
    Eigen::MatrixXd parameters_matrix;
    projection_dv_->getParameters(parameters_matrix);
    const Eigen::Map<const Eigen::VectorXd> parameters(
        parameters_matrix.data(), parameters_matrix.size());
    typename parent_t::error_t error;
    error[0] = parameters[0] - parameters[1];
    parent_t::setError(error);
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(
        1, projection_dv_->minimalDimensions());
    if (jacobian.cols() < 2) {
      throw std::runtime_error(
          "SquarePixelFocalError projection has fewer than two parameters.");
    }
    jacobian(0, 0) = 1.0;
    jacobian(0, 1) = -1.0;
    jacobians.add(projection_dv_, jacobian);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<1>;
  aslam::backend::DesignVariable* projection_dv_ = nullptr;
};

double PositiveOrDefault(double value, double fallback) {
  return value > 0.0 && std::isfinite(value) ? value : fallback;
}

double NextTrustRegionAnchorWeightScale(double current_scale,
                                        double violation_ratio) {
  const double safe_current =
      std::isfinite(current_scale) && current_scale > 1.0 ? current_scale : 1.0;
  const double safe_ratio =
      std::isfinite(violation_ratio) && violation_ratio > 1.0
          ? violation_ratio
          : 2.0;
  const double multiplier =
      std::min(64.0, std::max(4.0, 1.5 * safe_ratio * safe_ratio));
  return std::min(kMaxTrustRegionAnchorWeightScale,
                  std::max(safe_current + 1.0, safe_current * multiplier));
}

double PriorWeightFromSigma(double sigma) {
  const double safe_sigma = PositiveOrDefault(sigma, 1.0);
  return 1.0 / (safe_sigma * safe_sigma);
}

template <typename GeometryT>
class IncrementalReprojectionError : public aslam::backend::ErrorTermFs<2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using CameraDv = aslam::backend::CameraDesignVariable<GeometryT>;

  IncrementalReprojectionError(
      const Eigen::Vector2d& measurement,
      const Eigen::Matrix2d& inverse_covariance,
      double huber_delta_pixels,
      bool use_huber_loss,
      const aslam::backend::HomogeneousExpression& point_camera,
      const CameraDv& camera_dv,
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
    typename GeometryT::jacobian_homogeneous_t projection_jacobian;
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
  const CameraDv& camera_dv_;
  Eigen::Matrix2d inverse_covariance_ = Eigen::Matrix2d::Identity();
  double invalid_projection_penalty_pixels_ = 100.0;
};

template <typename GeometryT>
class IncrementalAngularReprojectionError
    : public aslam::backend::ErrorTermFs<2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using CameraDv = aslam::backend::CameraDesignVariable<GeometryT>;

  IncrementalAngularReprojectionError(
      const Eigen::Vector2d& observed_image_xy,
      const Eigen::Matrix2d& inverse_covariance,
      double huber_delta_radians,
      bool use_huber_loss,
      const aslam::backend::HomogeneousExpression& point_camera,
      const CameraDv& camera_dv,
      double invalid_projection_penalty_radians,
      bool use_normalize_jacobian,
      AslamBackendCalibrationOptions::AngularObservedRayMode observed_ray_mode,
      const AngularObservationGeometry& frozen_observation_geometry)
      : observed_image_xy_(observed_image_xy),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        inverse_covariance_(inverse_covariance.cwiseMax(0.0)),
        invalid_projection_penalty_radians_(invalid_projection_penalty_radians),
        use_normalize_jacobian_(use_normalize_jacobian),
        observed_ray_mode_(observed_ray_mode),
        frozen_observation_geometry_(frozen_observation_geometry) {
    parent_t::setInvR(inverse_covariance_);
    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    camera_dv_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(),
                                         design_variables.end());
    const double balance_weight =
        0.5 * (inverse_covariance_(0, 0) + inverse_covariance_(1, 1));
    if (use_huber_loss && huber_delta_radians > 0.0 &&
        balance_weight > 0.0) {
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(
                  std::sqrt(balance_weight) * huber_delta_radians)));
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
    const Eigen::Vector2d residual =
        ComputeResidual(&observation_geometry, &prediction_geometry,
                        &valid_projection);
    (void)residual;
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

    const Eigen::Matrix<double, 2, 3> d_residual_d_point =
        observation_geometry.tangent_basis.transpose() * d_unit_d_point;
    const Eigen::Matrix<double, 2, 4> d_residual_d_homogeneous =
        (Eigen::Matrix<double, 2, 4>() <<
             d_residual_d_point(0, 0), d_residual_d_point(0, 1),
             d_residual_d_point(0, 2), 0.0,
             d_residual_d_point(1, 0), d_residual_d_point(1, 1),
             d_residual_d_point(1, 2), 0.0)
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
    AddThreadSafeCameraFiniteDifferenceJacobian<2>(
        const_cast<CameraDv&>(camera_dv_).projectionDesignVariable(),
        base_camera, CameraParameterBlock::kProjection, evaluate_residual,
        &jacobians);
    AddThreadSafeCameraFiniteDifferenceJacobian<2>(
        const_cast<CameraDv&>(camera_dv_).distortionDesignVariable(),
        base_camera, CameraParameterBlock::kDistortion, evaluate_residual,
        &jacobians);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<2>;

  Eigen::Vector2d ComputeResidual(
      AngularObservationGeometry* observation_geometry,
      AngularPredictionGeometry* prediction_geometry,
      bool* valid_projection) const {
    return ComputeResidualForCamera(*camera_dv_.camera(), observation_geometry,
                                    prediction_geometry, valid_projection);
  }

  Eigen::Vector2d ComputeResidualForCamera(
      const GeometryT& camera,
      AngularObservationGeometry* observation_geometry,
      AngularPredictionGeometry* prediction_geometry,
      bool* valid_projection) const {
    if (observation_geometry == nullptr || prediction_geometry == nullptr ||
        valid_projection == nullptr) {
      throw std::runtime_error(
          "IncrementalAngularReprojectionError requires valid output pointers.");
    }
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
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    const Eigen::Vector3d point_camera = point_homogeneous.head<3>();
    *valid_projection = observation_valid &&
        ComputePredictionGeometryForCamera(camera, point_homogeneous,
                                           prediction_geometry);
    if (!(*valid_projection) || !observation_geometry->success) {
      return Eigen::Vector2d::Constant(invalid_projection_penalty_radians_);
    }
    return ComputeAngularResidualTangent(*observation_geometry,
                                         *prediction_geometry);
  }

  Eigen::Vector2d observed_image_xy_ = Eigen::Vector2d::Zero();
  aslam::backend::HomogeneousExpression point_camera_;
  const CameraDv& camera_dv_;
  Eigen::Matrix2d inverse_covariance_ = Eigen::Matrix2d::Identity();
  double invalid_projection_penalty_radians_ = 0.35;
  bool use_normalize_jacobian_ = false;
  AslamBackendCalibrationOptions::AngularObservedRayMode observed_ray_mode_ =
      AslamBackendCalibrationOptions::AngularObservedRayMode::
          DynamicCurrentCamera;
  AngularObservationGeometry frozen_observation_geometry_;
};

template <typename GeometryT>
class IncrementalChordalReprojectionError
    : public aslam::backend::ErrorTermFs<3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using CameraDv = aslam::backend::CameraDesignVariable<GeometryT>;

  IncrementalChordalReprojectionError(
      const Eigen::Vector2d& observed_image_xy,
      const Eigen::Matrix3d& inverse_covariance,
      double huber_delta_chordal,
      bool use_huber_loss,
      const aslam::backend::HomogeneousExpression& point_camera,
      const CameraDv& camera_dv,
      double invalid_projection_penalty_chordal,
      bool use_normalize_jacobian,
      AslamBackendCalibrationOptions::AngularObservedRayMode observed_ray_mode,
      const AngularObservationGeometry& frozen_observation_geometry)
      : observed_image_xy_(observed_image_xy),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        inverse_covariance_(inverse_covariance.cwiseMax(0.0)),
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
    const double balance_weight =
        (inverse_covariance_(0, 0) + inverse_covariance_(1, 1) +
         inverse_covariance_(2, 2)) /
        3.0;
    if (use_huber_loss && huber_delta_chordal > 0.0 &&
        balance_weight > 0.0) {
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(
                  std::sqrt(balance_weight) * huber_delta_chordal)));
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

    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
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
        const_cast<CameraDv&>(camera_dv_).projectionDesignVariable(),
        base_camera, CameraParameterBlock::kProjection, evaluate_residual,
        &jacobians);
    AddThreadSafeCameraFiniteDifferenceJacobian<3>(
        const_cast<CameraDv&>(camera_dv_).distortionDesignVariable(),
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
          "IncrementalChordalReprojectionError requires valid output pointers.");
    }
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
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    const Eigen::Vector3d point_camera = point_homogeneous.head<3>();
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
  const CameraDv& camera_dv_;
  Eigen::Matrix3d inverse_covariance_ = Eigen::Matrix3d::Identity();
  double invalid_projection_penalty_chordal_ = 0.35;
  bool use_normalize_jacobian_ = false;
  AslamBackendCalibrationOptions::AngularObservedRayMode observed_ray_mode_ =
      AslamBackendCalibrationOptions::AngularObservedRayMode::
          DynamicCurrentCamera;
  AngularObservationGeometry frozen_observation_geometry_;
};

template <typename GeometryT>
class IncrementalPolarContinuousHybridReprojectionError
    : public aslam::backend::ErrorTermFs<2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using CameraDv = aslam::backend::CameraDesignVariable<GeometryT>;

  IncrementalPolarContinuousHybridReprojectionError(
      const Eigen::Vector2d& observed_image_xy,
      const Eigen::Matrix2d& inverse_covariance,
      double huber_delta_pixels,
      bool use_huber_loss,
      const aslam::backend::HomogeneousExpression& point_camera,
      const CameraDv& camera_dv,
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
        camera_dv_.camera()->homogeneousToKeypoint(point_homogeneous,
                                                   predicted_image,
                                                   projection_jacobian) &&
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
        const_cast<CameraDv&>(camera_dv_).projectionDesignVariable(),
        base_camera, CameraParameterBlock::kProjection, evaluate_residual,
        &jacobians);
    AddThreadSafeCameraFiniteDifferenceJacobian<2>(
        const_cast<CameraDv&>(camera_dv_).distortionDesignVariable(),
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
          "IncrementalPolarContinuousHybridReprojectionError requires output.");
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
  const CameraDv& camera_dv_;
  Eigen::Matrix2d inverse_covariance_ = Eigen::Matrix2d::Identity();
  double invalid_projection_penalty_pixels_ = 100.0;
  double invalid_projection_penalty_radians_ = 0.35;
  bool use_normalize_jacobian_ = false;
  AslamBackendCalibrationOptions::AngularObservedRayMode observed_ray_mode_ =
      AslamBackendCalibrationOptions::AngularObservedRayMode::
          DynamicCurrentCamera;
  AngularObservationGeometry frozen_observation_geometry_;
  double threshold_deg_ = 50.0;
  double temperature_deg_ = 10.0;
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
  struct KeyStats {
    int count = 0;
    double squared_error = 0.0;
    int outer_count = 0;
    double outer_squared_error = 0.0;

    double Rmse() const {
      return count > 0
                 ? std::sqrt(squared_error / static_cast<double>(count))
                 : 0.0;
    }
    double OuterRmse() const {
      return outer_count > 0
                 ? std::sqrt(outer_squared_error /
                             static_cast<double>(outer_count))
                 : 0.0;
    }
  };

  int total_count = 0;
  int outer_count = 0;
  int internal_count = 0;
  double total_squared_error = 0.0;
  double outer_squared_error = 0.0;
  double internal_squared_error = 0.0;
  int invalid_projection_count = 0;
  int invalid_outer_projection_count = 0;
  int invalid_internal_projection_count = 0;
  std::vector<double> residual_norms;
  std::vector<double> outer_residual_norms;
  std::vector<double> internal_residual_norms;
  std::map<FrameBoardKey, KeyStats> frame_board_stats;

  static double Percentile(std::vector<double> values, double quantile) {
    if (values.empty()) {
      return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double clamped_quantile = std::max(0.0, std::min(1.0, quantile));
    const double position =
        clamped_quantile * static_cast<double>(values.size() - 1u);
    const std::size_t lower =
        static_cast<std::size_t>(std::floor(position));
    const std::size_t upper =
        std::min(values.size() - 1u, lower + 1u);
    const double alpha = position - static_cast<double>(lower);
    return (1.0 - alpha) * values[lower] + alpha * values[upper];
  }

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
  double P95() const { return Percentile(residual_norms, 0.95); }
  double OuterP95() const {
    return Percentile(outer_residual_norms, 0.95);
  }
  double InternalP95() const {
    return Percentile(internal_residual_norms, 0.95);
  }
  double WorstFrameBoardRmse() const {
    double worst = 0.0;
    for (const auto& entry : frame_board_stats) {
      worst = std::max(worst, entry.second.Rmse());
    }
    return worst;
  }
  void Add(const FrameBoardKey& key,
           JointPointType point_type,
           double residual_norm) {
    const double squared_error = residual_norm * residual_norm;
    ++total_count;
    total_squared_error += squared_error;
    residual_norms.push_back(residual_norm);
    KeyStats& key_stats = frame_board_stats[key];
    ++key_stats.count;
    key_stats.squared_error += squared_error;
    if (point_type == JointPointType::Outer) {
      ++outer_count;
      outer_squared_error += squared_error;
      outer_residual_norms.push_back(residual_norm);
      ++key_stats.outer_count;
      key_stats.outer_squared_error += squared_error;
    } else {
      ++internal_count;
      internal_squared_error += squared_error;
      internal_residual_norms.push_back(residual_norm);
    }
  }
};

struct FullTrainingPoseRefitStats {
  ResidualStats pixel_stats;
  std::map<FrameBoardKey, Eigen::Isometry3d> fitted_poses;
  // This is deliberately diagnostic-only.  The health gate still evaluates
  // the same complete observation set; these fields explain a count change
  // instead of silently interpreting it as a bad frame or a bad camera.
  std::map<FrameBoardKey, std::string> pose_failure_reasons;
  std::map<FrameBoardKey, int> point_count_by_key;
  std::map<FrameBoardKey, int> outer_point_count_by_key;
  std::map<FrameBoardKey, int> internal_point_count_by_key;
  std::map<FrameBoardKey, int> invalid_projection_count_by_key;
  std::map<FrameBoardKey, int> invalid_outer_projection_count_by_key;
  std::map<FrameBoardKey, int> invalid_internal_projection_count_by_key;
  std::map<FrameBoardKey, double> pose_rmse_by_key;
  int pose_total_count = 0;
  int pose_success_count = 0;
  int point_total_count = 0;
  bool camera_valid = false;

  double PoseSuccessRate() const {
    return pose_total_count > 0
               ? static_cast<double>(pose_success_count) /
                     static_cast<double>(pose_total_count)
               : 0.0;
  }

  bool IsUsable() const {
    return camera_valid && pose_total_count > 0 && point_total_count > 0 &&
           pose_success_count > 0 && pixel_stats.total_count > 0 &&
           std::isfinite(pixel_stats.Rmse()) &&
           std::isfinite(pixel_stats.P95());
  }
};

struct TrainingRobustCheckpointStats {
  bool usable = false;
  double frame_median_rmse = 0.0;
  double frame_p90_rmse = 0.0;
  double huber15_rmse = 0.0;
  double fold_median_mean_rmse = 0.0;
  double fold_median_max_rmse = 0.0;
};

TrainingRobustCheckpointStats SummarizeTrainingRobustCheckpoint(
    const FullTrainingPoseRefitStats& stats,
    const CalibrationSceneState& scene) {
  TrainingRobustCheckpointStats summary;
  if (!stats.IsUsable()) {
    return summary;
  }

  struct FrameAccumulator {
    int count = 0;
    double squared_error = 0.0;
  };
  std::map<int, FrameAccumulator> frame_accumulators;
  for (const auto& entry : stats.pixel_stats.frame_board_stats) {
    FrameAccumulator& frame = frame_accumulators[entry.first.first];
    frame.count += entry.second.count;
    frame.squared_error += entry.second.squared_error;
  }
  std::map<int, std::string> frame_labels;
  for (const JointSceneFrameState& frame : scene.frames) {
    frame_labels[frame.frame_index] = frame.frame_label;
  }

  std::vector<double> frame_rmses;
  std::array<std::vector<double>, 5> fold_frame_rmses;
  for (const auto& entry : frame_accumulators) {
    if (entry.second.count <= 0 ||
        !std::isfinite(entry.second.squared_error)) {
      continue;
    }
    const double rmse = std::sqrt(
        entry.second.squared_error / static_cast<double>(entry.second.count));
    if (!std::isfinite(rmse)) {
      continue;
    }
    frame_rmses.push_back(rmse);
    const auto label_it = frame_labels.find(entry.first);
    const std::string label =
        label_it != frame_labels.end() && !label_it->second.empty()
            ? label_it->second
            : std::to_string(entry.first);
    std::uint32_t hash = 2166136261u;
    for (const unsigned char ch : label) {
      hash ^= static_cast<std::uint32_t>(ch);
      hash *= 16777619u;
    }
    fold_frame_rmses[hash % fold_frame_rmses.size()].push_back(rmse);
  }
  if (frame_rmses.empty() || stats.pixel_stats.residual_norms.empty()) {
    return summary;
  }
  summary.frame_median_rmse =
      ResidualStats::Percentile(frame_rmses, 0.5);
  summary.frame_p90_rmse =
      ResidualStats::Percentile(frame_rmses, 0.9);

  std::vector<double> fold_medians;
  for (const std::vector<double>& fold_values : fold_frame_rmses) {
    if (!fold_values.empty()) {
      fold_medians.push_back(
          ResidualStats::Percentile(fold_values, 0.5));
    }
  }
  if (fold_medians.size() != fold_frame_rmses.size()) {
    return summary;
  }
  summary.fold_median_mean_rmse =
      std::accumulate(fold_medians.begin(), fold_medians.end(), 0.0) /
      static_cast<double>(fold_medians.size());
  summary.fold_median_max_rmse =
      *std::max_element(fold_medians.begin(), fold_medians.end());

  double huber_cost_sum = 0.0;
  constexpr double kHuberDeltaPixels = 1.5;
  for (const double residual : stats.pixel_stats.residual_norms) {
    if (!std::isfinite(residual)) {
      return TrainingRobustCheckpointStats();
    }
    huber_cost_sum +=
        residual <= kHuberDeltaPixels
            ? 0.5 * residual * residual
            : kHuberDeltaPixels *
                  (residual - 0.5 * kHuberDeltaPixels);
  }
  summary.huber15_rmse = std::sqrt(
      2.0 * huber_cost_sum /
      static_cast<double>(stats.pixel_stats.residual_norms.size()));
  summary.usable =
      std::isfinite(summary.frame_median_rmse) &&
      std::isfinite(summary.frame_p90_rmse) &&
      std::isfinite(summary.huber15_rmse) &&
      std::isfinite(summary.fold_median_mean_rmse) &&
      std::isfinite(summary.fold_median_max_rmse);
  return summary;
}

bool IsBetterTrainingRobustCheckpoint(
    const TrainingRobustCheckpointStats& candidate,
    const TrainingRobustCheckpointStats& best) {
  if (!candidate.usable) {
    return false;
  }
  if (!best.usable) {
    return true;
  }
  // The fold mean is the selection objective. Other metrics are guards that
  // prevent a narrow subset from improving at the expense of the center or
  // tail. These are training-only diagnostics and never inspect holdout data.
  constexpr double kGuardRelativeTolerance = 0.01;
  return candidate.fold_median_mean_rmse < best.fold_median_mean_rmse &&
      candidate.fold_median_max_rmse <=
          (1.0 + kGuardRelativeTolerance) *
              best.fold_median_max_rmse &&
      candidate.frame_median_rmse <=
          (1.0 + kGuardRelativeTolerance) * best.frame_median_rmse &&
      candidate.frame_p90_rmse <=
          (1.0 + kGuardRelativeTolerance) * best.frame_p90_rmse &&
      candidate.huber15_rmse <=
          (1.0 + kGuardRelativeTolerance) * best.huber15_rmse;
}

struct InstabilityQuarantineResult {
  std::set<FrameBoardKey> keys;
  bool within_budget = true;
  double candidate_rmse_threshold_px = 0.0;
  double regression_threshold_px = 0.0;
  int max_quarantine_count = 0;
  std::string reason;
};

double MedianAbsoluteDeviation(const std::vector<double>& values,
                               double median) {
  std::vector<double> deviations;
  deviations.reserve(values.size());
  for (double value : values) {
    deviations.push_back(std::abs(value - median));
  }
  return ResidualStats::Percentile(deviations, 0.5);
}

InstabilityQuarantineResult IdentifyUnstableFrameBoards(
    const FullTrainingPoseRefitStats& reference_stats,
    const FullTrainingPoseRefitStats& candidate_stats,
    int reference_board_id,
    const Stage5IncrementalBackendEstimatorOptions& options) {
  InstabilityQuarantineResult result;
  if (!options.full_training_instability_quarantine_enabled ||
      !reference_stats.IsUsable() || !candidate_stats.IsUsable()) {
    return result;
  }

  std::vector<double> candidate_rmses;
  std::vector<double> regressions;
  for (const auto& entry : candidate_stats.pixel_stats.frame_board_stats) {
    const auto reference_it =
        reference_stats.pixel_stats.frame_board_stats.find(entry.first);
    if (reference_it == reference_stats.pixel_stats.frame_board_stats.end()) {
      continue;
    }
    const double reference_rmse = reference_it->second.Rmse();
    const double candidate_rmse = entry.second.Rmse();
    if (!std::isfinite(reference_rmse) || !std::isfinite(candidate_rmse)) {
      continue;
    }
    candidate_rmses.push_back(candidate_rmse);
    regressions.push_back(candidate_rmse - reference_rmse);
  }
  if (candidate_rmses.empty()) {
    return result;
  }

  constexpr double kMadToRobustSigma = 1.4826;
  const double candidate_median =
      ResidualStats::Percentile(candidate_rmses, 0.5);
  const double candidate_sigma = kMadToRobustSigma *
      MedianAbsoluteDeviation(candidate_rmses, candidate_median);
  const double regression_median =
      ResidualStats::Percentile(regressions, 0.5);
  const double regression_sigma = kMadToRobustSigma *
      MedianAbsoluteDeviation(regressions, regression_median);
  const double mad_scale =
      std::max(0.0, options.full_training_instability_quarantine_mad_scale);
  result.candidate_rmse_threshold_px =
      candidate_median + mad_scale * candidate_sigma;
  result.regression_threshold_px = std::max(
      options.full_training_instability_quarantine_min_regression_px,
      regression_median + mad_scale * regression_sigma);

  std::set<FrameBoardKey> directly_unstable;
  for (const auto& entry : candidate_stats.pixel_stats.frame_board_stats) {
    const FrameBoardKey& key = entry.first;
    const auto reference_it =
        reference_stats.pixel_stats.frame_board_stats.find(key);
    if (reference_it == reference_stats.pixel_stats.frame_board_stats.end()) {
      continue;
    }
    const double reference_rmse = reference_it->second.Rmse();
    const double candidate_rmse = entry.second.Rmse();
    const double regression = candidate_rmse - reference_rmse;
    const double ratio = candidate_rmse /
        std::max(0.25, std::max(0.0, reference_rmse));
    const bool lost_pose =
        reference_stats.fitted_poses.count(key) > 0 &&
        candidate_stats.fitted_poses.count(key) == 0;
    const bool residual_jump =
        candidate_rmse > result.candidate_rmse_threshold_px &&
        regression > result.regression_threshold_px &&
        ratio >= options
                     .full_training_instability_quarantine_min_regression_ratio;
    if (lost_pose || residual_jump) {
      directly_unstable.insert(key);
    }
  }

  result.keys = directly_unstable;
  for (const FrameBoardKey& unstable_key : directly_unstable) {
    if (unstable_key.second != reference_board_id) {
      continue;
    }
    for (const auto& entry : reference_stats.pixel_stats.frame_board_stats) {
      if (entry.first.first == unstable_key.first) {
        result.keys.insert(entry.first);
      }
    }
  }

  const int comparable_count = static_cast<int>(candidate_rmses.size());
  result.max_quarantine_count = std::max(
      1, static_cast<int>(std::ceil(
             options.full_training_instability_quarantine_max_fraction *
             static_cast<double>(comparable_count))));
  result.within_budget =
      static_cast<int>(result.keys.size()) <= result.max_quarantine_count;
  if (!result.within_budget) {
    result.keys.clear();
  }

  std::ostringstream reason;
  reason << "adaptive_frame_board_instability_quarantine"
         << " candidate_median=" << candidate_median
         << " candidate_sigma=" << candidate_sigma
         << " candidate_threshold=" << result.candidate_rmse_threshold_px
         << " regression_median=" << regression_median
         << " regression_sigma=" << regression_sigma
         << " regression_threshold=" << result.regression_threshold_px
         << " direct_count=" << directly_unstable.size()
         << " expanded_count="
         << (result.within_budget ? result.keys.size()
                                  : directly_unstable.size())
         << " max_count=" << result.max_quarantine_count
         << " within_budget=" << (result.within_budget ? 1 : 0)
         << " keys=";
  bool first = true;
  const std::set<FrameBoardKey>& reported_keys =
      result.within_budget ? result.keys : directly_unstable;
  for (const FrameBoardKey& key : reported_keys) {
    if (!first) {
      reason << ";";
    }
    first = false;
    reason << key.first << "/B" << key.second;
  }
  result.reason = reason.str();
  return result;
}

bool ConvertPoseToCv(const Eigen::Isometry3d& pose,
                     cv::Mat* rvec,
                     cv::Mat* tvec) {
  if (rvec == nullptr || tvec == nullptr || !pose.matrix().allFinite()) {
    return false;
  }
  cv::Mat rotation(3, 3, CV_64F);
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      rotation.at<double>(row, column) = pose.linear()(row, column);
    }
  }
  cv::Rodrigues(rotation, *rvec);
  *tvec = (cv::Mat_<double>(3, 1)
      << pose.translation().x(), pose.translation().y(),
         pose.translation().z());
  return !rvec->empty() && !tvec->empty();
}

bool ConvertCvToPose(const cv::Mat& rvec,
                     const cv::Mat& tvec,
                     Eigen::Isometry3d* pose) {
  if (pose == nullptr || rvec.empty() || tvec.empty()) {
    return false;
  }
  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);
  cv::Mat rotation;
  cv::Mat translation;
  rotation_cv.convertTo(rotation, CV_64F);
  tvec.convertTo(translation, CV_64F);
  if (rotation.rows != 3 || rotation.cols != 3 || translation.total() < 3u) {
    return false;
  }
  Eigen::Isometry3d converted = Eigen::Isometry3d::Identity();
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      converted.linear()(row, column) = rotation.at<double>(row, column);
    }
    converted.translation()[row] = translation.at<double>(row, 0);
  }
  if (!converted.matrix().allFinite()) {
    return false;
  }
  *pose = converted;
  return true;
}

bool EstimatePoseFromObjectPointsWithSeed(
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const Eigen::Isometry3d* initial_pose,
    Eigen::Isometry3d* pose,
    double* rmse) {
  if (pose == nullptr || rmse == nullptr ||
      object_points.size() != image_points.size() || object_points.size() < 4u) {
    return false;
  }

  const DoubleSphereCameraModel camera = DoubleSphereCameraModel::FromConfig(
      MakePersistentIntermediateCameraConfig(intrinsics));
  if (!camera.IsValid()) {
    return false;
  }
  std::vector<cv::Point3f> object_points_cv;
  object_points_cv.reserve(object_points.size());
  for (const Eigen::Vector3d& point : object_points) {
    object_points_cv.emplace_back(static_cast<float>(point.x()),
                                  static_cast<float>(point.y()),
                                  static_cast<float>(point.z()));
  }
  cv::Mat initial_rvec;
  cv::Mat initial_tvec;
  const cv::Mat* initial_rvec_ptr = nullptr;
  const cv::Mat* initial_tvec_ptr = nullptr;
  if (initial_pose != nullptr) {
    if (!ConvertPoseToCv(*initial_pose, &initial_rvec, &initial_tvec)) {
      return false;
    }
    initial_rvec_ptr = &initial_rvec;
    initial_tvec_ptr = &initial_tvec;
  }
  cv::Mat fitted_rvec;
  cv::Mat fitted_tvec;
  if (!camera.estimateTransformation(object_points_cv, image_points,
                                     &fitted_rvec, &fitted_tvec,
                                     initial_rvec_ptr, initial_tvec_ptr) ||
      !ConvertCvToPose(fitted_rvec, fitted_tvec, pose)) {
    return false;
  }
  double squared_error = 0.0;
  for (std::size_t index = 0; index < object_points.size(); ++index) {
    const Eigen::Vector3d point_camera = (*pose) * object_points[index];
    if (!point_camera.allFinite() || point_camera.z() <= 0.0) {
      return false;
    }
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    if (!camera.vsEuclideanToKeypoint(point_camera, &predicted) ||
        !predicted.allFinite()) {
      return false;
    }
    const Eigen::Vector2d observed(image_points[index].x, image_points[index].y);
    squared_error += (predicted - observed).squaredNorm();
  }
  *rmse = std::sqrt(squared_error / static_cast<double>(object_points.size()));
  return std::isfinite(*rmse);
}

std::string FormatWorstFullTrainingFrameBoards(
    const FullTrainingPoseRefitStats& candidate_stats,
    const FullTrainingPoseRefitStats& reference_stats,
    int max_entries) {
  struct RankedKey {
    FrameBoardKey key;
    double candidate_rmse = 0.0;
  };
  std::vector<RankedKey> ranked;
  ranked.reserve(candidate_stats.pixel_stats.frame_board_stats.size());
  int above_10_px = 0;
  int above_25_px = 0;
  int above_100_px = 0;
  for (const auto& entry : candidate_stats.pixel_stats.frame_board_stats) {
    const double rmse = entry.second.Rmse();
    ranked.push_back(RankedKey{entry.first, rmse});
    above_10_px += rmse > 10.0 ? 1 : 0;
    above_25_px += rmse > 25.0 ? 1 : 0;
    above_100_px += rmse > 100.0 ? 1 : 0;
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const RankedKey& lhs, const RankedKey& rhs) {
              return lhs.candidate_rmse > rhs.candidate_rmse;
            });

  std::ostringstream stream;
  stream << " candidate_frame_board_tail_count_gt10=" << above_10_px
         << " gt25=" << above_25_px
         << " gt100=" << above_100_px
         << " candidate_worst_frame_boards=";
  const int count = std::min(max_entries, static_cast<int>(ranked.size()));
  for (int index = 0; index < count; ++index) {
    if (index > 0) {
      stream << ";";
    }
    const RankedKey& ranked_key = ranked[static_cast<std::size_t>(index)];
    double reference_rmse = 0.0;
    const auto reference_it =
        reference_stats.pixel_stats.frame_board_stats.find(ranked_key.key);
    if (reference_it != reference_stats.pixel_stats.frame_board_stats.end()) {
      reference_rmse = reference_it->second.Rmse();
    }
    stream << ranked_key.key.first << "/B" << ranked_key.key.second
           << ":" << reference_rmse << "->" << ranked_key.candidate_rmse;
  }
  return stream.str();
}

std::string FormatPoseRefitFailureDelta(
    const FullTrainingPoseRefitStats& candidate_stats,
    const FullTrainingPoseRefitStats& reference_stats,
    int max_entries) {
  std::vector<FrameBoardKey> newly_failed;
  std::vector<FrameBoardKey> recovered;
  for (const auto& entry : candidate_stats.pose_failure_reasons) {
    if (reference_stats.pose_failure_reasons.count(entry.first) == 0) {
      newly_failed.push_back(entry.first);
    }
  }
  for (const auto& entry : reference_stats.pose_failure_reasons) {
    if (candidate_stats.pose_failure_reasons.count(entry.first) == 0) {
      recovered.push_back(entry.first);
    }
  }

  const auto append_key = [](std::ostringstream* stream,
                             const FrameBoardKey& key,
                             const FullTrainingPoseRefitStats& stats) {
    const auto reason_it = stats.pose_failure_reasons.find(key);
    const auto points_it = stats.point_count_by_key.find(key);
    const auto outer_it = stats.outer_point_count_by_key.find(key);
    const auto internal_it = stats.internal_point_count_by_key.find(key);
    const auto invalid_it = stats.invalid_projection_count_by_key.find(key);
    *stream << key.first << "/B" << key.second
            << "(" << (reason_it == stats.pose_failure_reasons.end()
                              ? "recovered"
                              : reason_it->second)
            << ",points="
            << (points_it == stats.point_count_by_key.end() ? 0
                                                             : points_it->second)
            << ",outer="
            << (outer_it == stats.outer_point_count_by_key.end() ? 0
                                                                  : outer_it->second)
            << ",internal="
            << (internal_it == stats.internal_point_count_by_key.end()
                    ? 0
                    : internal_it->second)
            << ",invalid="
            << (invalid_it == stats.invalid_projection_count_by_key.end()
                    ? 0
                    : invalid_it->second)
            << ")";
  };

  std::ostringstream stream;
  stream << " pose_refit_new_failures=" << newly_failed.size()
         << " pose_refit_recovered=" << recovered.size();
  const int new_count =
      std::min(max_entries, static_cast<int>(newly_failed.size()));
  if (new_count > 0) {
    stream << " new_failure_keys=";
    for (int index = 0; index < new_count; ++index) {
      if (index > 0) {
        stream << ";";
      }
      append_key(&stream, newly_failed[static_cast<std::size_t>(index)],
                 candidate_stats);
    }
  }
  const int recovered_count =
      std::min(max_entries, static_cast<int>(recovered.size()));
  if (recovered_count > 0) {
    stream << " recovered_keys=";
    for (int index = 0; index < recovered_count; ++index) {
      if (index > 0) {
        stream << ";";
      }
      append_key(&stream, recovered[static_cast<std::size_t>(index)],
                 reference_stats);
    }
  }
  return stream.str();
}

double RegressionLimit(double before_rmse,
                       double ratio,
                       double absolute_margin_px) {
  if (!std::isfinite(before_rmse) || before_rmse <= 0.0) {
    return absolute_margin_px;
  }
  return std::max(before_rmse * ratio, before_rmse + absolute_margin_px);
}

bool CheckFullTrainingPoseRefitHealthGate(
    const FullTrainingPoseRefitStats& initial_stats,
    const FullTrainingPoseRefitStats& committed_before_stats,
    const FullTrainingPoseRefitStats& candidate_stats,
    const Stage5IncrementalBackendEstimatorOptions& options,
    std::string* reason,
    bool include_initial_reference = true) {
  if (!options.use_full_training_pose_refit_health_gate) {
    if (reason != nullptr) {
      reason->clear();
    }
    return true;
  }
  if (!initial_stats.IsUsable() || !committed_before_stats.IsUsable() ||
      !candidate_stats.IsUsable() ||
      initial_stats.point_total_count != candidate_stats.point_total_count ||
      committed_before_stats.point_total_count !=
          candidate_stats.point_total_count ||
      initial_stats.pose_total_count != candidate_stats.pose_total_count ||
      committed_before_stats.pose_total_count !=
          candidate_stats.pose_total_count) {
    if (reason != nullptr) {
      std::ostringstream stream;
      stream << "full_training_pose_refit_health_gate invalid_evaluation"
             << " initial_usable=" << (initial_stats.IsUsable() ? 1 : 0)
             << " before_usable="
             << (committed_before_stats.IsUsable() ? 1 : 0)
             << " candidate_usable="
             << (candidate_stats.IsUsable() ? 1 : 0)
             << " initial_points=" << initial_stats.point_total_count
             << " before_points=" << committed_before_stats.point_total_count
             << " candidate_points=" << candidate_stats.point_total_count
             << " initial_poses=" << initial_stats.pose_total_count
             << " before_poses=" << committed_before_stats.pose_total_count
             << " candidate_poses=" << candidate_stats.pose_total_count;
      *reason = stream.str();
    }
    return false;
  }

  const bool outer_only = options.full_training_pose_refit_outer_only_health;
  const auto scoped_rmse = [outer_only](const FullTrainingPoseRefitStats& stats) {
    return outer_only ? stats.pixel_stats.OuterRmse() : stats.pixel_stats.Rmse();
  };
  const auto scoped_p95 = [outer_only](const FullTrainingPoseRefitStats& stats) {
    return outer_only ? stats.pixel_stats.OuterP95() : stats.pixel_stats.P95();
  };
  const auto scoped_invalid = [outer_only](const FullTrainingPoseRefitStats& stats) {
    return outer_only ? stats.pixel_stats.invalid_outer_projection_count
                      : stats.pixel_stats.invalid_projection_count;
  };
  const double before_rmse_limit = RegressionLimit(
      scoped_rmse(committed_before_stats),
      options.full_training_pose_refit_max_rmse_regression_ratio,
      options.full_training_pose_refit_max_rmse_regression_abs_px);
  const double initial_rmse_limit = RegressionLimit(
      scoped_rmse(initial_stats),
      options.full_training_pose_refit_max_rmse_regression_ratio,
      options.full_training_pose_refit_max_rmse_regression_abs_px);
  const double before_p95_limit = RegressionLimit(
      scoped_p95(committed_before_stats),
      options.full_training_pose_refit_max_p95_regression_ratio,
      options.full_training_pose_refit_max_p95_regression_abs_px);
  const double initial_p95_limit = RegressionLimit(
      scoped_p95(initial_stats),
      options.full_training_pose_refit_max_p95_regression_ratio,
      options.full_training_pose_refit_max_p95_regression_abs_px);
  const double rmse_limit = include_initial_reference
                                ? std::min(before_rmse_limit,
                                           initial_rmse_limit)
                                : before_rmse_limit;
  const double p95_limit = include_initial_reference
                               ? std::min(before_p95_limit,
                                          initial_p95_limit)
                               : before_p95_limit;
  const double pose_rate_floor = std::max(
      0.0,
      (include_initial_reference
           ? std::min(initial_stats.PoseSuccessRate(),
                      committed_before_stats.PoseSuccessRate())
           : committed_before_stats.PoseSuccessRate()) -
          options.full_training_pose_refit_max_pose_success_rate_drop);
  const bool pass =
      scoped_rmse(candidate_stats) <= rmse_limit &&
      scoped_p95(candidate_stats) <= p95_limit &&
      candidate_stats.PoseSuccessRate() >= pose_rate_floor &&
      scoped_invalid(candidate_stats) <=
          (include_initial_reference
               ? std::max(scoped_invalid(initial_stats),
                          scoped_invalid(committed_before_stats))
               : scoped_invalid(committed_before_stats));
  if (!pass && reason != nullptr) {
    std::ostringstream stream;
    stream << "full_training_pose_refit_health_gate_"
           << (outer_only ? "outer4" : "all_points")
           << " initial_rmse=" << scoped_rmse(initial_stats)
           << " before_rmse=" << scoped_rmse(committed_before_stats)
           << " candidate_rmse=" << scoped_rmse(candidate_stats)
           << " rmse_limit=" << rmse_limit
           << " include_initial_reference="
           << (include_initial_reference ? 1 : 0)
           << " initial_p95=" << scoped_p95(initial_stats)
           << " before_p95=" << scoped_p95(committed_before_stats)
           << " candidate_p95=" << scoped_p95(candidate_stats)
           << " p95_limit=" << p95_limit
           << " initial_pose_rate=" << initial_stats.PoseSuccessRate()
           << " before_pose_rate="
           << committed_before_stats.PoseSuccessRate()
           << " candidate_pose_rate=" << candidate_stats.PoseSuccessRate()
           << " pose_rate_floor=" << pose_rate_floor
           << " initial_invalid=" << scoped_invalid(initial_stats)
           << " before_invalid=" << scoped_invalid(committed_before_stats)
           << " candidate_invalid=" << scoped_invalid(candidate_stats)
           << FormatPoseRefitFailureDelta(
                  candidate_stats, committed_before_stats, 16)
           << FormatWorstFullTrainingFrameBoards(
                  candidate_stats, committed_before_stats, 8);
    *reason = stream.str();
  } else if (reason != nullptr) {
    reason->clear();
  }
  return pass;
}

double AngularHealthThresholdRad(
    const Stage5IncrementalBackendEstimatorOptions& options) {
  return std::max(
      0.02,
      2.0 * std::max(PositiveOrDefault(options.outer_huber_delta_radians,
                                       0.02),
                     PositiveOrDefault(options.internal_huber_delta_radians,
                                       0.015)));
}

double AdaptiveAngularHealthThreshold(
    const Stage5IncrementalBackendEstimatorOptions& options,
    const ResidualStats& seed_stats,
    const ResidualStats& candidate_stats) {
  const double floor = AngularHealthThresholdRad(options);
  double robust_scale = floor;
  auto absorb = [&](double value, double scale) {
    if (std::isfinite(value) && value > 0.0) {
      robust_scale = std::max(robust_scale, scale * value);
    }
  };
  absorb(seed_stats.Rmse(), 2.5);
  absorb(seed_stats.P95(), 1.5);
  absorb(seed_stats.OuterRmse(), 2.5);
  absorb(seed_stats.OuterP95(), 1.5);
  absorb(seed_stats.InternalRmse(), 2.0);
  absorb(seed_stats.InternalP95(), 1.25);
  absorb(candidate_stats.Rmse(), 1.35);
  absorb(candidate_stats.P95(), 0.85);
  absorb(candidate_stats.OuterRmse(), 1.35);
  absorb(candidate_stats.OuterP95(), 0.85);
  absorb(candidate_stats.InternalRmse(), 1.2);
  absorb(candidate_stats.InternalP95(), 0.75);
  return std::max(floor, robust_scale);
}

double AdaptivePixelEquivalentHealthThreshold(const ResidualStats& seed_stats,
                                              const ResidualStats& candidate_stats) {
  double robust_scale = 2.5;
  auto absorb = [&](double value, double scale) {
    if (std::isfinite(value) && value > 0.0) {
      robust_scale = std::max(robust_scale, scale * value);
    }
  };
  absorb(seed_stats.Rmse(), 2.5);
  absorb(seed_stats.P95(), 1.5);
  absorb(seed_stats.OuterRmse(), 2.5);
  absorb(seed_stats.OuterP95(), 1.5);
  absorb(seed_stats.InternalRmse(), 2.0);
  absorb(seed_stats.InternalP95(), 1.25);
  absorb(candidate_stats.Rmse(), 1.35);
  absorb(candidate_stats.P95(), 0.85);
  absorb(candidate_stats.OuterRmse(), 1.35);
  absorb(candidate_stats.OuterP95(), 0.85);
  absorb(candidate_stats.InternalRmse(), 1.2);
  absorb(candidate_stats.InternalP95(), 0.75);
  return robust_scale;
}

double SelectionHealthThreshold(
    const Stage5IncrementalBackendEstimatorOptions& options,
    const Stage5IncrementalBackendBatchInput& input,
    SelectionResidualMetric metric) {
  if (options.single_board_dense_grid_profile) {
    return input.residual_health_threshold_px > 0.0
               ? input.residual_health_threshold_px
               : std::max(0.25, options.huber_delta_pixels);
  }
  if (metric == SelectionResidualMetric::kPixel) {
    return input.residual_health_threshold_px > 0.0
               ? input.residual_health_threshold_px
               : std::max(5.0, 1.25 * std::max(0.0, input.max_trial_rmse));
  }
	  if (SelectionMetricUsesAngularHealth(metric)) {
	    return input.residual_health_threshold_metric > 0.0
	               ? input.residual_health_threshold_metric
	               : AngularHealthThresholdRad(options);
	  }
  if (SelectionMetricIsHybridObjective(metric)) {
    return input.residual_health_threshold_metric > 0.0
               ? input.residual_health_threshold_metric
               : 2.5;
  }
  return 0.0;
}

double SelectionRegressionMargin(
    const Stage5IncrementalBackendEstimatorOptions& options,
    SelectionResidualMetric metric,
    bool internal) {
	  if (SelectionMetricUsesAngularHealth(metric)) {
	    const double base = AngularHealthThresholdRad(options);
	    return internal ? 0.35 * base : 0.25 * base;
	  }
  return internal ? options.split_residual_max_internal_rmse_regression_abs_px
                  : options.split_residual_max_outer_rmse_regression_abs_px;
}

bool CheckSplitResidualHealthGate(
    const ResidualStats& before_stats,
    const ResidualStats& after_stats,
    const ResidualStats& candidate_stats,
    const Stage5IncrementalBackendEstimatorOptions& options,
    const Stage5IncrementalBackendBatchInput& input,
    SelectionResidualMetric metric,
    std::string* reason) {
  if (!options.use_split_residual_health_gate || input.force) {
    if (reason != nullptr) {
      reason->clear();
    }
    return true;
  }
  const bool outer_only = options.split_residual_outer_only_health;
  const int candidate_health_count =
      outer_only ? candidate_stats.outer_count : candidate_stats.total_count;
  const int after_health_count =
      outer_only ? after_stats.outer_count : after_stats.total_count;
  if (candidate_health_count <= 0 || after_health_count <= 0) {
    if (reason != nullptr) {
      *reason = "split_residual_health_gate empty_residual_stats";
    }
    return false;
  }

  const double base_threshold =
      SelectionHealthThreshold(options, input, metric);
	  const double outer_p95_floor =
	      SelectionMetricUsesAngularHealth(metric) ? 0.03 : 2.5;
	  const double internal_p95_floor =
	      SelectionMetricUsesAngularHealth(metric) ? 0.035 : 3.5;
	  const double outer_p95_limit =
	      SelectionMetricIsHybridObjective(metric)
	          ? std::numeric_limits<double>::infinity()
	          : std::max(outer_p95_floor,
	                     options.split_residual_p95_threshold_scale *
	                         base_threshold);
	  const double internal_p95_limit =
	      SelectionMetricIsHybridObjective(metric)
	          ? std::numeric_limits<double>::infinity()
	          : std::max(internal_p95_floor,
	                     options.split_residual_p95_threshold_scale *
                         base_threshold);
  if (candidate_stats.outer_count > 0 &&
      candidate_stats.OuterP95() > outer_p95_limit) {
    if (reason != nullptr) {
      std::ostringstream stream;
      stream << "split_residual_health_gate candidate_outer_p95="
             << candidate_stats.OuterP95()
             << " limit=" << outer_p95_limit;
      *reason = stream.str();
    }
    return false;
  }
  if (!outer_only && candidate_stats.internal_count > 0 &&
      candidate_stats.InternalP95() > internal_p95_limit) {
    if (reason != nullptr) {
      std::ostringstream stream;
      stream << "split_residual_health_gate candidate_internal_p95="
             << candidate_stats.InternalP95()
             << " limit=" << internal_p95_limit;
      *reason = stream.str();
    }
    return false;
  }

  if (options.use_candidate_relative_residual_gate) {
    if (!outer_only && before_stats.total_count > 0 &&
        candidate_stats.Rmse() >
            RegressionLimit(before_stats.Rmse(),
                            options.split_residual_max_rmse_regression_ratio,
                            SelectionRegressionMargin(options, metric, true))) {
      if (reason != nullptr) {
        std::ostringstream stream;
        stream << "split_residual_health_gate candidate_total_rmse_regression "
               << "before=" << before_stats.Rmse()
               << " candidate=" << candidate_stats.Rmse();
        *reason = stream.str();
      }
      return false;
    }
    if (before_stats.outer_count > 0 && candidate_stats.outer_count > 0 &&
        candidate_stats.OuterRmse() >
            RegressionLimit(before_stats.OuterRmse(),
                            options.split_residual_max_rmse_regression_ratio,
                            SelectionRegressionMargin(options, metric, false))) {
      if (reason != nullptr) {
        std::ostringstream stream;
        stream << "split_residual_health_gate candidate_outer_rmse_regression "
               << "before=" << before_stats.OuterRmse()
               << " candidate=" << candidate_stats.OuterRmse();
        *reason = stream.str();
      }
      return false;
    }
    if (!outer_only && before_stats.internal_count > 0 &&
        candidate_stats.internal_count > 0 &&
        candidate_stats.InternalRmse() >
            RegressionLimit(before_stats.InternalRmse(),
                            options.split_residual_max_rmse_regression_ratio,
                            SelectionRegressionMargin(options, metric, true))) {
      if (reason != nullptr) {
        std::ostringstream stream;
        stream << "split_residual_health_gate candidate_internal_rmse_regression "
               << "before=" << before_stats.InternalRmse()
               << " candidate=" << candidate_stats.InternalRmse();
        *reason = stream.str();
      }
      return false;
    }
  }

  if (!outer_only && before_stats.total_count > 0 &&
      after_stats.Rmse() >
          RegressionLimit(before_stats.Rmse(),
                          options.split_residual_max_rmse_regression_ratio,
                          SelectionRegressionMargin(options, metric, true))) {
    if (reason != nullptr) {
      std::ostringstream stream;
      stream << "split_residual_health_gate committed_total_rmse_regression "
             << "before=" << before_stats.Rmse()
             << " after=" << after_stats.Rmse();
      *reason = stream.str();
    }
    return false;
  }
  if (before_stats.outer_count > 0 && after_stats.outer_count > 0 &&
      after_stats.OuterRmse() >
          RegressionLimit(before_stats.OuterRmse(),
                          options.split_residual_max_rmse_regression_ratio,
                          SelectionRegressionMargin(options, metric, false))) {
    if (reason != nullptr) {
      std::ostringstream stream;
      stream << "split_residual_health_gate committed_outer_rmse_regression "
             << "before=" << before_stats.OuterRmse()
             << " after=" << after_stats.OuterRmse();
      *reason = stream.str();
    }
    return false;
  }
  if (!outer_only && before_stats.internal_count > 0 &&
      after_stats.internal_count > 0 &&
      after_stats.InternalRmse() >
          RegressionLimit(before_stats.InternalRmse(),
                          options.split_residual_max_rmse_regression_ratio,
                          SelectionRegressionMargin(options, metric, true))) {
    if (reason != nullptr) {
      std::ostringstream stream;
      stream << "split_residual_health_gate committed_internal_rmse_regression "
             << "before=" << before_stats.InternalRmse()
             << " after=" << after_stats.InternalRmse();
      *reason = stream.str();
    }
    return false;
  }

  if (!outer_only && candidate_stats.outer_count > 0 &&
      candidate_stats.internal_count > 0) {
    const double outer_rmse = std::max(1e-9, candidate_stats.OuterRmse());
    const double internal_limit =
	        std::max(SelectionMetricUsesAngularHealth(metric) ? 0.03 : 3.0,
	                 options.split_residual_internal_outer_rmse_ratio *
	                         outer_rmse +
	                     (SelectionMetricUsesAngularHealth(metric)
	                          ? 0.25 * AngularHealthThresholdRad(options)
	                          : 0.5));
    if (candidate_stats.InternalRmse() > internal_limit) {
      if (reason != nullptr) {
        std::ostringstream stream;
        stream << "split_residual_health_gate internal_outer_mismatch "
               << "candidate_outer_rmse=" << candidate_stats.OuterRmse()
               << " candidate_internal_rmse="
               << candidate_stats.InternalRmse()
               << " limit=" << internal_limit;
        *reason = stream.str();
      }
      return false;
    }
  }

  if (reason != nullptr) {
    reason->clear();
  }
  return true;
}

struct ObservationBudget {
  int outer_count = 0;
  int internal_count = 0;
};

std::map<FrameBoardKey, ObservationBudget> ComputeObservationBudgets(
    const CalibrationMeasurementDataset& dataset) {
  std::map<FrameBoardKey, ObservationBudget> budgets;
  for (const JointPointObservation& observation : dataset.solver_observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    ObservationBudget& budget =
        budgets[FrameBoardKey(observation.frame_index,
                              observation.board_id)];
    if (observation.point_type == JointPointType::Outer) {
      ++budget.outer_count;
    } else {
      ++budget.internal_count;
    }
  }
  return budgets;
}

double ComputePersistentBalanceWeight(const ObservationBudget& budget,
                                      JointPointType point_type,
                                      bool uniform_control_point_mode,
                                      bool uniform_control_point_weighting,
                                      double internal_role_budget_when_mixed) {
  if (uniform_control_point_mode || uniform_control_point_weighting) {
    // Dense calibration targets contribute one independent measurement per
    // control point, matching Kalibr's per-corner unit covariance model.
    return 1.0;
  }
  const bool has_outer = budget.outer_count > 0;
  const bool has_internal = budget.internal_count > 0;
  double type_budget = 1.0;
  int type_count = 1;
  if (has_outer && has_internal) {
    const double internal_budget = std::max(
        0.0, std::min(1.0, internal_role_budget_when_mixed));
    type_budget = point_type == JointPointType::Internal
                      ? internal_budget
                      : 1.0 - internal_budget;
    type_count = point_type == JointPointType::Outer ? budget.outer_count
                                                     : budget.internal_count;
  } else if (point_type == JointPointType::Outer) {
    type_count = budget.outer_count;
  } else {
    type_count = budget.internal_count;
  }
  return type_budget / static_cast<double>(std::max(1, type_count));
}

template <typename GeometryT>
class PersistentProblemBuilder {
 public:
  using CameraDv = aslam::backend::CameraDesignVariable<GeometryT>;

  PersistentProblemBuilder(const CalibrationStateBundle& baseline_bundle,
                           const CalibrationStateBundle& candidate_pool_bundle,
                           const Stage5IncrementalBackendEstimatorOptions& options)
      : baseline_bundle_(baseline_bundle),
        candidate_pool_bundle_(candidate_pool_bundle),
        options_(options),
        observation_budgets_(
            ComputeObservationBudgets(candidate_pool_bundle.measurement_dataset)),
	        camera_geometry_(
	            MakePersistentGeometry<GeometryT>(baseline_bundle.scene_state.camera)),
	        camera_dv_(camera_geometry_),
	        chordal_reference_focal_px_(std::sqrt(
	            std::max(1.0, std::abs(baseline_bundle.scene_state.camera.fu)) *
	            std::max(1.0, std::abs(baseline_bundle.scene_state.camera.fv)))) {
	    camera_dv_.setActive(false, false, false);
	    camera_geometry_->projection().distortion().getParameters(
	        initial_distortion_parameters_);
	    BuildBoardVariables();
	  }

  struct StateSnapshot;

  struct IndependentCameraWarmupResult {
    bool attempted = false;
    bool initialized = false;
    bool state_valid = false;
    int pose_count = 0;
    int point_count = 0;
    aslam::backend::SolutionReturnValue solution;
    std::string failure_reason;
  };
	  struct ResidualConstructionCounts {
	    int image_plane_residual_count = 0;
	    int angular_residual_count = 0;
	    int chordal_residual_count = 0;
	    int hybrid_angular_selected_count = 0;
	    int hybrid_chordal_selected_count = 0;
	    int angular_observation_geometry_failure_count = 0;
	    int angular_local_whitening_success_count = 0;
	    int angular_local_whitening_failure_count = 0;
	    int angular_local_whitening_clamped_count = 0;
	    double angular_local_whitening_sigma_sum_rad = 0.0;
	    double angular_local_whitening_sigma_min_rad =
	        std::numeric_limits<double>::infinity();
	    double angular_local_whitening_sigma_max_rad = 0.0;
	    double angular_local_whitening_weight_sum = 0.0;
	    double angular_local_whitening_weight_min =
	        std::numeric_limits<double>::infinity();
	    double angular_local_whitening_weight_max = 0.0;
	  };

  boost::shared_ptr<CalibrationBatch> BuildBatch(
      const std::set<FrameBoardKey>& keys,
      bool force_add_frame_variables,
      CameraOptimizationPhase camera_phase,
      const StateSnapshot* camera_anchor_state,
      ResidualConstructionCounts* residual_counts = nullptr,
      double camera_anchor_weight_scale = 1.0,
      const KbDistortionReleaseState& kb_release_state =
          KbDistortionReleaseState(),
      bool outer_only = false,
      bool force_board_layout_active = false) {
    boost::shared_ptr<CalibrationBatch> batch =
        boost::make_shared<CalibrationBatch>();
    AddCameraVariables(camera_phase, batch);
    MaybeAddSquarePixelFocalPrior(batch);
    MaybeAddCameraAnchorPrior(camera_phase, camera_anchor_state, batch,
                              camera_anchor_weight_scale);
    MaybeAddDsSeedStabilityPrior(camera_phase, camera_anchor_state, batch);
    MaybeAddKbDistortionPrior(camera_phase, camera_anchor_state,
                              kb_release_state, batch,
                              camera_anchor_weight_scale);
    // A model-aware active-camera seed is a full-dataset camera bootstrap,
    // not a joint layout refinement.  Keep the rigid board layout fixed at
    // this stage so a small seed cannot compensate a camera step by moving
    // the shared board geometry.
    const bool board_layout_active =
        (!options_.fix_board_layout || force_board_layout_active) &&
        camera_phase != CameraOptimizationPhase::kSeedActiveCamera &&
        camera_phase != CameraOptimizationPhase::kSeedActiveCameraFixedFramePoses;
    SetBoardLayoutActive(board_layout_active);
    AddBoardVariables(batch);
    AddFrameVariables(keys, force_add_frame_variables, batch);
    AddResiduals(keys, batch, residual_counts, outer_only);
    return batch;
  }

  ResidualStats EvaluateAccepted(
      const std::set<FrameBoardKey>& accepted_keys,
      SelectionResidualMetric metric) const {
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
	      const double pixel_residual_norm =
	          (predicted - observation.image_xy).norm();
	      double residual_norm = pixel_residual_norm;
	      if (metric != SelectionResidualMetric::kPixel) {
	        AngularObservationGeometry observation_geometry;
	        const bool have_observation_geometry = ComputeObservationGeometryForCamera(
	            *camera_geometry_, observation.image_xy,
	            &observation_geometry);
	        const bool use_tangent_angular =
	            metric == SelectionResidualMetric::kAngularTangent ||
	            metric ==
	                SelectionResidualMetric::kLocallyWhitenedAngularTangent ||
	            (metric == SelectionResidualMetric::kHybridObjective &&
	             options_.residual_model == ResidualModel::HybridEdgeAngular &&
	             have_observation_geometry &&
	             ShouldUseAngularResidual(options_.residual_model,
	                                      observation_geometry.polar_angle_deg,
	                                      options_.hybrid_angular_threshold_deg));
	        const bool use_continuous_hybrid =
	            metric == SelectionResidualMetric::kHybridObjective &&
	            options_.residual_model == ResidualModel::PolarContinuousHybrid;
	        const bool use_chordal =
	            metric == SelectionResidualMetric::kChordal ||
	            metric == SelectionResidualMetric::kPixelChordalHybridObjective;
	        if (use_tangent_angular || use_chordal || use_continuous_hybrid) {
	          AngularPredictionGeometry prediction_geometry;
	          if (!have_observation_geometry ||
	              !ComputePredictionGeometryForCamera(
	                  *camera_geometry_, point_camera,
	                  &prediction_geometry)) {
	            ++stats.invalid_projection_count;
	            continue;
	          }
	          if (use_tangent_angular) {
	            const Eigen::Vector2d angular_residual =
	                ComputeAngularResidualTangent(observation_geometry,
	                                              prediction_geometry);
	            if (metric == SelectionResidualMetric::
	                              kLocallyWhitenedAngularTangent) {
	              BearingCovarianceResult covariance_result;
	              if (!ComputeLocalBearingWhiteningForCamera(
	                      *camera_geometry_, observation.image_xy,
	                      observation_geometry, options_,
	                      &covariance_result)) {
	                ++stats.invalid_projection_count;
	                continue;
	              }
	              residual_norm =
	                  (covariance_result.sqrt_information * angular_residual)
	                      .norm();
	            } else {
	              residual_norm = ComputeAngularResidualNorm(angular_residual);
	            }
	          } else if (use_continuous_hybrid) {
	            const double angular_weight = ComputePolarContinuousAngularWeight(
	                observation_geometry.polar_angle_deg,
	                options_.polar_continuous_hybrid_threshold_deg,
	                options_.polar_continuous_hybrid_temperature_deg);
	            const double angular_norm = ComputeAngularResidualNorm(
	                ComputeAngularResidualTangent(observation_geometry,
	                                              prediction_geometry));
	            const double pixel_weight = std::max(0.0, 1.0 - angular_weight);
	            residual_norm = std::sqrt(
	                pixel_weight * pixel_residual_norm * pixel_residual_norm +
	                std::max(0.0, angular_weight) *
	                    chordal_reference_focal_px_ *
	                    chordal_reference_focal_px_ *
	                    angular_norm * angular_norm);
	          } else if (metric == SelectionResidualMetric::kChordal) {
	            residual_norm =
	                (prediction_geometry.predicted_ray -
	                 observation_geometry.observed_ray).norm();
	          } else {
	            const double chordal_norm =
	                (prediction_geometry.predicted_ray -
	                 observation_geometry.observed_ray).norm();
	            const double pixel_weight =
	                std::max(0.0, options_.pixel_residual_weight);
	            const double chordal_weight =
	                std::max(0.0, options_.chordal_residual_weight);
	            residual_norm = std::sqrt(
	                pixel_weight * pixel_residual_norm * pixel_residual_norm +
	                chordal_weight * chordal_reference_focal_px_ *
	                    chordal_reference_focal_px_ * chordal_norm *
	                    chordal_norm);
	          }
	        }
	      }
      stats.Add(FrameBoardKey(observation.frame_index, observation.board_id),
                observation.point_type, residual_norm);
    }
    return stats;
  }

  FullTrainingPoseRefitStats EvaluateFullTrainingPoseRefitPixel(
      const FullTrainingPoseRefitStats* pose_seed_stats = nullptr,
      const std::set<FrameBoardKey>* excluded_keys = nullptr) const {
    FullTrainingPoseRefitStats result;
    const OuterBootstrapCameraIntrinsics intrinsics = CurrentCamera();
    const DoubleSphereCameraModel camera = DoubleSphereCameraModel::FromConfig(
        MakePersistentIntermediateCameraConfig(intrinsics));
    result.camera_valid = camera.IsValid();

    std::map<FrameBoardKey, std::vector<const JointPointObservation*> >
        observations_by_key;
    for (const JointPointObservation& observation :
         candidate_pool_bundle_.measurement_dataset.solver_observations) {
      const FrameBoardKey key(observation.frame_index, observation.board_id);
      if (!observation.used_in_solver ||
          (excluded_keys != nullptr && excluded_keys->count(key) > 0)) {
        continue;
      }
      observations_by_key[key].push_back(&observation);
    }

    const double invalid_penalty =
        std::max(1.0, options_.invalid_projection_penalty_pixels);
    for (const auto& entry : observations_by_key) {
      const FrameBoardKey& key = entry.first;
      const std::vector<const JointPointObservation*>& observations =
          entry.second;
      ++result.pose_total_count;
      result.point_total_count += static_cast<int>(observations.size());
      result.point_count_by_key[key] = static_cast<int>(observations.size());

      std::vector<Eigen::Vector3d> object_points;
      std::vector<cv::Point2f> image_points;
      object_points.reserve(observations.size());
      image_points.reserve(observations.size());
      bool finite_observations = true;
      for (const JointPointObservation* observation : observations) {
        finite_observations =
            finite_observations && observation != nullptr &&
            observation->target_xyz_board.allFinite() &&
            observation->image_xy.allFinite();
        if (observation == nullptr) {
          continue;
        }
        if (observation->point_type == JointPointType::Outer) {
          ++result.outer_point_count_by_key[key];
        } else {
          ++result.internal_point_count_by_key[key];
        }
        object_points.push_back(observation->target_xyz_board);
        image_points.emplace_back(
            static_cast<float>(observation->image_xy.x()),
            static_cast<float>(observation->image_xy.y()));
      }

      Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
      const Eigen::Isometry3d* initial_pose = nullptr;
      if (pose_seed_stats != nullptr) {
        const auto seed_it = pose_seed_stats->fitted_poses.find(key);
        if (seed_it != pose_seed_stats->fitted_poses.end()) {
          initial_pose = &seed_it->second;
        }
      }
      double pose_rmse = 0.0;
      bool pose_success = false;
      std::string pose_failure_reason;
      if (!result.camera_valid) {
        pose_failure_reason = "camera_invalid";
      } else if (!finite_observations) {
        pose_failure_reason = "nonfinite_observation";
      } else if (object_points.size() < 4u) {
        pose_failure_reason = "insufficient_points";
      } else if (!EstimatePoseFromObjectPointsWithSeed(
                     intrinsics, object_points, image_points, initial_pose,
                     &T_camera_board, &pose_rmse)) {
        // The direct model-aware solver rejects this branch if fitting,
        // cheirality, or a model projection is invalid.  Keep this category
        // explicit instead of attributing it to the input frame.
        pose_failure_reason = "pose_solver_or_projection_validity_failed";
      } else if (!T_camera_board.matrix().allFinite() ||
                 !std::isfinite(pose_rmse)) {
        pose_failure_reason = "nonfinite_pose_or_rmse";
      } else {
        pose_success = true;
      }
      if (pose_success) {
        ++result.pose_success_count;
        result.fitted_poses[key] = T_camera_board;
        result.pose_rmse_by_key[key] = pose_rmse;
      } else {
        result.pose_failure_reasons[key] = pose_failure_reason;
      }

      for (std::size_t index = 0; index < observations.size(); ++index) {
        const JointPointObservation& observation = *observations[index];
        double residual_norm = invalid_penalty;
        bool valid_projection = false;
        if (pose_success) {
          Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
          valid_projection = camera.vsEuclideanToKeypoint(
              T_camera_board * observation.target_xyz_board, &predicted);
          if (valid_projection && predicted.allFinite()) {
            residual_norm = (predicted - observation.image_xy).norm();
            valid_projection = std::isfinite(residual_norm);
          }
        }
        if (!valid_projection) {
          residual_norm = invalid_penalty;
          ++result.pixel_stats.invalid_projection_count;
          ++result.invalid_projection_count_by_key[key];
          if (observation.point_type == JointPointType::Outer) {
            ++result.pixel_stats.invalid_outer_projection_count;
            ++result.invalid_outer_projection_count_by_key[key];
          } else {
            ++result.pixel_stats.invalid_internal_projection_count;
            ++result.invalid_internal_projection_count_by_key[key];
          }
        }
        // Invalid projections are audit/health signals, not valid
        // reprojection observations. Keep their type-specific counters above,
        // but exclude the penalty value from RMSE/P95 and frame-board stats.
        if (valid_projection) {
          result.pixel_stats.Add(key, observation.point_type, residual_norm);
        }
      }
    }
    return result;
  }

  IndependentCameraWarmupResult WarmupIndependentFrameBoardCamera(
      int max_iterations,
      double convergence_delta_j,
      double convergence_delta_x,
      const std::set<FrameBoardKey>* excluded_keys = nullptr,
      const std::set<FrameBoardKey>* included_keys = nullptr) {
    IndependentCameraWarmupResult result;
    result.attempted = true;

    std::map<FrameBoardKey, std::vector<const JointPointObservation*> >
        observations_by_key;
    for (const JointPointObservation& observation :
         candidate_pool_bundle_.measurement_dataset.solver_observations) {
      const FrameBoardKey key(observation.frame_index, observation.board_id);
      if (!observation.used_in_solver ||
          (excluded_keys != nullptr && excluded_keys->count(key) > 0) ||
          (included_keys != nullptr && included_keys->count(key) == 0)) {
        continue;
      }
      observations_by_key[key].push_back(&observation);
    }

    using IndependentPoseMap =
        std::map<FrameBoardKey, std::unique_ptr<PoseVariable> >;
    IndependentPoseMap pose_variables;
    const OuterBootstrapCameraIntrinsics intrinsics = CurrentCamera();
    for (const auto& entry : observations_by_key) {
      if (entry.second.size() < 4u) {
        continue;
      }
      std::vector<Eigen::Vector3d> object_points;
      std::vector<cv::Point2f> image_points;
      object_points.reserve(entry.second.size());
      image_points.reserve(entry.second.size());
      bool finite = true;
      for (const JointPointObservation* observation : entry.second) {
        finite = finite && observation != nullptr &&
                 observation->target_xyz_board.allFinite() &&
                 observation->image_xy.allFinite();
        if (observation != nullptr) {
          object_points.push_back(observation->target_xyz_board);
          image_points.emplace_back(
              static_cast<float>(observation->image_xy.x()),
              static_cast<float>(observation->image_xy.y()));
        }
      }
      Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
      double pose_rmse = 0.0;
      if (!finite ||
          !EstimatePoseFromObjectPoints(intrinsics, object_points, image_points,
                                        &T_camera_board, &pose_rmse) ||
          !T_camera_board.matrix().allFinite() || !std::isfinite(pose_rmse)) {
        continue;
      }
      std::unique_ptr<PoseVariable> variable(new PoseVariable());
      InitializePoseVariable(variable.get(), T_camera_board.matrix(), true);
      pose_variables.emplace(entry.first, std::move(variable));
    }
    result.pose_count = static_cast<int>(pose_variables.size());
    if (pose_variables.empty()) {
      result.failure_reason =
          "no valid frame-board pose could be initialized";
      return result;
    }

    boost::shared_ptr<CalibrationBatch> problem =
        boost::make_shared<CalibrationBatch>();
    constexpr bool kHasDistortionDv =
        GeometryT::projection_t::distortion_t::DesignVariableDimension > 0;
    camera_dv_.setActive(true, kHasDistortionDv, false);
    problem->addDesignVariable(camera_dv_.projectionDesignVariable(),
                               kCameraInformationGroupId);
    problem->addDesignVariable(camera_dv_.distortionDesignVariable(),
                               kCameraInformationGroupId);
    problem->addDesignVariable(camera_dv_.shutterDesignVariable(),
                               kCameraInformationGroupId);
    for (const auto& entry : pose_variables) {
      AddPoseVariableDvs(*entry.second, kTransformationGroupId, problem);
    }

    for (const auto& entry : pose_variables) {
      const auto observations_it = observations_by_key.find(entry.first);
      if (observations_it == observations_by_key.end()) {
        continue;
      }
      const auto budget_it = observation_budgets_.find(entry.first);
      const ObservationBudget budget =
          budget_it == observation_budgets_.end() ? ObservationBudget{}
                                                  : budget_it->second;
      for (const JointPointObservation* observation : observations_it->second) {
        if (observation == nullptr) {
          continue;
        }
        const double weight =
            ComputePersistentBalanceWeight(
                budget, observation->point_type,
                options_.single_board_dense_grid_profile,
                options_.uniform_control_point_weighting,
                options_.internal_role_budget_when_mixed) *
            std::max(0.0, observation->final_observation_weight);
        if (!(weight > 0.0)) {
          continue;
        }
        const aslam::backend::HomogeneousExpression point_board(
            observation->target_xyz_board);
        const aslam::backend::HomogeneousExpression point_camera =
            entry.second->expression * point_board;
        boost::shared_ptr<IncrementalReprojectionError<GeometryT> > error(
            new IncrementalReprojectionError<GeometryT>(
                observation->image_xy,
                weight * Eigen::Matrix2d::Identity(),
                options_.huber_delta_pixels, true, point_camera, camera_dv_,
                options_.invalid_projection_penalty_pixels));
        problem->addErrorTerm(error);
        ++result.point_count;
      }
    }
    if (result.point_count <= 0) {
      result.failure_reason = "independent camera warmup has no residuals";
      return result;
    }
    result.initialized = true;

    aslam::backend::Optimizer2Options optimizer_options;
    optimizer_options.maxIterations = std::max(1, max_iterations);
    optimizer_options.convergenceDeltaJ = convergence_delta_j;
    optimizer_options.convergenceDeltaX = convergence_delta_x;
    optimizer_options.nThreads = 4;
    optimizer_options.verbose = options_.verbose;
    aslam::backend::Optimizer2 optimizer(optimizer_options);
    optimizer.setProblem(problem);
    result.solution = optimizer.optimize();

    result.state_valid = IsCameraStateValid(*camera_geometry_,
                                            &result.failure_reason);
    for (const auto& entry : pose_variables) {
      if (!IsPoseVariableFinite(*entry.second)) {
        result.state_valid = false;
        result.failure_reason =
            "nonfinite independent frame-board pose after warmup";
        break;
      }
    }
    return result;
  }

  std::set<FrameBoardKey> CollectIndependentPoseSupportedKeys(
      const std::set<FrameBoardKey>& keys,
      double max_pose_rmse_px) const {
    std::map<FrameBoardKey, std::vector<const JointPointObservation*> >
        observations_by_key;
    for (const JointPointObservation& observation :
         candidate_pool_bundle_.measurement_dataset.solver_observations) {
      const FrameBoardKey key(observation.frame_index, observation.board_id);
      if (observation.used_in_solver && keys.count(key) > 0) {
        observations_by_key[key].push_back(&observation);
      }
    }
    std::set<FrameBoardKey> supported;
    const OuterBootstrapCameraIntrinsics intrinsics = CurrentCamera();
    for (const auto& entry : observations_by_key) {
      std::vector<Eigen::Vector3d> object_points;
      std::vector<cv::Point2f> image_points;
      bool finite = true;
      for (const JointPointObservation* observation : entry.second) {
        finite = finite && observation != nullptr &&
                 observation->target_xyz_board.allFinite() &&
                 observation->image_xy.allFinite();
        if (observation != nullptr) {
          object_points.push_back(observation->target_xyz_board);
          image_points.emplace_back(
              static_cast<float>(observation->image_xy.x()),
              static_cast<float>(observation->image_xy.y()));
        }
      }
      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      double pose_rmse = 0.0;
      if (finite && object_points.size() >= 4u &&
          EstimatePoseFromObjectPoints(intrinsics, object_points, image_points,
                                       &pose, &pose_rmse) &&
          pose.matrix().allFinite() && std::isfinite(pose_rmse) &&
          pose_rmse <= max_pose_rmse_px) {
        supported.insert(entry.first);
      }
    }
    return supported;
  }

  CalibrationSceneState BuildSceneState() const {
    CalibrationSceneState scene = candidate_pool_bundle_.scene_state;
    scene.camera = CameraToIntrinsics<GeometryT>(*camera_geometry_);
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
    Eigen::MatrixXd distortion_parameters;
    PoseMatrixMap frame_poses;
    PoseMatrixMap board_poses;
  };

  aslam::backend::SolutionReturnValue PrefitCandidateFramePoses(
      const std::set<FrameBoardKey>& keys,
      int max_iterations,
      double convergence_delta_j,
      double convergence_delta_x) {
    for (auto& entry : board_variables_) {
      entry.second.rotation_dv->setActive(false);
      entry.second.translation_dv->setActive(false);
    }
    for (auto& entry : frame_variables_) {
      entry.second.rotation_dv->setActive(false);
      entry.second.translation_dv->setActive(false);
    }

    boost::shared_ptr<CalibrationBatch> problem = BuildBatch(
        keys, true, CameraOptimizationPhase::kPosePrefitFixedIntrinsics,
        nullptr, nullptr);
    // BuildBatch activates a mutable layout for regular candidate solves.
    // Pose prefit is deliberately narrower: it must align only the incoming
    // frame against the committed camera and layout.  Leaving these variables
    // active lets one candidate locally deform the shared board geometry
    // before its joint admission solve starts.
    for (auto& entry : board_variables_) {
      entry.second.rotation_dv->setActive(false);
      entry.second.translation_dv->setActive(false);
    }
    for (const FrameBoardKey& key : keys) {
      auto frame_it = frame_variables_.find(key.first);
      if (frame_it != frame_variables_.end()) {
        frame_it->second.rotation_dv->setActive(true);
        frame_it->second.translation_dv->setActive(true);
      }
    }

    aslam::backend::Optimizer2Options optimizer_options;
    optimizer_options.maxIterations = std::max(1, max_iterations);
    optimizer_options.convergenceDeltaJ = convergence_delta_j;
    optimizer_options.convergenceDeltaX = convergence_delta_x;
    optimizer_options.nThreads = 4;
    optimizer_options.verbose = options_.verbose;
    aslam::backend::Optimizer2 optimizer(optimizer_options);
    optimizer.setProblem(problem);
    const aslam::backend::SolutionReturnValue return_value =
        optimizer.optimize();

    for (auto& entry : board_variables_) {
      entry.second.rotation_dv->setActive(!options_.fix_board_layout);
      entry.second.translation_dv->setActive(!options_.fix_board_layout);
    }
    for (auto& entry : frame_variables_) {
      const bool frozen_seed_pose =
          frozen_frame_pose_indices_.count(entry.first) > 0;
      entry.second.rotation_dv->setActive(!frozen_seed_pose);
      entry.second.translation_dv->setActive(!frozen_seed_pose);
    }
    return return_value;
  }

  void FreezeFramePoses(const std::set<FrameBoardKey>& keys) {
    std::set<int> frame_indices;
    for (const FrameBoardKey& key : keys) {
      frame_indices.insert(key.first);
    }
    for (const int frame_index : frame_indices) {
      const auto it = frame_variables_.find(frame_index);
      if (it == frame_variables_.end()) {
        throw std::runtime_error(
            "cannot freeze a persistent seed frame without a pose variable");
      }
      frozen_frame_pose_indices_.insert(frame_index);
      it->second.rotation_dv->setActive(false);
      it->second.translation_dv->setActive(false);
    }
  }

  aslam::backend::SolutionReturnValue WarmupSeedIntrinsics(
      const std::set<FrameBoardKey>& keys,
      int max_iterations,
      double convergence_delta_j,
      double convergence_delta_x) {
    boost::shared_ptr<CalibrationBatch> problem = BuildBatch(
        keys, true, CameraOptimizationPhase::kCandidateTrustRegion,
        nullptr, nullptr);
    aslam::backend::Optimizer2Options optimizer_options;
    optimizer_options.maxIterations = std::max(1, max_iterations);
    optimizer_options.convergenceDeltaJ = convergence_delta_j;
    optimizer_options.convergenceDeltaX = convergence_delta_x;
    optimizer_options.nThreads = 4;
    optimizer_options.verbose = options_.verbose;
    if (options_.residual_model != ResidualModel::ImagePlane) {
      optimizer_options.trustRegionPolicy = boost::make_shared<
          aslam::backend::LevenbergMarquardtTrustRegionPolicy>(10.0);
    }
    aslam::backend::Optimizer2 optimizer(optimizer_options);
    optimizer.setProblem(problem);
    return optimizer.optimize();
  }

  aslam::backend::SolutionReturnValue AlignSeedLayoutOuter(
      const std::set<FrameBoardKey>& keys, int max_iterations,
      double convergence_delta_j, double convergence_delta_x) {
    // This aligns the rigid multi-board scene to the already selected camera.
    // Do not let regenerated internal points or camera intrinsics influence
    // this bootstrap: candidate admission starts from Outer4 geometry only.
    boost::shared_ptr<CalibrationBatch> problem = BuildBatch(
        keys, true, CameraOptimizationPhase::kPosePrefitFixedIntrinsics,
        nullptr, nullptr, 1.0, KbDistortionReleaseState(), true, true);
    aslam::backend::Optimizer2Options optimizer_options;
    optimizer_options.maxIterations = std::max(1, max_iterations);
    optimizer_options.convergenceDeltaJ = convergence_delta_j;
    optimizer_options.convergenceDeltaX = convergence_delta_x;
    optimizer_options.nThreads = 4;
    optimizer_options.verbose = options_.verbose;
    aslam::backend::Optimizer2 optimizer(optimizer_options);
    optimizer.setProblem(problem);
    return optimizer.optimize();
  }

  StateSnapshot CaptureState() const {
    StateSnapshot snapshot;
    camera_geometry_->projection().getParameters(
        snapshot.projection_parameters);
    camera_geometry_->projection().distortion().getParameters(
        snapshot.distortion_parameters);
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
    return CameraToIntrinsics<GeometryT>(*camera_geometry_);
  }

  KbRayCurveHealth EvaluateKbRayCurveHealth(
      const StateSnapshot& reference_state) const {
    KbRayCurveHealth health;
    const OuterBootstrapCameraIntrinsics current = CurrentCamera();
    if (!options_.use_kb_distortion_guard ||
        !IsBearingResidualModel(options_.residual_model) ||
        current.NormalizedFamilyString() != "pinhole-equi") {
      return health;
    }
    health.applicable = true;
    boost::shared_ptr<GeometryT> reference_geometry =
        MakePersistentGeometry<GeometryT>(current);
    reference_geometry->projection().setParameters(
        reference_state.projection_parameters);
    reference_geometry->projection().distortion().setParameters(
        reference_state.distortion_parameters);

    double squared_sum = 0.0;
    const double width = static_cast<double>(current.resolution.width);
    const double height = static_cast<double>(current.resolution.height);
    for (int polar_index = 0; polar_index <= 8; ++polar_index) {
      const double polar_rad = M_PI / 180.0 * (10.0 * polar_index);
      const int azimuth_count = polar_index == 0 ? 1 : 12;
      for (int azimuth_index = 0; azimuth_index < azimuth_count;
           ++azimuth_index) {
        const double azimuth_rad =
            2.0 * M_PI * azimuth_index / azimuth_count;
        Eigen::Vector3d reference_ray(
            std::sin(polar_rad) * std::cos(azimuth_rad),
            std::sin(polar_rad) * std::sin(azimuth_rad),
            std::cos(polar_rad));
        Eigen::Vector2d image_xy = Eigen::Vector2d::Zero();
        if (!reference_geometry->euclideanToKeypoint(reference_ray,
                                                      image_xy) ||
            !image_xy.allFinite() || image_xy.x() < 0.0 ||
            image_xy.y() < 0.0 || image_xy.x() >= width ||
            image_xy.y() >= height) {
          continue;
        }
        Eigen::Vector3d current_ray = Eigen::Vector3d::Zero();
        if (!camera_geometry_->keypointToEuclidean(image_xy, current_ray) ||
            !current_ray.allFinite() || current_ray.norm() <= 1e-12) {
          continue;
        }
        current_ray.normalize();
        const double angle_deg = 180.0 / M_PI * std::atan2(
            reference_ray.cross(current_ray).norm(),
            std::max(-1.0, std::min(1.0, reference_ray.dot(current_ray))));
        squared_sum += angle_deg * angle_deg;
        health.max_change_deg = std::max(health.max_change_deg, angle_deg);
        ++health.sample_count;
      }
    }
    if (health.sample_count > 0) {
      health.rms_change_deg =
          std::sqrt(squared_sum / static_cast<double>(health.sample_count));
    }

    const std::vector<double> distortion = current.DistortionVector();
    if (distortion.size() == 4u) {
      health.min_radial_derivative = std::numeric_limits<double>::infinity();
      for (int index = 0; index <= 170; ++index) {
        const double theta = 1.7 * index / 170.0;
        const double theta2 = theta * theta;
        const double derivative =
            1.0 + 3.0 * distortion[0] * theta2 +
            5.0 * distortion[1] * theta2 * theta2 +
            7.0 * distortion[2] * theta2 * theta2 * theta2 +
            9.0 * distortion[3] * theta2 * theta2 * theta2 * theta2;
        health.min_radial_derivative =
            std::min(health.min_radial_derivative, derivative);
      }
    }

    health.valid = health.sample_count >= 25 &&
        std::isfinite(health.rms_change_deg) &&
        std::isfinite(health.max_change_deg) &&
        std::isfinite(health.min_radial_derivative) &&
        health.rms_change_deg <= 1.0 &&
        health.max_change_deg <= 3.0 &&
        health.min_radial_derivative >= 0.05;
    if (!health.valid) {
      std::ostringstream stream;
      stream << "kb_ray_curve_validity samples=" << health.sample_count
             << " rms_change_deg=" << health.rms_change_deg
             << " max_change_deg=" << health.max_change_deg
             << " min_radial_derivative="
             << health.min_radial_derivative;
      health.failure_reason = stream.str();
    }
    return health;
  }

  void SetKbDistortionGuardReference(const StateSnapshot& reference_state) {
    if (reference_state.distortion_parameters.rows() == 4 &&
        reference_state.distortion_parameters.cols() == 1) {
      initial_distortion_parameters_ = reference_state.distortion_parameters;
    }
  }

  void RestoreState(const StateSnapshot& snapshot) {
    if (snapshot.projection_parameters.size() > 0) {
      camera_dv_.projectionDesignVariable()->setParameters(
          snapshot.projection_parameters);
    }
    if (snapshot.distortion_parameters.size() > 0) {
      camera_dv_.distortionDesignVariable()->setParameters(
          snapshot.distortion_parameters);
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
        // Rebuild the expression together with the restored DV values. A
        // rejected IncrementalEstimator batch can retain expression nodes
        // that reference the trial linearization; merely writing the matrix
        // back leaves later residuals evaluating a stale pose expression.
        const bool frozen_seed_pose =
            frozen_frame_pose_indices_.count(entry.first) > 0;
        InitializePoseVariable(&variable_it->second, entry.second,
                               !frozen_seed_pose);
      }
    }
    for (const auto& entry : snapshot.board_poses) {
      auto variable_it = board_variables_.find(entry.first);
      if (variable_it != board_variables_.end()) {
        InitializePoseVariable(&variable_it->second, entry.second,
                               !options_.fix_board_layout);
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

  bool CandidateCameraStepWithinTrustRegion(
      const StateSnapshot& anchor,
      std::string* reason,
      double* violation_ratio = nullptr) const {
    if (violation_ratio != nullptr) {
      *violation_ratio = 1.0;
    }
    if (!options_.optimize_candidate_intrinsics) {
      if (reason != nullptr) {
        reason->clear();
      }
      return true;
    }
	    if (anchor.projection_parameters.cols() < 1 ||
	        anchor.projection_parameters.rows() <= 0) {
	      if (reason != nullptr) {
	        reason->clear();
	      }
	      return true;
	    }
    boost::shared_ptr<GeometryT> anchor_geometry =
        MakePersistentGeometry<GeometryT>(CurrentCamera());
    anchor_geometry->projection().setParameters(anchor.projection_parameters);
    if (anchor.distortion_parameters.size() > 0) {
      anchor_geometry->projection().distortion().setParameters(
          anchor.distortion_parameters);
    }
    const OuterBootstrapCameraIntrinsics anchor_camera =
        CameraToIntrinsics<GeometryT>(*anchor_geometry);
    Eigen::MatrixXd current_parameters;
    camera_geometry_->projection().getParameters(current_parameters);
    if (current_parameters.rows() != anchor.projection_parameters.rows() ||
        current_parameters.cols() < 1 ||
        !current_parameters.allFinite()) {
      if (reason != nullptr) {
        *reason = "nonfinite_camera_parameters_after_candidate";
      }
      return false;
    }
    const OuterBootstrapCameraIntrinsics current_camera =
        CameraToIntrinsics<GeometryT>(*camera_geometry_);
    const bool focal_step_diagnostic_enabled =
        options_.max_candidate_focal_relative_step > 0.0;
    const bool principal_step_diagnostic_enabled =
        options_.max_candidate_principal_step_px > 0.0;
    const bool shape_step_diagnostic_enabled =
        options_.max_candidate_xi_alpha_step > 0.0;
    if (!focal_step_diagnostic_enabled &&
        !principal_step_diagnostic_enabled &&
        !shape_step_diagnostic_enabled) {
      if (reason != nullptr) {
        reason->clear();
      }
      return true;
    }
    const double anchor_fu = std::max(1.0, std::abs(anchor_camera.fu));
    const double anchor_fv = std::max(1.0, std::abs(anchor_camera.fv));
    const double max_fu_step =
        options_.max_candidate_focal_relative_step * anchor_fu;
    const double max_fv_step =
        options_.max_candidate_focal_relative_step * anchor_fv;
    const double d_xi = std::abs(current_camera.xi - anchor_camera.xi);
    const double d_alpha =
        std::abs(current_camera.alpha - anchor_camera.alpha);
    const double d_fu = std::abs(current_camera.fu - anchor_camera.fu);
    const double d_fv = std::abs(current_camera.fv - anchor_camera.fv);
    const double d_cu = std::abs(current_camera.cu - anchor_camera.cu);
    const double d_cv = std::abs(current_camera.cv - anchor_camera.cv);
    double max_distortion_delta = 0.0;
    const std::vector<double> anchor_distortion =
        anchor_camera.DistortionVector();
    const std::vector<double> current_distortion =
        current_camera.DistortionVector();
    if (anchor_distortion.size() == current_distortion.size()) {
      for (std::size_t index = 0; index < anchor_distortion.size(); ++index) {
        max_distortion_delta = std::max(
            max_distortion_delta,
            std::abs(current_distortion[index] - anchor_distortion[index]));
      }
    } else if (!anchor_distortion.empty() || !current_distortion.empty()) {
      max_distortion_delta = std::numeric_limits<double>::infinity();
    }
    double max_ratio = 1.0;
    if (shape_step_diagnostic_enabled) {
      const double max_xi_alpha_step = options_.max_candidate_xi_alpha_step;
      max_ratio = std::max(max_ratio, d_xi / max_xi_alpha_step);
      max_ratio = std::max(max_ratio, d_alpha / max_xi_alpha_step);
      max_ratio = std::max(max_ratio, max_distortion_delta / max_xi_alpha_step);
    }
    if (focal_step_diagnostic_enabled) {
      max_ratio = std::max(max_ratio, d_fu / std::max(1e-12, max_fu_step));
      max_ratio = std::max(max_ratio, d_fv / std::max(1e-12, max_fv_step));
    }
    if (principal_step_diagnostic_enabled) {
      const double max_principal_step =
          options_.max_candidate_principal_step_px;
      max_ratio = std::max(max_ratio, d_cu / max_principal_step);
      max_ratio = std::max(max_ratio, d_cv / max_principal_step);
    }
    if (violation_ratio != nullptr && std::isfinite(max_ratio)) {
      *violation_ratio = max_ratio;
    }
    // Kalibr's IncrementalEstimator accepts a batch using optimizer validity
    // plus information/rank gain. Do not hard-reject based on raw camera
    // parameter displacement here; the optional displacement ratio above is
    // diagnostic-only and is disabled by default.
    if (reason != nullptr) {
      reason->clear();
    }
    return true;
  }

 private:
  void AddCameraVariables(CameraOptimizationPhase camera_phase,
                          const boost::shared_ptr<CalibrationBatch>& batch) {
    constexpr bool kHasDistortionDv =
        GeometryT::projection_t::distortion_t::DesignVariableDimension > 0;
    bool projection_active = options_.optimize_candidate_intrinsics;
    if (camera_phase == CameraOptimizationPhase::kPosePrefitFixedIntrinsics) {
      projection_active = false;
    } else if (camera_phase ==
               CameraOptimizationPhase::kSeedFixedIntrinsics) {
      projection_active = options_.optimize_seed_intrinsics;
    }

    camera_dv_.setActive(projection_active,
                         projection_active && kHasDistortionDv,
                         false);
    batch->addDesignVariable(camera_dv_.projectionDesignVariable(),
                             kCameraInformationGroupId);
    batch->addDesignVariable(camera_dv_.distortionDesignVariable(),
                             kCameraInformationGroupId);
    batch->addDesignVariable(camera_dv_.shutterDesignVariable(),
                             kCameraInformationGroupId);
  }

  void MaybeAddCameraAnchorPrior(
      CameraOptimizationPhase camera_phase,
      const StateSnapshot* camera_anchor_state,
      const boost::shared_ptr<CalibrationBatch>& batch,
      double camera_anchor_weight_scale) {
    if (camera_phase != CameraOptimizationPhase::kCandidateTrustRegion ||
        camera_anchor_state == nullptr ||
        !options_.optimize_candidate_intrinsics ||
        !options_.use_candidate_intrinsics_anchor_prior ||
        camera_anchor_state->projection_parameters.rows() < 4 ||
        camera_anchor_state->projection_parameters.cols() < 1) {
      return;
    }
    const Eigen::MatrixXd& anchor = camera_anchor_state->projection_parameters;
    // Projection parameter dimensions differ by camera family: pinhole is 4,
    // UCM omni-none is 5, and DS/EUCM are 6.  Treat UCM explicitly here so
    // its xi/focal weak direction receives the same optional anchor prior.
    if (anchor.rows() != 4 && anchor.rows() != 5 && anchor.rows() != 6) {
      return;
    }
    const bool shape_prior_requested =
        options_.candidate_intrinsics_anchor_weight_xi_alpha > 0.0 ||
        options_.max_candidate_xi_alpha_step > 0.0;
    const bool focal_prior_requested =
        options_.candidate_intrinsics_anchor_weight_focal > 0.0 ||
        options_.max_candidate_focal_relative_step > 0.0;
    const bool principal_prior_requested =
        options_.candidate_intrinsics_anchor_weight_principal > 0.0 ||
        options_.max_candidate_principal_step_px > 0.0;
    if (!shape_prior_requested && !focal_prior_requested &&
        !principal_prior_requested) {
      return;
    }
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(anchor.rows());
    // Derive step-scaled prior sigmas from the frozen anchor, not the current
    // candidate state.  Otherwise a drifting focal length also changes the
    // scale of the trust-region prior that is supposed to restrain it.
    boost::shared_ptr<GeometryT> anchor_geometry =
        MakePersistentGeometry<GeometryT>(CurrentCamera());
    anchor_geometry->projection().setParameters(anchor);
    const OuterBootstrapCameraIntrinsics anchor_camera =
        CameraToIntrinsics<GeometryT>(*anchor_geometry);
    const double focal_sigma_u =
        options_.max_candidate_focal_relative_step > 0.0
            ? std::max(1.0, options_.max_candidate_focal_relative_step *
                                std::max(1.0, std::abs(anchor_camera.fu)))
            : std::numeric_limits<double>::infinity();
    const double focal_sigma_v =
        options_.max_candidate_focal_relative_step > 0.0
            ? std::max(1.0, options_.max_candidate_focal_relative_step *
                                std::max(1.0, std::abs(anchor_camera.fv)))
            : std::numeric_limits<double>::infinity();
    const double principal_sigma =
        options_.max_candidate_principal_step_px > 0.0
            ? options_.max_candidate_principal_step_px
            : std::numeric_limits<double>::infinity();
    const double xi_alpha_sigma =
        options_.max_candidate_xi_alpha_step > 0.0
            ? options_.max_candidate_xi_alpha_step
            : std::numeric_limits<double>::infinity();
    const double xi_alpha_weight =
        options_.candidate_intrinsics_anchor_weight_xi_alpha > 0.0
            ? options_.candidate_intrinsics_anchor_weight_xi_alpha
            : (std::isfinite(xi_alpha_sigma)
                   ? PriorWeightFromSigma(xi_alpha_sigma)
                   : 0.0);
    const double focal_weight_u =
        options_.candidate_intrinsics_anchor_weight_focal > 0.0
            ? options_.candidate_intrinsics_anchor_weight_focal
            : (std::isfinite(focal_sigma_u)
                   ? PriorWeightFromSigma(focal_sigma_u)
                   : 0.0);
    const double focal_weight_v =
        options_.candidate_intrinsics_anchor_weight_focal > 0.0
            ? options_.candidate_intrinsics_anchor_weight_focal
            : (std::isfinite(focal_sigma_v)
                   ? PriorWeightFromSigma(focal_sigma_v)
                   : 0.0);
    const double principal_weight =
        options_.candidate_intrinsics_anchor_weight_principal > 0.0
            ? options_.candidate_intrinsics_anchor_weight_principal
            : (std::isfinite(principal_sigma)
                   ? PriorWeightFromSigma(principal_sigma)
                   : 0.0);
    if (anchor.rows() == 6) {
      weights[0] = xi_alpha_weight;
      weights[1] = xi_alpha_weight;
      weights[2] = focal_weight_u;
      weights[3] = focal_weight_v;
      weights[4] = principal_weight;
      weights[5] = principal_weight;
    } else if (anchor.rows() == 5) {
      weights[0] = xi_alpha_weight;
      weights[1] = focal_weight_u;
      weights[2] = focal_weight_v;
      weights[3] = principal_weight;
      weights[4] = principal_weight;
    } else if (anchor.rows() == 4) {
      weights[0] = focal_weight_u;
      weights[1] = focal_weight_v;
      weights[2] = principal_weight;
      weights[3] = principal_weight;
    }
    if (weights.maxCoeff() <= 0.0) {
      return;
    }
    if (std::isfinite(camera_anchor_weight_scale) &&
        camera_anchor_weight_scale > 1.0) {
      weights *= camera_anchor_weight_scale;
    }
    if (anchor.rows() == 6) {
      boost::shared_ptr<ProjectionAnchorError<6> > prior(
          new ProjectionAnchorError<6>(
              camera_dv_.projectionDesignVariable().get(), anchor, weights));
      batch->addErrorTerm(prior);
    } else if (anchor.rows() == 5) {
      boost::shared_ptr<ProjectionAnchorError<5> > prior(
          new ProjectionAnchorError<5>(
              camera_dv_.projectionDesignVariable().get(), anchor, weights));
      batch->addErrorTerm(prior);
    } else if (anchor.rows() == 4) {
      boost::shared_ptr<ProjectionAnchorError<4> > prior(
          new ProjectionAnchorError<4>(
              camera_dv_.projectionDesignVariable().get(), anchor, weights));
      batch->addErrorTerm(prior);
    }
  }

  void MaybeAddDsSeedStabilityPrior(
      CameraOptimizationPhase camera_phase,
      const StateSnapshot* camera_anchor_state,
      const boost::shared_ptr<CalibrationBatch>& batch) {
    if ((camera_phase != CameraOptimizationPhase::kSeedActiveCamera &&
         camera_phase !=
             CameraOptimizationPhase::kSeedActiveCameraFixedFramePoses) ||
        !options_.normalize_information_gain_by_board_observation ||
        camera_anchor_state == nullptr ||
        camera_anchor_state->projection_parameters.rows() != 6 ||
        camera_anchor_state->projection_parameters.cols() != 1 ||
        CurrentCamera().NormalizedFamilyString() != "ds-none") {
      return;
    }

    // DS stores xi/alpha, focal lengths, and principal point in one projection
    // design variable. The seed needs an active camera to form its information
    // baseline, but releasing all six parameters from an outer-only seed is
    // underconstrained. Keep shape and principal point near the initializer;
    // focal lengths remain free and the normal candidate stages may refine all
    // parameters after the baseline is valid.
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(6);
    constexpr double kShapeSigma = 0.05;
    constexpr double kPrincipalPointSigmaPx = 25.0;
    weights[0] = PriorWeightFromSigma(kShapeSigma);
    weights[1] = PriorWeightFromSigma(kShapeSigma);
    weights[4] = PriorWeightFromSigma(kPrincipalPointSigmaPx);
    weights[5] = PriorWeightFromSigma(kPrincipalPointSigmaPx);
    boost::shared_ptr<ProjectionAnchorError<6> > prior(
        new ProjectionAnchorError<6>(
            camera_dv_.projectionDesignVariable().get(),
            camera_anchor_state->projection_parameters, weights));
    batch->addErrorTerm(prior);
  }

  void MaybeAddKbDistortionPrior(
      CameraOptimizationPhase camera_phase,
      const StateSnapshot* camera_anchor_state,
      const KbDistortionReleaseState& release_state,
      const boost::shared_ptr<CalibrationBatch>& batch,
      double camera_anchor_weight_scale) {
    const OuterBootstrapCameraIntrinsics current = CurrentCamera();
    if (camera_phase != CameraOptimizationPhase::kCandidateTrustRegion ||
        !options_.optimize_candidate_intrinsics ||
        !options_.use_kb_distortion_guard ||
        !IsBearingResidualModel(options_.residual_model) ||
        current.NormalizedFamilyString() != "pinhole-equi" ||
        camera_anchor_state == nullptr ||
        camera_anchor_state->distortion_parameters.rows() != 4 ||
        camera_anchor_state->distortion_parameters.cols() != 1 ||
        initial_distortion_parameters_.rows() != 4 ||
        initial_distortion_parameters_.cols() != 1) {
      return;
    }
    Eigen::MatrixXd anchor = camera_anchor_state->distortion_parameters;
    if (!release_state.k3_released) {
      anchor(2, 0) = initial_distortion_parameters_(2, 0);
    }
    if (!release_state.k4_released) {
      anchor(3, 0) = initial_distortion_parameters_(3, 0);
    }
    Eigen::Vector4d weights;
    weights[0] = PriorWeightFromSigma(0.02);
    weights[1] = PriorWeightFromSigma(0.02);
    weights[2] = PriorWeightFromSigma(
        release_state.k3_released ? 0.01 : 1e-6);
    weights[3] = PriorWeightFromSigma(
        release_state.k4_released ? 0.005 : 1e-6);
    if (std::isfinite(camera_anchor_weight_scale) &&
        camera_anchor_weight_scale > 1.0) {
      weights *= camera_anchor_weight_scale;
    }
    boost::shared_ptr<ProjectionAnchorError<4> > prior(
        new ProjectionAnchorError<4>(
            camera_dv_.distortionDesignVariable().get(), anchor, weights));
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
      InitializePoseVariable(&variable, board->T_reference_board,
                             !options_.fix_board_layout);
    }
  }

  void SetBoardLayoutActive(bool active) {
    for (auto& entry : board_variables_) {
      entry.second.rotation_dv->setActive(active);
      entry.second.translation_dv->setActive(active);
    }
  }

  void AddBoardVariables(const boost::shared_ptr<CalibrationBatch>& batch) {
    for (const auto& entry : board_variables_) {
      AddPoseVariableDvs(entry.second, kBoardLayoutGroupId, batch);
    }
  }

  void MaybeAddSquarePixelFocalPrior(
      const boost::shared_ptr<CalibrationBatch>& batch) {
    if (!options_.square_pixel_focal_prior) {
      return;
    }
    boost::shared_ptr<SquarePixelFocalError> prior(
        new SquarePixelFocalError(
            camera_dv_.projectionDesignVariable().get()));
    batch->addErrorTerm(prior);
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

  void SetFramePosesActive(const std::set<FrameBoardKey>& keys, bool active) {
    std::set<int> frame_indices;
    for (const FrameBoardKey& key : keys) {
      frame_indices.insert(key.first);
    }
    for (const int frame_index : frame_indices) {
      const auto it = frame_variables_.find(frame_index);
      if (it == frame_variables_.end()) {
        continue;
      }
      it->second.rotation_dv->setActive(active);
      it->second.translation_dv->setActive(active);
    }
  }

 private:
  const PoseVariable* FindFrameVariable(int frame_index) const {
    const auto it = frame_variables_.find(frame_index);
    return it == frame_variables_.end() ? nullptr : &it->second;
  }

  const PoseVariable* FindBoardVariable(int board_id) const {
    const auto it = board_variables_.find(board_id);
    return it == board_variables_.end() ? nullptr : &it->second;
  }

  void AddResiduals(const std::set<FrameBoardKey>& keys,
                    const boost::shared_ptr<CalibrationBatch>& batch,
                    ResidualConstructionCounts* residual_counts,
                    bool outer_only = false) {
    const aslam::backend::TransformationExpression identity_transform(
        Eigen::Matrix4d::Identity());
    for (const JointPointObservation& observation :
         candidate_pool_bundle_.measurement_dataset.solver_observations) {
      if (!observation.used_in_solver ||
          keys.count(FrameBoardKey(observation.frame_index,
                                   observation.board_id)) == 0) {
        continue;
      }
      if (outer_only && observation.point_type != JointPointType::Outer) {
        continue;
      }
      const FrameBoardKey observation_key(observation.frame_index,
                                          observation.board_id);
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
      const auto budget_it = observation_budgets_.find(observation_key);
      const ObservationBudget budget =
          budget_it == observation_budgets_.end() ? ObservationBudget{}
                                                  : budget_it->second;
      const double weight =
          ComputePersistentBalanceWeight(
              budget, observation.point_type,
              options_.single_board_dense_grid_profile,
              options_.uniform_control_point_weighting,
              options_.internal_role_budget_when_mixed) *
          std::max(0.0, observation.final_observation_weight);
      if (!(weight > 0.0)) {
        continue;
      }
	      const Eigen::Matrix2d inv_r =
	          weight * Eigen::Matrix2d::Identity();
	      const ResidualModel requested_model = options_.residual_model;
	      const bool bearing_capable_model =
	          requested_model == ResidualModel::SphereAngular ||
	          requested_model == ResidualModel::HybridEdgeAngular ||
	          requested_model == ResidualModel::PolarContinuousHybrid ||
	          requested_model == ResidualModel::Chordal ||
	          requested_model == ResidualModel::PixelChordalHybrid;
	      AngularObservationGeometry angular_observation_geometry;
	      bool have_angular_observation_geometry = false;
	      if (bearing_capable_model) {
	        try {
          have_angular_observation_geometry = ComputeObservationGeometryForCamera(
              *camera_geometry_, observation.image_xy,
              &angular_observation_geometry);
        } catch (const std::exception&) {
          have_angular_observation_geometry = false;
        }
        if (!have_angular_observation_geometry && residual_counts != nullptr) {
          ++residual_counts->angular_observation_geometry_failure_count;
        }
	        if (!have_angular_observation_geometry) {
	          // Bearing-space modes must not silently degrade to a pixel
	          // objective when the active camera cannot unproject a point.
	          continue;
	        }
      }

	      bool use_angular_residual = false;
	      double continuous_angular_weight = 0.0;
	      bool use_chordal_residual = false;
	      if (requested_model == ResidualModel::SphereAngular) {
	        use_angular_residual = true;
	      } else if (requested_model == ResidualModel::HybridEdgeAngular) {
        use_angular_residual =
            have_angular_observation_geometry &&
            ShouldUseAngularResidual(requested_model,
                                     angular_observation_geometry.polar_angle_deg,
                                     options_.hybrid_angular_threshold_deg);
      } else if (requested_model == ResidualModel::PolarContinuousHybrid &&
                 have_angular_observation_geometry) {
        continuous_angular_weight =
            ComputePolarContinuousAngularWeight(
	                angular_observation_geometry.polar_angle_deg,
	                options_.polar_continuous_hybrid_threshold_deg,
	                options_.polar_continuous_hybrid_temperature_deg);
	      } else if (requested_model == ResidualModel::Chordal ||
	                 requested_model == ResidualModel::PixelChordalHybrid) {
	        use_chordal_residual = have_angular_observation_geometry;
	      }

	      const bool use_continuous_hybrid =
	          requested_model == ResidualModel::PolarContinuousHybrid;
	      const bool use_pixel_chordal_hybrid =
	          requested_model == ResidualModel::PixelChordalHybrid;
	      const double pixel_weight_scale =
	          use_pixel_chordal_hybrid
	              ? std::max(0.0, options_.pixel_residual_weight)
	              : (use_continuous_hybrid
	                     ? std::max(0.0, 1.0 - continuous_angular_weight)
	                     : ((use_angular_residual || use_chordal_residual)
	                            ? 0.0
	                            : 1.0));
	      const double angular_weight_scale =
	          use_continuous_hybrid
	              ? continuous_angular_weight * chordal_reference_focal_px_ *
	                    chordal_reference_focal_px_
	              : (use_angular_residual
	                     ? chordal_reference_focal_px_ *
	                           chordal_reference_focal_px_
	                     : 0.0);
	      const double chordal_weight_scale =
	          use_chordal_residual
	              ? std::max(0.0, options_.chordal_residual_weight)
	              : 0.0;

      if (pixel_weight_scale > 0.0) {
        boost::shared_ptr<IncrementalReprojectionError<GeometryT> > error(
            new IncrementalReprojectionError<GeometryT>(
                observation.image_xy, inv_r * pixel_weight_scale,
                options_.huber_delta_pixels, options_.use_huber_loss,
                point_camera, camera_dv_,
                options_.invalid_projection_penalty_pixels));
        batch->addErrorTerm(error);
        if (residual_counts != nullptr) {
          ++residual_counts->image_plane_residual_count;
        }
      }

	      if (angular_weight_scale > 0.0 && have_angular_observation_geometry) {
	        Eigen::Matrix2d angular_inv_r = inv_r * angular_weight_scale;
	        if (options_.angular_local_whitening_enabled) {
	          BearingCovarianceResult covariance_result;
	          if (!ComputeLocalBearingWhiteningForCamera(
	                  *camera_geometry_, observation.image_xy,
	                  angular_observation_geometry, options_,
	                  &covariance_result)) {
	            if (residual_counts != nullptr) {
	              ++residual_counts->angular_local_whitening_failure_count;
	            }
	            continue;
	          }
	          angular_inv_r =
	              weight * covariance_result.sqrt_information.transpose() *
	              covariance_result.sqrt_information;
	          if (residual_counts != nullptr) {
	            ++residual_counts->angular_local_whitening_success_count;
	            if (covariance_result.whitening_clamped) {
	              ++residual_counts->angular_local_whitening_clamped_count;
	            }
	            residual_counts->angular_local_whitening_sigma_sum_rad +=
	                covariance_result.tangent_sigma_mean_rad;
	            residual_counts->angular_local_whitening_sigma_min_rad =
	                std::min(
	                    residual_counts->angular_local_whitening_sigma_min_rad,
	                    covariance_result.tangent_sigma_min_rad);
	            residual_counts->angular_local_whitening_sigma_max_rad =
	                std::max(
	                    residual_counts->angular_local_whitening_sigma_max_rad,
	                    covariance_result.tangent_sigma_max_rad);
	            residual_counts->angular_local_whitening_weight_sum +=
	                covariance_result.whitening_weight_mean;
	            residual_counts->angular_local_whitening_weight_min =
	                std::min(
	                    residual_counts->angular_local_whitening_weight_min,
	                    covariance_result.whitening_weight_min);
	            residual_counts->angular_local_whitening_weight_max =
	                std::max(
	                    residual_counts->angular_local_whitening_weight_max,
	                    covariance_result.whitening_weight_max);
	          }
	        }
	        const double huber_delta_radians =
	            options_.single_board_dense_grid_profile
	                ? std::min(options_.outer_huber_delta_radians,
	                           options_.internal_huber_delta_radians)
	                : observation.point_type == JointPointType::Outer
	                      ? options_.outer_huber_delta_radians
	                      : options_.internal_huber_delta_radians;
        boost::shared_ptr<IncrementalAngularReprojectionError<GeometryT> >
            error(new IncrementalAngularReprojectionError<GeometryT>(
                observation.image_xy, angular_inv_r,
                huber_delta_radians,
                options_.use_huber_loss, point_camera, camera_dv_,
                options_.invalid_projection_penalty_radians,
                options_.angular_use_normalize_jacobian,
                options_.angular_observed_ray_mode,
                angular_observation_geometry));
        batch->addErrorTerm(error);
        if (residual_counts != nullptr) {
          ++residual_counts->angular_residual_count;
          if (requested_model == ResidualModel::HybridEdgeAngular ||
              requested_model == ResidualModel::PolarContinuousHybrid) {
            ++residual_counts->hybrid_angular_selected_count;
	          }
	        }
		      }

		      if (chordal_weight_scale > 0.0 && have_angular_observation_geometry) {
	        const double huber_delta_chordal =
	            options_.single_board_dense_grid_profile
	                ? std::min(options_.outer_huber_delta_radians,
	                           options_.internal_huber_delta_radians)
	                : observation.point_type == JointPointType::Outer
	                      ? options_.outer_huber_delta_radians
	                      : options_.internal_huber_delta_radians;
	        const Eigen::Matrix3d inv_r_chordal =
	            weight * chordal_weight_scale * chordal_reference_focal_px_ *
	            chordal_reference_focal_px_ * Eigen::Matrix3d::Identity();
	        boost::shared_ptr<IncrementalChordalReprojectionError<GeometryT> >
	            error(new IncrementalChordalReprojectionError<GeometryT>(
	                observation.image_xy, inv_r_chordal,
	                huber_delta_chordal,
	                options_.use_huber_loss, point_camera, camera_dv_,
	                options_.invalid_projection_penalty_radians,
	                options_.angular_use_normalize_jacobian,
	                options_.angular_observed_ray_mode,
	                angular_observation_geometry));
	        batch->addErrorTerm(error);
	        if (residual_counts != nullptr) {
	          ++residual_counts->chordal_residual_count;
	          if (requested_model == ResidualModel::PixelChordalHybrid) {
	            ++residual_counts->hybrid_chordal_selected_count;
	          }
	        }
	      }
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
	  std::map<FrameBoardKey, ObservationBudget> observation_budgets_;
	  boost::shared_ptr<GeometryT> camera_geometry_;
	  CameraDv camera_dv_;
	  double chordal_reference_focal_px_ = 1.0;
	  Eigen::MatrixXd initial_distortion_parameters_;
  PoseVariableMap frame_variables_;
  // Seed poses are trusted scene-state inputs for the incremental camera
  // information baseline.  They remain in the residual graph, but must not
  // be re-optimized whenever a later candidate batch is tested.
  std::set<int> frozen_frame_pose_indices_;
  PoseVariableMap board_variables_;
	};

Stage5IncrementalBackendEstimatorOptions MakeOptions(
    const TrialBackendFrameBoardSelectionOptions& selection_options,
    const AslamBackendCalibrationOptions& backend_runner_options) {
  Stage5IncrementalBackendEstimatorOptions options;
  options.enabled = true;
  options.single_board_dense_grid_profile =
      selection_options.single_board_dense_grid_profile;
  options.information_gain_threshold =
      selection_options.acceptance_information_gain_threshold;
  options.information_gain_threshold_explicit =
      selection_options.acceptance_information_gain_threshold_explicit;
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
  options.residual_model = backend_runner_options.residual_model;
  if (options.residual_model != ResidualModel::ImagePlane) {
    // Match Kalibr's incremental camera-calibration state-step tolerance.
    // Bearing objectives contain weak pose/depth directions for which the
    // pixel profile's stricter 1e-4 threshold causes false non-convergence.
    options.convergence_delta_x =
        std::max(options.convergence_delta_x, 1e-3);
  }
  if (options.residual_model != ResidualModel::ImagePlane) {
    // Bearing-space modes use a larger per-pass safety ceiling and adaptive
    // continuation. Pixel mode keeps its established trial budget.
    constexpr int kBearingMaximumIterationsPerPass = 50;
    options.max_iterations = std::max(
        kBearingMaximumIterationsPerPass,
        std::max(options.max_iterations,
                 backend_runner_options.max_iterations));
  }
  options.huber_delta_pixels =
      std::min(backend_runner_options.outer_huber_delta_pixels,
               backend_runner_options.internal_huber_delta_pixels);
  options.outer_huber_delta_radians =
      backend_runner_options.outer_huber_delta_radians;
  options.internal_huber_delta_radians =
      backend_runner_options.internal_huber_delta_radians;
  options.invalid_projection_penalty_pixels =
      backend_runner_options.invalid_projection_penalty_pixels;
  options.invalid_projection_penalty_radians =
      backend_runner_options.invalid_projection_penalty_radians;
  options.hybrid_angular_threshold_deg =
      backend_runner_options.hybrid_angular_threshold_deg;
	  options.polar_continuous_hybrid_threshold_deg =
	      backend_runner_options.polar_continuous_hybrid_threshold_deg;
	  options.polar_continuous_hybrid_temperature_deg =
	      backend_runner_options.polar_continuous_hybrid_temperature_deg;
	  options.pixel_residual_weight =
	      backend_runner_options.pixel_residual_weight;
	  options.chordal_residual_weight =
	      backend_runner_options.chordal_residual_weight;
	  options.angular_use_normalize_jacobian =
	      backend_runner_options.angular_use_normalize_jacobian;
  options.angular_local_whitening_enabled =
      backend_runner_options.angular_local_whitening_enabled;
  options.angular_local_whitening_pixel_sigma_px =
      backend_runner_options.angular_local_whitening_pixel_sigma_px;
  options.angular_local_whitening_covariance_damping =
      backend_runner_options.angular_local_whitening_covariance_damping;
  options.angular_local_whitening_min_sigma_rad =
      backend_runner_options.angular_local_whitening_min_sigma_rad;
  options.angular_local_whitening_max_weight =
      backend_runner_options.angular_local_whitening_max_weight;
  options.angular_observed_ray_mode =
      backend_runner_options.angular_observed_ray_mode;
  options.optimize_seed_intrinsics =
      selection_options.optimize_intrinsics_in_trial &&
      selection_options.force_include_list_is_exact_input;
  options.independent_frame_board_camera_warmup =
      selection_options.independent_frame_board_camera_warmup &&
      !selection_options.single_board_dense_grid_profile &&
      !selection_options.model_aware_information_coreset;
  options.model_aware_ds_independent_seed_camera_stabilization =
      selection_options.model_aware_ds_independent_seed_camera_stabilization;
  options.optimize_candidate_intrinsics =
      selection_options.optimize_intrinsics_in_trial;
  // Round 2 establishes the shared layout before model-aware selection begins.
  // Keep that layout fixed while Persistent BA decides whether a new frame
  // provides usable camera information. Otherwise one candidate can jointly
  // trade camera, frame pose, and rigid layout, which makes the incremental
  // solve ill-conditioned. The established layout remains available to the
  // normal Round 2 shared-layout optimization; this only scopes selection.
  options.fix_board_layout =
      selection_options.persistent_fix_board_layout ||
      selection_options.model_aware_information_coreset;
  options.use_candidate_intrinsics_anchor_prior =
      selection_options.optimize_intrinsics_in_trial &&
      selection_options.persistent_intrinsics_anchor_prior_enabled;
  options.normalize_information_gain_by_board_observation =
      selection_options.model_aware_information_coreset;
  options.model_aware_progressive_seed =
      selection_options.model_aware_information_coreset &&
      selection_options.model_aware_progressive_seed;
  options.internal_role_budget_when_mixed = std::max(
      0.0, std::min(1.0,
                    selection_options.persistent_internal_role_budget_when_mixed));
  options.uniform_control_point_weighting =
      selection_options.persistent_uniform_control_point_weighting;
  options.training_robust_checkpoint_selection =
      selection_options.persistent_training_robust_checkpoint_selection;
  options.square_pixel_focal_prior =
      selection_options.persistent_square_pixel_focal_prior;
  options.align_model_aware_seed_layout_outer =
      selection_options.model_aware_seed_layout_alignment;
  options.use_model_aware_candidate_pose_prefit =
      selection_options.model_aware_candidate_pose_prefit;
  options.use_candidate_relative_residual_gate =
      !selection_options.model_aware_information_coreset;
  // Model-aware candidates already use their candidate-only and committed
  // residual health gates.  Keep the historical path free of the expensive
  // full-training pose-refit veto; that check made every candidate appear
  // non-converged on otherwise valid multi-board datasets.
  options.use_full_training_pose_refit_health_gate =
      !selection_options.model_aware_information_coreset;
  options.full_training_pose_refit_outer_only_health = false;
  options.split_residual_outer_only_health =
      selection_options.model_aware_information_coreset;
  if (selection_options.model_aware_information_coreset) {
    // A frame batch is initialized from a fixed shared multi-board layout,
    // unlike Kalibr's independent target view.  Preserve the historical
    // 20-step budget unless a caller explicitly requests more iterations;
    // all information and residual-health gates remain unchanged.
    options.max_iterations = std::max(20, options.max_iterations);
    options.max_continuation_rounds = 0;
  }
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
  options.adaptive_saturation_stop_enabled =
      selection_options.budget_mode ==
      TrialBackendFrameBoardSelectionOptions::BudgetMode::KalibrStyle;
  if (options.single_board_dense_grid_profile) {
    // kalibr_calibrate_cameras sets maxIterations=50 for each incremental
    // target-view batch. Our camera has already been initialized jointly from
    // all training views; keep that state fixed while the single forced seed
    // pose is stabilized. Releasing six DS parameters against one planar view
    // is rank-deficient and can destroy an otherwise valid initialization.
    // Candidate batches release intrinsics immediately afterwards.
    options.max_iterations = 50;
    options.convergence_delta_j = 1e-3;
    options.convergence_delta_x = 1e-3;
    options.check_validity = true;
    options.optimize_seed_intrinsics = false;
    options.optimize_candidate_intrinsics = true;
    options.use_candidate_intrinsics_anchor_prior = false;
    options.use_split_residual_health_gate = true;
    options.use_bearing_pixel_safety_gate = false;
    options.use_kb_distortion_guard = false;
    options.adaptive_saturation_stop_enabled = false;
    options.max_candidate_focal_relative_step = 0.0;
    options.max_candidate_principal_step_px = 0.0;
    options.max_candidate_xi_alpha_step = 0.0;
  }
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
  if (backend_runner_options.use_point_type_residual_split ||
      backend_runner_options.angular_auxiliary_enabled) {
    return set_reason(
        "persistent incremental estimator supports only a single primary "
        "residual model; point-type split and angular auxiliary are not yet "
        "implemented");
  }
	  if (backend_runner_options.residual_model != ResidualModel::ImagePlane &&
	      backend_runner_options.residual_model != ResidualModel::SphereAngular &&
	      backend_runner_options.residual_model != ResidualModel::HybridEdgeAngular &&
	      backend_runner_options.residual_model != ResidualModel::PolarContinuousHybrid &&
	      backend_runner_options.residual_model != ResidualModel::Chordal &&
	      backend_runner_options.residual_model != ResidualModel::PixelChordalHybrid) {
	    return set_reason(
	        "persistent incremental estimator supports image_plane, "
	        "sphere_angular, hybrid_edge_angular, polar_continuous_hybrid, "
	        "chordal, and pixel_chordal_hybrid residual models only");
	  }
  if (!baseline_bundle.IsReadyForBackend() ||
      !candidate_pool_bundle.IsReadyForBackend()) {
    return set_reason("bundle is not ready for backend");
  }
  const std::string baseline_family =
      baseline_bundle.scene_state.camera.NormalizedFamilyString();
  const std::string candidate_family =
      candidate_pool_bundle.scene_state.camera.NormalizedFamilyString();
  if (baseline_family != candidate_family) {
    return set_reason("baseline and candidate camera families differ");
  }
  if (baseline_family != "ds-none" &&
      baseline_family != "eucm-none" &&
      baseline_family != "pinhole-equi" &&
      baseline_family != "omni-radtan" &&
      baseline_family != "omni-none") {
    return set_reason(
        "persistent incremental estimator supports ds-none, eucm-none, pinhole-equi, omni-none, and omni-radtan only");
  }
  if (backend_runner_options.angular_local_whitening_enabled &&
      (backend_runner_options.residual_model != ResidualModel::SphereAngular ||
       baseline_family != "ds-none")) {
    return set_reason(
        "angular local whitening currently requires sphere_angular with "
        "ds-none");
  }
  if (!backend_options.optimize_frame_poses) {
    return set_reason("frame pose optimization must be enabled");
  }
  // The persistent estimator has an explicit fixed-layout mode.  This is the
  // intended mode after Round 1 when supplemental non-reference views are
  // admitted: their frame poses and camera may be optimized, while the
  // established reference-to-board transforms must remain unchanged.
  const bool layout_is_explicitly_fixed =
      selection_options.persistent_fix_board_layout ||
      selection_options.model_aware_information_coreset;
  if (!backend_options.optimize_board_poses && !layout_is_explicitly_fixed) {
    return set_reason(
        "board pose optimization must be enabled unless the persistent "
        "board layout is explicitly fixed");
  }
  if (reason != nullptr) {
    reason->clear();
  }
  return true;
}

template <typename GeometryT>
Stage5IncrementalBackendEstimatorResult RunStage5IncrementalBackendEstimatorTyped(
    const CalibrationStateBundle& baseline_bundle,
    const CalibrationStateBundle& candidate_pool_bundle,
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
  std::vector<Stage5IncrementalBackendBatchInput> effective_candidate_batches =
      candidate_batches;
  result.candidate_batch_count =
      static_cast<int>(effective_candidate_batches.size());
  result.compatible = true;
  result.information_gain_target = "camera_intrinsics_only";
  result.board_layout_in_information_group = false;
  result.camera_information_group_id =
      static_cast<int>(kCameraInformationGroupId);
  result.board_layout_group_id = static_cast<int>(kBoardLayoutGroupId);
  result.transformation_group_id = static_cast<int>(kTransformationGroupId);

  const Stage5IncrementalBackendEstimatorOptions estimator_options =
      MakeOptions(selection_options, backend_runner_options);
  bool fixed_forced_schedule =
      !effective_candidate_batches.empty() &&
      std::all_of(
          effective_candidate_batches.begin(), effective_candidate_batches.end(),
          [](const Stage5IncrementalBackendBatchInput& input) {
            return input.force;
          });
  result.board_layout_fixed = estimator_options.fix_board_layout;
  const double reference_fu =
      std::max(1.0, std::abs(baseline_bundle.scene_state.camera.fu));
  const double reference_fv =
      std::max(1.0, std::abs(baseline_bundle.scene_state.camera.fv));
  result.solver_bearing_reference_focal_px =
      std::sqrt(reference_fu * reference_fv);
  result.solver_bearing_residual_scale =
      estimator_options.residual_model == ResidualModel::ImagePlane ||
              estimator_options.angular_local_whitening_enabled
          ? 1.0
          : result.solver_bearing_reference_focal_px;
  result.solver_max_iterations = estimator_options.max_iterations;
  result.solver_convergence_delta_j = estimator_options.convergence_delta_j;
  result.solver_convergence_delta_x = estimator_options.convergence_delta_x;
  if (estimator_options.single_board_dense_grid_profile) {
    result.solver_profile_name =
        "kalibr_checkerboard_dense_grid_incremental";
    result.solver_objective_unit =
        estimator_options.residual_model == ResidualModel::ImagePlane
            ? "px"
            : "px_equivalent";
  } else if (estimator_options.residual_model == ResidualModel::ImagePlane) {
    result.solver_profile_name = "pixel_native_incremental_trust_region";
    result.solver_objective_unit = "px";
  } else if (estimator_options.angular_local_whitening_enabled) {
    result.solver_profile_name =
        "tangent_local_covariance_whitened_incremental_trust_region";
    result.solver_objective_unit = "normalized";
  } else if (estimator_options.residual_model == ResidualModel::SphereAngular) {
    result.solver_profile_name =
        "tangent_fixed_focal_px_equivalent_incremental_trust_region";
    result.solver_objective_unit = "px_equivalent";
  } else if (estimator_options.residual_model ==
             ResidualModel::PolarContinuousHybrid) {
    result.solver_profile_name =
        "polar_continuous_px_equivalent_incremental_trust_region";
    result.solver_objective_unit = "px_equivalent";
  } else {
    result.solver_profile_name =
        "bearing_fixed_focal_px_equivalent_incremental_trust_region";
    result.solver_objective_unit = "px_equivalent";
  }
  if (!std::isfinite(estimator_options.hybrid_angular_threshold_deg) ||
      estimator_options.hybrid_angular_threshold_deg < 0.0 ||
      estimator_options.hybrid_angular_threshold_deg > 180.0 ||
      !std::isfinite(
          estimator_options.polar_continuous_hybrid_threshold_deg) ||
      estimator_options.polar_continuous_hybrid_threshold_deg < 0.0 ||
      estimator_options.polar_continuous_hybrid_threshold_deg > 180.0 ||
      !std::isfinite(
          estimator_options.polar_continuous_hybrid_temperature_deg) ||
      estimator_options.polar_continuous_hybrid_temperature_deg <= 0.0 ||
      !std::isfinite(estimator_options.pixel_residual_weight) ||
      estimator_options.pixel_residual_weight < 0.0 ||
      !std::isfinite(estimator_options.chordal_residual_weight) ||
      estimator_options.chordal_residual_weight < 0.0 ||
      !std::isfinite(
          estimator_options.angular_local_whitening_pixel_sigma_px) ||
      estimator_options.angular_local_whitening_pixel_sigma_px <= 0.0 ||
      !std::isfinite(
          estimator_options.angular_local_whitening_covariance_damping) ||
      estimator_options.angular_local_whitening_covariance_damping < 0.0 ||
      !std::isfinite(
          estimator_options.angular_local_whitening_min_sigma_rad) ||
      estimator_options.angular_local_whitening_min_sigma_rad <= 0.0 ||
      !std::isfinite(
          estimator_options.angular_local_whitening_max_weight) ||
      estimator_options.angular_local_whitening_max_weight < 1.0) {
    result.failure_reason = "invalid residual-model configuration";
    return result;
  }
  if (estimator_options.residual_model ==
          ResidualModel::PixelChordalHybrid &&
      estimator_options.pixel_residual_weight == 0.0 &&
      estimator_options.chordal_residual_weight == 0.0) {
    result.failure_reason =
        "pixel_chordal_hybrid has no active residual block";
    return result;
  }
  const SelectionResidualMetric selection_metric =
      SelectionMetricForResidualModel(
          estimator_options.residual_model,
          estimator_options.angular_local_whitening_enabled);
  result.selection_metric_name = SelectionMetricName(selection_metric);
  result.selection_metric_unit = SelectionMetricUnit(selection_metric);
  result.residual_health_threshold_source =
      estimator_options.single_board_dense_grid_profile
          ? "diagnostics_only_kalibr_information_rank_acceptance"
          : selection_metric == SelectionResidualMetric::kPixel
          ? "pixel_trial_rmse_threshold"
          : "adaptive_seed_and_candidate_metric_stats";
  result.normalize_information_gain_by_board_observation =
      estimator_options.normalize_information_gain_by_board_observation;
  result.split_residual_health_gate_enabled =
      estimator_options.use_split_residual_health_gate;
  result.bearing_pixel_safety_gate_enabled =
      estimator_options.use_bearing_pixel_safety_gate &&
      IsBearingResidualModel(estimator_options.residual_model);
  result.full_training_pose_refit_health_gate_enabled =
      estimator_options.use_full_training_pose_refit_health_gate;
  const std::string active_camera_family =
      baseline_bundle.scene_state.camera.NormalizedFamilyString();
  result.kb_distortion_guard_enabled =
      estimator_options.use_kb_distortion_guard &&
      IsBearingResidualModel(estimator_options.residual_model) &&
      active_camera_family == "pinhole-equi";
  result.adaptive_saturation_stop_enabled =
      estimator_options.adaptive_saturation_stop_enabled;

  aslam::calibration::IncrementalEstimator::Options inc_options;
  // The persistent Stage5 layer owns accept/reject so the health metric can
  // follow the selected residual model. Keep each optimized candidate
  // temporarily, then explicitly commit or roll it back below.
  inc_options.infoGainDelta =
      estimator_options.single_board_dense_grid_profile
          ? estimator_options.information_gain_threshold
          : -std::numeric_limits<double>::max();
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
    PersistentProblemBuilder<GeometryT> builder(
        baseline_bundle, candidate_pool_bundle, estimator_options);
    aslam::calibration::IncrementalEstimator estimator(
        kCameraInformationGroupId, inc_options, solver_options,
        optimizer_options);

    const FullTrainingPoseRefitStats initial_full_training_stats =
        builder.EvaluateFullTrainingPoseRefitPixel();
    FullTrainingPoseRefitStats health_initial_full_training_stats =
        initial_full_training_stats;
    FullTrainingPoseRefitStats committed_full_training_stats =
        initial_full_training_stats;
    result.initial_full_training_pixel_rmse =
        initial_full_training_stats.pixel_stats.Rmse();
    result.initial_full_training_pixel_p95 =
        initial_full_training_stats.pixel_stats.P95();
    result.initial_full_training_pose_success_rate =
        initial_full_training_stats.PoseSuccessRate();
    result.initial_full_training_pose_success_count =
        initial_full_training_stats.pose_success_count;
    result.initial_full_training_pose_total_count =
        initial_full_training_stats.pose_total_count;
    result.initial_full_training_invalid_projection_count =
        initial_full_training_stats.pixel_stats.invalid_projection_count;
    result.initial_full_training_invalid_outer_projection_count =
        initial_full_training_stats.pixel_stats.invalid_outer_projection_count;
    result.initial_full_training_invalid_internal_projection_count =
        initial_full_training_stats.pixel_stats.invalid_internal_projection_count;
    if (result.full_training_pose_refit_health_gate_enabled &&
        !initial_full_training_stats.IsUsable()) {
      result.failure_reason =
          "initial full-training independent-pose refit evaluation failed";
      return result;
    }

    const bool model_aware_ds_independent_seed_stabilization =
        estimator_options.model_aware_ds_independent_seed_camera_stabilization &&
        estimator_options.normalize_information_gain_by_board_observation &&
        builder.CurrentCamera().NormalizedFamilyString() == "ds-none";
    result.independent_frame_board_camera_warmup_requested =
        estimator_options.independent_frame_board_camera_warmup ||
        model_aware_ds_independent_seed_stabilization;
    if (result.independent_frame_board_camera_warmup_requested) {
      result.independent_frame_board_camera_warmup_attempted = true;
      result.independent_frame_board_camera_warmup_rmse_before =
          initial_full_training_stats.pixel_stats.Rmse();
      result.independent_frame_board_camera_warmup_p95_before =
          initial_full_training_stats.pixel_stats.P95();
      const typename PersistentProblemBuilder<GeometryT>::StateSnapshot
          camera_before_independent_warmup = builder.CaptureState();
      const typename PersistentProblemBuilder<GeometryT>::
          IndependentCameraWarmupResult independent_warmup =
              builder.WarmupIndependentFrameBoardCamera(
                  estimator_options
                      .independent_frame_board_camera_warmup_max_iterations,
                  estimator_options.convergence_delta_j,
                  estimator_options.convergence_delta_x, nullptr,
                  model_aware_ds_independent_seed_stabilization
                      ? &result.accepted_keys
                      : nullptr);
      result.independent_frame_board_camera_warmup_pose_count =
          independent_warmup.pose_count;
      result.independent_frame_board_camera_warmup_point_count =
          independent_warmup.point_count;
      result.independent_frame_board_camera_warmup_iterations =
          independent_warmup.solution.iterations;
      result.independent_frame_board_camera_warmup_objective_start =
          independent_warmup.solution.JStart;
      result.independent_frame_board_camera_warmup_objective_final =
          independent_warmup.solution.JFinal;
      const FullTrainingPoseRefitStats independent_warmup_stats =
          builder.EvaluateFullTrainingPoseRefitPixel(
              &initial_full_training_stats);
      result.independent_frame_board_camera_warmup_rmse_after =
          independent_warmup_stats.pixel_stats.Rmse();
      result.independent_frame_board_camera_warmup_p95_after =
          independent_warmup_stats.pixel_stats.P95();
      std::string health_reason;
      result.independent_frame_board_camera_warmup_health_pass =
          CheckFullTrainingPoseRefitHealthGate(
              initial_full_training_stats, initial_full_training_stats,
              independent_warmup_stats, estimator_options, &health_reason);
      result.independent_frame_board_camera_warmup_success =
          independent_warmup.initialized && independent_warmup.state_valid &&
          !independent_warmup.solution.linearSolverFailure &&
          std::isfinite(independent_warmup.solution.JStart) &&
          std::isfinite(independent_warmup.solution.JFinal) &&
          independent_warmup.solution.JFinal < independent_warmup.solution.JStart &&
          result.independent_frame_board_camera_warmup_health_pass;

      const bool probe_solver_valid =
          independent_warmup.initialized && independent_warmup.state_valid &&
          !independent_warmup.solution.linearSolverFailure &&
          std::isfinite(independent_warmup.solution.JStart) &&
          std::isfinite(independent_warmup.solution.JFinal) &&
          independent_warmup.solution.JFinal < independent_warmup.solution.JStart;
      InstabilityQuarantineResult instability_quarantine;
      if (probe_solver_valid &&
          !result.independent_frame_board_camera_warmup_health_pass) {
        instability_quarantine = IdentifyUnstableFrameBoards(
            initial_full_training_stats, independent_warmup_stats,
            candidate_pool_bundle.scene_state.reference_board_id,
            estimator_options);
        result.independent_frame_board_camera_warmup_quarantine_reason =
            instability_quarantine.reason;
      }

      if (!result.independent_frame_board_camera_warmup_success &&
          instability_quarantine.within_budget &&
          !instability_quarantine.keys.empty()) {
        result.independent_frame_board_camera_warmup_quarantine_retry_attempted =
            true;
        result.quarantined_keys = instability_quarantine.keys;
        result
            .independent_frame_board_camera_warmup_instability_quarantined_count =
            static_cast<int>(result.quarantined_keys.size());
        builder.RestoreState(camera_before_independent_warmup);
        health_initial_full_training_stats =
            builder.EvaluateFullTrainingPoseRefitPixel(
                nullptr, &result.quarantined_keys);
        const typename PersistentProblemBuilder<GeometryT>::
            IndependentCameraWarmupResult quarantine_retry =
                builder.WarmupIndependentFrameBoardCamera(
                    estimator_options
                        .independent_frame_board_camera_warmup_max_iterations,
                    estimator_options.convergence_delta_j,
                    estimator_options.convergence_delta_x,
                    &result.quarantined_keys);
        const FullTrainingPoseRefitStats quarantine_retry_stats =
            builder.EvaluateFullTrainingPoseRefitPixel(
                &health_initial_full_training_stats, &result.quarantined_keys);
        std::string quarantine_retry_health_reason;
        const bool quarantine_retry_health_pass =
            CheckFullTrainingPoseRefitHealthGate(
                health_initial_full_training_stats,
                health_initial_full_training_stats, quarantine_retry_stats,
                estimator_options, &quarantine_retry_health_reason);
        const bool quarantine_retry_success =
            quarantine_retry.initialized && quarantine_retry.state_valid &&
            !quarantine_retry.solution.linearSolverFailure &&
            std::isfinite(quarantine_retry.solution.JStart) &&
            std::isfinite(quarantine_retry.solution.JFinal) &&
            quarantine_retry.solution.JFinal < quarantine_retry.solution.JStart &&
            quarantine_retry_health_pass;
        result.independent_frame_board_camera_warmup_quarantine_retry_success =
            quarantine_retry_success;
        result.independent_frame_board_camera_warmup_pose_count =
            quarantine_retry.pose_count;
        result.independent_frame_board_camera_warmup_point_count =
            quarantine_retry.point_count;
        result.independent_frame_board_camera_warmup_iterations =
            quarantine_retry.solution.iterations;
        result.independent_frame_board_camera_warmup_objective_start =
            quarantine_retry.solution.JStart;
        result.independent_frame_board_camera_warmup_objective_final =
            quarantine_retry.solution.JFinal;
        result.independent_frame_board_camera_warmup_rmse_before =
            health_initial_full_training_stats.pixel_stats.Rmse();
        result.independent_frame_board_camera_warmup_rmse_after =
            quarantine_retry_stats.pixel_stats.Rmse();
        result.independent_frame_board_camera_warmup_p95_before =
            health_initial_full_training_stats.pixel_stats.P95();
        result.independent_frame_board_camera_warmup_p95_after =
            quarantine_retry_stats.pixel_stats.P95();
        result.independent_frame_board_camera_warmup_health_pass =
            quarantine_retry_health_pass;
        result.independent_frame_board_camera_warmup_success =
            quarantine_retry_success;
        if (quarantine_retry_success) {
          result.independent_frame_board_camera_warmup_committed = true;
          committed_full_training_stats = quarantine_retry_stats;
          result.warnings.push_back(
              instability_quarantine.reason + " retry=committed");
        } else {
          builder.RestoreState(camera_before_independent_warmup);
          std::ostringstream reason;
          reason << "independent frame-board camera warmup quarantine retry "
                    "rolled back"
                 << " probe_health_reason=" << health_reason
                 << " quarantine=" << instability_quarantine.reason
                 << " retry_health_reason=" << quarantine_retry_health_reason
                 << " linear_solver_failure="
                 << (quarantine_retry.solution.linearSolverFailure ? 1 : 0)
                 << " JStart=" << quarantine_retry.solution.JStart
                 << " JFinal=" << quarantine_retry.solution.JFinal;
          result.independent_frame_board_camera_warmup_rollback_reason =
              reason.str();
          result.warnings.push_back(reason.str());
          result.quarantined_keys.clear();
          result
              .independent_frame_board_camera_warmup_instability_quarantined_count =
              0;
          health_initial_full_training_stats = initial_full_training_stats;
          committed_full_training_stats = initial_full_training_stats;
        }
      } else if (result.independent_frame_board_camera_warmup_success) {
        result.independent_frame_board_camera_warmup_committed = true;
        committed_full_training_stats = independent_warmup_stats;
      } else {
        builder.RestoreState(camera_before_independent_warmup);
        std::ostringstream reason;
        reason << "independent frame-board camera warmup rolled back";
        if (!independent_warmup.failure_reason.empty()) {
          reason << " state_reason=" << independent_warmup.failure_reason;
        }
        if (!health_reason.empty()) {
          reason << " health_reason=" << health_reason;
        }
        reason << " linear_solver_failure="
               << (independent_warmup.solution.linearSolverFailure ? 1 : 0)
               << " JStart=" << independent_warmup.solution.JStart
               << " JFinal=" << independent_warmup.solution.JFinal;
        result.independent_frame_board_camera_warmup_rollback_reason =
            reason.str();
        result.warnings.push_back(reason.str());
        committed_full_training_stats = initial_full_training_stats;
      }

      constexpr double kMaxIndependentBoardPoseRefitRmsePx = 25.0;
      const std::set<FrameBoardKey> pre_quarantine_seed_keys =
          result.accepted_keys;
      const std::set<FrameBoardKey> supported_seed_keys =
          builder.CollectIndependentPoseSupportedKeys(
              result.accepted_keys, kMaxIndependentBoardPoseRefitRmsePx);
      std::set<FrameBoardKey> retained_seed_keys = supported_seed_keys;
      for (const FrameBoardKey& key : result.quarantined_keys) {
        retained_seed_keys.erase(key);
      }
      result.independent_frame_board_camera_warmup_seed_quarantined_count =
          static_cast<int>(result.accepted_keys.size() -
                           retained_seed_keys.size());
      result.accepted_keys = retained_seed_keys;
      for (Stage5IncrementalBackendBatchInput& batch :
           effective_candidate_batches) {
        for (const FrameBoardKey& key : result.quarantined_keys) {
          batch.frame_board_keys.erase(key);
        }
      }
      effective_candidate_batches.erase(
          std::remove_if(
              effective_candidate_batches.begin(),
              effective_candidate_batches.end(),
              [](const Stage5IncrementalBackendBatchInput& batch) {
                return batch.frame_board_keys.empty();
              }),
          effective_candidate_batches.end());
      std::map<int, std::size_t> candidate_batch_index_by_frame;
      for (std::size_t index = 0; index < effective_candidate_batches.size();
           ++index) {
        candidate_batch_index_by_frame[
            effective_candidate_batches[index].frame_index] = index;
      }
      for (const FrameBoardKey& key : pre_quarantine_seed_keys) {
        if (retained_seed_keys.count(key) > 0 ||
            result.quarantined_keys.count(key) > 0) {
          continue;
        }
        const auto batch_it = candidate_batch_index_by_frame.find(key.first);
        if (batch_it != candidate_batch_index_by_frame.end()) {
          effective_candidate_batches[batch_it->second]
              .frame_board_keys.insert(key);
        } else {
          Stage5IncrementalBackendBatchInput batch;
          batch.frame_index = key.first;
          batch.frame_board_keys.insert(key);
          candidate_batch_index_by_frame[key.first] =
              effective_candidate_batches.size();
          effective_candidate_batches.push_back(batch);
        }
      }
      result.candidate_batch_count =
          static_cast<int>(effective_candidate_batches.size());
      fixed_forced_schedule =
          !effective_candidate_batches.empty() &&
          std::all_of(
              effective_candidate_batches.begin(),
              effective_candidate_batches.end(),
              [](const Stage5IncrementalBackendBatchInput& input) {
                return input.force;
              });
      if (result.accepted_keys.empty()) {
        result.failure_reason =
            "independent camera warmup quarantine removed every seed observation";
        return result;
      }
    }

    // The persistent seed is a state-less information linearization.  It must
    // represent the already curated stable views, rather than promote a
    // single exceptionally low-error frame into the health baseline for every
    // later candidate.  No seed solve is performed below, so retaining the
    // full set cannot move the initializer camera, frame poses, or layout.
    if (estimator_options.model_aware_progressive_seed) {
      const std::set<FrameBoardKey> source_seed_keys = result.accepted_keys;
      std::map<int, std::set<FrameBoardKey> > source_keys_by_frame;
      for (const FrameBoardKey& key : source_seed_keys) {
        source_keys_by_frame[key.first].insert(key);
      }
      if (source_keys_by_frame.empty()) {
        result.failure_reason =
            "model-aware progressive seed has no selected observations";
        return result;
      }

      const int reference_board_id =
          candidate_pool_bundle.scene_state.reference_board_id;
      // The progressive seed establishes Fisher information, not a joint
      // multi-board residual objective. Keep only the reference-board Outer4
      // observation for each seed frame: its pose defines T_camera_reference
      // directly and does not inherit a potentially imperfect fixed layout.
      // The selected seed frames themselves remain out of the candidate pool
      // so their auxiliary-board residuals cannot re-enter with a frozen pose.
      std::set<FrameBoardKey> information_seed_keys;
      for (const auto& entry : source_keys_by_frame) {
        const FrameBoardKey reference_key(entry.first, reference_board_id);
        if (entry.second.count(reference_key) != 0) {
          information_seed_keys.insert(reference_key);
        }
      }
      if (!information_seed_keys.empty()) {
        result.accepted_keys = information_seed_keys;
      } else {
        result.warnings.push_back(
            "model-aware progressive seed has no reference-board observations; "
            "retaining the selected seed observations");
      }
      // Materialize the existing scene poses without running a solve so the
      // anchor is ranked by its actual shared-layout Outer4 residual.  Point
      // count alone can pick a large but geometrically inconsistent frame.
      const boost::shared_ptr<CalibrationBatch> anchor_audit_batch =
          builder.BuildBatch(source_seed_keys, true,
                             CameraOptimizationPhase::kSeedFixedIntrinsics,
                             nullptr, nullptr, 1.0,
                             KbDistortionReleaseState(), true);
      (void)anchor_audit_batch;
      int anchor_frame_index = -1;
      int anchor_has_reference = -1;
      int anchor_outer_healthy = -1;
      double anchor_outer_p95 = std::numeric_limits<double>::infinity();
      double anchor_outer_rmse = std::numeric_limits<double>::infinity();
      int anchor_point_count = -1;
      int anchor_board_count = -1;
      for (const auto& entry : source_keys_by_frame) {
        const int frame_index = entry.first;
        const int has_reference =
            entry.second.count(FrameBoardKey(frame_index, reference_board_id))
                ? 1
                : 0;
        const ResidualStats outer_stats = builder.EvaluateAccepted(
            entry.second, SelectionResidualMetric::kPixel);
        const int outer_healthy =
            outer_stats.outer_count >= 4 &&
            outer_stats.invalid_outer_projection_count == 0 &&
            std::isfinite(outer_stats.OuterRmse()) &&
            std::isfinite(outer_stats.OuterP95())
                ? 1
                : 0;
        const double outer_p95 = outer_healthy
                                     ? outer_stats.OuterP95()
                                     : std::numeric_limits<double>::infinity();
        const double outer_rmse =
            outer_healthy ? outer_stats.OuterRmse()
                          : std::numeric_limits<double>::infinity();
        const int point_count = CountBatchPoints(
            candidate_pool_bundle.measurement_dataset, entry.second);
        const int board_count = static_cast<int>(entry.second.size());
        if (anchor_frame_index < 0 || has_reference > anchor_has_reference ||
            (has_reference == anchor_has_reference &&
             outer_healthy > anchor_outer_healthy) ||
            (has_reference == anchor_has_reference &&
             outer_healthy == anchor_outer_healthy &&
             outer_p95 < anchor_outer_p95) ||
            (has_reference == anchor_has_reference &&
             outer_healthy == anchor_outer_healthy &&
             outer_p95 == anchor_outer_p95 &&
             outer_rmse < anchor_outer_rmse) ||
            (has_reference == anchor_has_reference &&
             outer_healthy == anchor_outer_healthy &&
             outer_p95 == anchor_outer_p95 &&
             outer_rmse == anchor_outer_rmse &&
             point_count > anchor_point_count) ||
            (has_reference == anchor_has_reference &&
             outer_healthy == anchor_outer_healthy &&
             outer_p95 == anchor_outer_p95 &&
             outer_rmse == anchor_outer_rmse &&
             point_count == anchor_point_count &&
             board_count > anchor_board_count) ||
            (has_reference == anchor_has_reference &&
             outer_healthy == anchor_outer_healthy &&
             outer_p95 == anchor_outer_p95 &&
             outer_rmse == anchor_outer_rmse &&
             point_count == anchor_point_count &&
             board_count == anchor_board_count &&
             frame_index < anchor_frame_index)) {
          anchor_frame_index = frame_index;
          anchor_has_reference = has_reference;
          anchor_outer_healthy = outer_healthy;
          anchor_outer_p95 = outer_p95;
          anchor_outer_rmse = outer_rmse;
          anchor_point_count = point_count;
          anchor_board_count = board_count;
        }
      }

      result.progressive_seed_enabled = true;
      result.progressive_seed_source_frame_count =
          static_cast<int>(source_keys_by_frame.size());
      result.progressive_seed_moved_frame_count = 0;
      result.progressive_seed_anchor_frame_index = anchor_frame_index;
      const JointMeasurementFrameResult* anchor_frame = FindMeasurementFrame(
          candidate_pool_bundle.measurement_dataset, anchor_frame_index);
      result.progressive_seed_anchor_frame_label =
          anchor_frame != nullptr && !anchor_frame->frame_label.empty()
              ? anchor_frame->frame_label
              : std::to_string(anchor_frame_index);

      // Baseline and candidate keys are normally disjoint.  Remove any
      // overlap defensively so a residual can never be added twice. Every
      // source seed frame stays out of traversal, including its non-reference
      // boards, because its frame pose is frozen after Fisher construction.
      for (Stage5IncrementalBackendBatchInput& batch :
           effective_candidate_batches) {
        for (const auto& source_entry : source_keys_by_frame) {
          if (source_entry.first != batch.frame_index) {
            continue;
          }
          batch.frame_board_keys.clear();
          break;
        }
      }
      effective_candidate_batches.erase(
          std::remove_if(
              effective_candidate_batches.begin(), effective_candidate_batches.end(),
              [](const Stage5IncrementalBackendBatchInput& batch) {
                return batch.frame_board_keys.empty();
              }),
          effective_candidate_batches.end());
      result.candidate_batch_count =
          static_cast<int>(effective_candidate_batches.size());

      std::ostringstream stream;
      stream << "model-aware progressive seed anchor_frame="
             << result.progressive_seed_anchor_frame_label
             << " source_frames=" << result.progressive_seed_source_frame_count
             << " retained_seed_frames=" << result.progressive_seed_source_frame_count
             << " reference_connected=" << anchor_has_reference
             << " outer_healthy=" << anchor_outer_healthy
             << " outer_rmse=" << anchor_outer_rmse
             << " outer_p95=" << anchor_outer_p95
             << " point_count=" << anchor_point_count;
      result.warnings.push_back(stream.str());
    }

    // The initializer owns the incoming camera state.  The persistent seed
    // only establishes the estimator on the selected observations; it must
    // not make an unvalidated global camera step before the existing
    // frame-level health gate can inspect it.
    const std::set<FrameBoardKey> seed_keys = result.accepted_keys;
    result.seed_board_observation_count = static_cast<int>(seed_keys.size());
    std::set<int> seed_frames;
    for (const FrameBoardKey& key : seed_keys) {
      seed_frames.insert(key.first);
    }
    result.seed_frame_count = static_cast<int>(seed_frames.size());
    result.seed_point_count = CountBatchPoints(
        candidate_pool_bundle.measurement_dataset, seed_keys);
    const typename PersistentProblemBuilder<GeometryT>::StateSnapshot seed_state =
        builder.CaptureState();
    result.seed_intrinsics_warmup_attempted =
        estimator_options.optimize_seed_intrinsics &&
        !estimator_options.single_board_dense_grid_profile;
    if (result.seed_intrinsics_warmup_attempted) {
      const FullTrainingPoseRefitStats full_training_stats_before_seed_warmup =
          committed_full_training_stats;
      const aslam::backend::SolutionReturnValue seed_warmup =
          builder.WarmupSeedIntrinsics(
              seed_keys, estimator_options.max_iterations,
              estimator_options.convergence_delta_j,
              estimator_options.convergence_delta_x);
      result.seed_intrinsics_warmup_iterations = seed_warmup.iterations;
      result.seed_intrinsics_warmup_objective_start = seed_warmup.JStart;
      result.seed_intrinsics_warmup_objective_final = seed_warmup.JFinal;
      result.seed_intrinsics_warmup_last_delta_j = seed_warmup.dJFinal;
      result.seed_intrinsics_warmup_last_delta_x = seed_warmup.dXFinal;
      result.seed_intrinsics_warmup_converged_by_relative_objective =
          ConvergedByBearingRelativeObjective(
              estimator_options.residual_model, seed_warmup,
              estimator_options.max_iterations);
      const bool seed_warmup_hit_ceiling =
          seed_warmup.iterations >= estimator_options.max_iterations;
      std::string seed_warmup_state_reason;
      result.seed_intrinsics_warmup_success =
          !seed_warmup.linearSolverFailure &&
          (!seed_warmup_hit_ceiling ||
           result.seed_intrinsics_warmup_converged_by_relative_objective) &&
          std::isfinite(seed_warmup.JStart) &&
          std::isfinite(seed_warmup.JFinal) &&
          seed_warmup.JFinal <= seed_warmup.JStart &&
          builder.CurrentStateFinite(&seed_warmup_state_reason);
      const FullTrainingPoseRefitStats warmup_full_training_stats =
          builder.EvaluateFullTrainingPoseRefitPixel(
              &committed_full_training_stats, &result.quarantined_keys);
      std::string warmup_full_training_reason;
      result.seed_intrinsics_warmup_full_training_health_pass =
          fixed_forced_schedule ||
          CheckFullTrainingPoseRefitHealthGate(
              health_initial_full_training_stats,
              full_training_stats_before_seed_warmup,
              warmup_full_training_stats, estimator_options,
              &warmup_full_training_reason);
      result.seed_intrinsics_warmup_success =
          result.seed_intrinsics_warmup_success &&
          result.seed_intrinsics_warmup_full_training_health_pass;
      if (!result.seed_intrinsics_warmup_success) {
        builder.RestoreState(seed_state);
        std::ostringstream stream;
        stream << "Independent seed intrinsics warm-up was rolled back"
               << " iterations=" << seed_warmup.iterations
               << " JStart=" << seed_warmup.JStart
               << " JFinal=" << seed_warmup.JFinal
               << " dJFinal=" << seed_warmup.dJFinal
               << " dXFinal=" << seed_warmup.dXFinal
               << " linear_solver_failure="
               << (seed_warmup.linearSolverFailure ? 1 : 0);
        if (!seed_warmup_state_reason.empty()) {
          stream << " state_reason=" << seed_warmup_state_reason;
        }
        if (!warmup_full_training_reason.empty()) {
          stream << " full_training_reason="
                 << warmup_full_training_reason;
        }
        result.warnings.push_back(stream.str());
        committed_full_training_stats =
            full_training_stats_before_seed_warmup;
      } else {
        committed_full_training_stats = warmup_full_training_stats;
      }
    } else {
      result.seed_intrinsics_warmup_success = true;
    }
    typename PersistentProblemBuilder<GeometryT>::ResidualConstructionCounts
        seed_residual_counts;
    // The model-aware path must establish camera information on the already
    // curated seed, before evaluating any candidate frame.  Letting the first
    // traversal-order candidate activate the camera makes its admission
    // privileged and can seed the estimator from an unhealthy view.
    const bool model_aware_active_camera_seed =
        estimator_options.normalize_information_gain_by_board_observation;
    if (model_aware_active_camera_seed &&
        estimator_options.align_model_aware_seed_layout_outer &&
        !estimator_options.model_aware_progressive_seed) {
      const typename PersistentProblemBuilder<GeometryT>::StateSnapshot
          seed_layout_state = builder.CaptureState();
      const aslam::backend::SolutionReturnValue seed_layout_alignment =
          builder.AlignSeedLayoutOuter(
              seed_keys, estimator_options.max_iterations,
              estimator_options.convergence_delta_j,
              estimator_options.convergence_delta_x);
      std::string seed_layout_state_reason;
      const bool seed_layout_success =
          !seed_layout_alignment.linearSolverFailure &&
          std::isfinite(seed_layout_alignment.JStart) &&
          std::isfinite(seed_layout_alignment.JFinal) &&
          seed_layout_alignment.JFinal <= seed_layout_alignment.JStart &&
          builder.CurrentStateFinite(&seed_layout_state_reason);
      if (!seed_layout_success) {
        builder.RestoreState(seed_layout_state);
        std::ostringstream stream;
        stream << "model-aware Outer4 seed-layout alignment rolled back"
               << " iterations=" << seed_layout_alignment.iterations
               << " JStart=" << seed_layout_alignment.JStart
               << " JFinal=" << seed_layout_alignment.JFinal;
        if (!seed_layout_state_reason.empty()) {
          stream << " state_reason=" << seed_layout_state_reason;
        }
        result.warnings.push_back(stream.str());
      } else {
        std::ostringstream stream;
        stream << "model-aware Outer4 seed-layout alignment accepted"
               << " iterations=" << seed_layout_alignment.iterations
               << " JStart=" << seed_layout_alignment.JStart
               << " JFinal=" << seed_layout_alignment.JFinal;
        result.warnings.push_back(stream.str());
      }
    }
    const CameraOptimizationPhase seed_camera_phase =
        model_aware_active_camera_seed
            ? (estimator_options.model_aware_progressive_seed
                   ? CameraOptimizationPhase::kSeedActiveCameraFixedFramePoses
                   : CameraOptimizationPhase::kSeedActiveCamera)
            : CameraOptimizationPhase::kSeedFixedIntrinsics;
    // The model-aware seed establishes camera information at the current
    // shared-layout state.  Preserve the validated baseline behavior and use
    // every valid frozen measurement in this seed; candidate batches and the
    // final backend continue to use the normal mixed Outer4/internal
    // objective.  The seed does not change the layout or frame poses.
    const bool seed_outer_only = false;
    boost::shared_ptr<CalibrationBatch> seed_batch = builder.BuildBatch(
        seed_keys, true, seed_camera_phase, &seed_state,
        &seed_residual_counts,
        1.0, KbDistortionReleaseState(), seed_outer_only);
    const auto append_seed_shared_layout_audit =
        [&](const char* stage) {
          const ResidualStats stats = builder.EvaluateAccepted(
              seed_keys, SelectionResidualMetric::kPixel);
          std::ostringstream stream;
          stream << "persistent seed shared-layout pixel audit"
                 << " stage=" << stage
                 << " outer_rmse=" << stats.OuterRmse()
                 << " outer_p95=" << stats.OuterP95()
                 << " invalid_outer="
                 << stats.invalid_outer_projection_count
                 << " internal_rmse=" << stats.InternalRmse()
                 << " internal_p95=" << stats.InternalP95()
                 << " invalid_internal="
                 << stats.invalid_internal_projection_count;
          result.warnings.push_back(stream.str());
        };
    append_seed_shared_layout_audit("before_forced_seed");
    result.seed_outer_only_residuals = seed_outer_only;
    std::string seed_contract_reason;
    if (!ResidualConstructionMatchesMode(
            estimator_options.residual_model,
            estimator_options.pixel_residual_weight,
            estimator_options.chordal_residual_weight,
            seed_residual_counts.image_plane_residual_count,
            seed_residual_counts.angular_residual_count,
            seed_residual_counts.chordal_residual_count,
            &seed_contract_reason)) {
      result.failure_reason = seed_contract_reason;
      return result;
    }
    result.image_plane_residual_count +=
        seed_residual_counts.image_plane_residual_count;
	    result.angular_residual_count +=
	        seed_residual_counts.angular_residual_count;
	    result.chordal_residual_count +=
	        seed_residual_counts.chordal_residual_count;
	    result.hybrid_angular_selected_count +=
	        seed_residual_counts.hybrid_angular_selected_count;
	    result.hybrid_chordal_selected_count +=
	        seed_residual_counts.hybrid_chordal_selected_count;
	    result.angular_observation_geometry_failure_count +=
	        seed_residual_counts.angular_observation_geometry_failure_count;
    const int whitening_success_before_seed =
        result.angular_local_whitening_success_count;
    result.angular_local_whitening_success_count +=
        seed_residual_counts.angular_local_whitening_success_count;
    result.angular_local_whitening_failure_count +=
        seed_residual_counts.angular_local_whitening_failure_count;
    result.angular_local_whitening_clamped_count +=
        seed_residual_counts.angular_local_whitening_clamped_count;
    result.angular_local_whitening_sigma_sum_rad +=
        seed_residual_counts.angular_local_whitening_sigma_sum_rad;
    result.angular_local_whitening_weight_sum +=
        seed_residual_counts.angular_local_whitening_weight_sum;
    if (seed_residual_counts.angular_local_whitening_success_count > 0) {
      result.angular_local_whitening_sigma_min_rad =
          whitening_success_before_seed == 0
              ? seed_residual_counts.angular_local_whitening_sigma_min_rad
              : std::min(
                    result.angular_local_whitening_sigma_min_rad,
                    seed_residual_counts.angular_local_whitening_sigma_min_rad);
      result.angular_local_whitening_sigma_max_rad = std::max(
          result.angular_local_whitening_sigma_max_rad,
          seed_residual_counts.angular_local_whitening_sigma_max_rad);
      result.angular_local_whitening_weight_min =
          whitening_success_before_seed == 0
              ? seed_residual_counts.angular_local_whitening_weight_min
              : std::min(
                    result.angular_local_whitening_weight_min,
                    seed_residual_counts.angular_local_whitening_weight_min);
      result.angular_local_whitening_weight_max = std::max(
          result.angular_local_whitening_weight_max,
          seed_residual_counts.angular_local_whitening_weight_max);
    }
    const aslam::calibration::IncrementalEstimator::ReturnValue seed_ret =
        model_aware_active_camera_seed
            ? estimator.addBatchAtCurrentState(seed_batch)
            : estimator.addBatch(seed_batch, true);
    result.seed_batch_count = seed_ret.batchAccepted ? 1 : 0;
    if (!seed_ret.batchAccepted) {
      result.failure_reason = "forced seed batch was not accepted";
      return result;
    }
    if (model_aware_active_camera_seed &&
        estimator_options.model_aware_progressive_seed) {
      // addBatchAtCurrentState has already formed the camera information
      // baseline.  From this point on the retained seed is an anchored
      // observation set, not a growing pose optimization problem.
      builder.FreezeFramePoses(seed_keys);
      result.warnings.push_back(
          "model-aware retained seed frame poses frozen after information "
          "baseline construction");
    }
    append_seed_shared_layout_audit(
        model_aware_active_camera_seed ? "after_information_seed"
                                       : "after_forced_seed");
    // The fixed-intrinsics seed preserves the initializer camera, so its
    // existing independent-pose refit remains a valid reference state.
    const FullTrainingPoseRefitStats seed_batch_full_training_stats =
        builder.EvaluateFullTrainingPoseRefitPixel(
            &committed_full_training_stats,
            &result.quarantined_keys);
    std::string seed_batch_full_training_reason;
    if (!fixed_forced_schedule &&
        !CheckFullTrainingPoseRefitHealthGate(
            health_initial_full_training_stats, committed_full_training_stats,
            seed_batch_full_training_stats, estimator_options,
            &seed_batch_full_training_reason)) {
      builder.RestoreState(seed_state);
      result.failure_reason =
          "forced seed batch violated full-training pose-refit health: " +
          seed_batch_full_training_reason;
      return result;
    }
    committed_full_training_stats = seed_batch_full_training_stats;
    result.seed_information_rank = static_cast<int>(seed_ret.rankTheta);
    result.seed_information_rank_deficiency =
        static_cast<int>(seed_ret.rankThetaDeficiency);
    result.seed_information_baseline_valid =
        seed_ret.rankTheta >= 0 && seed_ret.rankThetaDeficiency >= 0 &&
        seed_ret.rankTheta + seed_ret.rankThetaDeficiency > 0;
    bool camera_information_activation_pending = false;
    if (!result.seed_information_baseline_valid) {
      if (model_aware_active_camera_seed) {
        std::ostringstream stream;
        stream << "model-aware active seed did not establish a valid camera "
                  "information baseline: rankTheta="
               << seed_ret.rankTheta
               << " rankThetaDeficiency=" << seed_ret.rankThetaDeficiency;
        result.failure_reason = stream.str();
        return result;
      }
      if (!estimator_options.optimize_seed_intrinsics &&
          estimator_options.optimize_candidate_intrinsics) {
        camera_information_activation_pending = true;
        result.warnings.push_back(
            "Persistent seed kept camera intrinsics fixed; the first valid "
            "candidate batch will establish the active-camera information "
            "baseline while releasing intrinsics.");
      } else {
        std::ostringstream stream;
        stream << "persistent seed did not establish a valid active-camera "
                  "information baseline: rankTheta="
               << seed_ret.rankTheta
               << " rankThetaDeficiency=" << seed_ret.rankThetaDeficiency
               << " optimize_candidate_intrinsics="
               << (estimator_options.optimize_candidate_intrinsics ? 1 : 0);
        result.failure_reason = stream.str();
        return result;
      }
    }
    result.seed_information_group_dim =
        result.seed_information_baseline_valid
            ? static_cast<int>(seed_ret.rankTheta +
                               seed_ret.rankThetaDeficiency)
            : -1;
    const Eigen::VectorXd& seed_scaled_singular_values =
        seed_ret.singularValuesScaled.size() > 0
            ? seed_ret.singularValuesScaled
            : seed_ret.singularValues;
    double min_singular_value = std::numeric_limits<double>::infinity();
    double max_singular_value = 0.0;
    for (Eigen::Index index = 0;
         index < seed_scaled_singular_values.size(); ++index) {
      const double value = std::abs(seed_scaled_singular_values[index]);
      if (!std::isfinite(value) || value <= 0.0) {
        continue;
      }
      min_singular_value = std::min(min_singular_value, value);
      max_singular_value = std::max(max_singular_value, value);
    }
    if (std::isfinite(min_singular_value) && max_singular_value > 0.0) {
      result.seed_information_scaled_min_singular_value = min_singular_value;
      result.seed_information_scaled_max_singular_value = max_singular_value;
      result.seed_information_scaled_condition_number =
          max_singular_value / min_singular_value;
    }
    if (active_camera_family == "ds-none" &&
        seed_ret.sigma2Theta.rows() == 6 &&
        seed_ret.sigma2Theta.cols() == 6) {
      const double cu_variance = seed_ret.sigma2Theta(4, 4);
      const double cv_variance = seed_ret.sigma2Theta(5, 5);
      if (std::isfinite(cu_variance) && cu_variance >= 0.0) {
        result.seed_information_ds_cu_stddev_px = std::sqrt(cu_variance);
      }
      if (std::isfinite(cv_variance) && cv_variance >= 0.0) {
        result.seed_information_ds_cv_stddev_px = std::sqrt(cv_variance);
      }
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

    if (seed_state_restored) {
      result.failure_reason =
          "persistent seed state became invalid after forced seed batch";
      if (!seed_state_invalid_reason.empty()) {
        result.failure_reason += ": " + seed_state_invalid_reason;
      }
      return result;
    }
    // Candidate guards compare against the committed post-warmup seed. The
    // incoming state remains available separately for warmup rollback.
    const typename PersistentProblemBuilder<GeometryT>::StateSnapshot
        guard_reference_state = builder.CaptureState();
    builder.SetKbDistortionGuardReference(guard_reference_state);
  result.training_robust_checkpoint_selection_enabled =
      estimator_options.training_robust_checkpoint_selection;
  result.square_pixel_focal_prior_enabled =
      estimator_options.square_pixel_focal_prior;
    typename PersistentProblemBuilder<GeometryT>::StateSnapshot
        best_training_checkpoint_state = guard_reference_state;
    std::set<FrameBoardKey> best_training_checkpoint_keys =
        result.accepted_keys;
    std::set<FrameBoardKey> best_training_checkpoint_quarantined_keys =
        result.quarantined_keys;
    FullTrainingPoseRefitStats best_training_checkpoint_full_stats =
        committed_full_training_stats;
    TrainingRobustCheckpointStats best_training_checkpoint_stats =
        SummarizeTrainingRobustCheckpoint(
            committed_full_training_stats,
            candidate_pool_bundle.scene_state);
    int best_training_checkpoint_attempt_order = -1;
    int best_training_checkpoint_accepted_batch_count = 0;
    int current_rank = result.seed_information_rank;
    const ResidualStats seed_metric_stats =
        builder.EvaluateAccepted(seed_keys, selection_metric);
    const ResidualStats seed_pixel_stats =
        builder.EvaluateAccepted(seed_keys, SelectionResidualMetric::kPixel);
    result.seed_acceptance_metric_rmse = seed_metric_stats.Rmse();
    result.seed_acceptance_metric_p95 = seed_metric_stats.P95();
    {
      std::ostringstream stream;
      stream << "persistent seed shared-layout pixel audit"
             << " outer_rmse=" << seed_pixel_stats.OuterRmse()
             << " outer_p95=" << seed_pixel_stats.OuterP95()
             << " invalid_outer="
             << seed_pixel_stats.invalid_outer_projection_count
             << " internal_rmse=" << seed_pixel_stats.InternalRmse()
             << " internal_p95=" << seed_pixel_stats.InternalP95()
             << " invalid_internal="
             << seed_pixel_stats.invalid_internal_projection_count;
      result.warnings.push_back(stream.str());
    }
    if (selection_metric != SelectionResidualMetric::kPixel) {
      result.residual_health_threshold_metric =
          SelectionMetricUsesAngularHealth(selection_metric)
              ? AdaptiveAngularHealthThreshold(estimator_options,
                                               seed_metric_stats,
                                               ResidualStats())
              : AdaptivePixelEquivalentHealthThreshold(seed_metric_stats,
                                                       ResidualStats());
      std::ostringstream stream;
      stream << "Stage5 persistent selection uses "
             << result.selection_metric_name
             << " health gates with adaptive threshold="
             << result.residual_health_threshold_metric
             << " " << result.selection_metric_unit
             << " from seed_rmse=" << result.seed_acceptance_metric_rmse
             << " seed_p95=" << result.seed_acceptance_metric_p95
             << "; pixel trial residual is not used as the residual-aware "
                "acceptance health metric.";
      result.warnings.push_back(stream.str());
    }
    std::set<int> candidate_board_ids;
    for (const FrameBoardKey& key : result.accepted_keys) {
      candidate_board_ids.insert(key.second);
    }
    for (const Stage5IncrementalBackendBatchInput& input :
         effective_candidate_batches) {
      for (const FrameBoardKey& key : input.frame_board_keys) {
        candidate_board_ids.insert(key.second);
      }
    }
    const int unique_board_count =
        std::max(1, static_cast<int>(candidate_board_ids.size()));
    const int min_accepted_for_saturation =
        estimator_options.adaptive_saturation_min_accepted_batches > 0
            ? estimator_options.adaptive_saturation_min_accepted_batches
            : std::max(4, std::min(12, result.seed_frame_count / 2));
    const int nonproductive_limit =
        estimator_options.adaptive_saturation_nonproductive_batch_limit > 0
            ? estimator_options.adaptive_saturation_nonproductive_batch_limit
            : std::max(6, std::min(16, 2 * unique_board_count));
    std::vector<double> batch_ordering_scores;
    batch_ordering_scores.reserve(effective_candidate_batches.size());
    for (const Stage5IncrementalBackendBatchInput& input :
         effective_candidate_batches) {
      if (std::isfinite(input.ordering_score)) {
        batch_ordering_scores.push_back(input.ordering_score);
      }
    }
    double tail_ordering_score_threshold =
        std::numeric_limits<double>::infinity();
    if (!batch_ordering_scores.empty()) {
      std::sort(batch_ordering_scores.begin(), batch_ordering_scores.end());
      tail_ordering_score_threshold =
          batch_ordering_scores[batch_ordering_scores.size() / 2u];
    }
    result.adaptive_saturation_min_accepted_batches =
        min_accepted_for_saturation;
    result.adaptive_saturation_nonproductive_batch_limit =
        nonproductive_limit;
    result.adaptive_saturation_tail_ordering_score_threshold =
        tail_ordering_score_threshold;
    int consecutive_nonproductive_batches = 0;
    double cumulative_accepted_information_gain = 0.0;
    // A forced candidate is only a one-shot bridge from the information seed
    // into normal incremental admission. Retrying that exceptional path after
    // every rejection bypasses the estimator's regular save/reject semantics
    // for the entire traversal.
    bool model_aware_seed_extension_attempted = false;
    const auto rejection_reason_code = [](const std::string& reason) {
      if (reason.find("full_training_pose_refit_health") !=
          std::string::npos) {
        return std::string("full_training_pose_refit_health_gate");
      }
      if (reason.find("optimizer_nonconvergence") != std::string::npos) {
        return std::string("incremental_optimizer_nonconvergence");
      }
      if (reason.find("information_activation") != std::string::npos) {
        return std::string("camera_information_activation");
      }
      if (reason.find("trust_region") != std::string::npos) {
        return std::string("camera_trust_region");
      }
      if (reason.find("residual_health") != std::string::npos ||
          reason.find("split_residual") != std::string::npos ||
          reason.find("pixel_safety") != std::string::npos) {
        return std::string("residual_health_gate");
      }
      if (reason.find("ray_curve") != std::string::npos) {
        return std::string("ray_curve_health_gate");
      }
      if (reason.find("objective") != std::string::npos) {
        return std::string("objective_gate");
      }
      return std::string("other");
    };

    for (const Stage5IncrementalBackendBatchInput& raw_input :
         effective_candidate_batches) {
      Stage5IncrementalBackendBatchInput input = raw_input;
      const bool camera_information_activation_batch =
          camera_information_activation_pending;
      if (camera_information_activation_batch) {
        input.force = true;
      }
      if (selection_metric != SelectionResidualMetric::kPixel) {
        const ResidualStats candidate_pre_stats =
            builder.EvaluateAccepted(input.frame_board_keys, selection_metric);
        input.residual_health_threshold_metric =
            SelectionMetricUsesAngularHealth(selection_metric)
                ? AdaptiveAngularHealthThreshold(estimator_options,
                                                 seed_metric_stats,
                                                 candidate_pre_stats)
                : AdaptivePixelEquivalentHealthThreshold(seed_metric_stats,
                                                         candidate_pre_stats);
      }
      result.adaptive_saturation_next_ordering_score = input.ordering_score;
      const bool next_is_ordering_tail =
          !std::isfinite(tail_ordering_score_threshold) ||
          input.ordering_score <= tail_ordering_score_threshold;
      const bool next_is_protected =
          input.force || input.has_intrinsics_diversity_anchor;
      if (estimator_options.adaptive_saturation_stop_enabled &&
          result.accepted_batch_count >= min_accepted_for_saturation &&
          consecutive_nonproductive_batches >= nonproductive_limit &&
          next_is_ordering_tail &&
          !next_is_protected) {
        result.adaptive_saturation_stop_hit = true;
        result.adaptive_saturation_consecutive_nonproductive_batches =
            consecutive_nonproductive_batches;
        std::ostringstream stream;
        stream << "adaptive_information_saturation consecutive_nonproductive="
               << consecutive_nonproductive_batches
               << " limit=" << nonproductive_limit
               << " accepted_batches=" << result.accepted_batch_count
               << " min_accepted=" << min_accepted_for_saturation
               << " next_ordering_score=" << input.ordering_score
               << " tail_ordering_score_threshold="
               << tail_ordering_score_threshold;
        result.adaptive_saturation_stop_reason = stream.str();
        result.warnings.push_back(result.adaptive_saturation_stop_reason);
        break;
      }
      Stage5IncrementalBackendBatchResult batch_result;
      batch_result.attempted = true;
      batch_result.force = input.force;
      batch_result.frame_index = input.frame_index;
      batch_result.frame_label = input.frame_label;
      batch_result.max_trial_rmse = input.max_trial_rmse;
      batch_result.residual_health_threshold_px =
          input.residual_health_threshold_px;
      batch_result.residual_health_threshold_metric =
          input.residual_health_threshold_metric;
      batch_result.batch_board_observation_count =
          static_cast<int>(input.frame_board_keys.size());
      batch_result.batch_point_count = CountBatchPoints(
          candidate_pool_bundle.measurement_dataset, input.frame_board_keys);
      batch_result.information_gain_threshold =
          estimator_options.information_gain_threshold;
      batch_result.rank_gain_threshold =
          estimator_options.rank_gain_threshold;
      batch_result.rank_theta_before = current_rank;
      batch_result.full_training_pixel_rmse_before =
          committed_full_training_stats.pixel_stats.Rmse();
      batch_result.full_training_pixel_p95_before =
          committed_full_training_stats.pixel_stats.P95();
      batch_result.full_training_pose_success_rate_before =
          committed_full_training_stats.PoseSuccessRate();
      batch_result.full_training_pose_success_count_before =
          committed_full_training_stats.pose_success_count;
      batch_result.full_training_pose_total_count =
          committed_full_training_stats.pose_total_count;
      batch_result.full_training_invalid_projection_count_before =
          committed_full_training_stats.pixel_stats.invalid_projection_count;
      FillCameraDiagnostics(builder.CurrentCamera(),
                            &batch_result.camera_xi_before,
                            &batch_result.camera_alpha_before,
                            &batch_result.camera_fu_before,
                            &batch_result.camera_fv_before,
                            &batch_result.camera_cu_before,
                            &batch_result.camera_cv_before);
      FillDistortionDiagnostics(builder.CurrentCamera(),
                                &batch_result.camera_k1_before,
                                &batch_result.camera_k2_before,
                                &batch_result.camera_k3_before,
                                &batch_result.camera_k4_before);
      const bool camera_information_full_rank =
          result.seed_information_baseline_valid && current_rank >= 0 &&
          result.seed_information_group_dim > 0 &&
          current_rank >= result.seed_information_group_dim;
      KbDistortionReleaseState kb_release_state;
      if (result.kb_distortion_guard_enabled) {
        kb_release_state.k3_released = camera_information_full_rank &&
            result.accepted_batch_count >= 3 &&
            cumulative_accepted_information_gain >= 0.75;
        kb_release_state.k4_released = camera_information_full_rank &&
            result.accepted_batch_count >= 6 &&
            cumulative_accepted_information_gain >= 1.50;
      }
      batch_result.kb_k3_released = kb_release_state.k3_released;
      batch_result.kb_k4_released = kb_release_state.k4_released;
      ++result.attempted_batch_count;

      if (input.frame_board_keys.empty() ||
          batch_result.batch_point_count <= 0) {
        batch_result.batch_accepted = false;
        batch_result.reject_reason = "empty_batch";
        batch_result.committed_or_rollback = "rollback";
        ++result.rejected_batch_count;
        result.batch_results.push_back(batch_result);
        ++consecutive_nonproductive_batches;
        result.adaptive_saturation_consecutive_nonproductive_batches =
            consecutive_nonproductive_batches;
        continue;
      }

      // Keep the committed scene state separate from the state that the
      // incremental estimator snapshots internally after the pose prefit.
      // rejectBatch() restores the latter; a rejected candidate must then
      // restore the former so the prefit cannot leak into the next batch.
      const typename PersistentProblemBuilder<GeometryT>::StateSnapshot
          committed_batch_state = builder.CaptureState();
      const ResidualStats committed_before_prefit_pixel_stats =
          builder.EvaluateAccepted(result.accepted_keys,
                                   SelectionResidualMetric::kPixel);
      // A model-aware frame is scored with its pose marginalized.  Align that
      // frame against the committed camera and fixed shared layout before the
      // joint candidate solve, regardless of whether the final residual is
      // pixel or bearing based.  Previously pixel-mode candidates skipped
      // this step and started by moving a new frame pose and all camera
      // parameters together, which made otherwise usable difficult views hit
      // the iteration ceiling before the health gates could evaluate them.
      const bool require_candidate_pose_prefit =
          estimator_options.residual_model != ResidualModel::ImagePlane ||
          (estimator_options.normalize_information_gain_by_board_observation &&
           estimator_options.use_model_aware_candidate_pose_prefit);
      if (require_candidate_pose_prefit) {
        constexpr int kPosePrefitMaximumIterations = 100;
        batch_result.pose_prefit_attempted = true;
        const aslam::backend::SolutionReturnValue pose_prefit =
            builder.PrefitCandidateFramePoses(
                input.frame_board_keys, kPosePrefitMaximumIterations,
                estimator_options.convergence_delta_j,
                estimator_options.convergence_delta_x);
        batch_result.pose_prefit_iterations = pose_prefit.iterations;
        batch_result.pose_prefit_objective_start = pose_prefit.JStart;
        batch_result.pose_prefit_objective_final = pose_prefit.JFinal;
        batch_result.pose_prefit_last_delta_j = pose_prefit.dJFinal;
        batch_result.pose_prefit_last_delta_x = pose_prefit.dXFinal;
        batch_result.pose_prefit_success =
            !pose_prefit.linearSolverFailure &&
            pose_prefit.iterations < kPosePrefitMaximumIterations &&
            std::isfinite(pose_prefit.JStart) &&
            std::isfinite(pose_prefit.JFinal) &&
            pose_prefit.JFinal <= pose_prefit.JStart;
        if (!batch_result.pose_prefit_success) {
          builder.RestoreState(committed_batch_state);
          std::ostringstream stream;
          stream << "candidate_pose_prefit_failed iterations="
                 << pose_prefit.iterations
                 << " JStart=" << pose_prefit.JStart
                 << " JFinal=" << pose_prefit.JFinal
                 << " dJFinal=" << pose_prefit.dJFinal
                 << " dXFinal=" << pose_prefit.dXFinal
                 << " linear_solver_failure="
                 << (pose_prefit.linearSolverFailure ? 1 : 0);
          batch_result.reject_reason = stream.str();
          batch_result.committed_or_rollback = "rollback";
          ++result.rejected_batch_count;
          result.batch_results.push_back(batch_result);
          ++consecutive_nonproductive_batches;
          continue;
        }
      }
      const typename PersistentProblemBuilder<GeometryT>::StateSnapshot
          batch_state = builder.CaptureState();
      const ResidualStats committed_before_stats =
          builder.EvaluateAccepted(result.accepted_keys, selection_metric);
      const ResidualStats committed_before_pixel_stats =
          builder.EvaluateAccepted(result.accepted_keys,
                                   SelectionResidualMetric::kPixel);
      batch_result.rmse_before = committed_before_stats.Rmse();
      batch_result.outer_rmse_before = committed_before_stats.OuterRmse();
      batch_result.internal_rmse_before =
          committed_before_stats.InternalRmse();
      batch_result.acceptance_metric_name =
          SelectionMetricName(selection_metric);
      batch_result.acceptance_metric_unit =
          SelectionMetricUnit(selection_metric);
      batch_result.acceptance_metric_threshold =
          SelectionHealthThreshold(estimator_options, input, selection_metric);
      batch_result.acceptance_metric_before = committed_before_stats.Rmse();
      batch_result.pixel_rmse_before = committed_before_pixel_stats.Rmse();
      batch_result.pixel_p95_before = committed_before_pixel_stats.P95();
      aslam::calibration::IncrementalEstimator::ReturnValue ret;
      boost::shared_ptr<CalibrationBatch> batch;
      bool incremental_accepted = false;
      std::string state_invalid_reason;
      double trust_region_violation_ratio = 1.0;
      double active_anchor_weight_scale = 1.0;
      bool add_batch_exception = false;
      bool positive_information_seed_extension = false;
      const bool try_model_aware_seed_extension =
          estimator_options.normalize_information_gain_by_board_observation &&
          !estimator_options.information_gain_threshold_explicit &&
          result.accepted_batch_count == 0 &&
          !model_aware_seed_extension_attempted;

      auto populate_from_return_value =
          [&](const aslam::calibration::IncrementalEstimator::ReturnValue&
                  active_ret,
              bool active_incremental_accepted, int retry_count,
              double anchor_weight_scale) {
            ret = active_ret;
            state_invalid_reason.clear();
            trust_region_violation_ratio = 1.0;
            batch_result.batch_accepted = active_incremental_accepted;
            batch_result.num_iterations =
                static_cast<int>(active_ret.numIterations);
            batch_result.information_gain = active_ret.informationGain;
            batch_result.information_gain_normalization_count =
                std::max(1, batch_result.batch_board_observation_count);
            batch_result.normalized_information_gain =
                estimator_options.normalize_information_gain_by_board_observation
                    ? active_ret.informationGain /
                          static_cast<double>(
                              batch_result
                                  .information_gain_normalization_count)
                    : active_ret.informationGain;
            batch_result.rank_psi_after =
                static_cast<int>(active_ret.rankPsi);
            batch_result.rank_psi_deficiency_after =
                static_cast<int>(active_ret.rankPsiDeficiency);
            batch_result.rank_theta_after =
                static_cast<int>(active_ret.rankTheta);
            batch_result.rank_theta_deficiency_after =
                static_cast<int>(active_ret.rankThetaDeficiency);
            batch_result.svd_tolerance = active_ret.svdTolerance;
            batch_result.qr_tolerance = active_ret.qrTolerance;
            batch_result.objective_start = active_ret.JStart;
            batch_result.objective_final = active_ret.JFinal;
            batch_result.objective_last_delta_j = active_ret.dJFinal;
            batch_result.state_last_delta_x = active_ret.dXFinal;
            batch_result.linear_solver_failure =
                active_ret.linearSolverFailure;
            batch_result.elapsed_time_seconds = active_ret.elapsedTime;
            batch_result.objective_finite =
                std::isfinite(active_ret.JStart) &&
                std::isfinite(active_ret.JFinal);
            batch_result.objective_decreased =
                batch_result.objective_finite &&
                active_ret.JFinal < active_ret.JStart;
		            batch_result.objective_gate_pass =
		                input.force ||
		                batch_result.objective_decreased;
            const bool rank_gain =
                batch_result.rank_theta_before >= 0 &&
                (static_cast<double>(active_ret.rankTheta) -
                 static_cast<double>(batch_result.rank_theta_before)) >
                    estimator_options.rank_gain_threshold;
            // A model-aware seed already establishes the active camera
            // information state.  If its fixed absolute threshold rejects
            // every candidate, the incremental estimator cannot start even
            // when the best information-ranked view is healthy and adds
            // positive camera information. Permit only that first healthy
            // extension when the caller did not request an explicit cutoff;
            // all later candidates retain the normal threshold and health
            // gates still run before a state can be committed.
            positive_information_seed_extension =
                estimator_options.normalize_information_gain_by_board_observation &&
                !estimator_options.information_gain_threshold_explicit &&
                result.accepted_batch_count == 0 &&
                std::isfinite(batch_result.information_gain) &&
                batch_result.information_gain > 0.0;
            batch_result.information_gate_pass =
                batch_result.information_gain >
                    estimator_options.information_gain_threshold ||
                rank_gain || positive_information_seed_extension;
            const bool residual_score_gate_pass =
                !estimator_options.single_board_dense_grid_profile &&
                SelectionMetricUsesResidualAwareScoreGate(selection_metric) &&
                std::isfinite(input.ordering_score) &&
                input.ordering_score >= tail_ordering_score_threshold;
            batch_result.residual_health_pass =
		                input.force ||
		                selection_metric != SelectionResidualMetric::kPixel ||
		                input.residual_health_threshold_px <= 0.0 ||
		                input.max_trial_rmse <= input.residual_health_threshold_px;
            FillCameraDiagnostics(builder.CurrentCamera(),
                                  &batch_result.camera_xi_after,
                                  &batch_result.camera_alpha_after,
                                  &batch_result.camera_fu_after,
                                  &batch_result.camera_fv_after,
                                  &batch_result.camera_cu_after,
                                  &batch_result.camera_cv_after);
            FillDistortionDiagnostics(builder.CurrentCamera(),
                                      &batch_result.camera_k1_after,
                                      &batch_result.camera_k2_after,
                                      &batch_result.camera_k3_after,
                                      &batch_result.camera_k4_after);
            const bool reached_iteration_ceiling =
                batch_result.last_solver_pass_iterations >=
                std::max(1, estimator_options.max_iterations);
            batch_result.converged_by_relative_objective =
                ConvergedByBearingRelativeObjective(
                    estimator_options.residual_model, active_ret,
                    estimator_options.max_iterations);
            batch_result.optimization_success =
                !active_ret.linearSolverFailure &&
                (!reached_iteration_ceiling ||
                 batch_result.converged_by_relative_objective ||
                 batch_result.converged_by_camera_step);
            batch_result.solution_valid =
                batch_result.optimization_success &&
                (input.force ||
                 (batch_result.objective_finite &&
		                 batch_result.objective_gate_pass &&
		                 (batch_result.information_gate_pass ||
		                  residual_score_gate_pass) &&
		                 batch_result.residual_health_pass));
            batch_result.trust_region_pass = true;
            batch_result.trust_region_retry_count = retry_count;
            batch_result.trust_region_backtracking_used = retry_count > 0;
            batch_result.trust_region_anchor_weight_scale =
                anchor_weight_scale;
            batch_result.solution_valid =
                batch_result.solution_valid &&
                builder.CurrentStateFinite(&state_invalid_reason);
            if (batch_result.solution_valid) {
              batch_result.trust_region_pass =
                  builder.CandidateCameraStepWithinTrustRegion(
                      batch_state, &state_invalid_reason,
                      &trust_region_violation_ratio);
              batch_result.solution_valid =
                  batch_result.solution_valid &&
                  batch_result.trust_region_pass;
            }
            batch_result.trust_region_violation_ratio =
                trust_region_violation_ratio;
            batch_result.batch_accepted =
                active_incremental_accepted && batch_result.solution_valid;
            batch_result.committed_or_rollback =
                batch_result.batch_accepted ? "committed" : "rollback";
          };

      auto run_candidate_attempt = [&](double anchor_weight_scale,
                                       int retry_count) {
        typename PersistentProblemBuilder<GeometryT>::ResidualConstructionCounts
            residual_counts;
        batch = builder.BuildBatch(input.frame_board_keys, true,
                                   CameraOptimizationPhase::kCandidateTrustRegion,
                                   &batch_state, &residual_counts,
                                   anchor_weight_scale, kb_release_state);
        batch_result.image_plane_residual_count =
            residual_counts.image_plane_residual_count;
	        batch_result.angular_residual_count =
	            residual_counts.angular_residual_count;
	        batch_result.chordal_residual_count =
	            residual_counts.chordal_residual_count;
	        batch_result.hybrid_angular_selected_count =
	            residual_counts.hybrid_angular_selected_count;
	        batch_result.hybrid_chordal_selected_count =
	            residual_counts.hybrid_chordal_selected_count;
	        batch_result.angular_observation_geometry_failure_count =
	            residual_counts.angular_observation_geometry_failure_count;
        result.image_plane_residual_count +=
            residual_counts.image_plane_residual_count;
	        result.angular_residual_count +=
	            residual_counts.angular_residual_count;
	        result.chordal_residual_count +=
	            residual_counts.chordal_residual_count;
	        result.hybrid_angular_selected_count +=
	            residual_counts.hybrid_angular_selected_count;
	        result.hybrid_chordal_selected_count +=
	            residual_counts.hybrid_chordal_selected_count;
	        result.angular_observation_geometry_failure_count +=
	            residual_counts.angular_observation_geometry_failure_count;
        const int whitening_success_before_batch =
            result.angular_local_whitening_success_count;
        result.angular_local_whitening_success_count +=
            residual_counts.angular_local_whitening_success_count;
        result.angular_local_whitening_failure_count +=
            residual_counts.angular_local_whitening_failure_count;
        result.angular_local_whitening_clamped_count +=
            residual_counts.angular_local_whitening_clamped_count;
        result.angular_local_whitening_sigma_sum_rad +=
            residual_counts.angular_local_whitening_sigma_sum_rad;
        result.angular_local_whitening_weight_sum +=
            residual_counts.angular_local_whitening_weight_sum;
        if (residual_counts.angular_local_whitening_success_count > 0) {
          result.angular_local_whitening_sigma_min_rad =
              whitening_success_before_batch == 0
                  ? residual_counts.angular_local_whitening_sigma_min_rad
                  : std::min(
                        result.angular_local_whitening_sigma_min_rad,
                        residual_counts.angular_local_whitening_sigma_min_rad);
          result.angular_local_whitening_sigma_max_rad = std::max(
              result.angular_local_whitening_sigma_max_rad,
              residual_counts.angular_local_whitening_sigma_max_rad);
          result.angular_local_whitening_weight_min =
              whitening_success_before_batch == 0
                  ? residual_counts.angular_local_whitening_weight_min
                  : std::min(
                        result.angular_local_whitening_weight_min,
                        residual_counts.angular_local_whitening_weight_min);
          result.angular_local_whitening_weight_max = std::max(
              result.angular_local_whitening_weight_max,
              residual_counts.angular_local_whitening_weight_max);
        }
        std::string residual_contract_reason;
        if (!ResidualConstructionMatchesMode(
                estimator_options.residual_model,
                estimator_options.pixel_residual_weight,
                estimator_options.chordal_residual_weight,
                residual_counts.image_plane_residual_count,
                residual_counts.angular_residual_count,
                residual_counts.chordal_residual_count,
                &residual_contract_reason)) {
          builder.RestoreState(committed_batch_state);
          incremental_accepted = false;
          add_batch_exception = true;
          batch_result.batch_accepted = false;
          batch_result.optimization_success = false;
          batch_result.objective_finite = false;
          batch_result.solution_valid = false;
          batch_result.reject_reason = residual_contract_reason;
          state_invalid_reason = residual_contract_reason;
          batch_result.committed_or_rollback = "rollback";
          return false;
        }
        try {
          if (try_model_aware_seed_extension) {
            model_aware_seed_extension_attempted = true;
          }
          OuterBootstrapCameraIntrinsics pass_camera_before =
              builder.CurrentCamera();
          aslam::calibration::IncrementalEstimator::ReturnValue attempt_ret =
              // Kalibr force-adds its initial target view before its normal
              // information-gain traversal.  Our model-aware seed already
              // owns the initial camera state, so use the same one-shot
              // mechanism only while no candidate has been accepted.  The
              // post-solve positive-information and all health gates below
              // still decide whether this candidate is committed.
              estimator.addBatch(batch,
                                 input.force ||
                                     try_model_aware_seed_extension);
          CameraStepConvergence camera_step = EvaluateCameraStepConvergence(
              pass_camera_before, builder.CurrentCamera());
          const double initial_objective = attempt_ret.JStart;
          const double information_gain = attempt_ret.informationGain;
          double total_elapsed_time = attempt_ret.elapsedTime;
          std::size_t total_iterations = attempt_ret.numIterations;
          std::size_t last_pass_iterations = attempt_ret.numIterations;
          int continuation_round_count = 0;
          while (attempt_ret.batchAccepted &&
                 (estimator_options.residual_model != ResidualModel::ImagePlane ||
                  camera_information_activation_batch) &&
                 last_pass_iterations >= static_cast<std::size_t>(
                     estimator_options.max_iterations) &&
                 !ConvergedByBearingRelativeObjective(
                     estimator_options.residual_model, attempt_ret,
                     estimator_options.max_iterations) &&
                 !camera_step.converged &&
                 continuation_round_count <
                     estimator_options.max_continuation_rounds) {
            pass_camera_before = builder.CurrentCamera();
            const aslam::calibration::IncrementalEstimator::ReturnValue
                continuation_ret = estimator.reoptimize();
            camera_step = EvaluateCameraStepConvergence(
                pass_camera_before, builder.CurrentCamera());
            ++continuation_round_count;
            last_pass_iterations = continuation_ret.numIterations;
            total_iterations += continuation_ret.numIterations;
            total_elapsed_time += continuation_ret.elapsedTime;
            attempt_ret = continuation_ret;
            attempt_ret.batchAccepted = true;
            attempt_ret.informationGain = information_gain;
            attempt_ret.JStart = initial_objective;
            attempt_ret.numIterations = total_iterations;
            attempt_ret.elapsedTime = total_elapsed_time;
            if (attempt_ret.linearSolverFailure ||
                !std::isfinite(attempt_ret.JFinal)) {
              break;
            }
          }
          batch_result.last_solver_pass_iterations =
              static_cast<int>(last_pass_iterations);
          batch_result.converged_by_camera_step = camera_step.converged;
          batch_result.last_camera_shape_step = camera_step.shape_step;
          batch_result.last_camera_focal_relative_step =
              camera_step.focal_relative_step;
          batch_result.last_camera_principal_step_px =
              camera_step.principal_step_px;
          batch_result.continuation_round_count = continuation_round_count;
          batch_result.continuation_guard_hit =
              continuation_round_count >=
                  estimator_options.max_continuation_rounds &&
              last_pass_iterations >= static_cast<std::size_t>(
                  estimator_options.max_iterations) &&
              !ConvergedByBearingRelativeObjective(
                  estimator_options.residual_model, attempt_ret,
                  estimator_options.max_iterations) &&
              !camera_step.converged;
          incremental_accepted = attempt_ret.batchAccepted;
          populate_from_return_value(attempt_ret, incremental_accepted,
                                     retry_count, anchor_weight_scale);
          return true;
        } catch (const std::exception& exception) {
          if (batch) {
            estimator.rejectBatch(batch);
          }
          builder.RestoreState(committed_batch_state);
          incremental_accepted = false;
          add_batch_exception = true;
          batch_result.batch_accepted = false;
          batch_result.optimization_success = false;
          batch_result.objective_finite = false;
          batch_result.objective_decreased = false;
          batch_result.solution_valid = false;
          batch_result.trust_region_pass = false;
          batch_result.trust_region_retry_count = retry_count;
          batch_result.trust_region_backtracking_used = retry_count > 0;
          batch_result.trust_region_anchor_weight_scale =
              anchor_weight_scale;
          batch_result.reject_reason =
              std::string("incremental_add_batch_exception: ") +
              exception.what();
          state_invalid_reason = batch_result.reject_reason;
          batch_result.committed_or_rollback = "rollback";
          return false;
        }
      };

      if (!run_candidate_attempt(active_anchor_weight_scale, 0)) {
        ++result.rejected_batch_count;
        result.batch_results.push_back(batch_result);
        continue;
      }

      if (camera_information_activation_batch &&
          (ret.rankTheta < 0 || ret.rankThetaDeficiency < 0 ||
           ret.rankTheta + ret.rankThetaDeficiency <= 0)) {
        if (incremental_accepted) {
          estimator.rejectBatch(batch);
        }
        builder.RestoreState(committed_batch_state);
        batch_result.batch_accepted = false;
        batch_result.solution_valid = false;
        batch_result.reject_reason =
            "camera_information_activation_failed";
        batch_result.committed_or_rollback = "rollback";
        ++result.rejected_batch_count;
        result.batch_results.push_back(batch_result);
        continue;
      }

      while (!batch_result.batch_accepted &&
             incremental_accepted &&
             !add_batch_exception &&
             !batch_result.trust_region_pass &&
             state_invalid_reason.find("camera_trust_region_gate") == 0 &&
             batch_result.trust_region_retry_count <
                 kMaxTrustRegionBacktrackingRetries &&
             estimator_options.optimize_candidate_intrinsics &&
             estimator_options.use_candidate_intrinsics_anchor_prior &&
             active_anchor_weight_scale <
                 kMaxTrustRegionAnchorWeightScale) {
        estimator.rejectBatch(batch);
        incremental_accepted = false;
        active_anchor_weight_scale = NextTrustRegionAnchorWeightScale(
            active_anchor_weight_scale, trust_region_violation_ratio);
        const int retry_count = batch_result.trust_region_retry_count + 1;
        if (!run_candidate_attempt(active_anchor_weight_scale, retry_count)) {
          break;
        }
      }

      auto make_reject_reason = [&]() {
        if (!state_invalid_reason.empty()) {
          return state_invalid_reason;
        }
        if (!batch_result.objective_finite) {
          return std::string("incremental_nonfinite_objective");
        }
        if (!batch_result.optimization_success) {
          std::ostringstream stream;
          stream << "incremental_optimizer_nonconvergence iterations="
                 << batch_result.num_iterations
                 << " last_pass_iterations="
                 << batch_result.last_solver_pass_iterations
                 << " max_iterations_per_pass="
                 << estimator_options.max_iterations
                 << " continuation_rounds="
                 << batch_result.continuation_round_count
                 << " continuation_guard_hit="
                 << (batch_result.continuation_guard_hit ? 1 : 0)
                 << " JStart=" << batch_result.objective_start
                 << " JFinal=" << batch_result.objective_final;
          return stream.str();
        }
        if (!batch_result.objective_gate_pass) {
          std::ostringstream stream;
          stream << "incremental_objective_increase_gate JStart="
                 << ret.JStart << " JFinal=" << ret.JFinal;
          return stream.str();
        }
		        if (SelectionMetricUsesResidualAwareScoreGate(selection_metric) &&
		            !batch_result.information_gate_pass &&
		            !(std::isfinite(input.ordering_score) &&
		              input.ordering_score >= tail_ordering_score_threshold)) {
          std::ostringstream stream;
          stream << "residual_aware_selection_score_gate metric="
                 << batch_result.acceptance_metric_name
                 << " ordering_score="
                 << input.ordering_score
                 << " threshold=" << tail_ordering_score_threshold
                 << " raw_info=" << batch_result.information_gain
                 << " normalized_info="
                 << batch_result.normalized_information_gain;
          return stream.str();
        }
		        if (!batch_result.information_gate_pass &&
		            !SelectionMetricUsesResidualAwareScoreGate(selection_metric)) {
          std::ostringstream stream;
          stream << "incremental_information_gain_gate normalized_info="
                 << batch_result.normalized_information_gain
                 << " raw_info=" << batch_result.information_gain
                 << " norm_count="
                 << batch_result.information_gain_normalization_count
                 << " gate_scope=frame_raw threshold="
                 << estimator_options.information_gain_threshold;
          return stream.str();
        }
        if (!batch_result.residual_health_pass) {
          std::ostringstream stream;
          stream << "incremental_residual_health_gate metric="
                 << batch_result.acceptance_metric_name
                 << " max_trial_rmse="
                 << input.max_trial_rmse
                 << " threshold="
                 << batch_result.acceptance_metric_threshold
                 << " unit=" << batch_result.acceptance_metric_unit;
          return stream.str();
        }
        if (!batch_result.split_residual_health_pass) {
          return std::string("incremental_split_residual_health_gate");
        }
        if (!batch_result.pixel_safety_gate_pass) {
          return std::string("incremental_bearing_pixel_safety_gate");
        }
        if (!batch_result.full_training_pose_refit_health_pass) {
          return std::string(
              "incremental_full_training_pose_refit_health_gate");
        }
        if (!batch_result.ray_curve_validity_pass) {
          return std::string("incremental_kb_ray_curve_validity_gate");
        }
        return incremental_accepted
                   ? std::string("incremental_solution_validity_gate")
                   : std::string("incremental_backend_rejected_batch");
      };
      ResidualStats committed_candidate_stats;
      ResidualStats candidate_only_stats;
      bool committed_candidate_stats_ready = false;
      FullTrainingPoseRefitStats candidate_full_training_stats =
          committed_full_training_stats;
      bool candidate_full_training_stats_ready = false;
      if (batch_result.batch_accepted) {
        std::set<FrameBoardKey> candidate_accepted_keys = result.accepted_keys;
        candidate_accepted_keys.insert(input.frame_board_keys.begin(),
                                       input.frame_board_keys.end());
        candidate_only_stats =
            builder.EvaluateAccepted(input.frame_board_keys, selection_metric);
        batch_result.candidate_rmse_after = candidate_only_stats.Rmse();
        batch_result.candidate_outer_rmse_after =
            candidate_only_stats.OuterRmse();
        batch_result.candidate_internal_rmse_after =
            candidate_only_stats.InternalRmse();
        batch_result.candidate_total_p95_after = candidate_only_stats.P95();
        batch_result.candidate_outer_p95_after =
            candidate_only_stats.OuterP95();
        batch_result.candidate_internal_p95_after =
            candidate_only_stats.InternalP95();
        batch_result.acceptance_metric_candidate =
            candidate_only_stats.Rmse();
        batch_result.acceptance_metric_candidate_p95 =
            candidate_only_stats.P95();
        batch_result.acceptance_metric_candidate_outer =
            candidate_only_stats.OuterRmse();
        batch_result.acceptance_metric_candidate_internal =
            candidate_only_stats.InternalRmse();
        committed_candidate_stats =
            builder.EvaluateAccepted(candidate_accepted_keys,
                                     selection_metric);
        committed_candidate_stats_ready = true;
        batch_result.rmse_after = committed_candidate_stats.Rmse();
        batch_result.outer_rmse_after = committed_candidate_stats.OuterRmse();
        batch_result.internal_rmse_after =
            committed_candidate_stats.InternalRmse();
        batch_result.total_p95_after = committed_candidate_stats.P95();
        batch_result.outer_p95_after = committed_candidate_stats.OuterP95();
        batch_result.internal_p95_after =
            committed_candidate_stats.InternalP95();
        batch_result.acceptance_metric_after =
            committed_candidate_stats.Rmse();

        const ResidualStats candidate_only_pixel_stats =
            builder.EvaluateAccepted(input.frame_board_keys,
                                     SelectionResidualMetric::kPixel);
        const ResidualStats committed_existing_pixel_stats =
            builder.EvaluateAccepted(result.accepted_keys,
                                     SelectionResidualMetric::kPixel);
        const ResidualStats committed_candidate_pixel_stats =
            builder.EvaluateAccepted(candidate_accepted_keys,
                                     SelectionResidualMetric::kPixel);
        batch_result.candidate_pixel_rmse_after =
            candidate_only_pixel_stats.Rmse();
        batch_result.candidate_pixel_p95_after =
            candidate_only_pixel_stats.P95();
        batch_result.pixel_rmse_after =
            committed_candidate_pixel_stats.Rmse();
        batch_result.pixel_p95_after =
            committed_candidate_pixel_stats.P95();

        if (batch_result.batch_accepted &&
            estimator_options.single_board_dense_grid_profile &&
            committed_before_stats.total_count > 0) {
          const ResidualStats committed_existing_after_stats =
              builder.EvaluateAccepted(result.accepted_keys,
                                       selection_metric);
          const double existing_rmse_limit = std::max(
              1.005 * committed_before_stats.Rmse(),
              committed_before_stats.Rmse() + 0.002);
          const double existing_p95_limit = std::max(
              1.01 * committed_before_stats.P95(),
              committed_before_stats.P95() + 0.005);
          if (committed_existing_after_stats.Rmse() > existing_rmse_limit ||
              committed_existing_after_stats.P95() > existing_p95_limit) {
            std::ostringstream stream;
            stream << "checkerboard_committed_stability_gate"
                   << " before_rmse=" << committed_before_stats.Rmse()
                   << " existing_after_rmse="
                   << committed_existing_after_stats.Rmse()
                   << " rmse_limit=" << existing_rmse_limit
                   << " before_p95=" << committed_before_stats.P95()
                   << " existing_after_p95="
                   << committed_existing_after_stats.P95()
                   << " p95_limit=" << existing_p95_limit;
            state_invalid_reason = stream.str();
            batch_result.solution_valid = false;
            batch_result.batch_accepted = false;
            batch_result.committed_or_rollback = "rollback";
            batch_result.split_residual_health_pass = false;
            ++result.split_residual_health_rejected_count;
          }
        }

        if (result.bearing_pixel_safety_gate_enabled && !input.force) {
          std::string pixel_safety_reason;
          batch_result.pixel_safety_gate_pass =
              CheckSplitResidualHealthGate(
                  committed_before_pixel_stats,
                  committed_existing_pixel_stats,
                  candidate_only_pixel_stats, estimator_options, input,
                  SelectionResidualMetric::kPixel, &pixel_safety_reason);
          const double seed_rmse_limit = RegressionLimit(
              seed_pixel_stats.Rmse(), 1.35, 0.50);
          const double seed_p95_limit = RegressionLimit(
              seed_pixel_stats.P95(), 1.50, 1.00);
          if (committed_existing_pixel_stats.Rmse() > seed_rmse_limit ||
              committed_existing_pixel_stats.P95() > seed_p95_limit) {
            batch_result.pixel_safety_gate_pass = false;
            std::ostringstream stream;
            stream << "bearing_pixel_safety_global seed_rmse="
                   << seed_pixel_stats.Rmse()
                   << " existing_after_rmse="
                   << committed_existing_pixel_stats.Rmse()
                   << " rmse_limit=" << seed_rmse_limit
                   << " seed_p95=" << seed_pixel_stats.P95()
                   << " existing_after_p95="
                   << committed_existing_pixel_stats.P95()
                   << " p95_limit=" << seed_p95_limit;
            pixel_safety_reason = stream.str();
          }
          if (!batch_result.pixel_safety_gate_pass) {
            state_invalid_reason = pixel_safety_reason;
            batch_result.solution_valid = false;
            batch_result.batch_accepted = false;
            batch_result.committed_or_rollback = "rollback";
            ++result.bearing_pixel_safety_rejected_count;
          }
        }

        if (batch_result.batch_accepted &&
            result.kb_distortion_guard_enabled) {
          const KbRayCurveHealth ray_health =
              builder.EvaluateKbRayCurveHealth(guard_reference_state);
          batch_result.ray_curve_rms_change_deg =
              ray_health.rms_change_deg;
          batch_result.ray_curve_max_change_deg =
              ray_health.max_change_deg;
          batch_result.ray_curve_min_radial_derivative =
              ray_health.min_radial_derivative;
          batch_result.ray_curve_validity_pass = ray_health.valid;
          if (!ray_health.valid) {
            state_invalid_reason = ray_health.failure_reason;
            batch_result.solution_valid = false;
            batch_result.batch_accepted = false;
            batch_result.committed_or_rollback = "rollback";
            ++result.kb_ray_curve_validity_rejected_count;
          }
        }
	        // The model-aware path seeds its camera-information state with a
	        // deliberately diverse multi-board bundle. Its mixed seed+candidate
	        // RMSE is therefore not comparable with a single-candidate absolute
	        // threshold. CheckSplitResidualHealthGate below still enforces the
	        // candidate's own health and protects residuals already committed.
	        if (!estimator_options.normalize_information_gain_by_board_observation &&
	            !input.force &&
            batch_result.acceptance_metric_threshold > 0.0 &&
            batch_result.rmse_after >
                batch_result.acceptance_metric_threshold) {
          std::ostringstream stream;
          stream << "persistent_committed_residual_health_gate metric="
                 << batch_result.acceptance_metric_name
                 << " rmse="
                 << batch_result.rmse_after
                 << " threshold="
                 << batch_result.acceptance_metric_threshold
                 << " unit=" << batch_result.acceptance_metric_unit;
          state_invalid_reason = stream.str();
          batch_result.solution_valid = false;
          batch_result.batch_accepted = false;
          batch_result.committed_or_rollback = "rollback";
          batch_result.residual_health_pass = false;
        }
        if (batch_result.batch_accepted) {
          // Do not compare the mixed seed+candidate residual to the seed
          // residual. A geometrically valid hard frame naturally has a larger
          // standalone error and would always make that mixed average rise.
          // The candidate is checked by its own absolute RMSE/P95 above;
          // this regression gate protects only observations already committed
          // before the candidate was introduced.
          const ResidualStats committed_existing_stats =
              builder.EvaluateAccepted(result.accepted_keys, selection_metric);
          std::string split_residual_reason;
          batch_result.split_residual_health_pass =
              CheckSplitResidualHealthGate(
                  committed_before_stats, committed_existing_stats,
                  candidate_only_stats, estimator_options, input,
                  selection_metric,
                  &split_residual_reason);
          if (!batch_result.split_residual_health_pass) {
            state_invalid_reason = split_residual_reason;
            batch_result.solution_valid = false;
            batch_result.batch_accepted = false;
            batch_result.committed_or_rollback = "rollback";
            ++result.split_residual_health_rejected_count;
          }
        }
        // Camera-information activation is forced only so IncrementalEstimator
        // can establish its first active-intrinsics information baseline. It
        // must still pass the full-dataset independent-pose health gate before
        // the camera update is committed.
        const bool require_full_training_health =
            !input.force || camera_information_activation_batch;
        if (batch_result.batch_accepted && require_full_training_health &&
            result.full_training_pose_refit_health_gate_enabled) {
          candidate_full_training_stats =
              builder.EvaluateFullTrainingPoseRefitPixel(
                  &committed_full_training_stats, &result.quarantined_keys);
          candidate_full_training_stats_ready = true;
          batch_result.full_training_pixel_rmse_after =
              candidate_full_training_stats.pixel_stats.Rmse();
          batch_result.full_training_pixel_p95_after =
              candidate_full_training_stats.pixel_stats.P95();
          batch_result.full_training_pose_success_rate_after =
              candidate_full_training_stats.PoseSuccessRate();
          batch_result.full_training_pose_success_count_after =
              candidate_full_training_stats.pose_success_count;
          batch_result.full_training_invalid_projection_count_after =
              candidate_full_training_stats.pixel_stats
                  .invalid_projection_count;
          std::string full_training_reason;
          // The first active-camera batch completes the camera warmup
          // transaction. Compare it with the pre-warmup baseline rather than
          // requiring it to preserve the temporary independent-pose optimum.
          const FullTrainingPoseRefitStats& full_training_before_reference =
              camera_information_activation_batch
                  ? health_initial_full_training_stats
                  : committed_full_training_stats;
          batch_result.full_training_pose_refit_health_pass =
              CheckFullTrainingPoseRefitHealthGate(
                  health_initial_full_training_stats,
                  full_training_before_reference,
                  candidate_full_training_stats, estimator_options,
                  &full_training_reason);
          if (!batch_result.full_training_pose_refit_health_pass) {
            state_invalid_reason = full_training_reason;
            batch_result.solution_valid = false;
            batch_result.batch_accepted = false;
            batch_result.committed_or_rollback = "rollback";
            ++result.full_training_pose_refit_health_rejected_count;
          }
        }
      }
      if (batch_result.batch_accepted) {
        result.accepted_keys.insert(input.frame_board_keys.begin(),
                                    input.frame_board_keys.end());
        current_rank = static_cast<int>(ret.rankTheta);
        if (camera_information_activation_batch) {
          camera_information_activation_pending = false;
          result.seed_information_rank = static_cast<int>(ret.rankTheta);
          result.seed_information_rank_deficiency =
              static_cast<int>(ret.rankThetaDeficiency);
          result.seed_information_group_dim = static_cast<int>(
              ret.rankTheta + ret.rankThetaDeficiency);
          result.seed_information_baseline_valid = true;
          batch_result.accept_reason = "camera_information_activation";
        }
        if (!committed_candidate_stats_ready) {
          committed_candidate_stats =
              builder.EvaluateAccepted(result.accepted_keys,
                                       selection_metric);
        }
        batch_result.rmse_after = committed_candidate_stats.Rmse();
        batch_result.outer_rmse_after = committed_candidate_stats.OuterRmse();
        batch_result.internal_rmse_after =
            committed_candidate_stats.InternalRmse();
        batch_result.total_p95_after = committed_candidate_stats.P95();
        batch_result.outer_p95_after = committed_candidate_stats.OuterP95();
        batch_result.internal_p95_after =
            committed_candidate_stats.InternalP95();
        batch_result.acceptance_metric_after =
            committed_candidate_stats.Rmse();
	        if (candidate_full_training_stats_ready) {
	          committed_full_training_stats = candidate_full_training_stats;
	        }
	        if (estimator_options.training_robust_checkpoint_selection) {
	          if (!candidate_full_training_stats_ready) {
	            committed_full_training_stats =
	                builder.EvaluateFullTrainingPoseRefitPixel(
	                    &committed_full_training_stats,
	                    &result.quarantined_keys);
	          }
	          const TrainingRobustCheckpointStats checkpoint_stats =
	              SummarizeTrainingRobustCheckpoint(
	                  committed_full_training_stats,
	                  candidate_pool_bundle.scene_state);
	          if (IsBetterTrainingRobustCheckpoint(
	                  checkpoint_stats, best_training_checkpoint_stats)) {
	            best_training_checkpoint_state = builder.CaptureState();
	            best_training_checkpoint_keys = result.accepted_keys;
	            best_training_checkpoint_quarantined_keys =
	                result.quarantined_keys;
	            best_training_checkpoint_full_stats =
	                committed_full_training_stats;
	            best_training_checkpoint_stats = checkpoint_stats;
	            best_training_checkpoint_attempt_order =
	                result.attempted_batch_count - 1;
	            best_training_checkpoint_accepted_batch_count =
	                result.accepted_batch_count + 1;
	          }
	        }
        if (!camera_information_activation_batch) {
          batch_result.accept_reason =
            input.force
                ? "force"
                : (positive_information_seed_extension
                       ? "positive_information_seed_extension"
                : (SelectionMetricUsesResidualAwareScoreGate(selection_metric)
                       ? "residual_aware_metric_score_health"
                       : (batch_result.information_gain >
                                  estimator_options.information_gain_threshold
                              ? "incremental_information_gain"
                              : "incremental_rank_gain")));
        }
        ++result.accepted_batch_count;
        cumulative_accepted_information_gain +=
            std::max(0.0, batch_result.normalized_information_gain);
        consecutive_nonproductive_batches = 0;
      } else {
        const std::string active_reject_reason = make_reject_reason();
        if (incremental_accepted) {
          estimator.rejectBatch(batch);
        }
        builder.RestoreState(committed_batch_state);
        const ResidualStats rollback_pixel_stats =
            builder.EvaluateAccepted(result.accepted_keys,
                                     SelectionResidualMetric::kPixel);
        const double rollback_tolerance_px = std::max(
            0.01,
            0.001 * std::max(1.0,
                             committed_before_prefit_pixel_stats.Rmse()));
        if (!std::isfinite(rollback_pixel_stats.Rmse()) ||
            std::abs(rollback_pixel_stats.Rmse() -
                     committed_before_prefit_pixel_stats.Rmse()) >
                rollback_tolerance_px) {
          std::ostringstream stream;
          stream << "persistent batch rollback state mismatch"
                 << " before_rmse="
                 << committed_before_prefit_pixel_stats.Rmse()
                 << " restored_rmse=" << rollback_pixel_stats.Rmse()
                 << " tolerance=" << rollback_tolerance_px
                 << " frame=" << input.frame_index;
          result.failure_reason = stream.str();
          result.warnings.push_back(result.failure_reason);
          return result;
        }
        batch_result.reject_reason = active_reject_reason;
        ++result.rejection_reason_counts[active_reject_reason];
        ++result.rejection_reason_code_counts[
            rejection_reason_code(active_reject_reason)];
        ++result.rejected_batch_count;
        const bool low_information =
            !input.force &&
            !batch_result.information_gate_pass &&
            !SelectionMetricUsesAngularHealth(selection_metric) &&
            selection_metric != SelectionResidualMetric::kHybridObjective;
        const bool residual_or_validity_reject =
            !batch_result.residual_health_pass ||
            !batch_result.split_residual_health_pass ||
            !batch_result.pixel_safety_gate_pass ||
            !batch_result.full_training_pose_refit_health_pass ||
            !batch_result.ray_curve_validity_pass ||
            !batch_result.trust_region_pass ||
            !batch_result.objective_gate_pass ||
            !batch_result.solution_valid;
        if (low_information || residual_or_validity_reject) {
          ++consecutive_nonproductive_batches;
        } else {
          consecutive_nonproductive_batches = 0;
        }
      }
      result.adaptive_saturation_consecutive_nonproductive_batches =
          consecutive_nonproductive_batches;
      if (batch_result.trust_region_retry_count > 0) {
        ++result.trust_region_backtracking_batch_count;
        result.trust_region_backtracking_attempt_count +=
            batch_result.trust_region_retry_count;
        if (batch_result.batch_accepted) {
          ++result.trust_region_backtracking_accepted_count;
        }
        result.trust_region_backtracking_max_anchor_scale =
            std::max(result.trust_region_backtracking_max_anchor_scale,
                     batch_result.trust_region_anchor_weight_scale);
      }
      if (batch_result.attempted) {
        if (batch_result.num_iterations <= 1) {
          ++result.solver_single_iteration_batch_count;
        }
        if (batch_result.last_solver_pass_iterations >=
            estimator_options.max_iterations) {
          ++result.solver_max_iteration_batch_count;
        }
        if (batch_result.objective_decreased) {
          ++result.solver_objective_decreased_batch_count;
        }
        if (batch_result.converged_by_relative_objective) {
          ++result.solver_relative_objective_converged_batch_count;
        }
        if (batch_result.converged_by_camera_step) {
          ++result.solver_camera_step_converged_batch_count;
        }
        if (batch_result.continuation_round_count > 0) {
          ++result.solver_continuation_batch_count;
          result.solver_continuation_round_count +=
              batch_result.continuation_round_count;
        }
        if (batch_result.continuation_guard_hit) {
          ++result.solver_continuation_guard_hit_count;
        }
      }
      result.batch_results.push_back(batch_result);
    }

    if (estimator_options.training_robust_checkpoint_selection &&
        best_training_checkpoint_stats.usable &&
        best_training_checkpoint_accepted_batch_count <
            result.accepted_batch_count) {
      const int accepted_before_restore = result.accepted_batch_count;
      builder.RestoreState(best_training_checkpoint_state);
      result.accepted_keys = best_training_checkpoint_keys;
      result.quarantined_keys =
          best_training_checkpoint_quarantined_keys;
      committed_full_training_stats = best_training_checkpoint_full_stats;
      result.accepted_batch_count =
          best_training_checkpoint_accepted_batch_count;
      result.training_robust_checkpoint_restored = true;
      result.training_robust_checkpoint_discarded_accepted_batch_count =
          accepted_before_restore - result.accepted_batch_count;
      for (Stage5IncrementalBackendBatchResult& batch_result :
           result.batch_results) {
        if (batch_result.batch_accepted &&
            batch_result.committed_or_rollback == "committed" &&
            batch_result.frame_index >= 0) {
          // attempt order is implicit in batch_results, including rejected
          // entries. Mark accepted states after the selected checkpoint as
          // explored but absent from the returned Persistent state.
          const int attempt_order = static_cast<int>(
              &batch_result - result.batch_results.data());
          if (attempt_order > best_training_checkpoint_attempt_order) {
            batch_result.batch_accepted = false;
            batch_result.committed_or_rollback =
                "training_checkpoint_rollback";
            batch_result.reject_reason =
                "training_robust_checkpoint_later_state_rollback";
          }
        }
      }
      std::ostringstream stream;
      stream << "Persistent training-only robust checkpoint restored"
             << " attempt_order="
             << best_training_checkpoint_attempt_order
             << " accepted_batches="
             << best_training_checkpoint_accepted_batch_count
             << " discarded_later_accepted_batches="
             << result.training_robust_checkpoint_discarded_accepted_batch_count
             << " fold_median_mean_rmse="
             << best_training_checkpoint_stats.fold_median_mean_rmse
             << " fold_median_max_rmse="
             << best_training_checkpoint_stats.fold_median_max_rmse
             << "; no final optimization was run.";
      result.warnings.push_back(stream.str());
    }
    result.training_robust_checkpoint_attempt_order =
        best_training_checkpoint_attempt_order;
    result.training_robust_checkpoint_accepted_batch_count =
        best_training_checkpoint_accepted_batch_count;
    if (best_training_checkpoint_stats.usable) {
      result.training_robust_checkpoint_frame_median_rmse =
          best_training_checkpoint_stats.frame_median_rmse;
      result.training_robust_checkpoint_frame_p90_rmse =
          best_training_checkpoint_stats.frame_p90_rmse;
      result.training_robust_checkpoint_huber15_rmse =
          best_training_checkpoint_stats.huber15_rmse;
      result.training_robust_checkpoint_fold_median_mean_rmse =
          best_training_checkpoint_stats.fold_median_mean_rmse;
      result.training_robust_checkpoint_fold_median_max_rmse =
          best_training_checkpoint_stats.fold_median_max_rmse;
    }

    const typename PersistentProblemBuilder<GeometryT>::StateSnapshot
        final_state = builder.CaptureState();
    for (const auto& initial_board : seed_state.board_poses) {
      const auto final_it = final_state.board_poses.find(initial_board.first);
      if (final_it == final_state.board_poses.end()) {
        continue;
      }
      ++result.board_layout_pose_count;
      const Eigen::Matrix4d& initial_pose = initial_board.second;
      const Eigen::Matrix4d& final_pose = final_it->second;
      result.board_layout_max_matrix_abs_delta = std::max(
          result.board_layout_max_matrix_abs_delta,
          (final_pose - initial_pose).cwiseAbs().maxCoeff());
      result.board_layout_max_translation_delta = std::max(
          result.board_layout_max_translation_delta,
          (final_pose.block<3, 1>(0, 3) -
           initial_pose.block<3, 1>(0, 3)).norm());
      const Eigen::Matrix3d relative_rotation =
          initial_pose.block<3, 3>(0, 0).transpose() *
          final_pose.block<3, 3>(0, 0);
      const double cosine = std::max(
          -1.0, std::min(1.0, 0.5 * (relative_rotation.trace() - 1.0)));
      constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
      result.board_layout_max_rotation_delta_deg = std::max(
          result.board_layout_max_rotation_delta_deg,
          std::acos(cosine) * kRadiansToDegrees);
    }
    const FullTrainingPoseRefitStats final_full_training_stats =
        builder.EvaluateFullTrainingPoseRefitPixel(
            &committed_full_training_stats, &result.quarantined_keys);
    result.final_full_training_pixel_rmse =
        final_full_training_stats.pixel_stats.Rmse();
    result.final_full_training_pixel_p95 =
        final_full_training_stats.pixel_stats.P95();
    result.final_full_training_pose_success_rate =
        final_full_training_stats.PoseSuccessRate();
    result.final_full_training_pose_success_count =
        final_full_training_stats.pose_success_count;
    result.final_full_training_pose_total_count =
        final_full_training_stats.pose_total_count;
    result.final_full_training_invalid_projection_count =
        final_full_training_stats.pixel_stats.invalid_projection_count;
    result.final_full_training_invalid_outer_projection_count =
        final_full_training_stats.pixel_stats.invalid_outer_projection_count;
    result.final_full_training_invalid_internal_projection_count =
        final_full_training_stats.pixel_stats.invalid_internal_projection_count;
    const ResidualStats committed_state_pixel_stats =
        builder.EvaluateAccepted(result.accepted_keys,
                                 SelectionResidualMetric::kPixel);
    result.optimized_scene_state = builder.BuildSceneState();
    result.curated_bundle = BuildCuratedBundle(
        baseline_bundle, candidate_pool_bundle, result.optimized_scene_state,
        result.accepted_keys);
    result.committed_state_pixel_rmse = committed_state_pixel_stats.Rmse();
    result.curated_bundle_pixel_rmse =
        result.curated_bundle.residual_result.overall_image_plane_rmse;
    if (!(result.curated_bundle_pixel_rmse > 0.0) &&
        result.curated_bundle.residual_result.overall_rmse > 0.0) {
      result.curated_bundle_pixel_rmse =
          result.curated_bundle.residual_result.overall_rmse;
    }
    result.curated_bundle_state_consistency_tolerance_px = std::max(
        0.01, 0.001 * std::max(1.0, result.committed_state_pixel_rmse));
    result.curated_bundle_state_consistency_pass =
        std::isfinite(result.committed_state_pixel_rmse) &&
        std::isfinite(result.curated_bundle_pixel_rmse) &&
        std::abs(result.committed_state_pixel_rmse -
                 result.curated_bundle_pixel_rmse) <=
            result.curated_bundle_state_consistency_tolerance_px;
    result.validated_baseline_pixel_rmse =
        baseline_bundle.residual_result.overall_image_plane_rmse;
    if (!(result.validated_baseline_pixel_rmse > 0.0) &&
        baseline_bundle.residual_result.overall_rmse > 0.0) {
      result.validated_baseline_pixel_rmse =
          baseline_bundle.residual_result.overall_rmse;
    }
    if (std::isfinite(result.validated_baseline_pixel_rmse) &&
        result.validated_baseline_pixel_rmse > 0.0) {
      result.curated_bundle_shared_scene_rmse_limit_px = std::max(
          2.0 * result.validated_baseline_pixel_rmse,
          result.validated_baseline_pixel_rmse + 2.0);
      result.curated_bundle_shared_scene_health_pass =
          std::isfinite(result.curated_bundle_pixel_rmse) &&
          result.curated_bundle_pixel_rmse <=
              result.curated_bundle_shared_scene_rmse_limit_px;
    }
    if ((!result.curated_bundle_state_consistency_pass ||
         !result.curated_bundle_shared_scene_health_pass) &&
        baseline_bundle.IsReadyForBackend()) {
      std::ostringstream warning;
      warning << "persistent curated bundle failed final shared-state health; "
                 "using validated baseline fallback"
              << " committed_rmse=" << result.committed_state_pixel_rmse
              << " curated_rmse=" << result.curated_bundle_pixel_rmse
              << " tolerance="
              << result.curated_bundle_state_consistency_tolerance_px
              << " baseline_rmse=" << result.validated_baseline_pixel_rmse
              << " shared_rmse_limit="
              << result.curated_bundle_shared_scene_rmse_limit_px
              << " consistency_pass="
              << (result.curated_bundle_state_consistency_pass ? 1 : 0)
              << " shared_health_pass="
              << (result.curated_bundle_shared_scene_health_pass ? 1 : 0);
      result.warnings.push_back(warning.str());
      result.curated_bundle = baseline_bundle;
      result.optimized_scene_state = baseline_bundle.scene_state;
      result.accepted_keys =
          CollectAcceptedKeys(baseline_bundle.measurement_dataset);
      result.curated_bundle_used_validated_baseline_fallback = true;
      result.final_full_training_pixel_rmse =
          initial_full_training_stats.pixel_stats.Rmse();
      result.final_full_training_pixel_p95 =
          initial_full_training_stats.pixel_stats.P95();
      result.final_full_training_pose_success_rate =
          initial_full_training_stats.PoseSuccessRate();
      result.final_full_training_pose_success_count =
          initial_full_training_stats.pose_success_count;
      result.final_full_training_pose_total_count =
          initial_full_training_stats.pose_total_count;
      result.final_full_training_invalid_projection_count =
          initial_full_training_stats.pixel_stats.invalid_projection_count;
      result.final_full_training_invalid_outer_projection_count =
          initial_full_training_stats.pixel_stats.invalid_outer_projection_count;
      result.final_full_training_invalid_internal_projection_count =
          initial_full_training_stats.pixel_stats.invalid_internal_projection_count;
    }
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
    // Stage5Benchmark reports the persistent fallback reason when a
    // model-aware run cannot continue. Preserve the typed backend exception
    // there as well so model-specific failures are not hidden.
    result.fallback_reason = result.failure_reason;
  }

  const auto time_end = std::chrono::steady_clock::now();
  result.total_elapsed_time_seconds =
      std::chrono::duration<double>(time_end - time_start).count();
  if (!result.success && result.fallback_reason.empty()) {
    result.fallback_reason = result.failure_reason;
  }
  return result;
}

Stage5IncrementalBackendEstimatorResult RunStage5IncrementalBackendEstimator(
    const CalibrationStateBundle& baseline_bundle,
    const CalibrationStateBundle& candidate_pool_bundle,
    const BackendProblemOptions& backend_options,
    const TrialBackendFrameBoardSelectionOptions& selection_options,
    const AslamBackendCalibrationOptions& backend_runner_options,
    const std::vector<Stage5IncrementalBackendBatchInput>& candidate_batches) {
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

  const std::string family =
      baseline_bundle.scene_state.camera.NormalizedFamilyString();
  if (family == "ds-none") {
    return RunStage5IncrementalBackendEstimatorTyped<DsGeometry>(
        baseline_bundle, candidate_pool_bundle, selection_options,
        backend_runner_options, candidate_batches);
  }
  if (family == "eucm-none") {
    return RunStage5IncrementalBackendEstimatorTyped<EucmGeometry>(
        baseline_bundle, candidate_pool_bundle, selection_options,
        backend_runner_options, candidate_batches);
  }
  if (family == "pinhole-equi") {
    return RunStage5IncrementalBackendEstimatorTyped<PinholeEquiGeometry>(
        baseline_bundle, candidate_pool_bundle, selection_options,
        backend_runner_options, candidate_batches);
  }
  if (family == "omni-radtan") {
    return RunStage5IncrementalBackendEstimatorTyped<MeiGeometry>(
        baseline_bundle, candidate_pool_bundle, selection_options,
        backend_runner_options, candidate_batches);
  }
  if (family == "omni-none") {
    return RunStage5IncrementalBackendEstimatorTyped<OmniNoneGeometry>(
        baseline_bundle, candidate_pool_bundle, selection_options,
        backend_runner_options, candidate_batches);
  }
  result.compatible = false;
  result.fallback_reason =
      "persistent incremental estimator unsupported family: " + family;
  result.failure_reason = result.fallback_reason;
  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
