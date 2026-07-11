#include <aslam/cameras/apriltag_internal/Stage5IncrementalBackendEstimator.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <limits>
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

#include <aslam/cameras/apriltag_internal/AngularResidualGeometry.hpp>
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
  kSeedFixedIntrinsics,
  kCandidateTrustRegion,
};

enum class SelectionResidualMetric {
  kPixel,
  kAngularTangent,
  kChordal,
  kHybridObjective,
  kPixelChordalHybridObjective,
};

SelectionResidualMetric SelectionMetricForResidualModel(
    ResidualModel residual_model) {
  switch (residual_model) {
    case ResidualModel::SphereAngular:
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
         metric == SelectionResidualMetric::kChordal ||
         metric == SelectionResidualMetric::kHybridObjective ||
         metric == SelectionResidualMetric::kPixelChordalHybridObjective;
}

bool SelectionMetricIsHybridObjective(SelectionResidualMetric metric) {
  return metric == SelectionResidualMetric::kHybridObjective ||
         metric == SelectionResidualMetric::kPixelChordalHybridObjective;
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

    auto add_finite_difference_camera_jacobian =
        [&](const auto& design_variable_adapter) {
          if (design_variable_adapter == nullptr ||
              !design_variable_adapter->isActive()) {
            return;
          }
          const Eigen::MatrixXd base_parameters =
              design_variable_adapter->getParameters();
          const int dimension = static_cast<int>(base_parameters.size());
          if (dimension <= 0) {
            return;
          }
          Eigen::MatrixXd camera_jacobian(2, dimension);
          camera_jacobian.setZero();
          for (int index = 0; index < dimension; ++index) {
            const Eigen::Index row =
                static_cast<Eigen::Index>(index % base_parameters.rows());
            const Eigen::Index col =
                static_cast<Eigen::Index>(index / base_parameters.rows());
            const double base_value = base_parameters(row, col);
            const double epsilon =
                std::max(1e-7, 1e-6 * std::max(1.0, std::fabs(base_value)));

            Eigen::MatrixXd positive = base_parameters;
            positive(row, col) = base_value + epsilon;
            design_variable_adapter->setParameters(positive);
            AngularObservationGeometry positive_observation;
            AngularPredictionGeometry positive_prediction;
            bool positive_valid = false;
            const Eigen::Vector2d positive_residual =
                ComputeResidual(&positive_observation, &positive_prediction,
                                &positive_valid);

            Eigen::MatrixXd negative = base_parameters;
            negative(row, col) = base_value - epsilon;
            design_variable_adapter->setParameters(negative);
            AngularObservationGeometry negative_observation;
            AngularPredictionGeometry negative_prediction;
            bool negative_valid = false;
            const Eigen::Vector2d negative_residual =
                ComputeResidual(&negative_observation, &negative_prediction,
                                &negative_valid);

            if (positive_valid && negative_valid &&
                positive_residual.allFinite() &&
                negative_residual.allFinite()) {
              camera_jacobian.col(index) =
                  (positive_residual - negative_residual) / (2.0 * epsilon);
            }
          }
          design_variable_adapter->setParameters(base_parameters);
          jacobians.add(design_variable_adapter.get(), camera_jacobian);
        };

    add_finite_difference_camera_jacobian(
        const_cast<CameraDv&>(camera_dv_).projectionDesignVariable());
    add_finite_difference_camera_jacobian(
        const_cast<CameraDv&>(camera_dv_).distortionDesignVariable());
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<2>;

  Eigen::Vector2d ComputeResidual(
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
          *camera_dv_.camera(), observed_image_xy_, observation_geometry);
    }
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    const Eigen::Vector3d point_camera = point_homogeneous.head<3>();
    *valid_projection = observation_valid &&
        ComputePredictionGeometryForCamera(
            *camera_dv_.camera(), point_homogeneous, prediction_geometry);
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

    auto add_finite_difference_camera_jacobian =
        [&](const auto& design_variable_adapter) {
          if (design_variable_adapter == nullptr ||
              !design_variable_adapter->isActive()) {
            return;
          }
          const Eigen::MatrixXd base_parameters =
              design_variable_adapter->getParameters();
          const int dimension = static_cast<int>(base_parameters.size());
          if (dimension <= 0) {
            return;
          }
          Eigen::MatrixXd camera_jacobian(3, dimension);
          camera_jacobian.setZero();
          for (int index = 0; index < dimension; ++index) {
            const Eigen::Index row =
                static_cast<Eigen::Index>(index % base_parameters.rows());
            const Eigen::Index col =
                static_cast<Eigen::Index>(index / base_parameters.rows());
            const double base_value = base_parameters(row, col);
            const double epsilon =
                std::max(1e-7, 1e-6 * std::max(1.0, std::fabs(base_value)));

            Eigen::MatrixXd positive = base_parameters;
            positive(row, col) = base_value + epsilon;
            design_variable_adapter->setParameters(positive);
            AngularObservationGeometry positive_observation;
            AngularPredictionGeometry positive_prediction;
            bool positive_valid = false;
            const Eigen::Matrix<double, 3, 1> positive_residual =
                ComputeResidual(&positive_observation, &positive_prediction,
                                &positive_valid);

            Eigen::MatrixXd negative = base_parameters;
            negative(row, col) = base_value - epsilon;
            design_variable_adapter->setParameters(negative);
            AngularObservationGeometry negative_observation;
            AngularPredictionGeometry negative_prediction;
            bool negative_valid = false;
            const Eigen::Matrix<double, 3, 1> negative_residual =
                ComputeResidual(&negative_observation, &negative_prediction,
                                &negative_valid);

            if (positive_valid && negative_valid &&
                positive_residual.allFinite() &&
                negative_residual.allFinite()) {
              camera_jacobian.col(index) =
                  (positive_residual - negative_residual) / (2.0 * epsilon);
            }
          }
          design_variable_adapter->setParameters(base_parameters);
          jacobians.add(design_variable_adapter.get(), camera_jacobian);
        };

    add_finite_difference_camera_jacobian(
        const_cast<CameraDv&>(camera_dv_).projectionDesignVariable());
    add_finite_difference_camera_jacobian(
        const_cast<CameraDv&>(camera_dv_).distortionDesignVariable());
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<3>;

  Eigen::Matrix<double, 3, 1> ComputeResidual(
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
          *camera_dv_.camera(), observed_image_xy_, observation_geometry);
    }
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    const Eigen::Vector3d point_camera = point_homogeneous.head<3>();
    *valid_projection = observation_valid &&
        ComputePredictionGeometryForCamera(
            *camera_dv_.camera(), point_homogeneous, prediction_geometry);
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

    auto add_finite_difference_camera_jacobian =
        [&](const auto& design_variable_adapter) {
          if (design_variable_adapter == nullptr ||
              !design_variable_adapter->isActive()) {
            return;
          }
          const Eigen::MatrixXd base_parameters =
              design_variable_adapter->getParameters();
          const int dimension = static_cast<int>(base_parameters.size());
          if (dimension <= 0) {
            return;
          }
          Eigen::MatrixXd camera_jacobian(2, dimension);
          camera_jacobian.setZero();
          for (int index = 0; index < dimension; ++index) {
            const Eigen::Index row =
                static_cast<Eigen::Index>(index % base_parameters.rows());
            const Eigen::Index col =
                static_cast<Eigen::Index>(index / base_parameters.rows());
            const double base_value = base_parameters(row, col);
            const double epsilon =
                std::max(1e-7, 1e-6 * std::max(1.0, std::fabs(base_value)));

            Eigen::MatrixXd positive = base_parameters;
            positive(row, col) = base_value + epsilon;
            design_variable_adapter->setParameters(positive);
            HybridEvaluation positive_eval;
            const Eigen::Vector2d positive_residual =
                ComputeResidual(&positive_eval);

            Eigen::MatrixXd negative = base_parameters;
            negative(row, col) = base_value - epsilon;
            design_variable_adapter->setParameters(negative);
            HybridEvaluation negative_eval;
            const Eigen::Vector2d negative_residual =
                ComputeResidual(&negative_eval);

            if (positive_eval.valid_projection &&
                negative_eval.valid_projection &&
                positive_residual.allFinite() &&
                negative_residual.allFinite()) {
              camera_jacobian.col(index) =
                  (positive_residual - negative_residual) / (2.0 * epsilon);
            }
          }
          design_variable_adapter->setParameters(base_parameters);
          jacobians.add(design_variable_adapter.get(), camera_jacobian);
        };

    add_finite_difference_camera_jacobian(
        const_cast<CameraDv&>(camera_dv_).projectionDesignVariable());
    add_finite_difference_camera_jacobian(
        const_cast<CameraDv&>(camera_dv_).distortionDesignVariable());
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
          *camera_dv_.camera(), observed_image_xy_,
          &evaluation->observation_geometry);
    }
    const bool image_valid =
        camera_dv_.camera()->homogeneousToKeypoint(
            point_homogeneous, evaluation->predicted_image_xy) &&
        evaluation->predicted_image_xy.allFinite();
    const bool angular_valid = observation_valid &&
        ComputePredictionGeometryForCamera(
            *camera_dv_.camera(), point_homogeneous,
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

    double Rmse() const {
      return count > 0
                 ? std::sqrt(squared_error / static_cast<double>(count))
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
    } else {
      ++internal_count;
      internal_squared_error += squared_error;
      internal_residual_norms.push_back(residual_norm);
    }
  }
};

double RegressionLimit(double before_rmse,
                       double ratio,
                       double absolute_margin_px) {
  if (!std::isfinite(before_rmse) || before_rmse <= 0.0) {
    return absolute_margin_px;
  }
  return std::max(before_rmse * ratio, before_rmse + absolute_margin_px);
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
  if (candidate_stats.total_count <= 0 || after_stats.total_count <= 0) {
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
  if (candidate_stats.internal_count > 0 &&
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

  if (before_stats.total_count > 0 &&
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
  if (before_stats.internal_count > 0 && candidate_stats.internal_count > 0 &&
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

  if (before_stats.total_count > 0 &&
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
  if (before_stats.internal_count > 0 && after_stats.internal_count > 0 &&
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

  if (candidate_stats.outer_count > 0 && candidate_stats.internal_count > 0) {
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
                                      JointPointType point_type) {
  const bool has_outer = budget.outer_count > 0;
  const bool has_internal = budget.internal_count > 0;
  double type_budget = 1.0;
  int type_count = 1;
  if (has_outer && has_internal) {
    type_budget = point_type == JointPointType::Internal ? 0.5 : 0.5;
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
	    BuildBoardVariables();
	  }

  struct StateSnapshot;
	  struct ResidualConstructionCounts {
	    int image_plane_residual_count = 0;
	    int angular_residual_count = 0;
	    int chordal_residual_count = 0;
	    int hybrid_angular_selected_count = 0;
	    int hybrid_chordal_selected_count = 0;
	    int angular_observation_geometry_failure_count = 0;
	  };

  boost::shared_ptr<CalibrationBatch> BuildBatch(
      const std::set<FrameBoardKey>& keys,
      bool force_add_frame_variables,
      CameraOptimizationPhase camera_phase,
      const StateSnapshot* camera_anchor_state,
      ResidualConstructionCounts* residual_counts = nullptr,
      double camera_anchor_weight_scale = 1.0) {
    boost::shared_ptr<CalibrationBatch> batch =
        boost::make_shared<CalibrationBatch>();
    AddCameraVariables(camera_phase, batch);
    MaybeAddCameraAnchorPrior(camera_phase, camera_anchor_state, batch,
                              camera_anchor_weight_scale);
    AddBoardVariables(batch);
    AddFrameVariables(keys, force_add_frame_variables, batch);
    AddResiduals(keys, batch, residual_counts);
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
	            residual_norm = ComputeAngularResidualNorm(
	                ComputeAngularResidualTangent(observation_geometry,
	                                              prediction_geometry));
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
    const bool projection_active =
        camera_phase == CameraOptimizationPhase::kSeedFixedIntrinsics
            ? options_.optimize_seed_intrinsics
            : options_.optimize_candidate_intrinsics;
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
        !options_.optimize_candidate_intrinsics ||
        !options_.use_candidate_intrinsics_anchor_prior ||
        camera_anchor_state == nullptr ||
        camera_anchor_state->projection_parameters.rows() < 4 ||
        camera_anchor_state->projection_parameters.cols() < 1) {
      return;
    }
    const Eigen::MatrixXd& anchor = camera_anchor_state->projection_parameters;
    if (anchor.rows() != 4 && anchor.rows() != 6) {
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
    const OuterBootstrapCameraIntrinsics anchor_camera = CurrentCamera();
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
    } else if (anchor.rows() == 4) {
      boost::shared_ptr<ProjectionAnchorError<4> > prior(
          new ProjectionAnchorError<4>(
              camera_dv_.projectionDesignVariable().get(), anchor, weights));
      batch->addErrorTerm(prior);
    }
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
      AddPoseVariableDvs(entry.second, kBoardLayoutGroupId, batch);
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
                    const boost::shared_ptr<CalibrationBatch>& batch,
                    ResidualConstructionCounts* residual_counts) {
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
      const FrameBoardKey observation_key(observation.frame_index,
                                          observation.board_id);
      const auto budget_it = observation_budgets_.find(observation_key);
      const ObservationBudget budget =
          budget_it == observation_budgets_.end() ? ObservationBudget{}
                                                  : budget_it->second;
      const double weight =
          ComputePersistentBalanceWeight(budget, observation.point_type) *
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
	                                : (use_angular_residual ? 1.0 : 0.0);
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
	        const double huber_delta_radians =
	            observation.point_type == JointPointType::Outer
                ? options_.outer_huber_delta_radians
                : options_.internal_huber_delta_radians;
        boost::shared_ptr<IncrementalAngularReprojectionError<GeometryT> >
            error(new IncrementalAngularReprojectionError<GeometryT>(
                observation.image_xy, inv_r * angular_weight_scale,
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
	            observation.point_type == JointPointType::Outer
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
  options.residual_model = backend_runner_options.residual_model;
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
  options.angular_observed_ray_mode =
      backend_runner_options.angular_observed_ray_mode;
  options.optimize_seed_intrinsics =
      selection_options.optimize_intrinsics_in_trial &&
      selection_options.force_include_list_is_exact_input;
  options.optimize_candidate_intrinsics =
      selection_options.optimize_intrinsics_in_trial;
  options.use_candidate_intrinsics_anchor_prior =
      selection_options.optimize_intrinsics_in_trial &&
      selection_options.persistent_intrinsics_anchor_prior_enabled;
  options.normalize_information_gain_by_board_observation = false;
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
      baseline_family != "omni-radtan") {
    return set_reason(
        "persistent incremental estimator supports ds-none, eucm-none, pinhole-equi, and omni-radtan only");
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
  result.candidate_batch_count = static_cast<int>(candidate_batches.size());
  result.compatible = true;
  result.information_gain_target = "camera_intrinsics_only";
  result.board_layout_in_information_group = false;
  result.camera_information_group_id =
      static_cast<int>(kCameraInformationGroupId);
  result.board_layout_group_id = static_cast<int>(kBoardLayoutGroupId);
  result.transformation_group_id = static_cast<int>(kTransformationGroupId);

  const Stage5IncrementalBackendEstimatorOptions estimator_options =
      MakeOptions(selection_options, backend_runner_options);
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
      estimator_options.chordal_residual_weight < 0.0) {
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
      SelectionMetricForResidualModel(estimator_options.residual_model);
  result.selection_metric_name = SelectionMetricName(selection_metric);
  result.selection_metric_unit = SelectionMetricUnit(selection_metric);
  result.residual_health_threshold_source =
      selection_metric == SelectionResidualMetric::kPixel
          ? "pixel_trial_rmse_threshold"
          : "adaptive_seed_and_candidate_metric_stats";
  result.normalize_information_gain_by_board_observation =
      estimator_options.normalize_information_gain_by_board_observation;
  result.split_residual_health_gate_enabled =
      estimator_options.use_split_residual_health_gate;
  result.adaptive_saturation_stop_enabled =
      estimator_options.adaptive_saturation_stop_enabled;

  aslam::calibration::IncrementalEstimator::Options inc_options;
  // The persistent Stage5 layer owns accept/reject so the health metric can
  // follow the selected residual model. Keep each optimized candidate
  // temporarily, then explicitly commit or roll it back below.
  inc_options.infoGainDelta = -std::numeric_limits<double>::max();
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
    typename PersistentProblemBuilder<GeometryT>::ResidualConstructionCounts
        seed_residual_counts;
    boost::shared_ptr<CalibrationBatch> seed_batch =
        builder.BuildBatch(seed_keys, true,
                           CameraOptimizationPhase::kSeedFixedIntrinsics,
                           nullptr, &seed_residual_counts);
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
    const aslam::calibration::IncrementalEstimator::ReturnValue seed_ret =
        estimator.addBatch(seed_batch, true);
    result.seed_batch_count = seed_ret.batchAccepted ? 1 : 0;
    if (!seed_ret.batchAccepted) {
      result.failure_reason = "forced seed batch was not accepted";
      return result;
    }
    result.seed_information_group_dim =
        static_cast<int>(seed_ret.rankTheta + seed_ret.rankThetaDeficiency);
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
    const ResidualStats seed_metric_stats =
        builder.EvaluateAccepted(seed_keys, selection_metric);
    result.seed_acceptance_metric_rmse = seed_metric_stats.Rmse();
    result.seed_acceptance_metric_p95 = seed_metric_stats.P95();
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
         candidate_batches) {
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
    batch_ordering_scores.reserve(candidate_batches.size());
    for (const Stage5IncrementalBackendBatchInput& input :
         candidate_batches) {
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

    for (const Stage5IncrementalBackendBatchInput& raw_input :
         candidate_batches) {
      Stage5IncrementalBackendBatchInput input = raw_input;
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
        ++consecutive_nonproductive_batches;
        result.adaptive_saturation_consecutive_nonproductive_batches =
            consecutive_nonproductive_batches;
        continue;
      }

      const typename PersistentProblemBuilder<GeometryT>::StateSnapshot
          batch_state = builder.CaptureState();
      const ResidualStats committed_before_stats =
          builder.EvaluateAccepted(result.accepted_keys, selection_metric);
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
      aslam::calibration::IncrementalEstimator::ReturnValue ret;
      boost::shared_ptr<CalibrationBatch> batch;
      bool incremental_accepted = false;
      std::string state_invalid_reason;
      double trust_region_violation_ratio = 1.0;
      double active_anchor_weight_scale = 1.0;
      bool add_batch_exception = false;

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
            batch_result.information_gate_pass =
                (batch_result.normalized_information_gain >
                     estimator_options.information_gain_threshold ||
                 (batch_result.rank_theta_before >= 0 &&
                  (static_cast<double>(active_ret.rankTheta) -
                   static_cast<double>(batch_result.rank_theta_before)) >
                      estimator_options.rank_gain_threshold));
            const bool residual_score_gate_pass =
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
            batch_result.optimization_success =
                active_ret.numIterations <
                static_cast<std::size_t>(
                    std::max(1, estimator_options.max_iterations));
            batch_result.solution_valid =
                input.force ||
                (batch_result.objective_finite &&
		                 batch_result.objective_gate_pass &&
			                 (batch_result.information_gate_pass ||
			                  residual_score_gate_pass) &&
			                 batch_result.residual_health_pass);
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
                                   anchor_weight_scale);
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
        std::string residual_contract_reason;
        if (!ResidualConstructionMatchesMode(
                estimator_options.residual_model,
                estimator_options.pixel_residual_weight,
                estimator_options.chordal_residual_weight,
                residual_counts.image_plane_residual_count,
                residual_counts.angular_residual_count,
                residual_counts.chordal_residual_count,
                &residual_contract_reason)) {
          builder.RestoreState(batch_state);
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
          const aslam::calibration::IncrementalEstimator::ReturnValue
              attempt_ret = estimator.addBatch(batch, false);
          incremental_accepted = attempt_ret.batchAccepted;
          populate_from_return_value(attempt_ret, incremental_accepted,
                                     retry_count, anchor_weight_scale);
          return true;
        } catch (const std::exception& exception) {
          if (batch) {
            estimator.rejectBatch(batch);
          }
          builder.RestoreState(batch_state);
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
        builder.RestoreState(batch_state);
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
                 << " threshold="
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
        return incremental_accepted
                   ? std::string("incremental_solution_validity_gate")
                   : std::string("incremental_backend_rejected_batch");
      };
      ResidualStats committed_candidate_stats;
      ResidualStats candidate_only_stats;
      bool committed_candidate_stats_ready = false;
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
	        if (batch_result.acceptance_metric_threshold > 0.0 &&
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
          std::string split_residual_reason;
          batch_result.split_residual_health_pass =
              CheckSplitResidualHealthGate(
                  committed_before_stats, committed_candidate_stats,
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
      }
      if (batch_result.batch_accepted) {
        result.accepted_keys.insert(input.frame_board_keys.begin(),
                                    input.frame_board_keys.end());
        current_rank = static_cast<int>(ret.rankTheta);
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
	        batch_result.accept_reason =
	            input.force
	                ? "force"
	                : (SelectionMetricUsesResidualAwareScoreGate(selection_metric)
	                       ? "residual_aware_metric_score_health"
	                       : (batch_result.normalized_information_gain >
	                                  estimator_options.information_gain_threshold
	                              ? "incremental_information_gain"
	                              : "incremental_rank_gain"));
        ++result.accepted_batch_count;
        consecutive_nonproductive_batches = 0;
      } else {
        const std::string active_reject_reason = make_reject_reason();
        if (incremental_accepted) {
          estimator.rejectBatch(batch);
        }
        builder.RestoreState(batch_state);
        batch_result.reject_reason = active_reject_reason;
        ++result.rejected_batch_count;
        const bool low_information =
            !input.force &&
            !batch_result.information_gate_pass &&
            !SelectionMetricUsesAngularHealth(selection_metric) &&
            selection_metric != SelectionResidualMetric::kHybridObjective;
        const bool residual_or_validity_reject =
            !batch_result.residual_health_pass ||
            !batch_result.split_residual_health_pass ||
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
  result.compatible = false;
  result.fallback_reason =
      "persistent incremental estimator unsupported family: " + family;
  result.failure_reason = result.fallback_reason;
  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
