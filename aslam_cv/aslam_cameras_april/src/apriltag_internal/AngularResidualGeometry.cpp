#include <aslam/cameras/apriltag_internal/AngularResidualGeometry.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Eigenvalues>

#include <aslam/cameras/apriltag_internal/PolarAngleResidualDiagnostics.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

Eigen::Vector3d NormalizeOrZero(const Eigen::Vector3d& vector) {
  const double norm = vector.norm();
  if (!(norm > 0.0) || !std::isfinite(norm)) {
    return Eigen::Vector3d::Zero();
  }
  return vector / norm;
}

Eigen::Matrix<double, 3, 2> BuildTangentBasis(const Eigen::Vector3d& unit_ray) {
  Eigen::Vector3d tangent_x =
      std::fabs(unit_ray.z()) < 0.9 ? unit_ray.cross(Eigen::Vector3d::UnitZ())
                                    : unit_ray.cross(Eigen::Vector3d::UnitX());
  tangent_x = NormalizeOrZero(tangent_x);
  Eigen::Vector3d tangent_y = NormalizeOrZero(unit_ray.cross(tangent_x));
  Eigen::Matrix<double, 3, 2> basis = Eigen::Matrix<double, 3, 2>::Zero();
  basis.col(0) = tangent_x;
  basis.col(1) = tangent_y;
  return basis;
}

std::string Lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return value;
}

}  // namespace

const char* ToString(ResidualModel model) {
  switch (model) {
    case ResidualModel::ImagePlane:
      return "image_plane";
    case ResidualModel::SphereAngular:
      return "sphere_angular";
    case ResidualModel::NormalizedSphereAngular:
      return "normalized_sphere_angular";
    case ResidualModel::HybridEdgeAngular:
      return "hybrid_edge_angular";
    case ResidualModel::PolarContinuousHybrid:
      return "polar_continuous_hybrid";
  }
  return "image_plane";
}

ResidualModel ParseResidualModel(const std::string& value) {
  const std::string lowered = Lowercase(value);
  if (lowered == "image_plane" || lowered == "imageplane") {
    return ResidualModel::ImagePlane;
  }
  if (lowered == "sphere_angular" || lowered == "sphereangular" ||
      lowered == "angular") {
    return ResidualModel::SphereAngular;
  }
  if (lowered == "normalized_sphere_angular" ||
      lowered == "normalizedsphereangular" ||
      lowered == "normalized_angular" ||
      lowered == "normalizedangular") {
    return ResidualModel::NormalizedSphereAngular;
  }
  if (lowered == "hybrid_edge_angular" || lowered == "hybridedgeangular" ||
      lowered == "hybrid") {
    return ResidualModel::HybridEdgeAngular;
  }
  if (lowered == "polar_continuous_hybrid" ||
      lowered == "polarcontinuoushybrid" ||
      lowered == "continuous_hybrid" || lowered == "continuous") {
    return ResidualModel::PolarContinuousHybrid;
  }
  throw std::runtime_error("Unknown residual model: " + value);
}

bool ComputeAngularObservationGeometry(
    const DoubleSphereCameraModel& camera,
    const Eigen::Vector2d& observed_image_xy,
    AngularObservationGeometry* geometry) {
  if (geometry == nullptr) {
    throw std::runtime_error(
        "ComputeAngularObservationGeometry requires a valid output pointer.");
  }
  *geometry = AngularObservationGeometry{};
  geometry->observed_image_xy = observed_image_xy;
  Eigen::Vector3d observed_ray = Eigen::Vector3d::Zero();
  if (!camera.keypointToEuclidean(observed_image_xy, &observed_ray)) {
    return false;
  }
  const Eigen::Vector3d unit_ray = NormalizeOrZero(observed_ray);
  if (!unit_ray.allFinite() || unit_ray.isZero(1e-12)) {
    return false;
  }
  geometry->observed_ray = unit_ray;
  geometry->tangent_basis = BuildTangentBasis(unit_ray);
  geometry->polar_angle_deg = ComputePolarAngleDegFromRay(unit_ray);
  geometry->success = geometry->tangent_basis.allFinite();
  return geometry->success;
}

bool ComputeAngularPredictionGeometry(
    const DoubleSphereCameraModel& camera,
    const Eigen::Vector3d& point_camera,
    AngularPredictionGeometry* geometry) {
  if (geometry == nullptr) {
    throw std::runtime_error(
        "ComputeAngularPredictionGeometry requires a valid output pointer.");
  }
  *geometry = AngularPredictionGeometry{};
  if (!camera.vsEuclideanToKeypoint(point_camera, &geometry->predicted_image_xy)) {
    geometry->predicted_image_xy =
        Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN());
    return false;
  }
  const Eigen::Vector3d unit_ray = NormalizeOrZero(point_camera);
  if (!unit_ray.allFinite() || unit_ray.isZero(1e-12)) {
    return false;
  }
  geometry->predicted_ray = unit_ray;
  geometry->valid_projection = true;
  return true;
}

Eigen::Vector2d ComputeAngularResidualTangent(
    const AngularObservationGeometry& observation_geometry,
    const AngularPredictionGeometry& prediction_geometry) {
  if (!observation_geometry.success || !prediction_geometry.valid_projection) {
    return Eigen::Vector2d::Zero();
  }
  const Eigen::Vector3d delta =
      prediction_geometry.predicted_ray - observation_geometry.observed_ray;
  return observation_geometry.tangent_basis.transpose() * delta;
}

double ComputeAngularResidualNorm(
    const Eigen::Vector2d& angular_residual_xy) {
  return angular_residual_xy.norm();
}

bool ShouldUseAngularResidual(
    ResidualModel model,
    double observed_polar_angle_deg,
    double hybrid_threshold_deg) {
  switch (model) {
    case ResidualModel::ImagePlane:
      return false;
    case ResidualModel::SphereAngular:
      return true;
    case ResidualModel::NormalizedSphereAngular:
      return true;
    case ResidualModel::HybridEdgeAngular:
      return std::isfinite(observed_polar_angle_deg) &&
             observed_polar_angle_deg >= hybrid_threshold_deg;
    case ResidualModel::PolarContinuousHybrid:
      return false;
  }
  return false;
}

double ComputePolarContinuousAngularWeight(
    double observed_polar_angle_deg,
    double threshold_deg,
    double temperature_deg) {
  if (!std::isfinite(observed_polar_angle_deg)) {
    return 0.0;
  }
  const double safe_temperature = std::max(1e-6, temperature_deg);
  const double normalized =
      (observed_polar_angle_deg - threshold_deg) / safe_temperature;
  if (normalized >= 40.0) {
    return 1.0;
  }
  if (normalized <= -40.0) {
    return 0.0;
  }
  return 1.0 / (1.0 + std::exp(-normalized));
}

double EstimateAngularSigmaPerPixel(
    const DoubleSphereCameraModel& camera,
    const Eigen::Vector2d& observed_image_xy,
    const AngularObservationGeometry& observation_geometry,
    double finite_difference_step_px) {
  if (!observation_geometry.success ||
      !(finite_difference_step_px > 0.0) ||
      !std::isfinite(finite_difference_step_px)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  auto tangent_delta_norm = [&](const Eigen::Vector2d& shifted_xy,
                                double* norm) {
    AngularObservationGeometry shifted_geometry;
    if (!ComputeAngularObservationGeometry(camera, shifted_xy,
                                           &shifted_geometry)) {
      return false;
    }
    const Eigen::Vector3d delta =
        shifted_geometry.observed_ray - observation_geometry.observed_ray;
    const Eigen::Vector2d tangent_delta =
        observation_geometry.tangent_basis.transpose() * delta;
    *norm = tangent_delta.norm();
    return std::isfinite(*norm);
  };

  std::vector<double> sigma_candidates;
  sigma_candidates.reserve(4);
  const Eigen::Vector2d offsets[] = {
      Eigen::Vector2d(finite_difference_step_px, 0.0),
      Eigen::Vector2d(-finite_difference_step_px, 0.0),
      Eigen::Vector2d(0.0, finite_difference_step_px),
      Eigen::Vector2d(0.0, -finite_difference_step_px),
  };
  for (const Eigen::Vector2d& offset : offsets) {
    double norm = 0.0;
    if (tangent_delta_norm(observed_image_xy + offset, &norm)) {
      sigma_candidates.push_back(norm / finite_difference_step_px);
    }
  }
  if (sigma_candidates.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double squared_sum = 0.0;
  int count = 0;
  for (const double sigma : sigma_candidates) {
    if (sigma > 0.0 && std::isfinite(sigma)) {
      squared_sum += sigma * sigma;
      ++count;
    }
  }
  if (count == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::sqrt(squared_sum / static_cast<double>(count));
}

bool ComputeBearingTangentCovariance(
    const IntermediateCameraConfig& camera_config,
    const Eigen::Vector2d& observed_image_xy,
    const AngularObservationGeometry& observation_geometry,
    const BearingCovarianceOptions& options,
    BearingCovarianceResult* result) {
  if (result == nullptr) {
    throw std::runtime_error(
        "ComputeBearingTangentCovariance requires a valid output pointer.");
  }
  *result = BearingCovarianceResult{};
  if (!observation_geometry.success ||
      !observation_geometry.tangent_basis.allFinite()) {
    return false;
  }

  Eigen::Matrix3d sigma_b = Eigen::Matrix3d::Zero();
  const double pixel_sigma = std::max(0.0, options.pixel_sigma_px);
  auto compute_shifted_ray = [&](const IntermediateCameraConfig& config,
                                 const Eigen::Vector2d& xy,
                                 Eigen::Vector3d* ray) {
    const DoubleSphereCameraModel camera = DoubleSphereCameraModel::FromConfig(config);
    AngularObservationGeometry shifted_geometry;
    if (!ComputeAngularObservationGeometry(camera, xy, &shifted_geometry)) {
      return false;
    }
    *ray = shifted_geometry.observed_ray;
    return ray->allFinite();
  };

  if (options.use_pixel_uncertainty && pixel_sigma > 0.0) {
    Eigen::Matrix<double, 3, 2> jacobian_u;
    jacobian_u.setZero();
    const double eps = 1.0;
    bool pixel_ok = true;
    for (int dim = 0; dim < 2; ++dim) {
      Eigen::Vector2d delta = Eigen::Vector2d::Zero();
      delta(dim) = eps;
      Eigen::Vector3d plus = Eigen::Vector3d::Zero();
      Eigen::Vector3d minus = Eigen::Vector3d::Zero();
      if (!compute_shifted_ray(camera_config, observed_image_xy + delta, &plus) ||
          !compute_shifted_ray(camera_config, observed_image_xy - delta, &minus)) {
        pixel_ok = false;
        break;
      }
      jacobian_u.col(dim) = (plus - minus) / (2.0 * eps);
    }
    if (!pixel_ok || !jacobian_u.allFinite()) {
      return false;
    }
    sigma_b += pixel_sigma * pixel_sigma * jacobian_u * jacobian_u.transpose();
  }

  const std::string camera_model = Lowercase(camera_config.camera_model);
  const std::string distortion_model = Lowercase(camera_config.distortion_model);
  const bool is_ds_none =
      (camera_model == "ds" || camera_model == "double_sphere" ||
       camera_model == "double-sphere") &&
      (distortion_model.empty() || distortion_model == "none");
  const bool can_use_model = options.use_model_uncertainty && is_ds_none &&
                             camera_config.intrinsics.size() == 6;
  if (can_use_model) {
    Eigen::Matrix<double, 3, 6> jacobian_theta;
    jacobian_theta.setZero();
    bool model_ok = true;
    for (int dim = 0; dim < 6; ++dim) {
      const double base = camera_config.intrinsics[dim];
      const double eps =
          std::max(1e-7, 1e-6 * std::max(1.0, std::fabs(base)));
      IntermediateCameraConfig plus_config = camera_config;
      IntermediateCameraConfig minus_config = camera_config;
      plus_config.intrinsics[dim] = base + eps;
      minus_config.intrinsics[dim] = base - eps;
      Eigen::Vector3d plus = Eigen::Vector3d::Zero();
      Eigen::Vector3d minus = Eigen::Vector3d::Zero();
      if (!compute_shifted_ray(plus_config, observed_image_xy, &plus) ||
          !compute_shifted_ray(minus_config, observed_image_xy, &minus)) {
        model_ok = false;
        break;
      }
      jacobian_theta.col(dim) = (plus - minus) / (2.0 * eps);
    }
    if (!model_ok || !jacobian_theta.allFinite()) {
      return false;
    }
    Eigen::Matrix<double, 6, 6> sigma_theta =
        Eigen::Matrix<double, 6, 6>::Zero();
    for (int dim = 0; dim < 6; ++dim) {
      const double sigma = std::max(0.0, options.model_sigma(dim));
      sigma_theta(dim, dim) = sigma * sigma;
    }
    sigma_b += jacobian_theta * sigma_theta * jacobian_theta.transpose();
    result->model_uncertainty_used = true;
  }

  Eigen::Matrix2d sigma_tan =
      observation_geometry.tangent_basis.transpose() * sigma_b *
      observation_geometry.tangent_basis;
  sigma_tan = 0.5 * (sigma_tan + sigma_tan.transpose());
  if (!sigma_tan.allFinite()) {
    return false;
  }
  const double min_variance =
      std::max(0.0, options.min_sigma_rad) *
      std::max(0.0, options.min_sigma_rad);
  double damping = std::max(0.0, options.covariance_damping);
  if (sigma_tan.trace() <= 0.0) {
    damping = std::max(damping, min_variance);
  }
  sigma_tan += damping * Eigen::Matrix2d::Identity();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(sigma_tan);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
      !solver.eigenvectors().allFinite()) {
    return false;
  }
  Eigen::Vector2d variances = solver.eigenvalues();
  const double max_weight = std::max(1.0, options.max_whitening_weight);
  Eigen::Vector2d sqrt_info_diag = Eigen::Vector2d::Zero();
  Eigen::Vector2d sigmas = Eigen::Vector2d::Zero();
  for (int dim = 0; dim < 2; ++dim) {
    double variance = variances(dim);
    if (!(variance > min_variance) || !std::isfinite(variance)) {
      variance = min_variance;
      result->damping_applied = true;
    }
    sigmas(dim) = std::sqrt(variance);
    double weight = 1.0 / std::max(sigmas(dim), options.min_sigma_rad);
    if (weight > max_weight) {
      weight = max_weight;
      result->whitening_clamped = true;
    }
    sqrt_info_diag(dim) = weight;
  }
  result->tangent_covariance = sigma_tan;
  result->sqrt_information =
      solver.eigenvectors() * sqrt_info_diag.asDiagonal() *
      solver.eigenvectors().transpose();
  result->tangent_sigma_mean_rad = 0.5 * (sigmas.x() + sigmas.y());
  result->tangent_sigma_min_rad = sigmas.minCoeff();
  result->tangent_sigma_max_rad = sigmas.maxCoeff();
  result->whitening_weight_mean =
      0.5 * (sqrt_info_diag.x() + sqrt_info_diag.y());
  result->whitening_weight_min = sqrt_info_diag.minCoeff();
  result->whitening_weight_max = sqrt_info_diag.maxCoeff();
  result->success = result->sqrt_information.allFinite();
  return result->success;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
