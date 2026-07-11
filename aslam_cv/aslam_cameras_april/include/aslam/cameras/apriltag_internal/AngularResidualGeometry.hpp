#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_ANGULAR_RESIDUAL_GEOMETRY_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_ANGULAR_RESIDUAL_GEOMETRY_HPP

#include <Eigen/Core>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct AngularObservationGeometry {
  bool success = false;
  Eigen::Vector2d observed_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector3d observed_ray = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 3, 2> tangent_basis =
      Eigen::Matrix<double, 3, 2>::Zero();
  double polar_angle_deg = 0.0;
};

struct AngularPredictionGeometry {
  bool valid_projection = false;
  Eigen::Vector2d predicted_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector3d predicted_ray = Eigen::Vector3d::Zero();
};

struct BearingCovarianceOptions {
  BearingCovarianceOptions() {
    model_sigma << 0.02, 0.02, 2.0, 2.0, 2.0, 2.0;
  }

  bool use_pixel_uncertainty = false;
  bool use_model_uncertainty = false;
  double pixel_sigma_px = 0.5;
  Eigen::Matrix<double, 6, 1> model_sigma;
  double covariance_damping = 1e-8;
  double min_sigma_rad = 1e-5;
  double max_whitening_weight = 1e5;
};

struct BearingCovarianceResult {
  bool success = false;
  bool model_uncertainty_used = false;
  bool damping_applied = false;
  bool whitening_clamped = false;
  Eigen::Matrix2d tangent_covariance = Eigen::Matrix2d::Identity();
  Eigen::Matrix2d sqrt_information = Eigen::Matrix2d::Identity();
  double tangent_sigma_mean_rad = 0.0;
  double tangent_sigma_min_rad = 0.0;
  double tangent_sigma_max_rad = 0.0;
  double whitening_weight_mean = 1.0;
  double whitening_weight_min = 1.0;
  double whitening_weight_max = 1.0;
};

enum class ResidualModel {
  ImagePlane,
  SphereAngular,
  NormalizedSphereAngular,
  HybridEdgeAngular,
  PolarContinuousHybrid,
  Chordal,
  PixelChordalHybrid,
};

const char* ToString(ResidualModel model);
ResidualModel ParseResidualModel(const std::string& value);

bool ComputeAngularObservationGeometry(
    const DoubleSphereCameraModel& camera,
    const Eigen::Vector2d& observed_image_xy,
    AngularObservationGeometry* geometry);

// Model-independent helpers used after the active camera geometry has
// unprojected/projected the measurement. Keeping these operations separate
// prevents non-DS camera families from being evaluated through a DS proxy.
bool ComputeAngularObservationGeometryFromRay(
    const Eigen::Vector2d& observed_image_xy,
    const Eigen::Vector3d& observed_ray,
    AngularObservationGeometry* geometry);

bool ComputeBearingTangentCovariance(
    const IntermediateCameraConfig& camera_config,
    const Eigen::Vector2d& observed_image_xy,
    const AngularObservationGeometry& observation_geometry,
    const BearingCovarianceOptions& options,
    BearingCovarianceResult* result);

bool ComputeAngularPredictionGeometry(
    const DoubleSphereCameraModel& camera,
    const Eigen::Vector3d& point_camera,
    AngularPredictionGeometry* geometry);

bool ComputeAngularPredictionGeometryFromPoint(
    const Eigen::Vector3d& point_camera,
    const Eigen::Vector2d& predicted_image_xy,
    AngularPredictionGeometry* geometry);

Eigen::Vector2d ComputeAngularResidualTangent(
    const AngularObservationGeometry& observation_geometry,
    const AngularPredictionGeometry& prediction_geometry);

double ComputeAngularResidualNorm(
    const Eigen::Vector2d& angular_residual_xy);

bool ShouldUseAngularResidual(
    ResidualModel model,
    double observed_polar_angle_deg,
    double hybrid_threshold_deg);

double ComputePolarContinuousAngularWeight(
    double observed_polar_angle_deg,
    double threshold_deg,
    double temperature_deg);

double EstimateAngularSigmaPerPixel(
    const DoubleSphereCameraModel& camera,
    const Eigen::Vector2d& observed_image_xy,
    const AngularObservationGeometry& observation_geometry,
    double finite_difference_step_px = 1.0);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_ANGULAR_RESIDUAL_GEOMETRY_HPP
