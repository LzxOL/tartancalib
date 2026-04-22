#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_APRILTAG_CANONICAL_MODEL_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_APRILTAG_CANONICAL_MODEL_HPP

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/CornerMeasurement.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

enum class InternalProjectionMode {
  Homography,
  VirtualPinholePatch,
};

const char* ToString(InternalProjectionMode mode);

struct IntermediateCameraConfig {
  std::string camera_yaml;
  std::string camera_model;
  std::string distortion_model;
  std::vector<double> intrinsics;
  std::vector<double> distortion_coeffs;
  std::vector<int> resolution;

  bool IsConfigured() const {
    return !camera_model.empty() && !intrinsics.empty() && resolution.size() == 2;
  }
};

struct ApriltagInternalConfig {
  std::string target_type = "apriltag_internal";
  std::string tag_family = "t36h11";
  int tag_id = 0;
  double tag_size = 0.0;
  int black_border_bits = 2;
  int min_visible_points = 12;
  int canonical_pixels_per_module = 24;
  int refinement_window_radius = 0;
  double internal_subpix_window_scale = 0.5;
  int internal_subpix_window_min = 4;
  int internal_subpix_window_max = 16;
  double max_subpix_displacement2 = 0.0;
  double internal_subpix_displacement_scale = 0.25;
  double max_internal_subpix_displacement = 6.0;
  bool enable_debug_output = false;
  InternalProjectionMode internal_projection_mode = InternalProjectionMode::VirtualPinholePatch;
  IntermediateCameraConfig intermediate_camera;
  MultiScaleOuterTagDetectorConfig outer_detector_config;
};

struct CanonicalCorner {
  int point_id = -1;
  int lattice_u = 0;
  int lattice_v = 0;
  Eigen::Vector3d target_xyz = Eigen::Vector3d::Zero();
  bool observable = false;
  CornerType corner_type = CornerType::LCorner;
  std::array<bool, 4> module_pattern{{false, false, false, false}};
};

class ApriltagCanonicalModel {
 public:
  static constexpr int kCodeDimension = 6;

  explicit ApriltagCanonicalModel(ApriltagInternalConfig config);

  const ApriltagInternalConfig& config() const { return config_; }

  int ModuleDimension() const { return module_dimension_; }
  int LatticeDimension() const { return lattice_dimension_; }
  std::size_t PointCount() const { return corners_.size(); }
  double Pitch() const { return pitch_; }

  int PointId(int lattice_u, int lattice_v) const;
  bool IsOuterCorner(int lattice_u, int lattice_v) const;
  bool IsModuleBlack(int module_x, int module_y) const;

  const CanonicalCorner& corner(int point_id) const;
  const std::vector<CanonicalCorner>& corners() const { return corners_; }

  int ObservablePointCount() const;
  std::vector<int> VisiblePointIds() const;
  std::vector<CornerMeasurement> MakeDefaultMeasurements() const;

  cv::Mat RenderBinaryPatch(int pixels_per_module) const;

 private:
  void ValidateConfig() const;
  void BuildModuleGrid();
  void BuildCornerMetadata();

  ApriltagInternalConfig config_;
  int module_dimension_ = 0;
  int lattice_dimension_ = 0;
  double pitch_ = 0.0;
  std::vector<bool> modules_;
  std::vector<CanonicalCorner> corners_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_APRILTAG_CANONICAL_MODEL_HPP
