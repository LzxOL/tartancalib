#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_POLAR_ANGLE_RESIDUAL_DIAGNOSTICS_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_POLAR_ANGLE_RESIDUAL_DIAGNOSTICS_HPP

#include <map>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionResidualEvaluator.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct PolarAngleDiagnosticsOptions {
  bool enabled = false;
  std::vector<double> bin_edges_deg = {0.0, 30.0, 50.0, 70.0, 85.0, 100.0};
};

struct PolarAngleBinStatistics {
  double bin_min_deg = 0.0;
  double bin_max_deg = 0.0;
  std::string point_type;
  int point_count = 0;
  double rmse = 0.0;
  double mean_abs_x = 0.0;
  double mean_abs_y = 0.0;
  double std_x = 0.0;
  double std_y = 0.0;
  double median_residual = 0.0;
  double p90_residual = 0.0;
  double p95_residual = 0.0;
  double max_residual = 0.0;
};

struct PolarAngleDiagnosticsResult {
  bool success = false;
  std::string failure_reason;
  std::vector<std::string> warnings;
  std::vector<PolarAngleBinStatistics> all_points_bins;
  std::vector<PolarAngleBinStatistics> outer_only_bins;
  std::vector<PolarAngleBinStatistics> internal_only_bins;
  std::map<int, std::vector<PolarAngleBinStatistics>> per_board_bins;
  std::map<int, std::vector<PolarAngleBinStatistics>> per_frame_bins;
};

class PolarAngleResidualDiagnostics {
 public:
  explicit PolarAngleResidualDiagnostics(PolarAngleDiagnosticsOptions options);

  PolarAngleDiagnosticsResult EvaluateWithResiduals(
      const CalibrationMeasurementDataset& measurement_dataset,
      const JointResidualEvaluationResult& residual_result,
      const JointReprojectionSceneState& scene_state,
      const std::string& output_dir) const;

  const PolarAngleDiagnosticsOptions& options() const { return options_; }

 private:
  double ComputePolarAngleDeg(
      const DoubleSphereCameraModel& camera,
      const Eigen::Vector2d& pixel) const;

  void ComputeBinStatistics(
      const std::vector<double>& polar_angles,
      const std::vector<double>& residual_norms,
      PolarAngleBinStatistics* stats) const;

  void WriteSummaryFile(
      const std::string& path,
      const PolarAngleDiagnosticsResult& result) const;

  void WriteCsvFile(
      const std::string& path,
      const PolarAngleDiagnosticsResult& result) const;

  PolarAngleDiagnosticsOptions options_;
};

double ComputePolarAngleDeg(
    const DoubleSphereCameraModel& camera,
    const Eigen::Vector2d& pixel);

double ComputePolarAngleDegFromRay(const Eigen::Vector3d& ray);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_POLAR_ANGLE_RESIDUAL_DIAGNOSTICS_HPP
