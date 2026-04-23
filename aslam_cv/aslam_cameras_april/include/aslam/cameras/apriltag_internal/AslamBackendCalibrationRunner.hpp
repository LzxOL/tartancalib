#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_ASLAM_BACKEND_CALIBRATION_RUNNER_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_ASLAM_BACKEND_CALIBRATION_RUNNER_HPP

#include <string>
#include <vector>

#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionResidualEvaluator.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct AslamBackendCalibrationOptions {
  int max_iterations = 12;
  double convergence_delta_j = 1e-3;
  double convergence_delta_x = 1e-4;
  double levenberg_marquardt_lambda_init = 1e-3;
  std::string linear_solver = "cholmod";
  bool verbose = false;
  bool use_huber_loss = true;
  double outer_huber_delta_pixels = 10.0;
  double internal_huber_delta_pixels = 6.0;
  double invalid_projection_penalty_pixels = 100.0;
};

struct AslamBackendOptimizationStageSummary {
  std::string stage_label;
  bool optimize_intrinsics = false;
  int max_iterations = 0;
  double objective_start = 0.0;
  double objective_final = 0.0;
  int iterations = 0;
  int failed_iterations = 0;
  double lm_lambda_final = 0.0;
  double delta_x_final = 0.0;
  double delta_j_final = 0.0;
  bool linear_solver_failure = false;
};

struct AslamBackendCalibrationResult {
  bool success = false;
  std::string dataset_label;
  std::string baseline_protocol_label;
  std::string training_split_signature;
  CalibrationBackendProblemInput problem_input;
  AslamBackendCalibrationOptions options;
  OuterBootstrapCameraIntrinsics anchor_camera;
  JointReprojectionSceneState initial_scene_state;
  JointReprojectionSceneState optimized_scene_state;
  JointResidualEvaluationResult initial_residual;
  JointResidualEvaluationResult optimized_residual;
  int design_variable_count = 0;
  int error_term_count = 0;
  std::vector<AslamBackendOptimizationStageSummary> stages;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class AslamBackendCalibrationRunner {
 public:
  explicit AslamBackendCalibrationRunner(
      AslamBackendCalibrationOptions options = AslamBackendCalibrationOptions{});

  AslamBackendCalibrationResult Run(
      const CalibrationBackendProblemInput& input) const;

  const AslamBackendCalibrationOptions& options() const { return options_; }

 private:
  AslamBackendCalibrationOptions options_;
};

void WriteAslamBackendCalibrationSummary(
    const std::string& path,
    const AslamBackendCalibrationResult& result);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_ASLAM_BACKEND_CALIBRATION_RUNNER_HPP
