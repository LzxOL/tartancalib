#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_ASLAM_BACKEND_CALIBRATION_RUNNER_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_ASLAM_BACKEND_CALIBRATION_RUNNER_HPP

#include <string>
#include <vector>

#include <Eigen/Core>

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
  bool export_cost_parity_diagnostics = false;
  bool run_jacobian_consistency_check = false;
  double jacobian_finite_difference_epsilon = 1e-6;
  int debug_max_frames = -1;
  int debug_max_nonreference_boards = -1;
  bool force_pose_only = false;
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

struct AslamBackendPointCostParityDiagnostics {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  JointPointType point_type = JointPointType::Outer;
  Eigen::Vector2d observed_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d frontend_predicted_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d backend_predicted_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d frontend_residual_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d backend_residual_xy = Eigen::Vector2d::Zero();
  bool frontend_valid_projection = false;
  bool backend_valid_projection = false;
  double frontend_balance_weight = 0.0;
  double frontend_huber_weight = 0.0;
  double frontend_final_weight = 0.0;
  double frontend_weighted_squared_error = 0.0;
  double backend_inv_r_scale = 0.0;
  double backend_m_estimator_weight = 0.0;
  double backend_raw_squared_error = 0.0;
  double backend_weighted_squared_error = 0.0;
  double predicted_difference_norm = 0.0;
  double residual_sign_consistency_norm = 0.0;
  double weighted_cost_difference = 0.0;
};

struct AslamBackendCostParityDiagnostics {
  bool success = false;
  std::string stage_label;
  int compared_point_count = 0;
  double frontend_total_squared_error = 0.0;
  double frontend_total_cost = 0.0;
  double backend_reprojection_total_raw_squared_error = 0.0;
  double backend_reprojection_total_weighted_cost = 0.0;
  double backend_problem_total_weighted_cost = 0.0;
  double total_abs_weighted_cost_difference = 0.0;
  double max_abs_weighted_cost_difference = 0.0;
  double max_predicted_difference_norm = 0.0;
  double max_residual_sign_consistency_norm = 0.0;
  std::vector<AslamBackendPointCostParityDiagnostics> point_diagnostics;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct AslamBackendJacobianBlockDiagnostics {
  std::string block_label;
  int dimension = 0;
  std::vector<double> analytic_gradient;
  std::vector<double> finite_difference_gradient;
  double max_abs_difference = 0.0;
};

struct AslamBackendJacobianDiagnostics {
  bool success = false;
  double finite_difference_epsilon = 0.0;
  std::vector<AslamBackendJacobianBlockDiagnostics> block_diagnostics;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct AslamBackendCalibrationResult {
  bool success = false;
  std::string dataset_label;
  std::string baseline_protocol_label;
  std::string training_split_signature;
  CalibrationBackendProblemInput problem_input;
  CalibrationBackendProblemInput effective_problem_input;
  AslamBackendCalibrationOptions options;
  OuterBootstrapCameraIntrinsics anchor_camera;
  JointReprojectionSceneState initial_scene_state;
  JointReprojectionSceneState optimized_scene_state;
  JointResidualEvaluationResult initial_residual;
  JointResidualEvaluationResult optimized_residual;
  AslamBackendCostParityDiagnostics initial_cost_parity;
  AslamBackendCostParityDiagnostics optimized_cost_parity;
  AslamBackendJacobianDiagnostics jacobian_diagnostics;
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

void WriteAslamBackendCostParitySummary(
    const std::string& path,
    const AslamBackendCostParityDiagnostics& diagnostics);

void WriteAslamBackendCostParityCsv(
    const std::string& path,
    const AslamBackendCostParityDiagnostics& diagnostics);

void WriteAslamBackendJacobianSummary(
    const std::string& path,
    const AslamBackendJacobianDiagnostics& diagnostics);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_ASLAM_BACKEND_CALIBRATION_RUNNER_HPP
