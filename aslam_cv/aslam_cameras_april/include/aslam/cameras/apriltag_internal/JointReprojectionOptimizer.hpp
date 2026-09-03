#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_OPTIMIZER_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_OPTIMIZER_HPP

#include <string>
#include <vector>

#include <aslam/cameras/apriltag_internal/JointMeasurementSelection.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct JointOptimizationOptions {
  int reference_board_id = 1;
  int max_joint_iterations = 8;
  double convergence_threshold = 1e-3;
  bool optimize_frame_poses = true;
  bool optimize_board_poses = true;
  bool optimize_intrinsics = false;
  int intrinsics_release_iteration = 3;
  JointReprojectionCostOptions cost_options;
  double intrinsics_anchor_weight_xi_alpha = 1e-2;
  double intrinsics_anchor_weight_focal = 1e-4;
  double intrinsics_anchor_weight_principal = 1e-4;
};

struct JointOptimizationIterationSummary {
  int iteration_index = -1;
  double cost_before = 0.0;
  double cost_after = 0.0;
  double cost_delta = 0.0;
  double overall_rmse_before = 0.0;
  double overall_rmse_after = 0.0;
  double outer_rmse_after = 0.0;
  double internal_rmse_after = 0.0;
  int accepted_frame_updates = 0;
  int rejected_frame_updates = 0;
  int accepted_board_updates = 0;
  int rejected_board_updates = 0;
  bool attempted_intrinsics_update = false;
  bool accepted_intrinsics_update = false;
  bool rejected_intrinsics_update = false;
  double intrinsics_step_norm = 0.0;
  double max_parameter_delta = 0.0;
};

struct JointOptimizationRuntimeBreakdown {
  double residual_evaluation_seconds = 0.0;
  int residual_evaluation_call_count = 0;
  double cost_evaluation_seconds = 0.0;
  int cost_evaluation_call_count = 0;
  double frame_update_seconds = 0.0;
  double board_update_seconds = 0.0;
  double intrinsics_update_seconds = 0.0;
};

struct JointOptimizationResult {
  bool success = false;
  int reference_board_id = 1;
  bool optimize_intrinsics = false;
  int intrinsics_release_iteration = 3;
  JointMeasurementSelectionResult selection_result;
  JointReprojectionSceneState initial_state;
  JointReprojectionSceneState optimized_state;
  JointResidualEvaluationResult initial_residual;
  JointResidualEvaluationResult optimized_residual;
  std::vector<JointOptimizationIterationSummary> iterations;
  JointOptimizationRuntimeBreakdown runtime_breakdown;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class JointReprojectionOptimizer {
 public:
  explicit JointReprojectionOptimizer(
      JointOptimizationOptions options = JointOptimizationOptions{});

  JointOptimizationResult Optimize(
      const JointMeasurementSelectionResult& selection_result,
      const JointReprojectionSceneState& initial_state) const;

  const JointOptimizationOptions& options() const { return options_; }

 private:
  JointOptimizationOptions options_;
  JointReprojectionCostCore cost_core_;
  JointReprojectionResidualEvaluator residual_evaluator_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_REPROJECTION_OPTIMIZER_HPP
