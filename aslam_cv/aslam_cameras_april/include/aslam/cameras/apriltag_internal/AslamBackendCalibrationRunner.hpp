#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_ASLAM_BACKEND_CALIBRATION_RUNNER_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_ASLAM_BACKEND_CALIBRATION_RUNNER_HPP

#include <string>
#include <vector>
#include <map>

#include <Eigen/Core>

#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>
#include <aslam/cameras/apriltag_internal/AngularResidualGeometry.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionResidualEvaluator.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct ConsistencyObservationWeightSummaryEntry {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int num_outer_points = 0;
  int num_internal_points = 0;
  double translation_error_mm = 0.0;
  double rotation_error_deg = 0.0;
  Eigen::Vector3d translation_correction_mm = Eigen::Vector3d::Zero();
  Eigen::Vector3d rotation_correction_deg = Eigen::Vector3d::Zero();
  double residual_rmse = 0.0;
  double polar_angle_deg = 0.0;
  double consistency_weight = 1.0;
  double final_weight = 1.0;
  bool local_pose_refit_success = false;
  bool reference_pose_from_local_refit = false;
  bool hard_rejected = false;
  std::string failure_reason;
};

struct AslamBackendCalibrationOptions {
  bool uniform_control_point_mode = false;
  enum class BoardPoseParameterization {
    ReferenceChain,
    IndependentFrameBoardPose,
  };

  enum class PolarAngleWeightMode {
    None,
    DiagnosticOnly,
    FixedBins,
    AdaptiveSigma,
  };

  enum class ConsistencyWeightMode {
    Cauchy,
  };

  enum class AngularObservedRayMode {
    DynamicCurrentCamera,
    FrozenAnchorCamera,
  };

  int max_iterations = 12;
  double convergence_delta_j = 1e-3;
  double convergence_delta_x = 1e-4;
  double levenberg_marquardt_lambda_init = 1e-3;
  std::string linear_solver = "cholmod";
  bool verbose = false;
  bool use_huber_loss = true;
  double outer_huber_delta_pixels = 10.0;
  double internal_huber_delta_pixels = 6.0;
  double outer_huber_delta_radians = 0.02;
  double internal_huber_delta_radians = 0.015;
  double invalid_projection_penalty_pixels = 100.0;
  double invalid_projection_penalty_radians = 0.35;
  ResidualModel residual_model = ResidualModel::ImagePlane;
  double hybrid_angular_threshold_deg = 50.0;
  ResidualModel outer_residual_model = ResidualModel::ImagePlane;
  ResidualModel internal_residual_model = ResidualModel::ImagePlane;
  bool use_point_type_residual_split = false;
  bool angular_auxiliary_enabled = false;
  double angular_auxiliary_weight = 0.0;
  bool angular_auxiliary_normalized = false;
  bool angular_auxiliary_apply_to_outer = true;
  bool angular_auxiliary_apply_to_internal = true;
  double polar_continuous_hybrid_threshold_deg = 50.0;
  double polar_continuous_hybrid_temperature_deg = 10.0;
  double normalized_angular_reference_sigma_px = 1.0;
  double normalized_angular_min_sigma_rad = 1e-6;
  double normalized_angular_max_weight_scale = 1.0e8;
  double pixel_residual_weight = 1.0;
  double chordal_residual_weight = 1.0;
  bool pixel_ray_hybrid_refinement_mode = false;
  double pixel_ray_hybrid_lambda = 0.5;
  // These weights are derived once from the pixel-committed camera before
  // final refinement, then remain fixed throughout the LM solve.
  bool pixel_ray_hybrid_polar_adaptive_enabled = false;
  double pixel_ray_hybrid_lambda_min = 0.2;
  double pixel_ray_hybrid_lambda_max = 0.8;
  double pixel_ray_hybrid_transition_start_deg = 30.0;
  double pixel_ray_hybrid_transition_end_deg = 70.0;
  double pixel_ray_hybrid_pixel_scale_floor = 1e-3;
  double pixel_ray_hybrid_ray_scale_floor = 1e-6;
  double pixel_ray_hybrid_huber_delta = 3.0;
  bool angular_use_normalize_jacobian = false;
  bool angular_local_whitening_enabled = false;
  double angular_local_whitening_pixel_sigma_px = 1.0;
  double angular_local_whitening_covariance_damping = 1e-12;
  double angular_local_whitening_min_sigma_rad = 1e-6;
  double angular_local_whitening_max_weight = 1e5;
  AngularObservedRayMode angular_observed_ray_mode =
      AngularObservedRayMode::DynamicCurrentCamera;
  bool export_cost_parity_diagnostics = false;
  bool export_variable_block_influence_diagnostics = false;
  bool run_jacobian_consistency_check = false;
  double jacobian_finite_difference_epsilon = 1e-6;
  std::string internal_anisotropic_weight_mode = "off";
  double internal_anisotropic_x_scale = 1.0;
  double internal_anisotropic_y_scale = 1.0;
  std::string observation_role_weight_mode = "balanced";
  double internal_role_budget_when_mixed = 0.5;
  PolarAngleWeightMode polar_angle_weight_mode = PolarAngleWeightMode::None;
  std::vector<double> polar_angle_weight_bin_edges_deg = {0.0, 30.0, 50.0, 70.0, 85.0, 100.0};
  std::vector<double> polar_angle_weight_fixed_bin_scales = {1.0, 1.0, 0.75, 0.5, 0.25};
  double polar_angle_weight_adaptive_sigma_reference_deg = 50.0;
  double polar_angle_weight_adaptive_sigma_growth = 1.0;
  double polar_angle_weight_min_scale = 0.25;
  bool multi_board_consistency_weighting = false;
  std::string consistency_pose_source = "outer_only";
  ConsistencyWeightMode consistency_weight_mode = ConsistencyWeightMode::Cauchy;
  double consistency_translation_sigma_mm = 3.0;
  double consistency_rotation_sigma_deg = 2.0;
  double consistency_min_weight = 0.25;
  bool consistency_apply_to_outer = true;
  bool consistency_apply_to_internal = true;
  bool consistency_hard_reject_enabled = false;
  double consistency_hard_reject_translation_mm = 8.0;
  double consistency_hard_reject_rotation_deg = 5.0;
  double consistency_hard_reject_residual_px = 8.0;
  bool consistency_dump_weight_summary = true;
  BoardPoseParameterization board_pose_parameterization =
      BoardPoseParameterization::ReferenceChain;
  bool board_pose_prior_enabled = false;
  double board_pose_prior_translation_sigma_mm = 20.0;
  double board_pose_prior_rotation_sigma_deg = 5.0;
  bool enable_angular_residual_diagnostics = false;
  std::vector<double> angular_residual_bin_edges_deg = {0.0, 30.0, 50.0, 70.0, 85.0, 100.0};
  int debug_max_frames = -1;
  int debug_max_nonreference_boards = -1;
  bool force_pose_only = false;
  bool skip_optimization = false;
};

const char* ToString(AslamBackendCalibrationOptions::PolarAngleWeightMode mode);
AslamBackendCalibrationOptions::PolarAngleWeightMode ParsePolarAngleWeightMode(
    const std::string& value);
const char* ToString(
    AslamBackendCalibrationOptions::BoardPoseParameterization mode);
AslamBackendCalibrationOptions::BoardPoseParameterization
ParseBoardPoseParameterization(const std::string& value);
const char* ToString(AslamBackendCalibrationOptions::ConsistencyWeightMode mode);
AslamBackendCalibrationOptions::ConsistencyWeightMode ParseConsistencyWeightMode(
    const std::string& value);
const char* ToString(AslamBackendCalibrationOptions::AngularObservedRayMode mode);
AslamBackendCalibrationOptions::AngularObservedRayMode ParseAngularObservedRayMode(
    const std::string& value);

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
  double frontend_angular_raw_squared_error = 0.0;
  double backend_angular_raw_squared_error = 0.0;
  double frontend_angular_weighted_squared_error = 0.0;
  double backend_angular_weighted_squared_error = 0.0;
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
  double backend_reprojection_total_angular_raw_squared_error = 0.0;
  double backend_reprojection_total_angular_weighted_cost = 0.0;
  double backend_problem_total_weighted_cost = 0.0;
  double total_abs_weighted_cost_difference = 0.0;
  double max_abs_weighted_cost_difference = 0.0;
  double max_predicted_difference_norm = 0.0;
  double max_residual_sign_consistency_norm = 0.0;
  std::vector<AslamBackendPointCostParityDiagnostics> point_diagnostics;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct AngularResidualDiagnosticOptions {
  bool enabled = false;
  std::vector<double> bin_edges_deg = {0.0, 30.0, 50.0, 70.0, 85.0, 100.0};
};

struct AngularResidualBinStatistics {
  double bin_min_deg = 0.0;
  double bin_max_deg = 0.0;
  int point_count = 0;
  int outer_count = 0;
  int internal_count = 0;
  double rmse = 0.0;
  double image_plane_rmse = 0.0;
  double median_residual = 0.0;
  double p90_residual = 0.0;
  double p95_residual = 0.0;
  double max_residual = 0.0;
  double std_x = 0.0;
  double std_y = 0.0;
};

struct AngularResidualDiagnosticsResult {
  bool success = false;
  int image_plane_residual_count = 0;
  int angular_residual_count = 0;
  int outer_image_plane_residual_count = 0;
  int outer_angular_residual_count = 0;
  int internal_image_plane_residual_count = 0;
  int internal_angular_residual_count = 0;
  int finite_polar_angle_count = 0;
  double polar_angle_min_deg = 0.0;
  double polar_angle_mean_deg = 0.0;
  double polar_angle_max_deg = 0.0;
  std::vector<AngularResidualBinStatistics> all_points_bins;
  std::vector<AngularResidualBinStatistics> outer_only_bins;
  std::vector<AngularResidualBinStatistics> internal_only_bins;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct ResidualBlockConstructionStats {
  int image_plane_residual_count = 0;
  int angular_residual_count = 0;
  int chordal_residual_count = 0;
  int pixel_ray_hybrid_residual_count = 0;
  int angular_auxiliary_residual_count = 0;
  int outer_image_plane_residual_count = 0;
  int outer_angular_residual_count = 0;
  int outer_chordal_residual_count = 0;
  int outer_pixel_ray_hybrid_residual_count = 0;
  int outer_angular_auxiliary_residual_count = 0;
  int internal_image_plane_residual_count = 0;
  int internal_angular_residual_count = 0;
  int internal_chordal_residual_count = 0;
  int internal_pixel_ray_hybrid_residual_count = 0;
  int internal_angular_auxiliary_residual_count = 0;
  int skipped_solver_observation_count = 0;
};

struct BackendResidualTypeAssignment {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  JointPointType point_type = JointPointType::Outer;
  double polar_angle_deg = 0.0;
  std::string residual_model_requested;
  std::string residual_model_effective;
  bool angular_observation_geometry_success = false;
  bool pixel_ray_hybrid_polar_adaptive = false;
  double pixel_ray_hybrid_lambda = 0.0;
  double image_plane_weight_scale = 1.0;
  double angular_weight_scale = 0.0;
  double angular_sigma_per_pixel_rad = 0.0;
  double normalized_angular_weight_scale = 0.0;
  bool angular_auxiliary_enabled = false;
  bool angular_auxiliary_normalized = false;
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
  std::string objective_model;
  std::vector<AslamBackendJacobianBlockDiagnostics> block_diagnostics;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct AslamBackendVariableBlockInfluenceEntry {
  std::string stage_label;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  std::string point_type;
  std::string residual_family;
  std::string variable_block;
  std::string variable_scope;
  int residual_count = 0;
  int residual_dimension = 0;
  int jacobian_columns = 0;
  double weighted_cost = 0.0;
  double hessian_trace = 0.0;
  double hessian_frobenius_norm = 0.0;
  double hessian_logdet = 0.0;
  double hessian_rank_proxy = 0.0;
  double gradient_norm = 0.0;
};

struct AslamBackendVariableBlockInfluenceDiagnostics {
  bool success = false;
  std::vector<AslamBackendVariableBlockInfluenceEntry> entries;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct AslamBackendCalibrationResult {
  bool success = false;
  std::string dataset_label;
  std::string baseline_protocol_label;
  std::string training_split_signature;
  std::string board_pose_parameterization = "reference_chain";
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
  AslamBackendVariableBlockInfluenceDiagnostics
      initial_variable_block_influence;
  AslamBackendVariableBlockInfluenceDiagnostics
      optimized_variable_block_influence;
  int design_variable_count = 0;
  int error_term_count = 0;
  ResidualBlockConstructionStats residual_block_construction;
  std::vector<BackendResidualTypeAssignment> residual_type_assignments;
  int consistency_observation_count = 0;
  int consistency_successful_observation_count = 0;
  int consistency_downweighted_observation_count = 0;
  int consistency_hard_rejected_observation_count = 0;
  double consistency_mean_weight = 1.0;
  double consistency_min_applied_weight = 1.0;
  double consistency_max_translation_error_mm = 0.0;
  double consistency_max_rotation_error_deg = 0.0;
  int board_pose_prior_count = 0;
  double board_pose_prior_translation_sigma_mm = 0.0;
  double board_pose_prior_rotation_sigma_deg = 0.0;
  std::vector<ConsistencyObservationWeightSummaryEntry>
      consistency_observation_summaries;
  bool pixel_ray_hybrid_scales_computed_once = false;
  int pixel_ray_hybrid_valid_observation_count = 0;
  int pixel_ray_hybrid_invalid_observation_count = 0;
  double pixel_ray_hybrid_pixel_scale = 0.0;
  double pixel_ray_hybrid_ray_scale = 0.0;
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

void WriteAslamBackendVariableBlockInfluenceCsv(
    const std::string& path,
    const AslamBackendVariableBlockInfluenceDiagnostics& diagnostics);

void WriteBackendResidualTypeAssignmentsCsv(
    const std::string& path,
    const AslamBackendCalibrationResult& result);

void WriteConsistencyWeightSummary(
    const std::string& path,
    const AslamBackendCalibrationResult& result);

void WriteConsistencyPerBoardSummary(
    const std::string& path,
    const AslamBackendCalibrationResult& result);

void WriteConsistencyPerFrameSummary(
    const std::string& path,
    const AslamBackendCalibrationResult& result);

void WriteTopDownweightedObservations(
    const std::string& path,
    const AslamBackendCalibrationResult& result);

AngularResidualDiagnosticsResult EvaluateAngularResidualDiagnostics(
    const JointResidualEvaluationResult& evaluation,
    const AngularResidualDiagnosticOptions& options);

void WriteAngularResidualSummary(
    const std::string& path,
    const AslamBackendCalibrationResult& result,
    const AngularResidualDiagnosticsResult& diagnostics);

void WriteAngularResidualBinsCsv(
    const std::string& path,
    const AngularResidualDiagnosticsResult& diagnostics);

void WriteAngularResidualPointSelectionCsv(
    const std::string& path,
    const JointResidualEvaluationResult& evaluation);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_ASLAM_BACKEND_CALIBRATION_RUNNER_HPP
