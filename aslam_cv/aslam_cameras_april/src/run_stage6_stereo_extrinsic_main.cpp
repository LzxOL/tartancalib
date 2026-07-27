#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardCoObservationConsistency.hpp>
#include <aslam/cameras/apriltag_internal/Stage5BackendDiagnosticWriters.hpp>
#include <aslam/cameras/apriltag_internal/StereoExtrinsicCalibrationRunner.hpp>
#include <aslam/cameras/apriltag_internal/StereoExtrinsicProblemInput.hpp>
#include <aslam/cameras/apriltag_internal/StereoResidualEvaluator.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

namespace {

namespace ati = aslam::cameras::apriltag_internal;
namespace fs = boost::filesystem;
using Clock = std::chrono::steady_clock;

struct MonocularFrontendBundleResult {
  ati::CalibrationStateBundle bundle;
  ati::FrozenRound2BaselineResult baseline_result;
  ati::FrozenRound2BaselineRuntimeBreakdown runtime_breakdown;
};

struct CmdArgs {
  std::string left_image_path;
  std::string right_image_path;
  std::string test_left_image_path;
  std::string test_right_image_path;
  std::string left_config_path;
  std::string right_config_path;
  std::string left_intrinsics_path;
  std::string right_intrinsics_path;
  std::string output_path;
  std::string cache_dir;
  std::string stereo_reference_camchain_path;
  std::string stereo_reference_left_intrinsics_path;
  std::string stereo_reference_right_intrinsics_path;
  std::string stage6_stereo_measurement_source = "all_valid";
  std::string stage6_frame_pairing_mode = "exact_timestamp";
  double stage6_frame_pairing_max_delta_ms = 250.0;
  int holdout_stride = 5;
  int holdout_offset = 0;
  int stage6_max_alternating_iterations = 10;
  double stage6_min_total_rmse_improvement = 1e-4;
  bool stage6_require_symmetric_pair_refit = true;
  int stage6_min_shared_boards_for_extrinsic_candidate = 1;
  int stage6_max_graph_propagation_iterations = 50;
  bool stage6_enable_shared_board_quality_gate = true;
  bool stage6_enable_shared_board_quality_hard_gate = false;
  double stage6_shared_board_quality_max_outer_rmse_px = 3.0;
  int stage6_shared_board_quality_min_outer_points_per_camera = 4;
  int stage6_shared_board_quality_min_good_shared_boards = 1;
  bool stage6_export_pair_board_consistency_audit = false;
  bool stage6_enable_pair_board_consistency_gate = false;
  double stage6_pair_board_consistency_local_good_max_outer_rmse_px = 3.0;
  double stage6_pair_board_consistency_global_bad_min_outer_rmse_px = 15.0;
  std::string stage6_view_selection_mode = "off";
  int stage6_selected_pair_count = 0;
  std::string stage6_solver_mode = "global_sparse_ba";
  std::string stage6_intrinsics_mode =
      "adaptive_regularized_joint_projection";
  std::string stage6_persistent_pose_structure = "independent_pair_board";
  std::string stage6_ba_mode = "pixel";
  std::string stage6_final_ba_residual_mode = "pixel";
  std::string stage6_selection_ba_residual_mode = "pixel";
  bool stage6_fixed_intrinsics_for_spherical = true;
  double stage6_spherical_weight = 1.0;
  bool stage6_spherical_polar_weighting = false;
  double stage6_spherical_min_polar_deg = 50.0;
  double stage6_spherical_max_weight = 4.0;
  std::string stage6_spherical_uncertainty_mode = "none";
  double stage6_spherical_pixel_sigma_px = 0.5;
  std::array<double, 6> stage6_spherical_model_sigma{
      {0.02, 0.02, 2.0, 2.0, 2.0, 2.0}};
  double stage6_spherical_covariance_damping = 1e-8;
  double stage6_spherical_min_sigma_rad = 1e-5;
  double stage6_spherical_max_whitening_weight = 1e5;
  bool stage6_spherical_use_normalize_jacobian = false;
  std::string stage6_rig_param_mode = "cam0_reference";
  double stage6_rig_camera_prior_translation_weight = 1e-6;
  double stage6_rig_camera_prior_rotation_weight = 1e-6;
  double stage6_rig_stereo_relative_prior_weight = 1e-6;
  bool stage6_coobs_enable = false;
  std::string stage6_coobs_output_dir;
  int stage6_coobs_min_corners_per_group = 12;
  double stage6_coobs_high_polar_threshold_deg = 50.0;
  double stage6_coobs_very_high_polar_threshold_deg = 70.0;
  bool stage6_coobs_enable_rescue_suggestions = true;
  double stage6_coobs_score_alpha_high_polar = 1.0;
  double stage6_coobs_score_beta_multiboard = 2.0;
  double stage6_coobs_score_gamma_balance = 1.0;
  double stage6_coobs_score_eta_conflict = 1.0;
  double stage6_coobs_rescue_min_high_polar_score = 8.0;
  double stage6_coobs_rescue_bad_conflict_threshold = 5.0;
  bool stage6_selection_coobs_factor_ba_enable = false;
  bool stage6_selection_coobs_factor_ba_apply_stereo_factor = true;
  bool stage6_selection_coobs_factor_ba_apply_layout_factor = false;
  double stage6_selection_coobs_factor_ba_stereo_weight = 0.1;
  double stage6_selection_coobs_factor_ba_layout_weight = 0.05;
  bool stage6_coobs_aware_acceptance_enable = false;
  double stage6_coobs_aware_acceptance_min_score = 1.0;
  double stage6_coobs_aware_acceptance_max_total_rmse_delta = 0.05;
  double stage6_coobs_aware_acceptance_max_camera_rmse_delta = 0.08;
  bool stage6_coobs_aware_acceptance_balance_guard_enable = false;
  double stage6_coobs_aware_acceptance_max_camera_delta_imbalance = 0.03;
  double stage6_coobs_aware_acceptance_max_camera_delta_ratio = 2.0;
  bool stage6_coobs_aware_acceptance_require_pair_completion = true;
  double stage6_coobs_aware_acceptance_stereo_rot_scale_deg = 1.0;
  double stage6_coobs_aware_acceptance_layout_rot_scale_deg = 1.0;
  double stage6_coobs_aware_acceptance_stereo_trans_scale_m = 0.01;
  double stage6_coobs_aware_acceptance_layout_trans_scale_m = 0.005;
  bool stage6_coobs_factor_ba_enable = false;
  bool stage6_coobs_factor_ba_run_experiment_matrix = true;
  std::string stage6_coobs_factor_ba_output_dir_suffix = "coobs_factor_ba";
  int stage6_coobs_factor_ba_min_corners_per_cam_board = 12;
  double stage6_coobs_factor_ba_max_local_pose_rmse = 3.0;
  double stage6_coobs_factor_ba_huber_delta = 1.0;
  std::vector<double> stage6_coobs_factor_ba_stereo_weights{
      0.005, 0.01, 0.03, 0.05};
  std::vector<double> stage6_coobs_factor_ba_layout_weights{
      0.001, 0.003, 0.005, 0.01, 0.03};
  std::vector<double> stage6_coobs_factor_ba_combined_stereo_weights{
      0.01, 0.03};
  std::vector<double> stage6_coobs_factor_ba_combined_layout_weights{
      0.003, 0.005, 0.01};
  std::vector<std::pair<int, int> > stage6_coobs_factor_ba_layout_selected_pairs{
      {2, 4}, {1, 4}, {2, 5}};
  bool stage6_final_ba_optimize_intrinsics = false;
  bool stage6_final_ba_optimize_stereo_extrinsic = true;
  bool stage6_final_ba_optimize_pair_poses = true;
  bool stage6_final_ba_optimize_board_poses = true;
  bool stage6_skip_final_global_ba = true;
  int stage6_ba_max_iterations = 20;
  double stage6_ba_convergence_threshold = 1e-6;
  double stage6_ba_shared_observation_weight_scale = 1.0;
  double stage6_ba_single_camera_only_observation_weight_scale = 1.0;
  std::string stage6_ba_single_camera_only_weight_mode = "fixed_scale";
  double stage6_ba_single_camera_only_base_scale = 1.0;
  double stage6_ba_single_camera_only_per_side_budget_ratio = 0.10;
  double stage6_ba_adaptive_single_camera_only_per_side_cap_ratio = 0.05;
  bool stage6_enable_pair_only_stereo_ba_init = true;
  int stage6_pair_init_max_iterations = 50;
  double stage6_pair_init_convergence_threshold = 1e-6;
  bool stage6_pair_init_use_huber_loss = true;
  bool stage6_enable_kalibr_style_pair_selection = true;
  bool stage6_enable_committing_pair_batch_selection = false;
  bool stage6_enable_persistent_incremental_stereo_ba = true;
  bool stage6_allow_legacy_selection_fallback_after_persistent_failure = false;
  bool stage6_enable_stage6_incremental_estimator = false;
  bool stage6_enable_incremental_pair_diversity_rescue = false;
  int stage6_incremental_pair_diversity_rescue_min_boards = 1;
  double stage6_incremental_mi_tol = 0.2;
  double stage6_incremental_rank_threshold = 1e-6;
  int stage6_persistent_incremental_max_iterations = 8;
  double stage6_persistent_incremental_convergence_delta_j = 1e-3;
  double stage6_persistent_incremental_convergence_delta_x = 1e-4;
  double stage6_persistent_incremental_baseline_prior_translation_weight = 1e-6;
  double stage6_persistent_incremental_baseline_prior_rotation_weight = 1e-6;
  double stage6_persistent_incremental_projection_prior_shape_sigma = 0.01;
  double stage6_persistent_incremental_projection_prior_focal_relative_sigma =
      0.01;
  double stage6_persistent_incremental_projection_prior_principal_sigma_px =
      5.0;
  int stage6_adaptive_joint_projection_min_training_pairs = 8;
  int stage6_adaptive_joint_projection_min_shared_pair_boards = 20;
  int stage6_adaptive_joint_projection_min_distinct_boards = 3;
  int stage6_adaptive_joint_projection_min_observation_points = 1000;
  double stage6_persistent_incremental_invalid_projection_penalty_px = 100.0;
  int stage6_persistent_incremental_seed_pair_count = 1;
  std::string stage6_incremental_info_block = "stereo_extrinsic";
  std::string stage6_batch_acceptance_policy = "residual_score";
  int stage6_pair_selection_seed_count = 10;
  std::string stage6_pair_selection_budget_mode;
  bool stage6_pair_selection_budget_mode_set = false;
  int stage6_pair_selection_max_candidate_additions = 30;
  double stage6_pair_selection_adaptive_budget_ratio = 0.20;
  int stage6_pair_selection_adaptive_budget_min = 30;
  int stage6_pair_selection_adaptive_budget_max = 120;
  int stage6_pair_selection_runtime_safety_ceiling = 1000;
  int stage6_pair_selection_min_shared_boards = 1;
  double stage6_pair_selection_max_rmse_delta = 0.02;
  double stage6_pair_selection_max_camera_rmse_delta = 0.05;
  double stage6_pair_selection_max_baseline_rotation_delta_deg = 0.2;
  double stage6_pair_selection_max_baseline_translation_delta_m = 0.005;
  bool stage6_enable_pair_board_trial_selection = true;
  std::string stage6_pairboard_selection_mode = "kalibr_style_batch";
  std::string stage6_pair_board_selection_budget_mode;
  bool stage6_pair_board_selection_budget_mode_set = false;
  int stage6_pair_board_selection_seed_count = 50;
  int stage6_pair_board_selection_max_candidate_additions = 40;
  double stage6_pair_board_selection_adaptive_budget_ratio = 0.15;
  int stage6_pair_board_selection_adaptive_budget_min = 40;
  int stage6_pair_board_selection_adaptive_budget_max = 200;
  int stage6_pair_board_selection_runtime_safety_ceiling = 1500;
  double stage6_pair_board_selection_min_candidate_score = 20.0;
  double stage6_pair_board_selection_min_coverage_gain = 0.0;
  int stage6_pair_board_selection_max_accepted_per_pair = 4;
  int stage6_pair_board_selection_max_accepted_per_board = 24;
  double stage6_pair_board_selection_max_rmse_delta = 0.02;
  double stage6_pair_board_selection_max_camera_rmse_delta = 0.05;
  double stage6_pair_board_selection_max_baseline_rotation_delta_deg = 0.2;
  double stage6_pair_board_selection_max_baseline_translation_delta_m = 0.005;
  bool stage6_enable_pair_cohesion = true;
  int stage6_pair_cohesion_min_boards_per_pair = 2;
  int stage6_pair_cohesion_max_companions_per_pair = 0;
  bool stage6_pair_cohesion_relax_score_gate = true;
  bool stage6_pair_cohesion_relax_cap_gates = true;
  std::string stage6_single_board_pair_policy = "audit";
  std::vector<std::pair<int, int> > stage6_ablation_excluded_pair_boards;
  bool stage6_export_stereo_reprojection_visualizations = false;
  int stage6_stereo_visualization_top_k = 50;
  bool stage6_export_extrinsic_uncertainty_diagnostics = false;
  bool stage6_export_angular_fixedk_diagnostic = false;
  bool stage6_enable_geometry_prior_outer_seed = true;
  bool stage6_geometry_prior_rescue_diagnostic_only = false;
  bool stage6_geometry_prior_rescue_use_as_observation = true;
  bool stage6_geometry_prior_rescue_keep_outer_on_internal_failure = false;
  bool stage6_geometry_prior_rescue_allow_geometry_only_pose_refit = false;
  int stage6_geometry_prior_rescue_subpix_window_radius = 0;
  double stage6_geometry_prior_rescue_max_corner_displacement_px = 0.0;
  double stage6_geometry_prior_rescue_min_corner_response_ratio = 0.03;
  bool stage6_geometry_prior_rescue_enable_spherical_refine = true;
  int stage6_geometry_prior_rescue_edge_sample_count = 80;
  int stage6_geometry_prior_rescue_edge_search_half_width_px = 6;
  double stage6_geometry_prior_rescue_min_edge_support_ratio = 0.45;
  double stage6_geometry_prior_rescue_min_edge_gradient_ratio = 0.02;
  double stage6_geometry_prior_rescue_accept_max_outer_rmse = 8.0;
  double stage6_geometry_prior_rescue_accept_max_rotation_error_deg = 5.0;
  double stage6_geometry_prior_rescue_accept_max_translation_error = 0.08;
  bool stage6_disable_geometry_prior_rescue_for_holdout = false;
  std::string stage6_board_masking_ablation = "none";
  int stage6_training_pair_sample_count = 0;
  int stage6_training_pair_sample_seed = 20260622;
};

bool ParseBool(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "1" || lowered == "true" || lowered == "yes" ||
      lowered == "on") {
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" ||
      lowered == "off") {
    return false;
  }
  throw std::runtime_error("Expected boolean value, got: " + value);
}

std::pair<int, int> ParsePairBoardKey(const std::string& value,
                                      const std::string& flag_name) {
  const std::size_t separator = value.find(':');
  if (separator == std::string::npos || separator == 0 ||
      separator + 1 >= value.size()) {
    throw std::runtime_error(flag_name + " expects PAIR:BOARD, got: " + value);
  }
  return std::make_pair(std::stoi(value.substr(0, separator)),
                        std::stoi(value.substr(separator + 1)));
}

std::vector<double> ParseDoubleList(const std::string& value,
                                    const std::string& flag_name) {
  std::vector<double> result;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(),
                               [](unsigned char ch) {
                                 return std::isspace(ch) != 0;
                               }),
                token.end());
    if (token.empty()) {
      continue;
    }
    result.push_back(std::stod(token));
  }
  if (result.empty()) {
    throw std::runtime_error(flag_name + " requires at least one value.");
  }
  return result;
}

std::vector<std::pair<int, int> > ParseBoardPairList(
    const std::string& value,
    const std::string& flag_name) {
  std::vector<std::pair<int, int> > result;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(),
                               [](unsigned char ch) {
                                 return std::isspace(ch) != 0;
                               }),
                token.end());
    if (token.empty()) {
      continue;
    }
    const std::size_t dash = token.find('-');
    if (dash == std::string::npos) {
      throw std::runtime_error(flag_name +
                               " expects board pairs like 2-4,1-4.");
    }
    int a = std::stoi(token.substr(0, dash));
    int b = std::stoi(token.substr(dash + 1));
    if (a == b) {
      throw std::runtime_error(flag_name + " cannot contain self-pairs.");
    }
    if (b < a) {
      std::swap(a, b);
    }
    result.push_back(std::make_pair(a, b));
  }
  if (result.empty()) {
    throw std::runtime_error(flag_name + " requires at least one pair.");
  }
  return result;
}

std::array<double, 6> ParseSixDoubles(const std::string& value,
                                      const std::string& flag_name);

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --left-image DIR --right-image DIR"
      << " [--test-left-image DIR --test-right-image DIR]"
      << " --left-config YAML --right-config YAML"
      << " --left-intrinsics YAML --right-intrinsics YAML"
      << " --output OUTPUT_DIR"
      << " [--cache-dir PATH]"
      << " [--stereo-reference-camchain YAML]"
      << " [--stereo-reference-left-intrinsics YAML]"
      << " [--stereo-reference-right-intrinsics YAML]"
      << " [--stage6-stereo-measurement-source backend_selected_only|all_valid]"
      << " [--stage6-frame-pairing-mode exact_timestamp|frame_index]"
      << " [--stage6-frame-pairing-max-delta-ms X]"
      << " [--holdout-stride N] [--holdout-offset N]"
      << " [--stage6-max-alternating-iterations N]"
      << " [--stage6-min-total-rmse-improvement X]"
      << " [--stage6-require-symmetric-pair-refit 0|1]"
      << " [--stage6-max-graph-propagation-iterations N]"
      << " [--stage6-min-shared-boards-for-extrinsic-candidate N]"
      << " [--stage6-enable-shared-board-quality-gate]"
      << " [--stage6-disable-shared-board-quality-audit]"
      << " [--stage6-enable-shared-board-quality-hard-gate]"
      << " [--stage6-shared-board-quality-max-outer-rmse-px X]"
      << " [--stage6-shared-board-quality-min-outer-points-per-camera N]"
      << " [--stage6-shared-board-quality-min-good-shared-boards N]"
      << " [--stage6-export-pair-board-consistency-audit]"
      << " [--stage6-enable-pair-board-consistency-gate]"
      << " [--stage6-pair-board-consistency-local-good-max-outer-rmse-px X]"
      << " [--stage6-pair-board-consistency-global-bad-min-outer-rmse-px X]"
      << " [--stage6-view-selection-mode off|topk]"
      << " [--stage6-selected-pair-count N]"
      << " [--stage6-solver-mode alternating|global_sparse_ba|shared_only_global_sparse_ba]"
      << " [--stage6-intrinsics-mode fixed_stage5|kalibr_joint_projection|regularized_joint_projection|adaptive_regularized_joint_projection]"
      << " [--stage6-persistent-pose-structure independent_pair_board|shared_frame_layout]"
      << " [--stage6-ba-mode pixel|angular|hybrid_polar]"
      << " [--stage6-residual-mode pixel|spherical_chordal|spherical_tangent|hybrid_pixel_spherical]"
      << " [--stage6-final-ba-residual-mode pixel|spherical_chordal|spherical_tangent|hybrid_pixel_spherical]"
      << " [--stage6-selection-ba-residual-mode pixel|spherical_chordal|spherical_tangent|hybrid_pixel_spherical]"
      << " [--stage6-fixed-intrinsics-for-spherical true|false]"
      << " [--stage6-spherical-weight X]"
      << " [--stage6-spherical-polar-weighting true|false]"
      << " [--stage6-spherical-min-polar-deg X]"
      << " [--stage6-spherical-max-weight X]"
      << " [--stage6-spherical-uncertainty-mode none|pixel|model|pixel_model]"
      << " [--stage6-spherical-pixel-sigma-px X]"
      << " [--stage6-spherical-model-sigma xi,alpha,fu,fv,cu,cv]"
      << " [--stage6-spherical-covariance-damping X]"
      << " [--stage6-spherical-min-sigma-rad X]"
      << " [--stage6-spherical-max-whitening-weight X]"
      << " [--stage6-rig-param-mode cam0_reference|rig_centric_symmetric]"
      << " [--stage6-rig-camera-prior-translation-weight X]"
      << " [--stage6-rig-camera-prior-rotation-weight X]"
      << " [--stage6-rig-stereo-relative-prior-weight X]"
      << " [--stage6-final-ba-optimize-stereo-extrinsic true|false]"
      << " [--stage6-final-ba-optimize-pair-poses true|false]"
      << " [--stage6-final-ba-optimize-board-poses true|false]"
      << " [--stage6-final-ba-optimize-intrinsics]"
      << " [--stage6-enable-final-global-ba]"
      << " [--stage6-skip-final-global-ba]"
      << " [--stage6-ba-max-iterations N]"
      << " [--stage6-ba-convergence-threshold X]"
      << " [--stage6-ba-shared-observation-weight-scale X]"
      << " [--stage6-ba-single-camera-only-observation-weight-scale X]"
      << " [--stage6-ba-single-camera-only-weight-mode fixed_scale|per_side_budget_cap|adaptive_independent_side_cap]"
      << " [--stage6-ba-single-camera-only-base-scale X]"
      << " [--stage6-ba-single-camera-only-per-side-budget-ratio X]"
      << " [--stage6-ba-adaptive-single-camera-only-per-side-cap-ratio X]"
      << " [--stage6-enable-pair-only-stereo-ba-init]"
      << " [--stage6-disable-pair-only-stereo-ba-init]"
      << " [--stage6-pair-init-max-iterations N]"
      << " [--stage6-pair-init-convergence-threshold X]"
      << " [--stage6-pair-init-use-huber-loss 0|1]"
      << " [--stage6-enable-kalibr-style-pair-selection]"
      << " [--stage6-disable-kalibr-style-pair-selection]"
      << " [--stage6-enable-committing-pair-batch-selection]"
      << " [--stage6-enable-persistent-incremental-stereo-ba]"
      << " [--stage6-disable-persistent-incremental-stereo-ba]"
      << " [--stage6-persistent-incremental-max-iterations N]"
      << " [--stage6-persistent-incremental-convergence-delta-j X]"
      << " [--stage6-persistent-incremental-convergence-delta-x X]"
      << " [--stage6-persistent-incremental-baseline-prior-translation-weight X]"
      << " [--stage6-persistent-incremental-baseline-prior-rotation-weight X]"
      << " [--stage6-persistent-incremental-projection-prior-shape-sigma X]"
      << " [--stage6-persistent-incremental-projection-prior-focal-relative-sigma X]"
      << " [--stage6-persistent-incremental-projection-prior-principal-sigma-px X]"
      << " [--stage6-adaptive-joint-projection-min-training-pairs N]"
      << " [--stage6-adaptive-joint-projection-min-shared-pair-boards N]"
      << " [--stage6-adaptive-joint-projection-min-distinct-boards N]"
      << " [--stage6-adaptive-joint-projection-min-observation-points N]"
      << " [--stage6-persistent-incremental-invalid-projection-penalty-px X]"
      << " [--stage6-persistent-incremental-seed-pair-count N]"
      << " [--stage6-enable-stage6-incremental-estimator]"
      << " [--stage6-disable-stage6-incremental-estimator]"
      << " [--stage6-incremental-mi-tol X]"
      << " [--stage6-incremental-rank-threshold X]"
      << " [--stage6-incremental-info-block stereo_extrinsic]"
      << " [--stage6-batch-acceptance-policy residual_score|kalibr_information_gain]"
      << " [--stage6-pair-selection-seed-count N]"
      << " [--stage6-pair-selection-budget-mode fixed|adaptive|kalibr_style]"
      << " [--stage6-pair-selection-max-candidate-additions N]"
      << " [--stage6-pair-selection-adaptive-budget-ratio X]"
      << " [--stage6-pair-selection-adaptive-budget-min N]"
      << " [--stage6-pair-selection-adaptive-budget-max N]"
      << " [--stage6-pair-selection-runtime-safety-ceiling N]"
      << " [--stage6-pair-selection-min-shared-boards N]"
      << " [--stage6-pair-selection-max-rmse-delta X]"
      << " [--stage6-pair-selection-max-camera-rmse-delta X]"
      << " [--stage6-pair-selection-max-baseline-rotation-delta-deg X]"
      << " [--stage6-pair-selection-max-baseline-translation-delta-m X]"
      << " [--stage6-enable-pair-board-trial-selection]"
      << " [--stage6-disable-pair-board-trial-selection]"
      << " [--stage6-pairboard-selection-mode strict_rmse|kalibr_style_batch]"
      << " (default kalibr_style_batch; strict_rmse for ablation)"
      << " [--stage6-pair-board-selection-budget-mode fixed|adaptive|kalibr_style]"
      << " [--stage6-pair-board-selection-seed-count N]"
      << " [--stage6-pair-board-selection-max-candidate-additions N]"
      << " [--stage6-pair-board-selection-adaptive-budget-ratio X]"
      << " [--stage6-pair-board-selection-adaptive-budget-min N]"
      << " [--stage6-pair-board-selection-adaptive-budget-max N]"
      << " [--stage6-pair-board-selection-runtime-safety-ceiling N]"
      << " [--stage6-pair-board-selection-min-candidate-score X]"
      << " [--stage6-pair-board-selection-min-coverage-gain X]"
      << " [--stage6-pair-board-selection-max-accepted-per-pair N]"
      << " [--stage6-pair-board-selection-max-accepted-per-board N]"
      << " [--stage6-pair-board-selection-max-rmse-delta X]"
      << " [--stage6-pair-board-selection-max-camera-rmse-delta X]"
      << " [--stage6-pair-board-selection-max-baseline-rotation-delta-deg X]"
      << " [--stage6-pair-board-selection-max-baseline-translation-delta-m X]"
      << " [--stage6-enable-pair-cohesion]"
      << " [--stage6-pair-cohesion-min-boards-per-pair N]"
      << " [--stage6-pair-cohesion-max-companions-per-pair N]"
      << " [--stage6-pair-cohesion-relax-score-gate 0|1]"
      << " [--stage6-pair-cohesion-relax-cap-gates 0|1]"
      << " [--stage6-single-board-pair-policy keep|audit|drop|low_weight]"
      << " [--stage6-export-stereo-reprojection-visualizations]"
      << " [--stage6-stereo-visualization-top-k N]"
      << " [--stage6-export-extrinsic-uncertainty-diagnostics]"
      << " [--stage6-enable-geometry-prior-outer-seed]"
      << " [--stage6-disable-geometry-prior-outer-seed]"
      << " [--stage6-geometry-prior-rescue-diagnostic-only 0|1]"
      << " [--stage6-geometry-prior-rescue-use-as-observation]"
      << " [--stage6-geometry-prior-rescue-disable-use-as-observation]"
      << " [--stage6-geometry-prior-rescue-keep-outer-on-internal-failure]"
      << " [--stage6-geometry-prior-rescue-allow-geometry-only-pose-refit]"
      << " [--stage6-geometry-prior-rescue-disable-geometry-only-pose-refit]"
      << " [--stage6-geometry-prior-rescue-subpix-window-radius N]"
      << " [--stage6-geometry-prior-rescue-max-corner-displacement-px X]"
      << " [--stage6-geometry-prior-rescue-min-corner-response-ratio X]"
      << " [--stage6-geometry-prior-rescue-enable-spherical-refine]"
      << " [--stage6-geometry-prior-rescue-disable-spherical-refine]"
      << " [--stage6-geometry-prior-rescue-edge-sample-count N]"
      << " [--stage6-geometry-prior-rescue-edge-search-half-width-px N]"
      << " [--stage6-geometry-prior-rescue-min-edge-support-ratio X]"
      << " [--stage6-geometry-prior-rescue-min-edge-gradient-ratio X]"
      << " [--stage6-geometry-prior-rescue-accept-max-outer-rmse X]"
      << " [--stage6-geometry-prior-rescue-accept-max-rotation-error-deg X]"
      << " [--stage6-geometry-prior-rescue-accept-max-translation-error X]"
      << " [--stage6-disable-geometry-prior-rescue-for-holdout]"
      << " [--stage6-board-masking-ablation none|split_pair_boards]"
      << " [--stage6-training-pair-sample-count N]"
      << " [--stage6-training-pair-sample-seed N]"
      << " [--stage6-export-angular-fixedk-diagnostic]\n";
  std::cout
      << "\nNotes:\n"
      << "  --stage6-pair-cohesion-max-companions-per-pair 0 uses the pair's actual shared-board capacity.\n"
      << "  --stage6-stereo-visualization-top-k 0 exports all available side-by-side visualizations.\n";
}

void ApplyStage6BaMode(CmdArgs* args, const std::string& mode) {
  if (args == nullptr) {
    return;
  }
  std::string normalized = mode;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  args->stage6_ba_mode = normalized;
  args->stage6_fixed_intrinsics_for_spherical = true;
  args->stage6_spherical_uncertainty_mode = "none";
  args->stage6_spherical_use_normalize_jacobian = false;
  if (normalized == "pixel") {
    args->stage6_final_ba_residual_mode = "pixel";
    args->stage6_selection_ba_residual_mode = "pixel";
    args->stage6_spherical_weight = 1.0;
    args->stage6_spherical_polar_weighting = false;
  } else if (normalized == "angular" ||
             normalized == "spherical_tangent") {
    args->stage6_final_ba_residual_mode = "spherical_tangent";
    args->stage6_selection_ba_residual_mode = "spherical_tangent";
    args->stage6_spherical_weight = 1.0;
    args->stage6_spherical_polar_weighting = false;
  } else if (normalized == "hybrid" ||
             normalized == "hybrid_polar" ||
             normalized == "hybrid_pixel_spherical") {
    args->stage6_final_ba_residual_mode = "hybrid_pixel_spherical";
    args->stage6_selection_ba_residual_mode = "hybrid_pixel_spherical";
    args->stage6_spherical_weight = 0.25;
    args->stage6_spherical_polar_weighting = true;
    args->stage6_spherical_min_polar_deg = 50.0;
    args->stage6_spherical_max_weight = 4.0;
  } else {
    throw std::runtime_error(
        "Unsupported --stage6-ba-mode: " + mode +
        " (expected pixel, angular, or hybrid_polar)");
  }
}

CmdArgs ParseArgs(int argc, char** argv) {
  CmdArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string token = argv[i];
    if (token == "--left-image" && i + 1 < argc) {
      args.left_image_path = argv[++i];
    } else if (token == "--right-image" && i + 1 < argc) {
      args.right_image_path = argv[++i];
    } else if (token == "--test-left-image" && i + 1 < argc) {
      args.test_left_image_path = argv[++i];
    } else if (token == "--test-right-image" && i + 1 < argc) {
      args.test_right_image_path = argv[++i];
    } else if (token == "--left-config" && i + 1 < argc) {
      args.left_config_path = argv[++i];
    } else if (token == "--right-config" && i + 1 < argc) {
      args.right_config_path = argv[++i];
    } else if (token == "--left-intrinsics" && i + 1 < argc) {
      args.left_intrinsics_path = argv[++i];
    } else if (token == "--right-intrinsics" && i + 1 < argc) {
      args.right_intrinsics_path = argv[++i];
    } else if (token == "--output" && i + 1 < argc) {
      args.output_path = argv[++i];
    } else if (token == "--cache-dir" && i + 1 < argc) {
      args.cache_dir = argv[++i];
    } else if (token == "--stereo-reference-camchain" && i + 1 < argc) {
      args.stereo_reference_camchain_path = argv[++i];
    } else if (token == "--stereo-reference-left-intrinsics" &&
               i + 1 < argc) {
      args.stereo_reference_left_intrinsics_path = argv[++i];
    } else if (token == "--stereo-reference-right-intrinsics" &&
               i + 1 < argc) {
      args.stereo_reference_right_intrinsics_path = argv[++i];
    } else if (token == "--stage6-stereo-measurement-source" && i + 1 < argc) {
      args.stage6_stereo_measurement_source = argv[++i];
    } else if (token == "--stage6-frame-pairing-mode" && i + 1 < argc) {
      args.stage6_frame_pairing_mode = argv[++i];
    } else if (token == "--stage6-frame-pairing-max-delta-ms" &&
               i + 1 < argc) {
      args.stage6_frame_pairing_max_delta_ms = std::stod(argv[++i]);
    } else if (token == "--holdout-stride" && i + 1 < argc) {
      args.holdout_stride = std::stoi(argv[++i]);
    } else if (token == "--holdout-offset" && i + 1 < argc) {
      args.holdout_offset = std::stoi(argv[++i]);
    } else if (token == "--stage6-max-alternating-iterations" && i + 1 < argc) {
      args.stage6_max_alternating_iterations = std::stoi(argv[++i]);
    } else if (token == "--stage6-min-total-rmse-improvement" && i + 1 < argc) {
      args.stage6_min_total_rmse_improvement = std::stod(argv[++i]);
    } else if (token == "--stage6-require-symmetric-pair-refit" && i + 1 < argc) {
      args.stage6_require_symmetric_pair_refit = std::stoi(argv[++i]) != 0;
    } else if (token == "--stage6-max-graph-propagation-iterations" &&
               i + 1 < argc) {
      args.stage6_max_graph_propagation_iterations = std::stoi(argv[++i]);
    } else if (token == "--stage6-min-shared-boards-for-extrinsic-candidate" &&
               i + 1 < argc) {
      args.stage6_min_shared_boards_for_extrinsic_candidate = std::stoi(argv[++i]);
    } else if (token == "--stage6-enable-shared-board-quality-gate") {
      args.stage6_enable_shared_board_quality_gate = true;
    } else if (token == "--stage6-disable-shared-board-quality-audit") {
      args.stage6_enable_shared_board_quality_gate = false;
    } else if (token == "--stage6-enable-shared-board-quality-hard-gate") {
      args.stage6_enable_shared_board_quality_gate = true;
      args.stage6_enable_shared_board_quality_hard_gate = true;
    } else if (token == "--stage6-shared-board-quality-max-outer-rmse-px" &&
               i + 1 < argc) {
      args.stage6_shared_board_quality_max_outer_rmse_px = std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-shared-board-quality-min-outer-points-per-camera" &&
               i + 1 < argc) {
      args.stage6_shared_board_quality_min_outer_points_per_camera =
          std::stoi(argv[++i]);
    } else if (token == "--stage6-shared-board-quality-min-good-shared-boards" &&
               i + 1 < argc) {
      args.stage6_shared_board_quality_min_good_shared_boards =
          std::stoi(argv[++i]);
    } else if (token == "--stage6-export-pair-board-consistency-audit") {
      args.stage6_export_pair_board_consistency_audit = true;
    } else if (token == "--stage6-enable-pair-board-consistency-gate") {
      args.stage6_enable_pair_board_consistency_gate = true;
    } else if (token ==
                   "--stage6-pair-board-consistency-local-good-max-outer-rmse-px" &&
               i + 1 < argc) {
      args.stage6_pair_board_consistency_local_good_max_outer_rmse_px =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-pair-board-consistency-global-bad-min-outer-rmse-px" &&
               i + 1 < argc) {
      args.stage6_pair_board_consistency_global_bad_min_outer_rmse_px =
          std::stod(argv[++i]);
    } else if (token == "--stage6-view-selection-mode" && i + 1 < argc) {
      args.stage6_view_selection_mode = argv[++i];
    } else if (token == "--stage6-selected-pair-count" && i + 1 < argc) {
      args.stage6_selected_pair_count = std::stoi(argv[++i]);
    } else if (token == "--stage6-solver-mode" && i + 1 < argc) {
      args.stage6_solver_mode = argv[++i];
    } else if (token == "--stage6-intrinsics-mode" && i + 1 < argc) {
      args.stage6_intrinsics_mode = argv[++i];
    } else if (token == "--stage6-persistent-pose-structure" &&
               i + 1 < argc) {
      args.stage6_persistent_pose_structure = argv[++i];
    } else if (token == "--stage6-ba-mode" && i + 1 < argc) {
      ApplyStage6BaMode(&args, argv[++i]);
    } else if ((token == "--stage6-final-ba-residual-mode" ||
                token == "--stage6-residual-mode") &&
               i + 1 < argc) {
      args.stage6_final_ba_residual_mode = argv[++i];
    } else if (token == "--stage6-selection-ba-residual-mode" &&
               i + 1 < argc) {
      args.stage6_selection_ba_residual_mode = argv[++i];
    } else if (token == "--stage6-fixed-intrinsics-for-spherical" &&
               i + 1 < argc) {
      args.stage6_fixed_intrinsics_for_spherical = ParseBool(argv[++i]);
    } else if (token == "--stage6-spherical-weight" && i + 1 < argc) {
      args.stage6_spherical_weight = std::stod(argv[++i]);
    } else if (token == "--stage6-spherical-polar-weighting" &&
               i + 1 < argc) {
      args.stage6_spherical_polar_weighting = ParseBool(argv[++i]);
    } else if (token == "--stage6-spherical-min-polar-deg" &&
               i + 1 < argc) {
      args.stage6_spherical_min_polar_deg = std::stod(argv[++i]);
    } else if (token == "--stage6-spherical-max-weight" && i + 1 < argc) {
      args.stage6_spherical_max_weight = std::stod(argv[++i]);
    } else if (token == "--stage6-spherical-uncertainty-mode" &&
               i + 1 < argc) {
      args.stage6_spherical_uncertainty_mode = argv[++i];
    } else if (token == "--stage6-spherical-pixel-sigma-px" &&
               i + 1 < argc) {
      args.stage6_spherical_pixel_sigma_px = std::stod(argv[++i]);
    } else if (token == "--stage6-spherical-model-sigma" &&
               i + 1 < argc) {
      args.stage6_spherical_model_sigma = ParseSixDoubles(
          argv[++i], "--stage6-spherical-model-sigma");
    } else if (token == "--stage6-spherical-covariance-damping" &&
               i + 1 < argc) {
      args.stage6_spherical_covariance_damping = std::stod(argv[++i]);
    } else if (token == "--stage6-spherical-min-sigma-rad" &&
               i + 1 < argc) {
      args.stage6_spherical_min_sigma_rad = std::stod(argv[++i]);
    } else if (token == "--stage6-spherical-max-whitening-weight" &&
               i + 1 < argc) {
      args.stage6_spherical_max_whitening_weight = std::stod(argv[++i]);
    } else if (token == "--stage6-spherical-use-normalize-jacobian" &&
               i + 1 < argc) {
      args.stage6_spherical_use_normalize_jacobian = ParseBool(argv[++i]);
    } else if (token == "--stage6-rig-param-mode" && i + 1 < argc) {
      args.stage6_rig_param_mode = argv[++i];
    } else if (token == "--stage6-rig-camera-prior-translation-weight" &&
               i + 1 < argc) {
      args.stage6_rig_camera_prior_translation_weight = std::stod(argv[++i]);
    } else if (token == "--stage6-rig-camera-prior-rotation-weight" &&
               i + 1 < argc) {
      args.stage6_rig_camera_prior_rotation_weight = std::stod(argv[++i]);
    } else if (token == "--stage6-rig-stereo-relative-prior-weight" &&
               i + 1 < argc) {
      args.stage6_rig_stereo_relative_prior_weight = std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-enable") {
      args.stage6_coobs_enable = true;
    } else if (token == "--stage6-coobs-output-dir" && i + 1 < argc) {
      args.stage6_coobs_output_dir = argv[++i];
    } else if (token == "--stage6-coobs-min-corners-per-group" &&
               i + 1 < argc) {
      args.stage6_coobs_min_corners_per_group = std::stoi(argv[++i]);
    } else if (token == "--stage6-coobs-high-polar-threshold-deg" &&
               i + 1 < argc) {
      args.stage6_coobs_high_polar_threshold_deg = std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-very-high-polar-threshold-deg" &&
               i + 1 < argc) {
      args.stage6_coobs_very_high_polar_threshold_deg = std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-enable-rescue-suggestions" &&
               i + 1 < argc) {
      args.stage6_coobs_enable_rescue_suggestions = ParseBool(argv[++i]);
    } else if (token == "--stage6-coobs-score-alpha-high-polar" &&
               i + 1 < argc) {
      args.stage6_coobs_score_alpha_high_polar = std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-score-beta-multiboard" &&
               i + 1 < argc) {
      args.stage6_coobs_score_beta_multiboard = std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-score-gamma-balance" &&
               i + 1 < argc) {
      args.stage6_coobs_score_gamma_balance = std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-score-eta-conflict" &&
               i + 1 < argc) {
      args.stage6_coobs_score_eta_conflict = std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-rescue-min-high-polar-score" &&
               i + 1 < argc) {
      args.stage6_coobs_rescue_min_high_polar_score = std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-rescue-bad-conflict-threshold" &&
               i + 1 < argc) {
      args.stage6_coobs_rescue_bad_conflict_threshold = std::stod(argv[++i]);
    } else if (token == "--stage6-selection-coobs-factor-ba-enable") {
      args.stage6_selection_coobs_factor_ba_enable = true;
    } else if (token ==
                   "--stage6-selection-coobs-factor-ba-apply-stereo-factor" &&
               i + 1 < argc) {
      args.stage6_selection_coobs_factor_ba_apply_stereo_factor =
          ParseBool(argv[++i]);
    } else if (token ==
                   "--stage6-selection-coobs-factor-ba-apply-layout-factor" &&
               i + 1 < argc) {
      args.stage6_selection_coobs_factor_ba_apply_layout_factor =
          ParseBool(argv[++i]);
    } else if (token == "--stage6-selection-coobs-factor-ba-stereo-weight" &&
               i + 1 < argc) {
      args.stage6_selection_coobs_factor_ba_stereo_weight =
          std::stod(argv[++i]);
    } else if (token == "--stage6-selection-coobs-factor-ba-layout-weight" &&
               i + 1 < argc) {
      args.stage6_selection_coobs_factor_ba_layout_weight =
          std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-aware-acceptance-enable") {
      args.stage6_coobs_aware_acceptance_enable = true;
    } else if (token == "--stage6-coobs-aware-acceptance-min-score" &&
               i + 1 < argc) {
      args.stage6_coobs_aware_acceptance_min_score = std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-coobs-aware-acceptance-max-total-rmse-delta" &&
               i + 1 < argc) {
      args.stage6_coobs_aware_acceptance_max_total_rmse_delta =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-coobs-aware-acceptance-max-camera-rmse-delta" &&
               i + 1 < argc) {
      args.stage6_coobs_aware_acceptance_max_camera_rmse_delta =
          std::stod(argv[++i]);
    } else if (token ==
               "--stage6-coobs-aware-acceptance-balance-guard-enable") {
      args.stage6_coobs_aware_acceptance_balance_guard_enable = true;
    } else if (token ==
                   "--stage6-coobs-aware-acceptance-max-camera-delta-imbalance" &&
               i + 1 < argc) {
      args.stage6_coobs_aware_acceptance_max_camera_delta_imbalance =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-coobs-aware-acceptance-max-camera-delta-ratio" &&
               i + 1 < argc) {
      args.stage6_coobs_aware_acceptance_max_camera_delta_ratio =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-coobs-aware-acceptance-require-pair-completion" &&
               i + 1 < argc) {
      args.stage6_coobs_aware_acceptance_require_pair_completion =
          ParseBool(argv[++i]);
    } else if (token ==
                   "--stage6-coobs-aware-acceptance-stereo-rot-scale-deg" &&
               i + 1 < argc) {
      args.stage6_coobs_aware_acceptance_stereo_rot_scale_deg =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-coobs-aware-acceptance-layout-rot-scale-deg" &&
               i + 1 < argc) {
      args.stage6_coobs_aware_acceptance_layout_rot_scale_deg =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-coobs-aware-acceptance-stereo-trans-scale-m" &&
               i + 1 < argc) {
      args.stage6_coobs_aware_acceptance_stereo_trans_scale_m =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-coobs-aware-acceptance-layout-trans-scale-m" &&
               i + 1 < argc) {
      args.stage6_coobs_aware_acceptance_layout_trans_scale_m =
          std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-factor-ba-enable") {
      args.stage6_coobs_factor_ba_enable = true;
    } else if (token == "--stage6-coobs-factor-ba-run-experiment-matrix" &&
               i + 1 < argc) {
      args.stage6_coobs_factor_ba_run_experiment_matrix =
          ParseBool(argv[++i]);
    } else if (token == "--stage6-coobs-factor-ba-output-dir-suffix" &&
               i + 1 < argc) {
      args.stage6_coobs_factor_ba_output_dir_suffix = argv[++i];
    } else if (token ==
                   "--stage6-coobs-factor-ba-min-corners-per-cam-board" &&
               i + 1 < argc) {
      args.stage6_coobs_factor_ba_min_corners_per_cam_board =
          std::stoi(argv[++i]);
    } else if (token == "--stage6-coobs-factor-ba-max-local-pose-rmse" &&
               i + 1 < argc) {
      args.stage6_coobs_factor_ba_max_local_pose_rmse = std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-factor-ba-huber-delta" &&
               i + 1 < argc) {
      args.stage6_coobs_factor_ba_huber_delta = std::stod(argv[++i]);
    } else if (token == "--stage6-coobs-factor-ba-stereo-weights" &&
               i + 1 < argc) {
      args.stage6_coobs_factor_ba_stereo_weights =
          ParseDoubleList(argv[++i], token);
    } else if (token == "--stage6-coobs-factor-ba-layout-weights" &&
               i + 1 < argc) {
      args.stage6_coobs_factor_ba_layout_weights =
          ParseDoubleList(argv[++i], token);
    } else if (token ==
                   "--stage6-coobs-factor-ba-combined-stereo-weights" &&
               i + 1 < argc) {
      args.stage6_coobs_factor_ba_combined_stereo_weights =
          ParseDoubleList(argv[++i], token);
    } else if (token ==
                   "--stage6-coobs-factor-ba-combined-layout-weights" &&
               i + 1 < argc) {
      args.stage6_coobs_factor_ba_combined_layout_weights =
          ParseDoubleList(argv[++i], token);
    } else if (token == "--stage6-coobs-factor-ba-layout-selected-pairs" &&
               i + 1 < argc) {
      args.stage6_coobs_factor_ba_layout_selected_pairs =
          ParseBoardPairList(argv[++i], token);
    } else if (token == "--stage6-final-ba-optimize-intrinsics") {
      args.stage6_final_ba_optimize_intrinsics = true;
    } else if (token == "--stage6-final-ba-optimize-stereo-extrinsic" &&
               i + 1 < argc) {
      args.stage6_final_ba_optimize_stereo_extrinsic = ParseBool(argv[++i]);
    } else if (token == "--stage6-final-ba-optimize-pair-poses" &&
               i + 1 < argc) {
      args.stage6_final_ba_optimize_pair_poses = ParseBool(argv[++i]);
    } else if (token == "--stage6-final-ba-optimize-board-poses" &&
               i + 1 < argc) {
      args.stage6_final_ba_optimize_board_poses = ParseBool(argv[++i]);
    } else if (token == "--stage6-enable-final-global-ba") {
      args.stage6_skip_final_global_ba = false;
    } else if (token == "--stage6-skip-final-global-ba") {
      args.stage6_skip_final_global_ba = true;
    } else if (token == "--stage6-ba-max-iterations" && i + 1 < argc) {
      args.stage6_ba_max_iterations = std::stoi(argv[++i]);
    } else if (token == "--stage6-ba-convergence-threshold" && i + 1 < argc) {
      args.stage6_ba_convergence_threshold = std::stod(argv[++i]);
    } else if (token == "--stage6-ba-shared-observation-weight-scale" &&
               i + 1 < argc) {
      args.stage6_ba_shared_observation_weight_scale = std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-ba-single-camera-only-observation-weight-scale" &&
               i + 1 < argc) {
      args.stage6_ba_single_camera_only_observation_weight_scale =
          std::stod(argv[++i]);
    } else if (token == "--stage6-ba-single-camera-only-weight-mode" &&
               i + 1 < argc) {
      args.stage6_ba_single_camera_only_weight_mode = argv[++i];
    } else if (token == "--stage6-ba-single-camera-only-base-scale" &&
               i + 1 < argc) {
      args.stage6_ba_single_camera_only_base_scale = std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-ba-single-camera-only-per-side-budget-ratio" &&
               i + 1 < argc) {
      args.stage6_ba_single_camera_only_per_side_budget_ratio =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage6-ba-adaptive-single-camera-only-per-side-cap-ratio" &&
        i + 1 < argc) {
      args.stage6_ba_adaptive_single_camera_only_per_side_cap_ratio =
          std::stod(argv[++i]);
    } else if (token == "--stage6-enable-pair-only-stereo-ba-init") {
      args.stage6_enable_pair_only_stereo_ba_init = true;
    } else if (token == "--stage6-disable-pair-only-stereo-ba-init") {
      args.stage6_enable_pair_only_stereo_ba_init = false;
    } else if (token == "--stage6-pair-init-max-iterations" &&
               i + 1 < argc) {
      args.stage6_pair_init_max_iterations = std::stoi(argv[++i]);
    } else if (token == "--stage6-pair-init-convergence-threshold" &&
               i + 1 < argc) {
      args.stage6_pair_init_convergence_threshold = std::stod(argv[++i]);
    } else if (token == "--stage6-pair-init-use-huber-loss" &&
               i + 1 < argc) {
      args.stage6_pair_init_use_huber_loss = std::stoi(argv[++i]) != 0;
    } else if (token == "--stage6-enable-kalibr-style-pair-selection") {
      args.stage6_enable_kalibr_style_pair_selection = true;
    } else if (token == "--stage6-disable-kalibr-style-pair-selection") {
      args.stage6_enable_kalibr_style_pair_selection = false;
    } else if (token == "--stage6-enable-committing-pair-batch-selection") {
      args.stage6_enable_committing_pair_batch_selection = true;
    } else if (token ==
               "--stage6-enable-persistent-incremental-stereo-ba") {
      args.stage6_enable_persistent_incremental_stereo_ba = true;
    } else if (token ==
               "--stage6-disable-persistent-incremental-stereo-ba") {
      args.stage6_enable_persistent_incremental_stereo_ba = false;
    } else if (token ==
               "--stage6-allow-legacy-selection-fallback") {
      args.stage6_allow_legacy_selection_fallback_after_persistent_failure = true;
    } else if (token == "--stage6-enable-stage6-incremental-estimator") {
      args.stage6_enable_stage6_incremental_estimator = true;
    } else if (token == "--stage6-disable-stage6-incremental-estimator") {
      args.stage6_enable_stage6_incremental_estimator = false;
    } else if (token ==
               "--stage6-enable-incremental-pair-diversity-rescue") {
      args.stage6_enable_incremental_pair_diversity_rescue = true;
    } else if (token ==
                   "--stage6-incremental-pair-diversity-rescue-min-boards" &&
               i + 1 < argc) {
      args.stage6_incremental_pair_diversity_rescue_min_boards =
          std::stoi(argv[++i]);
    } else if (token == "--stage6-incremental-mi-tol" && i + 1 < argc) {
      args.stage6_incremental_mi_tol = std::stod(argv[++i]);
    } else if (token == "--stage6-incremental-rank-threshold" &&
               i + 1 < argc) {
      args.stage6_incremental_rank_threshold = std::stod(argv[++i]);
    } else if (token == "--stage6-persistent-incremental-max-iterations" &&
               i + 1 < argc) {
      args.stage6_persistent_incremental_max_iterations = std::stoi(argv[++i]);
    } else if (token ==
                   "--stage6-persistent-incremental-convergence-delta-j" &&
               i + 1 < argc) {
      args.stage6_persistent_incremental_convergence_delta_j =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-persistent-incremental-convergence-delta-x" &&
               i + 1 < argc) {
      args.stage6_persistent_incremental_convergence_delta_x =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage6-persistent-incremental-baseline-prior-translation-weight" &&
        i + 1 < argc) {
      args.stage6_persistent_incremental_baseline_prior_translation_weight =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage6-persistent-incremental-baseline-prior-rotation-weight" &&
        i + 1 < argc) {
      args.stage6_persistent_incremental_baseline_prior_rotation_weight =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage6-persistent-incremental-projection-prior-shape-sigma" &&
        i + 1 < argc) {
      args.stage6_persistent_incremental_projection_prior_shape_sigma =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage6-persistent-incremental-projection-prior-focal-relative-sigma" &&
        i + 1 < argc) {
      args.stage6_persistent_incremental_projection_prior_focal_relative_sigma =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage6-persistent-incremental-projection-prior-principal-sigma-px" &&
        i + 1 < argc) {
      args.stage6_persistent_incremental_projection_prior_principal_sigma_px =
          std::stod(argv[++i]);
    } else if (
        token == "--stage6-adaptive-joint-projection-min-training-pairs" &&
        i + 1 < argc) {
      args.stage6_adaptive_joint_projection_min_training_pairs =
          std::stoi(argv[++i]);
    } else if (
        token ==
            "--stage6-adaptive-joint-projection-min-shared-pair-boards" &&
        i + 1 < argc) {
      args.stage6_adaptive_joint_projection_min_shared_pair_boards =
          std::stoi(argv[++i]);
    } else if (
        token == "--stage6-adaptive-joint-projection-min-distinct-boards" &&
        i + 1 < argc) {
      args.stage6_adaptive_joint_projection_min_distinct_boards =
          std::stoi(argv[++i]);
    } else if (
        token ==
            "--stage6-adaptive-joint-projection-min-observation-points" &&
        i + 1 < argc) {
      args.stage6_adaptive_joint_projection_min_observation_points =
          std::stoi(argv[++i]);
    } else if (
        token ==
            "--stage6-persistent-incremental-invalid-projection-penalty-px" &&
        i + 1 < argc) {
      args.stage6_persistent_incremental_invalid_projection_penalty_px =
          std::stod(argv[++i]);
    } else if (token == "--stage6-persistent-incremental-seed-pair-count" &&
               i + 1 < argc) {
      args.stage6_persistent_incremental_seed_pair_count =
          std::stoi(argv[++i]);
    } else if (token == "--stage6-incremental-info-block" &&
               i + 1 < argc) {
      args.stage6_incremental_info_block = argv[++i];
    } else if (token == "--stage6-batch-acceptance-policy" &&
               i + 1 < argc) {
      args.stage6_batch_acceptance_policy = argv[++i];
    } else if (token == "--stage6-pair-selection-seed-count" &&
               i + 1 < argc) {
      args.stage6_pair_selection_seed_count = std::stoi(argv[++i]);
    } else if (token == "--stage6-pair-selection-budget-mode" &&
               i + 1 < argc) {
      args.stage6_pair_selection_budget_mode = argv[++i];
      args.stage6_pair_selection_budget_mode_set = true;
    } else if (token == "--stage6-pair-selection-max-candidate-additions" &&
               i + 1 < argc) {
      args.stage6_pair_selection_max_candidate_additions = std::stoi(argv[++i]);
    } else if (token == "--stage6-pair-selection-adaptive-budget-ratio" &&
               i + 1 < argc) {
      args.stage6_pair_selection_adaptive_budget_ratio = std::stod(argv[++i]);
    } else if (token == "--stage6-pair-selection-adaptive-budget-min" &&
               i + 1 < argc) {
      args.stage6_pair_selection_adaptive_budget_min = std::stoi(argv[++i]);
    } else if (token == "--stage6-pair-selection-adaptive-budget-max" &&
               i + 1 < argc) {
      args.stage6_pair_selection_adaptive_budget_max = std::stoi(argv[++i]);
    } else if (token == "--stage6-pair-selection-runtime-safety-ceiling" &&
               i + 1 < argc) {
      args.stage6_pair_selection_runtime_safety_ceiling =
          std::stoi(argv[++i]);
    } else if (token == "--stage6-pair-selection-min-shared-boards" &&
               i + 1 < argc) {
      args.stage6_pair_selection_min_shared_boards = std::stoi(argv[++i]);
    } else if (token == "--stage6-pair-selection-max-rmse-delta" &&
               i + 1 < argc) {
      args.stage6_pair_selection_max_rmse_delta = std::stod(argv[++i]);
    } else if (token == "--stage6-pair-selection-max-camera-rmse-delta" &&
               i + 1 < argc) {
      args.stage6_pair_selection_max_camera_rmse_delta = std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-pair-selection-max-baseline-rotation-delta-deg" &&
               i + 1 < argc) {
      args.stage6_pair_selection_max_baseline_rotation_delta_deg =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-pair-selection-max-baseline-translation-delta-m" &&
               i + 1 < argc) {
      args.stage6_pair_selection_max_baseline_translation_delta_m =
          std::stod(argv[++i]);
    } else if (token == "--stage6-enable-pair-board-trial-selection") {
      args.stage6_enable_pair_board_trial_selection = true;
    } else if (token == "--stage6-disable-pair-board-trial-selection") {
      args.stage6_enable_pair_board_trial_selection = false;
    } else if (token == "--stage6-pairboard-selection-mode" &&
               i + 1 < argc) {
      args.stage6_pairboard_selection_mode = argv[++i];
    } else if (token == "--stage6-pair-board-selection-budget-mode" &&
               i + 1 < argc) {
      args.stage6_pair_board_selection_budget_mode = argv[++i];
      args.stage6_pair_board_selection_budget_mode_set = true;
    } else if (token == "--stage6-pair-board-selection-seed-count" &&
               i + 1 < argc) {
      args.stage6_pair_board_selection_seed_count = std::stoi(argv[++i]);
    } else if (token ==
                   "--stage6-pair-board-selection-max-candidate-additions" &&
               i + 1 < argc) {
      args.stage6_pair_board_selection_max_candidate_additions =
          std::stoi(argv[++i]);
    } else if (
        token == "--stage6-pair-board-selection-adaptive-budget-ratio" &&
        i + 1 < argc) {
      args.stage6_pair_board_selection_adaptive_budget_ratio =
          std::stod(argv[++i]);
    } else if (
        token == "--stage6-pair-board-selection-adaptive-budget-min" &&
        i + 1 < argc) {
      args.stage6_pair_board_selection_adaptive_budget_min =
          std::stoi(argv[++i]);
    } else if (
        token == "--stage6-pair-board-selection-adaptive-budget-max" &&
        i + 1 < argc) {
      args.stage6_pair_board_selection_adaptive_budget_max =
          std::stoi(argv[++i]);
    } else if (
        token == "--stage6-pair-board-selection-runtime-safety-ceiling" &&
        i + 1 < argc) {
      args.stage6_pair_board_selection_runtime_safety_ceiling =
          std::stoi(argv[++i]);
    } else if (token == "--stage6-pair-board-selection-min-candidate-score" &&
               i + 1 < argc) {
      args.stage6_pair_board_selection_min_candidate_score =
          std::stod(argv[++i]);
    } else if (token == "--stage6-pair-board-selection-min-coverage-gain" &&
               i + 1 < argc) {
      args.stage6_pair_board_selection_min_coverage_gain =
          std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-pair-board-selection-max-accepted-per-pair" &&
               i + 1 < argc) {
      args.stage6_pair_board_selection_max_accepted_per_pair =
          std::stoi(argv[++i]);
    } else if (token ==
                   "--stage6-pair-board-selection-max-accepted-per-board" &&
               i + 1 < argc) {
      args.stage6_pair_board_selection_max_accepted_per_board =
          std::stoi(argv[++i]);
    } else if (token == "--stage6-pair-board-selection-max-rmse-delta" &&
               i + 1 < argc) {
      args.stage6_pair_board_selection_max_rmse_delta = std::stod(argv[++i]);
    } else if (token ==
                   "--stage6-pair-board-selection-max-camera-rmse-delta" &&
               i + 1 < argc) {
      args.stage6_pair_board_selection_max_camera_rmse_delta =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage6-pair-board-selection-max-baseline-rotation-delta-deg" &&
        i + 1 < argc) {
      args.stage6_pair_board_selection_max_baseline_rotation_delta_deg =
          std::stod(argv[++i]);
    } else if (
        token ==
            "--stage6-pair-board-selection-max-baseline-translation-delta-m" &&
        i + 1 < argc) {
      args.stage6_pair_board_selection_max_baseline_translation_delta_m =
          std::stod(argv[++i]);
    } else if (token == "--stage6-enable-pair-cohesion") {
      args.stage6_enable_pair_cohesion = true;
    } else if (token == "--stage6-pair-cohesion-min-boards-per-pair" &&
               i + 1 < argc) {
      args.stage6_pair_cohesion_min_boards_per_pair = std::stoi(argv[++i]);
    } else if (token == "--stage6-pair-cohesion-max-companions-per-pair" &&
               i + 1 < argc) {
      args.stage6_pair_cohesion_max_companions_per_pair = std::stoi(argv[++i]);
    } else if (token == "--stage6-pair-cohesion-relax-score-gate" &&
               i + 1 < argc) {
      args.stage6_pair_cohesion_relax_score_gate = std::stoi(argv[++i]) != 0;
    } else if (token == "--stage6-pair-cohesion-relax-cap-gates" &&
               i + 1 < argc) {
      args.stage6_pair_cohesion_relax_cap_gates = std::stoi(argv[++i]) != 0;
    } else if (token == "--stage6-single-board-pair-policy" &&
               i + 1 < argc) {
      args.stage6_single_board_pair_policy = argv[++i];
    } else if (token == "--stage6-ablation-exclude-pair-board" &&
               i + 1 < argc) {
      args.stage6_ablation_excluded_pair_boards.push_back(
          ParsePairBoardKey(argv[++i], token));
    } else if (token == "--stage6-export-stereo-reprojection-visualizations") {
      args.stage6_export_stereo_reprojection_visualizations = true;
    } else if (token == "--stage6-stereo-visualization-top-k" &&
               i + 1 < argc) {
      args.stage6_stereo_visualization_top_k = std::stoi(argv[++i]);
      if (args.stage6_stereo_visualization_top_k < 0) {
        throw std::runtime_error(
            "--stage6-stereo-visualization-top-k must be >= 0; use 0 to export all.");
      }
    } else if (token == "--stage6-export-extrinsic-uncertainty-diagnostics") {
      args.stage6_export_extrinsic_uncertainty_diagnostics = true;
    } else if (token == "--stage6-enable-geometry-prior-outer-seed") {
      args.stage6_enable_geometry_prior_outer_seed = true;
    } else if (token == "--stage6-disable-geometry-prior-outer-seed") {
      args.stage6_enable_geometry_prior_outer_seed = false;
    } else if (token == "--stage6-geometry-prior-rescue-diagnostic-only" &&
               i + 1 < argc) {
      args.stage6_geometry_prior_rescue_diagnostic_only = ParseBool(argv[++i]);
    } else if (token == "--stage6-geometry-prior-rescue-use-as-observation") {
      args.stage6_geometry_prior_rescue_use_as_observation = true;
    } else if (token == "--stage6-geometry-prior-rescue-disable-use-as-observation") {
      args.stage6_geometry_prior_rescue_use_as_observation = false;
    } else if (token == "--stage6-geometry-prior-rescue-keep-outer-on-internal-failure") {
      args.stage6_geometry_prior_rescue_keep_outer_on_internal_failure = true;
    } else if (token == "--stage6-geometry-prior-rescue-allow-geometry-only-pose-refit") {
      args.stage6_geometry_prior_rescue_allow_geometry_only_pose_refit = true;
    } else if (token == "--stage6-geometry-prior-rescue-disable-geometry-only-pose-refit") {
      args.stage6_geometry_prior_rescue_allow_geometry_only_pose_refit = false;
    } else if (token == "--stage6-geometry-prior-rescue-subpix-window-radius" &&
               i + 1 < argc) {
      args.stage6_geometry_prior_rescue_subpix_window_radius = std::stoi(argv[++i]);
    } else if (token == "--stage6-geometry-prior-rescue-max-corner-displacement-px" &&
               i + 1 < argc) {
      args.stage6_geometry_prior_rescue_max_corner_displacement_px = std::stod(argv[++i]);
    } else if (token == "--stage6-geometry-prior-rescue-min-corner-response-ratio" &&
               i + 1 < argc) {
      args.stage6_geometry_prior_rescue_min_corner_response_ratio = std::stod(argv[++i]);
    } else if (token == "--stage6-geometry-prior-rescue-enable-spherical-refine") {
      args.stage6_geometry_prior_rescue_enable_spherical_refine = true;
    } else if (token == "--stage6-geometry-prior-rescue-disable-spherical-refine") {
      args.stage6_geometry_prior_rescue_enable_spherical_refine = false;
    } else if (token == "--stage6-geometry-prior-rescue-edge-sample-count" &&
               i + 1 < argc) {
      args.stage6_geometry_prior_rescue_edge_sample_count = std::stoi(argv[++i]);
    } else if (token == "--stage6-geometry-prior-rescue-edge-search-half-width-px" &&
               i + 1 < argc) {
      args.stage6_geometry_prior_rescue_edge_search_half_width_px = std::stoi(argv[++i]);
    } else if (token == "--stage6-geometry-prior-rescue-min-edge-support-ratio" &&
               i + 1 < argc) {
      args.stage6_geometry_prior_rescue_min_edge_support_ratio = std::stod(argv[++i]);
    } else if (token == "--stage6-geometry-prior-rescue-min-edge-gradient-ratio" &&
               i + 1 < argc) {
      args.stage6_geometry_prior_rescue_min_edge_gradient_ratio = std::stod(argv[++i]);
    } else if (token == "--stage6-geometry-prior-rescue-accept-max-outer-rmse" &&
               i + 1 < argc) {
      args.stage6_geometry_prior_rescue_accept_max_outer_rmse = std::stod(argv[++i]);
    } else if (token == "--stage6-geometry-prior-rescue-accept-max-rotation-error-deg" &&
               i + 1 < argc) {
      args.stage6_geometry_prior_rescue_accept_max_rotation_error_deg = std::stod(argv[++i]);
    } else if (token == "--stage6-geometry-prior-rescue-accept-max-translation-error" &&
               i + 1 < argc) {
      args.stage6_geometry_prior_rescue_accept_max_translation_error = std::stod(argv[++i]);
    } else if (token == "--stage6-disable-geometry-prior-rescue-for-holdout") {
      args.stage6_disable_geometry_prior_rescue_for_holdout = true;
    } else if (token == "--stage6-board-masking-ablation" &&
               i + 1 < argc) {
      args.stage6_board_masking_ablation = argv[++i];
    } else if (token == "--stage6-training-pair-sample-count" &&
               i + 1 < argc) {
      args.stage6_training_pair_sample_count = std::stoi(argv[++i]);
    } else if (token == "--stage6-training-pair-sample-seed" &&
               i + 1 < argc) {
      args.stage6_training_pair_sample_seed = std::stoi(argv[++i]);
    } else if (token == "--stage6-export-angular-fixedk-diagnostic") {
      args.stage6_export_angular_fixedk_diagnostic = true;
    } else if (token == "--help" || token == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + token);
    }
  }
  const bool has_test_left = !args.test_left_image_path.empty();
  const bool has_test_right = !args.test_right_image_path.empty();
  if (has_test_left != has_test_right) {
    throw std::runtime_error(
        "--test-left-image and --test-right-image must be provided together.");
  }
  const bool has_reference_left =
      !args.stereo_reference_left_intrinsics_path.empty();
  const bool has_reference_right =
      !args.stereo_reference_right_intrinsics_path.empty();
  if (has_reference_left != has_reference_right) {
    throw std::runtime_error(
        "--stereo-reference-left-intrinsics and "
        "--stereo-reference-right-intrinsics must be provided together.");
  }
  if (args.left_image_path.empty() || args.right_image_path.empty() ||
      args.left_config_path.empty() || args.right_config_path.empty() ||
      args.left_intrinsics_path.empty() || args.right_intrinsics_path.empty() ||
      args.output_path.empty()) {
    throw std::runtime_error(
        "All left/right image/config/intrinsics paths and --output are required.");
  }
  return args;
}

bool IsImageFile(const fs::path& path) {
  if (!fs::is_regular_file(path)) {
    return false;
  }
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
         extension == ".bmp" || extension == ".tif" || extension == ".tiff";
}

std::vector<std::string> CollectImagePaths(const std::string& image_path) {
  const fs::path input(image_path);
  if (!fs::exists(input)) {
    throw std::runtime_error("Input path does not exist: " + image_path);
  }
  fs::path directory = input;
  if (fs::is_regular_file(input)) {
    directory = input.parent_path();
  }
  if (!fs::is_directory(directory)) {
    throw std::runtime_error("Expected image path to point to a directory.");
  }
  std::vector<std::string> image_paths;
  for (fs::directory_iterator it(directory), end; it != end; ++it) {
    if (IsImageFile(it->path())) {
      image_paths.push_back(it->path().string());
    }
  }
  std::sort(image_paths.begin(), image_paths.end());
  if (image_paths.empty()) {
    throw std::runtime_error("No image files found in directory: " + directory.string());
  }
  return image_paths;
}

std::vector<ati::FrozenRound2BaselineFrameSource> BuildFrameSources(
    const std::vector<std::string>& image_paths) {
  std::vector<ati::FrozenRound2BaselineFrameSource> frame_sources;
  frame_sources.reserve(image_paths.size());
  for (std::size_t index = 0; index < image_paths.size(); ++index) {
    ati::FrozenRound2BaselineFrameSource source;
    source.frame_index = static_cast<int>(index);
    source.frame_label = fs::path(image_paths[index]).stem().string();
    source.image_path = image_paths[index];
    frame_sources.push_back(source);
  }
  return frame_sources;
}

void EnsureDirectoryExists(const fs::path& directory) {
  if (!directory.empty()) {
    fs::create_directories(directory);
  }
}

double RotationAngleDegrees(const Eigen::Matrix3d& rotation) {
  const double trace = std::max(-1.0, std::min(3.0, rotation.trace()));
  const double cos_theta = std::max(-1.0, std::min(1.0, 0.5 * (trace - 1.0)));
  return std::acos(cos_theta) * 180.0 / M_PI;
}

struct StereoReferenceComparison {
  bool success = false;
  std::string reference_camchain_path;
  Eigen::Matrix4d reference_T_cam1_cam0 = Eigen::Matrix4d::Identity();
  double rotation_delta_deg = std::numeric_limits<double>::infinity();
  double translation_delta_m = std::numeric_limits<double>::infinity();
  double baseline_length_reference = 0.0;
  double baseline_length_estimated = 0.0;
  double baseline_length_delta_m = std::numeric_limits<double>::infinity();
  std::string failure_reason;
};

StereoReferenceComparison CompareAgainstStereoReference(
    const std::string& camchain_path,
    const Eigen::Matrix4d& estimated_T_cam1_cam0) {
  StereoReferenceComparison comparison;
  comparison.reference_camchain_path = camchain_path;
  if (camchain_path.empty()) {
    comparison.failure_reason = "missing_reference_camchain";
    return comparison;
  }
  std::ifstream input(camchain_path.c_str());
  if (!input.is_open()) {
    comparison.failure_reason = "failed_to_open_reference_camchain: " +
                                camchain_path;
    return comparison;
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input, line)) {
    lines.push_back(line);
  }
  std::vector<std::vector<double> > rows;
  bool in_cam1 = false;
  bool found_transform = false;
  const std::regex number_regex(
      "[-+]?(?:\\d*\\.\\d+|\\d+)(?:[eE][-+]?\\d+)?");
  for (std::size_t index = 0; index < lines.size(); ++index) {
    std::string stripped = lines[index];
    const std::size_t comment = stripped.find('#');
    if (comment != std::string::npos) {
      stripped = stripped.substr(0, comment);
    }
    if (stripped.find("cam1:") != std::string::npos &&
        stripped.find_first_not_of(" \t") == stripped.find("cam1:")) {
      in_cam1 = true;
      continue;
    }
    if (in_cam1 && stripped.find_first_not_of(" \t") == 0 &&
        stripped.find("cam") == 0 && stripped.find("cam1:") == std::string::npos) {
      in_cam1 = false;
    }
    if (!in_cam1 || stripped.find("T_cn_cnm1") == std::string::npos) {
      continue;
    }
    found_transform = true;
    for (std::size_t row_index = index + 1;
         row_index < lines.size() && rows.size() < 4; ++row_index) {
      std::vector<double> values;
      const std::string row_line = lines[row_index];
      for (std::sregex_iterator it(row_line.begin(), row_line.end(),
                                   number_regex),
           end;
           it != end; ++it) {
        values.push_back(std::stod(it->str()));
      }
      if (values.size() == 4) {
        rows.push_back(values);
      }
    }
    break;
  }
  if (!found_transform || rows.size() != 4) {
    comparison.failure_reason =
        "reference_camchain_missing_valid_cam1_T_cn_cnm1";
    return comparison;
  }
  Eigen::Matrix4d reference = Eigen::Matrix4d::Identity();
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      reference(r, c) = rows[static_cast<std::size_t>(r)]
                            [static_cast<std::size_t>(c)];
    }
  }
  comparison.reference_T_cam1_cam0 = reference;
  const Eigen::Matrix3d delta_rotation =
      reference.block<3, 3>(0, 0).transpose() *
      estimated_T_cam1_cam0.block<3, 3>(0, 0);
  comparison.rotation_delta_deg = RotationAngleDegrees(delta_rotation);
  comparison.translation_delta_m =
      (reference.block<3, 1>(0, 3) - estimated_T_cam1_cam0.block<3, 1>(0, 3))
          .norm();
  comparison.baseline_length_reference = reference.block<3, 1>(0, 3).norm();
  comparison.baseline_length_estimated =
      estimated_T_cam1_cam0.block<3, 1>(0, 3).norm();
  comparison.baseline_length_delta_m =
      std::abs(comparison.baseline_length_reference -
               comparison.baseline_length_estimated);
  comparison.success = true;
  return comparison;
}

void WriteStereoReferenceComparison(const std::string& path,
                                    const StereoReferenceComparison& comparison) {
  std::ofstream output(path.c_str());
  output << "success: " << (comparison.success ? 1 : 0) << "\n";
  output << "failure_reason: " << comparison.failure_reason << "\n";
  output << "reference_camchain_path: " << comparison.reference_camchain_path
         << "\n";
  output << "rotation_delta_deg: " << comparison.rotation_delta_deg << "\n";
  output << "translation_delta_m: " << comparison.translation_delta_m << "\n";
  output << "baseline_length_reference: "
         << comparison.baseline_length_reference << "\n";
  output << "baseline_length_estimated: "
         << comparison.baseline_length_estimated << "\n";
  output << "baseline_length_delta_m: "
         << comparison.baseline_length_delta_m << "\n";
  output << "reference_translation_xyz: ["
         << comparison.reference_T_cam1_cam0(0, 3) << ", "
         << comparison.reference_T_cam1_cam0(1, 3) << ", "
         << comparison.reference_T_cam1_cam0(2, 3) << "]\n";
}

void WriteStereoReferenceHoldoutSummary(
    const std::string& path,
    const StereoReferenceComparison& comparison,
    const ati::StereoResidualSummary& ours_extrinsic_only,
    const ati::StereoResidualSummary& reference_extrinsic_only) {
  std::ofstream output(path.c_str());
  output << "success: "
         << (comparison.success && reference_extrinsic_only.success ? 1 : 0)
         << "\n";
  output << "failure_reason: "
         << (comparison.success ? reference_extrinsic_only.failure_reason
                                : comparison.failure_reason)
         << "\n";
  output << "reference_camchain_path: " << comparison.reference_camchain_path
         << "\n";
  output << "comparison_metric: extrinsic_only_holdout\n";
  output << "ours_extrinsic_only_holdout_total_stereo_rmse: "
         << ours_extrinsic_only.total_stereo_rmse << "\n";
  output << "reference_extrinsic_only_holdout_total_stereo_rmse: "
         << reference_extrinsic_only.total_stereo_rmse << "\n";
  output << "ours_minus_reference_extrinsic_only_holdout_total_stereo_rmse: "
         << (ours_extrinsic_only.total_stereo_rmse -
             reference_extrinsic_only.total_stereo_rmse) << "\n";
  output << "ours_extrinsic_only_holdout_cam0_rmse: "
         << ours_extrinsic_only.cam0_rmse << "\n";
  output << "reference_extrinsic_only_holdout_cam0_rmse: "
         << reference_extrinsic_only.cam0_rmse << "\n";
  output << "ours_extrinsic_only_holdout_cam1_rmse: "
         << ours_extrinsic_only.cam1_rmse << "\n";
  output << "reference_extrinsic_only_holdout_cam1_rmse: "
         << reference_extrinsic_only.cam1_rmse << "\n";
  output << "ours_extrinsic_only_holdout_outer_only_rmse: "
         << ours_extrinsic_only.outer_only_rmse << "\n";
  output << "reference_extrinsic_only_holdout_outer_only_rmse: "
         << reference_extrinsic_only.outer_only_rmse << "\n";
  output << "ours_extrinsic_only_holdout_internal_only_rmse: "
         << ours_extrinsic_only.internal_only_rmse << "\n";
  output << "reference_extrinsic_only_holdout_internal_only_rmse: "
         << reference_extrinsic_only.internal_only_rmse << "\n";
  output << "extrinsic_only_holdout_used_pair_count: "
         << ours_extrinsic_only.used_pair_count << "\n";
  output << "reference_extrinsic_only_holdout_used_pair_count: "
         << reference_extrinsic_only.used_pair_count << "\n";
  output << "extrinsic_only_holdout_mode: local_stereo_board_pose_refit\n";
}

void WriteStereoFullReferenceHoldoutSummary(
    const std::string& path,
    const StereoReferenceComparison& comparison,
    const std::string& reference_left_intrinsics_path,
    const std::string& reference_right_intrinsics_path,
    const ati::StereoResidualSummary& ours,
    const ati::StereoResidualSummary& reference_full) {
  std::ofstream output(path.c_str());
  output << "success: "
         << (comparison.success && reference_full.success ? 1 : 0) << "\n";
  output << "failure_reason: "
         << (comparison.success ? reference_full.failure_reason
                                : comparison.failure_reason)
         << "\n";
  output << "reference_camchain_path: " << comparison.reference_camchain_path
         << "\n";
  output << "reference_left_intrinsics_path: "
         << reference_left_intrinsics_path << "\n";
  output << "reference_right_intrinsics_path: "
         << reference_right_intrinsics_path << "\n";
  output << "comparison_metric: full_camera_frozen_measurement_holdout\n";
  output << "measurements_reused_without_redetection: 1\n";
  output << "ours_holdout_total_stereo_rmse: "
         << ours.total_stereo_rmse << "\n";
  output << "reference_full_holdout_total_stereo_rmse: "
         << reference_full.total_stereo_rmse << "\n";
  output << "ours_minus_reference_full_holdout_total_stereo_rmse: "
         << (ours.total_stereo_rmse - reference_full.total_stereo_rmse) << "\n";
  output << "ours_holdout_cam0_rmse: " << ours.cam0_rmse << "\n";
  output << "reference_full_holdout_cam0_rmse: "
         << reference_full.cam0_rmse << "\n";
  output << "ours_holdout_cam1_rmse: " << ours.cam1_rmse << "\n";
  output << "reference_full_holdout_cam1_rmse: "
         << reference_full.cam1_rmse << "\n";
  output << "ours_holdout_outer_only_rmse: " << ours.outer_only_rmse << "\n";
  output << "reference_full_holdout_outer_only_rmse: "
         << reference_full.outer_only_rmse << "\n";
  output << "ours_holdout_internal_only_rmse: "
         << ours.internal_only_rmse << "\n";
  output << "reference_full_holdout_internal_only_rmse: "
         << reference_full.internal_only_rmse << "\n";
  output << "ours_holdout_used_pair_count: " << ours.used_pair_count << "\n";
  output << "reference_full_holdout_used_pair_count: "
         << reference_full.used_pair_count << "\n";
  output << "holdout_mode: local_stereo_board_pose_refit\n";
}

void WriteStereoResidualPerBoardCsv(
    const std::string& path,
    const std::string& split_label,
    const ati::StereoResidualSummary& summary) {
  std::ofstream output(path.c_str());
  output << "split,board_id,point_count,cam0_point_count,cam1_point_count,"
         << "shared_pair_count,shared_point_count,shared_cam0_point_count,"
         << "shared_cam1_point_count,shared_outer_point_count,"
         << "shared_internal_point_count,rmse,shared_cam0_rmse,"
         << "shared_cam1_rmse,shared_outer_rmse,shared_internal_rmse\n";
  for (const ati::StereoBoardResidualSummary& board :
       summary.board_summaries) {
    output << split_label << "," << board.board_id << ","
           << board.point_count << "," << board.cam0_point_count << ","
           << board.cam1_point_count << "," << board.shared_pair_count << ","
           << board.shared_point_count << ","
           << board.shared_cam0_point_count << ","
           << board.shared_cam1_point_count << ","
           << board.shared_outer_point_count << ","
           << board.shared_internal_point_count << "," << board.rmse << ","
           << board.shared_cam0_rmse << "," << board.shared_cam1_rmse << ","
           << board.shared_outer_rmse << ","
           << board.shared_internal_rmse << "\n";
  }
}

template <typename T>
std::map<int, T> ShiftMapKeys(const std::map<int, T>& input, int key_offset) {
  std::map<int, T> output;
  for (const auto& entry : input) {
    output[entry.first + key_offset] = entry.second;
  }
  return output;
}

template <typename T>
void InsertShiftedMapEntries(std::map<int, T>* destination,
                             const std::map<int, T>& source,
                             int key_offset) {
  if (destination == nullptr) {
    throw std::runtime_error("InsertShiftedMapEntries requires destination.");
  }
  for (const auto& entry : source) {
    (*destination)[entry.first + key_offset] = entry.second;
  }
}

ati::StereoMeasurementDataset MergeTrainAndTestStereoDatasets(
    const ati::StereoMeasurementDataset& training_dataset,
    const ati::StereoMeasurementDataset& holdout_dataset) {
  ati::StereoMeasurementDataset merged = training_dataset;
  if (!training_dataset.success || !holdout_dataset.success) {
    merged.success = false;
    merged.failure_reason =
        !training_dataset.success ? training_dataset.failure_reason
                                  : holdout_dataset.failure_reason;
    return merged;
  }

  const int pair_offset = static_cast<int>(training_dataset.frame_pairs.size());
  const int frame_offset = training_dataset.paired_frame_count;
  merged.left_frame_count =
      training_dataset.left_frame_count + holdout_dataset.left_frame_count;
  merged.right_frame_count =
      training_dataset.right_frame_count + holdout_dataset.right_frame_count;
  merged.paired_frame_count =
      training_dataset.paired_frame_count + holdout_dataset.paired_frame_count;
  merged.unmatched_left_count =
      training_dataset.unmatched_left_count + holdout_dataset.unmatched_left_count;
  merged.unmatched_right_count =
      training_dataset.unmatched_right_count + holdout_dataset.unmatched_right_count;
  merged.shared_board_observation_count +=
      holdout_dataset.shared_board_observation_count;
  merged.cam0_only_board_observation_count +=
      holdout_dataset.cam0_only_board_observation_count;
  merged.cam1_only_board_observation_count +=
      holdout_dataset.cam1_only_board_observation_count;

  for (const ati::StereoFramePair& input_pair : holdout_dataset.frame_pairs) {
    ati::StereoFramePair pair = input_pair;
    pair.pair_index += pair_offset;
    pair.left_frame_index += frame_offset;
    pair.right_frame_index += frame_offset;
    pair.is_training = false;
    merged.frame_pairs.push_back(pair);
  }
  for (const ati::StereoObservation& input_observation :
       holdout_dataset.observations) {
    ati::StereoObservation observation = input_observation;
    observation.pair_index += pair_offset;
    observation.frame_index += frame_offset;
    merged.observations.push_back(observation);
  }

  for (int pair_index : holdout_dataset.holdout_pair_indices) {
    merged.holdout_pair_indices.push_back(pair_index + pair_offset);
  }
  InsertShiftedMapEntries(&merged.pair_shared_board_count,
                          holdout_dataset.pair_shared_board_count, pair_offset);
  InsertShiftedMapEntries(&merged.pair_cam0_only_board_count,
                          holdout_dataset.pair_cam0_only_board_count, pair_offset);
  InsertShiftedMapEntries(&merged.pair_cam1_only_board_count,
                          holdout_dataset.pair_cam1_only_board_count, pair_offset);
  InsertShiftedMapEntries(&merged.pair_shared_board_ids,
                          holdout_dataset.pair_shared_board_ids, pair_offset);
  InsertShiftedMapEntries(&merged.pair_cam0_only_board_ids,
                          holdout_dataset.pair_cam0_only_board_ids, pair_offset);
  InsertShiftedMapEntries(&merged.pair_cam1_only_board_ids,
                          holdout_dataset.pair_cam1_only_board_ids, pair_offset);
  InsertShiftedMapEntries(&merged.holdout_pair_board_ids,
                          holdout_dataset.holdout_pair_board_ids, pair_offset);
  merged.warnings.insert(merged.warnings.end(),
                         holdout_dataset.warnings.begin(),
                         holdout_dataset.warnings.end());
  merged.success = true;
  return merged;
}

void RestoreFrontendPairingPopulationCounts(
    const ati::StereoFrontendImagePairSelection& selection,
    ati::StereoMeasurementDataset* dataset) {
  if (dataset == nullptr) {
    throw std::runtime_error(
        "RestoreFrontendPairingPopulationCounts requires a dataset.");
  }
  dataset->left_frame_count = selection.original_left_frame_count;
  dataset->right_frame_count = selection.original_right_frame_count;
  dataset->unmatched_left_count = selection.unmatched_left_count;
  dataset->unmatched_right_count = selection.unmatched_right_count;
  std::ostringstream warning;
  warning << "frontend_pairing_prefilter pairing_mode="
          << selection.pairing_mode
          << " original_left=" << selection.original_left_frame_count
          << " original_right=" << selection.original_right_frame_count
          << " processed_pairs=" << selection.paired_frame_count
          << " skipped_left=" << selection.unmatched_left_count
          << " skipped_right=" << selection.unmatched_right_count;
  dataset->warnings.push_back(warning.str());
}

const ati::StereoFramePair* FindStereoFramePair(
    const ati::StereoMeasurementDataset& dataset, int pair_index) {
  for (const ati::StereoFramePair& pair : dataset.frame_pairs) {
    if (pair.pair_index == pair_index) {
      return &pair;
    }
  }
  return nullptr;
}

std::set<int> CollectObservationBoardsForPair(
    const ati::StereoMeasurementDataset& dataset, int pair_index) {
  std::set<int> board_ids;
  for (const ati::StereoObservation& observation : dataset.observations) {
    if (observation.pair_index == pair_index && observation.used_in_solver &&
        observation.board_id >= 0) {
      board_ids.insert(observation.board_id);
    }
  }
  return board_ids;
}

void RecomputeStereoPairBoardStatistics(
    ati::StereoMeasurementDataset* dataset) {
  if (dataset == nullptr) {
    throw std::runtime_error(
        "RecomputeStereoPairBoardStatistics requires dataset.");
  }
  dataset->shared_board_observation_count = 0;
  dataset->cam0_only_board_observation_count = 0;
  dataset->cam1_only_board_observation_count = 0;
  dataset->pair_shared_board_count.clear();
  dataset->pair_cam0_only_board_count.clear();
  dataset->pair_cam1_only_board_count.clear();
  dataset->pair_shared_board_ids.clear();
  dataset->pair_cam0_only_board_ids.clear();
  dataset->pair_cam1_only_board_ids.clear();
  dataset->training_pair_board_ids.clear();
  dataset->holdout_pair_board_ids.clear();

  std::map<int, std::set<int> > cam0_boards_by_pair;
  std::map<int, std::set<int> > cam1_boards_by_pair;
  for (const ati::StereoObservation& observation : dataset->observations) {
    if (!observation.used_in_solver || observation.pair_index < 0 ||
        observation.board_id < 0) {
      continue;
    }
    if (observation.camera_index == 0) {
      cam0_boards_by_pair[observation.pair_index].insert(observation.board_id);
    } else if (observation.camera_index == 1) {
      cam1_boards_by_pair[observation.pair_index].insert(observation.board_id);
    }
  }

  for (const ati::StereoFramePair& pair : dataset->frame_pairs) {
    const std::set<int>& cam0_boards = cam0_boards_by_pair[pair.pair_index];
    const std::set<int>& cam1_boards = cam1_boards_by_pair[pair.pair_index];
    std::set<int> union_boards = cam0_boards;
    union_boards.insert(cam1_boards.begin(), cam1_boards.end());
    if (union_boards.empty()) {
      continue;
    }
    std::set<int>* board_target =
        pair.is_training ? &dataset->training_pair_board_ids[pair.pair_index]
                         : &dataset->holdout_pair_board_ids[pair.pair_index];
    *board_target = union_boards;
    for (int board_id : union_boards) {
      const bool in_cam0 = cam0_boards.count(board_id) > 0;
      const bool in_cam1 = cam1_boards.count(board_id) > 0;
      if (in_cam0 && in_cam1) {
        ++dataset->shared_board_observation_count;
        ++dataset->pair_shared_board_count[pair.pair_index];
        dataset->pair_shared_board_ids[pair.pair_index].insert(board_id);
      } else if (in_cam0) {
        ++dataset->cam0_only_board_observation_count;
        ++dataset->pair_cam0_only_board_count[pair.pair_index];
        dataset->pair_cam0_only_board_ids[pair.pair_index].insert(board_id);
      } else if (in_cam1) {
        ++dataset->cam1_only_board_observation_count;
        ++dataset->pair_cam1_only_board_count[pair.pair_index];
        dataset->pair_cam1_only_board_ids[pair.pair_index].insert(board_id);
      }
    }
  }
}

ati::StereoMeasurementDataset ApplyStage6TrainingPairSample(
    const ati::StereoMeasurementDataset& dataset,
    int sample_count,
    int sample_seed) {
  if (sample_count <= 0 ||
      sample_count >= static_cast<int>(dataset.training_pair_indices.size())) {
    return dataset;
  }
  if (!dataset.success) {
    return dataset;
  }

  std::vector<int> shuffled_training_pairs = dataset.training_pair_indices;
  std::mt19937 rng(static_cast<std::mt19937::result_type>(sample_seed));
  std::shuffle(shuffled_training_pairs.begin(), shuffled_training_pairs.end(),
               rng);
  shuffled_training_pairs.resize(static_cast<std::size_t>(sample_count));
  std::sort(shuffled_training_pairs.begin(), shuffled_training_pairs.end());
  const std::set<int> selected_training_pairs(shuffled_training_pairs.begin(),
                                              shuffled_training_pairs.end());
  const std::set<int> holdout_pairs(dataset.holdout_pair_indices.begin(),
                                    dataset.holdout_pair_indices.end());

  ati::StereoMeasurementDataset sampled = dataset;
  sampled.frame_pairs.clear();
  sampled.observations.clear();
  sampled.training_pair_indices.clear();
  sampled.holdout_pair_indices.clear();

  for (const ati::StereoFramePair& pair : dataset.frame_pairs) {
    if (pair.is_training) {
      if (selected_training_pairs.count(pair.pair_index) == 0) {
        continue;
      }
      sampled.training_pair_indices.push_back(pair.pair_index);
    } else if (holdout_pairs.count(pair.pair_index) > 0) {
      sampled.holdout_pair_indices.push_back(pair.pair_index);
    } else {
      continue;
    }
    sampled.frame_pairs.push_back(pair);
  }

  for (const ati::StereoObservation& observation : dataset.observations) {
    const bool keep_training =
        selected_training_pairs.count(observation.pair_index) > 0;
    const bool keep_holdout = holdout_pairs.count(observation.pair_index) > 0;
    if (keep_training || keep_holdout) {
      sampled.observations.push_back(observation);
    }
  }

  sampled.paired_frame_count = static_cast<int>(sampled.frame_pairs.size());
  RecomputeStereoPairBoardStatistics(&sampled);
  std::ostringstream warning;
  warning << "stage6_training_pair_sample"
          << " requested_count=" << sample_count
          << " seed=" << sample_seed
          << " selected_training_pairs=";
  for (std::size_t index = 0; index < shuffled_training_pairs.size(); ++index) {
    if (index > 0) {
      warning << "|";
    }
    warning << shuffled_training_pairs[index];
  }
  sampled.warnings.push_back(warning.str());
  return sampled;
}

ati::StereoMeasurementDataset ApplyStage6BoardMaskingAblation(
    const ati::StereoMeasurementDataset& dataset, const std::string& mode) {
  if (mode.empty() || mode == "none") {
    return dataset;
  }
  if (mode != "split_pair_boards") {
    throw std::runtime_error(
        "Unsupported --stage6-board-masking-ablation mode: " + mode);
  }
  if (!dataset.success) {
    return dataset;
  }

  ati::StereoMeasurementDataset masked;
  masked.success = dataset.success;
  masked.reference_board_id = dataset.reference_board_id;
  masked.left_frame_count = dataset.left_frame_count;
  masked.right_frame_count = dataset.right_frame_count;
  masked.unmatched_left_count = dataset.unmatched_left_count;
  masked.unmatched_right_count = dataset.unmatched_right_count;
  masked.pairing_mode = dataset.pairing_mode + "|board_masking_split_pair_boards";
  masked.max_pair_timestamp_delta_ns = dataset.max_pair_timestamp_delta_ns;
  masked.mean_abs_pair_timestamp_delta_ms =
      dataset.mean_abs_pair_timestamp_delta_ms;
  masked.max_abs_pair_timestamp_delta_ms =
      dataset.max_abs_pair_timestamp_delta_ms;
  masked.warnings = dataset.warnings;
  masked.failure_reason = dataset.failure_reason;

  std::set<int> training_pairs(dataset.training_pair_indices.begin(),
                               dataset.training_pair_indices.end());
  std::map<std::pair<int, int>, int> training_pair_board_to_pseudo_pair;
  std::map<int, int> holdout_pair_to_new_pair;
  int next_pair_index = 0;
  int original_training_pair_count = 0;

  for (int old_pair_index : dataset.training_pair_indices) {
    const ati::StereoFramePair* old_pair =
        FindStereoFramePair(dataset, old_pair_index);
    if (old_pair == nullptr) {
      continue;
    }
    ++original_training_pair_count;
    std::set<int> board_ids;
    const auto training_boards_it =
        dataset.training_pair_board_ids.find(old_pair_index);
    if (training_boards_it != dataset.training_pair_board_ids.end()) {
      board_ids = training_boards_it->second;
    }
    if (board_ids.empty()) {
      board_ids = CollectObservationBoardsForPair(dataset, old_pair_index);
    }
    for (int board_id : board_ids) {
      ati::StereoFramePair pseudo_pair = *old_pair;
      pseudo_pair.pair_index = next_pair_index;
      pseudo_pair.is_training = true;
      pseudo_pair.left_frame_label += "_mask_board" + std::to_string(board_id);
      pseudo_pair.right_frame_label += "_mask_board" + std::to_string(board_id);
      masked.frame_pairs.push_back(pseudo_pair);
      masked.training_pair_indices.push_back(next_pair_index);
      training_pair_board_to_pseudo_pair[std::make_pair(old_pair_index, board_id)] =
          next_pair_index;
      ++next_pair_index;
    }
  }

  for (int old_pair_index : dataset.holdout_pair_indices) {
    const ati::StereoFramePair* old_pair =
        FindStereoFramePair(dataset, old_pair_index);
    if (old_pair == nullptr) {
      continue;
    }
    ati::StereoFramePair holdout_pair = *old_pair;
    holdout_pair.pair_index = next_pair_index;
    holdout_pair.is_training = false;
    masked.frame_pairs.push_back(holdout_pair);
    masked.holdout_pair_indices.push_back(next_pair_index);
    holdout_pair_to_new_pair[old_pair_index] = next_pair_index;
    ++next_pair_index;
  }

  int copied_training_observation_count = 0;
  int copied_holdout_observation_count = 0;
  for (const ati::StereoObservation& input_observation :
       dataset.observations) {
    ati::StereoObservation observation = input_observation;
    if (training_pairs.count(input_observation.pair_index) > 0) {
      const auto pseudo_it = training_pair_board_to_pseudo_pair.find(
          std::make_pair(input_observation.pair_index,
                         input_observation.board_id));
      if (pseudo_it == training_pair_board_to_pseudo_pair.end()) {
        continue;
      }
      observation.pair_index = pseudo_it->second;
      masked.observations.push_back(observation);
      ++copied_training_observation_count;
    } else {
      const auto holdout_it =
          holdout_pair_to_new_pair.find(input_observation.pair_index);
      if (holdout_it == holdout_pair_to_new_pair.end()) {
        continue;
      }
      observation.pair_index = holdout_it->second;
      masked.observations.push_back(observation);
      ++copied_holdout_observation_count;
    }
  }

  masked.paired_frame_count = static_cast<int>(masked.frame_pairs.size());
  RecomputeStereoPairBoardStatistics(&masked);
  std::ostringstream warning;
  warning << "stage6_board_masking_ablation=split_pair_boards"
          << " original_training_pairs=" << original_training_pair_count
          << " pseudo_training_pairs="
          << masked.training_pair_indices.size()
          << " holdout_pairs=" << masked.holdout_pair_indices.size()
          << " copied_training_observations="
          << copied_training_observation_count
          << " copied_holdout_observations="
          << copied_holdout_observation_count;
  masked.warnings.push_back(warning.str());
  return masked;
}

MonocularFrontendBundleResult RunFixedIntrinsicsMonocularFrontend(
    const std::string& config_path,
    const std::string& intrinsics_path,
    const std::vector<std::string>& image_paths,
    const std::string& dataset_label,
    const std::string& cache_dir,
    const CmdArgs& args) {
  ati::ApriltagInternalConfig config = ati::ApriltagInternalDetector::LoadConfig(config_path);
  const ati::IntermediateCameraConfig fixed_intrinsics =
      ati::LoadExternalCameraConfig(intrinsics_path);
  if (!fixed_intrinsics.IsConfigured()) {
    throw std::runtime_error(
        "Stage6 fixed-intrinsics frontend requires a complete camera model in " +
        intrinsics_path);
  }
  config.intermediate_camera = fixed_intrinsics;
  config.camera_initialization_mode = ati::CameraInitializationMode::Manual;

  ati::OuterBootstrapCameraIntrinsics explicit_initial_camera;
  explicit_initial_camera.camera_model = fixed_intrinsics.camera_model;
  explicit_initial_camera.distortion_model = fixed_intrinsics.distortion_model;
  explicit_initial_camera.SetIntrinsicsVector(fixed_intrinsics.intrinsics);
  explicit_initial_camera.SetDistortionVector(fixed_intrinsics.distortion_coeffs);
  if (fixed_intrinsics.resolution.size() == 2) {
    explicit_initial_camera.resolution =
        cv::Size(fixed_intrinsics.resolution[0],
                 fixed_intrinsics.resolution[1]);
  }

  ati::FrozenRound2BaselineOptions options;
  options.config = config;
  options.use_explicit_initial_camera = true;
  options.explicit_initial_camera = explicit_initial_camera;
  options.explicit_initial_camera_source_label =
      "stage6_explicit_fixed_intrinsics";
  options.optimize_intrinsics = false;
  options.run_second_pass = true;
  options.strict_board_observation_acceptance = true;
  options.enable_board_pose_fit_gate = false;
  options.enable_residual_sanity_gate = true;
  options.dataset_label = dataset_label;
  options.source_pipeline_label = "stage6_fixed_intrinsics_frontend";
  options.baseline_protocol_label = "stage6_fixed_intrinsics_frontend";
  options.training_split_signature = "stereo_all_pairs";
  options.enable_outer_detection_cache = !cache_dir.empty();
  options.outer_detection_cache_dir = cache_dir;
  options.enable_geometry_prior_outer_seed =
      args.stage6_enable_geometry_prior_outer_seed;
  options.geometry_prior_rescue_diagnostic_only =
      args.stage6_geometry_prior_rescue_diagnostic_only;
  options.geometry_prior_rescue_use_as_observation =
      args.stage6_geometry_prior_rescue_use_as_observation;
  options.geometry_prior_rescue_keep_outer_on_internal_failure =
      args.stage6_geometry_prior_rescue_keep_outer_on_internal_failure;
  options.geometry_prior_rescue_allow_geometry_only_pose_refit =
      args.stage6_geometry_prior_rescue_allow_geometry_only_pose_refit;
  options.geometry_prior_rescue_subpix_window_radius =
      args.stage6_geometry_prior_rescue_subpix_window_radius;
  options.geometry_prior_rescue_max_corner_displacement_px =
      args.stage6_geometry_prior_rescue_max_corner_displacement_px;
  options.geometry_prior_rescue_min_corner_response_ratio =
      args.stage6_geometry_prior_rescue_min_corner_response_ratio;
  options.geometry_prior_rescue_enable_spherical_refine =
      args.stage6_geometry_prior_rescue_enable_spherical_refine;
  options.geometry_prior_rescue_edge_sample_count =
      args.stage6_geometry_prior_rescue_edge_sample_count;
  options.geometry_prior_rescue_edge_search_half_width_px =
      args.stage6_geometry_prior_rescue_edge_search_half_width_px;
  options.geometry_prior_rescue_min_edge_support_ratio =
      args.stage6_geometry_prior_rescue_min_edge_support_ratio;
  options.geometry_prior_rescue_min_edge_gradient_ratio =
      args.stage6_geometry_prior_rescue_min_edge_gradient_ratio;
  options.geometry_prior_rescue_accept_max_outer_rmse =
      args.stage6_geometry_prior_rescue_accept_max_outer_rmse;
  options.geometry_prior_rescue_accept_max_rotation_error_deg =
      args.stage6_geometry_prior_rescue_accept_max_rotation_error_deg;
  options.geometry_prior_rescue_accept_max_translation_error =
      args.stage6_geometry_prior_rescue_accept_max_translation_error;

  ati::FrozenRound2BaselinePipeline pipeline(options);
  const ati::FrozenRound2BaselineResult result =
      pipeline.Run(BuildFrameSources(image_paths));
  if (!result.success || !result.stage5_bundle_available ||
      !result.final_stage5_bundle.IsReadyForBackend()) {
    throw std::runtime_error(
        "Failed to build fixed-intrinsics monocular frontend bundle for " +
        dataset_label + ": " + result.failure_reason);
  }
  MonocularFrontendBundleResult frontend_result;
  frontend_result.bundle = result.final_stage5_bundle;
  frontend_result.baseline_result = result;
  frontend_result.runtime_breakdown = result.runtime_breakdown;
  return frontend_result;
}

CmdArgs MakeHoldoutFrontendArgs(const CmdArgs& args) {
  CmdArgs holdout_args = args;
  if (holdout_args.stage6_disable_geometry_prior_rescue_for_holdout) {
    holdout_args.stage6_enable_geometry_prior_outer_seed = false;
    holdout_args.stage6_geometry_prior_rescue_diagnostic_only = true;
    holdout_args.stage6_geometry_prior_rescue_use_as_observation = false;
    holdout_args.stage6_geometry_prior_rescue_keep_outer_on_internal_failure = false;
    holdout_args.stage6_geometry_prior_rescue_allow_geometry_only_pose_refit = false;
    holdout_args.stage6_geometry_prior_rescue_enable_spherical_refine = false;
  }
  return holdout_args;
}

void WriteStage6GeometryPriorDiagnostics(
    const fs::path& output_dir,
    const std::string& frontend_label,
    const MonocularFrontendBundleResult& frontend_result,
    const CmdArgs& args) {
  if (!args.stage6_enable_geometry_prior_outer_seed) {
    return;
  }
  ati::Stage5BenchmarkReport report;
  report.success = frontend_result.baseline_result.success;
  report.diagnostic_only = true;
  report.dataset_label = frontend_label;
  report.baseline_protocol_label = "stage6_fixed_intrinsics_frontend";
  report.split_signature = "stage6_monocular_frontend";
  report.baseline_result = frontend_result.baseline_result;

  const fs::path diagnostic_dir =
      output_dir / "stage6_geometry_prior_diagnostics" / frontend_label;
  EnsureDirectoryExists(diagnostic_dir);
  ati::WriteGeometryPriorOuterSeedDiagnostics(diagnostic_dir, report);
}

ati::StereoCameraFixedCalibration ToStereoCalibration(
    const ati::IntermediateCameraConfig& config,
    const std::string& source_yaml_path) {
  ati::OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model = config.camera_model;
  intrinsics.distortion_model = config.distortion_model;
  intrinsics.SetIntrinsicsVector(config.intrinsics);
  intrinsics.SetDistortionVector(config.distortion_coeffs);
  intrinsics.resolution = cv::Size(config.resolution[0], config.resolution[1]);

  ati::StereoCameraFixedCalibration calibration;
  calibration.camera_model_family = intrinsics.NormalizedFamilyString();
  calibration.camera_model = intrinsics.NormalizedCameraModel();
  calibration.distortion_model = intrinsics.NormalizedDistortionModel();
  calibration.intrinsics = config.intrinsics;
  calibration.distortion_coeffs = config.distortion_coeffs;
  calibration.resolution = config.resolution;
  calibration.source_yaml_path = source_yaml_path;
  return calibration;
}

ati::StereoViewSelectionMode ParseViewSelectionMode(const std::string& value) {
  if (value == "off") {
    return ati::StereoViewSelectionMode::Off;
  }
  if (value == "topk") {
    return ati::StereoViewSelectionMode::TopK;
  }
  throw std::runtime_error("Unsupported --stage6-view-selection-mode: " + value);
}

ati::StereoSolverMode ParseSolverMode(const std::string& value) {
  if (value == "alternating") {
    return ati::StereoSolverMode::Alternating;
  }
  if (value == "global_sparse_ba") {
    return ati::StereoSolverMode::GlobalSparseBa;
  }
  if (value == "shared_only_global_sparse_ba") {
    return ati::StereoSolverMode::SharedOnlyGlobalSparseBa;
  }
  throw std::runtime_error("Unsupported --stage6-solver-mode: " + value);
}

ati::StereoIntrinsicsMode ParseIntrinsicsMode(const std::string& value) {
  if (value == "fixed_stage5") {
    return ati::StereoIntrinsicsMode::FixedStage5;
  }
  if (value == "kalibr_joint_projection") {
    return ati::StereoIntrinsicsMode::KalibrJointProjection;
  }
  if (value == "regularized_joint_projection") {
    return ati::StereoIntrinsicsMode::RegularizedJointProjection;
  }
  if (value == "adaptive_regularized_joint_projection") {
    return ati::StereoIntrinsicsMode::AdaptiveRegularizedJointProjection;
  }
  throw std::runtime_error("Unsupported --stage6-intrinsics-mode: " + value);
}

ati::StereoPersistentPoseStructure ParsePersistentPoseStructure(
    const std::string& value) {
  if (value == "independent_pair_board") {
    return ati::StereoPersistentPoseStructure::IndependentPairBoard;
  }
  if (value == "shared_frame_layout") {
    return ati::StereoPersistentPoseStructure::SharedFrameLayout;
  }
  throw std::runtime_error(
      "Unsupported --stage6-persistent-pose-structure: " + value);
}

ati::StereoFinalBaResidualMode ParseFinalBaResidualMode(
    const std::string& value) {
  if (value == "pixel") {
    return ati::StereoFinalBaResidualMode::Pixel;
  }
  if (value == "spherical_chordal" || value == "spherical-chordal" ||
      value == "chordal") {
    return ati::StereoFinalBaResidualMode::SphericalChordal;
  }
  if (value == "spherical_tangent" || value == "spherical-tangent" ||
      value == "tangent_angular" || value == "tangent-angular" ||
      value == "angular") {
    return ati::StereoFinalBaResidualMode::SphericalTangent;
  }
  if (value == "hybrid_pixel_spherical" ||
      value == "hybrid-pixel-spherical" || value == "hybrid") {
    return ati::StereoFinalBaResidualMode::HybridPixelSpherical;
  }
  throw std::runtime_error(
      "Unsupported --stage6-residual-mode/--stage6-final-ba-residual-mode: " +
      value);
}

ati::StereoSphericalUncertaintyMode ParseSphericalUncertaintyMode(
    const std::string& value) {
  if (value == "none" || value == "off") {
    return ati::StereoSphericalUncertaintyMode::None;
  }
  if (value == "pixel") {
    return ati::StereoSphericalUncertaintyMode::Pixel;
  }
  if (value == "model") {
    return ati::StereoSphericalUncertaintyMode::Model;
  }
  if (value == "pixel_model" || value == "pixel+model" ||
      value == "pixel-model") {
    return ati::StereoSphericalUncertaintyMode::PixelModel;
  }
  throw std::runtime_error(
      "Unsupported --stage6-spherical-uncertainty-mode: " + value);
}

ati::StereoRigParamMode ParseRigParamMode(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "cam0_reference" || lowered == "cam0-reference" ||
      lowered == "cam0") {
    return ati::StereoRigParamMode::Cam0Reference;
  }
  if (lowered == "rig_centric_symmetric" ||
      lowered == "rig-centric-symmetric" || lowered == "rig_centric" ||
      lowered == "rig-centric") {
    return ati::StereoRigParamMode::RigCentricSymmetric;
  }
  throw std::runtime_error("Unsupported --stage6-rig-param-mode: " + value);
}

std::array<double, 6> ParseSixDoubles(const std::string& value,
                                      const std::string& flag_name) {
  std::array<double, 6> result{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  std::stringstream stream(value);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',')) {
    if (index >= 6) {
      throw std::runtime_error(flag_name + " expects exactly 6 comma-separated values.");
    }
    result[index++] = std::stod(token);
  }
  if (index != 6) {
    throw std::runtime_error(flag_name + " expects exactly 6 comma-separated values.");
  }
  return result;
}

ati::StereoPairBoardSelectionMode ParsePairBoardSelectionMode(
    const std::string& value) {
  if (value == "strict_rmse") {
    return ati::StereoPairBoardSelectionMode::StrictRmse;
  }
  if (value == "kalibr_style_batch") {
    return ati::StereoPairBoardSelectionMode::KalibrStyleBatch;
  }
  throw std::runtime_error(
      "Unsupported --stage6-pairboard-selection-mode: " + value);
}

ati::Stage6IncrementalInfoBlock ParseStage6IncrementalInfoBlock(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "stereo_extrinsic" || lowered == "stereo-extrinsic" ||
      lowered == "extrinsic") {
    return ati::Stage6IncrementalInfoBlock::StereoExtrinsic;
  }
  throw std::runtime_error(
      "Unsupported --stage6-incremental-info-block: " + value);
}

ati::StereoCandidateBudgetMode ParseCandidateBudgetMode(
    const std::string& value,
    const std::string& flag_name) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "fixed") {
    return ati::StereoCandidateBudgetMode::Fixed;
  }
  if (lowered == "adaptive") {
    return ati::StereoCandidateBudgetMode::Adaptive;
  }
  if (lowered == "kalibr_style" || lowered == "kalibr-style") {
    return ati::StereoCandidateBudgetMode::KalibrStyle;
  }
  throw std::runtime_error("Unsupported " + flag_name + ": " + value);
}

ati::StereoSingleCameraOnlyWeightMode ParseSingleCameraOnlyWeightMode(
    const std::string& value) {
  if (value == "fixed_scale") {
    return ati::StereoSingleCameraOnlyWeightMode::FixedScale;
  }
  if (value == "per_side_budget_cap") {
    return ati::StereoSingleCameraOnlyWeightMode::PerSideBudgetCap;
  }
  if (value == "adaptive_independent_side_cap") {
    return ati::StereoSingleCameraOnlyWeightMode::AdaptiveIndependentSideCap;
  }
  throw std::runtime_error(
      "Unsupported --stage6-ba-single-camera-only-weight-mode: " + value);
}

ati::StereoSingleBoardPairPolicy ParseSingleBoardPairPolicy(
    const std::string& value) {
  if (value == "keep") {
    return ati::StereoSingleBoardPairPolicy::Keep;
  }
  if (value == "audit") {
    return ati::StereoSingleBoardPairPolicy::Audit;
  }
  if (value == "drop") {
    return ati::StereoSingleBoardPairPolicy::Drop;
  }
  if (value == "low_weight") {
    return ati::StereoSingleBoardPairPolicy::LowWeight;
  }
  throw std::runtime_error(
      "Unsupported --stage6-single-board-pair-policy: " + value);
}

ati::StereoMeasurementSourceMode ParseStereoMeasurementSourceMode(
    const std::string& value) {
  if (value == "backend_selected_only" || value == "backend-selected-only") {
    return ati::StereoMeasurementSourceMode::BackendSelectedOnly;
  }
  if (value == "all_valid" || value == "all-valid") {
    return ati::StereoMeasurementSourceMode::AllValid;
  }
  throw std::runtime_error(
      "Unsupported --stage6-stereo-measurement-source: " + value);
}

ati::StereoFramePairingMode ParseStereoFramePairingMode(
    const std::string& value) {
  if (value == "exact_timestamp" || value == "exact-timestamp") {
    return ati::StereoFramePairingMode::ExactTimestamp;
  }
  if (value == "frame_index" || value == "frame-index") {
    return ati::StereoFramePairingMode::FrameIndex;
  }
  throw std::runtime_error(
      "Unsupported --stage6-frame-pairing-mode: " + value);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Clock::time_point total_start = Clock::now();
    const CmdArgs args = ParseArgs(argc, argv);
    const fs::path output_dir(args.output_path);
    EnsureDirectoryExists(output_dir);
    const std::string cache_dir =
        args.cache_dir.empty() ? "result/.stage6_stereo_cache" : args.cache_dir;
    EnsureDirectoryExists(fs::path(cache_dir));

    std::cout << "[Stage6] collecting stereo image paths..." << std::endl;

    const std::vector<std::string> original_left_image_paths =
        CollectImagePaths(args.left_image_path);
    const std::vector<std::string> original_right_image_paths =
        CollectImagePaths(args.right_image_path);
    const bool use_cross_dataset_holdout =
        !args.test_left_image_path.empty() && !args.test_right_image_path.empty();
    const ati::StereoFramePairingMode frame_pairing_mode =
        ParseStereoFramePairingMode(args.stage6_frame_pairing_mode);
    const ati::StereoFrontendImagePairSelection training_frontend_selection =
        ati::SelectStereoFrontendImagePairs(
            original_left_image_paths, original_right_image_paths,
            frame_pairing_mode, args.stage6_frame_pairing_max_delta_ms);
    if (!training_frontend_selection.success) {
      throw std::runtime_error(
          training_frontend_selection.failure_reason.empty()
              ? "Stage6 frontend pairing prefilter failed."
              : training_frontend_selection.failure_reason);
    }
    const std::vector<std::string>& left_image_paths =
        training_frontend_selection.left_image_paths;
    const std::vector<std::string>& right_image_paths =
        training_frontend_selection.right_image_paths;
    std::vector<std::string> original_test_left_image_paths;
    std::vector<std::string> original_test_right_image_paths;
    ati::StereoFrontendImagePairSelection holdout_frontend_selection;
    std::vector<std::string> test_left_image_paths;
    std::vector<std::string> test_right_image_paths;
    std::cout << "[Stage6] original left frames: "
              << original_left_image_paths.size()
              << ", original right frames: "
              << original_right_image_paths.size()
              << ", frontend paired frames: " << left_image_paths.size()
              << std::endl;
    if (use_cross_dataset_holdout) {
      original_test_left_image_paths =
          CollectImagePaths(args.test_left_image_path);
      original_test_right_image_paths =
          CollectImagePaths(args.test_right_image_path);
      holdout_frontend_selection = ati::SelectStereoFrontendImagePairs(
          original_test_left_image_paths, original_test_right_image_paths,
          frame_pairing_mode, args.stage6_frame_pairing_max_delta_ms);
      if (!holdout_frontend_selection.success) {
        throw std::runtime_error(
            holdout_frontend_selection.failure_reason.empty()
                ? "Stage6 holdout frontend pairing prefilter failed."
                : holdout_frontend_selection.failure_reason);
      }
      test_left_image_paths = holdout_frontend_selection.left_image_paths;
      test_right_image_paths = holdout_frontend_selection.right_image_paths;
      std::cout << "[Stage6] original test left frames: "
                << original_test_left_image_paths.size()
                << ", original test right frames: "
                << original_test_right_image_paths.size()
                << ", frontend test paired frames: "
                << test_left_image_paths.size()
                << std::endl;
    }
    std::cout << "[Stage6] outer detection cache: " << cache_dir << std::endl;

    std::cout << "[Stage6] building fixed-intrinsics monocular frontend for cam0..."
              << std::endl;
    const MonocularFrontendBundleResult left_frontend =
        RunFixedIntrinsicsMonocularFrontend(
            args.left_config_path, args.left_intrinsics_path, left_image_paths,
            "stage6_left_monocular_frontend", cache_dir, args);
    WriteStage6GeometryPriorDiagnostics(
        output_dir, "train_cam0", left_frontend, args);
    const ati::CalibrationStateBundle& left_bundle = left_frontend.bundle;
    std::cout << "[Stage6] cam0 frontend ready." << std::endl;

    std::cout << "[Stage6] building fixed-intrinsics monocular frontend for cam1..."
              << std::endl;
    const MonocularFrontendBundleResult right_frontend =
        RunFixedIntrinsicsMonocularFrontend(
            args.right_config_path, args.right_intrinsics_path, right_image_paths,
            "stage6_right_monocular_frontend", cache_dir, args);
    WriteStage6GeometryPriorDiagnostics(
        output_dir, "train_cam1", right_frontend, args);
    const ati::CalibrationStateBundle& right_bundle = right_frontend.bundle;
    std::cout << "[Stage6] cam1 frontend ready." << std::endl;

    MonocularFrontendBundleResult test_left_frontend;
    MonocularFrontendBundleResult test_right_frontend;
    if (use_cross_dataset_holdout) {
      const CmdArgs holdout_args = MakeHoldoutFrontendArgs(args);
      std::cout
          << "[Stage6] building fixed-intrinsics monocular frontend for holdout cam0..."
          << std::endl;
      test_left_frontend = RunFixedIntrinsicsMonocularFrontend(
          args.left_config_path, args.left_intrinsics_path, test_left_image_paths,
          "stage6_left_monocular_holdout_frontend", cache_dir, holdout_args);
      WriteStage6GeometryPriorDiagnostics(
          output_dir, "holdout_cam0", test_left_frontend, holdout_args);
      std::cout << "[Stage6] holdout cam0 frontend ready." << std::endl;
      std::cout
          << "[Stage6] building fixed-intrinsics monocular frontend for holdout cam1..."
          << std::endl;
      test_right_frontend = RunFixedIntrinsicsMonocularFrontend(
          args.right_config_path, args.right_intrinsics_path, test_right_image_paths,
          "stage6_right_monocular_holdout_frontend", cache_dir, holdout_args);
      WriteStage6GeometryPriorDiagnostics(
          output_dir, "holdout_cam1", test_right_frontend, holdout_args);
      std::cout << "[Stage6] holdout cam1 frontend ready." << std::endl;
    }

    if (left_bundle.scene_state.reference_board_id !=
        right_bundle.scene_state.reference_board_id) {
      throw std::runtime_error(
          "Left/right reference_board_id mismatch; Stage6 v1 requires them to match.");
    }

    ati::StereoExtrinsicProblemInput problem_input;
    problem_input.dataset_label = "stage6_stereo_extrinsic_v1";
    problem_input.left_image_path = args.left_image_path;
    problem_input.right_image_path = args.right_image_path;
    problem_input.left_config_path = args.left_config_path;
    problem_input.right_config_path = args.right_config_path;
    problem_input.left_intrinsics_path = args.left_intrinsics_path;
    problem_input.right_intrinsics_path = args.right_intrinsics_path;
    problem_input.split_signature = use_cross_dataset_holdout
                                        ? "cross_dataset_holdout"
                                        : "deterministic_stride_" +
                                              std::to_string(args.holdout_stride) +
                                              "_offset_" +
                                              std::to_string(args.holdout_offset);
    problem_input.measurement_source_mode =
        args.stage6_stereo_measurement_source;
    problem_input.left_bundle = left_bundle;
    problem_input.right_bundle = right_bundle;
    if (use_cross_dataset_holdout) {
      ati::StereoMeasurementDataset training_dataset =
          ati::BuildStereoMeasurementDataset(
              left_image_paths, right_image_paths, 0, 0, left_bundle, right_bundle,
              ParseStereoMeasurementSourceMode(
                  args.stage6_stereo_measurement_source),
              frame_pairing_mode,
              args.stage6_frame_pairing_max_delta_ms);
      RestoreFrontendPairingPopulationCounts(training_frontend_selection,
                                             &training_dataset);
      if (test_left_frontend.bundle.scene_state.reference_board_id !=
          test_right_frontend.bundle.scene_state.reference_board_id) {
        throw std::runtime_error(
            "Holdout left/right reference_board_id mismatch in cross-dataset Stage6.");
      }
      ati::StereoMeasurementDataset holdout_dataset =
          ati::BuildStereoMeasurementDataset(
              test_left_image_paths, test_right_image_paths, 1, 0,
              test_left_frontend.bundle, test_right_frontend.bundle,
              ParseStereoMeasurementSourceMode(
                  args.stage6_stereo_measurement_source),
              frame_pairing_mode,
              args.stage6_frame_pairing_max_delta_ms);
      RestoreFrontendPairingPopulationCounts(holdout_frontend_selection,
                                             &holdout_dataset);
      problem_input.measurement_dataset =
          MergeTrainAndTestStereoDatasets(training_dataset, holdout_dataset);
    } else {
      problem_input.measurement_dataset =
          ati::BuildStereoMeasurementDataset(
              left_image_paths, right_image_paths, args.holdout_stride,
              args.holdout_offset, left_bundle, right_bundle,
              ParseStereoMeasurementSourceMode(
                  args.stage6_stereo_measurement_source),
              frame_pairing_mode,
              args.stage6_frame_pairing_max_delta_ms);
      RestoreFrontendPairingPopulationCounts(training_frontend_selection,
                                             &problem_input.measurement_dataset);
    }
    problem_input.measurement_dataset = ApplyStage6TrainingPairSample(
        problem_input.measurement_dataset,
        args.stage6_training_pair_sample_count,
        args.stage6_training_pair_sample_seed);
    problem_input.measurement_dataset = ApplyStage6BoardMaskingAblation(
        problem_input.measurement_dataset,
        args.stage6_board_masking_ablation);
    std::cout << "[Stage6] stereo dataset built:"
              << " paired=" << problem_input.measurement_dataset.paired_frame_count
              << ", training_pairs="
              << problem_input.measurement_dataset.training_pair_indices.size()
              << ", holdout_pairs="
              << problem_input.measurement_dataset.holdout_pair_indices.size()
              << ", shared_board_obs="
              << problem_input.measurement_dataset.shared_board_observation_count
              << std::endl;
    if (args.stage6_board_masking_ablation != "none") {
      std::cout << "[Stage6] board masking ablation: "
                << args.stage6_board_masking_ablation << std::endl;
    }
    ati::StereoExtrinsicCalibrationResult result;
    result.runtime_summary.cache_dir = cache_dir;
    result.runtime_summary.cache_enabled = true;
    result.runtime_summary.cam0_training_detection_cache_hits =
        left_frontend.runtime_breakdown.training_detection_cache.cache_hits;
    result.runtime_summary.cam0_training_detection_cache_misses =
        left_frontend.runtime_breakdown.training_detection_cache.cache_misses;
    result.runtime_summary.cam0_training_detection_cache_load_failures =
        left_frontend.runtime_breakdown.training_detection_cache.load_failures;
    result.runtime_summary.cam0_training_detection_cache_store_failures =
        left_frontend.runtime_breakdown.training_detection_cache.store_failures;
    result.runtime_summary.cam1_training_detection_cache_hits =
        right_frontend.runtime_breakdown.training_detection_cache.cache_hits;
    result.runtime_summary.cam1_training_detection_cache_misses =
        right_frontend.runtime_breakdown.training_detection_cache.cache_misses;
    result.runtime_summary.cam1_training_detection_cache_load_failures =
        right_frontend.runtime_breakdown.training_detection_cache.load_failures;
    result.runtime_summary.cam1_training_detection_cache_store_failures =
        right_frontend.runtime_breakdown.training_detection_cache.store_failures;
    result.runtime_summary.frontend_pairing_prefilter_enabled = true;
    result.runtime_summary.frontend_original_left_frame_count =
        training_frontend_selection.original_left_frame_count +
        holdout_frontend_selection.original_left_frame_count;
    result.runtime_summary.frontend_original_right_frame_count =
        training_frontend_selection.original_right_frame_count +
        holdout_frontend_selection.original_right_frame_count;
    result.runtime_summary.frontend_processed_left_frame_count =
        static_cast<int>(left_image_paths.size() + test_left_image_paths.size());
    result.runtime_summary.frontend_processed_right_frame_count =
        static_cast<int>(right_image_paths.size() + test_right_image_paths.size());
    result.runtime_summary.frontend_skipped_unpaired_left_frame_count =
        training_frontend_selection.unmatched_left_count +
        holdout_frontend_selection.unmatched_left_count;
    result.runtime_summary.frontend_skipped_unpaired_right_frame_count =
        training_frontend_selection.unmatched_right_count +
        holdout_frontend_selection.unmatched_right_count;
    result.runtime_summary.pairing_build_dataset_runtime_seconds =
        std::chrono::duration<double>(Clock::now() - total_start).count();
    if (!problem_input.measurement_dataset.success) {
      throw std::runtime_error(problem_input.measurement_dataset.failure_reason);
    }
    problem_input.initial_scene.cam0_is_reference = true;
    problem_input.initial_scene.gauge_fixed_board_id =
        left_bundle.scene_state.reference_board_id;
    problem_input.initial_scene.cam0 =
        ToStereoCalibration(ati::LoadExternalCameraConfig(args.left_intrinsics_path),
                            args.left_intrinsics_path);
    problem_input.initial_scene.cam1 =
        ToStereoCalibration(ati::LoadExternalCameraConfig(args.right_intrinsics_path),
                            args.right_intrinsics_path);
    problem_input.solver_options.reference_board_id =
        left_bundle.scene_state.reference_board_id;
    problem_input.solver_options.max_iterations =
        args.stage6_max_alternating_iterations;
    problem_input.solver_options.convergence_threshold =
        args.stage6_min_total_rmse_improvement;
    problem_input.solver_options.require_symmetric_pair_refit =
        args.stage6_require_symmetric_pair_refit;
    problem_input.solver_options.max_graph_propagation_iterations =
        args.stage6_max_graph_propagation_iterations;
    problem_input.solver_options.min_shared_boards_for_extrinsic_candidate =
        args.stage6_min_shared_boards_for_extrinsic_candidate;
    problem_input.solver_options.enable_shared_board_quality_gate =
        args.stage6_enable_shared_board_quality_gate;
    problem_input.solver_options.enable_shared_board_quality_hard_gate =
        args.stage6_enable_shared_board_quality_hard_gate;
    problem_input.solver_options.shared_board_quality_max_outer_rmse_px =
        args.stage6_shared_board_quality_max_outer_rmse_px;
    problem_input.solver_options.shared_board_quality_min_outer_points_per_camera =
        args.stage6_shared_board_quality_min_outer_points_per_camera;
    problem_input.solver_options.shared_board_quality_min_good_shared_boards =
        args.stage6_shared_board_quality_min_good_shared_boards;
    problem_input.solver_options.export_pair_board_consistency_audit =
        args.stage6_export_pair_board_consistency_audit;
    problem_input.solver_options.enable_pair_board_consistency_gate =
        args.stage6_enable_pair_board_consistency_gate;
    problem_input.solver_options
        .pair_board_consistency_local_good_max_outer_rmse_px =
        args.stage6_pair_board_consistency_local_good_max_outer_rmse_px;
    problem_input.solver_options
        .pair_board_consistency_global_bad_min_outer_rmse_px =
        args.stage6_pair_board_consistency_global_bad_min_outer_rmse_px;
    problem_input.solver_options.view_selection_mode =
        ParseViewSelectionMode(args.stage6_view_selection_mode);
    problem_input.solver_options.selected_pair_count =
        args.stage6_selected_pair_count;
    problem_input.solver_options.solver_mode =
        ParseSolverMode(args.stage6_solver_mode);
    problem_input.solver_options.intrinsics_mode =
        ParseIntrinsicsMode(args.stage6_intrinsics_mode);
    problem_input.solver_options.persistent_pose_structure =
        ParsePersistentPoseStructure(args.stage6_persistent_pose_structure);
    problem_input.solver_options.ba_mode_label = args.stage6_ba_mode;
    problem_input.solver_options.final_ba_residual_mode =
        ParseFinalBaResidualMode(args.stage6_final_ba_residual_mode);
    problem_input.solver_options.selection_ba_residual_mode =
        ParseFinalBaResidualMode(args.stage6_selection_ba_residual_mode);
    problem_input.solver_options.fixed_intrinsics_for_spherical =
        args.stage6_fixed_intrinsics_for_spherical;
    problem_input.solver_options.spherical_weight =
        args.stage6_spherical_weight;
    problem_input.solver_options.spherical_polar_weighting =
        args.stage6_spherical_polar_weighting;
    problem_input.solver_options.spherical_min_polar_deg =
        args.stage6_spherical_min_polar_deg;
    problem_input.solver_options.spherical_max_weight =
        args.stage6_spherical_max_weight;
    problem_input.solver_options.spherical_uncertainty_mode =
        ParseSphericalUncertaintyMode(args.stage6_spherical_uncertainty_mode);
    problem_input.solver_options.spherical_pixel_sigma_px =
        args.stage6_spherical_pixel_sigma_px;
    problem_input.solver_options.spherical_model_sigma =
        args.stage6_spherical_model_sigma;
    problem_input.solver_options.spherical_covariance_damping =
        args.stage6_spherical_covariance_damping;
    problem_input.solver_options.spherical_min_sigma_rad =
        args.stage6_spherical_min_sigma_rad;
    problem_input.solver_options.spherical_max_whitening_weight =
        args.stage6_spherical_max_whitening_weight;
    problem_input.solver_options.spherical_use_normalize_jacobian =
        args.stage6_spherical_use_normalize_jacobian;
    problem_input.solver_options.rig_param_mode =
        ParseRigParamMode(args.stage6_rig_param_mode);
    problem_input.solver_options.rig_camera_prior_translation_weight =
        args.stage6_rig_camera_prior_translation_weight;
    problem_input.solver_options.rig_camera_prior_rotation_weight =
        args.stage6_rig_camera_prior_rotation_weight;
    problem_input.solver_options.rig_stereo_relative_prior_weight =
        args.stage6_rig_stereo_relative_prior_weight;
    problem_input.solver_options.coobs_enable = args.stage6_coobs_enable;
    problem_input.solver_options.coobs_output_dir =
        args.stage6_coobs_output_dir;
    problem_input.solver_options.coobs_min_corners_per_group =
        args.stage6_coobs_min_corners_per_group;
    problem_input.solver_options.coobs_high_polar_threshold_deg =
        args.stage6_coobs_high_polar_threshold_deg;
    problem_input.solver_options.coobs_very_high_polar_threshold_deg =
        args.stage6_coobs_very_high_polar_threshold_deg;
    problem_input.solver_options.coobs_enable_rescue_suggestions =
        args.stage6_coobs_enable_rescue_suggestions;
    problem_input.solver_options.coobs_score_alpha_high_polar =
        args.stage6_coobs_score_alpha_high_polar;
    problem_input.solver_options.coobs_score_beta_multiboard =
        args.stage6_coobs_score_beta_multiboard;
    problem_input.solver_options.coobs_score_gamma_balance =
        args.stage6_coobs_score_gamma_balance;
    problem_input.solver_options.coobs_score_eta_conflict =
        args.stage6_coobs_score_eta_conflict;
    problem_input.solver_options.coobs_rescue_min_high_polar_score =
        args.stage6_coobs_rescue_min_high_polar_score;
    problem_input.solver_options.coobs_rescue_bad_conflict_threshold =
        args.stage6_coobs_rescue_bad_conflict_threshold;
    problem_input.solver_options.selection_coobs_factor_ba_enable =
        args.stage6_selection_coobs_factor_ba_enable;
    problem_input.solver_options
        .selection_coobs_factor_ba_apply_stereo_factor =
        args.stage6_selection_coobs_factor_ba_apply_stereo_factor;
    problem_input.solver_options
        .selection_coobs_factor_ba_apply_layout_factor =
        args.stage6_selection_coobs_factor_ba_apply_layout_factor;
    problem_input.solver_options.selection_coobs_factor_ba_stereo_weight =
        args.stage6_selection_coobs_factor_ba_stereo_weight;
    problem_input.solver_options.selection_coobs_factor_ba_layout_weight =
        args.stage6_selection_coobs_factor_ba_layout_weight;
    problem_input.solver_options.coobs_aware_acceptance_enable =
        args.stage6_coobs_aware_acceptance_enable;
    problem_input.solver_options.coobs_aware_acceptance_min_score =
        args.stage6_coobs_aware_acceptance_min_score;
    problem_input.solver_options.coobs_aware_acceptance_max_total_rmse_delta =
        args.stage6_coobs_aware_acceptance_max_total_rmse_delta;
    problem_input.solver_options.coobs_aware_acceptance_max_camera_rmse_delta =
        args.stage6_coobs_aware_acceptance_max_camera_rmse_delta;
    problem_input.solver_options
        .coobs_aware_acceptance_balance_guard_enable =
        args.stage6_coobs_aware_acceptance_balance_guard_enable;
    problem_input.solver_options
        .coobs_aware_acceptance_max_camera_delta_imbalance =
        args.stage6_coobs_aware_acceptance_max_camera_delta_imbalance;
    problem_input.solver_options.coobs_aware_acceptance_max_camera_delta_ratio =
        args.stage6_coobs_aware_acceptance_max_camera_delta_ratio;
    problem_input.solver_options
        .coobs_aware_acceptance_require_pair_completion =
        args.stage6_coobs_aware_acceptance_require_pair_completion;
    problem_input.solver_options.coobs_aware_acceptance_stereo_rot_scale_deg =
        args.stage6_coobs_aware_acceptance_stereo_rot_scale_deg;
    problem_input.solver_options.coobs_aware_acceptance_layout_rot_scale_deg =
        args.stage6_coobs_aware_acceptance_layout_rot_scale_deg;
    problem_input.solver_options.coobs_aware_acceptance_stereo_trans_scale_m =
        args.stage6_coobs_aware_acceptance_stereo_trans_scale_m;
    problem_input.solver_options.coobs_aware_acceptance_layout_trans_scale_m =
        args.stage6_coobs_aware_acceptance_layout_trans_scale_m;
    problem_input.solver_options.coobs_factor_ba_enable =
        args.stage6_coobs_factor_ba_enable;
    problem_input.solver_options.coobs_factor_ba_run_experiment_matrix =
        args.stage6_coobs_factor_ba_run_experiment_matrix;
    problem_input.solver_options.coobs_factor_ba_output_dir_suffix =
        args.stage6_coobs_factor_ba_output_dir_suffix;
    problem_input.solver_options.coobs_factor_ba_min_corners_per_cam_board =
        args.stage6_coobs_factor_ba_min_corners_per_cam_board;
    problem_input.solver_options.coobs_factor_ba_max_local_pose_rmse =
        args.stage6_coobs_factor_ba_max_local_pose_rmse;
    problem_input.solver_options.coobs_factor_ba_huber_delta =
        args.stage6_coobs_factor_ba_huber_delta;
    problem_input.solver_options.coobs_factor_ba_stereo_weights =
        args.stage6_coobs_factor_ba_stereo_weights;
    problem_input.solver_options.coobs_factor_ba_layout_weights =
        args.stage6_coobs_factor_ba_layout_weights;
    problem_input.solver_options.coobs_factor_ba_combined_stereo_weights =
        args.stage6_coobs_factor_ba_combined_stereo_weights;
    problem_input.solver_options.coobs_factor_ba_combined_layout_weights =
        args.stage6_coobs_factor_ba_combined_layout_weights;
    problem_input.solver_options.coobs_factor_ba_layout_selected_pairs =
        args.stage6_coobs_factor_ba_layout_selected_pairs;
    problem_input.solver_options.final_ba_optimize_intrinsics =
        args.stage6_final_ba_optimize_intrinsics;
    problem_input.solver_options.final_ba_optimize_stereo_extrinsic =
        args.stage6_final_ba_optimize_stereo_extrinsic;
    problem_input.solver_options.final_ba_optimize_pair_poses =
        args.stage6_final_ba_optimize_pair_poses;
    problem_input.solver_options.final_ba_optimize_board_poses =
        args.stage6_final_ba_optimize_board_poses;
    problem_input.solver_options.skip_final_global_ba =
        args.stage6_skip_final_global_ba;
    problem_input.solver_options.ba_max_iterations =
        args.stage6_ba_max_iterations;
    problem_input.solver_options.ba_convergence_threshold =
        args.stage6_ba_convergence_threshold;
    problem_input.solver_options.ba_shared_observation_weight_scale =
        args.stage6_ba_shared_observation_weight_scale;
    problem_input.solver_options.ba_single_camera_only_observation_weight_scale =
        args.stage6_ba_single_camera_only_observation_weight_scale;
    problem_input.solver_options.ba_single_camera_only_weight_mode =
        ParseSingleCameraOnlyWeightMode(
            args.stage6_ba_single_camera_only_weight_mode);
    problem_input.solver_options.ba_single_camera_only_base_scale =
        args.stage6_ba_single_camera_only_base_scale;
    problem_input.solver_options.ba_single_camera_only_per_side_budget_ratio =
        args.stage6_ba_single_camera_only_per_side_budget_ratio;
    problem_input.solver_options
        .ba_adaptive_single_camera_only_per_side_cap_ratio =
        args.stage6_ba_adaptive_single_camera_only_per_side_cap_ratio;
    problem_input.solver_options.enable_pair_only_stereo_ba_init =
        args.stage6_enable_pair_only_stereo_ba_init;
    problem_input.solver_options.pair_init_max_iterations =
        args.stage6_pair_init_max_iterations;
    problem_input.solver_options.pair_init_convergence_threshold =
        args.stage6_pair_init_convergence_threshold;
    problem_input.solver_options.pair_init_use_huber_loss =
        args.stage6_pair_init_use_huber_loss;
    problem_input.solver_options.enable_kalibr_style_pair_selection =
        args.stage6_enable_kalibr_style_pair_selection;
    problem_input.solver_options.enable_committing_pair_batch_selection =
        args.stage6_enable_committing_pair_batch_selection;
    problem_input.solver_options.enable_persistent_incremental_stereo_ba =
        args.stage6_enable_persistent_incremental_stereo_ba;
    problem_input.solver_options
        .allow_legacy_selection_fallback_after_persistent_failure =
        args.stage6_allow_legacy_selection_fallback_after_persistent_failure;
    problem_input.solver_options.enable_stage6_incremental_estimator =
        args.stage6_enable_stage6_incremental_estimator ||
        args.stage6_enable_committing_pair_batch_selection ||
        args.stage6_enable_persistent_incremental_stereo_ba;
    problem_input.solver_options.enable_incremental_pair_diversity_rescue =
        args.stage6_enable_incremental_pair_diversity_rescue;
    problem_input.solver_options.incremental_pair_diversity_rescue_min_boards =
        args.stage6_incremental_pair_diversity_rescue_min_boards;
    problem_input.solver_options.incremental_mi_tol =
        args.stage6_incremental_mi_tol;
    problem_input.solver_options.incremental_rank_threshold =
        args.stage6_incremental_rank_threshold;
    problem_input.solver_options.persistent_incremental_max_iterations =
        args.stage6_persistent_incremental_max_iterations;
    problem_input.solver_options.persistent_incremental_convergence_delta_j =
        args.stage6_persistent_incremental_convergence_delta_j;
    problem_input.solver_options.persistent_incremental_convergence_delta_x =
        args.stage6_persistent_incremental_convergence_delta_x;
    problem_input.solver_options
        .persistent_incremental_baseline_prior_translation_weight =
        args.stage6_persistent_incremental_baseline_prior_translation_weight;
    problem_input.solver_options
        .persistent_incremental_baseline_prior_rotation_weight =
        args.stage6_persistent_incremental_baseline_prior_rotation_weight;
    problem_input.solver_options
        .persistent_incremental_projection_prior_shape_sigma =
        args.stage6_persistent_incremental_projection_prior_shape_sigma;
    problem_input.solver_options
        .persistent_incremental_projection_prior_focal_relative_sigma =
        args.stage6_persistent_incremental_projection_prior_focal_relative_sigma;
    problem_input.solver_options
        .persistent_incremental_projection_prior_principal_sigma_px =
        args.stage6_persistent_incremental_projection_prior_principal_sigma_px;
    problem_input.solver_options.adaptive_joint_projection_min_training_pairs =
        args.stage6_adaptive_joint_projection_min_training_pairs;
    problem_input.solver_options
        .adaptive_joint_projection_min_shared_pair_boards =
        args.stage6_adaptive_joint_projection_min_shared_pair_boards;
    problem_input.solver_options.adaptive_joint_projection_min_distinct_boards =
        args.stage6_adaptive_joint_projection_min_distinct_boards;
    problem_input.solver_options
        .adaptive_joint_projection_min_observation_points =
        args.stage6_adaptive_joint_projection_min_observation_points;
    problem_input.solver_options
        .persistent_incremental_invalid_projection_penalty_px =
        args.stage6_persistent_incremental_invalid_projection_penalty_px;
    problem_input.solver_options.persistent_incremental_seed_pair_count =
        args.stage6_persistent_incremental_seed_pair_count;
    problem_input.solver_options.incremental_info_block =
        ParseStage6IncrementalInfoBlock(args.stage6_incremental_info_block);
    problem_input.solver_options.batch_acceptance_policy =
        ati::ParseKalibrStyleBatchAcceptancePolicy(
            args.stage6_batch_acceptance_policy);
    problem_input.solver_options.pair_selection_seed_count =
        args.stage6_pair_selection_seed_count;
    problem_input.solver_options.pair_selection_budget_mode =
        args.stage6_pair_selection_budget_mode_set
            ? ParseCandidateBudgetMode(
                  args.stage6_pair_selection_budget_mode,
                  "--stage6-pair-selection-budget-mode")
            : ati::StereoCandidateBudgetMode::KalibrStyle;
    problem_input.solver_options.pair_selection_max_candidate_additions =
        args.stage6_pair_selection_max_candidate_additions;
    problem_input.solver_options.pair_selection_adaptive_budget_ratio =
        args.stage6_pair_selection_adaptive_budget_ratio;
    problem_input.solver_options.pair_selection_adaptive_budget_min =
        args.stage6_pair_selection_adaptive_budget_min;
    problem_input.solver_options.pair_selection_adaptive_budget_max =
        args.stage6_pair_selection_adaptive_budget_max;
    problem_input.solver_options.pair_selection_runtime_safety_ceiling =
        args.stage6_pair_selection_runtime_safety_ceiling;
    problem_input.solver_options.pair_selection_min_shared_boards =
        args.stage6_pair_selection_min_shared_boards;
    problem_input.solver_options.pair_selection_max_rmse_delta =
        args.stage6_pair_selection_max_rmse_delta;
    problem_input.solver_options.pair_selection_max_camera_rmse_delta =
        args.stage6_pair_selection_max_camera_rmse_delta;
    problem_input.solver_options
        .pair_selection_max_baseline_rotation_delta_deg =
        args.stage6_pair_selection_max_baseline_rotation_delta_deg;
    problem_input.solver_options
        .pair_selection_max_baseline_translation_delta_m =
        args.stage6_pair_selection_max_baseline_translation_delta_m;
    problem_input.solver_options.enable_pair_board_trial_selection =
        args.stage6_enable_pair_board_trial_selection;
    const ati::StereoPairBoardSelectionMode pairboard_selection_mode =
        ParsePairBoardSelectionMode(args.stage6_pairboard_selection_mode);
    problem_input.solver_options.pairboard_selection_mode =
        pairboard_selection_mode;
    problem_input.solver_options.pair_board_selection_budget_mode =
        args.stage6_pair_board_selection_budget_mode_set
            ? ParseCandidateBudgetMode(
                  args.stage6_pair_board_selection_budget_mode,
                  "--stage6-pair-board-selection-budget-mode")
            : (pairboard_selection_mode ==
                       ati::StereoPairBoardSelectionMode::StrictRmse
                   ? ati::StereoCandidateBudgetMode::Fixed
                   : ati::StereoCandidateBudgetMode::KalibrStyle);
    problem_input.solver_options.pair_board_selection_seed_count =
        args.stage6_pair_board_selection_seed_count;
    problem_input.solver_options.pair_board_selection_max_candidate_additions =
        args.stage6_pair_board_selection_max_candidate_additions;
    problem_input.solver_options.pair_board_selection_adaptive_budget_ratio =
        args.stage6_pair_board_selection_adaptive_budget_ratio;
    problem_input.solver_options.pair_board_selection_adaptive_budget_min =
        args.stage6_pair_board_selection_adaptive_budget_min;
    problem_input.solver_options.pair_board_selection_adaptive_budget_max =
        args.stage6_pair_board_selection_adaptive_budget_max;
    problem_input.solver_options.pair_board_selection_runtime_safety_ceiling =
        args.stage6_pair_board_selection_runtime_safety_ceiling;
    problem_input.solver_options.pair_board_selection_min_candidate_score =
        args.stage6_pair_board_selection_min_candidate_score;
    problem_input.solver_options.pair_board_selection_min_coverage_gain =
        args.stage6_pair_board_selection_min_coverage_gain;
    problem_input.solver_options.pair_board_selection_max_accepted_per_pair =
        args.stage6_pair_board_selection_max_accepted_per_pair;
    problem_input.solver_options.pair_board_selection_max_accepted_per_board =
        args.stage6_pair_board_selection_max_accepted_per_board;
    problem_input.solver_options.pair_board_selection_max_rmse_delta =
        args.stage6_pair_board_selection_max_rmse_delta;
    problem_input.solver_options.pair_board_selection_max_camera_rmse_delta =
        args.stage6_pair_board_selection_max_camera_rmse_delta;
    problem_input.solver_options
        .pair_board_selection_max_baseline_rotation_delta_deg =
        args.stage6_pair_board_selection_max_baseline_rotation_delta_deg;
    problem_input.solver_options
        .pair_board_selection_max_baseline_translation_delta_m =
        args.stage6_pair_board_selection_max_baseline_translation_delta_m;
    problem_input.solver_options.enable_pair_cohesion =
        args.stage6_enable_pair_cohesion;
    problem_input.solver_options.pair_cohesion_min_boards_per_pair =
        args.stage6_pair_cohesion_min_boards_per_pair;
    problem_input.solver_options.pair_cohesion_max_companions_per_pair =
        args.stage6_pair_cohesion_max_companions_per_pair;
    problem_input.solver_options.pair_cohesion_relax_score_gate =
        args.stage6_pair_cohesion_relax_score_gate;
    problem_input.solver_options.pair_cohesion_relax_cap_gates =
        args.stage6_pair_cohesion_relax_cap_gates;
    problem_input.solver_options.single_board_pair_policy =
        ParseSingleBoardPairPolicy(args.stage6_single_board_pair_policy);
    problem_input.solver_options.ablation_excluded_pair_boards =
        args.stage6_ablation_excluded_pair_boards;
    problem_input.solver_options.export_stereo_reprojection_visualizations =
        args.stage6_export_stereo_reprojection_visualizations;
    problem_input.solver_options.stereo_visualization_top_k =
        args.stage6_stereo_visualization_top_k;
    problem_input.solver_options.export_extrinsic_uncertainty_diagnostics =
        args.stage6_export_extrinsic_uncertainty_diagnostics;
    problem_input.solver_options.export_angular_fixedk_diagnostic =
        args.stage6_export_angular_fixedk_diagnostic;
    problem_input.solver_options.board_masking_use_local_board_pose_ba =
        args.stage6_board_masking_ablation == "split_pair_boards";
    problem_input.solver_options.pair_pose_refit_mode =
        ati::StereoPairPoseRefitMode::StereoSymmetric;

    ati::StereoExtrinsicCalibrationRunner runner(problem_input.solver_options);
    std::cout << "[Stage6] starting stereo extrinsic runner..." << std::endl;
    const ati::Stage6RuntimeSummary frontend_runtime_summary =
        result.runtime_summary;
    result = runner.Run(problem_input);
    result.runtime_summary.cache_dir = frontend_runtime_summary.cache_dir;
    result.runtime_summary.cache_enabled = frontend_runtime_summary.cache_enabled;
    result.runtime_summary.cam0_training_detection_cache_hits =
        frontend_runtime_summary.cam0_training_detection_cache_hits;
    result.runtime_summary.cam0_training_detection_cache_misses =
        frontend_runtime_summary.cam0_training_detection_cache_misses;
    result.runtime_summary.cam0_training_detection_cache_load_failures =
        frontend_runtime_summary.cam0_training_detection_cache_load_failures;
    result.runtime_summary.cam0_training_detection_cache_store_failures =
        frontend_runtime_summary.cam0_training_detection_cache_store_failures;
    result.runtime_summary.cam1_training_detection_cache_hits =
        frontend_runtime_summary.cam1_training_detection_cache_hits;
    result.runtime_summary.cam1_training_detection_cache_misses =
        frontend_runtime_summary.cam1_training_detection_cache_misses;
    result.runtime_summary.cam1_training_detection_cache_load_failures =
        frontend_runtime_summary.cam1_training_detection_cache_load_failures;
    result.runtime_summary.cam1_training_detection_cache_store_failures =
        frontend_runtime_summary.cam1_training_detection_cache_store_failures;
    result.runtime_summary.frontend_pairing_prefilter_enabled =
        frontend_runtime_summary.frontend_pairing_prefilter_enabled;
    result.runtime_summary.frontend_original_left_frame_count =
        frontend_runtime_summary.frontend_original_left_frame_count;
    result.runtime_summary.frontend_original_right_frame_count =
        frontend_runtime_summary.frontend_original_right_frame_count;
    result.runtime_summary.frontend_processed_left_frame_count =
        frontend_runtime_summary.frontend_processed_left_frame_count;
    result.runtime_summary.frontend_processed_right_frame_count =
        frontend_runtime_summary.frontend_processed_right_frame_count;
    result.runtime_summary.frontend_skipped_unpaired_left_frame_count =
        frontend_runtime_summary.frontend_skipped_unpaired_left_frame_count;
    result.runtime_summary.frontend_skipped_unpaired_right_frame_count =
        frontend_runtime_summary.frontend_skipped_unpaired_right_frame_count;
    result.runtime_summary.pairing_build_dataset_runtime_seconds =
        std::chrono::duration<double>(Clock::now() - total_start).count() -
        result.runtime_summary.initialization_runtime_seconds -
        result.runtime_summary.training_optimization_runtime_seconds -
        result.runtime_summary.holdout_evaluation_runtime_seconds;

    if (result.success) {
      ati::WriteStereoExtrinsicYaml(
          (output_dir / "stereo_extrinsic.yaml").string(), result);
      ati::WriteStereoExtrinsicSummary(
          (output_dir / "stereo_extrinsic_summary.txt").string(), result);
    }
    if (result.success && !args.stereo_reference_camchain_path.empty()) {
      const StereoReferenceComparison reference_comparison =
          CompareAgainstStereoReference(args.stereo_reference_camchain_path,
                                        result.optimized_scene.T_cam1_cam0);
      WriteStereoReferenceComparison(
          (output_dir / "stereo_reference_comparison.txt").string(),
          reference_comparison);
      if (reference_comparison.success) {
        ati::StereoSceneState reference_scene = result.optimized_scene;
        reference_scene.T_cam1_cam0 =
            reference_comparison.reference_T_cam1_cam0;
        ati::StereoResidualEvaluator reference_extrinsic_only_holdout_evaluator(
            ati::StereoResidualEvaluationOptions{
                false,
                problem_input.solver_options.pair_pose_refit_mode,
                problem_input.solver_options.symmetric_refit_max_iterations,
                problem_input.solver_options.symmetric_refit_step,
                true});
        const ati::StereoResidualSummary
            reference_extrinsic_only_holdout_summary =
                reference_extrinsic_only_holdout_evaluator.Evaluate(
                    problem_input.measurement_dataset, reference_scene,
                    std::set<int>(
                        problem_input.measurement_dataset.holdout_pair_indices.begin(),
                        problem_input.measurement_dataset.holdout_pair_indices.end()),
                    "reference_holdout_extrinsic_only");
        WriteStereoReferenceHoldoutSummary(
            (output_dir / "stereo_reference_holdout_summary.txt").string(),
            reference_comparison,
            result.holdout_extrinsic_only_residual_summary,
            reference_extrinsic_only_holdout_summary);
        WriteStereoResidualPerBoardCsv(
            (output_dir /
             "stereo_reference_extrinsic_only_per_board_residuals.csv")
                .string(),
            "reference_holdout_extrinsic_only",
            reference_extrinsic_only_holdout_summary);
        if (!args.stereo_reference_left_intrinsics_path.empty() &&
            !args.stereo_reference_right_intrinsics_path.empty()) {
          ati::StereoSceneState full_reference_scene = reference_scene;
          full_reference_scene.cam0 = ToStereoCalibration(
              ati::LoadExternalCameraConfig(
                  args.stereo_reference_left_intrinsics_path),
              args.stereo_reference_left_intrinsics_path);
          full_reference_scene.cam1 = ToStereoCalibration(
              ati::LoadExternalCameraConfig(
                  args.stereo_reference_right_intrinsics_path),
              args.stereo_reference_right_intrinsics_path);
          const ati::StereoResidualSummary full_reference_holdout_summary =
              reference_extrinsic_only_holdout_evaluator.Evaluate(
                  problem_input.measurement_dataset, full_reference_scene,
                  std::set<int>(
                      problem_input.measurement_dataset.holdout_pair_indices.begin(),
                      problem_input.measurement_dataset.holdout_pair_indices.end()),
                  "reference_holdout_full_camera");
          WriteStereoFullReferenceHoldoutSummary(
              (output_dir / "stereo_reference_full_holdout_summary.txt").string(),
              reference_comparison,
              args.stereo_reference_left_intrinsics_path,
              args.stereo_reference_right_intrinsics_path,
              result.holdout_extrinsic_only_residual_summary,
              full_reference_holdout_summary);
          WriteStereoResidualPerBoardCsv(
              (output_dir /
               "stereo_reference_full_per_board_residuals.csv")
                  .string(),
              "reference_holdout_full_camera",
              full_reference_holdout_summary);
        }
        if (problem_input.solver_options.export_stereo_reprojection_visualizations) {
          ati::WriteStereoExtrinsicOnlyTopBadPairBoardVisualizations(
              (output_dir / "stereo_reprojection_visualizations" /
               "reference_holdout_extrinsic_only_top_bad_pair_boards")
                  .string(),
              result,
              reference_scene,
              "reference_holdout_extrinsic_only_top_bad_pair_boards",
              problem_input.solver_options.stereo_visualization_top_k);
        }
      }
    }
    if (result.success) {
      ati::WriteStereoReprojectionSummary(
          (output_dir / "stereo_reprojection_summary.txt").string(), result);
      ati::WriteStereoPerCameraResidualsCsv(
          (output_dir / "stereo_per_camera_residuals.csv").string(), result);
      ati::WriteStereoPerFrameResidualsCsv(
          (output_dir / "stereo_per_frame_residuals.csv").string(), result);
      ati::WriteStereoHoldoutLayoutTransferGapCsv(
          (output_dir / "stereo_holdout_layout_transfer_gap.csv").string(),
          result);
      ati::WriteStereoHoldoutLocalLayoutDriftCsv(
          (output_dir / "stereo_holdout_local_layout_drift.csv").string(),
          result);
      ati::WriteStereoBaFrameFactorTraceCsv(
          (output_dir / "stereo_ba_frame_factor_trace.csv").string(), result);
      ati::WriteStereoJacobianBlockDiagnosticsCsv(
          (output_dir / "stereo_jacobian_block_diagnostics.csv").string(),
          result);
      ati::WriteStereoPerBoardResidualsCsv(
          (output_dir / "stereo_per_board_residuals.csv").string(), result);
      ati::WriteStereoHoldoutBoardPolarRmseCsv(
          (output_dir / "stereo_holdout_board_polar_rmse.csv").string(),
          result);
      ati::WriteStereoIntrinsicsSanitySummary(
          (output_dir / "stereo_intrinsics_sanity_summary.txt").string(), result);
    }
    if (result.success && problem_input.solver_options.coobs_enable) {
      ati::MultiBoardCoObservationOptions coobs_options;
      coobs_options.enabled = true;
      coobs_options.output_dir =
          problem_input.solver_options.coobs_output_dir.empty()
              ? (output_dir / "coobs_diagnostics").string()
              : problem_input.solver_options.coobs_output_dir;
      coobs_options.min_corners_per_group =
          problem_input.solver_options.coobs_min_corners_per_group;
      coobs_options.high_polar_threshold_deg =
          problem_input.solver_options.coobs_high_polar_threshold_deg;
      coobs_options.very_high_polar_threshold_deg =
          problem_input.solver_options.coobs_very_high_polar_threshold_deg;
      coobs_options.enable_rescue_suggestions =
          problem_input.solver_options.coobs_enable_rescue_suggestions;
      coobs_options.score_alpha_high_polar =
          problem_input.solver_options.coobs_score_alpha_high_polar;
      coobs_options.score_beta_multiboard =
          problem_input.solver_options.coobs_score_beta_multiboard;
      coobs_options.score_gamma_balance =
          problem_input.solver_options.coobs_score_gamma_balance;
      coobs_options.score_eta_conflict =
          problem_input.solver_options.coobs_score_eta_conflict;
      coobs_options.rescue_min_high_polar_score =
          problem_input.solver_options.coobs_rescue_min_high_polar_score;
      coobs_options.rescue_bad_conflict_threshold =
          problem_input.solver_options.coobs_rescue_bad_conflict_threshold;
      const ati::MultiBoardCoObservationConsistency coobs(coobs_options);
      coobs.Evaluate(result);
    }
    if (result.success && problem_input.solver_options.coobs_factor_ba_enable) {
      const fs::path coobs_factor_dir =
          output_dir /
          problem_input.solver_options.coobs_factor_ba_output_dir_suffix;
      ati::WriteCoObsFactorBaExperiment(coobs_factor_dir.string(), result);
    }
    const bool write_angular_fixedk_diagnostic =
        problem_input.solver_options.export_angular_fixedk_diagnostic ||
        problem_input.solver_options.final_ba_residual_mode !=
            ati::StereoFinalBaResidualMode::Pixel;
    if (result.success && write_angular_fixedk_diagnostic) {
      const fs::path angular_dir =
          output_dir / "stage6_angular_fixedK_diagnostic";
      fs::create_directories(angular_dir);
      ati::WriteStereoAngularFixedKSummary(
          (angular_dir / "angular_diagnostic_summary.csv").string(),
          result);
      ati::WriteStereoAngularFixedKCornerTraceCsv(
          (angular_dir / "angular_diagnostic_corner_trace.csv").string(),
          result);
    }
    ati::WriteStereoPairingSummary(
        (output_dir / "stereo_pairing_summary.txt").string(), problem_input);
    ati::WriteStereoInitializationSummary(
        (output_dir / "stereo_initialization_summary.txt").string(), result);
    if (problem_input.solver_options.enable_persistent_incremental_stereo_ba) {
      ati::WriteStereoInitializationSummary(
          (output_dir / "stage6_init_summary.txt").string(), result);
    }
    ati::WriteStereoGraphSummary(
        (output_dir / "stereo_graph_summary.txt").string(), result);
    ati::WriteStereoPairSelectionSummary(
        (output_dir / "stereo_pair_selection_summary.txt").string(), result);
    ati::WriteStereoPairSelectionCsv(
        (output_dir / "stereo_pair_selection.csv").string(), result);
    if (problem_input.solver_options.enable_pair_only_stereo_ba_init) {
      ati::WriteStereoPairInitSummary(
          (output_dir / "stereo_pair_init_summary.txt").string(), result);
      ati::WriteStereoPairInitCandidatesCsv(
          (output_dir / "stereo_pair_init_candidates.csv").string(), result);
      ati::WriteStereoPairInitResidualsCsv(
          (output_dir / "stereo_pair_init_residuals_before_after.csv").string(),
          result);
    }
    if (problem_input.solver_options.enable_kalibr_style_pair_selection ||
        problem_input.solver_options.enable_committing_pair_batch_selection ||
        problem_input.solver_options.enable_persistent_incremental_stereo_ba) {
      ati::WriteStereoPairTrialSelectionSummary(
          (output_dir / "stereo_pair_trial_selection_summary.txt").string(),
          result);
      ati::WriteStereoPairTrialSelectionDecisionsCsv(
          (output_dir / "stereo_pair_trial_selection_decisions.csv").string(),
          result);
      ati::WriteStereoPairTrialSelectedPairsCsv(
          (output_dir / "stereo_pair_trial_selected_pairs.csv").string(), result);
    }
    if (problem_input.solver_options.enable_pair_board_trial_selection ||
        problem_input.solver_options.enable_committing_pair_batch_selection ||
        problem_input.solver_options.enable_persistent_incremental_stereo_ba) {
      ati::WriteStereoPairBoardTrialSelectionSummary(
          (output_dir / "stereo_pair_board_trial_selection_summary.txt").string(),
          result);
      ati::WriteStereoPairBoardTrialSelectionDecisionsCsv(
          (output_dir / "stereo_pair_board_trial_selection_decisions.csv").string(),
          result);
      ati::WriteStereoPairBoardTrialSelectedBoardsCsv(
          (output_dir / "stereo_pair_board_trial_selected_boards.csv").string(),
          result);
      if (problem_input.solver_options.enable_persistent_incremental_stereo_ba) {
        ati::WriteStereoPairBoardTrialSelectionSummary(
            (output_dir / "stage6_persistent_incremental_selection_summary.txt")
                .string(),
            result);
        ati::WriteStage6PersistentIncrementalBatchDecisionsCsv(
            (output_dir / "stage6_persistent_incremental_batch_decisions.csv")
                .string(),
            result);
        ati::WriteStereoPairBoardTrialSelectionDecisionsCsv(
            (output_dir /
             "stage6_persistent_incremental_pair_board_decisions.csv")
                .string(),
            result);
        ati::WriteStereoPairBoardTrialSelectedBoardsCsv(
            (output_dir / "stage6_persistent_incremental_selected_boards.csv")
                .string(),
            result);
      }
    }
    ati::WriteStereoGlobalSparseBaSummary(
        (output_dir / "stereo_global_sparse_ba_summary.txt").string(), result);
    ati::WriteStereoGlobalSparseBaInitialVsFinal(
        (output_dir / "stereo_global_sparse_ba_initial_vs_final.txt").string(),
        result);
    if (result.success &&
        (problem_input.solver_options.export_pair_board_consistency_audit ||
         problem_input.solver_options.enable_pair_board_consistency_gate)) {
      ati::WriteStereoPairBoardConsistencySummary(
          (output_dir / "stereo_pair_board_consistency_summary.txt").string(),
          result);
      ati::WriteStereoPairBoardLocalGlobalGapSummary(
          (output_dir / "stereo_pair_board_local_global_gap_summary.txt").string(),
          result);
      ati::WriteStereoPairBoardConsistencyCsv(
          (output_dir / "stereo_pair_board_consistency.csv").string(), result);
    }
    if (result.success &&
        (problem_input.solver_options.enable_shared_board_quality_gate ||
         problem_input.solver_options.enable_shared_board_quality_hard_gate)) {
      ati::WriteStereoSharedBoardQualityAuditCsv(
          (output_dir / "stereo_shared_board_quality_audit.csv").string(),
          result);
    }
    ati::WriteStage6RuntimeSummary(
        (output_dir / "stage6_runtime_summary.txt").string(), result);
    if (problem_input.solver_options.enable_pair_only_stereo_ba_init ||
        problem_input.solver_options.enable_kalibr_style_pair_selection ||
        problem_input.solver_options.enable_pair_board_trial_selection ||
        problem_input.solver_options.enable_committing_pair_batch_selection ||
        problem_input.solver_options.export_stereo_reprojection_visualizations ||
        problem_input.solver_options.export_extrinsic_uncertainty_diagnostics) {
      ati::WriteStereoRobustLossSummary(
          (output_dir / "stereo_robust_loss_summary.txt").string(), result);
    }
    if (result.success &&
        problem_input.solver_options.export_extrinsic_uncertainty_diagnostics) {
      ati::WriteStereoExtrinsicUncertaintySummary(
          (output_dir / "stereo_extrinsic_uncertainty_summary.txt").string(),
          result);
      ati::WriteStereoExtrinsicCandidateDispersionCsv(
          (output_dir / "stereo_extrinsic_candidate_dispersion.csv").string(),
          result);
      ati::WriteStereoExtrinsicJackknifeCsv(
          (output_dir / "stereo_extrinsic_jackknife.csv").string(), result);
    }
    if (result.success &&
        problem_input.solver_options.export_stereo_reprojection_visualizations) {
      ati::WriteStereoReprojectionVisualizations(
          (output_dir / "stereo_reprojection_visualizations" /
           "overview_side_by_side")
              .string(),
          result,
          problem_input.solver_options.stereo_visualization_top_k);
      ati::WriteStereoExtrinsicOnlyTopBadPairBoardVisualizations(
          (output_dir / "stereo_reprojection_visualizations" /
           "ours_holdout_extrinsic_only_top_bad_pair_boards")
              .string(),
          result,
          result.optimized_scene,
          "ours_holdout_extrinsic_only_top_bad_pair_boards",
          problem_input.solver_options.stereo_visualization_top_k);
    }
    if (result.success &&
        (problem_input.solver_options.export_stereo_reprojection_visualizations ||
         problem_input.solver_options.enable_persistent_incremental_stereo_ba)) {
      ati::WriteStereoBackendInputVisualizations(
          (output_dir / "stereo_backend_input_visualizations").string(),
          result,
          problem_input.solver_options.stereo_visualization_top_k);
    }

    std::cout << "Stage6 stereo extrinsic success: " << (result.success ? 1 : 0)
              << "\n";
    std::cout << "Stereo extrinsic yaml: "
              << (output_dir / "stereo_extrinsic.yaml").string() << "\n";
    std::cout << "Stereo summary: "
              << (output_dir / "stereo_extrinsic_summary.txt").string() << "\n";
    std::cout << "Stereo reprojection summary: "
              << (output_dir / "stereo_reprojection_summary.txt").string() << "\n";
    if (write_angular_fixedk_diagnostic) {
      std::cout << "Stage6 angular fixed-K diagnostic summary: "
                << (output_dir / "stage6_angular_fixedK_diagnostic" /
                    "angular_diagnostic_summary.csv")
                       .string()
                << "\n";
      std::cout << "Stage6 angular fixed-K diagnostic trace: "
                << (output_dir / "stage6_angular_fixedK_diagnostic" /
                    "angular_diagnostic_corner_trace.csv")
                       .string()
                << "\n";
    }
    return result.success ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
