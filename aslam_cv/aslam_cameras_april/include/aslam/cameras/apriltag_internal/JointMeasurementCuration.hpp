#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_MEASUREMENT_CURATION_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_MEASUREMENT_CURATION_HPP

#include <string>
#include <map>
#include <vector>

#include <Eigen/Core>

#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

enum class PreBackendFilterMode {
  Off,
  Diagnostic,
  Enabled,
};

const char* ToString(PreBackendFilterMode mode);
PreBackendFilterMode ParsePreBackendFilterMode(const std::string& value);

enum class PreBackendFilterThresholdMode {
  MeanStd,
  MedianMad,
};

const char* ToString(PreBackendFilterThresholdMode mode);
PreBackendFilterThresholdMode ParsePreBackendFilterThresholdMode(
    const std::string& value);

enum class InternalBlurFilterMode {
  Off,
  Diagnostic,
  Enabled,
};

const char* ToString(InternalBlurFilterMode mode);
InternalBlurFilterMode ParseInternalBlurFilterMode(const std::string& value);

enum class InternalObservationWeightMode {
  Off,
  Diagnostic,
  Enabled,
};

const char* ToString(InternalObservationWeightMode mode);
InternalObservationWeightMode ParseInternalObservationWeightMode(
    const std::string& value);

enum class InternalBlurBoardWeightMode {
  Off,
  Diagnostic,
  Enabled,
};

const char* ToString(InternalBlurBoardWeightMode mode);
InternalBlurBoardWeightMode ParseInternalBlurBoardWeightMode(
    const std::string& value);

enum class InternalJointRefineMode {
  Off,
  Diagnostic,
  Enabled,
};

const char* ToString(InternalJointRefineMode mode);
InternalJointRefineMode ParseInternalJointRefineMode(
    const std::string& value);

enum class InternalJointRefineTargetMode {
  All,
  HighResidualOnly,
  BlurBadBoardOnly,
  HighResidualOrBlurBadBoard,
  HighResidualAndBlurBadBoard,
};

const char* ToString(InternalJointRefineTargetMode mode);
InternalJointRefineTargetMode ParseInternalJointRefineTargetMode(
    const std::string& value);

struct PreBackendObservationFilterOptions {
  PreBackendFilterMode mode = PreBackendFilterMode::Off;
  PreBackendFilterThresholdMode threshold_mode =
      PreBackendFilterThresholdMode::MeanStd;
  double sigma_threshold = 2.0;
  double min_abs_threshold_px = 0.2;
};

struct PreBackendFilterPointDecision {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind = JointObservationSourceKind::InternalMeasurement;
  Eigen::Vector2d observed_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d predicted_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d residual_xy = Eigen::Vector2d::Zero();
  double residual_norm = 0.0;
  double board_mean_residual = 0.0;
  double board_std_residual = 0.0;
  double board_median_residual = 0.0;
  double board_mad_residual = 0.0;
  double board_threshold = 0.0;
  bool valid_projection = false;
  bool filtered = false;
  std::string filter_reason;
};

struct PreBackendFilterBoardSummary {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int input_internal_point_count = 0;
  int filtered_internal_point_count = 0;
  int remaining_internal_point_count = 0;
  double mean_residual = 0.0;
  double std_residual = 0.0;
  double median_residual = 0.0;
  double mad_residual = 0.0;
  double threshold = 0.0;
  double filtered_ratio = 0.0;
};

struct PreBackendFilterFrameSummary {
  int frame_index = -1;
  std::string frame_label;
  int board_observation_count = 0;
  int affected_board_count = 0;
  int input_internal_point_count = 0;
  int filtered_internal_point_count = 0;
  int remaining_internal_point_count = 0;
  double filtered_ratio = 0.0;
};

struct PreBackendObservationFilterResult {
  bool success = false;
  PreBackendObservationFilterOptions options;
  bool diagnostic_only = true;
  bool backend_input_changed = false;
  int input_internal_point_count = 0;
  int filtered_internal_point_count = 0;
  int remaining_internal_point_count = 0;
  double filtered_ratio = 0.0;
  int affected_board_count = 0;
  int affected_frame_count = 0;
  CalibrationStateBundle curated_bundle;
  std::vector<PreBackendFilterPointDecision> point_decisions;
  std::vector<PreBackendFilterBoardSummary> board_summaries;
  std::vector<PreBackendFilterFrameSummary> frame_summaries;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

PreBackendObservationFilterResult ApplyPreBackendObservationFilter(
    const CalibrationStateBundle& bundle,
    const PreBackendObservationFilterOptions& options);

void WritePreBackendFilterSummary(
    const std::string& path,
    const PreBackendObservationFilterResult& result);
void WritePreBackendFilterPointsCsv(
    const std::string& path,
    const PreBackendObservationFilterResult& result);
void WritePreBackendFilterBoardSummaryCsv(
    const std::string& path,
    const PreBackendObservationFilterResult& result);
void WritePreBackendFilterFrameSummaryCsv(
    const std::string& path,
    const PreBackendObservationFilterResult& result);

struct InternalBlurObservationFilterOptions {
  InternalBlurFilterMode mode = InternalBlurFilterMode::Off;
  double low_patch_gradient_quantile = 0.05;
  double min_board_internal_rmse_px = 5.0;
  double min_board_p95_residual_px = 5.0;
};

struct InternalBlurFilterPointDecision {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind =
      JointObservationSourceKind::InternalMeasurement;
  Eigen::Vector2d observed_image_xy = Eigen::Vector2d::Zero();
  double residual_norm = 0.0;
  bool filtered = false;
  std::string filter_reason;
};

struct InternalBlurFilterBoardDecision {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int input_internal_point_count = 0;
  int filtered_internal_point_count = 0;
  int remaining_internal_point_count = 0;
  double internal_rmse = 0.0;
  double max_residual = 0.0;
  double p90_residual = 0.0;
  double p95_residual = 0.0;
  double corner_patch_mean_gradient = 0.0;
  double patch_gradient_threshold = 0.0;
  bool low_patch_gradient = false;
  bool high_internal_residual = false;
  bool filtered = false;
  std::string filter_reason;
};

struct InternalBlurFilterFrameSummary {
  int frame_index = -1;
  std::string frame_label;
  int board_observation_count = 0;
  int filtered_board_count = 0;
  int input_internal_point_count = 0;
  int filtered_internal_point_count = 0;
  int remaining_internal_point_count = 0;
  double filtered_ratio = 0.0;
};

struct InternalBlurObservationFilterResult {
  bool success = false;
  InternalBlurObservationFilterOptions options;
  bool diagnostic_only = true;
  bool backend_input_changed = false;
  int input_internal_point_count = 0;
  int filtered_internal_point_count = 0;
  int remaining_internal_point_count = 0;
  double filtered_ratio = 0.0;
  int input_board_observation_count = 0;
  int filtered_board_observation_count = 0;
  int affected_frame_count = 0;
  double patch_gradient_threshold = 0.0;
  CalibrationStateBundle curated_bundle;
  std::vector<InternalBlurFilterBoardDecision> board_decisions;
  std::vector<InternalBlurFilterPointDecision> point_decisions;
  std::vector<InternalBlurFilterFrameSummary> frame_summaries;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

InternalBlurObservationFilterResult ApplyInternalBlurObservationFilter(
    const CalibrationStateBundle& bundle,
    const std::map<int, std::string>& frame_image_paths,
    const InternalBlurObservationFilterOptions& options);

void WriteInternalBlurFilterSummary(
    const std::string& path,
    const InternalBlurObservationFilterResult& result);
void WriteInternalBlurFilterBoardDecisionsCsv(
    const std::string& path,
    const InternalBlurObservationFilterResult& result);
void WriteInternalBlurFilterPointDecisionsCsv(
    const std::string& path,
    const InternalBlurObservationFilterResult& result);
void WriteInternalBlurFilterFrameSummaryCsv(
    const std::string& path,
    const InternalBlurObservationFilterResult& result);

struct InternalObservationWeightOptions {
  InternalObservationWeightMode mode = InternalObservationWeightMode::Off;
  std::string policy = "quality";
  double low_quality_quantile = 0.2;
  double min_weight = 0.25;
  double quality_exponent = 1.0;
  double residual_consistency_sigma_multiplier = 2.0;
  double residual_consistency_min_rmse = 0.5;
};

struct InternalObservationWeightDecision {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind =
      JointObservationSourceKind::InternalMeasurement;
  double quality = 0.0;
  double board_internal_rmse = 0.0;
  double board_outer_rmse = 0.0;
  double residual_consistency_ratio = 0.0;
  double input_weight = 1.0;
  double output_weight = 1.0;
  bool downweighted = false;
  std::string weight_reason;
};

struct InternalObservationWeightBoardSummary {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int internal_point_count = 0;
  int downweighted_internal_point_count = 0;
  double min_weight = 1.0;
  double mean_weight = 1.0;
  double max_weight = 1.0;
  double mean_quality = 0.0;
  double board_internal_rmse = 0.0;
  double board_outer_rmse = 0.0;
  double residual_consistency_ratio = 0.0;
};

struct InternalObservationWeightResult {
  bool success = false;
  InternalObservationWeightOptions options;
  bool diagnostic_only = true;
  bool backend_input_changed = false;
  int input_internal_point_count = 0;
  int downweighted_internal_point_count = 0;
  double downweighted_ratio = 0.0;
  double quality_threshold = 0.0;
  std::string policy = "quality";
  double residual_consistency_sigma_multiplier = 0.0;
  double residual_consistency_min_rmse = 0.0;
  double residual_consistency_ratio_threshold = 0.0;
  double min_weight = 1.0;
  double mean_weight = 1.0;
  double max_weight = 1.0;
  CalibrationStateBundle curated_bundle;
  std::vector<InternalObservationWeightDecision> point_decisions;
  std::vector<InternalObservationWeightBoardSummary> board_summaries;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

InternalObservationWeightResult ApplyInternalObservationWeights(
    const CalibrationStateBundle& bundle,
    const InternalObservationWeightOptions& options);

void WriteInternalObservationWeightSummary(
    const std::string& path,
    const InternalObservationWeightResult& result);
void WriteInternalObservationWeightsCsv(
    const std::string& path,
    const InternalObservationWeightResult& result);
void WriteInternalObservationWeightBoardSummaryCsv(
    const std::string& path,
    const InternalObservationWeightResult& result);

struct InternalBlurBoardWeightOptions {
  InternalBlurBoardWeightMode mode = InternalBlurBoardWeightMode::Off;
  double low_patch_gradient_quantile = 0.05;
  double min_board_internal_rmse_px = 5.0;
  double min_board_p95_residual_px = 5.0;
  double min_weight = 0.25;
  double gradient_exponent = 1.0;
};

struct InternalBlurBoardWeightPointDecision {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind =
      JointObservationSourceKind::InternalMeasurement;
  double board_corner_patch_mean_gradient = 0.0;
  double board_patch_gradient_threshold = 0.0;
  double board_internal_rmse = 0.0;
  double board_p95_residual = 0.0;
  double input_weight = 1.0;
  double output_weight = 1.0;
  bool downweighted = false;
  std::string weight_reason;
};

struct InternalBlurBoardWeightBoardSummary {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int internal_point_count = 0;
  int downweighted_internal_point_count = 0;
  double internal_rmse = 0.0;
  double p95_residual = 0.0;
  double corner_patch_mean_gradient = 0.0;
  double patch_gradient_threshold = 0.0;
  bool low_patch_gradient = false;
  bool high_internal_residual = false;
  bool targeted_for_downweight = false;
  double input_weight = 1.0;
  double output_weight = 1.0;
  std::string weight_reason;
};

struct InternalBlurBoardWeightResult {
  bool success = false;
  InternalBlurBoardWeightOptions options;
  bool diagnostic_only = true;
  bool backend_input_changed = false;
  int input_board_observation_count = 0;
  int downweighted_board_observation_count = 0;
  int input_internal_point_count = 0;
  int downweighted_internal_point_count = 0;
  double downweighted_internal_ratio = 0.0;
  double patch_gradient_threshold = 0.0;
  double min_weight = 1.0;
  double mean_weight = 1.0;
  double max_weight = 1.0;
  CalibrationStateBundle curated_bundle;
  std::vector<InternalBlurBoardWeightPointDecision> point_decisions;
  std::vector<InternalBlurBoardWeightBoardSummary> board_summaries;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

InternalBlurBoardWeightResult ApplyInternalBlurBoardWeights(
    const CalibrationStateBundle& bundle,
    const std::map<int, std::string>& frame_image_paths,
    const InternalBlurBoardWeightOptions& options);

void WriteInternalBlurBoardWeightSummary(
    const std::string& path,
    const InternalBlurBoardWeightResult& result);
void WriteInternalBlurBoardWeightPointsCsv(
    const std::string& path,
    const InternalBlurBoardWeightResult& result);
void WriteInternalBlurBoardWeightBoardSummaryCsv(
    const std::string& path,
    const InternalBlurBoardWeightResult& result);

struct InternalJointRefineOptions {
  InternalJointRefineMode mode = InternalJointRefineMode::Off;
  InternalJointRefineTargetMode target_mode =
      InternalJointRefineTargetMode::HighResidualAndBlurBadBoard;
  double search_radius_px = 2.0;
  double max_displacement_px = 1.5;
  double geometry_sigma_px = 1.0;
  double observation_sigma_px = 1.0;
  int subpix_window_radius = 1;
  double min_objective_improvement = 5e-4;
  double min_old_residual_px = 1.0;
  double low_patch_gradient_quantile = 0.05;
  double min_board_internal_rmse_px = 5.0;
  double min_board_p95_residual_px = 5.0;
  double min_corner_response_gain = 0.02;
  double min_board_internal_rmse_improvement_px = 0.1;
  int min_refined_point_count_per_board = 4;
  double accept_max_global_outer_delta_px = 0.01;
  double accept_max_frame_outer_delta_px = 0.05;
  int acceptance_backend_max_iterations = 4;
};

struct InternalJointRefinePointDecision {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind =
      JointObservationSourceKind::InternalMeasurement;
  Eigen::Vector2d observed_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d predicted_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d refined_image_xy = Eigen::Vector2d::Zero();
  double old_residual_norm = 0.0;
  double new_residual_norm = 0.0;
  double old_corner_response = 0.0;
  double new_corner_response = 0.0;
  double old_objective = 0.0;
  double new_objective = 0.0;
  double displacement_px = 0.0;
  double board_internal_rmse = 0.0;
  double board_p95_residual = 0.0;
  double board_corner_patch_mean_gradient = 0.0;
  double board_patch_gradient_threshold = 0.0;
  bool board_low_patch_gradient = false;
  bool board_high_internal_residual = false;
  bool targeted_by_high_residual = false;
  bool targeted_by_blur_bad_board = false;
  bool eligible_for_refine = true;
  bool tentative_refined = false;
  bool accepted_after_board_rollback = false;
  bool refined = false;
  double corner_response_gain = 0.0;
  std::string refine_reason;
};

struct InternalJointRefineBoardSummary {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int input_internal_point_count = 0;
  int eligible_internal_point_count = 0;
  int tentative_refined_point_count = 0;
  int accepted_refined_point_count = 0;
  double corner_patch_mean_gradient = 0.0;
  double patch_gradient_threshold = 0.0;
  double board_internal_rmse_before = 0.0;
  double board_internal_rmse_after_tentative = 0.0;
  double board_internal_rmse_improvement = 0.0;
  double board_p95_residual_before = 0.0;
  double global_outer_only_rmse_before = 0.0;
  double global_outer_only_rmse_after = 0.0;
  double global_outer_only_rmse_delta = 0.0;
  double frame_outer_only_rmse_before = 0.0;
  double frame_outer_only_rmse_after = 0.0;
  double frame_outer_only_rmse_delta = 0.0;
  double mean_corner_response_gain = 0.0;
  bool low_patch_gradient = false;
  bool high_internal_residual = false;
  bool targeted_for_refine = false;
  bool accepted = false;
  bool rolled_back = false;
  std::string rollback_reason;
};

struct InternalJointRefineFrameSummary {
  int frame_index = -1;
  std::string frame_label;
  int input_internal_point_count = 0;
  int eligible_internal_point_count = 0;
  int refined_internal_point_count = 0;
  int accepted_board_count = 0;
  int rolled_back_board_count = 0;
  double eligible_ratio = 0.0;
  double refined_ratio = 0.0;
  double mean_displacement_px = 0.0;
  double max_displacement_px = 0.0;
  double mean_residual_before = 0.0;
  double mean_residual_after = 0.0;
};

struct InternalJointRefineResult {
  bool success = false;
  InternalJointRefineOptions options;
  bool diagnostic_only = true;
  bool backend_input_changed = false;
  int input_internal_point_count = 0;
  int eligible_internal_point_count = 0;
  int refined_internal_point_count = 0;
  int candidate_board_count = 0;
  int accepted_board_count = 0;
  int rolled_back_board_count = 0;
  double eligible_ratio = 0.0;
  double refined_ratio = 0.0;
  double mean_displacement_px = 0.0;
  double max_displacement_px = 0.0;
  int targeted_blur_bad_board_count = 0;
  double patch_gradient_threshold = 0.0;
  CalibrationStateBundle curated_bundle;
  std::vector<InternalJointRefinePointDecision> point_decisions;
  std::vector<InternalJointRefineBoardSummary> board_summaries;
  std::vector<InternalJointRefineFrameSummary> frame_summaries;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

InternalJointRefineResult ApplyInternalJointRefinement(
    const CalibrationStateBundle& bundle,
    const std::map<int, std::string>& frame_image_paths,
    const InternalJointRefineOptions& options);

void WriteInternalJointRefineSummary(
    const std::string& path,
    const InternalJointRefineResult& result);
void WriteInternalJointRefinePointsCsv(
    const std::string& path,
    const InternalJointRefineResult& result);
void WriteInternalJointRefineBoardSummaryCsv(
    const std::string& path,
    const InternalJointRefineResult& result);
void WriteInternalJointRefineFrameSummaryCsv(
    const std::string& path,
    const InternalJointRefineResult& result);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_MEASUREMENT_CURATION_HPP
