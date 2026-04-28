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

enum class InternalBlurFilterMode {
  Off,
  Diagnostic,
  Enabled,
};

const char* ToString(InternalBlurFilterMode mode);
InternalBlurFilterMode ParseInternalBlurFilterMode(const std::string& value);

struct PreBackendObservationFilterOptions {
  PreBackendFilterMode mode = PreBackendFilterMode::Off;
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

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_JOINT_MEASUREMENT_CURATION_HPP
