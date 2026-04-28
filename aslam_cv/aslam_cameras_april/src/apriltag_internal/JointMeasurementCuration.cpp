#include <aslam/cameras/apriltag_internal/JointMeasurementCuration.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <aslam/cameras/apriltag_internal/JointReprojectionResidualEvaluator.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

struct ObservationKey {
  int frame_index = -1;
  int board_id = -1;
  int point_id = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind = JointObservationSourceKind::InternalMeasurement;

  bool operator<(const ObservationKey& other) const {
    return std::tie(frame_index, board_id, point_id, source_point_index, source_kind) <
           std::tie(other.frame_index, other.board_id, other.point_id,
                    other.source_point_index, other.source_kind);
  }
};

struct DatasetStats {
  std::set<int> accepted_frame_indices;
  std::set<std::pair<int, int> > accepted_board_observation_keys;
  int accepted_frame_count = 0;
  int accepted_board_observation_count = 0;
  int accepted_outer_point_count = 0;
  int accepted_internal_point_count = 0;
  int accepted_total_point_count = 0;
};

std::string CsvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) {
    return value;
  }
  std::ostringstream output;
  output << '"';
  for (char ch : value) {
    if (ch == '"') {
      output << "\"\"";
    } else {
      output << ch;
    }
  }
  output << '"';
  return output.str();
}

ObservationKey MakeObservationKey(const JointPointObservation& point) {
  ObservationKey key;
  key.frame_index = point.frame_index;
  key.board_id = point.board_id;
  key.point_id = point.point_id;
  key.source_point_index = point.source_point_index;
  key.source_kind = point.source_kind;
  return key;
}

ObservationKey MakeObservationKey(const JointResidualPointDiagnostics& point) {
  ObservationKey key;
  key.frame_index = point.frame_index;
  key.board_id = point.board_id;
  key.point_id = point.point_id;
  key.source_point_index = point.source_point_index;
  key.source_kind = point.source_kind;
  return key;
}

int CountInputInternalPoints(const CalibrationMeasurementDataset& dataset) {
  int count = 0;
  for (const JointPointObservation& point : dataset.solver_observations) {
    if (point.used_in_solver && point.point_type == JointPointType::Internal) {
      ++count;
    }
  }
  return count;
}

JointMeasurementBuildResult BuildMeasurementResultFromDataset(
    const CalibrationMeasurementDataset& dataset,
    int reference_board_id) {
  JointMeasurementBuildResult result;
  result.success = true;
  result.reference_board_id = reference_board_id;
  result.frames = dataset.frames;
  result.solver_observations = dataset.solver_observations;
  result.warnings = dataset.warnings;
  result.used_frame_count = dataset.accepted_frame_count;
  result.used_board_observation_count = dataset.accepted_board_observation_count;
  result.used_outer_point_count = dataset.accepted_outer_point_count;
  result.used_internal_point_count = dataset.accepted_internal_point_count;
  result.used_total_point_count = dataset.accepted_total_point_count;
  return result;
}

DatasetStats RecomputeDatasetStats(CalibrationMeasurementDataset* dataset) {
  if (dataset == nullptr) {
    throw std::runtime_error("RecomputeDatasetStats requires a valid dataset.");
  }

  DatasetStats stats;
  dataset->solver_observations.clear();

  for (JointMeasurementFrameResult& frame : dataset->frames) {
    bool frame_used = false;
    for (JointBoardObservation& board : frame.board_observations) {
      board.outer_point_count = 0;
      board.internal_point_count = 0;
      board.used_in_solver = false;
      for (JointPointObservation& point : board.points) {
        if (!point.used_in_solver) {
          continue;
        }
        if (point.point_type == JointPointType::Outer) {
          ++board.outer_point_count;
          ++stats.accepted_outer_point_count;
        } else {
          ++board.internal_point_count;
          ++stats.accepted_internal_point_count;
        }
        ++stats.accepted_total_point_count;
        dataset->solver_observations.push_back(point);
      }
      board.used_in_solver =
          board.outer_point_count > 0 || board.internal_point_count > 0;
      if (board.used_in_solver) {
        frame_used = true;
        stats.accepted_board_observation_keys.insert(
            std::make_pair(frame.frame_index, board.board_id));
      }
    }
    if (frame_used) {
      stats.accepted_frame_indices.insert(frame.frame_index);
    }
  }

  stats.accepted_frame_count =
      static_cast<int>(stats.accepted_frame_indices.size());
  stats.accepted_board_observation_count =
      static_cast<int>(stats.accepted_board_observation_keys.size());

  dataset->accepted_frame_indices = stats.accepted_frame_indices;
  dataset->accepted_board_observation_keys = stats.accepted_board_observation_keys;
  dataset->accepted_frame_count = stats.accepted_frame_count;
  dataset->accepted_board_observation_count = stats.accepted_board_observation_count;
  dataset->accepted_outer_point_count = stats.accepted_outer_point_count;
  dataset->accepted_internal_point_count = stats.accepted_internal_point_count;
  dataset->accepted_total_point_count = stats.accepted_total_point_count;
  return stats;
}

void ApplyFilteredKeysToDataset(const std::set<ObservationKey>& filtered_keys,
                                CalibrationMeasurementDataset* dataset) {
  if (dataset == nullptr || filtered_keys.empty()) {
    return;
  }
  for (JointMeasurementFrameResult& frame : dataset->frames) {
    for (JointBoardObservation& board : frame.board_observations) {
      for (JointPointObservation& point : board.points) {
        if (!point.used_in_solver ||
            point.point_type != JointPointType::Internal) {
          continue;
        }
        if (filtered_keys.find(MakeObservationKey(point)) == filtered_keys.end()) {
          continue;
        }
        point.used_in_solver = false;
        point.rejection_reason_code =
            JointRejectionReasonCode::InternalPointReprojectionOutlier;
        point.rejection_detail =
            "pre-backend kalibr-style internal reprojection filter";
      }
    }
  }
  RecomputeDatasetStats(dataset);
}

void ApplyFilteredInternalBoardsToDataset(
    const std::set<std::pair<int, int> >& filtered_boards,
    const std::string& rejection_detail,
    CalibrationMeasurementDataset* dataset) {
  if (dataset == nullptr || filtered_boards.empty()) {
    return;
  }
  for (JointMeasurementFrameResult& frame : dataset->frames) {
    for (JointBoardObservation& board : frame.board_observations) {
      if (filtered_boards.find(std::make_pair(frame.frame_index,
                                              board.board_id)) ==
          filtered_boards.end()) {
        continue;
      }
      for (JointPointObservation& point : board.points) {
        if (!point.used_in_solver ||
            point.point_type != JointPointType::Internal) {
          continue;
        }
        point.used_in_solver = false;
        point.rejection_reason_code =
            JointRejectionReasonCode::InternalPointReprojectionOutlier;
        point.rejection_detail = rejection_detail;
      }
    }
  }
  RecomputeDatasetStats(dataset);
}

double ComputeMean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double ComputeStd(const std::vector<double>& values, double mean) {
  if (values.empty()) {
    return 0.0;
  }
  double variance = 0.0;
  for (double value : values) {
    const double delta = value - mean;
    variance += delta * delta;
  }
  variance /= static_cast<double>(values.size());
  return std::sqrt(std::max(0.0, variance));
}

double SafeRatio(int numerator, int denominator) {
  if (denominator <= 0) {
    return 0.0;
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

double Quantile(std::vector<double> values, double q) {
  if (values.empty()) {
    return 0.0;
  }
  if (q <= 0.0) {
    return *std::min_element(values.begin(), values.end());
  }
  if (q >= 1.0) {
    return *std::max_element(values.begin(), values.end());
  }
  std::sort(values.begin(), values.end());
  const double scaled = q * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(scaled));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(scaled));
  if (lower == upper) {
    return values[lower];
  }
  const double t = scaled - static_cast<double>(lower);
  return values[lower] * (1.0 - t) + values[upper] * t;
}

double ComputeCornerPatchMeanGradient(
    const cv::Mat& image,
    const std::vector<InternalBlurFilterPointDecision*>& points) {
  if (image.empty() || points.empty()) {
    return 0.0;
  }
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Mat grad_x;
  cv::Mat grad_y;
  cv::Sobel(gray, grad_x, CV_64F, 1, 0, 3);
  cv::Sobel(gray, grad_y, CV_64F, 0, 1, 3);
  cv::Mat magnitude;
  cv::magnitude(grad_x, grad_y, magnitude);

  double sum = 0.0;
  int count = 0;
  for (const InternalBlurFilterPointDecision* point : points) {
    if (point == nullptr) {
      continue;
    }
    const int x =
        static_cast<int>(std::round(point->observed_image_xy.x()));
    const int y =
        static_cast<int>(std::round(point->observed_image_xy.y()));
    cv::Rect rect(x - 4, y - 4, 9, 9);
    rect &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (rect.width <= 0 || rect.height <= 0) {
      continue;
    }
    sum += cv::mean(magnitude(rect))[0];
    ++count;
  }
  return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

}  // namespace

const char* ToString(PreBackendFilterMode mode) {
  switch (mode) {
    case PreBackendFilterMode::Off:
      return "off";
    case PreBackendFilterMode::Diagnostic:
      return "diagnostic";
    case PreBackendFilterMode::Enabled:
      return "enabled";
  }
  return "unknown";
}

PreBackendFilterMode ParsePreBackendFilterMode(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "off") {
    return PreBackendFilterMode::Off;
  }
  if (lowered == "diagnostic") {
    return PreBackendFilterMode::Diagnostic;
  }
  if (lowered == "enabled") {
    return PreBackendFilterMode::Enabled;
  }
  throw std::runtime_error("Unsupported pre-backend filter mode: " + value);
}

const char* ToString(InternalBlurFilterMode mode) {
  switch (mode) {
    case InternalBlurFilterMode::Off:
      return "off";
    case InternalBlurFilterMode::Diagnostic:
      return "diagnostic";
    case InternalBlurFilterMode::Enabled:
      return "enabled";
  }
  return "unknown";
}

InternalBlurFilterMode ParseInternalBlurFilterMode(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "off") {
    return InternalBlurFilterMode::Off;
  }
  if (lowered == "diagnostic") {
    return InternalBlurFilterMode::Diagnostic;
  }
  if (lowered == "enabled") {
    return InternalBlurFilterMode::Enabled;
  }
  throw std::runtime_error("Unsupported internal blur filter mode: " + value);
}

PreBackendObservationFilterResult ApplyPreBackendObservationFilter(
    const CalibrationStateBundle& bundle,
    const PreBackendObservationFilterOptions& options) {
  PreBackendObservationFilterResult result;
  result.options = options;
  result.diagnostic_only = options.mode != PreBackendFilterMode::Enabled;
  result.backend_input_changed = options.mode == PreBackendFilterMode::Enabled;
  result.curated_bundle = bundle;
  result.input_internal_point_count =
      CountInputInternalPoints(bundle.measurement_dataset);
  result.remaining_internal_point_count = result.input_internal_point_count;

  if (!bundle.IsReadyForBackend()) {
    result.failure_reason =
        "ApplyPreBackendObservationFilter requires a ready-for-backend bundle.";
    return result;
  }
  if (options.sigma_threshold < 0.0) {
    result.failure_reason = "sigma_threshold must be non-negative.";
    return result;
  }
  if (options.min_abs_threshold_px < 0.0) {
    result.failure_reason = "min_abs_threshold_px must be non-negative.";
    return result;
  }

  if (options.mode == PreBackendFilterMode::Off) {
    result.success = true;
    return result;
  }

  const JointReprojectionSceneState scene_state =
      BuildJointSceneStateFromCalibrationSceneState(bundle.scene_state);
  const JointMeasurementBuildResult measurement_result =
      BuildMeasurementResultFromDataset(bundle.measurement_dataset,
                                        bundle.scene_state.reference_board_id);
  JointResidualEvaluationOptions residual_options;
  residual_options.top_k = -1;
  const JointReprojectionResidualEvaluator residual_evaluator(residual_options);
  const JointResidualEvaluationResult residual_result =
      residual_evaluator.Evaluate(measurement_result, scene_state);
  if (!residual_result.success) {
    result.failure_reason = residual_result.failure_reason;
    result.warnings = residual_result.warnings;
    return result;
  }

  std::map<std::pair<int, int>, std::vector<std::size_t> > board_to_decisions;
  std::map<int, std::string> frame_labels;
  for (const JointResidualPointDiagnostics& point :
       residual_result.point_diagnostics) {
    if (point.point_type != JointPointType::Internal || !point.used_in_solver) {
      continue;
    }
    PreBackendFilterPointDecision decision;
    decision.frame_index = point.frame_index;
    decision.frame_label = point.frame_label;
    decision.board_id = point.board_id;
    decision.point_id = point.point_id;
    decision.source_point_index = point.source_point_index;
    decision.source_kind = point.source_kind;
    decision.observed_image_xy = point.observed_image_xy;
    decision.predicted_image_xy = point.predicted_image_xy;
    decision.residual_xy = point.residual_xy;
    decision.residual_norm = point.residual_norm;
    decision.valid_projection =
        std::isfinite(point.predicted_image_xy.x()) &&
        std::isfinite(point.predicted_image_xy.y()) &&
        std::isfinite(point.residual_norm);
    frame_labels[point.frame_index] = point.frame_label;
    const std::size_t index = result.point_decisions.size();
    result.point_decisions.push_back(decision);
    board_to_decisions[std::make_pair(point.frame_index, point.board_id)]
        .push_back(index);
  }

  std::set<ObservationKey> filtered_keys;
  std::set<std::pair<int, int> > affected_boards;
  std::set<int> affected_frames;

  for (const auto& entry : board_to_decisions) {
    const std::pair<int, int>& board_key = entry.first;
    const std::vector<std::size_t>& indices = entry.second;
    std::vector<double> residuals;
    residuals.reserve(indices.size());
    for (std::size_t index : indices) {
      residuals.push_back(result.point_decisions[index].residual_norm);
    }
    const double mean = ComputeMean(residuals);
    const double stddev = ComputeStd(residuals, mean);
    const double threshold =
        std::max(options.min_abs_threshold_px,
                 mean + options.sigma_threshold * stddev);

    PreBackendFilterBoardSummary board_summary;
    board_summary.frame_index = board_key.first;
    board_summary.frame_label = frame_labels[board_key.first];
    board_summary.board_id = board_key.second;
    board_summary.input_internal_point_count =
        static_cast<int>(indices.size());
    board_summary.mean_residual = mean;
    board_summary.std_residual = stddev;
    board_summary.threshold = threshold;

    for (std::size_t index : indices) {
      PreBackendFilterPointDecision& decision = result.point_decisions[index];
      decision.board_mean_residual = mean;
      decision.board_std_residual = stddev;
      decision.board_threshold = threshold;
      decision.filtered = decision.residual_norm > threshold;
      if (decision.filtered) {
        decision.filter_reason = "residual_above_kalibr_style_threshold";
        ++board_summary.filtered_internal_point_count;
        affected_boards.insert(board_key);
        affected_frames.insert(board_key.first);
        ObservationKey key;
        key.frame_index = decision.frame_index;
        key.board_id = decision.board_id;
        key.point_id = decision.point_id;
        key.source_point_index = decision.source_point_index;
        key.source_kind = decision.source_kind;
        filtered_keys.insert(key);
      }
    }
    board_summary.remaining_internal_point_count =
        board_summary.input_internal_point_count -
        board_summary.filtered_internal_point_count;
    board_summary.filtered_ratio =
        SafeRatio(board_summary.filtered_internal_point_count,
                  board_summary.input_internal_point_count);
    result.board_summaries.push_back(board_summary);
  }

  std::map<int, PreBackendFilterFrameSummary> frame_map;
  for (const PreBackendFilterBoardSummary& board : result.board_summaries) {
    PreBackendFilterFrameSummary& frame = frame_map[board.frame_index];
    frame.frame_index = board.frame_index;
    frame.frame_label = board.frame_label;
    ++frame.board_observation_count;
    frame.input_internal_point_count += board.input_internal_point_count;
    frame.filtered_internal_point_count += board.filtered_internal_point_count;
    frame.remaining_internal_point_count += board.remaining_internal_point_count;
    if (board.filtered_internal_point_count > 0) {
      ++frame.affected_board_count;
    }
  }
  for (auto& entry : frame_map) {
    PreBackendFilterFrameSummary frame = entry.second;
    frame.filtered_ratio =
        SafeRatio(frame.filtered_internal_point_count,
                  frame.input_internal_point_count);
    result.frame_summaries.push_back(frame);
  }

  result.filtered_internal_point_count =
      static_cast<int>(filtered_keys.size());
  result.remaining_internal_point_count =
      result.input_internal_point_count - result.filtered_internal_point_count;
  result.filtered_ratio =
      SafeRatio(result.filtered_internal_point_count,
                result.input_internal_point_count);
  result.affected_board_count = static_cast<int>(affected_boards.size());
  result.affected_frame_count = static_cast<int>(affected_frames.size());

  if (options.mode == PreBackendFilterMode::Enabled) {
    ApplyFilteredKeysToDataset(filtered_keys,
                               &result.curated_bundle.measurement_dataset);
    const JointMeasurementBuildResult curated_measurement_result =
        BuildMeasurementResultFromDataset(
            result.curated_bundle.measurement_dataset,
            result.curated_bundle.scene_state.reference_board_id);
    const JointResidualEvaluationResult curated_residual_result =
        residual_evaluator.Evaluate(curated_measurement_result, scene_state);
    if (!curated_residual_result.success) {
      result.failure_reason = curated_residual_result.failure_reason;
      result.warnings = curated_residual_result.warnings;
      return result;
    }
    result.curated_bundle.residual_result = curated_residual_result;
    std::ostringstream warning;
    warning << "pre-backend Kalibr-style internal filter removed "
            << result.filtered_internal_point_count << " / "
            << result.input_internal_point_count << " internal points";
    result.curated_bundle.warnings.push_back(warning.str());
  }

  result.success = true;
  return result;
}

InternalBlurObservationFilterResult ApplyInternalBlurObservationFilter(
    const CalibrationStateBundle& bundle,
    const std::map<int, std::string>& frame_image_paths,
    const InternalBlurObservationFilterOptions& options) {
  InternalBlurObservationFilterResult result;
  result.options = options;
  result.diagnostic_only = options.mode != InternalBlurFilterMode::Enabled;
  result.backend_input_changed = options.mode == InternalBlurFilterMode::Enabled;
  result.curated_bundle = bundle;
  result.input_internal_point_count =
      CountInputInternalPoints(bundle.measurement_dataset);
  result.remaining_internal_point_count = result.input_internal_point_count;

  if (!bundle.IsReadyForBackend()) {
    result.failure_reason =
        "ApplyInternalBlurObservationFilter requires a ready-for-backend bundle.";
    return result;
  }
  if (options.low_patch_gradient_quantile < 0.0 ||
      options.low_patch_gradient_quantile > 1.0) {
    result.failure_reason =
        "low_patch_gradient_quantile must be in [0, 1].";
    return result;
  }
  if (options.min_board_internal_rmse_px < 0.0 ||
      options.min_board_p95_residual_px < 0.0) {
    result.failure_reason =
        "internal blur filter residual thresholds must be non-negative.";
    return result;
  }
  if (options.mode == InternalBlurFilterMode::Off) {
    result.success = true;
    return result;
  }

  const JointReprojectionSceneState scene_state =
      BuildJointSceneStateFromCalibrationSceneState(bundle.scene_state);
  const JointMeasurementBuildResult measurement_result =
      BuildMeasurementResultFromDataset(bundle.measurement_dataset,
                                        bundle.scene_state.reference_board_id);
  JointResidualEvaluationOptions residual_options;
  residual_options.top_k = -1;
  const JointReprojectionResidualEvaluator residual_evaluator(residual_options);
  const JointResidualEvaluationResult residual_result =
      residual_evaluator.Evaluate(measurement_result, scene_state);
  if (!residual_result.success) {
    result.failure_reason = residual_result.failure_reason;
    result.warnings = residual_result.warnings;
    return result;
  }

  std::map<std::pair<int, int>, std::vector<std::size_t> > board_to_points;
  std::map<int, std::string> frame_labels;
  for (const JointResidualPointDiagnostics& point :
       residual_result.point_diagnostics) {
    if (point.point_type != JointPointType::Internal || !point.used_in_solver) {
      continue;
    }
    InternalBlurFilterPointDecision decision;
    decision.frame_index = point.frame_index;
    decision.frame_label = point.frame_label;
    decision.board_id = point.board_id;
    decision.point_id = point.point_id;
    decision.source_point_index = point.source_point_index;
    decision.source_kind = point.source_kind;
    decision.observed_image_xy = point.observed_image_xy;
    decision.residual_norm = point.residual_norm;
    frame_labels[point.frame_index] = point.frame_label;
    const std::size_t index = result.point_decisions.size();
    result.point_decisions.push_back(decision);
    board_to_points[std::make_pair(point.frame_index, point.board_id)]
        .push_back(index);
  }

  std::map<int, cv::Mat> image_cache;
  std::vector<double> finite_patch_gradients;
  result.board_decisions.reserve(board_to_points.size());
  for (const auto& entry : board_to_points) {
    const std::pair<int, int>& board_key = entry.first;
    const std::vector<std::size_t>& indices = entry.second;
    InternalBlurFilterBoardDecision board;
    board.frame_index = board_key.first;
    board.frame_label = frame_labels[board_key.first];
    board.board_id = board_key.second;
    board.input_internal_point_count = static_cast<int>(indices.size());
    board.remaining_internal_point_count = board.input_internal_point_count;

    std::vector<double> residuals;
    residuals.reserve(indices.size());
    double squared_sum = 0.0;
    for (std::size_t index : indices) {
      const double residual = result.point_decisions[index].residual_norm;
      residuals.push_back(residual);
      squared_sum += residual * residual;
    }
    if (!residuals.empty()) {
      board.internal_rmse =
          std::sqrt(squared_sum / static_cast<double>(residuals.size()));
      board.max_residual =
          *std::max_element(residuals.begin(), residuals.end());
      board.p90_residual = Quantile(residuals, 0.90);
      board.p95_residual = Quantile(residuals, 0.95);
    }

    const auto image_path_it = frame_image_paths.find(board.frame_index);
    if (image_path_it == frame_image_paths.end()) {
      board.corner_patch_mean_gradient =
          std::numeric_limits<double>::infinity();
      board.filter_reason = "image_path_not_found";
      result.warnings.push_back(
          "internal blur filter skipped board because image path was not found");
    } else {
      cv::Mat& image = image_cache[board.frame_index];
      if (image.empty()) {
        image = cv::imread(image_path_it->second, cv::IMREAD_UNCHANGED);
      }
      if (image.empty()) {
        board.corner_patch_mean_gradient =
            std::numeric_limits<double>::infinity();
        board.filter_reason = "image_read_failed";
        result.warnings.push_back(
            "internal blur filter skipped board because image read failed");
      } else {
        std::vector<InternalBlurFilterPointDecision*> point_ptrs;
        point_ptrs.reserve(indices.size());
        for (std::size_t index : indices) {
          point_ptrs.push_back(&result.point_decisions[index]);
        }
        board.corner_patch_mean_gradient =
            ComputeCornerPatchMeanGradient(image, point_ptrs);
        if (std::isfinite(board.corner_patch_mean_gradient)) {
          finite_patch_gradients.push_back(board.corner_patch_mean_gradient);
        }
      }
    }
    result.board_decisions.push_back(board);
  }

  result.input_board_observation_count =
      static_cast<int>(result.board_decisions.size());
  result.patch_gradient_threshold =
      Quantile(finite_patch_gradients, options.low_patch_gradient_quantile);

  std::set<std::pair<int, int> > filtered_boards;
  std::set<int> affected_frames;
  for (InternalBlurFilterBoardDecision& board : result.board_decisions) {
    board.patch_gradient_threshold = result.patch_gradient_threshold;
    board.low_patch_gradient =
        std::isfinite(board.corner_patch_mean_gradient) &&
        board.corner_patch_mean_gradient <= result.patch_gradient_threshold;
    board.high_internal_residual =
        board.internal_rmse >= options.min_board_internal_rmse_px ||
        board.p95_residual >= options.min_board_p95_residual_px;
    board.filtered = board.low_patch_gradient && board.high_internal_residual;
    if (board.filtered) {
      board.filter_reason =
          "low_corner_patch_gradient_and_high_internal_residual";
      board.filtered_internal_point_count = board.input_internal_point_count;
      board.remaining_internal_point_count = 0;
      filtered_boards.insert(std::make_pair(board.frame_index, board.board_id));
      affected_frames.insert(board.frame_index);
    } else if (board.filter_reason.empty()) {
      if (!board.low_patch_gradient && !board.high_internal_residual) {
        board.filter_reason = "accepted_gradient_and_residual";
      } else if (!board.low_patch_gradient) {
        board.filter_reason = "accepted_patch_gradient_not_low";
      } else {
        board.filter_reason = "accepted_internal_residual_not_high";
      }
    }
  }

  for (InternalBlurFilterPointDecision& point : result.point_decisions) {
    if (filtered_boards.find(std::make_pair(point.frame_index,
                                            point.board_id)) !=
        filtered_boards.end()) {
      point.filtered = true;
      point.filter_reason =
          "board_removed_by_internal_blur_filter";
    }
  }

  std::map<int, InternalBlurFilterFrameSummary> frame_map;
  for (const InternalBlurFilterBoardDecision& board : result.board_decisions) {
    InternalBlurFilterFrameSummary& frame = frame_map[board.frame_index];
    frame.frame_index = board.frame_index;
    frame.frame_label = board.frame_label;
    ++frame.board_observation_count;
    frame.input_internal_point_count += board.input_internal_point_count;
    frame.filtered_internal_point_count += board.filtered_internal_point_count;
    frame.remaining_internal_point_count += board.remaining_internal_point_count;
    if (board.filtered) {
      ++frame.filtered_board_count;
    }
  }
  for (auto& entry : frame_map) {
    InternalBlurFilterFrameSummary frame = entry.second;
    frame.filtered_ratio =
        SafeRatio(frame.filtered_internal_point_count,
                  frame.input_internal_point_count);
    result.frame_summaries.push_back(frame);
  }

  result.filtered_board_observation_count =
      static_cast<int>(filtered_boards.size());
  result.affected_frame_count = static_cast<int>(affected_frames.size());
  result.filtered_internal_point_count = 0;
  for (const InternalBlurFilterBoardDecision& board : result.board_decisions) {
    result.filtered_internal_point_count += board.filtered_internal_point_count;
  }
  result.remaining_internal_point_count =
      result.input_internal_point_count - result.filtered_internal_point_count;
  result.filtered_ratio =
      SafeRatio(result.filtered_internal_point_count,
                result.input_internal_point_count);

  if (options.mode == InternalBlurFilterMode::Enabled) {
    ApplyFilteredInternalBoardsToDataset(
        filtered_boards,
        "pre-backend internal blur filter removed entire board observation",
        &result.curated_bundle.measurement_dataset);
    const JointMeasurementBuildResult curated_measurement_result =
        BuildMeasurementResultFromDataset(
            result.curated_bundle.measurement_dataset,
            result.curated_bundle.scene_state.reference_board_id);
    const JointResidualEvaluationResult curated_residual_result =
        residual_evaluator.Evaluate(curated_measurement_result, scene_state);
    if (!curated_residual_result.success) {
      result.failure_reason = curated_residual_result.failure_reason;
      result.warnings = curated_residual_result.warnings;
      return result;
    }
    result.curated_bundle.residual_result = curated_residual_result;
    std::ostringstream warning;
    warning << "pre-backend internal blur filter removed "
            << result.filtered_internal_point_count << " / "
            << result.input_internal_point_count
            << " internal points from "
            << result.filtered_board_observation_count
            << " board observations";
    result.curated_bundle.warnings.push_back(warning.str());
  }

  result.success = true;
  return result;
}

void WritePreBackendFilterSummary(
    const std::string& path,
    const PreBackendObservationFilterResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "mode: " << ToString(result.options.mode) << "\n";
  output << "diagnostic_only: " << (result.diagnostic_only ? 1 : 0) << "\n";
  output << "backend_input_changed: " << (result.backend_input_changed ? 1 : 0)
         << "\n";
  output << "sigma_threshold: " << result.options.sigma_threshold << "\n";
  output << "min_abs_threshold_px: " << result.options.min_abs_threshold_px
         << "\n";
  output << "input_internal_point_count: "
         << result.input_internal_point_count << "\n";
  output << "filtered_internal_point_count: "
         << result.filtered_internal_point_count << "\n";
  output << "remaining_internal_point_count: "
         << result.remaining_internal_point_count << "\n";
  output << "filtered_ratio: " << result.filtered_ratio << "\n";
  output << "affected_board_count: " << result.affected_board_count << "\n";
  output << "affected_frame_count: " << result.affected_frame_count << "\n";
  output << "board_summary_count: " << result.board_summaries.size() << "\n";
  output << "point_decision_count: " << result.point_decisions.size() << "\n";
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WritePreBackendFilterPointsCsv(
    const std::string& path,
    const PreBackendObservationFilterResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_id,point_id,source_kind,source_point_index,observed_x,"
      << "observed_y,predicted_x,predicted_y,residual_x,residual_y,"
      << "residual_norm,valid_projection,board_mean_residual,"
      << "board_std_residual,board_threshold,sigma_threshold,"
      << "min_abs_threshold_px,filtered,filter_reason\n";
  for (const PreBackendFilterPointDecision& point : result.point_decisions) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << point.frame_index << ","
           << CsvEscape(point.frame_label) << ","
           << point.board_id << ","
           << point.point_id << ","
           << ToString(point.source_kind) << ","
           << point.source_point_index << ","
           << point.observed_image_xy.x() << ","
           << point.observed_image_xy.y() << ","
           << point.predicted_image_xy.x() << ","
           << point.predicted_image_xy.y() << ","
           << point.residual_xy.x() << ","
           << point.residual_xy.y() << ","
           << point.residual_norm << ","
           << (point.valid_projection ? 1 : 0) << ","
           << point.board_mean_residual << ","
           << point.board_std_residual << ","
           << point.board_threshold << ","
           << result.options.sigma_threshold << ","
           << result.options.min_abs_threshold_px << ","
           << (point.filtered ? 1 : 0) << ","
           << CsvEscape(point.filter_reason) << "\n";
  }
}

void WritePreBackendFilterBoardSummaryCsv(
    const std::string& path,
    const PreBackendObservationFilterResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_id,input_internal_point_count,filtered_internal_point_count,"
      << "remaining_internal_point_count,filtered_ratio,mean_residual,"
      << "std_residual,threshold,sigma_threshold,min_abs_threshold_px\n";
  for (const PreBackendFilterBoardSummary& board : result.board_summaries) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << board.frame_index << ","
           << CsvEscape(board.frame_label) << ","
           << board.board_id << ","
           << board.input_internal_point_count << ","
           << board.filtered_internal_point_count << ","
           << board.remaining_internal_point_count << ","
           << board.filtered_ratio << ","
           << board.mean_residual << ","
           << board.std_residual << ","
           << board.threshold << ","
           << result.options.sigma_threshold << ","
           << result.options.min_abs_threshold_px << "\n";
  }
}

void WritePreBackendFilterFrameSummaryCsv(
    const std::string& path,
    const PreBackendObservationFilterResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_observation_count,affected_board_count,"
      << "input_internal_point_count,filtered_internal_point_count,"
      << "remaining_internal_point_count,filtered_ratio\n";
  for (const PreBackendFilterFrameSummary& frame : result.frame_summaries) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << frame.frame_index << ","
           << CsvEscape(frame.frame_label) << ","
           << frame.board_observation_count << ","
           << frame.affected_board_count << ","
           << frame.input_internal_point_count << ","
           << frame.filtered_internal_point_count << ","
           << frame.remaining_internal_point_count << ","
           << frame.filtered_ratio << "\n";
  }
}

void WriteInternalBlurFilterSummary(
    const std::string& path,
    const InternalBlurObservationFilterResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "mode: " << ToString(result.options.mode) << "\n";
  output << "diagnostic_only: " << (result.diagnostic_only ? 1 : 0)
         << "\n";
  output << "backend_input_changed: "
         << (result.backend_input_changed ? 1 : 0) << "\n";
  output << "low_patch_gradient_quantile: "
         << result.options.low_patch_gradient_quantile << "\n";
  output << "patch_gradient_threshold: "
         << result.patch_gradient_threshold << "\n";
  output << "min_board_internal_rmse_px: "
         << result.options.min_board_internal_rmse_px << "\n";
  output << "min_board_p95_residual_px: "
         << result.options.min_board_p95_residual_px << "\n";
  output << "input_board_observation_count: "
         << result.input_board_observation_count << "\n";
  output << "filtered_board_observation_count: "
         << result.filtered_board_observation_count << "\n";
  output << "affected_frame_count: " << result.affected_frame_count << "\n";
  output << "input_internal_point_count: "
         << result.input_internal_point_count << "\n";
  output << "filtered_internal_point_count: "
         << result.filtered_internal_point_count << "\n";
  output << "remaining_internal_point_count: "
         << result.remaining_internal_point_count << "\n";
  output << "filtered_ratio: " << result.filtered_ratio << "\n";
  output << "board_decision_count: " << result.board_decisions.size()
         << "\n";
  output << "point_decision_count: " << result.point_decisions.size()
         << "\n";
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteInternalBlurFilterBoardDecisionsCsv(
    const std::string& path,
    const InternalBlurObservationFilterResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_id,input_internal_point_count,filtered_internal_point_count,"
      << "remaining_internal_point_count,internal_rmse,max_residual,"
      << "p90_residual,p95_residual,corner_patch_mean_gradient,"
      << "patch_gradient_threshold,low_patch_gradient,high_internal_residual,"
      << "filtered,filter_reason\n";
  for (const InternalBlurFilterBoardDecision& board : result.board_decisions) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << board.frame_index << ","
           << CsvEscape(board.frame_label) << ","
           << board.board_id << ","
           << board.input_internal_point_count << ","
           << board.filtered_internal_point_count << ","
           << board.remaining_internal_point_count << ","
           << board.internal_rmse << ","
           << board.max_residual << ","
           << board.p90_residual << ","
           << board.p95_residual << ","
           << board.corner_patch_mean_gradient << ","
           << board.patch_gradient_threshold << ","
           << (board.low_patch_gradient ? 1 : 0) << ","
           << (board.high_internal_residual ? 1 : 0) << ","
           << (board.filtered ? 1 : 0) << ","
           << CsvEscape(board.filter_reason) << "\n";
  }
}

void WriteInternalBlurFilterPointDecisionsCsv(
    const std::string& path,
    const InternalBlurObservationFilterResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_id,point_id,source_kind,source_point_index,observed_x,"
      << "observed_y,residual_norm,filtered,filter_reason\n";
  for (const InternalBlurFilterPointDecision& point : result.point_decisions) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << point.frame_index << ","
           << CsvEscape(point.frame_label) << ","
           << point.board_id << ","
           << point.point_id << ","
           << ToString(point.source_kind) << ","
           << point.source_point_index << ","
           << point.observed_image_xy.x() << ","
           << point.observed_image_xy.y() << ","
           << point.residual_norm << ","
           << (point.filtered ? 1 : 0) << ","
           << CsvEscape(point.filter_reason) << "\n";
  }
}

void WriteInternalBlurFilterFrameSummaryCsv(
    const std::string& path,
    const InternalBlurObservationFilterResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_observation_count,filtered_board_count,"
      << "input_internal_point_count,filtered_internal_point_count,"
      << "remaining_internal_point_count,filtered_ratio\n";
  for (const InternalBlurFilterFrameSummary& frame : result.frame_summaries) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << frame.frame_index << ","
           << CsvEscape(frame.frame_label) << ","
           << frame.board_observation_count << ","
           << frame.filtered_board_count << ","
           << frame.input_internal_point_count << ","
           << frame.filtered_internal_point_count << ","
           << frame.remaining_internal_point_count << ","
           << frame.filtered_ratio << "\n";
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
