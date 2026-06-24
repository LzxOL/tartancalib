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

#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>
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

void ApplyObservationWeightsToDataset(
    const std::map<ObservationKey, double>& weights,
    CalibrationMeasurementDataset* dataset) {
  if (dataset == nullptr || weights.empty()) {
    return;
  }
  for (JointMeasurementFrameResult& frame : dataset->frames) {
    for (JointBoardObservation& board : frame.board_observations) {
      for (JointPointObservation& point : board.points) {
        if (!point.used_in_solver ||
            point.point_type != JointPointType::Internal) {
          continue;
        }
        const auto weight_it = weights.find(MakeObservationKey(point));
        if (weight_it == weights.end()) {
          continue;
        }
        point.observation_weight = weight_it->second;
      }
    }
  }
  RecomputeDatasetStats(dataset);
}

void ApplyRefinedPositionsToDataset(
    const std::map<ObservationKey, Eigen::Vector2d>& refined_positions,
    CalibrationMeasurementDataset* dataset) {
  if (dataset == nullptr || refined_positions.empty()) {
    return;
  }
  for (JointMeasurementFrameResult& frame : dataset->frames) {
    for (JointBoardObservation& board : frame.board_observations) {
      for (JointPointObservation& point : board.points) {
        if (!point.used_in_solver ||
            point.point_type != JointPointType::Internal) {
          continue;
        }
        const auto it = refined_positions.find(MakeObservationKey(point));
        if (it == refined_positions.end()) {
          continue;
        }
        point.image_xy = it->second;
      }
    }
  }
  RecomputeDatasetStats(dataset);
}

double ClampUnit(double value) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  return std::max(0.0, std::min(1.0, value));
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

double ComputeMedian(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t size = values.size();
  if (size % 2 == 1) {
    return values[size / 2];
  }
  return 0.5 * (values[size / 2 - 1] + values[size / 2]);
}

double ComputeMedianAbsoluteDeviation(const std::vector<double>& values,
                                      double median) {
  if (values.empty()) {
    return 0.0;
  }
  std::vector<double> deviations;
  deviations.reserve(values.size());
  for (double value : values) {
    deviations.push_back(std::abs(value - median));
  }
  return ComputeMedian(deviations);
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

double ComputeBoardInternalRmseFromResidual(
    const JointResidualEvaluationResult& residual_result,
    int frame_index,
    int board_id) {
  double squared_error_sum = 0.0;
  int count = 0;
  for (const JointResidualPointDiagnostics& point :
       residual_result.point_diagnostics) {
    if (!point.used_in_solver || point.point_type != JointPointType::Internal ||
        point.frame_index != frame_index || point.board_id != board_id) {
      continue;
    }
    squared_error_sum += point.residual_norm * point.residual_norm;
    ++count;
  }
  if (count <= 0) {
    return 0.0;
  }
  return std::sqrt(squared_error_sum / static_cast<double>(count));
}

struct BoardResidualConsistencyStats {
  double internal_squared_error_sum = 0.0;
  double outer_squared_error_sum = 0.0;
  int internal_count = 0;
  int outer_count = 0;
  double internal_rmse = 0.0;
  double outer_rmse = 0.0;
  double ratio = 1.0;
};

std::map<std::pair<int, int>, BoardResidualConsistencyStats>
ComputeBoardResidualConsistencyStats(
    const JointResidualEvaluationResult& residual_result,
    double min_outer_rmse) {
  std::map<std::pair<int, int>, BoardResidualConsistencyStats> stats_by_board;
  for (const JointResidualPointDiagnostics& point :
       residual_result.point_diagnostics) {
    if (!point.used_in_solver) {
      continue;
    }
    BoardResidualConsistencyStats& stats =
        stats_by_board[std::make_pair(point.frame_index, point.board_id)];
    const double squared_error = point.residual_norm * point.residual_norm;
    if (point.point_type == JointPointType::Internal) {
      stats.internal_squared_error_sum += squared_error;
      ++stats.internal_count;
    } else {
      stats.outer_squared_error_sum += squared_error;
      ++stats.outer_count;
    }
  }

  for (auto& entry : stats_by_board) {
    BoardResidualConsistencyStats& stats = entry.second;
    stats.internal_rmse =
        stats.internal_count > 0
            ? std::sqrt(stats.internal_squared_error_sum /
                        static_cast<double>(stats.internal_count))
            : 0.0;
    stats.outer_rmse =
        stats.outer_count > 0
            ? std::sqrt(stats.outer_squared_error_sum /
                        static_cast<double>(stats.outer_count))
            : 0.0;
    const double denominator = std::max(min_outer_rmse, stats.outer_rmse);
    stats.ratio = denominator > 0.0 ? stats.internal_rmse / denominator : 1.0;
  }
  return stats_by_board;
}

std::map<int, double> ComputeFrameOuterOnlyRmseMap(
    const JointResidualEvaluationResult& residual_result) {
  struct Accumulator {
    double squared_error_sum = 0.0;
    int count = 0;
  };
  std::map<int, Accumulator> accumulators;
  for (const JointResidualPointDiagnostics& point :
       residual_result.point_diagnostics) {
    if (!point.used_in_solver || point.point_type != JointPointType::Outer) {
      continue;
    }
    Accumulator& accumulator = accumulators[point.frame_index];
    accumulator.squared_error_sum += point.residual_norm * point.residual_norm;
    ++accumulator.count;
  }

  std::map<int, double> frame_outer_rmse;
  for (const auto& entry : accumulators) {
    if (entry.second.count <= 0) {
      continue;
    }
    frame_outer_rmse[entry.first] =
        std::sqrt(entry.second.squared_error_sum /
                  static_cast<double>(entry.second.count));
  }
  return frame_outer_rmse;
}

double LookupFrameOuterOnlyRmse(const std::map<int, double>& frame_outer_rmse,
                                int frame_index) {
  const auto it = frame_outer_rmse.find(frame_index);
  if (it == frame_outer_rmse.end()) {
    return 0.0;
  }
  return it->second;
}

cv::Mat ComputeCornerResponseMap(const cv::Mat& image) {
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Mat gray32f;
  gray.convertTo(gray32f, CV_32F);
  cv::Mat response;
  cv::cornerMinEigenVal(gray32f, response, 3, 3);
  return response;
}

cv::Mat ToGray8Image(const cv::Mat& image) {
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image.clone();
  } else if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    throw std::runtime_error("Unsupported image format for internal joint refine.");
  }
  if (gray.depth() == CV_8U) {
    return gray;
  }
  if (gray.depth() == CV_16U) {
    gray.convertTo(gray, CV_8U, 1.0 / 256.0);
  } else {
    gray.convertTo(gray, CV_8U);
  }
  return gray;
}

double SampleResponseNearest(const cv::Mat& response,
                             const Eigen::Vector2d& point) {
  if (response.empty()) {
    return 0.0;
  }
  const int x = static_cast<int>(std::lround(point.x()));
  const int y = static_cast<int>(std::lround(point.y()));
  if (x < 0 || y < 0 || x >= response.cols || y >= response.rows) {
    return 0.0;
  }
  return static_cast<double>(response.at<float>(y, x));
}

double EvaluateInternalJointObjective(
    const Eigen::Vector2d& candidate_xy,
    const Eigen::Vector2d& observed_xy,
    const Eigen::Vector2d& predicted_xy,
    double normalized_corner_response,
    const InternalJointRefineOptions& options) {
  const double sigma_geo =
      std::max(1e-6, options.geometry_sigma_px);
  const double sigma_obs =
      std::max(1e-6, options.observation_sigma_px);
  const double geometry_prior =
      std::exp(-0.5 * (candidate_xy - predicted_xy).squaredNorm() /
               (sigma_geo * sigma_geo));
  const double observation_prior =
      std::exp(-0.5 * (candidate_xy - observed_xy).squaredNorm() /
               (sigma_obs * sigma_obs));
  return ClampUnit(normalized_corner_response) * geometry_prior *
         observation_prior;
}

double ComputeCornerPatchMeanGradient(
    const cv::Mat& image,
    const std::vector<Eigen::Vector2d>& points) {
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
  for (const Eigen::Vector2d& point : points) {
    const int x = static_cast<int>(std::round(point.x()));
    const int y = static_cast<int>(std::round(point.y()));
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

double ComputeCornerPatchMeanGradient(
    const cv::Mat& image,
    const std::vector<InternalBlurFilterPointDecision*>& points) {
  if (image.empty() || points.empty()) {
    return 0.0;
  }
  std::vector<Eigen::Vector2d> image_points;
  image_points.reserve(points.size());
  for (const InternalBlurFilterPointDecision* point : points) {
    if (point == nullptr) {
      continue;
    }
    image_points.push_back(point->observed_image_xy);
  }
  return ComputeCornerPatchMeanGradient(image, image_points);
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

const char* ToString(PreBackendFilterThresholdMode mode) {
  switch (mode) {
    case PreBackendFilterThresholdMode::MeanStd:
      return "mean_std";
    case PreBackendFilterThresholdMode::MedianMad:
      return "median_mad";
  }
  return "mean_std";
}

PreBackendFilterThresholdMode ParsePreBackendFilterThresholdMode(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "mean_std" || lowered == "mean-std" ||
      lowered == "std") {
    return PreBackendFilterThresholdMode::MeanStd;
  }
  if (lowered == "median_mad" || lowered == "median-mad" ||
      lowered == "mad") {
    return PreBackendFilterThresholdMode::MedianMad;
  }
  throw std::runtime_error(
      "Unsupported pre-backend filter threshold mode: " + value);
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

const char* ToString(InternalObservationWeightMode mode) {
  switch (mode) {
    case InternalObservationWeightMode::Off:
      return "off";
    case InternalObservationWeightMode::Diagnostic:
      return "diagnostic";
    case InternalObservationWeightMode::Enabled:
      return "enabled";
  }
  return "unknown";
}

InternalObservationWeightMode ParseInternalObservationWeightMode(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "off") {
    return InternalObservationWeightMode::Off;
  }
  if (lowered == "diagnostic") {
    return InternalObservationWeightMode::Diagnostic;
  }
  if (lowered == "enabled") {
    return InternalObservationWeightMode::Enabled;
  }
  throw std::runtime_error("Unsupported internal observation weight mode: " +
                           value);
}

std::string NormalizeInternalObservationWeightPolicy(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "quality" || lowered == "residual_consistency") {
    return lowered;
  }
  throw std::runtime_error(
      "Unsupported internal observation weight policy: " + value);
}

const char* ToString(InternalBlurBoardWeightMode mode) {
  switch (mode) {
    case InternalBlurBoardWeightMode::Off:
      return "off";
    case InternalBlurBoardWeightMode::Diagnostic:
      return "diagnostic";
    case InternalBlurBoardWeightMode::Enabled:
      return "enabled";
  }
  return "unknown";
}

InternalBlurBoardWeightMode ParseInternalBlurBoardWeightMode(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "off") {
    return InternalBlurBoardWeightMode::Off;
  }
  if (lowered == "diagnostic") {
    return InternalBlurBoardWeightMode::Diagnostic;
  }
  if (lowered == "enabled") {
    return InternalBlurBoardWeightMode::Enabled;
  }
  throw std::runtime_error("Unsupported internal blur board weight mode: " +
                           value);
}

const char* ToString(InternalJointRefineMode mode) {
  switch (mode) {
    case InternalJointRefineMode::Off:
      return "off";
    case InternalJointRefineMode::Diagnostic:
      return "diagnostic";
    case InternalJointRefineMode::Enabled:
      return "enabled";
  }
  return "unknown";
}

InternalJointRefineMode ParseInternalJointRefineMode(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "off") {
    return InternalJointRefineMode::Off;
  }
  if (lowered == "diagnostic") {
    return InternalJointRefineMode::Diagnostic;
  }
  if (lowered == "enabled") {
    return InternalJointRefineMode::Enabled;
  }
  throw std::runtime_error("Unsupported internal joint refine mode: " +
                           value);
}

const char* ToString(InternalJointRefineTargetMode mode) {
  switch (mode) {
    case InternalJointRefineTargetMode::All:
      return "all";
    case InternalJointRefineTargetMode::HighResidualOnly:
      return "high_residual";
    case InternalJointRefineTargetMode::BlurBadBoardOnly:
      return "blur_bad_board";
    case InternalJointRefineTargetMode::HighResidualOrBlurBadBoard:
      return "high_residual_or_blur_bad_board";
    case InternalJointRefineTargetMode::HighResidualAndBlurBadBoard:
      return "high_residual_and_blur_bad_board";
  }
  return "unknown";
}

InternalJointRefineTargetMode ParseInternalJointRefineTargetMode(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "all") {
    return InternalJointRefineTargetMode::All;
  }
  if (lowered == "high_residual" || lowered == "high-residual") {
    return InternalJointRefineTargetMode::HighResidualOnly;
  }
  if (lowered == "blur_bad_board" || lowered == "blur-bad-board") {
    return InternalJointRefineTargetMode::BlurBadBoardOnly;
  }
  if (lowered == "high_residual_or_blur_bad_board" ||
      lowered == "high-residual-or-blur-bad-board") {
    return InternalJointRefineTargetMode::HighResidualOrBlurBadBoard;
  }
  if (lowered == "high_residual_and_blur_bad_board" ||
      lowered == "high-residual-and-blur-bad-board") {
    return InternalJointRefineTargetMode::HighResidualAndBlurBadBoard;
  }
  throw std::runtime_error(
      "Unsupported internal joint refine target mode: " + value);
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
    const double median = ComputeMedian(residuals);
    const double mad = ComputeMedianAbsoluteDeviation(residuals, median);
    double threshold = options.min_abs_threshold_px;
    if (options.threshold_mode == PreBackendFilterThresholdMode::MedianMad) {
      const double robust_sigma = 1.4826 * mad;
      threshold = std::max(options.min_abs_threshold_px,
                           median + options.sigma_threshold * robust_sigma);
    } else {
      threshold = std::max(options.min_abs_threshold_px,
                           mean + options.sigma_threshold * stddev);
    }

    PreBackendFilterBoardSummary board_summary;
    board_summary.frame_index = board_key.first;
    board_summary.frame_label = frame_labels[board_key.first];
    board_summary.board_id = board_key.second;
    board_summary.input_internal_point_count =
        static_cast<int>(indices.size());
    board_summary.mean_residual = mean;
    board_summary.std_residual = stddev;
    board_summary.median_residual = median;
    board_summary.mad_residual = mad;
    board_summary.threshold = threshold;

    for (std::size_t index : indices) {
      PreBackendFilterPointDecision& decision = result.point_decisions[index];
      decision.board_mean_residual = mean;
      decision.board_std_residual = stddev;
      decision.board_median_residual = median;
      decision.board_mad_residual = mad;
      decision.board_threshold = threshold;
      decision.filtered = decision.residual_norm > threshold;
      if (decision.filtered) {
        decision.filter_reason =
            options.threshold_mode == PreBackendFilterThresholdMode::MedianMad
                ? "residual_above_kalibr_style_median_mad_threshold"
                : "residual_above_kalibr_style_mean_std_threshold";
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

InternalBlurBoardWeightResult ApplyInternalBlurBoardWeights(
    const CalibrationStateBundle& bundle,
    const std::map<int, std::string>& frame_image_paths,
    const InternalBlurBoardWeightOptions& options) {
  InternalBlurBoardWeightResult result;
  result.options = options;
  result.diagnostic_only = options.mode != InternalBlurBoardWeightMode::Enabled;
  result.backend_input_changed =
      options.mode == InternalBlurBoardWeightMode::Enabled;
  result.curated_bundle = bundle;

  if (!bundle.IsReadyForBackend()) {
    result.failure_reason =
        "ApplyInternalBlurBoardWeights requires a ready-for-backend bundle.";
    return result;
  }
  if (options.low_patch_gradient_quantile < 0.0 ||
      options.low_patch_gradient_quantile > 1.0) {
    result.failure_reason =
        "internal blur board weight quantile must be in [0, 1].";
    return result;
  }
  if (options.min_board_internal_rmse_px < 0.0 ||
      options.min_board_p95_residual_px < 0.0) {
    result.failure_reason =
        "internal blur board weight residual thresholds must be non-negative.";
    return result;
  }
  if (options.min_weight < 0.0 || options.min_weight > 1.0) {
    result.failure_reason =
        "internal blur board weight min weight must be in [0, 1].";
    return result;
  }
  if (!(options.gradient_exponent > 0.0)) {
    result.failure_reason =
        "internal blur board weight gradient exponent must be positive.";
    return result;
  }
  if (options.mode == InternalBlurBoardWeightMode::Off) {
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

  struct BoardData {
    std::string frame_label;
    double input_weight = 1.0;
    std::vector<std::size_t> point_decision_indices;
    std::vector<double> residuals;
    std::vector<InternalBlurFilterPointDecision> blur_points;
    double internal_rmse = 0.0;
    double p95_residual = 0.0;
    double corner_patch_mean_gradient =
        std::numeric_limits<double>::infinity();
  };

  std::map<ObservationKey, std::size_t> decision_index_map;
  std::map<std::pair<int, int>, BoardData> board_data_map;
  for (const JointPointObservation& point :
       bundle.measurement_dataset.solver_observations) {
    if (!point.used_in_solver || point.point_type != JointPointType::Internal) {
      continue;
    }
    InternalBlurBoardWeightPointDecision decision;
    decision.frame_index = point.frame_index;
    decision.frame_label = point.frame_label;
    decision.board_id = point.board_id;
    decision.point_id = point.point_id;
    decision.source_point_index = point.source_point_index;
    decision.source_kind = point.source_kind;
    decision.input_weight = std::max(0.0, point.observation_weight);
    decision.output_weight = decision.input_weight;
    const std::size_t index = result.point_decisions.size();
    result.point_decisions.push_back(decision);

    ObservationKey key = MakeObservationKey(point);
    decision_index_map[key] = index;

    BoardData& board =
        board_data_map[std::make_pair(point.frame_index, point.board_id)];
    board.frame_label = point.frame_label;
    board.input_weight = decision.input_weight;
    board.point_decision_indices.push_back(index);
  }

  result.input_internal_point_count =
      static_cast<int>(result.point_decisions.size());

  for (const JointResidualPointDiagnostics& point :
       residual_result.point_diagnostics) {
    if (point.point_type != JointPointType::Internal || !point.used_in_solver) {
      continue;
    }
    ObservationKey key = MakeObservationKey(point);
    const auto index_it = decision_index_map.find(key);
    if (index_it == decision_index_map.end()) {
      continue;
    }
    BoardData& board =
        board_data_map[std::make_pair(point.frame_index, point.board_id)];
    board.residuals.push_back(point.residual_norm);

    InternalBlurFilterPointDecision blur_point;
    blur_point.frame_index = point.frame_index;
    blur_point.frame_label = point.frame_label;
    blur_point.board_id = point.board_id;
    blur_point.point_id = point.point_id;
    blur_point.source_point_index = point.source_point_index;
    blur_point.source_kind = point.source_kind;
    blur_point.observed_image_xy = point.observed_image_xy;
    blur_point.residual_norm = point.residual_norm;
    board.blur_points.push_back(blur_point);
  }

  std::map<int, cv::Mat> image_cache;
  std::vector<double> finite_patch_gradients;
  for (auto& entry : board_data_map) {
    BoardData& board = entry.second;
    if (!board.residuals.empty()) {
      double squared_sum = 0.0;
      for (double residual : board.residuals) {
        squared_sum += residual * residual;
      }
      board.internal_rmse =
          std::sqrt(squared_sum / static_cast<double>(board.residuals.size()));
      board.p95_residual = Quantile(board.residuals, 0.95);
    }

    const auto image_path_it = frame_image_paths.find(entry.first.first);
    if (image_path_it == frame_image_paths.end()) {
      result.warnings.push_back(
          "internal blur board weight skipped board because image path was not found");
      continue;
    }
    cv::Mat& image = image_cache[entry.first.first];
    if (image.empty()) {
      image = cv::imread(image_path_it->second, cv::IMREAD_UNCHANGED);
    }
    if (image.empty()) {
      result.warnings.push_back(
          "internal blur board weight skipped board because image read failed");
      continue;
    }
    std::vector<InternalBlurFilterPointDecision*> point_ptrs;
    point_ptrs.reserve(board.blur_points.size());
    for (InternalBlurFilterPointDecision& blur_point : board.blur_points) {
      point_ptrs.push_back(&blur_point);
    }
    board.corner_patch_mean_gradient =
        ComputeCornerPatchMeanGradient(image, point_ptrs);
    if (std::isfinite(board.corner_patch_mean_gradient)) {
      finite_patch_gradients.push_back(board.corner_patch_mean_gradient);
    }
  }

  result.input_board_observation_count =
      static_cast<int>(board_data_map.size());
  result.patch_gradient_threshold =
      Quantile(finite_patch_gradients, options.low_patch_gradient_quantile);

  std::map<ObservationKey, double> output_weights;
  std::vector<double> final_weights;
  for (auto& entry : board_data_map) {
    const std::pair<int, int>& board_key = entry.first;
    BoardData& board = entry.second;

    InternalBlurBoardWeightBoardSummary summary;
    summary.frame_index = board_key.first;
    summary.frame_label = board.frame_label;
    summary.board_id = board_key.second;
    summary.internal_point_count =
        static_cast<int>(board.point_decision_indices.size());
    summary.internal_rmse = board.internal_rmse;
    summary.p95_residual = board.p95_residual;
    summary.corner_patch_mean_gradient = board.corner_patch_mean_gradient;
    summary.patch_gradient_threshold = result.patch_gradient_threshold;
    summary.input_weight = board.input_weight;
    summary.low_patch_gradient =
        std::isfinite(board.corner_patch_mean_gradient) &&
        board.corner_patch_mean_gradient <= result.patch_gradient_threshold;
    summary.high_internal_residual =
        board.internal_rmse >= options.min_board_internal_rmse_px ||
        board.p95_residual >= options.min_board_p95_residual_px;
    summary.targeted_for_downweight =
        summary.low_patch_gradient && summary.high_internal_residual;

    double factor = 1.0;
    if (summary.targeted_for_downweight) {
      const double normalized_gradient =
          result.patch_gradient_threshold > 1e-9
              ? ClampUnit(board.corner_patch_mean_gradient /
                          result.patch_gradient_threshold)
              : 0.0;
      factor = options.min_weight +
               (1.0 - options.min_weight) *
                   std::pow(normalized_gradient, options.gradient_exponent);
      summary.weight_reason =
          "low_patch_gradient_and_high_internal_residual_downweight";
      ++result.downweighted_board_observation_count;
    } else if (!summary.low_patch_gradient && !summary.high_internal_residual) {
      summary.weight_reason = "accepted_gradient_and_residual";
    } else if (!summary.low_patch_gradient) {
      summary.weight_reason = "accepted_patch_gradient_not_low";
    } else {
      summary.weight_reason = "accepted_internal_residual_not_high";
    }
    summary.output_weight = summary.input_weight * factor;

    for (std::size_t point_index : board.point_decision_indices) {
      InternalBlurBoardWeightPointDecision& decision =
          result.point_decisions[point_index];
      decision.board_corner_patch_mean_gradient =
          summary.corner_patch_mean_gradient;
      decision.board_patch_gradient_threshold =
          summary.patch_gradient_threshold;
      decision.board_internal_rmse = summary.internal_rmse;
      decision.board_p95_residual = summary.p95_residual;
      decision.output_weight = decision.input_weight * factor;
      decision.downweighted =
          decision.output_weight < decision.input_weight - 1e-9;
      decision.weight_reason = summary.weight_reason;
      if (decision.downweighted) {
        ++summary.downweighted_internal_point_count;
        ++result.downweighted_internal_point_count;
      }

      ObservationKey key;
      key.frame_index = decision.frame_index;
      key.board_id = decision.board_id;
      key.point_id = decision.point_id;
      key.source_point_index = decision.source_point_index;
      key.source_kind = decision.source_kind;
      output_weights[key] = decision.output_weight;
      final_weights.push_back(decision.output_weight);
    }

    result.board_summaries.push_back(summary);
  }

  result.downweighted_internal_ratio =
      SafeRatio(result.downweighted_internal_point_count,
                result.input_internal_point_count);
  if (!final_weights.empty()) {
    result.min_weight =
        *std::min_element(final_weights.begin(), final_weights.end());
    result.max_weight =
        *std::max_element(final_weights.begin(), final_weights.end());
    result.mean_weight = ComputeMean(final_weights);
  }

  if (options.mode == InternalBlurBoardWeightMode::Enabled) {
    ApplyObservationWeightsToDataset(
        output_weights, &result.curated_bundle.measurement_dataset);
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
    warning << "pre-backend internal blur board weighting downweighted "
            << result.downweighted_internal_point_count << " / "
            << result.input_internal_point_count << " internal points across "
            << result.downweighted_board_observation_count
            << " board observations";
    result.curated_bundle.warnings.push_back(warning.str());
  }

  result.success = true;
  return result;
}

InternalJointRefineResult ApplyInternalJointRefinement(
    const CalibrationStateBundle& bundle,
    const std::map<int, std::string>& frame_image_paths,
    const InternalJointRefineOptions& options) {
  InternalJointRefineResult result;
  result.options = options;
  result.diagnostic_only = options.mode != InternalJointRefineMode::Enabled;
  result.backend_input_changed = false;
  result.curated_bundle = bundle;

  if (!bundle.IsReadyForBackend()) {
    result.failure_reason =
        "ApplyInternalJointRefinement requires a ready-for-backend bundle.";
    return result;
  }
  if (!(options.search_radius_px > 0.0) ||
      !(options.max_displacement_px > 0.0) ||
      !(options.geometry_sigma_px > 0.0) ||
      !(options.observation_sigma_px > 0.0) ||
      options.subpix_window_radius < 1 ||
      options.min_objective_improvement < 0.0 ||
      options.min_corner_response_gain < 0.0 ||
      options.min_board_internal_rmse_improvement_px < 0.0 ||
      options.min_refined_point_count_per_board < 1 ||
      options.accept_max_global_outer_delta_px < 0.0 ||
      options.accept_max_frame_outer_delta_px < 0.0 ||
      options.acceptance_backend_max_iterations < 1) {
    result.failure_reason =
        "internal joint refine options must be positive and valid.";
    return result;
  }
  if (options.min_old_residual_px < 0.0 ||
      options.low_patch_gradient_quantile < 0.0 ||
      options.low_patch_gradient_quantile > 1.0 ||
      options.min_board_internal_rmse_px < 0.0 ||
      options.min_board_p95_residual_px < 0.0) {
    result.failure_reason =
        "internal joint refine selective thresholds must be non-negative and quantiles in [0, 1].";
    return result;
  }
  if (options.mode == InternalJointRefineMode::Off) {
    result.success = true;
    return result;
  }
  if (options.target_mode !=
      InternalJointRefineTargetMode::HighResidualAndBlurBadBoard) {
    result.failure_reason =
        "outer-safe selective refine v1 only supports target_mode=high_residual_and_blur_bad_board.";
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

  struct FrameImageData {
    cv::Mat gray8;
    cv::Mat response;
    double response_max = 0.0;
    bool load_attempted = false;
    bool image_valid = false;
    bool response_evaluated = false;
    bool response_valid = false;
  };

  struct BoardTargetStats {
    int frame_index = -1;
    std::string frame_label;
    int board_id = -1;
    std::vector<double> residuals;
    std::vector<Eigen::Vector2d> observed_points;
    std::vector<std::size_t> decision_indices;
    double internal_rmse = 0.0;
    double p95_residual = 0.0;
    double corner_patch_mean_gradient = 0.0;
    bool low_patch_gradient = false;
    bool high_internal_residual = false;
    bool targeted_blur_bad_board = false;
  };

  std::map<int, FrameImageData> frame_cache;
  const auto ensure_frame_image_loaded =
      [&](int frame_index, FrameImageData* frame_data) -> bool {
    if (frame_data == nullptr) {
      return false;
    }
    if (!frame_data->load_attempted) {
      frame_data->load_attempted = true;
      const auto image_path_it = frame_image_paths.find(frame_index);
      if (image_path_it == frame_image_paths.end()) {
        return false;
      }
      const cv::Mat image =
          cv::imread(image_path_it->second, cv::IMREAD_UNCHANGED);
      if (image.empty()) {
        return false;
      }
      frame_data->gray8 = ToGray8Image(image);
      frame_data->image_valid = !frame_data->gray8.empty();
    }
    return frame_data->image_valid;
  };
  const auto ensure_frame_response_ready =
      [&](int frame_index, FrameImageData* frame_data) -> bool {
    if (!ensure_frame_image_loaded(frame_index, frame_data)) {
      return false;
    }
    if (!frame_data->response_evaluated) {
      frame_data->response_evaluated = true;
      frame_data->response = ComputeCornerResponseMap(frame_data->gray8);
      double response_min = 0.0;
      cv::minMaxLoc(frame_data->response, &response_min,
                    &frame_data->response_max);
      frame_data->response_valid =
          !frame_data->response.empty() && frame_data->response_max > 1e-9;
    }
    return frame_data->response_valid;
  };

  std::map<ObservationKey, std::size_t> decision_index_map;
  std::map<std::pair<int, int>, BoardTargetStats> board_stats_map;
  std::map<int, InternalJointRefineFrameSummary> frame_summary_map;
  for (const JointPointObservation& point :
       bundle.measurement_dataset.solver_observations) {
    if (!point.used_in_solver || point.point_type != JointPointType::Internal) {
      continue;
    }
    InternalJointRefinePointDecision decision;
    decision.frame_index = point.frame_index;
    decision.frame_label = point.frame_label;
    decision.board_id = point.board_id;
    decision.point_id = point.point_id;
    decision.source_point_index = point.source_point_index;
    decision.source_kind = point.source_kind;
    decision.observed_image_xy = point.image_xy;
    decision.refined_image_xy = point.image_xy;
    const std::size_t index = result.point_decisions.size();
    result.point_decisions.push_back(decision);
    decision_index_map[MakeObservationKey(point)] = index;

    InternalJointRefineFrameSummary& frame =
        frame_summary_map[point.frame_index];
    frame.frame_index = point.frame_index;
    frame.frame_label = point.frame_label;
    ++frame.input_internal_point_count;
  }
  result.input_internal_point_count =
      static_cast<int>(result.point_decisions.size());

  for (const JointResidualPointDiagnostics& point :
       residual_result.point_diagnostics) {
    if (point.point_type != JointPointType::Internal || !point.used_in_solver) {
      continue;
    }
    const auto decision_it = decision_index_map.find(MakeObservationKey(point));
    if (decision_it == decision_index_map.end()) {
      continue;
    }
    InternalJointRefinePointDecision& decision =
        result.point_decisions[decision_it->second];
    decision.predicted_image_xy = point.predicted_image_xy;
    decision.old_residual_norm = point.residual_norm;
    decision.new_residual_norm = point.residual_norm;

    BoardTargetStats& board =
        board_stats_map[std::make_pair(point.frame_index, point.board_id)];
    board.frame_index = point.frame_index;
    board.frame_label = point.frame_label;
    board.board_id = point.board_id;
    board.residuals.push_back(point.residual_norm);
    board.observed_points.push_back(point.observed_image_xy);
    board.decision_indices.push_back(decision_it->second);
  }

  std::vector<double> finite_patch_gradients;
  for (auto& entry : board_stats_map) {
    BoardTargetStats& board = entry.second;
    if (!board.residuals.empty()) {
      double squared_error_sum = 0.0;
      for (double residual : board.residuals) {
        squared_error_sum += residual * residual;
      }
      board.internal_rmse =
          std::sqrt(squared_error_sum /
                    static_cast<double>(board.residuals.size()));
      board.p95_residual = Quantile(board.residuals, 0.95);
    }

    FrameImageData& frame_data = frame_cache[board.frame_index];
    if (ensure_frame_image_loaded(board.frame_index, &frame_data)) {
      board.corner_patch_mean_gradient =
          ComputeCornerPatchMeanGradient(frame_data.gray8,
                                         board.observed_points);
      if (std::isfinite(board.corner_patch_mean_gradient)) {
        finite_patch_gradients.push_back(board.corner_patch_mean_gradient);
      }
    }
  }
  result.patch_gradient_threshold =
      Quantile(finite_patch_gradients, options.low_patch_gradient_quantile);
  for (auto& entry : board_stats_map) {
    BoardTargetStats& board = entry.second;
    board.low_patch_gradient =
        std::isfinite(board.corner_patch_mean_gradient) &&
        board.corner_patch_mean_gradient <= result.patch_gradient_threshold;
    board.high_internal_residual =
        board.internal_rmse >= options.min_board_internal_rmse_px ||
        board.p95_residual >= options.min_board_p95_residual_px;
    board.targeted_blur_bad_board =
        board.low_patch_gradient && board.high_internal_residual;
    if (board.targeted_blur_bad_board) {
      ++result.targeted_blur_bad_board_count;
    }
  }

  for (InternalJointRefinePointDecision& decision : result.point_decisions) {
    const auto board_it = board_stats_map.find(
        std::make_pair(decision.frame_index, decision.board_id));
    if (board_it != board_stats_map.end()) {
      const BoardTargetStats& board = board_it->second;
      decision.board_internal_rmse = board.internal_rmse;
      decision.board_p95_residual = board.p95_residual;
      decision.board_corner_patch_mean_gradient =
          board.corner_patch_mean_gradient;
      decision.board_patch_gradient_threshold = result.patch_gradient_threshold;
      decision.board_low_patch_gradient = board.low_patch_gradient;
      decision.board_high_internal_residual = board.high_internal_residual;
      decision.targeted_by_blur_bad_board = board.targeted_blur_bad_board;
    }
    decision.targeted_by_high_residual =
        decision.old_residual_norm >= options.min_old_residual_px;
    switch (options.target_mode) {
      case InternalJointRefineTargetMode::All:
        decision.eligible_for_refine = true;
        break;
      case InternalJointRefineTargetMode::HighResidualOnly:
        decision.eligible_for_refine = decision.targeted_by_high_residual;
        break;
      case InternalJointRefineTargetMode::BlurBadBoardOnly:
        decision.eligible_for_refine = decision.targeted_by_blur_bad_board;
        break;
      case InternalJointRefineTargetMode::HighResidualOrBlurBadBoard:
        decision.eligible_for_refine =
            decision.targeted_by_high_residual ||
            decision.targeted_by_blur_bad_board;
        break;
      case InternalJointRefineTargetMode::HighResidualAndBlurBadBoard:
        decision.eligible_for_refine =
            decision.targeted_by_high_residual &&
            decision.targeted_by_blur_bad_board;
        break;
    }
    InternalJointRefineFrameSummary& frame = frame_summary_map[decision.frame_index];
    frame.mean_residual_before += decision.old_residual_norm;
    frame.mean_residual_after += decision.old_residual_norm;
    if (decision.eligible_for_refine) {
      ++result.eligible_internal_point_count;
      ++frame.eligible_internal_point_count;
    } else {
      decision.refine_reason = "not_targeted_by_selective_gate";
    }
  }

  result.eligible_ratio = SafeRatio(result.eligible_internal_point_count,
                                    result.input_internal_point_count);

  std::vector<std::pair<std::pair<int, int>, BoardTargetStats*> > board_order;
  for (auto& entry : board_stats_map) {
    bool has_eligible_points = false;
    for (std::size_t decision_index : entry.second.decision_indices) {
      if (result.point_decisions[decision_index].eligible_for_refine) {
        has_eligible_points = true;
        break;
      }
    }
    if (!has_eligible_points) {
      continue;
    }
    board_order.push_back(std::make_pair(entry.first, &entry.second));
  }
  std::sort(board_order.begin(), board_order.end(),
            [](const std::pair<std::pair<int, int>, BoardTargetStats*>& lhs,
               const std::pair<std::pair<int, int>, BoardTargetStats*>& rhs) {
              if (lhs.second->internal_rmse != rhs.second->internal_rmse) {
                return lhs.second->internal_rmse > rhs.second->internal_rmse;
              }
              if (lhs.first.first != rhs.first.first) {
                return lhs.first.first < rhs.first.first;
              }
              return lhs.first.second < rhs.first.second;
            });
  result.candidate_board_count = static_cast<int>(board_order.size());

  CalibrationStateBundle working_bundle = bundle;
  working_bundle.residual_result = residual_result;

  BackendProblemOptions acceptance_backend_problem_options;
  acceptance_backend_problem_options.optimize_frame_poses = true;
  acceptance_backend_problem_options.optimize_board_poses = true;
  acceptance_backend_problem_options.optimize_intrinsics = false;
  acceptance_backend_problem_options.delayed_intrinsics_release = false;
  acceptance_backend_problem_options.intrinsics_release_iteration = 0;

  AslamBackendCalibrationOptions acceptance_runner_options;
  acceptance_runner_options.max_iterations =
      options.acceptance_backend_max_iterations;
  acceptance_runner_options.convergence_delta_j = 1e-3;
  acceptance_runner_options.convergence_delta_x = 1e-4;
  acceptance_runner_options.levenberg_marquardt_lambda_init = 1e-3;
  acceptance_runner_options.linear_solver = "cholmod";
  acceptance_runner_options.verbose = false;
  acceptance_runner_options.use_huber_loss = true;
  acceptance_runner_options.outer_huber_delta_pixels = 10.0;
  acceptance_runner_options.internal_huber_delta_pixels = 6.0;
  acceptance_runner_options.invalid_projection_penalty_pixels = 100.0;
  acceptance_runner_options.export_cost_parity_diagnostics = false;
  acceptance_runner_options.run_jacobian_consistency_check = false;
  acceptance_runner_options.force_pose_only = false;
  const AslamBackendCalibrationRunner acceptance_backend_runner(
      acceptance_runner_options);

  AslamBackendCalibrationResult current_acceptance_result;
  std::map<int, double> current_frame_outer_rmse;
  if (!board_order.empty()) {
    const CalibrationBackendProblemInput current_backend_problem_input =
        BuildBackendProblemInput(working_bundle, acceptance_backend_problem_options);
    current_acceptance_result =
        acceptance_backend_runner.Run(current_backend_problem_input);
    if (!current_acceptance_result.success) {
      result.failure_reason =
          "outer-safe selective refine baseline acceptance backend failed: " +
          current_acceptance_result.failure_reason;
      result.warnings = current_acceptance_result.warnings;
      return result;
    }
    current_frame_outer_rmse = ComputeFrameOuterOnlyRmseMap(
        current_acceptance_result.optimized_residual);
  }

  double accepted_displacement_sum = 0.0;
  for (const auto& board_entry : board_order) {
    const BoardTargetStats& board = *board_entry.second;
    InternalJointRefineBoardSummary summary;
    summary.frame_index = board.frame_index;
    summary.frame_label = board.frame_label;
    summary.board_id = board.board_id;
    summary.corner_patch_mean_gradient = board.corner_patch_mean_gradient;
    summary.patch_gradient_threshold = result.patch_gradient_threshold;
    summary.low_patch_gradient = board.low_patch_gradient;
    summary.high_internal_residual = board.high_internal_residual;
    summary.targeted_for_refine = true;
    summary.board_p95_residual_before = board.p95_residual;
    summary.board_internal_rmse_before = ComputeBoardInternalRmseFromResidual(
        working_bundle.residual_result, board.frame_index, board.board_id);
    summary.global_outer_only_rmse_before =
        current_acceptance_result.optimized_residual.outer_only_rmse;
    summary.frame_outer_only_rmse_before =
        LookupFrameOuterOnlyRmse(current_frame_outer_rmse, board.frame_index);

    std::map<ObservationKey, Eigen::Vector2d> board_refined_positions;
    double corner_response_gain_sum = 0.0;
    for (std::size_t decision_index : board.decision_indices) {
      InternalJointRefinePointDecision& decision =
          result.point_decisions[decision_index];
      ++summary.input_internal_point_count;
      if (!decision.eligible_for_refine) {
        continue;
      }
      ++summary.eligible_internal_point_count;

      if (!std::isfinite(decision.predicted_image_xy.x()) ||
          !std::isfinite(decision.predicted_image_xy.y())) {
        decision.refine_reason = "predicted_point_invalid";
        continue;
      }

      FrameImageData& frame_data = frame_cache[decision.frame_index];
      if (!ensure_frame_image_loaded(decision.frame_index, &frame_data)) {
        decision.refine_reason = "image_read_failed";
        continue;
      }
      if (!ensure_frame_response_ready(decision.frame_index, &frame_data)) {
        decision.refine_reason = "invalid_corner_response_map";
        continue;
      }

      const Eigen::Vector2d observed_xy = decision.observed_image_xy;
      const Eigen::Vector2d predicted_xy = decision.predicted_image_xy;
      const double observed_response =
          SampleResponseNearest(frame_data.response, observed_xy) /
          frame_data.response_max;
      const double old_objective =
          EvaluateInternalJointObjective(observed_xy, observed_xy, predicted_xy,
                                         observed_response, options);
      decision.old_corner_response = observed_response;
      decision.old_objective = old_objective;

      Eigen::Vector2d best_xy = observed_xy;
      double best_response = observed_response;
      double best_objective = old_objective;

      const Eigen::Vector2d search_center =
          0.5 * (observed_xy + predicted_xy);
      const int search_radius_int =
          static_cast<int>(std::ceil(options.search_radius_px));
      for (int dy = -search_radius_int; dy <= search_radius_int; ++dy) {
        for (int dx = -search_radius_int; dx <= search_radius_int; ++dx) {
          const Eigen::Vector2d candidate_xy =
              search_center + Eigen::Vector2d(dx, dy);
          if ((candidate_xy - observed_xy).norm() >
              options.max_displacement_px) {
            continue;
          }
          if (candidate_xy.x() < 0.0 || candidate_xy.y() < 0.0 ||
              candidate_xy.x() >= static_cast<double>(frame_data.gray8.cols) ||
              candidate_xy.y() >= static_cast<double>(frame_data.gray8.rows)) {
            continue;
          }
          const double response =
              SampleResponseNearest(frame_data.response, candidate_xy) /
              frame_data.response_max;
          const double objective = EvaluateInternalJointObjective(
              candidate_xy, observed_xy, predicted_xy, response, options);
          if (objective > best_objective) {
            best_xy = candidate_xy;
            best_response = response;
            best_objective = objective;
          }
        }
      }

      std::vector<cv::Point2f> corners(1);
      corners[0] =
          cv::Point2f(static_cast<float>(best_xy.x()),
                      static_cast<float>(best_xy.y()));
      cv::cornerSubPix(frame_data.gray8, corners,
                       cv::Size(options.subpix_window_radius,
                                options.subpix_window_radius),
                       cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::EPS |
                                            cv::TermCriteria::MAX_ITER,
                                        20, 0.01));
      const Eigen::Vector2d subpix_xy(corners[0].x, corners[0].y);
      if ((subpix_xy - observed_xy).norm() <= options.max_displacement_px &&
          subpix_xy.x() >= 0.0 && subpix_xy.y() >= 0.0 &&
          subpix_xy.x() < static_cast<double>(frame_data.gray8.cols) &&
          subpix_xy.y() < static_cast<double>(frame_data.gray8.rows)) {
        const double subpix_response =
            SampleResponseNearest(frame_data.response, subpix_xy) /
            frame_data.response_max;
        const double subpix_objective =
            EvaluateInternalJointObjective(subpix_xy, observed_xy, predicted_xy,
                                           subpix_response, options);
        if (subpix_objective > best_objective) {
          best_xy = subpix_xy;
          best_response = subpix_response;
          best_objective = subpix_objective;
        }
      }

      decision.refined_image_xy = best_xy;
      decision.new_corner_response = best_response;
      decision.new_objective = best_objective;
      decision.new_residual_norm = (best_xy - predicted_xy).norm();
      decision.displacement_px = (best_xy - observed_xy).norm();
      decision.corner_response_gain =
          decision.new_corner_response - decision.old_corner_response;

      decision.tentative_refined =
          best_objective >
              old_objective + options.min_objective_improvement &&
          decision.corner_response_gain >= options.min_corner_response_gain &&
          decision.displacement_px > 1e-6;
      if (!decision.tentative_refined) {
        decision.refine_reason = "insufficient_image_gain";
        decision.refined_image_xy = observed_xy;
        decision.new_corner_response = decision.old_corner_response;
        decision.new_objective = decision.old_objective;
        decision.new_residual_norm = decision.old_residual_norm;
        decision.displacement_px = 0.0;
        decision.corner_response_gain = 0.0;
        continue;
      }

      ObservationKey key;
      key.frame_index = decision.frame_index;
      key.board_id = decision.board_id;
      key.point_id = decision.point_id;
      key.source_point_index = decision.source_point_index;
      key.source_kind = decision.source_kind;
      board_refined_positions[key] = best_xy;
      ++summary.tentative_refined_point_count;
      corner_response_gain_sum += decision.corner_response_gain;
      decision.refine_reason = "tentative_refine_candidate";
    }

    if (summary.tentative_refined_point_count > 0) {
      summary.mean_corner_response_gain =
          corner_response_gain_sum /
          static_cast<double>(summary.tentative_refined_point_count);
    }

    const int frame_index = board.frame_index;
    InternalJointRefineFrameSummary& frame_summary =
        frame_summary_map[frame_index];

    if (summary.tentative_refined_point_count <
        options.min_refined_point_count_per_board) {
      summary.rolled_back = true;
      summary.rollback_reason = "insufficient_tentative_refined_points";
      ++result.rolled_back_board_count;
      ++frame_summary.rolled_back_board_count;
      for (std::size_t decision_index : board.decision_indices) {
        InternalJointRefinePointDecision& decision =
            result.point_decisions[decision_index];
        decision.accepted_after_board_rollback = false;
        decision.refined = false;
        if (decision.tentative_refined) {
          decision.refine_reason = summary.rollback_reason;
        }
      }
      result.board_summaries.push_back(summary);
      continue;
    }

    CalibrationStateBundle tentative_bundle = working_bundle;
    ApplyRefinedPositionsToDataset(board_refined_positions,
                                   &tentative_bundle.measurement_dataset);
    const JointMeasurementBuildResult tentative_measurement_result =
        BuildMeasurementResultFromDataset(
            tentative_bundle.measurement_dataset,
            tentative_bundle.scene_state.reference_board_id);
    const JointResidualEvaluationResult tentative_residual_result =
        residual_evaluator.Evaluate(tentative_measurement_result, scene_state);
    if (!tentative_residual_result.success) {
      summary.rolled_back = true;
      summary.rollback_reason = "tentative_residual_evaluation_failed";
      ++result.rolled_back_board_count;
      ++frame_summary.rolled_back_board_count;
      for (std::size_t decision_index : board.decision_indices) {
        InternalJointRefinePointDecision& decision =
            result.point_decisions[decision_index];
        decision.accepted_after_board_rollback = false;
        decision.refined = false;
        if (decision.tentative_refined) {
          decision.refine_reason = summary.rollback_reason;
        }
      }
      result.board_summaries.push_back(summary);
      continue;
    }
    tentative_bundle.residual_result = tentative_residual_result;
    summary.board_internal_rmse_after_tentative =
        ComputeBoardInternalRmseFromResidual(
            tentative_residual_result, board.frame_index, board.board_id);
    summary.board_internal_rmse_improvement =
        summary.board_internal_rmse_before -
        summary.board_internal_rmse_after_tentative;

    if (summary.board_internal_rmse_improvement <
        options.min_board_internal_rmse_improvement_px) {
      summary.rolled_back = true;
      summary.rollback_reason = "insufficient_board_internal_improvement";
      ++result.rolled_back_board_count;
      ++frame_summary.rolled_back_board_count;
      for (std::size_t decision_index : board.decision_indices) {
        InternalJointRefinePointDecision& decision =
            result.point_decisions[decision_index];
        decision.accepted_after_board_rollback = false;
        decision.refined = false;
        if (decision.tentative_refined) {
          decision.refine_reason = summary.rollback_reason;
        }
      }
      result.board_summaries.push_back(summary);
      continue;
    }

    const CalibrationBackendProblemInput tentative_backend_problem_input =
        BuildBackendProblemInput(tentative_bundle, acceptance_backend_problem_options);
    const AslamBackendCalibrationResult tentative_backend_result =
        acceptance_backend_runner.Run(tentative_backend_problem_input);
    if (!tentative_backend_result.success) {
      summary.rolled_back = true;
      summary.rollback_reason = "acceptance_backend_failed";
      ++result.rolled_back_board_count;
      ++frame_summary.rolled_back_board_count;
      for (std::size_t decision_index : board.decision_indices) {
        InternalJointRefinePointDecision& decision =
            result.point_decisions[decision_index];
        decision.accepted_after_board_rollback = false;
        decision.refined = false;
        if (decision.tentative_refined) {
          decision.refine_reason = summary.rollback_reason;
        }
      }
      result.board_summaries.push_back(summary);
      continue;
    }

    const std::map<int, double> tentative_frame_outer_rmse =
        ComputeFrameOuterOnlyRmseMap(tentative_backend_result.optimized_residual);
    summary.global_outer_only_rmse_after =
        tentative_backend_result.optimized_residual.outer_only_rmse;
    summary.global_outer_only_rmse_delta =
        summary.global_outer_only_rmse_after -
        summary.global_outer_only_rmse_before;
    summary.frame_outer_only_rmse_after =
        LookupFrameOuterOnlyRmse(tentative_frame_outer_rmse, board.frame_index);
    summary.frame_outer_only_rmse_delta =
        summary.frame_outer_only_rmse_after -
        summary.frame_outer_only_rmse_before;

    if (summary.global_outer_only_rmse_delta >
            options.accept_max_global_outer_delta_px ||
        summary.frame_outer_only_rmse_delta >
            options.accept_max_frame_outer_delta_px) {
      summary.rolled_back = true;
      summary.rollback_reason = "outer_consistency_protection_failed";
      ++result.rolled_back_board_count;
      ++frame_summary.rolled_back_board_count;
      for (std::size_t decision_index : board.decision_indices) {
        InternalJointRefinePointDecision& decision =
            result.point_decisions[decision_index];
        decision.accepted_after_board_rollback = false;
        decision.refined = false;
        if (decision.tentative_refined) {
          decision.refine_reason = summary.rollback_reason;
        }
      }
      result.board_summaries.push_back(summary);
      continue;
    }

    summary.accepted = true;
    summary.accepted_refined_point_count = summary.tentative_refined_point_count;
    ++result.accepted_board_count;
    ++frame_summary.accepted_board_count;
    for (std::size_t decision_index : board.decision_indices) {
      InternalJointRefinePointDecision& decision =
          result.point_decisions[decision_index];
      if (!decision.tentative_refined) {
        continue;
      }
      decision.accepted_after_board_rollback = true;
      decision.refined = true;
      decision.refine_reason = "accepted_after_outer_safe_backend_check";
      ++result.refined_internal_point_count;
      ++frame_summary.refined_internal_point_count;
      frame_summary.mean_displacement_px += decision.displacement_px;
      frame_summary.max_displacement_px =
          std::max(frame_summary.max_displacement_px, decision.displacement_px);
      accepted_displacement_sum += decision.displacement_px;
      result.max_displacement_px =
          std::max(result.max_displacement_px, decision.displacement_px);
      frame_summary.mean_residual_after +=
          (decision.new_residual_norm - decision.old_residual_norm);
    }

    working_bundle = tentative_bundle;
    working_bundle.residual_result = tentative_residual_result;
    current_acceptance_result = tentative_backend_result;
    current_frame_outer_rmse = tentative_frame_outer_rmse;
    result.board_summaries.push_back(summary);
  }

  for (InternalJointRefineFrameSummary& frame : result.frame_summaries) {
    (void)frame;
  }

  result.refined_ratio = SafeRatio(result.refined_internal_point_count,
                                   result.input_internal_point_count);
  if (result.refined_internal_point_count > 0) {
    result.mean_displacement_px =
        accepted_displacement_sum /
        static_cast<double>(result.refined_internal_point_count);
  }

  for (auto& entry : frame_summary_map) {
    InternalJointRefineFrameSummary frame = entry.second;
    frame.eligible_ratio =
        SafeRatio(frame.eligible_internal_point_count,
                  frame.input_internal_point_count);
    frame.refined_ratio =
        SafeRatio(frame.refined_internal_point_count,
                  frame.input_internal_point_count);
    if (frame.refined_internal_point_count > 0) {
      frame.mean_displacement_px /=
          static_cast<double>(frame.refined_internal_point_count);
    }
    if (frame.input_internal_point_count > 0) {
      frame.mean_residual_before /=
          static_cast<double>(frame.input_internal_point_count);
      frame.mean_residual_after /=
          static_cast<double>(frame.input_internal_point_count);
    }
    result.frame_summaries.push_back(frame);
  }

  if (result.eligible_internal_point_count == 0) {
    result.warnings.push_back(
        "internal joint refine selective gate produced zero eligible internal points");
  }

  if (options.mode == InternalJointRefineMode::Enabled &&
      result.accepted_board_count > 0) {
    result.curated_bundle = working_bundle;
    result.backend_input_changed = true;
    std::ostringstream warning;
    warning << "outer-safe selective internal joint refine accepted "
            << result.refined_internal_point_count << " refined internal points across "
            << result.accepted_board_count << " boards, rolled back "
            << result.rolled_back_board_count << " boards";
    result.curated_bundle.warnings.push_back(warning.str());
  }

  result.success = true;
  return result;
}

InternalObservationWeightResult ApplyInternalObservationWeights(
    const CalibrationStateBundle& bundle,
    const InternalObservationWeightOptions& options) {
  InternalObservationWeightResult result;
  result.options = options;
  result.policy = NormalizeInternalObservationWeightPolicy(options.policy);
  result.residual_consistency_sigma_multiplier =
      options.residual_consistency_sigma_multiplier;
  result.residual_consistency_min_rmse =
      options.residual_consistency_min_rmse;
  result.diagnostic_only = options.mode != InternalObservationWeightMode::Enabled;
  result.backend_input_changed =
      options.mode == InternalObservationWeightMode::Enabled;
  result.curated_bundle = bundle;

  if (!bundle.IsReadyForBackend()) {
    result.failure_reason =
        "ApplyInternalObservationWeights requires a ready-for-backend bundle.";
    return result;
  }
  if (options.min_weight < 0.0 || options.min_weight > 1.0) {
    result.failure_reason = "internal observation min weight must be in [0, 1].";
    return result;
  }
  if (options.low_quality_quantile < 0.0 || options.low_quality_quantile > 1.0) {
    result.failure_reason =
        "internal observation low quality quantile must be in [0, 1].";
    return result;
  }
  if (!(options.quality_exponent > 0.0)) {
    result.failure_reason = "internal observation quality exponent must be positive.";
    return result;
  }
  if (!(options.residual_consistency_sigma_multiplier >= 0.0)) {
    result.failure_reason =
        "internal observation residual consistency sigma multiplier must be non-negative.";
    return result;
  }
  if (!(options.residual_consistency_min_rmse > 0.0)) {
    result.failure_reason =
        "internal observation residual consistency min rmse must be positive.";
    return result;
  }
  if (options.mode == InternalObservationWeightMode::Off) {
    result.success = true;
    return result;
  }

  std::map<ObservationKey, double> output_weights;
  std::map<std::pair<int, int>, std::vector<std::size_t> > board_to_decisions;
  std::map<std::pair<int, int>, BoardResidualConsistencyStats>
      residual_consistency_by_board;
  if (result.policy == "residual_consistency") {
    const JointReprojectionSceneState scene_state =
        BuildJointSceneStateFromCalibrationSceneState(bundle.scene_state);
    const JointMeasurementBuildResult measurement_result =
        BuildMeasurementResultFromDataset(bundle.measurement_dataset,
                                          bundle.scene_state.reference_board_id);
    if (!measurement_result.success) {
      result.failure_reason = measurement_result.failure_reason;
      return result;
    }
    JointResidualEvaluationOptions residual_options;
    residual_options.top_k = -1;
    const JointReprojectionResidualEvaluator residual_evaluator(
        residual_options);
    const JointResidualEvaluationResult residual_result =
        residual_evaluator.Evaluate(measurement_result, scene_state);
    if (!residual_result.success) {
      result.failure_reason = residual_result.failure_reason;
      result.warnings = residual_result.warnings;
      return result;
    }
    residual_consistency_by_board =
        ComputeBoardResidualConsistencyStats(
            residual_result, options.residual_consistency_min_rmse);
  }
  std::vector<double> qualities;
  std::vector<double> consistency_ratios;
  std::vector<double> weights;
  for (const JointPointObservation& point :
       bundle.measurement_dataset.solver_observations) {
    if (!point.used_in_solver || point.point_type != JointPointType::Internal) {
      continue;
    }

    InternalObservationWeightDecision decision;
    decision.frame_index = point.frame_index;
    decision.frame_label = point.frame_label;
    decision.board_id = point.board_id;
    decision.point_id = point.point_id;
    decision.source_point_index = point.source_point_index;
    decision.source_kind = point.source_kind;
    decision.quality = point.quality;
    decision.input_weight = std::max(0.0, point.observation_weight);
    const auto residual_it =
        residual_consistency_by_board.find(
            std::make_pair(point.frame_index, point.board_id));
    if (residual_it != residual_consistency_by_board.end()) {
      decision.board_internal_rmse = residual_it->second.internal_rmse;
      decision.board_outer_rmse = residual_it->second.outer_rmse;
      decision.residual_consistency_ratio = residual_it->second.ratio;
      if (residual_it->second.internal_count > 0 &&
          residual_it->second.outer_count > 0) {
        consistency_ratios.push_back(residual_it->second.ratio);
      }
    }
    qualities.push_back(ClampUnit(point.quality));
    const std::size_t decision_index = result.point_decisions.size();
    result.point_decisions.push_back(decision);
    board_to_decisions[std::make_pair(point.frame_index, point.board_id)]
        .push_back(decision_index);
  }

  result.quality_threshold = Quantile(qualities, options.low_quality_quantile);
  if (!consistency_ratios.empty()) {
    const double mean_ratio = ComputeMean(consistency_ratios);
    const double std_ratio = ComputeStd(consistency_ratios, mean_ratio);
    result.residual_consistency_ratio_threshold =
        std::max(1.0, mean_ratio +
                          options.residual_consistency_sigma_multiplier *
                              std_ratio);
  }
  for (InternalObservationWeightDecision& decision : result.point_decisions) {
    if (result.policy == "residual_consistency") {
      const double ratio = decision.residual_consistency_ratio;
      if (ratio <= 0.0 ||
          ratio <= result.residual_consistency_ratio_threshold) {
        decision.output_weight = decision.input_weight;
        decision.downweighted = false;
        decision.weight_reason = "residual_consistent_with_outer_pose";
      } else {
        const double normalized =
            result.residual_consistency_ratio_threshold / ratio;
        const double factor =
            options.min_weight +
            (1.0 - options.min_weight) *
                std::pow(ClampUnit(normalized), options.quality_exponent);
        decision.output_weight = decision.input_weight * factor;
        decision.downweighted =
            decision.output_weight < decision.input_weight - 1e-9;
        decision.weight_reason =
            decision.downweighted ? "residual_inconsistent_board_downweight"
                                  : "residual_inconsistent_weight_is_one";
      }
    } else {
      const double quality = ClampUnit(decision.quality);
      if (quality > result.quality_threshold) {
        decision.output_weight = decision.input_weight;
        decision.downweighted = false;
        decision.weight_reason = "quality_above_quantile_threshold";
      } else {
        const double factor =
            options.min_weight +
            (1.0 - options.min_weight) *
                std::pow(quality, options.quality_exponent);
        decision.output_weight = decision.input_weight * factor;
        decision.downweighted =
            decision.output_weight < decision.input_weight - 1e-9;
        decision.weight_reason =
            decision.downweighted ? "low_quality_quantile_downweight"
                                  : "quantile_selected_but_weight_is_one";
      }
    }
    output_weights[ObservationKey{decision.frame_index, decision.board_id,
                                  decision.point_id,
                                  decision.source_point_index,
                                  decision.source_kind}] =
        decision.output_weight;
    weights.push_back(decision.output_weight);
  }

  result.input_internal_point_count =
      static_cast<int>(result.point_decisions.size());
  result.downweighted_internal_point_count = 0;
  for (const InternalObservationWeightDecision& decision :
       result.point_decisions) {
    if (decision.downweighted) {
      ++result.downweighted_internal_point_count;
    }
  }
  result.downweighted_ratio =
      SafeRatio(result.downweighted_internal_point_count,
                result.input_internal_point_count);
  if (!weights.empty()) {
    result.min_weight =
        *std::min_element(weights.begin(), weights.end());
    result.max_weight =
        *std::max_element(weights.begin(), weights.end());
    result.mean_weight = ComputeMean(weights);
  }

  for (const auto& entry : board_to_decisions) {
    const std::vector<std::size_t>& indices = entry.second;
    InternalObservationWeightBoardSummary summary;
    summary.frame_index = entry.first.first;
    summary.board_id = entry.first.second;
    summary.internal_point_count = static_cast<int>(indices.size());
    summary.min_weight = std::numeric_limits<double>::infinity();
    summary.max_weight = 0.0;
    double weight_sum = 0.0;
    double quality_sum = 0.0;
    double internal_rmse_sum = 0.0;
    double outer_rmse_sum = 0.0;
    double consistency_ratio_sum = 0.0;
    for (std::size_t index : indices) {
      const InternalObservationWeightDecision& decision =
          result.point_decisions[index];
      summary.frame_label = decision.frame_label;
      if (decision.downweighted) {
        ++summary.downweighted_internal_point_count;
      }
      summary.min_weight =
          std::min(summary.min_weight, decision.output_weight);
      summary.max_weight =
          std::max(summary.max_weight, decision.output_weight);
      weight_sum += decision.output_weight;
      quality_sum += decision.quality;
      internal_rmse_sum += decision.board_internal_rmse;
      outer_rmse_sum += decision.board_outer_rmse;
      consistency_ratio_sum += decision.residual_consistency_ratio;
    }
    if (summary.internal_point_count > 0) {
      summary.mean_weight =
          weight_sum / static_cast<double>(summary.internal_point_count);
      summary.mean_quality =
          quality_sum / static_cast<double>(summary.internal_point_count);
      summary.board_internal_rmse =
          internal_rmse_sum / static_cast<double>(summary.internal_point_count);
      summary.board_outer_rmse =
          outer_rmse_sum / static_cast<double>(summary.internal_point_count);
      summary.residual_consistency_ratio =
          consistency_ratio_sum /
          static_cast<double>(summary.internal_point_count);
    } else {
      summary.min_weight = 1.0;
      summary.mean_weight = 1.0;
      summary.max_weight = 1.0;
    }
    result.board_summaries.push_back(summary);
  }

  if (options.mode == InternalObservationWeightMode::Enabled) {
    ApplyObservationWeightsToDataset(
        output_weights, &result.curated_bundle.measurement_dataset);
    const JointReprojectionSceneState scene_state =
        BuildJointSceneStateFromCalibrationSceneState(
            result.curated_bundle.scene_state);
    const JointMeasurementBuildResult curated_measurement_result =
        BuildMeasurementResultFromDataset(
            result.curated_bundle.measurement_dataset,
            result.curated_bundle.scene_state.reference_board_id);
    JointResidualEvaluationOptions residual_options;
    residual_options.top_k = -1;
    const JointReprojectionResidualEvaluator residual_evaluator(
        residual_options);
    const JointResidualEvaluationResult curated_residual_result =
        residual_evaluator.Evaluate(curated_measurement_result, scene_state);
    if (!curated_residual_result.success) {
      result.failure_reason = curated_residual_result.failure_reason;
      result.warnings = curated_residual_result.warnings;
      return result;
    }
    result.curated_bundle.residual_result = curated_residual_result;
    std::ostringstream warning;
    warning << "pre-backend internal observation weighting downweighted "
            << result.downweighted_internal_point_count << " / "
            << result.input_internal_point_count << " internal points"
            << " policy=" << result.policy
            << " q=" << result.options.low_quality_quantile
            << " threshold=" << result.quality_threshold
            << " residual_consistency_ratio_threshold="
            << result.residual_consistency_ratio_threshold
            << " min_weight=" << result.min_weight
            << " mean_weight=" << result.mean_weight;
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
  output << "threshold_mode: "
         << ToString(result.options.threshold_mode) << "\n";
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
      << "board_std_residual,board_median_residual,board_mad_residual,"
      << "board_threshold,sigma_threshold,threshold_mode,"
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
           << point.board_median_residual << ","
           << point.board_mad_residual << ","
           << point.board_threshold << ","
           << result.options.sigma_threshold << ","
           << ToString(result.options.threshold_mode) << ","
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
      << "std_residual,median_residual,mad_residual,threshold,"
      << "sigma_threshold,threshold_mode,min_abs_threshold_px\n";
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
           << board.median_residual << ","
           << board.mad_residual << ","
           << board.threshold << ","
           << result.options.sigma_threshold << ","
           << ToString(result.options.threshold_mode) << ","
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

void WriteInternalObservationWeightSummary(
    const std::string& path,
    const InternalObservationWeightResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "mode: " << ToString(result.options.mode) << "\n";
  output << "policy: " << result.policy << "\n";
  output << "diagnostic_only: " << (result.diagnostic_only ? 1 : 0)
         << "\n";
  output << "backend_input_changed: "
         << (result.backend_input_changed ? 1 : 0) << "\n";
  output << "low_quality_quantile: " << result.options.low_quality_quantile
         << "\n";
  output << "quality_threshold: " << result.quality_threshold << "\n";
  output << "min_weight_option: " << result.options.min_weight << "\n";
  output << "quality_exponent: " << result.options.quality_exponent << "\n";
  output << "residual_consistency_sigma_multiplier: "
         << result.residual_consistency_sigma_multiplier << "\n";
  output << "residual_consistency_min_rmse: "
         << result.residual_consistency_min_rmse << "\n";
  output << "residual_consistency_ratio_threshold: "
         << result.residual_consistency_ratio_threshold << "\n";
  output << "input_internal_point_count: "
         << result.input_internal_point_count << "\n";
  output << "downweighted_internal_point_count: "
         << result.downweighted_internal_point_count << "\n";
  output << "downweighted_ratio: " << result.downweighted_ratio << "\n";
  output << "min_weight: " << result.min_weight << "\n";
  output << "mean_weight: " << result.mean_weight << "\n";
  output << "max_weight: " << result.max_weight << "\n";
  output << "board_summary_count: " << result.board_summaries.size()
         << "\n";
  output << "point_decision_count: " << result.point_decisions.size()
         << "\n";
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteInternalObservationWeightsCsv(
    const std::string& path,
    const InternalObservationWeightResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_id,point_id,source_kind,source_point_index,quality,"
      << "board_internal_rmse,board_outer_rmse,residual_consistency_ratio,"
      << "quality_threshold,residual_consistency_ratio_threshold,"
      << "input_weight,output_weight,downweighted,weight_reason\n";
  for (const InternalObservationWeightDecision& point :
       result.point_decisions) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << point.frame_index << ","
           << CsvEscape(point.frame_label) << ","
           << point.board_id << ","
           << point.point_id << ","
           << ToString(point.source_kind) << ","
           << point.source_point_index << ","
           << point.quality << ","
           << point.board_internal_rmse << ","
           << point.board_outer_rmse << ","
           << point.residual_consistency_ratio << ","
           << result.quality_threshold << ","
           << result.residual_consistency_ratio_threshold << ","
           << point.input_weight << ","
           << point.output_weight << ","
           << (point.downweighted ? 1 : 0) << ","
           << CsvEscape(point.weight_reason) << "\n";
  }
}

void WriteInternalObservationWeightBoardSummaryCsv(
    const std::string& path,
    const InternalObservationWeightResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_id,internal_point_count,downweighted_internal_point_count,"
      << "quality_threshold,residual_consistency_ratio_threshold,"
      << "min_weight,mean_weight,max_weight,mean_quality,"
      << "board_internal_rmse,board_outer_rmse,residual_consistency_ratio\n";
  for (const InternalObservationWeightBoardSummary& board :
       result.board_summaries) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << board.frame_index << ","
           << CsvEscape(board.frame_label) << ","
           << board.board_id << ","
           << board.internal_point_count << ","
           << board.downweighted_internal_point_count << ","
           << result.quality_threshold << ","
           << result.residual_consistency_ratio_threshold << ","
           << board.min_weight << ","
           << board.mean_weight << ","
           << board.max_weight << ","
           << board.mean_quality << ","
           << board.board_internal_rmse << ","
           << board.board_outer_rmse << ","
           << board.residual_consistency_ratio << "\n";
  }
}

void WriteInternalBlurBoardWeightSummary(
    const std::string& path,
    const InternalBlurBoardWeightResult& result) {
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
  output << "min_weight_option: " << result.options.min_weight << "\n";
  output << "gradient_exponent: " << result.options.gradient_exponent << "\n";
  output << "input_board_observation_count: "
         << result.input_board_observation_count << "\n";
  output << "downweighted_board_observation_count: "
         << result.downweighted_board_observation_count << "\n";
  output << "input_internal_point_count: "
         << result.input_internal_point_count << "\n";
  output << "downweighted_internal_point_count: "
         << result.downweighted_internal_point_count << "\n";
  output << "downweighted_internal_ratio: "
         << result.downweighted_internal_ratio << "\n";
  output << "min_weight: " << result.min_weight << "\n";
  output << "mean_weight: " << result.mean_weight << "\n";
  output << "max_weight: " << result.max_weight << "\n";
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteInternalBlurBoardWeightPointsCsv(
    const std::string& path,
    const InternalBlurBoardWeightResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_id,point_id,source_kind,source_point_index,"
      << "board_corner_patch_mean_gradient,board_patch_gradient_threshold,"
      << "board_internal_rmse,board_p95_residual,input_weight,output_weight,"
      << "downweighted,weight_reason\n";
  for (const InternalBlurBoardWeightPointDecision& point :
       result.point_decisions) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << point.frame_index << ","
           << CsvEscape(point.frame_label) << ","
           << point.board_id << ","
           << point.point_id << ","
           << ToString(point.source_kind) << ","
           << point.source_point_index << ","
           << point.board_corner_patch_mean_gradient << ","
           << point.board_patch_gradient_threshold << ","
           << point.board_internal_rmse << ","
           << point.board_p95_residual << ","
           << point.input_weight << ","
           << point.output_weight << ","
           << (point.downweighted ? 1 : 0) << ","
           << CsvEscape(point.weight_reason) << "\n";
  }
}

void WriteInternalBlurBoardWeightBoardSummaryCsv(
    const std::string& path,
    const InternalBlurBoardWeightResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_id,internal_point_count,downweighted_internal_point_count,"
      << "internal_rmse,p95_residual,corner_patch_mean_gradient,"
      << "patch_gradient_threshold,low_patch_gradient,high_internal_residual,"
      << "targeted_for_downweight,input_weight,output_weight,weight_reason\n";
  for (const InternalBlurBoardWeightBoardSummary& board :
       result.board_summaries) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << board.frame_index << ","
           << CsvEscape(board.frame_label) << ","
           << board.board_id << ","
           << board.internal_point_count << ","
           << board.downweighted_internal_point_count << ","
           << board.internal_rmse << ","
           << board.p95_residual << ","
           << board.corner_patch_mean_gradient << ","
           << board.patch_gradient_threshold << ","
           << (board.low_patch_gradient ? 1 : 0) << ","
           << (board.high_internal_residual ? 1 : 0) << ","
           << (board.targeted_for_downweight ? 1 : 0) << ","
           << board.input_weight << ","
           << board.output_weight << ","
           << CsvEscape(board.weight_reason) << "\n";
  }
}

void WriteInternalJointRefineSummary(
    const std::string& path,
    const InternalJointRefineResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "mode: " << ToString(result.options.mode) << "\n";
  output << "diagnostic_only: " << (result.diagnostic_only ? 1 : 0)
         << "\n";
  output << "backend_input_changed: "
         << (result.backend_input_changed ? 1 : 0) << "\n";
  output << "target_mode: " << ToString(result.options.target_mode) << "\n";
  output << "search_radius_px: " << result.options.search_radius_px << "\n";
  output << "max_displacement_px: " << result.options.max_displacement_px
         << "\n";
  output << "geometry_sigma_px: " << result.options.geometry_sigma_px << "\n";
  output << "observation_sigma_px: "
         << result.options.observation_sigma_px << "\n";
  output << "subpix_window_radius: "
         << result.options.subpix_window_radius << "\n";
  output << "min_objective_improvement: "
         << result.options.min_objective_improvement << "\n";
  output << "min_old_residual_px: " << result.options.min_old_residual_px
         << "\n";
  output << "low_patch_gradient_quantile: "
         << result.options.low_patch_gradient_quantile << "\n";
  output << "min_board_internal_rmse_px: "
         << result.options.min_board_internal_rmse_px << "\n";
  output << "min_board_p95_residual_px: "
         << result.options.min_board_p95_residual_px << "\n";
  output << "min_corner_response_gain: "
         << result.options.min_corner_response_gain << "\n";
  output << "min_board_internal_rmse_improvement_px: "
         << result.options.min_board_internal_rmse_improvement_px << "\n";
  output << "min_refined_point_count_per_board: "
         << result.options.min_refined_point_count_per_board << "\n";
  output << "accept_max_global_outer_delta_px: "
         << result.options.accept_max_global_outer_delta_px << "\n";
  output << "accept_max_frame_outer_delta_px: "
         << result.options.accept_max_frame_outer_delta_px << "\n";
  output << "acceptance_backend_max_iterations: "
         << result.options.acceptance_backend_max_iterations << "\n";
  output << "acceptance_backend_mode: "
         << "optimize_frame_poses+optimize_board_poses+fixed_intrinsics"
         << "\n";
  output << "candidate_board_count: " << result.candidate_board_count << "\n";
  output << "accepted_board_count: " << result.accepted_board_count << "\n";
  output << "rolled_back_board_count: " << result.rolled_back_board_count
         << "\n";
  output << "eligible_internal_point_count: "
         << result.eligible_internal_point_count << "\n";
  output << "eligible_ratio: " << result.eligible_ratio << "\n";
  output << "input_internal_point_count: "
         << result.input_internal_point_count << "\n";
  output << "refined_internal_point_count: "
         << result.refined_internal_point_count << "\n";
  output << "refined_ratio: " << result.refined_ratio << "\n";
  output << "mean_displacement_px: " << result.mean_displacement_px << "\n";
  output << "max_displacement_px_observed: " << result.max_displacement_px
         << "\n";
  output << "targeted_blur_bad_board_count: "
         << result.targeted_blur_bad_board_count << "\n";
  output << "patch_gradient_threshold: " << result.patch_gradient_threshold
         << "\n";
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteInternalJointRefinePointsCsv(
    const std::string& path,
    const InternalJointRefineResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_id,point_id,source_kind,source_point_index,observed_x,"
      << "observed_y,predicted_x,predicted_y,refined_x,refined_y,"
      << "old_residual_norm,new_residual_norm,old_corner_response,"
      << "new_corner_response,old_objective,new_objective,displacement_px,"
      << "corner_response_gain,tentative_refined,accepted_after_board_rollback,"
      << "board_internal_rmse,board_p95_residual,"
      << "board_corner_patch_mean_gradient,board_patch_gradient_threshold,"
      << "board_low_patch_gradient,board_high_internal_residual,"
      << "targeted_by_high_residual,targeted_by_blur_bad_board,"
      << "eligible_for_refine,refined,refine_reason\n";
  for (const InternalJointRefinePointDecision& point :
       result.point_decisions) {
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
           << point.refined_image_xy.x() << ","
           << point.refined_image_xy.y() << ","
           << point.old_residual_norm << ","
           << point.new_residual_norm << ","
           << point.old_corner_response << ","
           << point.new_corner_response << ","
           << point.old_objective << ","
           << point.new_objective << ","
           << point.displacement_px << ","
           << point.corner_response_gain << ","
           << (point.tentative_refined ? 1 : 0) << ","
           << (point.accepted_after_board_rollback ? 1 : 0) << ","
           << point.board_internal_rmse << ","
           << point.board_p95_residual << ","
           << point.board_corner_patch_mean_gradient << ","
           << point.board_patch_gradient_threshold << ","
           << (point.board_low_patch_gradient ? 1 : 0) << ","
           << (point.board_high_internal_residual ? 1 : 0) << ","
           << (point.targeted_by_high_residual ? 1 : 0) << ","
           << (point.targeted_by_blur_bad_board ? 1 : 0) << ","
           << (point.eligible_for_refine ? 1 : 0) << ","
           << (point.refined ? 1 : 0) << ","
           << CsvEscape(point.refine_reason) << "\n";
  }
}

void WriteInternalJointRefineBoardSummaryCsv(
    const std::string& path,
    const InternalJointRefineResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "board_id,input_internal_point_count,eligible_internal_point_count,"
      << "tentative_refined_point_count,accepted_refined_point_count,"
      << "corner_patch_mean_gradient,patch_gradient_threshold,"
      << "board_internal_rmse_before,board_internal_rmse_after_tentative,"
      << "board_internal_rmse_improvement,board_p95_residual_before,"
      << "global_outer_only_rmse_before,global_outer_only_rmse_after,"
      << "global_outer_only_rmse_delta,frame_outer_only_rmse_before,"
      << "frame_outer_only_rmse_after,frame_outer_only_rmse_delta,"
      << "mean_corner_response_gain,low_patch_gradient,high_internal_residual,"
      << "targeted_for_refine,accepted,rolled_back,rollback_reason\n";
  for (const InternalJointRefineBoardSummary& board : result.board_summaries) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << board.frame_index << ","
           << CsvEscape(board.frame_label) << ","
           << board.board_id << ","
           << board.input_internal_point_count << ","
           << board.eligible_internal_point_count << ","
           << board.tentative_refined_point_count << ","
           << board.accepted_refined_point_count << ","
           << board.corner_patch_mean_gradient << ","
           << board.patch_gradient_threshold << ","
           << board.board_internal_rmse_before << ","
           << board.board_internal_rmse_after_tentative << ","
           << board.board_internal_rmse_improvement << ","
           << board.board_p95_residual_before << ","
           << board.global_outer_only_rmse_before << ","
           << board.global_outer_only_rmse_after << ","
           << board.global_outer_only_rmse_delta << ","
           << board.frame_outer_only_rmse_before << ","
           << board.frame_outer_only_rmse_after << ","
           << board.frame_outer_only_rmse_delta << ","
           << board.mean_corner_response_gain << ","
           << (board.low_patch_gradient ? 1 : 0) << ","
           << (board.high_internal_residual ? 1 : 0) << ","
           << (board.targeted_for_refine ? 1 : 0) << ","
           << (board.accepted ? 1 : 0) << ","
           << (board.rolled_back ? 1 : 0) << ","
           << CsvEscape(board.rollback_reason) << "\n";
  }
}

void WriteInternalJointRefineFrameSummaryCsv(
    const std::string& path,
    const InternalJointRefineResult& result) {
  std::ofstream output(path.c_str());
  output
      << "mode,diagnostic_only,backend_input_changed,frame_index,frame_label,"
      << "input_internal_point_count,eligible_internal_point_count,"
      << "refined_internal_point_count,accepted_board_count,"
      << "rolled_back_board_count,eligible_ratio,refined_ratio,"
      << "mean_displacement_px,max_displacement_px,mean_residual_before,"
      << "mean_residual_after\n";
  for (const InternalJointRefineFrameSummary& frame :
       result.frame_summaries) {
    output << ToString(result.options.mode) << ","
           << (result.diagnostic_only ? 1 : 0) << ","
           << (result.backend_input_changed ? 1 : 0) << ","
           << frame.frame_index << ","
           << CsvEscape(frame.frame_label) << ","
           << frame.input_internal_point_count << ","
           << frame.eligible_internal_point_count << ","
           << frame.refined_internal_point_count << ","
           << frame.accepted_board_count << ","
           << frame.rolled_back_board_count << ","
           << frame.eligible_ratio << ","
           << frame.refined_ratio << ","
           << frame.mean_displacement_px << ","
           << frame.max_displacement_px << ","
           << frame.mean_residual_before << ","
           << frame.mean_residual_after << "\n";
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
