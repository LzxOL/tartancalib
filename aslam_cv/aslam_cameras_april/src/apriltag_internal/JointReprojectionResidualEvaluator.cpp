#include <aslam/cameras/apriltag_internal/JointReprojectionResidualEvaluator.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

double ComputeRmseFromSquaredSum(double squared_sum, int count) {
  if (count <= 0) {
    return 0.0;
  }
  return std::sqrt(squared_sum / static_cast<double>(count));
}

template <typename T>
std::vector<T> TopKByRmse(std::vector<T> values, int top_k) {
  std::sort(values.begin(), values.end(),
            [](const T& lhs, const T& rhs) { return lhs.rmse > rhs.rmse; });
  if (top_k >= 0 && static_cast<int>(values.size()) > top_k) {
    values.resize(static_cast<std::size_t>(top_k));
  }
  return values;
}

template <typename T>
std::vector<T> TopKByResidual(std::vector<T> values, int top_k) {
  std::sort(values.begin(), values.end(),
            [](const T& lhs, const T& rhs) { return lhs.residual_norm > rhs.residual_norm; });
  if (top_k >= 0 && static_cast<int>(values.size()) > top_k) {
    values.resize(static_cast<std::size_t>(top_k));
  }
  return values;
}

}  // namespace

JointReprojectionResidualEvaluator::JointReprojectionResidualEvaluator(
    JointResidualEvaluationOptions options)
    : options_(std::move(options)), cost_core_(options_.cost_options) {}

JointResidualEvaluationResult JointReprojectionResidualEvaluator::Evaluate(
    const JointMeasurementBuildResult& measurement_result,
    const JointReprojectionSceneState& scene_state) const {
  JointResidualEvaluationResult result;
  result.reference_board_id = scene_state.reference_board_id;

  const JointCostEvaluation cost_evaluation = cost_core_.Evaluate(measurement_result, scene_state);
  if (!cost_evaluation.success) {
    result.failure_reason = cost_evaluation.failure_reason;
    result.warnings = cost_evaluation.warnings;
    return result;
  }

  std::map<std::pair<int, int>, std::pair<double, int> > board_observation_accumulators;
  std::map<std::pair<int, int>, std::pair<int, std::string> > board_observation_labels;
  std::map<int, std::tuple<double, int, int, int> > board_accumulators;
  std::map<int, std::pair<double, int> > frame_accumulators;
  std::map<int, std::vector<int> > frame_visible_boards;

  result.point_diagnostics.reserve(cost_evaluation.point_evaluations.size());
  for (const JointCostPointEvaluation& point_eval : cost_evaluation.point_evaluations) {
    JointResidualPointDiagnostics diagnostics;
    diagnostics.frame_index = point_eval.frame_index;
    diagnostics.frame_label = point_eval.frame_label;
    diagnostics.board_id = point_eval.board_id;
    diagnostics.point_id = point_eval.point_id;
    diagnostics.point_type = point_eval.point_type;
    diagnostics.observed_image_xy = point_eval.observed_image_xy;
    diagnostics.predicted_image_xy = point_eval.predicted_image_xy;
    diagnostics.target_xyz_board = point_eval.target_xyz_board;
    diagnostics.residual_xy = point_eval.residual_xy;
    diagnostics.residual_norm = point_eval.residual_norm;
    diagnostics.quality = point_eval.quality;
    diagnostics.used_in_solver = point_eval.used_in_solver;
    diagnostics.frame_storage_index = point_eval.frame_storage_index;
    diagnostics.source_board_observation_index = point_eval.source_board_observation_index;
    diagnostics.source_point_index = point_eval.source_point_index;
    diagnostics.source_kind = point_eval.source_kind;
    result.point_diagnostics.push_back(diagnostics);

    const double squared_norm = diagnostics.residual_xy.squaredNorm();
    const std::pair<int, int> board_observation_key(
        diagnostics.frame_index, diagnostics.board_id);
    board_observation_accumulators[board_observation_key].first += squared_norm;
    board_observation_accumulators[board_observation_key].second += 1;
    board_observation_labels[board_observation_key] =
        std::make_pair(diagnostics.board_id, diagnostics.frame_label);

    std::tuple<double, int, int, int>& board_accumulator =
        board_accumulators[diagnostics.board_id];
    std::get<0>(board_accumulator) += squared_norm;
    std::get<1>(board_accumulator) += 1;
    if (diagnostics.point_type == JointPointType::Outer) {
      std::get<2>(board_accumulator) += 1;
    } else {
      std::get<3>(board_accumulator) += 1;
    }

    frame_accumulators[diagnostics.frame_index].first += squared_norm;
    frame_accumulators[diagnostics.frame_index].second += 1;
    std::vector<int>& visible_board_ids = frame_visible_boards[diagnostics.frame_index];
    if (std::find(visible_board_ids.begin(), visible_board_ids.end(),
                  diagnostics.board_id) == visible_board_ids.end()) {
      visible_board_ids.push_back(diagnostics.board_id);
    }
  }

  result.overall_rmse = cost_evaluation.overall_rmse;
  result.outer_only_rmse = cost_evaluation.outer_rmse;
  result.internal_only_rmse = cost_evaluation.internal_rmse;
  result.warnings = cost_evaluation.warnings;

  result.board_observation_diagnostics.reserve(board_observation_accumulators.size());
  for (const auto& entry : board_observation_accumulators) {
    const std::pair<int, int>& key = entry.first;
    const std::pair<double, int>& accumulator = entry.second;
    JointResidualBoardObservationDiagnostics diagnostics;
    diagnostics.frame_index = key.first;
    diagnostics.board_id = key.second;
    diagnostics.frame_label = board_observation_labels[key].second;
    diagnostics.point_count = accumulator.second;
    diagnostics.rmse = ComputeRmseFromSquaredSum(accumulator.first, accumulator.second);
    for (const JointResidualPointDiagnostics& point : result.point_diagnostics) {
      if (point.frame_index == diagnostics.frame_index &&
          point.board_id == diagnostics.board_id) {
        if (point.point_type == JointPointType::Outer) {
          ++diagnostics.outer_point_count;
        } else {
          ++diagnostics.internal_point_count;
        }
      }
    }
    result.board_observation_diagnostics.push_back(diagnostics);
  }

  result.board_diagnostics.reserve(board_accumulators.size());
  for (const auto& entry : board_accumulators) {
    JointResidualBoardDiagnostics diagnostics;
    diagnostics.board_id = entry.first;
    diagnostics.point_count = std::get<1>(entry.second);
    diagnostics.observation_count = 0;
    diagnostics.outer_point_count = std::get<2>(entry.second);
    diagnostics.internal_point_count = std::get<3>(entry.second);
    diagnostics.rmse = ComputeRmseFromSquaredSum(
        std::get<0>(entry.second), std::get<1>(entry.second));
    for (const JointResidualBoardObservationDiagnostics& board_observation :
         result.board_observation_diagnostics) {
      if (board_observation.board_id == diagnostics.board_id) {
        ++diagnostics.observation_count;
      }
    }
    result.board_diagnostics.push_back(diagnostics);
  }

  result.frame_diagnostics.reserve(frame_accumulators.size());
  for (const auto& entry : frame_accumulators) {
    JointResidualFrameDiagnostics diagnostics;
    diagnostics.frame_index = entry.first;
    diagnostics.point_count = entry.second.second;
    diagnostics.rmse = ComputeRmseFromSquaredSum(entry.second.first, entry.second.second);
    const JointSceneFrameState* frame_state = FindJointSceneFrameState(scene_state, entry.first);
    if (frame_state != nullptr) {
      diagnostics.frame_label = frame_state->frame_label;
    }
    diagnostics.visible_board_ids = frame_visible_boards[entry.first];
    for (const JointResidualPointDiagnostics& point : result.point_diagnostics) {
      if (point.frame_index == diagnostics.frame_index) {
        if (point.point_type == JointPointType::Outer) {
          ++diagnostics.outer_point_count;
        } else {
          ++diagnostics.internal_point_count;
        }
      }
    }
    result.frame_diagnostics.push_back(diagnostics);
  }

  result.worst_points = TopKByResidual(result.point_diagnostics, options_.top_k);
  result.worst_board_observations =
      TopKByRmse(result.board_observation_diagnostics, options_.top_k);
  result.worst_boards = TopKByRmse(result.board_diagnostics, options_.top_k);
  result.worst_frames = TopKByRmse(result.frame_diagnostics, options_.top_k);
  result.success = true;
  return result;
}

JointResidualEvaluationResult JointReprojectionResidualEvaluator::Evaluate(
    const JointMeasurementBuildResult& measurement_result,
    const OuterBootstrapResult& bootstrap_result) const {
  if (!bootstrap_result.success) {
    JointResidualEvaluationResult result;
    result.failure_reason = "bootstrap_result.success is false";
    return result;
  }
  return Evaluate(measurement_result, BuildSceneStateFromBootstrap(bootstrap_result));
}

void JointReprojectionResidualEvaluator::DrawFrameOverlay(
    const cv::Mat& image,
    int frame_index,
    const JointResidualEvaluationResult& evaluation_result,
    cv::Mat* output_image) const {
  if (output_image == nullptr) {
    throw std::runtime_error("DrawFrameOverlay requires a valid output pointer.");
  }
  *output_image = image.clone();

  int point_count = 0;
  for (const JointResidualPointDiagnostics& diagnostics : evaluation_result.point_diagnostics) {
    if (diagnostics.frame_index != frame_index) {
      continue;
    }
    ++point_count;
    const cv::Point observed(static_cast<int>(std::lround(diagnostics.observed_image_xy.x())),
                             static_cast<int>(std::lround(diagnostics.observed_image_xy.y())));
    const cv::Point predicted(static_cast<int>(std::lround(diagnostics.predicted_image_xy.x())),
                              static_cast<int>(std::lround(diagnostics.predicted_image_xy.y())));
    const cv::Scalar observed_color =
        diagnostics.point_type == JointPointType::Outer ? cv::Scalar(60, 220, 80)
                                                        : cv::Scalar(40, 180, 255);
    cv::circle(*output_image, observed, diagnostics.point_type == JointPointType::Outer ? 4 : 3,
               observed_color, 2, cv::LINE_AA);
    cv::drawMarker(*output_image, predicted, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 8, 1,
                   cv::LINE_AA);
    cv::line(*output_image, observed, predicted, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
  }

  const JointResidualFrameDiagnostics* frame_diagnostics =
      FindFrameDiagnostics(evaluation_result, frame_index);
  const int banner_height = 76;
  cv::rectangle(*output_image, cv::Rect(0, 0, output_image->cols, banner_height),
                cv::Scalar(18, 18, 18), cv::FILLED);

  std::ostringstream header;
  header << "frame " << frame_index << " residual_points=" << point_count;
  if (frame_diagnostics != nullptr) {
    header << " frame_rmse=" << frame_diagnostics->rmse
           << " outer=" << frame_diagnostics->outer_point_count
           << " internal=" << frame_diagnostics->internal_point_count;
  }
  cv::putText(*output_image, header.str(), cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.62,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

  std::ostringstream summary;
  summary << "overall_rmse=" << evaluation_result.overall_rmse
          << " outer_rmse=" << evaluation_result.outer_only_rmse
          << " internal_rmse=" << evaluation_result.internal_only_rmse;
  cv::putText(*output_image, summary.str(), cv::Point(18, 54), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(185, 185, 185), 1, cv::LINE_AA);
}

const JointResidualFrameDiagnostics* JointReprojectionResidualEvaluator::FindFrameDiagnostics(
    const JointResidualEvaluationResult& evaluation_result,
    int frame_index) const {
  for (const JointResidualFrameDiagnostics& diagnostics :
       evaluation_result.frame_diagnostics) {
    if (diagnostics.frame_index == frame_index) {
      return &diagnostics;
    }
  }
  return nullptr;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
