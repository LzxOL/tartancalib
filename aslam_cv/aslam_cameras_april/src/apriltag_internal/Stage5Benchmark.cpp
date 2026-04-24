#include <aslam/cameras/apriltag_internal/Stage5Benchmark.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagCanonicalModel.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

constexpr double kInvalidProjectionPenaltyPixels = 100.0;
constexpr double kKalibrCornerSigmaThreshold = 2.0;
constexpr double kKalibrCornerMinReprojErrorPixels = 0.2;

std::vector<int> NormalizeBoardIds(const std::vector<int>& configured_ids,
                                   int fallback_tag_id) {
  std::vector<int> board_ids;
  const auto append_if_valid = [&board_ids](int board_id) {
    if (board_id < 0) {
      return;
    }
    if (std::find(board_ids.begin(), board_ids.end(), board_id) == board_ids.end()) {
      board_ids.push_back(board_id);
    }
  };
  for (int board_id : configured_ids) {
    append_if_valid(board_id);
  }
  if (board_ids.empty()) {
    append_if_valid(fallback_tag_id);
  }
  return board_ids;
}

ApriltagInternalConfig NormalizeConfig(ApriltagInternalConfig config) {
  config.tag_ids = NormalizeBoardIds(config.tag_ids, config.tag_id);
  if (!config.tag_ids.empty()) {
    config.tag_id = config.tag_ids.front();
  }
  config.outer_detector_config.tag_ids = config.tag_ids;
  config.outer_detector_config.tag_id = config.tag_id;
  return config;
}

ApriltagInternalConfig BoardConfigForId(const ApriltagInternalConfig& config, int board_id) {
  ApriltagInternalConfig board_config = config;
  board_config.tag_id = board_id;
  board_config.tag_ids = {board_id};
  board_config.outer_detector_config.tag_id = board_id;
  board_config.outer_detector_config.tag_ids = {board_id};
  return board_config;
}

ApriltagInternalDetectionOptions MakeDetectionOptions(
    const ApriltagInternalConfig& config) {
  ApriltagInternalDetectionOptions options;
  options.do_subpix_refinement = true;
  options.max_subpix_displacement2 = config.max_subpix_displacement2;
  options.min_border_distance = 4.0;
  options.canonical_pixels_per_module = config.canonical_pixels_per_module;
  options.refinement_window_radius = config.refinement_window_radius;
  options.internal_subpix_window_scale = config.internal_subpix_window_scale;
  options.internal_subpix_window_min = config.internal_subpix_window_min;
  options.internal_subpix_window_max = config.internal_subpix_window_max;
  options.internal_subpix_displacement_scale = config.internal_subpix_displacement_scale;
  options.max_internal_subpix_displacement = config.max_internal_subpix_displacement;
  options.outer_detector_config = config.outer_detector_config;
  return options;
}

std::string JoinIndices(const std::vector<FrozenRound2BaselineFrameSource>& frames) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < frames.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << frames[index].frame_index;
  }
  return stream.str();
}

void AppendVisibleBoardId(int board_id, std::vector<int>* visible_board_ids) {
  if (visible_board_ids == nullptr || board_id < 0) {
    return;
  }
  if (std::find(visible_board_ids->begin(), visible_board_ids->end(), board_id) ==
      visible_board_ids->end()) {
    visible_board_ids->push_back(board_id);
  }
}

std::array<Eigen::Vector3d, 4> BuildOuterCornerTargets(const ApriltagInternalConfig& config,
                                                       int board_id) {
  const ApriltagCanonicalModel model(BoardConfigForId(config, board_id));
  const std::array<int, 4> point_ids{
      model.PointId(0, 0),
      model.PointId(model.ModuleDimension(), 0),
      model.PointId(model.ModuleDimension(), model.ModuleDimension()),
      model.PointId(0, model.ModuleDimension()),
  };
  std::array<Eigen::Vector3d, 4> points{};
  for (int index = 0; index < 4; ++index) {
    points[static_cast<std::size_t>(index)] =
        model.corner(point_ids[static_cast<std::size_t>(index)]).target_xyz;
  }
  return points;
}

bool IsOuterPoint(const CalibrationEvaluationPointObservation& point) {
  return point.point_type == JointPointType::Outer;
}

bool IsInternalPoint(const CalibrationEvaluationPointObservation& point) {
  return point.point_type == JointPointType::Internal;
}

cv::Rect ClampRectToImage(const cv::Rect& rect, const cv::Size& image_size) {
  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  return rect & image_rect;
}

const CameraModelRefitFrameDiagnostics* FindFrameDiagnostics(
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index) {
  for (const CameraModelRefitFrameDiagnostics& frame : evaluation.frame_diagnostics) {
    if (frame.frame_index == frame_index) {
      return &frame;
    }
  }
  return nullptr;
}

const CameraModelRefitBoardObservationDiagnostics* FindBoardDiagnostics(
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index,
    int board_id) {
  for (const CameraModelRefitBoardObservationDiagnostics& board :
       evaluation.board_observation_diagnostics) {
    if (board.frame_index == frame_index && board.board_id == board_id) {
      return &board;
    }
  }
  return nullptr;
}

double ComputeOuterRmseForFrame(const CameraModelRefitEvaluationResult& evaluation,
                                int frame_index) {
  double squared_error_sum = 0.0;
  int point_count = 0;
  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index || point.point_type != JointPointType::Outer) {
      continue;
    }
    squared_error_sum += point.residual_xy.squaredNorm();
    ++point_count;
  }
  if (point_count <= 0) {
    return 0.0;
  }
  return std::sqrt(squared_error_sum / static_cast<double>(point_count));
}

bool IsFiniteImagePoint(const Eigen::Vector2d& point) {
  return std::isfinite(point.x()) && std::isfinite(point.y());
}

void DrawObservedPredictedPoint(cv::Mat* image,
                                const CameraModelRefitPointDiagnostics& point,
                                const cv::Scalar& observed_color,
                                int radius,
                                bool annotate_point_id) {
  if (image == nullptr || !IsFiniteImagePoint(point.observed_image_xy) ||
      !IsFiniteImagePoint(point.predicted_image_xy)) {
    return;
  }

  const cv::Point observed(static_cast<int>(std::lround(point.observed_image_xy.x())),
                           static_cast<int>(std::lround(point.observed_image_xy.y())));
  const cv::Point predicted(static_cast<int>(std::lround(point.predicted_image_xy.x())),
                            static_cast<int>(std::lround(point.predicted_image_xy.y())));
  cv::circle(*image, observed, radius, observed_color, 2, cv::LINE_AA);
  cv::drawMarker(*image, predicted, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10, 1,
                 cv::LINE_AA);
  cv::line(*image, observed, predicted, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
  if (annotate_point_id) {
    cv::putText(*image, std::to_string(point.point_id),
                observed + cv::Point(6, -6), cv::FONT_HERSHEY_PLAIN, 1.0,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  }
}

bool AccumulatePointBounds(const CameraModelRefitPointDiagnostics& point,
                           double* min_x,
                           double* min_y,
                           double* max_x,
                           double* max_y) {
  if (min_x == nullptr || min_y == nullptr || max_x == nullptr || max_y == nullptr ||
      !IsFiniteImagePoint(point.observed_image_xy) ||
      !IsFiniteImagePoint(point.predicted_image_xy)) {
    return false;
  }

  *min_x = std::min(*min_x, std::min(point.observed_image_xy.x(), point.predicted_image_xy.x()));
  *min_y = std::min(*min_y, std::min(point.observed_image_xy.y(), point.predicted_image_xy.y()));
  *max_x = std::max(*max_x, std::max(point.observed_image_xy.x(), point.predicted_image_xy.x()));
  *max_y = std::max(*max_y, std::max(point.observed_image_xy.y(), point.predicted_image_xy.y()));
  return true;
}

void FilterInternalEvaluationPointsByReprojection(
    const OuterBootstrapCameraIntrinsics& camera,
    CalibrationEvaluationBoardObservation* board_observation) {
  if (board_observation == nullptr || !camera.IsValid()) {
    return;
  }

  std::vector<Eigen::Vector3d> outer_targets;
  std::vector<cv::Point2f> outer_pixels;
  std::vector<std::size_t> internal_indices;
  for (std::size_t index = 0; index < board_observation->points.size(); ++index) {
    const CalibrationEvaluationPointObservation& point = board_observation->points[index];
    if (point.point_type == JointPointType::Outer) {
      outer_targets.push_back(point.target_xyz_board);
      outer_pixels.push_back(
          cv::Point2f(static_cast<float>(point.image_xy.x()),
                      static_cast<float>(point.image_xy.y())));
    } else {
      internal_indices.push_back(index);
    }
  }

  if (outer_targets.size() < 4 || internal_indices.empty()) {
    return;
  }

  Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
  double pose_fit_rmse = 0.0;
  if (!EstimatePoseFromObjectPoints(camera, outer_targets, outer_pixels,
                                    &T_camera_board, &pose_fit_rmse)) {
    return;
  }

  const DoubleSphereCameraModel camera_model =
      DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(camera));
  std::vector<double> residual_norms;
  residual_norms.reserve(internal_indices.size());
  std::vector<double> per_point_residuals(
      internal_indices.size(), std::numeric_limits<double>::infinity());
  for (std::size_t i = 0; i < internal_indices.size(); ++i) {
    const CalibrationEvaluationPointObservation& point =
        board_observation->points[internal_indices[i]];
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    if (!camera_model.vsEuclideanToKeypoint(T_camera_board * point.target_xyz_board,
                                            &predicted)) {
      continue;
    }
    per_point_residuals[i] = (predicted - point.image_xy).norm();
    residual_norms.push_back(per_point_residuals[i]);
  }

  if (residual_norms.empty()) {
    board_observation->points.erase(
        std::remove_if(board_observation->points.begin(), board_observation->points.end(),
                       [](const CalibrationEvaluationPointObservation& point) {
                         return point.point_type == JointPointType::Internal;
                       }),
        board_observation->points.end());
    board_observation->internal_point_count = 0;
    return;
  }

  const double mean_residual = std::accumulate(
      residual_norms.begin(), residual_norms.end(), 0.0) /
      static_cast<double>(residual_norms.size());
  double variance = 0.0;
  for (double residual : residual_norms) {
    const double delta = residual - mean_residual;
    variance += delta * delta;
  }
  variance /= static_cast<double>(residual_norms.size());
  const double std_residual = std::sqrt(std::max(0.0, variance));
  const double threshold =
      mean_residual + kKalibrCornerSigmaThreshold * std_residual;

  std::vector<CalibrationEvaluationPointObservation> filtered_points;
  filtered_points.reserve(board_observation->points.size());
  std::size_t internal_offset = 0;
  int filtered_internal_count = 0;
  for (const CalibrationEvaluationPointObservation& point : board_observation->points) {
    if (point.point_type == JointPointType::Outer) {
      filtered_points.push_back(point);
      continue;
    }

    const double residual = per_point_residuals[internal_offset++];
    const bool keep =
        std::isfinite(residual) &&
        !(residual > threshold && residual > kKalibrCornerMinReprojErrorPixels);
    if (keep) {
      filtered_points.push_back(point);
      ++filtered_internal_count;
    }
  }

  board_observation->points.swap(filtered_points);
  board_observation->internal_point_count = filtered_internal_count;
}

}  // namespace

Stage5Benchmark::Stage5Benchmark(CalibrationBenchmarkSplitOptions split_options)
    : split_options_(std::move(split_options)) {}

CalibrationBenchmarkSplit Stage5Benchmark::BuildDeterministicSplit(
    const std::vector<FrozenRound2BaselineFrameSource>& frames) const {
  CalibrationBenchmarkSplit split;
  split.mode = split_options_.mode;
  split.holdout_stride = split_options_.holdout_stride;
  split.holdout_offset = split_options_.holdout_offset;

  if (frames.empty()) {
    split.failure_reason = "Stage 5 benchmark split requires non-empty frame sources.";
    return split;
  }
  if (split_options_.holdout_stride <= 1) {
    split.failure_reason = "holdout_stride must be greater than 1 for deterministic split.";
    return split;
  }

  for (std::size_t index = 0; index < frames.size(); ++index) {
    const int normalized_offset =
        ((split_options_.holdout_offset % split_options_.holdout_stride) +
         split_options_.holdout_stride) %
        split_options_.holdout_stride;
    const bool is_holdout =
        static_cast<int>((index + split_options_.holdout_stride - normalized_offset) %
                         split_options_.holdout_stride) == 0;
    if (is_holdout) {
      split.holdout_frames.push_back(frames[index]);
    } else {
      split.training_frames.push_back(frames[index]);
    }
  }

  split.split_signature =
      "deterministic_stride_" + std::to_string(split.holdout_stride) +
      "_offset_" + std::to_string(split.holdout_offset) +
      "_holdout_indices_" + JoinIndices(split.holdout_frames);

  if (static_cast<int>(split.training_frames.size()) <
      split_options_.minimum_training_frames) {
    split.failure_reason = "Training split is too small for Stage 5 benchmark.";
    return split;
  }
  if (static_cast<int>(split.holdout_frames.size()) <
      split_options_.minimum_holdout_frames) {
    split.failure_reason = "Hold-out split is too small for Stage 5 benchmark.";
    return split;
  }

  split.success = true;
  return split;
}

CalibrationEvaluationDataset Stage5Benchmark::BuildTrainingEvaluationDataset(
    const CalibrationStateBundle& bundle) const {
  CalibrationEvaluationDataset dataset;
  dataset.dataset_label = bundle.scene_state.dataset_label;
  dataset.split_label = "training";
  dataset.split_signature = bundle.training_split_signature;

  for (const JointMeasurementFrameResult& frame_result :
       bundle.measurement_dataset.frames) {
    CalibrationEvaluationFrameInput frame_input;
    frame_input.frame_index = frame_result.frame_index;
    frame_input.frame_label = frame_result.frame_label;
    frame_input.visible_board_ids = frame_result.visible_board_ids;

    for (const JointBoardObservation& board_observation :
         frame_result.board_observations) {
      CalibrationEvaluationBoardObservation eval_board;
      eval_board.frame_index = frame_result.frame_index;
      eval_board.frame_label = frame_result.frame_label;
      eval_board.board_id = board_observation.board_id;

      for (const JointPointObservation& point : board_observation.points) {
        if (!point.used_in_solver) {
          continue;
        }
        CalibrationEvaluationPointObservation eval_point;
        eval_point.frame_index = point.frame_index;
        eval_point.frame_label = point.frame_label;
        eval_point.board_id = point.board_id;
        eval_point.point_id = point.point_id;
        eval_point.point_type = point.point_type;
        eval_point.image_xy = point.image_xy;
        eval_point.target_xyz_board = point.target_xyz_board;
        eval_point.quality = point.quality;
        eval_point.frame_storage_index = point.frame_storage_index;
        eval_point.source_board_observation_index = point.source_board_observation_index;
        eval_point.source_point_index = point.source_point_index;
        eval_point.source_kind = point.source_kind;
        eval_board.points.push_back(eval_point);
        if (eval_point.point_type == JointPointType::Outer) {
          ++eval_board.outer_point_count;
        } else {
          ++eval_board.internal_point_count;
        }
      }

      eval_board.has_pose_fit_outer_points = (eval_board.outer_point_count >= 4);
      if (!eval_board.points.empty()) {
        frame_input.board_observations.push_back(eval_board);
        ++dataset.board_observation_count;
        dataset.outer_point_count += eval_board.outer_point_count;
        dataset.internal_point_count += eval_board.internal_point_count;
      }
    }

    if (!frame_input.board_observations.empty()) {
      dataset.frames.push_back(frame_input);
    }
  }

  dataset.frame_count = static_cast<int>(dataset.frames.size());
  dataset.total_point_count = dataset.outer_point_count + dataset.internal_point_count;
  dataset.success = dataset.frame_count > 0 && dataset.board_observation_count > 0 &&
                    dataset.total_point_count > 0;
  if (!dataset.success) {
    dataset.failure_reason = "Training evaluation dataset is empty.";
  }
  return dataset;
}

CalibrationEvaluationDataset Stage5Benchmark::BuildHoldoutEvaluationDataset(
    const std::vector<FrozenRound2BaselineFrameSource>& holdout_frames,
    const FrozenRound2BaselineOptions& baseline_options,
    const JointReprojectionSceneState& optimized_scene_state,
    const std::string& split_signature) const {
  CalibrationEvaluationDataset dataset;
  dataset.dataset_label = baseline_options.dataset_label;
  dataset.split_label = "holdout";
  dataset.split_signature = split_signature;

  const ApriltagInternalConfig config = NormalizeConfig(baseline_options.config);
  const ApriltagInternalDetectionOptions detection_options = MakeDetectionOptions(config);
  const MultiScaleOuterTagDetector outer_detector(config.outer_detector_config);
  const MultiBoardInternalMeasurementRegenerator regenerator(config, detection_options);

  for (std::size_t frame_storage_index = 0; frame_storage_index < holdout_frames.size();
       ++frame_storage_index) {
    const FrozenRound2BaselineFrameSource& frame_source = holdout_frames[frame_storage_index];
    const cv::Mat image = cv::imread(frame_source.image_path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      dataset.warnings.push_back("Failed to read hold-out image: " + frame_source.image_path);
      continue;
    }

    const OuterTagMultiDetectionResult outer_detection = outer_detector.DetectMultiple(image);
    InternalRegenerationFrameInput regen_input;
    regen_input.frame_index = frame_source.frame_index;
    regen_input.frame_label = frame_source.frame_label;
    regen_input.outer_detections = outer_detection;
    const InternalRegenerationFrameResult regen_result =
        regenerator.RegenerateFrame(image, regen_input, optimized_scene_state);
    for (const std::string& warning : regen_result.warnings) {
      dataset.warnings.push_back(warning);
    }

    CalibrationEvaluationFrameInput frame_input;
    frame_input.frame_index = frame_source.frame_index;
    frame_input.frame_label = frame_source.frame_label;

    for (std::size_t board_obs_index = 0;
         board_obs_index < outer_detection.frame_measurements.board_measurements.size();
         ++board_obs_index) {
      const OuterBoardMeasurement& outer_measurement =
          outer_detection.frame_measurements.board_measurements[board_obs_index];
      CalibrationEvaluationBoardObservation eval_board;
      eval_board.frame_index = frame_source.frame_index;
      eval_board.frame_label = frame_source.frame_label;
      eval_board.board_id = outer_measurement.board_id;

      if (outer_measurement.success && outer_measurement.valid_refined_corner_count == 4) {
        const std::array<Eigen::Vector3d, 4> outer_targets =
            BuildOuterCornerTargets(config, outer_measurement.board_id);
        for (int corner_index = 0; corner_index < 4; ++corner_index) {
          if (!outer_measurement.refined_corner_valid[static_cast<std::size_t>(corner_index)]) {
            continue;
          }
          CalibrationEvaluationPointObservation point;
          point.frame_index = frame_source.frame_index;
          point.frame_label = frame_source.frame_label;
          point.board_id = outer_measurement.board_id;
          point.point_id = corner_index;
          point.point_type = JointPointType::Outer;
          point.image_xy =
              outer_measurement.refined_outer_corners_original_image[static_cast<std::size_t>(
                  corner_index)];
          point.target_xyz_board = outer_targets[static_cast<std::size_t>(corner_index)];
          point.quality = outer_measurement.detection_quality;
          point.frame_storage_index = static_cast<int>(frame_storage_index);
          point.source_board_observation_index = static_cast<int>(board_obs_index);
          point.source_point_index = corner_index;
          point.source_kind = JointObservationSourceKind::OuterMeasurement;
          eval_board.points.push_back(point);
          ++eval_board.outer_point_count;
        }
        eval_board.has_pose_fit_outer_points = (eval_board.outer_point_count == 4);
        AppendVisibleBoardId(outer_measurement.board_id, &frame_input.visible_board_ids);
      }

      const RegeneratedBoardMeasurement* regenerated_board = nullptr;
      for (const RegeneratedBoardMeasurement& measurement : regen_result.board_measurements) {
        if (measurement.board_id == outer_measurement.board_id) {
          regenerated_board = &measurement;
          break;
        }
      }
      if (regenerated_board != nullptr) {
        for (std::size_t corner_index = 0;
             corner_index < regenerated_board->detection.corners.size();
             ++corner_index) {
          const CornerMeasurement& corner =
              regenerated_board->detection.corners[corner_index];
          if (!corner.valid || corner.corner_type == CornerType::Outer) {
            continue;
          }
          CalibrationEvaluationPointObservation point;
          point.frame_index = frame_source.frame_index;
          point.frame_label = frame_source.frame_label;
          point.board_id = outer_measurement.board_id;
          point.point_id = corner.point_id;
          point.point_type = JointPointType::Internal;
          point.image_xy = corner.image_xy;
          point.target_xyz_board = corner.target_xyz;
          point.quality = corner.quality;
          point.frame_storage_index = static_cast<int>(frame_storage_index);
          point.source_board_observation_index = static_cast<int>(board_obs_index);
          point.source_point_index = static_cast<int>(corner_index);
          point.source_kind = JointObservationSourceKind::InternalMeasurement;
          eval_board.points.push_back(point);
          ++eval_board.internal_point_count;
        }
      }

      FilterInternalEvaluationPointsByReprojection(
          optimized_scene_state.camera, &eval_board);

      if (!eval_board.points.empty() && eval_board.has_pose_fit_outer_points) {
        frame_input.board_observations.push_back(eval_board);
        ++dataset.board_observation_count;
        dataset.outer_point_count += eval_board.outer_point_count;
        dataset.internal_point_count += eval_board.internal_point_count;
      }
    }

    if (!frame_input.board_observations.empty()) {
      dataset.frames.push_back(frame_input);
    }
  }

  dataset.frame_count = static_cast<int>(dataset.frames.size());
  dataset.total_point_count = dataset.outer_point_count + dataset.internal_point_count;
  dataset.success = dataset.frame_count > 0 && dataset.board_observation_count > 0 &&
                    dataset.total_point_count > 0;
  if (!dataset.success) {
    dataset.failure_reason = "Hold-out evaluation dataset is empty.";
  }
  return dataset;
}

CameraModelRefitEvaluationResult Stage5Benchmark::EvaluateCameraModel(
    const CalibrationEvaluationDataset& dataset,
    const OuterBootstrapCameraIntrinsics& camera,
    const std::string& method_label) const {
  CameraModelRefitEvaluationResult result;
  result.method_label = method_label;
  result.split_label = dataset.split_label;
  result.split_signature = dataset.split_signature;
  result.camera = camera;

  if (!dataset.success) {
    result.failure_reason =
        "EvaluateCameraModel requires a successful evaluation dataset.";
    return result;
  }
  if (!camera.IsValid()) {
    result.failure_reason = "EvaluateCameraModel requires valid DS intrinsics.";
    return result;
  }

  const DoubleSphereCameraModel camera_model =
      DoubleSphereCameraModel::FromConfig(MakeIntermediateCameraConfig(camera));
  double total_squared_error = 0.0;
  double outer_squared_error = 0.0;
  double internal_squared_error = 0.0;
  int total_point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;

  for (const CalibrationEvaluationFrameInput& frame : dataset.frames) {
    double frame_squared_error = 0.0;
    int frame_point_count = 0;
    int frame_outer_count = 0;
    int frame_internal_count = 0;

    for (const CalibrationEvaluationBoardObservation& board : frame.board_observations) {
      std::vector<Eigen::Vector3d> outer_targets;
      std::vector<cv::Point2f> outer_pixels;
      for (const CalibrationEvaluationPointObservation& point : board.points) {
        if (IsOuterPoint(point)) {
          outer_targets.push_back(point.target_xyz_board);
          outer_pixels.push_back(
              cv::Point2f(static_cast<float>(point.image_xy.x()),
                          static_cast<float>(point.image_xy.y())));
        }
      }
      if (outer_targets.size() < 4) {
        result.warnings.push_back(
            "Skipped board observation without four outer pose-fit points: frame=" +
            frame.frame_label + " board=" + std::to_string(board.board_id));
        continue;
      }

      Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
      double pose_fit_outer_rmse = 0.0;
      if (!EstimatePoseFromObjectPoints(camera, outer_targets, outer_pixels,
                                        &T_camera_board, &pose_fit_outer_rmse)) {
        result.warnings.push_back(
            "Outer-only pose refit failed: frame=" + frame.frame_label + " board=" +
            std::to_string(board.board_id));
        continue;
      }

      CameraModelRefitBoardObservationDiagnostics board_diag;
      board_diag.method_label = method_label;
      board_diag.split_label = dataset.split_label;
      board_diag.frame_index = frame.frame_index;
      board_diag.frame_label = frame.frame_label;
      board_diag.board_id = board.board_id;
      board_diag.pose_fit_outer_rmse = pose_fit_outer_rmse;

      double board_squared_error = 0.0;
      int board_point_count = 0;
      for (const CalibrationEvaluationPointObservation& point : board.points) {
        CameraModelRefitPointDiagnostics point_diag;
        point_diag.method_label = method_label;
        point_diag.split_label = dataset.split_label;
        point_diag.frame_index = point.frame_index;
        point_diag.frame_label = point.frame_label;
        point_diag.board_id = point.board_id;
        point_diag.point_id = point.point_id;
        point_diag.point_type = point.point_type;
        point_diag.observed_image_xy = point.image_xy;
        point_diag.target_xyz_board = point.target_xyz_board;
        point_diag.quality = point.quality;
        point_diag.frame_storage_index = point.frame_storage_index;
        point_diag.source_board_observation_index = point.source_board_observation_index;
        point_diag.source_point_index = point.source_point_index;
        point_diag.source_kind = point.source_kind;

        Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
        double squared_error = 0.0;
        if (!camera_model.vsEuclideanToKeypoint(T_camera_board * point.target_xyz_board,
                                                &predicted)) {
          point_diag.predicted_image_xy = Eigen::Vector2d::Constant(
              std::numeric_limits<double>::quiet_NaN());
          point_diag.residual_xy = Eigen::Vector2d::Constant(kInvalidProjectionPenaltyPixels);
          point_diag.residual_norm = std::sqrt(2.0) * kInvalidProjectionPenaltyPixels;
          squared_error = 2.0 * kInvalidProjectionPenaltyPixels *
                          kInvalidProjectionPenaltyPixels;
        } else {
          point_diag.predicted_image_xy = predicted;
          point_diag.residual_xy = predicted - point.image_xy;
          point_diag.residual_norm = point_diag.residual_xy.norm();
          squared_error = point_diag.residual_xy.squaredNorm();
        }

        result.point_diagnostics.push_back(point_diag);
        board_squared_error += squared_error;
        ++board_point_count;
        if (point.point_type == JointPointType::Outer) {
          outer_squared_error += squared_error;
          ++outer_point_count;
          ++board_diag.outer_point_count;
          ++frame_outer_count;
        } else {
          internal_squared_error += squared_error;
          ++internal_point_count;
          ++board_diag.internal_point_count;
          ++frame_internal_count;
        }
      }

      if (board_point_count <= 0) {
        continue;
      }
      board_diag.point_count = board_point_count;
      board_diag.evaluation_rmse =
          std::sqrt(board_squared_error / static_cast<double>(board_point_count));
      result.board_observation_diagnostics.push_back(board_diag);
      total_squared_error += board_squared_error;
      total_point_count += board_point_count;
      frame_squared_error += board_squared_error;
      frame_point_count += board_point_count;
    }

    if (frame_point_count > 0) {
      CameraModelRefitFrameDiagnostics frame_diag;
      frame_diag.method_label = method_label;
      frame_diag.split_label = dataset.split_label;
      frame_diag.frame_index = frame.frame_index;
      frame_diag.frame_label = frame.frame_label;
      frame_diag.point_count = frame_point_count;
      frame_diag.outer_point_count = frame_outer_count;
      frame_diag.internal_point_count = frame_internal_count;
      frame_diag.rmse =
          std::sqrt(frame_squared_error / static_cast<double>(frame_point_count));
      result.frame_diagnostics.push_back(frame_diag);
    }
  }

  result.evaluated_frame_count = static_cast<int>(result.frame_diagnostics.size());
  result.evaluated_board_observation_count =
      static_cast<int>(result.board_observation_diagnostics.size());
  result.point_count = total_point_count;
  result.outer_point_count = outer_point_count;
  result.internal_point_count = internal_point_count;
  if (total_point_count <= 0) {
    result.failure_reason = "Camera-only refit evaluation produced zero valid points.";
    return result;
  }

  result.overall_rmse =
      std::sqrt(total_squared_error / static_cast<double>(total_point_count));
  result.outer_only_rmse =
      outer_point_count > 0
          ? std::sqrt(outer_squared_error / static_cast<double>(outer_point_count))
          : 0.0;
  result.internal_only_rmse =
      internal_point_count > 0
          ? std::sqrt(internal_squared_error / static_cast<double>(internal_point_count))
          : 0.0;
  result.success = true;
  return result;
}

Stage5BenchmarkReport Stage5Benchmark::Run(const Stage5BenchmarkInput& input) const {
  Stage5BenchmarkReport report;
  report.dataset_label = input.dataset_label.empty()
                             ? input.baseline_options.dataset_label
                             : input.dataset_label;

  report.split = BuildDeterministicSplit(input.all_frames);
  if (!report.split.success) {
    report.failure_reason = report.split.failure_reason;
    return report;
  }
  report.split_signature = report.split.split_signature;
  report.kalibr_reference = input.kalibr_reference;

  FrozenRound2BaselineOptions baseline_options = input.baseline_options;
  baseline_options.dataset_label = report.dataset_label;
  baseline_options.training_split_signature = report.split.split_signature;
  baseline_options.run_second_pass = true;
  const FrozenRound2BaselinePipeline baseline_pipeline(baseline_options);
  report.baseline_result = baseline_pipeline.Run(report.split.training_frames);
  report.baseline_protocol_label = report.baseline_result.baseline_protocol_label;
  if (!report.baseline_result.success) {
    report.failure_reason = report.baseline_result.failure_reason;
    return report;
  }
  if (!report.baseline_result.stage5_bundle_available) {
    report.failure_reason =
        "Frozen round2 baseline completed but final Stage 5 bundle is not ready.";
    return report;
  }

  report.backend_problem_input = BuildBackendProblemInput(
      report.baseline_result.final_stage5_bundle, input.backend_options);
  report.training_dataset =
      BuildTrainingEvaluationDataset(report.baseline_result.final_stage5_bundle);
  if (!report.training_dataset.success) {
    report.failure_reason = report.training_dataset.failure_reason;
    return report;
  }

  const JointReprojectionSceneState optimized_scene_state =
      report.baseline_result.round2_available
          ? report.baseline_result.round2.optimization_result.optimized_state
          : report.baseline_result.round1.optimization_result.optimized_state;
  report.holdout_dataset = BuildHoldoutEvaluationDataset(
      report.split.holdout_frames, baseline_options, optimized_scene_state,
      report.split.split_signature);
  if (!report.holdout_dataset.success) {
    report.failure_reason = report.holdout_dataset.failure_reason;
    return report;
  }

  OuterBootstrapCameraIntrinsics kalibr_intrinsics;
  std::string intrinsics_error;
  if (!LoadKalibrCamchainIntrinsics(input.kalibr_reference.camchain_yaml,
                                    &kalibr_intrinsics,
                                    &intrinsics_error)) {
    report.failure_reason = intrinsics_error;
    return report;
  }

  report.fair_protocol_matched =
      (input.kalibr_reference.camera_model_family == "ds") &&
      !input.kalibr_reference.training_split_signature.empty() &&
      input.kalibr_reference.training_split_signature == report.split.split_signature;
  report.diagnostic_only = !report.fair_protocol_matched;
  if (!report.fair_protocol_matched) {
    report.warnings.push_back(
        "Kalibr metadata did not match the deterministic training split signature; "
        "Stage 5 report is downgraded to diagnostic-only.");
  }

  report.our_training_evaluation = EvaluateCameraModel(
      report.training_dataset,
      report.baseline_result.final_stage5_bundle.scene_state.camera,
      "ours");
  report.kalibr_training_evaluation =
      EvaluateCameraModel(report.training_dataset, kalibr_intrinsics, "kalibr");
  report.our_holdout_evaluation = EvaluateCameraModel(
      report.holdout_dataset,
      report.baseline_result.final_stage5_bundle.scene_state.camera,
      "ours");
  report.kalibr_holdout_evaluation =
      EvaluateCameraModel(report.holdout_dataset, kalibr_intrinsics, "kalibr");

  if (!report.our_training_evaluation.success ||
      !report.kalibr_training_evaluation.success ||
      !report.our_holdout_evaluation.success ||
      !report.kalibr_holdout_evaluation.success) {
    report.failure_reason = "Camera-only refit evaluation failed for one or more methods.";
    return report;
  }

  KalibrBenchmarkInput diagnostic_input;
  diagnostic_input.dataset_label = report.dataset_label;
  diagnostic_input.kalibr_camchain_yaml = input.kalibr_reference.camchain_yaml;
  diagnostic_input.our_bundle = report.baseline_result.final_stage5_bundle;
  const KalibrBenchmark diagnostic_benchmark;
  report.diagnostic_compare = diagnostic_benchmark.Compare(diagnostic_input);
  if (!report.diagnostic_compare.success) {
    report.warnings.push_back(
        "Low-level Stage 5 diagnostic projection compare failed: " +
        report.diagnostic_compare.failure_reason);
  }

  report.success = true;
  report.warnings.insert(report.warnings.end(),
                         report.baseline_result.warnings.begin(),
                         report.baseline_result.warnings.end());
  return report;
}

cv::Mat Stage5Benchmark::RenderProjectionComparison(const Stage5BenchmarkReport& report,
                                                    int max_width,
                                                    int max_height) const {
  if (!report.diagnostic_compare.success) {
    return cv::Mat();
  }
  const KalibrBenchmark diagnostic_benchmark;
  return diagnostic_benchmark.RenderProjectionComparison(
      report.diagnostic_compare, max_width, max_height);
}

std::string Stage5Benchmark::FindFrameImagePath(const Stage5BenchmarkReport& report,
                                                int frame_index) const {
  const auto search = [frame_index](const std::vector<FrozenRound2BaselineFrameSource>& frames)
      -> std::string {
    for (const FrozenRound2BaselineFrameSource& frame : frames) {
      if (frame.frame_index == frame_index) {
        return frame.image_path;
      }
    }
    return std::string();
  };

  std::string image_path = search(report.split.holdout_frames);
  if (!image_path.empty()) {
    return image_path;
  }
  image_path = search(report.split.training_frames);
  if (!image_path.empty()) {
    return image_path;
  }
  return std::string();
}

cv::Mat Stage5Benchmark::RenderEvaluationFrameOverlay(
    const Stage5BenchmarkReport& report,
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index) const {
  const std::string image_path = FindFrameImagePath(report, frame_index);
  if (image_path.empty()) {
    return cv::Mat();
  }

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    return cv::Mat();
  }

  cv::Mat output = image.clone();
  int point_count = 0;
  int outer_count = 0;
  int internal_count = 0;
  double worst_residual = 0.0;

  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index) {
      continue;
    }
    ++point_count;
    if (point.point_type == JointPointType::Outer) {
      ++outer_count;
    } else {
      ++internal_count;
    }
    worst_residual = std::max(worst_residual, point.residual_norm);

    const cv::Point observed(static_cast<int>(std::lround(point.observed_image_xy.x())),
                             static_cast<int>(std::lround(point.observed_image_xy.y())));
    const cv::Point predicted(static_cast<int>(std::lround(point.predicted_image_xy.x())),
                              static_cast<int>(std::lround(point.predicted_image_xy.y())));
    const cv::Scalar observed_color =
        point.point_type == JointPointType::Outer ? cv::Scalar(60, 220, 80)
                                                  : cv::Scalar(40, 180, 255);
    cv::circle(output, observed, point.point_type == JointPointType::Outer ? 5 : 3,
               observed_color, 2, cv::LINE_AA);
    cv::drawMarker(output, predicted, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10, 1,
                   cv::LINE_AA);
    cv::line(output, observed, predicted, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
  }

  double frame_rmse = 0.0;
  std::string frame_label;
  for (const CameraModelRefitFrameDiagnostics& frame : evaluation.frame_diagnostics) {
    if (frame.frame_index == frame_index) {
      frame_rmse = frame.rmse;
      frame_label = frame.frame_label;
      break;
    }
  }

  const int banner_height = 82;
  cv::rectangle(output, cv::Rect(0, 0, output.cols, banner_height),
                cv::Scalar(18, 18, 18), cv::FILLED);
  std::ostringstream header;
  header << evaluation.method_label << " " << evaluation.split_label
         << " frame=" << frame_index;
  if (!frame_label.empty()) {
    header << " (" << frame_label << ")";
  }
  header << " rmse=" << frame_rmse;
  cv::putText(output, header.str(), cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.62,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

  std::ostringstream summary;
  summary << "points=" << point_count << " outer=" << outer_count
          << " internal=" << internal_count
          << " worst=" << worst_residual;
  cv::putText(output, summary.str(), cv::Point(18, 54), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(195, 195, 195), 1, cv::LINE_AA);

  return output;
}

cv::Mat Stage5Benchmark::RenderEvaluationBoardObservationOverlay(
    const Stage5BenchmarkReport& report,
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index,
    int board_id) const {
  cv::Mat full_overlay = RenderEvaluationFrameOverlay(report, evaluation, frame_index);
  if (full_overlay.empty()) {
    return cv::Mat();
  }

  bool has_points = false;
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  int point_count = 0;
  double board_rmse = 0.0;

  for (const CameraModelRefitBoardObservationDiagnostics& board :
       evaluation.board_observation_diagnostics) {
    if (board.frame_index == frame_index && board.board_id == board_id) {
      board_rmse = board.evaluation_rmse;
      break;
    }
  }

  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index || point.board_id != board_id) {
      continue;
    }
    has_points = true;
    ++point_count;
    min_x = std::min(min_x, std::min(point.observed_image_xy.x(), point.predicted_image_xy.x()));
    min_y = std::min(min_y, std::min(point.observed_image_xy.y(), point.predicted_image_xy.y()));
    max_x = std::max(max_x, std::max(point.observed_image_xy.x(), point.predicted_image_xy.x()));
    max_y = std::max(max_y, std::max(point.observed_image_xy.y(), point.predicted_image_xy.y()));
  }

  if (!has_points) {
    return cv::Mat();
  }

  const int padding = 80;
  cv::Rect crop_rect(static_cast<int>(std::floor(min_x)) - padding,
                     static_cast<int>(std::floor(min_y)) - padding,
                     static_cast<int>(std::ceil(max_x - min_x)) + 2 * padding,
                     static_cast<int>(std::ceil(max_y - min_y)) + 2 * padding);
  crop_rect = ClampRectToImage(crop_rect, full_overlay.size());
  if (crop_rect.width <= 0 || crop_rect.height <= 0) {
    return cv::Mat();
  }

  cv::Mat cropped = full_overlay(crop_rect).clone();
  cv::rectangle(cropped, cv::Rect(0, 0, cropped.cols, 54), cv::Scalar(18, 18, 18), cv::FILLED);
  std::ostringstream banner;
  banner << evaluation.method_label << " frame=" << frame_index
         << " board=" << board_id
         << " rmse=" << board_rmse
         << " points=" << point_count;
  cv::putText(cropped, banner.str(), cv::Point(12, 24), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
  cv::putText(cropped, "green/orange: observed, red cross: predicted",
              cv::Point(12, 44), cv::FONT_HERSHEY_PLAIN, 1.0,
              cv::Scalar(190, 190, 190), 1, cv::LINE_AA);
  return cropped;
}

cv::Mat Stage5Benchmark::RenderOuterPoseFitFrameOverlay(
    const Stage5BenchmarkReport& report,
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index) const {
  const std::string image_path = FindFrameImagePath(report, frame_index);
  if (image_path.empty()) {
    return cv::Mat();
  }

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    return cv::Mat();
  }

  cv::Mat output = image.clone();
  int outer_count = 0;
  double worst_outer_residual = 0.0;
  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index || point.point_type != JointPointType::Outer) {
      continue;
    }
    ++outer_count;
    worst_outer_residual = std::max(worst_outer_residual, point.residual_norm);
    DrawObservedPredictedPoint(&output, point, cv::Scalar(60, 220, 80), 5, true);
  }

  const CameraModelRefitFrameDiagnostics* frame_diag =
      FindFrameDiagnostics(evaluation, frame_index);
  const double outer_frame_rmse = ComputeOuterRmseForFrame(evaluation, frame_index);
  const int banner_height = 82;
  cv::rectangle(output, cv::Rect(0, 0, output.cols, banner_height),
                cv::Scalar(18, 18, 18), cv::FILLED);

  std::ostringstream header;
  header << evaluation.method_label << " " << evaluation.split_label
         << " outer-only frame=" << frame_index;
  if (frame_diag != nullptr && !frame_diag->frame_label.empty()) {
    header << " (" << frame_diag->frame_label << ")";
  }
  header << " outer_only_rmse=" << outer_frame_rmse;
  cv::putText(output, header.str(), cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.62,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

  std::ostringstream summary;
  summary << "outer_points=" << outer_count
          << " eval_outer_only_rmse=" << evaluation.outer_only_rmse
          << " worst_outer=" << worst_outer_residual;
  cv::putText(output, summary.str(), cv::Point(18, 54), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(195, 195, 195), 1, cv::LINE_AA);
  return output;
}

cv::Mat Stage5Benchmark::RenderOuterPoseFitBoardOverlay(
    const Stage5BenchmarkReport& report,
    const CameraModelRefitEvaluationResult& evaluation,
    int frame_index,
    int board_id) const {
  const std::string image_path = FindFrameImagePath(report, frame_index);
  if (image_path.empty()) {
    return cv::Mat();
  }

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    return cv::Mat();
  }

  bool has_outer_points = false;
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  int outer_point_count = 0;
  double worst_outer_residual = 0.0;
  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index || point.board_id != board_id ||
        point.point_type != JointPointType::Outer) {
      continue;
    }
    if (!AccumulatePointBounds(point, &min_x, &min_y, &max_x, &max_y)) {
      continue;
    }
    has_outer_points = true;
    ++outer_point_count;
    worst_outer_residual = std::max(worst_outer_residual, point.residual_norm);
  }
  if (!has_outer_points) {
    return cv::Mat();
  }

  const int padding = 80;
  cv::Rect crop_rect(static_cast<int>(std::floor(min_x)) - padding,
                     static_cast<int>(std::floor(min_y)) - padding,
                     static_cast<int>(std::ceil(max_x - min_x)) + 2 * padding,
                     static_cast<int>(std::ceil(max_y - min_y)) + 2 * padding);
  crop_rect = ClampRectToImage(crop_rect, image.size());
  if (crop_rect.width <= 0 || crop_rect.height <= 0) {
    return cv::Mat();
  }

  cv::Mat cropped = image(crop_rect).clone();
  for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.frame_index != frame_index || point.board_id != board_id ||
        point.point_type != JointPointType::Outer ||
        !IsFiniteImagePoint(point.observed_image_xy) ||
        !IsFiniteImagePoint(point.predicted_image_xy)) {
      continue;
    }

    CameraModelRefitPointDiagnostics shifted_point = point;
    shifted_point.observed_image_xy -=
        Eigen::Vector2d(static_cast<double>(crop_rect.x), static_cast<double>(crop_rect.y));
    shifted_point.predicted_image_xy -=
        Eigen::Vector2d(static_cast<double>(crop_rect.x), static_cast<double>(crop_rect.y));
    DrawObservedPredictedPoint(&cropped, shifted_point, cv::Scalar(60, 220, 80), 5, true);
  }

  const CameraModelRefitBoardObservationDiagnostics* board_diag =
      FindBoardDiagnostics(evaluation, frame_index, board_id);
  cv::rectangle(cropped, cv::Rect(0, 0, cropped.cols, 54), cv::Scalar(18, 18, 18),
                cv::FILLED);
  std::ostringstream banner;
  banner << evaluation.method_label << " frame=" << frame_index
         << " board=" << board_id
         << " pose_fit_outer_rmse="
         << (board_diag != nullptr ? board_diag->pose_fit_outer_rmse : 0.0)
         << " outer_points=" << outer_point_count;
  cv::putText(cropped, banner.str(), cv::Point(12, 24), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
  std::ostringstream detail;
  detail << "green: observed outer, red cross: predicted, worst_outer="
         << worst_outer_residual;
  cv::putText(cropped, detail.str(), cv::Point(12, 44), cv::FONT_HERSHEY_PLAIN, 1.0,
              cv::Scalar(190, 190, 190), 1, cv::LINE_AA);
  return cropped;
}

void WriteStage5BenchmarkProtocolSummary(const std::string& path,
                                         const Stage5BenchmarkReport& report) {
  std::ofstream output(path.c_str());
  output << "success: " << (report.success ? 1 : 0) << "\n";
  output << "failure_reason: " << report.failure_reason << "\n";
  output << "baseline_protocol_label: " << report.baseline_protocol_label << "\n";
  output << "dataset_label: " << report.dataset_label << "\n";
  output << "split_signature: " << report.split_signature << "\n";
  output << "split_mode: " << report.split.mode << "\n";
  output << "holdout_stride: " << report.split.holdout_stride << "\n";
  output << "holdout_offset: " << report.split.holdout_offset << "\n";
  output << "training_frame_count: " << report.split.training_frames.size() << "\n";
  output << "holdout_frame_count: " << report.split.holdout_frames.size() << "\n";
  output << "fair_protocol_matched: " << (report.fair_protocol_matched ? 1 : 0) << "\n";
  output << "diagnostic_only: " << (report.diagnostic_only ? 1 : 0) << "\n";
  output << "kalibr_camera_model_family: " << report.kalibr_reference.camera_model_family
         << "\n";
  output << "kalibr_training_split_signature: "
         << report.kalibr_reference.training_split_signature << "\n";
  output << "kalibr_source_label: " << report.kalibr_reference.source_label << "\n";
  output << "comparison_scope: output_level_and_evaluator_level_only\n";
  output << "evaluation_protocol: camera_only_outer_refit_pose_plus_outer_internal_reprojection\n";
  for (const std::string& warning : report.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteStage5BenchmarkTrainingSummary(const std::string& path,
                                         const Stage5BenchmarkReport& report) {
  std::ofstream output(path.c_str());
  output << "split_label: training\n";
  output << "split_signature: " << report.split_signature << "\n";
  output << "our_overall_rmse: " << report.our_training_evaluation.overall_rmse << "\n";
  output << "our_outer_only_rmse: " << report.our_training_evaluation.outer_only_rmse << "\n";
  output << "our_internal_only_rmse: " << report.our_training_evaluation.internal_only_rmse
         << "\n";
  output << "kalibr_overall_rmse: " << report.kalibr_training_evaluation.overall_rmse << "\n";
  output << "kalibr_outer_only_rmse: " << report.kalibr_training_evaluation.outer_only_rmse
         << "\n";
  output << "kalibr_internal_only_rmse: "
         << report.kalibr_training_evaluation.internal_only_rmse << "\n";
  output << "our_point_count: " << report.our_training_evaluation.point_count << "\n";
  output << "kalibr_point_count: " << report.kalibr_training_evaluation.point_count << "\n";
}

void WriteStage5BenchmarkHoldoutSummary(const std::string& path,
                                        const Stage5BenchmarkReport& report) {
  std::ofstream output(path.c_str());
  output << "split_label: holdout\n";
  output << "split_signature: " << report.split_signature << "\n";
  output << "our_overall_rmse: " << report.our_holdout_evaluation.overall_rmse << "\n";
  output << "our_outer_only_rmse: " << report.our_holdout_evaluation.outer_only_rmse << "\n";
  output << "our_internal_only_rmse: " << report.our_holdout_evaluation.internal_only_rmse
         << "\n";
  output << "kalibr_overall_rmse: " << report.kalibr_holdout_evaluation.overall_rmse << "\n";
  output << "kalibr_outer_only_rmse: " << report.kalibr_holdout_evaluation.outer_only_rmse
         << "\n";
  output << "kalibr_internal_only_rmse: "
         << report.kalibr_holdout_evaluation.internal_only_rmse << "\n";
  output << "our_point_count: " << report.our_holdout_evaluation.point_count << "\n";
  output << "kalibr_point_count: " << report.kalibr_holdout_evaluation.point_count << "\n";
}

void WriteStage5BenchmarkHoldoutPointsCsv(const std::string& path,
                                          const Stage5BenchmarkReport& report) {
  std::ofstream output(path.c_str());
  output << "method,split,frame_index,frame_label,board_id,point_id,point_type,"
         << "observed_x,observed_y,predicted_x,predicted_y,target_x,target_y,target_z,"
         << "residual_x,residual_y,residual_norm,debug_quality,source_kind,source_point_index\n";
  const auto write_points = [&output](const CameraModelRefitEvaluationResult& evaluation) {
    for (const CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
      output << point.method_label << ","
             << point.split_label << ","
             << point.frame_index << ","
             << point.frame_label << ","
             << point.board_id << ","
             << point.point_id << ","
             << ToString(point.point_type) << ","
             << point.observed_image_xy.x() << ","
             << point.observed_image_xy.y() << ","
             << point.predicted_image_xy.x() << ","
             << point.predicted_image_xy.y() << ","
             << point.target_xyz_board.x() << ","
             << point.target_xyz_board.y() << ","
             << point.target_xyz_board.z() << ","
             << point.residual_xy.x() << ","
             << point.residual_xy.y() << ","
             << point.residual_norm << ","
             << point.quality << ","
             << ToString(point.source_kind) << ","
             << point.source_point_index << "\n";
    }
  };
  write_points(report.our_holdout_evaluation);
  write_points(report.kalibr_holdout_evaluation);
}

void WriteStage5BenchmarkWorstCasesSummary(const std::string& path,
                                           const Stage5BenchmarkReport& report,
                                           int top_k) {
  std::ofstream output(path.c_str());
  output << "dataset_label: " << report.dataset_label << "\n";
  output << "split_signature: " << report.split_signature << "\n";
  output << "top_k: " << top_k << "\n";

  const auto write_eval = [&output, top_k](const CameraModelRefitEvaluationResult& evaluation) {
    output << "\n[" << evaluation.method_label << "_" << evaluation.split_label << "]\n";
    output << "overall_rmse: " << evaluation.overall_rmse << "\n";
    output << "outer_only_rmse: " << evaluation.outer_only_rmse << "\n";
    output << "internal_only_rmse: " << evaluation.internal_only_rmse << "\n";

    std::vector<CameraModelRefitFrameDiagnostics> worst_frames =
        evaluation.frame_diagnostics;
    std::sort(worst_frames.begin(), worst_frames.end(),
              [](const CameraModelRefitFrameDiagnostics& lhs,
                 const CameraModelRefitFrameDiagnostics& rhs) {
                return lhs.rmse > rhs.rmse;
              });
    if (top_k >= 0 && static_cast<int>(worst_frames.size()) > top_k) {
      worst_frames.resize(static_cast<std::size_t>(top_k));
    }
    output << "worst_frames:\n";
    for (const CameraModelRefitFrameDiagnostics& frame : worst_frames) {
      output << "  frame_index=" << frame.frame_index
             << " frame_label=" << frame.frame_label
             << " rmse=" << frame.rmse
             << " point_count=" << frame.point_count
             << " outer_point_count=" << frame.outer_point_count
             << " internal_point_count=" << frame.internal_point_count << "\n";
    }

    std::vector<CameraModelRefitBoardObservationDiagnostics> worst_boards =
        evaluation.board_observation_diagnostics;
    std::sort(worst_boards.begin(), worst_boards.end(),
              [](const CameraModelRefitBoardObservationDiagnostics& lhs,
                 const CameraModelRefitBoardObservationDiagnostics& rhs) {
                return lhs.evaluation_rmse > rhs.evaluation_rmse;
              });
    if (top_k >= 0 && static_cast<int>(worst_boards.size()) > top_k) {
      worst_boards.resize(static_cast<std::size_t>(top_k));
    }
    output << "worst_board_observations:\n";
    for (const CameraModelRefitBoardObservationDiagnostics& board : worst_boards) {
      output << "  frame_index=" << board.frame_index
             << " frame_label=" << board.frame_label
             << " board_id=" << board.board_id
             << " rmse=" << board.evaluation_rmse
             << " pose_fit_outer_rmse=" << board.pose_fit_outer_rmse
             << " point_count=" << board.point_count
             << " outer_point_count=" << board.outer_point_count
             << " internal_point_count=" << board.internal_point_count << "\n";
    }
  };

  write_eval(report.our_holdout_evaluation);
  write_eval(report.kalibr_holdout_evaluation);
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
