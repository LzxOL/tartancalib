#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

void AppendUniqueBoardId(int board_id, std::vector<int>* board_ids) {
  if (board_ids == nullptr || board_id < 0) {
    return;
  }
  if (std::find(board_ids->begin(), board_ids->end(), board_id) == board_ids->end()) {
    board_ids->push_back(board_id);
  }
}

void AppendUniqueWarning(const std::string& warning,
                         std::vector<std::string>* warnings) {
  if (warnings == nullptr || warning.empty()) {
    return;
  }
  if (std::find(warnings->begin(), warnings->end(), warning) == warnings->end()) {
    warnings->push_back(warning);
  }
}

std::string JoinBoardIds(const std::vector<int>& board_ids) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < board_ids.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << board_ids[index];
  }
  return stream.str();
}

Eigen::Matrix4d ComposeCameraBoardTransform(const Eigen::Matrix4d& T_camera_reference,
                                            const Eigen::Matrix4d& T_reference_board) {
  return T_camera_reference * T_reference_board;
}

ApriltagInternalDetectionResult BuildFailedDetectionResult(
    int board_id,
    const cv::Size& image_size,
    const OuterTagDetectionResult& outer_detection,
    const std::string& failure_reason) {
  ApriltagInternalDetectionResult detection;
  detection.board_id = board_id;
  detection.image_size = image_size;
  detection.outer_detection = outer_detection;
  detection.tag_detected = outer_detection.success;
  detection.failure_reason = failure_reason;
  return detection;
}

std::string BuildRegenerationFailureWarning(
    const InternalRegenerationFrameInput& frame_input,
    const std::string& state_source_label,
    int board_id,
    bool pose_prior_used,
    const std::string& failure_reason) {
  std::ostringstream stream;
  stream << "state=" << state_source_label
         << " frame=" << frame_input.frame_index;
  if (!frame_input.frame_label.empty()) {
    stream << " (" << frame_input.frame_label << ")";
  }
  stream << " board=" << board_id
         << " prior=" << (pose_prior_used ? 1 : 0)
         << " skipped: " << failure_reason;
  return stream.str();
}

void EmitRegenerationWarning(const std::string& warning,
                             std::vector<std::string>* warnings) {
  if (warning.empty()) {
    return;
  }
  AppendUniqueWarning(warning, warnings);
  std::cerr << "[internal_regen] " << warning << std::endl;
}

}  // namespace

int InternalRegenerationFrameResult::SuccessfulBoardCount() const {
  int count = 0;
  for (const RegeneratedBoardMeasurement& measurement : board_measurements) {
    count += measurement.detection.success ? 1 : 0;
  }
  return count;
}

int InternalRegenerationFrameResult::ValidInternalCornerCount() const {
  int count = 0;
  for (const RegeneratedBoardMeasurement& measurement : board_measurements) {
    count += measurement.detection.valid_internal_corner_count;
  }
  return count;
}

ApriltagInternalMultiDetectionResult InternalRegenerationFrameResult::AsMultiDetectionResult() const {
  ApriltagInternalMultiDetectionResult result;
  result.image_size = image_size;
  result.requested_board_ids.reserve(board_measurements.size());
  result.detections.reserve(board_measurements.size());
  for (const RegeneratedBoardMeasurement& measurement : board_measurements) {
    result.requested_board_ids.push_back(measurement.board_id);
    result.detections.push_back(measurement.detection);
  }
  return result;
}

MultiBoardInternalMeasurementRegenerator::MultiBoardInternalMeasurementRegenerator(
    ApriltagInternalConfig config,
    ApriltagInternalDetectionOptions options)
    : config_(std::move(config)),
      options_(std::move(options)),
      detector_(config_, options_) {}

InternalRegenerationFrameResult MultiBoardInternalMeasurementRegenerator::RegenerateFrame(
    const cv::Mat& image,
    const InternalRegenerationFrameInput& frame_input,
    const OuterBootstrapResult& bootstrap_result) const {
  if (image.empty()) {
    throw std::runtime_error("RegenerateFrame requires a non-empty image.");
  }

  InternalRegenerationFrameResult result;
  result.frame_index = frame_input.frame_index;
  result.frame_label = frame_input.frame_label;
  result.state_source_label = "bootstrap";
  result.image_size = image.size();

  const OuterBootstrapFrameState* frame_state = FindFrameState(bootstrap_result, frame_input);
  result.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;

  const IntermediateCameraConfig camera_override =
      MakeBootstrapCameraConfig(bootstrap_result.coarse_camera);

  result.board_measurements.reserve(frame_input.outer_detections.detections.size());
  for (std::size_t index = 0; index < frame_input.outer_detections.requested_board_ids.size(); ++index) {
    const int board_id = frame_input.outer_detections.requested_board_ids[index];
    OuterTagDetectionResult outer_detection;
    if (index < frame_input.outer_detections.detections.size()) {
      outer_detection = frame_input.outer_detections.detections[index];
    } else {
      outer_detection.board_id = board_id;
      outer_detection.failure_reason = OuterTagFailureReason::NoDetectionsAtAll;
      outer_detection.failure_reason_text = ToString(outer_detection.failure_reason);
    }

    const OuterBootstrapBoardState* board_state = FindBoardState(bootstrap_result, board_id);
    const bool pose_prior_available =
        frame_state != nullptr && frame_state->initialized &&
        board_state != nullptr && board_state->initialized;
    const Eigen::Matrix4d T_camera_board = pose_prior_available
                                               ? ComposeCameraBoardTransform(
                                                     frame_state->T_camera_reference,
                                                     board_state->T_reference_board)
                                               : Eigen::Matrix4d::Identity();

    RegeneratedBoardMeasurement measurement;
    measurement.board_id = board_id;
    measurement.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;
    measurement.board_bootstrap_initialized = board_state != nullptr && board_state->initialized;
    measurement.pose_prior_used = pose_prior_available;
    try {
      measurement.detection = detector_.DetectFromOuterDetection(
          image, board_id, outer_detection, &camera_override,
          pose_prior_available ? &T_camera_board : nullptr);
    } catch (const std::exception& error) {
      measurement.detection = BuildFailedDetectionResult(
          board_id, image.size(), outer_detection, error.what());
    }
    if (outer_detection.success && !measurement.detection.success &&
        !measurement.detection.failure_reason.empty()) {
      EmitRegenerationWarning(
          BuildRegenerationFailureWarning(
              frame_input, result.state_source_label, board_id,
              measurement.pose_prior_used, measurement.detection.failure_reason),
          &result.warnings);
    }
    result.board_measurements.push_back(measurement);

    if (outer_detection.success) {
      AppendUniqueBoardId(board_id, &result.visible_board_ids);
    }
  }

  return result;
}

InternalRegenerationFrameResult MultiBoardInternalMeasurementRegenerator::RegenerateFrame(
    const cv::Mat& image,
    const InternalRegenerationFrameInput& frame_input,
    const JointReprojectionSceneState& scene_state) const {
  if (image.empty()) {
    throw std::runtime_error("RegenerateFrame requires a non-empty image.");
  }

  InternalRegenerationFrameResult result;
  result.frame_index = frame_input.frame_index;
  result.frame_label = frame_input.frame_label;
  result.state_source_label = "optimized_scene";
  result.image_size = image.size();

  const JointSceneFrameState* frame_state = FindFrameState(scene_state, frame_input);
  result.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;

  const IntermediateCameraConfig camera_override =
      MakeSceneCameraConfig(scene_state.camera);

  result.board_measurements.reserve(frame_input.outer_detections.detections.size());
  for (std::size_t index = 0; index < frame_input.outer_detections.requested_board_ids.size(); ++index) {
    const int board_id = frame_input.outer_detections.requested_board_ids[index];
    OuterTagDetectionResult outer_detection;
    if (index < frame_input.outer_detections.detections.size()) {
      outer_detection = frame_input.outer_detections.detections[index];
    } else {
      outer_detection.board_id = board_id;
      outer_detection.failure_reason = OuterTagFailureReason::NoDetectionsAtAll;
      outer_detection.failure_reason_text = ToString(outer_detection.failure_reason);
    }

    const JointSceneBoardState* board_state = FindBoardState(scene_state, board_id);
    const bool pose_prior_available =
        frame_state != nullptr && frame_state->initialized &&
        board_state != nullptr && board_state->initialized;
    const Eigen::Matrix4d T_camera_board = pose_prior_available
                                               ? ComposeCameraBoardTransform(
                                                     frame_state->T_camera_reference,
                                                     board_state->T_reference_board)
                                               : Eigen::Matrix4d::Identity();

    RegeneratedBoardMeasurement measurement;
    measurement.board_id = board_id;
    measurement.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;
    measurement.board_bootstrap_initialized = board_state != nullptr && board_state->initialized;
    measurement.pose_prior_used = pose_prior_available;
    try {
      measurement.detection = detector_.DetectFromOuterDetection(
          image, board_id, outer_detection, &camera_override,
          pose_prior_available ? &T_camera_board : nullptr);
    } catch (const std::exception& error) {
      measurement.detection = BuildFailedDetectionResult(
          board_id, image.size(), outer_detection, error.what());
    }
    if (outer_detection.success && !measurement.detection.success &&
        !measurement.detection.failure_reason.empty()) {
      EmitRegenerationWarning(
          BuildRegenerationFailureWarning(
              frame_input, result.state_source_label, board_id,
              measurement.pose_prior_used, measurement.detection.failure_reason),
          &result.warnings);
    }
    result.board_measurements.push_back(measurement);

    if (outer_detection.success) {
      AppendUniqueBoardId(board_id, &result.visible_board_ids);
    }
  }

  return result;
}

void MultiBoardInternalMeasurementRegenerator::DrawFrameOverlay(
    const cv::Mat& image,
    const InternalRegenerationFrameResult& frame_result,
    cv::Mat* output_image) const {
  if (output_image == nullptr) {
    throw std::runtime_error("DrawFrameOverlay requires a valid output pointer.");
  }
  *output_image = image.clone();
  ApriltagInternalMultiDetectionResult multi_detection = frame_result.AsMultiDetectionResult();
  detector_.DrawDetections(multi_detection, output_image);

  const int banner_height = 78;
  cv::rectangle(*output_image, cv::Rect(0, 0, output_image->cols, banner_height),
                cv::Scalar(18, 18, 18), cv::FILLED);

  std::ostringstream header;
  header << "frame " << frame_result.frame_index << "  state="
         << frame_result.state_source_label << "  frame_init="
         << (frame_result.frame_bootstrap_initialized ? "yes" : "no")
         << "  successful_boards=" << frame_result.SuccessfulBoardCount() << "/"
         << frame_result.board_measurements.size()
         << "  valid_internal=" << frame_result.ValidInternalCornerCount();
  cv::putText(*output_image, header.str(), cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.62,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

  std::ostringstream board_line;
  board_line << "visible=" << JoinBoardIds(frame_result.visible_board_ids);
  cv::putText(*output_image, board_line.str(), cv::Point(18, 53), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(180, 180, 180), 1, cv::LINE_AA);

  int x = 18;
  for (const RegeneratedBoardMeasurement& measurement : frame_result.board_measurements) {
    std::ostringstream token;
    token << "#" << measurement.board_id
          << " prior=" << (measurement.pose_prior_used ? "Y" : "N")
          << " int=" << measurement.detection.valid_internal_corner_count;
    cv::putText(*output_image, token.str(), cv::Point(x, 71), cv::FONT_HERSHEY_PLAIN, 1.0,
                measurement.detection.success ? cv::Scalar(100, 220, 120)
                                              : cv::Scalar(150, 150, 150),
                1, cv::LINE_AA);
    x += 145;
  }
}

const OuterBootstrapFrameState* MultiBoardInternalMeasurementRegenerator::FindFrameState(
    const OuterBootstrapResult& bootstrap_result,
    const InternalRegenerationFrameInput& frame_input) const {
  for (const OuterBootstrapFrameState& frame_state : bootstrap_result.frames) {
    if (frame_state.frame_index == frame_input.frame_index) {
      return &frame_state;
    }
  }
  for (const OuterBootstrapFrameState& frame_state : bootstrap_result.frames) {
    if (!frame_input.frame_label.empty() && frame_state.frame_label == frame_input.frame_label) {
      return &frame_state;
    }
  }
  return nullptr;
}

const OuterBootstrapBoardState* MultiBoardInternalMeasurementRegenerator::FindBoardState(
    const OuterBootstrapResult& bootstrap_result,
    int board_id) const {
  for (const OuterBootstrapBoardState& board_state : bootstrap_result.boards) {
    if (board_state.board_id == board_id) {
      return &board_state;
    }
  }
  return nullptr;
}

IntermediateCameraConfig MultiBoardInternalMeasurementRegenerator::MakeBootstrapCameraConfig(
    const OuterBootstrapCameraIntrinsics& intrinsics) const {
  return MakeSceneCameraConfig(intrinsics);
}

const JointSceneFrameState* MultiBoardInternalMeasurementRegenerator::FindFrameState(
    const JointReprojectionSceneState& scene_state,
    const InternalRegenerationFrameInput& frame_input) const {
  for (const JointSceneFrameState& frame_state : scene_state.frames) {
    if (frame_state.frame_index == frame_input.frame_index) {
      return &frame_state;
    }
  }
  for (const JointSceneFrameState& frame_state : scene_state.frames) {
    if (!frame_input.frame_label.empty() && frame_state.frame_label == frame_input.frame_label) {
      return &frame_state;
    }
  }
  return nullptr;
}

const JointSceneBoardState* MultiBoardInternalMeasurementRegenerator::FindBoardState(
    const JointReprojectionSceneState& scene_state,
    int board_id) const {
  for (const JointSceneBoardState& board_state : scene_state.boards) {
    if (board_state.board_id == board_id) {
      return &board_state;
    }
  }
  return nullptr;
}

IntermediateCameraConfig MultiBoardInternalMeasurementRegenerator::MakeSceneCameraConfig(
    const OuterBootstrapCameraIntrinsics& intrinsics) const {
  IntermediateCameraConfig config = config_.intermediate_camera;
  config.camera_model = "ds";
  config.distortion_model = "none";
  config.distortion_coeffs.clear();
  config.intrinsics = {intrinsics.xi, intrinsics.alpha, intrinsics.fu, intrinsics.fv,
                       intrinsics.cu, intrinsics.cv};
  config.resolution = {intrinsics.resolution.width, intrinsics.resolution.height};
  return config;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
