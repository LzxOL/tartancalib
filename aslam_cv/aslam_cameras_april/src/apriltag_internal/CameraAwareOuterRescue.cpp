#include <aslam/cameras/apriltag_internal/CameraAwareOuterRescue.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <opencv2/imgcodecs.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

void AppendUniqueWarning(const std::string& warning,
                         std::vector<std::string>* warnings) {
  if (warnings == nullptr || warning.empty()) {
    return;
  }
  if (std::find(warnings->begin(), warnings->end(), warning) == warnings->end()) {
    warnings->push_back(warning);
  }
}

std::string MakeCameraAwareRescueSignatureImpl(
    const OuterBootstrapCameraIntrinsics& camera,
    const MultiScaleOuterTagDetectorConfig& config,
    int max_hamming,
    int reference_board_id) {
  std::ostringstream stream;
  stream << std::setprecision(17)
         << "camera_aware_outer_rescue_v6_robust_missing_board_recovery|family="
         << camera.NormalizedFamilyString()
         << "|distortion=" << camera.NormalizedDistortionModel()
         << "|camera=";
  for (double value : camera.CombinedParameterVector()) {
    stream << value << ",";
  }
  stream << "|reference_board_id=" << reference_board_id
         << "|max_hamming=" << max_hamming
         << "|patch_size=640"
         << "|zero_detection="
         << (config.camera_aware_sphere_patch_rescue_zero_detection_frames ? 1 : 0)
         << "|extended_atlas="
         << (config.camera_aware_sphere_patch_use_extended_atlas ? 1 : 0)
         << "|commit_mapped_corners="
         << (config.camera_aware_sphere_patch_commit_mapped_corners ? 1 : 0)
         << "|robust_missing_board_recovery="
         << (config.enable_robust_missing_board_recovery ? 1 : 0);
  return stream.str();
}

OuterRefineCameraConfig BuildOuterRefineCameraConfig(
    const OuterBootstrapCameraIntrinsics& camera) {
  OuterRefineCameraConfig config;
  config.camera_model = camera.NormalizedCameraModel();
  config.distortion_model = camera.NormalizedDistortionModel();
  config.intrinsics = camera.IntrinsicsVector();
  config.distortion_coeffs = camera.DistortionVector();
  config.resolution = {camera.resolution.width, camera.resolution.height};
  return config;
}

const OuterBootstrapFrameState* FindBootstrapFrameState(
    const OuterBootstrapResult& result,
    int frame_index) {
  const auto it = std::find_if(
      result.frames.begin(), result.frames.end(),
      [frame_index](const OuterBootstrapFrameState& frame) {
        return frame.frame_index == frame_index;
      });
  return it == result.frames.end() ? nullptr : &(*it);
}

const OuterBootstrapBoardState* FindBootstrapBoardState(
    const OuterBootstrapResult& result,
    int board_id) {
  const auto it = std::find_if(
      result.boards.begin(), result.boards.end(),
      [board_id](const OuterBootstrapBoardState& board) {
        return board.board_id == board_id;
      });
  return it == result.boards.end() ? nullptr : &(*it);
}

std::vector<OuterBootstrapFrameInput> BuildDirectOuterBootstrapFrames(
    const std::vector<OuterBootstrapFrameInput>& frames) {
  std::vector<OuterBootstrapFrameInput> direct_frames;
  direct_frames.reserve(frames.size());
  for (const OuterBootstrapFrameInput& frame : frames) {
    OuterBootstrapFrameInput direct_frame = frame;
    direct_frame.measurements.board_measurements.clear();
    for (const OuterBoardMeasurement& measurement :
         frame.measurements.board_measurements) {
      // A camera-aware patch result is the object being validated. Do not let
      // an older cached patch result establish the independent reference rig.
      if (measurement.success && !measurement.used_local_patch_rescue) {
        direct_frame.measurements.board_measurements.push_back(measurement);
      }
    }
    if (!direct_frame.measurements.board_measurements.empty()) {
      direct_frames.push_back(std::move(direct_frame));
    }
  }
  return direct_frames;
}

struct DirectLayoutRescueValidation {
  bool evaluable = false;
  double reprojection_rmse_px = std::numeric_limits<double>::infinity();
};

DirectLayoutRescueValidation ValidatePatchRescueAgainstDirectLayout(
    const OuterBootstrapResult& direct_layout,
    const OuterBootstrapCameraIntrinsics& camera,
    int frame_index,
    const OuterBoardMeasurement& measurement) {
  DirectLayoutRescueValidation validation;
  if (!direct_layout.success || !measurement.success ||
      !measurement.has_target_outer_corners || !camera.IsValid()) {
    return validation;
  }
  const OuterBootstrapFrameState* frame =
      FindBootstrapFrameState(direct_layout, frame_index);
  if (frame == nullptr || !frame->initialized) {
    return validation;
  }

  Eigen::Matrix4d T_reference_board = Eigen::Matrix4d::Identity();
  if (measurement.board_id != direct_layout.reference_board_id) {
    const OuterBootstrapBoardState* board =
        FindBootstrapBoardState(direct_layout, measurement.board_id);
    if (board == nullptr || !board->initialized) {
      return validation;
    }
    T_reference_board = board->T_reference_board;
  }

  DoubleSphereCameraModel camera_model;
  try {
    camera_model = DoubleSphereCameraModel::FromConfig(
        MakeIntermediateCameraConfig(camera));
  } catch (const std::exception&) {
    return validation;
  }

  double squared_error_sum = 0.0;
  for (std::size_t corner_index = 0; corner_index < 4; ++corner_index) {
    if (!measurement.refined_corner_valid[corner_index] ||
        !measurement.target_outer_corners_board[corner_index].allFinite() ||
        !measurement.refined_outer_corners_original_image[corner_index].allFinite()) {
      return validation;
    }
    const Eigen::Vector4d point_board(
        measurement.target_outer_corners_board[corner_index].x(),
        measurement.target_outer_corners_board[corner_index].y(),
        measurement.target_outer_corners_board[corner_index].z(), 1.0);
    const Eigen::Vector4d point_camera_h =
        frame->T_camera_reference * (T_reference_board * point_board);
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    if (!camera_model.vsEuclideanToKeypoint(point_camera_h.head<3>(),
                                             &predicted)) {
      return validation;
    }
    const Eigen::Vector2d residual =
        predicted - measurement.refined_outer_corners_original_image[corner_index];
    squared_error_sum += residual.squaredNorm();
  }
  validation.evaluable = true;
  validation.reprojection_rmse_px = std::sqrt(squared_error_sum / 4.0);
  return validation;
}

const OuterTagDetectionResult* FindDetectionByBoardId(
    const OuterTagMultiDetectionResult& detections,
    int board_id) {
  const auto it = std::find_if(
      detections.detections.begin(), detections.detections.end(),
      [board_id](const OuterTagDetectionResult& detection) {
        return detection.board_id == board_id;
      });
  return it == detections.detections.end() ? nullptr : &(*it);
}

const OuterBoardMeasurement* FindMeasurementByBoardId(
    const OuterTagMultiDetectionResult& detections,
    int board_id) {
  const auto& measurements = detections.frame_measurements.board_measurements;
  const auto it = std::find_if(
      measurements.begin(), measurements.end(),
      [board_id](const OuterBoardMeasurement& measurement) {
        return measurement.board_id == board_id;
      });
  return it == measurements.end() ? nullptr : &(*it);
}

void ReplaceDetectionByBoardId(
    const OuterTagDetectionResult& replacement,
    OuterTagMultiDetectionResult* detections) {
  if (detections == nullptr) {
    return;
  }
  for (OuterTagDetectionResult& detection : detections->detections) {
    if (detection.board_id == replacement.board_id) {
      detection = replacement;
      return;
    }
  }
}

// Keep decoder outputs as untrusted proposals until the regenerator validates
// them against the expected ID, image evidence, and the frame pose.
void MergeWrongIdProposalsByBoardId(
    const OuterTagDetectionResult& rescue_detection,
    OuterTagMultiDetectionResult* detections) {
  if (detections == nullptr || rescue_detection.wrong_id_proposals.empty()) {
    return;
  }
  for (OuterTagDetectionResult& detection : detections->detections) {
    if (detection.board_id != rescue_detection.board_id) {
      continue;
    }
    for (const OuterWrongIdProposal& proposal :
         rescue_detection.wrong_id_proposals) {
      const auto existing = std::find_if(
          detection.wrong_id_proposals.begin(),
          detection.wrong_id_proposals.end(),
          [&proposal](const OuterWrongIdProposal& candidate) {
            return candidate.detected_tag_id == proposal.detected_tag_id &&
                   candidate.source == proposal.source;
          });
      const bool proposal_is_better =
          existing == detection.wrong_id_proposals.end() ||
          proposal.hamming < existing->hamming ||
          (proposal.hamming == existing->hamming &&
           proposal.area_px > existing->area_px);
      if (!proposal_is_better) {
        continue;
      }
      if (existing == detection.wrong_id_proposals.end()) {
        detection.wrong_id_proposals.push_back(proposal);
      } else {
        *existing = proposal;
      }
    }
    return;
  }
}

void ReplaceMeasurementByBoardId(
    const OuterBoardMeasurement& replacement,
    OuterTagMultiDetectionResult* detections) {
  if (detections == nullptr) {
    return;
  }
  for (OuterBoardMeasurement& measurement :
       detections->frame_measurements.board_measurements) {
    if (measurement.board_id == replacement.board_id) {
      measurement = replacement;
      return;
    }
  }
}

OuterBoardMeasurement BuildRecoveryMeasurement(
    const OuterTagDetectionResult& detection) {
  OuterBoardMeasurement measurement;
  measurement.board_id = detection.board_id;
  measurement.detected_tag_id = detection.detected_tag_id;
  measurement.success = detection.success;
  measurement.attempted_local_patch_rescue =
      detection.attempted_local_patch_rescue;
  measurement.used_local_patch_rescue = detection.used_local_patch_rescue;
  measurement.local_patch_rescue_summary = detection.local_patch_rescue_summary;
  measurement.detection_quality = detection.quality;
  measurement.refined_outer_corners_original_image =
      detection.refined_corners_original_image;
  measurement.refined_corner_valid = detection.refined_valid;
  measurement.corner_verification_debug = detection.corner_verification_debug;
  measurement.failure_reason = detection.failure_reason;
  measurement.failure_reason_text = detection.failure_reason_text;
  measurement.valid_refined_corner_count = static_cast<int>(std::count(
      detection.refined_valid.begin(), detection.refined_valid.end(), true));
  return measurement;
}

bool IsExactTemporalAnchor(const OuterTagDetectionResult& detection,
                           int board_id) {
  // Direct full-image detections carry the expected decoded ID but do not
  // always populate the patch-only Hamming field (it is -1 in that path).
  // Such a detection is still a valid temporal search anchor when all four
  // refined corners are valid. Camera-aware patch anchors must continue to
  // satisfy Hamming 0; the current frame is always decoded again before any
  // temporal result can be committed.
  const bool exact_id_evidence =
      detection.hamming == 0 || detection.hamming < 0;
  return detection.success && detection.board_id == board_id &&
         detection.detected_tag_id == board_id && exact_id_evidence &&
         std::all_of(detection.refined_valid.begin(), detection.refined_valid.end(),
                     [](bool value) { return value; });
}

const OuterTagDetectionResult* FindExactTemporalAnchor(
    const std::vector<InternalRegenerationFrameInput>& inputs,
    std::size_t target_index,
    int board_id,
    int direction,
    std::size_t max_frame_gap,
    std::size_t* anchor_index) {
  if (anchor_index != nullptr) {
    *anchor_index = inputs.size();
  }
  for (std::size_t gap = 1; gap <= max_frame_gap; ++gap) {
    if ((direction < 0 && gap > target_index) ||
        (direction > 0 && target_index + gap >= inputs.size())) {
      break;
    }
    const std::size_t index = direction < 0 ? target_index - gap : target_index + gap;
    const OuterTagDetectionResult* detection =
        FindDetectionByBoardId(inputs[index].outer_detections, board_id);
    if (detection != nullptr && IsExactTemporalAnchor(*detection, board_id)) {
      if (anchor_index != nullptr) {
        *anchor_index = index;
      }
      return detection;
    }
  }
  return nullptr;
}

Eigen::Vector2d AverageOuterCorner(const OuterTagDetectionResult& detection) {
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  for (const Eigen::Vector2d& corner : detection.refined_corners_original_image) {
    center += corner;
  }
  return 0.25 * center;
}

}  // namespace

std::string MakeCameraAwareRescueSignature(
    const OuterBootstrapCameraIntrinsics& camera,
    const MultiScaleOuterTagDetectorConfig& config,
    int max_hamming,
    int reference_board_id) {
  return MakeCameraAwareRescueSignatureImpl(
      camera, config, max_hamming, reference_board_id);
}

void RunCameraAwareOuterRescue(
    const std::vector<FrozenRound2BaselineFrameSource>& frame_sources,
    const ApriltagInternalConfig& config,
    const OuterBootstrapCameraIntrinsics& provisional_camera,
    int reference_board_id,
    int max_hamming,
    int requested_worker_count,
    std::vector<OuterBootstrapFrameInput>* bootstrap_frames,
    std::vector<InternalRegenerationFrameInput>* regeneration_inputs,
    CameraAwareOuterRescueSummary* summary,
    OuterDetectionCache* rescue_cache,
    const std::function<void(std::size_t, std::size_t, const std::string&)>&
        progress_callback) {
  if (bootstrap_frames == nullptr || regeneration_inputs == nullptr ||
      summary == nullptr) {
    throw std::runtime_error(
        "RunCameraAwareOuterRescue requires valid output pointers.");
  }
  if (frame_sources.size() != bootstrap_frames->size() ||
      frame_sources.size() != regeneration_inputs->size()) {
    throw std::runtime_error(
        "Camera-aware outer rescue input vectors have inconsistent sizes.");
  }

  summary->enabled = true;
  summary->camera_family_supported = true;
  summary->camera_source =
      "stage5_provisional_outer_only_camera_initialization";
  summary->max_hamming = std::max(0, max_hamming);
  summary->frame_count = static_cast<int>(frame_sources.size());
  summary->provisional_camera = provisional_camera;
  summary->direct_layout_geometry_gate_enabled = true;

  MultiScaleOuterTagDetectorConfig rescue_config =
      config.outer_detector_config;
  rescue_config.refine_camera =
      BuildOuterRefineCameraConfig(provisional_camera);
  rescue_config.enable_camera_aware_sphere_patch_rescue = true;
  rescue_config.camera_aware_sphere_patch_max_hamming = summary->max_hamming;
  rescue_config.camera_aware_sphere_patch_commit_mapped_corners = true;
  summary->zero_detection_atlas_enabled =
      rescue_config.camera_aware_sphere_patch_rescue_zero_detection_frames;
  if (summary->zero_detection_atlas_enabled &&
      rescue_config.camera_aware_sphere_patch_use_extended_atlas) {
    summary->patch_plan =
        "dense_5x5_fov56_plus_wide_3x3_fov72_plus_extended_boundary_atlas_plus_zero_fine_9x9_fov42";
  } else if (summary->zero_detection_atlas_enabled) {
    summary->patch_plan =
        "dense_5x5_fov56_plus_wide_3x3_fov72_plus_zero_fine_9x9_fov42";
  } else if (rescue_config.camera_aware_sphere_patch_use_extended_atlas) {
    summary->patch_plan =
        "dense_5x5_fov56_plus_wide_3x3_fov72_plus_extended_boundary_atlas";
  }
  const std::string rescue_signature = MakeCameraAwareRescueSignature(
      provisional_camera, rescue_config, summary->max_hamming,
      reference_board_id);
  if (rescue_cache != nullptr && rescue_cache->enabled() &&
      !frame_sources.empty()) {
    std::string cache_warning;
    rescue_cache->PrepareForDataset(frame_sources.front().image_path,
                                    &cache_warning);
  }

  struct RescueAttempt {
    std::size_t frame_index = 0;
    bool baseline_has_zero_detections = false;
    std::vector<int> missing_board_ids;
    bool image_read_success = false;
    OuterTagMultiDetectionResult rescued;
  };
  std::vector<RescueAttempt> attempts;
  attempts.reserve(frame_sources.size());
  for (std::size_t frame_index = 0; frame_index < frame_sources.size();
       ++frame_index) {
    const OuterTagMultiDetectionResult& baseline =
        (*regeneration_inputs)[frame_index].outer_detections;
    summary->requested_board_observation_count +=
        static_cast<int>(baseline.frame_measurements.board_measurements.size());
    summary->baseline_success_count += baseline.SuccessfulBoardCount();
    if (baseline.SuccessfulBoardCount() ==
        static_cast<int>(baseline.requested_board_ids.size())) {
      ++summary->baseline_all_boards_frame_count;
    }
    const bool baseline_has_zero_detections =
        baseline.SuccessfulBoardCount() == 0;
    if (baseline_has_zero_detections) {
      ++summary->zero_detection_frame_count;
    }

    // Reuse only the stage-local rescue artifact keyed by this exact
    // provisional-camera signature. Raw direct detections remain untouched.
    if (rescue_cache != nullptr && rescue_cache->enabled()) {
      OuterTagMultiDetectionResult cached_rescue;
      std::string cache_warning;
      if (rescue_cache->Load(frame_sources[frame_index].image_path,
                             &cached_rescue, &cache_warning, nullptr)) {
        int cached_rescued_boards = 0;
        for (const OuterBoardMeasurement& baseline_measurement :
             baseline.frame_measurements.board_measurements) {
          if (baseline_measurement.success) {
            continue;
          }
          const OuterBoardMeasurement* cached_measurement =
              FindMeasurementByBoardId(cached_rescue, baseline_measurement.board_id);
          if (cached_measurement != nullptr && cached_measurement->success &&
              cached_measurement->used_local_patch_rescue) {
            ++cached_rescued_boards;
          }
        }
        ++summary->attempted_frame_count;
        summary->attempted_board_observation_count +=
            static_cast<int>(baseline.frame_measurements.board_measurements.size()) -
            baseline.SuccessfulBoardCount();
        summary->rescued_board_observation_count += cached_rescued_boards;
        summary->temporal_seed_rescued_board_observation_count +=
            std::count_if(
                cached_rescue.detections.begin(), cached_rescue.detections.end(),
                [](const OuterTagDetectionResult& detection) {
                  return detection.success && detection.used_local_patch_rescue &&
                         detection.local_patch_rescue_summary.find("temporal_") !=
                             std::string::npos;
                });
        (*regeneration_inputs)[frame_index].outer_detections =
            std::move(cached_rescue);
        (*bootstrap_frames)[frame_index].measurements =
            (*regeneration_inputs)[frame_index].outer_detections
                .frame_measurements;
        continue;
      }
    }

    // Camera-aware rescue is a frozen frontend artifact for this image and
    // detector configuration.  Do not key reuse on the provisional camera:
    // accepting recovered observations can change the subsequent initializer
    // and therefore change that camera on the next run, which otherwise makes
    // every cached rescue appear stale and triggers another atlas scan.  The
    // cache format/configuration version is the explicit invalidation point
    // when the rescue algorithm or atlas changes.
    if (baseline.camera_aware_rescue_attempted &&
        baseline.camera_aware_rescue_signature == rescue_signature) {
      continue;
    }

    std::vector<int> missing_board_ids;
    for (const OuterBoardMeasurement& measurement :
         baseline.frame_measurements.board_measurements) {
      if (!measurement.success) {
        missing_board_ids.push_back(measurement.board_id);
      }
    }
    if (missing_board_ids.empty()) {
      continue;
    }
    ++summary->attempted_frame_count;
    summary->attempted_board_observation_count +=
        static_cast<int>(missing_board_ids.size());
    RescueAttempt attempt;
    attempt.frame_index = frame_index;
    attempt.baseline_has_zero_detections = baseline_has_zero_detections;
    attempt.missing_board_ids = std::move(missing_board_ids);
    attempts.push_back(std::move(attempt));
  }

  if (!rescue_config.camera_aware_sphere_patch_atlas &&
      !frame_sources.empty() && !attempts.empty()) {
    const cv::Size image_size = cv::imread(
        frame_sources.front().image_path, cv::IMREAD_GRAYSCALE).size();
    rescue_config.camera_aware_sphere_patch_atlas =
        BuildCameraAwareSpherePatchAtlas(
            rescue_config.refine_camera, image_size,
            rescue_config.camera_aware_sphere_patch_use_extended_atlas,
            false);
    const bool has_zero_detection_attempt = std::any_of(
        attempts.begin(), attempts.end(),
        [](const RescueAttempt& attempt) {
          return attempt.baseline_has_zero_detections;
        });
    if (has_zero_detection_attempt &&
        rescue_config.camera_aware_sphere_patch_rescue_zero_detection_frames) {
      rescue_config.camera_aware_sphere_patch_zero_detection_atlas =
          BuildCameraAwareSpherePatchAtlas(
              rescue_config.refine_camera, image_size,
              rescue_config.camera_aware_sphere_patch_use_extended_atlas, true);
    }
  }

  // A cached generic-rescue result must not suppress the lightweight temporal
  // pass below.  The cache records that the generic atlas was attempted, but
  // it may contain a failed board that is recoverable from exact anchors on
  // both neighboring frames.  Keep the generic worker block conditional on
  // fresh attempts and always let temporal recovery inspect unresolved boards.

  // Build the rig only from direct full-image detections before accepting any
  // sphere-patch recovery. A patch can decode the expected ID exactly while
  // still map a heavily distorted quadrilateral to the wrong image location.
  const std::vector<OuterBootstrapFrameInput> direct_layout_frames =
      BuildDirectOuterBootstrapFrames(*bootstrap_frames);
  OuterBootstrapResult direct_layout;
  if (!direct_layout_frames.empty()) {
    OuterBootstrapOptions direct_layout_options;
    direct_layout_options.reference_board_id = reference_board_id;
    direct_layout_options.initial_camera = provisional_camera;
    direct_layout_options.max_coordinate_descent_iterations = 0;
    direct_layout_options.min_detection_quality =
        config.outer_detector_config.min_detection_quality;
    const MultiBoardOuterBootstrap direct_layout_bootstrap(
        config, direct_layout_options);
    direct_layout = direct_layout_bootstrap.Solve(direct_layout_frames);
    summary->direct_layout_geometry_gate_available = direct_layout.success;
    if (!direct_layout.success) {
      AppendUniqueWarning(
          "Camera-aware rescue direct-layout geometry gate unavailable: " +
              direct_layout.failure_reason,
          &summary->warnings);
    }
  }

  // Local sphere-patch recovery is frame-local. Each worker owns an independent
  // detector; the ordered merge below preserves the previous observable order.
  const unsigned int hardware_workers = std::thread::hardware_concurrency();
  const std::size_t automatic_worker_count = std::max<std::size_t>(
      1, std::min<std::size_t>(4, hardware_workers == 0 ? 1 : hardware_workers));
  const std::size_t requested_workers = requested_worker_count > 0
      ? static_cast<std::size_t>(requested_worker_count)
      : automatic_worker_count;
  const std::size_t worker_count = std::min<std::size_t>(
      attempts.size(), std::max<std::size_t>(1, requested_workers));
  summary->worker_count = static_cast<int>(worker_count == 0 ? 1 : worker_count);
  if (!attempts.empty()) {
    std::atomic<std::size_t> next_attempt(0);
    std::atomic<std::size_t> completed_attempts(0);
    const auto detect_attempts = [&]() {
      const MultiScaleOuterTagDetector rescue_detector(rescue_config);
      for (;;) {
        const std::size_t attempt_index = next_attempt.fetch_add(1);
        if (attempt_index >= attempts.size()) {
          return;
        }
        RescueAttempt& attempt = attempts[attempt_index];
        const auto report_attempt = [&](const std::string& detail) {
          if (progress_callback) {
            const std::size_t completed = completed_attempts.fetch_add(1) + 1;
            progress_callback(completed, attempts.size(), detail);
          }
        };
        const cv::Mat image = cv::imread(
            frame_sources[attempt.frame_index].image_path, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
          report_attempt("image_read_failed frame=" +
                        frame_sources[attempt.frame_index].frame_label);
          continue;
        }
        attempt.rescued = rescue_detector.DetectMultiple(image);
        attempt.image_read_success = true;
        report_attempt("frame=" + frame_sources[attempt.frame_index].frame_label);
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker_index = 0; worker_index < worker_count;
         ++worker_index) {
      workers.emplace_back(detect_attempts);
    }
    for (std::thread& worker : workers) {
      worker.join();
    }
  }

  // Preserve the previous observable order for warnings, records, and commits.
  for (const RescueAttempt& attempt : attempts) {
    const std::size_t frame_index = attempt.frame_index;
    if (!attempt.image_read_success) {
      AppendUniqueWarning(
          "Camera-aware outer rescue failed to read image: " +
              frame_sources[frame_index].image_path,
          &summary->warnings);
      continue;
    }
    const OuterTagMultiDetectionResult& rescued = attempt.rescued;
    OuterTagMultiDetectionResult* committed =
        &(*regeneration_inputs)[frame_index].outer_detections;
    for (int board_id : attempt.missing_board_ids) {
      const OuterTagDetectionResult* rescue_detection =
          FindDetectionByBoardId(rescued, board_id);
      const OuterBoardMeasurement* rescue_measurement =
          FindMeasurementByBoardId(rescued, board_id);
      const OuterTagDetectionResult* baseline_detection =
          FindDetectionByBoardId(*committed, board_id);
      if (attempt.baseline_has_zero_detections && rescue_detection != nullptr &&
          rescue_detection->attempted_local_patch_rescue) {
        ++summary->zero_detection_atlas_attempted_board_observation_count;
      }
      if (rescue_detection != nullptr) {
        MergeWrongIdProposalsByBoardId(*rescue_detection, committed);
      }
      if (rescue_detection == nullptr || rescue_measurement == nullptr ||
          !rescue_detection->success ||
          !rescue_detection->used_local_patch_rescue ||
          rescue_detection->hamming < 0 ||
          rescue_detection->hamming > summary->max_hamming) {
        continue;
      }

      const DirectLayoutRescueValidation geometry_validation =
          ValidatePatchRescueAgainstDirectLayout(
              direct_layout, provisional_camera,
              frame_sources[frame_index].frame_index, *rescue_measurement);
      if (geometry_validation.evaluable) {
        ++summary->direct_layout_geometry_gate_evaluated_count;
        if (geometry_validation.reprojection_rmse_px >
            summary->direct_layout_geometry_gate_max_rmse_px) {
          ++summary->direct_layout_geometry_gate_rejected_count;
          continue;
        }
        ++summary->direct_layout_geometry_gate_accepted_count;
      } else {
        // Zero-detection frames have no direct same-frame anchor. Preserve the
        // existing exact-ID recovery path for them; global bootstrap remains
        // responsible for rejecting inconsistent observations.
        ++summary->direct_layout_geometry_gate_not_evaluable_count;
      }

      const std::string baseline_failure_reason =
          baseline_detection != nullptr
              ? baseline_detection->failure_reason_text
              : "missing_detector_result";
      ReplaceDetectionByBoardId(*rescue_detection, committed);
      ReplaceMeasurementByBoardId(*rescue_measurement, committed);
      CameraAwareOuterRescueRecord record;
      record.frame_index = frame_sources[frame_index].frame_index;
      record.frame_label = frame_sources[frame_index].frame_label;
      record.board_id = board_id;
      record.baseline_failure_reason = baseline_failure_reason;
      record.rescue_summary = rescue_detection->local_patch_rescue_summary;
      record.hamming = rescue_detection->hamming;
      record.committed_corners = rescue_detection->refined_corners_original_image;
      summary->records.push_back(record);
      ++summary->rescued_board_observation_count;
    }
    (*bootstrap_frames)[frame_index].measurements =
        committed->frame_measurements;
  }

  // A generic atlas can miss one otherwise well-observed frame when the tag
  // straddles two patch centres.  When the same board is exactly decoded on
  // both nearby frames, interpolate only the DS patch centre and run another
  // exact-ID decode.  The temporal information is never itself an
  // observation: a result is committed only after the current image decodes
  // the expected ID at the normal Hamming threshold.
  constexpr std::size_t kTemporalSeedMaxFrameGap = 4;
  const MultiScaleOuterTagDetector temporal_detector(rescue_config);
  for (std::size_t frame_index = 0; frame_index < frame_sources.size();
       ++frame_index) {
    OuterTagMultiDetectionResult* committed =
        &(*regeneration_inputs)[frame_index].outer_detections;
    // A cached generic-rescue result may still have unresolved boards.  Do
    // not use the stage-level attempted bit as a board-level success bit:
    // cached incomplete frames must still get the lightweight temporal pass.
    std::vector<int> unresolved_board_ids;
    for (const OuterBoardMeasurement& measurement :
         committed->frame_measurements.board_measurements) {
      if (!measurement.success) {
        unresolved_board_ids.push_back(measurement.board_id);
      }
    }
    if (unresolved_board_ids.empty()) {
      continue;
    }

    const cv::Mat image = cv::imread(frame_sources[frame_index].image_path,
                                     cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      continue;
    }
    for (const int board_id : unresolved_board_ids) {
      std::size_t previous_index = regeneration_inputs->size();
      std::size_t next_index = regeneration_inputs->size();
      const OuterTagDetectionResult* previous = FindExactTemporalAnchor(
          *regeneration_inputs, frame_index, board_id, -1,
          kTemporalSeedMaxFrameGap, &previous_index);
      const OuterTagDetectionResult* next = FindExactTemporalAnchor(
          *regeneration_inputs, frame_index, board_id, 1,
          kTemporalSeedMaxFrameGap, &next_index);
      if (previous == nullptr || next == nullptr) {
        continue;
      }

      const double alpha = static_cast<double>(frame_index - previous_index) /
                           static_cast<double>(next_index - previous_index);
      const Eigen::Vector2d center =
          (1.0 - alpha) * AverageOuterCorner(*previous) +
          alpha * AverageOuterCorner(*next);
      const std::string label =
          "temporal_prev" + std::to_string(frame_sources[previous_index].frame_index) +
          "_next" + std::to_string(frame_sources[next_index].frame_index);
      const std::vector<CameraAwareSpherePatchSeed> seeds{
          CameraAwareSpherePatchSeed{label + "_wide", center, 72.0},
          CameraAwareSpherePatchSeed{label + "_extended_wide", center, 86.0},
          CameraAwareSpherePatchSeed{label + "_ultra", center, 108.0},
      };
      ++summary->temporal_seed_attempted_board_observation_count;
      const OuterTagDetectionResult recovered =
          temporal_detector.DetectTargetedSpherePatch(image, board_id, seeds);
      if (!recovered.success || !recovered.used_local_patch_rescue ||
          recovered.detected_tag_id != board_id || recovered.hamming < 0 ||
          recovered.hamming > summary->max_hamming) {
        continue;
      }
      const OuterBoardMeasurement recovered_measurement =
          BuildRecoveryMeasurement(recovered);
      const DirectLayoutRescueValidation geometry_validation =
          ValidatePatchRescueAgainstDirectLayout(
              direct_layout, provisional_camera,
              frame_sources[frame_index].frame_index, recovered_measurement);
      if (geometry_validation.evaluable) {
        ++summary->direct_layout_geometry_gate_evaluated_count;
        if (geometry_validation.reprojection_rmse_px >
            summary->direct_layout_geometry_gate_max_rmse_px) {
          ++summary->direct_layout_geometry_gate_rejected_count;
          continue;
        }
        ++summary->direct_layout_geometry_gate_accepted_count;
      } else {
        ++summary->direct_layout_geometry_gate_not_evaluable_count;
      }

      const OuterTagDetectionResult* baseline_detection =
          FindDetectionByBoardId(*committed, board_id);
      const std::string baseline_failure_reason =
          baseline_detection != nullptr
              ? baseline_detection->failure_reason_text
              : "missing_detector_result";
      ReplaceDetectionByBoardId(recovered, committed);
      ReplaceMeasurementByBoardId(recovered_measurement, committed);
      CameraAwareOuterRescueRecord record;
      record.frame_index = frame_sources[frame_index].frame_index;
      record.frame_label = frame_sources[frame_index].frame_label;
      record.board_id = board_id;
      record.baseline_failure_reason = baseline_failure_reason;
      record.rescue_summary = recovered.local_patch_rescue_summary;
      record.hamming = recovered.hamming;
      record.committed_corners = recovered.refined_corners_original_image;
      summary->records.push_back(record);
      ++summary->rescued_board_observation_count;
      ++summary->temporal_seed_rescued_board_observation_count;
    }
    (*bootstrap_frames)[frame_index].measurements = committed->frame_measurements;
  }

  for (const RescueAttempt& attempt : attempts) {
    if (!attempt.image_read_success) {
      continue;
    }
    OuterTagMultiDetectionResult& final_detection =
        (*regeneration_inputs)[attempt.frame_index].outer_detections;
    final_detection.camera_aware_rescue_attempted = true;
    final_detection.camera_aware_rescue_signature = rescue_signature;
  }

  if (rescue_cache != nullptr && rescue_cache->enabled()) {
    for (const RescueAttempt& attempt : attempts) {
      if (!attempt.image_read_success) {
        continue;
      }
      std::string cache_warning;
      rescue_cache->Save(
          frame_sources[attempt.frame_index].image_path,
          (*regeneration_inputs)[attempt.frame_index].outer_detections,
          &cache_warning);
    }
  }

  for (const InternalRegenerationFrameInput& input : *regeneration_inputs) {
    const OuterTagMultiDetectionResult& final_detection = input.outer_detections;
    summary->final_success_count += final_detection.SuccessfulBoardCount();
    if (final_detection.SuccessfulBoardCount() ==
        static_cast<int>(final_detection.requested_board_ids.size())) {
      ++summary->final_all_boards_frame_count;
    }
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
