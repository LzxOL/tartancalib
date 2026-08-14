#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>

#include <aslam/cameras/apriltag_internal/OuterDetectionResultUtils.hpp>
#include <aslam/cameras/apriltag_internal/GeometryPriorTopology.hpp>
#include <aslam/cameras/apriltag_internal/GeometryPriorOuterRecovery.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
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

void CollectVisibleBoardIds(
    const InternalRegenerationFrameInput& frame_input,
    std::vector<int>* board_ids) {
  if (board_ids == nullptr) {
    return;
  }
  for (std::size_t index = 0;
       index < frame_input.outer_detections.requested_board_ids.size();
       ++index) {
    if (index < frame_input.outer_detections.detections.size() &&
        frame_input.outer_detections.detections[index].success) {
      AppendUniqueBoardId(
          frame_input.outer_detections.requested_board_ids[index], board_ids);
    }
  }
}

// A single board provides no same-frame layout consensus. It may seed
// geometry-guided recovery only when it is a genuine, exact-ID detector
// observation with four verified corners, rather than a prior rescue.

bool IsExactImageValidatedOuterMeasurement(
    const OuterBoardMeasurement& measurement) {
  if (!measurement.success || !measurement.used_local_patch_rescue ||
      measurement.detected_tag_id != measurement.board_id ||
      measurement.valid_refined_corner_count != 4) {
    return false;
  }
  // The outer-corner target geometry is always reconstructed from the board
  // configuration at the call site. The regular detector-side measurement
  // intentionally does not carry a duplicate target-corner array, so requiring
  // has_target_outer_corners here silently discarded exact sphere-patch
  // recoveries before the local-frame pose refit.
  return std::all_of(
      measurement.refined_corner_valid.begin(),
      measurement.refined_corner_valid.end(),
      [](bool valid) { return valid; });
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

struct LocalFramePoseRefit {
  bool success = false;
  int source_board_id = -1;
  std::vector<int> support_board_ids;
  struct Candidate {
    int source_board_id = -1;
    double outer_rmse = std::numeric_limits<double>::infinity();
    Eigen::Matrix4d T_camera_reference = Eigen::Matrix4d::Identity();
  };
  std::vector<Candidate> candidates;
  double outer_rmse = 0.0;
  Eigen::Matrix4d T_camera_reference = Eigen::Matrix4d::Identity();
};


double MedianFinite(std::vector<double> values) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](double value) {
                                return !std::isfinite(value);
                              }),
               values.end());
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1] + values[mid]);
}

double ComputeFrameNormalOuterRefitRmseMedian(
    const ApriltagInternalConfig& config,
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const InternalRegenerationFrameInput& frame_input) {
  std::vector<double> rmses;
  for (const OuterBoardMeasurement& measurement :
       frame_input.outer_detections.frame_measurements.board_measurements) {
    const bool exact_image_validated_rescue =
        IsExactImageValidatedOuterMeasurement(measurement);
    if (!measurement.success ||
        (measurement.used_local_patch_rescue &&
         !exact_image_validated_rescue) ||
        measurement.valid_refined_corner_count < 4) {
      continue;
    }
    bool all_corners_valid = true;
    for (bool valid : measurement.refined_corner_valid) {
      all_corners_valid = all_corners_valid && valid;
    }
    if (!all_corners_valid) {
      continue;
    }
    const std::array<Eigen::Vector3d, 4> object_points_array =
        BuildOuterCornerPointsForBoard(config, measurement.board_id);
    Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
    double outer_rmse = 0.0;
    if (EstimatePoseFromObjectPoints(
            intrinsics, ToVector(object_points_array),
            ToImagePoints(measurement.refined_outer_corners_original_image),
            &T_camera_board, &outer_rmse) &&
        std::isfinite(outer_rmse)) {
      rmses.push_back(outer_rmse);
    }
  }
  return MedianFinite(rmses);
}

LocalFramePoseRefit EstimateLocalFramePoseFromVisibleBoards(
    const ApriltagInternalConfig& config,
    const OuterBootstrapCameraIntrinsics& intrinsics,
    const InternalRegenerationFrameInput& frame_input,
    const std::function<bool(int, Eigen::Matrix4d*)>& lookup_reference_board) {
  struct BoardPoseSupport {
    int board_id = -1;
    double local_outer_rmse = std::numeric_limits<double>::infinity();
    Eigen::Isometry3d T_camera_reference = Eigen::Isometry3d::Identity();
    std::vector<Eigen::Vector3d> object_points_reference;
    std::vector<cv::Point2f> image_points;
  };

  LocalFramePoseRefit best;
  std::vector<BoardPoseSupport> supports;
  for (const OuterBoardMeasurement& measurement :
       frame_input.outer_detections.frame_measurements.board_measurements) {
    const bool exact_image_validated_rescue =
        IsExactImageValidatedOuterMeasurement(measurement);
    if (!measurement.success ||
        (measurement.used_local_patch_rescue &&
         !exact_image_validated_rescue) ||
        measurement.valid_refined_corner_count < 4) {
      continue;
    }
    bool all_corners_valid = true;
    for (bool valid : measurement.refined_corner_valid) {
      all_corners_valid = all_corners_valid && valid;
    }
    if (!all_corners_valid) {
      continue;
    }

    Eigen::Matrix4d T_reference_board_matrix = Eigen::Matrix4d::Identity();
    if (!lookup_reference_board(measurement.board_id,
                                &T_reference_board_matrix)) {
      continue;
    }

    const std::array<Eigen::Vector3d, 4> object_points_array =
        BuildOuterCornerPointsForBoard(config, measurement.board_id);
    Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
    double outer_rmse = 0.0;
    if (!EstimatePoseFromObjectPoints(
            intrinsics, ToVector(object_points_array),
            ToImagePoints(measurement.refined_outer_corners_original_image),
            &T_camera_board, &outer_rmse)) {
      continue;
    }

    const Eigen::Isometry3d T_reference_board =
        ToIsometry3d(T_reference_board_matrix);
    const Eigen::Isometry3d T_camera_reference =
        T_camera_board * T_reference_board.inverse();
    LocalFramePoseRefit::Candidate pose_candidate;
    pose_candidate.source_board_id = measurement.board_id;
    pose_candidate.outer_rmse = outer_rmse;
    pose_candidate.T_camera_reference = T_camera_reference.matrix();
    best.candidates.push_back(pose_candidate);
    BoardPoseSupport support;
    support.board_id = measurement.board_id;
    support.local_outer_rmse = outer_rmse;
    support.T_camera_reference = T_camera_reference;
    support.image_points =
        ToImagePoints(measurement.refined_outer_corners_original_image);
    support.object_points_reference.reserve(object_points_array.size());
    for (const Eigen::Vector3d& object_point : object_points_array) {
      support.object_points_reference.push_back(
          T_reference_board * object_point);
    }
    supports.push_back(std::move(support));
    if (!best.success || outer_rmse < best.outer_rmse) {
      best.success = true;
      best.source_board_id = measurement.board_id;
      best.support_board_ids = {measurement.board_id};
      best.outer_rmse = outer_rmse;
      best.T_camera_reference = T_camera_reference.matrix();
    }
  }

  std::vector<const BoardPoseSupport*> locally_valid_supports;
  for (const BoardPoseSupport& support : supports) {
    if (std::isfinite(support.local_outer_rmse) &&
        support.local_outer_rmse < 3.0) {
      locally_valid_supports.push_back(&support);
    }
  }
  if (locally_valid_supports.size() < 2) {
    return best;
  }

  const auto fit_supports =
      [&](const std::vector<const BoardPoseSupport*>& fit_set,
          Eigen::Isometry3d* pose, double* rmse) {
        std::vector<Eigen::Vector3d> object_points;
        std::vector<cv::Point2f> image_points;
        object_points.reserve(fit_set.size() * 4);
        image_points.reserve(fit_set.size() * 4);
        for (const BoardPoseSupport* support : fit_set) {
          object_points.insert(object_points.end(),
                               support->object_points_reference.begin(),
                               support->object_points_reference.end());
          image_points.insert(image_points.end(), support->image_points.begin(),
                              support->image_points.end());
        }
        return EstimatePoseFromObjectPoints(intrinsics, object_points,
                                            image_points, pose, rmse);
      };

  IntermediateCameraConfig camera_config;
  camera_config.camera_model = intrinsics.camera_model;
  camera_config.distortion_model = intrinsics.distortion_model;
  camera_config.intrinsics = intrinsics.IntrinsicsVector();
  camera_config.distortion_coeffs = intrinsics.DistortionVector();
  camera_config.resolution = {intrinsics.resolution.width,
                              intrinsics.resolution.height};
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(camera_config);
  if (!camera.IsValid()) {
    return best;
  }

  const auto board_rmse =
      [&](const Eigen::Isometry3d& pose, const BoardPoseSupport& support) {
        double squared_error_sum = 0.0;
        int valid_count = 0;
        for (std::size_t index = 0;
             index < support.object_points_reference.size(); ++index) {
          Eigen::Vector2d projected;
          if (!camera.vsEuclideanToKeypoint(
                  pose * support.object_points_reference[index], &projected)) {
            continue;
          }
          const cv::Point2f& observed = support.image_points[index];
          squared_error_sum +=
              (projected - Eigen::Vector2d(observed.x, observed.y)).squaredNorm();
          ++valid_count;
        }
        return valid_count > 0
                   ? std::sqrt(squared_error_sum /
                               static_cast<double>(valid_count))
                   : std::numeric_limits<double>::infinity();
      };

  struct PoseConsensusCandidate {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double fit_rmse = std::numeric_limits<double>::infinity();
    double median_board_rmse = std::numeric_limits<double>::infinity();
  };
  std::vector<PoseConsensusCandidate> pose_candidates;
  const auto append_pose_candidate =
      [&](const Eigen::Isometry3d& pose, double fit_rmse) {
        std::vector<double> board_rmses;
        board_rmses.reserve(locally_valid_supports.size());
        for (const BoardPoseSupport* support : locally_valid_supports) {
          board_rmses.push_back(board_rmse(pose, *support));
        }
        PoseConsensusCandidate candidate;
        candidate.pose = pose;
        candidate.fit_rmse = fit_rmse;
        candidate.median_board_rmse = MedianFinite(board_rmses);
        if (std::isfinite(candidate.median_board_rmse)) {
          pose_candidates.push_back(candidate);
        }
      };

  for (const BoardPoseSupport* support : locally_valid_supports) {
    append_pose_candidate(support->T_camera_reference,
                          support->local_outer_rmse);
  }
  Eigen::Isometry3d all_pose = Eigen::Isometry3d::Identity();
  double all_rmse = 0.0;
  if (fit_supports(locally_valid_supports, &all_pose, &all_rmse)) {
    append_pose_candidate(all_pose, all_rmse);
  }
  if (locally_valid_supports.size() >= 3) {
    for (std::size_t excluded = 0; excluded < locally_valid_supports.size();
         ++excluded) {
      std::vector<const BoardPoseSupport*> subset;
      subset.reserve(locally_valid_supports.size() - 1);
      for (std::size_t index = 0; index < locally_valid_supports.size();
           ++index) {
        if (index != excluded) {
          subset.push_back(locally_valid_supports[index]);
        }
      }
      Eigen::Isometry3d subset_pose = Eigen::Isometry3d::Identity();
      double subset_rmse = 0.0;
      if (fit_supports(subset, &subset_pose, &subset_rmse)) {
        append_pose_candidate(subset_pose, subset_rmse);
      }
    }
  }
  if (pose_candidates.empty()) {
    return best;
  }
  const PoseConsensusCandidate* consensus = &pose_candidates.front();
  for (const PoseConsensusCandidate& candidate : pose_candidates) {
    if (candidate.median_board_rmse < consensus->median_board_rmse) {
      consensus = &candidate;
    }
  }

  std::vector<double> consensus_board_rmses;
  consensus_board_rmses.reserve(locally_valid_supports.size());
  for (const BoardPoseSupport* support : locally_valid_supports) {
    consensus_board_rmses.push_back(board_rmse(consensus->pose, *support));
  }
  const double median_rmse = MedianFinite(consensus_board_rmses);
  std::vector<double> absolute_deviations;
  absolute_deviations.reserve(consensus_board_rmses.size());
  for (double value : consensus_board_rmses) {
    absolute_deviations.push_back(std::abs(value - median_rmse));
  }
  const double mad = MedianFinite(absolute_deviations);
  const double inlier_threshold =
      std::max(3.0, median_rmse + std::max(0.75, 3.0 * 1.4826 * mad));
  std::vector<const BoardPoseSupport*> inlier_supports;
  for (std::size_t index = 0; index < locally_valid_supports.size(); ++index) {
    if (consensus_board_rmses[index] <= inlier_threshold) {
      inlier_supports.push_back(locally_valid_supports[index]);
    }
  }
  if (inlier_supports.size() < 2) {
    return best;
  }

  Eigen::Isometry3d joint_pose = consensus->pose;
  double joint_rmse = consensus->fit_rmse;
  Eigen::Isometry3d refit_pose = Eigen::Isometry3d::Identity();
  double refit_rmse = 0.0;
  if (fit_supports(inlier_supports, &refit_pose, &refit_rmse)) {
    joint_pose = refit_pose;
    joint_rmse = refit_rmse;
  }
  best.success = true;
  best.T_camera_reference = joint_pose.matrix();
  best.outer_rmse = joint_rmse;
  best.support_board_ids.clear();
  for (const BoardPoseSupport* support : inlier_supports) {
    best.support_board_ids.push_back(support->board_id);
  }
  return best;
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
  detection.internal_camera_source = "missing";
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

void AccumulateRuntimeBreakdown(
    const ApriltagInternalRuntimeBreakdown& detection_runtime,
    InternalRegenerationRuntimeBreakdown* frame_runtime) {
  if (frame_runtime == nullptr) {
    return;
  }
  frame_runtime->pose_estimation_seconds +=
      detection_runtime.pose_estimation_seconds;
  frame_runtime->boundary_model_seconds +=
      detection_runtime.boundary_model_seconds;
  frame_runtime->seed_search_seconds += detection_runtime.seed_search_seconds;
  frame_runtime->ray_refine_seconds += detection_runtime.ray_refine_seconds;
  frame_runtime->image_evidence_seconds +=
      detection_runtime.image_evidence_seconds;
  frame_runtime->subpix_seconds += detection_runtime.subpix_seconds;
  frame_runtime->pose_estimation_call_count +=
      detection_runtime.pose_estimation_call_count;
  frame_runtime->pose_rescue_attempt_count +=
      detection_runtime.pose_rescue_attempt_count;
  frame_runtime->pose_rescue_success_count +=
      detection_runtime.pose_rescue_success_count;
  frame_runtime->pose_rescue_used_count +=
      detection_runtime.pose_rescue_used_count;
  frame_runtime->boundary_model_build_count +=
      detection_runtime.boundary_model_build_count;
  frame_runtime->attempted_internal_corner_count +=
      detection_runtime.attempted_internal_corner_count;
  frame_runtime->valid_internal_corner_count +=
      detection_runtime.valid_internal_corner_count;
}

struct GeometryPriorPrediction {
  Eigen::Matrix4d T_camera_board = Eigen::Matrix4d::Identity();
  std::string source_label;
  int frame_pose_refit_source_board_id = -1;
  double frame_pose_refit_outer_rmse = 0.0;
  std::vector<int> visible_boards_used;
  std::vector<std::array<Eigen::Vector2d, 4>> competing_topology_slots;
  const OuterWrongIdProposal* wrong_id_proposal = nullptr;
};

struct GeometryPriorRecoverySelection {
  bool any_candidate_accepted = false;
  bool selected_rescued_detection = false;
  OuterTagDetectionResult detection;
  GeometryPriorOuterSeedCandidate selected_candidate;
};

// When several pose priors explain the same missing board, prefer independent
// image evidence over a lower-RMSE topology-only hypothesis.  The latter can
// be geometrically self-consistent while still placing all four corners on a
// nearby edge/structure, which is exactly the failure mode seen in extreme
// fisheye frames. Within a tier, payload and local image quality are compared
// before RMSE.
int GeometryPriorEvidenceRank(const GeometryPriorOuterSeedCandidate& candidate) {
  if (candidate.roi_redetect_success || candidate.rectified_patch_decode_success) {
    return 3;  // locally decoded expected tag ID
  }
  if (candidate.geometry_guided_tag_likelihood_passed ||
      candidate.tag_id_validated) {
    return 2;  // DS/model-aware code evidence passed
  }
  if (candidate.prediction_source_label.find("visible_refit") !=
      std::string::npos) {
    return 1;  // topology/pose-only recovery
  }
  return 0;
}

int GeometryPriorExpectedHamming(
    const GeometryPriorOuterSeedCandidate& candidate) {
  if (candidate.geometry_guided_tag_likelihood_checked) {
    return candidate.geometry_guided_tag_likelihood_expected_hamming;
  }
  if (candidate.rectified_patch_decode_success) {
    return candidate.rectified_patch_hamming;
  }
  if (candidate.roi_redetect_success) {
    return candidate.roi_redetect_hamming;
  }
  return std::numeric_limits<int>::max();
}

bool PreferGeometryPriorCandidate(
    const GeometryPriorOuterSeedCandidate& candidate,
    const GeometryPriorOuterSeedCandidate& current,
    bool has_current) {
  if (!has_current) {
    return true;
  }
  const int candidate_rank = GeometryPriorEvidenceRank(candidate);
  const int current_rank = GeometryPriorEvidenceRank(current);
  if (candidate_rank != current_rank) {
    return candidate_rank > current_rank;
  }

  // Within the same evidence tier, compare the ID evidence itself before pose
  // RMSE. This prevents two accepted model-aware patches from being ordered by
  // geometry alone when one has a cleaner payload match.
  const int candidate_hamming = GeometryPriorExpectedHamming(candidate);
  const int current_hamming = GeometryPriorExpectedHamming(current);
  if (candidate_hamming != current_hamming) {
    return candidate_hamming < current_hamming;
  }
  if (candidate.geometry_guided_tag_likelihood_hamming_margin !=
      current.geometry_guided_tag_likelihood_hamming_margin) {
    return candidate.geometry_guided_tag_likelihood_hamming_margin >
           current.geometry_guided_tag_likelihood_hamming_margin;
  }
  if (candidate.geometry_guided_tag_likelihood_contrast !=
      current.geometry_guided_tag_likelihood_contrast) {
    return candidate.geometry_guided_tag_likelihood_contrast >
           current.geometry_guided_tag_likelihood_contrast;
  }
  if (candidate.min_corner_response_ratio !=
      current.min_corner_response_ratio) {
    return candidate.min_corner_response_ratio >
           current.min_corner_response_ratio;
  }
  if (candidate.edge_support_ratio != current.edge_support_ratio) {
    return candidate.edge_support_ratio > current.edge_support_ratio;
  }
  return candidate.outer_reprojection_rmse < current.outer_reprojection_rmse;
}

// Evaluates every frame-level pose prediction through the same geometry-prior
// chain.  The two public RegenerateFrame overloads differ only in where their
// pose predictions come from; candidate validation and deterministic selection
// must remain a single implementation so the bootstrap and optimized-scene
// paths cannot drift apart.
GeometryPriorRecoverySelection EvaluateGeometryPriorPredictions(
    const cv::Mat& image,
    const ApriltagInternalConfig& config,
    const ApriltagInternalDetectionOptions& options,
    const IntermediateCameraConfig& camera_override,
    const InternalRegenerationFrameInput& frame_input,
    int board_id,
    const ApriltagCanonicalModel& model,
    double frame_normal_outer_refit_rmse_median,
    const std::string& original_outer_failure_reason,
    const std::vector<GeometryPriorPrediction>& predictions,
    std::vector<GeometryPriorOuterSeedCandidate>* candidates) {
  GeometryPriorRecoverySelection selection;
  if (candidates == nullptr) {
    return selection;
  }

  for (const GeometryPriorPrediction& prediction : predictions) {
    std::array<Eigen::Vector2d, 4> predicted_outer_corners{};
    if (!ProjectGeometryPriorOuterCorners(
            camera_override, model, prediction.T_camera_board, image.size(),
            &predicted_outer_corners)) {
      continue;
    }

    OuterWrongIdProposal topology_ordered_proposal;
    const OuterWrongIdProposal* effective_wrong_id_proposal =
        prediction.wrong_id_proposal;
    std::string effective_prediction_source_label = prediction.source_label;
    std::string topology_association_summary;
    if (prediction.wrong_id_proposal != nullptr) {
      const WrongIdTopologyAssociation association =
          AssociateWrongIdProposalToTopology(
              predicted_outer_corners, *prediction.wrong_id_proposal);
      topology_association_summary = association.summary;
      std::string unique_slot_summary;
      const bool unique_slot = IsUniqueWrongIdTopologyAssociation(
          predicted_outer_corners, *prediction.wrong_id_proposal,
          prediction.competing_topology_slots, &unique_slot_summary);
      topology_association_summary = unique_slot_summary;
      if (!unique_slot) {
        GeometryPriorOuterSeedCandidate rejected_candidate =
            BuildGeometryPriorOuterSeedCandidate(
                frame_input, board_id, prediction.visible_boards_used,
                predicted_outer_corners,
                prediction.source_label + "_topology_rejected_from" +
                    std::to_string(prediction.wrong_id_proposal->detected_tag_id),
                prediction.frame_pose_refit_source_board_id,
                prediction.frame_pose_refit_outer_rmse,
                original_outer_failure_reason);
        rejected_candidate.quad_topology_summary +=
            " " + topology_association_summary;
        rejected_candidate.reject_reason =
            "topology_id_association_rejected";
        candidates->push_back(std::move(rejected_candidate));
        continue;
      }
      topology_ordered_proposal = *prediction.wrong_id_proposal;
      topology_ordered_proposal.corners_original_image = association.ordered_corners;
      effective_wrong_id_proposal = &topology_ordered_proposal;
      effective_prediction_source_label += "_topology_assoc";
    }

    OuterTagDetectionResult rescued_detection;
    GeometryPriorOuterSeedCandidate candidate =
        EvaluateGeometryPriorOuterSeedCandidate(
            image, config, options, camera_override, frame_input, board_id,
            prediction.visible_boards_used, predicted_outer_corners,
            prediction.T_camera_board, effective_prediction_source_label,
            prediction.frame_pose_refit_source_board_id,
            prediction.frame_pose_refit_outer_rmse,
            frame_normal_outer_refit_rmse_median,
            original_outer_failure_reason, effective_wrong_id_proposal,
            &rescued_detection);
    if (!topology_association_summary.empty()) {
      candidate.quad_topology_summary += " " + topology_association_summary;
    }
    candidates->push_back(candidate);
    if (!candidate.accepted_as_rescued_observation) {
      continue;
    }
    selection.any_candidate_accepted = true;
    if (PreferGeometryPriorCandidate(
            candidate, selection.selected_candidate,
            selection.selected_rescued_detection)) {
      selection.selected_candidate = candidate;
      selection.detection = rescued_detection;
      selection.selected_rescued_detection = true;
    }
  }
  return selection;
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

struct BidirectionalTemporalPoseSeed {
  bool success = false;
  Eigen::Matrix4d T_camera_reference = Eigen::Matrix4d::Identity();
  std::vector<int> support_board_ids;
  double rotation_error_deg = std::numeric_limits<double>::infinity();
  double translation_error = std::numeric_limits<double>::infinity();
};

double RotationDifferenceDegrees(const Eigen::Matrix4d& lhs,
                                 const Eigen::Matrix4d& rhs) {
  constexpr double kPi = 3.14159265358979323846;
  const Eigen::Matrix3d relative_rotation =
      lhs.block<3, 3>(0, 0).transpose() * rhs.block<3, 3>(0, 0);
  const double cosine = std::max(
      -1.0, std::min(1.0, (relative_rotation.trace() - 1.0) * 0.5));
  return std::acos(cosine) * 180.0 / kPi;
}

bool IsDirectExactAnchor(const OuterBoardMeasurement& measurement) {
  return measurement.success &&
         measurement.detected_tag_id == measurement.board_id &&
         !measurement.used_local_patch_rescue &&
         measurement.valid_refined_corner_count == 4 &&
         std::all_of(measurement.refined_corner_valid.begin(),
                     measurement.refined_corner_valid.end(),
                     [](bool valid) { return valid; });
}

BidirectionalTemporalPoseSeed BuildBidirectionalTemporalPoseSeed(
    const InternalRegenerationFrameInput& frame_input,
    const OuterBootstrapResult& bootstrap_result,
    const LocalFramePoseRefit& current_anchor_refit) {
  BidirectionalTemporalPoseSeed seed;
  if (!current_anchor_refit.success ||
      current_anchor_refit.support_board_ids.size() != 1 ||
      frame_input.frame_index < 0) {
    return seed;
  }

  int direct_anchor_count = 0;
  int direct_anchor_id = -1;
  for (const OuterBoardMeasurement& measurement :
       frame_input.outer_detections.frame_measurements.board_measurements) {
    if (IsDirectExactAnchor(measurement)) {
      ++direct_anchor_count;
      direct_anchor_id = measurement.board_id;
    }
  }
  if (direct_anchor_count != 1 ||
      direct_anchor_id != current_anchor_refit.source_board_id) {
    return seed;
  }

  const OuterBootstrapFrameState* previous = nullptr;
  const OuterBootstrapFrameState* next = nullptr;
  for (const OuterBootstrapFrameState& candidate : bootstrap_result.frames) {
    if (!candidate.initialized || candidate.frame_index < 0 ||
        candidate.visible_board_ids.size() < 2) {
      continue;
    }
    const int distance = std::abs(candidate.frame_index - frame_input.frame_index);
    if (distance == 0 || distance > 4) {
      continue;
    }
    if (candidate.frame_index < frame_input.frame_index &&
        (previous == nullptr || candidate.frame_index > previous->frame_index)) {
      previous = &candidate;
    }
    if (candidate.frame_index > frame_input.frame_index &&
        (next == nullptr || candidate.frame_index < next->frame_index)) {
      next = &candidate;
    }
  }
  if (previous == nullptr || next == nullptr ||
      next->frame_index <= previous->frame_index) {
    return seed;
  }

  const double alpha = std::max(
      0.0, std::min(1.0,
                    static_cast<double>(frame_input.frame_index -
                                        previous->frame_index) /
                        static_cast<double>(next->frame_index -
                                            previous->frame_index)));
  Eigen::Isometry3d previous_pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d next_pose = Eigen::Isometry3d::Identity();
  previous_pose.matrix() = previous->T_camera_reference;
  next_pose.matrix() = next->T_camera_reference;
  Eigen::Quaterniond previous_rotation(previous_pose.rotation());
  Eigen::Quaterniond next_rotation(next_pose.rotation());
  if (previous_rotation.dot(next_rotation) < 0.0) {
    next_rotation.coeffs() *= -1.0;
  }
  Eigen::Isometry3d interpolated_pose = Eigen::Isometry3d::Identity();
  interpolated_pose.linear() =
      previous_rotation.slerp(alpha, next_rotation).normalized().toRotationMatrix();
  interpolated_pose.translation() =
      (1.0 - alpha) * previous_pose.translation() + alpha * next_pose.translation();

  seed.rotation_error_deg = RotationDifferenceDegrees(
      current_anchor_refit.T_camera_reference, interpolated_pose.matrix());
  seed.translation_error =
      (current_anchor_refit.T_camera_reference.block<3, 1>(0, 3) -
       interpolated_pose.translation())
          .norm();
  // A temporal pose is a search seed only. The current anchor must agree with
  // it before it is allowed to predict missing boards.
  if (!std::isfinite(seed.rotation_error_deg) ||
      !std::isfinite(seed.translation_error) ||
      seed.rotation_error_deg > 5.0 || seed.translation_error > 0.08) {
    return BidirectionalTemporalPoseSeed{};
  }

  seed.success = true;
  seed.T_camera_reference = interpolated_pose.matrix();
  seed.support_board_ids = previous->visible_board_ids;
  for (int board_id : next->visible_board_ids) {
    AppendUniqueBoardId(board_id, &seed.support_board_ids);
  }
  AppendUniqueBoardId(direct_anchor_id, &seed.support_board_ids);
  return seed;
}

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
  CollectVisibleBoardIds(frame_input, &result.visible_board_ids);

  const OuterBootstrapFrameState* frame_state = FindFrameState(bootstrap_result, frame_input);
  result.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;

  const IntermediateCameraConfig camera_override =
      MakeBootstrapCameraConfig(bootstrap_result.coarse_camera);
  const LocalFramePoseRefit visible_frame_refit =
      EstimateLocalFramePoseFromVisibleBoards(
          config_, bootstrap_result.coarse_camera, frame_input,
          [&](int visible_board_id, Eigen::Matrix4d* T_reference_board) {
            if (T_reference_board == nullptr) {
              return false;
            }
            if (visible_board_id == bootstrap_result.reference_board_id) {
              *T_reference_board = Eigen::Matrix4d::Identity();
              return true;
            }
            const OuterBootstrapBoardState* visible_board_state =
                FindBoardState(bootstrap_result, visible_board_id);
            if (visible_board_state == nullptr ||
                !visible_board_state->initialized) {
              return false;
            }
            *T_reference_board = visible_board_state->T_reference_board;
            return true;
          });
  const double frame_normal_outer_refit_rmse_median =
      ComputeFrameNormalOuterRefitRmseMedian(
          config_, bootstrap_result.coarse_camera, frame_input);
  const BidirectionalTemporalPoseSeed temporal_pose_seed =
      options_.outer_detector_config.enable_robust_missing_board_recovery
          ? BuildBidirectionalTemporalPoseSeed(frame_input, bootstrap_result,
                                               visible_frame_refit)
          : BidirectionalTemporalPoseSeed{};

  result.board_measurements.reserve(frame_input.outer_detections.detections.size());
  for (std::size_t index = 0; index < frame_input.outer_detections.requested_board_ids.size(); ++index) {
    const int board_id = frame_input.outer_detections.requested_board_ids[index];
    OuterTagDetectionResult outer_detection;
    if (index < frame_input.outer_detections.detections.size()) {
      outer_detection = frame_input.outer_detections.detections[index];
    } else {
      outer_detection = MakeMissingOuterTagDetection(board_id);
    }

    const OuterBootstrapBoardState* board_state = FindBoardState(bootstrap_result, board_id);
    const bool board_pose_available =
        board_state != nullptr && board_state->initialized;
    const bool scene_pose_prior_available =
        frame_state != nullptr && frame_state->initialized &&
        board_pose_available;
    const bool visible_refit_pose_prior_available =
        !scene_pose_prior_available && visible_frame_refit.success &&
        board_pose_available;
    const bool pose_prior_available =
        scene_pose_prior_available || visible_refit_pose_prior_available;
    Eigen::Matrix4d T_camera_board = Eigen::Matrix4d::Identity();
    if (scene_pose_prior_available) {
      T_camera_board = ComposeCameraBoardTransform(
          frame_state->T_camera_reference, board_state->T_reference_board);
    } else if (visible_refit_pose_prior_available) {
      T_camera_board =
          visible_frame_refit.T_camera_reference *
          board_state->T_reference_board;
    }

    const std::string original_outer_failure_reason =
        outer_detection.failure_reason_text.empty()
            ? ToString(outer_detection.failure_reason)
            : outer_detection.failure_reason_text;
    if (options_.enable_geometry_prior_outer_seed &&
        !outer_detection.success && pose_prior_available) {
      ApriltagInternalConfig board_config = config_;
      board_config.tag_id = board_id;
      const ApriltagCanonicalModel model(board_config);
      bool any_candidate_accepted = false;
      bool selected_rescued_detection = false;
      GeometryPriorOuterSeedCandidate selected_rescued_candidate;
      OuterTagDetectionResult selected_detection;
      Eigen::Matrix4d selected_rescued_pose = T_camera_board;
      bool selected_rescued_pose_available = false;
      const auto evaluate_prediction =
          [&](const Eigen::Matrix4d& T_camera_board_prediction,
              const std::string& prediction_source_label,
              int frame_pose_refit_source_board_id,
              double frame_pose_refit_outer_rmse,
              const std::vector<int>& visible_boards_used,
              const OuterWrongIdProposal* wrong_id_proposal = nullptr) {
            GeometryPriorPrediction prediction;
            prediction.T_camera_board = T_camera_board_prediction;
            prediction.source_label = prediction_source_label;
            prediction.frame_pose_refit_source_board_id =
                frame_pose_refit_source_board_id;
            prediction.frame_pose_refit_outer_rmse = frame_pose_refit_outer_rmse;
            prediction.visible_boards_used = visible_boards_used;
            if (board_state != nullptr) {
              const Eigen::Isometry3d T_camera_reference_prediction =
                  ToIsometry3d(T_camera_board_prediction) *
                  ToIsometry3d(board_state->T_reference_board).inverse();
              for (std::size_t competing_index = 0;
                   competing_index <
                   frame_input.outer_detections.requested_board_ids.size();
                   ++competing_index) {
                const int competing_board_id =
                    frame_input.outer_detections.requested_board_ids[competing_index];
                if (competing_board_id == board_id) {
                  continue;
                }
                const OuterBootstrapBoardState* competing_board_state =
                    FindBoardState(bootstrap_result, competing_board_id);
                if (competing_board_state == nullptr ||
                    !competing_board_state->initialized) {
                  continue;
                }
                ApriltagInternalConfig competing_config = config_;
                competing_config.tag_id = competing_board_id;
                const ApriltagCanonicalModel competing_model(competing_config);
                std::array<Eigen::Vector2d, 4> competing_corners{};
                if (ProjectGeometryPriorOuterCorners(
                        camera_override, competing_model,
                        (T_camera_reference_prediction *
                         ToIsometry3d(competing_board_state->T_reference_board)).matrix(),
                        image.size(), &competing_corners)) {
                  prediction.competing_topology_slots.push_back(competing_corners);
                }
              }
            }
            prediction.wrong_id_proposal = wrong_id_proposal;
            const GeometryPriorRecoverySelection selection =
                EvaluateGeometryPriorPredictions(
                    image, config_, options_, camera_override, frame_input,
                    board_id, model, frame_normal_outer_refit_rmse_median,
                    original_outer_failure_reason, {prediction},
                    &result.geometry_prior_outer_seed_candidates);
            any_candidate_accepted =
                any_candidate_accepted || selection.any_candidate_accepted;
            if (selection.selected_rescued_detection &&
                PreferGeometryPriorCandidate(
                    selection.selected_candidate,
                    selected_rescued_candidate,
                    selected_rescued_detection)) {
              selected_rescued_candidate = selection.selected_candidate;
              selected_detection = selection.detection;
              selected_rescued_detection = true;
              selected_rescued_pose = T_camera_board_prediction;
              selected_rescued_pose_available = true;
            }
          };
      evaluate_prediction(T_camera_board, result.state_source_label, -1, 0.0,
                          result.visible_board_ids);
      if (outer_detection.failure_reason ==
          OuterTagFailureReason::DetectionsExistButNoMatchingTagId) {
        for (const OuterWrongIdProposal& proposal :
             outer_detection.wrong_id_proposals) {
          evaluate_prediction(
              T_camera_board,
              result.state_source_label + "_wrong_id_proposal_from" +
                  std::to_string(proposal.detected_tag_id),
              -1, 0.0, result.visible_board_ids, &proposal);
        }
      }
      if (visible_frame_refit.success && board_pose_available) {
        const Eigen::Matrix4d T_camera_board_visible_refit =
            visible_frame_refit.T_camera_reference *
            board_state->T_reference_board;
        std::vector<int> visible_refit_board_ids =
            visible_frame_refit.support_board_ids;
        if (visible_refit_board_ids.empty()) {
          visible_refit_board_ids.push_back(
              visible_frame_refit.source_board_id);
        }
        const std::string visible_refit_label =
            result.state_source_label +
            (visible_frame_refit.support_board_ids.size() >= 2
                 ? "_visible_refit_consensus"
                 : "_visible_refit_single");
        evaluate_prediction(T_camera_board_visible_refit,
                            visible_refit_label,
                            visible_frame_refit.source_board_id,
                            visible_frame_refit.outer_rmse,
                            visible_refit_board_ids);
        std::vector<int> individual_refit_board_ids;
        for (const LocalFramePoseRefit::Candidate& pose_candidate :
             visible_frame_refit.candidates) {
          AppendUniqueBoardId(pose_candidate.source_board_id,
                              &individual_refit_board_ids);
        }
        if (individual_refit_board_ids.size() >= 2) {
          for (const LocalFramePoseRefit::Candidate& pose_candidate :
               visible_frame_refit.candidates) {
            const Eigen::Matrix4d T_camera_board_individual_refit =
                pose_candidate.T_camera_reference *
                board_state->T_reference_board;
            evaluate_prediction(
                T_camera_board_individual_refit,
                result.state_source_label + "_visible_refit_board" +
                    std::to_string(pose_candidate.source_board_id),
                pose_candidate.source_board_id, pose_candidate.outer_rmse,
                individual_refit_board_ids);
          }
        }
      }
      if (temporal_pose_seed.success && board_pose_available) {
        const Eigen::Matrix4d T_camera_board_temporal =
            temporal_pose_seed.T_camera_reference *
            board_state->T_reference_board;
        evaluate_prediction(
            T_camera_board_temporal,
            result.state_source_label +
                "_temporal_bidirectional_visible_refit_single_anchor",
            visible_frame_refit.source_board_id,
            visible_frame_refit.outer_rmse,
            temporal_pose_seed.support_board_ids);
      }
      if (selected_rescued_detection &&
          options_.geometry_prior_rescue_use_as_observation &&
          !options_.geometry_prior_rescue_diagnostic_only) {
        outer_detection = selected_detection;
        if (selected_rescued_pose_available) {
          T_camera_board = selected_rescued_pose;
        }
      } else if (any_candidate_accepted &&
                 options_.geometry_prior_rescue_use_as_observation &&
                 options_.geometry_prior_rescue_diagnostic_only) {
        EmitRegenerationWarning(
            "geometry prior rescue passed image validation but is kept "
            "diagnostic-only; pass --stage5-geometry-prior-rescue-diagnostic-only 0 "
            "to allow backend use.",
            &result.warnings);
      }
    }

    RegeneratedBoardMeasurement measurement;
    measurement.board_id = board_id;
    measurement.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;
    measurement.board_bootstrap_initialized = board_state != nullptr && board_state->initialized;
    // Keep the current frame's four image corners authoritative for every
    // exact-ID observation, including a camera-aware outer rescue.  The
    // persistent scene pose is only a seed for observations without a valid
    // current-frame outer quad.  Feeding that pose into a rescued quad can
    // move the spherical lattice hundreds of pixels before refinement.
    const bool has_local_outer_pose_seed =
        outer_detection.success &&
        std::all_of(outer_detection.refined_valid.begin(),
                    outer_detection.refined_valid.end(),
                    [](bool valid) { return valid; });
    const bool use_pose_prior_for_internal_generation =
        pose_prior_available && !outer_detection.used_local_patch_rescue &&
        !has_local_outer_pose_seed;
    measurement.pose_prior_used = use_pose_prior_for_internal_generation;
    try {
      measurement.detection = detector_.DetectFromOuterDetection(
          image, board_id, outer_detection, &camera_override,
          use_pose_prior_for_internal_generation ? &T_camera_board : nullptr);
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
    AccumulateRuntimeBreakdown(measurement.detection.runtime_breakdown,
                               &result.runtime_breakdown);
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
  CollectVisibleBoardIds(frame_input, &result.visible_board_ids);

  const JointSceneFrameState* frame_state = FindFrameState(scene_state, frame_input);
  result.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;

  const IntermediateCameraConfig camera_override =
      MakeSceneCameraConfig(scene_state.camera);
  const LocalFramePoseRefit visible_frame_refit =
      EstimateLocalFramePoseFromVisibleBoards(
          config_, scene_state.camera, frame_input,
          [&](int visible_board_id, Eigen::Matrix4d* T_reference_board) {
            if (T_reference_board == nullptr) {
              return false;
            }
            if (visible_board_id == scene_state.reference_board_id) {
              *T_reference_board = Eigen::Matrix4d::Identity();
              return true;
            }
            const JointSceneBoardState* visible_board_state =
                FindBoardState(scene_state, visible_board_id);
            if (visible_board_state == nullptr ||
                !visible_board_state->initialized) {
              return false;
            }
            *T_reference_board = visible_board_state->T_reference_board;
            return true;
          });
  const double frame_normal_outer_refit_rmse_median =
      ComputeFrameNormalOuterRefitRmseMedian(
          config_, scene_state.camera, frame_input);

  result.board_measurements.reserve(frame_input.outer_detections.detections.size());
  for (std::size_t index = 0; index < frame_input.outer_detections.requested_board_ids.size(); ++index) {
    const int board_id = frame_input.outer_detections.requested_board_ids[index];
    OuterTagDetectionResult outer_detection;
    if (index < frame_input.outer_detections.detections.size()) {
      outer_detection = frame_input.outer_detections.detections[index];
    } else {
      outer_detection = MakeMissingOuterTagDetection(board_id);
    }

    const JointSceneBoardState* board_state = FindBoardState(scene_state, board_id);
    const bool board_pose_available =
        board_state != nullptr && board_state->initialized;
    const bool scene_pose_prior_available =
        frame_state != nullptr && frame_state->initialized &&
        board_pose_available;
    const bool visible_refit_pose_prior_available =
        !scene_pose_prior_available && visible_frame_refit.success &&
        board_pose_available;
    const bool pose_prior_available =
        scene_pose_prior_available || visible_refit_pose_prior_available;
    Eigen::Matrix4d T_camera_board = Eigen::Matrix4d::Identity();
    if (scene_pose_prior_available) {
      T_camera_board = ComposeCameraBoardTransform(
          frame_state->T_camera_reference, board_state->T_reference_board);
    } else if (visible_refit_pose_prior_available) {
      T_camera_board =
          visible_frame_refit.T_camera_reference *
          board_state->T_reference_board;
    }

    const std::string original_outer_failure_reason =
        outer_detection.failure_reason_text.empty()
            ? ToString(outer_detection.failure_reason)
            : outer_detection.failure_reason_text;
    if (options_.enable_geometry_prior_outer_seed &&
        !outer_detection.success && pose_prior_available) {
      ApriltagInternalConfig board_config = config_;
      board_config.tag_id = board_id;
      const ApriltagCanonicalModel model(board_config);
      bool any_candidate_accepted = false;
      bool selected_rescued_detection = false;
      GeometryPriorOuterSeedCandidate selected_rescued_candidate;
      OuterTagDetectionResult selected_detection;
      const auto evaluate_prediction =
          [&](const Eigen::Matrix4d& T_camera_board_prediction,
              const std::string& prediction_source_label,
              int frame_pose_refit_source_board_id,
              double frame_pose_refit_outer_rmse,
              const std::vector<int>& visible_boards_used,
              const OuterWrongIdProposal* wrong_id_proposal = nullptr) {
            GeometryPriorPrediction prediction;
            prediction.T_camera_board = T_camera_board_prediction;
            prediction.source_label = prediction_source_label;
            prediction.frame_pose_refit_source_board_id =
                frame_pose_refit_source_board_id;
            prediction.frame_pose_refit_outer_rmse = frame_pose_refit_outer_rmse;
            prediction.visible_boards_used = visible_boards_used;
            if (board_state != nullptr) {
              const Eigen::Isometry3d T_camera_reference_prediction =
                  ToIsometry3d(T_camera_board_prediction) *
                  ToIsometry3d(board_state->T_reference_board).inverse();
              for (std::size_t competing_index = 0;
                   competing_index <
                   frame_input.outer_detections.requested_board_ids.size();
                   ++competing_index) {
                const int competing_board_id =
                    frame_input.outer_detections.requested_board_ids[competing_index];
                if (competing_board_id == board_id) {
                  continue;
                }
                const JointSceneBoardState* competing_board_state =
                    FindBoardState(scene_state, competing_board_id);
                if (competing_board_state == nullptr ||
                    !competing_board_state->initialized) {
                  continue;
                }
                ApriltagInternalConfig competing_config = config_;
                competing_config.tag_id = competing_board_id;
                const ApriltagCanonicalModel competing_model(competing_config);
                std::array<Eigen::Vector2d, 4> competing_corners{};
                if (ProjectGeometryPriorOuterCorners(
                        camera_override, competing_model,
                        (T_camera_reference_prediction *
                         ToIsometry3d(competing_board_state->T_reference_board)).matrix(),
                        image.size(), &competing_corners)) {
                  prediction.competing_topology_slots.push_back(competing_corners);
                }
              }
            }
            prediction.wrong_id_proposal = wrong_id_proposal;
            const GeometryPriorRecoverySelection selection =
                EvaluateGeometryPriorPredictions(
                    image, config_, options_, camera_override, frame_input,
                    board_id, model, frame_normal_outer_refit_rmse_median,
                    original_outer_failure_reason, {prediction},
                    &result.geometry_prior_outer_seed_candidates);
            any_candidate_accepted =
                any_candidate_accepted || selection.any_candidate_accepted;
            if (selection.selected_rescued_detection &&
                PreferGeometryPriorCandidate(
                    selection.selected_candidate,
                    selected_rescued_candidate,
                    selected_rescued_detection)) {
              selected_rescued_candidate = selection.selected_candidate;
              selected_detection = selection.detection;
              selected_rescued_detection = true;
            }
          };
      evaluate_prediction(T_camera_board, result.state_source_label, -1, 0.0,
                          result.visible_board_ids);
      if (outer_detection.failure_reason ==
          OuterTagFailureReason::DetectionsExistButNoMatchingTagId) {
        for (const OuterWrongIdProposal& proposal :
             outer_detection.wrong_id_proposals) {
          evaluate_prediction(
              T_camera_board,
              result.state_source_label + "_wrong_id_proposal_from" +
                  std::to_string(proposal.detected_tag_id),
              -1, 0.0, result.visible_board_ids, &proposal);
        }
      }
      if (visible_frame_refit.success && board_pose_available) {
        const Eigen::Matrix4d T_camera_board_visible_refit =
            visible_frame_refit.T_camera_reference *
            board_state->T_reference_board;
        std::vector<int> visible_refit_board_ids =
            visible_frame_refit.support_board_ids;
        if (visible_refit_board_ids.empty()) {
          visible_refit_board_ids.push_back(
              visible_frame_refit.source_board_id);
        }
        const std::string visible_refit_label =
            result.state_source_label +
            (visible_frame_refit.support_board_ids.size() >= 2
                 ? "_visible_refit_consensus"
                 : "_visible_refit_single");
        evaluate_prediction(T_camera_board_visible_refit,
                            visible_refit_label,
                            visible_frame_refit.source_board_id,
                            visible_frame_refit.outer_rmse,
                            visible_refit_board_ids);
        std::vector<int> individual_refit_board_ids;
        for (const LocalFramePoseRefit::Candidate& pose_candidate :
             visible_frame_refit.candidates) {
          AppendUniqueBoardId(pose_candidate.source_board_id,
                              &individual_refit_board_ids);
        }
        if (individual_refit_board_ids.size() >= 2) {
          for (const LocalFramePoseRefit::Candidate& pose_candidate :
               visible_frame_refit.candidates) {
            const Eigen::Matrix4d T_camera_board_individual_refit =
                pose_candidate.T_camera_reference *
                board_state->T_reference_board;
            evaluate_prediction(
                T_camera_board_individual_refit,
                result.state_source_label + "_visible_refit_board" +
                    std::to_string(pose_candidate.source_board_id),
                pose_candidate.source_board_id, pose_candidate.outer_rmse,
                individual_refit_board_ids);
          }
        }
      }
      if (selected_rescued_detection &&
          options_.geometry_prior_rescue_use_as_observation &&
          !options_.geometry_prior_rescue_diagnostic_only) {
        outer_detection = selected_detection;
      } else if (any_candidate_accepted &&
                 options_.geometry_prior_rescue_use_as_observation &&
                 options_.geometry_prior_rescue_diagnostic_only) {
        EmitRegenerationWarning(
            "geometry prior rescue passed image validation but is kept "
            "diagnostic-only; pass --stage5-geometry-prior-rescue-diagnostic-only 0 "
            "to allow backend use.",
            &result.warnings);
      }
    }

    RegeneratedBoardMeasurement measurement;
    measurement.board_id = board_id;
    measurement.frame_bootstrap_initialized = frame_state != nullptr && frame_state->initialized;
    measurement.board_bootstrap_initialized = board_state != nullptr && board_state->initialized;
    // Keep the same current-image exact-ID priority as the bootstrap overload.
    const bool has_local_outer_pose_seed =
        outer_detection.success &&
        std::all_of(outer_detection.refined_valid.begin(),
                    outer_detection.refined_valid.end(),
                    [](bool valid) { return valid; });
    const bool use_pose_prior_for_internal_generation =
        pose_prior_available && !outer_detection.used_local_patch_rescue &&
        !has_local_outer_pose_seed;
    measurement.pose_prior_used = use_pose_prior_for_internal_generation;
    try {
      measurement.detection = detector_.DetectFromOuterDetection(
          image, board_id, outer_detection, &camera_override,
          use_pose_prior_for_internal_generation ? &T_camera_board : nullptr);
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
    AccumulateRuntimeBreakdown(measurement.detection.runtime_breakdown,
                               &result.runtime_breakdown);
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
  config.camera_model = intrinsics.camera_model;
  config.distortion_model = intrinsics.distortion_model;
  config.distortion_coeffs = intrinsics.DistortionVector();
  config.intrinsics = intrinsics.IntrinsicsVector();
  config.resolution = {intrinsics.resolution.width, intrinsics.resolution.height};
  return config;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
