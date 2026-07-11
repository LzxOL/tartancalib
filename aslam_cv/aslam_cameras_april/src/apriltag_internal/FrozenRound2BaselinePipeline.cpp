#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

#include <opencv2/imgcodecs.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/JointMeasurementSelection.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionMeasurementBuilder.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionOptimizer.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionResidualEvaluator.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

double ElapsedSeconds(const std::chrono::steady_clock::time_point& start_time) {
  return std::chrono::duration_cast<std::chrono::duration<double> >(
             std::chrono::steady_clock::now() - start_time)
      .count();
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

void AppendWarnings(const std::vector<std::string>& new_warnings,
                    std::vector<std::string>* warnings) {
  if (warnings == nullptr) {
    return;
  }
  for (const std::string& warning : new_warnings) {
    AppendUniqueWarning(warning, warnings);
  }
}

void AccumulateRegenerationRuntime(
    const InternalRegenerationRuntimeBreakdown& frame_runtime,
    double* pose_estimation_seconds,
    double* boundary_model_seconds,
    double* seed_search_seconds,
    double* ray_refine_seconds,
    double* image_evidence_seconds,
    double* subpix_seconds) {
  if (pose_estimation_seconds != nullptr) {
    *pose_estimation_seconds += frame_runtime.pose_estimation_seconds;
  }
  if (boundary_model_seconds != nullptr) {
    *boundary_model_seconds += frame_runtime.boundary_model_seconds;
  }
  if (seed_search_seconds != nullptr) {
    *seed_search_seconds += frame_runtime.seed_search_seconds;
  }
  if (ray_refine_seconds != nullptr) {
    *ray_refine_seconds += frame_runtime.ray_refine_seconds;
  }
  if (image_evidence_seconds != nullptr) {
    *image_evidence_seconds += frame_runtime.image_evidence_seconds;
  }
  if (subpix_seconds != nullptr) {
    *subpix_seconds += frame_runtime.subpix_seconds;
  }
}

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

void PopulateOuterRefineCameraFromIntermediate(
    const IntermediateCameraConfig& intermediate_camera,
    OuterRefineCameraConfig* refine_camera) {
  if (refine_camera == nullptr || refine_camera->IsConfigured() ||
      !intermediate_camera.IsConfigured()) {
    return;
  }
  refine_camera->camera_model = intermediate_camera.camera_model;
  refine_camera->distortion_model = intermediate_camera.distortion_model;
  refine_camera->intrinsics = intermediate_camera.intrinsics;
  refine_camera->distortion_coeffs = intermediate_camera.distortion_coeffs;
  refine_camera->resolution = intermediate_camera.resolution;
}

ApriltagInternalConfig NormalizeConfig(ApriltagInternalConfig config) {
  config.tag_ids = NormalizeBoardIds(config.tag_ids, config.tag_id);
  if (!config.tag_ids.empty()) {
    config.tag_id = config.tag_ids.front();
  }
  config.outer_detector_config.tag_ids = config.tag_ids;
  config.outer_detector_config.tag_id = config.tag_id;
  PopulateOuterRefineCameraFromIntermediate(
      config.intermediate_camera, &config.outer_detector_config.refine_camera);
  return config;
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
  options.ignore_image_evidence_min_quality =
      config.ignore_image_evidence_min_quality;
  options.force_internal_seed_from_prediction =
      config.force_internal_seed_from_prediction;
  options.bypass_internal_seed_filters =
      config.bypass_internal_seed_filters;
  options.outer_detector_config = config.outer_detector_config;
  return options;
}

OuterBootstrapOptions MakeBootstrapOptions(const ApriltagInternalConfig& config,
                                           const FrozenRound2BaselineOptions& options) {
  OuterBootstrapOptions bootstrap_options;
  bootstrap_options.reference_board_id = options.reference_board_id;
  bootstrap_options.initial_camera = OuterBootstrapCameraIntrinsics();
  bootstrap_options.init_xi = config.sphere_lattice_init_xi;
  bootstrap_options.init_alpha = config.sphere_lattice_init_alpha;
  bootstrap_options.init_fu_scale = config.sphere_lattice_init_fu_scale;
  bootstrap_options.init_fv_scale = config.sphere_lattice_init_fv_scale;
  bootstrap_options.init_cu_offset = config.sphere_lattice_init_cu_offset;
  bootstrap_options.init_cv_offset = config.sphere_lattice_init_cv_offset;
  bootstrap_options.min_detection_quality = config.outer_detector_config.min_detection_quality;
  return bootstrap_options;
}

void SetBootstrapInitFromIntrinsics(const OuterBootstrapCameraIntrinsics& intrinsics,
                                    OuterBootstrapOptions* options) {
  if (options == nullptr) {
    throw std::runtime_error("SetBootstrapInitFromIntrinsics requires a valid options pointer.");
  }
  options->initial_camera = intrinsics;
  options->init_xi = intrinsics.xi;
  options->init_alpha = intrinsics.alpha;
  options->init_fu_scale =
      intrinsics.fu / static_cast<double>(intrinsics.resolution.width);
  options->init_fv_scale =
      intrinsics.fv / static_cast<double>(intrinsics.resolution.height);
  options->init_cu_offset =
      intrinsics.cu - 0.5 * static_cast<double>(intrinsics.resolution.width);
  options->init_cv_offset =
      intrinsics.cv - 0.5 * static_cast<double>(intrinsics.resolution.height);
}

std::set<std::tuple<int, int, int, int, int> > BuildSolverSignatureSet(
    const JointMeasurementBuildResult& result) {
  std::set<std::tuple<int, int, int, int, int> > signatures;
  for (const JointPointObservation& point : result.solver_observations) {
    signatures.insert(std::make_tuple(
        point.frame_index, point.board_id, point.point_id,
        static_cast<int>(point.point_type), static_cast<int>(point.source_kind)));
  }
  return signatures;
}

JointMeasurementBuildValidationSummary ValidateJointMeasurementBuilder(
    const std::vector<JointMeasurementFrameInput>& joint_inputs,
    const OuterBootstrapResult& bootstrap_result,
    const JointReprojectionMeasurementBuilder& builder,
    const JointMeasurementBuildResult& primary_result) {
  JointMeasurementBuildValidationSummary summary;
  if (!primary_result.success) {
    summary.failure_reason = "Primary joint measurement build failed.";
    return summary;
  }

  int hierarchical_used_points = 0;
  std::set<std::pair<int, int> > used_board_observation_keys;
  for (const JointMeasurementFrameResult& frame_result : primary_result.frames) {
    for (const JointBoardObservation& board_observation : frame_result.board_observations) {
      bool board_has_used_point = false;
      for (const JointPointObservation& point : board_observation.points) {
        if (point.used_in_solver) {
          ++hierarchical_used_points;
          board_has_used_point = true;
        }
      }
      if (board_has_used_point) {
        used_board_observation_keys.insert(
            std::make_pair(frame_result.frame_index, board_observation.board_id));
      }
    }
  }

  summary.flat_hierarchical_consistent =
      hierarchical_used_points == static_cast<int>(primary_result.solver_observations.size());
  summary.counting_consistent =
      primary_result.used_outer_point_count ==
          4 * primary_result.accepted_outer_board_observation_count &&
      primary_result.used_total_point_count ==
          static_cast<int>(primary_result.solver_observations.size()) &&
      primary_result.used_board_observation_count ==
          static_cast<int>(used_board_observation_keys.size()) &&
      primary_result.used_total_point_count ==
          primary_result.used_outer_point_count + primary_result.used_internal_point_count;

  std::vector<JointMeasurementFrameInput> reversed_inputs = joint_inputs;
  std::reverse(reversed_inputs.begin(), reversed_inputs.end());
  const JointMeasurementBuildResult reversed_result =
      builder.Build(reversed_inputs, bootstrap_result);
  summary.frame_order_invariant =
      reversed_result.success &&
      reversed_result.used_frame_count == primary_result.used_frame_count &&
      reversed_result.used_board_observation_count ==
          primary_result.used_board_observation_count &&
      reversed_result.used_outer_point_count == primary_result.used_outer_point_count &&
      reversed_result.used_internal_point_count == primary_result.used_internal_point_count &&
      BuildSolverSignatureSet(reversed_result) == BuildSolverSignatureSet(primary_result);

  if (!joint_inputs.empty()) {
    std::vector<JointMeasurementFrameInput> mismatch_inputs = joint_inputs;
    mismatch_inputs.front().frame_label += "_label_mismatch_probe";
    mismatch_inputs.front().regenerated_internal.frame_label =
        mismatch_inputs.front().frame_label;
    const JointMeasurementBuildResult mismatch_result =
        builder.Build(mismatch_inputs, bootstrap_result);
    bool found_label_warning = false;
    for (const std::string& warning : mismatch_result.warnings) {
      if (warning.find("label mismatch") != std::string::npos) {
        found_label_warning = true;
        break;
      }
    }
    summary.label_mismatch_warning_observed =
        mismatch_result.success &&
        mismatch_result.used_total_point_count == primary_result.used_total_point_count &&
        found_label_warning;
  } else {
    summary.label_mismatch_warning_observed = true;
  }

  if (!summary.counting_consistent) {
    summary.warnings.push_back("Builder counting semantics are inconsistent.");
  }
  if (!summary.flat_hierarchical_consistent) {
    summary.warnings.push_back(
        "Flat solver observations do not match hierarchical used points.");
  }
  if (!summary.frame_order_invariant) {
    summary.warnings.push_back(
        "Frame-order perturbation changed the joint measurement result.");
  }
  if (!summary.label_mismatch_warning_observed) {
    summary.warnings.push_back(
        "Label mismatch probe did not produce stable counts plus warning as expected.");
  }

  summary.success = summary.counting_consistent &&
                    summary.flat_hierarchical_consistent &&
                    summary.frame_order_invariant &&
                    summary.label_mismatch_warning_observed;
  if (!summary.success && summary.failure_reason.empty()) {
    summary.failure_reason = "Joint measurement builder validation failed.";
  }
  return summary;
}

bool ComputeStage42ValidationPass(const JointMeasurementSelectionResult& round1_selection,
                                  const JointOptimizationResult& round1_result,
                                  bool round2_available,
                                  const JointMeasurementSelectionResult& round2_selection,
                                  const JointOptimizationResult& round2_result) {
  if (!round2_available) {
    return false;
  }
  const bool round2_non_degrading_overall =
      round2_result.optimized_residual.overall_rmse <=
      round1_result.optimized_residual.overall_rmse;
  const bool round2_non_degrading_internal =
      round2_result.optimized_residual.internal_only_rmse <=
      round1_result.optimized_residual.internal_only_rmse;
  const bool selected_data_present =
      round2_selection.accepted_frame_count > 0 &&
      round2_selection.accepted_board_observation_count > 0;
  return round2_non_degrading_overall && round2_non_degrading_internal &&
         selected_data_present;
}

void RecomputeMeasurementCounts(JointMeasurementBuildResult* result) {
  if (result == nullptr) {
    return;
  }

  result->solver_observations.clear();
  result->used_frame_count = 0;
  result->accepted_outer_board_observation_count = 0;
  result->accepted_internal_board_observation_count = 0;
  result->used_board_observation_count = 0;
  result->used_outer_point_count = 0;
  result->used_internal_point_count = 0;
  result->used_total_point_count = 0;

  std::set<int> used_frames;
  std::set<std::pair<int, int> > used_board_keys;
  std::set<std::pair<int, int> > accepted_outer_keys;
  std::set<std::pair<int, int> > accepted_internal_keys;
  for (JointMeasurementFrameResult& frame_result : result->frames) {
    for (JointBoardObservation& board_observation :
         frame_result.board_observations) {
      board_observation.used_in_solver = false;
      board_observation.outer_point_count = 0;
      board_observation.internal_point_count = 0;
      for (JointPointObservation& point : board_observation.points) {
        if (!point.used_in_solver) {
          continue;
        }
        result->solver_observations.push_back(point);
        board_observation.used_in_solver = true;
        used_frames.insert(frame_result.frame_index);
        used_board_keys.insert(
            std::make_pair(frame_result.frame_index, board_observation.board_id));
        if (point.point_type == JointPointType::Outer) {
          ++board_observation.outer_point_count;
          ++result->used_outer_point_count;
          accepted_outer_keys.insert(
              std::make_pair(frame_result.frame_index, board_observation.board_id));
        } else {
          ++board_observation.internal_point_count;
          ++result->used_internal_point_count;
          accepted_internal_keys.insert(
              std::make_pair(frame_result.frame_index, board_observation.board_id));
        }
      }
    }
  }

  result->used_frame_count = static_cast<int>(used_frames.size());
  result->used_board_observation_count =
      static_cast<int>(used_board_keys.size());
  result->accepted_outer_board_observation_count =
      static_cast<int>(accepted_outer_keys.size());
  result->accepted_internal_board_observation_count =
      static_cast<int>(accepted_internal_keys.size());
  result->used_total_point_count =
      result->used_outer_point_count + result->used_internal_point_count;
  result->success = result->used_total_point_count > 0;
}

std::vector<JointMeasurementFrameInput> BuildOuterOnlyIntermediateInputs(
    const std::vector<InternalRegenerationFrameInput>& regeneration_inputs) {
  std::vector<JointMeasurementFrameInput> joint_inputs;
  joint_inputs.reserve(regeneration_inputs.size());
  for (const InternalRegenerationFrameInput& input : regeneration_inputs) {
    InternalRegenerationFrameResult regeneration_result;
    regeneration_result.frame_index = input.frame_index;
    regeneration_result.frame_label = input.frame_label;
    regeneration_result.frame_bootstrap_initialized = true;
    regeneration_result.state_source_label = "outer_only_intermediate_no_internal";
    regeneration_result.visible_board_ids =
        input.outer_detections.requested_board_ids;

    JointMeasurementFrameInput joint_input;
    joint_input.frame_index = input.frame_index;
    joint_input.frame_label = input.frame_label;
    joint_input.outer_detections = input.outer_detections;
    joint_input.regenerated_internal = regeneration_result;
    joint_inputs.push_back(joint_input);
  }
  return joint_inputs;
}

JointMeasurementSelectionResult SelectOuterOnlyIntermediateObservations(
    const JointMeasurementBuildResult& measurement_result,
    const JointResidualEvaluationResult& residual_result,
    const JointReprojectionSceneState& scene_state,
    double max_outer_rmse_px,
    int min_visible_boards) {
  JointMeasurementSelectionResult result;
  result.reference_board_id = scene_state.reference_board_id;
  result.selected_measurement_result = measurement_result;

  if (!measurement_result.success) {
    result.failure_reason =
        "outer-only intermediate measurement_result.success is false";
    return result;
  }
  if (!residual_result.success) {
    result.failure_reason =
        "outer-only intermediate residual_result.success is false";
    return result;
  }

  std::map<std::pair<int, int>, const JointResidualBoardObservationDiagnostics*>
      residual_by_key;
  for (const JointResidualBoardObservationDiagnostics& diagnostics :
       residual_result.board_observation_diagnostics) {
    residual_by_key[std::make_pair(diagnostics.frame_index,
                                   diagnostics.board_id)] = &diagnostics;
  }

  std::set<std::pair<int, int> > candidate_keys;
  std::map<int, std::vector<int> > candidate_boards_by_frame;
  for (const JointMeasurementFrameResult& frame_result :
       measurement_result.frames) {
    const JointSceneFrameState* scene_frame =
        FindJointSceneFrameState(scene_state, frame_result.frame_index);
    JointFrameSelectionDecision frame_decision;
    frame_decision.frame_index = frame_result.frame_index;
    frame_decision.frame_label = frame_result.frame_label;

    for (const JointBoardObservation& board_observation :
         frame_result.board_observations) {
      JointBoardObservationSelectionDecision decision;
      decision.frame_index = frame_result.frame_index;
      decision.frame_label = frame_result.frame_label;
      decision.board_id = board_observation.board_id;

      const std::pair<int, int> key(frame_result.frame_index,
                                   board_observation.board_id);
      const auto residual_it = residual_by_key.find(key);
      if (residual_it != residual_by_key.end()) {
        decision.rmse = residual_it->second->rmse;
        decision.point_count = residual_it->second->point_count;
        decision.outer_point_count = residual_it->second->outer_point_count;
        decision.internal_point_count =
            residual_it->second->internal_point_count;
      }

      const JointSceneBoardState* scene_board =
          FindJointSceneBoardState(scene_state, board_observation.board_id);
      const bool solver_ready = board_observation.used_in_solver &&
          scene_frame != nullptr && scene_frame->initialized &&
          (board_observation.board_id == scene_state.reference_board_id ||
           (scene_board != nullptr && scene_board->initialized));
      const bool outer_only =
          decision.outer_point_count > 0 && decision.internal_point_count == 0;
      const bool residual_ok =
          std::isfinite(decision.rmse) &&
          (max_outer_rmse_px <= 0.0 || decision.rmse <= max_outer_rmse_px);
      if (!solver_ready || !outer_only || !residual_ok) {
        decision.accepted = false;
        decision.reason_code =
            JointBoardObservationSelectionReasonCode::RejectedNotSolverReady;
        if (!solver_ready) {
          decision.reason_detail = "not solver-ready for intermediate";
        } else if (!outer_only) {
          decision.reason_detail =
              "intermediate requires outer-only real detections";
        } else {
          decision.reason_code =
              JointBoardObservationSelectionReasonCode::RejectedResidualSanity;
          decision.reason_detail = "outer rmse exceeds intermediate threshold";
        }
        result.board_observation_decisions.push_back(decision);
        continue;
      }

      candidate_keys.insert(key);
      candidate_boards_by_frame[frame_result.frame_index].push_back(
          board_observation.board_id);
      result.board_observation_decisions.push_back(decision);
    }
    result.frame_decisions.push_back(frame_decision);
  }

  const int min_boards = std::max(1, min_visible_boards);
  for (const auto& entry : candidate_boards_by_frame) {
    if (static_cast<int>(entry.second.size()) < min_boards) {
      continue;
    }
    result.accepted_frame_indices.insert(entry.first);
    for (int board_id : entry.second) {
      result.accepted_board_observation_keys.insert(
          std::make_pair(entry.first, board_id));
    }
  }

  for (JointBoardObservationSelectionDecision& decision :
       result.board_observation_decisions) {
    const std::pair<int, int> key(decision.frame_index, decision.board_id);
    if (result.accepted_board_observation_keys.find(key) !=
        result.accepted_board_observation_keys.end()) {
      decision.accepted = true;
      decision.reason_code = JointBoardObservationSelectionReasonCode::Accepted;
      decision.reason_detail = "accepted by outer-only intermediate gate";
    } else if (candidate_keys.find(key) != candidate_keys.end()) {
      decision.accepted = false;
      decision.reason_code =
          JointBoardObservationSelectionReasonCode::RejectedFrameRejected;
      decision.reason_detail =
          "frame has fewer than intermediate_min_visible_boards accepted boards";
    }
  }

  for (JointFrameSelectionDecision& frame_decision : result.frame_decisions) {
    const auto frame_it =
        candidate_boards_by_frame.find(frame_decision.frame_index);
    if (frame_it != candidate_boards_by_frame.end()) {
      frame_decision.usable_board_ids = frame_it->second;
      std::sort(frame_decision.usable_board_ids.begin(),
                frame_decision.usable_board_ids.end());
      frame_decision.usable_board_observation_count =
          static_cast<int>(frame_decision.usable_board_ids.size());
    }
    frame_decision.accepted =
        result.accepted_frame_indices.find(frame_decision.frame_index) !=
        result.accepted_frame_indices.end();
    if (frame_decision.accepted) {
      frame_decision.accepted_board_ids = frame_decision.usable_board_ids;
      frame_decision.accepted_board_observation_count =
          static_cast<int>(frame_decision.accepted_board_ids.size());
      frame_decision.reason_codes.push_back(
          JointFrameSelectionReasonCode::AcceptedMinViewsPerBoard);
      frame_decision.reason_detail = "accepted by outer-only intermediate gate";
    } else if (frame_decision.usable_board_observation_count > 0) {
      frame_decision.reason_codes.push_back(
          JointFrameSelectionReasonCode::RejectedRedundantView);
      frame_decision.reason_detail =
          "insufficient visible boards for outer-only intermediate";
    } else {
      frame_decision.reason_codes.push_back(
          JointFrameSelectionReasonCode::NoUsableBoardObservations);
      frame_decision.reason_detail =
          "no usable outer-only board observations";
    }
  }

  for (JointMeasurementFrameResult& frame_result :
       result.selected_measurement_result.frames) {
    for (JointBoardObservation& board_observation :
         frame_result.board_observations) {
      const bool selected =
          result.accepted_board_observation_keys.find(
              std::make_pair(frame_result.frame_index,
                             board_observation.board_id)) !=
          result.accepted_board_observation_keys.end();
      for (JointPointObservation& point : board_observation.points) {
        point.used_in_solver = point.used_in_solver && selected;
      }
    }
  }
  RecomputeMeasurementCounts(&result.selected_measurement_result);

  result.accepted_frame_count =
      static_cast<int>(result.accepted_frame_indices.size());
  result.accepted_board_observation_count =
      static_cast<int>(result.accepted_board_observation_keys.size());
  result.accepted_outer_point_count =
      result.selected_measurement_result.used_outer_point_count;
  result.accepted_internal_point_count =
      result.selected_measurement_result.used_internal_point_count;

  if (result.accepted_board_observation_count <= 0 ||
      result.selected_measurement_result.used_total_point_count <= 0) {
    result.failure_reason =
        "outer-only intermediate gate produced no accepted observations";
    result.selected_measurement_result.success = false;
    return result;
  }

  result.success = true;
  return result;
}

}  // namespace

FrozenRound2BaselinePipeline::FrozenRound2BaselinePipeline(
    FrozenRound2BaselineOptions options)
    : options_(std::move(options)) {}

FrozenRound2BaselineResult FrozenRound2BaselinePipeline::Run(
    const std::vector<FrozenRound2BaselineFrameSource>& frame_sources) const {
  FrozenRound2BaselineResult result;
  result.baseline_protocol_label = options_.baseline_protocol_label;
  result.dataset_label = options_.dataset_label;
  result.training_split_signature = options_.training_split_signature;
  result.reference_board_id = options_.reference_board_id;
  result.frame_sources = frame_sources;
  result.effective_options = options_;

  if (frame_sources.empty()) {
    result.failure_reason = "FrozenRound2BaselinePipeline requires at least one frame.";
    return result;
  }

  const ApriltagInternalConfig config = NormalizeConfig(options_.config);
  ApriltagInternalDetectionOptions detection_options = MakeDetectionOptions(config);
  detection_options.internal_pose_rescue_mode =
      options_.internal_pose_rescue_mode;
  detection_options.internal_pose_rescue_max_ray_angle_deg =
      options_.internal_pose_rescue_max_ray_angle_deg;
  detection_options.internal_pose_rescue_accept_max_outer_rmse =
      options_.internal_pose_rescue_accept_max_outer_rmse;
  detection_options.ignore_image_evidence_min_quality =
      options_.ignore_image_evidence_min_quality;
  detection_options.force_internal_seed_from_prediction =
      options_.force_internal_seed_from_prediction;
  detection_options.bypass_internal_seed_filters =
      options_.bypass_internal_seed_filters;
  detection_options.enable_geometry_prior_outer_seed =
      options_.enable_geometry_prior_outer_seed;
  detection_options.geometry_prior_rescue_diagnostic_only =
      options_.geometry_prior_rescue_diagnostic_only;
  detection_options.geometry_prior_rescue_use_as_observation =
      options_.geometry_prior_rescue_use_as_observation;
  detection_options.geometry_prior_rescue_allow_geometry_only_pose_refit =
      options_.geometry_prior_rescue_allow_geometry_only_pose_refit;
  detection_options.geometry_prior_rescue_subpix_window_radius =
      options_.geometry_prior_rescue_subpix_window_radius;
  detection_options.geometry_prior_rescue_max_corner_displacement_px =
      options_.geometry_prior_rescue_max_corner_displacement_px;
  detection_options.geometry_prior_rescue_min_corner_response_ratio =
      options_.geometry_prior_rescue_min_corner_response_ratio;
  detection_options.geometry_prior_rescue_enable_spherical_refine =
      options_.geometry_prior_rescue_enable_spherical_refine;
  detection_options.geometry_prior_rescue_edge_sample_count =
      options_.geometry_prior_rescue_edge_sample_count;
  detection_options.geometry_prior_rescue_edge_search_half_width_px =
      options_.geometry_prior_rescue_edge_search_half_width_px;
  detection_options.geometry_prior_rescue_min_edge_support_ratio =
      options_.geometry_prior_rescue_min_edge_support_ratio;
  detection_options.geometry_prior_rescue_min_edge_gradient_ratio =
      options_.geometry_prior_rescue_min_edge_gradient_ratio;
  detection_options.geometry_prior_rescue_accept_max_outer_rmse =
      options_.geometry_prior_rescue_accept_max_outer_rmse;
  detection_options.geometry_prior_rescue_accept_max_rotation_error_deg =
      options_.geometry_prior_rescue_accept_max_rotation_error_deg;
  detection_options.geometry_prior_rescue_accept_max_translation_error =
      options_.geometry_prior_rescue_accept_max_translation_error;
  const MultiScaleOuterTagDetector outer_detector(config.outer_detector_config);
  OuterBootstrapOptions bootstrap_options = MakeBootstrapOptions(config, options_);
  const MultiBoardInternalMeasurementRegenerator regenerator(config, detection_options);
  JointMeasurementBuildOptions build_options;
  build_options.reference_board_id = options_.reference_board_id;
  build_options.include_internal_points = options_.include_internal_points;
  build_options.include_outer_when_internal_failed =
      options_.outer_only_ablation_mode ||
      !options_.strict_board_observation_acceptance;
  build_options.include_rescued_outer_when_internal_failed =
      options_.geometry_prior_rescue_keep_outer_on_internal_failure;
  build_options.enable_internal_observation_quality_weighting =
      options_.enable_internal_observation_quality_weighting;
  build_options.internal_observation_low_quality_quantile =
      options_.internal_observation_low_quality_quantile;
  build_options.internal_observation_min_weight =
      options_.internal_observation_min_weight;
  build_options.internal_observation_quality_exponent =
      options_.internal_observation_quality_exponent;
  build_options.filter_internal_corner_mode =
      options_.internal_corner_filter_mode;
  build_options.filter_internal_corner_max_reproj_error =
      options_.internal_corner_filter_max_reproj_error;
  build_options.filter_internal_corner_quality_min =
      options_.internal_corner_filter_quality_min;
  build_options.filter_internal_corner_quality_relaxation_px =
      options_.internal_corner_filter_quality_relaxation_px;
  build_options.filter_internal_corner_adaptive_min_threshold_px =
      options_.internal_corner_filter_adaptive_min_threshold_px;
  const JointReprojectionMeasurementBuilder builder(config, build_options);
  JointResidualEvaluationOptions residual_options;
  const JointReprojectionResidualEvaluator residual_evaluator(residual_options);
  JointMeasurementSelectionOptions selection_options;
  selection_options.reference_board_id = options_.reference_board_id;
  selection_options.selection_mode = options_.selection_mode;
  selection_options.enable_residual_sanity_gate = options_.enable_residual_sanity_gate;
  selection_options.enable_board_pose_fit_gate = options_.enable_board_pose_fit_gate;
  selection_options.residual_sanity_factor =
      options_.selection_residual_sanity_factor;
  selection_options.max_board_observation_rmse =
      options_.selection_max_board_observation_rmse;
  selection_options.kalibr_style_outlier_sigma =
      options_.selection_kalibr_style_outlier_sigma;
  selection_options.kalibr_style_min_abs_threshold_px =
      options_.selection_kalibr_style_min_abs_threshold_px;
  selection_options.kalibr_style_min_views_before_filter =
      options_.selection_kalibr_style_min_views_before_filter;
  selection_options.preserve_frame_board_cohesion =
      options_.preserve_frame_board_cohesion;
  const JointMeasurementSelection selector(selection_options);
  JointOptimizationOptions optimization_options;
  optimization_options.reference_board_id = options_.reference_board_id;
  optimization_options.optimize_intrinsics = options_.optimize_intrinsics;
  optimization_options.intrinsics_release_iteration = options_.intrinsics_release_iteration;
  const JointReprojectionOptimizer optimizer(optimization_options);
  const OuterDetectionCache detection_cache(
      config.outer_detector_config,
      OuterDetectionCacheOptions{options_.enable_outer_detection_cache,
                                 options_.outer_detection_cache_dir});

  std::vector<OuterBootstrapFrameInput> bootstrap_frames;
  std::vector<InternalRegenerationFrameInput> regeneration_inputs;
  bootstrap_frames.reserve(frame_sources.size());
  regeneration_inputs.reserve(frame_sources.size());
  {
    const auto stage_start = std::chrono::steady_clock::now();
    for (const FrozenRound2BaselineFrameSource& frame_source : frame_sources) {
      const cv::Mat image = cv::imread(frame_source.image_path, cv::IMREAD_UNCHANGED);
      if (image.empty()) {
        result.failure_reason = "Failed to read image: " + frame_source.image_path;
        return result;
      }

      OuterTagMultiDetectionResult outer_detections;
      std::string cache_warning;
      if (detection_cache.Load(frame_source.image_path, &outer_detections, &cache_warning)) {
        ++result.runtime_breakdown.training_detection_cache.cache_hits;
      } else {
        ++result.runtime_breakdown.training_detection_cache.cache_misses;
        if (!cache_warning.empty()) {
          ++result.runtime_breakdown.training_detection_cache.load_failures;
          AppendUniqueWarning("Outer detection cache load warning: " + cache_warning,
                              &result.warnings);
        }
        outer_detections = outer_detector.DetectMultiple(image);
        if (detection_cache.enabled() &&
            !detection_cache.Save(frame_source.image_path, outer_detections,
                                  &cache_warning)) {
          ++result.runtime_breakdown.training_detection_cache.store_failures;
          if (!cache_warning.empty()) {
            AppendUniqueWarning("Outer detection cache store warning: " + cache_warning,
                                &result.warnings);
          }
        }
      }

      OuterBootstrapFrameInput bootstrap_input;
      bootstrap_input.frame_index = frame_source.frame_index;
      bootstrap_input.frame_label = frame_source.frame_label;
      bootstrap_input.measurements = outer_detections.frame_measurements;
      bootstrap_frames.push_back(bootstrap_input);

      InternalRegenerationFrameInput regeneration_input;
      regeneration_input.frame_index = frame_source.frame_index;
      regeneration_input.frame_label = frame_source.frame_label;
      regeneration_input.outer_detections = outer_detections;
      regeneration_inputs.push_back(regeneration_input);
    }
    result.runtime_breakdown.training_outer_detection_seconds =
        ElapsedSeconds(stage_start);
  }

  AutoCameraInitializationOptions initialization_options;
  initialization_options.mode = config.camera_initialization_mode;
  initialization_options.refine_mode = options_.camera_initialization_refine_mode;
  initialization_options.use_explicit_initial_camera =
      options_.use_explicit_initial_camera;
  initialization_options.explicit_initial_camera =
      options_.explicit_initial_camera;
  initialization_options.explicit_initial_camera_source_label =
      options_.explicit_initial_camera_source_label;
  const OuterOnlyCameraInitializer camera_initializer(config, initialization_options);
  {
    const auto stage_start = std::chrono::steady_clock::now();
    result.auto_camera_initialization = camera_initializer.Initialize(bootstrap_frames);
    result.runtime_breakdown.auto_camera_initialization_seconds =
        ElapsedSeconds(stage_start);
  }
  AppendWarnings(result.auto_camera_initialization.warnings, &result.warnings);
  if (!result.auto_camera_initialization.success) {
    result.failure_reason = result.auto_camera_initialization.failure_reason.empty()
                                ? "Automatic camera initialization failed."
                                : result.auto_camera_initialization.failure_reason;
    return result;
  }
  SetBootstrapInitFromIntrinsics(result.auto_camera_initialization.selected_camera,
                                 &bootstrap_options);

  {
    const auto stage_start = std::chrono::steady_clock::now();
    const MultiBoardOuterBootstrap bootstrap(config, bootstrap_options);
    result.bootstrap_result = bootstrap.Solve(bootstrap_frames);
    result.runtime_breakdown.outer_bootstrap_seconds = ElapsedSeconds(stage_start);
  }
  if (!result.bootstrap_result.success) {
    result.failure_reason = result.bootstrap_result.failure_reason.empty()
                                ? "Outer bootstrap failed."
                                : result.bootstrap_result.failure_reason;
    AppendWarnings(result.bootstrap_result.warnings, &result.warnings);
    return result;
  }

  const JointReprojectionSceneState initial_scene_state =
      BuildSceneStateFromBootstrap(result.bootstrap_result);

  result.outer_only_intermediate.enabled =
      options_.enable_outer_only_intermediate_calibration;
  result.outer_only_intermediate.diagnostic_only =
      options_.intermediate_diagnostic_only;
  result.outer_only_intermediate.use_for_round1_requested =
      options_.use_intermediate_for_round1_internal_regeneration;
  result.outer_only_intermediate.use_for_full_frontend_regeneration_requested =
      options_.use_intermediate_for_full_frontend_regeneration;
  result.outer_only_intermediate.max_outer_rmse_px =
      options_.intermediate_max_outer_rmse_px;
  result.outer_only_intermediate.min_visible_boards =
      options_.intermediate_min_visible_boards;
  if (options_.enable_outer_only_intermediate_calibration) {
    std::vector<JointMeasurementFrameInput> intermediate_inputs =
        BuildOuterOnlyIntermediateInputs(regeneration_inputs);
    JointMeasurementBuildOptions intermediate_build_options = build_options;
    intermediate_build_options.include_outer_points = true;
    intermediate_build_options.include_internal_points = false;
    intermediate_build_options.include_outer_when_internal_failed = true;
    intermediate_build_options.include_rescued_outer_when_internal_failed = false;
    intermediate_build_options.use_regenerated_rescued_outer_measurements = false;
    intermediate_build_options.filter_internal_corner_outliers = false;
    const JointReprojectionMeasurementBuilder intermediate_builder(
        config, intermediate_build_options);
    {
      const auto stage_start = std::chrono::steady_clock::now();
      result.outer_only_intermediate.measurement_result =
          intermediate_builder.Build(intermediate_inputs, result.bootstrap_result);
      result.runtime_breakdown
          .outer_only_intermediate_measurement_build_seconds =
          ElapsedSeconds(stage_start);
    }
    result.outer_only_intermediate.total_outer_board_observation_count =
        result.outer_only_intermediate.measurement_result
            .accepted_outer_board_observation_count;
    if (!result.outer_only_intermediate.measurement_result.success) {
      result.outer_only_intermediate.failure_reason =
          result.outer_only_intermediate.measurement_result.failure_reason.empty()
              ? "outer-only intermediate measurement build failed"
              : result.outer_only_intermediate.measurement_result.failure_reason;
      AppendWarnings(result.outer_only_intermediate.measurement_result.warnings,
                     &result.outer_only_intermediate.warnings);
      AppendWarnings(result.outer_only_intermediate.warnings, &result.warnings);
    } else {
      {
        const auto stage_start = std::chrono::steady_clock::now();
        result.outer_only_intermediate.initial_residual_result =
            residual_evaluator.Evaluate(
                result.outer_only_intermediate.measurement_result,
                initial_scene_state);
        result.runtime_breakdown
            .outer_only_intermediate_residual_evaluation_seconds =
            ElapsedSeconds(stage_start);
      }
      if (!result.outer_only_intermediate.initial_residual_result.success) {
        result.outer_only_intermediate.failure_reason =
            result.outer_only_intermediate.initial_residual_result
                    .failure_reason.empty()
                ? "outer-only intermediate initial residual evaluation failed"
                : result.outer_only_intermediate.initial_residual_result
                      .failure_reason;
        AppendWarnings(
            result.outer_only_intermediate.initial_residual_result.warnings,
            &result.outer_only_intermediate.warnings);
        AppendWarnings(result.outer_only_intermediate.warnings, &result.warnings);
      } else {
        {
          const auto stage_start = std::chrono::steady_clock::now();
          result.outer_only_intermediate.selection_result =
              SelectOuterOnlyIntermediateObservations(
                  result.outer_only_intermediate.measurement_result,
                  result.outer_only_intermediate.initial_residual_result,
                  initial_scene_state,
                  options_.intermediate_max_outer_rmse_px,
                  options_.intermediate_min_visible_boards);
          result.runtime_breakdown
              .outer_only_intermediate_selection_seconds =
              ElapsedSeconds(stage_start);
        }
        result.outer_only_intermediate.used_outer_board_observation_count =
            result.outer_only_intermediate.selection_result
                .accepted_board_observation_count;
        result.outer_only_intermediate.used_outer_point_count =
            result.outer_only_intermediate.selection_result
                .accepted_outer_point_count;
        result.outer_only_intermediate.used_internal_point_count =
            result.outer_only_intermediate.selection_result
                .accepted_internal_point_count;
        result.outer_only_intermediate.rejected_outer_board_observation_count =
            std::max(0,
                     result.outer_only_intermediate
                             .total_outer_board_observation_count -
                         result.outer_only_intermediate
                             .used_outer_board_observation_count);
        if (!result.outer_only_intermediate.selection_result.success) {
          result.outer_only_intermediate.failure_reason =
              result.outer_only_intermediate.selection_result
                      .failure_reason.empty()
                  ? "outer-only intermediate selection failed"
                  : result.outer_only_intermediate.selection_result
                        .failure_reason;
          AppendWarnings(result.outer_only_intermediate.selection_result.warnings,
                         &result.outer_only_intermediate.warnings);
          AppendWarnings(result.outer_only_intermediate.warnings,
                         &result.warnings);
        } else {
          JointOptimizationOptions intermediate_options = optimization_options;
          intermediate_options.optimize_intrinsics =
              options_.intermediate_optimize_intrinsics;
          intermediate_options.optimize_board_poses =
              options_.intermediate_optimize_board_poses;
          intermediate_options.optimize_frame_poses =
              options_.intermediate_optimize_frame_poses;
          intermediate_options.intrinsics_release_iteration =
              options_.intermediate_intrinsics_release_iteration;
          intermediate_options.cost_options.residual_model =
              ResidualModel::ImagePlane;
          const JointReprojectionOptimizer intermediate_optimizer(
              intermediate_options);
          {
            const auto stage_start = std::chrono::steady_clock::now();
            result.outer_only_intermediate.optimization_result =
                intermediate_optimizer.Optimize(
                    result.outer_only_intermediate.selection_result,
                    initial_scene_state);
            result.runtime_breakdown
                .outer_only_intermediate_optimization_seconds =
                ElapsedSeconds(stage_start);
          }
          result.outer_only_intermediate.success =
              result.outer_only_intermediate.optimization_result.success;
          if (result.outer_only_intermediate.success) {
            result.outer_only_intermediate.state_source_label =
                "outer_only_intermediate";
          } else {
            result.outer_only_intermediate.failure_reason =
                result.outer_only_intermediate.optimization_result
                        .failure_reason.empty()
                    ? "outer-only intermediate optimization failed"
                    : result.outer_only_intermediate.optimization_result
                          .failure_reason;
          }
          AppendWarnings(
              result.outer_only_intermediate.optimization_result.warnings,
              &result.outer_only_intermediate.warnings);
          AppendWarnings(result.outer_only_intermediate.warnings,
                         &result.warnings);
        }
      }
    }
  }

  const bool use_intermediate_for_round1 =
      options_.enable_outer_only_intermediate_calibration &&
      !options_.intermediate_diagnostic_only &&
      (options_.use_intermediate_for_round1_internal_regeneration ||
       options_.use_intermediate_for_full_frontend_regeneration) &&
      result.outer_only_intermediate.success;
  result.outer_only_intermediate.used_for_round1_internal_regeneration =
      use_intermediate_for_round1;
  result.outer_only_intermediate.used_for_full_frontend_regeneration =
      use_intermediate_for_round1 &&
      options_.use_intermediate_for_full_frontend_regeneration;
  const JointReprojectionSceneState* round1_regeneration_state =
      use_intermediate_for_round1
          ? &result.outer_only_intermediate.optimization_result.optimized_state
          : nullptr;

  result.round1.regeneration_results.reserve(frame_sources.size());
  result.round1.joint_inputs.reserve(frame_sources.size());
  if (options_.outer_only_ablation_mode) {
    for (std::size_t frame_index = 0; frame_index < frame_sources.size(); ++frame_index) {
      InternalRegenerationFrameResult regeneration_result;
      regeneration_result.frame_index = regeneration_inputs[frame_index].frame_index;
      regeneration_result.frame_label = regeneration_inputs[frame_index].frame_label;
      regeneration_result.frame_bootstrap_initialized = true;
      regeneration_result.state_source_label = "outer_only_ablation_skipped";
      regeneration_result.visible_board_ids =
          regeneration_inputs[frame_index].outer_detections.requested_board_ids;
      result.round1.regeneration_results.push_back(regeneration_result);

      JointMeasurementFrameInput joint_input;
      joint_input.frame_index = regeneration_inputs[frame_index].frame_index;
      joint_input.frame_label = regeneration_inputs[frame_index].frame_label;
      joint_input.outer_detections = regeneration_inputs[frame_index].outer_detections;
      joint_input.regenerated_internal = regeneration_result;
      result.round1.joint_inputs.push_back(joint_input);
    }
  } else {
    const auto stage_start = std::chrono::steady_clock::now();
    for (std::size_t frame_index = 0; frame_index < frame_sources.size(); ++frame_index) {
      const cv::Mat image =
          cv::imread(frame_sources[frame_index].image_path, cv::IMREAD_UNCHANGED);
      if (image.empty()) {
        result.failure_reason = "Failed to read image: " + frame_sources[frame_index].image_path;
        return result;
      }

      InternalRegenerationFrameResult regeneration_result;
      if (round1_regeneration_state != nullptr) {
        regeneration_result =
            regenerator.RegenerateFrame(image, regeneration_inputs[frame_index],
                                        *round1_regeneration_state);
        regeneration_result.state_source_label = "outer_only_intermediate";
      } else {
        regeneration_result =
            regenerator.RegenerateFrame(image, regeneration_inputs[frame_index],
                                        result.bootstrap_result);
      }
      AccumulateRegenerationRuntime(
          regeneration_result.runtime_breakdown,
          &result.runtime_breakdown.round1_regeneration_pose_estimation_seconds,
          &result.runtime_breakdown.round1_regeneration_boundary_model_seconds,
          &result.runtime_breakdown.round1_regeneration_seed_search_seconds,
          &result.runtime_breakdown.round1_regeneration_ray_refine_seconds,
          &result.runtime_breakdown.round1_regeneration_image_evidence_seconds,
          &result.runtime_breakdown.round1_regeneration_subpix_seconds);
      result.runtime_breakdown.round1_regeneration_attempted_internal_corners +=
          regeneration_result.runtime_breakdown.attempted_internal_corner_count;
      result.runtime_breakdown.round1_regeneration_valid_internal_corners +=
          regeneration_result.runtime_breakdown.valid_internal_corner_count;
      result.round1.regeneration_results.push_back(regeneration_result);
      for (const std::string& warning : regeneration_result.warnings) {
        AppendUniqueWarning(warning, &result.warnings);
      }

      JointMeasurementFrameInput joint_input;
      joint_input.frame_index = regeneration_inputs[frame_index].frame_index;
      joint_input.frame_label = regeneration_inputs[frame_index].frame_label;
      joint_input.outer_detections = regeneration_inputs[frame_index].outer_detections;
      joint_input.regenerated_internal = regeneration_result;
      result.round1.joint_inputs.push_back(joint_input);
    }
    result.runtime_breakdown.round1_regeneration_seconds =
        ElapsedSeconds(stage_start);
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    result.round1.measurement_result =
        builder.Build(result.round1.joint_inputs, result.bootstrap_result);
    result.runtime_breakdown.round1_measurement_build_seconds =
        ElapsedSeconds(stage_start);
  }
  result.round1.validation_summary = ValidateJointMeasurementBuilder(
      result.round1.joint_inputs, result.bootstrap_result, builder,
      result.round1.measurement_result);
  if (!result.round1.validation_summary.success) {
    result.failure_reason = result.round1.validation_summary.failure_reason;
    AppendWarnings(result.round1.validation_summary.warnings, &result.warnings);
    return result;
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    result.round1.residual_result =
        residual_evaluator.Evaluate(result.round1.measurement_result, initial_scene_state);
    result.runtime_breakdown.round1_residual_evaluation_seconds =
        ElapsedSeconds(stage_start);
  }
  if (!result.round1.residual_result.success) {
    result.failure_reason = result.round1.residual_result.failure_reason;
    AppendWarnings(result.round1.residual_result.warnings, &result.warnings);
    return result;
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    result.round1.selection_result =
        selector.Select(result.round1.measurement_result, result.round1.residual_result,
                        initial_scene_state);
    result.runtime_breakdown.round1_selection_seconds =
        ElapsedSeconds(stage_start);
  }
  if (!result.round1.selection_result.success) {
    result.failure_reason = result.round1.selection_result.failure_reason;
    AppendWarnings(result.round1.selection_result.warnings, &result.warnings);
    return result;
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    result.round1.optimization_result =
        optimizer.Optimize(result.round1.selection_result, initial_scene_state);
    result.runtime_breakdown.round1_optimization_seconds =
        ElapsedSeconds(stage_start);
    result.runtime_breakdown.round1_optimization_residual_evaluation_seconds =
        result.round1.optimization_result.runtime_breakdown.residual_evaluation_seconds;
    result.runtime_breakdown.round1_optimization_residual_evaluation_call_count =
        result.round1.optimization_result.runtime_breakdown.residual_evaluation_call_count;
    result.runtime_breakdown.round1_optimization_cost_evaluation_seconds =
        result.round1.optimization_result.runtime_breakdown.cost_evaluation_seconds;
    result.runtime_breakdown.round1_optimization_cost_evaluation_call_count =
        result.round1.optimization_result.runtime_breakdown.cost_evaluation_call_count;
    result.runtime_breakdown.round1_optimization_frame_update_seconds =
        result.round1.optimization_result.runtime_breakdown.frame_update_seconds;
    result.runtime_breakdown.round1_optimization_board_update_seconds =
        result.round1.optimization_result.runtime_breakdown.board_update_seconds;
    result.runtime_breakdown.round1_optimization_intrinsics_update_seconds =
        result.round1.optimization_result.runtime_breakdown.intrinsics_update_seconds;
  }
  if (!result.round1.optimization_result.success) {
    result.failure_reason = result.round1.optimization_result.failure_reason;
    AppendWarnings(result.round1.optimization_result.warnings, &result.warnings);
    return result;
  }

  CalibrationBundleMetadata metadata;
  metadata.bundle_version = "stage5_bundle_v1";
  metadata.baseline_protocol_label = options_.baseline_protocol_label;
  metadata.training_split_signature = options_.training_split_signature;
  metadata.dataset_label = options_.dataset_label;
  metadata.source_pipeline_label = options_.source_pipeline_label;
  result.stage5_round1_bundle = BuildCalibrationStateBundleFromJointOptimizationResult(
      result.round1.optimization_result,
      result.round1.selection_result,
      result.round1.measurement_result,
      1,
      metadata);

  result.final_stage5_bundle = result.stage5_round1_bundle;
  result.stage5_bundle_available = result.final_stage5_bundle.success;

  if (options_.run_second_pass && !options_.outer_only_ablation_mode) {
    result.round2_available = true;
    result.round2.regeneration_results.reserve(frame_sources.size());
    result.round2.joint_inputs.reserve(frame_sources.size());
    {
      const auto stage_start = std::chrono::steady_clock::now();
      for (std::size_t frame_index = 0; frame_index < frame_sources.size(); ++frame_index) {
        const cv::Mat image =
            cv::imread(frame_sources[frame_index].image_path, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
          result.failure_reason =
              "Failed to read image: " + frame_sources[frame_index].image_path;
          return result;
        }

        const InternalRegenerationFrameResult regeneration_result =
            regenerator.RegenerateFrame(
                image, regeneration_inputs[frame_index],
                result.round1.optimization_result.optimized_state);
        AccumulateRegenerationRuntime(
            regeneration_result.runtime_breakdown,
            &result.runtime_breakdown.round2_regeneration_pose_estimation_seconds,
            &result.runtime_breakdown.round2_regeneration_boundary_model_seconds,
            &result.runtime_breakdown.round2_regeneration_seed_search_seconds,
            &result.runtime_breakdown.round2_regeneration_ray_refine_seconds,
            &result.runtime_breakdown.round2_regeneration_image_evidence_seconds,
            &result.runtime_breakdown.round2_regeneration_subpix_seconds);
        result.runtime_breakdown.round2_regeneration_attempted_internal_corners +=
            regeneration_result.runtime_breakdown.attempted_internal_corner_count;
        result.runtime_breakdown.round2_regeneration_valid_internal_corners +=
            regeneration_result.runtime_breakdown.valid_internal_corner_count;
        result.round2.regeneration_results.push_back(regeneration_result);
        for (const std::string& warning : regeneration_result.warnings) {
          AppendUniqueWarning(warning, &result.warnings);
        }

        JointMeasurementFrameInput joint_input;
        joint_input.frame_index = regeneration_inputs[frame_index].frame_index;
        joint_input.frame_label = regeneration_inputs[frame_index].frame_label;
        joint_input.outer_detections = regeneration_inputs[frame_index].outer_detections;
        joint_input.regenerated_internal = regeneration_result;
        result.round2.joint_inputs.push_back(joint_input);
      }
      result.runtime_breakdown.round2_regeneration_seconds =
          ElapsedSeconds(stage_start);
    }

    {
      const auto stage_start = std::chrono::steady_clock::now();
      result.round2.measurement_result =
          builder.Build(result.round2.joint_inputs, result.bootstrap_result);
      result.runtime_breakdown.round2_measurement_build_seconds =
          ElapsedSeconds(stage_start);
    }
    result.round2.validation_summary = ValidateJointMeasurementBuilder(
        result.round2.joint_inputs, result.bootstrap_result, builder,
        result.round2.measurement_result);
    if (!result.round2.validation_summary.success) {
      result.failure_reason = result.round2.validation_summary.failure_reason;
      AppendWarnings(result.round2.validation_summary.warnings, &result.warnings);
      return result;
    }

    {
      const auto stage_start = std::chrono::steady_clock::now();
      result.round2.residual_result = residual_evaluator.Evaluate(
          result.round2.measurement_result,
          result.round1.optimization_result.optimized_state);
      result.runtime_breakdown.round2_residual_evaluation_seconds =
          ElapsedSeconds(stage_start);
    }
    if (!result.round2.residual_result.success) {
      result.failure_reason = result.round2.residual_result.failure_reason;
      AppendWarnings(result.round2.residual_result.warnings, &result.warnings);
      return result;
    }

    {
      const auto stage_start = std::chrono::steady_clock::now();
      result.round2.selection_result =
          selector.Select(result.round2.measurement_result, result.round2.residual_result,
                          result.round1.optimization_result.optimized_state);
      result.runtime_breakdown.round2_selection_seconds =
          ElapsedSeconds(stage_start);
    }
    if (!result.round2.selection_result.success) {
      result.failure_reason = result.round2.selection_result.failure_reason;
      AppendWarnings(result.round2.selection_result.warnings, &result.warnings);
      return result;
    }

    JointOptimizationOptions second_pass_options = optimization_options;
    second_pass_options.intrinsics_release_iteration =
        options_.second_pass_intrinsics_release_iteration;
    const JointReprojectionOptimizer round2_optimizer(second_pass_options);
    {
      const auto stage_start = std::chrono::steady_clock::now();
      result.round2.optimization_result = round2_optimizer.Optimize(
          result.round2.selection_result,
          result.round1.optimization_result.optimized_state);
      result.runtime_breakdown.round2_optimization_seconds =
          ElapsedSeconds(stage_start);
      result.runtime_breakdown.round2_optimization_residual_evaluation_seconds =
          result.round2.optimization_result.runtime_breakdown.residual_evaluation_seconds;
      result.runtime_breakdown.round2_optimization_residual_evaluation_call_count =
          result.round2.optimization_result.runtime_breakdown.residual_evaluation_call_count;
      result.runtime_breakdown.round2_optimization_cost_evaluation_seconds =
          result.round2.optimization_result.runtime_breakdown.cost_evaluation_seconds;
      result.runtime_breakdown.round2_optimization_cost_evaluation_call_count =
          result.round2.optimization_result.runtime_breakdown.cost_evaluation_call_count;
      result.runtime_breakdown.round2_optimization_frame_update_seconds =
          result.round2.optimization_result.runtime_breakdown.frame_update_seconds;
      result.runtime_breakdown.round2_optimization_board_update_seconds =
          result.round2.optimization_result.runtime_breakdown.board_update_seconds;
      result.runtime_breakdown.round2_optimization_intrinsics_update_seconds =
          result.round2.optimization_result.runtime_breakdown.intrinsics_update_seconds;
    }
    if (!result.round2.optimization_result.success) {
      result.failure_reason = result.round2.optimization_result.failure_reason;
      AppendWarnings(result.round2.optimization_result.warnings, &result.warnings);
      return result;
    }

    result.final_stage5_bundle = BuildCalibrationStateBundleFromJointOptimizationResult(
        result.round2.optimization_result,
        result.round2.selection_result,
        result.round2.measurement_result,
        2,
        metadata);
    result.stage5_bundle_available = result.final_stage5_bundle.success;
  }

  result.stage42_validation_pass = ComputeStage42ValidationPass(
      result.round1.selection_result,
      result.round1.optimization_result,
      result.round2_available,
      result.round2.selection_result,
      result.round2.optimization_result);
  result.success = true;
  result.warnings.clear();
  AppendWarnings(result.auto_camera_initialization.warnings, &result.warnings);
  AppendWarnings(result.round1.optimization_result.warnings, &result.warnings);
  if (result.round2_available) {
    AppendWarnings(result.round2.optimization_result.warnings, &result.warnings);
  }
  if (!result.stage5_bundle_available) {
    result.warnings.push_back(
        "Stage 5 bundle was not ready-for-backend after frozen round2 baseline.");
  }
  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
