#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
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

ApriltagInternalConfig NormalizeConfig(ApriltagInternalConfig config) {
  config.tag_ids = NormalizeBoardIds(config.tag_ids, config.tag_id);
  if (!config.tag_ids.empty()) {
    config.tag_id = config.tag_ids.front();
  }
  config.outer_detector_config.tag_ids = config.tag_ids;
  config.outer_detector_config.tag_id = config.tag_id;
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
  options.outer_detector_config = config.outer_detector_config;
  return options;
}

OuterBootstrapOptions MakeBootstrapOptions(const ApriltagInternalConfig& config,
                                           const FrozenRound2BaselineOptions& options) {
  OuterBootstrapOptions bootstrap_options;
  bootstrap_options.reference_board_id = options.reference_board_id;
  if (config.intermediate_camera.IsConfigured() &&
      config.intermediate_camera.camera_model == "ds" &&
      config.intermediate_camera.intrinsics.size() == 6 &&
      config.intermediate_camera.resolution.size() == 2 &&
      config.intermediate_camera.resolution[0] > 0 &&
      config.intermediate_camera.resolution[1] > 0) {
    bootstrap_options.init_xi = config.intermediate_camera.intrinsics[0];
    bootstrap_options.init_alpha = config.intermediate_camera.intrinsics[1];
    bootstrap_options.init_fu_scale =
        config.intermediate_camera.intrinsics[2] /
        static_cast<double>(config.intermediate_camera.resolution[0]);
    bootstrap_options.init_fv_scale =
        config.intermediate_camera.intrinsics[3] /
        static_cast<double>(config.intermediate_camera.resolution[1]);
    bootstrap_options.init_cu_offset =
        config.intermediate_camera.intrinsics[4] -
        0.5 * static_cast<double>(config.intermediate_camera.resolution[0]);
    bootstrap_options.init_cv_offset =
        config.intermediate_camera.intrinsics[5] -
        0.5 * static_cast<double>(config.intermediate_camera.resolution[1]);
  } else {
    bootstrap_options.init_xi = config.sphere_lattice_init_xi;
    bootstrap_options.init_alpha = config.sphere_lattice_init_alpha;
    bootstrap_options.init_fu_scale = config.sphere_lattice_init_fu_scale;
    bootstrap_options.init_fv_scale = config.sphere_lattice_init_fv_scale;
    bootstrap_options.init_cu_offset = config.sphere_lattice_init_cu_offset;
    bootstrap_options.init_cv_offset = config.sphere_lattice_init_cv_offset;
  }
  bootstrap_options.min_detection_quality = config.outer_detector_config.min_detection_quality;
  return bootstrap_options;
}

void SetBootstrapInitFromIntrinsics(const OuterBootstrapCameraIntrinsics& intrinsics,
                                    OuterBootstrapOptions* options) {
  if (options == nullptr) {
    throw std::runtime_error("SetBootstrapInitFromIntrinsics requires a valid options pointer.");
  }
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
  const ApriltagInternalDetectionOptions detection_options = MakeDetectionOptions(config);
  const MultiScaleOuterTagDetector outer_detector(config.outer_detector_config);
  OuterBootstrapOptions bootstrap_options = MakeBootstrapOptions(config, options_);
  const MultiBoardOuterBootstrap bootstrap(config, bootstrap_options);
  const MultiBoardInternalMeasurementRegenerator regenerator(config, detection_options);
  JointMeasurementBuildOptions build_options;
  build_options.reference_board_id = options_.reference_board_id;
  const JointReprojectionMeasurementBuilder builder(config, build_options);
  JointResidualEvaluationOptions residual_options;
  const JointReprojectionResidualEvaluator residual_evaluator(residual_options);
  JointMeasurementSelectionOptions selection_options;
  selection_options.reference_board_id = options_.reference_board_id;
  selection_options.enable_residual_sanity_gate = options_.enable_residual_sanity_gate;
  selection_options.enable_board_pose_fit_gate = options_.enable_board_pose_fit_gate;
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

  result.round1.regeneration_results.reserve(frame_sources.size());
  result.round1.joint_inputs.reserve(frame_sources.size());
  {
    const auto stage_start = std::chrono::steady_clock::now();
    for (std::size_t frame_index = 0; frame_index < frame_sources.size(); ++frame_index) {
      const cv::Mat image =
          cv::imread(frame_sources[frame_index].image_path, cv::IMREAD_UNCHANGED);
      if (image.empty()) {
        result.failure_reason = "Failed to read image: " + frame_sources[frame_index].image_path;
        return result;
      }

      const InternalRegenerationFrameResult regeneration_result =
          regenerator.RegenerateFrame(image, regeneration_inputs[frame_index],
                                      result.bootstrap_result);
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

  const JointReprojectionSceneState initial_scene_state =
      BuildSceneStateFromBootstrap(result.bootstrap_result);
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

  if (options_.run_second_pass) {
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
