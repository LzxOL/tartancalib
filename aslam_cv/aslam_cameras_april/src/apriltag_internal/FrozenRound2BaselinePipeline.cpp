#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>

#include <aslam/cameras/apriltag_internal/CameraAwareOuterRescue.hpp>

#include <aslam/cameras/apriltag_internal/BoardDetectionPipeline.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/JointMeasurementSelection.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionMeasurementBuilder.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionOptimizer.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionResidualEvaluator.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

class Stage5ProgressReporter {
 public:
  Stage5ProgressReporter(bool enabled, const std::string& log_path,
                         int interval_frames)
      : enabled_(enabled),
        interval_frames_(std::max(1, interval_frames)),
        start_time_(std::chrono::steady_clock::now()) {
    if (enabled_ && !log_path.empty()) {
      log_.open(log_path.c_str(), std::ios::out | std::ios::app);
    }
  }

  void StageStart(const std::string& stage, std::size_t total) {
    if (!enabled_) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    stage_completed_[stage] = 0;
    ReportUnlocked(stage, 0, total, "started");
  }

  void StageEnd(const std::string& stage, std::size_t total,
                const std::string& detail) {
    if (!enabled_) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = stage_completed_.find(stage);
    const std::size_t completed =
        it == stage_completed_.end() || it->second == 0 ? total : it->second;
    ReportUnlocked(stage, completed, total, detail);
  }

  void FrameDone(const std::string& stage, std::size_t total,
                 const std::string& detail) {
    if (!enabled_) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t completed = ++stage_completed_[stage];
    if (completed != total &&
        completed % static_cast<std::size_t>(interval_frames_) != 0) {
      return;
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time_).count();
    const double rate = completed > 0 ? static_cast<double>(completed) / elapsed : 0.0;
    const double eta = rate > 0.0 && total >= completed
                           ? static_cast<double>(total - completed) / rate
                           : 0.0;
    std::ostringstream message;
    message << "stage=" << stage << " completed=" << completed << "/" << total
            << " percent=" << std::fixed << std::setprecision(1)
            << (total > 0 ? 100.0 * static_cast<double>(completed) /
                                  static_cast<double>(total)
                            : 100.0)
            << " elapsed_s=" << std::setprecision(1) << elapsed
            << " eta_s=" << eta;
    if (!detail.empty()) {
      message << " " << detail;
    }
    WriteLocked(message.str());
  }

 private:
  // StageStart/StageEnd already hold mutex_. Avoid recursive locking here.
  void ReportUnlocked(const std::string& stage, std::size_t completed,
                      std::size_t total, const std::string& detail) {
    std::ostringstream message;
    message << "stage=" << stage << " completed=" << completed << "/" << total;
    if (total > 0) {
      message << " percent=" << std::fixed << std::setprecision(1)
              << (100.0 * static_cast<double>(completed) /
                  static_cast<double>(total));
    }
    if (!detail.empty()) {
      message << " " << detail;
    }
    WriteLocked(message.str());
  }

  void WriteLocked(const std::string& message) {
    const std::string line = "[stage5_progress] " + message;
    std::cout << line << std::endl;
    if (log_.is_open()) {
      log_ << line << "\n";
      log_.flush();
    }
  }

  bool enabled_ = false;
  int interval_frames_ = 1;
  std::chrono::steady_clock::time_point start_time_;
  std::mutex mutex_;
  std::map<std::string, std::size_t> stage_completed_;
  std::ofstream log_;
};

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

// Internal regeneration is independent per frame.  Keep the worker count
// bounded because each worker owns a large decoded image and the regenerator
// also performs OpenCV operations.  The final merge remains frame-ordered, so
// parallel execution does not change the observable result ordering.
std::size_t ChooseRegenerationWorkerCount(std::size_t frame_count) {
  if (frame_count == 0) {
    return 0;
  }
  const unsigned int hardware_workers = std::thread::hardware_concurrency();
  const std::size_t usable_workers =
      hardware_workers > 1 ? static_cast<std::size_t>(hardware_workers - 1) : 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(frame_count, std::min<std::size_t>(usable_workers, 8)));
}

struct ParallelRegenerationOutput {
  std::vector<InternalRegenerationFrameResult> regeneration_results;
  std::vector<std::vector<std::string> > frame_warnings;
  std::string failure_reason;
};

using RegenerateFrameFunction = std::function<InternalRegenerationFrameResult(
    BoardDetectionPipeline&, const cv::Mat&, const InternalRegenerationFrameInput&)>;

ParallelRegenerationOutput RegenerateFramesInParallel(
    const std::vector<FrozenRound2BaselineFrameSource>& frame_sources,
    const std::vector<InternalRegenerationFrameInput>& regeneration_inputs,
    const ApriltagInternalConfig& config,
    const ApriltagInternalDetectionOptions& detection_options,
    const InternalRegenerationCache& internal_regeneration_cache,
    const std::string& regeneration_state_signature,
    Stage5ProgressReporter* progress,
    const RegenerateFrameFunction& regenerate_frame) {
  ParallelRegenerationOutput output;
  if (frame_sources.size() != regeneration_inputs.size()) {
    output.failure_reason =
        "Parallel internal regeneration received mismatched frame/input counts.";
    return output;
  }

  const std::size_t frame_count = frame_sources.size();
  if (progress != nullptr) {
    progress->StageStart("internal_regeneration", frame_count);
  }
  output.regeneration_results.resize(frame_count);
  output.frame_warnings.resize(frame_count);
  if (frame_count == 0) {
    return output;
  }

  // Cache reads are immutable after stage preparation. Only artifact writes
  // need serialization because OpenCV FileStorage writes through temp files.
  std::mutex cache_mutex;
  std::mutex error_mutex;
  std::atomic<std::size_t> next_frame{0};
  std::string first_error;
  const std::size_t worker_count = ChooseRegenerationWorkerCount(frame_count);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  const auto worker = [&]() {
    // Detector/regenerator state is not shared between workers.  This avoids
    // introducing locks into the existing board recovery implementation.
    BoardDetectionPipeline worker_pipeline(config, detection_options);
    for (;;) {
      const std::size_t frame_index = next_frame.fetch_add(1);
      if (frame_index >= frame_count) {
        break;
      }

      try {
        InternalRegenerationFrameResult regeneration_result;
        std::string internal_cache_warning;
        bool internal_cache_hit = false;
        internal_cache_hit = internal_regeneration_cache.Load(
            frame_sources[frame_index].image_path,
            regeneration_inputs[frame_index], regeneration_state_signature,
            &regeneration_result, &internal_cache_warning);
        if (!internal_cache_hit) {
          if (!internal_cache_warning.empty()) {
            AppendUniqueWarning(
                "Internal regeneration cache load warning: " +
                    internal_cache_warning,
                &output.frame_warnings[frame_index]);
          }
          const cv::Mat image = cv::imread(
              frame_sources[frame_index].image_path, cv::IMREAD_UNCHANGED);
          if (image.empty()) {
            throw std::runtime_error("Failed to read image: " +
                                     frame_sources[frame_index].image_path);
          }
          // The detector accepts color images, but internal cornerSubPix is
          // defined only for a single-channel image. Normalize once at the
          // parallel regeneration boundary so all recovery paths agree.
          cv::Mat gray;
          if (image.channels() == 1) {
            gray = image;
          } else if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
          } else if (image.channels() == 4) {
            cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
          } else {
            throw std::runtime_error("Unsupported image channel count for internal regeneration: " +
                                     std::to_string(image.channels()));
          }
          regeneration_result = regenerate_frame(
              worker_pipeline, gray, regeneration_inputs[frame_index]);
          internal_cache_warning.clear();
          if (internal_regeneration_cache.enabled()) {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (!internal_regeneration_cache.Save(
                    frame_sources[frame_index].image_path,
                    regeneration_inputs[frame_index], regeneration_state_signature,
                    regeneration_result, &internal_cache_warning) &&
                !internal_cache_warning.empty()) {
              AppendUniqueWarning(
                  "Internal regeneration cache store warning: " +
                      internal_cache_warning,
                  &output.frame_warnings[frame_index]);
            }
          }
        }
        const int successful_boards = regeneration_result.SuccessfulBoardCount();
        const int valid_internal = regeneration_result.ValidInternalCornerCount();
        output.regeneration_results[frame_index] = std::move(regeneration_result);
        if (progress != nullptr) {
          progress->FrameDone(
              "internal_regeneration", frame_count,
              "frame=" + frame_sources[frame_index].frame_label +
                  " boards=" + std::to_string(successful_boards) +
                  " valid_internal=" + std::to_string(valid_internal));
        }
      } catch (const std::exception& exception) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error.empty()) {
          first_error = "Parallel internal regeneration failed at frame " +
                        std::to_string(frame_sources[frame_index].frame_index) +
                        ": " + exception.what();
        }
      }
    }
  };

  for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
    workers.emplace_back(worker);
  }
  for (std::thread& worker_thread : workers) {
    worker_thread.join();
  }
  output.failure_reason = first_error;
  if (progress != nullptr) {
    progress->StageEnd("internal_regeneration", frame_count,
                       first_error.empty() ? "finished" : "failed");
  }
  return output;
}

struct ParallelOuterDetectionOutput {
  std::vector<OuterTagMultiDetectionResult> detections;
  std::vector<OuterDetectionCacheLoadSource> cache_sources;
  std::vector<unsigned char> cache_hits;
  std::vector<unsigned char> cache_load_failures;
  std::vector<unsigned char> cache_store_failures;
  std::vector<std::vector<std::string> > frame_warnings;
  std::string failure_reason;
};

ParallelOuterDetectionOutput DetectOuterFramesInParallel(
    const std::vector<FrozenRound2BaselineFrameSource>& frame_sources,
    const ApriltagInternalConfig& config,
    const ApriltagInternalDetectionOptions& detection_options,
    const OuterDetectionCache& detection_cache,
    Stage5ProgressReporter* progress) {
  ParallelOuterDetectionOutput output;
  const std::size_t frame_count = frame_sources.size();
  if (progress != nullptr) {
    progress->StageStart("outer_detection", frame_count);
  }
  output.detections.resize(frame_count);
  output.cache_sources.resize(frame_count, OuterDetectionCacheLoadSource::None);
  output.cache_hits.resize(frame_count, 0);
  output.cache_load_failures.resize(frame_count, 0);
  output.cache_store_failures.resize(frame_count, 0);
  output.frame_warnings.resize(frame_count);
  if (frame_count == 0) {
    return output;
  }

  std::mutex cache_mutex;
  std::mutex error_mutex;
  std::atomic<std::size_t> next_frame{0};
  std::string first_error;
  const std::size_t worker_count = ChooseRegenerationWorkerCount(frame_count);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  const auto worker = [&]() {
    BoardDetectionPipeline worker_pipeline(config, detection_options);
    for (;;) {
      const std::size_t frame_index = next_frame.fetch_add(1);
      if (frame_index >= frame_count) {
        break;
      }
      try {
        OuterTagMultiDetectionResult outer_detections;
        std::string cache_warning;
        OuterDetectionCacheLoadSource cache_source =
            OuterDetectionCacheLoadSource::None;
        bool cache_hit = false;
        cache_hit = detection_cache.Load(
            frame_sources[frame_index].image_path, &outer_detections,
            &cache_warning, &cache_source);
        if (cache_hit) {
          output.cache_hits[frame_index] = 1;
          output.cache_sources[frame_index] = cache_source;
        } else {
          if (!cache_warning.empty()) {
            output.cache_load_failures[frame_index] = 1;
            AppendUniqueWarning(
                "Outer detection cache load warning: " + cache_warning,
                &output.frame_warnings[frame_index]);
          }
          const cv::Mat image = cv::imread(
              frame_sources[frame_index].image_path, cv::IMREAD_UNCHANGED);
          if (image.empty()) {
            throw std::runtime_error("Failed to read image: " +
                                     frame_sources[frame_index].image_path);
          }
          outer_detections = worker_pipeline.DetectOuter(image);
          cache_warning.clear();
          if (detection_cache.enabled()) {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (!detection_cache.Save(frame_sources[frame_index].image_path,
                                      outer_detections, &cache_warning) &&
                !cache_warning.empty()) {
              output.cache_store_failures[frame_index] = 1;
              AppendUniqueWarning(
                  "Outer detection cache store warning: " + cache_warning,
                  &output.frame_warnings[frame_index]);
            }
          }
        }
        const int successful_boards = outer_detections.SuccessfulBoardCount();
        output.detections[frame_index] = std::move(outer_detections);
        if (progress != nullptr) {
          progress->FrameDone(
              "outer_detection", frame_count,
              "frame=" + frame_sources[frame_index].frame_label +
                  " boards=" + std::to_string(successful_boards) +
                  (cache_hit ? " source=cache" : " source=detector"));
        }
      } catch (const std::exception& exception) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error.empty()) {
          first_error = "Parallel outer detection failed at frame " +
                        std::to_string(frame_sources[frame_index].frame_index) +
                        ": " + exception.what();
        }
      }
    }
  };

  for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
    workers.emplace_back(worker);
  }
  for (std::thread& worker_thread : workers) {
    worker_thread.join();
  }
  output.failure_reason = first_error;
  if (progress != nullptr) {
    progress->StageEnd("outer_detection", frame_count,
                       first_error.empty() ? "finished" : "failed");
  }
  return output;
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
  bootstrap_options.optimize_intrinsics = options.optimize_bootstrap_intrinsics;
  bootstrap_options.robust_board_layout_consensus =
      options.robust_board_layout_consensus;
  bootstrap_options.fixed_board_layout = options.fixed_board_layout;
  bootstrap_options.fixed_board_layout_source = options.fixed_board_layout_source;
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
    std::ostringstream diagnostics;
    diagnostics << "Joint measurement validation flags: counting_consistent="
                << (summary.counting_consistent ? 1 : 0)
                << " flat_hierarchical_consistent="
                << (summary.flat_hierarchical_consistent ? 1 : 0)
                << " frame_order_invariant="
                << (summary.frame_order_invariant ? 1 : 0)
                << " label_mismatch_warning_observed="
                << (summary.label_mismatch_warning_observed ? 1 : 0)
                << " primary_used_frames=" << primary_result.used_frame_count
                << " primary_used_boards="
                << primary_result.used_board_observation_count
                << " primary_used_outer=" << primary_result.used_outer_point_count
                << " primary_accepted_outer_boards="
                << primary_result.accepted_outer_board_observation_count
                << " primary_used_internal="
                << primary_result.used_internal_point_count
                << " primary_solver_points="
                << primary_result.solver_observations.size();
    const std::string diagnostic_text = diagnostics.str();
    summary.warnings.push_back(diagnostic_text);
    summary.failure_reason += " " + diagnostic_text;
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

bool HasSolverReadyReferenceBoard(
    const JointMeasurementFrameResult& frame, int reference_board_id) {
  for (const JointBoardObservation& board : frame.board_observations) {
    if (board.board_id != reference_board_id) {
      continue;
    }
    return std::any_of(
        board.points.begin(), board.points.end(),
        [](const JointPointObservation& point) {
          return point.used_in_solver;
        });
  }
  return false;
}

JointMeasurementBuildResult BuildNonReferenceSupplementMeasurements(
    const JointMeasurementBuildResult& measurements,
    int reference_board_id,
    int* candidate_frame_count = nullptr) {
  JointMeasurementBuildResult supplement = measurements;
  supplement.frames.clear();
  int retained_frame_count = 0;
  for (const JointMeasurementFrameResult& frame : measurements.frames) {
    if (HasSolverReadyReferenceBoard(frame, reference_board_id)) {
      continue;
    }
    supplement.frames.push_back(frame);
    ++retained_frame_count;
  }
  RecomputeMeasurementCounts(&supplement);
  if (candidate_frame_count != nullptr) {
    *candidate_frame_count = retained_frame_count;
  }
  return supplement;
}

bool SameMeasurementPoint(const JointPointObservation& lhs,
                          const JointPointObservation& rhs) {
  return lhs.frame_index == rhs.frame_index &&
         lhs.board_id == rhs.board_id && lhs.point_id == rhs.point_id &&
         lhs.source_point_index == rhs.source_point_index &&
         lhs.source_kind == rhs.source_kind;
}

JointMeasurementSelectionResult MergeSupplementalSelection(
    const JointMeasurementSelectionResult& reference_selection,
    const JointMeasurementSelectionResult& supplement_selection,
    int supplement_candidate_frame_count) {
  JointMeasurementSelectionResult merged = reference_selection;
  std::map<int, const JointMeasurementFrameResult*> supplement_frames;
  for (const JointMeasurementFrameResult& frame :
       supplement_selection.selected_measurement_result.frames) {
    supplement_frames[frame.frame_index] = &frame;
  }
  for (JointMeasurementFrameResult& frame :
       merged.selected_measurement_result.frames) {
    const auto frame_it = supplement_frames.find(frame.frame_index);
    if (frame_it == supplement_frames.end()) {
      continue;
    }
    for (JointBoardObservation& board : frame.board_observations) {
      const JointBoardObservation* supplement_board = nullptr;
      for (const JointBoardObservation& candidate :
           frame_it->second->board_observations) {
        if (candidate.board_id == board.board_id) {
          supplement_board = &candidate;
          break;
        }
      }
      if (supplement_board == nullptr) {
        continue;
      }
      for (JointPointObservation& point : board.points) {
        const auto source_it = std::find_if(
            supplement_board->points.begin(), supplement_board->points.end(),
            [&](const JointPointObservation& candidate) {
              return SameMeasurementPoint(point, candidate);
            });
        if (source_it == supplement_board->points.end() ||
            !source_it->used_in_solver) {
          continue;
        }
        point.used_in_solver = true;
        point.rejection_reason_code = source_it->rejection_reason_code;
        point.rejection_detail = source_it->rejection_detail;
      }
    }
  }
  RecomputeMeasurementCounts(&merged.selected_measurement_result);
  merged.accepted_frame_indices.insert(
      supplement_selection.accepted_frame_indices.begin(),
      supplement_selection.accepted_frame_indices.end());
  merged.accepted_board_observation_keys.insert(
      supplement_selection.accepted_board_observation_keys.begin(),
      supplement_selection.accepted_board_observation_keys.end());
  merged.frame_decisions.insert(merged.frame_decisions.end(),
                                supplement_selection.frame_decisions.begin(),
                                supplement_selection.frame_decisions.end());
  merged.board_observation_decisions.insert(
      merged.board_observation_decisions.end(),
      supplement_selection.board_observation_decisions.begin(),
      supplement_selection.board_observation_decisions.end());
  merged.accepted_frame_count =
      merged.selected_measurement_result.used_frame_count;
  merged.accepted_board_observation_count =
      merged.selected_measurement_result.used_board_observation_count;
  merged.accepted_outer_point_count =
      merged.selected_measurement_result.used_outer_point_count;
  merged.accepted_internal_point_count =
      merged.selected_measurement_result.used_internal_point_count;
  merged.success = merged.accepted_board_observation_count > 0 &&
                   merged.selected_measurement_result.used_total_point_count > 0;
  std::ostringstream warning;
  warning << "Round 2 supplemental non-reference selection considered "
          << supplement_candidate_frame_count << " frame(s) and admitted "
          << supplement_selection.accepted_frame_count
          << " frame(s) without reselecting reference-supported frames.";
  AppendUniqueWarning(warning.str(), &merged.warnings);
  return merged;
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

JointMeasurementBuildResult SynchronizePrecomputedMeasurementsWithBootstrap(
    const JointMeasurementBuildResult& imported,
    const OuterBootstrapResult& bootstrap) {
  JointMeasurementBuildResult synchronized = imported;
  synchronized.reference_board_id = bootstrap.reference_board_id;
  synchronized.bootstrap_seed = bootstrap;

  std::map<int, const OuterBootstrapFrameState*> frame_states;
  for (const OuterBootstrapFrameState& frame : bootstrap.frames) {
    frame_states[frame.frame_index] = &frame;
  }
  std::map<int, const OuterBootstrapBoardState*> board_states;
  for (const OuterBootstrapBoardState& board : bootstrap.boards) {
    board_states[board.board_id] = &board;
  }
  std::map<std::pair<int, int>, const OuterBootstrapObservationDiagnostics*>
      observation_states;
  for (const OuterBootstrapObservationDiagnostics& observation :
       bootstrap.observation_diagnostics) {
    observation_states[std::make_pair(observation.frame_index,
                                      observation.board_id)] = &observation;
  }

  for (JointMeasurementFrameResult& frame : synchronized.frames) {
    const auto frame_it = frame_states.find(frame.frame_index);
    const bool frame_initialized =
        frame_it != frame_states.end() && frame_it->second->initialized;
    frame.frame_bootstrap_initialized = frame_initialized;
    for (JointBoardObservation& board : frame.board_observations) {
      const auto board_it = board_states.find(board.board_id);
      const bool board_initialized =
          board.board_id == bootstrap.reference_board_id ||
          (board_it != board_states.end() && board_it->second->initialized);
      const auto observation_it = observation_states.find(
          std::make_pair(frame.frame_index, board.board_id));
      const bool reference_connected =
          observation_it != observation_states.end() &&
          observation_it->second->reference_connected;
      const bool solver_ready = frame_initialized && board_initialized &&
                                reference_connected;
      board.frame_bootstrap_initialized = frame_initialized;
      board.board_bootstrap_initialized = board_initialized;
      board.reference_connected = reference_connected;
      for (JointPointObservation& point : board.points) {
        point.used_in_solver = point.used_in_solver && solver_ready;
        if (!point.used_in_solver &&
            point.rejection_reason_code == JointRejectionReasonCode::None) {
          if (!frame_initialized) {
            point.rejection_reason_code =
                JointRejectionReasonCode::FrameNotInitialized;
            point.rejection_detail =
                "precomputed observation frame was not initialized by outer bootstrap";
          } else if (!board_initialized) {
            point.rejection_reason_code =
                JointRejectionReasonCode::BoardNotInitialized;
            point.rejection_detail =
                "precomputed observation board was not initialized by outer bootstrap";
          } else if (!reference_connected) {
            point.rejection_reason_code =
                JointRejectionReasonCode::NotReferenceConnected;
            point.rejection_detail =
                "precomputed observation was not connected to the reference board";
          }
        }
      }
    }
  }
  RecomputeMeasurementCounts(&synchronized);
  if (!synchronized.success) {
    synchronized.failure_reason =
        "No imported observations remained solver-ready after outer bootstrap.";
  }
  return synchronized;
}

JointMeasurementBuildValidationSummary ValidatePrecomputedMeasurements(
    const JointMeasurementBuildResult& measurements) {
  JointMeasurementBuildValidationSummary summary;
  int hierarchical_point_count = 0;
  std::set<std::pair<int, int> > board_keys;
  for (const JointMeasurementFrameResult& frame : measurements.frames) {
    for (const JointBoardObservation& board : frame.board_observations) {
      bool board_used = false;
      for (const JointPointObservation& point : board.points) {
        if (point.used_in_solver) {
          ++hierarchical_point_count;
          board_used = true;
        }
      }
      if (board_used) {
        board_keys.insert(std::make_pair(frame.frame_index, board.board_id));
      }
    }
  }
  summary.counting_consistent =
      measurements.used_total_point_count == hierarchical_point_count &&
      measurements.used_total_point_count ==
          measurements.used_outer_point_count +
              measurements.used_internal_point_count &&
      measurements.used_board_observation_count ==
          static_cast<int>(board_keys.size());
  summary.flat_hierarchical_consistent =
      hierarchical_point_count ==
      static_cast<int>(measurements.solver_observations.size());
  // The importer performs the equivalent strict ordering and label checks.
  summary.frame_order_invariant = true;
  summary.label_mismatch_warning_observed = true;
  summary.success = measurements.success && summary.counting_consistent &&
                    summary.flat_hierarchical_consistent;
  if (!summary.success) {
    summary.failure_reason =
        "Precomputed measurement hierarchy/count validation failed.";
  }
  return summary;
}

bool BuildSingleBoardDenseBootstrap(
    const JointMeasurementBuildResult& measurements,
    const OuterBootstrapCameraIntrinsics& camera,
    int reference_board_id,
    OuterBootstrapResult* bootstrap,
    std::string* failure_reason) {
  if (bootstrap == nullptr) {
    throw std::runtime_error(
        "BuildSingleBoardDenseBootstrap requires a bootstrap result.");
  }
  bootstrap->success = false;
  bootstrap->failure_reason.clear();
  bootstrap->reference_board_id = reference_board_id;
  bootstrap->coarse_camera = camera;
  bootstrap->boards.clear();
  bootstrap->frames.clear();
  bootstrap->observation_diagnostics.clear();
  bootstrap->used_frame_count = 0;
  bootstrap->used_board_observation_count = 0;
  bootstrap->used_corner_count = 0;
  bootstrap->global_rmse = std::numeric_limits<double>::infinity();

  OuterBootstrapBoardState board_state;
  board_state.board_id = reference_board_id;
  board_state.initialized = true;
  board_state.T_reference_board = Eigen::Matrix4d::Identity();

  double squared_rmse_sum = 0.0;
  for (const JointMeasurementFrameResult& frame : measurements.frames) {
    const JointBoardObservation* reference_observation = nullptr;
    for (const JointBoardObservation& board : frame.board_observations) {
      if (board.board_id == reference_board_id) {
        reference_observation = &board;
        break;
      }
    }
    if (reference_observation == nullptr) {
      continue;
    }

    std::vector<Eigen::Vector3d> object_points;
    std::vector<cv::Point2f> image_points;
    for (const JointPointObservation& point : reference_observation->points) {
      if (!point.used_in_solver || !point.target_xyz_board.allFinite() ||
          !point.image_xy.allFinite()) {
        continue;
      }
      object_points.push_back(point.target_xyz_board);
      image_points.emplace_back(static_cast<float>(point.image_xy.x()),
                                static_cast<float>(point.image_xy.y()));
    }

    Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
    double pose_rmse = std::numeric_limits<double>::infinity();
    const bool pose_success = EstimatePoseFromObjectPoints(
        camera, object_points, image_points, &T_camera_board, &pose_rmse);

    OuterBootstrapFrameState frame_state;
    frame_state.frame_index = frame.frame_index;
    frame_state.frame_label = frame.frame_label;
    frame_state.visible_board_ids.push_back(reference_board_id);
    frame_state.initialized = pose_success;
    frame_state.observation_count = pose_success ? 1 : 0;
    frame_state.rmse = pose_success ? pose_rmse : 0.0;
    if (pose_success) {
      frame_state.T_camera_reference = T_camera_board.matrix();
      ++bootstrap->used_frame_count;
      ++bootstrap->used_board_observation_count;
      bootstrap->used_corner_count += static_cast<int>(object_points.size());
      squared_rmse_sum += pose_rmse * pose_rmse;
      ++board_state.observation_count;
    }
    bootstrap->frames.push_back(frame_state);

    OuterBootstrapObservationDiagnostics diagnostics;
    diagnostics.frame_index = frame.frame_index;
    diagnostics.frame_label = frame.frame_label;
    diagnostics.board_id = reference_board_id;
    diagnostics.detection_quality = 1.0;
    diagnostics.reference_connected = pose_success;
    diagnostics.frame_initialized = pose_success;
    diagnostics.board_initialized = true;
    diagnostics.used_in_solve = pose_success;
    diagnostics.observation_rmse = pose_success ? pose_rmse : 0.0;
    bootstrap->observation_diagnostics.push_back(diagnostics);
  }

  board_state.rmse =
      board_state.observation_count > 0
          ? std::sqrt(squared_rmse_sum /
                      static_cast<double>(board_state.observation_count))
          : 0.0;
  bootstrap->boards.push_back(board_state);
  bootstrap->global_rmse = board_state.rmse;
  bootstrap->success = bootstrap->used_frame_count > 0;
  if (!bootstrap->success) {
    bootstrap->failure_reason =
        "SingleBoardDensePoseInitializationFailed";
    if (failure_reason != nullptr) {
      *failure_reason = bootstrap->failure_reason;
    }
    return false;
  }
  bootstrap->warnings.push_back(
      "Single-board dense bootstrap kept the camera initialization seed fixed and estimated each frame pose from all imported checkerboard control points.");
  return true;
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

  Stage5ProgressReporter progress(options_.enable_progress_reporting,
                                  options_.progress_log_path,
                                  options_.progress_report_interval_frames);

  ApriltagInternalConfig config = NormalizeConfig(options_.config);
  if (std::find(config.tag_ids.begin(), config.tag_ids.end(),
                options_.reference_board_id) == config.tag_ids.end()) {
    result.failure_reason =
        "Configured reference_board_id is absent from the explicit tag_ids topology.";
    return result;
  }
  config.outer_detector_config.enable_robust_missing_board_recovery =
      options_.enable_robust_missing_board_recovery;
  config.outer_detector_config.enable_opencv_apriltag_fallback =
      options_.enable_opencv_apriltag_fallback;
  result.effective_options.config.outer_detector_config
      .enable_opencv_apriltag_fallback =
      options_.enable_opencv_apriltag_fallback;
  if (options_.enable_camera_aware_outer_rescue) {
    // The first detection pass must remain independent of YAML/camchain
    // intrinsics. A camera is injected only after outer-only initialization.
    config.outer_detector_config.refine_camera = OuterRefineCameraConfig{};
  }
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
  detection_options.enable_internal_lattice_slot_ownership_check =
      options_.enable_internal_lattice_slot_ownership_check;
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
  detection_options.geometry_prior_rescue_scale_aware_outer_rmse_gate =
      options_.geometry_prior_rescue_scale_aware_outer_rmse_gate;
  detection_options.geometry_prior_rescue_accept_max_rotation_error_deg =
      options_.geometry_prior_rescue_accept_max_rotation_error_deg;
  detection_options.geometry_prior_rescue_accept_max_translation_error =
      options_.geometry_prior_rescue_accept_max_translation_error;
  detection_options.geometry_guided_tag_likelihood_enabled =
      options_.geometry_guided_tag_likelihood_enabled;
  detection_options.geometry_guided_tag_likelihood_min_visible_boards =
      options_.geometry_guided_tag_likelihood_min_visible_boards;
  detection_options.geometry_guided_tag_likelihood_max_expected_hamming =
      options_.geometry_guided_tag_likelihood_max_expected_hamming;
  detection_options.geometry_guided_tag_likelihood_min_hamming_margin =
      options_.geometry_guided_tag_likelihood_min_hamming_margin;
  detection_options.geometry_guided_tag_likelihood_min_contrast =
      options_.geometry_guided_tag_likelihood_min_contrast;
  detection_options.geometry_guided_tag_likelihood_allow_single_anchor =
      options_.geometry_guided_tag_likelihood_allow_single_anchor;
  detection_options.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse =
      options_.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse;
  detection_options.geometry_guided_tag_likelihood_single_anchor_max_expected_hamming =
      options_.geometry_guided_tag_likelihood_single_anchor_max_expected_hamming;
  detection_options.geometry_guided_tag_likelihood_single_anchor_min_hamming_margin =
      options_.geometry_guided_tag_likelihood_single_anchor_min_hamming_margin;
  detection_options.geometry_guided_tag_likelihood_single_anchor_min_contrast =
      options_.geometry_guided_tag_likelihood_single_anchor_min_contrast;
  OuterBootstrapOptions bootstrap_options = MakeBootstrapOptions(config, options_);
  JointMeasurementBuildOptions build_options;
  build_options.reference_board_id = options_.reference_board_id;
  build_options.include_internal_points = options_.include_internal_points;
  build_options.include_outer_when_internal_failed =
      options_.outer_only_ablation_mode ||
      !options_.strict_board_observation_acceptance;
  build_options.include_rescued_outer_when_internal_failed =
      options_.geometry_prior_rescue_keep_outer_on_internal_failure;
  build_options.robust_missing_board_recovery =
      options_.enable_robust_missing_board_recovery;
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
  build_options.enable_bidirectional_board_topology_consistency =
      options_.enable_bidirectional_board_topology_consistency;
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
  JointMeasurementSelectionOptions supplemental_selection_options = selection_options;
  supplemental_selection_options.require_reference_board_per_frame = false;
  const JointMeasurementSelection supplemental_selector(supplemental_selection_options);
  JointOptimizationOptions optimization_options;
  optimization_options.reference_board_id = options_.reference_board_id;
  optimization_options.optimize_intrinsics = options_.optimize_intrinsics;
  optimization_options.optimize_board_poses =
      options_.fixed_board_layout.empty();
  optimization_options.intrinsics_release_iteration = options_.intrinsics_release_iteration;
  const JointReprojectionOptimizer optimizer(optimization_options);
  const OuterDetectionCache detection_cache(
      config.outer_detector_config,
      OuterDetectionCacheOptions{options_.enable_outer_detection_cache,
                                 options_.outer_detection_cache_dir});
  const InternalRegenerationCache internal_regeneration_cache(
      config, detection_options,
      InternalRegenerationCacheOptions{options_.enable_outer_detection_cache,
                                        options_.outer_detection_cache_dir});

  if (!frame_sources.empty() && options_.enable_outer_detection_cache) {
    std::string cache_warning;
    if (!detection_cache.PrepareForDataset(frame_sources.front().image_path,
                                           &cache_warning) ||
        !internal_regeneration_cache.PrepareForDataset(
            frame_sources.front().image_path, &cache_warning)) {
      AppendUniqueWarning("Cache manifest preparation warning: " +
                              cache_warning,
                          &result.warnings);
    }
  }

  std::vector<OuterBootstrapFrameInput> bootstrap_frames;
  std::vector<InternalRegenerationFrameInput> regeneration_inputs;
  bootstrap_frames.reserve(frame_sources.size());
  regeneration_inputs.reserve(frame_sources.size());
  {
    const auto stage_start = std::chrono::steady_clock::now();
    const ParallelOuterDetectionOutput parallel_output =
        DetectOuterFramesInParallel(frame_sources, config, detection_options,
                                    detection_cache, &progress);
    if (!parallel_output.failure_reason.empty()) {
      result.failure_reason = parallel_output.failure_reason;
      return result;
    }
    for (std::size_t frame_index = 0; frame_index < frame_sources.size(); ++frame_index) {
      const FrozenRound2BaselineFrameSource& frame_source =
          frame_sources[frame_index];
      AppendWarnings(parallel_output.frame_warnings[frame_index], &result.warnings);
      if (parallel_output.cache_hits[frame_index] != 0) {
        ++result.runtime_breakdown.training_detection_cache.cache_hits;
        if (parallel_output.cache_sources[frame_index] ==
            OuterDetectionCacheLoadSource::StageLayout) {
          ++result.runtime_breakdown.training_detection_cache.stage_layout_cache_hits;
        } else if (parallel_output.cache_sources[frame_index] ==
                   OuterDetectionCacheLoadSource::LegacyLayout) {
          ++result.runtime_breakdown.training_detection_cache.legacy_layout_cache_hits;
        }
      } else {
        ++result.runtime_breakdown.training_detection_cache.cache_misses;
        if (parallel_output.cache_load_failures[frame_index] != 0) {
          ++result.runtime_breakdown.training_detection_cache.load_failures;
        }
        if (parallel_output.cache_store_failures[frame_index] != 0) {
          ++result.runtime_breakdown.training_detection_cache.store_failures;
        }
      }
      const OuterTagMultiDetectionResult& outer_detections =
          parallel_output.detections[frame_index];
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
  initialization_options.reference_board_id = options_.reference_board_id;
  initialization_options.refine_mode = options_.camera_initialization_refine_mode;
  initialization_options.selection_scorer =
      options_.camera_initialization_selection_scorer;
  initialization_options.shared_focal_during_outer_lm =
      options_.camera_initialization_shared_focal;
  initialization_options.fixed_board_layout = options_.fixed_board_layout;
  initialization_options.fixed_board_layout_source =
      options_.fixed_board_layout_source;
  initialization_options.enable_shared_frame_board_constraint =
      !options_.camera_initialization_use_independent_frame_board_poses;
  initialization_options.enable_principal_profile =
      options_.enable_camera_initialization_principal_profile;
  initialization_options.principal_profile_radius_px =
      options_.camera_initialization_principal_profile_radius_px;
  initialization_options.enable_fixed_layout_diagnostic =
      options_.enable_camera_initialization_fixed_layout_diagnostic;
  initialization_options.enable_board_jackknife_diagnostic =
      options_.enable_camera_initialization_board_jackknife_diagnostic;
  initialization_options.enable_coverage_weighted_diagnostic =
      options_.enable_camera_initialization_coverage_weighted_diagnostic;
  initialization_options.prefer_lower_focal_in_near_tie =
      options_.camera_initialization_prefer_lower_focal_in_near_tie;
  initialization_options.near_tie_relative_objective_tolerance =
      options_.camera_initialization_near_tie_relative_objective_tolerance;
  initialization_options.use_explicit_initial_camera =
      options_.use_explicit_initial_camera;
  initialization_options.explicit_initial_camera =
      options_.explicit_initial_camera;
  initialization_options.explicit_initial_camera_source_label =
      options_.explicit_initial_camera_source_label;
  initialization_options.dense_grid_lm_huber_delta_pixels =
      options_.checkerboard_initialization_huber_delta_pixels;
  initialization_options.use_direct_dense_control_points =
      options_.precomputed_initialization_use_all_points &&
      !options_.camera_initialization_auxiliary_bootstrap_frames.empty();
  initialization_options.direct_dense_control_point_scope =
      options_.precomputed_initialization_point_scope;
  std::vector<OuterBootstrapFrameInput> camera_initialization_frames =
      bootstrap_frames;
  camera_initialization_frames.insert(
      camera_initialization_frames.end(),
      options_.camera_initialization_auxiliary_bootstrap_frames.begin(),
      options_.camera_initialization_auxiliary_bootstrap_frames.end());
  const OuterOnlyCameraInitializer camera_initializer(config, initialization_options);
  {
    const auto stage_start = std::chrono::steady_clock::now();
    progress.StageStart("camera_initialization", 1);
    result.auto_camera_initialization =
        camera_initializer.Initialize(camera_initialization_frames);
    result.runtime_breakdown.auto_camera_initialization_seconds =
        ElapsedSeconds(stage_start);
    progress.StageEnd("camera_initialization", 1,
                      result.auto_camera_initialization.success ? "finished"
                                                                : "failed");
  }
  if (!result.auto_camera_initialization.success) {
    result.failure_reason = result.auto_camera_initialization.failure_reason.empty()
                                ? "Automatic camera initialization failed."
                                : result.auto_camera_initialization.failure_reason;
    AppendWarnings(result.auto_camera_initialization.warnings, &result.warnings);
    return result;
  }

  result.camera_aware_outer_rescue.requested =
      options_.enable_camera_aware_outer_rescue;
    if (options_.enable_camera_aware_outer_rescue) {
    const OuterBootstrapCameraIntrinsics provisional_camera =
        result.auto_camera_initialization.selected_camera;
    if (provisional_camera.NormalizedFamilyString() != "ds-none") {
      result.camera_aware_outer_rescue.skip_reason =
          "camera_family_not_supported_by_ds_sphere_patch_rescue";
      result.camera_aware_outer_rescue.provisional_camera =
          provisional_camera;
    } else if (!provisional_camera.IsValid()) {
      result.camera_aware_outer_rescue.skip_reason =
          "provisional_camera_is_invalid";
    } else {
      MultiScaleOuterTagDetectorConfig rescue_cache_config =
          config.outer_detector_config;
      rescue_cache_config.enable_camera_aware_sphere_patch_rescue = true;
      rescue_cache_config.camera_aware_sphere_patch_max_hamming =
          options_.camera_aware_outer_rescue_max_hamming;
      rescue_cache_config.camera_aware_sphere_patch_commit_mapped_corners =
          true;
      OuterDetectionCache rescue_cache(
          config.outer_detector_config,
          OuterDetectionCacheOptions{
              options_.enable_outer_detection_cache,
              options_.outer_detection_cache_dir,
              Stage5CacheStage::OuterRescue,
              MakeCameraAwareRescueSignature(
              provisional_camera, rescue_cache_config,
                  options_.camera_aware_outer_rescue_max_hamming,
                  options_.reference_board_id)});
      const auto rescue_start = std::chrono::steady_clock::now();
      progress.StageStart("camera_aware_outer_rescue", frame_sources.size());
      RunCameraAwareOuterRescue(
          frame_sources, config, provisional_camera, options_.reference_board_id,
          options_.camera_aware_outer_rescue_max_hamming,
          options_.camera_aware_outer_rescue_worker_count,
          &bootstrap_frames, &regeneration_inputs,
          &result.camera_aware_outer_rescue,
          &rescue_cache,
          [&progress](std::size_t completed, std::size_t total,
                      const std::string& detail) {
            (void)completed;
            progress.FrameDone("camera_aware_outer_rescue", total, detail);
          });
      result.runtime_breakdown.camera_aware_outer_rescue_seconds =
          ElapsedSeconds(rescue_start);
      progress.StageEnd("camera_aware_outer_rescue", frame_sources.size(),
                        "finished");
      result.camera_aware_outer_rescue.runtime_seconds =
          result.runtime_breakdown.camera_aware_outer_rescue_seconds;

      if (result.camera_aware_outer_rescue.rescued_board_observation_count > 0 &&
          options_.rerun_camera_initialization_after_outer_rescue) {
        result.camera_aware_outer_rescue.camera_initialization_rerun = true;
        camera_initialization_frames = bootstrap_frames;
        camera_initialization_frames.insert(
            camera_initialization_frames.end(),
            options_.camera_initialization_auxiliary_bootstrap_frames.begin(),
            options_.camera_initialization_auxiliary_bootstrap_frames.end());
        const auto reinitialization_start = std::chrono::steady_clock::now();
        const AutoCameraInitializationResult reinitialized =
            camera_initializer.Initialize(camera_initialization_frames);
        result.runtime_breakdown.auto_camera_initialization_seconds +=
            ElapsedSeconds(reinitialization_start);
        result.camera_aware_outer_rescue.camera_initialization_rerun_success =
            reinitialized.success;
        if (reinitialized.success) {
          result.auto_camera_initialization = reinitialized;
        } else {
          AppendUniqueWarning(
              "Camera-aware outer rescue succeeded, but camera reinitialization "
              "failed; retaining the provisional outer-only camera. Reason: " +
                  reinitialized.failure_reason,
              &result.camera_aware_outer_rescue.warnings);
        }
      }
    }
  }
  result.camera_aware_outer_rescue.final_initialization_camera =
      result.auto_camera_initialization.selected_camera;
  AppendWarnings(result.camera_aware_outer_rescue.warnings, &result.warnings);
  result.auto_camera_initialization.stage5_init_primary_frame_count =
      static_cast<int>(bootstrap_frames.size());
  result.auto_camera_initialization.stage5_init_auxiliary_session_count =
      options_.camera_initialization_auxiliary_session_count;
  result.auto_camera_initialization.stage5_init_auxiliary_frame_count =
      static_cast<int>(
          options_.camera_initialization_auxiliary_bootstrap_frames.size());
  result.auto_camera_initialization.stage5_init_uses_auxiliary_sessions =
      options_.camera_initialization_auxiliary_bootstrap_frames.empty() ? 0 : 1;
  if (result.auto_camera_initialization.stage5_init_uses_auxiliary_sessions != 0) {
    AppendUniqueWarning(
        "Auxiliary calibration sessions were used only by camera initialization with independent frame-board poses; auxiliary frames did not enter the primary layout, measurement selection, or backend problem.",
        &result.auto_camera_initialization.warnings);
  }
  AppendWarnings(result.auto_camera_initialization.warnings, &result.warnings);
  SetBootstrapInitFromIntrinsics(result.auto_camera_initialization.selected_camera,
                                 &bootstrap_options);

  {
    const auto stage_start = std::chrono::steady_clock::now();
    progress.StageStart("outer_bootstrap", 1);
    const MultiBoardOuterBootstrap bootstrap(config, bootstrap_options);
    result.bootstrap_result = bootstrap.Solve(bootstrap_frames);
    result.runtime_breakdown.outer_bootstrap_seconds = ElapsedSeconds(stage_start);
    progress.StageEnd("outer_bootstrap", 1,
                      result.bootstrap_result.success ? "finished" : "failed");
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
    const std::string regeneration_state_signature =
        round1_regeneration_state != nullptr
            ? InternalRegenerationCache::MakeSceneStateSignature(
                  *round1_regeneration_state)
            : InternalRegenerationCache::MakeBootstrapStateSignature(
                  result.bootstrap_result);
    const JointReprojectionSceneState* scene_state = round1_regeneration_state;
    const OuterBootstrapResult* bootstrap_result = &result.bootstrap_result;
    const ParallelRegenerationOutput parallel_output = RegenerateFramesInParallel(
        frame_sources, regeneration_inputs, config, detection_options,
        internal_regeneration_cache, regeneration_state_signature,
        &progress,
        [scene_state, bootstrap_result](
            BoardDetectionPipeline& worker_pipeline, const cv::Mat& image,
            const InternalRegenerationFrameInput& frame_input) {
          InternalRegenerationFrameResult regeneration_result;
          if (scene_state != nullptr) {
            regeneration_result = worker_pipeline.RegenerateFrame(
                image, frame_input, *scene_state);
            regeneration_result.state_source_label = "outer_only_intermediate";
          } else {
            regeneration_result = worker_pipeline.RegenerateFrame(
                image, frame_input, *bootstrap_result);
          }
          return regeneration_result;
        });
    if (!parallel_output.failure_reason.empty()) {
      result.failure_reason = parallel_output.failure_reason;
      return result;
    }
    for (std::size_t frame_index = 0; frame_index < frame_sources.size(); ++frame_index) {
      AppendWarnings(parallel_output.frame_warnings[frame_index], &result.warnings);
      InternalRegenerationFrameResult regeneration_result =
          parallel_output.regeneration_results[frame_index];
      result.runtime_breakdown.training_internal_regeneration_cache =
          internal_regeneration_cache.stats();
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
      AppendWarnings(regeneration_result.warnings, &result.warnings);
      result.round1.regeneration_results.push_back(std::move(regeneration_result));

      JointMeasurementFrameInput joint_input;
      joint_input.frame_index = regeneration_inputs[frame_index].frame_index;
      joint_input.frame_label = regeneration_inputs[frame_index].frame_label;
      joint_input.outer_detections = regeneration_inputs[frame_index].outer_detections;
      joint_input.regenerated_internal = result.round1.regeneration_results.back();
      result.round1.joint_inputs.push_back(std::move(joint_input));
    }
    result.runtime_breakdown.round1_regeneration_seconds =
        ElapsedSeconds(stage_start);
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    progress.StageStart("round1_measurement_build", 1);
    result.round1.measurement_result =
        builder.Build(result.round1.joint_inputs, result.bootstrap_result);
    result.runtime_breakdown.round1_measurement_build_seconds =
        ElapsedSeconds(stage_start);
    progress.StageEnd("round1_measurement_build", 1,
                      result.round1.measurement_result.success ? "finished" : "failed");
  }
  result.round1.validation_summary = ValidateJointMeasurementBuilder(
      result.round1.joint_inputs, result.bootstrap_result, builder,
      result.round1.measurement_result);
  if (!result.round1.validation_summary.success) {
    result.failure_reason = result.round1.validation_summary.failure_reason;
    AppendWarnings(result.round1.validation_summary.warnings, &result.warnings);
    return result;
  }

  if (options_.frontend_only) {
    result.success = true;
    AppendUniqueWarning(
        "Stage5 frontend-only mode completed after detection, initialization, "
        "rescue, regeneration, and measurement construction; measurement "
        "selection and all backend optimization were intentionally skipped.",
        &result.warnings);
    return result;
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    progress.StageStart("round1_residual_evaluation", 1);
    result.round1.residual_result =
        residual_evaluator.Evaluate(result.round1.measurement_result, initial_scene_state);
    result.runtime_breakdown.round1_residual_evaluation_seconds =
        ElapsedSeconds(stage_start);
    progress.StageEnd("round1_residual_evaluation", 1,
                      result.round1.residual_result.success ? "finished" : "failed");
  }
  if (!result.round1.residual_result.success) {
    result.failure_reason = result.round1.residual_result.failure_reason;
    AppendWarnings(result.round1.residual_result.warnings, &result.warnings);
    return result;
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    progress.StageStart("round1_selection", 1);
    result.round1.selection_result =
        selector.Select(result.round1.measurement_result, result.round1.residual_result,
                        initial_scene_state);
    result.runtime_breakdown.round1_selection_seconds =
        ElapsedSeconds(stage_start);
    progress.StageEnd("round1_selection", 1,
                      result.round1.selection_result.success ? "finished" : "failed");
  }
  if (!result.round1.selection_result.success) {
    result.failure_reason = result.round1.selection_result.failure_reason;
    AppendWarnings(result.round1.selection_result.warnings, &result.warnings);
    return result;
  }

  {
    const auto stage_start = std::chrono::steady_clock::now();
    progress.StageStart("round1_optimization", 1);
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
    progress.StageEnd("round1_optimization", 1,
                      result.round1.optimization_result.success ? "finished" : "failed");
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
      const std::string regeneration_state_signature =
          InternalRegenerationCache::MakeSceneStateSignature(
              result.round1.optimization_result.optimized_state);
      const JointReprojectionSceneState* scene_state =
          &result.round1.optimization_result.optimized_state;
    const ParallelRegenerationOutput parallel_output = RegenerateFramesInParallel(
        frame_sources, regeneration_inputs, config, detection_options,
        internal_regeneration_cache, regeneration_state_signature,
        &progress,
        [scene_state](BoardDetectionPipeline& worker_pipeline,
                        const cv::Mat& image,
                        const InternalRegenerationFrameInput& frame_input) {
            return worker_pipeline.RegenerateFrame(image, frame_input, *scene_state);
          });
      if (!parallel_output.failure_reason.empty()) {
        result.failure_reason = parallel_output.failure_reason;
        return result;
      }
      for (std::size_t frame_index = 0; frame_index < frame_sources.size(); ++frame_index) {
        AppendWarnings(parallel_output.frame_warnings[frame_index], &result.warnings);
        InternalRegenerationFrameResult regeneration_result =
            parallel_output.regeneration_results[frame_index];
        result.runtime_breakdown.training_internal_regeneration_cache =
            internal_regeneration_cache.stats();
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
        AppendWarnings(regeneration_result.warnings, &result.warnings);
        result.round2.regeneration_results.push_back(std::move(regeneration_result));

        JointMeasurementFrameInput joint_input;
        joint_input.frame_index = regeneration_inputs[frame_index].frame_index;
        joint_input.frame_label = regeneration_inputs[frame_index].frame_label;
        joint_input.outer_detections = regeneration_inputs[frame_index].outer_detections;
        joint_input.regenerated_internal = result.round2.regeneration_results.back();
        result.round2.joint_inputs.push_back(std::move(joint_input));
      }
      result.runtime_breakdown.round2_regeneration_seconds =
          ElapsedSeconds(stage_start);
    }

    {
      const auto stage_start = std::chrono::steady_clock::now();
      progress.StageStart("round2_measurement_build", 1);
      result.round2.measurement_result =
          builder.Build(result.round2.joint_inputs, result.bootstrap_result);
      result.runtime_breakdown.round2_measurement_build_seconds =
          ElapsedSeconds(stage_start);
      progress.StageEnd("round2_measurement_build", 1,
                        result.round2.measurement_result.success ? "finished" : "failed");
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
      progress.StageStart("round2_residual_evaluation", 1);
      result.round2.residual_result = residual_evaluator.Evaluate(
          result.round2.measurement_result,
          result.round1.optimization_result.optimized_state);
      result.runtime_breakdown.round2_residual_evaluation_seconds =
          ElapsedSeconds(stage_start);
      progress.StageEnd("round2_residual_evaluation", 1,
                        result.round2.residual_result.success ? "finished" : "failed");
    }
    if (!result.round2.residual_result.success) {
      result.failure_reason = result.round2.residual_result.failure_reason;
      AppendWarnings(result.round2.residual_result.warnings, &result.warnings);
      return result;
    }

    {
      const auto stage_start = std::chrono::steady_clock::now();
      progress.StageStart("round2_selection", 1);
      result.round2.selection_result = selector.Select(
          result.round2.measurement_result, result.round2.residual_result,
          result.round1.optimization_result.optimized_state);
      if (options_.allow_non_reference_board_frames_after_layout &&
          result.round2.selection_result.success) {
        int supplement_candidate_frame_count = 0;
        const JointMeasurementBuildResult supplement_measurements =
            BuildNonReferenceSupplementMeasurements(
                result.round2.measurement_result, options_.reference_board_id,
                &supplement_candidate_frame_count);
        const JointMeasurementSelectionResult supplement_selection =
            supplemental_selector.Select(
                supplement_measurements, result.round2.residual_result,
                result.round1.optimization_result.optimized_state);
        if (supplement_selection.success) {
          result.round2.selection_result = MergeSupplementalSelection(
              result.round2.selection_result, supplement_selection,
              supplement_candidate_frame_count);
        } else {
          AppendUniqueWarning(
              "Round 2 supplemental non-reference selection found no "
              "admissible reference-free frame; reference-only selection "
              "was retained.",
              &result.round2.selection_result.warnings);
        }
      }
      result.runtime_breakdown.round2_selection_seconds =
          ElapsedSeconds(stage_start);
      progress.StageEnd("round2_selection", 1,
                        result.round2.selection_result.success ? "finished" : "failed");
    }
    if (!result.round2.selection_result.success) {
      result.failure_reason = result.round2.selection_result.failure_reason;
      AppendWarnings(result.round2.selection_result.warnings, &result.warnings);
      return result;
    }

    JointOptimizationOptions second_pass_options = optimization_options;
    second_pass_options.intrinsics_release_iteration =
        options_.second_pass_intrinsics_release_iteration;
    second_pass_options.optimize_board_poses =
        options_.fixed_board_layout.empty() &&
        !options_.allow_non_reference_board_frames_after_layout;
    const JointReprojectionOptimizer round2_optimizer(second_pass_options);
    {
      const auto stage_start = std::chrono::steady_clock::now();
      progress.StageStart("round2_optimization", 1);
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
      progress.StageEnd("round2_optimization", 1,
                        result.round2.optimization_result.success ? "finished" : "failed");
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
    if (options_.allow_non_reference_board_frames_after_layout) {
      AppendUniqueWarning(
          "Round 2 admitted initialized non-reference-only frames after Round 1 layout refinement; shared layout updates are fixed for this supplemental pass.",
          &result.warnings);
    }
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

FrozenRound2BaselineResult FrozenRound2BaselinePipeline::RunPrecomputed(
    const FrozenPrecomputedMeasurementInput& input) const {
  FrozenRound2BaselineResult result;
  result.baseline_protocol_label = options_.baseline_protocol_label;
  result.dataset_label = options_.dataset_label;
  result.training_split_signature = options_.training_split_signature;
  result.reference_board_id = options_.reference_board_id;
  result.frame_sources = input.frame_sources;
  result.effective_options = options_;
  result.effective_options.run_second_pass = false;
  result.effective_options.enable_outer_only_intermediate_calibration = false;
  AppendWarnings(input.warnings, &result.warnings);

  if (!input.success) {
    result.failure_reason = input.failure_reason.empty()
                                ? "Precomputed measurement import failed."
                                : input.failure_reason;
    return result;
  }
  if (input.bootstrap_frames.empty() || input.frame_sources.empty() ||
      !input.measurement_result.success) {
    result.failure_reason =
        "RunPrecomputed requires non-empty bootstrap frames and measurements.";
    return result;
  }
  if (input.reference_board_id != options_.reference_board_id) {
    result.failure_reason =
        "Precomputed reference_board_id does not match Stage5 options.";
    return result;
  }
  if (input.image_size.width <= 0 || input.image_size.height <= 0) {
    result.failure_reason = "Precomputed observation image size is invalid.";
    return result;
  }

  const ApriltagInternalConfig config = NormalizeConfig(options_.config);
  if (std::find(config.tag_ids.begin(), config.tag_ids.end(),
                options_.reference_board_id) == config.tag_ids.end()) {
    result.failure_reason =
        "Configured reference_board_id is absent from the explicit tag_ids topology.";
    return result;
  }
  OuterBootstrapOptions bootstrap_options = MakeBootstrapOptions(config, options_);

  AutoCameraInitializationOptions initialization_options;
  initialization_options.mode = config.camera_initialization_mode;
  initialization_options.reference_board_id = options_.reference_board_id;
  initialization_options.refine_mode = options_.camera_initialization_refine_mode;
  initialization_options.selection_scorer =
      options_.camera_initialization_selection_scorer;
  initialization_options.shared_focal_during_outer_lm =
      options_.camera_initialization_shared_focal;
  initialization_options.fixed_board_layout = options_.fixed_board_layout;
  initialization_options.fixed_board_layout_source =
      options_.fixed_board_layout_source;
  initialization_options.enable_shared_frame_board_constraint =
      !options_.camera_initialization_use_independent_frame_board_poses;
  initialization_options.enable_principal_profile =
      options_.enable_camera_initialization_principal_profile;
  initialization_options.principal_profile_radius_px =
      options_.camera_initialization_principal_profile_radius_px;
  initialization_options.enable_fixed_layout_diagnostic =
      options_.enable_camera_initialization_fixed_layout_diagnostic;
  initialization_options.enable_board_jackknife_diagnostic =
      options_.enable_camera_initialization_board_jackknife_diagnostic;
  initialization_options.enable_coverage_weighted_diagnostic =
      options_.enable_camera_initialization_coverage_weighted_diagnostic;
  initialization_options.prefer_lower_focal_in_near_tie =
      options_.camera_initialization_prefer_lower_focal_in_near_tie;
  initialization_options.near_tie_relative_objective_tolerance =
      options_.camera_initialization_near_tie_relative_objective_tolerance;
  initialization_options.use_explicit_initial_camera =
      options_.use_explicit_initial_camera;
  initialization_options.explicit_initial_camera =
      options_.explicit_initial_camera;
  initialization_options.explicit_initial_camera_source_label =
      options_.explicit_initial_camera_source_label;
  initialization_options.dense_grid_lm_huber_delta_pixels =
      options_.checkerboard_initialization_huber_delta_pixels;
  initialization_options.use_direct_dense_control_points =
      input.single_board_mode ||
      options_.precomputed_initialization_use_all_points;
  // A single-board MAT target defines its own topology. The AprilTag YAML may
  // still list every board used by the image frontend, so applying that list
  // here would reject every checkerboard view before dense-grid initialization.
  initialization_options.require_all_configured_boards_per_frame =
      !input.single_board_mode;
  initialization_options.direct_dense_control_point_scope =
      options_.precomputed_initialization_point_scope;
  std::vector<OuterBootstrapFrameInput> camera_initialization_frames =
      input.bootstrap_frames;
  camera_initialization_frames.insert(
      camera_initialization_frames.end(),
      options_.camera_initialization_auxiliary_bootstrap_frames.begin(),
      options_.camera_initialization_auxiliary_bootstrap_frames.end());
  const OuterOnlyCameraInitializer camera_initializer(config,
                                                        initialization_options);
  {
    const auto stage_start = std::chrono::steady_clock::now();
    result.auto_camera_initialization =
        camera_initializer.Initialize(camera_initialization_frames);
    result.runtime_breakdown.auto_camera_initialization_seconds =
        ElapsedSeconds(stage_start);
  }
  result.auto_camera_initialization.stage5_init_primary_frame_count =
      static_cast<int>(input.bootstrap_frames.size());
  result.auto_camera_initialization.stage5_init_auxiliary_session_count =
      options_.camera_initialization_auxiliary_session_count;
  result.auto_camera_initialization.stage5_init_auxiliary_frame_count =
      static_cast<int>(
          options_.camera_initialization_auxiliary_bootstrap_frames.size());
  result.auto_camera_initialization.stage5_init_uses_auxiliary_sessions =
      options_.camera_initialization_auxiliary_bootstrap_frames.empty() ? 0 : 1;
  if (result.auto_camera_initialization.stage5_init_uses_auxiliary_sessions != 0) {
    AppendUniqueWarning(
        "Auxiliary calibration sessions were used only by camera initialization with independent frame-board poses; auxiliary frames did not enter the primary layout, measurement selection, or backend problem.",
        &result.auto_camera_initialization.warnings);
  }
  AppendWarnings(result.auto_camera_initialization.warnings, &result.warnings);
  if (!result.auto_camera_initialization.success) {
    result.failure_reason =
        result.auto_camera_initialization.failure_reason.empty()
            ? "Automatic camera initialization failed for precomputed observations."
            : result.auto_camera_initialization.failure_reason;
    return result;
  }
  SetBootstrapInitFromIntrinsics(
      result.auto_camera_initialization.selected_camera, &bootstrap_options);

  {
    const auto stage_start = std::chrono::steady_clock::now();
    const MultiBoardOuterBootstrap bootstrap(config, bootstrap_options);
    result.bootstrap_result = bootstrap.Solve(input.bootstrap_frames);
    result.runtime_breakdown.outer_bootstrap_seconds =
        ElapsedSeconds(stage_start);
  }
  if (input.single_board_mode) {
    std::string dense_bootstrap_failure;
    if (!BuildSingleBoardDenseBootstrap(
            input.measurement_result,
            result.auto_camera_initialization.selected_camera,
            options_.reference_board_id,
            &result.bootstrap_result,
            &dense_bootstrap_failure)) {
      result.failure_reason = dense_bootstrap_failure;
      return result;
    }
  }
  AppendWarnings(result.bootstrap_result.warnings, &result.warnings);
  if (!result.bootstrap_result.success) {
    result.failure_reason = result.bootstrap_result.failure_reason.empty()
                                ? "Outer bootstrap failed for precomputed observations."
                                : result.bootstrap_result.failure_reason;
    return result;
  }

  JointMeasurementBuildResult imported_measurements = input.measurement_result;
  if (!input.single_board_mode &&
      (!options_.include_internal_points || options_.outer_only_ablation_mode)) {
    for (JointMeasurementFrameResult& frame : imported_measurements.frames) {
      for (JointBoardObservation& board : frame.board_observations) {
        for (JointPointObservation& point : board.points) {
          if (point.point_type == JointPointType::Internal) {
            point.used_in_solver = false;
            point.rejection_reason_code =
                JointRejectionReasonCode::InternalRegenerationFailed;
            point.rejection_detail =
                "internal point disabled by Stage5 outer-only configuration";
          }
        }
      }
    }
    RecomputeMeasurementCounts(&imported_measurements);
  }
  result.round1.measurement_result =
      SynchronizePrecomputedMeasurementsWithBootstrap(
          imported_measurements, result.bootstrap_result);
  result.round1.validation_summary =
      ValidatePrecomputedMeasurements(result.round1.measurement_result);
  if (!result.round1.validation_summary.success) {
    result.failure_reason = result.round1.validation_summary.failure_reason;
    AppendWarnings(result.round1.validation_summary.warnings, &result.warnings);
    return result;
  }

  const JointReprojectionSceneState initial_scene_state =
      BuildSceneStateFromBootstrap(result.bootstrap_result);
  const JointReprojectionResidualEvaluator residual_evaluator(
      JointResidualEvaluationOptions{});
  {
    const auto stage_start = std::chrono::steady_clock::now();
    result.round1.residual_result = residual_evaluator.Evaluate(
        result.round1.measurement_result, initial_scene_state);
    result.runtime_breakdown.round1_residual_evaluation_seconds =
        ElapsedSeconds(stage_start);
  }
  if (!result.round1.residual_result.success) {
    result.failure_reason = result.round1.residual_result.failure_reason;
    AppendWarnings(result.round1.residual_result.warnings, &result.warnings);
    return result;
  }

  JointMeasurementSelectionOptions selection_options;
  selection_options.reference_board_id = options_.reference_board_id;
  selection_options.selection_mode = options_.selection_mode;
  selection_options.enable_residual_sanity_gate =
      options_.enable_residual_sanity_gate;
  selection_options.enable_board_pose_fit_gate =
      options_.enable_board_pose_fit_gate;
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
  {
    const auto stage_start = std::chrono::steady_clock::now();
    result.round1.selection_result = selector.Select(
        result.round1.measurement_result, result.round1.residual_result,
        initial_scene_state);
    result.runtime_breakdown.round1_selection_seconds =
        ElapsedSeconds(stage_start);
  }
  if (!result.round1.selection_result.success) {
    result.failure_reason = result.round1.selection_result.failure_reason;
    AppendWarnings(result.round1.selection_result.warnings, &result.warnings);
    return result;
  }

  JointOptimizationOptions optimization_options;
  optimization_options.reference_board_id = options_.reference_board_id;
  optimization_options.optimize_intrinsics = options_.optimize_intrinsics;
  optimization_options.optimize_board_poses =
      options_.fixed_board_layout.empty();
  optimization_options.intrinsics_release_iteration =
      options_.intrinsics_release_iteration;
  optimization_options.cost_options.uniform_control_point_mode =
      input.single_board_mode;
  const JointReprojectionOptimizer optimizer(optimization_options);
  {
    const auto stage_start = std::chrono::steady_clock::now();
    result.round1.optimization_result = optimizer.Optimize(
        result.round1.selection_result, initial_scene_state);
    result.runtime_breakdown.round1_optimization_seconds =
        ElapsedSeconds(stage_start);
    result.runtime_breakdown.round1_optimization_residual_evaluation_seconds =
        result.round1.optimization_result.runtime_breakdown
            .residual_evaluation_seconds;
    result.runtime_breakdown
        .round1_optimization_residual_evaluation_call_count =
        result.round1.optimization_result.runtime_breakdown
            .residual_evaluation_call_count;
    result.runtime_breakdown.round1_optimization_cost_evaluation_seconds =
        result.round1.optimization_result.runtime_breakdown.cost_evaluation_seconds;
    result.runtime_breakdown.round1_optimization_cost_evaluation_call_count =
        result.round1.optimization_result.runtime_breakdown
            .cost_evaluation_call_count;
    result.runtime_breakdown.round1_optimization_frame_update_seconds =
        result.round1.optimization_result.runtime_breakdown.frame_update_seconds;
    result.runtime_breakdown.round1_optimization_board_update_seconds =
        result.round1.optimization_result.runtime_breakdown.board_update_seconds;
    result.runtime_breakdown.round1_optimization_intrinsics_update_seconds =
        result.round1.optimization_result.runtime_breakdown
            .intrinsics_update_seconds;
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
  metadata.source_pipeline_label =
      options_.source_pipeline_label + "_precomputed_observations";
  result.stage5_round1_bundle =
      BuildCalibrationStateBundleFromJointOptimizationResult(
          result.round1.optimization_result, result.round1.selection_result,
          result.round1.measurement_result, 1, metadata);
  result.final_stage5_bundle = result.stage5_round1_bundle;
  result.stage5_bundle_available = result.final_stage5_bundle.success;
  result.round2_available = false;
  result.stage42_validation_pass = false;
  result.outer_only_intermediate.enabled = false;

  if (options_.allow_non_reference_board_frames_after_layout) {
    // Precomputed inputs have no image-driven Round 2 regeneration. Reuse the
    // frozen observations after the anchor-only Round 1 has refined layout.
    result.round2_available = true;
    result.round2.measurement_result = result.round1.measurement_result;
    result.round2.validation_summary = result.round1.validation_summary;
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
      // Preserve the regular reference-supported Round 2 seed.  The optional
      // selector below may only add frames that do not have such support.
      result.round2.selection_result = selector.Select(
          result.round2.measurement_result, result.round2.residual_result,
          result.round1.optimization_result.optimized_state);
      if (result.round2.selection_result.success) {
        int supplement_candidate_frame_count = 0;
        const JointMeasurementBuildResult supplement_measurements =
            BuildNonReferenceSupplementMeasurements(
                result.round2.measurement_result, options_.reference_board_id,
                &supplement_candidate_frame_count);
        JointMeasurementSelectionOptions supplemental_selection_options =
            selection_options;
        supplemental_selection_options.require_reference_board_per_frame = false;
        const JointMeasurementSelection supplemental_selector(
            supplemental_selection_options);
        const JointMeasurementSelectionResult supplement_selection =
            supplemental_selector.Select(
                supplement_measurements, result.round2.residual_result,
                result.round1.optimization_result.optimized_state);
        if (supplement_selection.success) {
          result.round2.selection_result = MergeSupplementalSelection(
              result.round2.selection_result, supplement_selection,
              supplement_candidate_frame_count);
        } else {
          AppendUniqueWarning(
              "Round 2 supplemental non-reference selection found no "
              "admissible reference-free frame; reference-only selection "
              "was retained.",
              &result.round2.selection_result.warnings);
        }
      }
      result.runtime_breakdown.round2_selection_seconds = ElapsedSeconds(stage_start);
    }
    if (!result.round2.selection_result.success) {
      result.failure_reason = result.round2.selection_result.failure_reason;
      AppendWarnings(result.round2.selection_result.warnings, &result.warnings);
      return result;
    }

    JointOptimizationOptions supplemental_optimization_options = optimization_options;
    supplemental_optimization_options.optimize_board_poses = false;
    const JointReprojectionOptimizer supplemental_optimizer(
        supplemental_optimization_options);
    {
      const auto stage_start = std::chrono::steady_clock::now();
      result.round2.optimization_result = supplemental_optimizer.Optimize(
          result.round2.selection_result,
          result.round1.optimization_result.optimized_state);
      result.runtime_breakdown.round2_optimization_seconds =
          ElapsedSeconds(stage_start);
    }
    if (!result.round2.optimization_result.success) {
      result.failure_reason = result.round2.optimization_result.failure_reason;
      AppendWarnings(result.round2.optimization_result.warnings, &result.warnings);
      return result;
    }
    result.final_stage5_bundle = BuildCalibrationStateBundleFromJointOptimizationResult(
        result.round2.optimization_result, result.round2.selection_result,
        result.round2.measurement_result, 2, metadata);
    result.stage5_bundle_available = result.final_stage5_bundle.success;
    AppendUniqueWarning(
        "Precomputed supplemental pass admitted initialized non-reference-only frames after anchor-only layout refinement; shared layout updates are fixed.",
        &result.warnings);
  }

  AppendUniqueWarning(
      "Stage5 consumed frozen precomputed observations: image detection, internal-point regeneration, and image-driven second pass were not run.",
      &result.warnings);
  AppendUniqueWarning(
      "Imported boards.Rt was not used; camera and multi-board scene were initialized from imported outer-corner observations.",
      &result.warnings);
  if (input.single_board_mode) {
    AppendUniqueWarning(
        "Precomputed checkerboard mode: every imported grid point is a uniform control point; Stage5 optimizes camera intrinsics and independent per-frame camera-to-board poses without outer/internal role weighting or board-layout variables.",
        &result.warnings);
  }
  AppendWarnings(result.round1.optimization_result.warnings, &result.warnings);
  if (!result.stage5_bundle_available) {
    result.failure_reason =
        "Precomputed Stage5 bundle was not ready for backend.";
    return result;
  }
  result.success = true;
  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
