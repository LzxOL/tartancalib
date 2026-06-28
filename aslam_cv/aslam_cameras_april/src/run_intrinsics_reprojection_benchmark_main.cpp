#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>
#include <aslam/cameras/apriltag_internal/KalibrBenchmark.hpp>
#include <aslam/cameras/apriltag_internal/Stage5Benchmark.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

namespace ati = aslam::cameras::apriltag_internal;
namespace fs = boost::filesystem;

struct CmdArgs {
  std::string image_path;
  std::string config_path;
  std::vector<std::string> intrinsics_specs;
  std::string active_intrinsics_path;
  std::string detector_intrinsics_yaml;
  std::string output_path;
  std::string benchmark_label = "benchmark";
  std::string frontend_mode = "detector";
  std::string frontend_cache_dir;
  bool all = false;
  bool export_overlays = false;
  int overlay_top_k = 20;
  int max_images = 0;
  int progress_every = 1;
  bool enable_geometry_prior_rescue = true;
  bool geometry_prior_rescue_use_as_observation = true;
  bool geometry_prior_rescue_allow_geometry_only_pose_refit = true;
  bool geometry_prior_rescue_enable_spherical_refine = true;
};

struct PolarBucketAccumulator {
  int point_count = 0;
  double pixel_squared_sum = 0.0;
  double angular_squared_sum_deg = 0.0;
  double max_pixel_error = 0.0;
  double max_angular_error_deg = 0.0;
};

struct FrontendRuntimeFrameStats {
  int frame_index = -1;
  std::string frame_label;
  double detect_seconds = 0.0;
  int detection_count = 0;
  int successful_detection_count = 0;
  int outer_debug_corner_count = 0;
  int close_edge_boost_corner_count = 0;
  int line_refinement_failed_count = 0;
  int line_seed_accepted_count = 0;
  int image_line_fallback_count = 0;
  int image_line_fallback_accepted_count = 0;
};

struct FrontendRuntimeStats {
  double total_detect_seconds = 0.0;
  int processed_image_count = 0;
  int detection_count = 0;
  int successful_detection_count = 0;
  int outer_debug_corner_count = 0;
  int close_edge_boost_corner_count = 0;
  int line_refinement_failed_count = 0;
  int line_seed_accepted_count = 0;
  int image_line_fallback_count = 0;
  int image_line_fallback_accepted_count = 0;
  std::vector<FrontendRuntimeFrameStats> frames;
};

struct ModelEvaluationRecord {
  std::string label;
  std::string intrinsics_yaml;
  ati::OuterBootstrapCameraIntrinsics intrinsics;
  ati::CameraModelRefitEvaluationResult evaluation;
};

struct IntrinsicsSpec {
  std::string label;
  std::string path;
};

struct RayCurveBucketAccumulator {
  int count = 0;
  double angular_sum = 0.0;
  double angular_square_sum = 0.0;
  double max_angular = 0.0;
  double baseline_polar_sum = 0.0;
  double reference_polar_sum = 0.0;
};

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return value;
}

bool ParseBool(const std::string& value) {
  const std::string lowered = ToLower(value);
  if (lowered == "1" || lowered == "true" || lowered == "yes" ||
      lowered == "on") {
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" ||
      lowered == "off") {
    return false;
  }
  throw std::runtime_error("Failed to parse bool value: " + value);
}

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --image IMAGE_OR_DIR --config APRILTAG_INTERNAL_YAML"
      << " --intrinsics-yaml CAMCHAIN_OR_CAMERA_YAML --output OUTPUT_DIR"
      << " [--detector-intrinsics-yaml BASELINE_CAMCHAIN_YAML]"
      << " [--benchmark-label LABEL] [--frontend-mode detector|frozen_round2]"
      << " [--frontend-cache-dir DIR] [--all]\n"
      << "  Repeat --intrinsics-yaml to evaluate multiple camera models with"
      << " one frontend detection pass.\n"
      << "Options:"
      << " [--max-images N] [--progress-every N]"
      << " [--export-overlays] [--overlay-top-k N]"
      << " [--enable-geometry-prior-rescue 0|1]"
      << " [--geometry-prior-rescue-use-as-observation 0|1]"
      << " [--geometry-prior-rescue-allow-geometry-only-pose-refit 0|1]"
      << " [--geometry-prior-rescue-enable-spherical-refine 0|1]\n";
}

CmdArgs ParseArgs(int argc, char** argv) {
  CmdArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string token = argv[i];
    if (token == "--image" && i + 1 < argc) {
      args.image_path = argv[++i];
    } else if (token == "--config" && i + 1 < argc) {
      args.config_path = argv[++i];
    } else if ((token == "--intrinsics-yaml" ||
                token == "--benchmark-intrinsics-yaml" ||
                token == "--camchain") &&
               i + 1 < argc) {
      args.intrinsics_specs.push_back(argv[++i]);
    } else if (token == "--detector-intrinsics-yaml" && i + 1 < argc) {
      args.detector_intrinsics_yaml = argv[++i];
    } else if (token == "--output" && i + 1 < argc) {
      args.output_path = argv[++i];
    } else if (token == "--benchmark-label" && i + 1 < argc) {
      args.benchmark_label = argv[++i];
    } else if (token == "--frontend-mode" && i + 1 < argc) {
      args.frontend_mode = ToLower(argv[++i]);
    } else if (token == "--frontend-cache-dir" && i + 1 < argc) {
      args.frontend_cache_dir = argv[++i];
    } else if (token == "--all") {
      args.all = true;
    } else if (token == "--export-overlays") {
      args.export_overlays = true;
    } else if (token == "--overlay-top-k" && i + 1 < argc) {
      args.overlay_top_k = std::stoi(argv[++i]);
    } else if (token == "--max-images" && i + 1 < argc) {
      args.max_images = std::stoi(argv[++i]);
    } else if (token == "--progress-every" && i + 1 < argc) {
      args.progress_every = std::max(1, std::stoi(argv[++i]));
    } else if (token == "--enable-geometry-prior-rescue" && i + 1 < argc) {
      args.enable_geometry_prior_rescue = ParseBool(argv[++i]);
    } else if (token == "--geometry-prior-rescue-use-as-observation" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_use_as_observation = ParseBool(argv[++i]);
    } else if (token ==
                   "--geometry-prior-rescue-allow-geometry-only-pose-refit" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_allow_geometry_only_pose_refit =
          ParseBool(argv[++i]);
    } else if (token == "--geometry-prior-rescue-enable-spherical-refine" &&
               i + 1 < argc) {
      args.geometry_prior_rescue_enable_spherical_refine = ParseBool(argv[++i]);
    } else if (token == "--help" || token == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + token);
    }
  }
  if (args.image_path.empty() || args.config_path.empty() ||
      args.intrinsics_specs.empty() || args.output_path.empty()) {
    throw std::runtime_error(
        "--image, --config, --intrinsics-yaml and --output are required.");
  }
  if (args.detector_intrinsics_yaml.empty()) {
    args.detector_intrinsics_yaml = args.intrinsics_specs.front();
    const std::size_t colon = args.detector_intrinsics_yaml.find(':');
    if (colon != std::string::npos &&
        colon + 1u < args.detector_intrinsics_yaml.size()) {
      args.detector_intrinsics_yaml =
          args.detector_intrinsics_yaml.substr(colon + 1u);
    }
  }
  return args;
}

std::string SanitizePathComponent(std::string value) {
  if (value.empty()) {
    return "benchmark";
  }
  for (char& ch : value) {
    const bool valid =
        std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_';
    if (!valid) {
      ch = '_';
    }
  }
  return value;
}

std::string Trim(std::string value) {
  const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
              value.end());
  return value;
}

IntrinsicsSpec ParseIntrinsicsSpec(const std::string& spec) {
  IntrinsicsSpec parsed;
  const std::size_t colon = spec.find(':');
  const bool has_label =
      colon != std::string::npos && colon > 0 && colon + 1u < spec.size();
  if (has_label) {
    parsed.label = spec.substr(0, colon);
    parsed.path = spec.substr(colon + 1u);
  } else {
    parsed.path = spec;
    parsed.label = fs::path(spec).stem().string();
  }
  parsed.label = SanitizePathComponent(parsed.label);
  return parsed;
}

bool TryReadKeyValueDouble(const std::string& line,
                           const std::string& key,
                           double* value) {
  if (value == nullptr) {
    return false;
  }
  const std::string prefix = key + ":";
  if (line.find(prefix) != 0) {
    return false;
  }
  try {
    *value = std::stod(Trim(line.substr(prefix.size())));
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool LoadStage5BackendSummaryIntrinsics(
    const std::string& summary_path,
    ati::OuterBootstrapCameraIntrinsics* intrinsics,
    std::string* error_message) {
  if (intrinsics == nullptr) {
    if (error_message != nullptr) {
      *error_message = "LoadStage5BackendSummaryIntrinsics requires output.";
    }
    return false;
  }
  std::ifstream input(summary_path.c_str());
  if (!input.is_open()) {
    if (error_message != nullptr) {
      *error_message = "Failed to open Stage5 summary: " + summary_path;
    }
    return false;
  }
  bool has_xi = false;
  bool has_alpha = false;
  bool has_fu = false;
  bool has_fv = false;
  bool has_cu = false;
  bool has_cv = false;
  std::string line;
  while (std::getline(input, line)) {
    line = Trim(line);
    has_xi = TryReadKeyValueDouble(line, "camera_xi", &intrinsics->xi) || has_xi;
    has_alpha =
        TryReadKeyValueDouble(line, "camera_alpha", &intrinsics->alpha) ||
        has_alpha;
    has_fu = TryReadKeyValueDouble(line, "camera_fu", &intrinsics->fu) || has_fu;
    has_fv = TryReadKeyValueDouble(line, "camera_fv", &intrinsics->fv) || has_fv;
    has_cu = TryReadKeyValueDouble(line, "camera_cu", &intrinsics->cu) || has_cu;
    has_cv = TryReadKeyValueDouble(line, "camera_cv", &intrinsics->cv) || has_cv;
  }
  if (!(has_xi && has_alpha && has_fu && has_fv && has_cu && has_cv)) {
    if (error_message != nullptr) {
      *error_message =
          "Stage5 summary does not contain camera_xi/alpha/fu/fv/cu/cv: " +
          summary_path;
    }
    return false;
  }
  intrinsics->camera_model = "ds";
  intrinsics->distortion_model = "none";
  intrinsics->beta = 1.0;
  intrinsics->distortion_coeffs.clear();
  return true;
}

bool LoadIntrinsicsSpec(const IntrinsicsSpec& spec,
                        const cv::Size& fallback_resolution,
                        ati::OuterBootstrapCameraIntrinsics* intrinsics,
                        std::string* error_message) {
  if (intrinsics == nullptr) {
    if (error_message != nullptr) {
      *error_message = "LoadIntrinsicsSpec requires output.";
    }
    return false;
  }
  const fs::path input_path(spec.path);
  fs::path path_to_load = input_path;
  if (fs::is_directory(input_path)) {
    const fs::path holdout_summary = input_path / "backend_holdout_summary.txt";
    const fs::path training_summary = input_path / "backend_training_summary.txt";
    if (fs::exists(holdout_summary)) {
      path_to_load = holdout_summary;
    } else if (fs::exists(training_summary)) {
      path_to_load = training_summary;
    }
  }

  std::string load_error;
  if (ati::LoadKalibrCamchainIntrinsics(path_to_load.string(),
                                        intrinsics,
                                        &load_error)) {
    return true;
  }
  ati::OuterBootstrapCameraIntrinsics stage5_intrinsics;
  if (LoadStage5BackendSummaryIntrinsics(path_to_load.string(),
                                         &stage5_intrinsics,
                                         &load_error)) {
    stage5_intrinsics.resolution = fallback_resolution;
    if (stage5_intrinsics.resolution.width <= 0 ||
        stage5_intrinsics.resolution.height <= 0) {
      if (error_message != nullptr) {
        *error_message =
            "Stage5 summary intrinsics require detector/camchain resolution.";
      }
      return false;
    }
    *intrinsics = stage5_intrinsics;
    return true;
  }
  if (error_message != nullptr) {
    *error_message = "Failed to load intrinsics spec " + spec.path +
                     " as camchain yaml or Stage5 backend summary: " +
                     load_error;
  }
  return false;
}

bool IsImageFile(const fs::path& path) {
  if (!fs::is_regular_file(path)) {
    return false;
  }
  const std::string extension = ToLower(path.extension().string());
  return extension == ".png" || extension == ".jpg" ||
         extension == ".jpeg" || extension == ".bmp" ||
         extension == ".tif" || extension == ".tiff";
}

std::vector<std::string> CollectImagePaths(const std::string& image_path,
                                           bool all) {
  const fs::path input(image_path);
  if (!all) {
    return {image_path};
  }
  if (!fs::exists(input)) {
    throw std::runtime_error("Input path does not exist: " + image_path);
  }
  fs::path directory = input;
  if (fs::is_regular_file(input)) {
    directory = input.parent_path();
  }
  if (!fs::is_directory(directory)) {
    throw std::runtime_error("--all requires an image directory.");
  }
  std::vector<std::string> paths;
  for (fs::directory_iterator it(directory), end; it != end; ++it) {
    if (IsImageFile(it->path())) {
      paths.push_back(it->path().string());
    }
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) {
    throw std::runtime_error("No image files found in: " + directory.string());
  }
  return paths;
}

ati::IntermediateCameraConfig ToIntermediateCameraConfig(
    const ati::OuterBootstrapCameraIntrinsics& intrinsics,
    const std::string& yaml_path) {
  ati::IntermediateCameraConfig config;
  config.camera_yaml = yaml_path;
  config.camera_model = intrinsics.NormalizedCameraModel();
  config.distortion_model = intrinsics.NormalizedDistortionModel();
  config.intrinsics = intrinsics.IntrinsicsVector();
  config.distortion_coeffs = intrinsics.DistortionVector();
  config.resolution = {intrinsics.resolution.width, intrinsics.resolution.height};
  return config;
}

ati::OuterRefineCameraConfig ToOuterRefineCameraConfig(
    const ati::IntermediateCameraConfig& camera) {
  ati::OuterRefineCameraConfig config;
  config.camera_model = camera.camera_model;
  config.distortion_model = camera.distortion_model;
  config.intrinsics = camera.intrinsics;
  config.distortion_coeffs = camera.distortion_coeffs;
  config.resolution = camera.resolution;
  return config;
}

ati::ApriltagInternalDetectionOptions MakeDetectionOptions(
    const ati::ApriltagInternalConfig& config,
    const CmdArgs& args) {
  ati::ApriltagInternalDetectionOptions options;
  options.canonical_pixels_per_module = config.canonical_pixels_per_module;
  options.refinement_window_radius = config.refinement_window_radius;
  options.internal_subpix_window_scale = config.internal_subpix_window_scale;
  options.internal_subpix_window_min = config.internal_subpix_window_min;
  options.internal_subpix_window_max = config.internal_subpix_window_max;
  options.max_subpix_displacement2 = config.max_subpix_displacement2;
  options.internal_subpix_displacement_scale =
      config.internal_subpix_displacement_scale;
  options.max_internal_subpix_displacement =
      config.max_internal_subpix_displacement;
  options.ignore_image_evidence_min_quality =
      config.ignore_image_evidence_min_quality;
  options.force_internal_seed_from_prediction =
      config.force_internal_seed_from_prediction;
  options.bypass_internal_seed_filters = config.bypass_internal_seed_filters;
  options.enable_geometry_prior_outer_seed =
      args.enable_geometry_prior_rescue;
  options.geometry_prior_rescue_diagnostic_only =
      !args.geometry_prior_rescue_use_as_observation;
  options.geometry_prior_rescue_use_as_observation =
      args.geometry_prior_rescue_use_as_observation;
  options.geometry_prior_rescue_allow_geometry_only_pose_refit =
      args.geometry_prior_rescue_allow_geometry_only_pose_refit;
  options.geometry_prior_rescue_enable_spherical_refine =
      args.geometry_prior_rescue_enable_spherical_refine;
  options.outer_detector_config = config.outer_detector_config;
  return options;
}

std::vector<ati::FrozenRound2BaselineFrameSource> BuildFrameSources(
    const std::vector<std::string>& image_paths) {
  std::vector<ati::FrozenRound2BaselineFrameSource> sources;
  sources.reserve(image_paths.size());
  for (std::size_t index = 0; index < image_paths.size(); ++index) {
    ati::FrozenRound2BaselineFrameSource source;
    source.frame_index = static_cast<int>(index);
    source.frame_label = fs::path(image_paths[index]).stem().string();
    source.image_path = image_paths[index];
    sources.push_back(source);
  }
  return sources;
}

ati::CalibrationEvaluationDataset BuildEvaluationDatasetFromBundle(
    const ati::CalibrationStateBundle& bundle,
    const std::string& label) {
  ati::CalibrationEvaluationDataset dataset;
  dataset.dataset_label = label;
  dataset.split_label = "validation";
  dataset.split_signature = bundle.training_split_signature;

  for (const ati::JointMeasurementFrameResult& frame_result :
       bundle.measurement_dataset.frames) {
    ati::CalibrationEvaluationFrameInput frame_input;
    frame_input.frame_index = frame_result.frame_index;
    frame_input.frame_label = frame_result.frame_label;
    frame_input.visible_board_ids = frame_result.visible_board_ids;

    for (const ati::JointBoardObservation& board_observation :
         frame_result.board_observations) {
      ati::CalibrationEvaluationBoardObservation eval_board;
      eval_board.frame_index = frame_result.frame_index;
      eval_board.frame_label = frame_result.frame_label;
      eval_board.board_id = board_observation.board_id;

      for (const ati::JointPointObservation& point :
           board_observation.points) {
        if (!point.used_in_solver) {
          continue;
        }
        ati::CalibrationEvaluationPointObservation eval_point;
        eval_point.frame_index = point.frame_index;
        eval_point.frame_label = point.frame_label;
        eval_point.board_id = point.board_id;
        eval_point.point_id = point.point_id;
        eval_point.point_type = point.point_type;
        eval_point.image_xy = point.image_xy;
        eval_point.target_xyz_board = point.target_xyz_board;
        eval_point.quality = point.quality;
        eval_point.frame_storage_index = point.frame_storage_index;
        eval_point.source_board_observation_index =
            point.source_board_observation_index;
        eval_point.source_point_index = point.source_point_index;
        eval_point.source_kind = point.source_kind;
        eval_board.points.push_back(eval_point);
        if (eval_point.point_type == ati::JointPointType::Outer) {
          ++eval_board.outer_point_count;
        } else {
          ++eval_board.internal_point_count;
        }
      }

      eval_board.has_pose_fit_outer_points = eval_board.outer_point_count >= 4;
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
  dataset.total_point_count =
      dataset.outer_point_count + dataset.internal_point_count;
  dataset.success = dataset.frame_count > 0 &&
                    dataset.board_observation_count > 0 &&
                    dataset.total_point_count > 0;
  if (!dataset.success) {
    dataset.failure_reason =
        "Evaluation dataset is empty after frozen_round2 frontend.";
  }
  for (const std::string& warning : bundle.warnings) {
    dataset.warnings.push_back("bundle_warning: " + warning);
  }
  for (const std::string& warning : bundle.measurement_dataset.warnings) {
    dataset.warnings.push_back("measurement_warning: " + warning);
  }
  return dataset;
}

ati::CalibrationEvaluationDataset BuildFrozenRound2EvaluationDataset(
    const std::vector<std::string>& image_paths,
    const ati::ApriltagInternalConfig& config,
    const CmdArgs& args,
    FrontendRuntimeStats* frontend_stats) {
  ati::FrozenRound2BaselineOptions options;
  options.config = config;
  ati::OuterBootstrapCameraIntrinsics explicit_initial_camera;
  explicit_initial_camera.camera_model = config.intermediate_camera.camera_model;
  explicit_initial_camera.distortion_model =
      config.intermediate_camera.distortion_model;
  explicit_initial_camera.SetIntrinsicsVector(
      config.intermediate_camera.intrinsics);
  explicit_initial_camera.SetDistortionVector(
      config.intermediate_camera.distortion_coeffs);
  if (config.intermediate_camera.resolution.size() == 2) {
    explicit_initial_camera.resolution =
        cv::Size(config.intermediate_camera.resolution[0],
                 config.intermediate_camera.resolution[1]);
  }
  options.use_explicit_initial_camera = true;
  options.explicit_initial_camera = explicit_initial_camera;
  options.explicit_initial_camera_source_label =
      "benchmark_explicit_detector_intrinsics";
  options.optimize_intrinsics = false;
  options.run_second_pass = true;
  options.strict_board_observation_acceptance = true;
  options.enable_board_pose_fit_gate = false;
  options.enable_residual_sanity_gate = true;
  options.dataset_label = args.benchmark_label;
  options.source_pipeline_label = "intrinsics_reprojection_benchmark_frozen_round2";
  options.baseline_protocol_label = "intrinsics_reprojection_benchmark_frozen_round2";
  options.training_split_signature = "validation_all_frames";
  options.enable_outer_detection_cache = !args.frontend_cache_dir.empty();
  options.outer_detection_cache_dir = args.frontend_cache_dir;
  options.enable_geometry_prior_outer_seed = args.enable_geometry_prior_rescue;
  options.geometry_prior_rescue_diagnostic_only =
      !args.geometry_prior_rescue_use_as_observation;
  options.geometry_prior_rescue_use_as_observation =
      args.geometry_prior_rescue_use_as_observation;
  options.geometry_prior_rescue_allow_geometry_only_pose_refit =
      args.geometry_prior_rescue_allow_geometry_only_pose_refit;
  options.geometry_prior_rescue_enable_spherical_refine =
      args.geometry_prior_rescue_enable_spherical_refine;

  ati::FrozenRound2BaselinePipeline pipeline(options);
  const ati::FrozenRound2BaselineResult result =
      pipeline.Run(BuildFrameSources(image_paths));
  if (!result.success || !result.stage5_bundle_available ||
      !result.final_stage5_bundle.IsReadyForBackend()) {
    throw std::runtime_error(
        "frozen_round2 frontend failed: " + result.failure_reason);
  }

  const ati::Stage5Benchmark benchmark;
  ati::OuterDetectionCacheStats cache_stats;
  const ati::JointReprojectionSceneState optimized_scene_state =
      ati::BuildJointSceneStateFromCalibrationSceneState(
          result.final_stage5_bundle.scene_state);
  ati::CalibrationEvaluationDataset dataset =
      benchmark.BuildHoldoutEvaluationDataset(
          BuildFrameSources(image_paths),
          options,
          optimized_scene_state,
          "validation_all_frames",
          &cache_stats);
  dataset.dataset_label = args.benchmark_label;
  dataset.split_label = "validation";
  if (frontend_stats != nullptr) {
    frontend_stats->processed_image_count =
        static_cast<int>(image_paths.size());
    frontend_stats->detection_count = dataset.board_observation_count;
    frontend_stats->successful_detection_count = dataset.board_observation_count;
    frontend_stats->total_detect_seconds =
        result.runtime_breakdown.round1_regeneration_seconds +
        result.runtime_breakdown.round2_regeneration_seconds;
  }
  return dataset;
}

std::string PointTypeString(ati::JointPointType type) {
  return type == ati::JointPointType::Outer ? "outer" : "internal";
}

void AddDetectionToDataset(
    const ati::ApriltagInternalDetectionResult& detection,
    const std::string& image_path,
    int frame_index,
    ati::CalibrationEvaluationFrameInput* frame_input) {
  if (frame_input == nullptr || !detection.success) {
    return;
  }
  ati::CalibrationEvaluationBoardObservation board;
  board.frame_index = frame_index;
  board.frame_label = fs::path(image_path).stem().string();
  board.board_id = detection.board_id;

  for (std::size_t corner_index = 0; corner_index < detection.corners.size();
       ++corner_index) {
    const ati::CornerMeasurement& corner = detection.corners[corner_index];
    if (!corner.valid) {
      continue;
    }
    ati::CalibrationEvaluationPointObservation point;
    point.frame_index = frame_index;
    point.frame_label = board.frame_label;
    point.board_id = detection.board_id;
    point.point_id = corner.point_id;
    point.point_type = corner.corner_type == ati::CornerType::Outer
                           ? ati::JointPointType::Outer
                           : ati::JointPointType::Internal;
    point.image_xy = corner.image_xy;
    point.target_xyz_board = corner.target_xyz;
    point.quality = corner.quality;
    point.frame_storage_index = frame_index;
    point.source_board_observation_index = 0;
    point.source_point_index = static_cast<int>(corner_index);
    point.source_kind = point.point_type == ati::JointPointType::Outer
                            ? ati::JointObservationSourceKind::OuterMeasurement
                            : ati::JointObservationSourceKind::InternalMeasurement;
    board.points.push_back(point);
    if (point.point_type == ati::JointPointType::Outer) {
      ++board.outer_point_count;
    } else {
      ++board.internal_point_count;
    }
  }

  board.has_pose_fit_outer_points = board.outer_point_count >= 4;
  if (!board.points.empty() && board.has_pose_fit_outer_points) {
    frame_input->visible_board_ids.push_back(detection.board_id);
    frame_input->board_observations.push_back(board);
  }
}

void AccumulateDetectionRuntimeStats(
    const ati::ApriltagInternalDetectionResult& detection,
    FrontendRuntimeFrameStats* frame_stats) {
  if (frame_stats == nullptr) {
    return;
  }
  ++frame_stats->detection_count;
  if (detection.success) {
    ++frame_stats->successful_detection_count;
  }
  for (const ati::OuterCornerVerificationDebugInfo& debug :
       detection.outer_detection.corner_verification_debug) {
    if (debug.corner_index < 0) {
      continue;
    }
    ++frame_stats->outer_debug_corner_count;
    if (debug.close_edge_subpix_boost_applied) {
      ++frame_stats->close_edge_boost_corner_count;
    }
    if (!debug.line_refinement_success) {
      ++frame_stats->line_refinement_failed_count;
    }
    if (debug.line_seed_accepted) {
      ++frame_stats->line_seed_accepted_count;
    }
    if (debug.image_line_valid) {
      ++frame_stats->image_line_fallback_count;
      if (debug.line_seed_accepted && !debug.line_refinement_success) {
        ++frame_stats->image_line_fallback_accepted_count;
      }
    }
  }
}

void AccumulateFrameRuntimeStats(const FrontendRuntimeFrameStats& frame,
                                 FrontendRuntimeStats* total) {
  if (total == nullptr) {
    return;
  }
  ++total->processed_image_count;
  total->total_detect_seconds += frame.detect_seconds;
  total->detection_count += frame.detection_count;
  total->successful_detection_count += frame.successful_detection_count;
  total->outer_debug_corner_count += frame.outer_debug_corner_count;
  total->close_edge_boost_corner_count += frame.close_edge_boost_corner_count;
  total->line_refinement_failed_count += frame.line_refinement_failed_count;
  total->line_seed_accepted_count += frame.line_seed_accepted_count;
  total->image_line_fallback_count += frame.image_line_fallback_count;
  total->image_line_fallback_accepted_count +=
      frame.image_line_fallback_accepted_count;
  total->frames.push_back(frame);
}

ati::CalibrationEvaluationDataset BuildEvaluationDataset(
    const std::vector<std::string>& image_paths,
    const ati::ApriltagInternalDetector& detector,
    const std::string& label,
    int progress_every,
    FrontendRuntimeStats* frontend_stats) {
  ati::CalibrationEvaluationDataset dataset;
  dataset.dataset_label = label;
  dataset.split_label = "validation";
  dataset.split_signature = "external_validation_all_frames";

  for (std::size_t index = 0; index < image_paths.size(); ++index) {
    if (progress_every > 0 &&
        (index % static_cast<std::size_t>(progress_every) == 0)) {
      std::cout << "[benchmark] detecting " << (index + 1) << "/"
                << image_paths.size() << " "
                << fs::path(image_paths[index]).filename().string()
                << std::endl;
    }
    cv::Mat image = cv::imread(image_paths[index], cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      dataset.warnings.push_back("Failed to read image: " + image_paths[index]);
      continue;
    }

    ati::CalibrationEvaluationFrameInput frame;
    frame.frame_index = static_cast<int>(index);
    frame.frame_label = fs::path(image_paths[index]).stem().string();
    FrontendRuntimeFrameStats frame_stats;
    frame_stats.frame_index = static_cast<int>(index);
    frame_stats.frame_label = frame.frame_label;

    const auto detect_start = std::chrono::steady_clock::now();
    if (detector.requested_board_ids().size() > 1u) {
      const ati::ApriltagInternalMultiDetectionResult multi =
          detector.DetectMultiple(image);
      for (const ati::ApriltagInternalDetectionResult& detection :
           multi.detections) {
        AccumulateDetectionRuntimeStats(detection, &frame_stats);
        AddDetectionToDataset(detection, image_paths[index],
                              static_cast<int>(index), &frame);
      }
    } else {
      const ati::ApriltagInternalDetectionResult detection =
          detector.Detect(image);
      AccumulateDetectionRuntimeStats(detection, &frame_stats);
      AddDetectionToDataset(detection, image_paths[index],
                            static_cast<int>(index), &frame);
    }
    const auto detect_end = std::chrono::steady_clock::now();
    frame_stats.detect_seconds =
        std::chrono::duration<double>(detect_end - detect_start).count();
    AccumulateFrameRuntimeStats(frame_stats, frontend_stats);

    if (!frame.board_observations.empty()) {
      for (const ati::CalibrationEvaluationBoardObservation& board :
           frame.board_observations) {
        ++dataset.board_observation_count;
        dataset.outer_point_count += board.outer_point_count;
        dataset.internal_point_count += board.internal_point_count;
      }
      dataset.frames.push_back(frame);
    }
  }

  dataset.frame_count = static_cast<int>(dataset.frames.size());
  dataset.total_point_count =
      dataset.outer_point_count + dataset.internal_point_count;
  dataset.success = dataset.frame_count > 0 &&
                    dataset.board_observation_count > 0 &&
                    dataset.total_point_count > 0;
  if (!dataset.success) {
    dataset.failure_reason =
        "Evaluation dataset is empty after frontend detection.";
  }
  return dataset;
}

void WriteSummary(const std::string& path,
                  const CmdArgs& args,
                  const ati::OuterBootstrapCameraIntrinsics& intrinsics,
                  const ati::OuterBootstrapCameraIntrinsics& detector_intrinsics,
                  const ati::CalibrationEvaluationDataset& dataset,
                  const ati::CameraModelRefitEvaluationResult& evaluation,
                  const FrontendRuntimeStats& frontend_stats) {
  std::ofstream out(path.c_str());
  out << std::setprecision(10);
  out << "success: " << (evaluation.success ? 1 : 0) << "\n";
  out << "failure_reason: " << evaluation.failure_reason << "\n";
  out << "benchmark_label: " << args.benchmark_label << "\n";
  out << "frontend_mode: " << args.frontend_mode << "\n";
  out << "frontend_cache_dir: " << args.frontend_cache_dir << "\n";
  out << "image_path: " << args.image_path << "\n";
  out << "config_path: " << args.config_path << "\n";
  out << "intrinsics_yaml: " << args.active_intrinsics_path << "\n";
  out << "detector_intrinsics_yaml: " << args.detector_intrinsics_yaml
      << "\n";
  out << "camera_model_family: " << intrinsics.NormalizedFamilyString()
      << "\n";
  out << "camera_model: " << intrinsics.NormalizedCameraModel() << "\n";
  out << "distortion_model: " << intrinsics.NormalizedDistortionModel()
      << "\n";
  out << "resolution: [" << intrinsics.resolution.width << ", "
      << intrinsics.resolution.height << "]\n";
  out << "detector_camera_model_family: "
      << detector_intrinsics.NormalizedFamilyString() << "\n";
  out << "detector_camera_model: "
      << detector_intrinsics.NormalizedCameraModel() << "\n";
  out << "detector_distortion_model: "
      << detector_intrinsics.NormalizedDistortionModel() << "\n";
  out << "detector_resolution: [" << detector_intrinsics.resolution.width
      << ", " << detector_intrinsics.resolution.height << "]\n";
  out << "dataset_frame_count: " << dataset.frame_count << "\n";
  out << "dataset_board_observation_count: "
      << dataset.board_observation_count << "\n";
  out << "dataset_total_point_count: " << dataset.total_point_count << "\n";
  out << "dataset_outer_point_count: " << dataset.outer_point_count << "\n";
  out << "dataset_internal_point_count: " << dataset.internal_point_count
      << "\n";
  out << "frontend_processed_image_count: "
      << frontend_stats.processed_image_count << "\n";
  out << "frontend_total_detect_seconds: "
      << frontend_stats.total_detect_seconds << "\n";
  out << "frontend_mean_detect_seconds: "
      << (frontend_stats.processed_image_count > 0
              ? frontend_stats.total_detect_seconds /
                    static_cast<double>(frontend_stats.processed_image_count)
              : 0.0)
      << "\n";
  out << "frontend_detection_count: " << frontend_stats.detection_count
      << "\n";
  out << "frontend_successful_detection_count: "
      << frontend_stats.successful_detection_count << "\n";
  out << "frontend_outer_debug_corner_count: "
      << frontend_stats.outer_debug_corner_count << "\n";
  out << "frontend_close_edge_boost_corner_count: "
      << frontend_stats.close_edge_boost_corner_count << "\n";
  out << "frontend_line_refinement_failed_count: "
      << frontend_stats.line_refinement_failed_count << "\n";
  out << "frontend_line_seed_accepted_count: "
      << frontend_stats.line_seed_accepted_count << "\n";
  out << "frontend_image_line_fallback_count: "
      << frontend_stats.image_line_fallback_count << "\n";
  out << "frontend_image_line_fallback_accepted_count: "
      << frontend_stats.image_line_fallback_accepted_count << "\n";
  out << "evaluated_frame_count: " << evaluation.evaluated_frame_count << "\n";
  out << "evaluated_board_observation_count: "
      << evaluation.evaluated_board_observation_count << "\n";
  out << "pose_only_refit_attempt_count: "
      << evaluation.pose_only_refit_attempt_count << "\n";
  out << "pose_only_refit_success_count: "
      << evaluation.pose_only_refit_success_count << "\n";
  out << "pose_only_refit_success_rate: "
      << evaluation.pose_only_refit_success_rate << "\n";
  out << "pose_only_refit_rmse: " << evaluation.pose_only_refit_rmse << "\n";
  out << "overall_rmse: " << evaluation.overall_rmse << "\n";
  out << "outer_only_rmse: " << evaluation.outer_only_rmse << "\n";
  out << "internal_only_rmse: " << evaluation.internal_only_rmse << "\n";
  out << "mean_residual_x: " << evaluation.mean_residual_x << "\n";
  out << "mean_residual_y: " << evaluation.mean_residual_y << "\n";
  out << "std_residual_x: " << evaluation.std_residual_x << "\n";
  out << "std_residual_y: " << evaluation.std_residual_y << "\n";
  for (const std::string& warning : dataset.warnings) {
    out << "dataset_warning: " << warning << "\n";
  }
  for (const std::string& warning : evaluation.warnings) {
    out << "evaluation_warning: " << warning << "\n";
  }
}

void WriteFrontendRuntimeCsv(const std::string& path,
                             const FrontendRuntimeStats& stats) {
  std::ofstream out(path.c_str());
  out << "frame_index,frame_label,detect_seconds,detection_count,"
      << "successful_detection_count,outer_debug_corner_count,"
      << "close_edge_boost_corner_count,line_refinement_failed_count,"
      << "line_seed_accepted_count,image_line_fallback_count,"
      << "image_line_fallback_accepted_count\n";
  out << std::setprecision(10);
  for (const FrontendRuntimeFrameStats& frame : stats.frames) {
    out << frame.frame_index << "," << frame.frame_label << ","
        << frame.detect_seconds << "," << frame.detection_count << ","
        << frame.successful_detection_count << ","
        << frame.outer_debug_corner_count << ","
        << frame.close_edge_boost_corner_count << ","
        << frame.line_refinement_failed_count << ","
        << frame.line_seed_accepted_count << ","
        << frame.image_line_fallback_count << ","
        << frame.image_line_fallback_accepted_count << "\n";
  }
}

void WritePointsCsv(const std::string& path,
                    const ati::CameraModelRefitEvaluationResult& evaluation) {
  std::ofstream out(path.c_str());
  out << "method,split,frame_index,frame_label,board_id,point_id,point_type,"
      << "observed_x,observed_y,predicted_x,predicted_y,residual_x,"
      << "residual_y,residual_norm,quality\n";
  out << std::setprecision(10);
  for (const auto& p : evaluation.point_diagnostics) {
    out << p.method_label << "," << p.split_label << "," << p.frame_index
        << "," << p.frame_label << "," << p.board_id << "," << p.point_id
        << "," << PointTypeString(p.point_type) << ","
        << p.observed_image_xy.x() << "," << p.observed_image_xy.y() << ","
        << p.predicted_image_xy.x() << "," << p.predicted_image_xy.y() << ","
        << p.residual_xy.x() << "," << p.residual_xy.y() << ","
        << p.residual_norm << "," << p.quality << "\n";
  }
}

void WriteBoardCsv(const std::string& path,
                   const ati::CameraModelRefitEvaluationResult& evaluation) {
  std::ofstream out(path.c_str());
  out << "method,split,frame_index,frame_label,board_id,"
      << "pose_only_refit_success,point_count,outer_point_count,"
      << "internal_point_count,pose_fit_outer_rmse,evaluation_rmse,"
      << "outer_evaluation_rmse,internal_evaluation_rmse,failure_reason\n";
  out << std::setprecision(10);
  for (const auto& b : evaluation.board_observation_diagnostics) {
    out << b.method_label << "," << b.split_label << "," << b.frame_index
        << "," << b.frame_label << "," << b.board_id << ","
        << (b.pose_only_refit_success ? 1 : 0) << "," << b.point_count
        << "," << b.outer_point_count << "," << b.internal_point_count
        << "," << b.pose_fit_outer_rmse << "," << b.evaluation_rmse << ","
        << b.outer_evaluation_rmse << "," << b.internal_evaluation_rmse
        << "," << b.failure_reason << "\n";
  }
}

void WriteFrameCsv(const std::string& path,
                   const ati::CameraModelRefitEvaluationResult& evaluation) {
  std::ofstream out(path.c_str());
  out << "method,split,frame_index,frame_label,pose_attempt_count,"
      << "pose_success_count,pose_success_rate,pose_fit_rmse,point_count,"
      << "outer_point_count,internal_point_count,rmse,outer_rmse,internal_rmse\n";
  out << std::setprecision(10);
  for (const auto& f : evaluation.frame_diagnostics) {
    out << f.method_label << "," << f.split_label << "," << f.frame_index
        << "," << f.frame_label << "," << f.pose_only_refit_attempt_count
        << "," << f.pose_only_refit_success_count << ","
        << f.pose_only_refit_success_rate << "," << f.pose_only_refit_rmse
        << "," << f.point_count << "," << f.outer_point_count << ","
        << f.internal_point_count << "," << f.rmse << "," << f.outer_rmse
        << "," << f.internal_rmse << "\n";
  }
}

std::string PolarBucket(double polar_deg) {
  if (polar_deg < 30.0) return "polar_0_30";
  if (polar_deg < 50.0) return "polar_30_50";
  if (polar_deg < 70.0) return "polar_50_70";
  return "polar_70_plus";
}

double PolarAngleDeg(const ati::DoubleSphereCameraModel& camera,
                     const Eigen::Vector2d& pixel) {
  Eigen::Vector3d ray = Eigen::Vector3d::Zero();
  if (!camera.keypointToEuclidean(pixel, &ray) || ray.norm() <= 1e-12) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  ray.normalize();
  const double z = std::max(-1.0, std::min(1.0, ray.z()));
  return std::acos(z) * 180.0 / M_PI;
}

double AngularErrorDeg(const ati::DoubleSphereCameraModel& camera,
                       const Eigen::Vector2d& observed,
                       const Eigen::Vector2d& predicted) {
  Eigen::Vector3d obs_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d pred_ray = Eigen::Vector3d::Zero();
  if (!camera.keypointToEuclidean(observed, &obs_ray) ||
      !camera.keypointToEuclidean(predicted, &pred_ray) ||
      obs_ray.norm() <= 1e-12 || pred_ray.norm() <= 1e-12) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  obs_ray.normalize();
  pred_ray.normalize();
  const double dot =
      std::max(-1.0, std::min(1.0, obs_ray.dot(pred_ray)));
  return std::acos(dot) * 180.0 / M_PI;
}

void WritePolarCsv(const std::string& path,
                   const ati::OuterBootstrapCameraIntrinsics& intrinsics,
                   const ati::CameraModelRefitEvaluationResult& evaluation) {
  const ati::DoubleSphereCameraModel camera =
      ati::DoubleSphereCameraModel::FromConfig(ToIntermediateCameraConfig(
          intrinsics, std::string()));
  std::map<std::string, PolarBucketAccumulator> buckets;
  for (const auto& p : evaluation.point_diagnostics) {
    const double polar = PolarAngleDeg(camera, p.observed_image_xy);
    if (!std::isfinite(polar)) {
      continue;
    }
    const std::string key = std::to_string(p.board_id) + "," +
                            PointTypeString(p.point_type) + "," +
                            PolarBucket(polar);
    PolarBucketAccumulator& acc = buckets[key];
    const double pixel_error = p.residual_norm;
    const double angular_error =
        AngularErrorDeg(camera, p.observed_image_xy, p.predicted_image_xy);
    ++acc.point_count;
    acc.pixel_squared_sum += pixel_error * pixel_error;
    acc.max_pixel_error = std::max(acc.max_pixel_error, pixel_error);
    if (std::isfinite(angular_error)) {
      acc.angular_squared_sum_deg += angular_error * angular_error;
      acc.max_angular_error_deg =
          std::max(acc.max_angular_error_deg, angular_error);
    }
  }

  std::ofstream out(path.c_str());
  out << "board_id,point_type,polar_bucket,point_count,pixel_rmse,"
      << "angular_rmse_deg,max_pixel_error,max_angular_error_deg\n";
  out << std::setprecision(10);
  for (const auto& entry : buckets) {
    std::stringstream key(entry.first);
    std::string board;
    std::string point_type;
    std::string bucket;
    std::getline(key, board, ',');
    std::getline(key, point_type, ',');
    std::getline(key, bucket, ',');
    const PolarBucketAccumulator& acc = entry.second;
    const double denom = std::max(1, acc.point_count);
    out << board << "," << point_type << "," << bucket << ","
        << acc.point_count << ","
        << std::sqrt(acc.pixel_squared_sum / denom) << ","
        << std::sqrt(acc.angular_squared_sum_deg / denom) << ","
        << acc.max_pixel_error << "," << acc.max_angular_error_deg << "\n";
  }
}

void WriteComparisonCsv(const std::string& path,
                        const std::vector<ModelEvaluationRecord>& records) {
  std::ofstream out(path.c_str());
  out << std::setprecision(10);
  out << "label,intrinsics_yaml,camera_model_family,camera_model,"
      << "distortion_model,fu,fv,cu,cv,xi,alpha,beta,"
      << "point_count,outer_point_count,internal_point_count,"
      << "evaluated_frame_count,evaluated_board_observation_count,"
      << "pose_only_refit_attempt_count,pose_only_refit_success_count,"
      << "pose_only_refit_success_rate,pose_only_refit_rmse,"
      << "overall_rmse,outer_only_rmse,internal_only_rmse,"
      << "mean_residual_x,mean_residual_y,std_residual_x,std_residual_y\n";
  for (const ModelEvaluationRecord& record : records) {
    const ati::CameraModelRefitEvaluationResult& e = record.evaluation;
    const ati::OuterBootstrapCameraIntrinsics& k = record.intrinsics;
    out << record.label << ","
        << record.intrinsics_yaml << ","
        << k.NormalizedFamilyString() << ","
        << k.NormalizedCameraModel() << ","
        << k.NormalizedDistortionModel() << ","
        << k.fu << ","
        << k.fv << ","
        << k.cu << ","
        << k.cv << ","
        << k.xi << ","
        << k.alpha << ","
        << k.beta << ","
        << e.point_count << ","
        << e.outer_point_count << ","
        << e.internal_point_count << ","
        << e.evaluated_frame_count << ","
        << e.evaluated_board_observation_count << ","
        << e.pose_only_refit_attempt_count << ","
        << e.pose_only_refit_success_count << ","
        << e.pose_only_refit_success_rate << ","
        << e.pose_only_refit_rmse << ","
        << e.overall_rmse << ","
        << e.outer_only_rmse << ","
        << e.internal_only_rmse << ","
        << e.mean_residual_x << ","
        << e.mean_residual_y << ","
        << e.std_residual_x << ","
        << e.std_residual_y << "\n";
  }
}

double UnitRayPolarDeg(const Eigen::Vector3d& ray) {
  const Eigen::Vector3d unit = ray.normalized();
  const double z = std::max(-1.0, std::min(1.0, unit.z()));
  return std::acos(z) * 180.0 / M_PI;
}

double UnitRayAngularDiffDeg(const Eigen::Vector3d& lhs,
                             const Eigen::Vector3d& rhs) {
  const double dot =
      std::max(-1.0, std::min(1.0, lhs.normalized().dot(rhs.normalized())));
  return std::acos(dot) * 180.0 / M_PI;
}

std::string RayCurvePolarBucket(double polar_deg) {
  if (polar_deg < 30.0) return "polar_0_30";
  if (polar_deg < 50.0) return "polar_30_50";
  if (polar_deg < 70.0) return "polar_50_70";
  if (polar_deg < 90.0) return "polar_70_90";
  return "polar_90_plus";
}

std::string RayCurveRadialBucket(double radial_fraction) {
  if (radial_fraction < 0.2) return "radial_0_0p2";
  if (radial_fraction < 0.4) return "radial_0p2_0p4";
  if (radial_fraction < 0.6) return "radial_0p4_0p6";
  if (radial_fraction < 0.8) return "radial_0p6_0p8";
  return "radial_0p8_plus";
}

void AddRayCurveBucket(
    const std::string& key,
    double angular_diff_deg,
    double baseline_polar_deg,
    double reference_polar_deg,
    std::map<std::string, RayCurveBucketAccumulator>* buckets) {
  RayCurveBucketAccumulator& acc = (*buckets)[key];
  ++acc.count;
  acc.angular_sum += angular_diff_deg;
  acc.angular_square_sum += angular_diff_deg * angular_diff_deg;
  acc.max_angular = std::max(acc.max_angular, angular_diff_deg);
  acc.baseline_polar_sum += baseline_polar_deg;
  acc.reference_polar_sum += reference_polar_deg;
}

void WriteRayCurveComparisonCsvs(
    const fs::path& output_dir,
    const std::vector<ModelEvaluationRecord>& records) {
  if (records.size() < 2u || !records.front().intrinsics.IsValid()) {
    return;
  }
  ati::DoubleSphereCameraModel baseline_camera;
  try {
    baseline_camera = ati::DoubleSphereCameraModel::FromConfig(
        ToIntermediateCameraConfig(records.front().intrinsics,
                                   records.front().intrinsics_yaml));
  } catch (const std::exception&) {
    return;
  }

  const cv::Size resolution = records.front().intrinsics.resolution;
  constexpr int kGridWidth = 41;
  constexpr int kGridHeight = 41;
  const Eigen::Vector2d center(
      0.5 * static_cast<double>(std::max(1, resolution.width - 1)),
      0.5 * static_cast<double>(std::max(1, resolution.height - 1)));
  const double max_radius =
      std::max(1e-9, std::sqrt(center.x() * center.x() +
                               center.y() * center.y()));

  std::ofstream samples(
      (output_dir / "intrinsics_ray_curve_samples.csv").string().c_str());
  samples << std::setprecision(10);
  samples << "baseline_label,reference_label,reference_family,image_x,image_y,"
          << "radial_fraction,baseline_polar_deg,reference_polar_deg,"
          << "angular_diff_deg\n";
  std::map<std::string, RayCurveBucketAccumulator> buckets;

  for (std::size_t index = 1; index < records.size(); ++index) {
    const ModelEvaluationRecord& reference = records[index];
    if (!reference.intrinsics.IsValid()) {
      continue;
    }
    ati::DoubleSphereCameraModel reference_camera;
    try {
      reference_camera = ati::DoubleSphereCameraModel::FromConfig(
          ToIntermediateCameraConfig(reference.intrinsics,
                                     reference.intrinsics_yaml));
    } catch (const std::exception&) {
      continue;
    }
    for (int y_index = 0; y_index < kGridHeight; ++y_index) {
      const double y = static_cast<double>(y_index) *
                       static_cast<double>(resolution.height - 1) /
                       static_cast<double>(kGridHeight - 1);
      for (int x_index = 0; x_index < kGridWidth; ++x_index) {
        const double x = static_cast<double>(x_index) *
                         static_cast<double>(resolution.width - 1) /
                         static_cast<double>(kGridWidth - 1);
        const Eigen::Vector2d pixel(x, y);
        Eigen::Vector3d baseline_ray = Eigen::Vector3d::Zero();
        Eigen::Vector3d reference_ray = Eigen::Vector3d::Zero();
        if (!baseline_camera.keypointToEuclidean(pixel, &baseline_ray) ||
            !reference_camera.keypointToEuclidean(pixel, &reference_ray) ||
            baseline_ray.norm() <= 1e-12 || reference_ray.norm() <= 1e-12 ||
            !baseline_ray.allFinite() || !reference_ray.allFinite()) {
          continue;
        }
        const double radial_fraction = (pixel - center).norm() / max_radius;
        const double baseline_polar = UnitRayPolarDeg(baseline_ray);
        const double reference_polar = UnitRayPolarDeg(reference_ray);
        const double angular_diff =
            UnitRayAngularDiffDeg(baseline_ray, reference_ray);
        samples << records.front().label << ","
                << reference.label << ","
                << reference.intrinsics.NormalizedFamilyString() << ","
                << x << ","
                << y << ","
                << radial_fraction << ","
                << baseline_polar << ","
                << reference_polar << ","
                << angular_diff << "\n";
        AddRayCurveBucket(reference.label + ",all,all",
                          angular_diff, baseline_polar, reference_polar,
                          &buckets);
        AddRayCurveBucket(reference.label + ",baseline_polar," +
                              RayCurvePolarBucket(baseline_polar),
                          angular_diff, baseline_polar, reference_polar,
                          &buckets);
        AddRayCurveBucket(reference.label + ",radial," +
                              RayCurveRadialBucket(radial_fraction),
                          angular_diff, baseline_polar, reference_polar,
                          &buckets);
      }
    }
  }

  std::ofstream summary(
      (output_dir / "intrinsics_ray_curve_summary.csv").string().c_str());
  summary << std::setprecision(10);
  summary << "reference_label,bucket_type,bucket_label,sample_count,"
          << "mean_angular_diff_deg,rms_angular_diff_deg,"
          << "max_angular_diff_deg,mean_baseline_polar_deg,"
          << "mean_reference_polar_deg\n";
  for (const auto& entry : buckets) {
    std::stringstream key(entry.first);
    std::string reference_label;
    std::string bucket_type;
    std::string bucket_label;
    std::getline(key, reference_label, ',');
    std::getline(key, bucket_type, ',');
    std::getline(key, bucket_label, ',');
    const RayCurveBucketAccumulator& acc = entry.second;
    const double denom = std::max(1, acc.count);
    summary << reference_label << ","
            << bucket_type << ","
            << bucket_label << ","
            << acc.count << ","
            << acc.angular_sum / denom << ","
            << std::sqrt(acc.angular_square_sum / denom) << ","
            << acc.max_angular << ","
            << acc.baseline_polar_sum / denom << ","
            << acc.reference_polar_sum / denom << "\n";
  }
}

void DrawOverlayForFrame(const std::string& image_path,
                         const std::vector<ati::CameraModelRefitPointDiagnostics>& points,
                         const std::string& output_path) {
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    return;
  }
  for (const auto& p : points) {
    const cv::Point observed(static_cast<int>(std::lround(p.observed_image_xy.x())),
                             static_cast<int>(std::lround(p.observed_image_xy.y())));
    const cv::Point predicted(static_cast<int>(std::lround(p.predicted_image_xy.x())),
                              static_cast<int>(std::lround(p.predicted_image_xy.y())));
    cv::circle(image, observed, 3, cv::Scalar(0, 220, 0), 1, cv::LINE_AA);
    cv::circle(image, predicted, 3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    cv::line(image, observed, predicted, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
  }
  cv::imwrite(output_path, image);
}

void WriteWorstFrameOverlays(
    const fs::path& output_dir,
    const std::vector<std::string>& image_paths,
    const ati::CameraModelRefitEvaluationResult& evaluation,
    int top_k) {
  if (top_k <= 0) {
    return;
  }
  fs::create_directories(output_dir);
  std::map<int, std::string> image_by_frame;
  for (std::size_t index = 0; index < image_paths.size(); ++index) {
    image_by_frame[static_cast<int>(index)] = image_paths[index];
  }
  std::vector<ati::CameraModelRefitFrameDiagnostics> frames =
      evaluation.frame_diagnostics;
  std::sort(frames.begin(), frames.end(),
            [](const auto& lhs, const auto& rhs) {
              return lhs.rmse > rhs.rmse;
            });
  if (static_cast<int>(frames.size()) > top_k) {
    frames.resize(static_cast<std::size_t>(top_k));
  }
  for (std::size_t rank = 0; rank < frames.size(); ++rank) {
    const auto& frame = frames[rank];
    std::vector<ati::CameraModelRefitPointDiagnostics> points;
    for (const auto& point : evaluation.point_diagnostics) {
      if (point.frame_index == frame.frame_index) {
        points.push_back(point);
      }
    }
    const auto image_it = image_by_frame.find(frame.frame_index);
    if (image_it == image_by_frame.end()) {
      continue;
    }
    std::ostringstream name;
    name << "rank_" << std::setw(3) << std::setfill('0') << rank
         << "_frame_" << frame.frame_index << "_" << frame.frame_label
         << "_rmse_" << std::fixed << std::setprecision(2) << frame.rmse
         << ".png";
    DrawOverlayForFrame(image_it->second, points,
                        (output_dir / name.str()).string());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CmdArgs args = ParseArgs(argc, argv);
    fs::create_directories(args.output_path);

    std::vector<IntrinsicsSpec> evaluation_intrinsics_specs;
    evaluation_intrinsics_specs.reserve(args.intrinsics_specs.size());
    for (const std::string& spec : args.intrinsics_specs) {
      evaluation_intrinsics_specs.push_back(ParseIntrinsicsSpec(spec));
    }

    ati::OuterBootstrapCameraIntrinsics detector_intrinsics;
    std::string detector_intrinsics_error;
    if (!ati::LoadKalibrCamchainIntrinsics(args.detector_intrinsics_yaml,
                                           &detector_intrinsics,
                                           &detector_intrinsics_error)) {
      throw std::runtime_error(detector_intrinsics_error);
    }

    ati::ApriltagInternalConfig config =
        ati::ApriltagInternalDetector::LoadConfig(args.config_path);
    config.intermediate_camera =
        ToIntermediateCameraConfig(detector_intrinsics,
                                   args.detector_intrinsics_yaml);
    config.camera_initialization_mode = ati::CameraInitializationMode::Manual;
    config.outer_detector_config.refine_camera =
        ToOuterRefineCameraConfig(config.intermediate_camera);
    config.outer_detector_config.enable_outer_spherical_refinement = false;
    config.outer_detector_config.do_outer_subpix_refinement = true;

    std::vector<std::string> image_paths =
        CollectImagePaths(args.image_path, args.all);
    if (args.max_images > 0 &&
        static_cast<int>(image_paths.size()) > args.max_images) {
      image_paths.resize(static_cast<std::size_t>(args.max_images));
    }

    std::cout << "[benchmark] images=" << image_paths.size()
              << " label=" << args.benchmark_label
              << " frontend_mode=" << args.frontend_mode
              << " intrinsics_count=" << evaluation_intrinsics_specs.size()
              << " detector_intrinsics=" << args.detector_intrinsics_yaml
              << std::endl;
    FrontendRuntimeStats frontend_stats;
    ati::CalibrationEvaluationDataset dataset;
    if (args.frontend_mode == "detector") {
      const ati::ApriltagInternalDetectionOptions detection_options =
          MakeDetectionOptions(config, args);
      const ati::ApriltagInternalDetector detector(config, detection_options);
      dataset =
          BuildEvaluationDataset(image_paths, detector, args.benchmark_label,
                                 args.progress_every, &frontend_stats);
    } else if (args.frontend_mode == "frozen_round2" ||
               args.frontend_mode == "baseline") {
      dataset = BuildFrozenRound2EvaluationDataset(
          image_paths, config, args, &frontend_stats);
    } else {
      throw std::runtime_error(
          "Unsupported --frontend-mode: " + args.frontend_mode +
          " (expected detector or frozen_round2)");
    }
    if (!dataset.success) {
      throw std::runtime_error(dataset.failure_reason);
    }

    const ati::Stage5Benchmark benchmark;
    const fs::path output_dir(args.output_path);
    std::vector<ModelEvaluationRecord> model_records;
    for (std::size_t model_index = 0;
         model_index < evaluation_intrinsics_specs.size(); ++model_index) {
      ati::OuterBootstrapCameraIntrinsics intrinsics;
      std::string intrinsics_error;
      const IntrinsicsSpec& spec = evaluation_intrinsics_specs[model_index];
      if (!LoadIntrinsicsSpec(spec,
                              detector_intrinsics.resolution,
                              &intrinsics,
                              &intrinsics_error)) {
        throw std::runtime_error(intrinsics_error);
      }
      CmdArgs per_model_args = args;
      per_model_args.active_intrinsics_path = spec.path;
      per_model_args.benchmark_label = spec.label;
      const ati::CameraModelRefitEvaluationResult evaluation =
          benchmark.EvaluateCameraModel(dataset, intrinsics,
                                        per_model_args.benchmark_label);
      if (!evaluation.success) {
        throw std::runtime_error(evaluation.failure_reason);
      }
      model_records.push_back(ModelEvaluationRecord{
          per_model_args.benchmark_label,
          spec.path,
          intrinsics,
          evaluation});

      fs::path model_output_dir = output_dir;
      if (evaluation_intrinsics_specs.size() > 1u) {
        model_output_dir /=
            SanitizePathComponent(per_model_args.benchmark_label);
      }
      fs::create_directories(model_output_dir);
      WriteSummary(
          (model_output_dir / "intrinsics_reprojection_summary.txt").string(),
          per_model_args, intrinsics, detector_intrinsics, dataset, evaluation,
          frontend_stats);
      WriteFrontendRuntimeCsv(
          (model_output_dir / "frontend_runtime_summary.csv").string(),
          frontend_stats);
      WritePointsCsv(
          (model_output_dir / "intrinsics_reprojection_points.csv").string(),
          evaluation);
      WriteBoardCsv(
          (model_output_dir / "intrinsics_reprojection_per_board.csv").string(),
          evaluation);
      WriteFrameCsv(
          (model_output_dir / "intrinsics_reprojection_per_frame.csv").string(),
          evaluation);
      WritePolarCsv(
          (model_output_dir / "intrinsics_reprojection_board_polar.csv").string(),
          intrinsics, evaluation);
      if (args.export_overlays) {
        WriteWorstFrameOverlays(
            model_output_dir / "intrinsics_reprojection_top_bad_frames",
            image_paths, evaluation, args.overlay_top_k);
      }

      std::cout << "[benchmark] model=" << per_model_args.benchmark_label
                << " success=1 overall_rmse=" << evaluation.overall_rmse
                << " outer_rmse=" << evaluation.outer_only_rmse
                << " internal_rmse=" << evaluation.internal_only_rmse
                << " frames=" << evaluation.evaluated_frame_count
                << " boards=" << evaluation.evaluated_board_observation_count
                << std::endl;
    }
    WriteComparisonCsv(
        (output_dir / "intrinsics_reprojection_compare.csv").string(),
        model_records);
    WriteRayCurveComparisonCsvs(output_dir, model_records);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << std::endl;
    return 1;
  }
}
