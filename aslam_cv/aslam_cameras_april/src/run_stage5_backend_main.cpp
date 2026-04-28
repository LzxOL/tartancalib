#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>
#include <aslam/cameras/apriltag_internal/JointMeasurementCuration.hpp>
#include <aslam/cameras/apriltag_internal/KalibrBenchmark.hpp>
#include <aslam/cameras/apriltag_internal/OuterOnlyCameraInitializer.hpp>
#include <aslam/cameras/apriltag_internal/Stage5Benchmark.hpp>
#include <aslam/cameras/apriltag_internal/Stage5Runtime.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

namespace ati = aslam::cameras::apriltag_internal;
namespace fs = boost::filesystem;

constexpr const char kFrozenBaselineLabel[] = "stage5_backend_auto_v1";

enum class IntrinsicsReleaseMode {
  Delayed,
  Immediate,
  PoseOnly,
};

struct CmdArgs {
  std::string config_path;
  std::string image_path;
  std::string output_path;
  std::string kalibr_camchain_yaml;
  std::string kalibr_training_split_signature;
  std::string kalibr_source_label;
  std::string camera_init_mode_override;
  std::string experiment_tag;
  std::string cache_dir;
  bool all = false;
  bool show = false;
  int reference_board_id = 1;
  bool optimize_intrinsics = true;
  int intrinsics_release_iteration = 3;
  int second_pass_intrinsics_release_iteration = 1;
  int holdout_stride = 5;
  int holdout_offset = 0;
  bool disable_second_pass = false;
  IntrinsicsReleaseMode intrinsics_release_mode = IntrinsicsReleaseMode::Delayed;
  bool disable_residual_sanity_gate = false;
  bool enable_board_pose_fit_gate = false;
  bool strict_board_observation_acceptance = false;
  bool internal_regeneration_diagnostics = false;
  bool internal_blur_diagnostics = false;
  ati::InternalBlurFilterMode internal_blur_filter_mode =
      ati::InternalBlurFilterMode::Off;
  double internal_blur_filter_low_patch_gradient_quantile = 0.05;
  double internal_blur_filter_min_board_rmse_px = 5.0;
  double internal_blur_filter_min_board_p95_px = 5.0;
  ati::InternalPoseRescueMode internal_pose_rescue_mode =
      ati::InternalPoseRescueMode::Off;
  double internal_pose_rescue_max_ray_angle_deg = 85.0;
  double internal_pose_rescue_accept_max_outer_rmse = 8.0;
  ati::PreBackendFilterMode pre_backend_filter_mode =
      ati::PreBackendFilterMode::Off;
  double pre_backend_filter_sigma = 2.0;
  double pre_backend_filter_min_abs_threshold_px = 0.2;
  double kalibr_runtime_seconds = -1.0;
  ati::Stage5RuntimeMode runtime_mode = ati::Stage5RuntimeMode::Research;
};

std::string BuildKalibrSourceLabel(const std::string& kalibr_camchain_yaml) {
  return fs::path(kalibr_camchain_yaml).lexically_normal().generic_string();
}

double ElapsedSeconds(const std::chrono::steady_clock::time_point& start_time) {
  return std::chrono::duration_cast<std::chrono::duration<double> >(
             std::chrono::steady_clock::now() - start_time)
      .count();
}

void AddRuntimeStage(ati::Stage5RuntimeSummary* summary,
                     const std::string& stage_label,
                     double seconds,
                     bool skipped_in_fast_mode) {
  if (summary == nullptr) {
    return;
  }
  ati::Stage5RuntimeStageRecord record;
  record.stage_label = stage_label;
  record.wall_time_seconds = seconds;
  record.skipped_in_fast_mode = skipped_in_fast_mode;
  summary->stage_records.push_back(record);
}

struct RequestedExperimentConfig {
  std::string frozen_baseline_label = kFrozenBaselineLabel;
  std::string experiment_tag;
  std::string effective_protocol_label = kFrozenBaselineLabel;
  ati::CameraInitializationMode camera_init_mode =
      ati::CameraInitializationMode::Auto;
  bool run_second_pass = true;
  bool frontend_optimize_intrinsics = true;
  int frontend_intrinsics_release_iteration = 3;
  int frontend_second_pass_intrinsics_release_iteration = 1;
  IntrinsicsReleaseMode frontend_intrinsics_release_mode =
      IntrinsicsReleaseMode::Delayed;
  bool backend_optimize_intrinsics = true;
  bool backend_delayed_intrinsics_release = true;
  int backend_intrinsics_release_iteration = 1;
  IntrinsicsReleaseMode backend_intrinsics_release_mode =
      IntrinsicsReleaseMode::Delayed;
  bool enable_residual_sanity_gate = true;
  bool enable_board_pose_fit_gate = false;
  bool strict_board_observation_acceptance = false;
  bool internal_regeneration_diagnostics = false;
  bool internal_blur_diagnostics = false;
  ati::InternalBlurFilterMode internal_blur_filter_mode =
      ati::InternalBlurFilterMode::Off;
  double internal_blur_filter_low_patch_gradient_quantile = 0.05;
  double internal_blur_filter_min_board_rmse_px = 5.0;
  double internal_blur_filter_min_board_p95_px = 5.0;
  ati::InternalPoseRescueMode internal_pose_rescue_mode =
      ati::InternalPoseRescueMode::Off;
  double internal_pose_rescue_max_ray_angle_deg = 85.0;
  double internal_pose_rescue_accept_max_outer_rmse = 8.0;
  ati::PreBackendFilterMode pre_backend_filter_mode =
      ati::PreBackendFilterMode::Off;
  double pre_backend_filter_sigma = 2.0;
  double pre_backend_filter_min_abs_threshold_px = 0.2;
};

const char* ToString(IntrinsicsReleaseMode mode) {
  switch (mode) {
    case IntrinsicsReleaseMode::Delayed:
      return "delayed";
    case IntrinsicsReleaseMode::Immediate:
      return "immediate";
    case IntrinsicsReleaseMode::PoseOnly:
      return "pose_only";
  }
  return "unknown";
}

IntrinsicsReleaseMode ParseIntrinsicsReleaseMode(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (lowered == "delayed") {
    return IntrinsicsReleaseMode::Delayed;
  }
  if (lowered == "immediate") {
    return IntrinsicsReleaseMode::Immediate;
  }
  if (lowered == "pose_only" || lowered == "pose-only") {
    return IntrinsicsReleaseMode::PoseOnly;
  }
  throw std::runtime_error("Unsupported intrinsics release mode: " + value);
}

IntrinsicsReleaseMode DeriveFrontendIntrinsicsReleaseMode(
    bool optimize_intrinsics,
    bool run_second_pass,
    int round1_release_iteration,
    int round2_release_iteration) {
  if (!optimize_intrinsics) {
    return IntrinsicsReleaseMode::PoseOnly;
  }
  if (round1_release_iteration <= 0 &&
      (!run_second_pass || round2_release_iteration <= 0)) {
    return IntrinsicsReleaseMode::Immediate;
  }
  return IntrinsicsReleaseMode::Delayed;
}

IntrinsicsReleaseMode DeriveBackendIntrinsicsReleaseMode(
    bool optimize_intrinsics,
    bool delayed_intrinsics_release) {
  if (!optimize_intrinsics) {
    return IntrinsicsReleaseMode::PoseOnly;
  }
  return delayed_intrinsics_release ? IntrinsicsReleaseMode::Delayed
                                    : IntrinsicsReleaseMode::Immediate;
}

bool HasAblationOverrides(const RequestedExperimentConfig& config) {
  return config.camera_init_mode != ati::CameraInitializationMode::Auto ||
         !config.run_second_pass ||
         config.frontend_intrinsics_release_mode != IntrinsicsReleaseMode::Delayed ||
         config.backend_intrinsics_release_mode != IntrinsicsReleaseMode::Delayed ||
         !config.enable_residual_sanity_gate ||
         config.enable_board_pose_fit_gate ||
         config.strict_board_observation_acceptance ||
         config.internal_blur_filter_mode != ati::InternalBlurFilterMode::Off ||
         config.internal_pose_rescue_mode != ati::InternalPoseRescueMode::Off ||
         config.pre_backend_filter_mode != ati::PreBackendFilterMode::Off;
}

std::string BuildDeterministicExperimentTag(const RequestedExperimentConfig& config) {
  std::vector<std::string> parts;
  if (config.camera_init_mode != ati::CameraInitializationMode::Auto) {
    parts.push_back("caminit_" + std::string(ati::ToString(config.camera_init_mode)));
  }
  if (!config.run_second_pass) {
    parts.push_back("no_round2");
  }
  if (config.frontend_intrinsics_release_mode != IntrinsicsReleaseMode::Delayed) {
    parts.push_back("intrinsics_" +
                    std::string(ToString(config.frontend_intrinsics_release_mode)));
  }
  if (!config.enable_residual_sanity_gate) {
    parts.push_back("no_residual_gate");
  }
  if (config.enable_board_pose_fit_gate) {
    parts.push_back("board_pose_gate_debug");
  }
  if (config.strict_board_observation_acceptance) {
    parts.push_back("kalibr_style_failed_board_drop");
  }
  if (config.internal_blur_filter_mode != ati::InternalBlurFilterMode::Off) {
    parts.push_back("internal_blur_filter_" +
                    std::string(ati::ToString(config.internal_blur_filter_mode)));
  }
  if (config.internal_pose_rescue_mode != ati::InternalPoseRescueMode::Off) {
    parts.push_back("internal_pose_rescue_" +
                    std::string(ati::ToString(config.internal_pose_rescue_mode)));
  }
  if (config.pre_backend_filter_mode != ati::PreBackendFilterMode::Off) {
    parts.push_back("pre_backend_filter_" +
                    std::string(ati::ToString(config.pre_backend_filter_mode)));
  }
  if (parts.empty()) {
    return std::string();
  }
  std::ostringstream stream;
  for (std::size_t index = 0; index < parts.size(); ++index) {
    if (index > 0) {
      stream << "__";
    }
    stream << parts[index];
  }
  return stream.str();
}

RequestedExperimentConfig BuildRequestedExperimentConfig(const CmdArgs& args) {
  RequestedExperimentConfig config;
  config.camera_init_mode = args.camera_init_mode_override.empty()
                                ? ati::CameraInitializationMode::Auto
                                : ati::ParseCameraInitializationMode(
                                      args.camera_init_mode_override);
  config.run_second_pass = !args.disable_second_pass;
  config.enable_residual_sanity_gate = !args.disable_residual_sanity_gate;
  config.enable_board_pose_fit_gate = args.enable_board_pose_fit_gate;
  config.strict_board_observation_acceptance =
      args.strict_board_observation_acceptance;
  config.internal_regeneration_diagnostics =
      args.internal_regeneration_diagnostics;
  config.internal_blur_diagnostics = args.internal_blur_diagnostics;
  config.internal_blur_filter_mode = args.internal_blur_filter_mode;
  config.internal_blur_filter_low_patch_gradient_quantile =
      args.internal_blur_filter_low_patch_gradient_quantile;
  config.internal_blur_filter_min_board_rmse_px =
      args.internal_blur_filter_min_board_rmse_px;
  config.internal_blur_filter_min_board_p95_px =
      args.internal_blur_filter_min_board_p95_px;
  config.internal_pose_rescue_mode = args.internal_pose_rescue_mode;
  config.internal_pose_rescue_max_ray_angle_deg =
      args.internal_pose_rescue_max_ray_angle_deg;
  config.internal_pose_rescue_accept_max_outer_rmse =
      args.internal_pose_rescue_accept_max_outer_rmse;
  config.pre_backend_filter_mode = args.pre_backend_filter_mode;
  config.pre_backend_filter_sigma = args.pre_backend_filter_sigma;
  config.pre_backend_filter_min_abs_threshold_px =
      args.pre_backend_filter_min_abs_threshold_px;

  switch (args.intrinsics_release_mode) {
    case IntrinsicsReleaseMode::Delayed:
      config.frontend_optimize_intrinsics = true;
      config.frontend_intrinsics_release_iteration =
          args.intrinsics_release_iteration;
      config.frontend_second_pass_intrinsics_release_iteration =
          args.second_pass_intrinsics_release_iteration;
      config.frontend_intrinsics_release_mode = IntrinsicsReleaseMode::Delayed;
      config.backend_optimize_intrinsics = true;
      config.backend_delayed_intrinsics_release = true;
      config.backend_intrinsics_release_iteration =
          args.second_pass_intrinsics_release_iteration;
      config.backend_intrinsics_release_mode = IntrinsicsReleaseMode::Delayed;
      break;
    case IntrinsicsReleaseMode::Immediate:
      config.frontend_optimize_intrinsics = true;
      config.frontend_intrinsics_release_iteration = 0;
      config.frontend_second_pass_intrinsics_release_iteration = 0;
      config.frontend_intrinsics_release_mode = IntrinsicsReleaseMode::Immediate;
      config.backend_optimize_intrinsics = true;
      config.backend_delayed_intrinsics_release = false;
      config.backend_intrinsics_release_iteration = 0;
      config.backend_intrinsics_release_mode = IntrinsicsReleaseMode::Immediate;
      break;
    case IntrinsicsReleaseMode::PoseOnly:
      config.frontend_optimize_intrinsics = false;
      config.frontend_intrinsics_release_iteration = 0;
      config.frontend_second_pass_intrinsics_release_iteration = 0;
      config.frontend_intrinsics_release_mode = IntrinsicsReleaseMode::PoseOnly;
      config.backend_optimize_intrinsics = false;
      config.backend_delayed_intrinsics_release = false;
      config.backend_intrinsics_release_iteration = 0;
      config.backend_intrinsics_release_mode = IntrinsicsReleaseMode::PoseOnly;
      break;
  }

  config.experiment_tag = args.experiment_tag;
  if (config.experiment_tag.empty() && HasAblationOverrides(config)) {
    config.experiment_tag = BuildDeterministicExperimentTag(config);
  }
  if (!config.experiment_tag.empty()) {
    config.effective_protocol_label =
        config.frozen_baseline_label + "__" + config.experiment_tag;
  }
  return config;
}

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --image IMAGE_OR_DIR --config APRILTAG_INTERNAL_YAML --output OUTPUT_DIR"
      << " --kalibr-camchain CAMCHAIN_YAML [--all] [--show]"
      << " [--reference-board-id ID] [--optimize-intrinsics]"
      << " [--intrinsics-release-iteration N]"
      << " [--second-pass-intrinsics-release-iteration N]"
      << " [--holdout-stride N] [--holdout-offset N]"
      << " [--camera-init-mode manual|auto|auto_with_manual_fallback]"
      << " [--experiment-tag TAG] [--disable-second-pass]"
      << " [--intrinsics-release-mode delayed|immediate|pose_only]"
      << " [--runtime-mode research|fast]"
      << " [--cache-dir PATH]"
      << " [--disable-residual-sanity-gate] [--enable-board-pose-fit-gate]"
      << " [--strict-board-observation-acceptance]"
      << " [--internal-regeneration-diagnostics]"
      << " [--internal-blur-diagnostics]"
      << " [--internal-blur-filter-mode off|diagnostic|enabled]"
      << " [--internal-blur-filter-low-patch-gradient-quantile Q]"
      << " [--internal-blur-filter-min-board-rmse-px PX]"
      << " [--internal-blur-filter-min-board-p95-px PX]"
      << " [--internal-pose-rescue-mode off|diagnostic|enabled]"
      << " [--internal-pose-rescue-accept-max-outer-rmse PX]"
      << " [--pre-backend-filter-mode off|diagnostic|enabled]"
      << " [--pre-backend-filter-sigma SIGMA]"
      << " [--pre-backend-filter-min-abs-threshold-px PX]"
      << " [--kalibr-training-split-signature SIGNATURE]"
      << " [--kalibr-source-label LABEL (deprecated, ignored)]"
      << " [--kalibr-runtime-seconds SEC]\n";
}

CmdArgs ParseArgs(int argc, char** argv) {
  CmdArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string token = argv[i];
    if (token == "--image" && i + 1 < argc) {
      args.image_path = argv[++i];
    } else if (token == "--config" && i + 1 < argc) {
      args.config_path = argv[++i];
    } else if (token == "--output" && i + 1 < argc) {
      args.output_path = argv[++i];
    } else if (token == "--kalibr-camchain" && i + 1 < argc) {
      args.kalibr_camchain_yaml = argv[++i];
    } else if (token == "--kalibr-training-split-signature" && i + 1 < argc) {
      args.kalibr_training_split_signature = argv[++i];
    } else if (token == "--kalibr-source-label" && i + 1 < argc) {
      args.kalibr_source_label = argv[++i];
    } else if (token == "--camera-init-mode" && i + 1 < argc) {
      args.camera_init_mode_override = argv[++i];
    } else if (token == "--experiment-tag" && i + 1 < argc) {
      args.experiment_tag = argv[++i];
    } else if (token == "--runtime-mode" && i + 1 < argc) {
      args.runtime_mode = ati::ParseStage5RuntimeMode(argv[++i]);
    } else if (token == "--cache-dir" && i + 1 < argc) {
      args.cache_dir = argv[++i];
    } else if (token == "--disable-second-pass") {
      args.disable_second_pass = true;
    } else if (token == "--intrinsics-release-mode" && i + 1 < argc) {
      args.intrinsics_release_mode = ParseIntrinsicsReleaseMode(argv[++i]);
    } else if (token == "--disable-residual-sanity-gate") {
      args.disable_residual_sanity_gate = true;
    } else if (token == "--enable-board-pose-fit-gate") {
      args.enable_board_pose_fit_gate = true;
    } else if (token == "--disable-board-pose-fit-gate") {
      args.enable_board_pose_fit_gate = false;
    } else if (token == "--strict-board-observation-acceptance") {
      args.strict_board_observation_acceptance = true;
    } else if (token == "--internal-regeneration-diagnostics") {
      args.internal_regeneration_diagnostics = true;
    } else if (token == "--internal-blur-diagnostics") {
      args.internal_blur_diagnostics = true;
    } else if (token == "--internal-blur-filter-mode" && i + 1 < argc) {
      args.internal_blur_filter_mode =
          ati::ParseInternalBlurFilterMode(argv[++i]);
    } else if (token == "--internal-blur-filter-low-patch-gradient-quantile" &&
               i + 1 < argc) {
      args.internal_blur_filter_low_patch_gradient_quantile =
          std::stod(argv[++i]);
    } else if (token == "--internal-blur-filter-min-board-rmse-px" &&
               i + 1 < argc) {
      args.internal_blur_filter_min_board_rmse_px = std::stod(argv[++i]);
    } else if (token == "--internal-blur-filter-min-board-p95-px" &&
               i + 1 < argc) {
      args.internal_blur_filter_min_board_p95_px = std::stod(argv[++i]);
    } else if (token == "--internal-pose-rescue-mode" && i + 1 < argc) {
      args.internal_pose_rescue_mode =
          ati::ParseInternalPoseRescueMode(argv[++i]);
    } else if (token == "--internal-pose-rescue-max-ray-angle-deg" && i + 1 < argc) {
      args.internal_pose_rescue_max_ray_angle_deg = std::stod(argv[++i]);
    } else if (token == "--internal-pose-rescue-accept-max-outer-rmse" &&
               i + 1 < argc) {
      args.internal_pose_rescue_accept_max_outer_rmse = std::stod(argv[++i]);
    } else if (token == "--pre-backend-filter-mode" && i + 1 < argc) {
      args.pre_backend_filter_mode =
          ati::ParsePreBackendFilterMode(argv[++i]);
    } else if (token == "--pre-backend-filter-sigma" && i + 1 < argc) {
      args.pre_backend_filter_sigma = std::stod(argv[++i]);
    } else if (token == "--pre-backend-filter-min-abs-threshold-px" &&
               i + 1 < argc) {
      args.pre_backend_filter_min_abs_threshold_px = std::stod(argv[++i]);
    } else if (token == "--kalibr-runtime-seconds" && i + 1 < argc) {
      args.kalibr_runtime_seconds = std::stod(argv[++i]);
    } else if (token == "--all") {
      args.all = true;
    } else if (token == "--show") {
      args.show = true;
    } else if (token == "--reference-board-id" && i + 1 < argc) {
      args.reference_board_id = std::stoi(argv[++i]);
    } else if (token == "--optimize-intrinsics") {
      args.optimize_intrinsics = true;
    } else if (token == "--intrinsics-release-iteration" && i + 1 < argc) {
      args.intrinsics_release_iteration = std::stoi(argv[++i]);
    } else if (token == "--second-pass-intrinsics-release-iteration" && i + 1 < argc) {
      args.second_pass_intrinsics_release_iteration = std::stoi(argv[++i]);
    } else if (token == "--holdout-stride" && i + 1 < argc) {
      args.holdout_stride = std::stoi(argv[++i]);
    } else if (token == "--holdout-offset" && i + 1 < argc) {
      args.holdout_offset = std::stoi(argv[++i]);
    } else if (token == "--help" || token == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + token);
    }
  }

  if (args.image_path.empty() || args.config_path.empty() || args.output_path.empty() ||
      args.kalibr_camchain_yaml.empty()) {
    throw std::runtime_error(
        "--image, --config, --output and --kalibr-camchain are required.");
  }
  return args;
}

std::string InferDatasetLabel(const CmdArgs& args) {
  const fs::path output_dir(args.output_path);
  if (!output_dir.filename().string().empty()) {
    return output_dir.filename().string();
  }
  return fs::path(args.image_path).stem().string();
}

bool IsImageFile(const fs::path& path) {
  if (!fs::is_regular_file(path)) {
    return false;
  }
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
         extension == ".bmp" || extension == ".tif" || extension == ".tiff";
}

std::vector<std::string> CollectImagePaths(const std::string& image_path, bool all) {
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
    throw std::runtime_error("--all requires --image to point to a directory or a file inside it.");
  }

  std::vector<std::string> image_paths;
  for (fs::directory_iterator it(directory), end; it != end; ++it) {
    if (IsImageFile(it->path())) {
      image_paths.push_back(it->path().string());
    }
  }
  std::sort(image_paths.begin(), image_paths.end());
  if (image_paths.empty()) {
    throw std::runtime_error("No image files found in directory: " + directory.string());
  }
  return image_paths;
}

void EnsureDirectoryExists(const fs::path& directory) {
  if (!directory.empty()) {
    fs::create_directories(directory);
  }
}

cv::Mat RenderLabeledCompare(
    const std::vector<std::pair<cv::Mat, std::string> >& images_and_labels) {
  std::vector<cv::Mat> valid_images;
  int target_height = 0;
  for (const auto& entry : images_and_labels) {
    if (entry.first.empty()) {
      return cv::Mat();
    }
    target_height = std::max(target_height, entry.first.rows);
  }
  if (target_height <= 0) {
    return cv::Mat();
  }

  const int banner_height = 30;
  valid_images.reserve(images_and_labels.size());
  for (const auto& entry : images_and_labels) {
    cv::Mat padded;
    const int bottom = std::max(0, target_height - entry.first.rows);
    cv::copyMakeBorder(entry.first, padded, banner_height, bottom, 0, 0,
                       cv::BORDER_CONSTANT, cv::Scalar(24, 24, 24));
    cv::putText(padded, entry.second, cv::Point(12, 21), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
    valid_images.push_back(padded);
  }

  cv::Mat compare;
  cv::hconcat(valid_images, compare);
  return compare;
}

void WriteEvaluationSummary(const std::string& path,
                            const ati::CameraModelRefitEvaluationResult& evaluation) {
  std::ofstream output(path.c_str());
  output << "success: " << (evaluation.success ? 1 : 0) << "\n";
  output << "failure_reason: " << evaluation.failure_reason << "\n";
  output << "method_label: " << evaluation.method_label << "\n";
  output << "split_label: " << evaluation.split_label << "\n";
  output << "split_signature: " << evaluation.split_signature << "\n";
  output << "overall_rmse: " << evaluation.overall_rmse << "\n";
  output << "outer_only_rmse: " << evaluation.outer_only_rmse << "\n";
  output << "internal_only_rmse: " << evaluation.internal_only_rmse << "\n";
  output << "mean_residual_x: " << evaluation.mean_residual_x << "\n";
  output << "mean_residual_y: " << evaluation.mean_residual_y << "\n";
  output << "std_residual_x: " << evaluation.std_residual_x << "\n";
  output << "std_residual_y: " << evaluation.std_residual_y << "\n";
  output << "point_count: " << evaluation.point_count << "\n";
  output << "outer_point_count: " << evaluation.outer_point_count << "\n";
  output << "internal_point_count: " << evaluation.internal_point_count << "\n";
  output << "camera_xi: " << evaluation.camera.xi << "\n";
  output << "camera_alpha: " << evaluation.camera.alpha << "\n";
  output << "camera_fu: " << evaluation.camera.fu << "\n";
  output << "camera_fv: " << evaluation.camera.fv << "\n";
  output << "camera_cu: " << evaluation.camera.cu << "\n";
  output << "camera_cv: " << evaluation.camera.cv << "\n";
  for (const std::string& warning : evaluation.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteBackendComparisonSummary(
    const std::string& path,
    const ati::AslamBackendCalibrationResult& backend_result,
    const ati::CameraModelRefitEvaluationResult& frontend_training,
    const ati::CameraModelRefitEvaluationResult& backend_training,
    const ati::CameraModelRefitEvaluationResult& kalibr_training,
    const ati::CameraModelRefitEvaluationResult& frontend_holdout,
    const ati::CameraModelRefitEvaluationResult& backend_holdout,
    const ati::CameraModelRefitEvaluationResult& kalibr_holdout) {
  std::ofstream output(path.c_str());
  output << "backend_success: " << (backend_result.success ? 1 : 0) << "\n";
  output << "backend_failure_reason: " << backend_result.failure_reason << "\n";
  output << "backend_initial_overall_rmse: "
         << backend_result.initial_residual.overall_rmse << "\n";
  output << "backend_optimized_overall_rmse: "
         << backend_result.optimized_residual.overall_rmse << "\n";
  output << "training_frontend_overall_rmse: " << frontend_training.overall_rmse << "\n";
  output << "training_backend_overall_rmse: " << backend_training.overall_rmse << "\n";
  output << "training_kalibr_overall_rmse: " << kalibr_training.overall_rmse << "\n";
  output << "training_frontend_outer_only_rmse: " << frontend_training.outer_only_rmse << "\n";
  output << "training_backend_outer_only_rmse: " << backend_training.outer_only_rmse << "\n";
  output << "training_kalibr_outer_only_rmse: " << kalibr_training.outer_only_rmse << "\n";
  output << "training_frontend_internal_only_rmse: "
         << frontend_training.internal_only_rmse << "\n";
  output << "training_backend_internal_only_rmse: "
         << backend_training.internal_only_rmse << "\n";
  output << "training_kalibr_internal_only_rmse: "
         << kalibr_training.internal_only_rmse << "\n";
  output << "training_frontend_mean_residual_x: " << frontend_training.mean_residual_x << "\n";
  output << "training_frontend_mean_residual_y: " << frontend_training.mean_residual_y << "\n";
  output << "training_frontend_std_residual_x: " << frontend_training.std_residual_x << "\n";
  output << "training_frontend_std_residual_y: " << frontend_training.std_residual_y << "\n";
  output << "training_backend_mean_residual_x: " << backend_training.mean_residual_x << "\n";
  output << "training_backend_mean_residual_y: " << backend_training.mean_residual_y << "\n";
  output << "training_backend_std_residual_x: " << backend_training.std_residual_x << "\n";
  output << "training_backend_std_residual_y: " << backend_training.std_residual_y << "\n";
  output << "training_kalibr_mean_residual_x: " << kalibr_training.mean_residual_x << "\n";
  output << "training_kalibr_mean_residual_y: " << kalibr_training.mean_residual_y << "\n";
  output << "training_kalibr_std_residual_x: " << kalibr_training.std_residual_x << "\n";
  output << "training_kalibr_std_residual_y: " << kalibr_training.std_residual_y << "\n";
  output << "holdout_frontend_overall_rmse: " << frontend_holdout.overall_rmse << "\n";
  output << "holdout_backend_overall_rmse: " << backend_holdout.overall_rmse << "\n";
  output << "holdout_kalibr_overall_rmse: " << kalibr_holdout.overall_rmse << "\n";
  output << "holdout_frontend_outer_only_rmse: " << frontend_holdout.outer_only_rmse << "\n";
  output << "holdout_backend_outer_only_rmse: " << backend_holdout.outer_only_rmse << "\n";
  output << "holdout_kalibr_outer_only_rmse: " << kalibr_holdout.outer_only_rmse << "\n";
  output << "holdout_frontend_internal_only_rmse: "
         << frontend_holdout.internal_only_rmse << "\n";
  output << "holdout_backend_internal_only_rmse: "
         << backend_holdout.internal_only_rmse << "\n";
  output << "holdout_kalibr_internal_only_rmse: "
         << kalibr_holdout.internal_only_rmse << "\n";
  output << "holdout_frontend_mean_residual_x: " << frontend_holdout.mean_residual_x << "\n";
  output << "holdout_frontend_mean_residual_y: " << frontend_holdout.mean_residual_y << "\n";
  output << "holdout_frontend_std_residual_x: " << frontend_holdout.std_residual_x << "\n";
  output << "holdout_frontend_std_residual_y: " << frontend_holdout.std_residual_y << "\n";
  output << "holdout_backend_mean_residual_x: " << backend_holdout.mean_residual_x << "\n";
  output << "holdout_backend_mean_residual_y: " << backend_holdout.mean_residual_y << "\n";
  output << "holdout_backend_std_residual_x: " << backend_holdout.std_residual_x << "\n";
  output << "holdout_backend_std_residual_y: " << backend_holdout.std_residual_y << "\n";
  output << "holdout_kalibr_mean_residual_x: " << kalibr_holdout.mean_residual_x << "\n";
  output << "holdout_kalibr_mean_residual_y: " << kalibr_holdout.mean_residual_y << "\n";
  output << "holdout_kalibr_std_residual_x: " << kalibr_holdout.std_residual_x << "\n";
  output << "holdout_kalibr_std_residual_y: " << kalibr_holdout.std_residual_y << "\n";
  output << "training_frontend_minus_kalibr: "
         << (frontend_training.overall_rmse - kalibr_training.overall_rmse) << "\n";
  output << "training_backend_minus_kalibr: "
         << (backend_training.overall_rmse - kalibr_training.overall_rmse) << "\n";
  output << "holdout_frontend_minus_kalibr: "
         << (frontend_holdout.overall_rmse - kalibr_holdout.overall_rmse) << "\n";
  output << "holdout_backend_minus_kalibr: "
         << (backend_holdout.overall_rmse - kalibr_holdout.overall_rmse) << "\n";
  output << "frontend_camera_xi: " << frontend_holdout.camera.xi << "\n";
  output << "frontend_camera_alpha: " << frontend_holdout.camera.alpha << "\n";
  output << "frontend_camera_fu: " << frontend_holdout.camera.fu << "\n";
  output << "frontend_camera_fv: " << frontend_holdout.camera.fv << "\n";
  output << "frontend_camera_cu: " << frontend_holdout.camera.cu << "\n";
  output << "frontend_camera_cv: " << frontend_holdout.camera.cv << "\n";
  output << "backend_camera_xi: " << backend_holdout.camera.xi << "\n";
  output << "backend_camera_alpha: " << backend_holdout.camera.alpha << "\n";
  output << "backend_camera_fu: " << backend_holdout.camera.fu << "\n";
  output << "backend_camera_fv: " << backend_holdout.camera.fv << "\n";
  output << "backend_camera_cu: " << backend_holdout.camera.cu << "\n";
  output << "backend_camera_cv: " << backend_holdout.camera.cv << "\n";
  output << "kalibr_camera_xi: " << kalibr_holdout.camera.xi << "\n";
  output << "kalibr_camera_alpha: " << kalibr_holdout.camera.alpha << "\n";
  output << "kalibr_camera_fu: " << kalibr_holdout.camera.fu << "\n";
  output << "kalibr_camera_fv: " << kalibr_holdout.camera.fv << "\n";
  output << "kalibr_camera_cu: " << kalibr_holdout.camera.cu << "\n";
  output << "kalibr_camera_cv: " << kalibr_holdout.camera.cv << "\n";
  for (const std::string& warning : backend_result.warnings) {
    output << "backend_warning: " << warning << "\n";
  }
}

void WriteExperimentConfigSummary(
    const std::string& path,
    const RequestedExperimentConfig& requested,
    const ati::Stage5BenchmarkReport& report,
    const ati::AslamBackendCalibrationResult* backend_result) {
  std::ofstream output(path.c_str());
  output << "frozen_baseline_label: " << requested.frozen_baseline_label << "\n";
  output << "experiment_tag: " << requested.experiment_tag << "\n";
  output << "effective_protocol_label: " << requested.effective_protocol_label << "\n";

  output << "requested_camera_init_mode: "
         << ati::ToString(requested.camera_init_mode) << "\n";
  output << "requested_run_second_pass: " << (requested.run_second_pass ? 1 : 0) << "\n";
  output << "requested_frontend_optimize_intrinsics: "
         << (requested.frontend_optimize_intrinsics ? 1 : 0) << "\n";
  output << "requested_frontend_intrinsics_release_mode: "
         << ToString(requested.frontend_intrinsics_release_mode) << "\n";
  output << "requested_frontend_intrinsics_release_iteration: "
         << requested.frontend_intrinsics_release_iteration << "\n";
  output << "requested_frontend_second_pass_intrinsics_release_iteration: "
         << requested.frontend_second_pass_intrinsics_release_iteration << "\n";
  output << "requested_backend_optimize_intrinsics: "
         << (requested.backend_optimize_intrinsics ? 1 : 0) << "\n";
  output << "requested_backend_delayed_intrinsics_release: "
         << (requested.backend_delayed_intrinsics_release ? 1 : 0) << "\n";
  output << "requested_backend_intrinsics_release_mode: "
         << ToString(requested.backend_intrinsics_release_mode) << "\n";
  output << "requested_backend_intrinsics_release_iteration: "
         << requested.backend_intrinsics_release_iteration << "\n";
  output << "requested_enable_residual_sanity_gate: "
         << (requested.enable_residual_sanity_gate ? 1 : 0) << "\n";
  output << "requested_enable_board_pose_fit_gate: "
         << (requested.enable_board_pose_fit_gate ? 1 : 0) << "\n";
  output << "requested_strict_board_observation_acceptance: "
         << (requested.strict_board_observation_acceptance ? 1 : 0) << "\n";
  output << "requested_failed_board_drop_policy: "
         << (requested.strict_board_observation_acceptance
                 ? "drop_entire_board_observation"
                 : "keep_outer_when_internal_failed")
         << "\n";
  output << "requested_kalibr_style_pose_ray_limit_deg: 80\n";
  output << "requested_internal_regeneration_diagnostics: "
         << (requested.internal_regeneration_diagnostics ? 1 : 0) << "\n";
  output << "requested_internal_blur_diagnostics: "
         << (requested.internal_blur_diagnostics ? 1 : 0) << "\n";
  output << "requested_internal_blur_filter_mode: "
         << ati::ToString(requested.internal_blur_filter_mode) << "\n";
  output << "requested_internal_blur_filter_low_patch_gradient_quantile: "
         << requested.internal_blur_filter_low_patch_gradient_quantile << "\n";
  output << "requested_internal_blur_filter_min_board_rmse_px: "
         << requested.internal_blur_filter_min_board_rmse_px << "\n";
  output << "requested_internal_blur_filter_min_board_p95_px: "
         << requested.internal_blur_filter_min_board_p95_px << "\n";
  output << "requested_internal_pose_rescue_mode: "
         << ati::ToString(requested.internal_pose_rescue_mode) << "\n";
  output << "requested_internal_pose_rescue_max_ray_angle_deg: "
         << requested.internal_pose_rescue_max_ray_angle_deg << "\n";
  output << "requested_internal_pose_rescue_accept_max_outer_rmse: "
         << requested.internal_pose_rescue_accept_max_outer_rmse << "\n";
  output << "requested_pre_backend_filter_mode: "
         << ati::ToString(requested.pre_backend_filter_mode) << "\n";
  output << "requested_pre_backend_filter_sigma: "
         << requested.pre_backend_filter_sigma << "\n";
  output << "requested_pre_backend_filter_min_abs_threshold_px: "
         << requested.pre_backend_filter_min_abs_threshold_px << "\n";

  output << "stage5_success: " << (report.success ? 1 : 0) << "\n";
  output << "stage5_failure_reason: " << report.failure_reason << "\n";
  output << "effective_stage5_baseline_protocol_label: "
         << report.baseline_protocol_label << "\n";
  output << "effective_camera_init_mode: "
         << ati::ToString(report.baseline_result.auto_camera_initialization.selected_mode) << "\n";
  output << "effective_camera_init_source: "
         << report.baseline_result.auto_camera_initialization.selected_source_label << "\n";
  output << "effective_camera_init_fallback_used: "
         << (report.baseline_result.auto_camera_initialization.fallback_used ? 1 : 0) << "\n";
  output << "effective_run_second_pass: "
         << (report.baseline_result.effective_options.run_second_pass ? 1 : 0) << "\n";
  output << "effective_round2_available: "
         << (report.baseline_result.round2_available ? 1 : 0) << "\n";
  output << "effective_frontend_optimize_intrinsics: "
         << (report.baseline_result.effective_options.optimize_intrinsics ? 1 : 0) << "\n";
  output << "effective_frontend_intrinsics_release_mode: "
         << ToString(DeriveFrontendIntrinsicsReleaseMode(
                report.baseline_result.effective_options.optimize_intrinsics,
                report.baseline_result.effective_options.run_second_pass,
                report.baseline_result.effective_options.intrinsics_release_iteration,
                report.baseline_result.effective_options
                    .second_pass_intrinsics_release_iteration))
         << "\n";
  output << "effective_frontend_intrinsics_release_iteration: "
         << report.baseline_result.round1.optimization_result.intrinsics_release_iteration << "\n";
  output << "effective_frontend_second_pass_intrinsics_release_iteration: "
         << (report.baseline_result.round2_available
                 ? report.baseline_result.round2.optimization_result
                       .intrinsics_release_iteration
                 : report.baseline_result.effective_options
                       .second_pass_intrinsics_release_iteration)
         << "\n";
  output << "effective_enable_residual_sanity_gate: "
         << (report.baseline_result.effective_options.enable_residual_sanity_gate ? 1 : 0)
         << "\n";
  output << "effective_enable_board_pose_fit_gate: "
         << (report.baseline_result.effective_options.enable_board_pose_fit_gate ? 1 : 0)
         << "\n";
  output << "effective_strict_board_observation_acceptance: "
         << (report.baseline_result.effective_options
                     .strict_board_observation_acceptance
                 ? 1
                 : 0)
         << "\n";
  output << "effective_failed_board_drop_policy: "
         << (report.baseline_result.effective_options
                     .strict_board_observation_acceptance
                 ? "drop_entire_board_observation"
                 : "keep_outer_when_internal_failed")
         << "\n";
  output << "effective_kalibr_style_pose_ray_limit_deg: 80\n";
  output << "effective_internal_regeneration_diagnostics: "
         << (requested.internal_regeneration_diagnostics ? 1 : 0) << "\n";
  output << "effective_internal_blur_diagnostics: "
         << (requested.internal_blur_diagnostics ? 1 : 0) << "\n";
  output << "effective_internal_blur_filter_mode: "
         << ati::ToString(report.internal_blur_filter_result.options.mode)
         << "\n";
  output << "effective_internal_blur_filter_diagnostic_only: "
         << (report.internal_blur_filter_result.diagnostic_only ? 1 : 0)
         << "\n";
  output << "effective_internal_blur_filter_backend_input_changed: "
         << (report.internal_blur_filter_result.backend_input_changed ? 1 : 0)
         << "\n";
  output << "effective_internal_blur_filter_low_patch_gradient_quantile: "
         << report.internal_blur_filter_result.options
                .low_patch_gradient_quantile
         << "\n";
  output << "effective_internal_blur_filter_patch_gradient_threshold: "
         << report.internal_blur_filter_result.patch_gradient_threshold << "\n";
  output << "effective_internal_blur_filter_min_board_rmse_px: "
         << report.internal_blur_filter_result.options
                .min_board_internal_rmse_px
         << "\n";
  output << "effective_internal_blur_filter_min_board_p95_px: "
         << report.internal_blur_filter_result.options
                .min_board_p95_residual_px
         << "\n";
  output << "effective_internal_blur_filter_input_internal_point_count: "
         << report.internal_blur_filter_result.input_internal_point_count
         << "\n";
  output << "effective_internal_blur_filter_filtered_internal_point_count: "
         << report.internal_blur_filter_result.filtered_internal_point_count
         << "\n";
  output << "effective_internal_blur_filter_remaining_internal_point_count: "
         << report.internal_blur_filter_result.remaining_internal_point_count
         << "\n";
  output << "effective_internal_blur_filter_filtered_board_observation_count: "
         << report.internal_blur_filter_result.filtered_board_observation_count
         << "\n";
  output << "effective_internal_pose_rescue_mode: "
         << ati::ToString(report.baseline_result.effective_options
                              .internal_pose_rescue_mode)
         << "\n";
  output << "effective_internal_pose_rescue_max_ray_angle_deg: "
         << report.baseline_result.effective_options
                .internal_pose_rescue_max_ray_angle_deg
         << "\n";
  output << "effective_internal_pose_rescue_accept_max_outer_rmse: "
         << report.baseline_result.effective_options
                .internal_pose_rescue_accept_max_outer_rmse
         << "\n";
  output << "effective_pre_backend_filter_mode: "
         << ati::ToString(report.pre_backend_filter_result.options.mode) << "\n";
  output << "effective_pre_backend_filter_diagnostic_only: "
         << (report.pre_backend_filter_result.diagnostic_only ? 1 : 0) << "\n";
  output << "effective_pre_backend_filter_backend_input_changed: "
         << (report.pre_backend_filter_result.backend_input_changed ? 1 : 0)
         << "\n";
  output << "effective_pre_backend_filter_sigma: "
         << report.pre_backend_filter_result.options.sigma_threshold << "\n";
  output << "effective_pre_backend_filter_min_abs_threshold_px: "
         << report.pre_backend_filter_result.options.min_abs_threshold_px << "\n";
  output << "effective_pre_backend_filter_input_internal_point_count: "
         << report.pre_backend_filter_result.input_internal_point_count << "\n";
  output << "effective_pre_backend_filter_filtered_internal_point_count: "
         << report.pre_backend_filter_result.filtered_internal_point_count << "\n";
  output << "effective_pre_backend_filter_remaining_internal_point_count: "
         << report.pre_backend_filter_result.remaining_internal_point_count << "\n";
  output << "effective_residual_sanity_factor: 2.5\n";
  output << "effective_max_pose_fit_outer_rmse: 8\n";

  output << "effective_backend_problem_optimize_intrinsics: "
         << (report.backend_problem_input.optimization_masks.optimize_intrinsics ? 1 : 0)
         << "\n";
  output << "effective_backend_problem_delayed_intrinsics_release: "
         << (report.backend_problem_input.optimization_masks.delayed_intrinsics_release ? 1 : 0)
         << "\n";
  output << "effective_backend_problem_intrinsics_release_mode: "
         << ToString(DeriveBackendIntrinsicsReleaseMode(
                report.backend_problem_input.optimization_masks.optimize_intrinsics,
                report.backend_problem_input.optimization_masks
                    .delayed_intrinsics_release))
         << "\n";
  output << "effective_backend_problem_intrinsics_release_iteration: "
         << report.backend_problem_input.optimization_masks.intrinsics_release_iteration << "\n";

  output << "round1_selected_frame_count: "
         << report.baseline_result.round1.selection_result.accepted_frame_count << "\n";
  output << "round1_selected_board_observation_count: "
         << report.baseline_result.round1.selection_result
                .accepted_board_observation_count
         << "\n";
  output << "round1_selected_internal_point_count: "
         << report.baseline_result.round1.selection_result.accepted_internal_point_count
         << "\n";
  output << "round2_selected_frame_count: "
         << (report.baseline_result.round2_available
                 ? report.baseline_result.round2.selection_result.accepted_frame_count
                 : 0)
         << "\n";
  output << "round2_selected_board_observation_count: "
         << (report.baseline_result.round2_available
                 ? report.baseline_result.round2.selection_result
                       .accepted_board_observation_count
                 : 0)
         << "\n";
  output << "round2_selected_internal_point_count: "
         << (report.baseline_result.round2_available
                 ? report.baseline_result.round2.selection_result
                       .accepted_internal_point_count
                 : 0)
         << "\n";

  if (backend_result != nullptr) {
    output << "backend_success: " << (backend_result->success ? 1 : 0) << "\n";
    output << "backend_failure_reason: " << backend_result->failure_reason << "\n";
    output << "effective_backend_runner_optimize_intrinsics: "
           << (backend_result->effective_problem_input.optimization_masks
                       .optimize_intrinsics
                   ? 1
                   : 0)
           << "\n";
    output << "effective_backend_runner_delayed_intrinsics_release: "
           << (backend_result->effective_problem_input.optimization_masks
                       .delayed_intrinsics_release
                   ? 1
                   : 0)
           << "\n";
    output << "effective_backend_runner_intrinsics_release_mode: "
           << ToString(DeriveBackendIntrinsicsReleaseMode(
                  backend_result->effective_problem_input.optimization_masks
                      .optimize_intrinsics,
                  backend_result->effective_problem_input.optimization_masks
                      .delayed_intrinsics_release))
           << "\n";
    output << "effective_backend_runner_intrinsics_release_iteration: "
           << backend_result->effective_problem_input.optimization_masks
                  .intrinsics_release_iteration
           << "\n";
  }
}

std::string CsvEscape(const std::string& value) {
  const bool needs_quotes =
      value.find(',') != std::string::npos ||
      value.find('"') != std::string::npos ||
      value.find('\n') != std::string::npos ||
      value.find('\r') != std::string::npos;
  if (!needs_quotes) {
    return value;
  }
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"') {
      escaped += "\"\"";
    } else {
      escaped += ch;
    }
  }
  escaped += "\"";
  return escaped;
}

double SafeRatio(int numerator, int denominator) {
  if (denominator <= 0) {
    return 0.0;
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

struct InternalRegenerationFrameAggregate {
  std::string round_label;
  std::string split_label;
  int frame_index = -1;
  std::string frame_label;
  int visible_board_count = 0;
  int board_observation_count = 0;
  int successful_board_count = 0;
  int failed_board_count = 0;
  int attempted_internal_corner_count = 0;
  int valid_internal_corner_count = 0;
  double valid_internal_ratio = 0.0;
  double total_runtime_seconds = 0.0;
  int pose_rescue_attempt_count = 0;
  int pose_rescue_success_count = 0;
  int pose_rescue_used_count = 0;
  std::string failure_reasons;
};

struct InternalRegenerationAcceptanceAggregate {
  std::string round_label;
  std::string split_label;
  int outer_detected_board_observation_count = 0;
  int accepted_board_observation_count = 0;
  int failed_board_observation_count = 0;
  int dropped_board_observation_count = 0;
};

std::vector<InternalRegenerationFrameAggregate> BuildInternalRegenerationAggregates(
    const std::string& round_label,
    const std::string& split_label,
    const std::vector<ati::InternalRegenerationFrameResult>& frame_results) {
  std::vector<InternalRegenerationFrameAggregate> aggregates;
  aggregates.reserve(frame_results.size());
  for (const ati::InternalRegenerationFrameResult& frame : frame_results) {
    InternalRegenerationFrameAggregate aggregate;
    aggregate.round_label = round_label;
    aggregate.split_label = split_label;
    aggregate.frame_index = frame.frame_index;
    aggregate.frame_label = frame.frame_label;
    aggregate.visible_board_count =
        static_cast<int>(frame.visible_board_ids.size());
    aggregate.board_observation_count =
        static_cast<int>(frame.board_measurements.size());
    aggregate.total_runtime_seconds =
        frame.runtime_breakdown.pose_estimation_seconds +
        frame.runtime_breakdown.boundary_model_seconds +
        frame.runtime_breakdown.seed_search_seconds +
        frame.runtime_breakdown.ray_refine_seconds +
        frame.runtime_breakdown.image_evidence_seconds +
        frame.runtime_breakdown.subpix_seconds;
    aggregate.pose_rescue_attempt_count =
        frame.runtime_breakdown.pose_rescue_attempt_count;
    aggregate.pose_rescue_success_count =
        frame.runtime_breakdown.pose_rescue_success_count;
    aggregate.pose_rescue_used_count =
        frame.runtime_breakdown.pose_rescue_used_count;

    std::ostringstream failures;
    bool first_failure = true;
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      const ati::ApriltagInternalDetectionResult& detection =
          measurement.detection;
      aggregate.attempted_internal_corner_count +=
          detection.runtime_breakdown.attempted_internal_corner_count;
      aggregate.valid_internal_corner_count +=
          detection.runtime_breakdown.valid_internal_corner_count;
      if (detection.outer_detection.success && detection.success) {
        ++aggregate.successful_board_count;
      } else if (detection.outer_detection.success && !detection.success) {
        ++aggregate.failed_board_count;
        if (!first_failure) {
          failures << " | ";
        }
        failures << "board=" << measurement.board_id << ":"
                 << detection.failure_reason;
        first_failure = false;
      }
    }
    aggregate.valid_internal_ratio =
        SafeRatio(aggregate.valid_internal_corner_count,
                  aggregate.attempted_internal_corner_count);
    aggregate.failure_reasons = failures.str();
    aggregates.push_back(aggregate);
  }
  return aggregates;
}

void WriteInternalRegenerationDiagnostics(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report) {
  const fs::path observations_path =
      output_dir / "internal_regeneration_observations.csv";
  const fs::path failures_path =
      output_dir / "internal_regeneration_failures.csv";
  const fs::path worst_frames_path =
      output_dir / "internal_regeneration_worst_frames.csv";
  const fs::path acceptance_summary_path =
      output_dir / "internal_regeneration_acceptance_summary.csv";

  std::ofstream observations(observations_path.string().c_str());
  std::ofstream failures(failures_path.string().c_str());
  observations
      << "round,split,state_source,frame_index,frame_label,board_id,"
      << "outer_detected,detection_success,failure_reason,"
      << "frame_bootstrap_initialized,board_bootstrap_initialized,pose_prior_used,"
      << "valid_corner_count,expected_visible_point_count,"
      << "attempted_internal_corner_count,valid_internal_corner_count,"
      << "valid_internal_ratio,total_seconds,pose_estimation_seconds,"
      << "boundary_model_seconds,seed_search_seconds,ray_refine_seconds,"
      << "image_evidence_seconds,subpix_seconds,pose_estimation_call_count,"
      << "boundary_model_build_count,pose_rescue_attempted,pose_rescue_success,"
      << "pose_rescue_used,pose_rescue_rmse,pose_rescue_max_ray_angle_deg,"
      << "pose_rescue_ray_angle_limit_deg,"
      << "pose_rescue_accept_max_outer_rmse,pose_rescue_failure_reason\n";
  failures
      << "round,split,state_source,frame_index,frame_label,board_id,"
      << "failure_reason,pose_prior_used,pose_rescue_attempted,"
      << "pose_rescue_success,pose_rescue_used,pose_rescue_rmse,"
      << "pose_rescue_max_ray_angle_deg,pose_rescue_failure_reason\n";

  std::vector<InternalRegenerationFrameAggregate> all_aggregates;
  std::vector<InternalRegenerationAcceptanceAggregate> acceptance_aggregates;
  const bool strict_board_observation_acceptance =
      report.baseline_result.effective_options
          .strict_board_observation_acceptance;
  const auto write_round =
      [&](const std::string& round_label,
          const std::string& split_label,
          const std::vector<ati::InternalRegenerationFrameResult>& frame_results) {
        const std::vector<InternalRegenerationFrameAggregate> aggregates =
            BuildInternalRegenerationAggregates(round_label, split_label,
                                                frame_results);
        all_aggregates.insert(all_aggregates.end(), aggregates.begin(),
                              aggregates.end());
        InternalRegenerationAcceptanceAggregate acceptance;
        acceptance.round_label = round_label;
        acceptance.split_label = split_label;

        for (const ati::InternalRegenerationFrameResult& frame : frame_results) {
          for (const ati::RegeneratedBoardMeasurement& measurement :
               frame.board_measurements) {
            const ati::ApriltagInternalDetectionResult& detection =
                measurement.detection;
            if (detection.outer_detection.success) {
              ++acceptance.outer_detected_board_observation_count;
              if (detection.success) {
                ++acceptance.accepted_board_observation_count;
              } else {
                ++acceptance.failed_board_observation_count;
                if (strict_board_observation_acceptance) {
                  ++acceptance.dropped_board_observation_count;
                }
              }
            }
            const int attempted =
                detection.runtime_breakdown.attempted_internal_corner_count;
            const int valid =
                detection.runtime_breakdown.valid_internal_corner_count;
            const double valid_ratio = SafeRatio(valid, attempted);
            observations
                << round_label << ","
                << split_label << ","
                << frame.state_source_label << ","
                << frame.frame_index << ","
                << CsvEscape(frame.frame_label) << ","
                << measurement.board_id << ","
                << (detection.outer_detection.success ? 1 : 0) << ","
                << (detection.success ? 1 : 0) << ","
                << CsvEscape(detection.failure_reason) << ","
                << (measurement.frame_bootstrap_initialized ? 1 : 0) << ","
                << (measurement.board_bootstrap_initialized ? 1 : 0) << ","
                << (measurement.pose_prior_used ? 1 : 0) << ","
                << detection.valid_corner_count << ","
                << detection.expected_visible_point_count << ","
                << attempted << ","
                << valid << ","
                << valid_ratio << ","
                << detection.runtime_breakdown.total_seconds << ","
                << detection.runtime_breakdown.pose_estimation_seconds << ","
                << detection.runtime_breakdown.boundary_model_seconds << ","
                << detection.runtime_breakdown.seed_search_seconds << ","
                << detection.runtime_breakdown.ray_refine_seconds << ","
                << detection.runtime_breakdown.image_evidence_seconds << ","
                << detection.runtime_breakdown.subpix_seconds << ","
                << detection.runtime_breakdown.pose_estimation_call_count << ","
                << detection.runtime_breakdown.boundary_model_build_count << ","
                << (detection.pose_rescue_attempted ? 1 : 0) << ","
                << (detection.pose_rescue_success ? 1 : 0) << ","
                << (detection.pose_rescue_used ? 1 : 0) << ","
                << detection.pose_rescue_rmse << ","
                << detection.pose_rescue_max_ray_angle_deg << ","
                << detection.pose_rescue_ray_angle_limit_deg << ","
                << detection.pose_rescue_accept_max_outer_rmse << ","
                << CsvEscape(detection.pose_rescue_failure_reason) << "\n";

            if (detection.outer_detection.success && !detection.success) {
              failures
                  << round_label << ","
                  << split_label << ","
                  << frame.state_source_label << ","
                  << frame.frame_index << ","
                  << CsvEscape(frame.frame_label) << ","
                  << measurement.board_id << ","
                  << CsvEscape(detection.failure_reason) << ","
                  << (measurement.pose_prior_used ? 1 : 0) << ","
                  << (detection.pose_rescue_attempted ? 1 : 0) << ","
                  << (detection.pose_rescue_success ? 1 : 0) << ","
                  << (detection.pose_rescue_used ? 1 : 0) << ","
                  << detection.pose_rescue_rmse << ","
                  << detection.pose_rescue_max_ray_angle_deg << ","
                  << CsvEscape(detection.pose_rescue_failure_reason) << "\n";
            }
          }
        }
        acceptance_aggregates.push_back(acceptance);
      };

  write_round("round1", "training",
              report.baseline_result.round1.regeneration_results);
  if (report.baseline_result.round2_available) {
    write_round("round2", "training",
                report.baseline_result.round2.regeneration_results);
  }
  write_round("holdout", "holdout",
              report.holdout_dataset.internal_regeneration_results);

  std::ofstream worst_frames(worst_frames_path.string().c_str());
  worst_frames
      << "ranking_metric,rank,round,split,frame_index,frame_label,"
      << "visible_board_count,board_observation_count,successful_board_count,"
      << "failed_board_count,attempted_internal_corner_count,"
      << "valid_internal_corner_count,valid_internal_ratio,total_runtime_seconds,"
      << "pose_rescue_attempt_count,pose_rescue_success_count,"
      << "pose_rescue_used_count,failure_reasons\n";

  const auto write_ranked =
      [&](const std::string& metric,
          std::vector<InternalRegenerationFrameAggregate> ranked) {
        if (metric == "failed_boards") {
          std::sort(ranked.begin(), ranked.end(),
                    [](const InternalRegenerationFrameAggregate& lhs,
                       const InternalRegenerationFrameAggregate& rhs) {
                      if (lhs.failed_board_count != rhs.failed_board_count) {
                        return lhs.failed_board_count > rhs.failed_board_count;
                      }
                      return lhs.valid_internal_ratio < rhs.valid_internal_ratio;
                    });
        } else if (metric == "low_valid_ratio") {
          std::sort(ranked.begin(), ranked.end(),
                    [](const InternalRegenerationFrameAggregate& lhs,
                       const InternalRegenerationFrameAggregate& rhs) {
                      if (lhs.valid_internal_ratio != rhs.valid_internal_ratio) {
                        return lhs.valid_internal_ratio < rhs.valid_internal_ratio;
                      }
                      return lhs.failed_board_count > rhs.failed_board_count;
                    });
        } else {
          std::sort(ranked.begin(), ranked.end(),
                    [](const InternalRegenerationFrameAggregate& lhs,
                       const InternalRegenerationFrameAggregate& rhs) {
                      return lhs.total_runtime_seconds > rhs.total_runtime_seconds;
                    });
        }
        const std::size_t top_k = std::min<std::size_t>(20, ranked.size());
        for (std::size_t index = 0; index < top_k; ++index) {
          const InternalRegenerationFrameAggregate& row = ranked[index];
          worst_frames
              << metric << ","
              << (index + 1) << ","
              << row.round_label << ","
              << row.split_label << ","
              << row.frame_index << ","
              << CsvEscape(row.frame_label) << ","
              << row.visible_board_count << ","
              << row.board_observation_count << ","
              << row.successful_board_count << ","
              << row.failed_board_count << ","
              << row.attempted_internal_corner_count << ","
              << row.valid_internal_corner_count << ","
              << row.valid_internal_ratio << ","
              << row.total_runtime_seconds << ","
              << row.pose_rescue_attempt_count << ","
              << row.pose_rescue_success_count << ","
              << row.pose_rescue_used_count << ","
              << CsvEscape(row.failure_reasons) << "\n";
        }
      };

  write_ranked("failed_boards", all_aggregates);
  write_ranked("low_valid_ratio", all_aggregates);
  write_ranked("runtime_seconds", all_aggregates);

  std::ofstream acceptance_summary(
      acceptance_summary_path.string().c_str());
  acceptance_summary
      << "round,split,strict_board_observation_acceptance,"
      << "failed_board_drop_policy,outer_detected_board_observation_count,"
      << "accepted_board_observation_count,failed_board_observation_count,"
      << "dropped_board_observation_count\n";
  for (const InternalRegenerationAcceptanceAggregate& row :
       acceptance_aggregates) {
    acceptance_summary
        << row.round_label << ","
        << row.split_label << ","
        << (strict_board_observation_acceptance ? 1 : 0) << ","
        << (strict_board_observation_acceptance
                ? "drop_entire_board_observation"
                : "keep_outer_when_internal_failed")
        << ","
        << row.outer_detected_board_observation_count << ","
        << row.accepted_board_observation_count << ","
        << row.failed_board_observation_count << ","
        << row.dropped_board_observation_count << "\n";
  }
}

struct InternalBlurBoardRow {
  std::string split;
  int round_index = -1;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int internal_point_count = 0;
  int valid_internal_point_count = 0;
  int attempted_internal_point_count = 0;
  double valid_internal_ratio = 0.0;
  double internal_rmse = 0.0;
  double mean_residual_x = 0.0;
  double mean_residual_y = 0.0;
  double std_residual_x = 0.0;
  double std_residual_y = 0.0;
  double max_residual = 0.0;
  double p90_residual = 0.0;
  double p95_residual = 0.0;
  double board_center_radius = 0.0;
  double board_crop_sharpness_score = 0.0;
  double board_crop_laplacian_variance = 0.0;
  double mean_gradient_magnitude = 0.0;
  double corner_patch_mean_gradient = 0.0;
  double ghosting_or_double_edge_score = -1.0;
  bool crop_valid = false;
  std::string failure_reason;
  int used_in_backend = 0;
  int used_in_evaluation = 0;
};

struct InternalBlurFrameRow {
  std::string split;
  int round_index = -1;
  int frame_index = -1;
  std::string frame_label;
  int board_count = 0;
  int internal_point_count = 0;
  int valid_internal_point_count = 0;
  double internal_rmse = 0.0;
  double max_residual = 0.0;
  double p95_residual = 0.0;
  double mean_board_crop_sharpness_score = 0.0;
  double mean_laplacian_variance = 0.0;
  double mean_gradient_magnitude = 0.0;
  int low_sharpness_high_rmse_board_count = 0;
};

double Quantile(std::vector<double> values, double q) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double scaled = q * static_cast<double>(values.size() - 1);
  const std::size_t index =
      static_cast<std::size_t>(std::max(0.0, std::min(scaled, static_cast<double>(values.size() - 1))));
  return values[index];
}

std::map<int, std::string> BuildFrameImagePathMap(
    const std::vector<ati::FrozenRound2BaselineFrameSource>& frames) {
  std::map<int, std::string> paths;
  for (const ati::FrozenRound2BaselineFrameSource& frame : frames) {
    paths[frame.frame_index] = frame.image_path;
  }
  return paths;
}

std::map<std::pair<int, int>, const ati::RegeneratedBoardMeasurement*>
BuildRegeneratedBoardMap(
    const std::vector<ati::InternalRegenerationFrameResult>& frames) {
  std::map<std::pair<int, int>, const ati::RegeneratedBoardMeasurement*> map;
  for (const ati::InternalRegenerationFrameResult& frame : frames) {
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      map[std::make_pair(frame.frame_index, measurement.board_id)] =
          &measurement;
    }
  }
  return map;
}

std::set<std::pair<int, int> > BuildBackendInternalBoardSet(
    const ati::CalibrationBackendProblemInput& backend_problem_input) {
  std::set<std::pair<int, int> > keys;
  for (const ati::JointPointObservation& point :
       backend_problem_input.measurement_dataset.solver_observations) {
    if (point.used_in_solver && point.point_type == ati::JointPointType::Internal) {
      keys.insert(std::make_pair(point.frame_index, point.board_id));
    }
  }
  return keys;
}

bool ComputeBoardCropMetrics(const cv::Mat& image,
                             const std::array<cv::Point2f, 4>& outer_corners,
                             double* center_radius,
                             double* laplacian_variance,
                             double* mean_gradient_magnitude,
                             std::string* failure_reason) {
  if (image.empty()) {
    if (failure_reason != nullptr) {
      *failure_reason = "image_empty";
    }
    return false;
  }
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }

  std::vector<cv::Point2f> corners(outer_corners.begin(), outer_corners.end());
  cv::Rect rect = cv::boundingRect(corners);
  const int margin =
      std::max(8, static_cast<int>(std::round(0.15 * std::max(rect.width, rect.height))));
  rect.x -= margin;
  rect.y -= margin;
  rect.width += 2 * margin;
  rect.height += 2 * margin;
  rect &= cv::Rect(0, 0, gray.cols, gray.rows);
  if (rect.width < 8 || rect.height < 8) {
    if (failure_reason != nullptr) {
      *failure_reason = "board_crop_too_small";
    }
    return false;
  }

  cv::Mat crop = gray(rect);
  cv::Mat laplacian;
  cv::Laplacian(crop, laplacian, CV_64F);
  cv::Scalar lap_mean;
  cv::Scalar lap_std;
  cv::meanStdDev(laplacian, lap_mean, lap_std);
  if (laplacian_variance != nullptr) {
    *laplacian_variance = lap_std[0] * lap_std[0];
  }

  cv::Mat grad_x;
  cv::Mat grad_y;
  cv::Sobel(crop, grad_x, CV_64F, 1, 0, 3);
  cv::Sobel(crop, grad_y, CV_64F, 0, 1, 3);
  cv::Mat magnitude;
  cv::magnitude(grad_x, grad_y, magnitude);
  if (mean_gradient_magnitude != nullptr) {
    *mean_gradient_magnitude = cv::mean(magnitude)[0];
  }

  if (center_radius != nullptr) {
    cv::Point2f center(0.0f, 0.0f);
    for (const cv::Point2f& corner : corners) {
      center += corner;
    }
    center *= 0.25f;
    const double dx = static_cast<double>(center.x) - 0.5 * gray.cols;
    const double dy = static_cast<double>(center.y) - 0.5 * gray.rows;
    const double half_diagonal =
        0.5 * std::sqrt(static_cast<double>(gray.cols * gray.cols + gray.rows * gray.rows));
    *center_radius = half_diagonal > 0.0 ? std::sqrt(dx * dx + dy * dy) / half_diagonal : 0.0;
  }
  return true;
}

double ComputeCornerPatchMeanGradient(
    const cv::Mat& image,
    const std::vector<ati::CameraModelRefitPointDiagnostics>& points) {
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
  for (const ati::CameraModelRefitPointDiagnostics& point : points) {
    const int x = static_cast<int>(std::round(point.observed_image_xy.x()));
    const int y = static_cast<int>(std::round(point.observed_image_xy.y()));
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

std::vector<InternalBlurBoardRow> BuildInternalBlurBoardRowsForSplit(
    const std::string& split,
    int round_index,
    const ati::CalibrationEvaluationDataset& dataset,
    const ati::CameraModelRefitEvaluationResult& evaluation,
    const std::map<int, std::string>& image_paths,
    const ati::CalibrationBackendProblemInput& backend_problem_input) {
  std::map<std::pair<int, int>, std::vector<ati::CameraModelRefitPointDiagnostics> > points_by_board;
  for (const ati::CameraModelRefitPointDiagnostics& point : evaluation.point_diagnostics) {
    if (point.point_type != ati::JointPointType::Internal) {
      continue;
    }
    points_by_board[std::make_pair(point.frame_index, point.board_id)].push_back(point);
  }
  std::map<int, std::string> regeneration_frame_labels;
  for (const ati::InternalRegenerationFrameResult& frame :
       dataset.internal_regeneration_results) {
    regeneration_frame_labels[frame.frame_index] = frame.frame_label;
    for (const ati::RegeneratedBoardMeasurement& board :
         frame.board_measurements) {
      points_by_board[std::make_pair(frame.frame_index, board.board_id)];
    }
  }
  const auto regenerated_by_board =
      BuildRegeneratedBoardMap(dataset.internal_regeneration_results);
  const auto backend_internal_boards =
      BuildBackendInternalBoardSet(backend_problem_input);

  std::vector<InternalBlurBoardRow> rows;
  rows.reserve(points_by_board.size());
  std::map<int, cv::Mat> image_cache;
  for (const auto& entry : points_by_board) {
    const std::pair<int, int>& key = entry.first;
    const std::vector<ati::CameraModelRefitPointDiagnostics>& points = entry.second;
    InternalBlurBoardRow row;
    row.split = split;
    row.round_index = round_index;
    row.frame_index = key.first;
    row.board_id = key.second;
    row.frame_label = points.empty() ? std::string() : points.front().frame_label;
    if (row.frame_label.empty()) {
      const auto label_it = regeneration_frame_labels.find(row.frame_index);
      if (label_it != regeneration_frame_labels.end()) {
        row.frame_label = label_it->second;
      }
    }
    row.internal_point_count = static_cast<int>(points.size());
    row.used_in_evaluation = row.internal_point_count > 0 ? 1 : 0;
    row.used_in_backend =
        backend_internal_boards.count(key) > 0 && split == "training" ? 1 : 0;

    std::vector<double> norms;
    double sq_sum = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (const ati::CameraModelRefitPointDiagnostics& point : points) {
      norms.push_back(point.residual_norm);
      sq_sum += point.residual_norm * point.residual_norm;
      sum_x += point.residual_xy.x();
      sum_y += point.residual_xy.y();
    }
    if (!points.empty()) {
      row.internal_rmse =
          std::sqrt(sq_sum / static_cast<double>(points.size()));
      row.mean_residual_x = sum_x / static_cast<double>(points.size());
      row.mean_residual_y = sum_y / static_cast<double>(points.size());
      double var_x = 0.0;
      double var_y = 0.0;
      for (const ati::CameraModelRefitPointDiagnostics& point : points) {
        const double dx = point.residual_xy.x() - row.mean_residual_x;
        const double dy = point.residual_xy.y() - row.mean_residual_y;
        var_x += dx * dx;
        var_y += dy * dy;
      }
      row.std_residual_x =
          std::sqrt(var_x / static_cast<double>(points.size()));
      row.std_residual_y =
          std::sqrt(var_y / static_cast<double>(points.size()));
      row.max_residual = *std::max_element(norms.begin(), norms.end());
      row.p90_residual = Quantile(norms, 0.90);
      row.p95_residual = Quantile(norms, 0.95);
    }

    const auto regen_it = regenerated_by_board.find(key);
    if (regen_it != regenerated_by_board.end()) {
      const ati::ApriltagInternalDetectionResult& detection =
          regen_it->second->detection;
      row.valid_internal_point_count = detection.valid_internal_corner_count;
      row.attempted_internal_point_count =
          detection.runtime_breakdown.attempted_internal_corner_count;
      row.valid_internal_ratio =
          SafeRatio(row.valid_internal_point_count,
                    row.attempted_internal_point_count);
      if (!detection.success) {
        row.failure_reason = detection.failure_reason;
      }

      const auto image_path_it = image_paths.find(row.frame_index);
      if (image_path_it != image_paths.end()) {
        cv::Mat& image = image_cache[row.frame_index];
        if (image.empty()) {
          image = cv::imread(image_path_it->second, cv::IMREAD_UNCHANGED);
        }
        std::string crop_failure;
        row.crop_valid = ComputeBoardCropMetrics(
            image, detection.outer_corners, &row.board_center_radius,
            &row.board_crop_laplacian_variance,
            &row.mean_gradient_magnitude, &crop_failure);
        row.board_crop_sharpness_score = row.board_crop_laplacian_variance;
        row.corner_patch_mean_gradient =
            ComputeCornerPatchMeanGradient(image, points);
        if (!row.crop_valid && row.failure_reason.empty()) {
          row.failure_reason = crop_failure;
        }
      } else if (row.failure_reason.empty()) {
        row.failure_reason = "image_path_not_found";
      }
    } else {
      row.failure_reason = "regeneration_result_not_found";
    }
    rows.push_back(row);
  }
  return rows;
}

std::vector<InternalBlurFrameRow> BuildInternalBlurFrameRows(
    const std::vector<InternalBlurBoardRow>& board_rows) {
  std::map<std::pair<std::string, int>, InternalBlurFrameRow> frames;
  for (const InternalBlurBoardRow& board : board_rows) {
    InternalBlurFrameRow& frame =
        frames[std::make_pair(board.split, board.frame_index)];
    frame.split = board.split;
    frame.round_index = board.round_index;
    frame.frame_index = board.frame_index;
    frame.frame_label = board.frame_label;
    ++frame.board_count;
    frame.internal_point_count += board.internal_point_count;
    frame.valid_internal_point_count += board.valid_internal_point_count;
    frame.max_residual = std::max(frame.max_residual, board.max_residual);
    frame.mean_board_crop_sharpness_score += board.board_crop_sharpness_score;
    frame.mean_laplacian_variance += board.board_crop_laplacian_variance;
    frame.mean_gradient_magnitude += board.mean_gradient_magnitude;
  }

  std::map<std::pair<std::string, int>, std::vector<double> > residuals_by_frame;
  for (const InternalBlurBoardRow& board : board_rows) {
    residuals_by_frame[std::make_pair(board.split, board.frame_index)].push_back(
        board.internal_rmse);
  }

  std::vector<InternalBlurFrameRow> rows;
  rows.reserve(frames.size());
  for (auto& entry : frames) {
    InternalBlurFrameRow frame = entry.second;
    const std::vector<double>& rmses = residuals_by_frame[entry.first];
    double sq_sum = 0.0;
    for (double rmse : rmses) {
      sq_sum += rmse * rmse;
    }
    frame.internal_rmse =
        rmses.empty() ? 0.0
                      : std::sqrt(sq_sum / static_cast<double>(rmses.size()));
    frame.p95_residual = Quantile(rmses, 0.95);
    if (frame.board_count > 0) {
      frame.mean_board_crop_sharpness_score /= frame.board_count;
      frame.mean_laplacian_variance /= frame.board_count;
      frame.mean_gradient_magnitude /= frame.board_count;
    }
    rows.push_back(frame);
  }
  return rows;
}

std::string RadiusBin(double radius) {
  if (radius < 0.33) {
    return "center";
  }
  if (radius < 0.55) {
    return "mid";
  }
  if (radius < 0.75) {
    return "edge";
  }
  return "extreme_edge";
}

std::string SharpnessBin(double value, double low_quantile, double high_quantile) {
  if (value <= low_quantile) {
    return "low";
  }
  if (value >= high_quantile) {
    return "high";
  }
  return "medium";
}

void WriteInternalBlurDiagnostics(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report,
    const ati::CameraModelRefitEvaluationResult& backend_training_evaluation,
    const ati::CameraModelRefitEvaluationResult& backend_holdout_evaluation,
    const std::vector<ati::FrozenRound2BaselineFrameSource>& all_frames) {
  const std::map<int, std::string> image_paths = BuildFrameImagePathMap(all_frames);
  std::vector<InternalBlurBoardRow> board_rows =
      BuildInternalBlurBoardRowsForSplit(
          "training", report.baseline_result.final_stage5_bundle.round_index,
          report.training_dataset, backend_training_evaluation, image_paths,
          report.backend_problem_input);
  const std::vector<InternalBlurBoardRow> holdout_board_rows =
      BuildInternalBlurBoardRowsForSplit(
          "holdout", report.baseline_result.final_stage5_bundle.round_index,
          report.holdout_dataset, backend_holdout_evaluation, image_paths,
          report.backend_problem_input);
  board_rows.insert(board_rows.end(), holdout_board_rows.begin(),
                    holdout_board_rows.end());
  std::vector<InternalBlurFrameRow> frame_rows =
      BuildInternalBlurFrameRows(board_rows);

  std::vector<double> sharpness_values;
  sharpness_values.reserve(board_rows.size());
  for (const InternalBlurBoardRow& row : board_rows) {
    if (row.crop_valid) {
      sharpness_values.push_back(row.board_crop_sharpness_score);
    }
  }
  const double low_sharpness = Quantile(sharpness_values, 0.33);
  const double high_sharpness = Quantile(sharpness_values, 0.67);

  for (InternalBlurFrameRow& frame : frame_rows) {
    for (const InternalBlurBoardRow& board : board_rows) {
      if (board.split == frame.split && board.frame_index == frame.frame_index &&
          SharpnessBin(board.board_crop_sharpness_score, low_sharpness,
                       high_sharpness) == "low" &&
          board.internal_rmse >= 6.0) {
        ++frame.low_sharpness_high_rmse_board_count;
      }
    }
  }

  std::ofstream by_board(
      (output_dir / "internal_blur_diagnostics_by_board.csv").string().c_str());
  by_board
      << "split,round_index,frame_index,frame_label,board_id,"
      << "internal_point_count,valid_internal_point_count,"
      << "attempted_internal_point_count,valid_internal_ratio,internal_rmse,"
      << "mean_residual_x,mean_residual_y,std_residual_x,std_residual_y,"
      << "max_residual,p90_residual,p95_residual,board_center_radius,"
      << "radius_bin,sharpness_bin,board_crop_sharpness_score,"
      << "board_crop_laplacian_variance,mean_gradient_magnitude,"
      << "corner_patch_mean_gradient,ghosting_or_double_edge_score,"
      << "crop_valid,failure_reason,used_in_backend,used_in_evaluation\n";
  for (const InternalBlurBoardRow& row : board_rows) {
    by_board
        << row.split << ","
        << row.round_index << ","
        << row.frame_index << ","
        << CsvEscape(row.frame_label) << ","
        << row.board_id << ","
        << row.internal_point_count << ","
        << row.valid_internal_point_count << ","
        << row.attempted_internal_point_count << ","
        << row.valid_internal_ratio << ","
        << row.internal_rmse << ","
        << row.mean_residual_x << ","
        << row.mean_residual_y << ","
        << row.std_residual_x << ","
        << row.std_residual_y << ","
        << row.max_residual << ","
        << row.p90_residual << ","
        << row.p95_residual << ","
        << row.board_center_radius << ","
        << RadiusBin(row.board_center_radius) << ","
        << SharpnessBin(row.board_crop_sharpness_score, low_sharpness,
                        high_sharpness) << ","
        << row.board_crop_sharpness_score << ","
        << row.board_crop_laplacian_variance << ","
        << row.mean_gradient_magnitude << ","
        << row.corner_patch_mean_gradient << ","
        << row.ghosting_or_double_edge_score << ","
        << (row.crop_valid ? 1 : 0) << ","
        << CsvEscape(row.failure_reason) << ","
        << row.used_in_backend << ","
        << row.used_in_evaluation << "\n";
  }

  std::ofstream by_frame(
      (output_dir / "internal_blur_diagnostics_by_frame.csv").string().c_str());
  by_frame
      << "split,round_index,frame_index,frame_label,board_count,"
      << "internal_point_count,valid_internal_point_count,internal_rmse,"
      << "max_residual,p95_residual,mean_board_crop_sharpness_score,"
      << "mean_laplacian_variance,mean_gradient_magnitude,"
      << "low_sharpness_high_rmse_board_count\n";
  for (const InternalBlurFrameRow& row : frame_rows) {
    by_frame
        << row.split << ","
        << row.round_index << ","
        << row.frame_index << ","
        << CsvEscape(row.frame_label) << ","
        << row.board_count << ","
        << row.internal_point_count << ","
        << row.valid_internal_point_count << ","
        << row.internal_rmse << ","
        << row.max_residual << ","
        << row.p95_residual << ","
        << row.mean_board_crop_sharpness_score << ","
        << row.mean_laplacian_variance << ","
        << row.mean_gradient_magnitude << ","
        << row.low_sharpness_high_rmse_board_count << "\n";
  }

  std::vector<InternalBlurBoardRow> worst_boards = board_rows;
  std::sort(worst_boards.begin(), worst_boards.end(),
            [](const InternalBlurBoardRow& lhs, const InternalBlurBoardRow& rhs) {
              return lhs.internal_rmse > rhs.internal_rmse;
            });
  std::vector<InternalBlurFrameRow> worst_frames = frame_rows;
  std::sort(worst_frames.begin(), worst_frames.end(),
            [](const InternalBlurFrameRow& lhs, const InternalBlurFrameRow& rhs) {
              return lhs.internal_rmse > rhs.internal_rmse;
            });

  std::ofstream worst(
      (output_dir / "internal_blur_worst_cases.csv").string().c_str());
  worst << "case_type,rank,split,round_index,frame_index,frame_label,board_id,"
        << "internal_rmse,max_residual,p95_residual,sharpness_score,"
        << "mean_gradient_magnitude,board_center_radius,radius_bin,sharpness_bin\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(30, worst_boards.size()); ++i) {
    const InternalBlurBoardRow& row = worst_boards[i];
    worst << "board,"
          << (i + 1) << ","
          << row.split << ","
          << row.round_index << ","
          << row.frame_index << ","
          << CsvEscape(row.frame_label) << ","
          << row.board_id << ","
          << row.internal_rmse << ","
          << row.max_residual << ","
          << row.p95_residual << ","
          << row.board_crop_sharpness_score << ","
          << row.mean_gradient_magnitude << ","
          << row.board_center_radius << ","
          << RadiusBin(row.board_center_radius) << ","
          << SharpnessBin(row.board_crop_sharpness_score, low_sharpness,
                          high_sharpness) << "\n";
  }
  for (std::size_t i = 0; i < std::min<std::size_t>(30, worst_frames.size()); ++i) {
    const InternalBlurFrameRow& row = worst_frames[i];
    worst << "frame,"
          << (i + 1) << ","
          << row.split << ","
          << row.round_index << ","
          << row.frame_index << ","
          << CsvEscape(row.frame_label) << ","
          << -1 << ","
          << row.internal_rmse << ","
          << row.max_residual << ","
          << row.p95_residual << ","
          << row.mean_board_crop_sharpness_score << ","
          << row.mean_gradient_magnitude << ","
          << -1 << ","
          << "frame,"
          << SharpnessBin(row.mean_board_crop_sharpness_score, low_sharpness,
                          high_sharpness) << "\n";
  }

  std::map<int, std::pair<double, int> > rmse_by_board_id;
  std::map<std::string, std::pair<double, int> > rmse_by_radius_bin;
  std::map<std::string, std::pair<double, int> > rmse_by_sharpness_bin;
  int low_sharpness_high_rmse_count = 0;
  for (const InternalBlurBoardRow& row : board_rows) {
    rmse_by_board_id[row.board_id].first += row.internal_rmse;
    rmse_by_board_id[row.board_id].second += 1;
    rmse_by_radius_bin[RadiusBin(row.board_center_radius)].first += row.internal_rmse;
    rmse_by_radius_bin[RadiusBin(row.board_center_radius)].second += 1;
    const std::string sharpness_bin =
        SharpnessBin(row.board_crop_sharpness_score, low_sharpness,
                     high_sharpness);
    rmse_by_sharpness_bin[sharpness_bin].first += row.internal_rmse;
    rmse_by_sharpness_bin[sharpness_bin].second += 1;
    if (sharpness_bin == "low" && row.internal_rmse >= 6.0) {
      ++low_sharpness_high_rmse_count;
    }
  }

  std::ofstream summary(
      (output_dir / "internal_blur_diagnostics_summary.txt").string().c_str());
  summary << "enabled: 1\n";
  summary << "board_record_count: " << board_rows.size() << "\n";
  summary << "frame_record_count: " << frame_rows.size() << "\n";
  summary << "sharpness_low_quantile: " << low_sharpness << "\n";
  summary << "sharpness_high_quantile: " << high_sharpness << "\n";
  summary << "low_sharpness_high_internal_rmse_board_count: "
          << low_sharpness_high_rmse_count << "\n";
  summary << "ghosting_or_double_edge_score: not_computed_v1\n";
  summary << "\n[average_internal_rmse_by_board_id]\n";
  for (const auto& entry : rmse_by_board_id) {
    summary << "board_" << entry.first << ": "
            << entry.second.first / std::max(1, entry.second.second)
            << " count=" << entry.second.second << "\n";
  }
  summary << "\n[average_internal_rmse_by_radius_bin]\n";
  for (const auto& entry : rmse_by_radius_bin) {
    summary << entry.first << ": "
            << entry.second.first / std::max(1, entry.second.second)
            << " count=" << entry.second.second << "\n";
  }
  summary << "\n[average_internal_rmse_by_sharpness_bin]\n";
  for (const auto& entry : rmse_by_sharpness_bin) {
    summary << entry.first << ": "
            << entry.second.first / std::max(1, entry.second.second)
            << " count=" << entry.second.second << "\n";
  }
}

void WriteBackendDiagnosticArtifacts(const fs::path& output_dir,
                                     const std::string& prefix,
                                     const ati::AslamBackendCalibrationResult& result) {
  ati::WriteAslamBackendCalibrationSummary(
      (output_dir / (prefix + "_summary.txt")).string(), result);
  if (result.initial_cost_parity.success || !result.initial_cost_parity.failure_reason.empty()) {
    ati::WriteAslamBackendCostParitySummary(
        (output_dir / (prefix + "_cost_parity_initial_summary.txt")).string(),
        result.initial_cost_parity);
    ati::WriteAslamBackendCostParityCsv(
        (output_dir / (prefix + "_cost_parity_initial_points.csv")).string(),
        result.initial_cost_parity);
  }
  if (result.optimized_cost_parity.success ||
      !result.optimized_cost_parity.failure_reason.empty()) {
    ati::WriteAslamBackendCostParitySummary(
        (output_dir / (prefix + "_cost_parity_optimized_summary.txt")).string(),
        result.optimized_cost_parity);
    ati::WriteAslamBackendCostParityCsv(
        (output_dir / (prefix + "_cost_parity_optimized_points.csv")).string(),
        result.optimized_cost_parity);
  }
  if (result.jacobian_diagnostics.success ||
      !result.jacobian_diagnostics.failure_reason.empty()) {
    ati::WriteAslamBackendJacobianSummary(
        (output_dir / (prefix + "_jacobian_summary.txt")).string(),
        result.jacobian_diagnostics);
  }
}

bool HasBackendCostDescent(const ati::AslamBackendCalibrationResult& result) {
  if (result.initial_cost_parity.success && result.optimized_cost_parity.success) {
    return result.optimized_cost_parity.backend_reprojection_total_weighted_cost + 1e-9 <
           result.initial_cost_parity.backend_reprojection_total_weighted_cost;
  }
  for (const ati::AslamBackendOptimizationStageSummary& stage : result.stages) {
    if (stage.objective_final + 1e-9 < stage.objective_start) {
      return true;
    }
  }
  return false;
}

void PrintProgress(const std::string& message) {
  std::cout << "[stage5_backend] " << message << std::endl;
}

void PrintEvaluationProgress(
    const std::string& label,
    const ati::CameraModelRefitEvaluationResult& evaluation) {
  std::cout << "[stage5_backend] " << label
            << " overall=" << evaluation.overall_rmse
            << " outer=" << evaluation.outer_only_rmse
            << " internal=" << evaluation.internal_only_rmse
            << " points=" << evaluation.point_count << std::endl;
}

void PrintBackendResultProgress(
    const ati::AslamBackendCalibrationResult& result) {
  std::cout << "[stage5_backend] backend success=" << (result.success ? 1 : 0)
            << " initial_rmse=" << result.initial_residual.overall_rmse
            << " optimized_rmse=" << result.optimized_residual.overall_rmse
            << std::endl;
  for (const ati::AslamBackendOptimizationStageSummary& stage : result.stages) {
    std::cout << "[stage5_backend] stage " << stage.stage_label
              << " cost=" << stage.objective_start << " -> "
              << stage.objective_final
              << " iterations=" << stage.iterations
              << " failed=" << stage.failed_iterations
              << " lambda=" << stage.lm_lambda_final << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto total_start = std::chrono::steady_clock::now();
    const CmdArgs args = ParseArgs(argc, argv);
    const RequestedExperimentConfig requested_config =
        BuildRequestedExperimentConfig(args);
    ati::Stage5RuntimeSummary runtime_summary;
    runtime_summary.runtime_mode = args.runtime_mode;
    runtime_summary.cache_dir =
        args.cache_dir.empty() ? "result/.stage5_backend_cache" : args.cache_dir;
    runtime_summary.cache_enabled = true;
    const std::string dataset_label = InferDatasetLabel(args);
    std::vector<std::string> image_paths;
    {
      const auto stage_start = std::chrono::steady_clock::now();
      image_paths = CollectImagePaths(args.image_path, args.all);
      AddRuntimeStage(&runtime_summary, "image_collection", ElapsedSeconds(stage_start),
                      false);
    }
    PrintProgress("dataset=" + dataset_label);
    PrintProgress("input=" + args.image_path);
    PrintProgress("output=" + args.output_path);
    PrintProgress("collected_images=" + std::to_string(image_paths.size()));
    PrintProgress("runtime_mode=" +
                  std::string(ati::ToString(args.runtime_mode)));
    PrintProgress("cache_dir=" + runtime_summary.cache_dir);
    const std::string kalibr_source_label =
        BuildKalibrSourceLabel(args.kalibr_camchain_yaml);
    if (!args.kalibr_source_label.empty() &&
        args.kalibr_source_label != kalibr_source_label) {
      PrintProgress("warning: ignoring deprecated --kalibr-source-label=" +
                    args.kalibr_source_label +
                    "; using actual camchain path label=" +
                    kalibr_source_label);
    }

    std::vector<ati::FrozenRound2BaselineFrameSource> all_frames;
    all_frames.reserve(image_paths.size());
    for (std::size_t index = 0; index < image_paths.size(); ++index) {
      ati::FrozenRound2BaselineFrameSource frame_source;
      frame_source.frame_index = static_cast<int>(index);
      frame_source.frame_label = fs::path(image_paths[index]).stem().string();
      frame_source.image_path = image_paths[index];
      all_frames.push_back(frame_source);
    }

    ati::FrozenRound2BaselineOptions baseline_options;
    baseline_options.config = ati::ApriltagInternalDetector::LoadConfig(args.config_path);
    baseline_options.config.camera_initialization_mode =
        requested_config.camera_init_mode;
    baseline_options.reference_board_id = args.reference_board_id;
    baseline_options.optimize_intrinsics =
        requested_config.frontend_optimize_intrinsics;
    baseline_options.intrinsics_release_iteration =
        requested_config.frontend_intrinsics_release_iteration;
    baseline_options.run_second_pass = requested_config.run_second_pass;
    baseline_options.second_pass_intrinsics_release_iteration =
        requested_config.frontend_second_pass_intrinsics_release_iteration;
    baseline_options.enable_residual_sanity_gate =
        requested_config.enable_residual_sanity_gate;
    baseline_options.enable_board_pose_fit_gate =
        requested_config.enable_board_pose_fit_gate;
    baseline_options.strict_board_observation_acceptance =
        requested_config.strict_board_observation_acceptance;
    baseline_options.internal_pose_rescue_mode =
        requested_config.internal_pose_rescue_mode;
    baseline_options.internal_pose_rescue_max_ray_angle_deg =
        requested_config.internal_pose_rescue_max_ray_angle_deg;
    baseline_options.internal_pose_rescue_accept_max_outer_rmse =
        requested_config.internal_pose_rescue_accept_max_outer_rmse;
    baseline_options.dataset_label = dataset_label;
    baseline_options.baseline_protocol_label =
        requested_config.effective_protocol_label;
    baseline_options.source_pipeline_label = "run_stage5_backend";
    baseline_options.enable_outer_detection_cache = true;
    baseline_options.outer_detection_cache_dir = runtime_summary.cache_dir;

    if (args.runtime_mode == ati::Stage5RuntimeMode::Fast && args.show) {
      PrintProgress("warning: --show is ignored in fast runtime mode.");
    }

    ati::BackendProblemOptions backend_options;
    backend_options.reference_board_id = args.reference_board_id;
    backend_options.optimize_frame_poses = true;
    backend_options.optimize_board_poses = true;
    backend_options.optimize_intrinsics =
        requested_config.backend_optimize_intrinsics;
    backend_options.delayed_intrinsics_release =
        requested_config.backend_delayed_intrinsics_release;
    backend_options.intrinsics_release_iteration =
        requested_config.backend_intrinsics_release_iteration;

    ati::CalibrationBenchmarkSplitOptions split_options;
    split_options.holdout_stride = args.holdout_stride;
    split_options.holdout_offset = args.holdout_offset;
    const ati::Stage5Benchmark benchmark(split_options);
    ati::CalibrationBenchmarkSplit preview_split;
    {
      const auto stage_start = std::chrono::steady_clock::now();
      preview_split = benchmark.BuildDeterministicSplit(all_frames);
      AddRuntimeStage(&runtime_summary, "split_preview", ElapsedSeconds(stage_start),
                      false);
    }
    if (preview_split.success) {
      PrintProgress("split=" + preview_split.split_signature +
                    " training=" +
                    std::to_string(preview_split.training_frames.size()) +
                    " holdout=" +
                    std::to_string(preview_split.holdout_frames.size()));
    } else {
      PrintProgress("split preview failed: " + preview_split.failure_reason);
    }
    const std::string kalibr_training_split_signature =
        !args.kalibr_training_split_signature.empty()
            ? args.kalibr_training_split_signature
            : (preview_split.success ? preview_split.split_signature : std::string());

    ati::KalibrBenchmarkReference kalibr_reference;
    kalibr_reference.camchain_yaml = args.kalibr_camchain_yaml;
    kalibr_reference.camera_model_family = "ds";
    kalibr_reference.training_split_signature = kalibr_training_split_signature;
    kalibr_reference.runtime_seconds = args.kalibr_runtime_seconds;
    kalibr_reference.source_label = kalibr_source_label;

    ati::Stage5BenchmarkInput benchmark_input;
    benchmark_input.all_frames = all_frames;
    benchmark_input.baseline_options = baseline_options;
    benchmark_input.backend_options = backend_options;
    benchmark_input.pre_backend_filter_options.mode =
        requested_config.pre_backend_filter_mode;
    benchmark_input.pre_backend_filter_options.sigma_threshold =
        requested_config.pre_backend_filter_sigma;
    benchmark_input.pre_backend_filter_options.min_abs_threshold_px =
        requested_config.pre_backend_filter_min_abs_threshold_px;
    benchmark_input.internal_blur_filter_options.mode =
        requested_config.internal_blur_filter_mode;
    benchmark_input.internal_blur_filter_options.low_patch_gradient_quantile =
        requested_config.internal_blur_filter_low_patch_gradient_quantile;
    benchmark_input.internal_blur_filter_options.min_board_internal_rmse_px =
        requested_config.internal_blur_filter_min_board_rmse_px;
    benchmark_input.internal_blur_filter_options.min_board_p95_residual_px =
        requested_config.internal_blur_filter_min_board_p95_px;
    benchmark_input.kalibr_reference = kalibr_reference;
    benchmark_input.dataset_label = dataset_label;
    benchmark_input.enable_diagnostic_compare =
        args.runtime_mode == ati::Stage5RuntimeMode::Research;

    const fs::path output_dir(args.output_path);
    EnsureDirectoryExists(output_dir);
    const auto write_runtime_summary = [&runtime_summary, &output_dir, &total_start]() {
      runtime_summary.total_runtime_seconds = ElapsedSeconds(total_start);
      ati::WriteStage5RuntimeSummary(
          (output_dir / "runtime_summary.txt").string(), runtime_summary);
    };

    PrintProgress("running frozen frontend baseline + Stage 5 benchmark...");
    const ati::Stage5BenchmarkReport report = benchmark.Run(benchmark_input);
    runtime_summary.training_detection_cache_hits =
        report.baseline_result.runtime_breakdown.training_detection_cache.cache_hits;
    runtime_summary.training_detection_cache_misses =
        report.baseline_result.runtime_breakdown.training_detection_cache.cache_misses;
    runtime_summary.holdout_detection_cache_hits =
        report.runtime_breakdown.holdout_detection_cache.cache_hits;
    runtime_summary.holdout_detection_cache_misses =
        report.runtime_breakdown.holdout_detection_cache.cache_misses;
    runtime_summary.round1_regeneration_attempted_internal_corners =
        report.baseline_result.runtime_breakdown
            .round1_regeneration_attempted_internal_corners;
    runtime_summary.round1_regeneration_valid_internal_corners =
        report.baseline_result.runtime_breakdown
            .round1_regeneration_valid_internal_corners;
    runtime_summary.round2_regeneration_attempted_internal_corners =
        report.baseline_result.runtime_breakdown
            .round2_regeneration_attempted_internal_corners;
    runtime_summary.round2_regeneration_valid_internal_corners =
        report.baseline_result.runtime_breakdown
            .round2_regeneration_valid_internal_corners;
    runtime_summary.round1_optimization_residual_evaluation_call_count =
        report.baseline_result.runtime_breakdown
            .round1_optimization_residual_evaluation_call_count;
    runtime_summary.round1_optimization_cost_evaluation_call_count =
        report.baseline_result.runtime_breakdown
            .round1_optimization_cost_evaluation_call_count;
    runtime_summary.round2_optimization_residual_evaluation_call_count =
        report.baseline_result.runtime_breakdown
            .round2_optimization_residual_evaluation_call_count;
    runtime_summary.round2_optimization_cost_evaluation_call_count =
        report.baseline_result.runtime_breakdown
            .round2_optimization_cost_evaluation_call_count;
    AddRuntimeStage(&runtime_summary, "training_outer_detection_load_build",
                    report.baseline_result.runtime_breakdown.training_outer_detection_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "auto_camera_initialization",
                    report.baseline_result.runtime_breakdown
                        .auto_camera_initialization_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "outer_bootstrap",
                    report.baseline_result.runtime_breakdown.outer_bootstrap_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration",
                    report.baseline_result.runtime_breakdown.round1_regeneration_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_pose_estimation",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_pose_estimation_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_boundary_model",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_boundary_model_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_seed_search",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_seed_search_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_ray_refine",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_ray_refine_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_image_evidence",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_image_evidence_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_regeneration_subpix",
                    report.baseline_result.runtime_breakdown
                        .round1_regeneration_subpix_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_measurement_build",
                    report.baseline_result.runtime_breakdown
                        .round1_measurement_build_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_residual_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round1_residual_evaluation_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_selection",
                    report.baseline_result.runtime_breakdown.round1_selection_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization",
                    report.baseline_result.runtime_breakdown.round1_optimization_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization_residual_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round1_optimization_residual_evaluation_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization_cost_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round1_optimization_cost_evaluation_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization_frame_updates",
                    report.baseline_result.runtime_breakdown
                        .round1_optimization_frame_update_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization_board_updates",
                    report.baseline_result.runtime_breakdown
                        .round1_optimization_board_update_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round1_optimization_intrinsics_updates",
                    report.baseline_result.runtime_breakdown
                        .round1_optimization_intrinsics_update_seconds,
                    false);
    AddRuntimeStage(&runtime_summary, "round2_regeneration",
                    report.baseline_result.runtime_breakdown.round2_regeneration_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_pose_estimation",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_pose_estimation_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_boundary_model",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_boundary_model_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_seed_search",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_seed_search_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_ray_refine",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_ray_refine_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_image_evidence",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_image_evidence_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_regeneration_subpix",
                    report.baseline_result.runtime_breakdown
                        .round2_regeneration_subpix_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_measurement_build",
                    report.baseline_result.runtime_breakdown
                        .round2_measurement_build_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_residual_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round2_residual_evaluation_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_selection",
                    report.baseline_result.runtime_breakdown.round2_selection_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization",
                    report.baseline_result.runtime_breakdown.round2_optimization_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization_residual_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round2_optimization_residual_evaluation_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization_cost_evaluation",
                    report.baseline_result.runtime_breakdown
                        .round2_optimization_cost_evaluation_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization_frame_updates",
                    report.baseline_result.runtime_breakdown
                        .round2_optimization_frame_update_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization_board_updates",
                    report.baseline_result.runtime_breakdown
                        .round2_optimization_board_update_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "round2_optimization_intrinsics_updates",
                    report.baseline_result.runtime_breakdown
                        .round2_optimization_intrinsics_update_seconds,
                    !report.baseline_result.effective_options.run_second_pass);
    AddRuntimeStage(&runtime_summary, "holdout_dataset_build",
                    report.runtime_breakdown.holdout_dataset_build_seconds, false);
    AddRuntimeStage(&runtime_summary, "pre_backend_filter",
                    report.runtime_breakdown.pre_backend_filter_seconds,
                    requested_config.pre_backend_filter_mode ==
                        ati::PreBackendFilterMode::Off);
    AddRuntimeStage(&runtime_summary, "internal_blur_filter",
                    report.runtime_breakdown.internal_blur_filter_seconds,
                    requested_config.internal_blur_filter_mode ==
                        ati::InternalBlurFilterMode::Off);
    AddRuntimeStage(&runtime_summary, "diagnostic_compare",
                    report.runtime_breakdown.diagnostic_compare_seconds,
                    args.runtime_mode == ati::Stage5RuntimeMode::Fast);
    PrintProgress("writing Stage 5 benchmark artifacts...");

    if (report.baseline_result.stage5_round1_bundle.success) {
      ati::WriteCalibrationStateBundleSummary(
          (output_dir / "stage5_round1_bundle_summary.txt").string(),
          report.baseline_result.stage5_round1_bundle);
    }
    if (report.baseline_result.stage5_bundle_available) {
      ati::WriteCalibrationStateBundleSummary(
          (output_dir / "stage5_bundle_summary.txt").string(),
          report.baseline_result.final_stage5_bundle);
      ati::WriteCalibrationBackendProblemSummary(
          (output_dir / "stage5_backend_problem_summary.txt").string(),
          report.backend_problem_input);
    }
    ati::WriteAutoCameraInitializationSummary(
        (output_dir / "auto_camera_initialization_summary.txt").string(),
        report.baseline_result.auto_camera_initialization);
    ati::WriteAutoCameraInitializationCandidatesCsv(
        (output_dir / "auto_camera_initialization_candidates.csv").string(),
        report.baseline_result.auto_camera_initialization);
    ati::WriteAutoCameraInitializationOuterResidualsCsv(
        (output_dir / "auto_camera_initialization_outer_residuals.csv").string(),
        report.baseline_result.auto_camera_initialization);
    WriteExperimentConfigSummary(
        (output_dir / "experiment_config_summary.txt").string(),
        requested_config,
        report,
        nullptr);
    ati::WritePreBackendFilterSummary(
        (output_dir / "pre_backend_filter_summary.txt").string(),
        report.pre_backend_filter_result);
    ati::WritePreBackendFilterPointsCsv(
        (output_dir / "pre_backend_filter_points.csv").string(),
        report.pre_backend_filter_result);
    ati::WritePreBackendFilterBoardSummaryCsv(
        (output_dir / "pre_backend_filter_board_summary.csv").string(),
        report.pre_backend_filter_result);
    ati::WritePreBackendFilterFrameSummaryCsv(
        (output_dir / "pre_backend_filter_frame_summary.csv").string(),
        report.pre_backend_filter_result);
    ati::WriteInternalBlurFilterSummary(
        (output_dir / "internal_blur_filter_summary.txt").string(),
        report.internal_blur_filter_result);
    ati::WriteInternalBlurFilterBoardDecisionsCsv(
        (output_dir / "internal_blur_filter_board_decisions.csv").string(),
        report.internal_blur_filter_result);
    ati::WriteInternalBlurFilterPointDecisionsCsv(
        (output_dir / "internal_blur_filter_point_decisions.csv").string(),
        report.internal_blur_filter_result);
    ati::WriteInternalBlurFilterFrameSummaryCsv(
        (output_dir / "internal_blur_filter_frame_summary.csv").string(),
        report.internal_blur_filter_result);
    ati::WriteStage5BenchmarkProtocolSummary(
        (output_dir / "benchmark_protocol_summary.txt").string(), report);
    ati::WriteStage5BenchmarkTrainingSummary(
        (output_dir / "benchmark_training_summary.txt").string(), report);
    ati::WriteStage5BenchmarkHoldoutSummary(
        (output_dir / "benchmark_holdout_summary.txt").string(), report);
    ati::WriteCameraModelRefitPointsCsv(
        (output_dir / "benchmark_training_points.csv").string(),
        std::vector<ati::CameraModelRefitEvaluationResult>{
            report.our_training_evaluation,
            report.kalibr_training_evaluation});
    ati::WriteStage5BenchmarkHoldoutPointsCsv(
        (output_dir / "benchmark_holdout_points.csv").string(), report);
    ati::WriteStage5BenchmarkWorstCasesSummary(
        (output_dir / "benchmark_worst_cases_summary.txt").string(), report, 10);
    if (requested_config.internal_regeneration_diagnostics) {
      WriteInternalRegenerationDiagnostics(output_dir, report);
    }
    if (args.runtime_mode == ati::Stage5RuntimeMode::Research &&
        report.diagnostic_compare.success) {
      ati::WriteKalibrBenchmarkIntrinsicsCsv(
          (output_dir / "benchmark_intrinsics_compare.csv").string(),
          report.diagnostic_compare);
    }

    if (args.runtime_mode == ati::Stage5RuntimeMode::Research) {
      const cv::Mat projection_compare = benchmark.RenderProjectionComparison(report);
      if (!projection_compare.empty()) {
        cv::imwrite((output_dir / "benchmark_projection_compare.png").string(),
                    projection_compare);
        if (args.show) {
          cv::imshow("stage5_benchmark_projection_compare", projection_compare);
          cv::waitKey(0);
        }
      }
    }

    if (!report.success) {
      write_runtime_summary();
      std::cout << "Stage 5 benchmark success: 0\n"
                << "Protocol summary: "
                << (output_dir / "benchmark_protocol_summary.txt").string() << "\n";
      return 1;
    }
    PrintProgress("Stage 5 benchmark completed.");
    PrintProgress("protocol=" + requested_config.effective_protocol_label);
    PrintProgress("camera init mode=" +
                  std::string(ati::ToString(
                      report.baseline_result.auto_camera_initialization.selected_mode)) +
                  " source=" +
                  report.baseline_result.auto_camera_initialization.selected_source_label +
                  " fallback=" +
                  std::to_string(
                      report.baseline_result.auto_camera_initialization.fallback_used ? 1 : 0));
    PrintEvaluationProgress("frontend training", report.our_training_evaluation);
    PrintEvaluationProgress("frontend holdout", report.our_holdout_evaluation);
    PrintEvaluationProgress("kalibr training", report.kalibr_training_evaluation);
    PrintEvaluationProgress("kalibr holdout", report.kalibr_holdout_evaluation);

    if (args.runtime_mode == ati::Stage5RuntimeMode::Research) {
      ati::AslamBackendCalibrationOptions minimal_runner_options;
      minimal_runner_options.max_iterations = 6;
      minimal_runner_options.convergence_delta_j = 1e-3;
      minimal_runner_options.convergence_delta_x = 1e-4;
      minimal_runner_options.levenberg_marquardt_lambda_init = 1e-3;
      minimal_runner_options.linear_solver = "cholmod";
      minimal_runner_options.verbose = false;
      minimal_runner_options.use_huber_loss = true;
      minimal_runner_options.outer_huber_delta_pixels = 10.0;
      minimal_runner_options.internal_huber_delta_pixels = 6.0;
      minimal_runner_options.invalid_projection_penalty_pixels = 100.0;
      minimal_runner_options.export_cost_parity_diagnostics = true;
      minimal_runner_options.run_jacobian_consistency_check = true;
      minimal_runner_options.jacobian_finite_difference_epsilon = 1e-6;
      minimal_runner_options.debug_max_frames = 3;
      minimal_runner_options.debug_max_nonreference_boards = 1;
      minimal_runner_options.force_pose_only = true;
      const ati::AslamBackendCalibrationRunner minimal_backend_runner(
          minimal_runner_options);
      PrintProgress("running minimal pose-only backend smoke check...");
      const auto minimal_stage_start = std::chrono::steady_clock::now();
      const ati::AslamBackendCalibrationResult minimal_backend_result =
          minimal_backend_runner.Run(report.backend_problem_input);
      AddRuntimeStage(&runtime_summary, "backend_minimal_smoke",
                      ElapsedSeconds(minimal_stage_start), false);
      PrintBackendResultProgress(minimal_backend_result);
      WriteBackendDiagnosticArtifacts(
          output_dir, "backend_minimal_pose_only", minimal_backend_result);

      if (!minimal_backend_result.success || !HasBackendCostDescent(minimal_backend_result)) {
        write_runtime_summary();
        std::cout << "Minimal pose-only backend summary: "
                  << (output_dir / "backend_minimal_pose_only_summary.txt").string() << "\n";
        return 1;
      }
    } else {
      AddRuntimeStage(&runtime_summary, "backend_minimal_smoke", 0.0, true);
    }

    ati::AslamBackendCalibrationOptions runner_options;
    runner_options.max_iterations = 12;
    runner_options.convergence_delta_j = 1e-3;
    runner_options.convergence_delta_x = 1e-4;
    runner_options.levenberg_marquardt_lambda_init = 1e-3;
    runner_options.linear_solver = "cholmod";
    runner_options.verbose = false;
    runner_options.use_huber_loss = true;
    runner_options.outer_huber_delta_pixels = 10.0;
    runner_options.internal_huber_delta_pixels = 6.0;
    runner_options.invalid_projection_penalty_pixels = 100.0;
    runner_options.export_cost_parity_diagnostics =
        args.runtime_mode == ati::Stage5RuntimeMode::Research;
    const ati::AslamBackendCalibrationRunner backend_runner(runner_options);
    PrintProgress("running full backend optimization...");
    const auto backend_stage_start = std::chrono::steady_clock::now();
    const ati::AslamBackendCalibrationResult backend_result =
        backend_runner.Run(report.backend_problem_input);
    AddRuntimeStage(&runtime_summary, "backend_full_optimization",
                    ElapsedSeconds(backend_stage_start), false);
    PrintBackendResultProgress(backend_result);
    WriteBackendDiagnosticArtifacts(output_dir, "backend_optimization", backend_result);
    WriteExperimentConfigSummary(
        (output_dir / "experiment_config_summary.txt").string(),
        requested_config,
        report,
        &backend_result);

    if (!backend_result.success) {
      write_runtime_summary();
      std::cout << "Backend summary: "
                << (output_dir / "backend_optimization_summary.txt").string() << "\n";
      return 1;
    }

    ati::CameraModelRefitEvaluationResult backend_training_evaluation;
    ati::CameraModelRefitEvaluationResult backend_holdout_evaluation;
    {
      const auto stage_start = std::chrono::steady_clock::now();
      backend_training_evaluation = benchmark.EvaluateCameraModel(
          report.training_dataset,
          backend_result.optimized_scene_state.camera,
          "backend");
      AddRuntimeStage(&runtime_summary, "backend_training_evaluation",
                      ElapsedSeconds(stage_start), false);
    }
    {
      const auto stage_start = std::chrono::steady_clock::now();
      backend_holdout_evaluation = benchmark.EvaluateCameraModel(
          report.holdout_dataset,
          backend_result.optimized_scene_state.camera,
          "backend");
      AddRuntimeStage(&runtime_summary, "backend_holdout_evaluation",
                      ElapsedSeconds(stage_start), false);
    }
    if (!backend_training_evaluation.success || !backend_holdout_evaluation.success) {
      std::ofstream output((output_dir / "backend_vs_frontend_summary.txt").string().c_str());
      output << "backend_evaluation_failed: 1\n";
      output << "training_failure_reason: " << backend_training_evaluation.failure_reason << "\n";
      output << "holdout_failure_reason: " << backend_holdout_evaluation.failure_reason << "\n";
      write_runtime_summary();
      return 1;
    }
    PrintEvaluationProgress("backend training", backend_training_evaluation);
    PrintEvaluationProgress("backend holdout", backend_holdout_evaluation);

    WriteEvaluationSummary(
        (output_dir / "backend_training_summary.txt").string(),
        backend_training_evaluation);
    WriteEvaluationSummary(
        (output_dir / "backend_holdout_summary.txt").string(),
        backend_holdout_evaluation);
    ati::WriteCameraModelRefitPointsCsv(
        (output_dir / "backend_training_points.csv").string(),
        std::vector<ati::CameraModelRefitEvaluationResult>{
            backend_training_evaluation});
    ati::WriteCameraModelRefitPointsCsv(
        (output_dir / "backend_holdout_points.csv").string(),
        std::vector<ati::CameraModelRefitEvaluationResult>{
            backend_holdout_evaluation});
    if (requested_config.internal_blur_diagnostics) {
      WriteInternalBlurDiagnostics(
          output_dir, report, backend_training_evaluation,
          backend_holdout_evaluation, all_frames);
    }
    WriteBackendComparisonSummary(
        (output_dir / "backend_vs_frontend_summary.txt").string(),
        backend_result,
        report.our_training_evaluation,
        backend_training_evaluation,
        report.kalibr_training_evaluation,
        report.our_holdout_evaluation,
        backend_holdout_evaluation,
        report.kalibr_holdout_evaluation);

    const fs::path compare_frame_dir = output_dir / "backend_compare_holdout_frames";
    const fs::path compare_board_dir = output_dir / "backend_compare_holdout_boards";
    const fs::path compare_outer_frame_dir =
        output_dir / "backend_compare_outer_pose_frames";
    const fs::path compare_outer_board_dir =
        output_dir / "backend_compare_outer_pose_boards";
    if (args.runtime_mode == ati::Stage5RuntimeMode::Research) {
      const auto overlay_stage_start = std::chrono::steady_clock::now();
      PrintProgress("rendering backend/frontend/Kalibr comparison overlays...");
      EnsureDirectoryExists(compare_frame_dir);
      EnsureDirectoryExists(compare_board_dir);
      EnsureDirectoryExists(compare_outer_frame_dir);
      EnsureDirectoryExists(compare_outer_board_dir);

      {
        std::vector<ati::CameraModelRefitFrameDiagnostics> worst_frames =
            report.our_holdout_evaluation.frame_diagnostics;
        std::sort(worst_frames.begin(), worst_frames.end(),
                  [](const ati::CameraModelRefitFrameDiagnostics& lhs,
                     const ati::CameraModelRefitFrameDiagnostics& rhs) {
                    return lhs.rmse > rhs.rmse;
                  });
        if (worst_frames.size() > 10) {
          worst_frames.resize(10);
        }
        for (std::size_t rank = 0; rank < worst_frames.size(); ++rank) {
          const ati::CameraModelRefitFrameDiagnostics& frame = worst_frames[rank];
          const cv::Mat frontend_overlay = benchmark.RenderEvaluationFrameOverlay(
              report, report.our_holdout_evaluation, frame.frame_index);
          const cv::Mat backend_overlay = benchmark.RenderEvaluationFrameOverlay(
              report, backend_holdout_evaluation, frame.frame_index);
          const cv::Mat kalibr_overlay = benchmark.RenderEvaluationFrameOverlay(
              report, report.kalibr_holdout_evaluation, frame.frame_index);
          const cv::Mat compare = RenderLabeledCompare({
              std::make_pair(frontend_overlay, "frontend"),
              std::make_pair(backend_overlay, "backend"),
              std::make_pair(kalibr_overlay, "kalibr"),
          });
          if (!compare.empty()) {
            const std::string filename =
                "rank" + std::to_string(static_cast<int>(rank + 1)) +
                "_frame_" + std::to_string(frame.frame_index) + "_" + frame.frame_label + ".png";
            cv::imwrite((compare_frame_dir / filename).string(), compare);
          }
        }
      }

      {
        std::vector<ati::CameraModelRefitFrameDiagnostics> worst_frames =
            report.our_holdout_evaluation.frame_diagnostics;
        std::sort(worst_frames.begin(), worst_frames.end(),
                  [](const ati::CameraModelRefitFrameDiagnostics& lhs,
                     const ati::CameraModelRefitFrameDiagnostics& rhs) {
                    return lhs.rmse > rhs.rmse;
                  });
        if (worst_frames.size() > 10) {
          worst_frames.resize(10);
        }
        for (std::size_t rank = 0; rank < worst_frames.size(); ++rank) {
          const ati::CameraModelRefitFrameDiagnostics& frame = worst_frames[rank];
          const cv::Mat frontend_overlay = benchmark.RenderOuterPoseFitFrameOverlay(
              report, report.our_holdout_evaluation, frame.frame_index);
          const cv::Mat backend_overlay = benchmark.RenderOuterPoseFitFrameOverlay(
              report, backend_holdout_evaluation, frame.frame_index);
          const cv::Mat kalibr_overlay = benchmark.RenderOuterPoseFitFrameOverlay(
              report, report.kalibr_holdout_evaluation, frame.frame_index);
          const cv::Mat compare = RenderLabeledCompare({
              std::make_pair(frontend_overlay, "frontend outer pose"),
              std::make_pair(backend_overlay, "backend outer pose"),
              std::make_pair(kalibr_overlay, "kalibr outer pose"),
          });
          if (!compare.empty()) {
            const std::string filename =
                "rank" + std::to_string(static_cast<int>(rank + 1)) +
                "_frame_" + std::to_string(frame.frame_index) + "_" + frame.frame_label + ".png";
            cv::imwrite((compare_outer_frame_dir / filename).string(), compare);
          }
        }
      }

      {
        std::vector<ati::CameraModelRefitBoardObservationDiagnostics> worst_boards =
            report.our_holdout_evaluation.board_observation_diagnostics;
        std::sort(worst_boards.begin(), worst_boards.end(),
                  [](const ati::CameraModelRefitBoardObservationDiagnostics& lhs,
                     const ati::CameraModelRefitBoardObservationDiagnostics& rhs) {
                    return lhs.evaluation_rmse > rhs.evaluation_rmse;
                  });
        if (worst_boards.size() > 10) {
          worst_boards.resize(10);
        }
        for (std::size_t rank = 0; rank < worst_boards.size(); ++rank) {
          const ati::CameraModelRefitBoardObservationDiagnostics& board = worst_boards[rank];
          const cv::Mat frontend_overlay = benchmark.RenderEvaluationBoardObservationOverlay(
              report, report.our_holdout_evaluation, board.frame_index, board.board_id);
          const cv::Mat backend_overlay = benchmark.RenderEvaluationBoardObservationOverlay(
              report, backend_holdout_evaluation, board.frame_index, board.board_id);
          const cv::Mat kalibr_overlay = benchmark.RenderEvaluationBoardObservationOverlay(
              report, report.kalibr_holdout_evaluation, board.frame_index, board.board_id);
          const cv::Mat compare = RenderLabeledCompare({
              std::make_pair(frontend_overlay, "frontend"),
              std::make_pair(backend_overlay, "backend"),
              std::make_pair(kalibr_overlay, "kalibr"),
          });
          if (!compare.empty()) {
            const std::string filename =
                "rank" + std::to_string(static_cast<int>(rank + 1)) +
                "_frame_" + std::to_string(board.frame_index) +
                "_board_" + std::to_string(board.board_id) + ".png";
            cv::imwrite((compare_board_dir / filename).string(), compare);
          }
        }
      }

      {
        std::vector<ati::CameraModelRefitBoardObservationDiagnostics> worst_boards =
            report.our_holdout_evaluation.board_observation_diagnostics;
        std::sort(worst_boards.begin(), worst_boards.end(),
                  [](const ati::CameraModelRefitBoardObservationDiagnostics& lhs,
                     const ati::CameraModelRefitBoardObservationDiagnostics& rhs) {
                    return lhs.evaluation_rmse > rhs.evaluation_rmse;
                  });
        if (worst_boards.size() > 10) {
          worst_boards.resize(10);
        }
        for (std::size_t rank = 0; rank < worst_boards.size(); ++rank) {
          const ati::CameraModelRefitBoardObservationDiagnostics& board = worst_boards[rank];
          const cv::Mat frontend_overlay = benchmark.RenderOuterPoseFitBoardOverlay(
              report, report.our_holdout_evaluation, board.frame_index, board.board_id);
          const cv::Mat backend_overlay = benchmark.RenderOuterPoseFitBoardOverlay(
              report, backend_holdout_evaluation, board.frame_index, board.board_id);
          const cv::Mat kalibr_overlay = benchmark.RenderOuterPoseFitBoardOverlay(
              report, report.kalibr_holdout_evaluation, board.frame_index, board.board_id);
          const cv::Mat compare = RenderLabeledCompare({
              std::make_pair(frontend_overlay, "frontend outer pose"),
              std::make_pair(backend_overlay, "backend outer pose"),
              std::make_pair(kalibr_overlay, "kalibr outer pose"),
          });
          if (!compare.empty()) {
            const std::string filename =
                "rank" + std::to_string(static_cast<int>(rank + 1)) +
                "_frame_" + std::to_string(board.frame_index) +
                "_board_" + std::to_string(board.board_id) + ".png";
            cv::imwrite((compare_outer_board_dir / filename).string(), compare);
          }
        }
      }
      AddRuntimeStage(&runtime_summary, "overlay_export",
                      ElapsedSeconds(overlay_stage_start), false);
    } else {
      AddRuntimeStage(&runtime_summary, "overlay_export", 0.0, true);
    }

    write_runtime_summary();
    std::cout << "Stage 5 benchmark summary: "
              << (output_dir / "benchmark_protocol_summary.txt").string() << "\n"
              << "Backend optimization summary: "
              << (output_dir / "backend_optimization_summary.txt").string() << "\n"
              << "Backend training summary: "
              << (output_dir / "backend_training_summary.txt").string() << "\n"
              << "Backend holdout summary: "
              << (output_dir / "backend_holdout_summary.txt").string() << "\n"
              << "Backend vs frontend summary: "
              << (output_dir / "backend_vs_frontend_summary.txt").string() << "\n"
              << "Runtime summary: "
              << (output_dir / "runtime_summary.txt").string() << "\n";
    if (args.runtime_mode == ati::Stage5RuntimeMode::Research) {
      std::cout << "Backend compare frame overlays: " << compare_frame_dir.string() << "\n"
                << "Backend compare board overlays: " << compare_board_dir.string() << "\n"
                << "Backend compare outer-pose frames: "
                << compare_outer_frame_dir.string() << "\n"
                << "Backend compare outer-pose boards: "
                << compare_outer_board_dir.string() << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
