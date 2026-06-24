#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
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
  std::string intrinsics_yaml;
  std::vector<std::string> additional_intrinsics_yamls;
  std::string detector_intrinsics_yaml;
  std::string output_path;
  std::string benchmark_label = "benchmark";
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
      << " [--benchmark-label LABEL] [--all]\n"
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
      const std::string yaml = argv[++i];
      if (args.intrinsics_yaml.empty()) {
        args.intrinsics_yaml = yaml;
      } else {
        args.additional_intrinsics_yamls.push_back(yaml);
      }
    } else if (token == "--detector-intrinsics-yaml" && i + 1 < argc) {
      args.detector_intrinsics_yaml = argv[++i];
    } else if (token == "--output" && i + 1 < argc) {
      args.output_path = argv[++i];
    } else if (token == "--benchmark-label" && i + 1 < argc) {
      args.benchmark_label = argv[++i];
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
      args.intrinsics_yaml.empty() || args.output_path.empty()) {
    throw std::runtime_error(
        "--image, --config, --intrinsics-yaml and --output are required.");
  }
  if (args.detector_intrinsics_yaml.empty()) {
    args.detector_intrinsics_yaml = args.intrinsics_yaml;
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
  out << "image_path: " << args.image_path << "\n";
  out << "config_path: " << args.config_path << "\n";
  out << "intrinsics_yaml: " << args.intrinsics_yaml << "\n";
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

    std::vector<std::string> evaluation_intrinsics_yamls;
    evaluation_intrinsics_yamls.push_back(args.intrinsics_yaml);
    evaluation_intrinsics_yamls.insert(evaluation_intrinsics_yamls.end(),
                                       args.additional_intrinsics_yamls.begin(),
                                       args.additional_intrinsics_yamls.end());

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

    const ati::ApriltagInternalDetectionOptions detection_options =
        MakeDetectionOptions(config, args);
    const ati::ApriltagInternalDetector detector(config, detection_options);
    std::vector<std::string> image_paths =
        CollectImagePaths(args.image_path, args.all);
    if (args.max_images > 0 &&
        static_cast<int>(image_paths.size()) > args.max_images) {
      image_paths.resize(static_cast<std::size_t>(args.max_images));
    }

    std::cout << "[benchmark] images=" << image_paths.size()
              << " label=" << args.benchmark_label
              << " intrinsics_count=" << evaluation_intrinsics_yamls.size()
              << " detector_intrinsics=" << args.detector_intrinsics_yaml
              << std::endl;
    FrontendRuntimeStats frontend_stats;
    const ati::CalibrationEvaluationDataset dataset =
        BuildEvaluationDataset(image_paths, detector, args.benchmark_label,
                               args.progress_every, &frontend_stats);
    if (!dataset.success) {
      throw std::runtime_error(dataset.failure_reason);
    }

    const ati::Stage5Benchmark benchmark;
    const fs::path output_dir(args.output_path);
    for (std::size_t model_index = 0;
         model_index < evaluation_intrinsics_yamls.size(); ++model_index) {
      ati::OuterBootstrapCameraIntrinsics intrinsics;
      std::string intrinsics_error;
      const std::string& yaml_path = evaluation_intrinsics_yamls[model_index];
      if (!ati::LoadKalibrCamchainIntrinsics(yaml_path, &intrinsics,
                                             &intrinsics_error)) {
        throw std::runtime_error(intrinsics_error);
      }
      CmdArgs per_model_args = args;
      per_model_args.intrinsics_yaml = yaml_path;
      if (evaluation_intrinsics_yamls.size() > 1u) {
        per_model_args.benchmark_label =
            fs::path(yaml_path).stem().string();
      }
      const ati::CameraModelRefitEvaluationResult evaluation =
          benchmark.EvaluateCameraModel(dataset, intrinsics,
                                        per_model_args.benchmark_label);
      if (!evaluation.success) {
        throw std::runtime_error(evaluation.failure_reason);
      }

      fs::path model_output_dir = output_dir;
      if (evaluation_intrinsics_yamls.size() > 1u) {
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
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << std::endl;
    return 1;
  }
}
