#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

namespace ati = aslam::cameras::apriltag_internal;
namespace fs = boost::filesystem;

struct CmdArgs {
  std::string config_path;
  std::string image_path;
  std::string output_path;
  bool all = false;
  bool show = false;
  bool save_overlays = true;
  int reference_board_id = 1;
};

struct FrameRecord {
  std::string image_path;
  ati::InternalRegenerationFrameInput regeneration_input;
  ati::OuterBootstrapFrameInput bootstrap_input;
};

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --image IMAGE_OR_DIR --config APRILTAG_INTERNAL_YAML --output OUTPUT_DIR"
      << " [--all] [--show] [--no-save-overlays] [--reference-board-id ID]\n";
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
    } else if (token == "--all") {
      args.all = true;
    } else if (token == "--show") {
      args.show = true;
    } else if (token == "--no-save-overlays") {
      args.save_overlays = false;
    } else if (token == "--reference-board-id" && i + 1 < argc) {
      args.reference_board_id = std::stoi(argv[++i]);
    } else if (token == "--help" || token == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + token);
    }
  }

  if (args.image_path.empty() || args.config_path.empty() || args.output_path.empty()) {
    throw std::runtime_error("--image, --config and --output are required.");
  }
  return args;
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

ati::OuterBootstrapOptions MakeBootstrapOptions(const ati::ApriltagInternalConfig& config,
                                                const CmdArgs& args) {
  ati::OuterBootstrapOptions options;
  options.reference_board_id = args.reference_board_id;
  if (config.intermediate_camera.IsConfigured() &&
      config.intermediate_camera.camera_model == "ds" &&
      config.intermediate_camera.intrinsics.size() == 6 &&
      config.intermediate_camera.resolution.size() == 2 &&
      config.intermediate_camera.resolution[0] > 0 &&
      config.intermediate_camera.resolution[1] > 0) {
    options.init_xi = config.intermediate_camera.intrinsics[0];
    options.init_alpha = config.intermediate_camera.intrinsics[1];
    options.init_fu_scale =
        config.intermediate_camera.intrinsics[2] /
        static_cast<double>(config.intermediate_camera.resolution[0]);
    options.init_fv_scale =
        config.intermediate_camera.intrinsics[3] /
        static_cast<double>(config.intermediate_camera.resolution[1]);
    options.init_cu_offset =
        config.intermediate_camera.intrinsics[4] -
        0.5 * static_cast<double>(config.intermediate_camera.resolution[0]);
    options.init_cv_offset =
        config.intermediate_camera.intrinsics[5] -
        0.5 * static_cast<double>(config.intermediate_camera.resolution[1]);
  } else {
    options.init_xi = config.sphere_lattice_init_xi;
    options.init_alpha = config.sphere_lattice_init_alpha;
    options.init_fu_scale = config.sphere_lattice_init_fu_scale;
    options.init_fv_scale = config.sphere_lattice_init_fv_scale;
    options.init_cu_offset = config.sphere_lattice_init_cu_offset;
    options.init_cv_offset = config.sphere_lattice_init_cv_offset;
  }
  options.min_detection_quality = config.outer_detector_config.min_detection_quality;
  return options;
}

std::vector<int> NormalizeBoardIds(const std::vector<int>& configured_ids, int fallback_tag_id) {
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

ati::ApriltagInternalDetectionOptions MakeDetectionOptions(const ati::ApriltagInternalConfig& config) {
  ati::ApriltagInternalDetectionOptions options;
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

void WriteBootstrapSummary(const fs::path& output_path,
                           const ati::OuterBootstrapResult& result) {
  std::ofstream output(output_path.string());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "used_frame_count: " << result.used_frame_count << "\n";
  output << "used_board_observation_count: " << result.used_board_observation_count << "\n";
  output << "used_corner_count: " << result.used_corner_count << "\n";
  output << "global_rmse: " << result.global_rmse << "\n";
  output << "coarse_camera: [" << result.coarse_camera.xi << ", " << result.coarse_camera.alpha
         << ", " << result.coarse_camera.fu << ", " << result.coarse_camera.fv << ", "
         << result.coarse_camera.cu << ", " << result.coarse_camera.cv << "]\n";
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteRegenerationSummary(
    const fs::path& output_path,
    const ati::OuterBootstrapResult& bootstrap_result,
    const std::vector<ati::InternalRegenerationFrameResult>& frame_results) {
  int successful_frames = 0;
  int successful_boards = 0;
  int valid_internal_corners = 0;
  for (const ati::InternalRegenerationFrameResult& frame_result : frame_results) {
    successful_frames += frame_result.SuccessfulBoardCount() > 0 ? 1 : 0;
    successful_boards += frame_result.SuccessfulBoardCount();
    valid_internal_corners += frame_result.ValidInternalCornerCount();
  }

  std::ofstream output(output_path.string());
  output << "bootstrap_success: " << (bootstrap_result.success ? 1 : 0) << "\n";
  output << "bootstrap_global_rmse: " << bootstrap_result.global_rmse << "\n";
  output << "frame_count: " << frame_results.size() << "\n";
  output << "successful_frame_count: " << successful_frames << "\n";
  output << "successful_board_count: " << successful_boards << "\n";
  output << "valid_internal_corner_count: " << valid_internal_corners << "\n";
  for (const ati::InternalRegenerationFrameResult& frame_result : frame_results) {
    output << "frame " << frame_result.frame_index << " " << frame_result.frame_label
           << " successful_boards=" << frame_result.SuccessfulBoardCount()
           << " valid_internal=" << frame_result.ValidInternalCornerCount() << "\n";
  }
}

void WriteRegenerationCsv(
    const fs::path& output_path,
    const std::vector<ati::InternalRegenerationFrameResult>& frame_results) {
  std::ofstream output(output_path.string());
  output << "frame_index,frame_label,board_id,frame_bootstrap_initialized,"
         << "board_bootstrap_initialized,pose_prior_used,tag_detected,success,"
         << "valid_outer_corners,valid_internal_corners\n";

  for (const ati::InternalRegenerationFrameResult& frame_result : frame_results) {
    for (const ati::RegeneratedBoardMeasurement& measurement : frame_result.board_measurements) {
      output << frame_result.frame_index << ","
             << frame_result.frame_label << ","
             << measurement.board_id << ","
             << (measurement.frame_bootstrap_initialized ? 1 : 0) << ","
             << (measurement.board_bootstrap_initialized ? 1 : 0) << ","
             << (measurement.pose_prior_used ? 1 : 0) << ","
             << (measurement.detection.tag_detected ? 1 : 0) << ","
             << (measurement.detection.success ? 1 : 0) << ","
             << measurement.detection.valid_corner_count << ","
             << measurement.detection.valid_internal_corner_count << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CmdArgs args = ParseArgs(argc, argv);
    ati::ApriltagInternalConfig config =
        ati::ApriltagInternalDetector::LoadConfig(args.config_path);
    config.tag_ids = NormalizeBoardIds(config.tag_ids, config.tag_id);
    config.tag_id = config.tag_ids.front();
    config.outer_detector_config.tag_ids = config.tag_ids;
    config.outer_detector_config.tag_id = config.tag_id;
    const ati::ApriltagInternalDetectionOptions detection_options = MakeDetectionOptions(config);
    const ati::MultiScaleOuterTagDetector outer_detector(config.outer_detector_config);
    const ati::OuterBootstrapOptions bootstrap_options = MakeBootstrapOptions(config, args);
    const ati::MultiBoardOuterBootstrap bootstrap(config, bootstrap_options);
    const ati::MultiBoardInternalMeasurementRegenerator regenerator(config, detection_options);

    const std::vector<std::string> image_paths = CollectImagePaths(args.image_path, args.all);
    const fs::path output_dir(args.output_path);
    EnsureDirectoryExists(output_dir);
    const fs::path overlay_dir = output_dir / "internal_regeneration_overlays";
    if (args.save_overlays) {
      EnsureDirectoryExists(overlay_dir);
    }

    std::vector<FrameRecord> frames;
    frames.reserve(image_paths.size());
    std::vector<ati::OuterBootstrapFrameInput> bootstrap_frames;
    bootstrap_frames.reserve(image_paths.size());

    std::cout << "Detecting multi-board outer measurements on " << image_paths.size()
              << " image(s)...\n";
    for (std::size_t image_index = 0; image_index < image_paths.size(); ++image_index) {
      const std::string& image_path = image_paths[image_index];
      const fs::path path(image_path);
      const std::string frame_label = path.stem().string();
      const cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
      if (image.empty()) {
        throw std::runtime_error("Failed to read image: " + image_path);
      }

      const ati::OuterTagMultiDetectionResult outer_detections = outer_detector.DetectMultiple(image);

      FrameRecord frame;
      frame.image_path = image_path;
      frame.regeneration_input.frame_index = static_cast<int>(image_index);
      frame.regeneration_input.frame_label = frame_label;
      frame.regeneration_input.outer_detections = outer_detections;
      frame.bootstrap_input.frame_index = frame.regeneration_input.frame_index;
      frame.bootstrap_input.frame_label = frame_label;
      frame.bootstrap_input.measurements = outer_detections.frame_measurements;
      bootstrap_frames.push_back(frame.bootstrap_input);
      frames.push_back(frame);

      std::cout << "  [" << (image_index + 1) << "/" << image_paths.size() << "] "
                << frame_label << " success_boards="
                << outer_detections.SuccessfulBoardCount() << "\n";
    }

    const ati::OuterBootstrapResult bootstrap_result = bootstrap.Solve(bootstrap_frames);
    WriteBootstrapSummary(output_dir / "bootstrap_summary.txt", bootstrap_result);
    if (!bootstrap_result.success) {
      std::cout << "Bootstrap failed. Summary written to: "
                << (output_dir / "bootstrap_summary.txt").string() << "\n";
      return 1;
    }

    std::vector<ati::InternalRegenerationFrameResult> frame_results;
    frame_results.reserve(frames.size());
    std::cout << "Regenerating multi-board internal measurements...\n";
    for (std::size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
      const cv::Mat image = cv::imread(frames[frame_index].image_path, cv::IMREAD_UNCHANGED);
      if (image.empty()) {
        throw std::runtime_error("Failed to read image: " + frames[frame_index].image_path);
      }

      const ati::InternalRegenerationFrameResult frame_result =
          regenerator.RegenerateFrame(image, frames[frame_index].regeneration_input, bootstrap_result);
      frame_results.push_back(frame_result);

      if (args.save_overlays || args.show) {
        cv::Mat overlay;
        regenerator.DrawFrameOverlay(image, frame_result, &overlay);
        if (args.save_overlays) {
          const fs::path overlay_path =
              overlay_dir / (frames[frame_index].regeneration_input.frame_label +
                             "_internal_regeneration_overlay.png");
          cv::imwrite(overlay_path.string(), overlay);
        }
        if (args.show) {
          cv::imshow("multi_board_internal_regeneration", overlay);
          cv::waitKey(1);
        }
      }

      std::cout << "  regenerated frame " << frame_result.frame_index
                << " successful_boards=" << frame_result.SuccessfulBoardCount()
                << " valid_internal=" << frame_result.ValidInternalCornerCount() << "\n";
    }

    WriteRegenerationSummary(output_dir / "internal_regeneration_summary.txt",
                             bootstrap_result, frame_results);
    WriteRegenerationCsv(output_dir / "internal_regeneration_measurements.csv", frame_results);

    std::cout << "\nBootstrap RMSE: " << bootstrap_result.global_rmse << "\n"
              << "Internal regeneration summary: "
              << (output_dir / "internal_regeneration_summary.txt").string() << "\n"
              << "Internal regeneration CSV: "
              << (output_dir / "internal_regeneration_measurements.csv").string() << "\n";
    if (args.save_overlays) {
      std::cout << "Overlays: " << overlay_dir.string() << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
