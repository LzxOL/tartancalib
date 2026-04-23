#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

#include <algorithm>
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
#include <utility>
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
  bool save_overlays = false;
  bool force_all_frames = false;
  bool no_post_filter = false;
  int reference_board_id = 1;
  int min_initial_views_per_board = 5;
  int coverage_grid_cols = 4;
  int coverage_grid_rows = 4;
  double post_filter_sigma = 4.0;
  int min_used_observations_for_post_filter = 20;
};

struct FrameCandidate {
  std::string image_path;
  ati::OuterBootstrapFrameInput frame_input;
};

struct DatasetConnectivity {
  bool reference_board_observed = false;
  std::set<int> reference_connected_boards;
};

struct FrameSelectionDecision {
  std::string image_path;
  int frame_index = -1;
  std::string frame_label;
  bool accepted = false;
  std::string reason;
  int usable_board_count = 0;
  std::vector<int> usable_board_ids;
};

struct ResidualStatistics {
  int used_corner_count = 0;
  double mean_x = 0.0;
  double mean_y = 0.0;
  double sigma_x = 0.0;
  double sigma_y = 0.0;
};

std::string JoinInts(const std::vector<int>& values) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << values[index];
  }
  return stream.str();
}

std::string JoinStrings(const std::vector<std::string>& values) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << values[index];
  }
  return stream.str();
}

std::string MatrixToString(const Eigen::Matrix4d& matrix) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(6);
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      if (col > 0) {
        stream << " ";
      }
      stream << matrix(row, col);
    }
    if (row + 1 < 4) {
      stream << "\n";
    }
  }
  return stream.str();
}

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --image IMAGE_OR_DIR --config APRILTAG_INTERNAL_YAML --output OUTPUT_DIR"
      << " [--all] [--show] [--save-overlays] [--reference-board-id ID]"
      << " [--force-all-frames] [--no-post-filter]\n\n"
      << "Example:\n"
      << "  " << program
      << " --image /data/images --all"
      << " --config ./config/example_apriltag_internal.yaml"
      << " --output /tmp/multi_board_outer_bootstrap\n";
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
    } else if (token == "--save-overlays") {
      args.save_overlays = true;
    } else if (token == "--reference-board-id" && i + 1 < argc) {
      args.reference_board_id = std::stoi(argv[++i]);
    } else if (token == "--min-initial-views-per-board" && i + 1 < argc) {
      args.min_initial_views_per_board = std::stoi(argv[++i]);
    } else if (token == "--coverage-grid-cols" && i + 1 < argc) {
      args.coverage_grid_cols = std::stoi(argv[++i]);
    } else if (token == "--coverage-grid-rows" && i + 1 < argc) {
      args.coverage_grid_rows = std::stoi(argv[++i]);
    } else if (token == "--post-filter-sigma" && i + 1 < argc) {
      args.post_filter_sigma = std::stod(argv[++i]);
    } else if (token == "--min-used-observations-for-post-filter" && i + 1 < argc) {
      args.min_used_observations_for_post_filter = std::stoi(argv[++i]);
    } else if (token == "--force-all-frames") {
      args.force_all_frames = true;
    } else if (token == "--no-post-filter") {
      args.no_post_filter = true;
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
  if (args.reference_board_id < 0) {
    throw std::runtime_error("--reference-board-id must be non-negative.");
  }
  if (args.min_initial_views_per_board < 1) {
    throw std::runtime_error("--min-initial-views-per-board must be positive.");
  }
  if (args.coverage_grid_cols < 1 || args.coverage_grid_rows < 1) {
    throw std::runtime_error("--coverage-grid-cols and --coverage-grid-rows must be positive.");
  }
  if (args.post_filter_sigma <= 0.0) {
    throw std::runtime_error("--post-filter-sigma must be positive.");
  }
  if (args.min_used_observations_for_post_filter < 1) {
    throw std::runtime_error("--min-used-observations-for-post-filter must be positive.");
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
  for (std::size_t index = 0; index < configured_ids.size(); ++index) {
    append_if_valid(configured_ids[index]);
  }
  if (board_ids.empty()) {
    append_if_valid(fallback_tag_id);
  }
  return board_ids;
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

bool IsUsableBoardMeasurement(const ati::OuterBoardMeasurement& measurement,
                              const ati::OuterBootstrapOptions& options) {
  return measurement.success &&
         measurement.valid_refined_corner_count == 4 &&
         std::all_of(measurement.refined_corner_valid.begin(),
                     measurement.refined_corner_valid.end(),
                     [](bool valid) { return valid; }) &&
         measurement.detection_quality >= options.min_detection_quality;
}

std::vector<int> GetUsableBoardIds(const ati::OuterBootstrapFrameInput& frame_input,
                                   const ati::OuterBootstrapOptions& options) {
  std::vector<int> board_ids;
  for (std::size_t index = 0; index < frame_input.measurements.board_measurements.size(); ++index) {
    const ati::OuterBoardMeasurement& measurement =
        frame_input.measurements.board_measurements[index];
    if (IsUsableBoardMeasurement(measurement, options) &&
        std::find(board_ids.begin(), board_ids.end(), measurement.board_id) == board_ids.end()) {
      board_ids.push_back(measurement.board_id);
    }
  }
  std::sort(board_ids.begin(), board_ids.end());
  return board_ids;
}

DatasetConnectivity AnalyzeDatasetConnectivity(const std::vector<FrameCandidate>& candidates,
                                               const ati::OuterBootstrapOptions& options) {
  DatasetConnectivity connectivity;
  std::map<int, std::set<int> > adjacency;
  std::set<int> usable_boards;

  for (std::size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const std::vector<int> usable_board_ids =
        GetUsableBoardIds(candidates[candidate_index].frame_input, options);
    for (std::size_t board_index = 0; board_index < usable_board_ids.size(); ++board_index) {
      const int board_id = usable_board_ids[board_index];
      usable_boards.insert(board_id);
      adjacency[board_id];
      if (board_id == options.reference_board_id) {
        connectivity.reference_board_observed = true;
      }
    }
    for (std::size_t first = 0; first < usable_board_ids.size(); ++first) {
      for (std::size_t second = first + 1; second < usable_board_ids.size(); ++second) {
        adjacency[usable_board_ids[first]].insert(usable_board_ids[second]);
        adjacency[usable_board_ids[second]].insert(usable_board_ids[first]);
      }
    }
  }

  if (!connectivity.reference_board_observed) {
    connectivity.reference_connected_boards = usable_boards;
    return connectivity;
  }

  std::vector<int> queue(1, options.reference_board_id);
  connectivity.reference_connected_boards.insert(options.reference_board_id);
  for (std::size_t offset = 0; offset < queue.size(); ++offset) {
    const int board_id = queue[offset];
    const std::set<int>& neighbors = adjacency[board_id];
    for (std::set<int>::const_iterator it = neighbors.begin(); it != neighbors.end(); ++it) {
      if (connectivity.reference_connected_boards.insert(*it).second) {
        queue.push_back(*it);
      }
    }
  }
  return connectivity;
}

int ScaleBinForMeasurement(const ati::OuterBoardMeasurement& measurement,
                           const cv::Size& image_size) {
  double average_edge_length = 0.0;
  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    const Eigen::Vector2d& first =
        measurement.refined_outer_corners_original_image[static_cast<std::size_t>(corner_index)];
    const Eigen::Vector2d& second =
        measurement.refined_outer_corners_original_image[static_cast<std::size_t>((corner_index + 1) % 4)];
    average_edge_length += (first - second).norm();
  }
  average_edge_length /= 4.0;
  const double normalized_scale =
      average_edge_length / static_cast<double>(std::max(image_size.width, image_size.height));
  if (normalized_scale < 0.04) {
    return 0;
  }
  if (normalized_scale < 0.08) {
    return 1;
  }
  if (normalized_scale < 0.14) {
    return 2;
  }
  return 3;
}

std::string BuildCoverageSignature(const ati::OuterBoardMeasurement& measurement,
                                   const cv::Size& image_size,
                                   int grid_cols,
                                   int grid_rows) {
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    centroid += measurement.refined_outer_corners_original_image[static_cast<std::size_t>(corner_index)];
  }
  centroid /= 4.0;

  const int cell_x = std::max(0, std::min(grid_cols - 1,
      static_cast<int>(std::floor(grid_cols * centroid.x() /
                                  std::max(1, image_size.width)))));
  const int cell_y = std::max(0, std::min(grid_rows - 1,
      static_cast<int>(std::floor(grid_rows * centroid.y() /
                                  std::max(1, image_size.height)))));
  const int scale_bin = ScaleBinForMeasurement(measurement, image_size);

  std::ostringstream stream;
  stream << cell_x << ":" << cell_y << ":" << scale_bin;
  return stream.str();
}

std::vector<FrameSelectionDecision> SelectFramesKalibrStyle(
    const std::vector<FrameCandidate>& candidates,
    const DatasetConnectivity& connectivity,
    const ati::OuterBootstrapOptions& bootstrap_options,
    const CmdArgs& args) {
  std::vector<FrameSelectionDecision> decisions;
  decisions.reserve(candidates.size());

  std::map<int, int> accepted_observation_count_per_board;
  std::map<int, std::set<std::string> > accepted_signatures_per_board;
  std::set<std::pair<int, int> > accepted_board_pairs;

  for (std::size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const FrameCandidate& candidate = candidates[candidate_index];

    std::vector<ati::OuterBoardMeasurement> usable_measurements;
    std::vector<int> usable_board_ids;
    for (std::size_t measurement_index = 0;
         measurement_index < candidate.frame_input.measurements.board_measurements.size();
         ++measurement_index) {
      const ati::OuterBoardMeasurement& measurement =
          candidate.frame_input.measurements.board_measurements[measurement_index];
      if (!IsUsableBoardMeasurement(measurement, bootstrap_options)) {
        continue;
      }
      if (connectivity.reference_board_observed &&
          connectivity.reference_connected_boards.find(measurement.board_id) ==
              connectivity.reference_connected_boards.end()) {
        continue;
      }
      usable_measurements.push_back(measurement);
      if (std::find(usable_board_ids.begin(), usable_board_ids.end(), measurement.board_id) ==
          usable_board_ids.end()) {
        usable_board_ids.push_back(measurement.board_id);
      }
    }
    std::sort(usable_board_ids.begin(), usable_board_ids.end());

    FrameSelectionDecision decision;
    decision.image_path = candidate.image_path;
    decision.frame_index = candidate.frame_input.frame_index;
    decision.frame_label = candidate.frame_input.frame_label;
    decision.usable_board_ids = usable_board_ids;
    decision.usable_board_count = static_cast<int>(usable_board_ids.size());

    if (usable_measurements.empty()) {
      decision.accepted = false;
      decision.reason = connectivity.reference_board_observed
                            ? "no_reference_connected_usable_boards"
                            : "no_usable_boards";
      decisions.push_back(decision);
      continue;
    }

    if (args.force_all_frames) {
      decision.accepted = true;
      decision.reason = "force_all_frames";
    } else {
      std::vector<std::string> reasons;

      bool board_needs_bootstrap_coverage = false;
      for (std::size_t measurement_index = 0; measurement_index < usable_measurements.size();
           ++measurement_index) {
        const int board_id = usable_measurements[measurement_index].board_id;
        if (accepted_observation_count_per_board[board_id] < args.min_initial_views_per_board) {
          board_needs_bootstrap_coverage = true;
          break;
        }
      }
      if (board_needs_bootstrap_coverage) {
        reasons.push_back("min_views_per_board");
      }

      bool new_board_pair = false;
      for (std::size_t first = 0; first < usable_board_ids.size(); ++first) {
        for (std::size_t second = first + 1; second < usable_board_ids.size(); ++second) {
          const std::pair<int, int> board_pair(usable_board_ids[first], usable_board_ids[second]);
          if (accepted_board_pairs.find(board_pair) == accepted_board_pairs.end()) {
            new_board_pair = true;
            break;
          }
        }
        if (new_board_pair) {
          break;
        }
      }
      if (new_board_pair) {
        reasons.push_back("new_board_pair");
      }

      bool new_image_coverage = false;
      for (std::size_t measurement_index = 0; measurement_index < usable_measurements.size();
           ++measurement_index) {
        const ati::OuterBoardMeasurement& measurement = usable_measurements[measurement_index];
        const std::string signature = BuildCoverageSignature(
            measurement,
            candidate.frame_input.measurements.image_size,
            args.coverage_grid_cols,
            args.coverage_grid_rows);
        if (accepted_signatures_per_board[measurement.board_id].find(signature) ==
            accepted_signatures_per_board[measurement.board_id].end()) {
          new_image_coverage = true;
          break;
        }
      }
      if (new_image_coverage) {
        reasons.push_back("new_image_coverage");
      }

      decision.accepted = !reasons.empty();
      decision.reason = decision.accepted ? JoinStrings(reasons) : "redundant_view";
    }

    if (decision.accepted) {
      for (std::size_t measurement_index = 0; measurement_index < usable_measurements.size();
           ++measurement_index) {
        const ati::OuterBoardMeasurement& measurement = usable_measurements[measurement_index];
        ++accepted_observation_count_per_board[measurement.board_id];
        accepted_signatures_per_board[measurement.board_id].insert(
            BuildCoverageSignature(measurement,
                                   candidate.frame_input.measurements.image_size,
                                   args.coverage_grid_cols,
                                   args.coverage_grid_rows));
      }
      for (std::size_t first = 0; first < usable_board_ids.size(); ++first) {
        for (std::size_t second = first + 1; second < usable_board_ids.size(); ++second) {
          accepted_board_pairs.insert(std::make_pair(usable_board_ids[first], usable_board_ids[second]));
        }
      }
    }

    decisions.push_back(decision);
  }

  return decisions;
}

std::vector<ati::OuterBootstrapFrameInput> GatherAcceptedFrames(
    const std::vector<FrameCandidate>& candidates,
    const std::vector<FrameSelectionDecision>& decisions) {
  std::vector<ati::OuterBootstrapFrameInput> accepted_frames;
  for (std::size_t index = 0; index < candidates.size() && index < decisions.size(); ++index) {
    if (decisions[index].accepted) {
      accepted_frames.push_back(candidates[index].frame_input);
    }
  }
  return accepted_frames;
}

ResidualStatistics ComputeResidualStatistics(const ati::OuterBootstrapResult& result) {
  ResidualStatistics stats;
  std::vector<double> residual_x;
  std::vector<double> residual_y;

  for (std::size_t index = 0; index < result.observation_diagnostics.size(); ++index) {
    const ati::OuterBootstrapObservationDiagnostics& diagnostics =
        result.observation_diagnostics[index];
    if (!diagnostics.used_in_solve) {
      continue;
    }
    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      residual_x.push_back(
          diagnostics.corner_residuals_xy[static_cast<std::size_t>(corner_index)].x());
      residual_y.push_back(
          diagnostics.corner_residuals_xy[static_cast<std::size_t>(corner_index)].y());
    }
  }

  stats.used_corner_count = static_cast<int>(residual_x.size());
  if (residual_x.empty()) {
    return stats;
  }

  for (std::size_t index = 0; index < residual_x.size(); ++index) {
    stats.mean_x += residual_x[index];
    stats.mean_y += residual_y[index];
  }
  stats.mean_x /= static_cast<double>(residual_x.size());
  stats.mean_y /= static_cast<double>(residual_y.size());

  for (std::size_t index = 0; index < residual_x.size(); ++index) {
    const double dx = residual_x[index] - stats.mean_x;
    const double dy = residual_y[index] - stats.mean_y;
    stats.sigma_x += dx * dx;
    stats.sigma_y += dy * dy;
  }
  stats.sigma_x = std::sqrt(stats.sigma_x / static_cast<double>(residual_x.size()));
  stats.sigma_y = std::sqrt(stats.sigma_y / static_cast<double>(residual_y.size()));
  return stats;
}

std::vector<ati::OuterBootstrapObservationDiagnostics> CollectOutlierObservations(
    const ati::OuterBootstrapResult& result,
    double sigma_multiplier) {
  std::vector<ati::OuterBootstrapObservationDiagnostics> outliers;
  const ResidualStatistics stats = ComputeResidualStatistics(result);
  if (stats.used_corner_count <= 0) {
    return outliers;
  }

  const double threshold_x = sigma_multiplier * stats.sigma_x;
  const double threshold_y = sigma_multiplier * stats.sigma_y;
  for (std::size_t index = 0; index < result.observation_diagnostics.size(); ++index) {
    const ati::OuterBootstrapObservationDiagnostics& diagnostics =
        result.observation_diagnostics[index];
    if (!diagnostics.used_in_solve) {
      continue;
    }

    bool is_outlier = false;
    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      const Eigen::Vector2d& residual =
          diagnostics.corner_residuals_xy[static_cast<std::size_t>(corner_index)];
      if (std::abs(residual.x()) > threshold_x || std::abs(residual.y()) > threshold_y) {
        is_outlier = true;
        break;
      }
    }
    if (is_outlier) {
      outliers.push_back(diagnostics);
    }
  }
  return outliers;
}

std::vector<ati::OuterBootstrapFrameInput> RemoveFlaggedObservations(
    const std::vector<ati::OuterBootstrapFrameInput>& frames,
    const std::vector<ati::OuterBootstrapObservationDiagnostics>& flagged_observations,
    const ati::OuterBootstrapOptions& bootstrap_options) {
  std::set<std::pair<int, int> > flagged_keys;
  for (std::size_t index = 0; index < flagged_observations.size(); ++index) {
    flagged_keys.insert(std::make_pair(flagged_observations[index].frame_index,
                                       flagged_observations[index].board_id));
  }

  std::vector<ati::OuterBootstrapFrameInput> filtered_frames;
  filtered_frames.reserve(frames.size());
  for (std::size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
    ati::OuterBootstrapFrameInput filtered_frame = frames[frame_index];
    std::vector<ati::OuterBoardMeasurement> kept_measurements;
    kept_measurements.reserve(filtered_frame.measurements.board_measurements.size());
    for (std::size_t measurement_index = 0;
         measurement_index < filtered_frame.measurements.board_measurements.size();
         ++measurement_index) {
      const ati::OuterBoardMeasurement& measurement =
          filtered_frame.measurements.board_measurements[measurement_index];
      if (flagged_keys.find(std::make_pair(filtered_frame.frame_index, measurement.board_id)) ==
          flagged_keys.end()) {
        kept_measurements.push_back(measurement);
      }
    }
    filtered_frame.measurements.board_measurements = kept_measurements;

    bool keep_frame = false;
    for (std::size_t measurement_index = 0;
         measurement_index < filtered_frame.measurements.board_measurements.size();
         ++measurement_index) {
      if (IsUsableBoardMeasurement(filtered_frame.measurements.board_measurements[measurement_index],
                                   bootstrap_options)) {
        keep_frame = true;
        break;
      }
    }
    if (keep_frame) {
      filtered_frames.push_back(filtered_frame);
    }
  }
  return filtered_frames;
}

void WriteObservationDiagnosticsCsv(const fs::path& output_path,
                                    const ati::OuterBootstrapResult& result) {
  std::ofstream output(output_path.string().c_str());
  output << "frame_index,frame_label,board_id,detection_quality,reference_connected,"
            "frame_initialized,board_initialized,used_in_solve,observation_rmse,"
            "max_abs_residual_x,max_abs_residual_y,"
            "c0_rx,c0_ry,c1_rx,c1_ry,c2_rx,c2_ry,c3_rx,c3_ry\n";
  output << std::fixed << std::setprecision(8);
  for (std::size_t index = 0; index < result.observation_diagnostics.size(); ++index) {
    const ati::OuterBootstrapObservationDiagnostics& diagnostics =
        result.observation_diagnostics[index];
    output << diagnostics.frame_index << ","
           << diagnostics.frame_label << ","
           << diagnostics.board_id << ","
           << diagnostics.detection_quality << ","
           << diagnostics.reference_connected << ","
           << diagnostics.frame_initialized << ","
           << diagnostics.board_initialized << ","
           << diagnostics.used_in_solve << ","
           << diagnostics.observation_rmse << ","
           << diagnostics.max_abs_residual_x << ","
           << diagnostics.max_abs_residual_y;
    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      const Eigen::Vector2d& residual =
          diagnostics.corner_residuals_xy[static_cast<std::size_t>(corner_index)];
      output << "," << residual.x() << "," << residual.y();
    }
    output << "\n";
  }
}

void WriteFrameSelectionSummary(const fs::path& output_path,
                                const DatasetConnectivity& connectivity,
                                const std::vector<FrameSelectionDecision>& decisions) {
  std::ofstream output(output_path.string().c_str());
  output << "reference_board_observed: " << connectivity.reference_board_observed << "\n";
  output << "reference_connected_boards: "
         << JoinInts(std::vector<int>(connectivity.reference_connected_boards.begin(),
                                      connectivity.reference_connected_boards.end()))
         << "\n\n";

  int accepted_count = 0;
  for (std::size_t index = 0; index < decisions.size(); ++index) {
    accepted_count += decisions[index].accepted ? 1 : 0;
  }
  output << "accepted_frames: " << accepted_count << "\n";
  output << "rejected_frames: " << static_cast<int>(decisions.size()) - accepted_count << "\n\n";

  for (std::size_t index = 0; index < decisions.size(); ++index) {
    const FrameSelectionDecision& decision = decisions[index];
    output << (decision.accepted ? "[ACCEPT] " : "[REJECT] ")
           << "frame=" << decision.frame_index
           << " label=" << decision.frame_label
           << " usable_board_count=" << decision.usable_board_count
           << " usable_board_ids=" << JoinInts(decision.usable_board_ids)
           << " reason=" << decision.reason
           << " image=" << decision.image_path
           << "\n";
  }
}

void WriteRemovedObservationSummary(
    const fs::path& output_path,
    const std::vector<ati::OuterBootstrapObservationDiagnostics>& removed_observations,
    double sigma_multiplier,
    const ResidualStatistics& residual_stats) {
  std::ofstream output(output_path.string().c_str());
  output << std::fixed << std::setprecision(8);
  output << "post_filter_sigma_multiplier: " << sigma_multiplier << "\n";
  output << "used_corner_count: " << residual_stats.used_corner_count << "\n";
  output << "mean_x: " << residual_stats.mean_x << "\n";
  output << "mean_y: " << residual_stats.mean_y << "\n";
  output << "sigma_x: " << residual_stats.sigma_x << "\n";
  output << "sigma_y: " << residual_stats.sigma_y << "\n";
  output << "threshold_x: " << sigma_multiplier * residual_stats.sigma_x << "\n";
  output << "threshold_y: " << sigma_multiplier * residual_stats.sigma_y << "\n";
  output << "removed_observation_count: " << removed_observations.size() << "\n\n";

  for (std::size_t index = 0; index < removed_observations.size(); ++index) {
    const ati::OuterBootstrapObservationDiagnostics& diagnostics = removed_observations[index];
    output << "frame=" << diagnostics.frame_index
           << " label=" << diagnostics.frame_label
           << " board=" << diagnostics.board_id
           << " observation_rmse=" << diagnostics.observation_rmse
           << " max_abs_x=" << diagnostics.max_abs_residual_x
           << " max_abs_y=" << diagnostics.max_abs_residual_y
           << "\n";
    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      const Eigen::Vector2d& residual =
          diagnostics.corner_residuals_xy[static_cast<std::size_t>(corner_index)];
      output << "  corner" << corner_index
             << ": rx=" << residual.x()
             << " ry=" << residual.y()
             << "\n";
    }
  }
}

void WriteBootstrapSummary(const fs::path& output_path,
                           const std::string& label,
                           const ati::OuterBootstrapResult& result) {
  std::ofstream output(output_path.string().c_str());
  output << std::fixed << std::setprecision(8);
  output << "label: " << label << "\n";
  output << "success: " << result.success << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "reference_board_id: " << result.reference_board_id << "\n";
  output << "used_frame_count: " << result.used_frame_count << "\n";
  output << "used_board_observation_count: " << result.used_board_observation_count << "\n";
  output << "used_corner_count: " << result.used_corner_count << "\n";
  output << "global_rmse: " << result.global_rmse << "\n\n";

  output << "coarse_camera:\n";
  output << "  xi: " << result.coarse_camera.xi << "\n";
  output << "  alpha: " << result.coarse_camera.alpha << "\n";
  output << "  fu: " << result.coarse_camera.fu << "\n";
  output << "  fv: " << result.coarse_camera.fv << "\n";
  output << "  cu: " << result.coarse_camera.cu << "\n";
  output << "  cv: " << result.coarse_camera.cv << "\n";
  output << "  resolution: [" << result.coarse_camera.resolution.width
         << ", " << result.coarse_camera.resolution.height << "]\n\n";

  output << "boards:\n";
  for (std::size_t index = 0; index < result.boards.size(); ++index) {
    const ati::OuterBootstrapBoardState& board = result.boards[index];
    output << "  board " << board.board_id
           << ": initialized=" << board.initialized
           << " observation_count=" << board.observation_count
           << " rmse=" << board.rmse << "\n";
    output << MatrixToString(board.T_reference_board) << "\n";
  }
  output << "\nframes:\n";
  for (std::size_t index = 0; index < result.frames.size(); ++index) {
    const ati::OuterBootstrapFrameState& frame = result.frames[index];
    output << "  frame " << frame.frame_index
           << " (" << frame.frame_label << ")"
           << ": initialized=" << frame.initialized
           << " visible_board_ids=" << JoinInts(frame.visible_board_ids)
           << " observation_count=" << frame.observation_count
           << " rmse=" << frame.rmse << "\n";
    output << MatrixToString(frame.T_camera_reference) << "\n";
  }

  if (!result.warnings.empty()) {
    output << "\nwarnings:\n";
    for (std::size_t index = 0; index < result.warnings.size(); ++index) {
      output << "  - " << result.warnings[index] << "\n";
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
    config.outer_detector_config.tag_id = config.tag_id;
    config.outer_detector_config.tag_ids = config.tag_ids;

    ati::MultiScaleOuterTagDetector outer_detector(config.outer_detector_config);
    const ati::OuterBootstrapOptions bootstrap_options = MakeBootstrapOptions(config, args);
    ati::MultiBoardOuterBootstrap bootstrap(config, bootstrap_options);

    const std::vector<std::string> image_paths = CollectImagePaths(args.image_path, args.all);
    const fs::path output_dir(args.output_path);
    EnsureDirectoryExists(output_dir);
    const fs::path overlay_dir = output_dir / "outer_overlays";
    if (args.save_overlays) {
      EnsureDirectoryExists(overlay_dir);
    }

    std::vector<FrameCandidate> candidates;
    candidates.reserve(image_paths.size());

    std::cout << "Detecting outer measurements on " << image_paths.size() << " image(s)...\n";
    for (std::size_t image_index = 0; image_index < image_paths.size(); ++image_index) {
      const std::string& image_path = image_paths[image_index];
      cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
      if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
      }

      const ati::OuterTagMultiDetectionResult detections = outer_detector.DetectMultiple(image);
      const fs::path image_fs_path(image_path);
      const std::string frame_label = image_fs_path.stem().string();
      if (args.save_overlays || args.show) {
        cv::Mat overlay = image.clone();
        outer_detector.DrawDetections(detections, &overlay, true);
        if (args.save_overlays) {
          const fs::path overlay_path = overlay_dir / (frame_label + "_outer_overlay.png");
          cv::imwrite(overlay_path.string(), overlay);
        }
        if (args.show) {
          cv::imshow("multi_board_outer_bootstrap", overlay);
          cv::waitKey(1);
        }
      }

      FrameCandidate candidate;
      candidate.image_path = image_path;
      candidate.frame_input.frame_index = static_cast<int>(image_index);
      candidate.frame_input.frame_label = frame_label;
      candidate.frame_input.measurements = detections.frame_measurements;
      candidates.push_back(candidate);

      std::cout << "  [" << (image_index + 1) << "/" << image_paths.size() << "] "
                << frame_label
                << " success_boards=" << detections.SuccessfulBoardCount() << std::endl;
    }

    const DatasetConnectivity connectivity =
        AnalyzeDatasetConnectivity(candidates, bootstrap_options);
    const std::vector<FrameSelectionDecision> selection =
        SelectFramesKalibrStyle(candidates, connectivity, bootstrap_options, args);
    const std::vector<ati::OuterBootstrapFrameInput> accepted_frames =
        GatherAcceptedFrames(candidates, selection);

    WriteFrameSelectionSummary(output_dir / "frame_selection.txt", connectivity, selection);

    std::cout << "Accepted " << accepted_frames.size() << " / " << candidates.size()
              << " frame(s) for raw bootstrap.\n";

    const ati::OuterBootstrapResult raw_result = bootstrap.Solve(accepted_frames);
    WriteBootstrapSummary(output_dir / "bootstrap_raw_summary.txt", "raw", raw_result);
    WriteObservationDiagnosticsCsv(output_dir / "bootstrap_raw_observations.csv", raw_result);

    ati::OuterBootstrapResult filtered_result = raw_result;
    std::vector<ati::OuterBootstrapObservationDiagnostics> removed_observations;
    ResidualStatistics residual_stats = ComputeResidualStatistics(raw_result);

    if (!args.no_post_filter &&
        raw_result.success &&
        raw_result.used_board_observation_count >= args.min_used_observations_for_post_filter) {
      removed_observations = CollectOutlierObservations(raw_result, args.post_filter_sigma);
      if (!removed_observations.empty()) {
        const std::vector<ati::OuterBootstrapFrameInput> filtered_frames =
            RemoveFlaggedObservations(accepted_frames, removed_observations, bootstrap_options);
        filtered_result = bootstrap.Solve(filtered_frames);
      }
    } else if (!args.no_post_filter && raw_result.success) {
      filtered_result.warnings.push_back(
          "post-filter skipped because used_board_observation_count < min_used_observations_for_post_filter");
    } else if (args.no_post_filter) {
      filtered_result.warnings.push_back("post-filter disabled by --no-post-filter");
    }

    WriteRemovedObservationSummary(output_dir / "post_filter_removed_observations.txt",
                                  removed_observations,
                                  args.post_filter_sigma,
                                  residual_stats);
    WriteBootstrapSummary(output_dir / "bootstrap_filtered_summary.txt", "filtered", filtered_result);
    WriteObservationDiagnosticsCsv(output_dir / "bootstrap_filtered_observations.csv",
                                   filtered_result);

    std::cout << "\nRaw bootstrap:\n"
              << "  success=" << raw_result.success
              << " used_frames=" << raw_result.used_frame_count
              << " used_observations=" << raw_result.used_board_observation_count
              << " global_rmse=" << raw_result.global_rmse << "\n";
    std::cout << "Filtered bootstrap:\n"
              << "  success=" << filtered_result.success
              << " used_frames=" << filtered_result.used_frame_count
              << " used_observations=" << filtered_result.used_board_observation_count
              << " global_rmse=" << filtered_result.global_rmse << "\n";
    std::cout << "Removed observations in post-filter: "
              << removed_observations.size() << "\n";
    std::cout << "Outputs written to: " << output_dir.string() << "\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "run_multi_board_outer_bootstrap failed: " << exception.what() << "\n";
    return 1;
  }
}
