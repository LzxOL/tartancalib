#include <aslam/cameras/apriltag_internal/IterativeCoarseCalibrationExperiment.hpp>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ati = aslam::cameras::apriltag_internal;

struct CmdArgs {
  std::string image_dir;
  std::string config_path;
  std::string output_dir;
  std::vector<int> board_ids;
  std::vector<std::string> group_filters;
  int max_iterations = 3;
  int pose_refinement_rounds = 2;
  bool no_subpix = false;
  bool use_internal_points_for_update = false;
  double internal_point_quality_threshold = 0.45;
  double convergence_threshold = 0.05;
  double init_xi = -0.2;
  double init_alpha = 0.6;
  double init_fu_scale = 0.55;
  double init_fv_scale = 0.55;
  double init_cu_offset = 0.0;
  double init_cv_offset = 0.0;
};

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --image-dir DIR --config YAML [--output-dir DIR] [--board-ids 1,2]"
      << " [--group image1,image2] [--max-iterations N]"
      << " [--use-internal-points-for-update] [--internal-point-quality-threshold Q]"
      << " [--convergence-threshold T]\n\n"
      << "Example:\n"
      << "  " << program
      << " --image-dir ./image/img_seq"
      << " --config ./aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
      << " --output-dir ./iterative_outputs --max-iterations 3\n";
}

std::vector<std::string> SplitCsv(const std::string& value) {
  std::vector<std::string> tokens;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      tokens.push_back(token);
    }
  }
  return tokens;
}

std::vector<int> ParseIntCsv(const std::string& value) {
  std::vector<int> numbers;
  const std::vector<std::string> tokens = SplitCsv(value);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    numbers.push_back(std::stoi(tokens[index]));
  }
  return numbers;
}

CmdArgs ParseArgs(int argc, char** argv) {
  CmdArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string token = argv[i];
    if (token == "--image-dir" && i + 1 < argc) {
      args.image_dir = argv[++i];
    } else if (token == "--config" && i + 1 < argc) {
      args.config_path = argv[++i];
    } else if (token == "--output-dir" && i + 1 < argc) {
      args.output_dir = argv[++i];
    } else if (token == "--board-ids" && i + 1 < argc) {
      args.board_ids = ParseIntCsv(argv[++i]);
    } else if (token == "--group" && i + 1 < argc) {
      args.group_filters = SplitCsv(argv[++i]);
    } else if (token == "--max-iterations" && i + 1 < argc) {
      args.max_iterations = std::atoi(argv[++i]);
    } else if (token == "--pose-refinement-rounds" && i + 1 < argc) {
      args.pose_refinement_rounds = std::atoi(argv[++i]);
    } else if (token == "--use-internal-points-for-update") {
      args.use_internal_points_for_update = true;
    } else if (token == "--internal-point-quality-threshold" && i + 1 < argc) {
      args.internal_point_quality_threshold = std::atof(argv[++i]);
    } else if (token == "--convergence-threshold" && i + 1 < argc) {
      args.convergence_threshold = std::atof(argv[++i]);
    } else if (token == "--init-xi" && i + 1 < argc) {
      args.init_xi = std::atof(argv[++i]);
    } else if (token == "--init-alpha" && i + 1 < argc) {
      args.init_alpha = std::atof(argv[++i]);
    } else if (token == "--init-fu-scale" && i + 1 < argc) {
      args.init_fu_scale = std::atof(argv[++i]);
    } else if (token == "--init-fv-scale" && i + 1 < argc) {
      args.init_fv_scale = std::atof(argv[++i]);
    } else if (token == "--init-cu-offset" && i + 1 < argc) {
      args.init_cu_offset = std::atof(argv[++i]);
    } else if (token == "--init-cv-offset" && i + 1 < argc) {
      args.init_cv_offset = std::atof(argv[++i]);
    } else if (token == "--no-subpix") {
      args.no_subpix = true;
    } else if (token == "--help" || token == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + token);
    }
  }

  if (args.image_dir.empty() || args.config_path.empty()) {
    throw std::runtime_error("Both --image-dir and --config are required.");
  }
  if (args.output_dir.empty()) {
    args.output_dir = "./iterative_coarse_experiment_outputs";
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CmdArgs args = ParseArgs(argc, argv);

    ati::ApriltagInternalConfig config =
        ati::ApriltagInternalDetector::LoadConfig(args.config_path);
    ati::ApriltagInternalDetectionOptions detection_options;
    detection_options.do_subpix_refinement = !args.no_subpix;

    ati::IterativeCoarseCalibrationExperimentRequest request;
    request.image_dir = args.image_dir;
    request.output_dir = args.output_dir;
    request.base_config = config;
    request.detection_options = detection_options;
    request.experiment_options.board_ids = args.board_ids;
    request.experiment_options.group_filters = args.group_filters;
    request.experiment_options.max_iterations = args.max_iterations;
    request.experiment_options.pose_refinement_rounds = args.pose_refinement_rounds;
    request.experiment_options.use_internal_points_for_update =
        args.use_internal_points_for_update;
    request.experiment_options.internal_point_quality_threshold =
        args.internal_point_quality_threshold;
    request.experiment_options.convergence_threshold = args.convergence_threshold;
    request.experiment_options.init_xi = args.init_xi;
    request.experiment_options.init_alpha = args.init_alpha;
    request.experiment_options.init_fu_scale = args.init_fu_scale;
    request.experiment_options.init_fv_scale = args.init_fv_scale;
    request.experiment_options.init_cu_offset = args.init_cu_offset;
    request.experiment_options.init_cv_offset = args.init_cv_offset;

    ati::IterativeCoarseCalibrationExperiment experiment(request);
    experiment.Run();

    std::cout << "iterative coarse calibration experiment finished\n";
    std::cout << "  image_dir: " << request.image_dir << "\n";
    std::cout << "  output_dir: " << request.output_dir << "\n";
    std::cout << "  max_iterations: " << request.experiment_options.max_iterations << "\n";
    std::cout << "  use_internal_points_for_update: "
              << (request.experiment_options.use_internal_points_for_update ? "yes" : "no") << "\n";
    std::cout << "  group rule: filename stem before the last '-'\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[iterative-apriltag-experiment] " << error.what() << "\n\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
