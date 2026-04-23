#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionMeasurementBuilder.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionResidualEvaluator.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
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
  ati::OuterBootstrapFrameInput bootstrap_input;
  ati::InternalRegenerationFrameInput regeneration_input;
};

struct BuilderValidationSummary {
  bool success = false;
  bool counting_consistent = false;
  bool flat_hierarchical_consistent = false;
  bool frame_order_invariant = false;
  bool label_mismatch_warning_observed = false;
  std::vector<std::string> warnings;
  std::string failure_reason;
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

ati::ApriltagInternalDetectionOptions MakeDetectionOptions(
    const ati::ApriltagInternalConfig& config) {
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

void WriteJointSummary(const fs::path& output_path,
                       const ati::JointMeasurementBuildResult& result) {
  std::ofstream output(output_path.string());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "reference_board_id: " << result.reference_board_id << "\n";
  output << "used_frame_count: " << result.used_frame_count << "\n";
  output << "accepted_outer_board_observation_count: "
         << result.accepted_outer_board_observation_count << "\n";
  output << "accepted_internal_board_observation_count: "
         << result.accepted_internal_board_observation_count << "\n";
  output << "used_board_observation_count: " << result.used_board_observation_count << "\n";
  output << "used_outer_point_count: " << result.used_outer_point_count << "\n";
  output << "used_internal_point_count: " << result.used_internal_point_count << "\n";
  output << "used_total_point_count: " << result.used_total_point_count << "\n";
  output << "bootstrap_global_rmse: " << result.bootstrap_seed.global_rmse << "\n";
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteJointCsv(const fs::path& output_path,
                   const ati::JointMeasurementBuildResult& result) {
  std::ofstream output(output_path.string());
  output << "frame_index,frame_label,board_id,point_id,point_type,image_x,image_y,"
         << "target_x,target_y,target_z,debug_quality,used_in_solver,rejection_reason_code,"
         << "rejection_detail,frame_storage_index,source_board_observation_index,"
         << "source_point_index,source_kind\n";
  for (const ati::JointMeasurementFrameResult& frame_result : result.frames) {
    for (const ati::JointBoardObservation& board_observation : frame_result.board_observations) {
      for (const ati::JointPointObservation& point : board_observation.points) {
        output << point.frame_index << ","
               << point.frame_label << ","
               << point.board_id << ","
               << point.point_id << ","
               << ati::ToString(point.point_type) << ","
               << point.image_xy.x() << ","
               << point.image_xy.y() << ","
               << point.target_xyz_board.x() << ","
               << point.target_xyz_board.y() << ","
               << point.target_xyz_board.z() << ","
               << point.quality << ","
               << (point.used_in_solver ? 1 : 0) << ","
               << ati::ToString(point.rejection_reason_code) << ","
               << point.rejection_detail << ","
               << point.frame_storage_index << ","
               << point.source_board_observation_index << ","
               << point.source_point_index << ","
               << ati::ToString(point.source_kind) << "\n";
      }
    }
  }
}

std::set<std::tuple<int, int, int, int, int> > BuildSolverSignatureSet(
    const ati::JointMeasurementBuildResult& result) {
  std::set<std::tuple<int, int, int, int, int> > signatures;
  for (const ati::JointPointObservation& point : result.solver_observations) {
    signatures.insert(std::make_tuple(
        point.frame_index, point.board_id, point.point_id,
        static_cast<int>(point.point_type), static_cast<int>(point.source_kind)));
  }
  return signatures;
}

BuilderValidationSummary ValidateJointMeasurementBuilder(
    const std::vector<ati::JointMeasurementFrameInput>& joint_inputs,
    const ati::OuterBootstrapResult& bootstrap_result,
    const ati::JointReprojectionMeasurementBuilder& builder,
    const ati::JointMeasurementBuildResult& primary_result) {
  BuilderValidationSummary summary;
  if (!primary_result.success) {
    summary.failure_reason = "Primary joint measurement build failed.";
    return summary;
  }

  int hierarchical_used_points = 0;
  std::set<std::pair<int, int> > used_board_observation_keys;
  for (const ati::JointMeasurementFrameResult& frame_result : primary_result.frames) {
    for (const ati::JointBoardObservation& board_observation :
         frame_result.board_observations) {
      bool board_has_used_point = false;
      for (const ati::JointPointObservation& point : board_observation.points) {
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

  std::vector<ati::JointMeasurementFrameInput> reversed_inputs = joint_inputs;
  std::reverse(reversed_inputs.begin(), reversed_inputs.end());
  const ati::JointMeasurementBuildResult reversed_result =
      builder.Build(reversed_inputs, bootstrap_result);
  summary.frame_order_invariant =
      reversed_result.success &&
      reversed_result.used_frame_count == primary_result.used_frame_count &&
      reversed_result.used_board_observation_count == primary_result.used_board_observation_count &&
      reversed_result.used_outer_point_count == primary_result.used_outer_point_count &&
      reversed_result.used_internal_point_count == primary_result.used_internal_point_count &&
      BuildSolverSignatureSet(reversed_result) == BuildSolverSignatureSet(primary_result);

  if (!joint_inputs.empty()) {
    std::vector<ati::JointMeasurementFrameInput> mismatch_inputs = joint_inputs;
    mismatch_inputs.front().frame_label += "_label_mismatch_probe";
    mismatch_inputs.front().regenerated_internal.frame_label = mismatch_inputs.front().frame_label;
    const ati::JointMeasurementBuildResult mismatch_result =
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
    summary.warnings.push_back("Flat solver_observations do not match hierarchical used points.");
  }
  if (!summary.frame_order_invariant) {
    summary.warnings.push_back("Frame-order perturbation changed the joint measurement result.");
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

void WriteValidationSummary(const fs::path& output_path,
                            const BuilderValidationSummary& summary) {
  std::ofstream output(output_path.string());
  output << "success: " << (summary.success ? 1 : 0) << "\n";
  output << "failure_reason: " << summary.failure_reason << "\n";
  output << "counting_consistent: " << (summary.counting_consistent ? 1 : 0) << "\n";
  output << "flat_hierarchical_consistent: "
         << (summary.flat_hierarchical_consistent ? 1 : 0) << "\n";
  output << "frame_order_invariant: " << (summary.frame_order_invariant ? 1 : 0) << "\n";
  output << "label_mismatch_warning_observed: "
         << (summary.label_mismatch_warning_observed ? 1 : 0) << "\n";
  for (const std::string& warning : summary.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteResidualSummary(const fs::path& output_path,
                          const ati::JointResidualEvaluationResult& result) {
  std::ofstream output(output_path.string());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "overall_rmse: " << result.overall_rmse << "\n";
  output << "outer_only_rmse: " << result.outer_only_rmse << "\n";
  output << "internal_only_rmse: " << result.internal_only_rmse << "\n";
  output << "point_count: " << result.point_diagnostics.size() << "\n";
  output << "frame_count: " << result.frame_diagnostics.size() << "\n";
  output << "board_count: " << result.board_diagnostics.size() << "\n";
  output << "board_observation_count: " << result.board_observation_diagnostics.size() << "\n";
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
  output << "worst_points:\n";
  for (const ati::JointResidualPointDiagnostics& point : result.worst_points) {
    output << "  frame=" << point.frame_index
           << " board=" << point.board_id
           << " point=" << point.point_id
           << " type=" << ati::ToString(point.point_type)
           << " residual_norm=" << point.residual_norm << "\n";
  }
  output << "worst_frames:\n";
  for (const ati::JointResidualFrameDiagnostics& frame : result.worst_frames) {
    output << "  frame=" << frame.frame_index
           << " rmse=" << frame.rmse
           << " points=" << frame.point_count << "\n";
  }
  output << "worst_boards:\n";
  for (const ati::JointResidualBoardDiagnostics& board : result.worst_boards) {
    output << "  board=" << board.board_id
           << " rmse=" << board.rmse
           << " points=" << board.point_count << "\n";
  }
}

void WriteResidualPointsCsv(const fs::path& output_path,
                            const ati::JointResidualEvaluationResult& result) {
  std::ofstream output(output_path.string());
  output << "frame_index,frame_label,board_id,point_id,point_type,observed_x,observed_y,"
         << "predicted_x,predicted_y,target_x,target_y,target_z,residual_x,residual_y,"
         << "residual_norm,debug_quality,used_in_solver,frame_storage_index,"
         << "source_board_observation_index,source_point_index,source_kind\n";
  for (const ati::JointResidualPointDiagnostics& point : result.point_diagnostics) {
    output << point.frame_index << ","
           << point.frame_label << ","
           << point.board_id << ","
           << point.point_id << ","
           << ati::ToString(point.point_type) << ","
           << point.observed_image_xy.x() << ","
           << point.observed_image_xy.y() << ","
           << point.predicted_image_xy.x() << ","
           << point.predicted_image_xy.y() << ","
           << point.target_xyz_board.x() << ","
           << point.target_xyz_board.y() << ","
           << point.target_xyz_board.z() << ","
           << point.residual_xy.x() << ","
           << point.residual_xy.y() << ","
           << point.residual_norm << ","
           << point.quality << ","
           << (point.used_in_solver ? 1 : 0) << ","
           << point.frame_storage_index << ","
           << point.source_board_observation_index << ","
           << point.source_point_index << ","
           << ati::ToString(point.source_kind) << "\n";
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
    ati::JointMeasurementBuildOptions build_options;
    build_options.reference_board_id = args.reference_board_id;
    const ati::JointReprojectionMeasurementBuilder builder(config, build_options);
    const ati::JointResidualEvaluationOptions residual_options;
    const ati::JointReprojectionResidualEvaluator residual_evaluator(residual_options);

    const std::vector<std::string> image_paths = CollectImagePaths(args.image_path, args.all);
    const fs::path output_dir(args.output_path);
    EnsureDirectoryExists(output_dir);
    const fs::path overlay_dir = output_dir / "joint_measurement_overlays";
    const fs::path residual_overlay_dir = output_dir / "joint_residual_overlays";
    if (args.save_overlays) {
      EnsureDirectoryExists(overlay_dir);
      EnsureDirectoryExists(residual_overlay_dir);
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
      frame.bootstrap_input.frame_index = static_cast<int>(image_index);
      frame.bootstrap_input.frame_label = frame_label;
      frame.bootstrap_input.measurements = outer_detections.frame_measurements;
      frame.regeneration_input.frame_index = frame.bootstrap_input.frame_index;
      frame.regeneration_input.frame_label = frame_label;
      frame.regeneration_input.outer_detections = outer_detections;
      bootstrap_frames.push_back(frame.bootstrap_input);
      frames.push_back(frame);

      std::cout << "  [" << (image_index + 1) << "/" << image_paths.size() << "] "
                << frame_label << " success_boards="
                << outer_detections.SuccessfulBoardCount() << "\n";
    }

    const ati::OuterBootstrapResult bootstrap_result = bootstrap.Solve(bootstrap_frames);
    if (!bootstrap_result.success) {
      WriteJointSummary(output_dir / "joint_measurement_summary.txt",
                        ati::JointMeasurementBuildResult{});
      std::cout << "Bootstrap failed.\n";
      return 1;
    }

    std::vector<ati::JointMeasurementFrameInput> joint_inputs;
    joint_inputs.reserve(frames.size());
    std::cout << "Regenerating internal measurements...\n";
    for (std::size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
      const cv::Mat image = cv::imread(frames[frame_index].image_path, cv::IMREAD_UNCHANGED);
      if (image.empty()) {
        throw std::runtime_error("Failed to read image: " + frames[frame_index].image_path);
      }

      const ati::InternalRegenerationFrameResult regeneration_result =
          regenerator.RegenerateFrame(image, frames[frame_index].regeneration_input, bootstrap_result);

      ati::JointMeasurementFrameInput joint_input;
      joint_input.frame_index = frames[frame_index].bootstrap_input.frame_index;
      joint_input.frame_label = frames[frame_index].bootstrap_input.frame_label;
      joint_input.outer_detections = frames[frame_index].regeneration_input.outer_detections;
      joint_input.regenerated_internal = regeneration_result;
      joint_inputs.push_back(joint_input);

      std::cout << "  regenerated frame " << joint_input.frame_index
                << " successful_boards=" << regeneration_result.SuccessfulBoardCount()
                << " valid_internal=" << regeneration_result.ValidInternalCornerCount() << "\n";
    }

    const ati::JointMeasurementBuildResult joint_result =
        builder.Build(joint_inputs, bootstrap_result);
    WriteJointSummary(output_dir / "joint_measurement_summary.txt", joint_result);
    WriteJointCsv(output_dir / "joint_measurements.csv", joint_result);
    const BuilderValidationSummary validation_summary =
        ValidateJointMeasurementBuilder(joint_inputs, bootstrap_result, builder, joint_result);
    WriteValidationSummary(output_dir / "joint_validation_summary.txt", validation_summary);

    if (!validation_summary.success) {
      std::cout << "\nJoint measurement builder validation failed.\n"
                << "Validation summary: "
                << (output_dir / "joint_validation_summary.txt").string() << "\n";
      return 1;
    }

    const ati::JointResidualEvaluationResult residual_result =
        residual_evaluator.Evaluate(joint_result, bootstrap_result);
    WriteResidualSummary(output_dir / "joint_residual_summary.txt", residual_result);
    WriteResidualPointsCsv(output_dir / "joint_residual_points.csv", residual_result);

    if (!residual_result.success) {
      std::cout << "\nFixed-state residual evaluation failed.\n"
                << "Residual summary: "
                << (output_dir / "joint_residual_summary.txt").string() << "\n";
      return 1;
    }

    if (args.save_overlays || args.show) {
      for (std::size_t frame_index = 0; frame_index < joint_inputs.size(); ++frame_index) {
        const cv::Mat image = cv::imread(frames[frame_index].image_path, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
          throw std::runtime_error("Failed to read image: " + frames[frame_index].image_path);
        }

        cv::Mat overlay;
        builder.DrawFrameOverlay(image, joint_result.frames[frame_index], &overlay);
        if (args.save_overlays) {
          const fs::path overlay_path =
              overlay_dir / (joint_inputs[frame_index].frame_label + "_joint_measurement_overlay.png");
          cv::imwrite(overlay_path.string(), overlay);
        }
        if (args.show) {
          cv::imshow("joint_reprojection_measurement_prep", overlay);
          cv::waitKey(1);
        }

        cv::Mat residual_overlay;
        residual_evaluator.DrawFrameOverlay(
            image, joint_inputs[frame_index].frame_index, residual_result, &residual_overlay);
        if (args.save_overlays) {
          const fs::path overlay_path =
              residual_overlay_dir / (joint_inputs[frame_index].frame_label +
                                      "_joint_residual_overlay.png");
          cv::imwrite(overlay_path.string(), residual_overlay);
        }
      }
    }

    std::cout << "\nJoint measurement success: " << (joint_result.success ? 1 : 0) << "\n"
              << "Builder validation success: " << (validation_summary.success ? 1 : 0) << "\n"
              << "Used frames: " << joint_result.used_frame_count << "\n"
              << "Used board observations: " << joint_result.used_board_observation_count << "\n"
              << "Used outer points: " << joint_result.used_outer_point_count << "\n"
              << "Used internal points: " << joint_result.used_internal_point_count << "\n"
              << "Residual overall RMSE: " << residual_result.overall_rmse << "\n"
              << "Residual outer RMSE: " << residual_result.outer_only_rmse << "\n"
              << "Residual internal RMSE: " << residual_result.internal_only_rmse << "\n"
              << "Summary: " << (output_dir / "joint_measurement_summary.txt").string() << "\n"
              << "CSV: " << (output_dir / "joint_measurements.csv").string() << "\n"
              << "Validation: " << (output_dir / "joint_validation_summary.txt").string() << "\n"
              << "Residual summary: " << (output_dir / "joint_residual_summary.txt").string() << "\n"
              << "Residual CSV: " << (output_dir / "joint_residual_points.csv").string() << "\n";
    if (args.save_overlays) {
      std::cout << "Measurement overlays: " << overlay_dir.string() << "\n"
                << "Residual overlays: " << residual_overlay_dir.string() << "\n";
    }
    return (joint_result.success && validation_summary.success && residual_result.success) ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
