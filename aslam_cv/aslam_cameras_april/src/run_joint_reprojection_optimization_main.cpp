#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>
#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>
#include <aslam/cameras/apriltag_internal/JointMeasurementSelection.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionMeasurementBuilder.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionOptimizer.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionResidualEvaluator.hpp>
#include <aslam/cameras/apriltag_internal/KalibrBenchmark.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardOuterBootstrap.hpp>
#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

#include <algorithm>
#include <cctype>
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
  bool optimize_intrinsics = false;
  int intrinsics_release_iteration = 3;
  bool run_second_pass = false;
  int second_pass_intrinsics_release_iteration = 1;
  std::string benchmark_kalibr_camchain;
};

struct FrameRecord {
  std::string image_path;
  ati::OuterBootstrapFrameInput bootstrap_input;
  ati::InternalRegenerationFrameInput regeneration_input;
};

using BuilderValidationSummary = ati::JointMeasurementBuildValidationSummary;

void WriteStage42PolicyNotes(std::ostream& output) {
  output << "note: Stage 4.2 treats edge-board outer residuals at extreme fisheye "
            "boundaries as secondary diagnostics.\n";
  output << "note: Stage 4.2 acceptance emphasizes overall_rmse, internal_only_rmse, "
            "and round1_vs_round2 trend rather than edge-board outer-only outliers.\n";
}

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --image IMAGE_OR_DIR --config APRILTAG_INTERNAL_YAML --output OUTPUT_DIR"
      << " [--all] [--show] [--no-save-overlays] [--reference-board-id ID]"
      << " [--optimize-intrinsics] [--intrinsics-release-iteration N]"
      << " [--run-second-pass] [--second-pass-intrinsics-release-iteration N]"
      << " [--benchmark-kalibr-camchain CAMCHAIN_YAML]\n";
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
    } else if (token == "--optimize-intrinsics") {
      args.optimize_intrinsics = true;
    } else if (token == "--intrinsics-release-iteration" && i + 1 < argc) {
      args.intrinsics_release_iteration = std::stoi(argv[++i]);
    } else if (token == "--run-second-pass") {
      args.run_second_pass = true;
    } else if (token == "--second-pass-intrinsics-release-iteration" && i + 1 < argc) {
      args.second_pass_intrinsics_release_iteration = std::stoi(argv[++i]);
    } else if (token == "--benchmark-kalibr-camchain" && i + 1 < argc) {
      args.benchmark_kalibr_camchain = argv[++i];
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
  WriteStage42PolicyNotes(output);
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
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
    for (const ati::JointBoardObservation& board_observation : frame_result.board_observations) {
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
  output << "reference_board_id: " << result.reference_board_id << "\n";
  output << "overall_rmse: " << result.overall_rmse << "\n";
  output << "outer_only_rmse: " << result.outer_only_rmse << "\n";
  output << "internal_only_rmse: " << result.internal_only_rmse << "\n";
  output << "point_count: " << result.point_diagnostics.size() << "\n";
  output << "frame_count: " << result.frame_diagnostics.size() << "\n";
  output << "board_count: " << result.board_diagnostics.size() << "\n";
  output << "board_observation_count: " << result.board_observation_diagnostics.size() << "\n";
  WriteStage42PolicyNotes(output);
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
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

void WriteSelectionSummary(const fs::path& output_path,
                           const ati::JointMeasurementSelectionResult& result) {
  std::ofstream output(output_path.string());
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "reference_board_id: " << result.reference_board_id << "\n";
  output << "accepted_frame_count: " << result.accepted_frame_count << "\n";
  output << "accepted_board_observation_count: "
         << result.accepted_board_observation_count << "\n";
  output << "accepted_outer_point_count: " << result.accepted_outer_point_count << "\n";
  output << "accepted_internal_point_count: " << result.accepted_internal_point_count << "\n";
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
  output << "frame_decisions:\n";
  for (const ati::JointFrameSelectionDecision& decision : result.frame_decisions) {
    output << "  frame=" << decision.frame_index
           << " label=" << decision.frame_label
           << " accepted=" << (decision.accepted ? 1 : 0)
           << " usable_boards=";
    for (std::size_t i = 0; i < decision.usable_board_ids.size(); ++i) {
      if (i > 0) {
        output << ",";
      }
      output << decision.usable_board_ids[i];
    }
    output << " accepted_boards=";
    for (std::size_t i = 0; i < decision.accepted_board_ids.size(); ++i) {
      if (i > 0) {
        output << ",";
      }
      output << decision.accepted_board_ids[i];
    }
    output << " reason_detail=" << decision.reason_detail << "\n";
  }
  output << "board_observation_decisions:\n";
  for (const ati::JointBoardObservationSelectionDecision& decision :
       result.board_observation_decisions) {
    output << "  frame=" << decision.frame_index
           << " board=" << decision.board_id
           << " accepted=" << (decision.accepted ? 1 : 0)
           << " reason=" << ati::ToString(decision.reason_code)
           << " rmse=" << decision.rmse
           << " pose_fit_outer_rmse=" << decision.pose_fit_outer_rmse
           << " points=" << decision.point_count
           << " coverage_signature=" << decision.coverage_signature
           << " detail=" << decision.reason_detail << "\n";
  }
}

void WriteOptimizationSummary(const fs::path& output_path,
                              const ati::JointOptimizationResult& result) {
  std::ofstream output(output_path.string());
  const auto write_intrinsics = [&output](const std::string& prefix,
                                          const ati::OuterBootstrapCameraIntrinsics& intrinsics) {
    output << prefix << "_xi: " << intrinsics.xi << "\n";
    output << prefix << "_alpha: " << intrinsics.alpha << "\n";
    output << prefix << "_fu: " << intrinsics.fu << "\n";
    output << prefix << "_fv: " << intrinsics.fv << "\n";
    output << prefix << "_cu: " << intrinsics.cu << "\n";
    output << prefix << "_cv: " << intrinsics.cv << "\n";
  };
  int accepted_intrinsics_updates = 0;
  int rejected_intrinsics_updates = 0;
  int attempted_intrinsics_updates = 0;
  for (const ati::JointOptimizationIterationSummary& iteration : result.iterations) {
    attempted_intrinsics_updates += iteration.attempted_intrinsics_update ? 1 : 0;
    accepted_intrinsics_updates += iteration.accepted_intrinsics_update ? 1 : 0;
    rejected_intrinsics_updates += iteration.rejected_intrinsics_update ? 1 : 0;
  }

  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "reference_board_id: " << result.reference_board_id << "\n";
  output << "optimize_intrinsics: " << (result.optimize_intrinsics ? 1 : 0) << "\n";
  output << "intrinsics_release_iteration: " << result.intrinsics_release_iteration << "\n";
  output << "selected_frame_count: " << result.selection_result.accepted_frame_count << "\n";
  output << "selected_board_observation_count: "
         << result.selection_result.accepted_board_observation_count << "\n";
  output << "initial_overall_rmse: " << result.initial_residual.overall_rmse << "\n";
  output << "optimized_overall_rmse: " << result.optimized_residual.overall_rmse << "\n";
  output << "initial_outer_rmse: " << result.initial_residual.outer_only_rmse << "\n";
  output << "optimized_outer_rmse: " << result.optimized_residual.outer_only_rmse << "\n";
  output << "initial_internal_rmse: " << result.initial_residual.internal_only_rmse << "\n";
  output << "optimized_internal_rmse: " << result.optimized_residual.internal_only_rmse << "\n";
  output << "iteration_count: " << result.iterations.size() << "\n";
  output << "attempted_intrinsics_update_count: " << attempted_intrinsics_updates << "\n";
  output << "accepted_intrinsics_update_count: " << accepted_intrinsics_updates << "\n";
  output << "rejected_intrinsics_update_count: " << rejected_intrinsics_updates << "\n";
  write_intrinsics("initial", result.initial_state.camera);
  write_intrinsics("optimized", result.optimized_state.camera);
  WriteStage42PolicyNotes(output);
  for (const std::string& warning : result.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteOptimizationIterationsCsv(const fs::path& output_path,
                                    const ati::JointOptimizationResult& result) {
  std::ofstream output(output_path.string());
  output << "iteration_index,cost_before,cost_after,cost_delta,overall_rmse_before,"
         << "overall_rmse_after,outer_rmse_after,internal_rmse_after,"
         << "accepted_frame_updates,rejected_frame_updates,accepted_board_updates,"
         << "rejected_board_updates,attempted_intrinsics_update,accepted_intrinsics_update,"
         << "rejected_intrinsics_update,intrinsics_step_norm,"
         << "max_parameter_delta\n";
  for (const ati::JointOptimizationIterationSummary& summary : result.iterations) {
    output << summary.iteration_index << ","
           << summary.cost_before << ","
           << summary.cost_after << ","
           << summary.cost_delta << ","
           << summary.overall_rmse_before << ","
           << summary.overall_rmse_after << ","
           << summary.outer_rmse_after << ","
           << summary.internal_rmse_after << ","
           << summary.accepted_frame_updates << ","
           << summary.rejected_frame_updates << ","
           << summary.accepted_board_updates << ","
           << summary.rejected_board_updates << ","
           << (summary.attempted_intrinsics_update ? 1 : 0) << ","
           << (summary.accepted_intrinsics_update ? 1 : 0) << ","
           << (summary.rejected_intrinsics_update ? 1 : 0) << ","
           << summary.intrinsics_step_norm << ","
           << summary.max_parameter_delta << "\n";
  }
}

void WriteRoundComparisonSummary(const fs::path& output_path,
                                 const ati::JointOptimizationResult& round1_result,
                                 const ati::JointMeasurementBuildResult& round2_measurement,
                                 const ati::JointMeasurementSelectionResult& round2_selection,
                                 const ati::JointOptimizationResult& round2_result) {
  std::ofstream output(output_path.string());
  const bool round2_non_degrading_overall =
      round2_result.optimized_residual.overall_rmse <=
      round1_result.optimized_residual.overall_rmse;
  const bool round2_non_degrading_internal =
      round2_result.optimized_residual.internal_only_rmse <=
      round1_result.optimized_residual.internal_only_rmse;
  const bool round2_non_degrading_outer =
      round2_result.optimized_residual.outer_only_rmse <=
      round1_result.optimized_residual.outer_only_rmse;

  output << "reference_board_id: " << round2_result.reference_board_id << "\n";
  output << "edge_outer_policy: diagnostic_only\n";
  output << "edge_outer_blocking_applied: 0\n";
  output << "round1_selected_frame_count: "
         << round1_result.selection_result.accepted_frame_count << "\n";
  output << "round2_selected_frame_count: "
         << round2_selection.accepted_frame_count << "\n";
  output << "round1_selected_board_observation_count: "
         << round1_result.selection_result.accepted_board_observation_count << "\n";
  output << "round2_selected_board_observation_count: "
         << round2_selection.accepted_board_observation_count << "\n";
  output << "round1_accepted_internal_point_count: "
         << round1_result.selection_result.accepted_internal_point_count << "\n";
  output << "round2_accepted_internal_point_count: "
         << round2_selection.accepted_internal_point_count << "\n";
  output << "round2_measurement_internal_point_count: "
         << round2_measurement.used_internal_point_count << "\n";
  output << "round1_overall_rmse: " << round1_result.optimized_residual.overall_rmse << "\n";
  output << "round2_overall_rmse: " << round2_result.optimized_residual.overall_rmse << "\n";
  output << "round1_outer_rmse: " << round1_result.optimized_residual.outer_only_rmse << "\n";
  output << "round2_outer_rmse: " << round2_result.optimized_residual.outer_only_rmse << "\n";
  output << "round1_internal_rmse: " << round1_result.optimized_residual.internal_only_rmse << "\n";
  output << "round2_internal_rmse: " << round2_result.optimized_residual.internal_only_rmse << "\n";
  output << "round2_delta_overall_rmse: "
         << (round1_result.optimized_residual.overall_rmse -
             round2_result.optimized_residual.overall_rmse) << "\n";
  output << "round2_delta_outer_rmse: "
         << (round1_result.optimized_residual.outer_only_rmse -
             round2_result.optimized_residual.outer_only_rmse) << "\n";
  output << "round2_delta_internal_rmse: "
         << (round1_result.optimized_residual.internal_only_rmse -
             round2_result.optimized_residual.internal_only_rmse) << "\n";
  output << "round2_non_degrading_overall: " << (round2_non_degrading_overall ? 1 : 0) << "\n";
  output << "round2_non_degrading_internal: " << (round2_non_degrading_internal ? 1 : 0) << "\n";
  output << "round2_non_degrading_outer: " << (round2_non_degrading_outer ? 1 : 0) << "\n";
  output << "round2_positive_trend: "
         << ((round2_non_degrading_overall && round2_non_degrading_internal) ? 1 : 0) << "\n";
  WriteStage42PolicyNotes(output);
}

void WriteStage42ValidationSummary(const fs::path& output_path,
                                   int input_frame_count,
                                   const ati::JointMeasurementSelectionResult& round1_selection,
                                   const ati::JointOptimizationResult& round1_result,
                                   bool round2_available,
                                   const ati::JointMeasurementSelectionResult* round2_selection,
                                   const ati::JointOptimizationResult* round2_result) {
  std::ofstream output(output_path.string());
  output << "success: 1\n";
  output << "reference_board_id: " << round1_result.reference_board_id << "\n";
  output << "input_frame_count: " << input_frame_count << "\n";
  output << "round2_available: " << (round2_available ? 1 : 0) << "\n";
  output << "edge_outer_policy: diagnostic_only\n";
  output << "edge_outer_blocking_applied: 0\n";
  output << "primary_acceptance_signal: overall_rmse_and_internal_only_rmse_trend\n";
  output << "round1_selected_frame_count: " << round1_selection.accepted_frame_count << "\n";
  output << "round1_selected_board_observation_count: "
         << round1_selection.accepted_board_observation_count << "\n";
  output << "round1_accepted_internal_point_count: "
         << round1_selection.accepted_internal_point_count << "\n";
  output << "round1_overall_rmse: " << round1_result.optimized_residual.overall_rmse << "\n";
  output << "round1_outer_only_rmse: "
         << round1_result.optimized_residual.outer_only_rmse << "\n";
  output << "round1_internal_only_rmse: "
         << round1_result.optimized_residual.internal_only_rmse << "\n";

  if (!round2_available || round2_selection == NULL || round2_result == NULL) {
    output << "stage42_validation_pass: 0\n";
    output << "warning: Stage 4.2 validation requires --run-second-pass to compare "
              "round1 vs round2.\n";
    WriteStage42PolicyNotes(output);
    return;
  }

  const bool round2_non_degrading_overall =
      round2_result->optimized_residual.overall_rmse <=
      round1_result.optimized_residual.overall_rmse;
  const bool round2_non_degrading_internal =
      round2_result->optimized_residual.internal_only_rmse <=
      round1_result.optimized_residual.internal_only_rmse;
  const bool round2_non_degrading_outer =
      round2_result->optimized_residual.outer_only_rmse <=
      round1_result.optimized_residual.outer_only_rmse;
  const bool selected_data_present =
      round2_selection->accepted_frame_count > 0 &&
      round2_selection->accepted_board_observation_count > 0;
  const bool stage42_validation_pass =
      round2_non_degrading_overall && round2_non_degrading_internal &&
      selected_data_present;

  output << "round2_selected_frame_count: " << round2_selection->accepted_frame_count << "\n";
  output << "round2_selected_board_observation_count: "
         << round2_selection->accepted_board_observation_count << "\n";
  output << "round2_accepted_internal_point_count: "
         << round2_selection->accepted_internal_point_count << "\n";
  output << "round2_overall_rmse: " << round2_result->optimized_residual.overall_rmse << "\n";
  output << "round2_outer_only_rmse: "
         << round2_result->optimized_residual.outer_only_rmse << "\n";
  output << "round2_internal_only_rmse: "
         << round2_result->optimized_residual.internal_only_rmse << "\n";
  output << "round2_delta_selected_frame_count: "
         << (round2_selection->accepted_frame_count -
             round1_selection.accepted_frame_count) << "\n";
  output << "round2_delta_selected_board_observation_count: "
         << (round2_selection->accepted_board_observation_count -
             round1_selection.accepted_board_observation_count) << "\n";
  output << "round2_delta_accepted_internal_point_count: "
         << (round2_selection->accepted_internal_point_count -
             round1_selection.accepted_internal_point_count) << "\n";
  output << "round2_delta_overall_rmse: "
         << (round1_result.optimized_residual.overall_rmse -
             round2_result->optimized_residual.overall_rmse) << "\n";
  output << "round2_delta_outer_only_rmse: "
         << (round1_result.optimized_residual.outer_only_rmse -
             round2_result->optimized_residual.outer_only_rmse) << "\n";
  output << "round2_delta_internal_only_rmse: "
         << (round1_result.optimized_residual.internal_only_rmse -
             round2_result->optimized_residual.internal_only_rmse) << "\n";
  output << "round2_non_degrading_overall: " << (round2_non_degrading_overall ? 1 : 0) << "\n";
  output << "round2_non_degrading_internal: " << (round2_non_degrading_internal ? 1 : 0) << "\n";
  output << "round2_non_degrading_outer: " << (round2_non_degrading_outer ? 1 : 0) << "\n";
  output << "selected_data_present: " << (selected_data_present ? 1 : 0) << "\n";
  output << "stage42_validation_pass: " << (stage42_validation_pass ? 1 : 0) << "\n";
  if (!round2_non_degrading_overall) {
    output << "warning: round2 worsened overall_rmse relative to round1.\n";
  }
  if (!round2_non_degrading_internal) {
    output << "warning: round2 worsened internal_only_rmse relative to round1.\n";
  }
  if (!selected_data_present) {
    output << "warning: round2 selection produced no accepted frames or board observations.\n";
  }
  WriteStage42PolicyNotes(output);
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
    const ati::MultiBoardInternalMeasurementRegenerator regenerator(config, detection_options);
    ati::JointMeasurementBuildOptions build_options;
    build_options.reference_board_id = args.reference_board_id;
    const ati::JointReprojectionMeasurementBuilder builder(config, build_options);
    ati::JointResidualEvaluationOptions residual_options;
    const ati::JointReprojectionResidualEvaluator residual_evaluator(residual_options);

    const std::vector<std::string> image_paths = CollectImagePaths(args.image_path, args.all);
    const fs::path output_dir(args.output_path);
    EnsureDirectoryExists(output_dir);
    const fs::path overlay_dir = output_dir / "joint_optimization_overlays";
    const fs::path round2_overlay_dir = output_dir / "joint_optimization_overlays_round2";
    if (args.save_overlays) {
      EnsureDirectoryExists(overlay_dir);
      if (args.run_second_pass) {
        EnsureDirectoryExists(round2_overlay_dir);
      }
    }

    const std::string dataset_label = InferDatasetLabel(args);
    std::vector<ati::FrozenRound2BaselineFrameSource> frame_sources;
    frame_sources.reserve(image_paths.size());
    for (std::size_t image_index = 0; image_index < image_paths.size(); ++image_index) {
      ati::FrozenRound2BaselineFrameSource frame_source;
      frame_source.frame_index = static_cast<int>(image_index);
      frame_source.image_path = image_paths[image_index];
      frame_source.frame_label = fs::path(image_paths[image_index]).stem().string();
      frame_sources.push_back(frame_source);
    }

    ati::FrozenRound2BaselineOptions baseline_options;
    baseline_options.config = config;
    baseline_options.reference_board_id = args.reference_board_id;
    baseline_options.optimize_intrinsics = args.optimize_intrinsics;
    baseline_options.intrinsics_release_iteration = args.intrinsics_release_iteration;
    baseline_options.run_second_pass = args.run_second_pass;
    baseline_options.second_pass_intrinsics_release_iteration =
        args.second_pass_intrinsics_release_iteration;
    baseline_options.dataset_label = dataset_label;
    baseline_options.source_pipeline_label = "run_joint_reprojection_optimization";

    const ati::FrozenRound2BaselinePipeline baseline_pipeline(baseline_options);
    const ati::FrozenRound2BaselineResult baseline_result =
        baseline_pipeline.Run(frame_sources);
    if (!baseline_result.success) {
      WriteJointSummary(output_dir / "joint_measurement_summary.txt",
                        ati::JointMeasurementBuildResult{});
      std::cout << "Frozen round2 baseline failed.\n";
      return 1;
    }

    const ati::OuterBootstrapResult& bootstrap_result = baseline_result.bootstrap_result;
    const ati::FrozenRoundArtifacts& round1 = baseline_result.round1;
    const ati::FrozenRoundArtifacts& round2 = baseline_result.round2;
    const ati::JointMeasurementBuildResult& joint_result = round1.measurement_result;
    const BuilderValidationSummary& validation_summary = round1.validation_summary;
    const ati::JointResidualEvaluationResult& residual_result = round1.residual_result;
    const ati::JointMeasurementSelectionResult& selection_result = round1.selection_result;
    const ati::JointOptimizationResult& optimization_result = round1.optimization_result;
    const std::vector<ati::JointMeasurementFrameInput>& joint_inputs = round1.joint_inputs;
    const std::vector<ati::JointMeasurementFrameInput>& round2_joint_inputs = round2.joint_inputs;
    const ati::JointMeasurementBuildResult& round2_joint_result = round2.measurement_result;
    const BuilderValidationSummary& round2_validation_summary = round2.validation_summary;
    const ati::JointResidualEvaluationResult& round2_residual_result = round2.residual_result;
    const ati::JointMeasurementSelectionResult& round2_selection_result =
        round2.selection_result;
    const ati::JointOptimizationResult& round2_optimization_result =
        round2.optimization_result;
    const ati::CalibrationStateBundle& stage5_round1_bundle =
        baseline_result.stage5_round1_bundle;
    const ati::CalibrationStateBundle& stage5_bundle =
        baseline_result.final_stage5_bundle;
    const bool stage5_bundle_available = baseline_result.stage5_bundle_available;

    WriteJointSummary(output_dir / "joint_measurement_summary.txt", joint_result);
    WriteValidationSummary(output_dir / "joint_validation_summary.txt", validation_summary);
    WriteResidualSummary(output_dir / "joint_residual_summary.txt", residual_result);
    WriteResidualPointsCsv(output_dir / "joint_residual_points.csv", residual_result);
    WriteSelectionSummary(output_dir / "joint_selection_summary.txt", selection_result);
    WriteOptimizationSummary(output_dir / "joint_optimization_summary.txt", optimization_result);
    WriteOptimizationIterationsCsv(output_dir / "joint_optimization_iterations.csv",
                                   optimization_result);
    WriteResidualSummary(output_dir / "joint_optimized_residual_summary.txt",
                         optimization_result.optimized_residual);
    WriteResidualPointsCsv(output_dir / "joint_optimized_residual_points.csv",
                           optimization_result.optimized_residual);

    if (baseline_result.round2_available) {
      WriteJointSummary(output_dir / "round2_joint_measurement_summary.txt", round2_joint_result);
      WriteValidationSummary(output_dir / "round2_joint_validation_summary.txt",
                             round2_validation_summary);
      WriteResidualSummary(output_dir / "round2_joint_residual_summary.txt",
                           round2_residual_result);
      WriteResidualPointsCsv(output_dir / "round2_joint_residual_points.csv",
                             round2_residual_result);
      WriteSelectionSummary(output_dir / "round2_joint_selection_summary.txt",
                            round2_selection_result);
      WriteOptimizationSummary(output_dir / "round2_joint_optimization_summary.txt",
                               round2_optimization_result);
      WriteOptimizationIterationsCsv(output_dir / "round2_joint_optimization_iterations.csv",
                                     round2_optimization_result);
      WriteResidualSummary(output_dir / "round2_joint_optimized_residual_summary.txt",
                           round2_optimization_result.optimized_residual);
      WriteResidualPointsCsv(output_dir / "round2_joint_optimized_residual_points.csv",
                             round2_optimization_result.optimized_residual);
      WriteRoundComparisonSummary(output_dir / "round1_vs_round2_summary.txt",
                                  optimization_result,
                                  round2_joint_result,
                                  round2_selection_result,
                                  round2_optimization_result);
    }

    WriteStage42ValidationSummary(output_dir / "stage42_validation_summary.txt",
                                  static_cast<int>(frame_sources.size()),
                                  selection_result,
                                  optimization_result,
                                  baseline_result.round2_available,
                                  baseline_result.round2_available
                                      ? &round2_selection_result
                                      : NULL,
                                  baseline_result.round2_available
                                      ? &round2_optimization_result
                                      : NULL);

    ati::WriteCalibrationStateBundleSummary(
        (output_dir / "stage5_round1_bundle_summary.txt").string(), stage5_round1_bundle);
    if (baseline_result.round2_available) {
      ati::WriteCalibrationStateBundleSummary(
          (output_dir / "stage5_bundle_summary.txt").string(), stage5_bundle);
    }

    ati::BackendProblemOptions backend_options;
    backend_options.reference_board_id = stage5_bundle.scene_state.reference_board_id;
    backend_options.optimize_frame_poses = true;
    backend_options.optimize_board_poses = true;
    backend_options.optimize_intrinsics = args.optimize_intrinsics;
    backend_options.delayed_intrinsics_release = true;
    backend_options.intrinsics_release_iteration =
        baseline_result.round2_available ? args.second_pass_intrinsics_release_iteration
                                         : args.intrinsics_release_iteration;
    if (stage5_bundle_available) {
      const ati::CalibrationBackendProblemInput backend_input =
          ati::BuildBackendProblemInput(stage5_bundle, backend_options);
      ati::WriteCalibrationBackendProblemSummary(
          (output_dir / "stage5_backend_problem_summary.txt").string(), backend_input);
    }

    if (stage5_bundle_available && !args.benchmark_kalibr_camchain.empty()) {
      ati::KalibrBenchmarkInput benchmark_input;
      benchmark_input.dataset_label = dataset_label;
      benchmark_input.kalibr_camchain_yaml = args.benchmark_kalibr_camchain;
      benchmark_input.our_bundle = stage5_bundle;
      const ati::KalibrBenchmark benchmark;
      const ati::KalibrBenchmarkReport benchmark_report = benchmark.Compare(benchmark_input);
      ati::WriteKalibrBenchmarkSummary((output_dir / "benchmark_summary.txt").string(),
                                       benchmark_report);
      ati::WriteKalibrBenchmarkIntrinsicsCsv(
          (output_dir / "benchmark_intrinsics_compare.csv").string(), benchmark_report);
      ati::WriteKalibrBenchmarkResidualSummary(
          (output_dir / "benchmark_residual_compare.txt").string(), benchmark_report);
      const cv::Mat benchmark_projection_compare =
          benchmark.RenderProjectionComparison(benchmark_report);
      if (!benchmark_projection_compare.empty()) {
        cv::imwrite((output_dir / "benchmark_projection_compare.png").string(),
                    benchmark_projection_compare);
      }
    }

    if (args.save_overlays || args.show) {
      for (std::size_t frame_index = 0; frame_index < joint_inputs.size(); ++frame_index) {
        const cv::Mat image =
            cv::imread(frame_sources[frame_index].image_path, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
          throw std::runtime_error("Failed to read image: " + frame_sources[frame_index].image_path);
        }

        cv::Mat detection_style_overlay;
        regenerator.DrawFrameOverlay(image, joint_inputs[frame_index].regenerated_internal,
                                     &detection_style_overlay);

        cv::Mat measurement_overlay;
        builder.DrawFrameOverlay(
            detection_style_overlay, selection_result.selected_measurement_result.frames[frame_index],
            &measurement_overlay);
        if (args.save_overlays) {
          cv::imwrite((overlay_dir / (joint_inputs[frame_index].frame_label +
                                      "_selected_measurement_overlay.png")).string(),
                      measurement_overlay);
        }

        cv::Mat fixed_residual_overlay;
        residual_evaluator.DrawFrameOverlay(detection_style_overlay,
                                            joint_inputs[frame_index].frame_index,
                                            residual_result, &fixed_residual_overlay);
        if (args.save_overlays) {
          cv::imwrite((overlay_dir / (joint_inputs[frame_index].frame_label +
                                      "_fixed_residual_overlay.png")).string(),
                      fixed_residual_overlay);
        }

        cv::Mat optimized_residual_overlay;
        residual_evaluator.DrawFrameOverlay(detection_style_overlay,
                                            joint_inputs[frame_index].frame_index,
                                            optimization_result.optimized_residual,
                                            &optimized_residual_overlay);
        if (args.save_overlays) {
          cv::imwrite((overlay_dir / (joint_inputs[frame_index].frame_label +
                                      "_optimized_residual_overlay.png")).string(),
                      optimized_residual_overlay);
        }

        if (args.show) {
          cv::imshow("joint_reprojection_optimization", optimized_residual_overlay);
          cv::waitKey(1);
        }
      }

      if (args.run_second_pass) {
        for (std::size_t frame_index = 0; frame_index < round2_joint_inputs.size(); ++frame_index) {
          const cv::Mat image =
              cv::imread(frame_sources[frame_index].image_path, cv::IMREAD_UNCHANGED);
          if (image.empty()) {
            throw std::runtime_error("Failed to read image: " + frame_sources[frame_index].image_path);
          }

          cv::Mat detection_style_overlay;
          regenerator.DrawFrameOverlay(image, round2_joint_inputs[frame_index].regenerated_internal,
                                       &detection_style_overlay);

          cv::Mat measurement_overlay;
          builder.DrawFrameOverlay(
              detection_style_overlay,
              round2_selection_result.selected_measurement_result.frames[frame_index],
              &measurement_overlay);
          if (args.save_overlays) {
            cv::imwrite((round2_overlay_dir / (round2_joint_inputs[frame_index].frame_label +
                                               "_selected_measurement_overlay.png")).string(),
                        measurement_overlay);
          }

          cv::Mat fixed_residual_overlay;
          residual_evaluator.DrawFrameOverlay(detection_style_overlay,
                                              round2_joint_inputs[frame_index].frame_index,
                                              round2_residual_result,
                                              &fixed_residual_overlay);
          if (args.save_overlays) {
            cv::imwrite((round2_overlay_dir / (round2_joint_inputs[frame_index].frame_label +
                                               "_fixed_residual_overlay.png")).string(),
                        fixed_residual_overlay);
          }

          cv::Mat optimized_residual_overlay;
          residual_evaluator.DrawFrameOverlay(detection_style_overlay,
                                              round2_joint_inputs[frame_index].frame_index,
                                              round2_optimization_result.optimized_residual,
                                              &optimized_residual_overlay);
          if (args.save_overlays) {
            cv::imwrite((round2_overlay_dir / (round2_joint_inputs[frame_index].frame_label +
                                               "_optimized_residual_overlay.png")).string(),
                        optimized_residual_overlay);
          }

          if (args.show) {
            cv::imshow("joint_reprojection_optimization_round2", optimized_residual_overlay);
            cv::waitKey(1);
          }
        }
      }
    }

    std::cout << "\nJoint optimization success: " << (optimization_result.success ? 1 : 0) << "\n"
              << "Reference board id: " << optimization_result.reference_board_id << "\n"
              << "Selection accepted frames: " << selection_result.accepted_frame_count << "\n"
              << "Selection accepted board observations: "
              << selection_result.accepted_board_observation_count << "\n"
              << "Initial residual RMSE: " << optimization_result.initial_residual.overall_rmse << "\n"
              << "Optimized residual RMSE: "
              << optimization_result.optimized_residual.overall_rmse << "\n"
              << "Measurement summary: "
              << (output_dir / "joint_measurement_summary.txt").string() << "\n"
              << "Validation summary: "
              << (output_dir / "joint_validation_summary.txt").string() << "\n"
              << "Fixed residual summary: "
              << (output_dir / "joint_residual_summary.txt").string() << "\n"
              << "Selection summary: "
              << (output_dir / "joint_selection_summary.txt").string() << "\n"
              << "Stage 4.2 validation summary: "
              << (output_dir / "stage42_validation_summary.txt").string() << "\n"
              << "Optimization summary: "
              << (output_dir / "joint_optimization_summary.txt").string() << "\n"
              << "Optimized residual summary: "
              << (output_dir / "joint_optimized_residual_summary.txt").string() << "\n";
    std::cout << "Stage 5 round-1 bundle summary: "
              << (output_dir / "stage5_round1_bundle_summary.txt").string() << "\n";
    if (stage5_bundle_available) {
      std::cout << "Stage 5 backend problem summary: "
                << (output_dir / "stage5_backend_problem_summary.txt").string() << "\n";
    }
    if (baseline_result.round2_available) {
      std::cout << "Round-2 optimized residual RMSE: "
                << round2_optimization_result.optimized_residual.overall_rmse << "\n"
                << "Round-2 measurement summary: "
                << (output_dir / "round2_joint_measurement_summary.txt").string() << "\n"
                << "Round-2 optimization summary: "
                << (output_dir / "round2_joint_optimization_summary.txt").string() << "\n"
                << "Round-1 vs Round-2 summary: "
                << (output_dir / "round1_vs_round2_summary.txt").string() << "\n";
      std::cout << "Stage 5 final bundle summary: "
                << (output_dir / "stage5_bundle_summary.txt").string() << "\n";
    }
    if (!args.benchmark_kalibr_camchain.empty()) {
      std::cout << "Stage 5 Kalibr benchmark summary: "
                << (output_dir / "benchmark_summary.txt").string() << "\n";
    }
    if (args.save_overlays) {
      std::cout << "Optimization overlays: " << overlay_dir.string() << "\n";
      if (baseline_result.round2_available) {
        std::cout << "Round-2 optimization overlays: " << round2_overlay_dir.string() << "\n";
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
