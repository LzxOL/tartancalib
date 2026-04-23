#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>
#include <aslam/cameras/apriltag_internal/KalibrBenchmark.hpp>
#include <aslam/cameras/apriltag_internal/Stage5Benchmark.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

namespace ati = aslam::cameras::apriltag_internal;
namespace fs = boost::filesystem;

struct CmdArgs {
  std::string config_path;
  std::string image_path;
  std::string output_path;
  std::string kalibr_camchain_yaml;
  std::string kalibr_training_split_signature;
  std::string kalibr_source_label;
  bool all = false;
  bool show = false;
  int reference_board_id = 1;
  bool optimize_intrinsics = false;
  int intrinsics_release_iteration = 3;
  int second_pass_intrinsics_release_iteration = 1;
  int holdout_stride = 5;
  int holdout_offset = 0;
  double kalibr_runtime_seconds = -1.0;
};

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
      << " [--kalibr-training-split-signature SIGNATURE]"
      << " [--kalibr-source-label LABEL] [--kalibr-runtime-seconds SEC]\n";
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const CmdArgs args = ParseArgs(argc, argv);
    const std::string dataset_label = InferDatasetLabel(args);
    const std::vector<std::string> image_paths = CollectImagePaths(args.image_path, args.all);

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
    baseline_options.reference_board_id = args.reference_board_id;
    baseline_options.optimize_intrinsics = args.optimize_intrinsics;
    baseline_options.intrinsics_release_iteration = args.intrinsics_release_iteration;
    baseline_options.run_second_pass = true;
    baseline_options.second_pass_intrinsics_release_iteration =
        args.second_pass_intrinsics_release_iteration;
    baseline_options.dataset_label = dataset_label;
    baseline_options.source_pipeline_label = "run_stage5_backend";

    ati::BackendProblemOptions backend_options;
    backend_options.reference_board_id = args.reference_board_id;
    backend_options.optimize_frame_poses = true;
    backend_options.optimize_board_poses = true;
    backend_options.optimize_intrinsics = args.optimize_intrinsics;
    backend_options.delayed_intrinsics_release = true;
    backend_options.intrinsics_release_iteration =
        args.second_pass_intrinsics_release_iteration;

    ati::KalibrBenchmarkReference kalibr_reference;
    kalibr_reference.camchain_yaml = args.kalibr_camchain_yaml;
    kalibr_reference.camera_model_family = "ds";
    kalibr_reference.training_split_signature = args.kalibr_training_split_signature;
    kalibr_reference.runtime_seconds = args.kalibr_runtime_seconds;
    kalibr_reference.source_label = args.kalibr_source_label.empty()
                                        ? fs::path(args.kalibr_camchain_yaml).stem().string()
                                        : args.kalibr_source_label;

    ati::Stage5BenchmarkInput benchmark_input;
    benchmark_input.all_frames = all_frames;
    benchmark_input.baseline_options = baseline_options;
    benchmark_input.backend_options = backend_options;
    benchmark_input.kalibr_reference = kalibr_reference;
    benchmark_input.dataset_label = dataset_label;

    ati::CalibrationBenchmarkSplitOptions split_options;
    split_options.holdout_stride = args.holdout_stride;
    split_options.holdout_offset = args.holdout_offset;
    const ati::Stage5Benchmark benchmark(split_options);
    const ati::Stage5BenchmarkReport report = benchmark.Run(benchmark_input);

    const fs::path output_dir(args.output_path);
    EnsureDirectoryExists(output_dir);

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
    ati::WriteStage5BenchmarkProtocolSummary(
        (output_dir / "benchmark_protocol_summary.txt").string(), report);
    ati::WriteStage5BenchmarkTrainingSummary(
        (output_dir / "benchmark_training_summary.txt").string(), report);
    ati::WriteStage5BenchmarkHoldoutSummary(
        (output_dir / "benchmark_holdout_summary.txt").string(), report);
    ati::WriteStage5BenchmarkHoldoutPointsCsv(
        (output_dir / "benchmark_holdout_points.csv").string(), report);
    ati::WriteStage5BenchmarkWorstCasesSummary(
        (output_dir / "benchmark_worst_cases_summary.txt").string(), report, 10);
    if (report.diagnostic_compare.success) {
      ati::WriteKalibrBenchmarkIntrinsicsCsv(
          (output_dir / "benchmark_intrinsics_compare.csv").string(),
          report.diagnostic_compare);
    }

    const cv::Mat projection_compare = benchmark.RenderProjectionComparison(report);
    if (!projection_compare.empty()) {
      cv::imwrite((output_dir / "benchmark_projection_compare.png").string(),
                  projection_compare);
      if (args.show) {
        cv::imshow("stage5_benchmark_projection_compare", projection_compare);
        cv::waitKey(0);
      }
    }

    if (!report.success) {
      std::cout << "Stage 5 benchmark success: 0\n"
                << "Protocol summary: "
                << (output_dir / "benchmark_protocol_summary.txt").string() << "\n";
      return 1;
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
    const ati::AslamBackendCalibrationRunner backend_runner(runner_options);
    const ati::AslamBackendCalibrationResult backend_result =
        backend_runner.Run(report.backend_problem_input);
    ati::WriteAslamBackendCalibrationSummary(
        (output_dir / "backend_optimization_summary.txt").string(), backend_result);

    if (!backend_result.success) {
      std::cout << "Backend summary: "
                << (output_dir / "backend_optimization_summary.txt").string() << "\n";
      return 1;
    }

    const ati::CameraModelRefitEvaluationResult backend_training_evaluation =
        benchmark.EvaluateCameraModel(report.training_dataset,
                                      backend_result.optimized_scene_state.camera,
                                      "backend");
    const ati::CameraModelRefitEvaluationResult backend_holdout_evaluation =
        benchmark.EvaluateCameraModel(report.holdout_dataset,
                                      backend_result.optimized_scene_state.camera,
                                      "backend");
    if (!backend_training_evaluation.success || !backend_holdout_evaluation.success) {
      std::ofstream output((output_dir / "backend_vs_frontend_summary.txt").string().c_str());
      output << "backend_evaluation_failed: 1\n";
      output << "training_failure_reason: " << backend_training_evaluation.failure_reason << "\n";
      output << "holdout_failure_reason: " << backend_holdout_evaluation.failure_reason << "\n";
      return 1;
    }

    WriteEvaluationSummary(
        (output_dir / "backend_training_summary.txt").string(),
        backend_training_evaluation);
    WriteEvaluationSummary(
        (output_dir / "backend_holdout_summary.txt").string(),
        backend_holdout_evaluation);
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
              << "Backend compare frame overlays: " << compare_frame_dir.string() << "\n"
              << "Backend compare board overlays: " << compare_board_dir.string() << "\n"
              << "Backend compare outer-pose frames: "
              << compare_outer_frame_dir.string() << "\n"
              << "Backend compare outer-pose boards: "
              << compare_outer_board_dir.string() << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
