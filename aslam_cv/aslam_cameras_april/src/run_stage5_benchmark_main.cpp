#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>
#include <aslam/cameras/apriltag_internal/KalibrBenchmark.hpp>
#include <aslam/cameras/apriltag_internal/Stage5Benchmark.hpp>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
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

cv::Mat RenderSideBySideCompare(const cv::Mat& left,
                                const cv::Mat& right,
                                const std::string& left_label,
                                const std::string& right_label) {
  if (left.empty() || right.empty()) {
    return cv::Mat();
  }

  const int target_height = std::max(left.rows, right.rows);
  const int banner_height = 30;
  const auto pad_and_label = [target_height, banner_height](const cv::Mat& image,
                                                            const std::string& label) {
    cv::Mat padded;
    const int bottom = std::max(0, target_height - image.rows);
    cv::copyMakeBorder(image, padded, banner_height, bottom, 0, 0, cv::BORDER_CONSTANT,
                       cv::Scalar(24, 24, 24));
    cv::putText(padded, label, cv::Point(12, 21), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
    return padded;
  };

  const cv::Mat left_labeled = pad_and_label(left, left_label);
  const cv::Mat right_labeled = pad_and_label(right, right_label);
  cv::Mat compare;
  cv::hconcat(std::vector<cv::Mat>{left_labeled, right_labeled}, compare);
  return compare;
}

std::string FindImagePathForFrame(
    const std::vector<ati::FrozenRound2BaselineFrameSource>& frames,
    int frame_index) {
  for (const ati::FrozenRound2BaselineFrameSource& frame : frames) {
    if (frame.frame_index == frame_index) {
      return frame.image_path;
    }
  }
  return std::string();
}

cv::Rect ClampRectToImage(const cv::Rect& rect, const cv::Size& image_size) {
  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  return rect & image_rect;
}

cv::Mat RenderAcceptedBundleFrameOverlay(
    const ati::CalibrationMeasurementDataset& dataset,
    const ati::JointMeasurementFrameResult& frame_result,
    const cv::Mat& image) {
  if (image.empty()) {
    return cv::Mat();
  }

  cv::Mat output;
  if (image.channels() == 1) {
    cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
  } else {
    output = image.clone();
  }

  int accepted_board_count = 0;
  int accepted_outer = 0;
  int accepted_internal = 0;
  for (const ati::JointBoardObservation& board : frame_result.board_observations) {
    const bool board_accepted =
        dataset.accepted_board_observation_keys.count(
            std::make_pair(frame_result.frame_index, board.board_id)) > 0;
    if (!board_accepted) {
      continue;
    }
    ++accepted_board_count;
    for (const ati::JointPointObservation& point : board.points) {
      if (!point.used_in_solver) {
        continue;
      }
      if (point.point_type == ati::JointPointType::Outer) {
        ++accepted_outer;
      } else {
        ++accepted_internal;
      }
      const cv::Point pixel(
          static_cast<int>(std::lround(point.image_xy.x())),
          static_cast<int>(std::lround(point.image_xy.y())));
      const cv::Scalar color = point.point_type == ati::JointPointType::Outer
                                   ? cv::Scalar(60, 220, 80)
                                   : cv::Scalar(40, 180, 255);
      cv::circle(output, pixel,
                 point.point_type == ati::JointPointType::Outer ? 5 : 3,
                 color, 2, cv::LINE_AA);
    }
  }

  const int banner_height = 82;
  cv::rectangle(output, cv::Rect(0, 0, output.cols, banner_height),
                cv::Scalar(18, 18, 18), cv::FILLED);
  std::ostringstream header;
  header << "stage5 bundle training frame=" << frame_result.frame_index
         << " (" << frame_result.frame_label << ")";
  cv::putText(output, header.str(), cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.62,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

  std::ostringstream summary;
  summary << "accepted_boards=" << accepted_board_count
          << " accepted_outer=" << accepted_outer
          << " accepted_internal=" << accepted_internal
          << " accepted_frame="
          << (dataset.accepted_frame_indices.count(frame_result.frame_index) > 0 ? 1 : 0);
  cv::putText(output, summary.str(), cv::Point(18, 54), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(195, 195, 195), 1, cv::LINE_AA);
  return output;
}

cv::Mat RenderAcceptedBundleBoardOverlay(
    const ati::CalibrationMeasurementDataset& dataset,
    const ati::JointMeasurementFrameResult& frame_result,
    const ati::JointBoardObservation& board_observation,
    const cv::Mat& image) {
  if (image.empty()) {
    return cv::Mat();
  }

  const bool board_accepted =
      dataset.accepted_board_observation_keys.count(
          std::make_pair(frame_result.frame_index, board_observation.board_id)) > 0;
  if (!board_accepted) {
    return cv::Mat();
  }

  cv::Mat color_image;
  if (image.channels() == 1) {
    cv::cvtColor(image, color_image, cv::COLOR_GRAY2BGR);
  } else {
    color_image = image.clone();
  }

  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  int accepted_outer = 0;
  int accepted_internal = 0;
  for (const ati::JointPointObservation& point : board_observation.points) {
    if (!point.used_in_solver) {
      continue;
    }
    const cv::Point pixel(
        static_cast<int>(std::lround(point.image_xy.x())),
        static_cast<int>(std::lround(point.image_xy.y())));
    const cv::Scalar color = point.point_type == ati::JointPointType::Outer
                                 ? cv::Scalar(60, 220, 80)
                                 : cv::Scalar(40, 180, 255);
    cv::circle(color_image, pixel,
               point.point_type == ati::JointPointType::Outer ? 5 : 3,
               color, 2, cv::LINE_AA);
    min_x = std::min(min_x, point.image_xy.x());
    min_y = std::min(min_y, point.image_xy.y());
    max_x = std::max(max_x, point.image_xy.x());
    max_y = std::max(max_y, point.image_xy.y());
    if (point.point_type == ati::JointPointType::Outer) {
      ++accepted_outer;
    } else {
      ++accepted_internal;
    }
  }

  if (!std::isfinite(min_x) || !std::isfinite(min_y)) {
    return cv::Mat();
  }

  const int padding = 80;
  cv::Rect crop_rect(static_cast<int>(std::floor(min_x)) - padding,
                     static_cast<int>(std::floor(min_y)) - padding,
                     static_cast<int>(std::ceil(max_x - min_x)) + 2 * padding,
                     static_cast<int>(std::ceil(max_y - min_y)) + 2 * padding);
  crop_rect = ClampRectToImage(crop_rect, color_image.size());
  if (crop_rect.width <= 0 || crop_rect.height <= 0) {
    return cv::Mat();
  }

  cv::Mat cropped = color_image(crop_rect).clone();
  cv::rectangle(cropped, cv::Rect(0, 0, cropped.cols, 54), cv::Scalar(18, 18, 18), cv::FILLED);
  std::ostringstream banner;
  banner << "stage5 bundle frame=" << frame_result.frame_index
         << " board=" << board_observation.board_id
         << " outer=" << accepted_outer
         << " internal=" << accepted_internal;
  cv::putText(cropped, banner.str(), cv::Point(12, 24), cv::FONT_HERSHEY_PLAIN, 1.2,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
  cv::putText(cropped, "green: accepted outer, orange: accepted internal",
              cv::Point(12, 44), cv::FONT_HERSHEY_PLAIN, 1.0,
              cv::Scalar(190, 190, 190), 1, cv::LINE_AA);
  return cropped;
}

void WriteAcceptedBundleOverlays(
    const fs::path& output_dir,
    const std::vector<ati::FrozenRound2BaselineFrameSource>& all_frames,
    const ati::CalibrationStateBundle& bundle) {
  if (!bundle.success || !bundle.ready_for_backend) {
    return;
  }

  const fs::path frame_dir = output_dir / "stage5_bundle_training_frames";
  const fs::path board_dir = output_dir / "stage5_bundle_training_boards";
  EnsureDirectoryExists(frame_dir);
  EnsureDirectoryExists(board_dir);

  for (const ati::JointMeasurementFrameResult& frame_result :
       bundle.measurement_dataset.frames) {
    const std::string image_path = FindImagePathForFrame(all_frames, frame_result.frame_index);
    if (image_path.empty()) {
      continue;
    }
    const cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      continue;
    }

    const cv::Mat frame_overlay = RenderAcceptedBundleFrameOverlay(
        bundle.measurement_dataset, frame_result, image);
    if (!frame_overlay.empty()) {
      const std::string filename =
          "frame_" + std::to_string(frame_result.frame_index) + "_" +
          frame_result.frame_label + ".png";
      cv::imwrite((frame_dir / filename).string(), frame_overlay);
    }

    for (const ati::JointBoardObservation& board_observation : frame_result.board_observations) {
      const cv::Mat board_overlay = RenderAcceptedBundleBoardOverlay(
          bundle.measurement_dataset, frame_result, board_observation, image);
      if (board_overlay.empty()) {
        continue;
      }
      const std::string filename =
          "frame_" + std::to_string(frame_result.frame_index) +
          "_board_" + std::to_string(board_observation.board_id) + ".png";
      cv::imwrite((board_dir / filename).string(), board_overlay);
    }
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
    baseline_options.source_pipeline_label = "run_stage5_benchmark";

    ati::BackendProblemOptions backend_options;
    backend_options.reference_board_id = args.reference_board_id;
    backend_options.optimize_frame_poses = true;
    backend_options.optimize_board_poses = true;
    backend_options.optimize_intrinsics = args.optimize_intrinsics;
    backend_options.delayed_intrinsics_release = true;
    backend_options.intrinsics_release_iteration =
        args.second_pass_intrinsics_release_iteration;

    ati::CalibrationBenchmarkSplitOptions split_options;
    split_options.holdout_stride = args.holdout_stride;
    split_options.holdout_offset = args.holdout_offset;
    const ati::Stage5Benchmark benchmark(split_options);
    const ati::CalibrationBenchmarkSplit preview_split =
        benchmark.BuildDeterministicSplit(all_frames);
    const std::string kalibr_training_split_signature =
        !args.kalibr_training_split_signature.empty()
            ? args.kalibr_training_split_signature
            : (preview_split.success ? preview_split.split_signature : std::string());

    ati::KalibrBenchmarkReference kalibr_reference;
    kalibr_reference.camchain_yaml = args.kalibr_camchain_yaml;
    kalibr_reference.camera_model_family = "ds";
    kalibr_reference.training_split_signature = kalibr_training_split_signature;
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
      WriteAcceptedBundleOverlays(output_dir, all_frames,
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

    const fs::path worst_frame_dir = output_dir / "benchmark_worst_holdout_frames";
    const fs::path worst_board_dir = output_dir / "benchmark_worst_holdout_boards";
    const fs::path compare_frame_dir = output_dir / "benchmark_compare_holdout_frames";
    const fs::path compare_board_dir = output_dir / "benchmark_compare_holdout_boards";
    const fs::path compare_outer_frame_dir =
        output_dir / "benchmark_compare_outer_pose_frames";
    const fs::path compare_outer_board_dir =
        output_dir / "benchmark_compare_outer_pose_boards";
    EnsureDirectoryExists(worst_frame_dir);
    EnsureDirectoryExists(worst_board_dir);
    EnsureDirectoryExists(compare_frame_dir);
    EnsureDirectoryExists(compare_board_dir);
    EnsureDirectoryExists(compare_outer_frame_dir);
    EnsureDirectoryExists(compare_outer_board_dir);

    const auto write_worst_case_overlays =
        [&](const ati::CameraModelRefitEvaluationResult& evaluation) {
          std::vector<ati::CameraModelRefitFrameDiagnostics> worst_frames =
              evaluation.frame_diagnostics;
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
            const cv::Mat overlay =
                benchmark.RenderEvaluationFrameOverlay(report, evaluation, frame.frame_index);
            if (!overlay.empty()) {
              const std::string filename =
                  evaluation.method_label + "_rank" + std::to_string(static_cast<int>(rank + 1)) +
                  "_frame_" + std::to_string(frame.frame_index) + "_" + frame.frame_label +
                  ".png";
              cv::imwrite((worst_frame_dir / filename).string(), overlay);
            }
          }

          std::vector<ati::CameraModelRefitBoardObservationDiagnostics> worst_boards =
              evaluation.board_observation_diagnostics;
          std::sort(worst_boards.begin(), worst_boards.end(),
                    [](const ati::CameraModelRefitBoardObservationDiagnostics& lhs,
                       const ati::CameraModelRefitBoardObservationDiagnostics& rhs) {
                      return lhs.evaluation_rmse > rhs.evaluation_rmse;
                    });
          if (worst_boards.size() > 10) {
            worst_boards.resize(10);
          }
          for (std::size_t rank = 0; rank < worst_boards.size(); ++rank) {
            const ati::CameraModelRefitBoardObservationDiagnostics& board =
                worst_boards[rank];
            const cv::Mat overlay = benchmark.RenderEvaluationBoardObservationOverlay(
                report, evaluation, board.frame_index, board.board_id);
            if (!overlay.empty()) {
              const std::string filename =
                  evaluation.method_label + "_rank" + std::to_string(static_cast<int>(rank + 1)) +
                  "_frame_" + std::to_string(board.frame_index) +
                  "_board_" + std::to_string(board.board_id) + ".png";
              cv::imwrite((worst_board_dir / filename).string(), overlay);
            }
          }
        };

    write_worst_case_overlays(report.our_holdout_evaluation);
    write_worst_case_overlays(report.kalibr_holdout_evaluation);

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
        const cv::Mat ours_overlay = benchmark.RenderEvaluationFrameOverlay(
            report, report.our_holdout_evaluation, frame.frame_index);
        const cv::Mat kalibr_overlay = benchmark.RenderEvaluationFrameOverlay(
            report, report.kalibr_holdout_evaluation, frame.frame_index);
        const cv::Mat compare =
            RenderSideBySideCompare(ours_overlay, kalibr_overlay, "ours", "kalibr");
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
        const cv::Mat ours_overlay = benchmark.RenderOuterPoseFitFrameOverlay(
            report, report.our_holdout_evaluation, frame.frame_index);
        const cv::Mat kalibr_overlay = benchmark.RenderOuterPoseFitFrameOverlay(
            report, report.kalibr_holdout_evaluation, frame.frame_index);
        const cv::Mat compare = RenderSideBySideCompare(
            ours_overlay, kalibr_overlay, "ours outer pose", "kalibr outer pose");
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
        const cv::Mat ours_overlay = benchmark.RenderEvaluationBoardObservationOverlay(
            report, report.our_holdout_evaluation, board.frame_index, board.board_id);
        const cv::Mat kalibr_overlay = benchmark.RenderEvaluationBoardObservationOverlay(
            report, report.kalibr_holdout_evaluation, board.frame_index, board.board_id);
        const cv::Mat compare =
            RenderSideBySideCompare(ours_overlay, kalibr_overlay, "ours", "kalibr");
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
        const cv::Mat ours_overlay = benchmark.RenderOuterPoseFitBoardOverlay(
            report, report.our_holdout_evaluation, board.frame_index, board.board_id);
        const cv::Mat kalibr_overlay = benchmark.RenderOuterPoseFitBoardOverlay(
            report, report.kalibr_holdout_evaluation, board.frame_index, board.board_id);
        const cv::Mat compare = RenderSideBySideCompare(
            ours_overlay, kalibr_overlay, "ours outer pose", "kalibr outer pose");
        if (!compare.empty()) {
          const std::string filename =
              "rank" + std::to_string(static_cast<int>(rank + 1)) +
              "_frame_" + std::to_string(board.frame_index) +
              "_board_" + std::to_string(board.board_id) + ".png";
          cv::imwrite((compare_outer_board_dir / filename).string(), compare);
        }
      }
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

    std::cout << "Stage 5 benchmark success: " << (report.success ? 1 : 0) << "\n"
              << "Bundle summary: " << (output_dir / "stage5_bundle_summary.txt").string()
              << "\n"
              << "Backend problem summary: "
              << (output_dir / "stage5_backend_problem_summary.txt").string() << "\n"
              << "Protocol summary: "
              << (output_dir / "benchmark_protocol_summary.txt").string() << "\n"
              << "Training summary: "
              << (output_dir / "benchmark_training_summary.txt").string() << "\n"
              << "Holdout summary: "
              << (output_dir / "benchmark_holdout_summary.txt").string() << "\n"
              << "Holdout points CSV: "
              << (output_dir / "benchmark_holdout_points.csv").string() << "\n"
              << "Worst-case summary: "
              << (output_dir / "benchmark_worst_cases_summary.txt").string() << "\n"
              << "Worst holdout frame overlays: " << worst_frame_dir.string() << "\n"
              << "Worst holdout board overlays: " << worst_board_dir.string() << "\n"
              << "Compare holdout frame overlays: " << compare_frame_dir.string() << "\n"
              << "Compare holdout board overlays: " << compare_board_dir.string() << "\n"
              << "Compare outer pose frame overlays: "
              << compare_outer_frame_dir.string() << "\n"
              << "Compare outer pose board overlays: "
              << compare_outer_board_dir.string() << "\n";
    if (report.diagnostic_compare.success) {
      std::cout << "Intrinsics compare CSV: "
                << (output_dir / "benchmark_intrinsics_compare.csv").string() << "\n";
    }
    if (!projection_compare.empty()) {
      std::cout << "Projection compare image: "
                << (output_dir / "benchmark_projection_compare.png").string() << "\n";
    }
    return report.success ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
