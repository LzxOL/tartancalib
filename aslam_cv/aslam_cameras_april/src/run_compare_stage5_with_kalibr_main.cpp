#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>
#include <aslam/cameras/apriltag_internal/KalibrBenchmark.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

namespace ati = aslam::cameras::apriltag_internal;
namespace fs = boost::filesystem;

struct CmdArgs {
  std::string bundle_summary_path;
  std::string kalibr_camchain_yaml;
  std::string output_path;
  bool show = false;
};

void PrintUsage(const char* program) {
  std::cout << "Usage:\n"
            << "  " << program
            << " --bundle-summary STAGE5_BUNDLE_SUMMARY --kalibr-camchain CAMCHAIN_YAML"
            << " --output OUTPUT_DIR [--show]\n";
}

CmdArgs ParseArgs(int argc, char** argv) {
  CmdArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string token = argv[i];
    if (token == "--bundle-summary" && i + 1 < argc) {
      args.bundle_summary_path = argv[++i];
    } else if (token == "--kalibr-camchain" && i + 1 < argc) {
      args.kalibr_camchain_yaml = argv[++i];
    } else if (token == "--output" && i + 1 < argc) {
      args.output_path = argv[++i];
    } else if (token == "--show") {
      args.show = true;
    } else if (token == "--help" || token == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + token);
    }
  }
  if (args.bundle_summary_path.empty() || args.kalibr_camchain_yaml.empty() ||
      args.output_path.empty()) {
    throw std::runtime_error("--bundle-summary, --kalibr-camchain and --output are required.");
  }
  return args;
}

void EnsureDirectoryExists(const fs::path& directory) {
  if (!directory.empty()) {
    fs::create_directories(directory);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CmdArgs args = ParseArgs(argc, argv);
    const fs::path output_dir(args.output_path);
    EnsureDirectoryExists(output_dir);

    ati::CalibrationStateBundle bundle;
    std::string error_message;
    if (!ati::LoadCalibrationStateBundleSummary(args.bundle_summary_path, &bundle, &error_message)) {
      throw std::runtime_error(error_message);
    }

    ati::KalibrBenchmarkInput benchmark_input;
    benchmark_input.dataset_label = bundle.scene_state.dataset_label;
    benchmark_input.kalibr_camchain_yaml = args.kalibr_camchain_yaml;
    benchmark_input.our_bundle = bundle;

    const ati::KalibrBenchmark benchmark;
    const ati::KalibrBenchmarkReport report = benchmark.Compare(benchmark_input);
    ati::WriteKalibrBenchmarkSummary((output_dir / "benchmark_summary.txt").string(), report);
    ati::WriteKalibrBenchmarkIntrinsicsCsv(
        (output_dir / "benchmark_intrinsics_compare.csv").string(), report);
    ati::WriteKalibrBenchmarkResidualSummary(
        (output_dir / "benchmark_residual_compare.txt").string(), report);

    const cv::Mat projection_compare = benchmark.RenderProjectionComparison(report);
    if (!projection_compare.empty()) {
      cv::imwrite((output_dir / "benchmark_projection_compare.png").string(),
                  projection_compare);
      if (args.show) {
        cv::imshow("stage5_kalibr_benchmark", projection_compare);
        cv::waitKey(0);
      }
    }

    std::cout << "Stage 5 Kalibr benchmark success: " << (report.success ? 1 : 0) << "\n"
              << "Benchmark summary: " << (output_dir / "benchmark_summary.txt").string() << "\n"
              << "Intrinsics compare CSV: "
              << (output_dir / "benchmark_intrinsics_compare.csv").string() << "\n"
              << "Residual compare summary: "
              << (output_dir / "benchmark_residual_compare.txt").string() << "\n";
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
