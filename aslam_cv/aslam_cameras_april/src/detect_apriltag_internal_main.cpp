#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

namespace ati = aslam::cameras::apriltag_internal;

struct CmdArgs {
  std::string config_path;
  std::string image_path;
  std::string output_path;
  std::string mode_override;
  bool show = false;
  bool no_subpix = false;
};

struct InternalMetricsSummary {
  int total_points = 0;
  int valid_points = 0;
  int image_evidence_valid_points = 0;
  int lcorner_points = 0;
  int lcorner_valid = 0;
  int xcorner_points = 0;
  int xcorner_valid = 0;
  double avg_q_refine = 0.0;
  double avg_template_quality = 0.0;
  double avg_gradient_quality = 0.0;
  double avg_final_quality = 0.0;
  double avg_image_template_quality = 0.0;
  double avg_image_gradient_quality = 0.0;
  double avg_image_centering_quality = 0.0;
  double avg_image_final_quality = 0.0;
  double avg_predicted_to_refined = 0.0;
};

ati::ApriltagInternalDetectionOptions MakeDetectionOptionsFromConfig(
    const ati::ApriltagInternalConfig& config) {
  ati::ApriltagInternalDetectionOptions options;
  options.canonical_pixels_per_module = config.canonical_pixels_per_module;
  options.refinement_window_radius = config.refinement_window_radius;
  options.internal_subpix_window_scale = config.internal_subpix_window_scale;
  options.internal_subpix_window_min = config.internal_subpix_window_min;
  options.internal_subpix_window_max = config.internal_subpix_window_max;
  options.max_subpix_displacement2 = config.max_subpix_displacement2;
  options.internal_subpix_displacement_scale = config.internal_subpix_displacement_scale;
  options.max_internal_subpix_displacement = config.max_internal_subpix_displacement;
  return options;
}

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --image IMAGE --config APRILTAG_INTERNAL_YAML [--output PNG] [--mode MODE]"
      << " [--show] [--no-subpix]\n\n"
      << "Example:\n"
      << "  " << program
      << " --image /data/frame.png --config ./config/example_apriltag_internal.yaml"
      << " --output /tmp/apriltag_internal.png\n";
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
    } else if (token == "--mode" && i + 1 < argc) {
      args.mode_override = argv[++i];
    } else if (token == "--show") {
      args.show = true;
    } else if (token == "--no-subpix") {
      args.no_subpix = true;
    } else if (token == "--help" || token == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + token);
    }
  }

  if (args.image_path.empty() || args.config_path.empty()) {
    throw std::runtime_error("Both --image and --config are required.");
  }
  return args;
}

ati::InternalProjectionMode ParseProjectionModeOrThrow(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (lowered == "homography") {
    return ati::InternalProjectionMode::Homography;
  }
  if (lowered == "virtual_pinhole_patch" || lowered == "virtual-pinhole-patch") {
    return ati::InternalProjectionMode::VirtualPinholePatch;
  }
  throw std::runtime_error("Unsupported --mode value: " + value);
}

std::string DefaultOutputPath(const std::string& image_path) {
  const boost::filesystem::path input(image_path);
  const boost::filesystem::path parent = input.has_parent_path() ? input.parent_path() : ".";
  return (parent / (input.stem().string() + "_apriltag_internal_detected.png")).string();
}

std::string DefaultCanonicalOutputPath(const std::string& image_path) {
  const boost::filesystem::path input(image_path);
  const boost::filesystem::path parent = input.has_parent_path() ? input.parent_path() : ".";
  return (parent / (input.stem().string() + "_apriltag_internal_canonical.png")).string();
}

std::string BuildMinuteStamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
  localtime_r(&now_time, &local_time);

  std::ostringstream stream;
  stream << std::put_time(&local_time, "_%Y%m%d_%H%M");
  return stream.str();
}

std::string AppendMinuteStamp(const std::string& path, const std::string& stamp) {
  const boost::filesystem::path input(path);
  const boost::filesystem::path parent = input.has_parent_path() ? input.parent_path() : ".";
  const std::string stamped_name = input.stem().string() + stamp + input.extension().string();
  return (parent / stamped_name).string();
}

cv::Mat ToGray(const cv::Mat& image) {
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image.clone();
  } else if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    throw std::runtime_error("Unsupported image format: expected 1, 3 or 4 channels.");
  }

  if (gray.depth() == CV_16U) {
    gray.convertTo(gray, CV_8U, 1.0 / 256.0);
  } else if (gray.depth() != CV_8U) {
    gray.convertTo(gray, CV_8U);
  }

  return gray;
}

cv::Mat BuildCanonicalPatch(const cv::Mat& image,
                            const ati::ApriltagInternalDetector& detector,
                            const ati::ApriltagInternalDetectionResult& result) {
  if (!result.tag_detected) {
    return cv::Mat();
  }

  const int module_dimension = detector.model().ModuleDimension();
  const int pixels_per_module = detector.options().canonical_pixels_per_module;
  const int patch_extent = module_dimension * pixels_per_module;

  const cv::Mat gray = ToGray(image);
  std::vector<cv::Point2f> image_outer(result.outer_corners.begin(), result.outer_corners.end());
  std::vector<cv::Point2f> patch_outer{
      cv::Point2f(0.0f, static_cast<float>(patch_extent)),
      cv::Point2f(static_cast<float>(patch_extent), static_cast<float>(patch_extent)),
      cv::Point2f(static_cast<float>(patch_extent), 0.0f),
      cv::Point2f(0.0f, 0.0f),
  };

  const cv::Mat image_to_patch = cv::getPerspectiveTransform(image_outer, patch_outer);
  cv::Mat patch;
  cv::warpPerspective(gray, patch, image_to_patch, cv::Size(patch_extent + 1, patch_extent + 1),
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
  return patch;
}

InternalMetricsSummary SummarizeInternalCorners(
    const std::vector<ati::InternalCornerDebugInfo>& debug_infos) {
  InternalMetricsSummary summary;
  if (debug_infos.empty()) {
    return summary;
  }

  for (const auto& debug : debug_infos) {
    ++summary.total_points;
    summary.valid_points += debug.valid ? 1 : 0;
    summary.image_evidence_valid_points += debug.image_evidence_valid ? 1 : 0;
    summary.avg_q_refine += debug.q_refine;
    summary.avg_template_quality += debug.template_quality;
    summary.avg_gradient_quality += debug.gradient_quality;
    summary.avg_final_quality += debug.final_quality;
    summary.avg_image_template_quality += debug.image_template_quality;
    summary.avg_image_gradient_quality += debug.image_gradient_quality;
    summary.avg_image_centering_quality += debug.image_centering_quality;
    summary.avg_image_final_quality += debug.image_final_quality;
    summary.avg_predicted_to_refined += debug.predicted_to_refined_displacement;

    if (debug.corner_type == ati::CornerType::LCorner) {
      ++summary.lcorner_points;
      summary.lcorner_valid += debug.valid ? 1 : 0;
    } else if (debug.corner_type == ati::CornerType::XCorner) {
      ++summary.xcorner_points;
      summary.xcorner_valid += debug.valid ? 1 : 0;
    }
  }

  const double count = static_cast<double>(summary.total_points);
  summary.avg_q_refine /= count;
  summary.avg_template_quality /= count;
  summary.avg_gradient_quality /= count;
  summary.avg_final_quality /= count;
  summary.avg_image_template_quality /= count;
  summary.avg_image_gradient_quality /= count;
  summary.avg_image_centering_quality /= count;
  summary.avg_image_final_quality /= count;
  summary.avg_predicted_to_refined /= count;
  return summary;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CmdArgs args = ParseArgs(argc, argv);

    ati::ApriltagInternalConfig config =
        ati::ApriltagInternalDetector::LoadConfig(args.config_path);
    if (!args.mode_override.empty()) {
      config.internal_projection_mode = ParseProjectionModeOrThrow(args.mode_override);
    }
    ati::ApriltagInternalDetectionOptions options = MakeDetectionOptionsFromConfig(config);
    options.do_subpix_refinement = !args.no_subpix;

    ati::ApriltagInternalDetector detector(config, options);

    cv::Mat image = cv::imread(args.image_path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      throw std::runtime_error("Failed to read image: " + args.image_path);
    }

    const ati::ApriltagInternalDetectionResult result = detector.Detect(image);
    const std::string minute_stamp = BuildMinuteStamp();

    cv::Mat overlay = image.clone();
    detector.DrawDetections(result, &overlay);

    const std::string requested_output_path =
        args.output_path.empty() ? DefaultOutputPath(args.image_path) : args.output_path;
    const std::string output_path = AppendMinuteStamp(requested_output_path, minute_stamp);
    if (!cv::imwrite(output_path, overlay)) {
      throw std::runtime_error("Failed to write output image: " + output_path);
    }

    std::string canonical_output_path;
    cv::Mat canonical_overlay = result.canonical_patch.clone();
    if (canonical_overlay.empty()) {
      canonical_overlay = BuildCanonicalPatch(image, detector, result);
    }
    if (!canonical_overlay.empty()) {
      detector.DrawCanonicalView(result, &canonical_overlay);
      canonical_output_path = AppendMinuteStamp(DefaultCanonicalOutputPath(args.image_path), minute_stamp);
      if (!cv::imwrite(canonical_output_path, canonical_overlay)) {
        throw std::runtime_error("Failed to write canonical output image: " + canonical_output_path);
      }
    }

    std::cout << "Apriltag internal config\n";
    std::cout << "  target_type: " << config.target_type << "\n";
    std::cout << "  tagFamily: " << config.tag_family << "\n";
    std::cout << "  tagId: " << config.tag_id << "\n";
    std::cout << "  tagSize: " << config.tag_size << "\n";
    std::cout << "  blackBorderBits: " << config.black_border_bits << "\n";
    std::cout << "  minVisiblePoints: " << config.min_visible_points << "\n";
    std::cout << "  canonical_pixels_per_module: " << config.canonical_pixels_per_module << "\n";
    std::cout << "  refinement_window_radius: " << config.refinement_window_radius << "\n";
    std::cout << "  internal_subpix_window_scale: " << config.internal_subpix_window_scale << "\n";
    std::cout << "  internal_subpix_window_min: " << config.internal_subpix_window_min << "\n";
    std::cout << "  internal_subpix_window_max: " << config.internal_subpix_window_max << "\n";
    std::cout << "  max_subpix_displacement2: " << config.max_subpix_displacement2 << "\n";
    std::cout << "  internal_subpix_displacement_scale: "
              << config.internal_subpix_displacement_scale << "\n";
    std::cout << "  max_internal_subpix_displacement: "
              << config.max_internal_subpix_displacement << "\n";
    std::cout << "  enable_debug_output: " << (config.enable_debug_output ? "true" : "false") << "\n";
    std::cout << "  internal_projection_mode: "
              << ati::ToString(config.internal_projection_mode) << "\n\n";
    std::cout << "Outer refinement chain\n";
    std::cout << "  chain: C-S\n";
    std::cout << "  outer_local_context_scale: "
              << config.outer_detector_config.outer_local_context_scale << "\n";
    std::cout << "  outer_corner_marker_ratio: "
              << config.outer_detector_config.outer_corner_marker_ratio << "\n";
    std::cout << "  outer_subpix_scale: "
              << config.outer_detector_config.outer_subpix_scale << "\n";
    std::cout << "  outer_refine_gate_scale: "
              << config.outer_detector_config.outer_refine_gate_scale << "\n";
    std::cout << "  outer_refine_gate_min: "
              << config.outer_detector_config.outer_refine_gate_min << "\n";
    std::cout << "  outer_subpix: "
              << (config.outer_detector_config.do_outer_subpix_refinement ? "on" : "off") << "\n\n";

    std::cout << "Outer wrapper scale plan\n";
    std::cout << "  original longest-side: "
              << result.outer_detection.original_longest_side << "\n";
    std::cout << "  scale mode: "
              << result.outer_detection.scale_configuration_mode << "\n";
    if (result.outer_detection.scale_configuration_mode == "fixed_schedule") {
      std::cout << "  fixed divisor -> target:";
      bool first_divisor = true;
      for (const auto& debug : result.outer_detection.scale_debug) {
        if (debug.configured_scale_divisor <= 0.0) {
          continue;
        }
        std::cout << (first_divisor ? " " : ", ")
                  << "/" << std::fixed << std::setprecision(2)
                  << debug.configured_scale_divisor
                  << "->" << debug.target_longest_side;
        first_divisor = false;
      }
      std::cout << "\n";
    }
    std::cout << std::defaultfloat << std::setprecision(6);
    std::cout << "  attempted target longest-sides:";
    bool first_scale = true;
    for (const auto& debug : result.outer_detection.scale_debug) {
      std::cout << (first_scale ? " [" : ", ") << debug.target_longest_side;
      first_scale = false;
    }
    std::cout << (first_scale ? " []\n" : "]\n");
    std::cout << "\n";

    if (config.enable_debug_output) {
      std::cout << "Outer wrapper per-scale debug\n";
      for (const auto& debug : result.outer_detection.scale_debug) {
        std::cout << "  scale longest-side=" << debug.target_longest_side
                  << " divisor=" << debug.configured_scale_divisor
                  << " factor=" << debug.scale_factor
                  << " size=" << debug.scaled_size.width << "x" << debug.scaled_size.height
                  << " raw=" << debug.raw_detection_count
                  << " matching=" << debug.matching_tag_count
                  << " accepted=" << debug.accepted_candidate_count
                  << " refined_success=" << debug.refined_success_count
                  << " fused=" << (debug.contributed_to_corner_fusion ? "yes" : "no");
        if (!debug.rejection_summary.empty()) {
          std::cout << " rejection=\"" << debug.rejection_summary << "\"";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }

    std::cout << "Detection summary\n";
    std::cout << "  tag detected: " << (result.tag_detected ? "yes" : "no") << "\n";
    std::cout << "  valid observation: " << (result.success ? "yes" : "no") << "\n";
    std::cout << "  outer wrapper success: " << (result.outer_detection.success ? "yes" : "no") << "\n";
    std::cout << "  outer failure reason: " << result.outer_detection.failure_reason_text << "\n";
    std::cout << "  outer scale mode: " << result.outer_detection.scale_configuration_mode << "\n";
    std::cout << "  outer chosen/reference scale: "
              << result.outer_detection.chosen_scale_longest_side << "\n";
    std::cout << "  outer used corner fusion: "
              << (result.outer_detection.used_corner_fusion ? "yes" : "no") << "\n";
    std::cout << "  outer quality: " << result.outer_detection.quality << "\n";
    std::cout << "  projection mode: " << ati::ToString(result.projection_mode) << "\n";
    std::cout << "  expected visible points: " << result.expected_visible_point_count << "\n";
    std::cout << "  valid points: " << result.valid_corner_count << "\n";
    std::cout << "  valid internal points: " << result.valid_internal_corner_count << "\n";
    std::cout << "  output image: " << output_path << "\n";
    if (!canonical_output_path.empty()) {
      std::cout << "  canonical view image: " << canonical_output_path << "\n";
    }

    if (config.enable_debug_output) {
      std::cout << "\nOuter corner multi-scale fusion debug\n";
      for (const auto& debug : result.outer_detection.corner_fusion_debug) {
        if (debug.corner_index < 0) {
          continue;
        }
        std::cout << "  corner=" << debug.corner_index
                  << " scales=" << debug.successful_scale_count
                  << " inliers=" << debug.inlier_count
                  << " outliers=" << debug.outlier_count
                  << " reject=" << (debug.used_outlier_rejection ? "yes" : "no")
                  << " stable=" << (debug.stable_after_fusion ? "yes" : "no")
                  << " avg_before=" << debug.average_deviation_before
                  << " max_before=" << debug.max_deviation_before
                  << " avg_after=" << debug.average_deviation_after
                  << " max_after=" << debug.max_deviation_after
                  << " threshold=" << debug.outlier_threshold
                  << " consensus=(" << debug.consensus_corner.x << ", " << debug.consensus_corner.y
                  << ")"
                  << " fused=(" << debug.fused_corner.x << ", " << debug.fused_corner.y << ")"
                  << "\n";
        if (!debug.scale_observations.empty()) {
          std::cout << "    observations:";
          for (const auto& observation : debug.scale_observations) {
            std::cout << " [" << observation.target_longest_side
                      << " (" << observation.coarse_corner.x << ", "
                      << observation.coarse_corner.y << ")"
                      << " dc=" << observation.deviation_from_consensus
                      << " df=" << observation.deviation_from_fused
                      << (observation.rejected_as_outlier ? " outlier" : " inlier")
                      << "]";
          }
          std::cout << "\n";
        }
      }

      std::cout << "\nOuter corner refinement debug\n";
      for (const auto& debug : result.outer_detection.corner_verification_debug) {
        if (debug.corner_index < 0) {
          continue;
        }
        std::cout << "  corner=" << debug.corner_index
                  << " chain=C-S"
                  << " refined_valid=" << (debug.refined_valid ? "yes" : "no")
                  << " local_scale=" << debug.local_scale
                  << " marker_width=" << debug.corner_marker_width
                  << " context_radius=" << debug.verification_roi_radius
                  << " coarse=(" << debug.coarse_corner.x << ", " << debug.coarse_corner.y << ")"
                  << " subpix=(" << debug.subpix_corner.x << ", " << debug.subpix_corner.y << ")"
                  << " d_cs=" << debug.coarse_to_subpix_displacement
                  << " d_cr=" << debug.coarse_to_refined_displacement
                  << " subpix_radius=" << debug.subpix_window_radius
                  << " refine_gate=" << debug.refine_displacement_limit;
        std::cout << "\n";
      }

      std::cout << "\nInternal corner debug\n";
      for (const auto& debug : result.internal_corner_debug) {
        std::cout << "  id=" << debug.point_id
                  << " type=" << ati::ToString(debug.corner_type)
                  << " valid=" << (debug.valid ? "yes" : "no")
                  << " image_valid=" << (debug.image_evidence_valid ? "yes" : "no")
                  << " predicted=(" << debug.predicted_image.x << ", " << debug.predicted_image.y << ")"
                  << " refined=(" << debug.refined_image.x << ", " << debug.refined_image.y << ")"
                  << " module_scale=" << debug.local_module_scale
                  << " subpix_radius=" << debug.subpix_window_radius
                  << " subpix_gate=" << debug.subpix_displacement_limit
                  << " image_search_radius=" << debug.image_evidence_search_radius
                  << " q_refine=" << debug.q_refine
                  << " template_quality=" << debug.template_quality
                  << " gradient_quality=" << debug.gradient_quality
                  << " final_quality=" << debug.final_quality
                  << " image_template_quality=" << debug.image_template_quality
                  << " image_gradient_quality=" << debug.image_gradient_quality
                  << " image_centering_quality=" << debug.image_centering_quality
                  << " image_final_quality=" << debug.image_final_quality
                  << "\n";
      }
    }

    const InternalMetricsSummary metrics = SummarizeInternalCorners(result.internal_corner_debug);
    std::cout << "\nInternal summary\n";
    std::cout << "  total internal points: " << metrics.total_points << "\n";
    std::cout << "  legacy valid internal points: " << metrics.valid_points << "\n";
    std::cout << "  image-evidence internal points: " << metrics.image_evidence_valid_points << "\n";
    std::cout << "  avg q_refine: " << metrics.avg_q_refine << "\n";
    std::cout << "  avg legacy template_quality: " << metrics.avg_template_quality << "\n";
    std::cout << "  avg legacy gradient_quality: " << metrics.avg_gradient_quality << "\n";
    std::cout << "  avg legacy final_quality: " << metrics.avg_final_quality << "\n";
    std::cout << "  avg image template_quality: " << metrics.avg_image_template_quality << "\n";
    std::cout << "  avg image gradient_quality: " << metrics.avg_image_gradient_quality << "\n";
    std::cout << "  avg image centering_quality: " << metrics.avg_image_centering_quality << "\n";
    std::cout << "  avg image final_quality: " << metrics.avg_image_final_quality << "\n";
    std::cout << "  avg predicted->refined displacement: " << metrics.avg_predicted_to_refined << "\n";
    std::cout << "  LCorner valid: " << metrics.lcorner_valid << "/" << metrics.lcorner_points << "\n";
    std::cout << "  XCorner valid: " << metrics.xcorner_valid << "/" << metrics.xcorner_points << "\n";

    if (args.show) {
      cv::imshow("m-kilbr Apriltag Internal Detection", overlay);
      cv::waitKey(0);
    }

    return result.success ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "[m-kilbr] " << error.what() << "\n\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
