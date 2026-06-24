#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/ApriltagInternalDebugVisualization.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <Eigen/Eigenvalues>
#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "apriltags/TagFamily.h"
#include "apriltags/Tag36h11.h"
#include "apriltags/TagDetection.h"
#include "apriltags/TagDetector.h"

namespace {

namespace ati = aslam::cameras::apriltag_internal;

struct CmdArgs {
  std::string config_path;
  std::string image_path;
  std::string output_path;
  std::string mode_override;
  bool all = false;
  bool show = false;
  bool no_subpix = false;
  bool no_output_images = false;
  bool no_debug_output = false;
  bool final_corners_only = false;
  bool rectified_roi_demo = false;
  double rectified_roi_crop_offset_x = 0.0;
  double rectified_roi_crop_offset_y = 0.0;
  double rectified_roi_fov_deg = 44.0;
  int rectified_roi_patch_size = 800;
  int corner_radius = 2;
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
  double avg_sphere_seed_quality = 0.0;
  double avg_ray_refine_edge_quality = 0.0;
  double avg_ray_refine_photometric_quality = 0.0;
  double avg_ray_refine_final_quality = 0.0;
  double avg_seed_to_refined_angular = 0.0;
  double avg_predicted_to_border_seed = 0.0;
  double avg_predicted_to_seed = 0.0;
  double avg_seed_to_refined = 0.0;
  double avg_predicted_to_refined = 0.0;
  int border_seed_count = 0;
};

std::string BuildOuterChainLabel(const ati::OuterCornerVerificationDebugInfo& debug) {
  if (debug.spherical_refinement_valid) {
    return debug.subpix_applied ? "C-SP-S" : "C-SP";
  }
  if (debug.subpix_applied) {
    return "C-S";
  }
  return "C";
}

bool UsesSphereSeedPipelineLocal(ati::InternalProjectionMode mode) {
  return mode == ati::InternalProjectionMode::SphereLattice ||
         mode == ati::InternalProjectionMode::SphereBorderLattice ||
         mode == ati::InternalProjectionMode::PureSphericalBoundarySeed ||
         mode == ati::InternalProjectionMode::SphereRayRefine;
}

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
  options.ignore_image_evidence_min_quality =
      config.ignore_image_evidence_min_quality;
  options.force_internal_seed_from_prediction =
      config.force_internal_seed_from_prediction;
  options.bypass_internal_seed_filters =
      config.bypass_internal_seed_filters;
  return options;
}

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --image IMAGE_OR_DIR --config APRILTAG_INTERNAL_YAML [--output PNG_OR_DIR]"
      << " [--mode MODE] [--all] [--show] [--no-subpix]"
      << " [--no-output-images] [--no-debug-output]"
      << " [--final-corners-only] [--corner-radius N]"
      << " [--rectified-roi-demo] [--rectified-roi-patch-size N]"
      << " [--rectified-roi-fov-deg DEG]"
      << " [--rectified-roi-crop-offset-x X --rectified-roi-crop-offset-y Y]\n\n"
      << "Example:\n"
      << "  " << program
      << " --image /data/frame.png --config ./config/example_apriltag_internal.yaml"
      << " --output /tmp/apriltag_internal.png\n"
      << "  " << program
      << " --image /data/images --all --config ./config/example_apriltag_internal.yaml"
      << " --output /tmp/apriltag_batch\n";
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
    } else if (token == "--all") {
      args.all = true;
    } else if (token == "--show") {
      args.show = true;
    } else if (token == "--no-subpix") {
      args.no_subpix = true;
    } else if (token == "--no-output-images") {
      args.no_output_images = true;
    } else if (token == "--no-debug-output") {
      args.no_debug_output = true;
    } else if (token == "--final-corners-only") {
      args.final_corners_only = true;
    } else if (token == "--rectified-roi-demo") {
      args.rectified_roi_demo = true;
    } else if (token == "--rectified-roi-patch-size" && i + 1 < argc) {
      args.rectified_roi_patch_size = std::max(128, std::stoi(argv[++i]));
    } else if (token == "--rectified-roi-fov-deg" && i + 1 < argc) {
      args.rectified_roi_fov_deg = std::max(5.0, std::stod(argv[++i]));
    } else if (token == "--rectified-roi-crop-offset-x" && i + 1 < argc) {
      args.rectified_roi_crop_offset_x = std::stod(argv[++i]);
    } else if (token == "--rectified-roi-crop-offset-y" && i + 1 < argc) {
      args.rectified_roi_crop_offset_y = std::stod(argv[++i]);
    } else if (token == "--corner-radius" && i + 1 < argc) {
      args.corner_radius = std::max(1, std::stoi(argv[++i]));
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
  if (lowered == "virtual_pinhole_image_subpix" ||
      lowered == "virtual-pinhole-image-subpix") {
    return ati::InternalProjectionMode::VirtualPinholeImageSubpix;
  }
  if (lowered == "sphere_lattice" || lowered == "sphere-lattice") {
    return ati::InternalProjectionMode::SphereLattice;
  }
  if (lowered == "sphere_border_lattice" || lowered == "sphere-border-lattice") {
    return ati::InternalProjectionMode::SphereBorderLattice;
  }
  if (lowered == "pure_spherical_boundary_seed" ||
      lowered == "pure-spherical-boundary-seed" ||
      lowered == "sphere_boundary_seed" ||
      lowered == "sphere-boundary-seed") {
    return ati::InternalProjectionMode::PureSphericalBoundarySeed;
  }
  if (lowered == "sphere_ray_refine" || lowered == "sphere-ray-refine") {
    return ati::InternalProjectionMode::SphereRayRefine;
  }
  throw std::runtime_error("Unsupported --mode value: " + value);
}

std::string DefaultOutputPath(const std::string& image_path) {
  const boost::filesystem::path input(image_path);
  const boost::filesystem::path parent = input.has_parent_path() ? input.parent_path() : ".";
  return (parent / (input.stem().string() + "_apriltag_internal_detected.png")).string();
}

bool IsImageFile(const boost::filesystem::path& path) {
  if (!boost::filesystem::is_regular_file(path)) {
    return false;
  }
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
         extension == ".bmp" || extension == ".tif" || extension == ".tiff";
}

std::vector<std::string> CollectImagePaths(const std::string& image_path, bool all) {
  const boost::filesystem::path input(image_path);
  if (!all) {
    return {image_path};
  }

  if (!boost::filesystem::exists(input)) {
    throw std::runtime_error("Input path does not exist: " + image_path);
  }

  boost::filesystem::path directory = input;
  if (boost::filesystem::is_regular_file(input)) {
    directory = input.parent_path();
  }
  if (!boost::filesystem::is_directory(directory)) {
    throw std::runtime_error("--all requires --image to point to an image directory or a file inside it.");
  }

  std::vector<std::string> image_paths;
  for (boost::filesystem::directory_iterator it(directory), end; it != end; ++it) {
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

void EnsureParentDirectoryExists(const std::string& output_path) {
  const boost::filesystem::path output(output_path);
  const boost::filesystem::path parent = output.has_parent_path() ? output.parent_path() : ".";
  if (!parent.empty()) {
    boost::filesystem::create_directories(parent);
  }
}

std::string BatchRequestedOutputPath(const std::string& configured_output_path,
                                     const std::string& image_path) {
  const boost::filesystem::path image(image_path);
  if (configured_output_path.empty()) {
    const boost::filesystem::path output_dir = image.parent_path() / "apriltag_internal_batch";
    return (output_dir / (image.stem().string() + "_apriltag_internal_detected.png")).string();
  }

  const boost::filesystem::path configured(configured_output_path);
  if (configured.has_extension()) {
    const boost::filesystem::path parent =
        configured.has_parent_path() ? configured.parent_path() : ".";
    return (parent /
            (image.stem().string() + "_" + configured.stem().string() +
             configured.extension().string()))
        .string();
  }

  return (configured / (image.stem().string() + "_apriltag_internal_detected.png")).string();
}

std::string DefaultCanonicalOutputPath(const std::string& image_path) {
  const boost::filesystem::path input(image_path);
  const boost::filesystem::path parent = input.has_parent_path() ? input.parent_path() : ".";
  return (parent / (input.stem().string() + "_apriltag_internal_canonical.png")).string();
}

std::string CanonicalOutputPathForRequestedOutput(const std::string& requested_output_path) {
  const boost::filesystem::path output(requested_output_path);
  const boost::filesystem::path parent = output.has_parent_path() ? output.parent_path() : ".";
  return (parent / (output.stem().string() + "_canonical" + output.extension().string())).string();
}

std::string SphereOutputPathForRequestedOutput(const std::string& requested_output_path) {
  const boost::filesystem::path output(requested_output_path);
  const boost::filesystem::path parent = output.has_parent_path() ? output.parent_path() : ".";
  return (parent / (output.stem().string() + "_sphere" + output.extension().string())).string();
}

std::string InternalSeedOutputPathForRequestedOutput(const std::string& requested_output_path) {
  const boost::filesystem::path output(requested_output_path);
  const boost::filesystem::path parent = output.has_parent_path() ? output.parent_path() : ".";
  return (parent / (output.stem().string() + "_internal_seed" + output.extension().string())).string();
}

std::string InternalSphereOutputPathForRequestedOutput(const std::string& requested_output_path) {
  const boost::filesystem::path output(requested_output_path);
  const boost::filesystem::path parent = output.has_parent_path() ? output.parent_path() : ".";
  return (parent / (output.stem().string() + "_internal_sphere" + output.extension().string())).string();
}

std::string AppendStemSuffix(const std::string& path, const std::string& suffix) {
  const boost::filesystem::path input(path);
  const boost::filesystem::path parent = input.has_parent_path() ? input.parent_path() : ".";
  return (parent / (input.stem().string() + suffix + input.extension().string())).string();
}

std::string JoinBoardIds(const std::vector<int>& board_ids) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < board_ids.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << board_ids[index];
  }
  return stream.str();
}

std::string JoinDetectionSummaries(const std::vector<std::string>& summaries,
                                   std::size_t max_count = 6) {
  if (summaries.empty()) {
    return "(none)";
  }

  std::ostringstream stream;
  const std::size_t count = std::min(max_count, summaries.size());
  for (std::size_t index = 0; index < count; ++index) {
    if (index > 0) {
      stream << " | ";
    }
    stream << summaries[index];
  }
  if (summaries.size() > count) {
    stream << " | ... +" << (summaries.size() - count);
  }
  return stream.str();
}

void PrintOuterScaleDebug(const ati::OuterTagDetectionResult& outer_detection) {
  if (outer_detection.scale_debug.empty()) {
    return;
  }

  std::cout << "  outer scale debug\n";
  for (const auto& scale_debug : outer_detection.scale_debug) {
    std::cout << "    scale longest_side=" << scale_debug.target_longest_side
              << " factor=" << std::fixed << std::setprecision(3) << scale_debug.scale_factor
              << " raw=" << scale_debug.raw_detection_count
              << " raw_good=" << scale_debug.raw_good_detection_count
              << " match=" << scale_debug.matching_tag_count
              << " match_good=" << scale_debug.matching_good_tag_count
              << " accepted=" << scale_debug.accepted_candidate_count;
    if (!scale_debug.rejection_summary.empty()) {
      std::cout << " reject=" << scale_debug.rejection_summary;
    }
    std::cout << "\n";
    std::cout << "      raw detections: "
              << JoinDetectionSummaries(scale_debug.raw_detection_summaries) << "\n";
  }
  std::cout << std::defaultfloat << std::setprecision(6);
}

ati::ApriltagInternalConfig MakeBoardOutputConfig(const ati::ApriltagInternalConfig& base_config,
                                                  int board_id) {
  ati::ApriltagInternalConfig config = base_config;
  config.tag_id = board_id;
  config.tag_ids.clear();
  config.outer_detector_config.tag_id = board_id;
  return config;
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

cv::Mat EnsureBgr(const cv::Mat& image) {
  cv::Mat bgr;
  if (image.channels() == 1) {
    cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
  } else {
    bgr = image.clone();
  }
  return bgr;
}

void DrawTinyCorner(cv::Mat* image,
                    const cv::Point2f& point,
                    const cv::Scalar& color,
                    int radius) {
  if (image == nullptr) {
    return;
  }
  cv::circle(*image, point, radius, color, cv::FILLED, cv::LINE_AA);
  if (radius >= 2) {
    cv::circle(*image, point, 1, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
  }
}

cv::Mat BuildFinalCornersOnlyOverlay(
    const ati::ApriltagInternalDetectionResult& result,
    const cv::Mat& image,
    int corner_radius) {
  cv::Mat overlay = EnsureBgr(image);
  if (!result.tag_detected) {
    return overlay;
  }

  const cv::Scalar kOuterColor(0, 255, 255);
  const cv::Scalar kLCornerColor(0, 220, 0);
  const cv::Scalar kXCornerColor(255, 0, 255);

  for (const auto& measurement : result.corners) {
    if (!measurement.valid) {
      continue;
    }
    const cv::Point2f point(static_cast<float>(measurement.image_xy.x()),
                            static_cast<float>(measurement.image_xy.y()));
    cv::Scalar color = kLCornerColor;
    if (measurement.corner_type == ati::CornerType::Outer) {
      color = kOuterColor;
    } else if (measurement.corner_type == ati::CornerType::XCorner) {
      color = kXCornerColor;
    }
    DrawTinyCorner(&overlay, point, color, corner_radius);
  }

  return overlay;
}

cv::Mat BuildFinalCornersOnlyOverlay(
    const ati::ApriltagInternalMultiDetectionResult& multi_result,
    const cv::Mat& image,
    int corner_radius) {
  cv::Mat overlay = EnsureBgr(image);
  for (const ati::ApriltagInternalDetectionResult& result : multi_result.detections) {
    overlay = BuildFinalCornersOnlyOverlay(result, overlay, corner_radius);
  }
  return overlay;
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

bool OrderQuadClockwise(const std::vector<cv::Point>& contour,
                        std::array<cv::Point2f, 4>* ordered) {
  if (ordered == nullptr || contour.size() < 4) {
    return false;
  }

  cv::RotatedRect rect = cv::minAreaRect(contour);
  cv::Point2f points[4];
  rect.points(points);
  std::vector<cv::Point2f> corners(points, points + 4);

  std::sort(corners.begin(), corners.end(),
            [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
              return lhs.y < rhs.y;
            });
  std::array<cv::Point2f, 2> top{{corners[0], corners[1]}};
  std::array<cv::Point2f, 2> bottom{{corners[2], corners[3]}};
  if (top[0].x > top[1].x) {
    std::swap(top[0], top[1]);
  }
  if (bottom[0].x > bottom[1].x) {
    std::swap(bottom[0], bottom[1]);
  }

  // The rectified patch convention here is top-left, top-right,
  // bottom-right, bottom-left. It is only for the demo patch.
  (*ordered)[0] = top[0];
  (*ordered)[1] = top[1];
  (*ordered)[2] = bottom[1];
  (*ordered)[3] = bottom[0];
  return true;
}

bool EstimateBrightBoardQuad(const cv::Mat& gray,
                             std::array<cv::Point2f, 4>* quad,
                             cv::Mat* mask_out = nullptr) {
  if (quad == nullptr || gray.empty()) {
    return false;
  }

  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0.0);

  cv::Mat mask;
  cv::threshold(blurred, mask, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21)));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    if (mask_out != nullptr) {
      *mask_out = mask;
    }
    return false;
  }

  const double image_area = static_cast<double>(gray.cols) * gray.rows;
  int best_index = -1;
  double best_area = 0.0;
  for (std::size_t index = 0; index < contours.size(); ++index) {
    const double area = std::fabs(cv::contourArea(contours[index]));
    if (area > best_area && area > image_area * 0.05) {
      best_area = area;
      best_index = static_cast<int>(index);
    }
  }
  if (best_index < 0) {
    if (mask_out != nullptr) {
      *mask_out = mask;
    }
    return false;
  }

  if (mask_out != nullptr) {
    *mask_out = mask;
  }
  return OrderQuadClockwise(contours[static_cast<std::size_t>(best_index)], quad);
}

bool NormalizeRay(Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    return false;
  }
  const double norm = ray->norm();
  if (!std::isfinite(norm) || norm <= 1e-12) {
    return false;
  }
  *ray /= norm;
  return true;
}

bool BuildLocalDsPatchFrame(const ati::DoubleSphereCameraModel& camera,
                            const cv::Point2f& full_image_center,
                            Eigen::Vector3d* center_ray,
                            Eigen::Vector3d* tangent_x,
                            Eigen::Vector3d* tangent_y) {
  if (center_ray == nullptr || tangent_x == nullptr || tangent_y == nullptr) {
    return false;
  }
  if (!camera.keypointToEuclidean(
          Eigen::Vector2d(full_image_center.x, full_image_center.y), center_ray) ||
      !NormalizeRay(center_ray)) {
    return false;
  }

  Eigen::Vector3d ray_x = Eigen::Vector3d::Zero();
  Eigen::Vector3d ray_y = Eigen::Vector3d::Zero();
  constexpr double kDeltaPx = 24.0;
  if (!camera.keypointToEuclidean(
          Eigen::Vector2d(full_image_center.x + kDeltaPx, full_image_center.y),
          &ray_x) ||
      !camera.keypointToEuclidean(
          Eigen::Vector2d(full_image_center.x, full_image_center.y + kDeltaPx),
          &ray_y) ||
      !NormalizeRay(&ray_x) || !NormalizeRay(&ray_y)) {
    return false;
  }

  *tangent_x = ray_x - (*center_ray) * center_ray->dot(ray_x);
  if (!NormalizeRay(tangent_x)) {
    return false;
  }
  *tangent_y = ray_y - (*center_ray) * center_ray->dot(ray_y);
  *tangent_y -= (*tangent_x) * tangent_x->dot(*tangent_y);
  if (!NormalizeRay(tangent_y)) {
    return false;
  }
  if (center_ray->dot(tangent_x->cross(*tangent_y)) < 0.0) {
    *tangent_y = -*tangent_y;
  }
  return true;
}

bool BuildDsRaySpacePatch(const cv::Mat& gray_crop,
                          const ati::DoubleSphereCameraModel& camera,
                          const cv::Point2f& crop_center,
                          const cv::Point2f& crop_offset_in_full_image,
                          double fov_deg,
                          int patch_size,
                          cv::Mat* patch) {
  if (patch == nullptr || gray_crop.empty()) {
    return false;
  }
  patch->release();
  const cv::Point2f full_center = crop_center + crop_offset_in_full_image;

  Eigen::Vector3d center_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangent_x = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangent_y = Eigen::Vector3d::Zero();
  if (!BuildLocalDsPatchFrame(camera, full_center, &center_ray, &tangent_x, &tangent_y)) {
    return false;
  }

  const double fov_rad = fov_deg * 3.14159265358979323846 / 180.0;
  const double focal = 0.5 * static_cast<double>(patch_size) / std::tan(0.5 * fov_rad);
  const double cx = 0.5 * static_cast<double>(patch_size - 1);
  const double cy = 0.5 * static_cast<double>(patch_size - 1);

  cv::Mat map_x(patch_size, patch_size, CV_32F);
  cv::Mat map_y(patch_size, patch_size, CV_32F);
  for (int y = 0; y < patch_size; ++y) {
    for (int x = 0; x < patch_size; ++x) {
      const double nx = (static_cast<double>(x) - cx) / focal;
      const double ny = (static_cast<double>(y) - cy) / focal;
      Eigen::Vector3d ray = center_ray + nx * tangent_x + ny * tangent_y;
      if (!NormalizeRay(&ray)) {
        map_x.at<float>(y, x) = -1.0f;
        map_y.at<float>(y, x) = -1.0f;
        continue;
      }

      Eigen::Vector2d full_keypoint = Eigen::Vector2d::Zero();
      if (!camera.vsEuclideanToKeypoint(ray, &full_keypoint)) {
        map_x.at<float>(y, x) = -1.0f;
        map_y.at<float>(y, x) = -1.0f;
        continue;
      }

      map_x.at<float>(y, x) =
          static_cast<float>(full_keypoint.x() - crop_offset_in_full_image.x);
      map_y.at<float>(y, x) =
          static_cast<float>(full_keypoint.y() - crop_offset_in_full_image.y);
    }
  }

  cv::remap(gray_crop, *patch, map_x, map_y, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar(127));
  return !patch->empty();
}

std::string RectifiedRoiPatchPathForRequestedOutput(const std::string& requested_output_path) {
  return AppendStemSuffix(requested_output_path, "_rectified_roi_patch");
}

std::string RectifiedRoiOverlayPathForRequestedOutput(const std::string& requested_output_path) {
  return AppendStemSuffix(requested_output_path, "_rectified_roi_overlay");
}

std::string RectifiedRoiSourcePathForRequestedOutput(const std::string& requested_output_path) {
  return AppendStemSuffix(requested_output_path, "_rectified_roi_source");
}

void RunRectifiedRoiDemo(const cv::Mat& image,
                         const std::string& requested_output_path,
                         const ati::DoubleSphereCameraModel* camera,
                         const cv::Point2f& crop_offset_in_full_image,
                         double fov_deg,
                         int patch_size) {
  const cv::Mat gray = ToGray(image);
  std::array<cv::Point2f, 4> quad{};
  cv::Mat mask;
  if (!EstimateBrightBoardQuad(gray, &quad, &mask)) {
    std::cout << "Rectified ROI demo\n";
    std::cout << "  status: failed_to_estimate_board_quad\n";
    return;
  }

  const int size = std::max(128, patch_size);
  cv::Point2f quad_center(0.0f, 0.0f);
  for (const cv::Point2f& corner : quad) {
    quad_center += corner;
  }
  quad_center *= 0.25f;

  std::vector<cv::Point2f> src(quad.begin(), quad.end());
  std::vector<cv::Point2f> dst{
      cv::Point2f(0.0f, 0.0f),
      cv::Point2f(static_cast<float>(size - 1), 0.0f),
      cv::Point2f(static_cast<float>(size - 1), static_cast<float>(size - 1)),
      cv::Point2f(0.0f, static_cast<float>(size - 1)),
  };
  const cv::Mat H = cv::getPerspectiveTransform(src, dst);
  cv::Mat patch;
  cv::warpPerspective(gray, patch, H, cv::Size(size, size), cv::INTER_LINEAR,
                      cv::BORDER_CONSTANT, cv::Scalar(255));

  AprilTags::TagDetector tag_detector(AprilTags::tagCodes36h11, 2);
  const std::vector<AprilTags::TagDetection> perspective_detections =
      tag_detector.extractTags(patch);

  cv::Mat ds_patch;
  std::vector<AprilTags::TagDetection> ds_detections;
  bool ds_patch_success = false;
  if (camera != nullptr && camera->IsValid()) {
    ds_patch_success = BuildDsRaySpacePatch(gray, *camera, quad_center,
                                            crop_offset_in_full_image, fov_deg,
                                            size, &ds_patch);
    if (ds_patch_success) {
      ds_detections = tag_detector.extractTags(ds_patch);
    }
  }

  cv::Mat source_overlay = EnsureBgr(image);
  for (int index = 0; index < 4; ++index) {
    cv::line(source_overlay, quad[static_cast<std::size_t>(index)],
             quad[static_cast<std::size_t>((index + 1) % 4)],
             cv::Scalar(0, 255, 255), 3, cv::LINE_AA);
    cv::putText(source_overlay, std::to_string(index),
                quad[static_cast<std::size_t>(index)] + cv::Point2f(8.0f, -8.0f),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2,
                cv::LINE_AA);
  }

  auto draw_detections_on_patch = [](const cv::Mat& gray_patch,
                                     const std::vector<AprilTags::TagDetection>& detections) {
    cv::Mat overlay;
    cv::cvtColor(gray_patch, overlay, cv::COLOR_GRAY2BGR);
    for (const AprilTags::TagDetection& detection : detections) {
      std::array<cv::Point2f, 4> corners{};
      for (int index = 0; index < 4; ++index) {
        corners[static_cast<std::size_t>(index)] =
            cv::Point2f(detection.p[index].first, detection.p[index].second);
      }
      const cv::Scalar color = detection.good ? cv::Scalar(0, 220, 0)
                                              : cv::Scalar(0, 165, 255);
      for (int index = 0; index < 4; ++index) {
        cv::line(overlay, corners[static_cast<std::size_t>(index)],
                 corners[static_cast<std::size_t>((index + 1) % 4)],
                 color, 2, cv::LINE_AA);
      }
      cv::putText(overlay,
                  "id=" + std::to_string(detection.id) +
                      " h=" + std::to_string(detection.hammingDistance),
                  cv::Point(static_cast<int>(std::lround(detection.cxy.first)) + 8,
                            static_cast<int>(std::lround(detection.cxy.second)) + 8),
                  cv::FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv::LINE_AA);
    }
    return overlay;
  };

  cv::Mat patch_overlay = draw_detections_on_patch(patch, perspective_detections);
  cv::putText(patch_overlay, "homography rectified patch", cv::Point(12, 28),
              cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2,
              cv::LINE_AA);

  cv::Mat ds_patch_overlay;
  if (ds_patch_success) {
    ds_patch_overlay = draw_detections_on_patch(ds_patch, ds_detections);
    cv::putText(ds_patch_overlay, "DS ray-space local patch", cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2,
                cv::LINE_AA);
  }

  const std::string source_path =
      RectifiedRoiSourcePathForRequestedOutput(requested_output_path);
  const std::string patch_path =
      RectifiedRoiPatchPathForRequestedOutput(requested_output_path);
  const std::string overlay_path =
      RectifiedRoiOverlayPathForRequestedOutput(requested_output_path);
  const std::string ds_patch_path =
      AppendStemSuffix(requested_output_path, "_ds_ray_patch");
  const std::string ds_overlay_path =
      AppendStemSuffix(requested_output_path, "_ds_ray_overlay");
  cv::imwrite(source_path, source_overlay);
  cv::imwrite(patch_path, patch);
  cv::imwrite(overlay_path, patch_overlay);
  if (ds_patch_success) {
    cv::imwrite(ds_patch_path, ds_patch);
    cv::imwrite(ds_overlay_path, ds_patch_overlay);
  }

  std::cout << "Rectified ROI demo\n";
  std::cout << "  estimated_board_quad: ";
  for (int index = 0; index < 4; ++index) {
    if (index > 0) {
      std::cout << " ";
    }
    std::cout << "(" << std::lround(quad[static_cast<std::size_t>(index)].x)
              << "," << std::lround(quad[static_cast<std::size_t>(index)].y)
              << ")";
  }
  std::cout << "\n";
  std::cout << "  patch_size: " << size << "\n";
  std::cout << "  homography_detection_count: "
            << perspective_detections.size() << "\n";
  for (const AprilTags::TagDetection& detection : perspective_detections) {
    std::cout << "    id=" << detection.id
              << " good=" << (detection.good ? 1 : 0)
              << " hamming=" << detection.hammingDistance
              << " center=(" << std::lround(detection.cxy.first)
              << "," << std::lround(detection.cxy.second) << ")\n";
  }
  std::cout << "  ds_ray_patch_success: " << (ds_patch_success ? 1 : 0)
            << "\n";
  std::cout << "  ds_ray_fov_deg: " << fov_deg << "\n";
  std::cout << "  crop_offset_full_image: ("
            << crop_offset_in_full_image.x << ","
            << crop_offset_in_full_image.y << ")\n";
  std::cout << "  ds_ray_detection_count: " << ds_detections.size() << "\n";
  for (const AprilTags::TagDetection& detection : ds_detections) {
    std::cout << "    id=" << detection.id
              << " good=" << (detection.good ? 1 : 0)
              << " hamming=" << detection.hammingDistance
              << " center=(" << std::lround(detection.cxy.first)
              << "," << std::lround(detection.cxy.second) << ")\n";
  }
  std::cout << "  source_overlay: " << source_path << "\n";
  std::cout << "  rectified_patch: " << patch_path << "\n";
  std::cout << "  rectified_overlay: " << overlay_path << "\n";
  if (ds_patch_success) {
    std::cout << "  ds_ray_patch: " << ds_patch_path << "\n";
    std::cout << "  ds_ray_overlay: " << ds_overlay_path << "\n";
  }
}

bool UnprojectImagePointsToRays(const ati::DoubleSphereCameraModel& camera,
                                const std::vector<cv::Point2f>& image_points,
                                std::vector<Eigen::Vector3d>* rays) {
  if (rays == nullptr) {
    throw std::runtime_error("UnprojectImagePointsToRays requires a valid output pointer.");
  }

  rays->clear();
  rays->reserve(image_points.size());
  for (const cv::Point2f& point : image_points) {
    Eigen::Vector3d ray = Eigen::Vector3d::Zero();
    if (!camera.keypointToEuclidean(Eigen::Vector2d(point.x, point.y), &ray)) {
      continue;
    }
    const double norm = ray.norm();
    if (!std::isfinite(norm) || norm <= 1e-9) {
      continue;
    }
    rays->push_back(ray / norm);
  }
  return !rays->empty();
}

bool FitPlaneToRays(const std::vector<Eigen::Vector3d>& rays,
                    Eigen::Vector3d* plane_normal,
                    double* rms_residual) {
  if (plane_normal == nullptr || rms_residual == nullptr) {
    throw std::runtime_error("FitPlaneToRays requires valid output pointers.");
  }
  if (rays.size() < 3) {
    return false;
  }

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const Eigen::Vector3d& ray : rays) {
    covariance += ray * ray.transpose();
  }

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  Eigen::Vector3d normal = solver.eigenvectors().col(0);
  const double normal_norm = normal.norm();
  if (!std::isfinite(normal_norm) || normal_norm <= 1e-9) {
    return false;
  }
  normal /= normal_norm;

  double residual_sum_sq = 0.0;
  for (const Eigen::Vector3d& ray : rays) {
    const double residual = std::abs(normal.dot(ray));
    residual_sum_sq += residual * residual;
  }

  *plane_normal = normal;
  *rms_residual = std::sqrt(residual_sum_sq / static_cast<double>(rays.size()));
  return std::isfinite(*rms_residual);
}

std::vector<Eigen::Vector3d> SamplePlaneGreatCircle(const Eigen::Vector3d& plane_normal,
                                                    const Eigen::Vector3d& anchor_ray,
                                                    int sample_count = 160) {
  std::vector<Eigen::Vector3d> rays;
  const double normal_norm = plane_normal.norm();
  if (!std::isfinite(normal_norm) || normal_norm <= 1e-9) {
    return rays;
  }

  const Eigen::Vector3d unit_normal = plane_normal / normal_norm;
  Eigen::Vector3d basis_a = anchor_ray - unit_normal * unit_normal.dot(anchor_ray);
  if (basis_a.norm() <= 1e-9) {
    basis_a = unit_normal.unitOrthogonal();
  } else {
    basis_a.normalize();
  }
  Eigen::Vector3d basis_b = unit_normal.cross(basis_a);
  if (basis_b.norm() <= 1e-9) {
    return rays;
  }
  basis_b.normalize();

  rays.reserve(static_cast<std::size_t>(sample_count));
  for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
    const double alpha =
        sample_count == 1 ? 0.0
                          : static_cast<double>(sample_index) / static_cast<double>(sample_count - 1);
    const double theta = -3.14159265358979323846 +
                         2.0 * 3.14159265358979323846 * alpha;
    Eigen::Vector3d ray = std::cos(theta) * basis_a + std::sin(theta) * basis_b;
    const double ray_norm = ray.norm();
    if (!std::isfinite(ray_norm) || ray_norm <= 1e-9) {
      continue;
    }
    ray /= ray_norm;
    if (ray.z() > 0.0) {
      rays.push_back(ray);
    }
  }
  return rays;
}

cv::Point2f MapRayToSpherePanel(const Eigen::Vector3d& ray,
                                const cv::Point2f& panel_center,
                                float panel_radius) {
  return cv::Point2f(panel_center.x + static_cast<float>(ray.x()) * panel_radius,
                     panel_center.y - static_cast<float>(ray.y()) * panel_radius);
}

bool CvVecToUnitRay(const cv::Vec3d& ray_vec, Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    throw std::runtime_error("CvVecToUnitRay requires a valid output pointer.");
  }
  *ray = Eigen::Vector3d(ray_vec[0], ray_vec[1], ray_vec[2]);
  const double norm = ray->norm();
  if (!std::isfinite(norm) || norm <= 1e-9) {
    return false;
  }
  *ray /= norm;
  return true;
}

bool BuildLocalSphereOffsetRay(const Eigen::Vector3d& anchor_ray,
                               const Eigen::Vector3d& tangent_u,
                               const Eigen::Vector3d& tangent_v,
                               double alpha,
                               double beta,
                               Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    throw std::runtime_error("BuildLocalSphereOffsetRay requires a valid output pointer.");
  }
  Eigen::Vector3d candidate = anchor_ray + alpha * tangent_u + beta * tangent_v;
  const double norm = candidate.norm();
  if (!std::isfinite(norm) || norm <= 1e-9) {
    return false;
  }
  *ray = candidate / norm;
  return true;
}

void DrawRayPolyline(cv::Mat* image,
                     const std::vector<Eigen::Vector3d>& rays,
                     const cv::Point2f& panel_center,
                     float panel_radius,
                     const cv::Scalar& color,
                     int thickness) {
  if (image == nullptr || rays.size() < 2) {
    return;
  }
  for (std::size_t index = 1; index < rays.size(); ++index) {
    cv::line(*image,
             MapRayToSpherePanel(rays[index - 1], panel_center, panel_radius),
             MapRayToSpherePanel(rays[index], panel_center, panel_radius),
             color, thickness, cv::LINE_AA);
  }
}

void DrawSpherePointCallout(cv::Mat* image,
                            const cv::Rect& panel_rect,
                            const cv::Point2f& panel_center,
                            const cv::Point2f& marker_point,
                            const std::string& text,
                            const cv::Scalar& color) {
  if (image == nullptr) {
    return;
  }

  const int baseline = 0;
  const double font_scale = 0.9;
  const int font_thickness = 1;
  const cv::Size text_size =
      cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN, font_scale, font_thickness, nullptr);

  const float horizontal_offset =
      marker_point.x < panel_center.x ? 14.0f : static_cast<float>(-text_size.width - 14);
  const float vertical_offset = marker_point.y < panel_center.y ? -14.0f : 22.0f;

  int box_x = static_cast<int>(std::lround(marker_point.x + horizontal_offset));
  int box_y = static_cast<int>(std::lround(marker_point.y + vertical_offset - text_size.height));
  const int padding_x = 5;
  const int padding_y = 4;
  const int box_width = text_size.width + 2 * padding_x;
  const int box_height = text_size.height + 2 * padding_y;
  box_x = std::max(panel_rect.x + 8, std::min(box_x, panel_rect.x + panel_rect.width - box_width - 8));
  box_y = std::max(panel_rect.y + 8, std::min(box_y, panel_rect.y + panel_rect.height - box_height - 8));

  const cv::Rect box_rect(box_x, box_y, box_width, box_height);
  const cv::Point2f box_anchor(
      static_cast<float>(box_rect.x + (marker_point.x < panel_center.x ? 0 : box_rect.width)),
      static_cast<float>(box_rect.y + box_rect.height * 0.5f));
  cv::line(*image, marker_point, box_anchor, color, 1, cv::LINE_AA);
  cv::rectangle(*image, box_rect, cv::Scalar(255, 255, 255), cv::FILLED);
  cv::rectangle(*image, box_rect, color, 1, cv::LINE_AA);
  cv::putText(*image, text,
              cv::Point(box_rect.x + padding_x, box_rect.y + box_height - padding_y - 1),
              cv::FONT_HERSHEY_PLAIN, font_scale, color, font_thickness, cv::LINE_AA);
}

float ComputePointClusterSpread(const std::vector<cv::Point2f>& points) {
  if (points.empty()) {
    return 10.0f;
  }
  if (points.size() == 1) {
    return 10.0f;
  }

  float max_pairwise_distance = 0.0f;
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (std::size_t j = i + 1; j < points.size(); ++j) {
      max_pairwise_distance =
          std::max(max_pairwise_distance,
                   static_cast<float>(cv::norm(points[i] - points[j])));
    }
  }

  return std::max(10.0f, std::min(26.0f, max_pairwise_distance * 0.9f + 8.0f));
}

cv::Point2f ComputeTriadLabelOffset(int slot, float spread, bool compact) {
  const float gain = compact ? 0.7f : 1.0f;
  switch (slot) {
    case 0:
      return cv::Point2f(-(spread + 20.0f * gain), -(10.0f + 0.55f * spread));
    case 1:
      return cv::Point2f(12.0f + 0.65f * spread, -(6.0f + 0.35f * spread));
    default:
      return cv::Point2f(12.0f + 0.55f * spread, 14.0f + 0.45f * spread);
  }
}

void DrawBoundedCallout(cv::Mat* image,
                        const cv::Rect& bounds,
                        const cv::Point2f& point,
                        const std::string& text,
                        const cv::Scalar& color,
                        const cv::Point2f& offset,
                        double font_scale,
                        int font_thickness) {
  if (image == nullptr) {
    return;
  }

  int baseline = 0;
  const cv::Size text_size =
      cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN, font_scale, font_thickness, &baseline);
  const int padding_x = 5;
  const int padding_y = 4;
  int box_x = static_cast<int>(std::lround(point.x + offset.x));
  int box_y = static_cast<int>(std::lround(point.y + offset.y));
  const int box_width = text_size.width + 2 * padding_x;
  const int box_height = text_size.height + baseline + 2 * padding_y;

  box_x = std::max(bounds.x + 4, std::min(box_x, bounds.x + bounds.width - box_width - 4));
  box_y = std::max(bounds.y + 4, std::min(box_y, bounds.y + bounds.height - box_height - 4));

  const cv::Rect box_rect(box_x, box_y, box_width, box_height);
  const cv::Point2f anchor(
      static_cast<float>(offset.x >= 0.0f ? box_rect.x : box_rect.x + box_rect.width),
      static_cast<float>(box_rect.y + box_rect.height * 0.5f));

  cv::line(*image, point, anchor, color, 1, cv::LINE_AA);
  cv::rectangle(*image, box_rect, cv::Scalar(255, 255, 255), cv::FILLED);
  cv::rectangle(*image, box_rect, color, 1, cv::LINE_AA);
  cv::putText(*image, text,
              cv::Point(box_rect.x + padding_x, box_rect.y + box_rect.height - padding_y - baseline),
              cv::FONT_HERSHEY_PLAIN, font_scale, color, font_thickness, cv::LINE_AA);
}

void DrawInsetLegendCallout(cv::Mat* image,
                            const cv::Rect& inset_rect,
                            const cv::Point2f& point,
                            const std::string& text,
                            const cv::Scalar& color,
                            int slot) {
  if (image == nullptr) {
    return;
  }

  const double font_scale = 0.75;
  const int font_thickness = 1;
  int baseline = 0;
  const cv::Size text_size =
      cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN, font_scale, font_thickness, &baseline);
  const int padding_x = 4;
  const int padding_y = 3;
  const int box_width = text_size.width + 2 * padding_x;
  const int box_height = text_size.height + baseline + 2 * padding_y;

  int box_x = inset_rect.x + 6;
  int box_y = inset_rect.y + 18;
  switch (slot) {
    case 0:
      box_x = inset_rect.x + 6;
      box_y = inset_rect.y + 18;
      break;
    case 1:
      box_x = inset_rect.x + inset_rect.width - box_width - 6;
      box_y = inset_rect.y + 18;
      break;
    default:
      box_x = inset_rect.x + inset_rect.width - box_width - 6;
      box_y = inset_rect.y + inset_rect.height - box_height - 6;
      break;
  }

  const cv::Rect box_rect(box_x, box_y, box_width, box_height);
  const cv::Point2f anchor(
      static_cast<float>(slot == 0 ? box_rect.x : box_rect.x + box_rect.width),
      static_cast<float>(box_rect.y + box_rect.height * 0.5f));
  cv::line(*image, point, anchor, color, 1, cv::LINE_AA);
  cv::rectangle(*image, box_rect, cv::Scalar(255, 255, 255), cv::FILLED);
  cv::rectangle(*image, box_rect, color, 1, cv::LINE_AA);
  cv::putText(*image, text,
              cv::Point(box_rect.x + padding_x,
                        box_rect.y + box_rect.height - padding_y - baseline),
              cv::FONT_HERSHEY_PLAIN, font_scale, color, font_thickness, cv::LINE_AA);
}

cv::Mat BuildOuterSphereDebugView(const ati::ApriltagInternalConfig& config,
                                  const ati::ApriltagInternalDetectionResult& result) {
  if (!config.intermediate_camera.IsConfigured()) {
    return cv::Mat();
  }

  ati::DoubleSphereCameraModel camera;
  try {
    camera = ati::DoubleSphereCameraModel::FromConfig(config.intermediate_camera);
  } catch (const std::exception&) {
    return cv::Mat();
  }
  if (!camera.IsValid()) {
    return cv::Mat();
  }

  bool has_any_sphere_debug = false;
  for (const auto& debug : result.outer_detection.corner_verification_debug) {
    if (!debug.prev_branch_points.empty() || !debug.next_branch_points.empty() ||
        debug.spherical_refinement_valid) {
      has_any_sphere_debug = true;
      break;
    }
  }
  if (!has_any_sphere_debug) {
    return cv::Mat();
  }

  constexpr int kCanvasWidth = 1600;
  constexpr int kCanvasHeight = 1200;
  constexpr int kMargin = 70;
  constexpr int kHeaderHeight = 90;
  const cv::Scalar kBgColor(248, 248, 248);
  const cv::Scalar kPanelColor(255, 255, 255);
  const cv::Scalar kBorderColor(90, 90, 90);
  const cv::Scalar kPrevColor(255, 120, 0);
  const cv::Scalar kNextColor(0, 180, 255);
  const cv::Scalar kCoarseColor(0, 165, 255);
  const cv::Scalar kSubpixColor(255, 255, 0);
  const cv::Scalar kSphereColor(255, 80, 255);
  const cv::Scalar kMoveColor(120, 120, 120);

  cv::Mat canvas(kCanvasHeight, kCanvasWidth, CV_8UC3, kBgColor);
  cv::putText(canvas, "Outer Sphere View: coarse/support rays -> boundary planes -> SP ray",
              cv::Point(40, 46), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(20, 20, 20), 2);
  cv::putText(canvas, "orange/blue dots: support rays, colored arcs: fitted boundary planes, gray segment: C->SP",
              cv::Point(40, 78), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(60, 60, 60), 1);

  const int panel_width = (kCanvasWidth - 3 * kMargin) / 2;
  const int panel_height = (kCanvasHeight - kHeaderHeight - 3 * kMargin) / 2;

  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    const ati::OuterCornerVerificationDebugInfo& debug =
        result.outer_detection.corner_verification_debug[static_cast<std::size_t>(corner_index)];

    const int row = corner_index / 2;
    const int col = corner_index % 2;
    const cv::Rect panel_rect(kMargin + col * (panel_width + kMargin),
                              kHeaderHeight + kMargin + row * (panel_height + kMargin),
                              panel_width, panel_height);
    cv::rectangle(canvas, panel_rect, kPanelColor, cv::FILLED);
    cv::rectangle(canvas, panel_rect, kBorderColor, 1);

    const cv::Point2f panel_center(panel_rect.x + panel_rect.width * 0.5f,
                                   panel_rect.y + panel_rect.height * 0.48f);
    const float panel_radius =
        0.36f * static_cast<float>(std::min(panel_rect.width, panel_rect.height));
    cv::circle(canvas, panel_center, static_cast<int>(std::lround(panel_radius)),
               cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
    cv::line(canvas,
             cv::Point(static_cast<int>(std::lround(panel_center.x - panel_radius)),
                       static_cast<int>(std::lround(panel_center.y))),
             cv::Point(static_cast<int>(std::lround(panel_center.x + panel_radius)),
                       static_cast<int>(std::lround(panel_center.y))),
             cv::Scalar(228, 228, 228), 1, cv::LINE_AA);
    cv::line(canvas,
             cv::Point(static_cast<int>(std::lround(panel_center.x)),
                       static_cast<int>(std::lround(panel_center.y - panel_radius))),
             cv::Point(static_cast<int>(std::lround(panel_center.x)),
                       static_cast<int>(std::lround(panel_center.y + panel_radius))),
             cv::Scalar(228, 228, 228), 1, cv::LINE_AA);

    std::vector<Eigen::Vector3d> prev_rays;
    std::vector<Eigen::Vector3d> next_rays;
    const bool prev_ok = UnprojectImagePointsToRays(camera, debug.prev_branch_points, &prev_rays);
    const bool next_ok = UnprojectImagePointsToRays(camera, debug.next_branch_points, &next_rays);

    Eigen::Vector3d coarse_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d subpix_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d sphere_ray = Eigen::Vector3d::Zero();
    const bool coarse_ok =
        camera.keypointToEuclidean(Eigen::Vector2d(debug.coarse_corner.x, debug.coarse_corner.y), &coarse_ray) &&
        coarse_ray.norm() > 1e-9;
    const bool subpix_ok =
        camera.keypointToEuclidean(Eigen::Vector2d(debug.subpix_corner.x, debug.subpix_corner.y), &subpix_ray) &&
        subpix_ray.norm() > 1e-9;
    const bool sphere_ok =
        camera.keypointToEuclidean(Eigen::Vector2d(debug.spherical_corner.x, debug.spherical_corner.y), &sphere_ray) &&
        sphere_ray.norm() > 1e-9;
    if (coarse_ok) coarse_ray.normalize();
    if (subpix_ok) subpix_ray.normalize();
    if (sphere_ok) sphere_ray.normalize();

    Eigen::Vector3d prev_plane = Eigen::Vector3d::Zero();
    Eigen::Vector3d next_plane = Eigen::Vector3d::Zero();
    double prev_rms = 0.0;
    double next_rms = 0.0;
    const bool prev_plane_ok = prev_ok && FitPlaneToRays(prev_rays, &prev_plane, &prev_rms);
    const bool next_plane_ok = next_ok && FitPlaneToRays(next_rays, &next_plane, &next_rms);

    if (prev_plane_ok) {
      const Eigen::Vector3d anchor = sphere_ok ? sphere_ray : (subpix_ok ? subpix_ray : prev_rays.front());
      DrawRayPolyline(&canvas, SamplePlaneGreatCircle(prev_plane, anchor), panel_center,
                      panel_radius, kPrevColor, 2);
    }
    if (next_plane_ok) {
      const Eigen::Vector3d anchor = sphere_ok ? sphere_ray : (subpix_ok ? subpix_ray : next_rays.front());
      DrawRayPolyline(&canvas, SamplePlaneGreatCircle(next_plane, anchor), panel_center,
                      panel_radius, kNextColor, 2);
    }

    if (prev_ok) {
      for (const Eigen::Vector3d& ray : prev_rays) {
        cv::circle(canvas, MapRayToSpherePanel(ray, panel_center, panel_radius), 3, kPrevColor, -1,
                   cv::LINE_AA);
      }
    }
    if (next_ok) {
      for (const Eigen::Vector3d& ray : next_rays) {
        cv::circle(canvas, MapRayToSpherePanel(ray, panel_center, panel_radius), 3, kNextColor, -1,
                   cv::LINE_AA);
      }
    }

    if (coarse_ok && sphere_ok) {
      cv::arrowedLine(canvas,
                      MapRayToSpherePanel(coarse_ray, panel_center, panel_radius),
                      MapRayToSpherePanel(sphere_ray, panel_center, panel_radius),
                      kMoveColor, 3, cv::LINE_AA, 0, 0.12);
    }

    if (coarse_ok) {
      const cv::Point2f coarse_point = MapRayToSpherePanel(coarse_ray, panel_center, panel_radius);
      cv::circle(canvas, coarse_point, 11, cv::Scalar(240, 245, 255), cv::FILLED, cv::LINE_AA);
      cv::circle(canvas, coarse_point, 8, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
      cv::circle(canvas, coarse_point, 6, kCoarseColor, 2, cv::LINE_AA);
      DrawSpherePointCallout(&canvas, panel_rect, panel_center, coarse_point, "C", kCoarseColor);
    }
    if (debug.subpix_applied && subpix_ok) {
      cv::drawMarker(canvas, MapRayToSpherePanel(subpix_ray, panel_center, panel_radius), kSubpixColor,
                     cv::MARKER_CROSS, 12, 1, cv::LINE_AA);
    }
    if (sphere_ok) {
      const cv::Point2f sphere_point = MapRayToSpherePanel(sphere_ray, panel_center, panel_radius);
      cv::drawMarker(canvas, sphere_point, cv::Scalar(255, 255, 255),
                     cv::MARKER_DIAMOND, 18, 4, cv::LINE_AA);
      cv::drawMarker(canvas, sphere_point, kSphereColor,
                     cv::MARKER_DIAMOND, 14, 2, cv::LINE_AA);
      DrawSpherePointCallout(&canvas, panel_rect, panel_center, sphere_point, "SP", kSphereColor);
    }

    const int text_x = panel_rect.x + 18;
    int text_y = panel_rect.y + panel_rect.height - 96;
    cv::putText(canvas, "corner " + std::to_string(corner_index) + " " + BuildOuterChainLabel(debug),
                cv::Point(text_x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(20, 20, 20), 2);
    text_y += 24;
    std::ostringstream line1;
    line1 << "C=(" << std::lround(debug.coarse_corner.x) << ","
          << std::lround(debug.coarse_corner.y) << ")";
    cv::putText(canvas, line1.str(), cv::Point(text_x, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                kCoarseColor, 1);
    text_y += 22;
    std::ostringstream line2;
    line2 << "SP=(" << std::lround(debug.spherical_corner.x) << ","
          << std::lround(debug.spherical_corner.y) << ")";
    cv::putText(canvas, line2.str(), cv::Point(text_x, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                kSphereColor, 1);
    text_y += 22;
    std::ostringstream line3;
    line3 << "d=" << std::fixed << std::setprecision(1) << debug.coarse_to_refined_displacement
          << "px  n=" << debug.prev_spherical_support_count << "/"
          << debug.next_spherical_support_count
          << "  rms=" << std::setprecision(4) << prev_rms << "/" << next_rms;
    if (!debug.spherical_refinement_valid && !debug.spherical_failure_reason.empty()) {
      line3 << "  " << debug.spherical_failure_reason;
    }
    cv::putText(canvas, line3.str(), cv::Point(text_x, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(50, 50, 50), 1);
  }

  return canvas;
}

cv::Mat BuildInternalSeedOverlay(const cv::Mat& image,
                                 const ati::ApriltagInternalDetectionResult& result) {
  if (!UsesSphereSeedPipelineLocal(result.projection_mode) ||
      result.internal_corner_debug.empty() || image.empty()) {
    return cv::Mat();
  }

  cv::Mat overlay = image.clone();
  if (overlay.channels() == 1) {
    cv::cvtColor(overlay, overlay, cv::COLOR_GRAY2BGR);
  } else if (overlay.channels() == 4) {
    cv::cvtColor(overlay, overlay, cv::COLOR_BGRA2BGR);
  }

  const cv::Scalar kPredictedColor(0, 165, 255);
  const cv::Scalar kBorderSeedColor(255, 180, 60);
  const cv::Scalar kSeedColor(255, 80, 255);
  const cv::Scalar kRefinedColor(0, 220, 80);
  const cv::Scalar kBoundaryUColor(190, 190, 190);
  const cv::Scalar kBoundaryVColor(115, 115, 115);
  const cv::Scalar kArrow1Color(180, 180, 180);
  const cv::Scalar kArrow2Color(120, 190, 120);

  if (result.tag_detected) {
    const cv::Scalar outer_outline_color(165, 165, 165);
    for (int index = 0; index < 4; ++index) {
      cv::line(overlay, result.outer_corners[index], result.outer_corners[(index + 1) % 4],
               outer_outline_color, 2, cv::LINE_AA);
    }
  }

  for (const auto& debug : result.internal_corner_debug) {
    const bool predicted_ok = debug.predicted_image.x >= 0.0f &&
                              debug.predicted_image.x < static_cast<float>(result.image_size.width) &&
                              debug.predicted_image.y >= 0.0f &&
                              debug.predicted_image.y < static_cast<float>(result.image_size.height);
    const bool seed_ok = debug.sphere_seed_image.x >= 0.0f &&
                         debug.sphere_seed_image.x < static_cast<float>(result.image_size.width) &&
                         debug.sphere_seed_image.y >= 0.0f &&
                         debug.sphere_seed_image.y < static_cast<float>(result.image_size.height);
    const bool border_seed_ok =
        debug.border_seed_valid &&
        debug.border_seed_image.x >= 0.0f &&
        debug.border_seed_image.x < static_cast<float>(result.image_size.width) &&
        debug.border_seed_image.y >= 0.0f &&
        debug.border_seed_image.y < static_cast<float>(result.image_size.height);
	    const bool refined_ok = debug.refined_image.x >= 0.0f &&
	                            debug.refined_image.x < static_cast<float>(result.image_size.width) &&
	                            debug.refined_image.y >= 0.0f &&
	                            debug.refined_image.y < static_cast<float>(result.image_size.height);
    if (!predicted_ok) {
      continue;
    }

    const cv::Point2f boundary_center = seed_ok ? debug.sphere_seed_image : debug.predicted_image;
    const double module_u_length = std::hypot(debug.module_u_axis.x, debug.module_u_axis.y);
    const double module_v_length = std::hypot(debug.module_v_axis.x, debug.module_v_axis.y);
    if (module_u_length > 1.0 && module_v_length > 1.0) {
      const cv::Point2f unit_u =
          debug.module_u_axis * static_cast<float>(1.0 / std::max(1e-9, module_u_length));
      const cv::Point2f unit_v =
          debug.module_v_axis * static_cast<float>(1.0 / std::max(1e-9, module_v_length));
      const float u_half_length = std::max(6.0f, static_cast<float>(0.55 * module_v_length));
      const float v_half_length = std::max(6.0f, static_cast<float>(0.55 * module_u_length));
      cv::line(overlay, boundary_center - u_half_length * unit_v,
               boundary_center + u_half_length * unit_v, kBoundaryUColor, 1, cv::LINE_AA);
      cv::line(overlay, boundary_center - v_half_length * unit_u,
               boundary_center + v_half_length * unit_u, kBoundaryVColor, 1, cv::LINE_AA);
    }

    const float search_radius_px =
        std::max(6.0f, static_cast<float>(0.35 * std::max(1.0, debug.local_module_scale)));
    cv::circle(overlay, debug.predicted_image, static_cast<int>(std::lround(search_radius_px)),
               cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    cv::drawMarker(overlay, debug.predicted_image, cv::Scalar(255, 255, 255),
                   cv::MARKER_CROSS, 8, 3, cv::LINE_AA);
    cv::drawMarker(overlay, debug.predicted_image, kPredictedColor,
                   cv::MARKER_CROSS, 6, 1, cv::LINE_AA);
    cv::circle(overlay, debug.predicted_image, 2, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
    cv::circle(overlay, debug.predicted_image, 1, kPredictedColor, cv::FILLED, cv::LINE_AA);

    if (border_seed_ok) {
      cv::arrowedLine(overlay, debug.predicted_image, debug.border_seed_image,
                      kArrow1Color, 1, cv::LINE_AA, 0, 0.12);
      cv::drawMarker(overlay, debug.border_seed_image, cv::Scalar(255, 255, 255),
                     cv::MARKER_TRIANGLE_UP, 8, 3, cv::LINE_AA);
      cv::drawMarker(overlay, debug.border_seed_image, kBorderSeedColor,
                     cv::MARKER_TRIANGLE_UP, 6, 1, cv::LINE_AA);
    }
	    if (seed_ok) {
      cv::arrowedLine(overlay, border_seed_ok ? debug.border_seed_image : debug.predicted_image,
                      debug.sphere_seed_image, kArrow1Color, 1, cv::LINE_AA, 0, 0.15);
      if (result.projection_mode == ati::InternalProjectionMode::SphereRayRefine &&
          debug.ray_refine_trust_radius > 0.0) {
        const int trust_radius_px = std::max(
            4, static_cast<int>(std::lround(0.20 * std::max(1.0, debug.local_module_scale))));
        cv::circle(overlay, debug.predicted_image, trust_radius_px, cv::Scalar(205, 205, 205),
                   1, cv::LINE_AA);
      }
      cv::drawMarker(overlay, debug.sphere_seed_image, cv::Scalar(255, 255, 255),
                     cv::MARKER_DIAMOND, 8, 3, cv::LINE_AA);
	      cv::drawMarker(overlay, debug.sphere_seed_image, kSeedColor,
	                     cv::MARKER_DIAMOND, 6, 1, cv::LINE_AA);
	    }
	    if (seed_ok && refined_ok) {
      cv::arrowedLine(overlay, debug.sphere_seed_image, debug.refined_image,
	                      kArrow2Color, 1, cv::LINE_AA, 0, 0.15);
    } else if (predicted_ok && refined_ok) {
      cv::arrowedLine(overlay, debug.predicted_image, debug.refined_image,
                      kArrow2Color, 1, cv::LINE_AA, 0, 0.15);
    }
    if (refined_ok) {
      cv::drawMarker(overlay, debug.refined_image, cv::Scalar(255, 255, 255),
                     cv::MARKER_SQUARE, 7, 3, cv::LINE_AA);
      cv::drawMarker(overlay, debug.refined_image, kRefinedColor,
                     cv::MARKER_SQUARE, 5, 1, cv::LINE_AA);
    }
  }

  const std::string title =
      result.projection_mode == ati::InternalProjectionMode::SphereRayRefine
          ? "Internal Ray-Seed Overlay: P -> BC -> SS(ray) -> R(subpix)"
          : "Internal Sphere Seed Overlay: P -> BC -> SS -> R";
  const std::string legend =
      result.projection_mode == ati::InternalProjectionMode::SphereRayRefine
          ? "Legend: P orange cross, BC blue triangle, SS magenta diamond, R green square, gray circle: predicted-ray trust region"
          : "Legend: P orange cross, BC blue triangle, SS magenta diamond, R green square, gray cross: aligned lattice boundaries";
  cv::putText(overlay, title,
              cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 3,
              cv::LINE_AA);
  cv::putText(overlay, title,
              cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(30, 30, 30), 1,
              cv::LINE_AA);
  cv::putText(overlay, legend,
              cv::Point(20, 56), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 3,
              cv::LINE_AA);
  cv::putText(overlay, legend,
              cv::Point(20, 56), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(40, 40, 40), 1,
              cv::LINE_AA);
  return overlay;
}

cv::Mat BuildInternalSphereDebugView(const ati::ApriltagInternalDetectionResult& result) {
  if (!UsesSphereSeedPipelineLocal(result.projection_mode) ||
      result.internal_corner_debug.empty()) {
    return cv::Mat();
  }

  const int panel_columns = 4;
  const int panel_width = 320;
  const int panel_height = 230;
  const int margin = 26;
  const int header_height = 85;
  const int panel_count = static_cast<int>(result.internal_corner_debug.size());
  const int panel_rows = std::max(1, (panel_count + panel_columns - 1) / panel_columns);
  const int canvas_width = panel_columns * panel_width + (panel_columns + 1) * margin;
  const int canvas_height = header_height + panel_rows * panel_height + (panel_rows + 1) * margin;

  cv::Mat canvas(canvas_height, canvas_width, CV_8UC3, cv::Scalar(248, 248, 248));
  const std::string header =
      result.projection_mode == ati::InternalProjectionMode::SphereRayRefine
          ? "Internal Sphere View: predicted ray -> ray-domain seed -> subpixel refined ray"
          : "Internal Sphere View: predicted ray -> sphere seed -> refined ray";
  const std::string subtitle =
      result.projection_mode == ati::InternalProjectionMode::SphereRayRefine
          ? "P orange, SS magenta, R green. Gray arrow: P->SS, green arrow: SS->R. Gray cross: aligned lattice boundaries. Gray ring: predicted-ray trust region."
          : "P orange, SS magenta, R green. Gray arrow: P->SS, green arrow: SS->R. Gray cross: aligned lattice boundaries.";
  cv::putText(canvas, header,
              cv::Point(28, 40), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(20, 20, 20), 2);
  cv::putText(canvas, subtitle,
              cv::Point(28, 70), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(60, 60, 60), 1);

  const cv::Scalar kPredictedColor(0, 165, 255);
  const cv::Scalar kSeedColor(255, 80, 255);
  const cv::Scalar kRefinedColor(0, 220, 80);
  const cv::Scalar kBoundaryUColor(190, 190, 190);
  const cv::Scalar kBoundaryVColor(115, 115, 115);
  const cv::Scalar kUAxisColor(150, 150, 150);
  const cv::Scalar kVAxisColor(90, 90, 90);
  const cv::Scalar kSearchBoxColor(190, 190, 190);
  const cv::Scalar kArrow1Color(180, 180, 180);
  const cv::Scalar kArrow2Color(120, 190, 120);

  for (int index = 0; index < panel_count; ++index) {
    const auto& debug = result.internal_corner_debug[static_cast<std::size_t>(index)];
    const int row = index / panel_columns;
    const int col = index % panel_columns;
    const cv::Rect panel_rect(margin + col * (panel_width + margin),
                              header_height + margin + row * (panel_height + margin),
                              panel_width, panel_height);
    cv::rectangle(canvas, panel_rect, cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(canvas, panel_rect, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);

    const cv::Point2f center(panel_rect.x + panel_rect.width * 0.5f,
                             panel_rect.y + panel_rect.height * 0.43f);
    const float radius = 0.33f * static_cast<float>(std::min(panel_rect.width, panel_rect.height));
    cv::circle(canvas, center, static_cast<int>(std::lround(radius)),
               cv::Scalar(215, 215, 215), 1, cv::LINE_AA);
    cv::line(canvas,
             cv::Point(static_cast<int>(std::lround(center.x - radius)), static_cast<int>(std::lround(center.y))),
             cv::Point(static_cast<int>(std::lround(center.x + radius)), static_cast<int>(std::lround(center.y))),
             cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    cv::line(canvas,
             cv::Point(static_cast<int>(std::lround(center.x)), static_cast<int>(std::lround(center.y - radius))),
             cv::Point(static_cast<int>(std::lround(center.x)), static_cast<int>(std::lround(center.y + radius))),
             cv::Scalar(230, 230, 230), 1, cv::LINE_AA);

    Eigen::Vector3d predicted_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d seed_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d refined_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent_u = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent_v = Eigen::Vector3d::Zero();
    cv::Point2f predicted_point{};
    cv::Point2f seed_point{};
    cv::Point2f refined_point{};
    cv::Point2f u_plus_point{};
    cv::Point2f v_plus_point{};
    std::array<cv::Point2f, 4> search_box_points{};
    std::array<cv::Point2f, 2> boundary_u_points{};
    std::array<cv::Point2f, 2> boundary_v_points{};
    std::vector<cv::Point2f> trust_circle_points;
    bool search_box_ok = false;
    bool u_plus_ok = false;
    bool v_plus_ok = false;
    bool boundary_u_ok = false;
    bool boundary_v_ok = false;
    const bool predicted_ok = CvVecToUnitRay(debug.predicted_ray, &predicted_ray);
    const bool seed_ok = CvVecToUnitRay(debug.sphere_seed_ray, &seed_ray);
    const bool refined_ok = CvVecToUnitRay(debug.refined_ray, &refined_ray);
    const bool tangent_u_ok = CvVecToUnitRay(debug.tangent_u_ray, &tangent_u);
    const bool tangent_v_ok = CvVecToUnitRay(debug.tangent_v_ray, &tangent_v);

    if (predicted_ok && tangent_u_ok && tangent_v_ok && debug.sphere_search_radius > 1e-9) {
      const double r = debug.sphere_search_radius;
      Eigen::Vector3d u_plus = Eigen::Vector3d::Zero();
      Eigen::Vector3d u_minus = Eigen::Vector3d::Zero();
      Eigen::Vector3d v_plus = Eigen::Vector3d::Zero();
      Eigen::Vector3d v_minus = Eigen::Vector3d::Zero();
      Eigen::Vector3d c00 = Eigen::Vector3d::Zero();
      Eigen::Vector3d c10 = Eigen::Vector3d::Zero();
      Eigen::Vector3d c11 = Eigen::Vector3d::Zero();
      Eigen::Vector3d c01 = Eigen::Vector3d::Zero();
      if (BuildLocalSphereOffsetRay(predicted_ray, tangent_u, tangent_v, r, 0.0, &u_plus) &&
          BuildLocalSphereOffsetRay(predicted_ray, tangent_u, tangent_v, -r, 0.0, &u_minus) &&
          BuildLocalSphereOffsetRay(predicted_ray, tangent_u, tangent_v, 0.0, r, &v_plus) &&
          BuildLocalSphereOffsetRay(predicted_ray, tangent_u, tangent_v, 0.0, -r, &v_minus) &&
          BuildLocalSphereOffsetRay(predicted_ray, tangent_u, tangent_v, -r, -r, &c00) &&
          BuildLocalSphereOffsetRay(predicted_ray, tangent_u, tangent_v, r, -r, &c10) &&
          BuildLocalSphereOffsetRay(predicted_ray, tangent_u, tangent_v, r, r, &c11) &&
          BuildLocalSphereOffsetRay(predicted_ray, tangent_u, tangent_v, -r, r, &c01)) {
        const std::array<cv::Point2f, 4> search_box{{
            MapRayToSpherePanel(c00, center, radius),
            MapRayToSpherePanel(c10, center, radius),
            MapRayToSpherePanel(c11, center, radius),
            MapRayToSpherePanel(c01, center, radius),
        }};
        search_box_points = search_box;
        search_box_ok = true;
        for (std::size_t edge_index = 0; edge_index < search_box.size(); ++edge_index) {
          cv::line(canvas, search_box[edge_index],
                   search_box[(edge_index + 1) % search_box.size()],
                   kSearchBoxColor, 1, cv::LINE_AA);
        }
        u_plus_point = MapRayToSpherePanel(u_plus, center, radius);
        v_plus_point = MapRayToSpherePanel(v_plus, center, radius);
        u_plus_ok = true;
        v_plus_ok = true;
        cv::arrowedLine(canvas, MapRayToSpherePanel(predicted_ray, center, radius),
                        u_plus_point,
                        kUAxisColor, 1, cv::LINE_AA, 0, 0.15);
        cv::arrowedLine(canvas, MapRayToSpherePanel(predicted_ray, center, radius),
                        v_plus_point,
                        kVAxisColor, 1, cv::LINE_AA, 0, 0.15);
        cv::putText(canvas, "u",
                    u_plus_point + cv::Point2f(6.0f, -4.0f),
                    cv::FONT_HERSHEY_PLAIN, 0.8, kUAxisColor, 1, cv::LINE_AA);
        cv::putText(canvas, "v",
                    v_plus_point + cv::Point2f(6.0f, -4.0f),
                    cv::FONT_HERSHEY_PLAIN, 0.8, kVAxisColor, 1, cv::LINE_AA);
      }
    }

    if (seed_ok && tangent_u_ok && tangent_v_ok) {
      auto project_to_seed_tangent = [&](const Eigen::Vector3d& source_tangent,
                                         Eigen::Vector3d* projected) {
        if (projected == nullptr) {
          return false;
        }
        *projected = source_tangent - seed_ray * seed_ray.dot(source_tangent);
        const double norm = projected->norm();
        if (!std::isfinite(norm) || norm <= 1e-9) {
          return false;
        }
        *projected /= norm;
        return true;
      };

      Eigen::Vector3d seed_tangent_u = Eigen::Vector3d::Zero();
      Eigen::Vector3d seed_tangent_v = Eigen::Vector3d::Zero();
      if (project_to_seed_tangent(tangent_u, &seed_tangent_u) &&
          project_to_seed_tangent(tangent_v, &seed_tangent_v)) {
        const double boundary_extent = std::max(
            0.06, std::min(0.18, 0.75 * std::max(debug.sphere_search_radius, 0.08)));
        Eigen::Vector3d boundary_u_minus = Eigen::Vector3d::Zero();
        Eigen::Vector3d boundary_u_plus = Eigen::Vector3d::Zero();
        Eigen::Vector3d boundary_v_minus = Eigen::Vector3d::Zero();
        Eigen::Vector3d boundary_v_plus = Eigen::Vector3d::Zero();
        if (BuildLocalSphereOffsetRay(seed_ray, seed_tangent_u, seed_tangent_v, 0.0,
                                      -boundary_extent, &boundary_u_minus) &&
            BuildLocalSphereOffsetRay(seed_ray, seed_tangent_u, seed_tangent_v, 0.0,
                                      boundary_extent, &boundary_u_plus)) {
          boundary_u_points = {{
              MapRayToSpherePanel(boundary_u_minus, center, radius),
              MapRayToSpherePanel(boundary_u_plus, center, radius),
          }};
          boundary_u_ok = true;
          cv::line(canvas, boundary_u_points[0], boundary_u_points[1], kBoundaryUColor, 1,
                   cv::LINE_AA);
        }
        if (BuildLocalSphereOffsetRay(seed_ray, seed_tangent_u, seed_tangent_v,
                                      -boundary_extent, 0.0, &boundary_v_minus) &&
            BuildLocalSphereOffsetRay(seed_ray, seed_tangent_u, seed_tangent_v,
                                      boundary_extent, 0.0, &boundary_v_plus)) {
          boundary_v_points = {{
              MapRayToSpherePanel(boundary_v_minus, center, radius),
              MapRayToSpherePanel(boundary_v_plus, center, radius),
          }};
          boundary_v_ok = true;
          cv::line(canvas, boundary_v_points[0], boundary_v_points[1], kBoundaryVColor, 1,
                   cv::LINE_AA);
        }
      }
    }

    if (result.projection_mode == ati::InternalProjectionMode::SphereRayRefine &&
        predicted_ok && tangent_u_ok && tangent_v_ok && debug.ray_refine_trust_radius > 1e-9) {
        constexpr int kTrustCircleSamples = 32;
        trust_circle_points.reserve(kTrustCircleSamples);
        for (int sample_index = 0; sample_index < kTrustCircleSamples; ++sample_index) {
          const double theta =
              2.0 * M_PI * static_cast<double>(sample_index) / static_cast<double>(kTrustCircleSamples);
          Eigen::Vector3d trust_ray = Eigen::Vector3d::Zero();
          if (!BuildLocalSphereOffsetRay(predicted_ray, tangent_u, tangent_v,
                                         debug.ray_refine_trust_radius * std::cos(theta),
                                         debug.ray_refine_trust_radius * std::sin(theta),
                                         &trust_ray)) {
            continue;
          }
          trust_circle_points.push_back(MapRayToSpherePanel(trust_ray, center, radius));
        }
        if (trust_circle_points.size() >= 2) {
          for (std::size_t edge_index = 0; edge_index < trust_circle_points.size(); ++edge_index) {
            cv::line(canvas, trust_circle_points[edge_index],
                     trust_circle_points[(edge_index + 1) % trust_circle_points.size()],
                     cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
          }
        }
    }

    if (predicted_ok && seed_ok) {
      cv::arrowedLine(canvas, MapRayToSpherePanel(predicted_ray, center, radius),
                      MapRayToSpherePanel(seed_ray, center, radius),
                      kArrow1Color, 2, cv::LINE_AA, 0, 0.14);
    }
    if (seed_ok && refined_ok) {
      cv::arrowedLine(canvas, MapRayToSpherePanel(seed_ray, center, radius),
                      MapRayToSpherePanel(refined_ray, center, radius),
                      kArrow2Color, 2, cv::LINE_AA, 0, 0.14);
    }

    if (predicted_ok) {
      predicted_point = MapRayToSpherePanel(predicted_ray, center, radius);
      cv::drawMarker(canvas, predicted_point, cv::Scalar(255, 255, 255),
                     cv::MARKER_CROSS, 9, 3, cv::LINE_AA);
      cv::drawMarker(canvas, predicted_point, kPredictedColor, cv::MARKER_CROSS, 7, 1, cv::LINE_AA);
      cv::circle(canvas, predicted_point, 2, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
      cv::circle(canvas, predicted_point, 1, kPredictedColor, cv::FILLED, cv::LINE_AA);
    }
    if (seed_ok) {
      seed_point = MapRayToSpherePanel(seed_ray, center, radius);
      cv::drawMarker(canvas, seed_point, cv::Scalar(255, 255, 255),
                     cv::MARKER_DIAMOND, 9, 3, cv::LINE_AA);
      cv::drawMarker(canvas, seed_point, kSeedColor, cv::MARKER_DIAMOND, 7, 1, cv::LINE_AA);
    }
    if (refined_ok) {
      refined_point = MapRayToSpherePanel(refined_ray, center, radius);
      cv::drawMarker(canvas, refined_point, cv::Scalar(255, 255, 255),
                     cv::MARKER_SQUARE, 8, 3, cv::LINE_AA);
      cv::drawMarker(canvas, refined_point, kRefinedColor,
                     cv::MARKER_SQUARE, 6, 1, cv::LINE_AA);
    }

    const cv::Rect inset_rect(panel_rect.x + panel_rect.width - 102, panel_rect.y + 12, 90, 90);
    cv::rectangle(canvas, inset_rect, cv::Scalar(252, 252, 252), cv::FILLED);
    cv::rectangle(canvas, inset_rect, cv::Scalar(150, 150, 150), 1, cv::LINE_AA);
    cv::putText(canvas, "zoom", cv::Point(inset_rect.x + 8, inset_rect.y + 14),
                cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);

    std::vector<cv::Point2f> zoom_points;
    if (predicted_ok) zoom_points.push_back(predicted_point);
    if (seed_ok) zoom_points.push_back(seed_point);
    if (refined_ok) zoom_points.push_back(refined_point);
    if (search_box_ok) {
      zoom_points.insert(zoom_points.end(), search_box_points.begin(), search_box_points.end());
    }
    if (boundary_u_ok) {
      zoom_points.insert(zoom_points.end(), boundary_u_points.begin(), boundary_u_points.end());
    }
    if (boundary_v_ok) {
      zoom_points.insert(zoom_points.end(), boundary_v_points.begin(), boundary_v_points.end());
    }
    if (u_plus_ok) zoom_points.push_back(u_plus_point);
    if (v_plus_ok) zoom_points.push_back(v_plus_point);
    zoom_points.insert(zoom_points.end(), trust_circle_points.begin(), trust_circle_points.end());

    if (!zoom_points.empty()) {
      float min_x = zoom_points.front().x;
      float max_x = zoom_points.front().x;
      float min_y = zoom_points.front().y;
      float max_y = zoom_points.front().y;
      for (const cv::Point2f& point : zoom_points) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
      }

      const float extent = std::max({max_x - min_x, max_y - min_y, 12.0f});
      const float padding = 0.45f * extent + 4.0f;
      min_x -= padding;
      max_x += padding;
      min_y -= padding;
      max_y += padding;
      const float inner_width = static_cast<float>(inset_rect.width - 12);
      const float inner_height = static_cast<float>(inset_rect.height - 22);
      const float scale_x = inner_width / std::max(1.0f, max_x - min_x);
      const float scale_y = inner_height / std::max(1.0f, max_y - min_y);
      const float zoom_scale = std::min(scale_x, scale_y);

      auto map_to_inset = [&](const cv::Point2f& point) {
        return cv::Point2f(
            static_cast<float>(inset_rect.x + 6) + (point.x - min_x) * zoom_scale,
            static_cast<float>(inset_rect.y + 18) + (point.y - min_y) * zoom_scale);
      };

      const cv::Point2f inset_center(
          static_cast<float>(inset_rect.x + inset_rect.width * 0.5f),
          static_cast<float>(inset_rect.y + inset_rect.height * 0.58f));
      cv::line(canvas,
               cv::Point(inset_rect.x + 6, static_cast<int>(std::lround(inset_center.y))),
               cv::Point(inset_rect.x + inset_rect.width - 6,
                         static_cast<int>(std::lround(inset_center.y))),
               cv::Scalar(236, 236, 236), 1, cv::LINE_AA);
      cv::line(canvas,
               cv::Point(static_cast<int>(std::lround(inset_center.x)), inset_rect.y + 18),
               cv::Point(static_cast<int>(std::lround(inset_center.x)),
                         inset_rect.y + inset_rect.height - 6),
               cv::Scalar(236, 236, 236), 1, cv::LINE_AA);

      if (search_box_ok) {
        std::array<cv::Point2f, 4> mapped_box{};
        for (std::size_t edge_index = 0; edge_index < search_box_points.size(); ++edge_index) {
          mapped_box[edge_index] = map_to_inset(search_box_points[edge_index]);
        }
        for (std::size_t edge_index = 0; edge_index < mapped_box.size(); ++edge_index) {
          cv::line(canvas, mapped_box[edge_index],
                   mapped_box[(edge_index + 1) % mapped_box.size()],
                   kSearchBoxColor, 1, cv::LINE_AA);
        }
      }
      if (boundary_u_ok) {
        cv::line(canvas, map_to_inset(boundary_u_points[0]), map_to_inset(boundary_u_points[1]),
                 kBoundaryUColor, 1, cv::LINE_AA);
      }
      if (boundary_v_ok) {
        cv::line(canvas, map_to_inset(boundary_v_points[0]), map_to_inset(boundary_v_points[1]),
                 kBoundaryVColor, 1, cv::LINE_AA);
      }
      if (trust_circle_points.size() >= 2) {
        std::vector<cv::Point2f> mapped_trust_circle;
        mapped_trust_circle.reserve(trust_circle_points.size());
        for (const cv::Point2f& point : trust_circle_points) {
          mapped_trust_circle.push_back(map_to_inset(point));
        }
        for (std::size_t edge_index = 0; edge_index < mapped_trust_circle.size(); ++edge_index) {
          cv::line(canvas, mapped_trust_circle[edge_index],
                   mapped_trust_circle[(edge_index + 1) % mapped_trust_circle.size()],
                   cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
        }
      }
      if (predicted_ok && u_plus_ok) {
        cv::arrowedLine(canvas, map_to_inset(predicted_point), map_to_inset(u_plus_point),
                        kUAxisColor, 1, cv::LINE_AA, 0, 0.15);
      }
      if (predicted_ok && v_plus_ok) {
        cv::arrowedLine(canvas, map_to_inset(predicted_point), map_to_inset(v_plus_point),
                        kVAxisColor, 1, cv::LINE_AA, 0, 0.15);
      }
      if (predicted_ok && seed_ok) {
        cv::arrowedLine(canvas, map_to_inset(predicted_point), map_to_inset(seed_point),
                        kArrow1Color, 1, cv::LINE_AA, 0, 0.12);
      }
      if (seed_ok && refined_ok) {
        cv::arrowedLine(canvas, map_to_inset(seed_point), map_to_inset(refined_point),
                        kArrow2Color, 1, cv::LINE_AA, 0, 0.12);
      }

      if (predicted_ok) {
        const cv::Point2f point = map_to_inset(predicted_point);
        cv::drawMarker(canvas, point, cv::Scalar(255, 255, 255),
                       cv::MARKER_CROSS, 9, 3, cv::LINE_AA);
        cv::drawMarker(canvas, point, kPredictedColor, cv::MARKER_CROSS, 7, 1, cv::LINE_AA);
        cv::circle(canvas, point, 2, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, point, 1, kPredictedColor, cv::FILLED, cv::LINE_AA);
        DrawInsetLegendCallout(&canvas, inset_rect, point, "P", kPredictedColor, 0);
      }
      if (seed_ok) {
        const cv::Point2f point = map_to_inset(seed_point);
        cv::drawMarker(canvas, point, cv::Scalar(255, 255, 255),
                       cv::MARKER_DIAMOND, 8, 3, cv::LINE_AA);
        cv::drawMarker(canvas, point, kSeedColor, cv::MARKER_DIAMOND, 6, 1, cv::LINE_AA);
        DrawInsetLegendCallout(&canvas, inset_rect, point, "SS", kSeedColor, 1);
      }
      if (refined_ok) {
        const cv::Point2f point = map_to_inset(refined_point);
        cv::drawMarker(canvas, point, cv::Scalar(255, 255, 255),
                       cv::MARKER_SQUARE, 7, 3, cv::LINE_AA);
        cv::drawMarker(canvas, point, kRefinedColor,
                       cv::MARKER_SQUARE, 5, 1, cv::LINE_AA);
        DrawInsetLegendCallout(&canvas, inset_rect, point, "R", kRefinedColor, 2);
      }
    }

    int text_y = panel_rect.y + panel_rect.height - 72;
    std::ostringstream title;
    title << "id " << debug.point_id << " "
          << (debug.corner_type == ati::CornerType::XCorner ? "X" : "L")
          << (debug.valid ? " valid" : " invalid");
    cv::putText(canvas, title.str(), cv::Point(panel_rect.x + 12, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
    text_y += 20;
    std::ostringstream line1;
    line1 << "u=" << std::lround(debug.sphere_template_quality * 100.0)
          << " v=" << std::lround(debug.sphere_gradient_quality * 100.0)
          << " seed=" << std::lround(debug.sphere_seed_quality * 100.0);
    cv::putText(canvas, line1.str(), cv::Point(panel_rect.x + 12, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    text_y += 18;
    std::ostringstream line2;
    line2 << "P->SS " << std::fixed << std::setprecision(1) << debug.predicted_to_seed_displacement
          << "  SS->R " << debug.seed_to_refined_displacement;
    cv::putText(canvas, line2.str(), cv::Point(panel_rect.x + 12, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    text_y += 18;
    std::ostringstream line3;
    line3 << "P->R " << std::fixed << std::setprecision(1)
          << debug.predicted_to_refined_displacement
          << "  r=" << std::setprecision(4) << debug.sphere_search_radius;
    cv::putText(canvas, line3.str(), cv::Point(panel_rect.x + 12, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    if (result.projection_mode == ati::InternalProjectionMode::SphereRayRefine) {
      text_y += 18;
      std::ostringstream line4;
      line4 << "edge=" << std::lround(debug.ray_refine_edge_quality * 100.0)
            << " photo=" << std::lround(debug.ray_refine_photometric_quality * 100.0)
            << " ray=" << std::lround(debug.ray_refine_final_quality * 100.0);
      cv::putText(canvas, line4.str(), cv::Point(panel_rect.x + 12, text_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
      text_y += 18;
      std::ostringstream line5;
      line5 << "tr=" << std::setprecision(5) << debug.ray_refine_trust_radius
            << " it=" << debug.ray_refine_iterations
            << " conv=" << (debug.ray_refine_converged ? "yes" : "no")
            << " ang=" << std::setprecision(4) << debug.seed_to_refined_angular;
      cv::putText(canvas, line5.str(), cv::Point(panel_rect.x + 12, text_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    }
  }

  return canvas;
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
    summary.avg_sphere_seed_quality += debug.sphere_seed_quality;
    summary.avg_ray_refine_edge_quality += debug.ray_refine_edge_quality;
    summary.avg_ray_refine_photometric_quality += debug.ray_refine_photometric_quality;
    summary.avg_ray_refine_final_quality += debug.ray_refine_final_quality;
    summary.avg_seed_to_refined_angular += debug.seed_to_refined_angular;
    if (debug.border_seed_valid) {
      summary.avg_predicted_to_border_seed += debug.predicted_to_border_seed_displacement;
      ++summary.border_seed_count;
    }
    summary.avg_predicted_to_seed += debug.predicted_to_seed_displacement;
    summary.avg_seed_to_refined += debug.seed_to_refined_displacement;
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
  summary.avg_sphere_seed_quality /= count;
  summary.avg_ray_refine_edge_quality /= count;
  summary.avg_ray_refine_photometric_quality /= count;
  summary.avg_ray_refine_final_quality /= count;
  summary.avg_seed_to_refined_angular /= count;
  if (summary.border_seed_count > 0) {
    summary.avg_predicted_to_border_seed /= static_cast<double>(summary.border_seed_count);
  }
  summary.avg_predicted_to_seed /= count;
  summary.avg_seed_to_refined /= count;
  summary.avg_predicted_to_refined /= count;
  return summary;
}

void PrintBorderConditionedDiagnostics(const ati::ApriltagInternalDetectionResult& result) {
  if (result.projection_mode != ati::InternalProjectionMode::SphereBorderLattice &&
      result.projection_mode != ati::InternalProjectionMode::PureSphericalBoundarySeed) {
    return;
  }

  const InternalMetricsSummary metrics = SummarizeInternalCorners(result.internal_corner_debug);
  if (metrics.border_seed_count > 0) {
    std::cout << "  mean |P-BC|: " << std::fixed << std::setprecision(2)
              << metrics.avg_predicted_to_border_seed
              << " over " << metrics.border_seed_count << " points\n";
  } else {
    std::cout << "  mean |P-BC|: (no valid BC points)\n";
  }
  const std::array<const char*, 4> edge_names{{"top", "right", "bottom", "left"}};
  for (std::size_t edge_index = 0; edge_index < edge_names.size(); ++edge_index) {
    std::cout << "  BC edge " << edge_names[edge_index] << ": ";
    if (result.border_edge_valid[edge_index]) {
      std::cout << "rms=" << std::fixed << std::setprecision(4)
                << result.border_edge_rms_residual[edge_index]
                << " n=" << result.border_edge_support_count[edge_index] << "\n";
    } else {
      std::cout << "invalid\n";
    }
  }
  std::cout << std::defaultfloat << std::setprecision(6);
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
    if (args.no_debug_output) {
      config.enable_debug_output = false;
    }
    // Respect the method configured in YAML unless --mode explicitly overrides it.
    // This entry point still fixes the outer path to the interactive default:
    // outer: C -> adaptive subpixel. Internal mode is now whatever the config requests.
    config.outer_detector_config.enable_outer_spherical_refinement = false;
    config.outer_detector_config.do_outer_subpix_refinement = true;
    ati::DoubleSphereCameraModel rectified_roi_camera =
        ati::DoubleSphereCameraModel::FromConfig(config.intermediate_camera);
    if (args.rectified_roi_demo && !rectified_roi_camera.IsValid()) {
      std::cout << "warning: --rectified-roi-demo could not build a valid DS "
                   "camera from config; DS ray-space patch will be skipped.\n";
    }
    ati::ApriltagInternalDetectionOptions options = MakeDetectionOptionsFromConfig(config);
    options.do_subpix_refinement = !args.no_subpix;

    ati::ApriltagInternalDetector detector(config, options);
    const auto process_image = [&](const std::string& image_path,
                                   const std::string& requested_output_path,
                                   bool show_image) -> bool {
      cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
      if (image.empty()) {
        throw std::runtime_error("Failed to read image: " + image_path);
      }

      const std::string minute_stamp = BuildMinuteStamp();

      std::cout << "\n==================================================\n";
      std::cout << "Processing image: " << image_path << "\n";

      if (detector.requested_board_ids().size() > 1) {
        const ati::ApriltagInternalMultiDetectionResult multi_result =
            detector.DetectMultiple(image);

        std::string output_path;
        cv::Mat combined_overlay;
        if (!args.no_output_images || show_image || args.rectified_roi_demo) {
          EnsureParentDirectoryExists(requested_output_path);
          combined_overlay = args.final_corners_only
                                 ? BuildFinalCornersOnlyOverlay(
                                       multi_result, image, args.corner_radius)
                                 : image.clone();
          if (!args.final_corners_only) {
            detector.DrawDetections(multi_result, &combined_overlay);
          }

          output_path = AppendMinuteStamp(requested_output_path, minute_stamp);
          if (!cv::imwrite(output_path, combined_overlay)) {
            throw std::runtime_error("Failed to write output image: " + output_path);
          }
          if (config.enable_debug_output) {
            for (const ati::ApriltagInternalDetectionResult& board_result :
                 multi_result.detections) {
              cv::Mat internal_seed_overlay =
                  ati::BuildInternalSeedOverlay(image, board_result);
              if (!internal_seed_overlay.empty()) {
                const std::string seed_path = AppendStemSuffix(
                    output_path,
                    "_board" + std::to_string(board_result.board_id) +
                        "_internal_seed");
                if (!cv::imwrite(seed_path, internal_seed_overlay)) {
                  throw std::runtime_error(
                      "Failed to write internal seed overlay: " + seed_path);
                }
              }
              cv::Mat internal_sphere_view =
                  ati::BuildInternalSphereDebugView(board_result);
              if (!internal_sphere_view.empty()) {
                const std::string sphere_path = AppendStemSuffix(
                    output_path,
                    "_board" + std::to_string(board_result.board_id) +
                        "_internal_sphere");
                if (!cv::imwrite(sphere_path, internal_sphere_view)) {
                  throw std::runtime_error(
                      "Failed to write internal sphere view: " + sphere_path);
                }
              }
            }
          }
          if (args.rectified_roi_demo) {
            RunRectifiedRoiDemo(
                image, output_path,
                rectified_roi_camera.IsValid() ? &rectified_roi_camera : nullptr,
                cv::Point2f(static_cast<float>(args.rectified_roi_crop_offset_x),
                            static_cast<float>(args.rectified_roi_crop_offset_y)),
                args.rectified_roi_fov_deg, args.rectified_roi_patch_size);
          }
        }

      std::cout << "Detection summary\n";
        std::cout << "  requested boards: [" << JoinBoardIds(multi_result.requested_board_ids)
                  << "]\n";
        std::cout << "  any detected: " << (multi_result.AnyTagDetected() ? "yes" : "no")
                  << "\n";
        std::cout << "  any success: " << (multi_result.AnySuccess() ? "yes" : "no") << "\n";
        std::cout << "  projection mode: " << ati::ToString(config.internal_projection_mode)
                  << "\n";
        std::cout << "  combined overlay: "
                  << (output_path.empty() ? "disabled" : output_path) << "\n";

        for (const ati::ApriltagInternalDetectionResult& board_result :
             multi_result.detections) {
          std::cout << "\nBoard " << board_result.board_id << "\n";
          std::cout << "  tag detected: " << (board_result.tag_detected ? "yes" : "no") << "\n";
          std::cout << "  valid observation: " << (board_result.success ? "yes" : "no")
                    << "\n";
          std::cout << "  outer wrapper success: "
                    << (board_result.outer_detection.success ? "yes" : "no") << "\n";
          std::cout << "  outer failure reason: "
                    << board_result.outer_detection.failure_reason_text << "\n";
          std::cout << "  local patch rescue attempted: "
                    << (board_result.outer_detection.attempted_local_patch_rescue ? "yes" : "no")
                    << "\n";
          if (board_result.outer_detection.attempted_local_patch_rescue) {
            std::cout << "  local patch rescue used: "
                      << (board_result.outer_detection.used_local_patch_rescue ? "yes" : "no")
                      << "\n";
            std::cout << "  local patch rescue summary: "
                      << board_result.outer_detection.local_patch_rescue_summary << "\n";
          }
          std::cout << "  valid corners: " << board_result.valid_corner_count << "\n";
          std::cout << "  valid internal points: "
                    << board_result.valid_internal_corner_count << "\n";
          if (config.enable_debug_output) {
            PrintBorderConditionedDiagnostics(board_result);
          }
          if (config.enable_debug_output) {
            if (!board_result.outer_detection.success) {
              PrintOuterScaleDebug(board_result.outer_detection);
            }
            for (const auto& debug : board_result.outer_detection.corner_verification_debug) {
              if (debug.corner_index < 0) {
                continue;
              }
              std::cout << "    outer corner=" << debug.corner_index
                        << " coarse=(" << std::lround(debug.coarse_corner.x) << ","
                        << std::lround(debug.coarse_corner.y) << ")"
                        << " subpix=(" << std::lround(debug.subpix_corner.x) << ","
                        << std::lround(debug.subpix_corner.y) << ")"
                        << " d_cs=" << std::fixed << std::setprecision(2)
                        << debug.coarse_to_subpix_displacement
                        << " subpix_r=" << debug.subpix_window_radius
                        << " line_ok=" << (debug.line_refinement_success ? "yes" : "no")
                        << " line_q=" << debug.line_refinement_quality
                        << " line_jump=" << debug.line_jump
                        << " line_inside=" << (debug.line_inside ? "yes" : "no")
                        << " line_gap=" << debug.line_seed_gap
                        << " line_seed=" << (debug.line_seed_accepted ? "used" : "rejected")
                        << " image_line=" << (debug.image_line_valid ? "yes" : "no")
                        << " image_line_corner=("
                        << std::lround(debug.image_line_corner.x) << ","
                        << std::lround(debug.image_line_corner.y) << ")"
                        << " image_line_res=(" << debug.prev_image_line_residual
                        << "," << debug.next_image_line_residual << ")"
                        << " image_line_support=("
                        << debug.prev_image_line_support_count << ","
                        << debug.next_image_line_support_count << ")"
                        << "\n";
            }
            std::cout << std::defaultfloat << std::setprecision(6);
          }

        }

        if (show_image) {
          cv::imshow("m-kilbr Apriltag Internal Detection", combined_overlay);
          cv::waitKey(0);
        }

        return multi_result.AnySuccess();
      }

      const ati::ApriltagInternalDetectionResult result = detector.Detect(image);

      std::string output_path;
      cv::Mat overlay;
      if (!args.no_output_images || show_image || args.rectified_roi_demo) {
        EnsureParentDirectoryExists(requested_output_path);
        overlay = args.final_corners_only
                      ? BuildFinalCornersOnlyOverlay(
                            result, image, args.corner_radius)
                      : image.clone();
        if (!args.final_corners_only) {
          detector.DrawDetections(result, &overlay);
        }

        output_path = AppendMinuteStamp(requested_output_path, minute_stamp);
        if (!cv::imwrite(output_path, overlay)) {
          throw std::runtime_error("Failed to write output image: " + output_path);
        }
        if (args.rectified_roi_demo) {
          RunRectifiedRoiDemo(
              image, output_path,
              rectified_roi_camera.IsValid() ? &rectified_roi_camera : nullptr,
              cv::Point2f(static_cast<float>(args.rectified_roi_crop_offset_x),
                          static_cast<float>(args.rectified_roi_crop_offset_y)),
              args.rectified_roi_fov_deg, args.rectified_roi_patch_size);
        }
      }

      std::cout << "Detection summary\n";
      std::cout << "  tag detected: " << (result.tag_detected ? "yes" : "no") << "\n";
      std::cout << "  valid observation: " << (result.success ? "yes" : "no") << "\n";
      std::cout << "  outer wrapper success: "
                << (result.outer_detection.success ? "yes" : "no") << "\n";
      std::cout << "  outer failure reason: " << result.outer_detection.failure_reason_text
                << "\n";
      std::cout << "  local patch rescue attempted: "
                << (result.outer_detection.attempted_local_patch_rescue ? "yes" : "no") << "\n";
      if (result.outer_detection.attempted_local_patch_rescue) {
        std::cout << "  local patch rescue used: "
                  << (result.outer_detection.used_local_patch_rescue ? "yes" : "no") << "\n";
        std::cout << "  local patch rescue summary: "
                  << result.outer_detection.local_patch_rescue_summary << "\n";
      }
      std::cout << "  projection mode: " << ati::ToString(result.projection_mode) << "\n";
      std::cout << "  valid points: " << result.valid_corner_count << "\n";
      std::cout << "  valid internal points: " << result.valid_internal_corner_count << "\n";
      std::cout << "  output image: "
                << (output_path.empty() ? "disabled" : output_path) << "\n";

      const InternalMetricsSummary metrics =
          SummarizeInternalCorners(result.internal_corner_debug);
      std::cout << "  total internal points: " << metrics.total_points << "\n";
      std::cout << "  valid internal points: " << metrics.valid_points << "\n";
      if (config.enable_debug_output) {
        PrintBorderConditionedDiagnostics(result);
      }
      if (config.enable_debug_output) {
        if (!result.outer_detection.success) {
          PrintOuterScaleDebug(result.outer_detection);
        }
        for (const auto& debug : result.outer_detection.corner_verification_debug) {
          if (debug.corner_index < 0) {
            continue;
          }
          std::cout << "  outer corner=" << debug.corner_index
                    << " coarse=(" << std::lround(debug.coarse_corner.x) << ","
                    << std::lround(debug.coarse_corner.y) << ")"
                    << " subpix=(" << std::lround(debug.subpix_corner.x) << ","
                    << std::lround(debug.subpix_corner.y) << ")"
                    << " d_cs=" << std::fixed << std::setprecision(2)
                    << debug.coarse_to_subpix_displacement
                    << " subpix_r=" << debug.subpix_window_radius
                    << " line_ok=" << (debug.line_refinement_success ? "yes" : "no")
                    << " line_q=" << debug.line_refinement_quality
                    << " line_jump=" << debug.line_jump
                    << " line_inside=" << (debug.line_inside ? "yes" : "no")
                    << " line_gap=" << debug.line_seed_gap
                    << " line_seed=" << (debug.line_seed_accepted ? "used" : "rejected")
                    << " image_line=" << (debug.image_line_valid ? "yes" : "no")
                    << " image_line_corner=("
                    << std::lround(debug.image_line_corner.x) << ","
                    << std::lround(debug.image_line_corner.y) << ")"
                    << " image_line_res=(" << debug.prev_image_line_residual
                    << "," << debug.next_image_line_residual << ")"
                    << " image_line_support=("
                    << debug.prev_image_line_support_count << ","
                    << debug.next_image_line_support_count << ")"
                    << "\n";
        }
        std::cout << std::defaultfloat << std::setprecision(6);
      }

      if (show_image) {
        cv::imshow("m-kilbr Apriltag Internal Detection", overlay);
        cv::waitKey(0);
      }

      return result.success;
    };

    const std::vector<std::string> image_paths = CollectImagePaths(args.image_path, args.all);
    if (args.all) {
      int success_count = 0;
      for (const std::string& image_path : image_paths) {
        const std::string requested_output_path =
            BatchRequestedOutputPath(args.output_path, image_path);
        success_count += process_image(image_path, requested_output_path, false) ? 1 : 0;
      }
      std::cout << "\nBatch summary\n";
      std::cout << "  images processed: " << image_paths.size() << "\n";
      std::cout << "  successful detections: " << success_count << "\n";
      std::cout << "  failed detections: "
                << (static_cast<int>(image_paths.size()) - success_count) << "\n";
      return success_count == static_cast<int>(image_paths.size()) ? 0 : 2;
    }

    const std::string requested_output_path =
        args.output_path.empty() ? DefaultOutputPath(args.image_path) : args.output_path;
    return process_image(args.image_path, requested_output_path, args.show) ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "[m-kilbr] " << error.what() << "\n\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
