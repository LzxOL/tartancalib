#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
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

std::string BuildOuterChainLabel(const ati::OuterCornerVerificationDebugInfo& debug) {
  if (debug.spherical_refinement_valid) {
    return debug.subpix_applied ? "C-S-SP" : "C-SP";
  }
  if (debug.subpix_applied) {
    return "C-S";
  }
  return "C";
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

    std::string sphere_output_path;
    cv::Mat sphere_overlay = BuildOuterSphereDebugView(config, result);
    if (!sphere_overlay.empty()) {
      sphere_output_path =
          AppendMinuteStamp(SphereOutputPathForRequestedOutput(requested_output_path), minute_stamp);
      if (!cv::imwrite(sphere_output_path, sphere_overlay)) {
        throw std::runtime_error("Failed to write sphere output image: " + sphere_output_path);
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
    const bool outer_has_subpix = config.outer_detector_config.do_outer_subpix_refinement;
    const bool outer_has_sphere = config.intermediate_camera.IsConfigured();
    std::cout << "  chain: "
              << (outer_has_sphere ? (outer_has_subpix ? "C-S-SP" : "C-SP")
                                   : (outer_has_subpix ? "C-S" : "C"))
              << "\n";
    std::cout << "  outer_local_context_scale: "
              << config.outer_detector_config.outer_local_context_scale << "\n";
    std::cout << "  outer_corner_marker_ratio: "
              << config.outer_detector_config.outer_corner_marker_ratio << "\n";
    std::cout << "  outer_subpix_scale: "
              << config.outer_detector_config.outer_subpix_scale << "\n";
    std::cout << "  outer_spherical_refinement: "
              << (outer_has_sphere ? "on" : "off") << "\n";
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
    if (!sphere_output_path.empty()) {
      std::cout << "  sphere view image: " << sphere_output_path << "\n";
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
                  << " chain="
                  << BuildOuterChainLabel(debug)
                  << " refined_valid=" << (debug.refined_valid ? "yes" : "no")
                  << " local_scale=" << debug.local_scale
                  << " marker_width=" << debug.corner_marker_width
                  << " context_radius=" << debug.verification_roi_radius
                  << " coarse=(" << debug.coarse_corner.x << ", " << debug.coarse_corner.y << ")"
                  << " subpix=(" << debug.subpix_corner.x << ", " << debug.subpix_corner.y << ")"
                  << " sphere=(" << debug.spherical_corner.x << ", " << debug.spherical_corner.y << ")"
                  << " d_cs=" << debug.coarse_to_subpix_displacement
                  << " d_cr=" << debug.coarse_to_refined_displacement
                  << " sphere_support=" << debug.prev_spherical_support_count
                  << "/" << debug.next_spherical_support_count
                  << " sphere_rms=" << debug.prev_spherical_residual
                  << "/" << debug.next_spherical_residual
                  << " subpix_radius=" << debug.subpix_window_radius
                  << " refine_gate=" << debug.refine_displacement_limit;
        if (!debug.spherical_failure_reason.empty()) {
          std::cout << " sphere_reason=" << debug.spherical_failure_reason;
        }
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
