#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "apriltags/TagFamily.h"
#include "apriltags/Tag36h11.h"
#include "apriltags/TagDetection.h"
#include "apriltags/TagDetector.h"

namespace ati = aslam::cameras::apriltag_internal;

namespace {

struct ProbeSpec {
  std::string label;
  int expected_id = -1;
  cv::Point2f center{};
};

struct CmdArgs {
  std::string config_path;
  std::string image_path;
  std::string output_dir;
  std::vector<ProbeSpec> probes;
};

struct DetectionSummary {
  int id = -1;
  bool good = false;
  int hamming = -1;
  double area = 0.0;
};

struct ProbeAttemptResult {
  ProbeSpec probe;
  int dx = 0;
  int dy = 0;
  double fov_deg = 0.0;
  bool center_valid = false;
  bool any_detection = false;
  bool matched_expected = false;
  std::vector<DetectionSummary> detections;
  cv::Mat patch;
};

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --image IMAGE --config YAML --output-dir DIR"
      << " --probe LABEL:ID:X:Y [--probe LABEL:ID:X:Y ...]\n\n"
      << "Example:\n"
      << "  " << program
      << " --image ./image.png --config ./config.yaml --output-dir ./result/local_patch_probe"
      << " --probe top:3:2256:860 --probe bottom:2:2256:3650\n";
}

ProbeSpec ParseProbe(const std::string& text) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, ':')) {
    parts.push_back(part);
  }
  if (parts.size() != 4) {
    throw std::runtime_error("Probe must be LABEL:ID:X:Y, got '" + text + "'.");
  }

  ProbeSpec probe;
  probe.label = parts[0];
  probe.expected_id = std::stoi(parts[1]);
  probe.center.x = std::stof(parts[2]);
  probe.center.y = std::stof(parts[3]);
  return probe;
}

CmdArgs ParseArgs(int argc, char** argv) {
  CmdArgs args;
  for (int index = 1; index < argc; ++index) {
    const std::string token = argv[index];
    if (token == "--config" && index + 1 < argc) {
      args.config_path = argv[++index];
    } else if (token == "--image" && index + 1 < argc) {
      args.image_path = argv[++index];
    } else if (token == "--output-dir" && index + 1 < argc) {
      args.output_dir = argv[++index];
    } else if (token == "--probe" && index + 1 < argc) {
      args.probes.push_back(ParseProbe(argv[++index]));
    } else if (token == "--help" || token == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + token);
    }
  }

  if (args.config_path.empty() || args.image_path.empty() || args.output_dir.empty() ||
      args.probes.empty()) {
    throw std::runtime_error("--config, --image, --output-dir and at least one --probe are required.");
  }
  return args;
}

cv::Mat ToGray(const cv::Mat& image) {
  if (image.empty()) {
    return cv::Mat();
  }

  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image.clone();
  } else if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    throw std::runtime_error("Unsupported image channel count.");
  }

  if (gray.depth() == CV_16U) {
    gray.convertTo(gray, CV_8U, 1.0 / 256.0);
  } else if (gray.depth() != CV_8U) {
    gray.convertTo(gray, CV_8U);
  }
  return gray;
}

double QuadArea(const std::array<cv::Point2f, 4>& corners) {
  double signed_area_twice = 0.0;
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f& current = corners[static_cast<std::size_t>(index)];
    const cv::Point2f& next = corners[static_cast<std::size_t>((index + 1) % 4)];
    signed_area_twice += static_cast<double>(current.x) * static_cast<double>(next.y) -
                         static_cast<double>(next.x) * static_cast<double>(current.y);
  }
  return 0.5 * std::abs(signed_area_twice);
}

bool Normalize(Eigen::Vector3d* vector) {
  const double norm = vector->norm();
  if (!std::isfinite(norm) || norm <= 1e-12) {
    return false;
  }
  *vector /= norm;
  return true;
}

bool BuildLocalFrame(const ati::DoubleSphereCameraModel& camera,
                     const cv::Point2f& center,
                     Eigen::Vector3d* center_ray,
                     Eigen::Vector3d* tangent_x,
                     Eigen::Vector3d* tangent_y) {
  if (center_ray == nullptr || tangent_x == nullptr || tangent_y == nullptr) {
    throw std::runtime_error("BuildLocalFrame requires valid output pointers.");
  }

  if (!camera.keypointToEuclidean(Eigen::Vector2d(center.x, center.y), center_ray)) {
    return false;
  }
  if (!Normalize(center_ray)) {
    return false;
  }

  Eigen::Vector3d ray_x = Eigen::Vector3d::Zero();
  Eigen::Vector3d ray_y = Eigen::Vector3d::Zero();
  const double delta_px = 24.0;
  if (!camera.keypointToEuclidean(Eigen::Vector2d(center.x + delta_px, center.y), &ray_x) ||
      !camera.keypointToEuclidean(Eigen::Vector2d(center.x, center.y + delta_px), &ray_y)) {
    return false;
  }
  if (!Normalize(&ray_x) || !Normalize(&ray_y)) {
    return false;
  }

  *tangent_x = ray_x - (*center_ray) * center_ray->dot(ray_x);
  if (!Normalize(tangent_x)) {
    return false;
  }
  *tangent_y = ray_y - (*center_ray) * center_ray->dot(ray_y);
  *tangent_y = *tangent_y - (*tangent_x) * tangent_x->dot(*tangent_y);
  if (!Normalize(tangent_y)) {
    return false;
  }
  if (center_ray->dot(tangent_x->cross(*tangent_y)) < 0.0) {
    *tangent_y = -*tangent_y;
  }
  return true;
}

cv::Mat BuildSpherePatch(const cv::Mat& gray,
                         const ati::DoubleSphereCameraModel& camera,
                         const cv::Point2f& center,
                         double fov_deg,
                         int patch_size,
                         bool* center_valid) {
  if (center_valid == nullptr) {
    throw std::runtime_error("BuildSpherePatch requires center_valid output.");
  }
  *center_valid = false;

  Eigen::Vector3d center_ray = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangent_x = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangent_y = Eigen::Vector3d::Zero();
  if (!BuildLocalFrame(camera, center, &center_ray, &tangent_x, &tangent_y)) {
    return cv::Mat();
  }
  *center_valid = true;

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
      if (!Normalize(&ray)) {
        map_x.at<float>(y, x) = -1.0f;
        map_y.at<float>(y, x) = -1.0f;
        continue;
      }

      Eigen::Vector2d keypoint;
      if (!camera.vsEuclideanToKeypoint(ray, &keypoint)) {
        map_x.at<float>(y, x) = -1.0f;
        map_y.at<float>(y, x) = -1.0f;
        continue;
      }

      map_x.at<float>(y, x) = static_cast<float>(keypoint.x());
      map_y.at<float>(y, x) = static_cast<float>(keypoint.y());
    }
  }

  cv::Mat patch;
  cv::remap(gray, patch, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(127));
  return patch;
}

std::vector<DetectionSummary> RunAprilTagDetector(const cv::Mat& patch) {
  AprilTags::TagDetector detector(AprilTags::tagCodes36h11, 2);
  const std::vector<AprilTags::TagDetection> detections = detector.extractTags(patch);
  std::vector<DetectionSummary> summaries;
  summaries.reserve(detections.size());
  for (const auto& detection : detections) {
    DetectionSummary summary;
    summary.id = detection.id;
    summary.good = detection.good;
    summary.hamming = detection.hammingDistance;
    std::array<cv::Point2f, 4> corners{};
    for (int index = 0; index < 4; ++index) {
      corners[static_cast<std::size_t>(index)] =
          cv::Point2f(detection.p[index].first, detection.p[index].second);
    }
    summary.area = QuadArea(corners);
    summaries.push_back(summary);
  }
  return summaries;
}

cv::Mat DrawPatchDetections(const cv::Mat& patch,
                            const std::vector<DetectionSummary>& detections,
                            const std::string& header) {
  cv::Mat color;
  cv::cvtColor(patch, color, cv::COLOR_GRAY2BGR);
  cv::rectangle(color, cv::Rect(0, 0, color.cols, 34), cv::Scalar(20, 20, 20), cv::FILLED);
  cv::putText(color, header, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.50,
              cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
  int y = 54;
  for (const auto& detection : detections) {
    std::ostringstream line;
    line << "id=" << detection.id << " good=" << (detection.good ? "1" : "0")
         << " ham=" << detection.hamming << " area=" << std::lround(detection.area);
    cv::putText(color, line.str(), cv::Point(8, y), cv::FONT_HERSHEY_PLAIN, 1.0,
                cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    y += 16;
  }
  return color;
}

std::string StemForProbe(const ProbeSpec& probe, int dx, int dy, double fov_deg) {
  std::ostringstream stream;
  stream << probe.label << "_id" << probe.expected_id
         << "_dx" << (dx >= 0 ? "+" : "") << dx
         << "_dy" << (dy >= 0 ? "+" : "") << dy
         << "_fov" << std::lround(fov_deg);
  return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CmdArgs args = ParseArgs(argc, argv);
    boost::filesystem::create_directories(args.output_dir);

    const ati::ApriltagInternalConfig config =
        ati::ApriltagInternalDetector::LoadConfig(args.config_path);
    if (!config.intermediate_camera.IsConfigured()) {
      throw std::runtime_error("Config does not contain a valid intermediate/camera DS model.");
    }
    const ati::IntermediateCameraConfig camera_config = config.intermediate_camera;
    const ati::DoubleSphereCameraModel camera =
        ati::DoubleSphereCameraModel::FromConfig(camera_config);

    const cv::Mat image = cv::imread(args.image_path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      throw std::runtime_error("Failed to read image: " + args.image_path);
    }
    const cv::Mat gray = ToGray(image);

    const std::vector<int> offset_values{-240, -120, 0, 120, 240};
    const std::vector<double> fov_values{14.0, 18.0, 22.0, 26.0, 30.0, 36.0, 44.0};
    constexpr int kPatchSize = 720;

    std::cout << "Local sphere patch AprilTag probe\n";
    std::cout << "  image: " << args.image_path << "\n";
    std::cout << "  output_dir: " << args.output_dir << "\n";
    std::cout << "  patch_size: " << kPatchSize << "\n";

    for (const ProbeSpec& probe : args.probes) {
      std::cout << "\nProbe " << probe.label << " expected_id=" << probe.expected_id
                << " center=(" << probe.center.x << "," << probe.center.y << ")\n";
      bool found_expected = false;
      bool has_best_attempt = false;
      ProbeAttemptResult best_attempt;
      std::size_t best_detection_count = 0;

      for (int dy : offset_values) {
        for (int dx : offset_values) {
          const cv::Point2f center = probe.center + cv::Point2f(static_cast<float>(dx),
                                                                static_cast<float>(dy));
          for (double fov_deg : fov_values) {
            ProbeAttemptResult attempt;
            attempt.probe = probe;
            attempt.dx = dx;
            attempt.dy = dy;
            attempt.fov_deg = fov_deg;
            attempt.patch = BuildSpherePatch(gray, camera, center, fov_deg, kPatchSize,
                                             &attempt.center_valid);
            if (!attempt.center_valid || attempt.patch.empty()) {
              continue;
            }

            attempt.detections = RunAprilTagDetector(attempt.patch);
            attempt.any_detection = !attempt.detections.empty();
            attempt.matched_expected = std::any_of(
                attempt.detections.begin(), attempt.detections.end(),
                [&](const DetectionSummary& summary) {
                  return summary.id == probe.expected_id && summary.good;
                });

            if (attempt.matched_expected) {
              found_expected = true;
              const std::string stem = StemForProbe(probe, dx, dy, fov_deg);
              const cv::Mat patch_debug = DrawPatchDetections(
                  attempt.patch, attempt.detections, stem + " HIT");
              cv::imwrite((boost::filesystem::path(args.output_dir) /
                           (stem + "_hit.png")).string(),
                          patch_debug);
              std::cout << "  HIT  dx=" << dx << " dy=" << dy
                        << " fov=" << fov_deg << " deg  detections=";
              for (std::size_t index = 0; index < attempt.detections.size(); ++index) {
                if (index > 0) {
                  std::cout << " | ";
                }
                const auto& detection = attempt.detections[index];
                std::cout << "id=" << detection.id
                          << " good=" << (detection.good ? "1" : "0")
                          << " ham=" << detection.hamming;
              }
              std::cout << "\n";
            }

            if (!has_best_attempt || attempt.detections.size() > best_detection_count) {
              has_best_attempt = true;
              best_detection_count = attempt.detections.size();
              best_attempt = attempt;
            }
          }
        }
      }

      if (!found_expected) {
        std::cout << "  no correct decode found in local sphere-patch sweep.\n";
      }

      if (has_best_attempt && best_attempt.center_valid && !best_attempt.patch.empty()) {
        const std::string stem = probe.label + "_best_probe";
        const cv::Mat patch_debug = DrawPatchDetections(
            best_attempt.patch, best_attempt.detections,
            stem + " dx=" + std::to_string(best_attempt.dx) +
                " dy=" + std::to_string(best_attempt.dy) +
                " fov=" + std::to_string(static_cast<int>(std::lround(best_attempt.fov_deg))));
        cv::imwrite((boost::filesystem::path(args.output_dir) /
                     (stem + ".png")).string(),
                    patch_debug);
        std::cout << "  best probe: dx=" << best_attempt.dx
                  << " dy=" << best_attempt.dy
                  << " fov=" << best_attempt.fov_deg
                  << " deg  detection_count=" << best_attempt.detections.size() << "\n";
        for (const auto& detection : best_attempt.detections) {
          std::cout << "    detected id=" << detection.id
                    << " good=" << (detection.good ? "1" : "0")
                    << " ham=" << detection.hamming
                    << " area=" << std::lround(detection.area) << "\n";
        }
      }
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[run_probe_local_sphere_patch_apriltag] " << error.what() << "\n";
    return 1;
  }
}
