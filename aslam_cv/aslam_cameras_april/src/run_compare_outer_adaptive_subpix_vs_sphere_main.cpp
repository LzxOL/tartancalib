#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

namespace ati = aslam::cameras::apriltag_internal;

struct CmdArgs {
  std::string image_dir;
  std::string config_path;
  std::string output_dir;
};

struct ImageRecord {
  int index = -1;
  boost::filesystem::path path;
  std::string stem;
  cv::Mat image;
};

struct DetectionRecord {
  bool available = false;
  ati::ApriltagInternalDetectionResult result;
  std::string error_text;
};

struct CornerMetric {
  std::string image_stem;
  int board_id = -1;
  int corner_index = -1;
  double d_c = 0.0;
  double d_cs = 0.0;
  double d_csp = 0.0;
  double imp_cs = 0.0;
  double imp_csp = 0.0;
  double delta_csp_vs_cs = 0.0;
};

struct ImageSummary {
  std::string image_stem;
  int valid_corners = 0;
  double avg_d_c = 0.0;
  double avg_d_cs = 0.0;
  double avg_d_csp = 0.0;
  double avg_imp_cs = 0.0;
  double avg_imp_csp = 0.0;
  double avg_delta_csp_vs_cs = 0.0;
};

struct GlobalSummary {
  int image_count = 0;
  int valid_corners = 0;
  double avg_d_c = 0.0;
  double avg_d_cs = 0.0;
  double avg_d_csp = 0.0;
  double avg_imp_cs = 0.0;
  double avg_imp_csp = 0.0;
  double avg_delta_csp_vs_cs = 0.0;
  int csp_better_count = 0;
  int cs_better_count = 0;
  int tie_count = 0;
};

std::string ToLower(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return lowered;
}

bool IsImageFile(const boost::filesystem::path& path) {
  const std::string extension = ToLower(path.extension().string());
  return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
         extension == ".bmp" || extension == ".tif" || extension == ".tiff";
}

std::string BuildTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
  localtime_r(&now_time, &local_time);
  std::ostringstream stream;
  stream << std::put_time(&local_time, "%Y%m%d_%H%M%S");
  return stream.str();
}

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program << " --image-dir DIR --config YAML [--output-dir DIR]\n";
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
    args.output_dir =
        (boost::filesystem::path("result") /
         boost::filesystem::path("outer_adaptive_subpix_vs_sphere_" + BuildTimestamp())).string();
  }
  return args;
}

std::vector<ImageRecord> LoadAllImages(const std::string& image_dir) {
  if (!boost::filesystem::exists(image_dir)) {
    throw std::runtime_error("Image directory does not exist: " + image_dir);
  }
  std::vector<boost::filesystem::path> image_paths;
  for (boost::filesystem::directory_iterator it(image_dir), end; it != end; ++it) {
    if (boost::filesystem::is_regular_file(it->path()) && IsImageFile(it->path())) {
      image_paths.push_back(it->path());
    }
  }
  std::sort(image_paths.begin(), image_paths.end());

  std::vector<ImageRecord> images;
  for (const auto& path : image_paths) {
    ImageRecord record;
    record.index = static_cast<int>(images.size());
    record.path = path;
    record.stem = path.stem().string();
    record.image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (record.image.empty()) {
      continue;
    }
    images.push_back(record);
  }
  if (images.empty()) {
    throw std::runtime_error("No readable images found in: " + image_dir);
  }
  return images;
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

cv::Mat ToGray8(const cv::Mat& image) {
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image.clone();
  } else if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    throw std::runtime_error("Unsupported image channels.");
  }
  if (gray.depth() == CV_16U) {
    gray.convertTo(gray, CV_8U, 1.0 / 256.0);
  } else if (gray.depth() != CV_8U) {
    gray.convertTo(gray, CV_8U);
  }
  return gray;
}

cv::Mat ToBgr8(const cv::Mat& image) {
  cv::Mat bgr;
  if (image.channels() == 1) {
    cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
  } else if (image.channels() == 3) {
    bgr = image.clone();
  } else if (image.channels() == 4) {
    cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
  } else {
    throw std::runtime_error("Unsupported image channels.");
  }
  if (bgr.depth() == CV_16U) {
    bgr.convertTo(bgr, CV_8U, 1.0 / 256.0);
  } else if (bgr.depth() != CV_8U) {
    bgr.convertTo(bgr, CV_8U);
  }
  return bgr;
}

cv::Point2f ToPoint(const Eigen::Vector2d& point) {
  return cv::Point2f(static_cast<float>(point.x()), static_cast<float>(point.y()));
}

double PointDistance(const cv::Point2f& a, const cv::Point2f& b) {
  return std::hypot(static_cast<double>(a.x - b.x), static_cast<double>(a.y - b.y));
}

DetectionRecord RunDetection(const ati::ApriltagInternalDetector& detector, const cv::Mat& image) {
  DetectionRecord record;
  try {
    record.result = detector.Detect(image);
    record.available = true;
  } catch (const std::exception& error) {
    record.available = false;
    record.error_text = error.what();
  }
  return record;
}

std::vector<DetectionRecord> RunDetectionPass(const std::vector<ImageRecord>& images,
                                              const ati::ApriltagInternalConfig& config,
                                              const ati::ApriltagInternalDetectionOptions& options) {
  ati::ApriltagInternalDetector detector(config, options);
  std::vector<DetectionRecord> detections;
  detections.reserve(images.size());
  for (const auto& image : images) {
    detections.push_back(RunDetection(detector, image.image));
  }
  return detections;
}

cv::Point2f ComputeOuterGtCorner(const ati::ApriltagInternalDetectionResult& result, int corner_index) {
  const ati::OuterCornerVerificationDebugInfo& debug =
      result.outer_detection.corner_verification_debug[static_cast<std::size_t>(corner_index)];
  if (debug.subpix_applied) {
    return debug.subpix_corner;
  }
  if (debug.spherical_refinement_valid) {
    return debug.spherical_corner;
  }
  return ToPoint(result.outer_detection.refined_corners_original_image[static_cast<std::size_t>(corner_index)]);
}

bool ComputeCoarseAdaptiveSubpixCorner(const cv::Mat& gray,
                                       const ati::OuterCornerVerificationDebugInfo& debug,
                                       cv::Point2f* corner) {
  if (corner == nullptr) {
    throw std::runtime_error("ComputeCoarseAdaptiveSubpixCorner requires a valid output pointer.");
  }
  if (debug.subpix_window_radius < 2) {
    return false;
  }
  const float border = static_cast<float>(debug.subpix_window_radius + 2);
  if (debug.coarse_corner.x < border || debug.coarse_corner.y < border ||
      debug.coarse_corner.x >= static_cast<float>(gray.cols) - border ||
      debug.coarse_corner.y >= static_cast<float>(gray.rows) - border) {
    return false;
  }

  std::vector<cv::Point2f> seeds(1, debug.coarse_corner);
  cv::cornerSubPix(gray, seeds,
                   cv::Size(debug.subpix_window_radius, debug.subpix_window_radius),
                   cv::Size(-1, -1),
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
  *corner = seeds.front();
  return std::isfinite(corner->x) && std::isfinite(corner->y);
}

std::vector<CornerMetric> CollectMetrics(const std::vector<ImageRecord>& images,
                                         const std::vector<DetectionRecord>& detections) {
  std::vector<CornerMetric> rows;
  for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
    if (image_index >= detections.size() || !detections[image_index].available) {
      continue;
    }
    const ati::ApriltagInternalDetectionResult& result = detections[image_index].result;
    if (!result.tag_detected) {
      continue;
    }
    const cv::Mat gray = ToGray8(images[image_index].image);

    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      const ati::OuterCornerVerificationDebugInfo& debug =
          result.outer_detection.corner_verification_debug[static_cast<std::size_t>(corner_index)];
      if (!result.outer_detection.refined_valid[static_cast<std::size_t>(corner_index)] ||
          !debug.spherical_refinement_valid) {
        continue;
      }

      cv::Point2f coarse_subpix;
      if (!ComputeCoarseAdaptiveSubpixCorner(gray, debug, &coarse_subpix)) {
        continue;
      }

      const cv::Point2f coarse = debug.coarse_corner;
      const cv::Point2f csp = debug.spherical_corner;
      const cv::Point2f gt = ComputeOuterGtCorner(result, corner_index);

      CornerMetric row;
      row.image_stem = images[image_index].stem;
      row.board_id = result.board_id;
      row.corner_index = corner_index;
      row.d_c = PointDistance(coarse, gt);
      row.d_cs = PointDistance(coarse_subpix, gt);
      row.d_csp = PointDistance(csp, gt);
      row.imp_cs = row.d_c - row.d_cs;
      row.imp_csp = row.d_c - row.d_csp;
      row.delta_csp_vs_cs = row.d_cs - row.d_csp;
      rows.push_back(row);
    }
  }
  return rows;
}

std::vector<ImageSummary> SummarizePerImage(const std::vector<CornerMetric>& rows) {
  std::map<std::string, std::vector<const CornerMetric*> > buckets;
  for (const auto& row : rows) {
    buckets[row.image_stem].push_back(&row);
  }
  std::vector<ImageSummary> summaries;
  for (const auto& bucket : buckets) {
    ImageSummary summary;
    summary.image_stem = bucket.first;
    summary.valid_corners = static_cast<int>(bucket.second.size());
    const double count = static_cast<double>(bucket.second.size());
    for (const auto* row : bucket.second) {
      summary.avg_d_c += row->d_c;
      summary.avg_d_cs += row->d_cs;
      summary.avg_d_csp += row->d_csp;
      summary.avg_imp_cs += row->imp_cs;
      summary.avg_imp_csp += row->imp_csp;
      summary.avg_delta_csp_vs_cs += row->delta_csp_vs_cs;
    }
    summary.avg_d_c /= count;
    summary.avg_d_cs /= count;
    summary.avg_d_csp /= count;
    summary.avg_imp_cs /= count;
    summary.avg_imp_csp /= count;
    summary.avg_delta_csp_vs_cs /= count;
    summaries.push_back(summary);
  }
  std::sort(summaries.begin(), summaries.end(),
            [](const ImageSummary& lhs, const ImageSummary& rhs) {
              return lhs.image_stem < rhs.image_stem;
            });
  return summaries;
}

GlobalSummary SummarizeGlobal(const std::vector<ImageSummary>& image_summaries,
                              const std::vector<CornerMetric>& rows) {
  GlobalSummary summary;
  summary.image_count = static_cast<int>(image_summaries.size());
  summary.valid_corners = static_cast<int>(rows.size());
  if (rows.empty()) {
    return summary;
  }
  const double count = static_cast<double>(rows.size());
  for (const auto& row : rows) {
    summary.avg_d_c += row.d_c;
    summary.avg_d_cs += row.d_cs;
    summary.avg_d_csp += row.d_csp;
    summary.avg_imp_cs += row.imp_cs;
    summary.avg_imp_csp += row.imp_csp;
    summary.avg_delta_csp_vs_cs += row.delta_csp_vs_cs;
    if (row.d_csp + 1e-6 < row.d_cs) {
      ++summary.csp_better_count;
    } else if (row.d_cs + 1e-6 < row.d_csp) {
      ++summary.cs_better_count;
    } else {
      ++summary.tie_count;
    }
  }
  summary.avg_d_c /= count;
  summary.avg_d_cs /= count;
  summary.avg_d_csp /= count;
  summary.avg_imp_cs /= count;
  summary.avg_imp_csp /= count;
  summary.avg_delta_csp_vs_cs /= count;
  return summary;
}

std::string FormatDouble(double value, int precision = 4) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

cv::Rect ComputeInsetRect(const ati::OuterCornerVerificationDebugInfo& debug,
                          const cv::Size& image_size) {
  const int radius = std::max(42, debug.verification_roi_radius + 28);
  const cv::Point2f center = debug.spherical_refinement_valid ? debug.spherical_corner : debug.coarse_corner;
  const int x0 = std::max(0, static_cast<int>(std::floor(center.x)) - radius);
  const int y0 = std::max(0, static_cast<int>(std::floor(center.y)) - radius);
  const int x1 = std::min(image_size.width, static_cast<int>(std::ceil(center.x)) + radius + 1);
  const int y1 = std::min(image_size.height, static_cast<int>(std::ceil(center.y)) + radius + 1);
  return cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
}

void DrawLabel(cv::Mat* image,
               const cv::Point2f& point,
               const std::string& text,
               const cv::Scalar& color,
               const cv::Point2f& offset) {
  if (image == nullptr) {
    return;
  }
  cv::putText(*image, text, point + offset, cv::FONT_HERSHEY_PLAIN, 1.0, color, 1, cv::LINE_AA);
}

cv::Mat BuildGlobalOverlay(const ImageRecord& image_record,
                           const ati::ApriltagInternalDetectionResult& result) {
  cv::Mat canvas = ToBgr8(image_record.image);
  const cv::Mat gray = ToGray8(image_record.image);
  const cv::Scalar kCoarseColor(0, 165, 255);
  const cv::Scalar kCsColor(255, 120, 0);
  const cv::Scalar kCspColor(255, 80, 255);
  const cv::Scalar kGtColor(0, 200, 80);

  cv::putText(canvas, image_record.stem + " outer comparison: C-S(adaptive) vs C-SP",
              cv::Point(26, 42), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(20, 20, 20), 2,
              cv::LINE_AA);
  cv::putText(canvas, "C orange, C-S blue, C-SP magenta, GT(SP->subpix) green",
              cv::Point(26, 72), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(70, 70, 70), 1,
              cv::LINE_AA);

  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    const ati::OuterCornerVerificationDebugInfo& debug =
        result.outer_detection.corner_verification_debug[static_cast<std::size_t>(corner_index)];
    if (!result.outer_detection.refined_valid[static_cast<std::size_t>(corner_index)] ||
        !debug.spherical_refinement_valid) {
      continue;
    }

    cv::Point2f coarse_subpix;
    if (!ComputeCoarseAdaptiveSubpixCorner(gray, debug, &coarse_subpix)) {
      continue;
    }
    const cv::Point2f gt = ComputeOuterGtCorner(result, corner_index);

    cv::line(canvas, debug.coarse_corner, coarse_subpix, kCsColor, 1, cv::LINE_AA);
    cv::line(canvas, debug.coarse_corner, debug.spherical_corner, kCspColor, 1, cv::LINE_AA);

    cv::circle(canvas, debug.coarse_corner, 4, kCoarseColor, 2, cv::LINE_AA);
    cv::drawMarker(canvas, coarse_subpix, kCsColor, cv::MARKER_CROSS, 10, 1, cv::LINE_AA);
    cv::drawMarker(canvas, debug.spherical_corner, kCspColor, cv::MARKER_DIAMOND, 10, 1, cv::LINE_AA);
    cv::rectangle(canvas,
                  cv::Rect(static_cast<int>(std::lround(gt.x)) - 3,
                           static_cast<int>(std::lround(gt.y)) - 3, 7, 7),
                  kGtColor, 1, cv::LINE_AA);

    DrawLabel(&canvas, debug.coarse_corner, "C" + std::to_string(corner_index), kCoarseColor,
              cv::Point2f(6.0f, -6.0f));
    DrawLabel(&canvas, coarse_subpix, "CS", kCsColor, cv::Point2f(6.0f, -6.0f));
    DrawLabel(&canvas, debug.spherical_corner, "CSP", kCspColor, cv::Point2f(6.0f, -6.0f));
    DrawLabel(&canvas, gt, "GT", kGtColor, cv::Point2f(6.0f, 12.0f));
  }

  return canvas;
}

cv::Mat BuildInsetMosaic(const ImageRecord& image_record,
                         const ati::ApriltagInternalDetectionResult& result) {
  const cv::Mat source = ToBgr8(image_record.image);
  const cv::Mat gray = ToGray8(image_record.image);
  constexpr int kCanvasWidth = 1600;
  constexpr int kCanvasHeight = 1200;
  constexpr int kMargin = 60;
  constexpr int kHeaderHeight = 90;
  const int panel_width = (kCanvasWidth - 3 * kMargin) / 2;
  const int panel_height = (kCanvasHeight - kHeaderHeight - 3 * kMargin) / 2;

  cv::Mat canvas(kCanvasHeight, kCanvasWidth, CV_8UC3, cv::Scalar(248, 248, 248));
  cv::putText(canvas, image_record.stem + " local corner comparison",
              cv::Point(30, 42), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(20, 20, 20), 2,
              cv::LINE_AA);
  cv::putText(canvas, "Each panel compares C-S(adaptive) and C-SP against GT = SP->subpix",
              cv::Point(30, 72), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(70, 70, 70), 1,
              cv::LINE_AA);

  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    const ati::OuterCornerVerificationDebugInfo& debug =
        result.outer_detection.corner_verification_debug[static_cast<std::size_t>(corner_index)];
    if (!result.outer_detection.refined_valid[static_cast<std::size_t>(corner_index)] ||
        !debug.spherical_refinement_valid) {
      continue;
    }

    cv::Point2f coarse_subpix;
    if (!ComputeCoarseAdaptiveSubpixCorner(gray, debug, &coarse_subpix)) {
      continue;
    }

    const cv::Rect roi = ComputeInsetRect(debug, source.size());
    cv::Mat crop = source(roi).clone();
    const auto to_local = [&](const cv::Point2f& point) {
      return point - cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
    };

    const cv::Point2f coarse = to_local(debug.coarse_corner);
    const cv::Point2f cs = to_local(coarse_subpix);
    const cv::Point2f csp = to_local(debug.spherical_corner);
    const cv::Point2f gt = to_local(ComputeOuterGtCorner(result, corner_index));

    cv::line(crop, coarse, cs, cv::Scalar(255, 120, 0), 1, cv::LINE_AA);
    cv::line(crop, coarse, csp, cv::Scalar(255, 80, 255), 1, cv::LINE_AA);
    cv::circle(crop, coarse, 4, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
    cv::drawMarker(crop, cs, cv::Scalar(255, 120, 0), cv::MARKER_CROSS, 10, 1, cv::LINE_AA);
    cv::drawMarker(crop, csp, cv::Scalar(255, 80, 255), cv::MARKER_DIAMOND, 10, 1, cv::LINE_AA);
    cv::rectangle(crop,
                  cv::Rect(static_cast<int>(std::lround(gt.x)) - 3,
                           static_cast<int>(std::lround(gt.y)) - 3, 7, 7),
                  cv::Scalar(0, 200, 80), 1, cv::LINE_AA);

    const double d_cs = PointDistance(coarse_subpix, ComputeOuterGtCorner(result, corner_index));
    const double d_csp = PointDistance(debug.spherical_corner, ComputeOuterGtCorner(result, corner_index));
    cv::putText(crop, "corner " + std::to_string(corner_index),
                cv::Point(10, 18), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(20, 20, 20), 1,
                cv::LINE_AA);
    cv::putText(crop, "CS=" + FormatDouble(d_cs, 2) + " px",
                cv::Point(10, 36), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 120, 0), 1,
                cv::LINE_AA);
    cv::putText(crop, "CSP=" + FormatDouble(d_csp, 2) + " px",
                cv::Point(10, 54), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 80, 255), 1,
                cv::LINE_AA);

    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(panel_width, panel_height));
    const int row = corner_index / 2;
    const int col = corner_index % 2;
    const cv::Rect dst(kMargin + col * (panel_width + kMargin),
                       kHeaderHeight + kMargin + row * (panel_height + kMargin),
                       panel_width, panel_height);
    resized.copyTo(canvas(dst));
    cv::rectangle(canvas, dst, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
  }

  return canvas;
}

void SaveVisualizations(const boost::filesystem::path& visuals_dir,
                        const std::vector<ImageRecord>& images,
                        const std::vector<DetectionRecord>& detections) {
  boost::filesystem::create_directories(visuals_dir);
  for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
    if (image_index >= detections.size() || !detections[image_index].available ||
        !detections[image_index].result.tag_detected) {
      continue;
    }
    const cv::Mat overlay = BuildGlobalOverlay(images[image_index], detections[image_index].result);
    const cv::Mat local = BuildInsetMosaic(images[image_index], detections[image_index].result);
    if (!overlay.empty()) {
      cv::imwrite((visuals_dir / (images[image_index].stem + "_overlay.png")).string(), overlay);
    }
    if (!local.empty()) {
      cv::imwrite((visuals_dir / (images[image_index].stem + "_local.png")).string(), local);
    }
  }
}

void WriteSummary(const boost::filesystem::path& summary_path,
                  const GlobalSummary& global_summary,
                  const std::vector<ImageSummary>& image_summaries) {
  std::ofstream stream(summary_path.string().c_str());
  stream << "# Outer C-S(adaptive) vs C-SP Summary\n\n";
  stream << "GT definition: `SP -> image-space subpixel`\n\n";
  stream << "## Global\n\n";
  stream << "- images: " << global_summary.image_count << "\n";
  stream << "- valid corners: " << global_summary.valid_corners << "\n";
  stream << "- avg `|C-GT|` = " << FormatDouble(global_summary.avg_d_c, 4) << "\n";
  stream << "- avg `|C-S(adaptive)-GT|` = " << FormatDouble(global_summary.avg_d_cs, 4) << "\n";
  stream << "- avg `|C-SP-GT|` = " << FormatDouble(global_summary.avg_d_csp, 4) << "\n";
  stream << "- avg `imp_CS = dC-dCS` = " << FormatDouble(global_summary.avg_imp_cs, 4) << "\n";
  stream << "- avg `imp_CSP = dC-dCSP` = " << FormatDouble(global_summary.avg_imp_csp, 4) << "\n";
  stream << "- avg `delta_CSP_vs_CS = dCS-dCSP` = "
         << FormatDouble(global_summary.avg_delta_csp_vs_cs, 4)
         << "  (positive means C-SP is better)\n";
  stream << "- corner wins: `C-SP better = " << global_summary.csp_better_count
         << "`, `C-S better = " << global_summary.cs_better_count
         << "`, `tie = " << global_summary.tie_count << "`\n\n";

  stream << "## Per-image\n\n";
  stream << "| image | valid corners | avg |C-GT| | avg |CS-GT| | avg |CSP-GT| | avg delta_CSP_vs_CS |\n";
  stream << "| --- | ---: | ---: | ---: | ---: | ---: |\n";
  for (const auto& summary : image_summaries) {
    stream << "| " << summary.image_stem
           << " | " << summary.valid_corners
           << " | " << FormatDouble(summary.avg_d_c, 4)
           << " | " << FormatDouble(summary.avg_d_cs, 4)
           << " | " << FormatDouble(summary.avg_d_csp, 4)
           << " | " << FormatDouble(summary.avg_delta_csp_vs_cs, 4) << " |\n";
  }
  stream << "\n## Conclusion\n\n";
  if (global_summary.avg_delta_csp_vs_cs > 0.0) {
    stream << "- Overall, `C-SP` is closer to `GT = SP->subpix` than the old `C-S(adaptive)` baseline.\n";
  } else if (global_summary.avg_delta_csp_vs_cs < 0.0) {
    stream << "- Overall, `C-S(adaptive)` is closer to `GT = SP->subpix` than `C-SP` on this image set.\n";
  } else {
    stream << "- Overall, the two methods are nearly tied on this image set.\n";
  }
  stream << "- The visual overlays are stored under `visuals/`, with one full-image comparison and one local inset mosaic for each image.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CmdArgs args = ParseArgs(argc, argv);
    ati::ApriltagInternalConfig config =
        ati::ApriltagInternalDetector::LoadConfig(args.config_path);
    config.tag_ids.clear();
    config.outer_detector_config.tag_id = config.tag_id;
    ati::ApriltagInternalDetectionOptions options =
        MakeDetectionOptionsFromConfig(config);

    config.internal_projection_mode = ati::InternalProjectionMode::Homography;
    config.outer_spherical_use_initial_camera = false;
    config.outer_detector_config.do_outer_subpix_refinement = true;
    options.do_subpix_refinement = true;

    const std::vector<ImageRecord> images = LoadAllImages(args.image_dir);
    const std::vector<DetectionRecord> detections = RunDetectionPass(images, config, options);
    const std::vector<CornerMetric> metrics = CollectMetrics(images, detections);
    const std::vector<ImageSummary> image_summaries = SummarizePerImage(metrics);
    const GlobalSummary global_summary = SummarizeGlobal(image_summaries, metrics);

    const boost::filesystem::path output_dir(args.output_dir);
    const boost::filesystem::path visuals_dir = output_dir / "visuals";
    boost::filesystem::create_directories(output_dir);
    boost::filesystem::create_directories(visuals_dir);

    SaveVisualizations(visuals_dir, images, detections);
    WriteSummary(output_dir / "summary.md", global_summary, image_summaries);

    std::cout << "[compare-outer-adaptive-subpix-vs-sphere] finished\n";
    std::cout << "  output_dir: " << output_dir.string() << "\n";
    std::cout << "  images: " << global_summary.image_count << "\n";
    std::cout << "  valid corners: " << global_summary.valid_corners << "\n";
    std::cout << "  avg |CS-GT|: " << FormatDouble(global_summary.avg_d_cs, 4) << "\n";
    std::cout << "  avg |CSP-GT|: " << FormatDouble(global_summary.avg_d_csp, 4) << "\n";
    std::cout << "  avg delta_CSP_vs_CS: " << FormatDouble(global_summary.avg_delta_csp_vs_cs, 4) << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[compare-outer-adaptive-subpix-vs-sphere] " << error.what() << "\n\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
