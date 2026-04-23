#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
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

struct PerCornerGeometryMetric {
  std::string image_stem;
  int board_id = -1;
  int corner_index = -1;
  double d_c = 0.0;
  double d_il = 0.0;
  double d_sp = 0.0;
  double d_final = 0.0;
  double imp_il = 0.0;
  double imp_sp = 0.0;
  double delta_sphere_vs_line = 0.0;
  double prev_line_rms = 0.0;
  double next_line_rms = 0.0;
  double prev_plane_rms = 0.0;
  double next_plane_rms = 0.0;
  int support_count_prev = 0;
  int support_count_next = 0;
};

struct PerImageGeometrySummary {
  std::string image_stem;
  int corner_count = 0;
  double avg_d_c = 0.0;
  double avg_d_il = 0.0;
  double avg_d_sp = 0.0;
  double avg_d_final = 0.0;
  double avg_imp_il = 0.0;
  double avg_imp_sp = 0.0;
  double avg_delta_sphere_vs_line = 0.0;
  double avg_line_rms = 0.0;
  double avg_plane_rms = 0.0;
};

struct GlobalGeometrySummary {
  int total_corners = 0;
  double avg_d_c = 0.0;
  double avg_d_il = 0.0;
  double avg_d_sp = 0.0;
  double avg_d_final = 0.0;
  double avg_imp_il = 0.0;
  double avg_imp_sp = 0.0;
  double avg_delta_sphere_vs_line = 0.0;
  double avg_line_rms = 0.0;
  double avg_plane_rms = 0.0;
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
      << "  " << program << " --image-dir DIR --config YAML [--output-dir DIR]\n\n"
      << "This runner uses all readable images under --image-dir.\n";
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
         boost::filesystem::path("outer_geometry_compare_" + BuildTimestamp())).string();
  }
  return args;
}

std::vector<ImageRecord> LoadSelectedImages(const std::string& image_dir) {
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
      throw std::runtime_error("Failed to read image: " + path.string());
    }
    images.push_back(record);
  }

  if (images.empty()) {
    throw std::runtime_error("No readable images were found under " + image_dir);
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

cv::Point2f ToPoint(const Eigen::Vector2d& point) {
  return cv::Point2f(static_cast<float>(point.x()), static_cast<float>(point.y()));
}

double PointDistance(const cv::Point2f& a, const cv::Point2f& b) {
  return std::hypot(static_cast<double>(a.x - b.x), static_cast<double>(a.y - b.y));
}

cv::Mat ToBgr(const cv::Mat& image) {
  cv::Mat bgr;
  if (image.channels() == 1) {
    cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
  } else if (image.channels() == 3) {
    bgr = image.clone();
  } else if (image.channels() == 4) {
    cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
  } else {
    throw std::runtime_error("Unsupported image channels for visualization.");
  }
  if (bgr.depth() == CV_16U) {
    bgr.convertTo(bgr, CV_8U, 1.0 / 256.0);
  } else if (bgr.depth() != CV_8U) {
    bgr.convertTo(bgr, CV_8U);
  }
  return bgr;
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

cv::Point2f ComputeOuterSubpixGtCorner(const ati::ApriltagInternalDetectionResult& result,
                                       int corner_index) {
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

bool IsValidCornerForComparison(const ati::ApriltagInternalDetectionResult& result,
                                const ati::OuterCornerVerificationDebugInfo& debug,
                                int corner_index) {
  return result.tag_detected &&
         result.outer_detection.refined_valid[static_cast<std::size_t>(corner_index)] &&
         debug.image_line_valid &&
         debug.spherical_refinement_valid &&
         debug.prev_image_line_support_count > 1 &&
         debug.next_image_line_support_count > 1 &&
         debug.prev_spherical_support_count > 1 &&
         debug.next_spherical_support_count > 1;
}

std::vector<PerCornerGeometryMetric> CollectPerCornerMetrics(
    const std::vector<ImageRecord>& images,
    const std::vector<DetectionRecord>& detections) {
  std::vector<PerCornerGeometryMetric> rows;
  for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
    if (image_index >= detections.size() || !detections[image_index].available) {
      continue;
    }
    const ati::ApriltagInternalDetectionResult& result = detections[image_index].result;
    if (!result.tag_detected) {
      continue;
    }

    for (int corner_index = 0; corner_index < 4; ++corner_index) {
      const ati::OuterCornerVerificationDebugInfo& debug =
          result.outer_detection.corner_verification_debug[static_cast<std::size_t>(corner_index)];
      if (!IsValidCornerForComparison(result, debug, corner_index)) {
        continue;
      }

      const cv::Point2f coarse = debug.coarse_corner;
      const cv::Point2f image_line = debug.image_line_corner;
      const cv::Point2f spherical = debug.spherical_corner;
      const cv::Point2f final_corner =
          ToPoint(result.outer_detection.refined_corners_original_image[static_cast<std::size_t>(corner_index)]);
      const cv::Point2f outer_gt = ComputeOuterSubpixGtCorner(result, corner_index);

      PerCornerGeometryMetric row;
      row.image_stem = images[image_index].stem;
      row.board_id = result.board_id;
      row.corner_index = corner_index;
      row.d_c = PointDistance(coarse, outer_gt);
      row.d_il = PointDistance(image_line, outer_gt);
      row.d_sp = PointDistance(spherical, outer_gt);
      row.d_final = PointDistance(final_corner, outer_gt);
      row.imp_il = row.d_c - row.d_il;
      row.imp_sp = row.d_c - row.d_sp;
      row.delta_sphere_vs_line = row.d_il - row.d_sp;
      row.prev_line_rms = debug.prev_image_line_residual;
      row.next_line_rms = debug.next_image_line_residual;
      row.prev_plane_rms = debug.prev_spherical_residual;
      row.next_plane_rms = debug.next_spherical_residual;
      row.support_count_prev = debug.prev_spherical_support_count;
      row.support_count_next = debug.next_spherical_support_count;
      rows.push_back(row);
    }
  }
  return rows;
}

std::vector<PerImageGeometrySummary> SummarizePerImage(
    const std::vector<PerCornerGeometryMetric>& rows) {
  std::map<std::string, std::vector<const PerCornerGeometryMetric*> > buckets;
  for (const auto& row : rows) {
    buckets[row.image_stem].push_back(&row);
  }

  std::vector<PerImageGeometrySummary> summaries;
  for (const auto& bucket : buckets) {
    PerImageGeometrySummary summary;
    summary.image_stem = bucket.first;
    summary.corner_count = static_cast<int>(bucket.second.size());
    const double count = static_cast<double>(bucket.second.size());
    for (const auto* row : bucket.second) {
      summary.avg_d_c += row->d_c;
      summary.avg_d_il += row->d_il;
      summary.avg_d_sp += row->d_sp;
      summary.avg_d_final += row->d_final;
      summary.avg_imp_il += row->imp_il;
      summary.avg_imp_sp += row->imp_sp;
      summary.avg_delta_sphere_vs_line += row->delta_sphere_vs_line;
      summary.avg_line_rms += 0.5 * (row->prev_line_rms + row->next_line_rms);
      summary.avg_plane_rms += 0.5 * (row->prev_plane_rms + row->next_plane_rms);
    }
    summary.avg_d_c /= count;
    summary.avg_d_il /= count;
    summary.avg_d_sp /= count;
    summary.avg_d_final /= count;
    summary.avg_imp_il /= count;
    summary.avg_imp_sp /= count;
    summary.avg_delta_sphere_vs_line /= count;
    summary.avg_line_rms /= count;
    summary.avg_plane_rms /= count;
    summaries.push_back(summary);
  }

  std::sort(summaries.begin(), summaries.end(),
            [](const PerImageGeometrySummary& lhs, const PerImageGeometrySummary& rhs) {
              return lhs.image_stem < rhs.image_stem;
            });
  return summaries;
}

GlobalGeometrySummary SummarizeGlobal(const std::vector<PerCornerGeometryMetric>& rows) {
  GlobalGeometrySummary summary;
  summary.total_corners = static_cast<int>(rows.size());
  if (rows.empty()) {
    return summary;
  }
  const double count = static_cast<double>(rows.size());
  for (const auto& row : rows) {
    summary.avg_d_c += row.d_c;
    summary.avg_d_il += row.d_il;
    summary.avg_d_sp += row.d_sp;
    summary.avg_d_final += row.d_final;
    summary.avg_imp_il += row.imp_il;
    summary.avg_imp_sp += row.imp_sp;
    summary.avg_delta_sphere_vs_line += row.delta_sphere_vs_line;
    summary.avg_line_rms += 0.5 * (row.prev_line_rms + row.next_line_rms);
    summary.avg_plane_rms += 0.5 * (row.prev_plane_rms + row.next_plane_rms);
  }
  summary.avg_d_c /= count;
  summary.avg_d_il /= count;
  summary.avg_d_sp /= count;
  summary.avg_d_final /= count;
  summary.avg_imp_il /= count;
  summary.avg_imp_sp /= count;
  summary.avg_delta_sphere_vs_line /= count;
  summary.avg_line_rms /= count;
  summary.avg_plane_rms /= count;
  return summary;
}

std::string FormatDouble(double value, int precision = 4) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

void WritePerCornerCsv(const boost::filesystem::path& csv_path,
                       const std::vector<PerCornerGeometryMetric>& rows) {
  std::ofstream stream(csv_path.string().c_str());
  stream << "image_stem,board_id,corner_index,d_C,d_IL,d_SP,d_final,imp_IL,imp_SP,"
         << "delta_sphere_vs_line,prev_line_rms,next_line_rms,prev_plane_rms,next_plane_rms,"
         << "support_count_prev,support_count_next\n";
  for (const auto& row : rows) {
    stream << row.image_stem << ","
           << row.board_id << ","
           << row.corner_index << ","
           << row.d_c << ","
           << row.d_il << ","
           << row.d_sp << ","
           << row.d_final << ","
           << row.imp_il << ","
           << row.imp_sp << ","
           << row.delta_sphere_vs_line << ","
           << row.prev_line_rms << ","
           << row.next_line_rms << ","
           << row.prev_plane_rms << ","
           << row.next_plane_rms << ","
           << row.support_count_prev << ","
           << row.support_count_next << "\n";
  }
}

void WritePerImageCsv(const boost::filesystem::path& csv_path,
                      const std::vector<PerImageGeometrySummary>& rows) {
  std::ofstream stream(csv_path.string().c_str());
  stream << "image_stem,corner_count,avg_d_C,avg_d_IL,avg_d_SP,avg_d_final,avg_imp_IL,avg_imp_SP,"
         << "avg_delta_sphere_vs_line,avg_line_rms,avg_plane_rms\n";
  for (const auto& row : rows) {
    stream << row.image_stem << ","
           << row.corner_count << ","
           << row.avg_d_c << ","
           << row.avg_d_il << ","
           << row.avg_d_sp << ","
           << row.avg_d_final << ","
           << row.avg_imp_il << ","
           << row.avg_imp_sp << ","
           << row.avg_delta_sphere_vs_line << ","
           << row.avg_line_rms << ","
           << row.avg_plane_rms << "\n";
  }
}

cv::Mat MakeChartCanvas(const std::string& title, const std::string& subtitle) {
  cv::Mat canvas(620, 980, CV_8UC3, cv::Scalar(248, 248, 248));
  cv::putText(canvas, title, cv::Point(36, 44), cv::FONT_HERSHEY_SIMPLEX, 0.95,
              cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
  cv::putText(canvas, subtitle, cv::Point(36, 76), cv::FONT_HERSHEY_SIMPLEX, 0.52,
              cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
  return canvas;
}

void DrawChartAxes(cv::Mat* canvas,
                   const cv::Rect& plot_rect,
                   double y_min,
                   double y_max,
                   const std::string& y_label) {
  if (canvas == nullptr) {
    return;
  }
  cv::rectangle(*canvas, plot_rect, cv::Scalar(255, 255, 255), cv::FILLED);
  cv::rectangle(*canvas, plot_rect, cv::Scalar(110, 110, 110), 1, cv::LINE_AA);
  for (int tick = 0; tick <= 5; ++tick) {
    const int y = plot_rect.y + static_cast<int>(std::lround(
        static_cast<double>(plot_rect.height) * tick / 5.0));
    cv::line(*canvas, cv::Point(plot_rect.x, y),
             cv::Point(plot_rect.x + plot_rect.width, y),
             cv::Scalar(228, 228, 228), 1, cv::LINE_AA);
    const double value = y_max - (y_max - y_min) * tick / 5.0;
    cv::putText(*canvas, FormatDouble(value, 2),
                cv::Point(plot_rect.x - 68, y + 4), cv::FONT_HERSHEY_PLAIN, 0.9,
                cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
  }
  cv::putText(*canvas, y_label, cv::Point(plot_rect.x - 68, plot_rect.y - 10),
              cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
}

cv::Point ChartPoint(int index,
                     int count,
                     double value,
                     double y_min,
                     double y_max,
                     const cv::Rect& plot_rect) {
  const double x_ratio = count <= 1 ? 0.5 : static_cast<double>(index) / static_cast<double>(count - 1);
  const double y_ratio =
      y_max <= y_min ? 0.5 : (value - y_min) / std::max(1e-9, y_max - y_min);
  const int x = plot_rect.x + static_cast<int>(std::lround(x_ratio * plot_rect.width));
  const int y = plot_rect.y + plot_rect.height -
                static_cast<int>(std::lround(y_ratio * plot_rect.height));
  return cv::Point(x, y);
}

cv::Mat BuildGlobalErrorBarChart(const GlobalGeometrySummary& summary) {
  const std::array<std::string, 4> labels{{"C", "IL", "SP", "Final"}};
  const std::array<double, 4> values{{summary.avg_d_c, summary.avg_d_il,
                                      summary.avg_d_sp, summary.avg_d_final}};
  double y_max = 0.0;
  for (double value : values) {
    y_max = std::max(y_max, value);
  }
  y_max = std::max(1.0, y_max * 1.2);

  cv::Mat canvas = MakeChartCanvas(
      "Global outer geometry error against proxy GT",
      "GT(proxy) = calibrated outer corner after SP followed by image-space subpixel");
  const cv::Rect plot_rect(110, 112, 820, 410);
  DrawChartAxes(&canvas, plot_rect, 0.0, y_max, "avg |method-GT| px");

  const std::array<cv::Scalar, 4> colors{
      cv::Scalar(0, 165, 255),
      cv::Scalar(255, 120, 0),
      cv::Scalar(255, 80, 255),
      cv::Scalar(0, 200, 80),
  };

  const int count = static_cast<int>(labels.size());
  for (int index = 0; index < count; ++index) {
    const double x_ratio = (static_cast<double>(index) + 0.5) / static_cast<double>(count);
    const int center_x = plot_rect.x + static_cast<int>(std::lround(x_ratio * plot_rect.width));
    const int bar_width = std::max(32, plot_rect.width / (count * 3));
    const int zero_y = ChartPoint(0, 1, 0.0, 0.0, y_max, plot_rect).y;
    const int value_y = ChartPoint(0, 1, values[static_cast<std::size_t>(index)], 0.0, y_max, plot_rect).y;
    const cv::Rect bar_rect(center_x - bar_width / 2, value_y, bar_width,
                            std::max(2, zero_y - value_y));
    cv::rectangle(canvas, bar_rect, colors[static_cast<std::size_t>(index)], cv::FILLED);
    cv::rectangle(canvas, bar_rect, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    cv::putText(canvas, labels[static_cast<std::size_t>(index)],
                cv::Point(center_x - 16, plot_rect.y + plot_rect.height + 26),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
    cv::putText(canvas, FormatDouble(values[static_cast<std::size_t>(index)], 2),
                cv::Point(center_x - 18, value_y - 8), cv::FONT_HERSHEY_PLAIN, 1.0,
                cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
  }
  return canvas;
}

cv::Mat BuildPerImageDeltaBarChart(const std::vector<PerImageGeometrySummary>& summaries) {
  double max_abs_value = 0.0;
  for (const auto& summary : summaries) {
    max_abs_value = std::max(max_abs_value, std::abs(summary.avg_delta_sphere_vs_line));
  }
  max_abs_value = std::max(0.5, max_abs_value * 1.2);

  cv::Mat canvas = MakeChartCanvas(
      "Per-image sphere-vs-line delta",
      "delta = |IL-GT| - |SP-GT|, positive means sphere-plane is closer to GT");
  const cv::Rect plot_rect(90, 112, 840, 410);
  DrawChartAxes(&canvas, plot_rect, -max_abs_value, max_abs_value, "delta px");

  const int count = static_cast<int>(summaries.size());
  for (int i = 0; i < count; ++i) {
    const double x_ratio =
        (static_cast<double>(i) + 0.5) / std::max(1.0, static_cast<double>(count));
    const int center_x = plot_rect.x + static_cast<int>(std::lround(x_ratio * plot_rect.width));
    const int bar_width = std::max(18, plot_rect.width / std::max(1, count * 2));
    const int zero_y = ChartPoint(0, 1, 0.0, -max_abs_value, max_abs_value, plot_rect).y;
    const int value_y = ChartPoint(0, 1, summaries[static_cast<std::size_t>(i)].avg_delta_sphere_vs_line,
                                   -max_abs_value, max_abs_value, plot_rect).y;
    const cv::Rect bar_rect(center_x - bar_width / 2, std::min(zero_y, value_y), bar_width,
                            std::max(2, std::abs(zero_y - value_y)));
    const cv::Scalar color =
        summaries[static_cast<std::size_t>(i)].avg_delta_sphere_vs_line >= 0.0
            ? cv::Scalar(255, 80, 255)
            : cv::Scalar(255, 120, 0);
    cv::rectangle(canvas, bar_rect, color, cv::FILLED);
    cv::rectangle(canvas, bar_rect, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    cv::putText(canvas, summaries[static_cast<std::size_t>(i)].image_stem,
                cv::Point(center_x - 26, plot_rect.y + plot_rect.height + 26),
                cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
  }
  return canvas;
}

cv::Mat BuildResidualComparisonChart(const std::vector<PerImageGeometrySummary>& summaries) {
  double line_max = 0.0;
  double plane_max = 0.0;
  for (const auto& summary : summaries) {
    line_max = std::max(line_max, summary.avg_line_rms);
    plane_max = std::max(plane_max, summary.avg_plane_rms);
  }
  line_max = std::max(1.0, line_max * 1.2);
  plane_max = std::max(0.01, plane_max * 1.2);

  cv::Mat canvas = MakeChartCanvas(
      "Residual comparison from the same support points",
      "left: image-line RMS in px, right: sphere-plane RMS in ray-plane residual");
  const cv::Rect left_rect(80, 112, 380, 410);
  const cv::Rect right_rect(540, 112, 380, 410);
  DrawChartAxes(&canvas, left_rect, 0.0, line_max, "line RMS px");
  DrawChartAxes(&canvas, right_rect, 0.0, plane_max, "plane RMS");

  const int count = static_cast<int>(summaries.size());
  for (int i = 0; i < count; ++i) {
    const double x_ratio =
        (static_cast<double>(i) + 0.5) / std::max(1.0, static_cast<double>(count));
    const int left_x = left_rect.x + static_cast<int>(std::lround(x_ratio * left_rect.width));
    const int right_x = right_rect.x + static_cast<int>(std::lround(x_ratio * right_rect.width));
    const int bar_width = std::max(12, left_rect.width / std::max(1, count * 2));

    const int left_zero_y = ChartPoint(0, 1, 0.0, 0.0, line_max, left_rect).y;
    const int left_value_y = ChartPoint(0, 1, summaries[static_cast<std::size_t>(i)].avg_line_rms,
                                        0.0, line_max, left_rect).y;
    const cv::Rect left_bar(left_x - bar_width / 2, left_value_y, bar_width,
                            std::max(2, left_zero_y - left_value_y));
    cv::rectangle(canvas, left_bar, cv::Scalar(255, 120, 0), cv::FILLED);
    cv::rectangle(canvas, left_bar, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);

    const int right_zero_y = ChartPoint(0, 1, 0.0, 0.0, plane_max, right_rect).y;
    const int right_value_y = ChartPoint(0, 1, summaries[static_cast<std::size_t>(i)].avg_plane_rms,
                                         0.0, plane_max, right_rect).y;
    const cv::Rect right_bar(right_x - bar_width / 2, right_value_y, bar_width,
                             std::max(2, right_zero_y - right_value_y));
    cv::rectangle(canvas, right_bar, cv::Scalar(255, 80, 255), cv::FILLED);
    cv::rectangle(canvas, right_bar, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);

    cv::putText(canvas, summaries[static_cast<std::size_t>(i)].image_stem,
                cv::Point(left_x - 24, left_rect.y + left_rect.height + 26),
                cv::FONT_HERSHEY_PLAIN, 0.85, cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
    cv::putText(canvas, summaries[static_cast<std::size_t>(i)].image_stem,
                cv::Point(right_x - 24, right_rect.y + right_rect.height + 26),
                cv::FONT_HERSHEY_PLAIN, 0.85, cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
  }
  return canvas;
}

bool FitImageLine(const std::vector<cv::Point2f>& points,
                  cv::Point2f* anchor,
                  cv::Point2f* direction,
                  double* rms_residual) {
  if (anchor == nullptr || direction == nullptr || rms_residual == nullptr) {
    throw std::runtime_error("FitImageLine requires valid output pointers.");
  }
  if (points.size() < 2) {
    return false;
  }
  cv::Vec4f line;
  cv::fitLine(points, line, cv::DIST_L2, 0.0, 0.01, 0.01);
  *anchor = cv::Point2f(line[2], line[3]);
  *direction = cv::Point2f(line[0], line[1]);
  const double norm = std::hypot(direction->x, direction->y);
  if (!std::isfinite(norm) || norm <= 1e-9) {
    return false;
  }
  *direction *= static_cast<float>(1.0 / norm);
  double residual_sum_sq = 0.0;
  for (const cv::Point2f& point : points) {
    const cv::Point2f delta = point - *anchor;
    const double residual = std::abs(delta.x * direction->y - delta.y * direction->x);
    residual_sum_sq += residual * residual;
  }
  *rms_residual = std::sqrt(residual_sum_sq / static_cast<double>(points.size()));
  return true;
}

void DrawSupportLineSegment(cv::Mat* image,
                            const std::vector<cv::Point2f>& points,
                            const cv::Scalar& color,
                            int thickness) {
  if (image == nullptr || points.size() < 2) {
    return;
  }
  cv::Point2f anchor;
  cv::Point2f direction;
  double rms = 0.0;
  if (!FitImageLine(points, &anchor, &direction, &rms)) {
    return;
  }
  double min_t = std::numeric_limits<double>::infinity();
  double max_t = -std::numeric_limits<double>::infinity();
  for (const cv::Point2f& point : points) {
    const cv::Point2f delta = point - anchor;
    const double projection = delta.x * direction.x + delta.y * direction.y;
    min_t = std::min(min_t, projection);
    max_t = std::max(max_t, projection);
  }
  min_t -= 10.0;
  max_t += 10.0;
  const cv::Point2f start = anchor + direction * static_cast<float>(min_t);
  const cv::Point2f end = anchor + direction * static_cast<float>(max_t);
  cv::line(*image, start, end, color, thickness, cv::LINE_AA);
}

cv::Rect ComputeInsetRect(const ati::OuterCornerVerificationDebugInfo& debug,
                          const cv::Size& image_size) {
  const cv::Point2f center =
      debug.spherical_refinement_valid ? debug.spherical_corner : debug.coarse_corner;
  const int radius = std::max(40, debug.verification_roi_radius + 28);
  const int x0 = std::max(0, static_cast<int>(std::floor(center.x)) - radius);
  const int y0 = std::max(0, static_cast<int>(std::floor(center.y)) - radius);
  const int x1 = std::min(image_size.width, static_cast<int>(std::ceil(center.x)) + radius + 1);
  const int y1 = std::min(image_size.height, static_cast<int>(std::ceil(center.y)) + radius + 1);
  return cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
}

void DrawPointLabel(cv::Mat* image,
                    const cv::Point2f& point,
                    const std::string& text,
                    const cv::Scalar& color,
                    const cv::Point2f& offset) {
  if (image == nullptr) {
    return;
  }
  cv::putText(*image, text,
              point + offset,
              cv::FONT_HERSHEY_PLAIN, 1.0, color, 1, cv::LINE_AA);
}

cv::Mat BuildRawImageGeometryView(const ImageRecord& image_record,
                                  const ati::ApriltagInternalDetectionResult& result) {
  cv::Mat canvas = ToBgr(image_record.image);
  const cv::Scalar kCoarseColor(0, 165, 255);
  const cv::Scalar kImageLineColor(255, 120, 0);
  const cv::Scalar kSphereColor(255, 80, 255);
  const cv::Scalar kGtColor(0, 200, 80);
  const cv::Scalar kPrevSupportColor(170, 210, 255);
  const cv::Scalar kNextSupportColor(255, 220, 180);

  cv::putText(canvas, image_record.stem + " raw fisheye geometry: same supports -> IL vs SP",
              cv::Point(26, 42), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(20, 20, 20), 2,
              cv::LINE_AA);
  cv::putText(canvas, "C orange, IL blue, SP magenta, GT green; light dots/lines use the same support points",
              cv::Point(26, 72), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(70, 70, 70), 1,
              cv::LINE_AA);

  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    const ati::OuterCornerVerificationDebugInfo& debug =
        result.outer_detection.corner_verification_debug[static_cast<std::size_t>(corner_index)];
      const cv::Point2f outer_gt = ComputeOuterSubpixGtCorner(result, corner_index);

    for (const cv::Point2f& point : debug.prev_branch_points) {
      cv::circle(canvas, point, 2, kPrevSupportColor, cv::FILLED, cv::LINE_AA);
    }
    for (const cv::Point2f& point : debug.next_branch_points) {
      cv::circle(canvas, point, 2, kNextSupportColor, cv::FILLED, cv::LINE_AA);
    }
    DrawSupportLineSegment(&canvas, debug.prev_branch_points, kPrevSupportColor, 1);
    DrawSupportLineSegment(&canvas, debug.next_branch_points, kNextSupportColor, 1);

    if (debug.image_line_valid) {
      cv::line(canvas, debug.coarse_corner, debug.image_line_corner, kImageLineColor, 1, cv::LINE_AA);
    }
    if (debug.spherical_refinement_valid) {
      cv::line(canvas, debug.coarse_corner, debug.spherical_corner, kSphereColor, 1, cv::LINE_AA);
    }

    cv::circle(canvas, debug.coarse_corner, 4, kCoarseColor, 2, cv::LINE_AA);
    if (debug.image_line_valid) {
      cv::drawMarker(canvas, debug.image_line_corner, kImageLineColor,
                     cv::MARKER_CROSS, 10, 1, cv::LINE_AA);
    }
    if (debug.spherical_refinement_valid) {
      cv::drawMarker(canvas, debug.spherical_corner, kSphereColor,
                     cv::MARKER_DIAMOND, 10, 1, cv::LINE_AA);
    }
    cv::rectangle(canvas,
                  cv::Rect(static_cast<int>(std::lround(outer_gt.x)) - 3,
                           static_cast<int>(std::lround(outer_gt.y)) - 3, 7, 7),
                  kGtColor, 1, cv::LINE_AA);

    DrawPointLabel(&canvas, debug.coarse_corner, "C" + std::to_string(corner_index),
                   kCoarseColor, cv::Point2f(6.0f, -6.0f));
    if (debug.image_line_valid) {
      DrawPointLabel(&canvas, debug.image_line_corner, "IL", kImageLineColor,
                     cv::Point2f(6.0f, -6.0f));
    }
    if (debug.spherical_refinement_valid) {
      DrawPointLabel(&canvas, debug.spherical_corner, "SP", kSphereColor,
                     cv::Point2f(6.0f, -6.0f));
    }
    DrawPointLabel(&canvas, outer_gt, "GT", kGtColor, cv::Point2f(6.0f, 12.0f));
  }

  return canvas;
}

cv::Mat BuildLineExplanationView(const ImageRecord& image_record,
                                 const ati::ApriltagInternalDetectionResult& result) {
  cv::Mat source = ToBgr(image_record.image);
  constexpr int kCanvasWidth = 1600;
  constexpr int kCanvasHeight = 1200;
  constexpr int kMargin = 60;
  constexpr int kHeaderHeight = 90;
  const int panel_width = (kCanvasWidth - 3 * kMargin) / 2;
  const int panel_height = (kCanvasHeight - kHeaderHeight - 3 * kMargin) / 2;

  cv::Mat canvas(kCanvasHeight, kCanvasWidth, CV_8UC3, cv::Scalar(248, 248, 248));
  cv::putText(canvas, image_record.stem + " local raw-image line fit explanation",
              cv::Point(30, 42), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(20, 20, 20), 2,
              cv::LINE_AA);
  cv::putText(canvas, "Each inset uses the same support points as SP, but solves the corner in raw image 2D lines",
              cv::Point(30, 72), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(70, 70, 70), 1,
              cv::LINE_AA);

  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    const ati::OuterCornerVerificationDebugInfo& debug =
        result.outer_detection.corner_verification_debug[static_cast<std::size_t>(corner_index)];
    const cv::Rect roi = ComputeInsetRect(debug, source.size());
    cv::Mat crop = source(roi).clone();

    const auto to_local = [&](const cv::Point2f& point) {
      return point - cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
    };

    for (const cv::Point2f& point : debug.prev_branch_points) {
      cv::circle(crop, to_local(point), 2, cv::Scalar(170, 210, 255), cv::FILLED, cv::LINE_AA);
    }
    for (const cv::Point2f& point : debug.next_branch_points) {
      cv::circle(crop, to_local(point), 2, cv::Scalar(255, 220, 180), cv::FILLED, cv::LINE_AA);
    }

    std::vector<cv::Point2f> local_prev;
    std::vector<cv::Point2f> local_next;
    local_prev.reserve(debug.prev_branch_points.size());
    local_next.reserve(debug.next_branch_points.size());
    for (const cv::Point2f& point : debug.prev_branch_points) {
      local_prev.push_back(to_local(point));
    }
    for (const cv::Point2f& point : debug.next_branch_points) {
      local_next.push_back(to_local(point));
    }
    DrawSupportLineSegment(&crop, local_prev, cv::Scalar(170, 210, 255), 1);
    DrawSupportLineSegment(&crop, local_next, cv::Scalar(255, 220, 180), 1);

    const cv::Point2f coarse = to_local(debug.coarse_corner);
    const cv::Point2f image_line = to_local(debug.image_line_corner);
    const cv::Point2f spherical = to_local(debug.spherical_corner);
    const cv::Point2f outer_gt = to_local(ComputeOuterSubpixGtCorner(result, corner_index));

    cv::circle(crop, coarse, 4, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
    if (debug.image_line_valid) {
      cv::drawMarker(crop, image_line, cv::Scalar(255, 120, 0),
                     cv::MARKER_CROSS, 10, 1, cv::LINE_AA);
    }
    if (debug.spherical_refinement_valid) {
      cv::drawMarker(crop, spherical, cv::Scalar(255, 80, 255),
                     cv::MARKER_DIAMOND, 10, 1, cv::LINE_AA);
    }
    cv::rectangle(crop,
                  cv::Rect(static_cast<int>(std::lround(outer_gt.x)) - 3,
                           static_cast<int>(std::lround(outer_gt.y)) - 3, 7, 7),
                  cv::Scalar(0, 200, 80), 1, cv::LINE_AA);

    cv::putText(crop, "corner " + std::to_string(corner_index),
                cv::Point(10, 18), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(20, 20, 20), 1,
                cv::LINE_AA);
    cv::putText(crop, "IL rms=" + FormatDouble(0.5 * (debug.prev_image_line_residual + debug.next_image_line_residual), 2),
                cv::Point(10, 36), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 120, 0), 1,
                cv::LINE_AA);
    cv::putText(crop, "SP rms=" + FormatDouble(0.5 * (debug.prev_spherical_residual + debug.next_spherical_residual), 4),
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
  return true;
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

bool KeypointToUnitRay(const ati::DoubleSphereCameraModel& camera,
                       const cv::Point2f& point,
                       Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    throw std::runtime_error("KeypointToUnitRay requires a valid output pointer.");
  }
  *ray = Eigen::Vector3d::Zero();
  if (!camera.keypointToEuclidean(Eigen::Vector2d(point.x, point.y), ray)) {
    return false;
  }
  const double norm = ray->norm();
  if (!std::isfinite(norm) || norm <= 1e-9) {
    return false;
  }
  *ray /= norm;
  return true;
}

cv::Mat BuildOuterSphereGeometryView(const ati::ApriltagInternalConfig& config,
                                     const ImageRecord& image_record,
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

  constexpr int kCanvasWidth = 1600;
  constexpr int kCanvasHeight = 1200;
  constexpr int kMargin = 70;
  constexpr int kHeaderHeight = 90;
  const int panel_width = (kCanvasWidth - 3 * kMargin) / 2;
  const int panel_height = (kCanvasHeight - kHeaderHeight - 3 * kMargin) / 2;

  cv::Mat canvas(kCanvasHeight, kCanvasWidth, CV_8UC3, cv::Scalar(248, 248, 248));
  cv::putText(canvas, image_record.stem + " sphere-plane geometry view",
              cv::Point(40, 46), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(20, 20, 20), 2);
  cv::putText(canvas, "support rays -> boundary planes -> SP ray; green square is the final GT proxy",
              cv::Point(40, 78), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(60, 60, 60), 1);

  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    const ati::OuterCornerVerificationDebugInfo& debug =
        result.outer_detection.corner_verification_debug[static_cast<std::size_t>(corner_index)];
    const int row = corner_index / 2;
    const int col = corner_index % 2;
    const cv::Rect panel_rect(kMargin + col * (panel_width + kMargin),
                              kHeaderHeight + kMargin + row * (panel_height + kMargin),
                              panel_width, panel_height);
    cv::rectangle(canvas, panel_rect, cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(canvas, panel_rect, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);

    const cv::Point2f panel_center(panel_rect.x + panel_rect.width * 0.5f,
                                   panel_rect.y + panel_rect.height * 0.48f);
    const float panel_radius =
        0.36f * static_cast<float>(std::min(panel_rect.width, panel_rect.height));
    cv::circle(canvas, panel_center, static_cast<int>(std::lround(panel_radius)),
               cv::Scalar(210, 210, 210), 1, cv::LINE_AA);

    std::vector<Eigen::Vector3d> prev_rays;
    std::vector<Eigen::Vector3d> next_rays;
    const bool prev_ok = UnprojectImagePointsToRays(camera, debug.prev_branch_points, &prev_rays);
    const bool next_ok = UnprojectImagePointsToRays(camera, debug.next_branch_points, &next_rays);

    Eigen::Vector3d coarse_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d sphere_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d gt_ray = Eigen::Vector3d::Zero();
    const cv::Point2f outer_gt = ComputeOuterSubpixGtCorner(result, corner_index);
    const bool coarse_ok = KeypointToUnitRay(camera, debug.coarse_corner, &coarse_ray);
    const bool sphere_ok = KeypointToUnitRay(camera, debug.spherical_corner, &sphere_ray);
    const bool gt_ok = KeypointToUnitRay(camera, outer_gt, &gt_ray);

    Eigen::Vector3d prev_plane = Eigen::Vector3d::Zero();
    Eigen::Vector3d next_plane = Eigen::Vector3d::Zero();
    double prev_rms = 0.0;
    double next_rms = 0.0;
    const bool prev_plane_ok = prev_ok && FitPlaneToRays(prev_rays, &prev_plane, &prev_rms);
    const bool next_plane_ok = next_ok && FitPlaneToRays(next_rays, &next_plane, &next_rms);

    if (prev_plane_ok) {
      const Eigen::Vector3d anchor = sphere_ok ? sphere_ray : prev_rays.front();
      DrawRayPolyline(&canvas, SamplePlaneGreatCircle(prev_plane, anchor), panel_center,
                      panel_radius, cv::Scalar(255, 120, 0), 2);
    }
    if (next_plane_ok) {
      const Eigen::Vector3d anchor = sphere_ok ? sphere_ray : next_rays.front();
      DrawRayPolyline(&canvas, SamplePlaneGreatCircle(next_plane, anchor), panel_center,
                      panel_radius, cv::Scalar(0, 180, 255), 2);
    }
    for (const Eigen::Vector3d& ray : prev_rays) {
      cv::circle(canvas, MapRayToSpherePanel(ray, panel_center, panel_radius), 3,
                 cv::Scalar(255, 120, 0), cv::FILLED, cv::LINE_AA);
    }
    for (const Eigen::Vector3d& ray : next_rays) {
      cv::circle(canvas, MapRayToSpherePanel(ray, panel_center, panel_radius), 3,
                 cv::Scalar(0, 180, 255), cv::FILLED, cv::LINE_AA);
    }
    if (coarse_ok && sphere_ok) {
      cv::arrowedLine(canvas,
                      MapRayToSpherePanel(coarse_ray, panel_center, panel_radius),
                      MapRayToSpherePanel(sphere_ray, panel_center, panel_radius),
                      cv::Scalar(120, 120, 120), 2, cv::LINE_AA, 0, 0.10);
    }
    if (coarse_ok) {
      cv::circle(canvas, MapRayToSpherePanel(coarse_ray, panel_center, panel_radius), 5,
                 cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
    }
    if (sphere_ok) {
      cv::drawMarker(canvas, MapRayToSpherePanel(sphere_ray, panel_center, panel_radius),
                     cv::Scalar(255, 80, 255), cv::MARKER_DIAMOND, 12, 1, cv::LINE_AA);
    }
    if (gt_ok) {
      const cv::Point2f gt_point = MapRayToSpherePanel(gt_ray, panel_center, panel_radius);
      cv::rectangle(canvas,
                    cv::Rect(static_cast<int>(std::lround(gt_point.x)) - 3,
                             static_cast<int>(std::lround(gt_point.y)) - 3, 7, 7),
                    cv::Scalar(0, 200, 80), 1, cv::LINE_AA);
    }

    std::ostringstream text1;
    text1 << "corner " << corner_index << "  C->SP="
          << FormatDouble(debug.coarse_to_refined_displacement, 2) << "px";
    cv::putText(canvas, text1.str(),
                cv::Point(panel_rect.x + 18, panel_rect.y + panel_rect.height - 74),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
    std::ostringstream text2;
    text2 << "support=" << debug.prev_spherical_support_count
          << "/" << debug.next_spherical_support_count;
    cv::putText(canvas, text2.str(),
                cv::Point(panel_rect.x + 18, panel_rect.y + panel_rect.height - 48),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    std::ostringstream text3;
    text3 << "plane_rms=" << FormatDouble(prev_rms, 4)
          << "/" << FormatDouble(next_rms, 4);
    cv::putText(canvas, text3.str(),
                cv::Point(panel_rect.x + 18, panel_rect.y + panel_rect.height - 24),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
  }
  return canvas;
}

std::vector<PerImageGeometrySummary> SelectRepresentativeImages(
    const std::vector<PerImageGeometrySummary>& summaries) {
  if (summaries.empty()) {
    return {};
  }
  std::vector<PerImageGeometrySummary> sorted = summaries;
  std::sort(sorted.begin(), sorted.end(),
            [](const PerImageGeometrySummary& lhs, const PerImageGeometrySummary& rhs) {
              return lhs.avg_delta_sphere_vs_line < rhs.avg_delta_sphere_vs_line;
            });

  std::vector<PerImageGeometrySummary> picks;
  const auto try_add = [&](const PerImageGeometrySummary& candidate) {
    for (const auto& existing : picks) {
      if (existing.image_stem == candidate.image_stem) {
        return;
      }
    }
    picks.push_back(candidate);
  };

  try_add(sorted.back());
  try_add(sorted[sorted.size() / 2]);
  bool found_non_positive = false;
  for (const auto& row : sorted) {
    if (row.avg_delta_sphere_vs_line <= 0.0) {
      try_add(row);
      found_non_positive = true;
      break;
    }
  }
  if (!found_non_positive) {
    try_add(sorted.front());
  }
  return picks;
}

const DetectionRecord* FindDetectionForStem(const std::vector<ImageRecord>& images,
                                            const std::vector<DetectionRecord>& detections,
                                            const std::string& image_stem,
                                            const ImageRecord** image_record) {
  for (std::size_t index = 0; index < images.size(); ++index) {
    if (images[index].stem == image_stem) {
      if (image_record != nullptr) {
        *image_record = &images[index];
      }
      if (index < detections.size()) {
        return &detections[index];
      }
      return nullptr;
    }
  }
  return nullptr;
}

void SaveRepresentativeVisuals(const boost::filesystem::path& visuals_dir,
                               const ati::ApriltagInternalConfig& config,
                               const std::vector<ImageRecord>& images,
                               const std::vector<DetectionRecord>& detections,
                               const std::vector<PerImageGeometrySummary>& representative_images) {
  boost::filesystem::create_directories(visuals_dir);
  for (const auto& representative : representative_images) {
    const ImageRecord* image_record = nullptr;
    const DetectionRecord* detection =
        FindDetectionForStem(images, detections, representative.image_stem, &image_record);
    if (image_record == nullptr || detection == nullptr || !detection->available ||
        !detection->result.tag_detected) {
      continue;
    }

    const cv::Mat raw_compare = BuildRawImageGeometryView(*image_record, detection->result);
    if (!raw_compare.empty()) {
      cv::imwrite((visuals_dir / (representative.image_stem + "_raw_compare.png")).string(),
                  raw_compare);
    }

    const cv::Mat sphere_view = BuildOuterSphereGeometryView(config, *image_record, detection->result);
    if (!sphere_view.empty()) {
      cv::imwrite((visuals_dir / (representative.image_stem + "_sphere_view.png")).string(),
                  sphere_view);
    }

    const cv::Mat line_view = BuildLineExplanationView(*image_record, detection->result);
    if (!line_view.empty()) {
      cv::imwrite((visuals_dir / (representative.image_stem + "_line_explain.png")).string(),
                  line_view);
    }
  }
}

void WriteReport(const boost::filesystem::path& report_path,
                 const GlobalGeometrySummary& global_summary,
                 const std::vector<PerImageGeometrySummary>& per_image_summaries,
                 const std::vector<PerImageGeometrySummary>& representative_images) {
  std::ofstream stream(report_path.string().c_str());
  stream << "# Outer 几何本质对比实验报告\n\n";
  stream << "## 1. 实验目的\n\n";
  stream << "这次实验直接回答一个问题：\n\n";
  stream << "> 既然已经在原图上找到了 support points，为什么不直接在 raw fisheye 图像里做两条 2D 直线的交点？\n\n";
  stream << "我们固定比较两种 **基于同一批 support points** 的 outer corner 求解方式：\n\n";
  stream << "- `Image-Line (IL)`：raw image 中对两组 support points 各自拟合 2D 直线，再求交点。\n";
  stream << "- `Sphere-Plane (SP)`：同一批 support points 先反投影成 rays，在单位球/射线域拟合两个边界平面，再求交 ray 并投回原图。\n\n";
  stream << "GT 采用 calibrated camera 下 `SP` 之后再做原图亚像素得到的 outer corner，"
         << "记为 `outer_gt (proxy)`。\n\n";

  stream << "## 2. 全局结果\n\n";
  stream << "- participating corners: " << global_summary.total_corners << "\n";
  stream << "- avg `|C-GT|` = " << FormatDouble(global_summary.avg_d_c, 4) << "\n";
  stream << "- avg `|IL-GT|` = " << FormatDouble(global_summary.avg_d_il, 4) << "\n";
  stream << "- avg `|SP-GT|` = " << FormatDouble(global_summary.avg_d_sp, 4) << "\n";
  stream << "- avg `|final-GT|` = " << FormatDouble(global_summary.avg_d_final, 4)
         << "  (这里的 final 与 SP->subpix GT 在当前实验设置下应基本一致)\n";
  stream << "- avg `imp_IL = dC-dIL` = " << FormatDouble(global_summary.avg_imp_il, 4) << "\n";
  stream << "- avg `imp_SP = dC-dSP` = " << FormatDouble(global_summary.avg_imp_sp, 4) << "\n";
  stream << "- avg `delta_sphere_vs_line = dIL-dSP` = "
         << FormatDouble(global_summary.avg_delta_sphere_vs_line, 4)
         << "  (positive means sphere-plane is better)\n";
  stream << "- avg image-line RMS = " << FormatDouble(global_summary.avg_line_rms, 4) << " px\n";
  stream << "- avg sphere-plane RMS = " << FormatDouble(global_summary.avg_plane_rms, 6) << "\n\n";

  stream << "![global_error_bar](charts/global_error_bar.png)\n\n";
  stream << "![per_image_delta_bar](charts/per_image_delta_bar.png)\n\n";
  stream << "![residual_comparison_bar](charts/residual_comparison_bar.png)\n\n";

  stream << "## 3. Per-image 结果\n\n";
  stream << "| image | corners | avg |C-GT| | avg |IL-GT| | avg |SP-GT| | avg delta = |IL-GT|-|SP-GT| |\n";
  stream << "| --- | ---: | ---: | ---: | ---: | ---: |\n";
  for (const auto& summary : per_image_summaries) {
    stream << "| " << summary.image_stem
           << " | " << summary.corner_count
           << " | " << FormatDouble(summary.avg_d_c, 4)
           << " | " << FormatDouble(summary.avg_d_il, 4)
           << " | " << FormatDouble(summary.avg_d_sp, 4)
           << " | " << FormatDouble(summary.avg_delta_sphere_vs_line, 4) << " |\n";
  }
  stream << "\n";

  stream << "## 4. 代表性可视化\n\n";
  stream << "选择规则：`max delta` / `median delta` / `typical non-positive delta`。\n\n";
  for (const auto& representative : representative_images) {
    stream << "### " << representative.image_stem << "\n\n";
    stream << "- corners: " << representative.corner_count << "\n";
    stream << "- avg `|IL-GT|` = " << FormatDouble(representative.avg_d_il, 4) << "\n";
    stream << "- avg `|SP-GT|` = " << FormatDouble(representative.avg_d_sp, 4) << "\n";
    stream << "- avg `delta_sphere_vs_line` = "
           << FormatDouble(representative.avg_delta_sphere_vs_line, 4) << "\n\n";
    stream << "![raw_compare](visuals/" << representative.image_stem << "_raw_compare.png)\n\n";
    stream << "![sphere_view](visuals/" << representative.image_stem << "_sphere_view.png)\n\n";
    stream << "![line_explain](visuals/" << representative.image_stem << "_line_explain.png)\n\n";
  }

  stream << "## 5. 结论总结\n\n";
  if (global_summary.avg_delta_sphere_vs_line > 0.0) {
    stream << "- 从全局均值看，`SP` 比 `IL` 更接近 proxy GT，说明在 raw fisheye 图像上，"
           << "同一条空间边对应的 support points 并不总是更适合直接看作 2D 直线。\n";
  } else {
    stream << "- 从全局均值看，这一组样例中 `SP` 没有稳定优于 `IL`，说明 raw-image 2D 近似在部分图像里仍然足够好。\n";
  }
  stream << "- 需要特别注意的是：`IL` 和 `SP` 使用的是 **同一批 support points**，因此差异主要来自几何建模方式本身，而不是采样点不同。\n";
  stream << "- 若某些中心区域图像里 `IL` 接近 `SP`，这是合理现象；真正能回答问题的关键证据，是边缘强畸变区域中两者是否开始分离。\n";
  stream << "- 这次实验更偏“方法本质解释”，不是 outer 全 pipeline ablation；因此主结论应围绕 `IL vs SP` 展开，而不是围绕最终 `C-SP-S` 指标展开。\n";
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

    const std::vector<ImageRecord> images = LoadSelectedImages(args.image_dir);
    const std::vector<DetectionRecord> detections = RunDetectionPass(images, config, options);
    const std::vector<PerCornerGeometryMetric> per_corner_rows =
        CollectPerCornerMetrics(images, detections);
    const std::vector<PerImageGeometrySummary> per_image_summaries =
        SummarizePerImage(per_corner_rows);
    const GlobalGeometrySummary global_summary = SummarizeGlobal(per_corner_rows);

    if (per_corner_rows.empty()) {
      throw std::runtime_error("No valid outer corners had C/IL/SP/GT simultaneously.");
    }

    const boost::filesystem::path output_dir(args.output_dir);
    const boost::filesystem::path charts_dir = output_dir / "charts";
    const boost::filesystem::path visuals_dir = output_dir / "visuals";
    boost::filesystem::create_directories(output_dir);
    boost::filesystem::create_directories(charts_dir);
    boost::filesystem::create_directories(visuals_dir);

    WritePerCornerCsv(output_dir / "per_corner_geometry_metrics.csv", per_corner_rows);
    WritePerImageCsv(output_dir / "per_image_geometry_summary.csv", per_image_summaries);

    const cv::Mat global_error_bar = BuildGlobalErrorBarChart(global_summary);
    const cv::Mat per_image_delta_bar = BuildPerImageDeltaBarChart(per_image_summaries);
    const cv::Mat residual_bar = BuildResidualComparisonChart(per_image_summaries);
    cv::imwrite((charts_dir / "global_error_bar.png").string(), global_error_bar);
    cv::imwrite((charts_dir / "per_image_delta_bar.png").string(), per_image_delta_bar);
    cv::imwrite((charts_dir / "residual_comparison_bar.png").string(), residual_bar);

    const std::vector<PerImageGeometrySummary> representative_images =
        SelectRepresentativeImages(per_image_summaries);
    SaveRepresentativeVisuals(visuals_dir, config, images, detections, representative_images);
    WriteReport(output_dir / "report.md", global_summary, per_image_summaries, representative_images);

    std::cout << "[compare-outer-geometry] finished\n";
    std::cout << "  output_dir: " << output_dir.string() << "\n";
    std::cout << "  corners: " << global_summary.total_corners << "\n";
    std::cout << "  avg |IL-GT|: " << FormatDouble(global_summary.avg_d_il, 4) << "\n";
    std::cout << "  avg |SP-GT|: " << FormatDouble(global_summary.avg_d_sp, 4) << "\n";
    std::cout << "  avg delta_sphere_vs_line: "
              << FormatDouble(global_summary.avg_delta_sphere_vs_line, 4) << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[compare-outer-geometry] " << error.what() << "\n\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
