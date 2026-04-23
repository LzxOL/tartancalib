#include <aslam/cameras/apriltag_internal/KalibrBenchmark.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

namespace {

IntermediateCameraConfig MakeCameraConfig(const OuterBootstrapCameraIntrinsics& intrinsics) {
  IntermediateCameraConfig config;
  config.camera_model = "ds";
  config.distortion_model = "none";
  config.intrinsics = {intrinsics.xi, intrinsics.alpha, intrinsics.fu, intrinsics.fv,
                       intrinsics.cu, intrinsics.cv};
  config.distortion_coeffs.clear();
  config.resolution = {intrinsics.resolution.width, intrinsics.resolution.height};
  return config;
}

std::string Trim(const std::string& value) {
  const std::size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::vector<double> ParseDoubleList(const std::string& value) {
  std::vector<double> result;
  const std::size_t left = value.find('[');
  const std::size_t right = value.find(']');
  if (left == std::string::npos || right == std::string::npos || right <= left) {
    return result;
  }
  std::stringstream stream(value.substr(left + 1, right - left - 1));
  std::string token;
  while (std::getline(stream, token, ',')) {
    token = Trim(token);
    if (!token.empty()) {
      result.push_back(std::stod(token));
    }
  }
  return result;
}

std::vector<int> ParseIntList(const std::string& value) {
  const std::vector<double> doubles = ParseDoubleList(value);
  std::vector<int> result;
  result.reserve(doubles.size());
  for (double entry : doubles) {
    result.push_back(static_cast<int>(std::lround(entry)));
  }
  return result;
}

}  // namespace

bool LoadKalibrCamchainIntrinsics(const std::string& yaml_path,
                                  OuterBootstrapCameraIntrinsics* intrinsics,
                                  std::string* error_message) {
  if (intrinsics == nullptr) {
    if (error_message != nullptr) {
      *error_message = "LoadKalibrCamchainIntrinsics requires a valid intrinsics pointer.";
    }
    return false;
  }

  std::ifstream input(yaml_path.c_str());
  if (!input.is_open()) {
    if (error_message != nullptr) {
      *error_message = "Failed to open Kalibr camchain yaml: " + yaml_path;
    }
    return false;
  }

  bool in_cam0 = false;
  std::string camera_model;
  std::string distortion_model;
  std::vector<double> intrinsics_values;
  std::vector<int> resolution_values;
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }
    if (trimmed == "cam0:") {
      in_cam0 = true;
      continue;
    }
    if (!in_cam0) {
      continue;
    }
    if (trimmed.find("camera_model:") == 0) {
      camera_model = Trim(trimmed.substr(std::string("camera_model:").size()));
    } else if (trimmed.find("distortion_model:") == 0) {
      distortion_model = Trim(trimmed.substr(std::string("distortion_model:").size()));
    } else if (trimmed.find("intrinsics:") == 0) {
      intrinsics_values = ParseDoubleList(trimmed);
    } else if (trimmed.find("resolution:") == 0) {
      resolution_values = ParseIntList(trimmed);
    } else if (trimmed.find("cam") == 0 && trimmed.back() == ':') {
      break;
    }
  }

  if (camera_model.empty()) {
    if (error_message != nullptr) {
      *error_message = "Kalibr camchain yaml does not contain a readable cam0 block.";
    }
    return false;
  }

  if (camera_model != "ds" || distortion_model != "none") {
    if (error_message != nullptr) {
      *error_message = "Kalibr benchmark v1 expects cam0 with ds / none.";
    }
    return false;
  }
  if (intrinsics_values.size() != 6 || resolution_values.size() != 2) {
    if (error_message != nullptr) {
      *error_message = "Kalibr camchain yaml intrinsics/resolution are malformed.";
    }
    return false;
  }

  intrinsics->xi = intrinsics_values[0];
  intrinsics->alpha = intrinsics_values[1];
  intrinsics->fu = intrinsics_values[2];
  intrinsics->fv = intrinsics_values[3];
  intrinsics->cu = intrinsics_values[4];
  intrinsics->cv = intrinsics_values[5];
  intrinsics->resolution = cv::Size(resolution_values[0], resolution_values[1]);
  return intrinsics->IsValid();
}

namespace {

cv::Scalar ColorForDelta(double value, double max_value) {
  const double scale = max_value > 1e-12 ? std::min(1.0, value / max_value) : 0.0;
  return cv::Scalar(255.0 * scale, 255.0 * (1.0 - scale), 32.0);
}

}  // namespace

KalibrBenchmarkReport KalibrBenchmark::Compare(const KalibrBenchmarkInput& input) const {
  KalibrBenchmarkReport report;
  report.dataset_label = input.dataset_label.empty()
                             ? input.our_bundle.scene_state.dataset_label
                             : input.dataset_label;
  report.our_reference_board_id = input.our_bundle.scene_state.reference_board_id;

  if (!input.our_bundle.scene_state.IsValid()) {
    report.failure_reason = "Our Stage 5 bundle does not contain a valid DS scene state.";
    return report;
  }
  if (input.camera_model != "ds") {
    report.failure_reason = "Kalibr benchmark v1 only supports camera_model=ds.";
    return report;
  }

  std::string error_message;
  if (!LoadKalibrCamchainIntrinsics(input.kalibr_camchain_yaml, &report.kalibr_intrinsics,
                                    &error_message)) {
    report.failure_reason = error_message;
    return report;
  }

  report.our_intrinsics = input.our_bundle.scene_state.camera;
  report.intrinsics_delta.xi = report.our_intrinsics.xi - report.kalibr_intrinsics.xi;
  report.intrinsics_delta.alpha = report.our_intrinsics.alpha - report.kalibr_intrinsics.alpha;
  report.intrinsics_delta.fu = report.our_intrinsics.fu - report.kalibr_intrinsics.fu;
  report.intrinsics_delta.fv = report.our_intrinsics.fv - report.kalibr_intrinsics.fv;
  report.intrinsics_delta.cu = report.our_intrinsics.cu - report.kalibr_intrinsics.cu;
  report.intrinsics_delta.cv = report.our_intrinsics.cv - report.kalibr_intrinsics.cv;
  report.intrinsics_delta.l2_norm =
      std::sqrt(report.intrinsics_delta.xi * report.intrinsics_delta.xi +
                report.intrinsics_delta.alpha * report.intrinsics_delta.alpha +
                report.intrinsics_delta.fu * report.intrinsics_delta.fu +
                report.intrinsics_delta.fv * report.intrinsics_delta.fv +
                report.intrinsics_delta.cu * report.intrinsics_delta.cu +
                report.intrinsics_delta.cv * report.intrinsics_delta.cv);

  report.residual_summary.our_overall_rmse = input.our_bundle.residual_result.overall_rmse;
  report.residual_summary.our_outer_only_rmse = input.our_bundle.residual_result.outer_only_rmse;
  report.residual_summary.our_internal_only_rmse =
      input.our_bundle.residual_result.internal_only_rmse;
  report.residual_summary.kalibr_residual_available = false;
  report.residual_summary.note =
      "Kalibr camchain.yaml does not expose directly comparable reprojection residual "
      "statistics, so this Stage 5 v1 benchmark reports our optimized residuals plus "
      "camera-model/output-level DS intrinsics and projection comparison.";

  try {
    const DoubleSphereCameraModel our_camera =
        DoubleSphereCameraModel::FromConfig(MakeCameraConfig(report.our_intrinsics));
    const DoubleSphereCameraModel kalibr_camera =
        DoubleSphereCameraModel::FromConfig(MakeCameraConfig(report.kalibr_intrinsics));

    KalibrProjectionComparison projection_compare;
    projection_compare.resolution = report.our_intrinsics.resolution;
    constexpr int kGridCols = 32;
    constexpr int kGridRows = 32;
    double delta_sum = 0.0;
    double max_delta = 0.0;

    for (int gy = 0; gy < kGridRows; ++gy) {
      for (int gx = 0; gx < kGridCols; ++gx) {
        KalibrProjectionComparisonSample sample;
        const double px =
            (static_cast<double>(gx) + 0.5) /
            static_cast<double>(kGridCols) *
            static_cast<double>(report.our_intrinsics.resolution.width - 1);
        const double py =
            (static_cast<double>(gy) + 0.5) /
            static_cast<double>(kGridRows) *
            static_cast<double>(report.our_intrinsics.resolution.height - 1);
        sample.anchor_pixel = Eigen::Vector2d(px, py);

        Eigen::Vector3d ray;
        if (!our_camera.keypointToEuclidean(sample.anchor_pixel, &ray)) {
          projection_compare.samples.push_back(sample);
          continue;
        }
        if (!our_camera.vsEuclideanToKeypoint(ray, &sample.our_projected_pixel)) {
          projection_compare.samples.push_back(sample);
          continue;
        }
        if (!kalibr_camera.vsEuclideanToKeypoint(ray, &sample.kalibr_projected_pixel)) {
          projection_compare.samples.push_back(sample);
          continue;
        }

        sample.delta_pixels =
            (sample.our_projected_pixel - sample.kalibr_projected_pixel).norm();
        sample.valid = true;
        delta_sum += sample.delta_pixels;
        max_delta = std::max(max_delta, sample.delta_pixels);
        ++projection_compare.valid_sample_count;
        projection_compare.samples.push_back(sample);
      }
    }

    if (projection_compare.valid_sample_count > 0) {
      projection_compare.mean_pixel_delta =
          delta_sum / static_cast<double>(projection_compare.valid_sample_count);
      projection_compare.max_pixel_delta = max_delta;
    } else {
      report.warnings.push_back("Projection comparison produced zero valid DS samples.");
    }
    report.distortion_or_projection_compare = projection_compare;
  } catch (const std::exception& error) {
    report.failure_reason = error.what();
    return report;
  }

  report.success = true;
  return report;
}

cv::Mat KalibrBenchmark::RenderProjectionComparison(const KalibrBenchmarkReport& report,
                                                    int max_width,
                                                    int max_height) const {
  if (!report.success || report.distortion_or_projection_compare.resolution.width <= 0 ||
      report.distortion_or_projection_compare.resolution.height <= 0) {
    return cv::Mat();
  }

  const cv::Size resolution = report.distortion_or_projection_compare.resolution;
  const double scale =
      std::min(static_cast<double>(max_width) / static_cast<double>(resolution.width),
               static_cast<double>(max_height) / static_cast<double>(resolution.height));
  const int draw_width = std::max(320, static_cast<int>(std::round(resolution.width * scale)));
  const int draw_height = std::max(320, static_cast<int>(std::round(resolution.height * scale)));
  cv::Mat canvas(draw_height + 120, draw_width, CV_8UC3, cv::Scalar(250, 250, 250));

  for (const KalibrProjectionComparisonSample& sample :
       report.distortion_or_projection_compare.samples) {
    if (!sample.valid) {
      continue;
    }
    const cv::Point point(static_cast<int>(std::round(sample.anchor_pixel.x() * scale)),
                          static_cast<int>(std::round(sample.anchor_pixel.y() * scale)));
    cv::circle(canvas, point, 4,
               ColorForDelta(sample.delta_pixels,
                             std::max(1.0, report.distortion_or_projection_compare.max_pixel_delta)),
               cv::FILLED, cv::LINE_AA);
  }

  cv::putText(canvas,
              "Stage 5 DS projection compare: our round2 stabilized state vs Kalibr camchain",
              cv::Point(20, draw_height + 32), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
  std::ostringstream metrics;
  metrics << "mean_delta=" << report.distortion_or_projection_compare.mean_pixel_delta
          << " px, max_delta=" << report.distortion_or_projection_compare.max_pixel_delta
          << " px, valid_samples="
          << report.distortion_or_projection_compare.valid_sample_count;
  cv::putText(canvas, metrics.str(), cv::Point(20, draw_height + 68),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(32, 32, 32), 2, cv::LINE_AA);
  return canvas;
}

void WriteKalibrBenchmarkSummary(const std::string& path,
                                 const KalibrBenchmarkReport& report) {
  std::ofstream output(path.c_str());
  output << "success: " << (report.success ? 1 : 0) << "\n";
  output << "failure_reason: " << report.failure_reason << "\n";
  output << "dataset_label: " << report.dataset_label << "\n";
  output << "our_reference_board_id: " << report.our_reference_board_id << "\n";
  output << "our_xi: " << report.our_intrinsics.xi << "\n";
  output << "our_alpha: " << report.our_intrinsics.alpha << "\n";
  output << "our_fu: " << report.our_intrinsics.fu << "\n";
  output << "our_fv: " << report.our_intrinsics.fv << "\n";
  output << "our_cu: " << report.our_intrinsics.cu << "\n";
  output << "our_cv: " << report.our_intrinsics.cv << "\n";
  output << "kalibr_xi: " << report.kalibr_intrinsics.xi << "\n";
  output << "kalibr_alpha: " << report.kalibr_intrinsics.alpha << "\n";
  output << "kalibr_fu: " << report.kalibr_intrinsics.fu << "\n";
  output << "kalibr_fv: " << report.kalibr_intrinsics.fv << "\n";
  output << "kalibr_cu: " << report.kalibr_intrinsics.cu << "\n";
  output << "kalibr_cv: " << report.kalibr_intrinsics.cv << "\n";
  output << "delta_l2_norm: " << report.intrinsics_delta.l2_norm << "\n";
  output << "projection_mean_pixel_delta: "
         << report.distortion_or_projection_compare.mean_pixel_delta << "\n";
  output << "projection_max_pixel_delta: "
         << report.distortion_or_projection_compare.max_pixel_delta << "\n";
  output << "projection_valid_sample_count: "
         << report.distortion_or_projection_compare.valid_sample_count << "\n";
  for (const std::string& warning : report.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteKalibrBenchmarkIntrinsicsCsv(const std::string& path,
                                       const KalibrBenchmarkReport& report) {
  std::ofstream output(path.c_str());
  output << "parameter,our_value,kalibr_value,delta\n";
  output << "xi," << report.our_intrinsics.xi << "," << report.kalibr_intrinsics.xi << ","
         << report.intrinsics_delta.xi << "\n";
  output << "alpha," << report.our_intrinsics.alpha << "," << report.kalibr_intrinsics.alpha
         << "," << report.intrinsics_delta.alpha << "\n";
  output << "fu," << report.our_intrinsics.fu << "," << report.kalibr_intrinsics.fu << ","
         << report.intrinsics_delta.fu << "\n";
  output << "fv," << report.our_intrinsics.fv << "," << report.kalibr_intrinsics.fv << ","
         << report.intrinsics_delta.fv << "\n";
  output << "cu," << report.our_intrinsics.cu << "," << report.kalibr_intrinsics.cu << ","
         << report.intrinsics_delta.cu << "\n";
  output << "cv," << report.our_intrinsics.cv << "," << report.kalibr_intrinsics.cv << ","
         << report.intrinsics_delta.cv << "\n";
}

void WriteKalibrBenchmarkResidualSummary(const std::string& path,
                                         const KalibrBenchmarkReport& report) {
  std::ofstream output(path.c_str());
  output << "our_overall_rmse: " << report.residual_summary.our_overall_rmse << "\n";
  output << "our_outer_only_rmse: " << report.residual_summary.our_outer_only_rmse << "\n";
  output << "our_internal_only_rmse: " << report.residual_summary.our_internal_only_rmse << "\n";
  output << "kalibr_residual_available: "
         << (report.residual_summary.kalibr_residual_available ? 1 : 0) << "\n";
  output << "note: " << report.residual_summary.note << "\n";
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
