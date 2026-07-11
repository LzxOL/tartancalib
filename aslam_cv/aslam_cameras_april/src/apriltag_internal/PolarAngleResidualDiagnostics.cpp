#include <aslam/cameras/apriltag_internal/PolarAngleResidualDiagnostics.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

constexpr double kRadToDeg = 180.0 / M_PI;

double ComputeMedian(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t size = values.size();
  if (size % 2 == 1) {
    return values[size / 2];
  }
  return 0.5 * (values[size / 2 - 1] + values[size / 2]);
}

double ComputePercentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double index = percentile * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(index));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
  if (lower == upper) {
    return values[lower];
  }
  const double fraction = index - static_cast<double>(lower);
  return (1.0 - fraction) * values[lower] + fraction * values[upper];
}

double ComputeStd(double sum, double sum_sq, int count) {
  if (count <= 0) {
    return 0.0;
  }
  const double mean = sum / static_cast<double>(count);
  const double variance =
      std::max(0.0, sum_sq / static_cast<double>(count) - mean * mean);
  return std::sqrt(variance);
}

double ComputeRmse(double squared_sum, int count) {
  if (count <= 0) {
    return 0.0;
  }
  return std::sqrt(squared_sum / static_cast<double>(count));
}

std::string FormatPointKey(int frame_index, int board_id, int point_id) {
  std::ostringstream oss;
  oss << frame_index << "_" << board_id << "_" << point_id;
  return oss.str();
}

}  // namespace

PolarAngleResidualDiagnostics::PolarAngleResidualDiagnostics(
    PolarAngleDiagnosticsOptions options)
    : options_(std::move(options)) {}

double PolarAngleResidualDiagnostics::ComputePolarAngleDeg(
    const DoubleSphereCameraModel& camera,
    const Eigen::Vector2d& pixel) const {
  Eigen::Vector3d ray;
  if (!camera.keypointToEuclidean(pixel, &ray)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return ComputePolarAngleDegFromRay(ray);
}

double ComputePolarAngleDegFromRay(const Eigen::Vector3d& ray) {
  const double norm = ray.norm();
  if (norm <= 0.0 || !std::isfinite(norm)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double cos_theta = ray.z() / norm;
  const double clamped_cos = std::max(-1.0, std::min(1.0, cos_theta));
  return std::acos(clamped_cos) * kRadToDeg;
}

double ComputePolarAngleDeg(
    const DoubleSphereCameraModel& camera,
    const Eigen::Vector2d& pixel) {
  Eigen::Vector3d ray;
  if (!camera.keypointToEuclidean(pixel, &ray)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return ComputePolarAngleDegFromRay(ray);
}

void PolarAngleResidualDiagnostics::ComputeBinStatistics(
    const std::vector<double>& polar_angles,
    const std::vector<double>& residual_norms,
    const std::vector<double>& angular_residual_norms,
    double angular_equiv_focal_px,
    PolarAngleBinStatistics* stats) const {
  if (stats == nullptr) {
    throw std::runtime_error("ComputeBinStatistics requires valid output pointer.");
  }
  if (polar_angles.size() != residual_norms.size()) {
    throw std::runtime_error("Polar angles and residuals must have same size.");
  }
  if (polar_angles.size() != angular_residual_norms.size()) {
    throw std::runtime_error(
        "Polar angles and angular residuals must have same size.");
  }

  const std::size_t n = polar_angles.size();
  if (n == 0) {
    return;
  }

  double squared_sum = 0.0;
  double sum_abs_x = 0.0;
  double sum_abs_y = 0.0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_x_sq = 0.0;
  double sum_y_sq = 0.0;
  double angular_squared_sum = 0.0;
  std::vector<double> residuals;
  residuals.reserve(n);
  std::vector<double> angular_residuals;
  angular_residuals.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    const double r = residual_norms[i];
    squared_sum += r * r;
    residuals.push_back(r);
    const double angular_r = angular_residual_norms[i];
    angular_squared_sum += angular_r * angular_r;
    angular_residuals.push_back(angular_r);
  }

  stats->rmse = ComputeRmse(squared_sum, static_cast<int>(n));
  stats->pixel_rmse_px = stats->rmse;
  stats->median_residual = ComputeMedian(residuals);
  stats->p90_residual = ComputePercentile(residuals, 0.90);
  stats->p95_residual = ComputePercentile(residuals, 0.95);
  stats->max_residual = residuals.empty() ? 0.0 : *std::max_element(residuals.begin(), residuals.end());
  stats->angular_rmse_rad =
      ComputeRmse(angular_squared_sum, static_cast<int>(n));
  stats->angular_equiv_px = stats->angular_rmse_rad * angular_equiv_focal_px;
  stats->angular_median_rad = ComputeMedian(angular_residuals);
  stats->angular_p90_rad = ComputePercentile(angular_residuals, 0.90);
  stats->angular_p95_rad = ComputePercentile(angular_residuals, 0.95);
  stats->angular_max_rad =
      angular_residuals.empty()
          ? 0.0
          : *std::max_element(angular_residuals.begin(),
                              angular_residuals.end());
  stats->mean_abs_x = sum_abs_x / static_cast<double>(n);
  stats->mean_abs_y = sum_abs_y / static_cast<double>(n);
  stats->std_x = ComputeStd(sum_x, sum_x_sq, static_cast<int>(n));
  stats->std_y = ComputeStd(sum_y, sum_y_sq, static_cast<int>(n));
}

void PolarAngleResidualDiagnostics::WriteSummaryFile(
    const std::string& path,
    const PolarAngleDiagnosticsResult& result) const {
  std::ofstream output(path.c_str());
  if (!output.is_open()) {
    throw std::runtime_error("Failed to open summary file: " + path);
  }

  output << "======================================================================\n";
  output << "POLAR ANGLE RESIDUAL DIAGNOSTICS SUMMARY\n";
  output << "======================================================================\n\n";

  output << "Configuration:\n";
  output << "  enabled: " << (options_.enabled ? "true" : "false") << "\n";
  output << "  bin_edges_deg:";
  for (const double edge : options_.bin_edges_deg) {
    output << " " << edge;
  }
  output << "\n";
  output << "  angular_equiv_focal_px: "
         << result.angular_equiv_focal_px;
  output << "\n\n";

  if (!result.success) {
    output << "DIAGNOSTICS FAILED: " << result.failure_reason << "\n";
    for (const std::string& warning : result.warnings) {
      output << "WARNING: " << warning << "\n";
    }
    return;
  }

  auto printBinTable = [&output](const std::string& title,
                                  const std::vector<PolarAngleBinStatistics>& bins) {
    if (bins.empty()) {
      return;
    }
    output << title << "\n";
    output << std::string(70, '-') << "\n";
    output << std::setw(10) << "Bin(deg)"
           << std::setw(8) << "Count"
           << std::setw(12) << "PixRMSE"
           << std::setw(12) << "AngRad"
           << std::setw(12) << "AngEqPx"
           << std::setw(12) << "PixP95"
           << std::setw(12) << "AngP95"
           << "\n";
    output << std::string(70, '-') << "\n";
    for (const PolarAngleBinStatistics& bin : bins) {
      output << std::setw(5) << std::fixed << std::setprecision(1)
             << bin.bin_min_deg << "-"
             << std::setw(4) << bin.bin_max_deg
             << std::setw(8) << std::resetiosflags(std::ios_base::fixed)
             << bin.point_count
             << std::setw(12) << std::fixed << std::setprecision(4)
             << bin.pixel_rmse_px
             << std::setw(12) << std::scientific << std::setprecision(3)
             << bin.angular_rmse_rad
             << std::setw(12) << std::fixed << std::setprecision(4)
             << bin.angular_equiv_px
             << std::setw(12) << bin.p95_residual
             << std::setw(12) << std::scientific << std::setprecision(3)
             << bin.angular_p95_rad
             << std::fixed
             << "\n";
    }
    output << "\n";
  };

  printBinTable("ALL POINTS", result.all_points_bins);
  printBinTable("OUTER ONLY", result.outer_only_bins);
  printBinTable("INTERNAL ONLY", result.internal_only_bins);

  if (!result.per_board_bins.empty()) {
    output << "PER-BOARD SUMMARY\n";
    output << std::string(70, '-') << "\n";
    for (const auto& entry : result.per_board_bins) {
      output << "Board " << entry.first << ":\n";
      printBinTable("", entry.second);
    }
  }

  if (!result.per_frame_bins.empty()) {
    output << "PER-FRAME SUMMARY\n";
    output << std::string(70, '-') << "\n";
    for (const auto& entry : result.per_frame_bins) {
      output << "Frame " << entry.first << ":\n";
      printBinTable("", entry.second);
    }
  }

  if (!result.warnings.empty()) {
    output << "\nWARNINGS:\n";
    for (const std::string& warning : result.warnings) {
      output << "  - " << warning << "\n";
    }
  }

  output << "\n======================================================================\n";
  output << "END OF REPORT\n";
  output << "======================================================================\n";
}

void PolarAngleResidualDiagnostics::WriteCsvFile(
    const std::string& path,
    const PolarAngleDiagnosticsResult& result) const {
  std::ofstream output(path.c_str());
  if (!output.is_open()) {
    throw std::runtime_error("Failed to open CSV file: " + path);
  }

  output << "bin_min_deg,bin_max_deg,point_type,point_count,rmse,"
         << "pixel_rmse_px,angular_rmse_rad,angular_equiv_px,"
         << "mean_abs_x,mean_abs_y,std_x,std_y,"
         << "median_residual,p90_residual,p95_residual,max_residual,"
         << "angular_median_rad,angular_p90_rad,angular_p95_rad,"
         << "angular_max_rad\n";

  auto writeBins = [&output](const std::string& type,
                              const std::vector<PolarAngleBinStatistics>& bins) {
    for (const PolarAngleBinStatistics& bin : bins) {
      output << std::fixed << std::setprecision(6)
             << bin.bin_min_deg << ","
             << bin.bin_max_deg << ","
             << type << ","
             << bin.point_count << ","
             << std::setprecision(6)
             << bin.rmse << ","
             << bin.pixel_rmse_px << ","
             << bin.angular_rmse_rad << ","
             << bin.angular_equiv_px << ","
             << bin.mean_abs_x << ","
             << bin.mean_abs_y << ","
             << bin.std_x << ","
             << bin.std_y << ","
             << bin.median_residual << ","
             << bin.p90_residual << ","
             << bin.p95_residual << ","
             << bin.max_residual << ","
             << bin.angular_median_rad << ","
             << bin.angular_p90_rad << ","
             << bin.angular_p95_rad << ","
             << bin.angular_max_rad << "\n";
    }
  };

  writeBins("all", result.all_points_bins);
  writeBins("outer", result.outer_only_bins);
  writeBins("internal", result.internal_only_bins);
}

PolarAngleDiagnosticsResult PolarAngleResidualDiagnostics::EvaluateWithResiduals(
    const CalibrationMeasurementDataset& measurement_dataset,
    const JointResidualEvaluationResult& residual_result,
    const JointReprojectionSceneState& scene_state,
    const std::string& output_dir) const {
  PolarAngleDiagnosticsResult result;

  if (!options_.enabled) {
    result.success = true;
    return result;
  }

  if (options_.bin_edges_deg.size() < 2) {
    result.failure_reason = "At least 2 bin edges are required.";
    return result;
  }

  IntermediateCameraConfig camera_config;
  camera_config.camera_model = scene_state.camera.camera_model;
  camera_config.distortion_model = scene_state.camera.distortion_model;
  camera_config.intrinsics = scene_state.camera.IntrinsicsVector();
  camera_config.distortion_coeffs = scene_state.camera.distortion_coeffs;
  camera_config.resolution = {
      scene_state.camera.resolution.width,
      scene_state.camera.resolution.height};

  DoubleSphereCameraModel camera = DoubleSphereCameraModel::FromConfig(camera_config);
  if (!camera.IsValid()) {
    result.failure_reason = "Invalid camera model configuration.";
    return result;
  }
  const std::vector<double> scene_intrinsics =
      scene_state.camera.IntrinsicsVector();
  if (scene_intrinsics.size() >= 4) {
    result.angular_equiv_focal_px =
        0.5 * (std::abs(scene_intrinsics[2]) +
               std::abs(scene_intrinsics[3]));
  }

  std::map<std::string, const JointResidualPointDiagnostics*> point_diag_by_key;
  for (const auto& diag : residual_result.point_diagnostics) {
    const std::string key = FormatPointKey(
        diag.frame_index, diag.board_id, diag.point_id);
    point_diag_by_key[key] = &diag;
  }

  const int num_bins = static_cast<int>(options_.bin_edges_deg.size()) - 1;
  std::vector<std::vector<double>> bin_angles_all(num_bins);
  std::vector<std::vector<double>> bin_residuals_all(num_bins);
  std::vector<std::vector<double>> bin_angular_residuals_all(num_bins);
  std::vector<std::vector<double>> bin_residual_x_all(num_bins);
  std::vector<std::vector<double>> bin_residual_y_all(num_bins);
  std::vector<std::vector<double>> bin_angles_outer(num_bins);
  std::vector<std::vector<double>> bin_residuals_outer(num_bins);
  std::vector<std::vector<double>> bin_angular_residuals_outer(num_bins);
  std::vector<std::vector<double>> bin_residual_x_outer(num_bins);
  std::vector<std::vector<double>> bin_residual_y_outer(num_bins);
  std::vector<std::vector<double>> bin_angles_internal(num_bins);
  std::vector<std::vector<double>> bin_residuals_internal(num_bins);
  std::vector<std::vector<double>> bin_angular_residuals_internal(num_bins);
  std::vector<std::vector<double>> bin_residual_x_internal(num_bins);
  std::vector<std::vector<double>> bin_residual_y_internal(num_bins);
  std::map<int, std::vector<std::vector<double>>> board_bin_angles;
  std::map<int, std::vector<std::vector<double>>> board_bin_residuals;
  std::map<int, std::vector<std::vector<double>>> board_bin_angular_residuals;
  std::map<int, std::vector<std::vector<double>>> board_bin_residual_x;
  std::map<int, std::vector<std::vector<double>>> board_bin_residual_y;
  std::map<int, std::vector<std::vector<double>>> frame_bin_angles;
  std::map<int, std::vector<std::vector<double>>> frame_bin_residuals;
  std::map<int, std::vector<std::vector<double>>> frame_bin_angular_residuals;
  std::map<int, std::vector<std::vector<double>>> frame_bin_residual_x;
  std::map<int, std::vector<std::vector<double>>> frame_bin_residual_y;

  std::set<int> observed_frames;
  for (const JointPointObservation& obs : measurement_dataset.solver_observations) {
    if (!obs.used_in_solver) {
      continue;
    }
    observed_frames.insert(obs.frame_index);
  }
  for (int frame_idx : observed_frames) {
    frame_bin_angles[frame_idx] = std::vector<std::vector<double>>(num_bins);
    frame_bin_residuals[frame_idx] = std::vector<std::vector<double>>(num_bins);
    frame_bin_angular_residuals[frame_idx] =
        std::vector<std::vector<double>>(num_bins);
    frame_bin_residual_x[frame_idx] = std::vector<std::vector<double>>(num_bins);
    frame_bin_residual_y[frame_idx] = std::vector<std::vector<double>>(num_bins);
  }

  auto find_bin = [this](double angle) -> int {
    for (std::size_t i = 0; i < options_.bin_edges_deg.size() - 1; ++i) {
      if (angle >= options_.bin_edges_deg[i] &&
          angle < options_.bin_edges_deg[i + 1]) {
        return static_cast<int>(i);
      }
    }
    return -1;
  };

  for (const JointPointObservation& obs : measurement_dataset.solver_observations) {
    if (!obs.used_in_solver) {
      continue;
    }

    const std::string key = FormatPointKey(
        obs.frame_index, obs.board_id, obs.point_id);
    const auto it = point_diag_by_key.find(key);
    if (it == point_diag_by_key.end()) {
      continue;
    }
    const JointResidualPointDiagnostics* diag = it->second;

    const double polar_angle = ComputePolarAngleDeg(camera, obs.image_xy);
    if (!std::isfinite(polar_angle)) {
      result.warnings.push_back(
          "Skipping observation with invalid polar angle: " + key);
      continue;
    }

    const int bin_idx = find_bin(polar_angle);
    if (bin_idx < 0) {
      continue;
    }

    const double residual_norm = diag->residual_norm;
    const double angular_residual_norm = diag->angular_residual_norm;
    const double residual_x = diag->residual_xy.x();
    const double residual_y = diag->residual_xy.y();

    bin_angles_all[bin_idx].push_back(polar_angle);
    bin_residuals_all[bin_idx].push_back(residual_norm);
    bin_angular_residuals_all[bin_idx].push_back(angular_residual_norm);
    bin_residual_x_all[bin_idx].push_back(residual_x);
    bin_residual_y_all[bin_idx].push_back(residual_y);

    if (obs.point_type == JointPointType::Outer) {
      bin_angles_outer[bin_idx].push_back(polar_angle);
      bin_residuals_outer[bin_idx].push_back(residual_norm);
      bin_angular_residuals_outer[bin_idx].push_back(angular_residual_norm);
      bin_residual_x_outer[bin_idx].push_back(residual_x);
      bin_residual_y_outer[bin_idx].push_back(residual_y);
    } else {
      bin_angles_internal[bin_idx].push_back(polar_angle);
      bin_residuals_internal[bin_idx].push_back(residual_norm);
      bin_angular_residuals_internal[bin_idx].push_back(angular_residual_norm);
      bin_residual_x_internal[bin_idx].push_back(residual_x);
      bin_residual_y_internal[bin_idx].push_back(residual_y);
    }

    if (board_bin_angles.find(obs.board_id) == board_bin_angles.end()) {
      board_bin_angles[obs.board_id] = std::vector<std::vector<double>>(num_bins);
      board_bin_residuals[obs.board_id] = std::vector<std::vector<double>>(num_bins);
      board_bin_angular_residuals[obs.board_id] =
          std::vector<std::vector<double>>(num_bins);
      board_bin_residual_x[obs.board_id] = std::vector<std::vector<double>>(num_bins);
      board_bin_residual_y[obs.board_id] = std::vector<std::vector<double>>(num_bins);
    }
    board_bin_angles[obs.board_id][bin_idx].push_back(polar_angle);
    board_bin_residuals[obs.board_id][bin_idx].push_back(residual_norm);
    board_bin_angular_residuals[obs.board_id][bin_idx].push_back(
        angular_residual_norm);
    board_bin_residual_x[obs.board_id][bin_idx].push_back(residual_x);
    board_bin_residual_y[obs.board_id][bin_idx].push_back(residual_y);

    auto frame_it = frame_bin_angles.find(obs.frame_index);
    if (frame_it != frame_bin_angles.end()) {
      frame_it->second[bin_idx].push_back(polar_angle);
      frame_bin_residuals[obs.frame_index][bin_idx].push_back(residual_norm);
      frame_bin_angular_residuals[obs.frame_index][bin_idx].push_back(
          angular_residual_norm);
      frame_bin_residual_x[obs.frame_index][bin_idx].push_back(residual_x);
      frame_bin_residual_y[obs.frame_index][bin_idx].push_back(residual_y);
    }
  }

  auto processBins = [this](const std::vector<std::vector<double>>& bin_angles,
                            const std::vector<std::vector<double>>& bin_residuals,
                            const std::vector<std::vector<double>>& bin_angular_residuals,
                            const std::vector<std::vector<double>>& bin_residual_x,
                            const std::vector<std::vector<double>>& bin_residual_y,
                            double angular_equiv_focal_px,
                            const std::string& type) {
    std::vector<PolarAngleBinStatistics> bins;
    for (std::size_t i = 0; i < bin_angles.size(); ++i) {
      PolarAngleBinStatistics stats;
      stats.bin_min_deg = options_.bin_edges_deg[i];
      stats.bin_max_deg = options_.bin_edges_deg[i + 1];
      stats.point_type = type;
      stats.point_count = static_cast<int>(bin_angles[i].size());
      ComputeBinStatistics(bin_angles[i], bin_residuals[i],
                           bin_angular_residuals[i],
                           angular_equiv_focal_px, &stats);
      if (bin_residual_x[i].size() == bin_residual_y[i].size() &&
          !bin_residual_x[i].empty()) {
        double sum_abs_x = 0.0;
        double sum_abs_y = 0.0;
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_x_sq = 0.0;
        double sum_y_sq = 0.0;
        for (std::size_t j = 0; j < bin_residual_x[i].size(); ++j) {
          const double rx = bin_residual_x[i][j];
          const double ry = bin_residual_y[i][j];
          sum_abs_x += std::abs(rx);
          sum_abs_y += std::abs(ry);
          sum_x += rx;
          sum_y += ry;
          sum_x_sq += rx * rx;
          sum_y_sq += ry * ry;
        }
        const int count = static_cast<int>(bin_residual_x[i].size());
        stats.mean_abs_x = sum_abs_x / static_cast<double>(count);
        stats.mean_abs_y = sum_abs_y / static_cast<double>(count);
        stats.std_x = ComputeStd(sum_x, sum_x_sq, count);
        stats.std_y = ComputeStd(sum_y, sum_y_sq, count);
      }
      bins.push_back(stats);
    }
    return bins;
  };

  result.all_points_bins = processBins(
      bin_angles_all, bin_residuals_all, bin_angular_residuals_all,
      bin_residual_x_all, bin_residual_y_all,
      result.angular_equiv_focal_px, "all");
  result.outer_only_bins = processBins(
      bin_angles_outer, bin_residuals_outer, bin_angular_residuals_outer,
      bin_residual_x_outer, bin_residual_y_outer,
      result.angular_equiv_focal_px, "outer");
  result.internal_only_bins =
      processBins(bin_angles_internal, bin_residuals_internal,
                  bin_angular_residuals_internal, bin_residual_x_internal,
                  bin_residual_y_internal, result.angular_equiv_focal_px,
                  "internal");

  for (const auto& entry : board_bin_angles) {
    result.per_board_bins[entry.first] = processBins(
        entry.second, board_bin_residuals[entry.first],
        board_bin_angular_residuals[entry.first],
        board_bin_residual_x[entry.first], board_bin_residual_y[entry.first],
        result.angular_equiv_focal_px, "all");
  }

  for (const auto& entry : frame_bin_angles) {
    result.per_frame_bins[entry.first] = processBins(
        entry.second, frame_bin_residuals[entry.first],
        frame_bin_angular_residuals[entry.first],
        frame_bin_residual_x[entry.first], frame_bin_residual_y[entry.first],
        result.angular_equiv_focal_px, "all");
  }

  result.success = true;

  if (!output_dir.empty()) {
    try {
      const std::string summary_path = output_dir + "/polar_angle_residual_summary.txt";
      const std::string csv_path = output_dir + "/polar_angle_residual_bins.csv";
      WriteSummaryFile(summary_path, result);
      WriteCsvFile(csv_path, result);
    } catch (const std::exception& e) {
      result.warnings.push_back(
          std::string("Failed to write output files: ") + e.what());
    }
  }

  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
