#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_KALIBR_BENCHMARK_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_KALIBR_BENCHMARK_HPP

#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct KalibrIntrinsicsDelta {
  double xi = 0.0;
  double alpha = 0.0;
  double fu = 0.0;
  double fv = 0.0;
  double cu = 0.0;
  double cv = 0.0;
  double l2_norm = 0.0;
};

struct KalibrResidualSummary {
  double our_overall_rmse = 0.0;
  double our_outer_only_rmse = 0.0;
  double our_internal_only_rmse = 0.0;
  bool kalibr_residual_available = false;
  std::string note;
};

struct KalibrProjectionComparisonSample {
  Eigen::Vector2d anchor_pixel = Eigen::Vector2d::Zero();
  Eigen::Vector2d our_projected_pixel = Eigen::Vector2d::Zero();
  Eigen::Vector2d kalibr_projected_pixel = Eigen::Vector2d::Zero();
  double delta_pixels = 0.0;
  bool valid = false;
};

struct KalibrProjectionComparison {
  cv::Size resolution;
  int valid_sample_count = 0;
  double mean_pixel_delta = 0.0;
  double max_pixel_delta = 0.0;
  std::vector<KalibrProjectionComparisonSample> samples;
};

struct KalibrBenchmarkInput {
  std::string dataset_label;
  std::string kalibr_camchain_yaml;
  CalibrationStateBundle our_bundle;
  std::string camera_model = "ds";
};

struct KalibrBenchmarkReport {
  bool success = false;
  std::string dataset_label;
  int our_reference_board_id = 1;
  OuterBootstrapCameraIntrinsics our_intrinsics;
  OuterBootstrapCameraIntrinsics kalibr_intrinsics;
  KalibrIntrinsicsDelta intrinsics_delta;
  KalibrResidualSummary residual_summary;
  KalibrProjectionComparison distortion_or_projection_compare;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class KalibrBenchmark {
 public:
  KalibrBenchmark() = default;

  KalibrBenchmarkReport Compare(const KalibrBenchmarkInput& input) const;

  cv::Mat RenderProjectionComparison(const KalibrBenchmarkReport& report,
                                     int max_width = 900,
                                     int max_height = 900) const;
};

void WriteKalibrBenchmarkSummary(const std::string& path,
                                 const KalibrBenchmarkReport& report);
void WriteKalibrBenchmarkIntrinsicsCsv(const std::string& path,
                                       const KalibrBenchmarkReport& report);
void WriteKalibrBenchmarkResidualSummary(const std::string& path,
                                         const KalibrBenchmarkReport& report);
bool LoadKalibrCamchainIntrinsics(const std::string& yaml_path,
                                  OuterBootstrapCameraIntrinsics* intrinsics,
                                  std::string* error_message);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_KALIBR_BENCHMARK_HPP
