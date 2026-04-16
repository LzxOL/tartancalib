#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_ITERATIVE_COARSE_CALIBRATION_EXPERIMENT_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_ITERATIVE_COARSE_CALIBRATION_EXPERIMENT_HPP

#include <string>
#include <vector>

#include <aslam/cameras/apriltag_internal/ApriltagCanonicalModel.hpp>
#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct IterativeCoarseCalibrationExperimentOptions {
  std::vector<int> board_ids;
  std::vector<std::string> group_filters;
  int max_iterations = 3;
  int pose_refinement_rounds = 2;
  bool use_internal_points_for_update = false;
  double internal_point_quality_threshold = 0.45;
  double convergence_threshold = 0.05;
  double init_xi = -0.2;
  double init_alpha = 0.6;
  double init_fu_scale = 0.55;
  double init_fv_scale = 0.55;
  double init_cu_offset = 0.0;
  double init_cv_offset = 0.0;
};

struct IterativeCoarseCalibrationExperimentRequest {
  std::string image_dir;
  std::string output_dir;
  ApriltagInternalConfig base_config;
  ApriltagInternalDetectionOptions detection_options;
  IterativeCoarseCalibrationExperimentOptions experiment_options;
};

class IterativeCoarseCalibrationExperiment {
 public:
  explicit IterativeCoarseCalibrationExperiment(
      IterativeCoarseCalibrationExperimentRequest request);

  void Run() const;

 private:
  IterativeCoarseCalibrationExperimentRequest request_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_ITERATIVE_COARSE_CALIBRATION_EXPERIMENT_HPP
