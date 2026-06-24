#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_COOBSERVATION_CONSISTENCY_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_COOBSERVATION_CONSISTENCY_HPP

#include <string>
#include <vector>

#include <aslam/cameras/apriltag_internal/StereoExtrinsicProblemInput.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct MultiBoardCoObservationOptions {
  bool enabled = false;
  std::string output_dir;
  int min_corners_per_group = 12;
  double high_polar_threshold_deg = 50.0;
  double very_high_polar_threshold_deg = 70.0;
  bool enable_rescue_suggestions = true;
  double score_alpha_high_polar = 1.0;
  double score_beta_multiboard = 2.0;
  double score_gamma_balance = 1.0;
  double score_eta_conflict = 1.0;
  double rescue_min_high_polar_score = 8.0;
  double rescue_bad_conflict_threshold = 5.0;
};

struct MultiBoardCoObservationSummary {
  bool success = false;
  std::string failure_reason;
  int total_frames_processed = 0;
  int frames_with_at_least_two_boards = 0;
  int frames_with_bicam_board_observations = 0;
  int high_polar_rescue_candidate_count = 0;
  double median_pose_rotation_deg = 0.0;
  double median_layout_rotation_deg = 0.0;
  double median_stereo_rotation_deg = 0.0;
  double median_pose_translation_m = 0.0;
  double median_layout_translation_m = 0.0;
  double median_stereo_translation_m = 0.0;
  std::vector<std::string> warnings;
};

class MultiBoardCoObservationConsistency {
 public:
  explicit MultiBoardCoObservationConsistency(
      MultiBoardCoObservationOptions options);

  MultiBoardCoObservationSummary Evaluate(
      const StereoExtrinsicCalibrationResult& result) const;

 private:
  MultiBoardCoObservationOptions options_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_MULTI_BOARD_COOBSERVATION_CONSISTENCY_HPP
