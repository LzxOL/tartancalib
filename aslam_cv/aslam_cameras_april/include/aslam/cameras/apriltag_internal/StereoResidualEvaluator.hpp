#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STEREO_RESIDUAL_EVALUATOR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STEREO_RESIDUAL_EVALUATOR_HPP

#include <map>
#include <set>

#include <aslam/cameras/apriltag_internal/StereoExtrinsicProblemInput.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct StereoResidualEvaluationOptions {
  bool refit_pair_pose = false;
  StereoPairPoseRefitMode pair_pose_refit_mode =
      StereoPairPoseRefitMode::StereoSymmetric;
  int symmetric_refit_max_iterations = 8;
  double symmetric_refit_step = 1e-3;
  bool extrinsic_only_local_board_pose = false;
  bool use_committed_pair_board_pose = false;
};

class StereoResidualEvaluator {
 public:
  explicit StereoResidualEvaluator(
      StereoResidualEvaluationOptions options = StereoResidualEvaluationOptions{});

  StereoResidualSummary Evaluate(
      const StereoMeasurementDataset& dataset,
      const StereoSceneState& scene_state,
      const std::set<int>& pair_indices,
      const std::string& split_label) const;

 private:
  StereoResidualEvaluationOptions options_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_STEREO_RESIDUAL_EVALUATOR_HPP
