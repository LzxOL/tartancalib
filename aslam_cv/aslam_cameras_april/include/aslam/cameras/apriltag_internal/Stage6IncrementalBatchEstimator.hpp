#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE6_INCREMENTAL_BATCH_ESTIMATOR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE6_INCREMENTAL_BATCH_ESTIMATOR_HPP

#include <set>
#include <string>
#include <utility>

#include <Eigen/Core>

#include <aslam/cameras/apriltag_internal/StereoExtrinsicProblemInput.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct Stage6IncrementalBatchEstimatorOptions {
  bool enabled = false;
  double info_gain_threshold = 0.2;
  double rank_threshold = 1e-6;
  Stage6IncrementalInfoBlock info_block =
      Stage6IncrementalInfoBlock::StereoExtrinsic;
  double max_baseline_rotation_delta_deg = 2.0;
  double max_baseline_translation_delta_m = 0.05;
};

struct Stage6IncrementalBatchCandidate {
  std::string batch_type;
  int pair_index = -1;
  std::set<std::pair<int, int> > selected_pair_boards_before;
  std::set<std::pair<int, int> > selected_pair_boards_after;
  bool force = false;
};

struct Stage6IncrementalBatchResult {
  bool batchAccepted = false;
  bool solution_valid = false;
  bool optimization_success = false;
  bool residual_finite = false;
  bool objective_finite = false;
  bool objective_decreased = false;
  bool no_solver_failure = false;
  bool baseline_stable = false;
  bool rank_proxy_increases = false;
  bool force = false;
  int num_iterations = 0;
  double objective_before = 0.0;
  double objective_after = 0.0;
  double rmse_before = 0.0;
  double rmse_after = 0.0;
  double total_rmse_delta = 0.0;
  double cam0_rmse_delta = 0.0;
  double cam1_rmse_delta = 0.0;
  double baseline_rotation_delta_deg = 0.0;
  double baseline_translation_delta_m = 0.0;
  double marginal_information_gain_proxy = 0.0;
  int rank_before = 0;
  int rank_after = 0;
  double info_gain_threshold = 0.0;
  double rank_threshold = 0.0;
  std::string info_block;
  std::string accept_reason;
  std::string reject_reason;
  std::string committed_or_rollback;
};

class Stage6IncrementalBatchEstimator {
 public:
  explicit Stage6IncrementalBatchEstimator(
      Stage6IncrementalBatchEstimatorOptions options);

  Stage6IncrementalBatchResult AddBatch(
      const StereoMeasurementDataset& dataset,
      const StereoSceneState& scene_before,
      const StereoSceneState& scene_after,
      const StereoResidualSummary& residual_before,
      const StereoResidualSummary& residual_after,
      const StereoGlobalSparseBaSummary& optimization_summary,
      const Stage6IncrementalBatchCandidate& candidate) const;

  Stage6IncrementalBatchResult EvaluateInformationGainOnly(
      const StereoMeasurementDataset& dataset,
      const StereoSceneState& scene,
      const Stage6IncrementalBatchCandidate& candidate) const;

  double ComputeMarginalInformationGainProxy(
      const StereoMeasurementDataset& dataset,
      const StereoSceneState& scene,
      const std::set<std::pair<int, int> >& selected_pair_boards,
      int* rank) const;

  const Stage6IncrementalBatchEstimatorOptions& options() const {
    return options_;
  }

 private:
  Stage6IncrementalBatchEstimatorOptions options_;
};

const char* ToString(Stage6IncrementalInfoBlock block);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE6_INCREMENTAL_BATCH_ESTIMATOR_HPP
