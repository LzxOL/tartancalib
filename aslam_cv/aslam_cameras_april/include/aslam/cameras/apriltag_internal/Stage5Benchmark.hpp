#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_BENCHMARK_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_BENCHMARK_HPP

#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>
#include <aslam/cameras/apriltag_internal/KalibrBenchmark.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct CalibrationBenchmarkSplitOptions {
  std::string mode = "deterministic_stride";
  int holdout_stride = 5;
  int holdout_offset = 0;
  int minimum_training_frames = 3;
  int minimum_holdout_frames = 1;
};

struct CalibrationBenchmarkSplit {
  bool success = false;
  std::string mode = "deterministic_stride";
  int holdout_stride = 5;
  int holdout_offset = 0;
  std::string split_signature;
  std::vector<FrozenRound2BaselineFrameSource> training_frames;
  std::vector<FrozenRound2BaselineFrameSource> holdout_frames;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct CalibrationEvaluationPointObservation {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  JointPointType point_type = JointPointType::Outer;
  Eigen::Vector2d image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector3d target_xyz_board = Eigen::Vector3d::Zero();
  double quality = 0.0;
  int frame_storage_index = -1;
  int source_board_observation_index = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind = JointObservationSourceKind::OuterMeasurement;
};

struct CalibrationEvaluationBoardObservation {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  std::vector<CalibrationEvaluationPointObservation> points;
  int outer_point_count = 0;
  int internal_point_count = 0;
  bool has_pose_fit_outer_points = false;
};

struct CalibrationEvaluationFrameInput {
  int frame_index = -1;
  std::string frame_label;
  std::vector<int> visible_board_ids;
  std::vector<CalibrationEvaluationBoardObservation> board_observations;
};

struct CalibrationEvaluationDataset {
  bool success = false;
  std::string dataset_label;
  std::string split_label;
  std::string split_signature;
  std::vector<CalibrationEvaluationFrameInput> frames;
  int frame_count = 0;
  int board_observation_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  int total_point_count = 0;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct CameraModelRefitPointDiagnostics {
  std::string method_label;
  std::string split_label;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_id = -1;
  JointPointType point_type = JointPointType::Outer;
  Eigen::Vector2d observed_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector2d predicted_image_xy = Eigen::Vector2d::Zero();
  Eigen::Vector3d target_xyz_board = Eigen::Vector3d::Zero();
  Eigen::Vector2d residual_xy = Eigen::Vector2d::Zero();
  double residual_norm = 0.0;
  double quality = 0.0;
  int frame_storage_index = -1;
  int source_board_observation_index = -1;
  int source_point_index = -1;
  JointObservationSourceKind source_kind = JointObservationSourceKind::OuterMeasurement;
};

struct CameraModelRefitBoardObservationDiagnostics {
  std::string method_label;
  std::string split_label;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double pose_fit_outer_rmse = 0.0;
  double evaluation_rmse = 0.0;
};

struct CameraModelRefitFrameDiagnostics {
  std::string method_label;
  std::string split_label;
  int frame_index = -1;
  std::string frame_label;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double rmse = 0.0;
};

struct CameraModelRefitEvaluationResult {
  bool success = false;
  std::string method_label;
  std::string split_label;
  std::string split_signature;
  OuterBootstrapCameraIntrinsics camera;
  int evaluated_frame_count = 0;
  int evaluated_board_observation_count = 0;
  int point_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  double overall_rmse = 0.0;
  double outer_only_rmse = 0.0;
  double internal_only_rmse = 0.0;
  std::vector<CameraModelRefitPointDiagnostics> point_diagnostics;
  std::vector<CameraModelRefitBoardObservationDiagnostics> board_observation_diagnostics;
  std::vector<CameraModelRefitFrameDiagnostics> frame_diagnostics;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

struct KalibrBenchmarkReference {
  std::string camchain_yaml;
  std::string camera_model_family = "ds";
  std::string training_split_signature;
  double runtime_seconds = -1.0;
  std::string source_label;
};

struct Stage5BenchmarkInput {
  std::vector<FrozenRound2BaselineFrameSource> all_frames;
  FrozenRound2BaselineOptions baseline_options;
  BackendProblemOptions backend_options;
  KalibrBenchmarkReference kalibr_reference;
  std::string dataset_label;
};

struct Stage5BenchmarkReport {
  bool success = false;
  bool fair_protocol_matched = false;
  bool diagnostic_only = false;
  std::string dataset_label;
  std::string baseline_protocol_label;
  std::string split_signature;
  CalibrationBenchmarkSplit split;
  FrozenRound2BaselineResult baseline_result;
  CalibrationBackendProblemInput backend_problem_input;
  CalibrationEvaluationDataset training_dataset;
  CalibrationEvaluationDataset holdout_dataset;
  CameraModelRefitEvaluationResult our_training_evaluation;
  CameraModelRefitEvaluationResult kalibr_training_evaluation;
  CameraModelRefitEvaluationResult our_holdout_evaluation;
  CameraModelRefitEvaluationResult kalibr_holdout_evaluation;
  KalibrBenchmarkReference kalibr_reference;
  KalibrBenchmarkReport diagnostic_compare;
  std::vector<std::string> warnings;
  std::string failure_reason;
};

class Stage5Benchmark {
 public:
  explicit Stage5Benchmark(
      CalibrationBenchmarkSplitOptions split_options =
          CalibrationBenchmarkSplitOptions{});

  CalibrationBenchmarkSplit BuildDeterministicSplit(
      const std::vector<FrozenRound2BaselineFrameSource>& frames) const;

  Stage5BenchmarkReport Run(const Stage5BenchmarkInput& input) const;

  cv::Mat RenderProjectionComparison(const Stage5BenchmarkReport& report,
                                     int max_width = 900,
                                     int max_height = 900) const;
  cv::Mat RenderEvaluationFrameOverlay(
      const Stage5BenchmarkReport& report,
      const CameraModelRefitEvaluationResult& evaluation,
      int frame_index) const;
  cv::Mat RenderEvaluationBoardObservationOverlay(
      const Stage5BenchmarkReport& report,
      const CameraModelRefitEvaluationResult& evaluation,
      int frame_index,
      int board_id) const;
  cv::Mat RenderOuterPoseFitFrameOverlay(
      const Stage5BenchmarkReport& report,
      const CameraModelRefitEvaluationResult& evaluation,
      int frame_index) const;
  cv::Mat RenderOuterPoseFitBoardOverlay(
      const Stage5BenchmarkReport& report,
      const CameraModelRefitEvaluationResult& evaluation,
      int frame_index,
      int board_id) const;

  const CalibrationBenchmarkSplitOptions& split_options() const {
    return split_options_;
  }

 private:
  CalibrationEvaluationDataset BuildTrainingEvaluationDataset(
      const CalibrationStateBundle& bundle) const;
  CalibrationEvaluationDataset BuildHoldoutEvaluationDataset(
      const std::vector<FrozenRound2BaselineFrameSource>& holdout_frames,
      const FrozenRound2BaselineOptions& baseline_options,
      const JointReprojectionSceneState& optimized_scene_state,
      const std::string& split_signature) const;
  CameraModelRefitEvaluationResult EvaluateCameraModel(
      const CalibrationEvaluationDataset& dataset,
      const OuterBootstrapCameraIntrinsics& camera,
      const std::string& method_label) const;
  std::string FindFrameImagePath(const Stage5BenchmarkReport& report,
                                 int frame_index) const;

  CalibrationBenchmarkSplitOptions split_options_;
};

void WriteStage5BenchmarkProtocolSummary(const std::string& path,
                                         const Stage5BenchmarkReport& report);
void WriteStage5BenchmarkTrainingSummary(const std::string& path,
                                         const Stage5BenchmarkReport& report);
void WriteStage5BenchmarkHoldoutSummary(const std::string& path,
                                        const Stage5BenchmarkReport& report);
void WriteStage5BenchmarkHoldoutPointsCsv(const std::string& path,
                                          const Stage5BenchmarkReport& report);
void WriteStage5BenchmarkWorstCasesSummary(const std::string& path,
                                           const Stage5BenchmarkReport& report,
                                           int top_k = 10);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_BENCHMARK_HPP
