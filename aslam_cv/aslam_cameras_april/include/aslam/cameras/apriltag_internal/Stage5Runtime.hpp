#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_RUNTIME_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_RUNTIME_HPP

#include <string>
#include <vector>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

enum class Stage5RuntimeMode {
  Research,
  Fast,
};

const char* ToString(Stage5RuntimeMode mode);
Stage5RuntimeMode ParseStage5RuntimeMode(const std::string& value);

struct Stage5RuntimeStageRecord {
  std::string stage_label;
  double wall_time_seconds = 0.0;
  bool skipped_in_fast_mode = false;
};

struct Stage5RuntimeSummary {
  Stage5RuntimeMode runtime_mode = Stage5RuntimeMode::Research;
  std::string cache_dir;
  bool cache_enabled = false;
  int training_detection_cache_hits = 0;
  int training_detection_cache_misses = 0;
  int holdout_detection_cache_hits = 0;
  int holdout_detection_cache_misses = 0;
  int round1_regeneration_attempted_internal_corners = 0;
  int round1_regeneration_valid_internal_corners = 0;
  int round2_regeneration_attempted_internal_corners = 0;
  int round2_regeneration_valid_internal_corners = 0;
  int round1_optimization_residual_evaluation_call_count = 0;
  int round1_optimization_cost_evaluation_call_count = 0;
  int round2_optimization_residual_evaluation_call_count = 0;
  int round2_optimization_cost_evaluation_call_count = 0;
  std::vector<Stage5RuntimeStageRecord> stage_records;
  double total_runtime_seconds = 0.0;
  std::vector<std::string> warnings;
};

void WriteStage5RuntimeSummary(const std::string& path,
                               const Stage5RuntimeSummary& summary);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_RUNTIME_HPP
