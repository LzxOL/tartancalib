#include <aslam/cameras/apriltag_internal/Stage5Runtime.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

const char* ToString(Stage5RuntimeMode mode) {
  switch (mode) {
    case Stage5RuntimeMode::Research:
      return "research";
    case Stage5RuntimeMode::Fast:
      return "fast";
  }
  return "unknown";
}

Stage5RuntimeMode ParseStage5RuntimeMode(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (lowered == "research") {
    return Stage5RuntimeMode::Research;
  }
  if (lowered == "fast") {
    return Stage5RuntimeMode::Fast;
  }
  throw std::runtime_error("Unsupported runtime mode: " + value);
}

void WriteStage5RuntimeSummary(const std::string& path,
                               const Stage5RuntimeSummary& summary) {
  std::ofstream output(path.c_str());
  output << "runtime_mode: " << ToString(summary.runtime_mode) << "\n";
  output << "cache_dir: " << summary.cache_dir << "\n";
  output << "cache_enabled: " << (summary.cache_enabled ? 1 : 0) << "\n";
  output << "training_detection_cache_hits: "
         << summary.training_detection_cache_hits << "\n";
  output << "training_detection_cache_misses: "
         << summary.training_detection_cache_misses << "\n";
  output << "holdout_detection_cache_hits: "
         << summary.holdout_detection_cache_hits << "\n";
  output << "holdout_detection_cache_misses: "
         << summary.holdout_detection_cache_misses << "\n";
  output << "round1_regeneration_attempted_internal_corners: "
         << summary.round1_regeneration_attempted_internal_corners << "\n";
  output << "round1_regeneration_valid_internal_corners: "
         << summary.round1_regeneration_valid_internal_corners << "\n";
  output << "round2_regeneration_attempted_internal_corners: "
         << summary.round2_regeneration_attempted_internal_corners << "\n";
  output << "round2_regeneration_valid_internal_corners: "
         << summary.round2_regeneration_valid_internal_corners << "\n";
  output << "round1_optimization_residual_evaluation_call_count: "
         << summary.round1_optimization_residual_evaluation_call_count << "\n";
  output << "round1_optimization_cost_evaluation_call_count: "
         << summary.round1_optimization_cost_evaluation_call_count << "\n";
  output << "round2_optimization_residual_evaluation_call_count: "
         << summary.round2_optimization_residual_evaluation_call_count << "\n";
  output << "round2_optimization_cost_evaluation_call_count: "
         << summary.round2_optimization_cost_evaluation_call_count << "\n";
  for (const Stage5RuntimeStageRecord& stage : summary.stage_records) {
    output << "stage_label: " << stage.stage_label << "\n";
    output << "stage_wall_time_seconds: " << stage.wall_time_seconds << "\n";
    output << "stage_skipped_in_fast_mode: "
           << (stage.skipped_in_fast_mode ? 1 : 0) << "\n";
  }
  output << "total_runtime_seconds: " << summary.total_runtime_seconds << "\n";
  for (const std::string& warning : summary.warnings) {
    output << "warning: " << warning << "\n";
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
