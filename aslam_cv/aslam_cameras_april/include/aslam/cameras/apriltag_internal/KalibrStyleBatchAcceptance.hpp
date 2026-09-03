#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_KALIBR_STYLE_BATCH_ACCEPTANCE_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_KALIBR_STYLE_BATCH_ACCEPTANCE_HPP

#include <string>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

enum class KalibrStyleBatchAcceptancePolicy {
  ResidualScore = 0,
  KalibrInformationGain = 1,
};

struct KalibrStyleBatchAcceptanceOptions {
  KalibrStyleBatchAcceptancePolicy policy =
      KalibrStyleBatchAcceptancePolicy::ResidualScore;
  double information_gain_threshold = 0.2;
  double rank_gain_threshold = 1e-6;
  bool protect_critical_views = true;
  double critical_view_max_residual_overage = 2.0;
};

struct KalibrStyleBatchAcceptanceInput {
  bool hard_validity_pass = false;
  bool catastrophic_residual = false;
  bool companion_completion = false;
  bool critical_view = false;
  double information_gain_proxy = 0.0;
  double rank_gain_proxy = 0.0;
  double residual_score = 0.0;
  double residual_overage_penalty = 0.0;
};

struct KalibrStyleBatchAcceptanceDecision {
  bool accepted = false;
  bool accepted_by_information_gain = false;
  bool accepted_by_rank_gain = false;
  bool accepted_by_companion_completion = false;
  bool accepted_by_critical_view_protection = false;
  std::string reason;
};

const char* ToString(KalibrStyleBatchAcceptancePolicy policy);
KalibrStyleBatchAcceptancePolicy ParseKalibrStyleBatchAcceptancePolicy(
    const std::string& value);

KalibrStyleBatchAcceptanceDecision EvaluateKalibrStyleBatchAcceptance(
    const KalibrStyleBatchAcceptanceOptions& options,
    const KalibrStyleBatchAcceptanceInput& input);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_KALIBR_STYLE_BATCH_ACCEPTANCE_HPP
