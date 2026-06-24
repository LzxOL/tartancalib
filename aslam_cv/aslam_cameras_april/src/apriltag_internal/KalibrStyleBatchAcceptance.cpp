#include <aslam/cameras/apriltag_internal/KalibrStyleBatchAcceptance.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

const char* ToString(KalibrStyleBatchAcceptancePolicy policy) {
  switch (policy) {
    case KalibrStyleBatchAcceptancePolicy::ResidualScore:
      return "residual_score";
    case KalibrStyleBatchAcceptancePolicy::KalibrInformationGain:
      return "kalibr_information_gain";
  }
  return "residual_score";
}

KalibrStyleBatchAcceptancePolicy ParseKalibrStyleBatchAcceptancePolicy(
    const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (lowered == "residual_score" || lowered == "residual-score" ||
      lowered == "residual") {
    return KalibrStyleBatchAcceptancePolicy::ResidualScore;
  }
  if (lowered == "kalibr_information_gain" ||
      lowered == "kalibr-information-gain" ||
      lowered == "information_gain" || lowered == "information-gain" ||
      lowered == "mi" || lowered == "marginal_information_gain") {
    return KalibrStyleBatchAcceptancePolicy::KalibrInformationGain;
  }
  throw std::runtime_error("Unsupported Kalibr-style batch acceptance policy: " +
                           value);
}

KalibrStyleBatchAcceptanceDecision EvaluateKalibrStyleBatchAcceptance(
    const KalibrStyleBatchAcceptanceOptions& options,
    const KalibrStyleBatchAcceptanceInput& input) {
  KalibrStyleBatchAcceptanceDecision decision;
  if (!input.hard_validity_pass) {
    decision.reason = "hard_validity_gate";
    return decision;
  }
  if (input.catastrophic_residual) {
    decision.reason = "batch_catastrophic_residual_gate";
    return decision;
  }

  if (input.companion_completion &&
      input.residual_overage_penalty <=
          options.critical_view_max_residual_overage) {
    decision.accepted = true;
    decision.accepted_by_companion_completion = true;
    decision.reason = "companion_completion";
    return decision;
  }

  if (options.policy == KalibrStyleBatchAcceptancePolicy::ResidualScore) {
    if (input.information_gain_proxy >= 1.0 &&
        input.residual_score >= 0.5) {
      decision.accepted = true;
      decision.reason = "batch_acceptance_score";
      return decision;
    }
    decision.reason = "batch_acceptance_score_gate";
    return decision;
  }

  const bool finite_information =
      std::isfinite(input.information_gain_proxy) &&
      std::isfinite(input.rank_gain_proxy);
  if (finite_information &&
      input.information_gain_proxy > options.information_gain_threshold) {
    decision.accepted = true;
    decision.accepted_by_information_gain = true;
    decision.reason = "marginal_information_gain";
    return decision;
  }
  if (finite_information &&
      input.rank_gain_proxy > options.rank_gain_threshold) {
    decision.accepted = true;
    decision.accepted_by_rank_gain = true;
    decision.reason = "rank_proxy_increase";
    return decision;
  }
  if (options.protect_critical_views && input.critical_view &&
      input.residual_overage_penalty <=
          options.critical_view_max_residual_overage) {
    decision.accepted = true;
    decision.accepted_by_critical_view_protection = true;
    decision.reason = "critical_view_protection";
    return decision;
  }

  decision.reason = "marginal_information_gain_gate";
  return decision;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
