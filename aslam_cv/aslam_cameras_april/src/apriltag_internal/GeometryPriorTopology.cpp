#include <aslam/cameras/apriltag_internal/GeometryPriorTopology.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

double Cross2d(const cv::Point2f& a,
               const cv::Point2f& b,
               const cv::Point2f& c) {
  return static_cast<double>(b.x - a.x) * static_cast<double>(c.y - a.y) -
         static_cast<double>(b.y - a.y) * static_cast<double>(c.x - a.x);
}

bool ProperSegmentsIntersect(const cv::Point2f& a,
                             const cv::Point2f& b,
                             const cv::Point2f& c,
                             const cv::Point2f& d) {
  constexpr double kCrossEpsilon = 1e-6;
  const double ab_c = Cross2d(a, b, c);
  const double ab_d = Cross2d(a, b, d);
  const double cd_a = Cross2d(c, d, a);
  const double cd_b = Cross2d(c, d, b);
  return ((ab_c > kCrossEpsilon && ab_d < -kCrossEpsilon) ||
          (ab_c < -kCrossEpsilon && ab_d > kCrossEpsilon)) &&
         ((cd_a > kCrossEpsilon && cd_b < -kCrossEpsilon) ||
          (cd_a < -kCrossEpsilon && cd_b > kCrossEpsilon));
}

std::array<cv::Point2f, 4> ToCvCorners(
    const std::array<Eigen::Vector2d, 4>& corners) {
  std::array<cv::Point2f, 4> converted{};
  for (int index = 0; index < 4; ++index) {
    const Eigen::Vector2d& corner = corners[static_cast<std::size_t>(index)];
    converted[static_cast<std::size_t>(index)] =
        cv::Point2f(static_cast<float>(corner.x()),
                    static_cast<float>(corner.y()));
  }
  return converted;
}

}  // namespace

double SignedPolygonArea(const std::array<cv::Point2f, 4>& corners) {
  double area = 0.0;
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f& a = corners[static_cast<std::size_t>(index)];
    const cv::Point2f& b = corners[static_cast<std::size_t>((index + 1) % 4)];
    area += static_cast<double>(a.x) * static_cast<double>(b.y) -
            static_cast<double>(b.x) * static_cast<double>(a.y);
  }
  return area * 0.5;
}

double PolygonArea(const std::array<cv::Point2f, 4>& corners) {
  return std::fabs(SignedPolygonArea(corners));
}

QuadTopologyCheck CheckQuadTopology(
    const std::array<cv::Point2f, 4>& corners) {
  QuadTopologyCheck check;
  check.finite = std::all_of(
      corners.begin(), corners.end(), [](const cv::Point2f& point) {
        return std::isfinite(point.x) && std::isfinite(point.y);
      });
  if (!check.finite) {
    check.summary = "nonfinite_corner";
    return check;
  }

  check.signed_area_px = SignedPolygonArea(corners);
  check.area_px = std::fabs(check.signed_area_px);
  check.min_edge_length_px = std::numeric_limits<double>::infinity();
  int positive_turn_count = 0;
  int negative_turn_count = 0;
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f edge =
        corners[static_cast<std::size_t>((index + 1) % 4)] -
        corners[static_cast<std::size_t>(index)];
    check.min_edge_length_px = std::min(
        check.min_edge_length_px,
        std::hypot(static_cast<double>(edge.x), static_cast<double>(edge.y)));
    const double turn = Cross2d(
        corners[static_cast<std::size_t>(index)],
        corners[static_cast<std::size_t>((index + 1) % 4)],
        corners[static_cast<std::size_t>((index + 2) % 4)]);
    positive_turn_count += turn > 1e-6 ? 1 : 0;
    negative_turn_count += turn < -1e-6 ? 1 : 0;
  }
  check.convex = positive_turn_count == 4 || negative_turn_count == 4;
  check.self_intersecting =
      ProperSegmentsIntersect(corners[0], corners[1], corners[2], corners[3]) ||
      ProperSegmentsIntersect(corners[1], corners[2], corners[3], corners[0]);
  check.valid = check.area_px >= 4.0 &&
                check.min_edge_length_px >= 2.0 && check.convex &&
                !check.self_intersecting;
  std::ostringstream summary;
  summary << "finite=" << (check.finite ? 1 : 0)
          << " convex=" << (check.convex ? 1 : 0)
          << " self_intersecting=" << (check.self_intersecting ? 1 : 0)
          << " signed_area=" << check.signed_area_px
          << " min_edge=" << check.min_edge_length_px;
  check.summary = summary.str();
  return check;
}

WrongIdTopologyAssociation AssociateWrongIdProposalToTopology(
    const std::array<Eigen::Vector2d, 4>& topology_predicted_corners,
    const OuterWrongIdProposal& proposal) {
  WrongIdTopologyAssociation association;
  const std::array<cv::Point2f, 4> predicted_cv =
      ToCvCorners(topology_predicted_corners);
  const std::array<cv::Point2f, 4> proposal_cv =
      ToCvCorners(proposal.corners_original_image);
  const QuadTopologyCheck predicted_topology = CheckQuadTopology(predicted_cv);
  const QuadTopologyCheck proposal_topology = CheckQuadTopology(proposal_cv);
  if (!predicted_topology.valid || !proposal_topology.valid) {
    association.summary = "topology_invalid_quad predicted{" +
                          predicted_topology.summary + "} proposal{" +
                          proposal_topology.summary + "}";
    return association;
  }

  const double predicted_scale =
      std::sqrt(std::max(1.0, predicted_topology.area_px));
  Eigen::Vector2d predicted_center = Eigen::Vector2d::Zero();
  Eigen::Vector2d proposal_center = Eigen::Vector2d::Zero();
  for (int index = 0; index < 4; ++index) {
    predicted_center += topology_predicted_corners[static_cast<std::size_t>(index)];
    proposal_center += proposal.corners_original_image[static_cast<std::size_t>(index)];
  }
  predicted_center *= 0.25;
  proposal_center *= 0.25;
  association.normalized_center_error =
      (predicted_center - proposal_center).norm() / predicted_scale;
  association.area_ratio = proposal_topology.area_px / predicted_topology.area_px;

  double best_squared_error = std::numeric_limits<double>::infinity();
  for (int reflected = 0; reflected <= 1; ++reflected) {
    for (int shift = 0; shift < 4; ++shift) {
      std::array<Eigen::Vector2d, 4> ordered{};
      double squared_error = 0.0;
      for (int index = 0; index < 4; ++index) {
        const int proposal_index =
            reflected == 0 ? (shift + index) % 4 : (shift - index + 4) % 4;
        ordered[static_cast<std::size_t>(index)] =
            proposal.corners_original_image[static_cast<std::size_t>(proposal_index)];
        squared_error +=
            (topology_predicted_corners[static_cast<std::size_t>(index)] -
             ordered[static_cast<std::size_t>(index)]).squaredNorm();
      }
      if (squared_error < best_squared_error) {
        best_squared_error = squared_error;
        association.ordered_corners = ordered;
        association.cyclic_shift = shift;
        association.reflected_order = reflected != 0;
      }
    }
  }
  association.normalized_corner_rmse =
      std::sqrt(best_squared_error / 4.0) / predicted_scale;

  constexpr double kMaxNormalizedCornerRmse = 0.18;
  constexpr double kMaxNormalizedCenterError = 0.15;
  constexpr double kMinAreaRatio = 0.55;
  constexpr double kMaxAreaRatio = 1.80;
  association.compatible =
      association.normalized_corner_rmse <= kMaxNormalizedCornerRmse &&
      association.normalized_center_error <= kMaxNormalizedCenterError &&
      association.area_ratio >= kMinAreaRatio &&
      association.area_ratio <= kMaxAreaRatio;
  std::ostringstream stream;
  stream << "topology_assoc compatible=" << (association.compatible ? 1 : 0)
         << " corner_rmse_norm=" << association.normalized_corner_rmse
         << " center_norm=" << association.normalized_center_error
         << " area_ratio=" << association.area_ratio
         << " cyclic_shift=" << association.cyclic_shift
         << " reflected=" << (association.reflected_order ? 1 : 0)
         << " proposal_id=" << proposal.detected_tag_id
         << " proposal_hamming=" << proposal.hamming
         << " proposal_source=" << proposal.source;
  association.summary = stream.str();
  return association;
}

TopologySlotAssignment AssignObservedQuadToTopologySlots(
    const std::vector<std::array<Eigen::Vector2d, 4>>& topology_slots,
    const OuterWrongIdProposal& proposal) {
  TopologySlotAssignment assignment;
  assignment.checked = true;
  if (topology_slots.empty()) {
    assignment.summary = "topology_slot_assignment no_slots";
    return assignment;
  }

  struct ScoredSlot {
    int index = -1;
    double cost = std::numeric_limits<double>::infinity();
    WrongIdTopologyAssociation association;
  };
  std::vector<ScoredSlot> scored_slots;
  scored_slots.reserve(topology_slots.size());
  for (std::size_t index = 0; index < topology_slots.size(); ++index) {
    ScoredSlot scored;
    scored.index = static_cast<int>(index);
    scored.association =
        AssociateWrongIdProposalToTopology(topology_slots[index], proposal);
    if (scored.association.compatible) {
      // Corner agreement is the main term. Center and area terms prevent a
      // similarly shaped neighboring board from winning solely through one
      // favorable corner ordering.
      scored.cost = scored.association.normalized_corner_rmse +
                    0.5 * scored.association.normalized_center_error +
                    0.25 * std::fabs(std::log(scored.association.area_ratio));
    }
    scored_slots.push_back(std::move(scored));
  }
  std::sort(scored_slots.begin(), scored_slots.end(),
            [](const ScoredSlot& lhs, const ScoredSlot& rhs) {
              if (lhs.cost != rhs.cost) {
                return lhs.cost < rhs.cost;
              }
              return lhs.index < rhs.index;
            });

  const ScoredSlot& best = scored_slots.front();
  assignment.compatible = std::isfinite(best.cost);
  if (assignment.compatible) {
    assignment.assigned_slot_index = best.index;
    assignment.ordered_corners = best.association.ordered_corners;
    assignment.best_normalized_cost = best.cost;
  }
  if (scored_slots.size() > 1u) {
    assignment.second_best_normalized_cost = scored_slots[1].cost;
  }
  assignment.normalized_cost_margin =
      assignment.second_best_normalized_cost - assignment.best_normalized_cost;
  constexpr double kMinimumSlotCostMargin = 0.02;
  assignment.unique =
      assignment.compatible &&
      (!std::isfinite(assignment.second_best_normalized_cost) ||
       assignment.normalized_cost_margin >= kMinimumSlotCostMargin);

  std::ostringstream stream;
  stream << "topology_slot_assignment compatible="
         << (assignment.compatible ? 1 : 0)
         << " unique=" << (assignment.unique ? 1 : 0)
         << " assigned_slot=" << assignment.assigned_slot_index
         << " best_cost=" << assignment.best_normalized_cost
         << " second_cost=" << assignment.second_best_normalized_cost
         << " margin=" << assignment.normalized_cost_margin;
  if (assignment.compatible) {
    stream << " best{" << best.association.summary << "}";
  }
  assignment.summary = stream.str();
  return assignment;
}

bool IsUniqueWrongIdTopologyAssociation(
    const std::array<Eigen::Vector2d, 4>& topology_predicted_corners,
    const OuterWrongIdProposal& proposal,
    const std::vector<std::array<Eigen::Vector2d, 4>>& competing_slots,
    std::string* summary) {
  std::vector<std::array<Eigen::Vector2d, 4>> slots;
  slots.reserve(competing_slots.size() + 1u);
  slots.push_back(topology_predicted_corners);
  slots.insert(slots.end(), competing_slots.begin(), competing_slots.end());
  const TopologySlotAssignment assignment =
      AssignObservedQuadToTopologySlots(slots, proposal);
  const bool unique = assignment.unique && assignment.assigned_slot_index == 0;
  if (summary != nullptr) {
    *summary = assignment.summary +
               " target_slot_unique=" + (unique ? "1" : "0");
  }
  return unique;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
