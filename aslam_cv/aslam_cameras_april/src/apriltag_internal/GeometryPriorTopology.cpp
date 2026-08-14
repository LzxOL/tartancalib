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

bool IsUniqueWrongIdTopologyAssociation(
    const std::array<Eigen::Vector2d, 4>& topology_predicted_corners,
    const OuterWrongIdProposal& proposal,
    const std::vector<std::array<Eigen::Vector2d, 4>>& competing_slots,
    std::string* summary) {
  const WrongIdTopologyAssociation target =
      AssociateWrongIdProposalToTopology(topology_predicted_corners, proposal);
  if (!target.compatible) {
    if (summary != nullptr) {
      *summary = target.summary + " unique_slot=0 target_incompatible";
    }
    return false;
  }

  // Keep a small, normalized margin between the requested slot and every
  // competing missing slot.  This is an identity ambiguity check, not a
  // corner displacement gate; the existing association compatibility bounds
  // remain the primary geometric limits.
  constexpr double kMinimumSlotMargin = 0.02;
  double best_competing_rmse = std::numeric_limits<double>::infinity();
  int competing_slot_index = -1;
  for (std::size_t index = 0; index < competing_slots.size(); ++index) {
    const WrongIdTopologyAssociation competing =
        AssociateWrongIdProposalToTopology(competing_slots[index], proposal);
    if (std::isfinite(competing.normalized_corner_rmse) &&
        competing.normalized_corner_rmse < best_competing_rmse) {
      best_competing_rmse = competing.normalized_corner_rmse;
      competing_slot_index = static_cast<int>(index);
    }
  }
  const bool unique =
      !std::isfinite(best_competing_rmse) ||
      target.normalized_corner_rmse + kMinimumSlotMargin <
          best_competing_rmse;
  if (summary != nullptr) {
    std::ostringstream stream;
    stream << target.summary << " unique_slot=" << (unique ? 1 : 0)
           << " target_rmse=" << target.normalized_corner_rmse
           << " best_competing_rmse=" << best_competing_rmse
           << " competing_slot_index=" << competing_slot_index
           << " slot_margin="
           << (std::isfinite(best_competing_rmse)
                   ? best_competing_rmse - target.normalized_corner_rmse
                   : std::numeric_limits<double>::infinity());
    *summary = stream.str();
  }
  return unique;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
