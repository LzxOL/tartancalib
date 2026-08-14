#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_GEOMETRY_PRIOR_TOPOLOGY_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_GEOMETRY_PRIOR_TOPOLOGY_HPP

#include <array>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

// Geometry-only diagnostics shared by topology association and geometry-prior
// candidate validation.  These checks describe the image quadrilateral; they
// do not decide whether a board observation is accepted by the regenerator.
struct QuadTopologyCheck {
  bool valid = false;
  bool finite = false;
  bool convex = false;
  bool self_intersecting = false;
  double signed_area_px = 0.0;
  double area_px = 0.0;
  double min_edge_length_px = 0.0;
  std::string summary;
};

double SignedPolygonArea(const std::array<cv::Point2f, 4>& corners);
double PolygonArea(const std::array<cv::Point2f, 4>& corners);
QuadTopologyCheck CheckQuadTopology(
    const std::array<cv::Point2f, 4>& corners);

// A wrong-ID decoder proposal is only a topology candidate. The caller still
// performs the existing image-evidence, tag-likelihood, and pose gates.
struct WrongIdTopologyAssociation {
  bool compatible = false;
  std::array<Eigen::Vector2d, 4> ordered_corners{};
  int cyclic_shift = 0;
  bool reflected_order = false;
  double normalized_corner_rmse = std::numeric_limits<double>::infinity();
  double normalized_center_error = std::numeric_limits<double>::infinity();
  double area_ratio = std::numeric_limits<double>::quiet_NaN();
  std::string summary;
};

WrongIdTopologyAssociation AssociateWrongIdProposalToTopology(
    const std::array<Eigen::Vector2d, 4>& topology_predicted_corners,
    const OuterWrongIdProposal& proposal);

// Verifies that a wrong-ID quad belongs uniquely to the requested topology
// slot.  Competing slots are only used for identity assignment; their
// projected corners are never committed as observations.
bool IsUniqueWrongIdTopologyAssociation(
    const std::array<Eigen::Vector2d, 4>& topology_predicted_corners,
    const OuterWrongIdProposal& proposal,
    const std::vector<std::array<Eigen::Vector2d, 4>>& competing_slots,
    std::string* summary);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif
