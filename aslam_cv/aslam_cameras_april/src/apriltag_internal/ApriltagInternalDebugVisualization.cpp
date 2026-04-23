#include <aslam/cameras/apriltag_internal/ApriltagInternalDebugVisualization.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

bool CvVecToUnitRay(const cv::Vec3d& ray_cv, Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    throw std::runtime_error("CvVecToUnitRay requires a valid output pointer.");
  }
  const Eigen::Vector3d candidate(ray_cv[0], ray_cv[1], ray_cv[2]);
  const double norm = candidate.norm();
  if (!std::isfinite(norm) || norm <= 1e-9) {
    return false;
  }
  *ray = candidate / norm;
  return true;
}

bool NormalizeRay(Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    throw std::runtime_error("NormalizeRay requires a valid output pointer.");
  }
  const double norm = ray->norm();
  if (!std::isfinite(norm) || norm <= 1e-9) {
    return false;
  }
  *ray /= norm;
  return true;
}

cv::Point2f MapRayToSpherePanel(const Eigen::Vector3d& ray,
                                const cv::Point2f& panel_center,
                                float panel_radius) {
  return cv::Point2f(panel_center.x + panel_radius * static_cast<float>(ray.x()),
                     panel_center.y - panel_radius * static_cast<float>(ray.y()));
}

bool BuildLocalSphereOffsetRay(const Eigen::Vector3d& anchor_ray,
                               const Eigen::Vector3d& tangent_u,
                               const Eigen::Vector3d& tangent_v,
                               double alpha,
                               double beta,
                               Eigen::Vector3d* ray) {
  if (ray == nullptr) {
    throw std::runtime_error("BuildLocalSphereOffsetRay requires a valid output pointer.");
  }
  const Eigen::Vector3d candidate = anchor_ray + alpha * tangent_u + beta * tangent_v;
  const double norm = candidate.norm();
  if (!std::isfinite(norm) || norm <= 1e-9) {
    return false;
  }
  *ray = candidate / norm;
  return true;
}

void DrawInsetLegendCallout(cv::Mat* image,
                            const cv::Rect& inset_rect,
                            const cv::Point2f& point,
                            const std::string& text,
                            const cv::Scalar& color,
                            int slot) {
  if (image == nullptr) {
    return;
  }

  const double font_scale = 0.75;
  const int font_thickness = 1;
  int baseline = 0;
  const cv::Size text_size =
      cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN, font_scale, font_thickness, &baseline);
  const int padding_x = 4;
  const int padding_y = 3;
  const int box_width = text_size.width + 2 * padding_x;
  const int box_height = text_size.height + baseline + 2 * padding_y;

  int box_x = inset_rect.x + 6;
  int box_y = inset_rect.y + 18;
  switch (slot) {
    case 0:
      box_x = inset_rect.x + 6;
      box_y = inset_rect.y + 18;
      break;
    case 1:
      box_x = inset_rect.x + inset_rect.width - box_width - 6;
      box_y = inset_rect.y + 18;
      break;
    case 2:
      box_x = inset_rect.x + 6;
      box_y = inset_rect.y + inset_rect.height - box_height - 6;
      break;
    default:
      box_x = inset_rect.x + inset_rect.width - box_width - 6;
      box_y = inset_rect.y + inset_rect.height - box_height - 6;
      break;
  }

  const cv::Rect box_rect(box_x, box_y, box_width, box_height);
  const cv::Point2f anchor(
      static_cast<float>(slot == 0 ? box_rect.x : box_rect.x + box_rect.width),
      static_cast<float>(box_rect.y + box_rect.height * 0.5f));
  cv::line(*image, point, anchor, color, 1, cv::LINE_AA);
  cv::rectangle(*image, box_rect, cv::Scalar(255, 255, 255), cv::FILLED);
  cv::rectangle(*image, box_rect, color, 1, cv::LINE_AA);
  cv::putText(*image, text,
              cv::Point(box_rect.x + padding_x,
                        box_rect.y + box_rect.height - padding_y - baseline),
              cv::FONT_HERSHEY_PLAIN, font_scale, color, font_thickness, cv::LINE_AA);
}

cv::Mat MakeColorCanvas(const cv::Mat& image) {
  if (image.empty()) {
    return cv::Mat();
  }

  cv::Mat color;
  if (image.channels() == 1) {
    cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, color, cv::COLOR_BGRA2BGR);
  } else {
    color = image.clone();
  }
  return color;
}

cv::Mat RenderTitledTile(const cv::Mat& image, const std::string& title) {
  if (image.empty()) {
    return cv::Mat();
  }

  const cv::Mat color = MakeColorCanvas(image);
  const int title_height = 34;
  cv::Mat tile(color.rows + title_height, color.cols, CV_8UC3, cv::Scalar(24, 24, 24));
  color.copyTo(tile(cv::Rect(0, title_height, color.cols, color.rows)));
  cv::putText(tile, title, cv::Point(12, 23), cv::FONT_HERSHEY_SIMPLEX, 0.60,
              cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
  return tile;
}

cv::Mat PadTileToHeight(const cv::Mat& tile, int target_height) {
  if (tile.empty() || tile.rows == target_height) {
    return tile.clone();
  }
  cv::Mat padded(target_height, tile.cols, tile.type(), cv::Scalar(24, 24, 24));
  tile.copyTo(padded(cv::Rect(0, 0, tile.cols, tile.rows)));
  return padded;
}

int ComputeSphereSearchRadiusOverlayPx(double local_module_scale) {
  return std::max(
      6, static_cast<int>(std::lround(0.35 * std::max(1.0, local_module_scale))));
}

int ComputeRayRefineTrustRadiusOverlayPx(double local_module_scale,
                                         int search_radius_px) {
  const int trust_radius_px = std::max(
      4, static_cast<int>(std::lround(0.28 * std::max(1.0, local_module_scale))));
  return std::min(trust_radius_px, std::max(4, search_radius_px - 2));
}

std::array<cv::Scalar, 4> BorderCurveColors() {
  return {cv::Scalar(110, 110, 230), cv::Scalar(90, 180, 235),
          cv::Scalar(110, 210, 120), cv::Scalar(220, 150, 110)};
}

void DrawImageBorderCurves(cv::Mat* image, const ApriltagInternalDetectionResult& result) {
  if (image == nullptr || !result.border_boundary_model_valid) {
    return;
  }

  const std::array<cv::Scalar, 4> colors = BorderCurveColors();
  for (std::size_t edge_index = 0; edge_index < result.border_curves_image.size(); ++edge_index) {
    const std::vector<cv::Point2f>& curve = result.border_curves_image[edge_index];
    if (curve.size() < 2) {
      continue;
    }
    for (std::size_t sample_index = 1; sample_index < curve.size(); ++sample_index) {
      cv::line(*image, curve[sample_index - 1], curve[sample_index], colors[edge_index], 1,
               cv::LINE_AA);
    }
  }
}

void DrawSphereBorderCurves(cv::Mat* image,
                            const cv::Point2f& center,
                            float radius,
                            const ApriltagInternalDetectionResult& result) {
  if (image == nullptr || !result.border_boundary_model_valid) {
    return;
  }

  const std::array<cv::Scalar, 4> colors = BorderCurveColors();
  for (std::size_t edge_index = 0; edge_index < result.border_curves_ray.size(); ++edge_index) {
    const std::vector<cv::Vec3d>& curve = result.border_curves_ray[edge_index];
    if (curve.size() < 2) {
      continue;
    }
    Eigen::Vector3d previous_ray = Eigen::Vector3d::Zero();
    if (!CvVecToUnitRay(curve.front(), &previous_ray)) {
      continue;
    }
    for (std::size_t sample_index = 1; sample_index < curve.size(); ++sample_index) {
      Eigen::Vector3d current_ray = Eigen::Vector3d::Zero();
      if (!CvVecToUnitRay(curve[sample_index], &current_ray)) {
        continue;
      }
      cv::line(*image, MapRayToSpherePanel(previous_ray, center, radius),
               MapRayToSpherePanel(current_ray, center, radius), colors[edge_index], 1,
               cv::LINE_AA);
      previous_ray = current_ray;
    }
  }
}

}  // namespace

bool UsesSphereSeedPipeline(InternalProjectionMode mode) {
  return mode == InternalProjectionMode::SphereLattice ||
         mode == InternalProjectionMode::SphereBorderLattice ||
         mode == InternalProjectionMode::SphereRayRefine;
}

bool UsesBorderConditionedSeedPipeline(InternalProjectionMode mode) {
  return mode == InternalProjectionMode::SphereBorderLattice;
}

bool HasExplicitSeedStage(InternalProjectionMode mode) {
  return mode == InternalProjectionMode::VirtualPinholePatchBoundarySeed ||
         UsesSphereSeedPipeline(mode);
}

cv::Mat BuildInternalSeedOverlay(const cv::Mat& image,
                                 const ApriltagInternalDetectionResult& result) {
  if (result.internal_corner_debug.empty() || image.empty()) {
    return cv::Mat();
  }

  cv::Mat overlay = MakeColorCanvas(image);
  const bool use_explicit_seed = HasExplicitSeedStage(result.projection_mode);
  const bool use_border_seed = UsesBorderConditionedSeedPipeline(result.projection_mode);
  const cv::Scalar kPredictedColor(0, 165, 255);
  const cv::Scalar kBorderSeedColor(255, 180, 0);
  const cv::Scalar kSeedColor(255, 80, 255);
  const cv::Scalar kRefinedColor(0, 220, 80);
  const cv::Scalar kBoundaryUColor(190, 190, 190);
  const cv::Scalar kBoundaryVColor(115, 115, 115);
  const cv::Scalar kArrow1Color(180, 180, 180);
  const cv::Scalar kArrowBorderColor(160, 110, 200);
  const cv::Scalar kArrow2Color(120, 190, 120);

  if (result.tag_detected) {
    const cv::Scalar outer_outline_color(165, 165, 165);
    for (int index = 0; index < 4; ++index) {
      cv::line(overlay, result.outer_corners[index], result.outer_corners[(index + 1) % 4],
               outer_outline_color, 2, cv::LINE_AA);
    }
  }
  if (result.projection_mode == InternalProjectionMode::SphereBorderLattice) {
    DrawImageBorderCurves(&overlay, result);
  }

  for (const auto& debug : result.internal_corner_debug) {
    const bool predicted_ok = debug.predicted_image.x >= 0.0f &&
                              debug.predicted_image.x < static_cast<float>(result.image_size.width) &&
                              debug.predicted_image.y >= 0.0f &&
                              debug.predicted_image.y < static_cast<float>(result.image_size.height);
    const bool border_seed_ok =
        use_border_seed && debug.border_seed_valid &&
        debug.border_seed_image.x >= 0.0f &&
        debug.border_seed_image.x < static_cast<float>(result.image_size.width) &&
        debug.border_seed_image.y >= 0.0f &&
        debug.border_seed_image.y < static_cast<float>(result.image_size.height);
    const bool seed_ok = use_explicit_seed &&
                         debug.sphere_seed_image.x >= 0.0f &&
                         debug.sphere_seed_image.x < static_cast<float>(result.image_size.width) &&
                         debug.sphere_seed_image.y >= 0.0f &&
                         debug.sphere_seed_image.y < static_cast<float>(result.image_size.height);
    const bool refined_ok = debug.refined_image.x >= 0.0f &&
                            debug.refined_image.x < static_cast<float>(result.image_size.width) &&
                            debug.refined_image.y >= 0.0f &&
                            debug.refined_image.y < static_cast<float>(result.image_size.height);
    if (!predicted_ok) {
      continue;
    }

    const cv::Point2f boundary_center =
        seed_ok ? debug.sphere_seed_image
                : (border_seed_ok ? debug.border_seed_image : debug.predicted_image);
    const double module_u_length = std::hypot(debug.module_u_axis.x, debug.module_u_axis.y);
    const double module_v_length = std::hypot(debug.module_v_axis.x, debug.module_v_axis.y);
    if (module_u_length > 1.0 && module_v_length > 1.0) {
      const cv::Point2f unit_u =
          debug.module_u_axis * static_cast<float>(1.0 / std::max(1e-9, module_u_length));
      const cv::Point2f unit_v =
          debug.module_v_axis * static_cast<float>(1.0 / std::max(1e-9, module_v_length));
      const float u_half_length = std::max(6.0f, static_cast<float>(0.55 * module_v_length));
      const float v_half_length = std::max(6.0f, static_cast<float>(0.55 * module_u_length));
      cv::line(overlay, boundary_center - u_half_length * unit_v,
               boundary_center + u_half_length * unit_v, kBoundaryUColor, 1, cv::LINE_AA);
      cv::line(overlay, boundary_center - v_half_length * unit_u,
               boundary_center + v_half_length * unit_u, kBoundaryVColor, 1, cv::LINE_AA);
    }

    int search_radius_px = 0;
    if (UsesSphereSeedPipeline(result.projection_mode)) {
      const int base_search_radius_px = ComputeSphereSearchRadiusOverlayPx(debug.local_module_scale);
      const double search_scale =
          debug.sphere_search_radius > 1e-9
              ? std::max(1.0, debug.adaptive_search_radius / debug.sphere_search_radius)
              : 1.0;
      search_radius_px =
          std::max(base_search_radius_px,
                   static_cast<int>(std::lround(base_search_radius_px * search_scale)));
      const cv::Point2f search_center = border_seed_ok ? debug.border_seed_image : debug.predicted_image;
      cv::circle(overlay, search_center, search_radius_px,
                 cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    }

    cv::drawMarker(overlay, debug.predicted_image, cv::Scalar(255, 255, 255),
                   cv::MARKER_CROSS, 8, 3, cv::LINE_AA);
    cv::drawMarker(overlay, debug.predicted_image, kPredictedColor,
                   cv::MARKER_CROSS, 6, 1, cv::LINE_AA);
    cv::circle(overlay, debug.predicted_image, 2, cv::Scalar(255, 255, 255), cv::FILLED,
               cv::LINE_AA);
    cv::circle(overlay, debug.predicted_image, 1, kPredictedColor, cv::FILLED, cv::LINE_AA);

    if (border_seed_ok) {
      cv::arrowedLine(overlay, debug.predicted_image, debug.border_seed_image,
                      kArrow1Color, 1, cv::LINE_AA, 0, 0.15);
      cv::drawMarker(overlay, debug.border_seed_image, cv::Scalar(255, 255, 255),
                     cv::MARKER_TRIANGLE_UP, 8, 3, cv::LINE_AA);
      cv::drawMarker(overlay, debug.border_seed_image, kBorderSeedColor,
                     cv::MARKER_TRIANGLE_UP, 6, 1, cv::LINE_AA);
    }
    if (seed_ok) {
      cv::arrowedLine(overlay, border_seed_ok ? debug.border_seed_image : debug.predicted_image,
                      debug.sphere_seed_image, border_seed_ok ? kArrowBorderColor : kArrow1Color,
                      1, cv::LINE_AA, 0, 0.15);
      if (result.projection_mode == InternalProjectionMode::SphereRayRefine &&
          debug.ray_refine_trust_radius > 0.0) {
        const int trust_radius_px =
            ComputeRayRefineTrustRadiusOverlayPx(debug.local_module_scale, search_radius_px);
        cv::circle(overlay, border_seed_ok ? debug.border_seed_image : debug.predicted_image,
                   trust_radius_px, cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
      }
      cv::drawMarker(overlay, debug.sphere_seed_image, cv::Scalar(255, 255, 255),
                     cv::MARKER_DIAMOND, 8, 3, cv::LINE_AA);
      cv::drawMarker(overlay, debug.sphere_seed_image, kSeedColor,
                     cv::MARKER_DIAMOND, 6, 1, cv::LINE_AA);
    }
    if (seed_ok && refined_ok) {
      cv::arrowedLine(overlay, debug.sphere_seed_image, debug.refined_image,
                      kArrow2Color, 1, cv::LINE_AA, 0, 0.15);
    } else if (predicted_ok && refined_ok) {
      cv::arrowedLine(overlay, debug.predicted_image, debug.refined_image,
                      kArrow2Color, 1, cv::LINE_AA, 0, 0.15);
    }
    if (refined_ok) {
      cv::drawMarker(overlay, debug.refined_image, cv::Scalar(255, 255, 255),
                     cv::MARKER_SQUARE, 7, 3, cv::LINE_AA);
      cv::drawMarker(overlay, debug.refined_image, kRefinedColor,
                     cv::MARKER_SQUARE, 5, 1, cv::LINE_AA);
    }
  }

  std::string title;
  std::string legend;
  if (result.projection_mode == InternalProjectionMode::VirtualPinholePatch) {
    title = "Internal Patch Overlay: P -> R";
    legend =
        "Legend: P orange cross, R green square, gray cross: aligned lattice boundaries";
  } else if (result.projection_mode == InternalProjectionMode::VirtualPinholeImageSubpix) {
    title = "Internal Patch-to-Image Overlay: P -> R(image subpix)";
    legend =
        "Legend: P orange cross, R green square, gray cross: aligned lattice boundaries";
  } else if (result.projection_mode ==
             InternalProjectionMode::VirtualPinholePatchBoundarySeed) {
    title = "Internal Patch-Seed Overlay: P -> SS(patch edge) -> R";
    legend =
        "Legend: P orange cross, SS magenta diamond, R green square, gray cross: aligned lattice boundaries";
  } else if (result.projection_mode == InternalProjectionMode::SphereBorderLattice) {
    title = "Internal Border-Seed Overlay: P -> BC -> SS -> R";
    legend =
        "Legend: P orange cross, BC blue triangle, SS magenta diamond, R green square, colored curves: top/right/bottom/left outer boundaries, gray cross: aligned lattice boundaries";
  } else if (result.projection_mode == InternalProjectionMode::SphereRayRefine) {
    title = "Internal Ray-Seed Overlay: P -> SS(ray) -> R(subpix)";
    legend =
        "Legend: P orange cross, SS magenta diamond, R green square, gray circle: predicted-ray trust region, gray cross: aligned lattice boundaries";
  } else {
    title = "Internal Sphere Seed Overlay: P -> SS -> R";
    legend =
        "Legend: P orange cross, SS magenta diamond, R green square, gray cross: aligned lattice boundaries";
  }
  cv::putText(overlay, title,
              cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 3,
              cv::LINE_AA);
  cv::putText(overlay, title,
              cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(30, 30, 30), 1,
              cv::LINE_AA);
  cv::putText(overlay, legend,
              cv::Point(20, 56), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 3,
              cv::LINE_AA);
  cv::putText(overlay, legend,
              cv::Point(20, 56), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(40, 40, 40), 1,
              cv::LINE_AA);
  return overlay;
}

cv::Mat BuildInternalSphereDebugView(const ApriltagInternalDetectionResult& result) {
  if (!UsesSphereSeedPipeline(result.projection_mode) ||
      result.internal_corner_debug.empty()) {
    return cv::Mat();
  }

  const int panel_columns = 4;
  const int panel_width = 320;
  const int panel_height = 230;
  const int margin = 26;
  const int header_height = 85;
  const int panel_count = static_cast<int>(result.internal_corner_debug.size());
  const int panel_rows = std::max(1, (panel_count + panel_columns - 1) / panel_columns);
  const int canvas_width = panel_columns * panel_width + (panel_columns + 1) * margin;
  const int canvas_height = header_height + panel_rows * panel_height + (panel_rows + 1) * margin;

  cv::Mat canvas(canvas_height, canvas_width, CV_8UC3, cv::Scalar(248, 248, 248));
  const std::string header =
      result.projection_mode == InternalProjectionMode::SphereBorderLattice
          ? "Internal Sphere View: predicted ray -> border-conditioned ray -> sphere seed -> refined ray"
          : result.projection_mode == InternalProjectionMode::SphereRayRefine
                ? "Internal Sphere View: predicted ray -> ray-domain seed -> subpixel refined ray"
                : "Internal Sphere View: predicted ray -> sphere seed -> refined ray";
  const std::string subtitle =
      result.projection_mode == InternalProjectionMode::SphereBorderLattice
          ? "P orange, BC blue, SS magenta, R green. Gray arrow: P->BC, violet arrow: BC->SS, green arrow: SS->R. Colored curves: top/right/bottom/left outer sphere boundaries. Thin gray lines: border-conditioned lattice cross."
          : result.projection_mode == InternalProjectionMode::SphereRayRefine
                ? "P orange, SS magenta, R green. Gray arrow: P->SS, green arrow: SS->R. Gray cross: aligned lattice boundaries. Gray ring: predicted-ray trust region."
                : "P orange, SS magenta, R green. Gray arrow: P->SS, green arrow: SS->R. Gray cross: aligned lattice boundaries.";
  cv::putText(canvas, header,
              cv::Point(28, 40), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(20, 20, 20), 2);
  cv::putText(canvas, subtitle,
              cv::Point(28, 70), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(60, 60, 60), 1);

  const cv::Scalar kPredictedColor(0, 165, 255);
  const cv::Scalar kBorderSeedColor(255, 180, 0);
  const cv::Scalar kSeedColor(255, 80, 255);
  const cv::Scalar kRefinedColor(0, 220, 80);
  const cv::Scalar kBoundaryUColor(190, 190, 190);
  const cv::Scalar kBoundaryVColor(115, 115, 115);
  const cv::Scalar kUAxisColor(150, 150, 150);
  const cv::Scalar kVAxisColor(90, 90, 90);
  const cv::Scalar kSearchBoxColor(190, 190, 190);
  const cv::Scalar kArrow1Color(180, 180, 180);
  const cv::Scalar kArrowBorderColor(160, 110, 200);
  const cv::Scalar kArrow2Color(120, 190, 120);

  for (int index = 0; index < panel_count; ++index) {
    const auto& debug = result.internal_corner_debug[static_cast<std::size_t>(index)];
    const int row = index / panel_columns;
    const int col = index % panel_columns;
    const cv::Rect panel_rect(margin + col * (panel_width + margin),
                              header_height + margin + row * (panel_height + margin),
                              panel_width, panel_height);
    cv::rectangle(canvas, panel_rect, cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(canvas, panel_rect, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);

    const cv::Point2f center(panel_rect.x + panel_rect.width * 0.5f,
                             panel_rect.y + panel_rect.height * 0.43f);
    const float radius = 0.33f * static_cast<float>(std::min(panel_rect.width, panel_rect.height));
    cv::circle(canvas, center, static_cast<int>(std::lround(radius)),
               cv::Scalar(215, 215, 215), 1, cv::LINE_AA);
    cv::line(canvas,
             cv::Point(static_cast<int>(std::lround(center.x - radius)),
                       static_cast<int>(std::lround(center.y))),
             cv::Point(static_cast<int>(std::lround(center.x + radius)),
                       static_cast<int>(std::lround(center.y))),
             cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    cv::line(canvas,
             cv::Point(static_cast<int>(std::lround(center.x)),
                       static_cast<int>(std::lround(center.y - radius))),
             cv::Point(static_cast<int>(std::lround(center.x)),
                       static_cast<int>(std::lround(center.y + radius))),
             cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    if (result.projection_mode == InternalProjectionMode::SphereBorderLattice) {
      DrawSphereBorderCurves(&canvas, center, radius, result);
    }

    Eigen::Vector3d predicted_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d border_seed_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d seed_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d refined_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent_u = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent_v = Eigen::Vector3d::Zero();
    cv::Point2f predicted_point{};
    cv::Point2f border_seed_point{};
    cv::Point2f seed_point{};
    cv::Point2f refined_point{};
    cv::Point2f u_plus_point{};
    cv::Point2f v_plus_point{};
    std::array<cv::Point2f, 4> search_box_points{};
    std::array<cv::Point2f, 2> boundary_u_points{};
    std::array<cv::Point2f, 2> boundary_v_points{};
    std::array<cv::Point2f, 2> border_vertical_points{};
    std::array<cv::Point2f, 2> border_horizontal_points{};
    std::vector<cv::Point2f> trust_circle_points;
    bool search_box_ok = false;
    bool u_plus_ok = false;
    bool v_plus_ok = false;
    bool boundary_u_ok = false;
    bool boundary_v_ok = false;
    bool border_vertical_ok = false;
    bool border_horizontal_ok = false;
    const bool predicted_ok = CvVecToUnitRay(debug.predicted_ray, &predicted_ray);
    const bool border_seed_ok =
        UsesBorderConditionedSeedPipeline(result.projection_mode) &&
        debug.border_seed_valid &&
        CvVecToUnitRay(debug.border_seed_ray, &border_seed_ray);
    const bool seed_ok = CvVecToUnitRay(debug.sphere_seed_ray, &seed_ray);
    const bool refined_ok = CvVecToUnitRay(debug.refined_ray, &refined_ray);
    const bool tangent_u_ok = CvVecToUnitRay(debug.tangent_u_ray, &tangent_u);
    const bool tangent_v_ok = CvVecToUnitRay(debug.tangent_v_ray, &tangent_v);
    Eigen::Vector3d border_top_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d border_bottom_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d border_left_ray = Eigen::Vector3d::Zero();
    Eigen::Vector3d border_right_ray = Eigen::Vector3d::Zero();
    const bool border_top_ok = border_seed_ok && CvVecToUnitRay(debug.border_top_ray, &border_top_ray);
    const bool border_bottom_ok =
        border_seed_ok && CvVecToUnitRay(debug.border_bottom_ray, &border_bottom_ray);
    const bool border_left_ok = border_seed_ok && CvVecToUnitRay(debug.border_left_ray, &border_left_ray);
    const bool border_right_ok =
        border_seed_ok && CvVecToUnitRay(debug.border_right_ray, &border_right_ray);

    if (predicted_ok && tangent_u_ok && tangent_v_ok && debug.adaptive_search_radius > 1e-9) {
      const Eigen::Vector3d search_anchor_ray = border_seed_ok ? border_seed_ray : predicted_ray;
      Eigen::Vector3d search_tangent_u = tangent_u;
      Eigen::Vector3d search_tangent_v = tangent_v;
      if (border_seed_ok) {
        search_tangent_u = tangent_u - search_anchor_ray * search_anchor_ray.dot(tangent_u);
        if (!NormalizeRay(&search_tangent_u)) {
          search_tangent_u = tangent_u;
        }
        search_tangent_v = tangent_v - search_anchor_ray * search_anchor_ray.dot(tangent_v);
        search_tangent_v =
            search_tangent_v - search_tangent_u * search_tangent_u.dot(search_tangent_v);
        if (!NormalizeRay(&search_tangent_v)) {
          search_tangent_v = tangent_v;
        }
        if (search_anchor_ray.dot(search_tangent_u.cross(search_tangent_v)) < 0.0) {
          search_tangent_v = -search_tangent_v;
        }
      }
      const double r = debug.adaptive_search_radius;
      Eigen::Vector3d u_plus = Eigen::Vector3d::Zero();
      Eigen::Vector3d u_minus = Eigen::Vector3d::Zero();
      Eigen::Vector3d v_plus = Eigen::Vector3d::Zero();
      Eigen::Vector3d v_minus = Eigen::Vector3d::Zero();
      Eigen::Vector3d c00 = Eigen::Vector3d::Zero();
      Eigen::Vector3d c10 = Eigen::Vector3d::Zero();
      Eigen::Vector3d c11 = Eigen::Vector3d::Zero();
      Eigen::Vector3d c01 = Eigen::Vector3d::Zero();
      if (BuildLocalSphereOffsetRay(search_anchor_ray, search_tangent_u, search_tangent_v, r, 0.0, &u_plus) &&
          BuildLocalSphereOffsetRay(search_anchor_ray, search_tangent_u, search_tangent_v, -r, 0.0, &u_minus) &&
          BuildLocalSphereOffsetRay(search_anchor_ray, search_tangent_u, search_tangent_v, 0.0, r, &v_plus) &&
          BuildLocalSphereOffsetRay(search_anchor_ray, search_tangent_u, search_tangent_v, 0.0, -r, &v_minus) &&
          BuildLocalSphereOffsetRay(search_anchor_ray, search_tangent_u, search_tangent_v, -r, -r, &c00) &&
          BuildLocalSphereOffsetRay(search_anchor_ray, search_tangent_u, search_tangent_v, r, -r, &c10) &&
          BuildLocalSphereOffsetRay(search_anchor_ray, search_tangent_u, search_tangent_v, r, r, &c11) &&
          BuildLocalSphereOffsetRay(search_anchor_ray, search_tangent_u, search_tangent_v, -r, r, &c01)) {
        search_box_points = {{
            MapRayToSpherePanel(c00, center, radius),
            MapRayToSpherePanel(c10, center, radius),
            MapRayToSpherePanel(c11, center, radius),
            MapRayToSpherePanel(c01, center, radius),
        }};
        search_box_ok = true;
        for (std::size_t edge_index = 0; edge_index < search_box_points.size(); ++edge_index) {
          cv::line(canvas, search_box_points[edge_index],
                   search_box_points[(edge_index + 1) % search_box_points.size()],
                   kSearchBoxColor, 1, cv::LINE_AA);
        }
        u_plus_point = MapRayToSpherePanel(u_plus, center, radius);
        v_plus_point = MapRayToSpherePanel(v_plus, center, radius);
        u_plus_ok = true;
        v_plus_ok = true;
        cv::arrowedLine(canvas, MapRayToSpherePanel(search_anchor_ray, center, radius),
                        u_plus_point, kUAxisColor, 1, cv::LINE_AA, 0, 0.15);
        cv::arrowedLine(canvas, MapRayToSpherePanel(search_anchor_ray, center, radius),
                        v_plus_point, kVAxisColor, 1, cv::LINE_AA, 0, 0.15);
        cv::putText(canvas, "u", u_plus_point + cv::Point2f(6.0f, -4.0f),
                    cv::FONT_HERSHEY_PLAIN, 0.8, kUAxisColor, 1, cv::LINE_AA);
        cv::putText(canvas, "v", v_plus_point + cv::Point2f(6.0f, -4.0f),
                    cv::FONT_HERSHEY_PLAIN, 0.8, kVAxisColor, 1, cv::LINE_AA);
      }
    }

    if (seed_ok && tangent_u_ok && tangent_v_ok) {
      auto project_to_seed_tangent = [&](const Eigen::Vector3d& source_tangent,
                                         Eigen::Vector3d* projected) {
        if (projected == nullptr) {
          return false;
        }
        *projected = source_tangent - seed_ray * seed_ray.dot(source_tangent);
        const double norm = projected->norm();
        if (!std::isfinite(norm) || norm <= 1e-9) {
          return false;
        }
        *projected /= norm;
        return true;
      };

      Eigen::Vector3d seed_tangent_u = Eigen::Vector3d::Zero();
      Eigen::Vector3d seed_tangent_v = Eigen::Vector3d::Zero();
      if (project_to_seed_tangent(tangent_u, &seed_tangent_u) &&
          project_to_seed_tangent(tangent_v, &seed_tangent_v)) {
        const double boundary_extent = std::max(
            0.06, std::min(0.18, 0.75 * std::max(debug.sphere_search_radius, 0.08)));
        Eigen::Vector3d boundary_u_minus = Eigen::Vector3d::Zero();
        Eigen::Vector3d boundary_u_plus = Eigen::Vector3d::Zero();
        Eigen::Vector3d boundary_v_minus = Eigen::Vector3d::Zero();
        Eigen::Vector3d boundary_v_plus = Eigen::Vector3d::Zero();
        if (BuildLocalSphereOffsetRay(seed_ray, seed_tangent_u, seed_tangent_v, 0.0,
                                      -boundary_extent, &boundary_u_minus) &&
            BuildLocalSphereOffsetRay(seed_ray, seed_tangent_u, seed_tangent_v, 0.0,
                                      boundary_extent, &boundary_u_plus)) {
          boundary_u_points = {{
              MapRayToSpherePanel(boundary_u_minus, center, radius),
              MapRayToSpherePanel(boundary_u_plus, center, radius),
          }};
          boundary_u_ok = true;
          cv::line(canvas, boundary_u_points[0], boundary_u_points[1], kBoundaryUColor, 1,
                   cv::LINE_AA);
        }
        if (BuildLocalSphereOffsetRay(seed_ray, seed_tangent_u, seed_tangent_v,
                                      -boundary_extent, 0.0, &boundary_v_minus) &&
            BuildLocalSphereOffsetRay(seed_ray, seed_tangent_u, seed_tangent_v,
                                      boundary_extent, 0.0, &boundary_v_plus)) {
          boundary_v_points = {{
              MapRayToSpherePanel(boundary_v_minus, center, radius),
              MapRayToSpherePanel(boundary_v_plus, center, radius),
          }};
          boundary_v_ok = true;
          cv::line(canvas, boundary_v_points[0], boundary_v_points[1], kBoundaryVColor, 1,
                   cv::LINE_AA);
        }
      }
    }

    if (border_top_ok && border_bottom_ok) {
      border_vertical_points = {{
          MapRayToSpherePanel(border_top_ray, center, radius),
          MapRayToSpherePanel(border_bottom_ray, center, radius),
      }};
      border_vertical_ok = true;
      cv::line(canvas, border_vertical_points[0], border_vertical_points[1],
               cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
    }
    if (border_left_ok && border_right_ok) {
      border_horizontal_points = {{
          MapRayToSpherePanel(border_left_ray, center, radius),
          MapRayToSpherePanel(border_right_ray, center, radius),
      }};
      border_horizontal_ok = true;
      cv::line(canvas, border_horizontal_points[0], border_horizontal_points[1],
               cv::Scalar(175, 175, 175), 1, cv::LINE_AA);
    }

    if (result.projection_mode == InternalProjectionMode::SphereRayRefine &&
        predicted_ok && tangent_u_ok && tangent_v_ok && debug.ray_refine_trust_radius > 1e-9) {
      constexpr int kTrustCircleSamples = 32;
      trust_circle_points.reserve(kTrustCircleSamples);
      for (int sample_index = 0; sample_index < kTrustCircleSamples; ++sample_index) {
        const double theta =
            2.0 * M_PI * static_cast<double>(sample_index) / static_cast<double>(kTrustCircleSamples);
        Eigen::Vector3d trust_ray = Eigen::Vector3d::Zero();
        if (!BuildLocalSphereOffsetRay(predicted_ray, tangent_u, tangent_v,
                                       debug.ray_refine_trust_radius * std::cos(theta),
                                       debug.ray_refine_trust_radius * std::sin(theta),
                                       &trust_ray)) {
          continue;
        }
        trust_circle_points.push_back(MapRayToSpherePanel(trust_ray, center, radius));
      }
      if (trust_circle_points.size() >= 2) {
        for (std::size_t edge_index = 0; edge_index < trust_circle_points.size(); ++edge_index) {
          cv::line(canvas, trust_circle_points[edge_index],
                   trust_circle_points[(edge_index + 1) % trust_circle_points.size()],
                   cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
        }
      }
    }

    if (predicted_ok && border_seed_ok) {
      cv::arrowedLine(canvas, MapRayToSpherePanel(predicted_ray, center, radius),
                      MapRayToSpherePanel(border_seed_ray, center, radius),
                      kArrow1Color, 2, cv::LINE_AA, 0, 0.14);
    }
    if (border_seed_ok && seed_ok) {
      cv::arrowedLine(canvas, MapRayToSpherePanel(border_seed_ray, center, radius),
                      MapRayToSpherePanel(seed_ray, center, radius),
                      kArrowBorderColor, 2, cv::LINE_AA, 0, 0.14);
    } else if (predicted_ok && seed_ok) {
      cv::arrowedLine(canvas, MapRayToSpherePanel(predicted_ray, center, radius),
                      MapRayToSpherePanel(seed_ray, center, radius),
                      kArrow1Color, 2, cv::LINE_AA, 0, 0.14);
    }
    if (seed_ok && refined_ok) {
      cv::arrowedLine(canvas, MapRayToSpherePanel(seed_ray, center, radius),
                      MapRayToSpherePanel(refined_ray, center, radius),
                      kArrow2Color, 2, cv::LINE_AA, 0, 0.14);
    }

    if (predicted_ok) {
      predicted_point = MapRayToSpherePanel(predicted_ray, center, radius);
      cv::drawMarker(canvas, predicted_point, cv::Scalar(255, 255, 255),
                     cv::MARKER_CROSS, 9, 3, cv::LINE_AA);
      cv::drawMarker(canvas, predicted_point, kPredictedColor, cv::MARKER_CROSS, 7, 1, cv::LINE_AA);
      cv::circle(canvas, predicted_point, 2, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
      cv::circle(canvas, predicted_point, 1, kPredictedColor, cv::FILLED, cv::LINE_AA);
    }
    if (border_seed_ok) {
      border_seed_point = MapRayToSpherePanel(border_seed_ray, center, radius);
      cv::drawMarker(canvas, border_seed_point, cv::Scalar(255, 255, 255),
                     cv::MARKER_TRIANGLE_UP, 9, 3, cv::LINE_AA);
      cv::drawMarker(canvas, border_seed_point, kBorderSeedColor,
                     cv::MARKER_TRIANGLE_UP, 7, 1, cv::LINE_AA);
    }
    if (seed_ok) {
      seed_point = MapRayToSpherePanel(seed_ray, center, radius);
      cv::drawMarker(canvas, seed_point, cv::Scalar(255, 255, 255),
                     cv::MARKER_DIAMOND, 9, 3, cv::LINE_AA);
      cv::drawMarker(canvas, seed_point, kSeedColor, cv::MARKER_DIAMOND, 7, 1, cv::LINE_AA);
    }
    if (refined_ok) {
      refined_point = MapRayToSpherePanel(refined_ray, center, radius);
      cv::drawMarker(canvas, refined_point, cv::Scalar(255, 255, 255),
                     cv::MARKER_SQUARE, 8, 3, cv::LINE_AA);
      cv::drawMarker(canvas, refined_point, kRefinedColor, cv::MARKER_SQUARE, 6, 1, cv::LINE_AA);
    }

    const cv::Rect inset_rect(panel_rect.x + panel_rect.width - 102, panel_rect.y + 12, 90, 90);
    cv::rectangle(canvas, inset_rect, cv::Scalar(252, 252, 252), cv::FILLED);
    cv::rectangle(canvas, inset_rect, cv::Scalar(150, 150, 150), 1, cv::LINE_AA);
    cv::putText(canvas, "zoom", cv::Point(inset_rect.x + 8, inset_rect.y + 14),
                cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);

    std::vector<cv::Point2f> zoom_points;
    if (predicted_ok) zoom_points.push_back(predicted_point);
    if (border_seed_ok) zoom_points.push_back(border_seed_point);
    if (seed_ok) zoom_points.push_back(seed_point);
    if (refined_ok) zoom_points.push_back(refined_point);
    if (search_box_ok) {
      zoom_points.insert(zoom_points.end(), search_box_points.begin(), search_box_points.end());
    }
    if (boundary_u_ok) {
      zoom_points.insert(zoom_points.end(), boundary_u_points.begin(), boundary_u_points.end());
    }
    if (boundary_v_ok) {
      zoom_points.insert(zoom_points.end(), boundary_v_points.begin(), boundary_v_points.end());
    }
    if (border_vertical_ok) {
      zoom_points.insert(zoom_points.end(), border_vertical_points.begin(), border_vertical_points.end());
    }
    if (border_horizontal_ok) {
      zoom_points.insert(zoom_points.end(), border_horizontal_points.begin(), border_horizontal_points.end());
    }
    if (u_plus_ok) zoom_points.push_back(u_plus_point);
    if (v_plus_ok) zoom_points.push_back(v_plus_point);
    zoom_points.insert(zoom_points.end(), trust_circle_points.begin(), trust_circle_points.end());

    if (!zoom_points.empty()) {
      float min_x = zoom_points.front().x;
      float max_x = zoom_points.front().x;
      float min_y = zoom_points.front().y;
      float max_y = zoom_points.front().y;
      for (const cv::Point2f& point : zoom_points) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
      }

      const float extent = std::max({max_x - min_x, max_y - min_y, 12.0f});
      const float padding = 0.45f * extent + 4.0f;
      min_x -= padding;
      max_x += padding;
      min_y -= padding;
      max_y += padding;
      const float inner_width = static_cast<float>(inset_rect.width - 12);
      const float inner_height = static_cast<float>(inset_rect.height - 22);
      const float scale_x = inner_width / std::max(1.0f, max_x - min_x);
      const float scale_y = inner_height / std::max(1.0f, max_y - min_y);
      const float zoom_scale = std::min(scale_x, scale_y);

      auto map_to_inset = [&](const cv::Point2f& point) {
        return cv::Point2f(
            static_cast<float>(inset_rect.x + 6) + (point.x - min_x) * zoom_scale,
            static_cast<float>(inset_rect.y + 18) + (point.y - min_y) * zoom_scale);
      };

      const cv::Point2f inset_center(
          static_cast<float>(inset_rect.x + inset_rect.width * 0.5f),
          static_cast<float>(inset_rect.y + inset_rect.height * 0.58f));
      cv::line(canvas,
               cv::Point(inset_rect.x + 6, static_cast<int>(std::lround(inset_center.y))),
               cv::Point(inset_rect.x + inset_rect.width - 6,
                         static_cast<int>(std::lround(inset_center.y))),
               cv::Scalar(236, 236, 236), 1, cv::LINE_AA);
      cv::line(canvas,
               cv::Point(static_cast<int>(std::lround(inset_center.x)), inset_rect.y + 18),
               cv::Point(static_cast<int>(std::lround(inset_center.x)),
                         inset_rect.y + inset_rect.height - 6),
               cv::Scalar(236, 236, 236), 1, cv::LINE_AA);

      if (search_box_ok) {
        std::array<cv::Point2f, 4> mapped_box{};
        for (std::size_t edge_index = 0; edge_index < search_box_points.size(); ++edge_index) {
          mapped_box[edge_index] = map_to_inset(search_box_points[edge_index]);
        }
        for (std::size_t edge_index = 0; edge_index < mapped_box.size(); ++edge_index) {
          cv::line(canvas, mapped_box[edge_index],
                   mapped_box[(edge_index + 1) % mapped_box.size()],
                   kSearchBoxColor, 1, cv::LINE_AA);
        }
      }
      if (boundary_u_ok) {
        cv::line(canvas, map_to_inset(boundary_u_points[0]), map_to_inset(boundary_u_points[1]),
                 kBoundaryUColor, 1, cv::LINE_AA);
      }
      if (boundary_v_ok) {
        cv::line(canvas, map_to_inset(boundary_v_points[0]), map_to_inset(boundary_v_points[1]),
                 kBoundaryVColor, 1, cv::LINE_AA);
      }
      if (border_vertical_ok) {
        cv::line(canvas, map_to_inset(border_vertical_points[0]),
                 map_to_inset(border_vertical_points[1]), cv::Scalar(205, 205, 205), 1,
                 cv::LINE_AA);
      }
      if (border_horizontal_ok) {
        cv::line(canvas, map_to_inset(border_horizontal_points[0]),
                 map_to_inset(border_horizontal_points[1]), cv::Scalar(175, 175, 175), 1,
                 cv::LINE_AA);
      }
      if (trust_circle_points.size() >= 2) {
        std::vector<cv::Point2f> mapped_trust_circle;
        mapped_trust_circle.reserve(trust_circle_points.size());
        for (const cv::Point2f& point : trust_circle_points) {
          mapped_trust_circle.push_back(map_to_inset(point));
        }
        for (std::size_t edge_index = 0; edge_index < mapped_trust_circle.size(); ++edge_index) {
          cv::line(canvas, mapped_trust_circle[edge_index],
                   mapped_trust_circle[(edge_index + 1) % mapped_trust_circle.size()],
                   cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
        }
      }
      if (predicted_ok && u_plus_ok) {
        cv::arrowedLine(canvas, map_to_inset(predicted_point), map_to_inset(u_plus_point),
                        kUAxisColor, 1, cv::LINE_AA, 0, 0.15);
      }
      if (predicted_ok && v_plus_ok) {
        cv::arrowedLine(canvas, map_to_inset(predicted_point), map_to_inset(v_plus_point),
                        kVAxisColor, 1, cv::LINE_AA, 0, 0.15);
      }
      if (predicted_ok && border_seed_ok) {
        cv::arrowedLine(canvas, map_to_inset(predicted_point), map_to_inset(border_seed_point),
                        kArrow1Color, 1, cv::LINE_AA, 0, 0.12);
      }
      if (border_seed_ok && seed_ok) {
        cv::arrowedLine(canvas, map_to_inset(border_seed_point), map_to_inset(seed_point),
                        kArrowBorderColor, 1, cv::LINE_AA, 0, 0.12);
      } else if (predicted_ok && seed_ok) {
        cv::arrowedLine(canvas, map_to_inset(predicted_point), map_to_inset(seed_point),
                        kArrow1Color, 1, cv::LINE_AA, 0, 0.12);
      }
      if (seed_ok && refined_ok) {
        cv::arrowedLine(canvas, map_to_inset(seed_point), map_to_inset(refined_point),
                        kArrow2Color, 1, cv::LINE_AA, 0, 0.12);
      }

      if (predicted_ok) {
        const cv::Point2f point = map_to_inset(predicted_point);
        cv::drawMarker(canvas, point, cv::Scalar(255, 255, 255),
                       cv::MARKER_CROSS, 9, 3, cv::LINE_AA);
        cv::drawMarker(canvas, point, kPredictedColor, cv::MARKER_CROSS, 7, 1, cv::LINE_AA);
        cv::circle(canvas, point, 2, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, point, 1, kPredictedColor, cv::FILLED, cv::LINE_AA);
        DrawInsetLegendCallout(&canvas, inset_rect, point, "P", kPredictedColor, 0);
      }
      if (border_seed_ok) {
        const cv::Point2f point = map_to_inset(border_seed_point);
        cv::drawMarker(canvas, point, cv::Scalar(255, 255, 255),
                       cv::MARKER_TRIANGLE_UP, 8, 3, cv::LINE_AA);
        cv::drawMarker(canvas, point, kBorderSeedColor, cv::MARKER_TRIANGLE_UP, 6, 1,
                       cv::LINE_AA);
        DrawInsetLegendCallout(&canvas, inset_rect, point, "BC", kBorderSeedColor, 1);
      }
      if (seed_ok) {
        const cv::Point2f point = map_to_inset(seed_point);
        cv::drawMarker(canvas, point, cv::Scalar(255, 255, 255),
                       cv::MARKER_DIAMOND, 8, 3, cv::LINE_AA);
        cv::drawMarker(canvas, point, kSeedColor, cv::MARKER_DIAMOND, 6, 1, cv::LINE_AA);
        DrawInsetLegendCallout(&canvas, inset_rect, point, "SS", kSeedColor,
                               border_seed_ok ? 2 : 1);
      }
      if (refined_ok) {
        const cv::Point2f point = map_to_inset(refined_point);
        cv::drawMarker(canvas, point, cv::Scalar(255, 255, 255),
                       cv::MARKER_SQUARE, 7, 3, cv::LINE_AA);
        cv::drawMarker(canvas, point, kRefinedColor, cv::MARKER_SQUARE, 5, 1, cv::LINE_AA);
        DrawInsetLegendCallout(&canvas, inset_rect, point, "R", kRefinedColor, 3);
      }
    }

    int text_y = panel_rect.y + panel_rect.height - 72;
    std::ostringstream title;
    title << "id " << debug.point_id << " "
          << (debug.corner_type == CornerType::XCorner ? "X" : "L")
          << (debug.valid ? " valid" : " invalid");
    cv::putText(canvas, title.str(), cv::Point(panel_rect.x + 12, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
    text_y += 20;
    std::ostringstream line1;
    line1 << "u=" << std::lround(debug.sphere_template_quality * 100.0)
          << " v=" << std::lround(debug.sphere_gradient_quality * 100.0)
          << " seed=" << std::lround(debug.sphere_seed_quality * 100.0);
    cv::putText(canvas, line1.str(), cv::Point(panel_rect.x + 12, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    text_y += 18;
    std::ostringstream line2;
    if (UsesBorderConditionedSeedPipeline(result.projection_mode) && debug.border_seed_valid) {
      line2 << "P->BC " << std::fixed << std::setprecision(1)
            << debug.predicted_to_border_seed_displacement
            << "  BC->SS " << debug.border_seed_to_sphere_seed_displacement;
    } else {
      line2 << "P->SS " << std::fixed << std::setprecision(1)
            << debug.predicted_to_seed_displacement
            << "  SS->R " << debug.seed_to_refined_displacement;
    }
    cv::putText(canvas, line2.str(), cv::Point(panel_rect.x + 12, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    text_y += 18;
    std::ostringstream line3;
    line3 << "P->R " << std::fixed << std::setprecision(1)
          << debug.predicted_to_refined_displacement
          << "  r=" << std::setprecision(4) << debug.adaptive_search_radius;
    cv::putText(canvas, line3.str(), cv::Point(panel_rect.x + 12, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    if (result.projection_mode == InternalProjectionMode::SphereRayRefine) {
      text_y += 18;
      std::ostringstream line4;
      line4 << "edge=" << std::lround(debug.ray_refine_edge_quality * 100.0)
            << " photo=" << std::lround(debug.ray_refine_photometric_quality * 100.0)
            << " ray=" << std::lround(debug.ray_refine_final_quality * 100.0);
      cv::putText(canvas, line4.str(), cv::Point(panel_rect.x + 12, text_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
      text_y += 18;
      std::ostringstream line5;
      line5 << "tr=" << std::setprecision(5) << debug.ray_refine_trust_radius
            << " it=" << debug.ray_refine_iterations
            << " conv=" << (debug.ray_refine_converged ? "yes" : "no")
            << " ang=" << std::setprecision(4) << debug.seed_to_refined_angular;
      cv::putText(canvas, line5.str(), cv::Point(panel_rect.x + 12, text_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    }
  }

  return canvas;
}

cv::Mat BuildInternalMethodSpaceDebugView(const ApriltagInternalDetectionResult& result) {
  if (UsesSphereSeedPipeline(result.projection_mode)) {
    return BuildInternalSphereDebugView(result);
  }
  if ((result.projection_mode != InternalProjectionMode::VirtualPinholePatch &&
       result.projection_mode != InternalProjectionMode::VirtualPinholeImageSubpix &&
       result.projection_mode != InternalProjectionMode::VirtualPinholePatchBoundarySeed) ||
      result.canonical_patch.empty()) {
    return cv::Mat();
  }

  cv::Mat canvas = MakeColorCanvas(result.canonical_patch);
  const cv::Scalar kPredictedColor(0, 165, 255);
  const cv::Scalar kSeedColor(255, 80, 255);
  const cv::Scalar kRefinedColor(0, 220, 80);
  const cv::Scalar kArrowColor(160, 160, 160);
  const bool use_explicit_seed = result.projection_mode ==
                                 InternalProjectionMode::VirtualPinholePatchBoundarySeed;

  for (const auto& debug : result.internal_corner_debug) {
    const bool predicted_ok =
        debug.predicted_patch.x >= 0.0f &&
        debug.predicted_patch.x < static_cast<float>(canvas.cols) &&
        debug.predicted_patch.y >= 0.0f &&
        debug.predicted_patch.y < static_cast<float>(canvas.rows);
    const bool refined_ok =
        debug.refined_patch.x >= 0.0f &&
        debug.refined_patch.x < static_cast<float>(canvas.cols) &&
        debug.refined_patch.y >= 0.0f &&
        debug.refined_patch.y < static_cast<float>(canvas.rows);
    if (!predicted_ok) {
      continue;
    }

    const bool seed_ok =
        use_explicit_seed &&
        debug.sphere_seed_patch.x >= 0.0f &&
        debug.sphere_seed_patch.x < static_cast<float>(canvas.cols) &&
        debug.sphere_seed_patch.y >= 0.0f &&
        debug.sphere_seed_patch.y < static_cast<float>(canvas.rows);

    if (seed_ok) {
      cv::arrowedLine(canvas, debug.predicted_patch, debug.sphere_seed_patch, kArrowColor, 1,
                      cv::LINE_AA, 0, 0.15);
    }
    if (seed_ok && refined_ok) {
      cv::arrowedLine(canvas, debug.sphere_seed_patch, debug.refined_patch,
                      cv::Scalar(120, 190, 120), 1, cv::LINE_AA, 0, 0.15);
    } else if (refined_ok) {
      cv::arrowedLine(canvas, debug.predicted_patch, debug.refined_patch, kArrowColor, 1,
                      cv::LINE_AA, 0, 0.15);
    }

    cv::drawMarker(canvas, debug.predicted_patch, cv::Scalar(255, 255, 255),
                   cv::MARKER_CROSS, 8, 3, cv::LINE_AA);
    cv::drawMarker(canvas, debug.predicted_patch, kPredictedColor,
                   cv::MARKER_CROSS, 6, 1, cv::LINE_AA);
    cv::circle(canvas, debug.predicted_patch, 2, cv::Scalar(255, 255, 255), cv::FILLED,
               cv::LINE_AA);
    cv::circle(canvas, debug.predicted_patch, 1, kPredictedColor, cv::FILLED, cv::LINE_AA);

    if (seed_ok) {
      cv::drawMarker(canvas, debug.sphere_seed_patch, cv::Scalar(255, 255, 255),
                     cv::MARKER_DIAMOND, 8, 3, cv::LINE_AA);
      cv::drawMarker(canvas, debug.sphere_seed_patch, kSeedColor,
                     cv::MARKER_DIAMOND, 6, 1, cv::LINE_AA);
    }

    if (refined_ok) {
      cv::drawMarker(canvas, debug.refined_patch, cv::Scalar(255, 255, 255),
                     cv::MARKER_SQUARE, 7, 3, cv::LINE_AA);
      cv::drawMarker(canvas, debug.refined_patch, kRefinedColor,
                     cv::MARKER_SQUARE, 5, 1, cv::LINE_AA);
    }
  }

  const std::string title =
      result.projection_mode == InternalProjectionMode::VirtualPinholeImageSubpix
          ? "Virtual-Pinhole Patch View: Ppatch -> Rimage(mapped back)"
          : result.projection_mode == InternalProjectionMode::VirtualPinholePatchBoundarySeed
                ? "Virtual-Pinhole Patch View: Ppatch -> SS(edge) -> Rimage(mapped back)"
          : "Virtual-Pinhole Patch View: P -> R";
  const std::string legend =
      result.projection_mode == InternalProjectionMode::VirtualPinholeImageSubpix
          ? "Legend: P orange cross, R green square, gray arrow: image-space refinement mapped to patch"
          : result.projection_mode == InternalProjectionMode::VirtualPinholePatchBoundarySeed
                ? "Legend: P orange cross, SS magenta diamond, R green square, gray arrow: patch seed path"
          : "Legend: P orange cross, R green square, gray arrow: patch refinement";
  cv::putText(canvas, title,
              cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 3,
              cv::LINE_AA);
  cv::putText(canvas, title,
              cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(30, 30, 30), 1,
              cv::LINE_AA);
  cv::putText(canvas, legend,
              cv::Point(18, 52), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 3,
              cv::LINE_AA);
  cv::putText(canvas, legend,
              cv::Point(18, 52), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(40, 40, 40), 1,
              cv::LINE_AA);
  return canvas;
}

cv::Mat BuildSideBySideComparisonCanvas(const cv::Mat& left,
                                        const cv::Mat& right,
                                        const std::string& left_title,
                                        const std::string& right_title,
                                        const std::string& title,
                                        const std::string& subtitle) {
  if (left.empty() || right.empty()) {
    return cv::Mat();
  }

  const cv::Mat left_tile = RenderTitledTile(left, left_title);
  const cv::Mat right_tile = RenderTitledTile(right, right_title);
  const int max_tile_height = std::max(left_tile.rows, right_tile.rows);
  std::vector<cv::Mat> tiles;
  tiles.push_back(PadTileToHeight(left_tile, max_tile_height));
  tiles.push_back(PadTileToHeight(right_tile, max_tile_height));

  cv::Mat row;
  cv::hconcat(tiles, row);

  const int header_height = 64;
  cv::Mat canvas(row.rows + header_height, row.cols, CV_8UC3, cv::Scalar(20, 20, 20));
  row.copyTo(canvas(cv::Rect(0, header_height, row.cols, row.rows)));
  cv::putText(canvas, title, cv::Point(14, 26), cv::FONT_HERSHEY_SIMPLEX, 0.72,
              cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
  cv::putText(canvas, subtitle, cv::Point(14, 50), cv::FONT_HERSHEY_SIMPLEX, 0.46,
              cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
  return canvas;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
