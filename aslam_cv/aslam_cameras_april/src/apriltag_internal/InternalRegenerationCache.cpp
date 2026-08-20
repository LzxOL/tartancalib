#include <aslam/cameras/apriltag_internal/InternalRegenerationCache.hpp>

#include <array>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

namespace fs = boost::filesystem;

constexpr const char kCacheFormatVersion[] =
      "internal_regeneration_cache_v36_outer_corner_debug";
constexpr const char kStageImplementationVersion[] =
      "internal_refinement_v36_outer_corner_debug";
constexpr const char kArtifactSchemaVersion[] =
    "internal_regeneration_frame_v2_outer_corner_debug";

std::uint64_t HashBytes(const std::string& text) {
  std::uint64_t hash = 1469598103934665603ull;
  for (unsigned char ch : text) {
    hash ^= static_cast<std::uint64_t>(ch);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string HashToHex(std::uint64_t value) {
  std::ostringstream stream;
  stream << std::hex << std::setw(16) << std::setfill('0') << value;
  return stream.str();
}

std::string AbsoluteImagePath(const std::string& image_path) {
  return fs::absolute(fs::path(image_path)).lexically_normal().string();
}

std::string VectorSignature(const std::vector<double>& values) {
  std::ostringstream stream;
  stream << std::setprecision(17);
  for (double value : values) {
    stream << value << ",";
  }
  return stream.str();
}

template <typename T>
void AppendScalar(std::ostringstream* stream, const char* name, const T& value) {
  *stream << name << "=" << value << "|";
}

void AppendBool(std::ostringstream* stream, const char* name, bool value) {
  AppendScalar(stream, name, value ? 1 : 0);
}

void AppendVector(std::ostringstream* stream,
                  const char* name,
                  const std::vector<double>& values) {
  *stream << name << "=[" << VectorSignature(values) << "]|";
}

void AppendVector(std::ostringstream* stream,
                  const char* name,
                  const std::vector<int>& values) {
  *stream << name << "=[";
  for (int value : values) {
    *stream << value << ",";
  }
  *stream << "]|";
}

void AppendMatrix(std::ostringstream* stream,
                  const char* name,
                  const Eigen::Matrix4d& matrix) {
  *stream << name << "=" << std::setprecision(17);
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      *stream << matrix(row, column) << ",";
    }
  }
  *stream << "|";
}

void AppendCamera(std::ostringstream* stream,
                  const OuterBootstrapCameraIntrinsics& camera) {
  *stream << "camera_model=" << camera.camera_model
          << "|distortion_model=" << camera.distortion_model << "|";
  AppendScalar(stream, "xi", camera.xi);
  AppendScalar(stream, "alpha", camera.alpha);
  AppendScalar(stream, "beta", camera.beta);
  AppendScalar(stream, "fu", camera.fu);
  AppendScalar(stream, "fv", camera.fv);
  AppendScalar(stream, "cu", camera.cu);
  AppendScalar(stream, "cv", camera.cv);
  AppendVector(stream, "distortion", camera.distortion_coeffs);
  AppendScalar(stream, "width", camera.resolution.width);
  AppendScalar(stream, "height", camera.resolution.height);
}

void AppendStateFrames(std::ostringstream* stream,
                       const std::vector<OuterBootstrapFrameState>& frames) {
  for (const OuterBootstrapFrameState& frame : frames) {
    AppendScalar(stream, "frame_index", frame.frame_index);
    *stream << "frame_label=" << frame.frame_label << "|";
    AppendBool(stream, "frame_initialized", frame.initialized);
    AppendVector(stream, "visible_board_ids", frame.visible_board_ids);
    AppendMatrix(stream, "T_camera_reference", frame.T_camera_reference);
    AppendScalar(stream, "frame_observation_count", frame.observation_count);
    AppendScalar(stream, "frame_rmse", frame.rmse);
  }
}

void AppendSceneFrames(std::ostringstream* stream,
                       const std::vector<JointSceneFrameState>& frames) {
  for (const JointSceneFrameState& frame : frames) {
    AppendScalar(stream, "frame_index", frame.frame_index);
    *stream << "frame_label=" << frame.frame_label << "|";
    AppendBool(stream, "frame_initialized", frame.initialized);
    AppendVector(stream, "visible_board_ids", frame.visible_board_ids);
    AppendMatrix(stream, "T_camera_reference", frame.T_camera_reference);
    AppendScalar(stream, "frame_observation_count", frame.observation_count);
    AppendScalar(stream, "frame_rmse", frame.rmse);
  }
}

void WritePoint2f(cv::FileStorage* storage,
                  const std::string& name,
                  const cv::Point2f& point) {
  *storage << name << "[" << point.x << point.y << "]";
}

cv::Point2f ReadPoint2f(const cv::FileNode& node) {
  cv::Point2f point;
  if (!node.empty() && node.isSeq() && node.size() >= 2) {
    point.x = static_cast<float>(node[0]);
    point.y = static_cast<float>(node[1]);
  }
  return point;
}

void WriteVec2d(cv::FileStorage* storage,
                const std::string& name,
                const Eigen::Vector2d& value) {
  *storage << name << "[" << value.x() << value.y() << "]";
}

Eigen::Vector2d ReadVec2d(const cv::FileNode& node) {
  Eigen::Vector2d value = Eigen::Vector2d::Zero();
  if (!node.empty() && node.isSeq() && node.size() >= 2) {
    value.x() = static_cast<double>(node[0]);
    value.y() = static_cast<double>(node[1]);
  }
  return value;
}

void WriteVec3d(cv::FileStorage* storage,
                const std::string& name,
                const Eigen::Vector3d& value) {
  *storage << name << "[" << value.x() << value.y() << value.z() << "]";
}

Eigen::Vector3d ReadVec3d(const cv::FileNode& node) {
  Eigen::Vector3d value = Eigen::Vector3d::Zero();
  if (!node.empty() && node.isSeq() && node.size() >= 3) {
    value.x() = static_cast<double>(node[0]);
    value.y() = static_cast<double>(node[1]);
    value.z() = static_cast<double>(node[2]);
  }
  return value;
}

void WriteVec3d(cv::FileStorage* storage,
                const std::string& name,
                const cv::Vec3d& value) {
  *storage << name << "[" << value[0] << value[1] << value[2] << "]";
}

cv::Vec3d ReadCvVec3d(const cv::FileNode& node) {
  cv::Vec3d value(0.0, 0.0, 0.0);
  if (!node.empty() && node.isSeq() && node.size() >= 3) {
    value[0] = static_cast<double>(node[0]);
    value[1] = static_cast<double>(node[1]);
    value[2] = static_cast<double>(node[2]);
  }
  return value;
}

template <typename T, std::size_t N, typename Writer>
void WriteArray(cv::FileStorage* storage,
                const std::string& name,
                const std::array<T, N>& values,
                Writer writer) {
  *storage << name << "[";
  for (const T& value : values) {
    writer(storage, value);
  }
  *storage << "]";
}

template <typename T, std::size_t N, typename Reader>
void ReadArray(const cv::FileNode& node,
               std::array<T, N>* values,
               Reader reader) {
  if (values == nullptr) {
    return;
  }
  values->fill(T{});
  if (node.empty() || !node.isSeq()) {
    return;
  }
  int index = 0;
  for (cv::FileNodeIterator it = node.begin();
       it != node.end() && index < static_cast<int>(N); ++it, ++index) {
    (*values)[static_cast<std::size_t>(index)] = reader(*it);
  }
}

void WriteBoolArray(cv::FileStorage* storage,
                    const std::string& name,
                    const std::array<bool, 4>& values) {
  *storage << name << "[";
  for (bool value : values) {
    *storage << (value ? 1 : 0);
  }
  *storage << "]";
}

void ReadBoolArray(const cv::FileNode& node, std::array<bool, 4>* values) {
  if (values == nullptr) {
    return;
  }
  values->fill(false);
  if (node.empty() || !node.isSeq()) {
    return;
  }
  int index = 0;
  for (cv::FileNodeIterator it = node.begin();
       it != node.end() && index < 4; ++it, ++index) {
    (*values)[static_cast<std::size_t>(index)] = static_cast<int>(*it) != 0;
  }
}

// The final regeneration cache owns the observations consumed by all Stage5
// diagnostics. Preserve the refinement trace here as well as in the outer
// cache, otherwise a cache hit silently turns every corner_index back to -1.
void WriteOuterCornerVerificationDebug(
    cv::FileStorage* storage,
    const OuterCornerVerificationDebugInfo& debug) {
  *storage << "{";
  *storage << "corner_index" << debug.corner_index;
  WritePoint2f(storage, "coarse_corner", debug.coarse_corner);
  WritePoint2f(storage, "verified_corner", debug.verified_corner);
  WritePoint2f(storage, "subpix_corner", debug.subpix_corner);
  *storage << "local_scale" << debug.local_scale;
  *storage << "verification_roi_radius" << debug.verification_roi_radius;
  *storage << "candidate_radius" << debug.candidate_radius;
  *storage << "branch_search_radius" << debug.branch_search_radius;
  *storage << "verification_quality" << debug.verification_quality;
  *storage << "coarse_to_verified_displacement"
           << debug.coarse_to_verified_displacement;
  *storage << "coarse_to_subpix_displacement"
           << debug.coarse_to_subpix_displacement;
  *storage << "coarse_to_refined_displacement"
           << debug.coarse_to_refined_displacement;
  *storage << "corner_marker_width" << debug.corner_marker_width;
  *storage << "configured_outer_subpix_scale"
           << debug.configured_outer_subpix_scale;
  *storage << "configured_outer_subpix_window_scale"
           << debug.configured_outer_subpix_window_scale;
  *storage << "configured_outer_subpix_window_radius"
           << debug.configured_outer_subpix_window_radius;
  *storage << "configured_outer_subpix_window_min"
           << debug.configured_outer_subpix_window_min;
  *storage << "configured_outer_subpix_window_max"
           << debug.configured_outer_subpix_window_max;
  *storage << "raw_subpix_window_radius" << debug.raw_subpix_window_radius;
  *storage << "pre_boost_subpix_window_radius"
           << debug.pre_boost_subpix_window_radius;
  *storage << "boosted_raw_subpix_window_radius"
           << debug.boosted_raw_subpix_window_radius;
  *storage << "subpix_window_clamp_limit" << debug.subpix_window_clamp_limit;
  *storage << "subpix_window_clamped" << (debug.subpix_window_clamped ? 1 : 0);
  *storage << "subpix_window_radius" << debug.subpix_window_radius;
  *storage << "subpix_unstable_rollback_detected"
           << (debug.subpix_unstable_rollback_detected ? 1 : 0);
  *storage << "subpix_unstable_rollback_iteration"
           << debug.subpix_unstable_rollback_iteration;
  *storage << "subpix_unstable_rollback_max_displacement"
           << debug.subpix_unstable_rollback_max_displacement;
  *storage << "close_edge_subpix_boost_applied"
           << (debug.close_edge_subpix_boost_applied ? 1 : 0);
  *storage << "close_edge_subpix_area_ratio"
           << debug.close_edge_subpix_area_ratio;
  *storage << "close_edge_subpix_max_polar_deg"
           << debug.close_edge_subpix_max_polar_deg;
  *storage << "close_edge_subpix_multiplier"
           << debug.close_edge_subpix_multiplier;
  *storage << "refine_displacement_limit" << debug.refine_displacement_limit;
  *storage << "refined_valid" << (debug.refined_valid ? 1 : 0);
  *storage << "verification_passed" << (debug.verification_passed ? 1 : 0);
  *storage << "subpix_applied" << (debug.subpix_applied ? 1 : 0);
  *storage << "failure_reason" << debug.failure_reason;
  *storage << "}";
}

void WriteOuterCornerVerificationDebugArray(
    cv::FileStorage* storage,
    const char* key,
    const std::array<OuterCornerVerificationDebugInfo, 4>& values) {
  *storage << key << "[";
  for (const OuterCornerVerificationDebugInfo& value : values) {
    WriteOuterCornerVerificationDebug(storage, value);
  }
  *storage << "]";
}

void WritePointVector(cv::FileStorage* storage,
                      const std::string& name,
                      const std::vector<cv::Point2f>& points) {
  if (!name.empty()) {
    *storage << name;
  }
  *storage << "[";
  for (const cv::Point2f& point : points) {
    *storage << "[" << point.x << point.y << "]";
  }
  *storage << "]";
}

std::vector<cv::Point2f> ReadPointVector(const cv::FileNode& node) {
  std::vector<cv::Point2f> points;
  if (node.empty() || !node.isSeq()) {
    return points;
  }
  for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it) {
    points.push_back(ReadPoint2f(*it));
  }
  return points;
}

void WriteVec3dVector(cv::FileStorage* storage,
                      const std::string& name,
                      const std::vector<cv::Vec3d>& values) {
  if (!name.empty()) {
    *storage << name;
  }
  *storage << "[";
  for (const cv::Vec3d& value : values) {
    *storage << "[" << value[0] << value[1] << value[2] << "]";
  }
  *storage << "]";
}

std::vector<cv::Vec3d> ReadVec3dVector(const cv::FileNode& node) {
  std::vector<cv::Vec3d> values;
  if (node.empty() || !node.isSeq()) {
    return values;
  }
  for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it) {
    values.push_back(ReadCvVec3d(*it));
  }
  return values;
}

void WriteStringVector(cv::FileStorage* storage,
                       const std::string& name,
                       const std::vector<std::string>& values) {
  *storage << name << "[";
  for (const std::string& value : values) {
    *storage << value;
  }
  *storage << "]";
}

std::vector<std::string> ReadStringVector(const cv::FileNode& node) {
  std::vector<std::string> values;
  if (node.empty() || !node.isSeq()) {
    return values;
  }
  for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it) {
    values.push_back(static_cast<std::string>(*it));
  }
  return values;
}

void WriteCornerMeasurement(cv::FileStorage* storage,
                            const CornerMeasurement& measurement) {
  *storage << "{";
  *storage << "board_id" << measurement.board_id;
  *storage << "point_id" << measurement.point_id;
  WriteVec2d(storage, "image_xy", measurement.image_xy);
  WriteVec3d(storage, "target_xyz", measurement.target_xyz);
  *storage << "valid" << (measurement.valid ? 1 : 0);
  *storage << "corner_type" << static_cast<int>(measurement.corner_type);
  *storage << "quality" << measurement.quality;
  *storage << "}";
}

CornerMeasurement ReadCornerMeasurement(const cv::FileNode& node) {
  CornerMeasurement measurement;
  measurement.board_id = static_cast<int>(node["board_id"]);
  measurement.point_id = static_cast<int>(node["point_id"]);
  measurement.image_xy = ReadVec2d(node["image_xy"]);
  measurement.target_xyz = ReadVec3d(node["target_xyz"]);
  measurement.valid = static_cast<int>(node["valid"]) != 0;
  measurement.corner_type =
      static_cast<CornerType>(static_cast<int>(node["corner_type"]));
  measurement.quality = static_cast<double>(node["quality"]);
  return measurement;
}

void WriteInternalCornerDebug(cv::FileStorage* storage,
                              const InternalCornerDebugInfo& debug) {
  *storage << "{";
  *storage << "point_id" << debug.point_id;
  *storage << "lattice_u" << debug.lattice_u;
  *storage << "lattice_v" << debug.lattice_v;
  *storage << "corner_type" << static_cast<int>(debug.corner_type);
  WritePoint2f(storage, "predicted_image", debug.predicted_image);
  WritePoint2f(storage, "border_seed_image", debug.border_seed_image);
  WritePoint2f(storage, "sphere_seed_image", debug.sphere_seed_image);
  WritePoint2f(storage, "structure_corrected_image", debug.structure_corrected_image);
  WritePoint2f(storage, "refined_image", debug.refined_image);
  WritePoint2f(storage, "predicted_patch", debug.predicted_patch);
  WritePoint2f(storage, "sphere_seed_patch", debug.sphere_seed_patch);
  WritePoint2f(storage, "refined_patch", debug.refined_patch);
  WriteVec3d(storage, "predicted_ray", debug.predicted_ray);
  WriteVec3d(storage, "sphere_seed_ray", debug.sphere_seed_ray);
  WriteVec3d(storage, "refined_ray", debug.refined_ray);
  *storage << "local_module_scale" << debug.local_module_scale;
  *storage << "sphere_search_radius" << debug.sphere_search_radius;
  *storage << "adaptive_search_radius" << debug.adaptive_search_radius;
  *storage << "sphere_template_quality" << debug.sphere_template_quality;
  *storage << "sphere_gradient_quality" << debug.sphere_gradient_quality;
  *storage << "sphere_seed_quality" << debug.sphere_seed_quality;
  *storage << "ray_refine_final_quality" << debug.ray_refine_final_quality;
  *storage << "ray_refine_iterations" << debug.ray_refine_iterations;
  *storage << "ray_refine_converged" << (debug.ray_refine_converged ? 1 : 0);
  *storage << "subpix_window_radius" << debug.subpix_window_radius;
  *storage << "subpix_displacement_limit" << debug.subpix_displacement_limit;
  *storage << "q_refine" << debug.q_refine;
  *storage << "template_quality" << debug.template_quality;
  *storage << "gradient_quality" << debug.gradient_quality;
  *storage << "final_quality" << debug.final_quality;
  *storage << "image_template_quality" << debug.image_template_quality;
  *storage << "image_gradient_quality" << debug.image_gradient_quality;
  *storage << "image_centering_quality" << debug.image_centering_quality;
  *storage << "image_final_quality" << debug.image_final_quality;
  *storage << "predicted_to_seed_displacement" << debug.predicted_to_seed_displacement;
  *storage << "seed_to_refined_displacement" << debug.seed_to_refined_displacement;
  *storage << "predicted_to_refined_displacement" << debug.predicted_to_refined_displacement;
  *storage << "border_seed_valid" << (debug.border_seed_valid ? 1 : 0);
  *storage << "structure_correction_valid" << (debug.structure_correction_valid ? 1 : 0);
  *storage << "forced_prediction_seed" << (debug.forced_prediction_seed ? 1 : 0);
  *storage << "bypass_seed_filters" << (debug.bypass_seed_filters ? 1 : 0);
  *storage << "original_seed_filter_success" << (debug.original_seed_filter_success ? 1 : 0);
  *storage << "original_seed_filter_would_reject" << (debug.original_seed_filter_would_reject ? 1 : 0);
  *storage << "image_refinement_applied" << (debug.image_refinement_applied ? 1 : 0);
  *storage << "valid" << (debug.valid ? 1 : 0);
  *storage << "image_evidence_valid" << (debug.image_evidence_valid ? 1 : 0);
  *storage << "}";
}

InternalCornerDebugInfo ReadInternalCornerDebug(const cv::FileNode& node) {
  InternalCornerDebugInfo debug;
  debug.point_id = static_cast<int>(node["point_id"]);
  debug.lattice_u = static_cast<int>(node["lattice_u"]);
  debug.lattice_v = static_cast<int>(node["lattice_v"]);
  debug.corner_type =
      static_cast<CornerType>(static_cast<int>(node["corner_type"]));
  debug.predicted_image = ReadPoint2f(node["predicted_image"]);
  debug.border_seed_image = ReadPoint2f(node["border_seed_image"]);
  debug.sphere_seed_image = ReadPoint2f(node["sphere_seed_image"]);
  debug.structure_corrected_image = ReadPoint2f(node["structure_corrected_image"]);
  debug.refined_image = ReadPoint2f(node["refined_image"]);
  debug.predicted_patch = ReadPoint2f(node["predicted_patch"]);
  debug.sphere_seed_patch = ReadPoint2f(node["sphere_seed_patch"]);
  debug.refined_patch = ReadPoint2f(node["refined_patch"]);
  debug.predicted_ray = ReadCvVec3d(node["predicted_ray"]);
  debug.sphere_seed_ray = ReadCvVec3d(node["sphere_seed_ray"]);
  debug.refined_ray = ReadCvVec3d(node["refined_ray"]);
  debug.local_module_scale = static_cast<double>(node["local_module_scale"]);
  debug.sphere_search_radius = static_cast<double>(node["sphere_search_radius"]);
  debug.adaptive_search_radius = static_cast<double>(node["adaptive_search_radius"]);
  debug.sphere_template_quality = static_cast<double>(node["sphere_template_quality"]);
  debug.sphere_gradient_quality = static_cast<double>(node["sphere_gradient_quality"]);
  debug.sphere_seed_quality = static_cast<double>(node["sphere_seed_quality"]);
  debug.ray_refine_final_quality = static_cast<double>(node["ray_refine_final_quality"]);
  debug.ray_refine_iterations = static_cast<int>(node["ray_refine_iterations"]);
  debug.ray_refine_converged = static_cast<int>(node["ray_refine_converged"]) != 0;
  debug.subpix_window_radius = static_cast<int>(node["subpix_window_radius"]);
  debug.subpix_displacement_limit = static_cast<double>(node["subpix_displacement_limit"]);
  debug.q_refine = static_cast<double>(node["q_refine"]);
  debug.template_quality = static_cast<double>(node["template_quality"]);
  debug.gradient_quality = static_cast<double>(node["gradient_quality"]);
  debug.final_quality = static_cast<double>(node["final_quality"]);
  debug.image_template_quality =
      static_cast<double>(node["image_template_quality"]);
  debug.image_gradient_quality =
      static_cast<double>(node["image_gradient_quality"]);
  debug.image_centering_quality =
      static_cast<double>(node["image_centering_quality"]);
  debug.image_final_quality =
      static_cast<double>(node["image_final_quality"]);
  debug.predicted_to_seed_displacement = static_cast<double>(node["predicted_to_seed_displacement"]);
  debug.seed_to_refined_displacement = static_cast<double>(node["seed_to_refined_displacement"]);
  debug.predicted_to_refined_displacement = static_cast<double>(node["predicted_to_refined_displacement"]);
  debug.border_seed_valid = static_cast<int>(node["border_seed_valid"]) != 0;
  debug.structure_correction_valid = static_cast<int>(node["structure_correction_valid"]) != 0;
  debug.forced_prediction_seed = static_cast<int>(node["forced_prediction_seed"]) != 0;
  debug.bypass_seed_filters = static_cast<int>(node["bypass_seed_filters"]) != 0;
  debug.original_seed_filter_success = static_cast<int>(node["original_seed_filter_success"]) != 0;
  debug.original_seed_filter_would_reject = static_cast<int>(node["original_seed_filter_would_reject"]) != 0;
  debug.image_refinement_applied = static_cast<int>(node["image_refinement_applied"]) != 0;
  debug.valid = static_cast<int>(node["valid"]) != 0;
  debug.image_evidence_valid = static_cast<int>(node["image_evidence_valid"]) != 0;
  return debug;
}

void WriteDetection(cv::FileStorage* storage,
                    const ApriltagInternalDetectionResult& detection) {
  *storage << "{";
  *storage << "success" << (detection.success ? 1 : 0);
  *storage << "reject_entire_board_observation"
           << (detection.reject_entire_board_observation ? 1 : 0);
  *storage << "tag_detected" << (detection.tag_detected ? 1 : 0);
  *storage << "board_id" << detection.board_id;
  *storage << "image_width" << detection.image_size.width;
  *storage << "image_height" << detection.image_size.height;
  *storage << "failure_reason" << detection.failure_reason;
  *storage << "internal_camera_source" << detection.internal_camera_source;
  *storage << "projection_mode" << static_cast<int>(detection.projection_mode);
  WritePoint2f(storage, "tag_center", detection.tag_center);
  *storage << "observed_perimeter" << detection.observed_perimeter;
  WriteArray(storage, "outer_corners", detection.outer_corners,
             [](cv::FileStorage* s, const cv::Point2f& p) {
               *s << "[" << p.x << p.y << "]";
             });
  WriteBoolArray(storage, "outer_corner_valid", detection.outer_corner_valid);
  WriteArray(storage, "patch_outer_corners", detection.patch_outer_corners,
             [](cv::FileStorage* s, const cv::Point2f& p) {
               *s << "[" << p.x << p.y << "]";
             });
  *storage << "corners" << "[";
  for (const CornerMeasurement& corner : detection.corners) {
    WriteCornerMeasurement(storage, corner);
  }
  *storage << "]";
  *storage << "internal_corner_debug" << "[";
  for (const InternalCornerDebugInfo& debug : detection.internal_corner_debug) {
    WriteInternalCornerDebug(storage, debug);
  }
  *storage << "]";
  *storage << "expected_visible_point_count" << detection.expected_visible_point_count;
  *storage << "valid_corner_count" << detection.valid_corner_count;
  *storage << "valid_internal_corner_count" << detection.valid_internal_corner_count;
  *storage << "pose_rescue_attempted" << (detection.pose_rescue_attempted ? 1 : 0);
  *storage << "pose_rescue_success" << (detection.pose_rescue_success ? 1 : 0);
  *storage << "pose_rescue_used" << (detection.pose_rescue_used ? 1 : 0);
  *storage << "pose_rescue_rmse" << detection.pose_rescue_rmse;
  *storage << "pose_rescue_max_ray_angle_deg" << detection.pose_rescue_max_ray_angle_deg;
  *storage << "pose_rescue_ray_angle_limit_deg" << detection.pose_rescue_ray_angle_limit_deg;
  *storage << "pose_rescue_failure_reason" << detection.pose_rescue_failure_reason;
  *storage << "border_boundary_model_valid" << (detection.border_boundary_model_valid ? 1 : 0);
  *storage << "border_boundary_model_failure_reason" << detection.border_boundary_model_failure_reason;
  WriteBoolArray(storage, "border_edge_valid", detection.border_edge_valid);
  *storage << "border_edge_rms_residual" << "[";
  for (double value : detection.border_edge_rms_residual) *storage << value;
  *storage << "]";
  *storage << "border_edge_support_count" << "[";
  for (int value : detection.border_edge_support_count) *storage << value;
  *storage << "]";
  *storage << "border_edge_support_ray_count" << "[";
  for (int value : detection.border_edge_support_ray_count) *storage << value;
  *storage << "]";
  *storage << "border_support_points" << "[";
  for (const auto& points : detection.border_support_points) {
    *storage << "[";
    for (const cv::Point2f& point : points) *storage << "[" << point.x << point.y << "]";
    *storage << "]";
  }
  *storage << "]";
  *storage << "border_curves_image" << "[";
  for (const auto& points : detection.border_curves_image) {
    WritePointVector(storage, "", points);
  }
  *storage << "]";
  *storage << "border_curves_ray" << "[";
  for (const auto& points : detection.border_curves_ray) {
    WriteVec3dVector(storage, "", points);
  }
  *storage << "]";
  if (!detection.canonical_patch.empty()) {
    *storage << "canonical_patch" << detection.canonical_patch;
  }
  const ApriltagInternalRuntimeBreakdown& runtime = detection.runtime_breakdown;
  *storage << "runtime_total_seconds" << runtime.total_seconds;
  *storage << "runtime_pose_estimation_seconds" << runtime.pose_estimation_seconds;
  *storage << "runtime_boundary_model_seconds" << runtime.boundary_model_seconds;
  *storage << "runtime_seed_search_seconds" << runtime.seed_search_seconds;
  *storage << "runtime_ray_refine_seconds" << runtime.ray_refine_seconds;
  *storage << "runtime_image_evidence_seconds" << runtime.image_evidence_seconds;
  *storage << "runtime_subpix_seconds" << runtime.subpix_seconds;
  *storage << "runtime_attempted_internal_corner_count" << runtime.attempted_internal_corner_count;
  *storage << "runtime_valid_internal_corner_count" << runtime.valid_internal_corner_count;
  // The detector may replace a missing outer observation with a geometry-
  // validated rescue. Preserve that effective outer result in the internal
  // cache; reattaching only the raw frame input loses the rescue on cache hits.
  *storage << "outer_detection" << "{";
  const OuterTagDetectionResult& outer = detection.outer_detection;
  *storage << "success" << (outer.success ? 1 : 0);
  *storage << "board_id" << outer.board_id;
  *storage << "detected_tag_id" << outer.detected_tag_id;
  *storage << "original_longest_side" << outer.original_longest_side;
  *storage << "chosen_scale_longest_side" << outer.chosen_scale_longest_side;
  *storage << "chosen_scale_factor" << outer.chosen_scale_factor;
  *storage << "scale_configuration_mode" << outer.scale_configuration_mode;
  *storage << "hamming" << outer.hamming;
  *storage << "good" << (outer.good ? 1 : 0);
  *storage << "attempted_local_patch_rescue"
           << (outer.attempted_local_patch_rescue ? 1 : 0);
  *storage << "used_local_patch_rescue"
           << (outer.used_local_patch_rescue ? 1 : 0);
  *storage << "local_patch_rescue_summary" << outer.local_patch_rescue_summary;
  *storage << "quality" << outer.quality;
  *storage << "failure_reason" << static_cast<int>(outer.failure_reason);
  *storage << "failure_reason_text" << outer.failure_reason_text;
  WriteArray(storage, "coarse_corners_scaled_image", outer.coarse_corners_scaled_image,
             [](cv::FileStorage* s, const Eigen::Vector2d& point) {
               *s << "[" << point.x() << point.y() << "]";
             });
  WriteArray(storage, "coarse_corners_original_image", outer.coarse_corners_original_image,
             [](cv::FileStorage* s, const Eigen::Vector2d& point) {
               *s << "[" << point.x() << point.y() << "]";
             });
  WriteArray(storage, "refined_corners_original_image", outer.refined_corners_original_image,
             [](cv::FileStorage* s, const Eigen::Vector2d& point) {
               *s << "[" << point.x() << point.y() << "]";
             });
  WriteBoolArray(storage, "refined_valid", outer.refined_valid);
  *storage << "board_quad_consistency_checked"
           << (outer.board_quad_consistency_checked ? 1 : 0);
  *storage << "board_quad_consistency_passed"
           << (outer.board_quad_consistency_passed ? 1 : 0);
  *storage << "board_quad_worst_corner_index" << outer.board_quad_worst_corner_index;
  *storage << "board_quad_worst_corner_displacement_px"
           << outer.board_quad_worst_corner_displacement_px;
  *storage << "board_quad_area_ratio" << outer.board_quad_area_ratio;
  *storage << "board_quad_consistency_diagnostic"
           << outer.board_quad_consistency_diagnostic;
  WriteOuterCornerVerificationDebugArray(
      storage, "corner_verification_debug", outer.corner_verification_debug);
  *storage << "}";
  *storage << "}";
}

template <typename T>
T ReadScalar(const cv::FileNode& node, const char* key, const T& fallback) {
  const cv::FileNode value = node[key];
  return value.empty() ? fallback : static_cast<T>(value);
}

void ReadOuterCornerVerificationDebug(
    const cv::FileNode& node,
    OuterCornerVerificationDebugInfo* debug) {
  if (debug == nullptr) {
    return;
  }
  *debug = OuterCornerVerificationDebugInfo{};
  if (node.empty() || !node.isMap()) {
    return;
  }
  debug->corner_index = ReadScalar<int>(node, "corner_index", -1);
  debug->coarse_corner = ReadPoint2f(node["coarse_corner"]);
  debug->verified_corner = ReadPoint2f(node["verified_corner"]);
  debug->subpix_corner = ReadPoint2f(node["subpix_corner"]);
  debug->local_scale = ReadScalar<double>(node, "local_scale", 0.0);
  debug->verification_roi_radius =
      ReadScalar<int>(node, "verification_roi_radius", 0);
  debug->candidate_radius = ReadScalar<int>(node, "candidate_radius", 0);
  debug->branch_search_radius =
      ReadScalar<int>(node, "branch_search_radius", 0);
  debug->verification_quality =
      ReadScalar<double>(node, "verification_quality", 0.0);
  debug->coarse_to_verified_displacement =
      ReadScalar<double>(node, "coarse_to_verified_displacement", 0.0);
  debug->coarse_to_subpix_displacement =
      ReadScalar<double>(node, "coarse_to_subpix_displacement", 0.0);
  debug->coarse_to_refined_displacement =
      ReadScalar<double>(node, "coarse_to_refined_displacement", 0.0);
  debug->corner_marker_width =
      ReadScalar<double>(node, "corner_marker_width", 0.0);
  debug->configured_outer_subpix_scale =
      ReadScalar<double>(node, "configured_outer_subpix_scale", 0.0);
  debug->configured_outer_subpix_window_scale =
      ReadScalar<double>(node, "configured_outer_subpix_window_scale", 0.0);
  debug->configured_outer_subpix_window_radius =
      ReadScalar<int>(node, "configured_outer_subpix_window_radius", 0);
  debug->configured_outer_subpix_window_min =
      ReadScalar<int>(node, "configured_outer_subpix_window_min", 0);
  debug->configured_outer_subpix_window_max =
      ReadScalar<int>(node, "configured_outer_subpix_window_max", 0);
  debug->raw_subpix_window_radius =
      ReadScalar<int>(node, "raw_subpix_window_radius", 0);
  debug->pre_boost_subpix_window_radius =
      ReadScalar<int>(node, "pre_boost_subpix_window_radius", 0);
  debug->boosted_raw_subpix_window_radius =
      ReadScalar<int>(node, "boosted_raw_subpix_window_radius", 0);
  debug->subpix_window_clamp_limit =
      ReadScalar<int>(node, "subpix_window_clamp_limit", 0);
  debug->subpix_window_clamped =
      ReadScalar<int>(node, "subpix_window_clamped", 0) != 0;
  debug->subpix_window_radius =
      ReadScalar<int>(node, "subpix_window_radius", 0);
  debug->subpix_unstable_rollback_detected =
      ReadScalar<int>(node, "subpix_unstable_rollback_detected", 0) != 0;
  debug->subpix_unstable_rollback_iteration =
      ReadScalar<int>(node, "subpix_unstable_rollback_iteration", 0);
  debug->subpix_unstable_rollback_max_displacement = ReadScalar<double>(
      node, "subpix_unstable_rollback_max_displacement", 0.0);
  debug->close_edge_subpix_boost_applied =
      ReadScalar<int>(node, "close_edge_subpix_boost_applied", 0) != 0;
  debug->close_edge_subpix_area_ratio =
      ReadScalar<double>(node, "close_edge_subpix_area_ratio", 0.0);
  debug->close_edge_subpix_max_polar_deg =
      ReadScalar<double>(node, "close_edge_subpix_max_polar_deg", 0.0);
  debug->close_edge_subpix_multiplier =
      ReadScalar<double>(node, "close_edge_subpix_multiplier", 1.0);
  debug->refine_displacement_limit =
      ReadScalar<double>(node, "refine_displacement_limit", 0.0);
  debug->refined_valid =
      ReadScalar<int>(node, "refined_valid", 0) != 0;
  debug->verification_passed =
      ReadScalar<int>(node, "verification_passed", 0) != 0;
  debug->subpix_applied =
      ReadScalar<int>(node, "subpix_applied", 0) != 0;
  debug->failure_reason =
      ReadScalar<std::string>(node, "failure_reason", "");
}

void ReadOuterCornerVerificationDebugArray(
    const cv::FileNode& node,
    std::array<OuterCornerVerificationDebugInfo, 4>* values) {
  if (values == nullptr) {
    return;
  }
  values->fill(OuterCornerVerificationDebugInfo{});
  if (node.empty() || !node.isSeq()) {
    return;
  }
  int index = 0;
  for (cv::FileNodeIterator it = node.begin();
       it != node.end() && index < 4; ++it, ++index) {
    ReadOuterCornerVerificationDebug(*it,
                                     &(*values)[static_cast<std::size_t>(index)]);
  }
}

void ReadDetection(const cv::FileNode& node,
                   ApriltagInternalDetectionResult* detection) {
  if (detection == nullptr) {
    return;
  }
  detection->success = ReadScalar<int>(node, "success", 0) != 0;
  detection->reject_entire_board_observation =
      ReadScalar<int>(node, "reject_entire_board_observation", 0) != 0;
  detection->tag_detected = ReadScalar<int>(node, "tag_detected", 0) != 0;
  detection->board_id = ReadScalar<int>(node, "board_id", -1);
  detection->image_size.width = ReadScalar<int>(node, "image_width", 0);
  detection->image_size.height = ReadScalar<int>(node, "image_height", 0);
  detection->failure_reason = ReadScalar<std::string>(node, "failure_reason", "");
  detection->internal_camera_source = ReadScalar<std::string>(node, "internal_camera_source", "");
  detection->projection_mode = static_cast<InternalProjectionMode>(
      ReadScalar<int>(node, "projection_mode", static_cast<int>(InternalProjectionMode::Homography)));
  detection->tag_center = ReadPoint2f(node["tag_center"]);
  detection->observed_perimeter = ReadScalar<float>(node, "observed_perimeter", 0.0f);
  ReadArray(node["outer_corners"], &detection->outer_corners,
            [](const cv::FileNode& value) { return ReadPoint2f(value); });
  ReadBoolArray(node["outer_corner_valid"], &detection->outer_corner_valid);
  ReadArray(node["patch_outer_corners"], &detection->patch_outer_corners,
            [](const cv::FileNode& value) { return ReadPoint2f(value); });
  detection->corners.clear();
  const cv::FileNode corners = node["corners"];
  if (!corners.empty() && corners.isSeq()) {
    for (cv::FileNodeIterator it = corners.begin(); it != corners.end(); ++it) {
      detection->corners.push_back(ReadCornerMeasurement(*it));
    }
  }
  detection->internal_corner_debug.clear();
  const cv::FileNode debug = node["internal_corner_debug"];
  if (!debug.empty() && debug.isSeq()) {
    for (cv::FileNodeIterator it = debug.begin(); it != debug.end(); ++it) {
      detection->internal_corner_debug.push_back(ReadInternalCornerDebug(*it));
    }
  }
  detection->expected_visible_point_count = ReadScalar<int>(node, "expected_visible_point_count", 0);
  detection->valid_corner_count = ReadScalar<int>(node, "valid_corner_count", 0);
  detection->valid_internal_corner_count = ReadScalar<int>(node, "valid_internal_corner_count", 0);
  detection->pose_rescue_attempted = ReadScalar<int>(node, "pose_rescue_attempted", 0) != 0;
  detection->pose_rescue_success = ReadScalar<int>(node, "pose_rescue_success", 0) != 0;
  detection->pose_rescue_used = ReadScalar<int>(node, "pose_rescue_used", 0) != 0;
  detection->pose_rescue_rmse = ReadScalar<double>(node, "pose_rescue_rmse", 0.0);
  detection->pose_rescue_max_ray_angle_deg = ReadScalar<double>(node, "pose_rescue_max_ray_angle_deg", 0.0);
  detection->pose_rescue_ray_angle_limit_deg = ReadScalar<double>(node, "pose_rescue_ray_angle_limit_deg", 0.0);
  detection->pose_rescue_failure_reason = ReadScalar<std::string>(node, "pose_rescue_failure_reason", "");
  detection->border_boundary_model_valid = ReadScalar<int>(node, "border_boundary_model_valid", 0) != 0;
  detection->border_boundary_model_failure_reason = ReadScalar<std::string>(node, "border_boundary_model_failure_reason", "");
  ReadBoolArray(node["border_edge_valid"], &detection->border_edge_valid);
  const cv::FileNode edge_rms = node["border_edge_rms_residual"];
  if (!edge_rms.empty() && edge_rms.isSeq()) {
    int index = 0;
    for (cv::FileNodeIterator it = edge_rms.begin(); it != edge_rms.end() && index < 4; ++it, ++index) {
      detection->border_edge_rms_residual[static_cast<std::size_t>(index)] = static_cast<double>(*it);
    }
  }
  const cv::FileNode support_count = node["border_edge_support_count"];
  if (!support_count.empty() && support_count.isSeq()) {
    int index = 0;
    for (cv::FileNodeIterator it = support_count.begin(); it != support_count.end() && index < 4; ++it, ++index) {
      detection->border_edge_support_count[static_cast<std::size_t>(index)] = static_cast<int>(*it);
    }
  }
  const cv::FileNode support_ray_count = node["border_edge_support_ray_count"];
  if (!support_ray_count.empty() && support_ray_count.isSeq()) {
    int index = 0;
    for (cv::FileNodeIterator it = support_ray_count.begin(); it != support_ray_count.end() && index < 4; ++it, ++index) {
      detection->border_edge_support_ray_count[static_cast<std::size_t>(index)] = static_cast<int>(*it);
    }
  }
  const cv::FileNode support_points = node["border_support_points"];
  if (!support_points.empty() && support_points.isSeq()) {
    int edge = 0;
    for (cv::FileNodeIterator edge_it = support_points.begin();
         edge_it != support_points.end() && edge < 4; ++edge_it, ++edge) {
      detection->border_support_points[static_cast<std::size_t>(edge)] = ReadPointVector(*edge_it);
    }
  }
  const cv::FileNode curves_image = node["border_curves_image"];
  if (!curves_image.empty() && curves_image.isSeq()) {
    int edge = 0;
    for (cv::FileNodeIterator edge_it = curves_image.begin();
         edge_it != curves_image.end() && edge < 4; ++edge_it, ++edge) {
      detection->border_curves_image[static_cast<std::size_t>(edge)] =
          ReadPointVector(*edge_it);
    }
  }
  const cv::FileNode curves_ray = node["border_curves_ray"];
  if (!curves_ray.empty() && curves_ray.isSeq()) {
    int edge = 0;
    for (cv::FileNodeIterator edge_it = curves_ray.begin();
         edge_it != curves_ray.end() && edge < 4; ++edge_it, ++edge) {
      detection->border_curves_ray[static_cast<std::size_t>(edge)] =
          ReadVec3dVector(*edge_it);
    }
  }
  const cv::FileNode canonical_patch = node["canonical_patch"];
  if (!canonical_patch.empty()) {
    canonical_patch >> detection->canonical_patch;
  }
  ApriltagInternalRuntimeBreakdown& runtime = detection->runtime_breakdown;
  runtime.total_seconds = ReadScalar<double>(node, "runtime_total_seconds", 0.0);
  runtime.pose_estimation_seconds = ReadScalar<double>(node, "runtime_pose_estimation_seconds", 0.0);
  runtime.boundary_model_seconds = ReadScalar<double>(node, "runtime_boundary_model_seconds", 0.0);
  runtime.seed_search_seconds = ReadScalar<double>(node, "runtime_seed_search_seconds", 0.0);
  runtime.ray_refine_seconds = ReadScalar<double>(node, "runtime_ray_refine_seconds", 0.0);
  runtime.image_evidence_seconds = ReadScalar<double>(node, "runtime_image_evidence_seconds", 0.0);
  runtime.subpix_seconds = ReadScalar<double>(node, "runtime_subpix_seconds", 0.0);
  runtime.attempted_internal_corner_count = ReadScalar<int>(node, "runtime_attempted_internal_corner_count", 0);
  runtime.valid_internal_corner_count = ReadScalar<int>(node, "runtime_valid_internal_corner_count", 0);
  const cv::FileNode outer_node = node["outer_detection"];
  if (!outer_node.empty() && outer_node.isMap()) {
    OuterTagDetectionResult& outer = detection->outer_detection;
    outer.success = ReadScalar<int>(outer_node, "success", 0) != 0;
    outer.board_id = ReadScalar<int>(outer_node, "board_id", -1);
    outer.detected_tag_id = ReadScalar<int>(outer_node, "detected_tag_id", -1);
    outer.original_longest_side = ReadScalar<int>(outer_node, "original_longest_side", 0);
    outer.chosen_scale_longest_side =
        ReadScalar<int>(outer_node, "chosen_scale_longest_side", 0);
    outer.chosen_scale_factor =
        ReadScalar<double>(outer_node, "chosen_scale_factor", 1.0);
    outer.scale_configuration_mode =
        ReadScalar<std::string>(outer_node, "scale_configuration_mode", "");
    outer.hamming = ReadScalar<int>(outer_node, "hamming", -1);
    outer.good = ReadScalar<int>(outer_node, "good", 0) != 0;
    outer.attempted_local_patch_rescue =
        ReadScalar<int>(outer_node, "attempted_local_patch_rescue", 0) != 0;
    outer.used_local_patch_rescue =
        ReadScalar<int>(outer_node, "used_local_patch_rescue", 0) != 0;
    outer.local_patch_rescue_summary =
        ReadScalar<std::string>(outer_node, "local_patch_rescue_summary", "");
    outer.quality = ReadScalar<double>(outer_node, "quality", 0.0);
    outer.failure_reason = static_cast<OuterTagFailureReason>(
        ReadScalar<int>(outer_node, "failure_reason",
                        static_cast<int>(OuterTagFailureReason::NoDetectionsAtAll)));
    outer.failure_reason_text =
        ReadScalar<std::string>(outer_node, "failure_reason_text", "");
    ReadArray(outer_node["coarse_corners_scaled_image"],
              &outer.coarse_corners_scaled_image,
              [](const cv::FileNode& value) { return ReadVec2d(value); });
    ReadArray(outer_node["coarse_corners_original_image"],
              &outer.coarse_corners_original_image,
              [](const cv::FileNode& value) { return ReadVec2d(value); });
    ReadArray(outer_node["refined_corners_original_image"],
              &outer.refined_corners_original_image,
              [](const cv::FileNode& value) { return ReadVec2d(value); });
    ReadBoolArray(outer_node["refined_valid"], &outer.refined_valid);
    outer.board_quad_consistency_checked =
        ReadScalar<int>(outer_node, "board_quad_consistency_checked", 0) != 0;
    outer.board_quad_consistency_passed =
        ReadScalar<int>(outer_node, "board_quad_consistency_passed", 0) != 0;
    outer.board_quad_worst_corner_index =
        ReadScalar<int>(outer_node, "board_quad_worst_corner_index", -1);
    outer.board_quad_worst_corner_displacement_px =
        ReadScalar<double>(outer_node, "board_quad_worst_corner_displacement_px", 0.0);
    outer.board_quad_area_ratio =
        ReadScalar<double>(outer_node, "board_quad_area_ratio", 0.0);
    outer.board_quad_consistency_diagnostic =
        ReadScalar<std::string>(outer_node, "board_quad_consistency_diagnostic", "");
    ReadOuterCornerVerificationDebugArray(
        outer_node["corner_verification_debug"],
        &outer.corner_verification_debug);
  }
}

void WriteFrameResult(cv::FileStorage* storage,
                      const InternalRegenerationFrameResult& result) {
  *storage << "frame_result" << "{";
  *storage << "frame_index" << result.frame_index;
  *storage << "frame_label" << result.frame_label;
  *storage << "frame_bootstrap_initialized" << (result.frame_bootstrap_initialized ? 1 : 0);
  *storage << "state_source_label" << result.state_source_label;
  *storage << "image_width" << result.image_size.width;
  *storage << "image_height" << result.image_size.height;
  *storage << "visible_board_ids" << "[";
  for (int board_id : result.visible_board_ids) *storage << board_id;
  *storage << "]";
  WriteStringVector(storage, "warnings", result.warnings);
  *storage << "board_measurements" << "[";
  for (const RegeneratedBoardMeasurement& measurement : result.board_measurements) {
    *storage << "{";
    *storage << "board_id" << measurement.board_id;
    *storage << "frame_bootstrap_initialized" << (measurement.frame_bootstrap_initialized ? 1 : 0);
    *storage << "board_bootstrap_initialized" << (measurement.board_bootstrap_initialized ? 1 : 0);
    *storage << "pose_prior_used" << (measurement.pose_prior_used ? 1 : 0);
    *storage << "detection";
    WriteDetection(storage, measurement.detection);
    *storage << "}";
  }
  *storage << "]";
  const InternalRegenerationRuntimeBreakdown& runtime = result.runtime_breakdown;
  *storage << "runtime_pose_estimation_seconds" << runtime.pose_estimation_seconds;
  *storage << "runtime_boundary_model_seconds" << runtime.boundary_model_seconds;
  *storage << "runtime_seed_search_seconds" << runtime.seed_search_seconds;
  *storage << "runtime_ray_refine_seconds" << runtime.ray_refine_seconds;
  *storage << "runtime_image_evidence_seconds" << runtime.image_evidence_seconds;
  *storage << "runtime_subpix_seconds" << runtime.subpix_seconds;
  *storage << "runtime_pose_estimation_call_count" << runtime.pose_estimation_call_count;
  *storage << "runtime_pose_rescue_attempt_count" << runtime.pose_rescue_attempt_count;
  *storage << "runtime_pose_rescue_success_count" << runtime.pose_rescue_success_count;
  *storage << "runtime_pose_rescue_used_count" << runtime.pose_rescue_used_count;
  *storage << "runtime_boundary_model_build_count" << runtime.boundary_model_build_count;
  *storage << "runtime_attempted_internal_corner_count" << runtime.attempted_internal_corner_count;
  *storage << "runtime_valid_internal_corner_count" << runtime.valid_internal_corner_count;
  *storage << "}";
}

bool ReadFrameResult(const cv::FileNode& root,
                     InternalRegenerationFrameResult* result) {
  if (result == nullptr) {
    return false;
  }
  const cv::FileNode node = root["frame_result"];
  if (node.empty() || !node.isMap()) {
    return false;
  }
  result->frame_index = ReadScalar<int>(node, "frame_index", -1);
  result->frame_label = ReadScalar<std::string>(node, "frame_label", "");
  result->frame_bootstrap_initialized = ReadScalar<int>(node, "frame_bootstrap_initialized", 0) != 0;
  result->state_source_label = ReadScalar<std::string>(node, "state_source_label", "cache");
  result->image_size.width = ReadScalar<int>(node, "image_width", 0);
  result->image_size.height = ReadScalar<int>(node, "image_height", 0);
  result->visible_board_ids.clear();
  const cv::FileNode visible = node["visible_board_ids"];
  if (!visible.empty() && visible.isSeq()) {
    for (cv::FileNodeIterator it = visible.begin(); it != visible.end(); ++it) {
      result->visible_board_ids.push_back(static_cast<int>(*it));
    }
  }
  result->warnings = ReadStringVector(node["warnings"]);
  result->board_measurements.clear();
  const cv::FileNode boards = node["board_measurements"];
  if (!boards.empty() && boards.isSeq()) {
    for (cv::FileNodeIterator it = boards.begin(); it != boards.end(); ++it) {
      RegeneratedBoardMeasurement measurement;
      measurement.board_id = ReadScalar<int>(*it, "board_id", -1);
      measurement.frame_bootstrap_initialized = ReadScalar<int>(*it, "frame_bootstrap_initialized", 0) != 0;
      measurement.board_bootstrap_initialized = ReadScalar<int>(*it, "board_bootstrap_initialized", 0) != 0;
      measurement.pose_prior_used = ReadScalar<int>(*it, "pose_prior_used", 0) != 0;
      ReadDetection((*it)["detection"], &measurement.detection);
      result->board_measurements.push_back(std::move(measurement));
    }
  }
  InternalRegenerationRuntimeBreakdown& runtime = result->runtime_breakdown;
  runtime.pose_estimation_seconds = ReadScalar<double>(node, "runtime_pose_estimation_seconds", 0.0);
  runtime.boundary_model_seconds = ReadScalar<double>(node, "runtime_boundary_model_seconds", 0.0);
  runtime.seed_search_seconds = ReadScalar<double>(node, "runtime_seed_search_seconds", 0.0);
  runtime.ray_refine_seconds = ReadScalar<double>(node, "runtime_ray_refine_seconds", 0.0);
  runtime.image_evidence_seconds = ReadScalar<double>(node, "runtime_image_evidence_seconds", 0.0);
  runtime.subpix_seconds = ReadScalar<double>(node, "runtime_subpix_seconds", 0.0);
  runtime.pose_estimation_call_count = ReadScalar<int>(node, "runtime_pose_estimation_call_count", 0);
  runtime.pose_rescue_attempt_count = ReadScalar<int>(node, "runtime_pose_rescue_attempt_count", 0);
  runtime.pose_rescue_success_count = ReadScalar<int>(node, "runtime_pose_rescue_success_count", 0);
  runtime.pose_rescue_used_count = ReadScalar<int>(node, "runtime_pose_rescue_used_count", 0);
  runtime.boundary_model_build_count = ReadScalar<int>(node, "runtime_boundary_model_build_count", 0);
  runtime.attempted_internal_corner_count = ReadScalar<int>(node, "runtime_attempted_internal_corner_count", 0);
  runtime.valid_internal_corner_count = ReadScalar<int>(node, "runtime_valid_internal_corner_count", 0);
  // Compatibility with the first cache artifacts, which stored the point
  // measurements but not the frame-level runtime counters.
  if (runtime.attempted_internal_corner_count == 0 &&
      runtime.valid_internal_corner_count == 0) {
    for (const RegeneratedBoardMeasurement& measurement : result->board_measurements) {
      runtime.attempted_internal_corner_count +=
          static_cast<int>(measurement.detection.corners.size());
      for (const CornerMeasurement& corner : measurement.detection.corners) {
        runtime.valid_internal_corner_count += corner.valid ? 1 : 0;
      }
    }
  }
  return true;
}

std::string MakeImageKey(const std::string& image_path,
                         const std::string& outer_signature,
                         const std::string& state_signature) {
  const fs::path path(AbsoluteImagePath(image_path));
  std::ostringstream stream;
  stream << path.string() << "|size=" << fs::file_size(path)
         << "|mtime=" << static_cast<long long>(fs::last_write_time(path))
         << "|outer=" << outer_signature << "|state=" << state_signature;
  return HashToHex(HashBytes(stream.str()));
}

std::string MakeInternalConfigSignature(
    const ApriltagInternalConfig& config,
    const ApriltagInternalDetectionOptions& options) {
  std::ostringstream stream;
  stream << std::setprecision(17);
  stream << "format=" << kCacheFormatVersion << "|";
  AppendScalar(&stream, "target_type", config.target_type);
  AppendScalar(&stream, "tag_family", config.tag_family);
  AppendScalar(&stream, "tag_id", config.tag_id);
  AppendVector(&stream, "tag_ids", config.tag_ids);
  AppendScalar(&stream, "tag_size", config.tag_size);
  AppendScalar(&stream, "black_border_bits", config.black_border_bits);
  AppendScalar(&stream, "min_visible_points", config.min_visible_points);
  AppendScalar(&stream, "canonical_pixels_per_module", config.canonical_pixels_per_module);
  AppendScalar(&stream, "refinement_window_radius", config.refinement_window_radius);
  AppendScalar(&stream, "internal_subpix_window_scale", config.internal_subpix_window_scale);
  AppendScalar(&stream, "internal_subpix_window_min", config.internal_subpix_window_min);
  AppendScalar(&stream, "internal_subpix_window_max", config.internal_subpix_window_max);
  AppendScalar(&stream, "max_subpix_displacement2", config.max_subpix_displacement2);
  AppendScalar(&stream, "internal_subpix_displacement_scale", config.internal_subpix_displacement_scale);
  AppendScalar(&stream, "max_internal_subpix_displacement", config.max_internal_subpix_displacement);
  AppendBool(&stream, "ignore_image_evidence_min_quality", config.ignore_image_evidence_min_quality);
  AppendBool(&stream, "force_internal_seed_from_prediction", config.force_internal_seed_from_prediction);
  AppendBool(&stream, "bypass_internal_seed_filters", config.bypass_internal_seed_filters);
  AppendBool(&stream, "enable_internal_structure_correction_after_ss", config.enable_internal_structure_correction_after_ss);
  AppendScalar(&stream, "internal_projection_mode", static_cast<int>(config.internal_projection_mode));
  AppendBool(&stream, "sphere_lattice_use_initial_camera", config.sphere_lattice_use_initial_camera);
  AppendBool(&stream, "outer_spherical_use_initial_camera", config.outer_spherical_use_initial_camera);
  AppendBool(&stream, "sphere_lattice_enable_seed_search", config.sphere_lattice_enable_seed_search);
  AppendScalar(&stream, "sphere_lattice_init_xi", config.sphere_lattice_init_xi);
  AppendScalar(&stream, "sphere_lattice_init_alpha", config.sphere_lattice_init_alpha);
  AppendScalar(&stream, "sphere_lattice_init_fu_scale", config.sphere_lattice_init_fu_scale);
  AppendScalar(&stream, "sphere_lattice_init_fv_scale", config.sphere_lattice_init_fv_scale);
  AppendScalar(&stream, "sphere_lattice_init_cu_offset", config.sphere_lattice_init_cu_offset);
  AppendScalar(&stream, "sphere_lattice_init_cv_offset", config.sphere_lattice_init_cv_offset);
  AppendScalar(&stream, "camera_initialization_mode", static_cast<int>(config.camera_initialization_mode));
  AppendScalar(&stream, "intermediate_camera_model", config.intermediate_camera.camera_model);
  AppendScalar(&stream, "intermediate_distortion_model", config.intermediate_camera.distortion_model);
  AppendVector(&stream, "intermediate_intrinsics", config.intermediate_camera.intrinsics);
  AppendVector(&stream, "intermediate_distortion", config.intermediate_camera.distortion_coeffs);
  AppendVector(&stream, "intermediate_resolution", config.intermediate_camera.resolution);
  AppendBool(&stream, "outer_enable_robust_missing_board_recovery",
             config.outer_detector_config.enable_robust_missing_board_recovery);

  AppendBool(&stream, "do_subpix_refinement", options.do_subpix_refinement);
  AppendScalar(&stream, "options_max_subpix_displacement2", options.max_subpix_displacement2);
  AppendBool(&stream, "reject_duplicate_ids", options.reject_duplicate_ids);
  AppendScalar(&stream, "min_border_distance", options.min_border_distance);
  AppendScalar(&stream, "options_canonical_pixels_per_module", options.canonical_pixels_per_module);
  AppendScalar(&stream, "options_refinement_window_radius", options.refinement_window_radius);
  AppendScalar(&stream, "options_internal_subpix_window_scale", options.internal_subpix_window_scale);
  AppendScalar(&stream, "options_internal_subpix_window_min", options.internal_subpix_window_min);
  AppendScalar(&stream, "options_internal_subpix_window_max", options.internal_subpix_window_max);
  AppendScalar(&stream, "min_quality", options.min_quality);
  AppendScalar(&stream, "min_template_contrast", options.min_template_contrast);
  AppendScalar(&stream, "virtual_patch_margin", options.virtual_patch_margin);
  AppendScalar(&stream, "options_internal_subpix_displacement_scale", options.internal_subpix_displacement_scale);
  AppendScalar(&stream, "options_max_internal_subpix_displacement", options.max_internal_subpix_displacement);
  AppendBool(&stream, "options_ignore_image_evidence_min_quality", options.ignore_image_evidence_min_quality);
  AppendBool(&stream, "options_force_internal_seed_from_prediction", options.force_internal_seed_from_prediction);
  AppendBool(&stream, "options_enable_internal_structure_correction_after_ss", options.enable_internal_structure_correction_after_ss);
  AppendBool(&stream, "options_bypass_internal_seed_filters", options.bypass_internal_seed_filters);
  AppendScalar(&stream, "internal_pose_rescue_mode", static_cast<int>(options.internal_pose_rescue_mode));
  AppendScalar(&stream, "internal_pose_rescue_max_ray_angle_deg", options.internal_pose_rescue_max_ray_angle_deg);
  AppendScalar(&stream, "internal_pose_rescue_accept_max_outer_rmse", options.internal_pose_rescue_accept_max_outer_rmse);
  AppendBool(&stream, "enable_geometry_prior_outer_seed", options.enable_geometry_prior_outer_seed);
  AppendBool(&stream, "geometry_prior_rescue_diagnostic_only", options.geometry_prior_rescue_diagnostic_only);
  AppendBool(&stream, "geometry_prior_rescue_use_as_observation", options.geometry_prior_rescue_use_as_observation);
  AppendBool(&stream, "geometry_prior_rescue_allow_geometry_only_pose_refit", options.geometry_prior_rescue_allow_geometry_only_pose_refit);
  AppendScalar(&stream, "geometry_prior_rescue_subpix_window_radius", options.geometry_prior_rescue_subpix_window_radius);
  AppendScalar(&stream, "geometry_prior_rescue_max_corner_displacement_px", options.geometry_prior_rescue_max_corner_displacement_px);
  AppendScalar(&stream, "geometry_prior_rescue_min_corner_response_ratio", options.geometry_prior_rescue_min_corner_response_ratio);
  AppendBool(&stream, "geometry_prior_rescue_enable_spherical_refine", options.geometry_prior_rescue_enable_spherical_refine);
  AppendScalar(&stream, "geometry_prior_rescue_edge_sample_count", options.geometry_prior_rescue_edge_sample_count);
  AppendScalar(&stream, "geometry_prior_rescue_edge_search_half_width_px", options.geometry_prior_rescue_edge_search_half_width_px);
  AppendScalar(&stream, "geometry_prior_rescue_min_edge_support_ratio", options.geometry_prior_rescue_min_edge_support_ratio);
  AppendScalar(&stream, "geometry_prior_rescue_min_edge_gradient_ratio", options.geometry_prior_rescue_min_edge_gradient_ratio);
  AppendScalar(&stream, "geometry_prior_rescue_accept_max_outer_rmse", options.geometry_prior_rescue_accept_max_outer_rmse);
  AppendScalar(&stream, "geometry_prior_rescue_accept_max_rotation_error_deg", options.geometry_prior_rescue_accept_max_rotation_error_deg);
  AppendScalar(&stream, "geometry_prior_rescue_accept_max_translation_error", options.geometry_prior_rescue_accept_max_translation_error);
  AppendBool(&stream, "geometry_guided_tag_likelihood_enabled", options.geometry_guided_tag_likelihood_enabled);
  AppendScalar(&stream, "geometry_guided_tag_likelihood_min_visible_boards", options.geometry_guided_tag_likelihood_min_visible_boards);
  AppendScalar(&stream, "geometry_guided_tag_likelihood_max_expected_hamming", options.geometry_guided_tag_likelihood_max_expected_hamming);
  AppendScalar(&stream, "geometry_guided_tag_likelihood_min_hamming_margin", options.geometry_guided_tag_likelihood_min_hamming_margin);
  AppendScalar(&stream, "geometry_guided_tag_likelihood_min_contrast", options.geometry_guided_tag_likelihood_min_contrast);
  AppendBool(&stream, "geometry_guided_tag_likelihood_allow_single_anchor", options.geometry_guided_tag_likelihood_allow_single_anchor);
  AppendScalar(&stream, "geometry_guided_tag_likelihood_single_anchor_max_outer_rmse", options.geometry_guided_tag_likelihood_single_anchor_max_outer_rmse);
  AppendScalar(&stream, "geometry_guided_tag_likelihood_single_anchor_max_expected_hamming", options.geometry_guided_tag_likelihood_single_anchor_max_expected_hamming);
  AppendScalar(&stream, "geometry_guided_tag_likelihood_single_anchor_min_hamming_margin", options.geometry_guided_tag_likelihood_single_anchor_min_hamming_margin);
  AppendScalar(&stream, "geometry_guided_tag_likelihood_single_anchor_min_contrast", options.geometry_guided_tag_likelihood_single_anchor_min_contrast);
  return stream.str();
}

std::string MakeSignature(const std::string& text) {
  return HashToHex(HashBytes(text));
}

}  // namespace

InternalRegenerationCache::InternalRegenerationCache(
    ApriltagInternalConfig config,
    ApriltagInternalDetectionOptions detection_options,
    InternalRegenerationCacheOptions options)
    : config_(std::move(config)),
      detection_options_(std::move(detection_options)),
      options_(std::move(options)),
      semantic_config_hash_(MakeSignature(
          MakeInternalConfigSignature(config_, detection_options_))) {}

bool InternalRegenerationCache::enabled() const {
  return options_.enabled && !options_.cache_dir.empty();
}

bool InternalRegenerationCache::PrepareForDataset(
    const std::string& image_path, std::string* warning) const {
  if (warning != nullptr) {
    warning->clear();
  }
  if (!enabled()) {
    return true;
  }
  Stage5CacheManifest manifest(options_.cache_dir);
  std::string manifest_warning;
  if (!manifest.EnsureDatasetManifest(
          MakeStage5DatasetCacheIdentity(image_path), &manifest_warning)) {
    if (warning != nullptr) {
      *warning = manifest_warning;
    }
    return false;
  }
  const Stage5CacheManifestEntry entry{
      Stage5CacheStage::InternalRefinement,
      kStageImplementationVersion,
      kArtifactSchemaVersion,
      semantic_config_hash_,
      {"outer_detection_final"},
      "internal regeneration result; per-record parent=outer-result+state"};
  if (!manifest.EnsureStageManifest(entry, &manifest_warning)) {
    if (warning != nullptr) {
      *warning = manifest_warning;
    }
    return false;
  }
  manifests_prepared_ = true;
  return true;
}

const std::string& InternalRegenerationCache::cache_dir() const {
  return options_.cache_dir;
}

const std::string& InternalRegenerationCache::semantic_config_hash() const {
  return semantic_config_hash_;
}

std::string InternalRegenerationCache::MakeOuterResultSignature(
    const OuterTagMultiDetectionResult& outer_detection) {
  std::ostringstream stream;
  stream << std::setprecision(17) << outer_detection.image_size.width << ","
         << outer_detection.image_size.height << "|";
  AppendVector(&stream, "requested_board_ids", outer_detection.requested_board_ids);
  for (const OuterTagDetectionResult& detection : outer_detection.detections) {
    AppendBool(&stream, "success", detection.success);
    AppendScalar(&stream, "board_id", detection.board_id);
    AppendScalar(&stream, "detected_tag_id", detection.detected_tag_id);
    AppendScalar(&stream, "hamming", detection.hamming);
    AppendBool(&stream, "good", detection.good);
    AppendBool(&stream, "used_local_patch_rescue", detection.used_local_patch_rescue);
    AppendScalar(&stream, "quality", detection.quality);
    AppendScalar(&stream, "failure_reason", static_cast<int>(detection.failure_reason));
    for (const Eigen::Vector2d& point : detection.refined_corners_original_image) {
      AppendScalar(&stream, "refined_x", point.x());
      AppendScalar(&stream, "refined_y", point.y());
    }
    for (const Eigen::Vector2d& point : detection.coarse_corners_original_image) {
      AppendScalar(&stream, "coarse_x", point.x());
      AppendScalar(&stream, "coarse_y", point.y());
    }
    for (bool valid : detection.refined_valid) {
      AppendBool(&stream, "refined_valid", valid);
    }
  }
  return MakeSignature(stream.str());
}

std::string InternalRegenerationCache::MakeBootstrapStateSignature(
    const OuterBootstrapResult& bootstrap_result) {
  std::ostringstream stream;
  stream << std::setprecision(17) << "bootstrap|";
  AppendScalar(&stream, "reference_board_id", bootstrap_result.reference_board_id);
  AppendBool(&stream, "success", bootstrap_result.success);
  AppendCamera(&stream, bootstrap_result.coarse_camera);
  for (const OuterBootstrapBoardState& board : bootstrap_result.boards) {
    AppendScalar(&stream, "board_id", board.board_id);
    AppendBool(&stream, "board_initialized", board.initialized);
    AppendMatrix(&stream, "T_reference_board", board.T_reference_board);
    AppendScalar(&stream, "board_observation_count", board.observation_count);
    AppendScalar(&stream, "board_rmse", board.rmse);
  }
  AppendStateFrames(&stream, bootstrap_result.frames);
  return MakeSignature(stream.str());
}

std::string InternalRegenerationCache::MakeSceneStateSignature(
    const JointReprojectionSceneState& scene_state) {
  std::ostringstream stream;
  stream << std::setprecision(17) << "scene|";
  AppendScalar(&stream, "reference_board_id", scene_state.reference_board_id);
  AppendCamera(&stream, scene_state.camera);
  for (const JointSceneBoardState& board : scene_state.boards) {
    AppendScalar(&stream, "board_id", board.board_id);
    AppendBool(&stream, "board_initialized", board.initialized);
    AppendMatrix(&stream, "T_reference_board", board.T_reference_board);
    AppendScalar(&stream, "board_observation_count", board.observation_count);
    AppendScalar(&stream, "board_rmse", board.rmse);
  }
  AppendSceneFrames(&stream, scene_state.frames);
  return MakeSignature(stream.str());
}

bool InternalRegenerationCache::Load(
    const std::string& image_path,
    const InternalRegenerationFrameInput& frame_input,
    const std::string& state_signature,
    InternalRegenerationFrameResult* frame_result,
    std::string* warning) const {
  if (warning != nullptr) {
    warning->clear();
  }
  if (!enabled() || frame_result == nullptr) {
    return false;
  }
  try {
    const std::string absolute_path = AbsoluteImagePath(image_path);
    const fs::path path(absolute_path);
    if (!fs::exists(path) || !fs::is_regular_file(path)) {
      return false;
    }
    const Stage5DatasetCacheIdentity identity =
        MakeStage5DatasetCacheIdentity(image_path);
    Stage5CacheManifest manifest(options_.cache_dir);
    std::string manifest_warning;
    if (!manifests_prepared_ && !manifest.EnsureDatasetManifest(identity, &manifest_warning)) {
      if (warning != nullptr) *warning = manifest_warning;
      ++stats_.load_failures;
      return false;
    }
    Stage5CacheManifestEntry entry;
    entry.stage = Stage5CacheStage::InternalRefinement;
    entry.implementation_version = kStageImplementationVersion;
    entry.artifact_schema_version = kArtifactSchemaVersion;
    entry.semantic_config_hash = semantic_config_hash_;
    entry.parent_artifact_hashes = {"outer_detection_final"};
    entry.semantic_config_description =
        "internal regeneration result; per-record parent=outer-result+state";
    if (!manifests_prepared_ && !manifest.EnsureStageManifest(entry, &manifest_warning)) {
      if (warning != nullptr) *warning = manifest_warning;
      ++stats_.load_failures;
      return false;
    }
    const std::string outer_signature =
        MakeOuterResultSignature(frame_input.outer_detections);
    const std::string image_key =
        MakeImageKey(image_path, outer_signature, state_signature);
    const fs::path cache_path =
        fs::path(manifest.StageDirectory(Stage5CacheStage::InternalRefinement,
                                          semantic_config_hash_)) /
        (image_key + ".yml");
    if (!fs::exists(cache_path)) {
      ++stats_.cache_misses;
      return false;
    }
    cv::FileStorage storage(cache_path.string(), cv::FileStorage::READ);
    if (!storage.isOpened()) {
      if (warning != nullptr) *warning = "Failed to open internal cache: " + cache_path.string();
      ++stats_.load_failures;
      return false;
    }
    if (static_cast<std::string>(storage["cache_format_version"]) != kCacheFormatVersion ||
        static_cast<std::string>(storage["semantic_config_hash"]) != semantic_config_hash_ ||
        static_cast<std::string>(storage["outer_result_signature"]) != outer_signature ||
        static_cast<std::string>(storage["state_signature"]) != state_signature) {
      ++stats_.cache_misses;
      return false;
    }
    if (!ReadFrameResult(storage.root(), frame_result) ||
        frame_result->frame_index != frame_input.frame_index) {
      if (warning != nullptr) *warning = "Internal cache artifact is incomplete: " + cache_path.string();
      ++stats_.load_failures;
      return false;
    }
    // New cache artifacts carry the effective outer detection selected during
    // regeneration (which may be a geometry-prior rescue). Only fall back to
    // the frame input for legacy artifacts that predate this field.
    for (RegeneratedBoardMeasurement& measurement : frame_result->board_measurements) {
      if (!measurement.detection.outer_detection.success &&
          measurement.detection.outer_detection.failure_reason_text.empty()) {
        for (const OuterTagDetectionResult& outer : frame_input.outer_detections.detections) {
          if (outer.board_id == measurement.board_id) {
            measurement.detection.outer_detection = outer;
            break;
          }
        }
      }
    }
    ++stats_.cache_hits;
    return true;
  } catch (const std::exception& exception) {
    if (warning != nullptr) *warning = exception.what();
    ++stats_.load_failures;
    return false;
  }
}

bool InternalRegenerationCache::Save(
    const std::string& image_path,
    const InternalRegenerationFrameInput& frame_input,
    const std::string& state_signature,
    const InternalRegenerationFrameResult& frame_result,
    std::string* warning) const {
  if (warning != nullptr) {
    warning->clear();
  }
  if (!enabled()) {
    return false;
  }
  try {
    const fs::path image_file(AbsoluteImagePath(image_path));
    if (!fs::exists(image_file) || !fs::is_regular_file(image_file)) {
      if (warning != nullptr) *warning = "Image does not exist: " + image_file.string();
      ++stats_.store_failures;
      return false;
    }
    const Stage5DatasetCacheIdentity identity =
        MakeStage5DatasetCacheIdentity(image_path);
    Stage5CacheManifest manifest(options_.cache_dir);
    std::string manifest_warning;
    if (!manifests_prepared_ && !manifest.EnsureDatasetManifest(identity, &manifest_warning)) {
      if (warning != nullptr) *warning = manifest_warning;
      ++stats_.store_failures;
      return false;
    }
    Stage5CacheManifestEntry entry;
    entry.stage = Stage5CacheStage::InternalRefinement;
    entry.implementation_version = kStageImplementationVersion;
    entry.artifact_schema_version = kArtifactSchemaVersion;
    entry.semantic_config_hash = semantic_config_hash_;
    entry.parent_artifact_hashes = {"outer_detection_final"};
    entry.semantic_config_description =
        "internal regeneration result; per-record parent=outer-result+state";
    if (!manifests_prepared_ && !manifest.EnsureStageManifest(entry, &manifest_warning)) {
      if (warning != nullptr) *warning = manifest_warning;
      ++stats_.store_failures;
      return false;
    }
    const std::string outer_signature =
        MakeOuterResultSignature(frame_input.outer_detections);
    const std::string image_key =
        MakeImageKey(image_path, outer_signature, state_signature);
    const fs::path directory = manifest.StageDirectory(
        Stage5CacheStage::InternalRefinement, semantic_config_hash_);
    fs::create_directories(directory);
    const fs::path cache_path = directory / (image_key + ".yml");
    if (fs::exists(cache_path)) {
      return true;
    }
    const fs::path temporary_path = cache_path.string() + ".tmp";
    cv::FileStorage storage(temporary_path.string(), cv::FileStorage::WRITE);
    if (!storage.isOpened()) {
      if (warning != nullptr) *warning = "Failed to write internal cache: " + temporary_path.string();
      ++stats_.store_failures;
      return false;
    }
    storage << "cache_format_version" << kCacheFormatVersion;
    storage << "semantic_config_hash" << semantic_config_hash_;
    storage << "absolute_image_path" << AbsoluteImagePath(image_path);
    storage << "image_file_size" << static_cast<long long>(fs::file_size(image_file));
    storage << "image_mtime" << static_cast<long long>(fs::last_write_time(image_file));
    storage << "outer_result_signature" << outer_signature;
    storage << "state_signature" << state_signature;
    WriteFrameResult(&storage, frame_result);
    storage.release();
    if (fs::exists(cache_path)) {
      fs::remove(temporary_path);
    } else {
      fs::rename(temporary_path, cache_path);
    }
    return true;
  } catch (const std::exception& exception) {
    if (warning != nullptr) *warning = exception.what();
    ++stats_.store_failures;
    return false;
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
