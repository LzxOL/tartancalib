#include <aslam/cameras/apriltag_internal/OuterDetectionCache.hpp>

#include <array>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

namespace fs = boost::filesystem;

constexpr const char kCacheFormatVersion[] = "outer_detection_cache_v2";

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

void HashCombine(std::uint64_t* seed, const std::string& value) {
  if (seed == nullptr) {
    return;
  }
  *seed ^= HashBytes(value) + 0x9e3779b97f4a7c15ull + (*seed << 6) + (*seed >> 2);
}

std::string MakeConfigSignature(const MultiScaleOuterTagDetectorConfig& config) {
  std::ostringstream stream;
  stream << "format=" << kCacheFormatVersion
         << "|tag_id=" << config.tag_id
         << "|tag_ids=";
  for (std::size_t index = 0; index < config.tag_ids.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << config.tag_ids[index];
  }
  stream << "|min_border_distance=" << config.min_border_distance
         << "|max_scales_to_try=" << config.max_scales_to_try
         << "|enable_outer_spherical_refinement="
         << (config.enable_outer_spherical_refinement ? 1 : 0)
         << "|do_outer_subpix_refinement="
         << (config.do_outer_subpix_refinement ? 1 : 0)
         << "|outer_local_context_scale=" << config.outer_local_context_scale
         << "|outer_subpix_scale=" << config.outer_subpix_scale
         << "|outer_refine_gate_scale=" << config.outer_refine_gate_scale
         << "|outer_refine_gate_min=" << config.outer_refine_gate_min
         << "|min_detection_quality=" << config.min_detection_quality
         << "|blur_before_detect=" << (config.blur_before_detect ? 1 : 0)
         << "|blur_kernel=" << config.blur_kernel
         << "|blur_sigma=" << config.blur_sigma
         << "|outer_subpix_window_radius=" << config.outer_subpix_window_radius
         << "|outer_subpix_window_scale=" << config.outer_subpix_window_scale
         << "|outer_subpix_window_min=" << config.outer_subpix_window_min
         << "|outer_subpix_window_max=" << config.outer_subpix_window_max
         << "|max_outer_refine_displacement=" << config.max_outer_refine_displacement
         << "|outer_refine_displacement_scale="
         << config.outer_refine_displacement_scale
         << "|enable_outer_corner_layout_check="
         << (config.enable_outer_corner_layout_check ? 1 : 0)
         << "|outer_corner_verification_roi_scale="
         << config.outer_corner_verification_roi_scale
         << "|outer_corner_verification_roi_min="
         << config.outer_corner_verification_roi_min
         << "|outer_corner_verification_roi_max="
         << config.outer_corner_verification_roi_max
         << "|outer_corner_candidate_scale="
         << config.outer_corner_candidate_scale
         << "|outer_corner_candidate_min=" << config.outer_corner_candidate_min
         << "|outer_corner_candidate_max=" << config.outer_corner_candidate_max
         << "|outer_corner_branch_search_scale="
         << config.outer_corner_branch_search_scale
         << "|outer_corner_branch_search_min="
         << config.outer_corner_branch_search_min
         << "|outer_corner_branch_search_max="
         << config.outer_corner_branch_search_max
         << "|outer_corner_min_direction_score="
         << config.outer_corner_min_direction_score
         << "|outer_corner_min_layout_score=" << config.outer_corner_min_layout_score
         << "|refine_camera_model=" << config.refine_camera.camera_model
         << "|refine_distortion_model=" << config.refine_camera.distortion_model
         << "|refine_intrinsics=";
  for (std::size_t index = 0; index < config.refine_camera.intrinsics.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << config.refine_camera.intrinsics[index];
  }
  stream << "|refine_resolution=";
  for (std::size_t index = 0; index < config.refine_camera.resolution.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << config.refine_camera.resolution[index];
  }
  return stream.str();
}

std::string AbsoluteNormalizedPath(const std::string& image_path) {
  return fs::absolute(fs::path(image_path)).lexically_normal().string();
}

std::uint64_t MakeImageKey(const std::string& absolute_image_path,
                           std::uintmax_t file_size,
                           std::time_t file_mtime) {
  std::ostringstream stream;
  stream << absolute_image_path
         << "|size=" << file_size
         << "|mtime=" << static_cast<long long>(file_mtime);
  return HashBytes(stream.str());
}

fs::path CacheFilePath(const OuterDetectionCacheOptions& options,
                       const std::string& detector_config_hash,
                       const std::string& absolute_image_path,
                       std::uintmax_t file_size,
                       std::time_t file_mtime) {
  const std::uint64_t image_key =
      MakeImageKey(absolute_image_path, file_size, file_mtime);
  return fs::path(options.cache_dir) / detector_config_hash /
         (HashToHex(image_key) + ".yml");
}

void WriteVector2dArray(cv::FileStorage* storage,
                        const char* key,
                        const std::array<Eigen::Vector2d, 4>& values) {
  *storage << key << "[";
  for (const Eigen::Vector2d& value : values) {
    *storage << "[" << value.x() << value.y() << "]";
  }
  *storage << "]";
}

void WritePoint2f(cv::FileStorage* storage,
                  const char* key,
                  const cv::Point2f& value) {
  *storage << key << "[" << value.x << value.y << "]";
}

void ReadPoint2f(const cv::FileNode& node,
                 cv::Point2f* value) {
  if (value == nullptr) {
    return;
  }
  *value = cv::Point2f();
  if (node.size() >= 2) {
    value->x = static_cast<float>(node[0]);
    value->y = static_cast<float>(node[1]);
  }
}

void WritePoint2fVector(cv::FileStorage* storage,
                        const char* key,
                        const std::vector<cv::Point2f>& values) {
  *storage << key << "[";
  for (const cv::Point2f& value : values) {
    *storage << "[" << value.x << value.y << "]";
  }
  *storage << "]";
}

void ReadPoint2fVector(const cv::FileNode& node,
                       std::vector<cv::Point2f>* values) {
  if (values == nullptr) {
    return;
  }
  values->clear();
  for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it) {
    cv::Point2f value;
    ReadPoint2f(*it, &value);
    values->push_back(value);
  }
}

void WriteRect(cv::FileStorage* storage,
               const char* key,
               const cv::Rect& value) {
  *storage << key << "["
           << value.x
           << value.y
           << value.width
           << value.height
           << "]";
}

void ReadRect(const cv::FileNode& node,
              cv::Rect* value) {
  if (value == nullptr) {
    return;
  }
  *value = cv::Rect();
  if (node.size() >= 4) {
    value->x = static_cast<int>(node[0]);
    value->y = static_cast<int>(node[1]);
    value->width = static_cast<int>(node[2]);
    value->height = static_cast<int>(node[3]);
  }
}

void WriteBoolArray(cv::FileStorage* storage,
                    const char* key,
                    const std::array<bool, 4>& values) {
  *storage << key << "[";
  for (bool value : values) {
    *storage << (value ? 1 : 0);
  }
  *storage << "]";
}

void ReadVector2dArray(const cv::FileNode& node,
                       std::array<Eigen::Vector2d, 4>* values) {
  if (values == nullptr) {
    return;
  }
  values->fill(Eigen::Vector2d::Zero());
  int index = 0;
  for (cv::FileNodeIterator it = node.begin(); it != node.end() && index < 4; ++it, ++index) {
    const cv::FileNode pair = *it;
    if (pair.size() >= 2) {
      (*values)[static_cast<std::size_t>(index)].x() = static_cast<double>(pair[0]);
      (*values)[static_cast<std::size_t>(index)].y() = static_cast<double>(pair[1]);
    }
  }
}

void ReadBoolArray(const cv::FileNode& node,
                   std::array<bool, 4>* values) {
  if (values == nullptr) {
    return;
  }
  values->fill(false);
  int index = 0;
  for (cv::FileNodeIterator it = node.begin(); it != node.end() && index < 4; ++it, ++index) {
    (*values)[static_cast<std::size_t>(index)] = static_cast<int>(*it) != 0;
  }
}

int CountValidCorners(const std::array<bool, 4>& values) {
  int count = 0;
  for (bool value : values) {
    count += value ? 1 : 0;
  }
  return count;
}

void WriteIntVector(cv::FileStorage* storage,
                    const char* key,
                    const std::vector<int>& values) {
  *storage << key << "[";
  for (int value : values) {
    *storage << value;
  }
  *storage << "]";
}

void ReadIntVector(const cv::FileNode& node,
                   std::vector<int>* values) {
  if (values == nullptr) {
    return;
  }
  values->clear();
  for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it) {
    values->push_back(static_cast<int>(*it));
  }
}

void WriteOuterCornerVerificationDebug(
    cv::FileStorage* storage,
    const OuterCornerVerificationDebugInfo& debug) {
  *storage << "{";
  *storage << "corner_index" << debug.corner_index;
  WritePoint2f(storage, "coarse_corner", debug.coarse_corner);
  WritePoint2f(storage, "verified_corner", debug.verified_corner);
  WritePoint2f(storage, "subpix_corner", debug.subpix_corner);
  WriteRect(storage, "verification_roi", debug.verification_roi);
  WritePoint2f(storage, "prev_edge_direction", debug.prev_edge_direction);
  WritePoint2f(storage, "next_edge_direction", debug.next_edge_direction);
  WritePoint2fVector(storage, "prev_marker_support_points",
                     debug.prev_marker_support_points);
  WritePoint2fVector(storage, "next_marker_support_points",
                     debug.next_marker_support_points);
  WritePoint2fVector(storage, "prev_branch_points", debug.prev_branch_points);
  WritePoint2fVector(storage, "next_branch_points", debug.next_branch_points);
  *storage << "local_scale" << debug.local_scale;
  *storage << "verification_roi_radius" << debug.verification_roi_radius;
  *storage << "candidate_radius" << debug.candidate_radius;
  *storage << "branch_search_radius" << debug.branch_search_radius;
  *storage << "direction_consistency_score" << debug.direction_consistency_score;
  *storage << "local_layout_score" << debug.local_layout_score;
  *storage << "verification_quality" << debug.verification_quality;
  *storage << "coarse_to_verified_displacement" << debug.coarse_to_verified_displacement;
  *storage << "coarse_to_subpix_displacement" << debug.coarse_to_subpix_displacement;
  *storage << "coarse_to_refined_displacement" << debug.coarse_to_refined_displacement;
  *storage << "corner_marker_width" << debug.corner_marker_width;
  WritePoint2f(storage, "image_line_corner", debug.image_line_corner);
  *storage << "prev_image_line_residual" << debug.prev_image_line_residual;
  *storage << "next_image_line_residual" << debug.next_image_line_residual;
  *storage << "prev_image_line_support_count" << debug.prev_image_line_support_count;
  *storage << "next_image_line_support_count" << debug.next_image_line_support_count;
  *storage << "image_line_valid" << (debug.image_line_valid ? 1 : 0);
  *storage << "line_refinement_success" << (debug.line_refinement_success ? 1 : 0);
  *storage << "line_refinement_quality" << debug.line_refinement_quality;
  *storage << "line_jump" << debug.line_jump;
  *storage << "line_jump_limit" << debug.line_jump_limit;
  *storage << "line_inside" << (debug.line_inside ? 1 : 0);
  *storage << "line_seed_gap" << debug.line_seed_gap;
  *storage << "line_seed_accepted" << (debug.line_seed_accepted ? 1 : 0);
  WritePoint2f(storage, "spherical_corner", debug.spherical_corner);
  WritePoint2fVector(storage, "prev_spherical_curve_points",
                     debug.prev_spherical_curve_points);
  WritePoint2fVector(storage, "next_spherical_curve_points",
                     debug.next_spherical_curve_points);
  *storage << "prev_spherical_residual" << debug.prev_spherical_residual;
  *storage << "next_spherical_residual" << debug.next_spherical_residual;
  *storage << "prev_spherical_support_count" << debug.prev_spherical_support_count;
  *storage << "next_spherical_support_count" << debug.next_spherical_support_count;
  *storage << "spherical_refinement_valid" << (debug.spherical_refinement_valid ? 1 : 0);
  *storage << "spherical_refinement_applied" << (debug.spherical_refinement_applied ? 1 : 0);
  *storage << "spherical_failure_reason" << debug.spherical_failure_reason;
  *storage << "subpix_window_radius" << debug.subpix_window_radius;
  *storage << "refine_displacement_limit" << debug.refine_displacement_limit;
  *storage << "refined_valid" << (debug.refined_valid ? 1 : 0);
  *storage << "verification_passed" << (debug.verification_passed ? 1 : 0);
  *storage << "subpix_applied" << (debug.subpix_applied ? 1 : 0);
  *storage << "failure_reason" << debug.failure_reason;
  *storage << "}";
}

void ReadOuterCornerVerificationDebug(
    const cv::FileNode& node,
    OuterCornerVerificationDebugInfo* debug) {
  if (debug == nullptr) {
    return;
  }
  *debug = OuterCornerVerificationDebugInfo{};
  debug->corner_index = static_cast<int>(node["corner_index"]);
  ReadPoint2f(node["coarse_corner"], &debug->coarse_corner);
  ReadPoint2f(node["verified_corner"], &debug->verified_corner);
  ReadPoint2f(node["subpix_corner"], &debug->subpix_corner);
  ReadRect(node["verification_roi"], &debug->verification_roi);
  ReadPoint2f(node["prev_edge_direction"], &debug->prev_edge_direction);
  ReadPoint2f(node["next_edge_direction"], &debug->next_edge_direction);
  ReadPoint2fVector(node["prev_marker_support_points"],
                    &debug->prev_marker_support_points);
  ReadPoint2fVector(node["next_marker_support_points"],
                    &debug->next_marker_support_points);
  ReadPoint2fVector(node["prev_branch_points"], &debug->prev_branch_points);
  ReadPoint2fVector(node["next_branch_points"], &debug->next_branch_points);
  debug->local_scale = static_cast<double>(node["local_scale"]);
  debug->verification_roi_radius = static_cast<int>(node["verification_roi_radius"]);
  debug->candidate_radius = static_cast<int>(node["candidate_radius"]);
  debug->branch_search_radius = static_cast<int>(node["branch_search_radius"]);
  debug->direction_consistency_score =
      static_cast<double>(node["direction_consistency_score"]);
  debug->local_layout_score = static_cast<double>(node["local_layout_score"]);
  debug->verification_quality = static_cast<double>(node["verification_quality"]);
  debug->coarse_to_verified_displacement =
      static_cast<double>(node["coarse_to_verified_displacement"]);
  debug->coarse_to_subpix_displacement =
      static_cast<double>(node["coarse_to_subpix_displacement"]);
  debug->coarse_to_refined_displacement =
      static_cast<double>(node["coarse_to_refined_displacement"]);
  debug->corner_marker_width = static_cast<double>(node["corner_marker_width"]);
  ReadPoint2f(node["image_line_corner"], &debug->image_line_corner);
  debug->prev_image_line_residual = static_cast<double>(node["prev_image_line_residual"]);
  debug->next_image_line_residual = static_cast<double>(node["next_image_line_residual"]);
  debug->prev_image_line_support_count =
      static_cast<int>(node["prev_image_line_support_count"]);
  debug->next_image_line_support_count =
      static_cast<int>(node["next_image_line_support_count"]);
  debug->image_line_valid = static_cast<int>(node["image_line_valid"]) != 0;
  debug->line_refinement_success =
      static_cast<int>(node["line_refinement_success"]) != 0;
  debug->line_refinement_quality = static_cast<double>(node["line_refinement_quality"]);
  debug->line_jump = static_cast<double>(node["line_jump"]);
  debug->line_jump_limit = static_cast<double>(node["line_jump_limit"]);
  debug->line_inside = static_cast<int>(node["line_inside"]) != 0;
  debug->line_seed_gap = static_cast<double>(node["line_seed_gap"]);
  debug->line_seed_accepted = static_cast<int>(node["line_seed_accepted"]) != 0;
  ReadPoint2f(node["spherical_corner"], &debug->spherical_corner);
  ReadPoint2fVector(node["prev_spherical_curve_points"],
                    &debug->prev_spherical_curve_points);
  ReadPoint2fVector(node["next_spherical_curve_points"],
                    &debug->next_spherical_curve_points);
  debug->prev_spherical_residual = static_cast<double>(node["prev_spherical_residual"]);
  debug->next_spherical_residual = static_cast<double>(node["next_spherical_residual"]);
  debug->prev_spherical_support_count =
      static_cast<int>(node["prev_spherical_support_count"]);
  debug->next_spherical_support_count =
      static_cast<int>(node["next_spherical_support_count"]);
  debug->spherical_refinement_valid =
      static_cast<int>(node["spherical_refinement_valid"]) != 0;
  debug->spherical_refinement_applied =
      static_cast<int>(node["spherical_refinement_applied"]) != 0;
  debug->spherical_failure_reason =
      static_cast<std::string>(node["spherical_failure_reason"]);
  debug->subpix_window_radius = static_cast<int>(node["subpix_window_radius"]);
  debug->refine_displacement_limit =
      static_cast<double>(node["refine_displacement_limit"]);
  debug->refined_valid = static_cast<int>(node["refined_valid"]) != 0;
  debug->verification_passed = static_cast<int>(node["verification_passed"]) != 0;
  debug->subpix_applied = static_cast<int>(node["subpix_applied"]) != 0;
  debug->failure_reason = static_cast<std::string>(node["failure_reason"]);
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

void ReadOuterCornerVerificationDebugArray(
    const cv::FileNode& node,
    std::array<OuterCornerVerificationDebugInfo, 4>* values) {
  if (values == nullptr) {
    return;
  }
  values->fill(OuterCornerVerificationDebugInfo{});
  int index = 0;
  for (cv::FileNodeIterator it = node.begin();
       it != node.end() && index < 4;
       ++it, ++index) {
    ReadOuterCornerVerificationDebug(
        *it, &(*values)[static_cast<std::size_t>(index)]);
  }
}

OuterBoardMeasurement BuildBoardMeasurement(const OuterTagDetectionResult& detection) {
  OuterBoardMeasurement measurement;
  measurement.board_id = detection.board_id;
  measurement.detected_tag_id = detection.detected_tag_id;
  measurement.success = detection.success;
  measurement.detection_quality = detection.quality;
  measurement.refined_outer_corners_original_image =
      detection.refined_corners_original_image;
  measurement.refined_corner_valid = detection.refined_valid;
  measurement.valid_refined_corner_count = CountValidCorners(detection.refined_valid);
  measurement.failure_reason = detection.failure_reason;
  measurement.failure_reason_text = detection.failure_reason_text;
  return measurement;
}

void WriteDetection(cv::FileStorage* storage,
                    const OuterTagDetectionResult& detection) {
  *storage << "{";
  *storage << "success" << (detection.success ? 1 : 0);
  *storage << "board_id" << detection.board_id;
  *storage << "detected_tag_id" << detection.detected_tag_id;
  *storage << "original_longest_side" << detection.original_longest_side;
  *storage << "chosen_scale_longest_side" << detection.chosen_scale_longest_side;
  *storage << "chosen_scale_factor" << detection.chosen_scale_factor;
  *storage << "scale_configuration_mode" << detection.scale_configuration_mode;
  *storage << "used_corner_fusion" << (detection.used_corner_fusion ? 1 : 0);
  *storage << "hamming" << detection.hamming;
  *storage << "good" << (detection.good ? 1 : 0);
  *storage << "attempted_local_patch_rescue"
           << (detection.attempted_local_patch_rescue ? 1 : 0);
  *storage << "used_local_patch_rescue"
           << (detection.used_local_patch_rescue ? 1 : 0);
  *storage << "local_patch_rescue_summary" << detection.local_patch_rescue_summary;
  *storage << "quality" << detection.quality;
  *storage << "failure_reason" << static_cast<int>(detection.failure_reason);
  *storage << "failure_reason_text" << detection.failure_reason_text;
  WriteVector2dArray(storage, "coarse_corners_scaled_image",
                     detection.coarse_corners_scaled_image);
  WriteVector2dArray(storage, "coarse_corners_original_image",
                     detection.coarse_corners_original_image);
  WriteVector2dArray(storage, "refined_corners_original_image",
                     detection.refined_corners_original_image);
  WriteBoolArray(storage, "refined_valid", detection.refined_valid);
  WriteIntVector(storage, "successful_scale_longest_sides",
                 detection.successful_scale_longest_sides);
  WriteOuterCornerVerificationDebugArray(storage, "corner_verification_debug",
                                         detection.corner_verification_debug);
  *storage << "}";
}

OuterTagDetectionResult ReadDetection(const cv::FileNode& node) {
  OuterTagDetectionResult detection;
  detection.success = static_cast<int>(node["success"]) != 0;
  detection.board_id = static_cast<int>(node["board_id"]);
  detection.detected_tag_id = static_cast<int>(node["detected_tag_id"]);
  detection.original_longest_side =
      static_cast<int>(node["original_longest_side"]);
  detection.chosen_scale_longest_side =
      static_cast<int>(node["chosen_scale_longest_side"]);
  detection.chosen_scale_factor = static_cast<double>(node["chosen_scale_factor"]);
  detection.scale_configuration_mode =
      static_cast<std::string>(node["scale_configuration_mode"]);
  detection.used_corner_fusion = static_cast<int>(node["used_corner_fusion"]) != 0;
  detection.hamming = static_cast<int>(node["hamming"]);
  detection.good = static_cast<int>(node["good"]) != 0;
  detection.attempted_local_patch_rescue =
      static_cast<int>(node["attempted_local_patch_rescue"]) != 0;
  detection.used_local_patch_rescue =
      static_cast<int>(node["used_local_patch_rescue"]) != 0;
  detection.local_patch_rescue_summary =
      static_cast<std::string>(node["local_patch_rescue_summary"]);
  detection.quality = static_cast<double>(node["quality"]);
  detection.failure_reason =
      static_cast<OuterTagFailureReason>(static_cast<int>(node["failure_reason"]));
  detection.failure_reason_text =
      static_cast<std::string>(node["failure_reason_text"]);
  if (detection.failure_reason_text.empty()) {
    detection.failure_reason_text = ToString(detection.failure_reason);
  }
  ReadVector2dArray(node["coarse_corners_scaled_image"],
                    &detection.coarse_corners_scaled_image);
  ReadVector2dArray(node["coarse_corners_original_image"],
                    &detection.coarse_corners_original_image);
  ReadVector2dArray(node["refined_corners_original_image"],
                    &detection.refined_corners_original_image);
  ReadBoolArray(node["refined_valid"], &detection.refined_valid);
  ReadIntVector(node["successful_scale_longest_sides"],
                &detection.successful_scale_longest_sides);
  ReadOuterCornerVerificationDebugArray(node["corner_verification_debug"],
                                        &detection.corner_verification_debug);
  return detection;
}

CachedOuterDetectionRecord MakeRecord(const std::string& image_path,
                                      const std::string& detector_config_hash,
                                      const OuterTagMultiDetectionResult& detection_result) {
  const std::string absolute_image_path = AbsoluteNormalizedPath(image_path);
  CachedOuterDetectionRecord record;
  record.absolute_image_path = absolute_image_path;
  record.image_file_size = fs::file_size(fs::path(absolute_image_path));
  record.image_mtime = fs::last_write_time(fs::path(absolute_image_path));
  record.detector_config_hash = detector_config_hash;
  record.detection_result = detection_result;
  return record;
}

OuterTagMultiDetectionResult RebuildFrameMeasurements(
    const OuterTagMultiDetectionResult& cached_detection) {
  OuterTagMultiDetectionResult result = cached_detection;
  result.frame_measurements.image_size = cached_detection.image_size;
  result.frame_measurements.requested_board_ids = cached_detection.requested_board_ids;
  result.frame_measurements.board_measurements.clear();
  result.frame_measurements.board_measurements.reserve(cached_detection.detections.size());
  for (const OuterTagDetectionResult& detection : cached_detection.detections) {
    result.frame_measurements.board_measurements.push_back(
        BuildBoardMeasurement(detection));
  }
  return result;
}

}  // namespace

OuterDetectionCache::OuterDetectionCache(MultiScaleOuterTagDetectorConfig config,
                                         OuterDetectionCacheOptions options)
    : config_(std::move(config)),
      options_(std::move(options)),
      detector_config_hash_(HashToHex(HashBytes(MakeConfigSignature(config_)))) {}

bool OuterDetectionCache::enabled() const {
  return options_.enabled && !options_.cache_dir.empty();
}

const std::string& OuterDetectionCache::cache_dir() const {
  return options_.cache_dir;
}

const std::string& OuterDetectionCache::detector_config_hash() const {
  return detector_config_hash_;
}

bool OuterDetectionCache::Load(const std::string& image_path,
                               OuterTagMultiDetectionResult* detection_result,
                               std::string* warning) const {
  if (warning != nullptr) {
    warning->clear();
  }
  if (!enabled()) {
    return false;
  }

  const std::string absolute_image_path = AbsoluteNormalizedPath(image_path);
  if (!fs::exists(absolute_image_path)) {
    return false;
  }
  const std::uintmax_t file_size = fs::file_size(fs::path(absolute_image_path));
  const std::time_t file_mtime = fs::last_write_time(fs::path(absolute_image_path));
  const fs::path cache_path = CacheFilePath(
      options_, detector_config_hash_, absolute_image_path, file_size, file_mtime);
  if (!fs::exists(cache_path)) {
    return false;
  }

  cv::FileStorage storage(cache_path.string(), cv::FileStorage::READ);
  if (!storage.isOpened()) {
    if (warning != nullptr) {
      *warning = "Failed to open outer detection cache: " + cache_path.string();
    }
    return false;
  }

  if (static_cast<std::string>(storage["format_version"]) != kCacheFormatVersion ||
      static_cast<std::string>(storage["detector_config_hash"]) != detector_config_hash_ ||
      static_cast<std::string>(storage["absolute_image_path"]) != absolute_image_path ||
      static_cast<std::uintmax_t>(static_cast<std::int64_t>(storage["image_file_size"])) !=
          file_size ||
      static_cast<std::time_t>(static_cast<long long>(storage["image_mtime"])) != file_mtime) {
    return false;
  }

  OuterTagMultiDetectionResult cached_detection;
  const cv::FileNode image_size = storage["image_size"];
  if (image_size.size() >= 2) {
    cached_detection.image_size.width = static_cast<int>(image_size[0]);
    cached_detection.image_size.height = static_cast<int>(image_size[1]);
  }
  const cv::FileNode requested_board_ids = storage["requested_board_ids"];
  for (cv::FileNodeIterator it = requested_board_ids.begin();
       it != requested_board_ids.end(); ++it) {
    cached_detection.requested_board_ids.push_back(static_cast<int>(*it));
  }
  const cv::FileNode detections = storage["detections"];
  for (cv::FileNodeIterator it = detections.begin(); it != detections.end(); ++it) {
    cached_detection.detections.push_back(ReadDetection(*it));
  }

  if (cached_detection.requested_board_ids.empty()) {
    if (warning != nullptr) {
      *warning = "Outer detection cache contained no requested_board_ids: " +
                 cache_path.string();
    }
    return false;
  }
  *detection_result = RebuildFrameMeasurements(cached_detection);
  return true;
}

bool OuterDetectionCache::Save(const std::string& image_path,
                               const OuterTagMultiDetectionResult& detection_result,
                               std::string* warning) const {
  if (warning != nullptr) {
    warning->clear();
  }
  if (!enabled()) {
    return false;
  }
  try {
    const CachedOuterDetectionRecord record =
        MakeRecord(image_path, detector_config_hash_, detection_result);
    const fs::path cache_path = CacheFilePath(
        options_,
        detector_config_hash_,
        record.absolute_image_path,
        record.image_file_size,
        record.image_mtime);
    fs::create_directories(cache_path.parent_path());

    cv::FileStorage storage(cache_path.string(), cv::FileStorage::WRITE);
    if (!storage.isOpened()) {
      if (warning != nullptr) {
        *warning = "Failed to open outer detection cache for write: " +
                   cache_path.string();
      }
      return false;
    }

    storage << "format_version" << kCacheFormatVersion;
    storage << "absolute_image_path" << record.absolute_image_path;
    storage << "image_file_size"
            << static_cast<std::int64_t>(record.image_file_size);
    storage << "image_mtime" << static_cast<long long>(record.image_mtime);
    storage << "detector_config_hash" << record.detector_config_hash;
    storage << "image_size"
            << "[" << record.detection_result.image_size.width
            << record.detection_result.image_size.height << "]";
    storage << "requested_board_ids" << "[";
    for (int board_id : record.detection_result.requested_board_ids) {
      storage << board_id;
    }
    storage << "]";
    storage << "detections" << "[";
    for (const OuterTagDetectionResult& detection : record.detection_result.detections) {
      WriteDetection(&storage, detection);
    }
    storage << "]";
    return true;
  } catch (const std::exception& error) {
    if (warning != nullptr) {
      *warning = error.what();
    }
    return false;
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
