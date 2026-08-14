#include <aslam/cameras/apriltag_internal/Stage5BackendDiagnosticWriters.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagInternalDebugVisualization.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/Stage5Runtime.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

namespace fs = boost::filesystem;
namespace ati = aslam::cameras::apriltag_internal;

void EnsureDirectoryExists(const fs::path& directory) {
  if (!directory.empty()) {
    fs::create_directories(directory);
  }
}

std::string SanitizeFilenameComponent(const std::string& input) {
  std::string sanitized;
  sanitized.reserve(input.size());
  bool last_was_underscore = false;
  for (char ch : input) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch)) {
      sanitized.push_back(ch);
      last_was_underscore = false;
    } else if (!last_was_underscore) {
      sanitized.push_back('_');
      last_was_underscore = true;
    }
  }
  while (!sanitized.empty() && sanitized.back() == '_') {
    sanitized.pop_back();
  }
  return sanitized.empty() ? "case" : sanitized;
}

std::string CsvEscape(const std::string& value) {
  const bool needs_quotes =
      value.find(',') != std::string::npos ||
      value.find('"') != std::string::npos ||
      value.find('\n') != std::string::npos ||
      value.find('\r') != std::string::npos;
  if (!needs_quotes) {
    return value;
  }
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"') {
      escaped += "\"\"";
    } else {
      escaped += ch;
    }
  }
  escaped += "\"";
  return escaped;
}

double SafeRatio(int numerator, int denominator) {
  if (denominator <= 0) {
    return 0.0;
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

double MedianValue(std::vector<double> values) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](double value) {
                                return !std::isfinite(value);
                              }),
               values.end());
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1] + values[mid]);
}

double MinFiniteValue(const std::vector<double>& values) {
  double result = std::numeric_limits<double>::infinity();
  for (double value : values) {
    if (std::isfinite(value)) {
      result = std::min(result, value);
    }
  }
  return std::isfinite(result) ? result
                               : std::numeric_limits<double>::quiet_NaN();
}

double MaxFiniteValue(const std::vector<double>& values) {
  double result = -std::numeric_limits<double>::infinity();
  for (double value : values) {
    if (std::isfinite(value)) {
      result = std::max(result, value);
    }
  }
  return std::isfinite(result) ? result
                               : std::numeric_limits<double>::quiet_NaN();
}

struct GeometryPriorSeedRejectStats {
  int count = 0;
  int rectified_tag_success_count = 0;
  int roi_tag_success_count = 0;
  int image_evidence_success_count = 0;
  int pose_refit_success_count = 0;
  int accepted_count = 0;
  std::vector<double> outer_rmse;
  std::vector<double> max_corner_displacement;
  std::vector<double> min_corner_response_ratio;
  std::vector<double> edge_support_ratio;
};

void AccumulateGeometryPriorSeedRejectStats(
    const ati::GeometryPriorOuterSeedCandidate& candidate,
    GeometryPriorSeedRejectStats* stats) {
  if (stats == nullptr) {
    return;
  }
  ++stats->count;
  if (candidate.rectified_patch_decode_success &&
      candidate.rectified_patch_detected_tag_id == candidate.missing_board_id) {
    ++stats->rectified_tag_success_count;
  }
  if (candidate.roi_redetect_success &&
      candidate.roi_redetect_detected_tag_id == candidate.missing_board_id) {
    ++stats->roi_tag_success_count;
  }
  if (candidate.image_evidence_success) {
    ++stats->image_evidence_success_count;
  }
  if (candidate.pose_refit_success) {
    ++stats->pose_refit_success_count;
  }
  if (candidate.accepted_as_rescued_observation) {
    ++stats->accepted_count;
  }
  stats->outer_rmse.push_back(candidate.outer_reprojection_rmse);
  stats->max_corner_displacement.push_back(
      candidate.max_corner_displacement_px);
  stats->min_corner_response_ratio.push_back(
      candidate.min_corner_response_ratio);
  stats->edge_support_ratio.push_back(candidate.edge_support_ratio);
}

IntermediateCameraConfig MakeIntermediateCameraConfigForDiagnostics(
    const OuterBootstrapCameraIntrinsics& intrinsics) {
  IntermediateCameraConfig config;
  config.camera_model = intrinsics.camera_model;
  config.distortion_model = intrinsics.distortion_model;
  config.intrinsics = intrinsics.IntrinsicsVector();
  config.distortion_coeffs = intrinsics.DistortionVector();
  config.resolution = {intrinsics.resolution.width,
                       intrinsics.resolution.height};
  return config;
}

ApriltagCanonicalModel ModelForBoardIdForDiagnostics(
    const ApriltagInternalConfig& base_config,
    int board_id) {
  ApriltagInternalConfig config = base_config;
  config.tag_id = board_id;
  config.tag_ids.clear();
  config.outer_detector_config.tag_id = board_id;
  config.outer_detector_config.tag_ids.clear();
  return ApriltagCanonicalModel(config);
}

Eigen::Isometry3d PoseFromCvForDiagnostics(const cv::Mat& rvec,
                                           const cv::Mat& tvec) {
  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);
  cv::Mat rotation64;
  cv::Mat tvec64;
  rotation_cv.convertTo(rotation64, CV_64F);
  tvec.convertTo(tvec64, CV_64F);
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      pose.linear()(row, col) = rotation64.at<double>(row, col);
    }
    pose.translation()[row] = tvec64.at<double>(row, 0);
  }
  return pose;
}

bool EstimatePoseFromOuterCornersForDiagnostics(
    const DoubleSphereCameraModel& camera,
    const ApriltagCanonicalModel& model,
    const ati::OuterTagDetectionResult& outer_detection,
    Eigen::Isometry3d* pose,
    double* outer_rmse) {
  if (pose == nullptr || outer_rmse == nullptr || !outer_detection.success) {
    return false;
  }
  const std::array<int, 4> outer_point_ids{{
      model.PointId(0, 0),
      model.PointId(model.ModuleDimension(), 0),
      model.PointId(model.ModuleDimension(), model.ModuleDimension()),
      model.PointId(0, model.ModuleDimension()),
  }};
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(4);
  image_points.reserve(4);
  for (int corner_index = 0; corner_index < 4; ++corner_index) {
    if (!outer_detection.refined_valid[static_cast<std::size_t>(corner_index)]) {
      return false;
    }
    const CanonicalCorner& corner =
        model.corner(outer_point_ids[static_cast<std::size_t>(corner_index)]);
    object_points.emplace_back(static_cast<float>(corner.target_xyz.x()),
                               static_cast<float>(corner.target_xyz.y()),
                               static_cast<float>(corner.target_xyz.z()));
    const Eigen::Vector2d& xy =
        outer_detection.refined_corners_original_image
            [static_cast<std::size_t>(corner_index)];
    image_points.emplace_back(static_cast<float>(xy.x()),
                              static_cast<float>(xy.y()));
  }
  cv::Mat rvec;
  cv::Mat tvec;
  if (!camera.estimateTransformation(object_points, image_points, &rvec, &tvec)) {
    return false;
  }
  *pose = PoseFromCvForDiagnostics(rvec, tvec);
  double squared_error_sum = 0.0;
  for (std::size_t index = 0; index < object_points.size(); ++index) {
    const Eigen::Vector3d object_point(object_points[index].x,
                                       object_points[index].y,
                                       object_points[index].z);
    Eigen::Vector2d projected;
    if (!camera.vsEuclideanToKeypoint((*pose) * object_point, &projected)) {
      return false;
    }
    const Eigen::Vector2d observed(image_points[index].x, image_points[index].y);
    squared_error_sum += (projected - observed).squaredNorm();
  }
  *outer_rmse = std::sqrt(squared_error_sum /
                          static_cast<double>(object_points.size()));
  return true;
}

bool EstimatePoseFromReferencePointsForDiagnostics(
    const DoubleSphereCameraModel& camera,
    const std::vector<Eigen::Vector3d>& object_points_reference,
    const std::vector<Eigen::Vector2d>& image_points,
    Eigen::Isometry3d* pose,
    double* rmse) {
  if (pose == nullptr || rmse == nullptr ||
      object_points_reference.size() != image_points.size() ||
      object_points_reference.size() < 4) {
    return false;
  }
  std::vector<cv::Point3f> object_points_cv;
  std::vector<cv::Point2f> image_points_cv;
  object_points_cv.reserve(object_points_reference.size());
  image_points_cv.reserve(image_points.size());
  for (std::size_t index = 0; index < object_points_reference.size(); ++index) {
    const Eigen::Vector3d& object = object_points_reference[index];
    const Eigen::Vector2d& image = image_points[index];
    object_points_cv.emplace_back(static_cast<float>(object.x()),
                                  static_cast<float>(object.y()),
                                  static_cast<float>(object.z()));
    image_points_cv.emplace_back(static_cast<float>(image.x()),
                                 static_cast<float>(image.y()));
  }
  cv::Mat rvec;
  cv::Mat tvec;
  if (!camera.estimateTransformation(object_points_cv, image_points_cv,
                                     &rvec, &tvec)) {
    return false;
  }
  *pose = PoseFromCvForDiagnostics(rvec, tvec);
  double squared_error_sum = 0.0;
  int valid_count = 0;
  for (std::size_t index = 0; index < object_points_reference.size(); ++index) {
    Eigen::Vector2d projected;
    if (!camera.vsEuclideanToKeypoint(
            (*pose) * object_points_reference[index], &projected)) {
      continue;
    }
    squared_error_sum += (projected - image_points[index]).squaredNorm();
    ++valid_count;
  }
  if (valid_count <= 0) {
    return false;
  }
  *rmse = std::sqrt(squared_error_sum / static_cast<double>(valid_count));
  return true;
}

double RotationAngleDegForDiagnostics(const Eigen::Matrix3d& rotation) {
  const Eigen::AngleAxisd angle_axis(rotation);
  return std::abs(angle_axis.angle()) * 180.0 / M_PI;
}

double TranslationNormForDiagnostics(const Eigen::Isometry3d& transform) {
  return transform.translation().norm();
}

const JointSceneFrameState* FindFrameStateForDiagnostics(
    const JointReprojectionSceneState& scene_state,
    int frame_index) {
  for (const JointSceneFrameState& frame : scene_state.frames) {
    if (frame.frame_index == frame_index) {
      return &frame;
    }
  }
  return nullptr;
}

const JointSceneBoardState* FindBoardStateForDiagnostics(
    const JointReprojectionSceneState& scene_state,
    int board_id) {
  for (const JointSceneBoardState& board : scene_state.boards) {
    if (board.board_id == board_id) {
      return &board;
    }
  }
  return nullptr;
}

void AccumulateProjectionResidualForDiagnostics(
    const DoubleSphereCameraModel& camera,
    const Eigen::Isometry3d& T_camera_board,
    const Eigen::Vector3d& target_xyz_board,
    const Eigen::Vector2d& observed,
    double* total_sq,
    int* total_count,
    double* outer_sq,
    int* outer_count,
    double* internal_sq,
    int* internal_count,
    bool internal_point) {
  Eigen::Vector2d projected;
  double sq = 1e12;
  if (camera.vsEuclideanToKeypoint(T_camera_board * target_xyz_board,
                                   &projected)) {
    sq = (projected - observed).squaredNorm();
  }
  *total_sq += sq;
  ++(*total_count);
  if (internal_point) {
    *internal_sq += sq;
    ++(*internal_count);
  } else {
    *outer_sq += sq;
    ++(*outer_count);
  }
}

cv::Scalar SourceColorForGeometryPriorSeed(
    const std::string& prediction_source_label) {
  if (prediction_source_label.find("visible_refit") != std::string::npos) {
    return cv::Scalar(255, 80, 255);  // vivid magenta
  }
  if (prediction_source_label.find("optimized_scene") != std::string::npos) {
    return cv::Scalar(0, 215, 255);  // bright yellow
  }
  if (prediction_source_label.find("bootstrap") != std::string::npos) {
    return cv::Scalar(255, 180, 0);  // bright blue-orange
  }
  return cv::Scalar(255, 255, 255);  // white
}

std::string ShortGeometryPriorSourceLabel(
    const std::string& prediction_source_label) {
  if (prediction_source_label.find("visible_refit") != std::string::npos) {
    return "visible_refit";
  }
  if (prediction_source_label.find("optimized_scene") != std::string::npos) {
    return "optimized_scene";
  }
  if (prediction_source_label.find("bootstrap") != std::string::npos) {
    return "bootstrap";
  }
  return prediction_source_label.empty() ? "unknown" : prediction_source_label;
}

void DrawLegendEntry(cv::Mat* image,
                     const cv::Point& origin,
                     const cv::Scalar& color,
                     const std::string& label) {
  cv::rectangle(*image, cv::Rect(origin.x, origin.y - 12, 14, 14), color,
                cv::FILLED, cv::LINE_AA);
  cv::putText(*image, label,
              cv::Point(origin.x + 20, origin.y),
              cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(255, 255, 255), 1,
              cv::LINE_AA);
}

void DrawTextWithBackground(cv::Mat* image,
                            const std::string& text,
                            const cv::Point& origin,
                            double scale,
                            const cv::Scalar& fg_color,
                            const cv::Scalar& bg_color,
                            int thickness) {
  int baseline = 0;
  const cv::Size text_size = cv::getTextSize(
      text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
  const cv::Rect background_rect(
      origin.x - 2,
      origin.y - text_size.height - 4,
      text_size.width + 6,
      text_size.height + baseline + 6);
  cv::rectangle(*image, background_rect, bg_color, cv::FILLED, cv::LINE_AA);
  cv::putText(*image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale,
              fg_color, thickness, cv::LINE_AA);
}

struct InternalRegenerationFrameAggregate {
  std::string round_label;
  std::string split_label;
  int frame_index = -1;
  std::string frame_label;
  int visible_board_count = 0;
  int board_observation_count = 0;
  int successful_board_count = 0;
  int failed_board_count = 0;
  int attempted_internal_corner_count = 0;
  int valid_internal_corner_count = 0;
  double valid_internal_ratio = 0.0;
  double total_runtime_seconds = 0.0;
  int pose_rescue_attempt_count = 0;
  int pose_rescue_success_count = 0;
  int pose_rescue_used_count = 0;
  std::string failure_reasons;
};

struct InternalRegenerationAcceptanceAggregate {
  std::string round_label;
  std::string split_label;
  int outer_detected_board_observation_count = 0;
  int accepted_board_observation_count = 0;
  int failed_board_observation_count = 0;
  int dropped_board_observation_count = 0;
};

std::vector<InternalRegenerationFrameAggregate> BuildInternalRegenerationAggregates(
    const std::string& round_label,
    const std::string& split_label,
    const std::vector<ati::InternalRegenerationFrameResult>& frame_results) {
  std::vector<InternalRegenerationFrameAggregate> aggregates;
  aggregates.reserve(frame_results.size());
  for (const ati::InternalRegenerationFrameResult& frame : frame_results) {
    InternalRegenerationFrameAggregate aggregate;
    aggregate.round_label = round_label;
    aggregate.split_label = split_label;
    aggregate.frame_index = frame.frame_index;
    aggregate.frame_label = frame.frame_label;
    aggregate.visible_board_count =
        static_cast<int>(frame.visible_board_ids.size());
    aggregate.board_observation_count =
        static_cast<int>(frame.board_measurements.size());
    aggregate.total_runtime_seconds =
        frame.runtime_breakdown.pose_estimation_seconds +
        frame.runtime_breakdown.boundary_model_seconds +
        frame.runtime_breakdown.seed_search_seconds +
        frame.runtime_breakdown.ray_refine_seconds +
        frame.runtime_breakdown.image_evidence_seconds +
        frame.runtime_breakdown.subpix_seconds;
    aggregate.pose_rescue_attempt_count =
        frame.runtime_breakdown.pose_rescue_attempt_count;
    aggregate.pose_rescue_success_count =
        frame.runtime_breakdown.pose_rescue_success_count;
    aggregate.pose_rescue_used_count =
        frame.runtime_breakdown.pose_rescue_used_count;

    std::ostringstream failures;
    bool first_failure = true;
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      const ati::ApriltagInternalDetectionResult& detection =
          measurement.detection;
      aggregate.attempted_internal_corner_count +=
          detection.runtime_breakdown.attempted_internal_corner_count;
      aggregate.valid_internal_corner_count +=
          detection.runtime_breakdown.valid_internal_corner_count;
      if (detection.outer_detection.success && detection.success) {
        ++aggregate.successful_board_count;
      } else if (detection.outer_detection.success && !detection.success) {
        ++aggregate.failed_board_count;
        if (!first_failure) {
          failures << " | ";
        }
        failures << "board=" << measurement.board_id << ":"
                 << detection.failure_reason;
        first_failure = false;
      }
    }
    aggregate.valid_internal_ratio =
        SafeRatio(aggregate.valid_internal_corner_count,
                  aggregate.attempted_internal_corner_count);
    aggregate.failure_reasons = failures.str();
    aggregates.push_back(aggregate);
  }
  return aggregates;
}

}  // namespace

void WriteInternalRegenerationDiagnostics(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report) {
  const fs::path observations_path =
      output_dir / "internal_regeneration_observations.csv";
  const fs::path failures_path =
      output_dir / "internal_regeneration_failures.csv";
  const fs::path worst_frames_path =
      output_dir / "internal_regeneration_worst_frames.csv";
  const fs::path acceptance_summary_path =
      output_dir / "internal_regeneration_acceptance_summary.csv";

  std::ofstream observations(observations_path.string().c_str());
  std::ofstream failures(failures_path.string().c_str());
  observations
      << "round,split,state_source,frame_index,frame_label,board_id,"
      << "outer_detected,detection_success,failure_reason,"
      << "frame_bootstrap_initialized,board_bootstrap_initialized,pose_prior_used,"
      << "valid_corner_count,expected_visible_point_count,"
      << "attempted_internal_corner_count,valid_internal_corner_count,"
      << "valid_internal_ratio,total_seconds,pose_estimation_seconds,"
      << "boundary_model_seconds,seed_search_seconds,ray_refine_seconds,"
      << "image_evidence_seconds,subpix_seconds,pose_estimation_call_count,"
      << "boundary_model_build_count,border_boundary_model_valid,"
      << "border_boundary_model_failure_reason,"
      << "border_edge0_valid,border_edge1_valid,border_edge2_valid,border_edge3_valid,"
      << "border_edge0_support_count,border_edge1_support_count,"
      << "border_edge2_support_count,border_edge3_support_count,"
      << "border_edge0_support_ray_count,border_edge1_support_ray_count,"
      << "border_edge2_support_ray_count,border_edge3_support_ray_count,"
      << "border_edge0_rms,border_edge1_rms,border_edge2_rms,border_edge3_rms,"
      << "pose_rescue_attempted,pose_rescue_success,"
      << "pose_rescue_used,pose_rescue_rmse,pose_rescue_max_ray_angle_deg,"
      << "pose_rescue_ray_angle_limit_deg,"
      << "pose_rescue_accept_max_outer_rmse,pose_rescue_failure_reason\n";
  failures
      << "round,split,state_source,frame_index,frame_label,board_id,"
      << "failure_reason,border_boundary_model_valid,"
      << "border_boundary_model_failure_reason,"
      << "border_edge0_support_count,border_edge1_support_count,"
      << "border_edge2_support_count,border_edge3_support_count,"
      << "border_edge0_support_ray_count,border_edge1_support_ray_count,"
      << "border_edge2_support_ray_count,border_edge3_support_ray_count,"
      << "border_edge0_rms,border_edge1_rms,border_edge2_rms,border_edge3_rms,"
      << "pose_prior_used,pose_rescue_attempted,"
      << "pose_rescue_success,pose_rescue_used,pose_rescue_rmse,"
      << "pose_rescue_max_ray_angle_deg,pose_rescue_failure_reason\n";

  std::vector<InternalRegenerationFrameAggregate> all_aggregates;
  std::vector<InternalRegenerationAcceptanceAggregate> acceptance_aggregates;
  const bool strict_board_observation_acceptance =
      report.baseline_result.effective_options
          .strict_board_observation_acceptance;
  const auto write_round =
      [&](const std::string& round_label,
          const std::string& split_label,
          const std::vector<ati::InternalRegenerationFrameResult>& frame_results) {
        const std::vector<InternalRegenerationFrameAggregate> aggregates =
            BuildInternalRegenerationAggregates(round_label, split_label,
                                                frame_results);
        all_aggregates.insert(all_aggregates.end(), aggregates.begin(),
                              aggregates.end());
        InternalRegenerationAcceptanceAggregate acceptance;
        acceptance.round_label = round_label;
        acceptance.split_label = split_label;

        for (const ati::InternalRegenerationFrameResult& frame : frame_results) {
          for (const ati::RegeneratedBoardMeasurement& measurement :
               frame.board_measurements) {
            const ati::ApriltagInternalDetectionResult& detection =
                measurement.detection;
            if (detection.outer_detection.success) {
              ++acceptance.outer_detected_board_observation_count;
              if (detection.success) {
                ++acceptance.accepted_board_observation_count;
              } else {
                ++acceptance.failed_board_observation_count;
                if (strict_board_observation_acceptance) {
                  ++acceptance.dropped_board_observation_count;
                }
              }
            }
            const int attempted =
                detection.runtime_breakdown.attempted_internal_corner_count;
            const int valid =
                detection.runtime_breakdown.valid_internal_corner_count;
            const double valid_ratio = SafeRatio(valid, attempted);
            observations
                << round_label << ","
                << split_label << ","
                << frame.state_source_label << ","
                << frame.frame_index << ","
                << CsvEscape(frame.frame_label) << ","
                << measurement.board_id << ","
                << (detection.outer_detection.success ? 1 : 0) << ","
                << (detection.success ? 1 : 0) << ","
                << CsvEscape(detection.failure_reason) << ","
                << (measurement.frame_bootstrap_initialized ? 1 : 0) << ","
                << (measurement.board_bootstrap_initialized ? 1 : 0) << ","
                << (measurement.pose_prior_used ? 1 : 0) << ","
                << detection.valid_corner_count << ","
                << detection.expected_visible_point_count << ","
                << attempted << ","
                << valid << ","
                << valid_ratio << ","
                << detection.runtime_breakdown.total_seconds << ","
                << detection.runtime_breakdown.pose_estimation_seconds << ","
                << detection.runtime_breakdown.boundary_model_seconds << ","
                << detection.runtime_breakdown.seed_search_seconds << ","
                << detection.runtime_breakdown.ray_refine_seconds << ","
                << detection.runtime_breakdown.image_evidence_seconds << ","
                << detection.runtime_breakdown.subpix_seconds << ","
                << detection.runtime_breakdown.pose_estimation_call_count << ","
                << detection.runtime_breakdown.boundary_model_build_count << ","
                << (detection.border_boundary_model_valid ? 1 : 0) << ","
                << CsvEscape(detection.border_boundary_model_failure_reason)
                << ","
                << (detection.border_edge_valid[0] ? 1 : 0) << ","
                << (detection.border_edge_valid[1] ? 1 : 0) << ","
                << (detection.border_edge_valid[2] ? 1 : 0) << ","
                << (detection.border_edge_valid[3] ? 1 : 0) << ","
                << detection.border_edge_support_count[0] << ","
                << detection.border_edge_support_count[1] << ","
                << detection.border_edge_support_count[2] << ","
                << detection.border_edge_support_count[3] << ","
                << detection.border_edge_support_ray_count[0] << ","
                << detection.border_edge_support_ray_count[1] << ","
                << detection.border_edge_support_ray_count[2] << ","
                << detection.border_edge_support_ray_count[3] << ","
                << detection.border_edge_rms_residual[0] << ","
                << detection.border_edge_rms_residual[1] << ","
                << detection.border_edge_rms_residual[2] << ","
                << detection.border_edge_rms_residual[3] << ","
                << (detection.pose_rescue_attempted ? 1 : 0) << ","
                << (detection.pose_rescue_success ? 1 : 0) << ","
                << (detection.pose_rescue_used ? 1 : 0) << ","
                << detection.pose_rescue_rmse << ","
                << detection.pose_rescue_max_ray_angle_deg << ","
                << detection.pose_rescue_ray_angle_limit_deg << ","
                << detection.pose_rescue_accept_max_outer_rmse << ","
                << CsvEscape(detection.pose_rescue_failure_reason) << "\n";

            if (detection.outer_detection.success && !detection.success) {
              failures
                  << round_label << ","
                  << split_label << ","
                  << frame.state_source_label << ","
                  << frame.frame_index << ","
                  << CsvEscape(frame.frame_label) << ","
                  << measurement.board_id << ","
                  << CsvEscape(detection.failure_reason) << ","
                  << (detection.border_boundary_model_valid ? 1 : 0) << ","
                  << CsvEscape(detection.border_boundary_model_failure_reason)
                  << ","
                  << detection.border_edge_support_count[0] << ","
                  << detection.border_edge_support_count[1] << ","
                  << detection.border_edge_support_count[2] << ","
                  << detection.border_edge_support_count[3] << ","
                  << detection.border_edge_support_ray_count[0] << ","
                  << detection.border_edge_support_ray_count[1] << ","
                  << detection.border_edge_support_ray_count[2] << ","
                  << detection.border_edge_support_ray_count[3] << ","
                  << detection.border_edge_rms_residual[0] << ","
                  << detection.border_edge_rms_residual[1] << ","
                  << detection.border_edge_rms_residual[2] << ","
                  << detection.border_edge_rms_residual[3] << ","
                  << (measurement.pose_prior_used ? 1 : 0) << ","
                  << (detection.pose_rescue_attempted ? 1 : 0) << ","
                  << (detection.pose_rescue_success ? 1 : 0) << ","
                  << (detection.pose_rescue_used ? 1 : 0) << ","
                  << detection.pose_rescue_rmse << ","
                  << detection.pose_rescue_max_ray_angle_deg << ","
                  << CsvEscape(detection.pose_rescue_failure_reason) << "\n";
            }
          }
        }
        acceptance_aggregates.push_back(acceptance);
      };

  write_round("round1", "training",
              report.baseline_result.round1.regeneration_results);
  if (report.baseline_result.round2_available) {
    write_round("round2", "training",
                report.baseline_result.round2.regeneration_results);
  }
  write_round("holdout", "holdout",
              report.holdout_dataset.internal_regeneration_results);

  std::ofstream worst_frames(worst_frames_path.string().c_str());
  worst_frames
      << "ranking_metric,rank,round,split,frame_index,frame_label,"
      << "visible_board_count,board_observation_count,successful_board_count,"
      << "failed_board_count,attempted_internal_corner_count,"
      << "valid_internal_corner_count,valid_internal_ratio,total_runtime_seconds,"
      << "pose_rescue_attempt_count,pose_rescue_success_count,"
      << "pose_rescue_used_count,failure_reasons\n";

  const auto write_ranked =
      [&](const std::string& metric,
          std::vector<InternalRegenerationFrameAggregate> ranked) {
        if (metric == "failed_boards") {
          std::sort(ranked.begin(), ranked.end(),
                    [](const InternalRegenerationFrameAggregate& lhs,
                       const InternalRegenerationFrameAggregate& rhs) {
                      if (lhs.failed_board_count != rhs.failed_board_count) {
                        return lhs.failed_board_count > rhs.failed_board_count;
                      }
                      return lhs.valid_internal_ratio < rhs.valid_internal_ratio;
                    });
        } else if (metric == "low_valid_ratio") {
          std::sort(ranked.begin(), ranked.end(),
                    [](const InternalRegenerationFrameAggregate& lhs,
                       const InternalRegenerationFrameAggregate& rhs) {
                      if (lhs.valid_internal_ratio != rhs.valid_internal_ratio) {
                        return lhs.valid_internal_ratio < rhs.valid_internal_ratio;
                      }
                      return lhs.failed_board_count > rhs.failed_board_count;
                    });
        } else {
          std::sort(ranked.begin(), ranked.end(),
                    [](const InternalRegenerationFrameAggregate& lhs,
                       const InternalRegenerationFrameAggregate& rhs) {
                      return lhs.total_runtime_seconds > rhs.total_runtime_seconds;
                    });
        }
        const std::size_t top_k = std::min<std::size_t>(20, ranked.size());
        for (std::size_t index = 0; index < top_k; ++index) {
          const InternalRegenerationFrameAggregate& row = ranked[index];
          worst_frames
              << metric << ","
              << (index + 1) << ","
              << row.round_label << ","
              << row.split_label << ","
              << row.frame_index << ","
              << CsvEscape(row.frame_label) << ","
              << row.visible_board_count << ","
              << row.board_observation_count << ","
              << row.successful_board_count << ","
              << row.failed_board_count << ","
              << row.attempted_internal_corner_count << ","
              << row.valid_internal_corner_count << ","
              << row.valid_internal_ratio << ","
              << row.total_runtime_seconds << ","
              << row.pose_rescue_attempt_count << ","
              << row.pose_rescue_success_count << ","
              << row.pose_rescue_used_count << ","
              << CsvEscape(row.failure_reasons) << "\n";
        }
      };

  write_ranked("failed_boards", all_aggregates);
  write_ranked("low_valid_ratio", all_aggregates);
  write_ranked("runtime_seconds", all_aggregates);

  std::ofstream acceptance_summary(
      acceptance_summary_path.string().c_str());
  acceptance_summary
      << "round,split,strict_board_observation_acceptance,"
      << "failed_board_drop_policy,outer_detected_board_observation_count,"
      << "accepted_board_observation_count,failed_board_observation_count,"
      << "dropped_board_observation_count\n";
  for (const InternalRegenerationAcceptanceAggregate& row :
       acceptance_aggregates) {
    acceptance_summary
        << row.round_label << ","
        << row.split_label << ","
        << (strict_board_observation_acceptance ? 1 : 0) << ","
        << (strict_board_observation_acceptance
                ? "drop_entire_board_observation"
                : "keep_outer_when_internal_failed")
        << ","
        << row.outer_detected_board_observation_count << ","
        << row.accepted_board_observation_count << ","
        << row.failed_board_observation_count << ","
        << row.dropped_board_observation_count << "\n";
  }
}

std::string JoinInts(const std::vector<int>& values) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      stream << ";";
    }
    stream << values[index];
  }
  return stream.str();
}

std::map<int, std::string> BuildTrainingFrameImagePathMap(
    const ati::Stage5BenchmarkReport& report) {
  std::map<int, std::string> image_paths;
  for (const ati::FrozenRound2BaselineFrameSource& source :
       report.baseline_result.frame_sources) {
    if (!source.image_path.empty()) {
      image_paths[source.frame_index] = source.image_path;
    }
  }
  return image_paths;
}

void WriteInternalSeedStepOverlaysForRound(
    const fs::path& output_dir,
    const std::string& round_label,
    const std::vector<ati::InternalRegenerationFrameResult>& frame_results,
    const std::map<int, std::string>& image_paths,
    int max_overlays_per_round,
    std::ofstream* index_csv,
    int* rendered_count) {
  if (index_csv == nullptr || rendered_count == nullptr) {
    return;
  }
  const fs::path round_dir =
      output_dir / "internal_seed_step_overlays" /
      SanitizeFilenameComponent(round_label);
  EnsureDirectoryExists(round_dir);

  int rendered_for_round = 0;
  for (const ati::InternalRegenerationFrameResult& frame : frame_results) {
    const auto path_it = image_paths.find(frame.frame_index);
    if (path_it == image_paths.end()) {
      continue;
    }
    cv::Mat image;
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      const ati::ApriltagInternalDetectionResult& detection =
          measurement.detection;
      if (!detection.outer_detection.success ||
          detection.internal_corner_debug.empty()) {
        continue;
      }
      if (max_overlays_per_round > 0 &&
          rendered_for_round >= max_overlays_per_round) {
        return;
      }
      if (image.empty()) {
        image = cv::imread(path_it->second, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
          break;
        }
      }
      cv::Mat overlay = ati::BuildInternalSeedOverlay(image, detection);
      if (overlay.empty()) {
        continue;
      }
      cv::rectangle(overlay, cv::Point(8, 118), cv::Point(1000, 220),
                    cv::Scalar(0, 0, 0), cv::FILLED);
      std::ostringstream header;
      header << round_label << " frame=" << frame.frame_index
             << " board=" << measurement.board_id
             << " valid=" << detection.valid_internal_corner_count << "/"
             << detection.runtime_breakdown.attempted_internal_corner_count
             << " boundary="
             << (detection.border_boundary_model_valid ? "ok" : "fail")
             << " failure=" << detection.failure_reason;
      cv::putText(overlay, header.str(), cv::Point(16, 142),
                  cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(255, 255, 255),
                  1, cv::LINE_AA);
      std::ostringstream border;
      border << "edge support="
             << detection.border_edge_support_count[0] << "/"
             << detection.border_edge_support_count[1] << "/"
             << detection.border_edge_support_count[2] << "/"
             << detection.border_edge_support_count[3]
             << " ray="
             << detection.border_edge_support_ray_count[0] << "/"
             << detection.border_edge_support_ray_count[1] << "/"
             << detection.border_edge_support_ray_count[2] << "/"
             << detection.border_edge_support_ray_count[3]
             << " rms=" << std::fixed << std::setprecision(2)
             << detection.border_edge_rms_residual[0] << "/"
             << detection.border_edge_rms_residual[1] << "/"
             << detection.border_edge_rms_residual[2] << "/"
             << detection.border_edge_rms_residual[3];
      cv::putText(overlay, border.str(), cv::Point(16, 166),
                  cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(255, 255, 255),
                  1, cv::LINE_AA);
      cv::putText(
          overlay,
          "orange=predicted, blue triangle=border seed, magenta diamond=sphere seed, green square=refined",
          cv::Point(16, 190), cv::FONT_HERSHEY_SIMPLEX, 0.42,
          cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

      std::ostringstream filename;
      filename << "frame_" << frame.frame_index << "_"
               << SanitizeFilenameComponent(frame.frame_label)
               << "_board_" << measurement.board_id << "_seed_steps.png";
      const fs::path overlay_path = round_dir / filename.str();
      cv::imwrite(overlay_path.string(), overlay);
      (*index_csv) << round_label << ","
                   << frame.frame_index << ","
                   << CsvEscape(frame.frame_label) << ","
                   << measurement.board_id << ","
                   << (detection.success ? 1 : 0) << ","
                   << CsvEscape(detection.failure_reason) << ","
                   << detection.valid_internal_corner_count << ","
                   << detection.runtime_breakdown.attempted_internal_corner_count
                   << ","
                   << (detection.border_boundary_model_valid ? 1 : 0) << ","
                   << CsvEscape(detection.border_boundary_model_failure_reason)
                   << ","
                   << detection.border_edge_support_count[0] << ","
                   << detection.border_edge_support_count[1] << ","
                   << detection.border_edge_support_count[2] << ","
                   << detection.border_edge_support_count[3] << ","
                   << detection.border_edge_support_ray_count[0] << ","
                   << detection.border_edge_support_ray_count[1] << ","
                   << detection.border_edge_support_ray_count[2] << ","
                   << detection.border_edge_support_ray_count[3] << ","
                   << CsvEscape(overlay_path.string()) << "\n";
      ++rendered_for_round;
      ++(*rendered_count);
    }
  }
}

void WriteInternalSeedStepOverlays(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report,
    const std::vector<ati::FrozenRound2BaselineFrameSource>& all_frames_for_lookup) {
  const std::map<int, std::string> image_paths =
      [&]() {
        std::map<int, std::string> paths;
        for (const ati::FrozenRound2BaselineFrameSource& frame :
             all_frames_for_lookup) {
          paths[frame.frame_index] = frame.image_path;
        }
        return paths;
      }();
  const fs::path overlay_root = output_dir / "internal_seed_step_overlays";
  EnsureDirectoryExists(overlay_root);
  std::ofstream index_csv((overlay_root / "index.csv").string().c_str());
  index_csv
      << "round,frame_index,frame_label,board_id,detection_success,"
      << "failure_reason,valid_internal_corner_count,"
      << "attempted_internal_corner_count,border_boundary_model_valid,"
      << "border_boundary_model_failure_reason,border_edge0_support_count,"
      << "border_edge1_support_count,border_edge2_support_count,"
      << "border_edge3_support_count,border_edge0_support_ray_count,"
      << "border_edge1_support_ray_count,border_edge2_support_ray_count,"
      << "border_edge3_support_ray_count,overlay_path\n";

  int rendered_count = 0;
  constexpr int kMaxOverlaysPerRound = 400;
  WriteInternalSeedStepOverlaysForRound(
      output_dir, "round1_training",
      report.baseline_result.round1.regeneration_results, image_paths,
      kMaxOverlaysPerRound, &index_csv, &rendered_count);
  if (report.baseline_result.round2_available) {
    WriteInternalSeedStepOverlaysForRound(
        output_dir, "round2_training",
        report.baseline_result.round2.regeneration_results, image_paths,
        kMaxOverlaysPerRound, &index_csv, &rendered_count);
  }
  WriteInternalSeedStepOverlaysForRound(
      output_dir, "holdout",
      report.holdout_dataset.internal_regeneration_results, image_paths,
      kMaxOverlaysPerRound, &index_csv, &rendered_count);

  std::ofstream summary((overlay_root / "summary.txt").string().c_str());
  summary << "description: detect_apriltag_internal-style internal seed step overlays\n";
  summary << "rendered_overlay_count: " << rendered_count << "\n";
  summary << "legend: orange=predicted, blue triangle=border seed, magenta diamond=sphere seed, green square=refined\n";
  summary << "note: border curves and support points are drawn when the border boundary model is available.\n";
}

void DrawGeometryPriorSeedCandidate(
    cv::Mat* image,
    const ati::GeometryPriorOuterSeedCandidate& candidate) {
  if (image == nullptr || image->empty()) {
    return;
  }
  const cv::Scalar predicted_color =
      SourceColorForGeometryPriorSeed(candidate.prediction_source_label);
  const cv::Scalar refined_color(80, 255, 120);
  const cv::Scalar subpix_window_color(80, 220, 255);
  const cv::Scalar rejected_color(60, 60, 255);
  const cv::Scalar accepted_outline_color(0, 255, 0);
  const cv::Scalar rejected_outline_color(0, 0, 255);
  const cv::Scalar text_fg_color(255, 255, 255);
  const cv::Scalar text_bg_color = candidate.accepted_as_rescued_observation
                                       ? cv::Scalar(20, 90, 20)
                                       : cv::Scalar(70, 20, 20);
  const cv::Scalar status_color = candidate.accepted_as_rescued_observation
                                      ? accepted_outline_color
                                      : rejected_outline_color;
  std::vector<cv::Point> polygon;
  polygon.reserve(4);
  for (const cv::Point2f& point : candidate.predicted_corners) {
    polygon.emplace_back(static_cast<int>(std::lround(point.x)),
                         static_cast<int>(std::lround(point.y)));
  }
  if (polygon.size() == 4) {
    cv::polylines(*image, polygon, true, predicted_color, 4, cv::LINE_AA);
  }
  std::vector<cv::Point> refined_polygon;
  refined_polygon.reserve(4);
  for (const cv::Point2f& point : candidate.refined_corners) {
    refined_polygon.emplace_back(static_cast<int>(std::lround(point.x)),
                                 static_cast<int>(std::lround(point.y)));
  }
  if (candidate.image_evidence_checked && refined_polygon.size() == 4) {
    const cv::Scalar refined_outline_color =
        candidate.accepted_as_rescued_observation ? refined_color
                                                  : rejected_color;
    cv::polylines(*image, refined_polygon, true, refined_outline_color,
                  candidate.accepted_as_rescued_observation ? 3 : 2,
                  cv::LINE_AA);
  }
  for (int index = 0; index < 4; ++index) {
    const cv::Point2f predicted =
        candidate.predicted_corners[static_cast<std::size_t>(index)];
    const cv::Point2f refined =
        candidate.refined_corners[static_cast<std::size_t>(index)];
    cv::drawMarker(*image, predicted, predicted_color,
                   cv::MARKER_TILTED_CROSS, 20, 2, cv::LINE_AA);
    if (candidate.image_evidence_checked) {
      const int radius = std::max(1, candidate.subpix_window_radius);
      const cv::Rect window_rect(
          static_cast<int>(std::lround(refined.x)) - radius,
          static_cast<int>(std::lround(refined.y)) - radius,
          radius * 2 + 1,
          radius * 2 + 1);
      cv::rectangle(*image, window_rect, subpix_window_color, 1, cv::LINE_AA);
      cv::circle(*image, refined, candidate.accepted_as_rescued_observation ? 6 : 4,
                 candidate.accepted_as_rescued_observation ? refined_color
                                                          : rejected_color,
                 candidate.accepted_as_rescued_observation ? 2 : 1,
                 cv::LINE_AA);
      cv::line(*image, predicted, refined,
               candidate.accepted_as_rescued_observation ? refined_color
                                                         : rejected_color,
               2, cv::LINE_AA);
      std::ostringstream radius_label;
      radius_label << "r=" << radius;
      cv::putText(*image, radius_label.str(),
                  cv::Point(static_cast<int>(std::lround(refined.x)) + radius + 3,
                            static_cast<int>(std::lround(refined.y)) - radius - 3),
                  cv::FONT_HERSHEY_SIMPLEX, 0.42, subpix_window_color, 1, cv::LINE_AA);
    }
  }
  const cv::Point2f anchor = candidate.predicted_corners[0] +
                             cv::Point2f(12.0f, -12.0f);
  std::ostringstream label;
  label << "board " << candidate.missing_board_id
        << " | src=" << ShortGeometryPriorSourceLabel(candidate.prediction_source_label)
        << " | " << (candidate.accepted_as_rescued_observation ? "ACCEPT" : "REJECT")
        << " | outer_rmse=" << std::fixed << std::setprecision(2)
        << candidate.outer_reprojection_rmse
        << " | pred_disp=" << candidate.max_corner_displacement_px
        << " | refine_disp=" << candidate.max_refinement_displacement_px
        << "/" << candidate.adaptive_max_corner_displacement_px
        << " | subpix=" << candidate.subpix_window_radius
        << " | scale=" << candidate.local_corner_scale_px
        << " | topology=" << (candidate.quad_topology_preserved ? 1 : 0)
        << " | id=" << (candidate.tag_id_validated ? 1 : 0)
        << " | resp=" << candidate.min_corner_response_ratio
        << " | edge=" << candidate.edge_support_ratio
        << "/" << candidate.mean_edge_gradient_ratio;
  DrawTextWithBackground(
      image, label.str(),
      cv::Point(static_cast<int>(anchor.x), static_cast<int>(anchor.y)),
      0.45, text_fg_color, text_bg_color, 1);

  if (!candidate.reject_reason.empty() &&
      !candidate.accepted_as_rescued_observation) {
    const cv::Point reject_anchor(
        static_cast<int>(std::lround(candidate.predicted_corners[0].x)),
        static_cast<int>(std::lround(candidate.predicted_corners[0].y)) - 28);
    DrawTextWithBackground(image,
                           "reason: " + candidate.reject_reason,
                           reject_anchor, 0.38, text_fg_color,
                           cv::Scalar(20, 20, 120), 1);
  }

  if (!candidate.accepted_as_rescued_observation && polygon.size() == 4) {
    for (int index = 0; index < 4; ++index) {
      cv::line(*image, polygon[static_cast<std::size_t>(index)],
               polygon[static_cast<std::size_t>((index + 2) % 4)],
               status_color, 1, cv::LINE_AA);
    }
  }
}

void WriteGeometryPriorOuterSeedOverlays(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report,
    const std::vector<ati::InternalRegenerationFrameResult>& frame_results,
    const std::string& round_label) {
  const std::map<int, std::string> image_paths =
      BuildTrainingFrameImagePathMap(report);
  const fs::path overlay_dir =
      output_dir / "geometry_prior_outer_seed_overlays" / round_label;
  fs::create_directories(overlay_dir);
  int rendered_count = 0;
  for (const ati::InternalRegenerationFrameResult& frame : frame_results) {
    if (frame.geometry_prior_outer_seed_candidates.empty()) {
      continue;
    }
    const auto path_it = image_paths.find(frame.frame_index);
    if (path_it == image_paths.end()) {
      continue;
    }
    cv::Mat image = cv::imread(path_it->second, cv::IMREAD_COLOR);
    if (image.empty()) {
      continue;
    }
    const cv::Rect legend_box(10, 10, 330, 125);
    cv::rectangle(image, legend_box, cv::Scalar(15, 15, 15), cv::FILLED,
                  cv::LINE_AA);
    cv::rectangle(image, legend_box, cv::Scalar(255, 255, 255), 1,
                  cv::LINE_AA);
    DrawLegendEntry(&image, cv::Point(22, 32),
                    SourceColorForGeometryPriorSeed("bootstrap"),
                    "bootstrap source");
    DrawLegendEntry(&image, cv::Point(22, 56),
                    SourceColorForGeometryPriorSeed("optimized_scene"),
                    "optimized_scene source");
    DrawLegendEntry(&image, cv::Point(22, 80),
                    SourceColorForGeometryPriorSeed("visible_refit"),
                    "visible_refit source");
    DrawLegendEntry(&image, cv::Point(22, 104), cv::Scalar(0, 255, 0),
                    "ACCEPT");
    DrawLegendEntry(&image, cv::Point(170, 104), cv::Scalar(0, 0, 255),
                    "REJECT");
    for (const ati::GeometryPriorOuterSeedCandidate& candidate :
         frame.geometry_prior_outer_seed_candidates) {
      DrawGeometryPriorSeedCandidate(&image, candidate);
    }
    std::ostringstream filename;
    filename << "frame_" << std::setw(5) << std::setfill('0')
             << frame.frame_index << "_geometry_prior_seed.png";
    cv::imwrite((overlay_dir / filename.str()).string(), image);
    ++rendered_count;
  }
  std::ofstream summary(
      (overlay_dir / "overlay_summary.txt").string().c_str());
  summary << "round: " << round_label << "\n";
  summary << "rendered_frame_count: " << rendered_count << "\n";
  summary << "legend:\n";
  summary << "  bootstrap: bright blue-orange, thicker predicted contour\n";
  summary << "  optimized_scene: bright yellow, optimized geometry prior\n";
  summary << "  visible_refit: vivid magenta, frame-pose-refit-guided prediction\n";
  summary << "  ACCEPT: green overlay text/background\n";
  summary << "  REJECT: red overlay text/background\n";
}

void WriteGeometryPriorOuterSeedDiagnostics(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report) {
  std::ofstream csv(
      (output_dir / "geometry_prior_outer_seed_candidates.csv")
          .string()
          .c_str());
  csv << "round,split,state_source,frame_index,frame_label,missing_board_id,"
      << "prediction_source,frame_pose_refit_source_board_id,"
      << "frame_pose_refit_outer_rmse,visible_boards_used,"
      << "predicted_corner_u0,predicted_corner_v0,"
      << "predicted_corner_u1,predicted_corner_v1,"
      << "predicted_corner_u2,predicted_corner_v2,"
      << "predicted_corner_u3,predicted_corner_v3,"
      << "refined_corner_u0,refined_corner_v0,"
      << "refined_corner_u1,refined_corner_v1,"
      << "refined_corner_u2,refined_corner_v2,"
      << "refined_corner_u3,refined_corner_v3,"
      << "predicted_area_px,predicted_signed_area_px,"
      << "refined_area_px,refined_signed_area_px,"
      << "refined_to_predicted_area_ratio,"
      << "predicted_quad_topology_valid,refined_quad_topology_valid,"
      << "quad_topology_preserved,quad_topology_summary,"
      << "local_corner_scale_px,subpix_window_radius,"
      << "spherical_refine_attempted,spherical_refine_success,"
      << "spherical_refine_successful_corner_count,"
      << "spherical_refine_max_displacement_px,"
      << "spherical_refine_min_quality,"
      << "spherical_refine_min_support_count,"
      << "spherical_refine_max_residual,"
      << "spherical_refine_failure_summary,"
      << "max_corner_displacement_px,max_refinement_displacement_px,"
      << "adaptive_max_corner_displacement_px,"
      << "min_corner_response_ratio,edge_support_ratio,"
      << "mean_edge_gradient_ratio,rectified_patch_checked,"
      << "rectified_patch_decode_success,rectified_patch_detected_tag_id,"
      << "rectified_patch_hamming,rectified_patch_summary,"
      << "geometry_guided_tag_likelihood_checked,"
      << "geometry_guided_tag_likelihood_passed,"
      << "geometry_guided_tag_likelihood_mode,"
      << "geometry_guided_tag_likelihood_expected_hamming,"
      << "geometry_guided_tag_likelihood_runner_up_id,"
      << "geometry_guided_tag_likelihood_runner_up_hamming,"
      << "geometry_guided_tag_likelihood_hamming_margin,"
      << "geometry_guided_tag_likelihood_contrast,"
      << "geometry_guided_tag_likelihood_summary,"
      << "roi_redetect_checked,roi_redetect_success,"
      << "roi_redetect_detected_tag_id,roi_redetect_hamming,"
      << "roi_redetect_bbox_x,roi_redetect_bbox_y,"
      << "roi_redetect_bbox_width,roi_redetect_bbox_height,"
      << "roi_redetect_summary,"
      << "roi_valid,image_evidence_checked,tag_id_validated,"
      << "image_evidence_success,local_redetect_success,"
      << "local_corner_refine_success,pose_refit_success,"
      << "local_vs_global_rotation_error_deg,"
      << "local_vs_global_translation_error,outer_reprojection_rmse,"
      << "frame_normal_outer_refit_rmse_median,"
      << "adaptive_accept_max_outer_rmse,"
      << "accepted_as_rescued_observation,reject_reason\n";

  int normal_detected_outer_count = 0;
  int geometry_prior_seed_count = 0;
  int image_validated_rescued_count = 0;
  int backend_used_rescued_count = 0;
  std::map<std::string, GeometryPriorSeedRejectStats> seed_stats_by_reason;
  std::vector<ati::GeometryPriorOuterSeedCandidate> rejected_candidates;

  const auto write_round =
      [&](const std::string& round_label,
          const std::string& split_label,
          const std::vector<ati::InternalRegenerationFrameResult>& frame_results) {
        for (const ati::InternalRegenerationFrameResult& frame : frame_results) {
          for (const ati::RegeneratedBoardMeasurement& measurement :
               frame.board_measurements) {
            if (measurement.detection.outer_detection.success &&
                !measurement.detection.outer_detection.used_local_patch_rescue) {
              ++normal_detected_outer_count;
            }
            if (measurement.detection.outer_detection.success &&
                measurement.detection.outer_detection.used_local_patch_rescue) {
              ++backend_used_rescued_count;
            }
          }
          for (const ati::GeometryPriorOuterSeedCandidate& candidate :
               frame.geometry_prior_outer_seed_candidates) {
            ++geometry_prior_seed_count;
            if (candidate.accepted_as_rescued_observation) {
              ++image_validated_rescued_count;
            }
            const std::string reject_reason =
                candidate.accepted_as_rescued_observation
                    ? "accepted"
                    : (candidate.reject_reason.empty()
                           ? "empty_reject_reason"
                           : candidate.reject_reason);
            AccumulateGeometryPriorSeedRejectStats(
                candidate, &seed_stats_by_reason[reject_reason]);
            if (!candidate.accepted_as_rescued_observation) {
              rejected_candidates.push_back(candidate);
            }
            csv << round_label << ","
                << split_label << ","
                << (candidate.prediction_source_label.empty()
                        ? frame.state_source_label
                        : candidate.prediction_source_label)
                << ","
                << candidate.frame_index << ","
                << CsvEscape(candidate.frame_label) << ","
                << candidate.missing_board_id << ","
                << CsvEscape(candidate.prediction_source_label) << ","
                << candidate.frame_pose_refit_source_board_id << ","
                << candidate.frame_pose_refit_outer_rmse << ","
                << CsvEscape(JoinInts(candidate.visible_boards_used)) << ",";
            for (int index = 0; index < 4; ++index) {
              csv << candidate
                         .predicted_corners[static_cast<std::size_t>(index)]
                         .x
                  << ","
                  << candidate
                         .predicted_corners[static_cast<std::size_t>(index)]
                         .y
                  << ",";
            }
            for (int index = 0; index < 4; ++index) {
              csv << candidate
                         .refined_corners[static_cast<std::size_t>(index)]
                         .x
                  << ","
                  << candidate
                         .refined_corners[static_cast<std::size_t>(index)]
                         .y
                  << ",";
            }
            csv << candidate.predicted_area_px << ","
                << candidate.predicted_signed_area_px << ","
                << candidate.refined_area_px << ","
                << candidate.refined_signed_area_px << ","
                << candidate.refined_to_predicted_area_ratio << ","
                << (candidate.predicted_quad_topology_valid ? 1 : 0) << ","
                << (candidate.refined_quad_topology_valid ? 1 : 0) << ","
                << (candidate.quad_topology_preserved ? 1 : 0) << ","
                << CsvEscape(candidate.quad_topology_summary) << ","
                << candidate.local_corner_scale_px << ","
                << candidate.subpix_window_radius << ","
                << (candidate.spherical_refine_attempted ? 1 : 0) << ","
                << (candidate.spherical_refine_success ? 1 : 0) << ","
                << candidate.spherical_refine_successful_corner_count << ","
                << candidate.spherical_refine_max_displacement_px << ","
                << candidate.spherical_refine_min_quality << ","
                << candidate.spherical_refine_min_support_count << ","
                << candidate.spherical_refine_max_residual << ","
                << CsvEscape(candidate.spherical_refine_failure_summary) << ","
                << candidate.max_corner_displacement_px << ","
                << candidate.max_refinement_displacement_px << ","
                << candidate.adaptive_max_corner_displacement_px << ","
                << candidate.min_corner_response_ratio << ","
                << candidate.edge_support_ratio << ","
                << candidate.mean_edge_gradient_ratio << ","
                << (candidate.rectified_patch_checked ? 1 : 0) << ","
                << (candidate.rectified_patch_decode_success ? 1 : 0) << ","
                << candidate.rectified_patch_detected_tag_id << ","
                << candidate.rectified_patch_hamming << ","
                << CsvEscape(candidate.rectified_patch_summary) << ","
                << (candidate.geometry_guided_tag_likelihood_checked ? 1 : 0)
                << ","
                << (candidate.geometry_guided_tag_likelihood_passed ? 1 : 0)
                << ","
                << CsvEscape(candidate.geometry_guided_tag_likelihood_mode) << ","
                << candidate.geometry_guided_tag_likelihood_expected_hamming
                << ","
                << candidate.geometry_guided_tag_likelihood_runner_up_id << ","
                << candidate.geometry_guided_tag_likelihood_runner_up_hamming
                << ","
                << candidate.geometry_guided_tag_likelihood_hamming_margin
                << ","
                << candidate.geometry_guided_tag_likelihood_contrast << ","
                << CsvEscape(candidate.geometry_guided_tag_likelihood_summary)
                << ","
                << (candidate.roi_redetect_checked ? 1 : 0) << ","
                << (candidate.roi_redetect_success ? 1 : 0) << ","
                << candidate.roi_redetect_detected_tag_id << ","
                << candidate.roi_redetect_hamming << ","
                << candidate.roi_redetect_bbox.x << ","
                << candidate.roi_redetect_bbox.y << ","
                << candidate.roi_redetect_bbox.width << ","
                << candidate.roi_redetect_bbox.height << ","
                << CsvEscape(candidate.roi_redetect_summary) << ","
                << (candidate.roi_valid ? 1 : 0) << ","
                << (candidate.image_evidence_checked ? 1 : 0) << ","
                << (candidate.tag_id_validated ? 1 : 0) << ","
                << (candidate.image_evidence_success ? 1 : 0) << ","
                << (candidate.local_redetect_success ? 1 : 0) << ","
                << (candidate.local_corner_refine_success ? 1 : 0) << ","
                << (candidate.pose_refit_success ? 1 : 0) << ","
                << candidate.local_vs_global_rotation_error_deg << ","
                << candidate.local_vs_global_translation_error << ","
                << candidate.outer_reprojection_rmse << ","
                << candidate.frame_normal_outer_refit_rmse_median << ","
                << candidate.adaptive_accept_max_outer_rmse << ","
                << (candidate.accepted_as_rescued_observation ? 1 : 0)
                << ","
                << CsvEscape(candidate.reject_reason) << "\n";
          }
        }
      };

  write_round("round1", "training",
              report.baseline_result.round1.regeneration_results);
  if (report.baseline_result.round2_available) {
    write_round("round2", "training",
                report.baseline_result.round2.regeneration_results);
  }
  write_round("holdout", "holdout",
              report.holdout_dataset.internal_regeneration_results);

  WriteGeometryPriorOuterSeedOverlays(
      output_dir, report, report.baseline_result.round1.regeneration_results,
      "round1_training");
  if (report.baseline_result.round2_available) {
    WriteGeometryPriorOuterSeedOverlays(
        output_dir, report, report.baseline_result.round2.regeneration_results,
        "round2_training");
  }

  std::ofstream summary(
      (output_dir / "geometry_prior_outer_seed_summary.txt")
          .string()
          .c_str());
  summary << "diagnostic_only: "
          << (report.baseline_result.effective_options
                      .geometry_prior_rescue_diagnostic_only
                  ? 1
                  : 0)
          << "\n";
  summary << "use_as_observation_requested: "
          << (report.baseline_result.effective_options
                      .geometry_prior_rescue_use_as_observation
                  ? 1
                  : 0)
          << "\n";
  summary << "allow_geometry_only_pose_refit: "
          << (report.baseline_result.effective_options
                      .geometry_prior_rescue_allow_geometry_only_pose_refit
                  ? 1
                  : 0)
          << "\n";
  summary << "geometry_only_updates_observations_by_default: "
          << (report.baseline_result.effective_options
                      .geometry_prior_rescue_allow_geometry_only_pose_refit &&
                  report.baseline_result.effective_options
                      .geometry_prior_rescue_use_as_observation &&
                  !report.baseline_result.effective_options
                       .geometry_prior_rescue_diagnostic_only
              ? 1
              : 0)
          << "\n";
  summary << "quad_topology_guard_enabled: 1\n";
  summary << "visible_frame_refit_mode: robust_multi_board_consensus\n";
  summary << "visible_frame_refit_outlier_policy: local_outer_rmse_lt_3px_then_board_median_mad_consensus\n";
  summary << "corner_displacement_guard_mode: diagnostic_only\n";
  summary << "normal_detected_outer_observation_count: "
          << normal_detected_outer_count << "\n";
  summary << "geometry_prior_seed_count: " << geometry_prior_seed_count
          << "\n";
  summary << "image_validated_rescued_outer_observation_count: "
          << image_validated_rescued_count << "\n";
  summary << "backend_used_rescued_outer_observation_count: "
          << backend_used_rescued_count << "\n";
  summary << "projected_geometry_prior_used_as_backend_observation: 0\n";
  summary << "note: geometry prior seeds define missing-board ROI. "
             "Tag/raw-quad image-validated rescue may be used when "
             "geometry_prior_rescue_use_as_observation=1 and "
             "diagnostic_only=0. The experimental geometry-only branch may "
             "continue to pose refit only when explicitly enabled and strong "
             "edge evidence is present; pure projected corners are never used "
             "as backend observations without image evidence and pose checks.\n";

  std::ofstream reject_summary(
      (output_dir / "geometry_prior_outer_seed_rejection_summary.txt")
          .string()
          .c_str());
  reject_summary << "geometry_prior_seed_count: " << geometry_prior_seed_count
                 << "\n";
  reject_summary << "image_validated_rescued_outer_observation_count: "
                 << image_validated_rescued_count << "\n";
  reject_summary << "backend_used_rescued_outer_observation_count: "
                 << backend_used_rescued_count << "\n";
  reject_summary << "accept_outer_rmse_threshold_px: "
                 << report.baseline_result.effective_options
                        .geometry_prior_rescue_accept_max_outer_rmse
                 << "\n";
  reject_summary << "adaptive_outer_rmse_rule: only after tag-id "
                    "validation, use max(configured_threshold, "
                    "frame_normal_outer_refit_rmse_median + 1.0)\n";
  reject_summary << "\nby_reject_reason:\n";
  reject_summary
      << "reason,count,accepted,rectified_tag_success,roi_tag_success,"
      << "image_evidence_success,pose_refit_success,outer_rmse_min,"
      << "outer_rmse_median,outer_rmse_max,corner_disp_median,"
      << "corner_response_median,edge_support_median\n";
  for (const auto& entry : seed_stats_by_reason) {
    const GeometryPriorSeedRejectStats& stats = entry.second;
    reject_summary << CsvEscape(entry.first) << ","
                   << stats.count << ","
                   << stats.accepted_count << ","
                   << stats.rectified_tag_success_count << ","
                   << stats.roi_tag_success_count << ","
                   << stats.image_evidence_success_count << ","
                   << stats.pose_refit_success_count << ","
                   << MinFiniteValue(stats.outer_rmse) << ","
                   << MedianValue(stats.outer_rmse) << ","
                   << MaxFiniteValue(stats.outer_rmse) << ","
                   << MedianValue(stats.max_corner_displacement) << ","
                   << MedianValue(stats.min_corner_response_ratio) << ","
                   << MedianValue(stats.edge_support_ratio) << "\n";
  }
  std::sort(rejected_candidates.begin(), rejected_candidates.end(),
            [](const ati::GeometryPriorOuterSeedCandidate& lhs,
               const ati::GeometryPriorOuterSeedCandidate& rhs) {
              return lhs.outer_reprojection_rmse < rhs.outer_reprojection_rmse;
            });
  reject_summary << "\nclosest_rejected_by_outer_rmse:\n";
  reject_summary
      << "frame_index,frame_label,board_id,reject_reason,outer_rmse,"
      << "frame_normal_outer_refit_rmse_median,adaptive_threshold,"
      << "rectified_tag_id,roi_tag_id,max_corner_displacement,"
      << "min_corner_response_ratio,edge_support_ratio\n";
  const std::size_t top_count =
      std::min<std::size_t>(20, rejected_candidates.size());
  for (std::size_t index = 0; index < top_count; ++index) {
    const ati::GeometryPriorOuterSeedCandidate& candidate =
        rejected_candidates[index];
    reject_summary << candidate.frame_index << ","
                   << CsvEscape(candidate.frame_label) << ","
                   << candidate.missing_board_id << ","
                   << CsvEscape(candidate.reject_reason) << ","
                   << candidate.outer_reprojection_rmse << ","
                   << candidate.frame_normal_outer_refit_rmse_median << ","
                   << candidate.adaptive_accept_max_outer_rmse << ","
                   << candidate.rectified_patch_detected_tag_id << ","
                   << candidate.roi_redetect_detected_tag_id << ","
                   << candidate.max_corner_displacement_px << ","
                   << candidate.min_corner_response_ratio << ","
                   << candidate.edge_support_ratio << "\n";
  }
}

void WriteIntermediateFrontendRegenerationSummary(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report) {
  const ati::FrozenRound2BaselineResult& baseline = report.baseline_result;
  const ati::OuterOnlyIntermediateCalibrationResult& intermediate =
      baseline.outer_only_intermediate;

  int round1_seed_count = 0;
  int round1_image_validated_rescue_count = 0;
  int round1_backend_used_rescue_count = 0;
  int round1_normal_outer_count = 0;
  int round1_attempted_internal_count = 0;
  int round1_valid_internal_count = 0;
  std::map<std::string, int> state_source_counts;
  std::map<std::string, int> seed_reject_reason_counts;
  for (const ati::InternalRegenerationFrameResult& frame :
       baseline.round1.regeneration_results) {
    ++state_source_counts[frame.state_source_label];
    round1_attempted_internal_count +=
        frame.runtime_breakdown.attempted_internal_corner_count;
    round1_valid_internal_count +=
        frame.runtime_breakdown.valid_internal_corner_count;
    for (const ati::GeometryPriorOuterSeedCandidate& candidate :
         frame.geometry_prior_outer_seed_candidates) {
      ++round1_seed_count;
      if (candidate.accepted_as_rescued_observation) {
        ++round1_image_validated_rescue_count;
      }
      ++seed_reject_reason_counts[candidate.reject_reason.empty()
                                      ? "empty_reject_reason"
                                      : candidate.reject_reason];
    }
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      if (!measurement.detection.outer_detection.success) {
        continue;
      }
      if (measurement.detection.outer_detection.used_local_patch_rescue) {
        ++round1_backend_used_rescue_count;
      } else {
        ++round1_normal_outer_count;
      }
    }
  }

  std::ofstream summary(
      (output_dir / "intermediate_frontend_regeneration_summary.txt")
          .string()
          .c_str());
  summary << "description: I3 intermediate model as geometry prior for full "
             "Round1 frontend regeneration\n";
  summary << "intermediate_enabled: " << (intermediate.enabled ? 1 : 0)
          << "\n";
  summary << "intermediate_success: " << (intermediate.success ? 1 : 0)
          << "\n";
  summary << "intermediate_diagnostic_only: "
          << (intermediate.diagnostic_only ? 1 : 0) << "\n";
  summary << "use_for_round1_requested: "
          << (intermediate.use_for_round1_requested ? 1 : 0) << "\n";
  summary << "use_for_full_frontend_regeneration_requested: "
          << (intermediate.use_for_full_frontend_regeneration_requested ? 1 : 0)
          << "\n";
  summary << "used_for_round1_internal_regeneration: "
          << (intermediate.used_for_round1_internal_regeneration ? 1 : 0)
          << "\n";
  summary << "used_for_full_frontend_regeneration: "
          << (intermediate.used_for_full_frontend_regeneration ? 1 : 0)
          << "\n";
  summary << "round1_geometry_source_when_enabled: "
          << (intermediate.used_for_full_frontend_regeneration
                  ? "outer_only_intermediate_optimized_state"
                  : "bootstrap_or_disabled")
          << "\n";
  summary << "missing_board_roi_uses_intermediate: "
          << (intermediate.used_for_full_frontend_regeneration ? 1 : 0)
          << "\n";
  summary << "ds_ray_patch_uses_intermediate: "
          << (intermediate.used_for_full_frontend_regeneration ? 1 : 0)
          << "\n";
  summary << "adaptive_subpix_uses_intermediate_prediction_scale: "
          << (intermediate.used_for_full_frontend_regeneration ? 1 : 0)
          << "\n";
  summary << "internal_regeneration_uses_intermediate: "
          << (intermediate.used_for_round1_internal_regeneration ? 1 : 0)
          << "\n";
  summary << "geometry_prior_seed_enabled: "
          << (baseline.effective_options.enable_geometry_prior_outer_seed ? 1 : 0)
          << "\n";
  summary << "geometry_prior_rescue_diagnostic_only: "
          << (baseline.effective_options.geometry_prior_rescue_diagnostic_only ? 1 : 0)
          << "\n";
  summary << "geometry_prior_rescue_use_as_observation: "
          << (baseline.effective_options.geometry_prior_rescue_use_as_observation ? 1 : 0)
          << "\n";
  summary << "geometry_prior_rescue_allow_geometry_only_pose_refit: "
          << (baseline.effective_options
                      .geometry_prior_rescue_allow_geometry_only_pose_refit
                  ? 1
                  : 0)
          << "\n";
  summary << "round1_normal_detected_outer_observation_count: "
          << round1_normal_outer_count << "\n";
  summary << "round1_geometry_prior_seed_count: " << round1_seed_count << "\n";
  summary << "round1_image_validated_rescued_outer_observation_count: "
          << round1_image_validated_rescue_count << "\n";
  summary << "round1_backend_used_rescued_outer_observation_count: "
          << round1_backend_used_rescue_count << "\n";
  summary << "round1_attempted_internal_corner_count: "
          << round1_attempted_internal_count << "\n";
  summary << "round1_valid_internal_corner_count: "
          << round1_valid_internal_count << "\n";
  summary << "round1_internal_valid_ratio: "
          << SafeRatio(round1_valid_internal_count,
                       round1_attempted_internal_count)
          << "\n";
  summary << "geometry_prior_candidates_csv: "
          << (output_dir / "geometry_prior_outer_seed_candidates.csv").string()
          << "\n";
  summary << "geometry_prior_round1_overlay_dir: "
          << (output_dir / "geometry_prior_outer_seed_overlays" /
              "round1_training").string()
          << "\n";
  summary << "frame_board_flow_csv: "
          << (output_dir / "frame_board_observation_flow.csv").string()
          << "\n";
  summary << "frame_board_flow_overlay_dir: "
          << (output_dir / "frame_board_observation_flow_overlays").string()
          << "\n";
  summary << "strict_internal_seed_overlay_dir: "
          << (output_dir / "strict_internal_seed_overlays").string()
          << "\n";
  summary << "forced_internal_seed_overlay_dir: "
          << (output_dir / "forced_internal_seed_overlays").string()
          << "\n";
  summary << "\nround1_state_source_counts:\n";
  for (const auto& entry : state_source_counts) {
    summary << "  " << entry.first << ": " << entry.second << "\n";
  }
  summary << "\nround1_geometry_prior_seed_reject_reason_counts:\n";
  for (const auto& entry : seed_reject_reason_counts) {
    summary << "  " << entry.first << ": " << entry.second << "\n";
  }
  summary << "\nnote: this mode changes the geometry prior used by Round1 "
             "frontend regeneration only when enabled explicitly. Pure "
             "projected geometry-prior corners are still not backend "
             "observations unless image validation passes and rescue use is "
             "explicitly enabled.\n";
}

struct FrameBoardObservationFlowRow {
  std::string round_label;
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  bool outer_detected = false;
  std::string outer_failure_reason;
  bool internal_result_available = false;
  bool internal_success = false;
  std::string internal_failure_reason;
  std::string internal_camera_source;
  bool frame_bootstrap_initialized = false;
  bool board_bootstrap_initialized = false;
  bool pose_prior_used = false;
  int attempted_internal_corner_count = 0;
  int valid_internal_corner_count = 0;
  double valid_internal_ratio = 0.0;
  bool measurement_built = false;
  bool pre_selection_solver_ready = false;
  int pre_selection_outer_point_count = 0;
  int pre_selection_internal_point_count = 0;
  int rejected_internal_regeneration_failed_count = 0;
  int rejected_internal_point_invalid_count = 0;
  int rejected_internal_point_outlier_count = 0;
  int rejected_other_count = 0;
  std::string point_rejection_reasons;
  bool strict_drop_candidate = false;
  bool selection_decision_available = false;
  bool selection_accepted = false;
  std::string selection_reason_code;
  std::string selection_reason_detail;
  double selection_rmse = 0.0;
  bool final_used_in_backend = false;
  int final_outer_point_count = 0;
  int final_internal_point_count = 0;
  std::string final_status;
};

struct BackendUsedFrameSummaryRow {
  int frame_index = -1;
  std::string frame_label;
  std::vector<int> board_ids;
  int board_observation_count = 0;
  int outer_point_count = 0;
  int internal_point_count = 0;
  int total_point_count = 0;
};

struct BackendTrainingResidualKey {
  int frame_index = -1;
  int board_id = -1;
  int point_id = -1;

  bool operator<(const BackendTrainingResidualKey& other) const {
    return std::tie(frame_index, board_id, point_id) <
           std::tie(other.frame_index, other.board_id, other.point_id);
  }
};

struct BackendTrainingResidualInfo {
  bool used_in_backend = false;
  double observed_x = 0.0;
  double observed_y = 0.0;
  double predicted_x = 0.0;
  double predicted_y = 0.0;
  double residual_x = 0.0;
  double residual_y = 0.0;
  double residual_norm = 0.0;
};

struct FrameBoardVisualGeometry {
  bool has_corners = false;
  std::array<cv::Point2f, 4> corners{};
};

const ati::RegeneratedBoardMeasurement* FindRegeneratedBoardMeasurement(
    const std::vector<ati::InternalRegenerationFrameResult>& frames,
    int frame_index,
    int board_id) {
  for (const ati::InternalRegenerationFrameResult& frame : frames) {
    if (frame.frame_index != frame_index) {
      continue;
    }
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      if (measurement.board_id == board_id) {
        return &measurement;
      }
    }
  }
  return nullptr;
}

const ati::JointBoardObservation* FindJointBoardObservation(
    const ati::JointMeasurementBuildResult& measurement_result,
    int frame_index,
    int board_id) {
  for (const ati::JointMeasurementFrameResult& frame :
       measurement_result.frames) {
    if (frame.frame_index != frame_index) {
      continue;
    }
    for (const ati::JointBoardObservation& board :
         frame.board_observations) {
      if (board.board_id == board_id) {
        return &board;
      }
    }
  }
  return nullptr;
}

const ati::JointBoardObservationSelectionDecision* FindSelectionDecision(
    const ati::JointMeasurementSelectionResult& selection_result,
    int frame_index,
    int board_id) {
  for (const ati::JointBoardObservationSelectionDecision& decision :
       selection_result.board_observation_decisions) {
    if (decision.frame_index == frame_index && decision.board_id == board_id) {
      return &decision;
    }
  }
  return nullptr;
}

std::map<std::pair<int, int>, std::pair<int, int> > BuildFinalBackendPointCounts(
    const ati::CalibrationBackendProblemInput& backend_problem_input) {
  std::map<std::pair<int, int>, std::pair<int, int> > counts;
  for (const ati::JointPointObservation& point :
       backend_problem_input.measurement_dataset.solver_observations) {
    if (!point.used_in_solver) {
      continue;
    }
    std::pair<int, int>& count =
        counts[std::make_pair(point.frame_index, point.board_id)];
    if (point.point_type == ati::JointPointType::Outer) {
      ++count.first;
    } else {
      ++count.second;
    }
  }
  return counts;
}

std::map<BackendTrainingResidualKey, BackendTrainingResidualInfo>
BuildBackendTrainingResidualMap(
    const ati::CameraModelRefitEvaluationResult* backend_training_evaluation) {
  std::map<BackendTrainingResidualKey, BackendTrainingResidualInfo> residuals;
  if (backend_training_evaluation == nullptr) {
    return residuals;
  }
  for (const ati::CameraModelRefitPointDiagnostics& point :
       backend_training_evaluation->point_diagnostics) {
    if (point.point_type != ati::JointPointType::Internal) {
      continue;
    }
    BackendTrainingResidualKey key;
    key.frame_index = point.frame_index;
    key.board_id = point.board_id;
    key.point_id = point.point_id;
    BackendTrainingResidualInfo info;
    info.used_in_backend = true;
    info.observed_x = point.observed_image_xy.x();
    info.observed_y = point.observed_image_xy.y();
    info.predicted_x = point.predicted_image_xy.x();
    info.predicted_y = point.predicted_image_xy.y();
    info.residual_x = point.residual_xy.x();
    info.residual_y = point.residual_xy.y();
    info.residual_norm = point.residual_norm;
    residuals[key] = info;
  }
  return residuals;
}

std::map<BackendTrainingResidualKey, BackendTrainingResidualInfo>
BuildBackendOptimizedResidualMap(
    const ati::JointResidualEvaluationResult& backend_residual) {
  std::map<BackendTrainingResidualKey, BackendTrainingResidualInfo> residuals;
  if (!backend_residual.success) {
    return residuals;
  }
  for (const ati::JointResidualPointDiagnostics& point :
       backend_residual.point_diagnostics) {
    if (!point.used_in_solver) {
      continue;
    }
    BackendTrainingResidualKey key;
    key.frame_index = point.frame_index;
    key.board_id = point.board_id;
    key.point_id = point.point_id;
    BackendTrainingResidualInfo info;
    info.used_in_backend = true;
    info.observed_x = point.observed_image_xy.x();
    info.observed_y = point.observed_image_xy.y();
    info.predicted_x = point.predicted_image_xy.x();
    info.predicted_y = point.predicted_image_xy.y();
    info.residual_x = point.residual_xy.x();
    info.residual_y = point.residual_xy.y();
    info.residual_norm = point.residual_norm;
    residuals[key] = info;
  }
  return residuals;
}

std::string JoinReasonCounts(const std::map<std::string, int>& counts) {
  std::ostringstream stream;
  bool first = true;
  for (const auto& entry : counts) {
    if (!first) {
      stream << ";";
    }
    first = false;
    stream << entry.first << "=" << entry.second;
  }
  return stream.str();
}

double MeanFiniteValue(const std::vector<double>& values) {
  double sum = 0.0;
  int count = 0;
  for (double value : values) {
    if (!std::isfinite(value)) {
      continue;
    }
    sum += value;
    ++count;
  }
  return count > 0 ? sum / static_cast<double>(count)
                   : std::numeric_limits<double>::quiet_NaN();
}

std::string DeriveFrameBoardFlowStatus(
    const FrameBoardObservationFlowRow& row) {
  if (row.final_used_in_backend) {
    if (row.internal_success && row.valid_internal_corner_count > 0 &&
        row.final_internal_point_count < row.valid_internal_corner_count) {
      return "used_partial_internal_points";
    }
    return "used_in_backend";
  }
  if (!row.outer_detected) {
    return "outer_not_detected";
  }
  if (row.strict_drop_candidate) {
    return "dropped_by_strict_internal_failure";
  }
  if (row.rejected_internal_point_outlier_count > 0 &&
      row.pre_selection_outer_point_count == 0 &&
      row.pre_selection_internal_point_count == 0) {
    return "all_points_rejected_by_internal_outlier_filter";
  }
  if (row.measurement_built && !row.pre_selection_solver_ready) {
    return "measurement_not_solver_ready";
  }
  if (row.selection_decision_available && !row.selection_accepted) {
    return std::string("rejected_by_selection_") + row.selection_reason_code;
  }
  return "not_used_unknown";
}

std::string StatusShortLabel(const std::string& status) {
  if (status == "used_in_backend") {
    return "USED";
  }
  if (status == "used_partial_internal_points") {
    return "PARTIAL";
  }
  if (status == "dropped_by_strict_internal_failure") {
    return "STRICT";
  }
  if (status == "outer_not_detected") {
    return "NO_OUTER";
  }
  if (status.find("selection") != std::string::npos) {
    return "SELECT";
  }
  if (status == "measurement_not_solver_ready") {
    return "NOT_READY";
  }
  if (status == "all_points_rejected_by_internal_outlier_filter") {
    return "OUTLIER";
  }
  return "OTHER";
}

cv::Scalar StatusColor(const std::string& status) {
  if (status == "used_in_backend") {
    return cv::Scalar(40, 190, 40);
  }
  if (status == "used_partial_internal_points") {
    return cv::Scalar(0, 210, 210);
  }
  if (status == "dropped_by_strict_internal_failure") {
    return cv::Scalar(40, 40, 230);
  }
  if (status == "outer_not_detected") {
    return cv::Scalar(160, 160, 160);
  }
  if (status.find("selection") != std::string::npos) {
    return cv::Scalar(0, 165, 255);
  }
  if (status == "all_points_rejected_by_internal_outlier_filter") {
    return cv::Scalar(220, 0, 220);
  }
  return cv::Scalar(255, 120, 0);
}

FrameBoardVisualGeometry BuildVisualGeometry(
    const ati::RegeneratedBoardMeasurement* regenerated,
    const ati::JointBoardObservation* board_observation) {
  FrameBoardVisualGeometry geometry;
  if (regenerated != nullptr && regenerated->detection.outer_detection.success) {
    geometry.has_corners = true;
    geometry.corners = regenerated->detection.outer_corners;
    return geometry;
  }
  if (board_observation == nullptr) {
    return geometry;
  }
  std::vector<cv::Point2f> outer_points;
  outer_points.reserve(4);
  for (const ati::JointPointObservation& point : board_observation->points) {
    if (point.point_type != ati::JointPointType::Outer) {
      continue;
    }
    outer_points.push_back(cv::Point2f(
        static_cast<float>(point.image_xy.x()),
        static_cast<float>(point.image_xy.y())));
  }
  if (outer_points.size() >= 4) {
    geometry.has_corners = true;
    for (int index = 0; index < 4; ++index) {
      geometry.corners[static_cast<std::size_t>(index)] =
          outer_points[static_cast<std::size_t>(index)];
    }
  }
  return geometry;
}

void DrawFrameBoardObservationFlowOverlay(
    const cv::Mat& image,
    const std::vector<FrameBoardObservationFlowRow>& rows,
    const std::map<std::pair<int, int>, FrameBoardVisualGeometry>& geometries,
    cv::Mat* output) {
  if (output == nullptr) {
    return;
  }
  if (image.empty()) {
    *output = cv::Mat();
    return;
  }
  if (image.channels() == 1) {
    cv::cvtColor(image, *output, cv::COLOR_GRAY2BGR);
  } else {
    *output = image.clone();
  }

  const int line_height = 22;
  cv::rectangle(*output, cv::Point(8, 8), cv::Point(620, 8 + 5 * line_height),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(*output, "Frame-board observation flow",
              cv::Point(16, 28), cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "green=used  cyan=used but partial internal  red=strict internal failure",
              cv::Point(16, 50), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "orange=selection rejected  magenta=internal outlier  gray=no outer",
              cv::Point(16, 70), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

  int text_y = 92;
  for (const FrameBoardObservationFlowRow& row : rows) {
    const cv::Scalar color = StatusColor(row.final_status);
    const std::pair<int, int> key(row.frame_index, row.board_id);
    const auto geometry_it = geometries.find(key);
    bool drew_on_board = false;
    if (geometry_it != geometries.end() && geometry_it->second.has_corners) {
      const std::array<cv::Point2f, 4>& corners = geometry_it->second.corners;
      std::vector<cv::Point> polygon;
      polygon.reserve(4);
      cv::Point2f center(0.0f, 0.0f);
      for (const cv::Point2f& corner : corners) {
        polygon.push_back(cv::Point(static_cast<int>(std::round(corner.x)),
                                    static_cast<int>(std::round(corner.y))));
        center += corner;
      }
      center *= 0.25f;
      cv::polylines(*output, polygon, true, color, 3, cv::LINE_AA);
      std::ostringstream label;
      label << "B" << row.board_id << " " << StatusShortLabel(row.final_status)
            << " I" << row.final_internal_point_count << "/"
            << row.valid_internal_corner_count;
      cv::putText(*output, label.str(),
                  cv::Point(static_cast<int>(std::round(center.x)),
                            static_cast<int>(std::round(center.y))),
                  cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
      drew_on_board = true;
    }
    if (!drew_on_board && text_y < output->rows - 12) {
      std::ostringstream label;
      label << "B" << row.board_id << " " << StatusShortLabel(row.final_status)
            << " outer=" << (row.outer_detected ? 1 : 0)
            << " internal=" << (row.internal_success ? 1 : 0);
      cv::putText(*output, label.str(), cv::Point(16, text_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
      text_y += line_height;
    }
  }
}

bool IsFinitePoint(const cv::Point2f& point) {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

bool IsPointNearImage(const cv::Point2f& point, const cv::Size& size,
                      float margin) {
  return IsFinitePoint(point) && point.x >= -margin && point.y >= -margin &&
         point.x < static_cast<float>(size.width) + margin &&
         point.y < static_cast<float>(size.height) + margin;
}

void DrawStrictInternalSeedOverlay(
    const cv::Mat& image,
    const std::vector<FrameBoardObservationFlowRow>& strict_rows,
    const std::vector<ati::InternalRegenerationFrameResult>& regeneration_results,
    cv::Mat* output) {
  if (output == nullptr) {
    return;
  }
  if (image.empty()) {
    *output = cv::Mat();
    return;
  }
  if (image.channels() == 1) {
    cv::cvtColor(image, *output, cv::COLOR_GRAY2BGR);
  } else {
    *output = image.clone();
  }

  cv::rectangle(*output, cv::Point(8, 8), cv::Point(780, 118),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(*output, "Strict-dropped internal seed overlay",
              cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX, 0.58,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "orange cross=predicted  cyan circle=sphere seed  GREEN/RED filled=final refined corner",
              cv::Point(16, 54), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "yellow polygon=detected outer; only boards dropped by strict internal failure are drawn",
              cv::Point(16, 76), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

  int board_text_y = 102;
  for (const FrameBoardObservationFlowRow& row : strict_rows) {
    const ati::RegeneratedBoardMeasurement* regenerated =
        FindRegeneratedBoardMeasurement(regeneration_results, row.frame_index,
                                        row.board_id);
    if (regenerated == nullptr) {
      continue;
    }
    const ati::ApriltagInternalDetectionResult& detection =
        regenerated->detection;
    if (detection.outer_detection.success) {
      std::vector<cv::Point> polygon;
      polygon.reserve(4);
      cv::Point2f center(0.0f, 0.0f);
      for (const cv::Point2f& corner : detection.outer_corners) {
        polygon.emplace_back(static_cast<int>(std::round(corner.x)),
                             static_cast<int>(std::round(corner.y)));
        center += corner;
      }
      center *= 0.25f;
      cv::polylines(*output, polygon, true, cv::Scalar(0, 220, 255), 3,
                    cv::LINE_AA);
      std::ostringstream board_label;
      board_label << "B" << row.board_id << " strict "
                  << row.valid_internal_corner_count << "/"
                  << row.attempted_internal_corner_count;
      cv::putText(*output, board_label.str(),
                  cv::Point(static_cast<int>(std::round(center.x)),
                            static_cast<int>(std::round(center.y))),
                  cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0, 220, 255),
                  2, cv::LINE_AA);
    }

    int drawn_count = 0;
    int valid_count = 0;
    int invalid_count = 0;
    const float margin = 80.0f;
    for (const ati::InternalCornerDebugInfo& debug :
         detection.internal_corner_debug) {
      const bool visible =
          IsPointNearImage(debug.predicted_image, output->size(), margin) ||
          IsPointNearImage(debug.sphere_seed_image, output->size(), margin) ||
          IsPointNearImage(debug.refined_image, output->size(), margin);
      if (!visible) {
        continue;
      }
      cv::drawMarker(*output, debug.predicted_image, cv::Scalar(0, 165, 255),
                     cv::MARKER_CROSS, 13, 1, cv::LINE_AA);
      if (IsPointNearImage(debug.sphere_seed_image, output->size(), margin)) {
        cv::circle(*output, debug.sphere_seed_image, 5,
                   cv::Scalar(255, 220, 0), 1, cv::LINE_AA);
      }
      if (IsPointNearImage(debug.refined_image, output->size(), margin)) {
        const cv::Scalar refined_color =
            debug.valid ? cv::Scalar(60, 230, 60) : cv::Scalar(0, 0, 255);
        cv::circle(*output, debug.refined_image, 7, refined_color,
                   cv::FILLED, cv::LINE_AA);
        cv::circle(*output, debug.refined_image, 9, cv::Scalar(255, 255, 255),
                   1, cv::LINE_AA);
      }
      if (IsPointNearImage(debug.sphere_seed_image, output->size(), margin) &&
          IsPointNearImage(debug.refined_image, output->size(), margin)) {
        cv::line(*output, debug.sphere_seed_image, debug.refined_image,
                 cv::Scalar(160, 160, 255), 1, cv::LINE_AA);
      }
      if (debug.valid) {
        ++valid_count;
      } else {
        ++invalid_count;
      }
      ++drawn_count;
    }
    if (board_text_y < output->rows - 12) {
      std::ostringstream label;
      label << "B" << row.board_id << ": debug_points=" << drawn_count
            << " valid=" << valid_count << " invalid=" << invalid_count
            << " status=" << row.final_status;
      cv::putText(*output, label.str(), cv::Point(16, board_text_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.42,
                  cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      board_text_y += 20;
    }
  }
}

void DrawForcedInternalSeedOverlay(
    const cv::Mat& image,
    int frame_index,
    const std::vector<const ati::RegeneratedBoardMeasurement*>& measurements,
    const std::map<BackendTrainingResidualKey, BackendTrainingResidualInfo>& residuals,
    cv::Mat* output) {
  if (output == nullptr) {
    return;
  }
  if (image.empty()) {
    *output = cv::Mat();
    return;
  }
  if (image.channels() == 1) {
    cv::cvtColor(image, *output, cv::COLOR_GRAY2BGR);
  } else {
    *output = image.clone();
  }
  cv::rectangle(*output, cv::Point(8, 8), cv::Point(850, 102),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(*output, "Forced predicted-seed internal corners",
              cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX, 0.58,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "orange cross=predicted seed  magenta=backend prediction  label q=image quality",
              cv::Point(16, 54), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "green=backend used/res<=3  yellow=backend used/res>3  blue=valid not selected",
              cv::Point(16, 76), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "red=invalid  cyan ring=original seed filter success  yellow X=original seed reject",
              cv::Point(16, 98), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

  int text_y = 122;
  int total_forced = 0;
  int total_used = 0;
  int total_original_seed_rejected = 0;
  for (const ati::RegeneratedBoardMeasurement* measurement : measurements) {
    if (measurement == nullptr) {
      continue;
    }
    const ati::ApriltagInternalDetectionResult& detection = measurement->detection;
    if (detection.outer_detection.success) {
      std::vector<cv::Point> polygon;
      polygon.reserve(4);
      for (const cv::Point2f& corner : detection.outer_corners) {
        polygon.emplace_back(static_cast<int>(std::round(corner.x)),
                             static_cast<int>(std::round(corner.y)));
      }
      cv::polylines(*output, polygon, true, cv::Scalar(0, 220, 255), 2,
                    cv::LINE_AA);
    }
    int board_forced = 0;
    int board_used = 0;
    for (const ati::InternalCornerDebugInfo& debug :
         detection.internal_corner_debug) {
      if (!debug.forced_prediction_seed) {
        continue;
      }
      ++total_forced;
      ++board_forced;
      if (debug.original_seed_filter_would_reject) {
        ++total_original_seed_rejected;
      }
      BackendTrainingResidualKey key;
      key.frame_index = frame_index;
      key.board_id = measurement->board_id;
      key.point_id = debug.point_id;
      const auto residual_it = residuals.find(key);
      const bool used = residual_it != residuals.end();
      if (used) {
        ++total_used;
        ++board_used;
      }
      const double residual =
          used ? residual_it->second.residual_norm
               : std::numeric_limits<double>::infinity();
      cv::Scalar refined_color(255, 90, 0);
      std::string status_label = "valid_not_selected";
      if (!debug.valid) {
        refined_color = cv::Scalar(0, 0, 255);
        status_label = "invalid";
      } else if (used && residual <= 3.0) {
        refined_color = cv::Scalar(60, 230, 60);
        status_label = "used";
      } else if (used) {
        refined_color = cv::Scalar(0, 220, 255);
        status_label = "used_hi_res";
      }
      cv::drawMarker(*output, debug.predicted_image, cv::Scalar(0, 165, 255),
                     cv::MARKER_CROSS, 13, 1, cv::LINE_AA);
      if (debug.original_seed_filter_would_reject) {
        cv::drawMarker(*output, debug.refined_image, cv::Scalar(0, 255, 255),
                       cv::MARKER_TILTED_CROSS, 17, 3, cv::LINE_AA);
      } else if (debug.original_seed_filter_success &&
                 IsPointNearImage(debug.original_seed_filter_image,
                                  output->size(), 80.0f)) {
        cv::circle(*output, debug.original_seed_filter_image, 5,
                   cv::Scalar(255, 220, 0), 1, cv::LINE_AA);
      }
      if (IsPointNearImage(debug.refined_image, output->size(), 80.0f)) {
        cv::circle(*output, debug.refined_image, 7, refined_color,
                   cv::FILLED, cv::LINE_AA);
        cv::circle(*output, debug.refined_image, 9, cv::Scalar(255, 255, 255),
                   1, cv::LINE_AA);
        cv::line(*output, debug.predicted_image, debug.refined_image,
                 cv::Scalar(160, 160, 255), 1, cv::LINE_AA);
      }
      if (used) {
        const cv::Point2f predicted(
            static_cast<float>(residual_it->second.predicted_x),
            static_cast<float>(residual_it->second.predicted_y));
        cv::drawMarker(*output, predicted, cv::Scalar(255, 0, 255),
                       cv::MARKER_TILTED_CROSS, 11, 1, cv::LINE_AA);
        cv::line(*output, debug.refined_image, predicted,
                 cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
      }
      if (IsPointNearImage(debug.refined_image, output->size(), 80.0f)) {
        std::ostringstream label;
        label << "B" << measurement->board_id << " p" << debug.point_id;
        label << " " << status_label;
        label << " q=" << std::fixed << std::setprecision(2)
              << debug.image_final_quality;
        if (used) {
          label << " r=" << std::setprecision(1) << residual;
        } else {
          label << " not_backend";
        }
        if (debug.original_seed_filter_would_reject) {
          label << " seed_reject";
        }
        cv::putText(*output, label.str(),
                    cv::Point(static_cast<int>(debug.refined_image.x) + 8,
                              static_cast<int>(debug.refined_image.y) - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.36, cv::Scalar(255, 255, 255),
                    1, cv::LINE_AA);
      }
    }
    if (board_forced > 0 && text_y < output->rows - 12) {
      std::ostringstream label;
      label << "B" << measurement->board_id << " forced=" << board_forced
            << " backend_used=" << board_used;
      cv::putText(*output, label.str(), cv::Point(16, text_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.42,
                  cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      text_y += 20;
    }
  }
  std::ostringstream footer;
  footer << "forced_total=" << total_forced << " backend_used=" << total_used
         << " original_seed_rejected=" << total_original_seed_rejected;
  cv::putText(*output, footer.str(), cv::Point(16, output->rows - 18),
              cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255, 255, 255),
              1, cv::LINE_AA);
}

void DrawRescuedInternalPointOverlay(
    const cv::Mat& image,
    int frame_index,
    const std::vector<const ati::RegeneratedBoardMeasurement*>& measurements,
    const std::map<BackendTrainingResidualKey, BackendTrainingResidualInfo>& residuals,
    cv::Mat* output) {
  if (output == nullptr) {
    return;
  }
  if (image.empty()) {
    *output = cv::Mat();
    return;
  }
  if (image.channels() == 1) {
    cv::cvtColor(image, *output, cv::COLOR_GRAY2BGR);
  } else {
    *output = image.clone();
  }

  cv::rectangle(*output, cv::Point(8, 8), cv::Point(980, 108),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(*output, "Rescued internal-point overlay",
              cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX, 0.58,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "Style follows detect_apriltag_internal internal generation visualization",
              cv::Point(16, 54), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "orange=predicted  blue triangle=border seed  magenta diamond=sphere seed  green/red=refined",
              cv::Point(16, 76), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "yellow polygon=rescued outer board  magenta tilted cross=backend prediction when available",
              cv::Point(16, 98), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

  int text_y = 122;
  int total_rescued = 0;
  int total_used = 0;
  for (const ati::RegeneratedBoardMeasurement* measurement : measurements) {
    if (measurement == nullptr) {
      continue;
    }
    const ati::ApriltagInternalDetectionResult& detection = measurement->detection;
    if (!detection.outer_detection.success ||
        !detection.outer_detection.used_local_patch_rescue) {
      continue;
    }

    std::vector<cv::Point> polygon;
    polygon.reserve(4);
    for (const cv::Point2f& corner : detection.outer_corners) {
      polygon.emplace_back(static_cast<int>(std::round(corner.x)),
                           static_cast<int>(std::round(corner.y)));
    }
    if (polygon.size() == 4) {
      cv::polylines(*output, polygon, true, cv::Scalar(0, 220, 255), 3,
                    cv::LINE_AA);
    }

    int board_rescued = 0;
    int board_used = 0;
    for (const ati::InternalCornerDebugInfo& debug :
         detection.internal_corner_debug) {
      ++total_rescued;
      ++board_rescued;
      BackendTrainingResidualKey key;
      key.frame_index = frame_index;
      key.board_id = measurement->board_id;
      key.point_id = debug.point_id;
      const auto residual_it = residuals.find(key);
      const bool used = residual_it != residuals.end();
      if (used) {
        ++total_used;
        ++board_used;
      }

      const bool valid = debug.valid;
      const cv::Scalar refined_color =
          valid ? cv::Scalar(60, 230, 60) : cv::Scalar(0, 0, 255);
      cv::drawMarker(*output, debug.predicted_image, cv::Scalar(0, 165, 255),
                     cv::MARKER_CROSS, 13, 1, cv::LINE_AA);
      if (debug.border_seed_valid &&
          IsPointNearImage(debug.border_seed_image, output->size(), 80.0f)) {
        cv::drawMarker(*output, debug.border_seed_image,
                       cv::Scalar(255, 180, 0), cv::MARKER_TRIANGLE_UP, 13,
                       1, cv::LINE_AA);
        cv::line(*output, debug.predicted_image, debug.border_seed_image,
                 cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
      }
      if (IsPointNearImage(debug.sphere_seed_image, output->size(), 80.0f)) {
        cv::drawMarker(*output, debug.sphere_seed_image,
                       cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 13, 1,
                       cv::LINE_AA);
      }
      if (IsPointNearImage(debug.refined_image, output->size(), 80.0f)) {
        cv::circle(*output, debug.refined_image, 7, refined_color,
                   cv::FILLED, cv::LINE_AA);
        cv::circle(*output, debug.refined_image, 9, cv::Scalar(255, 255, 255),
                   1, cv::LINE_AA);
      }
      if (IsPointNearImage(debug.sphere_seed_image, output->size(), 80.0f) &&
          IsPointNearImage(debug.refined_image, output->size(), 80.0f)) {
        cv::line(*output, debug.sphere_seed_image, debug.refined_image,
                 cv::Scalar(160, 160, 255), 1, cv::LINE_AA);
      }
      if (used) {
        const cv::Point2f predicted(
            static_cast<float>(residual_it->second.predicted_x),
            static_cast<float>(residual_it->second.predicted_y));
        cv::drawMarker(*output, predicted, cv::Scalar(255, 0, 255),
                       cv::MARKER_TILTED_CROSS, 11, 1, cv::LINE_AA);
        cv::line(*output, debug.refined_image, predicted,
                 cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
      }
      if (IsPointNearImage(debug.refined_image, output->size(), 80.0f)) {
        std::ostringstream label;
        label << "B" << measurement->board_id << " p" << debug.point_id;
        label << (valid ? " valid" : " invalid");
        if (used) {
          label << " r=" << std::fixed << std::setprecision(1)
                << residual_it->second.residual_norm;
        }
        cv::putText(*output, label.str(),
                    cv::Point(static_cast<int>(debug.refined_image.x) + 8,
                              static_cast<int>(debug.refined_image.y) - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.36, cv::Scalar(255, 255, 255),
                    1, cv::LINE_AA);
      }
    }

    if (board_rescued > 0 && text_y < output->rows - 12) {
      std::ostringstream label;
      label << "B" << measurement->board_id << " rescued=" << board_rescued
            << " backend_used=" << board_used;
      cv::putText(*output, label.str(), cv::Point(16, text_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.42,
                  cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      text_y += 20;
    }
  }

  std::ostringstream footer;
  footer << "rescued_total=" << total_rescued
         << " backend_used=" << total_used;
  cv::putText(*output, footer.str(), cv::Point(16, output->rows - 18),
              cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255, 255, 255),
              1, cv::LINE_AA);
}

cv::Scalar InternalPointFilterColor(
    const ati::JointPointObservation& point) {
  if (point.used_in_solver) {
    return cv::Scalar(60, 230, 60);
  }
  if (point.rejection_reason_code ==
      ati::JointRejectionReasonCode::InternalPointInvalid) {
    return cv::Scalar(0, 0, 255);
  }
  if (point.rejection_reason_code ==
      ati::JointRejectionReasonCode::InternalPointReprojectionOutlier) {
    return cv::Scalar(0, 165, 255);
  }
  if (point.rejection_reason_code ==
      ati::JointRejectionReasonCode::InternalRegenerationFailed) {
    return cv::Scalar(220, 0, 220);
  }
  // A frame pose can be unavailable even when an internal corner was
  // successfully regenerated. Keep that state visibly distinct from a
  // frontend-invalid corner, otherwise this diagnostic suggests a detector
  // failure where the bootstrap scene is actually the limiting stage.
  if (point.rejection_reason_code ==
      ati::JointRejectionReasonCode::FrameNotInitialized) {
    return cv::Scalar(255, 180, 40);
  }
  // The point can be frontend-valid yet excluded by a board/scene-level
  // condition (for example reference connectivity or a missing board pose).
  // Render that state in blue instead of the ambiguous gray fallback so a
  // valid detected corner is not mistaken for a frontend-invalid result.
  if (point.rejection_reason_code !=
      ati::JointRejectionReasonCode::InternalPointInvalid) {
    return cv::Scalar(255, 120, 40);
  }
  return cv::Scalar(170, 170, 170);
}

struct BackendInputBoardOverlaySource {
  bool has_decision = false;
  bool baseline_seed = false;
  bool incremental_trial_accepted = false;
  bool frame_cohesion_accepted = false;
  bool close_distance_frame_admission_accepted = false;
  bool soft_accepted = false;
  std::string reason;
};

bool IsIncrementalTrialAcceptedReason(const std::string& reason) {
  return reason == "accepted_incremental_trial" ||
         reason == "accepted_close_distance_frame_admission_trial";
}

std::string BackendInputBoardSourceLabel(
    const BackendInputBoardOverlaySource& source) {
  if (source.baseline_seed) {
    return "seed";
  }
  if (source.frame_cohesion_accepted) {
    return "cohesion";
  }
  if (source.close_distance_frame_admission_accepted) {
    return "near";
  }
  if (source.soft_accepted) {
    return "soft";
  }
  if (source.incremental_trial_accepted) {
    return "trial";
  }
  return source.has_decision ? "decision" : "unknown";
}

cv::Scalar BackendInputBoardSourceColor(
    const BackendInputBoardOverlaySource& source) {
  if (source.baseline_seed) {
    return cv::Scalar(60, 230, 60);
  }
  if (source.frame_cohesion_accepted) {
    return cv::Scalar(255, 0, 255);
  }
  if (source.close_distance_frame_admission_accepted) {
    return cv::Scalar(255, 210, 40);
  }
  if (source.soft_accepted) {
    return cv::Scalar(80, 220, 255);
  }
  if (source.incremental_trial_accepted) {
    return cv::Scalar(0, 165, 255);
  }
  return cv::Scalar(180, 180, 180);
}

void DrawLegendSwatch(cv::Mat* output,
                      const cv::Point& origin,
                      const cv::Scalar& color,
                      const std::string& label) {
  if (output == nullptr || output->empty()) {
    return;
  }
  cv::rectangle(*output, cv::Rect(origin.x, origin.y - 11, 16, 12), color,
                cv::FILLED);
  cv::putText(*output, label, cv::Point(origin.x + 22, origin.y),
              cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(255, 255, 255), 1,
              cv::LINE_AA);
}

void DrawInternalPointFilterOverlay(
    const cv::Mat& image,
    const std::vector<const ati::JointBoardObservation*>& board_observations,
    const std::map<std::pair<int, int>, const ati::RegeneratedBoardMeasurement*>&
        regenerated_by_frame_board,
    cv::Mat* output) {
  if (output == nullptr) {
    return;
  }
  if (image.empty()) {
    *output = cv::Mat();
    return;
  }
  if (image.channels() == 1) {
    cv::cvtColor(image, *output, cv::COLOR_GRAY2BGR);
  } else {
    *output = image.clone();
  }

  cv::rectangle(*output, cv::Point(8, 8), cv::Point(940, 118),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(*output, "Internal point filter overlay",
              cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX, 0.58,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "green=solver-ready  red=frontend-invalid  orange=local geometry outlier",
              cv::Point(16, 54), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "blue=frontend-valid, frame pose unavailable  magenta=regen failed  gray=other rejected",
              cv::Point(16, 76), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "cyan cross=outer corners; cyan boxes=outer subpix window; boost=1 means enlarged window.",
              cv::Point(16, 98), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

  int text_y = 122;
  for (const ati::JointBoardObservation* board : board_observations) {
    if (board == nullptr) {
      continue;
    }
    int outer_count = 0;
    int used_internal = 0;
    int invalid_internal = 0;
    int outlier_internal = 0;
    int regen_failed_internal = 0;
    int frame_pose_unavailable_internal = 0;
    int other_rejected_internal = 0;
    cv::Point2f board_center(0.0f, 0.0f);
    std::vector<cv::Point> outer_polygon;
    outer_polygon.reserve(4);

    for (const ati::JointPointObservation& point : board->points) {
      const cv::Point2f xy(static_cast<float>(point.image_xy.x()),
                           static_cast<float>(point.image_xy.y()));
      if (!IsPointNearImage(xy, output->size(), 80.0f)) {
        continue;
      }
      if (point.point_type == ati::JointPointType::Outer) {
        ++outer_count;
        board_center += xy;
        outer_polygon.emplace_back(static_cast<int>(std::round(xy.x)),
                                   static_cast<int>(std::round(xy.y)));
        cv::drawMarker(*output, xy, cv::Scalar(255, 220, 0),
                       cv::MARKER_CROSS, 13, 1, cv::LINE_AA);
        continue;
      }

      if (point.used_in_solver) {
        ++used_internal;
      } else if (point.rejection_reason_code ==
                 ati::JointRejectionReasonCode::InternalPointInvalid) {
        ++invalid_internal;
      } else if (point.rejection_reason_code ==
                 ati::JointRejectionReasonCode::InternalPointReprojectionOutlier) {
        ++outlier_internal;
      } else if (point.rejection_reason_code ==
                 ati::JointRejectionReasonCode::InternalRegenerationFailed) {
        ++regen_failed_internal;
      } else if (point.rejection_reason_code ==
                 ati::JointRejectionReasonCode::FrameNotInitialized) {
        ++frame_pose_unavailable_internal;
      } else {
        ++other_rejected_internal;
      }

      const cv::Scalar color = InternalPointFilterColor(point);
      if (point.used_in_solver) {
        cv::circle(*output, xy, 5, color, cv::FILLED, cv::LINE_AA);
        cv::circle(*output, xy, 7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      } else {
        cv::drawMarker(*output, xy, color, cv::MARKER_TILTED_CROSS, 11, 2,
                       cv::LINE_AA);
      }
      if (IsPointNearImage(xy, output->size(), 80.0f)) {
        std::ostringstream q_label;
        q_label << "q=" << std::fixed << std::setprecision(2)
                << point.quality;
        if (!point.used_in_solver &&
            point.rejection_reason_code ==
                ati::JointRejectionReasonCode::InternalPointReprojectionOutlier) {
          const std::string& detail = point.rejection_detail;
          const std::string key = "reprojection_error=";
          const std::size_t pos = detail.find(key);
          if (pos != std::string::npos) {
            std::istringstream value_stream(
                detail.substr(pos + key.size()));
            double residual = 0.0;
            if (value_stream >> residual) {
              q_label << " r=" << std::setprecision(1) << residual;
            }
          }
        }
        cv::putText(*output, q_label.str(),
                    cv::Point(static_cast<int>(xy.x) + 6,
                              static_cast<int>(xy.y) - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.32,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      }
    }

    if (outer_polygon.size() >= 4) {
      cv::polylines(*output, outer_polygon, true, cv::Scalar(255, 220, 0), 2,
                    cv::LINE_AA);
      board_center *= 1.0f / static_cast<float>(outer_count);
      const ati::RegeneratedBoardMeasurement* regenerated = nullptr;
      if (!board->points.empty()) {
        const int frame_index = board->points.front().frame_index;
        const auto regenerated_it =
            regenerated_by_frame_board.find(std::make_pair(frame_index, board->board_id));
        if (regenerated_it != regenerated_by_frame_board.end()) {
          regenerated = regenerated_it->second;
        }
      }
      bool outer_subpix_boost = false;
      double outer_subpix_area_ratio = 0.0;
      double outer_subpix_max_polar_deg = 0.0;
      int max_outer_subpix_radius = 0;
      int max_raw_outer_subpix_radius = 0;
      int max_pre_boost_outer_subpix_radius = 0;
      int min_outer_subpix_clamp_limit = 0;
      bool outer_subpix_clamped = false;
      double outer_subpix_config_scale = 0.0;
      if (regenerated != nullptr && regenerated->detection.outer_detection.success) {
        for (const ati::OuterCornerVerificationDebugInfo& debug :
             regenerated->detection.outer_detection.corner_verification_debug) {
          outer_subpix_boost =
              outer_subpix_boost || debug.close_edge_subpix_boost_applied;
          outer_subpix_area_ratio =
              std::max(outer_subpix_area_ratio, debug.close_edge_subpix_area_ratio);
          outer_subpix_max_polar_deg =
              std::max(outer_subpix_max_polar_deg,
                       debug.close_edge_subpix_max_polar_deg);
          max_outer_subpix_radius =
              std::max(max_outer_subpix_radius, debug.subpix_window_radius);
          max_raw_outer_subpix_radius =
              std::max(max_raw_outer_subpix_radius,
                       debug.raw_subpix_window_radius);
          max_pre_boost_outer_subpix_radius =
              std::max(max_pre_boost_outer_subpix_radius,
                       debug.pre_boost_subpix_window_radius);
          if (debug.subpix_window_clamp_limit > 0) {
            min_outer_subpix_clamp_limit =
                min_outer_subpix_clamp_limit == 0
                    ? debug.subpix_window_clamp_limit
                    : std::min(min_outer_subpix_clamp_limit,
                               debug.subpix_window_clamp_limit);
          }
          outer_subpix_clamped =
              outer_subpix_clamped || debug.subpix_window_clamped;
          outer_subpix_config_scale =
              std::max(outer_subpix_config_scale,
                       debug.configured_outer_subpix_scale);
          if (debug.subpix_window_radius > 0 &&
              IsPointNearImage(debug.subpix_corner, output->size(), 120.0f)) {
            const int pre_radius =
                std::max(1, debug.pre_boost_subpix_window_radius);
            const int radius = std::max(1, debug.subpix_window_radius);
            if (pre_radius != radius) {
              cv::rectangle(
                  *output,
                  cv::Rect(static_cast<int>(std::lround(debug.subpix_corner.x)) -
                               pre_radius,
                           static_cast<int>(std::lround(debug.subpix_corner.y)) -
                               pre_radius,
                           pre_radius * 2 + 1, pre_radius * 2 + 1),
                  cv::Scalar(255, 170, 60), 1, cv::LINE_AA);
            }
            cv::rectangle(
                *output,
                cv::Rect(static_cast<int>(std::lround(debug.subpix_corner.x)) - radius,
                         static_cast<int>(std::lround(debug.subpix_corner.y)) - radius,
                         radius * 2 + 1, radius * 2 + 1),
                debug.close_edge_subpix_boost_applied
                    ? cv::Scalar(80, 255, 255)
                    : cv::Scalar(255, 180, 60),
                debug.close_edge_subpix_boost_applied ? 2 : 1, cv::LINE_AA);
          }
        }
      }
      std::ostringstream label;
      label << "B" << board->board_id << " used=" << used_internal
            << " invalid=" << invalid_internal
            << " outlier=" << outlier_internal
            << " regen_fail=" << regen_failed_internal
            << " frame_pose_unavailable=" << frame_pose_unavailable_internal
            << " other=" << other_rejected_internal
            << " boost=" << (outer_subpix_boost ? 1 : 0)
            << " polar=" << std::fixed << std::setprecision(1)
            << outer_subpix_max_polar_deg
            << " area=" << std::setprecision(3) << outer_subpix_area_ratio
            << " subpix_r=" << max_pre_boost_outer_subpix_radius
            << "->" << max_outer_subpix_radius
            << " raw=" << max_raw_outer_subpix_radius
            << " clamp=" << min_outer_subpix_clamp_limit
            << " scale=" << std::setprecision(2) << outer_subpix_config_scale
            << " clipped=" << (outer_subpix_clamped ? 1 : 0);
      cv::putText(*output, label.str(),
                  cv::Point(static_cast<int>(std::round(board_center.x)),
                            static_cast<int>(std::round(board_center.y))),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255),
                  1, cv::LINE_AA);
    }

    if (text_y < output->rows - 12) {
      std::ostringstream label;
      label << "B" << board->board_id << ": internal used="
            << used_internal << " invalid=" << invalid_internal
            << " outlier=" << outlier_internal
            << " regen_failed=" << regen_failed_internal
            << " frame_pose_unavailable=" << frame_pose_unavailable_internal
            << " other=" << other_rejected_internal;
      if (!board->points.empty()) {
        const int frame_index = board->points.front().frame_index;
        const auto regenerated_it =
            regenerated_by_frame_board.find(std::make_pair(frame_index, board->board_id));
        if (regenerated_it != regenerated_by_frame_board.end() &&
            regenerated_it->second != nullptr &&
            regenerated_it->second->detection.outer_detection.success) {
          bool outer_subpix_boost = false;
          int max_outer_subpix_radius = 0;
          int max_raw_outer_subpix_radius = 0;
          int max_pre_boost_outer_subpix_radius = 0;
          int min_outer_subpix_clamp_limit = 0;
          bool outer_subpix_clamped = false;
          double max_polar_deg = 0.0;
          for (const ati::OuterCornerVerificationDebugInfo& debug :
               regenerated_it->second->detection.outer_detection.corner_verification_debug) {
            outer_subpix_boost =
                outer_subpix_boost || debug.close_edge_subpix_boost_applied;
            max_outer_subpix_radius =
                std::max(max_outer_subpix_radius, debug.subpix_window_radius);
            max_raw_outer_subpix_radius =
                std::max(max_raw_outer_subpix_radius,
                         debug.raw_subpix_window_radius);
            max_pre_boost_outer_subpix_radius =
                std::max(max_pre_boost_outer_subpix_radius,
                         debug.pre_boost_subpix_window_radius);
            if (debug.subpix_window_clamp_limit > 0) {
              min_outer_subpix_clamp_limit =
                  min_outer_subpix_clamp_limit == 0
                      ? debug.subpix_window_clamp_limit
                      : std::min(min_outer_subpix_clamp_limit,
                                 debug.subpix_window_clamp_limit);
            }
            outer_subpix_clamped =
                outer_subpix_clamped || debug.subpix_window_clamped;
            max_polar_deg =
                std::max(max_polar_deg, debug.close_edge_subpix_max_polar_deg);
          }
          label << " boost=" << (outer_subpix_boost ? 1 : 0)
                << " polar=" << std::fixed << std::setprecision(1)
                << max_polar_deg
                << " subpix_r=" << max_pre_boost_outer_subpix_radius
                << "->" << max_outer_subpix_radius
                << " raw=" << max_raw_outer_subpix_radius
                << " clamp=" << min_outer_subpix_clamp_limit
                << " clipped=" << (outer_subpix_clamped ? 1 : 0);
        }
      }
      cv::putText(*output, label.str(), cv::Point(16, text_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.42,
                  cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      text_y += 20;
    }
  }
}

void DrawBackendUsedObservationOverlay(
    const cv::Mat& image,
    int frame_index,
    const std::vector<ati::JointPointObservation>& backend_points,
    const std::map<int, BackendInputBoardOverlaySource>& source_by_board,
    const std::map<BackendTrainingResidualKey, BackendTrainingResidualInfo>&
        backend_training_residuals,
    cv::Mat* output) {
  if (output == nullptr) {
    return;
  }
  if (image.empty()) {
    *output = cv::Mat();
    return;
  }
  if (image.channels() == 1) {
    cv::cvtColor(image, *output, cv::COLOR_GRAY2BGR);
  } else {
    *output = image.clone();
  }

  cv::rectangle(*output, cv::Point(8, 8), cv::Point(1188, 138),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(*output, "Final backend-used observations with selection source",
              cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX, 0.58,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "cyan cross=backend outer  green=backend BA internal <=3px",
              cv::Point(16, 54), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "yellow=backend BA internal >3px  blue=BA residual unavailable",
              cv::Point(16, 76), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output, "board outline/label source:",
              cv::Point(16, 98), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  DrawLegendSwatch(output, cv::Point(210, 98), cv::Scalar(60, 230, 60),
                   "seed");
  DrawLegendSwatch(output, cv::Point(292, 98), cv::Scalar(0, 165, 255),
                   "trial");
  DrawLegendSwatch(output, cv::Point(382, 98), cv::Scalar(255, 0, 255),
                   "frame cohesion");
  DrawLegendSwatch(output, cv::Point(548, 98), cv::Scalar(180, 180, 180),
                   "unknown");
  cv::putText(*output, "Internal point fill still encodes final BA residual quality.",
              cv::Point(16, 122), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

  std::map<int, std::vector<const ati::JointPointObservation*> > points_by_board;
  for (const ati::JointPointObservation& point : backend_points) {
    if (!point.used_in_solver) {
      continue;
    }
    points_by_board[point.board_id].push_back(&point);
  }

  int total_outer = 0;
  int total_internal = 0;
  int high_residual_internal = 0;
  int text_y = 146;
  for (const auto& board_entry : points_by_board) {
    const int board_id = board_entry.first;
    const std::vector<const ati::JointPointObservation*>& points =
        board_entry.second;
    BackendInputBoardOverlaySource board_source;
    const auto source_it = source_by_board.find(board_id);
    if (source_it != source_by_board.end()) {
      board_source = source_it->second;
    }
    const cv::Scalar board_source_color =
        BackendInputBoardSourceColor(board_source);
    const std::string board_source_label =
        BackendInputBoardSourceLabel(board_source);
    int outer_count = 0;
    int internal_count = 0;
    int board_high_residual = 0;
    cv::Point2f board_center(0.0f, 0.0f);
    int board_center_count = 0;
    std::vector<cv::Point> outer_polygon;
    outer_polygon.reserve(4);

    for (const ati::JointPointObservation* point_ptr : points) {
      if (point_ptr == nullptr) {
        continue;
      }
      const ati::JointPointObservation& point = *point_ptr;
      const cv::Point2f xy(static_cast<float>(point.image_xy.x()),
                           static_cast<float>(point.image_xy.y()));
      if (!IsPointNearImage(xy, output->size(), 80.0f)) {
        continue;
      }
      board_center += xy;
      ++board_center_count;

      if (point.point_type == ati::JointPointType::Outer) {
        ++outer_count;
        ++total_outer;
        outer_polygon.emplace_back(static_cast<int>(std::round(xy.x)),
                                   static_cast<int>(std::round(xy.y)));
        cv::drawMarker(*output, xy, board_source_color,
                       cv::MARKER_CROSS, 15, 2, cv::LINE_AA);
        cv::circle(*output, xy, 8, board_source_color, 1, cv::LINE_AA);
        continue;
      }

      ++internal_count;
      ++total_internal;
      BackendTrainingResidualKey key;
      key.frame_index = point.frame_index;
      key.board_id = point.board_id;
      key.point_id = point.point_id;
      const auto residual_it = backend_training_residuals.find(key);
      const bool has_residual = residual_it != backend_training_residuals.end();
      const double residual_norm =
          has_residual ? residual_it->second.residual_norm
                       : std::numeric_limits<double>::quiet_NaN();
      const bool high_residual =
          has_residual && std::isfinite(residual_norm) && residual_norm > 3.0;
      if (high_residual) {
        ++high_residual_internal;
        ++board_high_residual;
      }

      const cv::Scalar color =
          !has_residual
              ? cv::Scalar(255, 120, 0)
              : (high_residual ? cv::Scalar(0, 215, 255)
                               : cv::Scalar(60, 230, 60));
      cv::circle(*output, xy, 5, color, cv::FILLED, cv::LINE_AA);
      cv::circle(*output, xy, 7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      cv::circle(*output, xy, 9, board_source_color, 1, cv::LINE_AA);
      std::ostringstream label;
      label << "q=" << std::fixed << std::setprecision(2) << point.quality;
      if (has_residual) {
        label << " r=" << std::setprecision(1) << residual_norm;
      }
      cv::putText(*output, label.str(),
                  cv::Point(static_cast<int>(xy.x) + 6,
                            static_cast<int>(xy.y) - 6),
                  cv::FONT_HERSHEY_SIMPLEX, 0.30,
                  cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    if (outer_polygon.size() >= 4) {
      cv::polylines(*output, outer_polygon, true, board_source_color, 3,
                    cv::LINE_AA);
    }
    if (board_center_count > 0) {
      board_center *= 1.0f / static_cast<float>(board_center_count);
      std::ostringstream label;
      label << "B" << board_id << " " << board_source_label
            << " outer=" << outer_count
            << " internal=" << internal_count
            << " high_r=" << board_high_residual;
      cv::putText(*output, label.str(),
                  cv::Point(static_cast<int>(std::round(board_center.x)),
                            static_cast<int>(std::round(board_center.y))),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, board_source_color, 2,
                  cv::LINE_AA);
    }

    if (text_y < output->rows - 12) {
      std::ostringstream label;
      label << "frame " << frame_index << " B" << board_id
            << " [" << board_source_label << "]: backend outer="
            << outer_count
            << " internal=" << internal_count
            << " internal_gt3px=" << board_high_residual;
      if (!board_source.reason.empty()) {
        label << " reason=" << board_source.reason;
      }
      cv::putText(*output, label.str(), cv::Point(16, text_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.42, board_source_color, 1,
                  cv::LINE_AA);
      text_y += 20;
    }
  }

  std::ostringstream footer;
  footer << "backend outer=" << total_outer
         << " internal=" << total_internal
         << " internal_gt3px=" << high_residual_internal;
  cv::putText(*output, footer.str(), cv::Point(16, output->rows - 18),
              cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255, 255, 255),
              1, cv::LINE_AA);
}

void DrawCloseEdgeSoftCandidateOverlay(
    const cv::Mat& image,
    const ati::TrialBackendFrameBoardObservationDecision& decision,
    const std::vector<ati::JointPointObservation>& points,
    const std::map<BackendTrainingResidualKey, BackendTrainingResidualInfo>&
        backend_optimized_residuals,
    cv::Mat* output) {
  if (output == nullptr) {
    return;
  }
  if (image.empty()) {
    *output = cv::Mat();
    return;
  }
  if (image.channels() == 1) {
    cv::cvtColor(image, *output, cv::COLOR_GRAY2BGR);
  } else {
    *output = image.clone();
  }

  const cv::Scalar accepted_color(60, 230, 60);
  const cv::Scalar rejected_color(0, 120, 255);
  const cv::Scalar attempted_color(0, 215, 255);
  const cv::Scalar status_color =
      decision.soft_accepted
          ? accepted_color
          : (decision.soft_attempted ? attempted_color : rejected_color);

  cv::rectangle(*output, cv::Point(8, 8), cv::Point(1160, 154),
                cv::Scalar(0, 0, 0), cv::FILLED);
  std::ostringstream title;
  title << "Close-edge soft candidate  frame=" << decision.frame_index
        << "  board=" << decision.board_id
        << "  status="
        << (decision.soft_accepted
                ? "soft_accepted"
                : (decision.soft_attempted ? "soft_rejected_after_trial"
                                           : "soft_candidate_not_attempted"));
  cv::putText(*output, title.str(), cv::Point(16, 32),
              cv::FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2, cv::LINE_AA);

  std::ostringstream metrics1;
  metrics1 << std::fixed << std::setprecision(2)
           << "score=" << decision.close_edge_score
           << "  max_polar=" << decision.max_polar_angle_deg
           << "deg  area_ratio=" << decision.projected_area_ratio
           << "  outer_refit_rmse=" << decision.outer_pose_refit_rmse
           << "px  soft_weight=" << decision.soft_weight;
  cv::putText(*output, metrics1.str(), cv::Point(16, 58),
              cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(255, 255, 255), 1,
              cv::LINE_AA);

  std::ostringstream metrics2;
  metrics2 << std::fixed << std::setprecision(4)
           << "soft_delta global=" << decision.soft_global_rmse_delta
           << " outer=" << decision.soft_outer_rmse_delta
           << " internal=" << decision.soft_internal_rmse_delta
           << "  partition left/right/center/edge="
           << std::setprecision(2) << decision.left_rmse << "/"
           << decision.right_rmse << "/" << decision.center_side_rmse << "/"
           << decision.edge_side_rmse;
  cv::putText(*output, metrics2.str(), cv::Point(16, 82),
              cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(255, 255, 255), 1,
              cv::LINE_AA);

  cv::putText(*output,
              "cyan circle=observed outer  white circle=observed internal  "
              "magenta cross=backend projection  line=BA residual",
              cv::Point(16, 106), cv::FONT_HERSHEY_SIMPLEX, 0.40,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(*output,
              "green line <=3px  yellow line >3px  blue point=no backend "
              "residual available",
              cv::Point(16, 130), cv::FONT_HERSHEY_SIMPLEX, 0.40,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

  int outer_count = 0;
  int internal_count = 0;
  int residual_count = 0;
  int high_residual_count = 0;
  std::vector<cv::Point> outer_polygon;
  outer_polygon.reserve(4);
  std::vector<cv::Point2f> outer_points_for_scale;
  outer_points_for_scale.reserve(4);

  for (const ati::JointPointObservation& point : points) {
    const cv::Point2f observed(static_cast<float>(point.image_xy.x()),
                               static_cast<float>(point.image_xy.y()));
    if (!IsPointNearImage(observed, output->size(), 120.0f)) {
      continue;
    }

    BackendTrainingResidualKey key;
    key.frame_index = point.frame_index;
    key.board_id = point.board_id;
    key.point_id = point.point_id;
    const auto residual_it = backend_optimized_residuals.find(key);
    const bool has_residual = residual_it != backend_optimized_residuals.end();
    double residual_norm = std::numeric_limits<double>::quiet_NaN();
    if (has_residual) {
      residual_norm = residual_it->second.residual_norm;
      ++residual_count;
      if (std::isfinite(residual_norm) && residual_norm > 3.0) {
        ++high_residual_count;
      }
    }

    const bool is_outer = point.point_type == ati::JointPointType::Outer;
    if (is_outer) {
      ++outer_count;
      outer_points_for_scale.push_back(observed);
      outer_polygon.emplace_back(static_cast<int>(std::round(observed.x)),
                                 static_cast<int>(std::round(observed.y)));
      cv::circle(*output, observed, 6, cv::Scalar(255, 220, 0), 2,
                 cv::LINE_AA);
    } else {
      ++internal_count;
      cv::circle(*output, observed, 4, cv::Scalar(255, 255, 255),
                 cv::FILLED, cv::LINE_AA);
      cv::circle(*output, observed, 6, cv::Scalar(40, 40, 40), 1,
                 cv::LINE_AA);
    }

    if (has_residual) {
      const cv::Point2f predicted(
          static_cast<float>(residual_it->second.predicted_x),
          static_cast<float>(residual_it->second.predicted_y));
      const bool high_residual =
          std::isfinite(residual_norm) && residual_norm > 3.0;
      const cv::Scalar line_color =
          high_residual ? cv::Scalar(0, 215, 255)
                        : cv::Scalar(60, 230, 60);
      if (IsPointNearImage(predicted, output->size(), 160.0f)) {
        cv::line(*output, observed, predicted, line_color, 1, cv::LINE_AA);
        cv::drawMarker(*output, predicted, cv::Scalar(255, 0, 255),
                       cv::MARKER_CROSS, is_outer ? 16 : 11, 2,
                       cv::LINE_AA);
      }
      std::ostringstream label;
      label << "r=" << std::fixed << std::setprecision(1) << residual_norm;
      cv::putText(*output, label.str(),
                  cv::Point(static_cast<int>(observed.x) + 6,
                            static_cast<int>(observed.y) - 6),
                  cv::FONT_HERSHEY_SIMPLEX, 0.30, cv::Scalar(255, 255, 255),
                  1, cv::LINE_AA);
    } else {
      cv::drawMarker(*output, observed, cv::Scalar(255, 120, 0),
                     cv::MARKER_TILTED_CROSS, is_outer ? 14 : 10, 2,
                     cv::LINE_AA);
    }
  }

  if (outer_polygon.size() >= 4) {
    cv::polylines(*output, outer_polygon, true, cv::Scalar(255, 220, 0), 2,
                  cv::LINE_AA);
  }

  double estimated_local_scale = std::numeric_limits<double>::quiet_NaN();
  int estimated_subpix_radius = 0;
  if (outer_points_for_scale.size() == 4) {
    double scale_sum = 0.0;
    int scale_count = 0;
    for (int index = 0; index < 4; ++index) {
      const cv::Point2f& corner = outer_points_for_scale[static_cast<std::size_t>(index)];
      const cv::Point2f& prev =
          outer_points_for_scale[static_cast<std::size_t>((index + 3) % 4)];
      const cv::Point2f& next =
          outer_points_for_scale[static_cast<std::size_t>((index + 1) % 4)];
      const double prev_len = std::hypot(static_cast<double>(prev.x - corner.x),
                                         static_cast<double>(prev.y - corner.y));
      const double next_len = std::hypot(static_cast<double>(next.x - corner.x),
                                         static_cast<double>(next.y - corner.y));
      const double local = std::min(prev_len, next_len);
      if (std::isfinite(local) && local > 0.0) {
        scale_sum += local;
        ++scale_count;
      }
    }
    if (scale_count > 0) {
      estimated_local_scale = scale_sum / static_cast<double>(scale_count);
      const double marker_width = 0.25 * estimated_local_scale;
      double estimated_radius = 0.25 * marker_width;
      if (decision.is_close_edge_hard_case || decision.soft_candidate ||
          decision.soft_attempted || decision.soft_accepted) {
        estimated_radius *= 1.8;
      }
      estimated_subpix_radius =
          std::max(2, static_cast<int>(std::lround(estimated_radius)));
      for (const cv::Point2f& corner : outer_points_for_scale) {
        const cv::Rect window_rect(
            static_cast<int>(std::lround(corner.x)) - estimated_subpix_radius,
            static_cast<int>(std::lround(corner.y)) - estimated_subpix_radius,
            estimated_subpix_radius * 2 + 1,
            estimated_subpix_radius * 2 + 1);
        cv::rectangle(*output, window_rect, cv::Scalar(80, 220, 255), 1,
                      cv::LINE_AA);
      }
    }
  }

  std::ostringstream footer;
  footer << "points outer=" << outer_count << " internal=" << internal_count
         << " residual_available=" << residual_count
         << " residual_gt3px=" << high_residual_count
         << "  est_subpix_r=" << estimated_subpix_radius
         << " win=" << (estimated_subpix_radius * 2 + 1)
         << "  reason=" << decision.reason;
  cv::rectangle(*output, cv::Point(8, output->rows - 42),
                cv::Point(std::min(output->cols - 8, 1160),
                          output->rows - 8),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(*output, footer.str(), cv::Point(16, output->rows - 18),
              cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(255, 255, 255), 1,
              cv::LINE_AA);
}

struct RescuedBoardResidualComparison {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  bool local_pose_success = false;
  bool global_pose_available = false;
  double local_outer_rmse = std::numeric_limits<double>::quiet_NaN();
  double local_total_rmse = std::numeric_limits<double>::quiet_NaN();
  double local_internal_rmse = std::numeric_limits<double>::quiet_NaN();
  double global_total_rmse = std::numeric_limits<double>::quiet_NaN();
  double global_outer_rmse = std::numeric_limits<double>::quiet_NaN();
  double global_internal_rmse = std::numeric_limits<double>::quiet_NaN();
  int outer_point_count = 0;
  int internal_point_count = 0;
  int image_evidence_valid_internal_count = 0;
  int backend_used_internal_count = 0;
};

double RmseFromSquaredError(double squared_error_sum, int count) {
  return count > 0 ? std::sqrt(squared_error_sum / static_cast<double>(count))
                   : std::numeric_limits<double>::quiet_NaN();
}

void WriteRescuedBoardLocalVsGlobalResidualDiagnostics(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report,
    const std::vector<ati::InternalRegenerationFrameResult>& regeneration_results,
    const std::map<BackendTrainingResidualKey, BackendTrainingResidualInfo>&
        backend_training_residuals) {
  if (!report.baseline_result.stage5_bundle_available ||
      !report.baseline_result.final_stage5_bundle.scene_state.camera.IsValid()) {
    return;
  }
  const JointReprojectionSceneState scene_state =
      BuildJointSceneStateFromCalibrationSceneState(
          report.baseline_result.final_stage5_bundle.scene_state);
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(
          MakeIntermediateCameraConfigForDiagnostics(scene_state.camera));
  if (!camera.IsValid()) {
    return;
  }

  std::vector<RescuedBoardResidualComparison> comparisons;
  for (const ati::InternalRegenerationFrameResult& frame :
       regeneration_results) {
    const JointSceneFrameState* frame_state =
        FindFrameStateForDiagnostics(scene_state, frame.frame_index);
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      const ati::ApriltagInternalDetectionResult& detection =
          measurement.detection;
      if (!detection.outer_detection.success ||
          !detection.outer_detection.used_local_patch_rescue) {
        continue;
      }
      const JointSceneBoardState* board_state =
          FindBoardStateForDiagnostics(scene_state, measurement.board_id);
      const ApriltagCanonicalModel model =
          ModelForBoardIdForDiagnostics(
              report.baseline_result.effective_options.config,
              measurement.board_id);

      RescuedBoardResidualComparison row;
      row.frame_index = frame.frame_index;
      row.frame_label = frame.frame_label;
      row.board_id = measurement.board_id;

      Eigen::Isometry3d T_camera_board_local = Eigen::Isometry3d::Identity();
      row.local_pose_success = EstimatePoseFromOuterCornersForDiagnostics(
          camera, model, detection.outer_detection, &T_camera_board_local,
          &row.local_outer_rmse);

      Eigen::Isometry3d T_camera_board_global = Eigen::Isometry3d::Identity();
      if (frame_state != nullptr && frame_state->initialized &&
          board_state != nullptr && board_state->initialized) {
        T_camera_board_global =
            Eigen::Isometry3d(frame_state->T_camera_reference) *
            Eigen::Isometry3d(board_state->T_reference_board);
        row.global_pose_available = true;
      }

      double local_total_sq = 0.0;
      double local_outer_sq = 0.0;
      double local_internal_sq = 0.0;
      int local_total_count = 0;
      int local_outer_count = 0;
      int local_internal_count = 0;
      double global_total_sq = 0.0;
      double global_outer_sq = 0.0;
      double global_internal_sq = 0.0;
      int global_total_count = 0;
      int global_outer_count = 0;
      int global_internal_count = 0;

      const std::array<int, 4> outer_point_ids{{
          model.PointId(0, 0),
          model.PointId(model.ModuleDimension(), 0),
          model.PointId(model.ModuleDimension(), model.ModuleDimension()),
          model.PointId(0, model.ModuleDimension()),
      }};
      for (int corner_index = 0; corner_index < 4; ++corner_index) {
        if (!detection.outer_detection.refined_valid
                 [static_cast<std::size_t>(corner_index)]) {
          continue;
        }
        const CanonicalCorner& corner =
            model.corner(outer_point_ids[static_cast<std::size_t>(corner_index)]);
        const Eigen::Vector2d& observed =
            detection.outer_detection.refined_corners_original_image
                [static_cast<std::size_t>(corner_index)];
        ++row.outer_point_count;
        if (row.local_pose_success) {
          AccumulateProjectionResidualForDiagnostics(
              camera, T_camera_board_local, corner.target_xyz, observed,
              &local_total_sq, &local_total_count, &local_outer_sq,
              &local_outer_count, &local_internal_sq, &local_internal_count,
              false);
        }
        if (row.global_pose_available) {
          AccumulateProjectionResidualForDiagnostics(
              camera, T_camera_board_global, corner.target_xyz, observed,
              &global_total_sq, &global_total_count, &global_outer_sq,
              &global_outer_count, &global_internal_sq,
              &global_internal_count, false);
        }
      }

      for (const ati::InternalCornerDebugInfo& debug :
           detection.internal_corner_debug) {
        if (!debug.valid) {
          continue;
        }
        ++row.internal_point_count;
        if (debug.image_evidence_valid) {
          ++row.image_evidence_valid_internal_count;
        }
        BackendTrainingResidualKey key;
        key.frame_index = frame.frame_index;
        key.board_id = measurement.board_id;
        key.point_id = debug.point_id;
        if (backend_training_residuals.find(key) !=
            backend_training_residuals.end()) {
          ++row.backend_used_internal_count;
        }
        const CanonicalCorner& corner = model.corner(debug.point_id);
        const Eigen::Vector2d observed(debug.refined_image.x,
                                       debug.refined_image.y);
        if (row.local_pose_success) {
          AccumulateProjectionResidualForDiagnostics(
              camera, T_camera_board_local, corner.target_xyz, observed,
              &local_total_sq, &local_total_count, &local_outer_sq,
              &local_outer_count, &local_internal_sq, &local_internal_count,
              true);
        }
        if (row.global_pose_available) {
          AccumulateProjectionResidualForDiagnostics(
              camera, T_camera_board_global, corner.target_xyz, observed,
              &global_total_sq, &global_total_count, &global_outer_sq,
              &global_outer_count, &global_internal_sq,
              &global_internal_count, true);
        }
      }

      row.local_total_rmse =
          RmseFromSquaredError(local_total_sq, local_total_count);
      row.local_internal_rmse =
          RmseFromSquaredError(local_internal_sq, local_internal_count);
      row.global_total_rmse =
          RmseFromSquaredError(global_total_sq, global_total_count);
      row.global_outer_rmse =
          RmseFromSquaredError(global_outer_sq, global_outer_count);
      row.global_internal_rmse =
          RmseFromSquaredError(global_internal_sq, global_internal_count);
      comparisons.push_back(row);
    }
  }

  std::ofstream csv(
      (output_dir / "rescued_board_local_vs_global_residuals.csv")
          .string()
          .c_str());
  csv << "frame_index,frame_label,board_id,local_pose_success,"
      << "global_pose_available,outer_point_count,internal_point_count,"
      << "image_evidence_valid_internal_count,backend_used_internal_count,"
      << "local_outer_rmse,local_internal_rmse,local_total_rmse,"
      << "global_outer_rmse,global_internal_rmse,global_total_rmse,"
      << "global_minus_local_internal_rmse,global_minus_local_total_rmse\n";
  for (const RescuedBoardResidualComparison& row : comparisons) {
    csv << row.frame_index << ","
        << CsvEscape(row.frame_label) << ","
        << row.board_id << ","
        << (row.local_pose_success ? 1 : 0) << ","
        << (row.global_pose_available ? 1 : 0) << ","
        << row.outer_point_count << ","
        << row.internal_point_count << ","
        << row.image_evidence_valid_internal_count << ","
        << row.backend_used_internal_count << ","
        << row.local_outer_rmse << ","
        << row.local_internal_rmse << ","
        << row.local_total_rmse << ","
        << row.global_outer_rmse << ","
        << row.global_internal_rmse << ","
        << row.global_total_rmse << ","
        << (row.global_internal_rmse - row.local_internal_rmse) << ","
        << (row.global_total_rmse - row.local_total_rmse) << "\n";
  }

  double local_internal_sq_sum = 0.0;
  double global_internal_sq_sum = 0.0;
  double local_total_sq_sum = 0.0;
  double global_total_sq_sum = 0.0;
  int valid_rows = 0;
  int global_worse_count = 0;
  int image_evidence_valid_count = 0;
  int internal_count = 0;
  for (const RescuedBoardResidualComparison& row : comparisons) {
    if (std::isfinite(row.local_internal_rmse) &&
        std::isfinite(row.global_internal_rmse)) {
      local_internal_sq_sum +=
          row.local_internal_rmse * row.local_internal_rmse;
      global_internal_sq_sum +=
          row.global_internal_rmse * row.global_internal_rmse;
      if (row.global_internal_rmse > row.local_internal_rmse) {
        ++global_worse_count;
      }
      ++valid_rows;
    }
    if (std::isfinite(row.local_total_rmse) &&
        std::isfinite(row.global_total_rmse)) {
      local_total_sq_sum += row.local_total_rmse * row.local_total_rmse;
      global_total_sq_sum += row.global_total_rmse * row.global_total_rmse;
    }
    image_evidence_valid_count += row.image_evidence_valid_internal_count;
    internal_count += row.internal_point_count;
  }
  std::ofstream summary(
      (output_dir / "rescued_board_local_vs_global_residuals_summary.txt")
          .string()
          .c_str());
  summary << "rescued_board_observation_count: " << comparisons.size() << "\n";
  summary << "valid_comparison_count: " << valid_rows << "\n";
  summary << "global_internal_worse_than_local_count: "
          << global_worse_count << "\n";
  summary << "rescued_internal_point_count: " << internal_count << "\n";
  summary << "image_evidence_valid_internal_count: "
          << image_evidence_valid_count << "\n";
  summary << "image_evidence_valid_internal_ratio: "
          << SafeRatio(image_evidence_valid_count, internal_count) << "\n";
  summary << "mean_local_internal_rmse: "
          << (valid_rows > 0 ? std::sqrt(local_internal_sq_sum /
                                         static_cast<double>(valid_rows))
                             : 0.0)
          << "\n";
  summary << "mean_global_internal_rmse: "
          << (valid_rows > 0 ? std::sqrt(global_internal_sq_sum /
                                         static_cast<double>(valid_rows))
                             : 0.0)
          << "\n";
  summary << "mean_local_total_rmse: "
          << (valid_rows > 0 ? std::sqrt(local_total_sq_sum /
                                         static_cast<double>(valid_rows))
                             : 0.0)
          << "\n";
  summary << "mean_global_total_rmse: "
          << (valid_rows > 0 ? std::sqrt(global_total_sq_sum /
                                         static_cast<double>(valid_rows))
                             : 0.0)
          << "\n";
  summary << "interpretation: if local residual is low but global residual is "
          << "high, rescued points are geometrically plausible but disagree "
          << "with the current global scene state. If both are high, rescued "
          << "internal localization is still poor.\n";
}

struct GlobalSceneConsistencyObservationRow {
  int frame_index = -1;
  std::string frame_label;
  int board_id = -1;
  std::string observation_source = "unknown";
  std::string state_stage = "backend_optimized";
  std::string T_reference_board_source = "final_stage5_bundle.backend_optimized";
  std::string T_camera_reference_source = "final_stage5_bundle.backend_optimized";
  std::string generation_state_source = "round2_final_regeneration";
  std::string selection_state_source = "round2_final_selection";
  std::string backend_input_state_source = "backend_problem_input";
  double global_outer_rmse = std::numeric_limits<double>::quiet_NaN();
  double local_outer_rmse = std::numeric_limits<double>::quiet_NaN();
  double global_minus_local_rmse = std::numeric_limits<double>::quiet_NaN();
  bool local_pose_refit_success = false;
  double pose_delta_vs_global_rotation_deg =
      std::numeric_limits<double>::quiet_NaN();
  double pose_delta_vs_global_translation_norm =
      std::numeric_limits<double>::quiet_NaN();
  bool T_cam_reference_from_board_available = false;
  bool is_reference_board = false;
  bool is_gauge_board = false;
  bool board_pose_is_identity = false;
  std::string diagnosis_label = "unknown";
  std::string reject_reason_or_note;
};

struct GlobalSceneConsistencyFrameAggregate {
  int frame_index = -1;
  std::string frame_label;
  int visible_board_count = 0;
  int normal_detected_board_count = 0;
  int rescued_board_count = 0;
  int observation_count = 0;
  int bad_global_good_local_board_count = 0;
  double global_outer_rmse_sq_sum = 0.0;
  double local_outer_rmse_sq_sum = 0.0;
  double max_global_outer_rmse = 0.0;
  double pose_rot_sum = 0.0;
  double pose_trans_sum = 0.0;
  double max_pose_rot = 0.0;
  double max_pose_trans = 0.0;
  int pose_delta_count = 0;
};

struct GlobalSceneConsistencyBoardAggregate {
  int board_id = -1;
  int observed_frame_count = 0;
  int normal_detected_frame_count = 0;
  int rescued_frame_count = 0;
  int bad_global_good_local_frame_count = 0;
  double global_outer_rmse_sq_sum = 0.0;
  double local_outer_rmse_sq_sum = 0.0;
  double max_global_outer_rmse = 0.0;
  double pose_rot_sum = 0.0;
  double pose_trans_sum = 0.0;
  int pose_delta_count = 0;
};

struct FramePoseRepairDiagnosticRow {
  int frame_index = -1;
  std::string frame_label;
  int candidate_board_count = 0;
  int candidate_point_count = 0;
  int evaluated_board_count = 0;
  bool refit_success = false;
  double refit_support_outer_rmse = std::numeric_limits<double>::quiet_NaN();
  double current_global_mean_outer_rmse = std::numeric_limits<double>::quiet_NaN();
  double repaired_mean_outer_rmse = std::numeric_limits<double>::quiet_NaN();
  double current_global_max_outer_rmse = std::numeric_limits<double>::quiet_NaN();
  double repaired_max_outer_rmse = std::numeric_limits<double>::quiet_NaN();
  double current_to_repaired_rotation_deg =
      std::numeric_limits<double>::quiet_NaN();
  double current_to_repaired_translation_norm =
      std::numeric_limits<double>::quiet_NaN();
  int improved_board_count = 0;
  int worsened_board_count = 0;
  std::string support_board_ids;
  std::string evaluated_board_ids;
  std::string diagnosis_label;
};

std::string DiagnosisLabelForGlobalSceneAudit(
    bool local_success,
    double local_rmse,
    double global_rmse,
    const std::string& observation_source) {
  constexpr double kLocalGoodThresholdPx = 3.0;
  constexpr double kGlobalBadThresholdPx = 10.0;
  if (!local_success || !std::isfinite(local_rmse)) {
    return "local_pose_failed";
  }
  if (local_rmse > kLocalGoodThresholdPx) {
    return "local_bad";
  }
  if (std::isfinite(global_rmse) && global_rmse > kGlobalBadThresholdPx) {
    if (observation_source == "rescued") {
      return "local_good_global_bad_rescued";
    }
    return "local_good_global_bad";
  }
  if (std::isfinite(global_rmse)) {
    return "ok_global_consistent";
  }
  return "global_pose_unavailable";
}

std::string JoinIntsForDiagnostics(const std::vector<int>& values) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      stream << "|";
    }
    stream << values[index];
  }
  return stream.str();
}

std::string SuspectFrameReason(
    const GlobalSceneConsistencyFrameAggregate& aggregate) {
  if (aggregate.observation_count <= 0) {
    return "no_observations";
  }
  if (aggregate.bad_global_good_local_board_count >= 2) {
    return "multiple_boards_local_good_global_bad";
  }
  if (aggregate.bad_global_good_local_board_count == 1) {
    return "single_board_local_good_global_bad";
  }
  return "no_strong_frame_pose_signal";
}

std::string SuspectBoardReason(
    const GlobalSceneConsistencyBoardAggregate& aggregate) {
  if (aggregate.observed_frame_count <= 0) {
    return "no_observations";
  }
  const double ratio =
      SafeRatio(aggregate.bad_global_good_local_frame_count,
                aggregate.observed_frame_count);
  if (aggregate.bad_global_good_local_frame_count >= 3 && ratio > 0.25) {
    return "repeated_cross_frame_local_good_global_bad";
  }
  if (aggregate.bad_global_good_local_frame_count > 0) {
    return "some_local_good_global_bad_observations";
  }
  return "no_strong_board_pose_signal";
}

void WriteGlobalSceneStateConsistencyAudit(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report) {
  if (!report.baseline_result.stage5_bundle_available ||
      !report.baseline_result.final_stage5_bundle.scene_state.camera.IsValid()) {
    return;
  }
  const ati::FrozenRoundArtifacts& artifacts =
      report.baseline_result.round2_available ? report.baseline_result.round2
                                               : report.baseline_result.round1;
  const std::string round_label =
      report.baseline_result.round2_available ? "round2_final" : "round1_final";
  const fs::path audit_dir =
      output_dir / "global_scene_state_consistency_audit";
  EnsureDirectoryExists(audit_dir);

  const JointReprojectionSceneState scene_state =
      BuildJointSceneStateFromCalibrationSceneState(
          report.baseline_result.final_stage5_bundle.scene_state);
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(
          MakeIntermediateCameraConfigForDiagnostics(scene_state.camera));
  if (!camera.IsValid()) {
    return;
  }

  std::vector<GlobalSceneConsistencyObservationRow> rows;
  std::map<std::pair<int, int>, const ati::RegeneratedBoardMeasurement*>
      regenerated_measurements_by_key;
  rows.reserve(artifacts.regeneration_results.size() * 4);
  for (const ati::InternalRegenerationFrameResult& frame :
       artifacts.regeneration_results) {
    const JointSceneFrameState* frame_state =
        FindFrameStateForDiagnostics(scene_state, frame.frame_index);
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      regenerated_measurements_by_key[std::make_pair(frame.frame_index,
                                                     measurement.board_id)] =
          &measurement;
      const ati::OuterTagDetectionResult& outer =
          measurement.detection.outer_detection;
      if (!outer.success) {
        continue;
      }
      int valid_corner_count = 0;
      for (bool valid : outer.refined_valid) {
        valid_corner_count += valid ? 1 : 0;
      }
      if (valid_corner_count < 4) {
        continue;
      }

      const JointSceneBoardState* board_state =
          FindBoardStateForDiagnostics(scene_state, measurement.board_id);
      const ApriltagCanonicalModel model =
          ModelForBoardIdForDiagnostics(
              report.baseline_result.effective_options.config,
              measurement.board_id);

      GlobalSceneConsistencyObservationRow row;
      row.frame_index = frame.frame_index;
      row.frame_label = frame.frame_label;
      row.board_id = measurement.board_id;
      row.observation_source =
          outer.used_local_patch_rescue ? "rescued" : "normal_detected";
      row.state_stage = "backend_optimized";
      row.generation_state_source =
          outer.used_local_patch_rescue
              ? (outer.local_patch_rescue_summary.empty()
                     ? "geometry_prior_rescue_unknown"
                     : outer.local_patch_rescue_summary)
              : round_label + "_normal_detection";
      row.selection_state_source = round_label + "_selection";
      row.backend_input_state_source =
          report.backend_problem_input.scene_state.coarse_or_optimized_level.empty()
              ? "backend_problem_input"
              : report.backend_problem_input.scene_state.coarse_or_optimized_level;
      row.is_reference_board =
          measurement.board_id == scene_state.reference_board_id;
      row.is_gauge_board = row.is_reference_board;

      Eigen::Isometry3d T_camera_board_local = Eigen::Isometry3d::Identity();
      row.local_pose_refit_success = EstimatePoseFromOuterCornersForDiagnostics(
          camera, model, outer, &T_camera_board_local, &row.local_outer_rmse);

      bool global_pose_available = false;
      Eigen::Isometry3d T_camera_board_global = Eigen::Isometry3d::Identity();
      Eigen::Isometry3d T_reference_board = Eigen::Isometry3d::Identity();
      if (row.is_reference_board) {
        row.board_pose_is_identity = true;
      } else if (board_state != nullptr && board_state->initialized) {
        T_reference_board = Eigen::Isometry3d(board_state->T_reference_board);
        row.board_pose_is_identity =
            (board_state->T_reference_board -
             Eigen::Matrix4d::Identity()).norm() < 1e-9;
      }
      if (frame_state != nullptr && frame_state->initialized &&
          (row.is_reference_board ||
           (board_state != nullptr && board_state->initialized))) {
        T_camera_board_global =
            Eigen::Isometry3d(frame_state->T_camera_reference) *
            T_reference_board;
        global_pose_available = true;
      }

      double global_outer_sq = 0.0;
      int global_outer_count = 0;
      const std::array<int, 4> outer_point_ids{{
          model.PointId(0, 0),
          model.PointId(model.ModuleDimension(), 0),
          model.PointId(model.ModuleDimension(), model.ModuleDimension()),
          model.PointId(0, model.ModuleDimension()),
      }};
      if (global_pose_available) {
        for (int corner_index = 0; corner_index < 4; ++corner_index) {
          const CanonicalCorner& corner = model.corner(
              outer_point_ids[static_cast<std::size_t>(corner_index)]);
          const Eigen::Vector2d& observed =
              outer.refined_corners_original_image
                  [static_cast<std::size_t>(corner_index)];
          Eigen::Vector2d projected;
          double sq = 1e12;
          if (camera.vsEuclideanToKeypoint(
                  T_camera_board_global * corner.target_xyz, &projected)) {
            sq = (projected - observed).squaredNorm();
          }
          global_outer_sq += sq;
          ++global_outer_count;
        }
        row.global_outer_rmse =
            RmseFromSquaredError(global_outer_sq, global_outer_count);
      }
      row.global_minus_local_rmse =
          row.global_outer_rmse - row.local_outer_rmse;

      if (row.local_pose_refit_success && global_pose_available) {
        const Eigen::Isometry3d T_camera_reference_from_board =
            T_camera_board_local * T_reference_board.inverse();
        const Eigen::Isometry3d delta =
            Eigen::Isometry3d(frame_state->T_camera_reference).inverse() *
            T_camera_reference_from_board;
        row.T_cam_reference_from_board_available = true;
        row.pose_delta_vs_global_rotation_deg =
            RotationAngleDegForDiagnostics(delta.linear());
        row.pose_delta_vs_global_translation_norm =
            TranslationNormForDiagnostics(delta);
      }

      row.diagnosis_label = DiagnosisLabelForGlobalSceneAudit(
          row.local_pose_refit_success, row.local_outer_rmse,
          row.global_outer_rmse, row.observation_source);
      if (row.is_reference_board &&
          row.diagnosis_label.find("local_good_global_bad") !=
              std::string::npos) {
        row.reject_reason_or_note =
            "reference_board_local_good_global_bad_prioritize_frame_pose";
      } else if (row.observation_source == "rescued" &&
                 row.diagnosis_label.find("local_good_global_bad") !=
                     std::string::npos) {
        row.reject_reason_or_note =
            "rescued_observation_disagrees_with_global_scene_state";
      }
      rows.push_back(row);
    }
  }

  std::map<int, GlobalSceneConsistencyFrameAggregate> by_frame;
  std::map<int, GlobalSceneConsistencyBoardAggregate> by_board;
  std::map<std::string, int> diagnosis_counts;
  int local_good_global_bad_count = 0;
  int reference_board_bad_count = 0;
  for (const GlobalSceneConsistencyObservationRow& row : rows) {
    ++diagnosis_counts[row.diagnosis_label];
    const bool local_good_global_bad =
        row.diagnosis_label.find("local_good_global_bad") != std::string::npos;
    local_good_global_bad_count += local_good_global_bad ? 1 : 0;
    reference_board_bad_count +=
        (local_good_global_bad && row.is_reference_board) ? 1 : 0;

    GlobalSceneConsistencyFrameAggregate& frame = by_frame[row.frame_index];
    frame.frame_index = row.frame_index;
    frame.frame_label = row.frame_label;
    ++frame.observation_count;
    if (row.observation_source == "rescued") {
      ++frame.rescued_board_count;
    } else {
      ++frame.normal_detected_board_count;
    }
    frame.visible_board_count =
        frame.normal_detected_board_count + frame.rescued_board_count;
    if (local_good_global_bad) {
      ++frame.bad_global_good_local_board_count;
    }
    if (std::isfinite(row.global_outer_rmse)) {
      frame.global_outer_rmse_sq_sum +=
          row.global_outer_rmse * row.global_outer_rmse;
      frame.max_global_outer_rmse =
          std::max(frame.max_global_outer_rmse, row.global_outer_rmse);
    }
    if (std::isfinite(row.local_outer_rmse)) {
      frame.local_outer_rmse_sq_sum +=
          row.local_outer_rmse * row.local_outer_rmse;
    }
    if (std::isfinite(row.pose_delta_vs_global_rotation_deg)) {
      frame.pose_rot_sum += row.pose_delta_vs_global_rotation_deg;
      frame.max_pose_rot =
          std::max(frame.max_pose_rot, row.pose_delta_vs_global_rotation_deg);
      frame.pose_trans_sum += row.pose_delta_vs_global_translation_norm;
      frame.max_pose_trans =
          std::max(frame.max_pose_trans,
                   row.pose_delta_vs_global_translation_norm);
      ++frame.pose_delta_count;
    }

    GlobalSceneConsistencyBoardAggregate& board = by_board[row.board_id];
    board.board_id = row.board_id;
    ++board.observed_frame_count;
    if (row.observation_source == "rescued") {
      ++board.rescued_frame_count;
    } else {
      ++board.normal_detected_frame_count;
    }
    if (local_good_global_bad) {
      ++board.bad_global_good_local_frame_count;
    }
    if (std::isfinite(row.global_outer_rmse)) {
      board.global_outer_rmse_sq_sum +=
          row.global_outer_rmse * row.global_outer_rmse;
      board.max_global_outer_rmse =
          std::max(board.max_global_outer_rmse, row.global_outer_rmse);
    }
    if (std::isfinite(row.local_outer_rmse)) {
      board.local_outer_rmse_sq_sum +=
          row.local_outer_rmse * row.local_outer_rmse;
    }
    if (std::isfinite(row.pose_delta_vs_global_rotation_deg)) {
      board.pose_rot_sum += row.pose_delta_vs_global_rotation_deg;
      board.pose_trans_sum += row.pose_delta_vs_global_translation_norm;
      ++board.pose_delta_count;
    }
  }

  std::ofstream per_obs(
      (audit_dir / "global_scene_consistency_per_observation.csv")
          .string()
          .c_str());
  per_obs
      << "frame_index,frame_label,board_id,observation_source,state_stage,"
      << "T_reference_board_source,T_cam_reference_source,"
      << "generation_state_source,selection_state_source,"
      << "backend_input_state_source,global_outer_rmse,local_outer_rmse,"
      << "global_minus_local_rmse,local_pose_refit_success,"
      << "pose_delta_vs_global_rotation_deg,"
      << "pose_delta_vs_global_translation_norm,"
      << "T_cam_reference_from_board_available,is_reference_board,"
      << "is_gauge_board,board_pose_is_identity,diagnosis_label,"
      << "reject_reason_or_note\n";
  for (const GlobalSceneConsistencyObservationRow& row : rows) {
    per_obs << row.frame_index << ","
            << CsvEscape(row.frame_label) << ","
            << row.board_id << ","
            << row.observation_source << ","
            << row.state_stage << ","
            << CsvEscape(row.T_reference_board_source) << ","
            << CsvEscape(row.T_camera_reference_source) << ","
            << CsvEscape(row.generation_state_source) << ","
            << CsvEscape(row.selection_state_source) << ","
            << CsvEscape(row.backend_input_state_source) << ","
            << row.global_outer_rmse << ","
            << row.local_outer_rmse << ","
            << row.global_minus_local_rmse << ","
            << (row.local_pose_refit_success ? 1 : 0) << ","
            << row.pose_delta_vs_global_rotation_deg << ","
            << row.pose_delta_vs_global_translation_norm << ","
            << (row.T_cam_reference_from_board_available ? 1 : 0) << ","
            << (row.is_reference_board ? 1 : 0) << ","
            << (row.is_gauge_board ? 1 : 0) << ","
            << (row.board_pose_is_identity ? 1 : 0) << ","
            << row.diagnosis_label << ","
            << CsvEscape(row.reject_reason_or_note) << "\n";
  }

  std::ofstream frame_csv(
      (audit_dir / "global_scene_consistency_by_frame.csv").string().c_str());
  frame_csv
      << "frame_index,frame_label,visible_board_count,"
      << "normal_detected_board_count,rescued_board_count,"
      << "mean_global_outer_rmse,mean_local_outer_rmse,max_global_outer_rmse,"
      << "bad_global_good_local_board_count,"
      << "mean_pose_delta_vs_global_rotation_deg,"
      << "max_pose_delta_vs_global_rotation_deg,"
      << "mean_pose_delta_vs_global_translation_norm,"
      << "max_pose_delta_vs_global_translation_norm,"
      << "suspect_frame_T_cam_reference,suspect_reason\n";
  for (const auto& entry : by_frame) {
    const GlobalSceneConsistencyFrameAggregate& frame = entry.second;
    const bool suspect = frame.bad_global_good_local_board_count >= 2;
    frame_csv << frame.frame_index << ","
              << CsvEscape(frame.frame_label) << ","
              << frame.visible_board_count << ","
              << frame.normal_detected_board_count << ","
              << frame.rescued_board_count << ","
              << RmseFromSquaredError(frame.global_outer_rmse_sq_sum,
                                      frame.observation_count)
              << ","
              << RmseFromSquaredError(frame.local_outer_rmse_sq_sum,
                                      frame.observation_count)
              << ","
              << frame.max_global_outer_rmse << ","
              << frame.bad_global_good_local_board_count << ","
              << (frame.pose_delta_count > 0
                      ? frame.pose_rot_sum /
                            static_cast<double>(frame.pose_delta_count)
                      : std::numeric_limits<double>::quiet_NaN())
              << ","
              << frame.max_pose_rot << ","
              << (frame.pose_delta_count > 0
                      ? frame.pose_trans_sum /
                            static_cast<double>(frame.pose_delta_count)
                      : std::numeric_limits<double>::quiet_NaN())
              << ","
              << frame.max_pose_trans << ","
              << (suspect ? 1 : 0) << ","
              << SuspectFrameReason(frame) << "\n";
  }

  std::ofstream board_csv(
      (audit_dir / "global_scene_consistency_by_board.csv").string().c_str());
  board_csv
      << "board_id,observed_frame_count,normal_detected_frame_count,"
      << "rescued_frame_count,mean_global_outer_rmse,mean_local_outer_rmse,"
      << "max_global_outer_rmse,bad_global_good_local_frame_count,"
      << "bad_global_good_local_ratio,"
      << "mean_pose_delta_vs_global_rotation_deg,"
      << "mean_pose_delta_vs_global_translation_norm,"
      << "suspect_board_T_reference_board,suspect_reason\n";
  for (const auto& entry : by_board) {
    const GlobalSceneConsistencyBoardAggregate& board = entry.second;
    const double bad_ratio =
        SafeRatio(board.bad_global_good_local_frame_count,
                  board.observed_frame_count);
    const bool suspect_board =
        board.bad_global_good_local_frame_count >= 3 && bad_ratio > 0.25;
    board_csv << board.board_id << ","
              << board.observed_frame_count << ","
              << board.normal_detected_frame_count << ","
              << board.rescued_frame_count << ","
              << RmseFromSquaredError(board.global_outer_rmse_sq_sum,
                                      board.observed_frame_count)
              << ","
              << RmseFromSquaredError(board.local_outer_rmse_sq_sum,
                                      board.observed_frame_count)
              << ","
              << board.max_global_outer_rmse << ","
              << board.bad_global_good_local_frame_count << ","
              << bad_ratio << ","
              << (board.pose_delta_count > 0
                      ? board.pose_rot_sum /
                            static_cast<double>(board.pose_delta_count)
                      : std::numeric_limits<double>::quiet_NaN())
              << ","
              << (board.pose_delta_count > 0
                      ? board.pose_trans_sum /
                            static_cast<double>(board.pose_delta_count)
                      : std::numeric_limits<double>::quiet_NaN())
              << ","
              << (suspect_board ? 1 : 0) << ","
              << SuspectBoardReason(board) << "\n";
  }

  std::vector<FramePoseRepairDiagnosticRow> repair_rows;
  for (const auto& entry : by_frame) {
    const GlobalSceneConsistencyFrameAggregate& frame_aggregate = entry.second;
    if (frame_aggregate.bad_global_good_local_board_count <= 0 &&
        frame_aggregate.max_global_outer_rmse < 20.0) {
      continue;
    }
    const JointSceneFrameState* frame_state =
        FindFrameStateForDiagnostics(scene_state, frame_aggregate.frame_index);
    if (frame_state == nullptr || !frame_state->initialized) {
      continue;
    }
    std::vector<Eigen::Vector3d> support_points_reference;
    std::vector<Eigen::Vector2d> support_points_image;
    std::vector<int> support_board_ids;
    std::vector<int> evaluated_board_ids;
    struct EvaluationBoardData {
      int board_id = -1;
      Eigen::Isometry3d T_reference_board = Eigen::Isometry3d::Identity();
      std::vector<Eigen::Vector3d> target_points_board;
      std::vector<Eigen::Vector2d> image_points;
    };
    std::vector<EvaluationBoardData> eval_boards;
    for (const GlobalSceneConsistencyObservationRow& row : rows) {
      if (row.frame_index != frame_aggregate.frame_index) {
        continue;
      }
      const auto measurement_it =
          regenerated_measurements_by_key.find(std::make_pair(row.frame_index,
                                                              row.board_id));
      if (measurement_it == regenerated_measurements_by_key.end()) {
        continue;
      }
      const ati::RegeneratedBoardMeasurement& measurement =
          *measurement_it->second;
      const ati::OuterTagDetectionResult& outer =
          measurement.detection.outer_detection;
      if (!outer.success) {
        continue;
      }
      const JointSceneBoardState* board_state =
          FindBoardStateForDiagnostics(scene_state, row.board_id);
      if (!row.is_reference_board &&
          (board_state == nullptr || !board_state->initialized)) {
        continue;
      }
      Eigen::Isometry3d T_reference_board = Eigen::Isometry3d::Identity();
      if (!row.is_reference_board) {
        T_reference_board =
            Eigen::Isometry3d(board_state->T_reference_board);
      }
      const ApriltagCanonicalModel model =
          ModelForBoardIdForDiagnostics(
              report.baseline_result.effective_options.config, row.board_id);
      const std::array<int, 4> outer_point_ids{{
          model.PointId(0, 0),
          model.PointId(model.ModuleDimension(), 0),
          model.PointId(model.ModuleDimension(), model.ModuleDimension()),
          model.PointId(0, model.ModuleDimension()),
      }};
      EvaluationBoardData eval;
      eval.board_id = row.board_id;
      eval.T_reference_board = T_reference_board;
      bool complete = true;
      for (int corner_index = 0; corner_index < 4; ++corner_index) {
        if (!outer.refined_valid[static_cast<std::size_t>(corner_index)]) {
          complete = false;
          break;
        }
        const CanonicalCorner& corner =
            model.corner(outer_point_ids[static_cast<std::size_t>(corner_index)]);
        eval.target_points_board.push_back(corner.target_xyz);
        eval.image_points.push_back(
            outer.refined_corners_original_image
                [static_cast<std::size_t>(corner_index)]);
      }
      if (!complete) {
        continue;
      }
      eval_boards.push_back(eval);
      evaluated_board_ids.push_back(row.board_id);
      if (row.observation_source == "normal_detected" &&
          row.local_pose_refit_success &&
          std::isfinite(row.local_outer_rmse) &&
          row.local_outer_rmse < 3.0) {
        support_board_ids.push_back(row.board_id);
        for (std::size_t point_index = 0;
             point_index < eval.target_points_board.size(); ++point_index) {
          support_points_reference.push_back(
              T_reference_board * eval.target_points_board[point_index]);
          support_points_image.push_back(eval.image_points[point_index]);
        }
      }
    }

    FramePoseRepairDiagnosticRow repair;
    repair.frame_index = frame_aggregate.frame_index;
    repair.frame_label = frame_aggregate.frame_label;
    repair.candidate_board_count = static_cast<int>(support_board_ids.size());
    repair.candidate_point_count =
        static_cast<int>(support_points_reference.size());
    repair.evaluated_board_count = static_cast<int>(eval_boards.size());
    repair.support_board_ids = JoinIntsForDiagnostics(support_board_ids);
    repair.evaluated_board_ids = JoinIntsForDiagnostics(evaluated_board_ids);

    Eigen::Isometry3d T_camera_reference_refit = Eigen::Isometry3d::Identity();
    repair.refit_success = EstimatePoseFromReferencePointsForDiagnostics(
        camera, support_points_reference, support_points_image,
        &T_camera_reference_refit, &repair.refit_support_outer_rmse);
    double current_sq = 0.0;
    double repaired_sq = 0.0;
    int current_count = 0;
    int repaired_count = 0;
    double current_max = 0.0;
    double repaired_max = 0.0;
    for (const EvaluationBoardData& eval : eval_boards) {
      double board_current_sq = 0.0;
      double board_repaired_sq = 0.0;
      int board_current_count = 0;
      int board_repaired_count = 0;
      const Eigen::Isometry3d T_current_board =
          Eigen::Isometry3d(frame_state->T_camera_reference) *
          eval.T_reference_board;
      const Eigen::Isometry3d T_repaired_board =
          T_camera_reference_refit * eval.T_reference_board;
      for (std::size_t point_index = 0;
           point_index < eval.target_points_board.size(); ++point_index) {
        Eigen::Vector2d projected_current;
        if (camera.vsEuclideanToKeypoint(
                T_current_board * eval.target_points_board[point_index],
                &projected_current)) {
          const double sq =
              (projected_current - eval.image_points[point_index]).squaredNorm();
          current_sq += sq;
          board_current_sq += sq;
          ++current_count;
          ++board_current_count;
        }
        if (repair.refit_success) {
          Eigen::Vector2d projected_repaired;
          if (camera.vsEuclideanToKeypoint(
                  T_repaired_board * eval.target_points_board[point_index],
                  &projected_repaired)) {
            const double sq =
                (projected_repaired - eval.image_points[point_index]).squaredNorm();
            repaired_sq += sq;
            board_repaired_sq += sq;
            ++repaired_count;
            ++board_repaired_count;
          }
        }
      }
      const double current_board_rmse =
          RmseFromSquaredError(board_current_sq, board_current_count);
      const double repaired_board_rmse =
          RmseFromSquaredError(board_repaired_sq, board_repaired_count);
      if (std::isfinite(current_board_rmse)) {
        current_max = std::max(current_max, current_board_rmse);
      }
      if (std::isfinite(repaired_board_rmse)) {
        repaired_max = std::max(repaired_max, repaired_board_rmse);
      }
      if (std::isfinite(current_board_rmse) &&
          std::isfinite(repaired_board_rmse)) {
        if (repaired_board_rmse + 1.0 < current_board_rmse) {
          ++repair.improved_board_count;
        } else if (repaired_board_rmse > current_board_rmse + 1.0) {
          ++repair.worsened_board_count;
        }
      }
    }
    repair.current_global_mean_outer_rmse =
        RmseFromSquaredError(current_sq, current_count);
    repair.repaired_mean_outer_rmse =
        RmseFromSquaredError(repaired_sq, repaired_count);
    repair.current_global_max_outer_rmse = current_max;
    repair.repaired_max_outer_rmse = repaired_max;
    if (repair.refit_success) {
      const Eigen::Isometry3d delta =
          Eigen::Isometry3d(frame_state->T_camera_reference).inverse() *
          T_camera_reference_refit;
      repair.current_to_repaired_rotation_deg =
          RotationAngleDegForDiagnostics(delta.linear());
      repair.current_to_repaired_translation_norm =
          TranslationNormForDiagnostics(delta);
    }
    if (!repair.refit_success) {
      repair.diagnosis_label = "repair_pose_refit_failed";
    } else if (repair.candidate_board_count <= 0) {
      repair.diagnosis_label = "no_local_good_normal_support";
    } else if (std::isfinite(repair.repaired_mean_outer_rmse) &&
               repair.repaired_mean_outer_rmse + 2.0 <
                   repair.current_global_mean_outer_rmse) {
      repair.diagnosis_label = "frame_pose_repair_likely_helpful";
    } else if (std::isfinite(repair.repaired_mean_outer_rmse) &&
               repair.repaired_mean_outer_rmse >
                   repair.current_global_mean_outer_rmse + 2.0) {
      repair.diagnosis_label = "frame_pose_repair_worse";
    } else {
      repair.diagnosis_label = "frame_pose_repair_neutral";
    }
    repair_rows.push_back(repair);
  }

  std::ofstream repair_csv(
      (audit_dir / "frame_pose_repair_diagnostics.csv").string().c_str());
  repair_csv
      << "frame_index,frame_label,candidate_board_count,"
      << "candidate_point_count,evaluated_board_count,refit_success,"
      << "refit_support_outer_rmse,current_global_mean_outer_rmse,"
      << "repaired_mean_outer_rmse,current_global_max_outer_rmse,"
      << "repaired_max_outer_rmse,current_to_repaired_rotation_deg,"
      << "current_to_repaired_translation_norm,improved_board_count,"
      << "worsened_board_count,support_board_ids,evaluated_board_ids,"
      << "diagnosis_label\n";
  for (const FramePoseRepairDiagnosticRow& repair : repair_rows) {
    repair_csv << repair.frame_index << ","
               << CsvEscape(repair.frame_label) << ","
               << repair.candidate_board_count << ","
               << repair.candidate_point_count << ","
               << repair.evaluated_board_count << ","
               << (repair.refit_success ? 1 : 0) << ","
               << repair.refit_support_outer_rmse << ","
               << repair.current_global_mean_outer_rmse << ","
               << repair.repaired_mean_outer_rmse << ","
               << repair.current_global_max_outer_rmse << ","
               << repair.repaired_max_outer_rmse << ","
               << repair.current_to_repaired_rotation_deg << ","
               << repair.current_to_repaired_translation_norm << ","
               << repair.improved_board_count << ","
               << repair.worsened_board_count << ","
               << CsvEscape(repair.support_board_ids) << ","
               << CsvEscape(repair.evaluated_board_ids) << ","
               << repair.diagnosis_label << "\n";
  }

  int repair_helpful_count = 0;
  int repair_worse_count = 0;
  for (const FramePoseRepairDiagnosticRow& repair : repair_rows) {
    if (repair.diagnosis_label == "frame_pose_repair_likely_helpful") {
      ++repair_helpful_count;
    } else if (repair.diagnosis_label == "frame_pose_repair_worse") {
      ++repair_worse_count;
    }
  }
  std::ofstream repair_summary(
      (audit_dir / "frame_pose_repair_summary.txt").string().c_str());
  repair_summary << "repair_candidate_frame_count: " << repair_rows.size()
                 << "\n";
  repair_summary << "frame_pose_repair_likely_helpful_count: "
                 << repair_helpful_count << "\n";
  repair_summary << "frame_pose_repair_worse_count: "
                 << repair_worse_count << "\n";
  repair_summary
      << "support_policy: normal_detected observations with local_outer_rmse < "
      << "3 px only; rescued observations are evaluated but not used as support."
      << "\n";

  std::vector<GlobalSceneConsistencyObservationRow> sorted_rows = rows;
  std::sort(sorted_rows.begin(), sorted_rows.end(),
            [](const GlobalSceneConsistencyObservationRow& a,
               const GlobalSceneConsistencyObservationRow& b) {
              return a.global_minus_local_rmse > b.global_minus_local_rmse;
            });
  std::ofstream top_bad(
      (audit_dir / "global_scene_consistency_top_bad_observations.csv")
          .string()
          .c_str());
  top_bad
      << "rank,frame_index,frame_label,board_id,observation_source,"
      << "global_outer_rmse,local_outer_rmse,global_minus_local_rmse,"
      << "pose_delta_vs_global_rotation_deg,"
      << "pose_delta_vs_global_translation_norm,diagnosis_label,note\n";
  const std::size_t top_count = std::min<std::size_t>(50, sorted_rows.size());
  for (std::size_t index = 0; index < top_count; ++index) {
    const GlobalSceneConsistencyObservationRow& row = sorted_rows[index];
    top_bad << (index + 1) << ","
            << row.frame_index << ","
            << CsvEscape(row.frame_label) << ","
            << row.board_id << ","
            << row.observation_source << ","
            << row.global_outer_rmse << ","
            << row.local_outer_rmse << ","
            << row.global_minus_local_rmse << ","
            << row.pose_delta_vs_global_rotation_deg << ","
            << row.pose_delta_vs_global_translation_norm << ","
            << row.diagnosis_label << ","
            << CsvEscape(row.reject_reason_or_note) << "\n";
  }

  std::ofstream source_audit(
      (audit_dir / "global_scene_state_source_audit.txt").string().c_str());
  source_audit << "audit_stage: " << round_label << "\n";
  source_audit << "global_scene_state_source: final_stage5_bundle.scene_state"
               << "\n";
  source_audit << "global_scene_state_level: "
               << report.baseline_result.final_stage5_bundle.scene_state
                      .coarse_or_optimized_level
               << "\n";
  source_audit << "backend_input_scene_state_level: "
               << report.backend_problem_input.scene_state
                      .coarse_or_optimized_level
               << "\n";
  source_audit << "generation_state_source: "
               << "round2/round1 regeneration artifacts; rescued rows carry "
               << "local_patch_rescue_summary when available\n";
  source_audit << "selection_state_source: " << round_label << "_selection\n";
  source_audit << "backend_input_state_source: backend_problem_input\n";
  source_audit << "note: this audit is read-only and does not alter selection, "
               << "backend input, residual weights, or scene state.\n";

  std::ofstream convention_audit(
      (audit_dir / "global_scene_transform_convention_audit.txt")
          .string()
          .c_str());
  convention_audit
      << "expected_transform_convention: T_cam_board = "
      << "T_cam_reference * T_reference_board\n";
  convention_audit
      << "local_frame_pose_backprojection: T_cam_reference_from_board = "
      << "T_cam_board_local * inverse(T_reference_board)\n";
  convention_audit
      << "reference_board_id: " << scene_state.reference_board_id << "\n";
  convention_audit
      << "reference_board_T_reference_board_expected_identity: 1\n";
  convention_audit
      << "reference_board_global_bad_interpretation: if board1/reference local "
      << "pose is good but global pose is bad, prioritize frame "
      << "T_cam_reference or state-source mismatch over board structure.\n";

  std::ofstream summary(
      (audit_dir / "global_scene_consistency_summary.txt").string().c_str());
  summary << "observation_count: " << rows.size() << "\n";
  summary << "local_good_global_bad_count: "
          << local_good_global_bad_count << "\n";
  summary << "reference_board_local_good_global_bad_count: "
          << reference_board_bad_count << "\n";
  summary << "diagnosis_counts:\n";
  for (const auto& entry : diagnosis_counts) {
    summary << "  " << entry.first << ": " << entry.second << "\n";
  }
  summary << "\ninterpretation_rules:\n";
  summary << "  if multiple boards in one frame are local_good_global_bad, "
          << "suspect frame T_cam_reference.\n";
  summary << "  if one board is local_good_global_bad across many frames, "
          << "suspect board T_reference_board.\n";
  summary << "  if reference board is local_good_global_bad, strongly suspect "
          << "frame T_cam_reference because reference board pose is identity.\n";
  summary << "  if only rescued observations are bad while normal detections are "
          << "good, suspect rescued/global state mismatch.\n";
}

void WriteFrameBoardObservationFlowDiagnostics(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report,
    const std::vector<ati::FrozenRound2BaselineFrameSource>& all_frames_for_lookup,
    const ati::CameraModelRefitEvaluationResult* backend_training_evaluation,
    const ati::JointResidualEvaluationResult* backend_optimized_residual) {
  const bool use_round2 = report.baseline_result.round2_available;
  const ati::FrozenRoundArtifacts& artifacts =
      use_round2 ? report.baseline_result.round2 : report.baseline_result.round1;
  const std::string round_label = use_round2 ? "round2_final" : "round1_final";
  const bool strict_board_observation_acceptance =
      report.baseline_result.effective_options.strict_board_observation_acceptance;

  std::set<std::pair<int, int> > keys;
  for (const ati::InternalRegenerationFrameResult& frame :
       artifacts.regeneration_results) {
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      keys.insert(std::make_pair(frame.frame_index, measurement.board_id));
    }
  }
  for (const ati::JointMeasurementFrameResult& frame :
       artifacts.measurement_result.frames) {
    for (const ati::JointBoardObservation& board :
         frame.board_observations) {
      keys.insert(std::make_pair(frame.frame_index, board.board_id));
    }
  }
  for (const ati::JointBoardObservationSelectionDecision& decision :
       artifacts.selection_result.board_observation_decisions) {
    keys.insert(std::make_pair(decision.frame_index, decision.board_id));
  }
  for (const std::pair<int, int>& key :
       report.backend_problem_input.measurement_dataset
           .accepted_board_observation_keys) {
    keys.insert(key);
  }

  const std::map<std::pair<int, int>, std::pair<int, int> > final_counts =
      BuildFinalBackendPointCounts(report.backend_problem_input);
  const std::map<BackendTrainingResidualKey, BackendTrainingResidualInfo>
      backend_training_residuals =
          BuildBackendTrainingResidualMap(backend_training_evaluation);
  const std::map<BackendTrainingResidualKey, BackendTrainingResidualInfo>
      backend_optimized_residuals =
          backend_optimized_residual != nullptr
              ? BuildBackendOptimizedResidualMap(*backend_optimized_residual)
              : std::map<BackendTrainingResidualKey,
                         BackendTrainingResidualInfo>();
  std::map<int, std::string> frame_labels;
  for (const ati::FrozenRound2BaselineFrameSource& frame :
       report.baseline_result.frame_sources) {
    frame_labels[frame.frame_index] = frame.frame_label;
  }
  for (const ati::InternalRegenerationFrameResult& frame :
       artifacts.regeneration_results) {
    frame_labels[frame.frame_index] = frame.frame_label;
  }

  std::vector<FrameBoardObservationFlowRow> rows;
  rows.reserve(keys.size());
  std::map<std::pair<int, int>, FrameBoardVisualGeometry> geometries;
  for (const std::pair<int, int>& key : keys) {
    const int frame_index = key.first;
    const int board_id = key.second;
    FrameBoardObservationFlowRow row;
    row.round_label = round_label;
    row.frame_index = frame_index;
    row.board_id = board_id;
    const auto frame_label_it = frame_labels.find(frame_index);
    if (frame_label_it != frame_labels.end()) {
      row.frame_label = frame_label_it->second;
    }

    const ati::RegeneratedBoardMeasurement* regenerated =
        FindRegeneratedBoardMeasurement(artifacts.regeneration_results,
                                        frame_index, board_id);
    if (regenerated != nullptr) {
      row.internal_result_available = true;
      row.frame_bootstrap_initialized = regenerated->frame_bootstrap_initialized;
      row.board_bootstrap_initialized = regenerated->board_bootstrap_initialized;
      row.pose_prior_used = regenerated->pose_prior_used;
      const ati::ApriltagInternalDetectionResult& detection =
          regenerated->detection;
      row.outer_detected = detection.outer_detection.success;
      row.outer_failure_reason = detection.outer_detection.failure_reason_text;
      row.internal_success = detection.success;
      row.internal_failure_reason = detection.failure_reason;
      row.internal_camera_source = detection.internal_camera_source;
      row.attempted_internal_corner_count =
          detection.runtime_breakdown.attempted_internal_corner_count;
      row.valid_internal_corner_count =
          detection.valid_internal_corner_count;
      row.valid_internal_ratio =
          SafeRatio(row.valid_internal_corner_count,
                    row.attempted_internal_corner_count);
    }

    const ati::JointBoardObservation* board_observation =
        FindJointBoardObservation(artifacts.measurement_result,
                                  frame_index, board_id);
    std::map<std::string, int> point_rejection_reasons;
    if (board_observation != nullptr) {
      row.measurement_built = true;
      row.pre_selection_outer_point_count =
          board_observation->outer_point_count;
      row.pre_selection_internal_point_count =
          board_observation->internal_point_count;
      row.pre_selection_solver_ready = board_observation->used_in_solver;
      for (const ati::JointPointObservation& point :
           board_observation->points) {
        if (point.rejection_reason_code == ati::JointRejectionReasonCode::None) {
          continue;
        }
        const std::string reason = ati::ToString(point.rejection_reason_code);
        ++point_rejection_reasons[reason];
        if (point.rejection_reason_code ==
            ati::JointRejectionReasonCode::InternalRegenerationFailed) {
          ++row.rejected_internal_regeneration_failed_count;
        } else if (point.rejection_reason_code ==
                   ati::JointRejectionReasonCode::InternalPointInvalid) {
          ++row.rejected_internal_point_invalid_count;
        } else if (point.rejection_reason_code ==
                   ati::JointRejectionReasonCode::InternalPointReprojectionOutlier) {
          ++row.rejected_internal_point_outlier_count;
        } else {
          ++row.rejected_other_count;
        }
      }
      row.point_rejection_reasons = JoinReasonCounts(point_rejection_reasons);
    }

    row.strict_drop_candidate =
        strict_board_observation_acceptance && row.outer_detected &&
        (!row.internal_result_available || !row.internal_success);

    const ati::JointBoardObservationSelectionDecision* decision =
        FindSelectionDecision(artifacts.selection_result, frame_index, board_id);
    if (decision != nullptr) {
      row.selection_decision_available = true;
      row.selection_accepted = decision->accepted;
      row.selection_reason_code = ati::ToString(decision->reason_code);
      row.selection_reason_detail = decision->reason_detail;
      row.selection_rmse = decision->rmse;
      if (row.frame_label.empty()) {
        row.frame_label = decision->frame_label;
      }
    }

    const auto final_count_it = final_counts.find(key);
    if (final_count_it != final_counts.end()) {
      row.final_outer_point_count = final_count_it->second.first;
      row.final_internal_point_count = final_count_it->second.second;
      row.final_used_in_backend =
          row.final_outer_point_count + row.final_internal_point_count > 0;
    }
    row.final_status = DeriveFrameBoardFlowStatus(row);

    geometries[key] = BuildVisualGeometry(regenerated, board_observation);
    rows.push_back(row);
  }

  std::map<int, std::vector<FrameBoardObservationFlowRow> > strict_rows_by_frame;
  for (const FrameBoardObservationFlowRow& row : rows) {
    if (row.final_status == "dropped_by_strict_internal_failure") {
      strict_rows_by_frame[row.frame_index].push_back(row);
    }
  }

  std::ofstream strict_seed_csv(
      (output_dir / "strict_internal_seed_points.csv").string().c_str());
  strict_seed_csv
      << "round,frame_index,frame_label,board_id,point_id,corner_type,"
      << "predicted_u,predicted_v,sphere_seed_u,sphere_seed_v,"
      << "refined_u,refined_v,predicted_inside,sphere_seed_inside,"
      << "refined_inside,debug_valid,image_evidence_valid,"
      << "forced_prediction_seed,"
      << "sphere_seed_quality,image_final_quality,template_quality,"
      << "gradient_quality,final_quality,q_refine,"
      << "predicted_to_seed_displacement,seed_to_refined_displacement,"
      << "predicted_to_refined_displacement,subpix_displacement_limit,"
      << "inferred_invalid_reason\n";
  std::ofstream strict_seed_status_csv(
      (output_dir / "strict_internal_seed_status.csv").string().c_str());
  strict_seed_status_csv
      << "round,frame_index,frame_label,board_id,point_id,corner_type,"
      << "seed_success,refined_inside,final_valid,image_evidence_valid,"
      << "forced_prediction_seed,"
      << "sphere_seed_quality,image_final_quality,final_quality,"
      << "seed_to_refined_displacement,predicted_to_refined_displacement,"
      << "status\n";
  std::map<std::string, int> strict_seed_status_counts;
  for (const auto& frame_entry : strict_rows_by_frame) {
    for (const FrameBoardObservationFlowRow& row : frame_entry.second) {
      const ati::RegeneratedBoardMeasurement* regenerated =
          FindRegeneratedBoardMeasurement(artifacts.regeneration_results,
                                          row.frame_index, row.board_id);
      if (regenerated == nullptr) {
        continue;
      }
      const ati::ApriltagInternalDetectionResult& detection =
          regenerated->detection;
      const cv::Size image_size = detection.image_size;
      for (const ati::InternalCornerDebugInfo& debug :
           detection.internal_corner_debug) {
        const bool predicted_inside =
            IsPointNearImage(debug.predicted_image, image_size, 0.0f);
        const bool sphere_seed_inside =
            IsPointNearImage(debug.sphere_seed_image, image_size, 0.0f);
        const bool refined_inside =
            IsPointNearImage(debug.refined_image, image_size, 0.0f);
        std::string inferred_reason = "valid";
        if (!debug.valid) {
          if (!refined_inside) {
            inferred_reason = "refined_outside_image";
          } else if (!sphere_seed_inside &&
                     debug.sphere_seed_quality <= 0.0) {
            inferred_reason = "sphere_seed_missing_or_invalid";
          } else if (!debug.image_evidence_valid) {
            inferred_reason = "image_evidence_below_threshold";
          } else {
            inferred_reason = "valid_flag_false_unknown";
          }
        }
        const bool seed_success =
            sphere_seed_inside && debug.sphere_seed_quality > 0.0;
        const std::string seed_status =
            debug.valid
                ? "final_valid"
                : (!seed_success
                       ? "seed_failed"
                       : (!refined_inside
                              ? "refined_outside_image"
                              : (!debug.image_evidence_valid
                                     ? "image_evidence_below_threshold"
                                     : "valid_flag_false_unknown")));
        ++strict_seed_status_counts[seed_status];
        strict_seed_csv
            << round_label << ","
            << row.frame_index << ","
            << CsvEscape(row.frame_label) << ","
            << row.board_id << ","
            << debug.point_id << ","
            << ati::ToString(debug.corner_type) << ","
            << debug.predicted_image.x << ","
            << debug.predicted_image.y << ","
            << debug.sphere_seed_image.x << ","
            << debug.sphere_seed_image.y << ","
            << debug.refined_image.x << ","
            << debug.refined_image.y << ","
            << (predicted_inside ? 1 : 0) << ","
            << (sphere_seed_inside ? 1 : 0) << ","
            << (refined_inside ? 1 : 0) << ","
            << (debug.valid ? 1 : 0) << ","
            << (debug.image_evidence_valid ? 1 : 0) << ","
            << (debug.forced_prediction_seed ? 1 : 0) << ","
            << debug.sphere_seed_quality << ","
            << debug.image_final_quality << ","
            << debug.template_quality << ","
            << debug.gradient_quality << ","
            << debug.final_quality << ","
            << debug.q_refine << ","
            << debug.predicted_to_seed_displacement << ","
            << debug.seed_to_refined_displacement << ","
            << debug.predicted_to_refined_displacement << ","
            << debug.subpix_displacement_limit << ","
            << CsvEscape(inferred_reason) << "\n";
        strict_seed_status_csv
            << round_label << ","
            << row.frame_index << ","
            << CsvEscape(row.frame_label) << ","
            << row.board_id << ","
            << debug.point_id << ","
            << ati::ToString(debug.corner_type) << ","
            << (seed_success ? 1 : 0) << ","
            << (refined_inside ? 1 : 0) << ","
            << (debug.valid ? 1 : 0) << ","
            << (debug.image_evidence_valid ? 1 : 0) << ","
            << (debug.forced_prediction_seed ? 1 : 0) << ","
            << debug.sphere_seed_quality << ","
            << debug.image_final_quality << ","
            << debug.final_quality << ","
            << debug.seed_to_refined_displacement << ","
            << debug.predicted_to_refined_displacement << ","
            << CsvEscape(seed_status) << "\n";
      }
    }
  }

  std::ofstream csv(
      (output_dir / "frame_board_observation_flow.csv").string().c_str());
  csv << "round,frame_index,frame_label,board_id,"
      << "outer_detected,outer_failure_reason,"
      << "internal_result_available,internal_success,internal_failure_reason,"
      << "internal_camera_source,"
      << "frame_bootstrap_initialized,board_bootstrap_initialized,pose_prior_used,"
      << "attempted_internal_corner_count,valid_internal_corner_count,"
      << "valid_internal_ratio,measurement_built,pre_selection_solver_ready,"
      << "pre_selection_outer_point_count,pre_selection_internal_point_count,"
      << "rejected_internal_regeneration_failed_count,"
      << "rejected_internal_point_invalid_count,"
      << "rejected_internal_point_outlier_count,rejected_other_count,"
      << "point_rejection_reasons,strict_drop_candidate,"
      << "selection_decision_available,selection_accepted,"
      << "selection_reason_code,selection_reason_detail,selection_rmse,"
      << "final_used_in_backend,final_outer_point_count,"
      << "final_internal_point_count,final_status\n";
  for (const FrameBoardObservationFlowRow& row : rows) {
    csv << row.round_label << ","
        << row.frame_index << ","
        << CsvEscape(row.frame_label) << ","
        << row.board_id << ","
        << (row.outer_detected ? 1 : 0) << ","
        << CsvEscape(row.outer_failure_reason) << ","
        << (row.internal_result_available ? 1 : 0) << ","
        << (row.internal_success ? 1 : 0) << ","
        << CsvEscape(row.internal_failure_reason) << ","
        << CsvEscape(row.internal_camera_source) << ","
        << (row.frame_bootstrap_initialized ? 1 : 0) << ","
        << (row.board_bootstrap_initialized ? 1 : 0) << ","
        << (row.pose_prior_used ? 1 : 0) << ","
        << row.attempted_internal_corner_count << ","
        << row.valid_internal_corner_count << ","
        << row.valid_internal_ratio << ","
        << (row.measurement_built ? 1 : 0) << ","
        << (row.pre_selection_solver_ready ? 1 : 0) << ","
        << row.pre_selection_outer_point_count << ","
        << row.pre_selection_internal_point_count << ","
        << row.rejected_internal_regeneration_failed_count << ","
        << row.rejected_internal_point_invalid_count << ","
        << row.rejected_internal_point_outlier_count << ","
        << row.rejected_other_count << ","
        << CsvEscape(row.point_rejection_reasons) << ","
        << (row.strict_drop_candidate ? 1 : 0) << ","
        << (row.selection_decision_available ? 1 : 0) << ","
        << (row.selection_accepted ? 1 : 0) << ","
        << CsvEscape(row.selection_reason_code) << ","
        << CsvEscape(row.selection_reason_detail) << ","
        << row.selection_rmse << ","
        << (row.final_used_in_backend ? 1 : 0) << ","
        << row.final_outer_point_count << ","
        << row.final_internal_point_count << ","
        << CsvEscape(row.final_status) << "\n";
  }

  std::map<std::string, int> selection_reason_counts;
  std::map<std::string, int> accepted_selection_reason_counts;
  std::map<int, int> accepted_boards;
  std::map<int, int> rejected_boards;
  int selection_decision_count = 0;
  int selection_accepted_count = 0;
  int selection_rejected_count = 0;
  for (const FrameBoardObservationFlowRow& row : rows) {
    if (!row.selection_decision_available) {
      continue;
    }
    ++selection_decision_count;
    ++selection_reason_counts[row.selection_reason_code];
    if (row.selection_accepted) {
      ++selection_accepted_count;
      ++accepted_selection_reason_counts[row.selection_reason_code];
      ++accepted_boards[row.board_id];
    } else {
      ++selection_rejected_count;
      ++rejected_boards[row.board_id];
    }
  }

  std::ofstream selection_summary(
      (output_dir / "kalibr_style_frame_board_selection_summary.txt")
          .string()
          .c_str());
  selection_summary << "round: " << round_label << "\n";
  selection_summary << "selection_mode: "
                    << ati::ToString(report.baseline_result.effective_options
                                         .selection_mode)
                    << "\n";
  selection_summary << "selection_decision_count: "
                    << selection_decision_count << "\n";
  selection_summary << "selection_accepted_count: "
                    << selection_accepted_count << "\n";
  selection_summary << "selection_rejected_count: "
                    << selection_rejected_count << "\n";
  selection_summary << "selection_residual_sanity_factor: "
                    << report.baseline_result.effective_options
                           .selection_residual_sanity_factor
                    << "\n";
  selection_summary << "selection_max_board_observation_rmse: "
                    << report.baseline_result.effective_options
                           .selection_max_board_observation_rmse
                    << "\n";
  selection_summary << "kalibr_style_outlier_sigma: "
                    << report.baseline_result.effective_options
                           .selection_kalibr_style_outlier_sigma
                    << "\n";
  selection_summary << "kalibr_style_min_abs_threshold_px: "
                    << report.baseline_result.effective_options
                           .selection_kalibr_style_min_abs_threshold_px
                    << "\n";
  selection_summary << "kalibr_style_min_views_before_filter: "
                    << report.baseline_result.effective_options
                           .selection_kalibr_style_min_views_before_filter
                    << "\n";
  selection_summary << "\nselection_reason_counts:\n";
  for (const auto& entry : selection_reason_counts) {
    selection_summary << "  " << entry.first << ": " << entry.second << "\n";
  }
  selection_summary << "\naccepted_reason_counts:\n";
  for (const auto& entry : accepted_selection_reason_counts) {
    selection_summary << "  " << entry.first << ": " << entry.second << "\n";
  }
  selection_summary << "\naccepted_by_board:\n";
  for (const auto& entry : accepted_boards) {
    selection_summary << "  board " << entry.first << ": " << entry.second
                      << "\n";
  }
  selection_summary << "\nrejected_by_board:\n";
  for (const auto& entry : rejected_boards) {
    selection_summary << "  board " << entry.first << ": " << entry.second
                      << "\n";
  }

  WriteTrialBackendFrameBoardSelectionDiagnostics(output_dir, report);

  std::map<int, BackendUsedFrameSummaryRow> used_frame_summaries;
  for (const FrameBoardObservationFlowRow& row : rows) {
    if (!row.final_used_in_backend) {
      continue;
    }
    BackendUsedFrameSummaryRow& frame_summary =
        used_frame_summaries[row.frame_index];
    frame_summary.frame_index = row.frame_index;
    if (frame_summary.frame_label.empty()) {
      frame_summary.frame_label = row.frame_label;
    }
    frame_summary.board_ids.push_back(row.board_id);
    ++frame_summary.board_observation_count;
    frame_summary.outer_point_count += row.final_outer_point_count;
    frame_summary.internal_point_count += row.final_internal_point_count;
    frame_summary.total_point_count +=
        row.final_outer_point_count + row.final_internal_point_count;
  }
  std::ofstream used_frames_csv(
      (output_dir / "backend_used_frames.csv").string().c_str());
  used_frames_csv << "frame_index,frame_label,used_board_ids,"
                  << "used_board_observation_count,outer_point_count,"
                  << "internal_point_count,total_point_count\n";
  for (auto& entry : used_frame_summaries) {
    BackendUsedFrameSummaryRow& frame_summary = entry.second;
    std::sort(frame_summary.board_ids.begin(), frame_summary.board_ids.end());
    used_frames_csv << frame_summary.frame_index << ","
                    << CsvEscape(frame_summary.frame_label) << ","
                    << CsvEscape(JoinInts(frame_summary.board_ids)) << ","
                    << frame_summary.board_observation_count << ","
                    << frame_summary.outer_point_count << ","
                    << frame_summary.internal_point_count << ","
                    << frame_summary.total_point_count << "\n";
  }
  std::ofstream used_frames_txt(
      (output_dir / "backend_used_frames.txt").string().c_str());
  used_frames_txt << "backend_used_frame_count: "
                  << used_frame_summaries.size() << "\n";
  used_frames_txt << "format: frame_index frame_label boards "
                     "outer_points internal_points total_points\n";
  for (const auto& entry : used_frame_summaries) {
    const BackendUsedFrameSummaryRow& frame_summary = entry.second;
    used_frames_txt << frame_summary.frame_index << " "
                    << frame_summary.frame_label << " boards="
                    << JoinInts(frame_summary.board_ids)
                    << " outer=" << frame_summary.outer_point_count
                    << " internal=" << frame_summary.internal_point_count
                    << " total=" << frame_summary.total_point_count << "\n";
  }

  std::map<std::string, int> status_counts;
  std::map<std::string, int> non_strict_not_used_counts;
  std::map<std::string, int> internal_failure_reason_counts;
  std::map<std::string, int> internal_camera_source_counts;
  std::map<std::string, int> point_rejection_reason_counts;
  int final_used_count = 0;
  int not_used_count = 0;
  int strict_drop_count = 0;
  int partial_internal_used_count = 0;
  for (const FrameBoardObservationFlowRow& row : rows) {
    ++status_counts[row.final_status];
    if (row.final_used_in_backend) {
      ++final_used_count;
    } else {
      ++not_used_count;
      if (row.final_status == "dropped_by_strict_internal_failure") {
        ++strict_drop_count;
      } else {
        ++non_strict_not_used_counts[row.final_status];
      }
    }
    if (row.final_status == "used_partial_internal_points") {
      ++partial_internal_used_count;
    }
    if (row.outer_detected && !row.internal_success) {
      const std::string reason =
          row.internal_failure_reason.empty() ? "empty_failure_reason"
                                              : row.internal_failure_reason;
      ++internal_failure_reason_counts[reason];
    }
    if (row.internal_result_available) {
      const std::string source =
          row.internal_camera_source.empty() ? "empty_camera_source"
                                             : row.internal_camera_source;
      ++internal_camera_source_counts[source];
    }
    if (!row.point_rejection_reasons.empty()) {
      std::stringstream stream(row.point_rejection_reasons);
      std::string token;
      while (std::getline(stream, token, ';')) {
        const std::size_t eq = token.find('=');
        if (eq == std::string::npos) {
          continue;
        }
        const std::string reason = token.substr(0, eq);
        const int count = std::atoi(token.substr(eq + 1).c_str());
        point_rejection_reason_counts[reason] += count;
      }
    }
  }

  std::ofstream summary(
      (output_dir / "frame_board_observation_flow_summary.txt").string().c_str());
  summary << "round: " << round_label << "\n";
  summary << "strict_board_observation_acceptance: "
          << (strict_board_observation_acceptance ? 1 : 0) << "\n";
  summary << "failed_board_drop_policy: "
          << (strict_board_observation_acceptance
                  ? "drop_entire_board_observation"
                  : "keep_outer_when_internal_failed")
          << "\n";
  summary << "total_frame_board_rows: " << rows.size() << "\n";
  summary << "final_used_count: " << final_used_count << "\n";
  summary << "not_used_count: " << not_used_count << "\n";
  summary << "strict_internal_failure_drop_count: " << strict_drop_count << "\n";
  summary << "partial_internal_used_count: " << partial_internal_used_count << "\n";
  summary << "\nstatus_counts:\n";
  for (const auto& entry : status_counts) {
    summary << "  " << entry.first << ": " << entry.second << "\n";
  }
  summary << "\nnon_strict_not_used_counts:\n";
  for (const auto& entry : non_strict_not_used_counts) {
    summary << "  " << entry.first << ": " << entry.second << "\n";
  }
  summary << "\ninternal_failure_reason_counts_outer_detected:\n";
  for (const auto& entry : internal_failure_reason_counts) {
    summary << "  " << entry.first << ": " << entry.second << "\n";
  }
  summary << "\ninternal_camera_source_counts:\n";
  for (const auto& entry : internal_camera_source_counts) {
    summary << "  " << entry.first << ": " << entry.second << "\n";
  }
  summary << "\npoint_rejection_reason_counts:\n";
  for (const auto& entry : point_rejection_reason_counts) {
    summary << "  " << entry.first << ": " << entry.second << "\n";
  }
  summary << "\nstrict_internal_seed_status_counts:\n";
  for (const auto& entry : strict_seed_status_counts) {
    summary << "  " << entry.first << ": " << entry.second << "\n";
  }
  if (!non_strict_not_used_counts.empty()) {
    const auto largest = std::max_element(
        non_strict_not_used_counts.begin(), non_strict_not_used_counts.end(),
        [](const std::pair<std::string, int>& lhs,
           const std::pair<std::string, int>& rhs) {
          return lhs.second < rhs.second;
        });
    summary << "\nlargest_non_strict_not_used_reason: "
            << largest->first << "\n";
    summary << "largest_non_strict_not_used_count: "
            << largest->second << "\n";
  }

  const fs::path overlay_dir = output_dir / "frame_board_observation_flow_overlays";
  const fs::path strict_seed_overlay_dir =
      output_dir / "strict_internal_seed_overlays";
  const fs::path strict_refined_overlay_dir =
      output_dir / "strict_internal_refined_corner_overlays";
  const fs::path internal_filter_overlay_dir =
      output_dir / "internal_point_filter_overlays";
  const fs::path backend_input_overlay_dir =
      output_dir / "backend_input_visualizations";
  const fs::path trial_accepted_overlay_dir =
      backend_input_overlay_dir / "trial_accepted_frame_boards";
  const fs::path close_edge_soft_overlay_dir =
      output_dir / "close_edge_soft_candidate_visualizations";
  std::map<int, std::string> image_paths;
  for (const ati::FrozenRound2BaselineFrameSource& frame :
       all_frames_for_lookup) {
    if (!frame.image_path.empty()) {
      image_paths[frame.frame_index] = frame.image_path;
    }
  }
  std::map<int, std::vector<FrameBoardObservationFlowRow> > rows_by_frame;
  for (const FrameBoardObservationFlowRow& row : rows) {
    rows_by_frame[row.frame_index].push_back(row);
  }

  std::map<int, std::vector<ati::JointPointObservation> >
      backend_used_points_by_frame;
  std::map<std::pair<int, int>, const ati::RegeneratedBoardMeasurement*>
      regenerated_by_frame_board;
  for (const ati::InternalRegenerationFrameResult& frame :
       artifacts.regeneration_results) {
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      regenerated_by_frame_board[std::make_pair(frame.frame_index,
                                                measurement.board_id)] =
          &measurement;
    }
  }

  std::ofstream outer_subpix_debug_csv(
      (output_dir / "outer_subpix_window_debug.csv").string().c_str());
  outer_subpix_debug_csv
      << "round,frame_index,frame_label,board_id,corner_index,"
      << "outer_detected,subpix_applied,configured_outer_subpix_scale,"
      << "configured_outer_subpix_window_scale,"
      << "configured_outer_subpix_window_radius,"
      << "configured_outer_subpix_window_min,"
      << "configured_outer_subpix_window_max,local_scale,"
      << "corner_marker_width,verification_roi_radius,"
      << "raw_subpix_window_radius,pre_boost_subpix_window_radius,"
      << "boosted_raw_subpix_window_radius,subpix_window_clamp_limit,"
      << "subpix_window_clamped,final_subpix_window_radius,"
      << "subpix_unstable_rollback_detected,"
      << "subpix_unstable_rollback_iteration,"
      << "subpix_unstable_rollback_max_displacement,"
      << "close_edge_subpix_boost_applied,"
      << "close_edge_subpix_area_ratio,"
      << "close_edge_subpix_max_polar_deg,"
      << "close_edge_subpix_multiplier,coarse_u,coarse_v,"
      << "subpix_u,subpix_v,failure_reason\n";
  for (const ati::InternalRegenerationFrameResult& frame :
       artifacts.regeneration_results) {
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      const bool outer_detected = measurement.detection.outer_detection.success;
      const std::array<ati::OuterCornerVerificationDebugInfo, 4>& debug_values =
          measurement.detection.outer_detection.corner_verification_debug;
      for (const ati::OuterCornerVerificationDebugInfo& debug : debug_values) {
        if (debug.corner_index < 0) {
          continue;
        }
        outer_subpix_debug_csv
            << round_label << ","
            << frame.frame_index << ","
            << CsvEscape(frame.frame_label) << ","
            << measurement.board_id << ","
            << debug.corner_index << ","
            << (outer_detected ? 1 : 0) << ","
            << (debug.subpix_applied ? 1 : 0) << ","
            << debug.configured_outer_subpix_scale << ","
            << debug.configured_outer_subpix_window_scale << ","
            << debug.configured_outer_subpix_window_radius << ","
            << debug.configured_outer_subpix_window_min << ","
            << debug.configured_outer_subpix_window_max << ","
            << debug.local_scale << ","
            << debug.corner_marker_width << ","
            << debug.verification_roi_radius << ","
            << debug.raw_subpix_window_radius << ","
            << debug.pre_boost_subpix_window_radius << ","
            << debug.boosted_raw_subpix_window_radius << ","
            << debug.subpix_window_clamp_limit << ","
            << (debug.subpix_window_clamped ? 1 : 0) << ","
            << debug.subpix_window_radius << ","
            << (debug.subpix_unstable_rollback_detected ? 1 : 0) << ","
            << debug.subpix_unstable_rollback_iteration << ","
            << debug.subpix_unstable_rollback_max_displacement << ","
            << (debug.close_edge_subpix_boost_applied ? 1 : 0) << ","
            << debug.close_edge_subpix_area_ratio << ","
            << debug.close_edge_subpix_max_polar_deg << ","
            << debug.close_edge_subpix_multiplier << ","
            << debug.coarse_corner.x << ","
            << debug.coarse_corner.y << ","
            << debug.subpix_corner.x << ","
            << debug.subpix_corner.y << ","
            << CsvEscape(debug.failure_reason) << "\n";
      }
    }
  }

  std::map<std::pair<int, int>, std::vector<ati::JointPointObservation> >
      backend_used_points_by_frame_board;
  std::map<std::pair<int, int>, std::vector<ati::JointPointObservation> >
      measurement_points_by_frame_board;
  for (const ati::JointMeasurementFrameResult& frame :
       artifacts.measurement_result.frames) {
    for (const ati::JointBoardObservation& board :
         frame.board_observations) {
      measurement_points_by_frame_board[std::make_pair(frame.frame_index,
                                                       board.board_id)] =
          board.points;
    }
  }
  int backend_used_overlay_point_count = 0;
  int backend_used_overlay_outer_point_count = 0;
  int backend_used_overlay_internal_point_count = 0;
  for (const ati::JointPointObservation& point :
       report.backend_problem_input.measurement_dataset.solver_observations) {
    if (!point.used_in_solver) {
      continue;
    }
    backend_used_points_by_frame[point.frame_index].push_back(point);
    backend_used_points_by_frame_board[std::make_pair(point.frame_index,
                                                      point.board_id)]
        .push_back(point);
    ++backend_used_overlay_point_count;
    if (point.point_type == ati::JointPointType::Outer) {
      ++backend_used_overlay_outer_point_count;
    } else {
      ++backend_used_overlay_internal_point_count;
    }
  }
  std::map<std::pair<int, int>, BackendInputBoardOverlaySource>
      backend_input_source_by_frame_board;
  for (const ati::TrialBackendFrameBoardObservationDecision& decision :
       report.trial_backend_selection_result.decisions) {
    BackendInputBoardOverlaySource& source =
        backend_input_source_by_frame_board[std::make_pair(
            decision.frame_index, decision.board_id)];
    source.has_decision = true;
    source.baseline_seed = source.baseline_seed || decision.baseline_seed;
    source.incremental_trial_accepted =
        source.incremental_trial_accepted ||
        IsIncrementalTrialAcceptedReason(decision.reason);
    source.frame_cohesion_accepted =
        source.frame_cohesion_accepted ||
        decision.frame_cohesion_accepted ||
        decision.reason == "accepted_frame_cohesion_trial";
    source.close_distance_frame_admission_accepted =
        source.close_distance_frame_admission_accepted ||
        decision.close_distance_frame_admission_accepted ||
        decision.reason ==
            "accepted_close_distance_frame_admission_trial";
    source.soft_accepted = source.soft_accepted || decision.soft_accepted;
    if (!decision.reason.empty() &&
        (source.reason.empty() || source.baseline_seed ||
         source.incremental_trial_accepted ||
         source.frame_cohesion_accepted ||
         source.close_distance_frame_admission_accepted ||
         source.soft_accepted)) {
      source.reason = decision.reason;
    }
  }

  int backend_used_overlay_count = 0;
  if (!backend_used_points_by_frame.empty()) {
    EnsureDirectoryExists(backend_input_overlay_dir);
  }
  std::ofstream backend_input_overlay_source_csv;
  if (!backend_used_points_by_frame.empty()) {
    backend_input_overlay_source_csv.open(
        (backend_input_overlay_dir /
         "backend_input_overlay_source_summary.csv")
            .string()
            .c_str());
    backend_input_overlay_source_csv
        << "frame_index,frame_label,image_file,backend_board_count,"
        << "seed_board_count,incremental_trial_board_count,"
        << "frame_cohesion_board_count,"
        << "close_distance_frame_admission_board_count,"
        << "soft_trial_board_count,"
        << "unknown_board_count\n";
  }
  for (const auto& entry : backend_used_points_by_frame) {
    const auto path_it = image_paths.find(entry.first);
    if (path_it == image_paths.end()) {
      continue;
    }
    const cv::Mat image = cv::imread(path_it->second, cv::IMREAD_UNCHANGED);
    std::string frame_label;
    const auto label_it = frame_labels.find(entry.first);
    if (label_it != frame_labels.end()) {
      frame_label = label_it->second;
    }
    std::map<int, BackendInputBoardOverlaySource> source_by_board;
    std::set<int> backend_board_ids;
    for (const ati::JointPointObservation& point : entry.second) {
      if (point.used_in_solver) {
        backend_board_ids.insert(point.board_id);
      }
    }
    int seed_board_count = 0;
    int incremental_trial_board_count = 0;
    int frame_cohesion_board_count = 0;
    int close_distance_frame_admission_board_count = 0;
    int soft_trial_board_count = 0;
    int unknown_board_count = 0;
    for (int board_id : backend_board_ids) {
      BackendInputBoardOverlaySource source;
      const auto source_it = backend_input_source_by_frame_board.find(
          std::make_pair(entry.first, board_id));
      if (source_it != backend_input_source_by_frame_board.end()) {
        source = source_it->second;
      }
      source_by_board[board_id] = source;
      if (source.baseline_seed) {
        ++seed_board_count;
      } else if (source.frame_cohesion_accepted) {
        ++frame_cohesion_board_count;
      } else if (source.close_distance_frame_admission_accepted) {
        ++close_distance_frame_admission_board_count;
      } else if (source.soft_accepted) {
        ++soft_trial_board_count;
      } else if (source.incremental_trial_accepted) {
        ++incremental_trial_board_count;
      } else {
        ++unknown_board_count;
      }
    }
    std::ostringstream filename;
    filename << "frame_" << entry.first << "_"
             << SanitizeFilenameComponent(frame_label)
             << "_backend_used.png";
    const std::string image_file = filename.str();
    cv::Mat overlay;
    DrawBackendUsedObservationOverlay(image, entry.first, entry.second,
                                      source_by_board,
                                      backend_optimized_residuals, &overlay);
    if (overlay.empty()) {
      continue;
    }
    cv::imwrite((backend_input_overlay_dir / filename.str()).string(),
                overlay);
    if (backend_input_overlay_source_csv) {
      backend_input_overlay_source_csv
          << entry.first << ","
          << CsvEscape(frame_label) << ","
          << CsvEscape(image_file) << ","
          << backend_board_ids.size() << ","
          << seed_board_count << ","
          << incremental_trial_board_count << ","
          << frame_cohesion_board_count << ","
          << close_distance_frame_admission_board_count << ","
          << soft_trial_board_count << ","
          << unknown_board_count << "\n";
    }
    ++backend_used_overlay_count;
  }

  int close_edge_soft_overlay_count = 0;
  int close_edge_soft_accepted_overlay_count = 0;
  int close_edge_soft_rejected_overlay_count = 0;
  bool has_close_edge_soft_cases = false;
  for (const ati::TrialBackendFrameBoardObservationDecision& decision :
       report.trial_backend_selection_result.decisions) {
    if (decision.soft_candidate || decision.soft_attempted ||
        decision.soft_accepted) {
      has_close_edge_soft_cases = true;
      break;
    }
  }

  int trial_accepted_overlay_count = 0;
  std::ofstream trial_accepted_overlay_csv(
      (backend_input_overlay_dir / "trial_accepted_frame_board_list.csv")
          .string()
          .c_str());
  trial_accepted_overlay_csv
      << "frame_index,frame_label,board_id,reason,baseline_seed,"
      << "frame_cohesion_candidate,frame_cohesion_accepted,"
      << "close_distance_frame_admission_candidate,"
      << "close_distance_frame_admission_accepted,"
      << "close_distance_candidate,close_distance_score_bonus,"
      << "candidate_score,coverage_gain,global_rmse_delta,"
      << "outer_rmse_delta,internal_rmse_delta,visualized_point_count,"
      << "outer_point_count,internal_point_count,image_file\n";
  bool trial_accepted_overlay_dir_created = false;
  for (const ati::TrialBackendFrameBoardObservationDecision& decision :
       report.trial_backend_selection_result.decisions) {
    const bool trial_accepted =
        decision.reason == "accepted_incremental_trial" ||
        decision.reason == "accepted_frame_cohesion_trial" ||
        decision.reason ==
            "accepted_close_distance_frame_admission_trial";
    if (!trial_accepted) {
      continue;
    }
    const std::pair<int, int> key =
        std::make_pair(decision.frame_index, decision.board_id);
    const auto backend_points_it = backend_used_points_by_frame_board.find(key);
    if (backend_points_it == backend_used_points_by_frame_board.end()) {
      continue;
    }
    const std::vector<ati::JointPointObservation>& points =
        backend_points_it->second;
    int outer_count = 0;
    int internal_count = 0;
    for (const ati::JointPointObservation& point : points) {
      if (point.point_type == ati::JointPointType::Outer) {
        ++outer_count;
      } else if (point.point_type == ati::JointPointType::Internal) {
        ++internal_count;
      }
    }
    std::string frame_label = decision.frame_label;
    if (frame_label.empty()) {
      const auto label_it = frame_labels.find(decision.frame_index);
      if (label_it != frame_labels.end()) {
        frame_label = label_it->second;
      }
    }
    std::string image_file;
    const auto path_it = image_paths.find(decision.frame_index);
    if (path_it != image_paths.end() && !points.empty()) {
      const cv::Mat image = cv::imread(path_it->second, cv::IMREAD_UNCHANGED);
      cv::Mat overlay;
      DrawCloseEdgeSoftCandidateOverlay(image, decision, points,
                                        backend_optimized_residuals, &overlay);
      if (!overlay.empty()) {
        if (!trial_accepted_overlay_dir_created) {
          EnsureDirectoryExists(trial_accepted_overlay_dir);
          trial_accepted_overlay_dir_created = true;
        }
        std::ostringstream filename;
        if (decision.frame_cohesion_accepted) {
          filename << "frame_cohesion_";
        } else if (decision.close_distance_frame_admission_accepted) {
          filename << "close_distance_frame_admission_";
        } else {
          filename << "trial_accepted_";
        }
        filename << "frame_" << decision.frame_index << "_"
                 << SanitizeFilenameComponent(frame_label)
                 << "_board_" << decision.board_id << ".png";
        image_file = filename.str();
        cv::imwrite((trial_accepted_overlay_dir / image_file).string(),
                    overlay);
        ++trial_accepted_overlay_count;
      }
    }
    trial_accepted_overlay_csv
        << decision.frame_index << ","
        << CsvEscape(frame_label) << ","
        << decision.board_id << ","
        << CsvEscape(decision.reason) << ","
        << (decision.baseline_seed ? 1 : 0) << ","
        << (decision.frame_cohesion_candidate ? 1 : 0) << ","
        << (decision.frame_cohesion_accepted ? 1 : 0) << ","
        << (decision.close_distance_frame_admission_candidate ? 1 : 0) << ","
        << (decision.close_distance_frame_admission_accepted ? 1 : 0) << ","
        << (decision.close_distance_candidate ? 1 : 0) << ","
        << decision.close_distance_score_bonus << ","
        << decision.candidate_score << ","
        << decision.coverage_gain << ","
        << decision.global_rmse_delta << ","
        << decision.outer_rmse_delta << ","
        << decision.internal_rmse_delta << ","
        << points.size() << ","
        << outer_count << ","
        << internal_count << ","
        << CsvEscape(image_file) << "\n";
  }
  std::ofstream trial_accepted_overlay_summary(
      (backend_input_overlay_dir / "trial_accepted_frame_board_summary.txt")
          .string()
          .c_str());
  trial_accepted_overlay_summary
      << "trial_accepted_overlay_count: " << trial_accepted_overlay_count
      << "\n";
  trial_accepted_overlay_summary
      << "output_dir: trial_accepted_frame_boards\n";
  if (has_close_edge_soft_cases) {
    std::ofstream close_edge_soft_overlay_csv(
        (output_dir / "close_edge_soft_candidate_visualization_list.csv")
            .string()
            .c_str());
    close_edge_soft_overlay_csv
        << "frame_index,frame_label,board_id,soft_candidate,soft_attempted,"
        << "soft_accepted,soft_weight,close_edge_score,max_polar_angle_deg,"
        << "projected_area_ratio,outer_pose_refit_rmse,"
        << "soft_global_rmse_delta,soft_outer_rmse_delta,"
        << "soft_internal_rmse_delta,visualized_point_source,"
        << "visualized_point_count,visualized_outer_count,"
        << "visualized_internal_count,residual_available_count,"
        << "residual_gt3px_count,image_file,reason\n";
    bool close_edge_soft_overlay_dir_created = false;
    for (const ati::TrialBackendFrameBoardObservationDecision& decision :
         report.trial_backend_selection_result.decisions) {
      if (!decision.soft_candidate && !decision.soft_attempted &&
          !decision.soft_accepted) {
        continue;
      }
      const std::pair<int, int> key =
          std::make_pair(decision.frame_index, decision.board_id);
      std::vector<ati::JointPointObservation> points;
      std::string point_source = "backend_problem_input";
      const auto backend_points_it = backend_used_points_by_frame_board.find(key);
      if (backend_points_it != backend_used_points_by_frame_board.end()) {
        points = backend_points_it->second;
      } else {
        const auto measurement_points_it =
            measurement_points_by_frame_board.find(key);
        if (measurement_points_it != measurement_points_by_frame_board.end()) {
          points = measurement_points_it->second;
          point_source = "round_measurement";
        }
      }

      int outer_count = 0;
      int internal_count = 0;
      int residual_available_count = 0;
      int residual_gt3px_count = 0;
      for (const ati::JointPointObservation& point : points) {
        if (point.point_type == ati::JointPointType::Outer) {
          ++outer_count;
        } else if (point.point_type == ati::JointPointType::Internal) {
          ++internal_count;
        }
        BackendTrainingResidualKey residual_key;
        residual_key.frame_index = point.frame_index;
        residual_key.board_id = point.board_id;
        residual_key.point_id = point.point_id;
        const auto residual_it = backend_optimized_residuals.find(residual_key);
        if (residual_it != backend_optimized_residuals.end()) {
          ++residual_available_count;
          if (std::isfinite(residual_it->second.residual_norm) &&
              residual_it->second.residual_norm > 3.0) {
            ++residual_gt3px_count;
          }
        }
      }

      std::string frame_label = decision.frame_label;
      if (frame_label.empty()) {
        const auto label_it = frame_labels.find(decision.frame_index);
        if (label_it != frame_labels.end()) {
          frame_label = label_it->second;
        }
      }

      std::string image_file;
      const auto path_it = image_paths.find(decision.frame_index);
      if (path_it != image_paths.end() && !points.empty()) {
        const cv::Mat image = cv::imread(path_it->second, cv::IMREAD_UNCHANGED);
        cv::Mat overlay;
        DrawCloseEdgeSoftCandidateOverlay(image, decision, points,
                                          backend_optimized_residuals, &overlay);
        if (!overlay.empty()) {
          if (!close_edge_soft_overlay_dir_created) {
            EnsureDirectoryExists(close_edge_soft_overlay_dir);
            close_edge_soft_overlay_dir_created = true;
          }
          std::ostringstream filename;
          filename << (decision.soft_accepted ? "soft_accepted_"
                                               : "soft_rejected_")
                   << "frame_" << decision.frame_index << "_"
                   << SanitizeFilenameComponent(frame_label)
                   << "_board_" << decision.board_id << ".png";
          image_file = filename.str();
          cv::imwrite((close_edge_soft_overlay_dir / image_file).string(),
                      overlay);
          ++close_edge_soft_overlay_count;
          if (decision.soft_accepted) {
            ++close_edge_soft_accepted_overlay_count;
          } else {
            ++close_edge_soft_rejected_overlay_count;
          }
        }
      }

      close_edge_soft_overlay_csv
          << decision.frame_index << ","
          << CsvEscape(frame_label) << ","
          << decision.board_id << ","
          << (decision.soft_candidate ? 1 : 0) << ","
          << (decision.soft_attempted ? 1 : 0) << ","
          << (decision.soft_accepted ? 1 : 0) << ","
          << decision.soft_weight << ","
          << decision.close_edge_score << ","
          << decision.max_polar_angle_deg << ","
          << decision.projected_area_ratio << ","
          << decision.outer_pose_refit_rmse << ","
          << decision.soft_global_rmse_delta << ","
          << decision.soft_outer_rmse_delta << ","
          << decision.soft_internal_rmse_delta << ","
          << CsvEscape(point_source) << ","
          << points.size() << ","
          << outer_count << ","
          << internal_count << ","
          << residual_available_count << ","
          << residual_gt3px_count << ","
          << CsvEscape(image_file) << ","
          << CsvEscape(decision.reason) << "\n";
    }
    std::ofstream close_edge_soft_overlay_summary(
        (output_dir / "close_edge_soft_candidate_visualization_summary.txt")
            .string()
            .c_str());
    close_edge_soft_overlay_summary
        << "overlay_count: " << close_edge_soft_overlay_count << "\n";
    close_edge_soft_overlay_summary
        << "soft_accepted_overlay_count: "
        << close_edge_soft_accepted_overlay_count << "\n";
    close_edge_soft_overlay_summary
        << "soft_rejected_overlay_count: "
        << close_edge_soft_rejected_overlay_count << "\n";
    close_edge_soft_overlay_summary
        << "output_dir: close_edge_soft_candidate_visualizations\n";
  }

  std::ofstream backend_input_csv(
      (output_dir / "backend_input_used_frame_board_list.csv").string().c_str());
  backend_input_csv
      << "frame_index,frame_label,board_id,used_point_count,outer_point_count,"
      << "internal_point_count\n";
  for (const auto& entry : backend_used_points_by_frame_board) {
    const int frame_index = entry.first.first;
    const int board_id = entry.first.second;
    const std::vector<ati::JointPointObservation>& points = entry.second;
    int outer_count = 0;
    int internal_count = 0;
    for (const ati::JointPointObservation& point : points) {
      if (point.point_type == ati::JointPointType::Outer) {
        ++outer_count;
      } else if (point.point_type == ati::JointPointType::Internal) {
        ++internal_count;
      }
    }
    std::string frame_label;
    const auto label_it = frame_labels.find(frame_index);
    if (label_it != frame_labels.end()) {
      frame_label = label_it->second;
    }
    backend_input_csv << frame_index << "," << frame_label << "," << board_id
                      << "," << points.size() << "," << outer_count << ","
                      << internal_count << "\n";
  }

  std::ofstream internal_filter_csv(
      (output_dir / "internal_point_filter_by_board.csv").string().c_str());
  internal_filter_csv
      << "round,frame_index,frame_label,board_id,outer_point_count,"
      << "internal_total_count,internal_used_count,"
      << "internal_invalid_count,internal_reprojection_outlier_count,"
      << "internal_regeneration_failed_count,internal_other_rejected_count,"
      << "used_ratio,point_rejection_reasons\n";
  std::map<int, std::vector<const ati::JointBoardObservation*> >
      internal_filter_boards_by_frame;
  int internal_filter_board_count = 0;
  int internal_filter_partial_board_count = 0;
  int internal_filter_rejected_point_count = 0;
  std::map<std::string, int> internal_filter_reason_counts;
  for (const ati::JointMeasurementFrameResult& frame :
       artifacts.measurement_result.frames) {
    for (const ati::JointBoardObservation& board :
         frame.board_observations) {
      int internal_total = 0;
      int internal_used = 0;
      int internal_invalid = 0;
      int internal_outlier = 0;
      int internal_regen_failed = 0;
      int internal_other = 0;
      std::map<std::string, int> reason_counts;
      for (const ati::JointPointObservation& point : board.points) {
        if (point.point_type != ati::JointPointType::Internal) {
          continue;
        }
        ++internal_total;
        if (point.used_in_solver) {
          ++internal_used;
          continue;
        }
        ++internal_filter_rejected_point_count;
        const std::string reason = ati::ToString(point.rejection_reason_code);
        ++reason_counts[reason];
        ++internal_filter_reason_counts[reason];
        if (point.rejection_reason_code ==
            ati::JointRejectionReasonCode::InternalPointInvalid) {
          ++internal_invalid;
        } else if (point.rejection_reason_code ==
                   ati::JointRejectionReasonCode::InternalPointReprojectionOutlier) {
          ++internal_outlier;
        } else if (point.rejection_reason_code ==
                   ati::JointRejectionReasonCode::InternalRegenerationFailed) {
          ++internal_regen_failed;
        } else {
          ++internal_other;
        }
      }
      if (internal_total <= 0) {
        continue;
      }
      ++internal_filter_board_count;
      // Keep every board with a complete outer observation in this overlay.
      // This diagnostic is also used to inspect outer detections; limiting it
      // to boards with rejected internal points silently hid healthy boards.
      int observed_outer = 0;
      for (const ati::JointPointObservation& point : board.points) {
        if (point.point_type == ati::JointPointType::Outer &&
            point.image_xy.allFinite()) {
          ++observed_outer;
        }
      }
      if (observed_outer == 4 || internal_used < internal_total) {
        ++internal_filter_partial_board_count;
        internal_filter_boards_by_frame[frame.frame_index].push_back(&board);
      }
      internal_filter_csv
          << round_label << ","
          << frame.frame_index << ","
          << CsvEscape(frame.frame_label) << ","
          << board.board_id << ","
          << board.outer_point_count << ","
          << internal_total << ","
          << internal_used << ","
          << internal_invalid << ","
          << internal_outlier << ","
          << internal_regen_failed << ","
          << internal_other << ","
          << SafeRatio(internal_used, internal_total) << ","
          << CsvEscape(JoinReasonCounts(reason_counts)) << "\n";
    }
  }
  int internal_filter_overlay_count = 0;
  if (!internal_filter_boards_by_frame.empty()) {
    EnsureDirectoryExists(internal_filter_overlay_dir);
  }
  for (const auto& entry : internal_filter_boards_by_frame) {
    const auto path_it = image_paths.find(entry.first);
    if (path_it == image_paths.end()) {
      continue;
    }
    const cv::Mat image = cv::imread(path_it->second, cv::IMREAD_UNCHANGED);
    cv::Mat overlay;
    DrawInternalPointFilterOverlay(image, entry.second,
                                   regenerated_by_frame_board, &overlay);
    if (overlay.empty()) {
      continue;
    }
    std::string frame_label;
    const auto label_it = frame_labels.find(entry.first);
    if (label_it != frame_labels.end()) {
      frame_label = label_it->second;
    }
    std::ostringstream filename;
    filename << "frame_" << entry.first << "_"
             << SanitizeFilenameComponent(frame_label)
             << "_internal_filter.png";
    cv::imwrite((internal_filter_overlay_dir / filename.str()).string(),
                overlay);
    ++internal_filter_overlay_count;
  }
  std::ofstream internal_filter_summary(
      (output_dir / "internal_point_filter_summary.txt").string().c_str());
  internal_filter_summary << "round: " << round_label << "\n";
  internal_filter_summary
      << "filter_internal_corner_outliers: "
      << 1 << "\n";
  internal_filter_summary
      << "filter_internal_corner_mode: "
      << report.baseline_result.effective_options
             .internal_corner_filter_mode
      << "\n";
  internal_filter_summary
      << "filter_internal_corner_sigma_threshold: "
      << 2.0
      << "\n";
  internal_filter_summary
      << "filter_internal_corner_min_reproj_error: "
      << 0.2
      << "\n";
  internal_filter_summary
      << "filter_internal_corner_max_reproj_error: "
      << report.baseline_result.effective_options
             .internal_corner_filter_max_reproj_error
      << "\n";
  internal_filter_summary
      << "filter_internal_corner_quality_min: "
      << report.baseline_result.effective_options
             .internal_corner_filter_quality_min
      << "\n";
  internal_filter_summary
      << "filter_internal_corner_quality_relaxation_px: "
      << report.baseline_result.effective_options
             .internal_corner_filter_quality_relaxation_px
      << "\n";
  internal_filter_summary
      << "filter_internal_corner_adaptive_min_threshold_px: "
      << report.baseline_result.effective_options
             .internal_corner_filter_adaptive_min_threshold_px
      << "\n";
  internal_filter_summary
      << "filter_option_source: JointMeasurementBuildOptions defaults "
         "used by FrozenRound2BaselinePipeline\n";
  internal_filter_summary << "internal_board_count: "
                          << internal_filter_board_count << "\n";
  internal_filter_summary << "partial_or_rejected_internal_board_count: "
                          << internal_filter_partial_board_count << "\n";
  internal_filter_summary << "rejected_internal_point_count: "
                          << internal_filter_rejected_point_count << "\n";
  internal_filter_summary << "internal_point_filter_overlay_count: "
                          << internal_filter_overlay_count << "\n";
  internal_filter_summary << "internal_point_filter_overlay_dir: "
                          << internal_filter_overlay_dir.string() << "\n";
  internal_filter_summary << "\nrejection_reason_counts:\n";
  for (const auto& entry : internal_filter_reason_counts) {
    internal_filter_summary << "  " << entry.first << ": "
                            << entry.second << "\n";
  }
  internal_filter_summary
      << "\nrule_explanation:\n"
      << "  internal_point_invalid means CornerMeasurement.valid=false from "
         "internal regeneration/refinement.\n"
      << "  internal_point_reprojection_outlier means the point was valid, "
         "but under an outer-only local pose refit its reprojection residual "
         "was greater than mean + sigma_threshold * std and also greater "
         "than min_reproj_error.\n"
      << "  quality_residual_adaptive additionally adjusts that local "
         "residual threshold using each internal point's image quality: "
         "low-quality points are stricter, high-quality points get a small "
         "relaxation.\n"
      << "  These diagnostics are read-only and do not change selection or "
         "backend optimization.\n";

  std::vector<cv::Mat> montage_tiles;
  int rendered_count = 0;
  const bool write_full_flow_overlays = false;
  if (write_full_flow_overlays) {
    EnsureDirectoryExists(overlay_dir);
    for (const auto& entry : rows_by_frame) {
      bool has_non_used_or_partial = false;
      for (const FrameBoardObservationFlowRow& row : entry.second) {
        if (!row.final_used_in_backend ||
            row.final_status == "used_partial_internal_points") {
          has_non_used_or_partial = true;
          break;
        }
      }
      if (!has_non_used_or_partial) {
        continue;
      }
      const auto path_it = image_paths.find(entry.first);
      if (path_it == image_paths.end()) {
        continue;
      }
      const cv::Mat image = cv::imread(path_it->second, cv::IMREAD_UNCHANGED);
      cv::Mat overlay;
      DrawFrameBoardObservationFlowOverlay(
          image, entry.second, geometries, &overlay);
      if (overlay.empty()) {
        continue;
      }
      std::ostringstream filename;
      filename << "frame_" << entry.first << "_"
               << SanitizeFilenameComponent(entry.second.front().frame_label)
               << "_flow.png";
      cv::imwrite((overlay_dir / filename.str()).string(), overlay);
      if (montage_tiles.size() < 24) {
        cv::Mat tile;
        const int tile_width = 520;
        const double scale =
            static_cast<double>(tile_width) / static_cast<double>(overlay.cols);
        cv::resize(overlay, tile, cv::Size(tile_width,
                                           std::max(1, static_cast<int>(
                                                          std::round(overlay.rows * scale)))));
        montage_tiles.push_back(tile);
      }
      ++rendered_count;
    }
  }

  int strict_seed_overlay_count = 0;
  const bool write_strict_seed_overlays = false;
  if (write_strict_seed_overlays) {
    EnsureDirectoryExists(strict_seed_overlay_dir);
    EnsureDirectoryExists(strict_refined_overlay_dir);
    for (const auto& entry : strict_rows_by_frame) {
      const auto path_it = image_paths.find(entry.first);
      if (path_it == image_paths.end()) {
        continue;
      }
      const cv::Mat image = cv::imread(path_it->second, cv::IMREAD_UNCHANGED);
      cv::Mat overlay;
      DrawStrictInternalSeedOverlay(image, entry.second,
                                    artifacts.regeneration_results, &overlay);
      if (overlay.empty()) {
        continue;
      }
      std::ostringstream filename;
      filename << "frame_" << entry.first << "_"
               << SanitizeFilenameComponent(entry.second.front().frame_label)
               << "_strict_internal_seed.png";
      cv::imwrite((strict_seed_overlay_dir / filename.str()).string(), overlay);
      cv::imwrite((strict_refined_overlay_dir / filename.str()).string(), overlay);
      ++strict_seed_overlay_count;
    }
    std::ofstream strict_seed_summary(
        (strict_seed_overlay_dir / "overlay_summary.txt").string().c_str());
    strict_seed_summary << "round: " << round_label << "\n";
    strict_seed_summary << "strict_internal_seed_overlay_count: "
                        << strict_seed_overlay_count << "\n";
    strict_seed_summary << "refined_corner_overlay_dir: "
                        << strict_refined_overlay_dir.string() << "\n";
    strict_seed_summary << "strict_dropped_frame_count: "
                        << strict_rows_by_frame.size() << "\n";
  }

  if (!montage_tiles.empty()) {
    const int cols = 2;
    const int tile_width = montage_tiles.front().cols;
    int tile_height = 0;
    for (const cv::Mat& tile : montage_tiles) {
      tile_height = std::max(tile_height, tile.rows);
    }
    const int rows_count =
        static_cast<int>((montage_tiles.size() + cols - 1) / cols);
    cv::Mat montage(rows_count * tile_height, cols * tile_width, CV_8UC3,
                    cv::Scalar(20, 20, 20));
    for (std::size_t index = 0; index < montage_tiles.size(); ++index) {
      const int row = static_cast<int>(index) / cols;
      const int col = static_cast<int>(index) % cols;
      cv::Mat roi = montage(cv::Rect(col * tile_width, row * tile_height,
                                     montage_tiles[index].cols,
                                     montage_tiles[index].rows));
      montage_tiles[index].copyTo(roi);
    }
    cv::imwrite((output_dir / "frame_board_observation_flow_montage.png").string(),
                montage);
  }

  const fs::path forced_seed_overlay_dir =
      output_dir / "forced_internal_seed_overlays";
  const fs::path rescued_internal_overlay_dir =
      output_dir / "rescued_internal_point_overlays";
  std::ostringstream forced_csv_buffer;
  forced_csv_buffer
      << "round,frame_index,frame_label,board_id,point_id,corner_type,"
      << "forced_prediction_seed,final_valid,image_evidence_valid,"
      << "bypass_seed_filters,original_seed_filter_success,"
      << "original_seed_filter_would_reject,"
      << "backend_used,residual_x,residual_y,residual_norm,"
      << "predicted_u,predicted_v,sphere_seed_u,sphere_seed_v,"
      << "original_seed_filter_u,original_seed_filter_v,"
      << "refined_u,refined_v,backend_predicted_u,backend_predicted_v,"
      << "sphere_seed_quality,image_final_quality,final_quality,"
      << "q_refine,seed_to_refined_displacement,"
      << "predicted_to_refined_displacement\n";
  std::map<int, std::vector<const ati::RegeneratedBoardMeasurement*> >
      forced_measurements_by_frame;
  std::map<int, std::vector<const ati::RegeneratedBoardMeasurement*> >
      rescued_measurements_by_frame;
  int forced_total_count = 0;
  int forced_backend_used_count = 0;
  int forced_valid_count = 0;
  int forced_original_seed_rejected_count = 0;
  int forced_original_seed_rejected_backend_used_count = 0;
  int forced_original_seed_rejected_high_residual_count = 0;
  int forced_high_residual_count = 0;
  double forced_residual_squared_sum = 0.0;
  std::map<int, int> forced_count_by_board;
  std::map<int, int> forced_backend_used_count_by_board;
  std::map<int, int> forced_high_residual_count_by_board;
  int rescued_internal_total_count = 0;
  int rescued_internal_valid_count = 0;
  int rescued_internal_backend_used_count = 0;
  std::map<int, int> rescued_internal_count_by_board;
  std::map<int, int> rescued_internal_backend_used_count_by_board;
  std::ostringstream rescued_csv_buffer;
  rescued_csv_buffer
      << "round,frame_index,frame_label,board_id,point_id,corner_type,"
      << "final_valid,image_evidence_valid,backend_used,"
      << "residual_x,residual_y,residual_norm,"
      << "predicted_u,predicted_v,sphere_seed_u,sphere_seed_v,"
      << "refined_u,refined_v,backend_predicted_u,backend_predicted_v,"
      << "sphere_seed_quality,image_final_quality,final_quality,"
      << "q_refine,seed_to_refined_displacement,"
      << "predicted_to_refined_displacement\n";
  for (const ati::InternalRegenerationFrameResult& frame :
       artifacts.regeneration_results) {
    for (const ati::RegeneratedBoardMeasurement& measurement :
         frame.board_measurements) {
      if (measurement.detection.outer_detection.success &&
          measurement.detection.outer_detection.used_local_patch_rescue) {
        rescued_measurements_by_frame[frame.frame_index].push_back(&measurement);
        for (const ati::InternalCornerDebugInfo& debug :
             measurement.detection.internal_corner_debug) {
          ++rescued_internal_total_count;
          ++rescued_internal_count_by_board[measurement.board_id];
          if (debug.valid) {
            ++rescued_internal_valid_count;
          }
          BackendTrainingResidualKey key;
          key.frame_index = frame.frame_index;
          key.board_id = measurement.board_id;
          key.point_id = debug.point_id;
          const auto residual_it = backend_training_residuals.find(key);
          const bool backend_used =
              residual_it != backend_training_residuals.end();
          double residual_x = 0.0;
          double residual_y = 0.0;
          double residual_norm = -1.0;
          double backend_predicted_x = std::numeric_limits<double>::quiet_NaN();
          double backend_predicted_y = std::numeric_limits<double>::quiet_NaN();
          if (backend_used) {
            ++rescued_internal_backend_used_count;
            ++rescued_internal_backend_used_count_by_board[measurement.board_id];
            residual_x = residual_it->second.residual_x;
            residual_y = residual_it->second.residual_y;
            residual_norm = residual_it->second.residual_norm;
            backend_predicted_x = residual_it->second.predicted_x;
            backend_predicted_y = residual_it->second.predicted_y;
          }
          rescued_csv_buffer
              << round_label << ","
              << frame.frame_index << ","
              << CsvEscape(frame.frame_label) << ","
              << measurement.board_id << ","
              << debug.point_id << ","
              << ati::ToString(debug.corner_type) << ","
              << (debug.valid ? 1 : 0) << ","
              << (debug.image_evidence_valid ? 1 : 0) << ","
              << (backend_used ? 1 : 0) << ","
              << residual_x << ","
              << residual_y << ","
              << residual_norm << ","
              << debug.predicted_image.x << ","
              << debug.predicted_image.y << ","
              << debug.sphere_seed_image.x << ","
              << debug.sphere_seed_image.y << ","
              << debug.refined_image.x << ","
              << debug.refined_image.y << ","
              << backend_predicted_x << ","
              << backend_predicted_y << ","
              << debug.sphere_seed_quality << ","
              << debug.image_final_quality << ","
              << debug.final_quality << ","
              << debug.q_refine << ","
              << debug.seed_to_refined_displacement << ","
              << debug.predicted_to_refined_displacement << "\n";
        }
      }

      bool measurement_has_forced_seed = false;
      for (const ati::InternalCornerDebugInfo& debug :
           measurement.detection.internal_corner_debug) {
        if (!debug.forced_prediction_seed) {
          continue;
        }
        measurement_has_forced_seed = true;
        ++forced_total_count;
        ++forced_count_by_board[measurement.board_id];
        if (debug.valid) {
          ++forced_valid_count;
        }
        if (debug.original_seed_filter_would_reject) {
          ++forced_original_seed_rejected_count;
        }
        BackendTrainingResidualKey key;
        key.frame_index = frame.frame_index;
        key.board_id = measurement.board_id;
        key.point_id = debug.point_id;
        const auto residual_it = backend_training_residuals.find(key);
        const bool backend_used = residual_it != backend_training_residuals.end();
        double residual_x = 0.0;
        double residual_y = 0.0;
        double residual_norm = -1.0;
        double backend_predicted_x = std::numeric_limits<double>::quiet_NaN();
        double backend_predicted_y = std::numeric_limits<double>::quiet_NaN();
        if (backend_used) {
          ++forced_backend_used_count;
          ++forced_backend_used_count_by_board[measurement.board_id];
          if (debug.original_seed_filter_would_reject) {
            ++forced_original_seed_rejected_backend_used_count;
          }
          residual_x = residual_it->second.residual_x;
          residual_y = residual_it->second.residual_y;
          residual_norm = residual_it->second.residual_norm;
          backend_predicted_x = residual_it->second.predicted_x;
          backend_predicted_y = residual_it->second.predicted_y;
          forced_residual_squared_sum += residual_norm * residual_norm;
          if (residual_norm > 3.0) {
            ++forced_high_residual_count;
            ++forced_high_residual_count_by_board[measurement.board_id];
            if (debug.original_seed_filter_would_reject) {
              ++forced_original_seed_rejected_high_residual_count;
            }
          }
        }
        forced_csv_buffer
            << round_label << ","
            << frame.frame_index << ","
            << CsvEscape(frame.frame_label) << ","
            << measurement.board_id << ","
            << debug.point_id << ","
            << ati::ToString(debug.corner_type) << ","
            << (debug.forced_prediction_seed ? 1 : 0) << ","
            << (debug.valid ? 1 : 0) << ","
            << (debug.image_evidence_valid ? 1 : 0) << ","
            << (debug.bypass_seed_filters ? 1 : 0) << ","
            << (debug.original_seed_filter_success ? 1 : 0) << ","
            << (debug.original_seed_filter_would_reject ? 1 : 0) << ","
            << (backend_used ? 1 : 0) << ","
            << residual_x << ","
            << residual_y << ","
            << residual_norm << ","
            << debug.predicted_image.x << ","
            << debug.predicted_image.y << ","
            << debug.sphere_seed_image.x << ","
            << debug.sphere_seed_image.y << ","
            << debug.original_seed_filter_image.x << ","
            << debug.original_seed_filter_image.y << ","
            << debug.refined_image.x << ","
            << debug.refined_image.y << ","
            << backend_predicted_x << ","
            << backend_predicted_y << ","
            << debug.sphere_seed_quality << ","
            << debug.image_final_quality << ","
            << debug.final_quality << ","
            << debug.q_refine << ","
            << debug.seed_to_refined_displacement << ","
            << debug.predicted_to_refined_displacement << "\n";
      }
      if (measurement_has_forced_seed) {
        forced_measurements_by_frame[frame.frame_index].push_back(&measurement);
      }
    }
  }
  int forced_overlay_count = 0;
  if (!forced_measurements_by_frame.empty()) {
    EnsureDirectoryExists(forced_seed_overlay_dir);
  }
  for (const auto& entry : forced_measurements_by_frame) {
    const auto path_it = image_paths.find(entry.first);
    if (path_it == image_paths.end()) {
      continue;
    }
    const cv::Mat image = cv::imread(path_it->second, cv::IMREAD_UNCHANGED);
    cv::Mat overlay;
    DrawForcedInternalSeedOverlay(image, entry.first, entry.second,
                                  backend_training_residuals, &overlay);
    if (overlay.empty()) {
      continue;
    }
    std::string frame_label;
    const auto label_it = frame_labels.find(entry.first);
    if (label_it != frame_labels.end()) {
      frame_label = label_it->second;
    }
    std::ostringstream filename;
    filename << "frame_" << entry.first << "_"
             << SanitizeFilenameComponent(frame_label)
             << "_forced_internal_seed.png";
    cv::imwrite((forced_seed_overlay_dir / filename.str()).string(), overlay);
    ++forced_overlay_count;
  }
  int rescued_internal_overlay_count = 0;
  if (!rescued_measurements_by_frame.empty()) {
    EnsureDirectoryExists(rescued_internal_overlay_dir);
  }
  for (const auto& entry : rescued_measurements_by_frame) {
    const auto path_it = image_paths.find(entry.first);
    if (path_it == image_paths.end()) {
      continue;
    }
    const cv::Mat image = cv::imread(path_it->second, cv::IMREAD_UNCHANGED);
    cv::Mat overlay;
    DrawRescuedInternalPointOverlay(image, entry.first, entry.second,
                                    backend_training_residuals, &overlay);
    if (overlay.empty()) {
      continue;
    }
    std::string frame_label;
    const auto label_it = frame_labels.find(entry.first);
    if (label_it != frame_labels.end()) {
      frame_label = label_it->second;
    }
    std::ostringstream filename;
    filename << "frame_" << entry.first << "_"
             << SanitizeFilenameComponent(frame_label)
             << "_rescued_internal_points.png";
    cv::imwrite((rescued_internal_overlay_dir / filename.str()).string(),
                overlay);
    ++rescued_internal_overlay_count;
  }
  if (rescued_internal_total_count > 0) {
    std::ofstream rescued_csv(
        (output_dir / "rescued_internal_point_report.csv").string().c_str());
    rescued_csv << rescued_csv_buffer.str();
    std::ofstream rescued_summary(
        (output_dir / "rescued_internal_point_summary.txt").string().c_str());
    rescued_summary << "round: " << round_label << "\n";
    rescued_summary << "rescued_internal_total_count: "
                    << rescued_internal_total_count << "\n";
    rescued_summary << "rescued_internal_valid_count: "
                    << rescued_internal_valid_count << "\n";
    rescued_summary << "rescued_internal_backend_used_count: "
                    << rescued_internal_backend_used_count << "\n";
    rescued_summary << "rescued_internal_backend_used_ratio: "
                    << SafeRatio(rescued_internal_backend_used_count,
                                 rescued_internal_total_count)
                    << "\n";
    rescued_summary << "rescued_internal_overlay_count: "
                    << rescued_internal_overlay_count << "\n";
    rescued_summary << "rescued_internal_overlay_dir: "
                    << rescued_internal_overlay_dir.string() << "\n";
    rescued_summary << "\nby_board:\n";
    for (const auto& entry : rescued_internal_count_by_board) {
      const int board_id = entry.first;
      rescued_summary << "  board " << board_id
                      << " rescued_internal=" << entry.second
                      << " backend_used="
                      << rescued_internal_backend_used_count_by_board[board_id]
                      << "\n";
    }
    WriteRescuedBoardLocalVsGlobalResidualDiagnostics(
        output_dir, report, artifacts.regeneration_results,
        backend_training_residuals);
  }
  if (forced_total_count > 0) {
    std::ofstream forced_csv(
        (output_dir / "forced_internal_seed_quality_report.csv").string().c_str());
    forced_csv << forced_csv_buffer.str();
    std::ofstream forced_summary(
        (output_dir / "forced_internal_seed_quality_summary.txt").string().c_str());
    forced_summary << "round: " << round_label << "\n";
    forced_summary << "forced_seed_total_count: " << forced_total_count << "\n";
    forced_summary << "forced_seed_final_valid_count: " << forced_valid_count << "\n";
    forced_summary << "forced_seed_backend_used_count: "
                   << forced_backend_used_count << "\n";
    forced_summary << "forced_seed_backend_used_ratio: "
                   << SafeRatio(forced_backend_used_count, forced_total_count)
                   << "\n";
    forced_summary << "forced_seed_original_filter_rejected_count: "
                   << forced_original_seed_rejected_count << "\n";
    forced_summary << "forced_seed_original_filter_rejected_backend_used_count: "
                   << forced_original_seed_rejected_backend_used_count << "\n";
    forced_summary << "forced_seed_original_filter_rejected_high_residual_gt3px_count: "
                   << forced_original_seed_rejected_high_residual_count << "\n";
    forced_summary << "forced_seed_backend_rmse: "
                   << (forced_backend_used_count > 0
                         ? std::sqrt(forced_residual_squared_sum /
                                     static_cast<double>(forced_backend_used_count))
                         : 0.0)
                 << "\n";
  forced_summary << "forced_seed_high_residual_gt3px_count: "
                 << forced_high_residual_count << "\n";
  forced_summary << "forced_seed_overlay_count: " << forced_overlay_count << "\n";
  forced_summary << "forced_seed_overlay_dir: "
                 << forced_seed_overlay_dir.string() << "\n";
  forced_summary << "\nby_board:\n";
  for (const auto& entry : forced_count_by_board) {
    const int board_id = entry.first;
    forced_summary << "  board " << board_id
                   << " forced=" << entry.second
                   << " backend_used=" << forced_backend_used_count_by_board[board_id]
                   << " high_residual_gt3px="
                   << forced_high_residual_count_by_board[board_id] << "\n";
  }
  }
  summary << "\nrendered_overlay_frame_count: " << rendered_count << "\n";
  summary << "overlay_dir: "
          << (overlay_dir.empty() ? std::string()
                                  : overlay_dir.string())
          << "\n";
  summary << "strict_internal_seed_overlay_count: "
          << strict_seed_overlay_count << "\n";
  summary << "strict_internal_seed_overlay_dir: "
          << strict_seed_overlay_dir.string() << "\n";
  summary << "internal_point_filter_overlay_count: "
          << internal_filter_overlay_count << "\n";
  summary << "internal_point_filter_overlay_dir: "
          << internal_filter_overlay_dir.string() << "\n";
  summary << "backend_used_observation_overlay_count: "
          << backend_used_overlay_count << "\n";
  summary << "backend_used_observation_overlay_point_count: "
          << backend_used_overlay_point_count << "\n";
  summary << "backend_used_observation_overlay_outer_point_count: "
          << backend_used_overlay_outer_point_count << "\n";
  summary << "backend_used_observation_overlay_internal_point_count: "
          << backend_used_overlay_internal_point_count << "\n";
  summary << "backend_input_visualization_dir: "
          << backend_input_overlay_dir.string() << "\n";
  summary << "backend_input_used_frame_board_list_csv: "
          << (output_dir / "backend_input_used_frame_board_list.csv").string()
          << "\n";
  summary << "trial_accepted_frame_board_overlay_count: "
          << trial_accepted_overlay_count << "\n";
  summary << "trial_accepted_frame_board_overlay_dir: "
          << trial_accepted_overlay_dir.string() << "\n";
  summary << "forced_internal_seed_total_count: "
          << forced_total_count << "\n";
  summary << "forced_internal_seed_backend_used_count: "
          << forced_backend_used_count << "\n";
  summary << "forced_internal_seed_overlay_dir: "
          << forced_seed_overlay_dir.string() << "\n";
  summary << "rescued_internal_point_total_count: "
          << rescued_internal_total_count << "\n";
  summary << "rescued_internal_point_backend_used_count: "
          << rescued_internal_backend_used_count << "\n";
  summary << "rescued_internal_point_overlay_dir: "
          << rescued_internal_overlay_dir.string() << "\n";
}

void WriteTrialBackendFrameBoardSelectionDiagnostics(
    const fs::path& output_dir,
    const ati::Stage5BenchmarkReport& report) {
  const ati::TrialBackendFrameBoardSelectionResult& result =
      report.trial_backend_selection_result;
  std::ofstream summary(
      (output_dir / "trial_backend_frame_board_selection_summary.txt")
          .string()
          .c_str());
  summary << "enabled: " << (result.enabled ? 1 : 0) << "\n";
  summary << "success: " << (result.success ? 1 : 0) << "\n";
  summary << "failure_reason: " << result.failure_reason << "\n";
  summary << "source_joint_input_frame_count: "
          << result.source_joint_input_frame_count << "\n";
  summary << "source_measurement_frame_count: "
          << result.source_measurement_frame_count << "\n";
  summary << "source_measurement_board_observation_count: "
          << result.source_measurement_board_observation_count << "\n";
  summary << "source_measurement_outer_point_count: "
          << result.source_measurement_outer_point_count << "\n";
  summary << "source_measurement_internal_point_count: "
          << result.source_measurement_internal_point_count << "\n";
  summary << "source_measurement_total_point_count: "
          << result.source_measurement_total_point_count << "\n";
  summary << "source_measurement_hierarchical_frame_count: "
          << result.source_measurement_hierarchical_frame_count << "\n";
  summary << "source_measurement_hierarchical_board_observation_count: "
          << result.source_measurement_hierarchical_board_observation_count
          << "\n";
  summary << "source_measurement_hierarchical_outer_point_count: "
          << result.source_measurement_hierarchical_outer_point_count << "\n";
  summary << "source_measurement_hierarchical_internal_point_count: "
          << result.source_measurement_hierarchical_internal_point_count
          << "\n";
  summary << "source_measurement_hierarchical_total_point_count: "
          << result.source_measurement_hierarchical_total_point_count << "\n";
  summary << "source_measurement_flat_solver_observation_count: "
          << result.source_measurement_flat_solver_observation_count << "\n";
  summary << "source_selection_frame_count: "
          << result.source_selection_frame_count << "\n";
  summary << "source_selection_board_observation_count: "
          << result.source_selection_board_observation_count << "\n";
  summary << "source_selection_outer_point_count: "
          << result.source_selection_outer_point_count << "\n";
  summary << "source_selection_internal_point_count: "
          << result.source_selection_internal_point_count << "\n";
  summary << "source_selection_total_point_count: "
          << result.source_selection_total_point_count << "\n";
  summary << "candidate_pool_frame_count: "
          << result.candidate_pool_frame_count << "\n";
  summary << "candidate_pool_board_observation_count: "
          << result.candidate_pool_board_observation_count << "\n";
  summary << "candidate_pool_outer_point_count: "
          << result.candidate_pool_outer_point_count << "\n";
  summary << "candidate_pool_internal_point_count: "
          << result.candidate_pool_internal_point_count << "\n";
  summary << "candidate_pool_total_point_count: "
          << result.candidate_pool_total_point_count << "\n";
  summary << "input_frame_count: " << result.input_frame_count << "\n";
  summary << "input_board_observation_count: "
          << result.input_board_observation_count << "\n";
  summary << "input_total_point_count: " << result.input_total_point_count
          << "\n";
  summary << "baseline_seed_frame_count: "
          << result.baseline_seed_frame_count << "\n";
  summary << "baseline_seed_board_observation_count: "
          << result.baseline_seed_board_observation_count << "\n";
  summary << "baseline_seed_outer_point_count: "
          << result.baseline_seed_outer_point_count << "\n";
  summary << "baseline_seed_internal_point_count: "
          << result.baseline_seed_internal_point_count << "\n";
  summary << "baseline_seed_total_point_count: "
          << result.baseline_seed_total_point_count << "\n";
  summary << "candidate_board_observation_count: "
          << result.candidate_board_observation_count << "\n";
  summary << "stage5_selection_profile: "
          << result.selection_profile << "\n";
  summary << "stage5_selection_is_kalibr_checkerboard_style: "
          << (result.selection_is_kalibr_checkerboard_style ? 1 : 0)
          << "\n";
  summary << "stage5_checkerboard_huber_delta_pixels: "
          << result.checkerboard_huber_delta_pixels << "\n";
  summary << "stage5_checkerboard_force_all_valid_views: "
          << (result.checkerboard_force_all_valid_views ? 1 : 0) << "\n";
  summary << "stage5_checkerboard_pose_marginalized_fisher: "
          << (result.checkerboard_pose_marginalized_fisher ? 1 : 0) << "\n";
  summary << "stage5_checkerboard_seed_strategy: "
          << result.checkerboard_seed_strategy << "\n";
  summary << "stage5_checkerboard_seed_target_frame_count: "
          << result.checkerboard_seed_target_frame_count << "\n";
  summary << "stage5_checkerboard_seed_fisher_logdet: "
          << result.checkerboard_seed_fisher_logdet << "\n";
  summary << "stage5_checkerboard_seed_fisher_rank_proxy: "
          << result.checkerboard_seed_fisher_rank_proxy << "\n";
  summary << "stage5_checkerboard_outlier_filter_enabled: "
          << (result.checkerboard_outlier_filter_enabled ? 1 : 0) << "\n";
  summary << "stage5_checkerboard_outlier_sigma: "
          << result.checkerboard_outlier_sigma << "\n";
  summary << "stage5_checkerboard_min_inlier_ratio: "
          << result.checkerboard_min_inlier_ratio << "\n";
  summary << "stage5_checkerboard_min_retained_points: "
          << result.checkerboard_min_retained_points << "\n";
  summary << "stage5_checkerboard_outlier_threshold_pixels: "
          << result.checkerboard_outlier_threshold_pixels << "\n";
  summary << "stage5_checkerboard_outlier_median_pixels: "
          << result.checkerboard_outlier_median_pixels << "\n";
  summary << "stage5_checkerboard_outlier_robust_sigma_pixels: "
          << result.checkerboard_outlier_robust_sigma_pixels << "\n";
  summary << "stage5_checkerboard_outlier_removed_point_count: "
          << result.checkerboard_outlier_removed_point_count << "\n";
  summary << "stage5_checkerboard_outlier_dropped_view_count: "
          << result.checkerboard_outlier_dropped_view_count << "\n";
  summary << "attempted_candidate_count: "
          << result.attempted_candidate_count << "\n";
  summary << "accepted_candidate_count: "
          << result.accepted_candidate_count << "\n";
  summary << "selection_mode: "
          << ati::ToString(result.selection_mode) << "\n";
  summary << "candidate_order_mode: "
          << ati::ToString(result.candidate_order_mode) << "\n";
  summary << "info_gain_proxy_mode: "
          << ati::ToString(result.info_gain_proxy_mode) << "\n";
  summary << "candidate_batch_granularity: "
          << ati::ToString(result.candidate_batch_granularity) << "\n";
  summary << "candidate_shuffle_seed_set: "
          << (result.candidate_shuffle_seed_set ? 1 : 0) << "\n";
  summary << "candidate_shuffle_seed: "
          << result.candidate_shuffle_seed << "\n";
  summary << "batch_acceptance_attempted_count: "
          << result.batch_acceptance_attempted_count << "\n";
  summary << "batch_acceptance_accepted_count: "
          << result.batch_acceptance_accepted_count << "\n";
  summary << "batch_acceptance_rescued_from_legacy_rmse_gate_count: "
          << result.batch_acceptance_rescued_from_legacy_rmse_gate_count
          << "\n";
  summary << "batch_acceptance_rejected_hard_validity_count: "
          << result.batch_acceptance_rejected_hard_validity_count << "\n";
  summary << "batch_acceptance_rejected_catastrophic_residual_count: "
          << result.batch_acceptance_rejected_catastrophic_residual_count
          << "\n";
  summary << "batch_acceptance_rejected_score_count: "
          << result.batch_acceptance_rejected_score_count << "\n";
  summary << "persistent_incremental_backend_estimator_attempted: "
          << (result.persistent_incremental_backend_estimator_attempted ? 1 : 0)
          << "\n";
  summary << "persistent_incremental_backend_estimator_used: "
          << (result.persistent_incremental_backend_estimator_used ? 1 : 0)
          << "\n";
  summary << "persistent_incremental_backend_estimator_compatible: "
          << (result.persistent_incremental_backend_estimator_compatible ? 1 : 0)
          << "\n";
  summary << "persistent_incremental_backend_estimator_fallback_reason: "
          << result.persistent_incremental_backend_estimator_fallback_reason
          << "\n";
  summary << "persistent_incremental_backend_estimator_failure_reason: "
          << result.persistent_incremental_backend_estimator_failure_reason
          << "\n";
  summary << "persistent_incremental_information_gain_target: "
          << result.persistent_incremental_information_gain_target << "\n";
  summary << "persistent_incremental_board_layout_in_information_group: "
          << (result.persistent_incremental_board_layout_in_information_group
                  ? 1
                  : 0)
          << "\n";
  summary << "persistent_incremental_board_layout_fixed: "
          << (result.persistent_incremental_board_layout_fixed ? 1 : 0)
          << "\n";
  summary << "persistent_incremental_board_layout_pose_count: "
          << result.persistent_incremental_board_layout_pose_count << "\n";
  summary << "persistent_incremental_board_layout_max_matrix_abs_delta: "
          << result.persistent_incremental_board_layout_max_matrix_abs_delta
          << "\n";
  summary << "persistent_incremental_board_layout_max_translation_delta: "
          << result.persistent_incremental_board_layout_max_translation_delta
          << "\n";
  summary << "persistent_incremental_board_layout_max_rotation_delta_deg: "
          << result.persistent_incremental_board_layout_max_rotation_delta_deg
          << "\n";
  summary << "persistent_incremental_camera_information_group_id: "
          << result.persistent_incremental_camera_information_group_id << "\n";
  summary << "persistent_incremental_board_layout_group_id: "
          << result.persistent_incremental_board_layout_group_id << "\n";
  summary << "persistent_incremental_transformation_group_id: "
          << result.persistent_incremental_transformation_group_id << "\n";
  summary << "persistent_incremental_seed_information_group_dim: "
          << result.persistent_incremental_seed_information_group_dim << "\n";
  summary << "persistent_incremental_seed_information_rank: "
          << result.persistent_incremental_seed_information_rank << "\n";
  summary << "persistent_incremental_seed_information_rank_deficiency: "
          << result.persistent_incremental_seed_information_rank_deficiency
          << "\n";
  summary << "persistent_incremental_seed_information_baseline_valid: "
          << (result.persistent_incremental_seed_information_baseline_valid ? 1
                                                                            : 0)
          << "\n";
  summary << "persistent_incremental_seed_information_scaled_min_singular_value: "
          << result
                 .persistent_incremental_seed_information_scaled_min_singular_value
          << "\n";
  summary << "persistent_incremental_seed_information_scaled_max_singular_value: "
          << result
                 .persistent_incremental_seed_information_scaled_max_singular_value
          << "\n";
  summary << "persistent_incremental_seed_information_scaled_condition_number: "
          << result
                 .persistent_incremental_seed_information_scaled_condition_number
          << "\n";
  summary << "persistent_incremental_seed_information_ds_cu_stddev_px: "
          << result.persistent_incremental_seed_information_ds_cu_stddev_px
          << "\n";
  summary << "persistent_incremental_seed_information_ds_cv_stddev_px: "
          << result.persistent_incremental_seed_information_ds_cv_stddev_px
          << "\n";
  summary << "persistent_incremental_seed_batch_count: "
          << result.persistent_incremental_seed_batch_count << "\n";
  summary << "persistent_incremental_seed_frame_count: "
          << result.persistent_incremental_seed_frame_count << "\n";
  summary << "persistent_incremental_seed_board_observation_count: "
          << result.persistent_incremental_seed_board_observation_count << "\n";
  summary << "persistent_incremental_seed_point_count: "
          << result.persistent_incremental_seed_point_count << "\n";
  summary << "persistent_incremental_seed_intrinsics_warmup_attempted: "
          << (result.persistent_incremental_seed_intrinsics_warmup_attempted
                  ? 1
                  : 0)
          << "\n";
  summary << "persistent_incremental_seed_intrinsics_warmup_success: "
          << (result.persistent_incremental_seed_intrinsics_warmup_success ? 1
                                                                           : 0)
          << "\n";
  summary << "persistent_incremental_seed_intrinsics_warmup_converged_by_relative_objective: "
          << (result
                      .persistent_incremental_seed_intrinsics_warmup_converged_by_relative_objective
                  ? 1
                  : 0)
          << "\n";
  summary << "persistent_incremental_seed_intrinsics_warmup_iterations: "
          << result.persistent_incremental_seed_intrinsics_warmup_iterations
          << "\n";
  summary << "persistent_incremental_seed_intrinsics_warmup_objective_start: "
          << result
                 .persistent_incremental_seed_intrinsics_warmup_objective_start
          << "\n";
  summary << "persistent_incremental_seed_intrinsics_warmup_objective_final: "
          << result
                 .persistent_incremental_seed_intrinsics_warmup_objective_final
          << "\n";
  summary << "persistent_incremental_seed_intrinsics_warmup_last_delta_j: "
          << result.persistent_incremental_seed_intrinsics_warmup_last_delta_j
          << "\n";
  summary << "persistent_incremental_seed_intrinsics_warmup_last_delta_x: "
          << result.persistent_incremental_seed_intrinsics_warmup_last_delta_x
          << "\n";
  summary << "persistent_incremental_candidate_batch_count: "
          << result.persistent_incremental_candidate_batch_count << "\n";
  summary << "persistent_incremental_attempted_batch_count: "
          << result.persistent_incremental_attempted_batch_count << "\n";
  summary << "persistent_incremental_accepted_batch_count: "
          << result.persistent_incremental_accepted_batch_count << "\n";
  summary << "persistent_incremental_rejected_batch_count: "
          << result.persistent_incremental_rejected_batch_count << "\n";
  summary << "persistent_incremental_solver_profile_name: "
          << result.persistent_incremental_solver_profile_name << "\n";
  summary << "persistent_incremental_solver_objective_unit: "
          << result.persistent_incremental_solver_objective_unit << "\n";
  summary << "persistent_incremental_solver_max_iterations: "
          << result.persistent_incremental_solver_max_iterations << "\n";
  summary << "persistent_incremental_solver_convergence_delta_j: "
          << result.persistent_incremental_solver_convergence_delta_j << "\n";
  summary << "persistent_incremental_solver_convergence_delta_x: "
          << result.persistent_incremental_solver_convergence_delta_x << "\n";
  summary << "persistent_incremental_solver_bearing_reference_focal_px: "
          << result.persistent_incremental_solver_bearing_reference_focal_px
          << "\n";
  summary << "persistent_incremental_solver_bearing_residual_scale: "
          << result.persistent_incremental_solver_bearing_residual_scale
          << "\n";
  summary << "persistent_incremental_solver_single_iteration_batch_count: "
          << result.persistent_incremental_solver_single_iteration_batch_count
          << "\n";
  summary << "persistent_incremental_solver_max_iteration_batch_count: "
          << result.persistent_incremental_solver_max_iteration_batch_count
          << "\n";
  summary << "persistent_incremental_solver_objective_decreased_batch_count: "
          << result
                 .persistent_incremental_solver_objective_decreased_batch_count
          << "\n";
  summary << "persistent_incremental_solver_relative_objective_converged_batch_count: "
          << result
                 .persistent_incremental_solver_relative_objective_converged_batch_count
          << "\n";
  summary << "persistent_incremental_solver_camera_step_converged_batch_count: "
          << result
                 .persistent_incremental_solver_camera_step_converged_batch_count
          << "\n";
  summary << "persistent_incremental_solver_continuation_batch_count: "
          << result.persistent_incremental_solver_continuation_batch_count
          << "\n";
  summary << "persistent_incremental_solver_continuation_round_count: "
          << result.persistent_incremental_solver_continuation_round_count
          << "\n";
  summary << "persistent_incremental_solver_continuation_guard_hit_count: "
          << result.persistent_incremental_solver_continuation_guard_hit_count
          << "\n";
  summary << "persistent_incremental_total_elapsed_time_seconds: "
          << result.persistent_incremental_total_elapsed_time_seconds << "\n";
  summary << "persistent_incremental_trust_region_backtracking_batch_count: "
          << result.persistent_incremental_trust_region_backtracking_batch_count
          << "\n";
  summary << "persistent_incremental_trust_region_backtracking_attempt_count: "
          << result.persistent_incremental_trust_region_backtracking_attempt_count
          << "\n";
  summary << "persistent_incremental_trust_region_backtracking_accepted_count: "
          << result.persistent_incremental_trust_region_backtracking_accepted_count
          << "\n";
  summary << "persistent_incremental_trust_region_backtracking_max_anchor_scale: "
          << result
                 .persistent_incremental_trust_region_backtracking_max_anchor_scale
          << "\n";
  summary << "persistent_incremental_normalize_information_gain_by_board_observation: "
          << (result
                      .persistent_incremental_normalize_information_gain_by_board_observation
                  ? 1
                  : 0)
          << "\n";
  summary << "persistent_incremental_split_residual_health_gate_enabled: "
          << (result.persistent_incremental_split_residual_health_gate_enabled
                  ? 1
                  : 0)
          << "\n";
  summary << "persistent_incremental_split_residual_health_rejected_count: "
          << result
                 .persistent_incremental_split_residual_health_rejected_count
          << "\n";
  summary << "persistent_incremental_bearing_pixel_safety_gate_enabled: "
          << (result.persistent_incremental_bearing_pixel_safety_gate_enabled
                  ? 1
                  : 0)
          << "\n";
  summary << "persistent_incremental_bearing_pixel_safety_rejected_count: "
          << result.persistent_incremental_bearing_pixel_safety_rejected_count
          << "\n";
  summary
      << "persistent_incremental_full_training_pose_refit_health_gate_enabled: "
      << (result
                  .persistent_incremental_full_training_pose_refit_health_gate_enabled
              ? 1
              : 0)
      << "\n";
  summary
      << "persistent_incremental_seed_intrinsics_warmup_full_training_health_pass: "
      << (result
                  .persistent_incremental_seed_intrinsics_warmup_full_training_health_pass
              ? 1
              : 0)
      << "\n";
  summary
      << "persistent_incremental_full_training_pose_refit_health_rejected_count: "
      << result
             .persistent_incremental_full_training_pose_refit_health_rejected_count
      << "\n";
  summary << "persistent_incremental_initial_full_training_pixel_rmse: "
          << result.persistent_incremental_initial_full_training_pixel_rmse
          << "\n";
  summary << "persistent_incremental_initial_full_training_pixel_p95: "
          << result.persistent_incremental_initial_full_training_pixel_p95
          << "\n";
  summary
      << "persistent_incremental_initial_full_training_pose_success_rate: "
      << result
             .persistent_incremental_initial_full_training_pose_success_rate
      << "\n";
  summary
      << "persistent_incremental_initial_full_training_pose_success_count: "
      << result
             .persistent_incremental_initial_full_training_pose_success_count
      << "\n";
  summary << "persistent_incremental_initial_full_training_pose_total_count: "
          << result
                 .persistent_incremental_initial_full_training_pose_total_count
          << "\n";
  summary
      << "persistent_incremental_initial_full_training_invalid_projection_count: "
      << result
             .persistent_incremental_initial_full_training_invalid_projection_count
      << "\n";
  summary << "persistent_incremental_final_full_training_pixel_rmse: "
          << result.persistent_incremental_final_full_training_pixel_rmse
          << "\n";
  summary << "persistent_incremental_final_full_training_pixel_p95: "
          << result.persistent_incremental_final_full_training_pixel_p95
          << "\n";
  summary << "persistent_incremental_final_full_training_pose_success_rate: "
          << result.persistent_incremental_final_full_training_pose_success_rate
          << "\n";
  summary << "persistent_incremental_final_full_training_pose_success_count: "
          << result
                 .persistent_incremental_final_full_training_pose_success_count
          << "\n";
  summary << "persistent_incremental_final_full_training_pose_total_count: "
          << result.persistent_incremental_final_full_training_pose_total_count
          << "\n";
  summary
      << "persistent_incremental_final_full_training_invalid_projection_count: "
      << result
             .persistent_incremental_final_full_training_invalid_projection_count
      << "\n";
  summary << "persistent_incremental_kb_distortion_guard_enabled: "
          << (result.persistent_incremental_kb_distortion_guard_enabled ? 1
                                                                        : 0)
          << "\n";
  summary << "persistent_incremental_kb_ray_curve_validity_rejected_count: "
          << result.persistent_incremental_kb_ray_curve_validity_rejected_count
          << "\n";
  summary << "persistent_incremental_adaptive_saturation_stop_enabled: "
          << (result.persistent_incremental_adaptive_saturation_stop_enabled
                  ? 1
                  : 0)
          << "\n";
  summary << "persistent_incremental_adaptive_saturation_stop_hit: "
          << (result.persistent_incremental_adaptive_saturation_stop_hit ? 1
                                                                         : 0)
          << "\n";
  summary << "persistent_incremental_adaptive_saturation_min_accepted_batches: "
          << result
                 .persistent_incremental_adaptive_saturation_min_accepted_batches
          << "\n";
  summary << "persistent_incremental_adaptive_saturation_nonproductive_batch_limit: "
          << result
                 .persistent_incremental_adaptive_saturation_nonproductive_batch_limit
          << "\n";
  summary << "persistent_incremental_adaptive_saturation_consecutive_nonproductive_batches: "
          << result
                 .persistent_incremental_adaptive_saturation_consecutive_nonproductive_batches
          << "\n";
  summary << "persistent_incremental_adaptive_saturation_tail_ordering_score_threshold: "
          << result
                 .persistent_incremental_adaptive_saturation_tail_ordering_score_threshold
          << "\n";
  summary << "persistent_incremental_adaptive_saturation_next_ordering_score: "
          << result
                 .persistent_incremental_adaptive_saturation_next_ordering_score
          << "\n";
  summary << "persistent_incremental_adaptive_saturation_stop_reason: "
          << result.persistent_incremental_adaptive_saturation_stop_reason
          << "\n";
  summary << "persistent_incremental_objective_decrease_gate_enabled: "
          << (result.selection_is_kalibr_checkerboard_style ? 1 : 0)
          << "\n";
  summary << "persistent_incremental_residual_model_aware_acceptance: "
          << (result.persistent_incremental_backend_estimator_used ? 1 : 0)
          << "\n";
  summary << "persistent_intrinsics_anchor_prior_enabled: "
          << (result.persistent_intrinsics_anchor_prior_enabled ? 1 : 0)
          << "\n";
  summary << "persistent_intrinsics_anchor_weight_xi_alpha: "
          << result.persistent_intrinsics_anchor_weight_xi_alpha << "\n";
  summary << "persistent_intrinsics_anchor_weight_focal: "
          << result.persistent_intrinsics_anchor_weight_focal << "\n";
  summary << "persistent_intrinsics_anchor_weight_principal: "
          << result.persistent_intrinsics_anchor_weight_principal << "\n";
  const bool explicit_anchor_weights =
      result.persistent_intrinsics_anchor_weight_xi_alpha > 0.0 ||
      result.persistent_intrinsics_anchor_weight_focal > 0.0 ||
      result.persistent_intrinsics_anchor_weight_principal > 0.0;
  const bool explicit_anchor_scales =
      result.persistent_max_focal_relative_step > 0.0 ||
      result.persistent_max_principal_step_px > 0.0 ||
      result.persistent_max_xi_alpha_step > 0.0;
  summary << "persistent_intrinsics_anchor_weight_mode: "
          << (!result.persistent_intrinsics_anchor_prior_enabled
                  ? "disabled"
                  : (explicit_anchor_weights
                         ? "explicit_weights"
                         : (explicit_anchor_scales
                                ? "configured_step_scale"
                                : "requested_but_no_weights_or_step")))
          << "\n";
  summary << "persistent_max_focal_relative_step: "
          << result.persistent_max_focal_relative_step << "\n";
  summary << "persistent_max_principal_step_px: "
          << result.persistent_max_principal_step_px << "\n";
  summary << "persistent_max_xi_alpha_step: "
          << result.persistent_max_xi_alpha_step << "\n";
	  if (result.persistent_incremental_backend_estimator_used) {
	    summary << "kalibr_style_acceptance_note: legacy_rmse_pass is residual "
	            << "health only; persistent incremental acceptance is decided "
	            << "from committed optimizer state, residual-model-aware "
	            << "health, information/rank gain, optimizer objective validity, "
	            << "finite/model-bound state checks, and explicit rollback. "
            << "Camera parameter step thresholds are not acceptance gates; "
            << "they are disabled by default and only reported when explicitly "
            << "configured, matching Kalibr's acceptance boundary more closely; "
	            << "pixel mode keeps the information/rank gate and angular mode "
	            << "uses tangent-plane angular health for acceptance.\n";
  } else {
    summary << "kalibr_style_acceptance_note: legacy_rmse_pass is residual "
            << "health only; proxy-path acceptance uses the configured "
            << "intrinsics information-gain proxy and batch acceptance policy.\n";
  }
  summary << "frame_cohesion_candidate_count: "
          << result.frame_cohesion_candidate_count << "\n";
  summary << "frame_cohesion_attempted_count: "
          << result.frame_cohesion_attempted_count << "\n";
  summary << "frame_cohesion_accepted_count: "
          << result.frame_cohesion_accepted_count << "\n";
  summary << "frame_cohesion_rejected_count: "
          << result.frame_cohesion_rejected_count << "\n";
  summary << "frame_batch_candidate_count: "
          << result.frame_batch_candidate_count << "\n";
  summary << "frame_batch_attempted_count: "
          << result.frame_batch_attempted_count << "\n";
  summary << "frame_batch_accepted_count: "
          << result.frame_batch_accepted_count << "\n";
  summary << "frame_batch_rejected_count: "
          << result.frame_batch_rejected_count << "\n";
  summary << "persistent_incremental_image_plane_residual_count: "
          << result.persistent_incremental_image_plane_residual_count << "\n";
	  summary << "persistent_incremental_angular_residual_count: "
	          << result.persistent_incremental_angular_residual_count << "\n";
	  summary << "persistent_incremental_chordal_residual_count: "
	          << result.persistent_incremental_chordal_residual_count << "\n";
	  summary << "persistent_incremental_hybrid_angular_selected_count: "
	          << result.persistent_incremental_hybrid_angular_selected_count
	          << "\n";
	  summary << "persistent_incremental_hybrid_chordal_selected_count: "
	          << result.persistent_incremental_hybrid_chordal_selected_count
	          << "\n";
	  summary << "persistent_incremental_angular_geometry_failure_count: "
	          << result.persistent_incremental_angular_geometry_failure_count
	          << "\n";
  summary << "persistent_incremental_angular_local_whitening_success_count: "
          << result.persistent_incremental_angular_local_whitening_success_count
          << "\n";
  summary << "persistent_incremental_angular_local_whitening_failure_count: "
          << result.persistent_incremental_angular_local_whitening_failure_count
          << "\n";
  summary << "persistent_incremental_angular_local_whitening_clamped_count: "
          << result.persistent_incremental_angular_local_whitening_clamped_count
          << "\n";
  summary << "persistent_incremental_angular_local_whitening_sigma_mean_rad: "
          << result.persistent_incremental_angular_local_whitening_sigma_mean_rad
          << "\n";
  summary << "persistent_incremental_angular_local_whitening_sigma_min_rad: "
          << result.persistent_incremental_angular_local_whitening_sigma_min_rad
          << "\n";
  summary << "persistent_incremental_angular_local_whitening_sigma_max_rad: "
          << result.persistent_incremental_angular_local_whitening_sigma_max_rad
          << "\n";
  summary << "persistent_incremental_angular_local_whitening_weight_mean: "
          << result.persistent_incremental_angular_local_whitening_weight_mean
          << "\n";
  summary << "persistent_incremental_angular_local_whitening_weight_min: "
          << result.persistent_incremental_angular_local_whitening_weight_min
          << "\n";
  summary << "persistent_incremental_angular_local_whitening_weight_max: "
          << result.persistent_incremental_angular_local_whitening_weight_max
          << "\n";
  summary << "persistent_incremental_selection_metric_name: "
          << result.persistent_incremental_selection_metric_name << "\n";
  summary << "persistent_incremental_bearing_geometry_source: "
          << "active_camera_model\n";
  summary << "persistent_incremental_selection_metric_unit: "
          << result.persistent_incremental_selection_metric_unit << "\n";
  summary << "persistent_incremental_residual_health_threshold_source: "
          << result.persistent_incremental_residual_health_threshold_source
          << "\n";
  summary << "persistent_incremental_residual_health_threshold_metric: "
          << result.persistent_incremental_residual_health_threshold_metric
          << "\n";
  summary << "persistent_incremental_seed_acceptance_metric_rmse: "
          << result.persistent_incremental_seed_acceptance_metric_rmse << "\n";
  summary << "persistent_incremental_seed_acceptance_metric_p95: "
          << result.persistent_incremental_seed_acceptance_metric_p95 << "\n";
  summary << "frame_consolidation_candidate_count: "
          << result.frame_consolidation_candidate_count << "\n";
  summary << "frame_consolidation_accepted_count: "
          << result.frame_consolidation_accepted_count << "\n";
  summary << "frame_consolidation_rejected_count: "
          << result.frame_consolidation_rejected_count << "\n";
  summary << "frame_consolidation_dropped_board_observation_count: "
          << result.frame_consolidation_dropped_board_observation_count
          << "\n";
  summary << "close_distance_candidate_count: "
          << result.close_distance_candidate_count << "\n";
  summary << "close_distance_accepted_count: "
          << result.close_distance_accepted_count << "\n";
  summary << "close_distance_frame_admission_candidate_count: "
          << result.close_distance_frame_admission_candidate_count << "\n";
  summary << "close_distance_frame_admission_attempted_count: "
          << result.close_distance_frame_admission_attempted_count << "\n";
  summary << "close_distance_frame_admission_accepted_count: "
          << result.close_distance_frame_admission_accepted_count << "\n";
  summary << "close_distance_frame_admission_rejected_count: "
          << result.close_distance_frame_admission_rejected_count << "\n";
  summary << "intrinsics_diversity_anchor_candidate_count: "
          << result.intrinsics_diversity_anchor_candidate_count << "\n";
  summary << "intrinsics_diversity_anchor_accepted_count: "
          << result.intrinsics_diversity_anchor_accepted_count << "\n";
  summary << "intrinsics_diversity_anchor_rejected_count: "
          << result.intrinsics_diversity_anchor_rejected_count << "\n";
  summary << "close_edge_hard_case_count: "
          << result.close_edge_hard_case_count << "\n";
  summary << "close_edge_soft_candidate_count: "
          << result.close_edge_soft_candidate_count << "\n";
  summary << "close_edge_soft_attempted_count: "
          << result.close_edge_soft_attempted_count << "\n";
  summary << "close_edge_soft_accepted_count: "
          << result.close_edge_soft_accepted_count << "\n";
  summary << "close_edge_soft_rejected_count: "
          << result.close_edge_soft_rejected_count << "\n";
  summary << "note: candidate_score/coverage_gain and cap parameters are "
          << "recorded per observation in the decisions CSV.\n";
  summary << "kept_frame_count: " << result.kept_frame_count << "\n";
  summary << "kept_board_observation_count: "
          << result.kept_board_observation_count << "\n";
  summary << "kept_outer_point_count: " << result.kept_outer_point_count
          << "\n";
  summary << "kept_internal_point_count: " << result.kept_internal_point_count
          << "\n";
  summary << "kept_total_point_count: " << result.kept_total_point_count
          << "\n";
  summary << "rejected_board_observation_count: "
          << result.rejected_board_observation_count << "\n";
  summary << "median_board_rmse: " << result.median_board_rmse << "\n";
  summary << "robust_sigma_board_rmse: "
          << result.robust_sigma_board_rmse << "\n";
  summary << "threshold_px: " << result.threshold_px << "\n";
  summary << "trial_backend_success: "
          << (result.trial_backend_result.success ? 1 : 0) << "\n";
  summary << "trial_backend_initial_rmse: "
          << result.trial_backend_result.initial_residual.overall_rmse
          << "\n";
  summary << "trial_backend_optimized_rmse: "
          << result.trial_backend_result.optimized_residual.overall_rmse
          << "\n";
  summary << "\nwarnings:\n";
  for (const std::string& warning : result.warnings) {
    summary << "  " << warning << "\n";
  }

  std::ofstream csv(
      (output_dir / "trial_backend_frame_board_selection_decisions.csv")
          .string()
          .c_str());
  csv << "frame_index,frame_label,board_id,baseline_seed,"
      << "attempted_incremental,kept,force_include_candidate,reason,"
      << "trial_rmse,threshold_px,"
      << "candidate_score,coverage_gain,polar_gain,edge_gain,"
      << "board_balance_gain,frame_novelty_gain,grid_gain,"
      << "covisibility_gain,residual_quality_score,"
      << "consistency_available,consistency_score,consistency_penalty,"
      << "consistency_translation_error_mm,consistency_rotation_error_deg,"
      << "consistency_local_outer_rmse,"
      << "mean_polar_angle_deg,max_polar_angle_deg,"
      << "intrinsics_diversity_anchor,"
      << "close_distance_candidate,close_distance_score_bonus,"
      << "close_distance_frame_admission_candidate,"
      << "close_distance_frame_admission_attempted,"
      << "close_distance_frame_admission_accepted,"
      << "is_close_edge_hard_case,projected_area_px,projected_area_ratio,"
      << "outer_pose_refit_rmse,close_edge_score,"
      << "frame_cohesion_candidate,frame_cohesion_attempted,"
      << "frame_cohesion_accepted,"
      << "frame_batch_candidate,frame_batch_attempted,"
      << "frame_batch_accepted,"
      << "frame_consolidation_candidate,"
      << "frame_consolidation_accepted,"
      << "soft_candidate,soft_attempted,soft_accepted,soft_weight,"
      << "soft_global_rmse_delta,soft_outer_rmse_delta,"
      << "soft_internal_rmse_delta,"
      << "left_rmse,right_rmse,center_side_rmse,edge_side_rmse,"
      << "global_rmse_before,global_rmse_after,"
      << "global_rmse_delta,outer_rmse_delta,internal_rmse_delta,"
      << "selection_mode,hard_validity_pass,legacy_rmse_pass,"
      << "catastrophic_residual,score_term,coverage_term,"
      << "intrinsics_jacobian_logdet_gain,"
      << "intrinsics_jacobian_trace_gain,"
      << "intrinsics_jacobian_rank_gain,"
      << "intrinsics_jacobian_info_term,"
      << "frame_completion_bonus,new_board_bonus,cap_penalty,"
      << "information_gain_proxy,residual_overage_penalty,"
      << "batch_acceptance_score,accepted_by_batch_acceptance,"
      << "persistent_incremental_attempted,"
      << "persistent_incremental_attempt_order,"
      << "persistent_incremental_batch_accepted,"
      << "persistent_incremental_force,"
      << "persistent_incremental_trust_region_pass,"
      << "persistent_incremental_trust_region_backtracking_used,"
      << "persistent_incremental_split_residual_health_pass,"
      << "persistent_incremental_pixel_safety_gate_pass,"
      << "persistent_incremental_full_training_pose_refit_health_pass,"
      << "persistent_incremental_ray_curve_validity_pass,"
      << "persistent_incremental_kb_k3_released,"
      << "persistent_incremental_kb_k4_released,"
      << "persistent_incremental_information_gain,"
      << "persistent_incremental_normalized_information_gain,"
      << "persistent_incremental_information_gain_normalization_count,"
      << "persistent_incremental_rank_psi_after,"
      << "persistent_incremental_rank_psi_deficiency_after,"
      << "persistent_incremental_rank_theta_before,"
      << "persistent_incremental_rank_theta_after,"
      << "persistent_incremental_rank_theta_deficiency_after,"
      << "persistent_incremental_svd_tolerance,"
      << "persistent_incremental_qr_tolerance,"
      << "persistent_incremental_iterations,"
      << "persistent_incremental_last_solver_pass_iterations,"
      << "persistent_incremental_continuation_round_count,"
      << "persistent_incremental_continuation_guard_hit,"
      << "persistent_incremental_pose_prefit_attempted,"
      << "persistent_incremental_pose_prefit_success,"
      << "persistent_incremental_pose_prefit_iterations,"
      << "persistent_incremental_pose_prefit_objective_start,"
      << "persistent_incremental_pose_prefit_objective_final,"
      << "persistent_incremental_pose_prefit_last_delta_j,"
      << "persistent_incremental_pose_prefit_last_delta_x,"
      << "persistent_incremental_objective_start,"
      << "persistent_incremental_objective_final,"
      << "persistent_incremental_objective_last_delta_j,"
      << "persistent_incremental_state_last_delta_x,"
      << "persistent_incremental_linear_solver_failure,"
      << "persistent_incremental_converged_by_relative_objective,"
      << "persistent_incremental_converged_by_camera_step,"
      << "persistent_incremental_last_camera_shape_step,"
      << "persistent_incremental_last_camera_focal_relative_step,"
      << "persistent_incremental_last_camera_principal_step_px,"
      << "persistent_incremental_objective_decreased,"
      << "persistent_incremental_rmse_before,"
      << "persistent_incremental_outer_rmse_before,"
      << "persistent_incremental_internal_rmse_before,"
      << "persistent_incremental_acceptance_metric_name,"
      << "persistent_incremental_acceptance_metric_unit,"
      << "persistent_incremental_acceptance_metric_threshold,"
      << "persistent_incremental_acceptance_metric_before,"
      << "persistent_incremental_acceptance_metric_after,"
      << "persistent_incremental_acceptance_metric_candidate,"
      << "persistent_incremental_acceptance_metric_candidate_p95,"
      << "persistent_incremental_acceptance_metric_candidate_outer,"
      << "persistent_incremental_acceptance_metric_candidate_internal,"
      << "persistent_incremental_total_p95_after,"
      << "persistent_incremental_outer_p95_after,"
      << "persistent_incremental_internal_p95_after,"
      << "persistent_incremental_candidate_rmse_after,"
      << "persistent_incremental_candidate_outer_rmse_after,"
      << "persistent_incremental_candidate_internal_rmse_after,"
      << "persistent_incremental_candidate_total_p95_after,"
      << "persistent_incremental_candidate_outer_p95_after,"
      << "persistent_incremental_candidate_internal_p95_after,"
      << "persistent_incremental_pixel_rmse_before,"
      << "persistent_incremental_pixel_rmse_after,"
      << "persistent_incremental_pixel_p95_before,"
      << "persistent_incremental_pixel_p95_after,"
      << "persistent_incremental_candidate_pixel_rmse_after,"
      << "persistent_incremental_candidate_pixel_p95_after,"
      << "persistent_incremental_full_training_pixel_rmse_before,"
      << "persistent_incremental_full_training_pixel_rmse_after,"
      << "persistent_incremental_full_training_pixel_p95_before,"
      << "persistent_incremental_full_training_pixel_p95_after,"
      << "persistent_incremental_full_training_pose_success_rate_before,"
      << "persistent_incremental_full_training_pose_success_rate_after,"
      << "persistent_incremental_full_training_pose_success_count_before,"
      << "persistent_incremental_full_training_pose_success_count_after,"
      << "persistent_incremental_full_training_pose_total_count,"
      << "persistent_incremental_full_training_invalid_projection_count_before,"
      << "persistent_incremental_full_training_invalid_projection_count_after,"
      << "persistent_incremental_ray_curve_rms_change_deg,"
      << "persistent_incremental_ray_curve_max_change_deg,"
      << "persistent_incremental_ray_curve_min_radial_derivative,"
	      << "persistent_incremental_image_plane_residual_count,"
	      << "persistent_incremental_angular_residual_count,"
	      << "persistent_incremental_chordal_residual_count,"
	      << "persistent_incremental_hybrid_angular_selected_count,"
	      << "persistent_incremental_hybrid_chordal_selected_count,"
	      << "persistent_incremental_angular_geometry_failure_count,"
      << "persistent_incremental_trust_region_retry_count,"
      << "persistent_incremental_trust_region_violation_ratio,"
      << "persistent_incremental_trust_region_anchor_weight_scale,"
      << "persistent_incremental_elapsed_time_seconds,"
      << "persistent_incremental_commit_state,"
      << "persistent_incremental_camera_xi_before,"
      << "persistent_incremental_camera_alpha_before,"
      << "persistent_incremental_camera_fu_before,"
      << "persistent_incremental_camera_fv_before,"
      << "persistent_incremental_camera_cu_before,"
      << "persistent_incremental_camera_cv_before,"
      << "persistent_incremental_camera_xi_after,"
      << "persistent_incremental_camera_alpha_after,"
      << "persistent_incremental_camera_fu_after,"
      << "persistent_incremental_camera_fv_after,"
      << "persistent_incremental_camera_cu_after,"
      << "persistent_incremental_camera_cv_after,"
      << "persistent_incremental_camera_k1_before,"
      << "persistent_incremental_camera_k2_before,"
      << "persistent_incremental_camera_k3_before,"
      << "persistent_incremental_camera_k4_before,"
      << "persistent_incremental_camera_k1_after,"
      << "persistent_incremental_camera_k2_after,"
      << "persistent_incremental_camera_k3_after,"
      << "persistent_incremental_camera_k4_after,"
      << "point_count,outer_point_count,internal_point_count\n";
  for (const ati::TrialBackendFrameBoardObservationDecision& decision :
       result.decisions) {
    csv << decision.frame_index << ","
        << CsvEscape(decision.frame_label) << ","
        << decision.board_id << ","
        << (decision.baseline_seed ? 1 : 0) << ","
        << (decision.attempted_incremental ? 1 : 0) << ","
        << (decision.kept ? 1 : 0) << ","
        << (decision.force_include_candidate ? 1 : 0) << ","
        << CsvEscape(decision.reason) << ","
        << decision.trial_rmse << ","
        << result.threshold_px << ","
        << decision.candidate_score << ","
        << decision.coverage_gain << ","
        << decision.polar_gain << ","
        << decision.edge_gain << ","
        << decision.board_balance_gain << ","
        << decision.frame_novelty_gain << ","
        << decision.grid_gain << ","
        << decision.covisibility_gain << ","
        << decision.residual_quality_score << ","
        << (decision.consistency_available ? 1 : 0) << ","
        << decision.consistency_score << ","
        << decision.consistency_penalty << ","
        << decision.consistency_translation_error_mm << ","
        << decision.consistency_rotation_error_deg << ","
        << decision.consistency_local_outer_rmse << ","
        << decision.mean_polar_angle_deg << ","
        << decision.max_polar_angle_deg << ","
        << (decision.intrinsics_diversity_anchor ? 1 : 0) << ","
        << (decision.close_distance_candidate ? 1 : 0) << ","
        << decision.close_distance_score_bonus << ","
        << (decision.close_distance_frame_admission_candidate ? 1 : 0) << ","
        << (decision.close_distance_frame_admission_attempted ? 1 : 0) << ","
        << (decision.close_distance_frame_admission_accepted ? 1 : 0) << ","
        << (decision.is_close_edge_hard_case ? 1 : 0) << ","
        << decision.projected_area_px << ","
        << decision.projected_area_ratio << ","
        << decision.outer_pose_refit_rmse << ","
        << decision.close_edge_score << ","
        << (decision.frame_cohesion_candidate ? 1 : 0) << ","
        << (decision.frame_cohesion_attempted ? 1 : 0) << ","
        << (decision.frame_cohesion_accepted ? 1 : 0) << ","
        << (decision.frame_batch_candidate ? 1 : 0) << ","
        << (decision.frame_batch_attempted ? 1 : 0) << ","
        << (decision.frame_batch_accepted ? 1 : 0) << ","
        << (decision.frame_consolidation_candidate ? 1 : 0) << ","
        << (decision.frame_consolidation_accepted ? 1 : 0) << ","
        << (decision.soft_candidate ? 1 : 0) << ","
        << (decision.soft_attempted ? 1 : 0) << ","
        << (decision.soft_accepted ? 1 : 0) << ","
        << decision.soft_weight << ","
        << decision.soft_global_rmse_delta << ","
        << decision.soft_outer_rmse_delta << ","
        << decision.soft_internal_rmse_delta << ","
        << decision.left_rmse << ","
        << decision.right_rmse << ","
        << decision.center_side_rmse << ","
        << decision.edge_side_rmse << ","
        << decision.global_rmse_before << ","
        << decision.global_rmse_after << ","
        << decision.global_rmse_delta << ","
        << decision.outer_rmse_delta << ","
        << decision.internal_rmse_delta << ","
        << ati::ToString(result.selection_mode) << ","
        << (decision.hard_validity_pass ? 1 : 0) << ","
        << (decision.legacy_rmse_pass ? 1 : 0) << ","
        << (decision.catastrophic_residual ? 1 : 0) << ","
        << decision.score_term << ","
        << decision.coverage_term << ","
        << decision.intrinsics_jacobian_logdet_gain << ","
        << decision.intrinsics_jacobian_trace_gain << ","
        << decision.intrinsics_jacobian_rank_gain << ","
        << decision.intrinsics_jacobian_info_term << ","
        << decision.frame_completion_bonus << ","
        << decision.new_board_bonus << ","
        << decision.cap_penalty << ","
        << decision.information_gain_proxy << ","
        << decision.residual_overage_penalty << ","
        << decision.batch_acceptance_score << ","
        << (decision.accepted_by_batch_acceptance ? 1 : 0) << ","
        << (decision.persistent_incremental_attempted ? 1 : 0) << ","
        << decision.persistent_incremental_attempt_order << ","
        << (decision.persistent_incremental_batch_accepted ? 1 : 0) << ","
        << (decision.persistent_incremental_force ? 1 : 0) << ","
        << (decision.persistent_incremental_trust_region_pass ? 1 : 0)
        << ","
        << (decision.persistent_incremental_trust_region_backtracking_used ? 1
                                                                           : 0)
        << ","
        << (decision.persistent_incremental_split_residual_health_pass ? 1
                                                                       : 0)
        << ","
        << (decision.persistent_incremental_pixel_safety_gate_pass ? 1 : 0)
        << ","
        << (decision
                    .persistent_incremental_full_training_pose_refit_health_pass
                ? 1
                : 0)
        << ","
        << (decision.persistent_incremental_ray_curve_validity_pass ? 1 : 0)
        << ","
        << (decision.persistent_incremental_kb_k3_released ? 1 : 0) << ","
        << (decision.persistent_incremental_kb_k4_released ? 1 : 0) << ","
        << decision.persistent_incremental_information_gain << ","
        << decision.persistent_incremental_normalized_information_gain << ","
        << decision
               .persistent_incremental_information_gain_normalization_count
        << ","
        << decision.persistent_incremental_rank_psi_after << ","
        << decision.persistent_incremental_rank_psi_deficiency_after << ","
        << decision.persistent_incremental_rank_theta_before << ","
        << decision.persistent_incremental_rank_theta_after << ","
        << decision.persistent_incremental_rank_theta_deficiency_after << ","
        << decision.persistent_incremental_svd_tolerance << ","
        << decision.persistent_incremental_qr_tolerance << ","
        << decision.persistent_incremental_iterations << ","
        << decision.persistent_incremental_last_solver_pass_iterations << ","
        << decision.persistent_incremental_continuation_round_count << ","
        << (decision.persistent_incremental_continuation_guard_hit ? 1 : 0)
        << ","
        << (decision.persistent_incremental_pose_prefit_attempted ? 1 : 0)
        << ","
        << (decision.persistent_incremental_pose_prefit_success ? 1 : 0)
        << ","
        << decision.persistent_incremental_pose_prefit_iterations << ","
        << decision.persistent_incremental_pose_prefit_objective_start << ","
        << decision.persistent_incremental_pose_prefit_objective_final << ","
        << decision.persistent_incremental_pose_prefit_last_delta_j << ","
        << decision.persistent_incremental_pose_prefit_last_delta_x << ","
        << decision.persistent_incremental_objective_start << ","
        << decision.persistent_incremental_objective_final << ","
        << decision.persistent_incremental_objective_last_delta_j << ","
        << decision.persistent_incremental_state_last_delta_x << ","
        << (decision.persistent_incremental_linear_solver_failure ? 1 : 0)
        << ","
        << (decision.persistent_incremental_converged_by_relative_objective
                ? 1
                : 0)
        << ","
        << (decision.persistent_incremental_converged_by_camera_step ? 1 : 0)
        << ","
        << decision.persistent_incremental_last_camera_shape_step << ","
        << decision.persistent_incremental_last_camera_focal_relative_step
        << ","
        << decision.persistent_incremental_last_camera_principal_step_px
        << ","
        << (decision.persistent_incremental_objective_decreased ? 1 : 0)
        << ","
        << decision.persistent_incremental_rmse_before << ","
        << decision.persistent_incremental_outer_rmse_before << ","
        << decision.persistent_incremental_internal_rmse_before << ","
        << CsvEscape(decision.persistent_incremental_acceptance_metric_name)
        << ","
        << CsvEscape(decision.persistent_incremental_acceptance_metric_unit)
        << ","
        << decision.persistent_incremental_acceptance_metric_threshold << ","
        << decision.persistent_incremental_acceptance_metric_before << ","
        << decision.persistent_incremental_acceptance_metric_after << ","
        << decision.persistent_incremental_acceptance_metric_candidate << ","
        << decision.persistent_incremental_acceptance_metric_candidate_p95
        << ","
        << decision.persistent_incremental_acceptance_metric_candidate_outer
        << ","
        << decision
               .persistent_incremental_acceptance_metric_candidate_internal
        << ","
        << decision.persistent_incremental_total_p95_after << ","
        << decision.persistent_incremental_outer_p95_after << ","
        << decision.persistent_incremental_internal_p95_after << ","
        << decision.persistent_incremental_candidate_rmse_after << ","
        << decision.persistent_incremental_candidate_outer_rmse_after << ","
        << decision.persistent_incremental_candidate_internal_rmse_after << ","
        << decision.persistent_incremental_candidate_total_p95_after << ","
        << decision.persistent_incremental_candidate_outer_p95_after << ","
        << decision.persistent_incremental_candidate_internal_p95_after << ","
        << decision.persistent_incremental_pixel_rmse_before << ","
        << decision.persistent_incremental_pixel_rmse_after << ","
        << decision.persistent_incremental_pixel_p95_before << ","
        << decision.persistent_incremental_pixel_p95_after << ","
        << decision.persistent_incremental_candidate_pixel_rmse_after << ","
        << decision.persistent_incremental_candidate_pixel_p95_after << ","
        << decision.persistent_incremental_full_training_pixel_rmse_before
        << ","
        << decision.persistent_incremental_full_training_pixel_rmse_after
        << ","
        << decision.persistent_incremental_full_training_pixel_p95_before
        << ","
        << decision.persistent_incremental_full_training_pixel_p95_after
        << ","
        << decision
               .persistent_incremental_full_training_pose_success_rate_before
        << ","
        << decision
               .persistent_incremental_full_training_pose_success_rate_after
        << ","
        << decision
               .persistent_incremental_full_training_pose_success_count_before
        << ","
        << decision
               .persistent_incremental_full_training_pose_success_count_after
        << ","
        << decision.persistent_incremental_full_training_pose_total_count
        << ","
        << decision
               .persistent_incremental_full_training_invalid_projection_count_before
        << ","
        << decision
               .persistent_incremental_full_training_invalid_projection_count_after
        << ","
        << decision.persistent_incremental_ray_curve_rms_change_deg << ","
        << decision.persistent_incremental_ray_curve_max_change_deg << ","
        << decision.persistent_incremental_ray_curve_min_radial_derivative
        << ","
	        << decision.persistent_incremental_image_plane_residual_count << ","
	        << decision.persistent_incremental_angular_residual_count << ","
	        << decision.persistent_incremental_chordal_residual_count << ","
	        << decision.persistent_incremental_hybrid_angular_selected_count
	        << ","
	        << decision.persistent_incremental_hybrid_chordal_selected_count
	        << ","
	        << decision.persistent_incremental_angular_geometry_failure_count
	        << ","
        << decision.persistent_incremental_trust_region_retry_count << ","
        << decision.persistent_incremental_trust_region_violation_ratio << ","
        << decision.persistent_incremental_trust_region_anchor_weight_scale
        << ","
        << decision.persistent_incremental_elapsed_time_seconds << ","
        << CsvEscape(decision.persistent_incremental_commit_state) << ","
        << decision.persistent_incremental_camera_xi_before << ","
        << decision.persistent_incremental_camera_alpha_before << ","
        << decision.persistent_incremental_camera_fu_before << ","
        << decision.persistent_incremental_camera_fv_before << ","
        << decision.persistent_incremental_camera_cu_before << ","
        << decision.persistent_incremental_camera_cv_before << ","
        << decision.persistent_incremental_camera_xi_after << ","
        << decision.persistent_incremental_camera_alpha_after << ","
        << decision.persistent_incremental_camera_fu_after << ","
        << decision.persistent_incremental_camera_fv_after << ","
        << decision.persistent_incremental_camera_cu_after << ","
        << decision.persistent_incremental_camera_cv_after << ","
        << decision.persistent_incremental_camera_k1_before << ","
        << decision.persistent_incremental_camera_k2_before << ","
        << decision.persistent_incremental_camera_k3_before << ","
        << decision.persistent_incremental_camera_k4_before << ","
        << decision.persistent_incremental_camera_k1_after << ","
        << decision.persistent_incremental_camera_k2_after << ","
        << decision.persistent_incremental_camera_k3_after << ","
        << decision.persistent_incremental_camera_k4_after << ","
        << decision.point_count << ","
        << decision.outer_point_count << ","
        << decision.internal_point_count << "\n";
  }

  std::ofstream close_edge_csv(
      (output_dir / "close_edge_candidate_diagnostics.csv")
          .string()
          .c_str());
  close_edge_csv
      << "frame_index,frame_label,board_id,is_close_edge_hard_case,"
      << "soft_candidate,soft_attempted,soft_accepted,reason,"
      << "projected_area_px,projected_area_ratio,mean_polar_angle_deg,"
      << "max_polar_angle_deg,outer_pose_refit_rmse,close_edge_score,"
      << "trial_rmse,left_rmse,right_rmse,center_side_rmse,edge_side_rmse,"
      << "soft_weight,soft_global_rmse_delta,soft_outer_rmse_delta,"
      << "soft_internal_rmse_delta,point_count,outer_point_count,"
      << "internal_point_count\n";
  std::vector<ati::TrialBackendFrameBoardObservationDecision> close_edge_rows;
  for (const ati::TrialBackendFrameBoardObservationDecision& decision :
       result.decisions) {
    if (decision.is_close_edge_hard_case || decision.soft_candidate ||
        decision.soft_attempted || decision.soft_accepted) {
      close_edge_rows.push_back(decision);
    }
  }
  std::sort(close_edge_rows.begin(),
            close_edge_rows.end(),
            [](const ati::TrialBackendFrameBoardObservationDecision& lhs,
               const ati::TrialBackendFrameBoardObservationDecision& rhs) {
              if (lhs.close_edge_score != rhs.close_edge_score) {
                return lhs.close_edge_score > rhs.close_edge_score;
              }
              if (lhs.outer_pose_refit_rmse != rhs.outer_pose_refit_rmse) {
                return lhs.outer_pose_refit_rmse > rhs.outer_pose_refit_rmse;
              }
              return lhs.trial_rmse > rhs.trial_rmse;
            });
  for (const ati::TrialBackendFrameBoardObservationDecision& decision :
       close_edge_rows) {
    close_edge_csv
        << decision.frame_index << ","
        << CsvEscape(decision.frame_label) << ","
        << decision.board_id << ","
        << (decision.is_close_edge_hard_case ? 1 : 0) << ","
        << (decision.soft_candidate ? 1 : 0) << ","
        << (decision.soft_attempted ? 1 : 0) << ","
        << (decision.soft_accepted ? 1 : 0) << ","
        << CsvEscape(decision.reason) << ","
        << decision.projected_area_px << ","
        << decision.projected_area_ratio << ","
        << decision.mean_polar_angle_deg << ","
        << decision.max_polar_angle_deg << ","
        << decision.outer_pose_refit_rmse << ","
        << decision.close_edge_score << ","
        << decision.trial_rmse << ","
        << decision.left_rmse << ","
        << decision.right_rmse << ","
        << decision.center_side_rmse << ","
        << decision.edge_side_rmse << ","
        << decision.soft_weight << ","
        << decision.soft_global_rmse_delta << ","
        << decision.soft_outer_rmse_delta << ","
        << decision.soft_internal_rmse_delta << ","
        << decision.point_count << ","
        << decision.outer_point_count << ","
        << decision.internal_point_count << "\n";
  }

  std::ofstream close_edge_summary(
      (output_dir / "close_edge_candidate_summary.txt")
          .string()
          .c_str());
  close_edge_summary << "close_edge_hard_case_count: "
                     << result.close_edge_hard_case_count << "\n";
  close_edge_summary << "close_edge_soft_candidate_count: "
                     << result.close_edge_soft_candidate_count << "\n";
  close_edge_summary << "close_edge_soft_attempted_count: "
                     << result.close_edge_soft_attempted_count << "\n";
  close_edge_summary << "close_edge_soft_accepted_count: "
                     << result.close_edge_soft_accepted_count << "\n";
  close_edge_summary << "close_edge_soft_rejected_count: "
                     << result.close_edge_soft_rejected_count << "\n";
  close_edge_summary << "diagnostics_csv: close_edge_candidate_diagnostics.csv\n";

  std::ofstream top_cases(
      (output_dir / "close_edge_top_hard_cases.csv")
          .string()
          .c_str());
  top_cases
      << "rank,frame_index,frame_label,board_id,close_edge_score,"
      << "projected_area_ratio,max_polar_angle_deg,outer_pose_refit_rmse,"
      << "trial_rmse,reason,soft_attempted,soft_accepted\n";
  const int top_k = std::min<int>(30, close_edge_rows.size());
  for (int index = 0; index < top_k; ++index) {
    const ati::TrialBackendFrameBoardObservationDecision& decision =
        close_edge_rows[index];
    top_cases
        << (index + 1) << ","
        << decision.frame_index << ","
        << CsvEscape(decision.frame_label) << ","
        << decision.board_id << ","
        << decision.close_edge_score << ","
        << decision.projected_area_ratio << ","
        << decision.max_polar_angle_deg << ","
        << decision.outer_pose_refit_rmse << ","
        << decision.trial_rmse << ","
        << CsvEscape(decision.reason) << ","
        << (decision.soft_attempted ? 1 : 0) << ","
        << (decision.soft_accepted ? 1 : 0) << "\n";
  }

  std::ofstream soft_summary(
      (output_dir / "close_edge_soft_use_summary.txt")
          .string()
          .c_str());
  soft_summary << "soft_candidate_count: "
               << result.close_edge_soft_candidate_count << "\n";
  soft_summary << "soft_attempted_count: "
               << result.close_edge_soft_attempted_count << "\n";
  soft_summary << "soft_accepted_count: "
               << result.close_edge_soft_accepted_count << "\n";
  soft_summary << "soft_rejected_count: "
               << result.close_edge_soft_rejected_count << "\n";

  std::ofstream soft_weights(
      (output_dir / "close_edge_soft_weight_assignments.csv")
          .string()
          .c_str());
  soft_weights
      << "frame_index,frame_label,board_id,soft_weight,soft_accepted,"
      << "soft_global_rmse_delta,soft_outer_rmse_delta,"
      << "soft_internal_rmse_delta,reason\n";
  for (const ati::TrialBackendFrameBoardObservationDecision& decision :
       result.decisions) {
    if (!decision.soft_candidate && !decision.soft_attempted &&
        !decision.soft_accepted) {
      continue;
    }
    soft_weights
        << decision.frame_index << ","
        << CsvEscape(decision.frame_label) << ","
        << decision.board_id << ","
        << decision.soft_weight << ","
        << (decision.soft_accepted ? 1 : 0) << ","
        << decision.soft_global_rmse_delta << ","
        << decision.soft_outer_rmse_delta << ","
        << decision.soft_internal_rmse_delta << ","
        << CsvEscape(decision.reason) << "\n";
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
