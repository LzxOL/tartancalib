#include <aslam/cameras/apriltag_internal/MultiBoardCoObservationConsistency.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <boost/filesystem.hpp>
#include <opencv2/calib3d.hpp>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

namespace fs = boost::filesystem;

constexpr double kRadToDeg = 180.0 / M_PI;

struct GroupKey {
  std::string split;
  int pair_index = -1;
  int camera_index = -1;
  int board_id = -1;

  bool operator<(const GroupKey& other) const {
    return std::tie(split, pair_index, camera_index, board_id) <
           std::tie(other.split, other.pair_index, other.camera_index,
                    other.board_id);
  }
};

struct LocalGroup {
  GroupKey key;
  std::vector<const StereoObservation*> observations;
  int num_corners = 0;
  int outer_corner_count = 0;
  int internal_corner_count = 0;
  int num_highpolar_50plus = 0;
  int num_highpolar_70plus = 0;
  bool selected_pair_board_used_in_backend = false;
  bool local_pose_success = false;
  double local_pose_rmse = std::numeric_limits<double>::quiet_NaN();
  Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_cam0_world_inferred = Eigen::Isometry3d::Identity();
  double frame_pose_rot_deg = std::numeric_limits<double>::quiet_NaN();
  double frame_pose_trans_m = std::numeric_limits<double>::quiet_NaN();
};

struct FrameMetrics {
  std::string split;
  int pair_index = -1;
  bool selected_pair_board_used_in_backend = false;
  int num_visible_boards = 0;
  int num_bicam_boards = 0;
  int num_total_corners = 0;
  int num_highpolar_50plus = 0;
  int num_highpolar_70plus = 0;
  int num_polar_50_70 = 0;
  double balance_entropy = 0.0;
  double pose_rot_median = std::numeric_limits<double>::quiet_NaN();
  double pose_trans_median = std::numeric_limits<double>::quiet_NaN();
  double pose_rot_max = std::numeric_limits<double>::quiet_NaN();
  double pose_trans_max = std::numeric_limits<double>::quiet_NaN();
  double layout_rot_median = std::numeric_limits<double>::quiet_NaN();
  double layout_trans_median = std::numeric_limits<double>::quiet_NaN();
  double stereo_rot_median = std::numeric_limits<double>::quiet_NaN();
  double stereo_trans_median = std::numeric_limits<double>::quiet_NaN();
  double score = 0.0;
  bool rescue_candidate = false;
};

struct StereoMetric {
  std::string split;
  int pair_index = -1;
  int board_id = -1;
  bool selected_pair_board_used_in_backend = false;
  int num_cam0_corners = 0;
  int num_cam1_corners = 0;
  double rot_deg = std::numeric_limits<double>::quiet_NaN();
  double trans_m = std::numeric_limits<double>::quiet_NaN();
  bool is_highpolar_board = false;
};

struct LayoutMetric {
  std::string split;
  int pair_index = -1;
  int camera_index = -1;
  int board_a = -1;
  int board_b = -1;
  bool selected_pair_board_a_used_in_backend = false;
  bool selected_pair_board_b_used_in_backend = false;
  double rot_deg = std::numeric_limits<double>::quiet_NaN();
  double trans_m = std::numeric_limits<double>::quiet_NaN();
};

IntermediateCameraConfig MakeCameraConfig(
    const StereoCameraFixedCalibration& calibration) {
  IntermediateCameraConfig config;
  config.camera_yaml = calibration.source_yaml_path;
  config.camera_model = calibration.camera_model;
  config.distortion_model = calibration.distortion_model;
  config.intrinsics = calibration.intrinsics;
  config.distortion_coeffs = calibration.distortion_coeffs;
  config.resolution = calibration.resolution;
  return config;
}

std::vector<cv::Point3f> BuildObjectPoints(
    const std::vector<Eigen::Vector3d>& points) {
  std::vector<cv::Point3f> object_points;
  object_points.reserve(points.size());
  for (const Eigen::Vector3d& point : points) {
    object_points.push_back(
        cv::Point3f(static_cast<float>(point.x()),
                    static_cast<float>(point.y()),
                    static_cast<float>(point.z())));
  }
  return object_points;
}

Eigen::Isometry3d MakePose(const cv::Mat& rvec, const cv::Mat& tvec) {
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      R(row, col) = rotation.at<double>(row, col);
    }
    t[row] = tvec.at<double>(row, 0);
  }
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = R;
  transform.translation() = t;
  return transform;
}

double RotationAngleDeg(const Eigen::Matrix3d& R) {
  const double trace = std::max(-1.0, std::min(3.0, R.trace()));
  const double cosine = std::max(-1.0, std::min(1.0, 0.5 * (trace - 1.0)));
  return std::acos(cosine) * kRadToDeg;
}

double RotationDistanceDeg(const Eigen::Isometry3d& lhs,
                           const Eigen::Isometry3d& rhs) {
  return RotationAngleDeg(lhs.linear() * rhs.linear().transpose());
}

double Median(std::vector<double> values) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](double v) { return !std::isfinite(v); }),
               values.end());
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  if (n % 2 == 1) {
    return values[n / 2];
  }
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double MaxFinite(const std::vector<double>& values) {
  double result = std::numeric_limits<double>::quiet_NaN();
  for (double v : values) {
    if (!std::isfinite(v)) {
      continue;
    }
    if (!std::isfinite(result) || v > result) {
      result = v;
    }
  }
  return result;
}

std::string ValueOrNan(double value) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream oss;
  oss << std::setprecision(12) << value;
  return oss.str();
}

double PolarAngleDeg(const DoubleSphereCameraModel& camera,
                     const Eigen::Vector2d& pixel) {
  Eigen::Vector3d ray = Eigen::Vector3d::Zero();
  if (!camera.keypointToEuclidean(pixel, &ray)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double norm = ray.norm();
  if (!(norm > 0.0) || !std::isfinite(norm)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double cos_theta = std::max(-1.0, std::min(1.0, ray.z() / norm));
  return std::acos(cos_theta) * kRadToDeg;
}

std::string PolarBucket(double polar_deg) {
  if (!std::isfinite(polar_deg)) {
    return "invalid";
  }
  if (polar_deg >= 70.0) {
    return "polar_70_plus";
  }
  if (polar_deg >= 50.0) {
    return "polar_50_70";
  }
  if (polar_deg >= 30.0) {
    return "polar_30_50";
  }
  return "polar_0_30";
}

bool PairBoardSelectedInBackend(
    const StereoExtrinsicCalibrationResult& result,
    int pair_index,
    int board_id) {
  return result.pair_selection_summary.selected_pair_board_keys.count(
             std::make_pair(pair_index, board_id)) > 0;
}

std::vector<std::string> SplitsForObservation(
    const StereoExtrinsicCalibrationResult& result,
    const StereoObservation& observation) {
  const StereoMeasurementDataset& dataset =
      result.problem_input.measurement_dataset;
  bool is_training = false;
  if (std::find(dataset.training_pair_indices.begin(),
                dataset.training_pair_indices.end(),
                observation.pair_index) != dataset.training_pair_indices.end()) {
    is_training = true;
  }
  std::vector<std::string> splits;
  if (is_training) {
    splits.push_back("training_all_valid");
    if (PairBoardSelectedInBackend(result, observation.pair_index,
                                   observation.board_id)) {
      splits.push_back("training_selected_backend");
    }
  } else {
    splits.push_back("holdout");
  }
  return splits;
}

bool EstimateLocalPose(const LocalGroup& group,
                       const DoubleSphereCameraModel& camera,
                       Eigen::Isometry3d* T_camera_board,
                       double* rmse) {
  if (T_camera_board == nullptr || rmse == nullptr) {
    throw std::runtime_error("EstimateLocalPose requires outputs.");
  }
  std::vector<Eigen::Vector3d> target_points;
  std::vector<cv::Point2f> image_points;
  for (const StereoObservation* observation : group.observations) {
    if (observation == nullptr ||
        observation->point_type != JointPointType::Outer) {
      continue;
    }
    target_points.push_back(observation->target_point_board);
    image_points.push_back(
        cv::Point2f(static_cast<float>(observation->observed_image_xy.x()),
                    static_cast<float>(observation->observed_image_xy.y())));
  }
  if (target_points.size() < 4) {
    return false;
  }
  cv::Mat rvec;
  cv::Mat tvec;
  if (!camera.estimateTransformation(BuildObjectPoints(target_points),
                                     image_points, &rvec, &tvec)) {
    return false;
  }
  const Eigen::Isometry3d pose = MakePose(rvec, tvec);
  double squared_error_sum = 0.0;
  int count = 0;
  for (std::size_t i = 0; i < target_points.size(); ++i) {
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    if (!camera.vsEuclideanToKeypoint(pose * target_points[i], &predicted)) {
      continue;
    }
    const Eigen::Vector2d observed(image_points[i].x, image_points[i].y);
    squared_error_sum += (predicted - observed).squaredNorm();
    ++count;
  }
  if (count <= 0) {
    return false;
  }
  *T_camera_board = pose;
  *rmse = std::sqrt(squared_error_sum / static_cast<double>(count));
  return true;
}

Eigen::Isometry3d MatrixToIsometry(const Eigen::Matrix4d& matrix) {
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.matrix() = matrix;
  return result;
}

Eigen::Matrix4d IsometryToMatrix(const Eigen::Isometry3d& transform) {
  return transform.matrix();
}

void WriteSummary(const std::string& path,
                  const MultiBoardCoObservationOptions& options,
                  const MultiBoardCoObservationSummary& summary,
                  const std::map<std::string, int>& frame_count_by_split,
                  const std::map<std::string, int>& rescue_count_by_split) {
  std::ofstream output(path.c_str());
  output << std::setprecision(12);
  output << "success: " << (summary.success ? 1 : 0) << "\n";
  output << "failure_reason: " << summary.failure_reason << "\n";
  output << "transform_convention: T_A_B maps points from frame B to frame A\n";
  output << "enabled: " << (options.enabled ? 1 : 0) << "\n";
  output << "min_corners_per_group: " << options.min_corners_per_group << "\n";
  output << "high_polar_threshold_deg: "
         << options.high_polar_threshold_deg << "\n";
  output << "very_high_polar_threshold_deg: "
         << options.very_high_polar_threshold_deg << "\n";
  output << "enable_rescue_suggestions: "
         << (options.enable_rescue_suggestions ? 1 : 0) << "\n";
  output << "total_frames_processed: "
         << summary.total_frames_processed << "\n";
  for (const auto& entry : frame_count_by_split) {
    output << "frames_processed_" << entry.first << ": "
           << entry.second << "\n";
  }
  output << "frames_with_at_least_two_boards: "
         << summary.frames_with_at_least_two_boards << "\n";
  output << "frames_with_bicam_board_observations: "
         << summary.frames_with_bicam_board_observations << "\n";
  output << "high_polar_rescue_candidate_count: "
         << summary.high_polar_rescue_candidate_count << "\n";
  for (const auto& entry : rescue_count_by_split) {
    output << "high_polar_rescue_candidate_count_" << entry.first << ": "
           << entry.second << "\n";
  }
  output << "median_C_pose_rot_deg: "
         << ValueOrNan(summary.median_pose_rotation_deg) << "\n";
  output << "median_C_pose_trans_m: "
         << ValueOrNan(summary.median_pose_translation_m) << "\n";
  output << "median_C_layout_rot_deg: "
         << ValueOrNan(summary.median_layout_rotation_deg) << "\n";
  output << "median_C_layout_trans_m: "
         << ValueOrNan(summary.median_layout_translation_m) << "\n";
  output << "median_C_stereo_rot_deg: "
         << ValueOrNan(summary.median_stereo_rotation_deg) << "\n";
  output << "median_C_stereo_trans_m: "
         << ValueOrNan(summary.median_stereo_translation_m) << "\n";
  for (const std::string& warning : summary.warnings) {
    output << "warning: " << warning << "\n";
  }
}

}  // namespace

MultiBoardCoObservationConsistency::MultiBoardCoObservationConsistency(
    MultiBoardCoObservationOptions options)
    : options_(std::move(options)) {}

MultiBoardCoObservationSummary MultiBoardCoObservationConsistency::Evaluate(
    const StereoExtrinsicCalibrationResult& result) const {
  MultiBoardCoObservationSummary summary;
  if (!options_.enabled) {
    summary.success = true;
    return summary;
  }
  if (options_.output_dir.empty()) {
    summary.failure_reason = "empty_output_dir";
    return summary;
  }

  fs::create_directories(options_.output_dir);

  if (result.optimized_scene.cam0.camera_model_family != "ds-none" ||
      result.optimized_scene.cam1.camera_model_family != "ds-none") {
    summary.failure_reason = "non_ds_none_camera_model";
    WriteSummary((fs::path(options_.output_dir) / "coobs_summary.txt").string(),
                 options_, summary, std::map<std::string, int>(),
                 std::map<std::string, int>());
    return summary;
  }

  const StereoMeasurementDataset& dataset =
      result.problem_input.measurement_dataset;
  const StereoSceneState& scene = result.optimized_scene;
  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene.cam1));
  const Eigen::Isometry3d T_cam1_cam0 = MatrixToIsometry(scene.T_cam1_cam0);

  std::map<GroupKey, LocalGroup> groups;
  std::map<std::pair<std::string, int>, std::map<int, std::set<int>>>
      frame_camera_boards;
  std::map<std::pair<std::string, int>,
           std::map<std::tuple<int, int, std::string>, int>>
      frame_balance_buckets;

  for (const StereoObservation& observation : dataset.observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    const DoubleSphereCameraModel& camera =
        observation.camera_index == 0 ? cam0 : cam1;
    const double polar_deg =
        PolarAngleDeg(camera, observation.observed_image_xy);
    for (const std::string& split : SplitsForObservation(result, observation)) {
      GroupKey key{split, observation.pair_index, observation.camera_index,
                   observation.board_id};
      LocalGroup& group = groups[key];
      group.key = key;
      group.observations.push_back(&observation);
      ++group.num_corners;
      if (observation.point_type == JointPointType::Outer) {
        ++group.outer_corner_count;
      } else {
        ++group.internal_corner_count;
      }
      group.selected_pair_board_used_in_backend =
          PairBoardSelectedInBackend(result, observation.pair_index,
                                     observation.board_id);
      if (polar_deg >= options_.high_polar_threshold_deg) {
        ++group.num_highpolar_50plus;
      }
      if (polar_deg >= options_.very_high_polar_threshold_deg) {
        ++group.num_highpolar_70plus;
      }
      const std::pair<std::string, int> frame_key(split,
                                                  observation.pair_index);
      frame_camera_boards[frame_key][observation.camera_index].insert(
          observation.board_id);
      frame_balance_buckets[frame_key]
                           [std::make_tuple(observation.camera_index,
                                            observation.board_id,
                                            PolarBucket(polar_deg))]++;
    }
  }

  std::map<std::pair<std::string, int>, std::vector<GroupKey>>
      valid_groups_by_frame;
  std::map<GroupKey, LocalGroup> valid_groups;
  for (auto& entry : groups) {
    LocalGroup group = entry.second;
    if (group.num_corners < options_.min_corners_per_group) {
      continue;
    }
    const DoubleSphereCameraModel& camera =
        group.key.camera_index == 0 ? cam0 : cam1;
    group.local_pose_success =
        EstimateLocalPose(group, camera, &group.T_camera_board,
                          &group.local_pose_rmse);
    if (!group.local_pose_success) {
      valid_groups[entry.first] = group;
      continue;
    }
    const auto board_it =
        scene.T_world_board_by_id.find(group.key.board_id);
    if (board_it == scene.T_world_board_by_id.end()) {
      group.local_pose_success = false;
      valid_groups[entry.first] = group;
      continue;
    }
    const Eigen::Isometry3d T_world_board = MatrixToIsometry(board_it->second);
    if (group.key.camera_index == 0) {
      // T_cam0_world = T_cam0_board * inverse(T_world_board).
      group.T_cam0_world_inferred =
          group.T_camera_board * T_world_board.inverse();
    } else {
      // T_cam0_world = inverse(T_cam1_cam0) * T_cam1_board *
      //                inverse(T_world_board).
      group.T_cam0_world_inferred =
          T_cam1_cam0.inverse() * group.T_camera_board *
          T_world_board.inverse();
    }
    valid_groups[entry.first] = group;
    valid_groups_by_frame[std::make_pair(group.key.split,
                                         group.key.pair_index)]
        .push_back(entry.first);
  }

  std::map<std::pair<std::string, int>, FrameMetrics> frame_metrics;
  std::vector<StereoMetric> stereo_metrics;
  std::vector<LayoutMetric> layout_metrics;
  std::vector<double> all_pose_rot;
  std::vector<double> all_pose_trans;
  std::vector<double> all_layout_rot;
  std::vector<double> all_layout_trans;
  std::vector<double> all_stereo_rot;
  std::vector<double> all_stereo_trans;

  for (const auto& frame_entry : valid_groups_by_frame) {
    const std::string split = frame_entry.first.first;
    const int pair_index = frame_entry.first.second;
    const std::vector<GroupKey>& frame_group_keys = frame_entry.second;
    FrameMetrics metrics;
    metrics.split = split;
    metrics.pair_index = pair_index;

    std::set<int> visible_boards;
    std::map<int, std::set<int>> board_cameras;
    for (const GroupKey& key : frame_group_keys) {
      const LocalGroup& group = valid_groups[key];
      if (!group.local_pose_success) {
        continue;
      }
      visible_boards.insert(key.board_id);
      board_cameras[key.board_id].insert(key.camera_index);
      metrics.selected_pair_board_used_in_backend =
          metrics.selected_pair_board_used_in_backend ||
          group.selected_pair_board_used_in_backend;
      metrics.num_total_corners += group.num_corners;
      metrics.num_highpolar_50plus += group.num_highpolar_50plus;
      metrics.num_highpolar_70plus += group.num_highpolar_70plus;
      metrics.num_polar_50_70 +=
          std::max(0, group.num_highpolar_50plus -
                          group.num_highpolar_70plus);
    }
    metrics.num_visible_boards = static_cast<int>(visible_boards.size());
    for (const auto& entry : board_cameras) {
      if (entry.second.count(0) > 0 && entry.second.count(1) > 0) {
        ++metrics.num_bicam_boards;
      }
    }

    Eigen::Isometry3d reference_pose = Eigen::Isometry3d::Identity();
    bool have_reference_pose = false;
    double best_reference_score = std::numeric_limits<double>::infinity();
    for (const GroupKey& candidate_key : frame_group_keys) {
      const LocalGroup& candidate = valid_groups[candidate_key];
      if (!candidate.local_pose_success) {
        continue;
      }
      double score = 0.0;
      int comparisons = 0;
      for (const GroupKey& other_key : frame_group_keys) {
        const LocalGroup& other = valid_groups[other_key];
        if (!other.local_pose_success) {
          continue;
        }
        score += RotationDistanceDeg(candidate.T_cam0_world_inferred,
                                     other.T_cam0_world_inferred) +
                 (candidate.T_cam0_world_inferred.translation() -
                  other.T_cam0_world_inferred.translation())
                     .norm();
        ++comparisons;
      }
      if (comparisons > 0) {
        score /= static_cast<double>(comparisons);
      }
      if (score < best_reference_score) {
        best_reference_score = score;
        reference_pose = candidate.T_cam0_world_inferred;
        have_reference_pose = true;
      }
    }

    std::vector<double> pose_rot;
    std::vector<double> pose_trans;
    if (have_reference_pose) {
      for (const GroupKey& key : frame_group_keys) {
        LocalGroup& group = valid_groups[key];
        if (!group.local_pose_success) {
          continue;
        }
        group.frame_pose_rot_deg =
            RotationDistanceDeg(reference_pose, group.T_cam0_world_inferred);
        group.frame_pose_trans_m =
            (reference_pose.translation() -
             group.T_cam0_world_inferred.translation())
                .norm();
        pose_rot.push_back(group.frame_pose_rot_deg);
        pose_trans.push_back(group.frame_pose_trans_m);
        all_pose_rot.push_back(group.frame_pose_rot_deg);
        all_pose_trans.push_back(group.frame_pose_trans_m);
      }
    }
    metrics.pose_rot_median = Median(pose_rot);
    metrics.pose_trans_median = Median(pose_trans);
    metrics.pose_rot_max = MaxFinite(pose_rot);
    metrics.pose_trans_max = MaxFinite(pose_trans);

    const std::pair<std::string, int> frame_key(split, pair_index);
    for (const auto& camera_entry : frame_camera_boards[frame_key]) {
      const int camera_index = camera_entry.first;
      std::vector<int> boards(camera_entry.second.begin(),
                              camera_entry.second.end());
      std::sort(boards.begin(), boards.end());
      for (std::size_t i = 0; i < boards.size(); ++i) {
        for (std::size_t j = i + 1; j < boards.size(); ++j) {
          const int board_a = boards[i];
          const int board_b = boards[j];
          GroupKey key_a{split, pair_index, camera_index, board_a};
          GroupKey key_b{split, pair_index, camera_index, board_b};
          const auto group_a_it = valid_groups.find(key_a);
          const auto group_b_it = valid_groups.find(key_b);
          if (group_a_it == valid_groups.end() ||
              group_b_it == valid_groups.end() ||
              !group_a_it->second.local_pose_success ||
              !group_b_it->second.local_pose_success) {
            continue;
          }
          const auto board_a_it = scene.T_world_board_by_id.find(board_a);
          const auto board_b_it = scene.T_world_board_by_id.find(board_b);
          if (board_a_it == scene.T_world_board_by_id.end() ||
              board_b_it == scene.T_world_board_by_id.end()) {
            continue;
          }
          const Eigen::Isometry3d T_cam_board_a =
              group_a_it->second.T_camera_board;
          const Eigen::Isometry3d T_cam_board_b =
              group_b_it->second.T_camera_board;
          const Eigen::Isometry3d T_board_a_board_b_observed =
              T_cam_board_a.inverse() * T_cam_board_b;
          const Eigen::Isometry3d T_world_board_a =
              MatrixToIsometry(board_a_it->second);
          const Eigen::Isometry3d T_world_board_b =
              MatrixToIsometry(board_b_it->second);
          const Eigen::Isometry3d T_board_a_board_b_layout =
              T_world_board_a.inverse() * T_world_board_b;
          const Eigen::Isometry3d delta =
              T_board_a_board_b_layout.inverse() *
              T_board_a_board_b_observed;
          LayoutMetric layout;
          layout.split = split;
          layout.pair_index = pair_index;
          layout.camera_index = camera_index;
          layout.board_a = board_a;
          layout.board_b = board_b;
          layout.selected_pair_board_a_used_in_backend =
              group_a_it->second.selected_pair_board_used_in_backend;
          layout.selected_pair_board_b_used_in_backend =
              group_b_it->second.selected_pair_board_used_in_backend;
          layout.rot_deg = RotationAngleDeg(delta.linear());
          layout.trans_m = delta.translation().norm();
          layout_metrics.push_back(layout);
          all_layout_rot.push_back(layout.rot_deg);
          all_layout_trans.push_back(layout.trans_m);
        }
      }
    }

    for (int board_id : visible_boards) {
      GroupKey key0{split, pair_index, 0, board_id};
      GroupKey key1{split, pair_index, 1, board_id};
      const auto group0_it = valid_groups.find(key0);
      const auto group1_it = valid_groups.find(key1);
      if (group0_it == valid_groups.end() ||
          group1_it == valid_groups.end() ||
          !group0_it->second.local_pose_success ||
          !group1_it->second.local_pose_success) {
        continue;
      }
      const Eigen::Isometry3d T_cam1_cam0_observed =
          group1_it->second.T_camera_board *
          group0_it->second.T_camera_board.inverse();
      const Eigen::Isometry3d delta =
          T_cam1_cam0.inverse() * T_cam1_cam0_observed;
      StereoMetric stereo;
      stereo.split = split;
      stereo.pair_index = pair_index;
      stereo.board_id = board_id;
      stereo.selected_pair_board_used_in_backend =
          group0_it->second.selected_pair_board_used_in_backend ||
          group1_it->second.selected_pair_board_used_in_backend;
      stereo.num_cam0_corners = group0_it->second.num_corners;
      stereo.num_cam1_corners = group1_it->second.num_corners;
      stereo.rot_deg = RotationAngleDeg(delta.linear());
      stereo.trans_m = delta.translation().norm();
      stereo.is_highpolar_board =
          group0_it->second.num_highpolar_50plus > 0 ||
          group1_it->second.num_highpolar_50plus > 0;
      stereo_metrics.push_back(stereo);
      all_stereo_rot.push_back(stereo.rot_deg);
      all_stereo_trans.push_back(stereo.trans_m);
    }

    std::vector<double> frame_layout_rot;
    std::vector<double> frame_layout_trans;
    for (const LayoutMetric& layout : layout_metrics) {
      if (layout.split == split && layout.pair_index == pair_index) {
        frame_layout_rot.push_back(layout.rot_deg);
        frame_layout_trans.push_back(layout.trans_m);
      }
    }
    std::vector<double> frame_stereo_rot;
    std::vector<double> frame_stereo_trans;
    for (const StereoMetric& stereo : stereo_metrics) {
      if (stereo.split == split && stereo.pair_index == pair_index) {
        frame_stereo_rot.push_back(stereo.rot_deg);
        frame_stereo_trans.push_back(stereo.trans_m);
      }
    }
    metrics.layout_rot_median = Median(frame_layout_rot);
    metrics.layout_trans_median = Median(frame_layout_trans);
    metrics.stereo_rot_median = Median(frame_stereo_rot);
    metrics.stereo_trans_median = Median(frame_stereo_trans);

    const auto balance_it = frame_balance_buckets.find(frame_key);
    if (balance_it != frame_balance_buckets.end()) {
      int total = 0;
      for (const auto& entry : balance_it->second) {
        total += entry.second;
      }
      if (total > 0) {
        for (const auto& entry : balance_it->second) {
          const double p = static_cast<double>(entry.second) /
                           static_cast<double>(total);
          if (p > 0.0) {
            metrics.balance_entropy -= p * std::log(p);
          }
        }
      }
    }

    const double high_polar_coverage =
        0.5 * static_cast<double>(metrics.num_polar_50_70) +
        static_cast<double>(metrics.num_highpolar_70plus);
    const double multiboard =
        std::log(1.0 + static_cast<double>(metrics.num_visible_boards)) +
        std::log(1.0 + static_cast<double>(metrics.num_bicam_boards));
    const double conflict =
        (std::isfinite(metrics.pose_rot_median) ? metrics.pose_rot_median : 0.0) +
        10.0 * (std::isfinite(metrics.pose_trans_median)
                    ? metrics.pose_trans_median
                    : 0.0) +
        (std::isfinite(metrics.layout_rot_median)
             ? metrics.layout_rot_median
             : 0.0) +
        10.0 * (std::isfinite(metrics.layout_trans_median)
                    ? metrics.layout_trans_median
                    : 0.0) +
        (std::isfinite(metrics.stereo_rot_median)
             ? metrics.stereo_rot_median
             : 0.0) +
        10.0 * (std::isfinite(metrics.stereo_trans_median)
                    ? metrics.stereo_trans_median
                    : 0.0);
    metrics.score =
        options_.score_alpha_high_polar * high_polar_coverage +
        options_.score_beta_multiboard * multiboard +
        options_.score_gamma_balance * metrics.balance_entropy -
        options_.score_eta_conflict * conflict;
    metrics.rescue_candidate =
        options_.enable_rescue_suggestions &&
        high_polar_coverage >= options_.rescue_min_high_polar_score &&
        metrics.num_visible_boards >= 2 &&
        metrics.num_bicam_boards >= 1 &&
        conflict < options_.rescue_bad_conflict_threshold;

    frame_metrics[frame_key] = metrics;
  }

  std::map<std::string, int> frame_count_by_split;
  std::map<std::string, int> rescue_count_by_split;
  summary.total_frames_processed = static_cast<int>(frame_metrics.size());
  for (const auto& entry : frame_metrics) {
    ++frame_count_by_split[entry.second.split];
    if (entry.second.num_visible_boards >= 2) {
      ++summary.frames_with_at_least_two_boards;
    }
    if (entry.second.num_bicam_boards >= 1) {
      ++summary.frames_with_bicam_board_observations;
    }
    if (entry.second.rescue_candidate) {
      ++summary.high_polar_rescue_candidate_count;
      ++rescue_count_by_split[entry.second.split];
    }
  }
  summary.median_pose_rotation_deg = Median(all_pose_rot);
  summary.median_pose_translation_m = Median(all_pose_trans);
  summary.median_layout_rotation_deg = Median(all_layout_rot);
  summary.median_layout_translation_m = Median(all_layout_trans);
  summary.median_stereo_rotation_deg = Median(all_stereo_rot);
  summary.median_stereo_translation_m = Median(all_stereo_trans);
  summary.success = true;

  const fs::path output_dir(options_.output_dir);
  {
    std::ofstream output((output_dir / "coobs_group_metrics.csv").string().c_str());
    output << std::setprecision(12);
    output << "split,frame_id,camera_id,board_id,"
           << "selected_pair_board_used_in_backend,"
           << "num_corners,outer_corner_count,internal_corner_count,"
           << "num_highpolar_50plus,num_highpolar_70plus,"
           << "local_pose_success,local_pose_rmse,"
           << "frame_pose_consistency_rot_deg,"
           << "frame_pose_consistency_trans\n";
    for (const auto& entry : valid_groups) {
      const LocalGroup& group = entry.second;
      output << group.key.split << "," << group.key.pair_index << ","
             << group.key.camera_index << "," << group.key.board_id << ","
             << (group.selected_pair_board_used_in_backend ? 1 : 0) << ","
             << group.num_corners << "," << group.outer_corner_count << ","
             << group.internal_corner_count << ","
             << group.num_highpolar_50plus << ","
             << group.num_highpolar_70plus << ","
             << (group.local_pose_success ? 1 : 0) << ","
             << ValueOrNan(group.local_pose_rmse) << ","
             << ValueOrNan(group.frame_pose_rot_deg) << ","
             << ValueOrNan(group.frame_pose_trans_m) << "\n";
    }
  }
  {
    std::ofstream output((output_dir / "coobs_frame_metrics.csv").string().c_str());
    output << std::setprecision(12);
    output << "split,frame_id,selected_pair_board_used_in_backend,"
           << "num_visible_boards,num_bicam_boards,"
           << "num_total_corners,num_highpolar_50plus,num_highpolar_70plus,"
           << "balance_entropy,C_pose_rot_deg_median,C_pose_trans_median,"
           << "C_pose_rot_deg_max,C_pose_trans_max,"
           << "C_layout_rot_deg_median,C_layout_trans_median,"
           << "C_stereo_rot_deg_median,C_stereo_trans_median,"
           << "S_coobs,rescue_candidate\n";
    for (const auto& entry : frame_metrics) {
      const FrameMetrics& metrics = entry.second;
      output << metrics.split << "," << metrics.pair_index << ","
             << (metrics.selected_pair_board_used_in_backend ? 1 : 0) << ","
             << metrics.num_visible_boards << ","
             << metrics.num_bicam_boards << "," << metrics.num_total_corners
             << "," << metrics.num_highpolar_50plus << ","
             << metrics.num_highpolar_70plus << ","
             << metrics.balance_entropy << ","
             << ValueOrNan(metrics.pose_rot_median) << ","
             << ValueOrNan(metrics.pose_trans_median) << ","
             << ValueOrNan(metrics.pose_rot_max) << ","
             << ValueOrNan(metrics.pose_trans_max) << ","
             << ValueOrNan(metrics.layout_rot_median) << ","
             << ValueOrNan(metrics.layout_trans_median) << ","
             << ValueOrNan(metrics.stereo_rot_median) << ","
             << ValueOrNan(metrics.stereo_trans_median) << ","
             << metrics.score << ","
             << (metrics.rescue_candidate ? 1 : 0) << "\n";
    }
  }
  {
    std::ofstream output((output_dir / "coobs_stereo_metrics.csv").string().c_str());
    output << std::setprecision(12);
    output << "split,frame_id,board_id,selected_pair_board_used_in_backend,"
           << "num_cam0_corners,num_cam1_corners,"
           << "stereo_consistency_rot_deg,stereo_consistency_trans,"
           << "is_highpolar_board\n";
    for (const StereoMetric& stereo : stereo_metrics) {
      output << stereo.split << "," << stereo.pair_index << ","
             << stereo.board_id << ","
             << (stereo.selected_pair_board_used_in_backend ? 1 : 0) << ","
             << stereo.num_cam0_corners << "," << stereo.num_cam1_corners
             << "," << ValueOrNan(stereo.rot_deg) << ","
             << ValueOrNan(stereo.trans_m) << ","
             << (stereo.is_highpolar_board ? 1 : 0) << "\n";
    }
  }
  {
    std::ofstream output((output_dir / "coobs_layout_metrics.csv").string().c_str());
    output << std::setprecision(12);
    output << "split,frame_id,camera_id,board_a,board_b,"
           << "selected_pair_board_a_used_in_backend,"
           << "selected_pair_board_b_used_in_backend,"
           << "layout_consistency_rot_deg,layout_consistency_trans\n";
    for (const LayoutMetric& layout : layout_metrics) {
      output << layout.split << "," << layout.pair_index << ","
             << layout.camera_index << ","
             << layout.board_a << "," << layout.board_b << ","
             << (layout.selected_pair_board_a_used_in_backend ? 1 : 0) << ","
             << (layout.selected_pair_board_b_used_in_backend ? 1 : 0) << ","
             << ValueOrNan(layout.rot_deg) << ","
             << ValueOrNan(layout.trans_m) << "\n";
    }
  }
  {
    std::ofstream output((output_dir / "coobs_rescue_candidates.csv").string().c_str());
    output << std::setprecision(12);
    output << "split,frame_id,selected_pair_board_used_in_backend,"
           << "num_visible_boards,num_bicam_boards,"
           << "num_highpolar_50plus,num_highpolar_70plus,"
           << "balance_entropy,S_coobs,C_pose_rot_deg_median,"
           << "C_layout_rot_deg_median,C_stereo_rot_deg_median\n";
    for (const auto& entry : frame_metrics) {
      const FrameMetrics& metrics = entry.second;
      if (!metrics.rescue_candidate) {
        continue;
      }
      output << metrics.split << "," << metrics.pair_index << ","
             << (metrics.selected_pair_board_used_in_backend ? 1 : 0) << ","
             << metrics.num_visible_boards
             << "," << metrics.num_bicam_boards << ","
             << metrics.num_highpolar_50plus << ","
             << metrics.num_highpolar_70plus << ","
             << metrics.balance_entropy << "," << metrics.score << ","
             << ValueOrNan(metrics.pose_rot_median) << ","
             << ValueOrNan(metrics.layout_rot_median) << ","
             << ValueOrNan(metrics.stereo_rot_median) << "\n";
    }
  }
  WriteSummary((output_dir / "coobs_summary.txt").string(), options_, summary,
               frame_count_by_split, rescue_count_by_split);
  std::cout << "[CoObs] total frames processed: "
            << summary.total_frames_processed << std::endl;
  std::cout << "[CoObs] frames with >=2 boards: "
            << summary.frames_with_at_least_two_boards << std::endl;
  std::cout << "[CoObs] frames with bicam board observations: "
            << summary.frames_with_bicam_board_observations << std::endl;
  std::cout << "[CoObs] high-polar rescue candidates: "
            << summary.high_polar_rescue_candidate_count << std::endl;
  std::cout << "[CoObs] median C_pose rot/trans: "
            << ValueOrNan(summary.median_pose_rotation_deg) << " deg / "
            << ValueOrNan(summary.median_pose_translation_m) << " m"
            << std::endl;
  std::cout << "[CoObs] median C_layout rot/trans: "
            << ValueOrNan(summary.median_layout_rotation_deg) << " deg / "
            << ValueOrNan(summary.median_layout_translation_m) << " m"
            << std::endl;
  std::cout << "[CoObs] median C_stereo rot/trans: "
            << ValueOrNan(summary.median_stereo_rotation_deg) << " deg / "
            << ValueOrNan(summary.median_stereo_translation_m) << " m"
            << std::endl;
  return summary;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
