#include <aslam/cameras/apriltag_internal/StereoResidualEvaluator.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <opencv2/calib3d.hpp>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

struct StereoOuterPoseObservation {
  int camera_index = -1;
  int board_id = -1;
  Eigen::Vector3d object_point_world = Eigen::Vector3d::Zero();
  Eigen::Vector2d observed_image_xy = Eigen::Vector2d::Zero();
};

struct StereoBoardPoseObservation {
  int camera_index = -1;
  Eigen::Vector3d object_point_board = Eigen::Vector3d::Zero();
  Eigen::Vector2d observed_image_xy = Eigen::Vector2d::Zero();
};

struct ResidualAccumulator {
  int count = 0;
  double squared_sum = 0.0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_sq_x = 0.0;
  double sum_sq_y = 0.0;
};

double ComputeRmse(double squared_sum, int count) {
  if (count <= 0) {
    return 0.0;
  }
  return std::sqrt(squared_sum / static_cast<double>(count));
}

double ComputeStd(double sum, double sum_sq, int count) {
  if (count <= 0) {
    return 0.0;
  }
  const double mean = sum / static_cast<double>(count);
  const double variance =
      std::max(0.0, sum_sq / static_cast<double>(count) - mean * mean);
  return std::sqrt(variance);
}

IntermediateCameraConfig MakeCameraConfig(
    const StereoCameraFixedCalibration& calibration) {
  IntermediateCameraConfig config;
  config.camera_model = calibration.camera_model;
  config.distortion_model = calibration.distortion_model;
  config.intrinsics = calibration.intrinsics;
  config.distortion_coeffs = calibration.distortion_coeffs;
  config.resolution = calibration.resolution;
  config.camera_yaml.clear();
  return config;
}

std::vector<cv::Point3f> BuildObjectPoints(
    const std::vector<Eigen::Vector3d>& points) {
  std::vector<cv::Point3f> object_points;
  object_points.reserve(points.size());
  for (const Eigen::Vector3d& point : points) {
    object_points.push_back(cv::Point3f(static_cast<float>(point.x()),
                                        static_cast<float>(point.y()),
                                        static_cast<float>(point.z())));
  }
  return object_points;
}

Eigen::Isometry3d MakePose(const cv::Mat& rvec, const cv::Mat& tvec) {
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);
  Eigen::Matrix3d R;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      R(row, col) = rotation.at<double>(row, col);
    }
  }
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = R;
  pose.translation() = Eigen::Vector3d(tvec.at<double>(0),
                                      tvec.at<double>(1),
                                      tvec.at<double>(2));
  return pose;
}

const StereoFramePair* FindPair(const StereoMeasurementDataset& dataset,
                                int pair_index) {
  for (const StereoFramePair& pair : dataset.frame_pairs) {
    if (pair.pair_index == pair_index) {
      return &pair;
    }
  }
  return nullptr;
}

bool PairUsesKnownBoards(const StereoMeasurementDataset& dataset,
                         const StereoSceneState& scene_state,
                         int pair_index) {
  const auto training_it = dataset.training_pair_board_ids.find(pair_index);
  const auto holdout_it = dataset.holdout_pair_board_ids.find(pair_index);
  const std::set<int>* board_ids = nullptr;
  if (training_it != dataset.training_pair_board_ids.end()) {
    board_ids = &training_it->second;
  } else if (holdout_it != dataset.holdout_pair_board_ids.end()) {
    board_ids = &holdout_it->second;
  }
  if (board_ids == nullptr || board_ids->empty()) {
    return false;
  }
  for (int board_id : *board_ids) {
    if (scene_state.T_world_board_by_id.count(board_id) > 0) {
      return true;
    }
  }
  return false;
}

bool ContainsBoard(const std::map<int, std::set<int> >& boards_by_pair,
                   int pair_index,
                   int board_id) {
  const auto it = boards_by_pair.find(pair_index);
  return it != boards_by_pair.end() && it->second.count(board_id) > 0;
}

bool RefitPairPoseFromOuterObservations(const StereoMeasurementDataset& dataset,
                                        const StereoSceneState& scene_state,
                                        int pair_index,
                                        Eigen::Matrix4d* T_cam0_world) {
  if (T_cam0_world == nullptr) {
    throw std::runtime_error("RefitPairPoseFromOuterObservations requires output pose.");
  }
  std::vector<Eigen::Vector3d> object_points;
  std::vector<cv::Point2f> image_points;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.camera_index != 0 ||
        observation.point_type != JointPointType::Outer ||
        !observation.used_in_solver) {
      continue;
    }
    const auto board_it = scene_state.T_world_board_by_id.find(observation.board_id);
    if (board_it == scene_state.T_world_board_by_id.end()) {
      continue;
    }
    const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                      observation.target_point_board.y(),
                                      observation.target_point_board.z(),
                                      1.0);
    const Eigen::Vector4d point_world = board_it->second * point_board;
    object_points.push_back(point_world.head<3>());
    image_points.push_back(cv::Point2f(
        static_cast<float>(observation.observed_image_xy.x()),
        static_cast<float>(observation.observed_image_xy.y())));
  }

  if (object_points.size() < 4) {
    return false;
  }

  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model = scene_state.cam0.camera_model;
  intrinsics.distortion_model = scene_state.cam0.distortion_model;
  intrinsics.resolution =
      cv::Size(scene_state.cam0.resolution[0], scene_state.cam0.resolution[1]);
  intrinsics.SetIntrinsicsVector(scene_state.cam0.intrinsics);
  intrinsics.SetDistortionVector(scene_state.cam0.distortion_coeffs);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  double rmse = 0.0;
  if (!EstimatePoseFromObjectPoints(intrinsics, object_points, image_points, &pose, &rmse)) {
    return false;
  }
  *T_cam0_world = ToMatrix4d(pose);
  return true;
}

bool RefitCam0BoardPoseFromOuterObservations(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    int pair_index,
    int board_id,
    Eigen::Matrix4d* T_cam0_board) {
  if (T_cam0_board == nullptr) {
    throw std::runtime_error(
        "RefitCam0BoardPoseFromOuterObservations requires output pose.");
  }
  std::vector<Eigen::Vector3d> object_points;
  std::vector<cv::Point2f> image_points;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.camera_index != 0 ||
        observation.board_id != board_id ||
        observation.point_type != JointPointType::Outer ||
        !observation.used_in_solver) {
      continue;
    }
    object_points.push_back(observation.target_point_board);
    image_points.push_back(cv::Point2f(
        static_cast<float>(observation.observed_image_xy.x()),
        static_cast<float>(observation.observed_image_xy.y())));
  }
  if (object_points.size() < 4) {
    return false;
  }

  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam0));
  cv::Mat rvec;
  cv::Mat tvec;
  if (!camera.estimateTransformation(BuildObjectPoints(object_points),
                                     image_points, &rvec, &tvec)) {
    return false;
  }
  *T_cam0_board = ToMatrix4d(MakePose(rvec, tvec));
  return true;
}

double EvaluateStereoBoardPoseOuterRmse(
    const std::vector<StereoBoardPoseObservation>& observations,
    const StereoSceneState& scene_state,
    const Eigen::Isometry3d& T_cam0_board) {
  if (observations.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam1));
  const Eigen::Isometry3d T_cam1_cam0 = ToIsometry3d(scene_state.T_cam1_cam0);

  double squared_error_sum = 0.0;
  int count = 0;
  for (const StereoBoardPoseObservation& observation : observations) {
    const Eigen::Vector3d point_cam0 =
        T_cam0_board * observation.object_point_board;
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    bool valid_projection = false;
    if (observation.camera_index == 0) {
      valid_projection = cam0.vsEuclideanToKeypoint(point_cam0, &predicted);
    } else {
      valid_projection = cam1.vsEuclideanToKeypoint(
          T_cam1_cam0 * point_cam0, &predicted);
    }
    if (!valid_projection) {
      continue;
    }
    const Eigen::Vector2d residual = predicted - observation.observed_image_xy;
    squared_error_sum += residual.squaredNorm();
    ++count;
  }
  if (count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(squared_error_sum / static_cast<double>(count));
}

bool RefitStereoBoardPoseFromOuterObservations(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    int pair_index,
    int board_id,
    const StereoResidualEvaluationOptions& options,
    Eigen::Matrix4d* T_cam0_board) {
  if (T_cam0_board == nullptr) {
    throw std::runtime_error(
        "RefitStereoBoardPoseFromOuterObservations requires output pose.");
  }
  std::vector<StereoBoardPoseObservation> observations;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.board_id != board_id ||
        observation.point_type != JointPointType::Outer ||
        !observation.used_in_solver) {
      continue;
    }
    StereoBoardPoseObservation pose_observation;
    pose_observation.camera_index = observation.camera_index;
    pose_observation.object_point_board = observation.target_point_board;
    pose_observation.observed_image_xy = observation.observed_image_xy;
    observations.push_back(pose_observation);
  }
  if (observations.size() < 8) {
    return RefitCam0BoardPoseFromOuterObservations(
        dataset, scene_state, pair_index, board_id, T_cam0_board);
  }

  Eigen::Matrix4d seed_pose = Eigen::Matrix4d::Identity();
  if (!RefitCam0BoardPoseFromOuterObservations(
          dataset, scene_state, pair_index, board_id, &seed_pose)) {
    return false;
  }

  Eigen::Isometry3d current_pose = ToIsometry3d(seed_pose);
  double current_rmse =
      EvaluateStereoBoardPoseOuterRmse(observations, scene_state, current_pose);
  if (!std::isfinite(current_rmse)) {
    *T_cam0_board = seed_pose;
    return true;
  }

  const double step = options.symmetric_refit_step;
  for (int iteration = 0; iteration < options.symmetric_refit_max_iterations;
       ++iteration) {
    Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    for (int axis = 0; axis < 6; ++axis) {
      Eigen::Matrix<double, 6, 1> plus_delta = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> minus_delta = Eigen::Matrix<double, 6, 1>::Zero();
      plus_delta(axis) = step;
      minus_delta(axis) = -step;
      const double plus_rmse = EvaluateStereoBoardPoseOuterRmse(
          observations, scene_state, ApplyPoseDelta(current_pose, plus_delta));
      const double minus_rmse = EvaluateStereoBoardPoseOuterRmse(
          observations, scene_state, ApplyPoseDelta(current_pose, minus_delta));
      if (!std::isfinite(plus_rmse) || !std::isfinite(minus_rmse)) {
        continue;
      }
      gradient(axis) = (plus_rmse - minus_rmse) / (2.0 * step);
      hessian(axis, axis) =
          std::max(1e-6,
                   (plus_rmse - 2.0 * current_rmse + minus_rmse) /
                       (step * step));
    }
    for (int axis = 0; axis < 6; ++axis) {
      hessian(axis, axis) += 1e-3;
    }
    if (!hessian.allFinite() || !gradient.allFinite()) {
      break;
    }
    const Eigen::Matrix<double, 6, 1> delta = -hessian.ldlt().solve(gradient);
    if (!delta.allFinite()) {
      break;
    }
    const Eigen::Isometry3d candidate_pose =
        ApplyPoseDelta(current_pose, delta);
    const double candidate_rmse = EvaluateStereoBoardPoseOuterRmse(
        observations, scene_state, candidate_pose);
    if (!std::isfinite(candidate_rmse) ||
        candidate_rmse + 1e-9 >= current_rmse) {
      break;
    }
    current_pose = candidate_pose;
    current_rmse = candidate_rmse;
    if (delta.norm() < 1e-5) {
      break;
    }
  }

  *T_cam0_board = ToMatrix4d(current_pose);
  return true;
}

std::vector<StereoOuterPoseObservation> CollectStereoOuterPoseObservations(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    int pair_index) {
  std::vector<StereoOuterPoseObservation> observations;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.point_type != JointPointType::Outer ||
        !observation.used_in_solver) {
      continue;
    }
    const auto board_it = scene_state.T_world_board_by_id.find(observation.board_id);
    if (board_it == scene_state.T_world_board_by_id.end()) {
      continue;
    }
    const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                      observation.target_point_board.y(),
                                      observation.target_point_board.z(),
                                      1.0);
    const Eigen::Vector4d point_world = board_it->second * point_board;
    StereoOuterPoseObservation stereo_observation;
    stereo_observation.camera_index = observation.camera_index;
    stereo_observation.board_id = observation.board_id;
    stereo_observation.object_point_world = point_world.head<3>();
    stereo_observation.observed_image_xy = observation.observed_image_xy;
    observations.push_back(stereo_observation);
  }
  return observations;
}

double EvaluateStereoOuterPoseRmse(
    const std::vector<StereoOuterPoseObservation>& observations,
    const StereoSceneState& scene_state,
    const Eigen::Isometry3d& T_cam0_world) {
  if (observations.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam1));
  const Eigen::Isometry3d T_cam1_cam0 = ToIsometry3d(scene_state.T_cam1_cam0);

  double squared_error_sum = 0.0;
  int count = 0;
  for (const StereoOuterPoseObservation& observation : observations) {
    const Eigen::Vector3d point_cam0 = T_cam0_world * observation.object_point_world;
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    bool valid_projection = false;
    if (observation.camera_index == 0) {
      valid_projection = cam0.vsEuclideanToKeypoint(point_cam0, &predicted);
    } else {
      valid_projection = cam1.vsEuclideanToKeypoint(
          T_cam1_cam0 * point_cam0, &predicted);
    }
    if (!valid_projection) {
      continue;
    }
    const Eigen::Vector2d residual = predicted - observation.observed_image_xy;
    squared_error_sum += residual.squaredNorm();
    ++count;
  }
  if (count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(squared_error_sum / static_cast<double>(count));
}

bool RefitPairPoseFromStereoOuterObservations(const StereoMeasurementDataset& dataset,
                                              const StereoSceneState& scene_state,
                                              int pair_index,
                                              const StereoResidualEvaluationOptions& options,
                                              StereoPairResidualSummary* pair_summary,
                                              Eigen::Matrix4d* T_cam0_world) {
  if (T_cam0_world == nullptr) {
    throw std::runtime_error(
        "RefitPairPoseFromStereoOuterObservations requires output pose.");
  }
  if (pair_summary == nullptr) {
    throw std::runtime_error(
        "RefitPairPoseFromStereoOuterObservations requires pair summary.");
  }
  pair_summary->used_symmetric_refit = true;

  Eigen::Matrix4d seed_pose = Eigen::Matrix4d::Identity();
  bool have_seed = false;
  const auto pose_it = scene_state.T_cam0_world_by_pair.find(pair_index);
  if (pose_it != scene_state.T_cam0_world_by_pair.end()) {
    seed_pose = pose_it->second;
    have_seed = true;
  }
  if (!have_seed &&
      !RefitPairPoseFromOuterObservations(dataset, scene_state, pair_index, &seed_pose)) {
    return false;
  }

  const std::vector<StereoOuterPoseObservation> stereo_observations =
      CollectStereoOuterPoseObservations(dataset, scene_state, pair_index);
  if (stereo_observations.size() < 4) {
    *T_cam0_world = seed_pose;
    pair_summary->pose_refit_success = have_seed;
    pair_summary->refit_fell_back_to_seed = true;
    return true;
  }

  Eigen::Isometry3d current_pose = ToIsometry3d(seed_pose);
  double current_rmse =
      EvaluateStereoOuterPoseRmse(stereo_observations, scene_state, current_pose);
  if (!std::isfinite(current_rmse)) {
    *T_cam0_world = seed_pose;
    pair_summary->pose_refit_success = have_seed;
    pair_summary->refit_fell_back_to_seed = true;
    return true;
  }

  const double step = options.symmetric_refit_step;
  bool refined = false;
  for (int iteration = 0; iteration < options.symmetric_refit_max_iterations;
       ++iteration) {
    Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    for (int axis = 0; axis < 6; ++axis) {
      Eigen::Matrix<double, 6, 1> plus_delta = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> minus_delta = Eigen::Matrix<double, 6, 1>::Zero();
      plus_delta(axis) = step;
      minus_delta(axis) = -step;
      const double plus_rmse = EvaluateStereoOuterPoseRmse(
          stereo_observations, scene_state, ApplyPoseDelta(current_pose, plus_delta));
      const double minus_rmse = EvaluateStereoOuterPoseRmse(
          stereo_observations, scene_state, ApplyPoseDelta(current_pose, minus_delta));
      if (!std::isfinite(plus_rmse) || !std::isfinite(minus_rmse)) {
        continue;
      }
      gradient(axis) = (plus_rmse - minus_rmse) / (2.0 * step);
      hessian(axis, axis) =
          std::max(1e-6, (plus_rmse - 2.0 * current_rmse + minus_rmse) / (step * step));
    }
    for (int axis = 0; axis < 6; ++axis) {
      hessian(axis, axis) += 1e-3;
    }
    if (!hessian.allFinite() || !gradient.allFinite()) {
      break;
    }
    const Eigen::Matrix<double, 6, 1> delta =
        -hessian.ldlt().solve(gradient);
    if (!delta.allFinite()) {
      break;
    }
    const Eigen::Isometry3d candidate_pose = ApplyPoseDelta(current_pose, delta);
    const double candidate_rmse =
        EvaluateStereoOuterPoseRmse(stereo_observations, scene_state, candidate_pose);
    if (!std::isfinite(candidate_rmse) || candidate_rmse + 1e-9 >= current_rmse) {
      break;
    }
    current_pose = candidate_pose;
    current_rmse = candidate_rmse;
    refined = true;
    if (delta.norm() < 1e-5) {
      break;
    }
  }

  if (refined) {
    *T_cam0_world = ToMatrix4d(current_pose);
    pair_summary->pose_refit_success = true;
    return true;
  }

  *T_cam0_world = seed_pose;
  pair_summary->pose_refit_success = have_seed;
  pair_summary->refit_fell_back_to_seed = true;
  return have_seed;
}

}  // namespace

StereoResidualEvaluator::StereoResidualEvaluator(
    StereoResidualEvaluationOptions options)
    : options_(std::move(options)) {}

StereoResidualSummary StereoResidualEvaluator::Evaluate(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    const std::set<int>& pair_indices,
    const std::string& split_label) const {
  StereoResidualSummary summary;
  summary.split_label = split_label;
  summary.holdout_refit =
      options_.refit_pair_pose || options_.extrinsic_only_local_board_pose;
  summary.pair_count = static_cast<int>(pair_indices.size());

  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam1));
  if (!cam0.IsValid() || !cam1.IsValid()) {
    summary.failure_reason = "Invalid fixed stereo camera calibration.";
    return summary;
  }

  ResidualAccumulator total_accumulator;
  ResidualAccumulator outer_accumulator;
  ResidualAccumulator internal_accumulator;
  ResidualAccumulator shared_total_accumulator;
  ResidualAccumulator shared_outer_accumulator;
  ResidualAccumulator shared_internal_accumulator;
  ResidualAccumulator cam0_only_total_accumulator;
  ResidualAccumulator cam1_only_total_accumulator;
  std::map<int, ResidualAccumulator> camera_accumulators;
  std::map<int, ResidualAccumulator> shared_camera_accumulators;
  std::map<int, ResidualAccumulator> cam0_only_camera_accumulators;
  std::map<int, ResidualAccumulator> cam1_only_camera_accumulators;
  std::map<int, ResidualAccumulator> board_accumulators;
  std::map<int, ResidualAccumulator> board_shared_accumulators;
  std::map<int, ResidualAccumulator> board_shared_cam0_accumulators;
  std::map<int, ResidualAccumulator> board_shared_cam1_accumulators;
  std::map<int, ResidualAccumulator> board_shared_outer_accumulators;
  std::map<int, ResidualAccumulator> board_shared_internal_accumulators;

  for (int pair_index : pair_indices) {
    StereoPairResidualSummary pair_summary;
    pair_summary.pair_index = pair_index;
    const StereoFramePair* pair = FindPair(dataset, pair_index);
    if (pair != nullptr) {
      pair_summary.left_frame_label = pair->left_frame_label;
      pair_summary.right_frame_label = pair->right_frame_label;
      pair_summary.is_training = pair->is_training;
      pair_summary.shared_board_count =
          dataset.pair_shared_board_count.count(pair_index) > 0
              ? dataset.pair_shared_board_count.at(pair_index)
              : 0;
      pair_summary.cam0_only_board_count =
          dataset.pair_cam0_only_board_count.count(pair_index) > 0
              ? dataset.pair_cam0_only_board_count.at(pair_index)
              : 0;
      pair_summary.cam1_only_board_count =
          dataset.pair_cam1_only_board_count.count(pair_index) > 0
              ? dataset.pair_cam1_only_board_count.at(pair_index)
              : 0;
    }

    Eigen::Matrix4d T_cam0_world = Eigen::Matrix4d::Identity();
    bool have_pose = false;
    if (options_.extrinsic_only_local_board_pose) {
      have_pose = true;
      pair_summary.pose_refit_success = true;
      pair_summary.pose_source = options_.use_committed_pair_board_pose
                                     ? "committed_pair_board_pose"
                                     : "local_stereo_board_refit_extrinsic_only";
    } else {
      const auto pose_it = scene_state.T_cam0_world_by_pair.find(pair_index);
      if (pose_it != scene_state.T_cam0_world_by_pair.end()) {
        T_cam0_world = pose_it->second;
        have_pose = true;
      }
    }
    if (!options_.extrinsic_only_local_board_pose && options_.refit_pair_pose) {
      if (options_.pair_pose_refit_mode ==
          StereoPairPoseRefitMode::StereoSymmetric) {
        have_pose = RefitPairPoseFromStereoOuterObservations(
            dataset, scene_state, pair_index, options_, &pair_summary,
            &T_cam0_world);
        pair_summary.pose_source = pair_summary.pose_refit_success
                                       ? (pair_summary.refit_fell_back_to_seed
                                              ? "bootstrapped"
                                              : "refit")
                                       : "failed";
      } else {
        have_pose = RefitPairPoseFromOuterObservations(
            dataset, scene_state, pair_index, &T_cam0_world);
        pair_summary.pose_refit_success = have_pose;
        pair_summary.pose_source = have_pose ? "bootstrapped" : "failed";
      }
    } else if (!options_.extrinsic_only_local_board_pose && have_pose) {
      pair_summary.pose_source = "propagated";
    }
    if (!options_.extrinsic_only_local_board_pose &&
        (!have_pose || !PairUsesKnownBoards(dataset, scene_state, pair_index))) {
      pair_summary.failure_reason = "pair pose unavailable or no known boards for evaluation";
      summary.pair_summaries.push_back(pair_summary);
      ++summary.unevaluable_pair_count;
      continue;
    }
    if (pair_summary.shared_board_count > 0) {
      ++summary.shared_board_pair_count;
    } else {
      ++summary.single_camera_only_pair_count;
    }

    ResidualAccumulator pair_total;
    ResidualAccumulator pair_outer;
    ResidualAccumulator pair_internal;
    ResidualAccumulator pair_shared_total;
    ResidualAccumulator pair_shared_outer;
    ResidualAccumulator pair_shared_internal;
    ResidualAccumulator pair_cam0_only_total;
    ResidualAccumulator pair_cam1_only_total;
    std::map<int, ResidualAccumulator> pair_camera_accumulators;
    std::map<int, ResidualAccumulator> pair_shared_camera_accumulators;
    std::map<int, Eigen::Matrix4d> local_T_cam0_board_by_board_id;
    std::set<int> failed_local_board_ids;
    const Eigen::Isometry3d T_cam1_cam0 =
        ToIsometry3d(scene_state.T_cam1_cam0);

    for (const StereoObservation& observation : dataset.observations) {
      if (observation.pair_index != pair_index || !observation.used_in_solver) {
        continue;
      }

      const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                        observation.target_point_board.y(),
                                        observation.target_point_board.z(), 1.0);
      Eigen::Vector3d point_cam0 = Eigen::Vector3d::Zero();
      if (options_.extrinsic_only_local_board_pose) {
        if (!ContainsBoard(dataset.pair_shared_board_ids, pair_index,
                           observation.board_id)) {
          continue;
        }
        if (failed_local_board_ids.count(observation.board_id) > 0) {
          continue;
        }
        auto local_pose_it =
            local_T_cam0_board_by_board_id.find(observation.board_id);
        if (local_pose_it == local_T_cam0_board_by_board_id.end()) {
          Eigen::Matrix4d T_cam0_board = Eigen::Matrix4d::Identity();
          bool have_local_pose = false;
          if (options_.use_committed_pair_board_pose) {
            const auto committed_pose_it =
                scene_state.T_cam0_board_by_pair_board.find(
                    std::make_pair(pair_index, observation.board_id));
            if (committed_pose_it !=
                scene_state.T_cam0_board_by_pair_board.end()) {
              T_cam0_board = committed_pose_it->second;
              have_local_pose = T_cam0_board.allFinite();
            }
          }
          if (!have_local_pose) {
            have_local_pose = RefitStereoBoardPoseFromOuterObservations(
                dataset, scene_state, pair_index, observation.board_id,
                options_, &T_cam0_board);
          }
          if (!have_local_pose) {
            failed_local_board_ids.insert(observation.board_id);
            continue;
          }
          const double local_stereo_outer_rmse =
              EvaluateStereoBoardPoseOuterRmse(
                  [&]() {
                    std::vector<StereoBoardPoseObservation> observations;
                    for (const StereoObservation& candidate :
                         dataset.observations) {
                      if (candidate.pair_index != pair_index ||
                          candidate.board_id != observation.board_id ||
                          candidate.point_type != JointPointType::Outer ||
                          !candidate.used_in_solver) {
                        continue;
                      }
                      StereoBoardPoseObservation pose_observation;
                      pose_observation.camera_index = candidate.camera_index;
                      pose_observation.object_point_board =
                          candidate.target_point_board;
                      pose_observation.observed_image_xy =
                          candidate.observed_image_xy;
                      observations.push_back(pose_observation);
                    }
                    return observations;
                  }(),
                  scene_state, ToIsometry3d(T_cam0_board));
          if (!std::isfinite(local_stereo_outer_rmse)) {
            failed_local_board_ids.insert(observation.board_id);
            continue;
          }
          local_pose_it =
              local_T_cam0_board_by_board_id
                  .insert(std::make_pair(observation.board_id, T_cam0_board))
                  .first;
        }
        point_cam0 = (local_pose_it->second * point_board).head<3>();
      } else {
        const auto board_it =
            scene_state.T_world_board_by_id.find(observation.board_id);
        if (board_it == scene_state.T_world_board_by_id.end()) {
          continue;
        }
        const Eigen::Vector4d point_world = board_it->second * point_board;
        point_cam0 = (T_cam0_world * point_world).head<3>();
      }
      Eigen::Vector3d point_camera = point_cam0;
      Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
      bool valid_projection = false;
      if (observation.camera_index == 0) {
        valid_projection = cam0.vsEuclideanToKeypoint(point_camera, &predicted);
      } else {
        valid_projection = cam1.vsEuclideanToKeypoint(
            T_cam1_cam0 * point_cam0, &predicted);
      }
      if (!valid_projection) {
        continue;
      }

      const Eigen::Vector2d residual_xy = predicted - observation.observed_image_xy;
      const double squared_error = residual_xy.squaredNorm();
      const bool is_outer = observation.point_type == JointPointType::Outer;
      const bool is_shared_pair =
          ContainsBoard(dataset.pair_shared_board_ids, pair_index,
                        observation.board_id);
      const bool is_cam0_only_pair =
          ContainsBoard(dataset.pair_cam0_only_board_ids, pair_index,
                        observation.board_id);
      const bool is_cam1_only_pair =
          ContainsBoard(dataset.pair_cam1_only_board_ids, pair_index,
                        observation.board_id);

      auto accumulate = [&](ResidualAccumulator* accumulator) {
        accumulator->count += 1;
        accumulator->squared_sum += squared_error;
        accumulator->sum_x += residual_xy.x();
        accumulator->sum_y += residual_xy.y();
        accumulator->sum_sq_x += residual_xy.x() * residual_xy.x();
        accumulator->sum_sq_y += residual_xy.y() * residual_xy.y();
      };

      accumulate(&pair_total);
      accumulate(&total_accumulator);
      accumulate(&pair_camera_accumulators[observation.camera_index]);
      accumulate(&camera_accumulators[observation.camera_index]);
      accumulate(&board_accumulators[observation.board_id]);
      if (is_outer) {
        accumulate(&pair_outer);
        accumulate(&outer_accumulator);
      } else {
        accumulate(&pair_internal);
        accumulate(&internal_accumulator);
      }
      if (is_shared_pair) {
        accumulate(&pair_shared_total);
        accumulate(&pair_shared_camera_accumulators[observation.camera_index]);
        accumulate(&board_shared_accumulators[observation.board_id]);
        accumulate(&shared_total_accumulator);
        accumulate(&shared_camera_accumulators[observation.camera_index]);
        if (observation.camera_index == 0) {
          accumulate(&board_shared_cam0_accumulators[observation.board_id]);
        } else if (observation.camera_index == 1) {
          accumulate(&board_shared_cam1_accumulators[observation.board_id]);
        }
        if (is_outer) {
          accumulate(&pair_shared_outer);
          accumulate(&board_shared_outer_accumulators[observation.board_id]);
          accumulate(&shared_outer_accumulator);
        } else {
          accumulate(&pair_shared_internal);
          accumulate(&board_shared_internal_accumulators[observation.board_id]);
          accumulate(&shared_internal_accumulator);
        }
      } else if (is_cam0_only_pair) {
        accumulate(&pair_cam0_only_total);
        accumulate(&cam0_only_total_accumulator);
        accumulate(&cam0_only_camera_accumulators[observation.camera_index]);
      } else if (is_cam1_only_pair) {
        accumulate(&pair_cam1_only_total);
        accumulate(&cam1_only_total_accumulator);
        accumulate(&cam1_only_camera_accumulators[observation.camera_index]);
      }
    }

    if (pair_total.count <= 0) {
      pair_summary.failure_reason = "no valid projected points";
      summary.pair_summaries.push_back(pair_summary);
      ++summary.unevaluable_pair_count;
      continue;
    }

    pair_summary.used_in_metrics = true;
    pair_summary.point_count = pair_total.count;
    pair_summary.outer_point_count = pair_outer.count;
    pair_summary.internal_point_count = pair_internal.count;
    pair_summary.cam0_point_count = pair_camera_accumulators[0].count;
    pair_summary.cam1_point_count = pair_camera_accumulators[1].count;
    pair_summary.shared_point_count = pair_shared_total.count;
    pair_summary.shared_outer_point_count = pair_shared_outer.count;
    pair_summary.shared_internal_point_count = pair_shared_internal.count;
    pair_summary.overall_rmse = ComputeRmse(pair_total.squared_sum, pair_total.count);
    pair_summary.cam0_rmse =
        ComputeRmse(pair_camera_accumulators[0].squared_sum,
                    pair_camera_accumulators[0].count);
    pair_summary.cam1_rmse =
        ComputeRmse(pair_camera_accumulators[1].squared_sum,
                    pair_camera_accumulators[1].count);
    pair_summary.outer_rmse = ComputeRmse(pair_outer.squared_sum, pair_outer.count);
    pair_summary.internal_rmse =
        ComputeRmse(pair_internal.squared_sum, pair_internal.count);
    pair_summary.shared_cam0_rmse =
        ComputeRmse(pair_shared_camera_accumulators[0].squared_sum,
                    pair_shared_camera_accumulators[0].count);
    pair_summary.shared_cam1_rmse =
        ComputeRmse(pair_shared_camera_accumulators[1].squared_sum,
                    pair_shared_camera_accumulators[1].count);
    pair_summary.shared_outer_rmse =
        ComputeRmse(pair_shared_outer.squared_sum, pair_shared_outer.count);
    pair_summary.shared_internal_rmse =
        ComputeRmse(pair_shared_internal.squared_sum, pair_shared_internal.count);
    pair_summary.cam0_only_rmse =
        ComputeRmse(pair_cam0_only_total.squared_sum, pair_cam0_only_total.count);
    pair_summary.cam1_only_rmse =
        ComputeRmse(pair_cam1_only_total.squared_sum, pair_cam1_only_total.count);
    pair_summary.mean_residual_x = pair_total.sum_x / static_cast<double>(pair_total.count);
    pair_summary.mean_residual_y = pair_total.sum_y / static_cast<double>(pair_total.count);
    pair_summary.std_residual_x =
        ComputeStd(pair_total.sum_x, pair_total.sum_sq_x, pair_total.count);
    pair_summary.std_residual_y =
        ComputeStd(pair_total.sum_y, pair_total.sum_sq_y, pair_total.count);
    summary.pair_summaries.push_back(pair_summary);
    ++summary.used_pair_count;
  }

  summary.point_count = total_accumulator.count;
  summary.outer_point_count = outer_accumulator.count;
  summary.internal_point_count = internal_accumulator.count;
  summary.total_stereo_rmse =
      ComputeRmse(total_accumulator.squared_sum, total_accumulator.count);
  summary.cam0_rmse =
      ComputeRmse(camera_accumulators[0].squared_sum, camera_accumulators[0].count);
  summary.cam1_rmse =
      ComputeRmse(camera_accumulators[1].squared_sum, camera_accumulators[1].count);
  if (summary.cam0_rmse > 0.0) {
    summary.cam1_over_cam0_rmse_ratio = summary.cam1_rmse / summary.cam0_rmse;
  }
  summary.cam_residual_balance_gap = std::abs(summary.cam1_rmse - summary.cam0_rmse);
  summary.outer_only_rmse =
      ComputeRmse(outer_accumulator.squared_sum, outer_accumulator.count);
  summary.internal_only_rmse =
      ComputeRmse(internal_accumulator.squared_sum, internal_accumulator.count);
  summary.shared_point_count = shared_total_accumulator.count;
  summary.shared_outer_point_count = shared_outer_accumulator.count;
  summary.shared_internal_point_count = shared_internal_accumulator.count;
  summary.shared_total_rmse = ComputeRmse(shared_total_accumulator.squared_sum,
                                          shared_total_accumulator.count);
  summary.shared_cam0_rmse = ComputeRmse(
      shared_camera_accumulators[0].squared_sum,
      shared_camera_accumulators[0].count);
  summary.shared_cam1_rmse = ComputeRmse(
      shared_camera_accumulators[1].squared_sum,
      shared_camera_accumulators[1].count);
  summary.cam0_only_point_count = cam0_only_total_accumulator.count;
  summary.cam1_only_point_count = cam1_only_total_accumulator.count;
  summary.cam0_only_total_rmse = ComputeRmse(
      cam0_only_total_accumulator.squared_sum,
      cam0_only_total_accumulator.count);
  summary.cam1_only_total_rmse = ComputeRmse(
      cam1_only_total_accumulator.squared_sum,
      cam1_only_total_accumulator.count);
  if (total_accumulator.count > 0) {
    summary.mean_residual_x =
        total_accumulator.sum_x / static_cast<double>(total_accumulator.count);
    summary.mean_residual_y =
        total_accumulator.sum_y / static_cast<double>(total_accumulator.count);
    summary.std_residual_x = ComputeStd(
        total_accumulator.sum_x, total_accumulator.sum_sq_x, total_accumulator.count);
    summary.std_residual_y = ComputeStd(
        total_accumulator.sum_y, total_accumulator.sum_sq_y, total_accumulator.count);
  }

  for (const auto& entry : camera_accumulators) {
    StereoCameraResidualSummary camera_summary;
    camera_summary.camera_index = entry.first;
    camera_summary.point_count = entry.second.count;
    camera_summary.rmse = ComputeRmse(entry.second.squared_sum, entry.second.count);
    camera_summary.shared_point_count =
        shared_camera_accumulators[entry.first].count;
    camera_summary.shared_rmse = ComputeRmse(
        shared_camera_accumulators[entry.first].squared_sum,
        shared_camera_accumulators[entry.first].count);
    camera_summary.cam0_only_point_count =
        cam0_only_camera_accumulators[entry.first].count;
    camera_summary.cam0_only_rmse = ComputeRmse(
        cam0_only_camera_accumulators[entry.first].squared_sum,
        cam0_only_camera_accumulators[entry.first].count);
    camera_summary.cam1_only_point_count =
        cam1_only_camera_accumulators[entry.first].count;
    camera_summary.cam1_only_rmse = ComputeRmse(
        cam1_only_camera_accumulators[entry.first].squared_sum,
        cam1_only_camera_accumulators[entry.first].count);
    summary.camera_summaries.push_back(camera_summary);
  }

  for (const auto& entry : board_accumulators) {
    StereoBoardResidualSummary board_summary;
    board_summary.board_id = entry.first;
    board_summary.point_count = entry.second.count;
    board_summary.rmse = ComputeRmse(entry.second.squared_sum, entry.second.count);
    board_summary.shared_point_count = board_shared_accumulators[entry.first].count;
    board_summary.shared_cam0_point_count =
        board_shared_cam0_accumulators[entry.first].count;
    board_summary.shared_cam1_point_count =
        board_shared_cam1_accumulators[entry.first].count;
    board_summary.shared_outer_point_count =
        board_shared_outer_accumulators[entry.first].count;
    board_summary.shared_internal_point_count =
        board_shared_internal_accumulators[entry.first].count;
    board_summary.shared_cam0_rmse = ComputeRmse(
        board_shared_cam0_accumulators[entry.first].squared_sum,
        board_shared_cam0_accumulators[entry.first].count);
    board_summary.shared_cam1_rmse = ComputeRmse(
        board_shared_cam1_accumulators[entry.first].squared_sum,
        board_shared_cam1_accumulators[entry.first].count);
    board_summary.shared_outer_rmse = ComputeRmse(
        board_shared_outer_accumulators[entry.first].squared_sum,
        board_shared_outer_accumulators[entry.first].count);
    board_summary.shared_internal_rmse = ComputeRmse(
        board_shared_internal_accumulators[entry.first].squared_sum,
        board_shared_internal_accumulators[entry.first].count);
    std::set<int> contributing_pairs;
    for (const StereoObservation& observation : dataset.observations) {
      if (!observation.used_in_solver || observation.board_id != entry.first) {
        continue;
      }
      if (pair_indices.count(observation.pair_index) == 0) {
        continue;
      }
      if (observation.camera_index == 0) {
        ++board_summary.cam0_point_count;
      } else if (observation.camera_index == 1) {
        ++board_summary.cam1_point_count;
      }
      if (dataset.pair_shared_board_count.count(observation.pair_index) > 0 &&
          dataset.pair_shared_board_count.at(observation.pair_index) > 0) {
        contributing_pairs.insert(observation.pair_index);
      }
    }
    board_summary.shared_pair_count = static_cast<int>(contributing_pairs.size());
    summary.board_summaries.push_back(board_summary);
  }

  summary.success = summary.point_count > 0;
  if (!summary.success && summary.failure_reason.empty()) {
    summary.failure_reason = "No stereo residuals were accumulated.";
  }
  return summary;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
