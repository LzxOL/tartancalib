#include <aslam/cameras/apriltag_internal/IterativeCoarseCalibrationExperiment.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

struct ImageRecord {
  int index = -1;
  boost::filesystem::path path;
  std::string stem;
  std::string group_name;
  cv::Mat image;
};

struct BoardKey {
  int image_index = -1;
  int board_id = -1;

  bool operator<(const BoardKey& other) const {
    if (image_index != other.image_index) {
      return image_index < other.image_index;
    }
    return board_id < other.board_id;
  }
};

struct BoardDetection {
  bool available = false;
  ApriltagInternalDetectionResult result;
  std::string error_text;
};

struct DsIntrinsics {
  double xi = 0.0;
  double alpha = 0.0;
  double fu = 0.0;
  double fv = 0.0;
  double cu = 0.0;
  double cv = 0.0;
  cv::Size resolution;
};

struct PoseEstimate {
  bool valid = false;
  cv::Mat rvec;
  cv::Mat tvec;
  double mean_reprojection_error = std::numeric_limits<double>::infinity();
};

struct CameraCorrespondence {
  Eigen::Vector3d target_xyz = Eigen::Vector3d::Zero();
  cv::Point2f image_xy{};
  CornerType corner_type = CornerType::Outer;
  double quality = 0.0;
  bool from_internal = false;
};

struct MetricsAccumulator {
  int observation_count = 0;
  int successful_observations = 0;
  int total_points = 0;
  int valid_points = 0;
  int image_evidence_valid_points = 0;
  int lcorner_points = 0;
  int lcorner_valid = 0;
  int xcorner_points = 0;
  int xcorner_valid = 0;
  double sum_q_refine = 0.0;
  double sum_template_quality = 0.0;
  double sum_gradient_quality = 0.0;
  double sum_final_quality = 0.0;
  double sum_image_template_quality = 0.0;
  double sum_image_gradient_quality = 0.0;
  double sum_image_centering_quality = 0.0;
  double sum_image_final_quality = 0.0;
  double sum_predicted_to_refined = 0.0;

  void AddObservation(const ApriltagInternalDetectionResult& result) {
    ++observation_count;
    successful_observations += result.success ? 1 : 0;
    for (std::size_t index = 0; index < result.internal_corner_debug.size(); ++index) {
      const InternalCornerDebugInfo& debug = result.internal_corner_debug[index];
      ++total_points;
      valid_points += debug.valid ? 1 : 0;
      image_evidence_valid_points += debug.image_evidence_valid ? 1 : 0;
      sum_q_refine += debug.q_refine;
      sum_template_quality += debug.template_quality;
      sum_gradient_quality += debug.gradient_quality;
      sum_final_quality += debug.final_quality;
      sum_image_template_quality += debug.image_template_quality;
      sum_image_gradient_quality += debug.image_gradient_quality;
      sum_image_centering_quality += debug.image_centering_quality;
      sum_image_final_quality += debug.image_final_quality;
      sum_predicted_to_refined += debug.predicted_to_refined_displacement;

      if (debug.corner_type == CornerType::LCorner) {
        ++lcorner_points;
        lcorner_valid += debug.valid ? 1 : 0;
      } else if (debug.corner_type == CornerType::XCorner) {
        ++xcorner_points;
        xcorner_valid += debug.valid ? 1 : 0;
      }
    }
  }

  double Average(double sum) const {
    return total_points > 0 ? sum / static_cast<double>(total_points) : 0.0;
  }

  double Score() const {
    return static_cast<double>(valid_points) + Average(sum_final_quality);
  }
};

struct IterationSummary {
  int iteration_index = 0;
  std::string label;
  DsIntrinsics camera;
  int board_observation_count = 0;
  int valid_pose_count = 0;
  int correspondence_count = 0;
  int outer_correspondence_count = 0;
  int internal_correspondence_count = 0;
  double camera_rmse = 0.0;
  double score = 0.0;
  MetricsAccumulator global_metrics;
  std::map<std::string, MetricsAccumulator> group_metrics;
};

typedef std::map<BoardKey, BoardDetection> DetectionMap;
typedef std::map<BoardKey, std::vector<CameraCorrespondence> > CorrespondenceMap;
typedef std::map<BoardKey, PoseEstimate> PoseMap;

std::string GroupNameFromStem(const std::string& stem) {
  const std::size_t dash = stem.find_last_of('-');
  if (dash == std::string::npos || dash == 0 || dash + 1 >= stem.size()) {
    return stem;
  }
  return stem.substr(0, dash);
}

bool IsImageFile(const boost::filesystem::path& path) {
  const std::string extension = path.extension().string();
  return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
         extension == ".bmp" || extension == ".tif" || extension == ".tiff";
}

bool ShouldKeepGroup(const std::string& group_name,
                     const std::set<std::string>& group_filters) {
  return group_filters.empty() || group_filters.count(group_name) > 0;
}

std::vector<ImageRecord> LoadDatasetImages(const std::string& image_dir,
                                           const std::vector<std::string>& group_filters) {
  if (!boost::filesystem::exists(image_dir)) {
    throw std::runtime_error("Image directory does not exist: " + image_dir);
  }

  std::set<std::string> filter_set(group_filters.begin(), group_filters.end());
  std::vector<boost::filesystem::path> image_paths;
  for (boost::filesystem::directory_iterator it(image_dir), end; it != end; ++it) {
    if (boost::filesystem::is_regular_file(it->path()) && IsImageFile(it->path())) {
      image_paths.push_back(it->path());
    }
  }
  std::sort(image_paths.begin(), image_paths.end());

  std::vector<ImageRecord> images;
  for (std::size_t index = 0; index < image_paths.size(); ++index) {
    const boost::filesystem::path& path = image_paths[index];
    const std::string stem = path.stem().string();
    const std::string group_name = GroupNameFromStem(stem);
    if (!ShouldKeepGroup(group_name, filter_set)) {
      continue;
    }

    cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      throw std::runtime_error("Failed to read image: " + path.string());
    }

    ImageRecord record;
    record.index = static_cast<int>(images.size());
    record.path = path;
    record.stem = stem;
    record.group_name = group_name;
    record.image = image;
    images.push_back(record);
  }

  if (images.empty()) {
    throw std::runtime_error("No images matched under " + image_dir);
  }

  const cv::Size reference_size = images.front().image.size();
  for (std::size_t index = 1; index < images.size(); ++index) {
    if (images[index].image.size() != reference_size) {
      throw std::runtime_error("All images must share the same resolution for this experiment.");
    }
  }

  return images;
}

cv::Mat ToGray(const cv::Mat& image) {
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image.clone();
  } else if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    throw std::runtime_error("Unsupported image format: expected 1, 3 or 4 channels.");
  }

  if (gray.depth() == CV_16U) {
    gray.convertTo(gray, CV_8U, 1.0 / 256.0);
  } else if (gray.depth() != CV_8U) {
    gray.convertTo(gray, CV_8U);
  }
  return gray;
}

IntermediateCameraConfig MakeCameraConfig(const DsIntrinsics& intrinsics) {
  IntermediateCameraConfig config;
  config.camera_model = "ds";
  config.distortion_model = "none";
  config.intrinsics.push_back(intrinsics.xi);
  config.intrinsics.push_back(intrinsics.alpha);
  config.intrinsics.push_back(intrinsics.fu);
  config.intrinsics.push_back(intrinsics.fv);
  config.intrinsics.push_back(intrinsics.cu);
  config.intrinsics.push_back(intrinsics.cv);
  config.resolution.push_back(intrinsics.resolution.width);
  config.resolution.push_back(intrinsics.resolution.height);
  return config;
}

ApriltagInternalConfig MakeBoardConfig(const ApriltagInternalConfig& base_config,
                                       int board_id,
                                       InternalProjectionMode mode,
                                       const DsIntrinsics* intrinsics) {
  ApriltagInternalConfig config = base_config;
  config.tag_id = board_id;
  config.internal_projection_mode = mode;
  if (intrinsics != nullptr) {
    config.intermediate_camera = MakeCameraConfig(*intrinsics);
  }
  return config;
}

DsIntrinsics MakeInitialIntrinsics(const cv::Size& resolution,
                                   const IterativeCoarseCalibrationExperimentOptions& options) {
  DsIntrinsics intrinsics;
  intrinsics.resolution = resolution;
  intrinsics.xi = options.init_xi;
  intrinsics.alpha = options.init_alpha;
  intrinsics.fu = options.init_fu_scale * static_cast<double>(resolution.width);
  intrinsics.fv = options.init_fv_scale * static_cast<double>(resolution.height);
  intrinsics.cu = 0.5 * static_cast<double>(resolution.width) + options.init_cu_offset;
  intrinsics.cv = 0.5 * static_cast<double>(resolution.height) + options.init_cv_offset;
  return intrinsics;
}

Eigen::Matrix<double, 6, 1> ToVector(const DsIntrinsics& intrinsics) {
  Eigen::Matrix<double, 6, 1> vector;
  vector << intrinsics.xi, intrinsics.alpha, intrinsics.fu, intrinsics.fv, intrinsics.cu, intrinsics.cv;
  return vector;
}

DsIntrinsics FromVector(const Eigen::Matrix<double, 6, 1>& vector, const cv::Size& resolution) {
  DsIntrinsics intrinsics;
  intrinsics.resolution = resolution;
  intrinsics.xi = vector[0];
  intrinsics.alpha = vector[1];
  intrinsics.fu = vector[2];
  intrinsics.fv = vector[3];
  intrinsics.cu = vector[4];
  intrinsics.cv = vector[5];
  return intrinsics;
}

bool ClampIntrinsicsInPlace(DsIntrinsics* intrinsics) {
  if (intrinsics == nullptr) {
    throw std::runtime_error("ClampIntrinsicsInPlace requires a valid pointer.");
  }

  intrinsics->xi = std::max(-0.95, std::min(2.5, intrinsics->xi));
  intrinsics->alpha = std::max(0.05, std::min(0.95, intrinsics->alpha));
  intrinsics->fu = std::max(50.0, std::min(3.0 * intrinsics->resolution.width, intrinsics->fu));
  intrinsics->fv = std::max(50.0, std::min(3.0 * intrinsics->resolution.height, intrinsics->fv));
  intrinsics->cu = std::max(0.0, std::min(static_cast<double>(intrinsics->resolution.width),
                                          intrinsics->cu));
  intrinsics->cv = std::max(0.0, std::min(static_cast<double>(intrinsics->resolution.height),
                                          intrinsics->cv));
  return intrinsics->fu > 0.0 && intrinsics->fv > 0.0;
}

bool ProjectDoubleSphere(const DsIntrinsics& intrinsics,
                         const Eigen::Vector3d& point_camera,
                         Eigen::Vector2d* keypoint) {
  if (keypoint == nullptr) {
    throw std::runtime_error("ProjectDoubleSphere requires a valid output pointer.");
  }

  const double x = point_camera.x();
  const double y = point_camera.y();
  const double z = point_camera.z();
  const double r2 = x * x + y * y;
  const double d1 = std::sqrt(r2 + z * z);
  const double temp = intrinsics.alpha <= 0.5
                          ? intrinsics.alpha / (1.0 - intrinsics.alpha)
                          : (1.0 - intrinsics.alpha) / intrinsics.alpha;
  const double fov_parameter =
      (temp + intrinsics.xi) /
      std::sqrt(2.0 * temp * intrinsics.xi + intrinsics.xi * intrinsics.xi + 1.0);

  if (z <= -(fov_parameter * d1)) {
    return false;
  }

  const double k = intrinsics.xi * d1 + z;
  const double d2 = std::sqrt(r2 + k * k);
  const double norm = intrinsics.alpha * d2 + (1.0 - intrinsics.alpha) * k;
  if (std::abs(norm) < 1e-12) {
    return false;
  }

  const double inv_norm = 1.0 / norm;
  (*keypoint)[0] = intrinsics.fu * x * inv_norm + intrinsics.cu;
  (*keypoint)[1] = intrinsics.fv * y * inv_norm + intrinsics.cv;
  return (*keypoint)[0] >= 0.0 &&
         (*keypoint)[0] < static_cast<double>(intrinsics.resolution.width) &&
         (*keypoint)[1] >= 0.0 &&
         (*keypoint)[1] < static_cast<double>(intrinsics.resolution.height);
}

Eigen::Vector3d TransformTargetPoint(const cv::Mat& rvec,
                                     const cv::Mat& tvec,
                                     const Eigen::Vector3d& target_xyz) {
  cv::Mat rotation_matrix;
  cv::Rodrigues(rvec, rotation_matrix);
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      rotation(row, col) = rotation_matrix.at<double>(row, col);
    }
  }

  Eigen::Vector3d translation(
      tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0));
  return rotation * target_xyz + translation;
}

std::vector<CameraCorrespondence> BuildOuterCorrespondences(
    const ApriltagInternalDetectionResult& result) {
  std::vector<CameraCorrespondence> correspondences;
  correspondences.reserve(result.corners.size());
  for (std::size_t index = 0; index < result.corners.size(); ++index) {
    const CornerMeasurement& measurement = result.corners[index];
    if (measurement.corner_type != CornerType::Outer) {
      continue;
    }

    CameraCorrespondence correspondence;
    correspondence.target_xyz = measurement.target_xyz;
    correspondence.image_xy = cv::Point2f(static_cast<float>(measurement.image_xy.x()),
                                          static_cast<float>(measurement.image_xy.y()));
    correspondence.corner_type = measurement.corner_type;
    correspondence.quality = measurement.quality;
    correspondence.from_internal = false;
    correspondences.push_back(correspondence);
  }
  return correspondences;
}

std::vector<CameraCorrespondence> BuildInternalCorrespondences(
    const ApriltagInternalDetectionResult& result,
    double quality_threshold) {
  std::vector<CameraCorrespondence> correspondences;
  correspondences.reserve(result.corners.size());
  for (std::size_t index = 0; index < result.corners.size(); ++index) {
    const CornerMeasurement& measurement = result.corners[index];
    if (measurement.corner_type == CornerType::Outer || !measurement.valid ||
        measurement.quality < quality_threshold) {
      continue;
    }

    CameraCorrespondence correspondence;
    correspondence.target_xyz = measurement.target_xyz;
    correspondence.image_xy = cv::Point2f(static_cast<float>(measurement.image_xy.x()),
                                          static_cast<float>(measurement.image_xy.y()));
    correspondence.corner_type = measurement.corner_type;
    correspondence.quality = measurement.quality;
    correspondence.from_internal = true;
    correspondences.push_back(correspondence);
  }
  return correspondences;
}

BoardDetection RunBoardDetection(const ImageRecord& image_record,
                                 const ApriltagInternalConfig& base_config,
                                 const ApriltagInternalDetectionOptions& options,
                                 int board_id,
                                 InternalProjectionMode mode,
                                 const DsIntrinsics* intrinsics) {
  BoardDetection detection;
  try {
    const ApriltagInternalConfig config =
        MakeBoardConfig(base_config, board_id, mode, intrinsics);
    ApriltagInternalDetector detector(config, options);
    detection.result = detector.Detect(image_record.image);
    detection.available = true;
  } catch (const std::exception& error) {
    detection.available = false;
    detection.error_text = error.what();
  }
  return detection;
}

DetectionMap RunDetectionPass(const std::vector<ImageRecord>& images,
                              const std::vector<int>& board_ids,
                              const ApriltagInternalConfig& base_config,
                              const ApriltagInternalDetectionOptions& options,
                              InternalProjectionMode mode,
                              const DsIntrinsics* intrinsics) {
  DetectionMap detections;
  for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
    for (std::size_t board_index = 0; board_index < board_ids.size(); ++board_index) {
      BoardKey key;
      key.image_index = static_cast<int>(image_index);
      key.board_id = board_ids[board_index];
      detections[key] = RunBoardDetection(images[image_index], base_config, options,
                                          board_ids[board_index], mode, intrinsics);
    }
  }
  return detections;
}

void SaveBoardVisualization(const boost::filesystem::path& output_dir,
                            const ImageRecord& image_record,
                            const ApriltagInternalConfig& base_config,
                            const ApriltagInternalDetectionOptions& options,
                            int board_id,
                            InternalProjectionMode mode,
                            const DsIntrinsics* intrinsics,
                            const BoardDetection& detection) {
  if (!detection.available) {
    return;
  }

  const ApriltagInternalConfig config = MakeBoardConfig(base_config, board_id, mode, intrinsics);
  ApriltagInternalDetector detector(config, options);

  cv::Mat overlay = image_record.image.clone();
  detector.DrawDetections(detection.result, &overlay);

  std::ostringstream stem;
  stem << image_record.stem << "_board" << board_id << "_" << ToString(mode);
  const boost::filesystem::path overlay_path = output_dir / (stem.str() + "_detected.png");
  if (!cv::imwrite(overlay_path.string(), overlay)) {
    throw std::runtime_error("Failed to write overlay image: " + overlay_path.string());
  }

  cv::Mat patch = detection.result.canonical_patch.clone();
  if (!patch.empty()) {
    detector.DrawCanonicalView(detection.result, &patch);
    const boost::filesystem::path patch_path = output_dir / (stem.str() + "_patch.png");
    if (!cv::imwrite(patch_path.string(), patch)) {
      throw std::runtime_error("Failed to write patch image: " + patch_path.string());
    }
  }
}

CorrespondenceMap BuildCorrespondenceMap(const DetectionMap& detections,
                                         bool include_internal,
                                         double internal_quality_threshold,
                                         int* outer_count,
                                         int* internal_count) {
  if (outer_count != nullptr) {
    *outer_count = 0;
  }
  if (internal_count != nullptr) {
    *internal_count = 0;
  }

  CorrespondenceMap correspondence_map;
  for (DetectionMap::const_iterator it = detections.begin(); it != detections.end(); ++it) {
    if (!it->second.available || !it->second.result.tag_detected) {
      continue;
    }

    std::vector<CameraCorrespondence> correspondences = BuildOuterCorrespondences(it->second.result);
    if (outer_count != nullptr) {
      *outer_count += static_cast<int>(correspondences.size());
    }

    if (include_internal) {
      const std::vector<CameraCorrespondence> internal =
          BuildInternalCorrespondences(it->second.result, internal_quality_threshold);
      if (internal_count != nullptr) {
        *internal_count += static_cast<int>(internal.size());
      }
      correspondences.insert(correspondences.end(), internal.begin(), internal.end());
    }

    if (!correspondences.empty()) {
      correspondence_map[it->first] = correspondences;
    }
  }
  return correspondence_map;
}

bool EstimatePose(const DsIntrinsics& intrinsics,
                  const std::vector<CameraCorrespondence>& correspondences,
                  PoseEstimate* pose_estimate) {
  if (pose_estimate == nullptr) {
    throw std::runtime_error("EstimatePose requires a valid output pointer.");
  }
  if (correspondences.size() < 4) {
    return false;
  }

  const DoubleSphereCameraModel camera = DoubleSphereCameraModel::FromConfig(MakeCameraConfig(intrinsics));
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(correspondences.size());
  image_points.reserve(correspondences.size());
  for (std::size_t index = 0; index < correspondences.size(); ++index) {
    const CameraCorrespondence& correspondence = correspondences[index];
    object_points.push_back(cv::Point3f(static_cast<float>(correspondence.target_xyz.x()),
                                        static_cast<float>(correspondence.target_xyz.y()),
                                        static_cast<float>(correspondence.target_xyz.z())));
    image_points.push_back(correspondence.image_xy);
  }

  cv::Mat rvec;
  cv::Mat tvec;
  if (!camera.estimateTransformation(object_points, image_points, &rvec, &tvec)) {
    return false;
  }

  pose_estimate->valid = true;
  pose_estimate->rvec = rvec.clone();
  pose_estimate->tvec = tvec.clone();
  return true;
}

double ComputeObservationReprojectionError(const DsIntrinsics& intrinsics,
                                           const PoseEstimate& pose,
                                           const std::vector<CameraCorrespondence>& correspondences) {
  if (!pose.valid || correspondences.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  double sum_error = 0.0;
  int count = 0;
  for (std::size_t index = 0; index < correspondences.size(); ++index) {
    const CameraCorrespondence& correspondence = correspondences[index];
    const Eigen::Vector3d point_camera =
        TransformTargetPoint(pose.rvec, pose.tvec, correspondence.target_xyz);
    Eigen::Vector2d projected = Eigen::Vector2d::Zero();
    if (!ProjectDoubleSphere(intrinsics, point_camera, &projected)) {
      sum_error += 100.0;
      ++count;
      continue;
    }

    const double dx = projected.x() - static_cast<double>(correspondence.image_xy.x);
    const double dy = projected.y() - static_cast<double>(correspondence.image_xy.y);
    sum_error += std::sqrt(dx * dx + dy * dy);
    ++count;
  }

  return count > 0 ? sum_error / static_cast<double>(count) : std::numeric_limits<double>::infinity();
}

PoseMap EstimateAllPoses(const DsIntrinsics& intrinsics,
                         const CorrespondenceMap& correspondences,
                         int* valid_pose_count,
                         int* board_observation_count) {
  if (valid_pose_count != nullptr) {
    *valid_pose_count = 0;
  }
  if (board_observation_count != nullptr) {
    *board_observation_count = 0;
  }

  PoseMap poses;
  for (CorrespondenceMap::const_iterator it = correspondences.begin(); it != correspondences.end(); ++it) {
    if (board_observation_count != nullptr) {
      ++(*board_observation_count);
    }

    PoseEstimate pose;
    if (EstimatePose(intrinsics, it->second, &pose)) {
      pose.mean_reprojection_error = ComputeObservationReprojectionError(intrinsics, pose, it->second);
      poses[it->first] = pose;
      if (valid_pose_count != nullptr) {
        ++(*valid_pose_count);
      }
    }
  }
  return poses;
}

Eigen::VectorXd BuildResidualVector(const DsIntrinsics& intrinsics,
                                    const CorrespondenceMap& correspondences,
                                    const PoseMap& poses,
                                    int* residual_point_count) {
  int point_count = 0;
  for (CorrespondenceMap::const_iterator it = correspondences.begin(); it != correspondences.end(); ++it) {
    PoseMap::const_iterator pose_it = poses.find(it->first);
    if (pose_it == poses.end() || !pose_it->second.valid) {
      continue;
    }
    point_count += static_cast<int>(it->second.size());
  }

  if (residual_point_count != nullptr) {
    *residual_point_count = point_count;
  }

  Eigen::VectorXd residuals = Eigen::VectorXd::Zero(2 * point_count);
  int row = 0;
  for (CorrespondenceMap::const_iterator it = correspondences.begin(); it != correspondences.end(); ++it) {
    PoseMap::const_iterator pose_it = poses.find(it->first);
    if (pose_it == poses.end() || !pose_it->second.valid) {
      continue;
    }

    for (std::size_t index = 0; index < it->second.size(); ++index) {
      const CameraCorrespondence& correspondence = it->second[index];
      const Eigen::Vector3d point_camera =
          TransformTargetPoint(pose_it->second.rvec, pose_it->second.tvec, correspondence.target_xyz);
      Eigen::Vector2d projected = Eigen::Vector2d::Zero();
      if (ProjectDoubleSphere(intrinsics, point_camera, &projected)) {
        residuals[row++] = projected.x() - static_cast<double>(correspondence.image_xy.x);
        residuals[row++] = projected.y() - static_cast<double>(correspondence.image_xy.y);
      } else {
        residuals[row++] = 100.0;
        residuals[row++] = 100.0;
      }
    }
  }
  return residuals;
}

double ComputeRmse(const Eigen::VectorXd& residuals, int residual_point_count) {
  if (residual_point_count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(residuals.squaredNorm() / static_cast<double>(residual_point_count));
}

double ParameterStep(double value, double fallback_step) {
  return std::max(std::abs(value) * 1e-4, fallback_step);
}

bool OptimizeIntrinsics(const CorrespondenceMap& correspondences,
                        const PoseMap& poses,
                        DsIntrinsics* intrinsics,
                        double* rmse) {
  if (intrinsics == nullptr || rmse == nullptr) {
    throw std::runtime_error("OptimizeIntrinsics requires valid output pointers.");
  }

  int residual_point_count = 0;
  Eigen::VectorXd residuals = BuildResidualVector(*intrinsics, correspondences, poses, &residual_point_count);
  if (residual_point_count <= 0) {
    return false;
  }

  double lambda = 1e-3;
  double best_cost = residuals.squaredNorm();
  Eigen::Matrix<double, 6, 1> parameters = ToVector(*intrinsics);
  const Eigen::Matrix<double, 6, 1> anchor_parameters = parameters;
  Eigen::Matrix<double, 6, 1> prior_sigma;
  prior_sigma << 0.20, 0.12,
                 0.20 * static_cast<double>(intrinsics->resolution.width),
                 0.20 * static_cast<double>(intrinsics->resolution.height),
                 0.03 * static_cast<double>(intrinsics->resolution.width),
                 0.03 * static_cast<double>(intrinsics->resolution.height);
  Eigen::Matrix<double, 6, 1> prior_weight;
  for (int index = 0; index < 6; ++index) {
    prior_weight[index] = 1.0 / std::max(1e-9, prior_sigma[index] * prior_sigma[index]);
  }

  for (int iteration = 0; iteration < 18; ++iteration) {
    Eigen::MatrixXd jacobian(residuals.rows(), 6);
    for (int column = 0; column < 6; ++column) {
      Eigen::Matrix<double, 6, 1> plus_params = parameters;
      Eigen::Matrix<double, 6, 1> minus_params = parameters;
      const double step = column == 0 ? ParameterStep(parameters[column], 1e-3)
                                      : (column == 1 ? ParameterStep(parameters[column], 1e-3)
                                                     : ParameterStep(parameters[column], 1e-1));
      plus_params[column] += step;
      minus_params[column] -= step;

      DsIntrinsics plus_intrinsics = FromVector(plus_params, intrinsics->resolution);
      DsIntrinsics minus_intrinsics = FromVector(minus_params, intrinsics->resolution);
      ClampIntrinsicsInPlace(&plus_intrinsics);
      ClampIntrinsicsInPlace(&minus_intrinsics);

      const Eigen::VectorXd residuals_plus =
          BuildResidualVector(plus_intrinsics, correspondences, poses, NULL);
      const Eigen::VectorXd residuals_minus =
          BuildResidualVector(minus_intrinsics, correspondences, poses, NULL);
      jacobian.col(column) = (residuals_plus - residuals_minus) / (2.0 * step);
    }

    const Eigen::Matrix<double, 6, 6> hessian = jacobian.transpose() * jacobian;
    const Eigen::Matrix<double, 6, 1> gradient = jacobian.transpose() * residuals;
    Eigen::Matrix<double, 6, 6> prior_hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> prior_gradient = Eigen::Matrix<double, 6, 1>::Zero();
    for (int index = 0; index < 6; ++index) {
      prior_hessian(index, index) = prior_weight[index];
      prior_gradient[index] = prior_weight[index] * (parameters[index] - anchor_parameters[index]);
    }
    const Eigen::Matrix<double, 6, 6> damped =
        hessian + prior_hessian + lambda * Eigen::Matrix<double, 6, 6>::Identity();
    const Eigen::Matrix<double, 6, 1> delta =
        damped.ldlt().solve(-(gradient + prior_gradient));
    if (!delta.allFinite()) {
      break;
    }

    DsIntrinsics candidate = FromVector(parameters + delta, intrinsics->resolution);
    ClampIntrinsicsInPlace(&candidate);
    const Eigen::VectorXd candidate_residuals =
        BuildResidualVector(candidate, correspondences, poses, NULL);
    const Eigen::Matrix<double, 6, 1> candidate_vector = ToVector(candidate);
    double prior_cost = 0.0;
    for (int index = 0; index < 6; ++index) {
      const double diff = candidate_vector[index] - anchor_parameters[index];
      prior_cost += prior_weight[index] * diff * diff;
    }
    const double candidate_cost = candidate_residuals.squaredNorm() + prior_cost;

    if (candidate_cost < best_cost) {
      parameters = ToVector(candidate);
      *intrinsics = candidate;
      residuals = candidate_residuals;
      best_cost = candidate_cost;
      lambda *= 0.5;
      if (delta.norm() < 1e-4) {
        break;
      }
    } else {
      lambda *= 4.0;
    }
  }

  *rmse = ComputeRmse(residuals, residual_point_count);
  return true;
}

std::string IntrinsicsToString(const DsIntrinsics& intrinsics) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(6)
         << "[xi=" << intrinsics.xi
         << ", alpha=" << intrinsics.alpha
         << ", fu=" << intrinsics.fu
         << ", fv=" << intrinsics.fv
         << ", cu=" << intrinsics.cu
         << ", cv=" << intrinsics.cv << "]";
  return stream.str();
}

std::string FormatMetrics(const MetricsAccumulator& metrics) {
  std::ostringstream stream;
  stream << "  observations: " << metrics.observation_count << "\n";
  stream << "  successful observations: " << metrics.successful_observations << "\n";
  stream << "  valid internal points: " << metrics.valid_points << "/" << metrics.total_points << "\n";
  stream << "  predicted->refined avg displacement: "
         << metrics.Average(metrics.sum_predicted_to_refined) << "\n";
  stream << "  avg template_quality: " << metrics.Average(metrics.sum_template_quality) << "\n";
  stream << "  avg gradient_quality: " << metrics.Average(metrics.sum_gradient_quality) << "\n";
  stream << "  avg final_quality: " << metrics.Average(metrics.sum_final_quality) << "\n";
  stream << "  avg image template_quality: "
         << metrics.Average(metrics.sum_image_template_quality) << "\n";
  stream << "  avg image gradient_quality: "
         << metrics.Average(metrics.sum_image_gradient_quality) << "\n";
  stream << "  avg image final_quality: " << metrics.Average(metrics.sum_image_final_quality) << "\n";
  stream << "  LCorner valid: " << metrics.lcorner_valid << "/" << metrics.lcorner_points << "\n";
  stream << "  XCorner valid: " << metrics.xcorner_valid << "/" << metrics.xcorner_points << "\n";
  return stream.str();
}

cv::Mat RenderTitledTile(const cv::Mat& image, const std::string& title) {
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

  const int title_height = 30;
  cv::Mat tile(color.rows + title_height, color.cols, CV_8UC3, cv::Scalar(24, 24, 24));
  color.copyTo(tile(cv::Rect(0, title_height, color.cols, color.rows)));
  cv::putText(tile, title, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
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

boost::filesystem::path IterationDirForIndex(const boost::filesystem::path& iterative_dir,
                                             int iteration_index) {
  std::ostringstream name;
  name << "iter_" << iteration_index;
  return iterative_dir / name.str();
}

void BuildCombinedVisualComparisons(const boost::filesystem::path& output_dir,
                                    const std::vector<ImageRecord>& images,
                                    const std::vector<int>& board_ids,
                                    const std::vector<IterationSummary>& summaries) {
  const boost::filesystem::path baseline_dir = output_dir / "baseline_homography";
  const boost::filesystem::path iterative_dir = output_dir / "iterative_coarse_model";
  const boost::filesystem::path combined_detected_dir = output_dir / "combined" / "detected";
  const boost::filesystem::path combined_patch_dir = output_dir / "combined" / "patch";
  boost::filesystem::create_directories(combined_detected_dir);
  boost::filesystem::create_directories(combined_patch_dir);

  for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
    for (std::size_t board_index = 0; board_index < board_ids.size(); ++board_index) {
      const int board_id = board_ids[board_index];
      std::vector<cv::Mat> detected_tiles;
      std::vector<cv::Mat> patch_tiles;

      {
        std::ostringstream base_name;
        base_name << images[image_index].stem << "_board" << board_id << "_homography";
        const boost::filesystem::path detected_path =
            baseline_dir / (base_name.str() + "_detected.png");
        const boost::filesystem::path patch_path =
            baseline_dir / (base_name.str() + "_patch.png");
        if (boost::filesystem::exists(detected_path)) {
          detected_tiles.push_back(
              RenderTitledTile(cv::imread(detected_path.string(), cv::IMREAD_COLOR),
                               "baseline_homography"));
        }
        if (boost::filesystem::exists(patch_path)) {
          patch_tiles.push_back(
              RenderTitledTile(cv::imread(patch_path.string(), cv::IMREAD_COLOR),
                               "baseline_homography"));
        }
      }

      for (std::size_t summary_index = 0; summary_index < summaries.size(); ++summary_index) {
        if (summaries[summary_index].iteration_index <= 0) {
          continue;
        }

        std::ostringstream iter_name;
        iter_name << images[image_index].stem << "_board" << board_id << "_virtual_pinhole_patch";
        const boost::filesystem::path iteration_dir =
            IterationDirForIndex(iterative_dir, summaries[summary_index].iteration_index);
        const boost::filesystem::path detected_path =
            iteration_dir / (iter_name.str() + "_detected.png");
        const boost::filesystem::path patch_path =
            iteration_dir / (iter_name.str() + "_patch.png");
        const std::string label = summaries[summary_index].label;

        if (boost::filesystem::exists(detected_path)) {
          detected_tiles.push_back(
              RenderTitledTile(cv::imread(detected_path.string(), cv::IMREAD_COLOR), label));
        }
        if (boost::filesystem::exists(patch_path)) {
          patch_tiles.push_back(
              RenderTitledTile(cv::imread(patch_path.string(), cv::IMREAD_COLOR), label));
        }
      }

      if (!detected_tiles.empty()) {
        int max_height = 0;
        for (std::size_t i = 0; i < detected_tiles.size(); ++i) {
          max_height = std::max(max_height, detected_tiles[i].rows);
        }
        std::vector<cv::Mat> padded_tiles;
        for (std::size_t i = 0; i < detected_tiles.size(); ++i) {
          padded_tiles.push_back(PadTileToHeight(detected_tiles[i], max_height));
        }
        cv::Mat combined;
        cv::hconcat(padded_tiles, combined);
        std::ostringstream output_name;
        output_name << images[image_index].stem << "_board" << board_id << "_combined_detected.png";
        cv::imwrite((combined_detected_dir / output_name.str()).string(), combined);
      }

      if (!patch_tiles.empty()) {
        int max_height = 0;
        for (std::size_t i = 0; i < patch_tiles.size(); ++i) {
          max_height = std::max(max_height, patch_tiles[i].rows);
        }
        std::vector<cv::Mat> padded_tiles;
        for (std::size_t i = 0; i < patch_tiles.size(); ++i) {
          padded_tiles.push_back(PadTileToHeight(patch_tiles[i], max_height));
        }
        cv::Mat combined;
        cv::hconcat(padded_tiles, combined);
        std::ostringstream output_name;
        output_name << images[image_index].stem << "_board" << board_id << "_combined_patch.png";
        cv::imwrite((combined_patch_dir / output_name.str()).string(), combined);
      }
    }
  }
}

void WriteIterationSummary(const boost::filesystem::path& output_dir,
                           const IterationSummary& summary) {
  std::ostringstream text;
  text << "label: " << summary.label << "\n";
  text << "iteration_index: " << summary.iteration_index << "\n";
  text << "camera: " << IntrinsicsToString(summary.camera) << "\n";
  text << "board observations used: " << summary.board_observation_count << "\n";
  text << "valid poses: " << summary.valid_pose_count << "\n";
  text << "camera correspondences: " << summary.correspondence_count
       << " (outer=" << summary.outer_correspondence_count
       << ", internal=" << summary.internal_correspondence_count << ")\n";
  text << "camera rmse: " << summary.camera_rmse << "\n";
  text << "score: " << summary.score << "\n";
  text << "\nGlobal metrics\n" << FormatMetrics(summary.global_metrics) << "\n";
  text << "Per-group metrics\n";
  for (std::map<std::string, MetricsAccumulator>::const_iterator it = summary.group_metrics.begin();
       it != summary.group_metrics.end(); ++it) {
    text << "- group: " << it->first << "\n" << FormatMetrics(it->second);
  }

  const boost::filesystem::path summary_path = output_dir / "summary.txt";
  std::ofstream stream(summary_path.string().c_str());
  stream << text.str();
  stream.close();
}

void AppendIterationCsvRow(std::ofstream* stream,
                           const IterationSummary& summary,
                           const std::string& scope,
                           const std::string& group_name,
                           const MetricsAccumulator& metrics) {
  if (stream == NULL) {
    throw std::runtime_error("AppendIterationCsvRow requires a valid output stream.");
  }

  (*stream)
      << summary.iteration_index << ","
      << summary.label << ","
      << scope << ","
      << group_name << ","
      << metrics.observation_count << ","
      << metrics.successful_observations << ","
      << metrics.valid_points << ","
      << metrics.total_points << ","
      << metrics.Average(metrics.sum_predicted_to_refined) << ","
      << metrics.Average(metrics.sum_template_quality) << ","
      << metrics.Average(metrics.sum_gradient_quality) << ","
      << metrics.Average(metrics.sum_final_quality) << ","
      << metrics.lcorner_valid << ","
      << metrics.lcorner_points << ","
      << metrics.xcorner_valid << ","
      << metrics.xcorner_points << ","
      << summary.camera.xi << ","
      << summary.camera.alpha << ","
      << summary.camera.fu << ","
      << summary.camera.fv << ","
      << summary.camera.cu << ","
      << summary.camera.cv << ","
      << summary.camera_rmse << ","
      << summary.score << "\n";
}

void WriteIterationCsv(const boost::filesystem::path& csv_path,
                       const std::vector<IterationSummary>& summaries) {
  std::ofstream stream(csv_path.string().c_str());
  stream << "iteration,label,scope,group,observations,successful_observations,"
         << "valid_internal_points,total_internal_points,avg_predicted_to_refined,"
         << "avg_template_quality,avg_gradient_quality,avg_final_quality,"
         << "lcorner_valid,lcorner_total,xcorner_valid,xcorner_total,"
         << "xi,alpha,fu,fv,cu,cv,camera_rmse,score\n";
  for (std::size_t index = 0; index < summaries.size(); ++index) {
    AppendIterationCsvRow(&stream, summaries[index], "global", "all",
                          summaries[index].global_metrics);
    for (std::map<std::string, MetricsAccumulator>::const_iterator it =
             summaries[index].group_metrics.begin();
         it != summaries[index].group_metrics.end(); ++it) {
      AppendIterationCsvRow(&stream, summaries[index], "group", it->first, it->second);
    }
  }
  stream.close();
}

IterationSummary SummarizeIteration(const std::string& label,
                                    int iteration_index,
                                    const DsIntrinsics& camera,
                                    int board_observation_count,
                                    int valid_pose_count,
                                    int correspondence_count,
                                    int outer_correspondence_count,
                                    int internal_correspondence_count,
                                    double camera_rmse,
                                    const std::vector<ImageRecord>& images,
                                    const DetectionMap& detections) {
  IterationSummary summary;
  summary.label = label;
  summary.iteration_index = iteration_index;
  summary.camera = camera;
  summary.board_observation_count = board_observation_count;
  summary.valid_pose_count = valid_pose_count;
  summary.correspondence_count = correspondence_count;
  summary.outer_correspondence_count = outer_correspondence_count;
  summary.internal_correspondence_count = internal_correspondence_count;
  summary.camera_rmse = camera_rmse;

  for (DetectionMap::const_iterator it = detections.begin(); it != detections.end(); ++it) {
    if (!it->second.available) {
      continue;
    }

    summary.global_metrics.AddObservation(it->second.result);
    const std::string& group_name = images[static_cast<std::size_t>(it->first.image_index)].group_name;
    summary.group_metrics[group_name].AddObservation(it->second.result);
  }

  summary.score = summary.global_metrics.Score();
  return summary;
}

void WriteDatasetSummary(const boost::filesystem::path& output_dir,
                         const std::vector<ImageRecord>& images,
                         const std::vector<int>& board_ids,
                         const IterativeCoarseCalibrationExperimentRequest& request,
                         const std::vector<IterationSummary>& summaries) {
  std::ostringstream stream;
  stream << "iterative coarse calibration experiment\n";
  stream << "image_dir: " << request.image_dir << "\n";
  stream << "output_dir: " << request.output_dir << "\n";
  stream << "group rule: filename stem before the last '-'\n";
  stream << "selected groups:";
  if (request.experiment_options.group_filters.empty()) {
    stream << " all";
  } else {
    for (std::size_t index = 0; index < request.experiment_options.group_filters.size(); ++index) {
      stream << " " << request.experiment_options.group_filters[index];
    }
  }
  stream << "\nboard ids:";
  for (std::size_t index = 0; index < board_ids.size(); ++index) {
    stream << " " << board_ids[index];
  }
  stream << "\nimages:";
  for (std::size_t index = 0; index < images.size(); ++index) {
    stream << "\n  - " << images[index].stem << " (group=" << images[index].group_name << ")";
  }
  stream << "\n\niteration summaries\n";
  for (std::size_t index = 0; index < summaries.size(); ++index) {
    stream << "\n[" << summaries[index].label << "]\n";
    stream << "camera=" << IntrinsicsToString(summaries[index].camera)
           << " camera_rmse=" << summaries[index].camera_rmse
           << " score=" << summaries[index].score
           << " valid_internal=" << summaries[index].global_metrics.valid_points << "\n";
  }

  const boost::filesystem::path summary_path = output_dir / "experiment_summary.txt";
  std::ofstream summary_stream(summary_path.string().c_str());
  summary_stream << stream.str();
  summary_stream.close();
}

}  // namespace

IterativeCoarseCalibrationExperiment::IterativeCoarseCalibrationExperiment(
    IterativeCoarseCalibrationExperimentRequest request)
    : request_(std::move(request)) {}

void IterativeCoarseCalibrationExperiment::Run() const {
  std::vector<ImageRecord> images =
      LoadDatasetImages(request_.image_dir, request_.experiment_options.group_filters);
  std::vector<int> board_ids = request_.experiment_options.board_ids;
  if (board_ids.empty()) {
    board_ids.push_back(request_.base_config.tag_id);
  }

  boost::filesystem::create_directories(request_.output_dir);
  const boost::filesystem::path baseline_dir =
      boost::filesystem::path(request_.output_dir) / "baseline_homography";
  const boost::filesystem::path iterative_dir =
      boost::filesystem::path(request_.output_dir) / "iterative_coarse_model";
  boost::filesystem::create_directories(baseline_dir);
  boost::filesystem::create_directories(iterative_dir);

  const DetectionMap baseline_detections =
      RunDetectionPass(images, board_ids, request_.base_config, request_.detection_options,
                       InternalProjectionMode::Homography, NULL);
  for (DetectionMap::const_iterator it = baseline_detections.begin(); it != baseline_detections.end(); ++it) {
    SaveBoardVisualization(baseline_dir, images[static_cast<std::size_t>(it->first.image_index)],
                           request_.base_config, request_.detection_options, it->first.board_id,
                           InternalProjectionMode::Homography, NULL, it->second);
  }

  std::vector<IterationSummary> summaries;
  summaries.push_back(
      SummarizeIteration("iteration_0_baseline_homography", 0,
                         MakeInitialIntrinsics(images.front().image.size(),
                                               request_.experiment_options),
                         0, 0, 0, 0, 0, 0.0, images, baseline_detections));
  WriteIterationSummary(baseline_dir, summaries.back());

  DsIntrinsics current_camera =
      MakeInitialIntrinsics(images.front().image.size(), request_.experiment_options);
  ClampIntrinsicsInPlace(&current_camera);
  DetectionMap previous_iterative_detections = baseline_detections;
  double previous_score = summaries.back().score;

  for (int iteration = 1; iteration <= request_.experiment_options.max_iterations; ++iteration) {
    const bool include_internal =
        iteration > 1 && request_.experiment_options.use_internal_points_for_update;
    int outer_correspondence_count = 0;
    int internal_correspondence_count = 0;
    const CorrespondenceMap correspondences =
        BuildCorrespondenceMap(previous_iterative_detections, include_internal,
                               request_.experiment_options.internal_point_quality_threshold,
                               &outer_correspondence_count, &internal_correspondence_count);
    if (correspondences.empty()) {
      throw std::runtime_error("No board correspondences were available for coarse camera update.");
    }

    int valid_pose_count = 0;
    int board_observation_count = 0;
    double camera_rmse = std::numeric_limits<double>::infinity();
    for (int round = 0; round < request_.experiment_options.pose_refinement_rounds; ++round) {
      const PoseMap poses =
          EstimateAllPoses(current_camera, correspondences, &valid_pose_count, &board_observation_count);
      if (poses.empty()) {
        throw std::runtime_error("Failed to estimate any board pose with the current coarse model.");
      }
      OptimizeIntrinsics(correspondences, poses, &current_camera, &camera_rmse);
    }

    const DetectionMap iterative_detections =
        RunDetectionPass(images, board_ids, request_.base_config, request_.detection_options,
                         InternalProjectionMode::VirtualPinholePatch, &current_camera);

    std::ostringstream label_stream;
    label_stream << "iteration_" << iteration << "_iterative_coarse_model";
    const IterationSummary summary =
        SummarizeIteration(label_stream.str(), iteration, current_camera,
                           board_observation_count, valid_pose_count,
                           outer_correspondence_count + internal_correspondence_count,
                           outer_correspondence_count, internal_correspondence_count,
                           camera_rmse, images, iterative_detections);
    summaries.push_back(summary);

    const boost::filesystem::path iteration_dir =
        iterative_dir / (std::string("iter_") + std::to_string(iteration));
    boost::filesystem::create_directories(iteration_dir);
    for (DetectionMap::const_iterator it = iterative_detections.begin(); it != iterative_detections.end(); ++it) {
      SaveBoardVisualization(iteration_dir,
                             images[static_cast<std::size_t>(it->first.image_index)],
                             request_.base_config, request_.detection_options, it->first.board_id,
                             InternalProjectionMode::VirtualPinholePatch, &current_camera, it->second);
    }
    WriteIterationSummary(iteration_dir, summaries.back());

    previous_iterative_detections = iterative_detections;
    const double improvement = summaries.back().score - previous_score;
    previous_score = summaries.back().score;
    if (std::abs(improvement) < request_.experiment_options.convergence_threshold) {
      break;
    }
  }

  WriteIterationCsv(boost::filesystem::path(request_.output_dir) / "iteration_summary.csv",
                    summaries);
  WriteDatasetSummary(request_.output_dir, images, board_ids, request_, summaries);
  BuildCombinedVisualComparisons(request_.output_dir, images, board_ids, summaries);
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
