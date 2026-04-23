#include <aslam/cameras/apriltag_internal/ApriltagInternalDebugVisualization.hpp>
#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
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

namespace {

namespace ati = aslam::cameras::apriltag_internal;

constexpr double kConsensusGtDistanceThresholdPx = 2.0;
constexpr int kLowCoverageThreshold = 10;

const std::array<const char*, 9> kSelectedImageStems = {{
    "image1-2", "image1-3", "image1-4",
    "image2-2", "image2-3", "image2-4",
    "image3-2", "image3-3", "image3-4",
}};

struct CmdArgs {
  std::string image_dir;
  std::string config_path;
  std::string output_dir;
  std::string old_mode = "sphere_lattice";
  std::string new_mode = "sphere_ray_refine";
  std::vector<int> board_ids;
  int max_iterations = 3;
  int pose_refinement_rounds = 2;
  bool no_subpix = false;
  bool use_internal_points_for_update = true;
  double internal_point_quality_threshold = 0.45;
  double convergence_threshold = 0.05;
  double init_xi = -0.2;
  double init_alpha = 0.6;
  double init_fu_scale = 0.55;
  double init_fv_scale = 0.55;
  double init_cu_offset = 0.0;
  double init_cv_offset = 0.0;
};

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
  ati::ApriltagInternalDetectionResult result;
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
  ati::CornerType corner_type = ati::CornerType::Outer;
  double quality = 0.0;
  bool from_internal = false;
};

struct MetricsAccumulator {
  int observation_count = 0;
  int successful_observations = 0;
  int total_points = 0;
  int valid_points = 0;
  double sum_final_quality = 0.0;

  void AddObservation(const ati::ApriltagInternalDetectionResult& result) {
    ++observation_count;
    successful_observations += result.success ? 1 : 0;
    for (const auto& debug : result.internal_corner_debug) {
      ++total_points;
      valid_points += debug.valid ? 1 : 0;
      sum_final_quality += debug.final_quality;
    }
  }

  double AverageFinalQuality() const {
    return total_points > 0 ? sum_final_quality / static_cast<double>(total_points) : 0.0;
  }

  double Score() const {
    return static_cast<double>(valid_points) + AverageFinalQuality();
  }
};

struct MethodIterationData {
  int iteration_index = 0;
  std::string label;
  DsIntrinsics camera;
  BoardDetection sample_detection;
  std::map<BoardKey, BoardDetection> detections;
  MetricsAccumulator metrics;
  int board_observation_count = 0;
  int valid_pose_count = 0;
  int correspondence_count = 0;
  int outer_correspondence_count = 0;
  int internal_correspondence_count = 0;
  double camera_rmse = 0.0;
  double score = 0.0;
};

struct MethodRunData {
  std::string method_name;
  ati::InternalProjectionMode mode = ati::InternalProjectionMode::SphereLattice;
  std::map<BoardKey, BoardDetection> calibrated_detections;
  std::vector<MethodIterationData> iterations;
};

struct ConsensusGtStats {
  int overlapping_valid_pairs = 0;
  int filtered_by_distance = 0;
  int gt_points = 0;
  std::map<std::string, int> overlapping_valid_pairs_per_image;
  std::map<std::string, int> filtered_by_distance_per_image;
  std::map<std::string, int> gt_points_per_image;
};

struct PerPointSeedMetric {
  int iteration = 0;
  std::string image_stem;
  std::string group_name;
  int board_id = -1;
  int point_id = -1;
  bool old_valid = false;
  bool new_valid = false;
  double d_p_old = 0.0;
  double d_p_new = 0.0;
  double d_ss_old = 0.0;
  double d_ss_new = 0.0;
  double d_r_old = 0.0;
  double d_r_new = 0.0;
  double imp_old = 0.0;
  double imp_new = 0.0;
  double delta_imp = 0.0;
  double move_old = 0.0;
  double move_new = 0.0;
  double predicted_gap = 0.0;
};

struct PerImageSeedSummary {
  int iteration = 0;
  std::string image_stem;
  std::string group_name;
  int gt_points = 0;
  int filtered_points = 0;
  bool low_coverage = false;
  double avg_d_p_old = 0.0;
  double avg_d_p_new = 0.0;
  double avg_d_ss_old = 0.0;
  double avg_d_ss_new = 0.0;
  double avg_d_r_old = 0.0;
  double avg_d_r_new = 0.0;
  double avg_imp_old = 0.0;
  double avg_imp_new = 0.0;
  double avg_delta_imp = 0.0;
  double avg_move_old = 0.0;
  double avg_move_new = 0.0;
  double avg_predicted_gap = 0.0;
};

struct IterationMethodSummary {
  std::string method_name;
  int iteration = 0;
  std::string label;
  int matched_gt_points = 0;
  double avg_d_p = 0.0;
  double avg_d_ss = 0.0;
  double avg_d_r = 0.0;
  double avg_improvement = 0.0;
  double avg_move = 0.0;
  double avg_predicted_gap = 0.0;
  int valid_internal_points = 0;
  int total_internal_points = 0;
  double camera_rmse = 0.0;
  double score = 0.0;
};

typedef std::map<BoardKey, BoardDetection> DetectionMap;
typedef std::map<BoardKey, std::vector<CameraCorrespondence> > CorrespondenceMap;
typedef std::map<BoardKey, PoseEstimate> PoseMap;
typedef std::map<int, cv::Point2f> GroundTruthPointMap;
typedef std::map<BoardKey, GroundTruthPointMap> GroundTruthMap;

std::string ToLower(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return lowered;
}

void PrintUsage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --image-dir DIR --config YAML [--output-dir DIR] [--board-ids 1,2]"
      << " [--old-mode MODE] [--new-mode MODE]"
      << " [--max-iterations N] [--pose-refinement-rounds N]"
      << " [--internal-point-quality-threshold Q] [--no-subpix]\n\n"
      << "Supported modes: virtual_pinhole_patch, virtual_pinhole_image_subpix, "
      << "virtual_pinhole_patch_boundary_seed, sphere_lattice, sphere_ray_refine, homography\n"
      << "Default comparison: sphere_lattice vs sphere_ray_refine on the fixed 9-image subset:\n"
      << "  image1-2/3/4, image2-2/3/4, image3-2/3/4\n";
}

std::vector<std::string> SplitCsv(const std::string& value) {
  std::vector<std::string> tokens;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      tokens.push_back(token);
    }
  }
  return tokens;
}

std::vector<int> ParseIntCsv(const std::string& value) {
  std::vector<int> numbers;
  for (const auto& token : SplitCsv(value)) {
    numbers.push_back(std::stoi(token));
  }
  return numbers;
}

std::string BuildTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
  localtime_r(&now_time, &local_time);
  std::ostringstream stream;
  stream << std::put_time(&local_time, "%Y%m%d_%H%M%S");
  return stream.str();
}

std::string GroupNameFromStem(const std::string& stem) {
  const std::size_t dash = stem.find_last_of('-');
  if (dash == std::string::npos || dash == 0 || dash + 1 >= stem.size()) {
    return stem;
  }
  return stem.substr(0, dash);
}

bool IsImageFile(const boost::filesystem::path& path) {
  const std::string extension = ToLower(path.extension().string());
  return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
         extension == ".bmp" || extension == ".tif" || extension == ".tiff";
}

ati::InternalProjectionMode ParseProjectionModeOrThrow(const std::string& value) {
  const std::string lowered = ToLower(value);
  if (lowered == "homography") {
    return ati::InternalProjectionMode::Homography;
  }
  if (lowered == "virtual_pinhole_patch" || lowered == "virtual-pinhole-patch") {
    return ati::InternalProjectionMode::VirtualPinholePatch;
  }
  if (lowered == "virtual_pinhole_image_subpix" ||
      lowered == "virtual-pinhole-image-subpix") {
    return ati::InternalProjectionMode::VirtualPinholeImageSubpix;
  }
  if (lowered == "virtual_pinhole_patch_boundary_seed" ||
      lowered == "virtual-pinhole-patch-boundary-seed" ||
      lowered == "virtual_pinhole_patch_edge_seed" ||
      lowered == "virtual-pinhole-patch-edge-seed") {
    return ati::InternalProjectionMode::VirtualPinholePatchBoundarySeed;
  }
  if (lowered == "sphere_lattice" || lowered == "sphere-lattice") {
    return ati::InternalProjectionMode::SphereLattice;
  }
  if (lowered == "sphere_border_lattice" || lowered == "sphere-border-lattice") {
    return ati::InternalProjectionMode::SphereBorderLattice;
  }
  if (lowered == "sphere_ray_refine" || lowered == "sphere-ray-refine") {
    return ati::InternalProjectionMode::SphereRayRefine;
  }
  throw std::runtime_error("Unsupported internal projection mode: " + value);
}

bool HasExplicitSphereSeedStage(ati::InternalProjectionMode mode) {
  return mode == ati::InternalProjectionMode::VirtualPinholePatchBoundarySeed ||
         mode == ati::InternalProjectionMode::SphereLattice ||
         mode == ati::InternalProjectionMode::SphereBorderLattice ||
         mode == ati::InternalProjectionMode::SphereRayRefine;
}

std::string DescribeMethodForReport(ati::InternalProjectionMode mode) {
  switch (mode) {
    case ati::InternalProjectionMode::VirtualPinholePatch:
      return "将外四角点张成的区域投到 virtual-pinhole patch，在 patch 上定位内点，再回到图像域。";
    case ati::InternalProjectionMode::VirtualPinholeImageSubpix:
      return "将 patch 上的几何预测点回投到原图，再直接在原图上做 cornerSubPix。";
    case ati::InternalProjectionMode::VirtualPinholePatchBoundarySeed:
      return "先在 visual-pinhole patch 上围绕预测点做黑白边界跃迁一致性 seed 搜索，再把 seed 回投到原图并做 cornerSubPix。";
    case ati::InternalProjectionMode::SphereLattice:
      return "先由位姿与相机模型给出预测射线，再在局部球面晶格候选中搜索 seed，最后在原图做 cornerSubPix。";
    case ati::InternalProjectionMode::SphereBorderLattice:
      return "先用外边界构造 border-conditioned ray seed，再围绕该 seed 做局部球面搜索，最后在原图做 cornerSubPix。";
    case ati::InternalProjectionMode::SphereRayRefine:
      return "先由位姿与相机模型给出预测射线，再在球面上做 ray-domain 连续 seed refinement，最后在原图做 cornerSubPix。";
    case ati::InternalProjectionMode::Homography:
      return "基于 homography 的图像域初始化与精修。";
  }
  return "unknown";
}

ati::ApriltagInternalDetectionOptions MakeDetectionOptionsFromConfig(
    const ati::ApriltagInternalConfig& config) {
  ati::ApriltagInternalDetectionOptions options;
  options.canonical_pixels_per_module = config.canonical_pixels_per_module;
  options.refinement_window_radius = config.refinement_window_radius;
  options.internal_subpix_window_scale = config.internal_subpix_window_scale;
  options.internal_subpix_window_min = config.internal_subpix_window_min;
  options.internal_subpix_window_max = config.internal_subpix_window_max;
  options.max_subpix_displacement2 = config.max_subpix_displacement2;
  options.internal_subpix_displacement_scale = config.internal_subpix_displacement_scale;
  options.max_internal_subpix_displacement = config.max_internal_subpix_displacement;
  return options;
}

CmdArgs ParseArgs(int argc, char** argv) {
  CmdArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string token = argv[i];
    if (token == "--image-dir" && i + 1 < argc) {
      args.image_dir = argv[++i];
    } else if (token == "--config" && i + 1 < argc) {
      args.config_path = argv[++i];
    } else if (token == "--output-dir" && i + 1 < argc) {
      args.output_dir = argv[++i];
    } else if (token == "--old-mode" && i + 1 < argc) {
      args.old_mode = ToLower(argv[++i]);
    } else if (token == "--new-mode" && i + 1 < argc) {
      args.new_mode = ToLower(argv[++i]);
    } else if (token == "--board-ids" && i + 1 < argc) {
      args.board_ids = ParseIntCsv(argv[++i]);
    } else if (token == "--max-iterations" && i + 1 < argc) {
      args.max_iterations = std::atoi(argv[++i]);
    } else if (token == "--pose-refinement-rounds" && i + 1 < argc) {
      args.pose_refinement_rounds = std::atoi(argv[++i]);
    } else if (token == "--internal-point-quality-threshold" && i + 1 < argc) {
      args.internal_point_quality_threshold = std::atof(argv[++i]);
    } else if (token == "--convergence-threshold" && i + 1 < argc) {
      args.convergence_threshold = std::atof(argv[++i]);
    } else if (token == "--init-xi" && i + 1 < argc) {
      args.init_xi = std::atof(argv[++i]);
    } else if (token == "--init-alpha" && i + 1 < argc) {
      args.init_alpha = std::atof(argv[++i]);
    } else if (token == "--init-fu-scale" && i + 1 < argc) {
      args.init_fu_scale = std::atof(argv[++i]);
    } else if (token == "--init-fv-scale" && i + 1 < argc) {
      args.init_fv_scale = std::atof(argv[++i]);
    } else if (token == "--init-cu-offset" && i + 1 < argc) {
      args.init_cu_offset = std::atof(argv[++i]);
    } else if (token == "--init-cv-offset" && i + 1 < argc) {
      args.init_cv_offset = std::atof(argv[++i]);
    } else if (token == "--no-subpix") {
      args.no_subpix = true;
    } else if (token == "--help" || token == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + token);
    }
  }

  if (args.image_dir.empty() || args.config_path.empty()) {
    throw std::runtime_error("Both --image-dir and --config are required.");
  }
  if (args.old_mode == args.new_mode) {
    throw std::runtime_error("--old-mode and --new-mode must be different.");
  }
  if (args.output_dir.empty()) {
    args.output_dir =
        (boost::filesystem::path("result") /
         boost::filesystem::path("internal_method_compare_" + args.old_mode + "_vs_" +
                                 args.new_mode + "_" + BuildTimestamp())).string();
  }
  return args;
}

std::vector<ImageRecord> LoadSelectedImages(const std::string& image_dir) {
  if (!boost::filesystem::exists(image_dir)) {
    throw std::runtime_error("Image directory does not exist: " + image_dir);
  }

  std::set<std::string> selected_stems;
  selected_stems.insert(kSelectedImageStems.begin(), kSelectedImageStems.end());

  std::vector<boost::filesystem::path> image_paths;
  for (boost::filesystem::directory_iterator it(image_dir), end; it != end; ++it) {
    if (boost::filesystem::is_regular_file(it->path()) && IsImageFile(it->path()) &&
        selected_stems.count(it->path().stem().string()) > 0) {
      image_paths.push_back(it->path());
    }
  }
  std::sort(image_paths.begin(), image_paths.end());

  std::vector<ImageRecord> images;
  for (const auto& path : image_paths) {
    ImageRecord record;
    record.index = static_cast<int>(images.size());
    record.path = path;
    record.stem = path.stem().string();
    record.group_name = GroupNameFromStem(record.stem);
    record.image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (record.image.empty()) {
      throw std::runtime_error("Failed to read image: " + path.string());
    }
    images.push_back(record);
  }

  if (images.size() != kSelectedImageStems.size()) {
    std::ostringstream stream;
    stream << "Expected " << kSelectedImageStems.size() << " selected images, but found "
           << images.size() << " under " << image_dir;
    throw std::runtime_error(stream.str());
  }

  const cv::Size reference_size = images.front().image.size();
  for (const auto& image : images) {
    if (image.image.size() != reference_size) {
      throw std::runtime_error("All selected images must share the same resolution.");
    }
  }

  return images;
}

ati::IntermediateCameraConfig MakeCameraConfig(const DsIntrinsics& intrinsics) {
  ati::IntermediateCameraConfig config;
  config.camera_model = "ds";
  config.distortion_model = "none";
  config.intrinsics = {intrinsics.xi, intrinsics.alpha, intrinsics.fu,
                       intrinsics.fv, intrinsics.cu, intrinsics.cv};
  config.resolution = {intrinsics.resolution.width, intrinsics.resolution.height};
  return config;
}

DsIntrinsics MakeInitialIntrinsics(const cv::Size& resolution, const CmdArgs& args) {
  DsIntrinsics intrinsics;
  intrinsics.resolution = resolution;
  intrinsics.xi = args.init_xi;
  intrinsics.alpha = args.init_alpha;
  intrinsics.fu = args.init_fu_scale * static_cast<double>(resolution.width);
  intrinsics.fv = args.init_fv_scale * static_cast<double>(resolution.height);
  intrinsics.cu = 0.5 * static_cast<double>(resolution.width) + args.init_cu_offset;
  intrinsics.cv = 0.5 * static_cast<double>(resolution.height) + args.init_cv_offset;
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

Eigen::Matrix<double, 6, 1> ToVector(const DsIntrinsics& intrinsics) {
  Eigen::Matrix<double, 6, 1> vector;
  vector << intrinsics.xi, intrinsics.alpha, intrinsics.fu, intrinsics.fv, intrinsics.cu,
      intrinsics.cv;
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

double ParameterStep(double value, double fallback_step) {
  return std::max(std::abs(value) * 1e-4, fallback_step);
}

ati::ApriltagInternalConfig MakeBoardConfig(const ati::ApriltagInternalConfig& base_config,
                                            int board_id,
                                            ati::InternalProjectionMode mode,
                                            const DsIntrinsics* intrinsics,
                                            bool use_calibrated_camera) {
  ati::ApriltagInternalConfig config = base_config;
  config.tag_id = board_id;
  config.tag_ids.clear();
  config.outer_detector_config.tag_id = board_id;
  config.internal_projection_mode = mode;
  if (intrinsics != nullptr) {
    config.sphere_lattice_use_initial_camera = false;
    config.outer_spherical_use_initial_camera = false;
    config.intermediate_camera = MakeCameraConfig(*intrinsics);
  } else if (use_calibrated_camera) {
    config.sphere_lattice_use_initial_camera = false;
    config.outer_spherical_use_initial_camera = false;
  }
  return config;
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

  const Eigen::Vector3d translation(tvec.at<double>(0, 0), tvec.at<double>(1, 0),
                                    tvec.at<double>(2, 0));
  return rotation * target_xyz + translation;
}

std::vector<CameraCorrespondence> BuildOuterCorrespondences(
    const ati::ApriltagInternalDetectionResult& result) {
  std::vector<CameraCorrespondence> correspondences;
  correspondences.reserve(result.corners.size());
  for (const auto& measurement : result.corners) {
    if (measurement.corner_type != ati::CornerType::Outer) {
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
    const ati::ApriltagInternalDetectionResult& result,
    double quality_threshold) {
  std::vector<CameraCorrespondence> correspondences;
  correspondences.reserve(result.corners.size());
  for (const auto& measurement : result.corners) {
    if (measurement.corner_type == ati::CornerType::Outer || !measurement.valid ||
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
                                 const ati::ApriltagInternalConfig& base_config,
                                 const ati::ApriltagInternalDetectionOptions& options,
                                 int board_id,
                                 ati::InternalProjectionMode mode,
                                 const DsIntrinsics* intrinsics,
                                 bool use_calibrated_camera) {
  BoardDetection detection;
  try {
    const ati::ApriltagInternalConfig config =
        MakeBoardConfig(base_config, board_id, mode, intrinsics, use_calibrated_camera);
    ati::ApriltagInternalDetector detector(config, options);
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
                              const ati::ApriltagInternalConfig& base_config,
                              const ati::ApriltagInternalDetectionOptions& options,
                              ati::InternalProjectionMode mode,
                              const DsIntrinsics* intrinsics,
                              bool use_calibrated_camera) {
  DetectionMap detections;
  for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
    ati::ApriltagInternalConfig pass_config = base_config;
    pass_config.internal_projection_mode = mode;
    pass_config.tag_ids = board_ids;
    if (!board_ids.empty()) {
      pass_config.tag_id = board_ids.front();
      pass_config.outer_detector_config.tag_id = board_ids.front();
    }
    if (intrinsics != nullptr) {
      pass_config.sphere_lattice_use_initial_camera = false;
      pass_config.outer_spherical_use_initial_camera = false;
      pass_config.intermediate_camera = MakeCameraConfig(*intrinsics);
    } else if (use_calibrated_camera) {
      pass_config.sphere_lattice_use_initial_camera = false;
      pass_config.outer_spherical_use_initial_camera = false;
    }

    try {
      ati::ApriltagInternalDetector detector(pass_config, options);
      const ati::ApriltagInternalMultiDetectionResult multi_result =
          detector.DetectMultiple(images[image_index].image);
      for (const int board_id : board_ids) {
        BoardKey key;
        key.image_index = static_cast<int>(image_index);
        key.board_id = board_id;

        BoardDetection board_detection;
        board_detection.available = true;
        const auto detection_it =
            std::find_if(multi_result.detections.begin(), multi_result.detections.end(),
                         [board_id](const ati::ApriltagInternalDetectionResult& detection) {
                           return detection.board_id == board_id;
                         });
        if (detection_it != multi_result.detections.end()) {
          board_detection.result = *detection_it;
        } else {
          board_detection.result.board_id = board_id;
        }
        detections[key] = board_detection;
      }
    } catch (const std::exception& error) {
      for (const int board_id : board_ids) {
        BoardKey key;
        key.image_index = static_cast<int>(image_index);
        key.board_id = board_id;

        BoardDetection board_detection;
        board_detection.available = false;
        board_detection.error_text = error.what();
        board_detection.result.board_id = board_id;
        detections[key] = board_detection;
      }
    }
  }
  return detections;
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
  for (const auto& detection_entry : detections) {
    if (!detection_entry.second.available || !detection_entry.second.result.tag_detected) {
      continue;
    }

    std::vector<CameraCorrespondence> correspondences =
        BuildOuterCorrespondences(detection_entry.second.result);
    if (outer_count != nullptr) {
      *outer_count += static_cast<int>(correspondences.size());
    }
    if (include_internal) {
      const std::vector<CameraCorrespondence> internal =
          BuildInternalCorrespondences(detection_entry.second.result, internal_quality_threshold);
      if (internal_count != nullptr) {
        *internal_count += static_cast<int>(internal.size());
      }
      correspondences.insert(correspondences.end(), internal.begin(), internal.end());
    }
    if (!correspondences.empty()) {
      correspondence_map[detection_entry.first] = correspondences;
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

  const ati::DoubleSphereCameraModel camera =
      ati::DoubleSphereCameraModel::FromConfig(MakeCameraConfig(intrinsics));
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(correspondences.size());
  image_points.reserve(correspondences.size());
  for (const auto& correspondence : correspondences) {
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
  for (const auto& correspondence : correspondences) {
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
  return count > 0 ? sum_error / static_cast<double>(count)
                   : std::numeric_limits<double>::infinity();
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
  for (const auto& entry : correspondences) {
    if (board_observation_count != nullptr) {
      ++(*board_observation_count);
    }
    PoseEstimate pose;
    if (EstimatePose(intrinsics, entry.second, &pose)) {
      pose.mean_reprojection_error =
          ComputeObservationReprojectionError(intrinsics, pose, entry.second);
      poses[entry.first] = pose;
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
  for (const auto& entry : correspondences) {
    const auto pose_it = poses.find(entry.first);
    if (pose_it == poses.end() || !pose_it->second.valid) {
      continue;
    }
    point_count += static_cast<int>(entry.second.size());
  }

  if (residual_point_count != nullptr) {
    *residual_point_count = point_count;
  }

  Eigen::VectorXd residuals = Eigen::VectorXd::Zero(2 * point_count);
  int row = 0;
  for (const auto& entry : correspondences) {
    const auto pose_it = poses.find(entry.first);
    if (pose_it == poses.end() || !pose_it->second.valid) {
      continue;
    }

    for (const auto& correspondence : entry.second) {
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

bool OptimizeIntrinsics(const CorrespondenceMap& correspondences,
                        const PoseMap& poses,
                        DsIntrinsics* intrinsics,
                        double* rmse) {
  if (intrinsics == nullptr || rmse == nullptr) {
    throw std::runtime_error("OptimizeIntrinsics requires valid output pointers.");
  }

  int residual_point_count = 0;
  Eigen::VectorXd residuals =
      BuildResidualVector(*intrinsics, correspondences, poses, &residual_point_count);
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

MethodIterationData SummarizeIteration(const DetectionMap& detections,
                                       int iteration_index,
                                       const std::string& label,
                                       const DsIntrinsics& camera,
                                       int board_observation_count,
                                       int valid_pose_count,
                                       int correspondence_count,
                                       int outer_correspondence_count,
                                       int internal_correspondence_count,
                                       double camera_rmse) {
  MethodIterationData data;
  data.iteration_index = iteration_index;
  data.label = label;
  data.camera = camera;
  data.board_observation_count = board_observation_count;
  data.valid_pose_count = valid_pose_count;
  data.correspondence_count = correspondence_count;
  data.outer_correspondence_count = outer_correspondence_count;
  data.internal_correspondence_count = internal_correspondence_count;
  data.camera_rmse = camera_rmse;
  data.detections = detections;
  for (const auto& entry : detections) {
    if (!entry.second.available) {
      continue;
    }
    data.metrics.AddObservation(entry.second.result);
  }
  data.score = data.metrics.Score();
  return data;
}

MethodRunData RunMethod(const std::string& method_name,
                        ati::InternalProjectionMode mode,
                        const std::vector<ImageRecord>& images,
                        const std::vector<int>& board_ids,
                        const ati::ApriltagInternalConfig& base_config,
                        const ati::ApriltagInternalDetectionOptions& options,
                        const CmdArgs& args) {
  MethodRunData run;
  run.method_name = method_name;
  run.mode = mode;

  run.calibrated_detections =
      RunDetectionPass(images, board_ids, base_config, options, mode, NULL, true);

  DsIntrinsics current_camera = MakeInitialIntrinsics(images.front().image.size(), args);
  ClampIntrinsicsInPlace(&current_camera);

  DetectionMap previous_detections =
      RunDetectionPass(images, board_ids, base_config, options, mode, &current_camera, false);
  run.iterations.push_back(
      SummarizeIteration(previous_detections, 0, "iteration_0_initial_camera", current_camera,
                         0, 0, 0, 0, 0, 0.0));

  double previous_score = run.iterations.back().score;
  for (int iteration = 1; iteration <= args.max_iterations; ++iteration) {
    const bool include_internal =
        iteration > 1 && args.use_internal_points_for_update;
    int outer_correspondence_count = 0;
    int internal_correspondence_count = 0;
    const CorrespondenceMap correspondences =
        BuildCorrespondenceMap(previous_detections, include_internal,
                               args.internal_point_quality_threshold,
                               &outer_correspondence_count, &internal_correspondence_count);
    if (correspondences.empty()) {
      throw std::runtime_error("No board correspondences were available for coarse camera update.");
    }

    int valid_pose_count = 0;
    int board_observation_count = 0;
    double camera_rmse = std::numeric_limits<double>::infinity();
    for (int round = 0; round < args.pose_refinement_rounds; ++round) {
      const PoseMap poses =
          EstimateAllPoses(current_camera, correspondences, &valid_pose_count, &board_observation_count);
      if (poses.empty()) {
        throw std::runtime_error("Failed to estimate any board pose with the current coarse model.");
      }
      OptimizeIntrinsics(correspondences, poses, &current_camera, &camera_rmse);
    }

    DetectionMap detections =
        RunDetectionPass(images, board_ids, base_config, options, mode, &current_camera, false);
    std::ostringstream label;
    label << "iteration_" << iteration;
    run.iterations.push_back(
        SummarizeIteration(detections, iteration, label.str(), current_camera,
                           board_observation_count, valid_pose_count,
                           outer_correspondence_count + internal_correspondence_count,
                           outer_correspondence_count, internal_correspondence_count, camera_rmse));

    previous_detections = detections;
    const double improvement = run.iterations.back().score - previous_score;
    previous_score = run.iterations.back().score;
    if (std::abs(improvement) < args.convergence_threshold) {
      break;
    }
  }

  return run;
}

std::map<int, const ati::InternalCornerDebugInfo*> BuildDebugIndex(
    const ati::ApriltagInternalDetectionResult& result) {
  std::map<int, const ati::InternalCornerDebugInfo*> debug_index;
  for (const auto& debug : result.internal_corner_debug) {
    debug_index[debug.point_id] = &debug;
  }
  return debug_index;
}

double PointDistance(const cv::Point2f& a, const cv::Point2f& b) {
  return std::hypot(static_cast<double>(a.x - b.x), static_cast<double>(a.y - b.y));
}

cv::Point2f SeedEquivalentPoint(const ati::InternalCornerDebugInfo& debug,
                                ati::InternalProjectionMode mode) {
  if (HasExplicitSphereSeedStage(mode)) {
    return debug.sphere_seed_image;
  }
  return debug.refined_image;
}

GroundTruthMap BuildConsensusGroundTruth(const std::vector<ImageRecord>& images,
                                         const DetectionMap& old_gt_detections,
                                         const DetectionMap& new_gt_detections,
                                         ConsensusGtStats* stats) {
  if (stats == nullptr) {
    throw std::runtime_error("BuildConsensusGroundTruth requires a valid stats pointer.");
  }

  GroundTruthMap references;
  for (const auto& old_entry : old_gt_detections) {
    const auto new_it = new_gt_detections.find(old_entry.first);
    if (new_it == new_gt_detections.end()) {
      continue;
    }
    if (!old_entry.second.available || !new_it->second.available ||
        !old_entry.second.result.tag_detected || !new_it->second.result.tag_detected) {
      continue;
    }

    const std::string& image_stem =
        images[static_cast<std::size_t>(old_entry.first.image_index)].stem;
    const std::map<int, const ati::InternalCornerDebugInfo*> old_debug =
        BuildDebugIndex(old_entry.second.result);
    const std::map<int, const ati::InternalCornerDebugInfo*> new_debug =
        BuildDebugIndex(new_it->second.result);
    GroundTruthPointMap point_map;

    for (const auto& debug_entry : old_debug) {
      const auto new_debug_it = new_debug.find(debug_entry.first);
      if (new_debug_it == new_debug.end()) {
        continue;
      }
      const ati::InternalCornerDebugInfo& old_debug_info = *debug_entry.second;
      const ati::InternalCornerDebugInfo& new_debug_info = *new_debug_it->second;
      if (!old_debug_info.valid || !new_debug_info.valid) {
        continue;
      }

      ++stats->overlapping_valid_pairs;
      ++stats->overlapping_valid_pairs_per_image[image_stem];

      const double refined_gap =
          PointDistance(old_debug_info.refined_image, new_debug_info.refined_image);
      if (refined_gap > kConsensusGtDistanceThresholdPx) {
        ++stats->filtered_by_distance;
        ++stats->filtered_by_distance_per_image[image_stem];
        continue;
      }

      point_map[debug_entry.first] =
          0.5f * (old_debug_info.refined_image + new_debug_info.refined_image);
      ++stats->gt_points;
      ++stats->gt_points_per_image[image_stem];
    }

    if (!point_map.empty()) {
      references[old_entry.first] = point_map;
    }
  }
  return references;
}

std::vector<PerPointSeedMetric> CollectPerPointMetrics(
    const std::vector<ImageRecord>& images,
    const MethodRunData& old_run,
    const MethodRunData& new_run,
    const GroundTruthMap& references) {
  std::vector<PerPointSeedMetric> rows;
  const std::size_t iteration_count =
      std::min(old_run.iterations.size(), new_run.iterations.size());
  for (std::size_t iteration_index = 0; iteration_index < iteration_count; ++iteration_index) {
    const DetectionMap& old_detections = old_run.iterations[iteration_index].detections;
    const DetectionMap& new_detections = new_run.iterations[iteration_index].detections;

    for (const auto& gt_entry : references) {
      const auto old_it = old_detections.find(gt_entry.first);
      const auto new_it = new_detections.find(gt_entry.first);
      if (old_it == old_detections.end() || new_it == new_detections.end() ||
          !old_it->second.available || !new_it->second.available ||
          !old_it->second.result.tag_detected || !new_it->second.result.tag_detected) {
        continue;
      }

      const std::map<int, const ati::InternalCornerDebugInfo*> old_debug =
          BuildDebugIndex(old_it->second.result);
      const std::map<int, const ati::InternalCornerDebugInfo*> new_debug =
          BuildDebugIndex(new_it->second.result);
      const std::string& image_stem =
          images[static_cast<std::size_t>(gt_entry.first.image_index)].stem;
      const std::string& group_name =
          images[static_cast<std::size_t>(gt_entry.first.image_index)].group_name;

      for (const auto& point_entry : gt_entry.second) {
        const auto old_debug_it = old_debug.find(point_entry.first);
        const auto new_debug_it = new_debug.find(point_entry.first);
        if (old_debug_it == old_debug.end() || new_debug_it == new_debug.end()) {
          continue;
        }

        const ati::InternalCornerDebugInfo& old_debug_info = *old_debug_it->second;
        const ati::InternalCornerDebugInfo& new_debug_info = *new_debug_it->second;
        PerPointSeedMetric row;
        row.iteration = static_cast<int>(iteration_index);
        row.image_stem = image_stem;
        row.group_name = group_name;
        row.board_id = gt_entry.first.board_id;
        row.point_id = point_entry.first;
        row.old_valid = old_debug_info.valid;
        row.new_valid = new_debug_info.valid;
        const cv::Point2f old_seed_point = SeedEquivalentPoint(old_debug_info, old_run.mode);
        const cv::Point2f new_seed_point = SeedEquivalentPoint(new_debug_info, new_run.mode);
        row.d_p_old = PointDistance(old_debug_info.predicted_image, point_entry.second);
        row.d_p_new = PointDistance(new_debug_info.predicted_image, point_entry.second);
        row.d_ss_old = PointDistance(old_seed_point, point_entry.second);
        row.d_ss_new = PointDistance(new_seed_point, point_entry.second);
        row.d_r_old = PointDistance(old_debug_info.refined_image, point_entry.second);
        row.d_r_new = PointDistance(new_debug_info.refined_image, point_entry.second);
        row.imp_old = row.d_p_old - row.d_ss_old;
        row.imp_new = row.d_p_new - row.d_ss_new;
        row.delta_imp = row.imp_new - row.imp_old;
        row.move_old = PointDistance(old_debug_info.predicted_image, old_seed_point);
        row.move_new = PointDistance(new_debug_info.predicted_image, new_seed_point);
        row.predicted_gap =
            PointDistance(old_debug_info.predicted_image, new_debug_info.predicted_image);
        rows.push_back(row);
      }
    }
  }
  return rows;
}

std::vector<PerImageSeedSummary> SummarizePerImage(
    const std::vector<PerPointSeedMetric>& point_rows,
    const ConsensusGtStats& gt_stats) {
  std::map<std::pair<int, std::string>, std::vector<const PerPointSeedMetric*> > buckets;
  std::map<std::string, std::string> image_to_group;
  for (const auto& row : point_rows) {
    buckets[std::make_pair(row.iteration, row.image_stem)].push_back(&row);
    image_to_group[row.image_stem] = row.group_name;
  }

  std::vector<PerImageSeedSummary> summaries;
  for (const auto& bucket : buckets) {
    PerImageSeedSummary summary;
    summary.iteration = bucket.first.first;
    summary.image_stem = bucket.first.second;
    summary.group_name = image_to_group[summary.image_stem];
    summary.gt_points = gt_stats.gt_points_per_image.count(summary.image_stem) > 0
                            ? gt_stats.gt_points_per_image.at(summary.image_stem)
                            : 0;
    summary.filtered_points = gt_stats.filtered_by_distance_per_image.count(summary.image_stem) > 0
                                  ? gt_stats.filtered_by_distance_per_image.at(summary.image_stem)
                                  : 0;
    summary.low_coverage = summary.gt_points < kLowCoverageThreshold;

    const double count = static_cast<double>(bucket.second.size());
    for (const auto* row : bucket.second) {
      summary.avg_d_p_old += row->d_p_old;
      summary.avg_d_p_new += row->d_p_new;
      summary.avg_d_ss_old += row->d_ss_old;
      summary.avg_d_ss_new += row->d_ss_new;
      summary.avg_d_r_old += row->d_r_old;
      summary.avg_d_r_new += row->d_r_new;
      summary.avg_imp_old += row->imp_old;
      summary.avg_imp_new += row->imp_new;
      summary.avg_delta_imp += row->delta_imp;
      summary.avg_move_old += row->move_old;
      summary.avg_move_new += row->move_new;
      summary.avg_predicted_gap += row->predicted_gap;
    }

    summary.avg_d_p_old /= count;
    summary.avg_d_p_new /= count;
    summary.avg_d_ss_old /= count;
    summary.avg_d_ss_new /= count;
    summary.avg_d_r_old /= count;
    summary.avg_d_r_new /= count;
    summary.avg_imp_old /= count;
    summary.avg_imp_new /= count;
    summary.avg_delta_imp /= count;
    summary.avg_move_old /= count;
    summary.avg_move_new /= count;
    summary.avg_predicted_gap /= count;
    summaries.push_back(summary);
  }

  std::sort(summaries.begin(), summaries.end(),
            [](const PerImageSeedSummary& lhs, const PerImageSeedSummary& rhs) {
              if (lhs.iteration != rhs.iteration) {
                return lhs.iteration < rhs.iteration;
              }
              return lhs.image_stem < rhs.image_stem;
            });
  return summaries;
}

std::vector<IterationMethodSummary> SummarizePerIteration(
    const std::vector<PerPointSeedMetric>& point_rows,
    const MethodRunData& old_run,
    const MethodRunData& new_run) {
  std::map<int, std::vector<const PerPointSeedMetric*> > buckets;
  for (const auto& row : point_rows) {
    buckets[row.iteration].push_back(&row);
  }

  std::vector<IterationMethodSummary> summaries;
  const std::size_t iteration_count =
      std::min(old_run.iterations.size(), new_run.iterations.size());
  for (std::size_t iteration_index = 0; iteration_index < iteration_count; ++iteration_index) {
    const auto bucket_it = buckets.find(static_cast<int>(iteration_index));
    const std::vector<const PerPointSeedMetric*> rows =
        bucket_it == buckets.end() ? std::vector<const PerPointSeedMetric*>() : bucket_it->second;

    const auto append_summary =
        [&](const MethodRunData& run, bool use_new_method_metrics) {
          IterationMethodSummary summary;
          summary.method_name = run.method_name;
          summary.iteration = static_cast<int>(iteration_index);
          summary.label = run.iterations[iteration_index].label;
          summary.matched_gt_points = static_cast<int>(rows.size());
          summary.valid_internal_points = run.iterations[iteration_index].metrics.valid_points;
          summary.total_internal_points = run.iterations[iteration_index].metrics.total_points;
          summary.camera_rmse = run.iterations[iteration_index].camera_rmse;
          summary.score = run.iterations[iteration_index].score;

          if (!rows.empty()) {
            for (const auto* row : rows) {
              summary.avg_d_p += use_new_method_metrics ? row->d_p_new : row->d_p_old;
              summary.avg_d_ss += use_new_method_metrics ? row->d_ss_new : row->d_ss_old;
              summary.avg_d_r += use_new_method_metrics ? row->d_r_new : row->d_r_old;
              summary.avg_improvement += use_new_method_metrics ? row->imp_new : row->imp_old;
              summary.avg_move += use_new_method_metrics ? row->move_new : row->move_old;
              summary.avg_predicted_gap += row->predicted_gap;
            }
            const double count = static_cast<double>(rows.size());
            summary.avg_d_p /= count;
            summary.avg_d_ss /= count;
            summary.avg_d_r /= count;
            summary.avg_improvement /= count;
            summary.avg_move /= count;
            summary.avg_predicted_gap /= count;
          }
          summaries.push_back(summary);
        };

    append_summary(old_run, false);
    append_summary(new_run, true);
  }

  return summaries;
}

std::string FormatDouble(double value, int precision = 4) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

void WritePerPointCsv(const boost::filesystem::path& csv_path,
                      const std::vector<PerPointSeedMetric>& rows) {
  std::ofstream stream(csv_path.string().c_str());
  stream << "iteration,image_stem,group_name,board_id,point_id,old_valid,new_valid,"
         << "d_p_old,d_p_new,d_ss_old,d_ss_new,d_r_old,d_r_new,"
         << "imp_old,imp_new,delta_imp,move_old,move_new,predicted_gap\n";
  for (const auto& row : rows) {
    stream << row.iteration << ","
           << row.image_stem << ","
           << row.group_name << ","
           << row.board_id << ","
           << row.point_id << ","
           << (row.old_valid ? 1 : 0) << ","
           << (row.new_valid ? 1 : 0) << ","
           << row.d_p_old << ","
           << row.d_p_new << ","
           << row.d_ss_old << ","
           << row.d_ss_new << ","
           << row.d_r_old << ","
           << row.d_r_new << ","
           << row.imp_old << ","
           << row.imp_new << ","
           << row.delta_imp << ","
           << row.move_old << ","
           << row.move_new << ","
           << row.predicted_gap << "\n";
  }
}

void WritePerImageCsv(const boost::filesystem::path& csv_path,
                      const std::vector<PerImageSeedSummary>& rows) {
  std::ofstream stream(csv_path.string().c_str());
  stream << "iteration,image_stem,group_name,gt_points,filtered_points,low_coverage,"
         << "avg_d_p_old,avg_d_p_new,avg_d_ss_old,avg_d_ss_new,avg_d_r_old,avg_d_r_new,"
         << "avg_imp_old,avg_imp_new,avg_delta_imp,avg_move_old,avg_move_new,avg_predicted_gap\n";
  for (const auto& row : rows) {
    stream << row.iteration << ","
           << row.image_stem << ","
           << row.group_name << ","
           << row.gt_points << ","
           << row.filtered_points << ","
           << (row.low_coverage ? 1 : 0) << ","
           << row.avg_d_p_old << ","
           << row.avg_d_p_new << ","
           << row.avg_d_ss_old << ","
           << row.avg_d_ss_new << ","
           << row.avg_d_r_old << ","
           << row.avg_d_r_new << ","
           << row.avg_imp_old << ","
           << row.avg_imp_new << ","
           << row.avg_delta_imp << ","
           << row.avg_move_old << ","
           << row.avg_move_new << ","
           << row.avg_predicted_gap << "\n";
  }
}

void WritePerIterationCsv(const boost::filesystem::path& csv_path,
                          const std::vector<IterationMethodSummary>& rows) {
  std::ofstream stream(csv_path.string().c_str());
  stream << "method_name,iteration,label,matched_gt_points,avg_d_p,avg_d_ss,avg_d_r,"
         << "avg_improvement,avg_move,avg_predicted_gap,valid_internal_points,total_internal_points,"
         << "camera_rmse,score\n";
  for (const auto& row : rows) {
    stream << row.method_name << ","
           << row.iteration << ","
           << row.label << ","
           << row.matched_gt_points << ","
           << row.avg_d_p << ","
           << row.avg_d_ss << ","
           << row.avg_d_r << ","
           << row.avg_improvement << ","
           << row.avg_move << ","
           << row.avg_predicted_gap << ","
           << row.valid_internal_points << ","
           << row.total_internal_points << ","
           << row.camera_rmse << ","
           << row.score << "\n";
  }
}

cv::Mat MakeChartCanvas(const std::string& title, const std::string& subtitle) {
  cv::Mat canvas(620, 980, CV_8UC3, cv::Scalar(248, 248, 248));
  cv::putText(canvas, title, cv::Point(36, 44), cv::FONT_HERSHEY_SIMPLEX, 0.95,
              cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
  cv::putText(canvas, subtitle, cv::Point(36, 76), cv::FONT_HERSHEY_SIMPLEX, 0.52,
              cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
  return canvas;
}

void DrawChartAxes(cv::Mat* canvas,
                   const cv::Rect& plot_rect,
                   double y_min,
                   double y_max,
                   const std::string& y_label) {
  if (canvas == nullptr) {
    return;
  }
  cv::rectangle(*canvas, plot_rect, cv::Scalar(255, 255, 255), cv::FILLED);
  cv::rectangle(*canvas, plot_rect, cv::Scalar(110, 110, 110), 1, cv::LINE_AA);
  for (int tick = 0; tick <= 5; ++tick) {
    const int y = plot_rect.y + static_cast<int>(std::lround(
        static_cast<double>(plot_rect.height) * tick / 5.0));
    cv::line(*canvas, cv::Point(plot_rect.x, y),
             cv::Point(plot_rect.x + plot_rect.width, y),
             cv::Scalar(228, 228, 228), 1, cv::LINE_AA);
    const double value = y_max - (y_max - y_min) * tick / 5.0;
    cv::putText(*canvas, FormatDouble(value, 2),
                cv::Point(plot_rect.x - 68, y + 4), cv::FONT_HERSHEY_PLAIN, 0.9,
                cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
  }
  cv::putText(*canvas, y_label, cv::Point(plot_rect.x - 68, plot_rect.y - 10),
              cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
}

cv::Point ChartPoint(int index,
                     int count,
                     double value,
                     double y_min,
                     double y_max,
                     const cv::Rect& plot_rect) {
  const double x_ratio = count <= 1 ? 0.5 : static_cast<double>(index) / static_cast<double>(count - 1);
  const double y_ratio =
      y_max <= y_min ? 0.5 : (value - y_min) / std::max(1e-9, y_max - y_min);
  const int x = plot_rect.x + static_cast<int>(std::lround(x_ratio * plot_rect.width));
  const int y = plot_rect.y + plot_rect.height -
                static_cast<int>(std::lround(y_ratio * plot_rect.height));
  return cv::Point(x, y);
}

cv::Mat BuildIterationLineChart(const std::vector<IterationMethodSummary>& rows,
                                const std::string& old_method_name,
                                const std::string& new_method_name,
                                const std::string& title,
                                const std::string& subtitle,
                                bool use_ss_error) {
  std::vector<IterationMethodSummary> old_rows;
  std::vector<IterationMethodSummary> new_rows;
  for (const auto& row : rows) {
    if (row.method_name == old_method_name) {
      old_rows.push_back(row);
    } else if (row.method_name == new_method_name) {
      new_rows.push_back(row);
    }
  }

  const int point_count = static_cast<int>(std::min(old_rows.size(), new_rows.size()));
  double y_min = std::numeric_limits<double>::infinity();
  double y_max = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < point_count; ++i) {
    const double old_value = use_ss_error ? old_rows[i].avg_d_ss : old_rows[i].avg_improvement;
    const double new_value = use_ss_error ? new_rows[i].avg_d_ss : new_rows[i].avg_improvement;
    y_min = std::min(y_min, std::min(old_value, new_value));
    y_max = std::max(y_max, std::max(old_value, new_value));
  }
  if (!std::isfinite(y_min) || !std::isfinite(y_max)) {
    y_min = 0.0;
    y_max = 1.0;
  }
  const double padding = std::max(0.1, 0.12 * (y_max - y_min + 1e-6));
  y_min -= padding;
  y_max += padding;

  cv::Mat canvas = MakeChartCanvas(title, subtitle);
  const cv::Rect plot_rect(110, 112, 820, 410);
  DrawChartAxes(&canvas, plot_rect, y_min, y_max,
                use_ss_error ? "avg |SS-GT|" : "avg(dP-dSS)");

  const cv::Scalar old_color(255, 90, 180);
  const cv::Scalar new_color(60, 170, 60);
  for (int i = 1; i < point_count; ++i) {
    const cv::Point old_prev = ChartPoint(i - 1, point_count,
                                          use_ss_error ? old_rows[i - 1].avg_d_ss
                                                       : old_rows[i - 1].avg_improvement,
                                          y_min, y_max, plot_rect);
    const cv::Point old_curr = ChartPoint(i, point_count,
                                          use_ss_error ? old_rows[i].avg_d_ss
                                                       : old_rows[i].avg_improvement,
                                          y_min, y_max, plot_rect);
    const cv::Point new_prev = ChartPoint(i - 1, point_count,
                                          use_ss_error ? new_rows[i - 1].avg_d_ss
                                                       : new_rows[i - 1].avg_improvement,
                                          y_min, y_max, plot_rect);
    const cv::Point new_curr = ChartPoint(i, point_count,
                                          use_ss_error ? new_rows[i].avg_d_ss
                                                       : new_rows[i].avg_improvement,
                                          y_min, y_max, plot_rect);
    cv::line(canvas, old_prev, old_curr, old_color, 2, cv::LINE_AA);
    cv::line(canvas, new_prev, new_curr, new_color, 2, cv::LINE_AA);
  }

  for (int i = 0; i < point_count; ++i) {
    const cv::Point old_point = ChartPoint(i, point_count,
                                           use_ss_error ? old_rows[i].avg_d_ss
                                                        : old_rows[i].avg_improvement,
                                           y_min, y_max, plot_rect);
    const cv::Point new_point = ChartPoint(i, point_count,
                                           use_ss_error ? new_rows[i].avg_d_ss
                                                        : new_rows[i].avg_improvement,
                                           y_min, y_max, plot_rect);
    cv::circle(canvas, old_point, 4, old_color, cv::FILLED, cv::LINE_AA);
    cv::circle(canvas, new_point, 4, new_color, cv::FILLED, cv::LINE_AA);
    cv::putText(canvas, std::to_string(i),
                cv::Point(old_point.x - 5, plot_rect.y + plot_rect.height + 28),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
  }

  cv::line(canvas, cv::Point(660, 38), cv::Point(705, 38), old_color, 2, cv::LINE_AA);
  cv::putText(canvas, old_method_name, cv::Point(714, 43), cv::FONT_HERSHEY_PLAIN, 1.1,
              cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
  cv::line(canvas, cv::Point(660, 62), cv::Point(705, 62), new_color, 2, cv::LINE_AA);
  cv::putText(canvas, new_method_name, cv::Point(714, 67), cv::FONT_HERSHEY_PLAIN, 1.1,
              cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
  return canvas;
}

cv::Mat BuildPerImageImprovementBarChart(const std::vector<PerImageSeedSummary>& rows,
                                         const std::string& old_method_name,
                                         const std::string& new_method_name) {
  std::vector<PerImageSeedSummary> iteration_zero_rows;
  for (const auto& row : rows) {
    if (row.iteration == 0) {
      iteration_zero_rows.push_back(row);
    }
  }
  std::sort(iteration_zero_rows.begin(), iteration_zero_rows.end(),
            [](const PerImageSeedSummary& lhs, const PerImageSeedSummary& rhs) {
              return lhs.image_stem < rhs.image_stem;
            });

  double max_abs_value = 0.0;
  for (const auto& row : iteration_zero_rows) {
    max_abs_value = std::max(max_abs_value, std::abs(row.avg_delta_imp));
  }
  max_abs_value = std::max(0.5, max_abs_value * 1.2);

  cv::Mat canvas = MakeChartCanvas(
      "Per-image SS improvement delta at iteration 0",
      "delta_imp = (dP-dSS)_" + new_method_name + " - (dP-dSS)_" + old_method_name +
          ", positive is better");
  const cv::Rect plot_rect(90, 112, 840, 410);
  DrawChartAxes(&canvas, plot_rect, -max_abs_value, max_abs_value, "delta_imp");

  const int count = static_cast<int>(iteration_zero_rows.size());
  for (int i = 0; i < count; ++i) {
    const double value = iteration_zero_rows[i].avg_delta_imp;
    const double x_ratio =
        (static_cast<double>(i) + 0.5) / std::max(1.0, static_cast<double>(count));
    const int center_x = plot_rect.x + static_cast<int>(std::lround(x_ratio * plot_rect.width));
    const int bar_width = std::max(18, plot_rect.width / std::max(1, count * 2));
    const int zero_y = ChartPoint(0, 1, 0.0, -max_abs_value, max_abs_value, plot_rect).y;
    const int value_y = ChartPoint(0, 1, value, -max_abs_value, max_abs_value, plot_rect).y;
    const cv::Rect bar_rect(center_x - bar_width / 2, std::min(zero_y, value_y), bar_width,
                            std::max(2, std::abs(zero_y - value_y)));
    const cv::Scalar color = value >= 0.0 ? cv::Scalar(70, 170, 70) : cv::Scalar(80, 120, 220);
    cv::rectangle(canvas, bar_rect, color, cv::FILLED);
    cv::rectangle(canvas, bar_rect, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    cv::putText(canvas, iteration_zero_rows[i].image_stem,
                cv::Point(center_x - 26, plot_rect.y + plot_rect.height + 26),
                cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
  }
  return canvas;
}

cv::Mat BuildMatchedPointsBarChart(const ConsensusGtStats& gt_stats) {
  std::vector<std::string> image_stems;
  for (const auto& image_stem : kSelectedImageStems) {
    image_stems.push_back(image_stem);
  }

  int max_total = 1;
  for (const auto& image_stem : image_stems) {
    const int gt_count = gt_stats.gt_points_per_image.count(image_stem) > 0
                             ? gt_stats.gt_points_per_image.at(image_stem)
                             : 0;
    const int filtered = gt_stats.filtered_by_distance_per_image.count(image_stem) > 0
                             ? gt_stats.filtered_by_distance_per_image.at(image_stem)
                             : 0;
    max_total = std::max(max_total, gt_count + filtered);
  }

  cv::Mat canvas = MakeChartCanvas(
      "Consensus GT coverage per image",
      "green: retained GT points, orange: filtered because |R_old-R_new| > 2 px");
  const cv::Rect plot_rect(90, 112, 840, 410);
  DrawChartAxes(&canvas, plot_rect, 0.0, static_cast<double>(max_total) * 1.15, "points");

  const int count = static_cast<int>(image_stems.size());
  for (int i = 0; i < count; ++i) {
    const std::string& image_stem = image_stems[static_cast<std::size_t>(i)];
    const int gt_count = gt_stats.gt_points_per_image.count(image_stem) > 0
                             ? gt_stats.gt_points_per_image.at(image_stem)
                             : 0;
    const int filtered = gt_stats.filtered_by_distance_per_image.count(image_stem) > 0
                             ? gt_stats.filtered_by_distance_per_image.at(image_stem)
                             : 0;
    const double x_ratio =
        (static_cast<double>(i) + 0.5) / std::max(1.0, static_cast<double>(count));
    const int center_x = plot_rect.x + static_cast<int>(std::lround(x_ratio * plot_rect.width));
    const int bar_width = std::max(18, plot_rect.width / std::max(1, count * 2));
    const int zero_y = ChartPoint(0, 1, 0.0, 0.0, static_cast<double>(max_total) * 1.15, plot_rect).y;
    const int gt_y = ChartPoint(0, 1, static_cast<double>(gt_count), 0.0,
                                static_cast<double>(max_total) * 1.15, plot_rect).y;
    const int total_y = ChartPoint(0, 1, static_cast<double>(gt_count + filtered), 0.0,
                                   static_cast<double>(max_total) * 1.15, plot_rect).y;

    cv::Rect gt_rect(center_x - bar_width / 2, gt_y, bar_width, std::max(2, zero_y - gt_y));
    cv::Rect filtered_rect(center_x - bar_width / 2, total_y, bar_width,
                           std::max(2, gt_y - total_y));
    cv::rectangle(canvas, gt_rect, cv::Scalar(70, 170, 70), cv::FILLED);
    cv::rectangle(canvas, filtered_rect, cv::Scalar(0, 165, 255), cv::FILLED);
    cv::rectangle(canvas,
                  cv::Rect(center_x - bar_width / 2, total_y, bar_width, zero_y - total_y),
                  cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    cv::putText(canvas, image_stem, cv::Point(center_x - 24, plot_rect.y + plot_rect.height + 26),
                cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
  }

  cv::rectangle(canvas, cv::Rect(690, 28, 14, 10), cv::Scalar(70, 170, 70), cv::FILLED);
  cv::putText(canvas, "retained", cv::Point(712, 38), cv::FONT_HERSHEY_PLAIN, 1.0,
              cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
  cv::rectangle(canvas, cv::Rect(690, 48, 14, 10), cv::Scalar(0, 165, 255), cv::FILLED);
  cv::putText(canvas, "filtered", cv::Point(712, 58), cv::FONT_HERSHEY_PLAIN, 1.0,
              cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
  return canvas;
}

void SaveComparisonVisuals(const boost::filesystem::path& visuals_dir,
                           const std::vector<ImageRecord>& images,
                           const MethodRunData& old_run,
                           const MethodRunData& new_run) {
  boost::filesystem::create_directories(visuals_dir);
  const std::size_t iteration_index = 0;
  for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
    BoardKey key;
    key.image_index = static_cast<int>(image_index);
    key.board_id = old_run.iterations[iteration_index].detections.begin()->first.board_id;

    const auto old_it = old_run.iterations[iteration_index].detections.find(key);
    const auto new_it = new_run.iterations[iteration_index].detections.find(key);
    if (old_it == old_run.iterations[iteration_index].detections.end() ||
        new_it == new_run.iterations[iteration_index].detections.end() ||
        !old_it->second.available || !new_it->second.available) {
      continue;
    }

    const cv::Mat old_seed = ati::BuildInternalSeedOverlay(images[image_index].image, old_it->second.result);
    const cv::Mat new_seed = ati::BuildInternalSeedOverlay(images[image_index].image, new_it->second.result);
    if (!old_seed.empty() && !new_seed.empty()) {
      const cv::Mat compare = ati::BuildSideBySideComparisonCanvas(
          old_seed, new_seed, "old: " + old_run.method_name, "new: " + new_run.method_name,
          images[image_index].stem + " iteration 0 P-SS-R comparison",
          "P orange, SS magenta when available, R green");
      if (!compare.empty()) {
        cv::imwrite((visuals_dir / (images[image_index].stem + "_iter0_seed_compare.png")).string(),
                    compare);
      }
    }

    const cv::Mat old_method_view = ati::BuildInternalMethodSpaceDebugView(old_it->second.result);
    const cv::Mat new_method_view = ati::BuildInternalMethodSpaceDebugView(new_it->second.result);
    if (!old_method_view.empty() && !new_method_view.empty()) {
      const cv::Mat compare = ati::BuildSideBySideComparisonCanvas(
          old_method_view, new_method_view,
          "old: " + old_run.method_name, "new: " + new_run.method_name,
          images[image_index].stem + " iteration 0 method-space comparison",
          "old/new method-space debug views");
      if (!compare.empty()) {
        cv::imwrite((visuals_dir / (images[image_index].stem + "_iter0_method_compare.png")).string(),
                    compare);
      }
    }
  }
}

std::vector<PerImageSeedSummary> SelectRepresentativeRows(
    const std::vector<PerImageSeedSummary>& rows) {
  std::vector<PerImageSeedSummary> iteration_zero_rows;
  for (const auto& row : rows) {
    if (row.iteration == 0) {
      iteration_zero_rows.push_back(row);
    }
  }
  if (iteration_zero_rows.empty()) {
    return {};
  }

  std::vector<PerImageSeedSummary> sorted_by_delta = iteration_zero_rows;
  std::sort(sorted_by_delta.begin(), sorted_by_delta.end(),
            [](const PerImageSeedSummary& lhs, const PerImageSeedSummary& rhs) {
              return lhs.avg_delta_imp < rhs.avg_delta_imp;
            });
  std::vector<PerImageSeedSummary> sorted_by_dp = iteration_zero_rows;
  std::sort(sorted_by_dp.begin(), sorted_by_dp.end(),
            [](const PerImageSeedSummary& lhs, const PerImageSeedSummary& rhs) {
              return lhs.avg_d_p_old > rhs.avg_d_p_old;
            });

  std::vector<PerImageSeedSummary> picks;
  auto try_add = [&](const PerImageSeedSummary& candidate) {
    for (const auto& existing : picks) {
      if (existing.image_stem == candidate.image_stem) {
        return;
      }
    }
    picks.push_back(candidate);
  };

  try_add(sorted_by_delta.back());
  try_add(sorted_by_delta[sorted_by_delta.size() / 2]);
  try_add(sorted_by_dp.front());
  bool found_non_positive = false;
  for (const auto& row : sorted_by_delta) {
    if (row.avg_delta_imp <= 0.0) {
      try_add(row);
      found_non_positive = true;
      break;
    }
  }
  if (!found_non_positive) {
    try_add(sorted_by_delta.front());
  }
  for (const auto& row : sorted_by_delta) {
    if (picks.size() >= 4) {
      break;
    }
    try_add(row);
  }
  return picks;
}

void SaveRepresentativeMontage(const boost::filesystem::path& visuals_dir,
                               const std::vector<PerImageSeedSummary>& representative_rows) {
  std::vector<cv::Mat> rows;
  for (const auto& row : representative_rows) {
    const boost::filesystem::path image_path =
        visuals_dir / (row.image_stem + "_iter0_seed_compare.png");
    if (!boost::filesystem::exists(image_path)) {
      continue;
    }
    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      continue;
    }
    rows.push_back(image);
  }

  if (rows.empty()) {
    return;
  }

  const int montage_width = 1800;
  std::vector<cv::Mat> resized_rows;
  int total_height = 80;
  for (const auto& row : rows) {
    const double scale = static_cast<double>(montage_width) / static_cast<double>(row.cols);
    cv::Mat resized;
    cv::resize(row, resized, cv::Size(montage_width, static_cast<int>(std::lround(row.rows * scale))));
    total_height += resized.rows;
    resized_rows.push_back(resized);
  }
  cv::Mat canvas(total_height, montage_width, CV_8UC3, cv::Scalar(245, 245, 245));
  cv::putText(canvas, "Representative iteration 0 seed comparisons",
              cv::Point(18, 36), cv::FONT_HERSHEY_SIMPLEX, 0.95, cv::Scalar(20, 20, 20), 2,
              cv::LINE_AA);
  cv::putText(canvas,
              "selection rule: max delta_imp / median delta_imp / max dP / typical non-positive delta",
              cv::Point(18, 62), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(70, 70, 70), 1,
              cv::LINE_AA);

  int y = 80;
  for (const auto& row : resized_rows) {
    row.copyTo(canvas(cv::Rect(0, y, row.cols, row.rows)));
    y += row.rows;
  }
  cv::imwrite((visuals_dir / "representative_montage.png").string(), canvas);
}

const IterationMethodSummary* FindIterationSummary(const std::vector<IterationMethodSummary>& summaries,
                                                   const std::string& method_name,
                                                   int iteration) {
  for (const auto& summary : summaries) {
    if (summary.method_name == method_name && summary.iteration == iteration) {
      return &summary;
    }
  }
  return nullptr;
}

void WriteReport(const boost::filesystem::path& report_path,
                 const std::vector<ImageRecord>& images,
                 const MethodRunData& old_run,
                 const MethodRunData& new_run,
                 const CmdArgs& args,
                 const ConsensusGtStats& gt_stats,
                 const std::vector<PerImageSeedSummary>& per_image_rows,
                 const std::vector<IterationMethodSummary>& per_iteration_rows,
                 const std::vector<PerImageSeedSummary>& representative_rows) {
  const IterationMethodSummary* iter0_old =
      FindIterationSummary(per_iteration_rows, old_run.method_name, 0);
  const IterationMethodSummary* iter0_new =
      FindIterationSummary(per_iteration_rows, new_run.method_name, 0);
  const IterationMethodSummary* final_old =
      FindIterationSummary(per_iteration_rows, old_run.method_name,
                           static_cast<int>(old_run.iterations.size() - 1));
  const IterationMethodSummary* final_new =
      FindIterationSummary(per_iteration_rows, new_run.method_name,
                           static_cast<int>(new_run.iterations.size() - 1));

  std::ofstream stream(report_path.string().c_str());
  const bool old_has_explicit_seed = HasExplicitSphereSeedStage(old_run.mode);
  const bool new_has_explicit_seed = HasExplicitSphereSeedStage(new_run.mode);
  stream << "# SS 方法对比实验报告\n\n";
  stream << "## 1. 实验设置\n\n";
  stream << "- 时间：2026-04-19\n";
  stream << "- 图像子集：";
  for (std::size_t index = 0; index < images.size(); ++index) {
    if (index > 0) {
      stream << "，";
    }
    stream << images[index].stem;
  }
  stream << "\n";
  stream << "- 旧方法：`" << old_run.method_name << "`，" << DescribeMethodForReport(old_run.mode) << "\n";
  stream << "- 新方法：`" << new_run.method_name << "`，" << DescribeMethodForReport(new_run.mode) << "\n";
  if (!old_has_explicit_seed || !new_has_explicit_seed) {
    stream << "- 统一口径说明：若某方法没有独立的图像域 `SS` 中间点，本报告将其"
           << "“方法内部精修后回到图像域的点”记作 `SS(eq)`，用于和带显式 `SS` 的方法对齐比较。\n";
  }
  stream << "- 迭代设置：`max_iterations=" << args.max_iterations
         << "`，`pose_refinement_rounds=" << args.pose_refinement_rounds << "`，"
         << "`use_internal_points_for_update="
         << (args.use_internal_points_for_update ? "true" : "false") << "`，"
         << "`internal_point_quality_threshold=" << FormatDouble(args.internal_point_quality_threshold, 2)
         << "`\n";
  stream << "- 共识 GT：`R_gt (consensus proxy)`，由 calibrated camera 下两种方法的最终 `R` 共识构造\n";
  stream << "  - overlapping valid pairs: " << gt_stats.overlapping_valid_pairs << "\n";
  stream << "  - filtered by |R_old-R_new| > 2 px: " << gt_stats.filtered_by_distance << "\n";
  stream << "  - retained GT points: " << gt_stats.gt_points << "\n\n";

  stream << "## 2. Iteration 0 纯方法对比\n\n";
  if (iter0_old != nullptr && iter0_new != nullptr) {
    stream << "| 指标 | " << old_run.method_name << " | " << new_run.method_name << " | 差值 |\n";
    stream << "| --- | ---: | ---: | ---: |\n";
    stream << "| matched gt points | " << iter0_old->matched_gt_points << " | "
           << iter0_new->matched_gt_points << " | 0 |\n";
    stream << "| avg P-GT | " << FormatDouble(iter0_old->avg_d_p, 4) << " | "
           << FormatDouble(iter0_new->avg_d_p, 4) << " | "
           << FormatDouble(iter0_new->avg_d_p - iter0_old->avg_d_p, 4) << " |\n";
    stream << "| avg SS-GT | " << FormatDouble(iter0_old->avg_d_ss, 4) << " | "
           << FormatDouble(iter0_new->avg_d_ss, 4) << " | "
           << FormatDouble(iter0_new->avg_d_ss - iter0_old->avg_d_ss, 4) << " |\n";
    stream << "| avg improvement = dP-dSS | " << FormatDouble(iter0_old->avg_improvement, 4) << " | "
           << FormatDouble(iter0_new->avg_improvement, 4) << " | "
           << FormatDouble(iter0_new->avg_improvement - iter0_old->avg_improvement, 4) << " |\n";
    stream << "| avg P-SS | " << FormatDouble(iter0_old->avg_move, 4) << " | "
           << FormatDouble(iter0_new->avg_move, 4) << " | "
           << FormatDouble(iter0_new->avg_move - iter0_old->avg_move, 4) << " |\n";
    stream << "| avg predicted gap old-vs-new | " << FormatDouble(iter0_old->avg_predicted_gap, 6)
           << " | " << FormatDouble(iter0_new->avg_predicted_gap, 6) << " | 0 |\n\n";
  }
  stream << "![iteration_ss_error_curve](charts/iteration_ss_error_curve.png)\n\n";
  stream << "![per_image_improvement_bar](charts/per_image_improvement_bar.png)\n\n";

  stream << (args.max_iterations > 0 ? "## 3. Full Iterative 对比\n\n"
                                     : "## 3. Iteration 0 结果总览\n\n");
  stream << "| iteration | method | avg SS-GT | avg R-GT | avg improvement | valid internal | camera_rmse |\n";
  stream << "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n";
  for (const auto& row : per_iteration_rows) {
    stream << "| " << row.iteration
           << " | " << row.method_name
           << " | " << FormatDouble(row.avg_d_ss, 4)
           << " | " << FormatDouble(row.avg_d_r, 4)
           << " | " << FormatDouble(row.avg_improvement, 4)
           << " | " << row.valid_internal_points << "/" << row.total_internal_points
           << " | " << FormatDouble(row.camera_rmse, 4) << " |\n";
  }
  stream << "\n![iteration_improvement_curve](charts/iteration_improvement_curve.png)\n\n";

  if (final_old != nullptr && final_new != nullptr) {
    stream << (args.max_iterations > 0 ? "最终迭代（old iter " : "本次运行仅保留 iteration 0（old iter ")
           << (old_run.iterations.empty() ? 0 : old_run.iterations.back().iteration_index)
           << " / new iter "
           << (new_run.iterations.empty() ? 0 : new_run.iterations.back().iteration_index)
           << "）下：\n\n";
    stream << "- `" << old_run.method_name << "`: avg `|SS-GT|` = " << FormatDouble(final_old->avg_d_ss, 4)
           << ", avg `|R-GT|` = " << FormatDouble(final_old->avg_d_r, 4)
           << ", valid internal = " << final_old->valid_internal_points << "/"
           << final_old->total_internal_points
           << ", camera RMSE = " << FormatDouble(final_old->camera_rmse, 4) << "\n";
    stream << "- `" << new_run.method_name << "`: avg `|SS-GT|` = " << FormatDouble(final_new->avg_d_ss, 4)
           << ", avg `|R-GT|` = " << FormatDouble(final_new->avg_d_r, 4)
           << ", valid internal = " << final_new->valid_internal_points << "/"
           << final_new->total_internal_points
           << ", camera RMSE = " << FormatDouble(final_new->camera_rmse, 4) << "\n\n";
  }

  stream << "## 4. Per-image 观察（iteration 0）\n\n";
  stream << "| image | gt points | filtered | low coverage | avg P-GT old | avg SS-GT old | avg SS-GT new | avg delta_imp |\n";
  stream << "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |\n";
  for (const auto& row : per_image_rows) {
    if (row.iteration != 0) {
      continue;
    }
    stream << "| " << row.image_stem
           << " | " << row.gt_points
           << " | " << row.filtered_points
           << " | " << (row.low_coverage ? "yes" : "no")
           << " | " << FormatDouble(row.avg_d_p_old, 4)
           << " | " << FormatDouble(row.avg_d_ss_old, 4)
           << " | " << FormatDouble(row.avg_d_ss_new, 4)
           << " | " << FormatDouble(row.avg_delta_imp, 4) << " |\n";
  }
  stream << "\n![matched_points_bar](charts/matched_points_bar.png)\n\n";

  stream << "## 5. 代表性可视化\n\n";
  stream << "自动选择规则：`max delta_imp / median delta_imp / max dP / typical non-positive delta`。\n\n";
  for (const auto& row : representative_rows) {
    stream << "- `" << row.image_stem << "`: avg `delta_imp` = "
           << FormatDouble(row.avg_delta_imp, 4)
           << ", avg `|P-GT|` = " << FormatDouble(row.avg_d_p_old, 4)
           << ", gt points = " << row.gt_points
           << (row.low_coverage ? " (low-coverage)" : "") << "\n";
    stream << "  - [seed compare](visuals/" << row.image_stem << "_iter0_seed_compare.png)\n";
    stream << "  - [method-space compare](visuals/" << row.image_stem << "_iter0_method_compare.png)\n";
  }
  stream << "\n![representative_montage](visuals/representative_montage.png)\n\n";

  stream << "## 6. 结论摘要\n\n";
  if (iter0_old != nullptr && iter0_new != nullptr) {
    stream << "- `iteration 0` 下，两种方法的 `avg |P-GT|` 仅相差 "
           << FormatDouble(iter0_new->avg_d_p - iter0_old->avg_d_p, 6)
           << "，说明比较基准基本一致。\n";
    stream << "- `" << new_run.method_name << "` 相比 `" << old_run.method_name << "`，`avg |SS-GT|` "
           << ((iter0_new->avg_d_ss <= iter0_old->avg_d_ss) ? "更小" : "更大")
           << "，差值为 "
           << FormatDouble(iter0_new->avg_d_ss - iter0_old->avg_d_ss, 4) << "。\n";
    stream << "- 从 `avg improvement = dP-dSS` 看，新方法相对旧方法的净提升为 "
           << FormatDouble(iter0_new->avg_improvement - iter0_old->avg_improvement, 4) << "。\n";
  }
  if (final_old != nullptr && final_new != nullptr) {
    if (args.max_iterations > 0) {
      stream << "- 完整迭代后，`" << new_run.method_name << "` 对下游 `R` 与相机更新的影响，可从 "
             << "`avg |R-GT|` 与 `camera_rmse` 两列直接比较。\n";
    } else {
      stream << "- 这次结果只回答“初始粗相机下，两种方法在 iteration 0 的表现差异”，未继续做后续相机迭代更新。\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CmdArgs args = ParseArgs(argc, argv);
    const ati::InternalProjectionMode old_mode = ParseProjectionModeOrThrow(args.old_mode);
    const ati::InternalProjectionMode new_mode = ParseProjectionModeOrThrow(args.new_mode);
    ati::ApriltagInternalConfig config =
        ati::ApriltagInternalDetector::LoadConfig(args.config_path);
    ati::ApriltagInternalDetectionOptions detection_options =
        MakeDetectionOptionsFromConfig(config);
    detection_options.do_subpix_refinement = !args.no_subpix;

    const std::vector<ImageRecord> images = LoadSelectedImages(args.image_dir);
    std::vector<int> board_ids = args.board_ids;
    if (board_ids.empty()) {
      board_ids.push_back(config.tag_id);
    }

    const boost::filesystem::path output_dir(args.output_dir);
    const boost::filesystem::path charts_dir = output_dir / "charts";
    const boost::filesystem::path visuals_dir = output_dir / "visuals";
    boost::filesystem::create_directories(output_dir);
    boost::filesystem::create_directories(charts_dir);
    boost::filesystem::create_directories(visuals_dir);

    std::cout << "[compare] running calibrated detections and iterative comparison on "
              << images.size() << " images\n";
    std::cout << "  old_mode: " << ati::ToString(old_mode)
              << " new_mode: " << ati::ToString(new_mode) << "\n";

    const MethodRunData old_run =
        RunMethod(ati::ToString(old_mode), old_mode,
                  images, board_ids, config, detection_options, args);
    const MethodRunData new_run =
        RunMethod(ati::ToString(new_mode), new_mode,
                  images, board_ids, config, detection_options, args);

    ConsensusGtStats gt_stats;
    const GroundTruthMap gt_references =
        BuildConsensusGroundTruth(images, old_run.calibrated_detections,
                                  new_run.calibrated_detections, &gt_stats);
    const std::vector<PerPointSeedMetric> per_point_rows =
        CollectPerPointMetrics(images, old_run, new_run, gt_references);
    const std::vector<PerImageSeedSummary> per_image_rows =
        SummarizePerImage(per_point_rows, gt_stats);
    const std::vector<IterationMethodSummary> per_iteration_rows =
        SummarizePerIteration(per_point_rows, old_run, new_run);

    SaveComparisonVisuals(visuals_dir, images, old_run, new_run);
    const std::vector<PerImageSeedSummary> representative_rows =
        SelectRepresentativeRows(per_image_rows);
    SaveRepresentativeMontage(visuals_dir, representative_rows);

    const cv::Mat ss_error_chart = BuildIterationLineChart(
        per_iteration_rows, old_run.method_name, new_run.method_name,
        "Iteration curve: avg |SS-GT|",
        "comparison against consensus proxy GT", true);
    const cv::Mat improvement_chart = BuildIterationLineChart(
        per_iteration_rows, old_run.method_name, new_run.method_name,
        "Iteration curve: avg(dP-dSS)",
        "higher is better because SS is closer to GT than P", false);
    const cv::Mat per_image_bar_chart =
        BuildPerImageImprovementBarChart(per_image_rows, old_run.method_name, new_run.method_name);
    const cv::Mat matched_bar_chart = BuildMatchedPointsBarChart(gt_stats);

    cv::imwrite((charts_dir / "iteration_ss_error_curve.png").string(), ss_error_chart);
    cv::imwrite((charts_dir / "iteration_improvement_curve.png").string(), improvement_chart);
    cv::imwrite((charts_dir / "per_image_improvement_bar.png").string(), per_image_bar_chart);
    cv::imwrite((charts_dir / "matched_points_bar.png").string(), matched_bar_chart);

    WritePerPointCsv(output_dir / "per_point_seed_metrics.csv", per_point_rows);
    WritePerImageCsv(output_dir / "per_image_seed_summary.csv", per_image_rows);
    WritePerIterationCsv(output_dir / "per_iteration_method_summary.csv", per_iteration_rows);
    WriteReport(output_dir / "report.md", images, old_run, new_run, args, gt_stats,
                per_image_rows, per_iteration_rows, representative_rows);

    std::cout << "[compare] finished\n";
    std::cout << "  output_dir: " << output_dir.string() << "\n";
    std::cout << "  gt_points: " << gt_stats.gt_points
              << " filtered: " << gt_stats.filtered_by_distance << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[compare-internal-seed-methods] " << error.what() << "\n\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
