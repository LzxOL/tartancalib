#include <aslam/cameras/apriltag_internal/StereoExtrinsicProblemInput.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>
#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

namespace fs = boost::filesystem;

bool IsImageFile(const fs::path& path) {
  if (!fs::is_regular_file(path)) {
    return false;
  }
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
         extension == ".bmp" || extension == ".tif" || extension == ".tiff";
}

std::vector<std::string> CollectImagePaths(const std::string& image_path) {
  const fs::path input(image_path);
  if (!fs::exists(input)) {
    throw std::runtime_error("Input path does not exist: " + image_path);
  }

  fs::path directory = input;
  if (fs::is_regular_file(input)) {
    directory = input.parent_path();
  }
  if (!fs::is_directory(directory)) {
    throw std::runtime_error("Expected image path to be a directory or file inside it.");
  }

  std::vector<std::string> image_paths;
  for (fs::directory_iterator it(directory), end; it != end; ++it) {
    if (IsImageFile(it->path())) {
      image_paths.push_back(it->path().string());
    }
  }
  std::sort(image_paths.begin(), image_paths.end());
  return image_paths;
}

std::vector<FrozenRound2BaselineFrameSource> BuildFrameSources(
    const std::vector<std::string>& image_paths) {
  std::vector<FrozenRound2BaselineFrameSource> frames;
  frames.reserve(image_paths.size());
  for (std::size_t index = 0; index < image_paths.size(); ++index) {
    FrozenRound2BaselineFrameSource frame_source;
    frame_source.frame_index = static_cast<int>(index);
    frame_source.frame_label = fs::path(image_paths[index]).stem().string();
    frame_source.image_path = image_paths[index];
    frames.push_back(frame_source);
  }
  return frames;
}

StereoCameraFixedCalibration ToStereoCameraFixedCalibration(
    const IntermediateCameraConfig& config,
    const std::string& source_label) {
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model = config.camera_model;
  intrinsics.distortion_model = config.distortion_model;
  intrinsics.SetIntrinsicsVector(config.intrinsics);
  intrinsics.SetDistortionVector(config.distortion_coeffs);
  if (config.resolution.size() == 2) {
    intrinsics.resolution = cv::Size(config.resolution[0], config.resolution[1]);
  }

  StereoCameraFixedCalibration calibration;
  calibration.camera_model_family = intrinsics.NormalizedFamilyString();
  calibration.camera_model = intrinsics.NormalizedCameraModel();
  calibration.distortion_model = intrinsics.NormalizedDistortionModel();
  calibration.intrinsics = config.intrinsics;
  calibration.distortion_coeffs = config.distortion_coeffs;
  calibration.resolution = config.resolution;
  calibration.source_label = source_label;
  return calibration;
}

std::map<int, const JointMeasurementFrameResult*> IndexFrameResults(
    const CalibrationMeasurementDataset& dataset) {
  std::map<int, const JointMeasurementFrameResult*> index;
  for (const JointMeasurementFrameResult& frame : dataset.frames) {
    index[frame.frame_index] = &frame;
  }
  return index;
}

bool BoardObservationUsed(const JointBoardObservation& observation) {
  return observation.used_in_solver;
}

bool PointObservationValidForStereoSource(
    const JointPointObservation& point,
    StereoMeasurementSourceMode source_mode) {
  if (point.used_in_solver) {
    return true;
  }
  if (source_mode != StereoMeasurementSourceMode::AllValid) {
    return false;
  }
  return point.rejection_reason_code == JointRejectionReasonCode::None;
}

bool BoardObservationVisibleForStereoSource(
    const JointBoardObservation& observation,
    StereoMeasurementSourceMode source_mode) {
  if (BoardObservationUsed(observation)) {
    return true;
  }
  if (source_mode != StereoMeasurementSourceMode::AllValid) {
    return false;
  }
  for (const JointPointObservation& point : observation.points) {
    if (PointObservationValidForStereoSource(point, source_mode)) {
      return true;
    }
  }
  return false;
}

struct StereoImageTimestampEntry {
  int frame_index = -1;
  long long timestamp_ns = 0;
  std::string image_path;
};

bool ParseTimestampTokenFromImagePath(const std::string& image_path,
                                      long long* timestamp_ns) {
  if (timestamp_ns == nullptr) {
    return false;
  }
  const std::string stem = fs::path(image_path).stem().string();
  const std::vector<std::string> side_tokens = {"_left_", "_right_"};
  for (const std::string& side_token : side_tokens) {
    const std::size_t side_pos = stem.find(side_token);
    if (side_pos == std::string::npos) {
      continue;
    }
    const std::size_t token_begin = side_pos + side_token.size();
    const std::size_t token_end = stem.find('_', token_begin);
    if (token_end == std::string::npos || token_end <= token_begin) {
      continue;
    }
    const std::string token = stem.substr(token_begin, token_end - token_begin);
    if (token.empty() ||
        !std::all_of(token.begin(), token.end(),
                     [](unsigned char ch) { return std::isdigit(ch); })) {
      continue;
    }
    try {
      *timestamp_ns = std::stoll(token);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

std::vector<StereoImageTimestampEntry> BuildTimestampEntries(
    const std::vector<std::string>& image_paths,
    bool* all_have_timestamps) {
  std::vector<StereoImageTimestampEntry> entries;
  entries.reserve(image_paths.size());
  bool ok = true;
  for (std::size_t index = 0; index < image_paths.size(); ++index) {
    long long timestamp_ns = 0;
    if (!ParseTimestampTokenFromImagePath(image_paths[index], &timestamp_ns)) {
      ok = false;
      break;
    }
    StereoImageTimestampEntry entry;
    entry.frame_index = static_cast<int>(index);
    entry.timestamp_ns = timestamp_ns;
    entry.image_path = image_paths[index];
    entries.push_back(entry);
  }
  if (all_have_timestamps != nullptr) {
    *all_have_timestamps = ok;
  }
  if (!ok) {
    entries.clear();
  }
  std::sort(entries.begin(), entries.end(),
            [](const StereoImageTimestampEntry& lhs,
               const StereoImageTimestampEntry& rhs) {
              if (lhs.timestamp_ns != rhs.timestamp_ns) {
                return lhs.timestamp_ns < rhs.timestamp_ns;
              }
              return lhs.frame_index < rhs.frame_index;
            });
  return entries;
}

struct StereoIndexPair {
  int left_index = -1;
  int right_index = -1;
  long long left_timestamp_ns = 0;
  long long right_timestamp_ns = 0;
  long long timestamp_delta_ns = 0;
};

bool ParseFrameOrdinalFromImagePath(const std::string& image_path,
                                    int* ordinal) {
  if (ordinal == nullptr) {
    return false;
  }
  const std::string stem = fs::path(image_path).stem().string();
  const std::size_t separator = stem.find('_');
  if (separator == std::string::npos || separator == 0) {
    return false;
  }
  const std::string token = stem.substr(0, separator);
  if (!std::all_of(token.begin(), token.end(),
                   [](unsigned char ch) { return std::isdigit(ch); })) {
    return false;
  }
  try {
    *ordinal = std::stoi(token);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

void FillPairingDeltaStatistics(const std::vector<StereoIndexPair>& pairs,
                                StereoMeasurementDataset* dataset) {
  if (dataset == nullptr) {
    return;
  }
  long long abs_delta_sum = 0;
  long long abs_delta_max = 0;
  for (const StereoIndexPair& pair : pairs) {
    const long long abs_delta = std::llabs(pair.timestamp_delta_ns);
    abs_delta_sum += abs_delta;
    abs_delta_max = std::max(abs_delta_max, abs_delta);
  }
  dataset->max_pair_timestamp_delta_ns = abs_delta_max;
  dataset->max_abs_pair_timestamp_delta_ms =
      static_cast<double>(abs_delta_max) / 1.0e6;
  dataset->mean_abs_pair_timestamp_delta_ms =
      pairs.empty()
          ? 0.0
          : static_cast<double>(abs_delta_sum) /
                static_cast<double>(pairs.size()) / 1.0e6;
}

std::vector<StereoIndexPair> BuildFrameIndexPairs(
    const std::vector<std::string>& left_image_paths,
    const std::vector<std::string>& right_image_paths,
    double max_delta_ms,
    StereoMeasurementDataset* dataset) {
  std::vector<StereoIndexPair> pairs;
  if (dataset == nullptr) {
    return pairs;
  }
  dataset->pairing_mode = "filename_frame_index_with_timestamp_guard";
  std::map<int, StereoImageTimestampEntry> left_by_ordinal;
  std::map<int, StereoImageTimestampEntry> right_by_ordinal;
  const auto build_index = [](const std::vector<std::string>& paths,
                              std::map<int, StereoImageTimestampEntry>* index) {
    for (std::size_t path_index = 0; path_index < paths.size(); ++path_index) {
      int ordinal = -1;
      if (!ParseFrameOrdinalFromImagePath(paths[path_index], &ordinal)) {
        continue;
      }
      StereoImageTimestampEntry entry;
      entry.frame_index = static_cast<int>(path_index);
      entry.image_path = paths[path_index];
      ParseTimestampTokenFromImagePath(paths[path_index], &entry.timestamp_ns);
      (*index)[ordinal] = entry;
    }
  };
  build_index(left_image_paths, &left_by_ordinal);
  build_index(right_image_paths, &right_by_ordinal);
  const long long max_delta_ns =
      max_delta_ms > 0.0
          ? static_cast<long long>(std::llround(max_delta_ms * 1.0e6))
          : std::numeric_limits<long long>::max();
  int rejected_by_delta = 0;
  for (const auto& left_entry : left_by_ordinal) {
    const auto right_it = right_by_ordinal.find(left_entry.first);
    if (right_it == right_by_ordinal.end()) {
      continue;
    }
    StereoIndexPair pair;
    pair.left_index = left_entry.second.frame_index;
    pair.right_index = right_it->second.frame_index;
    pair.left_timestamp_ns = left_entry.second.timestamp_ns;
    pair.right_timestamp_ns = right_it->second.timestamp_ns;
    pair.timestamp_delta_ns =
        pair.right_timestamp_ns - pair.left_timestamp_ns;
    if (std::llabs(pair.timestamp_delta_ns) > max_delta_ns) {
      ++rejected_by_delta;
      continue;
    }
    pairs.push_back(pair);
  }
  FillPairingDeltaStatistics(pairs, dataset);
  std::ostringstream warning;
  warning << "frame_index_pairing max_delta_ms=" << max_delta_ms
          << " rejected_by_delta=" << rejected_by_delta;
  dataset->warnings.push_back(warning.str());
  return pairs;
}

std::vector<StereoIndexPair> BuildTimestampAwarePairs(
    const std::vector<std::string>& left_image_paths,
    const std::vector<std::string>& right_image_paths,
    StereoMeasurementDataset* dataset) {
  std::vector<StereoIndexPair> pairs;
  if (dataset == nullptr) {
    return pairs;
  }

  bool left_ok = false;
  bool right_ok = false;
  const std::vector<StereoImageTimestampEntry> left_entries =
      BuildTimestampEntries(left_image_paths, &left_ok);
  const std::vector<StereoImageTimestampEntry> right_entries =
      BuildTimestampEntries(right_image_paths, &right_ok);
  if (!left_ok || !right_ok) {
    dataset->pairing_mode = "index_fallback_missing_timestamp";
    const int paired_count =
        std::min(static_cast<int>(left_image_paths.size()),
                 static_cast<int>(right_image_paths.size()));
    for (int index = 0; index < paired_count; ++index) {
      StereoIndexPair pair;
      pair.left_index = index;
      pair.right_index = index;
      pairs.push_back(pair);
    }
    return pairs;
  }

  dataset->pairing_mode = "filename_timestamp_exact";
  std::size_t left_cursor = 0;
  std::size_t right_cursor = 0;
  while (left_cursor < left_entries.size() &&
         right_cursor < right_entries.size()) {
    const StereoImageTimestampEntry& left = left_entries[left_cursor];
    const StereoImageTimestampEntry& right = right_entries[right_cursor];
    if (left.timestamp_ns == right.timestamp_ns) {
      StereoIndexPair pair;
      pair.left_index = left.frame_index;
      pair.right_index = right.frame_index;
      pair.left_timestamp_ns = left.timestamp_ns;
      pair.right_timestamp_ns = right.timestamp_ns;
      pair.timestamp_delta_ns = right.timestamp_ns - left.timestamp_ns;
      pairs.push_back(pair);
      ++left_cursor;
      ++right_cursor;
      continue;
    }
    if (left.timestamp_ns < right.timestamp_ns) {
      ++left_cursor;
    } else {
      ++right_cursor;
    }
  }
  FillPairingDeltaStatistics(pairs, dataset);
  return pairs;
}

}  // namespace

// Intentionally kept in this translation unit for Stage6 only.
StereoFrontendImagePairSelection SelectStereoFrontendImagePairs(
    const std::vector<std::string>& left_image_paths,
    const std::vector<std::string>& right_image_paths,
    StereoFramePairingMode pairing_mode,
    double pairing_max_delta_ms) {
  StereoFrontendImagePairSelection selection;
  selection.original_left_frame_count =
      static_cast<int>(left_image_paths.size());
  selection.original_right_frame_count =
      static_cast<int>(right_image_paths.size());

  StereoMeasurementDataset pairing_diagnostics;
  const std::vector<StereoIndexPair> pairs =
      pairing_mode == StereoFramePairingMode::FrameIndex
          ? BuildFrameIndexPairs(left_image_paths, right_image_paths,
                                 pairing_max_delta_ms, &pairing_diagnostics)
          : BuildTimestampAwarePairs(left_image_paths, right_image_paths,
                                     &pairing_diagnostics);
  selection.pairing_mode = pairing_diagnostics.pairing_mode;
  selection.warnings = pairing_diagnostics.warnings;
  selection.paired_frame_count = static_cast<int>(pairs.size());
  selection.unmatched_left_count =
      selection.original_left_frame_count - selection.paired_frame_count;
  selection.unmatched_right_count =
      selection.original_right_frame_count - selection.paired_frame_count;
  selection.left_image_paths.reserve(pairs.size());
  selection.right_image_paths.reserve(pairs.size());
  for (const StereoIndexPair& pair : pairs) {
    if (pair.left_index < 0 || pair.right_index < 0 ||
        pair.left_index >= static_cast<int>(left_image_paths.size()) ||
        pair.right_index >= static_cast<int>(right_image_paths.size())) {
      selection.failure_reason =
          "Stereo frontend pairing produced an invalid image index.";
      return selection;
    }
    selection.left_image_paths.push_back(left_image_paths[pair.left_index]);
    selection.right_image_paths.push_back(right_image_paths[pair.right_index]);
  }
  if (selection.left_image_paths.empty()) {
    selection.failure_reason =
        "No synchronized stereo frame pairs available for frontend.";
    return selection;
  }
  selection.success = true;
  return selection;
}

StereoMeasurementDataset BuildStereoMeasurementDataset(
    const std::vector<std::string>& left_image_paths,
    const std::vector<std::string>& right_image_paths,
    int holdout_stride,
    int holdout_offset,
    const CalibrationStateBundle& left_bundle,
    const CalibrationStateBundle& right_bundle,
    StereoMeasurementSourceMode source_mode,
    StereoFramePairingMode pairing_mode,
    double pairing_max_delta_ms) {
  StereoMeasurementDataset dataset;
  dataset.success = false;
  dataset.reference_board_id = left_bundle.scene_state.reference_board_id;
  dataset.left_frame_count = static_cast<int>(left_image_paths.size());
  dataset.right_frame_count = static_cast<int>(right_image_paths.size());

  const std::vector<StereoIndexPair> stereo_pairs =
      pairing_mode == StereoFramePairingMode::FrameIndex
          ? BuildFrameIndexPairs(left_image_paths, right_image_paths,
                                 pairing_max_delta_ms, &dataset)
          : BuildTimestampAwarePairs(left_image_paths, right_image_paths,
                                     &dataset);
  const int paired_count = static_cast<int>(stereo_pairs.size());
  dataset.paired_frame_count = paired_count;
  dataset.unmatched_left_count = dataset.left_frame_count - paired_count;
  dataset.unmatched_right_count = dataset.right_frame_count - paired_count;
  if (paired_count <= 0) {
    dataset.failure_reason = "No synchronized stereo frame pairs available.";
    dataset.warnings.push_back(
        "No synchronized stereo frame pairs available after timestamp-aware pairing.");
    return dataset;
  }

  const std::map<int, const JointMeasurementFrameResult*> left_frames =
      IndexFrameResults(left_bundle.measurement_dataset);
  const std::map<int, const JointMeasurementFrameResult*> right_frames =
      IndexFrameResults(right_bundle.measurement_dataset);

  for (int pair_index = 0; pair_index < paired_count; ++pair_index) {
    const StereoIndexPair& index_pair = stereo_pairs[pair_index];
    StereoFramePair pair;
    pair.pair_index = pair_index;
    pair.left_frame_index = index_pair.left_index;
    pair.right_frame_index = index_pair.right_index;
    pair.left_frame_label =
        fs::path(left_image_paths[index_pair.left_index]).stem().string();
    pair.right_frame_label =
        fs::path(right_image_paths[index_pair.right_index]).stem().string();
    pair.left_image_path = left_image_paths[index_pair.left_index];
    pair.right_image_path = right_image_paths[index_pair.right_index];
    pair.is_training = holdout_stride <= 0 ||
                       (((pair_index - holdout_offset) % holdout_stride + holdout_stride) %
                            holdout_stride) != 0;
    dataset.frame_pairs.push_back(pair);
    if (pair.is_training) {
      dataset.training_pair_indices.push_back(pair_index);
    } else {
      dataset.holdout_pair_indices.push_back(pair_index);
    }

    std::set<int> left_visible_boards;
    std::set<int> right_visible_boards;
    const auto left_it = left_frames.find(pair.left_frame_index);
    if (left_it != left_frames.end()) {
      for (const JointBoardObservation& observation : left_it->second->board_observations) {
        if (!BoardObservationVisibleForStereoSource(observation, source_mode)) {
          continue;
        }
        left_visible_boards.insert(observation.board_id);
        for (const JointPointObservation& point : observation.points) {
          if (!PointObservationValidForStereoSource(point, source_mode)) {
            continue;
          }
          StereoObservation stereo_point;
          stereo_point.camera_index = 0;
          stereo_point.pair_index = pair_index;
          stereo_point.frame_index = point.frame_index;
          stereo_point.frame_label = point.frame_label;
          stereo_point.board_id = point.board_id;
          stereo_point.point_id = point.point_id;
          stereo_point.point_type = point.point_type;
          stereo_point.target_point_board = point.target_xyz_board;
          stereo_point.observed_image_xy = point.image_xy;
          stereo_point.weight = point.observation_weight;
          stereo_point.quality = point.quality;
          stereo_point.used_in_solver = true;
          stereo_point.outer_subpix_window_radius =
              point.outer_subpix_window_radius;
          stereo_point.outer_pre_boost_subpix_window_radius =
              point.outer_pre_boost_subpix_window_radius;
          stereo_point.outer_boosted_raw_subpix_window_radius =
              point.outer_boosted_raw_subpix_window_radius;
          stereo_point.outer_close_edge_subpix_boost_applied =
              point.outer_close_edge_subpix_boost_applied;
          stereo_point.outer_close_edge_subpix_area_ratio =
              point.outer_close_edge_subpix_area_ratio;
          stereo_point.outer_close_edge_subpix_max_polar_deg =
              point.outer_close_edge_subpix_max_polar_deg;
          dataset.observations.push_back(stereo_point);
        }
      }
    }
    const auto right_it = right_frames.find(pair.right_frame_index);
    if (right_it != right_frames.end()) {
      for (const JointBoardObservation& observation : right_it->second->board_observations) {
        if (!BoardObservationVisibleForStereoSource(observation, source_mode)) {
          continue;
        }
        right_visible_boards.insert(observation.board_id);
        for (const JointPointObservation& point : observation.points) {
          if (!PointObservationValidForStereoSource(point, source_mode)) {
            continue;
          }
          StereoObservation stereo_point;
          stereo_point.camera_index = 1;
          stereo_point.pair_index = pair_index;
          stereo_point.frame_index = point.frame_index;
          stereo_point.frame_label = point.frame_label;
          stereo_point.board_id = point.board_id;
          stereo_point.point_id = point.point_id;
          stereo_point.point_type = point.point_type;
          stereo_point.target_point_board = point.target_xyz_board;
          stereo_point.observed_image_xy = point.image_xy;
          stereo_point.weight = point.observation_weight;
          stereo_point.quality = point.quality;
          stereo_point.used_in_solver = true;
          stereo_point.outer_subpix_window_radius =
              point.outer_subpix_window_radius;
          stereo_point.outer_pre_boost_subpix_window_radius =
              point.outer_pre_boost_subpix_window_radius;
          stereo_point.outer_boosted_raw_subpix_window_radius =
              point.outer_boosted_raw_subpix_window_radius;
          stereo_point.outer_close_edge_subpix_boost_applied =
              point.outer_close_edge_subpix_boost_applied;
          stereo_point.outer_close_edge_subpix_area_ratio =
              point.outer_close_edge_subpix_area_ratio;
          stereo_point.outer_close_edge_subpix_max_polar_deg =
              point.outer_close_edge_subpix_max_polar_deg;
          dataset.observations.push_back(stereo_point);
        }
      }
    }

    std::set<int> union_boards = left_visible_boards;
    union_boards.insert(right_visible_boards.begin(), right_visible_boards.end());
    std::set<int>* board_target =
        pair.is_training ? &dataset.training_pair_board_ids[pair_index]
                         : &dataset.holdout_pair_board_ids[pair_index];
    *board_target = union_boards;
    for (int board_id : union_boards) {
      const bool in_left = left_visible_boards.count(board_id) > 0;
      const bool in_right = right_visible_boards.count(board_id) > 0;
      if (in_left && in_right) {
        ++dataset.shared_board_observation_count;
        ++dataset.pair_shared_board_count[pair_index];
        dataset.pair_shared_board_ids[pair_index].insert(board_id);
      } else if (in_left) {
        ++dataset.cam0_only_board_observation_count;
        ++dataset.pair_cam0_only_board_count[pair_index];
        dataset.pair_cam0_only_board_ids[pair_index].insert(board_id);
      } else {
        ++dataset.cam1_only_board_observation_count;
        ++dataset.pair_cam1_only_board_count[pair_index];
        dataset.pair_cam1_only_board_ids[pair_index].insert(board_id);
      }
    }
  }

  dataset.success = true;
  return dataset;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
