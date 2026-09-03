#include <aslam/cameras/apriltag_internal/PrecomputedObservationImporter.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

namespace fs = boost::filesystem;

using CsvRow = std::unordered_map<std::string, std::string>;

std::vector<std::string> SplitCsvLine(const std::string& line) {
  std::vector<std::string> values;
  std::stringstream stream(line);
  std::string value;
  while (std::getline(stream, value, ',')) {
    if (!value.empty() && value.back() == '\r') {
      value.pop_back();
    }
    values.push_back(value);
  }
  if (!line.empty() && line.back() == ',') {
    values.push_back("");
  }
  return values;
}

std::vector<CsvRow> ReadCsv(const fs::path& path) {
  std::ifstream input(path.string().c_str());
  if (!input.is_open()) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::string line;
  if (!std::getline(input, line)) {
    throw std::runtime_error("empty CSV: " + path.string());
  }
  const std::vector<std::string> header = SplitCsvLine(line);
  if (header.empty()) {
    throw std::runtime_error("missing CSV header: " + path.string());
  }
  std::set<std::string> unique_header(header.begin(), header.end());
  if (unique_header.size() != header.size()) {
    throw std::runtime_error("duplicate CSV header field: " + path.string());
  }
  std::vector<CsvRow> rows;
  int line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> values = SplitCsvLine(line);
    if (values.size() != header.size()) {
      throw std::runtime_error(
          "CSV field-count mismatch at " + path.string() + ":" +
          std::to_string(line_number));
    }
    CsvRow row;
    for (std::size_t index = 0; index < header.size(); ++index) {
      row[header[index]] = values[index];
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

const std::string& Require(const CsvRow& row, const std::string& key) {
  const auto it = row.find(key);
  if (it == row.end()) {
    throw std::runtime_error("missing required CSV field: " + key);
  }
  return it->second;
}

std::string Optional(const CsvRow& row, const std::string& key,
                     const std::string& fallback) {
  const auto it = row.find(key);
  return it == row.end() ? fallback : it->second;
}

int ParseInt(const std::string& value, const std::string& label) {
  std::size_t parsed = 0;
  const int result = std::stoi(value, &parsed);
  if (parsed != value.size()) {
    throw std::runtime_error("invalid integer for " + label + ": " + value);
  }
  return result;
}

double ParseDouble(const std::string& value, const std::string& label) {
  std::size_t parsed = 0;
  const double result = std::stod(value, &parsed);
  if (parsed != value.size() || !std::isfinite(result)) {
    throw std::runtime_error("invalid finite number for " + label + ": " + value);
  }
  return result;
}

struct BoardDefinition {
  int board_id = -1;
  std::map<int, int> outer_corner_by_point_id;
  std::map<int, Eigen::Vector3d> target_by_point_id;
};

void RecomputeCounts(JointMeasurementBuildResult* result) {
  result->solver_observations.clear();
  result->used_frame_count = 0;
  result->accepted_outer_board_observation_count = 0;
  result->accepted_internal_board_observation_count = 0;
  result->used_board_observation_count = 0;
  result->used_outer_point_count = 0;
  result->used_internal_point_count = 0;
  result->used_total_point_count = 0;
  for (JointMeasurementFrameResult& frame : result->frames) {
    bool frame_used = false;
    for (JointBoardObservation& board : frame.board_observations) {
      board.outer_point_count = 0;
      board.internal_point_count = 0;
      bool board_used = false;
      for (JointPointObservation& point : board.points) {
        if (!point.used_in_solver) {
          continue;
        }
        result->solver_observations.push_back(point);
        board_used = true;
        if (point.point_type == JointPointType::Outer) {
          ++board.outer_point_count;
          ++result->used_outer_point_count;
        } else {
          ++board.internal_point_count;
          ++result->used_internal_point_count;
        }
      }
      board.used_in_solver = board_used;
      if (board_used) {
        frame_used = true;
        ++result->used_board_observation_count;
        if (board.outer_point_count > 0) {
          ++result->accepted_outer_board_observation_count;
        }
        if (board.internal_point_count > 0) {
          ++result->accepted_internal_board_observation_count;
        }
      }
    }
    if (frame_used) {
      ++result->used_frame_count;
    }
  }
  result->used_total_point_count =
      result->used_outer_point_count + result->used_internal_point_count;
  result->success = result->used_frame_count > 0 &&
                    result->used_board_observation_count > 0 &&
                    result->used_outer_point_count >= 4;
}

}  // namespace

FrozenPrecomputedMeasurementInput PrecomputedObservationImporter::Load(
    const std::string& directory,
    int frame_index_offset,
    const std::string& target_mode) const {
  FrozenPrecomputedMeasurementInput result;
  result.source_path = directory;
  result.target_mode_requested = target_mode;
  try {
    const fs::path root(directory);
    const fs::path metadata_path = root / "metadata.yaml";
    const fs::path boards_path = root / "boards.csv";
    const fs::path frames_path = root / "frames.csv";
    const fs::path points_path = root / "points.csv";
    if (!fs::is_directory(root)) {
      throw std::runtime_error("precomputed observation directory does not exist: " + directory);
    }

    cv::FileStorage metadata(metadata_path.string(), cv::FileStorage::READ);
    if (!metadata.isOpened()) {
      throw std::runtime_error("failed to read " + metadata_path.string());
    }
    metadata["schema_version"] >> result.schema_version;
    if (result.schema_version != "stage5_precomputed_observations_v1") {
      throw std::runtime_error("unsupported precomputed observation schema: " +
                               result.schema_version);
    }
    int image_width = 0;
    int image_height = 0;
    metadata["image_width"] >> image_width;
    metadata["image_height"] >> image_height;
    metadata["reference_board_id"] >> result.reference_board_id;
    result.image_size = cv::Size(image_width, image_height);
    if (image_width <= 0 || image_height <= 0 || result.reference_board_id <= 0) {
      throw std::runtime_error("invalid metadata image size or reference board id");
    }

    std::map<int, BoardDefinition> boards;
    for (const CsvRow& row : ReadCsv(boards_path)) {
      const int board_id = ParseInt(Require(row, "board_id"), "board_id");
      const int point_id = ParseInt(Require(row, "point_id"), "point_id");
      const int is_outer = ParseInt(Require(row, "is_outer"), "is_outer");
      const int outer_index =
          ParseInt(Require(row, "outer_corner_index"), "outer_corner_index");
      BoardDefinition& board = boards[board_id];
      board.board_id = board_id;
      const Eigen::Vector3d target(
          ParseDouble(Require(row, "target_x"), "target_x"),
          ParseDouble(Require(row, "target_y"), "target_y"),
          ParseDouble(Require(row, "target_z"), "target_z"));
      if (!board.target_by_point_id.emplace(point_id, target).second) {
        throw std::runtime_error("duplicate board point_id in boards.csv");
      }
      if (is_outer != 0) {
        if (outer_index < 0 || outer_index >= 4) {
          throw std::runtime_error("outer_corner_index must be in [0,3]");
        }
        board.outer_corner_by_point_id[point_id] = outer_index;
      }
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
          ParseDouble(Require(row, "rt" + std::to_string(r) + std::to_string(c)),
                      "board Rt");
        }
      }
    }
    if (boards.empty()) {
      throw std::runtime_error("boards.csv contains no boards");
    }
    result.board_count = static_cast<int>(boards.size());
    std::string normalized_target_mode = target_mode;
    std::transform(normalized_target_mode.begin(), normalized_target_mode.end(),
                   normalized_target_mode.begin(),
                   [](unsigned char value) {
                     return static_cast<char>(std::tolower(value));
                   });
    if (normalized_target_mode != "auto" &&
        normalized_target_mode != "single_board" &&
        normalized_target_mode != "multi_board") {
      throw std::runtime_error(
          "precomputed target mode must be auto, single_board, or multi_board");
    }
    result.target_mode_resolved =
        normalized_target_mode == "auto"
            ? (boards.size() == 1 ? "single_board" : "multi_board")
            : normalized_target_mode;
    if (result.target_mode_resolved == "single_board" && boards.size() != 1) {
      throw std::runtime_error(
          "single_board target mode requires exactly one board in the MAT data");
    }
    if (result.target_mode_resolved == "multi_board" && boards.size() < 2) {
      throw std::runtime_error(
          "multi_board target mode requires at least two boards in the MAT data");
    }
    result.single_board_mode = result.target_mode_resolved == "single_board";
    for (const auto& item : boards) {
      std::set<int> geometric_outer_indices;
      for (const auto& outer : item.second.outer_corner_by_point_id) {
        geometric_outer_indices.insert(outer.second);
      }
      if (geometric_outer_indices != std::set<int>({0, 1, 2, 3})) {
        throw std::runtime_error("board " + std::to_string(item.first) +
                                 " does not define all four geometric outer corners");
      }
    }

    std::map<int, std::pair<std::string, int> > frames;
    for (const CsvRow& row : ReadCsv(frames_path)) {
      const int source_index = ParseInt(Require(row, "frame_index"), "frame_index");
      const int frame_index = source_index + frame_index_offset;
      const std::string frame_label = Require(row, "frame_label");
      const int point_count = ParseInt(Require(row, "point_count"), "point_count");
      if (frame_label.empty() || point_count <= 0 || frames.count(frame_index) != 0) {
        throw std::runtime_error("invalid or duplicate frame in frames.csv");
      }
      frames[frame_index] = std::make_pair(frame_label, point_count);
      FrozenRound2BaselineFrameSource source;
      source.frame_index = frame_index;
      source.frame_label = frame_label;
      source.image_path.clear();
      result.frame_sources.push_back(source);
    }
    if (frames.empty()) {
      throw std::runtime_error("frames.csv contains no frames");
    }

    std::map<std::pair<int, int>, std::vector<JointPointObservation> > grouped_points;
    std::map<int, int> actual_frame_point_counts;
    std::map<std::pair<int, int>, std::array<Eigen::Vector2d, 4> > outer_corners;
    std::map<std::pair<int, int>, std::array<Eigen::Vector3d, 4> > outer_targets;
    std::map<std::pair<int, int>, std::array<bool, 4> > outer_valid;
    int source_point_index = 0;
    for (const CsvRow& row : ReadCsv(points_path)) {
      const int source_frame = ParseInt(Require(row, "frame_index"), "frame_index");
      const int frame_index = source_frame + frame_index_offset;
      const auto frame_it = frames.find(frame_index);
      if (frame_it == frames.end() || frame_it->second.first != Require(row, "frame_label")) {
        throw std::runtime_error("points.csv references an unknown/mismatched frame");
      }
      const int board_id = ParseInt(Require(row, "board_id"), "board_id");
      const int point_id = ParseInt(Require(row, "point_id"), "point_id");
      const auto board_it = boards.find(board_id);
      if (board_it == boards.end()) {
        throw std::runtime_error("points.csv references unknown board " +
                                 std::to_string(board_id));
      }
      const std::string point_type = Require(row, "point_type");
      const int outer_index =
          ParseInt(Require(row, "outer_corner_index"), "outer_corner_index");
      JointPointObservation point;
      point.frame_index = frame_index;
      point.frame_label = frame_it->second.first;
      point.board_id = board_id;
      point.point_id = point_id;
      point.image_xy = Eigen::Vector2d(
          ParseDouble(Require(row, "observed_x"), "observed_x"),
          ParseDouble(Require(row, "observed_y"), "observed_y"));
      point.target_xyz_board = Eigen::Vector3d(
          ParseDouble(Require(row, "target_x"), "target_x"),
          ParseDouble(Require(row, "target_y"), "target_y"),
          ParseDouble(Require(row, "target_z"), "target_z"));
      const auto target_it = board_it->second.target_by_point_id.find(point_id);
      if (target_it == board_it->second.target_by_point_id.end() ||
          (target_it->second - point.target_xyz_board).norm() > 1e-10) {
        throw std::runtime_error(
            "points.csv target coordinate does not match boards.csv topology");
      }
      point.quality = ParseDouble(Require(row, "quality"), "quality");
      point.observation_weight = ParseDouble(
          Optional(row, "observation_weight", "1.0"), "observation_weight");
      if (point.observation_weight < 0.0) {
        throw std::runtime_error("observation_weight must be non-negative");
      }
      point.consistency_weight = 1.0;
      point.final_observation_weight = point.observation_weight;
      point.used_in_solver = true;
      point.rejection_reason_code = JointRejectionReasonCode::None;
      point.source_point_index = source_point_index++;
      if (point_type == "outer") {
        if (outer_index < 0 || outer_index >= 4 ||
            board_it->second.outer_corner_by_point_id.at(point_id) != outer_index) {
          throw std::runtime_error("outer point topology mismatch in points.csv");
        }
        point.point_type = JointPointType::Outer;
        point.source_kind = JointObservationSourceKind::OuterMeasurement;
        const std::pair<int, int> key(frame_index, board_id);
        if (outer_valid[key][outer_index]) {
          throw std::runtime_error("duplicate outer corner in one frame-board observation");
        }
        outer_corners[key][outer_index] = point.image_xy;
        outer_targets[key][outer_index] = point.target_xyz_board;
        outer_valid[key][outer_index] = true;
      } else if (point_type == "internal") {
        if (outer_index != -1) {
          throw std::runtime_error("internal point has a non-negative outer_corner_index");
        }
        point.point_type = JointPointType::Internal;
        point.source_kind = JointObservationSourceKind::InternalMeasurement;
      } else {
        throw std::runtime_error("unsupported point_type: " + point_type);
      }
      grouped_points[std::make_pair(frame_index, board_id)].push_back(point);
      ++actual_frame_point_counts[frame_index];
    }

    for (const auto& frame : frames) {
      if (actual_frame_point_counts[frame.first] != frame.second.second) {
        throw std::runtime_error("frame point count mismatch for " + frame.second.first);
      }
    }

    result.measurement_result.reference_board_id = result.reference_board_id;
    result.measurement_result.bootstrap_seed.reference_board_id =
        result.reference_board_id;
    result.bootstrap_frames.reserve(frames.size());
    result.measurement_result.frames.reserve(frames.size());
    int frame_storage_index = 0;
    for (const auto& frame : frames) {
      OuterBootstrapFrameInput bootstrap_frame;
      bootstrap_frame.frame_index = frame.first;
      bootstrap_frame.frame_label = frame.second.first;
      bootstrap_frame.measurements.image_size = result.image_size;

      JointMeasurementFrameResult measurement_frame;
      measurement_frame.frame_index = frame.first;
      measurement_frame.frame_label = frame.second.first;
      measurement_frame.frame_bootstrap_initialized = true;

      for (const auto& board : boards) {
        const std::pair<int, int> key(frame.first, board.first);
        const auto points_it = grouped_points.find(key);
        if (points_it == grouped_points.end()) {
          continue;
        }
        const auto valid_it = outer_valid.find(key);
        const bool has_four_outer =
            valid_it != outer_valid.end() &&
            std::all_of(valid_it->second.begin(), valid_it->second.end(),
                        [](bool value) { return value; });
        if (!has_four_outer && !result.single_board_mode) {
          throw std::runtime_error("frame " + frame.second.first + " board " +
                                   std::to_string(board.first) +
                                   " does not contain four outer corners");
        }

        OuterBoardMeasurement outer;
        outer.board_id = board.first;
        outer.detected_tag_id = board.first;
        outer.success = true;
        outer.detection_quality = 1.0;
        outer.valid_refined_corner_count = 0;
        outer.refined_corner_valid = {{false, false, false, false}};
        if (valid_it != outer_valid.end()) {
          outer.refined_corner_valid = valid_it->second;
          outer.valid_refined_corner_count = static_cast<int>(std::count(
              valid_it->second.begin(), valid_it->second.end(), true));
          outer.refined_outer_corners_original_image = outer_corners.at(key);
        }
        outer.has_target_outer_corners = has_four_outer;
        if (has_four_outer) {
          outer.target_outer_corners_board = outer_targets.at(key);
        }
        outer.has_direct_dense_control_points = true;
        outer.direct_dense_target_points_board.reserve(points_it->second.size());
        outer.direct_dense_image_points.reserve(points_it->second.size());
        outer.direct_dense_point_is_outer.reserve(points_it->second.size());
        for (const JointPointObservation& point : points_it->second) {
          outer.direct_dense_target_points_board.push_back(
              point.target_xyz_board);
          outer.direct_dense_image_points.push_back(point.image_xy);
          outer.direct_dense_point_is_outer.push_back(
              point.point_type == JointPointType::Outer ? 1u : 0u);
        }
        outer.failure_reason_text.clear();
        bootstrap_frame.measurements.requested_board_ids.push_back(board.first);
        bootstrap_frame.measurements.board_measurements.push_back(outer);

        JointBoardObservation observation;
        observation.board_id = board.first;
        observation.frame_bootstrap_initialized = true;
        observation.board_bootstrap_initialized = true;
        observation.reference_connected = true;
        observation.used_in_solver = true;
        observation.points = points_it->second;
        const int observation_index =
            static_cast<int>(measurement_frame.board_observations.size());
        for (JointPointObservation& point : observation.points) {
          point.frame_storage_index = frame_storage_index;
          point.source_board_observation_index = observation_index;
        }
        measurement_frame.visible_board_ids.push_back(board.first);
        measurement_frame.board_observations.push_back(std::move(observation));
      }
      if (measurement_frame.board_observations.empty()) {
        throw std::runtime_error("frame has no usable board observations: " +
                                 frame.second.first);
      }
      result.bootstrap_frames.push_back(std::move(bootstrap_frame));
      result.measurement_result.frames.push_back(std::move(measurement_frame));
      ++frame_storage_index;
    }

    RecomputeCounts(&result.measurement_result);
    if (!result.measurement_result.success) {
      throw std::runtime_error("imported measurement dataset is empty or lacks outer points");
    }
    result.warnings.push_back(
        "Imported boards.Rt was validated but not used to initialize intrinsics or layout; Stage5 re-estimates the multi-board scene from outer observations.");
    result.measurement_result.warnings = result.warnings;
    result.success = true;
  } catch (const std::exception& error) {
    result.failure_reason = error.what();
  }
  return result;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
